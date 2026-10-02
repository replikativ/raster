(ns raster.ode.amr-transfer
  "Conservative cell-average transfer through the existing CSR operator.

   Exact normalized rectangles establish overlap coverage before FP64 coefficients are
   materialized. This is host-side geometry preparation, not a new compiler operation or
   ownership transfer. Floating evaluation preserves constants/mass only within rounding error;
   it does not reconstruct gradients or claim high-order remap accuracy."
  (:require [raster.compiler.ir.validate :refer [fail! positive-number?]]
            [raster.linalg.sparse :as sparse])
  (:import [java.util Comparator TreeMap]))

(defn- rectangles [projection]
  (let [{:keys [domain domain-lengths cells]} projection]
    (when-not (and (vector? domain) (= 2 (count domain)) (every? pos-int? domain)
                   (vector? domain-lengths) (= 2 (count domain-lengths))
                   (every? positive-number? domain-lengths)
                   (vector? cells) (seq cells) (<= (count cells) Integer/MAX_VALUE))
      (fail! "transfer requires a concrete two-dimensional active layout"
             :amr-transfer-layout {}))
    (mapv (fn [id {:keys [offsets shape]}]
            (when-not (and (vector? offsets) (= 2 (count offsets))
                           (every? #(and (integer? %) (not (neg? %))) offsets)
                           (vector? shape) (= 2 (count shape)) (every? #(and (integer? %) (pos? %)) shape)
                           (every? true? (map #(<= (+' %1 %2) %3) offsets shape domain)))
              (fail! "transfer cell lies outside its declared domain"
                     :amr-transfer-cell {:cell id :offsets offsets :shape shape}))
            (let [[x y] (mapv / offsets domain)
                  [nx ny] (mapv / shape domain)]
              {:id id :x x :end-x (+ x nx) :y y :end-y (+ y ny) :area (* nx ny)}))
          (range) cells)))

(defn- active-index []
  (TreeMap. (reify Comparator (compare [_ a b] (clojure.core/compare a b)))))

(defn- overlaps [^TreeMap active rectangle]
  (let [floor (.floorEntry active (:y rectangle))
        start (if floor floor (.ceilingEntry active (:y rectangle)))]
    ;; Consume lazily so a small nonzero budget can stop a large overlap query immediately.
    ;; Neither caller mutates this queried index while consuming its entries.
    (->> (iterate #(when % (.higherEntry active (.getKey ^java.util.Map$Entry %))) start)
         (take-while #(and % (< (.getKey ^java.util.Map$Entry %) (:end-y rectangle))))
         (map #(.getValue ^java.util.Map$Entry %))
         (filter #(> (:end-y %) (:y rectangle))))))

(defn- insert! [^TreeMap active rectangle]
  (when (seq (overlaps active rectangle))
    (fail! "active layout contains overlapping cell interiors"
           :amr-transfer-overlap {:cell (:id rectangle)}))
  (.put active (:y rectangle) rectangle))

(defn matrix
  "Build a fresh FP64 CSR matrix mapping source cell averages to target cell averages.

   Physical domain lengths must match. Finest lattice sizes may differ: geometry uses exact
   rational coordinates, not tile expansion. A sweep inserts/removes disjoint y intervals and
   visits actual source/target overlaps, rather than all cell pairs. :max-nonzeros bounds stored
   overlaps before primitive allocation. Cell order defines the matrix's input/output order.
   Geometry, coverage and coefficient representability are checked; no manifest compatibility,
   runtime ownership, temporal interpolation or compiler certificate is inferred."
  ([source target] (matrix source target {}))
  ([source target {:keys [max-nonzeros] :or {max-nonzeros 1000000}}]
   (when-not (and (integer? max-nonzeros) (<= 1 max-nonzeros Integer/MAX_VALUE))
     (fail! "transfer requires a positive Int nonzero budget" :amr-transfer-budget {}))
   (when-not (= (:domain-lengths source) (:domain-lengths target))
     (fail! "cell-average transfer requires the same physical domain"
            :amr-transfer-domain {:source (:domain-lengths source) :target (:domain-lengths target)}))
   (when (and (vector? (:cells source)) (vector? (:cells target)))
     (when (>= (count (:cells target)) Integer/MAX_VALUE)
       (fail! "CSR target row offsets require nrows + 1 Int-addressable entries"
              :amr-transfer-capacity {:target-rows (count (:cells target))}))
     (when (> (max (count (:cells source)) (count (:cells target))) max-nonzeros)
       (fail! "complete positive-area coverage cannot fit the declared CSR budget"
              :amr-transfer-budget {:source-rows (count (:cells source))
                                    :target-rows (count (:cells target)) :budget max-nonzeros})))
   (let [sources (rectangles source) targets (rectangles target)
         _ (doseq [rects [sources targets]]
             (when-not (= 1 (reduce + (map :area rects)))
               (fail! "active layout must cover one complete normalized domain"
                      :amr-transfer-coverage {})))
         events (sort-by (juxt :x :phase :side :id)
                        (for [[side rects] [[0 sources] [1 targets]] rectangle rects
                              [phase x] [[0 (:end-x rectangle)] [1 (:x rectangle)]]]
                          {:x x :phase phase :side side :id (:id rectangle) :rectangle rectangle}))
         source-index (active-index) target-index (active-index)
         rows (volatile! (vec (repeat (count targets) (sorted-map))))
         source-coverage (volatile! (vec (repeat (count sources) 0)))
         target-coverage (volatile! (vec (repeat (count targets) 0)))
         nonzeros (volatile! 0)]
     (doseq [{:keys [phase side rectangle]} events]
       (let [own (if (zero? side) source-index target-index)
             other (if (zero? side) target-index source-index)]
         (if (zero? phase)
           (.remove ^TreeMap own (:y rectangle))
           (do
             (insert! own rectangle)
             (doseq [opposite (overlaps other rectangle)]
               (when (>= @nonzeros max-nonzeros)
                 (fail! "cell overlaps exceed the declared CSR budget"
                        :amr-transfer-budget {:budget max-nonzeros}))
               (let [s (if (zero? side) rectangle opposite)
                     t (if (zero? side) opposite rectangle)
                     area (* (- (min (:end-x s) (:end-x t)) (max (:x s) (:x t)))
                             (- (min (:end-y s) (:end-y t)) (max (:y s) (:y t))))
                     weight (/ area (:area t))]
                 (when-not (and (pos? area) (positive-number? (double weight)))
                   (fail! "overlap coefficient is not representable as positive FP64"
                          :amr-transfer-coefficient {:source (:id s) :target (:id t) :weight weight}))
                 (vswap! rows assoc-in [(:id t) (:id s)] weight)
                 (vswap! source-coverage update (:id s) + area)
                 (vswap! target-coverage update (:id t) + area)
                 (vswap! nonzeros inc)))))))
     (when-not (and (= (mapv :area sources) @source-coverage)
                    (= (mapv :area targets) @target-coverage)
                    (every? #(= 1 (reduce + (vals %))) @rows))
       (fail! "overlap transfer does not cover both active layouts exactly"
              :amr-transfer-coverage {}))
     (let [entries (mapcat seq @rows)]
       (sparse/->CSRMatrix
        (int-array (reductions + 0 (map count @rows)))
        (int-array (map key entries)) (double-array (map (comp double val) entries))
        (long (count targets)) (long (count sources)) (long @nonzeros))))))
