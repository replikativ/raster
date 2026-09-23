(ns raster.compiler.passes.parallel.map-read-requirements
  "Minimum flat storage derived from actual typed map loads, not from output size."
  (:require [clojure.set :as set]
            [clojure.walk :as walk]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.index-algebra :as algebra]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scalar-range :as ranges]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.segmap-body :as map-body]))

(defn- span-expression
  [{:keys [const factors]}]
  (let [operands (cond-> (vec factors) (not= 1 const) (conj const))]
    (case (count operands)
      0 1
      1 (first operands)
      (apply launch/product operands))))

(defn symbolic-read-certificate
  "Certify exact symbolic spans for the flat reads of a typed one-dimensional map.

   The proof is structural: the map index is decomposed into mixed-radix digits and each address
   must enumerate a zero-based dense interval over the digits it represents. Omitted digits are
   permitted because a broadcast may repeat an input interval; padding, translation, indirect
   reads, and undecidable arithmetic decline. `scalar-definitions` contains already checked
   KernelLaunch definitions from the host prefix. Only monomial definitions are substituted, so
   quotient relations are never invented here."
  [operation {:keys [scalar-definitions] :or {scalar-definitions {}}}]
  (when (and (segop/seg-map? operation)
             (= 1 (count (get-in operation [:space :dims])))
             (empty? (set/intersection (set (:inputs operation)) (set (:outputs operation))))
             (not (seq (get-in operation [:scalar-region :effects]))))
    (let [{index :name bound :bound} (first (get-in operation [:space :dims]))
          {source-locals :locals result :result} (:scalar-region operation)
          monomial-definitions (into {} (filter (comp algebra/monomial val)) scalar-definitions)
          expand #(walk/postwalk-replace monomial-definitions %)
          bound (expand bound)
          locals (mapv #(update % :init (comp algebra/canonical-arithmetic expand)) source-locals)
          reads (->> (concat (map :init source-locals) [result])
                     (mapcat descriptor/aget-reads)
                     (filter #(contains? (:inputs operation) (:sym %)))
                     vec)
          read-facts
          (mapv (fn [{:keys [sym idx]}]
                  (let [coordinate (algebra/canonical-arithmetic (expand idx))
                        form (algebra/index-form
                              coordinate
                              index bound locals {})]
                    (when-let [span (algebra/zero-based-dense-span form)]
                      {:buffer sym :source-coordinate idx
                       :coordinate coordinate :form form
                       :span (span-expression span)})))
                reads)]
      (when (and (seq reads) (every? some? read-facts))
        {:kind :zero-based-dense-read-spans
         :index index :extent bound
         ;; Retain both sides of the proof. `:source-locals` and
         ;; `:scalar-definitions` let a later graph-bound target projection recompute this
         ;; certificate from the exact SegMap instead of trusting attached metadata. `:locals`
         ;; is the canonical expanded region used by the mixed-radix proof.
         :source-locals source-locals
         :scalar-definitions monomial-definitions
         :locals locals :reads read-facts
         :requirements
         (reduce (fn [result {:keys [buffer span]}]
                  (update result buffer
                          (fn [prior]
                            (cond
                              (nil? prior) span
                              (= prior span) prior
                              :else (launch/maximum prior span)))))
                 {} read-facts)}))))

(defn symbolic-read-requirements
  "Return only the buffer-capacity projection of `symbolic-read-certificate`."
  [operation options]
  (:requirements (symbolic-read-certificate operation options)))

(defn static-read-requirements
  "Optional all-load proof for a plain, positive static, one-dimensional result map.
   The map schedule establishes 0<=index<bound at each :map-active load. Indirect/local-SSA
   coordinates, effects and unknown domains decline. Returned counts are minimum capacities."
  [operation options]
  (when (and (segop/seg-map? operation)
             (:out-sym operation)
             (= 1 (count (get-in operation [:space :dims])))
             (empty? (set/intersection (set (:inputs operation)) (set (:outputs operation))))
             (not (seq (get-in operation [:scalar-region :effects]))))
    (let [{index :name bound :bound} (first (get-in operation [:space :dims]))]
      (when (and (integer? bound) (<= 1 bound Integer/MAX_VALUE))
        (try
          (let [lowered (map-body/lower operation options)
                operations (tree-seq coll? seq (get-in lowered [:kernel-body :operations]))
                loads (filter #(= "raster.compiler.ir.kernel_body.ScalarLoad"
                                  (some-> % class .getName)) operations)
                ;; A lexical map local may be retained as typed SSA rather than beta-expanded
                ;; into the load coordinate.  Reuse only the range certificates carried by its
                ;; validated ScalarCompute; otherwise a harmless `(let* [k (+ i j)] (aget a k))`
                ;; loses the same capacity proof as its inlined spelling.
                computed-ranges
                (reduce
                 (fn [known node]
                   (if (= "raster.compiler.ir.kernel_body.ScalarCompute"
                          (some-> node class .getName))
                     (let [id (get-in node [:result :id])
                           type (get-in node [:result :type])
                           expression (:expression node)
                           proof (get-in expression [:options :proof])
                           argument-range
                           (fn [argument]
                             (cond
                               (symbol? argument) (get-in known [argument :range])
                               (and (integer? (:value argument)) (:type argument))
                               (ranges/literal (:value argument) (:type argument))
                               :else nil))
                           arguments (mapv argument-range (:arguments expression))
                           derived
                           (or (when (and (= :typed-scalar-range (:kind proof))
                                          (ranges/contained-in-dtype? proof type))
                                 (select-keys proof [:lower :upper]))
                               (when (every? some? arguments)
                                 (case (:op expression)
                                   (:+ :- :*)
                                   (let [range (ranges/arithmetic (:op expression) arguments)]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :quot
                                   (let [[numerator divisor] arguments
                                         range (when (and (<= 0 (:lower numerator))
                                                          (= (:lower divisor) (:upper divisor))
                                                          (pos? (:lower divisor)))
                                                 (ranges/quotient arguments))]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :rem
                                   (let [[numerator divisor] arguments
                                         d (:lower divisor)
                                         range (when (and (<= 0 (:lower numerator))
                                                          (= d (:upper divisor)) (pos? d))
                                                 {:lower 0
                                                  :upper (min (:upper numerator) (dec d))})]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :cast
                                   (let [range (first arguments)]
                                     (when (and (= :exact (get-in expression [:options :overflow]))
                                                (ranges/contained-in-dtype? range type))
                                       range))
                                   nil)))]
                       (if (and (symbol? id) derived)
                         (assoc known id {:type type :range derived})
                         known))
                     known))
                 {index {:type :long :range {:lower 0 :upper (dec bound)}}}
                 operations)
                leaf-types (merge (:scalar-types options)
                                  {index :long}
                                  (into {} (map (fn [[id fact]] [id (:type fact)]))
                                        computed-ranges))
                leaf-ranges (merge {index {:lower 0 :upper (dec bound)}}
                                   (into {} (map (fn [[id fact]] [id (:range fact)]))
                                         computed-ranges))
                requirements
                (mapv (fn [{:keys [buffer coordinates predicate]}]
                        (when (and (= :map-active predicate) (= 1 (count coordinates)))
                          (when-let [range (ranges/typed-index-range
                                           (first coordinates)
                                           leaf-types leaf-ranges)]
                            (when (<= 0 (:lower range))
                              [buffer (inc' (:upper range))]))))
                      loads)]
            (when (and (seq loads) (every? some? requirements))
              (reduce (fn [result [id extent]] (update result id (fnil max 0) extent))
                      {} requirements)))
          (catch clojure.lang.ExceptionInfo _ nil))))))
