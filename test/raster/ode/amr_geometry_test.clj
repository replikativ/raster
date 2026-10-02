(ns raster.ode.amr-geometry-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.ode.amr-geometry :as geometry]
            [raster.ode.finite-volume :as fv]))

(defn- hierarchy [base refinements]
  (amr/hierarchy
   {:id :geometry :base-shape base :proper-nesting-width 0
    :levels (into [(amr/level {:id 0 :index 0
                              :patches [(amr/patch {:id :base :field :base-field :device :local
                                                   :level 0 :offsets [0 0] :shape base})]})]
                  (map-indexed
                   (fn [i {:keys [ratio patches]}]
                     (let [level (inc i)]
                       (amr/level {:id level :index level :ratio-to-parent ratio
                                   :patches (mapv (fn [p region]
                                                    (amr/patch (merge region
                                                                      {:id [level p] :field [:field level p]
                                                                       :device :local :level level})))
                                                  (range) patches)}))) refinements))}))

(defn- project [h & [options]]
  (geometry/project-hierarchy h (merge {:domain-lengths [2.0 3.0]} options)))

(defn- assert-coverage [projection]
  (let [{:keys [domain cells faces incidences cell-volumes patch-rows]} projection
        tiles (for [{[x y] :offsets [nx ny] :shape} cells
                    xx (range x (+ x nx)) yy (range y (+ y ny))] [xx yy])]
    (is (= (reduce * domain) (count tiles) (count (distinct tiles))))
    (is (< (Math/abs (- 6.0 (reduce + cell-volumes))) 1.0e-12))
    (is (= (count cells) (count (mapcat :rows patch-rows))))
    (is (= (set (range (count cells))) (set (map second (mapcat :rows patch-rows)))))
    (doseq [[f {:keys [left right]}] (map-indexed vector faces)]
      (is (= (frequencies [[left 1.0] [right -1.0]])
             (frequencies (for [[c entries] (map-indexed vector incidences)
                               [face sign] entries :when (= f face)] [c sign])))))
    ;; Independent perimeter accounting: each cell side has one physical extent, regardless
    ;; of how many fine neighbours split it. Self-neighbours contribute both sides.
    (doseq [[c {[nx ny] :shape}] (map-indexed vector cells)]
      (let [expected (* 2.0 (+ (* ny (/ 3.0 (domain 1))) (* nx (/ 2.0 (domain 0)))))
            actual (reduce + (map (fn [[f _]] (:measure (faces f))) (incidences c)))]
        (is (< (Math/abs (- expected actual)) 1.0e-12))))))

(deftest rectangular-anisotropic-multilevel-provenance
  (doseq [h [(hierarchy [3 4] [])
             (hierarchy [3 4] [{:ratio [2 3] :patches [{:offsets [2 3] :shape [2 6]}]}])
             (hierarchy [3 4] [{:ratio [2 3] :patches [{:offsets [2 3] :shape [2 6]}]}
                               {:ratio [3 2] :patches [{:offsets [6 6] :shape [3 4]}]}])
             (hierarchy [3 4] [{:ratio [2 3] :patches [{:offsets [0 0] :shape [6 12]}]}])
             (hierarchy [4 4] [{:ratio [2 2] :patches [{:offsets [0 0] :shape [2 2]}
                                                      {:offsets [6 6] :shape [2 2]}]}])]]
    (let [projection (project h)
          patches (into {} (map (juxt :id identity)) (mapcat :patches (:levels h)))]
      (assert-coverage projection)
      (doseq [{:keys [patch field device level local-index offsets shape]} (:cells projection)]
        (let [source (patches patch) ny (second (:shape source))
              xy [(quot local-index ny) (mod local-index ny)]]
          (is (= [field device level] ((juxt :field :device :level) source)))
          (is (= offsets (mapv * (mapv + xy (:offsets source)) shape)))))
      (is (= projection (project h))))))

(deftest periodic-distinct-faces-and-self-neighbours
  (doseq [base [[1 1] [1 2] [2 1] [2 2]]]
    (let [projection (project (hierarchy base []))]
      (assert-coverage projection)
      (is (= (* 2 (reduce * base)) (count (:faces projection))))))
  (let [projection (project (hierarchy [2 1] []))]
    (is (= 2 (count (filter #(= 0 (:axis %)) (:faces projection)))))
    (is (= 2 (count (filter #(= (:left %) (:right %)) (:faces projection)))))))

(deftest fresh-arrays-feed-the-existing-conservative-kernels
  (doseq [boundary [:periodic :no-flux]]
  (let [projection (project (hierarchy [3 4] [{:ratio [2 3]
                                             :patches [{:offsets [2 3] :shape [2 6]}]}])
                            {:boundary boundary})
        {:keys [left right conductance offsets indices orientation inverse-volume
                cell-count face-count] :as arrays} (geometry/materialize-connectivity projection)
        fresh (geometry/materialize-connectivity projection)
        field (double-array (map #(double (inc %)) (range cell-count)))
        out (double-array cell-count) flux (double-array face-count)
        change (double-array cell-count) dt 0.001]
    (is (not (identical? left (:left fresh))))
    (aset-int (:left fresh) 0 999)
    (is (not= 999 (aget left 0)))
    (fv/diffusive-face-fluxes! flux field left right conductance face-count)
    (fv/divergence-step! out field flux offsets indices orientation inverse-volume cell-count dt)
    ;; Independent extensive-mass scatter, not the CSR divergence implementation.
    (doseq [{:keys [left right conductance]} (:faces projection)]
      (let [amount (* dt conductance (- (aget field left) (aget field right)))]
        (aset-double change left (- (aget change left) amount))
        (aset-double change right (+ (aget change right) amount))))
    (doseq [i (range cell-count)]
      (is (< (Math/abs (- (aget out i) (+ (aget field i)
                                        (/ (aget change i) (nth (:cell-volumes projection) i)))))
             1.0e-12)))
    (is (< (Math/abs (- (reduce + (map * field (:cell-volumes projection)))
                        (reduce + (map * out (:cell-volumes projection))))) 1.0e-12))
    (java.util.Arrays/fill field 7.0)
    (fv/diffusive-face-fluxes! flux field left right conductance face-count)
    (fv/divergence-step! out field flux offsets indices orientation inverse-volume cell-count dt)
    (is (= (vec field) (vec out)))
    (doseq [{:keys [patch local-indices packed-indices]} (:patch-rows arrays)]
      (is (= (vec (for [cell (:cells projection) :when (= patch (:patch cell))] (:local-index cell)))
             (vec local-indices)))
      (is (= (count local-indices) (count packed-indices)))))))

(deftest many-small-patches-and-explicit-packing
  (let [h (hierarchy [10 10] [{:ratio [2 2]
                              :patches (vec (for [x (range 0 20 2) y (range 0 20 2)]
                                              {:offsets [x y] :shape [2 2]}))}])
        projection (project h {:max-patch-cells 500})
        arrays (geometry/materialize-connectivity projection)
        packed (double-array (:cell-count arrays))
        source (into {} (map-indexed (fn [p {:keys [patch shape]}]
                                      [patch (double-array (map #(+ (* 1000 p) %) (range (reduce * shape))))])
                                    (:patch-rows arrays)))]
    (is (= 400 (:cell-count arrays)))
    (is (empty? (:rows (first (:patch-rows projection)))) "covered coarse field is not packed")
    (doseq [{:keys [patch local-indices packed-indices]} (:patch-rows arrays)
            i (range (count local-indices))]
      (aset-double packed (aget ^ints packed-indices i)
                   (aget ^doubles (source patch) (aget ^ints local-indices i))))
    (is (= (vec packed) (mapv (fn [{:keys [patch local-index]}]
                               (aget ^doubles (source patch) local-index)) (:cells projection))))))

(deftest patch-permutation-retains-geometric-identity-and-provenance
  (let [h (hierarchy [4 4] [{:ratio [2 3]
                            :patches [{:offsets [0 0] :shape [2 3]}
                                      {:offsets [6 9] :shape [2 3]}]}])
        reordered (update-in h [:levels 1 :patches] #(vec (reverse %)))
        original (project h) permuted (project reordered)
        cells-by-geometry (fn [p] (into {} (map (fn [c] [[(:offsets c) (:shape c)] c])) (:cells p)))
        faces-by-geometry (fn [p]
                            (mapv (fn [f]
                                    (-> f (update :left #(get-in p [:cells % :offsets]))
                                        (update :right #(get-in p [:cells % :offsets])))) (:faces p)))]
    (is (not= (:cells original) (:cells permuted)) "packed order follows declared patch order")
    (is (= (cells-by-geometry original) (cells-by-geometry permuted)))
    (is (= (faces-by-geometry original) (faces-by-geometry permuted)))))

(deftest explicit-boundaries-and-budget-declines
  (let [p (project (hierarchy [2 2] []) {:boundary :no-flux})]
    (is (= 4 (count (:faces p))))
    (is (every? #(pos? (:plane %)) (:faces p))))
  (doseq [options [{:max-patch-cells 3} {:boundary :dirichlet} {:diffusivity Double/NaN}
                   {:domain-lengths [Double/POSITIVE_INFINITY 1.0]}
                   {:domain-lengths [1.0e-160 1.0e-160]}]]
    (is (thrown? clojure.lang.ExceptionInfo (project (hierarchy [2 2] []) options))))
  (let [one-dimensional (amr/hierarchy
                         {:id :line :base-shape [2]
                          :levels [(amr/level {:id 0 :index 0
                                              :patches [(amr/patch {:id :base :field :field :device :local
                                                                   :level 0 :offsets [0] :shape [2]})]})]})]
    (is (= :amr-geometry-rank
           (try (project one-dimensional) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
