(ns raster.ode.amr-transfer-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.ode.amr-geometry :as geometry]
            [raster.ode.amr-transfer :as transfer]
            [raster.linalg.sparse :as sparse]))

(defn- layout [ratio region]
  (geometry/project-hierarchy
   (amr/hierarchy
    {:id :transfer :base-shape [3 4] :proper-nesting-width 0
     :levels [(amr/level {:id 0 :index 0
                          :patches [(amr/patch {:id :base :field :base-field :device :local
                                               :level 0 :offsets [0 0] :shape [3 4]})]})
              (amr/level {:id 1 :index 1 :ratio-to-parent ratio
                          :patches [(amr/patch (merge region {:id :fine :field :fine-field
                                                              :device :local :level 1}))]})]})
   {:domain-lengths [2.0 3.0]}))

(defn- reference [source target values]
  ;; Independent all-pairs rational overlap oracle, deliberately bounded to small layouts.
  (mapv (fn [t]
          (reduce +
                  (map (fn [s value]
                         (let [lo-s (mapv / (:offsets s) (:domain source))
                               hi-s (mapv / (mapv + (:offsets s) (:shape s)) (:domain source))
                               lo-t (mapv / (:offsets t) (:domain target))
                               hi-t (mapv / (mapv + (:offsets t) (:shape t)) (:domain target))
                               area (reduce * (map #(max 0 (- (min %2 %4) (max %1 %3)))
                                                    lo-s hi-s lo-t hi-t))
                               target-area (reduce * (map - hi-t lo-t))]
                           (* value (double (/ area target-area))))) (:cells source) values)))
        (:cells target)))

(deftest changed-lattices-match-independent-overlaps-and-conserve-mass
  (let [layouts [(layout [2 2] {:offsets [2 2] :shape [2 4]})
                 (layout [2 3] {:offsets [0 0] :shape [2 3]})
                 (layout [3 2] {:offsets [3 4] :shape [3 2]})
                 (layout [2 3] {:offsets [0 0] :shape [6 12]})]]
    (doseq [source layouts target layouts]
      (let [matrix (transfer/matrix source target)
            values (double-array (map #(+ 1.0 (* 0.25 %)) (range (count (:cells source)))))
            actual (sparse/spmv matrix values (double-array (count (:cells target))) 1.0 0.0)
            expected (reference source target values)]
        (is (= [(count (:cells target)) (count (:cells source))] [(:nrows matrix) (:ncols matrix)]))
        (is (every? #(< (Math/abs (double %)) 1.0e-12) (map - expected actual)))
        (is (< (Math/abs (- (reduce + (map * values (:cell-volumes source)))
                            (reduce + (map * actual (:cell-volumes target))))) 1.0e-11))
        (let [ones (double-array (repeat (count values) 1.0))]
          (is (every? #(< (Math/abs (- 1.0 %)) 1.0e-14)
                      (sparse/spmv matrix ones (double-array (count (:cells target))) 1.0 0.0))))))))

(deftest transfer-declines-drift-overlap-holes-and-budget-exhaustion
  (let [source (layout [2 2] {:offsets [2 2] :shape [2 4]})
        reason (fn [s t options]
                 (try (transfer/matrix s t options) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= :amr-transfer-domain (reason source (assoc source :domain-lengths [1.0 3.0]) {})))
    (is (= :amr-transfer-budget (reason source source {:max-nonzeros 1})))
    (is (= :amr-transfer-coverage (reason source (update source :cells pop) {})))
    (is (= :amr-transfer-cell
           (reason source (assoc-in source [:cells 0 :offsets] [-1 0]) {})))
    ;; Duplicating a same-volume row preserves total area but must fail the sweep's overlap gate.
    (let [first-coarse (first (:cells source))
          duplicate (assoc-in source [:cells 1] first-coarse)]
      (is (= :amr-transfer-overlap (reason duplicate duplicate {}))))))

(deftest packed-order-and-fresh-storage-are-independent-of-sweep-order
  (let [original (layout [2 3] {:offsets [0 0] :shape [2 3]})
        source (update original :cells #(vec (reverse %)))
        target (update original :cells #(vec (concat (drop 3 %) (take 3 %))))
        matrix (transfer/matrix source target)
        fresh (transfer/matrix source target)
        values (double-array (map #(+ 0.25 %) (range (count (:cells source)))))
        actual (sparse/spmv matrix values (double-array (count (:cells target))) 1.0 0.0)]
    (is (= (count (:cells source)) (:nnz matrix)))
    (is (= (reference source target values) (vec actual)))
    (is (not (identical? (:values matrix) (:values fresh))))
    (aset-double ^doubles (:values fresh) 0 999.0)
    (is (= 1.0 (aget ^doubles (:values matrix) 0)))))
