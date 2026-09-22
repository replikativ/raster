(ns raster.compiler.passes.parallel.abm-atomic-route-test
  "Production-route coverage for value-returning atomics in the firms workload."
  (:require [clojure.test :refer [deftest is]]
            [raster.abm.firms.phases :as phases]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.pipeline :as pipeline]))

(defn- body-operations [artifact]
  (letfn [(walk [operations]
            (mapcat (fn [operation]
                      (cons operation
                            (concat (walk (or (:operations operation) []))
                                    (walk (or (:then-operations operation) []))
                                    (walk (or (:else-operations operation) [])))))
                    operations))]
    (walk (get-in artifact [:attributes :kernel-body :operations]))))

(deftest fill-members-retains-the-atomic-old-value-type
  (let [descriptor (pipeline/compile-gpu-program
                    #'phases/fill-members-par! :ze:debug
                    :dtype :float :on-non-resident :nil)]
    (is (some? descriptor))
    (is (= 2 (count (:steps descriptor))))
    (is (kernel-artifact/kernel-artifact? (get-in descriptor [:steps 0 :artifact])))
    (let [executable (get-in descriptor [:steps 1 :artifact])
          artifacts (if (kernel-graph/kernel-graph? executable)
                      (mapv :operation (:nodes executable))
                      [executable])
          atomics (for [artifact artifacts
                        operation (body-operations artifact)
                        :when (= "AtomicRMW" (some-> operation class .getSimpleName))]
                    operation)]
      (is (every? kernel-artifact/kernel-artifact? artifacts))
      (is (= 1 (count atomics)))
      (is (= :int (get-in (first atomics) [:result :type]))))))
