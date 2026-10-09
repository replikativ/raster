(ns raster.compiler.passes.region-copy-coverage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.equation-artifact-store :as artifact-store]
            [raster.compiler.ir.link-plan :as link]
            [raster.gpu.compiled :as compiled]))

(deftest copy-addresses-feed-the-shared-complete-write-proof
  (with-redefs [hardware/descriptor-for
               (constantly {:device-id :ocl:copy-coverage :device-type :gpu :backend :ocl :fp64? true
                            :subgroup-dialect :intel-opencl :max-workgroup-size 256})
               ;; This test checks retained arithmetic and coverage, not persistent identity.
               artifact-store/load-artifact (fn [& _] {:status :miss :reason :coverage-test})
               artifact-store/store-artifact! (fn [& _] {:status :not-stored})]
    (doseq [[src-size src-off dst-off len full?]
            [[16 0 0 16 true] [20 4 0 16 true] [16 0 1 15 false] [16 0 0 15 false]]]
      (let [prepared (compiled/lower #'arrays/acopy!
                                     [(double-array src-size) src-off (double-array 16) dst-off len]
                                     {:target :ocl:copy-coverage :compiler :equation-first :dtype :double
                                      :inline? true :donate '[dst]})
            local (compiled/plan prepared)
            value (first (link/output-value-ids local))
            node (first (link/value-node-ids local value))
            complete (:complete-writes (link/initialization-contract local))]
        (is (= full? (contains? complete node))
            (str "copy offsets " [src-off dst-off] " and extent " len))
        (is (= :unproven (:completion (link/memory-report local))))))))
