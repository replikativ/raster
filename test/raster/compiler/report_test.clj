(ns raster.compiler.report-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.report :as report]
            [raster.linalg.contract :as contract]
            [raster.nn :as nn]
            [raster.runtime.hardware :as hardware]))

(use-fixtures
  :once
  (fn [f]
    (hardware/register-target-device!
     :ze:compiler-report
     {:name "Synthetic Intel GPU for compiler-report tests"
      :capabilities {:total-eus 64
                     :threads-per-eu 8
                     :simd-width 16
                     :subgroup-sizes [16 32]
                     :max-workgroup-size 1024
                     :shared-local-memory 131072}})
    (f)))

(deftest normalization-is-pure-and-does-not-infer-residency
  (let [normalized
        (report/from-pipeline
         {:backend :opencl
          :soac-fused-stats {:route :typed-soac :typed-validated true
                             :vertical 2 :horizontal 1 :iterations 3}
          :segop-lowered-stats {:segops-lowered 2 :kernel-graphs-lowered 1
                                :typed-soac-reused 3 :typed-scalar-equations 4}
          :backend-applied-stats {:segop-reused 2 :segop-relowered 1 :fallback 0
                                  :segop-declined [{:reason :missing-rule}]}
          :kernels [{:target :opencl-c
                     :attributes {:strategy :portable
                                  :fallback-reason :symbolic-dims
                                  :declines [{:leaf :tensorized :reason :symbolic-dims}]}}]})]
    (is (= {:backend :opencl :source-dialect :typed-soac :typed-validated true
            :declines [{:stage :backend-applied-stats
                        :kind :segop-declined
                        :reason :missing-rule}]}
           (:route normalized)))
    (is (= {:vertical 2 :horizontal 1 :iterations 3 :placements 0}
           (:fusion normalized)))
    (is (= {:segops 2 :kernel-graphs 1 :typed-reused 3 :typed-scalar-equations 4
            :backend-reused 2 :backend-relowered 1 :fallback 0}
           (:lowering normalized)))
    (is (= {:kernel-count 1 :targets [:opencl-c] :strategies [:portable]
            :fallback-reasons [:symbolic-dims]
            :declines [{:stage :emission :kind :candidate
                        :leaf :tensorized :reason :symbolic-dims}]}
           (:emission normalized)))
    (is (= {:assessed? false :resident? nil :device-scratch-count nil
            :host-array-allocs-in-compute nil :internal-host-roundtrips nil}
           (:residency normalized)))))

(deftest representative-dense-pipeline-exposes-current-compatibility-debt
  (let [r (pipeline/compile-report #'nn/predict-fn)]
    (testing "functional fusion stays on the typed route"
      (is (= {:backend :simd :source-dialect :typed-soac
              :typed-validated true :declines []}
             (:route r)))
      (is (= 1 (get-in r [:fusion :vertical]))))
    (testing "remaining JVM re-lowering and fallback are explicit, exact debt"
      (is (= {:segops 2 :kernel-graphs 0 :typed-reused 2 :typed-scalar-equations 4
              :backend-reused 2 :backend-relowered 2 :fallback 2}
             (:lowering r))))))

(deftest representative-symbolic-contraction-records-its-real-kernelbody-route
  (let [r (pipeline/compile-report #'contract/contract-mm
                                   :target-device :ze:compiler-report
                                   :dtype :float)]
    (is (= {:backend :opencl :source-dialect :compatibility
            :typed-validated false :declines []}
           (:route r)))
    (is (= {:segops 1 :kernel-graphs 0 :typed-reused 0 :typed-scalar-equations 0
            :backend-reused 1 :backend-relowered 0 :fallback 0}
           (:lowering r)))
    (is (= 1 (get-in r [:emission :kernel-count])))
    (is (= [:opencl-c] (get-in r [:emission :targets])))
    (is (= [:portable-segred] (get-in r [:emission :strategies])))
    (is (= #{[:dpas :matrix-capability-unavailable]
             [:regtiled :symbolic-dims]}
           (set (map (juxt :leaf :reason) (get-in r [:emission :declines])))))))
