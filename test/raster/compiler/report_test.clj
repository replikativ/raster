(ns raster.compiler.report-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.report :as report]))

(deftest normalization-is-pure-and-does-not-infer-residency
  (let [normalized
        (report/from-pipeline
         {:backend :opencl
          :soac-fused-stats {:route :typed-soac :typed-validated true
                             :vertical 2 :horizontal 1 :iterations 3}
          :segop-lowered-stats {:segops-lowered 2 :kernel-graphs-lowered 1
                                :structured-loops-scheduled 1
                                :typed-soac-reused 3 :typed-scalar-equations 4}
          :backend-applied-stats {:segop-reused 2 :segop-relowered 1 :fallback 0
                                  :typed-contraction-dispatch-declines [{}]
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
    (is (= {:segops 2 :kernel-graphs 1 :structured-loops 1
            :typed-reused 3 :typed-scalar-equations 4
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

(deftest specialized-kernel-emission-is-visible-as-compatibility-not-scalar
  (let [r (report/from-pipeline {:backend :opencl
                                 :segop-lowered-stats {:segops-lowered 0}
                                 :backend-applied-stats {:fallback 0}
                                 :kernels [{:target :opencl-c}]})]
    (is (= :compatibility (get-in r [:route :source-dialect])))))

(deftest counted-decline-summaries-retain-their-reasons
  (let [r (report/from-pipeline
           {:backend :opencl
            :backend-applied-stats
            {:typed-contraction-dispatch-declines {:dynamic-scalar 2}}
            :kernels []})]
    (is (= [{:stage :backend-applied-stats
             :kind :typed-contraction-dispatch-declines
             :reason :dynamic-scalar}
            {:stage :backend-applied-stats
             :kind :typed-contraction-dispatch-declines
             :reason :dynamic-scalar}]
           (get-in r [:route :declines])))))
