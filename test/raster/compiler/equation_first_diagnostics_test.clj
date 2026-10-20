(ns raster.compiler.equation-first-diagnostics-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.distributed-training :as training]
            [raster.dl.loss :as loss]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.pipeline :as pipeline]))

(deftest training-loss-keeps-its-float-checkpoint-before-the-double-quotient
  ;; Returning a Double does not mean that the sum was accumulated in Double.
  ;; These exact Float inputs distinguish the source specialization from an
  ;; accidentally widened accumulation without choosing a parallel tree order.
  (let [prediction (float-array [4096.0 1.0 1.0 1.0])
        target (float-array 4)
        result (loss/mse-loss prediction target 4)
        double-reference (/ (+ 16777216.0 1.0 1.0 1.0) 4.0)
        legal-float-results #{4194304.0 4194304.5 4194305.0}]
    (is (instance? Double result))
    (is (contains? legal-float-results result))
    (is (not= double-reference result))
    (is (Double/isNaN (double (loss/mse-loss (float-array 0) (float-array 0) 0)))
        "an empty reduction still reaches its IEEE Double zero-denominator quotient")))

(deftest coverage-reports-the-actual-normalized-admission-refusal
  (let [attempt (atom nil)
        repeated-analyses (atom 0)
        coverage frontend/coverage-decline
        original pipeline/run-passes-diagnostic
        ;; A pure source-only target: no session, device allocation or vendor
        ;; compiler is necessary to reach semantic admission.
        target :ocl:normalized-coverage
        descriptor {:device-id target :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}
        error (with-redefs [pipeline/run-passes-diagnostic
                            (fn [form passes options start]
                              (let [result (original form passes options start)]
                                (when (= passes pipeline/gpu-semantic-post-soa-passes)
                                  (reset! attempt result))
                                result))
                           frontend/coverage-decline
                           (fn [form options]
                             (when @attempt (swap! repeated-analyses inc))
                             (coverage form options))]
                (try
                  (equation-first/compile #'training/local-gradient-sum
                                          {:target target :dtype :float
                                           :schedule {:precision :f32-scalar}}
                                          descriptor)
                  nil
                  (catch clojure.lang.ExceptionInfo error error)))
        actual (get-in @attempt [:stats :soac-fused :typed-soac-declined])]
    (is (= :equation-first-coverage (:reason (ex-data error))))
    (is (= :typed-soac-source-coverage (:reason actual)))
    (is (= actual (:coverage (ex-data error)))
        "report the refusal from the same pass that decided legality")
    (is (seq (:bindings actual)))
    (is (zero? @repeated-analyses)
        "do not replace actual admission evidence with a second source analysis")))
