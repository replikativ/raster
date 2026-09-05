(ns raster.perf.production-canary-test
  (:require [clojure.test :refer [deftest is]]
            [raster.perf.production-canary :as canary]
            [raster.runtime.microbench :as microbench]
            [raster.gpu.device-probe :as probe]))

(defn- once-only [f & _]
  (f)
  {:median-ns 100 :stationary? true})

(deftest explicit-baseline-and-comparability
  (let [sample {:identity {:workload :example :environment-tag "fixture"}
                :validated? true :measurement {:median-ns 100 :stationary? true}}]
    (is (= :pass (canary/verdict sample sample)))
    (is (= :regression (canary/verdict sample (assoc-in sample [:measurement :median-ns] 116))))
    (is (= :unbaselined (canary/verdict nil sample)))
    (is (= :incomparable (canary/verdict sample (assoc-in sample [:identity :environment-tag] "other"))))
    (is (= :nonstationary (canary/verdict sample (assoc-in sample [:measurement :stationary?] false))))
    (is (= :invalid-baseline (canary/verdict (assoc sample :validated? false) sample)))
    (doseq [v [nil 0 -1 ##NaN ##Inf]]
      (is (= :invalid-measurement
             (canary/verdict sample (assoc-in sample [:measurement :median-ns] v)))))))

(deftest cpu-canary-executes-the-public-aot-route
  ;; Real compilation and independent reference; no wall-time assertion in CI.
  (with-redefs [microbench/do-bench once-only]
    (let [result (canary/cpu! {:environment-tag "ci-correctness-only"})]
      (is (:validated? result))
      (is (= :sumsq-aot (get-in result [:identity :workload]))))))

(deftest opencl-resident-gemm-canary-numerics
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "production canary resident GEMM numerics")
    (with-redefs [microbench/do-bench once-only]
      (let [result (canary/gemm! {:target :ocl:0 :environment-tag "ci-correctness-only"})]
        (is (:validated? result))
        (is (= :host-synchronized-replay (get-in result [:identity :timing-source])))
        (is (seq (:execution result)))))))
