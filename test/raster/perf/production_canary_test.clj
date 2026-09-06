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

(deftest direct-contraction-result-is-a-public-resident-buffer
  (let [p (canary/prepare-gemm :ocl:0 (canary/gemm-arguments))
        evidence (canary/compilation-evidence p)
        candidates (mapcat :alternatives (:steps evidence))]
    (is (= 'C (get-in p [:descriptor :result-sym])))
    (is (= ['C] (mapv :sym (:out-tree p))))
    (is (every? #(= :executable (:convention %)) (get-in p [:descriptor :steps])))
    (is (= 1 (:resident-step-count evidence)))
    (is (= 0 (:descriptor-scratch-count evidence)))
    (is (seq candidates))
    (is (every? #(= {:kernel-body (:entry-point-count %)} (:emission-routes %)) candidates))
    (is (every? #(string? (get-in % [:signature :source-hash])) candidates))
    (is (= evidence (canary/compilation-evidence p))
        "evidence identifies the same already-compiled alternatives")))

(deftest opencl-resident-gemm-canary-numerics
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "production canary resident GEMM numerics")
    (with-redefs [microbench/do-bench once-only]
      (let [result (canary/gemm! {:target :ocl:0 :environment-tag "ci-correctness-only"})]
        (is (:validated? result))
        (is (= :host-synchronized-replay (get-in result [:identity :timing-source])))
        (is (seq (:execution result)))))))
