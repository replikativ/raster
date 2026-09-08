(ns raster.perf.production-canary-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.gpu.link :as link]
            [raster.perf.production-canary :as canary]
            [raster.runtime.microbench :as microbench]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as probe]))

(defn- once-only [f & _]
  (f)
  {:median-ns 100 :stationary? true})

(deftm static-gemm-relu! [A :- (Array float) B :- (Array float) C :- (Array float)] :- (Array float)
  (let [product (raster.par/contract C [[i 3] [j 4]] [[p 5]]
                 (* (arrays/aget A (+ (* i 5) p)) (arrays/aget B (+ (* p 4) j)))
                 :init (float 0.0))]
    (raster.par/map! C q 12 nil (max (float 0.0) (arrays/aget product q)))))

(deftest static-public-composition-fuses-through-the-generated-route
  (let [args (canary/gemm-arguments [3 4 5])
        prepared (compiled/lower #'static-gemm-relu! args
                                 {:target :ocl:0 :dtype :float :constants ['A 'B]
                                  :gemm-precision :f32-scalar :on-non-resident :throw})
        evidence (canary/compilation-evidence prepared)]
    (is (= 1 (:resident-step-count evidence)))
    (is (every? #(= {:kernel-body (:entry-point-count %)} (:emission-routes %))
                (mapcat :alternatives (:steps evidence))))
    (if-not @probe/opencl-available?
      (probe/opencl-skip! "static public GEMM plus map fusion")
      (let [live (compiled/instantiate! prepared)]
        (try
          (link/run! (:executable live))
          (let [output (first (:out-tree live))
                actual (vec (link/download (:executable live) (:node output)))
                expected (mapv #(max (float 0.0) %)
                               (canary/gemm-reference (first args) (second args) [3 4 5]))]
            (is (= expected actual)))
          (finally (compiled/close! live)))))))

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

(deftest gemm-shape-inputs-and-rejection
  (is (= [15 20 12] (mapv alength (canary/gemm-arguments [3 4 5]))))
  (is (= [22.0 28.0 49.0 64.0]
         (vec (canary/gemm-reference (float-array [1 2 3 4 5 6])
                                     (float-array [1 2 3 4 5 6]) [2 2 3]))))
  (doseq [shape [[0 4 5] [-1 4 5] [3 4] [3 4 1.5] [Integer/MAX_VALUE 2 1]]]
    (is (thrown? clojure.lang.ExceptionInfo (canary/gemm-arguments shape)))))

(deftest opencl-generated-gemm-epilogue-and-policy
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "public generated GEMM epilogue and precision policy")
    (with-redefs [microbench/do-bench once-only]
      (doseq [precision [:f32-scalar :mixed-f16-f32]]
        (let [result (canary/gemm! {:shape [3 4 5] :variant :relu
                                   :gemm-precision precision :target :ocl:0
                                   :environment-tag "ci-correctness-only"})
              evidence (:compilation result)
              candidates (mapcat :alternatives (:steps evidence))]
          (is (:validated? result))
          (is (= precision (get-in result [:identity :numerical-policy])))
          (is (= :gemm-relu-resident (get-in result [:identity :workload])))
          (is (= 1 (:resident-step-count evidence)))
          (is (seq candidates))
          (is (every? #(= {:kernel-body (:entry-point-count %)} (:emission-routes %)) candidates)))))))

(deftest composed-return-alias-uses-a-pointwise-inout-boundary
  (let [prepared (compiled/lower #'canary/gemm-relu-composed!
                                 (into (canary/gemm-arguments [3 4 5]) [3 4 5])
                                 {:target :ocl:0 :dtype :float :constants ['A 'B]
                                  :gemm-precision :f32-scalar :on-non-resident :throw})]
    (is (= 2 (get-in (canary/compilation-evidence prepared) [:resident-step-count])))
    (if-not @probe/opencl-available?
      (probe/opencl-skip! "composed GEMM return-alias numerics")
      (with-redefs [microbench/do-bench once-only]
        (let [result (canary/gemm! {:shape [3 4 5] :variant :relu-composed
                                   :gemm-precision :f32-scalar :target :ocl:0
                                   :environment-tag "ci-correctness-only"})]
          (is (:validated? result))
          (is (= :gemm-relu-composed-resident (get-in result [:identity :workload]))))))))

(deftest opencl-parameterized-gemm-shape-canary
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "parameterized production GEMM shape numerics")
    (with-redefs [microbench/do-bench once-only]
      (doseq [shape [[1 7 5] [3 4 5]]]
        (let [result (canary/gemm! {:shape shape :target :ocl:0
                                   :environment-tag "ci-correctness-only"})]
          (is (:validated? result))
          (is (= shape (get-in result [:identity :shape])))
          (is (= :gemm-mnk-resident (get-in result [:identity :workload])))
          (is (seq (:execution result))))))))
