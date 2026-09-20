(ns raster.perf.production-canary-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.gpu.link :as link]
            [raster.perf.production-canary :as canary]
            [raster.runtime.hardware :as hardware]
            [raster.runtime.microbench :as microbench]
            [raster.gpu.compiled :as compiled]
            [raster.compiler.core.hardware :as compiler-hardware]
            [raster.compiler.pipeline :as pipeline]
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

(deftest launch-aware-roofline-is-an-explicit-cliff-contract
  (let [descriptor {:peak-flops {:f32 1.0e12}
                    :bandwidth-bytes-s 1.0e11
                    :launch-overhead-ns 1000.0}
        work {:flops 5120 :warm-bytes 10240 :dtype :f32 :n-kernels 1}
        pass (canary/roofline-assessment descriptor work {:median-ns 10000.0} 30.0)
        regression (canary/roofline-assessment descriptor work {:median-ns 50000.0} 30.0)]
    (is (= :pass (:status pass)))
    (is (= :regression (:status regression)))
    (is (= :unmodelled
           (:status (canary/roofline-assessment {} work {:median-ns 1.0} 30.0))))
    (is (= :roofline-regression
           (canary/verdict nil {:identity {} :validated? true
                                :measurement {:median-ns 50000.0 :stationary? true}
                                :performance-contract regression})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (canary/roofline-assessment descriptor work {:median-ns 1.0} 0.0)))))

(deftest equation-first-rmsnorm-canary-keeps-validation-and-device-timing-on-public-artifact
  (let [calls (atom [])]
    (with-redefs [hardware/init! (constantly nil)
                  hardware/device-signature (fn [target] {:target target})
                  compiler-hardware/descriptor-for
                  (constantly {:peak-flops {:f32 1.0e12}
                               :bandwidth-bytes-s 1.0e11
                               :launch-overhead-ns 1000.0})
                  compiled/lower
                  (fn [entry args opts]
                    (swap! calls conj [:lower entry opts])
                    {:args args :schedule {:compiler :equation-first}})
                  compiled/instantiate!
                  (fn [prepared opts]
                    (swap! calls conj [:instantiate opts])
                    {:prepared prepared})
                  compiled/profile
                  (fn [live]
                    {:result {:out (canary/rmsnorm-reference
                                    (get-in live [:prepared :args]))}
                     :profile [{:kernel-name "generated-rmsnorm" :ms 0.01}]
                     :kernel-total-ms 0.01 :device-wall-ms 0.01})
                  compiled/measure
                  (fn [_ & opts]
                    (swap! calls conj [:measure (apply hash-map opts)])
                    {:median-ns 10000.0 :stationary? true :timing-source :device-event})
                  compiled/ir (constantly [{:convention :kernel-body}])
                  compiled/close! (fn [_] (swap! calls conj [:close]))]
      (let [result (canary/rmsnorm! {:target :ze:0 :shape [1 16]
                                     :environment-tag "fixture" :compiler-revision "test"
                                     :budget-ms 7.0})
            lower-options (nth (first @calls) 2)]
        (is (:validated? result))
        (is (= :equation-first (:compiler lower-options)))
        (is (= '[weight] (:constants lower-options)))
        (is (= {:profile? true} (second (second @calls))))
        (is (= 7.0 (get-in (nth @calls 2) [1 :budget-ms])))
        (is (= :device-event (get-in result [:measurement :timing-source])))
        (is (= :pass (get-in result [:performance-contract :status])))
        (is (= [:close] (last @calls)))))))

(deftest dynamic-prebound-composition-retains-one-generated-step
  (let [shape [3 4 5]
        args (canary/gemm-arguments shape)
        prepared (canary/prepare-gemm :ocl:0 args shape
                                      {:variant :relu-prebound :gemm-precision :f32-scalar})
        evidence (canary/compilation-evidence prepared)]
    (is (= 1 (:resident-step-count evidence)))
    (is (every? #(= {:kernel-body (:entry-point-count %)} (:emission-routes %))
                (mapcat :alternatives (:steps evidence))))
    (if-not @probe/opencl-available?
      (probe/opencl-skip! "dynamic prebound public GEMM plus map fusion")
      (let [live (compiled/instantiate! prepared)]
        (try
          (let [resident (:executable live)
                output (some #(when (= 'C (:sym %)) %) (:out-tree live))
                expected (mapv #(max (float 0.0) %)
                               (canary/gemm-reference (first args) (second args) shape))]
            (doseq [_ (range 2)]
              (link/upload! resident (:node output) (float-array (repeat 12 Float/NaN)))
              (link/run! resident)
              (is (= expected (vec (link/download resident (:node output)))))))
          (finally (compiled/close! live)))))))

(deftest public-prebound-extent-overflow-precedes-output-mutation
  (let [f (pipeline/compile-aot #'canary/gemm-relu-prebound! :dtype :float)
        output (float-array [17])]
    (is (thrown? ArithmeticException
                 (f (float-array 0) (float-array 0) output Long/MAX_VALUE 2 0)))
    (is (= [17.0] (vec output)))))

(deftest compilation-evidence-retains-existing-dispatch-declines
  (let [diagnostics {:selection :analytic-fixed
                     :declines [{:reason :symbolic-dims}]
                     :matrix-graph-decline {:reason :mixed-dpas-target-capability}}
        prepared {:descriptor {:steps [{:convention :executable
                                         :dispatch {:alternatives [] :attributes diagnostics}}]}}]
    (is (= diagnostics
           (get-in (canary/compilation-evidence prepared) [:steps 0 :dispatch-diagnostics])))))

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
    (is (every? #(contains? #{:contract :executable} (:convention %))
                (get-in p [:descriptor :steps]))
        "the public path may retain the typed contraction or its executable wrapper")
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
