(ns raster.gpu.dispatch-benchmark-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-benchmark :as benchmark]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.tuning-cache :as cache])
  (:import [java.nio.file Files]))

(def ^:private abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'width :scalar :long :role :shape)])

(defn- emitted
  [kernel-name strategy workgroup]
  (artifact/make
   {:kernel-name kernel-name
    :source (str "__kernel void " kernel-name
                 "(__global const float* x, __global float* out, long width) {}")
    :abi abi
    :arguments '[x out width]
    :launch (launch/spec {:workgroup-size [workgroup]
                          :group-count [(launch/ceil-div 'width workgroup)]})
    :effects {:kind :elementwise-map :reads ['x] :writes ['out]}
    :attributes {:strategy strategy}}))

(def ^:private reference (emitted "benchmark_reference" :reference 64))
(def ^:private subgroup (emitted "benchmark_subgroup" :subgroup 16))

(def ^:private dispatch
  (kdispatch/make
   {:id "resident-benchmark-test"
    :alternatives [reference subgroup]
    :default-strategy :reference
    :selector {:kind :runtime-scalar-threshold
               :argument 'width :threshold 256
               :at-least :subgroup :otherwise :reference}}))

(def ^:private descriptor
  {:device-id :ze:0 :vendor "Intel" :arch "test" :driver-version "test-driver"
   :machine-lanes 8192 :subgroup-size 16})

(defn- temporary-cache-root
  []
  (.toFile (Files/createTempDirectory "raster-dispatch-benchmark-"
                                      (make-array java.nio.file.attribute.FileAttribute 0))))

(deftest resident-driver-validates-before-device-measurement
  (let [actions (atom [])
        result
        (with-redefs [gpu/bind-kernel-call!
                      (fn [_ key candidate arguments opts]
                        (swap! actions conj [:bind key candidate arguments opts])
                        {:candidate candidate})
                      gpu/run-kernel-graph!
                      (fn [_ handle]
                        (swap! actions conj [:validate-run (:candidate handle)])
                        {'out :resident-output})
                      gpu/measure-bound-kernel-graph!
                      (fn [_ handle & {:keys [before-sample! compile-ms hashes]}]
                        (swap! actions conj [:measure (:candidate handle) compile-ms hashes])
                        (when before-sample! (before-sample!))
                        (measurement/summarize [10 10 10]
                                               :timing-source :device-event
                                               :compile-ms compile-ms :hashes hashes))
                      gpu/release-kernel-graph!
                      (fn [_ handle] (swap! actions conj [:release (:candidate handle)]))]
          (benchmark/benchmark-candidate!
           (atom {}) dispatch reference 128
           (fn [_]
             {:arguments [:x :out {:type :long :value 128}]
              :before-run! #(swap! actions conj [:restore (:runtime-value %)])
              :validate! (fn [{:keys [outputs]}]
                           (is (= {'out :resident-output} outputs))
                           {:passed? true :oracle-hash "host-reference-v1"})})))]
    (is (= :device-event (get-in result [:measurement :timing-source])))
    (is (= "host-reference-v1" (get-in result [:validation :oracle-hash])))
    (is (= (:source-hash (tuning/artifact-signature reference))
           (get-in result [:validation :candidate-hash])))
    (is (= [:bind :restore :validate-run :measure :restore :release]
           (mapv first @actions)))
    (is (number? (get-in result [:measurement :compile-ms])))))

(deftest failed-oracle-is-never-measured-and-always-releases
  (let [measured? (atom false)
        released? (atom false)]
    (with-redefs [gpu/bind-kernel-call! (fn [& _] :handle)
                  gpu/run-kernel-graph! (fn [& _] {})
                  gpu/measure-bound-kernel-graph! (fn [& _] (reset! measured? true))
                  gpu/release-kernel-graph! (fn [& _] (reset! released? true))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"failed its oracle"
           (benchmark/benchmark-candidate!
            (atom {}) dispatch reference 128
            (fn [_]
              {:arguments [:x :out {:type :long :value 128}]
               :validate! (constantly {:passed? false :oracle-hash "host-reference-v1"})}))))
      (is (false? @measured?))
      (is @released?))))

(deftest explicit-reset-is-required-for-read-write-candidates
  (let [effects {:kind :stateful :reads ['out] :writes ['out]}
        stateful (assoc reference :effects effects)
        stateful-dispatch (assoc dispatch :alternatives
                                 [stateful (assoc subgroup :effects effects)])]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"require :before-run!"
         (benchmark/benchmark-candidate!
          (atom {}) stateful-dispatch stateful 128
          (fn [_]
            {:arguments [:x :out {:type :long :value 128}]
             :validate! (constantly {:passed? true :oracle-hash "oracle"})}))))))

(deftest end-to-end-driver-produces-a-nonmonotonic-measured-selector
  (binding [cache/*cache-root* (temporary-cache-root)]
    (let [costs {:reference {128 10.0 256 10.0 768 5.0}
                 :subgroup {128 12.0 256 7.0 768 6.0}}
          binds (atom 0)
          releases (atom 0)]
      (with-redefs [gpu/bind-kernel-call!
                    (fn [_ _ candidate arguments _]
                      (swap! binds inc)
                      {:candidate candidate
                       :runtime-value (get-in arguments [2 :value])})
                    gpu/run-kernel-graph! (fn [& _] {'out :resident-output})
                    gpu/measure-bound-kernel-graph!
                    (fn [_ {:keys [candidate runtime-value]}
                         & {:keys [compile-ms hashes before-sample!]}]
                      (when before-sample! (before-sample!))
                      (let [cost (get-in costs [(kdispatch/artifact-strategy candidate)
                                                runtime-value])]
                        (measurement/summarize [cost cost cost]
                                               :timing-source :device-event
                                               :compile-ms compile-ms :hashes hashes)))
                    gpu/release-kernel-graph! (fn [& _] (swap! releases inc))]
        (let [result
              (benchmark/tune-dispatch!
               (atom {}) dispatch descriptor [768 128 256]
               (fn [runtime-value]
                 {:arguments [:x :out {:type :long :value runtime-value}]
                  :validate! (constantly {:passed? true
                                          :oracle-hash (str "oracle-" runtime-value)})})
               :numerical-mode {:input :f32 :accumulate :f32 :output :f32}
               :layout {:input :row-major :output :row-major}
               :force? true)]
          (is (= [{:at-least 256 :strategy :subgroup}
                  {:at-least 768 :strategy :reference}]
                 (get-in result [:selector :ranges])))
          (is (= 6 @binds @releases))
          (testing "cached result bypasses resident execution"
            (let [cached (benchmark/tune-dispatch!
                          (atom {}) dispatch descriptor [128 256 768]
                          (fn [_] (throw (ex-info "must not construct a case" {})))
                          :numerical-mode {:input :f32 :accumulate :f32 :output :f32}
                          :layout {:input :row-major :output :row-major})]
              (is (= (:selector result) (:selector cached)))
              (is (= 6 @binds @releases)))))))))

(deftest compiled-program-bridge-projects-a-case-and-returns-recompilation-data
  (let [tuning-contract
        {:schedule-path [:generic-reduction :measured-selectors]
         :schedule-key "resident-benchmark-test"
         :numerical-mode {:input :f32 :accumulate :f32 :output :f32}
         :layout {:input :contiguous :output :contiguous}}
        compiled-dispatch (assoc-in dispatch [:attributes :tuning] tuning-contract)
        program-descriptor
        {:all-params '[x width]
         :array-params '[x]
         :scalar-params '[width]
         :steps [{:phase :reduce
                  :convention :contract
                  :dispatch compiled-dispatch
                  :artifact reference
                  :abi abi
                  :argument-specs
                  [{:slot (nth abi 0) :kind :input :sym 'x}
                   {:slot (nth abi 1) :kind :output :sym 'out}
                   {:slot (nth abi 2) :kind :scalar :type :long :expression 'width
                    :value-fn (fn [args] (nth args 1))}]}]}
        session (atom {:programs {:compiled {:descriptor program-descriptor
                                             :param->key {'x :resident-x}
                                             :alloc->key {'out :resident-out}}}
                       :buffers {:resident-x (Object.) :resident-out (Object.)}})
        measured-selector {:kind :runtime-scalar-ranges
                           :argument 'width :below :reference
                           :ranges [{:at-least 256 :strategy :subgroup}]}
        tuning-identity (tuning/tuning-identity
                         compiled-dispatch descriptor [128 256]
                         (:numerical-mode tuning-contract) (:layout tuning-contract) 0.001)
        fake-tuning (tuning/->DispatchTuning
                     (tuning/cache-key tuning-identity) tuning-identity measured-selector [])
        observed (atom nil)
        result
        (with-redefs [benchmark/tune-dispatch!
                      (fn [_ passed-dispatch passed-descriptor runtime-values passed-case-fn
                           & options]
                        (let [case (passed-case-fn 256)]
                          (reset! observed
                                  {:dispatch passed-dispatch
                                   :descriptor passed-descriptor
                                   :runtime-values runtime-values
                                   :case case
                                   :options (apply hash-map options)}))
                        fake-tuning)]
          (benchmark/tune-program-dispatch!
           session program-descriptor descriptor [128 256]
           (fn [width]
             {:program-arguments [(float-array 1024) width]
              :validate! (constantly {:passed? true :oracle-hash "compiled-reference"})})))]
    (is (= [:resident-x :resident-out {:type :long :value 256}]
           (get-in @observed [:case :arguments])))
    (is (= 256 (get-in @observed
                       [:case :compiled-binding :reference-inputs :scalars 'width])))
    (is (= (:numerical-mode tuning-contract)
           (get-in @observed [:options :numerical-mode])))
    (is (= (:layout tuning-contract) (get-in @observed [:options :layout])))
    (is (= {:generic-reduction
            {:measured-selectors {"resident-benchmark-test" measured-selector}}}
           (:schedule-override result)))
    (is (= measured-selector (:selector result)))
    (is (= :reduce (:phase result)))))

(deftest schedule-override-rejects-dispatch-without-an-emitter-path
  (let [fake-tuning
        (tuning/->DispatchTuning
         "key" {} {:kind :runtime-scalar-ranges :argument 'width
                   :below :reference :ranges []} [])]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"does not declare a tuning schedule path"
         (benchmark/tuning-schedule-override dispatch fake-tuning descriptor
                                             {:input :f32} {:input :contiguous})))))
