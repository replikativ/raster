(ns raster.compiler.ir.kernel-dispatch-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.core :as gpu]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

(def ^:private abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'width :scalar :long :role :shape)])

(defn- artifact
  [name strategy workgroup]
  (kart/make
   {:kernel-name name
    :source (str "__kernel void " name
                 "(__global const float* x, __global float* out, long width) {}")
    :abi abi
    :arguments '[x out width]
    :launch (klaunch/spec {:workgroup-size [workgroup]
                           :group-count [(klaunch/ceil-div 'width workgroup)]})
    :effects {:kind :map :reads ['x] :writes ['out]}
    :attributes {:strategy strategy}}))

(def ^:private reference
  (artifact "dispatch_reference" :reference 64))

(def ^:private subgroup
  (artifact "dispatch_subgroup" :subgroup-score-reuse 16))

(def ^:private dispatch
  (kdispatch/make
   {:id "dispatch-test"
    :alternatives [reference subgroup]
    :default-strategy :reference
    :selector {:kind :runtime-scalar-threshold
               :argument 'width
               :threshold 256
               :at-least :subgroup-score-reuse
               :otherwise :reference}}))

(deftest runtime-scalars-select-an-abi-compatible-artifact
  (is (kdispatch/kernel-dispatch? dispatch))
  (is (= "dispatch_reference"
         (:kernel-name (kdispatch/select-artifact dispatch [:x :out {:type :long :value 128}]))))
  (is (= "dispatch_subgroup"
         (:kernel-name (kdispatch/select-artifact dispatch [:x :out {:type :long :value 256}]))))
  (is (= "dispatch_subgroup"
         (:kernel-name (kdispatch/select-artifact dispatch [:x :out {:type :long :value 2}]
                                                  :subgroup-score-reuse)))
      "an explicit schedule override is authoritative"))

(deftest dispatch-rejects-incompatible-or-ambiguous-alternatives
  (testing "an alternative cannot silently change the ordered ABI"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"share target, ABI"
         (kdispatch/make
          {:id "bad-abi"
           :alternatives [reference (assoc subgroup :arguments '[out x width])]
           :default-strategy :reference
           :selector (:selector dispatch)}))))
  (testing "strategy identity is unique"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"strategies must be unique"
         (kdispatch/make
          {:id "duplicate"
           :alternatives [reference (assoc subgroup :attributes {:strategy :reference})]
           :default-strategy :reference
           :selector (:selector dispatch)})))))

(deftest both-resident-backends-register-the-same-pure-dispatch
  (doseq [[register! entry] [[ze/register-kernel-dispatch!
                              ze/kernel-dispatch-registry-entry]
                             [ocl/register-kernel-dispatch!
                              ocl/kernel-dispatch-registry-entry]]]
    (register! dispatch)
    (is (identical? dispatch (entry (:id dispatch))))))

(deftest resident-step-selects-before-the-backend-binder
  (let [step {:kernel-name (:kernel-name reference)
              :phase :probe
              :convention :contract
              :artifact reference
              :dispatch dispatch
              :argument-specs [{:kind :input :sym 'x}
                               {:kind :output :sym 'out}
                               {:kind :scalar :type :long
                                :value-fn (fn [args] (:width args))}]}
        selected
        (fn [width]
          (let [session (atom {:device-id :probe
                               :buffers {'x :resident-x 'out :resident-out}})]
            (with-redefs-fn
              {#'raster.gpu.core/rt-resolve
               (fn [_ function-name]
                 (case function-name
                   "bind-kernel-call" identity
                   (throw (ex-info "unexpected runtime resolution"
                                   {:function function-name}))))}
              #(gpu/bind-step! session step {:width width} identity))
            (get-in @session [:prepared :probe :artifact :attributes :strategy])))]
    (is (= :reference (selected 128)))
    (is (= :subgroup-score-reuse (selected 256)))))
