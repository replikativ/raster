(ns raster.compiler.ir.kernel-dispatch-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
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

(defn- staged-graph
  []
  (let [temporary 'dispatch-temporary
        stage (fn [kernel-name input output]
                (kart/make
                 {:kernel-name kernel-name
                  :source (str "__kernel void " kernel-name
                               "(__global const float* " input ", __global float* " output
                               ", long width) {}")
                  :abi [(kabi/slot input :input :float)
                        (kabi/slot output :output :float)
                        (kabi/slot 'width :scalar :long)]
                  :arguments [input output 'width]
                  :launch (klaunch/spec {:workgroup-size [64]
                                         :group-count [(klaunch/ceil-div 'width 64)]})
                  :effects {:kind :stage}}))
        first-stage (stage "dispatch_stage_one" 'x temporary)
        second-stage (stage "dispatch_stage_two" temporary 'out)]
    (kgraph/make
     {:inputs [(kgraph/buffer 'x :float 'width :device :input)]
      :outputs [(kgraph/buffer 'out :float 'width :device :output)]
      :temporaries [(kgraph/buffer temporary :float 'width :device :temporary)]
      :abi abi
      :arguments '[x out width]
      :nodes [(kgraph/->ScheduledKernel
               :stage-one first-stage
               [(kgraph/->ValueUse 'x :read)
                (kgraph/->ValueUse temporary :write)] [])
              (kgraph/->ScheduledKernel
               :stage-two second-stage
               [(kgraph/->ValueUse temporary :read)
                (kgraph/->ValueUse 'out :write)] [:stage-one])]
      :effects (:effects reference)
      :attributes {:strategy :two-stage}})))

(deftest runtime-scalars-select-an-abi-compatible-artifact
  (is (kdispatch/kernel-dispatch? dispatch))
  (is (= "dispatch_reference"
         (:kernel-name (kdispatch/select-alternative dispatch [:x :out {:type :long :value 128}]))))
  (is (= "dispatch_subgroup"
         (:kernel-name (kdispatch/select-alternative dispatch [:x :out {:type :long :value 256}]))))
  (is (= "dispatch_subgroup"
         (:kernel-name (kdispatch/select-alternative dispatch [:x :out {:type :long :value 2}]
                                                     :subgroup-score-reuse)))
      "an explicit schedule override is authoritative"))

(deftest measured-runtime-ranges-select-without-runtime-tuning
  (let [measured (kdispatch/with-selector
                   dispatch
                   {:kind :runtime-scalar-ranges
                    :argument 'width
                    :below :reference
                    :ranges [{:at-least 192 :strategy :subgroup-score-reuse}
                             {:at-least 768 :strategy :reference}]})
        select #(kdispatch/alternative-strategy
                 (kdispatch/select-alternative measured [:x :out {:type :long :value %}]))]
    (is (= :reference (select 128)))
    (is (= :subgroup-score-reuse (select 192)))
    (is (= :subgroup-score-reuse (select 512)))
    (is (= :reference (select 768)))
    (is (= :subgroup-score-reuse
           (kdispatch/alternative-strategy
            (kdispatch/select-alternative measured [:x :out {:type :long :value 1}]
                                          :subgroup-score-reuse)))
        "an explicit override remains authoritative over measured selector data")))

(deftest ordered-expression-cases-select-from-checked-shape-arithmetic
  (let [scheduled
        (kdispatch/with-selector
          dispatch
          {:kind :runtime-expression-cases
           :cases [{:expression 'width :op :< :value 8 :strategy :reference}
                   {:expression (klaunch/product 'width 2)
                    :op :>= :value 512 :strategy :subgroup-score-reuse}]
           :default :reference})
        select #(kdispatch/alternative-strategy
                 (kdispatch/select-alternative
                  scheduled [:x :out {:type :long :value %}]))]
    (is (= :reference (select 4)))
    (is (= :reference (select 128)))
    (is (= :subgroup-score-reuse (select 256)))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"outside its scalar ABI"
         (kdispatch/with-selector
           dispatch
           {:kind :runtime-expression-cases
            :cases [{:expression 'not-in-the-abi :op :> :value 0 :strategy :reference}]
            :default :subgroup-score-reuse})))))

(deftest dispatch-preserves-one-interface-across-single-and-multi-kernel-schedules
  (let [graph (staged-graph)
        mixed (kdispatch/make
               {:id "mixed-executable-dispatch"
                :alternatives [reference graph]
                :default-strategy :reference
                :selector {:kind :runtime-scalar-threshold
                           :argument 'width :threshold 256
                           :at-least :two-stage :otherwise :reference}})]
    (is (= :kernel-artifact
           (kexec/kind (kdispatch/select-alternative
                        mixed [:x :out {:type :long :value 128}]))))
    (is (= :kernel-graph
           (kexec/kind (kdispatch/select-alternative
                        mixed [:x :out {:type :long :value 256}]))))
    (is (= '[x out width] (kexec/arguments graph)))
    (is (= ['dispatch-temporary] (mapv :id (:temporaries graph))))))

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
           :selector (:selector dispatch)}))))
  (testing "measured range boundaries are ordered and name available strategies"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"strictly increasing"
         (kdispatch/with-selector
           dispatch {:kind :runtime-scalar-ranges :argument 'width :below :reference
                     :ranges [{:at-least 256 :strategy :subgroup-score-reuse}
                              {:at-least 128 :strategy :reference}]})))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"absent strategy"
         (kdispatch/with-selector
           dispatch {:kind :runtime-scalar-ranges :argument 'width :below :unknown
                     :ranges []}))))
  (testing "a runtime selector must name exactly one scalar ABI position"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"must name a scalar"
         (kdispatch/make
          {:id "pointer-selector"
           :alternatives [reference subgroup]
           :default-strategy :reference
           :selector {:kind :runtime-scalar-threshold :argument 'x :threshold 1
                      :at-least :subgroup-score-reuse :otherwise :reference}}))))
  (testing "shared entry points are legal when they name the same module"
    (is (kdispatch/kernel-dispatch?
         (kdispatch/make
          {:id "shared-entry-point"
           :alternatives [reference
                          (assoc subgroup :kernel-name (:kernel-name reference)
                                 :source (:source reference))]
           :default-strategy :reference
           :selector (:selector dispatch)}))))
  (testing "one entry point cannot name conflicting emitted modules"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"conflicting modules"
         (kdispatch/make
          {:id "duplicate-entry-point"
           :alternatives
           [reference
            (assoc subgroup
                   :kernel-name (:kernel-name reference)
                   :source (-> (:source subgroup)
                               (str/replace "dispatch_subgroup" "dispatch_reference")
                               (str/replace "{}" "{ long gid = get_global_id(0); }")))]
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
