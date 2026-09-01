(ns raster.compiler.ir.invocation-plan
  "Typed semantic boundary from public function arguments to an internal ParallelProgram.

   ParallelProgram inputs are compiler SSA values: they may include shape projections, narrowed
   scalars, cloned state, and fresh scratch. InvocationPlan retains how ordered public parameters
   produce those values without asking a runtime to inspect the original function body or infer
   storage roles from names. Physical allocation, views, ownership, and transfers remain the job of
   ResidentPlan/LinkPlan lowering."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.scalar.effects :as effects]))

(defrecord InvocationParameter [id symbol value])
(defrecord ShapeProjection [id symbol operands source axis value])
(defrecord ScalarCompute [id symbol operands region value])
(defrecord BufferAllocation [id symbol operands initialization value])
(defrecord BufferClone [id symbol operands source value])
(defrecord ValueAlias [id symbol operands source value])
(defrecord InvocationPlan
           [id parameters steps bindings program-inputs program-outputs values attributes])

(defn- record-kind?
  [class-name value]
  (and value (= class-name (.getName (class value)))))

(defn invocation-parameter? [x]
  (record-kind? "raster.compiler.ir.invocation_plan.InvocationParameter" x))
(defn shape-projection? [x]
  (record-kind? "raster.compiler.ir.invocation_plan.ShapeProjection" x))
(defn scalar-compute? [x]
  (record-kind? "raster.compiler.ir.invocation_plan.ScalarCompute" x))
(defn buffer-allocation? [x]
  (record-kind? "raster.compiler.ir.invocation_plan.BufferAllocation" x))
(defn buffer-clone? [x]
  (record-kind? "raster.compiler.ir.invocation_plan.BufferClone" x))
(defn value-alias? [x]
  (record-kind? "raster.compiler.ir.invocation_plan.ValueAlias" x))
(defn invocation-plan? [x]
  (record-kind? "raster.compiler.ir.invocation_plan.InvocationPlan" x))

(defn invocation-step?
  [x]
  (or (shape-projection? x) (scalar-compute? x)
      (buffer-allocation? x) (buffer-clone? x) (value-alias? x)))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :invocation-plan))))

(defn- distinct-vector?
  [values]
  (and (vector? values) (= (count values) (count (distinct values)))))

(defn- scalar-value?
  [value]
  (and (= :tensor (:kind value)) (empty? (:shape value))))

(defn- buffer-value?
  [value]
  (and (= :tensor (:kind value)) (seq (:shape value))))

(defn- operand-map
  [step]
  (let [operands (:operands step)]
    (when-not (and (vector? operands)
                   (every? #(and (map? %) (symbol? (:symbol %))
                                 (contains? % :value)) operands)
                   (= (count operands) (count (distinct (map :symbol operands)))))
      (fail! :invocation-step-operands
             "invocation step operands must be ordered unique symbol/value references"
             {:step (:id step) :operands operands}))
    (into {} (map (juxt :symbol :value)) operands)))

(defn- compatible-value?
  [left right]
  (and left right
       (= (:kind left) (:kind right))
       (= (:dtype left) (:dtype right))
       (av/storage-contract-compatible? left right)))

(defn- validate-step!
  [step values available]
  (when-not (invocation-step? step)
    (fail! :invocation-step-type "invocation plan contains an unknown step"
           {:step step :actual (type step)}))
  (let [{:keys [id symbol value]} step
        operands (operand-map step)
        operand-values (set (vals operands))]
    (when (or (nil? id) (not (symbol? symbol)))
      (fail! :invocation-step-identity
             "invocation step requires a stable identity and lexical result symbol"
             {:step id :symbol symbol}))
    (when (contains? available id)
      (fail! :invocation-step-redefinition "invocation step result must be fresh SSA"
             {:step id}))
    (when-let [missing (seq (set/difference operand-values available))]
      (fail! :invocation-step-use-before-definition
             "invocation step operand is not a public parameter or earlier result"
             {:step id :values (set missing)}))
    (when-not (= value (get values id))
      (fail! :invocation-step-value
             "invocation step value differs from the plan value table"
             {:step id :declared value :table (get values id)}))
    (av/validate! value)
    (cond
      (shape-projection? step)
      (let [source-value (get values (:source step))]
        (when-not (and (= 1 (count operands))
                       (= (:source step) (first operand-values))
                       (buffer-value? source-value)
                       (integer? (:axis step))
                       (<= 0 (:axis step) (dec (count (:shape source-value))))
                       (scalar-value? value)
                       (contains? #{:int :long} (:dtype value)))
          (fail! :invocation-shape-projection
                 "shape projection requires one shaped source and an integer scalar result"
                 {:step id :source (:source step) :axis (:axis step) :value value})))

      (scalar-compute? step)
      (let [{:keys [parameters locals body-results]} (soac/lambda-parts (:region step))
            parameter-set (set parameters)
            free-symbols (when (= 1 (count body-results))
                           (util/free-syms (first body-results)))]
        (when-not (scalar-value? value)
          (fail! :invocation-scalar-value "scalar computation must produce a rank-zero value"
                 {:step id :value value}))
        (when-not (every? (fn [operand]
                            (scalar-value? (get values (:value operand))))
                          (:operands step))
          (fail! :invocation-scalar-operands
                 "invocation scalar computation operands must all be rank zero"
                 {:step id :operands (:operands step)}))
        (when-not (and (= parameters (mapv :symbol (:operands step)))
                       (empty? locals)
                       (= 1 (count body-results)))
          (fail! :invocation-scalar-region
                 "invocation scalar region must close one result over its ordered operands"
                 {:step id :region (:region step) :operands (:operands step)}))
        (when-let [unbound (seq (set/difference free-symbols parameter-set))]
          (fail! :invocation-scalar-free-symbol
                 "invocation scalar region references a value outside its closed operands"
                 {:step id :region (:region step) :free-symbols (set unbound)
                  :parameters parameter-set}))
        (when-not (= :pure (effects/analyze-effect (first body-results)))
          (fail! :invocation-scalar-effect
                 "invocation scalar computation must be proven pure"
                 {:step id :region (:region step)})))

      (buffer-allocation? step)
      (let [shape-symbols (into #{} (mapcat util/free-syms) (:shape value))]
        (when-not (and (buffer-value? value)
                       (contains? #{:zero :unspecified} (:initialization step))
                       (= shape-symbols (set (map :symbol (:operands step)))))
          (fail! :invocation-buffer-allocation
                 "buffer allocation operands must exactly close its typed shape contract"
                 {:step id :initialization (:initialization step) :value value
                  :shape-symbols shape-symbols
                  :operands (mapv :symbol (:operands step))})))

      (buffer-clone? step)
      (let [source-value (get values (:source step))]
        (when-not (and (buffer-value? value)
                       (= 1 (count operands))
                       (= (:source step) (first operand-values))
                       (compatible-value? source-value value))
          (fail! :invocation-buffer-clone
                 "buffer clone requires one storage-compatible source"
                 {:step id :source (:source step) :value value})))

      (value-alias? step)
      (let [source-value (get values (:source step))]
        (when-not (and (= 1 (count operands))
                       (= (:source step) (first operand-values))
                       (compatible-value? source-value value))
          (fail! :invocation-value-alias
                 "value alias requires one storage-compatible earlier value"
                 {:step id :source (:source step) :value value}))))
    (conj available id)))

(defn validate!
  [plan]
  (when-not (invocation-plan? plan)
    (fail! :invocation-plan-type "expected an InvocationPlan" {:actual (type plan)}))
  (let [{:keys [id parameters steps bindings program-inputs program-outputs values attributes]} plan]
    (when (nil? id)
      (fail! :invocation-plan-id "invocation plan requires a stable identity" {}))
    (when-not (and (vector? parameters) (every? invocation-parameter? parameters)
                   (distinct-vector? (mapv :id parameters))
                   (distinct-vector? (mapv :symbol parameters)))
      (fail! :invocation-parameters
             "public invocation parameters must have unique ordered IDs and symbols"
             {:parameters parameters}))
    (when-not (and (vector? steps) (vector? bindings)
                   (distinct-vector? program-inputs) (distinct-vector? program-outputs)
                   (map? values) (map? attributes))
      (fail! :invocation-plan-fields "invocation plan fields have invalid container types"
             {:steps steps :bindings bindings :inputs program-inputs
              :outputs program-outputs :values values :attributes attributes}))
    (doseq [{:keys [id value] :as parameter} parameters]
      (when-not (= value (get values id))
        (fail! :invocation-parameter-value
               "public parameter differs from the invocation value table"
               {:parameter parameter :table (get values id)}))
      (av/validate! value))
    (let [available (reduce (fn [available step]
                              (validate-step! step values available))
                            (set (map :id parameters)) steps)
          expected-bindings (mapv :program-value bindings)
          invocation-values (mapv :invocation-value bindings)]
      (when-not (= program-inputs expected-bindings)
        (fail! :invocation-program-input-order
               "invocation bindings must exactly follow the internal program input order"
               {:inputs program-inputs :bindings expected-bindings}))
      (when-let [missing (seq (set/difference (set invocation-values) available))]
        (fail! :invocation-program-input-value
               "internal program input is not produced by the invocation prefix"
               {:values (set missing)}))
      (doseq [{program-value :program-value invocation-value :invocation-value} bindings]
        (when-not (compatible-value? (get values program-value) (get values invocation-value))
          (fail! :invocation-program-input-contract
                 "public materialization differs from the internal program value contract"
                 {:program-value program-value :invocation-value invocation-value
                  :program-contract (get values program-value)
                  :invocation-contract (get values invocation-value)})))
      (doseq [output program-outputs]
        (when-not (contains? values output)
          (fail! :invocation-program-output
                 "invocation output has no retained AbstractValue" {:output output}))))
    plan))

(defn- parameter-id [plan-id ordinal symbol]
  [:invocation plan-id :parameter ordinal symbol])

(defn- binding-id [plan-id ordinal symbol]
  [:invocation plan-id :binding ordinal symbol])

(defn- referenced-operands
  [expression environment]
  (let [symbols (sort-by str (util/free-syms expression))]
    (mapv (fn [symbol]
            (if-let [value (get environment symbol)]
              {:symbol symbol :value value}
              (fail! :invocation-prefix-free-value
                     "invocation prefix references a value outside its public lexical boundary"
                     {:symbol symbol :expression expression
                      :available (set (keys environment))})))
          symbols)))

(defn- unwrap-casts
  [expression]
  (loop [expression expression]
    (if (and (seq? expression)
             (descriptor/cast-op? (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
      (recur (first (descriptor/call-args expression)))
      expression)))

(defn- shape-source
  [expression]
  (let [expression (unwrap-casts expression)]
    (when (and (seq? expression)
               (descriptor/alength-op? (descriptor/semantic-op expression))
               (= 1 (count (descriptor/call-args expression)))
               (symbol? (first (descriptor/call-args expression))))
      (first (descriptor/call-args expression)))))

(defn- allocation-operands
  [value environment]
  (->> (:shape value)
       (mapcat #(referenced-operands % environment))
       (reduce (fn [operands operand]
                 (if (some #(= (:symbol %) (:symbol operand)) operands)
                   operands
                   (conj operands operand)))
               [])))

(defn- prefix-step
  [plan-id ordinal symbol expression value environment]
  (let [id (binding-id plan-id ordinal symbol)
        operation (descriptor/semantic-op expression)
        initialization (descriptor/allocation-initialization operation)]
    (cond
      (shape-source expression)
      (let [operands (referenced-operands expression environment)
            source-symbol (shape-source expression)]
        (->ShapeProjection id symbol operands (get environment source-symbol) 0 value))

      (= :copy initialization)
      (let [operands (referenced-operands expression environment)
            source-symbol (some-> expression descriptor/call-args first unwrap-casts)]
        (when-not (symbol? source-symbol)
          (fail! :invocation-clone-source
                 "copy allocation requires one lexical source value"
                 {:symbol symbol :expression expression}))
        (->BufferClone id symbol operands (get environment source-symbol) value))

      initialization
      (->BufferAllocation id symbol (allocation-operands value environment)
                          initialization value)

      (symbol? expression)
      (->ValueAlias id symbol (referenced-operands expression environment)
                    (get environment expression) value)

      (scalar-value? value)
      (let [operands (referenced-operands expression environment)]
        (->ScalarCompute id symbol operands
                         (soac/lambda-form (mapv :symbol operands) [expression]) value))

      :else
      (fail! :invocation-prefix-unsupported
             "typed invocation planning does not support this host materialization"
             {:symbol symbol :expression expression :value value}))))

(defn from-prefix
  "Build a typed invocation plan from a certified flat host prefix.

   `parameters` is the public function order. `parameter-values` describes those values before any
   lexical shadowing; `binding-values` describes prefix results. `program-values`, inputs, and
   outputs are the already validated internal ParallelProgram boundary. Stable SSA IDs keep a
   narrowed binding such as `nsteps = (int nsteps)` distinct from its public long parameter."
  [{:keys [id parameters parameter-values bindings binding-values program-values
           program-inputs program-outputs attributes]
    :or {attributes {}}}]
  (let [parameters (vec parameters)
        public
        (mapv (fn [ordinal symbol]
                (let [value (get parameter-values symbol)]
                  (when-not value
                    (fail! :invocation-parameter-contract
                           "public parameter has no retained AbstractValue"
                           {:parameter symbol :parameters parameters}))
                  (->InvocationParameter (parameter-id id ordinal symbol) symbol value)))
              (range) parameters)
        initial-values (into {} (map (juxt :id :value)) public)
        initial-environment (into {} (map (juxt :symbol :id)) public)
        {:keys [steps values environment]}
        (reduce (fn [{:keys [steps values environment]} [ordinal [symbol expression]]]
                  (let [value (get binding-values symbol)]
                    (when-not value
                      (fail! :invocation-prefix-value
                             "invocation prefix binding has no retained AbstractValue"
                             {:symbol symbol :expression expression}))
                    (let [step (prefix-step id ordinal symbol expression value environment)]
                      {:steps (conj steps step)
                       :values (assoc values (:id step) value)
                       :environment (assoc environment symbol (:id step))})))
                {:steps [] :values initial-values :environment initial-environment}
                (map-indexed vector bindings))
        input-bindings
        (mapv (fn [program-value]
                (if-let [invocation-value (get environment program-value)]
                  {:program-value program-value :invocation-value invocation-value}
                  (fail! :invocation-program-input
                         "internal program input has no public or materialized producer"
                         {:program-value program-value
                          :available (set (keys environment))})))
              program-inputs)
        values (merge program-values values)]
    (validate!
     (->InvocationPlan id public steps input-bindings (vec program-inputs)
                       (vec program-outputs) values attributes))))

(defn validate-against!
  "Validate that `plan` still matches a program's explicit value/input/output boundary."
  [plan parallel-program]
  (let [plan (validate! plan)]
    (when-not (and (= (:program-inputs plan) (:inputs parallel-program))
                   (= (:program-outputs plan) (:outputs parallel-program))
                   (every? (fn [id] (= (get (:values plan) id)
                                       (get (:values parallel-program) id)))
                           (concat (:inputs parallel-program) (:outputs parallel-program))))
      (fail! :invocation-program-mismatch
             "invocation plan no longer matches its internal ParallelProgram boundary"
             {:plan-inputs (:program-inputs plan) :program-inputs (:inputs parallel-program)
              :plan-outputs (:program-outputs plan) :program-outputs (:outputs parallel-program)}))
    plan))
