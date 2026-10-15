(ns raster.compiler.ir.invocation-plan
  "Typed semantic boundary from public function arguments to an internal ParallelProgram.

   ParallelProgram inputs are compiler SSA values: they may include shape projections, narrowed
   scalars, cloned state, and fresh scratch. `storage-bindings` separately retain physical result
   destinations rather than numerical operands. They are either resolved caller-owned values or
   a closed unresolved external premise that only a later resident binder may discharge.
   InvocationPlan records how ordered public parameters produce both boundaries without asking a
   runtime to inspect the original function body or infer storage roles from names. Physical
   allocation, views, ownership, and transfers remain the job of LinkPlan lowering."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.params-flatten :as params-flatten]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.scalar.effects :as effects]))

(defrecord InvocationParameter [id symbol value])
(defrecord ShapeProjection [id symbol operands source axis value])
(defrecord ScalarCompute [id symbol operands region value])
(defrecord BufferAllocation [id symbol operands initialization value])
(defrecord BufferClone [id symbol operands source value])
(defrecord ValueAlias [id symbol operands source value])
(defrecord InvocationPlan
           [id parameters steps bindings storage-bindings
            program-inputs program-outputs values attributes])

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

(defn external-storage-binding?
  "True only for the closed unresolved-storage premise used by source-free resident binding."
  [binding]
  (and (map? binding)
       (= #{:program-value :external} (set (keys binding)))
       (true? (:external binding))))

(def ^:private source-free-shape-projection-key :source-free-shape-projections)

(declare fail! scalar-value?)

(defn source-free-shape-projections
  "Derive the bounded public-buffer shape projections required by a source-free program.

   A projected dimension is a retained private integral scalar used as the sole dimension of
   exactly one public rank-one buffer. It may be reused by other AbstractValue shapes, but it must
   not occur at the executable value boundary. The returned order follows public input order."
  [parallel-program]
  (let [inputs (vec (:inputs parallel-program))
        input-set (set inputs)
        values (:values parallel-program)
        inner-executable-values
        (into #{}
              (mapcat
               (fn [equation]
                 (let [algorithm (:algorithm equation)]
                   (when (soac/program-form? algorithm)
                     (mapcat
                      (fn [inner]
                        (concat (soac/operation-inputs inner)
                                (nth inner 2)
                                (mapcat util/free-syms (soac/operation-extents inner))))
                      (soac/equations algorithm)))))
              (:equations parallel-program)))
        executable-values
        (into (into (set (:outputs parallel-program)) inner-executable-values)
              (mapcat #(concat (:operands %) (:results %)))
              (:equations parallel-program))
        candidates
        (into []
              (keep (fn [source]
                      (let [source-value (get values source)
                            shape (:shape source-value)
                            dimension (when (= 1 (count shape)) (first shape))]
                        (when (and (= :tensor (:kind source-value))
                                   (symbol? dimension)
                                   (not (contains? input-set dimension)))
                          {:dimension dimension :source source :axis 0}))))
              inputs)
        by-dimension (group-by :dimension candidates)]
    (doseq [[dimension projections] by-dimension]
      (when-not (= 1 (count projections))
        (fail! :invocation-source-free-shape-projection
               "one private shape dimension requires exactly one public buffer source"
               {:dimension dimension :sources (mapv :source projections)}))
      (let [value (get values dimension)
            declared (:dtype value)
            canonical (when (and (keyword? declared) (dtype/known? declared))
                        (dtype/canon declared))]
        (when-not (and (scalar-value? value) (contains? #{:int :long} canonical))
          (fail! :invocation-source-free-shape-projection
                 "a projected shape dimension requires one retained integral scalar"
                 {:dimension dimension :value value})))
      (when (contains? executable-values dimension)
        (fail! :invocation-source-free-shape-projection
               "a projected shape dimension cannot also be an executable value"
               {:dimension dimension :executable-values executable-values})))
    candidates))

(defn- validate-source-free-shape-projection-marker!
  [plan]
  (let [attributes (:attributes plan)
        present? (contains? attributes source-free-shape-projection-key)
        projections (get attributes source-free-shape-projection-key)]
    (when present?
      (when-not (and (vector? projections) (seq projections)
                     (every? #(and (= #{:dimension :source :axis} (set (keys %)))
                                    (symbol? (:dimension %)) (symbol? (:source %))
                                    (= 0 (:axis %)))
                             projections)
                     (= (count projections) (count (distinct (map :dimension projections))))
                     (= (count projections) (count (distinct (map :source projections)))))
        (fail! :invocation-source-free-shape-projection
               "source-free shape projections require a closed ordered marker"
               {:projections projections}))
      (let [parameters (into {} (map (juxt :symbol identity)) (:parameters plan))
            steps (into {} (map (juxt :symbol identity)) (:steps plan))]
        (when-not (= (count projections) (count (:steps plan)))
          (fail! :invocation-source-free-shape-projection
                 "source-free shape projection steps must exactly match their marker"
                 {:projections projections :steps (mapv :id (:steps plan))}))
        (doseq [{:keys [dimension source axis]} projections
                :let [parameter (get parameters source)
                      step (get steps dimension)
                      dimension-value (get-in plan [:values dimension])]]
          (when-not (and parameter (shape-projection? step)
                         (= [(list 'extent source)] (get-in parameter [:value :shape]))
                         (= axis (:axis step))
                         (= (:id parameter) (:source step))
                         (= [{:symbol source :value (:id parameter)}] (:operands step))
                         (= dimension-value (:value step))
                         (scalar-value? dimension-value)
                         (keyword? (:dtype dimension-value))
                         (dtype/known? (:dtype dimension-value))
                         (contains? #{:int :long}
                                    (dtype/canon (:dtype dimension-value))))
            (fail! :invocation-source-free-shape-projection
                   "source-free shape marker differs from its parameter and ShapeProjection"
                   {:projection {:dimension dimension :source source :axis axis}
                    :parameter parameter :step step :dimension-value dimension-value}))))))
  plan)

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

(defn validate-parameter-projection!
  "Check the logical caller boundary against the invocation's ordered physical parameters."
  [plan]
  (when-let [{:keys [public-parameters physical-parameters trees] :as projection}
             (get-in plan [:attributes :parameter-projection])]
    (when-not (and (= (cond-> #{:public-parameters :physical-parameters} trees (conj :trees))
                      (set (keys projection)))
                   (or (nil? trees) (and (map? trees) (seq trees)
                                        (every? (set public-parameters) (keys trees))))
                   (distinct-vector? public-parameters) (every? symbol? public-parameters)
                   (vector? physical-parameters)
                   (= (mapv :symbol (:parameters plan)) (mapv :symbol physical-parameters))
                   (= public-parameters (vec (distinct (map :binding physical-parameters)))))
      (fail! :invocation-parameter-projection "aggregate projection differs from parameter order"
             {:projection projection :parameters (:parameters plan)}))
    (doseq [[leaf parameter] (map vector physical-parameters (:parameters plan))]
      (let [{:keys [symbol binding field path tag] class-name :class} leaf
            projected? (or field path)
            array-tag? (types/array-tag? tag)
            retained-dtype (dtype/dtype-for-scalar-tag
                            (if array-tag? (types/array-tag->cast tag) tag))]
        (when-not (and (symbol? symbol) (symbol? binding)
                       (= (set (keys leaf))
                          (cond field #{:symbol :binding :field :class :tag}
                                path #{:symbol :binding :path :tag}
                                :else #{:symbol :binding}))
                       (cond field (and (keyword? field) (string? class-name) (seq class-name) retained-dtype)
                             path (and (vector? path) (seq path) (contains? trees binding) retained-dtype)
                             :else (= symbol binding))
                       (or (not projected?)
                           (and retained-dtype
                                (= (dtype/canon retained-dtype) (dtype/canon (get-in parameter [:value :dtype])))
                                (= array-tag? (boolean (seq (get-in parameter [:value :shape])))))))
          (fail! :invocation-parameter-projection "aggregate leaf differs from its retained storage contract"
                 {:leaf leaf :parameter parameter}))))
    (doseq [[binding leaves] (group-by :binding physical-parameters)]
      (when-not (or (and (not (contains? trees binding))
                         (= 1 (count leaves)) (nil? (:field (first leaves)))
                         (nil? (:path (first leaves))))
                    (and (contains? trees binding)
                         (do (params-flatten/validate-tree-spec! (get trees binding)) true)
                         (= (mapv :path (params-flatten/flatten-spec (get trees binding)))
                            (mapv :path leaves))
                         (every? (fn [[declared leaf]]
                                   (let [declared-tag (types/annotation->tag (:type declared) nil)
                                         tag (:tag leaf)
                                         scalar-tag #(if (types/array-tag? %)
                                                       (types/array-tag->cast %) %)
                                         declared-dtype (dtype/dtype-for-scalar-tag (scalar-tag declared-tag))]
                                     (and declared-dtype
                                          (= declared-dtype (dtype/dtype-for-scalar-tag (scalar-tag tag)))
                                          (= (types/array-tag? declared-tag) (types/array-tag? tag)))))
                                 (map vector (params-flatten/flatten-spec (get trees binding)) leaves)))
                    (and (not (contains? trees binding)) (every? :field leaves)
                         (= 1 (count (distinct (map :class leaves))))
                         (= (count leaves) (count (distinct (map :field leaves))))))
        (fail! :invocation-parameter-projection "aggregate leaves require unique fields and one declared class"
               {:binding binding :leaves leaves}))))
  plan)

(defn validate!
  [plan]
  (when-not (invocation-plan? plan)
    (fail! :invocation-plan-type "expected an InvocationPlan" {:actual (type plan)}))
  (let [{:keys [id parameters steps bindings storage-bindings
                program-inputs program-outputs values attributes]} plan]
    (when (nil? id)
      (fail! :invocation-plan-id "invocation plan requires a stable identity" {}))
    (when-not (and (vector? parameters) (every? invocation-parameter? parameters)
                   (distinct-vector? (mapv :id parameters))
                   (distinct-vector? (mapv :symbol parameters)))
      (fail! :invocation-parameters
             "public invocation parameters must have unique ordered IDs and symbols"
             {:parameters parameters}))
    (when-not (and (vector? steps) (vector? bindings) (vector? storage-bindings)
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
    (validate-parameter-projection! plan)
    (validate-source-free-shape-projection-marker! plan)
    (let [available (binding [util/*shadowing-locals*
                             (into util/*shadowing-locals*
                                   (map :symbol (concat parameters steps)))]
                      (reduce (fn [available step]
                              (validate-step! step values available))
                              (set (map :id parameters)) steps))
          expected-bindings (mapv :program-value bindings)
          storage-values (mapv :program-value storage-bindings)
          external-storage (filterv external-storage-binding? storage-bindings)
          external-marker? (contains? attributes :external-storage)
          external-mode? (= :resident-premise (:external-storage attributes))
          ordinary-storage (remove external-storage-binding? storage-bindings)
          invocation-values (mapv :invocation-value (concat bindings ordinary-storage))]
      (when-not (every? #(or (external-storage-binding? %)
                             (and (map? %) (= #{:program-value :invocation-value}
                                             (set (keys %)))))
                        storage-bindings)
        (fail! :invocation-storage-binding-form
               "storage bindings must be resolved invocation values or closed external premises"
               {:bindings storage-bindings}))
      (when-not (= external-mode?
                   (boolean (and (seq external-storage)
                                 (= (count external-storage) (count storage-bindings)))))
        (fail! :invocation-external-storage-mode
               "unresolved external storage and its closed invocation mode must coincide"
               {:mode (:external-storage attributes)
                :external (mapv :program-value external-storage)
                :storage-count (count storage-bindings)}))
      (when (and external-marker? (not external-mode?))
        (fail! :invocation-external-storage-mode
               "the external-storage attribute is reserved for the exact resident premise"
               {:mode (:external-storage attributes)}))
      (when (and external-mode?
                 (seq (set/intersection (set storage-values) (set program-outputs))))
        (fail! :invocation-external-storage-boundary
               "an unresolved physical destination must be distinct from logical program outputs"
               {:storage storage-values :outputs program-outputs}))
      (when-not (= program-inputs expected-bindings)
        (fail! :invocation-program-input-order
               "invocation bindings must exactly follow the internal program input order"
               {:inputs program-inputs :bindings expected-bindings}))
      (when-not (and (distinct-vector? storage-values)
                     (empty? (set/intersection (set program-inputs) (set storage-values))))
        (fail! :invocation-storage-bindings
               "write-only storage bindings must be distinct from logical program inputs"
               {:inputs program-inputs :storage storage-values}))
      (when-let [missing (seq (set/difference (set invocation-values) available))]
        (fail! :invocation-program-input-value
               "internal program input is not produced by the invocation prefix"
               {:values (set missing)}))
      (doseq [{program-value :program-value invocation-value :invocation-value}
              (concat bindings ordinary-storage)]
        (when-not (compatible-value? (get values program-value) (get values invocation-value))
          (fail! :invocation-program-input-contract
                 "public materialization differs from the internal program value contract"
                 {:program-value program-value :invocation-value invocation-value
                  :program-contract (get values program-value)
                  :invocation-contract (get values invocation-value)})))
      (doseq [{:keys [program-value]} storage-bindings]
        (when-not (buffer-value? (get values program-value))
          (fail! :invocation-storage-value
                 "write-only storage binding must name a tensor buffer"
                 {:program-value program-value :value (get values program-value)})))
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
  (let [symbols (binding [util/*shadowing-locals*
                          (into util/*shadowing-locals* (keys environment))]
                  (sort-by str (util/free-syms expression)))]
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
             (not= :wrap (descriptor/cast-integral-narrowing (descriptor/semantic-op expression)))
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
   lexical shadowing; `binding-values` describes prefix results. `program-values`, logical inputs,
   resolved or unresolved external storage, and outputs are the validated internal program boundary. Stable SSA IDs keep a
   narrowed binding such as `nsteps = (int nsteps)` distinct from its public long parameter."
  [{:keys [id parameters parameter-values bindings binding-values program-values
           program-inputs program-storage external-program-storage program-outputs attributes]
    :or {program-storage [] external-program-storage [] attributes {}}}]
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
        binding-for
        (fn [kind program-value]
          (if-let [invocation-value (get environment program-value)]
            {:program-value program-value :invocation-value invocation-value}
            (fail! kind
                   "internal program value has no public or materialized producer"
                   {:program-value program-value
                    :available (set (keys environment))})))
        input-bindings (mapv #(binding-for :invocation-program-input %) program-inputs)
        storage-bindings (into (mapv #(binding-for :invocation-program-storage %) program-storage)
                               (map (fn [program-value]
                                      {:program-value program-value :external true}))
                               external-program-storage)
        values (merge program-values values)]
    (validate!
     (->InvocationPlan id public steps input-bindings storage-bindings (vec program-inputs)
                       (vec program-outputs) values attributes))))

(defn- program-external-storage
  [parallel-program]
  (into []
        (comp (map :algorithm)
              (filter soac/program-form?)
              (mapcat (fn [algorithm]
                        (let [algorithm (soac/validate! algorithm)
                              facts (soac/facts algorithm)]
                          (mapcat
                           (fn [equation]
                             (let [equation-id (second equation)
                                   results (nth equation 2)
                                   storage (soac/result-storage facts equation-id)
                                   destinations (mapv :destination storage)
                                   aliases (get-in facts [:equations equation-id :aliases])
                                   physical (soac/physical-results facts equation)]
                               (when (seq storage)
                                 (when-not (and (= (count results) (count storage))
                                                (empty? (set/intersection (set results)
                                                                          (set destinations)))
                                                (= aliases (zipmap results destinations))
                                                (= physical destinations))
                                   (fail! :invocation-external-storage-correspondence
                                          "external storage must retain exact result, alias and physical correspondence"
                                          {:equation equation-id :results results
                                           :storage storage :aliases aliases
                                           :physical-results physical})))
                               destinations))
                           (soac/equations algorithm))))))
        (:equations parallel-program)))

(def ^:private source-free-program-origins
  #{[:explicit-typed-algorithm :source-free-typed-invocation]
    [:explicit-typed-effect-algorithm :source-free-typed-effect-invocation]})

(defn- source-free-program?
  "Recognize the source-free boundary from the validated program, never from plan attributes."
  [program]
  (let [origin [(:front-end (:provenance program)) (:pass (:provenance program))]
        source-free-candidate? (and (= :typed-parallel (:dialect program))
                                    (nil? (:source program))
                                    (every? #(soac/program-form? (:algorithm %))
                                            (:equations program)))
        source-free? (and source-free-candidate?
                          (= :typed-invocation (get-in program [:attributes :host-control]))
                          (contains? source-free-program-origins origin))]
    (parallel-program/validate!
     program (constantly true)
     (if source-free-candidate?
       (fn [equation algorithm]
         (and (= :typed-soac (get-in equation [:attributes :algorithm-dialect]))
              (soac/program-form? algorithm)
              (= algorithm (soac/validate! algorithm))
              (= (:operands equation) (:inputs (soac/facts algorithm)))
              (= (:results equation) (soac/outputs algorithm))))
       (fn [_ _] true)))
    (when (and source-free-candidate? (not source-free?))
      (fail! :invocation-program-mismatch
             "source-free typed programs require their exact validated origin"
             {:origin origin :host-control (get-in program [:attributes :host-control])}))
    source-free?))

(defn validate-against!
  "Validate that `plan` still matches a program's explicit value/input/output boundary."
  [plan parallel-program]
  (let [plan (validate! plan)
        source-free? (source-free-program? parallel-program)
        expected-projections (when source-free?
                               (source-free-shape-projections parallel-program))
        declared-projections (get-in plan [:attributes source-free-shape-projection-key])
        external (mapv :program-value
                       (filter external-storage-binding? (:storage-bindings plan)))]
    (when-not (and (or (not source-free?)
                       (= {:source-dialect :typed-soac
                           :algorithm-dialect :typed-soac
                           :target-dialect :typed-invocation}
                          (select-keys (:attributes plan)
                                       [:source-dialect :algorithm-dialect :target-dialect])))
                   (= (vec expected-projections) (vec (or declared-projections [])))
                   (every?
                    (fn [{:keys [dimension source]}]
                      (let [parameter (some #(when (= source (:symbol %)) %) (:parameters plan))
                            source-bindings (filterv #(= source (:program-value %))
                                                     (:bindings plan))
                            original (get-in parallel-program [:values source])
                            dimension-value (get-in parallel-program [:values dimension])]
                        (and (= 1 (count source-bindings))
                             (= (:id parameter) (:invocation-value (first source-bindings)))
                             (= (:value parameter)
                                (assoc original :shape [(list 'extent source)]))
                             (= dimension-value (get-in plan [:values dimension]))
                             (= dimension-value
                                (:value (some #(when (= dimension (:symbol %)) %)
                                              (:steps plan)))))))
                    expected-projections)
                   (= (:program-inputs plan) (:inputs parallel-program))
                   (= (:program-outputs plan) (:outputs parallel-program))
                   (or (empty? external)
                       (= external (program-external-storage parallel-program)))
                   (every? (fn [id] (= (get (:values plan) id)
                                       (get (:values parallel-program) id)))
                           (concat (:inputs parallel-program)
                                   (:outputs parallel-program)
                                   (map :program-value (:storage-bindings plan)))))
      (fail! :invocation-program-mismatch
             "invocation plan no longer matches its internal ParallelProgram boundary"
             {:plan-inputs (:program-inputs plan) :program-inputs (:inputs parallel-program)
              :plan-outputs (:program-outputs plan) :program-outputs (:outputs parallel-program)
              :shape-projections declared-projections
              :expected-shape-projections expected-projections}))
    plan))
