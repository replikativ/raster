(ns raster.compiler.ir.invocation-materialization
  "Pure specialization of a typed InvocationPlan against ordered public arguments.

   The result contains typed scalars and logical buffer initializers for the exact internal
   ParallelProgram inputs plus explicitly resolved caller-owned result storage. Unresolved resident
   storage refuses before argument projection. It allocates no driver
   objects. Pure scalar regions are delegated as closed TypedSOAC regions to an injected compiler
   or interpreter. Observable source completions require a separate owner-supplied executor;
   opaque host references never cross the numerical program boundary. A later lowering realizes
   MaterializedBuffers as LinkValues/BufferViews."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.params-flatten :as params-flatten]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.index-expression :as index-expression]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scalar-range :as scalar-range]
            [raster.compiler.ir.invocation-plan :as invocation]))

(defrecord MaterializedBuffer [id value shape source initialization])
(defrecord MaterializedOpaque [id value source])
(defrecord MaterializedInvocation
           [plan values program-buffers program-scalars attributes])
(defrecord InvocationMaterializationState [plan values lexical next-step])

(defn materialized-buffer? [value]
  (and value (= "raster.compiler.ir.invocation_materialization.MaterializedBuffer"
                (.getName (class value)))))

(defn materialized-opaque? [value]
  (and value (= "raster.compiler.ir.invocation_materialization.MaterializedOpaque"
                (.getName (class value)))))

(defn materialized-invocation? [value]
  (and value (= "raster.compiler.ir.invocation_materialization.MaterializedInvocation"
                (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :invocation-materialization))))

(defn- refuse-external-storage!
  [plan]
  (when-let [storage (seq (filter invocation/external-storage-binding?
                                  (:storage-bindings plan)))]
    (fail! :invocation-materialization-external-storage
           "unresolved resident storage cannot be materialized from host arguments"
           {:storage (mapv :program-value storage)})))

(defn- refuse-external-dimensions!
  [plan]
  (when-let [dimensions (seq (get-in plan [:attributes :external-dimension-values]))]
    (fail! :invocation-materialization-external-dimensions
           "unresolved resident dimensions cannot be materialized from host arguments"
           {:dimensions (mapv :id dimensions)})))

(defn- typed-scalar? [value]
  (and (map? value) (keyword? (:type value)) (contains? value :value)))

(defn- scalar-value? [value]
  (and (= :tensor (:kind value)) (empty? (:shape value))))

(defn- buffer-value? [value]
  (and (= :tensor (:kind value)) (seq (:shape value))))

(defn- opaque-value? [value]
  (and (= :opaque (:kind value)) (empty? (:shape value))))

(defn- checked-opaque [id expected actual]
  ;; Neither equality, hashing nor printing of the payload is a validation
  ;; operation. Its host reference (including nil) remains entirely opaque.
  (when-not (and (opaque-value? expected) (materialized-opaque? actual)
                 (some? (:id actual)) (= expected (:value actual)))
    (fail! :invocation-materialization-opaque
           "host reference requires its retained opaque value wrapper" {:value-id id}))
  actual)

(defn- checked-opaque-step [{:keys [id value] :as step} actual]
  (checked-opaque id value actual)
  (when (and (invocation/program-result-import? step) (not= id (:id actual)))
    (fail! :invocation-materialization-import-identity
           "imported source value belongs to a different invocation step"
           {:step id :actual (:id actual)}))
  actual)

(defn- primitive-array-dtype [value]
  (when (and value (.isArray (class value)))
    (case (.getName (.getComponentType (class value)))
      "byte" :byte
      "short" :half
      "int" :int
      "long" :long
      "float" :float
      "double" :double
      nil)))

(defn- primitive-array-length [value]
  (when (primitive-array-dtype value)
    (long (java.lang.reflect.Array/getLength value))))

(defn ^:no-doc conflicting-source-alias
  "Return the first pair sharing a borrowed host source across different storage identities
   when either entry is mutable. Entries carry :source, :storage and :mutable? plus caller data.
   Read-only sharing and already-represented same-storage aliases are not conflicts. This is
   host object identity only, not an overlap/lifetime proof for resident views or native ranges."
  [entries]
  (let [seen (java.util.IdentityHashMap.)]
    (some (fn [{:keys [source storage mutable?] :as entry}]
            (when (some? source)
              (let [previous (or (.get seen source) [])
                    conflict (some #(when (and (not= storage (:storage %))
                                                (or mutable? (:mutable? %))) %)
                                   previous)]
                (if conflict
                  [conflict entry]
                  (do (.put seen source (conj previous entry)) nil))))) entries)))

(defn- checked-scalar [id abstract value]
  (let [expected (dtype/canon (:dtype abstract))
        value (if (typed-scalar? value) value {:type expected :value value})]
    (when-not (and (scalar-value? abstract)
                   (= expected (dtype/canon (:type value)))
                   (number? (:value value))
                   ;; A retained integral ABI is not a request to truncate or
                   ;; wrap an arbitrary Number. Validate the actual admitted
                   ;; value before shape resolution or numerical evaluation.
                   ;; Floating NaNs/infinities keep their existing IEEE contract.
                   (or (nil? (scalar-range/for-dtype expected))
                       (and (integer? (:value value))
                            (some? (scalar-range/literal (:value value) expected)))))
      (fail! :invocation-materialization-scalar
             "public or computed scalar differs from its retained rank-zero contract"
             {:value-id id :expected abstract :actual value}))
    value))

(defn ^:no-doc checked-scalar-value
  "Validate one explicit scalar specialization against its retained AbstractValue.

   This is the scalar half of invocation materialization only; it grants no program-input,
   storage, or execution authority."
  [id abstract value]
  (av/validate! abstract)
  (checked-scalar id abstract value))

(declare scalar-number)

(defn- resolve-dimension [dimension lexical values]
  (launch/resolve-expression
   #(scalar-number values (get lexical %))
   (index-expression/project-dimension
    dimension #(get-in values [(get lexical %) :type]) fail!)))

(defn- runtime-buffer-shape [parameter abstract source lexical values]
  (let [id (:id parameter)
        symbol (:symbol parameter)
        rank (count (:shape abstract))
        elements (primitive-array-length source)]
    (when-not elements
      (fail! :invocation-materialization-buffer
             "the initial invocation vertical requires primitive-array buffer arguments"
             {:value-id id :actual (some-> source type)}))
    (when-not (= 1 rank)
      (fail! :invocation-materialization-buffer-shape
             "a flat primitive-array argument requires a retained rank-one logical shape"
             {:value-id id :shape (:shape abstract) :elements elements}))
    (let [actual-dtype (primitive-array-dtype source)
          expected-dtype (dtype/canon (:dtype abstract))]
      (when-not (= expected-dtype actual-dtype)
        (fail! :invocation-materialization-buffer-dtype
               "public buffer storage dtype differs from its AbstractValue"
               {:value-id id :expected expected-dtype :actual actual-dtype})))
    (let [dimension (first (:shape abstract))
          expected
          (cond
            (integer? dimension) (long dimension)
            (symbol? dimension)
            (scalar-number values (get lexical dimension))
            (and (seq? dimension) (= 'extent (first dimension))
                 (= 2 (count dimension)) (= symbol (second dimension)))
            elements
            (seq? dimension) (resolve-dimension dimension lexical values)
            :else
            (fail! :invocation-materialization-buffer-shape
                   "public buffer shape must resolve from a static extent, scalar, or self extent"
                   {:value-id id :symbol symbol :shape (:shape abstract)
                    :available (set (keys lexical))}))]
      (when-not (= expected elements)
        (fail! :invocation-materialization-buffer-shape
               "public buffer storage length differs from its retained logical shape"
               {:value-id id :symbol symbol :expected [expected] :actual [elements]})))
    [elements]))

(defn- scalar-number [values id]
  (let [value (get values id)]
    (when-not (typed-scalar? value)
      (fail! :invocation-materialization-shape-value
             "symbolic buffer shape requires an earlier typed scalar"
             {:value-id id :actual value}))
    (let [number (:value value)]
      (when-not (and (contains? #{:int :long} (dtype/canon (:type value)))
                     (integer? number))
        (fail! :invocation-materialization-shape-value
               "buffer dimensions require integral scalar values"
               {:value-id id :actual value}))
      (long number))))

(defn- concrete-shape [id abstract lexical values]
  (mapv
   (fn [dimension]
     (let [resolved
           (cond
             (integer? dimension) (long dimension)
             (symbol? dimension)
             (if-let [value-id (get lexical dimension)]
               (scalar-number values value-id)
               (fail! :invocation-materialization-shape-symbol
                      "buffer shape references a scalar outside the invocation boundary"
                      {:value-id id :dimension dimension :available (set (keys lexical))}))
             (and (seq? dimension) (= 'extent (first dimension)) (= 2 (count dimension)))
             (let [source-symbol (second dimension)
                   source-id (get lexical source-symbol)
                   source (get values source-id)]
               (if (materialized-buffer? source)
                 (first (:shape source))
                 (fail! :invocation-materialization-shape-expression
                        "extent projection requires an earlier materialized buffer"
                        {:value-id id :dimension dimension :source source-symbol
                         :actual source})))
             (seq? dimension) (resolve-dimension dimension lexical values)
             :else
             (fail! :invocation-materialization-shape-expression
                    "buffer shape must be canonicalized to integer or scalar SSA dimensions"
                    {:value-id id :dimension dimension}))]
       (when (neg? resolved)
         (fail! :invocation-materialization-shape-negative
                "buffer dimensions must resolve non-negative"
                {:value-id id :dimension dimension :resolved resolved}))
       resolved))
   (:shape abstract)))

(defn- checked-computed-scalar [step value]
  (when (invocation/source-completion? step)
    (let [contract (:contract (completion/canonical-parts (:expression step)))
          carrier (get types/primitive-info (completion/source-cast-result-tag contract))]
      ;; A source completion already performed its actual conversion. A
      ;; dtype label is not permission to silently convert its result again.
      (when-not (and (typed-scalar? value) carrier
                     (identical? carrier (class (:value value))))
        (fail! :invocation-materialization-source-carrier
               "source completion executor did not return its declared physical result"
               {:step (:id step) :expected carrier :actual (some-> (:value value) class)}))))
  (checked-scalar (:id step) (:value step) value))

(defn- source-completion-producers
  "Follow only explicit invocation value aliases to their source completion.
   This is a plan-local dataflow projection, not an inferred numeric type."
  [plan]
  (reduce (fn [producers step]
            (cond
              (invocation/source-completion? step) (assoc producers (:id step) step)
              (invocation/value-alias? step)
              (if-let [producer (get producers (:source step))]
                (assoc producers (:id step) producer) producers)
              :else producers))
          {} (:steps plan)))

(defn- validate-buffer! [binding-id buffer expected]
  (when-not (materialized-buffer? buffer)
    (fail! :invocation-materialization-buffer-binding
           "program buffer input requires a MaterializedBuffer"
           {:value-id binding-id :actual buffer}))
  (let [{:keys [id value shape source initialization]} buffer]
    (when (nil? id)
      (fail! :invocation-materialization-buffer-id
             "materialized buffer requires a stable storage identity"
             {:value-id binding-id}))
    (when-not (av/storage-contract-compatible? expected value)
      (fail! :invocation-materialization-buffer-value
             "materialized buffer differs from its retained value contract"
             {:value-id binding-id :expected expected :actual value}))
    (when-not (and (buffer-value? value) (vector? shape)
                   (= (count (:shape value)) (count shape))
                   (every? #(and (integer? %) (not (neg? %))) shape))
      (fail! :invocation-materialization-buffer-shape
             "materialized buffer requires one concrete dimension per logical axis"
             {:value-id binding-id :value value :shape shape}))
    (when-not (contains? #{:copy :zero :unspecified} initialization)
      (fail! :invocation-materialization-buffer-initialization
             "materialized buffer has an unknown initialization contract"
             {:value-id binding-id :initialization initialization}))
    (when (not= (= :copy initialization) (some? source))
      (fail! :invocation-materialization-buffer-source
             "exactly copy-initialized buffers require host source storage"
             {:value-id binding-id :initialization initialization :source source}))
    (when source
      (let [actual-dtype (primitive-array-dtype source)
            actual-elements (primitive-array-length source)
            expected-elements (reduce * 1 shape)]
        (when-not (and (= (dtype/canon (:dtype value)) actual-dtype)
                       (= expected-elements actual-elements))
          (fail! :invocation-materialization-buffer-source
                 "buffer initializer differs from its concrete storage contract"
                 {:value-id binding-id :expected-dtype (:dtype value)
                  :actual-dtype actual-dtype :expected-elements expected-elements
                  :actual-elements actual-elements})))))
  buffer)

(defn validate!
  [materialization]
  (when-not (materialized-invocation? materialization)
    (fail! :invocation-materialization-type "expected a MaterializedInvocation"
           {:actual (type materialization)}))
  (let [{:keys [plan values program-buffers program-scalars attributes]} materialization
        plan (invocation/validate! plan)
        _ (refuse-external-dimensions! plan)
        _ (refuse-external-storage! plan)
        completion-producers (source-completion-producers plan)
        invocation-ids (set (concat (map :id (:parameters plan)) (map :id (:steps plan))))
        lexical (into {} (map (juxt :symbol :id))
                      (concat (:parameters plan) (:steps plan)))]
    (doseq [[field value] [[:values values] [:program-buffers program-buffers]
                           [:program-scalars program-scalars] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :invocation-materialization-field
               "materialized invocation fields must be maps" {:field field :value value})))
    (when-not (= invocation-ids (set (keys values)))
      (fail! :invocation-materialization-values
             "materialization must retain every public parameter and invocation SSA result"
             {:expected invocation-ids :actual (set (keys values))}))
    ;; Retained parameters/steps can feed shape computations without being
    ;; final program inputs. Their declared scalar domains still apply when a
    ;; materialization is validated again at a later boundary.
    (doseq [{:keys [id value] :as slot} (concat (:parameters plan) (:steps plan))
            :when (scalar-value? value)]
      (checked-computed-scalar (get completion-producers id slot) (get values id)))
    (doseq [{:keys [id value] :as slot} (concat (:parameters plan) (:steps plan))
            :when (opaque-value? value)]
      (checked-opaque-step slot (get values id)))
    (let [storage-values (set (map :program-value (:storage-bindings plan)))
          materialized-values (set/union (set (:program-inputs plan)) storage-values)]
      (when-not (= materialized-values
                   (set (concat (keys program-buffers) (keys program-scalars))))
        (fail! :invocation-materialization-inputs
               "materialized bindings must partition logical inputs and external result storage"
               {:expected materialized-values
                :buffers (keys program-buffers) :scalars (keys program-scalars)}))
      (when-let [scalar-storage (seq (set/intersection storage-values
                                                       (set (keys program-scalars))))]
        (fail! :invocation-materialization-storage-kind
               "external result storage cannot materialize as scalar values"
               {:values (set scalar-storage)})))
    (when (seq (set/intersection (set (keys program-buffers))
                                 (set (keys program-scalars))))
      (fail! :invocation-materialization-input-kind
             "one program input cannot be both a buffer and scalar" {}))
    (doseq [[id buffer] program-buffers]
      (validate-buffer! id buffer (get-in plan [:values id]))
      (let [expected (concrete-shape id (get-in plan [:values id]) lexical values)]
        (when-not (= expected (:shape buffer))
          (fail! :invocation-materialization-program-shape
                 "materialized producer shape differs from its internal program input"
                 {:program-value id :expected expected :actual (:shape buffer)}))))
    (doseq [[id scalar] program-scalars]
      (checked-scalar id (get-in plan [:values id]) scalar))
    (doseq [{:keys [program-value invocation-value]}
            (concat (:bindings plan) (:storage-bindings plan))]
      (when-let [producer (get completion-producers invocation-value)]
        (checked-computed-scalar producer (get program-scalars program-value)))
      (when-not (= (get values invocation-value)
                   (or (get program-buffers program-value)
                       (get program-scalars program-value)))
        (fail! :invocation-materialization-binding
               "program input differs from its certified invocation producer"
               {:program-value program-value :invocation-value invocation-value})))
    materialization))

(defn aggregate-field
  "Read one declared field after checking the logical argument's nominal class."
  [argument binding class-name field]
  (when-not (= class-name (some-> argument class .getName))
    (fail! :invocation-materialization-aggregate-class
           "aggregate argument differs from its declared class"
           {:parameter binding :expected class-name :actual (some-> argument class .getName)}))
  (try
    (let [f (.getField (class argument) (clojure.lang.Compiler/munge (name field)))]
      (.get ^java.lang.reflect.Field f argument))
    (catch java.lang.ReflectiveOperationException error
      (fail! :invocation-materialization-aggregate-field
             "aggregate argument has no declared physical field"
             {:parameter binding :field field :class class-name}))))

(defn validate-aggregate!
  "Validate a declared tree once, before selecting leaves or mutating resident storage."
  [argument binding tree-spec]
  (when tree-spec
    (params-flatten/assert-tree-shape! tree-spec argument)
    (params-flatten/assert-no-identity-collisions! tree-spec argument))
  argument)

(defn aggregate-leaf
  "Read a declared projection selector, without inferring types from runtime contents."
  [argument {:keys [binding field path] class-name :class}]
  (cond field (aggregate-field argument binding class-name field)
        path (get-in argument path)
        :else argument))

(defn projection-path [leaf]
  (or (:path leaf) (when-let [field (:field leaf)] [field])))

(defn- project-parameter-arguments
  [plan arguments]
  (let [projection (get-in plan [:attributes :parameter-projection])
        public (or (:public-parameters projection) (mapv :symbol (:parameters plan)))
        arguments (vec arguments)]
    (when-not (= (count public) (count arguments))
      (fail! :invocation-materialization-arity "public arguments must follow the declared parameter order"
             {:parameters public :expected (count public) :actual (count arguments)}))
    (if-not projection arguments
      (let [values (zipmap public arguments)]
        (doseq [[binding spec] (:trees projection)]
          (validate-aggregate! (get values binding) binding spec))
        (mapv (fn [{:keys [binding] :as leaf}]
                (aggregate-leaf (get values binding) leaf))
              (:physical-parameters projection))))))

(defn parameter-arguments
  "Project declared aggregate fields at the public invocation boundary.
   Flat callers retain their original argument order. This evaluates no source expressions."
  [plan arguments]
  (project-parameter-arguments (invocation/validate-parameter-projection! plan) arguments))

(defn- checked-parameter-plan [plan]
  (let [plan (invocation/validate! plan)]
    (refuse-external-dimensions! plan)
    (refuse-external-storage! plan)
    plan))

(defn- initialize-parameters
  [plan arguments]
  ;; Check before pairing: mapv vector would otherwise silently truncate a missing leaf.
  (when-not (and (vector? arguments) (= (count (:parameters plan)) (count arguments)))
    (fail! :invocation-materialization-arity
           "physical arguments must follow the declared parameter order"
           {:parameters (mapv :symbol (:parameters plan))
            :expected (count (:parameters plan)) :actual (count arguments)}))
  (let [parameter-pairs (mapv vector (:parameters plan) arguments)
          scalar-values
          (into {}
                (keep (fn [[parameter argument]]
                        (when (scalar-value? (:value parameter))
                          [(:id parameter)
                           (checked-scalar (:id parameter) (:value parameter) argument)])))
                parameter-pairs)
          initial-lexical (into {} (map (juxt :symbol :id)) (:parameters plan))
          parameter-values
          (reduce
           (fn [values [parameter argument]]
             (let [{:keys [id value]} parameter]
               (cond
                 (opaque-value? value)
                 (assoc values id (->MaterializedOpaque id value argument))

                 (buffer-value? value)
                 (assoc values id
                        (->MaterializedBuffer id value
                                              (runtime-buffer-shape parameter value argument
                                                                    initial-lexical scalar-values)
                                              argument :copy))

                 (scalar-value? value) values
                 :else (fail! :invocation-materialization-parameter-kind
                              "public argument has no supported host materialization"
                              {:value-id id :kind (:kind value)}))))
           scalar-values parameter-pairs)]
    (->InvocationMaterializationState plan parameter-values initial-lexical 0)))

(defn ^:no-doc begin-parameter-materialization
  "Initialize declared physical parameters after an owner's logical entry projection.
   This checks arity, scalar values/domains and buffer contracts, not the original root Class/tree or
   source completion. It cannot authorize a new public calling convention or scalar replacement."
  [plan parameter-arguments]
  (initialize-parameters (checked-parameter-plan plan) parameter-arguments))

(defn begin-materialization
  "Validate public arguments without evaluating any invocation step.

   The returned immutable state owns no driver resources or evaluator. It is not a completed
   MaterializedInvocation or permission to bind its unevaluated suffix. Advancing a state chain
   evaluates one step at a time; callers may place existing device completion boundaries between
   advances. Reusing an older state explicitly repeats computation, not a successful-check proof.
   Host arrays remain borrowed sources, not snapshots or newly acquired lifetime leases."
  [plan arguments]
  (let [plan (checked-parameter-plan plan)]
    (initialize-parameters plan (project-parameter-arguments plan arguments))))

(defn- step-operand-values [step values]
  (into {} (map (fn [{:keys [symbol value]}] [symbol (get values value)])) (:operands step)))

(defn- source-completion-executor [evaluators step-id]
  (let [execute (when (map? evaluators) (:source-completion evaluators))]
    (when-not (ifn? execute)
      (fail! :invocation-materialization-source-executor
             "source completion requires a separately supplied owned executor"
             {:step step-id}))
    execute))

(defn- advance-materialization
  [{:keys [plan values lexical next-step] :as state} evaluate-scalar]
  (let [step (nth (:steps plan) next-step)
        id (:id step)
        materialized
        (cond
          (invocation/shape-projection? step)
          (let [source (get values (:source step))]
            (when-not (materialized-buffer? source)
              (fail! :invocation-materialization-shape-source
                     "shape projection source is not a materialized buffer"
                     {:step id :source (:source step) :actual source}))
            (let [result (checked-computed-scalar
                          step {:type (get-in step [:value :dtype])
                                :value (nth (:shape source) (:axis step))})]
              ;; Shape dimensions are stored as Longs, but a source alength
              ;; produces an Int unless its retained result explicitly widens.
              ;; Check the domain BEFORE converting; this is this operation's
              ;; result production, not normalization of arbitrary public scalars.
              (update result :value #(case (dtype/canon (:type result))
                                       :int (int %)
                                       :long (long %)))))

          (invocation/source-completion? step)
          (let [execute (source-completion-executor evaluate-scalar id)]
            (checked-computed-scalar step (execute step (step-operand-values step values))))

          (invocation/program-result-import? step)
          (fail! :invocation-materialization-program-result
                 "numerical result imports require the owning ordered execution boundary"
                 {:step id :equation (:equation step) :result (:result step)})

          (invocation/scalar-compute? step)
          (let [evaluate (if (map? evaluate-scalar) (:scalar evaluate-scalar) evaluate-scalar)]
            (when-not (ifn? evaluate)
              (fail! :invocation-materialization-scalar-evaluator
                     "scalar invocation regions require a typed evaluator"
                     {:step id :region (:region step)}))
            (checked-computed-scalar
             step (evaluate step (step-operand-values step values))))

          (invocation/buffer-clone? step)
          (let [source (get values (:source step))]
            (when-not (materialized-buffer? source)
              (fail! :invocation-materialization-clone-source
                     "buffer clone source is not materialized storage"
                     {:step id :source (:source step) :actual source}))
            (->MaterializedBuffer id (:value step)
                                  (concrete-shape id (:value step) lexical values)
                                  (:source source) :copy))

          (invocation/buffer-allocation? step)
          (->MaterializedBuffer id (:value step)
                                (concrete-shape id (:value step) lexical values)
                                nil (:initialization step))

          (invocation/value-alias? step)
          (get values (:source step)))]
    (when-not materialized
      (fail! :invocation-materialization-step
             "invocation step did not produce a runtime value" {:step id}))
    (assoc state :values (assoc values id materialized)
                 :lexical (assoc lexical (:symbol step) id)
                 :next-step (inc next-step))))

(defn validate-state!
  "Check an incremental materialization's completed prefix, without evaluating scalar regions.
   As for MaterializedInvocation, computed values come from the caller's trusted evaluator;
   structural validation is not a certificate that arbitrary supplied numbers were computed."
  [state]
  (when-not (and state (= "raster.compiler.ir.invocation_materialization.InvocationMaterializationState"
                         (.getName (class state))))
    (fail! :invocation-materialization-state-type "expected an InvocationMaterializationState"
           {:actual (type state)}))
  (let [{:keys [plan values lexical next-step]} state
        plan (invocation/validate! plan)
        completion-producers (source-completion-producers plan)]
    (refuse-external-dimensions! plan)
    (refuse-external-storage! plan)
    (when-not (and (integer? next-step) (<= 0 next-step (count (:steps plan))))
      (fail! :invocation-materialization-position "materialization position is outside the plan"
             {:next-step next-step :steps (count (:steps plan))}))
    (let [prefix (concat (:parameters plan) (take next-step (:steps plan)))
          expected-ids (set (map :id prefix))
          expected-lexical (into {} (map (juxt :symbol :id)) prefix)]
      (when-not (and (map? values) (= expected-ids (set (keys values)))
                     (= expected-lexical lexical))
        (fail! :invocation-materialization-prefix "materialization state differs from its exact prefix"
               {:next-step next-step :expected-values expected-ids
                :actual-values (when (map? values) (set (keys values)))}))
      (let [parameters (:parameters plan)
            parameter-lexical (into {} (map (juxt :symbol :id)) parameters)
            parameter-values (select-keys values (map :id parameters))
            scalar-values (into {} (filter (comp typed-scalar? val)) parameter-values)
            check-scalar! (fn [{:keys [id] :as slot} actual]
                            (when-not (typed-scalar? actual)
                              (fail! :invocation-materialization-state-scalar
                                     "retained scalar state requires a typed value"
                                     {:value-id id :actual actual}))
                            (checked-computed-scalar (get completion-producers id slot) actual))]
        (doseq [{:keys [id value] :as parameter} parameters]
          (let [actual (get values id)]
            (cond
              (opaque-value? value) (checked-opaque id value actual)
              (scalar-value? value)
              (check-scalar! parameter actual)
              :else (do
                (validate-buffer! id actual value)
                (let [expected (->MaterializedBuffer
                                id value
                                (runtime-buffer-shape parameter value (:source actual)
                                                      parameter-lexical scalar-values)
                                (:source actual) :copy)]
                  (when-not (= expected actual)
                    (fail! :invocation-materialization-state-parameter
                           "retained buffer parameter differs from its original projection"
                           {:value-id id :expected expected :actual actual})))))))
        ;; Reuse the same interpreter to check deterministic shape, allocation,
        ;; clone and alias steps in their original lexical environment. Only
        ;; ScalarCompute and SourceCompletion results are supplied, never
        ;; reevaluated during validation.
        (reduce
         (fn [prior step]
           (let [{:keys [id symbol value]} step
                 actual (get values id)]
             (cond
               (invocation/program-result-import? step)
               (do
                 (checked-opaque-step step actual)
                 (-> prior
                     (assoc-in [:values id] actual)
                     (assoc-in [:lexical symbol] id)
                     (update :next-step inc)))

               (or (invocation/scalar-compute? step) (invocation/source-completion? step))
               (do (check-scalar! step actual)
                   (-> prior
                       (assoc-in [:values id] actual)
                       (assoc-in [:lexical symbol] id)
                       (update :next-step inc)))
               :else
               (let [_ (when-let [producer (get completion-producers id)]
                         (checked-computed-scalar producer actual))
                     expected (advance-materialization prior nil)]
                 (when-not (if (opaque-value? value)
                             (identical? actual (get-in expected [:values id]))
                             (= actual (get-in expected [:values id])))
                   (fail! :invocation-materialization-state-producer
                          "retained structured value differs from its ordered producer"
                          (cond-> {:value-id id}
                            (not (opaque-value? value))
                            (assoc :actual actual :expected (get-in expected [:values id])))))
                 expected))))
         (->InvocationMaterializationState plan parameter-values parameter-lexical 0)
         (take next-step (:steps plan))))
      state)))

(defn ^:no-doc initial-parameter-arguments
  "Read already checked entry slots without revisiting the logical argument containers.
   This borrows their sources; it neither snapshots bytes nor grants source-completion credit."
  [state]
  (when-not (= 0 (:next-step state))
    (fail! :invocation-materialization-position
           "entry slots must be captured before source materialization advances" {}))
  (let [state (validate-state! state)]
    (mapv (fn [{:keys [id]}]
            (let [value (get-in state [:values id])]
              (cond (or (materialized-buffer? value) (materialized-opaque? value)) (:source value)
                    (typed-scalar? value) (:value value)
                    :else value)))
          (get-in state [:plan :parameters]))))

(defn materialize-next
  "Evaluate exactly the next invocation step after validating the current prefix.
   A failure returns no advanced state and does not evaluate or materialize any later step."
  [state evaluate-scalar]
  (let [state (validate-state! state)]
    (when (= (:next-step state) (count (get-in state [:plan :steps])))
      (fail! :invocation-materialization-complete "materialization has no remaining step" {}))
    (advance-materialization state evaluate-scalar)))

(defn supply-program-result
  "Install one opaque source value at the exact next numerical-result import.
   This pure transition performs no transfer or conversion and proves no device
   completion. The ordered owner must first discharge the import's producer,
   payload and initialized-view obligations. Its actual snapshot step—not an
   equal reconstructed record—identifies where the supplied value belongs."
  [state step source]
  (let [{:keys [plan next-step] :as state} (validate-state! state)
        actual (nth (:steps plan) next-step nil)]
    (when-not (and (invocation/program-result-import? actual) (identical? step actual))
      (fail! :invocation-materialization-import-step
             "a numerical result may only supply its exact next source import"
             {:next-step next-step}))
    (validate-state!
     (-> state
         (assoc-in [:values (:id step)] (->MaterializedOpaque (:id step) (:value step) source))
         (assoc-in [:lexical (:symbol step)] (:id step))
         (update :next-step inc)))))

(defn- available-program-bindings
  [{:keys [plan values]}]
  (let [input-values
          (into {}
                (keep (fn [{:keys [program-value invocation-value]}]
                        (when (contains? values invocation-value)
                          [program-value (get values invocation-value)])))
                (concat (:bindings plan) (:storage-bindings plan)))
          program-buffers (into {} (filter (comp materialized-buffer? val)) input-values)
          program-scalars (into {} (filter (comp typed-scalar? val)) input-values)]
    {:program-buffers program-buffers :program-scalars program-scalars}))

(defn prefix-bindings
  "Project only available values from a checked materialization prefix, without completing it.
   Missing suffix bindings are absent, never nil placeholders or permission to bind a graph.
   Concrete storage/call validation must still discharge each ready equation's requirements."
  [state]
  (available-program-bindings (validate-state! state)))

(defn- complete-materialization
  [{:keys [plan values next-step] :as state}]
    (when-not (= next-step (count (:steps plan)))
      (fail! :invocation-materialization-incomplete "cannot publish an unevaluated invocation suffix"
             {:next-step next-step :steps (count (:steps plan))}))
    (let [{:keys [program-buffers program-scalars]} (available-program-bindings state)]
      (validate!
       (->MaterializedInvocation
        plan values program-buffers program-scalars
        {:source-inspected false :driver-allocations 0}))))

(defn finish-materialization
  "Publish a completed invocation only after validating every completed prefix step."
  [state]
  (complete-materialization (validate-state! state)))

(defn materialize
  "Specialize a typed plan against public arguments without allocating driver resources.

   This eager convenience entry uses the same step interpreter as incremental materialization.
   A function evaluator receives pure ScalarCompute steps. An explicit executor map may provide
   :scalar and :source-completion functions separately; each returns one typed scalar. Source
   execution authority belongs to the caller's exact prepared owner, never this semantic plan.
   Plans requiring a source executor refuse before evaluating any step when it is absent.
   Executors are not retained or called for shapes, allocations or aliases."
  [plan arguments evaluate-scalar]
  (let [plan (invocation/validate! plan)
        _ (when-let [step (first (filter invocation/program-result-import? (:steps plan)))]
            (fail! :invocation-materialization-program-result
                   "eager host materialization cannot supply numerical results"
                   {:step (:id step) :equation (:equation step) :result (:result step)}))
        _ (when-let [step (first (filter invocation/source-completion? (:steps plan)))]
            (source-completion-executor evaluate-scalar (:id step)))
        initial (begin-materialization plan arguments)
        final (reduce (fn [state _] (advance-materialization state evaluate-scalar))
                      initial (get-in initial [:plan :steps]))]
    ;; The eager chain is produced here, not supplied by a caller. Keep its
    ;; existing final validation without reinterpreting every structured step.
    (complete-materialization final)))
