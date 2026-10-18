(ns raster.compiler.ir.invocation-materialization
  "Pure specialization of a typed InvocationPlan against ordered public arguments.

   The result contains typed scalars and logical buffer initializers for the exact internal
   ParallelProgram inputs plus explicitly resolved caller-owned result storage. Unresolved resident
   storage refuses before argument projection. It allocates no driver
   objects and does not evaluate Clojure source: scalar regions are delegated as closed TypedSOAC
   regions to an injected compiler or interpreter. A later lowering realizes MaterializedBuffers
   as LinkValues/BufferViews."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.params-flatten :as params-flatten]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.index-expression :as index-expression]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scalar-range :as scalar-range]
            [raster.compiler.ir.invocation-plan :as invocation]))

(defrecord MaterializedBuffer [id value shape source initialization])
(defrecord MaterializedInvocation
           [plan values program-buffers program-scalars attributes])
(defrecord InvocationMaterializationState [plan values lexical next-step])

(defn materialized-buffer? [value]
  (and value (= "raster.compiler.ir.invocation_materialization.MaterializedBuffer"
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
  (checked-scalar (:id step) (:value step) value))

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
    (doseq [{:keys [id value]} (concat (:parameters plan) (:steps plan))
            :when (scalar-value? value)]
      (checked-scalar id value (get values id)))
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

(defn begin-materialization
  "Validate public arguments without evaluating any invocation step.

   The returned immutable state owns no driver resources or evaluator. It is not a completed
   MaterializedInvocation or permission to bind its unevaluated suffix. Advancing a state chain
   evaluates one step at a time; callers may place existing device completion boundaries between
   advances. Reusing an older state explicitly repeats computation, not a successful-check proof.
   Host arrays remain borrowed sources, not snapshots or newly acquired lifetime leases."
  [plan arguments]
  (let [plan (invocation/validate! plan)
        _ (refuse-external-dimensions! plan)
        _ (refuse-external-storage! plan)
        arguments (project-parameter-arguments plan arguments)]
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
               (if (buffer-value? value)
                 (assoc values id
                        (->MaterializedBuffer id value
                                              (runtime-buffer-shape parameter value argument
                                                                    initial-lexical scalar-values)
                                              argument :copy))
                 values)))
           scalar-values parameter-pairs)]
      (->InvocationMaterializationState plan parameter-values initial-lexical 0))))

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
            (checked-computed-scalar
             step {:type (get-in step [:value :dtype])
                   :value (nth (:shape source) (:axis step))}))

          (invocation/scalar-compute? step)
          (do
            (when-not (ifn? evaluate-scalar)
              (fail! :invocation-materialization-scalar-evaluator
                     "scalar invocation regions require a typed evaluator"
                     {:step id :region (:region step)}))
            (checked-computed-scalar
             step (evaluate-scalar
                   step (into {} (map (fn [{:keys [symbol value]}]
                                        [symbol (get values value)])) (:operands step)))))

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
        plan (invocation/validate! plan)]
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
            check-scalar! (fn [id value actual]
                            (when-not (typed-scalar? actual)
                              (fail! :invocation-materialization-state-scalar
                                     "retained scalar state requires a typed value"
                                     {:value-id id :actual actual}))
                            (checked-scalar id value actual))]
        (doseq [{:keys [id value] :as parameter} parameters]
          (let [actual (get values id)]
            (if (scalar-value? value)
              (check-scalar! id value actual)
              (do
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
        ;; ScalarCompute results are supplied, never reevaluated during validation.
        (reduce
         (fn [prior step]
           (let [{:keys [id symbol value]} step
                 actual (get values id)]
             (if (invocation/scalar-compute? step)
               (do (check-scalar! id value actual)
                   (-> prior
                       (assoc-in [:values id] actual)
                       (assoc-in [:lexical symbol] id)
                       (update :next-step inc)))
               (let [expected (advance-materialization prior nil)]
                 (when-not (= actual (get-in expected [:values id]))
                   (fail! :invocation-materialization-state-producer
                          "retained structured value differs from its ordered producer"
                          {:value-id id :actual actual
                           :expected (get-in expected [:values id])}))
                 expected))))
         (->InvocationMaterializationState plan parameter-values parameter-lexical 0)
         (take next-step (:steps plan))))
      state)))

(defn materialize-next
  "Evaluate exactly the next invocation step after validating the current prefix.
   A failure returns no advanced state and does not evaluate or materialize any later step."
  [state evaluate-scalar]
  (let [state (validate-state! state)]
    (when (= (:next-step state) (count (get-in state [:plan :steps])))
      (fail! :invocation-materialization-complete "materialization has no remaining step" {}))
    (advance-materialization state evaluate-scalar)))

(defn- complete-materialization
  [{:keys [plan values next-step]}]
    (when-not (= next-step (count (:steps plan)))
      (fail! :invocation-materialization-incomplete "cannot publish an unevaluated invocation suffix"
             {:next-step next-step :steps (count (:steps plan))}))
    (let [input-values
          (into {}
                (map (fn [{:keys [program-value invocation-value]}]
                       [program-value (get values invocation-value)]))
                (concat (:bindings plan) (:storage-bindings plan)))
          program-buffers (into {} (filter (comp materialized-buffer? val)) input-values)
          program-scalars (into {} (filter (comp typed-scalar? val)) input-values)]
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
   `evaluate-scalar` receives the retained closed ScalarCompute and typed operand map, and must
   return one typed scalar. It is neither retained nor called for shapes, allocations or aliases."
  [plan arguments evaluate-scalar]
  (let [initial (begin-materialization plan arguments)
        final (reduce (fn [state _] (advance-materialization state evaluate-scalar))
                      initial (get-in initial [:plan :steps]))]
    ;; The eager chain is produced here, not supplied by a caller. Keep its
    ;; existing final validation without reinterpreting every structured step.
    (complete-materialization final)))
