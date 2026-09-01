(ns raster.compiler.ir.invocation-materialization
  "Pure specialization of a typed InvocationPlan against ordered public arguments.

   The result contains typed scalars and logical buffer initializers for the exact internal
   ParallelProgram inputs plus explicitly bound caller-owned result storage. It allocates no driver
   objects and does not evaluate Clojure source: scalar regions are delegated as closed TypedSOAC
   regions to an injected compiler or interpreter. A later lowering realizes MaterializedBuffers
   as LinkValues/BufferViews."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]))

(defrecord MaterializedBuffer [id value shape source initialization])
(defrecord MaterializedInvocation
           [plan values program-buffers program-scalars attributes])

(defn materialized-buffer? [value]
  (and value (= "raster.compiler.ir.invocation_materialization.MaterializedBuffer"
                (.getName (class value)))))

(defn materialized-invocation? [value]
  (and value (= "raster.compiler.ir.invocation_materialization.MaterializedInvocation"
                (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :invocation-materialization))))

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
                   (number? (:value value)))
      (fail! :invocation-materialization-scalar
             "public or computed scalar differs from its retained rank-zero contract"
             {:value-id id :expected abstract :actual value}))
    value))

(declare scalar-number)

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
      (when-not (integer? number)
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

(defn- validate-buffer! [binding-id buffer]
  (when-not (materialized-buffer? buffer)
    (fail! :invocation-materialization-buffer-binding
           "program buffer input requires a MaterializedBuffer"
           {:value-id binding-id :actual buffer}))
  (let [{:keys [id value shape source initialization]} buffer]
    (when (nil? id)
      (fail! :invocation-materialization-buffer-id
             "materialized buffer requires a stable storage identity"
             {:value-id binding-id}))
    (av/validate! value)
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
      (validate-buffer! id buffer)
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

(defn materialize
  "Specialize `plan` against ordered public `arguments` without allocating device resources.

   `evaluate-scalar` receives `[step operand-values]`, where `step` owns a closed TypedSOAC scalar
   region and `operand-values` maps its ordered parameter symbols to typed scalar maps. It must
   return one typed scalar map. This is the only executable hook; shapes, copies, allocations and
   aliases are interpreted from structured invocation records."
  [plan arguments evaluate-scalar]
  (let [plan (invocation/validate! plan)
        arguments (vec arguments)]
    (when-not (= (count (:parameters plan)) (count arguments))
      (fail! :invocation-materialization-arity
             "public arguments must follow the InvocationPlan parameter order"
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
               (if (buffer-value? value)
                 (assoc values id
                        (->MaterializedBuffer id value
                                              (runtime-buffer-shape parameter value argument
                                                                    initial-lexical scalar-values)
                                              argument :copy))
                 values)))
           scalar-values parameter-pairs)
          {:keys [values lexical]}
          (reduce
           (fn [{:keys [values lexical]} step]
             (let [id (:id step)
                   materialized
                   (cond
                     (invocation/shape-projection? step)
                     (let [source (get values (:source step))
                           axis (:axis step)]
                       (when-not (materialized-buffer? source)
                         (fail! :invocation-materialization-shape-source
                                "shape projection source is not a materialized buffer"
                                {:step id :source (:source step) :actual source}))
                       (checked-computed-scalar
                        step {:type (get-in step [:value :dtype])
                              :value (nth (:shape source) axis)}))

                     (invocation/scalar-compute? step)
                     (do
                       (when-not (ifn? evaluate-scalar)
                         (fail! :invocation-materialization-scalar-evaluator
                                "scalar invocation regions require a typed evaluator"
                                {:step id :region (:region step)}))
                       (checked-computed-scalar
                        step
                        (evaluate-scalar
                         step
                         (into {}
                               (map (fn [{:keys [symbol value]}]
                                      [symbol (get values value)]))
                               (:operands step)))))

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
               {:values (assoc values id materialized)
                :lexical (assoc lexical (:symbol step) id)}))
           {:values parameter-values :lexical initial-lexical}
           (:steps plan))
          input-values
          (into {}
                (map (fn [{:keys [program-value invocation-value]}]
                       [program-value (get values invocation-value)]))
                (concat (:bindings plan) (:storage-bindings plan)))
          program-buffers (into {} (filter (comp materialized-buffer? val)) input-values)
          program-scalars (into {} (filter (comp typed-scalar? val)) input-values)]
      (validate!
       (->MaterializedInvocation
        plan values program-buffers program-scalars
        {:source-inspected false :driver-allocations 0})))))
