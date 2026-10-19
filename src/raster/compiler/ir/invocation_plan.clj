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
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.core.params-flatten :as params-flatten]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
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
(def ^:private external-dimension-values-key :external-dimension-values)

(declare fail! scalar-value?)

(defn- source-free-executable-values
  [parallel-program]
  (let [inner
        (into #{}
              (mapcat
               (fn [equation]
                 (let [algorithm (:algorithm equation)]
                   (when (soac/program-form? algorithm)
                     (let [facts (soac/facts algorithm)]
                       (concat
                        (get-in facts [:attributes :native-initialization-providers])
                        (get-in facts [:attributes :host-read-values])
                        (mapcat :values (get-in facts [:attributes :host-read-sites]))
                        (mapcat util/free-syms (:effects facts))
                        (mapcat
                         (fn [inner]
                           (let [equation-id (second inner)
                                 aliases (get-in facts [:equations equation-id :aliases])]
                             (concat (soac/operation-inputs inner)
                                     (nth inner 2)
                                     (mapcat util/free-syms (soac/operation-extents inner))
                                     (keys aliases) (vals aliases)
                                     (mapcat util/free-syms
                                             (get-in facts [:equations equation-id :effects]))
                                     (map :destination (soac/result-storage facts equation-id)))))
                         (soac/equations algorithm)))))))
              (:equations parallel-program)))]
    (into (into (set (:outputs parallel-program)) inner)
          (mapcat #(concat (:operands %) (:results %)))
          (:equations parallel-program))))

(defn source-free-shape-projections
  "Derive the bounded public-buffer shape projections required by a source-free program.

   A projected dimension is a retained private integral scalar used as the sole dimension of
   exactly one public rank-one buffer. It may be reused by other AbstractValue shapes, but it must
   not occur at the executable value boundary. The returned order follows public input order."
  [parallel-program]
  (let [inputs (vec (:inputs parallel-program))
        input-set (set inputs)
        values (:values parallel-program)
        executable-values (source-free-executable-values parallel-program)
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

(defn source-free-external-dimensions
  "Derive private direct-symbol shape dimensions that only a resident caller may supply."
  [parallel-program shape-projections]
  (let [values (:values parallel-program)
        public (set (:inputs parallel-program))
        projected (set (map :dimension shape-projections))
        executable (source-free-executable-values parallel-program)
        direct-dimensions
        (->> (sort-by pr-str (keys values))
             (mapcat #(get-in values [% :shape]))
             (filter symbol?)
             distinct
             (remove public)
             (remove projected)
             vec)]
    (mapv
     (fn [id]
       (let [value (get values id)
             declared (:dtype value)
             canonical (when (and (keyword? declared) (dtype/known? declared))
                         (dtype/canon declared))]
         (when-not (and (scalar-value? value) (contains? #{:int :long} canonical))
           (fail! :invocation-source-free-external-dimension
                  "external dimensions require retained integral scalar facts"
                  {:dimension id :value value}))
         (when (contains? executable id)
           (fail! :invocation-source-free-external-dimension
                  "external dimensions cannot also be executable values"
                  {:dimension id :executable-values executable}))
         {:id id :dtype canonical}))
     direct-dimensions)))

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

(defn- validate-external-dimension-marker!
  [plan]
  (let [attributes (:attributes plan)
        mode-present? (contains? attributes :external-dimensions)
        values-present? (contains? attributes external-dimension-values-key)
        dimensions (get attributes external-dimension-values-key)]
    (when-not (= mode-present? values-present?)
      (fail! :invocation-external-dimension-mode
             "external dimension mode and values must coincide" {}))
    (when mode-present?
      (when-not (and (= :resident-premise (:external-dimensions attributes))
                     (vector? dimensions) (seq dimensions)
                     (= (count dimensions) (count (distinct (map :id dimensions))))
                     (every? (fn [{:keys [id dtype] :as dimension}]
                               (let [value (get-in plan [:values id])]
                                 (and (= #{:id :dtype} (set (keys dimension)))
                                      (symbol? id) (contains? #{:int :long} dtype)
                                      (scalar-value? value)
                                      (= dtype (dtype/canon (:dtype value))))))
                             dimensions))
        (fail! :invocation-external-dimension-mode
               "external dimensions require one closed ordered retained-scalar marker"
               {:mode (:external-dimensions attributes) :dimensions dimensions}))))
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

(defn terminal-source-result
  "Describe the exact terminal source region and its logical producer ports.

   This is correspondence data for the InvocationPlan attribute boundary, NOT
   an executable, a physical scalar-export proof or permission to bypass source
   coverage. Policy and namespace are supplied by the actual source producer.
   A later owner must establish physical exports, compile under that retained
   context, guard dependencies, await completion and explicitly observe results.

   The initial boundary is one return expression over produced Double/Long
   scalars. Nested scopes and canonical completions remain intact; prefix
   computations, public inputs, array reads and private captures are not silently
   acquired as terminal parameters. Portable encoding does not prove purity or
   callee legality; logical producer IDs do not certify producer semantics.
   Existing resident APIs remain unchanged."
  [program {:keys [source-policy source-namespace]}]
  (parallel-program/validate! program)
  (source-policy/unchecked-math source-policy)
  (let [[head bindings & body] (:source program)]
    (when-not (and (contains? form/let-heads head)
                   (vector? bindings) (even? (count bindings)) (= 1 (count body))
                   (symbol? source-namespace) (nil? (namespace source-namespace)))
      (fail! :invocation-terminal-source
             "terminal observation requires one return at a flat source boundary" {}))
    (let [pairs (mapv vec (partition 2 bindings))
          symbols (mapv first pairs)
          _ (when-not (and (every? symbol? symbols) (distinct-vector? symbols))
              (fail! :invocation-terminal-source
                     "terminal observation requires distinct lexical source identities" {}))
          expression (first body)
          lexical (into (set symbols) (:inputs program))
          free (binding [util/*shadowing-locals* (into util/*shadowing-locals* lexical)]
                 (util/free-syms expression))
          producers (into {} (mapcat (fn [equation]
                                      (map-indexed (fn [position result]
                                                     [result [equation position]])
                                                   (:results equation))))
                          (:equations program))
          parameters
          (into []
                (keep (fn [result]
                        (when (contains? free result)
                          (let [[equation position] (get producers result)
                                value (get (:values program) result)]
                            (when-not (and (scalar-value? value)
                                           (contains? #{:double :long} (:dtype value)))
                              (fail! :invocation-terminal-carrier
                                     "terminal port requires a supported produced scalar"
                                     {:value result :abstract value}))
                            {:symbol result :value result :dtype (:dtype value)
                             :producer-equation (:id equation) :result-position position}))))
                (mapcat :results (:equations program)))
          missing (set/difference free (set (map :value parameters)))]
      (when (seq missing)
        (fail! :invocation-terminal-free-values
               "terminal source depends on values without retained numerical producers"
               {:values missing}))
      (when (empty? parameters)
        (fail! :invocation-terminal-free-values
               "terminal observation requires a numerical producer dependency" {}))
      (let [contract {:kind :terminal-host-result :site [:return]
                      :parameters parameters :body expression
                      :source-policy source-policy :source-namespace source-namespace}]
        ;; No runtime object, private completion Class or executable may enter
        ;; this portable semantic boundary. Such owners need a separate path.
        (semantic/canonical-bytes contract)
        contract))))

(defn validate-terminal-source-result!
  "Re-derive terminal correspondence from the actual program and source context.
   Matching this data alone does not attest ownership, exports or execution."
  [contract program context]
  (let [actual (terminal-source-result program context)]
    (when-not (semantic/equivalent? contract actual)
      (fail! :invocation-terminal-correspondence
             "terminal contract differs from its source or producer boundary" {}))
    contract))

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
    (validate-external-dimension-marker! plan)
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
  [plan-id ordinal symbol expression value environment lexical-values]
  (let [id (binding-id plan-id ordinal symbol)
        operation (descriptor/semantic-op expression)
        projection (when (buffer-value? value)
                     (descriptor/direct-allocation-projection expression lexical-values))
        initialization (descriptor/allocation-initialization
                        (if projection (descriptor/semantic-op (:constructor projection)) operation))]
    (cond
      (invocation-source/shape-projection-source expression)
      (let [operands (referenced-operands expression environment)
            source-symbol (invocation-source/shape-projection-source expression)]
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
      (cond-> (->BufferAllocation id symbol (allocation-operands value environment)
                                  initialization value)
        (:helper projection) (assoc :source-projection projection))

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
                    (let [lexical-values (into {} (map (fn [[symbol id]] [symbol (get values id)])) environment)
                          step (prefix-step id ordinal symbol expression value environment lexical-values)]
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

(declare derive-source-order)

(defn validate-against!
  "Validate that `plan` still matches a program's explicit value/input/output boundary."
  [plan parallel-program]
  (let [plan (validate! plan)
        source-free? (source-free-program? parallel-program)
        expected-projections (when source-free?
                               (source-free-shape-projections parallel-program))
        declared-projections (get-in plan [:attributes source-free-shape-projection-key])
        expected-dimensions (when source-free?
                              (source-free-external-dimensions
                               parallel-program expected-projections))
        declared-dimensions (get-in plan [:attributes external-dimension-values-key])
        external (mapv :program-value
                       (filter external-storage-binding? (:storage-bindings plan)))]
    (when-not (and (or (not source-free?)
                       (= {:source-dialect :typed-soac
                           :algorithm-dialect :typed-soac
                           :target-dialect :typed-invocation}
                          (select-keys (:attributes plan)
                                       [:source-dialect :algorithm-dialect :target-dialect])))
                   (= (vec expected-projections) (vec (or declared-projections [])))
                   (= (vec expected-dimensions) (vec (or declared-dimensions [])))
                   (= (boolean (seq expected-dimensions))
                      (= :resident-premise (get-in plan [:attributes :external-dimensions])))
                   (every? (fn [{:keys [id]}]
                             (= (get-in parallel-program [:values id])
                                (get-in plan [:values id])))
                           expected-dimensions)
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
              :expected-shape-projections expected-projections
              :external-dimensions declared-dimensions
              :expected-external-dimensions expected-dimensions}))
    (when (contains? (:attributes plan) :source-order)
      (let [actual (derive-source-order plan parallel-program)]
        (when-not (semantic/equivalent? actual (get-in plan [:attributes :source-order]))
          (fail! :invocation-source-order-mismatch
                 "retained invocation order differs from its exact source and typed producers"
                 {:expected actual :actual (get-in plan [:attributes :source-order])}))))
    plan))

(defn- derive-source-order
  [plan program]
  (let [[head bindings & body] (:source program)]
    (when-not (and (contains? '#{let let* clojure.core/let} head)
                   (vector? bindings) (even? (count bindings)))
      (fail! :invocation-source-order-boundary
             "source-ordered invocation requires a retained flat binding boundary"
             {:source (:source program)}))
    (let [pairs (mapv vec (partition 2 bindings))
          symbols (mapv first pairs)
          _ (when-not (= (count symbols) (count (set symbols)))
              (fail! :invocation-source-order-shadowing
                     "source positions require distinct retained binding identities" {}))
          steps-by-symbol (into {} (map (juxt :symbol identity)) (:steps plan))
          public (set (map :symbol (:parameters plan)))
          step-pairs (filterv #(contains? steps-by-symbol (first %)) pairs)
          expected-steps
          (:steps
           (reduce
            (fn [{:keys [steps environment]} [ordinal [symbol expression]]]
              (let [canonical (invocation-source/canonicalize-shape-projection
                               (:values plan) public symbol expression)
                    actual (get steps-by-symbol symbol)
                    _ (when (and (buffer-allocation? actual)
                                 (not (invocation-source/allocation-corresponds?
                                       canonical (:value actual) (:initialization actual)
                                       (into {} (map (fn [[symbol id]] [symbol (get-in plan [:values id])])) environment))))
                        (fail! :invocation-source-allocation
                               "source constructor differs from the retained allocation contract"
                               {:binding symbol :source canonical :value (:value actual)}))
                    _ (when (and (buffer-clone? actual)
                                 (not (invocation-source/clone-corresponds?
                                       canonical environment (:source actual))))
                        (fail! :invocation-source-clone
                               "source clone differs from the retained direct lexical copy"
                               {:binding symbol :source canonical}))
                    _ (when (and (shape-projection? actual)
                                 (not (invocation-source/projection-corresponds?
                                       canonical (:value actual))))
                        (fail! :invocation-source-projection
                               "source length conversions are not the retained exact projection"
                               {:binding symbol :source canonical}))
                    step (prefix-step (:id plan) ordinal symbol canonical
                                      (:value (get steps-by-symbol symbol)) environment
                                      (into {} (map (fn [[symbol id]] [symbol (get-in plan [:values id])])) environment))]
                {:steps (conj steps step) :environment (assoc environment symbol (:id step))}))
            {:steps [] :environment (into {} (map (juxt :symbol :id)) (:parameters plan))}
            (map-indexed vector step-pairs)))
          _ (when-not (semantic/equivalent? expected-steps (:steps plan))
              (fail! :invocation-source-step-mismatch
                     "invocation steps differ from their canonical source producers"
                     {:expected expected-steps :actual (:steps plan)}))
          allocation-claims (get-in program [:attributes :allocations])
          _ (when-not (= (count allocation-claims)
                         (count (set (map :destination allocation-claims))))
              (fail! :invocation-source-allocation-claims
                     "allocation destinations require unique source producers" {}))
          allocations (into {} (map (juxt :destination :source-expression)) allocation-claims)
          binding-order
          (mapv
           (fn [position [symbol expression]]
             (let [step (get steps-by-symbol symbol)
                   equation (parallel-program/equation-for-binding program symbol expression)
                   realization (invocation-source/realization-equation program symbol expression)
                   allocation? (and (contains? allocations symbol)
                                    (semantic/equivalent? expression (get allocations symbol)))]
               (when (and equation realization)
                 (fail! :invocation-source-order-ambiguous
                        "one source binding cannot be two different equation sites"
                        {:binding symbol}))
               (when (and allocation?
                          (not (invocation-source/allocation-corresponds?
                                expression (get-in program [:values symbol])
                                (descriptor/allocation-initialization
                                 (descriptor/semantic-op expression)) (:values program))))
                 (fail! :invocation-source-allocation
                        "source storage allocation differs from its retained physical contract"
                        {:binding symbol :source expression}))
               (when (and step equation)
                 (let [canonical (invocation-source/canonicalize-shape-projection
                                  (:values plan) public symbol expression)]
                   (when-not (and (true? (get-in equation [:attributes :host-only]))
                                  (= [symbol] (:results equation))
                                  (semantic/equivalent?
                                   canonical (invocation-source/scalar-equation-source equation)))
                     (fail! :invocation-source-order-shared-producer
                            "shared invocation/equation bindings require one identical host computation"
                            {:binding symbol :equation (:id equation)}))))
               (when-not (or step equation realization allocation? (effects/removable-expr? expression))
                 (fail! :invocation-source-order-uncovered
                        "retained source binding has no ordered invocation or equation producer"
                        {:binding symbol :position position}))
               (cond-> {:position position :binding symbol}
                 step (assoc :invocation-step (:id step))
                 equation (assoc :equation (:id equation))
                 realization (assoc :realization (:id realization))
                 allocation? (assoc :storage symbol))))
           (range) pairs)
          body-order
          (mapv
           (fn [index expression]
             (let [matches (filterv
                            #(and (or (= [:body index] (:site %))
                                      (and (= index (dec (count body))) (= [:return] (:site %))))
                                  (semantic/equivalent? expression (:source %)))
                            (:equations program))]
               (when (> (count matches) 1)
                 (fail! :invocation-source-order-ambiguous
                        "one body position cannot execute several equations" {:body-index index}))
               (when-not (or (seq matches) (effects/removable-expr? expression))
                 (fail! :invocation-source-order-uncovered
                        "retained source body has no ordered equation producer" {:body-index index}))
               (cond-> {:position (+ (count pairs) index) :body-index index}
                 (seq matches) (assoc :equation (:id (first matches))))))
           (range) body)
          order (into binding-order body-order)
          expected (mapv :id (:equations program))
          actual (into [] (keep :equation) order)]
      (when-not (= expected actual)
        (fail! :invocation-source-order-equations
               "source order must account for every equation exactly once"
               {:expected expected :actual actual}))
      order)))

(defn source-order
  "Derive source positions from the exact realized program, not pruned step ordinals.

   Entries reference existing invocation steps, equations and verified realization scaffolding.
   A shared host binding carries both IDs and denotes one computation, not two evaluations.
   This is correspondence data only: it does not authorize hoisting, binding unresolved storage,
   skipping checked evaluation, or treating a graph submission as completion. Source-free and
   non-flat programs require their separate explicit execution contract."
  [plan program]
  (let [plan (validate-against! plan program)]
    (if (contains? (:attributes plan) :source-order)
      (get-in plan [:attributes :source-order])
      (derive-source-order plan program))))
