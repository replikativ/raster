(ns raster.compiler.ir.resident-plan
  "Certifying conversion from one compiled resident descriptor to a one-instance LinkPlan.

   `compile-gpu-program` is the source dialect: it owns ordered parameters, executable ABIs,
   scalar/shape closures and default roles. `LinkPlan` is the target dialect: it owns stable value
   identities, realized views, ownership and ordered effects. This namespace is the only conversion
   between those contracts. It returns a checkable witness rather than asking a runtime to recover
   facts from symbols, buffer names, or emitted kernels."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.link-plan :as link-plan]))

(defrecord ResidentPlanCertificate
           [source-dialect target-dialect plan-id target instance-id parameter-order
            array-parameters scalar-parameters bindings scalars roles schedule values outputs
            aliases effect-evidence])
(defrecord CertifiedResidentPlan [plan certificate])

(defn certificate? [x]
  (instance? ResidentPlanCertificate x))

(defn certified-plan? [x]
  (instance? CertifiedResidentPlan x))

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

(defn- array-length [symbol value]
  (when-not (primitive-array-dtype value)
    (throw (ex-info "resident program array arguments must be primitive JVM arrays"
                    {:reason :resident-plan-array-argument :symbol symbol
                     :actual (some-> value type)})))
  (java.lang.reflect.Array/getLength value))

(defn- unique-vector! [reason label values]
  (let [values (vec values)]
    (when-not (= (count values) (count (distinct values)))
      (throw (ex-info (str label " must be unique and ordered")
                      {:reason reason :values values})))
    values))

(defn- descriptor-environment!
  [descriptor arguments]
  (let [{:keys [all-params array-params scalar-params allocs]} descriptor]
    (when-not (and (map? descriptor) (vector? all-params) (vector? array-params)
                   (vector? scalar-params) (vector? allocs) (vector? (:steps descriptor)))
      (throw (ex-info "resident-plan lowering requires a complete resident descriptor"
                      {:reason :resident-plan-descriptor :descriptor descriptor})))
    (unique-vector! :resident-plan-parameters "resident descriptor parameters" all-params)
    (when-not (= (count all-params) (count arguments))
      (throw (ex-info "resident-plan arguments differ from the descriptor parameter order"
                      {:reason :resident-plan-arguments :parameters all-params
                       :expected (count all-params) :actual (count arguments)})))
    (let [parameter-set (set all-params)
          array-set (set array-params)
          scalar-set (set scalar-params)]
      (when-not (and (set/subset? array-set parameter-set)
                     (set/subset? scalar-set parameter-set)
                     (empty? (set/intersection array-set scalar-set))
                     (= parameter-set (set/union array-set scalar-set)))
        (throw (ex-info "resident descriptor array/scalar parameters do not partition its ABI"
                        {:reason :resident-plan-parameter-partition :parameters parameter-set
                         :arrays array-set :scalars scalar-set})))
      (when-not (= (count allocs) (count (distinct (map :sym allocs))))
        (throw (ex-info "resident descriptor allocation symbols must be unique"
                        {:reason :resident-plan-allocation-symbols
                         :symbols (mapv :sym allocs)})))
      (zipmap all-params arguments))))

(defn- node-id-for [plan-id node-ids symbol]
  (get node-ids symbol [plan-id symbol]))

(defn- node-role [symbol parameter? descriptor roles output-symbols]
  (let [role (or (get roles symbol)
                 (when parameter? (get (:array-roles descriptor) symbol))
                 (when (contains? output-symbols symbol) :output)
                 (if parameter? :input :internal))]
    (if (= :scratch role) :internal role)))

(defn- physical-option [options symbol field default]
  (get options [symbol field] (get options symbol default)))

(defn- validate-value-spec! [symbol {:keys [abstract physical-layout leaves] :as spec}]
  (when-not (and (map? spec) (vector? leaves) (seq leaves))
    (throw (ex-info "resident logical value specification requires ordered leaves"
                    {:reason :resident-plan-value-spec :symbol symbol :spec spec})))
  (when-not abstract
    (throw (ex-info "resident composite value specification requires an AbstractValue"
                    {:reason :resident-plan-value-abstract :symbol symbol})))
  (abstract-value/validate! abstract)
  (when-not (or (nil? physical-layout) (map? physical-layout))
    (throw (ex-info "resident value physical layout must be a descriptor map"
                    {:reason :resident-plan-value-layout :symbol symbol
                     :physical-layout physical-layout})))
  (let [fields (mapv :field leaves)]
    (when (or (some nil? fields) (not= (count fields) (count (distinct fields))))
      (throw (ex-info "resident value fields must be present, unique, and ordered"
                      {:reason :resident-plan-value-fields :symbol symbol :fields fields}))))
  spec)

(defn- parameter-leaves [symbol argument value-spec]
  (if value-spec
    (let [{:keys [leaves]} (validate-value-spec! symbol value-spec)
          fields (mapv :field leaves)]
      (when-not (and (map? argument) (= (set fields) (set (keys argument))))
        (throw (ex-info "resident composite argument must map every declared field to storage"
                        {:reason :resident-plan-composite-argument :symbol symbol
                         :expected (set fields) :actual (when (map? argument)
                                                          (set (keys argument)))})))
      (mapv (fn [{:keys [field dtype] :as leaf}]
              (let [source (get argument field)
                    actual-dtype (primitive-array-dtype source)]
                (when-not actual-dtype
                  (throw (ex-info "resident composite fields must be primitive JVM arrays"
                                  {:reason :resident-plan-array-argument :symbol symbol
                                   :field field :actual (some-> source type)})))
                (when (and dtype (not= (dtype/canon dtype) actual-dtype))
                  (throw (ex-info "resident composite field dtype differs from its specification"
                                  {:reason :resident-plan-field-dtype :symbol symbol :field field
                                   :expected dtype :actual actual-dtype})))
                (assoc leaf :dtype actual-dtype :elements (array-length symbol source)
                       :source source)))
            leaves))
    [{:field :value :dtype (primitive-array-dtype argument)
      :elements (array-length symbol argument) :source argument}]))

(defn- allocation-leaves [symbol allocation default-dtype arguments]
  (mapv (fn [{:keys [field dtype size-fn] :as leaf}]
          (let [elements
                (try (long (size-fn arguments))
                     (catch Exception error
                       (throw (ex-info "resident allocation shape did not resolve"
                                       {:reason :resident-plan-allocation-shape
                                        :symbol symbol :field field}
                                       error))))]
            (assoc leaf :dtype (dtype/canon dtype) :elements elements :source nil)))
        (link-plan/descriptor-allocation-leaves allocation default-dtype)))

(defn- leaf-node-id [plan-id node-ids value-id symbol field leaf-count]
  (if (= 1 leaf-count)
    value-id
    (get node-ids [symbol field] [plan-id symbol field])))

(defn- leaf-shape [shapes symbol field leaf-count elements]
  (let [shape (vec (or (get shapes [symbol field])
                       (when (= 1 leaf-count) (get shapes symbol))
                       [elements]))]
    (when-not (= elements (reduce * 1 shape))
      (throw (ex-info "resident value shape differs from its storage extent"
                      {:reason :resident-plan-shape :symbol symbol :field field
                       :elements elements :shape shape})))
    shape))

(defn- leaf-contract [name node]
  {:name name
   :node (:id node)
   :dtype (dtype/canon (get-in node [:view :dtype]))
   :shape (get-in node [:view :shape])
   :strides (get-in node [:view :strides])
   :memory-space (get-in node [:view :allocation :memory-space])
   :device (get-in node [:view :allocation :device])
   :ownership (get-in node [:view :allocation :ownership])
   :role (:role node)})

(defn- value-contract [symbol origin value nodes]
  (let [leaves (mapv (fn [{:keys [name node]}]
                       (leaf-contract name (get nodes node)))
                     (:leaves value))
        flat (when (= 1 (count leaves)) (first leaves))]
    (merge {:symbol symbol
            :origin origin
            :value (:id value)
            :abstract (:abstract value)
            :physical-layout (:physical-layout value)
            :leaves leaves}
           ;; Preserve the original certificate query surface for one-leaf descriptors.
           (dissoc flat :name))))

(defn- derive-certificate [plan effect-evidence]
  (let [instance (first (:instances plan))
        descriptor (:descriptor instance)
        alloc-symbols (set (map :sym (:allocs descriptor)))
        values (into {}
                     (map (fn [[symbol value-id]]
                            [symbol (value-contract
                                     symbol
                                     (if (contains? alloc-symbols symbol)
                                       :allocation :parameter)
                                     (get-in plan [:values value-id])
                                     (:nodes plan))]))
                     (:bindings instance))]
    (->ResidentPlanCertificate
     :resident-program :link-plan (:id plan) (:target plan) (:id instance)
     (:all-params descriptor) (:array-params descriptor) (:scalar-params descriptor)
     (:bindings instance) (:scalars instance) (link-plan/instance-roles plan instance)
     (:schedule instance) values (:outputs plan) (:aliases plan) effect-evidence)))

(defn verify!
  "Verify and return a CertifiedResidentPlan.

   Verification revalidates the target plan and independently derives every witnessed fact from
   that plan's resident descriptor, stable bindings, scalar environment and views. A stale or
   modified certificate therefore fails before runtime instantiation."
  [lowering]
  (when-not (certified-plan? lowering)
    (throw (ex-info "expected a CertifiedResidentPlan"
                    {:reason :resident-plan-lowering-type :actual (type lowering)})))
  (let [{:keys [plan effect-evidence]}
        (link-plan/validate-with-effect-evidence! (:plan lowering))
        certificate (:certificate lowering)
        expected (derive-certificate plan effect-evidence)]
    (when-not (certificate? certificate)
      (throw (ex-info "resident-plan lowering requires a ResidentPlanCertificate"
                      {:reason :resident-plan-certificate-type :actual (type certificate)})))
    (when-not (= expected certificate)
      (throw (ex-info "resident-program to LinkPlan certificate does not match its target"
                      {:reason :resident-plan-certificate
                       :expected expected :actual certificate})))
    lowering))

(defn ^:no-doc source-free-template
  "Remove invocation-specific host sources from a freshly certified resident lowering.

   The returned value is an internal structural template, not an independently trusted artifact:
   callers must retain it only behind the compiler template cache and instantiate it through
   `bind-template`. Descriptor code, realized views, effects and the certificate are unchanged."
  [lowering]
  (when-not (certified-plan? lowering)
    (throw (ex-info "resident plan template requires a certified lowering"
                    {:reason :resident-plan-template-type :actual (type lowering)})))
  (update lowering :plan
          (fn [plan]
            (-> plan
                (update :nodes update-vals #(assoc % :source nil))
                (update :instances
                        (fn [instances]
                          (mapv #(assoc % :arguments []) instances)))))))

(defn ^:no-doc bind-template
  "Bind current primitive-array sources to a source-free resident plan template.

   This checks the complete descriptor argument partition, scalar environment, logical field
   order, dtype and realized element extent before installing sources. Those are precisely the
   invocation-varying facts; the cached topology/effects were validated when the template was
   constructed. The certificate remains valid because it intentionally witnesses contracts, not
   host object identities or contents."
  [template arguments]
  (when-not (certified-plan? template)
    (throw (ex-info "resident plan binding requires a certified template"
                    {:reason :resident-plan-template-type :actual (type template)})))
  (let [plan (:plan template)
        instances (:instances plan)
        _ (when-not (= 1 (count instances))
            (throw (ex-info "resident plan template requires exactly one instance"
                            {:reason :resident-plan-template-instances
                             :instances (count instances)})))
        instance (first instances)
        descriptor (:descriptor instance)
        argument-map (descriptor-environment! descriptor arguments)
        scalar-values (select-keys argument-map (:scalar-params descriptor))
        expected-scalars (:scalars (:certificate template))
        _ (when-not (= expected-scalars scalar-values)
            (throw (ex-info "resident plan template scalar specialization changed"
                            {:reason :resident-plan-template-scalars
                             :expected expected-scalars :actual scalar-values})))
        value-specs (or (:value-specs descriptor) {})
        nodes
        (reduce
         (fn [nodes symbol]
           (let [value-id (get-in template [:certificate :bindings symbol])
                 logical-leaves (get-in plan [:values value-id :leaves])
                 source-leaves (parameter-leaves symbol (get argument-map symbol)
                                                 (get value-specs symbol))]
             (when-not (= (count logical-leaves) (count source-leaves))
               (throw (ex-info "resident plan template field count changed"
                               {:reason :resident-plan-template-fields :symbol symbol
                                :expected (mapv :name logical-leaves)
                                :actual (mapv :field source-leaves)})))
             (reduce
              (fn [nodes [{:keys [name node]} {:keys [field dtype elements source]}]]
                (let [target (get nodes node)
                      target-dtype (dtype/canon (get-in target [:view :dtype]))
                      target-elements (reduce *' 1 (get-in target [:view :shape]))]
                  (when-not (and (= name field)
                                 (= target-dtype (dtype/canon dtype))
                                 (= target-elements elements))
                    (throw (ex-info "resident plan template array specialization changed"
                                    {:reason :resident-plan-template-array
                                     :symbol symbol :field field
                                     :expected {:name name :dtype target-dtype
                                                :elements target-elements}
                                     :actual {:name field :dtype (dtype/canon dtype)
                                              :elements elements}})))
                  (assoc nodes node
                         (cond-> (assoc target :source nil)
                           (= :owned (get-in target [:view :allocation :ownership]))
                           (assoc :source source)))))
              nodes (map vector logical-leaves source-leaves))))
         (:nodes plan) (:array-params descriptor))
        bound-plan (assoc plan :nodes nodes
                          :instances [(assoc instance :arguments (vec arguments)
                                             :scalars scalar-values)])]
    (->CertifiedResidentPlan bound-plan (:certificate template))))

(defn lower
  "Lower one resident descriptor and its specialization arguments to a certified LinkPlan.

   Options:
   - `:id`, `:target`, `:descriptor`, and ordered `:arguments` are required;
   - `:roles` overrides descriptor array roles;
   - `:outputs` is an ordered vector of public output symbols (default: `:result-sym`);
   - descriptor `:value-specs` describes composite array parameters as an AbstractValue, physical
     layout, and ordered leaves; composite `:allocs` use the same facets plus leaf size closures;
   - `:shapes` may refine flat values by symbol or physical leaves by `[symbol field]`;
   - `:node-ids` supplies logical identities by symbol and optional physical identities by
     `[symbol field]`;
   - `:ownership`, `:memory-space`, `:aliases`, and `:attributes` refine LinkPlan storage.

   Flat arguments remain primitive arrays. Composite arguments are field-to-primitive-array maps.
   They remain exact initializers for owned parameter leaves, including output/state parameters.
   Borrowed/external allocations use them only to specialize dtype and shape; their storage and
   readiness are supplied explicitly at instantiation."
  [{:keys [id target descriptor arguments roles outputs shapes node-ids ownership memory-space
           aliases attributes instance-id]
    :or {roles {} shapes {} node-ids {} ownership {} memory-space {}
         aliases #{} attributes {}}}]
  (when (nil? id)
    (throw (ex-info "resident-plan lowering requires a stable plan identity"
                    {:reason :resident-plan-id})))
  (let [argument-map (descriptor-environment! descriptor arguments)
        pointer-symbols (link-plan/descriptor-pointer-symbols descriptor)
        array-symbols (set (:array-params descriptor))
        alloc-map (into {} (map (juxt :sym identity)) (:allocs descriptor))
        known-pointers (set/union array-symbols (set (keys alloc-map)))
        _ (when-not (= pointer-symbols known-pointers)
            (throw (ex-info "resident descriptor pointer ABI has no exact value contract"
                            {:reason :resident-plan-pointer-contract
                             :pointers pointer-symbols :parameters array-symbols
                             :allocations (set (keys alloc-map))
                             :missing (set/difference pointer-symbols known-pointers)
                             :unused (set/difference known-pointers pointer-symbols)})))
        outputs (unique-vector!
                 :resident-plan-outputs "resident-plan output symbols"
                 (or outputs (when-let [result (:result-sym descriptor)] [result])))
        _ (when-not (set/subset? (set outputs) pointer-symbols)
            (throw (ex-info "resident-plan outputs must name descriptor pointer values"
                            {:reason :resident-plan-output-symbols :outputs outputs
                             :pointers pointer-symbols})))
        output-symbols (set outputs)
        scalar-values (select-keys argument-map (:scalar-params descriptor))
        ordered-symbols (vec (concat (:array-params descriptor) (map :sym (:allocs descriptor))))
        value-specs (or (:value-specs descriptor) {})
        _ (when-not (set/subset? (set (keys value-specs)) array-symbols)
            (throw (ex-info "resident descriptor value specs must name array parameters"
                            {:reason :resident-plan-value-spec-symbols
                             :specs (set (keys value-specs)) :arrays array-symbols})))
        value-descriptors
        (mapv
         (fn [symbol]
           (let [parameter? (contains? array-symbols symbol)
                 allocation (get alloc-map symbol)
                 value-spec (if parameter? (get value-specs symbol)
                                (when (contains? allocation :leaves) allocation))
                 _ (when value-spec (validate-value-spec! symbol value-spec))
                 leaves (if parameter?
                          (parameter-leaves symbol (get argument-map symbol) value-spec)
                          (allocation-leaves symbol allocation (:dtype descriptor) arguments))
                 value-id (node-id-for id node-ids symbol)
                 leaf-count (count leaves)
                 leaves (mapv (fn [{:keys [field] :as leaf}]
                                (assoc leaf :node-id
                                       (leaf-node-id id node-ids value-id symbol field leaf-count)))
                              leaves)]
             {:symbol symbol :parameter? parameter? :allocation allocation
              :value-spec value-spec :value-id value-id :leaves leaves
              :role (node-role symbol parameter? descriptor roles output-symbols)}))
         ordered-symbols)
        _ (unique-vector! :resident-plan-node-ids "resident-plan value identities"
                          (mapv :value-id value-descriptors))
        _ (unique-vector! :resident-plan-node-ids "resident-plan node identities"
                          (mapv :node-id (mapcat :leaves value-descriptors)))
        nodes
        (vec
         (mapcat
          (fn [{:keys [symbol parameter? role leaves]}]
            (let [leaf-count (count leaves)]
              (mapv
               (fn [{:keys [field dtype elements source node-id]}]
                 (let [value-ownership (physical-option ownership symbol field :owned)]
                   (link-plan/node
                    {:id node-id :allocation-id node-id :dtype dtype
                     :shape (leaf-shape shapes symbol field leaf-count elements)
                     :device target
                     :memory-space (physical-option memory-space symbol field :device)
                     :ownership value-ownership :role role
                     :source (when (and parameter? (= :owned value-ownership)) source)})))
               leaves)))
          value-descriptors))
        logical-values
        (vec
         (keep (fn [{:keys [value-id value-spec leaves]}]
                 (when value-spec
                   (link-plan/value
                    {:id value-id :abstract (:abstract value-spec)
                     :physical-layout (:physical-layout value-spec)
                     :leaves (mapv (fn [{:keys [field node-id]}]
                                     {:name field :node node-id})
                                   leaves)})))
               value-descriptors))
        value-by-symbol (into {} (map (juxt :symbol identity)) value-descriptors)
        bindings (into {} (map (juxt :symbol :value-id)) value-descriptors)
        instance (link-plan/instance
                  {:id (or instance-id [id :instance]) :descriptor descriptor
                   :bindings bindings :scalars scalar-values :schedule (:schedule descriptor)
                   :roles roles :arguments (vec arguments)})
        {:keys [plan effect-evidence]}
        (link-plan/make-with-effect-evidence
         {:id id :target target :nodes nodes :values logical-values :instances [instance]
          :outputs (vec (mapcat #(map :node-id (:leaves (get value-by-symbol %))) outputs))
          :aliases aliases
          :attributes (assoc attributes :lowered-from :resident-program)})
        lowering (->CertifiedResidentPlan plan (derive-certificate plan effect-evidence))]
    ;; `plan` and its certificate are constructed together from the same validated inputs. The
    ;; public `verify!` boundary independently re-derives this witness when it later crosses into
    ;; composition, a cache, or the runtime; immediately doing the same work here is redundant.
    lowering))
