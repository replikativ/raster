(ns raster.compiler.ir.resident-plan
  "Certifying conversion from one compiled resident descriptor to a one-instance LinkPlan.

   `compile-gpu-program` is the source dialect: it owns ordered parameters, executable ABIs,
   scalar/shape closures and default roles. `LinkPlan` is the target dialect: it owns stable value
   identities, realized views, ownership and ordered effects. This namespace is the only conversion
   between those contracts. It returns a checkable witness rather than asking a runtime to recover
   facts from symbols, buffer names, or emitted kernels."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.link-plan :as link-plan]))

(defrecord ResidentPlanCertificate
           [source-dialect target-dialect plan-id target instance-id parameter-order
            array-parameters scalar-parameters bindings scalars roles schedule values outputs
            aliases])
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

(defn- value-contract [symbol origin node]
  {:symbol symbol
   :origin origin
   :node (:id node)
   :dtype (dtype/canon (get-in node [:view :dtype]))
   :shape (get-in node [:view :shape])
   :strides (get-in node [:view :strides])
   :memory-space (get-in node [:view :allocation :memory-space])
   :device (get-in node [:view :allocation :device])
   :ownership (get-in node [:view :allocation :ownership])
   :role (:role node)})

(defn- derive-certificate [plan]
  (let [instance (first (:instances plan))
        descriptor (:descriptor instance)
        symbol-by-node (set/map-invert (:bindings instance))
        alloc-symbols (set (map :sym (:allocs descriptor)))
        values (into {}
                     (map (fn [[node-id node]]
                            (let [symbol (get symbol-by-node node-id)]
                              [symbol (value-contract symbol
                                                      (if (contains? alloc-symbols symbol)
                                                        :allocation :parameter)
                                                      node)])))
                     (:nodes plan))]
    (->ResidentPlanCertificate
     :resident-program :link-plan (:id plan) (:target plan) (:id instance)
     (:all-params descriptor) (:array-params descriptor) (:scalar-params descriptor)
     (:bindings instance) (:scalars instance) (link-plan/instance-roles plan instance)
     (:schedule instance) values (:outputs plan) (:aliases plan))))

(defn verify!
  "Verify and return a CertifiedResidentPlan.

   Verification revalidates the target plan and independently derives every witnessed fact from
   that plan's resident descriptor, stable bindings, scalar environment and views. A stale or
   modified certificate therefore fails before runtime instantiation."
  [lowering]
  (when-not (certified-plan? lowering)
    (throw (ex-info "expected a CertifiedResidentPlan"
                    {:reason :resident-plan-lowering-type :actual (type lowering)})))
  (let [plan (link-plan/validate! (:plan lowering))
        certificate (:certificate lowering)
        expected (derive-certificate plan)]
    (when-not (certificate? certificate)
      (throw (ex-info "resident-plan lowering requires a ResidentPlanCertificate"
                      {:reason :resident-plan-certificate-type :actual (type certificate)})))
    (when-not (= expected certificate)
      (throw (ex-info "resident-program to LinkPlan certificate does not match its target"
                      {:reason :resident-plan-certificate
                       :expected expected :actual certificate})))
    lowering))

(defn lower
  "Lower one resident descriptor and its specialization arguments to a certified LinkPlan.

   Options:
   - `:id`, `:target`, `:descriptor`, and ordered `:arguments` are required;
   - `:roles` overrides descriptor array roles;
   - `:outputs` is an ordered vector of public output symbols (default: `:result-sym`);
   - `:shapes` may refine the default flat parameter/allocation shapes;
   - `:node-ids` supplies stable public identities (default `[plan-id symbol]`);
   - `:ownership`, `:memory-space`, `:aliases`, and `:attributes` refine LinkPlan storage.

   Host array arguments remain exact initializers for owned parameter allocations, including
   output/state parameters. Borrowed/external allocations use the arguments only to specialize
   dtype and shape; their storage and readiness are supplied explicitly at instantiation. This
   preserves `bind-program!`'s initial-value semantics without confusing ownership with readiness."
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
        resolved-node-ids (mapv #(node-id-for id node-ids %) ordered-symbols)
        _ (unique-vector! :resident-plan-node-ids "resident-plan node identities"
                          resolved-node-ids)
        nodes
        (mapv
         (fn [symbol]
           (let [parameter? (contains? array-symbols symbol)
                 argument (when parameter? (get argument-map symbol))
                 allocation (get alloc-map symbol)
                 default-elements (if parameter?
                                    (array-length symbol argument)
                                    (try (long ((:size-fn allocation) arguments))
                                         (catch Exception error
                                           (throw (ex-info
                                                   "resident allocation shape did not resolve"
                                                   {:reason :resident-plan-allocation-shape
                                                    :symbol symbol}
                                                   error)))))
                 shape (vec (get shapes symbol [default-elements]))
                 _ (when-not (= default-elements (reduce * 1 shape))
                     (throw (ex-info "resident value shape differs from its storage extent"
                                     {:reason :resident-plan-shape :symbol symbol
                                      :elements default-elements :shape shape})))
                 storage-dtype (if parameter?
                                 (primitive-array-dtype argument)
                                 (or (:dtype allocation) (:dtype descriptor)))
                 node-id (node-id-for id node-ids symbol)
                 value-ownership (get ownership symbol :owned)]
             (link-plan/node
              {:id node-id :allocation-id node-id :dtype storage-dtype :shape shape
               :device target :memory-space (get memory-space symbol :device)
               :ownership value-ownership
               :role (node-role symbol parameter? descriptor roles output-symbols)
               :source (when (= :owned value-ownership) argument)})))
         ordered-symbols)
        bindings (zipmap ordered-symbols resolved-node-ids)
        instance (link-plan/instance
                  {:id (or instance-id [id :instance]) :descriptor descriptor
                   :bindings bindings :scalars scalar-values :schedule (:schedule descriptor)
                   :roles roles})
        plan (link-plan/make
              {:id id :target target :nodes nodes :instances [instance]
               :outputs (mapv bindings outputs) :aliases aliases
               :attributes (assoc attributes :lowered-from :resident-program)})
        lowering (->CertifiedResidentPlan plan (derive-certificate plan))]
    (verify! lowering)))
