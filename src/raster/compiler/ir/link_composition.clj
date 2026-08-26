(ns raster.compiler.ir.link-composition
  "Certified, allocation-free composition of independently lowered resident programs.

   Composition is a pure LinkPlan rewrite. Component node/allocation/instance identities are
   namespaced first; explicit dataflow connections and shared boundary values then canonicalize
   identities; one ordinary LinkPlan is validated last. The runtime therefore allocates and
   records the composite only once and cannot insert an implicit intermediate copy."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.resident-plan :as resident-plan]))

(defrecord LinkCompositionCertificate
           [source-dialect target-dialect plan-id target component-plan-ids
            connections shares value-mapping node-mapping allocation-mapping instance-mapping
            outputs])
(defrecord CertifiedLinkComposition [plan certificate components specification])

(defn certificate? [x]
  (instance? LinkCompositionCertificate x))

(defn certified-composition? [x]
  (instance? CertifiedLinkComposition x))

(declare verify!)

(defn- verify-component! [component]
  (cond
    (resident-plan/certified-plan? component) (resident-plan/verify! component)
    (certified-composition? component) (verify! component)
    :else
    (throw (ex-info "link composition components must carry a verified lowering certificate"
                    {:reason :link-composition-component-type :actual (type component)}))))

(defn- normalize-components! [components]
  (let [components (mapv (fn [component]
                           (when-not (and (map? component) (contains? component :id)
                                          (contains? component :lowering))
                             (throw (ex-info "each link component requires :id and :lowering"
                                             {:reason :link-composition-component
                                              :component component})))
                           (update component :lowering verify-component!))
                         components)
        ids (mapv :id components)]
    (when (empty? components)
      (throw (ex-info "link composition requires at least one certified component"
                      {:reason :link-composition-components})))
    (when-not (= (count ids) (count (distinct ids)))
      (throw (ex-info "link composition component identities must be unique and ordered"
                      {:reason :link-composition-component-ids :ids ids})))
    components))

(defn- value-ref! [component-plans [component-id value-id :as reference]]
  (when-not (and (vector? reference) (= 2 (count reference)))
    (throw (ex-info "a link composition value reference is [component-id value-id]"
                    {:reason :link-composition-value-reference :reference reference})))
  (let [plan (get component-plans component-id)]
    (when-not plan
      (throw (ex-info "a link composition node reference names an absent component"
                      {:reason :link-composition-component-reference
                       :reference reference :components (set (keys component-plans))})))
    (if (contains? (:values plan) value-id)
      reference
      (let [owners (into []
                         (keep (fn [[candidate value]]
                                 (when (some #(= value-id (:node %)) (:leaves value))
                                   candidate)))
                         (:values plan))]
        (if (= 1 (count owners))
          ;; Compatibility for callers holding a physical output identity from an older
          ;; certificate. Even one leaf resolves to its whole logical value atomically.
          [component-id (first owners)]
          (throw (ex-info "a link composition value reference names an absent component value"
                          {:reason :link-composition-value-reference :reference reference
                           :physical-owners owners
                           :values (set (keys (:values plan)))})))))))

(defn- namespace-id [composition-id component-id kind id]
  [composition-id component-id kind id])

(defn- namespace-node [composition-id component-id node]
  (let [view (:view node)
        allocation (:allocation view)]
    (assoc node
           :id (namespace-id composition-id component-id :node (:id node))
           :view (assoc view
                        :id (namespace-id composition-id component-id :view (:id view))
                        :allocation
                        (assoc allocation :id (namespace-id composition-id component-id
                                                            :allocation (:id allocation)))))))

(defn- namespace-value [composition-id component-id node-mapping value]
  (assoc value
         :id (namespace-id composition-id component-id :value (:id value))
         :leaves (mapv #(update % :node node-mapping) (:leaves value))))

(defn- namespace-instance [composition-id component-id value-mapping instance]
  (assoc instance
         :id (namespace-id composition-id component-id :instance (:id instance))
         :bindings (update-vals (:bindings instance) value-mapping)))

(defn- endpoint-contract [node]
  (let [view (:view node)
        allocation (:allocation view)]
    {:dtype (dtype/canon (:dtype view))
     :shape (:shape view)
     :strides (:strides view)
     :byte-offset (:byte-offset view)
     :byte-length (:byte-length view)
     :allocation (select-keys allocation
                              [:byte-size :memory-space :device :alignment :coherence
                               :ownership])}))

(defn- value-contract [value nodes]
  {:abstract (:abstract value)
   :physical-layout (:physical-layout value)
   :leaves (mapv (fn [{:keys [name node]}]
                   {:name name :view (endpoint-contract (get nodes node))})
                 (:leaves value))})

(defn- require-equal-contract! [kind references values nodes]
  (let [contracts (mapv #(value-contract (get values %) nodes) references)]
    (when-not (apply = contracts)
      (throw (ex-info "composed boundary values have different realized view contracts"
                      {:reason :link-composition-view-contract :kind kind
                       :references references :contracts contracts})))))

(defn- require-single-node-allocations! [references values nodes]
  ;; Step 4 generalizes this to ranged/subview boundary unification. Rejecting it here is crucial:
  ;; rebasing one endpoint of a multi-view allocation would silently change the other aliases.
  (let [counts (frequencies (map #(get-in % [:view :allocation :id]) (vals nodes)))]
    (doseq [reference references
            node-id (map :node (:leaves (get values reference)))
            :let [allocation-id (get-in nodes [node-id :view :allocation :id])]]
      (when-not (= 1 (get counts allocation-id))
        (throw (ex-info "composition across a multi-view allocation requires ranged-view linking"
                        {:reason :link-composition-ranged-view :value reference :node node-id
                         :allocation allocation-id :views (get counts allocation-id)}))))))

(defn- value-role [plan value-id]
  (get-in plan [:nodes (get-in plan [:values value-id :leaves 0 :node]) :role]))

(defn- public-output-value? [plan value-id]
  (contains? (set (link-plan/output-value-ids plan)) value-id))

(defn- normalize-connections! [connections component-plans component-order]
  (let [connections
        (mapv (fn [{:keys [from to] :as connection}]
                (when-not (= #{:from :to} (set (keys connection)))
                  (throw (ex-info "a dataflow connection requires exactly :from and :to"
                                  {:reason :link-composition-connection
                                   :connection connection})))
                {:from (value-ref! component-plans from)
                 :to (value-ref! component-plans to)})
              connections)
        targets (mapv :to connections)]
    (when-not (= (count targets) (count (distinct targets)))
      (throw (ex-info "a composed consumer node has more than one producer"
                      {:reason :link-composition-producers :targets targets})))
    (doseq [{:keys [from to]} connections
            :let [[from-component from-value] from
                  [to-component to-value] to
                  from-plan (get component-plans from-component)
                  to-plan (get component-plans to-component)]]
      (when-not (< (get component-order from-component) (get component-order to-component))
        (throw (ex-info "link dataflow connections must follow component schedule order"
                        {:reason :link-composition-order :from from :to to})))
      (when-not (public-output-value? from-plan from-value)
        (throw (ex-info "a link dataflow producer must be a declared logical component output"
                        {:reason :link-composition-producer-boundary :from from
                         :outputs (:outputs from-plan)})))
      (when-not (= :input (value-role to-plan to-value))
        (throw (ex-info "a link dataflow consumer must be a logical input boundary"
                        {:reason :link-composition-consumer-role :to to
                         :role (value-role to-plan to-value)}))))
    connections))

(defn- normalize-shares! [shares component-plans]
  (mapv
   (fn [share]
     (let [references (mapv #(value-ref! component-plans %) share)]
       (when-not (<= 2 (count references))
         (throw (ex-info "a shared boundary group requires at least two nodes"
                         {:reason :link-composition-share :share share})))
       (when-not (= (count references) (count (distinct references)))
         (throw (ex-info "a shared boundary group cannot repeat a node"
                         {:reason :link-composition-share-duplicates :share share})))
       (let [roles (mapv (fn [[component-id value-id]]
                           (value-role (get component-plans component-id) value-id))
                         references)]
         (when-not (and (apply = roles) (contains? #{:input :constant} (first roles)))
           (throw (ex-info "shared boundaries must all be inputs or all be constants"
                           {:reason :link-composition-share-role :share share :roles roles}))))
       references))
   shares))

(defn- distinct-groups! [connections shares]
  (let [connection-refs (set (mapcat (juxt :from :to) connections))
        share-refs (mapcat identity shares)]
    (when-not (= (count share-refs) (count (distinct share-refs)))
      (throw (ex-info "a boundary node occurs in more than one shared group"
                      {:reason :link-composition-share-overlap :shares shares})))
    (when-let [overlap (seq (set/intersection connection-refs (set share-refs)))]
      (throw (ex-info "dataflow and shared-value composition cannot claim the same boundary node"
                      {:reason :link-composition-boundary-overlap :nodes (set overlap)})))))

(defn- canonical-source [references nodes]
  (let [sources (keep (comp :source nodes) references)]
    (when-not (or (<= (count sources) 1)
                  (every? #(identical? (first sources) %) (rest sources)))
      (throw (ex-info "shared initialized boundaries must carry the same source object"
                      {:reason :link-composition-share-source :nodes references})))
    (first sources)))

(defn- leaf-groups [value-groups values]
  (mapcat
   (fn [group]
     (let [leaf-vectors (mapv #(get-in values [% :leaves]) group)]
       (apply mapv (fn [& leaves] (mapv :node leaves)) leaf-vectors)))
   value-groups))

(defn- derive-composition
  [id components {:keys [connections shares outputs attributes]
                  :or {connections [] shares [] attributes {}} :as specification}]
  (when (nil? id)
    (throw (ex-info "link composition requires a stable plan identity"
                    {:reason :link-composition-id})))
  (let [components (normalize-components! components)
        component-plans (into {} (map (juxt :id (comp :plan :lowering))) components)
        targets (set (map :target (vals component-plans)))
        _ (when-not (= 1 (count targets))
            (throw (ex-info "all composed LinkPlans must target the same device"
                            {:reason :link-composition-targets :targets targets})))
        target (first targets)
        component-order (zipmap (map :id components) (range))
        connections (normalize-connections! connections component-plans component-order)
        shares (normalize-shares! shares component-plans)
        _ (distinct-groups! connections shares)
        outputs (mapv #(value-ref! component-plans %) outputs)
        _ (when (empty? outputs)
            (throw (ex-info "link composition requires explicit public outputs"
                            {:reason :link-composition-outputs})))
        _ (doseq [[component-id value-id :as output] outputs]
            (when-not (public-output-value? (get component-plans component-id) value-id)
              (throw (ex-info "a composite output must be a declared logical component output"
                              {:reason :link-composition-output-boundary :output output}))))
        node-mapping0
        (into {}
              (mapcat (fn [{component-id :id lowering :lowering}]
                        (map (fn [node-id]
                               [[component-id node-id]
                                (namespace-id id component-id :node node-id)])
                             (keys (get-in lowering [:plan :nodes]))))
                      components))
        nodes0
        (into {}
              (mapcat (fn [{component-id :id lowering :lowering}]
                        (map (fn [[_ node]]
                               (let [node (namespace-node id component-id node)]
                                 [(:id node) node]))
                             (get-in lowering [:plan :nodes])))
                      components))
        value-mapping0
        (into {}
              (mapcat (fn [{component-id :id lowering :lowering}]
                        (map (fn [value-id]
                               [[component-id value-id]
                                (namespace-id id component-id :value value-id)])
                             (keys (get-in lowering [:plan :values]))))
                      components))
        values0
        (into {}
              (mapcat (fn [{component-id :id lowering :lowering}]
                        (map (fn [[_ value]]
                               (let [value (namespace-value
                                            id component-id
                                            #(get node-mapping0 [component-id %]) value)]
                                 [(:id value) value]))
                             (get-in lowering [:plan :values])))
                      components))
        namespaced-value-ref #(get value-mapping0 %)
        connection-pairs (mapv (fn [{:keys [from to]}]
                                 [(namespaced-value-ref from) (namespaced-value-ref to)])
                               connections)
        share-groups (mapv #(mapv namespaced-value-ref %) shares)
        value-groups (concat connection-pairs share-groups)
        _ (doseq [group value-groups]
            (require-equal-contract! :boundary group values0 nodes0)
            (require-single-node-allocations! group values0 nodes0))
        connection-leaf-pairs (vec (leaf-groups connection-pairs values0))
        share-leaf-groups (vec (leaf-groups share-groups values0))
        _ (doseq [group share-leaf-groups] (canonical-source group nodes0))
        node-replacements
        (into {}
              (concat (map (fn [[source target]] [target source]) connection-leaf-pairs)
                      (mapcat (fn [[canonical & others]]
                                (map (fn [other] [other canonical]) others))
                              share-leaf-groups)))
        value-replacements
        (into {}
              (concat (map (fn [[source target]] [target source]) connection-pairs)
                      (mapcat (fn [[canonical & others]]
                                (map (fn [other] [other canonical]) others))
                              share-groups)))
        allocation-replacements
        (into {}
              (map (fn [[discarded canonical]]
                     [(get-in nodes0 [discarded :view :allocation :id])
                      (get-in nodes0 [canonical :view :allocation :id])]))
              node-replacements)
        connected-sources (set (map first connection-leaf-pairs))
        shared-source-by-node
        (into {}
              (mapcat (fn [group]
                        (let [source (canonical-source group nodes0)]
                          (map (fn [node-id] [node-id source]) group)))
                      share-leaf-groups))
        nodes
        (into {}
              (keep (fn [[node-id node]]
                      (when-not (contains? node-replacements node-id)
                        (let [allocation (get-in node [:view :allocation])
                              allocation-id (:id allocation)
                              allocation-id' (get allocation-replacements allocation-id
                                                  allocation-id)
                              node (cond-> (assoc-in node [:view :allocation :id] allocation-id')
                                     (contains? connected-sources node-id)
                                     (assoc :role :internal)
                                     (contains? shared-source-by-node node-id)
                                     (assoc :source (get shared-source-by-node node-id)))]
                          [node-id node]))))
              nodes0)
        resolve-node #(get node-replacements % %)
        resolve-value #(get value-replacements % %)
        values
        (into {}
              (keep (fn [[value-id value]]
                      (when-not (contains? value-replacements value-id)
                        [value-id (update value :leaves
                                          (fn [leaves]
                                            (mapv #(update % :node resolve-node) leaves)))])))
              values0)
        node-mapping (into {} (map (fn [[reference node-id]]
                                     [reference (resolve-node node-id)])) node-mapping0)
        value-mapping (into {} (map (fn [[reference value-id]]
                                      [reference (resolve-value value-id)])) value-mapping0)
        allocation-mapping
        (into {}
              (mapcat (fn [{component-id :id lowering :lowering}]
                        (for [allocation-id (distinct
                                             (map #(get-in % [:view :allocation :id])
                                                  (vals (get-in lowering [:plan :nodes]))))
                              :let [namespaced (namespace-id id component-id :allocation
                                                             allocation-id)]]
                          [[component-id allocation-id]
                           (get allocation-replacements namespaced namespaced)]))
                      components))
        instance-mapping
        (into {}
              (mapcat (fn [{component-id :id lowering :lowering}]
                        (map (fn [instance]
                               [[component-id (:id instance)]
                                (namespace-id id component-id :instance (:id instance))])
                             (get-in lowering [:plan :instances])))
                      components))
        instances
        (vec
         (mapcat (fn [{component-id :id lowering :lowering}]
                   (map (fn [instance]
                          (namespace-instance id component-id
                                              (comp resolve-value
                                                    #(get value-mapping0 [component-id %]))
                                              instance))
                        (get-in lowering [:plan :instances])))
                 components))
        aliases (into #{}
                      (for [[left-id left] nodes
                            [right-id right] nodes
                            :when (neg? (compare (pr-str left-id) (pr-str right-id)))
                            :when (bview/overlaps? (:view left) (:view right))]
                        #{left-id right-id}))
        output-value-ids (mapv value-mapping outputs)
        output-ids (vec (mapcat #(map :node (get-in values [% :leaves])) output-value-ids))
        plan (link-plan/make
              {:id id :target target :nodes nodes :values values
               :instances instances :outputs output-ids
               :aliases aliases
               :attributes (merge attributes
                                  {:lowered-from :certified-link-composition
                                   :component-plan-ids
                                   (mapv (comp :id :plan :lowering) components)})})
        certificate
        (->LinkCompositionCertificate
         :certified-link-plans :link-plan id target
         (mapv (comp :id :plan :lowering) components)
         connections shares value-mapping node-mapping allocation-mapping instance-mapping
         output-ids)]
    {:plan plan :certificate certificate :components components
     :specification (assoc specification :connections connections :shares shares
                           :outputs outputs :attributes attributes)}))

(defn verify!
  "Re-derive a certified composition from its certified components and explicit boundary map."
  [composition]
  (when-not (certified-composition? composition)
    (throw (ex-info "expected a CertifiedLinkComposition"
                    {:reason :link-composition-type :actual (type composition)})))
  (let [actual (select-keys composition [:plan :certificate :components :specification])
        expected (derive-composition (get-in composition [:plan :id])
                                     (:components composition)
                                     (:specification composition))]
    (when-not (certificate? (:certificate composition))
      (throw (ex-info "link composition requires a LinkCompositionCertificate"
                      {:reason :link-composition-certificate-type
                       :actual (type (:certificate composition))})))
    (when-not (= expected actual)
      (throw (ex-info "link composition certificate does not match its source plans and target"
                      {:reason :link-composition-certificate
                       :expected (select-keys expected [:plan :certificate])
                       :actual (select-keys actual [:plan :certificate])})))
    composition))

(defn compose
  "Compose certified LinkPlans before allocation and return a checkable witness.

   `components` is an ordered vector of `{:id component-id :lowering certified-lowering}`.
   `specification` requires ordered `:outputs` and optionally contains:
   - `:connections` — `{:from [producer-id output-value] :to [consumer-id input-value]}`;
   - `:shares` — groups of equal input/constant logical value references;
   - `:attributes` — inspectable composite metadata.

   Composite values unify atomically across their ordered leaves. Endpoints whose individual
   physical allocations have multiple views remain rejected until ranged composition is explicit."
  [{:keys [id components] :as request}]
  (let [specification (select-keys request [:connections :shares :outputs :attributes])
        {:keys [plan certificate components specification]}
        (derive-composition id components specification)]
    (verify! (->CertifiedLinkComposition plan certificate components specification))))
