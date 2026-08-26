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
            connections shares node-mapping allocation-mapping instance-mapping outputs])
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

(defn- node-ref! [component-plans [component-id node-id :as reference]]
  (when-not (and (vector? reference) (= 2 (count reference)))
    (throw (ex-info "a link composition node reference is [component-id node-id]"
                    {:reason :link-composition-node-reference :reference reference})))
  (let [plan (get component-plans component-id)]
    (when-not plan
      (throw (ex-info "a link composition node reference names an absent component"
                      {:reason :link-composition-component-reference
                       :reference reference :components (set (keys component-plans))})))
    (when-not (contains? (:nodes plan) node-id)
      (throw (ex-info "a link composition node reference names an absent component node"
                      {:reason :link-composition-node-reference :reference reference
                       :nodes (set (keys (:nodes plan)))})))
    reference))

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

(defn- namespace-instance [composition-id component-id node-mapping instance]
  (assoc instance
         :id (namespace-id composition-id component-id :instance (:id instance))
         :bindings (update-vals (:bindings instance) node-mapping)))

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

(defn- require-equal-contract! [kind references nodes]
  (let [contracts (mapv (comp endpoint-contract nodes) references)]
    (when-not (apply = contracts)
      (throw (ex-info "composed boundary values have different realized view contracts"
                      {:reason :link-composition-view-contract :kind kind
                       :references references :contracts contracts})))))

(defn- require-single-node-allocations! [references nodes]
  ;; Step 4 generalizes this to ranged/subview boundary unification. Rejecting it here is crucial:
  ;; rebasing one endpoint of a multi-view allocation would silently change the other aliases.
  (let [counts (frequencies (map #(get-in % [:view :allocation :id]) (vals nodes)))]
    (doseq [reference references
            :let [allocation-id (get-in nodes [reference :view :allocation :id])]]
      (when-not (= 1 (get counts allocation-id))
        (throw (ex-info "composition across a multi-view allocation requires ranged-view linking"
                        {:reason :link-composition-ranged-view :node reference
                         :allocation allocation-id :views (get counts allocation-id)}))))))

(defn- normalize-connections! [connections component-plans component-order]
  (let [connections
        (mapv (fn [{:keys [from to] :as connection}]
                (when-not (= #{:from :to} (set (keys connection)))
                  (throw (ex-info "a dataflow connection requires exactly :from and :to"
                                  {:reason :link-composition-connection
                                   :connection connection})))
                {:from (node-ref! component-plans from)
                 :to (node-ref! component-plans to)})
              connections)
        targets (mapv :to connections)]
    (when-not (= (count targets) (count (distinct targets)))
      (throw (ex-info "a composed consumer node has more than one producer"
                      {:reason :link-composition-producers :targets targets})))
    (doseq [{:keys [from to]} connections
            :let [[from-component from-node] from
                  [to-component to-node] to
                  from-plan (get component-plans from-component)
                  to-plan (get component-plans to-component)]]
      (when-not (< (get component-order from-component) (get component-order to-component))
        (throw (ex-info "link dataflow connections must follow component schedule order"
                        {:reason :link-composition-order :from from :to to})))
      (when-not (some #{from-node} (:outputs from-plan))
        (throw (ex-info "a link dataflow producer must be a declared component output"
                        {:reason :link-composition-producer-boundary :from from
                         :outputs (:outputs from-plan)})))
      (when-not (= :input (get-in to-plan [:nodes to-node :role]))
        (throw (ex-info "a link dataflow consumer must be an input boundary"
                        {:reason :link-composition-consumer-role :to to
                         :role (get-in to-plan [:nodes to-node :role])}))))
    connections))

(defn- normalize-shares! [shares component-plans]
  (mapv
   (fn [share]
     (let [references (mapv #(node-ref! component-plans %) share)]
       (when-not (<= 2 (count references))
         (throw (ex-info "a shared boundary group requires at least two nodes"
                         {:reason :link-composition-share :share share})))
       (when-not (= (count references) (count (distinct references)))
         (throw (ex-info "a shared boundary group cannot repeat a node"
                         {:reason :link-composition-share-duplicates :share share})))
       (let [roles (mapv (fn [[component-id node-id]]
                           (get-in component-plans [component-id :nodes node-id :role]))
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
        outputs (mapv #(node-ref! component-plans %) outputs)
        _ (when (empty? outputs)
            (throw (ex-info "link composition requires explicit public outputs"
                            {:reason :link-composition-outputs})))
        _ (doseq [[component-id node-id :as output] outputs]
            (when-not (some #{node-id} (get-in component-plans [component-id :outputs]))
              (throw (ex-info "a composite output must be a declared component output"
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
        namespaced-ref #(get node-mapping0 %)
        connection-pairs (mapv (fn [{:keys [from to]}]
                                 [(namespaced-ref from) (namespaced-ref to)])
                               connections)
        share-groups (mapv #(mapv namespaced-ref %) shares)
        endpoint-groups (concat connection-pairs share-groups)
        _ (doseq [group endpoint-groups]
            (require-equal-contract! :boundary group nodes0)
            (require-single-node-allocations! group nodes0))
        _ (doseq [group share-groups] (canonical-source group nodes0))
        replacements
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
              replacements)
        connected-sources (set (map first connection-pairs))
        shared-source-by-node
        (into {}
              (mapcat (fn [group]
                        (let [source (canonical-source group nodes0)]
                          (map (fn [node-id] [node-id source]) group)))
                      share-groups))
        nodes
        (into {}
              (keep (fn [[node-id node]]
                      (when-not (contains? replacements node-id)
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
        resolve-node #(get replacements % %)
        node-mapping (into {} (map (fn [[reference node-id]]
                                     [reference (resolve-node node-id)])) node-mapping0)
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
                                              (comp resolve-node
                                                    #(get node-mapping0 [component-id %]))
                                              instance))
                        (get-in lowering [:plan :instances])))
                 components))
        aliases (into #{}
                      (for [[left-id left] nodes
                            [right-id right] nodes
                            :when (neg? (compare (pr-str left-id) (pr-str right-id)))
                            :when (bview/overlaps? (:view left) (:view right))]
                        #{left-id right-id}))
        output-ids (mapv node-mapping outputs)
        plan (link-plan/make
              {:id id :target target :nodes nodes :instances instances :outputs output-ids
               :aliases aliases
               :attributes (merge attributes
                                  {:lowered-from :certified-link-composition
                                   :component-plan-ids
                                   (mapv (comp :id :plan :lowering) components)})})
        certificate
        (->LinkCompositionCertificate
         :certified-link-plans :link-plan id target
         (mapv (comp :id :plan :lowering) components)
         connections shares node-mapping allocation-mapping instance-mapping output-ids)]
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
   - `:connections` — `{:from [producer-id output-node] :to [consumer-id input-node]}`;
   - `:shares` — groups of equal input/constant boundary node references;
   - `:attributes` — inspectable composite metadata.

   This first contract intentionally rejects endpoints whose allocation has multiple views. The
   following AbstractValue/view consolidation pass will make ranged boundary composition explicit."
  [{:keys [id components] :as request}]
  (let [specification (select-keys request [:connections :shares :outputs :attributes])
        {:keys [plan certificate components specification]}
        (derive-composition id components specification)]
    (verify! (->CertifiedLinkComposition plan certificate components specification))))
