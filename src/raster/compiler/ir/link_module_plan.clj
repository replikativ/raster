(ns raster.compiler.ir.link-module-plan
  "Target-neutral, validation-only named entry contracts over one resident LinkPlan spine.

   This prerequisite proves entry partitioning, resident entry interfaces, and cross-entry
   initialized-storage frontiers. It does not yet execute entries or define sessions, leases,
   replay, lifetime, or failure-completion authority."
  (:require [clojure.set :as set]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.link-plan :as link]))

(defrecord LinkModuleEntry [name instances contract])
(defrecord LinkModulePlan [id target spine entries frontiers retained-groups attributes])

(defn module-entry? [candidate]
  (and candidate (= "raster.compiler.ir.link_module_plan.LinkModuleEntry"
                    (.getName (class candidate)))))

(defn module-plan? [candidate]
  (and candidate (= "raster.compiler.ir.link_module_plan.LinkModulePlan"
                    (.getName (class candidate)))))

(defn- fail! [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :link-module-plan))))

(def ^:private interface-keys #{:inputs :state :outputs})

(defn- normalize-interface [entry]
  (let [interface (:interface entry)]
    (when-not (and (map? interface) (= interface-keys (set (keys interface))))
      (fail! :link-module-entry-interface
             "a named entry requires exactly inputs, state, and outputs interface vectors"
             {:entry (:name entry) :interface interface}))
    (doseq [field interface-keys
            :let [values (get interface field)]]
      (when-not (and (vector? values) (= (count values) (count (distinct values))))
        (fail! :link-module-entry-interface
               "entry interface categories must be duplicate-free vectors"
               {:entry (:name entry) :field field :values values})))
    (let [memberships (map (fn [field] [field (set (get interface field))]) interface-keys)]
      (doseq [[[left left-values] [right right-values]]
              (for [left memberships right memberships
                    :when (neg? (compare (name (first left)) (name (first right))))]
                [left right])
              :let [overlap (set/intersection left-values right-values)]
              :when (seq overlap)]
        (fail! :link-module-entry-interface-overlap
               "one logical value cannot occupy two public entry interface categories"
               {:entry (:name entry) :left left :right right :values overlap})))
    (select-keys interface [:inputs :state :outputs])))

(defn- normalize-entry [entry]
  (let [{:keys [name instances]} entry
        interface (normalize-interface entry)]
    (when-not (and (keyword? name)
                   (vector? instances) (seq instances)
                   (= (count instances) (count (distinct instances))))
      (fail! :link-module-entry
             "a named entry requires a keyword name and non-empty unique instance vector"
             {:entry entry}))
    (assoc (->LinkModuleEntry name instances nil) :interface interface)))

(defn- node-values [spine]
  (into {}
        (mapcat (fn [[value-id value]]
                  (map (fn [{:keys [node]}] [node value-id]) (:leaves value))))
        (:values spine)))

(defn- value-covered? [spine produced node-ids]
  (let [views (mapv #(get-in spine [:nodes % :view]) produced)]
    (every? #(bview/covered-contiguous? (get-in spine [:nodes % :view]) views) node-ids)))

(defn- writes-retained-value? [spine writes node-ids]
  (let [targets (mapv #(get-in spine [:nodes % :view]) node-ids)]
    (boolean
     (some (fn [written]
             (let [view (get-in spine [:nodes written :view])]
               (some #(bview/overlaps? view %) targets)))
           writes))))

(defn- external-requirement? [spine node-id]
  (let [{:keys [role source]} (get-in spine [:nodes node-id])]
    (or source (contains? #{:input :constant :state} role))))

(defn- view-allocation-key [view]
  [(get-in view [:allocation :device]) (get-in view [:allocation :id])])

(defn- indexed-access-views [spine node-ids]
  (group-by (comp view-allocation-key :view)
            (map #(get-in spine [:nodes %]) node-ids)))

(defn- overlapping-access-nodes [spine indexed leaf-node-ids]
  (into #{}
        (mapcat
         (fn [leaf-node-id]
           (let [leaf-view (get-in spine [:nodes leaf-node-id :view])]
             (keep (fn [{:keys [id view]}]
                     (when (bview/overlaps? leaf-view view) id))
                   (get indexed (view-allocation-key leaf-view)))))
        leaf-node-ids)))

(defn- value-roles [spine value-id]
  (into #{} (map #(get-in spine [:nodes (:node %) :role]))
        (get-in spine [:values value-id :leaves])))

(defn- reject-cross-value-access! [node->value entry value-id access-nodes]
  (let [other-values (into #{} (keep #(let [owner (node->value %)]
                                        (when (not= owner value-id) owner))) access-nodes)]
    (when (seq other-values)
      (fail! :link-module-interface-alias
             "a public entry value cannot overlap access through another logical value"
             {:entry (:name entry) :value value-id :aliased-values other-values
              :access-nodes access-nodes}))))

(defn- derive-access-interface!
  [spine entries entry-index entry]
  (let [node->value (node-values spine)
        contract (:contract entry)
        read-index (indexed-access-views spine (:reads contract))
        write-index (indexed-access-views spine (:writes contract))
        access-index (indexed-access-views spine (into (:reads contract) (:writes contract)))
        earlier-write-index
        (indexed-access-views spine
                              (into #{} (mapcat #(get-in % [:contract :writes]))
                                    (take entry-index entries)))
        derive
        (fn [role index]
          (into #{}
                (keep
                 (fn [[value-id value]]
                   (let [roles (value-roles spine value-id)]
                     (when (contains? roles role)
                       (let [leaves (mapv :node (:leaves value))
                             overlap (overlapping-access-nodes spine index leaves)]
                         (when (seq overlap)
                           (when-not (= #{role} roles)
                             (fail! :link-module-interface-value-roles
                                    "one public entry value requires one matching role across its leaves"
                                    {:entry (:name entry) :value value-id
                                     :expected role :roles roles}))
                           (when-not (value-covered? spine overlap leaves)
                             (fail! :link-module-interface-whole-value
                                    "a public entry interface cannot expose a partial logical value"
                                    {:entry (:name entry) :value value-id :role role
                                     :access-nodes overlap :leaves leaves}))
                           (let [all-overlap (overlapping-access-nodes spine access-index leaves)]
                             (reject-cross-value-access!
                              node->value entry value-id all-overlap))
                           value-id)))))
                (:values spine))))
        inputs (derive :input read-index)
        states (derive :state access-index)]
    (doseq [[value-id value] (:values spine)
            :when (contains? (value-roles spine value-id) :input)
            :let [leaves (mapv :node (:leaves value))
                  writes (overlapping-access-nodes spine write-index leaves)]
            :when (seq writes)]
      (fail! :link-module-interface-input-write
             "a public entry input cannot overlap storage written by the same entry"
             {:entry (:name entry) :value value-id :write-nodes writes}))
    (doseq [value-id inputs
            :let [leaves (mapv :node (get-in spine [:values value-id :leaves]))
                  writers (overlapping-access-nodes spine earlier-write-index leaves)]
            :when (seq writers)]
      (fail! :link-module-interface-input-producer
             "a public entry input cannot be initialized by an earlier module entry"
             {:entry (:name entry) :value value-id :writer-nodes writers}))
    {:inputs inputs :state states}))

(defn- values-overlap? [spine left-id right-id]
  (let [left (map #(get-in spine [:nodes (:node %) :view])
                  (get-in spine [:values left-id :leaves]))
        right (map #(get-in spine [:nodes (:node %) :view])
                   (get-in spine [:values right-id :leaves]))]
    (boolean (some (fn [left-view] (some #(bview/overlaps? left-view %) right)) left))))

(defn- validate-entry-interface!
  [spine entries entry-index entry]
  (let [interface (:interface entry)
        derived (derive-access-interface! spine entries entry-index entry)
        node->value (node-values spine)
        access-index (indexed-access-views
                      spine (into (get-in entry [:contract :reads])
                                  (get-in entry [:contract :writes])))
        available-values (set (keys (:values spine)))]
    (doseq [field [:inputs :state]
            :let [declared (set (get interface field))
                  expected (get derived field)]
            :when (not= declared expected)]
      (fail! :link-module-interface-access
             "entry input and state interfaces must exactly match resident physical access"
             {:entry (:name entry) :field field :declared declared :expected expected
              :missing (set/difference expected declared)
              :extra (set/difference declared expected)}))
    (doseq [value-id (:outputs interface)]
      (when-not (contains? available-values value-id)
        (fail! :link-module-interface-value
               "entry interface names an absent logical LinkValue"
               {:entry (:name entry) :field :outputs :value value-id
                :values available-values}))
      (when-not (= #{:output} (value-roles spine value-id))
        (fail! :link-module-interface-output-role
               "an entry output requires output-role physical storage"
               {:entry (:name entry) :value value-id :roles (value-roles spine value-id)}))
      (let [leaves (mapv :node (get-in spine [:values value-id :leaves]))]
        (when-not (value-covered? spine (get-in entry [:contract :full-produces]) leaves)
          (fail! :link-module-interface-output-production
                 "an entry output must be semantically complete-produced by that entry"
                 {:entry (:name entry) :value value-id :leaves leaves
                  :full-produces (get-in entry [:contract :full-produces])}))
        (reject-cross-value-access!
         node->value entry value-id
         (overlapping-access-nodes spine access-index leaves))))
    (let [categories (for [field [:inputs :state :outputs]
                           value-id (get interface field)]
                       [field value-id])]
      (doseq [[[left-field left-id] [right-field right-id]]
              (for [[left-index left] (map-indexed vector categories)
                    [right-index right] (map-indexed vector categories)
                    :when (< left-index right-index)]
                [left right])
              :when (and (not= left-id right-id)
                         (values-overlap? spine left-id right-id))]
        (fail! :link-module-interface-alias
               "distinct public entry values cannot overlap physical storage"
               {:entry (:name entry) :left-field left-field :left left-id
                :right-field right-field :right right-id})))
    entry))

(defn- derive-frontiers [spine entries]
  (let [node->value (node-values spine)
        entry-names (mapv :name entries)]
    (reduce
     (fn [state [consumer-index entry]]
       (let [required-nodes (get-in entry [:contract :frontier-requires])
             required-order (get-in entry [:contract :frontier-require-order])
             value-order (distinct (map node->value required-order))]
         (reduce
          (fn [state value-id]
            (let [required (filterv #(= value-id (node->value %)) required-nodes)]
            (when-not value-id
              (fail! :link-module-frontier-value
                     "a cross-entry requirement has no owning logical value"
                     {:entry (:name entry) :nodes (set required)}))
            (let [leaves (mapv :node (get-in spine [:values value-id :leaves]))
                  required (set required)]
              (when-not (= required (set leaves))
                (fail! :link-module-frontier-whole-value
                       "cross-entry frontiers must retain one whole logical value"
                       {:entry (:name entry) :value value-id
                        :required required :leaves (set leaves)}))
              (let [producers
                    (into []
                          (keep-indexed
                           (fn [producer-index producer]
                             (when (and (< producer-index consumer-index)
                                        (value-covered? spine
                                                        (get-in producer [:contract :full-produces])
                                                        leaves))
                               producer-index)))
                          entries)]
                (when-not (= 1 (count producers))
                  (fail! :link-module-frontier-producer
                         "one whole logical frontier requires one unique earlier complete producer"
                         {:entry (:name entry) :value value-id
                          :producer-entries (mapv entry-names producers)}))
                (let [writers (into []
                                    (keep-indexed
                                     (fn [entry-index candidate]
                                       (when (writes-retained-value?
                                              spine (get-in candidate [:contract :writes]) leaves)
                                         entry-index)))
                                    entries)]
                  (when-not (= writers producers)
                    (fail! :link-module-frontier-writers
                           "a retained logical frontier requires its complete producer to be its only writer entry"
                           {:entry (:name entry) :value value-id
                            :producer-entries (mapv entry-names producers)
                            :writer-entries (mapv entry-names writers)})))
                (doseq [node-id leaves]
                  (when-not (= :owned (get-in spine [:nodes node-id :view :allocation :ownership]))
                    (fail! :link-module-frontier-ownership
                           "a retained cross-entry frontier requires module-owned physical storage"
                           {:entry (:name entry) :value value-id :node node-id
                            :ownership (get-in spine
                                               [:nodes node-id :view :allocation :ownership])})))
                (let [producer-name (nth entry-names (first producers))
                      key [producer-name value-id]
                      frontier {:value value-id :nodes leaves :producer producer-name
                                :consumer (:name entry)}]
                  (-> state
                      (update :frontiers conj frontier)
                      (update-in [:groups key]
                                 (fn [prior]
                                   (if prior
                                     (update prior :consumers conj (:name entry))
                                     {:value value-id :nodes leaves :producer producer-name
                                      :consumers #{(:name entry)}})))))))))
          state value-order)))
     {:frontiers [] :groups {}}
     (map-indexed vector entries))))

(defn- derive-for-request!
  "Freshly validate a named-entry module and rederive every conditional effect frontier.

   The spine is a canonical LinkPlan with no global outputs. Entries partition its instances in
   order. Stored contracts/frontiers are never accepted as proof. Host source contents and native
   handles remain live objects in the spine and are neither fingerprinted nor serialized here."
  [module caller-options verify-derived?]
  (when-not (module-plan? module)
    (fail! :link-module-type "expected a LinkModulePlan" {:actual (type module)}))
  (let [{:keys [id target spine entries frontiers retained-groups attributes]} module]
    (when (nil? id)
      (fail! :link-module-id "a link module requires a nominal identity" {}))
    (when-not (map? attributes)
      (fail! :link-module-attributes "link module attributes must be a map"
             {:attributes attributes}))
    (when-not (and (link/link-plan? spine) (empty? (:outputs spine)))
      (fail! :link-module-spine
             "a link module requires one canonical LinkPlan spine with no global outputs"
             {:actual (type spine) :outputs (:outputs spine)}))
    (let [supplied-entries entries
          entries (mapv normalize-entry entries)
          names (mapv :name entries)]
      (when-not (and (seq entries) (= (count names) (count (distinct names))))
        (fail! :link-module-entry-names
               "link module entry names must be non-empty and unique" {:names names}))
      ;; This is intentionally fresh on every validation. The sealed evidence never becomes a
      ;; public module authority and cannot be supplied by the caller.
      (let [{checked-spine :plan evidence :effect-evidence}
            (link/validate-with-effect-evidence! spine nil caller-options)
            contracts (link/relative-entry-contracts!
                       checked-spine evidence (mapv :instances entries) caller-options)
            contracts
            (mapv (fn [contract]
                    (let [external (into #{} (filter #(external-requirement?
                                                       checked-spine %)) (:requires contract))]
                      (assoc contract
                             :external-requires external
                             :frontier-requires (set (remove external (:requires contract)))
                             :frontier-require-order
                             (vec (remove external (:require-order contract))))))
                  contracts)
            entries (mapv #(assoc %1 :contract %2) entries contracts)
            entries (mapv #(validate-entry-interface! checked-spine entries %1 %2)
                          (range) entries)
            {derived-frontiers :frontiers groups :groups}
            (derive-frontiers checked-spine entries)
            group-order (distinct (map (juxt :producer :value) derived-frontiers))
            derived-groups (mapv groups group-order)
            derived (->LinkModulePlan id (:target checked-spine) checked-spine entries
                                      derived-frontiers derived-groups attributes)]
        (when verify-derived?
          (when-not (= target (:target checked-spine))
            (fail! :link-module-derived-target
                   "stored module target differs from the freshly validated spine"
                   {:stored target :derived (:target checked-spine)}))
          (when-not (= (mapv :contract supplied-entries) (mapv :contract entries))
            (fail! :link-module-derived-contracts
                   "stored entry contracts differ from fresh relative effect analysis" {}))
          (when-not (= frontiers derived-frontiers)
            (fail! :link-module-derived-frontiers
                   "stored cross-entry frontiers differ from fresh analysis" {}))
          (when-not (= retained-groups derived-groups)
            (fail! :link-module-derived-retained-groups
                   "stored retained groups differ from fresh analysis" {})))
        derived))))

(defn validate!
  "Freshly validate a named-entry module under independent caller math intent."
  ([module] (derive-for-request! module nil true))
  ([module caller-options] (derive-for-request! module caller-options true)))

(defn make
  "Construct and freshly validate a target-neutral named-entry module contract."
  ([{:keys [id spine entries attributes] :or {attributes {}}}]
   (derive-for-request! (->LinkModulePlan id (:target spine) spine entries nil nil attributes)
                        nil false))
  ([{:keys [id spine entries attributes] :or {attributes {}}} caller-options]
   (derive-for-request! (->LinkModulePlan id (:target spine) spine entries nil nil attributes)
                        caller-options false)))
