(ns raster.compiler.ir.link-module-plan
  "Target-neutral, validation-only named entry contracts over one resident LinkPlan spine.

   This prerequisite proves entry partitioning and cross-entry initialized-storage frontiers. It
   does not yet define per-entry callable outputs, sessions, execution, leases, replay, lifetime,
   or failure-completion authority."
  (:require [raster.compiler.ir.buffer-view :as bview]
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

(defn- normalize-entry [entry]
  (let [{:keys [name instances]} entry]
    (when-not (and (keyword? name)
                   (vector? instances) (seq instances)
                   (= (count instances) (count (distinct instances))))
      (fail! :link-module-entry
             "a named entry requires a keyword name and non-empty unique instance vector"
             {:entry entry}))
    (->LinkModuleEntry name instances nil)))

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
