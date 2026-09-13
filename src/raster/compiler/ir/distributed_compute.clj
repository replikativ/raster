(ns raster.compiler.ir.distributed-compute
  "Structural compute bindings, reusing LinkPlan ABI and physical view contracts.
   This namespace creates no runtime resources or distributed executor."
  (:require [clojure.set :as set]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.validate :refer [fail!]]))

(defn- required-values [plan accesses]
  (into (set (link/output-value-ids plan))
        (keep (fn [[id value]]
                (let [nodes (map #(get-in plan [:nodes (:node %)]) (:leaves value))
                      role (:role (first nodes))]
                  (when (and (contains? accesses id)
                             (or (contains? #{:input :state} role)
                                 (and (= :constant role) (not-every? :source nodes))))
                    id))))
        (:values plan)))

(defn- entry-facts [device id entry]
  (when-not (and (some? id) (map? entry) (= #{:link-plan} (set (keys entry))))
    (fail! "a named compute entry owns one LinkPlan"
           :distributed-compute-entry {:device device :entry id}))
  (let [plan (:link-plan entry)
        accesses (link/value-accesses plan)]
    (when-not (= device (:target plan))
      (fail! "compute entry target differs from its mesh device"
             :distributed-compute-entry-device {:device device :entry id}))
    {:link-plan plan :accesses accesses :required (required-values plan accesses)}))

(defn- normalize-reference [reference]
  (cond
    (and (map? reference) (= #{:value :shard} (set (keys reference)))) reference

    (and (map? reference) (= #{:local-shape :placements} (set (keys reference))))
    (let [placements (:placements reference)
          owned (when (vector? placements) (first placements))]
      (when-not (and (vector? placements) (= 1 (count placements))
                     (map? owned) (= :owned (:kind owned))
                     (= #{:kind :value :shard :local-offsets} (set (keys owned))))
        (fail! "this materialization requires one owned placement; replica/boundary proofs are not yet implemented"
               :distributed-compute-placement-coverage {:reference reference}))
      (assoc (select-keys owned [:value :shard])
             :local-shape (:local-shape reference) :local-offsets (:local-offsets owned)
             :explicit-domain? true))

    :else
    (fail! "binding must name a qualified shard or an explicit local materialization"
           :distributed-compute-shard-reference {:reference reference})))

(defn- project-domain [local leaves shape]
  (let [abstract (:abstract local)
        logical-shape (:shape abstract)
        layout (:logical-layout abstract)
        base (:view (first leaves))
        concrete? #(and (vector? %) (seq %) (every? pos-int? %))]
    (when-not (and (= :tensor (:kind abstract)) (concrete? shape) (concrete? logical-shape)
                   (= 1 (count leaves))
                   (= {:kind :plain} (:representation abstract))
                   (= {:kind :dense} (:physical-layout local))
                   (or (nil? layout) (= {:order :row-major} layout)
                       (= {:strides (view/dense-strides logical-shape)} layout))
                   (view/contiguous? base)
                   (= (dtype/canon (:dtype abstract)) (:dtype base))
                   (= (reduce *' 1 shape) (reduce *' 1 logical-shape)
                      (reduce *' 1 (:shape base))))
      (fail! "local coordinates require one plain dense leaf with a proven equal-volume layout"
             :distributed-compute-local-domain
             {:value (:id local) :local-shape shape :abstract abstract
              :physical-layout (:physical-layout local)}))
    (view/subview base {:shape shape})))

(defn- bind-values [{plan :link-plan :keys [accesses required]} device globals shards bindings]
  (when-not (map? bindings)
    (fail! "compute bindings must map local values to qualified shard references"
           :distributed-compute-bindings {:bindings bindings}))
  (let [bindings (update-vals bindings normalize-reference)
        supplied (set (keys bindings))
        available (set/union (set (keys accesses)) (set (link/output-value-ids plan)))]
    (when-not (set/subset? required supplied)
      (fail! "compute binding omits a public local value"
             :distributed-compute-missing-values {:missing (set/difference required supplied)}))
    (when-not (set/subset? supplied available)
      (fail! "compute binding names an absent or unused local value"
             :distributed-compute-local-value {:unknown (set/difference supplied available)}))
    (when-not (= (count bindings)
                 (count (distinct (map #(select-keys % [:value :shard]) (vals bindings)))))
      (fail! "one compute call gives one shard distinct local identities"
             :distributed-compute-duplicate-shard {:bindings bindings}))
    ;; LinkPlan validates declared aliases locally, but logical ABI access facts do not
    ;; propagate through them. Do not hide private mutations behind a bound read-only value.
    (let [bound-nodes (into #{} (mapcat #(map :node (:leaves (get-in plan [:values %]))))
                            supplied)]
      (doseq [bound-node bound-nodes
              [private-node node] (:nodes plan)
              :when (not (contains? bound-nodes private-node))]
        (when (view/overlaps? (get-in plan [:nodes bound-node :view]) (:view node))
          (fail! "bound shard storage cannot alias private entry storage without an access proof"
                 :distributed-compute-private-alias
                 {:bound-node bound-node :private-node private-node}))))
    (into {}
          (map (fn [[id reference]]
                 (let [{:keys [value shard]} reference
                       global (get globals value)
                       candidate (first (filter #(= shard (:id %)) (get shards value)))
                       local (get-in plan [:values id])
                       leaves (mapv (fn [{:keys [name node]}]
                                      {:name name :view (get-in plan [:nodes node :view])})
                                    (:leaves local))]
                   (when-not (and global candidate)
                     (fail! "compute binding names an absent value shard"
                            :distributed-compute-shard {:reference reference}))
                   (when-not (= device (:device candidate))
                     (fail! "compute binding names a shard on another device"
                            :distributed-compute-shard-device {:device device :shard candidate}))
                   (when-not (and (= (:shape candidate) (if (:explicit-domain? reference)
                                                        (:local-shape reference)
                                                        (get-in local [:abstract :shape])))
                                  (av/storage-contract-compatible? global (:abstract local)))
                     (fail! "local value does not realize the shard's logical storage contract"
                            :distributed-compute-value-contract
                            {:reference reference :local (:abstract local)
                             :global global :shape (:shape candidate)}))
                   (when (and (:memory-space global)
                              (not-every? #(= (:memory-space global)
                                              (get-in % [:view :allocation :memory-space])) leaves))
                     (fail! "local storage violates the global value's memory-space constraint"
                            :distributed-compute-memory-space {:reference reference}))
                   (let [domain (when (:explicit-domain? reference)
                                  (project-domain local leaves (:local-shape reference)))
                         owned (when domain
                                 (view/rectangular-subview domain
                                                          {:offsets (:local-offsets reference)
                                                           :shape (:shape candidate)}))]
                     [id (cond-> {:value value :shard shard :access (get accesses id)
                                  :physical-layout (:physical-layout local) :leaves leaves}
                           domain (assoc :domain {:shape (:local-shape reference) :view domain
                                                  :placements [{:kind :owned :value value :shard shard
                                                                :view owned}]}))])))
               bindings))))

(defn- owned-realization [{:keys [physical-layout leaves domain]}]
  (let [owned (some #(when (= :owned (:kind %)) (:view %)) (:placements domain))
        leaves (if owned [(assoc (first leaves) :view owned)] leaves)]
    {:physical-layout physical-layout
     :leaves (mapv (fn [leaf]
                     (update leaf :view
                             (fn [v]
                               ;; Logical shape agreement is checked when binding the shard.
                               ;; Dense ABI reshapes do not change its ordered physical cells.
                               ;; Noncontiguous views must retain their coordinate mapping.
                               (cond-> (dissoc v :id)
                                 (view/contiguous? v) (dissoc :shape :strides)))))
                   leaves)}))

(defn bindings
  "Derive bindings after the enclosing DistributedPlan validates its DAG and shards.
   Unbound analytical compute is explicit. Fully bound compute is not a distributed executor:
   device-scoped allocation, transfer realization, events and arena ownership remain runtime
   obligations. Full owned-domain coverage is required, with optional explicit dense
   reinterpretation; halo subregions still require coverage and provenance proofs."
  [{:keys [device-plans steps values shards]}]
  (let [step-by-id (into {} (map (juxt :id identity)) steps)
        bound
        (reduce-kv
         (fn [bound device local]
           (when-not (set/subset? (set (keys local))
                                  #{:entries :steps :link-plan :execution-plan :attributes})
             (fail! "unknown device-local compute contract keys"
                    :distributed-compute-device-keys {:device device :keys (set (keys local))}))
           (let [entries (get local :entries {}) calls (get local :steps {})]
             (when-not (and (map? entries) (map? calls))
               (fail! "device compute entries and calls must be maps"
                      :distributed-compute-entries {:device device}))
             (when (and (or (seq entries) (seq calls))
                        (or (:link-plan local) (:execution-plan local)))
               (fail! "analytical plans cannot also declare named compute entries"
                      :distributed-compute-mixed-plans {:device device}))
             (let [entries (into {} (map (fn [[id entry]]
                                          [id (entry-facts device id entry)])) entries)]
             (reduce-kv
              (fn [bound id call]
                (when-not (and (map? call) (= #{:entry :bindings} (set (keys call))))
                  (fail! "compute call requires a named entry and logical bindings"
                         :distributed-compute-call {:step id :call call}))
                (let [entry (:entry call) step (get step-by-id id)
                      plan (get-in entries [entry :link-plan])]
                  (when-not (and (= :compute (:kind step)) (= device (:device step)))
                    (fail! "local call does not name a compute step on this device"
                           :distributed-compute-step {:device device :step id}))
                  (when-not plan
                    (fail! "compute call names an absent local entry"
                           :distributed-compute-entry {:device device :entry entry}))
                  (assoc bound id {:entry entry :link-plan plan
                                   :values (bind-values (get entries entry) device values shards
                                                        (:bindings call))})))
              bound calls))))
         {} device-plans)
        realized (volatile! {})]
    ;; Distinct entry points cannot silently assign one resident shard different storage.
    ;; IDs of views may differ; allocation identity, ranges and ordered field packing may not.
    (doseq [[id entry] bound
            [_ {:keys [value shard leaves] :as binding}] (:values entry)]
      (let [key [(:device (get step-by-id id)) value shard]
            realization (owned-realization binding)]
        (when (and (contains? @realized key)
                   (not= (:realization (get @realized key)) realization))
          (fail! "compute entries disagree on a resident shard's physical realization"
                 :distributed-compute-shard-storage {:step id :shard key}))
        (doseq [[other-key other] @realized
                :when (and (not= key other-key) (= (first key) (first other-key)))
                left leaves right (:leaves other)]
          (when (view/overlaps? (:view left) (:view right))
            (fail! "distinct distributed shards cannot alias physical storage without a relation"
                   :distributed-compute-shard-alias {:shard key :other other-key})))
        (vswap! realized assoc key {:realization realization :leaves leaves})))
    {:bindings bound
     :unbound (mapv :id (filter #(and (= :compute (:kind %))
                                     (not (contains? bound (:id %)))) steps))}))
