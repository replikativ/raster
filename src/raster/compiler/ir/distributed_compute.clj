(ns raster.compiler.ir.distributed-compute
  "Structural compute bindings, reusing LinkPlan ABI and physical view contracts.
   This namespace creates no runtime resources or distributed executor."
  (:require [clojure.set :as set]
            [raster.compiler.ir.abstract-value :as av]
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

(defn- bind-values [{plan :link-plan :keys [accesses required]} device globals shards bindings]
  (when-not (map? bindings)
    (fail! "compute bindings must map local values to qualified shard references"
           :distributed-compute-bindings {:bindings bindings}))
  (let [supplied (set (keys bindings))
        available (set/union (set (keys accesses)) (set (link/output-value-ids plan)))]
    (when-not (set/subset? required supplied)
      (fail! "compute binding omits a public local value"
             :distributed-compute-missing-values {:missing (set/difference required supplied)}))
    (when-not (set/subset? supplied available)
      (fail! "compute binding names an absent or unused local value"
             :distributed-compute-local-value {:unknown (set/difference supplied available)}))
    (when-not (= (count bindings) (count (distinct (vals bindings))))
      (fail! "one compute call gives one shard distinct local identities"
             :distributed-compute-duplicate-shard {:bindings bindings}))
    (into {}
          (map (fn [[id reference]]
                 (when-not (and (map? reference) (= #{:value :shard} (set (keys reference))))
                   (fail! "shard identity must be qualified by its global value"
                          :distributed-compute-shard-reference {:reference reference}))
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
                   (when-not (and (= (:shape candidate) (get-in local [:abstract :shape]))
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
                   [id {:value value :shard shard :access (get accesses id)
                        :physical-layout (:physical-layout local)
                        :leaves leaves}]))
               bindings))))

(defn bindings
  "Derive bindings after the enclosing DistributedPlan validates its DAG and shards.
   Unbound analytical compute is explicit. Fully bound compute is not a distributed executor:
   device-scoped allocation, transfer realization, events and arena ownership remain runtime
   obligations. Whole-shard shapes are required; halo subregions need an explicit view relation."
  [{:keys [device-plans steps values shards]}]
  (let [step-by-id (into {} (map (juxt :id identity)) steps)
        bound
        (reduce-kv
         (fn [bound device local]
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
            [_ {:keys [value shard physical-layout leaves]}] (:values entry)]
      (let [key [(:device (get step-by-id id)) value shard]
            realization {:physical-layout physical-layout
                         :leaves (mapv #(update % :view dissoc :id) leaves)}]
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
