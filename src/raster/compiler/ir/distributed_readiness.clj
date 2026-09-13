(ns raster.compiler.ir.distributed-readiness
  "Conditional physical initialization and producer freshness over a validated distributed DAG.
   No runtime resources are created and declared sources are not treated as immutable snapshots."
  (:require [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.distributed-compute :as compute]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.validate :refer [fail!]]))

(defn- allocation-key [v]
  [(get-in v [:allocation :device]) (get-in v [:allocation :id])])

(defn- dense! [v]
  (when-not (view/contiguous? v)
    (fail! "readiness needs a contiguous physical region projection"
           :distributed-readiness-layout {:view (:id v)}))
  v)

(defn- node-views [plan ids]
  (mapv #(dense! (get-in plan [:nodes % :view])) (sort-by pr-str ids)))

(defn- compute-action [step entry]
  (let [plan (:link-plan entry)
        contract (link/initialization-contract plan)
        ;; Declared initializers are conditional local preconditions too. Until local analysis
        ;; distinguishes used initializer subregions, retain their complete scopes conservatively.
        requires (node-views plan (into (:requires contract) (:initializers contract)))]
    {:id (:id step) :kind :compute
     :requires requires :reads (node-views plan (:reads contract))
     :writes (node-views plan (:writes contract))
     :produces (node-views plan (:produces contract))
     :fresh (vec (for [[_ binding] (:values entry)
                       placement (get-in binding [:domain :placements])
                       :when (and (contains? #{:replica :boundary} (:kind placement))
                                  (some #(view/overlaps? % (:view placement)) requires))]
                   {:view (dense! (:view placement))
                    :producer (case (:kind placement)
                                :replica (:transfer placement)
                                :boundary (get-in placement [:provider :step]))}))
     :initializers (mapv (fn [id]
                          {:id [(:device step) (:entry entry) id]
                           :view (dense! (get-in plan [:nodes id :view]))
                           :source (get-in plan [:nodes id :source])})
                        (sort-by pr-str (:initializers contract)))}))

(defn- initialization-facts [initializers]
  (reduce
   (fn [accepted {:keys [view source] :as candidate}]
     (let [overlapping (filter #(view/overlaps? view (:view %)) accepted)]
       (when (some #(not (and (= (dissoc view :id) (dissoc (:view %) :id))
                              (identical? source (:source %)))) overlapping)
         (fail! "overlapping startup initializers need one unambiguous source realization"
                :distributed-readiness-initializers {:initializer (:id candidate)}))
       (if (seq overlapping) accepted (conj accepted candidate)))) [] initializers))

(defn- check-races [history ancestors action]
  (reduce
   (fn [history [v access]]
     (let [key (allocation-key v)]
       (doseq [{other :step previous :view previous-access :access} (get history key)
               :when (and (not= other (:id action))
                          (or (= :write access) (= :write previous-access))
                          (view/overlaps? v previous)
                          (not (contains? ancestors other)))]
         (fail! "conflicting physical effects must be ordered by the distributed DAG"
                :distributed-readiness-race {:step (:id action) :other other :allocation key}))
       (update history key (fnil conj []) {:step (:id action) :view v :access access})))
   ;; Preconditions also observe storage: pass-through outputs need no ABI load, but
   ;; their producer must happen before this action, not merely earlier in this traversal.
   history (concat (map #(vector % :read)
                        (distinct (concat (:reads action) (:requires action)
                                          (map :view (:fresh action)))))
                   (map #(vector % :write) (:writes action)))))

(defn- require-covered! [facts v producer step]
  (let [covers (map :view (if (some? producer) (filter #(= producer (:producer %)) facts) facts))]
    (when-not (view/covered-contiguous? v covers)
      (fail! (if (some? producer) "required producer's region is absent or stale"
                 "physical read has no complete initialization evidence")
             (if (some? producer) :distributed-readiness-stale :distributed-readiness-uninitialized)
             {:step step :view (:id v) :producer producer}))))

(defn check
  "Check a structurally validated plan, conditional on startup sources and successful events.
   Every compute must bind a LinkPlan and every transfer must resolve exact copy endpoints.
   All declared initializers must be realized once before the DAG, without later implicit
   reuploads. Their source objects remain caller-owned mutable obligations, not certificates.
   Returned actions and source requirements are not a runnable executable: allocation sharing,
   ownership/lifetimes, runtime input gates and actual transport/event completion still need proof."
  [{:keys [steps] :as plan}]
  (let [{:keys [bindings unbound]} (compute/bindings plan)
        _ (when (seq unbound)
            (fail! "readiness requires all compute steps to be bound"
                   :distributed-readiness-unbound {:steps unbound}))
        endpoints (compute/transfer-bindings plan)
        actions (mapv (fn [step]
                        (case (:kind step)
                          :compute (compute-action step (get bindings (:id step)))
                          :transfer (let [{:keys [source target]} (get endpoints (:id step))]
                                      {:id (:id step) :kind :transfer
                                       :requires [(:view source)] :reads [(:view source)]
                                       :writes [(:view target)] :produces [(:view target)]}))) steps)
        views (mapcat #(concat (:requires %) (:reads %) (:writes %) (:produces %)
                               (map :view (:initializers %))) actions)
        _ (reduce (fn [seen v]
                    (let [key (allocation-key v) a (:allocation v)]
                      (when (and (contains? seen key) (not= (get seen key) a))
                        (fail! "one physical allocation cannot have conflicting contracts"
                               :distributed-readiness-allocation {:allocation key}))
                      (assoc seen key a))) {} views)
        initializers (initialization-facts (mapcat :initializers actions))
        ancestors (reduce (fn [known step]
                            (assoc known (:id step)
                                   (into (set (:dependencies step))
                                         (mapcat #(get known %) (:dependencies step))))) {} steps)]
    (loop [remaining actions
           facts (mapv #(hash-map :view (:view %) :initializer (:id %)) initializers)
           history {}]
      (if-let [action (first remaining)]
        (let [history (check-races history (get ancestors (:id action)) action)]
          (doseq [v (:requires action)] (require-covered! facts v nil (:id action)))
          (doseq [{:keys [view producer]} (:fresh action)]
            (require-covered! facts view producer (:id action)))
          (let [facts (reduce (fn [facts write]
                                (into [] (mapcat (fn [fact]
                                                  (map #(assoc fact :view %)
                                                       (view/subtract-contiguous (:view fact) write)))) facts))
                              facts (:writes action))
                facts (into facts (map #(hash-map :view % :producer (:id action)) (:produces action)))]
            (recur (next remaining) facts history)))
        {:initializers initializers :actions (mapv #(dissoc % :initializers) actions)
         :final-regions facts}))))
