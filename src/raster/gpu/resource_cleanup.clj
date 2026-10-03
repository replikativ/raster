(ns raster.gpu.resource-cleanup
  "Owner-local native resource teardown. This is runtime state, not portable evidence.
   Release callbacks must throw on failure and must return only after successful destruction.
   Failed resources remain owned. Retry requires the failure's ex-data to explicitly declare
   :cleanup-retry-safe? true; an indeterminate outcome is retained without another native call.")

(defrecord Cleanup [state])

(defn owner
  "Capture an ordered destruction plan. Each entry has :id, :release (zero-argument function),
   and optional :after IDs that must have been successfully released first. Dependencies must
   precede their dependants. Independent entries continue after failures; dependants remain owned.
   Construct this once alongside the native owner, never anew on each release attempt.
   IDs must be diagnostic identifiers, never native handles or release callbacks."
  [resources]
  (when-not (vector? resources)
    (throw (ex-info "Cleanup resources require an ordered vector" {:reason :invalid-cleanup-plan})))
  (reduce (fn [seen {:keys [id release after] :as resource}]
            (when-not (and (map? resource) (some? id) (not (contains? seen id))
                           (not (contains? resource :failure))
                           (not (contains? resource :retry-safe?))
                           (fn? release) (or (nil? after) (set? after))
                           (every? #(contains? seen %) after))
              (throw (ex-info "Cleanup resource has invalid identity, callback or dependency order"
                              {:reason :invalid-cleanup-plan :id id :after after})))
            (conj seen id))
          #{} resources)
  (->Cleanup (atom {:phase :live :remaining resources})))

(defn pending
  "Remaining diagnostic identities in destruction order, excluding resource callbacks."
  [cleanup]
  (mapv :id (:remaining @(:state cleanup))))

(defn status
  "Inspect teardown disposition using diagnostic IDs, excluding callbacks/exception objects."
  [cleanup]
  (let [{:keys [phase remaining]} @(:state cleanup)]
    {:phase phase
     :remaining (mapv (fn [{:keys [id failure retry-safe?]}]
                        {:id id :disposition (cond (nil? failure) :pending
                                                   retry-safe? :retryable
                                                   :else :indeterminate)})
                      remaining)}))

(defn assert-live!
  "Prevent execution or rebinding once any teardown attempt has started. This check does not
   pin resources: the containing session must serialize use and teardown under its own lock."
  [cleanup]
  (when-not (= :live (:phase @(:state cleanup)))
    (throw (ex-info "Native owner teardown has started"
                    {:reason :owner-releasing :pending (pending cleanup)})))
  cleanup)

(defn release!
  "Release each currently independent resource, retaining only failed/blocked entries.
   Successful callbacks never run again. The first failure is rethrown with later independent
   failures suppressed. The owner stays non-live even after a failed attempt; subsequent calls
   retry explicitly retry-safe failures only. An unclassified/indeterminate failure is reported
   again without retrying its callback. Calls serialize; recursive teardown is rejected."
  [cleanup]
  (let [state (:state cleanup)]
    (locking state
      (when (= :releasing (:phase @state))
        (throw (ex-info "Recursive native owner teardown"
                        {:reason :recursive-cleanup})))
      (let [resources (:remaining @state)
            failure (volatile! nil)
            record-error! (fn [error]
                            (if-let [primary @failure]
                              (when-not (or (identical? primary error)
                                            (some #(identical? error %)
                                                  (.getSuppressed ^Throwable primary)))
                                (.addSuppressed ^Throwable primary error))
                              (vreset! failure error)))]
        (swap! state assoc :phase :releasing)
        (try
          (doseq [{:keys [id release after] prior-failure :failure retry-safe? :retry-safe?} resources]
            (when-not (some #(contains? (set (pending cleanup)) %) after)
              (if (and prior-failure (not retry-safe?))
                (record-error! prior-failure)
                (try
                  (release)
                  (swap! state update :remaining
                         (fn [entries] (filterv #(not= id (:id %)) entries)))
                  (catch Throwable error
                    (swap! state update :remaining
                           (fn [entries]
                             (mapv #(if (= id (:id %))
                                      (assoc % :failure error
                                               :retry-safe? (true? (:cleanup-retry-safe? (ex-data error))))
                                      %) entries)))
                    (record-error! error))))))
          (finally
            (swap! state assoc :phase
                   (cond (empty? (:remaining @state)) :released
                         (some #(and (:failure %) (not (:retry-safe? %))) (:remaining @state)) :poisoned
                         :else :failed))))
        (when-let [error @failure] (throw error)))))
  nil)
