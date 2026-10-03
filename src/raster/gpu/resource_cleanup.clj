(ns raster.gpu.resource-cleanup
  "Owner-local native resource teardown. This is runtime state, not portable evidence.
   Release callbacks must throw on failure and must return only after successful destruction.
   Failed resources remain owned. Retry requires the failure's ex-data to explicitly declare
   :cleanup-retry-safe? true; an indeterminate outcome is retained without another native call.")

(defrecord Cleanup [state])

(defn lifetime-owner
  "Exact native lifetime identity for a root or a non-owning view. A borrowed lifetime reference
   is not destruction authority; only ::owner can release the value's native resource."
  [value]
  (or (::owner value) (::lifetime-owner value)))

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
          (doseq [{:keys [id release after] prior-failure :failure retry-safe? :retry-safe?} resources
                  :let [remaining-ids (set (pending cleanup))]]
            (when-not (some #(contains? remaining-ids %) after)
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

(defn publish-replacement!
  "Replace one exact runtime registry generation, retiring the old entry before publication.
   Holds the registry monitor. On failure after retirement, remove only the exact old/candidate
   entry; preserve unrelated reentrant generations. The caller owns candidate rollback. A failed
   old teardown remains registered as debt. Arbitrary validators can reject recovery mutations;
   these errors are suppressed on the primary. This is runtime authority, never compiler evidence."
  [registry path candidate destroy!]
  (when-not (and (instance? clojure.lang.Atom registry)
                 (vector? path) (seq path) (every? some? path)
                 (some? candidate) (fn? destroy!))
    (throw (ex-info "Registry replacement requires an atom, nonempty path, candidate and teardown callback"
                    {:reason :invalid-replacement-contract})))
  (locking registry
    (let [old (get-in @registry path)
          retired? (volatile! false)]
      (when (identical? old candidate)
        (throw (ex-info "Replacement must be a distinct registry generation"
                        {:reason :replacement-generation-unchanged :path path})))
      (try
        (when (some? old) (destroy! old))
        (vreset! retired? true)
        (when-not (identical? old (get-in @registry path))
          (throw (ex-info "GPU binding changed during retirement"
                          {:reason :reentrant-binding-publication :path path})))
        (swap! registry
               (fn [state]
                 (when-not (identical? old (get-in state path))
                   (throw (ex-info "GPU binding changed before publication"
                                   {:reason :reentrant-binding-publication :path path})))
                 (assoc-in state path candidate)))
        (when-not (identical? candidate (get-in @registry path))
          (throw (ex-info "GPU binding changed during publication"
                          {:reason :reentrant-binding-publication :path path})))
        candidate
        (catch Throwable primary
          (when @retired?
            (try
              (swap! registry
                     (fn [state]
                       (let [installed (get-in state path)]
                         (if (or (identical? installed old) (identical? installed candidate))
                           (if (= 1 (count path))
                             (dissoc state (first path))
                             (update-in state (pop path) dissoc (peek path)))
                           state))))
              (catch Throwable secondary
                (when-not (or (identical? primary secondary)
                              (some #(identical? secondary %) (.getSuppressed primary)))
                  (.addSuppressed primary secondary)))))
          (throw primary))))))

(defn build!
  "Build an owning map using an already reserved cleanup owner; attach ::owner automatically.
   Rollback/adoption has the same
   failure contract as construct!, including original-primary identity when adoption succeeds."
  [cleanup build adopt-cleanup!]
  (when-not (and (fn? build) (or (nil? adopt-cleanup!) (fn? adopt-cleanup!)))
    (throw (ex-info "Native construction requires callbacks" {:reason :invalid-cleanup-plan})))
  (try
    (let [value (build)]
      (when-not (map? value)
        (throw (ex-info "Native construction must return an owning map"
                        {:reason :invalid-owned-value})))
      (assoc value ::owner cleanup))
    (catch Throwable primary
      (try (release! cleanup)
           (catch Throwable secondary
             (when-not (or (identical? primary secondary)
                           (some #(identical? secondary %) (.getSuppressed ^Throwable primary)))
               (.addSuppressed primary secondary))))
      (when (seq (pending cleanup))
        (let [fallback #(ex-info "Native construction retains unresolved cleanup ownership"
                                 {::unresolved cleanup} primary)]
          (if adopt-cleanup!
            (try (adopt-cleanup! cleanup)
                 (catch Throwable adoption-error
                   (let [wrapper (fallback)]
                     (.addSuppressed wrapper adoption-error)
                     (throw wrapper))))
            (throw (fallback)))))
      (throw primary))))

(defn acquisition-slot
  "Reserve backend-local acquisition state before native contact. Not compiler IR/evidence."
  []
  (volatile! {:phase :not-acquired}))

(defn acquire-native!
  "Capture a native create-and-readback result. A throwing native callback/readback has an
   indeterminate outcome; never infer that no object was created because no value returned."
  [slot acquire]
  (when-not (fn? acquire)
    (throw (ex-info "Native acquisition requires a callback" {:reason :invalid-cleanup-plan})))
  (when-not (= :not-acquired (:phase @slot))
    (throw (ex-info "Native acquisition slot is not fresh" {:reason :acquisition-already-started})))
  (vreset! slot {:phase :acquiring})
  (try
    (let [resource (acquire)]
      (vreset! slot {:phase :live :resource resource})
      resource)
    (catch Throwable error
      (vreset! slot {:phase :indeterminate :error error})
      (throw error))))

(defn release-native!
  "Release a successful native acquisition, or report an indeterminate acquisition unchanged.
   Use only within the containing reserved cleanup owner's serialized callback."
  [slot release]
  (when-not (fn? release)
    (throw (ex-info "Native release requires a callback" {:reason :invalid-cleanup-plan})))
  (case (:phase @slot)
    (:not-acquired :released) nil
    :indeterminate (throw (:error @slot))
    :live (let [resource (:resource @slot)]
            (try
              (release resource)
              (vreset! slot {:phase :released})
              nil
              (catch Throwable error
                (when-not (true? (:cleanup-retry-safe? (ex-data error)))
                  (vreset! slot {:phase :indeterminate :error error}))
                (throw error))))
    (throw (ex-info "Native acquisition has not completed" {:reason :acquisition-in-progress}))))

(defn construct!
  "Acquire one native resource and build its owning value under immediate rollback ownership.
   build receives [resource cleanup] and must retain cleanup in its returned value. If build
   fails, attempt cleanup and preserve the exact primary Throwable. Unresolved ownership is
   passed to adopt-cleanup! before rethrowing the exact primary. Without an adoption callback,
   throw an ExceptionInfo that owns cleanup and keeps the original failure as its cause.
   acquire must itself account for any native resources it allocates without returning."
  ([id acquire release build] (construct! id acquire release build nil))
  ([id acquire release build adopt-cleanup!]
  (when-not (and (every? fn? [acquire release build])
                 (or (nil? adopt-cleanup!) (fn? adopt-cleanup!)))
    (throw (ex-info "Native construction requires callbacks" {:reason :invalid-cleanup-plan})))
  ;; Reserve rollback ownership before acquisition so ordinary post-acquisition setup failures
  ;; cannot strand a returned handle. A failed acquire that returns no handle has nothing here.
  (let [not-acquired (Object.)
        acquired (volatile! not-acquired)
        cleanup (owner [{:id id :release #(when-not (identical? not-acquired @acquired)
                                            (release @acquired))}])]
    (build! cleanup
            #(do (vreset! acquired (acquire)) (build @acquired cleanup))
            adopt-cleanup!))))
