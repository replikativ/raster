(ns raster.gpu.resource-cleanup
  "Owner-local native resource teardown. This is runtime state, not portable evidence.
   Release callbacks must throw on failure and must return only after successful destruction.
   Failed resources remain owned. Retry requires the failure's ex-data to explicitly declare
   :cleanup-retry-safe? true; an indeterminate outcome is retained without another native call.")

(defrecord Cleanup [state])

(def ^:dynamic *registry-uses* [])
(defn- assert-not-building! [cleanup]
  (when (::construction-token @(:state cleanup))
    (throw (ex-info "Native owner construction is still in progress"
                    {:reason :owner-construction-in-progress :cleanup-retry-safe? true}))))

(defn- clear-construction! [cleanup token]
  (locking (:state cleanup)
    (when-not (identical? token (::construction-token @(:state cleanup)))
      (throw (ex-info "Native construction marker changed generation"
                      {:reason :owner-construction-generation-mismatch})))
    (swap! (:state cleanup)
           #(if (identical? token (::construction-token %))
              (dissoc % ::construction-token)
              (throw (ex-info "Native construction marker changed generation"
                              {:reason :owner-construction-generation-mismatch}))))
    (when (::construction-token @(:state cleanup))
      (throw (ex-info "Native construction marker was reinserted during retirement"
                      {:reason :owner-construction-generation-mismatch})))))

(defn assert-registry-mutable!
  "Reject same-thread lifecycle mutation while native registration acquisition is in flight.
   The registry monitor serializes other threads; this guard also covers reentrant callbacks."
  [registry]
  (when (some #(identical? registry %) *registry-uses*)
    (throw (ex-info "Registration lifecycle mutation during native use"
                    {:reason :registration-in-use}))))

(defmacro with-registry-use
  "Serialize native use and prevent reentrant registration retirement on the same thread."
  [registry & body]
  `(let [registry# ~registry]
     (assert-registry-mutable! registry#)
     (locking registry#
       (binding [*registry-uses* (conj *registry-uses* registry#)]
         ~@body))))

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
      (assert-not-building! cleanup)
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

(defn retain-owned-entry!
  "Retain an exact owning entry after a watch loses its generation. Preserve unrelated entries.
   Retained fields must be runtime state only, never portable compiler metadata."
  [registry entry]
  (locking registry
    (when-not (::owner entry)
      (throw (ex-info "Retained entry requires cleanup authority" {:reason :missing-cleanup-owner})))
    (when-not (some #(identical? (::owner entry) (::owner %)) (vals @registry))
      (let [key (random-uuid)
            entry (assoc entry ::failed-registration true)]
        (swap! registry assoc key entry)
        (when-not (identical? entry (get @registry key))
          (throw (ex-info "Registry rejected unresolved registration cleanup"
                          {:reason :cleanup-retention-failed}))))))
  nil)

(defn retain-registration!
  "Keep unresolved cleanup reachable by arena teardown if a registry watch lost its generation.
   Preserve unrelated registrations. This hidden entry carries authority, never native metadata."
  [registry registration cleanup]
  (retain-owned-entry! registry {:arena-id (:arena-id registration) ::owner cleanup}))

(defn retaining-registration!
  "Run a registration operation without losing its existing owner on publication failure.
   Unlike candidate rollback, this also retains a live parent after successful child rollback."
  [registry registration operation]
  (try
    (operation)
    (catch Throwable primary
      (let [cleanup (::owner registration)]
        (when (seq (pending cleanup))
          (try (retain-registration! registry registration cleanup)
               (catch Throwable retention-error
                 (let [wrapper (ex-info "Registration operation retains unresolved cleanup"
                                        {::unresolved cleanup} primary)]
                   (.addSuppressed wrapper retention-error)
                   (throw wrapper))))))
      (throw primary))))

(defn assert-registration-current!
  "Require the exact registration snapshot, not merely a live replacement generation."
  [registry key registration]
  (when-not (identical? registration (get @registry key))
    (throw (ex-info "Registration generation changed during native use"
                    {:reason :registry-generation-changed :key key})))
  registration)

(defn publish-owned-update!
  "Publish metadata/cache fields within one canonical owner generation, without retiring it.
   Failure preserves whatever exact generation is installed; construction callers still own
   rollback. Reject owner changes and stale CAS input before publishing, and detect atom-watch
   mutations before reporting success. The caller must admit owner liveness before native use."
  [registry path prior updated]
  (when-not (and (instance? clojure.lang.Atom registry)
                 (vector? path) (seq path) (every? some? path)
                 (::owner prior) (identical? (::owner prior) (::owner updated)))
    (throw (ex-info "Registry update requires one canonical owner generation"
                    {:reason :invalid-registry-update})))
  (locking registry
    (swap! registry
           (fn [state]
             (when-not (identical? prior (get-in state path))
               (throw (ex-info "Registry generation changed before update"
                               {:reason :registry-generation-changed :path path})))
             (assoc-in state path updated)))
    (when-not (identical? updated (get-in @registry path))
      (throw (ex-info "Registry generation changed during update publication"
                      {:reason :registry-generation-changed :path path})))
    updated))

(defn release-entries!
  "Release independently owned entries from a runtime atom. Remove only exact successfully
   released generations; preserve failed entries and unrelated reentrant replacements. Snapshot
   entries must be supplied by the containing owner's serialized teardown, not inferred handles."
  [registry entries destroy!]
  (locking registry
    (let [failure (volatile! nil)]
      (doseq [[key entry] entries]
        (when (identical? entry (get @registry key))
          (try
            (destroy! entry)
            (swap! registry #(if (identical? entry (get % key)) (dissoc % key) %))
            (when (identical? entry (get @registry key))
              (throw (ex-info "Registry watch reinserted a released generation"
                              {:reason :registry-generation-changed :key key})))
            (catch Throwable error
              (if-let [primary @failure]
                (when-not (or (identical? primary error)
                              (some #(identical? error %) (.getSuppressed primary)))
                  (.addSuppressed primary error))
                (vreset! failure error))))))
      (when-let [primary @failure] (throw primary))))
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
  ;; A prepublished owner may already be visible to another thread. Decline teardown while
  ;; building, without holding its monitor across arbitrary callbacks or native contact.
  (let [token (Object.) admitted? (volatile! false)]
    (try
      (locking (:state cleanup)
        (assert-not-building! cleanup)
        (assert-live! cleanup)
        (vreset! admitted? true)
        (swap! (:state cleanup) assoc ::construction-token token)
        (when-not (identical? token (::construction-token @(:state cleanup)))
          (throw (ex-info "Native construction marker changed during publication"
                          {:reason :owner-construction-generation-mismatch})))
        (assert-live! cleanup))
      (let [value (build)]
        (when-not (map? value)
          (throw (ex-info "Native construction must return an owning map"
                          {:reason :invalid-owned-value})))
        (clear-construction! cleanup token)
        (assoc value ::owner cleanup))
      (catch Throwable primary
      ;; An overlapping build owns the existing marker. Never clear or retire its generation.
        (when-not @admitted? (throw primary))
        (try (when (identical? token (::construction-token @(:state cleanup)))
               (clear-construction! cleanup token))
             (catch Throwable secondary
               (when-not (identical? primary secondary) (.addSuppressed primary secondary))))
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
        (throw primary)))))

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
