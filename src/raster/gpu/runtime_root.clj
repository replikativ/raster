(ns raster.gpu.runtime-root
  "Exact backend root construction ownership. Live teardown stays closed until child leases land.
   This is native runtime authority, never compiler metadata or portable planning evidence."
  (:require [raster.gpu.resource-cleanup :as cleanup]))

(defn- entries [state]
  (filterv (fn [[_ value]] (and (map? value) (::root? value))) (vec @state)))

(defn- retire-projection! [state entry]
  (swap! state
         (fn [current]
           (if (identical? entry (::entry current))
             (assoc (apply dissoc current (keys @(:projection entry))) :initialized? false)
             current))))

(defn- retire-released! [state entry]
  (when (empty? (cleanup/pending (::cleanup/owner entry)))
    (retire-projection! state entry)
    (cleanup/release-entries! state
                              (filterv (fn [[_ value]] (identical? (::cleanup/owner entry) (::cleanup/owner value)))
                                       (entries state))
                              #(cleanup/release! (::cleanup/owner %)))))

(defn assert-live!
  "Admit only the exact live root and its unchanged native projections. Does not pin resources."
  [state]
  (locking state
    (let [entry (::entry @state)]
      (when-not (and entry (= 1 (count (entries state)))
                     (= :live @(:phase entry)) (:initialized? @state))
        (throw (ex-info "Runtime has no admitted live root"
                        {:reason :runtime-root-unavailable})))
      (cleanup/assert-live! (::cleanup/owner entry))
      (doseq [[key value] @(:projection entry)]
        (when-not (identical? value (get @state key))
          (throw (ex-info "Runtime root projection changed generation"
                          {:reason :runtime-generation-mismatch :field key}))))
      entry)))

(defn acquire!
  "Acquire into a pre-reserved root slot. Caller holds the root's state lifecycle scope."
  [entry id acquire]
  (cleanup/assert-live! (::cleanup/owner entry))
  (let [slot (or (get (:slots entry) id)
                 (throw (ex-info "Runtime root slot was not reserved"
                                 {:reason :invalid-root-slot :slot id})))]
    (try
      (cleanup/acquire-native! slot acquire)
      (catch Throwable error
        (vreset! (:phase entry) :failed)
        (throw error)))))

(defn initialize!
  "Reserve/publish one generation before any acquisition. plan uses Cleanup's resource DAG;
   each :release receives its acquired resource. build returns backend-native projections.
   Failed rollback remains in state; known successful rollback removes only its exact generation."
  [state plan build]
  ;; Validate the original callbacks before wrapping them in Cleanup callbacks.
  ;; Otherwise a malformed release would be discovered only after native acquisition.
  (when-not (and (vector? plan) (every? #(and (map? %) (fn? (:release %))) plan)
                 (fn? build))
    (throw (ex-info "Runtime root requires a checked construction plan"
                    {:reason :invalid-cleanup-plan})))
  (cleanup/with-registry-use state
    (if (seq (entries state))
      (do (assert-live! state) nil)
      (do
        (when (:initialized? @state)
          (throw (ex-info "Initialized runtime has no root owner"
                          {:reason :missing-cleanup-owner})))
        (let [slots (into {} (map (fn [{:keys [id]}] [id (cleanup/acquisition-slot)]) plan))
              owner (cleanup/owner
                     (mapv (fn [{:keys [id release] :as resource}]
                             (assoc resource :release #(cleanup/release-native! (get slots id) release)))
                           plan))
              entry {::root? true :generation (random-uuid) :phase (volatile! :constructing)
                     :ever-live? (volatile! false) :projection (volatile! {})
                     :leases (volatile! {})
                     :slots slots ::cleanup/owner owner}]
          (try
            (cleanup/build! owner
                            (fn []
                              (cleanup/publish-replacement! state [::entry] entry
                                                            #(cleanup/release! (::cleanup/owner %)))
                              (let [projection (build entry)]
                                (when-not (and (map? projection) (not (contains? projection ::entry))
                                               (not (contains? projection ::cleanup/owner)))
                                  (throw (ex-info "Runtime initialization requires native projections"
                                                  {:reason :invalid-runtime-projection})))
                  ;; Record fields before publication so post-swap watch failures can remove them.
                                (vreset! (:projection entry) projection)
                                (swap! state
                                       (fn [current]
                                         (when-not (identical? entry (::entry current))
                                           (throw (ex-info "Runtime root changed before publication"
                                                           {:reason :runtime-generation-mismatch})))
                                         (merge current projection {:initialized? true})))
                                (when-not (identical? entry (::entry @state))
                                  (throw (ex-info "Runtime root changed during publication"
                                                  {:reason :runtime-generation-mismatch})))
                                (doseq [[key value] projection]
                                  (when-not (identical? value (get @state key))
                                    (throw (ex-info "Runtime projection changed during publication"
                                                    {:reason :runtime-generation-mismatch :field key}))))
                                (vreset! (:phase entry) :live)
                                (vreset! (:ever-live? entry) true)
                                entry))
                            (fn [_] (cleanup/retain-owned-entry! state entry)))
            nil
            (catch Throwable primary
              (vreset! (:phase entry) :failed)
              (try (retire-released! state entry)
                   (catch Throwable retirement-error
                     (when-not (identical? primary retirement-error)
                       (.addSuppressed primary retirement-error))))
              (throw primary))))))))

(defn acquire-child!
  "Acquire a lazy root child exactly once. Unknown outcome poisons admission, never retries.
   prepare runs before slot acquisition and returns the native acquisition callback."
  [state id prepare]
  (cleanup/with-registry-use state
    (let [entry (assert-live! state)
          slot (or (get (:slots entry) id)
                   (throw (ex-info "Runtime root slot was not reserved"
                                   {:reason :invalid-root-slot :slot id})))]
      (case (:phase @slot)
        :live (:resource @slot)
        :not-acquired (let [acquire (prepare entry)]
                        (acquire! entry id acquire))
        (throw (ex-info "Runtime root child has unresolved acquisition"
                        {:reason :runtime-root-unavailable :slot id}))))))

(defn- lease-under-lock!
  [state]
  (let [entry (assert-live! state)
        token (random-uuid)
        owner-slot (volatile! nil)
        owner (cleanup/owner
               [{:id :runtime-root-lease
                 :release
                 #(locking state
                      ;; Failed lazy acquisition still permits known child retirement.
                      ;; Lost-generation authority must remain reachable, never be guessed.
                    (when-not (some (fn [[_ value]] (identical? entry value)) (entries state))
                      (throw (ex-info "Runtime lease lost its exact root generation"
                                      {:reason :runtime-generation-mismatch})))
                    (when-not (identical? @owner-slot (get @(:leases entry) token))
                      (throw (ex-info "Runtime lease lost its exact owner"
                                      {:reason :cleanup-owner-mismatch})))
                    (vswap! (:leases entry) dissoc token))}])]
    (vreset! owner-slot owner)
    (vswap! (:leases entry) assoc token owner)
    {::entry entry ::lease-token token ::cleanup/owner owner}))

(defn lease!
  "Pin the admitted root generation until the returned Cleanup owner is released.
   Runtime-only authority: callers retain this in their resource owner's cleanup DAG.
   The private lease table holds owners, not native pointers or compiler metadata."
  [state]
  (cleanup/with-registry-use state (lease-under-lock! state)))

(defn construct-child!
  "Construct one canonical child owner, with the root lease as its final dependency.
   build receives that composite Cleanup and the exact root entry under the lifecycle lock.
   Retain/adopt the composite owner, never just the standalone root lease."
  [state resources build adopt-cleanup!]
  (when-not (and (vector? resources) (fn? build)
                 (or (nil? adopt-cleanup!) (fn? adopt-cleanup!)))
    (throw (ex-info "Runtime child construction requires a checked plan"
                    {:reason :invalid-cleanup-plan})))
  (let [lease-slot (volatile! nil)
        owner (cleanup/owner
               (conj resources
                     {:id :runtime-root-lease :after (set (map :id resources))
                      :release #(when-let [lease @lease-slot]
                                  (cleanup/release! (::cleanup/owner lease)))}))]
    (cleanup/with-registry-use state
      (cleanup/build!
       owner
       (fn []
         (let [lease (lease-under-lock! state)]
           (vreset! lease-slot lease)
           (build owner (::entry lease))))
       adopt-cleanup!))))

(defn assert-lease-live!
  "Check lease authority and current root admission. Does not create another lease."
  [state lease]
  (locking state
    (let [entry (assert-live! state)
          owner (::cleanup/owner lease)]
      (when-not (and (identical? entry (::entry lease)) owner
                     (identical? owner (get @(:leases entry) (::lease-token lease))))
        (throw (ex-info "Runtime lease does not belong to the admitted root"
                        {:reason :runtime-generation-mismatch})))
      (cleanup/assert-live! owner)
      lease)))

(defn lease-count
  "Diagnostic live lease count of the exact admitted root, never destruction authority."
  [state]
  (locking state
    (count @(:leases (assert-live! state)))))

(defn shutdown-construction!
  "Clean failed initialization only. Ever-live roots fail closed until all child leases exist.
   An async-child failure after initialization does not waive the live-child preflight."
  [state]
  (cleanup/with-registry-use state
    (let [roots (entries state)]
      (when (some #(true? @(:ever-live? (second %))) roots)
        (throw (ex-info "Live runtime teardown awaits complete resource leases"
                        {:reason :runtime-root-leases-incomplete})))
      (when (and (:initialized? @state) (empty? roots))
        (throw (ex-info "Initialized runtime has no root owner" {:reason :missing-cleanup-owner})))
      (cleanup/release-entries! state roots
                                (fn [entry]
                                  (cleanup/release! (::cleanup/owner entry))
                                  (retire-projection! state entry)))))
  nil)
