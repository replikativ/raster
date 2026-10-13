(ns raster.gpu.cuda-execution
  "Ordered stream replay, not native CUDA Graph instantiation. The common cleanup
   owners and pending-use ledger remain the lifetime and completion authorities."
  (:require [raster.gpu.cuda-prepared-execution :as pending]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]))

(defn- owner! [value]
  (or (::cleanup/owner value)
      (throw (ex-info "CUDA execution value has no owner" {:reason :missing-cleanup-owner}))))

(defn- retire-under-lock! [slot expected]
  (locking slot
    (when (and expected (not (identical? (owner! expected) (some-> @slot owner!))))
      (throw (ex-info "CUDA submission is no longer current" {:reason :submission-not-current})))
    (when-let [submission @slot]
      (cleanup/release! (owner! submission))
      (when (empty? (cleanup/pending (owner! submission)))
        (compare-and-set! slot submission nil)))))

(defn- retire!
  ([state slot] (retire! state slot nil))
  ([state slot expected]
   (cleanup/with-registry-use state (retire-under-lock! slot expected))))

(defn record-graph!
  ;; Graph adoption belongs to the common session/graph table, not ledger.debts.
  ;; Debt sweeping holds state before owner; graph destruction holds owner before state.
  [state prepareds {:keys [profile? adopt-cleanup!] :or {profile? false}}]
  (when-not (and (vector? prepareds) (boolean? profile?))
    (throw (ex-info "CUDA replay requires ordered prepared calls" {:reason :cuda-graph-order})))
  (let [slot (atom nil) leases (mapv (fn [_] (cleanup/acquisition-slot)) prepareds)
        root-slot (cleanup/acquisition-slot)
        resources (into [{:id :submission :release #(retire! state slot)}]
                        (map-indexed
                         (fn [i lease]
                           {:id [:prepared-use i] :after #{:submission}
                            :release #(cleanup/release-native!
                                       lease (fn [v] (cleanup/release! (owner! v))))}) leases))
        owner (cleanup/owner
               (conj resources {:id :runtime-root :after (set (map :id resources))
                                :release #(cleanup/release-native!
                                           root-slot (fn [v] (cleanup/release! (owner! v))))}))]
    (cleanup/build!
     owner
     #(do
        ;; Dependency owner admission precedes root locking: prepared destruction drains
        ;; under the root monitor, so reversing this order could deadlock.
        (doseq [[lease prepared] (map vector leases prepareds)]
          (cleanup/acquire-native! lease (fn [] (cleanup/retain-use! (owner! prepared)))))
        (root/capture-lease! state root-slot)
        (cleanup/with-registry-use state
          (root/assert-live! state)
          (pending/assert-admitted! (:prepared-execution @state))
          (doseq [prepared prepareds]
            (when-not (identical? (:raster.gpu.cuda-runtime/context prepared) (:context @state))
              (throw (ex-info "CUDA replay contains another context generation"
                              {:reason :runtime-generation-mismatch}))))
          {:prepareds prepareds :submission-state slot :profile? profile? :state state
           :context (:context @state)}))
     adopt-cleanup!)))

(defn submit-graph! [state graph native!]
  (when-not (identical? state (:state graph))
    (throw (ex-info "CUDA recording belongs to another runtime root"
                    {:reason :runtime-generation-mismatch})))
  (cleanup/using!
   (owner! graph)
   #(cleanup/with-registry-use state
      (locking (:submission-state graph)
        (when @(:submission-state graph)
          (throw (ex-info "CUDA graph still owns its previous submission"
                          {:reason :submission-still-owned})))
        (let [prepareds (:prepareds graph) profile? (:profile? graph)
              ledger (:prepared-execution @state) stream (:stream @state)
              binding (pending/binding-state)
              slots (mapv (fn [_] (cleanup/acquisition-slot))
                          (range (if profile? (* 2 (count prepareds)) (if (seq prepareds) 1 0))))
              resources (into [{:id :completion :release (fn [] (pending/drain! ledger binding stream native!))}]
                              (map-indexed
                               (fn [i slot]
                                 {:id [:event i] :after #{:completion}
                                  :release (fn [] (cleanup/release-native! slot (fn [e] (native! :event-destroy [e]))))}) slots))
              owner (cleanup/owner resources)
              submission {::cleanup/owner owner :graph graph :binding binding
                          :ledger ledger :stream stream :profile? profile? :complete? (empty? prepareds)}
              slot (:submission-state graph)
              admit! (fn []
                       (root/assert-live! state)
                       (when-not (identical? (:context graph) (:context @state))
                         (throw (ex-info "CUDA replay generation changed" {:reason :runtime-generation-mismatch}))))
              enqueue! (fn [operation arguments]
                         (pending/submit! ledger binding stream native! admit! operation arguments true))]
        ;; Retain the exact owner before any enqueue or event acquisition.
          (reset! slot submission)
          (try
            (cleanup/build!
             owner
             (fn []
               (admit!)
               (pending/assert-admitted! ledger)
             ;; Replay is not a fresh admission authority for an unresolved standalone call.
               (doseq [prepared prepareds]
                 (pending/assert-binding-admitted! ledger (:pending prepared)))
               (let [events (mapv (fn [s] (cleanup/acquire-native! s (fn [] (native! :event-create [])))) slots)]
               ;; Reserve the graph's pending completion before any launch. The same event
               ;; is recorded again at the end for unprofiled completion, not timing.
                 (when (and (seq events) (not profile?))
                   (enqueue! :event-record [(first events) stream]))
                 (doseq [[i prepared] (map-indexed vector prepareds)]
                   (when profile? (enqueue! :event-record [(nth events (* 2 i)) stream]))
                   (pending/submit! ledger (:pending prepared) stream native! admit!
                                    :launch [(:function prepared) (:geometry prepared) (:parameters prepared) stream] true)
                   (when profile? (enqueue! :event-record [(nth events (inc (* 2 i))) stream])))
                 (when (and (seq events) (not profile?))
                   (enqueue! :event-record [(peek events) stream]))
                 (let [token (assoc submission :events events :event (peek events))]
                   (reset! slot token) token)))
             (fn [_] nil))
            (catch Throwable primary
              (when (empty? (cleanup/pending owner)) (reset! slot nil))
              (throw primary))))))))

(defn- with-token [token f]
  (cleanup/using!
   (owner! (:graph token))
   #(cleanup/using!
     (owner! token)
     (fn []
       (cleanup/with-registry-use (:state (:graph token))
         (locking (:submission-state (:graph token))
           (root/assert-live! (:state (:graph token)))
           (when-not (identical? (:context (:graph token)) (:context @(:state (:graph token))))
             (throw (ex-info "CUDA replay generation changed" {:reason :runtime-generation-mismatch})))
           (when-not (identical? (owner! token) (some-> @(:submission-state (:graph token)) owner!))
             (throw (ex-info "CUDA submission is no longer current" {:reason :submission-not-current})))
           (f)))))))

(defn event-complete? [token native!]
  (with-token token #(if (:complete? token) true (boolean (native! :event-query [(:event token)])))))

(defn await-event! [token _native!]
  (with-token token #(pending/drain! (:ledger token) (:binding token) (:stream token) _native!))
  nil)

(defn release-event! [token]
  (if (:profile? token)
    ;; Profile events belong to the recording until timestamps are consumed/reset.
    (with-token token (fn [] nil))
    (cleanup/using!
     (owner! (:graph token))
     #(let [slot (:submission-state (:graph token)) state (:state (:graph token))]
        (cleanup/with-registry-use state
          (locking slot
            (when-not (identical? (owner! token) (some-> @slot owner!))
              (throw (ex-info "CUDA submission is no longer current" {:reason :submission-not-current})))
            (retire-under-lock! slot token))))))
  nil)

(defn destroy-graph! [graph] (cleanup/release! (owner! graph)))

(defn reset-graph-events! [graph]
  (cleanup/using! (owner! graph) #(retire! (:state graph) (:submission-state graph)))
  nil)

(defn read-graph-timestamps! [graph native!]
  (cleanup/using!
   (owner! graph)
   #(do
      (when-not (:profile? graph)
        (throw (ex-info "CUDA replay is not profiled" {:reason :cuda-graph-not-profiled})))
      (let [token @(:submission-state graph)]
        (when-not token (throw (ex-info "CUDA replay has no submission" {:reason :cuda-graph-no-submission})))
        (let [result
              (with-token
                token
                (fn []
                  (pending/drain! (:ledger token) (:binding token) (:stream token) native!)
                  (let [events (:events token)
                        rows (mapv (fn [i prepared]
                                     (let [ns (native! :event-elapsed [(nth events (* i 2)) (nth events (inc (* i 2)))])]
                                       {:kernel-name (:kernel-name prepared) :phase (:phase prepared)
                                        :ms (/ (double ns) 1.0e6) :context-ms (/ (double ns) 1.0e6)}))
                                   (range (count (:prepareds graph))) (:prepareds graph))]
                    {:kernels rows :timing-source :device-event :ns-per-tick 1.0
                     :wall-ms (if (seq events) (/ (double (native! :event-elapsed [(first events) (peek events)])) 1.0e6) 0.0)})))]
          (retire! (:state graph) (:submission-state graph) token) result)))))
