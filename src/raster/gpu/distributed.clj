(ns raster.gpu.distributed
  "Synchronous execution of checked DistributedPlans with explicit worker target placement.
   Logical workers have explicit physical targets; placement does not imply execution overlap."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.resident-value :as resident-value]
            [raster.gpu.storage-representation :as storage]
            [raster.runtime.artifact-provenance :as provenance])
  (:import [java.lang.foreign Arena]))

(declare close!)

(def ^:private runtime-issuer (provenance/issuer))
(defn- seal-runtime-value [value] ((:seal runtime-issuer) value))

(defrecord DistributedExecutable [plan schedule readiness bindings projections sessions transport staging-bytes state allocation-budgets provenance-seal staging]
  java.io.Closeable
  (close [this] (close! this)))

(defn original-executable?
  "Whether this is the exact distributed owner issued by instantiate!, not a copied record.
   This proves in-process issuance only, not completion, byte content or durable provenance."
  [value]
  (and (instance? DistributedExecutable value) ((:authentic? runtime-issuer) value)))

(defrecord ResidentRepresentationEvidence [data executable session session-id provenance-seal]
  clojure.lang.IDeref
  (deref [this]
    (when-not ((:authentic? runtime-issuer) this)
      (throw (ex-info "distributed representation evidence requires its original object"
                      {:reason :distributed-representation-owner})))
    data))

(defn representation-evidence? [value]
  (and (instance? ResidentRepresentationEvidence value) ((:authentic? runtime-issuer) value)))

(defn- schedule [plan]
  (let [queues {:compute (execution/compute-queue) :transfer (execution/transfer-queue)}
        events (into {} (map (fn [step] [(:id step) (execution/->LogicalEvent [:complete (:id step)])])) (:steps plan))]
    (execution/validate!
     (execution/->ExecutionPlan
      (vec (vals queues)) []
      (mapv (fn [step]
              (execution/->ScheduledOperation (:id step) step (queues (:kind step))
                                              (mapv events (:dependencies step)) (events (:id step))))
            (:steps plan))
      (mapv events (:outputs plan))))))

(defn- physical-view [sessions view]
  (gpu/buffer-view (get sessions (get-in view [:allocation :device]))
                   (get-in view [:allocation :id])
                   (select-keys view [:id :byte-offset :dtype :shape :strides])))

(defn- close-sessions! [sessions]
  (let [failure (volatile! nil)]
    (doseq [session (reverse (vec (vals sessions)))]
      (try (gpu/close-session! session)
           (catch Throwable e (if-let [first @failure]
                                (when-not (identical? first e) (.addSuppressed ^Throwable first e))
                                  (vreset! failure e)))))
    (when-let [error @failure] (throw error))))

(defn- profile-transfer-leg! [submit! session entries]
  (let [event (submit! session entries)
        failure (volatile! nil)]
    (try
      (gpu/await-event! session event)
      (gpu/event-measurement session event)
      (catch Throwable e (vreset! failure e) (throw e))
      (finally
        ;; Drain while the enclosing staging arena is still live, including failed awaits.
        (try (gpu/release-event! session event)
             (catch Throwable cleanup
               (if-let [primary @failure]
                 (when-not (identical? primary cleanup)
                   (.addSuppressed ^Throwable primary cleanup))
                 (throw cleanup))))))))

(defn- owner-staging! [staging bytes]
  (or @staging
      (let [arena (Arena/ofShared)]
        (try
          (let [entry {:arena arena :segment (.allocate arena (long bytes) 16)}]
            (reset! staging entry)
            (when-not (identical? entry @staging)
              (throw (ex-info "staging owner rejected its lifetime"
                              {:reason :distributed-staging-publication})))
            entry)
          (catch Throwable e
            (try (.close arena)
                 (catch Throwable cleanup
                   (when-not (identical? e cleanup) (.addSuppressed ^Throwable e cleanup))))
            (throw e))))))

(defn- transfer-host-staged! [sessions source target staging-bytes profile? staging]
  (let [elements (reduce * 1 (:shape source))
        element-bytes (dtype/bytes-of (:dtype source))
        chunk-elements (min elements (quot staging-bytes element-bytes))
        observations (volatile! [])]
    (when (pos? elements)
      ;; One bounded shared buffer belongs to the execution, not an individual transfer call.
      ;; Submission/await/retirement failures keep it live until all sessions drain at close.
      (let [segment (:segment (owner-staging! staging staging-bytes))]
          (loop [offset 0]
            (when (< offset elements)
              (let [n (min chunk-elements (- elements offset))]
                (if profile?
                  (doseq [[direction submit! v opts]
                          [[:download gpu/submit-download-ranges! source {:src-element offset :elements n}]
                           [:upload gpu/submit-upload-ranges! target {:dst-element offset :elements n}]]]
                    (let [device (get-in v [:allocation :device])
                          measurement (profile-transfer-leg!
                                       submit! (get sessions device)
                                       [[(physical-view sessions v) segment opts]])]
                      (vswap! observations conj {:direction direction :target device
                                                :element-offset offset :elements n
                                                :bytes (* n element-bytes)
                                                :measurement measurement})))
                  (do
                    (gpu/download-range! (get sessions (get-in source [:allocation :device]))
                                         (physical-view sessions source) segment {:src-element offset :elements n})
                    (gpu/upload-range! (get sessions (get-in target [:allocation :device]))
                                       (physical-view sessions target) segment {:dst-element offset :elements n})))
                (recur (+ offset n)))))))
    (when profile? @observations)))

(defn instantiate!
  "Validate and initialize an owning, one-shot distributed execution.
   Local LinkPlans must target real devices and all local allocations must be owned. Explicit
   `{:transport :host-staged}` permits synchronous cross-device copies via a temporary native
   segment; it is not P2P/MPI or the topology's predicted transfer performance. Without that
   option, plans containing transfers are refused before contacting a device. Host staging is
   bounded by :max-staging-bytes (default 1 MiB), reused sequentially across transfer chunks.
   Co-located workers declare :target in each device-plan, use :transport :resident-copy, and
   supply an aggregate physical :device-capacities budget (bytes) for remapped targets. Local
   copies still execute; logical topology predictions are not physical runtime cost evidence.
   Budgets cover declared owned LinkPlan roots. Optional :include-graph-temporaries? includes
   conservative graph scratch bounds for this serial runner, not total driver memory.
   Source objects must remain valid and stable through this synchronous initialization."
  ([plan] (instantiate! plan {}))
  ([plan {:keys [transport max-staging-bytes device-capacities include-graph-temporaries?]
          :or {max-staging-bytes 1048576 device-capacities {} include-graph-temporaries? false}}]
   (let [ready (distributed/check-readiness plan)
         schedule (schedule plan)
         _ (when-not (contains? #{nil :host-staged :resident-copy} transport)
             (throw (ex-info "unsupported distributed transport"
                             {:reason :distributed-runtime-transport :transport transport})))
         _ (when-not (and (integer? max-staging-bytes) (<= 16 max-staging-bytes Long/MAX_VALUE))
             (throw (ex-info "host staging budget must be at least 16 bytes and fit a long"
                             {:reason :distributed-runtime-staging-size :bytes max-staging-bytes})))
         _ (when (and (some #(= :transfer (:kind %)) (:steps plan)) (nil? transport))
             (throw (ex-info "distributed copies require an explicit supported transport"
                             {:reason :distributed-runtime-transport :transport transport})))
         _ (when (= :resident-copy transport)
             (doseq [action (:actions ready) :when (= :transfer (:kind action))]
               (when-not (= (get-in action [:reads 0 :allocation :device])
                            (get-in action [:writes 0 :allocation :device]))
                 (throw (ex-info "resident-copy requires co-located physical endpoints"
                                 {:reason :distributed-runtime-resident-copy :step (:id action)})))))
         {:keys [bindings specs allocation-budgets]}
         (distributed/resident-storage-plan plan {:device-capacities device-capacities
                                                 :include-graph-temporaries? include-graph-temporaries?})
         projections (update-vals bindings #(link-plan/borrow-owned-storage (:link-plan %)))
         _ (doseq [[index action] (map-indexed vector (:actions ready))
                   :when (contains? (set (:outputs plan)) (:id action))
                   :let [local (get-in bindings [(:id action) :link-plan])]
                   node-id (:outputs local)
                   later (drop (inc index) (:actions ready))
                   write (:writes later)]
             (when (view/overlaps? (get-in local [:nodes node-id :view]) write)
               (throw (ex-info "a retained distributed output is overwritten before completion"
                               {:reason :distributed-runtime-output-overwritten :step (:id action)
                                :node node-id :writer (:id later)}))))
         sessions (atom {})]
     (try
       (doseq [[device entries] (sort-by (comp pr-str key) (group-by (comp first key) specs))]
         (let [session (gpu/make-session device)]
           (swap! sessions assoc device session)
           (gpu/alloc! session
                       (into {} (map (fn [[[_ id] {:keys [allocation dtype]}]]
                                       [id [dtype (quot (:byte-size allocation) (dtype/bytes-of dtype)) nil
                                            {:allocation-id id :memory-space (:memory-space allocation)
                                             :coherence (:coherence allocation) :alignment (:alignment allocation)}]]))
                             entries))))
       (doseq [{:keys [view source]} (:initializers ready)]
         (gpu/upload-range! (get @sessions (get-in view [:allocation :device]))
                            (physical-view @sessions view) source
                            {:elements (reduce * 1 (:shape view))}))
       (seal-runtime-value
        (->DistributedExecutable plan schedule ready bindings projections @sessions transport
                                 max-staging-bytes (atom :ready) allocation-budgets nil (atom nil)))
       (catch Throwable e
         (try (close-sessions! @sessions) (catch Throwable cleanup (.addSuppressed e cleanup)))
         (throw e))))))

(defn- device-observations [executable]
  (into {} (map (fn [[target session]]
                  [target {:session-id (:session-id @session)
                           :device (gpu/execution-device-info session)
                           :hardware-evidence
                           (hardware/evidence-signature
                            (hardware/descriptor-for (:device-id @session)))}]))
        (:sessions executable)))

(defn- route-context [executable devices]
  {:devices (update-vals devices #(dissoc % :session-id))
   :transport (:transport executable)
   :max-staging-bytes (:staging-bytes executable)
   :transfer-layouts
   (into {} (for [action (get-in executable [:readiness :actions])
                  :when (= :transfer (:kind action))]
              [(:id action) {:source (distributed/route-layout-description (first (:reads action)))
                            :target (distributed/route-layout-description (first (:writes action)))}]))})

(defn cost-context
  "Snapshot live physical context for explicit empirical route-cost simulation.
   Requires the original ready owner; does not execute it or admit measurements.
   Session identities are deliberately excluded so fresh owners can share matching context."
  [executable]
  (when-not (original-executable? executable)
    (throw (ex-info "route cost context requires the original ready distributed owner"
                    {:reason :distributed-runtime-owner})))
  (locking (:state executable)
    (when-not (= :ready @(:state executable))
      (throw (ex-info "route cost context requires a ready distributed owner"
                      {:reason :distributed-runtime-owner})))
    (route-context executable (device-observations executable))))

(defn- execute! [executable profile?]
  (locking (:state executable)
    (when-not (= :ready @(:state executable))
      (throw (ex-info "distributed execution is not ready" {:reason :distributed-runtime-state :state @(:state executable)})))
    (reset! (:state executable) :running)
    (try
      (let [start (when profile? (System/nanoTime))
            observed-at (when profile? {:clock :unix-epoch-ms :value (System/currentTimeMillis)})
            before (when profile? (device-observations executable))
            actions (into {} (map (juxt :id identity)) (get-in executable [:readiness :actions]))
            observations (volatile! [])
            completed (volatile! #{})]
        (doseq [{:keys [id operation waits completion]} (get-in executable [:schedule :operations])]
          (when-not (every? @completed (map :id waits))
            (throw (ex-info "distributed operation has incomplete dependencies" {:reason :distributed-runtime-waits :step id})))
          (let [step-start (when profile? (System/nanoTime))
                measurement
                (case (:kind operation)
            :compute
            (let [plan (get-in executable [:projections id :plan])
                  session (get (:sessions executable) (:target plan))
                  ids (set (map #(get-in % [:view :allocation :id]) (vals (:nodes plan))))
                  buffers (into {} (map (fn [id] [id (gpu/buffer session id)])) ids)]
              (with-open [local (link/instantiate! plan {:session session :external-buffers buffers
                                                       :profile? profile?})]
                (if profile? (link/profile! local) (do (link/run! local) nil))))
            :transfer
            (let [action (actions id)]
              (if (= :resident-copy (:transport executable))
                (let [source (first (:reads action)) target (first (:writes action))]
                  (gpu/copy-range! (get (:sessions executable) (get-in source [:allocation :device]))
                                   (physical-view (:sessions executable) source)
                                   (physical-view (:sessions executable) target)
                                   {:elements (reduce * 1 (:shape source))}))
                (transfer-host-staged! (:sessions executable) (first (:reads action))
                                       (first (:writes action)) (:staging-bytes executable) profile?
                                       (:staging executable)))))]
            (when profile?
              (vswap! observations conj
                      (cond-> {:step id :kind (:kind operation) :dependencies (:dependencies operation)
                               :host-wall-ns (- (System/nanoTime) step-start)
                               :host-timing-scope :binding-execution-and-release}
                        (= :compute (:kind operation))
                        (assoc :target (get-in executable [:projections id :plan :target])
                               :kernel-profile measurement)
                        (= :transfer (:kind operation))
                        (assoc :transport (:transport executable) :bytes (:bytes operation)
                               :route (:route operation)
                               :source-view (first (:reads (actions id)))
                               :target-view (first (:writes (actions id)))
                               :transfer-legs (if (= :host-staged (:transport executable)) measurement [])
                               :route-timing-source :host-monotonic)))))
          (vswap! completed conj (:id completion)))
        (let [after (when profile? (device-observations executable))
              _ (when (and profile? (not= before after))
                  (throw (ex-info "execution hardware identity changed during profiling"
                                  {:reason :distributed-profile-device-drift
                                   :before before :after after})))
              report (when profile?
                       {:plan (:plan executable) :execution-model :synchronous-serialized
                        :calibration? false :devices-before before
                        :observed-at observed-at :route-cost-context (route-context executable before)
                        :transport (:transport executable)
                        :max-staging-bytes (:staging-bytes executable)
                        :allocation-budgets (:allocation-budgets executable)
                        :devices-after after
                        :steps @observations :host-wall-ns (- (System/nanoTime) start)
                        :host-timing-scope :execution-with-observation-overhead})]
          (reset! (:state executable) :complete)
          (if profile? report executable)))
      (catch Throwable e (reset! (:state executable) :failed) (throw e)))))

(defn run!
  "Execute the DAG once, synchronously. Completion is recorded after each call returns.
   Failed/completed owners cannot replay stale initialization evidence."
  [executable]
  (execute! executable false))

(defn profile!
  "Execute once with existing Link kernel profiling and awaited transfer-event measurements.
   Not an extra replay: consumes the same ready owner as run!, and leaves outputs available.
   Returns a complete observation only on success. Host step times include binding/cleanup;
   resident copies have host timing only. Host-staged legs retain their backend timing sources.
   Before/after context includes the planner's hardware evidence signature, queried using the
   physical session device, not the logical worker name. Calibration drift invalidates a report
   just as live device/driver drift does; missing hardware facts remain absent.
   Logical route time is not attributed to individual topology links. This serialized execution
   observation neither proves overlap nor updates calibration or topology automatically."
  [executable]
  (when-not (original-executable? executable)
    (throw (ex-info "profiling requires the original distributed owner"
                    {:reason :distributed-runtime-owner})))
  (execute! executable true))

(defn output-values
  "Return retained compute outputs as step-id -> logical-value-id -> resident value.
   Views borrow the enclosing execution lifetime and are available only after successful completion.
   A transfer-only completion has no local logical output values."
  [executable]
  (locking (:state executable)
    (when-not (= :complete @(:state executable))
      (throw (ex-info "distributed outputs require completed execution"
                      {:reason :distributed-runtime-state :state @(:state executable)})))
    (into {}
          (map (fn [step]
                 (let [local (get-in executable [:bindings step :link-plan])]
                   [step (if-not local {}
                             (into {}
                                   (map (fn [id]
                                          (let [fields (mapv (fn [{:keys [name node]}]
                                                               {:name name :value (physical-view (:sessions executable)
                                                                                                 (get-in local [:nodes node :view]))})
                                                             (get-in local [:values id :leaves]))]
                                            [id (if (= 1 (count fields)) (:value (first fields))
                                                    (resident-value/composite id fields))])))
                                   (link-plan/output-value-ids local)))])))
          (get-in executable [:plan :outputs]))))

(defn- with-output-scope! [executable read! state-after]
  (let [values (locking (:state executable)
                 (let [values (output-values executable)]
                   (reset! (:state executable) :reading-outputs)
                   values))]
    (try
      (read! values)
      (finally
        (locking (:state executable)
          (reset! (:state executable) (state-after)))))))

(defn with-output-values!
  "Call `read!` synchronously with completed outputs while retaining the owner's lifetime.
   Close from any thread is refused while this scope is active. No owner monitor is held
   across user code, so a provider worker can observe that refusal without deadlocking. The
   callback must finish all reads (including asynchronous transfer waits) before returning.
   Resident views must not escape or be mutated, and direct session mutation/close is outside
   this contract. Return copied data, not borrowed views. This is a lifetime boundary, not a
   sealed compiler completion receipt, content verification or durable publication."
  [executable read!]
  (with-output-scope! executable read! (constantly :complete)))

(defn measure-storage-representation!
  "Observe one dtype on an original completed distributed owner's physical target.
   The exclusive output scope retains all owner sessions without holding a monitor across
   callbacks. The generated probe writes only its private temporary, never output storage.
   Its two-element allocation must fit the retained explicit resident-buffer budget; driver
   and context overhead is not modeled by that budget. No async events are admitted.
   Admission declines preserve completion; failure after entering the probe conservatively
   marks the owner failed at scope release, preventing consumption of potentially lost storage.

   Returns original owner/session-bound representation evidence. Deref is historical data;
   this is neither a program/completion receipt nor portable authentication/publication. A
   future capture must revalidate this owner, live session and its independent compiler plan."
  [executable target element-dtype]
  (when-not (original-executable? executable)
    (throw (ex-info "representation measurement requires the original distributed owner"
                    {:reason :distributed-runtime-owner})))
  (let [probe-failed? (volatile! false)]
    (with-output-scope!
     executable
     (fn [_]
       (let [session (or (get (:sessions executable) target)
                         (throw (ex-info "target is not owned by this distributed execution"
                                         {:reason :distributed-representation-target :target target})))
             dt (dtype/canon element-dtype)
             {:keys [capacity-bytes resident-bytes]} (get (:allocation-budgets executable) target)
             probe-bytes (* 2 (dtype/bytes-of dt))]
         (when-not (and (integer? capacity-bytes) (integer? resident-bytes)
                        (<= 0 resident-bytes capacity-bytes Long/MAX_VALUE)
                        (<= probe-bytes (- capacity-bytes resident-bytes)))
           (throw (ex-info "storage probe exceeds the owner's resident allocation budget"
                           {:reason :distributed-representation-budget :target target
                            :probe-bytes probe-bytes :capacity-bytes capacity-bytes
                            :resident-bytes resident-bytes})))
         (let [observation (storage/observe!
                            session dt
                            (fn [observe]
                              (try (observe)
                                   (catch Throwable error
                                     (vreset! probe-failed? true)
                                     (throw error)))))]
           (seal-runtime-value
            (->ResidentRepresentationEvidence
             (assoc (dissoc observation :session-id)
                    :kind :raster.distributed/resident-representation-v1 :target target)
             executable session (:session-id observation) nil)))))
     (fn [] (if @probe-failed? :failed :complete)))))

(defn storage-representation-description
  "Return checked representation data inside an active output read scope.
   Requires the original distributed fact for this exact owner, target, dtype, live session
   identity and current device snapshot. Returned maps are historical data, not transferable
   authority; capture must validate again in the scope that actually reads bytes."
  [executable target element-dtype fact]
  (when-not (original-executable? executable)
    (throw (ex-info "storage description requires the original distributed owner"
                    {:reason :distributed-runtime-owner})))
  (locking (:state executable)
    (when-not (= :reading-outputs @(:state executable))
      (throw (ex-info "storage description requires a pinned completed output scope"
                      {:reason :distributed-runtime-state :state @(:state executable)})))
    (let [session (get (:sessions executable) target)
          dt (dtype/canon element-dtype)]
      (when-not (and (representation-evidence? fact)
                     (identical? executable (:executable fact))
                     session (identical? session (:session fact))
                     (= (:session-id @session) (:session-id fact))
                     (not (:closed? @session)) (empty? (:events @session))
                     (= :raster.distributed/resident-representation-v1 (:kind @fact))
                     (= target (:target @fact)) (= dt (:dtype @fact))
                     (= (gpu/execution-device-info session) (:device @fact)))
        (throw (ex-info "distributed storage fact does not match this live owner frontier"
                        {:reason :distributed-representation-mismatch :target target :dtype dt})))
      @fact)))

(defn close! [executable]
  (locking (:state executable)
    (when (= :reading-outputs @(:state executable))
      (throw (ex-info "distributed output read scope retains the owner lifetime"
                      {:reason :distributed-runtime-output-scope-active})))
    (when-not (= :closed @(:state executable))
      ;; A failed close prohibits reuse but retains retryable teardown and staging lifetime.
      (reset! (:state executable) :closing)
      (close-sessions! (:sessions executable))
      (when-let [staging (:staging executable)]
        (when-let [^Arena arena (:arena @staging)]
          (when (.isAlive (.scope arena)) (.close arena)))
        (reset! staging nil))
      (reset! (:state executable) :closed)))
  nil)
