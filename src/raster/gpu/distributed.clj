(ns raster.gpu.distributed
  "Synchronous execution of checked DistributedPlans with explicit worker target placement.
   Logical workers have explicit physical targets; placement does not imply execution overlap."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.resident-value :as resident-value])
  (:import [java.lang.foreign Arena]))

(declare close!)

(defrecord DistributedExecutable [plan schedule readiness bindings projections sessions transport staging-bytes state]
  java.io.Closeable
  (close [this] (close! this)))

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
           (catch Throwable e (if-let [first @failure] (.addSuppressed ^Throwable first e)
                                  (vreset! failure e)))))
    (when-let [error @failure] (throw error))))

(defn- transfer-host-staged! [sessions source target staging-bytes]
  (let [elements (reduce * 1 (:shape source))
        element-bytes (dtype/bytes-of (:dtype source))
        chunk-elements (min elements (quot staging-bytes element-bytes))]
    (when (pos? elements)
      (with-open [arena (Arena/ofConfined)]
        (let [segment (.allocate arena (long (* chunk-elements element-bytes)) (long element-bytes))]
          (loop [offset 0]
            (when (< offset elements)
              (let [n (min chunk-elements (- elements offset))]
                (gpu/download-range! (get sessions (get-in source [:allocation :device]))
                                     (physical-view sessions source) segment {:src-element offset :elements n})
                (gpu/upload-range! (get sessions (get-in target [:allocation :device]))
                                   (physical-view sessions target) segment {:dst-element offset :elements n})
                (recur (+ offset n))))))))))

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
   Source objects must remain valid and stable through this synchronous initialization."
  ([plan] (instantiate! plan {}))
  ([plan {:keys [transport max-staging-bytes device-capacities]
          :or {max-staging-bytes 1048576 device-capacities {}}}]
   (let [ready (distributed/check-readiness plan)
         {:keys [bindings]} (distributed/compute-bindings plan)
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
         _ (when-not (map? device-capacities)
             (throw (ex-info "physical device capacities must be a map"
                             {:reason :distributed-runtime-physical-budget})))
         remapped-targets (into #{} (keep (fn [[worker local]]
                                            (let [target (get local :target worker)]
                                              (when (not= worker target) target)))) (:device-plans plan))
         projections (update-vals bindings #(link-plan/borrow-owned-storage (:link-plan %)))
         specs (reduce
                (fn [specs [_ {:keys [link-plan]}]]
                  (reduce
                   (fn [specs [_ {:keys [view]}]]
                     (let [a (:allocation view) device (:device a) id (:id a) dt (dtype/canon (:dtype view))
                           bytes (dtype/bytes-of dt) key [device id]]
                       (when-not (and (= :owned (:ownership a)) (= device (:target link-plan))
                                      (zero? (mod (:byte-size a) bytes)))
                         (throw (ex-info "distributed owner needs explicit owned device storage"
                                         {:reason :distributed-runtime-allocation :allocation a})))
                       (when-let [previous (get specs key)]
                         (when-not (= previous {:allocation a :dtype dt})
                           (throw (ex-info "one resident allocation needs one exact storage contract"
                                           {:reason :distributed-runtime-storage-contract :allocation key
                                            :previous previous :actual {:allocation a :dtype dt}}))))
                       (assoc specs key {:allocation a :dtype dt})))
                   specs (:nodes link-plan))) {} bindings)
         _ (doseq [[device entries] (group-by (comp first key) specs)]
             (let [bytes (reduce +' 0 (map (comp :byte-size :allocation val) entries))
                   capacity (get device-capacities device
                                 (when-not (contains? remapped-targets device)
                                   (get-in plan [:topology :devices device :memory-capacity-bytes])))]
               (when-not (and (integer? capacity) (<= 0 capacity Long/MAX_VALUE))
                 (throw (ex-info "remapped devices require an explicit aggregate physical budget"
                                 {:reason :distributed-runtime-physical-budget :device device :capacity capacity})))
               (when (> bytes capacity)
                 (throw (ex-info "resident allocation pool exceeds the declared device budget"
                                 {:reason :distributed-runtime-memory :device device :bytes bytes :capacity capacity})))))
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
       (->DistributedExecutable plan schedule ready bindings projections @sessions transport max-staging-bytes (atom :ready))
       (catch Throwable e
         (try (close-sessions! @sessions) (catch Throwable cleanup (.addSuppressed e cleanup)))
         (throw e))))))

(defn run!
  "Execute the DAG once, synchronously. Completion events are recorded only after each call
   returns. Local executables are bound after their waits, then closed before owner storage.
   A failed or completed execution cannot be replayed using stale initialization evidence."
  [executable]
  (locking (:state executable)
    (when-not (= :ready @(:state executable))
      (throw (ex-info "distributed execution is not ready" {:reason :distributed-runtime-state :state @(:state executable)})))
    (reset! (:state executable) :running)
    (try
      (let [actions (into {} (map (juxt :id identity)) (get-in executable [:readiness :actions]))
            completed (volatile! #{})]
        (doseq [{:keys [id operation waits completion]} (get-in executable [:schedule :operations])]
          (when-not (every? @completed (map :id waits))
            (throw (ex-info "distributed operation has incomplete dependencies" {:reason :distributed-runtime-waits :step id})))
          (case (:kind operation)
            :compute
            (let [plan (get-in executable [:projections id :plan])
                  session (get (:sessions executable) (:target plan))
                  ids (set (map #(get-in % [:view :allocation :id]) (vals (:nodes plan))))
                  buffers (into {} (map (fn [id] [id (gpu/buffer session id)])) ids)]
              (with-open [local (link/instantiate! plan {:session session :external-buffers buffers})]
                (link/run! local)))
            :transfer
            (let [action (actions id)]
              (if (= :resident-copy (:transport executable))
                (let [source (first (:reads action)) target (first (:writes action))]
                  (gpu/copy-range! (get (:sessions executable) (get-in source [:allocation :device]))
                                   (physical-view (:sessions executable) source)
                                   (physical-view (:sessions executable) target)
                                   {:elements (reduce * 1 (:shape source))}))
                (transfer-host-staged! (:sessions executable) (first (:reads action))
                                       (first (:writes action)) (:staging-bytes executable)))))
          (vswap! completed conj (:id completion)))
        (reset! (:state executable) :complete)
        executable)
      (catch Throwable e (reset! (:state executable) :failed) (throw e)))))

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

(defn close! [executable]
  (locking (:state executable)
    (when-not (= :closed @(:state executable))
      (reset! (:state executable) :closed)
      (close-sessions! (:sessions executable))))
  nil)
