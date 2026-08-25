(ns raster.gpu.link
  "Runtime instantiation of pure compiler LinkPlans.

   The plan is validated completely before a session is created or touched. Instantiation allocates
   each unique owned allocation once, imports caller-owned allocations explicitly, materializes
   node views, binds every descriptor through raster.gpu.core/bind-step!, and records one replay
   graph. Attention, GEMM, reductions and quant kernels are not special cases here."
  (:refer-clojure :exclude [run!])
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]))

(declare close! run!)

(defrecord LinkedExecutable
           [plan session owns-session? graph-key phases allocation-keys node-views pending-inputs
            closed?]
  java.io.Closeable
  (close [this] (close! this))
  clojure.lang.IFn
  (invoke [this] (run! this)))

(defn linked-executable? [x]
  (and x (= "raster.gpu.link.LinkedExecutable" (.getName (class x)))))

(defn- allocation-groups [plan]
  (group-by #(get-in % [:view :allocation :id]) (vals (:nodes plan))))

(defn- allocation-key [execution-id allocation-id]
  [::allocation execution-id allocation-id])

(defn- phase-key [execution-id instance-id step-index]
  [::phase execution-id instance-id step-index])

(defn- graph-key [execution-id]
  [::graph execution-id])

(defn- allocation-spec
  [nodes]
  (let [view (:view (first nodes))
        allocation (:allocation view)
        dt (dtype/canon (:dtype view))
        bytes (long (:byte-size allocation))
        element-bytes (long (dtype/bytes-of dt))]
    (when-not (zero? (mod bytes element-bytes))
      (throw (ex-info "linked allocation byte size is not integral in its storage dtype"
                      {:reason :link-allocation-size :allocation (:id allocation)
                       :byte-size bytes :dtype dt})))
    [dt (quot bytes element-bytes) nil]))

(defn- cleanup-attached!
  [session graph-key phases allocation-keys]
  ;; A failed backend destructor must not strand the resources that follow it. Preserve the first
  ;; failure for the caller, but attempt the complete reverse-order teardown.
  (let [failure (volatile! nil)
        attempt! (fn [f]
                   (try (f)
                        (catch Throwable error
                          (when-not @failure (vreset! failure error)))))]
    (when graph-key (attempt! #(gpu/release-recorded-graph! session graph-key)))
    (doseq [phase (reverse phases)] (attempt! #(gpu/release-prepared! session phase)))
    (doseq [key (reverse allocation-keys)] (attempt! #(gpu/free-buffer! session key)))
    (when-let [error @failure] (throw error))))

(defn instantiate!
  "Instantiate a validated LinkPlan as one replayable LinkedExecutable.

   opts:
   - `:session` attaches to an existing same-device session; otherwise the executable owns one.
   - `:external-buffers` maps every borrowed/external allocation identity to a backend DeviceBuffer.

   Attached executables own only their phases, graph recording and allocation registrations. Their
   close does not close the caller session and never frees caller-owned buffers."
  ([plan] (instantiate! plan {}))
  ([plan {:keys [session external-buffers] :or {external-buffers {}}}]
   ;; This is intentionally the first operation. Everything below may contact a backend.
   (let [plan (link-plan/validate! plan)
         target (:target plan)
         owns-session? (nil? session)
         session (or session (gpu/make-session target))
         _ (when-not (= target (:device-id @session))
             (when owns-session? (gpu/close-session! session))
             (throw (ex-info "link plan target differs from its runtime session"
                             {:reason :link-session-target :target target
                              :session-device (:device-id @session)})))
         execution-id (random-uuid)
         groups (allocation-groups plan)
         external-ids
         (into #{}
               (keep (fn [[allocation-id nodes]]
                       (when (contains? #{:borrowed :external}
                                        (get-in (first nodes) [:view :allocation :ownership]))
                         allocation-id)))
               groups)
         supplied-external-ids (set (keys external-buffers))
         allocation-keys (volatile! [])
         phases (volatile! [])
         recorded-key (volatile! nil)]
     (when-not (= external-ids supplied-external-ids)
       (when owns-session? (gpu/close-session! session))
       (throw (ex-info "external buffer bindings differ from the LinkPlan ownership contract"
                       {:reason :link-external-bindings :expected external-ids
                        :actual supplied-external-ids
                        :missing (set/difference external-ids supplied-external-ids)
                        :extra (set/difference supplied-external-ids external-ids)})))
     (try
       (let [key-by-allocation
             (into {}
                   (map (fn [[allocation-id _]]
                          [allocation-id (allocation-key execution-id allocation-id)]))
                   groups)
             owned-specs
             (into {}
                   (keep (fn [[allocation-id nodes]]
                           (when (= :owned (get-in (first nodes) [:view :allocation :ownership]))
                             [(get key-by-allocation allocation-id) (allocation-spec nodes)])))
                   groups)]
         (when (seq owned-specs)
           (gpu/alloc! session owned-specs)
           (vswap! allocation-keys into (keys owned-specs)))
         (doseq [[allocation-id nodes] groups
                 :let [allocation (get-in (first nodes) [:view :allocation])
                       ownership (:ownership allocation)]
                 :when (contains? #{:borrowed :external} ownership)]
           (let [key (get key-by-allocation allocation-id)]
             (gpu/register-buffer! session key (get external-buffers allocation-id)
                                   {:ownership ownership
                                    :allocation-id allocation-id
                                    :memory-space (:memory-space allocation)
                                    :coherence (:coherence allocation)
                                    :alignment (:alignment allocation)})
             (vswap! allocation-keys conj key)))
         (let [node-views
               (into {}
                     (map (fn [[node-id {:keys [view]}]]
                            [node-id
                             (gpu/buffer-view
                              session (get key-by-allocation (get-in view [:allocation :id]))
                              {:id (:id view) :byte-offset (:byte-offset view)
                               :dtype (:dtype view) :shape (:shape view)
                               :strides (:strides view)})]))
                     (:nodes plan))]
           (doseq [[node-id {:keys [source view]}] (:nodes plan)
                   :when source]
             (gpu/upload-range! session (get node-views node-id) source
                                {:elements (reduce * 1 (:shape view))}))
           (doseq [instance (:instances plan)
                   [step-index step] (map-indexed vector (get-in instance [:descriptor :steps]))]
             (let [phase (phase-key execution-id (:id instance) step-index)
                   bindings (:bindings instance)]
               (gpu/bind-step!
                session (assoc step :phase phase) (link-plan/instance-arguments instance)
                (fn [symbol]
                  (or (get node-views (get bindings symbol))
                      (throw (ex-info "validated link binding disappeared during instantiation"
                                      {:instance (:id instance) :symbol symbol}))))
                {:schedule (or (:schedule instance) (get-in instance [:descriptor :schedule]))
                 :roles (link-plan/instance-roles plan instance)})
               (vswap! phases conj phase)))
           (let [gkey (graph-key execution-id)]
             (gpu/record-graph! session @phases gkey)
             (vreset! recorded-key gkey)
             (->LinkedExecutable plan session owns-session? gkey @phases @allocation-keys
                                 node-views
                                 (atom (into #{}
                                             (keep (fn [[node-id {:keys [role source view]}]]
                                                     (when (and (contains? #{:input :constant :state}
                                                                           role)
                                                                (nil? source)
                                                                (= :owned (get-in view
                                                                                  [:allocation
                                                                                   :ownership])))
                                                       node-id)))
                                             (:nodes plan)))
                                 (atom false)))))
       (catch Throwable error
         (if owns-session?
           (try (gpu/close-session! session) (catch Throwable _))
           (try (cleanup-attached! session @recorded-key @phases @allocation-keys)
                (catch Throwable _)))
         (throw error))))))

(defn- ensure-live! [executable operation]
  (when-not (linked-executable? executable)
    (throw (ex-info "operation requires a LinkedExecutable"
                    {:operation operation :actual (type executable)})))
  (when @(:closed? executable)
    (throw (ex-info "linked executable is closed"
                    {:operation operation :plan (get-in executable [:plan :id])})))
  executable)

(defn node-view
  "Return a stable ResidentBufferView for one public or internal LinkNode."
  [executable node-id]
  (let [executable (ensure-live! executable :node-view)]
    (or (get (:node-views executable) node-id)
        (throw (ex-info "linked executable has no such node"
                        {:reason :link-runtime-node :node node-id
                         :nodes (set (keys (:node-views executable)))})))))

(defn outputs
  "Return the plan's ordered output node identities mapped to resident views."
  [executable]
  (let [executable (ensure-live! executable :outputs)
        output-ids (get-in executable [:plan :outputs])
        rank (zipmap output-ids (range))]
    ;; Array maps lose insertion order after their small-map threshold. A plan-ranked sorted map
    ;; keeps the declared order for model boundaries with arbitrarily many outputs.
    (into (sorted-map-by (fn [left right]
                           (let [position-order (compare (get rank left Long/MAX_VALUE)
                                                         (get rank right Long/MAX_VALUE))]
                             (if (zero? position-order)
                               (compare (pr-str left) (pr-str right))
                               position-order))))
          (map (fn [node-id] [node-id (node-view executable node-id)]))
          output-ids)))

(defn run!
  "Replay the linked graph synchronously and return its resident output views. No host copies."
  [executable]
  (let [executable (ensure-live! executable :run!)]
    (when (seq @(:pending-inputs executable))
      (throw (ex-info "linked executable has owned inputs or state that have not been initialized"
                      {:reason :link-pending-inputs :plan (get-in executable [:plan :id])
                       :nodes @(:pending-inputs executable)})))
    (gpu/replay! (:session executable) (:graph-key executable))
    (outputs executable)))

(defn upload!
  "Upload an exact contiguous host value into one live LinkNode view. Returns the executable."
  [executable node-id source]
  (let [executable (ensure-live! executable :upload!)
        node (get-in executable [:plan :nodes node-id])]
    (when-not node
      (node-view executable node-id))
    ;; Reject dtype/length mistakes before the backend copy sees them.
    (link-plan/validate-node-source! node source)
    (gpu/upload-range! (:session executable) (node-view executable node-id) source
                       {:elements (reduce * 1 (get-in node [:view :shape]))})
    (swap! (:pending-inputs executable) disj node-id)
    executable))

(defn download
  "Download one complete contiguous LinkNode view. Debug/host-boundary helper, not invocation."
  [executable node-id]
  (let [executable (ensure-live! executable :download)
        node (get-in executable [:plan :nodes node-id])
        _ (when-not node (node-view executable node-id))
        n (reduce * 1 (get-in node [:view :shape]))
        dt (get-in node [:view :dtype])
        out (case (dtype/canon dt)
              :byte (byte-array n)
              :half (short-array n)
              :int (int-array n)
              :long (long-array n)
              :float (float-array n)
              :double (double-array n))]
    (gpu/download-range! (:session executable) (node-view executable node-id) out
                         {:elements n})
    out))

(defn close!
  "Release a LinkedExecutable. Idempotent. An owned session is closed wholesale; an attached
   executable releases only its recording, phases, and allocation registrations."
  [executable]
  (when-not (linked-executable? executable)
    (throw (ex-info "close! requires a LinkedExecutable" {:actual (type executable)})))
  (when (compare-and-set! (:closed? executable) false true)
    (if (:owns-session? executable)
      (gpu/close-session! (:session executable))
      (cleanup-attached! (:session executable) (:graph-key executable)
                         (:phases executable) (:allocation-keys executable))))
  nil)
