(ns raster.gpu.link-module
  "Owned-session selective execution for validated named LinkModule entries.

   This first runtime slice accepts only static equation-first programs over owned storage,
   exact-node interfaces, and semantic complete writes. One opaque lease pins a successful
   entry generation; it is not a compiler certificate or a portable residual."
  (:refer-clojure :exclude [run!])
  (:require [clojure.set :as set]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.link-module-plan :as module]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as parallel-program]
            [raster.gpu.resource-cleanup :as cleanup]))

(declare close! close-lease! lease-value)

(deftype NamedEntryOwner [payload]
  java.io.Closeable
  (close [this] (close! this)))

(deftype EntryLease [payload]
  java.io.Closeable
  (close [this] (close-lease! this))
  clojure.lang.IDeref
  (deref [this] (lease-value this)))

(def ^:private identity-token (Object.))

(defn- opaque-payload [value]
  @(cond (instance? NamedEntryOwner value) (.-payload ^NamedEntryOwner value)
         (instance? EntryLease value) (.-payload ^EntryLease value)
         :else (atom nil)))

(defn- opaque-field [value field]
  (get (opaque-payload value) field))

(defn- sealed [kind fields]
  (let [payload (atom nil)
        value (case kind :owner (NamedEntryOwner. payload) :lease (EntryLease. payload))
        reference (volatile! nil)
        seal (fn [candidate]
               (when (identical? candidate (some-> @reference .get)) identity-token))
        fields (assoc fields :seal seal)]
    (vreset! reference (java.lang.ref.WeakReference. value))
    (reset! payload fields)
    value))

(defn- exact-owner? [value]
  (and (instance? NamedEntryOwner value)
       (ifn? (opaque-field value :seal))
       (identical? identity-token ((opaque-field value :seal) value))))

(defn- exact-lease? [value]
  (and (instance? EntryLease value)
       (ifn? (opaque-field value :seal))
       (identical? identity-token ((opaque-field value :seal) value))))

(defn- fail! [reason message data]
  (throw (ex-info message (assoc data :reason reason :runtime :link-module))))

(defn- caller-options [opts]
  (when (contains? opts :scalar-math) (select-keys opts [:scalar-math])))

(defn- value-node [spine value-id]
  (let [leaves (get-in spine [:values value-id :leaves])]
    (when-not (= 1 (count leaves))
      (fail! :link-module-runtime-exact-node
             "named entry runtime requires exact one-node logical values"
             {:value value-id :leaves leaves}))
    (:node (first leaves))))

(defn- entry-frontiers [module entry-name]
  (filterv #(= entry-name (:consumer %)) (:frontiers module)))

(defn- producer-values [module entry-name]
  (into #{}
        (keep #(when (= entry-name (:producer %)) (:value %)))
        (:retained-groups module)))

(defn- raw-order! [call request instance-id]
  (let [graph-order (fn [_ graph]
                      {:record-time-prologue []
                       :per-replay (mapv #(hash-map :kernel-phase (:id %)) (:nodes graph))})
        order (if request
                (program-call/execution-order call graph-order request)
                (program-call/execution-order call))]
    (when (seq (:record-time-prologue order))
      (fail! :link-module-runtime-prologue
             "named entry execution refuses record-time prologues"
             {:instance instance-id :record-time-prologue (:record-time-prologue order)}))
    order))

(defn- admit!
  [candidate opts]
  (when-not (and (map? opts) (set/subset? (set (keys opts)) #{:scalar-math :profile?}))
    (fail! :link-module-runtime-options "unsupported named entry runtime options"
           {:keys (when (map? opts) (set (keys opts)))}))
  (let [request (caller-options opts)
        checked (if request (module/validate! candidate request) (module/validate! candidate))
        spine (:spine checked)
        instances (into {} (map (juxt :id identity)) (:instances spine))]
    (doseq [[_ node] (:nodes spine)]
      (when-not (= :owned (get-in node [:view :allocation :ownership]))
        (fail! :link-module-runtime-ownership
               "named entry runtime initially owns every physical allocation"
               {:node (:id node) :ownership (get-in node [:view :allocation :ownership])}))
      (when (zero? (long (get-in node [:view :byte-length])))
        (fail! :link-module-runtime-empty-view
               "named entry runtime refuses zero-byte physical values"
               {:node (:id node)})))
    (when (seq (:aliases spine))
      (fail! :link-module-runtime-aliases
             "named entry runtime initially requires disjoint physical node identities"
             {:aliases (:aliases spine)}))
    (doseq [entry (:entries checked)]
      (when (seq (get-in entry [:interface :state]))
        (fail! :link-module-runtime-state
               "named entry runtime does not yet expose mutable state interfaces"
               {:entry (:name entry) :state (get-in entry [:interface :state])}))
      (doseq [value-id (concat (get-in entry [:interface :inputs])
                               (get-in entry [:interface :outputs]))]
        (value-node spine value-id))
      (let [{:keys [writes full-produces]} (:contract entry)]
        (when-not (set/subset? writes full-produces)
          (fail! :link-module-runtime-complete-write
                 "every named entry write must be a semantic complete result"
                 {:entry (:name entry) :writes writes :full-produces full-produces})))
      (doseq [value-id (into (producer-values checked (:name entry))
                             (get-in entry [:interface :outputs]))
              :let [node-id (value-node spine value-id)]]
        (when-not (and (contains? (get-in entry [:contract :writes]) node-id)
                       (contains? (get-in entry [:contract :full-produces]) node-id))
          (fail! :link-module-runtime-direct-production
                 "leased entry values require exact-node semantic production"
                 {:entry (:name entry) :value value-id :node node-id
                  :writes (get-in entry [:contract :writes])
                  :full-produces (get-in entry [:contract :full-produces])})))
      (doseq [instance-id (:instances entry)
              :let [instance (get instances instance-id)]]
        (when-not (and (link-plan/program-link-instance? instance)
                       (parallel-program/straight-line-call? (:call instance)))
          (fail! :link-module-runtime-program
                 "named entries require static equation-first program instances"
                 {:entry (:name entry) :instance instance-id :actual (type instance)}))
        (raw-order! (:call instance) request instance-id)))
    (doseq [{:keys [value nodes producer]} (:frontiers checked)]
      (when-not (and (= 1 (count nodes)) (= (first nodes) (value-node spine value)))
        (fail! :link-module-runtime-exact-node
               "named entry frontiers must be exact one-node values"
               {:value value :nodes nodes}))
      (let [producer-entry (some #(when (= producer (:name %)) %) (:entries checked))]
        (when-not (contains? (get-in producer-entry [:contract :full-produces]) (first nodes))
          (fail! :link-module-runtime-frontier-production
                 "a runtime frontier requires its exact node as a semantic producer result"
                 {:value value :producer producer :node (first nodes)}))))
    (doseq [{:keys [value nodes consumer]} (:frontiers checked)
            :let [consumer-entry (some #(when (= consumer (:name %)) %) (:entries checked))]]
      (when-not (contains? (get-in consumer-entry [:contract :requires]) (first nodes))
        (fail! :link-module-runtime-direct-requirement
               "a runtime frontier consumer must directly require its exact node"
               {:value value :consumer consumer :node (first nodes)
                :requires (get-in consumer-entry [:contract :requires])})))
    {:module checked :caller-options request :profile? (boolean (:profile? opts))}))

(defn- overlaps-any? [spine node-id node-ids]
  (let [view (get-in spine [:nodes node-id :view])]
    (boolean (some #(bview/overlaps? view (get-in spine [:nodes % :view])) node-ids))))

(defn- assert-owner! [owner operation]
  (when-not (exact-owner? owner)
    (fail! :link-module-runtime-owner "operation requires the exact named entry owner"
           {:operation operation :actual (type owner)}))
  (let [{:keys [closed? failure]} @(opaque-field owner :state)]
    (when closed?
      (fail! :link-module-runtime-closed "named entry owner is closed" {:operation operation}))
    (when failure
      (throw (ex-info "named entry execution previously failed"
                      {:reason :link-module-runtime-poisoned :operation operation}
                      failure))))
  (cleanup/assert-live! (opaque-field owner :cleanup))
  owner)

(defn- assert-idle! [owner operation]
  (when (:active? @(opaque-field owner :state))
    (fail! :link-module-runtime-active
           "named entry owner rejects reentrant lifecycle or execution operations"
           {:operation operation})))

(defn- entry! [owner entry-name]
  (or (some #(when (= entry-name (:name %)) %) (get-in (opaque-field owner :module) [:entries]))
      (fail! :link-module-runtime-entry "unknown named entry"
             {:entry entry-name :entries (mapv :name (get-in (opaque-field owner :module) [:entries]))})))

(defn- registered-lease-info! [owner lease]
  (when-not (exact-lease? lease)
    (fail! :link-module-runtime-lease "dependency requires an opaque entry lease"
           {:actual (type lease)}))
  (when-not (identical? owner (opaque-field lease :owner))
    (fail! :link-module-runtime-lease-owner "entry lease belongs to another owner" {}))
  (when @(opaque-field lease :released?)
    (fail! :link-module-runtime-lease-released "entry lease was released" {}))
  (let [info (get-in @(opaque-field owner :state)
                     [:leases (opaque-field lease :token)])]
    (when-not (and info (identical? lease (:lease info)) (= :live (:phase info)))
      (fail! :link-module-runtime-lease-generation
             "entry lease is not registered in this owner generation" {}))
    info))

(defn- validate-dependencies! [owner entry dependencies]
  (when-not (map? dependencies)
    (fail! :link-module-runtime-dependencies "entry dependencies must be a value-to-lease map"
           {:actual (type dependencies)}))
  (let [frontiers (entry-frontiers (opaque-field owner :module) (:name entry))
        expected (set (map :value frontiers))]
    (when-not (= expected (set (keys dependencies)))
      (fail! :link-module-runtime-dependencies
             "entry dependencies must exactly cover derived frontier requirements"
             {:entry (:name entry) :expected expected :actual (set (keys dependencies))}))
    (doseq [{:keys [value producer nodes]} frontiers
            :let [info (registered-lease-info! owner (get dependencies value))]]
      (when-not (and (= producer (:entry info)) (contains? (:values info) value))
        (fail! :link-module-runtime-dependency-producer
               "entry dependency lease does not pin the required producer frontier"
               {:entry (:name entry) :value value :producer producer
                :lease-entry (:entry info) :lease-values (:values info)}))
      (doseq [node-id nodes]
        (when-not (= (get (:generations info) node-id)
                     (get-in @(opaque-field owner :state) [:generations node-id]))
          (fail! :link-module-runtime-dependency-generation
                 "entry dependency no longer names the resident frontier generation"
                 {:entry (:name entry) :value value :node node-id}))))
    frontiers))

(defn- input-bindings! [owner entry inputs]
  (when-not (map? inputs)
    (fail! :link-module-runtime-inputs "entry inputs must be a value-to-host-buffer map"
           {:actual (type inputs)}))
  (let [spine (get-in (opaque-field owner :module) [:spine])
        expected (set (get-in entry [:interface :inputs]))]
    (when-not (= expected (set (keys inputs)))
      (fail! :link-module-runtime-inputs "entry inputs must exactly match its public interface"
             {:entry (:name entry) :expected expected :actual (set (keys inputs))}))
    (mapv (fn [value-id]
            (let [node-id (value-node spine value-id)
                  source (get inputs value-id)]
              (when (nil? source)
                (fail! :link-module-runtime-input-source
                       "entry inputs require concrete host buffers"
                       {:entry (:name entry) :value value-id :node node-id}))
              (link-plan/validate-node-source! (get-in spine [:nodes node-id]) source)
              {:value value-id :node node-id :source source}))
          (get-in entry [:interface :inputs]))))

(defn- live-pinned-nodes [state]
  (into #{} (mapcat :nodes) (filter #(= :live (:phase %)) (vals (:leases state)))))

(defn- assert-no-leased-write! [owner node-ids]
  (let [spine (get-in (opaque-field owner :module) [:spine])
        pinned (live-pinned-nodes @(opaque-field owner :state))]
    (when-let [overlap (seq (filter #(overlaps-any? spine % pinned) node-ids))]
      (fail! :link-module-runtime-lease-active
             "a live entry lease prevents an overlapping upload or replay"
             {:nodes (set overlap)}))))

(defn- poison! [owner error tainted]
  (swap! (opaque-field owner :state)
         (fn [state]
           (-> state
               (update :tainted into tainted)
               (assoc :failure (or (:failure state) error)))))
  error)

(defn- publish-nodes! [owner node-ids]
  (let [affected (set node-ids)]
    (swap! (opaque-field owner :state)
           (fn [state]
             (let [generation (inc (:epoch state))]
               (-> state
                   (assoc :epoch generation)
                   (update :initialized into affected)
                   (update :tainted #(apply disj % affected))
                   (update :generations #(reduce (fn [m node] (assoc m node generation))
                                                 % affected))))))))

(defn- invalidate-nodes! [owner node-ids]
  (let [affected (set node-ids)]
    (swap! (opaque-field owner :state)
           #(-> % (update :initialized (fn [ready] (apply disj ready affected)))
                (update :tainted into affected)))
    affected))

(defn- remove-registration! [owner lease]
  (let [token (opaque-field lease :token)]
    (swap! (opaque-field owner :state)
           (fn [state]
             (let [registered (get-in state [:leases token :lease])]
               (when-not (identical? lease registered)
                 (fail! :link-module-runtime-lease-generation
                        "entry lease registration changed identity" {}))
               (update state :leases dissoc token))))))

(defn- publish-lease! [owner entry pin-values]
  (let [spine (get-in (opaque-field owner :module) [:spine])
        nodes (into #{} (map #(value-node spine %)) pin-values)
        generations (select-keys (:generations @(opaque-field owner :state)) nodes)]
    (when-not (= nodes (set (keys generations)))
      (fail! :link-module-runtime-output-readiness
             "successful entry did not publish every leased output/frontier"
             {:entry (:name entry) :nodes nodes :generations (set (keys generations))}))
    ;; The owner lock and :active? guard pin the session through synchronous execution. Acquire
    ;; the long-lived root use only after success. Until the opaque lease is published, retain its
    ;; cleanup in owner state so every issuance failure remains reachable by owner teardown.
    (let [token (Object.)
          root-use
          (try
            (cleanup/retain-use! (opaque-field owner :cleanup))
            (catch Throwable primary
              ;; Common Cleanup reports the exact unresolved owner when acquisition rollback
              ;; itself cannot finish. Adopt only that namespaced receipt, never arbitrary
              ;; backend exception data.
              (when-let [unresolved (::cleanup/unresolved (ex-data primary))]
                (try
                  (swap! (opaque-field owner :state)
                         assoc-in [:issuance-debts token] unresolved)
                  (catch Throwable secondary
                    (when-not (identical? primary secondary)
                      (.addSuppressed ^Throwable primary secondary)))))
              (throw primary)))
          lease-cleanup (::cleanup/owner root-use)]
      (try
        (swap! (opaque-field owner :state) assoc-in [:issuance-debts token] lease-cleanup)
        (let [lease (sealed :lease {:owner owner :token token :cleanup lease-cleanup
                                    :released? (atom false)})]
          (swap! (opaque-field owner :state)
                 (fn [state]
                   (-> state
                       (update :issuance-debts dissoc token)
                       (assoc-in [:leases token]
                                 {:lease lease :phase :live :entry (:name entry)
                                  :values pin-values :nodes nodes
                                  :generations generations}))))
          lease)
        (catch Throwable primary
          ;; A private state observer may throw after an atom update. Reclassify either an
          ;; in-flight debt or a just-published lease back to owner-owned issuance debt before
          ;; rollback, so the caller never needs an unpublished token to retry cleanup.
          (swap! (opaque-field owner :state)
                 (fn [state]
                   (-> state
                       (update :leases dissoc token)
                       (assoc-in [:issuance-debts token] lease-cleanup))))
          (try
            (cleanup/release! lease-cleanup)
            (swap! (opaque-field owner :state) update :issuance-debts dissoc token)
            (catch Throwable secondary
              (when-not (identical? primary secondary)
                (.addSuppressed ^Throwable primary secondary))))
          (throw primary))))))

(defn- ordered-output-map [output-ids resolve-value]
  ;; array-map preserves this caller-declared projection order without installing a partial
  ;; comparator that could throw if a consumer later extends the immutable map.
  (apply array-map (mapcat (fn [value-id] [value-id (resolve-value value-id)]) output-ids)))

(defn- release-issuance-debts! [owner]
  (doseq [[token debt] (:issuance-debts @(opaque-field owner :state))]
    (cleanup/release! debt)
    (swap! (opaque-field owner :state) update :issuance-debts dissoc token)))

(defn instantiate!
  "Instantiate one validated named-entry module in a private owned session."
  ([candidate] (instantiate! candidate {}))
  ([candidate opts]
   (link-plan/without-validation-context
    (fn []
      (let [{:keys [module caller-options profile?]} (admit! candidate opts)
            spine (:spine module)
            session (gpu/make-session (:target spine))
            prepared-slots (into {} (map (fn [entry] [(:name entry) (volatile! nil)]))
                                 (:entries module))
            entry-resource-ids (mapv #(vector :entry (:name %)) (:entries module))
            owner-cleanup
            (cleanup/owner
             (conj
              (mapv (fn [entry]
                      {:id [:entry (:name entry)]
                       :release #(when-let [prepared @(get prepared-slots (:name entry))]
                                   (parallel-program/release-prepared! prepared))})
                    (:entries module))
              {:id :session :after (set entry-resource-ids)
               :release #(gpu/close-session! session)}))
            allocation-keys (volatile! [])
            timings (volatile! {})]
        (:value (cleanup/build!
         owner-cleanup
         (fn []
           (let [{:keys [node-views]}
                 (link/realize-plan-storage!
                  spine session (random-uuid) (link/allocation-groups spine) {}
                  allocation-keys timings)
                 executor (link/resident-program-executor
                           spine session (random-uuid) node-views false profile? allocation-keys)
                 instances (into {} (map (juxt :id identity)) (:instances spine))]
             (doseq [entry (:entries module)]
               (let [calls (mapv (fn [instance-id]
                                   {:id instance-id :kind :program
                                    :call (:call (get instances instance-id))})
                                 (:instances entry))
                     prepared (if caller-options
                                (parallel-program/prepare-sequence-with!
                                 calls executor caller-options)
                                (parallel-program/prepare-sequence-with! calls executor))
                     _ (vreset! (get prepared-slots (:name entry)) prepared)
                     order (parallel-program/execution-order
                            prepared #(gpu/kernel-graph-execution-order session %))]
                 (when (seq (:record-time-prologue order))
                   (fail! :link-module-runtime-prologue-changed
                          "bound entry execution order gained a record-time prologue"
                          {:entry (:name entry)
                           :record-time-prologue (:record-time-prologue order)}))))
             (let [direct-source-nodes
                   (into #{} (keep (fn [[id node]] (when (:source node) id))) (:nodes spine))
                   source-nodes direct-source-nodes
                   initial-generations (zipmap source-nodes (repeat 0))
                   owner
                   (sealed
                    :owner
                    {:module module :session session
                     :prepared (into {} (map (fn [[name slot]] [name @slot])) prepared-slots)
                     :node-views node-views :cleanup owner-cleanup
                     :state (atom {:closed? false :active? false :failure nil :epoch 0
                                   :initialized source-nodes :tainted #{}
                                   :generations initial-generations :leases {}
                                   :issuance-debts {}})
                     :lock (Object.) :caller-options caller-options})]
               {:value owner})))
         nil)))))))

(defn run-and-lease!
  "Upload exact public inputs, selectively run one entry, and lease its resident generation."
  [owner entry-name {:keys [inputs dependencies] :or {inputs {} dependencies {}} :as request}]
  (when-not (and (map? request) (set/subset? (set (keys request)) #{:inputs :dependencies}))
    (fail! :link-module-runtime-request "unsupported named entry request fields"
           {:keys (when (map? request) (set (keys request)))}))
  (assert-owner! owner :run-and-lease!)
  (locking (opaque-field owner :lock)
    (assert-owner! owner :run-and-lease!)
    (assert-idle! owner :run-and-lease!)
    (let [entry (entry! owner entry-name)
          _ (validate-dependencies! owner entry dependencies)
          input-bindings (input-bindings! owner entry inputs)
          input-nodes (set (map :node input-bindings))
          write-nodes (set (get-in entry [:contract :writes]))
          mutation-nodes (into input-nodes write-nodes)
          _ (assert-no-leased-write! owner mutation-nodes)
          required (set (get-in entry [:contract :requires]))
          prospectively-ready
          (into (:initialized @(opaque-field owner :state)) input-nodes)
          _ (when-not (set/subset? required prospectively-ready)
              (fail! :link-module-runtime-readiness
                     "entry has resident requirements that are not initialized"
                     {:entry entry-name :required required
                      :missing (set/difference required prospectively-ready)}))
          pin-values (into (producer-values (opaque-field owner :module) entry-name)
                           (get-in entry [:interface :outputs]))]
      (swap! (opaque-field owner :state) assoc :active? true)
      (let [backend-started? (volatile! false)]
       (try
        (try
          (doseq [{:keys [node source]} input-bindings]
            (invalidate-nodes! owner #{node})
            (vreset! backend-started? true)
            (try
              (gpu/upload-range! (opaque-field owner :session)
                                 (get (opaque-field owner :node-views) node) source
                                 {:elements (reduce * 1
                                                    (get-in (opaque-field owner :module)
                                                            [:spine :nodes node :view :shape]))})
              (publish-nodes! owner #{node})
              (catch Throwable error
                (throw (poison! owner error #{node})))))
          (let [tainted (invalidate-nodes! owner write-nodes)]
            (vreset! backend-started? true)
            (try
              (parallel-program/run-prepared!
               (get (opaque-field owner :prepared) entry-name))
              (publish-nodes! owner write-nodes)
              (catch Throwable error (throw (poison! owner error tainted)))))
          (publish-lease! owner entry pin-values)
          (catch Throwable error
            (when (and @backend-started?
                       (nil? (:failure @(opaque-field owner :state))))
              (poison! owner error #{}))
            (throw error)))
        (finally (swap! (opaque-field owner :state) assoc :active? false)))))))

(defn lease-value [lease]
  (when-not (exact-lease? lease)
    (fail! :link-module-runtime-lease "deref requires an opaque entry lease"
           {:actual (type lease)}))
  (let [owner (opaque-field lease :owner)]
    (locking (opaque-field owner :lock)
      (assert-owner! owner :lease-value)
      (assert-idle! owner :lease-value)
      (let [info (registered-lease-info! owner lease)
            spine (get-in (opaque-field owner :module) [:spine])
            output-ids (get-in (entry! owner (:entry info)) [:interface :outputs])]
        {:entry (:entry info)
         :outputs
         (ordered-output-map
          output-ids
          #(get (opaque-field owner :node-views) (value-node spine %)))}))))

(defn close-lease! [lease]
  (when-not (exact-lease? lease)
    (fail! :link-module-runtime-lease "close requires an opaque entry lease"
           {:actual (type lease)}))
  (let [owner (opaque-field lease :owner)]
    (locking (opaque-field owner :lock)
      (when-not @(opaque-field lease :released?)
        (assert-idle! owner :close-lease!)
        (registered-lease-info! owner lease)
        (cleanup/release! (opaque-field lease :cleanup))
        (remove-registration! owner lease)
        (reset! (opaque-field lease :released?) true))))
  nil)

(defn close! [owner]
  (when-not (exact-owner? owner)
    (fail! :link-module-runtime-owner "close requires the exact named entry owner"
           {:actual (type owner)}))
  (locking (opaque-field owner :lock)
    (when-not (:closed? @(opaque-field owner :state))
      (assert-idle! owner :close!)
      (when (seq (:leases @(opaque-field owner :state)))
        (fail! :link-module-runtime-lease-active
               "close requires every entry lease to be released"
               {:leases (count (:leases @(opaque-field owner :state)))}))
      (swap! (opaque-field owner :state) assoc :closed? true))
    ;; Issuance debt owns retained root uses but was never exposed as a lease. Drain it before
    ;; releasing the root; retry-safe failures remain in this owner-local table for close retry.
    (release-issuance-debts! owner)
    ;; Retry only cleanup outcomes explicitly classified retry-safe by the common owner.
    (cleanup/release! (opaque-field owner :cleanup)))
  nil)
