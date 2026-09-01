(ns raster.gpu.link
  "Runtime instantiation of pure compiler LinkPlans.

   The plan is validated completely before a session is created or touched. Instantiation allocates
   each unique owned allocation once, imports caller-owned allocations explicitly, materializes
   node views, and binds either descriptor instances through raster.gpu.core/bind-step! or one
   equation-first emitted program through reusable KernelGraphs. Attention, GEMM, reductions and
   quant kernels are not special cases here."
  (:refer-clojure :exclude [run!])
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.parallel-program :as parallel-program]
            [raster.gpu.resident-value :as resident-value]
            [raster.gpu.value :as value]))

(declare close! run!)

(defrecord LinkedExecutable
           [plan session owns-session? graph-key phases prepared-program allocation-keys node-views
            pending-inputs profile? closed?]
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
    [dt (quot bytes element-bytes) nil
     {:allocation-id (:id allocation)
      :memory-space (:memory-space allocation)
      :coherence (:coherence allocation)
      :alignment (:alignment allocation)}]))

(defn- resident-link-value [plan node-views value-id]
  (let [link-value (get-in plan [:values value-id])
        fields (mapv (fn [{:keys [name node]}]
                       {:name name :value (get node-views node)})
                     (:leaves link-value))]
    (when-not (and link-value (every? :value fields))
      (throw (ex-info "validated link value disappeared during instantiation"
                      {:reason :link-runtime-value :value value-id})))
    (if (= 1 (count fields))
      (:value (first fields))
      (resident-value/composite value-id fields))))

(defn- cleanup-attached!
  [session graph-key phases prepared-program allocation-keys]
  ;; A failed backend destructor must not strand the resources that follow it. Preserve the first
  ;; failure for the caller, but attempt the complete reverse-order teardown.
  (let [failure (volatile! nil)
        attempt! (fn [f]
                   (try (f)
                        (catch Throwable error
                          (when-not @failure (vreset! failure error)))))]
    (when graph-key (attempt! #(gpu/release-recorded-graph! session graph-key)))
    (doseq [phase (reverse phases)] (attempt! #(gpu/release-prepared! session phase)))
    (when prepared-program (attempt! #(parallel-program/release-prepared! prepared-program)))
    (doseq [key (reverse allocation-keys)] (attempt! #(gpu/free-buffer! session key)))
    (when-let [error @failure] (throw error))))

(defn instantiate!
  "Instantiate a validated LinkPlan as one replayable LinkedExecutable.

   opts:
   - `:session` attaches to an existing same-device session; otherwise the executable owns one.
   - `:external-buffers` maps every borrowed/external allocation identity to a backend DeviceBuffer.

   Attached executables own only their phases or prepared equation graphs, graph recording and
   allocation registrations. Their close does not close the caller session and never frees
   caller-owned buffers. The initial equation-first runtime accepts one ProgramLinkInstance; mixed
   descriptor/program scheduling remains a fail-loud compiler boundary."
  ([plan] (instantiate! plan {}))
  ([plan {:keys [session external-buffers profile?] :or {external-buffers {} profile? false}}]
   ;; This is intentionally the first operation. Everything below may contact a backend.
   (let [plan (link-plan/validate! plan)
         program-instances (filterv link-plan/program-link-instance? (:instances plan))
         _ (when (and (seq program-instances) (not= 1 (count (:instances plan))))
             (throw (ex-info
                     "runtime composition of emitted programs with other instances is not yet scheduled"
                     {:reason :link-runtime-mixed-program-instances
                      :instances (mapv :id (:instances plan))
                      :program-instances (mapv :id program-instances)})))
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
         prepared-program (volatile! nil)
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
           (if-let [instance (first program-instances)]
             (vreset!
              prepared-program
              (parallel-program/prepare-with!
               (:call instance)
               {:bind! (fn [key graph buffers scalars]
                         (let [resident-buffers
                               (into {}
                                     (map (fn [[compiler-value value-id]]
                                            [compiler-value
                                             (resident-link-value plan node-views value-id)]))
                                     buffers)
                               extent-scalars
                               (into {}
                                     (keep (fn [[compiler-value resident]]
                                             (when-let [extent (first (get-in resident [:view :shape]))]
                                               [(list 'extent compiler-value)
                                                {:type :long :value extent}])))
                                     resident-buffers)]
                           (gpu/bind-kernel-graph!
                            session [::program-graph execution-id key] graph
                            resident-buffers (merge extent-scalars scalars)
                            {:profile? profile?})))
                :run! #(gpu/run-kernel-graph! session %)
                :release! #(gpu/release-kernel-graph! session %)}))
             (do
               (doseq [instance (:instances plan)
                       [step-index step]
                       (map-indexed vector (get-in instance [:descriptor :steps]))]
                 (let [phase (phase-key execution-id (:id instance) step-index)
                       bindings (:bindings instance)]
                   (gpu/bind-step!
                    session (assoc step :phase phase) (link-plan/instance-arguments instance)
                    (fn [symbol]
                      (let [value-id (get bindings symbol)]
                        (try (resident-link-value plan node-views value-id)
                             (catch clojure.lang.ExceptionInfo error
                               (throw (ex-info (.getMessage error)
                                               (assoc (ex-data error)
                                                      :instance (:id instance) :symbol symbol)
                                               error))))))
                    {:schedule (or (:schedule instance)
                                   (get-in instance [:descriptor :schedule]))
                     :roles (link-plan/instance-roles plan instance)})
                   (vswap! phases conj phase)))
               (let [gkey (graph-key execution-id)]
                 (gpu/record-graph! session @phases gkey {:profile? profile?})
                 (vreset! recorded-key gkey))))
           (->LinkedExecutable plan session owns-session? @recorded-key @phases @prepared-program
                               @allocation-keys node-views
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
                               (boolean profile?)
                               (atom false))))
       (catch Throwable error
         (if owns-session?
           (try (gpu/close-session! session) (catch Throwable _))
           (try (cleanup-attached! session @recorded-key @phases @prepared-program
                                   @allocation-keys)
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

(defn value-view
  "Return one logical resident value. Dense values return their ResidentBufferView; composite
   values return an ordered ResidentComposite whose field identities match the certified ABI."
  [executable value-id]
  (let [executable (ensure-live! executable :value-view)]
    (resident-link-value (:plan executable) (:node-views executable) value-id)))

(defn- select-instance
  [plan selector]
  (let [instances (:instances plan)
        selected
        (cond
          (nil? selector)
          (when (= 1 (count instances)) (first instances))

          (integer? selector)
          (when (<= 0 (long selector) (dec (count instances)))
            (nth instances (long selector)))

          :else
          (first (filter #(= selector (:id %)) instances)))]
    (or selected
        (throw
         (ex-info
          (if (nil? selector)
            "a composed executable requires an explicit instance selector"
            "linked executable instance selector did not match")
          {:reason (if (nil? selector)
                     :ambiguous-linked-instance
                     :linked-instance-not-found)
           :selector selector
           :instances (mapv :id instances)})))))

(defn- select-dispatch-step
  [descriptor selector]
  (let [steps (:steps descriptor)
        indexed (mapv vector (range) steps)
        selected
        (cond
          (nil? selector)
          (let [matches (filterv (fn [[_ step]] (some? (:dispatch step))) indexed)]
            (when (= 1 (count matches)) (first matches)))

          (integer? selector)
          (when (<= 0 (long selector) (dec (count steps)))
            [(long selector) (nth steps (long selector))])

          (keyword? selector)
          (first (filterv (fn [[_ step]] (= selector (:phase step))) indexed))

          :else
          (throw (ex-info "linked dispatch step selector must be nil, an index, or a phase keyword"
                          {:reason :invalid-linked-dispatch-step-selector
                           :selector selector})))]
    (when-not selected
      (throw (ex-info (if (nil? selector)
                        "a linked instance requires exactly one dispatch step when the step is omitted"
                        "linked dispatch step selector did not match")
                      {:reason (if (nil? selector)
                                 :ambiguous-linked-dispatch-step
                                 :linked-dispatch-step-not-found)
                       :selector selector
                       :dispatch-phases
                       (mapv (comp :phase second)
                             (filterv (fn [[_ step]] (some? (:dispatch step))) indexed))})))
    (let [[index step] selected]
      (when-not (:dispatch step)
        (throw (ex-info "selected linked step has no KernelDispatch"
                        {:reason :linked-step-has-no-dispatch
                         :selector selector :index index :phase (:phase step)})))
      {:step-index index :step step :dispatch (kdispatch/validate! (:dispatch step))})))

(defn linked-dispatch
  "Return one KernelDispatch from a live certified executable.

   `instance-selector` is nil for a one-instance plan, a zero-based instance index, or a stable
   LinkInstance id. `step-selector` is nil when that instance has exactly one dispatch, a
   zero-based step index, or a phase keyword. Composed plans require an explicit instance so
   compiler inspection never falls back to session registry or symbol-name inference."
  ([executable]
   (linked-dispatch executable nil nil))
  ([executable step-selector]
   (linked-dispatch executable nil step-selector))
  ([executable instance-selector step-selector]
   (let [executable (ensure-live! executable :linked-dispatch)
         instance (select-instance (:plan executable) instance-selector)]
     (when (link-plan/program-link-instance? instance)
       (throw (ex-info "equation-first program instances do not expose one descriptor dispatch"
                       {:reason :linked-program-has-no-descriptor-dispatch
                        :instance (:id instance)})))
     (assoc (select-dispatch-step (:descriptor instance) step-selector)
            :instance-id (:id instance)
            :descriptor (:descriptor instance)))))

(defn- physical-step-arguments
  [step logical-arguments]
  (if-not (:logical-bindings? step)
    logical-arguments
    (vec
     (mapcat (fn [spec value]
               (if (= :scalar (:kind spec))
                 [value]
                 (let [slots (:slots spec)]
                   (if (resident-value/resident-composite? value)
                     (let [fields (:fields value)]
                       (when-not (= (count slots) (count fields))
                         (throw (ex-info "linked dispatch composite differs from its ABI"
                                         {:reason :linked-dispatch-composite-count
                                          :phase (:phase step) :binding (:binding spec)
                                          :slots slots :fields fields})))
                       (mapv (fn [slot field]
                               (let [view (:view (:value field))]
                                 (when (and (:field slot)
                                            (not= (:field slot) (:name field)))
                                   (throw
                                    (ex-info "linked dispatch composite field order differs from its ABI"
                                             {:reason :linked-dispatch-composite-field
                                              :phase (:phase step) :slot slot :field field})))
                                 (when-not (= (dtype/canon (:dtype slot))
                                              (dtype/canon (:dtype view)))
                                   (throw
                                    (ex-info "linked dispatch composite dtype differs from its ABI"
                                             {:reason :linked-dispatch-composite-dtype
                                              :phase (:phase step) :slot slot :field field})))
                                 (:value field)))
                             slots fields))
                     (do
                       (when-not (= 1 (count slots))
                         (throw (ex-info
                                 "linked dispatch multi-slot binding requires a composite value"
                                 {:reason :linked-dispatch-multiview-binding
                                  :phase (:phase step) :binding (:binding spec) :slots slots})))
                       [value])))))
             (:argument-specs step) logical-arguments))))

(defn- resident-fields [value]
  (if (resident-value/resident-composite? value)
    (:fields value)
    [{:name :value :value value}]))

(defn dispatch-arguments
  "Project a tuning sample onto one linked instance's resident views and physical kernel ABI.

   `descriptor-arguments` follow the selected descriptor's `:all-params` order. Array values remain
   host-side reference inputs; device arguments come only from the instance's certified bindings.
   Samples may use shorter arrays or smaller dynamic scratch extents than the allocated node view,
   but may never exceed it."
  ([executable descriptor-arguments]
   (dispatch-arguments executable nil nil descriptor-arguments))
  ([executable step-selector descriptor-arguments]
   (dispatch-arguments executable nil step-selector descriptor-arguments))
  ([executable instance-selector step-selector descriptor-arguments]
   (let [executable (ensure-live! executable :dispatch-arguments)
         plan (:plan executable)
         instance (select-instance plan instance-selector)
         descriptor (:descriptor instance)
         {:keys [step-index step dispatch] :as selected}
         (select-dispatch-step descriptor step-selector)
         all-params (:all-params descriptor)
         _argument-count
         (when-not (and (sequential? descriptor-arguments)
                        (= (count all-params) (count descriptor-arguments)))
           (throw (ex-info "linked dispatch arguments must follow descriptor :all-params"
                           {:reason :linked-dispatch-argument-count
                            :instance (:id instance)
                            :expected (count all-params)
                            :actual (when (sequential? descriptor-arguments)
                                      (count descriptor-arguments))
                            :all-params all-params})))
         argmap (zipmap all-params descriptor-arguments)
         bindings (:bindings instance)
         resident-of
         (fn [sym]
           (let [value-id (get bindings sym ::missing)]
             (when (= ::missing value-id)
               (throw (ex-info "linked dispatch symbol has no certified value binding"
                               {:reason :linked-dispatch-binding
                                :instance (:id instance) :symbol sym})))
             (if (contains? (get-in executable [:plan :values]) value-id)
               (value-view executable value-id)
               ;; Compatibility for synthetic dispatch fixtures predating LinkValue. A validated
               ;; production plan always takes the branch above.
               (node-view executable value-id))))
         require-capacity!
         (fn [sym field resident required]
           (let [view (:view resident)
                 capacity (quot (:byte-length view) (dtype/bytes-of (:dtype view)))]
             (when (> (long required) (long capacity))
               (throw (ex-info "linked dispatch sample exceeds its resident node view"
                               {:reason :linked-dispatch-buffer-capacity
                                :instance (:id instance) :step-index step-index
                                :phase (:phase step) :symbol sym :field field
                                :required-elements (long required)
                                :capacity-elements (long capacity)})))))
         _array-capacities
         (doseq [sym (:array-params descriptor)]
           (let [argument (get argmap sym)
                 spec (get-in descriptor [:value-specs sym])
                 fields (resident-fields (resident-of sym))]
             (if spec
               (let [expected (mapv :field (:leaves spec))]
                 (when-not (and (map? argument) (= (set expected) (set (keys argument)))
                                (= expected (mapv :name fields)))
                   (throw (ex-info "linked dispatch composite parameter differs from its value spec"
                                   {:reason :linked-dispatch-composite-argument
                                    :instance (:id instance) :symbol sym
                                    :expected expected :argument argument :fields fields})))
                 (doseq [[field resident] (map vector expected (map :value fields))
                         :let [array (get argument field)]]
                   (when-not (and array (.isArray (class array)))
                     (throw (ex-info "linked dispatch composite field must be a JVM array"
                                     {:reason :linked-dispatch-array-argument
                                      :instance (:id instance) :symbol sym :field field
                                      :value-type (some-> array class)})))
                   (require-capacity! sym field resident
                                      (java.lang.reflect.Array/getLength array))))
               (do
                 (when-not (and argument (.isArray (class argument)))
                   (throw (ex-info "linked dispatch array parameter must be a JVM array"
                                   {:reason :linked-dispatch-array-argument
                                    :instance (:id instance) :symbol sym
                                    :value-type (some-> argument class)})))
                 (require-capacity! sym :value (:value (first fields))
                                    (java.lang.reflect.Array/getLength argument))))))
         _scratch-capacities
         (doseq [{:keys [sym] :as allocation} (:allocs descriptor)
                 :let [fields (resident-fields (resident-of sym))
                       specs (link-plan/descriptor-allocation-leaves
                              allocation (:dtype descriptor))]
                 [spec field] (map vector specs fields)]
           (when-not (= (:field spec) (:name field))
             (throw (ex-info "linked dispatch allocation fields differ from the resident value"
                             {:reason :linked-dispatch-allocation-field :symbol sym
                              :expected (:field spec) :actual (:name field)})))
           (require-capacity! sym (:field spec) (:value field)
                              ((:size-fn spec) descriptor-arguments)))
         logical-arguments
         (mapv (fn [{:keys [kind sym type value-fn]}]
                 (if (= :scalar kind)
                   {:type type :value (value-fn descriptor-arguments)}
                   (resident-of sym)))
               (:argument-specs step))
         arguments (physical-step-arguments step logical-arguments)
         _abi (kabi/validate-arguments! (:abi (kdispatch/default-alternative dispatch)) arguments)
         resident-bindings
         (into {}
               (keep (fn [[{:keys [kind sym]} value]]
                       (when-not (= :scalar kind) [sym value])))
               (map vector (:argument-specs step) logical-arguments))
         direct-scalars (select-keys argmap (:scalar-params descriptor))
         derived-scalars
         (reduce (fn [values [{:keys [kind expression slot]} value]]
                   (if (= :scalar kind)
                     (let [raw (:value value)
                           slot-name (:name slot)]
                       (cond-> values
                         slot-name (assoc slot-name raw)
                         (symbol? expression) (assoc expression raw)))
                     values))
                 {}
                 (map vector (:argument-specs step) logical-arguments))]
     (assoc selected
            :instance-id (:id instance)
            :descriptor descriptor
            :arguments arguments
            :resident-bindings resident-bindings
            :reference-inputs
            {:buffers (select-keys argmap (:array-params descriptor))
             :scalars (merge direct-scalars derived-scalars)}))))

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

(defn output-values
  "Return the plan's public logical values in declared output order without copying to the host."
  [executable]
  (let [executable (ensure-live! executable :output-values)
        value-ids (link-plan/output-value-ids (:plan executable))
        rank (zipmap value-ids (range))]
    (into (sorted-map-by (fn [left right]
                           (let [position-order (compare (get rank left Long/MAX_VALUE)
                                                         (get rank right Long/MAX_VALUE))]
                             (if (zero? position-order)
                               (compare (pr-str left) (pr-str right))
                               position-order))))
          (map (fn [value-id] [value-id (value-view executable value-id)]))
          value-ids)))

(defn run!
  "Replay the linked graph synchronously and return its resident output views. No host copies."
  [executable]
  (let [executable (ensure-live! executable :run!)]
    (when (seq @(:pending-inputs executable))
      (throw (ex-info "linked executable has owned inputs or state that have not been initialized"
                      {:reason :link-pending-inputs :plan (get-in executable [:plan :id])
                       :nodes @(:pending-inputs executable)})))
    (if-let [prepared (:prepared-program executable)]
      (parallel-program/run-prepared! prepared)
      (gpu/replay! (:session executable) (:graph-key executable)))
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

(defn- same-buffer-range?
  [left-buffer left-view right-buffer right-view]
  (and (identical? left-buffer right-buffer)
       (= (:byte-offset left-view) (:byte-offset right-view))
       (= (:byte-length left-view) (:byte-length right-view))))

(defn- overlapping-buffer-range?
  [left-buffer left-view right-buffer right-view]
  (and (identical? left-buffer right-buffer)
       (< (:byte-offset left-view) (bview/byte-end right-view))
       (< (:byte-offset right-view) (bview/byte-end left-view))))

(defn write!
  "Initialize or replace one complete contiguous LinkNode from a host value or DeviceArray.

   Host values use upload!. A compatible DeviceArray already naming the exact destination range
   is accepted without a copy; every other compatible resident value uses a backend device copy,
   never device→host→device. Partially overlapping ranges fail explicitly because backend
   overlap semantics are not a portable memory contract. Returns the executable."
  [executable node-id source]
  (if-not (value/device-array? source)
    (upload! executable node-id source)
    (let [executable (ensure-live! executable :write!)
          node (get-in executable [:plan :nodes node-id])
          _ (when-not node (node-view executable node-id))
          _ (when-not (value/live? source)
              (throw (ex-info "cannot write from a consumed or freed DeviceArray"
                              {:reason :link-device-input-lifetime :node node-id})))
          _ (when-not (= (:device source) (get-in executable [:plan :target]))
              (throw (ex-info "DeviceArray target differs from linked executable target"
                              {:reason :link-device-input-target :node node-id
                               :source (:device source)
                               :target (get-in executable [:plan :target])})))
          source-view (bview/validate-view! (:view source))
          destination (node-view executable node-id)
          destination-view (:view destination)
          _ (when-not (= [(:dtype destination-view) (:shape destination-view)]
                         [(:dtype source-view) (:shape source-view)])
              (throw (ex-info "DeviceArray dtype/shape differs from LinkNode"
                              {:reason :link-device-input-contract :node node-id
                               :source {:dtype (:dtype source-view) :shape (:shape source-view)}
                               :destination {:dtype (:dtype destination-view)
                                             :shape (:shape destination-view)}})))
          _ (when-not (and (bview/contiguous? source-view)
                           (bview/contiguous? destination-view))
              (throw (ex-info "linked DeviceArray writes require contiguous views"
                              {:reason :link-device-input-layout :node node-id})))
          session (:session executable)
          destination-buffer (gpu/buffer session (:key destination))
          source-buffer (:buffer source)
          elements (reduce * 1 (:shape destination-view))]
      (cond
        (same-buffer-range? source-buffer source-view destination-buffer destination-view)
        nil

        (overlapping-buffer-range? source-buffer source-view destination-buffer destination-view)
        (throw (ex-info "linked DeviceArray write has partially overlapping source/destination"
                        {:reason :link-device-input-overlap :node node-id}))

        (identical? source-buffer destination-buffer)
        (let [source-resident
              (gpu/buffer-view session (:key destination)
                               {:id (:id source-view)
                                :byte-offset (:byte-offset source-view)
                                :dtype (:dtype source-view)
                                :shape (:shape source-view)
                                :strides (:strides source-view)})]
          (gpu/copy-range! session source-resident destination {:elements elements}))

        :else
        (let [temporary-key [::device-input (random-uuid)]
              allocation (:allocation source-view)]
          (gpu/register-buffer! session temporary-key source-buffer
                                {:ownership :borrowed
                                 :allocation-id [::device-input-allocation (random-uuid)]
                                 :memory-space (:memory-space allocation)
                                 :coherence (:coherence allocation)
                                 :alignment (:alignment allocation)})
          (try
            (let [source-resident
                  (gpu/buffer-view session temporary-key
                                   {:id (:id source-view)
                                    :byte-offset (:byte-offset source-view)
                                    :dtype (:dtype source-view)
                                    :shape (:shape source-view)
                                    :strides (:strides source-view)})]
              (gpu/copy-range! session source-resident destination {:elements elements}))
            (finally
              ;; Borrowed registrations are detached, never freed.
              (gpu/free-buffer! session temporary-key)))))
      (swap! (:pending-inputs executable) disj node-id)
      executable)))

(defn profile!
  "Profile one replay of an executable instantiated with `{:profile? true}`. Inputs must be ready."
  [executable]
  (let [executable (ensure-live! executable :profile!)]
    (when (:prepared-program executable)
      (throw (ex-info "equation-first program profiling requires an aggregate event schedule"
                      {:reason :link-program-profile-unsupported})))
    (when (seq @(:pending-inputs executable))
      (throw (ex-info "linked executable has inputs or state that have not been initialized"
                      {:reason :link-pending-inputs :nodes @(:pending-inputs executable)})))
    (gpu/profile-recorded-graph! (:session executable) (:graph-key executable))))

(defn measure!
  "Measure a profiled LinkedExecutable with device events. Stateful plans require the explicit
   `:before-sample!` restore hook so repeated samples cannot silently measure mutated state."
  [executable & {:keys [before-sample!] :as opts}]
  (let [executable (ensure-live! executable :measure!)
        state-nodes (into #{} (keep (fn [[node-id node]]
                                      (when (= :state (:role node)) node-id)))
                          (get-in executable [:plan :nodes]))]
    (when (:prepared-program executable)
      (throw (ex-info "equation-first program measurement requires an aggregate event schedule"
                      {:reason :link-program-measure-unsupported})))
    (when (seq @(:pending-inputs executable))
      (throw (ex-info "linked executable has inputs or state that have not been initialized"
                      {:reason :link-pending-inputs :nodes @(:pending-inputs executable)})))
    (when (and (seq state-nodes) (nil? before-sample!))
      (throw (ex-info "stateful linked executables require :before-sample! restoration"
                      {:reason :link-stateful-measurement :state-nodes state-nodes})))
    (apply gpu/measure-recorded-graph! (:session executable) (:graph-key executable)
           (mapcat identity opts))))

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
                         (:phases executable) (:prepared-program executable)
                         (:allocation-keys executable))))
  nil)
