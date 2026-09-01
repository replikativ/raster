(ns raster.gpu.core
  "Unified GPU session layer for Raster.

   Manages kernel compilation, buffer allocation, invocation, and cleanup
   in a single session object. Problem-agnostic: works for PDE, DL, ABM,
   numerical computing — all within the same session with shared buffers.

   Supports multiple GPU backends:
     :ze:N  — Intel Level Zero (low-level, Intel-only)
     :ocl:N — OpenCL ICD (portable: Intel, NVIDIA, AMD)

   Usage:
     (with-gpu-session [sess :ocl:0]
       ;; Compile kernels
       (compile! sess :step #'gray-scott-step!)

       ;; Allocate buffers
       (alloc! sess {:u [:float 1024 my-array]
                     :v [:float 1024 nil]})

       ;; Invoke kernels (buffers looked up from session)
       (invoke! sess :step {\"U\" :u \"V\" :v} [{:type :int :value 32}] 1024)

       ;; Data transfer by buffer key
       (upload! sess :u new-data)
       (def result (download sess :u)))"
  (:refer-clojure :exclude [])
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph-call :as kgcall]
            [raster.compiler.pipeline :as pl]
            [raster.core :as rcore]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.resident-value :as resident-value])
  (:import [java.lang AutoCloseable]))

;; ================================================================
;; Backend dispatch
;; ================================================================

(defn backend-type
  "Determine GPU backend from device-id keyword.
   :ze:0 → :ze, :ocl:0 → :ocl"
  [device-id]
  (let [s (name device-id)]
    (cond
      (str/starts-with? s "ze")  :ze
      (str/starts-with? s "ocl") :ocl
      :else (throw (ex-info (str "Unknown GPU backend: " device-id
                                 ". Use :ze:N or :ocl:N")
                            {:device-id device-id})))))

(defn- rt-resolve-soft
  "Like rt-resolve but returns nil instead of throwing when the backend lacks the fn.
  Used for the bound-dispatch destroyers, which are ze-only."
  [device-id fn-name]
  (let [ns-sym (case (backend-type device-id)
                 :ze  'raster.gpu.ze-runtime
                 :ocl 'raster.gpu.ocl-runtime)]
    (requiring-resolve (symbol (str ns-sym) fn-name))))

(defn- rt-resolve
  "Resolve a function from the appropriate runtime namespace."
  [device-id fn-name]
  (let [ns-sym (case (backend-type device-id)
                 :ze  'raster.gpu.ze-runtime
                 :ocl 'raster.gpu.ocl-runtime)]
    (or (requiring-resolve (symbol (str ns-sym) fn-name))
        (throw (ex-info (str "Cannot resolve " fn-name " in " ns-sym)
                        {:device-id device-id :fn fn-name})))))

(defn- rt-arena-var
  "Get the *current-arena* var for the backend."
  [device-id]
  (let [ns-sym (case (backend-type device-id)
                 :ze  'raster.gpu.ze-runtime
                 :ocl 'raster.gpu.ocl-runtime)]
    (requiring-resolve (symbol (str ns-sym) "*current-arena*"))))

;; ================================================================
;; deftm var resolution
;; ================================================================

(defn resolve-deftm-var
  "Resolve a deftm var through dispatch table to the mangled backing var. Thin
   wrapper over the canonical raster.core/resolve-deftm-var with the GPU policy:
   any concrete overload's par-forms suffice, so pick the first (:ambiguity
   :first); returns nil when v is not a deftm/dispatch var."
  [v]
  (rcore/resolve-deftm-var v {:ambiguity :first}))

(defn get-walked-body
  "Get the walker-processed body from a deftm var, walked FOR the kernel's
   dtype. Delegates to the pipeline's dtype-directed walk (the single
   monomorphized-walk seam): the definition-time walked body is a JVM walk —
   dtype-blind, so a concrete-float kernel's `0.0` accumulators carry double
   stamps that disagree with the float code the GPU backend emits. Falls back
   to the stored definition-time body when the source body is unavailable.
   Throws if the var has no walked body."
  [v dtype]
  (let [resolved (or (resolve-deftm-var v) v)]
    (or (try (pl/get-walked-body v dtype)
             (catch Exception _ nil))
        (:raster.core/deftm-walked-body (meta resolved))
        (rcore/ensure-walked-body! resolved)
        (throw (ex-info "Var has no deftm walked body" {:var v})))))

;; ================================================================
;; Internal: kernel compilation
;; ================================================================

(defn- compile-deftm-internal!
  "Compile a deftm var's par forms to GPU kernels and register them.
   Returns the complete backend result so a session retains first-class dispatches and graphs
   instead of reconstructing them later from the flat kernel list."
  [v device-id {:keys [dtype min-elements] :or {dtype :float min-elements 0}}]
  (let [walked-body (get-walked-body v dtype)
        resolved (or (resolve-deftm-var v) v)
        params (:raster.core/deftm-params (meta resolved))
        tags   (:raster.core/deftm-tags (meta resolved))
        ;; Declared scalar/array element types — the SINGLE shared derivation, used by the
        ;; pipeline's pass-backend too (opencl-pass/derive-param-types). One source of truth.
        {:keys [scalar-types array-types]} (opencl-pass/derive-param-types params tags dtype)
        form (let [f (if (= 1 (count walked-body))
                       (first walked-body)
                       (cons 'do walked-body))
                   f (if (seq scalar-types) (vary-meta f assoc :scalar-types scalar-types) f)
                   f (if (seq array-types)  (vary-meta f assoc :array-types array-types) f)]
               f)
        scheduled (:form (pl/schedule-parallel-form
                          form {:target-device device-id
                                :dtype dtype
                                :array-types array-types
                                :scalar-types scalar-types}))
        par-opencl opencl-pass/opencl-pass
        register!  (rt-resolve device-id "register-kernel!")
        result (par-opencl scheduled
                           :device-id device-id
                           :dtype dtype
                           :min-elements min-elements)]
    (doseq [k (:kernels result)]
      (register! (:kernel-name k) k))
    result))

;; ================================================================
;; Internal: buffer allocation
;; ================================================================

(defn- alloc-buffers-internal
  "Allocate DeviceBuffers from a spec map. Topology-aware."
  [buffer-specs device-id]
  (let [hw-topo  (try ((requiring-resolve 'raster.runtime.hardware/memory-topology) device-id)
                      (catch Exception _ {:model :discrete :integrated? false}))
        unified? (= :unified (:model hw-topo))
        ;; Level Zero's integrated allocation is genuinely host-coherent. OpenCL's OclBuffer,
        ;; however, always owns a cl_mem plus a separate host staging segment: writing that segment
        ;; is not an upload even when the physical GPU shares system memory. Treating topology
        ;; `:unified` as API-level coherence left newly allocated OpenCL inputs full of zeroes.
        coherent-host-view? (and unified? (= :ze (backend-type device-id)))
        mk       (rt-resolve device-id "make-buffer")
        upload   (rt-resolve device-id "array->buffer!")
        as-fbuf  (rt-resolve device-id "buffer-as-float-buffer")
        as-ibuf  (rt-resolve device-id "buffer-as-int-buffer")

        buf-of (fn [arr dtype n]
                 (let [buf (mk n dtype)]
                   (if arr
                     (if coherent-host-view?
                       (case dtype
                         :float (let [fb (as-fbuf buf)]
                                  (.put fb ^floats arr 0 (int n))
                                  buf)
                         :int   (let [ib (as-ibuf buf)]
                                  (.put ib ^ints arr 0 (int n))
                                  buf)
                         ;; The fast coherent host views are currently specialized for the two
                         ;; common storage types. All other typed buffers still use the ordinary
                         ;; backend upload contract; allocation must never guess from dtype.
                         (upload buf arr))
                         (upload buf arr))
                     buf)))]
    (into {}
          (map (fn [[k [dtype n source-arr]]]
                 [k (buf-of source-arr dtype n)]))
          buffer-specs)))

(defn- free-buffers-internal!
  "Free all DeviceBuffers in a buffer map."
  [bufs device-id]
  (let [free! (rt-resolve device-id "free-buffer!")]
    (doseq [[_ buf] bufs]
      (free! buf))))

(defn- allocation-contract
  [device-id session-id key buffer ownership opts]
  (let [backend (backend-type device-id)]
    (bview/allocation
     {:id (or (:allocation-id opts) [session-id key (random-uuid)])
      :byte-size (:byte-size buffer)
      :memory-space (or (:memory-space opts) (case backend :ze :shared :ocl :device))
      :device device-id
      :alignment (or (:alignment opts) (:alignment buffer) 1)
      :coherence (or (:coherence opts)
                     (case backend :ze :host-coherent :ocl :explicit-transfer))
      :ownership ownership})))

(defn- free-session-buffers!
  "Free only session-owned allocations. Borrowed/external registrations are detached, never
   destroyed by Raster."
  [buffers allocations device-id]
  (let [free! (rt-resolve device-id "free-buffer!")]
    (doseq [[key buffer] buffers
            :when (= :owned (:ownership (get allocations key)))]
      (free! buffer))))

(defn- alloc-buffers-transactional
  "Allocate buffer specs one at a time and free the successful prefix on failure."
  [buffer-specs device-id]
  (let [allocated (volatile! {})]
    (try
      (doseq [[key spec] buffer-specs]
        (vswap! allocated merge (alloc-buffers-internal {key spec} device-id)))
      @allocated
      (catch Exception e
        (when (seq @allocated)
          (free-buffers-internal! @allocated device-id))
        (throw e)))))

(defrecord BoundExecutableStep [prepareds temporary-buffers owned-view-buffers])

(defn- bound-executable-step?
  [value]
  (instance? BoundExecutableStep value))

(defn- prepared-bindings
  "Return the ordered backend bindings represented by one prepared session entry. Plain bindings
   from the low-level prepare! API remain valid; compiler steps use BoundExecutableStep because one
   semantic step may select a multi-kernel schedule."
  [entry]
  (if (bound-executable-step? entry) (:prepareds entry) [entry]))

(defn- destroy-prepared-entry!
  "Destroy one plain prepared binding or one compiler-owned executable step. Kernel handles die
   before their private storage because their argument state still refers to those buffers."
  [device-id entry]
  (when entry
    (when-let [destroy-prepared! (rt-resolve-soft device-id "destroy-prepared!")]
      (doseq [prepared (prepared-bindings entry)]
        (try (destroy-prepared! prepared) (catch Exception _))))
    ;; Level Zero slices are non-owning pointers. OpenCL sub-buffers are independently
    ;; reference-counted cl_mem values and follow the executable step that materialized them.
    (when (and (bound-executable-step? entry) (seq (:owned-view-buffers entry)))
      (when-let [free! (rt-resolve-soft device-id "free-buffer!")]
        (doseq [buffer (:owned-view-buffers entry)]
          (try (free! buffer) (catch Exception _)))))
    (when (and (bound-executable-step? entry) (seq (:temporary-buffers entry)))
      (try (free-buffers-internal! (:temporary-buffers entry) device-id)
           (catch Exception _)))))

(defn- recorded-graph-entry?
  [value]
  (true? (::recorded-graph value)))

(defn- destroy-recorded-graph-entry!
  [device-id entry]
  (when entry
    (when-let [destroy-graph! (rt-resolve-soft device-id "destroy-graph!")]
      (doseq [graph (if (recorded-graph-entry? entry)
                      [(:replay-graph entry) (:prologue-graph entry)]
                      [entry])
              :when graph]
        (try (destroy-graph! graph) (catch Exception _))))))

(defn- destroy-kernel-graph-entry!
  "Destroy one bound graph's backend recording, dedicated kernel handles and graph-owned
   temporaries/view handles. External root buffers remain session-owned and are never freed here."
  [device-id {:keys [runtime-graph prepareds owned-view-buffers temporary-buffers]}]
  (let [destroy-graph! (rt-resolve-soft device-id "destroy-graph!")
        destroy-prepared! (rt-resolve-soft device-id "destroy-prepared!")]
    (when (and destroy-graph! runtime-graph)
      (try (destroy-graph! runtime-graph) (catch Exception _)))
    (when destroy-prepared!
      (doseq [prepared prepareds]
        (try (destroy-prepared! prepared) (catch Exception _))))
    ;; OpenCL cl_mem sub-buffers are independently reference-counted native objects. Kernel
    ;; bindings must die first because their argument state still refers to these handles.
    (when (seq owned-view-buffers)
      (let [free! (rt-resolve device-id "free-buffer!")]
        (doseq [buffer owned-view-buffers]
          (try (free! buffer) (catch Exception _)))))
    (when (seq temporary-buffers)
      (free-buffers-internal! temporary-buffers device-id))))

(def ^:private array-tag->dtype
  {'doubles :double
   'floats  :float
   'longs   :long
   'ints    :int
   'bytes   :byte})

(defn- object-field
  [obj field-name]
  (let [cls (class obj)
        candidates [field-name (str/replace field-name "-" "_")]
        field (some (fn [candidate]
                      (try
                        (doto (.getDeclaredField cls candidate)
                          (.setAccessible true))
                        (catch NoSuchFieldException _ nil)))
                    candidates)]
    (when-not field
      (throw (NoSuchFieldException.
              (str field-name " (tried " (str/join ", " candidates) ")"))))
    (.get ^java.lang.reflect.Field field obj)))

(defn- bundle-type-tag
  [bundle]
  (symbol (.getSimpleName (class bundle))))

(defn array-bundle-buffer-specs
  "Derive GPU buffer specs from a defvalue that bundles primitive arrays.

   Returns {buffer-key [dtype n source-array]} using the field-type registry
   populated by defvalue. Non-array fields are ignored.

   opts:
   - :aliases {field-key -> buffer-key} to rename derived buffer keys"
  ([bundle]
   (array-bundle-buffer-specs bundle {}))
  ([bundle {:keys [aliases] :or {aliases {}}}]
   (let [registry @inf/field-type-registry
         type-tag (bundle-type-tag bundle)
         field-types (or (get registry type-tag)
                         (throw (ex-info "No field metadata registered for array bundle"
                                         {:type type-tag :registered (keys registry)})))]
     (into {}
           (keep (fn [[field-name array-tag]]
                   (when-let [dtype (get array-tag->dtype array-tag)]
                     (let [field-key (-> field-name
                                         (str/replace "_" "-")
                                         keyword)
                           buffer-key (get aliases field-key field-key)
                           source-arr (object-field bundle field-name)]
                       [buffer-key [dtype (java.lang.reflect.Array/getLength source-arr) source-arr]]))))
           field-types))))

;; ================================================================
;; GPU Session
;; ================================================================

(defn make-session
  "Create a new GPU session. Manages kernels, buffers, and cleanup.
   Returns an atom holding session state.

   The session owns all resources allocated through it and frees them
   when closed via close-session! or with-gpu-session."
  [device-id]
  (let [make-arena! (rt-resolve device-id "make-kernel-arena!")
        arena-id (make-arena!)]
    (atom {:device-id device-id
           :session-id (random-uuid)
           :arena-id  arena-id
           :kernels   {}       ;; {phase-key → [kernel-info ...]}
           :dispatches {}      ;; {phase-key → [KernelDispatch ...]}
           :buffers   {}       ;; {buf-key → DeviceBuffer}
           :allocations {}     ;; {buf-key → backend-neutral BufferAllocation}
           :kernel-graphs {}    ;; {graph-key → bound emitted KernelGraph}
           :events {}           ;; {event-id → session-owned asynchronous completion}
           :closed?   false})))

(defn transfer-capabilities
  "Return backend transfer execution capabilities for `sess`.

  `:independent-physical-queue?` reports whether transfer submission uses a
  physical queue distinct from compute. It is a scheduling capability, not a
  promise that a particular device overlaps copy and kernels under load."
  [sess]
  (let [device-id (:device-id @sess)]
    (assoc ((rt-resolve device-id "transfer-capabilities"))
           :backend (backend-type device-id)
           :device-id device-id)))

(declare release-event!)

(defn close-session!
  "Free all buffers and kernels in a session. Idempotent and thread-safe.

  Also destroys the per-binding fresh kernel handles (:prepared) and recorded command graphs
  (:graphs). These hold dedicated driver objects (zeKernel per bind, zeCommandQueue+List per
  graph) that are NOT in the kernel registry, so close-kernel-arena! never reaches them — without
  this every session leaks them and the driver eventually aborts (the source of the SIGABRTs)."
  [sess]
  (locking sess
    (when-not (:closed? @sess)
      ;; Completion owns the right to keep graph recordings, bound kernels, and buffers alive.
      ;; Drain and release every event before tearing any of those resources down.
      (doseq [[_ {:keys [event]}] (:events @sess)]
        (release-event! sess event))
      (let [{:keys [device-id arena-id buffers allocations prepared graphs kernel-graphs]} @sess]
        (doseq [[_ graph-entry] graphs]
          (destroy-recorded-graph-entry! device-id graph-entry))
        (doseq [[_ prepared-entry] prepared]
          (destroy-prepared-entry! device-id prepared-entry))
        (doseq [[_ graph-entry] kernel-graphs]
          (destroy-kernel-graph-entry! device-id graph-entry))
        (free-session-buffers! buffers allocations device-id)
        (let [close-arena! (rt-resolve device-id "close-kernel-arena!")]
          (close-arena! arena-id))
        (swap! sess assoc :closed? true :buffers {} :allocations {} :kernels {} :dispatches {}
               :prepared {} :graphs {} :kernel-graphs {} :events {})))))

(defn with-gpu-session*
  "Functional implementation for with-gpu-session macro."
  [device-id body-fn]
  (let [sess (make-session device-id)
        arena-var (rt-arena-var device-id)]
    (push-thread-bindings {arena-var (:arena-id @sess)})
    (try
      (body-fn sess)
      (finally
        (pop-thread-bindings)
        (close-session! sess)))))

(defmacro with-gpu-session
  "Execute body with a GPU session. All compiled kernels and allocated
   buffers are automatically freed on exit (normal or exceptional).

   Usage:
     (with-gpu-session [sess :ocl:0]
       (compile! sess :step #'gray-scott-step!)
       (alloc! sess {:u [:float 1024 nil]})
       (invoke! sess :step {\"U\" :u} [] 1024))"
  [[sess-sym device-id] & body]
  `(with-gpu-session* ~device-id (fn [~sess-sym] ~@body)))

;; ================================================================
;; Kernel compilation
;; ================================================================

(defn compile!
  "Compile a deftm var and register its kernels in the session.

   sess: session atom
   phase-key: keyword to identify this kernel group (e.g. :step, :colorize)
   v: var pointing to a deftm function
   opts: {:dtype :float, :min-elements 0}"
  ([sess phase-key v] (compile! sess phase-key v {}))
  ([sess phase-key v opts]
   (let [device-id (:device-id @sess)
         ;; Dedup generation by (op, dtype): compile-deftm-internal! emits a gensym-named kernel
         ;; each call, so without this N phases of the SAME deftm produce N distinct kernel SOURCES
         ;; → N SPIR-V compiles (ocloc) at first prepare!, even though the bodies are identical.
         ;; Caching the generated kernel-vec per (op, dtype) makes those phases SHARE one kernel
         ;; name → one SPIR-V compile. The bound path already mints a fresh handle per binding from
         ;; the shared module, so distinct phases keep independent arg sets. (e.g. the 18-layer
         ;; gemma forward: 453 steps / ~8 distinct kernels → first token 171s → ~3s.)
         cache-key [v (get opts :dtype :float)]
         compiled (or (get-in @sess [:kernel-cache cache-key])
                      (let [result (compile-deftm-internal! v device-id opts)]
                        (swap! sess assoc-in [:kernel-cache cache-key] result)
                        result))
         kernels (:kernels compiled)]
     (swap! sess (fn [state]
                   (-> state
                       (assoc-in [:kernels phase-key] kernels)
                       (assoc-in [:dispatches phase-key] (:dispatches compiled)))))
     kernels)))

(defn compile-phases!
  "Compile multiple deftm vars into the session. Returns {key → kernel-info-vec}.

   sess: session atom
   phase-map: {keyword → var}, e.g. {:produce #'produce! :distribute #'distribute!}
   opts: compilation options passed to each compile!"
  ([sess phase-map] (compile-phases! sess phase-map {}))
  ([sess phase-map opts]
   (into {} (map (fn [[k v]] [k (compile! sess k v opts)])) phase-map)))

;; ================================================================
;; Buffer allocation
;; ================================================================

(defn alloc!
  "Allocate DeviceBuffers and register them in the session.
   Topology-aware: uses zero-copy on integrated GPUs, memcpy on discrete.

   sess: session atom
   buffer-specs: {key → [dtype n source-array-or-nil allocation-opts?]}

   `allocation-opts`, when present, carries the same stable allocation metadata accepted by
   register-buffer!, including :allocation-id. The fourth element affects the public allocation
   contract only; allocation and initial upload still use the first three elements.

   Buffers are merged into the session — call multiple times to add more.
   If allocation fails partway through, already-allocated buffers are freed."
  [sess buffer-specs]
  (locking sess
    (let [{:keys [device-id session-id buffers closed?]} @sess
          duplicate-keys (set (filter #(contains? buffers %) (keys buffer-specs)))]
      (when closed?
        (throw (ex-info "cannot allocate in a closed GPU session" {})))
      (when (seq duplicate-keys)
        (throw (ex-info "session buffer keys must identify one stable allocation lifetime"
                        {:duplicate-keys duplicate-keys})))
      (let [new-bufs (alloc-buffers-transactional buffer-specs device-id)]
        (let [new-allocations
              (try
                (into {}
                      (map (fn [[key buffer]]
                             [key (allocation-contract device-id session-id key buffer :owned
                                                       (or (nth (get buffer-specs key) 3 nil) {}))]))
                      new-bufs)
                (catch Exception e
                  (free-buffers-internal! new-bufs device-id)
                  (throw e)))]
          (swap! sess (fn [state]
                        (-> state
                            (update :buffers merge new-bufs)
                            (update :allocations merge new-allocations))))
          new-bufs)))))

(defn register-buffer!
  "Register an existing backend buffer without taking ownership by default.

   This is the safe zero-copy/import seam: callers must first obtain a buffer that the selected
   backend can legally address (an arbitrary Panama MemorySegment is not necessarily GPU-visible).
   `opts` may specify :ownership (:external or :borrowed), :memory-space, :coherence, :alignment,
   and a stable :allocation-id. Raster never frees external/borrowed registrations."
  ([sess key buffer] (register-buffer! sess key buffer {}))
  ([sess key buffer opts]
   (locking sess
     (let [{:keys [device-id session-id buffers closed?]} @sess
           ownership (or (:ownership opts) :external)
           device-buffer? (rt-resolve device-id "device-buffer?")]
       (when closed?
         (throw (ex-info "cannot register a buffer in a closed GPU session" {:key key})))
       (when (contains? buffers key)
         (throw (ex-info "session buffer key already names a live allocation" {:key key})))
       (when-not (contains? #{:external :borrowed} ownership)
         (throw (ex-info "registered buffers must remain caller-owned"
                         {:key key :ownership ownership})))
       (when-not (device-buffer? buffer)
         (throw (ex-info "registered value is not a buffer for this session backend"
                         {:key key :device-id device-id :actual (type buffer)})))
       (let [allocation (allocation-contract device-id session-id key buffer ownership opts)]
         (swap! sess (fn [state]
                       (-> state
                           (assoc-in [:buffers key] buffer)
                           (assoc-in [:allocations key] allocation))))
         buffer)))))

(defn free-buffer!
  "Release a specific buffer registration. Raster frees it only when the allocation is owned."
  [sess key]
  (locking sess
    (let [{:keys [device-id buffers allocations kernel-graphs]} @sess
          bound-graphs (->> kernel-graphs
                            (keep (fn [[graph-key entry]]
                                    (when (some #(= key (:key %))
                                                (vals (:resident-views entry)))
                                      graph-key)))
                            vec)]
      (when-let [buf (get buffers key)]
        (when (seq bound-graphs)
          (throw (ex-info "cannot release a buffer while a kernel graph holds one of its views"
                          {:key key :kernel-graphs bound-graphs})))
        (when (= :owned (:ownership (get allocations key)))
          ((rt-resolve device-id "free-buffer!") buf))
        (swap! sess (fn [state]
                      (-> state
                          (update :buffers dissoc key)
                          (update :allocations dissoc key))))))))

;; ================================================================
;; Buffer argument resolution
;; ================================================================

(defn- resolve-kernel-bufs
  "Resolve buffer arguments for a kernel from session buffers.
   sym->buf-key maps kernel param names to session buffer keys.  When present,
   the ordered ABI is authoritative and includes a functional map's output."
  [kernel-info bufs sym->buf-key]
  (mapv (fn [sym]
          (let [sym-name (name sym)
                normalized-key (-> sym-name
                                   (str/replace "_" "-")
                                   keyword)
                k (or (get sym->buf-key sym-name)
                      (get sym->buf-key (symbol sym-name))
                      (get sym->buf-key normalized-key)
                      normalized-key)]
            (or (get bufs k)
                (throw (ex-info (str "No buffer for kernel param: " sym-name)
                                {:sym sym :available (keys bufs)})))))
        (if-let [abi (:abi kernel-info)]
          (kabi/pointer-binding-names abi)
          (:array-params kernel-info))))

;; ================================================================
;; Kernel invocation
;; ================================================================

(defn invoke!
  "Invoke a compiled kernel from the session.

   sess: session atom
   phase-key: keyword identifying the kernel group
   sym->buf-key: {\"param_name\" → :buffer-key} mapping
   scalars: vector of {:type :int/:float/:long :value v}
   n: number of work items
   opts: {:index 0} — which kernel in multi-kernel phases"
  ([sess phase-key sym->buf-key scalars n]
   (invoke! sess phase-key sym->buf-key scalars n {}))
  ([sess phase-key sym->buf-key scalars n {:keys [index] :or {index 0}}]
   (let [{:keys [kernels buffers]} @sess
         kernel-vec (or (get kernels phase-key)
                        (throw (ex-info (str "No kernel for phase: " phase-key)
                                        {:available (keys kernels)})))
         kernel-info (nth kernel-vec index)
         buf-vec (resolve-kernel-bufs kernel-info buffers sym->buf-key)
         device-id (:device-id @sess)
         invoke-fn! (rt-resolve device-id "invoke-registered-map-void-kernel")]
     (invoke-fn! (:kernel-name kernel-info) buf-vec scalars n))))

(defn prepare!
  "Pre-bind a kernel's arguments ONCE for fast repeated dispatch (the launch-overhead fix).
  Resolves the session buffers for the kernel's params, binds them + scalars + n, and caches
  the bound handle in the session under [:prepared phase-key]. Subsequent invoke-bound! calls
  skip per-launch arg setup and the barrier (measured 2.6-5× faster than invoke!).

  Requires all kernel array params to map to session buffers (DeviceBuffers) — the residency-
  friendly path. Re-call prepare! only if a buffer is reallocated or n changes; buffer CONTENTS
  may change freely between invoke-bound! calls (the bound pointers are stable)."
  ([sess phase-key sym->buf-key scalars n]
   (prepare! sess phase-key sym->buf-key scalars n {}))
  ([sess phase-key sym->buf-key scalars n {:keys [index async? kernel-phase] :or {index 0}}]
   (let [{:keys [kernels buffers]} @sess
         ;; The COMPILED kernel comes from kernel-phase (defaults to phase-key); the bound
         ;; argument-set is stored under phase-key. This lets one compiled kernel back many
         ;; distinct bindings (e.g. every matmul in a decode token shares one dp4a kernel).
         klookup (or kernel-phase phase-key)
         kernel-vec (or (get kernels klookup)
                        (throw (ex-info (str "No kernel for phase: " klookup)
                                        {:available (keys kernels)})))
         kernel-info (nth kernel-vec index)
         buf-vec (resolve-kernel-bufs kernel-info buffers sym->buf-key)
         device-id (:device-id @sess)
         bind-fn (rt-resolve device-id "bind-registered-map-void-kernel")
         prepared (bind-fn (:kernel-name kernel-info) buf-vec scalars n {:async? (boolean async?)})]
     (destroy-prepared-entry! device-id (get-in @sess [:prepared phase-key]))
     (swap! sess assoc-in [:prepared phase-key] prepared)
     prepared)))

(defn invoke-bound!
  "Dispatch a kernel previously bound with prepare!. No arg setup, no barrier — the
  low-overhead dispatch path. With an async-prepared kernel the dispatch returns immediately
  (call sync! before reading results); otherwise it completes synchronously.
  Throws if the phase was not prepared."
  [sess phase-key]
  (let [prepared (or (get-in @sess [:prepared phase-key])
                     (throw (ex-info (str "Phase not prepared: " phase-key " — call prepare! first")
                                     {:prepared (keys (:prepared @sess))})))
        bindings (prepared-bindings prepared)
        _ (when-not (= 1 (count bindings))
            (throw (ex-info (str "Phase " phase-key " is a " (count bindings)
                                 "-kernel executable step — record it with record-graph! instead of "
                                 "invoking one backend binding")
                            {:phase phase-key :kernel-count (count bindings)})))
        device-id (:device-id @sess)
        launch-fn (rt-resolve device-id "launch-registered-bound!")]
    (launch-fn (first bindings))))

(defn sync!
  "Block until all async-dispatched kernels on this device have completed. Call once after a
  batch of async invoke-bound! calls, before downloading results."
  [sess]
  (let [device-id (:device-id @sess)]
    ((rt-resolve device-id "synchronize-async!"))))

(defn record-graph!
  "Record an ordered sequence of prepared kernels into a replayable command graph (the AOT
  decode-graph). Pays the per-launch host-append cost ONCE; replay! then runs the whole
  sequence with a single queue execute — eliminating the per-token dispatch floor.

  phase-keys: ordered vector of phase-keys previously bound via prepare!. The kernel sequence
  and buffer pointers are fixed; buffer CONTENTS may change between replays. Stored under :graph
  (or graph-key). Re-record only if the sequence or a buffer is reallocated."
  ([sess phase-keys] (record-graph! sess phase-keys :graph {}))
  ([sess phase-keys graph-key] (record-graph! sess phase-keys graph-key {}))
  ([sess phase-keys graph-key {:keys [profile?] :or {profile? false}}]
   (let [device-id (:device-id @sess)
         record-fn (rt-resolve device-id "record-graph!")
         entries (mapv (fn [pk]
                         (or (get-in @sess [:prepared pk])
                             (throw (ex-info (str "Phase not prepared: " pk " — call prepare! first")
                                             {:prepared (keys (:prepared @sess))}))))
                       phase-keys)
         prepareds (vec (mapcat prepared-bindings entries))
         prologue-prepareds (filterv :const-prologue? prepareds)
         replay-prepareds (filterv (complement :const-prologue?) prepareds)
         prologue-graph (when (seq prologue-prepareds) (record-fn prologue-prepareds))
         graph (try
                 (when prologue-graph
                   ((rt-resolve device-id "replay-graph!") prologue-graph))
                 ;; Keep the non-profiling recording on the backend's exact fast path. Profiling
                 ;; events are a graph-construction property, not a replay flag.
                 (if profile?
                   (record-fn replay-prepareds {:barriers? true :profile? true})
                   (record-fn replay-prepareds))
                 (catch Exception e
                   (destroy-recorded-graph-entry! device-id prologue-graph)
                   (throw e)))
         entry (if (or prologue-graph profile?)
                 {::recorded-graph true
                  :replay-graph graph
                  :prologue-graph prologue-graph
                  :profile? (boolean profile?)}
                 graph)]
     (destroy-recorded-graph-entry! device-id (get-in @sess [:graphs graph-key]))
     (swap! sess assoc-in [:graphs graph-key] entry)
     graph)))

(defn replay!
  "Execute a recorded command graph once (synchronous). Reads current buffer contents."
  ([sess] (replay! sess :graph))
  ([sess graph-key]
   (let [device-id (:device-id @sess)
         entry (or (get-in @sess [:graphs graph-key])
                   (throw (ex-info (str "No graph: " graph-key " — call record-graph! first") {})))
         graph (if (recorded-graph-entry? entry) (:replay-graph entry) entry)]
     ((rt-resolve device-id "replay-graph!") graph)
     ;; A normal replay intentionally discards profiling data, just like LinkedExecutable/run!.
     ;; Profiling events must be reset before the next replay; profile-recorded-graph! reads and
     ;; resets them instead.
     (when (and (recorded-graph-entry? entry) (:profile? entry))
       (when-let [reset-fn (rt-resolve-soft device-id "reset-graph-events!")]
         (reset-fn graph))))))

(defn release-recorded-graph!
  "Release one graph recorded by record-graph!. Idempotent. Prepared executable steps and their
   resident buffers remain live until released separately."
  [sess graph-key]
  (locking sess
    (when-let [entry (get-in @sess [:graphs graph-key])]
      (swap! sess update :graphs dissoc graph-key)
      (destroy-recorded-graph-entry! (:device-id @sess) entry)))
  nil)

(defn release-prepared!
  "Release one phase prepared by prepare! or bind-step!. Idempotent. A recorded graph referring to
   the phase must be released first; callers that own both should use that order."
  [sess phase-key]
  (locking sess
    (when-let [entry (get-in @sess [:prepared phase-key])]
      (swap! sess update :prepared dissoc phase-key)
      (destroy-prepared-entry! (:device-id @sess) entry)))
  nil)

(declare bind-kernel-executable! run-kernel-graph! release-kernel-graph!)

(defn invoke-scan!
  "Invoke a compiled exclusive scan through its scheduled KernelExecutable.

   sess: session atom
   phase-key: keyword identifying the compiled scan dispatch
   input-keys: vector of buffer keys for inputs
   output-key: buffer key for output
   n: number of elements"
  [sess phase-key input-keys output-key n]
  (let [dispatches (get-in @sess [:dispatches phase-key])
        _ (when-not (= 1 (count dispatches))
            (throw (ex-info "compiled scan phase must retain exactly one KernelDispatch"
                            {:phase phase-key :dispatch-count (count dispatches)
                             :reason :compiled-scan-dispatch-missing})))
        executable (kdispatch/default-alternative (first dispatches))
        remaining-inputs (volatile! (seq input-keys))
        arguments
        (mapv (fn [slot]
                (cond
                  (and (= :scalar (:kind slot)) (= :bound (:role slot)))
                  {:type (:kernel-dtype slot) :value n}

                  (= :scalar (:kind slot))
                  (throw (ex-info "invoke-scan! cannot guess a captured scalar binding"
                                  {:phase phase-key :slot slot
                                   :reason :compiled-scan-captured-scalar-unbound}))

                  (kabi/writable? slot)
                  output-key

                  (kabi/readable? slot)
                  (let [key (first @remaining-inputs)]
                    (when-not key
                      (throw (ex-info "compiled scan has more input slots than caller bindings"
                                      {:phase phase-key :slot slot :input-keys input-keys})))
                    (vswap! remaining-inputs next)
                    key)

                  :else
                  (throw (ex-info "compiled scan ABI slot has no runtime binding policy"
                                  {:phase phase-key :slot slot}))))
              (kexec/abi executable))
        _ (when (seq @remaining-inputs)
            (throw (ex-info "compiled scan caller supplied unused input bindings"
                            {:phase phase-key :unused (vec @remaining-inputs)})))
        handle (bind-kernel-executable! sess [:compiled-scan phase-key]
                                        executable arguments)]
    (try
      (run-kernel-graph! sess handle)
      (get-in @sess [:buffers output-key])
      (finally
        (release-kernel-graph! sess handle)))))

(defn invoke-rng-fill!
  "Invoke a compiled parallel RNG fill kernel from the session.

   sess: session atom
   phase-key: keyword identifying the RNG fill kernel
   buf-key: buffer key for output
   n: number of elements
   base-seed: long seed value"
  [sess phase-key buf-key n base-seed]
  (let [{:keys [kernels buffers]} @sess
        kernel-info (first (get kernels phase-key))
        device-id (:device-id @sess)
        invoke! (rt-resolve device-id "invoke-registered-rng-fill-kernel")]
    (invoke! (:kernel-name kernel-info) (get buffers buf-key) n base-seed)))

(defn invoke-active-ids!
  "Invoke a compiled parallel active-id generation kernel from the session.

   sess: session atom
   phase-key: keyword identifying the active-ids kernel
   buf-key: buffer key for output indices
   n-active: number of active elements to generate
   n-total: total population size (modulus)
   base-seed: long seed value"
  [sess phase-key buf-key n-active n-total base-seed]
  (let [{:keys [kernels buffers]} @sess
        kernel-info (first (get kernels phase-key))
        device-id (:device-id @sess)
        invoke! (rt-resolve device-id "invoke-registered-active-ids-kernel")]
    (invoke! (:kernel-name kernel-info) (get buffers buf-key) n-active n-total base-seed)))

;; ================================================================
;; Data transfer
;; ================================================================

(defn upload!
  "Copy JVM array into a session buffer by key.

   sess: session atom
   key: buffer key
   arr: JVM array to upload"
  [sess key arr]
  (let [{:keys [device-id buffers]} @sess
        buf (or (get buffers key)
                (throw (ex-info (str "No buffer for key: " key)
                                {:available (keys buffers)})))]
    ((rt-resolve device-id "array->buffer!") buf arr)))

(defn download
  "Download a session buffer to a new JVM array.

   sess: session atom
   key: buffer key"
  [sess key]
  (let [{:keys [device-id buffers]} @sess
        buf (or (get buffers key)
                (throw (ex-info (str "No buffer for key: " key)
                                {:available (keys buffers)})))]
    ((rt-resolve device-id "buffer->array") buf)))

(defrecord ResidentBufferView [session-id key view])

(defn resident-buffer-view?
  [x]
  (and x (= "raster.gpu.core.ResidentBufferView" (.getName (class x)))))

(defn- checked-resident-view
  [sess resident]
  (when-not (resident-buffer-view? resident)
    (throw (ex-info "expected a ResidentBufferView" {:value resident :actual (type resident)})))
  (let [{current-session :session-id buffers :buffers allocations :allocations closed? :closed?} @sess
        key (:key resident)
        current-allocation (get allocations key)]
    (when closed?
      (throw (ex-info "cannot use a buffer view from a closed GPU session" {:key key})))
    (when-not (= current-session (:session-id resident))
      (throw (ex-info "buffer view belongs to a different GPU session"
                      {:key key :expected current-session :actual (:session-id resident)})))
    (when-not (and (contains? buffers key)
                   current-allocation
                   (= (:id current-allocation) (get-in resident [:view :allocation :id])))
      (throw (ex-info "buffer view no longer names its original live allocation"
                      {:key key :allocation (get-in resident [:view :allocation :id])})))
    (bview/validate-view! (:view resident))
    resident))

(defn buffer-view
  "Return a stable checked view of a session allocation.

   With no opts this is the whole flat buffer. Options describe a typed view and accept
   :byte-offset, :shape, :strides, and :id. Shape defaults to the remaining one-dimensional
   capacity. Byte offsets are relative to the allocation, and all bounds are checked now."
  ([sess key] (buffer-view sess key {}))
  ([sess key opts]
   (let [{:keys [session-id buffers allocations closed?]} @sess
         buffer (get buffers key)
         allocation (get allocations key)]
     (when closed?
       (throw (ex-info "cannot create a buffer view in a closed GPU session" {:key key})))
     (when-not (and buffer allocation)
       (throw (ex-info (str "No buffer for key: " key " (or no live allocation contract)")
                       {:key key :available (keys buffers)})))
     (let [view-dtype (or (:dtype opts) (:dtype buffer))
           byte-offset (long (or (:byte-offset opts) 0))
           element-bytes (long (dtype/bytes-of view-dtype))
           remaining (- (:byte-size allocation) byte-offset)
           _ (when (or (neg? byte-offset) (neg? remaining)
                       (not (zero? (mod remaining element-bytes))))
               (throw (ex-info "buffer view offset leaves no integral typed capacity"
                               {:key key :byte-offset byte-offset :dtype view-dtype
                                :allocation-bytes (:byte-size allocation)})))
           shape (or (:shape opts) [(quot remaining element-bytes)])
           descriptor (bview/view allocation
                                  (assoc opts :byte-offset byte-offset
                                         :dtype view-dtype
                                         :shape shape))]
       (->ResidentBufferView session-id key descriptor)))))

(defn sub-buffer-view
  "Create a checked resident view contained by `base`; :byte-offset is relative to `base`."
  [sess base opts]
  (let [base (checked-resident-view sess base)]
    (->ResidentBufferView (:session-id base) (:key base)
                          (bview/subview (:view base) opts))))

(defn- resolve-resident-binding
  [sess key-or-view]
  (let [resident (if (resident-buffer-view? key-or-view)
                   (checked-resident-view sess key-or-view)
                   (buffer-view sess key-or-view))]
    {:buffer (get-in @sess [:buffers (:key resident)])
     :resident resident
     :view (:view resident)}))

(defn- checked-view-range-spec
  [buffer view spec direction]
  (let [view (bview/validate-view! view)
        raw-dtype (dtype/canon (:dtype buffer))
        view-dtype (dtype/canon (:dtype view))
        element-bytes (long (dtype/bytes-of raw-dtype))
        buffer-field (case direction :upload :dst-element :download :src-element)
        relative (long (get spec buffer-field 0))
        elements (:elements spec)
        capacity (quot (:byte-length view) element-bytes)]
    (when-not (= raw-dtype view-dtype)
      (throw (ex-info "ranged transfers cannot reinterpret a resident view's storage dtype"
                      {:buffer-dtype raw-dtype :view-dtype view-dtype})))
    (when-not (bview/contiguous? view)
      (throw (ex-info "ranged transfers require a contiguous resident view"
                      {:view (:id view) :shape (:shape view) :strides (:strides view)})))
    (when-not (and (integer? elements) (not (neg? elements)))
      (throw (ex-info "ranged transfer requires a non-negative integer element count"
                      {:elements elements})))
    (when (neg? relative)
      (throw (ex-info "ranged transfer has a negative buffer-view offset"
                      {:view (:id view) :offset relative :elements elements})))
    (when (> (+ relative elements) capacity)
      (throw (ex-info "ranged transfer exceeds the buffer view"
                      {:view (:id view) :offset relative :elements elements
                       :view-elements capacity})))
    (assoc spec buffer-field (+ (quot (:byte-offset view) element-bytes) relative))))

(defn upload-range!
  "Copy a SUB-RANGE of a host array or MemorySegment into a session buffer.

     (upload-range! sess :kc0 src {:src-element 0 :dst-element 0 :elements (* tokens kvrow)})

   `src` may be a JVM primitive array or a MemorySegment — an mmap'd file is copied directly, no
   JVM array in between. Offsets and length are in ELEMENTS of the buffer's dtype; byte size is
   never the caller's to get wrong. Out-of-range is an error, not a clamp.

   Why this exists: `upload!`/`download` move the WHOLE buffer. A KV cache is allocated at
   `maxpos` positions and position-major, so a continuation of `t` tokens is one contiguous
   prefix — exporting it should move `t` rows, not `maxpos`."
  [sess key-or-view src spec]
  (let [{:keys [buffer view]} (resolve-resident-binding sess key-or-view)
        spec (checked-view-range-spec buffer view spec :upload)]
    ((rt-resolve (:device-id @sess) "upload-range!") buffer src spec)))

(defn download-range!
  "Copy a SUB-RANGE of a session buffer into a host array or MemorySegment; mirror of
   `upload-range!`. Returns `dst`."
  [sess key-or-view dst spec]
  (let [{:keys [buffer view]} (resolve-resident-binding sess key-or-view)
        spec (checked-view-range-spec buffer view spec :download)]
    ((rt-resolve (:device-id @sess) "download-range!") buffer dst spec)))

(defn copy-range!
  "Copy a contiguous element range between resident buffers or views.

   `spec` requires `:elements`; `:src-element` and `:dst-element` default to
   zero and are relative to their respective views. Source and destination must
   belong to this session and have the same storage dtype. The backend performs
   a direct resident copy without materializing a JVM array. Returns `dst`."
  [sess src dst {:keys [src-element dst-element elements]
                 :or {src-element 0 dst-element 0}
                 :as spec}]
  (let [{src-buffer :buffer src-view :view} (resolve-resident-binding sess src)
        {dst-buffer :buffer dst-view :view} (resolve-resident-binding sess dst)
        src-spec (checked-view-range-spec src-buffer src-view spec :download)
        dst-spec (checked-view-range-spec dst-buffer dst-view spec :upload)
        src-dtype (dtype/canon (:dtype src-buffer))
        dst-dtype (dtype/canon (:dtype dst-buffer))]
    (when-not (= src-dtype dst-dtype)
      (throw (ex-info "resident range copy requires identical storage dtypes"
                      {:source-dtype src-dtype :destination-dtype dst-dtype})))
    ((rt-resolve (:device-id @sess) "copy-buffer-range!")
     src-buffer dst-buffer
     (:src-element src-spec) (:dst-element dst-spec) elements)
    dst))

(defn- plan-transfer-ranges
  "Resolve and validate every range before either synchronous or asynchronous execution."
  [sess entries direction]
  (let [device-id (:device-id @sess)
        plan (rt-resolve device-id "plan-range")]
    (mapv (fn [[key-or-view host spec]]
            (let [{:keys [buffer view]} (resolve-resident-binding sess key-or-view)
                  spec (checked-view-range-spec buffer view spec direction)]
              [buffer (plan buffer host spec direction) host]))
          entries)))

(defn- transfer-ranges!
  "The synchronous batched core. VALIDATES EVERY entry (`plan-range`) before EXECUTING ANY. A batch is how a
   whole KV cache moves — 36 per-layer buffers for gemma-270m — and a batched API newly makes a
   partial state possible: a bad spec in the 30th entry leaving 29 layers written. All-or-nothing
   on validation removes that class; nothing is copied until every range has been proved in
   bounds. (A failure DURING execution — a device fault — is still partial; that is a different
   class and is not promised here.)"
  [sess entries direction]
  (let [device-id (:device-id @sess)
        exec (rt-resolve device-id "execute-range!")
        plans (plan-transfer-ranges sess entries direction)]
    ;; phase 2: execute in order
    (mapv (fn [[buf p host]] (exec buf p direction) (if (= :upload direction) buf host)) plans)))

(defn upload-ranges!
  "BATCHED `upload-range!`: many `[key src spec]` entries in one call, e.g. every layer of a KV
   continuation:

     (upload-ranges! sess (for [l (range 18)]
                            [(keyword (str \"kc\" l)) (kc-segment l) {:elements (* tokens kvrow)}]))

   Every entry is bounds-validated BEFORE any is copied, so a bad spec cannot leave the cache
   half-restored. Returns the buffers, in entry order. This is the shape LMCache's
   `batched_to_gpu` has; ours is a loop over validated plans rather than a fused kernel because a
   position-major cache needs no gather."
  [sess entries]
  (transfer-ranges! sess entries :upload))

(defn download-ranges!
  "BATCHED `download-range!`: many `[key dst spec]` entries, validated all-or-nothing, executed
   in order. Returns the destinations, in entry order."
  [sess entries]
  (transfer-ranges! sess entries :download))

(defn buffer
  "Get a DeviceBuffer from the session by key."
  [sess key]
  (get-in @sess [:buffers key]))

;; ================================================================
;; Executable KernelGraphs
;; ================================================================

(defrecord KernelGraphHandle [key])
(defrecord GPUEvent [session-id id queue])

(defn kernel-graph-handle? [x]
  (and x (= "raster.gpu.core.KernelGraphHandle" (.getName (class x)))))

(defn gpu-event? [x]
  (and x (= "raster.gpu.core.GPUEvent" (.getName (class x)))))

(defn- submit-transfer-ranges!
  ([sess entries direction]
   (submit-transfer-ranges! sess entries direction []))
  ([sess entries direction retained-resources]
   (when-not (and (vector? retained-resources)
                  (every? #(instance? AutoCloseable %) retained-resources))
     (throw (ex-info "retained transfer resources must be a vector of AutoCloseable values"
                     {:direction direction :resources retained-resources})))
   (locking sess
     (let [{:keys [device-id session-id closed?]} @sess]
       (when closed?
         (throw (ex-info "cannot submit a transfer to a closed GPU session"
                         {:direction direction})))
       ;; Validation and host-side upload staging both complete before the event becomes visible.
       ;; The backend owns any native staging until await/release consumes its completion token.
       ;; Retained resources cover a stronger future path in which the backend borrows a mapped
       ;; source/destination directly: successful submission transfers their close responsibility
       ;; to this event, and completion releases them exactly once.
       (let [plans (plan-transfer-ranges sess entries direction)
             values (mapv (fn [[buffer _ host]]
                            (if (= :upload direction) buffer host))
                          plans)
             submitted-ns (System/nanoTime)
             backend-event ((rt-resolve device-id "submit-range-batch!")
                            (mapv (fn [[buffer plan _]] [buffer plan]) plans)
                            direction)
             submit-return-ns (System/nanoTime)
             event-id (random-uuid)
             event (->GPUEvent session-id event-id (execution/transfer-queue))]
         (swap! sess assoc-in [:events event-id]
                {:event event
                 :kind :transfer
                 :direction direction
                 :status :pending
                 :backend-event backend-event
                 :submitted-ns submitted-ns
                 :submit-return-ns submit-return-ns
                 :retained-resources retained-resources
                 :value values})
         event)))))

(defn submit-upload-ranges!
  "Validate and submit a batch of host-to-resident ranges without waiting.

   Returns a session-owned `GPUEvent` on the logical transfer queue. OpenCL submits through an
   independent physical in-order transfer queue and owns an immutable native staging copy until
   the event is awaited/released, so callers may reuse their source after submission. Level Zero
   shared allocations may complete inline; inspect `transfer-capabilities` and
   `event-measurement` rather than assuming physical overlap."
  [sess entries]
  (submit-transfer-ranges! sess entries :upload))

(defn submit-upload-ranges-retained!
  "Submit host-to-resident ranges and transfer ownership of scoped host resources to the event.

   `retained-resources` must be a vector of AutoCloseable leases. On successful submission they
   remain live until await-event! or release-event! establishes completion; session close also
   drains the event. If submission throws, ownership remains with the caller. This is the safe seam
   for mmap/LMDB payloads and future direct DMA that does not make an owned staging copy."
  [sess entries retained-resources]
  (submit-transfer-ranges! sess entries :upload retained-resources))

(defn submit-download-ranges!
  "Validate and submit a batch of resident-to-host ranges without waiting.

   Host destinations become observable only after `await-event!`, even if `event-complete?` reports
   device completion. Returns a session-owned `GPUEvent`; read its measured byte/time provenance
   with `event-measurement` after awaiting it."
  [sess entries]
  (submit-transfer-ranges! sess entries :download))

(defn submit-download-ranges-retained!
  "Download into scoped host resources retained through device completion. See
   submit-upload-ranges-retained! for ownership semantics."
  [sess entries retained-resources]
  (submit-transfer-ranges! sess entries :download retained-resources))

(defn- external-graph-buffer-ids
  [graph]
  (set (map :id (concat (:inputs graph) (:outputs graph)))))

(defn- write-access?
  [access]
  (contains? #{:write :read-write} access))

(defn- resolve-graph-elements
  [scalar-values expression]
  (let [bound (get scalar-values expression ::not-bound)
        value (if (= ::not-bound bound)
                (kgcall/resolve-integer scalar-values expression)
                (if (and (map? bound) (contains? bound :value)) (:value bound) bound))]
    (when-not (and (integer? value) (not (neg? value)))
      (throw (ex-info "kernel graph buffer extent must resolve to a non-negative integer"
                      {:expression expression :resolved value})))
    (long value)))

(defn- validate-external-view!
  [graph-buffer view scalar-values]
  (let [view (bview/validate-view! view)
        expected-dtype (dtype/canon (:dtype graph-buffer))
        actual-dtype (dtype/canon (:dtype view))
        expected-elements (when (some? (:elements graph-buffer))
                            (resolve-graph-elements scalar-values (:elements graph-buffer)))
        capacity (quot (:byte-length view) (dtype/bytes-of actual-dtype))]
    (when-not (= expected-dtype actual-dtype)
      (throw (ex-info "kernel graph buffer dtype differs from its resident view"
                      {:graph-buffer (:id graph-buffer) :expected expected-dtype
                       :actual actual-dtype :view (:id view)})))
    (when (and expected-elements (> expected-elements capacity))
      (throw (ex-info "kernel graph buffer extent exceeds its resident view"
                      {:graph-buffer (:id graph-buffer) :elements expected-elements
                       :view-elements capacity :view (:id view)})))
    (when-not (bview/contiguous? view)
      (throw (ex-info "kernel ABI binding currently requires a contiguous resident view"
                      {:graph-buffer (:id graph-buffer) :view (:id view)
                       :shape (:shape view) :strides (:strides view)})))
    view))

(defn- node-physical-hazard?
  [external-bindings earlier later]
  (boolean
   (some (fn [earlier-use]
           (some (fn [later-use]
                   (let [earlier-view (get-in external-bindings [(:buffer earlier-use) :view])
                         later-view (get-in external-bindings [(:buffer later-use) :view])]
                     (and earlier-view later-view
                          (bview/overlaps? earlier-view later-view)
                          (or (write-access? (:access earlier-use))
                              (write-access? (:access later-use))))))
                 (:uses later)))
         (:uses earlier))))

(defn- validate-physical-aliases!
  "Prove hazards introduced when distinct graph identities are bound to overlapping views. The
   symbolic graph validator cannot see these aliases, so binding must reject same-kernel writable
   aliases and require the same explicit dependency rule across kernels."
  [graph external-bindings]
  (doseq [node (:nodes graph)
          [index left] (map-indexed vector (:uses node))
          right (drop (inc index) (:uses node))]
    (let [left-view (get-in external-bindings [(:buffer left) :view])
          right-view (get-in external-bindings [(:buffer right) :view])]
      (when (and left-view right-view
                 (bview/overlaps? left-view right-view)
                 (or (write-access? (:access left)) (write-access? (:access right))))
        (throw (ex-info "one kernel cannot bind overlapping writable graph buffer views"
                        {:node (:id node) :left (:buffer left) :right (:buffer right)})))))
  (doseq [[later-index later] (map-indexed vector (:nodes graph))
          earlier (take later-index (:nodes graph))
          :when (node-physical-hazard? external-bindings earlier later)
          :when (not (contains? (set (:dependencies later)) (:id earlier)))]
    (throw (ex-info "kernel graph omits a dependency introduced by overlapping resident views"
                    {:node (:id later) :missing (:id earlier)})))
  external-bindings)

(defn- runtime-buffer-for-view
  [device-id buffer view]
  (let [raw-dtype (dtype/canon (:dtype buffer))
        view-dtype (dtype/canon (:dtype view))]
    (when-not (= raw-dtype view-dtype)
      (throw (ex-info "kernel ABI cannot reinterpret a resident buffer dtype"
                      {:buffer-dtype raw-dtype :view-dtype view-dtype :view (:id view)})))
    (if (zero? (:byte-offset view))
      {:buffer buffer :owned-view? false}
      (case (backend-type device-id)
        :ze {:buffer ((rt-resolve device-id "slice-buffer") buffer (:byte-offset view)
                                                            (:byte-length view) view-dtype)
             :owned-view? false}
        :ocl {:buffer ((rt-resolve device-id "slice-buffer") buffer (:byte-offset view)
                                                             (:byte-length view) view-dtype)
              :owned-view? true}))))

(defn- materialize-external-buffers!
  "Turn checked external BufferViews into backend ABI buffers. Level Zero slices are non-owning
   pointers. OpenCL slices are owned cl_mem sub-buffers and are transactionally released if any
   later view fails to materialize."
  [device-id external-bindings]
  (let [owned (volatile! [])]
    (try
      {:buffers
       (into {}
             (map (fn [[id {:keys [buffer view]}]]
                    (let [{runtime-buffer :buffer owned-view? :owned-view?}
                          (runtime-buffer-for-view device-id buffer view)]
                      (when owned-view? (vswap! owned conj runtime-buffer))
                      [id runtime-buffer])))
             external-bindings)
       :owned-view-buffers @owned}
      (catch Exception e
        (when (seq @owned)
          (let [free! (rt-resolve device-id "free-buffer!")]
            (doseq [buffer @owned]
              (try (free! buffer) (catch Exception _)))))
        (throw e)))))

(defn- resolve-kernel-graph-entry
  [sess handle]
  (when-not (kernel-graph-handle? handle)
    (throw (ex-info "kernel graph runner requires a KernelGraphHandle"
                    {:handle handle :actual (type handle)})))
  (or (get-in @sess [:kernel-graphs (:key handle)])
      (throw (ex-info "kernel graph is not bound in this session"
                      {:key (:key handle)
                       :bound (keys (:kernel-graphs @sess))}))))

(defn- resolve-event-entry
  [sess event]
  (when-not (gpu-event? event)
    (throw (ex-info "GPU event operation requires a GPUEvent"
                    {:event event :actual (type event)})))
  (when-not (= (:session-id @sess) (:session-id event))
    (throw (ex-info "GPU event belongs to a different session"
                    {:event event :session-id (:session-id @sess)})))
  (or (get-in @sess [:events (:id event)])
      (throw (ex-info "GPU event is no longer owned by this session"
                      {:event event :events (keys (:events @sess))}))))

(defn- close-retained-resources
  [resources]
  (reduce (fn [errors resource]
            (try
              (.close ^AutoCloseable resource)
              errors
              (catch Exception error
                (conj errors error))))
          []
          (reverse resources)))

(defn- await-event-under-lock!
  [sess event]
  (let [{:keys [device-id closed?]} @sess
        {:keys [status backend-event kind submitted-ns submit-return-ns retained-resources]
         :as entry}
        (resolve-event-entry sess event)]
    (when closed?
      (throw (ex-info "cannot use an event from a closed GPU session" {:event event})))
    (if (= :complete status)
      entry
      ;; A successful status query is not necessarily a host memory-synchronization point
      ;; (notably in OpenCL). Await always calls the backend wait before releasing the token.
      (let [backend-completion ((rt-resolve device-id "await-event!") backend-event)
            completed-ns (System/nanoTime)
            measurement (when (= :transfer kind)
                          (merge (when (map? backend-completion) backend-completion)
                                 {:host-wall-ns (- completed-ns submitted-ns)
                                  :submit-host-ns (- submit-return-ns submitted-ns)}))]
        ((rt-resolve device-id "release-event!") backend-event)
        (let [release-errors (close-retained-resources retained-resources)
              completed (cond-> (assoc entry
                                       :status :complete
                                       :backend-event nil
                                       :retained-resources [])
                          measurement (assoc :measurement measurement)
                          (seq release-errors) (assoc :retention-release-errors release-errors))]
          ;; Record completion before surfacing a lease-close failure: the native event has already
          ;; been consumed and must never be released a second time.
          (swap! sess assoc-in [:events (:id event)] completed)
          (when (seq release-errors)
            (throw (ex-info "transfer completed but retained resource release failed"
                            {:event event :errors release-errors} (first release-errors))))
          completed)))))

(defn event-complete?
  "Return true when a GPUEvent has completed, without blocking. This is a status query, not a
   substitute for await-event!: native handles remain live until a host wait establishes memory
   visibility or release-event! safely consumes the event."
  [sess event]
  (locking sess
    (let [{:keys [device-id closed?]} @sess
          {:keys [status backend-event]} (resolve-event-entry sess event)]
      (when closed?
        (throw (ex-info "cannot use an event from a closed GPU session" {:event event})))
      (or (= :complete status)
          ((rt-resolve device-id "event-complete?") backend-event)))))

(defn await-event!
  "Block until a GPUEvent completes and return the submission's value. For a KernelGraph this is
   its resident output-buffer map. Waiting is idempotent until release-event! consumes the event."
  [sess event]
  (locking sess
    (:value (await-event-under-lock! sess event))))

(defn event-measurement
  "Return a completed transfer event's timing and byte-count measurement.

   The event must first be awaited: a nonblocking completion observation does not establish host
   visibility or finalize download staging. Kernel-graph events do not carry this one-shot transfer
   measurement; graph profiling remains available through `measure-graph!`."
  [sess event]
  (locking sess
    (let [{:keys [status kind measurement]} (resolve-event-entry sess event)]
      (when-not (= :transfer kind)
        (throw (ex-info "event does not describe a transfer submission"
                        {:event event :kind kind})))
      (when-not (= :complete status)
        (throw (ex-info "transfer event must be awaited before reading its measurement"
                        {:event event :status status})))
      measurement)))

(defn release-event!
  "Establish completion and remove a session-owned GPUEvent. This is deliberately safe rather
   than cancellation-like: releasing an in-flight event waits before permitting its graph or
   buffers to be destroyed."
  [sess event]
  (locking sess
    (await-event-under-lock! sess event)
    (swap! sess update :events dissoc (:id event)))
  nil)

(defn- release-graph-events!
  [sess graph-key]
  (doseq [event (->> (:events @sess)
                     vals
                     (filter #(= graph-key (:graph-key %)))
                     (map :event)
                     vec)]
    (release-event! sess event)))

(defn bind-kernel-graph!
  "Bind an emitted KernelGraph for repeated resident execution.

   `buffer-keys` maps every external GraphBuffer identity to an existing session-buffer key or a
   ResidentBufferView. Distinct graph identities may share an allocation when their physical
   ranges and declared accesses are legal. `scalar-values` maps symbolic compiler values to
   explicitly typed runtime scalars, e.g. `{'n {:type :int :value 4096}}`.

   The graph owns its temporary allocations and dedicated bound kernel handles. Binding validates
   the complete graph call, lowers dependencies to logical queue/event edges, then records a
   conservative dependency-safe sequence. Current backends map the logical plan to one in-order
   compute queue; submit-kernel-graph! exposes asynchronous completion without native handles.

   Option :profile? records device timestamp events for explicit offline measurement."
  ([sess graph-key graph buffer-keys scalar-values]
   (bind-kernel-graph! sess graph-key graph buffer-keys scalar-values {}))
  ([sess graph-key graph buffer-keys scalar-values {:keys [profile?]
                                                    :or {profile? false}}]
   (let [{:keys [device-id closed?]} @sess
         graph (kexec/validate! graph)
         external-ids (external-graph-buffer-ids graph)]
     (when closed?
       (throw (ex-info "cannot bind a kernel graph in a closed GPU session" {:key graph-key})))
     (when-not (= external-ids (set (keys buffer-keys)))
       (throw (ex-info "kernel graph external bindings differ from graph inputs/outputs"
                       {:expected external-ids :bound (set (keys buffer-keys))})))
     (let [graph-buffer-by-id (into {} (map (juxt :id identity))
                                    (concat (:inputs graph) (:outputs graph)))
           external-bindings (into {}
                                   (map (fn [[id key-or-view]]
                                          [id (resolve-resident-binding sess key-or-view)]))
                                   buffer-keys)
           _ (doseq [[id {:keys [view]}] external-bindings]
               (validate-external-view! (get graph-buffer-by-id id) view scalar-values))
           _ (validate-physical-aliases! graph external-bindings)
           _ (release-graph-events! sess graph-key)
           temporary-specs (kgcall/temporary-specs graph scalar-values)
           temporary-buffers (alloc-buffers-transactional temporary-specs device-id)
           owned-view-buffers (volatile! [])
           prepareds (volatile! [])
           runtime-graph (volatile! nil)]
       (try
         (let [{:keys [buffers] :as materialized}
               (materialize-external-buffers! device-id external-bindings)
               _ (vreset! owned-view-buffers (:owned-view-buffers materialized))
               all-buffers (merge buffers temporary-buffers)
               register! (rt-resolve device-id "register-kernel!")
               bind-call! (rt-resolve device-id "bind-kernel-call")
               record! (rt-resolve device-id "record-graph!")
               graph-call (kgcall/make graph all-buffers scalar-values)
               execution-plan (execution/from-kernel-graph-call graph-call)]
           (doseq [node (:nodes graph)]
             (let [artifact (:operation node)]
               (register! (:kernel-name artifact) artifact)))
           (doseq [node-call (:nodes graph-call)]
             (let [prepared (assoc (bind-call! (:call node-call))
                                   :phase (:id node-call))]
               (vswap! prepareds conj prepared)))
          ;; Graph verification proves every dependency names an earlier node and every hazard is
          ;; represented. Serial recording is therefore a safe implementation of that partial
          ;; order on today's single in-order compute queues; the logical plan retains the DAG.
           (vreset! runtime-graph (record! @prepareds (cond-> {:barriers? true}
                                                        profile? (assoc :profile? true))))
           (let [entry {:graph-call graph-call
                        :execution-plan execution-plan
                        :runtime-graph @runtime-graph
                        :prepareds @prepareds
                        :owned-view-buffers @owned-view-buffers
                        :temporary-buffers temporary-buffers
                        :buffer-keys buffer-keys
                        :resident-views (into {} (map (fn [[id binding]]
                                                        [id (:resident binding)]))
                                              external-bindings)
                        :outputs (select-keys all-buffers (map :id (:outputs graph)))
                        :profile? (boolean profile?)}
                 old (get-in @sess [:kernel-graphs graph-key])]
             (swap! sess assoc-in [:kernel-graphs graph-key] entry)
             (when old (destroy-kernel-graph-entry! device-id old))
             (->KernelGraphHandle graph-key)))
         (catch Exception e
           (destroy-kernel-graph-entry!
            device-id {:runtime-graph @runtime-graph
                       :prepareds @prepareds
                       :owned-view-buffers @owned-view-buffers
                       :temporary-buffers temporary-buffers})
           (throw e)))))))

(defn bind-kernel-call!
  "Bind one emitted KernelArtifact over session-resident arguments as a replayable graph.

   `arguments` follow the artifact's PHYSICAL ABI order. Pointer entries are session buffer keys
   or ResidentBufferViews; scalar entries are typed values such as `{:type :int :value 4096}`.
   The resulting KernelCall is checked before driver contact and then recorded through the same
   queue/event machinery as a multi-kernel graph. This is the generic runtime seam used to
   validate and device-time one schedule alternative during explicit offline tuning.

   Options: :profile? records device timestamp events; :group-count may override only the realized
   grid, never the emitted workgroup geometry. Rebinding `call-key` replaces and releases the old
   recording transactionally after the new recording succeeds."
  ([sess call-key artifact arguments]
   (bind-kernel-call! sess call-key artifact arguments {}))
  ([sess call-key artifact arguments {:keys [profile? group-count]
                                      :or {profile? false}}]
   (let [{:keys [device-id closed?]} @sess
         artifact (kart/validate! artifact)
         abi (:abi artifact)]
     (when closed?
       (throw (ex-info "cannot bind a kernel call in a closed GPU session" {:key call-key})))
     (when (nil? call-key)
       (throw (ex-info "bound kernel call requires a non-nil key" {})))
     (kabi/validate-arguments! abi arguments)
     (let [pointer-bindings
           (into {}
                 (keep-indexed (fn [index [slot value]]
                                 (when-not (= :scalar (:kind slot))
                                   [index (resolve-resident-binding sess value)])))
                 (mapv vector abi arguments))
           _ (release-graph-events! sess call-key)
           owned-view-buffers (volatile! [])
           prepareds (volatile! [])
           runtime-graph (volatile! nil)]
       (try
         (let [{:keys [buffers] :as materialized}
               (materialize-external-buffers! device-id pointer-bindings)
               _ (vreset! owned-view-buffers (:owned-view-buffers materialized))
               runtime-arguments
               (mapv (fn [index [slot value]]
                       (if (= :scalar (:kind slot)) value (get buffers index)))
                     (range) (mapv vector abi arguments))
               call (kcall/make artifact runtime-arguments
                                (cond-> {} group-count (assoc :group-count group-count)))
               execution-plan (execution/from-kernel-call call-key call)
               register! (rt-resolve device-id "register-kernel!")
               bind-call! (rt-resolve device-id "bind-kernel-call")
               record! (rt-resolve device-id "record-graph!")
               _ (register! (:kernel-name artifact) artifact)
               prepared (assoc (bind-call! call) :phase call-key)
               _ (vreset! prepareds [prepared])
               _ (vreset! runtime-graph
                          (record! [prepared] {:barriers? true
                                               :profile? (boolean profile?)}))
               outputs
               (into {}
                     (keep-indexed
                      (fn [index slot]
                        (when (kabi/writable? slot)
                          [(or (:binding slot) (:name slot))
                           (nth runtime-arguments index)])))
                     abi)
               entry {:kernel-call call
                      :execution-plan execution-plan
                      :runtime-graph @runtime-graph
                      :prepareds @prepareds
                      :owned-view-buffers @owned-view-buffers
                      :temporary-buffers {}
                      :outputs outputs
                      :profile? (boolean profile?)}
               old (get-in @sess [:kernel-graphs call-key])]
           (swap! sess assoc-in [:kernel-graphs call-key] entry)
           (when old (destroy-kernel-graph-entry! device-id old))
           (->KernelGraphHandle call-key))
         (catch Exception e
           (destroy-kernel-graph-entry!
            device-id {:runtime-graph @runtime-graph
                       :prepareds @prepareds
                       :owned-view-buffers @owned-view-buffers
                       :temporary-buffers {}})
           (throw e)))))))

(defn bind-kernel-executable!
  "Bind one KernelArtifact or emitted KernelGraph through its common ordered external ABI.

   Graph alternatives project pointer arguments to external GraphBuffer identities and typed
   scalar arguments to the graph's symbolic scalar environment. Graph-owned temporaries and
   derived bounds stay private. :group-count is meaningful only for a single artifact."
  ([sess executable-key executable arguments]
   (bind-kernel-executable! sess executable-key executable arguments {}))
  ([sess executable-key executable arguments {:keys [group-count] :as options}]
   (let [executable (kexec/validate! executable)]
     (case (kexec/kind executable)
       :kernel-artifact
       (bind-kernel-call! sess executable-key executable arguments options)

       :kernel-graph
       (do
         (when group-count
           (throw (ex-info "group-count cannot override a multi-kernel graph schedule"
                           {:key executable-key :group-count group-count})))
         (let [{:keys [buffers scalar-values]} (kexec/graph-bindings executable arguments)]
           (bind-kernel-graph! sess executable-key executable buffers scalar-values
                               (select-keys options [:profile?]))))))))

(defn- staged-pointer-plan
  "Validate and group ABI pointer values by object identity before any allocation.

   One host array or resident buffer supplied at several ABI positions receives one session
   allocation/registration, preserving legal in-place calls and exposing illegal stable-input
   aliases to the ordinary graph validator."
  [device-id executable typed-arguments]
  (let [device-buffer? (rt-resolve device-id "device-buffer?")
        identities (java.util.IdentityHashMap.)
        groups (volatile! [])]
    (doseq [[index [slot value]] (map-indexed vector (map vector (kexec/abi executable)
                                                          typed-arguments))
            :when (not= :scalar (:kind slot))]
      (let [array-dtype (dtype/dtype-for-jvm-array value)
            resident? (boolean (device-buffer? value))
            actual-dtype (cond array-dtype array-dtype resident? (dtype/canon (:dtype value)))]
        (when-not actual-dtype
          (throw (ex-info "staged kernel executable pointer is not a primitive array or backend buffer"
                          {:device-id device-id :index index :slot slot
                           :actual (type value)})))
        (when-not (= (dtype/canon (:dtype slot)) actual-dtype)
          (throw (ex-info "staged kernel executable pointer has the wrong storage dtype"
                          {:device-id device-id :index index :slot slot
                           :expected (dtype/canon (:dtype slot)) :actual actual-dtype})))
        (let [existing (.get identities value)
              read? (kabi/readable? slot)
              write? (kabi/writable? slot)]
          (if (some? existing)
            (vswap! groups update existing
                    (fn [group]
                      (-> group
                          (update :indexes conj index)
                          (update :read? #(or % read?))
                          (update :write? #(or % write?)))))
            (let [group-index (count @groups)
                  key [:staged-executable-pointer group-index]
                  elements (if array-dtype
                             (java.lang.reflect.Array/getLength value)
                             (:n-elements value))]
              (.put identities value group-index)
              (vswap! groups conj {:key key :value value :dtype actual-dtype
                                   :elements elements :resident? resident?
                                   :indexes [index] :read? read? :write? write?}))))))
    (let [groups @groups
          key-by-index (reduce (fn [result {:keys [key indexes]}]
                                 (reduce #(assoc %1 %2 key) result indexes))
                               {} groups)]
      {:groups groups
       :arguments (mapv (fn [index [slot value]]
                          (if (= :scalar (:kind slot)) value (get key-by-index index)))
                        (range) (map vector (kexec/abi executable) typed-arguments))})))

(declare run-kernel-graph!)

(defn invoke-staged-executable!
  "Execute a registered KernelDispatch through the common KernelExecutable session machinery.

   This is the ordinary compiled-function staging contract: JVM primitive arrays are copied in
   and ABI-declared writes are copied back; backend DeviceBuffers are borrowed without copying.
   Every call owns and closes its graph recording and temporary buffers. Long-lived/replayable
   execution belongs to compile-gpu-program plus LinkPlan, which uses the same executable binder."
  [device-id dispatch-id runtime-arguments]
  (let [dispatch-entry (rt-resolve device-id "kernel-dispatch-registry-entry")
        dispatch (or (dispatch-entry dispatch-id)
                     (throw (ex-info "staged kernel executable dispatch is not registered"
                                     {:device-id device-id :dispatch-id dispatch-id})))
        dispatch (kdispatch/validate! dispatch)
        common (kdispatch/default-alternative dispatch)
        typed-arguments (kexec/typed-runtime-arguments common runtime-arguments)
        executable (kdispatch/select-alternative dispatch typed-arguments)
        {:keys [groups arguments]} (staged-pointer-plan device-id executable typed-arguments)
        result-pairs (filterv (fn [[slot _]] (= :result (:role slot)))
                              (map vector (kexec/abi executable) typed-arguments))]
    (when (> (count result-pairs) 1)
      (throw (ex-info "staged kernel executable has more than one primary result"
                      {:dispatch-id dispatch-id :result-slots (mapv first result-pairs)})))
    (with-gpu-session* device-id
      (fn [session]
        (doseq [{:keys [key value dtype elements resident? read?]} groups]
          (if resident?
            (register-buffer! session key value {:ownership :borrowed})
            (alloc! session {key [dtype elements (when read? value)]})))
        (let [handle (bind-kernel-executable! session
                                              [:staged-executable dispatch-id]
                                              executable arguments)]
          (run-kernel-graph! session handle)
          (doseq [{:keys [key value elements resident? write?]} groups
                  :when (and write? (not resident?))]
            (download-range! session key value {:elements elements})))
        (some-> (first result-pairs) second)))))

(defn kernel-graph-execution-plan
  "Return the pure backend-neutral queue/event plan for a bound KernelGraph."
  [sess handle]
  (:execution-plan (resolve-kernel-graph-entry sess handle)))

(defn submit-kernel-graph!
  "Submit a bound graph without waiting and return a session-owned GPUEvent.

   One submission per graph may be in flight. This common guarantee matches Level Zero's
   replayable command-list ownership and remains correct on OpenCL's in-order queue. Await or
   release the prior event before submitting the same graph again."
  [sess handle]
  (locking sess
    (let [{:keys [device-id session-id closed? events]} @sess]
      (when closed?
        (throw (ex-info "cannot submit a kernel graph in a closed GPU session" {:handle handle})))
      (let [{:keys [runtime-graph outputs execution-plan]}
            (resolve-kernel-graph-entry sess handle)
            pending (some (fn [[_ entry]]
                            (when (and (= (:key handle) (:graph-key entry))
                                       (= :pending (:status entry)))
                              (:event entry)))
                          events)]
        (when pending
          (throw (ex-info "kernel graph already has an in-flight submission"
                          {:handle handle :event pending})))
        (let [backend-event ((rt-resolve device-id "submit-graph!") runtime-graph)
              event-id (random-uuid)
              queue (first (:queues execution-plan))
              event (->GPUEvent session-id event-id queue)]
          (swap! sess assoc-in [:events event-id]
                 {:event event
                  :graph-key (:key handle)
                  :status :pending
                  :backend-event backend-event
                  :value outputs})
          event)))))

(defn run-kernel-graph!
  "Submit a bound graph, wait for completion, and return its resident output buffers."
  [sess handle]
  (let [{:keys [device-id]} @sess
        {:keys [runtime-graph profile?]} (resolve-kernel-graph-entry sess handle)
        event (submit-kernel-graph! sess handle)]
    (try
      (let [outputs (await-event! sess event)]
        ;; A validation replay of a profiled graph intentionally discards its timestamps. Reset
        ;; the per-kernel events so a subsequent measure-graph! replay can signal them legally.
        (when profile?
          ((rt-resolve device-id "reset-graph-events!") runtime-graph))
        outputs)
      (finally
        (release-event! sess event)))))

(defn release-kernel-graph!
  "Release one bound graph and its graph-owned temporaries. External session buffers survive.
   Any in-flight submission is completed and released before its resources are destroyed."
  [sess handle]
  (when-not (kernel-graph-handle? handle)
    (throw (ex-info "release-kernel-graph! requires a KernelGraphHandle"
                    {:handle handle :actual (type handle)})))
  (locking sess
    (release-graph-events! sess (:key handle))
    (when-let [entry (get-in @sess [:kernel-graphs (:key handle)])]
      (swap! sess update :kernel-graphs dissoc (:key handle))
      (destroy-kernel-graph-entry! (:device-id @sess) entry)))
  nil)

;; ================================================================
;; Resident GPU programs (Option A: pipeline → bound-dispatch path)
;; ================================================================

(defn- allocate-executable-temporaries
  [device-id temporary-specs]
  (let [make-buffer (rt-resolve device-id "make-buffer")
        allocated (volatile! {})]
    (try
      (doseq [[id [temporary-dtype elements _]] temporary-specs]
        (vswap! allocated assoc id (make-buffer elements temporary-dtype)))
      @allocated
      (catch Exception e
        (when (seq @allocated)
          (free-buffers-internal! @allocated device-id))
        (throw e)))))

(defn- step-selection-override
  [step schedule]
  (if-let [{:keys [path mapping default]} (:strategy-selection step)]
    (get mapping (get-in schedule path) default)
    (when (= :reduce (:convention step))
      (get-in schedule [:segmented-weighted-reduction :strategy] :auto))))

(defn- bind-selected-executable
  "Bind one selected KernelArtifact/KernelGraph without recording it. The returned value is the
   common per-step ownership unit used by whole programs and descriptor composition."
  [device-id executable runtime-arguments phase constant-buffer-ids group-count]
  (let [executable (kexec/validate! executable)
        register! (rt-resolve device-id "register-kernel!")
        bind-call! (rt-resolve device-id "bind-kernel-call")]
    (case (kexec/kind executable)
      :kernel-artifact
      (do
        (register! (:kernel-name executable) executable)
        (->BoundExecutableStep
         [(assoc (bind-call! (kcall/make executable runtime-arguments
                                         (cond-> {} group-count
                                                 (assoc :group-count group-count))))
                 :phase phase)] {} []))

      :kernel-graph
      (let [{:keys [buffers scalar-values]}
            (kexec/graph-bindings executable runtime-arguments)
            temporary-buffers
            (allocate-executable-temporaries
             device-id (kgcall/temporary-specs executable scalar-values))
            prepareds (volatile! [])]
        (try
          (let [graph-call (kgcall/make executable (merge buffers temporary-buffers) scalar-values)]
            ;; Cacheable transforms form a graph-generic constant prologue: once every read is a
            ;; captured constant, its private writes are constant for dependent transforms too.
            (loop [scheduled-nodes (:nodes executable)
                   called-nodes (:nodes graph-call)
                   constants (set constant-buffer-ids)]
              (when-let [scheduled-node (first scheduled-nodes)]
                (let [called-node (first called-nodes)
                      artifact (get-in called-node [:call :artifact])
                      reads (->> (:uses scheduled-node)
                                 (filter #(contains? #{:read :read-write} (:access %)))
                                 (map :buffer) set)
                      writes (->> (:uses scheduled-node)
                                  (filter #(contains? #{:write :read-write} (:access %)))
                                  (map :buffer) set)
                      constant? (and (true? (get-in artifact
                                                    [:attributes :cacheable-transform?]))
                                     (every? constants reads))]
                  (register! (:kernel-name artifact) artifact)
                  (vswap! prepareds conj
                          (assoc (bind-call! (:call called-node))
                                 :phase (or (:id scheduled-node) phase)
                                 :const-prologue? constant?))
                  (recur (next scheduled-nodes) (next called-nodes)
                         (if constant? (into constants writes) constants)))))
            (->BoundExecutableStep @prepareds temporary-buffers []))
          (catch Exception e
            (destroy-prepared-entry!
             device-id (->BoundExecutableStep @prepareds temporary-buffers []))
            (throw e)))))))

(defn- bind-resident-step
  "The single descriptor-step binder. `resolve-buffer` maps a compiler array symbol to a resident
   DeviceBuffer. Both whole-program binding and composition call this function; convention-specific
   runtime expansion is confined here."
  [device-id step args resolve-buffer schedule roles]
  (let [{:keys [kernel-name phase convention artifact argument-specs arrays n-fn scalar-specs]}
        step]
    (case convention
      (:map :reduce :map-void :contract :gemm :executable)
      (let [logical-or-physical-args
            (mapv (fn [{:keys [kind sym type value-fn]}]
                    (if (= :scalar kind)
                      {:type type :value (value-fn args)}
                      (resolve-buffer sym)))
                  argument-specs)
            interface (or artifact
                          (some-> (:dispatch step) kdispatch/default-alternative))
            ordered-args
            (if (:logical-bindings? step)
              (kcall/expand-logical-arguments
               interface logical-or-physical-args
               (rt-resolve device-id "expand-pointer-binding"))
              logical-or-physical-args)
            selected (if-let [dispatch (:dispatch step)]
                       (kdispatch/select-alternative
                        dispatch ordered-args (step-selection-override step schedule))
                       artifact)
            constant-buffer-ids
            (when (= :kernel-graph (kexec/kind selected))
              (into #{}
                    (keep (fn [{:keys [id]}]
                            (when (= :constant (get roles id)) id)))
                    (:inputs selected)))
            ;; A resident SegRed is deliberately a single-workgroup schedule. KernelGraphs own
            ;; their complete launch geometry and therefore cannot accept this artifact override.
            group-count (when (and (= :reduce convention)
                                   (= :kernel-artifact (kexec/kind selected)))
                          [1])]
        (bind-selected-executable
         device-id selected ordered-args phase constant-buffer-ids group-count))

      :scatter
      (do
        (when (not= :ze (backend-type device-id))
          (throw (ex-info "resident scatter steps are Level-Zero-only (no OpenCL implementation yet)"
                          {:backend (backend-type device-id)
                           :device-id device-id :step kernel-name})))
        (when (> (count scalar-specs) 1)
          (throw (ex-info (str "resident scatter step " kernel-name " has "
                               (count scalar-specs)
                               " scalars — only a single stride is modeled")
                          {:kernel kernel-name :scalar-specs scalar-specs})))
        (let [[out-sym src-sym idx-sym] arrays
              out-buf (resolve-buffer out-sym)
              src-buf (resolve-buffer src-sym)
              idx-buf (resolve-buffer idx-sym)
              n (long (n-fn args))
              stride (when (seq scalar-specs)
                       (long ((:value-fn (first scalar-specs)) args)))
              ensure-zero! (rt-resolve device-id "ensure-zero-fill-kernel!")
              specialized-bind! (rt-resolve device-id "bind-registered-map-void-kernel")
              scatter-bind! (rt-resolve device-id "bind-registered-scatter-kernel!")
              zero-kernel (ensure-zero! (:dtype out-buf))
              prepareds (volatile! [])]
          (try
            (vswap! prepareds conj
                    (assoc (specialized-bind! zero-kernel [out-buf] []
                                              (long (:n-elements out-buf)))
                           :phase phase))
            (vswap! prepareds conj
                    (assoc (scatter-bind! kernel-name [out-buf src-buf idx-buf] n stride)
                           :phase phase))
            (->BoundExecutableStep @prepareds {} [])
            (catch Exception e
              (destroy-prepared-entry! device-id (->BoundExecutableStep @prepareds {} []))
              (throw e)))))

      (throw (ex-info (str "resident step binder cannot bind a " convention " step ("
                           kernel-name ")")
                      {:convention convention :kernel kernel-name})))))

(defn bind-step!
  "Bind ONE compiled kernel step (a descriptor :steps entry) into the session under its :phase.
   Resolves the kernel's array args to resident buffers via `sym->key` (arg-name-symbol → session
   buffer-key keyword) and its scalars via the step's value-fns over `args`. Handles executable
   conventions uniformly—the per-step core used by LinkPlan instantiation and multi-instance
   decoder composition. Does NOT record a graph; the caller collects :phase keys and records once.

     :map/:reduce/:map-void/:contract/:gemm
                Select one ABI-compatible KernelArtifact or KernelGraph schedule. Graph-private
                conversion/layout/split temporaries remain owned by the bound step.
     :scatter   Expands to zero-fill + scatter behind the same ordered prepared-step boundary.

   Optional opts carry descriptor context needed by composition: {:schedule <resolved schedule>
   :roles {compiler-sym -> :constant|...}}. Captured constants make eligible graph transforms a
   one-time prologue when record-graph! links the phases."
  ([sess step args sym->key]
   (bind-step! sess step args sym->key {}))
  ([sess step args sym->key {:keys [schedule roles] :or {roles {}}}]
   (let [device-id (:device-id @sess)
         {:keys [kernel-name phase]} step
         materialized (volatile! {})
         owned-view-buffers (volatile! [])
         materialize
         (fn materialize [sym key-or-view]
           (cond
             (resident-value/resident-composite? key-or-view)
             (resident-value/map-values #(materialize sym %) key-or-view)

             (resident-buffer-view? key-or-view)
             (or (get @materialized key-or-view)
                 (let [{:keys [buffer view]} (resolve-resident-binding sess key-or-view)
                       _ (when-not (bview/contiguous? view)
                           (throw (ex-info
                                   "resident descriptor binding requires a contiguous view"
                                   {:kernel kernel-name :symbol sym :view (:id view)
                                    :shape (:shape view) :strides (:strides view)})))
                       {runtime-buffer :buffer owned-view? :owned-view?}
                       (runtime-buffer-for-view device-id buffer view)]
                   (when owned-view?
                     (vswap! owned-view-buffers conj runtime-buffer))
                   (vswap! materialized assoc key-or-view runtime-buffer)
                   runtime-buffer))

             :else
             (or (get-in @sess [:buffers key-or-view])
                 (throw (ex-info (str "No buffer for kernel arg: " sym " → " key-or-view)
                                 {:kernel kernel-name
                                  :available (keys (:buffers @sess))})))))
         resolve-buf (fn [sym] (materialize sym (sym->key sym)))
         bound-step
         (try
           (assoc (bind-resident-step device-id step args resolve-buf schedule roles)
                  :owned-view-buffers @owned-view-buffers)
           (catch Exception e
             (when (seq @owned-view-buffers)
               (when-let [free! (rt-resolve-soft device-id "free-buffer!")]
                 (doseq [buffer @owned-view-buffers]
                   (try (free! buffer) (catch Exception _)))))
             (throw e)))
         old (get-in @sess [:prepared phase])]
     (swap! sess assoc-in [:prepared phase] bound-step)
     (destroy-prepared-entry! device-id old)
     sess)))

;; ----------------------------------------------------------------
;; Hand-authored op-chain (the manual resident decoder layer — gemma-first; converges to a single
;; fused compile-gpu-program later). Each op is compiled individually and chained into ONE command
;; graph over SHARED resident buffers with residency roles, so per token only :input moves up and
;; :output moves down; :constant weights + :state KV stay resident.
;; ----------------------------------------------------------------

(defn- typed-scalars-for
  "Build the ordered, typed scalar arg list for a chain step from the kernel's :scalar-params and
   the deftm's DECLARED scalar types — so hand-wiring can't mis-order or mis-type. :scalar-params
   EXCLUDES the par bound (it becomes the work-item count _n_bound, passed separately as :n), and
   the declared type is what the kernel actually emits (e.g. a Double `scale` is emitted float at
   :float dtype). scalars = {param-name → value} from the caller, in any order. Mirrors
   compile-gpu-program's scalar typing (same derive-param-types) — not a special case."
  [op ki scalars dtype]
  (let [v (or (resolve-deftm-var op) op)
        scalar-params (or (:scalar-params ki) (get-in ki [:attributes :scalar-params]))
        types (:scalar-types (opencl-pass/derive-param-types
                              (:raster.core/deftm-params (meta v))
                              (:raster.core/deftm-tags (meta v)) dtype))]
    (mapv (fn [sp]
            (let [t (get types sp :float)
                  raw (if (contains? scalars (name sp)) (get scalars (name sp))
                          (if (contains? scalars sp) (get scalars sp)
                              (throw (ex-info (str "chain step for " (:name (meta v))
                                                   " missing scalar: " sp)
                                              {:need scalar-params :have (keys scalars)}))))]
              {:type t :value (case t :int (int raw) :long (long raw) :double (double raw) (float raw))}))
          scalar-params)))

(def ^:private valid-chain-roles
  "Residency roles a chain buffer may carry. run-chain!/run-chain-ctx! move :input up and
   download :output; :constant/:state/:scratch stay put. A role OUTSIDE this set (e.g. a
   typo `:ouput`) is never matched by the `(= r :output)` filters and would SILENTLY never
   be downloaded — so an unknown role is rejected at bind time, not lost at replay."
  #{:constant :state :input :output :scratch})

(defn- chain-roles-of
  "Extract {buf-key → role} from the buffers spec, defaulting :scratch, and REJECT any
   unknown role by name rather than storing it (where it would silently never download)."
  [buffers]
  (into {}
        (map (fn [[k v]]
               (let [r (or (nth v 3 nil) :scratch)]
                 (when-not (contains? valid-chain-roles r)
                   (throw (ex-info (str "chain buffer " k " has unknown role " (pr-str r)
                                        " — must be one of " valid-chain-roles)
                                   {:buffer k :role r :valid valid-chain-roles})))
                 [k r])))
        buffers))

(defn chain-program!
  "Bind a hand-authored op-chain as one resident command graph.
     buffers: {buf-key → [dtype size init-array-or-nil role]} — role ∈
              #{:constant :state :input :output :scratch} (default :scratch).
     steps:   [{:op #'deftm-var :phase kw
                :bind {kernel-param-name-string → buf-key}   ; EACH array param → a buffer
                :scalars {param-name → value}                ; NON-bound scalars, any order/raw values
                :n work-items}]                              ; the par bound (work-item count)
   Allocates the buffers (uploading any init array), compiles each op, prepares each step against
   the shared buffers (scalars ordered + typed from the kernel signature), and records the kernel
   sequence under :chain. run-chain! then replays it. dtype defaults :float (GPU decode)."
  ([sess buffers steps] (chain-program! sess buffers steps :float))
  ([sess buffers steps dtype]
   (let [specs (into {} (map (fn [[k [dt sz init _]]] [k [dt sz init]]) buffers))
         roles (chain-roles-of buffers)]
     (alloc! sess specs)
     ;; A multi-par-form op compiles to SEVERAL kernels — bind them ALL, in order (the old
     ;; `first` silently dropped every kernel after the first). All kernels of a step share
     ;; the step's :bind and :n (ops whose par forms need different bounds don't fit the
     ;; chain API's single :n — use compile-gpu-program/bind-step! for those).
     (let [all-phases
           (vec (mapcat
                 (fn [{:keys [op phase bind scalars n]}]
                   (compile! sess phase op)
                   (let [ks (get-in @sess [:kernels phase])]
                     (mapv (fn [i]
                             (let [ph (if (zero? (long i)) phase
                                          (keyword (str (name phase) "_" i)))
                                   tsc (typed-scalars-for op (nth ks i) (or scalars {}) dtype)]
                               (prepare! sess ph bind tsc (long n)
                                         {:kernel-phase phase :index i})
                               ph))
                           (range (count ks)))))
                 steps))]
       (record-graph! sess all-phases :chain))
     (swap! sess assoc :chain-roles roles)
     sess)))

(defn run-chain!
  "Replay a bound op-chain: refresh the given :input buffers (buffer POINTERS stable), replay the
   graph, download the :output buffers. inputs = {buf-key → jvm-array}. Returns
   {output-buf-key → downloaded jvm-array}. :constant/:state/:scratch buffers are never moved."
  [sess inputs]
  (let [roles (:chain-roles @sess)]
    (doseq [[k arr] inputs] (upload! sess k arr))
    (replay! sess :chain)
    (into {} (for [[k r] roles :when (= r :output)] [k (download sess k)]))))

(defn bind-chain!
  "Allocate the resident buffers + compile each op of a multi-step chain ONCE. Buffers
   (:constant weights, :state KV cache sized to MAX positions, :scratch) persist across
   run-chain-ctx! calls. Steps + roles are stored on the session. This is the decode-loop split of
   chain-program!: bind once, then run-chain-ctx! per token re-prepares only the position-dependent
   steps while weights + KV stay resident on-device."
  ([sess buffers steps] (bind-chain! sess buffers steps :float))
  ([sess buffers steps dtype]
   (let [specs (into {} (map (fn [[k [dt sz init _]]] [k [dt sz init]]) buffers))
         roles (chain-roles-of buffers)]
     (alloc! sess specs)
     (doseq [{:keys [op phase]} steps] (compile! sess phase op))
     (swap! sess assoc :chain-steps steps :chain-roles roles :chain-dtype dtype)
     sess)))

(defn run-chain-ctx!
  "Run a bound chain (bind-chain!) for one token: resolve each step's position-dependent scalars
   and work-item count via ctx (a scalar value or :n that is a KEYWORD is looked up in ctx — e.g.
   `\"pos-offset\" :pos` or `:n :nq` → (get ctx …)), prepare each step, record the graph, refresh
   the given :input buffers, replay, and download the :output buffers. Re-callable per token with a
   new ctx; the resident weights + KV (:state, written in place by kv-append) persist. The KV cache
   + attention scratch are sized to MAX positions at bind; cache-len/pos vary per token as scalars."
  ([sess ctx inputs] (run-chain-ctx! sess ctx inputs (:chain-dtype @sess)))
  ([sess ctx inputs dtype]
   (let [steps (:chain-steps @sess) roles (:chain-roles @sess)
         resolve* (fn [v] (if (keyword? v) (get ctx v) v))
         ;; A step is POSITION-DEPENDENT iff a scalar value or its work-item count is a ctx keyword
         ;; (pos/cache-len). Only those need re-preparing per token; the other ~420 keep their
         ;; first-call kernel handles. The first call (no :chain-prepared) prepares ALL to establish
         ;; the static handles. This is what makes per-token cheap: re-prepare drops 453→~30 (the
         ;; 429ms→~28ms host cost); the static handles persist across tokens, only the position
         ;; steps + the graph re-record. (record-graph! must still re-bake the changed handles.)
         prepared? (:chain-prepared @sess)
         pos-dep? (fn [{:keys [scalars n]}] (or (keyword? n) (some keyword? (vals scalars))))]
     (doseq [{:keys [op phase bind scalars n] :as step} steps]
       (when (or (not prepared?) (pos-dep? step))
         (let [ki (first (get-in @sess [:kernels phase]))
               rscalars (into {} (map (fn [[k v]] [k (resolve* v)]) scalars))
               tsc (typed-scalars-for op ki rscalars dtype)]
           (prepare! sess phase bind tsc (long (resolve* n)) {:kernel-phase phase}))))
     (swap! sess assoc :chain-prepared true)
     (record-graph! sess (mapv :phase steps) :chain)
     (doseq [[k arr] inputs] (upload! sess k arr))
     (replay! sess :chain)
     (into {} (for [[k r] roles :when (= r :output)] [k (download sess k)])))))

(defn kernel
  "Get kernel info vector from the session by phase key."
  [sess phase-key]
  (get-in @sess [:kernels phase-key]))

(defn profile-recorded-graph!
  "Replay a graph recorded by record-graph! with `{:profile? true}` and return its backend-neutral
   device-event profile. Unlike replay!, this consumes (and resets) the timestamps instead of
   discarding them. The graph representation remains private to the session layer."
  [sess graph-key]
  (let [device-id (:device-id @sess)
        entry (or (get-in @sess [:graphs graph-key])
                  (throw (ex-info (str "No graph: " graph-key " — call record-graph! first")
                                  {:graph-key graph-key})))
        graph (if (recorded-graph-entry? entry) (:replay-graph entry) entry)]
    (when-not (and (recorded-graph-entry? entry) (:profile? entry))
      (throw (ex-info "recorded graph was not created with {:profile? true}"
                      {:graph-key graph-key})))
    (let [replay-fn (rt-resolve device-id "replay-graph!")
          read-ts-fn (rt-resolve device-id "read-graph-timestamps!")
          t0 (System/nanoTime)
          _ (replay-fn graph)
          t1 (System/nanoTime)
          prof (read-ts-fn graph)]
      {:profile (mapv #(select-keys % [:kernel-name :phase :ms :context-ms]) (:kernels prof))
       :kernel-total-ms (reduce + 0.0 (map :ms (:kernels prof)))
       :device-wall-ms (:wall-ms prof)
       :host-wall-ms (/ (- t1 t0) 1.0e6)})))

(defn measure-graph!
  "Repeatedly measure a PROFILING runtime graph with backend device events.

   This is the backend-neutral autotune ruler. It delegates sampling discipline and the stable
   Measurement value to raster.gpu.measurement, while the selected runtime supplies replay and
   timestamp conversion. It never falls back to host wall time.

   `:before-sample!`, when supplied, runs before EVERY replay (including warmup/probe) and is the
   explicit restore seam for stateful candidates. Other options are forwarded to
   measurement/measure!, including :budget-ms, :warmup-iterations, :flush-fn, :cold-warm,
   :compile-ms, and :hashes."
  [sess graph & {:keys [before-sample!] :as opts}]
  (let [device-id (:device-id @sess)
        replay-fn (rt-resolve device-id "replay-graph!")
        read-ts-fn (rt-resolve device-id "read-graph-timestamps!")
        sample-fn (fn []
                    (when before-sample! (before-sample!))
                    (replay-fn graph)
                    (let [wall-ms (:wall-ms (read-ts-fn graph))]
                      (when-not (and (number? wall-ms)
                                     (Double/isFinite (double wall-ms))
                                     (not (neg? (double wall-ms))))
                        (throw (ex-info "device graph profiler returned no finite wall duration"
                                        {:device-id device-id :wall-ms wall-ms})))
                      (* 1.0e6 (double wall-ms))))
        measurement-opts (-> opts
                             (dissoc :before-sample!)
                             (assoc :timing-source :device-event))]
    (apply measurement/measure! sample-fn (mapcat identity measurement-opts))))

(defn measure-recorded-graph!
  "Repeatedly measure a session graph recorded with `{:profile? true}`. This is the stable-key
   counterpart to measure-graph! and keeps backend graph handles out of compiler/runtime APIs."
  [sess graph-key & {:as opts}]
  (let [entry (or (get-in @sess [:graphs graph-key])
                  (throw (ex-info (str "No graph: " graph-key " — call record-graph! first")
                                  {:graph-key graph-key})))
        graph (if (recorded-graph-entry? entry) (:replay-graph entry) entry)]
    (when-not (and (recorded-graph-entry? entry) (:profile? entry))
      (throw (ex-info "recorded graph was not created with {:profile? true}"
                      {:graph-key graph-key})))
    (apply measure-graph! sess graph (mapcat identity opts))))

(defn measure-bound-kernel-graph!
  "Device-event measurement of a bound KernelGraphHandle.

   The graph must have been recorded with `:profile? true` (currently supplied by
   bind-kernel-call!). This keeps runtime graph representations private while exposing the same
   bounded Measurement contract to generic offline tuning code."
  [sess handle & {:as opts}]
  (let [{:keys [runtime-graph profile?]} (resolve-kernel-graph-entry sess handle)]
    (when-not profile?
      (throw (ex-info "bound kernel graph was not recorded with :profile? true"
                      {:handle handle})))
    (apply measure-graph! sess runtime-graph (mapcat identity opts))))

(defn sync-to-arrays!
  "Download GPU buffers back into JVM arrays.

   sess: session atom
   mappings: seq of [jvm-array buffer-key] pairs

   Example:
     (sync-to-arrays! sess
       [[(.effort agents) :effort]
        [(.income agents) :income]])"
  [sess mappings]
  (let [{:keys [device-id buffers]} @sess
        download-fn (rt-resolve device-id "buffer->array")
        bufs buffers]
    (doseq [[dst-arr buf-key] mappings]
      (let [src (download-fn (get bufs buf-key))
            n   (java.lang.reflect.Array/getLength dst-arr)]
        (System/arraycopy src 0 dst-arr 0 n)))))
