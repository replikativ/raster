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
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as kgcall]
            [raster.compiler.pipeline :as pl]
            [raster.core :as rcore]
            [raster.gpu.measurement :as measurement]))

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

(defn- destroy-superseded!
  "Destroy a prepared binding / graph being overwritten under the same key, so re-prepare! /
  re-record! don't leak the previous dedicated kernel handle (or queue+list). Nil-safe."
  [device-id destroyer-name old]
  (when old
    (when-let [d (rt-resolve-soft device-id destroyer-name)]
      (try (d old) (catch Exception _)))))

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
   Returns vector of kernel-info maps."
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
        par-opencl opencl-pass/opencl-pass
        register!  (rt-resolve device-id "register-kernel!")
        result (par-opencl form
                           :device-id device-id
                           :dtype dtype
                           :min-elements min-elements)]
    (doseq [k (:kernels result)]
      (register! (:kernel-name k) k))
    (:kernels result)))

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
           :buffers   {}       ;; {buf-key → DeviceBuffer}
           :allocations {}     ;; {buf-key → backend-neutral BufferAllocation}
           :programs  {}       ;; {program-key → bound resident program (see bind-program!)}
           :kernel-graphs {}    ;; {graph-key → bound emitted KernelGraph}
           :events {}           ;; {event-id → session-owned asynchronous completion}
           :closed?   false})))

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
      (let [{:keys [device-id arena-id buffers allocations prepared graphs programs kernel-graphs]} @sess]
        ;; backend-specific: the bound-dispatch + command-graph path is ze-only, so resolve the
        ;; destroyers nil-safely rather than via rt-resolve (which throws on backends lacking them).
        (let [ns-sym (case (backend-type device-id) :ze 'raster.gpu.ze-runtime :ocl 'raster.gpu.ocl-runtime)
              destroy-prepared! (requiring-resolve (symbol (str ns-sym) "destroy-prepared!"))
              destroy-graph!    (requiring-resolve (symbol (str ns-sym) "destroy-graph!"))]
          (when destroy-graph!    (doseq [[_ g] graphs]   (try (destroy-graph! g)    (catch Exception _))))
          (when destroy-prepared! (doseq [[_ p] prepared] (try (destroy-prepared! p) (catch Exception _))))
          ;; bound resident programs: each holds a recorded graph (queue+list) AND the per-step
          ;; bound kernel handles (create-kernel-fresh per bind, NOT in the registry) — both must
          ;; be destroyed here or every session leaks them (the SIGABRT class of driver leak).
          (doseq [[_ prog] programs]
            (when destroy-graph! (try (destroy-graph! (:graph prog)) (catch Exception _)))
            (when (and destroy-graph! (:prologue-graph prog))
              (try (destroy-graph! (:prologue-graph prog)) (catch Exception _)))
            (when destroy-prepared!
              (doseq [b (:bounds prog)] (try (destroy-prepared! b) (catch Exception _))))
            ;; f16 GEMM-conversion scratch is allocated outside :buffers — free it here.
            (when (seq (:scratch-buffers prog))
              (let [free! (rt-resolve device-id "free-buffer!")]
                (doseq [b (:scratch-buffers prog)] (try (free! b) (catch Exception _)))))))
        (doseq [[_ graph-entry] kernel-graphs]
          (destroy-kernel-graph-entry! device-id graph-entry))
        (free-session-buffers! buffers allocations device-id)
        (let [close-arena! (rt-resolve device-id "close-kernel-arena!")]
          (close-arena! arena-id))
        (swap! sess assoc :closed? true :buffers {} :allocations {} :kernels {} :prepared {} :graphs {}
               :programs {} :kernel-graphs {} :events {})))))

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
         kernels (or (get-in @sess [:kernel-cache cache-key])
                     (let [ks (compile-deftm-internal! v device-id opts)]
                       (swap! sess assoc-in [:kernel-cache cache-key] ks)
                       ks))]
     (swap! sess assoc-in [:kernels phase-key] kernels)
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
   buffer-specs: {key → [dtype n source-array-or-nil]}

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
                             [key (allocation-contract device-id session-id key buffer :owned {})]))
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
     (destroy-superseded! device-id "destroy-prepared!" (get-in @sess [:prepared phase-key]))
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
        device-id (:device-id @sess)
        launch-fn (rt-resolve device-id "launch-registered-bound!")]
    (launch-fn prepared)))

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
  ([sess phase-keys] (record-graph! sess phase-keys :graph))
  ([sess phase-keys graph-key]
   (let [device-id (:device-id @sess)
         record-fn (rt-resolve device-id "record-graph!")
         prepareds (mapv (fn [pk]
                           (or (get-in @sess [:prepared pk])
                               (throw (ex-info (str "Phase not prepared: " pk " — call prepare! first")
                                               {:prepared (keys (:prepared @sess))}))))
                         phase-keys)
         graph (record-fn prepareds)]
     (destroy-superseded! device-id "destroy-graph!" (get-in @sess [:graphs graph-key]))
     (swap! sess assoc-in [:graphs graph-key] graph)
     graph)))

(defn replay!
  "Execute a recorded command graph once (synchronous). Reads current buffer contents."
  ([sess] (replay! sess :graph))
  ([sess graph-key]
   (let [device-id (:device-id @sess)
         graph (or (get-in @sess [:graphs graph-key])
                   (throw (ex-info (str "No graph: " graph-key " — call record-graph! first") {})))]
     ((rt-resolve device-id "replay-graph!") graph))))

(defn invoke-scan!
  "Invoke a compiled Blelloch exclusive-scan kernel pair from the session.

   sess: session atom
   phase-key: keyword identifying the scan kernel pair
   input-keys: vector of buffer keys for inputs
   output-key: buffer key for output
   n: number of elements"
  [sess phase-key input-keys output-key n]
  (let [{:keys [kernels buffers]} @sess
        kernel-vec (get kernels phase-key)
        device-id (:device-id @sess)
        invoke! (rt-resolve device-id "invoke-registered-scan-exclusive-kernel")]
    (invoke! (:kernel-name (first kernel-vec))
             (:kernel-name (second kernel-vec))
             (mapv #(get buffers %) input-keys)
             (get buffers output-key)
             n)))

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

(defn- transfer-ranges!
  "The batched core. VALIDATES EVERY entry (`plan-range`) before EXECUTING ANY. A batch is how a
   whole KV cache moves — 36 per-layer buffers for gemma-270m — and a batched API newly makes a
   partial state possible: a bad spec in the 30th entry leaving 29 layers written. All-or-nothing
   on validation removes that class; nothing is copied until every range has been proved in
   bounds. (A failure DURING execution — a device fault — is still partial; that is a different
   class and is not promised here.)"
  [sess entries direction]
  (let [device-id (:device-id @sess)
        plan (rt-resolve device-id "plan-range")
        exec (rt-resolve device-id "execute-range!")
        ;; phase 1: resolve + validate everything, collecting plans in order
        plans (mapv (fn [[key-or-view host spec]]
                      (let [{:keys [buffer view]} (resolve-resident-binding sess key-or-view)
                            spec (checked-view-range-spec buffer view spec direction)]
                        [buffer (plan buffer host spec direction) host]))
                    entries)]
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

(defn- await-event-under-lock!
  [sess event]
  (let [{:keys [device-id closed?]} @sess
        {:keys [status backend-event] :as entry} (resolve-event-entry sess event)]
    (when closed?
      (throw (ex-info "cannot use an event from a closed GPU session" {:event event})))
    (if (= :complete status)
      entry
      (do
        ;; A successful status query is not necessarily a host memory-synchronization point
        ;; (notably in OpenCL). Await always calls the backend wait before releasing the token.
        ((rt-resolve device-id "await-event!") backend-event)
        ((rt-resolve device-id "release-event!") backend-event)
        (let [completed (assoc entry :status :complete :backend-event nil)]
          (swap! sess assoc-in [:events (:id event)] completed)
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
   compute queue; submit-kernel-graph! exposes asynchronous completion without native handles."
  [sess graph-key graph buffer-keys scalar-values]
  (let [{:keys [device-id closed?]} @sess
        graph (kgraph/validate! graph)
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
          (vreset! runtime-graph (record! @prepareds {:barriers? true}))
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
                       :outputs (select-keys all-buffers (map :id (:outputs graph)))}
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
          (throw e))))))

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
                        (when (= :output (:kind slot))
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

(defn bind-step!
  "Bind ONE compiled kernel step (a descriptor :steps entry) into the session under its :phase.
   Resolves the kernel's array args to resident buffers via `sym->key` (arg-name-symbol → session
   buffer-key keyword) and its scalars via the step's value-fns over `args`. Handles all three
   conventions uniformly — the per-step core shared by bind-program! (a single program, sym→key =
   name) and a multi-instance decode binder (one layer program bound once per layer, sym→key maps
   weights/KV per layer and scratch shared, like the decoder's pbk). Does NOT record a graph; the
   caller collects :phase keys and records once.

     :map/:reduce/:map-void/:contract
                Artifact ABI values and realized geometry bind through one KernelCall contract.
                SegRed explicitly schedules one resident workgroup; maps realize their grid.
                Logical SoA arguments expand through the backend representation adapter before
                the physical KernelCall is constructed."
  [sess step args sym->key]
  (let [device-id (:device-id @sess)
        {:keys [kernel-name phase convention artifact argument-specs]} step]
    (when-not (#{:map-void :map :reduce :contract} convention)
      (throw (ex-info (str "bind-step! cannot bind a " convention " step (" kernel-name
                           ") — only artifact-backed kernel steps are supported on the resident path")
                      {:convention convention :kernel kernel-name})))
    (let [bufs (:buffers @sess)
          resolve-buf (fn [a] (or (get bufs (sym->key a))
                                  (throw (ex-info (str "No buffer for kernel arg: " a " → " (sym->key a))
                                                  {:kernel kernel-name :available (keys bufs)}))))
          bind-call-fn (rt-resolve device-id "bind-kernel-call")]
      (case convention
        (:map :reduce :map-void :contract)
        (let [logical-or-physical-args
              (mapv (fn [{:keys [kind sym type value-fn]}]
                      (if (= :scalar kind)
                        {:type type :value (value-fn args)}
                        (resolve-buf sym)))
                    argument-specs)
              ordered-args
              (if (:logical-bindings? step)
                (kcall/expand-logical-arguments
                 artifact logical-or-physical-args
                 (rt-resolve device-id "expand-pointer-binding"))
                logical-or-physical-args)
              selected-artifact (if-let [dispatch (:dispatch step)]
                                  (kdispatch/select-artifact dispatch ordered-args)
                                  artifact)
              call (kcall/make selected-artifact ordered-args
                               (if (= :reduce convention) {:group-count [1]} {}))
              prepared (bind-call-fn call)]
          (swap! sess assoc-in [:prepared phase] prepared))
        nil))
    sess))

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

;; ================================================================
;; Resident GPU programs (whole-offload: pipeline descriptor → bound command graph)
;; ================================================================
;;
;; RE-TARGETED onto fusion's runtime graph primitives: params-on-main's per-step
;; prepare!/bind-kernel!(4-arg)/launch-registered-bound! + session-level
;; record-graph!/replay! are REPLACED by ze-runtime/bind-registered-map-void-kernel
;; (returns a {:kernel :gc-seg …} bound map over GPU-RESIDENT buffers) collected into
;; a vector for ze-runtime/record-graph! (barrier-separated), replayed by replay-graph!.

;; ── split-k schedule knobs (see the `splitk` fn in bind-program!) ──────────────
;; The XMX GEMM launches ceil(n/128) x ceil(m/128) workgroups of 256 items (= 16
;; hw threads). This iGPU (64 EU x 8 threads) holds 32 such workgroups, so any GEMM
;; below that count is OCCUPANCY-bound, not memory- or compute-bound. Splitting k
;; buys workgroups at constant DRAM traffic; the knobs bound how far.

(def ^:dynamic *gemm-splitk-fill-wgs*
  "Workgroups that saturate the device. nil = DERIVE it from the target's HardwareDescriptor
   (machine-lanes / 256 — Arc 140V: 8192/256 = 32); bind to a number to override. A GEMM at
   or above this count is NOT split.

   This used to be the literal 32, which is this laptop's iGPU and no other machine's." nil)

(def ^:dynamic *gemm-splitk-target-wgs*
  "Workgroup count a split GEMM aims for — a few waves over the fill count, so memory
   latency has something to hide behind. nil = 4x the fill count." nil)

(def ^:dynamic *gemm-splitk-min-chunk*
  "Smallest k-chunk worth giving a workgroup: below this the per-chunk fixed cost (A
   prefetch, storing a full partial C tile) outweighs the reduction." 1024)

(def ^:dynamic *gemm-splitk-max-splits*
  "Hard cap on k-chunks — the partials buffer is splits·m·n f32." 64)

(defn gemm-fill-workgroups
  "Workgroups that fill `device-id` at the XMX GEMM's 256-item workgroup — the bound that
   decides whether a GEMM is occupancy-starved. From the target's HardwareDescriptor unless
   *gemm-splitk-fill-wgs* overrides."
  [device-id]
  (long (or *gemm-splitk-fill-wgs* (hw/fill-workgroups (hw/descriptor-for device-id) 256))))

(defn gemm-schedule
  "THE GEMM schedule decision, as a pure function of the shape and the machine width —
   never of the model. Returns [splits k-chunk]; splits = 1 means the plain XMX GEMM.

   The XMX GEMM tiles C into 128x128 blocks, so its grid is ceil(n/128) x ceil(m/128)
   workgroups. When that is fewer than `fill-wgs`, the device runs partly EMPTY and no
   inner-loop tuning can help — the only way to buy workgroups is to split the k-reduction
   across a third grid dimension and combine the partials afterwards (same operands, same
   DRAM traffic, same result up to f32 summation order).

   Callers: bind-program! (the binder) and gemm_splitk_test (the executable spec). ONE
   copy — the policy is not re-implemented anywhere."
  ([m n k fill-wgs] (gemm-schedule m n k fill-wgs (or *gemm-splitk-target-wgs* (* 4 (long fill-wgs)))))
  ([m n k fill-wgs target-wgs]
   (let [{:keys [block-m block-n block-k]}
         ((requiring-resolve 'raster.compiler.core.hardware/gemm-tile-for)
          (try ((requiring-resolve 'raster.compiler.core.hardware/descriptor-for)
                (:device-id @(requiring-resolve 'raster.gpu.ze-runtime/state)))
               (catch Throwable _ nil)))
         ;; the split-k policy's occupancy estimate and K-chunk quantum are the SAME tile the
         ;; kernel is emitted with — they were independent `/128.0` and `*32` literals, so a tile
         ;; change silently decoupled the policy from the kernel it schedules
         base (* (Math/ceil (/ (double n) (double block-n)))
                 (Math/ceil (/ (double m) (double block-m))))]
     (if (>= base (double fill-wgs))
       [1 k]                              ;; already fills the machine
       (let [want (long (Math/ceil (/ (double target-wgs) base)))
             ;; each chunk must be a multiple of the K-unroll (block-k) and at least
             ;; *gemm-splitk-min-chunk* long, or the per-chunk fixed cost (A prefetch + the
             ;; store of a full partial C tile) swamps the reduction it does.
             cap (quot (long k) (long *gemm-splitk-min-chunk*))
             s (min want cap (long *gemm-splitk-max-splits*))]
         (if (< s 2)
           [1 k]
           (let [ku (long block-k)
                 kc (* ku (quot (+ (quot (long k) s) (dec ku)) ku))]
             [(quot (+ (long k) kc -1) kc) kc])))))))

(defn bind-program!
  "Bind a resident GPU program (a descriptor from pipeline/compile-gpu-program) to this session:
   allocate resident buffers for the array params + intermediate scratch, bind each kernel
   step against them (a fresh kernel handle per step, group count pre-set into its :gc-seg), and
   record the kernel sequence as ONE replayable command graph. After binding, run-program!
   replays the whole sequence with NO re-binding. The bound machinery is convention-agnostic, so
   map! and map-void! kernels bind identically (a map! kernel's output is just another resident
   buffer in its :array-params).

   args = values in the descriptor's :all-params order (JVM arrays for array params, numbers for
   scalars). Buffer keys are the param/intermediate sym names as keywords.

   roles = optional {param-sym → :constant|:state|:input|:output} override of the descriptor's
   derived defaults (read-only→:input, written→:output). Declare cross-call persistence the
   program can't derive: :constant = weights (uploaded once here, never re-uploaded by
   run-program!); :state = persistent device state e.g. a KV cache or on-device-updated adapters
   (never downloaded). All buffer CONTENTS are uploaded once here at bind; run-program! then
   moves only :input (up) and :output (down).

   Returns a program HANDLE {::program-key k :descriptor d} — pass it to run-program!. A session
   can hold MULTIPLE programs (bound under distinct :key opts) over SHARED resident buffers.

   opts (5-arity):
     :key            program key (keyword, default :program). Binding an already-bound key
                     throws. With a non-default key the program's scratch (intermediate alloc)
                     buffers are namespaced `<key>.<sym>` so scratch never collides across
                     programs; PARAM buffer keys stay the plain param names (the sharing seam).
     :reuse-buffers  the explicit buffer-sharing rule. When true, an array param whose resolved
                     buffer key ALREADY EXISTS in the session reuses that DeviceBuffer — the
                     bound kernels of this program read/write the SAME device memory as the
                     program that allocated it, and the param's host array is NOT uploaded
                     (device contents are authoritative; that is the point: a VJP program's
                     :state adapters are seen by a forward program with no host round-trip,
                     frozen :constant weights upload once). Element-count or dtype mismatch
                     FAILS LOUD (ex-info), never silently rebinds. When false/absent, ANY
                     buffer-key collision throws — name collision alone never aliases memory.
     :rename         {param-sym → buffer-key-keyword} per-param key override: share two
                     differently-named params (rename both onto one key + :reuse-buffers), or
                     keep two same-named params separate (rename one away from the collision).
                     run-program!/upload!/download address the renamed buffer by the new key.
     :profile?       record the graph in PROFILING mode: a device-timestamp event per launch.
                     Level Zero and OpenCL expose the same profile-program! contract. Opt-in:
                     without it the recorded graph is exactly the fast path (no profiling queue,
                     events, or overhead); run-program! on a profiled program still works.

   :gemm steps bind per the resolved S6 schedule's precision — [:schedule :precision] (set at
   compile time by compile-gpu-program, default :f16-xmx). :gemm-precision on the descriptor is a
   back-compat fallback for a hand-built descriptor with no :schedule. A bind-time caller overrides
   the plain-data descriptor with (assoc-in descriptor [:schedule :precision] …):
     :f16-xmx    — convert A/B f32→f16 and run the XMX gemm (f32 accumulate/output): the
                   mixed-precision (AMP) policy — f16 inputs, f32 math. Valid for BACKWARD
                   programs too (measured on the real gemma-3-270m layer VJP: adapter grads
                   rel-err ~9e-4 / cosine 1.000 vs the :f32-scalar grads; kernel time
                   65.5 → 38.8 ms, its GEMM part 36.8 → 12.0 ms).
     :f32-scalar — bind the plain scalar f32 GEMM for ALL :gemm steps: reads the f32
                   residents directly, no convert/transpose expansion kernels. Exact f32
                   grads (~1e-6-level parity) — the exactness escape hatch.
   The XMX hardware pitch gate (n<8 or k<8 → scalar) applies regardless of policy. Loss
   scaling (for cotangents small enough to hit the f16 min-normal 6.1e-5) is a CALLER
   concern — the VJP is linear in the seed, so scale the seed by S and use lr/S.
   Under :f16-xmx the f32→f16 conversion (and, for :nt/:tn, the transpose of that f16 copy)
   of any operand whose role is :constant is recorded into a PROLOGUE graph replayed once
   here at bind — frozen weights are converted once, not once per replay. This assumes the
   :constant contract literally: a buffer bound :constant must not be mutated afterwards."
  ([sess descriptor args] (bind-program! sess descriptor args {} {}))
  ([sess descriptor args roles] (bind-program! sess descriptor args roles {}))
  ([sess descriptor args roles {:keys [key reuse-buffers rename profile?]
                                :or {key :program rename {}}}]
   (let [device-id (:device-id @sess)
         pkey key
         _ (when (contains? (:programs @sess) pkey)
             (throw (ex-info (str "bind-program!: program key " pkey " already bound in this session"
                                  " — bind each program under a distinct :key")
                             {:key pkey :bound (keys (:programs @sess))})))
         {:keys [dtype all-params array-params allocs steps]} descriptor
         ;; precision reads from the resolved S6 schedule (source of truth); :gemm-precision is
         ;; back-compat sugar for a descriptor built before the schedule field, or one hand-assoc'd.
         gemm-precision (or (get-in descriptor [:schedule :precision])
                            (:gemm-precision descriptor) :f16-xmx)
         reduction-strategy
         (get-in descriptor [:schedule :segmented-weighted-reduction :strategy] :auto)
         effective-roles (merge (:array-roles descriptor) roles)
         argmap (zipmap all-params args)
         dt (if (= dtype :double) :double :float)
         nel (fn [arr] (java.lang.reflect.Array/getLength arr))
         ;; per-array element dtype from the ACTUAL JVM array — a program can mix dtypes (quant
         ;; kernels carry byte weights + float scales + int bsums + float output), so a single
         ;; program dtype mis-allocates (CCE [B→[F). The runtime array type is authoritative.
         arr-dtype (fn [arr]
                     (condp instance? arr
                       (Class/forName "[B") :byte
                       (Class/forName "[S") :half
                       (Class/forName "[I") :int
                       (Class/forName "[J") :long
                       (Class/forName "[F") :float
                       (Class/forName "[D") :double
                       dt))
         param->key (into {} (map (fn [p] [p (get rename p (keyword (name p)))])) array-params)
         alloc->key (into {} (map (fn [{:keys [sym]}]
                                    [sym (if (= pkey :program)
                                           (keyword (name sym))
                                           (keyword (str (name pkey) "." (name sym))))]))
                          allocs)
         existing-bufs (:buffers @sess)
         ;; The sharing rule, enforced per resolved buffer key:
         ;;   free key            → allocate (+ upload the param's host array)
         ;;   collision, no reuse → THROW (a silent rebind would orphan the earlier program's
         ;;                         graph pointers and leak the buffer)
         ;;   collision + reuse   → same n-elements AND dtype → reuse the DeviceBuffer, skip
         ;;                         the upload (device contents authoritative); else THROW.
         reuse-or-spec (fn [k want-dtype want-n spec ctx]
                         (if-let [buf (get existing-bufs k)]
                           (if-not reuse-buffers
                             (throw (ex-info (str "bind-program!: buffer key " k " already exists in "
                                                  "this session (" ctx "). Sharing must be explicit: "
                                                  "pass {:reuse-buffers true} to share it, or :rename "
                                                  "to give this program's buffer a distinct key.")
                                             {:key k :ctx ctx :program pkey}))
                             (if (and (= (long want-n) (long (:n-elements buf)))
                                      (= want-dtype (:dtype buf)))
                               nil ;; reuse: no alloc, no upload
                               (throw (ex-info (str "bind-program!: buffer key " k " exists with a "
                                                    "DIFFERENT shape — refusing to alias (" ctx ")")
                                               {:key k :ctx ctx :program pkey
                                                :existing {:n (:n-elements buf) :dtype (:dtype buf)}
                                                :wanted {:n want-n :dtype want-dtype}}))))
                           [k spec]))
         param-specs (into {} (keep (fn [p]
                                      (let [arr (get argmap p)
                                            adt (arr-dtype arr)]
                                        (reuse-or-spec (param->key p) adt (nel arr)
                                                       [adt (nel arr) arr]
                                                       (str "param " p)))))
                           array-params)
         alloc-specs (into {} (keep (fn [{:keys [sym size-fn dtype]}]
                                      (let [n (long (size-fn args))
                                            alloc-dtype (or dtype dt)]
                                        (reuse-or-spec (alloc->key sym) alloc-dtype n
                                                       [alloc-dtype n nil]
                                                       (str "scratch " sym)))))
                           allocs)
         info-fn   (rt-resolve device-id "kernel-registry-entry")
         specialized-bind-fn (rt-resolve device-id "bind-registered-map-void-kernel")
         bind-call-fn (rt-resolve device-id "bind-kernel-call")
         gemm-fn   (rt-resolve device-id "bind-registered-gemm!")
         conv-fn   (rt-resolve device-id "bind-registered-convert!")
         trans-fn  (rt-resolve device-id "bind-registered-transpose!")
         mkbuf-fn  (rt-resolve device-id "make-buffer")
         record-fn (rt-resolve device-id "record-graph!")
         ;; ── the SCHEDULE seam ────────────────────────────────────────────────
         ;; Every launch geometry below is a function of (shape, MACHINE WIDTH) and
         ;; nothing else — never of the model. The machine width comes from the target's
         ;; HardwareDescriptor, so the same program schedules itself differently on a
         ;; different GPU instead of inheriting this laptop's iGPU as a literal.
         desc      (hw/descriptor-for device-id)
         fill-wgs  (gemm-fill-workgroups device-id)
         ;; elementwise/convert vector width: widest that still fills the machine.
         vec-width (fn [n] (hw/stream-vector-width desc n))
         ;; SPLIT-K: see `gemm-schedule` — a GEMM whose grid is below the machine's fill
         ;; count is occupancy-bound, and splitting k buys workgroups at constant DRAM
         ;; traffic. See ze-runtime/bind-registered-gemm-splitk!.
         splitk (fn [m n k] (gemm-schedule m n k fill-wgs))
         alloc-size-of (fn [sym-kw]
                         (some (fn [{:keys [sym size-fn]}]
                                 (when (= (keyword (name sym)) sym-kw) (long (size-fn args))))
                               allocs))
         gemm-scratch (atom [])]
     (alloc! sess (merge param-specs alloc-specs))
     (let [buffers (:buffers @sess)
           key-of (fn [sym] (or (get param->key sym) (get alloc->key sym) (keyword (name sym))))
           buf-of (fn [sym ctx]
                    (or (get buffers (key-of sym))
                        (throw (ex-info (str "bind-program!: no resident buffer for step array " sym)
                                        {:sym sym :key (key-of sym) :ctx ctx :have (keys buffers)}))))
           ;; A GEMM operand that is a :constant param (frozen weights) never changes on device,
           ;; so its f32→f16 conversion — and, for the transposed variants, the transpose of that
           ;; f16 copy — is the SAME work every replay. Those kernels are recorded into a separate
           ;; PROLOGUE graph replayed exactly once at bind (see const-prologue? below), not into
           ;; the per-step graph. The f16 scratch buffers are allocated either way, so this costs
           ;; no VRAM; it just stops re-converting the weights every step. Measured on the gemma
           ;; layer VJP (5.6M frozen weight elements vs ~0.2M activation elements): f32_to_f16
           ;; 7.2 → 0.6 ms, transpose_half 3.2 → 0.2 ms per replay.
           const-operand? (fn [sym] (= :constant (get effective-roles sym)))
           step->bounds
           (fn [{:keys [kernel-name arrays n-fn scalar-specs convention accumulator
                        artifact argument-specs] :as step}]
             (case convention
               ;; GEMM (Option B): [convert A f32→f16][convert B f32→f16][fp16 XMX gemm → f32 C].
               ;; A/B are converted into per-GEMM f16 scratch (kept alive on the session); the
               ;; conversions of :constant operands are hoisted to the bind-time prologue graph.
               :gemm
               (let [m (long ((:m-fn step) args)) n (long ((:n-fn step) args)) k (long ((:k-fn step) args))
                     abuf (buf-of (:A step) :gemm-A) bbuf (buf-of (:B step) :gemm-B)
                     cbuf (buf-of (:C step) :gemm-C)
                     a-const? (const-operand? (:A step))
                     b-const? (const-operand? (:B step))]
                 (if (or (= :f32-scalar gemm-precision) (< n 8) (< k 8))
                   ;; Scalar f32 path, taken when (a) the :gemm-precision policy is
                   ;; :f32-scalar (exact-grad training — see the docstring), or (b) the
                   ;; XMX hardware pitch gate fires regardless of policy: the XMX
                   ;; kernel's 2D-block reads need a >=16-byte pitch — the B-operand
                   ;; VNNI read's pitch is N*2 bytes (fp16) and the A-operand row read's
                   ;; pitch is K*2 bytes, so N<8 OR K<8 violates it and yields garbage
                   ;; (relerr ~1; K<8 shows as scrambled + zeroed C rows).
                   ;; Binds the plain scalar f32 GEMM: it reads the f32 residents
                   ;; directly, so NO convert/transpose expansion kernels at all.
                   ;; Resolved lazily — Level-Zero-only for now (like the scatter binder).
                   (let [scalar-gemm-fn (rt-resolve device-id "bind-registered-gemm-scalar!")]
                     [{:bound (scalar-gemm-fn abuf bbuf cbuf m n k (:variant step))
                       :kernel-name (str "gemm_scalar_" (name (:variant step)))
                       :phase (:phase step)}])
                   (let [a16 (mkbuf-fn (* m k) :half)
                     ;; The GEMM proper: one bound kernel, or — when the (m,n) tiling can't
                     ;; fill the machine — a split-k PAIR (partial GEMM over a 3D grid +
                     ;; a partials combine), which is a pure SCHEDULE change: same operands,
                     ;; same DRAM traffic, same result up to f32 summation order.
                         mk-gemm
                         (fn [abuf* bbuf*]
                           (let [[splits kc] (splitk m n k)]
                             (if (= 1 (long splits))
                               [["gemm_nonsquare_float" (gemm-fn abuf* bbuf* cbuf m n k :float) false]]
                               (let [sk-fn (rt-resolve device-id "bind-registered-gemm-splitk!")
                                     red-fn (rt-resolve device-id "bind-registered-splitk-reduce!")
                                     parts (mkbuf-fn (* (long splits) m n) :float)]
                                 (swap! gemm-scratch conj parts)
                                 [["gemm_nonsquare_splitk"
                                   (sk-fn abuf* bbuf* parts m n k kc splits) false]
                                  ["gemm_splitk_reduce"
                                   (red-fn parts cbuf (* m n) splits) false]]))))
                     ;; each expansion kernel (convert/transpose/gemm) pre-bakes its FULL gc-seg —
                     ;; wrap as {:bound bnd} with NO :group-count so record-graph! keeps the grid.
                     ;; Each entry is [kernel-name bnd const?]: kernel-name so profiling can
                     ;; attribute device time, const? = "depends only on :constant operands" ⇒
                     ;; recorded into the bind-time prologue graph instead of the replay graph.
                         raw
                         (case (:variant step)
                       ;; C = A[m,k] · B[k,n]
                           :nn
                           (let [b16 (mkbuf-fn (* k n) :half)]
                             (swap! gemm-scratch conj a16 b16)
                             (into [["f32_to_f16" (conv-fn abuf a16 (* m k) (vec-width (* m k))) a-const?]
                                    ["f32_to_f16" (conv-fn bbuf b16 (* k n) (vec-width (* k n))) b-const?]]
                                   (mk-gemm a16 b16)))
                       ;; C = A[m,k] · B[n,k]ᵀ — convert B then transpose [n,k]→[k,n], then :nn gemm.
                       ;; (HF linear weights [out,in] and attention Q·Kᵀ are :nt.)
                           :nt
                           (let [b16 (mkbuf-fn (* n k) :half) bt16 (mkbuf-fn (* k n) :half)]
                             (swap! gemm-scratch conj a16 b16 bt16)
                             (into [["f32_to_f16" (conv-fn abuf a16 (* m k) (vec-width (* m k))) a-const?]
                                    ["f32_to_f16" (conv-fn bbuf b16 (* n k) (vec-width (* n k))) b-const?]
                                    ["transpose_half" (trans-fn b16 bt16 n k :half) b-const?]]
                                   (mk-gemm a16 bt16)))
                       ;; C[m,n] = Aᵀ·B — A stored [k,m], B [k,n]. Convert A then transpose
                       ;; [k,m]→[m,k], convert B ([k,n] already the :nn B layout), then :nn gemm.
                       ;; (linear-dW = dgemm-tn! : the weight-gradient backward matmul.)
                           :tn
                           (let [at16 (mkbuf-fn (* m k) :half) b16 (mkbuf-fn (* k n) :half)]
                             (swap! gemm-scratch conj a16 at16 b16)
                             (into [["f32_to_f16" (conv-fn abuf a16 (* k m) (vec-width (* k m))) a-const?]
                                    ["transpose_half" (trans-fn a16 at16 k m :half) a-const?]
                                    ["f32_to_f16" (conv-fn bbuf b16 (* k n) (vec-width (* k n))) b-const?]]
                                   (mk-gemm at16 b16)))
                           (throw (ex-info (str "GEMM variant not yet wired on resident path: " (:variant step)
                                                " (only :nn / :nt / :tn)") {:variant (:variant step)})))]
                     (mapv (fn [[nm b c]] {:bound b :kernel-name nm :phase (:phase step)
                                           :const-prologue? (boolean c)})
                           raw))))
               ;; Artifact-backed kernels share the complete executable value: emitted artifact,
               ;; ordered ABI values and realized launch. Reduction's single-group resident
               ;; schedule is explicit here rather than hidden in a special binder.
               (:map :reduce :map-void :contract)
               (let [logical-or-physical-args
                     (mapv (fn [{:keys [kind sym type value-fn]}]
                             (if (= :scalar kind)
                               {:type type :value (value-fn args)}
                               (buf-of sym kernel-name)))
                           argument-specs)
                     ordered-args
                     (if (:logical-bindings? step)
                       (kcall/expand-logical-arguments
                        artifact logical-or-physical-args
                        (rt-resolve device-id "expand-pointer-binding"))
                       logical-or-physical-args)
                     selected-artifact (if-let [dispatch (:dispatch step)]
                                         (kdispatch/select-artifact
                                          dispatch ordered-args reduction-strategy)
                                         artifact)
                     call (kcall/make selected-artifact ordered-args
                                      (if (= :reduce convention) {:group-count [1]} {}))]
                 [(assoc (bind-call-fn call) :phase (:phase step))])
               ;; scatter-add: out[index[e]*stride+d] += src[e*stride+d]. Expands to TWO bound
               ;; kernels — a zero-fill of the accumulator, then the atomic-add scatter — so the
               ;; recorded graph re-zeroes `out` each replay (zeros-like semantics) and fans
               ;; overlapping destination indices in safely. arrays = [out src index]; the extra
               ;; scalar (if present) is `stride`.
               :scatter
               (do (when (not= :ze (backend-type device-id))
                     (throw (ex-info "resident scatter steps are Level-Zero-only (no OpenCL implementation yet)"
                                     {:backend (backend-type device-id)
                                      :device-id device-id
                                      :step kernel-name})))
                   (or (info-fn kernel-name)
                       (throw (ex-info (str "Program kernel not registered: " kernel-name) {:kernel kernel-name})))
                   (let [[out-sym src-sym idx-sym] arrays
                         out-buf (buf-of out-sym :scatter-out)
                         src-buf (buf-of src-sym :scatter-src)
                         idx-buf (buf-of idx-sym :scatter-idx)
                         n       (long (n-fn args))
                         ;; A scatter step has AT MOST one scalar (the optional stride). Taking
                         ;; `(first scalar-specs)` would silently drop any further scalars — so
                         ;; assert the shape instead of miscompiling a multi-scalar scatter.
                         _       (when (> (count scalar-specs) 1)
                                   (throw (ex-info (str "resident scatter step " kernel-name
                                                        " has " (count scalar-specs)
                                                        " scalars — only a single stride is modeled")
                                                   {:kernel kernel-name :scalar-specs scalar-specs})))
                         stride  (when (seq scalar-specs) (long ((:value-fn (first scalar-specs)) args)))
                         out-size (or (alloc-size-of accumulator)
                                      ;; fallback: accumulator is a param, use its length
                                      (nel (get argmap (symbol (name accumulator)))))
                         ;; Resolved lazily (only a program with a scatter step needs them),
                         ;; so a non-scatter program on a runtime lacking these fns (OpenCL) still
                         ;; binds — resident scatter is Level-Zero-only for now.
                         zerofill-fn (rt-resolve device-id "ensure-zero-fill-kernel!")
                         scatter-fn (rt-resolve device-id "bind-registered-scatter-kernel!")
                         zk (zerofill-fn (if (= dtype :double) :double :float))]
                     [(assoc (specialized-bind-fn zk [out-buf] [] (long out-size))
                             :phase (:phase step))
                      (assoc (scatter-fn kernel-name [out-buf src-buf idx-buf] n stride)
                             :phase (:phase step))]))
               (throw (ex-info (str "bind-program! cannot bind a " convention " step — only "
                                    ":map / :map-void / :contract / :reduce / :gemm / :scatter "
                                    "are wired on the resident path")
                               {:convention convention :kernel kernel-name}))))
           bounds (vec (mapcat step->bounds steps))
           ;; CONSTANT PROLOGUE: the f16 conversions/transposes of :constant GEMM operands run
           ;; ONCE, in their own recorded graph replayed here at bind — the per-step graph holds
           ;; only the kernels whose inputs can actually change between replays. Order within the
           ;; prologue is preserved (a :nt/:tn transpose still follows its own convert).
           prologue-bounds (filterv :const-prologue? bounds)
           replay-bounds   (filterv (complement :const-prologue?) bounds)
           prologue-graph (when (seq prologue-bounds) (record-fn prologue-bounds))
           _ (when prologue-graph ((rt-resolve device-id "replay-graph!") prologue-graph))
           ;; The non-profiling call is EXACTLY the 1-arity fast path (no opts map) so a
           ;; recorded non-profiling graph is byte-for-byte what it was before profiling existed.
           graph (if profile?
                   (record-fn replay-bounds {:barriers? true :profile? true})
                   (record-fn replay-bounds))]
       (swap! sess update :programs assoc pkey
              {:descriptor descriptor
               :roles effective-roles
               :graph graph
               ;; kept for close-session! (its queue/list must be destroyed) and as the seam for a
               ;; future refresh-constants! — replaying it re-derives the f16 copies of the weights.
               :prologue-graph prologue-graph
               :bounds bounds
               :profile? (boolean profile?)
               ;; per-GEMM f16 conversion scratch (NOT in :buffers — allocated directly via
               ;; make-buffer) — kept here so close-session! can free it instead of leaking it.
               :scratch-buffers @gemm-scratch
               :param->key param->key
               :alloc->key alloc->key
               :buffer-capacities
               (into {}
                     (map (fn [buffer-key]
                            [buffer-key (:n-elements (get buffers buffer-key))]))
                     (concat (vals param->key) (vals alloc->key)))
               ;; resolved buffer key of the functional :result-sym (may be a scratch alloc,
               ;; hence resolved through THIS program's key-fn, not a raw name->keyword).
               :result-key (when-let [rs (:result-sym descriptor)]
                             (or (get param->key rs) (get alloc->key rs) (keyword (name rs))))}))
     {::program-key pkey :descriptor descriptor})))

(defn- resolve-program
  "Find the bound-program entry for run-program!'s second argument: a HANDLE from bind-program!
   (looked up by its ::program-key), or — the single-program back-compat shape — the descriptor
   itself (unambiguous when the session holds one program; with several, matched by descriptor
   identity). Throws when nothing (or more than one thing) matches."
  [sess prog-or-handle]
  (let [programs (:programs @sess)]
    (if-let [pkey (::program-key prog-or-handle)]
      (or (get programs pkey)
          (throw (ex-info (str "run-program!: no program bound under key " pkey)
                          {:key pkey :bound (keys programs)})))
      (case (count programs)
        0 (throw (ex-info "run-program!: no program bound in this session — call bind-program! first" {}))
        1 (val (first programs))
        (let [matches (filter #(identical? prog-or-handle (:descriptor (val %))) programs)]
          (if (= 1 (count matches))
            (val (first matches))
            (throw (ex-info (str "run-program!: session holds " (count programs) " programs — "
                                 "pass the handle bind-program! returned (or a distinct descriptor)")
                            {:bound (keys programs)}))))))))

(defn- select-dispatch-step
  [descriptor selector]
  (let [steps (:steps descriptor)
        indexed (mapv vector (range) steps)
        selected
        (cond
          (nil? selector)
          (let [matches (filterv (fn [[_ step]] (some? (:dispatch step))) indexed)]
            (when-not (= 1 (count matches))
              (throw (ex-info "bound program requires exactly one dispatch step when :step is omitted"
                              {:reason :ambiguous-program-dispatch-step
                               :dispatch-phases (mapv (comp :phase second) matches)})))
            (first matches))

          (integer? selector)
          (when (<= 0 (long selector) (dec (count steps)))
            [(long selector) (nth steps (long selector))])

          (keyword? selector)
          (first (filterv (fn [[_ step]] (= selector (:phase step))) indexed))

          :else
          (throw (ex-info "program dispatch step selector must be nil, an index, or a phase keyword"
                          {:reason :invalid-program-dispatch-step-selector
                           :selector selector})))]
    (when-not selected
      (throw (ex-info "program dispatch step selector did not match a compiled step"
                      {:reason :program-dispatch-step-not-found
                       :selector selector :phases (mapv :phase steps)})))
    (let [[index step] selected]
      (when-not (:dispatch step)
        (throw (ex-info "selected compiled program step has no KernelDispatch"
                        {:reason :program-step-has-no-dispatch
                         :selector selector :index index :phase (:phase step)})))
      {:step-index index :step step :dispatch (kdispatch/validate! (:dispatch step))})))

(defn bound-program-dispatch
  "Return one compiled KernelDispatch step from a live resident program binding.

   `step-selector` is nil (requiring exactly one dispatch step), a zero-based step index, or a
   phase keyword. This is read-only compiler/runtime metadata; candidate binding remains explicit
   through program-dispatch-arguments."
  ([sess prog-or-handle]
   (bound-program-dispatch sess prog-or-handle nil))
  ([sess prog-or-handle step-selector]
   (let [program (resolve-program sess prog-or-handle)]
     (assoc (select-dispatch-step (:descriptor program) step-selector)
            :descriptor (:descriptor program)))))

(defn- physical-program-step-arguments
  [step logical-arguments]
  (if-not (:logical-bindings? step)
    logical-arguments
    (vec
     (mapcat (fn [spec value]
               (if (= :scalar (:kind spec))
                 [value]
                 (let [slots (:slots spec)]
                   (when-not (= 1 (count slots))
                     (throw (ex-info
                             "offline dispatch tuning cannot project a multi-slot logical resident binding"
                             {:reason :program-dispatch-multislot-logical-binding
                              :phase (:phase step) :binding (:binding spec) :slots slots})))
                   [value])))
             (:argument-specs step) logical-arguments))))

(defn program-dispatch-arguments
  "Project one compiled dispatch step onto a bound program's resident buffers.

   `program-arguments` follow the descriptor's :all-params order, just like bind-program!. The
   returned :arguments are in the selected alternatives' physical ABI order and contain stable
   session buffer keys plus typed scalar values. :reference-inputs retains the corresponding host
   buffers and a scalar environment (including compiler scalar lets) for an independent evaluator.
   Multi-slot logical SoA bindings fail explicitly until the resident program binder models their
   field allocations as stable views."
  ([sess prog-or-handle program-arguments]
   (program-dispatch-arguments sess prog-or-handle nil program-arguments))
  ([sess prog-or-handle step-selector program-arguments]
   (let [program (resolve-program sess prog-or-handle)
         descriptor (:descriptor program)
         {:keys [step-index step dispatch] :as selected}
         (select-dispatch-step descriptor step-selector)
         all-params (:all-params descriptor)
         _argument-count
         (when-not (and (sequential? program-arguments)
                        (= (count all-params) (count program-arguments)))
           (throw (ex-info "program dispatch arguments must follow descriptor :all-params"
                           {:reason :program-dispatch-argument-count
                            :expected (count all-params)
                            :actual (when (sequential? program-arguments)
                                      (count program-arguments))
                            :all-params all-params})))
         argmap (zipmap all-params program-arguments)
         key-of (fn [sym]
                  (or (get (:param->key program) sym)
                      (get (:alloc->key program) sym)
                      (keyword (name sym))))
         capacity-of (fn [key]
                       (or (get (:buffer-capacities program) key)
                           (get-in @sess [:buffers key :n-elements])))
         require-capacity!
         (fn [sym required]
           (let [key (key-of sym)
                 capacity (capacity-of key)]
             (when (and (some? capacity) (> (long required) (long capacity)))
               (throw (ex-info "compiled dispatch sample exceeds its resident buffer capacity"
                               {:reason :program-dispatch-buffer-capacity
                                :step-index step-index :phase (:phase step)
                                :sym sym :key key
                                :required-elements (long required)
                                :capacity-elements (long capacity)})))))
         _array-capacities
         (doseq [sym (:array-params descriptor)]
           (let [array (get argmap sym)]
             (when-not (and array (.isArray (class array)))
               (throw (ex-info "compiled dispatch array parameter must remain a JVM array"
                               {:reason :program-dispatch-array-argument
                                :sym sym :value-type (some-> array class)})))
             (require-capacity! sym (java.lang.reflect.Array/getLength array))))
         _scratch-capacities
         (doseq [{:keys [sym size-fn]} (:allocs descriptor)]
           (require-capacity! sym (size-fn program-arguments)))
         logical-arguments
         (mapv (fn [{:keys [kind sym type value-fn]}]
                 (if (= :scalar kind)
                   {:type type :value (value-fn program-arguments)}
                   (let [key (key-of sym)]
                     (when-not (contains? (:buffers @sess) key)
                       (throw (ex-info "compiled dispatch argument has no live resident buffer"
                                       {:reason :program-dispatch-buffer-not-resident
                                        :step-index step-index :phase (:phase step)
                                        :sym sym :key key})))
                     key)))
               (:argument-specs step))
         arguments (physical-program-step-arguments step logical-arguments)
         _abi (kabi/validate-arguments! (:abi (kdispatch/default-artifact dispatch)) arguments)
         resident-bindings
         (into {}
               (keep (fn [[{:keys [kind sym]} value]]
                       (when-not (= :scalar kind) [sym value])))
               (map vector (:argument-specs step) logical-arguments))
         direct-scalars (select-keys argmap (:scalar-params descriptor))
         derived-scalars
         (reduce (fn [bindings [{:keys [kind expression slot]} value]]
                   (if (= :scalar kind)
                     (let [raw (:value value)
                           slot-name (:name slot)]
                       (cond-> bindings
                         slot-name (assoc slot-name raw)
                         (symbol? expression) (assoc expression raw)))
                     bindings))
                 {}
                 (map vector (:argument-specs step) logical-arguments))]
     (assoc selected
            :descriptor descriptor
            :arguments arguments
            :resident-bindings resident-bindings
            :reference-inputs
            {:buffers (select-keys argmap (:array-params descriptor))
             :scalars (merge direct-scalars derived-scalars)}))))

(defn run-program!
  "Replay a bound resident GPU program: refresh ONLY the :input array params (buffer POINTERS are
   stable — only CONTENTS change), replay the recorded command graph, and download ONLY the
   :output params. :constant (weights) and :state (KV cache / on-device adapters) buffers are
   NEVER moved — they stay resident from bind-program!, and when several programs of this session
   share them (bind-program! :reuse-buffers) each program reads the LIVE device state the others
   left. prog-or-handle = the handle bind-program! returned (required for multi-program sessions)
   or, back-compat, the descriptor of a single-program session. args = values in :all-params
   order (same as bind-program!). Returns {output-param-sym → downloaded JVM array}."
  [sess prog-or-handle args]
  (let [{:keys [descriptor roles graph param->key result-key profile?]}
        (resolve-program sess prog-or-handle)
        {:keys [all-params array-params result-sym]} descriptor
        device-id (:device-id @sess)
        argmap (zipmap all-params args)
        replay-fn (rt-resolve device-id "replay-graph!")]
    ;; upload only per-call inputs (constant uploaded once at bind; state mutated in place on
    ;; device; output produced by the kernels so its prior content is irrelevant).
    (doseq [p array-params :when (= :input (get roles p :input))]
      (upload! sess (get param->key p) (get argmap p)))
    (replay-fn graph)
    ;; a PROFILED graph's events must be reset between replays (re-signaling a signaled event
    ;; is invalid); this replay's timestamps are intentionally discarded — use profile-program!
    ;; to read them.
    (when profile?
      (when-let [reset-fn (rt-resolve-soft device-id "reset-graph-events!")]
        (reset-fn graph)))
    ;; download :output array-params (in-place-mutated results) PLUS the functional :result-sym
    ;; (a fresh alloc returned by the deftm — the common SOAC case; it is not an array-param so
    ;; it has no :output role, but it IS the program's return value).
    (cond-> (into {} (for [p array-params :when (= :output (get roles p))]
                       [p (download sess (get param->key p))]))
      ;; download the functional :result-sym only when it is a distinct resident buffer (not
      ;; already an :output param, and actually allocated — a Void map-void has no result buffer).
      (and result-sym (not (some #(= result-sym %) array-params))
           (contains? (:buffers @sess) result-key))
      (assoc result-sym (download sess result-key)))))

(defn replay-program!
  "Replay a bound program WITHOUT downloading any output — the device-value path (S4).
   Refreshes only :input array params (same as run-program!), replays the recorded graph, and
   returns the session. Output/:state/:result buffers stay RESIDENT; the caller wraps them as
   DeviceArrays (raster.gpu.value) and downloads explicitly only when a host value is needed.
   This is what `raster.gpu.compiled/invoke-compiled` calls — run-program!'s upload+replay with
   the host round-trip removed. `resident-key` (below) resolves the buffer key for a param/result
   symbol so the wrapper can fetch the live buffer via `buffer`."
  [sess prog-or-handle args]
  (let [{:keys [descriptor roles graph param->key profile?]}
        (resolve-program sess prog-or-handle)
        {:keys [all-params array-params]} descriptor
        device-id (:device-id @sess)
        argmap (zipmap all-params args)
        replay-fn (rt-resolve device-id "replay-graph!")]
    (doseq [p array-params :when (= :input (get roles p :input))]
      (upload! sess (get param->key p) (get argmap p)))
    (replay-fn graph)
    (when profile?
      (when-let [reset-fn (rt-resolve-soft device-id "reset-graph-events!")]
        (reset-fn graph)))
    sess))

(defn resident-key
  "Resolve the session buffer key for a param/result symbol of a bound program — the seam
   `raster.gpu.compiled` uses to fetch a live resident buffer (via `buffer`) and wrap it as a
   DeviceArray. Handles array params (param->key) and the functional :result-sym (result-key)."
  [sess prog-or-handle sym]
  (let [{:keys [param->key result-key descriptor]} (resolve-program sess prog-or-handle)]
    (or (get param->key sym)
        (when (= sym (:result-sym descriptor)) result-key)
        (keyword (name sym)))))

(defn profile-program!
  "The profiling twin of run-program!: same upload → replay → download sequence over a program
   bound with {:profile? true}, but reads the per-kernel DEVICE timestamps the replay produced.
   A separate verb (rather than an opts flag on run-program!) because profiling is a BIND-time
   property — the events are recorded into the command graph — and because run-program!'s
   return shape stays stable for every existing caller.

   Returns
     {:result          {output-param → array}        ;; exactly run-program!'s return value
      :profile         [{:kernel-name str :phase kw :ms double :context-ms double} …]
                       ;; execution order; :ms = device (global-timestamp) kernel duration
      :kernel-total-ms double     ;; Σ per-kernel device time
      :device-wall-ms  double     ;; device span: first kernel start → last kernel end
                                  ;; (includes inter-kernel gaps = dispatch/barrier overhead)
      :host-wall-ms    double}    ;; System/nanoTime around the replay call, for comparison

   Device kernel times come from backend timestamp events (Level Zero or OpenCL), immune to host
   scheduling and far more stable under platform power-state swings than host wall time. Throws
   if the program was not bound with {:profile? true}."
  [sess prog-or-handle args]
  (let [{:keys [descriptor roles graph param->key result-key profile?]}
        (resolve-program sess prog-or-handle)
        {:keys [all-params array-params result-sym]} descriptor
        device-id (:device-id @sess)
        argmap (zipmap all-params args)
        replay-fn (rt-resolve device-id "replay-graph!")
        read-ts-fn (rt-resolve device-id "read-graph-timestamps!")]
    (when-not profile?
      (throw (ex-info "profile-program!: program was not bound with {:profile? true} — rebind it with (bind-program! sess descriptor args roles {:profile? true})"
                      {:program (or (::program-key prog-or-handle) :program)})))
    (doseq [p array-params :when (= :input (get roles p :input))]
      (upload! sess (get param->key p) (get argmap p)))
    (let [t0 (System/nanoTime)
          _ (replay-fn graph)
          t1 (System/nanoTime)
          prof (read-ts-fn graph)   ;; reads AND resets the events
          result (cond-> (into {} (for [p array-params :when (= :output (get roles p))]
                                    [p (download sess (get param->key p))]))
                   (and result-sym (not (some #(= result-sym %) array-params))
                        (contains? (:buffers @sess) result-key))
                   (assoc result-sym (download sess result-key)))]
      {:result result
       :profile (mapv #(select-keys % [:kernel-name :phase :ms :context-ms]) (:kernels prof))
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

(defn measure-program!
  "Explicit offline device-event measurement of a bound resident program.

   The program must have been bound with {:profile? true}. Inputs upload once before sampling;
   output downloads are intentionally absent because they are not part of device timing. The
   autotuner must validate a candidate against its oracle before calling this function.

   A program with a :state resident role requires :before-sample! so repeated timing cannot
   silently benchmark successively mutated state. Pure/output-only programs need no restore hook.
   Returns a raster.gpu.measurement/Measurement."
  [sess prog-or-handle args & {:keys [before-sample!] :as opts}]
  (let [{:keys [descriptor roles graph param->key profile?]}
        (resolve-program sess prog-or-handle)
        {:keys [all-params array-params]} descriptor
        argmap (zipmap all-params args)]
    (when-not profile?
      (throw (ex-info "measure-program!: program was not bound with {:profile? true}"
                      {:program (or (::program-key prog-or-handle) :program)})))
    (when (and (some #(= :state %) (vals roles)) (nil? before-sample!))
      (throw (ex-info "measure-program!: stateful programs require :before-sample! restoration"
                      {:state-params (into #{} (keep (fn [[sym role]]
                                                       (when (= :state role) sym))) roles)})))
    (doseq [param array-params :when (= :input (get roles param :input))]
      (upload! sess (get param->key param) (get argmap param)))
    (apply measure-graph! sess graph (mapcat identity opts))))

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
