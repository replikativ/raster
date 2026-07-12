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
            [raster.compiler.core.inference :as inf]
            [raster.compiler.pipeline :as pl]
            [raster.core :as rcore]))

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
        mk       (rt-resolve device-id "make-buffer")
        upload   (rt-resolve device-id "array->buffer!")
        as-fbuf  (rt-resolve device-id "buffer-as-float-buffer")
        as-ibuf  (rt-resolve device-id "buffer-as-int-buffer")

        buf-of (fn [arr dtype n]
                 (let [buf (mk n dtype)]
                   (if arr
                     (if unified?
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
           :arena-id  arena-id
           :kernels   {}       ;; {phase-key → [kernel-info ...]}
           :buffers   {}       ;; {buf-key → DeviceBuffer}
           :programs  {}       ;; {program-key → bound resident program (see bind-program!)}
           :closed?   false})))

(defn close-session!
  "Free all buffers and kernels in a session. Idempotent and thread-safe.

  Also destroys the per-binding fresh kernel handles (:prepared) and recorded command graphs
  (:graphs). These hold dedicated driver objects (zeKernel per bind, zeCommandQueue+List per
  graph) that are NOT in the kernel registry, so close-kernel-arena! never reaches them — without
  this every session leaks them and the driver eventually aborts (the source of the SIGABRTs)."
  [sess]
  (locking sess
    (let [{:keys [device-id arena-id buffers prepared graphs programs closed?]} @sess]
      (when-not closed?
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
            (when destroy-prepared!
              (doseq [b (:bounds prog)] (try (destroy-prepared! b) (catch Exception _))))))
        (free-buffers-internal! buffers device-id)
        (let [close-arena! (rt-resolve device-id "close-kernel-arena!")]
          (close-arena! arena-id))
        (swap! sess assoc :closed? true :buffers {} :kernels {} :prepared {} :graphs {} :programs {})))))

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
  (let [device-id (:device-id @sess)
        ;; Allocate one-by-one so we can free on partial failure.
        ;; alloc-buffers-internal uses `into` which is eager, so a failure
        ;; partway through would leak already-allocated buffers.
        allocated (volatile! {})
        new-bufs (try
                   (doseq [[k spec] buffer-specs]
                     (let [single (alloc-buffers-internal {k spec} device-id)]
                       (vswap! allocated merge single)))
                   @allocated
                   (catch Exception e
                     (when (seq @allocated)
                       (free-buffers-internal! @allocated device-id))
                     (throw e)))]
    (swap! sess update :buffers merge new-bufs)
    new-bufs))

(defn free-buffer!
  "Free a specific buffer from the session by key."
  [sess key]
  (let [{:keys [device-id buffers]} @sess]
    (when-let [buf (get buffers key)]
      ((rt-resolve device-id "free-buffer!") buf)
      (swap! sess update :buffers dissoc key))))

;; ================================================================
;; Buffer argument resolution
;; ================================================================

(defn- resolve-kernel-bufs
  "Resolve buffer arguments for a kernel from session buffers.
   sym->buf-key maps kernel param names to session buffer keys."
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
        (:array-params kernel-info)))

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

(defn buffer
  "Get a DeviceBuffer from the session by key."
  [sess key]
  (get-in @sess [:buffers key]))

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

     :reduce    SegRed sig (inputs…, output, scl…, _n_bound); bind inputs ++ [output], SINGLE
                workgroup (:group-count 1) so the grid-stride loop writes output[0] device-resident.
     :map       SegMap sig (inputs…, out, scl…, _n_bound); the output is a SEPARATE param not in the
                registry :array-params, so bind the step's full :arrays (inputs ++ [out]).
     :map-void  output is an in-place array among :array-params; bind by name via prepare!."
  [sess step args sym->key]
  (let [device-id (:device-id @sess)
        {:keys [kernel-name arrays n-fn scalar-specs phase convention output]} step
        info-fn (rt-resolve device-id "kernel-registry-entry")]
    (when-not (#{:map-void :map :reduce} convention)
      (throw (ex-info (str "bind-step! cannot bind a " convention " step (" kernel-name
                           ") — only :map-void / :map / :reduce are supported on the resident path")
                      {:convention convention :kernel kernel-name})))
    (let [ki (or (info-fn kernel-name)
                 (throw (ex-info (str "Program kernel not registered: " kernel-name)
                                 {:kernel kernel-name})))
          bufs (:buffers @sess)
          resolve-buf (fn [a] (or (get bufs (sym->key a))
                                  (throw (ex-info (str "No buffer for kernel arg: " a " → " (sym->key a))
                                                  {:kernel kernel-name :available (keys bufs)}))))
          scalars (mapv (fn [{:keys [type value-fn]}] {:type type :value (value-fn args)})
                        scalar-specs)
          bind-fn (rt-resolve device-id "bind-registered-map-void-kernel")]
      (case convention
        :reduce
        (let [array-bufs (conj (mapv resolve-buf (:array-params ki)) (resolve-buf output))
              prepared (bind-fn kernel-name array-bufs scalars (long (n-fn args)) {:group-count 1})]
          (swap! sess assoc-in [:prepared phase] prepared))

        :map
        (let [array-bufs (mapv resolve-buf arrays)
              prepared (bind-fn kernel-name array-bufs scalars (long (n-fn args)) {})]
          (swap! sess assoc-in [:prepared phase] prepared))

        (let [sym->buf (into {} (map (fn [p] [(name p) (sym->key p)]) (:array-params ki)))]
          (swap! sess assoc-in [:kernels phase] [ki])
          (prepare! sess phase sym->buf scalars (long (n-fn args)) {:kernel-phase phase}))))
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
        types (:scalar-types (opencl-pass/derive-param-types
                              (:raster.core/deftm-params (meta v))
                              (:raster.core/deftm-tags (meta v)) dtype))]
    (mapv (fn [sp]
            (let [t (get types sp :float)
                  raw (if (contains? scalars (name sp)) (get scalars (name sp))
                          (if (contains? scalars sp) (get scalars sp)
                              (throw (ex-info (str "chain step for " (:name (meta v))
                                                   " missing scalar: " sp)
                                              {:need (:scalar-params ki) :have (keys scalars)}))))]
              {:type t :value (case t :int (int raw) :long (long raw) :double (double raw) (float raw))}))
          (:scalar-params ki))))

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
         roles (into {} (map (fn [[k v]] [k (or (nth v 3 nil) :scratch)]) buffers))]
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
         roles (into {} (map (fn [[k v]] [k (or (nth v 3 nil) :scratch)]) buffers))]
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

   :gemm steps bind per the descriptor's :gemm-precision policy (set at compile time by
   compile-gpu-program, default :f16-xmx; a bind-time caller may override with
   (assoc descriptor :gemm-precision …) since the descriptor is plain data):
     :f16-xmx    — convert A/B f32→f16 and run the XMX gemm (f32 accumulate/output). Fast
                   (systolic), but the f16 input conversion costs gradient precision
                   (~1e-3-level composed-grad noise) — the decode/inference default.
     :f32-scalar — bind the plain scalar f32 GEMM for ALL :gemm steps: reads the f32
                   residents directly, no convert/transpose expansion kernels. Exact f32
                   grads (~1e-6-level parity) — what training wants.
   The XMX hardware pitch gate (n<8 or k<8 → scalar) applies regardless of policy."
  ([sess descriptor args] (bind-program! sess descriptor args {} {}))
  ([sess descriptor args roles] (bind-program! sess descriptor args roles {}))
  ([sess descriptor args roles {:keys [key reuse-buffers rename] :or {key :program rename {}}}]
   (let [device-id (:device-id @sess)
         pkey key
         _ (when (contains? (:programs @sess) pkey)
             (throw (ex-info (str "bind-program!: program key " pkey " already bound in this session"
                                  " — bind each program under a distinct :key")
                             {:key pkey :bound (keys (:programs @sess))})))
         {:keys [dtype all-params array-params allocs steps]} descriptor
         gemm-precision (or (:gemm-precision descriptor) :f16-xmx)
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
                       (Class/forName "[S") :short
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
         alloc-specs (into {} (keep (fn [{:keys [sym size-fn]}]
                                      (let [n (long (size-fn args))]
                                        (reuse-or-spec (alloc->key sym) dt n [dt n nil]
                                                       (str "scratch " sym)))))
                           allocs)
         info-fn   (rt-resolve device-id "kernel-registry-entry")
         bind-fn   (rt-resolve device-id "bind-registered-map-void-kernel")
         gemm-fn   (rt-resolve device-id "bind-registered-gemm!")
         conv-fn   (rt-resolve device-id "bind-registered-convert!")
         trans-fn  (rt-resolve device-id "bind-registered-transpose!")
         mkbuf-fn  (rt-resolve device-id "make-buffer")
         record-fn (rt-resolve device-id "record-graph!")
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
           step->bounds
           (fn [{:keys [kernel-name arrays n-fn scalar-specs convention accumulator output] :as step}]
             (case convention
               ;; GEMM (Option B): [convert A f32→f16][convert B f32→f16][fp16 XMX gemm → f32 C].
               ;; A/B are converted into per-GEMM f16 scratch (kept alive on the session); weights
               ;; convert redundantly per replay for now (correctness-first) — hoist to a once-at-
               ;; bind :constant f16 upload later.
               :gemm
               (let [m (long ((:m-fn step) args)) n (long ((:n-fn step) args)) k (long ((:k-fn step) args))
                     abuf (buf-of (:A step) :gemm-A) bbuf (buf-of (:B step) :gemm-B)
                     cbuf (buf-of (:C step) :gemm-C)]
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
                     [{:bound (scalar-gemm-fn abuf bbuf cbuf m n k (:variant step))}])
                   (let [a16 (mkbuf-fn (* m k) :half)
                     ;; each expansion kernel (convert/transpose/gemm) pre-bakes its FULL gc-seg —
                     ;; wrap as {:bound bnd} with NO :group-count so record-graph! keeps the grid.
                         raw
                         (case (:variant step)
                       ;; C = A[m,k] · B[k,n]
                           :nn
                           (let [b16 (mkbuf-fn (* k n) :half)]
                             (swap! gemm-scratch conj a16 b16)
                             [(conv-fn abuf a16 (* m k)) (conv-fn bbuf b16 (* k n))
                              (gemm-fn a16 b16 cbuf m n k :float)])
                       ;; C = A[m,k] · B[n,k]ᵀ — convert B then transpose [n,k]→[k,n], then :nn gemm.
                       ;; (HF linear weights [out,in] and attention Q·Kᵀ are :nt.)
                           :nt
                           (let [b16 (mkbuf-fn (* n k) :half) bt16 (mkbuf-fn (* k n) :half)]
                             (swap! gemm-scratch conj a16 b16 bt16)
                             [(conv-fn abuf a16 (* m k)) (conv-fn bbuf b16 (* n k))
                              (trans-fn b16 bt16 n k :half) (gemm-fn a16 bt16 cbuf m n k :float)])
                       ;; C[m,n] = Aᵀ·B — A stored [k,m], B [k,n]. Convert A then transpose
                       ;; [k,m]→[m,k], convert B ([k,n] already the :nn B layout), then :nn gemm.
                       ;; (linear-dW = dgemm-tn! : the weight-gradient backward matmul.)
                           :tn
                           (let [at16 (mkbuf-fn (* m k) :half) b16 (mkbuf-fn (* k n) :half)]
                             (swap! gemm-scratch conj a16 at16 b16)
                             [(conv-fn abuf a16 (* k m)) (trans-fn a16 at16 k m :half)
                              (conv-fn bbuf b16 (* k n)) (gemm-fn at16 b16 cbuf m n k :float)])
                           (throw (ex-info (str "GEMM variant not yet wired on resident path: " (:variant step)
                                                " (only :nn / :nt / :tn)") {:variant (:variant step)})))]
                     (mapv (fn [b] {:bound b}) raw))))
               ;; map / map-void bind through bind-registered-map-void-kernel (output is just
               ;; another resident buffer). Resolve buffers from the STEP's :arrays (full C-sig
               ;; order incl. output).
               (:map :map-void)
               (do (or (info-fn kernel-name)
                       (throw (ex-info (str "Program kernel not registered: " kernel-name) {:kernel kernel-name})))
                   (let [buf-vec (mapv #(buf-of % kernel-name) arrays)
                         scalars (mapv (fn [{:keys [type value-fn]}] {:type type :value (value-fn args)})
                                       scalar-specs)]
                     [(bind-fn kernel-name buf-vec scalars (long (n-fn args)))]))
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
                     [(bind-fn zk [out-buf] [] (long out-size))
                      (scatter-fn kernel-name [out-buf src-buf idx-buf] n stride)]))
               ;; reduce: SegRed sig (inputs…, output, scl…, _n_bound) — bind the registry's
               ;; :array-params ++ [output] and launch a SINGLE workgroup (:group-count 1) so
               ;; the kernel's grid-stride loop covers all n and writes output[0] into its
               ;; resident 1-elem buffer (mirrors bind-step!'s :reduce; the #42 leftover).
               :reduce
               (let [ki (or (info-fn kernel-name)
                            (throw (ex-info (str "Program kernel not registered: " kernel-name)
                                            {:kernel kernel-name})))
                     array-bufs (conj (mapv #(buf-of % kernel-name) (:array-params ki))
                                      (buf-of output kernel-name))
                     scalars (mapv (fn [{:keys [type value-fn]}] {:type type :value (value-fn args)})
                                   scalar-specs)]
                 [(bind-fn kernel-name array-bufs scalars (long (n-fn args)) {:group-count 1})])
               (throw (ex-info (str "bind-program! cannot bind a " convention " step — only "
                                    ":map / :map-void / :reduce / :gemm / :scatter are wired on the resident path")
                               {:convention convention :kernel kernel-name}))))
           bounds (vec (mapcat step->bounds steps))
           graph (record-fn bounds)]
       (swap! sess update :programs assoc pkey
              {:descriptor descriptor
               :roles effective-roles
               :graph graph
               :bounds bounds
               :param->key param->key
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
  (let [{:keys [descriptor roles graph param->key result-key]} (resolve-program sess prog-or-handle)
        {:keys [all-params array-params result-sym]} descriptor
        device-id (:device-id @sess)
        argmap (zipmap all-params args)
        replay-fn (rt-resolve device-id "replay-graph!")]
    ;; upload only per-call inputs (constant uploaded once at bind; state mutated in place on
    ;; device; output produced by the kernels so its prior content is irrelevant).
    (doseq [p array-params :when (= :input (get roles p :input))]
      (upload! sess (get param->key p) (get argmap p)))
    (replay-fn graph)
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
