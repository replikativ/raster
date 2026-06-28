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
  "Resolve a deftm var through dispatch table to the mangled backing var.
   Returns the var that carries :raster.core/deftm metadata."
  [v]
  (if (:raster.core/deftm (meta v))
    v
    (when-let [dt (:raster.core/dispatch-table (meta v))]
      (let [entries (vals @dt)
            method (first (first entries))
            ns-obj (:ns (meta v))
            mangled-name (symbol (str (:name (meta v)) "_m_"
                                      (clojure.string/join "_" (:tags method))))]
        (ns-resolve ns-obj mangled-name)))))

(defn get-walked-body
  "Get the walker-processed body from a deftm var.
   Throws if the var has no walked body."
  [v]
  (let [resolved (or (resolve-deftm-var v) v)]
    (or (:raster.core/deftm-walked-body (meta resolved))
        (rcore/ensure-walked-body! resolved)
        (throw (ex-info "Var has no deftm walked body" {:var v})))))

;; ================================================================
;; Internal: kernel compilation
;; ================================================================

(defn- compile-deftm-internal!
  "Compile a deftm var's par forms to GPU kernels and register them.
   Returns vector of kernel-info maps."
  [v device-id {:keys [dtype min-elements] :or {dtype :float min-elements 0}}]
  (let [walked-body (get-walked-body v)
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
           :closed?   false})))

(defn close-session!
  "Free all buffers and kernels in a session. Idempotent and thread-safe.

  Also destroys the per-binding fresh kernel handles (:prepared) and recorded command graphs
  (:graphs). These hold dedicated driver objects (zeKernel per bind, zeCommandQueue+List per
  graph) that are NOT in the kernel registry, so close-kernel-arena! never reaches them — without
  this every session leaks them and the driver eventually aborts (the source of the SIGABRTs)."
  [sess]
  (locking sess
    (let [{:keys [device-id arena-id buffers prepared graphs closed?]} @sess]
      (when-not closed?
        ;; backend-specific: the bound-dispatch + command-graph path is ze-only, so resolve the
        ;; destroyers nil-safely rather than via rt-resolve (which throws on backends lacking them).
        (let [ns-sym (case (backend-type device-id) :ze 'raster.gpu.ze-runtime :ocl 'raster.gpu.ocl-runtime)
              destroy-prepared! (requiring-resolve (symbol (str ns-sym) "destroy-prepared!"))
              destroy-graph!    (requiring-resolve (symbol (str ns-sym) "destroy-graph!"))]
          (when destroy-graph!    (doseq [[_ g] graphs]   (try (destroy-graph! g)    (catch Exception _))))
          (when destroy-prepared! (doseq [[_ p] prepared] (try (destroy-prepared! p) (catch Exception _)))))
        (free-buffers-internal! buffers device-id)
        (let [close-arena! (rt-resolve device-id "close-kernel-arena!")]
          (close-arena! arena-id))
        (swap! sess assoc :closed? true :buffers {} :kernels {} :prepared {} :graphs {})))))

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
         kernels (compile-deftm-internal! v device-id opts)]
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

(defn bind-program!
  "Bind a resident GPU program (a descriptor from pipeline/compile-gpu-program) to this session
   ONCE: allocate resident buffers for the array params + intermediate scratch, install +
   prepare! each kernel step against them, and record the kernel sequence as a command graph.
   After binding, run-program! replays the whole sequence with NO re-binding — the resident
   bound-dispatch path, vs make-gpu-fn's per-call JVM-array staging. The bound machinery is
   convention-agnostic, so map! and map-void! kernels bind identically (a map! kernel's output
   is just another resident buffer in its :array-params).

   args = values in the descriptor's :all-params order (JVM arrays for array params, numbers for
   scalars). Buffer keys are the param/intermediate sym names as keywords.

   roles = optional {param-sym → :constant|:state|:input|:output} override of the descriptor's
   derived defaults (read-only→:input, written→:output). Declare cross-call persistence the
   program can't derive: :constant = weights (uploaded once here, never re-uploaded by
   run-program!); :state = persistent device state e.g. a KV cache (never downloaded). All buffer
   CONTENTS are uploaded once here at bind; run-program! then moves only :input (up) and :output
   (down)."
  ([sess descriptor args] (bind-program! sess descriptor args {}))
  ([sess descriptor args roles]
   (let [device-id (:device-id @sess)
         {:keys [dtype all-params array-params allocs steps]} descriptor
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
         param-specs (into {} (map (fn [p]
                                     (let [arr (get argmap p)]
                                       [(keyword (name p)) [(arr-dtype arr) (nel arr) arr]]))
                                   array-params))
         alloc-specs (into {} (map (fn [{:keys [sym size-fn]}]
                                     [(keyword (name sym)) [dt (long (size-fn args)) nil]])
                                   allocs))
         info-fn (rt-resolve device-id "kernel-registry-entry")]
     (alloc! sess (merge param-specs alloc-specs))
     (doseq [{:keys [kernel-name n-fn scalar-specs phase convention]} steps]
       ;; Only the map conventions bind through bind-registered-map-void-kernel (output is just
       ;; another resident buffer). :reduce has a different launch/partial-sum shape and would
       ;; SILENTLY mis-bind here — reject it loudly rather than miscompile. (Decode needs no
       ;; reduce STEP: reductions live inside map-void kernels. Add a reduce bind path when one
       ;; appears.)
       (when-not (#{:map-void :map} convention)
         (throw (ex-info (str "bind-program! cannot bind a " convention " step (" kernel-name
                              ") — only :map-void / :map are supported on the resident path")
                         {:convention convention :kernel kernel-name})))
       (let [ki (or (info-fn kernel-name)
                    (throw (ex-info (str "Program kernel not registered: " kernel-name)
                                    {:kernel kernel-name})))
            ;; kernel array-param NAME → resident buffer key. For map-void the invoke arrays ARE
            ;; the kernel's own param syms, and intermediates/params are keyed by the same names,
            ;; so name→same-name-keyword is exact (and SoA-order-robust: prepare! orders by
            ;; :array-params itself via resolve-kernel-bufs).
             sym->buf (into {} (map (fn [p] [(name p) (keyword (name p))]) (:array-params ki)))
            ;; typed scalars ({:type :int|:float|:double :value v}) so int params bind as int.
             scalars (mapv (fn [{:keys [type value-fn]}] {:type type :value (value-fn args)})
                           scalar-specs)]
         (swap! sess assoc-in [:kernels phase] [ki])
         (prepare! sess phase sym->buf scalars (long (n-fn args)) {:kernel-phase phase})))
     (record-graph! sess (mapv :phase steps) :program)
     (swap! sess assoc :program-descriptor descriptor :program-roles effective-roles)
     sess)))

(defn run-program!
  "Replay a bound resident GPU program: refresh ONLY the :input array params (buffer POINTERS are
   stable — only CONTENTS change), replay the recorded command graph, and download ONLY the
   :output params. :constant (weights) and :state (KV cache) buffers are NEVER moved — they stay
   resident from bind-program!. args = values in :all-params order (same as bind-program!).
   Returns {output-param-sym → downloaded JVM array}."
  [sess descriptor args]
  (let [{:keys [all-params array-params]} descriptor
        roles (:program-roles @sess)
        argmap (zipmap all-params args)]
    ;; upload only per-call inputs (constant uploaded once at bind; state mutated in place on
    ;; device; output produced by the kernels so its prior content is irrelevant).
    (doseq [p array-params :when (= :input (get roles p :input))]
      (upload! sess (keyword (name p)) (get argmap p)))
    (replay! sess :program)
    ;; download only outputs (inputs/constants/state are not host-visible results).
    (into {} (for [p array-params :when (= :output (get roles p))]
               [p (download sess (keyword (name p)))]))))

(defn kernel
  "Get kernel info vector from the session by phase key."
  [sess phase-key]
  (get-in @sess [:kernels phase-key]))

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
