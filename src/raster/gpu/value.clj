(ns raster.gpu.value
  "The device-value type — `DeviceArray`, a `jax.Array` analogue (S4 §1).

   A `DeviceArray` is a first-class value that *is* device residency: it wraps a
   backend `DeviceBuffer` (ze_runtime/DeviceBuffer or ocl equivalent) and adds the
   three things a *value* needs that the byte-storage buffer lacks — WHICH device it
   lives on, its logical SHAPE, and an OWNERSHIP/lifetime discipline. This is the
   prerequisite for functional artifact invocation (values-in / values-out) and for
   donation-as-residency: without a device value, `:donate` is just today's `:state`
   plus a per-step host copy (the design's C2 finding).

   Ownership (`owner` field), the lifetime discipline:
     ::owned    — raster allocated it; freed on `free!`, session-close, or GC (Cleaner
                  safety net). The only kind reclaim frees.
     ::donated  — ownership was transferred INTO a `step` call; the value is marked
                  consumed (freed? = true) and its buffer now belongs to an output
                  value. Reads throw; its Cleaner is disarmed (never frees the buffer).
     ::aliased  — a view onto another value's buffer (a KV-cache slice). Never frees on
                  reclaim; the base owner does.
     ::external — wraps a buffer raster did not allocate. Never freed by raster.

   Backend-neutral by construction: every runtime call routes through `device-id`
   (`:ze:0` / `:ocl:0`) via `requiring-resolve`, never a hardcoded `:ze`. Depends only
   on the runtime records, never on the session layer — this ns sits at the bottom of
   the GPU dependency graph so `raster.gpu.compiled` / `raster.gpu.core` can build on it
   without a cycle."
  (:import [java.lang.ref Cleaner]
           [java.util.concurrent.atomic AtomicBoolean]))

;; ================================================================
;; Backend dispatch (mirror of core/rt-resolve; kept local so this
;; ns depends on neither the session nor a specific runtime)
;; ================================================================

(defn- backend-ns
  [device-id]
  (let [s (name device-id)]
    (cond
      (.startsWith s "ze")  'raster.gpu.ze-runtime
      (.startsWith s "ocl") 'raster.gpu.ocl-runtime
      :else (throw (ex-info (str "Unknown GPU backend for device " device-id
                                 " — use :ze:N or :ocl:N")
                            {:device-id device-id})))))

(defn- rt-fn
  "Resolve a runtime fn by name for the given device-id's backend."
  [device-id fn-name]
  (let [ns-sym (backend-ns device-id)]
    (or (requiring-resolve (symbol (str ns-sym) fn-name))
        (throw (ex-info (str "Cannot resolve " fn-name " in " ns-sym)
                        {:device-id device-id :fn fn-name})))))

;; ================================================================
;; The Cleaner (GC-backed reclamation safety net)
;; ================================================================

(def ^:private ^Cleaner cleaner (Cleaner/create))

(defn- arm-cleaner!
  "Register a GC cleaning action that frees `buffer` on `device-id` IF this value is
   still the live owner when it becomes unreachable. Captures only value-independent
   state (never the DeviceArray) so the action does not pin the value in memory.

   Idempotent with explicit `free!` and donation via the shared AtomicBoolean:
   whoever wins `compareAndSet(false,true)` performs (or suppresses) the free exactly
   once. Donation wins the CAS without freeing → the buffer survives on the output
   value and this Cleaner is disarmed."
  [holder ^AtomicBoolean freed buffer device-id free-on-reclaim?]
  (.register cleaner holder
             (reify Runnable
               (run [_]
                 (when (and free-on-reclaim? (.compareAndSet freed false true))
                   (try ((rt-fn device-id "free-buffer!") buffer)
                        (catch Throwable _)))))))

;; ================================================================
;; The type
;; ================================================================

(defrecord DeviceArray
           [buffer      ;; backend DeviceBuffer (byte storage: segment + n-elements + dtype)
            device      ;; device-id keyword (:ze:0) — which runtime it lives on
            dtype       ;; :float :half :int … (redundant with buffer.dtype; cheap inspection)
            shape       ;; [d0 d1 …] logical shape (the buffer knows only flat n-elements)
            owner       ;; ::owned | ::donated | ::aliased | ::external
            freed])     ;; AtomicBoolean — use-after-free / double-free / donation guard

(defn device-array? [x] (instance? DeviceArray x))

(defn- make-device-array
  "Build a DeviceArray over an existing backend buffer, arming the Cleaner for
   ::owned values. Low-level constructor used by `->device` and output projection."
  [buffer device-id dtype shape owner]
  (let [freed (AtomicBoolean. false)
        da    (->DeviceArray buffer device-id dtype (vec shape) owner freed)]
    (arm-cleaner! da freed buffer device-id (= owner ::owned))
    da))

;; ================================================================
;; Lifetime guards
;; ================================================================

(defn live?
  "True if the value has not been freed, consumed (donated), or reclaimed."
  [^DeviceArray da]
  (not (.get ^AtomicBoolean (:freed da))))

(defn- ensure-live!
  [^DeviceArray da op]
  (when (.get ^AtomicBoolean (:freed da))
    (throw (ex-info (str "DeviceArray use-after-free: " op
                         " on a value that was already freed, consumed by donation,"
                         " or reclaimed")
                    {:op op :owner (:owner da) :shape (:shape da) :device (:device da)})))
  da)

;; ================================================================
;; Host <-> device transfer
;; ================================================================

(defn ->device
  "Lift a JVM primitive array onto `device-id` as a fresh ::owned DeviceArray.
   dtype is auto-detected from the array type unless supplied in opts.

   opts: {:shape [d0 d1 …]   ;; logical shape (default [n])
          :dtype :float|…}    ;; override auto-detected dtype"
  ([arr device-id] (->device arr device-id nil))
  ([arr device-id opts]
   (let [dtype   (:dtype opts)
         buffer  (if dtype
                   ((rt-fn device-id "buffer-of-array") arr dtype)
                   ((rt-fn device-id "buffer-of-array") arr))
         n       (:n-elements buffer)
         shape   (or (:shape opts) [n])]
     (make-device-array buffer device-id (:dtype buffer) shape ::owned))))

(defn ->host
  "Download a DeviceArray to a fresh JVM array. Throws if the value is not live.
   For :half/:float16 returns the encoded short array (matches buffer->array)."
  [^DeviceArray da]
  (ensure-live! da :->host)
  ((rt-fn (:device da) "buffer->array") (:buffer da)))

;; ================================================================
;; Explicit free + donation
;; ================================================================

(defn free!
  "Explicitly free a DeviceArray. Idempotent and race-safe with GC reclamation and
   donation (whoever wins the CAS acts once). Only ::owned values free their buffer;
   ::aliased/::external values just mark themselves dead without touching the buffer
   (the base owner frees it)."
  [^DeviceArray da]
  (let [^AtomicBoolean freed (:freed da)]
    (when (.compareAndSet freed false true)
      (when (= (:owner da) ::owned)
        (try ((rt-fn (:device da) "free-buffer!") (:buffer da))
             (catch Throwable _)))))
  nil)

(defn consume!
  "Consume a DeviceArray for donation: mark it dead (reads now throw) and hand its
   buffer to the caller WITHOUT freeing it. Wins the CAS so the input's Cleaner is
   disarmed — the buffer's ownership moves to the output value the caller builds with
   `donate-output`. Throws if the value was already freed/consumed."
  [^DeviceArray da]
  (ensure-live! da :consume!)
  (let [^AtomicBoolean freed (:freed da)]
    (when-not (.compareAndSet freed false true)
      (throw (ex-info "DeviceArray consumed concurrently"
                      {:owner (:owner da) :shape (:shape da)})))
    (:buffer da)))

(defn donate-output
  "Build the output DeviceArray of a donated in→out pair: a fresh ::owned value over
   the consumed input's buffer, with the output's logical shape/dtype. Call `consume!`
   on the input first (this fn takes the buffer it returned)."
  [buffer device-id dtype shape]
  (make-device-array buffer device-id dtype shape ::owned))

;; ================================================================
;; Constructors for the other ownership kinds
;; ================================================================

(defn wrap-owned
  "Wrap an existing raster-allocated backend buffer as an ::owned DeviceArray
   (Cleaner-armed). Used by output projection for freshly allocated result buffers."
  [buffer device-id dtype shape]
  (make-device-array buffer device-id dtype shape ::owned))

(defn wrap-external
  "Wrap a caller-owned backend buffer raster did not allocate. Never freed by raster."
  [buffer device-id dtype shape]
  (make-device-array buffer device-id dtype shape ::external))

(defn alias-of
  "Build an ::aliased view onto `base`'s buffer with a (possibly different) logical
   shape — the cross-artifact / KV-slice sharing case. Never frees on reclaim; the
   base owner's lifetime governs the buffer."
  ([^DeviceArray base] (alias-of base (:shape base) (:dtype base)))
  ([^DeviceArray base shape dtype]
   (make-device-array (:buffer base) (:device base) dtype shape ::aliased)))
