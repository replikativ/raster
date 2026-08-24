(ns raster.gpu.value
  "The device-value type — `DeviceArray`, a `jax.Array` analogue (S4 §1).

   A `DeviceArray` is a first-class value that *is* device residency: it wraps a
   backend `DeviceBuffer` (ze_runtime/DeviceBuffer or ocl equivalent) and a canonical
   backend-neutral `BufferView`: stable allocation identity, exact checked byte range,
   logical shape/strides, dtype, and memory properties. It also adds the ownership and
   lifetime discipline a byte-storage buffer lacks. This is the
   prerequisite for functional artifact invocation (values-in / values-out) and for
   donation-as-residency: without a device value, `:donate` is just today's `:state`
   plus a per-step host copy (the design's C2 finding).

   Ownership (`owner` field), the lifetime discipline:
     ::owned    — raster allocated it; freed on `free!`, session-close, or GC (Cleaner
                  safety net). The only kind reclaim frees.
     ::aliased  — a view onto another value's buffer (a KV-cache slice). Never frees on
                  reclaim; the base owner does.
     ::external — wraps a buffer raster did not allocate. Never freed by raster.

   Backend-neutral by construction: every runtime call routes through `device-id`
   (`:ze:0` / `:ocl:0`) via `requiring-resolve`, never a hardcoded `:ze`. Depends only
   on the runtime records, never on the session layer — this ns sits at the bottom of
   the GPU dependency graph so `raster.gpu.compiled` / `raster.gpu.core` can build on it
   without a cycle."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as bview])
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
  "Register a GC cleaning action that frees `buffer` on `device-id` IF `holder` is
   still the live owner when it becomes unreachable. Captures only holder-independent
   state (never the DeviceArray or DonatedBuffer) so the action does not pin it.

   Idempotent with explicit `free!` and donation via the shared AtomicBoolean:
   whoever wins `compareAndSet(false,true)` performs (or suppresses) the free exactly
   once. Claiming a donation wins the token's CAS without freeing → the buffer survives
   on the output value and that token Cleaner is disarmed."
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
            view        ;; canonical backend-neutral BufferView; dtype/shape above mirror it for
                        ;; compatibility and are checked at construction, never independently set
            owner       ;; ::owned | ::aliased | ::external
            freed       ;; AtomicBoolean — use-after-free / double-free / donation guard
            base])      ;; for ::aliased: the base DeviceArray it views (else nil). Held as a
                        ;; STRONG ref so reachability of an alias keeps the base — and thus the
                        ;; base's Cleaner-guarded buffer — alive; also the seam ensure-live! reads
                        ;; to reject reads through a base that was freed/consumed.

(defn device-array? [x] (instance? DeviceArray x))

(defn- allocation-ownership
  [owner]
  (case owner
    ::owned :owned
    ::aliased :borrowed
    ::external :external
    (throw (ex-info "unknown DeviceArray ownership" {:owner owner}))))

(defn- default-view
  [buffer device-id dtype shape owner]
  (let [dtype (dtype/canon dtype)
        byte-size (long (or (:byte-size buffer)
                            (* (long (:n-elements buffer)) (dtype/bytes-of (:dtype buffer)))))
        ze? (.startsWith (name device-id) "ze")
        allocation (bview/allocation
                    {:id [:device-array (random-uuid)]
                     :byte-size byte-size
                     :memory-space (if ze? :shared :device)
                     :device device-id
                     :alignment (or (:alignment buffer) 1)
                     :coherence (if ze? :host-coherent :explicit-transfer)
                     :ownership (allocation-ownership owner)})]
    (bview/view allocation {:dtype dtype :shape (vec shape)})))

(defn- validate-device-view!
  [buffer device-id dtype shape view]
  (let [view (bview/validate-view! view)
        dtype (dtype/canon dtype)
        shape (vec shape)
        buffer-dtype (dtype/canon (:dtype buffer))]
    (when-not (= dtype (:dtype view))
      (throw (ex-info "DeviceArray dtype differs from its BufferView"
                      {:dtype dtype :view-dtype (:dtype view)})))
    (when-not (= shape (:shape view))
      (throw (ex-info "DeviceArray shape differs from its BufferView"
                      {:shape shape :view-shape (:shape view)})))
    (when-not (= buffer-dtype dtype)
      (throw (ex-info "DeviceArray cannot reinterpret its backend buffer dtype"
                      {:buffer-dtype buffer-dtype :dtype dtype})))
    (when-not (or (nil? (get-in view [:allocation :device]))
                  (= device-id (get-in view [:allocation :device])))
      (throw (ex-info "DeviceArray device differs from its BufferAllocation"
                      {:device device-id :allocation-device (get-in view [:allocation :device])})))
    view))

(defn- make-device-array
  "Build a DeviceArray over an existing backend buffer, arming the Cleaner for
   ::owned values. Low-level constructor used by `->device` and output projection.
   `base` is non-nil only for ::aliased views (§ alias-of)."
  ([buffer device-id dtype shape owner]
   (make-device-array buffer device-id dtype shape owner nil nil))
  ([buffer device-id dtype shape owner base]
   (make-device-array buffer device-id dtype shape owner base nil))
  ([buffer device-id dtype shape owner base view]
   (let [dtype (dtype/canon dtype)
         shape (vec shape)
         view (validate-device-view! buffer device-id dtype shape
                                     (or view (default-view buffer device-id dtype shape owner)))
         freed (AtomicBoolean. false)
         da    (->DeviceArray buffer device-id dtype shape view owner freed base)]
     (arm-cleaner! da freed buffer device-id (= owner ::owned))
     da)))

;; ================================================================
;; Lifetime guards
;; ================================================================

(defn live?
  "True if the value has not been freed, consumed (donated), or reclaimed."
  [^DeviceArray da]
  (and (not (.get ^AtomicBoolean (:freed da)))
       (if-let [^DeviceArray base (:base da)] (live? base) true)))

(defn- ensure-live!
  [^DeviceArray da op]
  (when (.get ^AtomicBoolean (:freed da))
    (throw (ex-info (str "DeviceArray use-after-free: " op
                         " on a value that was already freed, consumed by donation,"
                         " or reclaimed")
                    {:op op :owner (:owner da) :shape (:shape da) :device (:device da)})))
  ;; An alias chain is only as live as its ultimate allocation owner.
  (when-let [^DeviceArray b (:base da)]
    (ensure-live! b op))
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
     (try
       (make-device-array buffer device-id (:dtype buffer) shape ::owned)
       (catch Throwable e
         ((rt-fn device-id "free-buffer!") buffer)
         (throw e))))))

(defn ->host
  "Download a DeviceArray's exact contiguous view to a fresh JVM array. Throws if the
   value is not live. For :half/:float16 returns the encoded short array."
  [^DeviceArray da]
  (ensure-live! da :->host)
  ;; reachabilityFence: without it the JIT may treat `da` as dead after the (:buffer da) field
  ;; load, letting GC run its Cleaner and free the segment WHILE download-range! copies from it —
  ;; a native use-after-free. Keep `da` (and, via its :base field, any aliased base) reachable
  ;; across the whole ranged native copy.
  (let [view (bview/validate-view! (:view da))
        dt (:dtype view)
        element-bytes (long (dtype/bytes-of dt))
        elements (quot (:byte-length view) element-bytes)
        out (case dt
              :float (float-array elements)
              :double (double-array elements)
              :int (int-array elements)
              :long (long-array elements)
              :half (short-array elements)
              :byte (byte-array elements))]
    (when-not (bview/contiguous? view)
      (throw (ex-info "DeviceArray host transfer requires a contiguous BufferView"
                      {:view (:id view) :shape (:shape view) :strides (:strides view)})))
    (try ((rt-fn (:device da) "download-range!")
          (:buffer da) out {:src-element (quot (:byte-offset view) element-bytes)
                            :elements elements})
         (finally (java.lang.ref.Reference/reachabilityFence da)))))

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
        ;; explicit free surfaces runtime errors (fail-loud) — unlike the Cleaner thread, whose
        ;; reclamation MUST swallow (arm-cleaner!). The CAS guarantees this fires at most once.
        ((rt-fn (:device da) "free-buffer!") (:buffer da)))))
  nil)

(defrecord DonatedBuffer [buffer device view owner claimed])

(defn donated-buffer? [x] (instance? DonatedBuffer x))

(defn consume!
  "Consume a DeviceArray for donation and return a single-use DonatedBuffer token carrying its
   buffer and canonical view. The input becomes dead without freeing storage. Aliases cannot be
   donated independently because their lifetime remains subordinate to a base owner."
  [^DeviceArray da]
  (ensure-live! da :consume!)
  (when (= ::aliased (:owner da))
    (throw (ex-info "an aliased DeviceArray cannot transfer its base allocation ownership"
                    {:view (get-in da [:view :id])})))
  (let [^AtomicBoolean freed (:freed da)]
    (when-not (.compareAndSet freed false true)
      (throw (ex-info "DeviceArray consumed concurrently"
                      {:owner (:owner da) :shape (:shape da)})))
    (let [claimed (AtomicBoolean. false)
          donation (->DonatedBuffer (:buffer da) (:device da) (:view da) (:owner da) claimed)]
      ;; Ownership must remain reclaimable while it is between the consumed input and output.
      (arm-cleaner! donation claimed (:buffer da) (:device da) (= ::owned (:owner da)))
      donation)))

(defn donate-output
  "Claim a DonatedBuffer exactly once and build its output DeviceArray without losing allocation
   identity. The output view must fit within the consumed input view."
  [^DonatedBuffer donation dtype shape]
  (when-not (donated-buffer? donation)
    (throw (ex-info "donate-output requires the token returned by consume!"
                    {:value donation :actual (type donation)})))
  (let [owner (case (:owner donation)
                ::owned ::owned
                ::external ::external
                (throw (ex-info "donation token has a non-transferable owner"
                                {:owner (:owner donation)})))
        dtype (dtype/canon dtype)
        _ (when-not (= dtype (get-in donation [:view :dtype]))
            (throw (ex-info "donation cannot reinterpret its backend buffer dtype"
                            {:dtype dtype :buffer-dtype (get-in donation [:view :dtype])})))
        view (bview/subview (:view donation) {:dtype dtype :shape (vec shape)})
        _ (validate-device-view! (:buffer donation) (:device donation) dtype shape view)]
    (when-not (.compareAndSet ^AtomicBoolean (:claimed donation) false true)
      (throw (ex-info "donated buffer token was already claimed" {})))
    (make-device-array (:buffer donation) (:device donation) dtype shape owner nil view)))

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

(defn wrap-external-view
  "Wrap a caller/session-owned backend buffer using an existing canonical BufferView identity."
  [buffer device-id view]
  (let [view (bview/validate-view! view)]
    (make-device-array buffer device-id (:dtype view) (:shape view) ::external nil view)))

(defn alias-of
  "Build an ::aliased view onto `base` without allocating or copying.

   The map form accepts :byte-offset (relative to base), :shape, :dtype, :strides, and :id.
   Aliases retain the base strongly; freeing or consuming any owner in the chain invalidates
   every descendant."
  ([^DeviceArray base] (alias-of base {}))
  ([^DeviceArray base opts]
   (ensure-live! base :alias-of)
   (let [opts (merge {:dtype (:dtype base) :shape (:shape base)} opts)
         view (bview/subview (:view base) opts)]
     (make-device-array (:buffer base) (:device base) (:dtype view) (:shape view)
                        ::aliased base view))))
