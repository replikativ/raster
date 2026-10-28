(ns raster.runtime.callsites
  "MutableCallSite registry for invokedynamic dispatch.

  Each .invk call on a deftm -impl var is linked via invokedynamic to a
  MutableCallSite. The JIT treats the callsite target as a compile-time
  constant and can inline through it — zero runtime check on the hot path.

  On REPL redefinition, registered callsites are invalidated via
  setTarget() + syncAll(), triggering C2 deoptimization and recompilation
  with the new target. A shared root watch covers normal notifying Var root
  mutations, including restoration after with-redefs. Explicit invalidation
  remains necessary for unbindRoot, non-notifying/throwing mutation paths, or
  removed watches. Already executing invocations are not cancelled; this is
  not a concurrent world-age guarantee.

  Start a fresh process after upgrading: the defonce registry's type changed."
  (:import [java.lang.invoke MutableCallSite MethodHandle MethodHandles
            MethodHandles$Lookup MethodType]
           [java.lang.ref WeakReference]
           [java.util WeakHashMap]
           [clojure.lang Var RT]))

;; Weak keys must not be reachable through their values. In particular, neither
;; the Var nor a MethodType (which can retain generated classes) is stored in an
;; entry. The shared watch below also captures no watched Var.
;; Lock order is always Var -> registry. Registry critical sections only prune
;; and update references, never build handles, call watches, or publish targets.
(defonce ^:private registry (WeakHashMap.))

(declare invalidate!)

(defn- root-changed! [_key reference _old-root new-root]
  (invalidate! reference new-root))

(defn- register-reference! [^Var impl-var ^MutableCallSite site]
  (locking registry
    (.put registry impl-var
          (conj (into [] (filter #(.get ^WeakReference %)) (.get registry impl-var))
                (WeakReference. site)))))

(defn- live-sites! [^Var impl-var]
  (locking registry
    (let [[references sites]
          (reduce (fn [[references sites] ^WeakReference reference]
                    (if-let [site (.get reference)]
                      ;; One read both prunes the reference and holds this site
                      ;; strongly while publishing outside the registry lock.
                      [(conj references reference) (conj sites site)]
                      [references sites]))
                  [[] []] (.get registry impl-var))]
      (if (seq references)
        (.put registry impl-var references)
        (.remove registry impl-var))
      sites)))

(defn create-target-handle
  "Create a MethodHandle targeting the impl's compute_prim (zero-boxing) or invk method.
  When the callsite has a primitive return type, tries compute_prim first — this method
  returns the primitive directly without boxing through Object. Falls back to invk
  with asType adaptation for pre-JIT deftype stubs that lack compute_prim."
  [impl ^MethodType mt]
  (let [cls (class impl)
        lookup (MethodHandles/lookup)]
    ;; Try compute_prim first — same params as invk but primitive return (no boxing)
    (if-let [prim-handle
             (try (.bindTo (.findVirtual lookup cls "compute_prim" mt) impl)
                  (catch NoSuchMethodException _ nil))]
      prim-handle
      ;; Fallback: use invk (Object return) with asType adaptation
      (let [invk-mt (MethodType/methodType Object (.parameterArray mt))
            invk-handle (.findVirtual lookup cls "invk" invk-mt)
            bound (.bindTo invk-handle impl)]
        (.asType bound mt)))))

(defn- relink-failure-handle [^MethodType mt cause]
  (let [failure (ex-info "typed invokedynamic callsite could not link its current root"
                         {:reason :typed-callsite-relink} cause)
        throwing (MethodHandles/throwException (.returnType mt) clojure.lang.ExceptionInfo)
        bound (.bindTo ^MethodHandle throwing failure)]
    (MethodHandles/dropArguments bound (int 0) (.parameterArray mt))))

(defn- relink-site! [^MutableCallSite site impl]
  (let [mt (.type site)]
    (try
      (.setTarget site ^MethodHandle (create-target-handle impl mt))
      (catch Exception cause
        (.setTarget site ^MethodHandle (relink-failure-handle mt cause)))
      (catch LinkageError cause
        (.setTarget site ^MethodHandle (relink-failure-handle mt cause))))))

(defn register-callsite!
  "Register and link a site to the CURRENT raw root, not its initial target.
   Normal notifying root mutations serialize on the same Var monitor. Failed
   links install an exact-type throwing target and remain registered for repair."
  [^Var impl-var ^MutableCallSite site ^MethodType mt]
  (when-not (and (instance? Var impl-var) (instance? MutableCallSite site)
                 (instance? MethodType mt) (= mt (.type site)))
    (throw (ex-info "typed callsite registration requires its exact MethodType"
                    {:reason :typed-callsite-registration})))
  (locking impl-var
    ;; Installing the shared watch is outside the registry lock. It retains no
    ;; reference to impl-var beyond Var's own watch table.
    (add-watch impl-var ::root-change root-changed!)
    (register-reference! impl-var site)
    (relink-site! site (.getRawRoot impl-var))
    (MutableCallSite/syncAll (into-array MutableCallSite [site])))
  site)

(defn bootstrap
  "Bootstrap method for invokedynamic call sites.
  Called by the JVM the first time each invokedynamic instruction is executed.

  name:     invocation name (unused, e.g. \"invoke\")
  type:     the call site's method type (matches the .invk signature)
  ns-name:  namespace of the -impl var (passed as extra bootstrap arg)
  var-name: name of the -impl var (passed as extra bootstrap arg)

  JVM member names cannot contain dots or slashes, so ns-name and var-name
  are passed as separate ConstantDesc bootstrap arguments."
  [^MethodHandles$Lookup _lookup ^String _name ^MethodType type
   ^String ns-name ^String var-name]
  (let [impl-var (RT/var ns-name var-name)
        site (MutableCallSite. type)]
    ;; Do not let an invalid initial root become a sticky BootstrapMethodError.
    ;; Registration reads and links the root atomically with watch installation.
    (register-callsite! impl-var site type)
    site))

(defn invalidate!
  "Relink every live registered site from the Var's current raw root.
   The notification argument is advisory only: a delayed notification cannot
   reinstall an older root. Failed sites are synchronized and retained so a
   later valid root can repair them. Core's explicit lazy-JIT invalidation stays."
  [^Var impl-var _new-impl]
  (locking impl-var
    (let [sites (live-sites! impl-var)
          impl (.getRawRoot impl-var)]
      (doseq [site sites] (relink-site! site impl))
      (when (seq sites)
        (MutableCallSite/syncAll (into-array MutableCallSite sites))))))
