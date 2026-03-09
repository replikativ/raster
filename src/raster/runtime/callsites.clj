(ns raster.runtime.callsites
  "MutableCallSite registry for invokedynamic dispatch.

  Each .invk call on a deftm -impl var is linked via invokedynamic to a
  MutableCallSite. The JIT treats the callsite target as a compile-time
  constant and can inline through it — zero runtime check on the hot path.

  On REPL redefinition, registered callsites are invalidated via
  setTarget() + syncAll(), triggering C2 deoptimization and recompilation
  with the new target. This matches Julia's backedge invalidation model."
  (:import [java.lang.invoke MutableCallSite MethodHandle MethodHandles
            MethodHandles$Lookup MethodType CallSite]
           [java.lang.ref WeakReference]
           [clojure.lang Var RT]))

;; Registry: {Var → [{:site MutableCallSite, :method-type MethodType} ...]}
(defonce ^:private registry (atom {}))

(defn register-callsite!
  "Register a MutableCallSite for invalidation tracking.
  Uses WeakReference to avoid retaining GC'd classes during REPL development."
  [^Var impl-var ^MutableCallSite site ^MethodType mt]
  (swap! registry update impl-var (fnil conj [])
         {:site-ref (WeakReference. site) :method-type mt}))

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

(defn bootstrap
  "Bootstrap method for invokedynamic call sites.
  Called by the JVM the first time each invokedynamic instruction is executed.

  name:     invocation name (unused, e.g. \"invoke\")
  type:     the call site's method type (matches the .invk signature)
  ns-name:  namespace of the -impl var (passed as extra bootstrap arg)
  var-name: name of the -impl var (passed as extra bootstrap arg)

  JVM member names cannot contain dots or slashes, so ns-name and var-name
  are passed as separate ConstantDesc bootstrap arguments."
  [^MethodHandles$Lookup lookup ^String name ^MethodType type
   ^String ns-name ^String var-name]
  (let [impl-var (RT/var ns-name var-name)
        impl (.getRawRoot impl-var)
        target (create-target-handle impl type)
        site (MutableCallSite. ^MethodHandle target)]
    (register-callsite! impl-var site type)
    site))

(defn invalidate!
  "Invalidate all callsites registered for the given impl var.
  Called when the var's value changes (lazy JIT, REPL redef).

  new-impl: the new impl instance (deftype or BC-compiled)

  After this call, C2 will deoptimize any compiled code that inlined
  through the old callsite target, and recompile with the new target
  on next invocation. One-time cost: ~100µs."
  [^Var impl-var new-impl]
  (when-let [entries (get @registry impl-var)]
    (let [sites (java.util.ArrayList.)
          live-entries (java.util.ArrayList.)]
      (doseq [{:keys [^WeakReference site-ref ^MethodType method-type]} entries]
        (when-let [^MutableCallSite site (.get site-ref)]
          (try
            (let [new-handle (create-target-handle new-impl method-type)]
              (.setTarget site ^MethodHandle new-handle)
              (.add sites site)
              (.add live-entries {:site-ref site-ref :method-type method-type}))
            (catch Exception e
              (binding [*out* *err*]
                (println (str "WARNING: Stale invokedynamic callsite for " impl-var
                              " — signature changed. Re-eval calling namespace. "
                              (.getMessage e))))))))
      ;; Prune GC'd and stale entries from registry
      (swap! registry assoc impl-var (vec live-entries))
      (when (pos? (.size sites))
        (MutableCallSite/syncAll (into-array MutableCallSite sites))))))
