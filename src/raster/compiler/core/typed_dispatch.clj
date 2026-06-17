(ns raster.compiler.core.typed-dispatch
  "Generates typed dispatch classes that implement IFn + all IFn__ interfaces.

  Instead of a plain Clojure fn closure, the dispatch var holds an instance
  of a dynamically generated class that implements:
    - clojure.lang.IFn (for Clojure interop, dispatch with boxing)
    - Every IFn__ interface from the dispatch table (typed fast path)

  This eliminates the instanceof-fails-to-slow-path problem. When a dispatch
  var is passed as a (Fn [...]) parameter, the instanceof check on the IFn__
  interface SUCCEEDS, routing to the typed .invk path. C2 profiles the
  monomorphic receiver → 1-cycle class-word guard → inlines the .invk body."
  (:require [clojure.string :as str]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as me])
  (:import [java.lang.classfile ClassFile ClassBuilder CodeBuilder]
           [java.lang.constant ClassDesc MethodTypeDesc]))

(def ^:private obj-cd (ClassDesc/of "java.lang" "Object"))
(def ^:private V-cd   (ClassDesc/ofDescriptor "V"))
(def ^:private D-cd   (ClassDesc/ofDescriptor "D"))
(def ^:private J-cd   (ClassDesc/ofDescriptor "J"))
(def ^:private F-cd   (ClassDesc/ofDescriptor "F"))
(def ^:private I-cd   (ClassDesc/ofDescriptor "I"))
(def ^:private afn-cd (ClassDesc/of "clojure.lang" "AFn"))
(def ^:private ifn-cd (ClassDesc/of "clojure.lang" "IFn"))
(def ^:private iseq-cd (ClassDesc/of "clojure.lang" "ISeq"))

(def ^:private class-counter (atom 0))

(defn- iface-tag->cd [iface-tag]
  (let [s (str iface-tag)
        dot (.lastIndexOf s (int \.))]
    (if (pos? dot)
      (ClassDesc/of (subs s 0 dot) (subs s (inc dot)))
      (ClassDesc/of "" s))))

(defn- param-tag->cd [tag]
  (case tag
    (double Double) D-cd
    (long Long)     J-cd
    (float Float)   F-cd
    (int Integer)   I-cd
    doubles         (ClassDesc/ofDescriptor "[D")
    floats          (ClassDesc/ofDescriptor "[F")
    longs           (ClassDesc/ofDescriptor "[J")
    ints            (ClassDesc/ofDescriptor "[I")
    objects         (ClassDesc/ofDescriptor "[Ljava/lang/Object;")
    obj-cd))

(defn- emit-load-params [^CodeBuilder code p-cds]
  (loop [slot 1, i 0]
    (when (< i (count p-cds))
      (let [pd (.descriptorString ^ClassDesc (nth p-cds i))]
        (cond
          (= pd "D") (do (.dload code slot) (recur (+ slot 2) (inc i)))
          (= pd "J") (do (.lload code slot) (recur (+ slot 2) (inc i)))
          (= pd "F") (do (.fload code slot) (recur (+ slot 1) (inc i)))
          (#{"I" "Z" "B" "S" "C"} pd) (do (.iload code slot) (recur (+ slot 1) (inc i)))
          :else (do (.aload code slot) (recur (+ slot 1) (inc i))))))))

(defn make-typed-dispatch
  "Generate a typed dispatch instance implementing IFn + all IFn__ interfaces.
  Each .invk method loads the corresponding impl field and delegates.
  IFn.invoke delegates to the stored dispatch-fn."
  [fn-name table-atom dispatch-fn]
  (let [table @table-atom
        ;; Collect entries with typed interfaces
        typed-entries (vec (for [[_arity methods] table
                                 m methods
                                 :when (:typed-iface m)]
                             m))
        ;; Deduplicate by interface name (multiple overloads may share an interface
        ;; if they have the same param types after normalization)
        iface-infos (->> typed-entries
                         (map (fn [e] {:iface-tag (:typed-iface e)
                                       :tags (:tags e)
                                       :typed-impl (:typed-impl e)}))
                         (reduce (fn [acc info]
                                   (if (some #(= (:iface-tag %) (:iface-tag info)) acc)
                                     acc
                                     (conj acc info)))
                                 []))]
    (if (empty? iface-infos)
      ;; No typed interfaces — return plain dispatch fn
      dispatch-fn
      ;; Generate typed dispatch class
      (let [n (swap! class-counter inc)
            cname (str "raster.dispatch.D_"
                       (str/replace (str fn-name) #"[^a-zA-Z0-9_]" "_")
                       "_" n)
            dot (.lastIndexOf cname (int \.))
            dispatch-cd (ClassDesc/of (subs cname 0 dot) (subs cname (inc dot)))
            iface-cds (mapv #(iface-tag->cd (:iface-tag %)) iface-infos)
            field-names (mapv #(str "impl_" %) (range (count iface-infos)))
            cf (ClassFile/of)
            bytes
            (.build cf dispatch-cd
                    (reify java.util.function.Consumer
                      (accept [_ cb]
                        (let [^ClassBuilder cb cb]
                          (.withFlags cb (int 0x0011)) ;; PUBLIC FINAL
                          (.withSuperclass cb afn-cd)
                          (.withInterfaceSymbols cb (into-array ClassDesc iface-cds))
                  ;; Fields: one per overload impl + dispatchFn
                          (doseq [[fname icd] (map vector field-names iface-cds)]
                            (.withField cb fname icd (int 0x0001)))
                          (.withField cb "dispatchFn" ifn-cd (int 0x0001))
                  ;; Constructor
                          (.withMethodBody cb "<init>"
                                           (MethodTypeDesc/of V-cd (into-array ClassDesc [ifn-cd]))
                                           (int 0x0001)
                                           (reify java.util.function.Consumer
                                             (accept [_ code]
                                               (let [^CodeBuilder code code]
                                                 (.aload code 0)
                                                 (.invokespecial code afn-cd "<init>"
                                                                 (MethodTypeDesc/of V-cd (into-array ClassDesc [])))
                                                 (.aload code 0)
                                                 (.aload code 1)
                                                 (.putfield code dispatch-cd "dispatchFn" ifn-cd)
                                                 (.return_ code)))))
                  ;; .invk methods: one per unique method descriptor.
                  ;; Multiple IFn__ interfaces may share the same invk signature
                  ;; (e.g. IFn__Object_long and IFn__String_long both have invk(Object,long)→Object).
                  ;; Emit each descriptor only once; first matching entry wins.
                          (let [emitted-descs (atom #{})]
                            (doseq [[{:keys [tags]} icd fname]
                                    (map vector iface-infos iface-cds field-names)]
                              (let [p-cds (mapv param-tag->cd tags)
                                    invk-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc p-cds))
                                    desc (.descriptorString invk-mt)]
                                (when-not (contains? @emitted-descs desc)
                                  (swap! emitted-descs conj desc)
                                  (.withMethodBody cb "invk" invk-mt (int 0x0001)
                                                   (reify java.util.function.Consumer
                                                     (accept [_ code]
                                                       (let [^CodeBuilder code code]
                                                         (.aload code 0)
                                                         (.getfield code dispatch-cd fname icd)
                                                         (emit-load-params code p-cds)
                                                         (.invokeinterface code icd "invk" invk-mt)
                                                         (.areturn code)))))))))
                  ;; IFn.invoke overloads 0-20: delegate all to dispatchFn.
                  ;; Must generate all arities because Clojure variadic fns
                  ;; (e.g. (- a b c)) use direct invoke(3), not applyTo.
                  ;; The dispatch fn handles variadic reduction internally.
                          (doseq [arity (range 0 21)]
                            (let [mt (MethodTypeDesc/of obj-cd (into-array ClassDesc (repeat arity obj-cd)))]
                              (.withMethodBody cb "invoke" mt (int 0x0001)
                                               (reify java.util.function.Consumer
                                                 (accept [_ code]
                                                   (let [^CodeBuilder code code]
                                                     (.aload code 0)
                                                     (.getfield code dispatch-cd "dispatchFn" ifn-cd)
                                                     (dotimes [i arity] (.aload code (inc i)))
                                                     (.invokeinterface code ifn-cd "invoke" mt)
                                                     (.areturn code)))))))
                  ;; applyTo: delegate to dispatchFn.applyTo(seq)
                  ;; This preserves variadic/reduce semantics from the dispatch fn
                          (let [apply-to-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc [iseq-cd]))]
                            (.withMethodBody cb "applyTo" apply-to-mt (int 0x0001)
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                                                 (let [^CodeBuilder code code]
                                                   (.aload code 0)
                                                   (.getfield code dispatch-cd "dispatchFn" ifn-cd)
                                                   (.aload code 1)
                                                   (.invokeinterface code ifn-cd "applyTo" apply-to-mt)
                                                   (.areturn code))))))))))
            cl (or me/*compilation-classloader*
                   (clojure.lang.DynamicClassLoader.
                    (.getContextClassLoader (Thread/currentThread))))
            dispatch-class (.defineClass ^clojure.lang.DynamicClassLoader cl cname bytes nil)
            ctor (.getDeclaredConstructor dispatch-class (into-array Class [clojure.lang.IFn]))
            instance (.newInstance ctor (into-array Object [dispatch-fn]))]
        ;; Set impl fields. parametric-specialize! registers the impl *var* (so a
        ;; later recompile is visible) while regular specialization registers the
        ;; impl instance directly — deref the var here so the typed field holds an
        ;; IFn__ instance either way. Without this, the parametric path (e.g. float
        ;; (All [T]) kernels) fails the field set ("can not set IFn__… to Var"),
        ;; silently falls back to untyped dispatch, and runs ~30x slower interpreted.
        (doseq [[{:keys [typed-impl]} fname] (map vector iface-infos field-names)]
          (when-let [ti (if (var? typed-impl) (deref typed-impl) typed-impl)]
            (let [f (.getDeclaredField dispatch-class fname)]
              (.setAccessible f true)
              (.set f instance ti))))
        instance))))

(defn update-dispatch-impl!
  "Update impl fields on the dispatch object after JIT upgrade."
  [dispatch-obj old-impl new-impl]
  (let [cls (class dispatch-obj)]
    (doseq [f (.getDeclaredFields cls)]
      (when (not= (.getName f) "dispatchFn")
        (try
          (.setAccessible f true)
          (when (identical? (.get f dispatch-obj) old-impl)
            (.set f dispatch-obj new-impl))
          (catch Exception _ nil))))))
