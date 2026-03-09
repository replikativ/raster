(ns raster.compiler.backend.jvm.bytecode
  "Bytecode specialization — compiles devirtualized Clojure forms to JVM
  static methods, eliminating IFn.invoke overhead entirely.

  Architecture: macroexpand-all first, then handle only Clojure's ~12
  special forms + function calls.  Same approach as partial-cps/ioc.clj.
  This generalizes to arbitrary Clojure code in deftm bodies."
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.core.method-entry :as me]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.util :as util]
            [raster.compiler.backend.jvm.split :as split])
  (:import (java.lang.classfile ClassFile ClassFile$ClassHierarchyResolverOption ClassFile$Option)
           (java.lang.classfile.attribute SourceFileAttribute)
           (java.lang.constant ClassDesc MethodTypeDesc
                               DynamicCallSiteDesc DirectMethodHandleDesc$Kind
                               MethodHandleDesc ConstantDesc)))

;; ================================================================
;; ClassFile with Clojure-aware class hierarchy resolver
;; ================================================================

(defn- compilation-loader
  "Get classloader for defining compiled classes.
  Uses *compilation-classloader* when bound (from compile-aot),
  otherwise creates a fresh DCL from thread context CL."
  []
  (or me/*compilation-classloader*
      (clojure.lang.DynamicClassLoader.
       (.getContextClassLoader (Thread/currentThread)))))

(defn- clojure-classfile
  "Create a ClassFile builder that can resolve Clojure DynamicClassLoader classes.
  When the CL can't find a class (e.g., defrecord loaded by a different DCL),
  falls back to treating it as extending Object (conservative: enables virtual
  dispatch instead of interface dispatch, preventing AbstractMethodError)."
  []
  (let [cl (compilation-loader)
        base (java.lang.classfile.ClassHierarchyResolver/ofClassLoading cl)
        ;; Fallback: treat unresolvable classes as extending Object.
        ;; Without this, the ClassFile API may treat them as interfaces,
        ;; generating invokeinterface instead of invokevirtual for methods
        ;; like meta() — causing AbstractMethodError at runtime.
        obj-info (java.lang.classfile.ClassHierarchyResolver$ClassHierarchyInfo/ofClass
                  (java.lang.constant.ClassDesc/of "java.lang" "Object"))
        safe (reify java.lang.classfile.ClassHierarchyResolver
               (getClassInfo [_ cd]
                 (try (.getClassInfo base cd)
                      (catch Throwable _ obj-info))))]
    (ClassFile/of (into-array ClassFile$Option [(ClassFile$ClassHierarchyResolverOption/of safe)]))))

;; ================================================================
;; ClassDesc constants
;; ================================================================

(def ^:private obj-cd   (ClassDesc/ofDescriptor "Ljava/lang/Object;"))
(def ^:private objarr-cd (ClassDesc/ofDescriptor "[Ljava/lang/Object;"))
(def ^:private dblarr-cd (ClassDesc/ofDescriptor "[D"))
(def ^:private number-cd (ClassDesc/ofDescriptor "Ljava/lang/Number;"))
(def ^:private math-cd   (ClassDesc/ofDescriptor "Ljava/lang/Math;"))
(def ^:private D-cd (ClassDesc/ofDescriptor "D"))
(def ^:private I-cd (ClassDesc/ofDescriptor "I"))
(def ^:private J-cd (ClassDesc/ofDescriptor "J"))
(def ^:private V-cd (ClassDesc/ofDescriptor "V"))
(def ^:private no-cd (into-array ClassDesc []))
(def ^:private F-cd (ClassDesc/ofDescriptor "F"))
(def ^:private fltarr-cd (ClassDesc/ofDescriptor "[F"))
(def ^:private lngarr-cd (ClassDesc/ofDescriptor "[J"))
(def ^:private intarr-cd (ClassDesc/ofDescriptor "[I"))
(def ^:private flt-box-cd (ClassDesc/ofDescriptor "Ljava/lang/Float;"))
(def ^:private int-box-cd  (ClassDesc/ofDescriptor "Ljava/lang/Integer;"))
(def ^:private long-box-cd (ClassDesc/ofDescriptor "Ljava/lang/Long;"))
(def ^:private dbl-box-cd  (ClassDesc/ofDescriptor "Ljava/lang/Double;"))
(def ^:private ifn-cd (ClassDesc/ofDescriptor "Lclojure/lang/IFn;"))
(def ^:private var-cd (ClassDesc/ofDescriptor "Lclojure/lang/Var;"))
(def ^:private rt-cd  (ClassDesc/ofDescriptor "Lclojure/lang/RT;"))

;; ================================================================
;; Stack type predicates — centralized instead of inline #{} literals
;; ================================================================

(defn- two-slot?
  "True if this stack type occupies two JVM stack/local slots (double, long)."
  [t] (or (identical? t :double) (identical? t :long)))

(defn- primitive?
  "True if this stack type is a JVM primitive (not :ref, :void, or :diverge)."
  [t] (case t (:double :float :int :long) true false))

(defn- diverge?
  "True if this form never completes normally (throw, recur).
  Distinguished from :void which means 'no value but execution continues'."
  [t] (identical? t :diverge))

(defn- typed-interface?
  "True if this symbol/string names a Raster typed function interface (IFn__)."
  [s] (and s (.contains (str s) "IFn__")))

;; ================================================================
;; invokedynamic bootstrap (MutableCallSite dispatch)
;; ================================================================

(def ^:private string-cd  (ClassDesc/ofDescriptor "Ljava/lang/String;"))
(def ^:private mhlookup-cd (ClassDesc/ofDescriptor "Ljava/lang/invoke/MethodHandles$Lookup;"))
(def ^:private methodtype-cd (ClassDesc/ofDescriptor "Ljava/lang/invoke/MethodType;"))
(def ^:private callsite-cd (ClassDesc/ofDescriptor "Ljava/lang/invoke/CallSite;"))
(def ^:private bootstrap-cd (ClassDesc/of "raster.runtime.Bootstrap"))

;; Bootstrap method type: (Lookup, String, MethodType, String, String) -> CallSite
(def ^:private bsm-mt
  (MethodTypeDesc/of callsite-cd
                     (into-array ClassDesc [mhlookup-cd string-cd methodtype-cd string-cd string-cd])))

;; DirectMethodHandleDesc pointing to Bootstrap.bsm
(def ^:private bsm-handle
  (MethodHandleDesc/ofMethod
   DirectMethodHandleDesc$Kind/STATIC
   bootstrap-cd
   "bsm"
   bsm-mt))

(defonce ^:private bootstrap-class-loaded? (atom false))

(defn- ensure-bootstrap-class!
  "Generate and load the raster.runtime.Bootstrap class if not already loaded.
  Contains a static bsm method that delegates to callsites/bootstrap."
  []
  (when-not @bootstrap-class-loaded?
    (require 'raster.runtime.callsites)
    (let [bytes (.build (clojure-classfile)
                        bootstrap-cd
                        (reify java.util.function.Consumer
                          (accept [_ cb]
                            (.withSuperclass cb obj-cd)
                      ;; public static bsm(Lookup, String, MethodType, String, String) -> CallSite
                            (.withMethodBody cb "bsm" bsm-mt
                                             (int 0x0009) ;; ACC_PUBLIC | ACC_STATIC
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                            ;; Get the Clojure bootstrap fn:
                            ;; RT.var("raster.runtime.callsites", "bootstrap").getRawRoot()
                                                 (.ldc code "raster.runtime.callsites")
                                                 (.ldc code "bootstrap")
                                                 (.invokestatic code rt-cd "var"
                                                                (MethodTypeDesc/of var-cd (into-array ClassDesc [string-cd string-cd])))
                                                 (.invokevirtual code var-cd "getRawRoot"
                                                                 (MethodTypeDesc/of obj-cd no-cd))
                                                 (.checkcast code ifn-cd)
                            ;; Push all 5 args: lookup, name, type, nsName, varName
                                                 (.aload code 0)
                                                 (.aload code 1)
                                                 (.aload code 2)
                                                 (.aload code 3)
                                                 (.aload code 4)
                            ;; IFn.invoke(Object, Object, Object, Object, Object) -> Object
                                                 (.invokeinterface code ifn-cd "invoke"
                                                                   (MethodTypeDesc/of obj-cd
                                                                                      (into-array ClassDesc [obj-cd obj-cd obj-cd obj-cd obj-cd])))
                                                 (.checkcast code callsite-cd)
                                                 (.areturn code)))))))
          loader (compilation-loader)]
      (.defineClass loader "raster.runtime.Bootstrap" bytes nil)
      (reset! bootstrap-class-loaded? true))))

;; ================================================================
;; Helpers
;; ================================================================

(defn- class-desc-of
  "Create a ClassDesc from a Class object."
  ^ClassDesc [^Class c]
  (cond
    (.isPrimitive c)
    (ClassDesc/ofDescriptor
     (str ({Double/TYPE "D" Long/TYPE "J" Integer/TYPE "I"
            Float/TYPE "F" Void/TYPE "V" Boolean/TYPE "Z"
            Byte/TYPE "B" Short/TYPE "S" Character/TYPE "C"} c)))
    (.isArray c)
    (ClassDesc/ofDescriptor
     (str "[" (.descriptorString (class-desc-of (.getComponentType c)))))
    :else
    (ClassDesc/ofDescriptor
     (str "L" (.replace (.getName c) "." "/") ";"))))

(defn- class-type->stack
  "Map a Java Class to its stack type keyword."
  [^Class c]
  (cond (.equals c Double/TYPE) :double
        (.equals c Long/TYPE) :long
        (.equals c Integer/TYPE) :int
        (.equals c Float/TYPE) :float
        ;; boolean, byte, short, char are all int at JVM bytecode level
        (.equals c Boolean/TYPE) :int
        (.equals c Byte/TYPE) :int
        (.equals c Short/TYPE) :int
        (.equals c Character/TYPE) :int
        (.equals c Void/TYPE) :void
        :else :ref))

(defn- infer-arg-stack-type
  "Infer the stack type of a form WITHOUT emitting bytecode.
  Used to select the correct overloaded method (e.g. Math.abs(int) vs Math.abs(double))."
  [form locals]
  (cond
    (nil? form) :ref
    (true? form) :int
    (false? form) :int
    (float? form) :double
    (integer? form) :int
    (symbol? form) (or (when-let [{:keys [type]} (get locals form)] type)
                       ;; Fallback: read walker/AD-stamped type tag
                       (when-let [tag (or (:raster.type/tag (meta form))
                                          (:tag (meta form)))]
                         (get {'double :double 'long :long 'int :int
                               'float :float 'doubles :ref 'longs :ref
                               'floats :ref 'ints :ref} tag)))
    (seq? form)
    (let [h (first form)]
      (cond
        (not (symbol? h)) nil
        ;; Cast expressions
        (= (name h) "double") :double
        (= (name h) "long")   :long
        (= (name h) "int")    :int
        (= (name h) "float")  :float
        ;; Arithmetic: float×float→float, otherwise→double
        ;; (matches emit-arithmetic behavior)
        (contains? #{"+" "-" "*" "/"} (name h))
        (let [args (rest form)]
          (if (every? #(= :float (infer-arg-stack-type % locals)) args)
            :float
            :double))
        ;; Comparison ops produce :ref (Boolean.TRUE/FALSE from emit-comparison)
        (contains? #{"<" "<=" ">" ">=" "==" "not="} (name h)) :ref
        ;; if → infer from branches (enables skip-box for nested ifs)
        ;; Only returns a primitive type when BOTH branches are known
        ;; to produce the same primitive. Conservative: returns nil when
        ;; either branch is unknown, forcing the caller to box.
        (= (name h) "if")
        (let [[_ _pred then else] form
              t1 (infer-arg-stack-type then locals)
              t2 (infer-arg-stack-type else locals)]
          (when (and t1 t2 (= t1 t2) (primitive? t1)) t1))
        ;; do → type of last form
        (= (name h) "do")
        (when-let [last-form (last (rest form))]
          (infer-arg-stack-type last-form locals))
        ;; let/let* → type of last body form
        (contains? #{"let" "let*"} (name h))
        (when-let [last-form (last (nnext form))]
          (infer-arg-stack-type last-form locals))
        ;; inc/dec → preserves input type
        (contains? #{"inc" "dec" "unchecked-inc" "unchecked-dec"} (name h))
        (or (infer-arg-stack-type (second form) locals) :long)
        ;; min/max → preserves first arg type
        (contains? #{"min" "max"} (name h))
        (or (infer-arg-stack-type (second form) locals) :double)
        ;; Unchecked int ops
        (contains? #{"unchecked-add-int" "unchecked-inc-int" "unchecked-dec-int"
                     "unchecked-subtract-int" "unchecked-multiply-int"} (name h)) :int
        ;; .invk: return type from walker metadata on the impl symbol.
        ;; Handles both pre-expand (.invk impl args) and post-expand (. impl invk args).
        (or (= (str h) ".invk")
            (and (= (str h) ".") (= 'invk (nth form 2 nil))))
        (let [impl-sym (if (= (str h) ".") (second form) (second form))]
          (when (symbol? impl-sym)
            (let [ret (or (:raster.type/ret-tag (meta impl-sym))
                          (try (when-let [v (resolve impl-sym)]
                                 (:raster.core/return-tag (meta v)))
                               (catch Exception _ nil)))]
              (case ret
                (double Double) :double
                (long Long) :long
                (float Float) :float
                (int Integer) :int
                nil))))
        ;; Math static methods — infer from the first arg type
        ;; (Math/abs, Math/sqrt, Math/fma all preserve arg type)
        (let [s (str h)] (and (.contains s "/") (.startsWith s "Math")))
        (or (infer-arg-stack-type (second form) locals) :double)
        ;; deftm calls: infer return type from var metadata
        (namespace h)
        (try (when-let [v (ns-resolve *ns* h)]
               (when (var? v)
                 (when-let [rt (:raster.core/return-tag (meta v))]
                   (case rt
                     Double :double Long :long Float :float
                     (double) :double (long) :long (float) :float (int) :int
                     nil))))
             (catch Exception _ nil))
        ;; Default: unknown
        :else nil))
    :else nil))

(defn- resolve-static-call
  "Given a deftm walked body that is a single static method call,
  extract class, method name, and method descriptor via reflection."
  [body params tags return-tag]
  (when (and (= 1 (count body))
             (seq? (first body))
             (symbol? (ffirst body))
             (let [s (str (ffirst body))]
               (and (.contains s "/") (not (.startsWith s ".")))))
    (let [s (str (ffirst body))
          slash-idx (.indexOf s "/")
          cls-short (subs s 0 slash-idx)
          meth-name (subs s (inc slash-idx))
          call-args (rest (first body))
          cls-class (or (try (Class/forName cls-short) (catch Exception _ nil))
                        (when-let [resolved (ns-resolve *ns* (symbol cls-short))]
                          (when (class? resolved) resolved)))
          n-args (count call-args)
          candidates (when cls-class
                       (filterv #(and (= (.getName ^java.lang.reflect.Method %) meth-name)
                                      (= (.getParameterCount ^java.lang.reflect.Method %) n-args))
                                (.getMethods cls-class)))
          ;; Match overload by deftm parameter types
          tag->class (fn [t] (case t (double Double) Double/TYPE (long Long) Long/TYPE
                                   (int Integer) Integer/TYPE (float Float) Float/TYPE nil))
          method (when (seq candidates)
                   (if (= 1 (count candidates))
                     (first candidates)
                     (or (first (filter (fn [^java.lang.reflect.Method m]
                                          (let [ptypes (.getParameterTypes m)]
                                            (every? true?
                                                    (map (fn [^Class pt tag]
                                                           (if-let [tc (tag->class tag)]
                                                             (= pt tc)
                                                             true))
                                                         ptypes tags))))
                                        candidates))
                         (first candidates))))]
      (when method
        {:class (class-desc-of cls-class)
         :method meth-name
         :method-type (MethodTypeDesc/of (class-desc-of (.getReturnType method))
                                         (into-array ClassDesc (mapv class-desc-of (.getParameterTypes method))))
         :deftm-params params
         :deftm-call-args call-args
         :java-param-types (vec (.getParameterTypes method))}))))

;; ================================================================
;; Bytecode coercion
;; ================================================================

(defn- emit-unbox-double
  "Emit unbox from :ref to double via Number.doubleValue()."
  [code]
  (.checkcast code number-cd)
  (.invokevirtual code number-cd "doubleValue" (MethodTypeDesc/of D-cd no-cd)))

(defn- emit-unbox-to-prim
  "Unbox Object on stack to a primitive type via Number.xxxValue()."
  [code target-type]
  (case target-type
    :double (emit-unbox-double code)
    :float  (do (.checkcast code number-cd)
                (.invokevirtual code number-cd "floatValue" (MethodTypeDesc/of F-cd no-cd)))
    :long   (do (.checkcast code number-cd)
                (.invokevirtual code number-cd "longValue" (MethodTypeDesc/of J-cd no-cd)))
    :int    (do (.checkcast code number-cd)
                (.invokevirtual code number-cd "intValue" (MethodTypeDesc/of I-cd no-cd)))))

(defn- emit-box-to-ref
  "Box a primitive stack value to Object. Returns :ref."
  [code stack-type]
  (case stack-type
    :double (do (.invokestatic code dbl-box-cd "valueOf"
                               (MethodTypeDesc/of dbl-box-cd (into-array ClassDesc [D-cd])))
                :ref)
    :float  (do (.invokestatic code flt-box-cd "valueOf"
                               (MethodTypeDesc/of flt-box-cd (into-array ClassDesc [F-cd])))
                :ref)
    :int    (do (.invokestatic code int-box-cd "valueOf"
                               (MethodTypeDesc/of int-box-cd (into-array ClassDesc [I-cd])))
                :ref)
    :long   (do (.invokestatic code long-box-cd "valueOf"
                               (MethodTypeDesc/of long-box-cd (into-array ClassDesc [J-cd])))
                :ref)
    :ref     :ref
    :void    :void
    :diverge :diverge))

(defn- emit-coerce
  "Emit type coercion bytecode. Returns the target type."
  [code from-type to-type]
  (case [from-type to-type]
    [:ref :double]  (do ;; Cast to Number (throws CCE for non-Number) and unbox
                      (.checkcast code number-cd)
                      (.invokevirtual code number-cd "doubleValue" (MethodTypeDesc/of D-cd no-cd))
                      :double)
    [:int :double]  (do (.i2d code) :double)
    [:long :double] (do (.l2d code) :double)
    [:float :double] (do (.f2d code) :double)
    [:double :int]  (do (.d2i code) :int)
    [:long :int]    (do (.l2i code) :int)
    [:float :int]   (do (.f2i code) :int)
    [:int :long]    (do (.i2l code) :long)
    [:double :long] (do (.d2l code) :long)
    [:float :long]  (do (.f2l code) :long)
    [:double :float] (do (.d2f code) :float)
    [:int :float]   (do (.i2f code) :float)
    [:long :float]  (do (.l2f code) :float)
    [:ref :float]   (do (.checkcast code number-cd)
                        (.invokevirtual code number-cd "floatValue" (MethodTypeDesc/of F-cd no-cd))
                        :float)
    [:ref :int]     (let [bool-cd (ClassDesc/ofDescriptor "Ljava/lang/Boolean;")
                          num-label (.newLabel code)
                          end-label (.newLabel code)]
                      ;; Boolean is not a Number — handle it specially:
                      ;; instanceof Boolean → booleanValue, else → Number.intValue
                      (.dup code)
                      (.instanceOf code bool-cd)
                      (.ifeq code num-label)
                      ;; Boolean path: Boolean.booleanValue() → int (true=1, false=0)
                      (.checkcast code bool-cd)
                      (.invokevirtual code bool-cd "booleanValue"
                                      (MethodTypeDesc/of (ClassDesc/ofDescriptor "Z") no-cd))
                      (.goto_ code end-label)
                      ;; Number path
                      (.labelBinding code num-label)
                      (.checkcast code number-cd)
                      (.invokevirtual code number-cd "intValue" (MethodTypeDesc/of I-cd no-cd))
                      (.labelBinding code end-label)
                      :int)
    [:ref :long]    (do (.checkcast code number-cd)
                        (.invokevirtual code number-cd "longValue" (MethodTypeDesc/of J-cd no-cd))
                        :long)
    ;; Boxing: primitive → ref
    [:double :ref]  (do (emit-box-to-ref code :double) :ref)
    [:float :ref]   (do (emit-box-to-ref code :float) :ref)
    [:int :ref]     (do (emit-box-to-ref code :int) :ref)
    [:long :ref]    (do (emit-box-to-ref code :long) :ref)
    ;; already the right type
    to-type))

(defn- walk-preserving-meta
  "Like clojure.walk/walk but preserves metadata on forms."
  [inner outer form]
  (let [m (meta form)
        result (cond
                 (list? form)    (outer (apply list (map inner form)))
                 (seq? form)     (outer (doall (map inner form)))
                 (vector? form)  (outer (mapv inner form))
                 (map? form)     (outer (into (empty form)
                                              (map (fn [[k v]] [(inner k) (inner v)]) form)))
                 (set? form)     (outer (into (empty form) (map inner form)))
                 :else           (outer form))]
    (if (and m (instance? clojure.lang.IObj result))
      (with-meta result (merge (meta result) m))
      result)))

(defn- postwalk-preserving
  "Like clojure.walk/postwalk but preserves metadata."
  [f form]
  (walk-preserving-meta (partial postwalk-preserving f) f form))

(defn- macroexpand-all-preserving
  "Like clojure.walk/macroexpand-all but preserves metadata on forms."
  [form]
  (let [expanded (if (seq? form) (macroexpand form) form)
        m (meta form)
        result (walk-preserving-meta macroexpand-all-preserving identity expanded)]
    (if (and m (instance? clojure.lang.IObj result))
      (with-meta result (merge (meta result) m))
      result)))

(defn desugar-invk
  "Process walker .invk forms for bytecode compilation.
  (.invk ^IFace ns/fn-impl args...) is kept as-is — the bytecode compiler's
  . handler emits invokeinterface using the -impl var's :tag metadata.
  This preserves typed dispatch through the interface instead of falling back
  to IFn.invoke via var lookup."
  [form]
  (cond
    (seq? form) (with-meta (apply list (map desugar-invk form)) (meta form))
    (vector? form) (with-meta (mapv desugar-invk form) (meta form))
    :else form))

(defn- comparison-form?
  "True if form is a comparison call: (< a b), (>= a b), (== a b), etc."
  [form]
  (and (seq? form) (symbol? (first form))
       (contains? #{"<" "<=" ">" ">=" "==" "not="} (name (first form)))))

(defn- pre-expand-par-forms
  "Pre-process forms for bytecode compilation:
  1. Fuse and/or+comparison patterns into direct (if (cmp) ...) for branch folding
  2. Replace raster.par/map! and reduce with int-counted loop* expansions
  3. Replace clojure.core/dotimes with int-counted loop* (dotimes promotes
     to long, preventing C2 counted loop recognition)
  The par macros and dotimes use long counters for Clojure eval, but the
  bytecode compiler needs int counters for IF_ICMPGE."
  [form]
  (cond
    ;; and/or + comparison fusion:
    ;; (let* [t (cmp)] (if t expr t)) → (if (cmp) expr false)   [and pattern]
    ;; (let* [t (cmp)] (if t t expr)) → (if (cmp) true expr)    [or pattern]
    ;; Safe because comparisons only produce true/false, so t's value
    ;; can be replaced with the constant. Enables branch folding in
    ;; emit-if-handler (single IF_ICMPxx instead of Boolean materialization
    ;; + truthiness test = 10 instructions saved per and/or).
    (and (seq? form) (= 'let* (first form))
         (let [bindings (second form)]
           (and (vector? bindings) (= 2 (count bindings))))
         (let [[sym pred] (second form)
               body (nth form 2 nil)]
           (and (symbol? sym) (comparison-form? pred)
                (seq? body) (= 'if (first body))
                (= sym (second body))
                (or (= sym (nth body 3 nil))   ;; (if t EXPR t) → and
                    (= sym (nth body 2 nil))))))  ;; (if t t EXPR) → or
    (let [[sym pred] (second form)
          [_ _ then-expr else-expr] (nth form 2)]
      (if (= sym else-expr)
        ;; and pattern: (let [t (cmp)] (if t expr t)) → (if (cmp) expr false)
        (pre-expand-par-forms (list 'if pred then-expr false))
        ;; or pattern: (let [t (cmp)] (if t t expr)) → (if (cmp) true expr)
        (pre-expand-par-forms (list 'if pred true else-expr))))

    (par/par-map-form? form)
    (try (par/expand-par-map! form)
         (catch Throwable _ form))
    ;; map-void! → dotimes (side-effect only, no output array)
    (par/par-map-void-form? form)
    (let [[_ idx-sym bound-expr body-expr] form]
      (pre-expand-par-forms
       (list 'dotimes [idx-sym bound-expr] body-expr)))

    (par/par-reduce-form? form)
    (try (par/expand-par-reduce form)
         (catch Throwable _ form))
    ;; Replace (dotimes [i n] body) with int-counted loop*
    (and (seq? form)
         (let [h (first form)]
           (or (= h 'clojure.core/dotimes) (= h 'dotimes)))
         (vector? (second form))
         (= 2 (count (second form))))
    (let [[_ [i-sym bound-expr] & body] form
          n-sym (gensym "n__")]
      (pre-expand-par-forms
       (list 'let* [n-sym (list 'int bound-expr)]
             (list 'loop* [i-sym '(int 0)]
                   (list 'if (list '< i-sym n-sym)
                         (apply list 'do (concat body
                                                 [(list 'recur (list 'clojure.core/unchecked-inc-int i-sym))]))
                         nil)))))
    (seq? form)
    (with-meta (apply list (map pre-expand-par-forms form)) (meta form))
    (vector? form)
    (with-meta (mapv pre-expand-par-forms form) (meta form))
    :else form))

;; Forward declarations for mutual recursion
(declare emit-form)
(declare compile-and-cache-deftm!)
(declare no-inline-deftm?)

(defn- emit-body
  "Emit a sequence of body forms, popping intermediate results.
  Returns the stack type of the last form (or :void if empty).
  This is the correct way to emit multi-form bodies (let*, loop*, do, try, etc.).
  Non-last forms are emitted in :void-context (result discarded), allowing
  intrinsics like aset to skip temp storage of the return value."
  [code forms locals ctx]
  (let [stmts (butlast forms)
        last-form (last forms)
        void-ctx (assoc ctx :void-context true)]
    (doseq [s stmts]
      (let [t (emit-form code s locals void-ctx)]
        (when-not (#{:void :diverge} t)
          (if (two-slot? t) (.pop2 code) (.pop code)))))
    (if (seq forms)
      ;; Preserve void-context if the enclosing form is already in void context.
      ;; Otherwise dissoc so the last form's value is kept on the stack.
      (emit-form code last-form locals
                 (if (:void-context ctx) ctx (dissoc ctx :void-context)))
      :void)))

;; Counter for unique anonymous class names
(def ^:private anon-counter (atom 0))

;; ================================================================
;; fn* support — anonymous function / closure compilation
;; ================================================================

(defn- collect-free-vars
  "Walk arity bodies to find symbols referenced from `locals`.
  Correctly tracks scoping: let*/loop* bindings, fn*/ftm params,
  letfn* mutual recursion, and try/catch exception bindings shadow
  outer names and are NOT counted as free variables.
  Returns a sorted vector of [sym local-info] for closed-over vars."
  [arities locals]
  (let [free (atom {})]
    (letfn [(walk-form [form bound]
              (cond
                (symbol? form)
                (when (and (contains? locals form)
                           (not (contains? bound form)))
                  (swap! free assoc form (get locals form)))

                (seq? form)
                (let [h (first form)]
                  (case (when (symbol? h) h)
                    ;; let*/loop*: sequential bindings — each sym visible
                    ;; in subsequent init exprs and in the body
                    (let* loop*)
                    (let [bindings (second form)
                          pairs (partition 2 bindings)
                          body (nnext form)]
                      (loop [ps (seq pairs) b bound]
                        (when ps
                          (let [[sym init] (first ps)]
                            (walk-form init b)
                            (recur (next ps) (conj b sym)))))
                      (let [all-bound (into bound (map first pairs))]
                        (doseq [f body] (walk-form f all-bound))))

                    ;; let: macroexpand to let* (handles destructuring)
                    let
                    (let [expanded (macroexpand-1 (list* 'let (rest form)))]
                      (walk-form expanded bound))

                    ;; fn*/fn: params bind in body, each arity independent
                    (fn* fn)
                    (let [rest-args (rest form)
                          ;; skip optional name symbol
                          arities (cond
                                    (symbol? (first rest-args))
                                    (let [r (rest rest-args)]
                                      (if (vector? (first r)) [r] r))
                                    (vector? (first rest-args))
                                    [rest-args]
                                    :else rest-args)]
                      (doseq [arity arities]
                        (let [params (first arity)
                              body (rest arity)
                              inner (into bound (remove #{'&} params))]
                          (doseq [f body] (walk-form f inner)))))

                    ;; letfn*: all names visible in all fn bodies + body
                    letfn*
                    (let [bindings (second form)
                          pairs (partition 2 bindings)
                          body (nnext form)
                          all-bound (into bound (map first pairs))]
                      (doseq [[_ fn-form] pairs]
                        (walk-form fn-form all-bound))
                      (doseq [f body] (walk-form f all-bound)))

                    ;; try/catch/finally: exception sym binds in catch body
                    try
                    (doseq [sub (rest form)]
                      (if (and (seq? sub) (= 'catch (first sub)))
                        (let [[_ _ex-class ex-sym & catch-body] sub
                              catch-bound (conj bound ex-sym)]
                          (doseq [f catch-body] (walk-form f catch-bound)))
                        (walk-form sub bound)))

                    ;; default: walk all subforms with current bound set
                    (doseq [f form] (walk-form f bound))))

                (vector? form)
                (doseq [f form] (walk-form f bound))

                (map? form)
                (doseq [[k v] form]
                  (walk-form k bound)
                  (walk-form v bound))

                (set? form)
                (doseq [f form] (walk-form f bound))))]
      (doseq [arity arities]
        (let [params (first arity)
              body   (rest arity)
              bound  (set (remove #{'&} params))]
          (doseq [b body] (walk-form b bound)))))
    (vec (sort-by (comp str first) @free))))

(defn- emit-anon-fn-class
  "Generate an anonymous IFn class with captured fields and invoke methods.
  Returns {:class-desc ClassDesc, :class Class, :field-names [str...],
           :field-types [ClassDesc...]}"
  [free-vars arities ctx]
  (let [class-num  (swap! anon-counter inc)
        class-name (str "raster.compiler.backend.jvm.bytecode.AnonFn__" class-num)
        dot-idx    (.lastIndexOf ^String class-name (int \.))
        pkg        (subs class-name 0 dot-idx)
        simple     (subs class-name (inc dot-idx))
        class-desc (ClassDesc/of pkg simple)
        ;; Field info for captured vars
        field-names (mapv (fn [[sym _]] (str sym)) free-vars)
        field-types (mapv (fn [[_ info]]
                            (case (:type info)
                              :double D-cd
                              :long   J-cd
                              :int    I-cd
                              :float  F-cd
                              obj-cd))
                          free-vars)
        cf (clojure-classfile)
        bytes (.build cf class-desc
                      (reify java.util.function.Consumer
                        (accept [_ cb]
                          (.withFlags cb (int 0x0001))  ;; PUBLIC
                          (.withVersion cb (ClassFile/latestMajorVersion) (ClassFile/latestMinorVersion))
                    ;; Implement AFn (which implements IFn)
                          (.withSuperclass cb (ClassDesc/ofDescriptor "Lclojure/lang/AFn;"))
                    ;; Fields for captured vars
                          (doseq [[fname ftype] (map vector field-names field-types)]
                            (.withField cb fname ftype (int 0x0012))) ;; PRIVATE FINAL
                    ;; Constructor: (captured1, captured2, ...)
                          (let [ctor-mt (MethodTypeDesc/of V-cd (into-array ClassDesc field-types))]
                            (.withMethodBody cb "<init>" ctor-mt
                                             (int 0x0001)  ;; PUBLIC
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                                           ;; super()
                                                 (.aload code 0)
                                                 (.invokespecial code
                                                                 (ClassDesc/ofDescriptor "Lclojure/lang/AFn;")
                                                                 "<init>" (MethodTypeDesc/of V-cd no-cd))
                                           ;; Store fields
                                                 (loop [idx 0, slot 1]
                                                   (when (< idx (count field-names))
                                                     (let [fname (nth field-names idx)
                                                           ftype (nth field-types idx)]
                                                       (.aload code 0)
                                                       (cond
                                                         (.equals ftype D-cd) (.dload code slot)
                                                         (.equals ftype J-cd) (.lload code slot)
                                                         (.equals ftype I-cd) (.iload code slot)
                                                         (.equals ftype F-cd) (.fload code slot)
                                                         :else (.aload code slot))
                                                       (.putfield code class-desc fname ftype)
                                                       (recur (inc idx)
                                                              (+ slot (if (or (.equals ftype D-cd)
                                                                              (.equals ftype J-cd))
                                                                        2 1))))))
                                                 (.return_ code)))))
                    ;; Invoke methods for each arity
                          (doseq [arity arities]
                            (let [params (first arity)
                                  body   (rest arity)
                                  n      (count params)
                                  invoke-mt (MethodTypeDesc/of obj-cd
                                                               (into-array ClassDesc (repeat n obj-cd)))]
                              (.withMethodBody cb "invoke" invoke-mt
                                               (int 0x0001)  ;; PUBLIC
                                               (reify java.util.function.Consumer
                                                 (accept [_ code]
                                             ;; Build locals: slot 0 = this, slots 1..n = params (Object)
                                                   (let [next-slot (atom (inc n))
                                                   ;; Captured vars → load from fields
                                                         captured-locals
                                                         (reduce
                                                          (fn [locs [sym info]]
                                                            (let [fname (str sym)
                                                                  ftype (case (:type info)
                                                                          :double D-cd :long J-cd
                                                                          :int I-cd :float F-cd obj-cd)
                                                                  st (:type info)
                                                                  slot (let [s @next-slot]
                                                                         (swap! next-slot + (if (two-slot? st) 2 1))
                                                                         s)]
                                                        ;; Load field from this
                                                              (.aload code 0)
                                                              (.getfield code class-desc fname ftype)
                                                              ;; Checkcast ref captures with known types
                                                              (when (and (= st :ref) (:hint info)
                                                                         (not= (:hint info) 'Object))
                                                                (let [hint (:hint info)
                                                                      cls (or (try (Class/forName (str hint))
                                                                                   (catch Exception _ nil))
                                                                              (when-let [r (ns-resolve
                                                                                            (or (:source-ns ctx) *ns*)
                                                                                            (symbol (str hint)))]
                                                                                (when (class? r) r)))]
                                                                  (when cls
                                                                    (.checkcast code (class-desc-of cls)))))
                                                              (case st
                                                                :double (.dstore code slot)
                                                                :long   (.lstore code slot)
                                                                :int    (.istore code slot)
                                                                :float  (.fstore code slot)
                                                                (.astore code slot))
                                                              (assoc locs sym {:slot slot :type st
                                                                               :hint (:hint info)})))
                                                          {} free-vars)
                                                   ;; Params → Object refs at slots 1..n
                                                         param-locals
                                                         (reduce
                                                          (fn [locs [i p]]
                                                            (assoc locs p {:slot (inc i) :type :ref}))
                                                          captured-locals
                                                          (map-indexed vector params))
                                                         inner-ctx {:class-desc class-desc
                                                                    :next-slot  next-slot
                                                                    :source-ns  (:source-ns ctx)}
                                                         expanded-body (map macroexpand-all-preserving body)
                                                         ret-type (emit-body code expanded-body param-locals inner-ctx)]
                                               ;; Box and return
                                                     (case ret-type
                                                       :ref    nil
                                                       :double (.invokestatic code dbl-box-cd "valueOf"
                                                                              (MethodTypeDesc/of dbl-box-cd (into-array ClassDesc [D-cd])))
                                                       :float  (.invokestatic code flt-box-cd "valueOf"
                                                                              (MethodTypeDesc/of flt-box-cd (into-array ClassDesc [F-cd])))
                                                       :int    (.invokestatic code int-box-cd "valueOf"
                                                                              (MethodTypeDesc/of int-box-cd (into-array ClassDesc [I-cd])))
                                                       :long   (.invokestatic code long-box-cd "valueOf"
                                                                              (MethodTypeDesc/of long-box-cd (into-array ClassDesc [J-cd])))
                                                       :void   (.aconst_null code)
                                                       :diverge nil)
                                                     (.areturn code))))))))))
        ;; Verify bytecode before loading
        _ (try (.verify (java.lang.classfile.ClassFile/of) bytes)
               (catch Exception e
                 (println "WARNING: Bytecode verification failed for" class-name ":" (.getMessage e))))
        loader (compilation-loader)
        cls (.defineClass loader class-name bytes nil)]
    {:class-desc class-desc
     :class      cls
     :field-names field-names
     :field-types field-types}))

;; ================================================================
;; reify* support — typed interface implementation with closures
;; ================================================================

(defn- resolve-interface-class
  "Resolve an interface symbol/class to a Class object."
  [iface-sym source-ns]
  (let [s (str iface-sym)]
    (or (try (Class/forName s) (catch Exception _ nil))
        (try (Class/forName (.replace s "/" ".")) (catch Exception _ nil))
        (when source-ns
          (when-let [r (ns-resolve source-ns (symbol s))]
            (when (class? r) r))))))

(defn- find-interface-method
  "Find a method by name and param count (excluding `this`) in a list of interface classes.
  Returns the java.lang.reflect.Method or nil."
  [method-name n-params iface-classes]
  (some (fn [^Class cls]
          (first (filter (fn [^java.lang.reflect.Method m]
                           (and (= (.getName m) method-name)
                                (= (.getParameterCount m) n-params)))
                         (.getMethods cls))))
        iface-classes))

(defn- emit-reify-class
  "Generate a class implementing the given interfaces with typed methods.
  Free variables are captured as fields, same as fn* closures.
  Implements IObj/IMeta for vary-meta compatibility (needed by ftm expansion).
  Returns {:class-desc ClassDesc, :field-names [...], :field-types [...]}"
  [iface-syms method-specs free-vars ctx]
  (let [class-num  (swap! anon-counter inc)
        class-name (str "raster.compiler.backend.jvm.bytecode.Reify__" class-num)
        dot-idx    (.lastIndexOf ^String class-name (int \.))
        pkg        (subs class-name 0 dot-idx)
        simple     (subs class-name (inc dot-idx))
        class-desc (ClassDesc/of pkg simple)
        source-ns  (or (:source-ns ctx) *ns*)
        ;; Resolve interface classes
        iface-classes (mapv #(resolve-interface-class % source-ns) iface-syms)
        _  (doseq [[sym cls] (map vector iface-syms iface-classes)]
             (when-not cls
               (throw (ex-info (str "Cannot resolve interface: " sym) {:interface sym}))))
        ;; Determine superclass: if IFn is among interfaces, extend AFn
        has-ifn? (some #(= clojure.lang.IFn %) iface-classes)
        super-cd (if has-ifn?
                   (ClassDesc/ofDescriptor "Lclojure/lang/AFn;")
                   obj-cd)
        ;; Interface ClassDescs (exclude IFn if extending AFn, since AFn implements IFn)
        ;; Always add IObj for vary-meta compatibility
        iobj-cd (ClassDesc/ofDescriptor "Lclojure/lang/IObj;")
        iface-cds (-> (mapv (fn [^Class c] (class-desc-of c))
                            (if has-ifn?
                              (remove #(= clojure.lang.IFn %) iface-classes)
                              iface-classes))
                      (conj iobj-cd))
        ;; ClassDescs for IObj/IMeta support
        ipmap-cd (ClassDesc/ofDescriptor "Lclojure/lang/IPersistentMap;")
        ;; Field info for captured vars
        field-names (mapv (fn [[sym _]] (str sym)) free-vars)
        field-types (mapv (fn [[_ info]]
                            (case (:type info)
                              :double D-cd
                              :long   J-cd
                              :int    I-cd
                              :float  F-cd
                              obj-cd))
                          free-vars)
        ;; Full constructor takes all captured fields + __meta as last param
        all-ctor-types (conj field-types ipmap-cd)
        cf (clojure-classfile)
        bytes (.build cf class-desc
                      (reify java.util.function.Consumer
                        (accept [_ cb]
                          (.withFlags cb (int 0x0001))  ;; PUBLIC
                          (.withVersion cb (ClassFile/latestMajorVersion) (ClassFile/latestMinorVersion))
                          (.withSuperclass cb super-cd)
                    ;; Add interfaces (including IObj)
                          (.withInterfaceSymbols cb (into-array ClassDesc iface-cds))
                    ;; Fields for captured vars
                          (doseq [[fname ftype] (map vector field-names field-types)]
                            (.withField cb fname ftype (int 0x0012))) ;; PRIVATE FINAL
                    ;; __meta field for IObj
                          (.withField cb "__meta" ipmap-cd (int 0x0012)) ;; PRIVATE FINAL
                    ;; Primary constructor: (captured1, captured2, ...) — sets __meta=null
                          (let [ctor-mt (MethodTypeDesc/of V-cd (into-array ClassDesc field-types))]
                            (.withMethodBody cb "<init>" ctor-mt
                                             (int 0x0001)  ;; PUBLIC
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                                           ;; super()
                                                 (.aload code 0)
                                                 (.invokespecial code super-cd
                                                                 "<init>" (MethodTypeDesc/of V-cd no-cd))
                                           ;; Store captured fields
                                                 (loop [idx 0, slot 1]
                                                   (when (< idx (count field-names))
                                                     (let [fname (nth field-names idx)
                                                           ftype (nth field-types idx)]
                                                       (.aload code 0)
                                                       (cond
                                                         (.equals ftype D-cd) (.dload code slot)
                                                         (.equals ftype J-cd) (.lload code slot)
                                                         (.equals ftype I-cd) (.iload code slot)
                                                         (.equals ftype F-cd) (.fload code slot)
                                                         :else (.aload code slot))
                                                       (.putfield code class-desc fname ftype)
                                                       (recur (inc idx)
                                                              (+ slot (if (or (.equals ftype D-cd)
                                                                              (.equals ftype J-cd))
                                                                        2 1))))))
                                           ;; __meta = null
                                                 (.aload code 0)
                                                 (.aconst_null code)
                                                 (.putfield code class-desc "__meta" ipmap-cd)
                                                 (.return_ code)))))
                    ;; Full constructor: (captured1, ..., __meta) — for withMeta
                          (let [full-ctor-mt (MethodTypeDesc/of V-cd (into-array ClassDesc all-ctor-types))]
                            (.withMethodBody cb "<init>" full-ctor-mt
                                             (int 0x0001)  ;; PUBLIC
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                                           ;; super()
                                                 (.aload code 0)
                                                 (.invokespecial code super-cd
                                                                 "<init>" (MethodTypeDesc/of V-cd no-cd))
                                           ;; Store captured fields
                                                 (loop [idx 0, slot 1]
                                                   (when (< idx (count field-names))
                                                     (let [fname (nth field-names idx)
                                                           ftype (nth field-types idx)]
                                                       (.aload code 0)
                                                       (cond
                                                         (.equals ftype D-cd) (.dload code slot)
                                                         (.equals ftype J-cd) (.lload code slot)
                                                         (.equals ftype I-cd) (.iload code slot)
                                                         (.equals ftype F-cd) (.fload code slot)
                                                         :else (.aload code slot))
                                                       (.putfield code class-desc fname ftype)
                                                       (recur (inc idx)
                                                              (+ slot (if (or (.equals ftype D-cd)
                                                                              (.equals ftype J-cd))
                                                                        2 1))))))
                                           ;; Store __meta (last ctor param)
                                                 (let [meta-slot (loop [idx 0, slot 1]
                                                                   (if (>= idx (count field-names))
                                                                     slot
                                                                     (let [ftype (nth field-types idx)]
                                                                       (recur (inc idx)
                                                                              (+ slot (if (or (.equals ftype D-cd)
                                                                                              (.equals ftype J-cd))
                                                                                        2 1))))))]
                                                   (.aload code 0)
                                                   (.aload code meta-slot)
                                                   (.putfield code class-desc "__meta" ipmap-cd))
                                                 (.return_ code)))))
                    ;; meta() method — returns __meta field
                          (.withMethodBody cb "meta"
                                           (MethodTypeDesc/of ipmap-cd no-cd)
                                           (int 0x0001)  ;; PUBLIC
                                           (reify java.util.function.Consumer
                                             (accept [_ code]
                                               (.aload code 0)
                                               (.getfield code class-desc "__meta" ipmap-cd)
                                               (.areturn code))))
                    ;; withMeta(IPersistentMap) — creates new instance with same captures + new meta
                          (.withMethodBody cb "withMeta"
                                           (MethodTypeDesc/of iobj-cd (into-array ClassDesc [ipmap-cd]))
                                           (int 0x0001)  ;; PUBLIC
                                           (reify java.util.function.Consumer
                                             (accept [_ code]
                                         ;; new ThisClass(field1, field2, ..., newMeta)
                                               (.new_ code class-desc)
                                               (.dup code)
                                         ;; Load all captured fields from this
                                               (doseq [[fname ftype] (map vector field-names field-types)]
                                                 (.aload code 0)
                                                 (.getfield code class-desc fname ftype))
                                         ;; Load newMeta from arg slot 1
                                               (.aload code 1)
                                         ;; Invoke full constructor
                                               (.invokespecial code class-desc "<init>"
                                                               (MethodTypeDesc/of V-cd (into-array ClassDesc all-ctor-types)))
                                               (.areturn code))))
                    ;; Emit each method
                          (doseq [method-spec method-specs]
                            (let [meth-name-sym (first method-spec)
                                  meth-name-str (str meth-name-sym)
                                  params        (second method-spec)
                                  body          (nnext method-spec)
                                  this-sym      (first params)
                                  method-params (vec (rest params))
                                  n-params      (count method-params)
                            ;; Find the matching method via reflection
                                  method        (find-interface-method meth-name-str n-params iface-classes)]
                              (if-not method
                          ;; No interface method found — emit as Object invoke
                          ;; (fallback for IFn.invoke which may not appear in reflection
                          ;;  if AFn is the superclass)
                                (let [invoke-mt (MethodTypeDesc/of obj-cd
                                                                   (into-array ClassDesc (repeat n-params obj-cd)))]
                                  (.withMethodBody cb meth-name-str invoke-mt
                                                   (int 0x0001) ;; PUBLIC
                                                   (reify java.util.function.Consumer
                                                     (accept [_ code]
                                                       (let [next-slot (atom (inc n-params))
                                                       ;; Load captured vars from fields
                                                             captured-locals
                                                             (reduce
                                                              (fn [locs [sym info]]
                                                                (let [fname (str sym)
                                                                      ftype (case (:type info)
                                                                              :double D-cd :long J-cd
                                                                              :int I-cd :float F-cd obj-cd)
                                                                      st (:type info)
                                                                      slot (let [s @next-slot]
                                                                             (swap! next-slot + (if (two-slot? st) 2 1))
                                                                             s)]
                                                                  (.aload code 0)
                                                                  (.getfield code class-desc fname ftype)
                                                                  (when (and (= st :ref) (:hint info)
                                                                             (not= (:hint info) 'Object))
                                                                    (let [hint (:hint info)
                                                                          cls (or (try (Class/forName (str hint))
                                                                                       (catch Exception _ nil))
                                                                                  (when-let [r (ns-resolve source-ns (symbol (str hint)))]
                                                                                    (when (class? r) r)))]
                                                                      (when cls (.checkcast code (class-desc-of cls)))))
                                                                  (case st
                                                                    :double (.dstore code slot)
                                                                    :long   (.lstore code slot)
                                                                    :int    (.istore code slot)
                                                                    :float  (.fstore code slot)
                                                                    (.astore code slot))
                                                                  (assoc locs sym {:slot slot :type st :hint (:hint info)})))
                                                              {} free-vars)
                                                       ;; Params: all Object for IFn.invoke
                                                             param-locals
                                                             (reduce
                                                              (fn [locs [i p]]
                                                                (assoc locs p {:slot (inc i) :type :ref}))
                                                              captured-locals
                                                              (map-indexed vector method-params))
                                                             inner-ctx {:class-desc class-desc
                                                                        :next-slot  next-slot
                                                                        :source-ns  source-ns}
                                                             expanded-body (map macroexpand-all-preserving body)
                                                             ret-type (emit-body code expanded-body param-locals inner-ctx)]
                                                   ;; Box and return Object
                                                         (case ret-type
                                                           :ref     nil
                                                           :double  (.invokestatic code dbl-box-cd "valueOf"
                                                                                   (MethodTypeDesc/of dbl-box-cd (into-array ClassDesc [D-cd])))
                                                           :float   (.invokestatic code flt-box-cd "valueOf"
                                                                                   (MethodTypeDesc/of flt-box-cd (into-array ClassDesc [F-cd])))
                                                           :int     (.invokestatic code int-box-cd "valueOf"
                                                                                   (MethodTypeDesc/of int-box-cd (into-array ClassDesc [I-cd])))
                                                           :long    (.invokestatic code long-box-cd "valueOf"
                                                                                   (MethodTypeDesc/of long-box-cd (into-array ClassDesc [J-cd])))
                                                           :void    (.aconst_null code)
                                                           :diverge nil)
                                                         (.areturn code))))))
                          ;; Found interface method — emit with correct types
                                (let [^java.lang.reflect.Method method method
                                      param-types  (.getParameterTypes method)
                                      ret-class    (.getReturnType method)
                                      ret-st       (class-type->stack ret-class)
                                      param-cds    (mapv class-desc-of param-types)
                                      ret-cd       (class-desc-of ret-class)
                                      meth-mt      (MethodTypeDesc/of ret-cd (into-array ClassDesc param-cds))]
                                  (.withMethodBody cb meth-name-str meth-mt
                                                   (int 0x0001) ;; PUBLIC
                                                   (reify java.util.function.Consumer
                                                     (accept [_ code]
                                                 ;; Build locals: slot 0 = this (ref)
                                                 ;; Typed params get correct slot widths
                                                       (let [;; Calculate param slots accounting for double/long width
                                                             param-slot-info
                                                             (loop [i 0, slot 1, acc []]
                                                               (if (>= i n-params)
                                                                 acc
                                                                 (let [^Class pt (aget param-types i)
                                                                       st (class-type->stack pt)]
                                                                   (recur (inc i)
                                                                          (+ slot (if (two-slot? st) 2 1))
                                                                          (conj acc {:slot slot :type st})))))
                                                             first-free-slot (if (seq param-slot-info)
                                                                               (let [{:keys [slot type]} (last param-slot-info)]
                                                                                 (+ slot (if (two-slot? type) 2 1)))
                                                                               1)
                                                             next-slot (atom first-free-slot)
                                                       ;; Load captured vars from fields
                                                             captured-locals
                                                             (reduce
                                                              (fn [locs [sym info]]
                                                                (let [fname (str sym)
                                                                      ftype (case (:type info)
                                                                              :double D-cd :long J-cd
                                                                              :int I-cd :float F-cd obj-cd)
                                                                      st (:type info)
                                                                      slot (let [s @next-slot]
                                                                             (swap! next-slot + (if (two-slot? st) 2 1))
                                                                             s)]
                                                                  (.aload code 0)
                                                                  (.getfield code class-desc fname ftype)
                                                                  (when (and (= st :ref) (:hint info)
                                                                             (not= (:hint info) 'Object))
                                                                    (let [hint (:hint info)
                                                                          cls (or (try (Class/forName (str hint))
                                                                                       (catch Exception _ nil))
                                                                                  (when-let [r (ns-resolve source-ns (symbol (str hint)))]
                                                                                    (when (class? r) r)))]
                                                                      (when cls (.checkcast code (class-desc-of cls)))))
                                                                  (case st
                                                                    :double (.dstore code slot)
                                                                    :long   (.lstore code slot)
                                                                    :int    (.istore code slot)
                                                                    :float  (.fstore code slot)
                                                                    (.astore code slot))
                                                                  (assoc locs sym {:slot slot :type st :hint (:hint info)})))
                                                              {} free-vars)
                                                       ;; Build param locals with correct types from interface
                                                             param-locals
                                                             (reduce
                                                              (fn [locs [i p]]
                                                                (let [{:keys [slot type]} (nth param-slot-info i)]
                                                                  (assoc locs p {:slot slot :type type})))
                                                              captured-locals
                                                              (map-indexed vector method-params))
                                                             inner-ctx {:class-desc class-desc
                                                                        :next-slot  next-slot
                                                                        :source-ns  source-ns}
                                                             expanded-body (map macroexpand-all-preserving body)
                                                             body-type (emit-body code expanded-body param-locals inner-ctx)]
                                                   ;; Coerce body result to method return type
                                                         (when (and (not= body-type ret-st)
                                                                    (not (diverge? body-type)))
                                                           (cond
                                                       ;; void return — pop the value
                                                             (= ret-st :void)
                                                             (when-not (#{:void :diverge} body-type)
                                                               (if (two-slot? body-type) (.pop2 code) (.pop code)))
                                                       ;; need to coerce
                                                             :else
                                                             (emit-coerce code body-type ret-st)))
                                                   ;; Return with correct instruction
                                                         (case ret-st
                                                           :double  (.dreturn code)
                                                           :long    (.lreturn code)
                                                           :int     (.ireturn code)
                                                           :float   (.freturn code)
                                                           :void    (.return_ code)
                                                           (.areturn code)))))))))))))
        loader (compilation-loader)
        cls (.defineClass loader class-name bytes nil)]
    {:class-desc class-desc
     :class      cls
     :field-names field-names
     :field-types field-types}))

;; ================================================================
;; emit-core-intrinsic — clojure.core function bytecode
;; ================================================================

(defn- emit-predicate-intrinsic
  "Emit bytecode for type predicates, boolean logic, coercions, and
   sign/parity predicates."
  [code head-name args locals ctx]
  (case head-name
    ;; ---- Type predicates ----
    "nil?"
    (let [t (emit-form code (first args) locals ctx)
          null-label (.newLabel code)
          end-label  (.newLabel code)]
      (when (primitive? t) (emit-box-to-ref code t))
      (.ifnull code null-label)
      (.ldc code (int 0)) (.goto_ code end-label)
      (.labelBinding code null-label) (.ldc code (int 1))
      (.labelBinding code end-label) :int)

    "some?"
    (let [t (emit-form code (first args) locals ctx)
          null-label (.newLabel code)
          end-label  (.newLabel code)]
      (when (primitive? t) (emit-box-to-ref code t))
      (.ifnull code null-label)
      (.ldc code (int 1)) (.goto_ code end-label)
      (.labelBinding code null-label) (.ldc code (int 0))
      (.labelBinding code end-label) :int)

    "identical?"
    (let [t1 (emit-form code (first args) locals ctx)]
      (when (primitive? t1) (emit-box-to-ref code t1))
      (let [t2 (emit-form code (second args) locals ctx)]
        (when (primitive? t2) (emit-box-to-ref code t2)))
      (let [eq-label  (.newLabel code)
            end-label (.newLabel code)]
        (.if_acmpeq code eq-label)
        (.ldc code (int 0)) (.goto_ code end-label)
        (.labelBinding code eq-label) (.ldc code (int 1))
        (.labelBinding code end-label) :int))

    "instance?"
    (let [[cls-form val-form] args
          src-ns (or (:source-ns ctx) *ns*)
          cls (or (try (Class/forName (str cls-form)) (catch Exception _ nil))
                  (when-let [r (ns-resolve src-ns (symbol (str cls-form)))]
                    (when (class? r) r)))
          t (emit-form code val-form locals ctx)]
      (when (primitive? t) (emit-box-to-ref code t))
      (.instanceOf code (class-desc-of cls))
      :int)

    "boolean?"
    (let [t (emit-form code (first args) locals ctx)]
      (when (primitive? t) (emit-box-to-ref code t))
      (.instanceOf code (ClassDesc/ofDescriptor "Ljava/lang/Boolean;"))
      :int)

    "number?"
    (let [t (emit-form code (first args) locals ctx)]
      (when (primitive? t) (emit-box-to-ref code t))
      (.instanceOf code number-cd)
      :int)

    "string?"
    (let [t (emit-form code (first args) locals ctx)]
      (when (primitive? t) (emit-box-to-ref code t))
      (.instanceOf code (ClassDesc/ofDescriptor "Ljava/lang/String;"))
      :int)

    ;; ---- Boolean logic ----
    "not"
    (let [t (emit-form code (first args) locals ctx)
          false-label (.newLabel code)
          end-label   (.newLabel code)]
      (case t
        :int  (.ifeq code false-label)
        :ref  (let [not-null (.newLabel code)]
                (.dup code)
                (.ifnonnull code not-null)
                (.pop code)
                (.goto_ code false-label)
                (.labelBinding code not-null)
                (.getstatic code (ClassDesc/ofDescriptor "Ljava/lang/Boolean;")
                            "FALSE" (ClassDesc/ofDescriptor "Ljava/lang/Boolean;"))
                (.if_acmpeq code false-label)
                ;; truthy → not = false
                (.ldc code (int 0)) (.goto_ code end-label)
                (.labelBinding code false-label) (.ldc code (int 1))
                (.labelBinding code end-label)
                (do))  ;; prevent falling through
        :double (do (.ldc code 0.0) (.dcmpl code) (.ifeq code false-label))
        :float  (do (.ldc code (float 0.0)) (.fcmpl code) (.ifeq code false-label))
        :long   (do (.ldc code (long 0)) (.lcmp code) (.ifeq code false-label)))
      (when (#{:int :double :float :long} t)
        (.ldc code (int 0)) (.goto_ code end-label)
        (.labelBinding code false-label) (.ldc code (int 1))
        (.labelBinding code end-label))
      :int)

    ;; ---- Numeric coercions ----
    "byte"  (let [t (emit-form code (first args) locals ctx)]
              (when (not= t :int) (emit-coerce code t :int))
              (.i2b code) :int)
    "short" (let [t (emit-form code (first args) locals ctx)]
              (when (not= t :int) (emit-coerce code t :int))
              (.i2s code) :int)
    "char"  (let [t (emit-form code (first args) locals ctx)]
              (when (not= t :int) (emit-coerce code t :int))
              (.i2c code) :int)
    "boolean" (let [t (emit-form code (first args) locals ctx)
                    true-label (.newLabel code)
                    end-label  (.newLabel code)]
                (case t
                  :int  (do (.ifne code true-label)
                            (.ldc code (int 0)) (.goto_ code end-label)
                            (.labelBinding code true-label) (.ldc code (int 1))
                            (.labelBinding code end-label))
                  :ref  (let [not-null (.newLabel code)]
                          (.dup code)
                          (.ifnonnull code not-null)
                          (.pop code)
                          (.ldc code (int 0)) (.goto_ code end-label)
                          (.labelBinding code not-null)
                          (.getstatic code (ClassDesc/ofDescriptor "Ljava/lang/Boolean;")
                                      "FALSE" (ClassDesc/ofDescriptor "Ljava/lang/Boolean;"))
                          (.if_acmpeq code true-label)
                          ;; truthy
                          (.ldc code (int 1)) (.goto_ code end-label)
                          (.labelBinding code true-label) (.ldc code (int 0))
                          (.labelBinding code end-label))
                  :double (do (.ldc code 0.0) (.dcmpl code)
                              (.ifne code true-label)
                              (.ldc code (int 0)) (.goto_ code end-label)
                              (.labelBinding code true-label) (.ldc code (int 1))
                              (.labelBinding code end-label))
                  :float  (do (.ldc code (float 0.0)) (.fcmpl code)
                              (.ifne code true-label)
                              (.ldc code (int 0)) (.goto_ code end-label)
                              (.labelBinding code true-label) (.ldc code (int 1))
                              (.labelBinding code end-label))
                  :long   (do (.ldc code (long 0)) (.lcmp code)
                              (.ifne code true-label)
                              (.ldc code (int 0)) (.goto_ code end-label)
                              (.labelBinding code true-label) (.ldc code (int 1))
                              (.labelBinding code end-label)))
                :int)

    ;; ---- Array constructors ----
    "long-array"    (cond
                      (and (= 1 (count args)) (vector? (first args)))
                      (let [elts (first args) n (count elts)]
                        (.ldc code (int n))
                        (.newarray code java.lang.classfile.TypeKind/LONG)
                        (dotimes [i n]
                          (.dup code) (.ldc code (int i))
                          (.ldc code (long (nth elts i))) (.lastore code))
                        :ref)
                      (= 2 (count args))
                      (do (let [t (emit-form code (first args) locals ctx)]
                            (when (not= t :int) (emit-coerce code t :int)))
                          (.newarray code java.lang.classfile.TypeKind/LONG)
                          (.dup code)
                          (let [t (emit-form code (second args) locals ctx)]
                            (when (not= t :long) (emit-coerce code t :long)))
                          (.invokestatic code
                                         (ClassDesc/of "java.util.Arrays") "fill"
                                         (MethodTypeDesc/of V-cd (into-array ClassDesc [lngarr-cd J-cd])))
                          :ref)
                      :else
                      (do (let [t (emit-form code (first args) locals ctx)]
                            (when (not= t :int) (emit-coerce code t :int)))
                          (.newarray code java.lang.classfile.TypeKind/LONG) :ref))
    "int-array"     (cond
                      (and (= 1 (count args)) (vector? (first args)))
                      (let [elts (first args) n (count elts)]
                        (.ldc code (int n))
                        (.newarray code java.lang.classfile.TypeKind/INT)
                        (dotimes [i n]
                          (.dup code) (.ldc code (int i))
                          (.ldc code (int (nth elts i))) (.iastore code))
                        :ref)
                      (= 2 (count args))
                      (do (let [t (emit-form code (first args) locals ctx)]
                            (when (not= t :int) (emit-coerce code t :int)))
                          (.newarray code java.lang.classfile.TypeKind/INT)
                          (.dup code)
                          (let [t (emit-form code (second args) locals ctx)]
                            (when (not= t :int) (emit-coerce code t :int)))
                          (.invokestatic code
                                         (ClassDesc/of "java.util.Arrays") "fill"
                                         (MethodTypeDesc/of V-cd (into-array ClassDesc [intarr-cd I-cd])))
                          :ref)
                      :else
                      (do (let [t (emit-form code (first args) locals ctx)]
                            (when (not= t :int) (emit-coerce code t :int)))
                          (.newarray code java.lang.classfile.TypeKind/INT) :ref))
    "byte-array"    (do (let [t (emit-form code (first args) locals ctx)]
                          (when (not= t :int) (emit-coerce code t :int)))
                        (.newarray code java.lang.classfile.TypeKind/BYTE) :ref)
    "short-array"   (do (let [t (emit-form code (first args) locals ctx)]
                          (when (not= t :int) (emit-coerce code t :int)))
                        (.newarray code java.lang.classfile.TypeKind/SHORT) :ref)
    "boolean-array" (do (let [t (emit-form code (first args) locals ctx)]
                          (when (not= t :int) (emit-coerce code t :int)))
                        (.newarray code java.lang.classfile.TypeKind/BOOLEAN) :ref)
    "char-array"    (do (let [t (emit-form code (first args) locals ctx)]
                          (when (not= t :int) (emit-coerce code t :int)))
                        (.newarray code java.lang.classfile.TypeKind/CHAR) :ref)
    "make-array"    (let [src-ns (or (:source-ns ctx) *ns*)
                          cls (or (try (Class/forName (str (first args))) (catch Exception _ nil))
                                  (when-let [r (ns-resolve src-ns (symbol (str (first args))))]
                                    (when (class? r) r))
                                  Object)]
                      (let [t (emit-form code (second args) locals ctx)]
                        (when (not= t :int) (emit-coerce code t :int)))
                      (.anewarray code (class-desc-of cls)) :ref)

    ;; ---- abs (polymorphic) ----
    "abs" (let [t (emit-form code (first args) locals ctx)]
            (case t
              :double (do (.invokestatic code math-cd "abs"
                                         (MethodTypeDesc/of D-cd (into-array ClassDesc [D-cd])))
                          :double)
              :long   (do (.invokestatic code math-cd "abs"
                                         (MethodTypeDesc/of J-cd (into-array ClassDesc [J-cd])))
                          :long)
              :int    (do (.invokestatic code math-cd "abs"
                                         (MethodTypeDesc/of I-cd (into-array ClassDesc [I-cd])))
                          :int)
              :float  (do (.invokestatic code math-cd "abs"
                                         (MethodTypeDesc/of F-cd (into-array ClassDesc [F-cd])))
                          :float)
              ;; ref → unbox to double, abs, result double
              (do (emit-unbox-double code)
                  (.invokestatic code math-cd "abs"
                                 (MethodTypeDesc/of D-cd (into-array ClassDesc [D-cd])))
                  :double)))

    ;; ---- zero? ----
    "zero?"
    (let [t (emit-form code (first args) locals ctx)
          zero-label (.newLabel code)
          end-label  (.newLabel code)]
      (case t
        :int    (do (.ifeq code zero-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code zero-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :long   (do (.ldc code (long 0)) (.lcmp code) (.ifeq code zero-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code zero-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :double (do (.ldc code 0.0) (.dcmpl code) (.ifeq code zero-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code zero-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :float  (do (.ldc code (float 0.0)) (.fcmpl code) (.ifeq code zero-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code zero-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        ;; ref → unbox and compare
        (do (emit-unbox-double code) (.ldc code 0.0) (.dcmpl code) (.ifeq code zero-label)
            (.ldc code (int 0)) (.goto_ code end-label)
            (.labelBinding code zero-label) (.ldc code (int 1))
            (.labelBinding code end-label) :int)))

    ;; ---- pos? / neg? ----
    "pos?"
    (let [t (emit-form code (first args) locals ctx)
          true-label (.newLabel code)
          end-label  (.newLabel code)]
      (case t
        :int    (do (.ifgt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :long   (do (.ldc code (long 0)) (.lcmp code) (.ifgt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :float  (do (.ldc code (float 0.0)) (.fcmpg code) (.ifgt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :double (do (.ldc code 0.0) (.dcmpg code) (.ifgt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        (do (emit-unbox-double code) (.ldc code 0.0) (.dcmpg code) (.ifgt code true-label)
            (.ldc code (int 0)) (.goto_ code end-label)
            (.labelBinding code true-label) (.ldc code (int 1))
            (.labelBinding code end-label) :int)))

    "neg?"
    (let [t (emit-form code (first args) locals ctx)
          true-label (.newLabel code)
          end-label  (.newLabel code)]
      (case t
        :int    (do (.iflt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :long   (do (.ldc code (long 0)) (.lcmp code) (.iflt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :float  (do (.ldc code (float 0.0)) (.fcmpg code) (.iflt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        :double (do (.ldc code 0.0) (.dcmpg code) (.iflt code true-label)
                    (.ldc code (int 0)) (.goto_ code end-label)
                    (.labelBinding code true-label) (.ldc code (int 1))
                    (.labelBinding code end-label) :int)
        (do (emit-unbox-double code) (.ldc code 0.0) (.dcmpg code) (.iflt code true-label)
            (.ldc code (int 0)) (.goto_ code end-label)
            (.labelBinding code true-label) (.ldc code (int 1))
            (.labelBinding code end-label) :int)))

    ;; ---- even? / odd? ----
    "even?"
    (let [t (emit-form code (first args) locals ctx)
          true-label (.newLabel code)
          end-label  (.newLabel code)]
      (if (= t :long)
        (do (.ldc code (long 1)) (.land code) (.l2i code)
            (.ifeq code true-label)
            (.ldc code (int 0)) (.goto_ code end-label)
            (.labelBinding code true-label) (.ldc code (int 1))
            (.labelBinding code end-label) :int)
        (do (when (not= t :int) (emit-coerce code t :int))
            (.ldc code (int 1)) (.iand code)
            (.ifeq code true-label)
            (.ldc code (int 0)) (.goto_ code end-label)
            (.labelBinding code true-label) (.ldc code (int 1))
            (.labelBinding code end-label) :int)))

    "odd?"
    (let [t (emit-form code (first args) locals ctx)
          true-label (.newLabel code)
          end-label  (.newLabel code)]
      (if (= t :long)
        (do (.ldc code (long 1)) (.land code) (.l2i code)
            (.ifne code true-label)
            (.ldc code (int 0)) (.goto_ code end-label)
            (.labelBinding code true-label) (.ldc code (int 1))
            (.labelBinding code end-label) :int)
        (do (when (not= t :int) (emit-coerce code t :int))
            (.ldc code (int 1)) (.iand code)
            (.ifne code true-label)
            (.ldc code (int 0)) (.goto_ code end-label)
            (.labelBinding code true-label) (.ldc code (int 1))
            (.labelBinding code end-label) :int)))

    ;; Not handled here
    nil))

(defn- widen-arithmetic-type
  "Determine the widened result type for two arithmetic operands.
   Preserves integer types: int×int→int, long×long→long, int×long→long.
   Preserves float: float×float→float. Otherwise→double.
   Division always uses float/double (integer division is quot, not /)."
  [t1 t2 division?]
  (if division?
    ;; Division: float×float→float, everything else→double
    (if (and (= t1 :float) (= t2 :float)) :float :double)
    (case t1
      :int    (case t2 :int :int, :long :long, :float :float, :double)
      :long   (case t2 (:int :long) :long, :double)
      :float  (case t2 :float :float, :double)
      :double :double
      ;; :ref — unbox, treat as double
      :double)))

(defn- emit-arith-op
  "Emit a single arithmetic instruction for the given type and op string."
  [code op type]
  (case type
    :int    (case op "+" (.iadd code) "-" (.isub code) "*" (.imul code) "/" (.idiv code))
    :long   (case op "+" (.ladd code) "-" (.lsub code) "*" (.lmul code) "/" (.ldiv code))
    :float  (case op "+" (.fadd code) "-" (.fsub code) "*" (.fmul code) "/" (.fdiv code))
    :double (case op "+" (.dadd code) "-" (.dsub code) "*" (.dmul code) "/" (.ddiv code))))

(defn- emit-arithmetic
  "Emit type-aware arithmetic bytecode. Preserves integer types:
   int×int→int, long×long→long, int×long→long, float×float→float, else→double.
   Division (/) always uses float or double (integer division is quot)."
  [code op args locals ctx]
  (cond
    ;; Unary negation: (- x) → negate, preserving type
    (and (= op "-") (= 1 (count args)))
    (let [t (emit-form code (first args) locals ctx)]
      (case t
        :int    (do (.ineg code) :int)
        :long   (do (.lneg code) :long)
        :float  (do (.fneg code) :float)
        :double (do (.dneg code) :double)
        ;; :ref → coerce to double
        (do (emit-coerce code t :double) (.dneg code) :double)))

    ;; Unary reciprocal: (/ x) → 1.0 / x (always floating point)
    (and (= op "/") (= 1 (count args)))
    (let [t (emit-form code (first args) locals ctx)]
      (if (= t :float)
        (do (.ldc code (float 1.0)) (.swap code) (.fdiv code) :float)
        (do (when (not= t :double) (emit-coerce code t :double))
            (let [tmp @(:next-slot ctx)]
              (swap! (:next-slot ctx) + 2)
              (.dstore code tmp)
              (.ldc code 1.0)
              (.dload code tmp)
              (.ddiv code) :double))))

    :else
    ;; Multi-arg: emit first arg, then fold remaining with type widening
    (let [division? (= op "/")
          t1 (emit-form code (first args) locals ctx)
          ;; For :ref, coerce to double eagerly
          t1 (if (= t1 :ref) (do (emit-coerce code t1 :double) :double) t1)
          ;; For division with integer first arg, coerce to double
          t1 (if (and division? (#{:int :long} t1))
               (do (emit-coerce code t1 :double) :double)
               t1)]
      (loop [remaining (rest args) cur t1]
        (if (empty? remaining)
          cur
          (let [t2 (emit-form code (first remaining) locals ctx)
                ;; For :ref, coerce to double eagerly
                t2 (if (= t2 :ref) (do (emit-coerce code t2 :double) :double) t2)
                target (widen-arithmetic-type cur t2 division?)]
            ;; Coerce t2 (on top of stack) to target if needed
            (when (not= t2 target)
              (emit-coerce code t2 target))
            ;; Coerce cur (below t2) to target if needed
            ;; This requires store/coerce/reload when sizes differ
            (when (not= cur target)
              (let [tmp @(:next-slot ctx)]
                ;; Store t2 (on top)
                (case target
                  (:double :long) (do (swap! (:next-slot ctx) + 2)
                                      (if (= target :double) (.dstore code tmp) (.lstore code tmp)))
                  (do (swap! (:next-slot ctx) inc)
                      (if (= target :float) (.fstore code tmp) (.istore code tmp))))
                ;; Coerce cur
                (emit-coerce code cur target)
                ;; Reload t2
                (case target
                  :double (.dload code tmp) :long (.lload code tmp)
                  :float (.fload code tmp) :int (.iload code tmp))))
            (emit-arith-op code op target)
            (recur (rest remaining) target)))))))

(defn- emit-comparison
  "Emit comparison bytecode. Uses int/long comparison when both operands are
   integer types (enables C2 loop optimizations: unrolling, auto-vectorization).
   Falls back to dcmpg for float/double operands.
   Only binary comparisons supported; chained comparisons like (< a b c)
   must be desugared to (and (< a b) (< b c)) before reaching bytecode."
  [code op args locals ctx]
  (when (not= 2 (count args))
    (throw (ex-info (str "Comparison " op " requires exactly 2 args in bytecode, got " (count args)
                         ". Chained comparisons must be desugared before compilation.")
                    {:op op :arg-count (count args)})))
  (let [end-label  (.newLabel code)
        true-label (.newLabel code)
        ;; Peek at operand types before emitting, to choose the right strategy
        a1 (first args) a2 (second args)
        t1-hint (cond (and (seq? a1) (= 'int (first a1))) :int
                      (and (seq? a1) (= 'long (first a1))) :long
                      (and (seq? a1) (= 'float (first a1))) :float
                      (and (symbol? a1) (= :int (:type (get locals a1)))) :int
                      (and (symbol? a1) (= :long (:type (get locals a1)))) :long
                      (and (symbol? a1) (= :float (:type (get locals a1)))) :float
                      :else nil)
        t2-hint (cond (and (seq? a2) (= 'int (first a2))) :int
                      (and (seq? a2) (= 'long (first a2))) :long
                      (and (seq? a2) (= 'float (first a2))) :float
                      (and (symbol? a2) (= :int (:type (get locals a2)))) :int
                      (and (symbol? a2) (= :long (:type (get locals a2)))) :long
                      (and (symbol? a2) (= :float (:type (get locals a2)))) :float
                      :else nil)
        use-int? (and (= t1-hint :int) (= t2-hint :int))
        use-long? (and (= t1-hint :long) (= t2-hint :long))
        ;; Mixed int/long: promote both to long (avoids double conversion)
        use-long-mixed? (and (#{:int :long} t1-hint) (#{:int :long} t2-hint)
                             (not use-int?))
        use-float? (and (= t1-hint :float) (= t2-hint :float))]
    (if use-int?
      ;; Both int: use if_icmpXX (no conversion, fastest path for loop counters)
      (let [t1 (emit-form code a1 locals ctx)
            _ (when (not= t1 :int) (emit-coerce code t1 :int))
            t2 (emit-form code a2 locals ctx)
            _ (when (not= t2 :int) (emit-coerce code t2 :int))]
        (case op
          "<"  (.if_icmplt code true-label) "<=" (.if_icmple code true-label)
          ">"  (.if_icmpgt code true-label) ">=" (.if_icmpge code true-label)
          "==" (.if_icmpeq code true-label) "not=" (.if_icmpne code true-label)))
      (if (or use-long? use-long-mixed?)
        ;; Both long (or mixed int/long → promote to long): use lcmp + ifXX
        (let [t1 (emit-form code a1 locals ctx)
              _ (when (not= t1 :long) (emit-coerce code t1 :long))
              t2 (emit-form code a2 locals ctx)
              _ (when (not= t2 :long) (emit-coerce code t2 :long))]
          (.lcmp code)
          (case op
            "<"  (.iflt code true-label) "<=" (.ifle code true-label)
            ">"  (.ifgt code true-label) ">=" (.ifge code true-label)
            "==" (.ifeq code true-label) "not=" (.ifne code true-label)))
        (if use-float?
          ;; Both float: use fcmpg + ifXX (avoids f2d widening)
          (let [t1 (emit-form code a1 locals ctx)
                _ (when (not= t1 :float) (emit-coerce code t1 :float))
                t2 (emit-form code a2 locals ctx)
                _ (when (not= t2 :float) (emit-coerce code t2 :float))]
            (.fcmpg code)
            (case op
              "<"  (.iflt code true-label) "<=" (.ifle code true-label)
              ">"  (.ifgt code true-label) ">=" (.ifge code true-label)
              "==" (.ifeq code true-label) "not=" (.ifne code true-label)))
          ;; Default: promote both to double, use dcmpg
          (let [t1 (emit-form code a1 locals ctx)
                _ (when (not= t1 :double) (emit-coerce code t1 :double))
                t2 (emit-form code a2 locals ctx)
                _ (when (not= t2 :double) (emit-coerce code t2 :double))]
            (.dcmpg code)
            (case op
              "<"  (.iflt code true-label) "<=" (.ifle code true-label)
              ">"  (.ifgt code true-label) ">=" (.ifge code true-label)
              "==" (.ifeq code true-label) "not=" (.ifne code true-label))))))
    ;; Return Boolean.FALSE/TRUE (not int 0/1) so Clojure truthiness works.
    ;; int 0 is truthy in Clojure, which breaks (and (> j 0) ...) short-circuit.
    (.getstatic code (ClassDesc/of "java.lang" "Boolean") "FALSE"
                (ClassDesc/of "java.lang" "Boolean"))
    (.goto_ code end-label)
    (.labelBinding code true-label)
    (.getstatic code (ClassDesc/of "java.lang" "Boolean") "TRUE"
                (ClassDesc/of "java.lang" "Boolean"))
    (.labelBinding code end-label)
    :ref))

(defn- emit-minmax
  "Emit min/max bytecode. Variadic: (min a b c) → Math.min(Math.min(a, b), c).
   Selects Math.min/max overload based on first arg's actual emitted type:
   int → int, long → long, float → float, else → double."
  [code op args locals ctx]
  (let [t1 (emit-form code (first args) locals ctx)
        target (case t1 :int :int :long :long :float :float :double)
        ret-cd (case target :int I-cd :long (ClassDesc/ofDescriptor "J")
                     :float (ClassDesc/ofDescriptor "F") D-cd)
        mm-mt (MethodTypeDesc/of ret-cd (into-array ClassDesc [ret-cd ret-cd]))]
    (when (not= t1 target) (emit-coerce code t1 target))
    (doseq [arg (rest args)]
      (let [t (emit-form code arg locals ctx)]
        (when (not= t target) (emit-coerce code t target)))
      (.invokestatic code math-cd op mm-mt))
    target))

(defn- emit-aget-intrinsic
  "Emit bytecode for aget (array read). Handles typed array checkcast and
   dispatches to the correct JVM *aload instruction based on arr-tag."
  [code args locals ctx]
  (let [[arr idx] args
        arr-tag (or (:tag (meta arr))
                    (when (symbol? arr) (:hint (get locals arr))))
        ;; Skip checkcast when the local's type is verified by the
        ;; JVM method descriptor (typed param, not Object)
        arr-verified? (when (symbol? arr) (:verified (get locals arr)))
        arr-type (emit-form code arr locals ctx)]
    ;; Checkcast array to correct type when on stack as Object ref.
    ;; Skip when the type is already verified by the method descriptor.
    (when (and (= arr-type :ref) (not arr-verified?))
      (let [arr-cls (case arr-tag
                      objects (Class/forName "[Ljava.lang.Object;")
                      doubles (Class/forName "[D")
                      floats  (Class/forName "[F")
                      longs   (Class/forName "[J")
                      ints    (Class/forName "[I")
                      bytes   (Class/forName "[B")
                      shorts  (Class/forName "[S")
                      ;; Unknown arr-tag: assume Object[] for aaload
                      (Class/forName "[Ljava.lang.Object;"))]
        (.checkcast code (class-desc-of arr-cls))))
    (let [t (emit-form code idx locals ctx)]
      (when (not= t :int) (emit-coerce code t :int)))
    (case arr-tag
      doubles (do (.daload code) :double)
      floats  (do (.faload code) :float)
      longs   (do (.laload code) :long)
      ints    (do (.iaload code) :int)
      bytes   (do (.baload code) :int)
      shorts  (do (.saload code) :int)
      objects (do (.aaload code) :ref)
      (do (when arr-tag
            (binding [*out* *err*]
              (println "WARNING: emit-aget-intrinsic: unrecognized arr-tag" arr-tag
                       "for" arr "— falling back to aaload (Object[])")))
          (.aaload code) :ref))))

(defn- emit-aset-intrinsic
  "Emit bytecode for aset (array write). Handles typed array checkcast,
   void-context optimization (plain store vs temp-local for return value),
   and dispatches to the correct JVM *astore instruction based on arr-tag."
  [code args locals ctx]
  (let [[arr idx val] args
        arr-tag (or (:tag (meta arr))
                    (when (symbol? arr) (:hint (get locals arr))))
        arr-verified? (when (symbol? arr) (:verified (get locals arr)))
        arr-type (emit-form code arr locals ctx)]
    ;; Checkcast array — skip when type verified by method descriptor
    (when (and (= arr-type :ref) (not arr-verified?))
      (let [arr-cls (case arr-tag
                      objects (Class/forName "[Ljava.lang.Object;")
                      doubles (Class/forName "[D")
                      floats  (Class/forName "[F")
                      longs   (Class/forName "[J")
                      ints    (Class/forName "[I")
                      bytes   (Class/forName "[B")
                      shorts  (Class/forName "[S")
                      nil)]
        (when arr-cls (.checkcast code (class-desc-of arr-cls)))))
    (let [t (emit-form code idx locals ctx)]
      (when (not= t :int) (emit-coerce code t :int)))
    ;; In void context (non-last statement), emit plain store (1 instr).
    ;; In value context (result used), use temp local to preserve
    ;; the value across the store (Clojure semantics: aset returns val).
    (let [void? (:void-context ctx)]
      (case arr-tag
        doubles (let [t (emit-form code val locals ctx)]
                  (when (not= t :double) (emit-coerce code t :double))
                  (if void?
                    (do (.dastore code) :void)
                    (let [tmp @(:next-slot ctx)]
                      (swap! (:next-slot ctx) + 2)
                      (.dstore code tmp) (.dload code tmp)
                      (.dastore code) (.dload code tmp) :double)))
        floats  (let [t (emit-form code val locals ctx)]
                  (when (not= t :float) (emit-coerce code t :float))
                  (if void?
                    (do (.fastore code) :void)
                    (let [tmp @(:next-slot ctx)]
                      (swap! (:next-slot ctx) inc)
                      (.fstore code tmp) (.fload code tmp)
                      (.fastore code) (.fload code tmp) :float)))
        longs   (let [t (emit-form code val locals ctx)]
                  (when (not= t :long) (emit-coerce code t :long))
                  (if void?
                    (do (.lastore code) :void)
                    (let [tmp @(:next-slot ctx)]
                      (swap! (:next-slot ctx) + 2)
                      (.lstore code tmp) (.lload code tmp)
                      (.lastore code) (.lload code tmp) :long)))
        ints    (let [t (emit-form code val locals ctx)]
                  (when (not= t :int) (emit-coerce code t :int))
                  (if void?
                    (do (.iastore code) :void)
                    (let [tmp @(:next-slot ctx)]
                      (swap! (:next-slot ctx) inc)
                      (.istore code tmp) (.iload code tmp)
                      (.iastore code) (.iload code tmp) :int)))
        bytes  (let [t (emit-form code val locals ctx)]
                 (when (not= t :int) (emit-coerce code t :int))
                 (if void?
                   (do (.bastore code) :void)
                   (let [tmp @(:next-slot ctx)]
                     (swap! (:next-slot ctx) inc)
                     (.istore code tmp) (.iload code tmp)
                     (.bastore code) (.iload code tmp) :int)))
        shorts (let [t (emit-form code val locals ctx)]
                 (when (not= t :int) (emit-coerce code t :int))
                 (if void?
                   (do (.sastore code) :void)
                   (let [tmp @(:next-slot ctx)]
                     (swap! (:next-slot ctx) inc)
                     (.istore code tmp) (.iload code tmp)
                     (.sastore code) (.iload code tmp) :int)))
        (do (when (and arr-tag (not= arr-tag 'objects))
              (binding [*out* *err*]
                (println "WARNING: emit-aset-intrinsic: unrecognized arr-tag" arr-tag
                         "for" arr "— falling back to aastore (Object[])")))
            (let [val-type (emit-form code val locals ctx)]
              (when (not= val-type :ref)
                (emit-coerce code val-type :ref))
              (if void?
                (do (.aastore code) :void)
                ;; dup_x2: [arr, idx, val] → [val, arr, idx, val]
                ;; aastore pops [arr, idx, val], leaving [val] as result
                (do (.dup_x2 code) (.aastore code) :ref))))))))

(defn- emit-alength-intrinsic
  "Emit bytecode for alength. Uses arraylength when the array type is known,
   falls back to Array.getLength for untyped references."
  [code args locals ctx]
  (let [arg (first args)
        arg-tag (or (:tag (meta arg))
                    (when (symbol? arg) (:hint (get locals arg))))
        t (emit-form code arg locals ctx)]
    (if (and (= t :ref) arg-tag)
      ;; Known array type from hint — checkcast then arraylength
      (let [arr-cls (cond (= arg-tag 'objects) (Class/forName "[Ljava.lang.Object;")
                          (= arg-tag 'doubles) (Class/forName "[D")
                          (= arg-tag 'floats) (Class/forName "[F")
                          :else nil)]
        (if arr-cls
          (do (.checkcast code (class-desc-of arr-cls))
              (.arraylength code) :int)
          ;; Unknown hint, use Array.getLength
          (do (.invokestatic code
                             (ClassDesc/ofDescriptor "Ljava/lang/reflect/Array;")
                             "getLength"
                             (MethodTypeDesc/of I-cd (into-array ClassDesc [obj-cd])))
              :int)))
      (if (= t :ref)
        ;; No hint at all, use Array.getLength for safety
        (do (.invokestatic code
                           (ClassDesc/ofDescriptor "Ljava/lang/reflect/Array;")
                           "getLength"
                           (MethodTypeDesc/of I-cd (into-array ClassDesc [obj-cd])))
            :int)
        ;; Already a typed array on the stack
        (do (.arraylength code) :int)))))

(defn- emit-core-intrinsic
  "Emit bytecode for clojure.core intrinsic functions."
  [code head-name args locals ctx]
  (case head-name
    ;; Array ops
    "aget"    (emit-aget-intrinsic code args locals ctx)
    "aset"    (emit-aset-intrinsic code args locals ctx)
    "alength" (emit-alength-intrinsic code args locals ctx)
    ;; nth — RT.nth(coll, idx) for vector/seq access
    "nth"          (let [[coll idx] args
                         t-coll (emit-form code coll locals ctx)]
                     (when (= t-coll :ref)
                       (let [t-idx (emit-form code idx locals ctx)]
                         (when (not= t-idx :int) (emit-coerce code t-idx :int))
                         (.invokestatic code rt-cd "nth"
                                        (MethodTypeDesc/of obj-cd (into-array ClassDesc [obj-cd I-cd])))
                         :ref)))
    ;; get — RT.get(map, key) or RT.get(map, key, default)
    "get"          (if (= 3 (count args))
                     (let [[map-form key-form default-form] args]
                       (let [t (emit-form code map-form locals ctx)]
                         (when (primitive? t) (emit-box-to-ref code t)))
                       (let [t (emit-form code key-form locals ctx)]
                         (when (primitive? t) (emit-box-to-ref code t)))
                       (let [t (emit-form code default-form locals ctx)]
                         (when (primitive? t) (emit-box-to-ref code t)))
                       (.invokestatic code rt-cd "get"
                                      (MethodTypeDesc/of obj-cd (into-array ClassDesc [obj-cd obj-cd obj-cd])))
                       :ref)
                     (let [[map-form key-form] args]
                       (let [t (emit-form code map-form locals ctx)]
                         (when (primitive? t) (emit-box-to-ref code t)))
                       (let [t (emit-form code key-form locals ctx)]
                         (when (primitive? t) (emit-box-to-ref code t)))
                       (.invokestatic code rt-cd "get"
                                      (MethodTypeDesc/of obj-cd (into-array ClassDesc [obj-cd obj-cd])))
                       :ref))
    ;; aclone — direct array clone
    "aclone"       (let [arg (first args)
                         arg-tag (or (:tag (meta arg))
                                     (when (symbol? arg) (:hint (get locals arg))))
                         t (emit-form code arg locals ctx)]
                     (let [arr-cd (case arg-tag
                                    doubles dblarr-cd
                                    floats  fltarr-cd
                                    longs   lngarr-cd
                                    ints    intarr-cd
                                    nil)]
                       (if arr-cd
                         (do (when (= t :ref) (.checkcast code arr-cd))
                             (.invokevirtual code arr-cd "clone" (MethodTypeDesc/of obj-cd no-cd))
                             (.checkcast code arr-cd)
                             :ref)
                         (do (.invokevirtual code obj-cd "clone" (MethodTypeDesc/of obj-cd no-cd))
                             :ref))))
    "object-array" (let [arg (first args)]
                     (if (vector? arg)
                       ;; (object-array [a b c]) — create and populate from elements
                       (let [n (count arg)]
                         (.ldc code (int n))
                         (.anewarray code obj-cd)
                         (doseq [[i elem] (map-indexed vector arg)]
                           (.dup code)
                           (.ldc code (int i))
                           (let [t (emit-form code elem locals ctx)]
                             (when (#{:double :float :int :long} t)
                               (emit-box-to-ref code t)))
                           (.aastore code))
                         :ref)
                       ;; (object-array n) — create empty array of size n
                       (do (let [t (emit-form code arg locals ctx)]
                             (when (not= t :int) (emit-coerce code t :int)))
                           (.anewarray code obj-cd) :ref)))
    "double-array" (let [arg (first args)]
                     (cond
                       ;; (double-array [1.0 2.0 3.0]) — literal vector
                       (and (= 1 (count args)) (vector? arg))
                       (let [elts arg
                             n (count elts)]
                         (.ldc code (int n))
                         (.newarray code java.lang.classfile.TypeKind/DOUBLE)
                         (if (every? number? elts)
                           (dotimes [i n]
                             (.dup code)
                             (.ldc code (int i))
                             (.ldc code (double (nth elts i)))
                             (.dastore code))
                           (dotimes [i n]
                             (.dup code)
                             (.ldc code (int i))
                             (let [t (emit-form code (nth elts i) locals ctx)]
                               (when (not= t :double) (emit-coerce code t :double)))
                             (.dastore code)))
                         :ref)
                       ;; (double-array n init) — size + fill value
                       (= 2 (count args))
                       (do (let [t (emit-form code arg locals ctx)]
                             (when (not= t :int) (emit-coerce code t :int)))
                           (.newarray code java.lang.classfile.TypeKind/DOUBLE)
                           ;; Stack: array. Dup, emit fill value, call Arrays/fill
                           (.dup code)
                           (let [t (emit-form code (second args) locals ctx)]
                             (when (not= t :double) (emit-coerce code t :double)))
                           (.invokestatic code
                                          (ClassDesc/of "java.util.Arrays") "fill"
                                          (MethodTypeDesc/of V-cd (into-array ClassDesc [dblarr-cd D-cd])))
                           :ref)
                       ;; (double-array n) — size-only, zero-filled
                       :else
                       (do (let [t (emit-form code arg locals ctx)]
                             (when (not= t :int) (emit-coerce code t :int)))
                           (.newarray code java.lang.classfile.TypeKind/DOUBLE) :ref)))
    "float-array"  (let [arg (first args)]
                     (cond
                       ;; (float-array [1.0 2.0 3.0]) — literal vector
                       (and (= 1 (count args)) (vector? arg))
                       (let [elts arg
                             n (count elts)]
                         (.ldc code (int n))
                         (.newarray code java.lang.classfile.TypeKind/FLOAT)
                         (if (every? number? elts)
                           (dotimes [i n]
                             (.dup code)
                             (.ldc code (int i))
                             (.ldc code (float (nth elts i)))
                             (.fastore code))
                           (dotimes [i n]
                             (.dup code)
                             (.ldc code (int i))
                             (let [t (emit-form code (nth elts i) locals ctx)]
                               (when (not= t :float) (emit-coerce code t :float)))
                             (.fastore code)))
                         :ref)
                       ;; (float-array n init) — size + fill value
                       (= 2 (count args))
                       (do (let [t (emit-form code arg locals ctx)]
                             (when (not= t :int) (emit-coerce code t :int)))
                           (.newarray code java.lang.classfile.TypeKind/FLOAT)
                           (.dup code)
                           (let [t (emit-form code (second args) locals ctx)]
                             (when (not= t :float) (emit-coerce code t :float)))
                           (.invokestatic code
                                          (ClassDesc/of "java.util.Arrays") "fill"
                                          (MethodTypeDesc/of V-cd (into-array ClassDesc [fltarr-cd F-cd])))
                           :ref)
                       ;; (float-array n) — size-only, zero-filled
                       :else
                       (do (let [t (emit-form code arg locals ctx)]
                             (when (not= t :int) (emit-coerce code t :int)))
                           (.newarray code java.lang.classfile.TypeKind/FLOAT) :ref)))
    ;; Casts
    "double" (let [t (emit-form code (first args) locals ctx)]
               (if (= t :double) t (do (emit-coerce code t :double) :double)))
    "float"  (let [t (emit-form code (first args) locals ctx)]
               (if (= t :float) t (do (emit-coerce code t :float) :float)))
    "int"    (let [t (emit-form code (first args) locals ctx)]
               (if (= t :int) t (do (emit-coerce code t :int) :int)))
    "long"   (let [t (emit-form code (first args) locals ctx)]
               (if (= t :long) t (do (emit-coerce code t :long) :long)))
    ;; Arithmetic — dispatched to emit-arithmetic helper
    ("+" "-" "*" "/") (emit-arithmetic code head-name args locals ctx)
    ;; Inc/dec (polymorphic, including unchecked variants)
    ;; NOTE: IINC optimization is NOT safe here because (unchecked-inc-int i)
    ;; may appear in a multi-arg recur where other args reference i.
    ;; IINC modifies i in-place before other args are emitted.
    ;; IINC should only be used in the recur handler for single-arg cases.
    ("inc" "dec" "unchecked-inc" "unchecked-dec" "unchecked-inc-int" "unchecked-dec-int")
    (let [t    (emit-form code (first args) locals ctx)
          inc? (contains? #{"inc" "unchecked-inc" "unchecked-inc-int"} head-name)]
      (case t
        :int    (do (.ldc code (int 1))  (if inc? (.iadd code) (.isub code)) :int)
        :long   (do (.ldc code (long 1)) (if inc? (.ladd code) (.lsub code)) :long)
        :double (do (.ldc code 1.0)      (if inc? (.dadd code) (.dsub code)) :double)
        :float  (do (.ldc code (float 1.0)) (if inc? (.fadd code) (.fsub code)) :float)
        :ref    (do (emit-coerce code :ref :long)
                    (.ldc code (long 1)) (if inc? (.ladd code) (.lsub code)) :long)))
    ;; Comparisons → int 0/1 — dispatched to emit-comparison helper
    ("<" "<=" ">" ">=" "==" "not=") (emit-comparison code head-name args locals ctx)
    ;; min/max — dispatched to emit-minmax helper
    ("min" "max") (emit-minmax code head-name args locals ctx)
    ;; str — StringBuilder chain
    "str" (let [sb-cd (ClassDesc/ofDescriptor "Ljava/lang/StringBuilder;")
                str-cd (ClassDesc/ofDescriptor "Ljava/lang/String;")]
            (.new_ code sb-cd)
            (.dup code)
            (.invokespecial code sb-cd "<init>" (MethodTypeDesc/of V-cd no-cd))
            (doseq [a args]
              (let [t (emit-form code a locals ctx)]
                (when (primitive? t) (emit-box-to-ref code t)))
              (.invokevirtual code sb-cd "append"
                              (MethodTypeDesc/of sb-cd (into-array ClassDesc [obj-cd]))))
            (.invokevirtual code sb-cd "toString" (MethodTypeDesc/of str-cd no-cd))
            :ref)
    ;; count — for strings (.length), for collections RT.count
    "count" (let [arg (first args)
                  t (emit-form code arg locals ctx)]
              (if (and (= t :ref)
                       (or (string? arg)
                           (let [tag (or (:tag (meta arg))
                                         (when (symbol? arg) (:hint (get locals arg))))]
                             (= tag 'String))))
                ;; Known string → .length()
                (do (.checkcast code (ClassDesc/ofDescriptor "Ljava/lang/String;"))
                    (.invokevirtual code (ClassDesc/ofDescriptor "Ljava/lang/String;")
                                    "length" (MethodTypeDesc/of I-cd no-cd))
                    :int)
                ;; Generic → RT.count
                (do (when (primitive? t) (emit-box-to-ref code t))
                    (.invokestatic code rt-cd "count"
                                   (MethodTypeDesc/of I-cd (into-array ClassDesc [obj-cd])))
                    :int)))
    ;; ---- Unchecked integer ops ----
    ("unchecked-add-int" "unchecked-subtract-int" "unchecked-multiply-int"
                         "unchecked-divide-int" "unchecked-negate-int")
    (case head-name
      "unchecked-negate-int"
      (let [t (emit-form code (first args) locals ctx)]
        (when (not= t :int) (emit-coerce code t :int))
        (.ineg code) :int)
      ;; binary ops
      (let [t1 (emit-form code (first args) locals ctx)]
        (when (not= t1 :int) (emit-coerce code t1 :int))
        (let [t2 (emit-form code (second args) locals ctx)]
          (when (not= t2 :int) (emit-coerce code t2 :int)))
        (case head-name
          "unchecked-add-int"      (do (.iadd code) :int)
          "unchecked-subtract-int" (do (.isub code) :int)
          "unchecked-multiply-int" (do (.imul code) :int)
          "unchecked-divide-int"   (do (.idiv code) :int))))

    ("unchecked-add" "unchecked-subtract" "unchecked-multiply" "unchecked-negate")
    (case head-name
      "unchecked-negate"
      (let [t (emit-form code (first args) locals ctx)]
        (when (not= t :long) (emit-coerce code t :long))
        (.lneg code) :long)
      (let [t1 (emit-form code (first args) locals ctx)]
        (when (not= t1 :long) (emit-coerce code t1 :long))
        (let [t2 (emit-form code (second args) locals ctx)]
          (when (not= t2 :long) (emit-coerce code t2 :long)))
        (case head-name
          "unchecked-add"      (do (.ladd code) :long)
          "unchecked-subtract" (do (.lsub code) :long)
          "unchecked-multiply" (do (.lmul code) :long))))

    ;; ---- Bitwise operations ----
    ("bit-and" "bit-or" "bit-xor")
    (let [t1 (emit-form code (first args) locals ctx)
          use-long? (= t1 :long)]
      (if use-long?
        (do (let [t2 (emit-form code (second args) locals ctx)]
              (when (not= t2 :long) (emit-coerce code t2 :long)))
            (case head-name
              "bit-and" (.land code)
              "bit-or"  (.lor code)
              "bit-xor" (.lxor code))
            :long)
        (do (when (not= t1 :int) (emit-coerce code t1 :int))
            (let [t2 (emit-form code (second args) locals ctx)]
              (when (not= t2 :int) (emit-coerce code t2 :int)))
            (case head-name
              "bit-and" (.iand code)
              "bit-or"  (.ior code)
              "bit-xor" (.ixor code))
            :int)))

    "bit-not"
    (let [t (emit-form code (first args) locals ctx)]
      (if (= t :long)
        (do (.ldc code (long -1)) (.lxor code) :long)
        (do (when (not= t :int) (emit-coerce code t :int))
            (.ldc code (int -1)) (.ixor code) :int)))

    ("bit-shift-left" "bit-shift-right" "unsigned-bit-shift-right")
    (let [_ (when (not= 2 (count args))
              (throw (ex-info (str head-name " requires exactly 2 args, got " (count args))
                              {:op head-name :arg-count (count args)})))
          t1 (emit-form code (first args) locals ctx)
          use-long? (= t1 :long)]
      (if use-long?
        (do (let [t2 (emit-form code (second args) locals ctx)]
              (when (not= t2 :int) (emit-coerce code t2 :int)))
            (case head-name
              "bit-shift-left"          (.lshl code)
              "bit-shift-right"         (.lshr code)
              "unsigned-bit-shift-right" (.lushr code))
            :long)
        (do (when (not= t1 :int) (emit-coerce code t1 :int))
            (let [t2 (emit-form code (second args) locals ctx)]
              (when (not= t2 :int) (emit-coerce code t2 :int)))
            (case head-name
              "bit-shift-left"          (.ishl code)
              "bit-shift-right"         (.ishr code)
              "unsigned-bit-shift-right" (.iushr code))
            :int)))

    ;; ---- mod / rem / quot ----
    "rem" (let [_ (when (not= 2 (count args))
                    (throw (ex-info (str "rem requires exactly 2 args, got " (count args))
                                    {:op "rem" :arg-count (count args)})))
                t1 (emit-form code (first args) locals ctx)
                use-long? (= t1 :long)]
            (if use-long?
              (do (let [t2 (emit-form code (second args) locals ctx)]
                    (when (not= t2 :long) (emit-coerce code t2 :long)))
                  (.lrem code) :long)
              (if (= t1 :double)
                (do (let [t2 (emit-form code (second args) locals ctx)]
                      (when (not= t2 :double) (emit-coerce code t2 :double)))
                    (.drem code) :double)
                (if (= t1 :float)
                  (do (let [t2 (emit-form code (second args) locals ctx)]
                        (when (not= t2 :float) (emit-coerce code t2 :float)))
                      (.frem code) :float)
                  (do (when (not= t1 :int) (emit-coerce code t1 :int))
                      (let [t2 (emit-form code (second args) locals ctx)]
                        (when (not= t2 :int) (emit-coerce code t2 :int)))
                      (.irem code) :int)))))

    "quot" (let [_ (when (not= 2 (count args))
                     (throw (ex-info (str "quot requires exactly 2 args, got " (count args))
                                     {:op "quot" :arg-count (count args)})))
                 t1 (emit-form code (first args) locals ctx)]
             (case t1
               :long (do (let [t2 (emit-form code (second args) locals ctx)]
                           (when (not= t2 :long) (emit-coerce code t2 :long)))
                         (.ldiv code) :long)
               :double (do (let [t2 (emit-form code (second args) locals ctx)]
                             (when (not= t2 :double) (emit-coerce code t2 :double)))
                           (.ddiv code) (.d2l code) (.l2d code) :double)
               :float (do (let [t2 (emit-form code (second args) locals ctx)]
                            (when (not= t2 :float) (emit-coerce code t2 :float)))
                          (.fdiv code) (.f2i code) (.i2f code) :float)
               (do (when (not= t1 :int) (emit-coerce code t1 :int))
                   (let [t2 (emit-form code (second args) locals ctx)]
                     (when (not= t2 :int) (emit-coerce code t2 :int)))
                   (.idiv code) :int)))

    ;; Delegate remaining intrinsics to sub-function
    (emit-predicate-intrinsic code head-name args locals ctx)))

;; ================================================================
;; emit-fn-call — function/method call dispatch
;; ================================================================

(defn- is-clojure-core? [head ctx]
  (and (symbol? head)
       (let [src-ns (or (:source-ns ctx) *ns*)]
         (or (= (namespace head) "clojure.core")
             (and (nil? (namespace head))
                  (when-let [v (ns-resolve src-ns head)]
                    (and (var? v)
                         (= (the-ns 'clojure.core) (:ns (meta v))))))))))

(defn- emit-deftm-invokestatic
  "Emit invokestatic for a compiled deftm method with typed argument coercions."
  [code bc-info caller-args locals ctx]
  (let [{:keys [class-desc method-type return-stack-type]} bc-info]
    (doseq [[arg i] (map vector caller-args (range (.parameterCount method-type)))]
      (let [t         (emit-form code arg locals ctx)
            target-cd (.parameterType method-type i)
            target-st (cond
                        (.equals target-cd D-cd) :double
                        (.equals target-cd J-cd) :long
                        (.equals target-cd I-cd) :int
                        (.equals target-cd F-cd) :float
                        :else :ref)]
        (if (and (= t :ref) (= target-st :ref) (not (.equals target-cd obj-cd)))
          (.checkcast code target-cd)
          (when (not= t target-st)
            (emit-coerce code t target-st)))))
    (.invokestatic code class-desc "invoke" method-type)
    return-stack-type))

(defn- emit-java-static-call
  "Emit bytecode for Java static method calls (e.g. Math/sin, System/arraycopy).
  Resolves the class and method via reflection, disambiguates overloads by
  matching inferred arg stack types and :hint metadata, and emits invokestatic."
  [code head head-name args locals ctx]
  (let [s         (str head)
        slash-idx (.indexOf s "/")
        cls-short (subs s 0 slash-idx)
        meth-name (subs s (inc slash-idx))
        src-ns    (or (:source-ns ctx) *ns*)
        cls       (or (try (Class/forName cls-short) (catch Exception _ nil))
                      (when-let [resolved (ns-resolve src-ns (symbol cls-short))]
                        (when (class? resolved) resolved))
                      (throw (ex-info (str "Cannot resolve class: " cls-short) {:symbol head})))
        candidates (filterv #(and (= (.getName ^java.lang.reflect.Method %) meth-name)
                                  (= (.getParameterCount ^java.lang.reflect.Method %) (count args)))
                            (.getMethods cls))
          ;; When multiple overloads match, select the one whose param types
          ;; best match the inferred arg types (prevents Math.min(II) vs (DD)).
          ;; Two-phase: 1) stack-type match, 2) hint-based match for ref types
          ;; (e.g. MemorySegment.ofArray has byte[], double[], etc. overloads).
        arg-types (mapv #(infer-arg-stack-type % locals) args)
        stack-matched (filterv (fn [^java.lang.reflect.Method m]
                                 (every? true?
                                         (map (fn [^Class pt at]
                                                (or (nil? at) (= at (class-type->stack pt))))
                                              (.getParameterTypes m) arg-types)))
                               candidates)
        method    (cond
                    (<= (count candidates) 1) (first candidates)
                    (= 1 (count stack-matched)) (first stack-matched)
                    ;; Ambiguous ref types — use :hint from locals for precise matching
                    :else
                    (let [hint-resolve
                          (fn [arg]
                            (when (symbol? arg)
                              (when-let [h (:hint (get locals arg))]
                                (case h
                                  doubles (Class/forName "[D")
                                  floats  (Class/forName "[F")
                                  longs   (Class/forName "[J")
                                  ints    (Class/forName "[I")
                                  objects (Class/forName "[Ljava.lang.Object;")
                                  (try (Class/forName (str h)) (catch Exception _ nil))))))
                          arg-classes (mapv hint-resolve args)]
                      (or (first (filter (fn [^java.lang.reflect.Method m]
                                           (every? true?
                                                   (map (fn [^Class pt ac]
                                                          (or (nil? ac) (.isAssignableFrom pt ac)))
                                                        (.getParameterTypes m) arg-classes)))
                                         (if (seq stack-matched) stack-matched candidates)))
                          (first stack-matched)
                          (first candidates))))]
    (if (and (nil? method) (zero? (count args)))
      ;; No method with 0 args: try static field access
      (let [field (try (.getField cls meth-name) (catch Exception _ nil))]
        (if field
          (let [^Class ftype (.getType field)]
            (.getstatic code (class-desc-of cls) meth-name (class-desc-of ftype))
            (class-type->stack ftype))
          (throw (ex-info (str "Cannot resolve static method or field: " head)
                          {:class (.getName cls) :method meth-name :arity 0}))))
      (do
        (when-not method
          (throw (ex-info (str "Cannot resolve static method: " head)
                          {:class (.getName cls) :method meth-name :arity (count args)})))
        (let [param-types (.getParameterTypes method)]
          (doseq [[arg ^Class pt] (map vector args param-types)]
            (let [t      (emit-form code arg locals ctx)
                  target (class-type->stack pt)]
              (if (and (= t :ref) (= target :ref) (not (.equals pt Object)))
                (.checkcast code (class-desc-of pt))
                (when (not= t target) (emit-coerce code t target))))))
        (let [^Class ret-class (.getReturnType method)
              mt (MethodTypeDesc/of (class-desc-of ret-class)
                                    (into-array ClassDesc (mapv class-desc-of (.getParameterTypes method))))]
          ;; Interface static methods require the isInterface flag
          (if (.isInterface cls)
            (.invokestatic code (class-desc-of cls) meth-name mt true)
            (.invokestatic code (class-desc-of cls) meth-name mt))
          ;; Void methods → push nil (matches Clojure semantics)
          (let [st (class-type->stack ret-class)]
            (if (= st :void) (do (.aconst_null code) :ref) st)))))))

(defn- emit-deftm-call
  "Emit bytecode for a mangled deftm call using a three-tier dispatch strategy:
  1. Already compiled to bytecode → invokestatic via bc-info metadata
  2. Fallback static-call lowering when bytecode metadata is absent
  3. Compile deftm body on demand → invokestatic"
  [code head head-name args locals ctx]
  (let [v  (ns-resolve (or (:source-ns ctx) *ns*) head)
        m  (meta v)]
      ;; 1. Already compiled to bytecode → invokestatic
    (if-let [bc-info (:raster.core/deftm-bytecode m)]
      (emit-deftm-invokestatic code bc-info args locals ctx)
        ;; 2. Fallback static-call lowering when bytecode metadata is absent
      (let [si (resolve-static-call
                (:raster.core/deftm-walked-body m) (:raster.core/deftm-params m)
                (:raster.core/deftm-tags m) (:raster.core/return-tag m))]
        (if si
          (let [{:keys [class method method-type deftm-call-args java-param-types]} si
                param-map (zipmap (:raster.core/deftm-params m) args)]
            (doseq [[call-arg ^Class jpt] (map vector deftm-call-args
                                               (or java-param-types (repeat nil)))]
              (let [resolved-arg (if (and (symbol? call-arg) (get param-map call-arg))
                                   (get param-map call-arg)
                                   (postwalk-preserving
                                    (fn [form]
                                      (if (and (symbol? form) (get param-map form))
                                        (let [replacement (get param-map form)
                                              orig-meta (meta form)]
                                          (if (and orig-meta
                                                   (instance? clojure.lang.IObj replacement))
                                            (with-meta replacement
                                              (merge (meta replacement) orig-meta))
                                            replacement))
                                        form))
                                    call-arg))
                    t (emit-form code resolved-arg locals ctx)]
                (cond
                  (and jpt (.equals jpt Double/TYPE) (not= t :double))
                  (emit-coerce code t :double)
                  (and jpt (not (.isPrimitive jpt)) (= t :ref) (not (.equals jpt Object)))
                  (.checkcast code (class-desc-of jpt)))))
            (.invokestatic code class method method-type)
            (let [ret-str (str (.descriptorString (.returnType method-type)))]
              (case (first ret-str)
                \D :double \J :long \I :int \V :void :ref)))
            ;; 3. Compile deftm body on demand → invokestatic
          (let [bc-info (compile-and-cache-deftm! v)]
            (emit-deftm-invokestatic code bc-info args locals ctx)))))))

(defn- emit-fn-call
  "Emit bytecode for function calls (Java static, deftm, sibling, var, local IFn)."
  [code head head-name args locals ctx]
  ;; Try clojure.core intrinsic first (returns nil for unhandled fns → fallthrough)
  (or (when (is-clojure-core? head ctx)
        (emit-core-intrinsic code head-name args locals ctx))
      (cond
    ;; .method sugar → rewrite to (. obj method args...) and recurse
        (and (symbol? head) (.startsWith (str head) ".") (not= (str head) "."))
        (let [meth-sym (symbol (subs (str head) 1))
              [obj & margs] args]
          (emit-form code (apply list '. obj meth-sym margs) locals ctx))

    ;; ---- Java static method: ClassName/method ----
    ;; Skip if it resolves to a Clojure var (e.g. raster.nn/dense_m_doubles_doubles_doubles)
        (and (symbol? head)
             (let [s (str head)]
               (and (.contains s "/") (not (.startsWith s "."))
                    (not (when-let [v (or (ns-resolve (or (:source-ns ctx) *ns*) head)
                                         ;; Fully qualified: resolve namespace directly
                                          (when-let [ns-sym (namespace head)]
                                            (when-let [ns-obj (find-ns (symbol ns-sym))]
                                              (ns-resolve ns-obj (symbol (name head))))))]
                           (var? v))))))
        (emit-java-static-call code head head-name args locals ctx)

    ;; ---- Typed deftm sibling method (same-class, JIT-friendly) ----
        (and (symbol? head) (:typed-sibling-methods ctx)
             (get (:typed-sibling-methods ctx) (name head)))
        (let [{:keys [method-name method-type return-stack-type param-infos]}
              (get (:typed-sibling-methods ctx) (name head))]
          (doseq [[arg info] (map vector args param-infos)]
            (let [t (emit-form code arg locals ctx)
                  target-st (:stack-type info)
                  target-cd (:class-desc info)]
              (if (and (= t :ref) (= target-st :ref) (not (.equals target-cd obj-cd)))
                (.checkcast code target-cd)
                (when (not= t target-st)
                  (emit-coerce code t target-st)))))
          (.invokestatic code (:class-desc ctx) method-name method-type)
          return-stack-type)

    ;; ---- Mangled deftm call → three-tier dispatch ----
    ;; Skip ^:no-inline deftms (BLAS, etc.) — call via IFn fallback
        (and (symbol? head)
             (when-let [v (ns-resolve (or (:source-ns ctx) *ns*) head)]
               (and (:raster.core/deftm-walked-body (meta v))
                    (not (no-inline-deftm? v)))))
        (emit-deftm-call code head head-name args locals ctx)

    ;; ---- Sibling method call ----
        (and (symbol? head) (get (:sibling-methods ctx) (str head)))
        (let [{:keys [method-name method-type param-descs]} (get (:sibling-methods ctx) (str head))]
          (doseq [[a pd] (map vector args (concat param-descs (repeat obj-cd)))]
            (let [t (emit-form code a locals ctx)
                  target-st (cond (.equals ^ClassDesc pd D-cd) :double
                                  (.equals ^ClassDesc pd J-cd) :long
                                  (.equals ^ClassDesc pd I-cd) :int
                                  (.equals ^ClassDesc pd F-cd) :float
                                  :else :ref)]
              (if (and (= t :ref) (= target-st :ref) (not (.equals ^ClassDesc pd obj-cd)))
                (.checkcast code pd)
                (when (not= t target-st)
                  (emit-coerce code t target-st)))))
          (.invokestatic code (:class-desc ctx) method-name method-type)
          ;; Return type from method descriptor
          (let [ret-cd (.returnType method-type)]
            (cond (.equals ^ClassDesc ret-cd D-cd) :double
                  (.equals ^ClassDesc ret-cd J-cd) :long
                  (.equals ^ClassDesc ret-cd I-cd) :int
                  (.equals ^ClassDesc ret-cd F-cd) :float
                  (.equals ^ClassDesc ret-cd (ClassDesc/ofDescriptor "V")) :void
                  :else :ref)))

    ;; ---- Bytecode-compiled var (has ::bytecode-class metadata) ----
        (and (symbol? head)
             (when-let [v (ns-resolve (or (:source-ns ctx) *ns*) head)]
               (:raster.core/bytecode-class (meta v))))
        (let [v   (ns-resolve (or (:source-ns ctx) *ns*) head)
              m   (meta v)
              bc-class (:raster.core/bytecode-class m)
              bc-method ^java.lang.reflect.Method (:raster.core/bytecode-method m)
              bc-cls-cd (class-desc-of bc-class)
              bc-meth-name (.getName bc-method)
              param-types (.getParameterTypes bc-method)
              param-cds (mapv class-desc-of param-types)
              ret-cd-actual (class-desc-of (.getReturnType bc-method))
              bc-mt (MethodTypeDesc/of ret-cd-actual (into-array ClassDesc param-cds))]
          (doseq [[a pd] (map vector args param-cds)]
            (let [t (emit-form code a locals ctx)
                  target-st (cond (.equals ^ClassDesc pd D-cd) :double
                                  (.equals ^ClassDesc pd J-cd) :long
                                  (.equals ^ClassDesc pd I-cd) :int
                                  (.equals ^ClassDesc pd F-cd) :float
                                  :else :ref)]
              (cond
            ;; Box primitives to Object when method expects Object
                (and (primitive? t)
                     (or (nil? pd) (= pd obj-cd)))
                (emit-box-to-ref code t)
            ;; Coerce to primitive when method expects primitive
                (and (not= t target-st) (not= target-st :ref))
                (emit-coerce code t target-st)
            ;; Checkcast ref to specific ref type when method expects specific ref type
                (and (= t :ref) (= target-st :ref) (not= pd obj-cd))
                (.checkcast code pd))))
          (.invokestatic code bc-cls-cd bc-meth-name bc-mt)
          ;; Return stack type from actual method return type
          (cond (.equals ^ClassDesc ret-cd-actual D-cd) :double
                (.equals ^ClassDesc ret-cd-actual J-cd) :long
                (.equals ^ClassDesc ret-cd-actual I-cd) :int
                (.equals ^ClassDesc ret-cd-actual F-cd) :float
                (.equals ^ClassDesc ret-cd-actual V-cd) :void
                :else :ref))

    ;; ---- Deftype/defvalue factory fn (->ClassName args...) → new ClassName ----
    ;; Emit `new ClassName(...)` instead of var-deref + IFn.invoke.
    ;; Only safe when the class is resolvable from any classloader in the
    ;; DCL chain (e.g., Valhalla value classes defined via defineClass on TCL).
    ;; defrecord classes live in ephemeral eval DCLs that can't be reached
    ;; from the compiled class's classloader — for those, we must fall back
    ;; to var-deref so the factory creates the correct class instance.
        (and (symbol? head)
             (let [hn (name head)]
               (and (.startsWith hn "->")
                    (> (count hn) 2)
                    (when-let [v (ns-resolve (or (:source-ns ctx) *ns*) head)]
                      (when (var? v)
                        (let [cls-name (subs hn 2)
                              full-name (str (ns-name (.ns ^clojure.lang.Var v)) "." cls-name)
                              factory-cl (try (.getClassLoader (class (.getRawRoot ^clojure.lang.Var v)))
                                              (catch Exception _ nil))
                              ;; Resolve via factory's classloader (correct class)
                              cls-via-factory (when factory-cl
                                                (try (Class/forName full-name true factory-cl)
                                                     (catch Exception _ nil)))
                              ;; Resolve via the COMPILATION classloader — this is where
                              ;; the compiled bytecode will live, so its `new DP5()` resolves
                              ;; from this CL at runtime. If this CL can't see the class
                              ;; (defrecord in a sibling DCL branch), fall through to var-deref.
                              comp-cl (compilation-loader)
                              cls-via-comp (try (Class/forName full-name true comp-cl)
                                                (catch Exception _ nil))]
                          ;; Only emit `new` when both resolve to the SAME class.
                          (and cls-via-factory cls-via-comp
                               (identical? cls-via-factory cls-via-comp))))))))
        (let [hn (name head)
              cls-name (subs hn 2)
              src-ns (or (:source-ns ctx) *ns*)
              v (ns-resolve src-ns head)
              full-name (str (ns-name (.ns ^clojure.lang.Var v)) "." cls-name)
              factory-cl (.getClassLoader (class (.getRawRoot ^clojure.lang.Var v)))
              cls (Class/forName full-name true factory-cl)
              cls-cd (class-desc-of cls)]
          (.new_ code cls-cd)
          (.dup code)
          (let [ctor (first (filter #(= (.getParameterCount %) (count args))
                                    (.getConstructors cls)))
                param-types (.getParameterTypes ctor)]
            (doseq [[arg ^Class pt] (map vector args param-types)]
              (let [t      (emit-form code arg locals ctx)
                    target (class-type->stack pt)]
                (when (not= t target) (emit-coerce code t target))))
            (.invokespecial code cls-cd "<init>"
                            (MethodTypeDesc/of V-cd (into-array ClassDesc (mapv class-desc-of param-types)))))
          :ref)

    ;; ---- Fallback: Clojure var call via IFn.invoke ----
        (and (symbol? head)
             (when-let [v (or (ns-resolve (or (:source-ns ctx) *ns*) head)
                              (when-let [ns-sym (namespace head)]
                                (when-let [ns-obj (find-ns (symbol ns-sym))]
                                  (ns-resolve ns-obj (symbol (name head))))))]
               (var? v)))
        (let [v (or (ns-resolve (or (:source-ns ctx) *ns*) head)
                    (when-let [ns-sym (namespace head)]
                      (when-let [ns-obj (find-ns (symbol ns-sym))]
                        (ns-resolve ns-obj (symbol (name head))))))]
          (.ldc code (str (.name (.ns ^clojure.lang.Var v))))
          (.ldc code (str (.sym ^clojure.lang.Var v)))
          (.invokestatic code rt-cd "var" (MethodTypeDesc/of var-cd (into-array ClassDesc [(ClassDesc/ofDescriptor "Ljava/lang/String;") (ClassDesc/ofDescriptor "Ljava/lang/String;")])))
          (.invokevirtual code var-cd "getRawRoot" (MethodTypeDesc/of obj-cd no-cd))
          (.checkcast code ifn-cd)
          (doseq [a args]
            (let [t (emit-form code a locals ctx)]
              (when (primitive? t) (emit-box-to-ref code t))))
          (let [n (count args)
                invoke-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc (repeat n obj-cd)))]
            (.invokeinterface code ifn-cd "invoke" invoke-mt))
          :ref)

    ;; ---- Local variable as IFn (including Fn-typed params) ----
    ;; Uses boxed IFn.invoke to handle both typed ftm and plain defn callers.
    ;; C2 profiles monomorphic call sites and devirtualizes after warmup.
        (and (symbol? head) (get locals head))
        (let [{:keys [slot]} (get locals head)]
          (.aload code slot)
          (.checkcast code ifn-cd)
          (doseq [a args]
            (let [t (emit-form code a locals ctx)]
              (when (primitive? t) (emit-box-to-ref code t))))
          (let [n (count args)
                invoke-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc (repeat n obj-cd)))]
            (.invokeinterface code ifn-cd "invoke" invoke-mt))
          :ref)

    ;; ---- Expression in call position: ((f x) y z) ----
    ;; The head is a sub-expression that evaluates to an IFn.
    ;; Emit it, checkcast to IFn, invoke with args.
        :else
        (let [ht (emit-form code head locals ctx)]
          (when (primitive? ht) (emit-box-to-ref code ht))
          (.checkcast code ifn-cd)
          (doseq [a args]
            (let [t (emit-form code a locals ctx)]
              (when (primitive? t) (emit-box-to-ref code t))))
          (let [n (count args)
                invoke-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc (repeat n obj-cd)))]
            (.invokeinterface code ifn-cd "invoke" invoke-mt))
          :ref))))

;; ================================================================
;; emit-form — the core bytecode emitter
;; ================================================================
;;
;; Input:  a macroexpanded, walked Clojure form (only special forms)
;; Output: the JVM stack type left on the operand stack
;;
;; Split into handler functions to stay under the JVM 64KB method limit.

(declare emit-form)

(defn- alloc-slot!
  "Allocate a local variable slot in the current method. Double/long take 2 slots."
  [ctx type]
  (let [s @(:next-slot ctx)]
    (swap! (:next-slot ctx) + (if (two-slot? type) 2 1))
    s))

;; ---------------------------------------------------------------------------
;; Extracted emit-form handlers (each compiles to its own JVM class method,
;; keeping emit-form itself under the 64KB method size limit).
;; ---------------------------------------------------------------------------

(defn- emit-let*-handler
  "Emit bytecode for (let* [bindings...] body...).
  Handles type inference for let bindings including:
  - Explicit :tag metadata, alias inheritance, deftm return-tag
  - Java static field/method return types
  - Instance method return types
  - CHECKCAST for typed bindings (arrays, records, etc.)"
  [code args locals ctx]
  (let [[bindings & body] args
        pairs (partition 2 bindings)
        binding-idx (atom 0)
        new-locals (reduce
                    (fn [locs [sym init]]
                      (swap! binding-idx inc)
                      ;; Emit line number from source metadata if available,
                      ;; otherwise use binding index as synthetic line number
                      (let [line-num (or (:line (meta init))
                                         (:line (meta sym))
                                         @binding-idx)]
                        (try (.lineNumber code (int line-num))
                             (catch Exception _)))
                      ;; Binding inits must always produce a value (for the local
                      ;; variable slot), so strip :void-context from ctx.
                      (let [t (emit-form code init locs (dissoc ctx :void-context))
                             ;; Handle type hints on let bindings
                             ;; ^objects → checkcast [Ljava/lang/Object;
                             ;; ^doubles → checkcast [D
                             ;; ^ClassName → checkcast to that class
                            ;; Infer tag: explicit hint, alias source, deftm return-tag,
                            ;; .invk return-tag, or Java static field/method return type
                            tag (or (:tag (meta sym))
                                    ;; Alias: inherit from source
                                    (when (symbol? init)
                                      (:hint (get locs init)))
                                    ;; Call to deftm: use return-tag from var metadata
                                    (when (and (seq? init) (symbol? (first init)))
                                      (try (when-let [v (ns-resolve (or (:source-ns ctx) *ns*) (first init))]
                                             (when (var? v)
                                               (:raster.core/return-tag (meta v))))
                                           (catch Exception _ nil)))
                                    ;; .invk call: read return-tag from walker metadata on impl symbol
                                    (when (and (seq? init) (= '.invk (first init))
                                               (symbol? (second init)))
                                      (:raster.type/ret-tag (meta (second init))))
                                    ;; Java static field/method return type (ClassName/name)
                                    ;; Only for actual Java classes (not Clojure vars)
                                    (when (and (seq? init) (symbol? (first init))
                                               (let [s (str (first init))]
                                                 (and (.contains s "/") (not (.startsWith s "."))))
                                               ;; Skip if it resolves to a Clojure var
                                               (not (try (when-let [v (ns-resolve (or (:source-ns ctx) *ns*) (first init))]
                                                           (var? v))
                                                         (catch Exception _ false))))
                                      (try
                                        (let [s (str (first init))
                                              slash (.indexOf ^String s "/")]
                                          (when (pos? slash)
                                            (let [cls-name (subs s 0 slash)
                                                  meth-name (subs s (inc slash))
                                                  src-ns (or (:source-ns ctx) *ns*)
                                                  cls (or (try (Class/forName cls-name) (catch Exception _ nil))
                                                          (when-let [r (ns-resolve src-ns (symbol cls-name))]
                                                            (when (class? r) r)))
                                                  n-args (dec (count init))]
                                              (when cls
                                                (or ;; Static field (0 args)
                                                 (when (zero? n-args)
                                                   (try (symbol (.getName (.getType (.getField cls meth-name))))
                                                        (catch Exception _ nil)))
                                                    ;; Static method
                                                 (when-let [m (first (filter #(and (= (.getName ^java.lang.reflect.Method %) meth-name)
                                                                                   (= n-args (.getParameterCount ^java.lang.reflect.Method %)))
                                                                             (.getMethods cls)))]
                                                   (let [ret (.getReturnType ^java.lang.reflect.Method m)]
                                                     (when-not (.isPrimitive ret)
                                                       (symbol (.getName ret))))))))))
                                        (catch Exception _ nil)))
                                    ;; Instance method: (. obj method args...) or (.method obj args...)
                                    ;; Infer return type from the method's declaring class
                                    (when (and (seq? init) (symbol? (first init)))
                                      (let [h (first init)]
                                        (cond
                                          ;; (.method obj args...)
                                          (and (.startsWith (str h) ".") (not= "." (str h)))
                                          (let [meth-str (subs (str h) 1)
                                                obj-form (second init)
                                                obj-tag (when (symbol? obj-form) (:hint (get locs obj-form)))
                                                src-ns (or (:source-ns ctx) *ns*)
                                                cls (when obj-tag
                                                      (or (try (Class/forName (str obj-tag)) (catch Exception _ nil))
                                                          (when-let [r (ns-resolve src-ns (symbol (str obj-tag)))]
                                                            (when (class? r) r))))]
                                            (when cls
                                              ;; Find most specific return type among matching methods
                                              ;; Java bridge methods return the erased type; prefer
                                              ;; the non-bridge overload with the most specific return
                                              (let [matches (seq (filter #(= (.getName ^java.lang.reflect.Method %) meth-str)
                                                                         (.getMethods cls)))
                                                    best (or (first (remove #(.isBridge ^java.lang.reflect.Method %) matches))
                                                             (first matches))]
                                                (when best
                                                  (let [ret (.getReturnType ^java.lang.reflect.Method best)]
                                                    (when-not (.isPrimitive ret)
                                                      (symbol (.getName ret))))))))
                                          ;; (. obj method args...)
                                          (= '. h)
                                          (let [obj-form (second init)
                                                meth-sym (nth init 2 nil)
                                                obj-tag (when (symbol? obj-form) (:hint (get locs obj-form)))
                                                src-ns (or (:source-ns ctx) *ns*)
                                                cls (when obj-tag
                                                      (or (try (Class/forName (str obj-tag)) (catch Exception _ nil))
                                                          (when-let [r (ns-resolve src-ns (symbol (str obj-tag)))]
                                                            (when (class? r) r))))]
                                            (when (and cls meth-sym)
                                              (let [matches (seq (filter #(= (.getName ^java.lang.reflect.Method %) (str meth-sym))
                                                                         (.getMethods cls)))
                                                    best (or (first (remove #(.isBridge ^java.lang.reflect.Method %) matches))
                                                             (first matches))]
                                                (when best
                                                  (let [ret (.getReturnType ^java.lang.reflect.Method best)]
                                                    (when-not (.isPrimitive ret)
                                                      (symbol (.getName ret))))))))
                                          :else nil)))
                                    ;; (new ClassName ...) → ClassName
                                    (when (and (seq? init) (= 'new (first init)))
                                      (let [cls-sym (second init)]
                                        (when cls-sym (symbol (str cls-sym)))))
                                    ;; Fallback: use infer-arg-tag for if/do/loop/etc.
                                    ;; This covers forms the bytecode handler's
                                    ;; pattern matching doesn't handle.
                                    (try
                                      (let [local-env (into {} (keep (fn [[k v]]
                                                                       (when-let [h (:hint v)]
                                                                         [k h]))
                                                                     locs))]
                                        (inf/infer-arg-tag init local-env))
                                      (catch Throwable _ nil)))
                            ;; Checkcast for typed let bindings (arrays, records, etc.)
                            ;; These CHECKCASTs are redundant when getfield already returns
                            ;; the typed value, but C2 uses them as type proofs for downstream
                            ;; optimizations. Removing them causes 2x regression.
                            ;; Skip IFn__ types — handled by wrapper guard.
                            _ (when (and tag (= t :ref)
                                         (not (typed-interface? tag)))
                                (let [src-ns (or (:source-ns ctx) *ns*)
                                      cls (cond
                                            (= tag 'objects)
                                            (Class/forName "[Ljava.lang.Object;")
                                            (= tag 'doubles)
                                            (Class/forName "[D")
                                            (= tag 'floats)
                                            (Class/forName "[F")
                                            (= tag 'ints)
                                            (Class/forName "[I")
                                            (= tag 'longs)
                                            (Class/forName "[J")
                                            (= tag 'bytes)
                                            (Class/forName "[B")
                                            (= tag 'shorts)
                                            (Class/forName "[S")
                                            :else
                                            (let [tag-str (str tag)]
                                              (when (pos? (count tag-str))
                                                (or (try (Class/forName tag-str)
                                                         (catch Exception _ nil))
                                                    (when-let [r (ns-resolve src-ns (symbol tag-str))]
                                                      (when (class? r) r))))))]
                                  (when cls
                                    (.checkcast code (class-desc-of cls)))))
                            ;; Skip store for void results (effectful calls like Arrays.fill)
                            slot (when-not (#{:void :diverge} t) (alloc-slot! ctx t))]
                        (when slot
                          (case t
                            :double (.dstore code slot)
                            :long   (.lstore code slot)
                            :int    (.istore code slot)
                            :float  (.fstore code slot)
                            (.astore code slot)))
                        (if slot
                          (assoc locs sym {:slot slot :type t :hint tag
                                           ;; Mark as verified if we just checkcast'd it
                                           ;; — skips redundant checkcasts on aget/aset
                                           :verified (boolean (and tag (not= tag 'Object)))})
                          locs)))
                    locals pairs)]
    (emit-body code body new-locals ctx)))

(defn- emit-if-handler
  "Emit bytecode for (if pred then else).
  Handles branch folding for int/long/double comparisons, Clojure truthiness
  for refs, and type merging between branches (boxing only when needed)."
  [code args locals ctx]
  (let [[pred then else] args
        else-label (.newLabel code)
        end-label  (.newLabel code)
        ;; Branch folding: if pred is (< a b), emit IF_ICMPGE directly
        ;; instead of materializing boolean. Critical for C2 counted loop
        ;; recognition which requires IF_ICMPxx at the loop back-edge.
        branch-folded?
        (when (and (seq? pred) (symbol? (first pred)))
          (let [pn (name (first pred))]
            (when (and (contains? #{"<" "<=" ">" ">=" "==" "not="} pn)
                       (= 2 (count (rest pred))))
              (let [a1 (nth pred 1) a2 (nth pred 2)
                    val-ctx (dissoc ctx :void-context)
                    t1 (emit-form code a1 locals val-ctx)
                    t2 (emit-form code a2 locals val-ctx)]
                (cond
                  ;; Both int → IF_ICMPxx (inverted for else branch)
                  (and (= t1 :int) (= t2 :int))
                  (do (case pn "<" (.if_icmpge code else-label) "<=" (.if_icmpgt code else-label)
                            ">" (.if_icmple code else-label) ">=" (.if_icmplt code else-label)
                            "==" (.if_icmpne code else-label) "not=" (.if_icmpeq code else-label))
                      true)
                  ;; Both long → LCMP + IFxx
                  (and (= t1 :long) (= t2 :long))
                  (do (.lcmp code)
                      (case pn "<" (.ifge code else-label) "<=" (.ifgt code else-label)
                            ">" (.ifle code else-label) ">=" (.iflt code else-label)
                            "==" (.ifne code else-label) "not=" (.ifeq code else-label))
                      true)
                  ;; Both double → DCMPG + IFxx
                  (and (= t1 :double) (= t2 :double))
                  (do (.dcmpg code)
                      (case pn "<" (.ifge code else-label) "<=" (.ifgt code else-label)
                            ">" (.ifle code else-label) ">=" (.iflt code else-label)
                            "==" (.ifne code else-label) "not=" (.ifeq code else-label))
                      true)
                  ;; Mixed types → coerce both to double
                  :else
                  (let [;; t2 on top, t1 below. Store t2, coerce t1, reload+coerce t2
                        tmp (let [s @(:next-slot ctx)]
                              (case t2
                                :int    (do (.istore code s) (swap! (:next-slot ctx) inc) s)
                                :float  (do (.fstore code s) (swap! (:next-slot ctx) inc) s)
                                :long   (do (.lstore code s) (swap! (:next-slot ctx) + 2) s)
                                :double (do (.dstore code s) (swap! (:next-slot ctx) + 2) s)
                                ;; :ref — store as Object, will coerce to double on reload
                                (do (.astore code s) (swap! (:next-slot ctx) inc) s)))]
                    (when (not= t1 :double) (emit-coerce code t1 :double))
                    (case t2 :int (do (.iload code tmp) (.i2d code))
                          :long (do (.lload code tmp) (.l2d code))
                          :float (do (.fload code tmp) (.f2d code))
                          :double (.dload code tmp)
                             ;; :ref — reload Object, unbox to double
                          (do (.aload code tmp) (emit-coerce code :ref :double)))
                    (.dcmpg code)
                    (case pn "<" (.ifge code else-label) "<=" (.ifgt code else-label)
                          ">" (.ifle code else-label) ">=" (.iflt code else-label)
                          "==" (.ifne code else-label) "not=" (.ifeq code else-label))
                    true))))))
        ;; Predicate must always produce a value (to branch on),
        ;; so strip :void-context from ctx.
        pred-type (when-not branch-folded?
                    (emit-form code pred locals (dissoc ctx :void-context)))]
    (when-not branch-folded?
      (case pred-type
        :int    (.ifeq code else-label)
      ;; Clojure truthiness: null and Boolean.FALSE are falsy, everything else truthy
        :ref    (let [not-null-label (.newLabel code)]
                  (.dup code)
                  (.ifnonnull code not-null-label)
                ;; null → pop and jump to else
                  (.pop code)
                  (.goto_ code else-label)
                  (.labelBinding code not-null-label)
                ;; Check for Boolean.FALSE
                  (.getstatic code (ClassDesc/ofDescriptor "Ljava/lang/Boolean;")
                              "FALSE" (ClassDesc/ofDescriptor "Ljava/lang/Boolean;"))
                  (.if_acmpeq code else-label))
        :double (do (.ldc code 0.0) (.dcmpl code) (.ifeq code else-label))
        :float  (do (.ldc code (float 0.0)) (.fcmpl code) (.ifeq code else-label))
        :long   (do (.ldc code (long 0)) (.lcmp code) (.ifeq code else-label))))
    (let [then-type (emit-form code then locals ctx)]
      (cond
        (diverge? then-type)
        ;; Then branch diverges (throw/recur) — never falls through.
        ;; No goto needed. Else branch is the only value-producing path.
        (do (.labelBinding code else-label)
            (let [else-type (if else
                              (emit-form code else locals ctx)
                              (do (.aconst_null code) :ref))]
              (when (primitive? else-type)
                (emit-box-to-ref code else-type))
              (.labelBinding code end-label)
              (if (primitive? else-type) :ref else-type)))
        ;; Normal case: both branches produce values.
        ;; Predict else type to avoid unnecessary boxing.
        ;; When both branches return the same primitive type, keeping
        ;; them unboxed is critical: (and (> j 0) expr) expands to
        ;; (let [t (> j 0)] (if t expr t)) — both branches are :int.
        ;; Boxing int 0 to Integer(0) breaks Clojure truthiness
        ;; (Integer(0) is truthy, but int 0 with ifeq is correctly falsy).
        :else
        (let [else-type-pred (cond
                               (nil? else) :ref
                               (symbol? else) (:type (get locals else))
                               :else (infer-arg-stack-type else locals))
              skip-box? (and (primitive? then-type)
                             (= then-type else-type-pred))]
          ;; Handle :void then-branch: push null so both paths have a
          ;; value on the stack at the join point.
          (cond
            (= then-type :void) (.aconst_null code)
            (and (primitive? then-type) (not skip-box?))
            (emit-box-to-ref code then-type))
          (.goto_ code end-label)
          (.labelBinding code else-label)
          (let [else-type (if else
                            (emit-form code else locals ctx)
                            (do (.aconst_null code) :ref))]
            (cond
              ;; Else diverges (recur/throw) — then is the only value path
              (diverge? else-type)
              (do (.labelBinding code end-label)
                  (if (= then-type :void) :ref
                      (if skip-box? then-type
                          (if (primitive? then-type) :ref then-type))))
              ;; Both void — push null for else too (then already has null)
              (and (= then-type :void) (= else-type :void))
              (do (.aconst_null code)
                  (.labelBinding code end-label) :ref)
              ;; Same primitive type confirmed — keep unboxed
              (and skip-box? (= then-type else-type))
              (do (.labelBinding code end-label)
                  then-type)
              ;; :void else outside void-context: push null
              (= else-type :void)
              (do (.aconst_null code)
                  (.labelBinding code end-label)
                  :ref)
              ;; Different types or prediction was wrong — box else
              :else
              (do (when (primitive? else-type)
                    (emit-box-to-ref code else-type))
                  (.labelBinding code end-label)
                  (if skip-box? then-type :ref)))))))))

(defn- emit-dot-handler
  "Emit bytecode for (. obj method args...) — instance method or field access.
  Handles two paths:
  - invokedynamic path: .invk on a namespace-qualified -impl var with IFn__ interface
  - Regular path: invokeinterface/invokevirtual with optional instanceof guard"
  [code args locals ctx]
  (let [[obj member & margs] args
        [meth-name method-args] (if (seq? member)
                                  [(first member) (rest member)]
                                  [member margs])
        meth-str (str meth-name)]
    ;; invokedynamic path: .invk on a namespace-qualified -impl var
    ;; with an IFn__ interface tag. Emits invokedynamic instead of
    ;; RT.var().getRawRoot() + invokeinterface, giving C2 a
    ;; MutableCallSite target it can inline as a constant.
    (if (and (= meth-str "invk")
             (symbol? obj)
             ;; Check for typed interface via metadata (set by walker's stamp-type-meta)
             ;; or via the locals map hint (for function parameters).
             ;; This is the single source of truth — no -impl suffix pattern matching.
             (let [tag (or (:raster.type/tag (meta obj))
                           (:tag (meta obj))
                           (when-let [li (get locals obj)] (:hint li)))]
               (and tag (typed-interface? tag))))
      (let [src-ns (or (:source-ns ctx) *ns*)
            tag (or (:raster.type/tag (meta obj)) (:tag (meta obj))
                    (:hint (get locals obj)))
            cls (or (try (Class/forName (str tag)) (catch Exception _ nil))
                    (when-let [r (ns-resolve src-ns (symbol (str tag)))]
                      (when (class? r) r)))
            _ (when-not cls
                (throw (ex-info (str "Cannot resolve IFn__ interface: " tag)
                                {:obj obj :tag tag})))
            method (first (filter #(and (= (.getName ^java.lang.reflect.Method %) "invk")
                                        (= (.getParameterCount ^java.lang.reflect.Method %) (count method-args)))
                                  (.getMethods cls)))
            _ (when-not method
                (throw (ex-info (str "Cannot resolve invk method on " tag " with " (count method-args) " args")
                                {:obj obj :tag tag :arity (count method-args)})))
            param-types (.getParameterTypes method)
            ^Class ret-class (.getReturnType method)
            ;; Primitive return type from walker (for invokedynamic zero-boxing).
            ;; Only used for the var-reference path; local path keeps Object return.
            ret-tag-meta (:raster.type/ret-tag (meta obj))
            ^Class prim-ret-class (case ret-tag-meta
                                    (double Double) Double/TYPE
                                    (long Long)     Long/TYPE
                                    (int Integer)   Integer/TYPE
                                    (float Float)   Float/TYPE
                                    (void Void)     Void/TYPE
                                    nil)
            is-var? (and (namespace obj) (not (get locals obj)))]
        (if is-var?
          ;; Var reference (e.g. raster.numeric/_plus__m_double_double-impl)
          ;; → emit args first, then invokedynamic (static-like, no receiver).
          ;; Uses primitive return type when available → bootstrap targets compute_prim
          ;; for zero-boxing cross-function calls.
          (let [^Class indy-ret-class (or prim-ret-class ret-class)]
            (doseq [[arg ^Class pt] (map vector method-args param-types)]
              (let [t (emit-form code arg locals ctx)
                    target (class-type->stack pt)]
                (if (not= t target)
                  (emit-coerce code t target)
                  (when (and (= t :ref) (not (.equals pt Object)))
                    (.checkcast code (class-desc-of pt))))))
            (ensure-bootstrap-class!)
            (let [call-mt (MethodTypeDesc/of (class-desc-of indy-ret-class)
                                             (into-array ClassDesc (mapv class-desc-of param-types)))
                  indy-desc (DynamicCallSiteDesc/of
                             bsm-handle
                             "invoke"
                             call-mt
                             (into-array ConstantDesc
                                         [(str (namespace obj))
                                          (str (name obj))]))]
              (.invokedynamic code indy-desc)))
          ;; Local symbol with IFn__ tag.
          ;; If the local was checkcast at method entry (:verified in locals),
          ;; emit direct invokeinterface — no instanceof guard needed.
          ;; Otherwise emit instanceof guard with boxed IFn.invoke fallback.
          (let [local-info (get locals obj)
                verified? (:verified local-info)
                iface-cd  (class-desc-of cls)
                call-mt   (MethodTypeDesc/of (class-desc-of ret-class)
                                             (into-array ClassDesc (mapv class-desc-of param-types)))
                ret-st    (class-type->stack ret-class)]
            (if verified?
              ;; === VERIFIED: direct invokeinterface, no guard ===
              ;; The method signature guarantees f is IFn__. C2 devirtualizes
              ;; via type profiling — zero instanceof overhead.
              (do (emit-form code obj locals ctx)
                  (doseq [[arg ^Class pt] (map vector method-args param-types)]
                    (let [t (emit-form code arg locals ctx)
                          target (class-type->stack pt)]
                      (if (not= t target)
                        (emit-coerce code t target)
                        (when (and (= t :ref) (not (.equals pt Object)))
                          (.checkcast code (class-desc-of pt))))))
                  (.invokeinterface code iface-cd "invk" call-mt))
              ;; === UNVERIFIED: instanceof guard with boxed fallback ===
              (let [fast-label (.newLabel code)
                    end-label  (.newLabel code)]
                (emit-form code obj locals ctx)
                (.dup code)
                (.instanceOf code iface-cd)
                (.ifne code fast-label)
                ;; SLOW PATH: plain IFn.invoke with boxing
                (doseq [arg method-args]
                  (let [t (emit-form code arg locals ctx)]
                    (when (primitive? t) (emit-box-to-ref code t))))
                (let [n-args (count method-args)
                      invoke-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc (repeat n-args obj-cd)))]
                  (.invokeinterface code ifn-cd "invoke" invoke-mt))
                (cond
                  (= ret-st :void) (.pop code)
                  (primitive? ret-st) (emit-unbox-to-prim code ret-st))
                (.goto_ code end-label)
                ;; FAST PATH: typed .invk with primitives
                (.labelBinding code fast-label)
                (.checkcast code iface-cd)
                (doseq [[arg ^Class pt] (map vector method-args param-types)]
                  (let [t (emit-form code arg locals ctx)
                        target (class-type->stack pt)]
                    (if (not= t target)
                      (emit-coerce code t target)
                      (when (and (= t :ref) (not (.equals pt Object)))
                        (.checkcast code (class-desc-of pt))))))
                (.invokeinterface code iface-cd "invk" call-mt)
                (.labelBinding code end-label)))))
        ;; Stack type: primitive for invokedynamic with compute_prim, :ref for local path
        (if (and is-var? prim-ret-class)
          (class-type->stack prim-ret-class)
          (class-type->stack ret-class)))
      ;; Regular . path — var deref + invokeinterface/invokevirtual
      (let [obj-type (emit-form code obj locals ctx)
            obj-meta-tag (or (:raster.type/tag (meta obj)) (:tag (meta obj)))
            obj-hint (when (and (symbol? obj) (get locals obj))
                       (:hint (get locals obj)))
            obj-tag  (or obj-meta-tag
                         obj-hint
                       ;; Infer class from literal types
                         (cond
                           (float? obj)   'java.lang.Number
                           (integer? obj) 'java.lang.Number
                           (string? obj)  'java.lang.String
                           :else nil)
                       ;; Infer tag from resolved var metadata (covers cases where
                       ;; macroexpand strips :tag from the symbol, e.g. ftm→reify bodies)
                         (when (and (symbol? obj) (namespace obj)
                                    (not (get locals obj)))
                           (try
                             (let [resolved (ns-resolve (or (:source-ns ctx) *ns*) obj)]
                               (when (var? resolved)
                                 (let [tag (:tag (meta resolved))]
                                 ;; tag may be a Class or a symbol; normalize to symbol
                                   (cond
                                     (class? tag) (symbol (.getName ^Class tag))
                                     (symbol? tag) tag
                                     (string? tag) (symbol tag)
                                     :else nil))))
                             (catch Exception _ nil)))
                       ;; Infer return type from static method call:
                       ;; (Foo/bar args...) → reflect on Foo.bar return type
                         (when (and (seq? obj) (symbol? (first obj))
                                    (let [ns-part (namespace (first obj))]
                                      (and ns-part (pos? (count ns-part)))))
                           (try
                             (let [s (first obj)
                                   cls-name (namespace s)
                                   meth-name (name s)
                                   src-ns (or (:source-ns ctx) *ns*)
                                   cls (or (try (Class/forName cls-name) (catch Exception _ nil))
                                           (when-let [r (ns-resolve src-ns (symbol cls-name))]
                                             (when (class? r) r)))]
                               (when cls
                                 (when-let [m (first (filter #(= (.getName ^java.lang.reflect.Method %)
                                                                 meth-name)
                                                             (.getMethods cls)))]
                                   (symbol (.getName (.getReturnType ^java.lang.reflect.Method m))))))
                             (catch Exception _ nil))))
          ;; Skip checkcast when type is guaranteed by local hint
          ;; (set by typed method params or prior checkcast in let*)
            type-verified? (and (nil? obj-meta-tag) (some? obj-hint))
            src-ns (or (:source-ns ctx) *ns*)
            cls (when (and obj-tag (pos? (count (str obj-tag))))
                  (or (try (Class/forName (str obj-tag)) (catch Exception _ nil))
                      (when-let [r (ns-resolve src-ns (symbol (str obj-tag)))]
                        (when (class? r) r))))]
        (if-not cls
          (throw (ex-info (str "Cannot resolve class for: (. " obj " " meth-str ")")
                          {:obj obj :method meth-str}))
          (let [cls-cd (class-desc-of cls)
                candidates (filter #(and (= (.getName ^java.lang.reflect.Method %) meth-str)
                                         (= (.getParameterCount ^java.lang.reflect.Method %) (count method-args)))
                                   (.getMethods cls))
              ;; Select best method: prefer assignable arg types over coercion.
              ;; When multiple overloads match, use arg :tag metadata and
              ;; local :hint to pick the right one (e.g. max(Vector) vs max(double)).
                non-bridge (vec (remove #(.isBridge ^java.lang.reflect.Method %) candidates))
                method (or
                       ;; Try arg-tag-guided selection when ambiguous
                        (when (> (count non-bridge) 1)
                          (let [arg-tags (mapv (fn [arg]
                                                 (cond
                                                   (symbol? arg)
                                                   (or (:tag (meta arg))
                                                       (:hint (get locals arg)))
                                                   (seq? arg) (:tag (meta arg))
                                                   :else nil))
                                               method-args)
                               ;; Resolve arg classes from tags
                                arg-classes (mapv (fn [tag]
                                                    (when (and tag (pos? (count (str tag))))
                                                      (try (Class/forName (str tag))
                                                           (catch Exception _
                                                             (try (when-let [r (ns-resolve src-ns (symbol (str tag)))]
                                                                    (when (class? r) r))
                                                                  (catch Exception _ nil))))))
                                                  arg-tags)]
                            (when (some some? arg-classes)
                              (first (filter (fn [^java.lang.reflect.Method m]
                                               (let [pts (.getParameterTypes m)]
                                                 (every? true?
                                                         (map (fn [^Class pt ^Class ac]
                                                                (or (nil? ac) (.isAssignableFrom pt ac)))
                                                              pts arg-classes))))
                                             non-bridge)))))
                        (first non-bridge)
                        (first candidates))]
            (if method
            ;; Method call — invokeinterface/invokevirtual.
            ;; For IFn__ .invk on LOCAL vars (Fn-typed params like `f`),
            ;; emit instanceof guard: typed path (.invk, primitive) +
            ;; boxed fallback (IFn.invoke). This handles both typed ftm
            ;; and plain defn callers correctly.
              (let [param-types (.getParameterTypes method)
                    ^Class ret-class (.getReturnType method)
                    ret-st (class-type->stack ret-class)
                    needs-guard? (and (= meth-str "invk")
                                      (.isInterface cls)
                                      (typed-interface? (.getName cls))
                                      (symbol? obj)
                                      (get locals obj)
                                      (not (:verified (get locals obj))))
                    obj-type (if (primitive? obj-type)
                               (do (emit-box-to-ref code obj-type) :ref)
                               obj-type)]
                (if needs-guard?
                ;; instanceof guard: typed .invk when available, boxed IFn.invoke fallback
                  (let [boxed-label (.newLabel code)
                        end-label (.newLabel code)
                        obj-slot (:slot (get locals obj))]
                  ;; obj is already on stack from emit-form above
                  ;; instanceof check (uses stack value, doesn't consume it)
                    (.dup code)
                    (.instanceOf code cls-cd)
                    (.ifeq code boxed-label)
                  ;; === Typed path ===
                    (.checkcast code cls-cd)
                    (doseq [[arg ^Class pt] (map vector method-args param-types)]
                      (let [t (emit-form code arg locals ctx)
                            target (class-type->stack pt)]
                        (when (not= t target) (emit-coerce code t target))))
                    (let [mt (MethodTypeDesc/of (class-desc-of ret-class)
                                                (into-array ClassDesc (mapv class-desc-of param-types)))]
                      (.invokeinterface code cls-cd meth-str mt))
                  ;; Normalize stack for merge: void → null, primitive → box
                    (case ret-st
                      :void (.aconst_null code)
                      (:double :float :int :long) (emit-box-to-ref code ret-st)
                      nil)
                    (.goto_ code end-label)
                  ;; === Boxed fallback ===
                    (.labelBinding code boxed-label)
                  ;; obj ref is still on stack
                    (.checkcast code ifn-cd)
                    (doseq [a method-args]
                      (let [t (emit-form code a locals ctx)]
                        (when (#{:double :float :int :long} t)
                          (emit-box-to-ref code t))))
                    (let [n (count method-args)
                          invoke-mt (MethodTypeDesc/of obj-cd
                                                       (into-array ClassDesc (repeat n obj-cd)))]
                      (.invokeinterface code ifn-cd "invoke" invoke-mt))
                  ;; === Merge point ===
                    (.labelBinding code end-label)
                  ;; Both paths leave Object on stack. Unbox if typed path returns primitive.
                    (case ret-st
                      :void (do (.pop code) :void)
                      (:double :float :int :long) (do (emit-unbox-to-prim code ret-st) ret-st)
                      :ref))
                ;; Regular method call (no guard needed)
                  (do
                    (when (and (= obj-type :ref) (not type-verified?))
                      (.checkcast code cls-cd))
                    (doseq [[arg ^Class pt] (map vector method-args param-types)]
                      (let [t      (emit-form code arg locals ctx)
                            target (class-type->stack pt)]
                        (when (not= t target) (emit-coerce code t target))))
                    (let [mt (MethodTypeDesc/of (class-desc-of ret-class)
                                                (into-array ClassDesc (mapv class-desc-of param-types)))]
                      (if (.isInterface cls)
                        (.invokeinterface code cls-cd meth-str mt)
                        (.invokevirtual code cls-cd meth-str mt)))
                    (class-type->stack ret-class))))
            ;; Field access (strip leading - from Clojure's .-field syntax,
            ;; replace hyphens with underscores for JVM field names)
              (let [field-name (cond-> (if (.startsWith meth-str "-")
                                         (subs meth-str 1)
                                         meth-str)
                                 true (.replace "-" "_"))
                    field (or (try (.getField cls field-name) (catch NoSuchFieldException _ nil))
                           ;; Try without underscore munging (Java classes)
                              (try (.getField cls (if (.startsWith meth-str "-")
                                                    (subs meth-str 1)
                                                    meth-str))
                                   (catch NoSuchFieldException _ nil)))
                    ^Class ft (when field (.getType field))]
                (when-not field
                  (throw (ex-info (str "Cannot resolve method or field: " meth-str " on " (.getName cls))
                                  {:class (.getName cls) :member meth-str :field-name field-name
                                   :available-fields (mapv #(.getName %) (.getFields cls))})))
                (when (and (= obj-type :ref) (not type-verified?))
                  (.checkcast code cls-cd))
                (.getfield code cls-cd field-name (class-desc-of ft))
              ;; When field is Object but the field-type-registry knows a
              ;; more specific type (from defvalue annotations), add CHECKCAST.
              ;; This is zero-cost in C2 and enables correct downstream dispatch.
                (let [jvm-type (class-type->stack ft)
                      registry-tag (when (and (= jvm-type :ref) (= ft Object))
                                     (let [cls-sym (symbol (.getSimpleName cls))]
                                       (when-let [fields (get @inf/field-type-registry cls-sym)]
                                         (get fields field-name))))]
                  (if registry-tag
                  ;; Registry knows the specific type — CHECKCAST for refs,
                  ;; unbox for primitives (field returns Object, we need double/long)
                    (case registry-tag
                      doubles (do (.checkcast code dblarr-cd) :ref)
                      floats  (do (.checkcast code fltarr-cd) :ref)
                      longs   (do (.checkcast code lngarr-cd) :ref)
                      ints    (do (.checkcast code intarr-cd) :ref)
                      objects (do (.checkcast code objarr-cd) :ref)
                      double  (do (emit-unbox-to-prim code :double) :double)
                      float   (do (emit-unbox-to-prim code :float) :float)
                      long    (do (emit-unbox-to-prim code :long) :long)
                      int     (do (emit-unbox-to-prim code :int) :int)
                    ;; Default: try as class, CHECKCAST if ref
                      (if-let [cls (try (Class/forName (str registry-tag))
                                        (catch Exception _ nil))]
                        (do (.checkcast code (class-desc-of cls))
                            (class-type->stack cls))
                        jvm-type))
                    jvm-type))))))))))

;; ================================================================
;; emit-form handlers — extracted from the main dispatch for clarity
;; Each handler takes [code form/args locals ctx] → stack-type
;; ================================================================

(declare emit-form)

(defn- emit-symbol
  "Emit bytecode for a symbol: local variable load, var deref, constant fold, or static field."
  [code form locals ctx]
  (if-let [{:keys [slot type]} (get locals form)]
    (do (case type
          :double (.dload code slot)
          :long   (.lload code slot)
          :int    (.iload code slot)
          :float  (.fload code slot)
          (.aload code slot))
        type)
    (if-let [v (try (ns-resolve (or (:source-ns ctx) *ns*) form)
                    (catch Exception _ nil))]
      (if (var? v)
        (let [root (try (.getRawRoot ^clojure.lang.Var v) (catch Exception _ nil))]
          (cond
            (instance? Double root)  (do (.ldc code (double root)) :double)
            (instance? Long root)    (do (.ldc code (long root)) :long)
            (instance? Float root)   (do (.ldc code (float root)) :float)
            (instance? Integer root) (do (.ldc code (int root)) :int)
            :else
            (do (.ldc code (str (.name (.ns ^clojure.lang.Var v))))
                (.ldc code (str (.sym ^clojure.lang.Var v)))
                (.invokestatic code rt-cd "var"
                               (MethodTypeDesc/of var-cd
                                                  (into-array ClassDesc
                                                              [(ClassDesc/ofDescriptor "Ljava/lang/String;")
                                                               (ClassDesc/ofDescriptor "Ljava/lang/String;")])))
                (.invokevirtual code var-cd "getRawRoot" (MethodTypeDesc/of obj-cd no-cd))
                :ref)))
        (if (class? v)
          (do (.ldc code (class-desc-of v)) :ref)
          (throw (ex-info (str "Unresolved symbol in bytecode emit: " form)
                          {:symbol form :locals (keys locals)}))))
      (if-let [slash-idx (when (namespace form) (.indexOf ^String (str form) "/"))]
        (let [s (str form)
              cls-name (subs s 0 slash-idx)
              field-name (subs s (inc slash-idx))
              src-ns (or (:source-ns ctx) *ns*)
              cls (or (try (Class/forName cls-name) (catch Exception _ nil))
                      (when-let [r (ns-resolve src-ns (symbol cls-name))]
                        (when (class? r) r)))]
          (if (and cls (pos? (count field-name)))
            (let [field (try (.getField cls field-name) (catch Exception _ nil))]
              (if field
                (let [^Class ftype (.getType field)]
                  (.getstatic code (class-desc-of cls) field-name (class-desc-of ftype))
                  (class-type->stack ftype))
                (throw (ex-info (str "Cannot resolve static field: " form)
                                {:class cls-name :field field-name}))))
            (throw (ex-info (str "Unresolved symbol in bytecode emit: " form)
                            {:symbol form :locals (keys locals)}))))
        (throw (ex-info (str "Unresolved symbol in bytecode emit: " form)
                        {:symbol form :locals (keys locals)}))))))

(defn- emit-keyword-invoke
  "Emit bytecode for (:key obj) or (:key obj default) — keyword invocation."
  [code head args locals ctx]
  (let [obj-form (first args)
        obj-hint (when (symbol? obj-form) (:hint (get locals obj-form)))
        field-name (name head)
        jvm-field-name (.replace ^String field-name "-" "_")
        src-ns (or (:source-ns ctx) *ns*)
        field-cls (when obj-hint
                    (or (try (Class/forName (str obj-hint)) (catch Exception _ nil))
                        (when-let [r (ns-resolve src-ns (symbol (str obj-hint)))]
                          (when (class? r) r))))
        field (when field-cls
                (try (.getField ^Class field-cls jvm-field-name) (catch Exception _ nil)))]
    (if field
      (let [obj-type (emit-form code obj-form locals ctx)
            _ (when (and (= obj-type :ref) (not (:verified (get locals obj-form))))
                (.checkcast code (class-desc-of field-cls)))
            ^Class ftype (.getType ^java.lang.reflect.Field field)
            fcd (class-desc-of ftype)]
        (.getfield code (class-desc-of field-cls) jvm-field-name fcd)
        (class-type->stack ftype))
      (let [kw-cd (ClassDesc/of "clojure.lang" "Keyword")
            kw-intern-mt (MethodTypeDesc/of kw-cd (into-array ClassDesc [(ClassDesc/ofDescriptor "Ljava/lang/String;") (ClassDesc/ofDescriptor "Ljava/lang/String;")]))]
        (if (namespace head)
          (.ldc code (namespace head))
          (.aconst_null code))
        (.ldc code (name head))
        (.invokestatic code kw-cd "intern" kw-intern-mt)
        (let [t (emit-form code (first args) locals ctx)]
          (when (primitive? t) (emit-box-to-ref code t)))
        (if (second args)
          (let [t (emit-form code (second args) locals ctx)]
            (when (primitive? t) (emit-box-to-ref code t))
            (.invokeinterface code ifn-cd "invoke"
                              (MethodTypeDesc/of obj-cd (into-array ClassDesc [obj-cd obj-cd])))
            :ref)
          (do (.invokeinterface code ifn-cd "invoke"
                                (MethodTypeDesc/of obj-cd (into-array ClassDesc [obj-cd])))
              :ref))))))

(defn- emit-loop*
  "Emit bytecode for (loop* [bindings] body...)."
  [code args locals ctx]
  (let [[bindings & body] args
        pairs (partition 2 bindings)
        loop-label (.newLabel code)
        loop-locals (reduce
                     (fn [locs [sym init]]
                       (let [t (emit-form code init locs (dissoc ctx :void-context))
                             slot (alloc-slot! ctx t)]
                         (case t
                           :double (.dstore code slot)
                           :long   (.lstore code slot)
                           :int    (.istore code slot)
                           :float  (.fstore code slot)
                           (.astore code slot))
                         (assoc locs sym {:slot slot :type t})))
                     locals pairs)
        loop-var-syms (mapv first pairs)]
    (.labelBinding code loop-label)
    (let [ctx (assoc ctx :loop-label loop-label
                     :loop-vars (mapv #(get loop-locals %) loop-var-syms))]
      (emit-body code body loop-locals ctx))))

(defn- emit-recur
  "Emit bytecode for (recur args...) — loop back-edge with IINC optimization."
  [code args locals ctx]
  (let [{:keys [loop-label loop-vars]} ctx
        single-iinc?
        (and (= 1 (count args) (count loop-vars))
             (seq? (first args))
             (let [arg (first args)
                   lv (first loop-vars)
                   h (first arg)
                   operand (second arg)]
               (and (= :int (:type lv))
                    (or (= h 'clojure.core/unchecked-inc-int)
                        (= h 'unchecked-inc-int)
                        (= h 'clojure.core/unchecked-dec-int)
                        (= h 'unchecked-dec-int))
                    (symbol? operand)
                    (when-let [info (get locals operand)]
                      (= (:slot info) (:slot lv))))))]
    (if single-iinc?
      (let [slot (:slot (first loop-vars))
            h (first (first args))
            delta (if (or (= h 'clojure.core/unchecked-inc-int) (= h 'unchecked-inc-int)) 1 -1)]
        (.iinc code slot (int delta))
        (.goto_ code loop-label)
        :diverge)
      (let [vals (mapv (fn [arg loop-var]
                         (let [t (emit-form code arg locals ctx)
                               target (:type loop-var)]
                           (when (not= t target)
                             (emit-coerce code t target))
                           {:type target}))
                       args loop-vars)]
        (doseq [[_ {:keys [slot type]}]
                (reverse (map vector vals loop-vars))]
          (case type
            :double (.dstore code slot)
            :long   (.lstore code slot)
            :int    (.istore code slot)
            :float  (.fstore code slot)
            (.astore code slot)))
        (.goto_ code loop-label)
        :diverge))))

(defn- emit-new
  "Emit bytecode for (new ClassName args...)."
  [code args locals ctx]
  (let [[cls-sym & ctor-args] args
        src-ns (or (:source-ns ctx) *ns*)
        cls    (or (try (Class/forName (str cls-sym)) (catch Exception _ nil))
                   (when-let [r (ns-resolve src-ns (symbol (str cls-sym)))]
                     (when (class? r) r))
                   (throw (ex-info (str "Cannot resolve class for new: " cls-sym) {:form (list* 'new args)})))
        cls-cd (class-desc-of cls)]
    (.new_ code cls-cd)
    (.dup code)
    (let [ctor (first (filter #(= (.getParameterCount %) (count ctor-args))
                              (.getConstructors cls)))
          param-types (.getParameterTypes ctor)]
      (doseq [[arg ^Class pt] (map vector ctor-args param-types)]
        (let [t      (emit-form code arg locals ctx)
              target (class-type->stack pt)]
          (if (not= t target)
            (emit-coerce code t target)
            (when (and (= t :ref) (not (.equals pt Object)))
              (.checkcast code (class-desc-of pt))))))
      (.invokespecial code cls-cd "<init>"
                      (MethodTypeDesc/of V-cd (into-array ClassDesc (mapv class-desc-of param-types)))))
    :ref))

(defn- emit-try
  "Emit bytecode for (try body... (catch ExType e handler...) (finally fin...))."
  [code args locals ctx]
  (let [catch-or-finally? #(and (seq? %) (#{'catch 'finally} (first %)))
        [body-forms cfs] (split-with #(not (catch-or-finally? %)) args)
        catches (filter #(= 'catch (first %)) cfs)
        finally-forms (when-let [fin (first (filter #(= 'finally (first %)) cfs))]
                        (rest fin))
        try-start  (.newLabel code)
        try-end    (.newLabel code)
        after-try  (.newLabel code)]
    (.labelBinding code try-start)
    (let [body-type (emit-body code body-forms locals ctx)]
      (when (primitive? body-type) (emit-box-to-ref code body-type)))
    (.labelBinding code try-end)
    (when finally-forms
      (doseq [f finally-forms]
        (let [t (emit-form code f locals ctx)]
          (when-not (#{:void :diverge} t) (if (two-slot? t) (.pop2 code) (.pop code))))))
    (.goto_ code after-try)
    (doseq [[_ ex-class ex-sym & catch-body] catches]
      (let [handler-label (.newLabel code)
            src-ns (or (:source-ns ctx) *ns*)
            ex-cls (or (try (Class/forName (str ex-class)) (catch Exception _ nil))
                       (when-let [r (ns-resolve src-ns (symbol (str ex-class)))]
                         (when (class? r) r))
                       Exception)
            ex-cd (class-desc-of ex-cls)]
        (.exceptionCatch code try-start try-end handler-label ex-cd)
        (.labelBinding code handler-label)
        (let [slot (let [s @(:next-slot ctx)] (swap! (:next-slot ctx) inc) s)
              catch-locals (assoc locals ex-sym {:slot slot :type :ref
                                                 :hint (symbol (.getName ex-cls))})]
          (.astore code slot)
          (let [ct (emit-body code catch-body catch-locals ctx)]
            (when (primitive? ct) (emit-box-to-ref code ct))))
        (when finally-forms
          (doseq [f finally-forms]
            (let [t (emit-form code f locals ctx)]
              (when-not (#{:void :diverge} t) (if (two-slot? t) (.pop2 code) (.pop code))))))
        (.goto_ code after-try)))
    (when finally-forms
      (let [fin-handler (.newLabel code)]
        (.exceptionCatch code try-start try-end fin-handler (ClassDesc/ofDescriptor "Ljava/lang/Throwable;"))
        (.labelBinding code fin-handler)
        (let [ex-slot (let [s @(:next-slot ctx)] (swap! (:next-slot ctx) inc) s)]
          (.astore code ex-slot)
          (doseq [f finally-forms]
            (let [t (emit-form code f locals ctx)]
              (when-not (#{:void :diverge} t) (if (two-slot? t) (.pop2 code) (.pop code)))))
          (.aload code ex-slot)
          (.athrow code))))
    (.labelBinding code after-try)
    :ref))

(defn- emit-case*
  "Emit bytecode for (case* ge shift mask default imap &rest)."
  [code args locals ctx]
  (let [[ge shift mask default-expr imap & _rest] args
        after-label (.newLabel code)]
    (let [t (emit-form code ge locals ctx)]
      (when (not= t :int)
        (if (= t :ref)
          ;; case* dispatches on hashCode, not intValue — keywords, strings, etc.
          (.invokevirtual code (ClassDesc/of "java.lang" "Object") "hashCode"
                          (MethodTypeDesc/of I-cd no-cd))
          (emit-coerce code t :int))))
    (when (and shift (not (zero? shift)))
      (.ldc code (int shift)) (.iushr code))
    (when (and mask (not (zero? mask)) (not= mask -1))
      (.ldc code (int mask)) (.iand code))
    (let [sorted-entries (sort-by first (seq imap))
          case-labels (mapv (fn [_] (.newLabel code)) sorted-entries)
          default-label (.newLabel code)
          switch-cases (java.util.ArrayList.)]
      (doseq [[hash-key label] (map vector (mapv first sorted-entries) case-labels)]
        (.add switch-cases (java.lang.classfile.instruction.SwitchCase/of (int hash-key) label)))
      (.lookupswitch code default-label switch-cases)
      (let [branch-exprs (mapv (fn [[_ [_ result-expr]]] result-expr) sorted-entries)
            all-exprs (conj branch-exprs default-expr)
            inferred-types (mapv #(infer-arg-stack-type % locals) all-exprs)
            value-types (filterv some? inferred-types)
            uniform-prim? (and (seq value-types)
                               (apply = value-types)
                               (primitive? (first value-types)))
            result-type (if uniform-prim? (first value-types) :ref)]
        (doseq [[[_ [test-val result-expr]] label] (map vector sorted-entries case-labels)]
          (.labelBinding code label)
          (let [t (emit-form code result-expr locals ctx)]
            (when (and (not uniform-prim?) (primitive? t))
              (emit-box-to-ref code t))
            (when-not (diverge? t)
              (.goto_ code after-label))))
        (.labelBinding code default-label)
        (let [t (emit-form code default-expr locals ctx)]
          (when (and (not uniform-prim?) (primitive? t))
            (emit-box-to-ref code t))
          (when-not (diverge? t)
            (.goto_ code after-label)))
        (.labelBinding code after-label)
        result-type))))

(defn- emit-quote
  "Emit bytecode for (quote v) — literal values.
  Handles atoms (nil, keywords, symbols, strings, numbers) natively.
  For complex data structures (vectors, lists, maps), uses RT.readString
  with the pr-str serialization as a general fallback."
  [code args locals ctx]
  (let [v (first args)]
    (cond
      (nil? v)     (do (.aconst_null code) :ref)
      (keyword? v) (let [kw-cd (ClassDesc/of "clojure.lang" "Keyword")
                         str-cd (ClassDesc/ofDescriptor "Ljava/lang/String;")]
                     (if (namespace v)
                       (.ldc code (namespace v))
                       (.aconst_null code))
                     (.ldc code (name v))
                     (.invokestatic code kw-cd "intern"
                                    (MethodTypeDesc/of kw-cd (into-array ClassDesc [str-cd str-cd])))
                     :ref)
      (symbol? v)  (let [sym-cd (ClassDesc/of "clojure.lang" "Symbol")
                         str-cd (ClassDesc/ofDescriptor "Ljava/lang/String;")]
                     (if (namespace v)
                       (.ldc code (namespace v))
                       (.aconst_null code))
                     (.ldc code (name v))
                     (.invokestatic code sym-cd "intern"
                                    (MethodTypeDesc/of sym-cd (into-array ClassDesc [str-cd str-cd])))
                     :ref)
      (string? v)  (do (.ldc code v) :ref)
      (integer? v) (do (.ldc code (int v)) :int)
      (float? v)   (do (.ldc code (double v)) :double)
      (true? v)    (do (.getstatic code (ClassDesc/ofDescriptor "Ljava/lang/Boolean;")
                                   "TRUE" (ClassDesc/ofDescriptor "Ljava/lang/Boolean;"))
                       :ref)
      (false? v)   (do (.getstatic code (ClassDesc/ofDescriptor "Ljava/lang/Boolean;")
                                   "FALSE" (ClassDesc/ofDescriptor "Ljava/lang/Boolean;"))
                       :ref)
      ;; Complex data structures: serialize to string, reconstruct at runtime
      ;; via RT.readString. Handles vectors, lists, maps, sets, nested forms.
      :else        (do (.ldc code (pr-str v))
                       (.invokestatic code rt-cd "readString"
                                      (MethodTypeDesc/of obj-cd (into-array ClassDesc
                                                                            [(ClassDesc/ofDescriptor "Ljava/lang/String;")])))
                       :ref))))

(defn- emit-fn*
  "Emit bytecode for (fn* arities...) — anonymous function/closure."
  [code args locals ctx]
  (let [arities (if (vector? (first args))
                  [(cons (first args) (rest args))]
                  args)
        free-vars (collect-free-vars arities locals)
        anon-info (emit-anon-fn-class free-vars arities ctx)]
    (.new_ code (:class-desc anon-info))
    (.dup code)
    (doseq [[sym info] free-vars]
      (let [{:keys [slot type]} (get locals sym)]
        (case type
          :double (.dload code slot)
          :long   (.lload code slot)
          :int    (.iload code slot)
          :float  (.fload code slot)
          (.aload code slot))))
    (.invokespecial code (:class-desc anon-info) "<init>"
                    (MethodTypeDesc/of V-cd (into-array ClassDesc (:field-types anon-info))))
    :ref))

(defn- emit-letfn*
  "Emit bytecode for (letfn* [bindings] body...) — mutually recursive local fns."
  [code args locals ctx]
  (let [[bindings & body] args
        pairs (partition 2 bindings)
        fn-locals (reduce
                   (fn [locs [sym _fn-form]]
                     (let [slot (let [s @(:next-slot ctx)]
                                  (swap! (:next-slot ctx) inc) s)]
                       (.aconst_null code)
                       (.astore code slot)
                       (assoc locs sym {:slot slot :type :ref})))
                   locals pairs)]
    (doseq [[sym fn-form] pairs]
      (let [expanded (macroexpand-all-preserving fn-form)
            t (emit-form code expanded fn-locals ctx)
            {:keys [slot]} (get fn-locals sym)]
        (when (primitive? t) (emit-box-to-ref code t))
        (.astore code slot)))
    (emit-body code body fn-locals ctx)))

(defn- emit-set!
  "Emit bytecode for (set! (. obj field) val) — field mutation."
  [code args locals ctx]
  (let [[target val-form] args]
    (if (and (seq? target) (= '. (first target)))
      (let [[_ obj field-name] target
            field-str (str field-name)
            obj-type (emit-form code obj locals ctx)
            obj-tag  (or (:tag (meta obj))
                         (when (symbol? obj) (:hint (get locals obj))))
            src-ns   (or (:source-ns ctx) *ns*)
            cls      (when obj-tag
                       (or (try (Class/forName (str obj-tag)) (catch Exception _ nil))
                           (when-let [r (ns-resolve src-ns (symbol (str obj-tag)))]
                             (when (class? r) r))))
            cls-cd   (when cls (class-desc-of cls))
            field    (when cls (.getField cls field-str))
            ft       (when field (.getType field))]
        (when cls-cd
          (when (= obj-type :ref) (.checkcast code cls-cd)))
        (let [vt (emit-form code val-form locals ctx)]
          (if (and ft cls-cd)
            (let [target-st (class-type->stack ft)]
              (when (not= vt target-st) (emit-coerce code vt target-st))
              (.putfield code cls-cd field-str (class-desc-of ft))
              :void)
            (throw (ex-info (str "Cannot resolve field for set!: " field-str)
                            {:target target})))))
      (throw (ex-info (str "set! only supports field mutation: (set! (. obj field) val)")
                      {:target target})))))

(defn- emit-ftm
  "Emit bytecode for (ftm [params :- Types] body...) — strip annotations, delegate to fn*."
  [code args locals ctx]
  (let [param-vec (first args)
        rest-args (rest args)
        [_ret-type body] (if (= :- (first rest-args))
                           [(second rest-args) (nnext rest-args)]
                           [nil rest-args])
        plain-params (vec (loop [items (seq param-vec) result []]
                            (if-not items
                              result
                              (let [item (first items)]
                                (if (and (next items) (= :- (second items)))
                                  (recur (nnext (next items)) (conj result item))
                                  (recur (next items) (conj result item)))))))
        fn-form (list* 'fn* (list (cons plain-params body)))]
    (emit-form code fn-form locals ctx)))

(defn- emit-reify*
  "Emit bytecode for (reify* [ifaces] method-forms...) — interface implementation."
  [code args locals ctx]
  (let [iface-syms   (first args)
        method-forms (rest args)
        pseudo-arities (mapv (fn [mf]
                               (let [[_name params & body] mf]
                                 (cons params body)))
                             method-forms)
        free-vars  (collect-free-vars pseudo-arities locals)
        reify-info (emit-reify-class iface-syms method-forms free-vars ctx)]
    (.new_ code (:class-desc reify-info))
    (.dup code)
    (doseq [[sym _info] free-vars]
      (let [{:keys [slot type]} (get locals sym)]
        (case type
          :double (.dload code slot)
          :long   (.lload code slot)
          :int    (.iload code slot)
          :float  (.fload code slot)
          (.aload code slot))))
    (.invokespecial code (:class-desc reify-info) "<init>"
                    (MethodTypeDesc/of V-cd (into-array ClassDesc (:field-types reify-info))))
    :ref))

(defn- emit-map-literal
  "Emit bytecode for a map literal → RT.mapUniqueKeys(k1,v1,k2,v2,...)."
  [code form locals ctx]
  (let [entries (seq form)
        n (* 2 (count entries))]
    (.ldc code (int n))
    (.anewarray code obj-cd)
    (doseq [[i [k v]] (map-indexed vector entries)]
      (.dup code)
      (.ldc code (int (* 2 i)))
      (let [t (emit-form code k locals ctx)]
        (when (primitive? t) (emit-box-to-ref code t)))
      (.aastore code)
      (.dup code)
      (.ldc code (int (inc (* 2 i))))
      (let [t (emit-form code v locals ctx)]
        (when (primitive? t) (emit-box-to-ref code t)))
      (.aastore code))
    (.invokestatic code rt-cd "mapUniqueKeys"
                   (MethodTypeDesc/of
                    (ClassDesc/ofDescriptor "Lclojure/lang/IPersistentMap;")
                    (into-array ClassDesc [objarr-cd])))
    :ref))

(defn- emit-form
  "Emit bytecode for a macroexpanded, walked Clojure form.
  code:   CodeBuilder
  form:   the Clojure form (after macroexpand-all)
  locals: {symbol -> {:slot int, :type (:ref :double :long :int)}}
  ctx:    {:class-desc ClassDesc, :element-type symbol,
           :sibling-methods {name-str -> {:method-name str, :method-type MethodTypeDesc}},
           :next-slot (atom int), :loop-label Label, :loop-vars [...],
           :source-ns Namespace (for symbol resolution)}
  Returns :ref :double :long :int :void"
  [code form locals ctx]
  ;; Coerce lazy seqs to proper lists — SIMD codegen may produce lazy seqs
  ;; from map/for that need to be materialized for pattern matching
  (let [form (if (and (seq? form) (not (list? form)))
               (with-meta (apply list form) (meta form))
               form)]
    (cond
      ;; --- Atoms ---
      (nil? form)     (do (.aconst_null code) :ref)
      (true? form)    (do (.ldc code (int 1)) :int)
      (false? form)   (do (.ldc code (int 0)) :int)
      (integer? form) (if (and (instance? Long form)
                               (or (> (long form) Integer/MAX_VALUE)
                                   (< (long form) Integer/MIN_VALUE)))
                        (do (.ldc code (long form)) :long)
                        (do (.ldc code (int form)) :int))
      (float? form)   (do (.ldc code (double form)) :double)
      (char? form)    (do (.ldc code (int form)) :int) ;; JVM chars are ints
      (string? form)  (do (.ldc code form) :ref)
      (keyword? form) (let [kw-cd (ClassDesc/of "clojure.lang" "Keyword")
                            str-cd (ClassDesc/ofDescriptor "Ljava/lang/String;")]
                        (if (namespace form)
                          (.ldc code (namespace form))
                          (.aconst_null code))
                        (.ldc code (name form))
                        (.invokestatic code kw-cd "intern"
                                       (MethodTypeDesc/of kw-cd (into-array ClassDesc [str-cd str-cd])))
                        :ref)

      (symbol? form) (emit-symbol code form locals ctx)

      ;; --- List (special form or function call) ---
      (seq? form)
      (let [[head & args] form
            head-name (when (symbol? head) (name head))]
        (if (keyword? head)
          (emit-keyword-invoke code head args locals ctx)
        ;; .method and .-field sugar: (.method obj args...) → (. obj method args...)
        ;; The walker may not macroexpand these forms, so handle them here.
          (if (and (symbol? head) head-name
                   (.startsWith head-name ".") (not= head-name "."))
            (let [member-sym (symbol (subs head-name 1))]
              (emit-dot-handler code (cons (first args) (cons member-sym (rest args))) locals ctx))

            (case (when (symbol? head) head)

          ;; let*
              let*
              (emit-let*-handler code args locals ctx)

          ;; let — macroexpand to let* (handles destructuring → nth calls)
              let
              (let [expanded (macroexpand-1 (list* 'let args))
                    [_ bindings & body] expanded]
                (emit-let*-handler code (list* bindings body) locals ctx))

          ;; do
              do
              (let [stmts (butlast args)
                    last-form (last args)
                    void-ctx (assoc ctx :void-context true)]
                (doseq [s stmts]
                  (let [t (emit-form code s locals void-ctx)]
                    (when-not (#{:void :diverge} t)
                      (if (two-slot? t) (.pop2 code) (.pop code)))))
                (if last-form
                  (emit-form code last-form locals
                             (if (:void-context ctx) ctx (dissoc ctx :void-context)))
                  :void))

          ;; if — both branches must produce same stack type for verifier
              if
              (emit-if-handler code args locals ctx)

              loop*
              (emit-loop* code args locals ctx)

              recur
              (emit-recur code args locals ctx)

          ;; throw — unconditional transfer, returns :diverge (not :void)
              throw
              (let [t (emit-form code (first args) locals ctx)]
                (when (primitive? t) (emit-box-to-ref code t))
                (.checkcast code (ClassDesc/of "java.lang" "Throwable"))
                (.athrow code)
                :diverge)

              new
              (emit-new code args locals ctx)

          ;; . (dot) — instance method or field access
              .
              (emit-dot-handler code args locals ctx)

              try
              (emit-try code args locals ctx)

              case*
              (emit-case* code args locals ctx)

              quote
              (emit-quote code args locals ctx)

              fn*
              (emit-fn* code args locals ctx)

              letfn*
              (emit-letfn* code args locals ctx)

              set!
              (emit-set! code args locals ctx)

              ftm
              (emit-ftm code args locals ctx)

          ;; import* — class import at macroexpansion time. No-op in BC.
          ;; May appear qualified as clojure.core/import* after macroexpand-all.
              (import* clojure.core/import*)
              :void

              reify*
              (emit-reify* code args locals ctx)

          ;; var — (var x) returns the Var itself
              var
              (let [sym (first args)
                    v   (ns-resolve (or (:source-ns ctx) *ns*) sym)]
                (if (and v (var? v))
                  (do (.ldc code (str (.name (.ns ^clojure.lang.Var v))))
                      (.ldc code (str (.sym ^clojure.lang.Var v)))
                      (.invokestatic code rt-cd "var"
                                     (MethodTypeDesc/of var-cd
                                                        (into-array ClassDesc [(ClassDesc/ofDescriptor "Ljava/lang/String;")
                                                                               (ClassDesc/ofDescriptor "Ljava/lang/String;")])))
                      :ref)
                  (throw (ex-info (str "Cannot resolve var: " sym)
                                  {:symbol sym}))))

          ;; Default: function call dispatch
              (emit-fn-call code head head-name args locals ctx))
        ;; close the (if (.startsWith ...) and (if (keyword? head) ...) from above
            )))

      ;; --- Vector literal → RT.vector(elements...) ---
      (vector? form)
      (let [n (count form)]
        ;; Create Object array, populate, call RT.vector
        (.ldc code (int n))
        (.anewarray code obj-cd)
        (doseq [[i elem] (map-indexed vector form)]
          (.dup code)
          (.ldc code (int i))
          (let [t (emit-form code elem locals ctx)]
            (when (primitive? t) (emit-box-to-ref code t)))
          (.aastore code))
        (.invokestatic code rt-cd "vector"
                       (MethodTypeDesc/of
                        (ClassDesc/ofDescriptor "Lclojure/lang/IPersistentVector;")
                        (into-array ClassDesc [objarr-cd])))
        :ref)

      (map? form) (emit-map-literal code form locals ctx)

      ;; --- Set literal → RT.set(elements...) ---
      (set? form)
      (let [elems (seq form)
            n (count elems)]
        (.ldc code (int n))
        (.anewarray code obj-cd)
        (doseq [[i elem] (map-indexed vector elems)]
          (.dup code)
          (.ldc code (int i))
          (let [t (emit-form code elem locals ctx)]
            (when (primitive? t) (emit-box-to-ref code t)))
          (.aastore code))
        (.invokestatic code rt-cd "set"
                       (MethodTypeDesc/of
                        (ClassDesc/ofDescriptor "Lclojure/lang/IPersistentSet;")
                        (into-array ClassDesc [objarr-cd])))
        :ref)

      ;; --- Anything else ---
      :else
      (throw (ex-info (str "Unknown form type in bytecode: " (type form) " " (pr-str form))
                      {:form form})))))

;; ================================================================
;; scan-deftm-refs — discover deftm vars referenced in walked code
;; ================================================================

(defn- no-inline-deftm?
  "Check if a deftm var or its dispatch parent has ^:no-inline."
  [v]
  (or (:no-inline (meta v))
      ;; Check the dispatch var (base name before _m_)
      (let [vname (str (:name (meta v)))]
        (when-let [idx (clojure.string/index-of vname "_m_")]
          (let [dispatch-name (subs vname 0 idx)
                dispatch-var (ns-resolve (:ns (meta v)) (symbol dispatch-name))]
            (when dispatch-var
              (:no-inline (meta dispatch-var))))))))

(defn- scan-deftm-refs
  "Walk forms to find all referenced deftm vars.
  Returns a set of vars that have ::deftm-walked-body metadata.
  Handles both direct calls (ns/fn args...) and .invk forms
  (.invk ^IFace ns/fn-impl args...)."
  [forms source-ns]
  (let [refs (atom #{})]
    (letfn [(scan [form]
              (cond
                (and (seq? form) (symbol? (first form)))
                (let [h (first form)]
                  (if (= h '.invk)
                    ;; .invk form: extract deftm var from -impl symbol
                    (when-let [impl-sym (second form)]
                      (when (and (symbol? impl-sym) (namespace impl-sym)
                                 (.endsWith (name impl-sym) "-impl"))
                        (let [base-name (subs (name impl-sym) 0 (- (count (name impl-sym)) 5))
                              base-sym (symbol (namespace impl-sym) base-name)]
                          (when-let [v (try (ns-resolve source-ns base-sym) (catch Exception _ nil))]
                            (when (and (var? v)
                                       (:raster.core/deftm-walked-body (meta v))
                                       (not (no-inline-deftm? v)))
                              (swap! refs conj v)))))
                      (doseq [f (drop 2 form)] (scan f)))
                    ;; Normal call
                    (do (when-let [v (try (ns-resolve source-ns h) (catch Exception _ nil))]
                          (when (and (var? v)
                                     (:raster.core/deftm-walked-body (meta v))
                                     (not (no-inline-deftm? v)))
                            (swap! refs conj v)))
                        (doseq [f (rest form)] (scan f)))))
                (seq? form) (doseq [f form] (scan f))
                (vector? form) (doseq [f form] (scan f))
                :else nil))]
      (doseq [f forms] (scan f)))
    @refs))

;; ================================================================
;; tag->jvm-info — type tag to JVM descriptor mapping
;; ================================================================

(defn- tag->jvm-info
  "Map a deftm type tag to JVM type information.
  Returns {:class Class, :class-desc ClassDesc, :stack-type keyword}"
  [tag source-ns]
  (case tag
    (double Double) {:class Double/TYPE :class-desc D-cd :stack-type :double}
    (long Long)     {:class Long/TYPE :class-desc J-cd :stack-type :long}
    (int Integer)   {:class Integer/TYPE :class-desc I-cd :stack-type :int}
    boolean         {:class Boolean/TYPE :class-desc (ClassDesc/ofDescriptor "Z") :stack-type :int}
    (float Float)   {:class Float/TYPE :class-desc F-cd :stack-type :float}
    (void Void)     {:class Void/TYPE :class-desc V-cd :stack-type :void}
    ;; Array types — use typed descriptors ([D, [J, etc.) for precise
    ;; method signatures. Matches param-tag->jvm and typed interface params.
    doubles         {:class (Class/forName "[D") :class-desc dblarr-cd :stack-type :ref}
    floats          {:class (Class/forName "[F") :class-desc fltarr-cd :stack-type :ref}
    longs           {:class (Class/forName "[J") :class-desc lngarr-cd :stack-type :ref}
    ints            {:class (Class/forName "[I") :class-desc intarr-cd :stack-type :ref}
    bytes           {:class (Class/forName "[B") :class-desc (ClassDesc/ofDescriptor "[B") :stack-type :ref}
    shorts          {:class (Class/forName "[S") :class-desc (ClassDesc/ofDescriptor "[S") :stack-type :ref}
    chars           {:class (Class/forName "[C") :class-desc (ClassDesc/ofDescriptor "[C") :stack-type :ref}
    booleans        {:class (Class/forName "[Z") :class-desc (ClassDesc/ofDescriptor "[Z") :stack-type :ref}
    objects         {:class (Class/forName "[Ljava.lang.Object;") :class-desc objarr-cd :stack-type :ref}
    Number          {:class Number :class-desc number-cd :stack-type :ref}
    Object          {:class Object :class-desc obj-cd :stack-type :ref}
    ;; Default: resolve class from tag symbol
    (if (nil? tag)
      {:class Object :class-desc obj-cd :stack-type :ref}
      (let [tag-str (str tag)]
        (if (typed-interface? tag-str)
          ;; Fn-typed params: keep JVM type as Object so the wrapper doesn't
          ;; CHECKCAST (which fails for plain defn callers). The BC body uses
          ;; the :hint metadata for typed invokeinterface when available.
          {:class Object :class-desc obj-cd :stack-type :ref}
          (let [cls (or (try (Class/forName tag-str) (catch Exception _ nil))
                        (when source-ns
                          (when-let [r (ns-resolve source-ns (symbol tag-str))]
                            (when (class? r) r)))
                        Object)]
            {:class cls
             :class-desc (class-desc-of cls)
             :stack-type :ref}))))))

(defn- param-tag->jvm
  "Map a specialized function param type hint to JVM descriptor info for fn-spec params.
  Returns {:class-desc ClassDesc, :stack-type keyword}"
  [tag]
  (case tag
    double   {:class-desc D-cd :stack-type :double}
    float    {:class-desc F-cd :stack-type :float}
    long     {:class-desc J-cd :stack-type :long}
    int      {:class-desc I-cd :stack-type :int}
    boolean  {:class-desc (ClassDesc/ofDescriptor "Z") :stack-type :int}
    objects  {:class-desc objarr-cd :stack-type :ref}
    doubles  {:class-desc dblarr-cd :stack-type :ref}
    floats   {:class-desc fltarr-cd :stack-type :ref}
    longs    {:class-desc (ClassDesc/ofDescriptor "[J") :stack-type :ref}
    ints     {:class-desc (ClassDesc/ofDescriptor "[I") :stack-type :ref}
    bytes    {:class-desc (ClassDesc/ofDescriptor "[B") :stack-type :ref}
    shorts   {:class-desc (ClassDesc/ofDescriptor "[S") :stack-type :ref}
    chars    {:class-desc (ClassDesc/ofDescriptor "[C") :stack-type :ref}
    booleans {:class-desc (ClassDesc/ofDescriptor "[Z") :stack-type :ref}
    Number   {:class-desc obj-cd :stack-type :ref}
    Object   {:class-desc obj-cd :stack-type :ref}
    ;; Default: try to resolve class (handles IFn__ interfaces, defvalue types, etc.)
    (let [tag-str (str tag)]
      (if-let [cls (when (pos? (count tag-str))
                     (try (Class/forName tag-str) (catch Exception _ nil)))]
        {:class-desc (class-desc-of cls) :stack-type :ref}
        {:class-desc obj-cd :stack-type :ref}))))

;; ================================================================
;; compile-specialized-class!
;; ================================================================

(defn compile-specialized-class!
  "Compile multiple specialized functions into a single class with static methods.
  Each function becomes a static method. Calls between them use invokestatic.
  Bodies are macroexpand-all'd before bytecode emission.

  Auto-discovers deftm vars referenced in walked bodies and compiles them as
  typed static methods in the same class. This enables JIT inlining across
  deftm call boundaries (same-class invokestatic is always inlined).

  fn-specs: [{:name sym, :params [...], :walked-body [...], :element-type sym,
              :source-ns Namespace (for symbol resolution)}]

  Returns {:class Class, :methods {fn-name -> Method}}"
  [class-name fn-specs]
  (let [cf         (clojure-classfile)
        dot-idx    (.lastIndexOf ^String class-name (int \.))
        pkg        (subs class-name 0 dot-idx)
        simple     (subs class-name (inc dot-idx))
        class-desc (ClassDesc/of pkg simple)
        ;; Build method type descriptors for fn-specs
        ;; Handle both symbol params and destructuring vectors
        ;; Detect ^double/^long hints to generate primitive param types
        method-specs (mapv (fn [{:keys [name params return-tag]}]
                             (let [method-name (clojure.lang.Compiler/munge (str name))
                                   ;; Each param is either a symbol or a vector (destructuring)
                                   ;; Both become a single method param
                                   method-params (filterv #(or (symbol? %) (vector? %)) params)
                                   param-jvm-infos (mapv (fn [p]
                                                           (if (symbol? p)
                                                             (param-tag->jvm (:tag (meta p)))
                                                             ;; destructuring vector → Object
                                                             {:class-desc obj-cd :stack-type :ref}))
                                                         method-params)
                                   param-descs (mapv :class-desc param-jvm-infos)
                                   param-stack-types (mapv :stack-type param-jvm-infos)
                                   ret-jvm (param-tag->jvm return-tag)
                                   ret-cd (:class-desc ret-jvm)
                                   ret-st (:stack-type ret-jvm)
                                   mt (MethodTypeDesc/of ret-cd
                                                         (into-array ClassDesc param-descs))]
                               {:name name
                                :method-name method-name
                                :method-type mt
                                :params method-params
                                :param-descs param-descs
                                :param-stack-types param-stack-types
                                :return-stack-type ret-st}))
                           fn-specs)
        sibling-methods (into {} (map (fn [ms] [(str (:name ms)) ms]) method-specs))
        ;; Auto-discover deftm vars referenced in fn-spec bodies
        src-ns (or (:source-ns (first fn-specs)) *ns*)
        deftm-refs (scan-deftm-refs (mapcat :walked-body fn-specs) src-ns)
        deftm-method-specs (mapv
                            (fn [v]
                              (let [m (meta v)
                                    deftm-ns (or (:ns m) src-ns)
                                    deftm-params (:raster.core/deftm-params m)
                                    deftm-tags (:raster.core/deftm-tags m)
                                    return-tag (:raster.core/return-tag m)
                                    param-infos (mapv #(tag->jvm-info % deftm-ns) deftm-tags)
                                    return-info (tag->jvm-info (or return-tag 'Object) deftm-ns)
                                    param-cds (mapv :class-desc param-infos)
                                    mt (MethodTypeDesc/of (:class-desc return-info)
                                                          (into-array ClassDesc param-cds))
                                ;; Qualify symbols in the walked body using the deftm's source ns
                                    param-set (set (map #(if (symbol? %) % (symbol (name %))) deftm-params))
                                    qualified-wb (mapv #(inf/qualify-body-symbols % deftm-ns param-set)
                                                       (:raster.core/deftm-walked-body m))]
                                {:method-name (clojure.lang.Compiler/munge (str (:name m)))
                                 :method-type mt
                                 :params deftm-params
                                 :tags deftm-tags
                                 :return-tag return-tag
                                 :param-infos param-infos
                                 :return-info return-info
                                 :return-stack-type (:stack-type return-info)
                                 :walked-body qualified-wb
                                 :source-ns deftm-ns}))
                            deftm-refs)
        typed-sibling-methods (into {} (map (fn [ms] [(:method-name ms) ms])
                                            deftm-method-specs))
        source-file (or (:source-file (first fn-specs))
                        (when-let [ns-obj (:source-ns (first fn-specs))]
                          (some-> ns-obj str
                                  (.replace "." "/")
                                  (str ".clj")))
                        "compiled.clj")
        bytes (.build cf class-desc
                      (reify java.util.function.Consumer
                        (accept [_ cb]
                          (.withFlags cb (int 0x0011))  ;; PUBLIC + FINAL
                          (.withVersion cb (ClassFile/latestMajorVersion) (ClassFile/latestMinorVersion))
                          ;; SourceFile — shows filename in stack traces
                          (let [^java.lang.classfile.ClassBuilder cb cb]
                            (.with cb ^java.lang.classfile.ClassElement
                                   (SourceFileAttribute/of (str source-file))))
                    ;; Emit fn-spec methods (with typed params)
                          (doseq [[{:keys [method-name method-type params param-descs param-stack-types
                                           return-stack-type]}
                                   {:keys [walked-body element-type source-ns]}]
                                  (map vector method-specs fn-specs)]
                            (.withMethodBody cb method-name method-type
                                             (int 0x0009)  ;; PUBLIC + STATIC
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                                                 (let [;; JVM params occupy slots with double/long taking 2 slots.
                                  ;; Compute slot assignments respecting primitive widths.
                                                       slot-assignments (loop [idx 0, slot 0, acc []]
                                                                          (if (>= idx (count params))
                                                                            acc
                                                                            (let [st (nth param-stack-types idx :ref)
                                                                                  width (if (two-slot? st) 2 1)]
                                                                              (recur (inc idx) (+ slot width)
                                                                                     (conj acc {:slot slot :stack-type st})))))
                                  ;; Destructured elements go into slots after all params
                                                       next-slot (atom (if (seq slot-assignments)
                                                                         (+ (:slot (last slot-assignments))
                                                                            (let [st (:stack-type (last slot-assignments))]
                                                                              (if (two-slot? st) 2 1)))
                                                                         0))
                                                       locals (reduce
                                                               (fn [locs [idx p]]
                                                                 (let [{:keys [slot stack-type]} (nth slot-assignments idx)]
                                                                   (if (symbol? p)
                                                                     (let [tag (:tag (meta p))]
                                                                       ;; Checkcast ref params with known types at entry.
                                                                       ;; Skip IFn__ types — the .invk handler has its own
                                                                       ;; instanceof guard for typed/untyped callers.
                                                                       (when (and (= stack-type :ref) tag
                                                                                  (not (contains? #{'Object 'doubles 'floats
                                                                                                    'longs 'ints 'objects} tag))
                                                                                  (not (typed-interface? tag)))
                                                                         (let [cls (or (try (Class/forName (str tag))
                                                                                            (catch Exception _ nil))
                                                                                       (when-let [r (ns-resolve
                                                                                                     (or source-ns *ns*)
                                                                                                     (symbol (str tag)))]
                                                                                         (when (class? r) r)))]
                                                                           (when cls
                                                                             (.aload code slot)
                                                                             (.checkcast code (class-desc-of cls))
                                                                             (.astore code slot))))
                                                                       (assoc locs p {:slot slot :type stack-type
                                                                                      :hint (when (= stack-type :ref) tag)
                                                                                      ;; Verified only for non-Object, non-IFn__ ref types
                                                                                      ;; (Object needs checkcast to array type for aget)
                                                                                      :verified (boolean (and (= stack-type :ref) tag
                                                                                                              (not= tag 'Object)
                                                                                                              (not (typed-interface? tag))))}))
                                                 ;; Vector destructuring: p is [t0 tf ...]
                                                 ;; The JVM param at slot is Object (a seq/vector)
                                                 ;; Generate RT.nth calls to extract elements
                                                                     (reduce
                                                                      (fn [locs2 [i sym]]
                                                                        (let [s (int @next-slot)]
                                                                          (swap! next-slot inc)
                                                       ;; Emit: local_s = RT.nth(param_slot, i)
                                                                          (.aload code slot)
                                                                          (.ldc code (int i))
                                                                          (.invokestatic code rt-cd "nth"
                                                                                         (MethodTypeDesc/of obj-cd
                                                                                                            (into-array ClassDesc [obj-cd I-cd])))
                                                                          (.astore code s)
                                                                          (assoc locs2 sym {:slot s :type :ref})))
                                                                      locs
                                                                      (map-indexed vector p)))))
                                                               {} (map-indexed vector params))
                                                       ctx {:class-desc       class-desc
                                                            :element-type     element-type
                                                            :sibling-methods  sibling-methods
                                                            :typed-sibling-methods typed-sibling-methods
                                                            :next-slot        next-slot
                                                            :source-ns        source-ns}
                                                       expanded-body (binding [*ns* source-ns]
                                                                       (mapv (comp pre-expand-par-forms macroexpand-all-preserving desugar-invk pre-expand-par-forms) walked-body))
                                                       ret-type (emit-body code expanded-body locals ctx)]
                              ;; Return according to method signature type
                                                   (when-not (#{:void :diverge} ret-type)
                                                     (case return-stack-type
                                                       :double (do (when (not= ret-type :double)
                                                                     (emit-coerce code ret-type :double))
                                                                   (.dreturn code))
                                                       :long   (do (when (not= ret-type :long)
                                                                     (emit-coerce code ret-type :long))
                                                                   (.lreturn code))
                                                       :int    (do (when (not= ret-type :int)
                                                                     (emit-coerce code ret-type :int))
                                                                   (.ireturn code))
                                                       :float  (do (when (not= ret-type :float)
                                                                     (emit-coerce code ret-type :float))
                                                                   (.freturn code))
                                  ;; :ref or default — box primitives, checkcast typed return, areturn
                                                       (do (case ret-type
                                                             :ref    nil
                                                             :double (.invokestatic code dbl-box-cd "valueOf"
                                                                                    (MethodTypeDesc/of dbl-box-cd (into-array ClassDesc [D-cd])))
                                                             :float  (.invokestatic code flt-box-cd "valueOf"
                                                                                    (MethodTypeDesc/of flt-box-cd (into-array ClassDesc [F-cd])))
                                                             :int    (.invokestatic code int-box-cd "valueOf"
                                                                                    (MethodTypeDesc/of int-box-cd (into-array ClassDesc [I-cd])))
                                                             :long   (.invokestatic code long-box-cd "valueOf"
                                                                                    (MethodTypeDesc/of long-box-cd (into-array ClassDesc [J-cd])))
                                                             :void   (.aconst_null code))
                                                           ;; Checkcast when method declares typed ref return
                                                           ;; (e.g. Dual) but stack has generic Object
                                                           (let [ret-cd (.returnType ^MethodTypeDesc method-type)]
                                                             (when (and (not (.equals ^ClassDesc ret-cd obj-cd))
                                                                        (= ret-type :ref))
                                                               (.checkcast code ret-cd)))
                                                           (.areturn code))))
                                                   (when (= ret-type :void) (.return_ code)))))))
                    ;; Emit typed deftm methods as same-class siblings
                          (doseq [{:keys [method-name method-type params tags param-infos
                                          return-info walked-body source-ns]} deftm-method-specs]
                            (.withMethodBody cb method-name method-type
                                             (int 0x0009)  ;; PUBLIC + STATIC
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                                                 (let [next-slot (atom 0)
                                                       locals (reduce
                                                               (fn [locs [p info tag]]
                                                                 (let [slot (int @next-slot)
                                                                       st (:stack-type info)]
                                                                   (swap! next-slot + (if (two-slot? st) 2 1))
                                                                   ;; Checkcast ref params at entry.
                                                                   ;; Skip IFn__ types — handled by instanceof guard in .invk
                                                                   (when (and (= st :ref) tag
                                                                              (not= tag 'Object)
                                                                              (not (typed-interface? tag)))
                                                                     (let [cls (or (try (Class/forName (str tag))
                                                                                        (catch Exception _ nil))
                                                                                   (when-let [r (ns-resolve
                                                                                                 (or source-ns *ns*)
                                                                                                 (symbol (str tag)))]
                                                                                     (when (class? r) r)))]
                                                                       (when cls
                                                                         (.aload code slot)
                                                                         (.checkcast code (class-desc-of cls))
                                                                         (.astore code slot))))
                                                                   (assoc locs p {:slot slot :type st
                                                                                  :hint (when (= st :ref) tag)
                                                                                  :verified (boolean (and (= st :ref) tag
                                                                                                          (not= tag 'Object)
                                                                                                          (not (typed-interface? tag))))})))
                                                               {} (map vector params param-infos tags))
                                                       ctx {:class-desc class-desc
                                                            :next-slot  next-slot
                                                            :source-ns  source-ns
                                                            :typed-sibling-methods typed-sibling-methods}
                                                       expanded-body (binding [*ns* source-ns]
                                                                       (mapv (comp pre-expand-par-forms macroexpand-all-preserving desugar-invk pre-expand-par-forms) walked-body))
                                                       ret-type (emit-body code expanded-body locals ctx)
                                                       target-st (:stack-type return-info)
                                                       ret-cd (:class-desc return-info)]
                                                   (when-not (#{:void :diverge} ret-type)
                                                     (when (not= ret-type target-st)
                                                       (emit-coerce code ret-type target-st))
                                                     ;; Checkcast when method declares typed reference return
                                                     ;; (e.g. Dual) but body returns generic Object
                                                     (when (and (= target-st :ref)
                                                                (= ret-type :ref)
                                                                ret-cd
                                                                (not (.equals ^ClassDesc ret-cd obj-cd)))
                                                       (.checkcast code ret-cd))
                                                     (case target-st
                                                       :double (.dreturn code) :long (.lreturn code)
                                                       :int (.ireturn code) :float (.freturn code)
                                                       :void (.return_ code)
                                                       :ref (.areturn code)))
                                                   (when (= ret-type :void)
                                                     (.return_ code))))))))))
        ;; Verify bytecode before loading
        _ (try (.verify (java.lang.classfile.ClassFile/of) bytes)
               (catch Exception e
                 (println "WARNING: Bytecode verification failed for" class-name ":" (.getMessage e))))
        loader (compilation-loader)
        cls (.defineClass loader class-name bytes nil)]
    {:class cls
     :bytes bytes
     :methods (into {} (map (fn [ms]
                              [(:name ms)
                               (first (filter #(= (.getName %) (:method-name ms))
                                              (.getMethods cls)))])
                            method-specs))}))

;; ================================================================
;; deftm bytecode compilation
;; ================================================================
;;
;; Compiles arbitrary deftm bodies to typed JVM static methods.
;; Called on-demand when the bytecode compiler encounters a deftm
;; call with a complex body (not a single Java static call).

(defn compile-deftm-method!
  "Compile a deftm body to a typed JVM static method.
  Delegates to compile-specialized-class! with a single fn-spec,
  ensuring one code path for param handling, body expansion, and return coercion.
  Returns {:class Class, :method Method, :class-desc ClassDesc,
           :method-type MethodTypeDesc, :return-stack-type keyword}"
  [class-name method-name params tags return-tag walked-body source-ns]
  (let [;; Build fn-spec with tagged params
        tagged-params (mapv (fn [p t] (vary-meta p assoc :tag t)) params tags)
        spec {:name (symbol method-name)
              :params tagged-params
              :return-tag (or return-tag 'Object)
              :walked-body walked-body
              :source-ns source-ns}
        result (compile-specialized-class! class-name [spec])
        method-sym (symbol method-name)
        method (get (:methods result) method-sym)
        ;; Compute method-type and return-stack-type for callers that need them
        param-jvm-infos (mapv #(param-tag->jvm %) tags)
        ret-jvm (param-tag->jvm (or return-tag 'Object))
        mt (MethodTypeDesc/of (:class-desc ret-jvm)
                              (into-array ClassDesc (mapv :class-desc param-jvm-infos)))]
    {:class (:class result)
     :method method
     :class-desc (let [dot (.lastIndexOf ^String class-name (int \.))]
                   (ClassDesc/of (subs class-name 0 dot) (subs class-name (inc dot))))
     :method-type mt
     :return-stack-type (:stack-type ret-jvm)
     :bytes (:bytes result)}))

(def ^:dynamic *compiling-deftms*
  "Set of deftm var symbols currently being compiled, for cycle detection."
  #{})

(defn- compile-and-cache-deftm!
  "Lazily compile a deftm body to bytecode, caching the result on var metadata."
  [deftm-var]
  (let [m (meta deftm-var)
        var-sym (symbol (str (ns-name (:ns m))) (str (:name m)))]
    (when (contains? *compiling-deftms* var-sym)
      (throw (ex-info (str "Cycle detected compiling deftm: " var-sym)
                      {:var var-sym})))
    (let [walked-body (:raster.core/deftm-walked-body m)
          params      (:raster.core/deftm-params m)
          tags        (:raster.core/deftm-tags m)
          return-tag  (:raster.core/return-tag m)
          source-ns   (:ns m)
          class-name  (str (ns-name source-ns) "." (:name m) "__DeftmBC")
          bc-info     (binding [*compiling-deftms* (conj *compiling-deftms* var-sym)]
                        (compile-deftm-method! class-name "invoke"
                                               params tags return-tag walked-body source-ns))]
      (alter-meta! deftm-var assoc :raster.core/deftm-bytecode bc-info)
      bc-info)))

;; ================================================================
;; compile-typed-impl! — replace deftm deftype with bytecode-compiled impl
;; ================================================================

(defn- class-desc-of-class
  "Create a ClassDesc from a Class object, handling primitives and arrays."
  [^Class c]
  (cond
    (= c Double/TYPE) D-cd
    (= c Long/TYPE) J-cd
    (= c Integer/TYPE) I-cd
    (= c Float/TYPE) F-cd
    (= c Boolean/TYPE) (ClassDesc/ofDescriptor "Z")
    (= c Byte/TYPE) (ClassDesc/ofDescriptor "B")
    (= c Short/TYPE) (ClassDesc/ofDescriptor "S")
    (= c Character/TYPE) (ClassDesc/ofDescriptor "C")
    (= c Void/TYPE) V-cd
    (.isArray c) (ClassDesc/ofDescriptor (str "[" (.descriptorString (class-desc-of-class (.getComponentType c)))))
    :else (ClassDesc/ofDescriptor (str "L" (.replace (.getName c) "." "/") ";"))))

;; ================================================================
;; Body splitting for large methods
;; ================================================================

(defn- free-syms-in
  "Collect unqualified free symbols in a form. Delegates to the canonical
   scope-aware util/free-syms which handles all binding forms and correctly
   includes unqualified symbols in function position (local fn calls)."
  [form bound]
  (util/free-syms form bound))

(defn- optimize-chunk
  "Apply normalize pass to chunk body before bytecoding.
   Converts .invk calls to direct ops (Math/fma, clojure.core/+, etc.)
   which the bytecoder compiles to smaller, tighter bytecode.
   This reduces the total inlined code size so C2's compilation unit
   stays within its node budget after chain-inlining."
  [walked-body]
  (try
    (let [normalize (requiring-resolve 'raster.compiler.passes.scalar.normalize/normalize)]
      (mapv normalize walked-body))
    (catch Exception _
      walked-body)))

(defn- split-body-into-helpers
  "Extract par forms from a let-body into helper static methods.
   Scans both body forms and let-binding init expressions for par forms.
   Par forms are self-contained array loops with clear inputs (free variables)
   and outputs (array or scalar). Non-par forms stay in the main method.

   Uses descriptors/par-form? for detection (centralized, not hardcoded).
   Each helper receives only its free variables as typed params.
   Primitive-constant vars are excluded (bytecoder constant-folds them to LDC).
   Normalize pass is applied to extracted par forms for tighter bytecode.

   Returns {:main-bindings [sym init ...], :main-body [forms], :helpers [fn-spec...]}"
  [body-forms let-bindings fn-params fn-tags source-ns _target-chunk-size]
  (let [par-desc (requiring-resolve 'raster.compiler.passes.parallel.descriptors/par-form-info)
        ;; Build type env from walker-annotated symbol metadata + fn-params
        let-pairs (vec (partition 2 let-bindings))
        let-type-env (into {}
                           (for [[sym _init] let-pairs
                                 :let [tag (or (:raster.type/tag (meta sym))
                                               (:tag (meta sym)))]
                                 :when tag]
                             [sym tag]))
        ;; Bound set for free-syms: only let-binding names.
        ;; fn-params are NOT included because extracted helpers need them
        ;; as explicit params (they don't have access to the parent method's locals).
        ;; clojure.core vars are excluded by free-syms itself (ns-resolve check).
        let-bound (set (map first let-pairs))
        ;; Infer tag for a symbol from fn-params, walker metadata, or let env
        tag-for-sym (fn [sym]
                      (let [idx (.indexOf (vec fn-params) sym)]
                        (if (>= idx 0)
                          (nth fn-tags idx)
                          (or (get let-type-env sym)
                              (:raster.type/tag (meta sym))
                              (:tag (meta sym))
                              'Object))))
        ;; Exclude primitive-constant vars — bytecoder constant-folds them to LDC
        primitive-const? (fn [sym]
                           (when-let [v (try (ns-resolve source-ns sym)
                                             (catch Exception _ nil))]
                             (when (var? v)
                               (let [root (try (.getRawRoot ^clojure.lang.Var v)
                                               (catch Exception _ nil))]
                                 (or (instance? Double root)
                                     (instance? Long root)
                                     (instance? Integer root))))))
        helpers (atom [])
        helper-counter (atom 0)
        extract-par (fn [form]
                      (let [info (par-desc form)]
                        (if info
                          ;; Extract par form as helper
                          (let [helper-name (symbol (str "par_" (swap! helper-counter inc)))
                                ;; Empty bound set: helpers are separate methods, all
                                ;; referenced symbols must be passed as params.
                                ;; clojure.core vars excluded by free-syms ns-resolve.
                                free (util/free-syms form #{})
                                free (set (remove primitive-const? free))
                                ordered-free (vec (concat
                                                   (filter free (vec fn-params))
                                                   (sort (remove (set fn-params) free))))
                                helper-params (mapv (fn [p]
                                                      (vary-meta p assoc :tag (tag-for-sym p)))
                                                    ordered-free)
                                ;; Infer array tag from :cast field (float → floats, double → doubles)
                                cast->arr-tag (fn [cast-sym]
                                                (case cast-sym
                                                  float 'floats
                                                  (double nil) 'doubles))
                                ;; Return type from par-form-info :type + output array tag
                                ret-tag (case (:type info)
                                          :reduce (or (when-let [et (:elem-type info)]
                                                        (case et :float 'float :double 'double nil))
                                                      (let [init (:init info)]
                                                        (when (and (seq? init) (= 'float (first init)))
                                                          'float))
                                                      (throw (ex-info "par/reduce missing :elem-type — type metadata lost in pipeline"
                                                                      {:init (:init info)})))
                                          :map-void 'Object
                                          :map2 'Object
                                          :butterfly 'Object
                                          ;; :map, :stencil — return type matches the output array
                                          (or (when-let [out-sym (:out info)]
                                                (let [t (tag-for-sym out-sym)]
                                                  (when (not= t 'Object) t)))
                                              (cast->arr-tag (:cast info))
                                              'doubles))]
                            (swap! helpers conj
                                   {:name helper-name
                                    :params helper-params
                                    :return-tag ret-tag
                                    :walked-body (optimize-chunk [form])
                                    :source-ns source-ns})
                            (apply list helper-name ordered-free))
                          ;; Not a par form — keep as-is
                          form)))
        ;; Rewrite let-binding init expressions
        rewritten-bindings (vec (mapcat (fn [[sym init]]
                                          [sym (extract-par init)])
                                        let-pairs))
        ;; Rewrite body forms
        rewritten-body (mapv extract-par body-forms)]
    {:main-bindings rewritten-bindings
     :main-body rewritten-body
     :helpers @helpers}))

(defn compile-typed-impl!
  "Compile a deftm walked body into a class implementing the typed interface.
  Returns an instance of the compiled class, or nil on failure.

  The compiled class has a single `invk` method that:
  1. Checkcasts Object params to the types expected by the static method
  2. Calls invokestatic on the compiled static method
  3. Returns the result

  This replaces the Clojure-compiled deftype body with optimized bytecode
  (branch folding, int loops, checkcast elimination)."
  [class-name params tags return-tag walked-body source-ns typed-iface-name
   & [fallback-ns fallback-name]]
  (when (> (count params) 20)
    (throw (ex-info "BC compilation: function has more than 20 parameters (IFn limit)"
                    {:count (count params) :class class-name})))
  (let [;; Par-extraction splitting: extract par/map! and par/reduce forms into
        ;; separate static methods. Keeps scalar code in the main method.
        ;; Uses size estimate to avoid splitting small bodies unnecessarily.
        form (first walked-body)
        needs-split? (and (seq? form)
                          (contains? #{'let 'let*} (first form))
                          (let [[_ binds & body] form]
                            (let [par-form? (requiring-resolve 'raster.compiler.passes.parallel.descriptors/par-form?)
                                  bind-inits (map second (partition 2 binds))
                                  all-forms (concat bind-inits body)]
                              (and (some par-form? all-forms)
                                   (> (reduce + (map split/estimate-form-size all-forms))
                                      split/TARGET-SIZE)))))
        ;; Step 1: compile walked body to static method(s)
        result (if needs-split?
                 ;; Split path: extract par forms into helper methods
                 (let [[_ binds & body] form
                       {:keys [main-bindings main-body helpers]}
                       (split-body-into-helpers body binds params tags source-ns 2)
                       ;; Rebuild let with rewritten bindings (par forms replaced by helper calls)
                       main-form (list* 'let main-bindings main-body)
                       ;; Tag main fn-spec params
                       tagged-params (mapv (fn [p t]
                                             (vary-meta p assoc :tag t))
                                           params tags)
                       main-spec {:name 'compute_static
                                  :params tagged-params
                                  :return-tag (or return-tag 'Object)
                                  :walked-body [main-form]
                                  :source-ns source-ns}
                       all-specs (vec (cons main-spec helpers))
                       compiled (compile-specialized-class! class-name all-specs)]
                   {:class (:class compiled)
                    :method (get (:methods compiled) 'compute_static)
                    :bytes (:bytes compiled)})
                 ;; Simple path: single method
                 (compile-deftm-method!
                  class-name "compute_static"
                  params tags return-tag
                  walked-body source-ns))
        statics-class (:class result)
        compute-method (:method result)
        ;; Step 2: generate wrapper class implementing the typed interface
        iface-class (if (class? typed-iface-name)
                      typed-iface-name
                      (Class/forName (str typed-iface-name)))
        iface-invk (first (filter #(= "invk" (.getName %))
                                  (.getDeclaredMethods iface-class)))
        iface-param-types (.getParameterTypes iface-invk)
        param-cds (mapv class-desc-of-class iface-param-types)
        ret-cd (class-desc-of-class (.getReturnType iface-invk))
        compute-params (.getParameterTypes compute-method)
        compute-param-cds (mapv class-desc-of-class compute-params)
        compute-ret-cd (class-desc-of-class (.getReturnType compute-method))
        cf (ClassFile/of)
        wrapper-name (str class-name "$TI")
        wdot (.lastIndexOf ^String wrapper-name (int \.))
        wrapper-cd (ClassDesc/of (subs wrapper-name 0 wdot)
                                 (subs wrapper-name (inc wdot)))
        iface-cd (class-desc-of-class iface-class)
        statics-cd (class-desc-of-class statics-class)
        invk-mt (MethodTypeDesc/of ret-cd (into-array ClassDesc param-cds))
        compute-mt (MethodTypeDesc/of compute-ret-cd
                                      (into-array ClassDesc compute-param-cds))
        bytes (.build cf wrapper-cd
                      (reify java.util.function.Consumer
                        (accept [_ cb]
                          (.withFlags cb (int 0x0011)) ;; PUBLIC FINAL
                          (.withVersion cb (ClassFile/latestMajorVersion)
                                        (ClassFile/latestMinorVersion))
                    ;; Implement both the typed interface AND IFn so the
                    ;; compiled impl can be used as a regular Clojure function.
                    ;; This enables deftm vars to hold the impl directly.
                          (.withInterfaceSymbols cb (into-array ClassDesc [iface-cd ifn-cd]))
                    ;; No-arg constructor
                          (.withMethodBody cb "<init>"
                                           (MethodTypeDesc/of V-cd (into-array ClassDesc []))
                                           (int 0x0001)
                                           (reify java.util.function.Consumer
                                             (accept [_ code]
                                               (.aload code 0)
                                               (.invokespecial code obj-cd "<init>"
                                                               (MethodTypeDesc/of V-cd (into-array ClassDesc [])))
                                               (.return_ code))))
                    ;; Shared helper: load params, checkcast IFn__ types, call compute_static
                          (let [;; Find IFn__ params: iface=Object but compute expects IFn__
                                param-slots (loop [i 0, slot 1, acc []]
                                              (if (>= i (count param-cds))
                                                acc
                                                (let [pd (.descriptorString ^ClassDesc (nth param-cds i))
                                                      w (if (#{"D" "J"} pd) 2 1)]
                                                  (recur (inc i) (+ slot w) (conj acc slot)))))
                                guarded-indices (vec (keep (fn [i]
                                                             (let [ipd (.descriptorString ^ClassDesc (nth param-cds i))
                                                                   cpd (.descriptorString ^ClassDesc (nth compute-param-cds i))]
                                                               (when (and (= ipd "Ljava/lang/Object;")
                                                                          (typed-interface? cpd))
                                                                 i)))
                                                           (range (count param-cds))))
                                emit-load-and-cast
                                (fn [code]
                                  (loop [slot 1 i 0]
                                    (when (< i (count param-cds))
                                      (let [iface-pd (.descriptorString ^ClassDesc (nth param-cds i))
                                            compute-pd (.descriptorString ^ClassDesc (nth compute-param-cds i))]
                                        (cond
                                          (= iface-pd "D") (do (.dload code slot) (recur (+ slot 2) (inc i)))
                                          (= iface-pd "J") (do (.lload code slot) (recur (+ slot 2) (inc i)))
                                          (= iface-pd "F") (do (.fload code slot) (recur (+ slot 1) (inc i)))
                                          (#{"I" "Z" "B" "S" "C"} iface-pd) (do (.iload code slot) (recur (+ slot 1) (inc i)))
                                          :else (do (.aload code slot)
                                                    (when (not= iface-pd compute-pd)
                                                      (.checkcast code (nth compute-param-cds i)))
                                                    (recur (+ slot 1) (inc i)))))))
                                  (.invokestatic code statics-cd "compute_static" compute-mt))]
                      ;; invk: IFn__ interface method. For IFn__ params, emit instanceof
                      ;; guard since callers may pass plain dispatch functions (not IFn__).
                            (.withMethodBody cb "invk" invk-mt (int 0x0001)
                                             (reify java.util.function.Consumer
                                               (accept [_ code]
                                                 (if (empty? guarded-indices)
                                                   (do (emit-load-and-cast code)
                                                       (let [crd (.descriptorString ^ClassDesc compute-ret-cd)]
                                                         (cond
                                                           (= crd "D") (emit-box-to-ref code :double)
                                                           (= crd "J") (emit-box-to-ref code :long)
                                                           (= crd "F") (emit-box-to-ref code :float)
                                                           (= crd "Z") (let [bool-cd (ClassDesc/of "java.lang" "Boolean")]
                                                                         (.invokestatic code bool-cd "valueOf"
                                                                                        (MethodTypeDesc/of bool-cd (into-array ClassDesc [(ClassDesc/ofDescriptor "Z")]))))
                                                           (#{"I" "B" "S" "C"} crd) (emit-box-to-ref code :int)
                                                           (= crd "V") (.aconst_null code)
                                                           :else nil)
                                                         (.areturn code)))
                              ;; Has IFn__ params — instanceof guard at boundary
                                                   (let [boxed-label (.newLabel code)]
                                                     (doseq [gi guarded-indices]
                                                       (.aload code (nth param-slots gi))
                                                       (.instanceOf code (nth compute-param-cds gi))
                                                       (.ifeq code boxed-label))
                                                     (emit-load-and-cast code)
                                                     (let [crd (.descriptorString ^ClassDesc compute-ret-cd)]
                                                       (cond
                                                         (= crd "D") (emit-box-to-ref code :double)
                                                         (= crd "J") (emit-box-to-ref code :long)
                                                         (= crd "F") (emit-box-to-ref code :float)
                                                         (= crd "Z") (let [bool-cd (ClassDesc/of "java.lang" "Boolean")]
                                                                       (.invokestatic code bool-cd "valueOf"
                                                                                      (MethodTypeDesc/of bool-cd (into-array ClassDesc [(ClassDesc/ofDescriptor "Z")]))))
                                                         (#{"I" "B" "S" "C"} crd) (emit-box-to-ref code :int)
                                                         (= crd "V") (.aconst_null code)
                                                         :else nil)
                                                       (.areturn code))
                                ;; Boxed fallback
                                                     (.labelBinding code boxed-label)
                                                     (.ldc code (or fallback-ns (str (ns-name source-ns))))
                                                     (.ldc code (or fallback-name (str class-name)))
                                                     (.invokestatic code rt-cd "var"
                                                                    (MethodTypeDesc/of var-cd (into-array ClassDesc [string-cd string-cd])))
                                                     (.invokevirtual code var-cd "getRawRoot"
                                                                     (MethodTypeDesc/of obj-cd no-cd))
                                                     (.checkcast code ifn-cd)
                                                     (doseq [i (range (count param-cds))]
                                                       (let [pd (.descriptorString ^ClassDesc (nth param-cds i))
                                                             slot (nth param-slots i)]
                                                         (cond
                                                           (= pd "D") (do (.dload code slot) (emit-box-to-ref code :double))
                                                           (= pd "J") (do (.lload code slot) (emit-box-to-ref code :long))
                                                           (= pd "F") (do (.fload code slot) (emit-box-to-ref code :float))
                                                           (= pd "Z") (do (.iload code slot)
                                                                          (let [bool-cd (ClassDesc/of "java.lang" "Boolean")]
                                                                            (.invokestatic code bool-cd "valueOf"
                                                                                           (MethodTypeDesc/of bool-cd (into-array ClassDesc [(ClassDesc/ofDescriptor "Z")])))))
                                                           (#{"I" "B" "S" "C"} pd) (do (.iload code slot) (emit-box-to-ref code :int))
                                                           :else (.aload code slot))))
                                                     (.invokeinterface code ifn-cd "invoke"
                                                                       (MethodTypeDesc/of obj-cd (into-array ClassDesc (repeat (count param-cds) obj-cd))))
                                                     (.areturn code))))))
                      ;; compute_prim: same params but primitive return (zero-boxing invokedynamic).
                      ;; No instanceof guard — only reached via invokedynamic from JIT'd code
                      ;; where the walker guarantees typed arguments.
                            (let [crd (.descriptorString ^ClassDesc compute-ret-cd)]
                              (when (#{"D" "J" "F" "I" "Z" "B" "S" "C"} crd)
                                (let [prim-mt (MethodTypeDesc/of compute-ret-cd (into-array ClassDesc param-cds))]
                                  (.withMethodBody cb "compute_prim" prim-mt (int 0x0001)
                                                   (reify java.util.function.Consumer
                                                     (accept [_ code]
                                                       (emit-load-and-cast code)
                                                       (cond
                                                         (= crd "D") (.dreturn code)
                                                         (= crd "J") (.lreturn code)
                                                         (= crd "F") (.freturn code)
                                                         (#{"I" "Z" "B" "S" "C"} crd) (.ireturn code)))))))))
                    ;; IFn.invoke(Object...) → Object: unbox args, call compute_static, box result.
                    ;; For IFn__ params: checkcast to the typed interface. If the caller
                    ;; passes a plain fn, this throws ClassCastException — the contractual
                    ;; behavior for the typed path. Plain fn callers should use IFn.invoke
                    ;; on the dispatch var (which routes through the boxed dispatch fn).
                          (when (<= (count param-cds) 20)
                            (let [invoke-param-cds (into-array ClassDesc (repeat (count param-cds) obj-cd))
                                  invoke-mt (MethodTypeDesc/of obj-cd invoke-param-cds)]
                              (.withMethodBody cb "invoke" invoke-mt (int 0x0001)
                                               (reify java.util.function.Consumer
                                                 (accept [_ code]
                              ;; Unbox each Object arg to the compute_static param type
                                                   (loop [slot 1 i 0]
                                                     (when (< i (count compute-param-cds))
                                                       (let [cpd (.descriptorString ^ClassDesc (nth compute-param-cds i))]
                                                         (.aload code slot)
                                                         (cond
                                                           (= cpd "D")
                                                           (let [num-lbl (.newLabel code)
                                                                 end-lbl (.newLabel code)]
                                        ;; instanceof Number guard — matches emit-coerce [:ref :double]
                                                             (.dup code)
                                                             (.instanceOf code number-cd)
                                                             (.ifeq code num-lbl)
                                                             (.checkcast code number-cd)
                                                             (.invokevirtual code number-cd "doubleValue"
                                                                             (MethodTypeDesc/of D-cd no-cd))
                                                             (.goto_ code end-lbl)
                                                             (.labelBinding code num-lbl)
                                                             (.pop code)
                                                             (.ldc code (double 0.0))
                                                             (.labelBinding code end-lbl))
                                                           (= cpd "J") (do (.checkcast code number-cd)
                                                                           (.invokevirtual code number-cd "longValue"
                                                                                           (MethodTypeDesc/of J-cd no-cd)))
                                                           (= cpd "F") (do (.checkcast code number-cd)
                                                                           (.invokevirtual code number-cd "floatValue"
                                                                                           (MethodTypeDesc/of F-cd no-cd)))
                                                           (= cpd "Z")
                                                           (let [bool-cd (ClassDesc/of "java.lang" "Boolean")]
                                                             (.checkcast code bool-cd)
                                                             (.invokevirtual code bool-cd "booleanValue"
                                                                             (MethodTypeDesc/of (ClassDesc/ofDescriptor "Z") no-cd)))
                                                           (#{"I" "B" "S" "C"} cpd)
                                                           (do (.checkcast code number-cd)
                                                               (.invokevirtual code number-cd "intValue"
                                                                               (MethodTypeDesc/of I-cd no-cd)))
                                      ;; Ref types: checkcast if not Object
                                                           :else (when (not= cpd "Ljava/lang/Object;")
                                                                   (.checkcast code (nth compute-param-cds i))))
                                                         (recur (inc slot) (inc i)))))
                              ;; Call compute_static (invokestatic — cheapest call)
                                                   (.invokestatic code statics-cd "compute_static" compute-mt)
                              ;; Box primitive return to Object
                                                   (let [crd (.descriptorString ^ClassDesc compute-ret-cd)]
                                                     (cond
                                                       (= crd "D") (.invokestatic code dbl-box-cd "valueOf"
                                                                                  (MethodTypeDesc/of dbl-box-cd (into-array ClassDesc [D-cd])))
                                                       (= crd "J") (.invokestatic code long-box-cd "valueOf"
                                                                                  (MethodTypeDesc/of long-box-cd (into-array ClassDesc [J-cd])))
                                                       (= crd "F") (.invokestatic code flt-box-cd "valueOf"
                                                                                  (MethodTypeDesc/of flt-box-cd (into-array ClassDesc [F-cd])))
                                                       (= crd "Z")
                                                       (let [bool-box-cd (ClassDesc/of "java.lang" "Boolean")
                                                             bool-cd (ClassDesc/ofDescriptor "Z")]
                                                         (.invokestatic code bool-box-cd "valueOf"
                                                                        (MethodTypeDesc/of bool-box-cd (into-array ClassDesc [bool-cd]))))
                                                       (#{"I" "B" "S" "C"} crd)
                                                       (.invokestatic code int-box-cd "valueOf"
                                                                      (MethodTypeDesc/of int-box-cd (into-array ClassDesc [I-cd])))
                                                       (= crd "V") (.aconst_null code)
                                                       :else nil))
                                                   (.areturn code)))))
                      ;; applyTo for varargs
                            (let [afn-cd (ClassDesc/of "clojure.lang" "AFn")
                                  iseq-cd (ClassDesc/of "clojure.lang" "ISeq")
                                  apply-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc [ifn-cd iseq-cd]))]
                              (.withMethodBody cb "applyTo"
                                               (MethodTypeDesc/of obj-cd (into-array ClassDesc [iseq-cd]))
                                               (int 0x0001)
                                               (reify java.util.function.Consumer
                                                 (accept [_ code]
                                                   (.aload code 0)
                                                   (.aload code 1)
                                                   (.invokestatic code afn-cd "applyToHelper" apply-mt)
                                                   (.areturn code)))))))))
        loader (compilation-loader)
        wrapper-class (.defineClass loader wrapper-name bytes nil)]
    (.newInstance (.getDeclaredConstructor wrapper-class (into-array Class []))
                  (into-array Object []))))

;; ================================================================
;; compile-pipeline! — batch recompilation into a single class
;; ================================================================

(defn compile-pipeline!
  "Recompile specialized functions into a single JVM class.
  All inter-function calls become same-class invokestatic with typed params.

  vars: a sequence of vars that have ::pipeline-fn-spec metadata
  (set by specialize-fn! in core.clj).

  Returns {:class Class, :methods {fn-name -> Method}}"
  [class-name vars]
  (let [fn-specs (mapv (fn [v]
                         (let [m (meta v)
                               spec (:raster.core/pipeline-fn-spec m)]
                           (when-not spec
                             (throw (ex-info (str "Var " v " has no ::pipeline-fn-spec metadata. "
                                                  "Was it created with specialize?")
                                             {:var v})))
                           spec))
                       vars)
        bc-result (compile-specialized-class! class-name fn-specs)]
    ;; Update each var's metadata and wrapper function
    (doseq [v vars]
      (let [m (meta v)
            spec (:raster.core/pipeline-fn-spec m)
            fn-name (:name spec)
            bc-method (get (:methods bc-result) fn-name)
            _ (when-not bc-method
                (throw (ex-info (str "compile-pipeline!: method not found for " fn-name)
                                {:fn-name fn-name :available (keys (:methods bc-result))})))
            source-params (:params spec)
            param-syms (filterv #(or (symbol? %) (vector? %)) source-params)
            n (count param-syms)
            wrapper (case n
                      0 (fn [] (.invoke bc-method nil (object-array 0)))
                      1 (fn [a] (.invoke bc-method nil (object-array [a])))
                      2 (fn [a b] (.invoke bc-method nil (object-array [a b])))
                      3 (fn [a b c] (.invoke bc-method nil (object-array [a b c])))
                      4 (fn [a b c d] (.invoke bc-method nil (object-array [a b c d])))
                      5 (fn [a b c d e] (.invoke bc-method nil (object-array [a b c d e])))
                      6 (fn [a b c d e f] (.invoke bc-method nil (object-array [a b c d e f])))
                      (fn [& args] (.invoke bc-method nil (object-array args))))
            target-ns (or (:source-ns spec) *ns*)]
        ;; Replace the var's root binding with the new wrapper
        (.bindRoot ^clojure.lang.Var v wrapper)
        ;; Update metadata to point to the new class/method
        (alter-meta! v assoc
                     :raster.core/bytecode-class (:class bc-result)
                     :raster.core/bytecode-method bc-method
                     :raster.core/bytecode-bytes (:bytes bc-result))))
    bc-result))
