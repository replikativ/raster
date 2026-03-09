(ns raster.compiler.backend.jvm.hoisted
  "Generates hoisted wrapper classes for compiled pipeline functions.
   The wrapper extends AFn (implements IFn) and holds hoisted buffers
   as instance fields. Delegates to static compute methods.

   The wrapper handles its own lifecycle: on the first invoke() call,
   it allocates buffers by calling the initFn. No Clojure fn wrapper
   needed on the hot path — the wrapper IS the callable."
  (:require [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as me])
  (:import [java.lang.classfile ClassFile]
           [java.lang.classfile.attribute SourceFileAttribute]
           [java.lang.constant ClassDesc MethodTypeDesc]))

(def ^:private obj-cd (ClassDesc/of "java.lang" "Object"))
(def ^:private number-cd (ClassDesc/of "java.lang" "Number"))
(def ^:private ifn-cd (ClassDesc/of "clojure.lang" "IFn"))
(def ^:private D-cd (ClassDesc/ofDescriptor "D"))
(def ^:private J-cd (ClassDesc/ofDescriptor "J"))
(def ^:private I-cd (ClassDesc/ofDescriptor "I"))
(def ^:private F-cd (ClassDesc/ofDescriptor "F"))
(def ^:private Z-cd (ClassDesc/ofDescriptor "Z"))
(def ^:private dbl-box-cd (ClassDesc/of "java.lang" "Double"))
(def ^:private long-box-cd (ClassDesc/of "java.lang" "Long"))
(def ^:private int-box-cd (ClassDesc/of "java.lang" "Integer"))
(def ^:private flt-box-cd (ClassDesc/of "java.lang" "Float"))
;; AFunction extends AFn and implements IObj (metadata), Comparator, Serializable
(def ^:private afn-cd (ClassDesc/of "clojure.lang" "AFunction"))

(defn- class-desc-of [^Class cls]
  (ClassDesc/ofDescriptor (.descriptorString cls)))

(defn- tag->jvm-info [tag _source-ns]
  (let [ptj @(resolve 'raster.compiler.backend.jvm.bytecode/param-tag->jvm)
        info (ptj tag)]
    (assoc info :class Object)))

(defn- emit-load [code stack-type slot]
  (case stack-type
    :double (.dload code (int slot))
    :long (.lload code (int slot))
    :float (.fload code (int slot))
    :int (.iload code (int slot))
    (.aload code (int slot))))

(defn- emit-return [code stack-type]
  (case stack-type
    :double (.dreturn code)
    :long (.lreturn code)
    :float (.freturn code)
    :int (.ireturn code)
    :void (.return_ code)
    (.areturn code)))

(defn- emit-unbox
  "Emit bytecode to unbox an Object on the stack to a typed primitive/reference."
  [code stack-type target-cd empty-cds]
  (case stack-type
    :double (do (.checkcast code number-cd)
                (.invokevirtual code number-cd "doubleValue" (MethodTypeDesc/of D-cd empty-cds)))
    :long (do (.checkcast code number-cd)
              (.invokevirtual code number-cd "longValue" (MethodTypeDesc/of J-cd empty-cds)))
    :float (do (.checkcast code number-cd)
               (.invokevirtual code number-cd "floatValue" (MethodTypeDesc/of F-cd empty-cds)))
    :int (do (.checkcast code number-cd)
             (.invokevirtual code number-cd "intValue" (MethodTypeDesc/of I-cd empty-cds)))
    (when (not (.equals ^ClassDesc target-cd obj-cd))
      (.checkcast code target-cd))))

(defn- emit-box
  "Emit bytecode to box a primitive stack value to Object."
  [code stack-type empty-cds]
  (case stack-type
    :double (.invokestatic code dbl-box-cd "valueOf"
                           (MethodTypeDesc/of dbl-box-cd (into-array ClassDesc [D-cd])))
    :long (.invokestatic code long-box-cd "valueOf"
                         (MethodTypeDesc/of long-box-cd (into-array ClassDesc [J-cd])))
    :float (.invokestatic code flt-box-cd "valueOf"
                          (MethodTypeDesc/of flt-box-cd (into-array ClassDesc [F-cd])))
    :int (.invokestatic code int-box-cd "valueOf"
                        (MethodTypeDesc/of int-box-cd (into-array ClassDesc [I-cd])))
    nil)) ;; :ref — already Object

(defn compile-hoisted-class!
  "Compile static methods + instance wrapper class for hoisted pipeline.

   The wrapper class:
   - Extends clojure.lang.AFn (implements IFn)
   - Holds hoisted buffers as instance fields
   - Has an initFn field for lazy buffer allocation
   - invoke(Object...) checks initialized flag, allocates on first call,
     then unboxes args → invk → boxes result. No Clojure fn wrapper needed.

   Returns {:statics-class Class, :wrapper-class Class, :methods {...}}."
  [class-name fn-specs instance-spec]
  (let [statics-result (bytecode/compile-specialized-class! class-name fn-specs)
        statics-class (:class statics-result)
        statics-cd (class-desc-of statics-class)
        {:keys [field-tags fn-param-tags return-tag compute-name iface-class]} instance-spec
        src-ns (or (:source-ns (first fn-specs)) *ns*)
        field-infos (mapv #(tag->jvm-info % src-ns) field-tags)
        field-names (mapv #(str "f" %) (range (count field-tags)))
        field-cds (mapv :class-desc field-infos)
        fn-param-infos (mapv #(tag->jvm-info % src-ns) fn-param-tags)
        ;; IFn__ typed params (function parameters) use Object in the method
        ;; signature. The compiled body has an instanceof guard for typed dispatch.
        ;; This allows callers to pass plain IFn dispatch functions.
        fn-param-cds (mapv (fn [info tag]
                             (if (and (= :ref (:stack-type info))
                                      (symbol? tag)
                                      (.contains (str tag) "IFn__"))
                               obj-cd
                               (:class-desc info)))
                           fn-param-infos fn-param-tags)
        ret-info (tag->jvm-info return-tag src-ns)
        ret-cd (:class-desc ret-info)
        iface-cd (class-desc-of iface-class)
        compute-method (get (:methods statics-result) compute-name)
        compute-param-types (.getParameterTypes ^java.lang.reflect.Method compute-method)
        compute-mt (MethodTypeDesc/of
                    (class-desc-of (.getReturnType ^java.lang.reflect.Method compute-method))
                    (into-array ClassDesc (mapv class-desc-of compute-param-types)))
        compute-mn (.getName ^java.lang.reflect.Method compute-method)
        void-cd (ClassDesc/ofDescriptor "V")
        empty-cds (into-array ClassDesc [])
        wrapper-name (str class-name "$W")
        cf (ClassFile/of)
        wdot (.lastIndexOf ^String wrapper-name (int \.))
        wpkg (subs wrapper-name 0 wdot)
        wsimple (subs wrapper-name (inc wdot))
        wrapper-cd (ClassDesc/of wpkg wsimple)
        n-fn-params (count fn-param-tags)
        n-fields (count field-tags)
        invoke-params (into-array ClassDesc (repeat n-fn-params obj-cd))
        invoke-mt (MethodTypeDesc/of obj-cd invoke-params)
        ;; init method: takes same Object params as invoke, calls initFn, stores fields
        init-mt (MethodTypeDesc/of void-cd invoke-params)
        wrapper-bytes
        (.build cf wrapper-cd
                (reify java.util.function.Consumer
                  (accept [_ cb]
                    (.withFlags cb (int 0x0011)) ;; PUBLIC + FINAL
                    (.withVersion cb (ClassFile/latestMajorVersion) (ClassFile/latestMinorVersion))
                    (.withSuperclass cb afn-cd)
                    (.withInterfaceSymbols cb (into-array ClassDesc [iface-cd]))
              ;; SourceFile — shows meaningful name in stack traces
                    (let [^java.lang.classfile.ClassBuilder cb cb]
                      (.with cb ^java.lang.classfile.ClassElement
                             (SourceFileAttribute/of (str wrapper-name ".clj"))))
              ;; Buffer fields (typed)
                    (doseq [[fname fcd] (map vector field-names field-cds)]
                      (.withField cb fname fcd (int 0x0002))) ;; PRIVATE
              ;; initFn field: holds the Clojure fn that allocates buffers
                    (.withField cb "initFn" obj-cd (int 0x0002)) ;; PRIVATE
              ;; initialized flag
                    (.withField cb "initialized" Z-cd (int 0x0002)) ;; PRIVATE

              ;; Constructor: takes initFn only
                    (let [ctor-mt (MethodTypeDesc/of void-cd (into-array ClassDesc [obj-cd]))]
                      (.withMethodBody cb "<init>" ctor-mt (int 0x0001)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                      ;; super()
                                           (.aload code 0)
                                           (.invokespecial code afn-cd "<init>" (MethodTypeDesc/of void-cd empty-cds))
                      ;; this.initFn = arg0
                                           (.aload code 0)
                                           (.aload code 1)
                                           (.putfield code wrapper-cd "initFn" obj-cd)
                      ;; this.initialized = false
                                           (.aload code 0)
                                           (.ldc code (int 0))
                                           (.putfield code wrapper-cd "initialized" Z-cd)
                                           (.return_ code)))))

              ;; doInit: call initFn(args...), extract results, store in fields
              ;; initFn returns a seq/vector of buffer arrays
                    (.withMethodBody cb "doInit" init-mt (int 0x0002) ;; PRIVATE
                                     (reify java.util.function.Consumer
                                       (accept [_ code]
                    ;; Call initFn with all args
                                         (.aload code 0)
                                         (.getfield code wrapper-cd "initFn" obj-cd)
                                         (.checkcast code ifn-cd)
                                         (if (<= n-fn-params 20)
                                           (do ;; Direct invoke for ≤20 params
                                             (dotimes [i n-fn-params] (.aload code (inc i)))
                                             (let [ifn-invoke-mt (MethodTypeDesc/of obj-cd invoke-params)]
                                               (.invokeinterface code ifn-cd "invoke" ifn-invoke-mt)))
                                           (do ;; >20 params: pack into Object[] → ArraySeq → applyTo
                                             (let [rt-cd (ClassDesc/of "clojure.lang" "RT")
                                                   aseq-cd (ClassDesc/of "clojure.lang" "ArraySeq")
                                                   iseq-cd (ClassDesc/of "clojure.lang" "ISeq")
                                                   oa-cd (.arrayType obj-cd)]
                          ;; new Object[n]
                                               (.ldc code (int n-fn-params))
                                               (.anewarray code obj-cd)
                          ;; arr[i] = arg_i
                                               (dotimes [i n-fn-params]
                                                 (.dup code)
                                                 (.ldc code (int i))
                                                 (.aload code (inc i))
                                                 (.aastore code))
                          ;; ArraySeq.create(Object[]) → ArraySeq
                                               (.invokestatic code aseq-cd "create"
                                                              (MethodTypeDesc/of aseq-cd (into-array ClassDesc [oa-cd])))
                          ;; initFn.applyTo(ISeq) → Object
                                               (.invokeinterface code ifn-cd "applyTo"
                                                                 (MethodTypeDesc/of obj-cd (into-array ClassDesc [iseq-cd]))))))
                    ;; Result is on stack — it's a vector/seq of buffer arrays
                    ;; Extract each buffer: (nth result i) → store to field
                                         (let [rt-cd (ClassDesc/of "clojure.lang" "RT")
                                               nth-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc [obj-cd I-cd]))]
                                           (dotimes [i n-fields]
                        ;; dup result for nth call (except last)
                                             (when (< i (dec n-fields)) (.dup code))
                                             (.ldc code (int i))
                                             (.invokestatic code rt-cd "nth" nth-mt)
                        ;; Store to field: this.fi = (cast)result
                                             (let [fcd (nth field-cds i)
                                                   fname (nth field-names i)]
                                               (when (not (.equals ^ClassDesc fcd obj-cd))
                                                 (.checkcast code fcd))
                          ;; Need to store: aload 0, swap, putfield
                          ;; But stack has [result] and we need [this result]
                          ;; Use a local to hold the value
                                               (let [tmp (+ n-fn-params 1 i)]
                                                 (.astore code tmp)
                                                 (.aload code 0)
                                                 (.aload code tmp)
                                                 (.putfield code wrapper-cd fname fcd))))
                      ;; Pop remaining dup if last nth consumed it
                                           )
                    ;; Set initialized = true
                                         (.aload code 0)
                                         (.ldc code (int 1))
                                         (.putfield code wrapper-cd "initialized" Z-cd)
                    ;; Clear initFn to allow GC
                                         (.aload code 0)
                                         (.aconst_null code)
                                         (.putfield code wrapper-cd "initFn" obj-cd)
                                         (.return_ code))))

              ;; Typed invk: check initialized, load params + fields, invokestatic compute
                    (let [invk-mt (MethodTypeDesc/of ret-cd (into-array ClassDesc fn-param-cds))]
                      (.withMethodBody cb "invk" invk-mt (int 0x0001)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                      ;; if (!this.initialized) this.doInit(boxed-a0, ..., boxed-aN)
                                           (.aload code 0)
                                           (.getfield code wrapper-cd "initialized" Z-cd)
                                           (let [skip-init (.newLabel code)]
                                             (.ifne code skip-init)
                                             (.aload code 0)
                                             (let [slot (atom 1)]
                                               (doseq [fi fn-param-infos]
                                                 (emit-load code (:stack-type fi) @slot)
                                                 (emit-box code (:stack-type fi) empty-cds)
                                                 (swap! slot + (if (#{:double :long} (:stack-type fi)) 2 1))))
                                             (.invokevirtual code wrapper-cd "doInit" init-mt)
                                             (.labelBinding code skip-init))
                      ;; Load typed params
                                           (let [slot (atom 1)]
                                             (doseq [fi fn-param-infos]
                                               (emit-load code (:stack-type fi) @slot)
                                               (swap! slot + (if (#{:double :long} (:stack-type fi)) 2 1))))
                      ;; Load hoisted fields
                                           (doseq [[fname fcd] (map vector field-names field-cds)]
                                             (.aload code 0)
                                             (.getfield code wrapper-cd fname fcd))
                                           (.invokestatic code statics-cd compute-mn compute-mt)
                                           (emit-return code (:stack-type ret-info))))))

              ;; IFn.invoke(Object...) → Object
              ;; Check initialized, call doInit if needed, then unbox→invk→box
                    (when (<= n-fn-params 20)
                      (.withMethodBody cb "invoke" invoke-mt (int 0x0001)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                      ;; if (!this.initialized) this.doInit(a0, a1, ..., aN)
                                           (.aload code 0)
                                           (.getfield code wrapper-cd "initialized" Z-cd)
                                           (let [skip-init (.newLabel code)]
                                             (.ifne code skip-init)
                        ;; doInit call
                                             (.aload code 0)
                                             (dotimes [i n-fn-params] (.aload code (inc i)))
                                             (.invokevirtual code wrapper-cd "doInit" init-mt)
                                             (.labelBinding code skip-init))
                      ;; this.invk(unbox(a0), unbox(a1), ..., unbox(aN))
                                           (.aload code 0)
                                           (doseq [[i fi fcd] (map vector (range) fn-param-infos fn-param-cds)]
                                             (.aload code (inc i))
                                             (emit-unbox code (:stack-type fi) fcd empty-cds))
                                           (let [invk-mt (MethodTypeDesc/of ret-cd (into-array ClassDesc fn-param-cds))]
                                             (.invokevirtual code wrapper-cd "invk" invk-mt))
                                           (emit-box code (:stack-type ret-info) empty-cds)
                                           (.areturn code)))))

              ;; applyTo(ISeq) — required for >20 params, but always useful
              ;; Extracts args from the seq, unboxes, and calls invk
                    (let [iseq-cd (ClassDesc/of "clojure.lang" "ISeq")
                          apply-mt (MethodTypeDesc/of obj-cd (into-array ClassDesc [iseq-cd]))]
                      (.withMethodBody cb "applyTo" apply-mt (int 0x0001)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                      ;; Extract args from ISeq into local vars (slot 2+)
                      ;; slot 0 = this, slot 1 = iseq
                                           (let [base-slot 2]
                                             (dotimes [i n-fn-params]
                          ;; localN = seq.first()
                                               (.aload code 1)
                                               (.invokeinterface code iseq-cd "first"
                                                                 (MethodTypeDesc/of obj-cd empty-cds))
                                               (.astore code (+ base-slot i))
                          ;; seq = seq.next()
                                               (when (< i (dec n-fn-params))
                                                 (.aload code 1)
                                                 (.invokeinterface code iseq-cd "next"
                                                                   (MethodTypeDesc/of iseq-cd empty-cds))
                                                 (.astore code 1)))
                        ;; doInit if needed
                                             (.aload code 0)
                                             (.getfield code wrapper-cd "initialized" Z-cd)
                                             (let [skip-init (.newLabel code)]
                                               (.ifne code skip-init)
                                               (.aload code 0)
                                               (dotimes [i n-fn-params] (.aload code (+ base-slot i)))
                                               (.invokevirtual code wrapper-cd "doInit" init-mt)
                                               (.labelBinding code skip-init))
                        ;; this.invk(unbox(local2), unbox(local3), ...)
                                             (.aload code 0)
                                             (doseq [[i fi fcd] (map vector (range) fn-param-infos fn-param-cds)]
                                               (.aload code (+ base-slot i))
                                               (emit-unbox code (:stack-type fi) fcd empty-cds))
                                             (let [invk-mt (MethodTypeDesc/of ret-cd (into-array ClassDesc fn-param-cds))]
                                               (.invokevirtual code wrapper-cd "invk" invk-mt))
                                             (emit-box code (:stack-type ret-info) empty-cds)
                                             (.areturn code)))))))))
        ;; Verify bytecode before loading — catches type errors early
        _ (try (.verify (ClassFile/of) wrapper-bytes)
               (catch Exception e
                 (throw (ex-info (str "Bytecode verification failed for wrapper " wrapper-name
                                      ": " (.getMessage e))
                                 {:wrapper-name wrapper-name} e))))
        loader (or me/*compilation-classloader*
                   (clojure.lang.DynamicClassLoader.
                    (.getContextClassLoader (Thread/currentThread))))
        wrapper-cls (.defineClass loader wrapper-name wrapper-bytes nil)]
    {:statics-class statics-class
     :wrapper-class wrapper-cls
     :methods (:methods statics-result)
     :statics-bytes (:bytes statics-result)
     :wrapper-bytes wrapper-bytes}))
