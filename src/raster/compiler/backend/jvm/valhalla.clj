(ns raster.compiler.backend.jvm.valhalla
  "Valhalla value class bytecode generation (JDK 27+).

  Generates JVM value classes with:
  - Zero-allocation stack semantics
  - Array flattening
  - ILookup for keyword field access
  - toString, makeArray, arrayGet, arraySet, arrayLength statics")

;; ================================================================
;; JDK detection
;; ================================================================

(def valhalla-available?
  "True if JDK supports Valhalla value classes (JDK 27+ with --enable-preview).
  The ClassFile API alone (JDK 24+) is not sufficient — value classes require
  the Valhalla extensions (no ACC_IDENTITY flag) which are only in JDK 27+."
  (try
    (Class/forName "java.lang.classfile.ClassFile")
    ;; ClassFile API exists; now check if value classes are actually supported
    ;; by verifying JDK version >= 27 (Valhalla preview)
    (let [version (.feature (Runtime/version))]
      (>= version 27))
    (catch ClassNotFoundException _ false)))

;; ================================================================
;; Bytecode generation (only defined when Valhalla is available)
;; ================================================================

(when valhalla-available?
  (eval
   '(do
      (import '(java.lang.classfile ClassFile)
              '(java.lang.constant ClassDesc MethodTypeDesc))

      (defn- type-tag->descriptor
        "Map type tags to JVM descriptors. Returns nil for unrecognized tags."
        [tag]
        (case tag
          double "D"
          float  "F"
          long   "J"
          int    "I"
          short  "S"
          byte   "B"
          char   "C"
          boolean "Z"
          doubles "[D"
          floats  "[F"
          longs   "[J"
          ints    "[I"
          nil))

      (defn- tag->box-class-name
        "Map primitive type tag to its boxed class name. Returns nil for non-primitives."
        [tag]
        (case tag
          double  "java.lang.Double"
          float   "java.lang.Float"
          long    "java.lang.Long"
          int     "java.lang.Integer"
          short   "java.lang.Short"
          byte    "java.lang.Byte"
          char    "java.lang.Character"
          boolean "java.lang.Boolean"
          nil))

      (defn- type-tag->slot-size
        "JVM slot size for a type tag. double/long take 2 slots, everything else 1."
        [tag]
        (case tag
          (double long) 2
          1))

      (defn generate-value-class-bytes
        "Generate Valhalla value class bytecode using JDK ClassFile API.
         class-name: full class name string (e.g. \"raster.core.Complex\")
         fields: [{:name \"v\" :tag double} {:name \"d0\" :tag double} ...]
         interfaces: [ClassDesc ...] optional interfaces to implement
         Automatically implements clojure.lang.ILookup for keyword field access.
         Field names are munged to JVM convention (- → _) for Clojure interop
         compatibility, while ILookup retains the original keyword names."
        [^String class-name fields & {:keys [interfaces] :or {interfaces []}}]
        (let [;; Munge field names for JVM (- → _), matching defrecord behavior.
              ;; Each field gets :jvm-name (munged, for field/getfield) and
              ;; :name (original, for keyword lookup via ILookup).
              fields (mapv (fn [f] (assoc f :jvm-name (.replace ^String (:name f) "-" "_"))) fields)
              cf (ClassFile/of)
              dot-idx (.lastIndexOf class-name (int \.))
              pkg (subs class-name 0 dot-idx)
              simple (subs class-name (inc dot-idx))
              class-desc (ClassDesc/of pkg simple)
              obj-desc (ClassDesc/of "java.lang.Object")
              void-desc (ClassDesc/ofDescriptor "V")
              int-desc (ClassDesc/ofDescriptor "I")
              arr-desc (.arrayType class-desc)
              sb-desc (ClassDesc/of "java.lang" "StringBuilder")
              str-desc (ClassDesc/of "java.lang" "String")
              kw-desc (ClassDesc/of "clojure.lang" "Keyword")
              ilookup-desc (ClassDesc/of "clojure.lang" "ILookup")
              tag->desc (fn [tag]
                          (cond
                            ;; value-class / array-of-value-class field: tag is a
                            ;; ClassDesc, or a JVM descriptor STRING ("Lpkg/Name;",
                            ;; "[Lpkg/Name;") — embeddable from the defvalue macro.
                            ;; A typed reference field (nested Parameters); the rest
                            ;; of the generator handles reference fields already
                            ;; (default aload, 1 slot, no box).
                            (instance? java.lang.constant.ClassDesc tag) tag
                            (string? tag) (ClassDesc/ofDescriptor tag)
                            (type-tag->descriptor tag) (ClassDesc/ofDescriptor (type-tag->descriptor tag))
                            :else obj-desc))
               ;; Box a primitive on the stack: emits invokestatic Type.valueOf(prim)
              emit-box (fn [code tag]
                         (when-let [box-cls (tag->box-class-name tag)]
                           (let [box-cd (ClassDesc/ofDescriptor (str "L" (.replace ^String box-cls "." "/") ";"))
                                 prim-cd (ClassDesc/ofDescriptor (type-tag->descriptor tag))]
                             (.invokestatic code box-cd "valueOf"
                                            (MethodTypeDesc/of box-cd (into-array ClassDesc [prim-cd]))))))]
          (.build cf class-desc
                  (reify java.util.function.Consumer
                    (accept [_ cb]
                 ;; Class flags: PUBLIC + FINAL (no ACC_IDENTITY = value class)
                      (.withFlags cb (int 0x0011))
                 ;; Version 71.65535 (JDK 27 preview)
                      (.withVersion cb 71 65535)
                 ;; Interfaces: user-specified + ILookup
                      (let [all-ifaces (into (vec interfaces) [ilookup-desc])]
                        (.withInterfaceSymbols cb (into-array ClassDesc all-ifaces)))
                 ;; Static keyword fields for ILookup (kw__fieldname)
                      (doseq [{:keys [name]} fields]
                        (.withField cb (str "kw__" name) kw-desc
                                    (reify java.util.function.Consumer
                                      (accept [_ fb]
                                        (.withFlags fb (int 0x001A)))))) ;; PRIVATE + STATIC + FINAL
                 ;; Static initializer: init keyword fields
                      (.withMethodBody cb "<clinit>"
                                       (MethodTypeDesc/of void-desc (into-array ClassDesc []))
                                       (int 0x0008) ;; STATIC
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (doseq [{:keys [name]} fields]
                                             (.ldc code name)
                                             (.invokestatic code kw-desc "intern"
                                                            (MethodTypeDesc/of kw-desc (into-array ClassDesc [str-desc])))
                                             (.putstatic code class-desc (str "kw__" name) kw-desc))
                                           (.return_ code))))
                 ;; Data fields with STRICT_INIT (use JVM-munged names)
                      (doseq [{:keys [jvm-name tag]} fields]
                        (.withField cb jvm-name (tag->desc tag)
                                    (reify java.util.function.Consumer
                                      (accept [_ fb]
                                        (.withFlags fb (int 0x0811)))))) ;; PUBLIC + FINAL + STRICT_INIT
                 ;; Constructor: putfield before invokespecial super()
                      (let [param-descs (into-array ClassDesc (map #(tag->desc (:tag %)) fields))
                            ctor-desc (MethodTypeDesc/of void-desc param-descs)]
                        (.withMethodBody cb "<init>" ctor-desc (int 0x0001)
                                         (reify java.util.function.Consumer
                                           (accept [_ code]
                                             (loop [slot 1 flds (seq fields)]
                                               (when flds
                                                 (let [{:keys [jvm-name tag]} (first flds)
                                                       fd (tag->desc tag)]
                                                   (.aload code 0)
                                                   (case tag
                                                     double (.dload code slot)
                                                     long (.lload code slot)
                                                     float (.fload code slot)
                                                     (int boolean byte short char) (.iload code slot)
                                                     (.aload code slot))
                                                   (.putfield code class-desc jvm-name fd)
                                                   (recur (+ slot (type-tag->slot-size tag)) (next flds)))))
                                             (.aload code 0)
                                             (.invokespecial code obj-desc "<init>"
                                                             (MethodTypeDesc/of void-desc (into-array ClassDesc [])))
                                             (.return_ code)))))
                 ;; ILookup.valAt(Object key) — keyword field access
                 ;; Keyword comparison uses :name (original), field access uses :jvm-name (munged)
                      (.withMethodBody cb "valAt"
                                       (MethodTypeDesc/of obj-desc (into-array ClassDesc [obj-desc]))
                                       (int 0x0001) ;; PUBLIC
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (doseq [{:keys [name jvm-name tag]} fields]
                                             (let [next-label (.newLabel code)]
                                               (.aload code 1) ;; key
                                               (.getstatic code class-desc (str "kw__" name) kw-desc)
                                               (.if_acmpne code next-label)
                           ;; Match: return boxed field value
                                               (.aload code 0)
                                               (.getfield code class-desc jvm-name (tag->desc tag))
                                               (if (type-tag->descriptor tag)
                                                 (emit-box code tag)
                                                 :nop) ;; already Object
                                               (.areturn code)
                                               (.labelBinding code next-label)))
                       ;; No match
                                           (.aconst_null code)
                                           (.areturn code))))
                 ;; ILookup.valAt(Object key, Object notFound)
                      (.withMethodBody cb "valAt"
                                       (MethodTypeDesc/of obj-desc (into-array ClassDesc [obj-desc obj-desc]))
                                       (int 0x0001) ;; PUBLIC
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (doseq [{:keys [name jvm-name tag]} fields]
                                             (let [next-label (.newLabel code)]
                                               (.aload code 1) ;; key
                                               (.getstatic code class-desc (str "kw__" name) kw-desc)
                                               (.if_acmpne code next-label)
                                               (.aload code 0)
                                               (.getfield code class-desc jvm-name (tag->desc tag))
                                               (if (type-tag->descriptor tag)
                                                 (emit-box code tag)
                                                 :nop)
                                               (.areturn code)
                                               (.labelBinding code next-label)))
                       ;; No match: return notFound
                                           (.aload code 2)
                                           (.areturn code))))
                 ;; toString method
                      (.withMethodBody cb "toString"
                                       (MethodTypeDesc/of str-desc (into-array ClassDesc []))
                                       (int 0x0001)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (.new_ code sb-desc)
                                           (.dup code)
                                           (.invokespecial code sb-desc "<init>"
                                                           (MethodTypeDesc/of void-desc (into-array ClassDesc [])))
                                           (.ldc code (str simple "("))
                                           (.invokevirtual code sb-desc "append"
                                                           (MethodTypeDesc/of sb-desc (into-array ClassDesc [str-desc])))
                                           (let [arrays-cd (ClassDesc/of "java.util" "Arrays")]
                                             (doseq [[idx {:keys [name jvm-name tag]}] (map-indexed vector fields)]
                                               (when (> idx 0)
                                                 (.ldc code ", ")
                                                 (.invokevirtual code sb-desc "append"
                                                                 (MethodTypeDesc/of sb-desc (into-array ClassDesc [str-desc]))))
                                               (.aload code 0)
                                               (.getfield code class-desc jvm-name (tag->desc tag))
                                               (let [fd (tag->desc tag)
                                                     desc-str (when (type-tag->descriptor tag)
                                                                (type-tag->descriptor tag))]
                                                 (if (and desc-str (.startsWith ^String desc-str "["))
                                                   ;; Array field: use Arrays.toString(type[]) → String, then sb.append(String)
                                                   (do (.invokestatic code arrays-cd "toString"
                                                                      (MethodTypeDesc/of str-desc (into-array ClassDesc [fd])))
                                                       (.invokevirtual code sb-desc "append"
                                                                       (MethodTypeDesc/of sb-desc (into-array ClassDesc [str-desc]))))
                                                   ;; Primitive or Object field: sb.append(type) directly
                                                   (.invokevirtual code sb-desc "append"
                                                                   (MethodTypeDesc/of sb-desc (into-array ClassDesc [fd])))))))
                                           (.ldc code ")")
                                           (.invokevirtual code sb-desc "append"
                                                           (MethodTypeDesc/of sb-desc (into-array ClassDesc [str-desc])))
                                           (.invokevirtual code sb-desc "toString"
                                                           (MethodTypeDesc/of str-desc (into-array ClassDesc [])))
                                           (.areturn code))))
                 ;; Static: makeArray(int n) -> ClassName[]
                      (.withMethodBody cb "makeArray"
                                       (MethodTypeDesc/of arr-desc (into-array ClassDesc [int-desc]))
                                       (int 0x0009) ;; PUBLIC + STATIC
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (.iload code 0)
                                           (.anewarray code class-desc)
                                           (.areturn code))))
                 ;; Static: arrayGet(ClassName[] arr, int i) -> ClassName
                      (.withMethodBody cb "arrayGet"
                                       (MethodTypeDesc/of class-desc (into-array ClassDesc [arr-desc int-desc]))
                                       (int 0x0009)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (.aload code 0)
                                           (.iload code 1)
                                           (.aaload code)
                                           (.checkcast code class-desc)
                                           (.areturn code))))
                 ;; Static: arraySet(ClassName[] arr, int i, ClassName val) -> void
                      (.withMethodBody cb "arraySet"
                                       (MethodTypeDesc/of void-desc (into-array ClassDesc [arr-desc int-desc class-desc]))
                                       (int 0x0009)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (.aload code 0)
                                           (.iload code 1)
                                           (.aload code 2)
                                           (.aastore code)
                                           (.return_ code))))
                 ;; Static: arrayLength(ClassName[] arr) -> int
                      (.withMethodBody cb "arrayLength"
                                       (MethodTypeDesc/of int-desc (into-array ClassDesc [arr-desc]))
                                       (int 0x0009)
                                       (reify java.util.function.Consumer
                                         (accept [_ code]
                                           (.aload code 0)
                                           (.arraylength code)
                                           (.ireturn code)))))))))

      (defn load-value-class!
        "Generate and load a Valhalla value class via DynamicClassLoader.
         Uses *compilation-classloader* if available for classloader consistency
         (parametric specialization). Without it, uses the current eval's DCL
         which enables REPL redefinition (new DCL shadows old class).
         On duplicate class in the same DCL, returns the existing class."
        [class-name field-specs & {:keys [interfaces] :or {interfaces []}}]
        (let [loader (or raster.compiler.core.method-entry/*compilation-classloader*
                         (let [tcl (.getContextClassLoader (Thread/currentThread))]
                           (if (instance? clojure.lang.DynamicClassLoader tcl)
                             tcl
                             (clojure.lang.DynamicClassLoader. tcl))))
              bytes (generate-value-class-bytes class-name field-specs :interfaces interfaces)]
          (try
            (.defineClass ^clojure.lang.DynamicClassLoader loader class-name bytes nil)
            (catch LinkageError _
              ;; Class already defined in this loader (parametric cascading) — return existing.
              ;; This is expected during parametric specialization where multiple templates
              ;; reference the same defvalue type.
              (when (System/getProperty "raster.debug")
                (binding [*out* *err*]
                  (println (str "DEBUG: value class already defined: " class-name))))
              (Class/forName class-name true loader))))))))

(when-not valhalla-available?
  (defn load-value-class!
    "Stub for non-Valhalla JDKs. Value class generation requires JDK 27+."
    [class-name field-specs & {:keys [interfaces] :or {interfaces []}}]
    (throw (UnsupportedOperationException.
            (str "load-value-class! requires Valhalla JDK 27+ (current: JDK "
                 (.feature (Runtime/version)) ")")))))
