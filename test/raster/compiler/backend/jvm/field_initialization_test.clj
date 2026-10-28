(ns raster.compiler.backend.jvm.field-initialization-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types])
  (:import [java.lang.classfile ClassFile]
           [java.lang.constant ClassDesc MethodTypeDesc]
           [java.lang.reflect InvocationTargetException Method]
           [java.net URL URLClassLoader]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(defn- outcome [run]
  (try (run) :returned
       (catch Throwable error
         (loop [error error]
           (if (instance? InvocationTargetException error)
             (recur (.getCause error)) (class error))))))

(defn- owner-bytes [class-name property fail?]
  ;; Counter is external to the owner: reading it cannot initialize the owner.
  ;; Every comparison arm gets a fresh Class, including throwing initializers.
  (let [string (ClassDesc/of "java.lang.String")
        void (ClassDesc/ofDescriptor "V")
        bytes (.build (ClassFile/of) (ClassDesc/of class-name)
                      (reify java.util.function.Consumer
                        (accept [_ builder]
                          (.withFlags builder (int 0x0001))
                          (.withSuperclass builder (ClassDesc/of "java.lang.Object"))
                          (.withField builder "coefficient" (ClassDesc/ofDescriptor "D") (int 0x0001))
                          (.withMethodBody builder "<init>" (MethodTypeDesc/of void (make-array ClassDesc 0))
                                           (int 0x0001)
                                           (reify java.util.function.Consumer
                                             (accept [_ code]
                                               (.aload code 0)
                                               (.invokespecial code (ClassDesc/of "java.lang.Object") "<init>"
                                                               (MethodTypeDesc/of void (make-array ClassDesc 0)))
                                               (.return_ code))))
                          (.withMethodBody builder "<clinit>" (MethodTypeDesc/of void (make-array ClassDesc 0))
                                           (int 0x0008)
                                           (reify java.util.function.Consumer
                                             (accept [_ code]
                                               (.ldc code property)
                                               (.ldc code "1")
                                               (.invokestatic code (ClassDesc/of "java.lang.System") "setProperty"
                                                              (MethodTypeDesc/of string (into-array ClassDesc [string string])))
                                               (.pop code)
                                               (if fail?
                                                 (let [failure (ClassDesc/of "java.lang.IllegalStateException")]
                                                   (.new_ code failure)
                                                   (.dup code)
                                                   (.ldc code "field owner initialized")
                                                   (.invokespecial code failure "<init>"
                                                                   (MethodTypeDesc/of void (into-array ClassDesc [string])))
                                                   (.athrow code))
                                                 (.return_ code))))))))]
    bytes))

(defn- define-owner [^clojure.lang.DynamicClassLoader loader class-name property fail?]
  (.defineClass loader class-name (owner-bytes class-name property fail?) nil))

(defn- compile-reader [arm class-name member]
  (let [receiver (with-meta 'value {:tag (symbol class-name)})
        parameters [(with-meta 'value {:tag 'Object}) (with-meta 'trace {:tag 'doubles})]
        body (list 'do '(clojure.core/aset-double trace 0 1.0)
                   (list 'let ['coefficient (list '. receiver member)]
                         '(clojure.core/aset-double trace 1 coefficient) 'coefficient))]
    (if (= arm :clojure)
      (eval (list 'fn parameters body))
      (let [artifact (bytecode/compile-specialized-class!
                      "raster.test.FieldInitialization"
                      [{:name 'run :params parameters :return-tag 'double
                        :walked-body [body] :source-ns *ns*}])
            ^Method method (get-in artifact [:methods 'run])]
        (fn [value trace] (.invoke method nil (object-array [value trace])))))))

(defn- check-reader! [arm class-name property loader ^Class owner fail? member]
  (is (nil? (System/getProperty property)))
  (let [read-field (compile-reader arm class-name member)]
    (is (nil? (System/getProperty property)) (str arm " compilation must not execute <clinit>"))
    (doseq [[receiver failure] [[nil NullPointerException] [(Object.) ClassCastException]]]
      (let [trace (double-array 2)]
        (is (= failure (outcome #(read-field receiver trace))))
        (is (= [1.0 0.0] (vec trace)) "earlier store runs; later store remains blocked")
        (is (nil? (System/getProperty property)))))
    (is (= (if fail? ExceptionInInitializerError :returned)
           (outcome #(Class/forName class-name true loader))))
    (is (= "1" (System/getProperty property)))
    (is (identical? owner (Class/forName class-name false loader)))
    (when-not fail?
      (let [instance (.newInstance (.getConstructor owner (make-array Class 0)) (object-array 0))
            trace (double-array 2)]
        (.setDouble (.getField owner "coefficient") instance 2.5)
        (is (= 2.5 (read-field instance trace)) "the emitted artifact must resolve the exact owner Class")
        (is (= [1.0 2.5] (vec trace)))))
    read-field))

(deftest field-selection-does-not-initialize-the-owner
  (doseq [fail? [false true] arm [:clojure :raster] member ['-coefficient 'coefficient]]
    (let [loader (.getClassLoader (class @(ns-resolve 'raster.compiler.backend.jvm.bytecode 'emit-dot-handler)))
          class-name (str "raster.fixture.FieldInitialization" (clojure.lang.RT/nextID))
          property (str "raster.test.field-initialization." (clojure.lang.RT/nextID))
          owner (define-owner loader class-name property fail?)]
      (try
        (check-reader! arm class-name property loader owner fail? member)
        (finally (System/clearProperty property))))))

(deftest namespace-class-fallback-does-not-reintroduce-initialization
  (doseq [fail? [false true] arm [:clojure :raster]]
    (let [root (Files/createTempDirectory "raster-field-init-" (make-array FileAttribute 0))
          package (.resolve root "raster/fixture")
          simple (str "FallbackField" (clojure.lang.RT/nextID))
          class-name (str "raster.fixture." simple)
          file (.resolve package (str simple ".class"))
          property (str "raster.test.field-fallback." (clojure.lang.RT/nextID))]
      (try
        (Files/createDirectories package (make-array FileAttribute 0))
        (Files/write file ^bytes (owner-bytes class-name property fail?) (make-array java.nio.file.OpenOption 0))
        (with-open [loader (URLClassLoader. (into-array URL [(.toURL (.toUri root))])
                                           (ClassLoader/getSystemClassLoader))]
          (let [owner (Class/forName class-name false loader)
                emitter-loader (.getClassLoader (class @(ns-resolve 'raster.compiler.backend.jvm.bytecode
                                                                    'emit-dot-handler)))]
            (is (= ClassNotFoundException (outcome #(Class/forName class-name false emitter-loader)))
                "the emitter's first loader really misses; this is a fallback test")
            (with-bindings {clojure.lang.Compiler/LOADER loader}
              (binding [entry/*compilation-classloader* (clojure.lang.DynamicClassLoader. loader)]
                (let [read-field (check-reader! arm class-name property loader owner fail? '-coefficient)]
                  (when-not fail?
                    (with-open [foreign-loader (URLClassLoader. (.getURLs loader) (ClassLoader/getSystemClassLoader))]
                      (let [foreign (Class/forName class-name false foreign-loader)
                            instance (.newInstance (.getConstructor foreign (make-array Class 0)) (object-array 0))
                            trace (double-array 2)]
                        (is (not (identical? owner foreign)))
                        (is (= ClassCastException (outcome #(read-field instance trace)))
                            "equal binary names from another loader are not the selected owner")
                        (is (= [1.0 0.0] (vec trace)))))))))))
        (finally
          (System/clearProperty property)
          (Files/deleteIfExists file)
          (Files/deleteIfExists package)
          (Files/deleteIfExists (.resolve root "raster"))
          (Files/deleteIfExists root))))))

(deftest noninitializing-type-fallback-retains-the-actual-resolver-loader
  (let [loader (.getClassLoader (class types/resolve-type-ref->class))
        name (str "raster.fixture.TypeResolution" (clojure.lang.RT/nextID))
        property (str "raster.test.field-type-resolution." (clojure.lang.RT/nextID))
        owner (define-owner loader name property true)]
    (try
      (is (identical? owner (types/resolve-type-ref->class (symbol name) false)))
      (is (identical? owner (types/tag->check-class (symbol name) false)))
      (is (nil? (System/getProperty property)))
      (is (= ExceptionInInitializerError (outcome #(Class/forName name true loader))))
      (is (= "1" (System/getProperty property)))
      (finally (System/clearProperty property)))))

(deftest default-type-resolution-retains-its-initializing-policy
  ;; Lookup-only consumers opt out explicitly. Dispatch/construction callers
  ;; using the existing arities must not silently acquire another policy.
  (doseq [resolver [types/resolve-type-ref->class types/tag->check-class]]
    (let [loader (.getClassLoader (class types/resolve-type-ref->class))
          name (str "raster.fixture.DefaultTypeResolution" (clojure.lang.RT/nextID))
          property (str "raster.test.default-type-resolution." (clojure.lang.RT/nextID))]
      (define-owner loader name property true)
      (try
        (is (nil? (System/getProperty property)))
        ;; RT may return its in-memory cache directly under a non-DCL base
        ;; loader. Fix the actual lookup policy instead of relying on whatever
        ;; ClassLoader the enclosing test runner happens to bind.
        (with-bindings {clojure.lang.Compiler/LOADER loader}
          (is (= ExceptionInInitializerError (outcome #(resolver (symbol name))))))
        (is (= "1" (System/getProperty property)))
        (finally (System/clearProperty property))))))
