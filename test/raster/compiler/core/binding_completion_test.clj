(ns raster.compiler.core.binding-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.binding-completion :as completion]
            [raster.compiler.core.method-entry :as entry])
  (:import [clojure.asm ClassWriter Opcodes]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))

(defn- class-bytes [name parent initialization-property]
  (let [writer (ClassWriter. 0)
        _ (.visit writer Opcodes/V1_8 Opcodes/ACC_PUBLIC (.replace ^String name "." "/") nil parent nil)
        constructor (.visitMethod writer Opcodes/ACC_PUBLIC "<init>" "()V" nil nil)]
    (.visitCode constructor)
    (.visitVarInsn constructor Opcodes/ALOAD 0)
    (.visitMethodInsn constructor Opcodes/INVOKESPECIAL parent "<init>" "()V" false)
    (.visitInsn constructor Opcodes/RETURN)
    (.visitMaxs constructor 1 1)
    (.visitEnd constructor)
    (when initialization-property
      (let [initializer (.visitMethod writer Opcodes/ACC_STATIC "<clinit>" "()V" nil nil)]
        (.visitCode initializer)
        (.visitLdcInsn initializer initialization-property)
        (.visitLdcInsn initializer "initialized")
        (.visitMethodInsn initializer Opcodes/INVOKESTATIC "java/lang/System" "setProperty"
                          "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;" false)
        (.visitInsn initializer Opcodes/POP)
        (.visitInsn initializer Opcodes/RETURN)
        (.visitMaxs initializer 2 0)
        (.visitEnd initializer)))
    (.visitEnd writer)
    (.toByteArray writer)))

(defn- with-resolution [run]
  (let [source (create-ns (gensym "raster.binding-source."))
        policy-scope (create-ns (gensym "raster.binding-policy."))
        loader (clojure.lang.DynamicClassLoader. (.getContextClassLoader (Thread/currentThread)))
        policy-name (str "raster.test." (gensym "BindingPolicy"))
        policy-class (.defineClass loader policy-name (class-bytes policy-name "clojure/lang/AFn" nil) nil)
        root (.newInstance (.getDeclaredConstructor policy-class (make-array Class 0)) (object-array 0))
        policy (intern policy-scope 'binding-policy root)
        context (completion/capture-resolution-context policy)]
    (try (run source policy loader context)
         (finally (remove-ns (ns-name source)) (remove-ns (ns-name policy-scope))))))

(deftest declared-hints-and-local-facts-preserve-the-existing-emitter-policy
  (let [binder (with-meta 'snapshot {:tag 'Object :raster.type/tag 'double})
        context {:source-ns *ns* :hint-classloader (.getClassLoader (class reason))}]
    (is (= 'Object (completion/declared-binding-hint binder)))
    (doseq [[hint actual expected]
            [['Object :ref {:type :ref :hint 'Object :verified false :reference-class Object}]
             ['java.lang.Object :ref {:type :ref :hint 'java.lang.Object :verified true :reference-class Object}]
             ['String :ref {:type :ref :hint 'String :verified true :reference-class String}]
             ['double :double {:type :double :hint 'double :verified true}]]]
      (is (= expected (:local-facts (completion/binding-completion-plan hint actual context)))))
    (is (identical? Object (:resolved-hint-class (completion/binding-completion-plan 'Object :ref context))))
    (is (nil? (:checkcast-class (completion/binding-completion-plan 'Object :ref context))))
    (is (identical? String (:checkcast-class (completion/binding-completion-plan 'String :ref context))))
    (is (= (class (double-array 0))
           (:resolved-hint-class (completion/binding-completion-plan 'doubles :ref context))))))

(deftest optional-object-proof-needs-original-owned-context-and-an-explicit-hint
  (with-resolution
    (fn [source policy _ context]
      (.importClass source 'Object Object)
      (let [binder (with-meta 'snapshot {:tag 'Object :raster.type/tag 'double})
            owner (atom {:static [] :values []})]
        (is (false? (completion/object-binding-without-check? binder source context)))
        (binding [entry/*emission-inputs* owner]
          (is (false? (completion/object-binding-without-check? binder source)))
          (is (false? (completion/object-binding-without-check? binder source {:hint-classloader nil})))
          (doseq [hint [nil 'String 'doubles 'IFn__double_double 'not.loaded/Class 'not.loaded.Class]]
            (is (false? (completion/object-binding-without-check?
                         (with-meta 'snapshot (when hint {:tag hint})) source context))))
          (is (= {:static [] :values []} @owner) "declined queries adopt no policy observations")
          (is (completion/object-binding-without-check? binder source context))
          (is (contains? (:definition-proofs @owner) policy))
          (is (completion/object-binding-without-check?
               (with-meta 'snapshot {:tag 'java.lang.Object}) source context)))
        (with-redefs-fn {policy (fn [] nil)}
          #(is (= :stale-definition-proof-input
                  (reason (fn [] (binding [entry/*emission-inputs* (atom {:static [] :values []})]
                                   (completion/object-binding-without-check? binder source context)))))))))))

(deftest reference-proof-uses-actual-upper-bound-classes-not-logical-array-tags
  (with-resolution
    (fn [source _ _ context]
      (.importClass source 'Object Object)
      (.importClass source 'Cloneable Cloneable)
      (let [floats (class (float-array 0))
            owner (atom {:static [] :values []})]
        (binding [entry/*emission-inputs* owner]
          (doseq [hint '[floats Object Cloneable]]
            (is (completion/reference-binding-without-check?
                 (with-meta 'snapshot {:tag hint}) source context floats)))
          (doseq [[hint carrier] [['doubles floats] ['String floats] ['floats Object]
                                  ['floats Float/TYPE] ['floats :float]]]
            (is (false? (completion/reference-binding-without-check?
                         (with-meta 'snapshot {:tag hint}) source context carrier))))
          (is (false? (completion/object-binding-without-check?
                       (with-meta 'snapshot {:tag 'floats}) source context))))))))

(deftest reference-proof-preserves-real-class-identity-and-original-lookup-history
  (with-resolution
    (fn [source _ loader context]
      (let [name (str "raster.test." (gensym "ReferenceCarrier"))
            bytes (class-bytes name "java/lang/Object" nil)
            other-loader (clojure.lang.DynamicClassLoader. (.getParent ^ClassLoader loader))
            actual (.defineClass ^clojure.lang.DynamicClassLoader loader name bytes nil)
            same-name (.defineClass other-loader name bytes nil)
            hint (gensym "ReferenceAlias")
            binder (with-meta 'snapshot {:tag hint})
            owner (atom {:static [] :values []})]
        (.importClass source hint actual)
        (is (= (.getName actual) (.getName same-name)))
        (binding [entry/*emission-inputs* owner]
          (is (completion/reference-binding-without-check? binder source context actual))
          (is (false? (completion/reference-binding-without-check? binder source context same-name))))
        (.unmap source hint)
        (.importClass source hint same-name)
        (is (= :stale-binding-completion-resolution
               (reason #(entry/assert-definition-inputs-current! @owner))))))))

(deftest object-proof-retains-full-first-loader-and-namespace-mapping-decision
  (with-resolution
    (fn [source _ loader context]
      (let [hint (gensym "ObjectAlias")
            binder (with-meta 'snapshot {:tag hint})
            owner (atom {:static [] :values []})
            property (str "raster.binding-initializer." (gensym))]
        (.importClass source hint Object)
        (binding [entry/*emission-inputs* owner]
          (is (completion/object-binding-without-check? binder source context)))
        (.unmap source hint)
        (.importClass source hint String)
        (is (= :stale-binding-completion-resolution
               (reason #(entry/assert-definition-inputs-current! @owner))))
        (.unmap source hint)
        (.importClass source hint Object)
        (is (nil? (reason #(entry/assert-definition-inputs-current! @owner))))
        ;; Defining a previously missing first-loader Class changes precedence,
        ;; even when the namespace still maps the hint to java.lang.Object.
        (let [defined (.defineClass ^clojure.lang.DynamicClassLoader loader (str hint)
                                   (class-bytes (str hint) "java/lang/Object" property) nil)]
        (is (= :stale-binding-completion-resolution
               (reason #(entry/assert-definition-inputs-current! @owner))))
        (is (nil? (System/getProperty property)) "guard replay must not run static initializers")
        (binding [entry/*emission-inputs* (atom {:static [] :values []})]
          (is (false? (completion/object-binding-without-check? binder source context)))
          (is (completion/reference-binding-without-check? binder source context defined)))
        (is (nil? (System/getProperty property)) "optional construction must not initialize either"))))))

(deftest object-proof-retains-source-namespace-identity
  (with-resolution
    (fn [source _ _ context]
      (let [owner (atom {:static [] :values []}) source-name (ns-name source)]
        (.importClass source 'Object Object)
        (binding [entry/*emission-inputs* owner]
          (is (completion/object-binding-without-check? (with-meta 'snapshot {:tag 'Object}) source context)))
        (remove-ns source-name)
        (create-ns source-name)
        (is (= :stale-binding-completion-resolution
               (reason #(entry/assert-definition-inputs-current! @owner))))))))
