(ns raster.ad.generated-call-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.ordered-call :as ordered]
            [raster.compiler.core.types :as types])
  (:import [clojure.asm ClassWriter Opcodes]))

(def policy {:unchecked-math false})
(def ^:dynamic *target* nil)

(defn- compile-owned [plan params]
  (#'reverse/make-runtime-value+grad-fn plan params {:generated-source-policy (:source-policy plan)}))

(deftest actual-owner-compiles-under-explicit-generated-policy
  (let [method (.getMethod Math "pow" (into-array Class [Double/TYPE Double/TYPE]))
        plan (ordered/plan-resolved-call method nil '[x (do (swap! events conj :later) 2.0)] policy)
        f (binding [*unchecked-math* true] (compile-owned plan '[x events]))]
    (dotimes [_ 2]
      (let [events (atom [])]
        (is (thrown? NullPointerException (f nil events)))
        (is (= [] @events)))
      (let [events (atom [])]
        (is (= 9.0 (f 3.0 events)))
        (is (= [:later] @events))))
    (is (= :unclosed-generated-call-policy
           (try (#'reverse/make-runtime-value+grad-fn plan '[x events]
                 {:generated-source-policy {:unchecked-math true}}) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest owner-preserves-default-and-large-parameter-paths
  (let [old (#'reverse/make-runtime-value+grad-fn '[(clojure.core/+ x 1.0)] '[x])]
    (is (= 3.0 (old 2.0))))
  (let [params (mapv #(symbol (str "p" %)) (range 24))
        old (#'reverse/make-runtime-value+grad-fn [(list 'clojure.core/+ (first params) (last params))] params)
        method (.getMethod Math "pow" (into-array Class [Double/TYPE Double/TYPE]))
        plan (ordered/plan-resolved-call method nil [(first params) (last params)] policy)
        generated (compile-owned plan params)
        values (vec (concat [3.0] (repeat 22 0.0) [2.0]))]
    (is (= 5.0 (apply old values)))
    (is (= 9.0 (apply generated values)))))

(deftest generated-policy-is-not-the-emission-threads-ambient-policy
  (let [method (.getMethod Integer "rotateLeft" (into-array Class [Integer/TYPE Integer/TYPE]))
        checked (ordered/plan-resolved-call method nil '[(long x) 0] {:unchecked-math false})
        unchecked (ordered/plan-resolved-call method nil '[(long x) 0] {:unchecked-math true})
        checked-fn (binding [*unchecked-math* true] (compile-owned checked '[x]))
        unchecked-fn (binding [*unchecked-math* false] (compile-owned unchecked '[x]))]
    (is (thrown? ArithmeticException (checked-fn Long/MAX_VALUE)))
    (is (= -1 (unchecked-fn Long/MAX_VALUE)))))

(deftest actual-owner-keeps-typed-input-failures-before-effects
  (doseq [[tags input expected-class] [[['double 'double] nil NullPointerException]
                                     [['doubles 'double] 42 ClassCastException]]]
    (let [iface (types/ensure-fn-interface! tags)
          ^Class carrier (:iface-class iface)
          method (first (filter #(= "invk" (.getName ^java.lang.reflect.Method %)) (.getDeclaredMethods carrier)))
          receiver (with-meta 'target {:tag (:iface-name iface)})
          target (java.lang.reflect.Proxy/newProxyInstance
                  (.getClassLoader carrier) (into-array Class [carrier])
                  (reify java.lang.reflect.InvocationHandler (invoke [_ _ _ _] 2.0)))
          arguments '[x (do (swap! events conj :later) 2.0)]
          source (eval (list 'fn* [receiver 'x 'events] (list* '.invk receiver arguments)))
          generated (compile-owned (ordered/plan-resolved-call method receiver arguments policy)
                                    [receiver 'x 'events])
          observe (fn [f]
                    (let [events (atom [])]
                      (try (f target input events) nil
                           (catch Throwable e {:exception (class e) :events @events}))))]
      (is (= {:exception expected-class :events []} (observe source)))
      (is (= (observe source) (observe generated))))))

(deftest actual-owner-keeps-runtime-target-selection-and-object-identity
  (let [method (.getMethod clojure.lang.IFn "invoke" (into-array Class [Object]))
        arguments '[(do (set! raster.ad.generated-call-owner-test/*target* (fn [x] [:after x])) 7)]
        plan (ordered/plan-resolved-call method 'raster.ad.generated-call-owner-test/*target* arguments policy)
        f (compile-owned plan [])]
    (binding [*target* (fn [x] [:before x])]
      (is (= [:before 7] (f)))))
  (let [method (.getMethod clojure.lang.IFn "invoke" (into-array Class [Object Object]))]
    (doseq [value [1000 5.125]]
      (let [receiver (with-meta 'target {:tag 'clojure.lang.IFn})
            source (eval (list 'fn* [receiver] (list receiver value value)))
            generated (compile-owned (ordered/plan-resolved-call method receiver [value value] policy) [receiver])]
        (is (true? (source identical?)))
        (is (= (source identical?) (generated identical?)))))))

(defn- repeated-interface-methods []
  (let [name (str "raster.test.GeneratedOwner" (gensym))
        writer (ClassWriter. 0)
        _ (.visit writer Opcodes/V1_8 (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_INTERFACE Opcodes/ACC_ABSTRACT)
                  (.replace name "." "/") nil "java/lang/Object" nil)
        m (.visitMethod writer (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_ABSTRACT)
                        "invk" "(D)Ljava/lang/Object;" nil nil)
        _ (.visitEnd m)
        _ (.visitEnd writer)
        bytes (.toByteArray writer)]
    (mapv (fn [_]
            (let [loader (clojure.lang.DynamicClassLoader. (.getContextClassLoader (Thread/currentThread)))
                  carrier (.defineClass loader name bytes nil)]
              (.getMethod carrier "invk" (into-array Class [Double/TYPE]))))
          (range 2))))

(deftest actual-invocation-loader-must-match-not-merely-reference-capture
  (let [[a b] (repeated-interface-methods)
        a-plan (ordered/plan-resolved-call a 'target '[x] policy)
        b-plan (ordered/plan-resolved-call b 'target '[x] policy)
        ^Class carrier (.getDeclaringClass ^java.lang.reflect.Method b)
        target (java.lang.reflect.Proxy/newProxyInstance
                (.getClassLoader carrier) (into-array Class [carrier])
                (reify java.lang.reflect.InvocationHandler
                  (invoke [_ _ _ args] (aget ^objects args 0))))]
    (is (= :source-invocation-class-mismatch
           (try (compile-owned a-plan '[target x]) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (let [f (compile-owned b-plan '[target x])]
      (is (= 2.0 (f target 2.0))))))

(deftest factory-supports-more-than-twenty-private-class-captures
  (let [name (str "raster.test.ManyReferenceInputs" (gensym))
        carriers (mapv #(Class/forName (str (apply str (repeat % "[")) "D")) (range 1 23))
        descriptor (str "(" (apply str (map #(.descriptorString ^Class %) carriers)) ")Ljava/lang/Object;")
        writer (ClassWriter. 0)
        _ (.visit writer Opcodes/V1_8 Opcodes/ACC_PUBLIC (.replace name "." "/") nil "java/lang/Object" nil)
        m (.visitMethod writer (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_STATIC) "call" descriptor nil nil)
        _ (.visitCode m)
        _ (.visitInsn m Opcodes/ACONST_NULL)
        _ (.visitInsn m Opcodes/ARETURN)
        _ (.visitMaxs m 1 22)
        _ (.visitEnd m)
        _ (.visitEnd writer)
        loader (clojure.lang.DynamicClassLoader. (.getContextClassLoader (Thread/currentThread)))
        carrier (.defineClass loader name (.toByteArray writer) nil)
        method (.getMethod carrier "call" (into-array Class carriers))
        params (mapv #(symbol (str "x" %)) (range 22))
        plan (ordered/plan-resolved-call method nil params policy)
        f (compile-owned plan params)]
    (is (nil? (apply f (repeat 22 nil))))))

(deftest parameter-signature-alone-cannot-select-a-return-descriptor
  (let [name (str "raster.test.ReturnDescriptor" (gensym))
        writer (ClassWriter. 0)
        _ (.visit writer Opcodes/V1_8 (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_INTERFACE Opcodes/ACC_ABSTRACT)
                  (.replace name "." "/") nil "java/lang/Object" nil)
        _ (doseq [return ["Ljava/lang/Object;" "Ljava/lang/String;"]]
            (.visitEnd (.visitMethod writer (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_ABSTRACT)
                                     "invk" (str "(D)" return) nil nil)))
        _ (.visitEnd writer)
        loader (clojure.lang.DynamicClassLoader. (.getContextClassLoader (Thread/currentThread)))
        carrier (.defineClass loader name (.toByteArray writer) nil)]
    (doseq [method (.getDeclaredMethods carrier)]
      (is (= :source-invocation-method-ambiguous
             (try (compile-owned (ordered/plan-resolved-call method 'target '[x] policy) '[target x]) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
