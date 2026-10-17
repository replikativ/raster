(ns raster.compiler.core.reference-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.value-completion :as completion])
  (:import [clojure.asm ClassWriter Opcodes]))

(defn- checked-source [carrier expression]
  (completion/project-to-jvm-source
   (completion/complete-value (completion/reference-check-contract carrier) expression)
   nil {carrier 'reference-class}))

(defn- compile-checked [carrier]
  (let [f (eval (list 'fn* '[reference-class value] (checked-source carrier 'value)))]
    (partial f carrier)))

(deftest exact-reference-check-preserves-nil-identity-and-rejection
  (doseq [[carrier value] [[String "value"] [(class (double-array 0)) (double-array [1.0])]
                           [clojure.lang.IFn vector] [Object (Object.)]]]
    (let [f (compile-checked carrier)]
      (is (nil? (f nil)))
      (is (identical? value (f value)))
      (when-not (= Object carrier)
        (is (thrown? ClassCastException (f (Object.))))))))

(deftest receiver-check-and-invocation-remain-separate-events
  (let [source '(f (do (swap! events conj :argument) 7))
        captured (with-meta 'captured {:tag 'clojure.lang.IFn})
        normalized (list 'let* [captured (checked-source clojure.lang.IFn 'f)]
                         (list captured '(do (swap! events conj :argument) 7)))
        compile-fn (fn [body]
                     (partial (eval (list 'fn* '[reference-class f events] body)) clojure.lang.IFn))
        observe (fn [f input]
                  (let [events (atom [])]
                    (try {:result (f input events) :events @events}
                         (catch Throwable error {:exception (class error) :events @events}))))
        original (compile-fn source)
        transformed (compile-fn normalized)]
    (is (= {:exception ClassCastException :events []} (observe original 42)))
    (is (= {:exception NullPointerException :events [:argument]} (observe original nil)))
    (is (= {:result [7] :events [:argument]} (observe original vector)))
    (doseq [input [42 nil vector]]
      (is (= (observe original input) (observe transformed input))))))

(deftest exact-check-is-not-a-functional-interface-adapter
  (let [f (compile-checked java.util.function.Function)]
    ;; This contract means an existing reference check, not permission to adapt
    ;; a Clojure function to a SAM interface during a host call.
    (is (thrown? ClassCastException (f identity)))
    (let [value (reify java.util.function.Function (apply [_ x] x))]
      (is (identical? value (f value))))))

(deftest reference-check-never-accepts-a-tag-as-class-evidence
  (doseq [carrier ['String :double nil Double/TYPE Void/TYPE]]
    (is (thrown? clojure.lang.ExceptionInfo (completion/reference-check-contract carrier))))
  (let [operand (with-meta 'x {:raster.type/tag 'double :tag 'double :line 9})
        term (completion/complete-value (completion/reference-check-contract String) operand)]
    (is (= {:line 9} (meta term)))
    (is (identical? operand (:operand (completion/canonical-parts term))))))

(defn- duplicate-interface-classes []
  (let [name (str "raster.test.ReferenceCompletion" (gensym))
        writer (ClassWriter. 0)
        _ (.visit writer Opcodes/V1_8 (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_INTERFACE Opcodes/ACC_ABSTRACT)
                  (.replace name "." "/") nil "java/lang/Object" nil)
        _ (.visitEnd writer)
        bytes (.toByteArray writer)]
    (mapv (fn [_]
            (let [loader (clojure.lang.DynamicClassLoader. (.getContextClassLoader (Thread/currentThread)))]
              (.defineClass loader name bytes nil)))
          (range 2))))

(deftest reference-class-identity-is-not-a-binary-name
  (let [[a b] (duplicate-interface-classes)
        proxy-for (fn [^Class carrier]
                    (java.lang.reflect.Proxy/newProxyInstance
                     (.getClassLoader carrier) (into-array Class [carrier])
                     (reify java.lang.reflect.InvocationHandler
                       (invoke [_ _ _ _] nil))))
        a-value (proxy-for a)
        b-value (proxy-for b)
        f (compile-checked a)]
    (is (= (.getName ^Class a) (.getName ^Class b)))
    (is (not (identical? a b)))
    (is (identical? a-value (f a-value)))
    (is (thrown? ClassCastException (f b-value)))))

(deftest reference-check-requires-actual-owner-bound-class-transport
  (let [term (completion/complete-value (completion/reference-check-contract String) 'x)]
    (doseq [bindings [{} {String String} {String 'java.lang.String/value}]]
      (is (= :unclosed-reference-completion-class
             (try (completion/project-to-jvm-source term nil bindings) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))
