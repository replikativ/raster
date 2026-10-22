(ns raster.compiler.core.call-boundary-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.call-boundary :as boundary]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types])
  (:import [java.lang.reflect Proxy InvocationHandler Method]
           [clojure.asm ClassWriter Opcodes]))

(defn- invocation-method [^Class interface]
  ;; These fixtures create interfaces with one declared invocation. This is not
  ;; a production overload selector; production requires the resolved descriptor.
  (first (filter #(= "invk" (.getName ^Method %)) (.getDeclaredMethods interface))))

(defn- receiver [^Class interface calls]
  (Proxy/newProxyInstance
   (.getClassLoader interface) (into-array Class [interface])
   (reify InvocationHandler
     (invoke [_ _ _ _]
       (swap! calls inc)
       (throw (ex-info "Signature analysis executed the callee" {}))))))

(defn- with-selected-definition
  ([tags body] (with-selected-definition tags (types/ensure-fn-interface! tags) body))
  ([tags interface body]
  (let [scope (create-ns (gensym "raster.test.call_boundary_"))
        scope-name (ns-name scope)
        operation (symbol (str scope-name) "fixture")
        source-name (types/mangle 'fixture tags)
        implementation (symbol (str scope-name) (str source-name "-impl"))
        calls (atom 0)
        executable (receiver (:iface-class interface) calls)
        source-fn (fn [& _] (swap! calls inc))
        params (mapv #(symbol (str "arg" %)) (range (count tags)))
        source (intern scope source-name source-fn)
        receiver-var (intern scope (symbol (name implementation)) executable)
        entry {:tags tags :mangled-ns scope-name :typed-impl executable
               :target-fn source-fn :typed-iface (:iface-name interface)}
        table (atom {(count tags) [entry]})
        generic (intern scope 'fixture (fn [& _] (swap! calls inc)))]
    (alter-meta! source assoc :raster.core/deftm-params params
                 :raster.core/deftm-tags tags :raster.core/return-tag 'double
                 :raster.core/deftm-source-body '((throw (Exception. "not run"))))
    (alter-meta! generic assoc :raster.core/dispatch-table table)
    (try
      (body {:operation operation :implementation implementation
             :arity (count tags) :interface interface
             :method (invocation-method (:iface-class interface))
             :source source :receiver-var receiver-var :table table :calls calls})
      (finally (remove-ns scope-name))))))

(defn- signature [{:keys [operation implementation arity method]}]
  (boundary/selected-typed-signature operation implementation arity method))

(deftest signature-reflects-physical-carriers-not-logical-tags
  (with-selected-definition
    '[double custom-record doubles]
    (fn [{:keys [interface calls] :as fixture}]
      (let [result (signature fixture)]
        (is (= :known (:status result)))
        (is (identical? (:iface-class interface) (:declaring-class result)))
        (is (= [Double/TYPE Object (class (double-array 0))]
               (:parameter-classes result)))
        (is (= Object (:return-class result)))
        (is (= 'double (get-in result [:definition :return-tag])))
        (is (= '[double custom-record doubles] (get-in result [:definition :tags])))
        (is (not-any? #(contains? result %) [:presence :ownership :pure? :conversions]))
        (is (zero? @calls))))))

(deftest primitive-and-empty-signatures-are-read-from-the-method
  (doseq [[tags expected]
          [[[] []]
           ['[long int float boolean byte short char]
            [Long/TYPE Integer/TYPE Float/TYPE Boolean/TYPE Byte/TYPE Short/TYPE Character/TYPE]]]]
    (with-selected-definition
      tags
      (fn [{:keys [calls] :as fixture}]
        (let [result (signature fixture)]
          (is (= :known (:status result)))
          (is (= expected (:parameter-classes result)))
          (is (= Object (:return-class result)))
          (is (zero? @calls)))))))

(deftest selected-capture-never-restamps-a-definition-changed-by-preparation
  (doseq [nested? [false true]]
   (with-selected-definition
    '[double]
    (fn [{:keys [operation implementation arity method receiver-var calls]}]
      (let [preparations (atom 0)
            reason (binding [entry/*emission-inputs* (when nested? (atom {:static [] :values []}))]
                    (try
                     (boundary/capture-selected-emission!
                      operation implementation arity method
                      (fn [selected]
                        (is (identical? receiver-var selected))
                        (swap! preparations inc)
                        ;; Each preparation changes the originally observed
                        ;; definition. A fresh post-prepare snapshot cannot make
                        ;; that attempt valid, even when its root is unchanged.
                        (alter-meta! selected assoc ::preparation-revision (Object.))))
                     :accepted
                     (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
        (is (= :stale-definition-proof-input reason))
        (is (= (if nested? 1 3) @preparations)
            "A nested owner must propagate staleness, never retry around the parent's history")
        (is (zero? @calls)))))))

(deftest stale-source-or-executable-evidence-is-unknown
  (with-selected-definition
    '[double]
    (fn [{:keys [source receiver-var calls] :as fixture}]
      (doseq [v [source receiver-var]]
        (with-redefs-fn {v (fn [& _] (swap! calls inc))}
          #(is (= {:status :unknown :reason :unverified-selected-definition}
                  (signature fixture)))))
      (is (= :known (:status (signature fixture))))
      (is (zero? @calls)))))

(deftest matching-definition-roots-do-not-prove-receiver-interface
  (with-selected-definition
    '[double]
    (fn [{:keys [table receiver-var calls] :as fixture}]
      (let [other (Object.) saved @table]
        (try
          (swap! table assoc-in [1 0 :typed-impl] other)
          ;; Both roots agree, so this must reach the physical class check,
          ;; not accidentally pass through an earlier stale-root refusal.
          (with-redefs-fn {receiver-var other}
            #(is (= {:status :unknown :reason :invocation-receiver-class-mismatch}
                    (signature fixture))))
          (finally (reset! table saved))))
      (is (= :known (:status (signature fixture))))
      (is (zero? @calls)))))

(defn- duplicate-named-interfaces []
  ;; Unique test-only name: never replace a generated production interface in
  ;; Clojure's class cache or the existing interface registry.
  (let [name (str "raster.test.boundary." (gensym "Invocation"))
        writer (ClassWriter. 0)
        flags (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_INTERFACE Opcodes/ACC_ABSTRACT)]
    (.visit writer Opcodes/V1_8 flags (.replace name "." "/") nil "java/lang/Object" nil)
    (.visitMethod writer (bit-or Opcodes/ACC_PUBLIC Opcodes/ACC_ABSTRACT)
                  "invk" "(D)Ljava/lang/Object;" nil nil)
    (.visitEnd writer)
    (let [bytes (.toByteArray writer)
          parent (.getClassLoader types/TypedFn-class)
          first-loader (clojure.lang.DynamicClassLoader. parent)
          second-loader (clojure.lang.DynamicClassLoader. parent)]
      [(.defineClass first-loader name bytes nil)
       (.defineClass second-loader name bytes nil)])))

(deftest same-class-name-does-not-establish-descriptor-identity
  (let [[^Class original ^Class duplicate] (duplicate-named-interfaces)
        interface-name (symbol (.getName original))
        interface {:iface-name interface-name :iface-class original :param-tags '[double]}]
    (is (= (.getName original) (.getName duplicate)))
    (is (not (identical? original duplicate)))
    (with-redefs [types/fn-interface-by-name
                  (atom (assoc @types/fn-interface-by-name interface-name interface))]
      (with-selected-definition
        '[double] interface
        (fn [{:keys [calls] :as fixture}]
          (is (= :known (:status (signature fixture))))
          (is (= {:status :unknown :reason :invocation-descriptor-mismatch}
                 (signature (assoc fixture :method (invocation-method duplicate)))))
          (is (zero? @calls)))))))

(deftest descriptor-and-registration-must-belong-to-the-selected-call
  (with-selected-definition
    '[double]
    (fn [{:keys [interface calls] :as fixture}]
      (is (= {:status :unknown :reason :unresolved-invocation-method}
             (signature (assoc fixture :method nil))))
      (is (= {:status :unknown :reason :unverified-selected-definition}
             (signature (assoc fixture :arity 2))))
      (is (= {:status :unknown :reason :unverified-selected-definition}
             (signature (assoc fixture :implementation 'unregistered/receiver))))
      (let [other (types/ensure-fn-interface! '[long])]
        (is (= {:status :unknown :reason :invocation-descriptor-mismatch}
               (signature (assoc fixture :method (invocation-method (:iface-class other)))))))
      (with-redefs [types/fn-interface-by-name
                    (atom (dissoc @types/fn-interface-by-name (:iface-name interface)))]
        (is (= {:status :unknown :reason :unregistered-invocation-interface}
               (signature fixture))))
      (is (zero? @calls)))))
