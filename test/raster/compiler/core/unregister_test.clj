(ns raster.compiler.core.unregister-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]))

(defn- with-scope [f]
  (let [scope (create-ns (gensym "raster.unregister."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (f scope))
      (finally
        (doseq [[_ v] (ns-publics scope)
                :when (:raster.core/generic-function (meta v))]
          (core/unregister! v))
        (remove-ns (ns-name scope))))))

(deftest unregister-preserves-the-var-and-replaces-the-whole-generic
  (with-scope
    (fn [scope]
      (eval '(raster.core/deftm model [x :- Long] :- Long x))
      (eval '(raster.core/deftm model [x :- Double] :- Double x))
      (let [v (ns-resolve scope 'model)
            table (:raster.core/dispatch-table (meta v))
            captured-dispatcher @v
            captured-target (:target-fn (first (get @table 1)))
            operation (symbol (str (ns-name scope)) "model")
            signature (inference/registered-call-signature operation '[long])
            inputs (atom {:static [] :values []})
            cache (#'dispatch/get-specialization-cache operation)
            claim (Object.)
            revision (dispatch/compiler-definition-revision)]
        (is (= 3 (v 3)))
        (is (some? signature))
        (binding [entry/*emission-inputs* inputs]
          (entry/observe-definition-input! v scope 'model))
        (is (true? (entry/assert-definition-inputs-current! @inputs)))
        (reset! cache {String {:claim claim :pending? true}})
        (is (identical? v (core/unregister! v)))
        (is (identical? table (:raster.core/dispatch-table (meta v))))
        (is (empty? @table))
        (is (nil? (:arglists (meta v))))
        (is (< revision (dispatch/compiler-definition-revision)))
        (is (empty? @cache))
        (#'dispatch/complete-specialization! cache String claim identity)
        (is (empty? @cache) "a revoked in-flight claim cannot refill the cache")
        (is (= :stale-definition-proof-input
               (try (entry/assert-definition-inputs-current! @inputs) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (nil? (inference/registered-call-signature operation '[long])))
        (is (thrown? clojure.lang.ExceptionInfo (v 3)))
        (is (thrown? clojure.lang.ExceptionInfo (captured-dispatcher 3)))
        (is (number? (captured-target 3))
            "selected implementations are not revoked or unloaded")
        (is (identical? v (core/unregister! v)) "idempotent")
        (eval '(raster.core/deftm model [x :- Long y :- Long] :- Long x))
        (is (identical? v (ns-resolve scope 'model)))
        (is (= 7 (v 7 9)))
        (is (= 7 (captured-dispatcher 7 9))
            "captured generic dispatchers observe the same mutable table")
        (is (thrown? clojure.lang.ExceptionInfo (v 7)))))))

(deftest templates-cannot-regenerate-unregistered-overloads
  (with-scope
    (fn [scope]
      (eval '(raster.core/deftm model (All [T] [x :- T] :- T x)))
      (let [v (ns-resolve scope 'model)
            operation (symbol (str (ns-name scope)) "model")]
        (is (seq (get @dispatch/parametric-registry operation)))
        (core/unregister! v)
        (is (nil? (get @dispatch/parametric-registry operation)))
        (is (nil? (dispatch/try-parametric-register! operation [(float 2)])))
        (is (thrown? clojure.lang.ExceptionInfo (v (float 2))))
        (eval '(raster.core/deftm model (All [T] [x :- T] :- T x)))
        (is (= 2.0 (v (double 2))))))))

(deftest a-replacement-does-not-inherit-reducible-folding
  (with-scope
    (fn [scope]
      (eval '(raster.core/deftm model [x :- Long y :- Long] :- Long x))
      (let [v (ns-resolve scope 'model)]
        (dispatch/set-reducible! v)
        (is (= 1 (v 1 2 3)))
        (core/unregister! v)
        (is (nil? (:raster.core/reducible (meta v))))
        (eval '(raster.core/deftm model [x :- Long y :- Long] :- Long y))
        (is (= 2 (v 1 2)))
        (is (thrown? clojure.lang.ExceptionInfo (v 1 2 3)))))))

(deftest unregister-rejects-non-generics-without-mutating-them
  (is (= :invalid-generic-function
         (try (core/unregister! #'identity) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (is (= :invalid-generic-function
         (try (core/unregister! 'identity) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (is (= 4 (identity 4))))

(deftest a-detached-generic-var-cannot-clear-the-live-registry
  (with-scope
    (fn [scope]
      (eval '(raster.core/deftm model [x :- Long] :- Long x))
      (let [detached (ns-resolve scope 'model)
            old-table (:raster.core/dispatch-table (meta detached))]
        (ns-unmap scope 'model)
        (eval '(raster.core/deftm model [x :- Long] :- Long x))
        (let [live (ns-resolve scope 'model)]
          (is (not (identical? detached live)))
          (is (identical? old-table (:raster.core/dispatch-table (meta live))))
          (is (= :invalid-generic-function
                 (try (core/unregister! detached) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= 3 (live 3))))))))

(deftest empty-generics-replace-the-checkers-old-callable-signature
  (with-scope
    (fn [scope]
      (eval '(raster.core/deftm model [x :- Long] :- Long x))
      (let [v (ns-resolve scope 'model)
            operation (symbol (str (ns-name scope)) "model")
            annotations (atom [])]
        (with-redefs-fn
          {#'dispatch/tc-annotations-ready? (atom true)
           #'clojure.core/requiring-resolve
           (fn [symbol]
             (is (= 'clojure.core.typed/-ann symbol))
             (fn [annotation] (swap! annotations conj annotation)))}
          #(core/unregister! v))
        (is (= [[(ns-name scope) operation 'typed.clojure/Any {} nil nil]]
               @annotations))
        (is (identical? (:raster.core/dispatch-table (meta v))
                        (nth (get @(var-get #'dispatch/pending-tc-annotations)
                                  operation) 2)))))))
