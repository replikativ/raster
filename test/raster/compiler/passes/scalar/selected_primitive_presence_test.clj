(ns raster.compiler.passes.scalar.selected-primitive-presence-test
  (:refer-clojure :exclude [*])
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]
            [raster.compiler.passes.scalar.local-products :as products]))

(defn- selected-product []
  (with-meta '(.invk raster.numeric/_star__m_double_double-impl x y)
    {:raster.op/original 'raster.numeric/*}))

(defn- selected-result [call]
  (entry/call-with-definition-inputs #(scalar/checked-selected-primitive-result call)))

(defn- lower-owned [source]
  (entry/call-with-definition-inputs #(products/lower-local-products source)))

(deftest selected-presence-abstains-without-a-definition-owner
  (binding [entry/*emission-inputs* nil]
    (is (nil? (scalar/checked-selected-primitive-result (selected-product))))))

(deftest selected-body-proves-only-the-output-on-normal-return
  (let [call (selected-product)
        source (list 'let* ['x nil 'y 2.0 'z call]
                     '(if (clojure.core/nil? z) :absent :present))
        lowered (lower-owned source)
        failure (fn [body]
                  (try ((eval (list 'fn* [] body))) nil
                       (catch Throwable e (class e))))]
    (is (= {:type :double} (selected-result call)))
    (is (= :present (last lowered)))
    (is (= (second source) (second lowered)) "the nullable input and call remain executed")
    (is (some? (failure source)))
    (is (= (failure source) (failure lowered)))))

(deftest declaration-and-name-alone-never-prove-presence
  (doseq [call ['(raster.numeric/* x y)
                (with-meta '(unknown.helper/product x y) {:raster.type/tag 'double})
                (with-meta '(.invk missing.namespace/receiver x y)
                  {:raster.op/original 'raster.numeric/*})
                (with-meta (selected-product) {:raster.op/original 'raster.numeric/*
                                               :raster.op/lexical-callee true})]]
    (is (nil? (selected-result call))))
  (with-redefs [raster.numeric/_star__m_double_double-impl (fn [& _] nil)]
    (is (nil? (selected-result (selected-product)))))
  (with-redefs [raster.numeric/_star__m_double_double (fn [& _] nil)]
    (is (nil? (selected-result (selected-product))))))

(deftest actual-source-body-must-be-a-strict-producer
  (let [source-var (find-var 'raster.numeric/_star__m_double_double)
        original (meta source-var)]
    (try
      (doseq [body ['[nil] '[x] '[(identity nil)] '[(if x nil y)]]]
        (alter-meta! source-var assoc :raster.core/deftm-source-body body)
        (is (nil? (selected-result (selected-product)))))
      (finally (reset-meta! source-var original)))
    (is (= {:type :double} (selected-result (selected-product))))))

(deftest cold-stub-source-redefinition-cannot-borrow-stale-metadata
  ;; Recreate the isolated fixture on EACH run: a repeated warm-REPL suite
  ;; must still exercise a genuinely cold stub, independent of test order.
  (binding [*ns* (the-ns 'raster.compiler.passes.scalar.selected-primitive-presence-test)]
    (eval '(deftm * [x :- Double y :- Double] :- Double (clojure.core/* x y))))
  (let [generic #'n/* original (meta generic)
        fixture (find-var 'raster.compiler.passes.scalar.selected-primitive-presence-test/*)
        fixture-table (:raster.core/dispatch-table (meta fixture))
        call (with-meta
               '(.invk raster.compiler.passes.scalar.selected-primitive-presence-test/_star__m_double_double-impl x y)
               {:raster.op/original 'raster.numeric/*})
        receiver (find-var 'raster.compiler.passes.scalar.selected-primitive-presence-test/_star__m_double_double-impl)
        source (find-var 'raster.compiler.passes.scalar.selected-primitive-presence-test/_star__m_double_double)]
    (try
      (alter-meta! generic assoc :raster.core/dispatch-table fixture-table)
      (is (some? (:raster.core/jit-spec (meta receiver))) "the receiver is a real cold lazy stub")
      (is (= {:type :double} (selected-result call)))
      (with-redefs-fn {source (fn [& _] nil)}
        (fn []
          (is (nil? (selected-result call)))
          ;; The cold stub can in fact return nil through its delegated source
          ;; Var: matching only the typed receiver would have been unsound.
          (is (nil? (clojure.lang.Reflector/invokeInstanceMethod
                     @receiver "invk" (object-array [2.0 3.0]))))))
      (finally (reset-meta! generic original)))))

(deftest selected-presence-does-not-rewrite-floating-checkpoints
  (let [call (with-meta '(.invk raster.numeric/_star__m_float_float-impl x y)
               {:raster.op/original 'raster.numeric/*})
        source (list 'let* ['x (Float/valueOf (float -0.0)) 'y (Float/valueOf (float 2.0)) 'z call]
                     '(if (clojure.core/nil? z) nil z))
        lowered (lower-owned source)
        evaluate #((eval (list 'fn* [] %)))
        before (evaluate source) after (evaluate lowered)]
    (is (= (second source) (second lowered)))
    (is (instance? Float after))
    (is (= (Float/floatToRawIntBits before) (Float/floatToRawIntBits after)))))
