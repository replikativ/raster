(ns raster.compiler.core.walker-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.walker :as walker]
            [raster.numeric]))

;; Helper: build type-env from flat {sym → tag}
(defn- te [flat-env]
  (reduce-kv (fn [m s t] (assoc m s {:tag t})) {} flat-env))

(defn- wb
  "Walk a form with a flat env."
  ([form] (walker/walk-body form {:type-env {}}))
  ([form flat-env] (walker/walk-body form {:type-env (te flat-env)})))

;; ================================================================
;; classify-form
;; ================================================================

(deftest classify-let-test
  (testing "let forms"
    (let [ctx (walker/make-ctx {})]
      (is (= :let (walker/classify-form '(let [x 1] x) ctx)))
      (is (= :let (walker/classify-form '(let* [x 1] x) ctx)))
      (is (= :let (walker/classify-form '(loop [i 0] i) ctx))))))

(deftest classify-fn-test
  (testing "fn forms"
    (let [ctx (walker/make-ctx {})]
      (is (= :fn (walker/classify-form '(fn [x] x) ctx)))
      (is (= :fn (walker/classify-form '(fn* [x] x) ctx))))))

(deftest classify-leaf-test
  (testing "literals and symbols"
    (let [ctx (walker/make-ctx {})]
      (is (= :leaf (walker/classify-form 42 ctx)))
      (is (= :leaf (walker/classify-form 3.14 ctx)))
      (is (= :leaf (walker/classify-form 'x ctx)))
      (is (= :leaf (walker/classify-form nil ctx))))))

(deftest classify-deftm-call-test
  (testing "deftm function calls"
    (let [ctx (walker/make-ctx {:type-env (te {'x 'double 'y 'double})
                                :source-ns (the-ns 'raster.compiler.core.walker-test)})]
      (is (= :deftm-call (walker/classify-form '(raster.numeric/+ x y) ctx))))))

(deftest classify-array-op-test
  (testing "array operations"
    (let [ctx (walker/make-ctx {:type-env (te {'a 'doubles})
                                :source-ns (the-ns 'raster.compiler.core.walker-test)})]
      (is (= :array-op (walker/classify-form '(aget a 0) ctx))))))

(deftest classify-vector-test
  (testing "vector forms"
    (let [ctx (walker/make-ctx {})]
      (is (= :vector (walker/classify-form [1 2 3] ctx))))))

;; ================================================================
;; walk — integration via walk-body
;; ================================================================

(deftest walk-literal-test
  (testing "literals pass through"
    (is (= 42 (wb 42)))
    (is (= 3.14 (wb 3.14)))))

(deftest walk-symbol-test
  (testing "symbols pass through"
    (is (= 'x (wb 'x {'x 'double})))))

(deftest walk-let-hinting-test
  (testing "array bindings get type hints"
    (let [walked (wb '(let [buf (double-array 16)] buf))]
      (is (seq? walked))
      (is (= 'let (first walked)))
      (let [bindings (partition 2 (second walked))
            [sym _init] (first bindings)]
        ;; buf should be hinted as doubles (array type)
        (is (= 'doubles (:tag (meta sym))))))))

(deftest walk-deftm-devirtualization-test
  (testing "deftm call devirtualized to .invk"
    (let [walked (wb '(let [z (raster.numeric/+ x y)] z)
                     {'x 'double 'y 'double})]
      ;; Should produce .invk call
      (let [bindings (partition 2 (second walked))
            [_ init] (first bindings)]
        (is (seq? init))
        (is (= '.invk (first init)))))))

(deftest walk-op-metadata-test
  (testing ".invk forms carry :raster.op/original metadata"
    (let [walked (wb '(let [z (raster.numeric/+ x y)] z)
                     {'x 'double 'y 'double})]
      (let [bindings (partition 2 (second walked))
            [_ init] (first bindings)]
        (is (= 'raster.numeric/+ (:raster.op/original (meta init))))))))

(deftest walk-nested-let-test
  (testing "nested let bindings are walked"
    (let [walked (wb '(let [a (raster.numeric/+ x y)
                            b (raster.numeric/* a z)]
                        b)
                     {'x 'double 'y 'double 'z 'double})]
      ;; Both bindings should be devirtualized
      (let [bindings (vec (partition 2 (second walked)))]
        (is (= 2 (count bindings)))
        (is (= '.invk (first (second (first bindings)))))
        (is (= '.invk (first (second (second bindings)))))))))

(deftest walk-vector-test
  (testing "vectors are walked element-wise"
    (let [walked (wb '(let [a (raster.numeric/+ x y)] [a 42])
                     {'x 'double 'y 'double})]
      (let [body (last (nnext walked))]
        (is (vector? body))))))

(deftest walk-if-test
  (testing "if branches are walked"
    (let [walked (wb '(if true x y) {'x 'double 'y 'double})]
      (is (seq? walked))
      (is (= 'if (first walked))))))

;; ================================================================
;; Object devirtualization guard
;; ================================================================

(deftest walk-object-no-devirtualize-test
  (testing "Object-tagged methods are NOT devirtualized (promotion fallbacks)"
    ;; When both args are Object, walker should leave as qualified dispatch call
    (let [walked (wb '(raster.numeric/* x y) {'x 'Object 'y 'Object})]
      (is (seq? walked))
      (is (= 'raster.numeric/* (first walked))
          "Object+Object should stay as dispatch call, not devirtualize to _star__m_Object_Object"))
    ;; When one arg is inferred Object (e.g. from Object-returning fn),
    ;; the call should still not devirtualize to the Object fallback
    (let [walked (wb '(raster.numeric/* x y) {'x 'Number 'y 'Object})]
      (is (not (and (seq? walked)
                    (let [head (first walked)]
                      (and (symbol? head)
                           (.contains (name head) "Object")))))
          "Should not devirtualize when any arg resolves to Object fallback")))

  (testing "Number-tagged methods ARE devirtualized (concrete implementations)"
    (let [walked (wb '(raster.numeric/* x y) {'x 'Number 'y 'Number})]
      (is (seq? walked))
      (is (= '.invk (first walked))
          "Number+Number should devirtualize to .invk")
      (is (let [impl (second walked)]
            (and (symbol? impl)
                 (.contains (name impl) "Number")))
          "Number+Number impl should contain 'Number'")))

  (testing "double-tagged methods ARE devirtualized"
    (let [walked (wb '(raster.numeric/* x y) {'x 'double 'y 'double})]
      (is (seq? walked))
      (is (= '.invk (first walked))
          "double+double should devirtualize to .invk"))))
