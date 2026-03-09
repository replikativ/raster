(ns raster.compiler.ad.bind-ctx-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.ad.bind-ctx :as ctx]))

(defn- test-gensym
  "Deterministic gensym for testing."
  ([] (let [counter (atom 0)]
        (fn [prefix] (symbol (str prefix "__" (swap! counter inc))))))
  ([prefix] (symbol (str prefix "_t"))))

;; ================================================================
;; make-ctx
;; ================================================================

(deftest make-ctx-test
  (testing "creates empty context"
    (let [c (ctx/make-ctx test-gensym)]
      (is (= [] (:bindings c)))
      (is (fn? (:gensym-fn c))))))

;; ================================================================
;; genlet
;; ================================================================

(deftest genlet-test
  (testing "adds binding and returns symbol"
    (let [gs (test-gensym)
          c (ctx/make-ctx gs)
          [c2 sym] (ctx/genlet c "buf" '(nc/raw x))]
      (is (symbol? sym))
      (is (= 2 (count (:bindings c2))))
      (is (= sym (first (:bindings c2))))
      (is (= '(nc/raw x) (second (:bindings c2))))))

  (testing "multiple genlets accumulate bindings"
    (let [gs (test-gensym)
          c (ctx/make-ctx gs)
          [c1 s1] (ctx/genlet c "a" 1)
          [c2 s2] (ctx/genlet c1 "b" 2)
          [c3 s3] (ctx/genlet c2 "c" 3)]
      (is (= 6 (count (:bindings c3))))
      (is (= [s1 1 s2 2 s3 3] (:bindings c3))))))

;; ================================================================
;; emit-let*
;; ================================================================

(deftest emit-let-test
  (testing "emits let* with bindings"
    (let [gs (test-gensym)
          c (ctx/make-ctx gs)
          [c1 s1] (ctx/genlet c "a" '(+ x 1))
          [c2 s2] (ctx/genlet c1 "b" '(* x 2))
          result (ctx/emit-let* c2 s2)]
      (is (seq? result))
      (is (= 'let* (first result)))
      (is (= [s1 '(+ x 1) s2 '(* x 2)] (second result)))
      (is (= s2 (nth (vec result) 2)))))

  (testing "empty context returns body directly"
    (let [c (ctx/make-ctx test-gensym)]
      (is (= 'x (ctx/emit-let* c 'x))))))
