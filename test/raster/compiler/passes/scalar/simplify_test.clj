(ns raster.compiler.passes.scalar.simplify-test
  "Unit tests for the canonical algebraic simplifier.
   Complements the integration-level simplify_test.clj in compiler/
   with targeted pass-level tests including .invk normalization,
   power rules, and derivative cleanup."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.simplify :as simp]))

;; ================================================================
;; Arithmetic identity elimination
;; ================================================================

(deftest arithmetic-identity-test
  (testing "additive identity: (+ x 0) => x"
    (is (= 'x (simp/simplify-1 '(+ x 0))))
    (is (= 'x (simp/simplify-1 '(+ x 0.0))))
    (is (= 'x (simp/simplify-1 '(+ 0 x))))
    (is (= 'x (simp/simplify-1 '(+ 0.0 x)))))

  (testing "subtractive identity: (- x 0) => x"
    (is (= 'x (simp/simplify-1 '(- x 0))))
    (is (= 'x (simp/simplify-1 '(- x 0.0)))))

  (testing "multiplicative identity: (* x 1) => x"
    (is (= 'x (simp/simplify-1 '(* x 1))))
    (is (= 'x (simp/simplify-1 '(* x 1.0))))
    (is (= 'x (simp/simplify-1 '(* 1 x))))
    (is (= 'x (simp/simplify-1 '(* 1.0 x)))))

  (testing "division identity: (/ x 1) => x"
    (is (= 'x (simp/simplify-1 '(/ x 1))))
    (is (= 'x (simp/simplify-1 '(/ x 1.0)))))

  (testing "self-subtraction: (- x x) => 0.0"
    (is (= 0.0 (simp/simplify-1 '(- x x))))))

;; ================================================================
;; Zero multiplication
;; ================================================================

(deftest zero-multiplication-test
  (testing "(* x 0) => 0.0"
    (is (= 0.0 (simp/simplify-1 '(* x 0))))
    (is (= 0.0 (simp/simplify-1 '(* x 0.0)))))

  (testing "(* 0 x) => 0.0"
    (is (= 0.0 (simp/simplify-1 '(* 0 x))))
    (is (= 0.0 (simp/simplify-1 '(* 0.0 x)))))

  (testing "(/ 0 x) => 0.0"
    (is (= 0.0 (simp/simplify-1 '(/ 0 x))))
    (is (= 0.0 (simp/simplify-1 '(/ 0.0 x))))))

;; ================================================================
;; Constant folding
;; ================================================================

(deftest constant-folding-test
  (testing "addition of constants"
    (is (= 5 (simp/simplify-1 '(+ 2 3))))
    (is (= 5.5 (simp/simplify-1 '(+ 2.0 3.5)))))

  (testing "subtraction of constants"
    (is (= 1 (simp/simplify-1 '(- 3 2))))
    (is (= 1.5 (simp/simplify-1 '(- 4.5 3.0)))))

  (testing "multiplication of constants"
    (is (= 12 (simp/simplify-1 '(* 3 4))))
    (is (= 7.5 (simp/simplify-1 '(* 2.5 3.0)))))

  (testing "division of constants"
    (is (= 2.0 (simp/simplify-1 '(/ 6.0 3.0))))
    (is (= 0.5 (simp/simplify-1 '(/ 1.0 2.0)))))

  (testing "division by zero is not folded"
    (is (seq? (simp/simplify-1 '(/ 1.0 0.0)))
        "division by zero should not be folded")))

;; ================================================================
;; Power simplification
;; ================================================================

(deftest power-simplification-test
  (testing "x^0 => 1.0"
    (is (= 1.0 (simp/simplify-1 '(Math/pow x 0))))
    (is (= 1.0 (simp/simplify-1 '(Math/pow x 0.0)))))

  (testing "x^1 => x"
    (is (= 'x (simp/simplify-1 '(Math/pow x 1))))
    (is (= 'x (simp/simplify-1 '(Math/pow x 1.0)))))

  (testing "x^2 => (* x x)"
    (is (= '(raster.numeric/* x x)
           (simp/simplify-1 '(Math/pow x 2.0)))))

  (testing "x^0.5 => (Math/sqrt x)"
    (is (= '(Math/sqrt x)
           (simp/simplify-1 '(Math/pow x 0.5)))))

  (testing "x^-1 => (/ 1.0 x)"
    (is (= '(raster.numeric// 1.0 x)
           (simp/simplify-1 '(Math/pow x -1.0)))))

  (testing "constant power folding"
    (is (= 8.0 (simp/simplify-1 '(Math/pow 2.0 3.0))))))

;; ================================================================
;; Math constant folding
;; ================================================================

(deftest math-const-folding-test
  (testing "Math/sin of constant"
    (is (= (Math/sin 0.0) (simp/simplify-1 '(Math/sin 0.0)))))

  (testing "Math/cos of constant"
    (is (= (Math/cos 0.0) (simp/simplify-1 '(Math/cos 0.0)))))

  (testing "Math/exp of constant"
    (is (= (Math/exp 1.0) (simp/simplify-1 '(Math/exp 1.0)))))

  (testing "Math/sqrt of constant"
    (is (= 3.0 (simp/simplify-1 '(Math/sqrt 9.0)))))

  (testing "Math/abs of constant"
    (is (= 5.0 (simp/simplify-1 '(Math/abs -5.0))))))

;; ================================================================
;; .invk normalization and simplification
;; ================================================================

(deftest invk-normalization-test
  (testing ".invk with :raster.op/original metadata is simplified"
    (let [form (with-meta '(.invk impl__plus x 0)
                 {:raster.op/original 'raster.numeric/+})]
      (is (= 'x (simp/simplify-1 form)))))

  (testing ".invk with mangled _star_ name is simplified"
    (let [form '(.invk some_star__m_double_double-impl x 1)]
      (is (= 'x (simp/simplify-1 form)))))

  (testing ".invk non-arithmetic form passes through unchanged"
    (let [form '(.invk some_custom_fn x y)]
      (is (= form (simp/simplify-1 form))))))

;; ================================================================
;; Bottom-up recursive simplification
;; ================================================================

(deftest recursive-simplify-test
  (testing "nested expressions are simplified bottom-up"
    (is (= 'x (simp/simplify '(+ (+ x 0) 0))))
    (is (= 'x (simp/simplify '(* (* x 1) 1)))))

  (testing "let* body is simplified"
    (is (= '(let* [a x] a)
           (simp/simplify '(let* [a x] (+ a 0))))))

  (testing "constant folding in nested expressions"
    (is (= 24.0 (simp/simplify '(* (+ 3.0 5.0) (- 4.0 1.0)))))))

;; ================================================================
;; simplify-derivative fixed-point iteration
;; ================================================================

(deftest simplify-derivative-test
  (testing "AD chain rule cleanup"
    ;; After forward AD: (+ (* 1.0 x) (* 0.0 y)) => x
    (is (= 'x (simp/simplify-derivative '(+ (* 1.0 x) (* 0.0 y))))))

  (testing "deeply nested AD expression"
    ;; (+ (+ (* 1.0 (* dy y_dot)) (* 0.0 z)) 0.0)
    ;; => (* dy y_dot)
    (is (= '(* dy y_dot)
           (simp/simplify-derivative
            '(+ (+ (* 1.0 (* dy y_dot)) (* 0.0 z)) 0.0)))))

  (testing "fixpoint terminates when no further simplification possible"
    (is (= '(+ x y)
           (simp/simplify-derivative '(+ x y))))))
