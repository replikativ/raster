(ns raster.compiler.simplify-test
  (:require [clojure.test :refer [deftest testing is are]]
            [raster.compiler.passes.scalar.simplify :as simp]
            [raster.compiler.passes.scalar.pe :as pe]))

;; ================================================================
;; Identity / Annihilator rules
;; ================================================================

(deftest addition-identity-test
  (testing "(+ x 0) => x"
    (is (= 'x (simp/simplify '(+ x 0))))
    (is (= 'x (simp/simplify '(+ x 0.0)))))
  (testing "(+ 0 x) => x"
    (is (= 'x (simp/simplify '(+ 0 x))))
    (is (= 'x (simp/simplify '(+ 0.0 x)))))
  (testing "(+ x y) unchanged"
    (is (= '(+ x y) (simp/simplify '(+ x y))))))

(deftest subtraction-identity-test
  (testing "(- x 0) => x"
    (is (= 'x (simp/simplify '(- x 0))))
    (is (= 'x (simp/simplify '(- x 0.0)))))
  (testing "(- x x) => 0.0"
    (is (= 0.0 (simp/simplify '(- x x)))))
  (testing "(- x y) unchanged"
    (is (= '(- x y) (simp/simplify '(- x y))))))

(deftest multiplication-identity-test
  (testing "(* x 1) => x"
    (is (= 'x (simp/simplify '(* x 1))))
    (is (= 'x (simp/simplify '(* x 1.0)))))
  (testing "(* 1 x) => x"
    (is (= 'x (simp/simplify '(* 1 x))))
    (is (= 'x (simp/simplify '(* 1.0 x)))))
  (testing "(* x 0) => 0.0"
    (is (= 0.0 (simp/simplify '(* x 0))))
    (is (= 0.0 (simp/simplify '(* 0 x)))))
  (testing "(* x y) unchanged"
    (is (= '(* x y) (simp/simplify '(* x y))))))

(deftest division-identity-test
  (testing "(/ x 1) => x"
    (is (= 'x (simp/simplify '(/ x 1))))
    (is (= 'x (simp/simplify '(/ x 1.0)))))
  (testing "(/ 0 x) => 0.0"
    (is (= 0.0 (simp/simplify '(/ 0 x)))))
  (testing "(/ x y) unchanged"
    (is (= '(/ x y) (simp/simplify '(/ x y))))))

;; ================================================================
;; Constant folding
;; ================================================================

(deftest constant-folding-test
  (testing "addition"
    (is (= 7.0 (simp/simplify '(+ 3.0 4.0))))
    (is (= 10 (simp/simplify '(+ 3 7)))))
  (testing "subtraction"
    (is (= 1.0 (simp/simplify '(- 4.0 3.0)))))
  (testing "multiplication"
    (is (= 12.0 (simp/simplify '(* 3.0 4.0)))))
  (testing "division"
    (is (= 2.0 (simp/simplify '(/ 6.0 3.0))))))

;; ================================================================
;; Nested simplification (bottom-up)
;; ================================================================

(deftest nested-simplification-test
  (testing "nested addition identity"
    (is (= 'x (simp/simplify '(+ (+ x 0) 0)))))
  (testing "nested multiplication identity"
    (is (= 'x (simp/simplify '(* (* x 1) 1)))))
  (testing "constant folding in nested"
    (is (= 24.0 (simp/simplify '(* (+ 3.0 5.0) (- 4.0 1.0)))))))

;; ================================================================
;; Derivative cleanup
;; ================================================================

(deftest derivative-cleanup-test
  (testing "(* 1.0 x) => x (AD identity)"
    (is (= 'x (simp/simplify '(* 1.0 x)))))
  (testing "(* x 1.0) => x"
    (is (= 'x (simp/simplify '(* x 1.0)))))
  (testing "(+ x 0.0) => x"
    (is (= 'x (simp/simplify '(+ x 0.0)))))
  (testing "(+ 0.0 x) => x"
    (is (= 'x (simp/simplify '(+ 0.0 x)))))
  (testing "nested derivative cleanup"
    (is (= 'x (simp/simplify-derivative '(+ (* 1.0 x) 0.0))))))

;; ================================================================
;; Let form simplification
;; ================================================================

(deftest let-simplification-test
  (testing "constant init is folded"
    (is (= '(let* [a 7.0] a)
           (simp/simplify '(let* [a (+ 3.0 4.0)] a)))))
  (testing "body is simplified"
    (is (= '(let* [a x] a)
           (simp/simplify '(let* [a x] (+ a 0)))))))

;; ================================================================
;; .invk call simplification
;; ================================================================

(deftest invk-simplification-test
  (testing ".invk with _plus_ in name"
    (let [form '(.invk some_plus__m_double_double-impl x 0)]
      ;; Should recognize as addition and simplify (+ x 0) => x
      (is (= 'x (simp/simplify form)))))
  (testing ".invk with _star_ in name"
    (let [form '(.invk some_star__m_double_double-impl x 1)]
      (is (= 'x (simp/simplify form)))))
  (testing ".invk constant fold"
    (let [form '(.invk _plus__m_double_double-impl 3.0 4.0)]
      (is (= 7.0 (simp/simplify form))))))

;; ================================================================
;; Math function simplification
;; ================================================================

(deftest math-simplification-test
  (testing "Math/sin constant"
    (is (= 0.0 (simp/simplify '(Math/sin 0.0)))))
  (testing "Math/cos constant"
    (is (= 1.0 (simp/simplify '(Math/cos 0.0)))))
  (testing "Math/sqrt constant"
    (is (= 2.0 (simp/simplify '(Math/sqrt 4.0)))))
  (testing "Math/exp constant"
    (is (= 1.0 (simp/simplify '(Math/exp 0.0))))))

;; ================================================================
;; Power/exponent simplification
;; ================================================================

(deftest power-simplification-test
  (testing "x^0 => 1.0"
    (is (= 1.0 (simp/simplify '(Math/pow x 0))))
    (is (= 1.0 (simp/simplify '(Math/pow x 0.0)))))
  (testing "x^1 => x"
    (is (= 'x (simp/simplify '(Math/pow x 1))))
    (is (= 'x (simp/simplify '(Math/pow x 1.0)))))
  (testing "x^2 => (* x x)"
    (is (= '(raster.numeric/* x x) (simp/simplify '(Math/pow x 2.0)))))
  (testing "x^0.5 => (Math/sqrt x)"
    (is (= '(Math/sqrt x) (simp/simplify '(Math/pow x 0.5)))))
  (testing "x^-1 => (/ 1.0 x)"
    (is (= '(raster.numeric// 1.0 x) (simp/simplify '(Math/pow x -1.0)))))
  (testing "constant folding"
    (is (= 8.0 (simp/simplify '(Math/pow 2.0 3.0))))))

;; ================================================================
;; Edge cases
;; ================================================================

(deftest edge-cases-test
  (testing "non-numeric forms pass through"
    (is (= '(str a b) (simp/simplify '(str a b)))))
  (testing "nil passes through"
    (is (nil? (simp/simplify nil))))
  (testing "symbol passes through"
    (is (= 'x (simp/simplify 'x))))
  (testing "empty list"
    (is (= '() (simp/simplify '()))))
  (testing "vector simplification"
    (is (= ['x 7.0] (simp/simplify ['x '(+ 3.0 4.0)])))))

;; ================================================================
;; Integration: simplifier + PE on realistic patterns
;; ================================================================

(deftest lorenz-specialization-test
  (testing "Lorenz with known sigma/rho/beta"
    (let [walked '(let* [dx (* sigma (- y x))
                         dy (- (* x (- rho z)) y)
                         dz (- (* x y) (* beta z))]
                        [dx dy dz])
          result (pe/pe walked {'sigma 10.0 'rho 28.0 'beta (/ 8.0 3.0)})]
      ;; sigma, rho, beta should be replaced with constants
      (is (not (some #{'sigma 'rho 'beta}
                     (flatten (if (seq? result) result [result]))))
          "All known parameters should be substituted")
      ;; x, y, z should remain
      (is (some #{'x} (flatten (if (seq? result) result [result])))
          "Unknown variable x should be preserved")))

  (testing "Walked .invk body with known params"
    (let [walked '(.invk _star__m_double_double-impl sigma (- y x))
          result (pe/pe walked {'sigma 10.0})]
      (is (= '(raster.numeric/* 10.0 (- y x))
             result)))))

(deftest derivative-chain-cleanup-test
  (testing "AD-generated expression cleanup"
    ;; After forward AD, expressions often look like:
    ;; (+ (* 1.0 (* dy y_dot)) (* 0.0 (* dx x_dot)))
    ;; Should simplify to: (* dy y_dot)
    (let [ad-expr '(+ (* 1.0 (* dy y_dot)) (* 0.0 (* dx x_dot)))]
      (is (= '(* dy y_dot) (simp/simplify-derivative ad-expr)))))

  (testing "Nested derivative identities"
    ;; (+ (+ (* 1.0 x) 0.0) (* 0.0 y)) => x
    (is (= 'x (simp/simplify-derivative '(+ (+ (* 1.0 x) 0.0) (* 0.0 y)))))))
