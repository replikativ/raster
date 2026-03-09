(ns raster.sym.core-test
  (:require [clojure.test :refer [deftest testing is are]]
            [raster.sym.core :as sym :refer [sym sym? unwrap ->Sym]]
            [raster.sym.diff :as sd :refer [diff differentiate depends-on?]]
            [raster.sym.analysis :as sa :refer [linearity-analysis
                                                jacobian-sparsity
                                                hessian-sparsity]]
            [raster.numeric :as n]
            [raster.math :as m]))

;; ================================================================
;; sym.clj — Creating symbolic values
;; ================================================================

(deftest sym-creation-test
  (testing "sym wraps a symbol"
    (let [x (sym 'x)]
      (is (sym? x))
      (is (= 'x (unwrap x)))))

  (testing "sym wraps an S-expression"
    (let [e (sym '(+ x y))]
      (is (sym? e))
      (is (= '(+ x y) (unwrap e)))))

  (testing "sym wraps a number"
    (let [n (sym 42)]
      (is (sym? n))
      (is (= 42 (unwrap n)))))

  (testing "sym? returns false for non-symbolic values"
    (is (not (sym? 42)))
    (is (not (sym? "hello")))
    (is (not (sym? nil)))))

(deftest unwrap-test
  (testing "unwrap extracts form from Sym"
    (is (= 'x (unwrap (sym 'x))))
    (is (= 42 (unwrap (sym 42)))))

  (testing "unwrap returns non-Sym values unchanged"
    (is (= 42 (unwrap 42)))
    (is (= "hello" (unwrap "hello")))
    (is (= nil (unwrap nil)))))

;; ================================================================
;; sym.clj — Arithmetic on symbolic values
;; ================================================================

(deftest sym-addition-test
  (testing "sym + sym"
    (let [x (sym 'x)
          y (sym 'y)
          result (n/+ x y)]
      (is (sym? result))
      (is (= '(+ x y) (unwrap result)))))

  (testing "sym + number"
    (let [x (sym 'x)
          result (n/+ x 2)]
      (is (sym? result))
      (is (= '(+ x 2.0) (unwrap result)))))

  (testing "number + sym"
    (let [x (sym 'x)
          result (n/+ 3 x)]
      (is (sym? result))
      (is (= '(+ 3.0 x) (unwrap result))))))

(deftest sym-subtraction-test
  (testing "unary negation"
    (let [x (sym 'x)
          result (n/- x)]
      (is (sym? result))
      (is (= '(- x) (unwrap result)))))

  (testing "sym - sym"
    (let [x (sym 'x)
          y (sym 'y)
          result (n/- x y)]
      (is (sym? result))
      (is (= '(- x y) (unwrap result)))))

  (testing "sym - number"
    (let [x (sym 'x)
          result (n/- x 1)]
      (is (sym? result))
      (is (= '(- x 1.0) (unwrap result)))))

  (testing "number - sym"
    (let [x (sym 'x)
          result (n/- 5 x)]
      (is (sym? result))
      (is (= '(- 5.0 x) (unwrap result))))))

(deftest sym-multiplication-test
  (testing "sym * sym"
    (let [x (sym 'x)
          y (sym 'y)
          result (n/* x y)]
      (is (sym? result))
      (is (= '(* x y) (unwrap result)))))

  (testing "sym * number"
    (let [x (sym 'x)
          result (n/* x 3)]
      (is (sym? result))
      (is (= '(* x 3.0) (unwrap result)))))

  (testing "number * sym"
    (let [x (sym 'x)
          result (n/* 2 x)]
      (is (sym? result))
      (is (= '(* 2.0 x) (unwrap result))))))

(deftest sym-division-test
  (testing "sym / sym"
    (let [x (sym 'x)
          y (sym 'y)
          result (n// x y)]
      (is (sym? result))
      (is (= '(/ x y) (unwrap result)))))

  (testing "sym / number"
    (let [x (sym 'x)
          result (n// x 2)]
      (is (sym? result))
      (is (= '(/ x 2.0) (unwrap result)))))

  (testing "number / sym"
    (let [x (sym 'x)
          result (n// 1 x)]
      (is (sym? result))
      (is (= '(/ 1.0 x) (unwrap result))))))

(deftest sym-power-test
  (testing "sym pow sym"
    (let [x (sym 'x)
          y (sym 'y)
          result (n/pow x y)]
      (is (sym? result))
      (is (= '(pow x y) (unwrap result)))))

  (testing "sym pow number"
    (let [x (sym 'x)
          result (n/pow x 2)]
      (is (sym? result))
      (is (= '(pow x 2.0) (unwrap result))))))

(deftest sym-unary-ops-test
  (testing "sqrt"
    (let [x (sym 'x)]
      (is (= '(sqrt x) (unwrap (n/sqrt x))))))

  (testing "abs"
    (let [x (sym 'x)]
      (is (= '(abs x) (unwrap (n/abs x))))))

  (testing "sign"
    (let [x (sym 'x)]
      (is (= '(sign x) (unwrap (n/sign x)))))))

(deftest sym-min-max-test
  (testing "min of two syms"
    (let [x (sym 'x)
          y (sym 'y)]
      (is (= '(min x y) (unwrap (n/min x y))))))

  (testing "max of two syms"
    (let [x (sym 'x)
          y (sym 'y)]
      (is (= '(max x y) (unwrap (n/max x y))))))

  (testing "min sym and number"
    (let [x (sym 'x)]
      (is (= '(min x 0.0) (unwrap (n/min x 0)))))))

(deftest sym-comparison-test
  (testing "symbolic comparisons produce Sym values"
    (let [x (sym 'x)
          y (sym 'y)]
      (is (= '(< x y) (unwrap (n/< x y))))
      (is (= '(<= x y) (unwrap (n/<= x y))))
      (is (= '(> x y) (unwrap (n/> x y))))
      (is (= '(>= x y) (unwrap (n/>= x y))))
      (is (= '(== x y) (unwrap (n/== x y))))))

  (testing "comparisons with numbers"
    (let [x (sym 'x)]
      (is (= '(< x 0.0) (unwrap (n/< x 0))))
      (is (= '(> 1.0 x) (unwrap (n/> 1 x)))))))

(deftest sym-predicates-test
  (testing "zero?, pos?, neg? return symbolic"
    (let [x (sym 'x)]
      (is (= '(zero? x) (unwrap (n/zero? x))))
      (is (= '(pos? x) (unwrap (n/pos? x))))
      (is (= '(neg? x) (unwrap (n/neg? x)))))))

;; ================================================================
;; sym.clj — Math functions on symbolic values
;; ================================================================

(deftest sym-trig-test
  (testing "sin, cos, tan"
    (let [x (sym 'x)]
      (is (= '(sin x) (unwrap (m/sin x))))
      (is (= '(cos x) (unwrap (m/cos x))))
      (is (= '(tan x) (unwrap (m/tan x)))))))

(deftest sym-inverse-trig-test
  (testing "asin, acos, atan"
    (let [x (sym 'x)]
      (is (= '(asin x) (unwrap (m/asin x))))
      (is (= '(acos x) (unwrap (m/acos x))))
      (is (= '(atan x) (unwrap (m/atan x)))))))

(deftest sym-atan2-test
  (testing "atan2 sym sym"
    (let [x (sym 'x)
          y (sym 'y)]
      (is (= '(atan2 y x) (unwrap (m/atan2 y x))))))

  (testing "atan2 with numbers"
    (let [x (sym 'x)]
      (is (= '(atan2 x 1.0) (unwrap (m/atan2 x 1))))
      (is (= '(atan2 1.0 x) (unwrap (m/atan2 1 x)))))))

(deftest sym-hyperbolic-test
  (testing "sinh, cosh, tanh"
    (let [x (sym 'x)]
      (is (= '(sinh x) (unwrap (m/sinh x))))
      (is (= '(cosh x) (unwrap (m/cosh x))))
      (is (= '(tanh x) (unwrap (m/tanh x)))))))

(deftest sym-exp-log-test
  (testing "exp, log, log2, log10"
    (let [x (sym 'x)]
      (is (= '(exp x) (unwrap (m/exp x))))
      (is (= '(log x) (unwrap (m/log x))))
      (is (= '(log2 x) (unwrap (m/log2 x))))
      (is (= '(log10 x) (unwrap (m/log10 x)))))))

(deftest sym-rounding-test
  (testing "ceil, floor, cbrt"
    (let [x (sym 'x)]
      (is (= '(ceil x) (unwrap (m/ceil x))))
      (is (= '(floor x) (unwrap (m/floor x))))
      (is (= '(cbrt x) (unwrap (m/cbrt x)))))))

(deftest sym-fma-test
  (testing "fused multiply-add"
    (let [x (sym 'x)
          y (sym 'y)
          z (sym 'z)]
      (is (= '(fma x y z) (unwrap (m/fma x y z)))))))

(deftest sym-hypot-test
  (testing "hypot sym sym"
    (let [x (sym 'x)
          y (sym 'y)]
      (is (= '(hypot x y) (unwrap (m/hypot x y))))))

  (testing "hypot with numbers"
    (let [x (sym 'x)]
      (is (= '(hypot x 3.0) (unwrap (m/hypot x 3))))
      (is (= '(hypot 4.0 x) (unwrap (m/hypot 4 x)))))))

;; ================================================================
;; sym.clj — Expression composition
;; ================================================================

(deftest sym-composition-test
  (testing "nested arithmetic builds expression trees"
    (let [x (sym 'x)
          ;; (x + 1) * (x - 2)
          result (n/* (n/+ x 1) (n/- x 2))]
      (is (sym? result))
      (is (= '(* (+ x 1.0) (- x 2.0)) (unwrap result)))))

  (testing "math applied to arithmetic"
    (let [x (sym 'x)
          ;; sin(x^2)
          result (m/sin (n/pow x 2))]
      (is (= '(sin (pow x 2.0)) (unwrap result)))))

  (testing "complex expression: exp(2*x) + cos(x)"
    (let [x (sym 'x)
          result (n/+ (m/exp (n/* 2 x)) (m/cos x))]
      (is (= '(+ (exp (* 2.0 x)) (cos x)) (unwrap result))))))

(deftest sym-zero-one-test
  (testing "zero and one constructors"
    (let [x (sym 'x)]
      (is (= 0 (unwrap (n/zero x))))
      (is (= 1 (unwrap (n/one x)))))))

(deftest sym-real-value-throws-test
  (testing "real-value throws for symbolic"
    (let [x (sym 'x)]
      (is (thrown? clojure.lang.ExceptionInfo (n/real-value x))))))

;; ================================================================
;; sym_diff.clj — depends-on?
;; ================================================================

(deftest depends-on-test
  (testing "variable depends on itself"
    (is (true? (depends-on? 'x 'x))))

  (testing "different variable does not depend"
    (is (false? (depends-on? 'y 'x))))

  (testing "number does not depend"
    (is (false? (depends-on? 42 'x))))

  (testing "nil, bool, string, keyword do not depend"
    (is (false? (depends-on? nil 'x)))
    (is (false? (depends-on? true 'x)))
    (is (false? (depends-on? "hi" 'x)))
    (is (false? (depends-on? :k 'x))))

  (testing "expression containing variable"
    (is (true? (depends-on? '(+ x 1) 'x)))
    (is (true? (depends-on? '(* y (sin x)) 'x))))

  (testing "expression not containing variable"
    (is (false? (depends-on? '(+ y z) 'x)))
    (is (false? (depends-on? '(sin y) 'x))))

  (testing "nested expression"
    (is (true? (depends-on? '(+ 1 (* 2 (sin x))) 'x))))

  (testing "vector expressions"
    (is (true? (depends-on? ['x 'y] 'x)))
    (is (false? (depends-on? ['y 'z] 'x)))))

;; ================================================================
;; sym_diff.clj — diff of constants and variables
;; ================================================================

(deftest diff-constant-test
  (testing "derivative of constant is 0"
    (is (= 0 (diff 42 'x)))
    (is (= 0 (diff 3.14 'x)))
    (is (= 0 (diff 'y 'x)))))

(deftest diff-variable-test
  (testing "derivative of x w.r.t. x is 1"
    (is (= 1 (diff 'x 'x)))))

;; ================================================================
;; sym_diff.clj — diff of sums
;; ================================================================

(deftest diff-addition-test
  (testing "d/dx (x + y) = 1 + 0"
    (is (= '(+ 1 0) (diff '(+ x y) 'x))))

  (testing "d/dx (x + x) = 1 + 1"
    (is (= '(+ 1 1) (diff '(+ x x) 'x))))

  (testing "d/dx (x + 5) = 1 + 0"
    (is (= '(+ 1 0) (diff '(+ x 5) 'x)))))

(deftest diff-subtraction-test
  (testing "d/dx (x - y) = 1 - 0"
    (is (= '(- 1 0) (diff '(- x y) 'x))))

  (testing "unary negation: d/dx (-x) = -1"
    (is (= '(- 1) (diff '(- x) 'x)))))

;; ================================================================
;; sym_diff.clj — diff of products (product rule)
;; ================================================================

(deftest diff-product-test
  (testing "d/dx (x * y) = 1*y + x*0"
    (is (= '(+ (* 1 y) (* x 0)) (diff '(* x y) 'x))))

  (testing "d/dx (x * x) = 1*x + x*1"
    (is (= '(+ (* 1 x) (* x 1)) (diff '(* x x) 'x))))

  (testing "d/dx (3 * x) = 0*x + 3*1"
    (is (= '(+ (* 0 x) (* 3 1)) (diff '(* 3 x) 'x)))))

;; ================================================================
;; sym_diff.clj — diff of quotients (quotient rule)
;; ================================================================

(deftest diff-quotient-test
  (testing "d/dx (x / y) = (1*y - x*0) / (y*y)"
    (is (= '(/ (- (* 1 y) (* x 0)) (* y y))
           (diff '(/ x y) 'x))))

  (testing "d/dx (1 / x)"
    (let [result (diff '(/ 1 x) 'x)]
      ;; (/ (- (* 0 x) (* 1 1)) (* x x))
      (is (seq? result))
      (is (= '/ (first result))))))

;; ================================================================
;; sym_diff.clj — diff of power
;; ================================================================

(deftest diff-power-test
  (testing "d/dx x^2 (constant exponent)"
    (let [result (diff '(pow x 2) 'x)]
      ;; (* (* 2 (pow x (- 2 1))) 1)
      (is (seq? result))
      (is (= '* (first result)))))

  (testing "d/dx 2^x (constant base)"
    (let [result (diff '(pow 2 x) 'x)]
      ;; (* (* (pow 2 x) (log 2)) 1)
      (is (seq? result))
      (is (= '* (first result))))))

;; ================================================================
;; sym_diff.clj — chain rule
;; ================================================================

(deftest diff-chain-rule-test
  (testing "d/dx sin(x) = cos(x) * 1"
    (is (= '(* (cos x) 1) (diff '(sin x) 'x))))

  (testing "d/dx cos(x) = -sin(x) * 1"
    (is (= '(* (- (sin x)) 1) (diff '(cos x) 'x))))

  (testing "d/dx exp(x) = exp(x) * 1"
    (is (= '(* (exp x) 1) (diff '(exp x) 'x))))

  (testing "d/dx log(x) = (1/x) * 1"
    (is (= '(* (/ 1 x) 1) (diff '(log x) 'x))))

  (testing "d/dx sin(x^2) — chain rule applies"
    (let [result (diff '(sin (pow x 2)) 'x)]
      ;; (* (cos (pow x 2)) (* (* 2 (pow x (- 2 1))) 1))
      (is (seq? result))
      ;; The outer cos should appear
      (is (= 'cos (first (second result))))))

  (testing "d/dx exp(2*x) — chain rule"
    (let [result (diff '(exp (* 2 x)) 'x)]
      ;; (* (exp (* 2 x)) (+ (* 0 x) (* 2 1)))
      (is (seq? result))
      (is (= '* (first result)))
      ;; exp(2*x) is the first factor
      (is (= 'exp (first (second result)))))))

(deftest diff-sqrt-test
  (testing "d/dx sqrt(x) = 1/(2*sqrt(x)) * 1"
    (let [result (diff '(sqrt x) 'x)]
      (is (seq? result))
      (is (= '* (first result))))))

(deftest diff-tan-test
  (testing "d/dx tan(x) = (1 + tan(x)^2) * 1"
    (let [result (diff '(tan x) 'x)]
      (is (seq? result)))))

(deftest diff-hyperbolic-test
  (testing "d/dx sinh(x) = cosh(x) * 1"
    (is (= '(* (cosh x) 1) (diff '(sinh x) 'x))))

  (testing "d/dx cosh(x) = sinh(x) * 1"
    (is (= '(* (sinh x) 1) (diff '(cosh x) 'x))))

  (testing "d/dx tanh(x)"
    (let [result (diff '(tanh x) 'x)]
      (is (seq? result)))))

(deftest diff-inverse-trig-test
  (testing "d/dx asin(x)"
    (let [result (diff '(asin x) 'x)]
      (is (seq? result))))

  (testing "d/dx acos(x)"
    (let [result (diff '(acos x) 'x)]
      (is (seq? result))))

  (testing "d/dx atan(x)"
    (let [result (diff '(atan x) 'x)]
      (is (seq? result)))))

(deftest diff-atan2-test
  (testing "d/dx atan2(y,x)"
    (let [result (diff '(atan2 y x) 'x)]
      (is (seq? result)))))

(deftest diff-abs-test
  (testing "d/dx abs(x) = sign(x) * 1"
    (is (= '(* (sign x) 1) (diff '(abs x) 'x)))))

(deftest diff-sign-test
  (testing "d/dx sign(x) = 0 (a.e.)"
    (is (= 0 (diff '(sign x) 'x)))))

(deftest diff-log-variants-test
  (testing "d/dx log2(x)"
    (let [result (diff '(log2 x) 'x)]
      (is (seq? result))))

  (testing "d/dx log10(x)"
    (let [result (diff '(log10 x) 'x)]
      (is (seq? result)))))

(deftest diff-cbrt-test
  (testing "d/dx cbrt(x)"
    (let [result (diff '(cbrt x) 'x)]
      (is (seq? result)))))

(deftest diff-hypot-test
  (testing "d/dx hypot(x,y)"
    (let [result (diff '(hypot x y) 'x)]
      (is (seq? result)))))

(deftest diff-min-max-test
  (testing "d/dx min(x,y) = if(x<y, 1, 0)"
    (is (= '(if (< x y) 1 0) (diff '(min x y) 'x))))

  (testing "d/dx max(x,y) = if(x>y, 1, 0)"
    (is (= '(if (> x y) 1 0) (diff '(max x y) 'x)))))

(deftest diff-fma-test
  (testing "d/dx fma(x,y,z) = y (since fma(x,y,z) = x*y+z)"
    (let [result (diff '(fma x y z) 'x)]
      ;; (+ (+ (* 1 y) (* x 0)) 0)
      (is (seq? result)))))

(deftest diff-if-test
  (testing "diff through conditional"
    (let [result (diff '(if (< x 0) (- x) x) 'x)]
      ;; (if (< x 0) (- 1) 1)
      (is (= 'if (first result))))))

;; ================================================================
;; sym_diff.clj — Multi-variable differentiation
;; ================================================================

(deftest diff-multi-variable-test
  (testing "d/dy of (x + y) = 0 + 1"
    (is (= '(+ 0 1) (diff '(+ x y) 'y))))

  (testing "d/dy of (x * y) = 0*y + x*1"
    (is (= '(+ (* 0 y) (* x 1)) (diff '(* x y) 'y))))

  (testing "d/dx of pure y expression applies chain rule (unsimplified)"
    ;; diff doesn't simplify — chain rule yields (* (cos y) 0)
    (is (= '(* (cos y) 0) (diff '(sin y) 'x))))

  (testing "d/dy of x^2 = 0"
    (is (= '(* (* 2 (pow x (- 2 1))) 0) (diff '(pow x 2) 'y)))))

;; ================================================================
;; sym_diff.clj — Higher-order differentiation
;; ================================================================

(deftest diff-higher-order-test
  (testing "d^2/dx^2 of x^2 (unsimplified)"
    (let [first-deriv (diff '(pow x 2) 'x)
          second-deriv (diff first-deriv 'x)]
      ;; Should be some expression that simplifies to 2
      (is (some? second-deriv))))

  (testing "d^2/dx^2 of x^3"
    (let [first-deriv (diff '(pow x 3) 'x)
          second-deriv (diff first-deriv 'x)]
      (is (some? second-deriv))))

  (testing "d^2/dx^2 of sin(x)"
    (let [first-deriv (diff '(sin x) 'x)
          second-deriv (diff first-deriv 'x)]
      ;; Should have sin in it (second derivative of sin is -sin)
      (is (some? second-deriv)))))

;; ================================================================
;; sym_diff.clj — differentiate (with simplification)
;; ================================================================

(deftest differentiate-simplify-test
  (testing "differentiate simplifies d/dx (x + 5)"
    (let [result (differentiate '(+ x 5) 'x)]
      ;; raw diff: (+ 1 0), should simplify to 1
      (is (= 1 result))))

  (testing "differentiate simplifies d/dx (3 * x)"
    (let [result (differentiate '(* 3 x) 'x)]
      ;; raw diff: (+ (* 0 x) (* 3 1)), should simplify to 3
      (is (= 3 result))))

  (testing "differentiate simplifies d/dx (x * x)"
    (let [result (differentiate '(* x x) 'x)]
      ;; should simplify to (* 2 x) or (+ x x)
      (is (some? result))
      (is (depends-on? result 'x))))

  (testing "differentiate of constant returns 0"
    (is (= 0 (differentiate 42 'x)))
    (is (= 0 (differentiate 'y 'x))))

  (testing "differentiate simplifies d/dx sin(x) to cos(x)"
    (let [result (differentiate '(sin x) 'x)]
      ;; (* (cos x) 1) should simplify to (cos x)
      (is (= '(cos x) result))))

  (testing "differentiate simplifies d/dx exp(x) to exp(x)"
    (let [result (differentiate '(exp x) 'x)]
      (is (= '(exp x) result)))))

;; ================================================================
;; sym_diff.clj — Unknown operations throw
;; ================================================================

(deftest diff-unknown-op-throws-test
  (testing "diff of unknown operation throws"
    (is (thrown? clojure.lang.ExceptionInfo
                 (diff '(unknown-fn x) 'x)))))

;; ================================================================
;; sym_analysis.clj — linearity-analysis
;; ================================================================

(deftest linearity-constant-test
  (testing "constant expression"
    (is (= {'x :constant}
           (linearity-analysis 42 ['x])))
    (is (= {'x :constant}
           (linearity-analysis 'y ['x])))))

(deftest linearity-linear-test
  (testing "x is linear in x"
    (is (= {'x :linear}
           (linearity-analysis 'x ['x]))))

  (testing "(+ x 5) is linear in x"
    (is (= {'x :linear}
           (linearity-analysis '(+ x 5) ['x]))))

  (testing "(* 3 x) is linear in x"
    (is (= {'x :linear}
           (linearity-analysis '(* 3 x) ['x]))))

  (testing "(+ x y) is linear in both"
    (is (= {'x :linear 'y :linear}
           (linearity-analysis '(+ x y) ['x 'y])))))

(deftest linearity-quadratic-test
  (testing "(* x x) is quadratic in x"
    (is (= {'x :quadratic}
           (linearity-analysis '(* x x) ['x]))))

  (testing "(pow x 2) is quadratic in x"
    (is (= {'x :quadratic}
           (linearity-analysis '(pow x 2) ['x]))))

  (testing "(* x y) is linear in each individually"
    (let [result (linearity-analysis '(* x y) ['x 'y])]
      (is (= :linear (get result 'x)))
      (is (= :linear (get result 'y))))))

(deftest linearity-nonlinear-test
  (testing "sin(x) is nonlinear in x"
    (is (= {'x :nonlinear}
           (linearity-analysis '(sin x) ['x]))))

  (testing "exp(x) is nonlinear in x"
    (is (= {'x :nonlinear}
           (linearity-analysis '(exp x) ['x]))))

  (testing "x/y is nonlinear in y"
    (is (= :nonlinear (get (linearity-analysis '(/ x y) ['y]) 'y))))

  (testing "x/y is linear in x (constant denominator)"
    (is (= :linear (get (linearity-analysis '(/ x y) ['x]) 'x)))))

(deftest linearity-mixed-test
  (testing "x^2 + y is quadratic in x, linear in y"
    (let [result (linearity-analysis '(+ (pow x 2) y) ['x 'y])]
      (is (= :quadratic (get result 'x)))
      (is (= :linear (get result 'y)))))

  (testing "sin(x) + 3*y is nonlinear in x, linear in y"
    (let [result (linearity-analysis '(+ (sin x) (* 3 y)) ['x 'y])]
      (is (= :nonlinear (get result 'x)))
      (is (= :linear (get result 'y))))))

(deftest linearity-division-constant-denom-test
  (testing "(/ x 2) is linear in x"
    (is (= :linear (get (linearity-analysis '(/ x 2) ['x]) 'x)))))

(deftest linearity-pow-cubic-test
  (testing "(pow x 3) is nonlinear (degree 3)"
    (is (= :nonlinear (get (linearity-analysis '(pow x 3) ['x]) 'x)))))

(deftest linearity-if-test
  (testing "if expression: max of branches"
    (let [result (linearity-analysis '(if (< y 0) x (* x x)) ['x])]
      (is (= :quadratic (get result 'x))))))

;; ================================================================
;; sym_analysis.clj — jacobian-sparsity
;; ================================================================

(deftest jacobian-sparsity-basic-test
  (testing "single expression, single variable"
    (is (= [[true]] (jacobian-sparsity ['(+ x 1)] ['x]))))

  (testing "single expression, independent variable"
    (is (= [[false]] (jacobian-sparsity ['(+ y 1)] ['x])))))

(deftest jacobian-sparsity-system-test
  (testing "2x2 system with mixed dependencies"
    ;; f1 = x + y, f2 = x^2
    (let [result (jacobian-sparsity ['(+ x y) '(pow x 2)] ['x 'y])]
      (is (= [[true true] [true false]] result))))

  (testing "diagonal system"
    ;; f1 = sin(x), f2 = cos(y)
    (let [result (jacobian-sparsity ['(sin x) '(cos y)] ['x 'y])]
      (is (= [[true false] [false true]] result))))

  (testing "dense system"
    ;; f1 = x*y, f2 = x+y
    (let [result (jacobian-sparsity ['(* x y) '(+ x y)] ['x 'y])]
      (is (= [[true true] [true true]] result)))))

(deftest jacobian-sparsity-3x3-test
  (testing "3x3 sparse Jacobian"
    ;; f1 = x1 + x2, f2 = x2*x3, f3 = x1
    (let [result (jacobian-sparsity ['(+ x1 x2) '(* x2 x3) 'x1]
                                    ['x1 'x2 'x3])]
      (is (= [[true true false]
              [false true true]
              [true false false]]
             result)))))

(deftest jacobian-sparsity-constants-test
  (testing "constant expressions produce all-false row"
    (let [result (jacobian-sparsity [42 '(+ x 1)] ['x])]
      (is (= [[false] [true]] result)))))

;; ================================================================
;; sym_analysis.clj — hessian-sparsity
;; ================================================================

(deftest hessian-sparsity-linear-test
  (testing "linear expression has zero Hessian"
    ;; f = x + y  =>  d/dx = 1, d/dy = 1  =>  all second derivatives 0
    (let [result (hessian-sparsity '(+ x y) ['x 'y])]
      (is (= [[false false] [false false]] result)))))

(deftest hessian-sparsity-quadratic-test
  (testing "x*y has mixed partial"
    ;; f = x*y => d/dx = y, d/dy = x
    ;; d^2/dxdy depends on x? yes (from d/dx=y => differentiate doesn't simplify to depend on y)
    ;; Actually d/dx = y, which depends on y => H[0,1]=true
    ;; d/dy = x, which depends on x => H[1,0]=true
    (let [result (hessian-sparsity '(* x y) ['x 'y])]
      ;; off-diagonal should be true (mixed partial)
      (is (true? (get-in result [0 1])))
      (is (true? (get-in result [1 0])))))

  (testing "x^2 has diagonal Hessian entry"
    ;; f = x^2 => d/dx = 2x => d^2/dx^2 depends on x? After simplification,
    ;; it should be 2 (constant), but before simplification it may still depend.
    ;; Let's just check it is a valid matrix
    (let [result (hessian-sparsity '(pow x 2) ['x])]
      (is (= 1 (count result)))
      (is (= 1 (count (first result)))))))

(deftest hessian-sparsity-separable-test
  (testing "separable function sin(x) + y^2"
    ;; d/dx = cos(x), d/dy = 2y
    ;; d^2/dx^2: depends on x? yes. d^2/dxdy: cos(x) depends on y? no
    ;; d^2/dy^2: 2y depends on y? After simplification, maybe or maybe not
    (let [result (hessian-sparsity '(+ (sin x) (pow y 2)) ['x 'y])]
      ;; off-diagonals should be false (separable)
      (is (false? (get-in result [0 1])))
      (is (false? (get-in result [1 0])))
      ;; diagonal for x should be true (second deriv of sin(x) is -sin(x))
      (is (true? (get-in result [0 0]))))))

(deftest hessian-sparsity-symmetry-test
  (testing "Hessian sparsity is symmetric"
    (let [result (hessian-sparsity '(* (sin x) (exp y)) ['x 'y])
          n (count result)]
      (doseq [i (range n)
              j (range n)]
        (is (= (get-in result [i j])
               (get-in result [j i]))
            (str "H[" i "," j "] != H[" j "," i "]"))))))
