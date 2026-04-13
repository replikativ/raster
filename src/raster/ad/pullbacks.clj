(ns raster.ad.pullbacks
  "Runtime pullback factories for reverse-mode AD.

  Each pullback factory takes the forward result and original inputs, and returns
  a function (the 'pullback') that maps output adjoints to input adjoints.

  Pullback arithmetic uses raster.numeric dispatch so that Dual numbers
  flow through for forward-over-reverse composition (Hessians, etc.).

  All pullbacks are registered into the unified template registry
  (raster.ad.templates) under the :pullback-factory key.

  The compiler uses :grads/:grads-fn templates for code emission.
  Runtime value+grad uses :pullback-factory (pre-built closure)."
  (:refer-clojure :exclude [+ - * / abs])
  (:require [raster.numeric :as n :refer [+ - * /]]
            [raster.math :as math]
            [raster.sci.special]
            [raster.ad.templates :as tmpl]
            [raster.linalg.core :as la]))

;; ================================================================
;; Arithmetic pullbacks
;; ================================================================

;; Addition: d/dx(x+y) = dy, d/dy(x+y) = dy
(tmpl/merge-into-template! '+
                           {:pullback-factory (fn [_result x y] (fn [dy] [dy dy]))})

(tmpl/merge-into-template! 'clojure.core/+
                           {:pullback-factory (fn [_result x y] (fn [dy] [dy dy]))})

;; grad-acc is nil-safe +, same derivative: d/da = dy, d/db = dy
(tmpl/merge-into-template! 'raster.ad.reverse/grad-acc
                           {:pullback-factory (fn [_result a b] (fn [dy] [dy dy]))})

;; Subtraction: d/dx(x-y) = dy, d/dy(x-y) = -dy
(tmpl/merge-into-template! '-
                           {:pullback-factory (fn [_result & args]
                                                (if (= 1 (count args))
                         ;; Unary negation
                                                  (fn [dy] [(- dy)])
                         ;; Binary subtraction
                                                  (fn [dy] [dy (- dy)])))})

(tmpl/merge-into-template! 'clojure.core/-
                           {:pullback-factory (fn [_result & args]
                                                (if (= 1 (count args))
                                                  (fn [dy] [(- dy)])
                                                  (fn [dy] [dy (- dy)])))})

;; Multiplication: d/dx(x*y) = y*dy, d/dy(x*y) = x*dy
(tmpl/merge-into-template! '*
                           {:pullback-factory (fn [_result x y] (fn [dy] [(* dy y) (* dy x)]))})

(tmpl/merge-into-template! 'clojure.core/*
                           {:pullback-factory (fn [_result x y] (fn [dy] [(* dy y) (* dy x)]))})

;; Division: d/dx(x/y) = dy/y, d/dy(x/y) = -x*dy/y²
(tmpl/merge-into-template! '/
                           {:pullback-factory (fn [_result x y]
                                                (fn [dy]
                                                  [(/ dy y)
                                                   (- (/ (* dy x) (* y y)))]))})

(tmpl/merge-into-template! 'clojure.core//
                           {:pullback-factory (fn [_result x y]
                                                (fn [dy]
                                                  [(/ dy y)
                                                   (- (/ (* dy x) (* y y)))]))})

;; ================================================================
;; Math function pullbacks
;; ================================================================

;; sin: d/dx sin(x) = cos(x) * dy
(tmpl/merge-into-template! 'Math/sin
                           {:pullback-factory (fn [_result x] (fn [dy] [(* dy (math/cos x))]))})

;; cos: d/dx cos(x) = -sin(x) * dy
(tmpl/merge-into-template! 'Math/cos
                           {:pullback-factory (fn [_result x] (fn [dy] [(* dy (- (math/sin x)))]))})

;; exp: d/dx exp(x) = exp(x) * dy
(tmpl/merge-into-template! 'Math/exp
                           {:pullback-factory (fn [result x] (fn [dy] [(* dy result)]))})

;; log: d/dx log(x) = dy/x
(tmpl/merge-into-template! 'Math/log
                           {:pullback-factory (fn [_result x] (fn [dy] [(/ dy x)]))})

;; sqrt: d/dx sqrt(x) = dy / (2*sqrt(x))
(tmpl/merge-into-template! 'Math/sqrt
                           {:pullback-factory (fn [result x] (fn [dy] [(/ dy (* 2.0 result))]))})

;; pow: d/dx x^n = n*x^(n-1) * dy, d/dn x^n = x^n*log(x) * dy
(tmpl/merge-into-template! 'Math/pow
                           {:pullback-factory (fn [result x n]
                                                (fn [dy]
                                                  [(* dy (* n (n/pow x (- n 1.0))))
                                                   (* dy (* result (math/log x)))]))})

;; abs: d/dx |x| = sign(x) * dy
(tmpl/merge-into-template! 'Math/abs
                           {:pullback-factory (fn [_result x] (fn [dy] [(* dy (math/signum (double x)))]))})

;; tan: d/dx tan(x) = (1 + tan²(x)) * dy = dy/cos²(x)
(tmpl/merge-into-template! 'Math/tan
                           {:pullback-factory (fn [result x] (fn [dy] [(* dy (+ 1.0 (* result result)))]))})

;; asin: d/dx asin(x) = dy / sqrt(1 - x²)
(tmpl/merge-into-template! 'Math/asin
                           {:pullback-factory (fn [_result x] (fn [dy] [(/ dy (n/sqrt (- 1.0 (* x x))))]))})

;; acos: d/dx acos(x) = -dy / sqrt(1 - x²)
(tmpl/merge-into-template! 'Math/acos
                           {:pullback-factory (fn [_result x] (fn [dy] [(- (/ dy (n/sqrt (- 1.0 (* x x)))))]))})

;; atan: d/dx atan(x) = dy / (1 + x²)
(tmpl/merge-into-template! 'Math/atan
                           {:pullback-factory (fn [_result x] (fn [dy] [(/ dy (+ 1.0 (* x x)))]))})

;; atan2: d/dx atan2(y,x) = x/(x²+y²)*dy, d/dy atan2(y,x) = -y/(x²+y²)*dy
;; Note: atan2(y,x) convention — first arg is y, second is x
(tmpl/merge-into-template! 'Math/atan2
                           {:pullback-factory (fn [_result y x]
                                                (let [denom (+ (* x x) (* y y))]
                                                  (fn [dy]
                                                    [(/ (* dy x) denom)
                                                     (- (/ (* dy y) denom))])))})

;; min: d/dx min(x,y) = dy if x <= y, 0 otherwise (subgradient)
(tmpl/merge-into-template! 'Math/min
                           {:pullback-factory (fn [_result x y]
                                                (fn [dy]
                                                  (if (<= (double x) (double y))
                                                    [dy 0.0]
                                                    [0.0 dy])))})

;; max: d/dx max(x,y) = dy if x >= y, 0 otherwise (subgradient)
(tmpl/merge-into-template! 'Math/max
                           {:pullback-factory (fn [_result x y]
                                                (fn [dy]
                                                  (if (>= (double x) (double y))
                                                    [dy 0.0]
                                                    [0.0 dy])))})

;; fma: d/dx fma(a,b,c) = b*dy, d/db fma(a,b,c) = a*dy, d/dc fma(a,b,c) = dy
(tmpl/merge-into-template! 'Math/fma
                           {:pullback-factory (fn [_result a b c] (fn [dy] [(* dy b) (* dy a) dy]))})

;; ================================================================
;; Numeric cast pullbacks (identity passthrough)
;; ================================================================
;; The walker inserts (double x), (float x) casts for type safety.
;; These are differentiable identity operations.

(tmpl/merge-into-template! 'double
                           {:pullback-factory (fn [_result x] (fn [dy] [dy]))})

(tmpl/merge-into-template! 'clojure.core/double
                           {:pullback-factory (fn [_result x] (fn [dy] [dy]))})

(tmpl/merge-into-template! 'float
                           {:pullback-factory (fn [_result x] (fn [dy] [dy]))})

(tmpl/merge-into-template! 'clojure.core/float
                           {:pullback-factory (fn [_result x] (fn [dy] [dy]))})

;; Integer casts: zero gradient (non-differentiable)
(tmpl/merge-into-template! 'long
                           {:pullback-factory (fn [_result x] (fn [dy] [0.0]))})

(tmpl/merge-into-template! 'clojure.core/long
                           {:pullback-factory (fn [_result x] (fn [dy] [0.0]))})

(tmpl/merge-into-template! 'int
                           {:pullback-factory (fn [_result x] (fn [dy] [0.0]))})

(tmpl/merge-into-template! 'clojure.core/int
                           {:pullback-factory (fn [_result x] (fn [dy] [0.0]))})

;; ================================================================
;; Special function pullbacks
;; ================================================================

;; lgamma: d/dx lgamma(x) = digamma(x)
(tmpl/merge-into-template! 'raster.sci.special/lgamma
                           {:pullback-factory (fn [_result x]
                                                (fn [dy] [(* dy (raster.sci.special/digamma x))]))})

;; digamma: d/dx digamma(x) = trigamma(x)
(tmpl/merge-into-template! 'raster.sci.special/digamma
                           {:pullback-factory (fn [_result x]
                                                (fn [dy] [(* dy (raster.sci.special/trigamma x))]))})

;; erf: d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
(tmpl/merge-into-template! 'raster.sci.special/erf
                           {:pullback-factory (fn [_result x]
                                                (fn [dy]
                                                  [(* dy (* (/ 2.0 (n/sqrt n/pi))
                                                            (math/exp (- (* x x)))))]))})

;; erfinv: d/dx erfinv(y) = sqrt(pi)/2 * exp(erfinv(y)^2)
(tmpl/merge-into-template! 'raster.sci.special/erfinv
                           {:pullback-factory (fn [result y]
                                                (fn [dy]
                                                  [(* dy (* (/ (n/sqrt n/pi) 2.0)
                                                            (math/exp (* result result))))]))})

;; betainc: partial derivatives w.r.t. x only (a,b treated as constants)
;; d/dx I_x(a,b) = x^(a-1) * (1-x)^(b-1) / B(a,b)
(tmpl/merge-into-template! 'raster.sci.special/betainc
                           {:pullback-factory (fn [_result a b x]
                                                (fn [dy]
                                                  [0.0 0.0
                                                   (* dy (/ (* (n/pow x (- a 1.0))
                                                               (n/pow (- 1.0 x) (- b 1.0)))
                                                            (raster.sci.special/beta a b)))]))})

;; besselj0: d/dx J0(x) = -J1(x)
(tmpl/merge-into-template! 'raster.sci.special/besselj0
                           {:pullback-factory (fn [_result x]
                                                (fn [dy] [(* dy (- (raster.sci.special/besselj1 x)))]))})

;; besselj1: d/dx J1(x) = J0(x) - J1(x)/x
(tmpl/merge-into-template! 'raster.sci.special/besselj1
                           {:pullback-factory (fn [_result x]
                                                (fn [dy]
                                                  [(* dy (- (raster.sci.special/besselj0 x)
                                                            (/ (raster.sci.special/besselj1 x) x)))]))})

;; expit (sigmoid): d/dx sigma(x) = sigma(x) * (1 - sigma(x))
(tmpl/merge-into-template! 'raster.sci.special/expit
                           {:pullback-factory (fn [result x]
                                                (fn [dy] [(* dy (* result (- 1.0 result)))]))})

;; logit: d/dx logit(x) = 1/(x*(1-x))
(tmpl/merge-into-template! 'raster.sci.special/logit
                           {:pullback-factory (fn [_result x]
                                                (fn [dy] [(* dy (/ 1.0 (* x (- 1.0 x))))]))})

;; log1pexp: d/dx log(1+exp(x)) = sigmoid(x)
(tmpl/merge-into-template! 'raster.sci.special/log1pexp
                           {:pullback-factory (fn [_result x]
                                                (fn [dy] [(* dy (/ 1.0 (+ 1.0 (math/exp (- x)))))]))})

;; ================================================================
;; Linear algebra pullbacks
;; ================================================================

;; solve: Ax = b => db = A^{-T} dy, dA = -db x^T
(tmpl/merge-into-template! 'raster.linalg.core/solve
                           {:pullback-factory (fn [result A b n]
                                                (let [solve-fn la/solve]
                                                  (fn [dy]
                                                    (let [x result
                                                          At (double-array (* n n))
                                                          _ (dotimes [i n]
                                                              (dotimes [j n]
                                                                (aset At (+ (* i n) j) (aget A (+ (* j n) i)))))
                                                          db (solve-fn At dy n)
                                                          dA (double-array (* n n))]
                                                      (dotimes [i n]
                                                        (dotimes [j n]
                                                          (aset dA (+ (* i n) j) (- (* (aget db i) (aget x j))))))
                                                      [dA db]))))})

;; det: d/dA det(A) = det(A) * (A^{-1})^T
(tmpl/merge-into-template! 'raster.linalg.core/array-det
                           {:pullback-factory (fn [result A n]
                                                (let [inv-fn la/array-inv]
                                                  (fn [dy]
                                                    (let [Ainv (inv-fn A n)
                                                          dA (double-array (* n n))]
                                                      (dotimes [i n]
                                                        (dotimes [j n]
                                                          (aset dA (+ (* i n) j)
                                                                (* dy result (aget Ainv (+ (* j n) i))))))
                                                      [dA]))))})
