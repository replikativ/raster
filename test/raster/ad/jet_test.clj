(ns raster.ad.jet-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ad.jet :as jet]
            [raster.numeric :as n]
            [raster.math :as math]
            [raster.ad.forward :as ad]))

(defn- approx=
  "Approximate equality with tolerance."
  ([a b] (approx= a b 1e-8))
  ([a b tol] (< (Math/abs (- (double a) (double b))) tol)))

;; ================================================================
;; Jet arithmetic
;; ================================================================

(deftest jet-construction
  (testing "identity jet"
    (let [j (jet/jet 3.0 2)]
      (is (approx= 3.0 (jet/primal j)))
      (is (approx= 3.0 (jet/taylor-derivative j 0)))
      (is (approx= 1.0 (jet/taylor-derivative j 1)))
      (is (approx= 0.0 (jet/taylor-derivative j 2)))))

  (testing "constant jet"
    (let [j (jet/jet-const 5.0 3)]
      (is (approx= 5.0 (jet/primal j)))
      (is (approx= 0.0 (jet/taylor-derivative j 1)))
      (is (approx= 0.0 (jet/taylor-derivative j 2)))
      (is (approx= 0.0 (jet/taylor-derivative j 3))))))

(deftest jet-addition
  (testing "jet + jet"
    (let [a (jet/jet 2.0 2)
          b (jet/jet 3.0 2)
          c (n/+ a b)]
      (is (approx= 5.0 (jet/primal c)))
      (is (approx= 2.0 (jet/taylor-derivative c 1)))))

  (testing "jet + number"
    (let [a (jet/jet 2.0 2)
          c (n/+ a 3.0)]
      (is (approx= 5.0 (jet/primal c)))
      (is (approx= 1.0 (jet/taylor-derivative c 1))))))

(deftest jet-multiplication
  (testing "truncated Cauchy product: (a+x)*(b+x) = ab + (a+b)x + x²"
    ;; jet(a,1) represents (a + x), jet(b,1) represents (b + x)
    ;; Product: ab + (a+b)x + x²
    ;; Scaled coeffs: [ab, (a+b)/1!, 1/1!] but we just test derivatives
    (let [a (jet/jet 2.0 2)  ; 2+x → coeffs [2, 1, 0]
          b (jet/jet 3.0 2)  ; 3+x → coeffs [3, 1, 0]
          c (n/* a b)]       ; → coeffs [6, 5, 1]
      (is (approx= 6.0 (jet/primal c)))          ; 2*3
      (is (approx= 5.0 (jet/taylor-derivative c 1)))  ; 2+3
      (is (approx= 2.0 (jet/taylor-derivative c 2))))) ; 2*1! = 2 (coeff[2]*2!)

  (testing "jet * scalar"
    (let [a (jet/jet 3.0 1)
          c (n/* a 2.0)]
      (is (approx= 6.0 (jet/primal c)))
      (is (approx= 2.0 (jet/taylor-derivative c 1))))))

(deftest jet-division
  (testing "jet / jet: 1/x at x=2"
    ;; f(x)=1/x, f'=-1/x², f''=2/x³
    (let [one-j (jet/jet-const 1.0 2)
          x-j (jet/jet 2.0 2)
          c (n// one-j x-j)]
      (is (approx= 0.5 (jet/primal c)))
      (is (approx= -0.25 (jet/taylor-derivative c 1)))
      (is (approx= 0.25 (jet/taylor-derivative c 2))))))

;; ================================================================
;; Math function recurrences
;; ================================================================

(deftest jet-exp
  (testing "exp derivatives at x=0"
    ;; exp(0)=1, exp'(0)=1, exp''(0)=1, ...
    (let [derivs (jet/higher-derivatives math/exp 0.0 4)]
      (dotimes [i 5]
        (is (approx= 1.0 (aget derivs i)) (str "exp^(" i ")(0) should be 1"))))))

(deftest jet-log
  (testing "log derivatives at x=1"
    ;; log(1)=0, log'(1)=1, log''(1)=-1, log'''(1)=2, log^(4)(1)=-6
    (let [derivs (jet/higher-derivatives math/log 1.0 4)]
      (is (approx= 0.0 (aget derivs 0)))
      (is (approx= 1.0 (aget derivs 1)))
      (is (approx= -1.0 (aget derivs 2)))
      (is (approx= 2.0 (aget derivs 3)))
      (is (approx= -6.0 (aget derivs 4))))))

(deftest jet-sin
  (testing "sin derivatives at x=0"
    ;; sin(0)=0, cos(0)=1, -sin(0)=0, -cos(0)=-1
    (let [derivs (jet/higher-derivatives math/sin 0.0 4)]
      (is (approx= 0.0 (aget derivs 0)))
      (is (approx= 1.0 (aget derivs 1)))
      (is (approx= 0.0 (aget derivs 2)))
      (is (approx= -1.0 (aget derivs 3)))
      (is (approx= 0.0 (aget derivs 4))))))

(deftest jet-cos
  (testing "cos derivatives at x=0"
    ;; cos(0)=1, -sin(0)=0, -cos(0)=-1, sin(0)=0
    (let [derivs (jet/higher-derivatives math/cos 0.0 4)]
      (is (approx= 1.0 (aget derivs 0)))
      (is (approx= 0.0 (aget derivs 1)))
      (is (approx= -1.0 (aget derivs 2)))
      (is (approx= 0.0 (aget derivs 3)))
      (is (approx= 1.0 (aget derivs 4))))))

(deftest jet-sqrt
  (testing "sqrt derivatives at x=4"
    ;; sqrt(4)=2, 1/(2*2)=0.25, -(1/(4*8))=-1/32
    (let [derivs (jet/higher-derivatives n/sqrt 4.0 2)]
      (is (approx= 2.0 (aget derivs 0)))
      (is (approx= 0.25 (aget derivs 1)))
      (is (approx= -0.03125 (aget derivs 2))))))

;; ================================================================
;; Composition tests — Faà di Bruno
;; ================================================================

(deftest jet-composition
  (testing "sin(x²) derivatives at x=1 match analytic"
    (let [f (fn [x] (math/sin (n/* x x)))
          jet-derivs (jet/higher-derivatives f 1.0 3)
          ;; Analytic: f(x)=sin(x²)
          ;; f'(x)=2x*cos(x²), f'(1)=2*cos(1)
          ;; f''(x)=2*cos(x²)-4x²*sin(x²), f''(1)=2*cos(1)-4*sin(1)
          ;; f'''(x)=-12x*sin(x²)-8x³*cos(x²), f'''(1)=-12*sin(1)-8*cos(1)
          d1-exact (clojure.core/* 2.0 (Math/cos 1.0))
          d2-exact (clojure.core/- (clojure.core/* 2.0 (Math/cos 1.0))
                                   (clojure.core/* 4.0 (Math/sin 1.0)))
          d3-exact (clojure.core/- (clojure.core/* -12.0 (Math/sin 1.0))
                                   (clojure.core/* 8.0 (Math/cos 1.0)))]
      (is (approx= (Math/sin 1.0) (aget jet-derivs 0)))
      (is (approx= d1-exact (aget jet-derivs 1) 1e-10))
      (is (approx= d2-exact (aget jet-derivs 2) 1e-10))
      (is (approx= d3-exact (aget jet-derivs 3) 1e-10)))))

(deftest jet-polynomial
  (testing "x³ derivatives are exact"
    (let [f (fn [x] (n/* x (n/* x x)))
          derivs (jet/higher-derivatives f 2.0 4)]
      (is (approx= 8.0 (aget derivs 0)))   ; 2³
      (is (approx= 12.0 (aget derivs 1)))  ; 3*2²
      (is (approx= 12.0 (aget derivs 2)))  ; 6*2
      (is (approx= 6.0 (aget derivs 3)))   ; 6
      (is (approx= 0.0 (aget derivs 4))))))  ; 0

(deftest jet-exp-sin-composition
  (testing "exp(sin(x)) derivatives at x=0"
    (let [f (fn [x] (math/exp (math/sin x)))
          derivs (jet/higher-derivatives f 0.0 3)]
      ;; exp(sin(0)) = exp(0) = 1
      (is (approx= 1.0 (aget derivs 0)))
      ;; d/dx exp(sin(x)) = cos(x)*exp(sin(x)), at 0 = 1*1 = 1
      (is (approx= 1.0 (aget derivs 1)))
      ;; d²/dx² = (cos²(x) - sin(x))*exp(sin(x)), at 0 = (1-0)*1 = 1
      (is (approx= 1.0 (aget derivs 2))))))

;; ================================================================
;; Promotion tests
;; ================================================================

(deftest jet-promotion
  (testing "Number + Jet promotion"
    (let [j (jet/jet 2.0 1)
          r (n/+ 3 j)]
      (is (instance? raster.ad.jet.Jet r))
      (is (approx= 5.0 (jet/primal r)))))

  (testing "Jet + Number promotion"
    (let [j (jet/jet 2.0 1)
          r (n/+ j 3)]
      (is (instance? raster.ad.jet.Jet r))
      (is (approx= 5.0 (jet/primal r))))))

;; ================================================================
;; jet-derivative convenience
;; ================================================================

(deftest jet-derivative-fn
  (testing "k-th derivative via jet-derivative"
    (let [f (fn [x] (n/* x (n/* x x)))]
      (is (approx= 12.0 (jet/jet-derivative f 2.0 1)))  ; 3x² at 2
      (is (approx= 12.0 (jet/jet-derivative f 2.0 2)))  ; 6x at 2
      (is (approx= 6.0 (jet/jet-derivative f 2.0 3)))   ; 6
      (is (approx= 0.0 (jet/jet-derivative f 2.0 4)))))) ; 0
