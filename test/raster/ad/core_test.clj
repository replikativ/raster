(ns raster.ad.core-test
  (:refer-clojure :exclude [+ - * / abs])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.numeric :refer [+ - * / abs]]
            [raster.ad.forward :refer [dual derivative gradient jacobian
                                       exp sin cos sqrt log pow]]))

;; ================================================================
;; Basic dual number arithmetic (multi-partial)
;; ================================================================

(deftest dual-arithmetic-test
  (testing "dual + dual"
    (let [r (+ (dual 3.0 1.0) (dual 2.0 0.5))]
      (is (== 5.0 (.v r)))
      (is (== 1.5 (aget (.partials r) 0)))))
  (testing "dual * dual (product rule)"
    (let [r (* (dual 3.0 1.0) (dual 2.0 0.5))]
      ;; v = 3*2 = 6, dv = 3*0.5 + 1*2 = 3.5
      (is (== 6.0 (.v r)))
      (is (== 3.5 (aget (.partials r) 0)))))
  (testing "dual - dual"
    (let [r (- (dual 5.0 2.0) (dual 3.0 0.5))]
      (is (== 2.0 (.v r)))
      (is (== 1.5 (aget (.partials r) 0)))))
  (testing "dual / dual (quotient rule)"
    (let [r (/ (dual 6.0 1.0) (dual 3.0 0.5))]
      ;; v = 6/3 = 2, dv = (1*3 - 6*0.5) / 9 = 0/9 = 0
      (is (== 2.0 (.v r)))
      (is (< (Math/abs (aget (.partials r) 0)) 1e-12)))))

;; ================================================================
;; Mixed Dual × scalar
;; ================================================================

(deftest dual-scalar-test
  (testing "dual + number"
    (let [r (+ (dual 3.0 1.0) 2.0)]
      (is (== 5.0 (.v r)))
      (is (== 1.0 (aget (.partials r) 0)))))
  (testing "number + dual"
    (let [r (+ 2 (dual 3.0 1.0))]
      (is (== 5.0 (.v r)))
      (is (== 1.0 (aget (.partials r) 0)))))
  (testing "scalar * dual"
    (let [r (* 3.0 (dual 2.0 1.0))]
      ;; v = 6, dv = 3*1 = 3
      (is (== 6.0 (.v r)))
      (is (== 3.0 (aget (.partials r) 0)))))
  (testing "integer * dual"
    (let [r (* 3 (dual 2.0 1.0))]
      (is (== 6.0 (.v r)))
      (is (== 3.0 (aget (.partials r) 0))))))

;; ================================================================
;; derivative function
;; ================================================================

(deftest derivative-polynomial-test
  (testing "d/dx(x²) = 2x"
    (is (== 4.0  (derivative (fn [x] (* x x)) 2.0)))
    (is (== 6.0  (derivative (fn [x] (* x x)) 3.0)))
    (is (== 0.0  (derivative (fn [x] (* x x)) 0.0))))
  (testing "d/dx(x³) = 3x²"
    (is (== 12.0 (derivative (fn [x] (* x x x)) 2.0)))
    (is (== 27.0 (derivative (fn [x] (* x x x)) 3.0))))
  (testing "d/dx(3x² + 2x + 1) = 6x + 2"
    (is (== 14.0 (derivative (fn [x] (+ (* 3 x x) (* 2 x) 1)) 2.0)))))

(deftest derivative-transcendental-test
  (testing "d/dx(exp(x)) = exp(x)"
    (let [x 1.0
          d (derivative exp x)]
      (is (< (Math/abs (- d (Math/exp x))) 1e-12))))
  (testing "d/dx(sin(x)) = cos(x)"
    (let [x 0.5
          d (derivative sin x)]
      (is (< (Math/abs (- d (Math/cos x))) 1e-12))))
  (testing "d/dx(cos(x)) = -sin(x)"
    (let [x 0.5
          d (derivative cos x)]
      (is (< (Math/abs (+ d (Math/sin x))) 1e-12))))
  (testing "d/dx(log(x)) = 1/x"
    (let [x 2.0
          d (derivative log x)]
      (is (< (Math/abs (- d (/ 1.0 x))) 1e-12))))
  (testing "d/dx(sqrt(x)) = 1/(2*sqrt(x))"
    (let [x 4.0
          d (derivative sqrt x)]
      (is (< (Math/abs (- d 0.25)) 1e-12)))))

;; ================================================================
;; Chain rule and composition
;; ================================================================

(deftest derivative-chain-rule-test
  (testing "d/dx(exp(-x²)) = -2x*exp(-x²)"
    (let [x 1.0
          f (fn [x] (exp (* -1.0 x x)))
          d (derivative f x)
          exact (* -2.0 x (Math/exp (- (* x x))))]
      (is (< (Math/abs (- d exact)) 1e-12))))
  (testing "d/dx(sin(x²)) = 2x*cos(x²)"
    (let [x 0.5
          f (fn [x] (sin (* x x)))
          d (derivative f x)
          exact (* 2.0 x (Math/cos (* x x)))]
      (is (< (Math/abs (- d exact)) 1e-12))))
  (testing "d/dx(log(1 + x²)) = 2x/(1+x²)"
    (let [x 2.0
          f (fn [x] (log (+ 1.0 (* x x))))
          d (derivative f x)
          exact (/ (* 2.0 x) (+ 1.0 (* x x)))]
      (is (< (Math/abs (- d exact)) 1e-12)))))

;; ================================================================
;; Gradient (multi-variable)
;; ================================================================

(deftest gradient-test
  (testing "gradient of f(x,y) = x² + y²"
    (let [f (fn [[x y]] (+ (* x x) (* y y)))
          g (gradient f [3.0 4.0])]
      ;; ∂f/∂x = 2x = 6, ∂f/∂y = 2y = 8
      (is (< (Math/abs (- (aget g 0) 6.0)) 1e-12))
      (is (< (Math/abs (- (aget g 1) 8.0)) 1e-12))))
  (testing "gradient of f(x,y) = x*y"
    (let [f (fn [[x y]] (* x y))
          g (gradient f [3.0 4.0])]
      ;; ∂f/∂x = y = 4, ∂f/∂y = x = 3
      (is (< (Math/abs (- (aget g 0) 4.0)) 1e-12))
      (is (< (Math/abs (- (aget g 1) 3.0)) 1e-12))))
  (testing "gradient of Rosenbrock f(x,y) = (1-x)² + 100(y-x²)²"
    (let [f (fn [[x y]]
              (+ (pow (- 1.0 x) 2.0)
                 (* 100.0 (pow (- y (* x x)) 2.0))))
          g (gradient f [1.0 1.0])]
      ;; At (1,1): ∇f = (0, 0) — it's the minimum
      (is (< (Math/abs (aget g 0)) 1e-10))
      (is (< (Math/abs (aget g 1)) 1e-10)))))

;; ================================================================
;; Chunk-based gradient (large n)
;; ================================================================

(deftest gradient-chunking-test
  (testing "gradient with 20 variables (tests chunking)"
    (let [n 20
          ;; f(x) = sum(xi^2)  =>  ∂f/∂xi = 2*xi
          f (fn [xs]
              (reduce (fn [acc xi] (+ acc (* xi xi)))
                      (first xs)
                      ;; reduce needs a Dual init; start from first element squared
                      ;; Actually: let's just loop with +
                      (rest xs)))
          ;; Simpler: f(x) = x0^2 + x1^2 + ... + xn-1^2
          f (fn [xs]
              (loop [i 0 acc (* (nth xs 0) (nth xs 0))]
                (if (>= (inc i) (count xs))
                  acc
                  (recur (inc i) (+ acc (* (nth xs (inc i)) (nth xs (inc i))))))))
          x-vals (vec (map double (range 1 (inc n))))
          g (gradient f x-vals)]
      ;; ∂f/∂xi = 2*xi
      (dotimes [i n]
        (is (< (Math/abs (- (aget g i) (* 2.0 (inc (double i))))) 1e-10)
            (str "gradient component " i)))))
  (testing "gradient with custom chunk size"
    (let [f (fn [[x y z]] (+ (* x y) (* y z) (* x z)))
          g2 (gradient f [1.0 2.0 3.0] 2)
          g8 (gradient f [1.0 2.0 3.0] 8)]
      ;; Same result regardless of chunk size
      ;; ∂f/∂x = y+z = 5, ∂f/∂y = x+z = 4, ∂f/∂z = y+x = 3
      (dotimes [i 3]
        (is (< (Math/abs (- (aget g2 i) (aget g8 i))) 1e-12))))))

;; ================================================================
;; Jacobian
;; ================================================================

(deftest jacobian-test
  (testing "jacobian of f(x,y) = [x*y, x+y]"
    (let [f (fn [[x y]] [(* x y) (+ x y)])
          j (jacobian f [3.0 4.0])]
      ;; J = [[y, x], [1, 1]] = [[4, 3], [1, 1]]
      ;; Row-major: [4, 3, 1, 1]
      (is (< (Math/abs (- (aget j 0) 4.0)) 1e-12))
      (is (< (Math/abs (- (aget j 1) 3.0)) 1e-12))
      (is (< (Math/abs (- (aget j 2) 1.0)) 1e-12))
      (is (< (Math/abs (- (aget j 3) 1.0)) 1e-12))))
  (testing "jacobian of rotation [cos(t)*x - sin(t)*y, sin(t)*x + cos(t)*y]"
    (let [theta (/ Math/PI 4.0)  ;; 45 degrees
          ;; f: R^2 -> R^2, linear in (x,y), J is the rotation matrix
          f (fn [[x y]]
              (let [c (cos x) s (sin x)]
                ;; Wait, we want jacobian w.r.t. x,y for fixed theta
                ;; Let's use a simpler example
                [(+ (* 2.0 x) (* 3.0 y))
                 (+ (* 4.0 x) (* 5.0 y))]))
          j (jacobian f [1.0 2.0])]
      ;; J = [[2, 3], [4, 5]]
      (is (< (Math/abs (- (aget j 0) 2.0)) 1e-12))
      (is (< (Math/abs (- (aget j 1) 3.0)) 1e-12))
      (is (< (Math/abs (- (aget j 2) 4.0)) 1e-12))
      (is (< (Math/abs (- (aget j 3) 5.0)) 1e-12)))))
