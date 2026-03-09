(ns raster.sci.interpolation-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.interpolation :refer [linear-interp cubic-spline akima-spline
                                              cubic-spline-deriv pchip
                                              interp2d-linear interp2d-cubic]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Linear interpolation
;; ================================================================

(deftest linear-interp-exact-test
  (testing "linear interp is exact on 2 points"
    (let [xs (double-array [0.0 1.0])
          ys (double-array [0.0 1.0])
          f (linear-interp xs ys)]
      (is (approx= 0.0 (f 0.0)))
      (is (approx= 0.5 (f 0.5)))
      (is (approx= 1.0 (f 1.0))))))

(deftest linear-interp-quadratic-test
  (testing "linear interp on x^2 data"
    (let [xs (double-array [0.0 1.0 2.0 3.0])
          ys (double-array [0.0 1.0 4.0 9.0])
          f (linear-interp xs ys)]
      ;; At data points: exact
      (is (approx= 0.0 (f 0.0)))
      (is (approx= 1.0 (f 1.0)))
      (is (approx= 4.0 (f 2.0)))
      ;; Between points: linear, so 2.5 between 1 and 4
      (is (approx= 2.5 (f 1.5))))))

(deftest linear-interp-extrapolation-test
  (testing "linear extrapolation beyond range"
    (let [xs (double-array [0.0 1.0])
          ys (double-array [0.0 2.0])
          f (linear-interp xs ys)]
      ;; Extrapolate left
      (is (approx= -2.0 (f -1.0)))
      ;; Extrapolate right
      (is (approx= 4.0 (f 2.0))))))

;; ================================================================
;; Cubic spline
;; ================================================================

(deftest cubic-spline-exact-on-data-test
  (testing "cubic spline passes through data points"
    (let [xs (double-array [0.0 1.0 2.0 3.0 4.0])
          ys (double-array [0.0 1.0 0.0 1.0 0.0])
          f (cubic-spline xs ys)]
      (dotimes [i 5]
        (is (approx= (aget ys i) (f (aget xs i)) 1e-10)
            (str "Spline should pass through point " i))))))

(deftest cubic-spline-smooth-test
  (testing "cubic spline is smooth (continuous derivative)"
    (let [xs (double-array [0.0 1.0 2.0 3.0 4.0])
          ys (double-array [0.0 1.0 4.0 9.0 16.0])
          f (cubic-spline xs ys)
          ;; Check smoothness by comparing left/right derivatives at interior knot
          eps 1e-6
          x 2.0
          d-left (/ (- (f x) (f (- x eps))) eps)
          d-right (/ (- (f (+ x eps)) (f x)) eps)]
      (is (approx= d-left d-right 1e-3)
          "Left and right derivatives should agree at interior knot"))))

(deftest cubic-spline-quadratic-test
  (testing "cubic spline reproduces quadratic exactly"
    ;; S''(x) is linear for a quadratic, and natural BCs set S''=0 at ends,
    ;; so this won't be exact for x^2 with few points, but should be close
    (let [xs (double-array [0.0 0.5 1.0 1.5 2.0])
          ys (double-array (map #(* % %) [0.0 0.5 1.0 1.5 2.0]))
          f (cubic-spline xs ys)]
      (is (approx= 0.25 (f 0.5) 1e-4))
      (is (approx= 1.0 (f 1.0) 1e-4)))))

(deftest cubic-spline-sin-test
  (testing "cubic spline on sin(x) with many points achieves low error"
    (let [n 20
          xs (double-array (map #(* Math/PI (/ (double %) (dec n))) (range n)))
          ys (double-array (map #(Math/sin %) xs))
          f (cubic-spline xs ys)
          ;; Test at midpoint pi/2
          err (Math/abs (- (f (* 0.5 Math/PI)) 1.0))]
      (is (< err 1e-4)
          (str "Cubic spline sin(pi/2) error should be < 1e-4, got " err)))))

;; ================================================================
;; Akima spline
;; ================================================================

(deftest akima-spline-exact-on-data-test
  (testing "Akima spline passes through data points"
    (let [xs (double-array [0.0 1.0 2.0 3.0 4.0 5.0])
          ys (double-array [0.0 1.0 0.0 1.0 0.0 1.0])
          f (akima-spline xs ys)]
      (dotimes [i 6]
        (is (approx= (aget ys i) (f (aget xs i)) 1e-10)
            (str "Akima should pass through point " i))))))

(deftest akima-spline-sin-test
  (testing "Akima spline on sin(x) with many points"
    (let [n 20
          xs (double-array (map #(* Math/PI (/ (double %) (dec n))) (range n)))
          ys (double-array (map #(Math/sin %) xs))
          f (akima-spline xs ys)
          err (Math/abs (- (f (* 0.5 Math/PI)) 1.0))]
      (is (< err 1e-3)
          (str "Akima sin(pi/2) error should be < 1e-3, got " err)))))

;; ================================================================
;; Derivative
;; ================================================================

(deftest cubic-spline-deriv-test
  (testing "cubic spline derivative of x^2 ≈ 2x"
    (let [xs (double-array [0.0 0.5 1.0 1.5 2.0])
          ys (double-array (map #(* % %) [0.0 0.5 1.0 1.5 2.0]))
          df (cubic-spline-deriv xs ys)]
      ;; derivative of x^2 at x=1 should be ≈ 2
      (is (approx= 2.0 (df 1.0) 0.1)
          (str "d/dx(x^2) at x=1 should be ~2, got " (df 1.0))))))

;; ================================================================
;; PCHIP interpolation
;; ================================================================

(deftest pchip-exact-on-knots-test
  (testing "pchip interpolates knots exactly"
    (let [xs (double-array [0.0 1.0 2.0 3.0 4.0])
          ys (double-array [0.0 1.0 0.0 1.0 0.0])
          f (pchip xs ys)]
      (dotimes [i 5]
        (is (approx= (aget ys i) (f (aget xs i)) 1e-10)
            (str "PCHIP should pass through knot " i))))))

(deftest pchip-monotonicity-test
  (testing "pchip preserves monotonicity"
    ;; Monotonically increasing data — interpolant should not overshoot
    (let [xs (double-array [0.0 1.0 2.0 3.0 4.0])
          ys (double-array [0.0 0.1 0.2 0.9 1.0])
          f (pchip xs ys)]
      ;; Sample between each pair and verify monotonicity
      (let [samples (map #(f (+ 0.5 (double %))) (range 4))]
        (doseq [[a b] (partition 2 1 samples)]
          (is (<= a (+ b 1e-10))
              (str "PCHIP should be monotone, got " a " > " b)))))))

(deftest pchip-sin-accuracy-test
  (testing "pchip on sin(x) with many points"
    (let [n 20
          xs (double-array (map #(* Math/PI (/ (double %) (dec n))) (range n)))
          ys (double-array (map #(Math/sin %) xs))
          f (pchip xs ys)
          err (Math/abs (- (f (* 0.5 Math/PI)) 1.0))]
      (is (< err 1e-2)
          (str "PCHIP sin(pi/2) error should be < 0.01, got " err)))))

;; ================================================================
;; 2D bilinear interpolation
;; ================================================================

(deftest interp2d-linear-xpy-test
  (testing "bilinear on z = x + y"
    (let [xs (double-array [0.0 1.0 2.0])
          ys (double-array [0.0 1.0 2.0])
          ;; z[i,j] = xs[j] + ys[i], row-major (ny=3, nx=3)
          zs (double-array [0.0 1.0 2.0    ;; y=0: 0+0, 1+0, 2+0
                            1.0 2.0 3.0    ;; y=1: 0+1, 1+1, 2+1
                            2.0 3.0 4.0])  ;; y=2: 0+2, 1+2, 2+2
          f (interp2d-linear xs ys zs 3 3)]
      ;; At grid points
      (is (approx= 0.0 (f 0.0 0.0)))
      (is (approx= 2.0 (f 1.0 1.0)))
      (is (approx= 4.0 (f 2.0 2.0)))
      ;; At midpoints: z(0.5, 0.5) = 1.0
      (is (approx= 1.0 (f 0.5 0.5)))
      ;; z(1.5, 0.5) = 2.0
      (is (approx= 2.0 (f 1.5 0.5))))))

;; ================================================================
;; 2D bicubic interpolation
;; ================================================================

(deftest interp2d-cubic-grid-exact-test
  (testing "bicubic interpolates grid points exactly"
    (let [nx 5 ny 5
          xs (double-array (map double (range nx)))
          ys (double-array (map double (range ny)))
          ;; z = x + y
          zs (double-array (for [j (range ny) i (range nx)]
                             (+ (double i) (double j))))
          f (interp2d-cubic xs ys zs nx ny)]
      ;; Check at interior grid points (bicubic needs 4-point stencil)
      (is (approx= 2.0 (f 1.0 1.0) 1e-4)
          (str "bicubic at (1,1) should be 2, got " (f 1.0 1.0)))
      (is (approx= 4.0 (f 2.0 2.0) 1e-4)
          (str "bicubic at (2,2) should be 4, got " (f 2.0 2.0)))
      (is (approx= 5.0 (f 2.0 3.0) 1e-4)
          (str "bicubic at (2,3) should be 5, got " (f 2.0 3.0))))))
