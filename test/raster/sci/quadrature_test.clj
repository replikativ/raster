(ns raster.sci.quadrature-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.quadrature :refer [quadgk trapz trapz-xy simps]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Gauss-Kronrod quadrature
;; ================================================================

(deftest quadgk-polynomial-test
  (testing "integrate x^2 from 0 to 1 = 1/3"
    (let [[integral err] (quadgk (fn [x] (* x x)) 0.0 1.0)]
      (is (approx= (/ 1.0 3.0) integral 1e-10)
          (str "Expected 1/3, got " integral))
      (is (< err 1e-10)
          (str "Error estimate should be tiny, got " err)))))

(deftest quadgk-sin-test
  (testing "integrate sin(x) from 0 to pi = 2"
    (let [[integral err] (quadgk (fn [x] (Math/sin x)) 0.0 Math/PI)]
      (is (approx= 2.0 integral 1e-10)
          (str "Expected 2.0, got " integral)))))

(deftest quadgk-exp-test
  (testing "integrate exp(x) from 0 to 1 = e-1"
    (let [[integral err] (quadgk (fn [x] (Math/exp x)) 0.0 1.0)]
      (is (approx= (- Math/E 1.0) integral 1e-10)
          (str "Expected " (- Math/E 1.0) ", got " integral)))))

(deftest quadgk-constant-test
  (testing "integrate 1 from 0 to 5 = 5"
    (let [[integral _] (quadgk (fn [_] 1.0) 0.0 5.0)]
      (is (approx= 5.0 integral 1e-10)))))

(deftest quadgk-cubic-test
  (testing "integrate x^3 from 0 to 2 = 4"
    (let [[integral _] (quadgk (fn [x] (* x x x)) 0.0 2.0)]
      (is (approx= 4.0 integral 1e-10)))))

(deftest quadgk-oscillatory-test
  (testing "integrate sin(10x) from 0 to 2*pi = 0"
    (let [[integral err] (quadgk (fn [x] (Math/sin (* 10.0 x))) 0.0 (* 2.0 Math/PI))]
      (is (approx= 0.0 integral 1e-6)
          (str "Expected 0, got " integral)))))

;; ================================================================
;; Trapezoidal rule
;; ================================================================

(deftest trapz-constant-test
  (testing "trapz on constant function"
    (let [ys (double-array [3.0 3.0 3.0 3.0 3.0])
          dx 0.25]
      (is (approx= 3.0 (trapz ys dx))))))

(deftest trapz-linear-test
  (testing "trapz on linear function (exact)"
    (let [ys (double-array [0.0 1.0 2.0 3.0 4.0])
          dx 1.0]
      ;; integral of f(x)=x from 0 to 4 = 8
      (is (approx= 8.0 (trapz ys dx))))))

(deftest trapz-quadratic-test
  (testing "trapz on x^2 data"
    ;; integral of x^2 from 0 to 4 = 64/3 ≈ 21.333
    ;; trapz with 5 points (dx=1): underestimates
    (let [ys (double-array [0.0 1.0 4.0 9.0 16.0])
          dx 1.0
          result (trapz ys dx)]
      (is (approx= 22.0 result 0.1)))))

(deftest trapz-xy-test
  (testing "trapz-xy with non-uniform spacing"
    (let [xs (double-array [0.0 0.5 1.5 3.0])
          ys (double-array [0.0 1.0 1.0 0.0])
          result (trapz-xy xs ys)]
      ;; Manual: 0.5*0.5*(0+1) + 0.5*1.0*(1+1) + 0.5*1.5*(1+0)
      ;; = 0.25 + 1.0 + 0.75 = 2.0
      (is (approx= 2.0 result)))))

;; ================================================================
;; Simpson's rule
;; ================================================================

(deftest simps-polynomial-test
  (testing "Simpson's rule integrates x^2 exactly"
    (let [result (simps (fn [x] (* x x)) 0.0 1.0 100)]
      (is (approx= (/ 1.0 3.0) result 1e-10)))))

(deftest simps-sin-test
  (testing "Simpson's rule integrates sin(x) from 0 to pi"
    (let [result (simps (fn [x] (Math/sin x)) 0.0 Math/PI 100)]
      (is (approx= 2.0 result 1e-7)))))

(deftest simps-cubic-test
  (testing "Simpson's rule integrates x^3 exactly"
    ;; Simpson's is exact for polynomials up to degree 3
    (let [result (simps (fn [x] (* x x x)) 0.0 2.0 4)]
      (is (approx= 4.0 result 1e-10)))))
