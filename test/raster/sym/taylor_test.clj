(ns raster.sym.taylor-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sym.taylor :as taylor]))

(defn- approx= [^double a ^double b ^double tol]
  (< (Math/abs (- a b)) tol))

(deftest sin-taylor-test
  (testing "sin(x) Taylor coefficients around 0"
    (let [coeffs (taylor/taylor-coeffs '(sin x) 'x 0.0 5)]
      (is (approx= 0.0 (nth coeffs 0) 1e-14))
      (is (approx= 1.0 (nth coeffs 1) 1e-14))
      (is (approx= 0.0 (nth coeffs 2) 1e-14))
      (is (approx= (/ -1.0 6) (nth coeffs 3) 1e-14))
      (is (approx= 0.0 (nth coeffs 4) 1e-14))
      (is (approx= (/ 1.0 120) (nth coeffs 5) 1e-14)))))

(deftest exp-taylor-test
  (testing "exp(x) Taylor coefficients around 0"
    (let [coeffs (taylor/taylor-coeffs '(exp x) 'x 0.0 4)]
      (is (approx= 1.0 (nth coeffs 0) 1e-14))
      (is (approx= 1.0 (nth coeffs 1) 1e-14))
      (is (approx= 0.5 (nth coeffs 2) 1e-14))
      (is (approx= (/ 1.0 6) (nth coeffs 3) 1e-14))
      (is (approx= (/ 1.0 24) (nth coeffs 4) 1e-14)))))

(deftest cos-taylor-test
  (testing "cos(x) Taylor coefficients around 0"
    (let [coeffs (taylor/taylor-coeffs '(cos x) 'x 0.0 4)]
      (is (approx= 1.0 (nth coeffs 0) 1e-14))
      (is (approx= 0.0 (nth coeffs 1) 1e-14))
      (is (approx= -0.5 (nth coeffs 2) 1e-14))
      (is (approx= 0.0 (nth coeffs 3) 1e-14))
      (is (approx= (/ 1.0 24) (nth coeffs 4) 1e-14)))))

(deftest log1px-taylor-test
  (testing "log(1+x) Taylor coefficients around 0"
    (let [coeffs (taylor/taylor-coeffs '(log (+ 1 x)) 'x 0.0 4)]
      (is (approx= 0.0 (nth coeffs 0) 1e-14))
      (is (approx= 1.0 (nth coeffs 1) 1e-14))
      (is (approx= -0.5 (nth coeffs 2) 1e-14))
      (is (approx= (/ 1.0 3) (nth coeffs 3) 1e-14))
      (is (approx= -0.25 (nth coeffs 4) 1e-14)))))

(deftest taylor-expand-test
  (testing "taylor-expand produces S-expression"
    (let [result (taylor/taylor-expand '(sin x) 'x 0 3)]
      (is (seq? result)))))

(deftest taylor-nonzero-center-test
  (testing "exp(x) around x=1 has coefficients e/n!"
    (let [e (Math/E)
          coeffs (taylor/taylor-coeffs '(exp x) 'x 1.0 3)]
      (is (approx= e (nth coeffs 0) 1e-12))
      (is (approx= e (nth coeffs 1) 1e-12))
      (is (approx= (/ e 2) (nth coeffs 2) 1e-12))
      (is (approx= (/ e 6) (nth coeffs 3) 1e-12)))))

(deftest taylor-remainder-test
  (testing "remainder estimation returns next derivative info"
    (let [result (taylor/taylor-remainder '(exp x) 'x 0.0 3)]
      (is (some? (:remainder-bound result)))
      (is (approx= 1.0 (:next-deriv-at-a result) 1e-12)))))
