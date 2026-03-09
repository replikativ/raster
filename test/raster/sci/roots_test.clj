(ns raster.sci.roots-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [ftm]]
            [raster.sci.roots :refer [bisect newton newton-fd brent fixed-point]]))

;; ================================================================
;; Bisection tests
;; ================================================================

(deftest bisect-sqrt2-test
  (testing "Bisection finds sqrt(2)"
    (let [result (bisect (ftm [x :- Double] :- Double (- (* x x) 2.0)) 0.0 2.0)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) (Math/sqrt 2))) 1e-10)))))

(deftest bisect-sin-test
  (testing "Bisection finds root of sin near pi"
    (let [result (bisect (ftm [x :- Double] :- Double (Math/sin x)) 3.0 3.5)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) Math/PI)) 1e-10)))))

(deftest bisect-cubic-test
  (testing "Bisection finds root of x^3 - 8"
    (let [result (bisect (ftm [x :- Double] :- Double (- (* x x x) 8.0)) 0.0 3.0)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) 2.0)) 1e-10)))))

(deftest bisect-opposite-signs-test
  (testing "Bisection throws when signs not opposite"
    (is (thrown? clojure.lang.ExceptionInfo
                 (bisect (ftm [x :- Double] :- Double (* x x)) 1.0 2.0)))))

;; ================================================================
;; Newton's method tests
;; ================================================================

(deftest newton-sqrt2-test
  (testing "Newton finds sqrt(2) with explicit derivative"
    (let [result (newton (ftm [x :- Double] :- Double (- (* x x) 2.0))
                         (ftm [x :- Double] :- Double (* 2.0 x))
                         1.5)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) (Math/sqrt 2))) 1e-12))
      (is (< (:iterations result) 10)))))

(deftest newton-sin-test
  (testing "Newton finds root of sin near pi"
    (let [result (newton (ftm [x :- Double] :- Double (Math/sin x))
                         (ftm [x :- Double] :- Double (Math/cos x))
                         3.0)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) Math/PI)) 1e-12)))))

(deftest newton-fd-sqrt2-test
  (testing "Newton with finite-difference finds sqrt(2)"
    (let [result (newton-fd (ftm [x :- Double] :- Double (- (* x x) 2.0)) 1.5)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) (Math/sqrt 2))) 1e-10)))))

;; ================================================================
;; Brent's method tests
;; ================================================================

(deftest brent-sqrt2-test
  (testing "Brent finds sqrt(2)"
    (let [result (brent (ftm [x :- Double] :- Double (- (* x x) 2.0)) 0.0 2.0)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) (Math/sqrt 2))) 1e-10)))))

(deftest brent-sin-test
  (testing "Brent finds root of sin near pi"
    (let [result (brent (ftm [x :- Double] :- Double (Math/sin x)) 3.0 3.5)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) Math/PI)) 1e-10)))))

(deftest brent-cubic-test
  (testing "Brent finds root of polynomial"
    ;; x^3 - x - 2 has root near 1.5214
    (let [result (brent (ftm [x :- Double] :- Double (- (* x x x) x 2.0)) 1.0 2.0)]
      (is (:converged? result))
      (is (< (Math/abs (:f-root result)) 1e-10)))))

;; ================================================================
;; Fixed-point tests
;; ================================================================

(deftest fixed-point-cos-test
  (testing "Fixed point of cos(x) ≈ 0.7391"
    (let [result (fixed-point (ftm [x :- Double] :- Double (Math/cos x)) 1.0)]
      (is (:converged? result))
      ;; Dottie number: cos(x) = x ≈ 0.739085
      (is (< (Math/abs (- (:root result) 0.7390851332151607)) 1e-10)))))

(deftest fixed-point-sqrt-test
  (testing "Fixed point of sqrt(x) = 1"
    (let [result (fixed-point (ftm [x :- Double] :- Double (Math/sqrt x)) 2.0)]
      (is (:converged? result))
      (is (< (Math/abs (- (:root result) 1.0)) 1e-10)))))
