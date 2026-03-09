(ns raster.types.complex-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.numeric :refer [+ - * /]]
            [raster.types.complex :refer [complex re im mag angle conj*
                                          cexp csin ccos clog csqrt]]))

(defn- close? [^double a ^double b]
  (< (Math/abs (- a b)) 1e-12))

;; ================================================================
;; Basic arithmetic
;; ================================================================

(deftest complex-add-test
  (testing "complex + complex"
    (let [r (+ (complex 1.0 2.0) (complex 3.0 4.0))]
      (is (close? 4.0 (re r)))
      (is (close? 6.0 (im r)))))
  (testing "complex + real"
    (let [r (+ (complex 1.0 2.0) 3.0)]
      (is (close? 4.0 (re r)))
      (is (close? 2.0 (im r)))))
  (testing "real + complex"
    (let [r (+ 3 (complex 1.0 2.0))]
      (is (close? 4.0 (re r)))
      (is (close? 2.0 (im r))))))

(deftest complex-sub-test
  (testing "complex - complex"
    (let [r (- (complex 5.0 3.0) (complex 2.0 1.0))]
      (is (close? 3.0 (re r)))
      (is (close? 2.0 (im r)))))
  (testing "negate complex"
    (let [r (- (complex 3.0 4.0))]
      (is (close? -3.0 (re r)))
      (is (close? -4.0 (im r))))))

(deftest complex-mul-test
  (testing "(1+2i)(3+4i) = -5+10i"
    (let [r (* (complex 1.0 2.0) (complex 3.0 4.0))]
      (is (close? -5.0 (re r)))
      (is (close? 10.0 (im r)))))
  (testing "scalar * complex"
    (let [r (* 2.0 (complex 3.0 4.0))]
      (is (close? 6.0 (re r)))
      (is (close? 8.0 (im r)))))
  (testing "i*i = -1"
    (let [i (complex 0.0 1.0)
          r (* i i)]
      (is (close? -1.0 (re r)))
      (is (close? 0.0 (im r))))))

(deftest complex-div-test
  (testing "(1+2i)/(3+4i)"
    (let [r (/ (complex 1.0 2.0) (complex 3.0 4.0))]
      ;; (1+2i)(3-4i) / 25 = (11+2i)/25
      (is (close? 0.44 (re r)))
      (is (close? 0.08 (im r)))))
  (testing "complex / scalar"
    (let [r (/ (complex 6.0 8.0) 2.0)]
      (is (close? 3.0 (re r)))
      (is (close? 4.0 (im r))))))

;; ================================================================
;; Complex operations
;; ================================================================

(deftest complex-ops-test
  (testing "magnitude"
    (is (close? 5.0 (mag (complex 3.0 4.0)))))
  (testing "angle"
    (is (close? (/ Math/PI 2.0) (angle (complex 0.0 1.0))))
    (is (close? 0.0 (angle (complex 1.0 0.0)))))
  (testing "conjugate"
    (let [r (conj* (complex 3.0 4.0))]
      (is (close? 3.0 (re r)))
      (is (close? -4.0 (im r))))))

;; ================================================================
;; Complex math functions
;; ================================================================

(deftest complex-exp-test
  (testing "exp(iπ) = -1 (Euler's formula)"
    (let [r (cexp (complex 0.0 Math/PI))]
      (is (close? -1.0 (re r)))
      (is (close? 0.0 (im r)))))
  (testing "exp(1+0i) = e"
    (let [r (cexp (complex 1.0 0.0))]
      (is (close? Math/E (re r)))
      (is (close? 0.0 (im r))))))

(deftest complex-trig-test
  (testing "sin²(z) + cos²(z) = 1 for complex z"
    (let [z (complex 1.0 2.0)
          s (csin z)
          c (ccos z)
          sum (+ (* s s) (* c c))]
      (is (close? 1.0 (re sum)))
      (is (close? 0.0 (im sum))))))

(deftest complex-log-test
  (testing "log(exp(z)) = z"
    (let [z (complex 1.5 0.8)
          r (clog (cexp z))]
      (is (close? 1.5 (re r)))
      (is (close? 0.8 (im r))))))

(deftest complex-sqrt-test
  (testing "sqrt(z)² = z"
    (let [z (complex 3.0 4.0)
          s (csqrt z)
          r (* s s)]
      (is (close? 3.0 (re r)))
      (is (close? 4.0 (im r)))))
  (testing "sqrt(-1) = i"
    (let [r (csqrt (complex -1.0 0.0))]
      (is (close? 0.0 (re r)))
      (is (close? 1.0 (im r))))))

;; ================================================================
;; Variadic with complex (reducible dispatch)
;; ================================================================

(deftest complex-variadic-test
  (testing "3-arg +"
    (let [r (+ (complex 1.0 1.0) (complex 2.0 2.0) (complex 3.0 3.0))]
      (is (close? 6.0 (re r)))
      (is (close? 6.0 (im r)))))
  (testing "3-arg *"
    ;; (1+i)(1-i)(2+0i) = (1+1)(2) = 4
    (let [r (* (complex 1.0 1.0) (complex 1.0 -1.0) (complex 2.0 0.0))]
      (is (close? 4.0 (re r)))
      (is (close? 0.0 (im r))))))
