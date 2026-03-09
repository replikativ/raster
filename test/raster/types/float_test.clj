(ns raster.types.float-test
  (:refer-clojure :exclude [+ - * / zero? min max abs])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.numeric :refer [+ - * / zero one similar abs
                                    min max zero? sqrt pow real-value]]
            [raster.types.promote :as promote :refer [promote promote-type]]))

;; ================================================================
;; Float+Float arithmetic
;; ================================================================

(deftest float-addition-test
  (testing "Float + Float"
    (let [r (+ (float 1.5) (float 2.5))]
      (is (== 4.0 r))
      (is (instance? Float r)))))

(deftest float-subtraction-test
  (testing "unary Float negation"
    (let [r (- (float 3.0))]
      (is (== -3.0 r))
      (is (instance? Float r))))
  (testing "binary Float subtraction"
    (let [r (- (float 5.5) (float 2.0))]
      (is (== 3.5 r))
      (is (instance? Float r)))))

(deftest float-multiplication-test
  (testing "Float * Float"
    (let [r (* (float 2.0) (float 3.0))]
      (is (== 6.0 r))
      (is (instance? Float r)))))

(deftest float-division-test
  (testing "Float / Float"
    (let [r (/ (float 10.0) (float 4.0))]
      (is (== 2.5 r))
      (is (instance? Float r)))))

;; ================================================================
;; Float math functions
;; ================================================================

(deftest float-abs-test
  (testing "Float abs"
    (let [r (abs (float -3.14))]
      (is (< (Math/abs (clojure.core/- (double r) 3.14)) 0.001))
      (is (instance? Float r)))))

(deftest float-min-max-test
  (testing "Float min"
    (let [r (min (float 1.5) (float 2.5))]
      (is (== 1.5 r))
      (is (instance? Float r))))
  (testing "Float max"
    (let [r (max (float 1.5) (float 2.5))]
      (is (== 2.5 r))
      (is (instance? Float r)))))

(deftest float-sqrt-test
  (testing "Float sqrt"
    (let [r (sqrt (float 4.0))]
      (is (== 2.0 r))
      (is (instance? Float r)))))

(deftest float-pow-test
  (testing "Float pow"
    (let [r (pow (float 2.0) (float 3.0))]
      (is (== 8.0 r))
      (is (instance? Float r)))))

(deftest float-real-value-test
  (testing "Float real-value returns Double"
    (let [r (real-value (float 3.14))]
      (is (instance? Double r))
      (is (< (Math/abs (clojure.core/- r 3.14)) 0.01)))))

(deftest float-zero-pred-test
  (testing "Float zero?"
    (is (zero? (float 0.0)))
    (is (not (zero? (float 1.0))))))

;; ================================================================
;; Float generic constructors
;; ================================================================

(deftest float-zero-test
  (testing "Float zero"
    (let [r (zero (float 3.14))]
      (is (== 0.0 r))
      (is (instance? Float r)))))

(deftest float-one-test
  (testing "Float one"
    (let [r (one (float 3.14))]
      (is (== 1.0 r))
      (is (instance? Float r)))))

;; ================================================================
;; Float promotion
;; ================================================================

(deftest float-promote-type-test
  (testing "Long + Float -> Float"
    (is (= Float (promote-type Long Float)))
    (is (= Float (promote-type Float Long))))
  (testing "Float + Double -> Double"
    (is (= Double (promote-type Float Double)))
    (is (= Double (promote-type Double Float)))))

(deftest float-promote-test
  (testing "Long + Float promotes to [Float Float]"
    (let [[a b] (promote 1 (float 2.0))]
      (is (instance? Float a))
      (is (instance? Float b))
      (is (== 1.0 a))
      (is (== 2.0 b))))
  (testing "Float + Double promotes to [Double Double]"
    (let [[a b] (promote (float 1.5) 2.5)]
      (is (instance? Double a))
      (is (instance? Double b)))))

(deftest float-mixed-arithmetic-test
  (testing "Long + Float -> Float via promotion"
    (let [r (+ 1 (float 2.0))]
      (is (== 3.0 r))
      (is (instance? Float r))))
  (testing "Float + Double -> Double via promotion"
    (let [r (+ (float 1.5) 2.5)]
      (is (== 4.0 r))
      (is (instance? Double r)))))

;; ================================================================
;; Float-array operations
;; ================================================================

(deftest float-array-addition-test
  (testing "float-array + float-array"
    (let [r (+ (float-array [1 2 3]) (float-array [4 5 6]))]
      (is (= [5.0 7.0 9.0] (mapv float (seq r))))
      (is (instance? (Class/forName "[F") r)))))

(deftest float-array-subtraction-test
  (testing "float-array - float-array"
    (let [r (- (float-array [4 5 6]) (float-array [1 2 3]))]
      (is (= [3.0 3.0 3.0] (mapv float (seq r))))))
  (testing "negate float-array"
    (let [r (- (float-array [1 2 3]))]
      (is (= [-1.0 -2.0 -3.0] (mapv float (seq r)))))))

(deftest float-array-scalar-mul-test
  (testing "Float * float-array"
    (let [r (* (float 2.0) (float-array [1 2 3]))]
      (is (= [2.0 4.0 6.0] (mapv float (seq r))))))
  (testing "float-array * Float"
    (let [r (* (float-array [1 2 3]) (float 3.0))]
      (is (= [3.0 6.0 9.0] (mapv float (seq r)))))))

(deftest float-array-scalar-div-test
  (testing "float-array / Float"
    (let [r (/ (float-array [10 20 30]) (float 5.0))]
      (is (= [2.0 4.0 6.0] (mapv float (seq r)))))))

(deftest float-array-zero-test
  (testing "zero creates zeroed float-array of same size"
    (let [a (float-array [1 2 3])
          z (zero a)]
      (is (= 3 (alength z)))
      (is (instance? (Class/forName "[F") z))
      (is (every? #(== 0.0 %) (seq z))))))

(deftest float-array-similar-test
  (testing "similar creates float-array of same size"
    (let [a (float-array [1 2 3])
          s (similar a)]
      (is (= 3 (alength s)))
      (is (instance? (Class/forName "[F") s))
      (is (not (identical? a s))))))

;; ================================================================
;; Float precision round-trip
;; ================================================================

(deftest float-precision-test
  (testing "float -> double -> float preserves float precision"
    (let [f (float 1.23456)
          d (double f)
          f2 (float d)]
      (is (== f f2))))
  (testing "float has less precision than double"
    ;; Float32 has ~7 decimal digits; this value differs in float vs double
    (let [d 1.23456789012345
          f (float d)]
      (is (not= d (double f))))))
