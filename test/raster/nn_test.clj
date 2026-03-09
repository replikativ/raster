(ns raster.nn-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.nn :as nn]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; Helpers
;; ================================================================

(defn- approx=
  "True if a and b are within eps."
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) eps)))

(defn- arr-approx=
  "True if all elements of double arrays a and b are within eps."
  ([a b] (arr-approx= a b 1e-6))
  ([^doubles a ^doubles b eps]
   (and (= (alength a) (alength b))
        (every? true?
                (for [i (range (alength a))]
                  (< (Math/abs (- (aget a i) (aget b i))) eps))))))

;; ================================================================
;; double[] overload tests
;; ================================================================

(deftest dense-doubles-test
  (testing "Dense layer forward with double[]"
    ;; W is row-major: 2 rows, 3 cols
    ;; Row 0: [1 3 5], Row 1: [2 4 6]
    (let [W (double-array [1 3 5 2 4 6])
          x (double-array [1 2 3])
          b (double-array [0.1 0.2])
          y (nn/dense W x b)]
      (is (instance? (Class/forName "[D") y))
      (is (= 2 (alength ^doubles y)))
      ;; y[0] = 1*1 + 3*2 + 5*3 + 0.1 = 22.1
      ;; y[1] = 2*1 + 4*2 + 6*3 + 0.2 = 28.2
      (is (approx= 22.1 (aget ^doubles y 0)))
      (is (approx= 28.2 (aget ^doubles y 1))))))

(deftest relu-doubles-test
  (testing "ReLU forward with double[]"
    (let [x (double-array [-2 -1 0 1 2])
          y (nn/relu x)]
      (is (instance? (Class/forName "[D") y))
      (is (approx= 0.0 (aget ^doubles y 0)))
      (is (approx= 0.0 (aget ^doubles y 1)))
      (is (approx= 0.0 (aget ^doubles y 2)))
      (is (approx= 1.0 (aget ^doubles y 3)))
      (is (approx= 2.0 (aget ^doubles y 4))))))

(deftest softmax-doubles-test
  (testing "Softmax with double[] sums to 1"
    (let [x (double-array [1 2 3])
          s (nn/softmax x)
          sum (loop [i 0 acc 0.0]
                (if (< i 3)
                  (recur (inc i) (+ acc (aget ^doubles s i)))
                  acc))]
      (is (approx= 1.0 sum))))
  (testing "Softmax ordering preserved"
    (let [x (double-array [1 2 3])
          s (nn/softmax x)]
      (is (< (aget ^doubles s 0) (aget ^doubles s 1)))
      (is (< (aget ^doubles s 1) (aget ^doubles s 2))))))

(deftest cross-entropy-doubles-test
  (testing "Cross-entropy with double[]"
    (let [p (double-array [0.1 0.7 0.2])
          t (double-array [0 1 0])]
      (is (approx= (- (Math/log 0.7)) (nn/cross-entropy p t))))))

(deftest relu-doubles-rrule-test
  (testing "ReLU rrule works with double[]"
    (let [x (double-array [-2 -1 0.5 1 2])
          y (nn/relu x)
          rrule-fn (tmpl/get-pullback-factory 'raster.nn/relu)
          pb (rrule-fn y x)
          dy (double-array [1 1 1 1 1])
          [dx] (pb dy)]
      (is (instance? (Class/forName "[D") dx))
      (is (approx= 0.0 (aget ^doubles dx 0)))
      (is (approx= 0.0 (aget ^doubles dx 1)))
      (is (approx= 1.0 (aget ^doubles dx 2)))
      (is (approx= 1.0 (aget ^doubles dx 3)))
      (is (approx= 1.0 (aget ^doubles dx 4))))))

(deftest softmax-doubles-rrule-test
  (testing "Softmax rrule works with double[]"
    (let [x (double-array [1.0 2.0 3.0])
          s (nn/softmax x)
          rrule-fn (tmpl/get-pullback-factory 'raster.nn/softmax)
          pb (rrule-fn s x)
          dy (double-array [1.0 0.0 0.0])
          [dx] (pb dy)]
      (is (instance? (Class/forName "[D") dx))
      ;; Numerical check
      (let [eps 1e-5
            n (alength x)
            num-dx (double-array n)]
        (dotimes [i n]
          (let [orig (aget x i)]
            (aset x i (+ orig eps))
            (let [s+ (nn/softmax x)
                  f+ (aget ^doubles s+ 0)]
              (aset x i orig)
              (let [s0 (nn/softmax x)
                    f0 (aget ^doubles s0 0)]
                (aset num-dx i (/ (- f+ f0) eps))))))
        (is (arr-approx= dx num-dx 1e-4))))))

(deftest cross-entropy-doubles-rrule-test
  (testing "Cross-entropy rrule works with double[]"
    (let [p (double-array [0.2 0.5 0.3])
          t (double-array [0 1 0])
          loss (nn/cross-entropy p t)
          rrule-fn (tmpl/get-pullback-factory 'raster.nn/cross-entropy)
          pb (rrule-fn loss p t)
          [dp _dt] (pb 1.0)]
      (is (instance? (Class/forName "[D") dp))
      ;; dp[1] should be -t[1]/p[1] = -1/0.5 = -2.0
      (is (approx= -2.0 (aget ^doubles dp 1))))))
