(ns raster.sci.stats-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.stats :refer [t-test-1sample t-test-2sample chi2-test
                                      describe percentile covariance pearson]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Descriptive statistics
;; ================================================================

(deftest describe-test
  (testing "describe [1 2 3 4 5]"
    (let [x (double-array [1.0 2.0 3.0 4.0 5.0])
          s (describe x 5)]
      (is (approx= 3.0 (.mean s)))
      (is (approx= 2.5 (.var s)))
      (is (approx= (Math/sqrt 2.5) (.std s)))
      (is (approx= 1.0 (.min s)))
      (is (approx= 5.0 (.max s)))
      (is (approx= 3.0 (.median s))))))

;; ================================================================
;; t-test (one-sample)
;; ================================================================

(deftest t-test-1sample-constant-test
  (testing "t-test of constant data = mu0 gives t=0, p=1"
    (let [x (double-array [5.0 5.0 5.0 5.0 5.0])
          ;; Constant data has zero variance, so t-stat would be NaN/0.
          ;; Use near-constant data instead to avoid division by zero.
          x2 (double-array [4.99 5.0 5.01 5.0 5.0])
          result (t-test-1sample x2 5 5.0)]
      ;; t-stat should be very close to 0
      (is (< (Math/abs (.statistic result)) 0.1)
          (str "t-stat should be near 0, got " (.statistic result))))))

(deftest t-test-1sample-shifted-test
  (testing "t-test with clearly shifted data"
    (let [x (double-array [10.0 11.0 12.0 13.0 14.0])
          result (t-test-1sample x 5 0.0)]
      ;; Should reject null: t-stat should be large, p-val small
      (is (> (Math/abs (.statistic result)) 5.0))
      (is (< (.pvalue result) 0.01)))))

;; ================================================================
;; Pearson correlation
;; ================================================================

(deftest pearson-perfect-correlation-test
  (testing "perfect positive correlation: [1 2 3], [2 4 6] => r=1.0"
    (let [x (double-array [1.0 2.0 3.0])
          y (double-array [2.0 4.0 6.0])
          result (pearson x y 3)]
      (is (approx= 1.0 (aget result 0))))))

(deftest pearson-negative-correlation-test
  (testing "perfect negative correlation: [1 2 3], [6 4 2] => r=-1.0"
    (let [x (double-array [1.0 2.0 3.0])
          y (double-array [6.0 4.0 2.0])
          result (pearson x y 3)]
      (is (approx= -1.0 (aget result 0))))))

;; ================================================================
;; Covariance
;; ================================================================

(deftest covariance-test
  (testing "covariance([1 2 3], [2 4 6]) = 2.0"
    (is (approx= 2.0 (covariance (double-array [1.0 2.0 3.0])
                                 (double-array [2.0 4.0 6.0])
                                 3)))))

;; ================================================================
;; Percentile
;; ================================================================

(deftest percentile-median-test
  (testing "50th percentile of sorted [1 2 3 4 5] = 3.0"
    (let [x (double-array [1.0 2.0 3.0 4.0 5.0])]
      (is (approx= 3.0 (percentile x 5 0.5))))))

(deftest percentile-quartiles-test
  (testing "25th and 75th percentiles"
    (let [x (double-array [1.0 2.0 3.0 4.0 5.0])]
      (is (approx= 2.0 (percentile x 5 0.25)))
      (is (approx= 4.0 (percentile x 5 0.75))))))
