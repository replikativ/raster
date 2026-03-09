(ns raster.sci.stats
  "Statistical tests and descriptive statistics.

   Provides t-tests (one-sample, Welch two-sample), chi-squared test,
   F-test, Kolmogorov-Smirnov test, Pearson/Spearman correlation,
   descriptive statistics, percentiles, and covariance.

   All numerical functions use deftm for typed multiple dispatch.

   Usage:
     (require '[raster.sci.stats :refer [t-test-1sample describe pearson]])
     (t-test-1sample (double-array [1 2 3 4 5]) 5 0.0)
     (describe (double-array [1 2 3 4 5]) 5)
     (pearson (double-array [1 2 3]) (double-array [2 4 6]) 3)"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue reduce!]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.sci.special :refer [lgamma gammainc betainc]]))

;; ================================================================
;; Result types
;; ================================================================

(defvalue TestResult [statistic :- Double, pvalue :- Double, df :- Long])
(defvalue DescriptiveStats [mean :- Double, var :- Double, std :- Double, skew :- Double, kurtosis :- Double,
                            min :- Double, max :- Double, median :- Double, q1 :- Double, q3 :- Double])

;; ================================================================
;; Helper functions
;; ================================================================

;; INPUT-GENERIC: accept (Array T), accumulate and return Double
(deftm array-mean (All [T] [x :- (Array T), n :- Long] :- Double
                       (/ (reduce! [acc 0.0] [x] (+ acc x))
                          (double n))))

(deftm array-var (All [T] [x :- (Array T), n :- Long] :- Double
                      (let [mu (array-mean x n)]
                        (/ (reduce! [acc 0.0] [x]
                                    (let [d (- x mu)]
                                      (+ acc (* d d))))
                           (double (dec n))))))

(deftm insertion-sort! (All [T] [x :- (Array T), n :- Long] :- (Array T)
                            (loop [i 1]
                              (if (>= i n)
                                x
                                (let [key-val (aget x i)
            ;; Find insertion position: scan leftward
                                      j (loop [j (dec i)]
                                          (if (and (>= j 0) (> (aget x j) key-val))
                                            (do (aset x (inc j) (aget x j))
                                                (recur (dec j)))
                                            j))]
                                  (aset x (inc j) key-val)
                                  (recur (inc i)))))))

(deftm studentt-cdf-2tail [t-stat :- Double, df :- Double] :- Double
  ;; Two-tailed p-value from Student-t distribution
  ;; p = I_x(df/2, 1/2) where x = df/(df + t^2)
  (let [t2 (* t-stat t-stat)
        x (/ df (+ df t2))]
    (betainc (* 0.5 df) 0.5 x)))

;; ================================================================
;; Statistical Tests
;; ================================================================

(deftm t-test-1sample [x :- (Array double), n :- Long, mu0 :- Double] :- TestResult
  (let [mu (array-mean x n)
        s2 (array-var x n)
        s (n/sqrt s2)
        se (/ s (n/sqrt (double n)))
        t-stat (/ (- mu mu0) se)
        df-val (dec n)
        pval (studentt-cdf-2tail t-stat (double df-val))]
    (->TestResult t-stat pval df-val)))

(deftm t-test-2sample [x1 :- (Array double), n1 :- Long,
                       x2 :- (Array double), n2 :- Long] :- TestResult
  ;; Welch's t-test (unequal variances)
  (let [mu1 (array-mean x1 n1)
        mu2 (array-mean x2 n2)
        s1-2 (array-var x1 n1)
        s2-2 (array-var x2 n2)
        se1 (/ s1-2 (double n1))
        se2 (/ s2-2 (double n2))
        se (n/sqrt (+ se1 se2))
        t-stat (/ (- mu1 mu2) se)
        ;; Welch-Satterthwaite degrees of freedom
        num (* (+ se1 se2) (+ se1 se2))
        den (+ (/ (* se1 se1) (double (dec n1)))
               (/ (* se2 se2) (double (dec n2))))
        df-approx (/ num den)
        df-val (long (m/round df-approx))
        pval (studentt-cdf-2tail t-stat df-approx)]
    (->TestResult t-stat pval df-val)))

(deftm chi2-test [observed :- (Array double), expected :- (Array double),
                  n :- Long] :- TestResult
  (let [chi2 (loop [i 0 acc 0.0]
               (if (>= i n)
                 acc
                 (let [o (aget observed i)
                       e (aget expected i)
                       d (- o e)]
                   (recur (inc i) (+ acc (/ (* d d) e))))))
        df-val (dec n)
        ;; p = 1 - gammainc(df/2, chi2/2)
        pval (- 1.0 (gammainc (* 0.5 (double df-val)) (* 0.5 chi2)))]
    (->TestResult chi2 pval df-val)))

(deftm f-test [x1 :- (Array double), n1 :- Long,
               x2 :- (Array double), n2 :- Long] :- TestResult
  (let [v1 (array-var x1 n1)
        v2 (array-var x2 n2)
        ;; Ensure F >= 1 by putting larger variance in numerator
        f-stat (if (>= v1 v2) (/ v1 v2) (/ v2 v1))
        df1 (if (>= v1 v2) (dec n1) (dec n2))
        df2 (if (>= v1 v2) (dec n2) (dec n1))
        ;; p = 1 - I_x(df1/2, df2/2) where x = df1*F/(df1*F + df2)
        x (/ (* (double df1) f-stat)
             (+ (* (double df1) f-stat) (double df2)))
        pval (* 2.0 (- 1.0 (betainc (* 0.5 (double df1))
                                    (* 0.5 (double df2)) x)))]
    ;; Two-tailed: min(2*p, 1.0)
    (->TestResult f-stat (n/min 1.0 pval) (long df1))))

(deftm ks-test [x :- (Array double), n :- Long,
                cdf-fn :- (Fn [Double] Double)] :- TestResult
  ;; Kolmogorov-Smirnov test: sort data, compute D = max|F_n(x) - cdf(x)|
  (let [sorted (aclone x)
        _ (insertion-sort! sorted n)
        nd (double n)
        d-stat (loop [i 0 d-max 0.0]
                 (if (>= i n)
                   d-max
                   (let [xi (aget sorted i)
                         f-emp (/ (double (inc i)) nd)
                         f-emp-prev (/ (double i) nd)
                         f-cdf (.invk cdf-fn xi)
                         ;; Check both sides of the step function
                         d1 (n/abs (- f-emp f-cdf))
                         d2 (n/abs (- f-emp-prev f-cdf))
                         d-new (n/max d1 d2)]
                     (recur (inc i) (n/max d-max d-new)))))
        ;; Kolmogorov distribution p-value: Smirnov series approximation
        ;; P(D > d) ~ 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2 k^2 n d^2)
        lambda (* d-stat (n/sqrt nd))
        pval (loop [k 1 acc 0.0]
               (if (> k 100)
                 (* 2.0 acc)
                 (let [sign (if (== (mod k 2) 1) 1.0 -1.0)
                       term (* sign (m/exp (* -2.0 (double k) (double k)
                                              lambda lambda)))]
                   (if (and (> k 3) (< (n/abs term) 1e-15))
                     (* 2.0 acc)
                     (recur (inc k) (+ acc term))))))]
    (->TestResult d-stat (n/max 0.0 (n/min 1.0 pval)) (long n))))

;; ================================================================
;; Correlation
;; ================================================================

(deftm pearson [x :- (Array double), y :- (Array double), n :- Long] :- (Array double)
  ;; Returns [r, pvalue] as double-array of length 2
  (let [mx (array-mean x n)
        my (array-mean y n)
        sums (loop [i 0 sxy 0.0 sxx 0.0 syy 0.0]
               (if (>= i n)
                 (double-array [sxy sxx syy])
                 (let [dx (- (aget x i) mx)
                       dy (- (aget y i) my)]
                   (recur (inc i)
                          (+ sxy (* dx dy))
                          (+ sxx (* dx dx))
                          (+ syy (* dy dy))))))
        sxy (aget sums 0)
        sxx (aget sums 1)
        syy (aget sums 2)
        r (/ sxy (n/sqrt (* sxx syy)))
        ;; t-test for significance: t = r * sqrt((n-2)/(1-r^2))
        df (- n 2)
        t-stat (* r (n/sqrt (/ (double df) (- 1.0 (* r r)))))
        pval (studentt-cdf-2tail t-stat (double df))
        result (double-array 2)]
    (aset result 0 r)
    (aset result 1 pval)
    result))

(deftm spearman [x :- (Array double), y :- (Array double), n :- Long] :- (Array double)
  ;; Rank-based correlation. Compute ranks, then apply Pearson to ranks.
  ;; Simple ranking (no tie correction): sort indices by value, assign ranks.
  (let [rank-x (double-array n)
        rank-y (double-array n)
        ;; Compute ranks for x: create index array, sort by x values
        idx-x (double-array n)
        idx-y (double-array n)
        ;; Initialize index arrays
        _ (loop [i 0]
            (when (< i n)
              (aset idx-x i (double i))
              (aset idx-y i (double i))
              (recur (inc i))))
        ;; Sort idx-x by x values (insertion sort on indices)
        _ (loop [i 1]
            (when (< i n)
              (let [key-idx (long (aget idx-x i))
                    key-val (aget x key-idx)
                    j (loop [j (dec i)]
                        (if (and (>= j 0) (> (aget x (long (aget idx-x j))) key-val))
                          (do (aset idx-x (inc j) (aget idx-x j))
                              (recur (dec j)))
                          j))]
                (aset idx-x (inc j) (double key-idx)))
              (recur (inc i))))
        ;; Assign ranks for x
        _ (loop [i 0]
            (when (< i n)
              (aset rank-x (long (aget idx-x i)) (+ (double i) 1.0))
              (recur (inc i))))
        ;; Sort idx-y by y values
        _ (loop [i 1]
            (when (< i n)
              (let [key-idx (long (aget idx-y i))
                    key-val (aget y key-idx)
                    j (loop [j (dec i)]
                        (if (and (>= j 0) (> (aget y (long (aget idx-y j))) key-val))
                          (do (aset idx-y (inc j) (aget idx-y j))
                              (recur (dec j)))
                          j))]
                (aset idx-y (inc j) (double key-idx)))
              (recur (inc i))))
        ;; Assign ranks for y
        _ (loop [i 0]
            (when (< i n)
              (aset rank-y (long (aget idx-y i)) (+ (double i) 1.0))
              (recur (inc i))))]
    ;; Pearson on ranks
    (pearson rank-x rank-y n)))

;; ================================================================
;; Descriptive Statistics
;; ================================================================

(deftm percentile [x :- (Array double), n :- Long, p :- Double] :- Double
  ;; Linear interpolation between order statistics
  ;; x must already be sorted
  (let [pos (* p (double (dec n)))
        lo (long (m/floor pos))
        hi (long (m/ceil pos))
        lo (if (< lo 0) (long 0) lo)
        hi (if (>= hi n) (dec n) hi)]
    (if (== lo hi)
      (aget x lo)
      (let [frac (- pos (double lo))]
        (+ (* (- 1.0 frac) (aget x lo))
           (* frac (aget x hi)))))))

(deftm describe [x :- (Array double), n :- Long] :- DescriptiveStats
  (let [nd (double n)
        ;; Mean
        mu (array-mean x n)
        ;; Variance and std
        v (array-var x n)
        s (n/sqrt v)
        ;; Min and max
        mn-mx (loop [i 0 mn Double/MAX_VALUE mx (- Double/MAX_VALUE)]
                (if (>= i n)
                  (double-array [mn mx])
                  (let [xi (aget x i)]
                    (recur (inc i) (n/min mn xi) (n/max mx xi)))))
        mn-val (aget mn-mx 0)
        mx-val (aget mn-mx 1)
        ;; Skewness and kurtosis (using mean and std)
        sk-ku (loop [i 0 m3 0.0 m4 0.0]
                (if (>= i n)
                  (double-array [m3 m4])
                  (let [d (/ (- (aget x i) mu) s)
                        d3 (* d d d)
                        d4 (* d3 d)]
                    (recur (inc i) (+ m3 d3) (+ m4 d4)))))
        skew (/ (aget sk-ku 0) nd)
        kurtosis (- (/ (aget sk-ku 1) nd) 3.0) ;; excess kurtosis
        ;; Sort a copy for median and quartiles
        sorted (aclone x)
        _ (insertion-sort! sorted n)
        med (percentile sorted n 0.5)
        q1-val (percentile sorted n 0.25)
        q3-val (percentile sorted n 0.75)]
    (->DescriptiveStats mu v s skew kurtosis mn-val mx-val med q1-val q3-val)))

(deftm covariance [x :- (Array double), y :- (Array double), n :- Long] :- Double
  ;; Sample covariance (divide by n-1)
  (let [mx (array-mean x n)
        my (array-mean y n)]
    (/ (reduce! [acc 0.0] [x y]
                (+ acc (* (- x mx) (- y my))))
       (double (dec n)))))
