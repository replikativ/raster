(ns raster.ad.ordered-loop-jvp-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse :as rev]))

(deftm nonzero-nonlinear [x :- Double, cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt) (recur (inc i) (n/* acc acc)) acc)))

(deftm inclusive-active-init [x :- Double, cnt :- Long] :- Double
  (loop [acc (n/* x x) i 1]
    (if (<= i cnt) (recur (n/+ acc x) (inc i)) acc)))

(deftm wide-induction [x :- Double, cnt :- Long] :- Double
  (loop [i 2147483648 acc 0.0]
    (if (< i cnt) (recur (inc i) (n/+ acc x)) acc)))

(deftm cancellation-and-data [x :- Double, ys :- (Array double), cnt :- Long] :- Double
  (loop [i 1 acc (n/* x x)]
    (if (< i cnt)
      (let [y (aget ys i)] (recur (inc i) (n/+ acc (n/* x y))))
      acc)))

(deftm active-bound [x :- Double] :- Double
  (loop [i 1 acc 0.0]
    (if (< i x) (recur (inc i) (n/+ acc x)) acc)))

(deftm active-origin [x :- Double] :- Double
  (loop [i (long x) acc 0.0]
    (if (< i 4) (recur (inc i) (n/+ acc x)) acc)))

(deftm trapping-origin [x :- Double, denom :- Long, ys :- (Array double)] :- Double
  (loop [i (quot 1 denom) acc (n/* x (aget ys 10))]
    (if (< i 3) (recur (inc i) (n/+ acc x)) acc)))

(deftm trapping-step [x :- Double, ys :- (Array double)] :- Double
  (loop [i 9223372036854775807 acc x]
    (if (<= i 9223372036854775807)
      (recur (inc i) (n/+ acc (aget ys 10))) acc)))

(deftm trapping-prelude [x :- Double, ys :- (Array double)] :- Double
  (loop [i 9223372036854775807 acc x]
    (if (<= i 9223372036854775807)
      (let [v (aget ys 10)] (recur (inc i) (n/+ acc v))) acc)))

(deftm carry-first-trapping-step [x :- Double, ys :- (Array double)] :- Double
  (loop [acc x i 9223372036854775807]
    (if (<= i 9223372036854775807)
      (recur (n/+ acc (aget ys 10)) (inc i)) acc)))

(deftm effectful-step [x :- Double, out :- (Array double)] :- Double
  (loop [i 1 acc x]
    (if (< i 3)
      (let [ignored (aset out i x)] (recur (inc i) (n/+ acc x)))
      acc)))

(deftest public-ordered-recurrence-jvp-keeps-domain-and-active-init
  (doseq [[v cases]
          [[#'nonzero-nonlinear [[0 2.0 0.5] [1 2.0 0.5] [3 16.0 16.0]]]
           [#'inclusive-active-init [[0 4.0 2.0] [1 6.0 2.5] [3 10.0 3.5]]]
           [#'wide-induction [[2147483648 0.0 0.0] [2147483650 4.0 1.0]]]]]
    (let [forward (jvp/jvp v :wrt [0])]
      (doseq [[cnt primal tangent] cases]
        (is (= primal (@v 2.0 cnt)))
        (is (= [primal tangent] (forward 2.0 cnt 0.5)))))))

(deftest ordered-recurrence-preserves-floating-checkpoints-and-data-activity
  (let [forward (jvp/jvp #'cancellation-and-data :wrt [0])
        reverse (rev/value+grad #'cancellation-and-data :wrt [0])]
    (doseq [[ys cnt] [[(double-array [0 1.0e16 1 -1.0e16]) 4]
                     [(double-array [0 2 3 4]) 4]
                     [(double-array [0]) 0]]]
      ;; Independent ordered primal and tangent recurrences, not an algebraic
      ;; closed form: reassociating the cancellation case changes its answer.
      (let [[primal tangent]
            (loop [i 1 acc 1.0 dacc 1.0]
              (if (< i cnt)
                (recur (inc i) (+ acc (aget ys i)) (+ dacc (* 0.5 (aget ys i))))
                [acc dacc]))]
        (is (= primal (cancellation-and-data 1.0 ys cnt)))
        (is (= [primal tangent] (forward 1.0 ys cnt 0.5)))
        (is (= primal (first (reverse 1.0 ys cnt))))
        (is (= [nil nil] (vec (drop 2 (reverse 1.0 ys cnt)))))))))

(deftest unsupported-ordered-loops-remain-explicit-refusals
  (doseq [v [#'active-bound #'active-origin #'effectful-step]]
    (is (= :jvp-ordered-loop-unsupported
           (try (jvp/jvp v :wrt [0]) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest ordered-linearization-retains-trap-order
  (doseq [[v args expected]
          [[#'trapping-origin [2.0 0 (double-array 0)] ArithmeticException]
           [#'trapping-step [2.0 (double-array 0)] ArithmeticException]
           [#'trapping-prelude [2.0 (double-array 0)] ArrayIndexOutOfBoundsException]
           [#'carry-first-trapping-step [2.0 (double-array 0)] ArrayIndexOutOfBoundsException]]]
    (let [exception-class (fn [f args]
                            (try (apply f args) nil (catch Throwable e (class e))))
          forward (jvp/jvp v :wrt [0])]
      (is (= expected (exception-class @v args)))
      (is (= expected (exception-class forward (conj args 0.5)))))))
