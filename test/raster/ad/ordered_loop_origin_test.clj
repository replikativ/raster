(ns raster.ad.ordered-loop-origin-test
  "Reverse AD of an ordered reduction whose counter starts at a nonzero literal
  takes the same reduction path as one starting at 0 (no per-iteration closure
  tape), with the same gradient; and a reference to a `:const` or primitive-tagged
  Var is typed at the call it feeds, so the arithmetic around it devirtualizes."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.compiler.pipeline :as pipeline]))

;; Σ_{t=lag}^{cnt-1} −(y_t − a − b·y_{t−lag})², counting from `lag`
(deftm lagged-from-origin [a :- Double, b :- Double, ys :- (Array double), cnt :- Long] :- Double
  (loop [t 3 acc (n/* a b)]
    (if (< t cnt)
      (let [r (n/- (ra/aget ys t) (n/+ a (n/* b (ra/aget ys (- t 3)))))]
        (recur (inc t) (n/- acc (n/* r r))))
      acc)))

(defn- reference
  "[value d/da d/db] of lagged-from-origin."
  [a b ^doubles ys cnt]
  (reduce (fn [[v da db] t]
            (let [r (- (aget ys t) a (* b (aget ys (- t 3))))]
              [(- v (* r r)) (+ da (* 2 r)) (+ db (* 2 r (aget ys (- t 3))))]))
          [(* a b) b a] (range 3 (max 3 cnt))))

(def ^:private ys (double-array (map #(Math/sin (* 0.7 %)) (range 64))))

(defn- close? [a b] (<= (Math/abs (- (double a) (double b))) (* 1e-10 (max 1.0 (Math/abs (double b))))))

(deftest nonzero-origin-reductions-differentiate-exactly
  (doseq [[label gradient] [[:evaluated (rev/value+grad #'lagged-from-origin :wrt [0 1])]
                            [:compiled (rev/value+grad #'lagged-from-origin :wrt [0 1] :compile? true)]]
          [a b cnt] [[0.3 0.5 64] [-1.0 2.0 10] [0.3 0.5 4] [0.3 0.5 3] [0.3 0.5 0]]]
    (testing [label a b cnt]
      (is (every? true? (map close? (take 3 (gradient a b ys cnt)) (reference a b ys cnt)))))))

(defn- lowered-forms [v]
  (tree-seq coll? seq (:lowered (binding [*err* (java.io.StringWriter.)]
                                  (pipeline/show-pipeline v)))))

(deftm lagged-gradient [a :- Double, b :- Double, ys :- (Array double), cnt :- Long] :- (Array double)
  (let [result ((rev/value+grad (var lagged-from-origin) :wrt [0 1]) a b ys cnt)]
    (double-array [(nth result 0) (nth result 1) (nth result 2)])))

(deftest nonzero-origin-reductions-need-no-closure-tape
  (is (not-any? #(and (seq? %) (= 'fn* (first %))) (lowered-forms #'lagged-gradient))
      "the reduction's pullback is not a per-iteration closure"))

(def ^:private ^:const half-log-2pi 0.9189385332046727)
(def ^:private ^{:tag 'double} log-two 0.6931471805599453)

(deftm uses-constant-vars [ys :- (Array double), mu :- Double] :- Double
  (loop [i 0 acc 0.0]
    (if (< i (alength ys))
      (let [r (n/- (ra/aget ys i) mu)]
        (recur (inc i) (n/- acc (n/+ (n/* 0.5 (n/* r r)) (n/+ half-log-2pi log-two)))))
      acc)))

(deftest constant-var-references-devirtualize
  (let [walked (tree-seq coll? seq (pipeline/get-walked-body #'uses-constant-vars nil))]
    (is (not-any? #(and (seq? %) (= 'raster.numeric/+ (first %))) walked)
        "(n/+ half-log-2pi log-two) resolves to its double overload"))
  (let [f (rev/value+grad #'uses-constant-vars :wrt [1] :compile? true)
        mu 0.25
        [v _ d] (f ys mu)]
    (is (close? v (reduce + (map #(- (+ (* 0.5 (- % mu) (- % mu)) 0.9189385332046727 0.6931471805599453)) ys))))
    (is (close? d (reduce + (map #(- % mu) ys))))))

;; Two carries where only one reaches the result: the other carry's cotangent at
;; loop exit is zero. Its per-step pullback treats the double-tagged cotangent as
;; present, so the backward loop must seed it with 0.0, not nil.
(deftm carried-variance [theta :- (Array double), ys :- (Array double), cnt :- Long] :- Double
  (let [mu (ra/aget theta 0) a0 (ra/aget theta 1) a1 (ra/aget theta 2) b1 (ra/aget theta 3)]
    (loop [t 1 s 1.0 acc 0.0]
      (if (< t cnt)
        (let [e (n/- (ra/aget ys (- t 1)) mu)
              s2 (n/+ a0 (n/+ (n/* a1 (n/* e e)) (n/* b1 s)))
              r (n// (n/- (ra/aget ys t) mu) s2)]
          (recur (inc t) s2 (n/- acc (n/+ (n/* 0.5 (n/* r r)) s2))))
        acc))))

(deftest absent-carry-cotangents-stay-nil-safe-when-compiled
  (let [compiled (rev/value+grad #'carried-variance :wrt [0] :compile? true)
        theta [0.1 1.2 0.3 0.4]
        [v g] (compiled (double-array theta) ys 20)
        h 1e-6
        f #(carried-variance (double-array %) ys 20)]
    (is (close? v (f theta)))
    (doseq [j (range 4)]
      (is (< (Math/abs (- (aget ^doubles g j) (/ (- (f (update theta j + h)) (f (update theta j - h))) (* 2 h))))
             1e-5)))))
