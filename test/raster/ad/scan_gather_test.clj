(ns raster.ad.scan-gather-test
  "Carry loops whose step reads an active array at a data-dependent index
  (a varying-intercept model's alpha[group[i]]) differentiate through the
  par/scan pullback: the gather index is replayed in the backward loop and
  the cotangent scatter-adds there, instead of a per-iteration closure tape."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.arrays :as ra]
            [raster.math :as rm]
            [raster.numeric :as n]))

(deftm varying-intercepts [mu :- Double, tau :- Double, beta :- Double, alpha :- (Array double),
                           group :- (Array long), x :- (Array double), y :- (Array double),
                           cnt :- Long, groups :- Long] :- Double
  (let [prior (loop [g 0 acc 0.0]
                (if (< g groups)
                  (let [z (n// (n/- (ra/aget alpha g) mu) tau)]
                    (recur (inc g) (n/- acc (n/+ (n/* 0.5 (n/* z z)) (rm/log tau)))))
                  acc))]
    (loop [i 0 acc prior]
      (if (< i cnt)
        (let [r (n/- (ra/aget y i) (n/+ (ra/aget alpha (ra/aget group i)) (n/* beta (ra/aget x i))))]
          (recur (inc i) (n/- acc (n/* 0.5 (n/* r r)))))
        acc))))

(def ^:private cnt 60)
(def ^:private groups 7)
(def ^:private group (let [r (java.util.Random. 3)] (long-array (repeatedly cnt #(.nextInt r groups)))))
(def ^:private x (let [r (java.util.Random. 4)] (double-array (repeatedly cnt #(.nextGaussian r)))))
(def ^:private y (let [r (java.util.Random. 5)] (double-array (repeatedly cnt #(.nextGaussian r)))))
(def ^:private alpha (double-array (map #(* 0.3 (Math/sin %)) (range groups))))

(defn- lp [mu tau beta alpha]
  (varying-intercepts mu tau beta alpha group x y cnt groups))

(defn- finite-difference [k]
  (let [h 1e-6
        at (fn [d]
             (if (< k 3)
               (apply lp (conj (update [0.2 0.8 -0.4] k + d) alpha))
               (let [a (aclone ^doubles alpha)]
                 (aset a (- k 3) (+ (aget a (- k 3)) d))
                 (lp 0.2 0.8 -0.4 a))))]
    (/ (- (at h) (at (- h))) (* 2 h))))

(defn- check-gradient [vg]
  (let [[v dmu dtau dbeta dalpha] (vg 0.2 0.8 -0.4 alpha group x y cnt groups)]
    (is (< (Math/abs (- v (lp 0.2 0.8 -0.4 alpha))) 1e-12))
    (doseq [[k g] (map vector (range) (concat [dmu dtau dbeta] (seq dalpha)))]
      (testing k
        (is (< (Math/abs (- g (finite-difference k))) 1e-5))))))

(deftest gather-read-in-a-carry-loop-matches-finite-differences
  (testing "evaluated" (check-gradient (rev/value+grad #'varying-intercepts :wrt [0 1 2 3])))
  (testing "compiled"
    (let [vg (rev/value+grad #'varying-intercepts :wrt [0 1 2 3] :compile? true)]
      (is (::rev/compiled? (meta vg)) (str (::rev/compile-failure (meta vg))))
      (check-gradient vg))))

(deftest gather-read-takes-the-scan-pullback
  (let [{:keys [walked-body]} (#'rev/build-grad-walked-body #'varying-intercepts [0 1 2 3])
        heads (atom #{})]
    (walk/postwalk (fn [f] (when (seq? f) (swap! heads conj (first f))) f) walked-body)
    (is (not (contains? @heads 'java.util.ArrayList.)) "no closure tape")
    (is (not (contains? @heads 'fn*)) "no per-iteration pullbacks")
    (is (not (contains? @heads 'raster.par/scan)) "an additive likelihood keeps no carry tape")))

;; term + acc (accumulator on the right) and a product recurrence, which is not
;; additive and keeps the scan's carry tape
(deftm right-sum [w :- Double, xs :- (Array double), cnt :- Long] :- Double
  (loop [i 0 acc 0.0]
    (if (< i cnt)
      (recur (inc i) (n/+ (n/* w (ra/aget xs i)) acc))
      acc)))

(deftm product-recurrence [w :- Double, xs :- (Array double), cnt :- Long] :- Double
  (loop [i 0 acc 1.0]
    (if (< i cnt)
      (recur (inc i) (n/* acc (n/+ 1.0 (n/* w (ra/aget xs i)))))
      acc)))

(deftest additive-and-multiplicative-recurrences
  (let [xs (double-array [0.5 -1.0 2.0])
        fd (fn [f w] (/ (- (f (+ w 1e-6) xs 3) (f (- w 1e-6) xs 3)) 2e-6))]
    (doseq [[f v] [[right-sum #'right-sum] [product-recurrence #'product-recurrence]]
            compile? [false true]]
      (let [[_ dw] ((rev/value+grad v :wrt [0] :compile? compile?) 0.3 xs 3)]
        (testing [v compile?]
          (is (< (Math/abs (- dw (fd f 0.3))) 1e-6)))))))
