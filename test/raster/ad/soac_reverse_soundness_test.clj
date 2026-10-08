(ns raster.ad.soac-reverse-soundness-test
  "Reverse mode through par/map! and par/reduce keeps primal effects, sees an
  active initializer, and transposes every admitted read; reads it cannot
  transpose are rejected instead of silently mis-scattered."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.par :as par]))

(defn- close? [a b] (< (Math/abs (- (double a) (double b))) 1e-9))

;; Additive shape does not license replaying an effectful step. The ordered
;; closure tape must retain each forward read instead of executing its write
;; again in backward.
(deftm effectful-additive-step
  [w :- Double, scratch :- (Array double), m :- Long] :- Double
  (par/reduce acc 0.0 i m
    (let [r (ra/aget scratch 0)
          _ (ra/aset scratch 0 (n/+ r 1.0))]
      (n/+ acc (n/* w r)))))

(deftest additive-effects-execute-only-in-forward
  (doseq [m [0 1 3]]
    (let [scratch (double-array [0.0])
          [value dw] ((rev/value+grad #'effectful-additive-step :wrt [0]) 2.0 scratch m)
          sum (/ (* m (dec m)) 2.0)]
      (testing m
        (is (close? (* 2.0 sum) value))
        ;; A zero-trip closure tape produces the existing absent/zero tangent.
        (if (zero? m) (is (nil? dw)) (is (close? sum dw)))
        (is (= [(double m)] (vec scratch)))))))

;; A map with nothing to differentiate still fills its output.
(deftm constant-fill-plus [x :- Double, m :- Long] :- Double
  (let [out (double-array m)]
    (par/map! out i m double 3.0)
    (n/+ x (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (ra/aget out i))) s)))))

(deftest an-inactive-map-keeps-its-forward-fill
  (let [[v dx] ((rev/value+grad #'constant-fill-plus :wrt [0]) 2.0 4)]
    (is (close? 14.0 v))
    (is (close? 1.0 dx))))

;; The reduction's value depends on its initializer even when the step does not
;; read an active value.
(deftm count-from [h0 :- Double, m :- Long] :- Double
  (let [r (par/reduce acc h0 i m (n/+ acc 1.0))]
    (n/* 2.0 r)))

(deftest a-reduction-is-active-through-its-initializer
  (doseq [m [0 3]]
    (let [[v dh0] ((rev/value+grad #'count-from :wrt [0]) 0.5 m)]
      (testing m
        (is (close? (* 2.0 (+ 0.5 m)) v))
        (is (close? 2.0 dh0))))))

;; Two reads of one array at the map index: both contributions count.
(deftm squared-map [xs :- (Array double), m :- Long] :- Double
  (let [out (double-array m)]
    (par/map! out i m double (let [a (ra/aget xs i) b (ra/aget xs i)] (n/* a b)))
    (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (ra/aget out i))) s))))

(deftest repeated-reads-of-one-array-accumulate
  (let [xs (double-array [1.0 -2.0 3.0])
        [v dxs] ((rev/value+grad #'squared-map :wrt [0]) xs 3)]
    (is (close? 14.0 v))
    (is (= [2.0 -4.0 6.0] (vec dxs)))))

;; A read at another lane would need a conflict-free scatter in the parallel
;; backward; it is rejected rather than written to the wrong element.
(deftm shifted-map [xs :- (Array double), m :- Long] :- Double
  (let [out (double-array m)]
    (par/map! out i m double (let [a (ra/aget xs (inc i))] (n/* 2.0 a)))
    (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (ra/aget out i))) s))))

(deftest a-shifted-map-read-scatters-its-cotangent
  (doseq [compile? [false true]]
    (let [[v dxs] ((rev/value+grad #'shifted-map :wrt [0] :compile? compile?) (double-array [1.0 2.0 3.0]) 2)]
      (is (close? 10.0 v))
      (is (= [0.0 2.0 2.0] (vec dxs))))))

;; An index computed by the step itself cannot be recomputed in the backward map.
(deftm locally-indexed-map [xs :- (Array double), m :- Long] :- Double
  (let [out (double-array m)]
    (par/map! out i m double (let [k (ra/aget xs i) a (ra/aget xs (long k))] (n/* 2.0 a)))
    (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (ra/aget out i))) s))))

(deftest a-map-read-at-a-step-local-index-is-rejected
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"par/map! AD"
                        ((rev/value+grad #'locally-indexed-map :wrt [0]) (double-array [1.0 0.0]) 2))))

;; Varying intercepts as a map: lanes gather alpha[group[i]], repeated groups
;; accumulate.
(deftm intercept-map [alpha :- (Array double), group :- (Array long), x :- (Array double), m :- Long] :- Double
  (let [out (double-array m)]
    (par/map! out i m double (let [r (n/- (ra/aget x i) (ra/aget alpha (ra/aget group i)))] (n/* r r)))
    (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (ra/aget out i))) s))))

(deftest a-gathered-map-read-accumulates
  (doseq [compile? [false true]]
    (let [alpha (double-array [0.5 -1.0])
          group (long-array [0 1 1 0])
          x (double-array [1.0 2.0 0.0 -1.0])
          [v d-alpha] ((rev/value+grad #'intercept-map :wrt [0] :compile? compile?) alpha group x 4)]
      (testing compile?
        ;; residuals 0.5, 3.0, 1.0, -1.5
        (is (close? (+ 0.25 9.0 1.0 2.25) v))
        ;; d alpha_g = -2 Σ residuals in group g
        (is (= [(* -2.0 (+ 0.5 -1.5)) (* -2.0 (+ 3.0 1.0))] (vec d-alpha)))))))

;; dotimes: two reads of one array accumulate into its shadow.
(deftm squared-dotimes [xs :- (Array double), m :- Long] :- Double
  (let [out (double-array m)]
    (dotimes [i m]
      (let [a (ra/aget xs i) b (ra/aget xs i)]
        (ra/aset out i (n/* a b))))
    (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (ra/aget out i))) s))))

(deftest repeated-dotimes-reads-accumulate
  (let [xs (double-array [1.0 -2.0 3.0])
        [v dxs] ((rev/value+grad #'squared-dotimes :wrt [0]) xs 3)]
    (is (close? 14.0 v))
    (is (= [2.0 -4.0 6.0] (vec dxs)))))

;; A pullback that re-reads an array needs the array's forward contents.
(deftm sum-then-overwrite [w :- Double, xs :- (Array double), m :- Long] :- Double
  (let [s (loop [i 0 acc 0.0] (if (< i m) (recur (inc i) (n/+ acc (n/* w (ra/aget xs i)))) acc))
        _ (ra/aset xs 0 100.0)]
    s))

(deftm sum-then-write-fresh [w :- Double, xs :- (Array double), m :- Long] :- Double
  (let [s (loop [i 0 acc 0.0] (if (< i m) (recur (inc i) (n/+ acc (n/* w (ra/aget xs i)))) acc))
        ys (double-array m)
        _ (ra/aset ys 0 100.0)]
    (n/+ s (ra/aget ys 0))))

(deftest a-replayed-array-is-not-overwritten
  (is (= :replayed-array-overwritten
         (:reason (ex-data (try ((rev/value+grad #'sum-then-overwrite :wrt [0]) 0.5 (double-array [1.0 2.0]) 2)
                                (catch clojure.lang.ExceptionInfo e e))))))
  (let [[v dw] ((rev/value+grad #'sum-then-write-fresh :wrt [0]) 0.5 (double-array [1.0 2.0]) 2)]
    (is (close? 101.5 v))
    (is (close? 3.0 dw))))

(deftm sum-then-overwrite-alias
  [w :- Double, xs :- (Array double), m :- Long] :- Double
  (let [ys xs zs ys
        s (par/reduce acc 0.0 i m (n/+ acc (n/* w (ra/aget xs i))))
        _ (ra/aset zs 0 100.0)]
    s))

(deftest replay-protection-follows-inactive-transitive-aliases
  (is (= :replayed-array-overwritten
         (:reason (ex-data
                    (try ((rev/value+grad #'sum-then-overwrite-alias :wrt [0])
                           0.5 (double-array [1.0 2.0]) 2)
                         (catch clojure.lang.ExceptionInfo e e)))))))

(deftm mapped-then-overwrite-output [w :- Double] :- Double
  (let [out (double-array 1)
        mapped (par/map! out i 1 double 2.0)
        s (par/reduce acc 0.0 i 1 (n/+ acc (n/* w (ra/aget mapped i))))
        _ (ra/aset out 0 100.0)]
    s))

(deftest operation-result-aliases-share-replay-storage
  (is (= :replayed-array-overwritten
         (:reason (ex-data
                    (try ((rev/value+grad #'mapped-then-overwrite-output :wrt [0]) 0.5)
                         (catch clojure.lang.ExceptionInfo e e)))))))

(deftm gather-then-change-index
  [xs :- (Array double), index :- (Array int)] :- Double
  (let [out (double-array 1)
        _ (par/gather out xs index 1)
        changed index
        _ (ra/aset changed 0 1)]
    (n/* (ra/aget out 0) (ra/aget out 0))))

(deftm mapped-then-change-index
  [xs :- (Array double), index :- (Array int)] :- Double
  (let [out (double-array 1)
        _ (par/map! out i 1 double (ra/aget xs (ra/aget index i)))
        _ (ra/aset index 0 1)]
    (ra/aget out 0)))

(deftm dotimes-then-change-index
  [xs :- (Array double), index :- (Array int)] :- Double
  (let [out (double-array 1)
        _ (dotimes [i 1] (ra/aset out i (ra/aget xs (ra/aget index i))))
        _ (ra/aset index 0 1)]
    (ra/aget out 0)))

(deftm dotimes-core-then-change-index
  [xs :- (Array double), index :- (Array int)] :- Double
  (let [out (double-array 1)
        _ (dotimes [i 1] (let [v (aget xs (aget index i))] (aset out i v)))
        _ (aset index 0 1)]
    (aget out 0)))

(deftm ordered-float-then-change-index
  [xs :- (Array float), index :- (Array int)] :- Float
  (let [s (par/reduce acc (float 1.0) i 1
            (n/* acc (ra/aget xs (aget index i))))
        _ (aset index 0 1)]
    s))

(deftm ordered-float-step-changes-index
  [xs :- (Array float), index :- (Array int)] :- Float
  (par/reduce acc (float 1.0) i 1
    (let [v (ra/aget xs (aget index i))
          _ (aset index i 1)]
      (n/* acc v))))

(deftest transposed-indices-retain-their-forward-storage
  (doseq [[f xs] [[#'gather-then-change-index (double-array [2.0 7.0])]
                  [#'mapped-then-change-index (double-array [2.0 7.0])]
                  [#'dotimes-core-then-change-index (double-array [2.0 7.0])]
                  [#'ordered-float-then-change-index (float-array [2.0 7.0])]
                  [#'ordered-float-step-changes-index (float-array [2.0 7.0])]]]
    (testing (str f)
      (is (= :replayed-array-overwritten
             (:reason (ex-data
                        (try ((rev/value+grad f :wrt [0])
                               xs (int-array [0]))
                             (catch clojure.lang.ExceptionInfo e e)))))))))

(deftest captured-dotimes-indices-do-not-need-replay-protection
  ;; This form's inline call pullback captures its index during forward,
  ;; unlike the separately analyzed read transposed above. Do not reject a
  ;; safe mutation just because another representation needs index replay.
  (let [[v dx] ((rev/value+grad #'dotimes-then-change-index :wrt [0])
                (double-array [2.0 7.0]) (int-array [0]))]
    (is (= 2.0 v))
    (is (= [1.0 0.0] (vec dx)))))

;; par/gather's pullback is the scatter-add of its output cotangent.
(deftm gathered-squares [alpha :- (Array double), group :- (Array int), m :- Long] :- Double
  (let [g (double-array m)]
    (par/gather g alpha group m)
    (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (n/* (ra/aget g i) (ra/aget g i)))) s))))

(deftest a-gather-transposes-to-a-scatter-add
  (doseq [compile? [false true]]
    (let [alpha (double-array [1.0 -2.0 3.0])
          group (int-array [0 2 2 1 0])
          vg (rev/value+grad #'gathered-squares :wrt [0] :compile? compile?)
          [v d-alpha] (vg alpha group 5)]
      (testing compile?
        (is (= compile? (boolean (::rev/compiled? (meta vg)))) (str (::rev/compile-failure (meta vg))))
        (is (close? (+ 1.0 9.0 9.0 4.0 1.0) v))
        (is (= [4.0 -4.0 12.0] (vec d-alpha)))))))

;; dotimes reading a gathered element scatters its cotangent there.
(deftm intercept-dotimes [alpha :- (Array double), group :- (Array long), x :- (Array double), m :- Long] :- Double
  (let [out (double-array m)]
    (dotimes [i m]
      (let [a (ra/aget alpha (ra/aget group i))]
        (ra/aset out i (n/* (n/- (ra/aget x i) a) (n/- (ra/aget x i) a)))))
    (loop [i 0 s 0.0] (if (< i m) (recur (inc i) (n/+ s (ra/aget out i))) s))))

(deftest a-gathered-dotimes-read-accumulates
  (let [[v d-alpha] ((rev/value+grad #'intercept-dotimes :wrt [0])
                     (double-array [0.5 -1.0]) (long-array [0 1 1 0]) (double-array [1.0 2.0 0.0 -1.0]) 4)]
    (is (close? (+ 0.25 9.0 1.0 2.25) v))
    (is (= [(* -2.0 (+ 0.5 -1.5)) (* -2.0 (+ 3.0 1.0))] (vec d-alpha)))))

;; An ordered (non-additive) reduction gathering alpha[group[i]].
(deftm gathered-product [alpha :- (Array double), group :- (Array long), m :- Long] :- Double
  (par/reduce acc 1.0 i m (n/* acc (n/+ 1.0 (ra/aget alpha (ra/aget group i))))))

(deftest an-ordered-reduction-scatters-gathered-reads
  (let [alpha (double-array [0.5 -0.25 0.1])
        group (long-array [0 2 0 1])
        f (fn [a] (gathered-product a group 4))
        [v d-alpha] ((rev/value+grad #'gathered-product :wrt [0]) alpha group 4)]
    (is (close? (f alpha) v))
    (doseq [k (range 3)]
      (let [bump (fn [d] (let [a (aclone alpha)] (aset a k (+ (aget a k) d)) (f a)))]
        (is (< (Math/abs (- (aget ^doubles d-alpha k) (/ (- (bump 1e-6) (bump -1e-6)) 2e-6))) 1e-6))))))
