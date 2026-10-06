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

(deftest a-shifted-map-read-is-rejected
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"par/map! AD"
                        ((rev/value+grad #'shifted-map :wrt [0]) (double-array [1.0 2.0 3.0]) 2))))

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
