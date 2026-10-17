(ns raster.ad.nested-reduction-call-test
  "External MMM boundary: retain analytic oracles for future helper composition
   and refuse an unowned derived pullback before runtime execution."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as ra]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]))

(deftm scalar-pair [a :- Double b :- Double xs :- (Array double) cnt :- Long] :- Double
  (par/reduce acc 0.0 i cnt (n/+ acc (n/* a (n/+ b (ra/aget xs i))))))

(deftm vector-pair [theta :- (Array double) xs :- (Array double) cnt :- Long] :- Double
  (let [a (ra/aget theta 0) b (ra/aget theta 1)]
    (scalar-pair a b xs cnt)))

(deftm inline-pair [theta :- (Array double) xs :- (Array double) cnt :- Long] :- Double
  (let [a (ra/aget theta 0) b (ra/aget theta 1)]
    (par/reduce acc 0.0 i cnt (n/+ acc (n/* a (n/+ b (ra/aget xs i)))))))

(deftest direct-and-inline-reduction-analytic-oracles
  (let [inline (reverse/value+grad #'inline-pair :wrt [0])
        all (reverse/value+grad #'inline-pair)
        direct (reverse/value+grad #'scalar-pair :wrt [0 1])]
    (doseq [[theta values] [[[2.0 3.0] [1.0 2.0]]
                            [[-1.5 0.25] [4.0 -2.0 3.0]]
                            [[4.0 -3.0] []]]]
      (let [th (double-array theta) xs (double-array values) cnt (count values)
            [a b] theta
            total (reduce + 0.0 (map #(+ b %) values))
            expected (* a total)
            gradient [total (* a cnt)]
            inline-result (inline th xs cnt)
            all-result (all th xs cnt)
            direct-result (direct a b xs cnt)]
        (is (= expected (vector-pair th xs cnt) (first inline-result)))
        (is (= gradient (vec (second inline-result))))
        (is (= [expected (first gradient) (second gradient) nil nil] direct-result))
        (is (= gradient (vec (second all-result))))
        (is (= (vec (repeat cnt a)) (vec (nth all-result 2))))
        (is (nil? (nth all-result 3)))
        (is (= theta (vec th)))
        (is (= values (vec xs)))))))

(deftest unowned-helper-pullback-refuses-during-derivation-not-at-invocation
  ;; This is a safety regression, NOT nested-helper AD acceptance. Keep the
  ;; numerical oracle above for the future forward-owned residual implementation.
  (doseq [options [[] [:wrt [0]]]]
    (let [error (try (apply reverse/value+grad #'vector-pair options)
                     (catch clojure.lang.ExceptionInfo e e))]
      (is (instance? clojure.lang.ExceptionInfo error))
      (is (= :ad-unowned-derived-pullback (:reason (ex-data error))))
      (is (seq (:active (ex-data error))))
      (is (seq (:form (ex-data error)))))))
