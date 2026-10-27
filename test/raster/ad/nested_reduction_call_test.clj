(ns raster.ad.nested-reduction-call-test
  "External MMM boundary: analytic and finite-difference oracles for helper
   composition over retained array roots and completed scalar inputs."
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

(deftm compound-vector-pair [theta :- (Array double) xs :- (Array double) cnt :- Long] :- Double
  (let [a (ra/aget theta 0)]
    (scalar-pair (n/* a a) 0.5 xs cnt)))

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

(deftest owned-helper-pullbacks-match-analytic-and-finite-difference-oracles
  ;; Stable retained array ports now discharge their redundant CHECKCASTs.
  ;; This replaces the former refusal expectation with numerical acceptance;
  ;; unverified/compound reference refusals remain in owned-inactive-input-test.
  (doseq [[source compound?] [[#'vector-pair false] [#'compound-vector-pair true]]
          selected? [false true]]
    (let [gradient (if selected? (reverse/value+grad source :wrt [0])
                       (reverse/value+grad source))]
      (doseq [[theta values] [[[2.0 3.0] [1.0 2.0]]
                              [[-1.5 0.25] [4.0 -2.0 3.0]]
                              [[4.0 -3.0] []]]]
        (let [[a b] theta
              th (double-array theta) xs (double-array values) cnt (count values)
              total (reduce + 0.0 (map #(+ (if compound? 0.5 b) %) values))
              scale (if compound? (* a a) a)
              expected (* scale total)
              expected-gradient (if compound? [(* 2.0 a total) 0.0] [total (* a cnt)])
              [value actual-gradient data-gradient count-gradient] (gradient th xs cnt)]
          (is (= expected (source th xs cnt) value))
          (is (= expected-gradient (vec actual-gradient)))
          (is (nil? count-gradient))
          (if selected?
            (is (nil? data-gradient))
            (is (= (vec (repeat cnt scale)) (vec data-gradient))))
          (doseq [coordinate (range 2)]
            (let [plus (aclone th) minus (aclone th) h 1.0e-5]
              (aset plus coordinate (+ (aget th coordinate) h))
              (aset minus coordinate (- (aget th coordinate) h))
              (is (< (Math/abs (- (aget actual-gradient coordinate)
                                  (/ (- (source plus xs cnt) (source minus xs cnt)) (* 2.0 h))))
                     1.0e-7))))
          (is (= theta (vec th)))
          (is (= values (vec xs))))))))
