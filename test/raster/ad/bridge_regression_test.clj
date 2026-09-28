(ns raster.ad.bridge-regression-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]
            [raster.sci.distributions :as dist]))

(deftm normal-density-sum [mu :- Double, sigma :- Double,
                           ys :- (Array double), count :- Long] :- Double
  (loop [i 0 sum 0.0]
    (if (< i count)
      (recur (inc i)
             (n/+ sum (dist/logpdf (dist/->Normal mu sigma)
                                   (aget ys i))))
      sum)))

(deftm normal-density-reduce [mu :- Double, sigma :- Double,
                              ys :- (Array double), count :- Long] :- Double
  (par/reduce sum 0.0 i count
    (n/+ sum (dist/logpdf (dist/->Normal mu sigma) (aget ys i)))))

(deftest generated-gradient-helper-namespace-is-available
  (is (some? (find-ns 'raster.dl.nn))))

(deftest constructed-normal-in-reduction-has-gradient
  (let [ys (double-array [0.1 0.7 -0.3])
        [value dmu dsigma _] ((rev/value+grad #'normal-density-sum)
                               0.2 1.4 ys 3)
        expected (reduce + (map #(dist/logpdf (dist/->Normal 0.2 1.4) %) ys))
        expected-dmu (reduce + (map #(/ (- % 0.2) (* 1.4 1.4)) ys))
        expected-dsigma
        (reduce + (map #(let [delta (- % 0.2)]
                          (+ (- (/ 1.0 1.4))
                             (/ (* delta delta) (* 1.4 1.4 1.4)))) ys))]
    (doseq [[f [v dm ds _]]
            [["lifted loop" [value dmu dsigma nil]]
             ["explicit reduce" ((rev/value+grad #'normal-density-reduce)
                                  0.2 1.4 ys 3)]]]
      (is (< (Math/abs (- v expected)) 1e-10) f)
      (is (< (Math/abs (- dm expected-dmu)) 1e-9) f)
      (is (< (Math/abs (- ds expected-dsigma)) 1e-9) f))))
