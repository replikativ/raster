(ns raster.ad.bridge-regression-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
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
    (is (< (Math/abs (- value expected)) 1e-10))
    (is (< (Math/abs (- dmu expected-dmu)) 1e-9))
    (is (< (Math/abs (- dsigma expected-dsigma)) 1e-9))))
