(ns raster.ad.ordered-loop-initializer-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.numeric :as n]))

(deftm origin-before-active-initializer
  [x :- Double, divisor :- Long, data :- (Array double)] :- Double
  (loop [i (quot 1 divisor) acc (n/* x (aget data 0))]
    (if (< i 3) (recur (inc i) (n/+ acc x)) acc)))

(deftm active-initializer-before-origin
  [x :- Double, divisor :- Long, data :- (Array double)] :- Double
  (loop [acc (n/* x (aget data 0)) i (quot 1 divisor)]
    (if (< i 3) (recur (n/+ acc x) (inc i)) acc)))

(deftm initializer-reads-earlier-binding [x :- Double, origin :- Long] :- Double
  (loop [i origin acc (n/* x (double i))]
    (if (< i 3) (recur (inc i) (n/+ acc x)) acc)))

(deftm dependent-active-initializers [x :- Double, cnt :- Long] :- Double
  (loop [i 1 acc (n/* 2.0 x) aux (n/* acc x)]
    (if (< i cnt)
      (recur (inc i) (n/+ acc aux) (n/+ aux acc))
      acc)))

(deftm zero-trip-initializer-chain [x :- Double, cnt :- Long] :- Double
  (loop [i 1 a (n/* 2.0 x) b (n/* 3.0 a)]
    (if (< i cnt) (recur (inc i) a b) b)))

(deftest reverse-tape-initializers-keep-lexical-order
  (doseq [[v expected]
          [[#'origin-before-active-initializer ArithmeticException]
           [#'active-initializer-before-origin ArrayIndexOutOfBoundsException]]]
    (let [gradient (rev/value+grad v :wrt [0])
          exception-class (fn [f] (try (f 2.0 0 (double-array 0)) nil
                                      (catch Throwable e (class e))))]
      (is (= expected (exception-class @v)))
      (is (= expected (exception-class gradient)))))
  (let [gradient (rev/value+grad #'initializer-reads-earlier-binding :wrt [0])]
    (doseq [origin [1 2 3 4]]
      (let [coefficient (max 3 origin)]
        (is (= (* 2.0 coefficient) (initializer-reads-earlier-binding 2.0 origin)))
        (is (= [(* 2.0 coefficient) (double coefficient) nil]
               (gradient 2.0 origin)))))))

(deftest reverse-tape-initializer-substitution-keeps-active-dependencies
  (let [gradient (rev/value+grad #'dependent-active-initializers :wrt [0])]
    (doseq [[x cnt expected-value expected-gradient]
            [[2.0 1 4.0 2.0] [2.0 2 12.0 10.0] [3.0 2 24.0 14.0]]]
      (is (= expected-value (dependent-active-initializers x cnt)))
      (is (= [expected-value expected-gradient nil] (gradient x cnt)))))
  (let [gradient (rev/value+grad #'zero-trip-initializer-chain :wrt [0])]
    (doseq [x [2.0 3.0]]
      (is (= (* 6.0 x) (zero-trip-initializer-chain x 0)))
      (is (= [(* 6.0 x) 6.0 nil] (gradient x 0))))))
