(ns raster.ad.conditional-replay-projection-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]))

(deftm nested-selected-replay
  [x :- Double observations :- (Array double) destination :- (Array double)
   counts :- (Array long) outer :- Long inner :- Long] :- Double
  (let [value (if (= outer 1)
                (let [tick (aset counts 0 (inc (aget counts 0)))]
                  (if (= inner 1)
                    (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget observations k))))
                    (n/* x 2.0)))
                (n/* x 3.0))
        changed (aset destination 0 99.0)]
    value))

(deftm nested-cancellation-checkpoint [x :- Double cnt :- Long] :- Double
  (loop [i 1 acc 0.0]
    (if (< i cnt)
      (let [value (loop [j 1 a 0.0 b 0.0]
                    (if (< j 2)
                      (recur (inc j) (n/* x 1.0e16) (n/* x -1.0e16))
                      (n/+ a b)))]
        (recur (inc i) (n/+ acc (n/+ value x))))
      acc)))

(deftest residual-condition-projection-does-not-change-adjoint-fold-order
  (is (= [1.0 0.0 nil]
         ((reverse/value+grad #'nested-cancellation-checkpoint :wrt [0]) 1.0 2))))

(deftest nested-selection-exports-evaluated-replay-obligations
  (doseq [compile? [false true]]
   (let [gradient (reverse/value+grad #'nested-selected-replay :wrt [0] :compile? compile?)]
    (doseq [[outer inner coefficient effects] [[0 0 3.0 0] [0 1 3.0 0] [1 0 2.0 1]]]
      (let [shared (double-array [3.0]) counts (long-array 1)]
        (is (= [(n/* 2.0 coefficient) coefficient nil nil nil nil nil]
               (gradient 2.0 shared shared counts outer inner)))
        (is (= [99.0] (vec shared)))
        (is (= [effects] (vec counts)))))
    (let [observations (double-array [3.0]) destination (double-array [7.0])
          counts (long-array 1)]
      (is (= [6.0 3.0 nil nil nil nil nil] (gradient 2.0 observations destination counts 1 1)))
      (is (= [3.0] (vec observations)))
      (is (= [99.0] (vec destination)))
      (is (= [1] (vec counts))))
    (let [shared (double-array [3.0]) counts (long-array 1)]
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 shared shared counts 1 1) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [3.0] (vec shared)))
      ;; The selected primal executes once before the later write is fenced.
      (is (= [1] (vec counts)))))))

(deftest conditional-records-cannot-fall-back-to-flattened-selector-summaries
  (is (= :ad-region-replay-projection
         (try (#'reverse/record-replay-conditions {:type :if-region :sym 'value} {} {}) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
