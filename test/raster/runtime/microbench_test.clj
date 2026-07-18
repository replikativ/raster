(ns raster.runtime.microbench-test
  "Phase-1: the measurement layer. do-bench methodology + the :measured overlay into the descriptor.
   Timing-value assertions are machine-independent (positive/finite/stationary-shape only)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.runtime.hardware :as rt]
            [raster.runtime.microbench :as mb]
            [raster.compiler.core.hardware :as chw]))

(deftest do-bench-shape
  (testing "do-bench returns the stationary-machine statistics over a fixed-work thunk"
    (let [work (fn [] (loop [i 0 acc 0.0] (if (< i 10000) (recur (inc i) (+ acc (Math/sqrt i))) acc)))
          r (mb/do-bench work :warmup-ms 5 :budget-ms 20)]
      (is (>= (:n r) 3) "at least the minimum sample count")
      (is (pos? (:median-ns r)))
      (is (pos? (:min-ns r)))
      (is (<= (:min-ns r) (:median-ns r)) "min ≤ median")
      (is (number? (:cv r)))
      (is (boolean? (:stationary? r))))))

(deftest cpu-bandwidth-is-positive
  (testing "the STREAM triad yields a positive, finite bandwidth"
    (let [r (mb/measure-cpu-bandwidth :llc-bytes (* 8 1024 1024))]
      (is (pos? (:bandwidth-bytes-s r)))
      (is (Double/isFinite (:bandwidth-bytes-s r)))
      (is (= (* 3.0 (:n-elements r) 8.0) (:bytes-moved r))))))

(deftest measured-overlay-overrides-and-tags
  (testing "a stored :measured map overrides the descriptor's bandwidth and carries provenance"
    (rt/init!)
    (rt/set-measured! :cpu:0 {:bandwidth-bytes-s 1.37e11
                              :peak-flops {:f32 5.0e11}
                              :provenance {:bandwidth-bytes-s :measured :peak-flops :measured}})
    (try
      (let [d (chw/descriptor-for :cpu:0)]
        (is (= 1.37e11 (:bandwidth-bytes-s d)) "measured bandwidth overrides")
        (is (= 5.0e11 (chw/peak-flops-for d :f32)) "measured peak-flops merged")
        (is (= :measured (get-in d [:provenance :bandwidth-bytes-s])))
        (is (map? (:measured d)) "the raw measured map is kept for inspection")
        (testing "and the roofline ridge now derives from the measured numbers"
          (is (< 3.0 (chw/balance-for d :f32) 4.0) "5e11/1.37e11 ≈ 3.65")))
      (finally (rt/set-measured! :cpu:0 nil)))))
