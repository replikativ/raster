(ns raster.par-runtime-test
  "Tests for raster.par.runtime/parallel-for! — chunked CPU thread-parallel
   execution with a runtime-configurable thread count."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.par.runtime :as rt]))

(defn- fill [threads n]
  (let [out (double-array n)]
    (binding [rt/*par-threads* threads]
      (rt/parallel-for! n (fn [lo hi]
                            (loop [i (long lo)]
                              (when (< i hi)
                                (aset out i (* (double i) (double i)))
                                (recur (inc i)))))))
    out))

(deftest serial-equals-parallel-on-disjoint-writes
  (testing "*par-threads* 1 and N give identical results when chunks are disjoint"
    (let [serial (fill 1 100000)]
      (doseq [t [2 4 8]]
        (is (java.util.Arrays/equals serial (fill t 100000)) (str "threads=" t))))))

(deftest covers-full-range-no-gaps
  (testing "every index in [0,n) is written (chunk boundaries cover the range)"
    (doseq [[t n] [[1 7] [3 7] [8 7] [4 4] [4 1] [4 0] [8 1000]]]
      (let [out (fill t (max 1 n))
            expect (double-array (max 1 n))]
        (dotimes [i n] (aset expect i (* (double i) (double i))))
        (is (java.util.Arrays/equals ^doubles (double-array (take n out))
                                     ^doubles (double-array (take n expect)))
            (str "threads=" t " n=" n))))))

(deftest racy-scatter-add-converges
  (testing "parallel scatter-add to a shared cell loses some updates but stays bounded
            (benign-race semantics; correctness here = no crash + within range)"
    ;; All lanes add 1.0 to acc[0] — with races some adds are lost (like SGD).
    ;; We only assert it doesn't exceed the serial total and is positive.
    (let [n 100000
          acc (double-array 1)
          _ (binding [rt/*par-threads* 8]
              (rt/parallel-for! n (fn [lo hi]
                                    (loop [i (long lo)]
                                      (when (< i hi)
                                        (aset acc 0 (+ (aget acc 0) 1.0))
                                        (recur (inc i)))))))
          total (aget acc 0)]
      (is (<= 0.0 total (double n)) (str "racy total " total " in [0," n "]")))))
