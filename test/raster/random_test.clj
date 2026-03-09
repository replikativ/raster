(ns raster.random-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.random :as rng]))

(deftest rand-int-test
  (testing "uniform random int in [0, bound)"
    (let [r (java.util.Random. 42)]
      (dotimes [_ 100]
        (let [v (rng/rand-int r 10)]
          (is (>= v 0))
          (is (< v 10)))))))

(deftest rand-double-test
  (testing "uniform random double in [origin, bound)"
    (let [r (java.util.Random. 42)]
      (dotimes [_ 100]
        (let [v (rng/rand-double r 1.0 5.0)]
          (is (>= v 1.0))
          (is (< v 5.0)))))))

(deftest rand-gaussian-test
  (testing "standard normal — mean near 0, finite"
    (let [r (java.util.Random. 42)
          samples (double-array 1000)]
      (dotimes [i 1000]
        (aset samples i (rng/rand-gaussian r)))
      (let [mean (/ (reduce + (seq samples)) 1000.0)]
        (is (< (Math/abs mean) 0.2) "mean should be near 0")))))

(deftest randn!-test
  (testing "fills double array with normal samples"
    (let [r (java.util.Random. 42)
          out (double-array 100)]
      (rng/randn! r out)
      (is (every? #(Double/isFinite %) out))
      (is (not (every? zero? out)))))

  (testing "fills float array with normal samples"
    (let [r (java.util.Random. 42)
          out (float-array 100)]
      (rng/randn! r out)
      (is (every? #(Float/isFinite (float %)) out))
      (is (not (every? zero? out))))))

(deftest rand!-test
  (testing "fills double array with uniform [0,1) samples"
    (let [r (java.util.Random. 42)
          out (double-array 100)]
      (rng/rand! r out)
      (is (every? #(and (>= % 0.0) (< % 1.0)) out))))

  (testing "fills float array with uniform [0,1) samples"
    (let [r (java.util.Random. 42)
          out (float-array 100)]
      (rng/rand! r out)
      (is (every? #(let [v (float %)] (and (>= v 0.0) (< v 1.0))) out)))))

(deftest fill-seeds!-test
  (testing "fills long array with random seeds"
    (let [r (java.util.Random. 42)
          seeds (long-array 10)]
      (rng/fill-seeds! r seeds 10)
      (is (not (every? zero? seeds))))))

(deftest deterministic-test
  (testing "same seed produces same sequence"
    (let [r1 (java.util.Random. 123)
          r2 (java.util.Random. 123)
          a1 (double-array 10)
          a2 (double-array 10)]
      (rng/randn! r1 a1)
      (rng/randn! r2 a2)
      (is (= (seq a1) (seq a2))))))
