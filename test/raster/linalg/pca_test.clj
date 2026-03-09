(ns raster.linalg.pca-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.linalg.pca :as pca]
            [raster.linalg.lapack :as lapack]))

(use-fixtures :once
  (fn [f] (if (lapack/available?) (f) (println "[SKIP] No LAPACK library"))))

(defn- approx=
  [a b tol]
  (< (Math/abs (- (double a) (double b))) tol))

(defn- make-test-data
  "Create a simple test dataset: 2 clear principal directions + noise.
  Returns [X, m, n]."
  [m]
  (let [n 5
        rng (java.util.Random. 42)
        X (double-array (* m n))]
    (dotimes [i m]
      (let [t1 (.nextGaussian rng)
            t2 (* 0.5 (.nextGaussian rng))]
        ;; Strong signal in first two directions
        (aset X (+ (* i n) 0) (+ (* 3.0 t1) (* 0.1 (.nextGaussian rng))))
        (aset X (+ (* i n) 1) (+ (* 2.0 t1) (* 0.1 (.nextGaussian rng))))
        (aset X (+ (* i n) 2) (+ (* 1.5 t2) (* 0.1 (.nextGaussian rng))))
        (aset X (+ (* i n) 3) (* 0.1 (.nextGaussian rng)))
        (aset X (+ (* i n) 4) (* 0.1 (.nextGaussian rng)))))
    [X m n]))

(deftest pca-full-basic-test
  (testing "PCA full solver: explained variance ratios sum to <= 1"
    (let [[X m n] (make-test-data 100)
          result (pca/pca-full X m n 3)
          ratios ^doubles (:explained-ratio result)
          total (loop [i 0, acc 0.0]
                  (if (>= i 3) acc
                      (recur (inc i) (+ acc (aget ratios i)))))]
      (is (<= total 1.0001) (str "Ratios sum " total " should be <= 1"))
      (is (> total 0.8) (str "Top 3 components should explain > 80%, got " total)))))

(deftest pca-full-components-test
  (testing "First component captures most variance"
    (let [[X m n] (make-test-data 100)
          result (pca/pca-full X m n 3)
          ratios ^doubles (:explained-ratio result)]
      (is (> (aget ratios 0) (aget ratios 1))
          "First component should have highest explained ratio")
      (is (> (aget ratios 1) (aget ratios 2))
          "Second component > third"))))

(deftest pca-covariance-eigh-test
  (testing "Covariance eigh solver agrees with full SVD"
    (let [[X m n] (make-test-data 100)
          r1 (pca/pca-full X m n 3)
          r2 (pca/pca-covariance-eigh X m n 3)]
      ;; Singular values should match
      (dotimes [i 3]
        (is (approx= (aget ^doubles (:singular-values r1) i)
                     (aget ^doubles (:singular-values r2) i)
                     0.1)
            (str "Singular value " i " mismatch"))))))

(deftest pca-randomized-test
  (testing "Randomized solver gives reasonable approximation"
    (let [[X m n] (make-test-data 200)
          r1 (pca/pca-full X m n 2)
          r2 (pca/pca-randomized X m n 2 10 4)]
      ;; Explained variance should be close
      (dotimes [i 2]
        (let [ev1 (aget ^doubles (:explained-var r1) i)
              ev2 (aget ^doubles (:explained-var r2) i)
              rel-err (/ (Math/abs (- ev1 ev2)) (Math/abs ev1))]
          (is (< rel-err 0.15)
              (str "Explained var " i " relative error " rel-err)))))))

(deftest pca-transform-roundtrip-test
  (testing "transform -> inverse-transform approximately recovers data"
    (let [m 50, n 4
          rng (java.util.Random. 42)
          X (double-array (* m n))
          _ (dotimes [i (* m n)]
              (aset X i (.nextGaussian rng)))
          ;; Full rank PCA (4 components for 4 features)
          result (pca/pca-full X m n 4)
          ;; Transform and inverse
          X-proj (pca/transform X result m)
          X-rec  (pca/inverse-transform X-proj result m)]
      ;; Should recover original data
      (dotimes [i (* m n)]
        (is (approx= (aget X i) (aget X-rec i) 1e-8)
            (str "Round-trip error at " i))))))

(deftest pca-auto-solver-test
  (testing "Auto solver produces valid results"
    (let [[X m n] (make-test-data 100)
          result (pca/pca X m n 2)]
      (is (== 2 (long (:n-components result))))
      (is (== 5 (long (:n-features result))))
      (is (== 100 (long (:n-samples result))))
      (is (> (aget ^doubles (:explained-ratio result) 0) 0.0)))))

(deftest pca-mean-centering-test
  (testing "Mean is computed correctly"
    (let [m 3, n 2
          X (double-array [1 2  3 4  5 6])
          result (pca/pca-full X m n 2)
          mean ^doubles (:mean result)]
      (is (approx= (aget mean 0) 3.0 1e-10) "Mean of col 0")
      (is (approx= (aget mean 1) 4.0 1e-10) "Mean of col 1"))))
