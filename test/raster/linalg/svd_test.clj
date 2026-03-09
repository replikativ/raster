(ns raster.linalg.svd-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.linalg.svd :as svd]
            [raster.linalg.blas :as blas]
            [raster.linalg.lapack :as lapack]))

(use-fixtures :once
  (fn [f] (if (lapack/available?) (f) (println "[SKIP] No LAPACK library"))))

(defn- approx=
  [a b tol]
  (< (Math/abs (- (double a) (double b))) tol))

(deftest svd-reconstruction-test
  (testing "A = U * diag(S) * Vt reconstruction"
    (let [m 4, n 3
          A (double-array [1 2 3  4 5 6  7 8 10  11 12 14])
          result (svd/svd A m n)
          U  ^doubles (aget ^objects result 0)
          S  ^doubles (aget ^objects result 1)
          Vt ^doubles (aget ^objects result 2)
          k (min m n)
          ;; U * diag(S): scale each column of U by S[j]
          US (double-array (* m k))
          _ (dotimes [i m]
              (dotimes [j k]
                (aset US (+ (* i k) j)
                      (* (aget U (+ (* i k) j)) (aget S j)))))
          ;; US @ Vt = A_reconstructed
          A-rec (double-array (* m n))]
      (blas/dgemm! US Vt A-rec m k n 1.0 0.0)
      (dotimes [i (* m n)]
        (is (approx= (aget A-rec i) (aget A i) 1e-10)
            (str "SVD reconstruction at index " i))))))

(deftest svd-orthogonality-test
  (testing "U^T @ U = I and V^T @ V = I"
    (let [m 5, n 3
          A (double-array [1 2 3  4 5 6  7 8 10  11 12 14  15 17 19])
          result (svd/svd A m n)
          U  ^doubles (aget ^objects result 0)
          Vt ^doubles (aget ^objects result 2)
          k (min m n)]
      ;; U^T @ U [k, k]
      (let [UtU (double-array (* k k))]
        (blas/dgemm-tn! U U UtU k m k 1.0 0.0)
        (dotimes [i k]
          (dotimes [j k]
            (is (approx= (aget UtU (+ (* i k) j))
                         (if (== i j) 1.0 0.0) 1e-10)
                (str "U^T@U [" i "," j "]")))))
      ;; Vt @ Vt^T [k, k] = V^T @ V = I
      (let [VtVtT (double-array (* k k))]
        (blas/dgemm-nt! Vt Vt VtVtT k k n 1.0 0.0)
        (dotimes [i k]
          (dotimes [j k]
            (is (approx= (aget VtVtT (+ (* i k) j))
                         (if (== i j) 1.0 0.0) 1e-10)
                (str "Vt@Vt^T [" i "," j "]"))))))))

(deftest svd-singular-values-descending-test
  (testing "Singular values are in descending order"
    (let [m 4, n 3
          A (double-array [1 2 3  4 5 6  7 8 10  11 12 14])
          result (svd/svd A m n)
          S ^doubles (aget ^objects result 1)
          k (min m n)]
      (dotimes [i (dec k)]
        (is (>= (aget S i) (aget S (inc i)))
            (str "S[" i "] >= S[" (inc i) "]"))))))

(deftest randomized-svd-reconstruction-test
  (testing "Randomized SVD gives good low-rank approximation"
    (let [m 50, n 20
          ;; Create a rank-3 matrix + small noise
          rng (java.util.Random. 123)
          A (double-array (* m n))
          _ (let [U0 (double-array (* m 3))
                  V0 (double-array (* 3 n))]
              (dotimes [i (* m 3)] (aset U0 i (.nextGaussian rng)))
              (dotimes [i (* 3 n)] (aset V0 i (.nextGaussian rng)))
              (blas/dgemm! U0 V0 A m 3 n 1.0 0.0)
              ;; Add small noise
              (dotimes [i (* m n)]
                (aset A i (+ (aget A i) (* 0.001 (.nextGaussian rng))))))
          ;; Randomized SVD with k=3
          result (svd/randomized-svd A m n 3 10 4)
          U  ^doubles (aget ^objects result 0)
          S  ^doubles (aget ^objects result 1)
          Vt ^doubles (aget ^objects result 2)
          ;; Reconstruct
          US (double-array (* m 3))
          _ (dotimes [i m]
              (dotimes [j 3]
                (aset US (+ (* i 3) j)
                      (* (aget U (+ (* i 3) j)) (aget S j)))))
          A-rec (double-array (* m n))
          _ (blas/dgemm! US Vt A-rec m 3 n 1.0 0.0)
          ;; Relative error
          norm-diff (loop [i 0, acc 0.0]
                      (if (>= i (* m n))
                        (Math/sqrt acc)
                        (let [d (- (aget A i) (aget A-rec i))]
                          (recur (inc i) (+ acc (* d d))))))
          norm-A (loop [i 0, acc 0.0]
                   (if (>= i (* m n))
                     (Math/sqrt acc)
                     (let [v (aget A i)]
                       (recur (inc i) (+ acc (* v v))))))
          rel-err (/ norm-diff norm-A)]
      (is (< rel-err 0.01)
          (str "Relative reconstruction error " rel-err " should be < 1%")))))
