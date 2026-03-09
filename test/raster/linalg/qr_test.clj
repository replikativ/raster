(ns raster.linalg.qr-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.linalg.qr :as qr]
            [raster.linalg.blas :as blas]
            [raster.linalg.lapack :as lapack]))

(use-fixtures :once
  (fn [f] (if (lapack/available?) (f) (println "[SKIP] No LAPACK library"))))

(defn- approx=
  "Check if two doubles are approximately equal."
  [a b tol]
  (< (Math/abs (- (double a) (double b))) tol))

(defn- mat-mul
  "Multiply A[m,k] @ B[k,n] -> C[m,n] via BLAS."
  [A B m k n]
  (let [C (double-array (* m n))]
    (blas/dgemm! A B C m k n 1.0 0.0)
    C))

(deftest qr-basic-test
  (testing "QR of 3x3 identity is identity"
    (let [A (double-array [1 0 0  0 1 0  0 0 1])
          Q (double-array 9)
          R (double-array 9)]
      (qr/qr! A Q R 3 3)
      ;; Q should be identity (up to sign)
      (dotimes [i 3]
        (is (approx= (Math/abs (aget Q (+ (* i 3) i))) 1.0 1e-12)
            (str "Q diagonal " i)))
      ;; R should be identity (up to sign)
      (dotimes [i 3]
        (is (approx= (Math/abs (aget R (+ (* i 3) i))) 1.0 1e-12)
            (str "R diagonal " i))))))

(deftest qr-reconstruction-test
  (testing "A = Q * R reconstruction"
    (let [m 4, n 3
          A (double-array [1 2 3  4 5 6  7 8 10  11 12 14])
          Q (double-array (* m n))
          R (double-array (* n n))]
      (qr/qr! A Q R m n)
      ;; Reconstruct: Q[m,n] @ R[n,n] should equal A
      (let [QR (mat-mul Q R m n n)]
        (dotimes [i (* m n)]
          (is (approx= (aget QR i) (aget A i) 1e-10)
              (str "QR reconstruction at index " i)))))))

(deftest qr-orthogonality-test
  (testing "Q^T @ Q = I (thin Q is orthonormal)"
    (let [m 5, n 3
          A (double-array [1 2 3  4 5 6  7 8 10  11 12 14  15 17 19])
          Q (double-array (* m n))
          R (double-array (* n n))]
      (qr/qr! A Q R m n)
      ;; Q^T @ Q [n, n] should be identity
      (let [QtQ (double-array (* n n))]
        (blas/dgemm-tn! Q Q QtQ n m n 1.0 0.0)
        (dotimes [i n]
          (dotimes [j n]
            (let [expected (if (== i j) 1.0 0.0)]
              (is (approx= (aget QtQ (+ (* i n) j)) expected 1e-10)
                  (str "Q^T@Q [" i "," j "]")))))))))

(deftest qr-upper-triangular-test
  (testing "R is upper triangular"
    (let [m 4, n 3
          A (double-array [1 2 3  4 5 6  7 8 10  11 12 14])
          Q (double-array (* m n))
          R (double-array (* n n))]
      (qr/qr! A Q R m n)
      ;; Below diagonal should be zero
      (dotimes [i n]
        (dotimes [j i]
          (is (approx= (aget R (+ (* i n) j)) 0.0 1e-10)
              (str "R[" i "," j "] should be zero")))))))

(deftest qr-q-only-test
  (testing "qr-q! produces same Q as qr!"
    (let [m 4, n 3
          A (double-array [1 2 3  4 5 6  7 8 10  11 12 14])
          Q1 (double-array (* m n))
          R (double-array (* n n))
          Q2 (double-array (* m n))]
      (qr/qr! A Q1 R m n)
      (qr/qr-q! A Q2 m n)
      (dotimes [i (* m n)]
        (is (approx= (aget Q1 i) (aget Q2 i) 1e-12)
            (str "Q mismatch at " i))))))
