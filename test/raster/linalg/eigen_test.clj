(ns raster.linalg.eigen-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.linalg.eigen :as eigen]
            [raster.linalg.blas :as blas]
            [raster.linalg.lapack :as lapack]))

(use-fixtures :once
  (fn [f] (if (lapack/available?) (f) (println "[SKIP] No LAPACK library"))))

(defn- approx=
  [a b tol]
  (< (Math/abs (- (double a) (double b))) tol))

(deftest eigh-eigenvalues-test
  (testing "Eigenvalues of known symmetric matrix"
    ;; A = [[2 1] [1 2]] has eigenvalues 1 and 3
    (let [A (double-array [2 1 1 2])
          result (eigen/eigh A 2)
          eigenvalues ^doubles (aget ^objects result 0)]
      (is (approx= (aget eigenvalues 0) 1.0 1e-10) "First eigenvalue should be 1")
      (is (approx= (aget eigenvalues 1) 3.0 1e-10) "Second eigenvalue should be 3"))))

(deftest eigh-eigenvector-test
  (testing "A * v = lambda * v for each eigenpair"
    (let [n 3
          ;; Symmetric positive definite matrix
          A (double-array [4 2 1  2 5 3  1 3 6])
          result (eigen/eigh A n)
          eigenvalues  ^doubles (aget ^objects result 0)
          eigenvectors ^doubles (aget ^objects result 1)]
      (dotimes [k n]
        (let [lambda (aget eigenvalues k)
              ;; Extract eigenvector k (row k of eigenvectors)
              v (double-array n)
              _ (System/arraycopy eigenvectors (* k n) v 0 n)
              ;; A @ v
              Av (double-array n)]
          (blas/dgemv! A v Av n n 1.0 0.0)
          ;; Check Av = lambda * v
          (dotimes [i n]
            (is (approx= (aget Av i) (* lambda (aget v i)) 1e-10)
                (str "A*v = lambda*v for eigenpair " k ", component " i))))))))

(deftest eigh-orthogonality-test
  (testing "Eigenvectors are orthonormal"
    (let [n 4
          A (double-array [5 1 0 1  1 4 1 0  0 1 3 1  1 0 1 2])
          result (eigen/eigh A n)
          eigenvectors ^doubles (aget ^objects result 1)]
      ;; V^T @ V should be identity (eigenvectors as rows)
      (let [VtV (double-array (* n n))]
        (blas/dgemm-nt! eigenvectors eigenvectors VtV n n n 1.0 0.0)
        (dotimes [i n]
          (dotimes [j n]
            (is (approx= (aget VtV (+ (* i n) j))
                         (if (== i j) 1.0 0.0) 1e-10)
                (str "V^T@V [" i "," j "]"))))))))

(deftest eigh-ascending-order-test
  (testing "Eigenvalues are in ascending order"
    (let [n 3
          A (double-array [4 2 1  2 5 3  1 3 6])
          result (eigen/eigh A n)
          eigenvalues ^doubles (aget ^objects result 0)]
      (dotimes [i (dec n)]
        (is (<= (aget eigenvalues i) (aget eigenvalues (inc i)))
            (str "eigenvalue[" i "] <= eigenvalue[" (inc i) "]"))))))

(deftest eigh-preserves-input-test
  (testing "Input matrix is not modified"
    (let [A (double-array [2 1 1 2])
          A-orig (double-array [2 1 1 2])]
      (eigen/eigh A 2)
      (dotimes [i 4]
        (is (== (aget A i) (aget A-orig i))
            "Input should not be modified")))))
