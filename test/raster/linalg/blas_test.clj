(ns raster.linalg.blas-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.linalg.blas :as blas]))

(use-fixtures :once
  (fn [f] (if (blas/available?) (f) (println "[SKIP] No BLAS library"))))

(deftest test-available
  (is (blas/available?) "BLAS should be loaded and functional (MKL or OpenBLAS)"))

(deftest test-dgemm-identity
  (testing "C = I @ B should give B"
    (let [I (double-array [1 0 0 1])
          B (double-array [5 6 7 8])
          C (double-array 4)]
      (blas/dgemm! I B C 2 2 2 1.0 0.0)
      (is (== 5.0 (aget C 0)))
      (is (== 6.0 (aget C 1)))
      (is (== 7.0 (aget C 2)))
      (is (== 8.0 (aget C 3))))))

(deftest test-dgemm-basic
  (testing "C = A @ B, 2x3 @ 3x2 -> 2x2"
    (let [A (double-array [1 2 3 4 5 6])       ;; [2,3]
          B (double-array [7 8 9 10 11 12])     ;; [3,2]
          C (double-array 4)]
      (blas/dgemm! A B C 2 3 2 1.0 0.0)
      ;; [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
      ;; [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
      (is (== 58.0 (aget C 0)))
      (is (== 64.0 (aget C 1)))
      (is (== 139.0 (aget C 2)))
      (is (== 154.0 (aget C 3))))))

(deftest test-dgemm-alpha-beta
  (testing "C = 2*A@B + 3*C"
    (let [A (double-array [1 0 0 1])
          B (double-array [5 6 7 8])
          C (double-array [10 20 30 40])]
      (blas/dgemm! A B C 2 2 2 2.0 3.0)
      ;; C[0] = 2*(1*5+0*7) + 3*10 = 10+30 = 40
      ;; C[1] = 2*(1*6+0*8) + 3*20 = 12+60 = 72
      ;; C[2] = 2*(0*5+1*7) + 3*30 = 14+90 = 104
      ;; C[3] = 2*(0*6+1*8) + 3*40 = 16+120 = 136
      (is (== 40.0 (aget C 0)))
      (is (== 72.0 (aget C 1)))
      (is (== 104.0 (aget C 2)))
      (is (== 136.0 (aget C 3))))))

(deftest test-dgemm-tn
  (testing "C = A^T @ B, A:[3,2] transposed to [2,3], B:[3,2] -> C:[2,2]"
    (let [A (double-array [1 4 2 5 3 6])       ;; [3,2] stored row-major
          B (double-array [7 8 9 10 11 12])     ;; [3,2]
          C (double-array 4)]
      ;; A^T = [[1,2,3],[4,5,6]]  (2x3)
      ;; A^T @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12],
      ;;            [4*7+5*9+6*11, 4*8+5*10+6*12]]
      ;;         = [[58,64],[139,154]]
      (blas/dgemm-tn! A B C 2 3 2 1.0 0.0)
      (is (== 58.0 (aget C 0)))
      (is (== 64.0 (aget C 1)))
      (is (== 139.0 (aget C 2)))
      (is (== 154.0 (aget C 3))))))

(deftest test-dgemm-nt
  (testing "C = A @ B^T, A:[2,3], B:[2,3] -> C:[2,2]"
    (let [A (double-array [1 2 3 4 5 6])       ;; [2,3]
          B (double-array [7 9 11 8 10 12])     ;; [2,3] stored row-major
          C (double-array 4)]
      ;; B^T = [[7,8],[9,10],[11,12]]  (3x2)
      ;; A @ B^T = [[1*7+2*9+3*11, 1*8+2*10+3*12],
      ;;            [4*7+5*9+6*11, 4*8+5*10+6*12]]
      ;;         = [[58,64],[139,154]]
      (blas/dgemm-nt! A B C 2 3 2 1.0 0.0)
      (is (== 58.0 (aget C 0)))
      (is (== 64.0 (aget C 1)))
      (is (== 139.0 (aget C 2)))
      (is (== 154.0 (aget C 3))))))

(deftest test-dgemv-basic
  (testing "y = A @ x, 2x3 @ 3 -> 2"
    (let [A (double-array [1 2 3 4 5 6])       ;; [2,3]
          x (double-array [1 2 3])
          y (double-array 2)]
      (blas/dgemv! A x y 2 3 1.0 0.0)
      ;; y[0] = 1*1+2*2+3*3 = 14
      ;; y[1] = 4*1+5*2+6*3 = 32
      (is (== 14.0 (aget y 0)))
      (is (== 32.0 (aget y 1))))))

(deftest test-dgemv-alpha-beta
  (testing "y = 2*A@x + 3*y"
    (let [A (double-array [1 0 0 1])
          x (double-array [5 7])
          y (double-array [10 20])]
      (blas/dgemv! A x y 2 2 2.0 3.0)
      ;; y[0] = 2*5 + 3*10 = 40
      ;; y[1] = 2*7 + 3*20 = 74
      (is (== 40.0 (aget y 0)))
      (is (== 74.0 (aget y 1))))))

(deftest test-dgemv-t
  (testing "y = A^T @ x, A:[3,2], x:[3] -> y:[2]"
    (let [A (double-array [1 4 2 5 3 6])  ;; [3,2] row-major
          x (double-array [1 2 3])
          y (double-array 2)]
      ;; A^T = [[1,2,3],[4,5,6]], y = A^T @ x
      ;; y[0] = 1*1+2*2+3*3 = 14
      ;; y[1] = 4*1+5*2+6*3 = 32
      (blas/dgemv-t! A x y 3 2 1.0 0.0)
      (is (== 14.0 (aget y 0)))
      (is (== 32.0 (aget y 1))))))

(deftest test-dger
  (testing "A += alpha * x * y^T, rank-1 update"
    (let [x (double-array [1 2 3])
          y (double-array [4 5])
          A (double-array [10 20 30 40 50 60])]  ;; [3,2]
      ;; A += 1.0 * x * y^T
      ;; A[0,0] += 1*4=4  -> 14
      ;; A[0,1] += 1*5=5  -> 25
      ;; A[1,0] += 2*4=8  -> 38
      ;; A[1,1] += 2*5=10 -> 50
      ;; A[2,0] += 3*4=12 -> 62
      ;; A[2,1] += 3*5=15 -> 75
      (blas/dger! x y A 3 2 1.0)
      (is (== 14.0 (aget A 0)))
      (is (== 25.0 (aget A 1)))
      (is (== 38.0 (aget A 2)))
      (is (== 50.0 (aget A 3)))
      (is (== 62.0 (aget A 4)))
      (is (== 75.0 (aget A 5))))))

(deftest test-daxpy
  (testing "y += alpha * x"
    (let [x (double-array [1 2 3 4])
          y (double-array [10 20 30 40])]
      (blas/daxpy! x y 4 2.0)
      ;; y[i] += 2*x[i]
      (is (== 12.0 (aget y 0)))
      (is (== 24.0 (aget y 1)))
      (is (== 36.0 (aget y 2)))
      (is (== 48.0 (aget y 3)))))
  (testing "SGD: W -= lr * dW  (alpha = -lr)"
    (let [w (double-array [1.0 2.0 3.0])
          dw (double-array [0.1 0.2 0.3])]
      (blas/daxpy! dw w 3 -0.01)
      (is (< (Math/abs (- (aget w 0) 0.999)) 1e-10))
      (is (< (Math/abs (- (aget w 1) 1.998)) 1e-10))
      (is (< (Math/abs (- (aget w 2) 2.997)) 1e-10)))))

(deftest test-dgemm-rectangular
  (testing "Non-square: 1x4 @ 4x1 -> 1x1 (dot product)"
    (let [A (double-array [1 2 3 4])
          B (double-array [5 6 7 8])
          C (double-array 1)]
      (blas/dgemm! A B C 1 4 1 1.0 0.0)
      ;; 1*5+2*6+3*7+4*8 = 70
      (is (== 70.0 (aget C 0)))))
  (testing "Non-square: 4x1 @ 1x4 -> 4x4 (outer product)"
    (let [A (double-array [1 2 3 4])
          B (double-array [5 6 7 8])
          C (double-array 16)]
      (blas/dgemm! A B C 4 1 4 1.0 0.0)
      (is (== 5.0 (aget C 0)))
      (is (== 8.0 (aget C 3)))
      (is (== 20.0 (aget C 12)))
      (is (== 32.0 (aget C 15))))))
