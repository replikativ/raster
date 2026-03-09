(ns raster.linalg.sparse-csr-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.linalg.sparse :refer [->CSRMatrix ->COOMatrix
                                          csr-from-dense csr-to-dense
                                          spmv spmv-t spmm csr-scale csr-diag
                                          csr-eye csr-transpose coo-to-csr]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Dense <-> CSR round-trip
;; ================================================================

(deftest csr-roundtrip-test
  (testing "dense -> CSR -> dense round-trip"
    (let [A (double-array [1.0 0.0 2.0
                           0.0 3.0 0.0
                           4.0 0.0 5.0])
          csr (csr-from-dense A 3 3)
          A2 (csr-to-dense csr)]
      (dotimes [i 9]
        (is (approx= (aget A i) (aget A2 i))
            (str "Mismatch at index " i))))))

;; ================================================================
;; spmv: sparse matrix-vector multiply
;; ================================================================

(deftest spmv-identity-test
  (testing "identity matrix * vector = vector"
    (let [I (csr-eye 3)
          x (double-array [1.0 2.0 3.0])
          y (double-array 3)]
      (spmv I x y 1.0 0.0)
      (is (approx= 1.0 (aget y 0)))
      (is (approx= 2.0 (aget y 1)))
      (is (approx= 3.0 (aget y 2))))))

(deftest spmv-known-matrix-test
  (testing "known 3x3 sparse matrix * vector"
    ;; A = [[1 0 2][0 3 0][4 0 5]], x = [1 2 3]
    ;; y = [1*1+2*3, 3*2, 4*1+5*3] = [7, 6, 19]
    (let [A (csr-from-dense (double-array [1.0 0.0 2.0
                                           0.0 3.0 0.0
                                           4.0 0.0 5.0]) 3 3)
          x (double-array [1.0 2.0 3.0])
          y (double-array 3)]
      (spmv A x y 1.0 0.0)
      (is (approx= 7.0 (aget y 0)))
      (is (approx= 6.0 (aget y 1)))
      (is (approx= 19.0 (aget y 2))))))

;; ================================================================
;; spmv-t: transpose multiply
;; ================================================================

(deftest spmv-t-test
  (testing "transpose multiply: A^T * x"
    ;; A = [[1 0][0 3][4 0]], A^T = [[1 0 4][0 3 0]]
    ;; x = [1 2 3], A^T * x = [1+12, 6] = [13, 6]
    (let [A (csr-from-dense (double-array [1.0 0.0
                                           0.0 3.0
                                           4.0 0.0]) 3 2)
          x (double-array [1.0 2.0 3.0])
          y (double-array 2)]
      (spmv-t A x y 1.0 0.0)
      (is (approx= 13.0 (aget y 0)))
      (is (approx= 6.0 (aget y 1))))))

;; ================================================================
;; spmm: sparse * dense matrix
;; ================================================================

(deftest spmm-test
  (testing "sparse * dense matrix multiply"
    ;; A(sparse 2x2) = [[1 2][3 4]], B(dense 2x2) = [[5 6][7 8]]
    ;; C = [[19 22][43 50]]
    (let [A (csr-from-dense (double-array [1.0 2.0 3.0 4.0]) 2 2)
          B (double-array [5.0 6.0 7.0 8.0])
          C (double-array 4)]
      (spmm A B C 2)
      (is (approx= 19.0 (aget C 0)))
      (is (approx= 22.0 (aget C 1)))
      (is (approx= 43.0 (aget C 2)))
      (is (approx= 50.0 (aget C 3))))))

;; ================================================================
;; csr-diag: extract diagonal
;; ================================================================

(deftest csr-diag-test
  (testing "extract diagonal"
    (let [A (csr-from-dense (double-array [1.0 2.0 3.0
                                           4.0 5.0 6.0
                                           7.0 8.0 9.0]) 3 3)
          d (csr-diag A)]
      (is (approx= 1.0 (aget d 0)))
      (is (approx= 5.0 (aget d 1)))
      (is (approx= 9.0 (aget d 2))))))

;; ================================================================
;; csr-eye: identity properties
;; ================================================================

(deftest csr-eye-test
  (testing "identity matrix properties"
    (let [I (csr-eye 4)]
      ;; nrows = ncols = 4
      (is (= 4 (.-nrows I)))
      (is (= 4 (.-ncols I)))
      ;; nnz = 4
      (is (= 4 (.-nnz I)))
      ;; diagonal is all 1s
      (let [d (csr-diag I)]
        (dotimes [i 4]
          (is (approx= 1.0 (aget d i))))))))

;; ================================================================
;; csr-transpose: A^T^T = A
;; ================================================================

(deftest csr-transpose-double-test
  (testing "A^T^T ≈ A (double transpose is identity)"
    (let [A-dense (double-array [1.0 0.0 2.0
                                 0.0 3.0 0.0
                                 4.0 0.0 5.0])
          A (csr-from-dense A-dense 3 3)
          Att (csr-transpose (csr-transpose A))
          Att-dense (csr-to-dense Att)]
      (dotimes [i 9]
        (is (approx= (aget A-dense i) (aget Att-dense i))
            (str "Mismatch at index " i))))))

;; ================================================================
;; COO to CSR
;; ================================================================

(deftest coo-to-csr-test
  (testing "COO to CSR conversion"
    (let [coo (->COOMatrix (int-array [0 1 2 0])
                           (int-array [0 1 2 2])
                           (double-array [1.0 2.0 3.0 4.0])
                           3 3 4)
          csr (coo-to-csr coo)
          dense (csr-to-dense csr)]
      ;; Should have: [1 0 4; 0 2 0; 0 0 3]
      (is (approx= 1.0 (aget dense 0)))
      (is (approx= 0.0 (aget dense 1)))
      (is (approx= 4.0 (aget dense 2)))
      (is (approx= 0.0 (aget dense 3)))
      (is (approx= 2.0 (aget dense 4)))
      (is (approx= 0.0 (aget dense 5)))
      (is (approx= 0.0 (aget dense 6)))
      (is (approx= 0.0 (aget dense 7)))
      (is (approx= 3.0 (aget dense 8))))))

;; ================================================================
;; csr-scale
;; ================================================================

(deftest csr-scale-test
  (testing "scale CSR by 2.0"
    (let [A (csr-from-dense (double-array [1.0 0.0 0.0 2.0]) 2 2)
          A2 (csr-scale A 2.0)
          dense (csr-to-dense A2)]
      (is (approx= 2.0 (aget dense 0)))
      (is (approx= 0.0 (aget dense 1)))
      (is (approx= 0.0 (aget dense 2)))
      (is (approx= 4.0 (aget dense 3))))))
