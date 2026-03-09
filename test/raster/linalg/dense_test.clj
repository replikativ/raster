(ns raster.linalg.dense-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.linalg.core :refer [solve solve-multiple cholesky cholesky-solve
                                        lu lstsq array-inv array-det cond-number matrix-norm]]
            [raster.linalg.lapack :as lapack]))

(use-fixtures :once
  (fn [f] (if (lapack/available?) (f) (println "[SKIP] No LAPACK library"))))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; solve: Ax = b
;; ================================================================

(deftest solve-2x2-test
  (testing "solve 2x2 system [[2 1][1 3]]x = [5 7]"
    ;; det = 5, x = (5*3-7*1)/5 = 8/5 = 1.6, y = (2*7-5*1)/5 = 9/5 = 1.8
    (let [A (double-array [2.0 1.0 1.0 3.0])
          b (double-array [5.0 7.0])
          x (solve A b 2)]
      (is (approx= 1.6 (aget x 0)))
      (is (approx= 1.8 (aget x 1))))))

(deftest solve-3x3-test
  (testing "solve 3x3 system"
    ;; [[1 2 3][0 1 4][5 6 0]]x = [1 1 1]
    ;; det = 1, known solution via Cramer's rule
    (let [A (double-array [1.0 2.0 3.0
                           0.0 1.0 4.0
                           5.0 6.0 0.0])
          b (double-array [1.0 1.0 1.0])
          x (solve A b 3)]
      ;; Verify Ax = b
      (is (approx= 1.0 (+ (* 1.0 (aget x 0)) (* 2.0 (aget x 1)) (* 3.0 (aget x 2)))))
      (is (approx= 1.0 (+ (* 0.0 (aget x 0)) (* 1.0 (aget x 1)) (* 4.0 (aget x 2)))))
      (is (approx= 1.0 (+ (* 5.0 (aget x 0)) (* 6.0 (aget x 1)) (* 0.0 (aget x 2))))))))

;; ================================================================
;; solve-multiple: AX = B
;; ================================================================

(deftest solve-multiple-test
  (testing "solve 2x2 with 2 RHS"
    (let [A (double-array [2.0 1.0 1.0 3.0])
          ;; B is 2x2, row-major: column 0 = [5 7], column 1 = [3 1]
          B (double-array [5.0 3.0 7.0 1.0])
          X (solve-multiple A B 2 2)]
      ;; First RHS: [5 7] => x = [1.6 1.8]
      (is (approx= 1.6 (aget X 0)))
      (is (approx= 1.8 (aget X 2)))
      ;; Second RHS: [3 1] => x = (3*3-1)/5=1.6, (2-3)/5=-0.2
      (is (approx= 1.6 (aget X 1)))
      (is (approx= -0.2 (aget X 3))))))

;; ================================================================
;; cholesky: SPD matrix
;; ================================================================

(deftest cholesky-test
  (testing "Cholesky of SPD matrix [[4 2][2 5]]"
    (let [A (double-array [4.0 2.0 2.0 5.0])
          L (cholesky A 2)]
      ;; Verify L * L^T ≈ A
      ;; L is lower triangular 2x2
      (let [l00 (aget L 0) l01 (aget L 1)
            l10 (aget L 2) l11 (aget L 3)]
        ;; l01 should be 0 (upper triangle zeroed)
        (is (approx= 0.0 l01))
        ;; L*L^T
        (is (approx= 4.0 (+ (* l00 l00) (* l01 l01))))   ;; (0,0)
        (is (approx= 2.0 (+ (* l10 l00) (* l11 l01))))   ;; (1,0)
        (is (approx= 5.0 (+ (* l10 l10) (* l11 l11))))   ;; (1,1)
        ))))

;; ================================================================
;; cholesky-solve
;; ================================================================

(deftest cholesky-solve-test
  (testing "Cholesky solve via L from [[4 2][2 5]]"
    (let [A (double-array [4.0 2.0 2.0 5.0])
          L (cholesky A 2)
          b (double-array [10.0 13.0])
          x (cholesky-solve L b 2)]
      ;; Verify Ax ≈ b
      (is (approx= 10.0 (+ (* 4.0 (aget x 0)) (* 2.0 (aget x 1)))))
      (is (approx= 13.0 (+ (* 2.0 (aget x 0)) (* 5.0 (aget x 1))))))))

;; ================================================================
;; lu
;; ================================================================

(deftest lu-test
  (testing "LU decomposition of 3x3 matrix"
    (let [A (double-array [2.0 1.0 1.0
                           4.0 3.0 3.0
                           8.0 7.0 9.0])
          result (lu A 3)]
      ;; Result is Object array [LU-combined, pivots]
      (is (some? (aget result 0)))
      (is (some? (aget result 1))))))

;; ================================================================
;; lstsq: overdetermined system
;; ================================================================

(deftest lstsq-test
  (testing "lstsq for overdetermined 3x2 system"
    ;; A = [[1 1][1 2][1 3]], b = [1 2 2]
    ;; Least squares fit of y = a + b*x
    (let [A (double-array [1.0 1.0
                           1.0 2.0
                           1.0 3.0])
          b (double-array [1.0 2.0 2.0])
          x (lstsq A b 3 2)]
      ;; Normal equations: A^T A x = A^T b
      ;; A^T A = [[3 6][6 14]], A^T b = [5 12]
      ;; x = [2/3, 1/2]
      (is (approx= (/ 2.0 3.0) (aget x 0) 1e-4))
      (is (approx= 0.5 (aget x 1) 1e-4)))))

;; ================================================================
;; array-inv
;; ================================================================

(deftest array-inv-test
  (testing "2x2 matrix inverse, A*A^{-1} ≈ I"
    (let [A (double-array [1.0 2.0 3.0 4.0])
          Ainv (array-inv A 2)
          ;; Compute A * A^{-1}
          i00 (+ (* 1.0 (aget Ainv 0)) (* 2.0 (aget Ainv 2)))
          i01 (+ (* 1.0 (aget Ainv 1)) (* 2.0 (aget Ainv 3)))
          i10 (+ (* 3.0 (aget Ainv 0)) (* 4.0 (aget Ainv 2)))
          i11 (+ (* 3.0 (aget Ainv 1)) (* 4.0 (aget Ainv 3)))]
      (is (approx= 1.0 i00))
      (is (approx= 0.0 i01))
      (is (approx= 0.0 i10))
      (is (approx= 1.0 i11)))))

;; ================================================================
;; array-det
;; ================================================================

(deftest array-det-2x2-test
  (testing "2x2 determinant"
    ;; [[1 2][3 4]] => det = 1*4 - 2*3 = -2
    (is (approx= -2.0 (array-det (double-array [1.0 2.0 3.0 4.0]) 2)))))

(deftest array-det-3x3-test
  (testing "3x3 determinant"
    ;; [[1 2 3][0 1 4][5 6 0]] => det = 1
    (is (approx= 1.0 (array-det (double-array [1.0 2.0 3.0
                                               0.0 1.0 4.0
                                               5.0 6.0 0.0]) 3)))))

;; ================================================================
;; cond-number
;; ================================================================

(deftest cond-number-identity-test
  (testing "condition number of identity matrix = 1"
    (let [I (double-array [1.0 0.0 0.0
                           0.0 1.0 0.0
                           0.0 0.0 1.0])]
      (is (approx= 1.0 (cond-number I 3 3))))))

;; ================================================================
;; matrix-norm
;; ================================================================

(deftest matrix-norm-frobenius-test
  (testing "Frobenius norm (ord=-1)"
    ;; [[1 2][3 4]] => sqrt(1+4+9+16) = sqrt(30)
    (let [A (double-array [1.0 2.0 3.0 4.0])]
      (is (approx= (Math/sqrt 30.0) (matrix-norm A 2 2 -1))))))

(deftest matrix-norm-1norm-test
  (testing "1-norm: max column sum of absolute values"
    ;; [[1 2][3 4]] => col sums: |1|+|3|=4, |2|+|4|=6 => 6
    (let [A (double-array [1.0 2.0 3.0 4.0])]
      (is (approx= 6.0 (matrix-norm A 2 2 1))))))

(deftest matrix-norm-2norm-test
  (testing "2-norm: largest singular value"
    ;; Identity: singular values all 1 => 2-norm = 1
    (let [I (double-array [1.0 0.0 0.0 1.0])]
      (is (approx= 1.0 (matrix-norm I 2 2 2))))))
