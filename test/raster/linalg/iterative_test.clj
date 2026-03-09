(ns raster.linalg.iterative-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.linalg.iterative :refer [cg gmres bicgstab pcg jacobi-precond]]
            [raster.core :refer [ftm]]))

(def ^:const TOL 1e-3)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) TOL))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; 3x3 Laplacian (SPD): [[2 -1 0][-1 2 -1][0 -1 2]]
;; True solution for b = [1 0 1]: x = [1 1 1]

(def laplacian-matvec
  (ftm [x :- (Array double) y :- (Array double)] :- (Array double)
       (let [x0 (aget x 0) x1 (aget x 1) x2 (aget x 2)]
         (aset y 0 (- (* 2.0 x0) x1))
         (aset y 1 (+ (- (* 2.0 x1) x0) (- x2)))
         (aset y 2 (- (* 2.0 x2) x1)))
       y))

;; b = A * [1 1 1] = [1 0 1]
(def b-vec (double-array [1.0 0.0 1.0]))

;; ================================================================
;; Conjugate Gradient
;; ================================================================

(deftest cg-laplacian-test
  (testing "CG solves 3x3 Laplacian system"
    (let [x0 (double-array [0.0 0.0 0.0])
          result (cg laplacian-matvec b-vec x0 3 1e-10 100)
          x (aget result 0)]
      (is (approx= 1.0 (aget ^doubles x 0)))
      (is (approx= 1.0 (aget ^doubles x 1)))
      (is (approx= 1.0 (aget ^doubles x 2))))))

;; ================================================================
;; GMRES
;; ================================================================

(deftest gmres-laplacian-test
  (testing "GMRES solves 3x3 Laplacian system"
    (let [x0 (double-array [0.0 0.0 0.0])
          result (gmres laplacian-matvec b-vec x0 3 1e-10 100 30)
          x (aget result 0)]
      (is (approx= 1.0 (aget ^doubles x 0)))
      (is (approx= 1.0 (aget ^doubles x 1)))
      (is (approx= 1.0 (aget ^doubles x 2))))))

;; ================================================================
;; BiCGSTAB
;; ================================================================

(deftest bicgstab-laplacian-test
  (testing "BiCGSTAB solves 3x3 Laplacian system"
    (let [x0 (double-array [0.0 0.0 0.0])
          result (bicgstab laplacian-matvec b-vec x0 3 1e-10 100)
          x (aget result 0)]
      (is (approx= 1.0 (aget ^doubles x 0)))
      (is (approx= 1.0 (aget ^doubles x 1)))
      (is (approx= 1.0 (aget ^doubles x 2))))))

;; ================================================================
;; Preconditioned Conjugate Gradient
;; ================================================================

(deftest pcg-laplacian-test
  (testing "PCG with Jacobi preconditioner solves Laplacian"
    (let [diag (double-array [2.0 2.0 2.0])  ;; diagonal of Laplacian
          precond-fn (ftm [r :- (Array double) z :- (Array double)] :- (Array double)
                          (jacobi-precond diag r z 3))
          x0 (double-array [0.0 0.0 0.0])
          result (pcg laplacian-matvec precond-fn b-vec x0 3 1e-10 100)
          x (aget result 0)]
      (is (approx= 1.0 (aget ^doubles x 0)))
      (is (approx= 1.0 (aget ^doubles x 1)))
      (is (approx= 1.0 (aget ^doubles x 2))))))

;; ================================================================
;; Convergence: residual should be small
;; ================================================================

(deftest cg-convergence-test
  (testing "CG residual is small"
    (let [x0 (double-array [0.0 0.0 0.0])
          result (cg laplacian-matvec b-vec x0 3 1e-10 100)
          residual (double (.doubleValue ^Double (aget result 1)))]
      (is (< residual 1e-8)
          (str "Residual should be < 1e-8, got " residual)))))

;; ================================================================
;; Helper: 2D Laplacian on NxN grid (N^2 unknowns, SPD)
;; ================================================================

(defn- make-laplacian-2d-matvec
  "Returns a matvec function for the 2D 5-point Laplacian on an NxN grid.
   Matrix is N^2 x N^2, SPD with diagonal = 4."
  [^long grid-n]
  (let [n (* grid-n grid-n)]
    (ftm [x :- (Array double) y :- (Array double)] :- (Array double)
         (dotimes [idx n]
           (let [row (quot idx grid-n)
                 col (rem idx grid-n)
              ;; diagonal: 4 * x[idx]
                 v (* 4.0 (clojure.core/aget ^doubles x idx))
              ;; left neighbor
                 v (if (> col 0)
                     (- v (clojure.core/aget ^doubles x (- idx 1)))
                     v)
              ;; right neighbor
                 v (if (< col (- grid-n 1))
                     (- v (clojure.core/aget ^doubles x (+ idx 1)))
                     v)
              ;; top neighbor
                 v (if (> row 0)
                     (- v (clojure.core/aget ^doubles x (- idx grid-n)))
                     v)
              ;; bottom neighbor
                 v (if (< row (- grid-n 1))
                     (- v (clojure.core/aget ^doubles x (+ idx grid-n)))
                     v)]
             (clojure.core/aset ^doubles y idx v)))
         y)))

(defn- compute-residual-norm
  "Computes ||Ax - b|| / ||b|| for result validation."
  [matvec-fn ^doubles x ^doubles b ^long n]
  (let [ax (double-array n)
        _ (matvec-fn x ax)
        bnorm (Math/sqrt (loop [i 0 s 0.0]
                           (if (>= i n) s
                               (recur (inc i) (+ s (* (clojure.core/aget b i)
                                                      (clojure.core/aget b i)))))))
        rnorm (Math/sqrt (loop [i 0 s 0.0]
                           (if (>= i n) s
                               (let [ri (- (clojure.core/aget ^doubles ax i)
                                           (clojure.core/aget b i))]
                                 (recur (inc i) (+ s (* ri ri)))))))]
    (if (> bnorm 0.0) (/ rnorm bnorm) rnorm)))

;; ================================================================
;; 1. Larger system: 2D Laplacian on 10x10 grid (100 unknowns)
;; ================================================================

(deftest cg-laplacian-10x10-test
  (testing "CG converges on 100-unknown 2D Laplacian"
    (let [grid-n 10
          n (* grid-n grid-n)
          mv (make-laplacian-2d-matvec grid-n)
          ;; Manufacture solution: x_true = [1, 1, ..., 1]
          x-true (double-array (repeat n 1.0))
          b (double-array n)
          _ (mv x-true b)
          x0 (double-array n)
          result (cg mv b x0 n 1e-10 2000)
          x (aget result 0)
          residual (double (.doubleValue ^Double (aget result 1)))
          iters (long (.longValue ^Long (aget result 2)))]
      (is (< residual 1e-6) (str "CG residual too large: " residual))
      (is (< iters 2000) (str "CG should converge before maxiter, took " iters))
      (doseq [i (range n)]
        (is (approx= 1.0 (clojure.core/aget ^doubles x i) 1e-4)
            (str "CG x[" i "] should be ~1.0"))))))

(deftest gmres-laplacian-10x10-test
  (testing "GMRES converges on 100-unknown 2D Laplacian"
    (let [grid-n 10
          n (* grid-n grid-n)
          mv (make-laplacian-2d-matvec grid-n)
          x-true (double-array (repeat n 1.0))
          b (double-array n)
          _ (mv x-true b)
          x0 (double-array n)
          result (gmres mv b x0 n 1e-10 2000 50)
          x (aget result 0)
          residual (double (.doubleValue ^Double (aget result 1)))
          iters (long (.longValue ^Long (aget result 2)))]
      (is (< residual 1e-6) (str "GMRES residual too large: " residual))
      (is (< iters 2000) (str "GMRES should converge before maxiter, took " iters))
      (doseq [i (range n)]
        (is (approx= 1.0 (clojure.core/aget ^doubles x i) 1e-4)
            (str "GMRES x[" i "] should be ~1.0"))))))

;; ================================================================
;; 2. Zero RHS: cg(A, zeros) should return zeros
;; ================================================================

(deftest cg-zero-rhs-test
  (testing "CG with zero RHS returns zero solution"
    (let [n 3
          b-zero (double-array n)
          x0 (double-array n)
          result (cg laplacian-matvec b-zero x0 n 1e-10 100)
          x (aget result 0)
          iters (long (.longValue ^Long (aget result 2)))]
      (doseq [i (range n)]
        (is (approx= 0.0 (clojure.core/aget ^doubles x i) 1e-14)
            (str "x[" i "] should be 0.0 for zero RHS")))
      (is (<= iters 1) "Zero RHS should converge immediately"))))

(deftest gmres-zero-rhs-test
  (testing "GMRES with zero RHS returns zero solution"
    (let [n 3
          b-zero (double-array n)
          x0 (double-array n)
          result (gmres laplacian-matvec b-zero x0 n 1e-10 100 10)
          x (aget result 0)]
      (doseq [i (range n)]
        (is (approx= 0.0 (clojure.core/aget ^doubles x i) 1e-14)
            (str "GMRES x[" i "] should be 0.0 for zero RHS"))))))

(deftest bicgstab-zero-rhs-test
  (testing "BiCGSTAB with zero RHS returns zero solution"
    (let [n 3
          b-zero (double-array n)
          x0 (double-array n)
          result (bicgstab laplacian-matvec b-zero x0 n 1e-10 100)
          x (aget result 0)]
      (doseq [i (range n)]
        (is (approx= 0.0 (clojure.core/aget ^doubles x i) 1e-14)
            (str "BiCGSTAB x[" i "] should be 0.0 for zero RHS"))))))

;; ================================================================
;; 3. Diagonal system: trivial solve, Jacobi preconditioner is exact
;; ================================================================

(deftest diagonal-system-cg-test
  (testing "CG solves diagonal system in 1 iteration"
    (let [n 5
          diag-vals (double-array [3.0 7.0 2.0 5.0 1.0])
          mv (ftm [x :- (Array double) y :- (Array double)] :- (Array double)
                  (dotimes [i 5]
                    (clojure.core/aset ^doubles y i
                                       (* (clojure.core/aget ^doubles diag-vals i)
                                          (clojure.core/aget ^doubles x i))))
                  y)
          ;; x_true = [1, 2, 3, 4, 5], b = diag * x_true
          b (double-array [3.0 14.0 6.0 20.0 5.0])
          x0 (double-array n)
          result (cg mv b x0 n 1e-12 100)
          x (aget result 0)
          iters (long (.longValue ^Long (aget result 2)))]
      (is (<= iters 5) (str "Diagonal system should converge fast, took " iters))
      (is (approx= 1.0 (clojure.core/aget ^doubles x 0) 1e-10))
      (is (approx= 2.0 (clojure.core/aget ^doubles x 1) 1e-10))
      (is (approx= 3.0 (clojure.core/aget ^doubles x 2) 1e-10))
      (is (approx= 4.0 (clojure.core/aget ^doubles x 3) 1e-10))
      (is (approx= 5.0 (clojure.core/aget ^doubles x 4) 1e-10)))))

(deftest diagonal-system-pcg-jacobi-exact-test
  (testing "PCG with exact Jacobi preconditioner on diagonal system converges in 1 iteration"
    (let [n 5
          diag-vals (double-array [3.0 7.0 2.0 5.0 1.0])
          mv (ftm [x :- (Array double) y :- (Array double)] :- (Array double)
                  (dotimes [i 5]
                    (clojure.core/aset ^doubles y i
                                       (* (clojure.core/aget ^doubles diag-vals i)
                                          (clojure.core/aget ^doubles x i))))
                  y)
          precond-fn (ftm [r :- (Array double) z :- (Array double)] :- (Array double)
                          (jacobi-precond diag-vals r z 5))
          b (double-array [3.0 14.0 6.0 20.0 5.0])
          x0 (double-array n)
          result (pcg mv precond-fn b x0 n 1e-12 100)
          x (aget result 0)
          iters (long (.longValue ^Long (aget result 2)))]
      (is (<= iters 1) (str "Exact Jacobi on diagonal should converge in 1 step, took " iters))
      (is (approx= 1.0 (clojure.core/aget ^doubles x 0) 1e-10))
      (is (approx= 2.0 (clojure.core/aget ^doubles x 1) 1e-10))
      (is (approx= 3.0 (clojure.core/aget ^doubles x 2) 1e-10))
      (is (approx= 4.0 (clojure.core/aget ^doubles x 3) 1e-10))
      (is (approx= 5.0 (clojure.core/aget ^doubles x 4) 1e-10)))))

;; ================================================================
;; 4. Non-symmetric system: GMRES and BiCGSTAB should handle it
;; ================================================================

(def nonsym-matvec
  "Non-symmetric 3x3 matrix: [[4 1 0] [0 3 1] [0 0 2]]
   Upper triangular, eigenvalues 4, 3, 2 (all positive).
   Solution for b = [5 5 4]: x = [1 1 2]"
  (ftm [x :- (Array double) y :- (Array double)] :- (Array double)
       (let [x0 (clojure.core/aget ^doubles x 0)
             x1 (clojure.core/aget ^doubles x 1)
             x2 (clojure.core/aget ^doubles x 2)]
         (clojure.core/aset ^doubles y 0 (+ (* 4.0 x0) (* 1.0 x1)))
         (clojure.core/aset ^doubles y 1 (+ (* 3.0 x1) (* 1.0 x2)))
         (clojure.core/aset ^doubles y 2 (* 2.0 x2)))
       y))

(def nonsym-b (double-array [5.0 5.0 4.0]))

(deftest gmres-nonsymmetric-test
  (testing "GMRES solves non-symmetric upper triangular system"
    (let [x0 (double-array [0.0 0.0 0.0])
          result (gmres nonsym-matvec nonsym-b x0 3 1e-10 100 10)
          x (aget result 0)]
      (is (approx= 1.0 (clojure.core/aget ^doubles x 0) 1e-6))
      (is (approx= 1.0 (clojure.core/aget ^doubles x 1) 1e-6))
      (is (approx= 2.0 (clojure.core/aget ^doubles x 2) 1e-6)))))

(deftest bicgstab-nonsymmetric-test
  (testing "BiCGSTAB solves non-symmetric upper triangular system"
    (let [x0 (double-array [0.0 0.0 0.0])
          result (bicgstab nonsym-matvec nonsym-b x0 3 1e-10 100)
          x (aget result 0)]
      (is (approx= 1.0 (clojure.core/aget ^doubles x 0) 1e-6))
      (is (approx= 1.0 (clojure.core/aget ^doubles x 1) 1e-6))
      (is (approx= 2.0 (clojure.core/aget ^doubles x 2) 1e-6)))))

;; ================================================================
;; 5. Residual check: ||Ax - b|| / ||b|| < tolerance for each solver
;; ================================================================

(deftest cg-residual-check-test
  (testing "CG final solution satisfies relative residual bound"
    (let [grid-n 10
          n (* grid-n grid-n)
          mv (make-laplacian-2d-matvec grid-n)
          ;; Random-ish RHS
          b (double-array (map #(Math/sin (double %)) (range n)))
          x0 (double-array n)
          tol 1e-8
          result (cg mv b x0 n tol 5000)
          x ^doubles (aget result 0)
          rel-res (compute-residual-norm mv x b n)]
      (is (< rel-res (* 10.0 tol))
          (str "CG relative residual " rel-res " should be < " (* 10.0 tol))))))

(deftest gmres-residual-check-test
  (testing "GMRES final solution satisfies relative residual bound"
    (let [grid-n 10
          n (* grid-n grid-n)
          mv (make-laplacian-2d-matvec grid-n)
          b (double-array (map #(Math/sin (double %)) (range n)))
          x0 (double-array n)
          tol 1e-8
          result (gmres mv b x0 n tol 5000 50)
          x ^doubles (aget result 0)
          rel-res (compute-residual-norm mv x b n)]
      (is (< rel-res (* 10.0 tol))
          (str "GMRES relative residual " rel-res " should be < " (* 10.0 tol))))))

(deftest bicgstab-residual-check-test
  (testing "BiCGSTAB final solution satisfies relative residual bound on SPD system"
    (let [grid-n 10
          n (* grid-n grid-n)
          mv (make-laplacian-2d-matvec grid-n)
          b (double-array (map #(Math/sin (double %)) (range n)))
          x0 (double-array n)
          tol 1e-8
          result (bicgstab mv b x0 n tol 5000)
          x ^doubles (aget result 0)
          rel-res (compute-residual-norm mv x b n)]
      (is (< rel-res (* 10.0 tol))
          (str "BiCGSTAB relative residual " rel-res " should be < " (* 10.0 tol))))))

;; ================================================================
;; 6. Different tolerances: tighter tol gives more accurate result
;; ================================================================

(deftest cg-tolerance-comparison-test
  (testing "Tighter CG tolerance produces more accurate solution"
    (let [grid-n 7
          n (* grid-n grid-n)
          mv (make-laplacian-2d-matvec grid-n)
          x-true (double-array (map #(Math/cos (double %)) (range n)))
          b (double-array n)
          _ (mv x-true b)
          ;; Loose tolerance
          result-loose (cg mv b (double-array n) n 1e-4 5000)
          x-loose ^doubles (aget result-loose 0)
          ;; Tight tolerance
          result-tight (cg mv b (double-array n) n 1e-10 5000)
          x-tight ^doubles (aget result-tight 0)
          ;; Compute max error for each
          err-loose (loop [i 0 mx 0.0]
                      (if (>= i n) mx
                          (recur (inc i) (max mx (Math/abs (- (clojure.core/aget x-loose i)
                                                              (clojure.core/aget ^doubles x-true i)))))))
          err-tight (loop [i 0 mx 0.0]
                      (if (>= i n) mx
                          (recur (inc i) (max mx (Math/abs (- (clojure.core/aget x-tight i)
                                                              (clojure.core/aget ^doubles x-true i)))))))]
      (is (< err-tight err-loose)
          (str "Tight tol error " err-tight " should be less than loose tol error " err-loose)))))

(deftest gmres-tolerance-comparison-test
  (testing "Tighter GMRES tolerance produces more accurate solution"
    (let [grid-n 7
          n (* grid-n grid-n)
          mv (make-laplacian-2d-matvec grid-n)
          x-true (double-array (map #(Math/cos (double %)) (range n)))
          b (double-array n)
          _ (mv x-true b)
          result-loose (gmres mv b (double-array n) n 1e-4 5000 30)
          x-loose ^doubles (aget result-loose 0)
          result-tight (gmres mv b (double-array n) n 1e-10 5000 30)
          x-tight ^doubles (aget result-tight 0)
          err-loose (loop [i 0 mx 0.0]
                      (if (>= i n) mx
                          (recur (inc i) (max mx (Math/abs (- (clojure.core/aget x-loose i)
                                                              (clojure.core/aget ^doubles x-true i)))))))
          err-tight (loop [i 0 mx 0.0]
                      (if (>= i n) mx
                          (recur (inc i) (max mx (Math/abs (- (clojure.core/aget x-tight i)
                                                              (clojure.core/aget ^doubles x-true i)))))))]
      (is (< err-tight err-loose)
          (str "Tight tol error " err-tight " should be less than loose tol error " err-loose)))))
