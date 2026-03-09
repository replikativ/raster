(ns raster.linalg.svd
  "Singular Value Decomposition.

  - Full thin SVD via LAPACK dgesdd
  - Randomized SVD (Halko et al. 2009) for truncated low-rank approximation
  - SVD sign flip for deterministic results

  All matrices are flat double[] in row-major order."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.linalg.lapack :as lapack]
            [raster.linalg.qr :as qr]
            [raster.linalg.blas :as blas]))

(deftm svd-flip!
  "Ensure deterministic SVD signs: for each component, flip so that the
  largest absolute value in each Vt row is positive.
  U is [m,k], Vt is [k,n]. Modifies both in-place."
  [U :- (Array double) Vt :- (Array double)
   m :- Long k :- Long n :- Long] :- (Array double)
  (dotimes [i k]
    ;; Find max abs element in Vt row i
    (let [max-idx (loop [j 0, best-j 0, best-abs 0.0]
                    (if (>= j n)
                      best-j
                      (let [v (n/abs (aget Vt (+ (* i n) j)))]
                        (if (> v best-abs)
                          (recur (inc j) j v)
                          (recur (inc j) best-j best-abs)))))
          sign (if (< (aget Vt (+ (* i n) max-idx)) 0.0) -1.0 1.0)]
      (when (< sign 0.0)
        ;; Flip Vt row i
        (dotimes [j n]
          (let [idx (+ (* i n) j)]
            (aset Vt idx (- (aget Vt idx)))))
        ;; Flip U column i
        (dotimes [j m]
          (let [idx (+ (* j k) i)]
            (aset U idx (- (aget U idx))))))))
  U)

(deftm svd
  "Full thin SVD: A[m,n] -> [U, S, Vt] where U[m,k], S[k], Vt[k,n], k=min(m,n).
  A is not modified."
  [A :- (Array double) m :- Long n :- Long] :- (Array Object)
  (let [k (min m n)
        A-copy (double-array (* m n))
        _ (acopy! A 0 A-copy 0 (* m n))
        U (double-array (* m k))
        S (double-array k)
        Vt (double-array (* k n))
        info (lapack/dgesdd! A-copy U S Vt m n)]
    (when-not (== info 0)
      (throw (ex-info "LAPACK dgesdd failed" {:info info})))
    (svd-flip! U Vt m k n)
    (object-array [U S Vt])))

(deftm randomized-range-finder
  "Find an approximate orthonormal basis Q for the range of A[m,n].
  Uses random projection + power iterations with QR normalization.
  Returns Q[m, size]."
  [A :- (Array double) m :- Long n :- Long
   size :- Long n-iter :- Long] :- (Array double)
  (let [rng (java.util.Random. 42)
        ;; 1. Generate random Gaussian Omega[n, size]
        omega (double-array (* n size))
        _ (dotimes [i (* n size)]
            (aset omega i (.nextGaussian rng)))
        ;; 2. Y = A @ Omega  [m, size]
        Y (double-array (* m size))
        _ (blas/dgemm! A omega Y m n size 1.0 0.0)
        ;; Scratch buffers for QR
        Q (double-array (* m size))
        R-scratch (double-array (* size size))
        Z (double-array (* n size))]
    ;; 3. Power iterations with QR normalization
    (dotimes [_iter n-iter]
      ;; Q, _ = QR(Y)
      (qr/qr! Y Q R-scratch m size)
      ;; Z = A^T @ Q  [n, size]
      (blas/dgemm-tn! A Q Z n m size 1.0 0.0)
      ;; Q2, _ = QR(Z)  — reuse Q/R as scratch (Q2 is [n, size])
      (let [Q2 (double-array (* n size))
            R2 (double-array (* size size))]
        (qr/qr! Z Q2 R2 n size)
        ;; Y = A @ Q2  [m, size]
        (blas/dgemm! A Q2 Y m n size 1.0 0.0)))
    ;; 4. Final QR
    (qr/qr! Y Q R-scratch m size)
    Q))

(deftm randomized-svd
  "Randomized SVD (Halko et al. 2009).
  A[m,n] -> [U, S, Vt] truncated to n-components.
  n-oversamples: extra dimensions for accuracy (default 10).
  n-iter: power iterations for spectral decay (default 4)."
  [A :- (Array double) m :- Long n :- Long
   n-components :- Long n-oversamples :- Long n-iter :- Long]
  :- (Array Object)
  (let [size (min (+ n-components n-oversamples) (min m n))
        ;; 1. Q = randomized range finder [m, size]
        Q (randomized-range-finder A m n size n-iter)
        ;; 2. B = Q^T @ A  [size, n]
        B (double-array (* size n))
        _ (blas/dgemm-tn! Q A B size m n 1.0 0.0)
        ;; 3. Full SVD of small B [size, n]
        svd-result (svd B size n)
        Uhat (aget ^objects svd-result 0)
        S    (aget ^objects svd-result 1)
        Vt   (aget ^objects svd-result 2)
        ;; 4. U = Q @ Uhat  [m, size]
        U-full (double-array (* m size))
        _ (blas/dgemm! Q ^doubles Uhat U-full m size size 1.0 0.0)
        ;; 5. Truncate to n-components
        k n-components
        U-trunc (double-array (* m k))
        S-trunc (double-array k)
        Vt-trunc (double-array (* k n))]
    ;; Copy truncated U columns
    (dotimes [i m]
      (acopy! U-full (* i size) U-trunc (* i k) k))
    ;; Copy truncated S
    (acopy! S 0 S-trunc 0 k)
    ;; Copy truncated Vt rows
    (dotimes [i k]
      (acopy! Vt (* i n) Vt-trunc (* i n) n))
    (svd-flip! U-trunc Vt-trunc m k n)
    (object-array [U-trunc S-trunc Vt-trunc])))
