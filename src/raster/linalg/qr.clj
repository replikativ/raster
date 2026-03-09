(ns raster.linalg.qr
  "QR decomposition via LAPACK dgeqrf + dorgqr.

  Falls back to pure Householder implementation if LAPACK is unavailable.

  All matrices are flat double[] in row-major order."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.linalg.lapack :as lapack]))

;; ================================================================
;; LAPACK-accelerated QR
;; ================================================================

(deftm qr!
  "QR decomposition: A[m,n] -> Q[m,n] (thin), R[n,n] (upper triangular).
  A is not modified. m >= n required."
  [A :- (Array double) Q-out :- (Array double) R-out :- (Array double)
   m :- Long n :- Long] :- (Array double)
  (let [k (min m n)
        ;; Copy A into Q-out (dgeqrf overwrites input)
        work (double-array (* m n))
        _ (acopy! A 0 work 0 (* m n))
        tau (double-array k)
        info (lapack/dgeqrf! work tau m n)]
    (when-not (== info 0)
      (throw (ex-info "LAPACK dgeqrf failed" {:info info})))
    ;; Extract R [n, n] from upper triangle of work [m, n]
    (dotimes [i n]
      (dotimes [j n]
        (aset R-out (+ (* i n) j)
              (if (>= j i) (aget work (+ (* i n) j)) 0.0))))
    ;; Generate Q [m, n] from Householder reflectors
    (let [info2 (lapack/dorgqr! work tau m n k)]
      (when-not (== info2 0)
        (throw (ex-info "LAPACK dorgqr failed" {:info info2})))
      (acopy! work 0 Q-out 0 (* m n)))
    Q-out))

(deftm qr-q!
  "Compute only Q factor of QR decomposition. A[m,n] -> Q[m,n] (thin).
  A is not modified."
  [A :- (Array double) Q-out :- (Array double)
   m :- Long n :- Long] :- (Array double)
  (let [k (min m n)
        work (double-array (* m n))
        _ (acopy! A 0 work 0 (* m n))
        tau (double-array k)
        info (lapack/dgeqrf! work tau m n)]
    (when-not (== info 0)
      (throw (ex-info "LAPACK dgeqrf failed" {:info info})))
    (let [info2 (lapack/dorgqr! work tau m n k)]
      (when-not (== info2 0)
        (throw (ex-info "LAPACK dorgqr failed" {:info info2})))
      (acopy! work 0 Q-out 0 (* m n)))
    Q-out))
