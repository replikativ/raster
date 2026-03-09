(ns raster.linalg.eigen
  "Symmetric eigendecomposition via LAPACK dsyevd.

  All matrices are flat double[] in row-major order."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.linalg.lapack :as lapack]))

(deftm eigh
  "Symmetric eigendecomposition: A[n,n] -> [eigenvalues, eigenvectors].
  eigenvalues[n] in ascending order, eigenvectors[n,n] row-major
  (row i = eigenvector i). A is not modified."
  [A :- (Array double) n :- Long] :- (Array Object)
  (let [A-copy (double-array (* n n))
        _ (acopy! A 0 A-copy 0 (* n n))
        eigenvalues (double-array n)
        info (lapack/dsyevd! A-copy eigenvalues n)]
    (when-not (== info 0)
      (throw (ex-info "LAPACK dsyevd failed" {:info info})))
    ;; Fortran dsyevd_ with symmetric matrix: row-major = col-major,
    ;; eigenvectors are stored as columns in col-major = rows in row-major.
    ;; A-copy already has eigenvectors as rows — no transpose needed.
    (object-array [eigenvalues A-copy])))
