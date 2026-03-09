(ns raster.linalg.pca
  "Principal Component Analysis with three solver paths:
  - full: thin SVD via LAPACK
  - covariance-eigh: eigendecomposition of covariance matrix
  - randomized: Halko et al. randomized SVD

  Mirrors scikit-learn's PCA API with typed multi-dispatch.

  All matrices are flat double[] in row-major order."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.linalg.svd :as svd]
            [raster.linalg.eigen :as eigen]
            [raster.linalg.blas :as blas]))

(defvalue PCAResult
  [components :- (Array double)       ;; [n_components, n_features] row-major
   singular-values :- (Array double)  ;; [n_components]
   explained-var :- (Array double)    ;; [n_components]
   explained-ratio :- (Array double)  ;; [n_components]
   mean :- (Array double)             ;; [n_features]
   n-components :- Long
   n-features :- Long
   n-samples :- Long
   noise-variance :- Double])

;; ================================================================
;; Helpers
;; ================================================================

(deftm column-mean!
  "Compute column means of X[m,n] into mean[n]."
  [X :- (Array double) mean :- (Array double) m :- Long n :- Long]
  :- (Array double)
  (dotimes [j n]
    (let [s (loop [i 0, acc 0.0]
              (if (>= i m)
                acc
                (recur (inc i) (+ acc (aget X (+ (* i n) j))))))]
      (aset mean j (/ s (double m)))))
  mean)

(deftm center!
  "Center X[m,n] in-place by subtracting mean[n] from each row."
  [X :- (Array double) mean :- (Array double) m :- Long n :- Long]
  :- (Array double)
  (dotimes [i m]
    (dotimes [j n]
      (let [idx (+ (* i n) j)]
        (aset X idx (- (aget X idx) (aget mean j))))))
  X)

;; ================================================================
;; Full SVD solver
;; ================================================================

(deftm pca-full
  "PCA via full thin SVD. X[m,n] row-major, not modified."
  [X :- (Array double) m :- Long n :- Long n-components :- Long]
  :- PCAResult
  (let [;; Copy and center
        X-c (double-array (* m n))
        _ (acopy! X 0 X-c 0 (* m n))
        mean (double-array n)
        _ (column-mean! X-c mean m n)
        _ (center! X-c mean m n)
        ;; Full thin SVD
        result (svd/svd X-c m n)
        U  ^doubles (aget ^objects result 0)
        S  ^doubles (aget ^objects result 1)
        Vt ^doubles (aget ^objects result 2)
        k (min n-components (min m n))
        ;; Extract top k components
        components (double-array (* k n))
        singular-values (double-array k)
        explained-var (double-array k)
        total-var (loop [i 0, acc 0.0]
                    (if (>= i (min m n))
                      acc
                      (let [si (aget S i)]
                        (recur (inc i) (+ acc (/ (* si si) (double (dec m))))))))
        explained-ratio (double-array k)
        ;; Noise variance from remaining singular values
        noise-var (if (< k (min m n))
                    (let [remaining (- (min m n) k)]
                      (loop [i k, acc 0.0]
                        (if (>= i (min m n))
                          (/ acc (double remaining) (double (dec m)))
                          (let [si (aget S i)]
                            (recur (inc i) (+ acc (* si si)))))))
                    0.0)]
    ;; Copy top k rows of Vt into components
    (dotimes [i k]
      (acopy! Vt (* i n) components (* i n) n))
    ;; Singular values and explained variance
    (dotimes [i k]
      (let [si (aget S i)
            ev (/ (* si si) (double (dec m)))]
        (aset singular-values i si)
        (aset explained-var i ev)
        (aset explained-ratio i (/ ev total-var))))
    (->PCAResult components singular-values explained-var explained-ratio
                 mean k n m noise-var)))

;; ================================================================
;; Covariance eigendecomposition solver
;; ================================================================

(deftm pca-covariance-eigh
  "PCA via eigendecomposition of covariance matrix. Best when m >> n.
  X[m,n] row-major, not modified."
  [X :- (Array double) m :- Long n :- Long n-components :- Long]
  :- PCAResult
  (let [;; Copy and center
        X-c (double-array (* m n))
        _ (acopy! X 0 X-c 0 (* m n))
        mean (double-array n)
        _ (column-mean! X-c mean m n)
        _ (center! X-c mean m n)
        ;; Covariance matrix C = X^T @ X / (m-1)  [n, n]
        C (double-array (* n n))
        _ (blas/dgemm-tn! X-c X-c C n m n 1.0 0.0)
        _ (let [scale (/ 1.0 (double (dec m)))]
            (dotimes [i (* n n)]
              (aset C i (* (aget C i) scale))))
        ;; Eigendecomposition
        eig-result (eigen/eigh C n)
        eigenvalues  ^doubles (aget ^objects eig-result 0)
        eigenvectors ^doubles (aget ^objects eig-result 1)
        ;; LAPACK returns ascending order — we want descending
        k (min n-components n)
        components (double-array (* k n))
        singular-values (double-array k)
        explained-var (double-array k)
        total-var (loop [i 0, acc 0.0]
                    (if (>= i n)
                      acc
                      (recur (inc i) (+ acc (max 0.0 (aget eigenvalues i))))))
        explained-ratio (double-array k)
        noise-var (if (< k n)
                    (let [remaining (- n k)]
                      (loop [i 0, acc 0.0]
                        (if (>= i remaining)
                          (/ acc (double remaining))
                          (recur (inc i)
                                 (+ acc (max 0.0 (aget eigenvalues i)))))))
                    0.0)]
    ;; Reverse: take last k eigenvectors (highest eigenvalues)
    (dotimes [i k]
      (let [eig-idx (- n 1 i)
            ev (max 0.0 (aget eigenvalues eig-idx))
            sv (n/sqrt (* ev (double (dec m))))]
        (aset singular-values i sv)
        (aset explained-var i ev)
        (aset explained-ratio i (if (> total-var 0.0) (/ ev total-var) 0.0))
        ;; Eigenvector row from eigenvectors matrix (row eig-idx)
        (acopy! eigenvectors (* eig-idx n) components (* i n) n)))
    ;; Sign flip for consistency
    (svd/svd-flip! components components k k n)
    (->PCAResult components singular-values explained-var explained-ratio
                 mean k n m noise-var)))

;; ================================================================
;; Randomized SVD solver
;; ================================================================

(deftm pca-randomized
  "PCA via randomized SVD. Best for large matrices with k << min(m,n).
  X[m,n] row-major, not modified."
  [X :- (Array double) m :- Long n :- Long
   n-components :- Long n-oversamples :- Long n-iter :- Long]
  :- PCAResult
  (let [;; Copy and center
        X-c (double-array (* m n))
        _ (acopy! X 0 X-c 0 (* m n))
        mean (double-array n)
        _ (column-mean! X-c mean m n)
        _ (center! X-c mean m n)
        ;; Randomized SVD
        result (svd/randomized-svd X-c m n n-components n-oversamples n-iter)
        U  ^doubles (aget ^objects result 0)
        S  ^doubles (aget ^objects result 1)
        Vt ^doubles (aget ^objects result 2)
        k n-components
        ;; Explained variance
        components (double-array (* k n))
        singular-values (double-array k)
        explained-var (double-array k)
        ;; Total variance from centered data: sum of column variances
        total-var (loop [j 0, acc 0.0]
                    (if (>= j n)
                      acc
                      (let [col-var (loop [i 0, s 0.0]
                                      (if (>= i m)
                                        (/ s (double (dec m)))
                                        (let [v (aget X-c (+ (* i n) j))]
                                          (recur (inc i) (+ s (* v v))))))]
                        (recur (inc j) (+ acc col-var)))))
        explained-ratio (double-array k)]
    ;; Copy results
    (dotimes [i k]
      (acopy! Vt (* i n) components (* i n) n))
    (acopy! S 0 singular-values 0 k)
    (dotimes [i k]
      (let [si (aget S i)
            ev (/ (* si si) (double (dec m)))]
        (aset explained-var i ev)
        (aset explained-ratio i (/ ev total-var))))
    (->PCAResult components singular-values explained-var explained-ratio
                 mean k n m 0.0)))

;; ================================================================
;; Auto solver selection
;; ================================================================

(deftm pca
  "PCA with automatic solver selection (sklearn heuristic).
  X[m,n] row-major, not modified."
  [X :- (Array double) m :- Long n :- Long n-components :- Long]
  :- PCAResult
  (let [mn (min m n)]
    (cond
      ;; Small matrices: use covariance_eigh
      (<= (max m n) 500)
      (pca-covariance-eigh X m n n-components)
      ;; Truncated: randomized SVD
      (< n-components (long (* 0.8 (double mn))))
      (pca-randomized X m n n-components 10 4)
      ;; Otherwise: covariance_eigh
      :else
      (pca-covariance-eigh X m n n-components))))

;; ================================================================
;; Transform / Inverse transform
;; ================================================================

(deftm transform
  "Project X[n-samples, n-features] onto principal components.
  Returns X_new[n-samples, n-components]."
  [X :- (Array double) result :- PCAResult n-samples :- Long]
  :- (Array double)
  (let [n-feat (long (:n-features result))
        n-comp (long (:n-components result))
        mean ^doubles (:mean result)
        components ^doubles (:components result)
        ;; Center X
        X-c (double-array (* n-samples n-feat))
        _ (acopy! X 0 X-c 0 (* n-samples n-feat))
        _ (center! X-c mean n-samples n-feat)
        ;; X_new = X_centered @ components^T  [n-samples, n-components]
        X-new (double-array (* n-samples n-comp))]
    (blas/dgemm-nt! X-c components X-new n-samples n-comp n-feat 1.0 0.0)
    X-new))

(deftm inverse-transform
  "Reconstruct X from transformed X_new[n-samples, n-components].
  Returns X_approx[n-samples, n-features]."
  [X-new :- (Array double) result :- PCAResult n-samples :- Long]
  :- (Array double)
  (let [n-feat (long (:n-features result))
        n-comp (long (:n-components result))
        mean ^doubles (:mean result)
        components ^doubles (:components result)
        ;; X_approx = X_new @ components + mean
        X-approx (double-array (* n-samples n-feat))]
    (blas/dgemm! X-new components X-approx n-samples n-comp n-feat 1.0 0.0)
    ;; Add mean
    (dotimes [i n-samples]
      (dotimes [j n-feat]
        (let [idx (+ (* i n-feat) j)]
          (aset X-approx idx (+ (aget X-approx idx) (aget mean j))))))
    X-approx))
