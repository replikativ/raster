(ns raster.umap.graph
  "Fuzzy simplicial set construction — ports umap.umap_.smooth_knn_dist and
  compute_membership_strengths (the per-point numeric kernels are deftm), plus
  the t-conorm symmetrization (graph bookkeeping, plain Clojure for now).

  Pipeline: kNN (dist,idx) -> smooth-knn-dist! (sigmas,rhos)
            -> membership-strengths! (per-edge weights)
            -> fuzzy-graph (symmetrize: A + Aᵀ - A∘Aᵀ) -> head/tail/weights."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == abs])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == abs]]
            [raster.math :as rm]
            [raster.linalg.sparse :as sp]))

;; SMOOTH_K_TOLERANCE=1e-5, MIN_K_DIST_SCALE=1e-3, local_connectivity=1, bandwidth=1.
;; dst: euclidean knn distances double[n*k] (sorted asc, self at col 0).
;; Writes sigmas[n], rhos[n]. rho = first non-zero distance in the row.
(deftm smooth-knn-dist!
  [dst :- (Array double) n :- Long k :- Long
   sigmas :- (Array double) rhos :- (Array double)]
  :- (Array double)
  (let [target (rm/log2 (double k))
        tol 1.0e-5
        minscale 0.001
        nk (* n k)
        total (loop [t 0 s 0.0] (if (< t nk) (recur (inc t) (+ s (aget dst t))) s))
        mean-all (/ total (double nk))]
    (dotimes [i n]
      (let [ib (* i k)
            rho (loop [t 0]
                  (if (< t k)
                    (if (> (aget dst (+ ib t)) 0.0) (aget dst (+ ib t)) (recur (inc t)))
                    0.0))
            rsum (loop [t 0 s 0.0] (if (< t k) (recur (inc t) (+ s (aget dst (+ ib t)))) s))
            row-mean (/ rsum (double k))]
        (aset rhos i rho)
        (let [sigma (loop [it 0 lo 0.0 hi 1.0e308 mid 1.0]
                      (if (< it 64)
                        (let [psum (loop [j 1 s 0.0]
                                     (if (< j k)
                                       (let [d (- (aget dst (+ ib j)) rho)]
                                         (recur (inc j)
                                                (+ s (if (> d 0.0) (rm/exp (/ (- 0.0 d) mid)) 1.0))))
                                       s))]
                          (if (< (abs (- psum target)) tol)
                            mid
                            (if (> psum target)
                              (recur (inc it) lo mid (/ (+ lo mid) 2.0))
                              (if (== hi 1.0e308)
                                (recur (inc it) mid hi (* mid 2.0))
                                (recur (inc it) mid hi (/ (+ mid hi) 2.0))))))
                        mid))
              sigma2 (if (> rho 0.0)
                       (if (< sigma (* minscale row-mean)) (* minscale row-mean) sigma)
                       (if (< sigma (* minscale mean-all)) (* minscale mean-all) sigma))]
          (aset sigmas i sigma2))))
    sigmas))

;; Per-edge membership: val = exp(-(d-rho)/sigma), with self -> 0, (d-rho)<=0 or
;; sigma==0 -> 1. Writes vals[n*k] aligned with the knn idx/dst layout.
(deftm membership-strengths!
  [idx :- (Array int) dst :- (Array double)
   sigmas :- (Array double) rhos :- (Array double)
   n :- Long k :- Long vals :- (Array double)]
  :- (Array double)
  (dotimes [i n]
    (let [ib (* i k) rho (aget rhos i) sig (aget sigmas i)]
      (dotimes [j k]
        (let [c (long (aget idx (+ ib j)))
              v (if (== c -1)
                  0.0
                  (if (== c i)
                    0.0
                    (let [dr (- (aget dst (+ ib j)) rho)]
                      (if (or (<= dr 0.0) (== sig 0.0))
                        1.0
                        (rm/exp (/ (- 0.0 dr) sig))))))]
          (aset vals (+ ib j) v)))))
  vals)

;; --- symmetrization via raster.linalg.sparse (reuse the CSR substrate) ---
;; The directed memberships form a sparse matrix A (A[i, knn_idx[i,j]] = val);
;; the symmetric fuzzy graph is the t-conorm W = A + Aᵀ - A∘Aᵀ, built from
;; coo-to-csr + csr-transpose + csr-add + csr-hadamard. Same CSR type that
;; spmv / spectral-Lanczos / EVoC's neighbor graph consume — no bespoke sparse code.

;; Fill COO triplets from the knn membership layout, dropping self-edges and
;; zero memberships (umap's eliminate_zeros). Output arrays are sized n*k (the
;; max); returns the actual nnz written so coo-to-csr reads only the prefix.
(deftm fill-membership-coo!
  [idx :- (Array int) vals :- (Array double) n :- Long k :- Long
   rowidx :- (Array int) colidx :- (Array int) values :- (Array double)] :- Long
  (loop [i 0 w 0]
    (if (< i n)
      (recur (inc i)
             (loop [j 0 w2 w]
               (if (< j k)
                 (let [p (+ (* i k) j)
                       c (aget idx p)
                       v (aget vals p)]
                   (if (and (not (== (long c) -1)) (> v 0.0))
                     (do (aset rowidx w2 (int i))
                         (aset colidx w2 c)
                         (aset values w2 v)
                         (recur (inc j) (inc w2)))
                     (recur (inc j) w2)))
                 w2)))
      w)))

;; Expand a symmetric CSR's (rowptr,colidx,values) into directed edge arrays.
(deftm graph->edges!
  [rowptr :- (Array int) colidx :- (Array int) values :- (Array double) n :- Long
   head :- (Array int) tail :- (Array int) ws :- (Array double)] :- (Array int)
  (dotimes [i n]
    (loop [p (long (aget rowptr i))]
      (when (< p (long (aget rowptr (+ i 1))))
        (aset head p (int i))
        (aset tail p (aget colidx p))
        (aset ws p (aget values p))
        (recur (+ p 1)))))
  head)

(defn fuzzy-graph
  "Symmetric fuzzy simplicial set as a CSRMatrix: W = A + Aᵀ - A∘Aᵀ, where
  A[i, knn_idx[i,j]] = membership val. Thin glue around the deftm COO fill +
  raster.linalg.sparse CSR ops."
  [^ints idx ^doubles vals n k]
  (let [n (long n) k (long k)
        cap (clojure.core/* n k)
        rowidx (int-array cap) colidx (int-array cap) values (double-array cap)
        nnz (fill-membership-coo! idx vals n k rowidx colidx values)
        coo (sp/->COOMatrix rowidx colidx values n n nnz)
        a    (sp/coo-to-csr coo)
        at   (sp/csr-transpose a)
        prod (sp/csr-hadamard a at)
        summ (sp/csr-add a 1.0 at 1.0)]
    (sp/csr-add summ 1.0 prod -1.0)))

(defn graph->edges
  "Expand a symmetric CSRMatrix into directed edge arrays for the layout."
  [W]
  (let [nnz (long (.-nnz W))
        head (int-array nnz) tail (int-array nnz) ws (double-array nnz)]
    (graph->edges! (.-rowptr W) (.-colidx W) (.-values W) (long (.-nrows W)) head tail ws)
    {:head head :tail tail :weights ws}))

(defn symmetrize
  "Convenience: fuzzy graph -> directed edge arrays {:head :tail :weights}."
  [idx vals n k]
  (graph->edges (fuzzy-graph idx vals n k)))
