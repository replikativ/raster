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
            [raster.math :as rm]))

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

;; --- symmetrization (graph bookkeeping; plain Clojure) ---
;; Combine directed memberships into the symmetric fuzzy graph
;;   W[i,j] = A[i,j] + A[j,i] - A[i,j]*A[j,i]
;; Returns {:head int[] :tail int[] :weights double[]} over all nonzero (i,j).
(defn symmetrize
  [^ints idx ^doubles vals ^long n ^long k]
  (let [a (java.util.HashMap.)]                       ; (i*n+j) -> A[i,j]
    (dotimes [i n]
      (dotimes [j k]
        (let [c (clojure.core/aget idx (clojure.core/+ (clojure.core/* i k) j))
              v (clojure.core/aget vals (clojure.core/+ (clojure.core/* i k) j))]
          (when (clojure.core/and (clojure.core/not= c -1) (clojure.core/> v 0.0))
            (.put a (clojure.core/+ (clojure.core/* (long i) n) (long c)) v)))))
    (let [seen (java.util.HashSet.)
          heads (java.util.ArrayList.) tails (java.util.ArrayList.) ws (java.util.ArrayList.)]
      (doseq [e (.entrySet a)]
        (let [key (long (.getKey e))
              i (quot key n) j (rem key n)
              kk (clojure.core/+ (clojure.core/* j n) i)]
          (when (clojure.core/not (.contains seen key))
            (.add seen key) (.add seen kk)
            (let [aij (double (.getValue e))
                  aji (double (or (.get a kk) 0.0))
                  w (clojure.core/- (clojure.core/+ aij aji) (clojure.core/* aij aji))]
              (when (clojure.core/> w 0.0)
                ;; emit both directions (undirected edge -> 2 COO entries, as umap does)
                (.add heads (int i)) (.add tails (int j)) (.add ws w)
                (.add heads (int j)) (.add tails (int i)) (.add ws w))))))
      {:head (int-array heads) :tail (int-array tails)
       :weights (double-array ws)})))
