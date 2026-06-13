(ns raster.umap.fit
  "End-to-end UMAP: kNN -> fuzzy simplicial set -> low-dim init -> SGD layout.

  A thin orchestrator (plain defn) over the deftm kernels in raster.umap,
  raster.umap.{knn,graph,spectral} and raster.spatial.nndescent. Python+numba
  UMAP (umap.umap_.UMAP / simplicial_set_embedding) is the gold standard.

  Pipeline (mirrors umap.umap_):
    1. cosine kNN  — exact brute (small n) or NN-descent (large n)
    2. smooth-knn-dist! + membership-strengths! -> fuzzy simplicial set
    3. symmetrize (t-conorm A+Aᵀ-A∘Aᵀ) -> directed edge list
    4. init      — spectral (small n) or random uniform[-10,10] (large n)
    5. make-epochs-per-sample + optimize-layout! (per-edge attractive +
       negative-sampled SGD, Tausworthe RNG)

  The numeric kernels stay deftm/compile-aot; only the wiring is plain Clojure."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == mod])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == mod]]
            [raster.umap :as u]
            [raster.umap.knn :as knn]
            [raster.umap.graph :as graph]
            [raster.umap.spectral :as spectral]
            [raster.spatial.nndescent :as nnd]))

;; a,b curve params for spread=1.0, min_dist=0.1 (umap.umap_.find_ab_params).
(def A 1.5769434603113077)
(def B 0.8950608779109733)

;; make_epochs_per_sample: an edge of weight w is sampled ∝ w, so its spacing in
;; epochs is wmax/w (the strongest edge every epoch). w==0 -> -1 (never sampled).
(deftm make-epochs-per-sample!
  [weights :- (Array double) eps :- (Array double) m :- Long] :- (Array double)
  (let [wmax (loop [i 0 mx 0.0]
               (if (< i m)
                 (recur (inc i) (if (> (aget weights i) mx) (aget weights i) mx))
                 mx))]
    (dotimes [i m]
      (let [w (aget weights i)]
        (aset eps i (if (> w 0.0) (/ wmax w) -1.0))))
    eps))

;; epn[i] = eps[i] / negative_sample_rate  (epochs_per_negative_sample).
(deftm epochs-per-neg!
  [eps :- (Array double) epn :- (Array double) m :- Long neg-rate :- Double]
  :- (Array double)
  (dotimes [i m] (aset epn i (/ (aget eps i) neg-rate)))
  epn)

;; ---- init / RNG seeding (data setup — plain Clojure, clojure.core arrays) ----

(defn- random-init
  "Uniform [-10,10] init, umap.umap_ init='random'."
  ^doubles [n dim seed]
  (let [r (java.util.Random. (long seed))
        e (double-array (clojure.core/* (long n) (long dim)))]
    (dotimes [i (clojure.core/alength e)]
      (clojure.core/aset-double e i (clojure.core/- (clojure.core/* 20.0 (.nextDouble r)) 10.0)))
    e))

(defn- add-noise!
  "Add N(0,1e-4) jitter, as umap does after spectral init."
  ^doubles [^doubles e seed]
  (let [r (java.util.Random. (clojure.core/+ (long seed) 1))]
    (dotimes [i (clojure.core/alength e)]
      (clojure.core/aset-double e i (clojure.core/+ (clojure.core/aget e i)
                                                    (clojure.core/* 1.0e-4 (.nextGaussian r)))))
    e))

(defn- init-states
  "Per-vertex LFSR113 state long[n*3]. Low bits forced high enough (≥0x1000) to
  avoid the degenerate seed region of the three Tausworthe generators."
  ^longs [n seed]
  (let [r (java.util.Random. (long seed))
        st (long-array (clojure.core/* (long n) 3))]
    (dotimes [v (clojure.core/alength st)]
      (clojure.core/aset st v (clojure.core/bit-or
                                (clojure.core/bit-and (.nextLong r) 0xFFFFFFFF) 0x1000)))
    st))

(defn fit
  "Fit a UMAP embedding of X (flat row-major double[n*dim]).
  Options: :k (neighbors, 15) :out-dim (2) :n-epochs (auto 500/200)
           :neg-rate (5.0) :gamma (1.0) :init (:auto|:spectral|:random) :seed (42).
  Returns {:emb double[n*out-dim] :n n :dim out-dim :n-edges ...}."
  [^doubles X n dim & {:keys [k out-dim n-epochs neg-rate gamma init seed]
                       :or {k 15 out-dim 2 neg-rate 5.0 gamma 1.0 init :auto seed 42}}]
  (let [n (long n) dim (long dim) k (long k) out-dim (long out-dim)
        ne (long (or n-epochs (if (clojure.core/> n 10000) 200 500)))
        nk (clojure.core/* n k)
        ;; 1. cosine kNN (normalizes X in place; -log2 cos distances, self first)
        {:keys [idx dst]} (nnd/cosine-knn X n dim k)
        ;; 2. fuzzy simplicial set
        sigmas (double-array n) rhos (double-array n)
        _ (graph/smooth-knn-dist! dst n k sigmas rhos)
        vals (double-array nk)
        _ (graph/membership-strengths! idx dst sigmas rhos n k vals)
        ;; 3. symmetric fuzzy graph -> directed edges
        {:keys [head tail weights]} (graph/symmetrize idx vals n k)
        n-edges (long (alength weights))
        mode (if (= init :auto) (if (clojure.core/> n 8000) :random :spectral) init)
        ;; 4. low-dim init
        emb (case mode
              :spectral (let [{e :emb} (spectral/spectral-init head tail weights n out-dim)]
                          (spectral/scale-embedding! e 10.0)
                          (add-noise! e seed))
              :random (random-init n out-dim seed))
        ;; 5. epochs-per-sample + per-vertex RNG state, then the SGD solve
        eps (double-array n-edges)
        _ (make-epochs-per-sample! weights eps n-edges)
        epn (double-array n-edges)
        _ (epochs-per-neg! eps epn n-edges (double neg-rate))
        eons (aclone eps) eonsn (aclone epn)
        states (init-states n seed)]
    (u/optimize-layout! emb head tail eps epn eons eonsn states
                        A B (double gamma) 1.0 n out-dim ne)
    {:emb emb :n n :dim out-dim :n-edges n-edges :init mode}))
