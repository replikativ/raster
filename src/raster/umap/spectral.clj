(ns raster.umap.spectral
  "Spectral initialization — the dim smallest non-trivial eigenvectors of the
  normalized graph Laplacian L = I - D^-1/2 A D^-1/2 (umap.spectral.spectral_layout).

  Uses dense LAPACK eigh (exact, run once). Fine for prototype-scale n; the
  scalable path is matrix-free Lanczos (raster.linalg.iterative/lanczos) with a
  (cI - L) spectral transform to reach the bottom of the spectrum — swap in later
  for large graphs.

  All array number-crunching is deftm; `spectral-init` is a thin orchestrator."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == abs max])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == abs max sqrt]]
            [raster.linalg.eigen :as eigen]))

;; Degree of each vertex = sum of incident edge weights. The graph is stored
;; symmetric (both directions), so summing over `head` covers every incidence.
(deftm degrees!
  [head :- (Array int) weights :- (Array double) deg :- (Array double) n-edges :- Long]
  :- (Array double)
  (dotimes [e n-edges]
    (let [i (long (aget head e))]
      (aset deg i (+ (aget deg i) (aget weights e)))))
  deg)

;; inv-sqrt-deg[i] = 1 / sqrt(deg[i])
(deftm inv-sqrt!
  [deg :- (Array double) isd :- (Array double) n :- Long] :- (Array double)
  (dotimes [i n] (aset isd i (/ 1.0 (sqrt (aget deg i)))))
  isd)

;; Dense normalized Laplacian from a symmetric edge list (both directions present).
;; Writes L[n*n] row-major: L = I - D^-1/2 A D^-1/2.
(deftm build-norm-laplacian!
  [head :- (Array int) tail :- (Array int) weights :- (Array double)
   inv-sqrt-deg :- (Array double) L :- (Array double) n :- Long n-edges :- Long]
  :- (Array double)
  (dotimes [t (* n n)] (aset L t 0.0))
  (dotimes [i n] (aset L (+ (* i n) i) 1.0))
  (dotimes [e n-edges]
    (let [i (long (aget head e))
          j (long (aget tail e))
          v (* (aget weights e) (* (aget inv-sqrt-deg i) (aget inv-sqrt-deg j)))]
      (aset L (+ (* i n) j) (- (aget L (+ (* i n) j)) v))))
  L)

;; Copy eigenvectors 1..dim (skip the trivial ~0 one) into the embedding columns.
;; evecs is row-major n×n with row r = eigenvector r (eigh output, ascending).
(deftm extract-eigvecs!
  [evecs :- (Array double) emb :- (Array double) n :- Long dim :- Long] :- (Array double)
  (dotimes [c dim]
    (let [src (* (+ c 1) n)]
      (dotimes [i n]
        (aset emb (+ (* i dim) c) (aget evecs (+ src i))))))
  emb)

;; Scale an embedding so max|coord| = expansion (umap scales spectral init to ~10).
(deftm scale-embedding!
  [emb :- (Array double) expansion :- Double] :- (Array double)
  (let [n (alength emb)
        mx (loop [i 0 m 0.0] (if (< i n) (recur (inc i) (max m (abs (aget emb i)))) m))
        s (/ expansion (max 1.0e-12 mx))]
    (dotimes [i n] (aset emb i (* (aget emb i) s)))
    emb))

;; Thin orchestrator: allocate, run the deftm kernels + LAPACK eigh, assemble.
(defn spectral-init
  [^ints head ^ints tail ^doubles weights n dim]
  (let [n (long n) dim (long dim)
        ne (alength weights)
        deg (double-array n)
        _   (degrees! head weights deg ne)
        isd (double-array n)
        _   (inv-sqrt! deg isd n)
        L   (double-array (* n n))
        _   (build-norm-laplacian! head tail weights isd L n ne)
        res (eigen/eigh L n)
        evals ^doubles (clojure.core/aget ^objects res 0)
        evecs ^doubles (clojure.core/aget ^objects res 1)
        emb (double-array (* n dim))
        _   (extract-eigvecs! evecs emb n dim)]
    {:emb emb
     :evals (vec (take (clojure.core/inc dim) evals))}))
