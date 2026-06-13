(ns raster.umap.spectral
  "Spectral initialization — the dim smallest non-trivial eigenvectors of the
  normalized graph Laplacian L = I - D^-1/2 A D^-1/2 (umap.spectral.spectral_layout).

  Uses dense LAPACK eigh (exact, run once). Fine for prototype-scale n; the
  scalable path is matrix-free Lanczos (raster.linalg.iterative/lanczos) with a
  (cI - L) spectral transform to reach the bottom of the spectrum — swap in later
  for large graphs."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= ==])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= ==]]
            [raster.linalg.eigen :as eigen]))

;; Dense normalized Laplacian from a symmetric edge list (both directions present).
;; inv-sqrt-deg[i] = 1/sqrt(deg_i). Writes L[n*n] row-major.
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

;; Returns a flat double[n*dim] spectral embedding: columns are eigenvectors
;; 1..dim of L (skipping the trivial ~0 eigenvector). Not yet scaled.
(defn spectral-init
  [^ints head ^ints tail ^doubles weights n dim]
  (let [n (long n) dim (long dim)
        ne (alength weights)
        deg (double-array n)
        _ (dotimes [e ne]
            (let [i (clojure.core/aget head e)]
              (aset deg i (clojure.core/+ (clojure.core/aget deg i)
                                          (clojure.core/aget weights e)))))
        isd (double-array n)
        _ (dotimes [i n]
            (aset isd i (clojure.core// 1.0 (Math/sqrt (clojure.core/aget deg i)))))
        L (double-array (clojure.core/* n n))
        _ (build-norm-laplacian! head tail weights isd L n ne)
        res (eigen/eigh L n)
        evals ^doubles (clojure.core/aget ^objects res 0)
        evecs ^doubles (clojure.core/aget ^objects res 1)   ; row c = eigenvector c
        emb (double-array (clojure.core/* n dim))]
    ;; skip eigenvector 0 (trivial), take 1..dim into embedding columns
    (dotimes [c dim]
      (let [src-row (clojure.core/* (clojure.core/+ c 1) n)]
        (dotimes [i n]
          (aset emb (clojure.core/+ (clojure.core/* i dim) c)
                (clojure.core/aget evecs (clojure.core/+ src-row i))))))
    {:emb emb
     :evals (vec (take (clojure.core/inc dim) evals))}))

;; umap scales the spectral embedding to ~[-10,10] and adds small gaussian noise.
;; (noise omitted here — deterministic; layout SGD tolerates the scale.)
(defn scale-embedding!
  [^doubles emb expansion]
  (let [n (alength emb)
        mx (loop [i 0 m 0.0] (if (clojure.core/< i n)
                               (recur (clojure.core/inc i)
                                      (Math/max m (Math/abs (clojure.core/aget emb i)))) m))
        s (clojure.core// (double expansion) (Math/max 1.0e-12 mx))]
    (dotimes [i n] (aset emb i (clojure.core/* (clojure.core/aget emb i) s)))
    emb))
