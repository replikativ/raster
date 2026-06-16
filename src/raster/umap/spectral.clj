(ns raster.umap.spectral
  "Spectral initialization — the dim smallest non-trivial eigenvectors of the
  normalized graph Laplacian L = I - D^-1/2 A D^-1/2 (umap.spectral.spectral_layout).

  Two paths, selected by graph size:
    - tiny n (<= dense-cutoff): dense LAPACK eigh (exact, O(n^3) but n is small).
    - large n: matrix-free Lanczos on the spectral-transformed operator
        M = I + D^-1/2 A D^-1/2          (eigenvalues in [0,2])
      The smallest eigenvectors of L are the LARGEST of M, and Lanczos converges
      fastest to the largest end of the spectrum. We take the top dim+1 of M,
      drop the trivial one (eigenvalue ~2, the all-ones-ish vector), and keep dim.

  All array number-crunching is deftm; `spectral-init` is a thin orchestrator."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == abs max min])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == abs max min sqrt]]
            [raster.math :as m]
            [raster.linalg.eigen :as eigen]
            [raster.linalg.iterative :as it]))

;; Below this vertex count, the dense O(n^3) eigh is faster than Lanczos setup.
(def ^:const dense-cutoff 256)

;; Degree of each vertex = sum of incident edge weights. The graph is stored
;; symmetric (both directions), so summing over `head` covers every incidence.
(deftm degrees!
  [head :- (Array int) weights :- (Array double) deg :- (Array double) n-edges :- Long]
  :- (Array double)
  (dotimes [e n-edges]
    (let [i (long (aget head e))]
      (aset deg i (+ (aget deg i) (aget weights e)))))
  deg)

;; inv-sqrt-deg[i] = 1 / sqrt(deg[i]); guard isolated vertices (deg 0) -> 0.
(deftm inv-sqrt!
  [deg :- (Array double) isd :- (Array double) n :- Long] :- (Array double)
  (dotimes [i n]
    (let [d (aget deg i)]
      (aset isd i (if (> d 0.0) (/ 1.0 (sqrt d)) 0.0))))
  isd)

;; ---------------------------------------------------------------------------
;; Dense path (tiny n)
;; ---------------------------------------------------------------------------

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

;; ---------------------------------------------------------------------------
;; Matrix-free path (large n)
;; ---------------------------------------------------------------------------

;; out = M x = x + D^-1/2 A (D^-1/2 x), matrix-free over the symmetric edge list.
;; tmp is scratch of length n. A is symmetric (both edge directions present), so
;; the single sweep over `head` accumulates the full sparse mat-vec.
(deftm norm-adj-matvec!
  [head :- (Array int) tail :- (Array int) weights :- (Array double)
   isd :- (Array double) x :- (Array double) out :- (Array double)
   tmp :- (Array double) n :- Long n-edges :- Long]
  :- (Array double)
  (dotimes [i n] (aset tmp i (* (aget isd i) (aget x i))))
  (dotimes [i n] (aset out i 0.0))
  (dotimes [e n-edges]
    (let [i (long (aget head e))
          j (long (aget tail e))]
      (aset out i (+ (aget out i) (* (aget weights e) (aget tmp j))))))
  (dotimes [i n] (aset out i (+ (aget x i) (* (aget isd i) (aget out i)))))
  out)

;; Copy lanczos eigenvector c+1 (object-array of double[], index 0 = trivial top
;; of M, dropped) into embedding column c.
(deftm pack-eigvec-column!
  [ev :- (Array double) emb :- (Array double) n :- Long dim :- Long c :- Long]
  :- (Array double)
  (dotimes [i n]
    (aset emb (+ (* i dim) c) (aget ev i)))
  emb)

;; ---------------------------------------------------------------------------
;; Shared post-processing
;; ---------------------------------------------------------------------------

;; Scale an embedding so max|coord| = expansion (umap scales spectral init to ~10).
(deftm scale-embedding!
  [emb :- (Array double) expansion :- Double] :- (Array double)
  (let [n (alength emb)
        mx (loop [i 0 m 0.0] (if (< i n) (recur (inc i) (max m (abs (aget emb i)))) m))
        s (/ expansion (max 1.0e-12 mx))]
    (dotimes [i n] (aset emb i (* (aget emb i) s)))
    emb))

;; ---------------------------------------------------------------------------
;; Orchestrator
;; ---------------------------------------------------------------------------

;; Exact eigh of the dense normalized Laplacian. Returns the n*dim row-major
;; embedding (smallest non-trivial eigenvectors). Used for tiny graphs.
(deftm dense-spectral
  [head :- (Array int) tail :- (Array int) weights :- (Array double)
   isd :- (Array double) n :- Long dim :- Long ne :- Long] :- (Array double)
  (let [L     (double-array (* n n))
        _     (build-norm-laplacian! head tail weights isd L n ne)
        res   (eigen/eigh L n)
        evecs (aget res 1)
        emb   (double-array (* n dim))
        _     (extract-eigvecs! evecs emb n dim)]
    emb))

;; Matrix-free Lanczos on M = I + D^-1/2 A D^-1/2. Returns the n*dim embedding,
;; or a zero-length array if Lanczos returned fewer than dim+1 eigenpairs (the
;; orchestrator detects this and falls back to the dense path).
(deftm lanczos-spectral
  [head :- (Array int) tail :- (Array int) weights :- (Array double)
   isd :- (Array double) n :- Long dim :- Long ne :- Long] :- (Array double)
  (let [tmp     (double-array n)
        mv      (ftm [x :- (Array double) out :- (Array double)] :- (Array double)
                  (norm-adj-matvec! head tail weights isd x out tmp n ne))
        k       (+ dim 1)
        ;; Enough Krylov steps to resolve the top k eigenpairs of a clustered
        ;; bottom-of-Laplacian spectrum, capped at n.
        maxiter (long (min (long n) (max (long 256) (* 8 k))))
        res     (it/lanczos mv n k 1.0e-4 maxiter)
        eigvecs (aget res 1)]
    (if (< (alength eigvecs) k)
      (double-array 0)
      (let [emb (double-array (* n dim))]
        (dotimes [c dim]
          (pack-eigvec-column! (aget eigvecs (+ c 1)) emb n dim c))
        emb))))

;; Spectral embedding of a CONNECTED symmetric weighted graph. Returns the n*dim
;; row-major embedding. Dense eigh for tiny graphs, matrix-free Lanczos otherwise;
;; falls back to dense if Lanczos under-delivers. (`spectral-init` handles the
;; disconnected case by calling this per component.)
(deftm connected-spectral
  [head :- (Array int) tail :- (Array int) weights :- (Array double)
   n :- Long dim :- Long] :- (Array double)
  (let [ne  (alength weights)
        deg (double-array n)
        _   (degrees! head weights deg ne)
        isd (double-array n)
        _   (inv-sqrt! deg isd n)]
    (if (<= n dense-cutoff)
      (dense-spectral head tail weights isd n dim ne)
      (let [e (lanczos-spectral head tail weights isd n dim ne)]
        (if (== (alength e) 0)
          (dense-spectral head tail weights isd n dim ne)
          e)))))

;; ---------------------------------------------------------------------------
;; Connected components (union-find) — for disconnected graphs
;; ---------------------------------------------------------------------------

;; Root of x's set (no compression here; union does the linking).
(deftm find-root [parent :- (Array int) x :- Long] :- Long
  (loop [r (long x)]
    (let [p (long (aget parent r))]
      (if (== p r) r (recur p)))))

;; Union-find over the edge list; writes parent[i] = component root for each i.
(deftm connected-components!
  [head :- (Array int) tail :- (Array int) parent :- (Array int)
   n :- Long ne :- Long] :- (Array int)
  (dotimes [i n] (aset parent i (int i)))
  (dotimes [e ne]
    (let [a (find-root parent (long (aget head e)))
          b (find-root parent (long (aget tail e)))]
      (when (not (== a b))
        (if (< a b) (aset parent b (int a)) (aset parent a (int b))))))
  (dotimes [i n] (aset parent i (int (find-root parent (long i)))))
  parent)

;; Relabel roots to dense ids in [0,c): label[i] = component id. Returns c.
;; remap is scratch of length n (root index -> dense id, -1 = unseen).
(deftm densify-labels!
  [parent :- (Array int) label :- (Array int) remap :- (Array int) n :- Long] :- Long
  (dotimes [i n] (aset remap i (int -1)))
  (loop [i 0 c 0]
    (if (< i n)
      (let [r (long (aget parent i))]
        (if (== (aget remap r) -1)
          (do (aset remap r (int c)) (aset label i (int c)) (recur (inc i) (inc c)))
          (do (aset label i (aget remap r)) (recur (inc i) c))))
      c)))

;; ---------------------------------------------------------------------------
;; Multi-component layout — per-component spectral + ±eye meta-placement
;; ---------------------------------------------------------------------------

;; ±eye meta-embedding of c components into dim-space (umap's `else` branch, used
;; when c <= 2*dim or no data is available). Walks the coordinate axes with
;; alternating sign and growing radius: components 0..dim-1 -> +e_axis, the next
;; dim -> -e_axis, etc. For c <= 2*dim this is exactly umap's ±eye placement.
(deftm meta-embedding!
  [meta :- (Array double) c :- Long dim :- Long] :- (Array double)
  (dotimes [t (* c dim)] (aset meta t 0.0))
  (loop [i 0 axis 0 sign 1.0 rad 1.0]
    (when (< i c)
      (aset meta (+ (* i dim) axis) (* sign rad))
      (let [axis2 (+ axis 1)]
        (if (>= axis2 dim)
          (if (> sign 0.0)
            (recur (+ i 1) 0 -1.0 rad)
            (recur (+ i 1) 0 1.0 (+ rad 1.0)))
          (recur (+ i 1) axis2 sign rad)))))
  meta)

;; --- Tier C: data-centroid component layout (umap component_layout) ---

;; Mean of each component's data points (X is n*data-dim row-major, any float type).
(deftm component-centroids!
  (All [T]
   [X :- (Array T) label :- (Array int) centroid :- (Array double)
    counts :- (Array double) n :- Long data-dim :- Long c :- Long]
   :- (Array double)
   (dotimes [t (* c data-dim)] (aset centroid t 0.0))
   (dotimes [i c] (aset counts i 0.0))
   (dotimes [i n]
     (let [lab (long (aget label i))]
       (aset counts lab (+ (aget counts lab) 1.0))
       (dotimes [d data-dim]
         (let [k (+ (* lab data-dim) d)]
           (aset centroid k (+ (aget centroid k) (double (aget X (+ (* i data-dim) d)))))))))
   (dotimes [lab c]
     (let [cnt (max 1.0 (aget counts lab))]
       (dotimes [d data-dim]
         (let [k (+ (* lab data-dim) d)]
           (aset centroid k (/ (aget centroid k) cnt))))))
   centroid))

;; Gaussian affinity exp(-||c_i - c_j||^2) between component centroids (c*c dense).
(deftm centroid-affinity!
  [centroid :- (Array double) affinity :- (Array double) c :- Long data-dim :- Long]
  :- (Array double)
  (dotimes [i c]
    (aset affinity (+ (* i c) i) 1.0)
    (dotimes [j c]
      (when (< j i)
        (let [d2 (loop [d 0 s 0.0]
                   (if (< d data-dim)
                     (let [diff (- (aget centroid (+ (* i data-dim) d))
                                   (aget centroid (+ (* j data-dim) d)))]
                       (recur (inc d) (+ s (* diff diff))))
                     s))
              a (m/exp (- 0.0 d2))]
          (aset affinity (+ (* i c) j) a)
          (aset affinity (+ (* j c) i) a)))))
  affinity)

;; Spectral embedding of the c*c dense affinity into dim dims (= meta centers).
;; Normalized Laplacian of the affinity (a complete, hence connected, graph), eigh,
;; eigenvectors 1..dim, scaled so max|coord| = 1 (umap component_layout tail).
(deftm affinity-meta-embedding!
  [affinity :- (Array double) meta :- (Array double) c :- Long dim :- Long]
  :- (Array double)
  (let [isd (double-array c)
        _   (dotimes [i c]
              (let [s (loop [j 0 acc 0.0]
                        (if (< j c) (recur (inc j) (+ acc (aget affinity (+ (* i c) j)))) acc))]
                (aset isd i (if (> s 0.0) (/ 1.0 (sqrt s)) 0.0))))
        L   (double-array (* c c))
        _   (dotimes [i c]
              (dotimes [j c]
                (let [v (* (aget affinity (+ (* i c) j)) (* (aget isd i) (aget isd j)))]
                  (aset L (+ (* i c) j) (- (if (== i j) 1.0 0.0) v)))))
        res (eigen/eigh L c)
        _   (extract-eigvecs! (aget res 1) meta c dim)
        nd  (* c dim)
        mx  (loop [t 0 mm 0.0] (if (< t nd) (recur (inc t) (max mm (abs (aget meta t)))) mm))
        s   (/ 1.0 (max 1.0e-12 mx))]
    (dotimes [t nd] (aset meta t (* (aget meta t) s)))
    meta))

;; Half the distance from component ci's meta-center to its nearest other center
;; (umap's data_range — sizes each component to fit the gap to its neighbour).
(deftm meta-data-ranges!
  [meta :- (Array double) ranges :- (Array double) c :- Long dim :- Long]
  :- (Array double)
  (dotimes [i c]
    (let [mn (loop [j 0 m 1.0e30]
               (if (< j c)
                 (if (== j i)
                   (recur (inc j) m)
                   (let [d2 (loop [d 0 s 0.0]
                              (if (< d dim)
                                (let [diff (- (aget meta (+ (* i dim) d)) (aget meta (+ (* j dim) d)))]
                                  (recur (inc d) (+ s (* diff diff))))
                                s))]
                     (recur (inc j) (min m (sqrt d2)))))
                 m))]
      (aset ranges i (/ (if (>= mn 1.0e30) 1.0 mn) 2.0))))
  ranges)

;; --- placement kernels (use meta-centers directly; per-component data-range scale) ---

;; Scale a (per-component) embedding in place so max|coord| = 1.
(deftm normalize-unit! [emb :- (Array double) n :- Long dim :- Long] :- (Array double)
  (let [nd (* n dim)
        mx (loop [i 0 m 0.0] (if (< i nd) (recur (inc i) (max m (abs (aget emb i)))) m))
        s  (/ 1.0 (max 1.0e-12 mx))]
    (dotimes [i nd] (aset emb i (* (aget emb i) s)))
    emb))

;; Write a component's unit-normalized sub-embedding into the global embedding at
;; its global vertex ids: sub * data-range + meta-center.
(deftm scatter-component!
  [sub :- (Array double) emb :- (Array double) verts :- (Array int)
   meta :- (Array double) ci :- Long sz :- Long dim :- Long data-range :- Double]
  :- (Array double)
  (dotimes [l sz]
    (let [g (long (aget verts l))]
      (dotimes [d dim]
        (aset emb (+ (* g dim) d)
              (+ (* data-range (aget sub (+ (* l dim) d)))
                 (aget meta (+ (* ci dim) d)))))))
  emb)

;; Place a too-small component (no meaningful spectral structure): all its vertices
;; sit at the meta-center plus a small deterministic spread scaled by data-range.
(deftm place-small!
  [emb :- (Array double) verts :- (Array int) meta :- (Array double)
   ci :- Long sz :- Long dim :- Long data-range :- Double] :- (Array double)
  (dotimes [l sz]
    (let [g (long (aget verts l))]
      (dotimes [d dim]
        (aset emb (+ (* g dim) d)
              (+ (aget meta (+ (* ci dim) d))
                 (* data-range (* 0.5 (- (/ (+ (double l) (* 0.37 (double d))) (double sz)) 0.5))))))))
  emb)

;; --- collection glue (dynamic per-component grouping; deftm can't express this) ---

(defn- component-vertices
  "Vector (indexed by component id) of int[] global vertex ids in that component."
  [^ints label ^long n ^long c]
  (let [lists (vec (repeatedly c #(java.util.ArrayList.)))]
    (dotimes [i n]
      (.add ^java.util.ArrayList (nth lists (clojure.core/aget label i)) (int i)))
    (mapv (fn [^java.util.ArrayList al]
            (let [a (int-array (.size al))]
              (dotimes [k (.size al)] (clojure.core/aset a k (int (.get al k))))
              a))
          lists)))

(defn- induced-subgraph
  "Sub-edge-list for component `ci` with LOCAL vertex ids (g2l must be filled for
  this component's vertices). Returns [head' tail' weights']. Edges are internal
  to a component by construction, so filtering on head's label suffices."
  [^ints head ^ints tail ^doubles weights ^ints label ci ^ints g2l ne]
  (let [ci (long ci) ne (long ne)
        sh (java.util.ArrayList.) st (java.util.ArrayList.) sw (java.util.ArrayList.)]
    (dotimes [e ne]
      (when (clojure.core/== ci (clojure.core/aget label (clojure.core/aget head e)))
        (.add sh (clojure.core/aget g2l (clojure.core/aget head e)))
        (.add st (clojure.core/aget g2l (clojure.core/aget tail e)))
        (.add sw (clojure.core/aget weights e))))
    (let [m (.size sh)
          h (int-array m) t (int-array m) w (double-array m)]
      (dotimes [k m]
        (clojure.core/aset h k (int (.get sh k)))
        (clojure.core/aset t k (int (.get st k)))
        (clojure.core/aset w k (double (.get sw k))))
      [h t w])))

(defn- data-component-centers!
  "Tier C: meta-centers from data-space component centroids (umap component_layout).
  Fills `meta` (c*dim) via the spectral embedding of the centroid affinity matrix."
  [X ^ints label ^doubles meta n c dim data-dim]
  (let [n (long n) c (long c) dim (long dim) data-dim (long data-dim)
        centroid (double-array (clojure.core/* c data-dim))
        counts   (double-array c)
        _ (component-centroids! X label centroid counts n data-dim c)
        affinity (double-array (clojure.core/* c c))
        _ (centroid-affinity! centroid affinity c data-dim)]
    (affinity-meta-embedding! affinity meta c dim)
    meta))

(defn- multi-component-layout
  "Embed each connected component spectrally and place it at its meta-center, scaled
  to its data-range (half the gap to the nearest neighbouring component). Meta-centers
  come from data centroids (Tier C) when c > 2*dim and X is available, else ±eye."
  [^ints head ^ints tail ^doubles weights ^ints label X n c dim data-dim]
  (let [n (long n) c (long c) dim (long dim)
        ne (alength weights)
        emb (double-array (clojure.core/* n dim))
        meta (double-array (clojure.core/* c dim))
        _ (if (and X (clojure.core/> c (clojure.core/* 2 dim)))
            (data-component-centers! X label meta n c dim data-dim)
            (meta-embedding! meta c dim))
        ranges (double-array c)
        _ (meta-data-ranges! meta ranges c dim)
        comp-verts (component-vertices label n c)
        g2l (int-array n)]
    (dotimes [ci c]
      (let [verts ^ints (nth comp-verts ci)
            sz (alength verts)
            dr (clojure.core/aget ranges ci)]
        (dotimes [l sz] (clojure.core/aset g2l (clojure.core/aget verts l) (int l)))
        (if (clojure.core/< sz (clojure.core/* 2 dim))
          (place-small! emb verts meta ci sz dim dr)
          (let [[sh st sw] (induced-subgraph head tail weights label ci g2l ne)
                sub (connected-spectral sh st sw sz dim)]
            (normalize-unit! sub sz dim)
            (scatter-component! sub emb verts meta ci sz dim dr)))))
    emb))

(defn spectral-init
  "Spectral embedding of a symmetric weighted graph (edge list head/tail/weights).
  Returns the n*dim row-major embedding. Single connected component -> direct
  spectral solve; multiple components -> per-component spectral + meta-placement.
  Pass `X` (data, n*data-dim) to enable data-centroid meta-placement (Tier C) for
  highly-fragmented graphs (c > 2*dim); without it, ±eye placement is used."
  ([head tail weights n dim] (spectral-init head tail weights n dim nil 0))
  ([^ints head ^ints tail ^doubles weights n dim X data-dim]
   (let [n (long n) dim (long dim)
         ne (alength weights)
         parent (int-array n)
         _ (connected-components! head tail parent n ne)
         label (int-array n)
         remap (int-array n)
         c (long (densify-labels! parent label remap n))]
     (if (clojure.core/== c 1)
       (connected-spectral head tail weights n dim)
       (multi-component-layout head tail weights label X n c dim data-dim)))))
