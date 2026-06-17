(ns raster.cluster.mst
  "Mutual-reachability minimum spanning tree — the heart of EVoC's (HDBSCAN*-family)
  clustering, run on the node embedding. Ports the result of EVoC's
  parallel_boruvka: an MST over d_mr(a,b) = max(core_a, core_b, rdist(a,b)), where
  core_i = rdist (squared euclidean) to the min_samples-th neighbor and rdist is
  squared euclidean. The MST is unique for distinct weights, so an exact dense
  Prim MST reproduces Boruvka's tree.

  Dense Prim is O(n²) — correct and exact for prototype scale; the KDTree+Boruvka
  approach EVoC uses is the large-n optimization (a later swap). Reuses
  raster.knn/knn-brute! for the core distances."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == max])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == max sqrt]]
            [raster.knn :as knn]))

;; core[i] = rdist (squared euclidean) to the min_samples-th neighbor. dst is the
;; euclidean knn-brute! output with k = min_samples+1 (self at col 0), so the
;; min_samples-th neighbor is column `ms`; squaring gives the reduced distance.
(deftm extract-core!
  [dst :- (Array double) n :- Long k :- Long ms :- Long core :- (Array double)]
  :- (Array double)
  (dotimes [i n]
    (let [d (aget dst (+ (* i k) ms))]
      (aset core i (* d d))))
  core)

;; Dense Prim MST over mutual-reachability distance. Fills mst-from/to/w with the
;; n-1 edges (weights in rdist units, matching EVoC); returns the edge count.
(deftm prim-mst!
  [emb :- (Array double) core :- (Array double) n :- Long dim :- Long
   mst-from :- (Array int) mst-to :- (Array int) mst-w :- (Array double)] :- Long
  (let [key (double-array n)
        parent (int-array n)
        visited (int-array n)]
    (dotimes [j n] (aset key j 1.0e308) (aset parent j (int -1)))
    (aset key 0 0.0)
    (loop [iter 0 e 0]
      (if (< iter n)
        ;; pick the unvisited vertex with the smallest key
        (let [u (loop [j 0 best -1 bestk 1.0e308]
                  (if (< j n)
                    (if (and (== (aget visited j) 0) (< (aget key j) bestk))
                      (recur (inc j) j (aget key j))
                      (recur (inc j) best bestk))
                    best))]
          (if (< u 0)
            e
            (let [_  (aset visited u 1)
                  ub (* (long u) dim)
                  cu (aget core u)
                  e2 (if (>= (aget parent u) 0)
                       (do (aset mst-from e (aget parent u))
                           (aset mst-to e (int u))
                           ;; store actual-distance mutual reachability (EVoC stores
                           ;; sqrt of the rdist d_mr used for comparison)
                           (aset mst-w e (sqrt (aget key u)))
                           (+ e 1))
                       e)]
              ;; relax keys of remaining vertices through u
              (dotimes [v n]
                (when (== (aget visited v) 0)
                  (let [vb (* (long v) dim)
                        d2 (loop [d 0 acc 0.0]
                             (if (< d dim)
                               (recur (inc d)
                                      (let [df (- (aget emb (+ ub d)) (aget emb (+ vb d)))]
                                        (+ acc (* df df))))
                               acc))
                        mr (max cu (max (aget core v) d2))]
                    (when (< mr (aget key v))
                      (aset key v mr)
                      (aset parent v (int u))))))
              (recur (inc iter) e2))))
        e))))

;; --- sort MST edges by weight (replaces boxed Clojure sort-by + marshaling) ---
;; In-place max-heap sift-down on an index array `ord`, keyed by w[ord[.]].
;; Total order on edge indices: by weight, ties broken by index (so the heapsort
;; is deterministic and equivalent to a stable ascending sort — matches EVoC).
(deftm edge-less? [w :- (Array double) a :- Long b :- Long] :- Long
  (if (or (< (aget w a) (aget w b))
          (and (== (aget w a) (aget w b)) (< a b)))
    1 0))

(deftm sift-down! [w :- (Array double) ord :- (Array int) root0 :- Long end :- Long] :- Long
  (loop [root root0]
    (let [child (+ (* 2 root) 1)]
      (if (> child end)
        root
        (let [child2 (if (and (<= (+ child 1) end)
                              (== 1 (edge-less? w (aget ord child) (aget ord (+ child 1)))))
                       (+ child 1) child)]
          (if (== 1 (edge-less? w (aget ord root) (aget ord child2)))
            (let [tmp (aget ord root)]
              (aset ord root (aget ord child2))
              (aset ord child2 tmp)
              (recur child2))
            root))))))

;; Heapsort: fill `ord` with edge indices sorted ascending by weight. O(m log m).
(deftm argsort-weights! [w :- (Array double) m :- Long ord :- (Array int)] :- (Array int)
  (dotimes [i m] (aset ord i (int i)))
  (loop [i (- (quot m 2) 1)]
    (when (>= i 0) (sift-down! w ord i (- m 1)) (recur (- i 1))))
  (loop [end (- m 1)]
    (when (> end 0)
      (let [tmp (aget ord 0)] (aset ord 0 (aget ord end)) (aset ord end tmp))
      (sift-down! w ord 0 (- end 1))
      (recur (- end 1))))
  ord)

;; Gather edges into weight-sorted order given the sorted index array.
(deftm gather-edges!
  [from :- (Array int) to :- (Array int) w :- (Array double) ord :- (Array int) m :- Long
   sf :- (Array int) st :- (Array int) sd :- (Array double)] :- (Array double)
  (dotimes [i m]
    (let [e (aget ord i)]
      (aset sf i (aget from e)) (aset st i (aget to e)) (aset sd i (aget w e))))
  sd)

;; Thin glue: sort an MST edge list by weight ascending.
(defn sort-edges
  [{:keys [^ints from ^ints to ^doubles w]}]
  (let [m (alength w)
        ord (int-array m) sf (int-array m) st (int-array m) sd (double-array m)]
    (argsort-weights! w m ord)
    (gather-edges! from to w ord m sf st sd)
    {:from sf :to st :w sd}))

;; Size-thresholded MST: dense O(n²) Prim wins below ~5k (tight loop, low
;; constants); KD-tree Boruvka O(n log n) wins above. Picks the faster path so
;; clustering is optimal across scales. (boruvka required lazily to avoid a cycle.)
(def ^:const boruvka-threshold 5000)
(declare mutual-reachability-mst)

(defn mst
  "Mutual-reachability MST, auto-selecting dense Prim (small n) vs KD-tree
   Boruvka (large n). Returns {:from :to :w :core}."
  [^doubles emb n dim min-samples]
  (if (clojure.core/< (long n) boruvka-threshold)
    (mutual-reachability-mst emb n dim min-samples)
    ((requiring-resolve 'raster.cluster.boruvka/boruvka-mst) emb n dim min-samples)))

;; Thin orchestrator: kNN -> core distances -> dense Prim MST.
;; Returns {:from int[] :to int[] :w double[]} (n-1 edges, weights in rdist units).
(defn mutual-reachability-mst
  [^doubles emb n dim min-samples]
  (let [n (long n) dim (long dim) ms (long min-samples)
        k (clojure.core/inc ms)                       ; self + ms neighbors
        kidx (int-array (clojure.core/* n k))
        kdst (double-array (clojure.core/* n k))
        _ (knn/knn-brute! emb n dim k kidx kdst)
        core (double-array n)
        _ (extract-core! kdst n k ms core)
        from (int-array (clojure.core/dec n))
        to (int-array (clojure.core/dec n))
        w (double-array (clojure.core/dec n))
        _ (prim-mst! emb core n dim from to w)]
    {:from from :to to :w w :core core}))
