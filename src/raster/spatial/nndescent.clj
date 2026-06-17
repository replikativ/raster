(ns raster.spatial.nndescent
  "Approximate kNN via nearest-neighbor descent — ports EVoC's float NN-descent
  (the large-n, high-dim graph-kNN path where KD-trees fail). RP-tree forest init
  + random init seed a per-point max-heap kNN graph; iterative local join
  (neighbor-of-neighbor) refines it; deheap-sort + -log2(cos) gives the result.

  Faithful-in-result: same algorithm (RP-tree init + local join), validated by
  recall vs brute-force (NN-descent is approximate; EVoC's own output is
  non-deterministic). EVoC's blocking/working-set details are cache micro-opts
  that don't change recall, so they're omitted. Cosine metric on L2-normalized
  rows (distance = -dot = fast_cosine), matching knn_graph's float path.

  All numeric work deftm; reuses rptree + tau-rand-int! + raster.math/log2."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == mod bit-and bit-or])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == mod bit-and bit-or]]
            [raster.math :as rm]
            [raster.par :as par]
            [raster.tausworthe :as u]
            [raster.knn :as knn]
            [raster.spatial.rptree :as rp]))

;; distance = -dot (fast_cosine on L2-normalized rows; smaller = nearer)
(deftm cos-dist
  "Negative cosine between rows a and b of a flat n*dim array (= -dot on
   L2-normalized rows). Parametric over element type so it serves both the f64
   default and umap's f32 path; written as par/reduce so the dot vectorizes once
   the backend handles affine-offset loads. (Accumulates in f64 even for f32 data
   — more precise than umap's f32 rdist, irrelevant to kNN ordering/recall.)"
  (All [T] [data :- (Array T) a :- Long b :- Long dim :- Long] :- Double
    (let [ab (* a dim) bb (* b dim)]
      (- 0.0 (par/reduce acc 0.0 d dim
                         (+ acc (* (aget data (+ ab d)) (aget data (+ bb d)))))))))

(deftm eucl-dist
  "Squared euclidean distance between rows a and b (monotone in euclidean, so the
   kNN ordering/heap is identical; the orchestrator sqrts the kept distances)."
  (All [T] [data :- (Array T) a :- Long b :- Long dim :- Long] :- Double
    (let [ab (* a dim) bb (* b dim)]
      (par/reduce acc 0.0 d dim
                  (let [df (- (aget data (+ ab d)) (aget data (+ bb d)))]
                    (+ acc (* df df)))))))

;; Bounded max-heap push with dedup + "new" flag, on point's sub-array [base,base+k).
;; Root = worst (max) distance. Returns 1 if inserted, 0 otherwise.
(deftm flagged-heap-push!
  [dist :- (Array double) ind :- (Array int) flags :- (Array int)
   base :- Long size :- Long p :- Double nn :- Long flag :- Long] :- Long
  (if (>= p (aget dist base))
    0
    (let [dup (loop [t 0] (if (< t size)
                            (if (== (aget ind (+ base t)) nn) 1 (recur (inc t)))
                            0))]
      (if (== dup 1)
        0
        (let [fin (loop [i 0]
                    (let [ic1 (+ (* 2 i) 1) ic2 (+ ic1 1)]
                      (if (>= ic1 size)
                        i
                        (let [isw (if (>= ic2 size)
                                    (if (> (aget dist (+ base ic1)) p) ic1 -1)
                                    (if (>= (aget dist (+ base ic1)) (aget dist (+ base ic2)))
                                      (if (< p (aget dist (+ base ic1))) ic1 -1)
                                      (if (< p (aget dist (+ base ic2))) ic2 -1)))]
                          (if (== isw -1)
                            i
                            (do (aset dist (+ base i) (aget dist (+ base isw)))
                                (aset ind (+ base i) (aget ind (+ base isw)))
                                (aset flags (+ base i) (aget flags (+ base isw)))
                                (recur isw)))))))]
          (aset dist (+ base fin) p) (aset ind (+ base fin) (int nn)) (aset flags (+ base fin) (int flag))
          1)))))

;; seed: self (dist=-1 -> cos 1) + k random neighbors per point
(deftm init-random!
  (All [T] [data :- (Array T) dist :- (Array double) ind :- (Array int) flags :- (Array int)
   rng-state :- (Array long) n :- Long dim :- Long k :- Long metric :- Long] :- (Array double)
  (dotimes [i n]
    (flagged-heap-push! dist ind flags (* i k) k
                        (if (== metric 1) (eucl-dist data i i dim) (cos-dist data i i dim)) i 1)
    (dotimes [t k]
      (let [j (mod (u/tau-rand-int! rng-state 0) n)]
        (when (not (== j i))
          (flagged-heap-push! dist ind flags (* i k) k
                              (if (== metric 1) (eucl-dist data i j dim) (cos-dist data i j dim)) j 1)))))
  dist))

;; seed from RP-tree leaves: all-pairs within each leaf
(deftm init-leaves!
  (All [T] [data :- (Array T) dist :- (Array double) ind :- (Array int) flags :- (Array int)
   idx :- (Array int) leaf-start :- (Array int) leaf-end :- (Array int)
   n-leaves :- Long dim :- Long k :- Long] :- (Array double)
  (dotimes [lf n-leaves]
    (let [s (long (aget leaf-start lf)) e (long (aget leaf-end lf))]
      (loop [p s]
        (when (< p e)
          (loop [q (+ p 1)]
            (when (< q e)
              (let [a (long (aget idx p)) b (long (aget idx q))
                    dd (cos-dist data a b dim)]
                (flagged-heap-push! dist ind flags (* a k) k dd b 1)
                (flagged-heap-push! dist ind flags (* b k) k dd a 1))
              (recur (inc q))))
          (recur (inc p))))))
  dist))

;; one refinement iteration: local join over each point's neighbor list (push
;; both directions makes the graph symmetric -> reverse neighbors propagate).
(deftm local-join!
  (All [T] [data :- (Array T) dist :- (Array double) ind :- (Array int) flags :- (Array int)
   n :- Long dim :- Long k :- Long] :- Long
  (loop [i 0 upd 0]
    (if (< i n)
      (let [ib (* i k)
            u2 (loop [j 0 u upd]
                 (if (< j k)
                   (let [a (long (aget ind (+ ib j)))]
                     (if (< a 0)
                       (recur (inc j) u)
                       (let [u3 (loop [l (+ j 1) uu u]
                                  (if (< l k)
                                    (let [b (long (aget ind (+ ib l)))]
                                      (if (< b 0)
                                        (recur (inc l) uu)
                                        (let [dd (cos-dist data a b dim)
                                              r1 (flagged-heap-push! dist ind flags (* a k) k dd b 1)
                                              r2 (flagged-heap-push! dist ind flags (* b k) k dd a 1)]
                                          (recur (inc l) (+ (+ uu r1) r2)))))
                                    uu))]
                         (recur (inc j) u3))))
                   u))]
        (recur (inc i) u2))
      upd))))

;; local join over a block of points [bstart,bend) — for CPU-multicore via
;; Clojure futures (raster par = SIMD/GPU, not CPU-thread; these irregular heap
;; loops need thread-level parallelism). Concurrent heap pushes across blocks are
;; Hogwild races, as EVoC's parallel mode accepts.
(deftm local-join-block!
  [data :- (Array double) dist :- (Array double) ind :- (Array int) flags :- (Array int)
   bstart :- Long bend :- Long dim :- Long k :- Long] :- Long
  (loop [i bstart upd 0]
    (if (< i bend)
      (let [ib (* i k)
            u2 (loop [j 0 u upd]
                 (if (< j k)
                   (let [a (long (aget ind (+ ib j)))]
                     (if (< a 0)
                       (recur (inc j) u)
                       (let [u3 (loop [l (+ j 1) uu u]
                                  (if (< l k)
                                    (let [b (long (aget ind (+ ib l)))]
                                      (if (< b 0)
                                        (recur (inc l) uu)
                                        (let [dd (cos-dist data a b dim)
                                              r1 (flagged-heap-push! dist ind flags (* a k) k dd b 1)
                                              r2 (flagged-heap-push! dist ind flags (* b k) k dd a 1)]
                                          (recur (inc l) (+ (+ uu r1) r2)))))
                                    uu))]
                         (recur (inc j) u3))))
                   u))]
        (recur (inc i) u2))
      upd)))

;; Cache-blocked / race-free local join: each point pulls candidates from its
;; neighbors-of-neighbors and updates ONLY ITS OWN heap. When run per block, a
;; thread writes only its block's heaps (owned, contiguous -> cache-local, no
;; write races); neighbor-list reads of other blocks are benign stale reads
;; (Hogwild-tolerant). This is the parallelizable form of EVoC's block-owned apply.
(deftm local-join-owned!
  (All [T] [data :- (Array T) dist :- (Array double) ind :- (Array int) flags :- (Array int)
   bstart :- Long bend :- Long dim :- Long k :- Long] :- Long
  (loop [p bstart upd 0]
    (if (< p bend)
      (let [pb (* p k)
            u2 (loop [j 0 u upd]
                 (if (< j k)
                   (let [a (long (aget ind (+ pb j)))]
                     (if (< a 0)
                       (recur (inc j) u)
                       (let [ab (* a k)
                             u3 (loop [l 0 uu u]
                                  (if (< l k)
                                    (let [b (long (aget ind (+ ab l)))]
                                      (if (or (< b 0) (== b p))
                                        (recur (inc l) uu)
                                        (let [dd (cos-dist data p b dim)
                                              r (flagged-heap-push! dist ind flags pb k dd b 1)]
                                          (recur (inc l) (+ uu r)))))
                                    uu))]
                         (recur (inc j) u3))))
                   u))]
        (recur (inc p) u2))
      upd))))

(defn parallel-local-join!
  "CPU-multicore local-join via futures over point-blocks. Race-free writes
   (each thread owns its block's heaps); cache-local."
  [data dist ind flags n dim k n-threads]
  (let [n (long n) n-threads (long n-threads)
        bs (long (Math/ceil (/ (double n) n-threads)))]
    (->> (range n-threads)
         (mapv (fn [t] (future (local-join-owned! data dist ind flags
                                                  (long (* t bs)) (long (min n (* (inc (long t)) bs)))
                                                  (long dim) (long k)))))
         (mapv deref)
         (reduce clojure.core/+))))

;; Reverse adjacency for one round: rev[a*rmax + c] lists up to rmax points that
;; have `a` as a forward neighbour (capped — like pynndescent's max_candidates).
;; rcnt[a] = how many were stored.
(deftm build-reverse!
  [ind :- (Array int) rev :- (Array int) rcnt :- (Array int)
   n :- Long k :- Long rmax :- Long] :- (Array int)
  (dotimes [i n] (aset rcnt i (int 0)))
  (dotimes [i n]
    (dotimes [j k]
      (let [a (long (aget ind (+ (* i k) j)))]
        (when (>= a 0)
          (let [c (long (aget rcnt a))]
            (when (< c rmax)
              (aset rev (+ (* a rmax) c) (int i))
              (aset rcnt a (int (+ c 1)))))))))
  rev)

;; Reverse-augmented owned local-join. For point p, explore 2-hop candidates via
;; BOTH forward neighbours (forward(p) -> forward(a)) AND reverse neighbours
;; (q points at p -> push q, and explore forward(q)). Writes only p's heap
;; (race-free, cache-local). This is the full NN-descent neighbourhood join.
(deftm local-join-owned-rev!
  (All [T] [data :- (Array T) dist :- (Array double) ind :- (Array int) flags :- (Array int)
            rev :- (Array int) rcnt :- (Array int)
            bstart :- Long bend :- Long dim :- Long k :- Long rmax :- Long metric :- Long] :- Long
  (loop [p bstart upd 0]
    (if (< p bend)
      (let [pb (* p k)
            ;; (1) forward first-hop a, explore forward(a)
            u1 (loop [j 0 u upd]
                 (if (< j k)
                   (let [a (long (aget ind (+ pb j)))]
                     (if (< a 0)
                       (recur (inc j) u)
                       (let [ab (* a k)
                             uu (loop [l 0 v u]
                                  (if (< l k)
                                    (let [b (long (aget ind (+ ab l)))]
                                      (if (or (< b 0) (== b p))
                                        (recur (inc l) v)
                                        (let [dd (if (== metric 1) (eucl-dist data p b dim) (cos-dist data p b dim))]
                                          (recur (inc l) (+ v (flagged-heap-push! dist ind flags pb k dd b 1))))))
                                    v))]
                         (recur (inc j) uu))))
                   u))
            ;; (2) reverse first-hop q: push q, then explore forward(q)
            rc (long (aget rcnt p))
            u2 (loop [j 0 u u1]
                 (if (< j rc)
                   (let [q (long (aget rev (+ (* p rmax) j)))]
                     (if (or (< q 0) (== q p))
                       (recur (inc j) u)
                       (let [dq (if (== metric 1) (eucl-dist data p q dim) (cos-dist data p q dim))
                             u' (+ u (flagged-heap-push! dist ind flags pb k dq q 1))
                             qb (* q k)
                             uu (loop [l 0 v u']
                                  (if (< l k)
                                    (let [b (long (aget ind (+ qb l)))]
                                      (if (or (< b 0) (== b p))
                                        (recur (inc l) v)
                                        (let [dd (if (== metric 1) (eucl-dist data p b dim) (cos-dist data p b dim))]
                                          (recur (inc l) (+ v (flagged-heap-push! dist ind flags pb k dd b 1))))))
                                    v))]
                         (recur (inc j) uu))))
                   u))]
        (recur (inc p) u2))
      upd))))

(defn parallel-local-join-rev!
  "Reverse-augmented owned local-join across point-blocks (futures). Race-free."
  [data dist ind flags rev rcnt n dim k rmax metric n-threads]
  (let [n (long n) n-threads (long (max 1 (long n-threads)))
        bs (long (Math/ceil (/ (double n) n-threads)))]
    (->> (range n-threads)
         (mapv (fn [t] (future (local-join-owned-rev! data dist ind flags rev rcnt
                                                      (long (* t bs)) (long (min n (* (inc (long t)) bs)))
                                                      (long dim) (long k) (long rmax) (long metric)))))
         (mapv deref)
         (reduce clojure.core/+))))

;; sort each point's k entries ascending by stored distance (-cos), then emit the
;; raw -cos values (unfilled heap slots stay Double/MAX_VALUE). cosine-knn applies
;; neg-cos->log2! once to convert to -log2(cos) (1e308 for far/unfilled) — matching
;; the brute path. (Converting here too would double-convert and destroy the graph.)
(deftm finalize!
  [dist :- (Array double) ind :- (Array int) n :- Long k :- Long
   out-idx :- (Array int) out-dst :- (Array double)] :- (Array int)
  (dotimes [i n]
    (let [ib (* i k)]
      (loop [a 1]
        (when (< a k)
          (let [dv (aget dist (+ ib a)) iv (aget ind (+ ib a))]
            (loop [b (- a 1)]
              (if (and (>= b 0) (> (aget dist (+ ib b)) dv))
                (do (aset dist (+ ib (+ b 1)) (aget dist (+ ib b)))
                    (aset ind (+ ib (+ b 1)) (aget ind (+ ib b)))
                    (recur (- b 1)))
                (do (aset dist (+ ib (+ b 1)) dv) (aset ind (+ ib (+ b 1)) (int iv))))))
          (recur (inc a))))
      (dotimes [t k]
        (aset out-idx (+ ib t) (aget ind (+ ib t)))
        (aset out-dst (+ ib t) (aget dist (+ ib t))))))
  out-idx)

;; --- orchestrator ---
(defn nn-descent
  "Approximate cosine kNN of X (flat n*dim, will be L2-normalized in place).
   n-trees RP-trees + random init, then n-iters local-join refinement.
   Returns {:idx int[n*k] :dst double[n*k] (raw -cos, self first)}; cosine-knn
   applies neg-cos->log2! to convert to -log2(cos)."
  ;; X may be double[] or float[] — the kernels are parametric (All [T]).
  ;; :metric 0 = cosine (L2-normalize + RP-tree init), 1 = euclidean (no normalize,
  ;; random init only — angular RP-trees don't fit euclidean geometry).
  [X n dim k & {:keys [n-trees n-iters leaf-size seed n-threads max-candidates metric]
               :or {leaf-size 30 seed 42 n-threads 1 max-candidates 30 metric 0}}]
  (let [n (long n) dim (long dim) k (long k) metric (long metric)
        ;; scale forest size + iterations with n (pynndescent heuristics) — a fixed
        ;; small forest gives terrible recall at large n (garbage graph).
        n-trees (long (or n-trees (min 32 (+ 5 (long (Math/round (/ (Math/sqrt (double n)) 20.0)))))))
        n-iters (long (or n-iters (max 12 (long (Math/round (/ (Math/log (double n)) (Math/log 2.0)))))))
        _ (when (clojure.core/zero? metric) (knn/l2-normalize! X n dim))
        dist (double-array (clojure.core/* n k))
        ind (int-array (clojure.core/* n k))
        flags (int-array (clojure.core/* n k))
        _ (java.util.Arrays/fill dist Double/MAX_VALUE)
        _ (java.util.Arrays/fill ind (int -1))
        rng (long-array [(long seed) (long (clojure.core/+ seed 7919)) (long (clojure.core/+ seed 104729))])
        ;; RP-tree forest init (parallel over trees when n-threads>1; each tree
        ;; has its own idx/hp/stk + rng, init-leaves pushes to the shared heap Hogwild)
        build-tree! (fn [tree-i]
                      (let [idx (int-array n) hp (double-array dim)
                            stk (int-array (clojure.core/* 4 n)) ls (int-array n) le (int-array n)
                            rng-t (long-array [(long (clojure.core/+ seed (clojure.core/* tree-i 1000003)))
                                               (long (clojure.core/+ seed (clojure.core/* tree-i 1000033) 7919))
                                               (long (clojure.core/+ seed (clojure.core/* tree-i 1000037) 104729))])
                            nl (rp/build-rptree-leaves! X n dim leaf-size idx rng-t hp stk ls le)]
                        (init-leaves! X dist ind flags idx ls le (long nl) dim k)))]
    ;; angular RP-trees only seed the cosine metric; euclidean uses random init
    (when (clojure.core/zero? metric)
      (if (clojure.core/> n-threads 1)
        (->> (range n-trees) (mapv (fn [ti] (future (build-tree! ti)))) (mapv deref))
        (dotimes [ti n-trees] (build-tree! ti))))
    (init-random! X dist ind flags rng n dim k metric)
    ;; refine — reverse-augmented owned local-join (rev capped at rmax candidates)
    (let [rmax (long (max (long max-candidates) k))
          rev (int-array (clojure.core/* n rmax))
          rcnt (int-array n)
          ;; warm the parametric kernel for X's dtype on an empty range BEFORE the
          ;; futures fan out — concurrent first-call specialization races to null.
          _ (local-join-owned-rev! X dist ind flags rev rcnt 0 0 dim k rmax metric)]
    (loop [it 0]
      (when (clojure.core/< it n-iters)
        ;; Build the reverse adjacency for this round, then do a reverse-augmented
        ;; owned local-join: each point explores 2-hop neighbours via BOTH its
        ;; forward neighbours and the points that point AT it (reverse). This is the
        ;; core NN-descent recall mechanism; forward-only plateaus ~0.78 recall.
        (build-reverse! ind rev rcnt n k rmax)
        (let [upd (parallel-local-join-rev! X dist ind flags rev rcnt n dim k rmax metric n-threads)]
          (when (clojure.core/> upd (quot (clojure.core/* n k) 50))   ; stop when <2% updated
            (recur (clojure.core/inc it))))))
    (let [out-idx (int-array (clojure.core/* n k)) out-dst (double-array (clojure.core/* n k))]
      (finalize! dist ind n k out-idx out-dst)
      ;; euclidean stored squared distances (monotone) — convert to true euclidean
      (when (clojure.core/== metric 1)
        (dotimes [i (clojure.core/* n k)]
          (clojure.core/aset ^doubles out-dst i (Math/sqrt (clojure.core/aget ^doubles out-dst i)))))
      {:idx out-idx :dst out-dst}))))

;; nn-descent stores -cos (on L2-normalized rows). Convert to -log2(cos) so the
;; approximate path matches the brute path and feeds UMAP's fuzzy-set a proper
;; [0,∞) distance (self=0). cos = -stored; order is preserved (monotone in cos).
(deftm neg-cos->log2! [d :- (Array double) m :- Long] :- (Array double)
  (dotimes [i m]
    (let [cos (- 0.0 (aget d i))]
      (aset d i (if (> cos 0.0) (- 0.0 (rm/log2 cos)) 1.0e308))))
  d)

(defn cosine-knn
  "Cosine kNN that auto-selects exact brute-force (small n) vs approximate
   NN-descent (large n). Returns {:idx :dst} (dst = -log2 cos, self first),
   matching knn_graph's float path. X is L2-normalized in place either way.

   :n-threads parallelizes the NN-descent path (per-tree RP-forest build + block-
   owned local-join, Hogwild-tolerant — like pynndescent's parallel mode, which is
   also non-deterministic). Defaults to all cores; the per-op scalar cos-dist dot
   is at the JVM floor, so multicore is the lever that matches numba (~2.8x @8c)."
  [X n dim k & {:keys [threshold n-threads]
                :or {threshold 4096
                     n-threads (.availableProcessors (Runtime/getRuntime))}}]
  (let [n (long n) dim (long dim) k (long k)]
    ;; cos-dist (and knn-brute-cosine!) both assume L2-normalized rows, so
    ;; normalize once up front — covers BOTH paths (norms vary in real data).
    (knn/l2-normalize! X n dim)
    (if (clojure.core/< n threshold)
      (let [oi (int-array (clojure.core/* n k)) od (double-array (clojure.core/* n k))]
        (knn/knn-brute-cosine! X n dim k oi od)
        {:idx oi :dst od})
      (let [r (nn-descent X n dim k :n-threads n-threads)]
        (neg-cos->log2! (:dst r) (clojure.core/* n k))
        r))))

(defn euclidean-knn
  "Euclidean kNN that auto-selects exact brute-force (small n) vs approximate
   NN-descent (large n). Returns {:idx :dst} (dst = euclidean distance, self
   first at 0). X is NOT normalized. Mirrors umap's metric='euclidean'."
  [X n dim k & {:keys [threshold n-threads]
                :or {threshold 4096
                     n-threads (.availableProcessors (Runtime/getRuntime))}}]
  (let [n (long n) dim (long dim) k (long k)]
    (if (clojure.core/< n threshold)
      (let [oi (int-array (clojure.core/* n k)) od (double-array (clojure.core/* n k))]
        (knn/knn-brute! X n dim k oi od)
        {:idx oi :dst od})
      (nn-descent X n dim k :n-threads n-threads :metric 1))))

;; ====================================================================
;; int8 / uint8 NN-descent — quantized large-data path (EVoC dtype paths).
;; int8 and uint8 are both byte[] in the JVM (can't dispatch by type), so one
;; byte NN-descent selects the distance via `mode`: 0 = int8 (signed quantized
;; inner product, -Σx·y), 1 = uint8 (bit-Jaccard, -popcount(x&y)/popcount(x|y)).
;; Reuses the flagged-heap; random init + owned local-join (RP-trees are float-
;; specific in EVoC; random init + descent converges for the quantized paths).
;; ====================================================================

(deftm byte-dist [data :- (Array byte) a :- Long b :- Long dim :- Long mode :- Long] :- Double
  (let [ab (* a dim) bb (* b dim)]
    (if (== mode 0)
      ;; int8: negative signed inner product
      (loop [i 0 s 0]
        (if (< i dim)
          (recur (inc i) (+ s (* (long (aget data (+ ab i))) (long (aget data (+ bb i))))))
          (- 0.0 (double s))))
      ;; uint8: negative bit-Jaccard (popcount over unsigned bytes)
      (loop [i 0 r 0 dn 0]
        (if (< i dim)
          (let [x (bit-and (long (aget data (+ ab i))) 255)
                y (bit-and (long (aget data (+ bb i))) 255)]
            (recur (inc i)
                   (+ r (Long/bitCount (bit-and x y)))
                   (+ dn (Long/bitCount (bit-or x y)))))
          (if (> dn 0) (- 0.0 (/ (double r) (double dn))) 0.0))))))

(deftm init-random-bytes!
  [data :- (Array byte) dist :- (Array double) ind :- (Array int) flags :- (Array int)
   rng-state :- (Array long) n :- Long dim :- Long k :- Long mode :- Long] :- (Array double)
  (dotimes [i n]
    (flagged-heap-push! dist ind flags (* i k) k (byte-dist data i i dim mode) i 1)
    (dotimes [t k]
      (let [j (mod (u/tau-rand-int! rng-state 0) n)]
        (when (not (== j i))
          (flagged-heap-push! dist ind flags (* i k) k (byte-dist data i j dim mode) j 1)))))
  dist)

(deftm local-join-bytes!
  [data :- (Array byte) dist :- (Array double) ind :- (Array int) flags :- (Array int)
   n :- Long dim :- Long k :- Long mode :- Long] :- Long
  (loop [p 0 upd 0]
    (if (< p n)
      (let [pb (* p k)
            u2 (loop [j 0 u upd]
                 (if (< j k)
                   (let [a (long (aget ind (+ pb j)))]
                     (if (< a 0)
                       (recur (inc j) u)
                       (let [ab (* a k)
                             u3 (loop [l 0 uu u]
                                  (if (< l k)
                                    (let [b (long (aget ind (+ ab l)))]
                                      (if (or (< b 0) (== b p))
                                        (recur (inc l) uu)
                                        (let [dd (byte-dist data p b dim mode)
                                              r (flagged-heap-push! dist ind flags pb k dd b 1)]
                                          (recur (inc l) (+ uu r)))))
                                    uu))]
                         (recur (inc j) u3))))
                   u))]
        (recur (inc p) u2))
      upd)))

;; sort each point's k entries ascending by stored distance; emit idx + dist.
(deftm finalize-bytes!
  [dist :- (Array double) ind :- (Array int) n :- Long k :- Long
   out-idx :- (Array int) out-dst :- (Array double)] :- (Array int)
  (dotimes [i n]
    (let [ib (* i k)]
      (loop [a 1]
        (when (< a k)
          (let [dv (aget dist (+ ib a)) iv (aget ind (+ ib a))]
            (loop [b (- a 1)]
              (if (and (>= b 0) (> (aget dist (+ ib b)) dv))
                (do (aset dist (+ ib (+ b 1)) (aget dist (+ ib b)))
                    (aset ind (+ ib (+ b 1)) (aget ind (+ ib b)))
                    (recur (- b 1)))
                (do (aset dist (+ ib (+ b 1)) dv) (aset ind (+ ib (+ b 1)) (int iv))))))
          (recur (inc a))))
      (dotimes [t k]
        (aset out-idx (+ ib t) (aget ind (+ ib t)))
        (aset out-dst (+ ib t) (aget dist (+ ib t))))))
  out-idx)

(defn nn-descent-bytes
  "Approximate kNN of quantized byte data (n*dim). mode 0=int8 (signed inner
   product), 1=uint8 (bit-Jaccard). Random init + owned local-join descent.
   Returns {:idx :dst} (dst = the stored metric, smaller=nearer, self first)."
  [^bytes X n dim k mode & {:keys [n-iters seed] :or {n-iters 15 seed 42}}]
  (let [n (long n) dim (long dim) k (long k) mode (long mode)
        dist (double-array (clojure.core/* n k)) ind (int-array (clojure.core/* n k))
        flags (int-array (clojure.core/* n k))
        _ (java.util.Arrays/fill dist Double/MAX_VALUE) _ (java.util.Arrays/fill ind (int -1))
        rng (long-array [(long seed) (long (clojure.core/+ seed 7919)) (long (clojure.core/+ seed 104729))])]
    (init-random-bytes! X dist ind flags rng n dim k mode)
    (loop [it 0]
      (when (clojure.core/< it n-iters)
        (let [upd (local-join-bytes! X dist ind flags n dim k mode)]
          (when (clojure.core/> upd (quot (clojure.core/* n k) 50))
            (recur (clojure.core/inc it))))))
    (let [oi (int-array (clojure.core/* n k)) od (double-array (clojure.core/* n k))]
      (finalize-bytes! dist ind n k oi od)
      {:idx oi :dst od})))
