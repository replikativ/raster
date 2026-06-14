(ns raster.umap
  "UMAP embedding layout optimization — a faithful port of
  umap.layouts.optimize_layout_euclidean (Python+numba is the gold standard).

  This is the SGD inner loop shared by UMAP and EVoC node-embedding: per-edge
  attractive forces + negative-sampled repulsive forces, with a Tausworthe RNG
  for the negative samples. The whole multi-epoch solve compiles to a single
  JVM method via `compile-aot`.

  Validation strategy (UMAP SGD is chaotically sensitive, so exact coordinate
  matching is not meaningful):
    1. deterministic gate — single attractive-only epoch matches the float64
       reference to ~1e-12,
    2. RNG gate — `tau-rand-int!` reproduces umap.utils.tau_rand_int bit-for-bit
       (JVM long == numba int64),
    3. quality gate — full-run trustworthiness(X) matches Python UMAP (~0.956).

  Arithmetic goes through raster.numeric / raster.arrays so the walker sees it."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= ==
                            mod bit-and bit-or bit-xor
                            bit-shift-left bit-shift-right unsigned-bit-shift-right])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == pow mod
                                    bit-and bit-or bit-xor
                                    bit-shift-left bit-shift-right
                                    unsigned-bit-shift-right]]))

;; ----------------------------------------------------------------------
;; Tausworthe RNG — replicates umap.utils.tau_rand_int exactly.
;; numba int64 arithmetic (wraparound, arithmetic >>) == JVM long, so mirroring
;; the same expressions reproduces the negative-sample stream bit-for-bit.
;; state is a flat long[] of n_vertices*3; `base` = vertex*3. Mutates in place,
;; returns the draw as a *signed* int32 widened to long (matching the i4 return).
;; ----------------------------------------------------------------------
(deftm tau-rand-int! [state :- (Array long) base :- Long] :- Long
  (let [s0 (aget state base)
        s1 (aget state (+ base 1))
        s2 (aget state (+ base 2))
        n0 (bit-xor (bit-and (bit-shift-left (bit-and s0 4294967294) 12) 4294967295)
                    (bit-shift-right (bit-xor (bit-and (bit-shift-left s0 13) 4294967295) s0) 19))
        n1 (bit-xor (bit-and (bit-shift-left (bit-and s1 4294967288) 4) 4294967295)
                    (bit-shift-right (bit-xor (bit-and (bit-shift-left s1 2) 4294967295) s1) 25))
        n2 (bit-xor (bit-and (bit-shift-left (bit-and s2 4294967280) 17) 4294967295)
                    (bit-shift-right (bit-xor (bit-and (bit-shift-left s2 3) 4294967295) s2) 11))]
    (aset state base n0)
    (aset state (+ base 1) n1)
    (aset state (+ base 2) n2)
    (let [m (bit-and (bit-xor (bit-xor n0 n1) n2) 4294967295)]
      ;; reinterpret low 32 bits as signed int32
      (if (>= m 2147483648) (- m 4294967296) m))))

;; Clamp gradient into [-4, 4] (umap.layouts.clip).
(deftm uclip [v :- Double] :- Double
  (if (> v 4.0) 4.0 (if (< v -4.0) -4.0 v)))

;; ----------------------------------------------------------------------
;; The SGD layout optimizer. Mirrors optimize_layout_euclidean with
;; move_other=true (head_embedding IS tail_embedding — standard UMAP).
;; All buffers are mutated in place; returns the embedding.
;;
;;   emb     flat double[n*dim], row-major          (mutated)
;;   head/tail  int[n_edges] 1-simplex endpoints
;;   eps     epochs_per_sample        double[n_edges]
;;   epn     epochs_per_negative_sample = eps/neg_rate
;;   eons    epoch_of_next_sample      (init = eps copy, mutated)
;;   eonsn   epoch_of_next_negative_sample (init = epn copy, mutated)
;;   states  long[n_vertices*3] per-vertex RNG state (mutated)
;; ----------------------------------------------------------------------
(deftm optimize-layout!
  (All [T] [emb :- (Array T) head :- (Array int) tail :- (Array int)
   eps :- (Array double) epn :- (Array double)
   eons :- (Array double) eonsn :- (Array double)
   states :- (Array long)
   a :- Double b :- Double gamma :- Double init-alpha :- Double
   n-vertices :- Long dim :- Long n-epochs :- Long]
  :- (Array T)
  (let [n-edges (alength eps)
        bm1 (- b 1.0)]
    (dotimes [ep n-epochs]
      (let [epd (double ep)
            ;; umap updates alpha at the end of each epoch; epoch ep therefore
            ;; runs with init*(1-(ep-1)/N) (and ep=0 samples nothing).
            alpha (if (== ep 0)
                    init-alpha
                    (* init-alpha (- 1.0 (/ (double (dec ep)) (double n-epochs)))))]
        (dotimes [i n-edges]
          (when (<= (aget eons i) epd)
            (let [j (long (aget head i))
                  k (long (aget tail i))
                  jb (* j dim)
                  kb (* k dim)
                  d2 (loop [d 0 acc 0.0]
                       (if (< d dim)
                         (recur (inc d)
                                (let [df (- (aget emb (+ jb d)) (aget emb (+ kb d)))]
                                  (+ acc (* df df))))
                         acc))
                  gc (if (> d2 0.0)
                       (/ (* (* (* -2.0 a) b) (pow d2 bm1))
                          (+ (* a (pow d2 b)) 1.0))
                       0.0)]
              ;; attractive update (move both endpoints)
              (dotimes [d dim]
                (let [cd (aget emb (+ jb d))
                      od (aget emb (+ kb d))
                      g (uclip (* gc (- cd od)))
                      ga (* g alpha)]
                  (aset emb (+ jb d) (+ cd ga))
                  (aset emb (+ kb d) (- od ga))))
              (aset eons i (+ (aget eons i) (aget eps i)))
              ;; negative samples
              (let [n-neg (long (/ (- epd (aget eonsn i)) (aget epn i)))]
                (dotimes [p n-neg]
                  (let [kk (mod (tau-rand-int! states (* j 3)) n-vertices)
                        kb2 (* kk dim)
                        d2n (loop [d 0 acc 0.0]
                              (if (< d dim)
                                (recur (inc d)
                                       (let [df (- (aget emb (+ jb d)) (aget emb (+ kb2 d)))]
                                         (+ acc (* df df))))
                                acc))
                        gcn (if (> d2n 0.0)
                              (/ (* (* 2.0 gamma) b)
                                 (* (+ 0.001 d2n) (+ (* a (pow d2n b)) 1.0)))
                              0.0)]
                    (dotimes [d dim]
                      (let [cd (aget emb (+ jb d))
                            od (aget emb (+ kb2 d))
                            g (if (> gcn 0.0) (uclip (* gcn (- cd od))) 0.0)]
                        (aset emb (+ jb d) (+ cd (* g alpha)))))))
                (aset eonsn i (+ (aget eonsn i) (* (double n-neg) (aget epn i))))))))))
    emb)))
