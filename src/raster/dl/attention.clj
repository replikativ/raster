(ns raster.dl.attention
  "Attention mechanisms for the Raster deep learning framework.

  Provides:
    scaled-dot-product-attention - standard QKV attention
    multi-head-attention         - multi-head self-attention with projections
    graph-attention              - graph attention with scatter-based message passing

  All ops are deftm with rrules for reverse AD.
  Parametric over element type T (works with both double[] and float[])."
  (:refer-clojure :exclude [aget aset alength aclone + - * /])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone acopy! alloc-like]]
            [raster.ad.templates :as tmpl]
            [raster.math :as m]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.dl.nn :as nn]
            [raster.dl.array-ops :as ops]
            [raster.linalg.blas :as blas]))

;; ================================================================
;; Scaled dot-product attention (single head)
;; Q:[seq_q, dk], K:[seq_k, dk], V:[seq_k, dv] -> output:[seq_q, dv]
;; ================================================================

(deftm scaled-dot-product-attn (All [T]
                                    [Q :- (Array T) K :- (Array T) V :- (Array T)
                                     seq-q :- Long seq-k :- Long dk :- Long dv :- Long]
                                    :- (Array T)
                                    (let [scale (/ 1.0 (n/sqrt (double dk)))
        ;; scores = Q @ K^T / sqrt(dk) -> [seq_q, seq_k]
                                          scores (alloc-like Q (* seq-q seq-k))
                                          _ (blas/dgemm-nt! Q K scores seq-q dk seq-k
                                                            (n/oftype Q 1.0) (n/oftype Q 0.0))
        ;; Scale scores
                                          _ (dotimes [i (* seq-q seq-k)]
                                              (aset scores i (* (aget scores i) scale)))
        ;; softmax over seq_k dimension for each query
                                          _ (dotimes [i seq-q]
                                              (let [offset (* i (int seq-k))
                                                    max-s (loop [j 0 m (n/neg-inf-val (aget scores 0))]
                                                            (if (< j seq-k)
                                                              (recur (inc j) (n/max m (aget scores (+ offset j))))
                                                              m))
                                                    sum-exp (loop [j 0 s 0.0]
                                                              (if (< j seq-k)
                                                                (let [e (m/exp (- (aget scores (+ offset j)) max-s))]
                                                                  (aset scores (+ offset j) e)
                                                                  (recur (inc j) (+ s e)))
                                                                s))
                                                    inv-sum (/ 1.0 sum-exp)]
                                                (dotimes [j seq-k]
                                                  (aset scores (+ offset j)
                                                        (* (aget scores (+ offset j)) inv-sum)))))
        ;; output = weights @ V -> [seq_q, dv]
                                          out (nn/matmul scores V seq-q seq-k dv)]
                                      out)))

;; rrule for scaled-dot-product-attn (pullback operates on double[])

;; ================================================================
;; Causal scaled dot-product attention (for autoregressive models)
;; Same as above but masks future positions (j > i) with -inf before softmax.
;; ================================================================

;; --- Rotary position embedding (RoPE) ---
;; NeoX/HF "rotate-half" convention over the full head_dim, positions 0..seq-1.
;; x is [seq, heads, head_dim] flattened. theta is the RoPE base — Gemma uses
;; 10000 for local/sliding layers and 1e6 for global layers (pass per layer);
;; Llama/Qwen use a single base. Generic across all RoPE decoder LMs.
(deftm rope (All [T] [x :- (Array T) seq-len :- Long heads :- Long
                      head-dim :- Long theta :- Double] :- (Array T)
                 (let [out (alloc-like x (* seq-len (* heads head-dim)))
                       half (quot head-dim 2)
        ;; inv_freq_i = theta^(-2i/d) = exp(-2i/d * ln theta); no pow intrinsic.
                       ln-theta (m/log theta)]
                   (dotimes [p seq-len]
                     (dotimes [h heads]
                       (let [base (+ (* p (* heads head-dim)) (* h (int head-dim)))]
                         (dotimes [i half]
                           (let [freq (m/exp (* (/ (* -2.0 (double i)) (double head-dim)) ln-theta))
                                 ang (* (double p) freq)
                                 c (m/cos ang) s (m/sin ang)
                                 x0 (aget x (+ base i))
                                 x1 (aget x (+ (+ base i) half))]
                             (aset out (+ base i) (- (* x0 c) (* x1 s)))
                             (aset out (+ (+ base i) half) (+ (* x1 c) (* x0 s))))))))
                   out)))

;; RoPE backward: RoPE applies the orthogonal rotation R(ang) to each (x0,x1)
;; pair, so the pullback is the transpose rotation R(-ang) applied to (dy0,dy1):
;;   dx0 =  c·dy0 + s·dy1
;;   dx1 = −s·dy0 + c·dy1
;; No trainable params (theta/positions are fixed) — only x gets a gradient.
(deftm rope-backward-dx (All [T] [dy :- (Array T) seq-len :- Long heads :- Long
                                  head-dim :- Long theta :- Double] :- (Array T)
                             (let [dx (alloc-like dy (* seq-len (* heads head-dim)))
                                   half (quot head-dim 2)
                                   ln-theta (m/log theta)]
                               (dotimes [p seq-len]
                                 (dotimes [h heads]
                                   (let [base (+ (* p (* heads head-dim)) (* h (int head-dim)))]
                                     (dotimes [i half]
                                       (let [freq (m/exp (* (/ (* -2.0 (double i)) (double head-dim)) ln-theta))
                                             ang (* (double p) freq)
                                             c (m/cos ang) s (m/sin ang)
                                             d0 (aget dy (+ base i))
                                             d1 (aget dy (+ (+ base i) half))]
                                         (aset dx (+ base i) (+ (* d0 c) (* d1 s)))
                                         (aset dx (+ (+ base i) half) (- (* d1 c) (* d0 s))))))))
                               dx)))

(tmpl/merge-into-template! 'raster.dl.attention/rope
                           {:params '[x seq-len heads head-dim theta] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [_x seq-len heads head-dim theta] _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.attention/rope-backward-dx
                                                             adjoint-sym seq-len heads head-dim theta)])
                                           [dx nil nil nil nil]]))})

;; --- Grouped / multi-query causal attention ---
;; q:[seq,n_q,hd]  k,v:[seq,n_kv,hd]  (n_kv divides n_q; n_kv<n_q = GQA, n_kv=1 = MQA).
;; Causal softmax with an EXPLICIT scale (Gemma: query_pre_attn_scalar^-0.5, which
;; need not equal 1/sqrt(head_dim)). Returns [seq, n_q*hd]. Each query head hq reads
;; kv head (hq / (n_q/n_kv)). Generic across Llama/Qwen/Gemma attention.
(deftm gqa-causal-attention (All [T]
                                 [q :- (Array T) k :- (Array T) v :- (Array T)
                                  seq-len :- Long n-q :- Long n-kv :- Long
                                  head-dim :- Long scale :- Double] :- (Array T)
                                 (let [out (alloc-like q (* seq-len (* n-q head-dim)))
                                       group (quot n-q n-kv)
                                       neg-inf (n/neg-inf-val (aget q 0))]
                                   (dotimes [hq n-q]
                                     (let [hkv (quot hq group)]
                                       (dotimes [i seq-len]
                                         (let [qb (+ (* i (* n-q head-dim)) (* hq (int head-dim)))
                                               sc (alloc-like q (inc i))
                                               _ (dotimes [j (inc i)]
                                                   (let [kb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))
                                                         dot (loop [d 0 acc 0.0]
                                                               (if (< d head-dim)
                                                                 (recur (inc d)
                                                                        (+ acc (* (aget q (+ qb d))
                                                                                  (aget k (+ kb d)))))
                                                                 acc))]
                                                     (aset sc j (* dot scale))))
                                               mx (loop [j 0 mm neg-inf]
                                                    (if (<= j i) (recur (inc j) (n/max mm (aget sc j))) mm))
                                               sum (loop [j 0 s 0.0]
                                                     (if (<= j i)
                                                       (let [e (m/exp (- (aget sc j) mx))]
                                                         (aset sc j e) (recur (inc j) (+ s e)))
                                                       s))
                                               inv (/ 1.0 sum)
                                               ob (+ (* i (* n-q head-dim)) (* hq (int head-dim)))]
                                           (dotimes [d head-dim]
                                             (aset out (+ ob d)
                                                   (loop [j 0 a 0.0]
                                                     (if (<= j i)
                                                       (let [kvb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))]
                                                         (recur (inc j)
                                                                (+ a (* (* (aget sc j) inv)
                                                                        (aget v (+ kvb d))))))
                                                       a))))))))
                                   out)))

;; --- RoPE at an absolute position offset (KV-cache decode) ---
;; Like rope, but positions run pos-offset .. pos-offset+seq-1. Decode passes
;; seq-len 1 and pos-offset = the new token's absolute position.
(deftm rope-pos (All [T] [x :- (Array T) seq-len :- Long heads :- Long
                          head-dim :- Long theta :- Double pos-offset :- Long]
                     :- (Array T)
                     (let [out (alloc-like x (* seq-len (* heads head-dim)))
                           half (quot head-dim 2)
                           ln-theta (m/log theta)]
                       (dotimes [p seq-len]
                         (dotimes [h heads]
                           (let [base (+ (* p (* heads head-dim)) (* h (int head-dim)))
                                 pos (+ pos-offset p)]
                             (dotimes [i half]
                               (let [freq (m/exp (* (/ (* -2.0 (double i)) (double head-dim)) ln-theta))
                                     ang (* (double pos) freq)
                                     c (m/cos ang) s (m/sin ang)
                                     x0 (aget x (+ base i))
                                     x1 (aget x (+ (+ base i) half))]
                                 (aset out (+ base i) (- (* x0 c) (* x1 s)))
                                 (aset out (+ (+ base i) half) (+ (* x1 c) (* x0 s))))))))
                       out)))

;; --- Single-query attention over a KV cache (decode step) ---
;; q:[1, n_q, hd], k/v:[>=cache_len, n_kv, hd] (the cache; only first cache_len
;; positions are read). The single query attends ALL cache_len keys (all causal).
;; Returns [n_q*hd]. MQA/GQA via n_kv<n_q.
(deftm gqa-decode-attention (All [T]
                                 [q :- (Array T) k :- (Array T) v :- (Array T)
                                  cache-len :- Long n-q :- Long n-kv :- Long
                                  head-dim :- Long scale :- Double] :- (Array T)
                                 (let [out (alloc-like q (* n-q head-dim))
                                       group (quot n-q n-kv)
                                       neg-inf (n/neg-inf-val (aget q 0))]
                                   (dotimes [hq n-q]
                                     (let [hkv (quot hq group)
                                           qb (* hq (int head-dim))
                                           sc (alloc-like q cache-len)
                                           _ (dotimes [j cache-len]
                                               (let [kb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))
                                                     dot (loop [d 0 acc 0.0]
                                                           (if (< d head-dim)
                                                             (recur (inc d)
                                                                    (+ acc (* (aget q (+ qb d))
                                                                              (aget k (+ kb d)))))
                                                             acc))]
                                                 (aset sc j (* dot scale))))
                                           mx (loop [j 0 mm neg-inf]
                                                (if (< j cache-len) (recur (inc j) (n/max mm (aget sc j))) mm))
                                           sum (loop [j 0 s 0.0]
                                                 (if (< j cache-len)
                                                   (let [e (m/exp (- (aget sc j) mx))]
                                                     (aset sc j e) (recur (inc j) (+ s e)))
                                                   s))
                                           inv (/ 1.0 sum)
                                           ob (* hq (int head-dim))]
                                       (dotimes [d head-dim]
                                         (aset out (+ ob d)
                                               (loop [j 0 a 0.0]
                                                 (if (< j cache-len)
                                                   (let [kvb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))]
                                                     (recur (inc j)
                                                            (+ a (* (* (aget sc j) inv)
                                                                    (aget v (+ kvb d))))))
                                                   a))))))
                                   out)))

(deftm gqa-decode-attention-heads! (All [T]
                                        [q :- (Array T) k :- (Array T) v :- (Array T) out :- (Array T)
                                         cache-len :- Long kv-start :- Long h0 :- Long h-cnt :- Long
                                         group :- Long n-kv :- Long head-dim :- Long scale :- Double] :- Long
                                        ;; Like gqa-decode-attention but computes only heads [h0, h0+h-cnt)
                                        ;; into a CALLER-provided `out` (so the decode loop can split heads
                                        ;; across the pool), and attends only KV positions [kv-start, cache-len)
                                        ;; — kv-start>0 implements sliding-window attention. Returns 0.
                                        (let [neg-inf (n/neg-inf-val (aget q 0))
                                              j0 (long kv-start)]
                                          (dotimes [hi h-cnt]
                                            (let [hq (clojure.core/+ (long h0) hi)
                                                  hkv (quot hq group)
                                                  qb (* hq (int head-dim))
                                                  sc (alloc-like q cache-len)
                                                  _ (loop [j j0]
                                                      (when (< j cache-len)
                                                        (let [kb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))
                                                              dot (loop [d 0 acc 0.0]
                                                                    (if (< d head-dim)
                                                                      (recur (inc d)
                                                                             (+ acc (* (aget q (+ qb d))
                                                                                       (aget k (+ kb d)))))
                                                                      acc))]
                                                          (aset sc j (* dot scale)))
                                                        (recur (inc j))))
                                                  mx (loop [j j0 mm neg-inf]
                                                       (if (< j cache-len) (recur (inc j) (n/max mm (aget sc j))) mm))
                                                  sum (loop [j j0 s 0.0]
                                                        (if (< j cache-len)
                                                          (let [e (m/exp (- (aget sc j) mx))]
                                                            (aset sc j e) (recur (inc j) (+ s e)))
                                                          s))
                                                  inv (/ 1.0 sum)
                                                  ob (* hq (int head-dim))]
                                              (dotimes [d head-dim]
                                                (aset out (+ ob d)
                                                      (loop [j j0 a 0.0]
                                                        (if (< j cache-len)
                                                          (let [kvb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))]
                                                            (recur (inc j)
                                                                   (+ a (* (* (aget sc j) inv)
                                                                           (aget v (+ kvb d))))))
                                                          a))))))
                                          0)))

;; Decode RoPE (single token), par/map-void! over heads — the same NeoX/HF rotate-half as
;; rope-pos with seq-len=1 (each head rotates its head-dim/2 pairs at absolute position
;; pos-offset). Lowers to OpenCL and SIMD-vectorizes on CPU. Index math clojure.core; trig
;; via raster.math. out is caller-provided (in-place-safe: reads x, writes out).
(deftm rope-pos-gpu! (All [T] [x :- (Array T) out :- (Array T)
                               heads :- Long head-dim :- Long
                               theta :- Double pos-offset :- Long] :- Void
                          (raster.par/map-void! h heads
                                                (let [hdim2 (quot head-dim 2)
                                                      base (clojure.core/* h head-dim)
                                                      ln-theta (m/log theta)
                                                      pos (double pos-offset)]
                                                  (loop [i 0]
                                                    (if (< i hdim2)
                                                      (let [freq (m/exp (* (/ (* -2.0 (double i)) (double head-dim)) ln-theta))
                                                            ang (* pos freq)
                                                            c (m/cos ang) s (m/sin ang)
                                                            x0 (aget x (clojure.core/+ base i))
                                                            x1 (aget x (clojure.core/+ (clojure.core/+ base i) hdim2))]
                                                        (aset out (clojure.core/+ base i) (- (* x0 c) (* x1 s)))
                                                        (aset out (clojure.core/+ (clojure.core/+ base i) hdim2) (+ (* x1 c) (* x0 s)))
                                                        (recur (inc i)))
                                                      nil))))))

;; KV-cache append (decode): write the current token's K (or V) slab of length kvrow = n_kv*head_dim
;; into the cache at absolute position `pos` (offset pos*kvrow). par/map-void! over kvrow — the
;; on-device equivalent of the CPU System/arraycopy append. Index math clojure.core (integer).
(deftm kv-append! (All [T] [src :- (Array T) cache :- (Array T) kvrow :- Long pos :- Long] :- Void
                       (raster.par/map-void! i kvrow
                                             (aset cache (clojure.core/+ (clojure.core/* pos kvrow) i)
                                                   (aget src i)))))

;; GPU/vectorizable decode attention: the SAME per-head computation as
;; gqa-decode-attention-heads!, but the head loop is a par/map-void! (one work-item per query
;; head) so it lowers to OpenCL and SIMD-vectorizes on CPU. Per-head scratch is caller-provided
;; (sc, size n-q*cache-len — GPU work-items can't allocate); index math is clojure.core
;; (integer subscripts), float compute is raster.numeric/raster.math. Attends all cache positions.
(deftm gqa-decode-attention-gpu! (All [T]
                                      [q :- (Array T) k :- (Array T) v :- (Array T)
                                       out :- (Array T) sc :- (Array T)
                                       cache-len :- Long n-q :- Long group :- Long
                                       n-kv :- Long head-dim :- Long scale :- Double] :- Void
                                      (raster.par/map-void! hq n-q
                                                            (let [hkv (quot hq group)
                                                                  qb (clojure.core/* hq head-dim)
                                                                  scb (clojure.core/* hq cache-len)
                                                                  kvstride (clojure.core/* n-kv head-dim)
                                                                  hkvb (clojure.core/* hkv head-dim)
                                              ;; float sentinel below any real score (OpenCL has
                                              ;; no NEGATIVE_INFINITY const; keeps the max in float
                                              ;; so n/max → fmax(float,float) isn't ambiguous).
                                                                  neg-inf -1.0e38
                                                                  _ (loop [j 0]
                                                                      (if (< j cache-len)
                                                                        (let [kb (clojure.core/+ (clojure.core/* j kvstride) hkvb)
                                                                              dot (loop [d 0 acc 0.0]
                                                                                    (if (< d head-dim)
                                                                                      (recur (inc d)
                                                                                             (+ acc (* (aget q (clojure.core/+ qb d))
                                                                                                       (aget k (clojure.core/+ kb d)))))
                                                                                      acc))]
                                                                          (aset sc (clojure.core/+ scb j) (* dot scale))
                                                                          (recur (inc j)))
                                                                        nil))
                                                                  mx (loop [j 0 mm neg-inf]
                                                                       (if (< j cache-len)
                                                                         (recur (inc j) (n/max mm (aget sc (clojure.core/+ scb j)))) mm))
                                                                  sum (loop [j 0 s 0.0]
                                                                        (if (< j cache-len)
                                                                          (let [e (m/exp (- (aget sc (clojure.core/+ scb j)) mx))]
                                                                            (aset sc (clojure.core/+ scb j) e)
                                                                            (recur (inc j) (+ s e)))
                                                                          s))
                                                                  inv (/ 1.0 sum)]
                                                              (loop [d 0]
                                                                (if (< d head-dim)
                                                                  (do (aset out (clojure.core/+ qb d)
                                                                            (loop [j 0 a 0.0]
                                                                              (if (< j cache-len)
                                                                                (let [kvb (clojure.core/+ (clojure.core/* j kvstride) hkvb)]
                                                                                  (recur (inc j)
                                                                                         (+ a (* (* (aget sc (clojure.core/+ scb j)) inv)
                                                                                                 (aget v (clojure.core/+ kvb d))))))
                                                                                a)))
                                                                      (recur (inc d)))
                                                                  nil))))))

;; ── Device-side-pos decode variants (#32): pos / cache_len come from 1-element DEVICE
;; buffers read INSIDE the par body (so each is a kernel array param, not a host scalar baked at
;; prepare!). The resident graph then binds ONCE and per token the host just writes posbuf/clenbuf
;; + replays — no re-prepare / re-record. Reading the buffer inside the kernel (vs passing
;; (aget posbuf 0) as a scalar arg) avoids the CSE-hoist that would otherwise pull the read out to
;; a host-evaluated scalar-let.
;; Flattened over heads×(head-dim/2) so the grid is heads*hdim2 work-items (one rotation each)
;; instead of `heads` threads each looping hdim2 serially — the n=4 serial version wasted ~98% of
;; the GPU (low occupancy is the dominant decode cost, NOT dispatch ~2µs nor kernel count).
(deftm rope-pos-buf! (All [T] [x :- (Array T) out :- (Array T)
                               heads :- Long head-dim :- Long
                               theta :- Double posbuf :- (Array long)] :- Void
                          (raster.par/map-void! idx (clojure.core/* heads (quot head-dim 2))
                                                (let [hdim2 (quot head-dim 2)
                                                      h (quot idx hdim2)
                                                      i (rem idx hdim2)
                                                      base (clojure.core/* h head-dim)
                                                      ln-theta (m/log theta)
                                                      pos (double (aget posbuf 0))
                                                      freq (m/exp (* (/ (* -2.0 (double i)) (double head-dim)) ln-theta))
                                                      ang (* pos freq)
                                                      c (m/cos ang) s (m/sin ang)
                                                      x0 (aget x (clojure.core/+ base i))
                                                      x1 (aget x (clojure.core/+ (clojure.core/+ base i) hdim2))]
                                                  (aset out (clojure.core/+ base i) (- (* x0 c) (* x1 s)))
                                                  (aset out (clojure.core/+ (clojure.core/+ base i) hdim2) (+ (* x1 c) (* x0 s)))))))

(deftm kv-append-buf! (All [T] [src :- (Array T) cache :- (Array T) kvrow :- Long posbuf :- (Array long)] :- Void
                           (raster.par/map-void! i kvrow
                                                 (aset cache (clojure.core/+ (clojure.core/* (aget posbuf 0) kvrow) i)
                                                       (aget src i)))))

;; 3-phase parallel decode attention. The old single-map version ran ONE work-item per Q head
;; (n-q=4) with three serial loops over cache-len×head-dim — measured 1.6ms/layer on Arc iGPU,
;; 66% of the whole decode step (occupancy, not dispatch: 4 threads on hundreds of lanes).
;; Grid sizes are BAKED at graph-record time while cache-len grows per token, so the score and
;; weighted-sum grids over-provision to `maxpos` work-items and guard on clenbuf[0] (idle items
;; exit immediately). sc is strided by maxpos (size n-q*maxpos) — the old cache-len stride also
;; overflowed a 32-float sc buffer at clen>8 (page padding hid it). Math and rounding order are
;; identical to the serial version (probs stored as e*inv, summed ascending).
(deftm gqa-decode-attention-buf! (All [T]
                                      [q :- (Array T) k :- (Array T) v :- (Array T)
                                       out :- (Array T) sc :- (Array T)
                                       clenbuf :- (Array long) n-q :- Long group :- Long
                                       n-kv :- Long head-dim :- Long maxpos :- Long scale :- T] :- Void
                                      (do
  ;; phase 1: sc[hq,j] = scale * dot(q[hq,:], k[j,hkv,:]) — one work-item per (hq, j)
                                        (raster.par/map-void! ij (clojure.core/* n-q maxpos)
                                                              (let [cache-len (aget clenbuf 0)
                                                                    hq (quot ij maxpos)
                                                                    j (rem ij maxpos)]
                                                                (when (< j cache-len)
                                                                  (let [hkv (quot hq group)
                                                                        qb (clojure.core/* hq head-dim)
                                                                        kb (clojure.core/+ (clojure.core/* j (clojure.core/* n-kv head-dim))
                                                                                           (clojure.core/* hkv head-dim))
                                                                        dot (loop [d 0 acc 0.0]
                                                                              (if (< d head-dim)
                                                                                (recur (inc d)
                                                                                       (+ acc (* (aget q (clojure.core/+ qb d))
                                                                                                 (aget k (clojure.core/+ kb d)))))
                                                                                acc))]
                                                                    ;; scale is :- T (element-typed), so dot*scale is T×T and devirtualizes
                                                                    ;; for every T — no hard-coded precision (a Double scale here would not
                                                                    ;; lower to GPU C: float×double has no monomorphic overload).
                                                                    (aset sc (clojure.core/+ (clojure.core/* hq maxpos) j) (* dot scale))))))
  ;; phase 2: softmax per head (serial over cache-len — tiny); sc ← e * (1/sum)
                                        (raster.par/map-void! hq n-q
                                                              (let [cache-len (aget clenbuf 0)
                                                                    scb (clojure.core/* hq maxpos)
                                                                    neg-inf -1.0e38
                                                                    mx (loop [j 0 mm neg-inf]
                                                                         (if (< j cache-len)
                                                                           (recur (inc j) (n/max mm (aget sc (clojure.core/+ scb j)))) mm))
                                                                    sum (loop [j 0 s 0.0]
                                                                          (if (< j cache-len)
                                                                            (let [e (m/exp (- (aget sc (clojure.core/+ scb j)) mx))]
                                                                              (aset sc (clojure.core/+ scb j) e)
                                                                              (recur (inc j) (+ s e)))
                                                                            s))
                                                                    inv (/ 1.0 sum)]
                                                                (loop [j 0]
                                                                  (if (< j cache-len)
                                                                    (do (aset sc (clojure.core/+ scb j) (* (aget sc (clojure.core/+ scb j)) inv))
                                                                        (recur (inc j)))
                                                                    nil))))
  ;; phase 3: out[hq,d] = Σ_j sc[hq,j] * v[j,hkv,d] — one work-item per (hq, d)
                                        (raster.par/map-void! hd (clojure.core/* n-q head-dim)
                                                              (let [cache-len (aget clenbuf 0)
                                                                    hq (quot hd head-dim)
                                                                    d (rem hd head-dim)
                                                                    scb (clojure.core/* hq maxpos)
                                                                    kvstride (clojure.core/* n-kv head-dim)
                                                                    hkvb (clojure.core/+ (clojure.core/* (quot hq group) head-dim) d)]
                                                                (aset out (clojure.core/+ (clojure.core/* hq head-dim) d)
                                                                      (loop [j 0 a 0.0]
                                                                        (if (< j cache-len)
                                                                          (recur (inc j)
                                                                                 (+ a (* (aget sc (clojure.core/+ scb j))
                                                                                         (aget v (clojure.core/+ (clojure.core/* j kvstride) hkvb)))))
                                                                          a))))))))

(deftm causal-scaled-dot-product-attn (All [T]
                                           [Q :- (Array T) K :- (Array T) V :- (Array T)
                                            seq-len :- Long dk :- Long dv :- Long]
                                           :- (Array T)
                                           (let [scale (/ 1.0 (n/sqrt (double dk)))
        ;; scores = Q @ K^T / sqrt(dk) -> [seq_len, seq_len]
                                                 scores (alloc-like Q (* seq-len seq-len))
                                                 _ (blas/dgemm-nt! Q K scores seq-len dk seq-len
                                                                   (n/oftype Q 1.0) (n/oftype Q 0.0))
        ;; Scale + causal mask: set scores[i,j] = -inf for j > i
                                                 ;; T-typed mask sentinel via oftype (NO aget) — a
                                                 ;; device-buffer read (neg-inf-val (aget scores 0))
                                                 ;; defeats resident-GPU extraction. -1e38 exp()s to 0
                                                 ;; like -inf; mirrors the decode kernels' literal.
                                                 neg-inf (n/oftype scores -1.0e38)
                                                 _ (dotimes [i seq-len]
                                                     (let [offset (* i (int seq-len))]
                                                       (dotimes [j seq-len]
                                                         (aset scores (+ offset j)
                                                               (if (> j i)
                                                                 neg-inf
                                                                 (* (aget scores (+ offset j)) scale))))))
        ;; softmax over seq_len dimension for each query
                                                 _ (dotimes [i seq-len]
                                                     (let [offset (* i (int seq-len))
                                                           max-s (loop [j 0 m neg-inf]
                                                                   (if (< j seq-len)
                                                                     (recur (inc j) (n/max m (aget scores (+ offset j))))
                                                                     m))
                                                           sum-exp (loop [j 0 s 0.0]
                                                                     (if (< j seq-len)
                                                                       (let [e (m/exp (- (aget scores (+ offset j)) max-s))]
                                                                         (aset scores (+ offset j) e)
                                                                         (recur (inc j) (+ s e)))
                                                                       s))
                                                           inv-sum (/ 1.0 sum-exp)]
                                                       (dotimes [j seq-len]
                                                         (aset scores (+ offset j)
                                                               (* (aget scores (+ offset j)) inv-sum)))))
        ;; output = weights @ V -> [seq_len, dv]
                                                 out (nn/matmul scores V seq-len seq-len dv)]
                                             out)))

;; Backward for causal-scaled-dot-product-attn. Returns Object[3] = [dQ dK dV].
;; Same algorithm as the pullback closure below (forward recomputation + softmax
;; backprop + d_V/dQ/dK gemms), but as a regular deftm so :grads-fn can splice
;; a single call into the AD body's flat let* (compile-aot needs flat IR).
(deftm causal-scaled-dot-product-attn-backward
  (All [T]
    [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
     seq-len :- Long dk :- Long dv :- Long]
    :- (Array Object)
    ;; scale is :- T (element-typed) so dot*scale and scale*(weights·…) devirtualize
    ;; for every T — a Double scale would be T×Double (no monomorphic overload → GPU
    ;; garbage). one/z0 likewise element-typed for the BLAS alpha/beta and masked grad.
    (let [scale (n/oftype Q (/ 1.0 (n/sqrt (double dk))))
          one   (n/oftype Q 1.0)
          z0    (n/oftype Q 0.0)
          neg-inf (n/oftype Q -1.0e38)   ; T-typed sentinel, no device-buffer read (see forward)
          scores  (alloc-like Q (* seq-len seq-len))
          weights (alloc-like Q (* seq-len seq-len))]
      ;; Forward recomputation with causal mask (masked entries = neg-inf so
      ;; exp(neg-inf - max) = 0; no explicit branch — mirrors the forward softmax).
      (dotimes [i seq-len]
        (dotimes [j seq-len]
          (let [dot (loop [d 0 acc 0.0]
                      (if (< d dk)
                        (recur (inc d)
                               (+ acc (* (aget Q (+ (* i (int dk)) d))
                                         (aget K (+ (* j (int dk)) d)))))
                        acc))]
            (aset scores (+ (* i (int seq-len)) j)
                  (if (> j i) neg-inf (* dot scale))))))
      (dotimes [i seq-len]
        (let [offset (* i (int seq-len))
              max-s (loop [j 0 mm neg-inf]
                      (if (< j seq-len)
                        (recur (inc j) (n/max mm (aget scores (+ offset j))))
                        mm))
              sum-exp (loop [j 0 s 0.0]
                        (if (< j seq-len)
                          (let [e (m/exp (- (aget scores (+ offset j)) max-s))]
                            (aset weights (+ offset j) e)
                            (recur (inc j) (+ s e)))
                          s))
              inv-sum (/ one sum-exp)]
          (dotimes [j seq-len]
            (aset weights (+ offset j)
                  (* (aget weights (+ offset j)) inv-sum)))))
      (let [d-weights (alloc-like Q (* seq-len seq-len))
            _ (blas/dgemm-nt! d-out V d-weights seq-len dv seq-len one z0)
            dV (alloc-like Q (* seq-len dv))
            _ (blas/dgemm-tn! weights d-out dV seq-len seq-len dv one z0)
            d-scores (alloc-like Q (* seq-len seq-len))]
        (dotimes [i seq-len]
          (let [offset (* i (int seq-len))
                dot-wdw (loop [j 0 acc 0.0]
                          (if (< j seq-len)
                            (recur (inc j)
                                   (+ acc (* (aget weights (+ offset j))
                                             (aget d-weights (+ offset j)))))
                            acc))]
            (dotimes [j seq-len]
              (aset d-scores (+ offset j)
                    (if (> j i)
                      z0
                      (* scale
                         (aget weights (+ offset j))
                         (- (aget d-weights (+ offset j)) dot-wdw)))))))
        (let [dQ (nn/matmul d-scores K seq-len seq-len dk)
              dK (alloc-like Q (* seq-len dk))
              _ (blas/dgemm-tn! d-scores Q dK seq-len seq-len dk one z0)
              out (object-array 3)]
          (clojure.core/aset out 0 dQ)
          (clojure.core/aset out 1 dK)
          (clojure.core/aset out 2 dV)
          out)))))

;; grads-fn (compile-aot flat codegen): calls the backward deftm above and
;; extracts the three gradients via aget — a single flat let-binding shape
;; for PE/CSE/DCE.

(tmpl/merge-into-template! 'raster.dl.attention/causal-scaled-dot-product-attn
                           {:params '[Q K V seq-len dk dv]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [Q K V seq-len dk dv] _result-sym adjoint-sym gensym-fn]
                              (let [bundle (gensym-fn "cspa_grads")
                                    dQ (gensym-fn "dQ")
                                    dK (gensym-fn "dK")
                                    dV (gensym-fn "dV")]
                                [(update ctx :bindings into
                                         [bundle (list 'raster.dl.attention/causal-scaled-dot-product-attn-backward
                                                       adjoint-sym Q K V seq-len dk dv)
                                          dQ (list 'clojure.core/aget bundle 0)
                                          dK (list 'clojure.core/aget bundle 1)
                                          dV (list 'clojure.core/aget bundle 2)])
                                 [dQ dK dV nil nil nil]]))})

;; Forward tangent (JVP, §13 A3) of causal SDPA — the standard form, single
;; output array (mirrors the backward's recompute-then-gemm structure):
;;   W  = causal-softmax(s·Q·Kᵀ)              (recomputed)
;;   dZ = s·(dQ·Kᵀ + Q·dKᵀ)
;;   dW = W ⊙ (dZ − rowsum(W⊙dZ))             (softmax JVP per query row;
;;                                             masked j>i have W=0 ⇒ dW=0)
;;   dO = dW·V + W·dV
(deftm causal-scaled-dot-product-attn-jvp
  (All [T]
    [dQ :- (Array T) dK :- (Array T) dV :- (Array T)
     Q :- (Array T) K :- (Array T) V :- (Array T)
     seq-len :- Long dk :- Long dv :- Long]
    :- (Array T)
    ;; scale/one/z0 element-typed (see the backward) — a Double scalar × T has no
    ;; monomorphic overload and would emit GPU garbage.
    (let [scale (n/oftype Q (/ 1.0 (n/sqrt (double dk))))
          one   (n/oftype Q 1.0)
          z0    (n/oftype Q 0.0)
          neg-inf (n/oftype Q -1.0e38)   ; T-typed sentinel, no device-buffer read (see forward)
          ;; Recompute causal softmax weights (same as the backward)
          weights (alloc-like Q (* seq-len seq-len))
          _ (blas/dgemm-nt! Q K weights seq-len dk seq-len scale z0)
          _ (dotimes [i seq-len]
              (let [offset (* i (int seq-len))]
                (dotimes [j seq-len]
                  (when (> j i)
                    (aset weights (+ offset j) neg-inf)))))
          _ (dotimes [i seq-len]
              (let [offset (* i (int seq-len))
                    max-s (loop [j 0 mm neg-inf]
                            (if (< j seq-len)
                              (recur (inc j) (n/max mm (aget weights (+ offset j))))
                              mm))
                    sum-exp (loop [j 0 s 0.0]
                              (if (< j seq-len)
                                (let [e (m/exp (- (aget weights (+ offset j)) max-s))]
                                  (aset weights (+ offset j) e)
                                  (recur (inc j) (+ s e)))
                                s))
                    inv-sum (/ one sum-exp)]
                (dotimes [j seq-len]
                  (aset weights (+ offset j) (* (aget weights (+ offset j)) inv-sum)))))
          ;; dZ = scale·(dQ·Kᵀ + Q·dKᵀ)
          dZ (alloc-like Q (* seq-len seq-len))
          _ (blas/dgemm-nt! dQ K dZ seq-len dk seq-len scale z0)
          _ (blas/dgemm-nt! Q dK dZ seq-len dk seq-len scale one)
          ;; softmax JVP per row: dW = W⊙(dZ − ⟨W,dZ⟩_row)
          dW (alloc-like Q (* seq-len seq-len))
          _ (dotimes [i seq-len]
              (let [offset (* i (int seq-len))
                    wdz (loop [j 0 s 0.0]
                          (if (< j seq-len)
                            (recur (inc j) (+ s (* (aget weights (+ offset j))
                                                   (aget dZ (+ offset j)))))
                            s))]
                (dotimes [j seq-len]
                  (aset dW (+ offset j)
                        (* (aget weights (+ offset j))
                           (- (aget dZ (+ offset j)) wdz))))))
          ;; dO = dW·V + W·dV
          dO (alloc-like Q (* seq-len dv))
          _ (blas/dgemm! dW V dO seq-len seq-len dv one z0)
          _ (blas/dgemm! weights dV dO seq-len seq-len dv one one)]
      dO)))

;; causal SDPA :jvp-fn — one kernel call; absent tangents get typed zeros
;; (the pushforward is jointly linear in (dQ,dK,dV), so zeros are exact).
(tmpl/merge-into-template!
 'raster.dl.attention/causal-scaled-dot-product-attn
 {:jvp-fn
  (fn [ctx [Q K V seq-len dk dv] tangent-args _result-sym gensym-fn]
    (let [dQ (nth tangent-args 0 nil)
          dK (nth tangent-args 1 nil)
          dV (nth tangent-args 2 nil)]
      (when-not (or dQ dK dV)
        (throw (ex-info "causal-scaled-dot-product-attn jvp: no active tangent reached the call"
                        {:op 'raster.dl.attention/causal-scaled-dot-product-attn})))
      (tmpl/bind-jvp-term ctx gensym-fn "jcspa"
                          (list 'raster.dl.attention/causal-scaled-dot-product-attn-jvp
                                (or dQ (tmpl/jvp-zero-like Q))
                                (or dK (tmpl/jvp-zero-like K))
                                (or dV (tmpl/jvp-zero-like V))
                                Q K V seq-len dk dv))))})

;; ================================================================
;; Causal single-head attention (AD-friendly: no head-split shuffles).
;; Composes only deftms with registered AD pullbacks so value+grad inlines
;; cleanly. For training-from-scratch demonstrations of GPT-2-style models;
;; the multi-head version below uses dotimes/aset shuffles that don't
;; differentiate without an explicit pullback on the whole block.
;; ================================================================

(deftm causal-single-head-attention (All [T]
                                         [x  :- (Array T)
                                          Wq :- (Array T) bq :- (Array T)
                                          Wk :- (Array T) bk :- (Array T)
                                          Wv :- (Array T) bv :- (Array T)
                                          Wo :- (Array T) bo :- (Array T)
                                          seq-len :- Long d-model :- Long]
                                         :- (Array T)
                                         (let [Q   (nn/linear x Wq bq seq-len d-model d-model)
                                               K   (nn/linear x Wk bk seq-len d-model d-model)
                                               V   (nn/linear x Wv bv seq-len d-model d-model)
                                               attn-out (causal-scaled-dot-product-attn
                                                         Q K V seq-len d-model d-model)]
                                           (nn/linear attn-out Wo bo seq-len d-model d-model))))

;; ================================================================
;; Causal multi-head attention (for autoregressive models like GPT-2)
;; x:[seq_len, d_model], W{q,k,v,o}:[d_model, d_model], b{q,k,v,o}:[d_model]
;; ================================================================

(deftm causal-multi-head-attention
  "Causal multi-head self-attention. Composed via slice-strided-2d /
  scatter-strided-2d over a loop fold so each step is a templated AD primitive
  — no nested-dotimes shuffles for AD to walk through. The output projection
  lives in the loop's result branch so AD sees the canonical
  (let* bindings (loop … result)) shape that gen-reverse-loop-with-let handles."
  [x :- (Array double) Wq :- (Array double) bq :- (Array double)
   Wk :- (Array double) bk :- (Array double)
   Wv :- (Array double) bv :- (Array double)
   Wo :- (Array double) bo :- (Array double)
   seq-len :- Long d-model :- Long n-heads :- Long]
  :- (Array double)
  (let [dk (quot d-model n-heads)
        n  (* seq-len d-model)
        Q  (nn/linear x Wq bq seq-len d-model d-model)
        K  (nn/linear x Wk bk seq-len d-model d-model)
        V  (nn/linear x Wv bv seq-len d-model d-model)]
    ;; Fold over heads: at each step, slice the head's QKV slabs out of
    ;; the full projections, run causal single-head attention, scatter
    ;; the result into a [seq_len × d_model] zeros-padded slab, and add
    ;; into the running accumulator. The result branch projects the
    ;; concatenated heads through Wo/bo.
    (loop [h 0 acc (double-array n)]
      (if (< h n-heads)
        (let [col-off (* h (int dk))
              Qh (ops/slice-strided-2d Q seq-len d-model col-off dk)
              Kh (ops/slice-strided-2d K seq-len d-model col-off dk)
              Vh (ops/slice-strided-2d V seq-len d-model col-off dk)
              head-out (causal-scaled-dot-product-attn
                        Qh Kh Vh seq-len dk dk)
              out-h (ops/scatter-strided-2d
                     head-out seq-len d-model col-off dk)]
          (recur (inc h) (ops/array-add acc out-h n)))
        (nn/linear acc Wo bo seq-len d-model d-model)))))

;; ================================================================
;; Differentiable GQA/MQA causal attention (Llama/Qwen/Gemma decoders)
;; ================================================================

;; ----------------------------------------------------------------
;; batched-causal-sdpa: run the flat GEMM-based causal SDPA over `batch`
;; contiguous [seq,hd] head slabs (Q,K,V all packed [batch,seq,hd]). Opaque to AD
;; (its own flat :grads-fn below calls the batched backward) so the head iteration
;; NEVER enters the reverse-AD tape — that is what keeps gqa-causal-mha's gradient
;; a straight-line let* (no fn* pullback closure / ArrayList tape). The runtime
;; body loops over batches calling the tested single-head causal-scaled-dot-
;; product-attn(+backward); (All [T]), all scalars element-typed inside SDPA.
;; ----------------------------------------------------------------
(deftm batched-causal-sdpa
  (All [T]
       [Q :- (Array T) K :- (Array T) V :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array T)
       (let [slab (* seq-len head-dim)
             out  (alloc-like Q (* batch slab))]
         (dotimes [b batch]
           (let [boff (* b (int slab))
                 Qb (alloc-like Q slab) Kb (alloc-like Q slab) Vb (alloc-like Q slab)]
             (dotimes [i slab]
               (aset Qb i (aget Q (+ boff i)))
               (aset Kb i (aget K (+ boff i)))
               (aset Vb i (aget V (+ boff i))))
             (let [ob (causal-scaled-dot-product-attn Qb Kb Vb seq-len head-dim head-dim)]
               (dotimes [i slab]
                 (aset out (+ boff i) (aget ob i))))))
         out)))

(deftm batched-causal-sdpa-backward
  "Backward for batched-causal-sdpa. Returns Object[3] = [dQ dK dV]. Batches own
  disjoint slabs (no cross-batch accumulation — the GQA kv-head fan-in is handled
  by sum-kv-heads OUTSIDE this op, since K/V are already broadcast to `batch=n_q`)."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array Object)
       (let [slab (* seq-len head-dim)
             dQ (alloc-like Q (* batch slab))
             dK (alloc-like Q (* batch slab))
             dV (alloc-like Q (* batch slab))]
         (dotimes [b batch]
           (let [boff (* b (int slab))
                 Qb (alloc-like Q slab) Kb (alloc-like Q slab)
                 Vb (alloc-like Q slab) dOb (alloc-like Q slab)]
             (dotimes [i slab]
               (aset Qb i (aget Q (+ boff i)))
               (aset Kb i (aget K (+ boff i)))
               (aset Vb i (aget V (+ boff i)))
               (aset dOb i (aget d-out (+ boff i))))
             (let [g   (causal-scaled-dot-product-attn-backward
                        dOb Qb Kb Vb seq-len head-dim head-dim)
                   dQb (clojure.core/aget g 0)
                   dKb (clojure.core/aget g 1)
                   dVb (clojure.core/aget g 2)]
               ;; dQb/dKb/dVb are (Array T) but extracted from an Object[] so their
               ;; compile-time tag is Object — scatter through the typed blit-slab!
               ;; helper (not a direct raster.arrays/aget) so element access
               ;; devirtualizes to the real double[]/float[] under compile-aot.
               (ops/blit-slab! dQ boff dQb slab)
               (ops/blit-slab! dK boff dKb slab)
               (ops/blit-slab! dV boff dVb slab))))
         (let [o (object-array 3)]
           (clojure.core/aset o 0 dQ)
           (clojure.core/aset o 1 dK)
           (clojure.core/aset o 2 dV)
           o))))

(tmpl/merge-into-template! 'raster.dl.attention/batched-causal-sdpa
                           {:params '[Q K V batch seq-len head-dim]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [Q K V batch seq-len head-dim]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [bundle (gensym-fn "bcspa_grads")
                                    dQ (gensym-fn "dQ")
                                    dK (gensym-fn "dK")
                                    dV (gensym-fn "dV")]
                                [(update ctx :bindings into
                                         [bundle (list 'raster.dl.attention/batched-causal-sdpa-backward
                                                       adjoint-sym Q K V batch seq-len head-dim)
                                          dQ (list 'clojure.core/aget bundle 0)
                                          dK (list 'clojure.core/aget bundle 1)
                                          dV (list 'clojure.core/aget bundle 2)])
                                 [dQ dK dV nil nil nil]]))})

(deftm gqa-causal-mha
  "Differentiable grouped/multi-query causal self-attention over PRE-PROJECTED,
  PRE-ROPED Q:[seq,n_q·hd] and K/V:[seq,n_kv·hd] (n_kv divides n_q; n_kv<n_q = GQA,
  n_kv=1 = MQA). Query head hq reads kv head hq/(n_q/n_kv). Returns [seq, n_q·hd].

  FLAT decomposition (no per-head carry loop): pack Q/K/V to per-head-contiguous
  layout, BROADCAST the n_kv kv heads up to the n_q query heads (broadcast-kv-heads),
  run a batched causal SDPA over the n_q head slabs, then unpack the disjoint
  per-head outputs back to [seq, n_q·hd]. Every step is an AD-registered primitive
  with a FLAT grads-fn (pack↔unpack, broadcast-kv-heads↔sum-kv-heads,
  batched-causal-sdpa↔its backward), so value+grad differentiates it as a
  straight-line let* — NO fn* pullback / ArrayList tape (the old raw carry loop
  emitted one, which cannot lower to GPU). The MQA/GQA dK/dV fan-in (the `group`
  query heads sharing one kv head) falls out exactly as sum-kv-heads (the
  broadcast dual). Scale is 1/sqrt(hd) (Llama/Qwen; Gemma when
  query_pre_attn_scalar = head_dim). The output projection is a separate linear-nb."
  (All [T]
    [Q :- (Array T) K :- (Array T) V :- (Array T)
     seq-len :- Long n-q :- Long n-kv :- Long head-dim :- Long]
    :- (Array T)
  (let [group (quot n-q n-kv)
        slab  (* seq-len head-dim)
        Qp (ops/pack-heads Q seq-len n-q head-dim)
        Kp (ops/pack-heads K seq-len n-kv head-dim)
        Vp (ops/pack-heads V seq-len n-kv head-dim)
        Ke (ops/broadcast-kv-heads Kp n-kv group slab)
        Ve (ops/broadcast-kv-heads Vp n-kv group slab)
        Op (batched-causal-sdpa Qp Ke Ve n-q seq-len head-dim)]
    (ops/unpack-heads Op seq-len n-q head-dim))))

;; gqa-causal-mha has NO hand-written reverse rule of its own: it is composed
;; entirely from AD-registered primitives with flat grads-fn templates
;; (pack-heads/unpack-heads, broadcast-kv-heads/sum-kv-heads, batched-causal-sdpa),
;; so value+grad inlines and differentiates it into a straight-line let* — flat,
;; GPU-lowerable IR with no closures-as-tape. The :jvp-fn (forward mode) is KEPT
;; below — its removability is a separate forward-mode audit.

;; Forward tangent (JVP, §13 A3) of GQA/MQA causal attention — mirrors
;; gqa-causal-mha-backward's head iteration: per query head hq, slice the
;; (Q,K,V) and (dQ,dK,dV) head slabs, run the single-head causal-SDPA JVP,
;; and WRITE the head tangent into hq's output slab (each hq owns its slab —
;; no accumulation on the output side; the kv-head fan-IN that forces dK/dV
;; accumulation in the backward has no forward counterpart).
(deftm gqa-causal-mha-jvp
  (All [T]
    [dQ :- (Array T) dK :- (Array T) dV :- (Array T)
     Q :- (Array T) K :- (Array T) V :- (Array T)
     seq-len :- Long n-q :- Long n-kv :- Long head-dim :- Long]
    :- (Array T)
  (let [group (quot n-q n-kv)
        qstride (* n-q head-dim)
        kvstride (* n-kv head-dim)
        dOut (alloc-like Q (* seq-len qstride))]
    (dotimes [hq n-q]
      (let [hkv (quot hq (int group))
            Qh (ops/slice-strided-2d Q seq-len qstride (* hq (int head-dim)) head-dim)
            Kh (ops/slice-strided-2d K seq-len kvstride (* hkv (int head-dim)) head-dim)
            Vh (ops/slice-strided-2d V seq-len kvstride (* hkv (int head-dim)) head-dim)
            dQh (ops/slice-strided-2d dQ seq-len qstride (* hq (int head-dim)) head-dim)
            dKh (ops/slice-strided-2d dK seq-len kvstride (* hkv (int head-dim)) head-dim)
            dVh (ops/slice-strided-2d dV seq-len kvstride (* hkv (int head-dim)) head-dim)
            dOh (causal-scaled-dot-product-attn-jvp dQh dKh dVh Qh Kh Vh
                                                    seq-len head-dim head-dim)]
        (dotimes [r seq-len]
          (let [qoff (+ (* r (int qstride)) (* hq (int head-dim)))
                hoff (* r (int head-dim))]
            (dotimes [c head-dim]
              (aset dOut (+ qoff c) (aget dOh (+ hoff c))))))))
    dOut)))

;; gqa-causal-mha :jvp-fn — one kernel call; absent tangents get typed zeros
;; (jointly linear pushforward in (dQ,dK,dV)).
(tmpl/merge-into-template!
 'raster.dl.attention/gqa-causal-mha
 {:jvp-fn
  (fn [ctx [Q K V seq-len n-q n-kv head-dim] tangent-args _result-sym gensym-fn]
    (let [dQ (nth tangent-args 0 nil)
          dK (nth tangent-args 1 nil)
          dV (nth tangent-args 2 nil)]
      (when-not (or dQ dK dV)
        (throw (ex-info "gqa-causal-mha jvp: no active tangent reached the call"
                        {:op 'raster.dl.attention/gqa-causal-mha})))
      (tmpl/bind-jvp-term ctx gensym-fn "jgqa"
                          (list 'raster.dl.attention/gqa-causal-mha-jvp
                                (or dQ (tmpl/jvp-zero-like Q))
                                (or dK (tmpl/jvp-zero-like K))
                                (or dV (tmpl/jvp-zero-like V))
                                Q K V seq-len n-q n-kv head-dim))))})

;; ================================================================
;; Multi-head self-attention (bidirectional, for BERT-style models)
;; x:[seq_len, d_model], Wq/Wk/Wv:[d_model, d_model], Wo:[d_model, d_model]
;; Separate bias parameters bq/bk/bv/bo for BERT-style models.
;; ================================================================

(deftm fast-exp
  "Portable vectorizing exp(x) for x <= 0 (softmax range): e^x = (e^(x/1024))^1024
  via a degree-6 Taylor of the small argument then 10 squarings. Uses only n/*
  and n/+ (no libm call, no SVML, no bitcast) so it lowers to a pure FMA/mul chain
  that the compiler SIMD-vectorizes on every backend. Rel error <1e-8 over
  [-88,0]. (SVML VectorOperators/EXP is not intrinsified on this JVM, so Math/exp
  stays scalar; this does not depend on it.)"
  (All [T] [x :- T] :- T
       (let [t (n/* x 9.765625E-4)                    ; x / 1024
             p (n/+ 0.008333334 (n/* t 0.0013888889)) ; 1/120 + t/720
             p (n/+ 0.041666668 (n/* t p))            ; 1/24  + t p
             p (n/+ 0.16666667 (n/* t p))             ; 1/6
             p (n/+ 0.5 (n/* t p))                    ; 1/2
             p (n/+ 1.0 (n/* t p))                    ; 1
             p (n/+ 1.0 (n/* t p))                    ; e^t (Taylor deg 6)
             p (n/* p p) p (n/* p p) p (n/* p p) p (n/* p p) p (n/* p p)
             p (n/* p p) p (n/* p p) p (n/* p p) p (n/* p p) p (n/* p p)]
         p)))

(deftm softmax-rows!
  "In-place row-wise softmax of a [rows, cols] row-major array — one fused pass
  per row (max → exp → sum → normalize). Scale is assumed already folded into the
  values (via the QK^T GEMM alpha). Parametric; float-pure."
  (All [T] [x :- (Array T) rows :- Long cols :- Long] :- (Array T)
       (let [n (* rows (int cols))
          ;; per-row max → broadcast to per-element (cheap, memory-bound scalar)
             maxes (alloc-like x n)
             _ (dotimes [r rows]
                 (let [off (* r (int cols))
                       mx (loop [j 0 m (n/neg-inf-val (aget x off))]
                            (if (< j cols) (recur (inc j) (n/max m (aget x (+ off j)))) m))]
                   (dotimes [j cols] (aset maxes (+ off j) mx))))
          ;; VECTORIZED exp = exp(x - max): a flat `broadcast` (recognized par
          ;; form → SIMD-vectorized SegMap under compile-aot) with the fast-exp
          ;; polynomial INLINED (e^(v/1024)^1024, deg-6 Taylor + 10 squarings;
          ;; only +/* → a pure FMA/mul lane chain). A raw nested dotimes here
          ;; stays scalar; a deftm call stays opaque; SVML EXP is not intrinsified.
             e (broadcast [x maxes]
                          (let [v (- x maxes) t (* v 9.765625E-4)
                                a (+ 0.008333334 (* t 0.0013888889)) b (+ 0.041666668 (* t a))
                                c (+ 0.16666667 (* t b)) d (+ 0.5 (* t c))
                                ee (+ 1.0 (* t d)) f (+ 1.0 (* t ee))
                                p1 (* f f) p2 (* p1 p1) p3 (* p2 p2) p4 (* p3 p3) p5 (* p4 p4)
                                p6 (* p5 p5) p7 (* p6 p6) p8 (* p7 p7) p9 (* p8 p8) p10 (* p9 p9)]
                            p10))
          ;; per-row sum of e, normalize back into x (cheap, memory-bound scalar)
             _ (dotimes [r rows]
                 (let [off (* r (int cols))
                       s (loop [j 0 acc 0.0]
                           (if (< j cols) (recur (inc j) (+ acc (aget e (+ off j)))) acc))
                       inv (/ 1.0 s)]
                   (dotimes [j cols]
                     (aset x (+ off j) (* (aget e (+ off j)) inv)))))]
         x)))

;; Batched multi-head attention: composed from the layout/GEMM/softmax
;; combinators (pack-heads → batched QK^T with scale folded → fused row-softmax →
;; batched scores@V → unpack-heads → output proj). Reproduces ORT's attention
;; computation (scaled-QK^T FusedMatMul + Softmax + scores@V) with NO per-head
;; scalar copies and 2 batched GEMM calls/head-group instead of ~2*n-heads tiny
;; ones. NOTE: the forward path is fully composable; training AD through
;; batched-gemm-*!/softmax-rows! is not yet registered (rrules deferred — the
;; fastembed inference path is forward-only).
(deftm multi-head-attention (All [T]
                                 [x :- (Array T) Wq :- (Array T) bq :- (Array T)
                                  Wk :- (Array T) bk :- (Array T)
                                  Wv :- (Array T) bv :- (Array T)
                                  Wo :- (Array T) bo :- (Array T)
                                  seq-len :- Long d-model :- Long n-heads :- Long]
                                 :- (Array T)
                                 (let [dk (quot d-model n-heads)
                                       scale (n/oftype x (clojure.core// 1.0 (Math/sqrt (double dk))))
        ;; Q,K,V projections: [seq_len, d_model]
                                       Q (nn/linear x Wq bq seq-len d-model d-model)
                                       K (nn/linear x Wk bk seq-len d-model d-model)
                                       V (nn/linear x Wv bv seq-len d-model d-model)
        ;; pack to head-major contiguous [n_heads, seq_len, dk]
                                       Qh (ops/pack-heads Q seq-len n-heads dk)
                                       Kh (ops/pack-heads K seq-len n-heads dk)
                                       Vh (ops/pack-heads V seq-len n-heads dk)
        ;; scores[n_heads, seq, seq] = scale * Qh @ Kh^T  (one batched GEMM)
                                       scores (alloc-like x (* n-heads seq-len seq-len))
                                       _ (blas/batched-gemm-nt! Qh Kh scores n-heads seq-len dk seq-len scale)
        ;; fused row-softmax over the key dimension
                                       _ (softmax-rows! scores (* n-heads seq-len) seq-len)
        ;; ctx[n_heads, seq, dk] = scores @ Vh  (one batched GEMM)
                                       ctxh (alloc-like x (* n-heads seq-len dk))
                                       _ (blas/batched-gemm-nn! scores Vh ctxh n-heads seq-len seq-len dk (n/oftype x 1.0))
        ;; unpack to [seq_len, d_model] and project out
                                       ctx (ops/unpack-heads ctxh seq-len n-heads dk)]
                                   (nn/linear ctx Wo bo seq-len d-model d-model))))

;; ================================================================
;; KV-cache: incremental attention for autoregressive generation
;; These are inference-only (no AD registration).
;; ================================================================

(deftm causal-attn-with-cache! (All [T]
                                    [Q-new :- (Array T)     ;; [dk] single query vector
                                     K-new :- (Array T)     ;; [dk] single key vector
                                     V-new :- (Array T)     ;; [dv] single value vector
                                     cache-K :- (Array T)   ;; [max_seq * dk] mutable KV cache
                                     cache-V :- (Array T)   ;; [max_seq * dv] mutable KV cache
                                     cache-pos :- Long      ;; current write position (0-indexed)
                                     dk :- Long dv :- Long]
                                    :- (Array T)
  ;; 1. Write K-new, V-new into cache at cache-pos
                                    (let [k-base (* cache-pos dk)
                                          v-base (* cache-pos dv)]
                                      (dotimes [d dk]
                                        (aset cache-K (+ k-base d) (aget K-new d)))
                                      (dotimes [d dv]
                                        (aset cache-V (+ v-base d) (aget V-new d)))
    ;; 2. Compute scores[j] = dot(Q-new, cache-K[j]) / sqrt(dk) for j in 0..cache-pos
                                      (let [n-valid (+ cache-pos 1)
                                            scale (/ 1.0 (n/sqrt (double dk)))
          ;; Find max score for numerical stability
                                            max-score (loop [j 0 m (n/neg-inf-val (aget Q-new 0))]
                                                        (if (< j n-valid)
                                                          (let [score (loop [d 0 acc 0.0]
                                                                        (if (< d dk)
                                                                          (recur (inc d)
                                                                                 (+ acc (* (aget Q-new d)
                                                                                           (aget cache-K (+ (* j dk) d)))))
                                                                          (* acc scale)))]
                                                            (recur (inc j) (n/max m score)))
                                                          m))
          ;; Compute softmax weights and weighted sum simultaneously
          ;; First pass: exp(score - max) and accumulate sum
                                            weights (alloc-like Q-new n-valid)
                                            sum-exp (loop [j 0 s 0.0]
                                                      (if (< j n-valid)
                                                        (let [score (loop [d 0 acc 0.0]
                                                                      (if (< d dk)
                                                                        (recur (inc d)
                                                                               (+ acc (* (aget Q-new d)
                                                                                         (aget cache-K (+ (* j dk) d)))))
                                                                        (* acc scale)))
                                                              w (m/exp (- score max-score))]
                                                          (aset weights j w)
                                                          (recur (inc j) (+ s w)))
                                                        s))
                                            inv-sum (/ 1.0 sum-exp)
          ;; Output = weighted sum of cache-V rows
                                            out (alloc-like V-new dv)]
                                        (dotimes [j n-valid]
                                          (let [w (* (aget weights j) inv-sum)]
                                            (dotimes [d dv]
                                              (aset out d (+ (aget out d) (* w (aget cache-V (+ (* j dv) d))))))))
                                        out))))

(deftm causal-multi-head-attention-cached! (All [T]
                                                [x :- (Array T)         ;; [d_model] single token embedding
                                                 Wq :- (Array T) bq :- (Array T)
                                                 Wk :- (Array T) bk :- (Array T)
                                                 Wv :- (Array T) bv :- (Array T)
                                                 Wo :- (Array T) bo :- (Array T)
                                                 cache-K :- (Array T)   ;; [n_heads * max_seq * dk] flat
                                                 cache-V :- (Array T)   ;; [n_heads * max_seq * dk] flat
                                                 cache-pos :- Long
                                                 d-model :- Long n-heads :- Long max-seq :- Long]
                                                :- (Array T)
                                                (let [dk (quot d-model n-heads)
        ;; Project single token: x:[d_model] -> Q,K,V:[d_model]
                                                      Q (nn/linear x Wq bq 1 d-model d-model)
                                                      K (nn/linear x Wk bk 1 d-model d-model)
                                                      V (nn/linear x Wv bv 1 d-model d-model)
        ;; Process each head with its own cache slice
                                                      concat-out (alloc-like x d-model)
                                                      head-cache-size (* max-seq dk)]
                                                  (dotimes [h n-heads]
                                                    (let [h-offset (* h (int dk))
                                                          cache-k-offset (* h head-cache-size)
                                                          cache-v-offset (* h head-cache-size)
            ;; Extract head h from Q, K, V
                                                          Qh (alloc-like Q dk)
                                                          Kh (alloc-like K dk)
                                                          Vh (alloc-like V dk)]
                                                      (dotimes [d dk]
                                                        (aset Qh d (aget Q (+ h-offset d)))
                                                        (aset Kh d (aget K (+ h-offset d)))
                                                        (aset Vh d (aget V (+ h-offset d))))
        ;; Slice into this head's cache region
        ;; We need to pass the full cache arrays + compute offsets inside
        ;; Since deftm can't do array slicing, use a per-head cache view
        ;; by offsetting manually. For now, allocate per-head caches and
        ;; copy in/out. A more optimized version would use offset arithmetic.
                                                      (let [head-ck (alloc-like cache-K (* max-seq dk))
                                                            head-cv (alloc-like cache-V (* max-seq dk))]
          ;; Copy existing cache for this head
                                                        (dotimes [i (* cache-pos dk)]
                                                          (aset head-ck i (aget cache-K (+ cache-k-offset i))))
                                                        (dotimes [i (* cache-pos dk)]
                                                          (aset head-cv i (aget cache-V (+ cache-v-offset i))))
          ;; Run cached attention
                                                        (let [head-out (causal-attn-with-cache! Qh Kh Vh head-ck head-cv cache-pos dk dk)]
            ;; Copy updated cache back
                                                          (let [new-k-base (* cache-pos dk)]
                                                            (dotimes [d dk]
                                                              (aset cache-K (+ cache-k-offset new-k-base d)
                                                                    (aget head-ck (+ new-k-base d)))
                                                              (aset cache-V (+ cache-v-offset new-k-base d)
                                                                    (aget head-cv (+ new-k-base d)))))
            ;; Copy head output to concatenated output
                                                          (dotimes [d dk]
                                                            (aset concat-out (+ h-offset d) (aget head-out d)))))))
    ;; Output projection: concat-out:[d_model] -> out:[d_model]
                                                  (nn/linear concat-out Wo bo 1 d-model d-model))))

(defn prefill-cache!
  "Fill KV caches from a full sequence using non-cached forward pass.
  Returns the output of the last layer for continued generation.

  layers: vector of {:Wq :bq :Wk :bk :Wv :bv :Wo :bo} per layer
  caches: vector of {:cache-K array, :cache-V array} per layer
  x: [seq_len, d_model] input embeddings
  seq-len, d-model, n-heads, max-seq: dimension parameters"
  [layers caches x seq-len d-model n-heads max-seq]
  (let [dk (quot d-model n-heads)]
    ;; For each layer, run non-cached attention and extract K,V into caches
    (doseq [layer-idx (range (count layers))]
      (let [{:keys [Wk bk Wv bv]} (nth layers layer-idx)
            {:keys [cache-K cache-V]} (nth caches layer-idx)
            ;; Compute K,V projections for full sequence
            K-full (nn/linear x Wk bk seq-len d-model d-model)
            V-full (nn/linear x Wv bv seq-len d-model d-model)
            head-cache-size (* max-seq dk)]
        ;; Copy K,V into per-head cache slots
        (dotimes [h n-heads]
          (let [h-offset (* h (int dk))
                cache-k-offset (* h head-cache-size)
                cache-v-offset (* h head-cache-size)]
            (dotimes [s seq-len]
              (dotimes [d dk]
                (let [src-idx (+ (* s (int d-model)) h-offset d)
                      dst-idx (+ cache-k-offset (* s dk) d)]
                  (clojure.core/aset cache-K dst-idx (clojure.core/aget K-full src-idx))
                  (clojure.core/aset cache-V dst-idx (clojure.core/aget V-full src-idx)))))))))))

;; ================================================================
;; Graph attention (scatter-based message passing)
;; h:[n_nodes, d_model], W:[d_model, d_model]
;; src_edges, dst_edges: [n_edges] (long[])
;; ================================================================

(deftm graph-attention
  [h :- (Array double)
   Wq :- (Array double) Wk :- (Array double)
   Wv :- (Array double) Wo :- (Array double)
   src-edges :- (Array long) dst-edges :- (Array long)
   n-nodes :- Long n-edges :- Long d-model :- Long]
  :- (Array double)
  (let [;; Project Q, K, V from node features
        Q (nn/matmul h Wq n-nodes d-model d-model)
        K (nn/matmul h Wk n-nodes d-model d-model)
        V (nn/matmul h Wv n-nodes d-model d-model)
        ;; Compute attention scores at edges
        ;; score[e] = exp(sum_d Q[dst[e],d] * K[src[e],d] / sqrt(dk))
        scale (/ 1.0 (n/sqrt (double d-model)))
        scores (double-array n-edges)
        _ (dotimes [e n-edges]
            (let [src (aget src-edges e)
                  dst (aget dst-edges e)
                  dot (loop [d 0 acc 0.0]
                        (if (< d d-model)
                          (recur (inc d)
                                 (+ acc (* (aget Q (+ (* dst (int d-model)) d))
                                           (aget K (+ (* src (int d-model)) d)))))
                          acc))]
              (aset scores e (m/exp (n/min 5.0 (n/max -5.0 (* dot scale)))))))
        ;; Scatter scores to get normalization Z per node
        Z (double-array n-nodes)
        _ (dotimes [e n-edges]
            (let [dst (aget dst-edges e)]
              (aset Z dst
                    (+ (aget Z dst) (aget scores e)))))
        ;; Compute weighted messages and scatter to nodes
        out (double-array (* n-nodes d-model))
        _ (dotimes [e n-edges]
            (let [src (aget src-edges e)
                  dst (aget dst-edges e)
                  w (aget scores e)]
              (dotimes [d d-model]
                (let [dst-idx (+ (* dst (int d-model)) d)
                      src-idx (+ (* src (int d-model)) d)]
                  (aset out dst-idx
                        (+ (aget out dst-idx)
                           (* w (aget V src-idx))))))))
        ;; Normalize: out[node] /= Z[node]
        _ (dotimes [node n-nodes]
            (let [z (+ (aget Z node) 1e-6)]
              (dotimes [d d-model]
                (let [idx (+ (* node (int d-model)) d)]
                  (aset out idx (/ (aget out idx) z))))))]
    ;; Output projection
    (nn/matmul out Wo n-nodes d-model d-model)))

;; ================================================================
;; Sinusoidal positional embedding
;; ================================================================

(deftm sinusoidal-embedding [timesteps :- (Array long)
                             n :- Long dim :- Long] :- (Array double)
  (let [out (double-array (* n dim))
        half-dim (quot dim 2)
        log-scale (/ (m/log 10000.0) (double (dec half-dim)))]
    (dotimes [t-idx n]
      (let [t (double (aget timesteps t-idx))]
        (dotimes [d half-dim]
          (let [freq (m/exp (* (- (double d)) log-scale))
                angle (* t freq)]
            (aset out (+ (* t-idx (int dim)) d)
                  (m/sin angle))
            (aset out (+ (* t-idx (int dim)) half-dim d)
                  (m/cos angle))))))
    out))

;; ================================================================
;; Backward helper for scaled-dot-product attention
;; ================================================================

(deftm scaled-dot-product-attn-backward
  [d-out :- (Array double)
   Q :- (Array double) K :- (Array double) V :- (Array double)
   seq-q :- Long seq-k :- Long dk :- Long dv :- Long]
  :- (Array Object)
  (let [scale (/ 1.0 (n/sqrt (double dk)))
        ;; Recompute scores and weights
        scores (double-array (* seq-q seq-k))
        weights (double-array (* seq-q seq-k))]
    (dotimes [i seq-q]
      (dotimes [j seq-k]
        (let [dot (loop [d 0 acc 0.0]
                    (if (< d dk)
                      (recur (inc d) (+ acc (* (aget Q (+ (* i (int dk)) d))
                                               (aget K (+ (* j (int dk)) d)))))
                      acc))]
          (aset scores (+ (* i (int seq-k)) j) (* dot scale)))))
    (dotimes [i seq-q]
      (let [offset (* i (int seq-k))
            max-s (loop [j 0 m n/neg-inf]
                    (if (< j seq-k) (recur (inc j) (n/max m (aget scores (+ offset j)))) m))
            sum-exp (loop [j 0 s 0.0]
                      (if (< j seq-k)
                        (let [e (m/exp (- (aget scores (+ offset j)) max-s))]
                          (aset weights (+ offset j) e)
                          (recur (inc j) (+ s e))) s))
            inv-sum (/ 1.0 sum-exp)]
        (dotimes [j seq-k]
          (aset weights (+ offset j) (* (aget weights (+ offset j)) inv-sum)))))
    ;; d_weights = d_out @ V^T -> [seq_q, seq_k]
    (let [d-weights (double-array (* seq-q seq-k))
          _ (blas/dgemm-nt! d-out V d-weights seq-q dv seq-k 1.0 0.0)
          ;; d_V = weights^T @ d_out -> [seq_k, dv]
          dV (double-array (* seq-k dv))
          _ (blas/dgemm-tn! weights d-out dV seq-k seq-q dv 1.0 0.0)
          ;; Backprop through softmax
          d-scores (double-array (* seq-q seq-k))]
      (dotimes [i seq-q]
        (let [offset (* i (int seq-k))
              dot-wdw (loop [j 0 acc 0.0]
                        (if (< j seq-k)
                          (recur (inc j) (+ acc (* (aget weights (+ offset j))
                                                   (aget d-weights (+ offset j)))))
                          acc))]
          (dotimes [j seq-k]
            (aset d-scores (+ offset j)
                  (* scale (aget weights (+ offset j))
                     (- (aget d-weights (+ offset j)) dot-wdw))))))
      ;; dQ = d_scores @ K -> [seq_q, dk]
      (let [dQ (nn/matmul d-scores K seq-q seq-k dk)
            ;; dK = d_scores^T @ Q -> [seq_k, dk]
            dK (double-array (* seq-k dk))
            _ (blas/dgemm-tn! d-scores Q dK seq-k seq-q dk 1.0 0.0)]
        (object-array [dQ dK dV])))))

;; ================================================================
;; Template registration for scaled-dot-product attention
;; ================================================================

(tmpl/merge-into-template! 'raster.dl.attention/scaled-dot-product-attn
                           {:params '[Q K V seq-q seq-k dk dv] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [Q K V seq-q seq-k dk dv] _result-sym adjoint-sym gensym-fn]
                                        (let [grads-arr (gensym-fn "attn_grads")
                                              dQ (gensym-fn "dQ")
                                              dK (gensym-fn "dK")
                                              dV (gensym-fn "dV")]
                                          [(update ctx :bindings into
                                                   [grads-arr (list 'raster.dl.attention/scaled-dot-product-attn-backward
                                                                    adjoint-sym Q K V seq-q seq-k dk dv)
                                                    dQ (list 'clojure.core/aget grads-arr 0)
                                                    dK (list 'clojure.core/aget grads-arr 1)
                                                    dV (list 'clojure.core/aget grads-arr 2)])
                                           [dQ dK dV nil nil nil nil]]))})

;; Forward tangent (JVP, §13 A3) of (bidirectional) SDPA — the standard form,
;; single output array (mirrors the backward's recompute-then-gemm structure):
;;   W  = softmax(s·Q·Kᵀ);  dZ = s·(dQ·Kᵀ + Q·dKᵀ)
;;   dW = W ⊙ (dZ − rowsum(W⊙dZ));  dO = dW·V + W·dV
(deftm scaled-dot-product-attn-jvp
  [dQ :- (Array double) dK :- (Array double) dV :- (Array double)
   Q :- (Array double) K :- (Array double) V :- (Array double)
   seq-q :- Long seq-k :- Long dk :- Long dv :- Long]
  :- (Array double)
  (let [scale (/ 1.0 (n/sqrt (double dk)))
        ;; Recompute softmax weights (same as the backward)
        weights (double-array (* seq-q seq-k))
        _ (blas/dgemm-nt! Q K weights seq-q dk seq-k scale 0.0)
        _ (dotimes [i seq-q]
            (let [offset (* i (int seq-k))
                  max-s (loop [j 0 m n/neg-inf]
                          (if (< j seq-k)
                            (recur (inc j) (n/max m (aget weights (+ offset j))))
                            m))
                  sum-exp (loop [j 0 s 0.0]
                            (if (< j seq-k)
                              (let [e (m/exp (- (aget weights (+ offset j)) max-s))]
                                (aset weights (+ offset j) e)
                                (recur (inc j) (+ s e)))
                              s))
                  inv-sum (/ 1.0 sum-exp)]
              (dotimes [j seq-k]
                (aset weights (+ offset j) (* (aget weights (+ offset j)) inv-sum)))))
        ;; dZ = scale·(dQ·Kᵀ + Q·dKᵀ)
        dZ (double-array (* seq-q seq-k))
        _ (blas/dgemm-nt! dQ K dZ seq-q dk seq-k scale 0.0)
        _ (blas/dgemm-nt! Q dK dZ seq-q dk seq-k scale 1.0)
        ;; softmax JVP per row: dW = W⊙(dZ − ⟨W,dZ⟩_row)
        dW (double-array (* seq-q seq-k))
        _ (dotimes [i seq-q]
            (let [offset (* i (int seq-k))
                  wdz (loop [j 0 s 0.0]
                        (if (< j seq-k)
                          (recur (inc j) (+ s (* (aget weights (+ offset j))
                                                 (aget dZ (+ offset j)))))
                          s))]
              (dotimes [j seq-k]
                (aset dW (+ offset j)
                      (* (aget weights (+ offset j))
                         (- (aget dZ (+ offset j)) wdz))))))
        ;; dO = dW·V + W·dV
        dO (double-array (* seq-q dv))
        _ (blas/dgemm! dW V dO seq-q seq-k dv 1.0 0.0)
        _ (blas/dgemm! weights dV dO seq-q seq-k dv 1.0 1.0)]
    dO))

;; SDPA :jvp-fn — one kernel call; absent tangents get typed zeros (the
;; pushforward is jointly linear in (dQ,dK,dV), so zeros are exact).
(tmpl/merge-into-template!
 'raster.dl.attention/scaled-dot-product-attn
 {:jvp-fn
  (fn [ctx [Q K V seq-q seq-k dk dv] tangent-args _result-sym gensym-fn]
    (let [dQ (nth tangent-args 0 nil)
          dK (nth tangent-args 1 nil)
          dV (nth tangent-args 2 nil)]
      (when-not (or dQ dK dV)
        (throw (ex-info "scaled-dot-product-attn jvp: no active tangent reached the call"
                        {:op 'raster.dl.attention/scaled-dot-product-attn})))
      (tmpl/bind-jvp-term ctx gensym-fn "jsdpa"
                          (list 'raster.dl.attention/scaled-dot-product-attn-jvp
                                (or dQ (tmpl/jvp-zero-like Q))
                                (or dK (tmpl/jvp-zero-like K))
                                (or dV (tmpl/jvp-zero-like V))
                                Q K V seq-q seq-k dk dv))))})

;; ---------------------------------------------------------------------------
;; PREFILL variants (S3 GPU embed mode): a whole T-token block per replay, no KV
;; cache, positions from the work-item index. Causal mask via -1e30 scores.
;; ---------------------------------------------------------------------------

(deftm rope-prefill! (All [T] [x :- (Array T) out :- (Array T)
                               nrows :- Long heads :- Long head-dim :- Long
                               theta :- Double] :- Void
                          (raster.par/map-void! idx (clojure.core/* nrows (clojure.core/* heads (quot head-dim 2)))
                                                (let [hdim2 (quot head-dim 2)
                                                      per-row (clojure.core/* heads hdim2)
                                                      t (quot idx per-row)
                                                      rest0 (rem idx per-row)
                                                      h (quot rest0 hdim2)
                                                      i (rem rest0 hdim2)
                                                      base (clojure.core/+ (clojure.core/* t (clojure.core/* heads head-dim))
                                                                           (clojure.core/* h head-dim))
                                                      ln-theta (m/log theta)
                                                      freq (m/exp (* (/ (* -2.0 (double i)) (double head-dim)) ln-theta))
                                                      ang (* (double t) freq)
                                                      c (m/cos ang) s (m/sin ang)
                                                      x0 (aget x (clojure.core/+ base i))
                                                      x1 (aget x (clojure.core/+ (clojure.core/+ base i) hdim2))]
                                                  (aset out (clojure.core/+ base i) (- (* x0 c) (* x1 s)))
                                                  (aset out (clojure.core/+ (clojure.core/+ base i) hdim2) (+ (* x1 c) (* x0 s)))))))

;; Causal batched attention, 3 phases. q:[T,nq*hd] k,v:[T,nkv*hd] row-major;
;; sc:[T*nq, T] scores/probs scratch; out:[T, nq*hd].
(deftm attn-prefill-scores! (All [T] [q :- (Array T) k :- (Array T) sc :- (Array T)
                                      nrows :- Long n-q :- Long group :- Long
                                      n-kv :- Long head-dim :- Long
                                      scale :- Double] :- Void
                                 (raster.par/map-void! idx (clojure.core/* nrows (clojure.core/* n-q nrows))
                                                       (let [per-i (clojure.core/* n-q nrows)
                                                             i (quot idx per-i)
                                                             rest0 (rem idx per-i)
                                                             hq (quot rest0 nrows)
                                                             j (rem rest0 nrows)
                                                             row (clojure.core/+ (clojure.core/* i n-q) hq)]
                                                         (if (< i j)
                                                           (aset sc (clojure.core/+ (clojure.core/* row nrows) j) -1.0e30)
                                                           (let [hkv (quot hq group)
                                                                 qb (clojure.core/+ (clojure.core/* i (clojure.core/* n-q head-dim))
                                                                                    (clojure.core/* hq head-dim))
                                                                 kb (clojure.core/+ (clojure.core/* j (clojure.core/* n-kv head-dim))
                                                                                    (clojure.core/* hkv head-dim))
                                                                 dot (loop [d 0 acc 0.0]
                                                                       (if (< d head-dim)
                                                                         (recur (inc d)
                                                                                (+ acc (* (aget q (clojure.core/+ qb d))
                                                                                          (aget k (clojure.core/+ kb d)))))
                                                                         acc))]
                                                             (aset sc (clojure.core/+ (clojure.core/* row nrows) j) (* dot scale))))))))

(deftm attn-prefill-softmax! (All [T] [sc :- (Array T)
                                       nrows :- Long n-q :- Long] :- Void
                                  (raster.par/map-void! row (clojure.core/* nrows n-q)
                                                        (let [scb (clojure.core/* row nrows)
                                                              mx (loop [j 0 mm -1.0e30]
                                                                   (if (< j nrows)
                                                                     (recur (inc j) (n/max mm (aget sc (clojure.core/+ scb j)))) mm))
                                                              sum (loop [j 0 s 0.0]
                                                                    (if (< j nrows)
                                                                      (let [e (m/exp (- (aget sc (clojure.core/+ scb j)) mx))]
                                                                        (aset sc (clojure.core/+ scb j) e)
                                                                        (recur (inc j) (+ s e)))
                                                                      s))
                                                              inv (/ 1.0 sum)]
                                                          (loop [j 0]
                                                            (if (< j nrows)
                                                              (do (aset sc (clojure.core/+ scb j) (* (aget sc (clojure.core/+ scb j)) inv))
                                                                  (recur (inc j)))
                                                              nil))))))

(deftm attn-prefill-out! (All [T] [sc :- (Array T) v :- (Array T) out :- (Array T)
                                   nrows :- Long n-q :- Long group :- Long
                                   n-kv :- Long head-dim :- Long] :- Void
                              (raster.par/map-void! idx (clojure.core/* nrows (clojure.core/* n-q head-dim))
                                                    (let [per-i (clojure.core/* n-q head-dim)
                                                          i (quot idx per-i)
                                                          rest0 (rem idx per-i)
                                                          hq (quot rest0 head-dim)
                                                          d (rem rest0 head-dim)
                                                          row (clojure.core/+ (clojure.core/* i n-q) hq)
                                                          scb (clojure.core/* row nrows)
                                                          hkvb (clojure.core/+ (clojure.core/* (quot hq group) head-dim) d)
                                                          kvstride (clojure.core/* n-kv head-dim)]
                                                      (aset out (clojure.core/+ (clojure.core/* i per-i) (clojure.core/+ (clojure.core/* hq head-dim) d))
                                                            (loop [j 0 a 0.0]
                                                              (if (< j nrows)
                                                                (recur (inc j)
                                                                       (+ a (* (aget sc (clojure.core/+ scb j))
                                                                               (aget v (clojure.core/+ (clojure.core/* j kvstride) hkvb)))))
                                                                a)))))))

;; Bidirectional scores (EmbeddingGemma-style encoder): all-to-all, no causal mask.
;; (Symmetric sliding window |i-j| < w only matters for T > window — the binder
;; asserts T <= window, so full attention here is exact.)
(deftm attn-prefill-scores-bidir! (All [T] [q :- (Array T) k :- (Array T) sc :- (Array T)
                                            nrows :- Long n-q :- Long group :- Long
                                            n-kv :- Long head-dim :- Long
                                            scale :- Double] :- Void
                                       (raster.par/map-void! idx (clojure.core/* nrows (clojure.core/* n-q nrows))
                                                             (let [per-i (clojure.core/* n-q nrows)
                                                                   i (quot idx per-i)
                                                                   rest0 (rem idx per-i)
                                                                   hq (quot rest0 nrows)
                                                                   j (rem rest0 nrows)
                                                                   row (clojure.core/+ (clojure.core/* i n-q) hq)
                                                                   hkv (quot hq group)
                                                                   qb (clojure.core/+ (clojure.core/* i (clojure.core/* n-q head-dim))
                                                                                      (clojure.core/* hq head-dim))
                                                                   kb (clojure.core/+ (clojure.core/* j (clojure.core/* n-kv head-dim))
                                                                                      (clojure.core/* hkv head-dim))
                                                                   dot (loop [d 0 acc 0.0]
                                                                         (if (< d head-dim)
                                                                           (recur (inc d)
                                                                                  (+ acc (* (aget q (clojure.core/+ qb d))
                                                                                            (aget k (clojure.core/+ kb d)))))
                                                                           acc))]
                                                               (aset sc (clojure.core/+ (clojure.core/* row nrows) j) (* dot scale))))))
