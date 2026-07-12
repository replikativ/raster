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
;; Resident training RoPE forward: same NeoX/HF rotate-half as rope-prefill!,
;; but ALLOCATES + RETURNS out (the AD-templated (Array T) primitive). The grid
;; is flattened over seq-len·heads·(head-dim/2) work-items (one rotation each),
;; so it lowers to a single resident :map-void kernel — the old raw nested
;; dotimes the resident extractor rejected. Pairing/theta semantics are bit-
;; identical to rope-prefill! (t = position p): out[base+i] = x0·c − x1·s,
;; out[base+i+hdim2] = x1·c + x0·s, with freq = θ^(−2i/d), ang = p·freq.
(deftm rope (All [T] [x :- (Array T) seq-len :- Long heads :- Long
                      head-dim :- Long theta :- Double] :- (Array T)
                 (let [out (alloc-like x (* seq-len (* heads head-dim)))]
                   (raster.par/map-void! idx (clojure.core/* seq-len (clojure.core/* heads (quot head-dim 2)))
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
                                           (aset out (clojure.core/+ (clojure.core/+ base i) hdim2) (+ (* x1 c) (* x0 s)))))
                   out)))

;; RoPE backward: RoPE applies the orthogonal rotation R(ang) to each (x0,x1)
;; pair, so the pullback is the transpose rotation R(-ang) applied to (dy0,dy1):
;;   dx0 =  c·dy0 + s·dy1
;;   dx1 = −s·dy0 + c·dy1
;; No trainable params (theta/positions are fixed) — only x gets a gradient.
;; Resident RoPE backward: the transpose rotation R(−ang) on (dy0,dy1), flattened
;; over the same seq-len·heads·(head-dim/2) grid as the forward → one resident
;; :map-void kernel. c/s recomputed identically to the forward (theta-exact):
;;   dx[base+i]       =  c·dy0 + s·dy1
;;   dx[base+i+hdim2] =  c·dy1 − s·dy0
(deftm rope-backward-dx (All [T] [dy :- (Array T) seq-len :- Long heads :- Long
                                  head-dim :- Long theta :- Double] :- (Array T)
                             (let [dx (alloc-like dy (* seq-len (* heads head-dim)))]
                               (raster.par/map-void! idx (clojure.core/* seq-len (clojure.core/* heads (quot head-dim 2)))
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
                                                           d0 (aget dy (clojure.core/+ base i))
                                                           d1 (aget dy (clojure.core/+ (clojure.core/+ base i) hdim2))]
                                                       (aset dx (clojure.core/+ base i) (+ (* d0 c) (* d1 s)))
                                                       (aset dx (clojure.core/+ (clojure.core/+ base i) hdim2) (- (* d1 c) (* d0 s)))))
                               dx)))

(tmpl/merge-into-template! 'raster.dl.attention/rope
                           {:params '[x seq-len heads head-dim theta] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x seq-len heads head-dim theta] _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx" (tmpl/grad-tag x))]
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

;; --- Partial INTERLEAVED RoPE at an absolute position (GPT-J convention) ---
;; Rotates only the first rotary-dim dims of each head, pairing ADJACENT
;; elements (2j, 2j+1) — the GPT-J/moonshine convention. NOT interchangeable
;; with rope-pos above, which is NeoX HALF-SPLIT (i, i+half) over the full
;; head_dim: different pairing AND partial coverage (THE moonshine port trap,
;; validated against HF activations). x is ONE position [heads, head_dim]
;; flattened; dims [rotary-dim, head_dim) pass through untouched. Frequencies
;; via pow (theta^(-2j/rotary-dim)) to match reference implementations
;; bit-exactly (exp/log recomposition differs in ulps). In place; returns x.
(deftm rope-pos-partial! (All [T] [x :- (Array T) heads :- Long head-dim :- Long
                                   rotary-dim :- Long theta :- Double pos :- Long]
                              :- (Array T)
                              (let [pairs (quot rotary-dim 2)
                                    posd (double pos)]
                                (dotimes [h heads]
                                  (let [base (* h (int head-dim))]
                                    (dotimes [j pairs]
                                      (let [freq (n/pow theta (/ (* -2.0 (double j)) (double rotary-dim)))
                                            ang (* posd freq)
                                            c (m/cos ang) s (m/sin ang)
                                            i0 (+ base (* 2 j))
                                            i1 (inc i0)
                                            x0 (aget x i0) x1 (aget x i1)]
                                        (aset x i0 (- (* x0 c) (* x1 s)))
                                        (aset x i1 (+ (* x1 c) (* x0 s)))))))
                                x)))

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

;; --- Decode attention that also captures the attention-weight distribution ---
;; Identical layout/scale semantics (and bit-identical output) to
;; gqa-decode-attention above, plus ACCUMULATES the head-averaged softmax
;; weights into wsink (length >= cache-len): wsink[j] += (1/n_q) * weight[hq,j].
;; This is the cross-attention alignment signal for DTW word timestamps
;; (whisper/moonshine style) — call with a zeroed wsink per decode step, or let
;; it accumulate across layers for a layer+head average.
(deftm gqa-decode-attention-weights! (All [T]
                                          [q :- (Array T) k :- (Array T) v :- (Array T)
                                           cache-len :- Long n-q :- Long n-kv :- Long
                                           head-dim :- Long scale :- Double
                                           wsink :- (Array T)] :- (Array T)
                                          (let [out (alloc-like q (* n-q head-dim))
                                                group (quot n-q n-kv)
                                                havg (/ 1.0 (double n-q))
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
                                                (dotimes [j cache-len]
                                                  (aset wsink j (+ (aget wsink j)
                                                                   (* havg (* (aget sc j) inv)))))
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

;; Backward for causal-scaled-dot-product-attn — THREE flat resident par/map-void!
;; kernels, one per gradient (dQ / dK / dV), each returning a SEPARATE (Array T)
;; (NOT an object-array bundle → NO lost float tag). Same recompute-the-row-softmax
;; philosophy as batched-causal-sdpa-dq/dk/dv (see there): every work-item owns a
;; DISJOINT output row (no atomics/host-dotimes/blit). Conventions:
;;   score_ij = (Q_i·K_j) / sqrt(dk)   (causal j≤i)   w_ij = softmax_j(score)
;;   dw_ij    = ⟨dO_i, V_j⟩            D_i = Σ_{j≤i} w_ij·dw_ij
;;   dQ_i = (1/√dk)·Σ_{j≤i} w_ij·(dw_ij−D_i)·K_j
;;   dK_j = (1/√dk)·Σ_{i≥j} w_ij·(dw_ij−D_i)·Q_i     dV_j = Σ_{i≥j} w_ij·dO_i
;; UNLIKE batched-causal-sdpa these split the Q/K feature dim (dk) from the V/out
;; feature dim (dv). Every T-touching scalar stays T-typed via the SAME two
;; mechanisms as the forward: reduction seeds are bare floating literals that narrow
;; to the element dtype, and the 1/√dk scale is a DIVISION by a data-scalar-typed
;; (n/oftype <val> (n/sqrt (double dk))) — NEVER a Double literal × T.

(deftm causal-scaled-dot-product-attn-dq
  "dQ of causal-scaled-dot-product-attn. Parallel over query row i (disjoint):
  recomputes the causal row softmax + D_i, then
  dQ[i,d] = (1/√dk)·Σ_{j≤i} w_ij·(⟨dO_i,V_j⟩ − D_i)·K[j,d]."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        seq-len :- Long dk :- Long dv :- Long]
       :- (Array T)
       (let [dQ (alloc-like Q (* seq-len dk))]
         (raster.par/map-void! i seq-len
                               (let [qrow (clojure.core/* i dk)
                                     orow (clojure.core/* i dv)
                                     mx (loop [j 0 mm -1.0e38]
                                          (if (<= j i)
                                            (let [krow (clojure.core/* j dk)
                                                  dot (loop [d 0 acc 0.0]
                                                        (if (< d dk)
                                                          (recur (inc d)
                                                                 (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                           (aget K (clojure.core/+ krow d)))))
                                                          acc))]
                                              (recur (inc j) (n/max mm (/ dot (n/oftype dot (n/sqrt (double dk)))))))
                                            mm))
                                     sum (loop [j 0 s 0.0]
                                           (if (<= j i)
                                             (let [krow (clojure.core/* j dk)
                                                   dot (loop [d 0 acc 0.0]
                                                         (if (< d dk)
                                                           (recur (inc d)
                                                                  (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                            (aget K (clojure.core/+ krow d)))))
                                                           acc))]
                                               (recur (inc j) (+ s (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)))))
                                             s))
                                     inv (/ 1.0 sum)
                                     Di (loop [j 0 acc 0.0]
                                          (if (<= j i)
                                            (let [krow (clojure.core/* j dk)
                                                  vrow (clojure.core/* j dv)
                                                  dot (loop [d 0 a 0.0]
                                                        (if (< d dk)
                                                          (recur (inc d)
                                                                 (+ a (* (aget Q (clojure.core/+ qrow d))
                                                                         (aget K (clojure.core/+ krow d)))))
                                                          a))
                                                  w  (* (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)) inv)
                                                  dw (loop [d 0 a 0.0]
                                                       (if (< d dv)
                                                         (recur (inc d)
                                                                (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                        (aget V (clojure.core/+ vrow d)))))
                                                         a))]
                                              (recur (inc j) (+ acc (* w dw))))
                                            acc))]
                                 (loop [d 0]
                                   (if (< d dk)
                                     (do (aset dQ (clojure.core/+ qrow d) 0.0) (recur (inc d)))
                                     nil))
                                 (loop [j 0]
                                   (if (<= j i)
                                     (let [krow (clojure.core/* j dk)
                                           vrow (clojure.core/* j dv)
                                           dot (loop [d 0 a 0.0]
                                                 (if (< d dk)
                                                   (recur (inc d)
                                                          (+ a (* (aget Q (clojure.core/+ qrow d))
                                                                  (aget K (clojure.core/+ krow d)))))
                                                   a))
                                           w  (* (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)) inv)
                                           dw (loop [d 0 a 0.0]
                                                (if (< d dv)
                                                  (recur (inc d)
                                                         (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                 (aget V (clojure.core/+ vrow d)))))
                                                  a))
                                           c  (* w (- dw Di))]
                                       (loop [d 0]
                                         (if (< d dk)
                                           (do (aset dQ (clojure.core/+ qrow d)
                                                     (+ (aget dQ (clojure.core/+ qrow d))
                                                        (* c (aget K (clojure.core/+ krow d)))))
                                               (recur (inc d)))
                                           nil))
                                       (recur (inc j)))
                                     nil))
                                 (loop [d 0]
                                   (if (< d dk)
                                     (let [v (aget dQ (clojure.core/+ qrow d))]
                                       (aset dQ (clojure.core/+ qrow d) (/ v (n/oftype v (n/sqrt (double dk)))))
                                       (recur (inc d)))
                                     nil))))
         dQ)))

(deftm causal-scaled-dot-product-attn-dv
  "dV of causal-scaled-dot-product-attn. Parallel over key row j (disjoint). Loops
  queries i≥j (causal), recomputing query i's row softmax to get w_ij, and
  accumulates dV[j,d] = Σ_{i≥j} w_ij·dO[i,d]."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        seq-len :- Long dk :- Long dv :- Long]
       :- (Array T)
       (let [dV (alloc-like Q (* seq-len dv))]
         (raster.par/map-void! j seq-len
                               (let [jrowk (clojure.core/* j dk)
                                     jrowv (clojure.core/* j dv)]
                                 (loop [d 0]
                                   (if (< d dv)
                                     (do (aset dV (clojure.core/+ jrowv d) 0.0) (recur (inc d)))
                                     nil))
                                 (loop [i j]
                                   (if (< i seq-len)
                                     (let [qrow (clojure.core/* i dk)
                                           orow (clojure.core/* i dv)
                                           mx (loop [k 0 mm -1.0e38]
                                                (if (<= k i)
                                                  (let [krow (clojure.core/* k dk)
                                                        dot (loop [d 0 acc 0.0]
                                                              (if (< d dk)
                                                                (recur (inc d)
                                                                       (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                 (aget K (clojure.core/+ krow d)))))
                                                                acc))]
                                                    (recur (inc k) (n/max mm (/ dot (n/oftype dot (n/sqrt (double dk)))))))
                                                  mm))
                                           sum (loop [k 0 s 0.0]
                                                 (if (<= k i)
                                                   (let [krow (clojure.core/* k dk)
                                                         dot (loop [d 0 acc 0.0]
                                                               (if (< d dk)
                                                                 (recur (inc d)
                                                                        (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                  (aget K (clojure.core/+ krow d)))))
                                                                 acc))]
                                                     (recur (inc k) (+ s (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)))))
                                                   s))
                                           inv (/ 1.0 sum)
                                           dotij (loop [d 0 acc 0.0]
                                                   (if (< d dk)
                                                     (recur (inc d)
                                                            (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                      (aget K (clojure.core/+ jrowk d)))))
                                                     acc))
                                           w (* (m/exp (- (/ dotij (n/oftype dotij (n/sqrt (double dk)))) mx)) inv)]
                                       (loop [d 0]
                                         (if (< d dv)
                                           (do (aset dV (clojure.core/+ jrowv d)
                                                     (+ (aget dV (clojure.core/+ jrowv d))
                                                        (* w (aget d-out (clojure.core/+ orow d)))))
                                               (recur (inc d)))
                                           nil))
                                       (recur (inc i)))
                                     nil))))
         dV)))

(deftm causal-scaled-dot-product-attn-dk
  "dK of causal-scaled-dot-product-attn. Parallel over key row j (disjoint). Loops
  queries i≥j (causal); for each, recomputes query i's row softmax + D_i and
  accumulates dK[j,d] = (1/√dk)·Σ_{i≥j} w_ij·(⟨dO_i,V_j⟩ − D_i)·Q[i,d]."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        seq-len :- Long dk :- Long dv :- Long]
       :- (Array T)
       (let [dK (alloc-like Q (* seq-len dk))]
         (raster.par/map-void! j seq-len
                               (let [jrowk (clojure.core/* j dk)
                                     jrowv (clojure.core/* j dv)]
                                 (loop [d 0]
                                   (if (< d dk)
                                     (do (aset dK (clojure.core/+ jrowk d) 0.0) (recur (inc d)))
                                     nil))
                                 (loop [i j]
                                   (if (< i seq-len)
                                     (let [qrow (clojure.core/* i dk)
                                           orow (clojure.core/* i dv)
                                           mx (loop [k 0 mm -1.0e38]
                                                (if (<= k i)
                                                  (let [krow (clojure.core/* k dk)
                                                        dot (loop [d 0 acc 0.0]
                                                              (if (< d dk)
                                                                (recur (inc d)
                                                                       (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                 (aget K (clojure.core/+ krow d)))))
                                                                acc))]
                                                    (recur (inc k) (n/max mm (/ dot (n/oftype dot (n/sqrt (double dk)))))))
                                                  mm))
                                           sum (loop [k 0 s 0.0]
                                                 (if (<= k i)
                                                   (let [krow (clojure.core/* k dk)
                                                         dot (loop [d 0 acc 0.0]
                                                               (if (< d dk)
                                                                 (recur (inc d)
                                                                        (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                  (aget K (clojure.core/+ krow d)))))
                                                                 acc))]
                                                     (recur (inc k) (+ s (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)))))
                                                   s))
                                           inv (/ 1.0 sum)
                                           Di (loop [k 0 acc 0.0]
                                                (if (<= k i)
                                                  (let [krow (clojure.core/* k dk)
                                                        vrow (clojure.core/* k dv)
                                                        dot (loop [d 0 a 0.0]
                                                              (if (< d dk)
                                                                (recur (inc d)
                                                                       (+ a (* (aget Q (clojure.core/+ qrow d))
                                                                               (aget K (clojure.core/+ krow d)))))
                                                                a))
                                                        w  (* (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)) inv)
                                                        dw (loop [d 0 a 0.0]
                                                             (if (< d dv)
                                                               (recur (inc d)
                                                                      (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                              (aget V (clojure.core/+ vrow d)))))
                                                               a))]
                                                    (recur (inc k) (+ acc (* w dw))))
                                                  acc))
                                           dotij (loop [d 0 acc 0.0]
                                                   (if (< d dk)
                                                     (recur (inc d)
                                                            (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                      (aget K (clojure.core/+ jrowk d)))))
                                                     acc))
                                           wij (* (m/exp (- (/ dotij (n/oftype dotij (n/sqrt (double dk)))) mx)) inv)
                                           dwij (loop [d 0 a 0.0]
                                                  (if (< d dv)
                                                    (recur (inc d)
                                                           (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                   (aget V (clojure.core/+ jrowv d)))))
                                                    a))
                                           c (* wij (- dwij Di))]
                                       (loop [d 0]
                                         (if (< d dk)
                                           (do (aset dK (clojure.core/+ jrowk d)
                                                     (+ (aget dK (clojure.core/+ jrowk d))
                                                        (* c (aget Q (clojure.core/+ qrow d)))))
                                               (recur (inc d)))
                                           nil))
                                       (recur (inc i)))
                                     nil))
                                 (loop [d 0]
                                   (if (< d dk)
                                     (let [v (aget dK (clojure.core/+ jrowk d))]
                                       (aset dK (clojure.core/+ jrowk d) (/ v (n/oftype v (n/sqrt (double dk)))))
                                       (recur (inc d)))
                                     nil))))
         dK)))

;; grads-fn (compile-aot flat codegen): three direct (Array T) kernel calls — one
;; per gradient (dQ/dK/dV). No object-array bundle, no clojure.core/aget-on-Object:
;; each grad carries its own float/double tag so composed cotangents devirtualize.

(tmpl/merge-into-template! 'raster.dl.attention/causal-scaled-dot-product-attn
                           {:params '[Q K V seq-len dk dv]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [Q K V seq-len dk dv] _result-sym adjoint-sym gensym-fn]
                              (let [dQ (gensym-fn "dQ" (tmpl/grad-tag Q))
                                    dK (gensym-fn "dK" (tmpl/grad-tag K))
                                    dV (gensym-fn "dV" (tmpl/grad-tag V))]
                                [(update ctx :bindings into
                                         [dQ (list 'raster.dl.attention/causal-scaled-dot-product-attn-dq
                                                   adjoint-sym Q K V seq-len dk dv)
                                          dK (list 'raster.dl.attention/causal-scaled-dot-product-attn-dk
                                                   adjoint-sym Q K V seq-len dk dv)
                                          dV (list 'raster.dl.attention/causal-scaled-dot-product-attn-dv
                                                   adjoint-sym Q K V seq-len dk dv)])
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
;; batched-causal-sdpa: causal SDPA over `batch` contiguous [seq,hd] head slabs
;; (Q,K,V all packed [batch,seq,hd]). Opaque to AD (its own flat :grads-fn below
;; calls the batched backward) so the head iteration NEVER enters the reverse-AD
;; tape — that is what keeps gqa-causal-mha's gradient a straight-line let* (no
;; fn* pullback closure / ArrayList tape).
;;
;; SHAPE (the occupancy rule for ALL of forward+backward here). A kernel parallel
;; over (batch, query row) is only b·seq work items — 256 at gemma training dims
;; (b = n_q = 4 heads, seq = 64) — on a GPU with thousands of lanes. EVERY kernel
;; in this group is therefore parallel over OUTPUT ELEMENTS (b·seq·hd = 65536) or
;; over SCORE ELEMENTS (b·seq² = 16384); the only b·seq-wide kernels left are the
;; two O(seq) row sweeps (softmax stats, D_i), which carry no hd factor.
;;
;; The enabler is the MATERIALIZED weight matrix W[b,i,j] = softmax_j(causal scores)
;; (batched-causal-attn-weights): with W in hand, every other kernel is a plain
;; disjoint accumulation over its own output element. The forward reads it (out =
;; W·V), and so do dQ/dK/dV. The three score-level intermediates —
;;   W    = softmax_j(Q·Kᵀ/√hd)            (batched-causal-attn-weights)
;;   DW   = ⟨dO_i, V_j⟩                     (batched-causal-attn-dw)
;;   D_i  = Σ_{j≤i} W_ij·DW_ij  (= ⟨dO_i,out_i⟩)  (batched-causal-attn-dsum)
;; — are CALLED, not inlined by hand, precisely so that a fused value+grad program
;; CSEs them: the VJP's own forward re-run and dQ/dK/dV then share ONE W (and dQ/dK
;; share one DW and one D), instead of each kernel recomputing the row softmax.
;; That is the store-vs-recompute choice made explicit: we store O(batch·seq²)
;; scores (64 KB at gemma training dims) and recompute nothing.
;; Cost: memory is O(batch·seq²) rather than the streaming forward's O(batch·seq·hd)
;; — the ordinary non-flash-attention footprint (a flash-style tiled kernel would
;; trade it back for recompute; it is not what this hardware path needs at seq 64).
;;
;; (All [T]), fully element-typed (float on GPU / double on CPU). Every scalar that
;; touches T-data stays T-typed via the same two mechanisms throughout: (1) reduction
;; accumulator seeds are BARE floating literals (0.0 / -1.0e38) inlined into the loop
;; head — a bare floating-literal loop-var init NARROWS to the element dtype, while a
;; let-bound/oftype seed of a CONSTANT folds to an untyped literal and loses the type;
;; (2) the 1/sqrt(head-dim) scale is a DIVISION by a data-scalar-typed
;; `(n/oftype <dot|acc> (n/sqrt (double head-dim)))` — NEVER a Double literal × T (no
;; monomorphic overload → undevirtualized dispatch → GPU garbage). `-1.0e38` is the
;; OpenCL-safe -inf sentinel (OpenCL has no NEG_INFINITY). Batches own disjoint slabs
;; (no cross-batch accumulation — the GQA kv-head fan-in is handled by sum-kv-heads
;; OUTSIDE this op, since K/V are already broadcast to `batch = n_q`).
;; ----------------------------------------------------------------

(deftm batched-causal-attn-weights
  "Materialized causal softmax weight matrix of batched-causal-sdpa:
  W[b,i,j] = softmax_j((Qb[i]·Kb[j])/√hd) for j≤i, 0 above the diagonal —
  [batch, seq, seq], query-row-major. TWO kernels: (1) raw scaled scores, parallel
  over (b,i,j) — one hd-dot per work item (n = b·seq², the occupancy carrier);
  (2) per query row (b,i) — running max + denom over the ≤i scores (an O(seq)
  serial sweep with NO hd factor, so the low-width b·seq shape is cheap here) and
  the normalized-weight writeback covering the FULL row (explicit 0 above the
  diagonal — no reliance on alloc zero-init). Scores land in a separate scratch
  buffer S (never rewritten in place — no read/write alias within a kernel).
  Shared by the forward and by dQ/dK/dV (CSE'd to ONE computation in a fused
  value+grad program); not AD-registered itself."
  (All [T]
       [Q :- (Array T) K :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array T)
       (let [slab (* seq-len head-dim)
             ss (* seq-len seq-len)
             S (alloc-like Q (* batch ss))
             W (alloc-like Q (* batch ss))]
         ;; kernel 1: raw scaled causal scores S[b,i,j] (0 above the diagonal)
         (raster.par/map-void! t (clojure.core/* batch ss)
                               (let [b (quot t ss)
                                     r (rem t ss)
                                     i (quot r seq-len)
                                     j (rem r seq-len)
                                     boff (clojure.core/* b slab)]
                                 (if (<= j i)
                                   (let [qrow (clojure.core/+ boff (clojure.core/* i head-dim))
                                         krow (clojure.core/+ boff (clojure.core/* j head-dim))
                                         dot (loop [d 0 acc 0.0]
                                               (if (< d head-dim)
                                                 (recur (inc d)
                                                        (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                  (aget K (clojure.core/+ krow d)))))
                                                 acc))]
                                     (aset S t (/ dot (n/oftype dot (n/sqrt (double head-dim))))))
                                   (aset S t 0.0))))
         ;; kernel 2: per query row — mx/Z over the ≤i scores, write normalized weights
         (raster.par/map-void! bi (clojure.core/* batch seq-len)
                               (let [b (quot bi seq-len)
                                     i (rem bi seq-len)
                                     roff (clojure.core/+ (clojure.core/* b ss) (clojure.core/* i seq-len))
                                     mx (loop [j 0 mm -1.0e38]
                                          (if (<= j i)
                                            (recur (inc j) (n/max mm (aget S (clojure.core/+ roff j))))
                                            mm))
                                     sum (loop [j 0 s 0.0]
                                           (if (<= j i)
                                             (recur (inc j) (+ s (m/exp (- (aget S (clojure.core/+ roff j)) mx))))
                                             s))
                                     inv (/ 1.0 sum)]
                                 (loop [j 0]
                                   (if (< j seq-len)
                                     (do (if (<= j i)
                                           (aset W (clojure.core/+ roff j)
                                                 (* (m/exp (- (aget S (clojure.core/+ roff j)) mx)) inv))
                                           (aset W (clojure.core/+ roff j) 0.0))
                                         (recur (inc j)))
                                     nil))))
         W)))

(deftm batched-causal-attn-dw
  "DW[b,i,j] = ⟨dO_i, V_j⟩ for j≤i (0 above the diagonal) — the UNNORMALIZED weight
  adjoint of batched-causal-sdpa, [batch, seq, seq] query-row-major. Parallel over
  score elements (b,i,j): one hd-dot per work item (n = b·seq²). Shared by dQ and dK
  (CSE'd to one computation); not AD-registered itself."
  (All [T]
       [d-out :- (Array T) V :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array T)
       (let [slab (* seq-len head-dim)
             ss (* seq-len seq-len)
             DW (alloc-like V (* batch ss))]
         (raster.par/map-void! t (clojure.core/* batch ss)
                               (let [b (quot t ss)
                                     r (rem t ss)
                                     i (quot r seq-len)
                                     j (rem r seq-len)
                                     boff (clojure.core/* b slab)]
                                 (if (<= j i)
                                   (let [orow (clojure.core/+ boff (clojure.core/* i head-dim))
                                         vrow (clojure.core/+ boff (clojure.core/* j head-dim))
                                         dw (loop [d 0 acc 0.0]
                                              (if (< d head-dim)
                                                (recur (inc d)
                                                       (+ acc (* (aget d-out (clojure.core/+ orow d))
                                                                 (aget V (clojure.core/+ vrow d)))))
                                                acc))]
                                     (aset DW t dw))
                                   (aset DW t 0.0))))
         DW)))

(deftm batched-causal-attn-dsum
  "D[b,i] = Σ_{j≤i} W[b,i,j]·DW[b,i,j] (= ⟨dO_i, out_i⟩) — the softmax-Jacobian row
  term of batched-causal-sdpa's backward, [batch, seq]. Parallel over query rows
  (b,i): an O(seq) serial sweep with NO hd factor, so the narrow b·seq shape costs
  nothing here. Shared by dQ and dK (CSE'd to one computation)."
  (All [T]
       [W :- (Array T) DW :- (Array T)
        batch :- Long seq-len :- Long]
       :- (Array T)
       (let [ss (* seq-len seq-len)
             D (alloc-like W (* batch seq-len))]
         (raster.par/map-void! bi (clojure.core/* batch seq-len)
                               (let [b (quot bi seq-len)
                                     i (rem bi seq-len)
                                     roff (clojure.core/+ (clojure.core/* b ss) (clojure.core/* i seq-len))
                                     acc (loop [j 0 acc 0.0]
                                           (if (<= j i)
                                             (recur (inc j)
                                                    (+ acc (* (aget W (clojure.core/+ roff j))
                                                              (aget DW (clojure.core/+ roff j)))))
                                             acc))]
                                 (aset D bi acc)))
         D)))

(deftm batched-causal-sdpa
  "Causal SDPA over `batch` contiguous [seq,hd] head slabs: out = softmax_causal(Q·Kᵀ/√hd)·V.
  Materializes the weight matrix W (batched-causal-attn-weights — ONE QKᵀ dot per score,
  vs three recomputes in a streaming row kernel) and accumulates the output parallel over
  OUTPUT ELEMENTS (b, query row i, dim d): each work item owns out[b,i,d] (disjoint,
  n = b·seq·hd) and serially sums Σ_{j≤i} W[b,i,j]·V[b,j,d]."
  (All [T]
       [Q :- (Array T) K :- (Array T) V :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array T)
       (let [slab (* seq-len head-dim)
             ss (* seq-len seq-len)
             W (batched-causal-attn-weights Q K batch seq-len head-dim)
             out (alloc-like Q (* batch slab))]
         (raster.par/map-void! t (clojure.core/* batch slab)
                               (let [b (quot t slab)
                                     r (rem t slab)
                                     i (quot r head-dim)
                                     d (rem r head-dim)
                                     boff (clojure.core/* b slab)
                                     woff (clojure.core/+ (clojure.core/* b ss) (clojure.core/* i seq-len))
                                     acc (loop [j 0 acc 0.0]
                                           (if (<= j i)
                                             (recur (inc j)
                                                    (+ acc (* (aget W (clojure.core/+ woff j))
                                                              (aget V (clojure.core/+ boff (clojure.core/+ (clojure.core/* j head-dim) d))))))
                                             acc))]
                                 (aset out t acc)))
         out)))

;; ----------------------------------------------------------------
;; batched-causal-sdpa BACKWARD — flat resident par/map-void! kernels, one deftm
;; per gradient (dQ / dK / dV), each returning a SEPARATE (Array T) — NOT an
;; object-array bundle. Conventions match batched-causal-sdpa:
;;   score_ij = (Qb[i]·Kb[j]) / sqrt(head-dim)   (causal j≤i)
;;   w_ij     = softmax_j(score)                  (per query row i)
;;   dw_ij    = <dO_i, V_j>                        (unnormalized weight adjoint)
;;   D_i      = Σ_{j≤i} w_ij·dw_ij  (= <dO_i, out_i>)
;;   dS_ij    = w_ij·(dw_ij − D_i) / sqrt(head-dim)
;;   dQ_i = Σ_{j≤i} dS_ij·K_j     dK_j = Σ_{i≥j} dS_ij·Q_i     dV_j = Σ_{i≥j} w_ij·dO_i
;;
;; All three read the SHARED score-level intermediates (W / DW / D above) and then
;; accumulate with one work item per OUTPUT ELEMENT (b, row, dim d) — n = b·seq·hd
;; items, each an O(seq) serial sum over the causal partner rows. Every output element
;; keeps a single owner: no atomics, no host dotimes, no blit.
;; History (why this shape): the original dK/dV recomputed the row softmax per (b, kv
;; row) work item, repeating each query row's O(seq·hd) stats seq times — O(b·seq³·hd)
;; on only b·seq items (dK 307 ms + dV 181 ms = 79% of the backward at gemma dims).
;; Materializing W fixed those (dK 452→1.9 ms, dV 241→1.4 ms standalone). dQ kept the
;; per-(b,query-row) recompute shape — legal (each row's stats are used by exactly one
;; item) but only b·seq = 256 work items wide, which left it the single largest kernel
;; in the resident gemma VJP (10.9 ms of 38.8 ms). It now uses the same W/DW/D + output-
;; element shape, and — because the three intermediates are shared deftm CALLS — the
;; VJP's forward re-run and dQ/dK/dV all CSE onto ONE W, one DW, one D.
;; ----------------------------------------------------------------

(deftm batched-causal-sdpa-dq
  "dQ of batched-causal-sdpa. Reads the shared W / DW / D intermediates and accumulates
  parallel over OUTPUT ELEMENTS (batch b, query row i, dim d) — each work item owns
  dQ[b,i,d] (disjoint, n = b·seq·hd) and serially sums over the ≤i keys:
  dQ[i,d] = (1/√hd)·Σ_{j≤i} W[b,i,j]·(DW[b,i,j] − D[b,i])·K[j,d]."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array T)
       (let [slab (* seq-len head-dim)
             ss (* seq-len seq-len)
             W (batched-causal-attn-weights Q K batch seq-len head-dim)
             DW (batched-causal-attn-dw d-out V batch seq-len head-dim)
             D (batched-causal-attn-dsum W DW batch seq-len)
             dQ (alloc-like Q (* batch slab))]
         (raster.par/map-void! t (clojure.core/* batch slab)
                               (let [b (quot t slab)
                                     r (rem t slab)
                                     i (quot r head-dim)
                                     d (rem r head-dim)
                                     boff (clojure.core/* b slab)
                                     woff (clojure.core/+ (clojure.core/* b ss) (clojure.core/* i seq-len))
                                     di (aget D (clojure.core/+ (clojure.core/* b seq-len) i))
                                     acc (loop [j 0 acc 0.0]
                                           (if (<= j i)
                                             (let [wij (aget W (clojure.core/+ woff j))
                                                   dwij (aget DW (clojure.core/+ woff j))]
                                               (recur (inc j)
                                                      (+ acc (* (* wij (- dwij di))
                                                                (aget K (clojure.core/+ boff (clojure.core/+ (clojure.core/* j head-dim) d)))))))
                                             acc))]
                                 (aset dQ t (/ acc (n/oftype acc (n/sqrt (double head-dim)))))))
         dQ)))

(deftm batched-causal-sdpa-dv
  "dV of batched-causal-sdpa. Reads the shared weight matrix W and accumulates parallel
  over OUTPUT ELEMENTS (batch b, key row j, dim d) — each work item owns dV[b,j,d]
  (disjoint, n = b·seq·hd) and serially sums dV[j,d] = Σ_{i≥j} W[b,i,j]·dO[i,d] over
  the ≥j queries (causal)."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array T)
       (let [slab (* seq-len head-dim)
             ss (* seq-len seq-len)
             W (batched-causal-attn-weights Q K batch seq-len head-dim)
             dV (alloc-like Q (* batch slab))]
         (raster.par/map-void! t (clojure.core/* batch slab)
                               (let [b (quot t slab)
                                     r (rem t slab)
                                     j (quot r head-dim)
                                     d (rem r head-dim)
                                     boff (clojure.core/* b slab)
                                     woff (clojure.core/+ (clojure.core/* b ss) j)
                                     acc (loop [i j acc 0.0]
                                           (if (< i seq-len)
                                             (recur (inc i)
                                                    (+ acc (* (aget W (clojure.core/+ woff (clojure.core/* i seq-len)))
                                                              (aget d-out (clojure.core/+ boff (clojure.core/+ (clojure.core/* i head-dim) d))))))
                                             acc))]
                                 (aset dV t acc)))
         dV)))

(deftm batched-causal-sdpa-dk
  "dK of batched-causal-sdpa. Reads the shared W / DW / D intermediates and accumulates
  parallel over OUTPUT ELEMENTS (batch b, key row j, dim d) — each work item owns
  dK[b,j,d] (disjoint, n = b·seq·hd) and serially sums over the ≥j queries:
  dK[j,d] = (1/√hd)·Σ_{i≥j} W[b,i,j]·(DW[b,i,j] − D[b,i])·Q[i,d]."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        batch :- Long seq-len :- Long head-dim :- Long]
       :- (Array T)
       (let [slab (* seq-len head-dim)
             ss (* seq-len seq-len)
             W (batched-causal-attn-weights Q K batch seq-len head-dim)
             DW (batched-causal-attn-dw d-out V batch seq-len head-dim)
             D (batched-causal-attn-dsum W DW batch seq-len)
             dK (alloc-like Q (* batch slab))]
         (raster.par/map-void! t (clojure.core/* batch slab)
                               (let [b (quot t slab)
                                     r (rem t slab)
                                     j (quot r head-dim)
                                     d (rem r head-dim)
                                     boff (clojure.core/* b slab)
                                     woff (clojure.core/+ (clojure.core/* b ss) j)
                                     doff (clojure.core/* b seq-len)
                                     acc (loop [i j acc 0.0]
                                           (if (< i seq-len)
                                             (let [wij (aget W (clojure.core/+ woff (clojure.core/* i seq-len)))
                                                   dwij (aget DW (clojure.core/+ woff (clojure.core/* i seq-len)))
                                                   di (aget D (clojure.core/+ doff i))]
                                               (recur (inc i)
                                                      (+ acc (* (* wij (- dwij di))
                                                                (aget Q (clojure.core/+ boff (clojure.core/+ (clojure.core/* i head-dim) d)))))))
                                             acc))]
                                 (aset dK t (/ acc (n/oftype acc (n/sqrt (double head-dim)))))))
         dK)))

(tmpl/merge-into-template! 'raster.dl.attention/batched-causal-sdpa
                           {:params '[Q K V batch seq-len head-dim]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [Q K V batch seq-len head-dim]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [dQ (gensym-fn "dQ" (tmpl/grad-tag Q))
                                    dK (gensym-fn "dK" (tmpl/grad-tag K))
                                    dV (gensym-fn "dV" (tmpl/grad-tag V))]
                                [(update ctx :bindings into
                                         [dQ (list 'raster.dl.attention/batched-causal-sdpa-dq
                                                   adjoint-sym Q K V batch seq-len head-dim)
                                          dK (list 'raster.dl.attention/batched-causal-sdpa-dk
                                                   adjoint-sym Q K V batch seq-len head-dim)
                                          dV (list 'raster.dl.attention/batched-causal-sdpa-dv
                                                   adjoint-sym Q K V batch seq-len head-dim)])
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

;; Backward for scaled-dot-product-attn (bidirectional, non-causal) — THREE flat
;; resident par/map-void! kernels, one per gradient (dQ/dK/dV), each a SEPARATE
;; (Array T) (NOT an object-array bundle → NO lost float tag). Same recompute-the-
;; row-softmax structure as the causal kernels, but the j-sums run over ALL seq-k
;; (no j≤i mask) and the query loops over all seq-q. Q/K feature dim = dk, V/out = dv.
;;   score_ij = (Q_i·K_j)/√dk   w_ij = softmax_j(score)   dw_ij = ⟨dO_i,V_j⟩
;;   D_i = Σ_j w_ij·dw_ij   dQ_i = (1/√dk)·Σ_j w_ij·(dw_ij−D_i)·K_j
;;   dK_j = (1/√dk)·Σ_i w_ij·(dw_ij−D_i)·Q_i   dV_j = Σ_i w_ij·dO_i
;; T-typing discipline identical to the causal kernels (reduction seeds are bare
;; floating literals; the 1/√dk scale is a DIVISION by an (n/oftype …)-typed scalar).

(deftm scaled-dot-product-attn-dq
  "dQ of scaled-dot-product-attn. Parallel over query row i (disjoint)."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        seq-q :- Long seq-k :- Long dk :- Long dv :- Long]
       :- (Array T)
       (let [dQ (alloc-like Q (* seq-q dk))]
         (raster.par/map-void! i seq-q
                               (let [qrow (clojure.core/* i dk)
                                     orow (clojure.core/* i dv)
                                     mx (loop [j 0 mm -1.0e38]
                                          (if (< j seq-k)
                                            (let [krow (clojure.core/* j dk)
                                                  dot (loop [d 0 acc 0.0]
                                                        (if (< d dk)
                                                          (recur (inc d)
                                                                 (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                           (aget K (clojure.core/+ krow d)))))
                                                          acc))]
                                              (recur (inc j) (n/max mm (/ dot (n/oftype dot (n/sqrt (double dk)))))))
                                            mm))
                                     sum (loop [j 0 s 0.0]
                                           (if (< j seq-k)
                                             (let [krow (clojure.core/* j dk)
                                                   dot (loop [d 0 acc 0.0]
                                                         (if (< d dk)
                                                           (recur (inc d)
                                                                  (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                            (aget K (clojure.core/+ krow d)))))
                                                           acc))]
                                               (recur (inc j) (+ s (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)))))
                                             s))
                                     inv (/ 1.0 sum)
                                     Di (loop [j 0 acc 0.0]
                                          (if (< j seq-k)
                                            (let [krow (clojure.core/* j dk)
                                                  vrow (clojure.core/* j dv)
                                                  dot (loop [d 0 a 0.0]
                                                        (if (< d dk)
                                                          (recur (inc d)
                                                                 (+ a (* (aget Q (clojure.core/+ qrow d))
                                                                         (aget K (clojure.core/+ krow d)))))
                                                          a))
                                                  w  (* (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)) inv)
                                                  dw (loop [d 0 a 0.0]
                                                       (if (< d dv)
                                                         (recur (inc d)
                                                                (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                        (aget V (clojure.core/+ vrow d)))))
                                                         a))]
                                              (recur (inc j) (+ acc (* w dw))))
                                            acc))]
                                 (loop [d 0]
                                   (if (< d dk)
                                     (do (aset dQ (clojure.core/+ qrow d) 0.0) (recur (inc d)))
                                     nil))
                                 (loop [j 0]
                                   (if (< j seq-k)
                                     (let [krow (clojure.core/* j dk)
                                           vrow (clojure.core/* j dv)
                                           dot (loop [d 0 a 0.0]
                                                 (if (< d dk)
                                                   (recur (inc d)
                                                          (+ a (* (aget Q (clojure.core/+ qrow d))
                                                                  (aget K (clojure.core/+ krow d)))))
                                                   a))
                                           w  (* (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)) inv)
                                           dw (loop [d 0 a 0.0]
                                                (if (< d dv)
                                                  (recur (inc d)
                                                         (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                 (aget V (clojure.core/+ vrow d)))))
                                                  a))
                                           c  (* w (- dw Di))]
                                       (loop [d 0]
                                         (if (< d dk)
                                           (do (aset dQ (clojure.core/+ qrow d)
                                                     (+ (aget dQ (clojure.core/+ qrow d))
                                                        (* c (aget K (clojure.core/+ krow d)))))
                                               (recur (inc d)))
                                           nil))
                                       (recur (inc j)))
                                     nil))
                                 (loop [d 0]
                                   (if (< d dk)
                                     (let [v (aget dQ (clojure.core/+ qrow d))]
                                       (aset dQ (clojure.core/+ qrow d) (/ v (n/oftype v (n/sqrt (double dk)))))
                                       (recur (inc d)))
                                     nil))))
         dQ)))

(deftm scaled-dot-product-attn-dv
  "dV of scaled-dot-product-attn. Parallel over key row j (disjoint); loops all
  queries i and accumulates dV[j,d] = Σ_i w_ij·dO[i,d]."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        seq-q :- Long seq-k :- Long dk :- Long dv :- Long]
       :- (Array T)
       (let [dV (alloc-like Q (* seq-k dv))]
         (raster.par/map-void! j seq-k
                               (let [jrowk (clojure.core/* j dk)
                                     jrowv (clojure.core/* j dv)]
                                 (loop [d 0]
                                   (if (< d dv)
                                     (do (aset dV (clojure.core/+ jrowv d) 0.0) (recur (inc d)))
                                     nil))
                                 (loop [i 0]
                                   (if (< i seq-q)
                                     (let [qrow (clojure.core/* i dk)
                                           orow (clojure.core/* i dv)
                                           mx (loop [k 0 mm -1.0e38]
                                                (if (< k seq-k)
                                                  (let [krow (clojure.core/* k dk)
                                                        dot (loop [d 0 acc 0.0]
                                                              (if (< d dk)
                                                                (recur (inc d)
                                                                       (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                 (aget K (clojure.core/+ krow d)))))
                                                                acc))]
                                                    (recur (inc k) (n/max mm (/ dot (n/oftype dot (n/sqrt (double dk)))))))
                                                  mm))
                                           sum (loop [k 0 s 0.0]
                                                 (if (< k seq-k)
                                                   (let [krow (clojure.core/* k dk)
                                                         dot (loop [d 0 acc 0.0]
                                                               (if (< d dk)
                                                                 (recur (inc d)
                                                                        (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                  (aget K (clojure.core/+ krow d)))))
                                                                 acc))]
                                                     (recur (inc k) (+ s (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)))))
                                                   s))
                                           inv (/ 1.0 sum)
                                           dotij (loop [d 0 acc 0.0]
                                                   (if (< d dk)
                                                     (recur (inc d)
                                                            (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                      (aget K (clojure.core/+ jrowk d)))))
                                                     acc))
                                           w (* (m/exp (- (/ dotij (n/oftype dotij (n/sqrt (double dk)))) mx)) inv)]
                                       (loop [d 0]
                                         (if (< d dv)
                                           (do (aset dV (clojure.core/+ jrowv d)
                                                     (+ (aget dV (clojure.core/+ jrowv d))
                                                        (* w (aget d-out (clojure.core/+ orow d)))))
                                               (recur (inc d)))
                                           nil))
                                       (recur (inc i)))
                                     nil))))
         dV)))

(deftm scaled-dot-product-attn-dk
  "dK of scaled-dot-product-attn. Parallel over key row j (disjoint); loops all
  queries i, recomputing query i's softmax + D_i, and accumulates
  dK[j,d] = (1/√dk)·Σ_i w_ij·(⟨dO_i,V_j⟩ − D_i)·Q[i,d]."
  (All [T]
       [d-out :- (Array T) Q :- (Array T) K :- (Array T) V :- (Array T)
        seq-q :- Long seq-k :- Long dk :- Long dv :- Long]
       :- (Array T)
       (let [dK (alloc-like Q (* seq-k dk))]
         (raster.par/map-void! j seq-k
                               (let [jrowk (clojure.core/* j dk)
                                     jrowv (clojure.core/* j dv)]
                                 (loop [d 0]
                                   (if (< d dk)
                                     (do (aset dK (clojure.core/+ jrowk d) 0.0) (recur (inc d)))
                                     nil))
                                 (loop [i 0]
                                   (if (< i seq-q)
                                     (let [qrow (clojure.core/* i dk)
                                           orow (clojure.core/* i dv)
                                           mx (loop [k 0 mm -1.0e38]
                                                (if (< k seq-k)
                                                  (let [krow (clojure.core/* k dk)
                                                        dot (loop [d 0 acc 0.0]
                                                              (if (< d dk)
                                                                (recur (inc d)
                                                                       (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                 (aget K (clojure.core/+ krow d)))))
                                                                acc))]
                                                    (recur (inc k) (n/max mm (/ dot (n/oftype dot (n/sqrt (double dk)))))))
                                                  mm))
                                           sum (loop [k 0 s 0.0]
                                                 (if (< k seq-k)
                                                   (let [krow (clojure.core/* k dk)
                                                         dot (loop [d 0 acc 0.0]
                                                               (if (< d dk)
                                                                 (recur (inc d)
                                                                        (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                                  (aget K (clojure.core/+ krow d)))))
                                                                 acc))]
                                                     (recur (inc k) (+ s (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)))))
                                                   s))
                                           inv (/ 1.0 sum)
                                           Di (loop [k 0 acc 0.0]
                                                (if (< k seq-k)
                                                  (let [krow (clojure.core/* k dk)
                                                        vrow (clojure.core/* k dv)
                                                        dot (loop [d 0 a 0.0]
                                                              (if (< d dk)
                                                                (recur (inc d)
                                                                       (+ a (* (aget Q (clojure.core/+ qrow d))
                                                                               (aget K (clojure.core/+ krow d)))))
                                                                a))
                                                        w  (* (m/exp (- (/ dot (n/oftype dot (n/sqrt (double dk)))) mx)) inv)
                                                        dw (loop [d 0 a 0.0]
                                                             (if (< d dv)
                                                               (recur (inc d)
                                                                      (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                              (aget V (clojure.core/+ vrow d)))))
                                                               a))]
                                                    (recur (inc k) (+ acc (* w dw))))
                                                  acc))
                                           dotij (loop [d 0 acc 0.0]
                                                   (if (< d dk)
                                                     (recur (inc d)
                                                            (+ acc (* (aget Q (clojure.core/+ qrow d))
                                                                      (aget K (clojure.core/+ jrowk d)))))
                                                     acc))
                                           wij (* (m/exp (- (/ dotij (n/oftype dotij (n/sqrt (double dk)))) mx)) inv)
                                           dwij (loop [d 0 a 0.0]
                                                  (if (< d dv)
                                                    (recur (inc d)
                                                           (+ a (* (aget d-out (clojure.core/+ orow d))
                                                                   (aget V (clojure.core/+ jrowv d)))))
                                                    a))
                                           c (* wij (- dwij Di))]
                                       (loop [d 0]
                                         (if (< d dk)
                                           (do (aset dK (clojure.core/+ jrowk d)
                                                     (+ (aget dK (clojure.core/+ jrowk d))
                                                        (* c (aget Q (clojure.core/+ qrow d)))))
                                               (recur (inc d)))
                                           nil))
                                       (recur (inc i)))
                                     nil))
                                 (loop [d 0]
                                   (if (< d dk)
                                     (let [v (aget dK (clojure.core/+ jrowk d))]
                                       (aset dK (clojure.core/+ jrowk d) (/ v (n/oftype v (n/sqrt (double dk)))))
                                       (recur (inc d)))
                                     nil))))
         dK)))

;; ================================================================
;; Template registration for scaled-dot-product attention
;; ================================================================

(tmpl/merge-into-template! 'raster.dl.attention/scaled-dot-product-attn
                           {:params '[Q K V seq-q seq-k dk dv] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [Q K V seq-q seq-k dk dv] _result-sym adjoint-sym gensym-fn]
                                        (let [dQ (gensym-fn "dQ" (tmpl/grad-tag Q))
                                              dK (gensym-fn "dK" (tmpl/grad-tag K))
                                              dV (gensym-fn "dV" (tmpl/grad-tag V))]
                                          [(update ctx :bindings into
                                                   [dQ (list 'raster.dl.attention/scaled-dot-product-attn-dq
                                                             adjoint-sym Q K V seq-q seq-k dk dv)
                                                    dK (list 'raster.dl.attention/scaled-dot-product-attn-dk
                                                             adjoint-sym Q K V seq-q seq-k dk dv)
                                                    dV (list 'raster.dl.attention/scaled-dot-product-attn-dv
                                                             adjoint-sym Q K V seq-q seq-k dk dv)])
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

;; Sliding-window scores (moonshine-style 'ergodic' encoder): query i attends j
;; iff 0 <= i-j <= left-1 (past incl. self) or 0 < j-i <= right-1 (future).
;; left >= 1 required; right = 0 and right = 1 both mean "no future" (the
;; diagonal always attends — same as moonshine's j1 = i + max(0, right-1)).
;; Degenerate windows recover the siblings exactly: [nrows, 1] == the causal
;; kernel, [nrows, nrows] == the bidir kernel; per-block calls with
;; left = right = block-size give block-diagonal attention. Out-of-window
;; scores get the same -1.0e30 sentinel as the causal kernel, so
;; attn-prefill-softmax!/attn-prefill-out! compose unchanged: exp(-1e30 - mx)
;; underflows to exactly 0.0 and masked lanes contribute exact zeros.
(deftm attn-prefill-scores-windowed! (All [T] [q :- (Array T) k :- (Array T) sc :- (Array T)
                                               nrows :- Long n-q :- Long group :- Long
                                               n-kv :- Long head-dim :- Long
                                               scale :- Double left :- Long right :- Long] :- Void
                                          (raster.par/map-void! idx (clojure.core/* nrows (clojure.core/* n-q nrows))
                                                                (let [per-i (clojure.core/* n-q nrows)
                                                                      i (quot idx per-i)
                                                                      rest0 (rem idx per-i)
                                                                      hq (quot rest0 nrows)
                                                                      j (rem rest0 nrows)
                                                                      row (clojure.core/+ (clojure.core/* i n-q) hq)]
                                                                  (if (or (> (clojure.core/- i j) (dec left))
                                                                          (and (< i j) (> (clojure.core/- j i) (dec right))))
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
