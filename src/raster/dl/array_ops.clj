(ns raster.dl.array-ops
  "AD-differentiable array primitive operations for graph neural networks.

  Generic indexed-array primitives that can be composed to build any
  graph neural network architecture. Only these primitives need hand-written
  rrules — compositions auto-differentiate through them.

  Core primitives:
    scatter-add      - scatter values to indexed destinations (adjoint of gather)
    gather           - gather values from indexed sources (adjoint of scatter-add)
    indexed-dot      - batched indexed dot product (multi-head aware)
    scatter-mul-add  - weighted scatter (multi-head aware)
    scale-clamp-exp  - pointwise scale+clamp+exp (attention nonlinearity)
    reduce-axis      - sum-reduce first axis
    segment-div      - divide array by per-segment scalars (normalization)

  Utility ops:
    array-add        - elementwise addition
    array-copy       - array copy utility
    broadcast-add    - h[b*dim+d] + t[d] broadcast

  Domain ops (kept for complex patterns):
    flat-embed-op    - value + space embedding
    dot-rows         - per-row dot product (unembed)
    masked-mse-loss  - MSE on non-observed variables"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm broadcast]]
            [raster.arrays :refer [aget aset alength aclone acopy! alloc-like]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; array-add: out[i] = a[i] + b[i]
;; Exists as a named primitive (rather than using generic +) because:
;; 1. The AD tape needs an explicit op to attach an rrule for skip connections
;; 2. Used by gsdm.clj ResnetBlock and graph attention residual paths
;; 3. Has both a closure rrule and a colocated flat AD template
;; ================================================================

(deftm array-add (All [T] [a :- (Array T) b :- (Array T) n :- Long] :- (Array T)
                      (broadcast [a b] (n/+ a b))))

(deftm array-add-backward
  "Backward for array-add: d_a = dy, d_b = dy (both copies). Expressed as a pure
  `broadcast` copy (the same SOAC the forward uses) so it lowers to a resident
  :map kernel on GPU and stays SIMD-vectorizable on CPU — see residual-add-backward."
  (All [T] [dy :- (Array T) n :- Long] :- (Array T)
       (broadcast [dy] dy)))

;; ================================================================
;; array-copy: utility for gradient copying
;; ================================================================

(deftm array-copy
  [src :- (Array double) n :- Long]
  :- (Array double)
  (let [out (double-array n)]
    (acopy! src 0 out 0 n)
    out))

;; ================================================================
;; broadcast-add: out[b*dim+d] = h[b*dim+d] + t[d]
;; ================================================================

(deftm broadcast-add
  [h :- (Array double) t :- (Array double)
   batch :- Long dim :- Long]
  :- (Array double)
  (let [n (* batch dim)
        out (double-array n)]
    (dotimes [b batch]
      (dotimes [d dim]
        (let [idx (+ (* b (int dim)) d)]
          (aset out idx
                (+ (aget h idx)
                   (aget t d))))))
    out))

(deftm broadcast-add-dh
  "Backward for h: d_h = copy(dy)."
  [dy :- (Array double) n :- Long]
  :- (Array double)
  (let [out (double-array n)]
    (acopy! dy 0 out 0 n)
    out))

(deftm broadcast-add-dt
  "Backward for t: d_t[d] = sum_b(dy[b*dim+d])."
  [dy :- (Array double) batch :- Long dim :- Long]
  :- (Array double)
  (let [out (double-array dim)]
    (dotimes [b batch]
      (dotimes [d dim]
        (let [idx (+ (* b (int dim)) d)]
          (aset out d
                (+ (aget out d)
                   (aget dy idx))))))
    out))

;; ================================================================
;; GENERIC PRIMITIVES
;; ================================================================

;; ================================================================
;; slice-strided-2d / scatter-strided-2d
;;
;; Conceptually src is a 2D matrix [rows × row-stride]; we read or write a
;; packed [rows × n-cols] sub-block at column offset col-offset. Used for
;; multi-head attention's head split (slice) and head concat (scatter).
;;
;; Forward:  out[r*n-cols + c]               = src   [r*row-stride + col-offset + c]
;; Forward:  out[r*row-stride + col-offset + c] = packed[r*n-cols + c]
;;
;; Both ops are pure (allocate fresh dst); they're each other's pullbacks
;; (slice ↔ scatter), so the gradient code is just the same op in the dual
;; direction.
;; ================================================================

(deftm slice-strided-2d
  "Read a packed [rows × n-cols] block from src starting at column col-offset
  with row stride row-stride. Returns a freshly allocated array of length
  (rows * n-cols)."
  (All [T] [src :- (Array T)
            rows :- Long row-stride :- Long col-offset :- Long n-cols :- Long]
       :- (Array T)
       (let [out (alloc-like src (* rows n-cols))]
         (dotimes [r rows]
           (let [src-off (+ (* r (int row-stride)) (int col-offset))
                 dst-off (* r (int n-cols))]
             (dotimes [c n-cols]
               (aset out (+ dst-off c)
                     (aget src (+ src-off c))))))
         out)))

(deftm scatter-strided-2d
  "Write a packed [rows × n-cols] block into a freshly allocated
  [rows × row-stride] dst at column offset col-offset, with the rest left as
  zero. Returns a freshly allocated array of length (rows * row-stride)."
  (All [T] [packed :- (Array T)
            rows :- Long row-stride :- Long col-offset :- Long n-cols :- Long]
       :- (Array T)
       (let [out (alloc-like packed (* rows row-stride))]
         (dotimes [r rows]
           (let [src-off (* r (int n-cols))
                 dst-off (+ (* r (int row-stride)) (int col-offset))]
             (dotimes [c n-cols]
               (aset out (+ dst-off c)
                     (aget packed (+ src-off c))))))
         out)))

;; ================================================================
;; pack-heads / unpack-heads: multi-head attention layout combinators.
;; Both are parametric (All [T]) and expressed as a resident strided copy:
;; a `par/map-void!` over the n-heads*seq-len output BLOCKS (one work-item per
;; [head, seq] block), each copying `head-dim` contiguous elements from the
;; permuted source offset. Index math is clojure.core integer arithmetic
;; (quot/rem/*/+ — the flat block coordinate → source/dest offsets), so it
;; lowers in-kernel; the inner contiguous copy vectorizes on CPU and becomes a
;; resident :map-void OpenCL kernel on GPU (same shape as par/gather /
;; kv-append! / the decode kernels). No index table, no atomics — a pure
;; permutation. pack-heads and unpack-heads are exact duals (the AD templates
;; below reference each other by name, so the resident rewrite leaves them
;; untouched).
;; ================================================================

(deftm pack-heads
  "Reshape [seq-len, n-heads*head-dim] (row-major) → [n-heads, seq-len, head-dim]
  so each head slab is contiguous: out[h*seq*hd + s*hd + d] = x[s*(nh*hd) + h*hd + d].
  Resident strided copy: work-item e = h*seq-len + s copies a head-dim slab."
  (All [T] [x :- (Array T) seq-len :- Long n-heads :- Long head-dim :- Long] :- (Array T)
       (let [out (alloc-like x (* seq-len n-heads head-dim))]
         (par/map-void! e (clojure.core/* n-heads seq-len)
                        (let [h (quot e seq-len)
                              s (rem e seq-len)
                              dst (clojure.core/* e head-dim)
                              src (clojure.core/+ (clojure.core/* s (clojure.core/* n-heads head-dim))
                                                  (clojure.core/* h head-dim))]
                          (loop [d 0]
                            (if (< d head-dim)
                              (do (aset out (clojure.core/+ dst d) (aget x (clojure.core/+ src d)))
                                  (recur (inc d)))
                              nil))))
         out)))

(deftm unpack-heads
  "Dual of pack-heads: [n-heads, seq-len, head-dim] → [seq-len, n-heads*head-dim]:
  out[s*(nh*hd) + h*hd + d] = x[h*seq*hd + s*hd + d].
  Resident strided copy: work-item e = s*n-heads + h copies a head-dim slab."
  (All [T] [x :- (Array T) seq-len :- Long n-heads :- Long head-dim :- Long] :- (Array T)
       (let [out (alloc-like x (* seq-len n-heads head-dim))]
         (par/map-void! e (clojure.core/* seq-len n-heads)
                        (let [s (quot e n-heads)
                              h (rem e n-heads)
                              dst (clojure.core/* e head-dim)
                              src (clojure.core/* (clojure.core/+ (clojure.core/* h seq-len) s) head-dim)]
                          (loop [d 0]
                            (if (< d head-dim)
                              (do (aset out (clojure.core/+ dst d) (aget x (clojure.core/+ src d)))
                                  (recur (inc d)))
                              nil))))
         out)))

(deftm blit-slab!
  "Copy `len` contiguous elements from `src[0..len)` into `dst[dst-off .. dst-off+len)`.
  Both are typed (Array T), so the element aget/aset devirtualize under compile-aot.
  Used to scatter object-array-bundled per-batch gradients — extracted from an Object[]
  via clojure.core/aget, hence UNTYPED (compile-time Object) — back into a strided output
  WITHOUT an Object[]-element re-index. A `(raster.arrays/aget <Object> i)` would
  mis-devirtualize to the Object overload and ClassCast a double[]/float[] at runtime;
  routing the untyped source through this (Array T) param makes compile-aot insert the
  checkcast to the monomorphized element array. Returns dst."
  (All [T] [dst :- (Array T) dst-off :- Long src :- (Array T) len :- Long] :- (Array T)
       (do (dotimes [i len]
             (aset dst (+ dst-off i) (aget src i)))
           dst)))

;; pack ↔ unpack are duals (both are non-diff on the Long shape params). Each has
;; a pullback-factory (interpreted AD) AND a flat grads-fn (compile-aot / resident
;; codegen): pack's gradient is unpack of the adjoint, and vice versa — a single
;; flat let-binding so the reverse pass stays straight-line (no fn* tape closure).
(tmpl/merge-into-template! 'raster.dl.array-ops/pack-heads
                           {:pullback-factory
                            (fn [_result _x seq-len n-heads head-dim]
                              (fn [d-out]
                                [(unpack-heads d-out seq-len n-heads head-dim) nil nil nil]))
                            :params '[x seq-len n-heads head-dim]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [x seq-len n-heads head-dim]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [dx (gensym-fn "d_x" (tmpl/grad-tag x))]
                                [(update ctx :bindings into
                                         [dx (list 'raster.dl.array-ops/unpack-heads
                                                   adjoint-sym seq-len n-heads head-dim)])
                                 [dx nil nil nil]]))})
(tmpl/merge-into-template! 'raster.dl.array-ops/unpack-heads
                           {:pullback-factory
                            (fn [_result _x seq-len n-heads head-dim]
                              (fn [d-out]
                                [(pack-heads d-out seq-len n-heads head-dim) nil nil nil]))
                            :params '[x seq-len n-heads head-dim]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [x seq-len n-heads head-dim]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [dx (gensym-fn "d_x" (tmpl/grad-tag x))]
                                [(update ctx :bindings into
                                         [dx (list 'raster.dl.array-ops/pack-heads
                                                   adjoint-sym seq-len n-heads head-dim)])
                                 [dx nil nil nil]]))})

;; ================================================================
;; broadcast-kv-heads / sum-kv-heads: GQA/MQA key-value head fan-out and its
;; dual (fan-in / segment reduce over the query-head group).
;;
;; A grouped/multi-query decoder shares one KV head across `group` query heads
;; (group = n_q / n_kv). Given a packed [n_kv, slab] KV tensor (slab = seq·hd),
;; broadcast-kv-heads REPEATS each kv head `group` times to align with the
;; n_q = n_kv·group query heads:  out[(g·group+r)·slab + i] = src[g·slab + i].
;; Its exact dual sum-kv-heads SUMS the `group` query-head contributions back
;; onto each kv head — this is the MQA dK/dV fan-in gradient. Both are (All [T])
;; and each other's pullback (flat grads-fn, no index array, no loop-carry), so
;; the reverse pass stays straight-line and GPU-lowerable (fan-out = a resident
;; broadcast/map, fan-in = a resident segment reduce).
;; ================================================================

(deftm broadcast-kv-heads
  "Repeat each of n-kv contiguous [slab] KV-head blocks `group` times:
  out[(g*group+r)*slab + i] = src[g*slab + i], for g<n-kv, r<group, i<slab.
  Output length = n-kv*group*slab. Used to align KV heads with query heads in GQA/MQA.
  Resident broadcast: `par/map-void!` over the n-kv*group output blocks (work-item
  e = g*group+r), each copying the slab of source kv-head g = e/group. In-kernel
  integer index math (clojure.core) → resident :map-void kernel; a pure fan-out
  copy (no atomics)."
  (All [T] [src :- (Array T) n-kv :- Long group :- Long slab :- Long] :- (Array T)
       (let [out (alloc-like src (* n-kv group slab))]
         (par/map-void! e (clojure.core/* n-kv group)
                        (let [g (quot e group)
                              dst (clojure.core/* e slab)
                              soff (clojure.core/* g slab)]
                          (loop [i 0]
                            (if (< i slab)
                              (do (aset out (clojure.core/+ dst i) (aget src (clojure.core/+ soff i)))
                                  (recur (inc i)))
                              nil))))
         out)))

(deftm sum-kv-heads
  "Dual of broadcast-kv-heads: sum each group of `group` query-head blocks back
  onto its kv head. out[g*slab + i] = Σ_{r<group} src[(g*group+r)*slab + i].
  Output length = n-kv*slab. This is the GQA/MQA dK/dV fan-in accumulation.
  Resident fan-in as an OUTPUT-PARALLEL segment reduce: `par/map-void!` over the
  n-kv*slab output elements (work-item o = g*slab+i), each summing its `group`
  contributing source elements with an in-kernel loop. Because the reduction is
  per output element (disjoint writes) it needs NO atomics — a resident :map-void
  kernel. The accumulator is seeded with the r=0 element (a T-typed aget, not a
  Double literal) so the sum stays polymorphic; n/+ adds T+T (devirtualizes on
  both CPU and GPU)."
  (All [T] [src :- (Array T) n-kv :- Long group :- Long slab :- Long] :- (Array T)
       (let [out (alloc-like src (* n-kv slab))]
         (par/map-void! o (clojure.core/* n-kv slab)
                        (let [g (quot o slab)
                              i (rem o slab)
                              gg (clojure.core/* g group)]
                          (aset out o
                                (loop [r 1
                                       acc (aget src (clojure.core/+ (clojure.core/* gg slab) i))]
                                  (if (< r group)
                                    (recur (inc r)
                                           (n/+ acc (aget src (clojure.core/+ (clojure.core/* (clojure.core/+ gg r) slab) i))))
                                    acc)))))
         out)))

;; broadcast-kv-heads ↔ sum-kv-heads duals. Flat grads-fn (each other) + a
;; pullback-factory for interpreted AD. Long shape params are non-diff (nil grads).
(tmpl/merge-into-template! 'raster.dl.array-ops/broadcast-kv-heads
                           {:pullback-factory
                            (fn [_result _src n-kv group slab]
                              (fn [d-out]
                                [(sum-kv-heads d-out n-kv group slab) nil nil nil]))
                            :params '[src n-kv group slab] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [src n-kv group slab] _result-sym adjoint-sym gensym-fn]
                              (let [d-src (gensym-fn "d_src" (tmpl/grad-tag src))]
                                [(update ctx :bindings into
                                         [d-src (list 'raster.dl.array-ops/sum-kv-heads
                                                      adjoint-sym n-kv group slab)])
                                 [d-src nil nil nil]]))})
(tmpl/merge-into-template! 'raster.dl.array-ops/sum-kv-heads
                           {:pullback-factory
                            (fn [_result _src n-kv group slab]
                              (fn [d-out]
                                [(broadcast-kv-heads d-out n-kv group slab) nil nil nil]))
                            :params '[src n-kv group slab] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [src n-kv group slab] _result-sym adjoint-sym gensym-fn]
                              (let [d-src (gensym-fn "d_src" (tmpl/grad-tag src))]
                                [(update ctx :bindings into
                                         [d-src (list 'raster.dl.array-ops/broadcast-kv-heads
                                                      adjoint-sym n-kv group slab)])
                                 [d-src nil nil nil]]))})

;; slice ↔ scatter are duals: gradient of slice goes through scatter, and
;; vice versa. A grads-fn (compile-aot flat codegen) is registered. Indices
;; passed back as nil since they're non-differentiable Long params.

(tmpl/merge-into-template! 'raster.dl.array-ops/slice-strided-2d
                           {:params '[src rows row-stride col-offset n-cols]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [src rows row-stride col-offset n-cols]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-src (gensym-fn "d_src" (tmpl/grad-tag src))]
                                [(update ctx :bindings into
                                         [d-src (list 'raster.dl.array-ops/scatter-strided-2d
                                                      adjoint-sym rows row-stride col-offset n-cols)])
                                 [d-src nil nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/scatter-strided-2d
                           {:params '[packed rows row-stride col-offset n-cols]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [packed rows row-stride col-offset n-cols]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-packed (gensym-fn "d_packed" (tmpl/grad-tag packed))]
                                [(update ctx :bindings into
                                         [d-packed (list 'raster.dl.array-ops/slice-strided-2d
                                                         adjoint-sym rows row-stride col-offset n-cols)])
                                 [d-packed nil nil nil nil]]))})

;; ================================================================
;; gather: out[e*stride+s] = src[indices[e]*stride+s]
;; Adjoint of scatter-add.
;; ================================================================

(deftm gather
  "Gather values from indexed positions.
  out[e*stride+s] = src[indices[e]*stride+s] for e∈[0,n-pairs), s∈[0,stride).
  n-src: number of source entities (for backward scatter allocation).

  Expressed over the par/gather primitive so it lowers to a resident GPU gather
  kernel (and stays a plain vectorizable loop on CPU). (All [T]): T=double on CPU,
  T=float on GPU — no hard-typed precision."
  (All [T] [src :- (Array T) indices :- (Array long)
            n-src :- Long n-pairs :- Long stride :- Long]
       :- (Array T)
       (let [out (alloc-like src (* n-pairs stride))]
         (par/gather out src indices n-pairs stride)
         out)))

;; ================================================================
;; scatter-add: out[indices[e]*stride+s] += vals[e*stride+s]
;; Adjoint of gather.
;; ================================================================

(deftm scatter-add
  "Scatter-add values to indexed destinations.
  out[indices[e]*stride+s] += vals[e*stride+s] for e∈[0,n-pairs), s∈[0,stride).
  n-dst: number of destination entities (output size = n-dst*stride).

  Expressed over the par/scatter! primitive so it lowers to a resident GPU
  scatter-add kernel (atomic += for concurrent fan-in) and stays a plain loop on
  CPU. The output accumulator is alloc-like (zero-initialized). (All [T]):
  T=double on CPU, T=float on GPU — no hard-typed precision."
  (All [T] [vals :- (Array T) indices :- (Array long)
            n-dst :- Long n-pairs :- Long stride :- Long]
       :- (Array T)
       (let [out (alloc-like vals (* n-dst stride))]
         (par/scatter! out vals indices n-pairs stride)
         out)))

;; scatter-add backward = gather, gather backward = scatter-add

;; ================================================================
;; indexed-dot: batched indexed dot product (multi-head aware)
;; out[e*n-slices+s] = Σ_{d=0}^{slice-dim-1} A[idx-a[e]*total-dim+s*slice-dim+d]
;;                                           * B[idx-b[e]*total-dim+s*slice-dim+d]
;; ================================================================

(deftm indexed-dot
  "Batched indexed dot product with multi-head support.
  For each pair e and slice s, computes dot product of slice-dim elements
  from A (at idx-a[e]) and B (at idx-b[e]).
  n-a, n-b: number of entities in A, B (for backward allocation)."
  [A :- (Array double) B :- (Array double)
   idx-a :- (Array long) idx-b :- (Array long)
   n-a :- Long n-b :- Long n-pairs :- Long
   slice-dim :- Long total-dim :- Long n-slices :- Long]
  :- (Array double)
  (let [out (double-array (* n-pairs n-slices))]
    (dotimes [e n-pairs]
      (let [ia (aget idx-a e)
            ib (aget idx-b e)]
        (dotimes [s n-slices]
          (let [off-a (+ (* ia (int total-dim)) (* s (int slice-dim)))
                off-b (+ (* ib (int total-dim)) (* s (int slice-dim)))]
            (loop [d 0 acc 0.0]
              (if (< d slice-dim)
                (recur (inc d)
                       (+ acc (* (aget A (+ off-a d))
                                 (aget B (+ off-b d)))))
                (aset out (+ (* e (int n-slices)) s) acc)))))))
    out))

(deftm indexed-dot-dA
  "Backward for A in indexed-dot.
  dA[idx-a[e]*total-dim+s*slice-dim+d] += dy[e*n-slices+s] * B[idx-b[e]*total-dim+s*slice-dim+d]"
  [dy :- (Array double) B :- (Array double)
   idx-a :- (Array long) idx-b :- (Array long)
   n-a :- Long n-pairs :- Long
   slice-dim :- Long total-dim :- Long n-slices :- Long]
  :- (Array double)
  (let [out (double-array (* n-a total-dim))]
    (dotimes [e n-pairs]
      (let [ia (aget idx-a e)
            ib (aget idx-b e)]
        (dotimes [s n-slices]
          (let [coeff (aget dy (+ (* e (int n-slices)) s))
                off-a (+ (* ia (int total-dim)) (* s (int slice-dim)))
                off-b (+ (* ib (int total-dim)) (* s (int slice-dim)))]
            (dotimes [d slice-dim]
              (let [a-idx (+ off-a d)
                    b-idx (+ off-b d)]
                (aset out a-idx
                      (+ (aget out a-idx)
                         (* coeff (aget B b-idx))))))))))
    out))

(deftm indexed-dot-dB
  "Backward for B in indexed-dot.
  dB[idx-b[e]*total-dim+s*slice-dim+d] += dy[e*n-slices+s] * A[idx-a[e]*total-dim+s*slice-dim+d]"
  [dy :- (Array double) A :- (Array double)
   idx-a :- (Array long) idx-b :- (Array long)
   n-b :- Long n-pairs :- Long
   slice-dim :- Long total-dim :- Long n-slices :- Long]
  :- (Array double)
  (let [out (double-array (* n-b total-dim))]
    (dotimes [e n-pairs]
      (let [ia (aget idx-a e)
            ib (aget idx-b e)]
        (dotimes [s n-slices]
          (let [coeff (aget dy (+ (* e (int n-slices)) s))
                off-a (+ (* ia (int total-dim)) (* s (int slice-dim)))
                off-b (+ (* ib (int total-dim)) (* s (int slice-dim)))]
            (dotimes [d slice-dim]
              (let [a-idx (+ off-a d)
                    b-idx (+ off-b d)]
                (aset out b-idx
                      (+ (aget out b-idx)
                         (* coeff (aget A a-idx))))))))))
    out))

;; ================================================================
;; scatter-mul-add: weighted scatter (multi-head aware)
;; out[dst[e]*total-dim+s*slice-dim+d] += coeffs[e*n-slices+s]
;;                                       * src[src[e]*total-dim+s*slice-dim+d]
;; ================================================================

(deftm scatter-mul-add
  "Weighted scatter with multi-head support.
  For each pair e, slice s, and dim d:
    out[dst[e]*total-dim+s*slice-dim+d] += coeffs[e*n-slices+s] * src[src[e]*total-dim+s*slice-dim+d]
  n-dst, n-src: entity counts for output allocation and backward."
  [coeffs :- (Array double) src :- (Array double)
   dst-indices :- (Array long) src-indices :- (Array long)
   n-dst :- Long n-src :- Long n-pairs :- Long
   slice-dim :- Long total-dim :- Long n-slices :- Long]
  :- (Array double)
  (let [out (double-array (* n-dst total-dim))]
    (dotimes [e n-pairs]
      (let [dst-node (aget dst-indices e)
            src-node (aget src-indices e)]
        (dotimes [s n-slices]
          (let [w (aget coeffs (+ (* e (int n-slices)) s))
                off-dst (+ (* dst-node (int total-dim)) (* s (int slice-dim)))
                off-src (+ (* src-node (int total-dim)) (* s (int slice-dim)))]
            (dotimes [d slice-dim]
              (let [dst-idx (+ off-dst d)
                    src-idx (+ off-src d)]
                (aset out dst-idx
                      (+ (aget out dst-idx)
                         (* w (aget src src-idx))))))))))
    out))

(deftm scatter-mul-add-d-coeffs
  "Backward for coeffs in scatter-mul-add.
  d_coeffs[e*n-slices+s] = Σ_d dy[dst[e]*td+s*sd+d] * src[src[e]*td+s*sd+d]"
  [dy :- (Array double) src :- (Array double)
   dst-indices :- (Array long) src-indices :- (Array long)
   n-pairs :- Long slice-dim :- Long total-dim :- Long n-slices :- Long]
  :- (Array double)
  (let [out (double-array (* n-pairs n-slices))]
    (dotimes [e n-pairs]
      (let [dst-node (aget dst-indices e)
            src-node (aget src-indices e)]
        (dotimes [s n-slices]
          (let [off-dst (+ (* dst-node (int total-dim)) (* s (int slice-dim)))
                off-src (+ (* src-node (int total-dim)) (* s (int slice-dim)))]
            (loop [d 0 acc 0.0]
              (if (< d slice-dim)
                (recur (inc d)
                       (+ acc (* (aget dy (+ off-dst d))
                                 (aget src (+ off-src d)))))
                (aset out (+ (* e (int n-slices)) s) acc)))))))
    out))

(deftm scatter-mul-add-d-src
  "Backward for src in scatter-mul-add.
  d_src[src[e]*td+s*sd+d] += coeffs[e*ns+s] * dy[dst[e]*td+s*sd+d]"
  [dy :- (Array double) coeffs :- (Array double)
   dst-indices :- (Array long) src-indices :- (Array long)
   n-src :- Long n-pairs :- Long
   slice-dim :- Long total-dim :- Long n-slices :- Long]
  :- (Array double)
  (let [out (double-array (* n-src total-dim))]
    (dotimes [e n-pairs]
      (let [dst-node (aget dst-indices e)
            src-node (aget src-indices e)]
        (dotimes [s n-slices]
          (let [w (aget coeffs (+ (* e (int n-slices)) s))
                off-src (+ (* src-node (int total-dim)) (* s (int slice-dim)))
                off-dst (+ (* dst-node (int total-dim)) (* s (int slice-dim)))]
            (dotimes [d slice-dim]
              (let [src-idx (+ off-src d)
                    dst-idx (+ off-dst d)]
                (aset out src-idx
                      (+ (aget out src-idx)
                         (* w (aget dy dst-idx))))))))))
    out))

;; ================================================================
;; scale-clamp-exp: out[i] = exp(clamp(x[i]*scale, -bound, bound))
;; Separates attention nonlinearity from dot product.
;; ================================================================

(deftm scale-clamp-exp
  "Pointwise scale, clamp, and exp.
  out[i] = exp(clamp(x[i] * scale, -clamp-bound, clamp-bound))"
  [x :- (Array double) scale :- Double
   clamp-bound :- Double n :- Long]
  :- (Array double)
  (broadcast [x]
             (m/exp (n/min clamp-bound (n/max (- clamp-bound) (* x scale))))))

(deftm scale-clamp-exp-backward
  "Backward for scale-clamp-exp.
  dx[i] = dy[i] * out[i] * scale * (|x[i]*scale| <= bound ? 1 : 0)"
  [dy :- (Array double) x :- (Array double) result :- (Array double)
   scale :- Double clamp-bound :- Double n :- Long]
  :- (Array double)
  (let [out (double-array n)]
    (dotimes [i n]
      (let [raw (* (aget x i) scale)
            clamp-mask (if (and (>= raw (- clamp-bound)) (<= raw clamp-bound)) 1.0 0.0)]
        (aset out i
              (* (* (aget dy i)
                    (aget result i))
                 (* scale clamp-mask)))))
    out))

;; ================================================================
;; reduce-axis: out[d] = Σ_b src[b*n-cols+d]
;; Sum-reduce along first axis.
;; ================================================================

(deftm reduce-axis
  "Sum-reduce along first axis.
  out[d] = Σ_{b=0}^{n-rows-1} src[b*n-cols+d] for d∈[0,n-cols)."
  [src :- (Array double) n-rows :- Long n-cols :- Long]
  :- (Array double)
  (let [out (double-array n-cols)]
    (dotimes [b n-rows]
      (dotimes [d n-cols]
        (let [idx (+ (* b (int n-cols)) d)]
          (aset out d
                (+ (aget out d)
                   (aget src idx))))))
    out))

(deftm reduce-axis-backward
  "Backward for reduce-axis: broadcast dy to each row.
  d_src[b*n-cols+d] = dy[d]"
  [dy :- (Array double) n-rows :- Long n-cols :- Long]
  :- (Array double)
  (let [out (double-array (* n-rows n-cols))]
    (dotimes [b n-rows]
      (dotimes [d n-cols]
        (aset out (+ (* b (int n-cols)) d)
              (aget dy d))))
    out))

;; ================================================================
;; segment-div: out[n*d+h_off+j] = wV[n*d+h_off+j] / (Z[n*nh+h] + eps)
;; Generic normalization: divide array by per-segment scalars with broadcast.
;; ================================================================

(deftm segment-div
  "Divide array by per-segment scalars with broadcast over slice dimension.
  out[node*emb-dim+head*dk+d] = wV[...] / (Z[node*n-heads+head] + eps)
  Generic enough for any per-segment normalization (attention, layer-norm, etc.)."
  [wV :- (Array double) Z :- (Array double)
   n-nodes :- Long emb-dim :- Long n-heads :- Long eps :- Double]
  :- (Array double)
  (let [dk (quot emb-dim n-heads)
        out (double-array (* n-nodes emb-dim))]
    (dotimes [node n-nodes]
      (dotimes [head n-heads]
        (let [z (+ (aget Z (+ (* node (int n-heads)) head)) eps)
              off (+ (* node (int emb-dim)) (* head (int dk)))]
          (dotimes [d dk]
            (let [idx (+ off d)]
              (aset out idx
                    (/ (aget wV idx) z)))))))
    out))

(deftm segment-div-dwV
  "Backward for wV: dwV[i] = dy[i] / (Z[n*nh+h] + eps)"
  [dy :- (Array double) Z :- (Array double)
   n-nodes :- Long emb-dim :- Long n-heads :- Long eps :- Double]
  :- (Array double)
  (let [dk (quot emb-dim n-heads)
        dwV (double-array (* n-nodes emb-dim))]
    (dotimes [node n-nodes]
      (dotimes [head n-heads]
        (let [z (+ (aget Z (+ (* node (int n-heads)) head)) eps)
              off (+ (* node (int emb-dim)) (* head (int dk)))]
          (dotimes [d dk]
            (let [idx (+ off d)]
              (aset dwV idx
                    (/ (aget dy idx) z)))))))
    dwV))

(deftm segment-div-jvp-dZ
  "Forward tangent (JVP, §13 A3) of segment-div in the Z direction — the
  quotient rule's second term: out = wV/(Z+eps) so
    d_i = −out_i·dZ_seg/(Z_seg+eps) = −wV_i·dZ_seg/(Z_seg+eps)².
  (The wV-direction term dwV/(Z+eps) is segment-div itself — linear in wV;
  see the :jvp-fn registration below.) Mirrors segment-div's loop structure."
  [wV :- (Array double) dZ :- (Array double) Z :- (Array double)
   n-nodes :- Long emb-dim :- Long n-heads :- Long eps :- Double]
  :- (Array double)
  (let [dk (quot emb-dim n-heads)
        out (double-array (* n-nodes emb-dim))]
    (dotimes [node n-nodes]
      (dotimes [head n-heads]
        (let [z-idx (+ (* node (int n-heads)) head)
              z (+ (aget Z z-idx) eps)
              factor (- (/ (aget dZ z-idx) (* z z)))
              off (+ (* node (int emb-dim)) (* head (int dk)))]
          (dotimes [d dk]
            (let [idx (+ off d)]
              (aset out idx (* (aget wV idx) factor)))))))
    out))

(deftm segment-div-dZ
  "Backward for Z: dZ[n*nh+h] = -sum_d(dy[..] * wV[..] / z²)"
  [dy :- (Array double) wV :- (Array double) Z :- (Array double)
   n-nodes :- Long emb-dim :- Long n-heads :- Long eps :- Double]
  :- (Array double)
  (let [dk (quot emb-dim n-heads)
        dZ (double-array (* n-nodes n-heads))]
    (dotimes [node n-nodes]
      (dotimes [head n-heads]
        (let [z-idx (+ (* node (int n-heads)) head)
              z (+ (aget Z z-idx) eps)
              z2 (* z z)
              off (+ (* node (int emb-dim)) (* head (int dk)))]
          (loop [d 0 acc 0.0]
            (if (< d dk)
              (let [idx (+ off d)]
                (recur (inc d)
                       (+ acc (* (aget dy idx)
                                 (aget wV idx)))))
              (aset dZ z-idx (- (/ acc z2))))))))
    dZ))

;; ================================================================
;; flat-embed-op: out[v*d+j] = values[v]*We[j] + be[j]
;;                            + space_emb[spaces[v]*d+j]
;;                            + state_emb[states[v]*d+j]
;;                            + pos_emb[v*d+j]
;; ================================================================

(deftm flat-embed-op
  "Embed flat variable values into embedding space.
  Combines value projection, space embedding, state embedding, and position embedding (all additive)."
  [values :- (Array double) space-emb :- (Array double) spaces :- (Array long)
   state-emb :- (Array double) states :- (Array long) pos-emb :- (Array double)
   We :- (Array double) be :- (Array double)
   n-vars :- Long emb-dim :- Long n-spaces :- Long n-states :- Long]
  :- (Array double)
  (let [out (double-array (* n-vars emb-dim))]
    (dotimes [v n-vars]
      (let [val-v (aget values v)
            sp (aget spaces v)
            st (aget states v)]
        (dotimes [d emb-dim]
          (let [idx (+ (* v (int emb-dim)) d)
                val-emb (+ (* (aget We d) val-v) (aget be d))
                space-e (aget space-emb (+ (* sp (int emb-dim)) d))
                state-e (aget state-emb (+ (* st (int emb-dim)) d))
                pos-e (aget pos-emb idx)]
            (aset out idx (+ val-emb space-e state-e pos-e))))))
    out))

(deftm flat-embed-d-values
  "Backward for values: d_values[v] = sum_d(dy[v*d+j] * We[j])"
  [dy :- (Array double) We :- (Array double)
   n-vars :- Long emb-dim :- Long]
  :- (Array double)
  (let [out (double-array n-vars)]
    (dotimes [v n-vars]
      (let [off (* v (int emb-dim))]
        (loop [d 0 acc 0.0]
          (if (< d emb-dim)
            (recur (inc d)
                   (+ acc (* (aget dy (+ off d))
                             (aget We d))))
            (aset out v acc)))))
    out))

(deftm flat-embed-dWe
  "Backward for We: dWe[j] = sum_v(dy[v*d+j] * values[v])"
  [dy :- (Array double) values :- (Array double)
   n-vars :- Long emb-dim :- Long]
  :- (Array double)
  (let [out (double-array emb-dim)]
    (dotimes [v n-vars]
      (let [val-v (aget values v)]
        (dotimes [d emb-dim]
          (aset out d
                (+ (aget out d)
                   (* (aget dy (+ (* v (int emb-dim)) d)) val-v))))))
    out))

(deftm flat-embed-dbe
  "Backward for be: dbe[j] = sum_v(dy[v*d+j])"
  [dy :- (Array double) n-vars :- Long emb-dim :- Long]
  :- (Array double)
  (let [out (double-array emb-dim)]
    (dotimes [v n-vars]
      (dotimes [d emb-dim]
        (aset out d
              (+ (aget out d)
                 (aget dy (+ (* v (int emb-dim)) d))))))
    out))

(deftm flat-embed-d-space-emb
  "Backward for space-emb: d_se[spaces[v]*d+j] += dy[v*d+j]"
  [dy :- (Array double) spaces :- (Array long)
   n-vars :- Long emb-dim :- Long n-spaces :- Long]
  :- (Array double)
  (let [out (double-array (* n-spaces emb-dim))]
    (dotimes [v n-vars]
      (let [sp (aget spaces v)]
        (dotimes [d emb-dim]
          (let [se-idx (+ (* sp (int emb-dim)) d)]
            (aset out se-idx
                  (+ (aget out se-idx)
                     (aget dy (+ (* v (int emb-dim)) d))))))))
    out))

(deftm flat-embed-d-state-emb
  "Backward for state-emb: d_ste[states[v]*d+j] += dy[v*d+j]"
  [dy :- (Array double) states :- (Array long)
   n-vars :- Long emb-dim :- Long n-states :- Long]
  :- (Array double)
  (let [out (double-array (* n-states emb-dim))]
    (dotimes [v n-vars]
      (let [st (aget states v)]
        (dotimes [d emb-dim]
          (let [ste-idx (+ (* st (int emb-dim)) d)]
            (aset out ste-idx
                  (+ (aget out ste-idx)
                     (aget dy (+ (* v (int emb-dim)) d))))))))
    out))

;; ================================================================
;; dot-rows: out[v] = h[v,:] · W + bias[0]
;; ================================================================

(deftm dot-rows
  [h :- (Array double) W :- (Array double) bias :- (Array double)
   n-rows :- Long dim :- Long]
  :- (Array double)
  (let [out (double-array n-rows)
        b0 (aget bias 0)]
    (dotimes [v n-rows]
      (let [off (* v (int dim))]
        (loop [d 0 acc 0.0]
          (if (< d dim)
            (recur (inc d)
                   (+ acc (* (aget h (+ off d))
                             (aget W d))))
            (aset out v (+ acc b0))))))
    out))

(deftm dot-rows-dh
  "Backward for h: dh[v*d+j] = dy[v] * W[j]"
  [dy :- (Array double) W :- (Array double)
   n-rows :- Long dim :- Long]
  :- (Array double)
  (let [out (double-array (* n-rows dim))]
    (dotimes [v n-rows]
      (let [dy-v (aget dy v)]
        (dotimes [d dim]
          (aset out (+ (* v (int dim)) d)
                (* dy-v (aget W d))))))
    out))

(deftm dot-rows-dW
  "Backward for W: dW[j] = sum_v(dy[v] * h[v*d+j])"
  [dy :- (Array double) h :- (Array double)
   n-rows :- Long dim :- Long]
  :- (Array double)
  (let [out (double-array dim)]
    (dotimes [v n-rows]
      (let [dy-v (aget dy v)]
        (dotimes [d dim]
          (aset out d
                (+ (aget out d)
                   (* dy-v (aget h (+ (* v (int dim)) d))))))))
    out))

(deftm dot-rows-dbias
  "Backward for bias: dbias[0] = sum_v(dy[v])"
  [dy :- (Array double) n-rows :- Long]
  :- (Array double)
  (let [out (double-array 1)]
    (loop [v 0 acc 0.0]
      (if (< v n-rows)
        (recur (inc v) (+ acc (aget dy v)))
        (aset out 0 acc)))
    out))

;; ================================================================
;; masked-mse-loss: loss = mean((pred[i]-target[i])² for i where states[i]!=1)
;; ================================================================

(deftm masked-mse-loss
  [pred :- (Array double) target :- (Array double)
   states :- (Array long) n :- Long]
  :- Double
  (loop [i 0 acc 0.0 cnt 0]
    (if (< i n)
      (if (not= (aget states i) 1)
        (let [diff (- (aget pred i) (aget target i))]
          (recur (inc i) (+ acc (* diff diff)) (inc cnt)))
        (recur (inc i) acc cnt))
      (if (pos? cnt)
        (/ acc (double cnt))
        0.0))))

(deftm masked-mse-loss-backward
  "Backward for pred in masked-mse-loss.
  d_pred[i] = 2*(pred[i]-target[i])/cnt * dy where states[i]!=1, else 0."
  [dy :- Double pred :- (Array double) target :- (Array double)
   states :- (Array long) n :- Long]
  :- (Array double)
  (let [;; First count non-observed
        cnt (loop [i 0 c 0]
              (if (< i n)
                (if (not= (aget states i) 1)
                  (recur (inc i) (inc c))
                  (recur (inc i) c))
                c))
        out (double-array n)]
    (when (pos? cnt)
      (let [scale (/ (* 2.0 dy) (double cnt))]
        (dotimes [i n]
          (when (not= (aget states i) 1)
            (aset out i
                  (* scale (- (aget pred i)
                              (aget target i))))))))
    out))

;; ================================================================
;; Flat AD templates
;;
;; Keep these registrations next to the forward op and backward helper code
;; so the closure rrule and flat-codegen rule stay in sync.
;; ================================================================

(tmpl/merge-into-template! 'raster.dl.array-ops/array-add
                           {:params '[a b n] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [a b n] _result-sym adjoint-sym gensym-fn]
                              (let [da (gensym-fn "da" (tmpl/grad-tag a))
                                    db (gensym-fn "db" (tmpl/grad-tag b))]
                                [(update ctx :bindings into
                                         [da (list 'raster.dl.array-ops/array-add-backward adjoint-sym n)
                                          db (list 'raster.dl.array-ops/array-add-backward adjoint-sym n)])
                                 [da db nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/broadcast-add
                           {:params '[h t batch dim] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [h t batch dim] _result-sym adjoint-sym gensym-fn]
                              (let [dh (gensym-fn "dh" (tmpl/grad-tag h))
                                    dt (gensym-fn "dt" (tmpl/grad-tag t))
                                    ;; integer element-count intermediate: Π is ⊥
                                    ;; on long — stays untagged by design
                                    n (gensym-fn "ba_n")]
                                [(update ctx :bindings into
                                         [n (list 'raster.numeric/* batch dim)
                                          dh (list 'raster.dl.array-ops/broadcast-add-dh adjoint-sym n)
                                          dt (list 'raster.dl.array-ops/broadcast-add-dt adjoint-sym batch dim)])
                                 [dh dt nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/scatter-add
                           {:params '[vals indices n-dst n-pairs stride]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [vals indices n-dst n-pairs stride]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-vals (gensym-fn "d_vals" (tmpl/grad-tag vals))]
                                [(update ctx :bindings into
                                         [d-vals (list 'raster.dl.array-ops/gather
                                                       adjoint-sym indices n-dst n-pairs stride)])
                                 [d-vals nil nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/gather
                           {:params '[src indices n-src n-pairs stride]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [src indices n-src n-pairs stride]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-src (gensym-fn "d_src" (tmpl/grad-tag src))]
                                [(update ctx :bindings into
                                         [d-src (list 'raster.dl.array-ops/scatter-add
                                                      adjoint-sym indices n-src n-pairs stride)])
                                 [d-src nil nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/indexed-dot
                           {:params '[A B idx-a idx-b n-a n-b n-pairs slice-dim total-dim n-slices]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [A B idx-a idx-b n-a n-b n-pairs slice-dim total-dim n-slices]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [dA (gensym-fn "dA" (tmpl/grad-tag A))
                                    dB (gensym-fn "dB" (tmpl/grad-tag B))]
                                [(update ctx :bindings into
                                         [dA (list 'raster.dl.array-ops/indexed-dot-dA
                                                   adjoint-sym B idx-a idx-b n-a n-pairs
                                                   slice-dim total-dim n-slices)
                                          dB (list 'raster.dl.array-ops/indexed-dot-dB
                                                   adjoint-sym A idx-a idx-b n-b n-pairs
                                                   slice-dim total-dim n-slices)])
                                 [dA dB nil nil nil nil nil nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/scatter-mul-add
                           {:params '[coeffs src dst-indices src-indices n-dst n-src n-pairs
                                      slice-dim total-dim n-slices]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [coeffs src dst-indices src-indices n-dst n-src n-pairs
                                      slice-dim total-dim n-slices]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-coeffs (gensym-fn "d_coeffs" (tmpl/grad-tag coeffs))
                                    d-src (gensym-fn "d_src" (tmpl/grad-tag src))]
                                [(update ctx :bindings into
                                         [d-coeffs (list 'raster.dl.array-ops/scatter-mul-add-d-coeffs
                                                         adjoint-sym src dst-indices src-indices
                                                         n-pairs slice-dim total-dim n-slices)
                                          d-src (list 'raster.dl.array-ops/scatter-mul-add-d-src
                                                      adjoint-sym coeffs dst-indices src-indices
                                                      n-src n-pairs slice-dim total-dim n-slices)])
                                 [d-coeffs d-src nil nil nil nil nil nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/scale-clamp-exp
                           {:params '[x scale clamp-bound n]
                            :result 'result :adjoint 'dy
                            :grads-fn
                            (fn [ctx [x scale clamp-bound n]
                                 result-sym adjoint-sym gensym-fn]
                              (let [dx (gensym-fn "dx" (tmpl/grad-tag x))]
                                [(update ctx :bindings into
                                         [dx (list 'raster.dl.array-ops/scale-clamp-exp-backward
                                                   adjoint-sym x result-sym scale clamp-bound n)])
                                 [dx nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/reduce-axis
                           {:params '[src n-rows n-cols]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [src n-rows n-cols]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-src (gensym-fn "d_src" (tmpl/grad-tag src))]
                                [(update ctx :bindings into
                                         [d-src (list 'raster.dl.array-ops/reduce-axis-backward
                                                      adjoint-sym n-rows n-cols)])
                                 [d-src nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/segment-div
                           {:params '[wV Z n-nodes emb-dim n-heads eps]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [wV Z n-nodes emb-dim n-heads eps]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [dwV (gensym-fn "dwV" (tmpl/grad-tag wV))
                                    dZ (gensym-fn "dZ" (tmpl/grad-tag Z))]
                                [(update ctx :bindings into
                                         [dwV (list 'raster.dl.array-ops/segment-div-dwV
                                                    adjoint-sym Z n-nodes emb-dim n-heads eps)
                                          dZ (list 'raster.dl.array-ops/segment-div-dZ
                                                   adjoint-sym wV Z n-nodes emb-dim n-heads eps)])
                                 [dwV dZ nil nil nil nil]]))})

;; segment-div forward tangent (JVP, §13 A3) — quotient rule:
;;   d = dwV/(Z+eps) ⊕ (−out⊙dZ/(Z+eps))
;; The wV term is segment-div ITSELF on the tangent (linear in wV); the Z
;; term is the dedicated segment-div-jvp-dZ kernel above.
(tmpl/merge-into-template!
 'raster.dl.array-ops/segment-div
 {:jvp-fn
  (fn [ctx [wV Z n-nodes emb-dim n-heads eps] tangent-args _result-sym gensym-fn]
    (let [dwV (nth tangent-args 0 nil)
          dZ (nth tangent-args 1 nil)]
      (when-not (or dwV dZ)
        (throw (ex-info "segment-div jvp: no active tangent reached the call"
                        {:op 'raster.dl.array-ops/segment-div})))
      (let [[ctx terms]
            (if dwV
              (let [[ctx' s] (tmpl/bind-jvp-term ctx gensym-fn "jsdiv_wV"
                                                 (list 'raster.dl.array-ops/segment-div
                                                       dwV Z n-nodes emb-dim n-heads eps))]
                [ctx' [s]])
              [ctx []])
            [ctx terms]
            (if dZ
              (let [[ctx' s] (tmpl/bind-jvp-term ctx gensym-fn "jsdiv_Z"
                                                 (list 'raster.dl.array-ops/segment-div-jvp-dZ
                                                       wV dZ Z n-nodes emb-dim n-heads eps))]
                [ctx' (conj terms s)])
              [ctx terms])]
        (tmpl/sum-tangent-contribs ctx terms gensym-fn))))})

(tmpl/merge-into-template! 'raster.dl.array-ops/flat-embed-op
                           {:params '[values space-emb spaces state-emb states pos-emb
                                      We be n-vars emb-dim n-spaces n-states]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [values space-emb spaces state-emb states pos-emb
                                      We be n-vars emb-dim n-spaces n-states]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-values (gensym-fn "d_values" (tmpl/grad-tag values))
                                    d-space-emb (gensym-fn "d_space_emb" (tmpl/grad-tag space-emb))
                                    d-state-emb (gensym-fn "d_state_emb" (tmpl/grad-tag state-emb))
                                    d-We (gensym-fn "d_We" (tmpl/grad-tag We))
                                    d-be (gensym-fn "d_be" (tmpl/grad-tag be))]
                                [(update ctx :bindings into
                                         [d-values (list 'raster.dl.array-ops/flat-embed-d-values
                                                         adjoint-sym We n-vars emb-dim)
                                          d-space-emb (list 'raster.dl.array-ops/flat-embed-d-space-emb
                                                            adjoint-sym spaces n-vars emb-dim n-spaces)
                                          d-state-emb (list 'raster.dl.array-ops/flat-embed-d-state-emb
                                                            adjoint-sym states n-vars emb-dim n-states)
                                          d-We (list 'raster.dl.array-ops/flat-embed-dWe
                                                     adjoint-sym values n-vars emb-dim)
                                          d-be (list 'raster.dl.array-ops/flat-embed-dbe
                                                     adjoint-sym n-vars emb-dim)])
                                 ;; [values space-emb spaces state-emb states pos-emb We be n-vars emb-dim n-spaces n-states]
                                 [d-values d-space-emb nil d-state-emb nil nil d-We d-be nil nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/dot-rows
                           {:params '[h W bias n-rows dim] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [h W bias n-rows dim] _result-sym adjoint-sym gensym-fn]
                              (let [dh (gensym-fn "dh" (tmpl/grad-tag h))
                                    dW (gensym-fn "dW" (tmpl/grad-tag W))
                                    dbias (gensym-fn "dbias" (tmpl/grad-tag bias))]
                                [(update ctx :bindings into
                                         [dh (list 'raster.dl.array-ops/dot-rows-dh adjoint-sym W n-rows dim)
                                          dW (list 'raster.dl.array-ops/dot-rows-dW adjoint-sym h n-rows dim)
                                          dbias (list 'raster.dl.array-ops/dot-rows-dbias adjoint-sym n-rows)])
                                 [dh dW dbias nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/masked-mse-loss
                           {:params '[pred target states n] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [pred target states n] _result-sym adjoint-sym gensym-fn]
                              (let [d-pred (gensym-fn "d_pred" (tmpl/grad-tag pred))]
                                [(update ctx :bindings into
                                         [d-pred (list 'raster.dl.array-ops/masked-mse-loss-backward
                                                       adjoint-sym pred target states n)])
                                 [d-pred nil nil nil]]))})
