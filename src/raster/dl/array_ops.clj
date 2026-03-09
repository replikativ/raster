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
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; array-add: out[i] = a[i] + b[i]
;; Exists as a named primitive (rather than using generic +) because:
;; 1. The AD tape needs an explicit op to attach an rrule for skip connections
;; 2. Used by gsdm.clj ResnetBlock and graph attention residual paths
;; 3. Has both a closure rrule and a colocated flat AD template
;; ================================================================

(deftm array-add
  [a :- (Array double) b :- (Array double) n :- Long]
  :- (Array double)
  (broadcast [a b] (+ a b)))

(deftm array-add-backward
  "Backward for array-add: d_a = dy, d_b = dy (both copies)."
  [dy :- (Array double) n :- Long]
  :- (Array double)
  (let [out (double-array n)]
    (acopy! dy 0 out 0 n)
    out))

(tmpl/merge-into-template! 'raster.dl.array-ops/array-add
                           {:pullback-factory (fn [_result a b n]
                                                (fn [dy]
                                                  (let [n (long n)
                                                        d-a (double-array n)
                                                        d-b (double-array n)]
                                                    (acopy! dy 0 d-a 0 n)
                                                    (acopy! dy 0 d-b 0 n)
                                                    [d-a d-b nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/broadcast-add
                           {:pullback-factory (fn [_result h t batch dim]
                                                (fn [dy]
                                                  (let [batch (long batch) dim (long dim)
                                                        n (* batch dim)
                                                        d-h (double-array n)
                                                        d-t (double-array dim)]
                                                    (acopy! dy 0 d-h 0 n)
                                                    (dotimes [b batch]
                                                      (dotimes [d dim]
                                                        (let [idx (+ (* b (int dim)) d)]
                                                          (aset d-t d
                                                                (+ (aget d-t d)
                                                                   (aget dy idx))))))
                                                    [d-h d-t nil nil])))})

;; ================================================================
;; GENERIC PRIMITIVES
;; ================================================================

;; ================================================================
;; gather: out[e*stride+s] = src[indices[e]*stride+s]
;; Adjoint of scatter-add.
;; ================================================================

(deftm gather
  "Gather values from indexed positions.
  out[e*stride+s] = src[indices[e]*stride+s] for e∈[0,n-pairs), s∈[0,stride).
  n-src: number of source entities (for backward scatter allocation)."
  [src :- (Array double) indices :- (Array long)
   n-src :- Long n-pairs :- Long stride :- Long]
  :- (Array double)
  (let [out (double-array (* n-pairs stride))]
    (dotimes [e n-pairs]
      (let [src-idx (aget indices e)]
        (dotimes [s stride]
          (aset out (+ (* e (int stride)) s)
                (aget src (+ (* src-idx (int stride)) s))))))
    out))

;; ================================================================
;; scatter-add: out[indices[e]*stride+s] += vals[e*stride+s]
;; Adjoint of gather.
;; ================================================================

(deftm scatter-add
  "Scatter-add values to indexed destinations.
  out[indices[e]*stride+s] += vals[e*stride+s] for e∈[0,n-pairs), s∈[0,stride).
  n-dst: number of destination entities (output size = n-dst*stride)."
  [vals :- (Array double) indices :- (Array long)
   n-dst :- Long n-pairs :- Long stride :- Long]
  :- (Array double)
  (let [out (double-array (* n-dst stride))]
    (dotimes [e n-pairs]
      (let [dst-idx (aget indices e)]
        (dotimes [s stride]
          (let [out-idx (+ (* dst-idx (int stride)) s)
                val-idx (+ (* e (int stride)) s)]
            (aset out out-idx
                  (+ (aget out out-idx)
                     (aget vals val-idx)))))))
    out))

;; scatter-add backward = gather, gather backward = scatter-add
(tmpl/merge-into-template! 'raster.dl.array-ops/scatter-add
                           {:pullback-factory (fn [_result vals indices n-dst n-pairs stride]
                                                (fn [dy]
                                                  (let [d-vals (gather dy indices n-dst n-pairs stride)]
                                                    [d-vals nil nil nil nil])))})

(tmpl/merge-into-template! 'raster.dl.array-ops/gather
                           {:pullback-factory (fn [_result src indices n-src n-pairs stride]
                                                (fn [dy]
                                                  (let [d-src (scatter-add dy indices n-src n-pairs stride)]
                                                    [d-src nil nil nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/indexed-dot
                           {:pullback-factory (fn [_result A B idx-a idx-b n-a n-b n-pairs slice-dim total-dim n-slices]
                                                (fn [dy]
                                                  (let [dA (indexed-dot-dA dy B idx-a idx-b n-a n-pairs slice-dim total-dim n-slices)
                                                        dB (indexed-dot-dB dy A idx-a idx-b n-b n-pairs slice-dim total-dim n-slices)]
                                                    [dA dB nil nil nil nil nil nil nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/scatter-mul-add
                           {:pullback-factory (fn [_result coeffs src dst-indices src-indices n-dst n-src n-pairs
                                                   slice-dim total-dim n-slices]
                                                (fn [dy]
                                                  (let [d-coeffs (scatter-mul-add-d-coeffs dy src dst-indices src-indices
                                                                                           n-pairs slice-dim total-dim n-slices)
                                                        d-src (scatter-mul-add-d-src dy coeffs dst-indices src-indices
                                                                                     n-src n-pairs slice-dim total-dim n-slices)]
                                                    [d-coeffs d-src nil nil nil nil nil nil nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/scale-clamp-exp
                           {:pullback-factory (fn [result x scale clamp-bound n]
                                                (fn [dy]
                                                  (let [dx (scale-clamp-exp-backward dy x result scale clamp-bound n)]
                                                    [dx nil nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/reduce-axis
                           {:pullback-factory (fn [_result src n-rows n-cols]
                                                (fn [dy]
                                                  (let [d-src (reduce-axis-backward dy n-rows n-cols)]
                                                    [d-src nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/segment-div
                           {:pullback-factory (fn [_result wV Z n-nodes emb-dim n-heads eps]
                                                (fn [dy]
                                                  (let [dwV (segment-div-dwV dy Z n-nodes emb-dim n-heads eps)
                                                        dZ (segment-div-dZ dy wV Z n-nodes emb-dim n-heads eps)]
                                                    [dwV dZ nil nil nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/flat-embed-op
                           {:pullback-factory
                            (fn [_result values space-emb spaces state-emb states pos-emb
                                 We be n-vars emb-dim n-spaces n-states]
                              (fn [dy]
                                (let [n-vars (long n-vars) emb-dim (long emb-dim)
                                      n-spaces (long n-spaces) n-states (long n-states)
                                      d-values (flat-embed-d-values dy We n-vars emb-dim)
                                      d-space-emb (flat-embed-d-space-emb dy spaces n-vars emb-dim n-spaces)
                                      d-state-emb (flat-embed-d-state-emb dy states n-vars emb-dim n-states)
                                      d-We (flat-embed-dWe dy values n-vars emb-dim)
                                      d-be (flat-embed-dbe dy n-vars emb-dim)]
                                  ;; [values space-emb spaces state-emb states pos-emb We be n-vars emb-dim n-spaces n-states]
                                  [d-values d-space-emb nil d-state-emb nil nil d-We d-be nil nil nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/dot-rows
                           {:pullback-factory (fn [_result h W bias n-rows dim]
                                                (fn [dy]
                                                  (let [n-rows (long n-rows) dim (long dim)
                                                        d-h (dot-rows-dh dy W n-rows dim)
                                                        d-W (dot-rows-dW dy h n-rows dim)
                                                        d-bias (dot-rows-dbias dy n-rows)]
                                                    [d-h d-W d-bias nil nil])))})

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

(tmpl/merge-into-template! 'raster.dl.array-ops/masked-mse-loss
                           {:pullback-factory (fn [_result pred target states n]
                                                (fn [dy]
                                                  (let [d-pred (masked-mse-loss-backward (double dy) pred target states n)]
                                                    [d-pred nil nil nil])))})

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
                              (let [da (gensym-fn "da")
                                    db (gensym-fn "db")]
                                [(update ctx :bindings into
                                         [da (list 'raster.dl.array-ops/array-add-backward adjoint-sym n)
                                          db (list 'raster.dl.array-ops/array-add-backward adjoint-sym n)])
                                 [da db nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/broadcast-add
                           {:params '[h t batch dim] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [h t batch dim] _result-sym adjoint-sym gensym-fn]
                              (let [dh (gensym-fn "dh")
                                    dt (gensym-fn "dt")
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
                              (let [d-vals (gensym-fn "d_vals")]
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
                              (let [d-src (gensym-fn "d_src")]
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
                              (let [dA (gensym-fn "dA")
                                    dB (gensym-fn "dB")]
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
                              (let [d-coeffs (gensym-fn "d_coeffs")
                                    d-src (gensym-fn "d_src")]
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
                              (let [dx (gensym-fn "dx")]
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
                              (let [d-src (gensym-fn "d_src")]
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
                              (let [dwV (gensym-fn "dwV")
                                    dZ (gensym-fn "dZ")]
                                [(update ctx :bindings into
                                         [dwV (list 'raster.dl.array-ops/segment-div-dwV
                                                    adjoint-sym Z n-nodes emb-dim n-heads eps)
                                          dZ (list 'raster.dl.array-ops/segment-div-dZ
                                                   adjoint-sym wV Z n-nodes emb-dim n-heads eps)])
                                 [dwV dZ nil nil nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/flat-embed-op
                           {:params '[values space-emb spaces state-emb states pos-emb
                                      We be n-vars emb-dim n-spaces n-states]
                            :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [values space-emb spaces state-emb states pos-emb
                                      We be n-vars emb-dim n-spaces n-states]
                                 _result-sym adjoint-sym gensym-fn]
                              (let [d-values (gensym-fn "d_values")
                                    d-space-emb (gensym-fn "d_space_emb")
                                    d-state-emb (gensym-fn "d_state_emb")
                                    d-We (gensym-fn "d_We")
                                    d-be (gensym-fn "d_be")]
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
                              (let [dh (gensym-fn "dh")
                                    dW (gensym-fn "dW")
                                    dbias (gensym-fn "dbias")]
                                [(update ctx :bindings into
                                         [dh (list 'raster.dl.array-ops/dot-rows-dh adjoint-sym W n-rows dim)
                                          dW (list 'raster.dl.array-ops/dot-rows-dW adjoint-sym h n-rows dim)
                                          dbias (list 'raster.dl.array-ops/dot-rows-dbias adjoint-sym n-rows)])
                                 [dh dW dbias nil nil]]))})

(tmpl/merge-into-template! 'raster.dl.array-ops/masked-mse-loss
                           {:params '[pred target states n] :result nil :adjoint 'dy
                            :grads-fn
                            (fn [ctx [pred target states n] _result-sym adjoint-sym gensym-fn]
                              (let [d-pred (gensym-fn "d_pred")]
                                [(update ctx :bindings into
                                         [d-pred (list 'raster.dl.array-ops/masked-mse-loss-backward
                                                       adjoint-sym pred target states n)])
                                 [d-pred nil nil nil]]))})
