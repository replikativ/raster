(ns raster.dl.nn
  "Deep learning layer operations for Raster.

  All layers are parametric deftm functions operating on flat arrays (double[]
  or float[]) with explicit shape parameters. The compiler sees through all
  layer boundaries as S-expressions and can fuse/optimize end-to-end.

  Layers:
    matmul              - batched matrix multiply (BLAS-backed)
    linear              - y = x @ W^T + b
    silu, gelu, sigmoid, tanh-act, leaky-relu  - activations
    layer-norm, group-norm, batch-norm         - normalization
    conv1d, conv2d      - convolution via im2col + GEMM
    maxpool2d           - max pooling (non-overlapping windows)
    embedding           - lookup table
    dropout             - stochastic regularization

  Each layer has:
    1. Forward computation (deftm)
    2. rrule registration (reverse AD)
    3. rrule template registration (flat AD codegen)"
  (:refer-clojure :exclude [aget aset alength aclone + - * /])
  (:require [raster.core :refer [deftm ftm broadcast]]
            [raster.arrays :refer [aget aset alength aclone acopy! alloc-like]]
            [raster.par]
            [raster.math :as m]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.ad.templates :as tmpl]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.device :as device]
            [raster.linalg.blas :as blas]))

;; ================================================================
;; Transpose helper
;; ================================================================

(deftm transpose-2d (All [T] [a :- (Array T) rows :- Long cols :- Long]
                         :- (Array T)
                         (let [out (alloc-like a (* rows cols))]
                           (dotimes [i rows]
                             (dotimes [j cols]
                               (aset out (+ (* j rows) i)
                                     (aget a (+ (* i cols) j)))))
                           out)))

(deftm transpose-2d! (All [T] [a :- (Array T) rows :- Long cols :- Long
                               out :- (Array T)] :- (Array T)
                          (dotimes [i rows]
                            (dotimes [j cols]
                              (aset out (+ (* j rows) i)
                                    (aget a (+ (* i cols) j)))))
                          out))

;; transpose rrule: d_a = transpose(d_out, cols, rows)
(tmpl/merge-into-template! 'raster.dl.nn/transpose-2d
                           {:pullback-factory (fn [_result _a rows cols]
                                                (fn [d-out]
                                                  [(transpose-2d d-out cols rows) nil nil]))})

;; ================================================================
;; Matrix multiply: C = A @ B, A:[M,K], B:[K,N] -> C:[M,N]
;; Row-major flat double[] representation
;; ================================================================

(deftm matmul (All [T] [A :- (Array T) B :- (Array T)
                        m :- Long k :- Long n :- Long] :- (Array T)
                   (let [out (alloc-like A (* m n))]
                     (blas/dgemm! A B out m k n (n/oftype A 1.0) (n/oftype A 0.0))
                     out)))

;; In-place matmul: C = A @ B, writes into pre-allocated out
(deftm matmul! (All [T] [A :- (Array T) B :- (Array T)
                         out :- (Array T)
                         m :- Long k :- Long n :- Long] :- (Array T)
                    (blas/dgemm! A B out m k n (n/oftype A 1.0) (n/oftype A 0.0))
                    out))

;; ================================================================
;; Matmul backward helpers (deftm — visible to compiler templates)
;; ================================================================

;; dA = dC @ B^T, dC:[M,N] B:[K,N] -> dA:[M,K]
(deftm matmul-dA (All [T] [dC :- (Array T) B :- (Array T)
                           m :- Long k :- Long n :- Long] :- (Array T)
                      (let [dA (alloc-like dC (* m k))]
                        (blas/dgemm-nt! dC B dA m n k (n/oftype dC 1.0) (n/oftype dC 0.0))
                        dA)))

;; dB = A^T @ dC, A:[M,K] dC:[M,N] -> dB:[K,N]
(deftm matmul-dB (All [T] [A :- (Array T) dC :- (Array T)
                           m :- Long k :- Long n :- Long] :- (Array T)
                      (let [dB (alloc-like A (* k n))]
                        (blas/dgemm-tn! A dC dB k m n (n/oftype A 1.0) (n/oftype A 0.0))
                        dB)))

(deftm matmul-dA! (All [T] [dC :- (Array T) B :- (Array T)
                            m :- Long k :- Long n :- Long
                            out :- (Array T)] :- (Array T)
                       (blas/dgemm-nt! dC B out m n k (n/oftype dC 1.0) (n/oftype dC 0.0))
                       out))

(deftm matmul-dB! (All [T] [A :- (Array T) dC :- (Array T)
                            m :- Long k :- Long n :- Long
                            out :- (Array T)] :- (Array T)
                       (blas/dgemm-tn! A dC out k m n (n/oftype A 1.0) (n/oftype A 0.0))
                       out))

;; matmul rrule: dA = dC @ B^T, dB = A^T @ dC
;; Delegates to backward deftms (parametric, float/double agnostic)
(tmpl/merge-into-template! 'raster.dl.nn/matmul
                           {:pullback-factory (fn [_result A B m k n]
                                                (fn [dC]
                                                  [(matmul-dA dC B m k n)
                                                   (matmul-dB A dC m k n)
                                                   nil nil nil]))})

;; ================================================================
;; Linear backward helpers (deftm — visible to compiler templates)
;; ================================================================

;; dx = dy @ W, dy:[B,out] W:[out,in] -> dx:[B,in]
(deftm linear-dx (All [T] [dy :- (Array T) W :- (Array T)
                           batch :- Long in-f :- Long out-f :- Long] :- (Array T)
                      (let [dx (alloc-like dy (* batch in-f))]
                        (blas/dgemm! dy W dx batch out-f in-f (n/oftype dy 1.0) (n/oftype dy 0.0))
                        dx)))

;; dW = dy^T @ x, dy:[B,out] x:[B,in] -> dW:[out,in]
(deftm linear-dW (All [T] [dy :- (Array T) x :- (Array T)
                           batch :- Long in-f :- Long out-f :- Long] :- (Array T)
                      (let [dW (alloc-like dy (* out-f in-f))]
                        (blas/dgemm-tn! dy x dW out-f batch in-f (n/oftype dy 1.0) (n/oftype dy 0.0))
                        dW)))

;; db = sum(dy, dim=0) -> [out_f]
(deftm linear-db (All [T] [dy :- (Array T) batch :- Long out-f :- Long] :- (Array T)
                      (let [db (alloc-like dy out-f)]
                        (dotimes [bi batch]
                          (dotimes [j out-f]
                            (aset db j (+ (aget db j) (aget dy (+ (* bi (int out-f)) j))))))
                        db)))

;; ================================================================
;; SiLU backward helper (deftm — visible to compiler)
;; dx[i] = dy[i] * sigma(x[i]) * (1 + x[i]*(1 - sigma(x[i])))
;; ================================================================

(deftm silu-backward (All [T] [dy :- (Array T) x :- (Array T) n :- Long]
                          :- (Array T)
                          (broadcast [dy x] (let [si (/ 1.0 (+ 1.0 (m/exp (- x))))
                                                  grad (* si (+ 1.0 (* x (- 1.0 si))))]
                                              (* dy grad)))))

;; ================================================================
;; Linear: y = x @ W^T + b
;; x:[batch, in_f], W:[out_f, in_f], b:[out_f] -> y:[batch, out_f]
;; ================================================================

(deftm linear (All [T] [x :- (Array T) W :- (Array T) b :- (Array T)
                        batch :- Long in-f :- Long out-f :- Long] :- (Array T)
                   (let [;; y = x @ W^T, x:[batch,in_f] W:[out_f,in_f] -> y:[batch,out_f]
        ;; Use BLAS with B-transposed to avoid explicit transpose
                         y (alloc-like x (* batch out-f))
        ;; Pre-fill y with bias (broadcast b over batch rows) for epilogue fusion
                         _ (dotimes [i batch]
                             (acopy! b 0 y (* i out-f) out-f))
        ;; y = 1.0 * x @ W^T + 1.0 * y (bias already in y)
                         _ (blas/dgemm-nt! x W y batch in-f out-f (n/oftype x 1.0) (n/oftype x 1.0))]
                     y)))

;; In-place linear: writes into pre-allocated y
(deftm linear! (All [T] [x :- (Array T) W :- (Array T) b :- (Array T)
                         y :- (Array T)
                         batch :- Long in-f :- Long out-f :- Long] :- (Array T)
                    (let [_ (dotimes [i batch]
                              (acopy! b 0 y (* i out-f) out-f))
                          _ (blas/dgemm-nt! x W y batch in-f out-f (n/oftype x 1.0) (n/oftype x 1.0))]
                      y)))

;; linear rrule:
;; dx = dy @ W, dW = dy^T @ x, db = sum(dy, dim=0)
;; Delegates to backward deftms (parametric, float/double agnostic)
(tmpl/merge-into-template! 'raster.dl.nn/linear
                           {:pullback-factory (fn [_result x W _b batch in-f out-f]
                                                (fn [dy]
                                                  [(linear-dx dy W batch in-f out-f)
                                                   (linear-dW dy x batch in-f out-f)
                                                   (linear-db dy batch out-f)
                                                   nil nil nil]))})

;; ================================================================
;; Activations
;; ================================================================

;; --- SiLU (Swish): x * sigmoid(x) ---
(deftm silu (All [T] [x :- (Array T) n :- Long] :- (Array T)
                 (broadcast [x] (let [si (/ 1.0 (+ 1.0 (m/exp (- x))))] (* x si)))))

;; silu': sigma(x) * (1 + x*(1-sigma(x)))
;; Delegates to backward deftm (parametric, float/double agnostic)
(tmpl/merge-into-template! 'raster.dl.nn/silu
                           {:pullback-factory (fn [_result x n]
                                                (fn [dy]
                                                  [(silu-backward dy x n) nil]))})

;; SwiGLU activation, fused: out = silu(gate) * up. par/map-void! !-variant (one work-item /
;; element) — the GPU-resident decode FFN form, and SIMD-vectorizes on CPU. The par index is the
;; subscript directly (no computed index); float compute via raster.numeric + raster.math/exp.
(deftm silu-mul! (All [T] [gate :- (Array T) up :- (Array T) out :- (Array T) n :- Long] :- Void
                      (raster.par/map-void! i n
                                            (let [g (aget gate i)]
                                              (aset out i (* (* g (/ 1.0 (+ 1.0 (m/exp (- g))))) (aget up i)))))))

;; GeGLU activation, fused: out = gelu(gate) * up (tanh approximation, gemma's FFN). Mirrors
;; silu-mul! — one work-item/element, GPU-resident decode FFN form. sqrt(2/pi)=0.7978845608028654
;; inlined as a literal so no scalar is bound outside the par body.
(deftm gelu-mul! (All [T] [gate :- (Array T) up :- (Array T) out :- (Array T) n :- Long] :- Void
                      (raster.par/map-void! i n
                                            (let [g (aget gate i)]
                                              (aset out i (* (* 0.5 g
                                                                (+ 1.0 (m/tanh (* 0.7978845608028654
                                                                                  (+ g (* 0.044715 g g g))))))
                                                             (aget up i)))))))

;; residual add, !-variant: out = a + b (one work-item / element).
(deftm residual-add! (All [T] [a :- (Array T) b :- (Array T) out :- (Array T) n :- Long] :- Void
                          (raster.par/map-void! i n
                                                (aset out i (+ (aget a i) (aget b i))))))
;; --- SkipLayerNorm: fused residual-add + layer-norm (ORT SkipLayerNormalization) ---
;; out = LayerNorm(a + b). Folds the residual add into layer-norm's reads so the
;; separate residual-add pass AND its [batch,features] intermediate buffer vanish —
;; (a+b) is consumed per row, never materialized. One coarse kernel instead of two.
(deftm skip-layer-norm
  (All [T] [a :- (Array T) b :- (Array T) gamma :- (Array T) beta :- (Array T)
            batch :- Long features :- Long eps :- Double] :- (Array T)
    (let [out  (alloc-like a (* batch features))
          finv (/ 1.0 (double features))]
      (dotimes [r batch]
        (let [offset (* r (int features))
              mean (loop [i 0 s 0.0]
                     (if (< i features)
                       (recur (inc i) (+ s (+ (aget a (+ offset i)) (aget b (+ offset i)))))
                       (* s finv)))
              var  (loop [i 0 s 0.0]
                     (if (< i features)
                       (let [d (- (+ (aget a (+ offset i)) (aget b (+ offset i))) mean)]
                         (recur (inc i) (+ s (* d d))))
                       (* s finv)))
              inv-std (/ 1.0 (n/sqrt (+ var eps)))]
          (dotimes [i features]
            (let [v (+ (aget a (+ offset i)) (aget b (+ offset i)))]
              (aset out (+ offset i)
                    (+ (* (aget gamma i) (* (- v mean) inv-std)) (aget beta i)))))))
      out)))

;; --- GELU: x * Phi(x) (tanh approximation) ---
;; The tanh is inlined as a vectorizable odd-rational approximation (Eigen ptanh:
;; clamp + polynomial in u², only +/*//min/max → a pure SIMD lane chain). A libm
;; m/tanh call stays scalar inside the broadcast (dtanh_stub) and blocks the map.
(deftm gelu (All [T] [x :- (Array T) n :- Long] :- (Array T)
                 (let [c (n/sqrt (/ 2.0 n/pi))]
                   (broadcast [x]
                     (let [u  (n/min 9.0 (n/max -9.0 (* c (+ x (* 0.044715 x x x)))))
                           u2 (* u u)
                           np (+ 4.89352455891786e-3
                                 (* u2 (+ 6.37261928875436e-4
                                          (* u2 (+ 1.48572235717979e-5
                                                   (* u2 (+ 5.12229709037114e-8
                                                            (* u2 (+ -8.60467152213735e-11
                                                                     (* u2 (+ 2.00018790482477e-13
                                                                              (* u2 -2.76076847742355e-16))))))))))))
                           dp (+ 4.89352518554385e-3
                                 (* u2 (+ 2.268434632439e-3
                                          (* u2 (+ 1.18534705686654e-4
                                                   (* u2 1.19825839466702e-6))))))
                           th (/ (* u np) dp)]
                       (* 0.5 x (+ 1.0 th)))))))

;; gelu': 0.5*(1+tanh(inner)) + 0.5*x*sech^2(inner)*c*(1+3*0.044715*x^2)
(deftm gelu-backward (All [T] [dy :- (Array T) x :- (Array T) n :- Long]
                          :- (Array T)
                          (let [dx (alloc-like dy n)
                                c (n/sqrt (/ 2.0 n/pi))]
                            (dotimes [i n]
                              (let [xi (aget x i)
                                    inner (* c (+ xi (* 0.044715 xi xi xi)))
                                    t (raster.math/tanh inner)
                                    sech2 (- 1.0 (* t t))
                                    grad (+ (* 0.5 (+ 1.0 t))
                                            (* 0.5 xi sech2 c (+ 1.0 (* 3.0 0.044715 xi xi))))]
                                (aset dx i (* (aget dy i) grad))))
                            dx)))

(tmpl/merge-into-template! 'raster.dl.nn/gelu
                           {:pullback-factory (fn [_result x n]
                                                (fn [dy]
                                                  [(gelu-backward dy x n) nil]))})

(tmpl/merge-into-template! 'raster.dl.nn/gelu
                           {:params '[x n] :adjoint 'dy
                            :grads-fn (fn [ctx [x n] _result adjoint gensym-fn]
                                        (let [dx (gensym-fn "dx")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/gelu-backward adjoint x n)])
                                           [dx nil]]))})

;; --- Sigmoid ---
(deftm sigmoid (All [T] [x :- (Array T) n :- Long] :- (Array T)
                    (broadcast [x] (/ 1.0 (+ 1.0 (m/exp (- x)))))))

;; sigmoid': s*(1-s)
(deftm sigmoid-backward (All [T] [dy :- (Array T) result :- (Array T) n :- Long]
                             :- (Array T)
                             (let [dx (alloc-like dy n)]
                               (dotimes [i n]
                                 (let [si (aget result i)]
                                   (aset dx i (* (aget dy i) si (- 1.0 si)))))
                               dx)))

(tmpl/merge-into-template! 'raster.dl.nn/sigmoid
                           {:pullback-factory (fn [result x n]
                                                (fn [dy]
                                                  [(sigmoid-backward dy result n) nil]))})

(tmpl/merge-into-template! 'raster.dl.nn/sigmoid
                           {:params '[x n] :adjoint 'dy
                            :grads-fn (fn [ctx [x n] result adjoint gensym-fn]
                                        (let [dx (gensym-fn "dx")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/sigmoid-backward adjoint result n)])
                                           [dx nil]]))})

;; --- Tanh activation ---
(deftm tanh-act (All [T] [x :- (Array T) n :- Long] :- (Array T)
                     (broadcast [x] (m/tanh x))))

;; tanh': 1 - tanh^2
(deftm tanh-act-backward (All [T] [dy :- (Array T) result :- (Array T) n :- Long]
                              :- (Array T)
                              (let [dx (alloc-like dy n)]
                                (dotimes [i n]
                                  (let [ti (aget result i)]
                                    (aset dx i (* (aget dy i) (- 1.0 (* ti ti))))))
                                dx)))

(tmpl/merge-into-template! 'raster.dl.nn/tanh-act
                           {:pullback-factory (fn [result x n]
                                                (fn [dy]
                                                  [(tanh-act-backward dy result n) nil]))})

(tmpl/merge-into-template! 'raster.dl.nn/tanh-act
                           {:params '[x n] :adjoint 'dy
                            :grads-fn (fn [ctx [x n] result adjoint gensym-fn]
                                        (let [dx (gensym-fn "dx")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/tanh-act-backward adjoint result n)])
                                           [dx nil]]))})

;; --- Leaky ReLU ---
(deftm leaky-relu (All [T] [x :- (Array T) n :- Long alpha :- Double]
                       :- (Array T)
                       (broadcast [x] (if (>= x 0.0) x (* alpha x)))))

;; In-place leaky-relu: writes into pre-allocated out
(deftm leaky-relu! (All [T] [x :- (Array T) out :- (Array T) n :- Long alpha :- Double]
                        :- (Array T)
                        (broadcast [x] (if (>= x 0.0) x (* alpha x)))))

;; leaky-relu': x>=0 ? 1 : alpha
(deftm leaky-relu-backward (All [T] [dy :- (Array T) x :- (Array T) n :- Long alpha :- Double]
                                :- (Array T)
                                (let [dx (alloc-like dy n)]
                                  (dotimes [i n]
                                    (let [xi (aget x i)
                                          grad (if (>= xi 0.0) 1.0 alpha)]
                                      (aset dx i (* (aget dy i) grad))))
                                  dx)))

(tmpl/merge-into-template! 'raster.dl.nn/leaky-relu
                           {:pullback-factory (fn [_result x n alpha]
                                                (fn [dy]
                                                  [(leaky-relu-backward dy x n alpha) nil nil]))})

(tmpl/merge-into-template! 'raster.dl.nn/leaky-relu
                           {:params '[x n alpha] :adjoint 'dy
                            :grads-fn (fn [ctx [x n alpha] _result adjoint gensym-fn]
                                        (let [dx (gensym-fn "dx")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/leaky-relu-backward adjoint x n alpha)])
                                           [dx nil nil]]))})

;; ================================================================
;; Normalization
;; ================================================================

;; --- Layer Norm ---
;; x:[batch, features], gamma:[features], beta:[features]
(deftm layer-norm (All [T] [x :- (Array T) gamma :- (Array T)
                            beta :- (Array T) batch :- Long features :- Long
                            eps :- Double] :- (Array T)
                       (let [out (alloc-like x (* batch features))]
                         (dotimes [b batch]
                           (let [offset (* b (int features))
            ;; compute mean
                                 mean (loop [i 0 s 0.0]
                                        (if (< i features)
                                          (recur (inc i) (+ s (aget x (+ offset i))))
                                          (/ s features)))
            ;; compute variance
                                 var (loop [i 0 s 0.0]
                                       (if (< i features)
                                         (let [d (- (aget x (+ offset i)) mean)]
                                           (recur (inc i) (+ s (* d d))))
                                         (/ s features)))
                                 inv-std (/ 1.0 (n/sqrt (+ var eps)))]
                             (dotimes [i features]
                               (let [x-hat (* (- (aget x (+ offset i)) mean) inv-std)]
                                 (aset out (+ offset i)
                                       (+ (* (aget gamma i) x-hat)
                                          (aget beta i)))))))
                         out)))

;; !-variant for resident GPU programs: one work-item per row, writes into a
;; pre-allocated out buffer (same convention as rms-norm!). The AuT audio
;; encoder (Qwen3-ASR) and BERT-family layers use this (LN with bias), where
;; the decoder LMs use rms-norm!.
(deftm layer-norm! (All [T] [x :- (Array T) gamma :- (Array T) beta :- (Array T)
                             out :- (Array T) rows :- Long features :- Long
                             eps :- Double] :- Void
  (raster.par/map-void! r rows
    (let [offset (clojure.core/* r features)
          mean (loop [i 0 s 0.0]
                 (if (< i features)
                   (recur (inc i) (+ s (aget x (clojure.core/+ offset i))))
                   (/ s (double features))))
          var (loop [i 0 s 0.0]
                (if (< i features)
                  (let [dv (- (aget x (clojure.core/+ offset i)) mean)]
                    (recur (inc i) (+ s (* dv dv))))
                  (/ s (double features))))
          inv (/ 1.0 (n/sqrt (+ var eps)))]
      (loop [i 0]
        (if (< i features)
          (do (aset out (clojure.core/+ offset i)
                    (+ (* (aget gamma i)
                          (* (- (aget x (clojure.core/+ offset i)) mean) inv))
                       (aget beta i)))
              (recur (inc i)))
          nil))))))

;; !-variant elementwise GELU (tanh approximation, matches `gelu`/gelu-mul!).
(deftm gelu! (All [T] [x :- (Array T) out :- (Array T) n :- Long] :- Void
  (raster.par/map-void! i n
    (let [v (aget x i)]
      (aset out i (* 0.5 v
                     (+ 1.0 (m/tanh (* 0.7978845608028654
                                       (+ v (* 0.044715 v v v)))))))))))

;; Broadcast row-bias add in place semantics via separate out (resident-graph
;; friendly): out[r,j] = x[r,j] + b[j]. Follows every biased linear in
;; encoder-family layers (AuT, BERT) — the quantized GEMM kernels are bias-free.
(deftm add-bias-rows! (All [T] [x :- (Array T) b :- (Array T) out :- (Array T)
                                rows :- Long features :- Long] :- Void
  (raster.par/map-void! i (* rows features)
    (aset out i (+ (aget x i) (aget b (clojure.core/rem i features)))))))

;; layer-norm rrule — pullback registered after backward deftm (below)

;; --- RMS Norm ---
;; Root-mean-square normalization (no mean subtraction, no bias). Used by the
;; modern decoder LMs (Llama, Qwen, Gemma, Mistral). Per-row over `features`:
;;   y_i = x_i / sqrt(mean_j(x_j^2) + eps) * (gain-offset + weight_i)
;; gain-offset = 0.0 for Llama/Qwen (plain weight gain); 1.0 for Gemma, whose
;; norm weights are centered at 0 and applied as (1 + weight). Also used for
;; per-head QK-norm (rows = heads*seq, features = head_dim).
;; par-combinator primitive: parallel-map over the (independent) row dimension, scalar
;; reduce+map inside. This is the form that vectorizes on CPU (the inner feature loops lift
;; to SIMD) and is the GPU-friendly shape (par/map-void! → one work-item/row, out caller-
;; provided), so one source serves prefill, decode, and (eventually) the GPU-resident graph.
;; Index arithmetic uses clojure.core (integer subscripts); float compute uses raster.numeric.
;; Validated on both CPU and the OpenCL/GPU lowering (maxerr ~5e-7 vs CPU).
(deftm rms-norm! (All [T] [x :- (Array T) weight :- (Array T) out :- (Array T)
                           rows :- Long features :- Long
                           eps :- Double gain-offset :- Double] :- Void
                      (raster.par/map-void! r rows
                        ;; index arithmetic stays clojure.core (integer); only float compute
                        ;; goes through raster.numeric (devirtualizes + vectorizes).
                                            (let [offset (clojure.core/* r features)
                                                  ms (loop [i 0 s 0.0]
                                                       (if (< i features)
                                                         (let [v (aget x (clojure.core/+ offset i))]
                                                           (recur (inc i) (+ s (* v v))))
                                                         (/ s (double features))))
                                                  inv (/ 1.0 (n/sqrt (+ ms eps)))]
                                              (loop [i 0]
                                                (if (< i features)
                                                  (do (aset out (clojure.core/+ offset i)
                                                            (* (aget x (clojure.core/+ offset i)) inv
                                                               (+ gain-offset (aget weight i))))
                                                      (recur (inc i)))
                                                  nil))))))

(deftm rms-norm (All [T] [x :- (Array T) weight :- (Array T)
                          rows :- Long features :- Long
                          eps :- Double gain-offset :- Double] :- (Array T)
                     (let [out (alloc-like x (* rows features))]
                       (rms-norm! x weight out rows features eps gain-offset)
                       out)))

;; --- Bias-free linear + gated MLP (modern decoder LMs) ---
;; Modern LLMs (Llama, Qwen, Gemma, Mistral) use bias-free projections.
;; linear-nb: y = x @ W^T, W:[out_f,in_f] (HF layout), no bias.
(deftm linear-nb (All [T] [x :- (Array T) W :- (Array T)
                           batch :- Long in-f :- Long out-f :- Long] :- (Array T)
                      (let [y (alloc-like x (* batch out-f))]
                        (blas/dgemm-nt! x W y batch in-f out-f
                                        (n/oftype x 1.0) (n/oftype x 0.0))
                        y)))

;; Elementwise (Hadamard) product a ⊙ b, with an explicit AD rule. A raw
;; (broadcast [a b] (* a b)) computes the same value but has NO AD template for
;; TWO active inputs (the SOAC map differentiates only single-active), so gated
;; MLPs need this dedicated op. Pullback: d_a = dy⊙b, d_b = dy⊙a.
(deftm hadamard (All [T] [a :- (Array T) b :- (Array T) n :- Long] :- (Array T)
                    (broadcast [a b] (* a b))))

(deftm hadamard-backward (All [T] [dy :- (Array T) other :- (Array T) n :- Long] :- (Array T)
                             (broadcast [dy other] (* dy other))))

;; Gated MLP (GeGLU): down( gelu(x@gate^T) ⊙ (x@up^T) ), bias-free.
;; This is Gemma's / T5-v1.1's MLP. The SwiGLU variant (Llama/Qwen) is identical
;; with silu in place of gelu — see swiglu.
(deftm geglu (All [T] [x :- (Array T) gate-w :- (Array T) up-w :- (Array T)
                       down-w :- (Array T)
                       rows :- Long d-model :- Long d-ff :- Long] :- (Array T)
                  (let [g (linear-nb x gate-w rows d-model d-ff)
                        g (gelu g (* rows d-ff))
                        u (linear-nb x up-w rows d-model d-ff)
                        h (hadamard g u (* rows d-ff))]
                    (linear-nb h down-w rows d-ff d-model))))

;; SwiGLU (Llama/Qwen gated MLP): down( silu(x@gate^T) ⊙ (x@up^T) ), bias-free.
(deftm swiglu (All [T] [x :- (Array T) gate-w :- (Array T) up-w :- (Array T)
                        down-w :- (Array T)
                        rows :- Long d-model :- Long d-ff :- Long] :- (Array T)
                   (let [g (linear-nb x gate-w rows d-model d-ff)
                         g (silu g (* rows d-ff))
                         u (linear-nb x up-w rows d-model d-ff)
                         h (hadamard g u (* rows d-ff))]
                     (linear-nb h down-w rows d-ff d-model))))

;; Elementwise residual add: out = a + b (parametric; for transformer residuals).
(deftm residual-add (All [T] [a :- (Array T) b :- (Array T) n :- Long] :- (Array T)
                         (broadcast [a b] (+ a b))))

;; --- Group Norm ---
;; x:[batch, channels, spatial], gamma:[channels], beta:[channels]
(deftm group-norm (All [T] [x :- (Array T) gamma :- (Array T)
                            beta :- (Array T) batch :- Long channels :- Long
                            spatial :- Long groups :- Long eps :- Double]
                       :- (Array T)
                       (let [out (alloc-like x (* batch channels spatial))
                             cpg (quot channels groups)  ;; channels per group
                             group-size (* cpg spatial)]
                         (dotimes [b batch]
                           (dotimes [g groups]
                             (let [;; compute mean and var for this group
                                   mean (loop [c 0 acc 0.0]
                                          (if (< c cpg)
                                            (let [ch (+ (* g (int cpg)) c)
                                                  acc (loop [sp 0 inner-acc acc]
                                                        (if (< sp spatial)
                                                          (recur (inc sp)
                                                                 (+ inner-acc (aget x
                                                                                    (+ (* b (int (* channels spatial)))
                                                                                       (* ch (int spatial))
                                                                                       sp))))
                                                          inner-acc))]
                                              (recur (inc c) acc))
                                            (/ acc group-size)))
                                   var (loop [c 0 acc 0.0]
                                         (if (< c cpg)
                                           (let [ch (+ (* g (int cpg)) c)
                                                 acc (loop [sp 0 inner-acc acc]
                                                       (if (< sp spatial)
                                                         (let [idx (+ (* b (int (* channels spatial)))
                                                                      (* ch (int spatial)) sp)
                                                               d (- (aget x idx) mean)]
                                                           (recur (inc sp) (+ inner-acc (* d d))))
                                                         inner-acc))]
                                             (recur (inc c) acc))
                                           (/ acc group-size)))
                                   inv-std (/ 1.0 (n/sqrt (+ var eps)))]
          ;; normalize and scale
                               (dotimes [c cpg]
                                 (let [ch (+ (* g (int cpg)) c)]
                                   (dotimes [sp spatial]
                                     (let [idx (+ (* b (int (* channels spatial)))
                                                  (* ch (int spatial)) sp)
                                           x-hat (* (- (aget x idx) mean) inv-std)]
                                       (aset out idx
                                             (+ (* (aget gamma ch) x-hat)
                                                (aget beta ch))))))))))
                         out)))

;; group-norm rrule — pullback registered after backward deftm (below)

;; --- Batch Norm ---
;; x:[batch, features], gamma:[features], beta:[features]
;; training=1 uses batch stats, training=0 uses running stats
(deftm batch-norm (All [T] [x :- (Array T) gamma :- (Array T)
                            beta :- (Array T) running-mean :- (Array T)
                            running-var :- (Array T) batch :- Long
                            features :- Long eps :- Double momentum :- Double
                            training :- Long] :- (Array T)
                       (let [out (alloc-like x (* batch features))]
                         (if (== training 1)
      ;; Training: use batch statistics
                           (dotimes [j features]
                             (let [;; batch mean
                                   mean (loop [i 0 s 0.0]
                                          (if (< i batch)
                                            (recur (inc i) (+ s (aget x (+ (* i (int features)) j))))
                                            (/ s batch)))
              ;; batch variance
                                   var (loop [i 0 s 0.0]
                                         (if (< i batch)
                                           (let [d (- (aget x (+ (* i (int features)) j)) mean)]
                                             (recur (inc i) (+ s (* d d))))
                                           (/ s batch)))
                                   inv-std (/ 1.0 (n/sqrt (+ var eps)))]
          ;; update running stats
                               (aset running-mean j
                                     (+ (* (- 1.0 momentum) (aget running-mean j))
                                        (* momentum mean)))
                               (aset running-var j
                                     (+ (* (- 1.0 momentum) (aget running-var j))
                                        (* momentum var)))
          ;; normalize
                               (dotimes [i batch]
                                 (let [idx (+ (* i (int features)) j)
                                       x-hat (* (- (aget x idx) mean) inv-std)]
                                   (aset out idx
                                         (+ (* (aget gamma j) x-hat)
                                            (aget beta j)))))))
      ;; Eval: use running statistics
                           (dotimes [j features]
                             (let [inv-std (/ 1.0 (n/sqrt (+ (aget running-var j) eps)))]
                               (dotimes [i batch]
                                 (let [idx (+ (* i (int features)) j)
                                       x-hat (* (- (aget x idx)
                                                   (aget running-mean j))
                                                inv-std)]
                                   (aset out idx
                                         (+ (* (aget gamma j) x-hat)
                                            (aget beta j))))))))
                         out)))

;; batch-norm rrule — pullback registered after backward deftm (below)

;; ================================================================
;; Convolution helpers: im2col
;; ================================================================

;; im2col-1d: rearrange input patches into columns for GEMM
;; x:[batch, c_in, length] -> cols:[c_in*kernel, batch*l_out]
(deftm im2col-1d (All [T] [x :- (Array T) batch :- Long c-in :- Long
                           length :- Long kernel :- Long stride :- Long
                           pad :- Long] :- (Array T)
                      (let [l-out (+ 1 (quot (+ length (* 2 pad) (- kernel)) stride))
                            col-rows (* c-in kernel)
                            col-cols (* batch l-out)
                            cols (alloc-like x (* col-rows col-cols))]
                        (dotimes [b batch]
                          (dotimes [c c-in]
                            (dotimes [kk kernel]
                              (dotimes [out-pos l-out]
                                (let [in-pos (+ (- (* out-pos (int stride)) pad) kk)
                                      row (+ (* c (int kernel)) kk)
                                      col (+ (* b (int l-out)) out-pos)]
                                  (when (and (>= in-pos 0) (< in-pos length))
                                    (aset cols (+ (* row col-cols) col)
                                          (aget x
                                                (+ (* b (int (* c-in length)))
                                                   (* c (int length))
                                                   in-pos)))))))))
                        cols)))

;; col2im-1d: reverse of im2col (accumulate)
;; cols:[c_in*kernel, batch*l_out] -> dx:[batch, c_in, length]
(deftm col2im-1d (All [T] [cols :- (Array T) batch :- Long c-in :- Long
                           length :- Long kernel :- Long stride :- Long
                           pad :- Long] :- (Array T)
                      (let [l-out (+ 1 (quot (+ length (* 2 pad) (- kernel)) stride))
                            col-cols (* batch l-out)
                            dx (alloc-like cols (* batch c-in length))]
                        (dotimes [b batch]
                          (dotimes [c c-in]
                            (dotimes [kk kernel]
                              (dotimes [out-pos l-out]
                                (let [in-pos (+ (- (* out-pos (int stride)) pad) kk)]
                                  (when (and (>= in-pos 0) (< in-pos length))
                                    (let [row (+ (* c (int kernel)) kk)
                                          col (+ (* b (int l-out)) out-pos)
                                          x-idx (+ (* b (int (* c-in length)))
                                                   (* c (int length)) in-pos)]
                                      (aset dx x-idx
                                            (+ (aget dx x-idx)
                                               (aget cols (+ (* row col-cols) col)))))))))))
                        dx)))

;; ================================================================
;; Conv1d: x:[B,C_in,L], W:[C_out,C_in,K], b:[C_out] -> y:[B,C_out,L_out]
;; ================================================================

(deftm conv1d (All [T] [x :- (Array T) W :- (Array T) b :- (Array T)
                        batch :- Long c-in :- Long length :- Long
                        c-out :- Long kernel :- Long stride :- Long
                        pad :- Long] :- (Array T)
                   (let [l-out (+ 1 (quot (+ length (* 2 pad) (- kernel)) stride))
        ;; im2col: x -> cols:[c_in*kernel, batch*l_out]
                         cols (im2col-1d x batch c-in length kernel stride pad)
        ;; GEMM: W_reshaped:[c_out, c_in*kernel] @ cols -> y_cols:[c_out, batch*l_out]
                         ck (* c-in kernel)
                         bl (* batch l-out)
                         y-cols (matmul W cols c-out ck bl)
        ;; Rearrange y_cols[c_out, batch*l_out] -> y[batch, c_out, l_out]
        ;; and add bias
                         y (alloc-like x (* batch c-out l-out))]
                     (dotimes [co c-out]
                       (dotimes [bi batch]
                         (dotimes [li l-out]
                           (let [src-idx (+ (* co (int bl)) (* bi (int l-out)) li)
                                 dst-idx (+ (* bi (int (* c-out l-out)))
                                            (* co (int l-out)) li)]
                             (aset y dst-idx
                                   (+ (aget y-cols src-idx)
                                      (aget b co)))))))
                     y)))

;; conv1d rrule — pullback registered after backward deftm (below)

;; ================================================================
;; im2col-2d and Conv2d
;; ================================================================

;; im2col-2d: x:[B,C_in,H,W] -> cols:[C_in*kH*kW, B*H_out*W_out]
(deftm ^:no-inline im2col-2d (All [T] [x :- (Array T) batch :- Long c-in :- Long
                                       h :- Long w :- Long kh :- Long kw :- Long
                                       stride-h :- Long stride-w :- Long
                                       pad-h :- Long pad-w :- Long] :- (Array T)
                                  (let [h-out (+ 1 (quot (+ h (* 2 pad-h) (- kh)) stride-h))
                                        w-out (+ 1 (quot (+ w (* 2 pad-w) (- kw)) stride-w))
                                        col-rows (* c-in kh kw)
                                        col-cols (* batch h-out w-out)
                                        cols (alloc-like x (* col-rows col-cols))]
                                    (dotimes [bi batch]
                                      (dotimes [c c-in]
                                        (dotimes [khi kh]
                                          (dotimes [kwi kw]
                                            (dotimes [oh h-out]
                                              (dotimes [ow w-out]
                                                (let [ih (+ (- (* oh (int stride-h)) pad-h) khi)
                                                      iw (+ (- (* ow (int stride-w)) pad-w) kwi)]
                                                  (when (and (>= ih 0) (< ih h) (>= iw 0) (< iw w))
                                                    (let [row (+ (* c (int (* kh kw))) (* khi (int kw)) kwi)
                                                          col (+ (* bi (int (* h-out w-out)))
                                                                 (* oh (int w-out)) ow)
                                                          x-idx (+ (* bi (int (* c-in h w)))
                                                                   (* c (int (* h w)))
                                                                   (* ih (int w)) iw)]
                                                      (aset cols (+ (* row col-cols) col)
                                                            (aget x x-idx)))))))))))
                                    cols)))

;; In-place im2col-2d: writes into pre-allocated cols buffer
;; Fast path: stride=1, pad=0 uses acopy! for contiguous rows
(deftm ^:no-inline im2col-2d! (All [T] [x :- (Array T) cols :- (Array T)
                                        batch :- Long c-in :- Long
                                        h :- Long w :- Long kh :- Long kw :- Long
                                        stride-h :- Long stride-w :- Long
                                        pad-h :- Long pad-w :- Long] :- (Array T)
                                   (let [h-out (+ 1 (quot (+ h (* 2 pad-h) (- kh)) stride-h))
                                         w-out (+ 1 (quot (+ w (* 2 pad-w) (- kw)) stride-w))
                                         col-cols (* batch h-out w-out)]
                                     (if (and (== stride-h 1) (== stride-w 1) (== pad-h 0) (== pad-w 0))
      ;; FAST PATH: contiguous copy with acopy!
                                       (let [hw (* h w)
                                             chw (* c-in hw)
                                             hw-out (* h-out w-out)]
                                         (dotimes [c c-in]
                                           (dotimes [khi kh]
                                             (dotimes [kwi kw]
                                               (let [row (+ (* c (int (* kh kw))) (* khi (int kw)) kwi)]
                                                 (dotimes [bi batch]
                                                   (dotimes [oh h-out]
                    ;; Copy w-out contiguous elements from x to cols
                                                     (let [x-start (+ (* bi (int chw)) (* c (int hw))
                                                                      (* (+ khi oh) (int w)) kwi)
                                                           col-start (+ (* row (int col-cols))
                                                                        (* bi (int hw-out))
                                                                        (* oh (int w-out)))]
                                                       (acopy! x x-start cols col-start w-out)))))))))
      ;; GENERIC PATH: element-by-element with bounds checking
                                       (dotimes [bi batch]
                                         (dotimes [c c-in]
                                           (dotimes [khi kh]
                                             (dotimes [kwi kw]
                                               (dotimes [oh h-out]
                                                 (dotimes [ow w-out]
                                                   (let [ih (+ (- (* oh (int stride-h)) pad-h) khi)
                                                         iw (+ (- (* ow (int stride-w)) pad-w) kwi)]
                                                     (when (and (>= ih 0) (< ih h) (>= iw 0) (< iw w))
                                                       (let [row (+ (* c (int (* kh kw))) (* khi (int kw)) kwi)
                                                             col (+ (* bi (int (* h-out w-out)))
                                                                    (* oh (int w-out)) ow)
                                                             x-idx (+ (* bi (int (* c-in h w)))
                                                                      (* c (int (* h w)))
                                                                      (* ih (int w)) iw)]
                                                         (aset cols (+ (* row col-cols) col)
                                                               (aget x x-idx))))))))))))
                                     cols)))

;; col2im-2d: reverse of im2col-2d
(deftm ^:no-inline col2im-2d (All [T] [cols :- (Array T) batch :- Long c-in :- Long
                                       h :- Long w :- Long kh :- Long kw :- Long
                                       stride-h :- Long stride-w :- Long
                                       pad-h :- Long pad-w :- Long] :- (Array T)
                                  (let [h-out (+ 1 (quot (+ h (* 2 pad-h) (- kh)) stride-h))
                                        w-out (+ 1 (quot (+ w (* 2 pad-w) (- kw)) stride-w))
                                        col-cols (* batch h-out w-out)
                                        dx (alloc-like cols (* batch c-in h w))]
                                    (dotimes [bi batch]
                                      (dotimes [c c-in]
                                        (dotimes [khi kh]
                                          (dotimes [kwi kw]
                                            (dotimes [oh h-out]
                                              (dotimes [ow w-out]
                                                (let [ih (+ (- (* oh (int stride-h)) pad-h) khi)
                                                      iw (+ (- (* ow (int stride-w)) pad-w) kwi)]
                                                  (when (and (>= ih 0) (< ih h) (>= iw 0) (< iw w))
                                                    (let [row (+ (* c (int (* kh kw))) (* khi (int kw)) kwi)
                                                          col (+ (* bi (int (* h-out w-out)))
                                                                 (* oh (int w-out)) ow)
                                                          x-idx (+ (* bi (int (* c-in h w)))
                                                                   (* c (int (* h w)))
                                                                   (* ih (int w)) iw)]
                                                      (aset dx x-idx
                                                            (+ (aget dx x-idx)
                                                               (aget cols (+ (* row col-cols) col)))))))))))))
                                    dx)))

;; In-place col2im-2d: writes into pre-allocated dx
;; Fast path: stride=1, pad=0 avoids bounds checking
(deftm ^:no-inline col2im-2d! (All [T] [cols :- (Array T) dx :- (Array T)
                                        batch :- Long c-in :- Long
                                        h :- Long w :- Long kh :- Long kw :- Long
                                        stride-h :- Long stride-w :- Long
                                        pad-h :- Long pad-w :- Long] :- (Array T)
                                   (let [h-out (+ 1 (quot (+ h (* 2 pad-h) (- kh)) stride-h))
                                         w-out (+ 1 (quot (+ w (* 2 pad-w) (- kw)) stride-w))
                                         col-cols (* batch h-out w-out)
                                         n-dx (* batch c-in h w)]
    ;; Zero the output (polymorphic — works for double[] and float[])
                                     (dotimes [i n-dx] (aset dx i 0.0))
                                     (if (and (== stride-h 1) (== stride-w 1) (== pad-h 0) (== pad-w 0))
      ;; FAST PATH: no bounds checking, accumulate directly
                                       (let [hw (* h w)
                                             chw (* c-in hw)
                                             hw-out (* h-out w-out)]
                                         (dotimes [c c-in]
                                           (dotimes [khi kh]
                                             (dotimes [kwi kw]
                                               (let [row (+ (* c (int (* kh kw))) (* khi (int kw)) kwi)]
                                                 (dotimes [bi batch]
                                                   (dotimes [oh h-out]
                                                     (dotimes [ow w-out]
                                                       (let [x-idx (+ (* bi (int chw)) (* c (int hw))
                                                                      (* (+ khi oh) (int w)) (+ kwi ow))
                                                             col-idx (+ (* row (int col-cols))
                                                                        (* bi (int hw-out))
                                                                        (* oh (int w-out)) ow)]
                                                         (aset dx x-idx
                                                               (+ (aget dx x-idx)
                                                                  (aget cols col-idx))))))))))))
      ;; GENERIC PATH
                                       (dotimes [bi batch]
                                         (dotimes [c c-in]
                                           (dotimes [khi kh]
                                             (dotimes [kwi kw]
                                               (dotimes [oh h-out]
                                                 (dotimes [ow w-out]
                                                   (let [ih (+ (- (* oh (int stride-h)) pad-h) khi)
                                                         iw (+ (- (* ow (int stride-w)) pad-w) kwi)]
                                                     (when (and (>= ih 0) (< ih h) (>= iw 0) (< iw w))
                                                       (let [row (+ (* c (int (* kh kw))) (* khi (int kw)) kwi)
                                                             col (+ (* bi (int (* h-out w-out)))
                                                                    (* oh (int w-out)) ow)
                                                             x-idx (+ (* bi (int (* c-in h w)))
                                                                      (* c (int (* h w)))
                                                                      (* ih (int w)) iw)]
                                                         (aset dx x-idx
                                                               (+ (aget dx x-idx)
                                                                  (aget cols (+ (* row col-cols) col))))))))))))))
                                     dx)))

;; Conv2d: x:[B,C_in,H,W], W:[C_out,C_in,kH,kW], b:[C_out]
(deftm ^:no-inline conv2d (All [T] [x :- (Array T) W :- (Array T) b :- (Array T)
                                    batch :- Long c-in :- Long h :- Long w :- Long
                                    c-out :- Long kh :- Long kw :- Long
                                    stride-h :- Long stride-w :- Long
                                    pad-h :- Long pad-w :- Long] :- (Array T)
                               (let [h-out (+ 1 (quot (+ h (* 2 pad-h) (- kh)) stride-h))
                                     w-out (+ 1 (quot (+ w (* 2 pad-w) (- kw)) stride-w))
                                     ckk (* c-in kh kw)
                                     bhw (* batch h-out w-out)
        ;; im2col
                                     cols (im2col-2d x batch c-in h w kh kw stride-h stride-w pad-h pad-w)
                                     hw-out (* h-out w-out)
                                     chw-out (* c-out hw-out)
        ;; GEMM: W:[c_out, ckk] @ cols:[ckk, bhw] -> y_cols:[c_out, bhw]
                                     y-cols (matmul W cols c-out ckk bhw)
        ;; Rearrange y_cols[c_out, B*hw] -> y[B, c_out, hw] + fused bias add
        ;; Tight copy+bias per (co, bi) block — sequential memory, vectorizable
                                     y (alloc-like x (* batch c-out hw-out))]
                                 (dotimes [co c-out]
                                   (let [bias-val (aget b co)]
                                     (dotimes [bi batch]
                                       (let [src-base (+ (* co (int bhw)) (* bi (int hw-out)))
                                             dst-base (+ (* bi (int chw-out)) (* co (int hw-out)))]
                                         (dotimes [j hw-out]
                                           (aset y (+ dst-base j)
                                                 (+ (aget y-cols (+ src-base j)) bias-val)))))))
                                 y)))

;; conv2d rrule — uses dedicated in-place helpers for rearrange/gemm/bias
;; conv2d rrule is registered after the backward helper deftms (see below)

;; ================================================================
;; MaxPool2d
;; x:[batch, channels, h, w] -> y:[batch, channels, h/kh, w/kw]
;; Non-overlapping pooling windows (stride = kernel size)
;; ================================================================

(deftm ^:no-inline maxpool2d (All [T] [x :- (Array T) batch :- Long channels :- Long
                                       h :- Long w :- Long kh :- Long kw :- Long]
                                  :- (Array T)
                                  (let [h-out (quot h kh)
                                        w-out (quot w kw)
                                        chw (* channels h w)
                                        hw (* h w)
                                        chw-out (* channels h-out w-out)
                                        hw-out (* h-out w-out)
                                        w-int (int w)
                                        kh-int (int kh)
                                        kw-int (int kw)
                                        w-out-int (int w-out)
                                        out (alloc-like x (* batch chw-out))
        ;; Single holder reused across all output positions (same type as x)
                                        holder (alloc-like x 1)]
                                    (dotimes [bi batch]
                                      (dotimes [ci channels]
                                        (let [base-in (+ (* bi (int chw)) (* ci (int hw)))
                                              base-out (+ (* bi (int chw-out)) (* ci (int hw-out)))]
                                          (dotimes [oh h-out]
                                            (dotimes [ow w-out]
              ;; Scan kernel window using nested dotimes (no loop/recur)
                                              (let [ih0 (* oh kh-int)
                                                    iw0 (* ow kw-int)]
                ;; Reset holder
                                                (aset holder 0 Double/NEGATIVE_INFINITY)
                                                (dotimes [khi kh]
                                                  (dotimes [kwi kw]
                                                    (let [x-idx (+ base-in (* (+ ih0 khi) w-int) (+ iw0 kwi))
                                                          v (aget x x-idx)]
                                                      (when (> v (aget holder 0))
                                                        (aset holder 0 v)))))
                                                (aset out (+ base-out (* oh w-out-int) ow)
                                                      (aget holder 0))))))))
                                    out)))

;; In-place maxpool2d: writes into pre-allocated out, also fills argmax for backward
(deftm ^:no-inline maxpool2d! (All [T] [x :- (Array T) out :- (Array T) argmax :- (Array long)
                                        batch :- Long channels :- Long
                                        h :- Long w :- Long kh :- Long kw :- Long]
                                   :- (Array T)
                                   (let [h-out (quot h kh)
                                         w-out (quot w kw)
                                         chw (* channels h w)
                                         hw (* h w)
                                         chw-out (* channels h-out w-out)
                                         hw-out (* h-out w-out)
                                         w-int (int w)
                                         kh-int (int kh)
                                         kw-int (int kw)
                                         w-out-int (int w-out)
        ;; Single holder pair reused across all output positions (same type as x)
                                         holder (alloc-like x 1)
                                         iholder (long-array 1)]
                                     (dotimes [bi batch]
                                       (dotimes [ci channels]
                                         (let [base-in (+ (* bi (int chw)) (* ci (int hw)))
                                               base-out (+ (* bi (int chw-out)) (* ci (int hw-out)))]
                                           (dotimes [oh h-out]
                                             (dotimes [ow w-out]
                                               (let [ih0 (* oh kh-int)
                                                     iw0 (* ow kw-int)]
                ;; Reset holders
                                                 (aset holder 0 Double/NEGATIVE_INFINITY)
                                                 (aset iholder 0 0)
                                                 (dotimes [khi kh]
                                                   (dotimes [kwi kw]
                                                     (let [x-idx (+ base-in (* (+ ih0 khi) w-int) (+ iw0 kwi))
                                                           v (aget x x-idx)]
                                                       (when (> v (aget holder 0))
                                                         (aset holder 0 v)
                                                         (aset iholder 0 (long x-idx))))))
                                                 (let [oi (+ base-out (* oh w-out-int) ow)]
                                                   (aset out oi (aget holder 0))
                                                   (aset argmax oi (aget iholder 0)))))))))
                                     out)))

;; In-place maxpool2d backward: uses pre-computed argmax, writes into pre-allocated dx
(deftm ^:no-inline maxpool2d-bwd! (All [T] [dy :- (Array T) argmax :- (Array long)
                                            dx :- (Array T) n-out :- Long n-in :- Long]
                                       :- (Array T)
  ;; Zero dx
                                       (dotimes [i n-in] (aset dx i 0.0))
  ;; Scatter gradients
                                       (dotimes [i n-out]
                                         (let [idx (aget argmax i)
                                               grad (aget dy i)]
                                           (aset dx idx (+ (aget dx idx) grad))))
                                       dx))

;; ================================================================
;; Conv2d backward helpers for template-based AD
;; ================================================================

;; Rearrange dy[B,c_out,h_out,w_out] -> dy_cols[c_out, B*h_out*w_out]
(deftm ^:no-inline conv2d-rearrange-dy! (All [T]
                                             [dy :- (Array T) dy-cols :- (Array T)
                                              batch :- Long c-out :- Long h-out :- Long w-out :- Long] :- (Array T)
                                             (let [hw-out (* h-out w-out)
                                                   chw-out (* c-out hw-out)
                                                   bhw (* batch hw-out)]
    ;; Rearrange dy[B, c_out, hw] -> dy_cols[c_out, B*hw]
    ;; Tight copy per (co, bi) block — sequential memory access
                                               (dotimes [co c-out]
                                                 (dotimes [bi batch]
                                                   (let [src-base (+ (* bi (int chw-out)) (* co (int hw-out)))
                                                         dst-base (+ (* co (int bhw)) (* bi (int hw-out)))]
                                                     (System/arraycopy dy src-base dy-cols dst-base hw-out)))))
                                             dy-cols))

;; dW = dy_cols @ cols^T via BLAS NT
(deftm ^:no-inline conv2d-backward-dW-into! (All [T]
                                                 [dy-cols :- (Array T) cols :- (Array T) dW :- (Array T)
                                                  c-out :- Long bhw :- Long ckk :- Long] :- (Array T)
                                                 (blas/dgemm-nt! dy-cols cols dW c-out bhw ckk (n/oftype dy-cols 1.0) (n/oftype dy-cols 0.0))))

;; d_cols = W^T @ dy_cols via BLAS TN
(deftm ^:no-inline conv2d-backward-dcols-into! (All [T]
                                                    [W :- (Array T) dy-cols :- (Array T) d-cols :- (Array T)
                                                     ckk :- Long c-out :- Long bhw :- Long] :- (Array T)
                                                    (blas/dgemm-tn! W dy-cols d-cols ckk c-out bhw (n/oftype W 1.0) (n/oftype W 0.0))))

;; db = sum dy over spatial dims — tight 2-deep loop
(deftm ^:no-inline conv2d-backward-db-into! (All [T]
                                                 [dy :- (Array T) db :- (Array T)
                                                  batch :- Long c-out :- Long h-out :- Long w-out :- Long] :- (Array T)
                                                 (let [hw-out (* h-out w-out)
                                                       chw-out (* c-out hw-out)]
                                                   (dotimes [i c-out] (aset db i 0.0))
                                                   (dotimes [bi batch]
                                                     (dotimes [co c-out]
                                                       (let [base (+ (* bi (int chw-out)) (* co (int hw-out)))]
                                                         (dotimes [j hw-out]
                                                           (aset db co (+ (aget db co) (aget dy (+ base j))))))))
                                                   db)))

;; conv2d rrule — uses dedicated in-place helpers for rearrange/gemm/bias
(tmpl/merge-into-template! 'raster.dl.nn/conv2d
                           {:pullback-factory (fn [_result x W _b batch c-in h w c-out kh kw stride-h stride-w pad-h pad-w]
                                                (fn [dy]
                                                  (let [batch (long batch) c-in (long c-in) h (long h) w (long w)
                                                        c-out (long c-out) kh (long kh) kw (long kw)
                                                        stride-h (long stride-h) stride-w (long stride-w)
                                                        pad-h (long pad-h) pad-w (long pad-w)
                                                        h-out (+ 1 (quot (+ h (* 2 pad-h) (- kh)) stride-h))
                                                        w-out (+ 1 (quot (+ w (* 2 pad-w) (- kw)) stride-w))
                                                        ckk (* c-in kh kw)
                                                        bhw (* batch h-out w-out)
                               ;; rearrange dy[B,c_out,h_out,w_out] -> dy_cols[c_out, bhw]
                                                        dy-cols (alloc-like dy (* c-out bhw))
                                                        _ (conv2d-rearrange-dy! dy dy-cols batch c-out h-out w-out)
                               ;; im2col for dx computation
                                                        cols (im2col-2d x batch c-in h w kh kw stride-h stride-w pad-h pad-w)
                               ;; dW = dy_cols @ cols^T
                                                        dW (alloc-like dy (* c-out ckk))
                                                        _ (conv2d-backward-dW-into! dy-cols cols dW c-out bhw ckk)
                               ;; d_cols = W^T @ dy_cols
                                                        d-cols (alloc-like dy (* ckk bhw))
                                                        _ (conv2d-backward-dcols-into! W dy-cols d-cols ckk c-out bhw)
                               ;; col2im to get dx
                                                        dx (col2im-2d d-cols batch c-in h w kh kw stride-h stride-w pad-h pad-w)
                               ;; db = sum of dy over spatial dims
                                                        db (alloc-like dy c-out)
                                                        _ (conv2d-backward-db-into! dy db batch c-out h-out w-out)]
                                                    [dx dW db nil nil nil nil nil nil nil nil nil nil])))})

;; maxpool2d rrule: scatter gradient to argmax positions
(tmpl/merge-into-template! 'raster.dl.nn/maxpool2d
                           {:pullback-factory (fn [_result x batch channels h w kh kw]
                                                (fn [dy]
                                                  (let [batch (long batch) channels (long channels)
                                                        h (long h) w (long w) kh (long kh) kw (long kw)
                                                        h-out (quot h kh)
                                                        w-out (quot w kw)
                                                        chw (* channels h w)
                                                        hw (* h w)
                                                        chw-out (* channels h-out w-out)
                                                        hw-out (* h-out w-out)
                                                        w-int (int w)
                                                        kh-int (int kh)
                                                        kw-int (int kw)
                                                        w-out-int (int w-out)
                                                        dx (alloc-like x (* batch chw))
                               ;; Use volatiles for max tracking (type-agnostic)
                                                        max-val (volatile! Double/NEGATIVE_INFINITY)
                                                        max-idx (volatile! (long 0))]
                                                    (dotimes [bi batch]
                                                      (dotimes [ci channels]
                                                        (let [base-in (+ (* bi (int chw)) (* ci (int hw)))
                                                              base-out (+ (* bi (int chw-out)) (* ci (int hw-out)))]
                                                          (dotimes [oh h-out]
                                                            (dotimes [ow w-out]
                                                              (let [ih0 (* oh kh-int)
                                                                    iw0 (* ow kw-int)]
                                       ;; Reset trackers
                                                                (vreset! max-val Double/NEGATIVE_INFINITY)
                                                                (vreset! max-idx 0)
                                                                (dotimes [khi kh]
                                                                  (dotimes [kwi kw]
                                                                    (let [x-idx (+ base-in (* (+ ih0 khi) w-int) (+ iw0 kwi))
                                                                          v (aget x x-idx)]
                                                                      (when (> v @max-val)
                                                                        (vreset! max-val v)
                                                                        (vreset! max-idx (long x-idx))))))
                                                                (let [out-idx (+ base-out (* oh w-out-int) ow)
                                                                      grad (aget dy out-idx)
                                                                      argmax-idx (long @max-idx)]
                                                                  (aset dx argmax-idx
                                                                        (+ (aget dx argmax-idx) grad)))))))))
                                                    [dx nil nil nil nil nil nil])))})

;; ================================================================
;; Embedding
;; ================================================================

;; table:[vocab, dim], indices:[N] (long[]) -> output:[N, dim]
(deftm embedding (All [T] [table :- (Array T) indices :- (Array long)
                           n :- Long vocab :- Long dim :- Long] :- (Array T)
                      (let [out (alloc-like table (* n dim))]
                        (dotimes [i n]
                          (let [idx (aget indices i)]
                            (dotimes [d dim]
                              (aset out (+ (* i (int dim)) d)
                                    (aget table (+ (* idx (int dim)) d))))))
                        out)))

;; embedding backward: d_table = scatter_add(dy at index positions)
(deftm embedding-backward (All [T] [dy :- (Array T) indices :- (Array long)
                                    n :- Long vocab :- Long dim :- Long]
                               :- (Array T)
                               (let [d-table (alloc-like dy (* vocab dim))]
                                 (dotimes [i n]
                                   (let [idx (aget indices i)]
                                     (dotimes [d dim]
                                       (aset d-table (+ (* idx (int dim)) d)
                                             (+ (aget d-table (+ (* idx (int dim)) d))
                                                (aget dy (+ (* i (int dim)) d)))))))
                                 d-table)))

(tmpl/merge-into-template! 'raster.dl.nn/embedding
                           {:pullback-factory (fn [_result table indices n vocab dim]
                                                (fn [dy]
                                                  [(embedding-backward dy indices n vocab dim) nil nil nil nil]))})

(tmpl/merge-into-template! 'raster.dl.nn/embedding
                           {:params '[table indices n vocab dim] :adjoint 'dy
                            :grads-fn (fn [ctx [table indices n vocab dim] _result adjoint gensym-fn]
                                        (let [dt (gensym-fn "d_table")]
                                          [(update ctx :bindings into
                                                   [dt (list 'raster.dl.nn/embedding-backward adjoint indices n vocab dim)])
                                           [dt nil nil nil nil]]))})

;; ================================================================
;; Dropout
;; ================================================================

;; mask is pre-generated: 0.0 for dropped, 1/(1-p) for kept
(deftm dropout (All [T] [x :- (Array T) mask :- (Array T) n :- Long]
                    :- (Array T)
                    (let [out (alloc-like x n)]
                      (dotimes [i n]
                        (aset out i (* (aget x i)
                                       (aget mask i))))
                      out)))

;; dropout backward: dx = dy * mask
(deftm dropout-backward (All [T] [dy :- (Array T) mask :- (Array T) n :- Long]
                             :- (Array T)
                             (let [dx (alloc-like dy n)]
                               (dotimes [i n]
                                 (aset dx i (* (aget dy i) (aget mask i))))
                               dx)))

(tmpl/merge-into-template! 'raster.dl.nn/dropout
                           {:pullback-factory (fn [_result x mask n]
                                                (fn [dy]
                                                  [(dropout-backward dy mask n) nil nil]))})

(tmpl/merge-into-template! 'raster.dl.nn/dropout
                           {:params '[x mask n] :adjoint 'dy
                            :grads-fn (fn [ctx [x mask n] _result adjoint gensym-fn]
                                        (let [dx (gensym-fn "dx")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/dropout-backward adjoint mask n)])
                                           [dx nil nil]]))})

(deftm generate-dropout-mask [n :- Long p :- Double] :- (Array double)
  (let [mask (double-array n)
        scale (/ 1.0 (- 1.0 p))
        rng (java.util.Random.)]
    (dotimes [i n]
      (aset mask i
            (if (>= (.nextDouble rng) p) scale 0.0)))
    mask))

;; ================================================================
;; Softmax (batched, over last dim)
;; ================================================================

;; softmax for flat array (single vector)
(deftm softmax-1d (All [T] [x :- (Array T) n :- Long] :- (Array T)
                       (let [out (alloc-like x n)
                             max-x (loop [i 0 m Double/NEGATIVE_INFINITY]
                                     (if (< i n)
                                       (recur (inc i) (Math/max m (aget x i)))
                                       m))
                             sum-exp (loop [i 0 s 0.0]
                                       (if (< i n)
                                         (let [e (Math/exp (- (aget x i) max-x))]
                                           (aset out i e)
                                           (recur (inc i) (+ s e)))
                                         s))
                             inv-sum (/ 1.0 sum-exp)]
                         (dotimes [i n]
                           (aset out i (* (aget out i) inv-sum)))
                         out)))

;; softmax-1d backward: dx_i = s_i * (dy_i - dot(s, dy))
(deftm softmax-1d-backward (All [T] [dy :- (Array T) result :- (Array T) n :- Long]
                                :- (Array T)
                                (let [s-dot-dy (loop [i 0 acc 0.0]
                                                 (if (< i n)
                                                   (recur (inc i) (+ acc (* (aget result i)
                                                                            (aget dy i))))
                                                   acc))
                                      dx (alloc-like dy n)]
                                  (dotimes [i n]
                                    (aset dx i
                                          (* (aget result i)
                                             (- (aget dy i) s-dot-dy))))
                                  dx)))

(tmpl/merge-into-template! 'raster.dl.nn/softmax-1d
                           {:pullback-factory (fn [result x n]
                                                (fn [dy]
                                                  [(softmax-1d-backward dy result n) nil]))})

(tmpl/merge-into-template! 'raster.dl.nn/softmax-1d
                           {:params '[x n] :adjoint 'dy
                            :grads-fn (fn [ctx [x n] result adjoint gensym-fn]
                                        (let [dx (gensym-fn "dx")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/softmax-1d-backward adjoint result n)])
                                           [dx nil]]))})

;; ================================================================
;; Weight initialization utilities
;; ================================================================

(deftm he-init [rows :- Long cols :- Long] :- (Array double)
  (let [scale (Math/sqrt (/ 6.0 cols))
        n (* rows cols)
        data (double-array n)
        rng (java.util.Random.)]
    (dotimes [i n]
      (aset data i (* scale (- (* 2.0 (.nextDouble rng)) 1.0))))
    data))

(deftm xavier-init [rows :- Long cols :- Long] :- (Array double)
  (let [scale (Math/sqrt (/ 6.0 (+ rows cols)))
        n (* rows cols)
        data (double-array n)
        rng (java.util.Random.)]
    (dotimes [i n]
      (aset data i (* scale (- (* 2.0 (.nextDouble rng)) 1.0))))
    data))

;; ================================================================
;; Compiler descriptors for buffer fusion and shape propagation
;; ================================================================

;; transpose-2d: allocates rows*cols, no in-place input reuse
(descriptor/register-buffer-semantics! 'raster.dl.nn/transpose-2d
                                       {:allocates? true
                                        :in-place-arg nil
                                        :alloc-form (fn [[_a rows cols] {:keys [dtype] :or {dtype :double}}]
                                                      (device/alloc-expr dtype (list '* rows cols)))
                                        :rewrite-fn (fn [[a rows cols] buf] (list 'raster.dl.nn/transpose-2d! a rows cols buf))})

;; matmul: allocates m*n, no in-place input reuse (BLAS needs fresh output)
(descriptor/register-buffer-semantics! 'raster.dl.nn/matmul
                                       {:allocates? true
                                        :in-place-arg nil
                                        :alloc-form (fn [[_A _B m _k n] {:keys [dtype] :or {dtype :double}}]
                                                      (device/alloc-expr dtype (list '* m n)))
                                        :rewrite-fn (fn [[A B m k n] buf] (list 'raster.dl.nn/matmul! A B buf m k n))})

;; daxpy-diff!: allocates n, CAN reuse arg 0 (elementwise: reads a[i] before writing out[i])
(descriptor/register-buffer-semantics! 'raster.linalg.blas/daxpy-diff!
                                       {:allocates? true
                                        :in-place-arg 0
                                        :alloc-form (fn [[_a _b _scale n] {:keys [dtype] :or {dtype :double}}]
                                                      (device/alloc-expr dtype n))
                                        :rewrite-fn (fn [[a b scale n] buf] (list 'raster.linalg.blas/daxpy-diff-into! a b scale n buf))})

;; matmul-dA: allocates m*k, no in-place
(descriptor/register-buffer-semantics! 'raster.dl.nn/matmul-dA
                                       {:allocates? true
                                        :in-place-arg nil
                                        :alloc-form (fn [[_dC _B m k _n] {:keys [dtype] :or {dtype :double}}]
                                                      (device/alloc-expr dtype (list '* m k)))
                                        :rewrite-fn (fn [[dC B m k n] buf] (list 'raster.dl.nn/matmul-dA! dC B m k n buf))})

;; matmul-dB: allocates k*n, no in-place
(descriptor/register-buffer-semantics! 'raster.dl.nn/matmul-dB
                                       {:allocates? true
                                        :in-place-arg nil
                                        :alloc-form (fn [[_A _dC _m k n] {:keys [dtype] :or {dtype :double}}]
                                                      (device/alloc-expr dtype (list '* k n)))
                                        :rewrite-fn (fn [[A dC m k n] buf] (list 'raster.dl.nn/matmul-dB! A dC m k n buf))})

;; conv2d: allocates batch*c_out*h_out*w_out
(descriptor/register-buffer-semantics! 'raster.dl.nn/conv2d
                                       {:allocates? true
                                        :in-place-arg nil})

;; maxpool2d: allocates batch*channels*(h/kh)*(w/kw)
(descriptor/register-buffer-semantics! 'raster.dl.nn/maxpool2d
                                       {:allocates? true
                                        :in-place-arg nil})

;; Dim rules for shape propagation through conv/pool layers
;; conv2d: output length = batch * c_out * h_out * w_out
(descriptor/register-dim-rule! 'raster.dl.nn/conv2d
                               (fn [[_x _W _b batch c-in h w c-out kh kw stride-h stride-w pad-h pad-w] _dim-env _params-set]
                                 (let [h-out (list 'clojure.core/+ 1
                                                   (list 'clojure.core/quot
                                                         (list 'clojure.core/+ h (list 'clojure.core/* 2 pad-h) (list 'clojure.core/- kh))
                                                         stride-h))
                                       w-out (list 'clojure.core/+ 1
                                                   (list 'clojure.core/quot
                                                         (list 'clojure.core/+ w (list 'clojure.core/* 2 pad-w) (list 'clojure.core/- kw))
                                                         stride-w))]
                                   (list 'clojure.core/* batch c-out h-out w-out))))

;; maxpool2d: output length = batch * channels * (h/kh) * (w/kw)
(descriptor/register-dim-rule! 'raster.dl.nn/maxpool2d
                               (fn [[_x batch channels h w kh kw] _dim-env _params-set]
                                 (list 'clojure.core/* batch channels
                                       (list 'clojure.core/quot h kh)
                                       (list 'clojure.core/quot w kw))))

;; ================================================================
;; Per-gradient backward deftms for template-based AD codegen
;; Each returns (Array T) so TypedClojure can trace types end-to-end.
;; ================================================================

;; ----------------------------------------------------------------
;; Layer-norm backward: dx, dgamma, dbeta
;; ----------------------------------------------------------------

(deftm layer-norm-backward-dx (All [T] [dy :- (Array T) x :- (Array T)
                                        gamma :- (Array T)
                                        batch :- Long features :- Long eps :- Double]
                                   :- (Array T)
                                   (let [dx (alloc-like dy (* batch features))]
                                     (dotimes [b batch]
                                       (let [offset (* b (int features))
                                             mean (loop [i 0 s 0.0]
                                                    (if (< i features)
                                                      (recur (inc i) (+ s (aget x (+ offset i))))
                                                      (/ s features)))
                                             var (loop [i 0 s 0.0]
                                                   (if (< i features)
                                                     (let [d (- (aget x (+ offset i)) mean)]
                                                       (recur (inc i) (+ s (* d d))))
                                                     (/ s features)))
                                             inv-std (/ 1.0 (n/sqrt (+ var eps)))
                                             sums (loop [i 0 s1 0.0 s2 0.0]
                                                    (if (< i features)
                                                      (let [x-hat (* (- (aget x (+ offset i)) mean) inv-std)
                                                            dg (* (aget dy (+ offset i))
                                                                  (aget gamma i))]
                                                        (recur (inc i) (+ s1 (* dg x-hat)) (+ s2 dg)))
                                                      [s1 s2]))
                                             s1 (nth sums 0)
                                             s2 (nth sums 1)
                                             n-inv (/ 1.0 features)]
                                         (dotimes [i features]
                                           (let [x-hat (* (- (aget x (+ offset i)) mean) inv-std)
                                                 dyi (aget dy (+ offset i))
                                                 gi (aget gamma i)]
                                             (aset dx (+ offset i)
                                                   (* inv-std (- (* dyi gi) (* n-inv (+ s2 (* s1 x-hat))))))))))
                                     dx)))

(deftm layer-norm-backward-dgamma (All [T] [dy :- (Array T) x :- (Array T)
                                            batch :- Long features :- Long eps :- Double]
                                       :- (Array T)
                                       (let [dgamma (alloc-like dy features)]
                                         (dotimes [b batch]
                                           (let [offset (* b (int features))
                                                 mean (loop [i 0 s 0.0]
                                                        (if (< i features)
                                                          (recur (inc i) (+ s (aget x (+ offset i))))
                                                          (/ s features)))
                                                 var (loop [i 0 s 0.0]
                                                       (if (< i features)
                                                         (let [d (- (aget x (+ offset i)) mean)]
                                                           (recur (inc i) (+ s (* d d))))
                                                         (/ s features)))
                                                 inv-std (/ 1.0 (n/sqrt (+ var eps)))]
                                             (dotimes [i features]
                                               (let [x-hat (* (- (aget x (+ offset i)) mean) inv-std)]
                                                 (aset dgamma i (+ (aget dgamma i) (* (aget dy (+ offset i)) x-hat)))))))
                                         dgamma)))

(deftm layer-norm-backward-dbeta (All [T] [dy :- (Array T)
                                           batch :- Long features :- Long]
                                      :- (Array T)
                                      (let [dbeta (alloc-like dy features)]
                                        (dotimes [b batch]
                                          (let [offset (* b (int features))]
                                            (dotimes [i features]
                                              (aset dbeta i (+ (aget dbeta i) (aget dy (+ offset i)))))))
                                        dbeta)))

;; ----------------------------------------------------------------
;; RMS-norm backward: dx, dweight
;;   y_i = x_i · inv · (g0 + w_i),  inv = 1/sqrt(mean_j(x_j²) + eps)
;;   c   = Σ_i (g0+w_i)·x_i·dy_i   (per row)
;;   dx_j = inv·(g0+w_j)·dy_j − (inv³/F)·x_j·c
;;   dw_i = Σ_rows dy_i·x_i·inv   (independent of the gain offset g0)
;; ----------------------------------------------------------------

(deftm rms-norm-backward-dx (All [T] [dy :- (Array T) x :- (Array T) weight :- (Array T)
                                      rows :- Long features :- Long
                                      eps :- Double gain-offset :- Double] :- (Array T)
                                (let [dx (alloc-like dy (* rows features))]
                                  (dotimes [r rows]
                                    (let [offset (* r (int features))
                                          ms (loop [i 0 s 0.0]
                                               (if (< i features)
                                                 (let [v (aget x (+ offset i))]
                                                   (recur (inc i) (+ s (* v v))))
                                                 (/ s features)))
                                          inv (/ 1.0 (n/sqrt (+ ms eps)))
                                          c (loop [i 0 s 0.0]
                                              (if (< i features)
                                                (let [gi (+ gain-offset (aget weight i))]
                                                  (recur (inc i)
                                                         (+ s (* gi (* (aget x (+ offset i))
                                                                       (aget dy (+ offset i)))))))
                                                s))
                                          inv3f (/ (* inv (* inv inv)) features)]
                                      (dotimes [i features]
                                        (let [gi (+ gain-offset (aget weight i))]
                                          (aset dx (+ offset i)
                                                (- (* inv (* gi (aget dy (+ offset i))))
                                                   (* inv3f (* (aget x (+ offset i)) c))))))))
                                  dx)))

(deftm rms-norm-backward-dweight (All [T] [dy :- (Array T) x :- (Array T)
                                           rows :- Long features :- Long eps :- Double] :- (Array T)
                                     (let [dw (alloc-like dy features)]
                                       (dotimes [r rows]
                                         (let [offset (* r (int features))
                                               ms (loop [i 0 s 0.0]
                                                    (if (< i features)
                                                      (let [v (aget x (+ offset i))]
                                                        (recur (inc i) (+ s (* v v))))
                                                      (/ s features)))
                                               inv (/ 1.0 (n/sqrt (+ ms eps)))]
                                           (dotimes [i features]
                                             (aset dw i (+ (aget dw i)
                                                           (* (aget dy (+ offset i))
                                                              (* (aget x (+ offset i)) inv)))))))
                                       dw)))

;; ----------------------------------------------------------------
;; Group-norm backward: dx, dgamma, dbeta
;; ----------------------------------------------------------------

(deftm group-norm-backward-dx (All [T] [dy :- (Array T) x :- (Array T)
                                        gamma :- (Array T)
                                        batch :- Long channels :- Long
                                        spatial :- Long groups :- Long eps :- Double]
                                   :- (Array T)
                                   (let [cpg (quot channels groups)
                                         group-size (* cpg spatial)
                                         dx (alloc-like dy (* batch channels spatial))]
                                     (dotimes [b batch]
                                       (dotimes [g groups]
                                         (let [mean (loop [c 0 acc 0.0]
                                                      (if (< c cpg)
                                                        (let [ch (+ (* g (int cpg)) c)
                                                              acc (loop [sp 0 inner-acc acc]
                                                                    (if (< sp spatial)
                                                                      (recur (inc sp) (+ inner-acc
                                                                                         (aget x (+ (* b (int (* channels spatial)))
                                                                                                    (* ch (int spatial)) sp))))
                                                                      inner-acc))]
                                                          (recur (inc c) acc))
                                                        (/ acc group-size)))
                                               var (loop [c 0 acc 0.0]
                                                     (if (< c cpg)
                                                       (let [ch (+ (* g (int cpg)) c)
                                                             acc (loop [sp 0 inner-acc acc]
                                                                   (if (< sp spatial)
                                                                     (let [idx (+ (* b (int (* channels spatial)))
                                                                                  (* ch (int spatial)) sp)
                                                                           d (- (aget x idx) mean)]
                                                                       (recur (inc sp) (+ inner-acc (* d d))))
                                                                     inner-acc))]
                                                         (recur (inc c) acc))
                                                       (/ acc group-size)))
                                               inv-std (/ 1.0 (n/sqrt (+ var eps)))
                                               ;; Accumulate into double-array to avoid vector return
                                               ;; (nth on vector loses type tags after inlining)
                                               sums-buf (double-array 2)
                                               _ (dotimes [c cpg]
                                                   (let [ch (+ (* g (int cpg)) c)]
                                                     (dotimes [sp spatial]
                                                       (let [idx (+ (* b (int (* channels spatial)))
                                                                    (* ch (int spatial)) sp)
                                                             x-hat (* (- (aget x idx) mean) inv-std)
                                                             dg (* (aget dy idx) (aget gamma ch))]
                                                         (aset sums-buf 0 (+ (aget sums-buf 0) (* dg x-hat)))
                                                         (aset sums-buf 1 (+ (aget sums-buf 1) dg))))))
                                               s1 (aget sums-buf 0)
                                               s2 (aget sums-buf 1)
                                               n-inv (/ 1.0 group-size)]
                                           (dotimes [c cpg]
                                             (let [ch (+ (* g (int cpg)) c)]
                                               (dotimes [sp spatial]
                                                 (let [idx (+ (* b (int (* channels spatial)))
                                                              (* ch (int spatial)) sp)
                                                       x-hat (* (- (aget x idx) mean) inv-std)
                                                       dyi (aget dy idx)
                                                       gi (aget gamma ch)]
                                                   (aset dx idx
                                                         (* inv-std (- (* dyi gi)
                                                                       (* n-inv (+ s2 (* s1 x-hat)))))))))))))
                                     dx)))

(deftm group-norm-backward-dgamma (All [T] [dy :- (Array T) x :- (Array T)
                                            batch :- Long channels :- Long
                                            spatial :- Long groups :- Long eps :- Double]
                                       :- (Array T)
                                       (let [cpg (quot channels groups)
                                             group-size (* cpg spatial)
                                             dgamma (alloc-like dy channels)]
                                         (dotimes [b batch]
                                           (dotimes [g groups]
                                             (let [mean (loop [c 0 acc 0.0]
                                                          (if (< c cpg)
                                                            (let [ch (+ (* g (int cpg)) c)
                                                                  acc (loop [sp 0 inner-acc acc]
                                                                        (if (< sp spatial)
                                                                          (recur (inc sp) (+ inner-acc
                                                                                             (aget x (+ (* b (int (* channels spatial)))
                                                                                                        (* ch (int spatial)) sp))))
                                                                          inner-acc))]
                                                              (recur (inc c) acc))
                                                            (/ acc group-size)))
                                                   var (loop [c 0 acc 0.0]
                                                         (if (< c cpg)
                                                           (let [ch (+ (* g (int cpg)) c)
                                                                 acc (loop [sp 0 inner-acc acc]
                                                                       (if (< sp spatial)
                                                                         (let [idx (+ (* b (int (* channels spatial)))
                                                                                      (* ch (int spatial)) sp)
                                                                               d (- (aget x idx) mean)]
                                                                           (recur (inc sp) (+ inner-acc (* d d))))
                                                                         inner-acc))]
                                                             (recur (inc c) acc))
                                                           (/ acc group-size)))
                                                   inv-std (/ 1.0 (n/sqrt (+ var eps)))]
                                               (dotimes [c cpg]
                                                 (let [ch (+ (* g (int cpg)) c)]
                                                   (dotimes [sp spatial]
                                                     (let [idx (+ (* b (int (* channels spatial)))
                                                                  (* ch (int spatial)) sp)
                                                           x-hat (* (- (aget x idx) mean) inv-std)]
                                                       (aset dgamma ch (+ (aget dgamma ch)
                                                                          (* (aget dy idx) x-hat))))))))))
                                         dgamma)))

(deftm group-norm-backward-dbeta (All [T] [dy :- (Array T)
                                           batch :- Long channels :- Long spatial :- Long]
                                      :- (Array T)
                                      (let [dbeta (alloc-like dy channels)]
                                        (dotimes [b batch]
                                          (dotimes [ch channels]
                                            (dotimes [sp spatial]
                                              (let [idx (+ (* b (int (* channels spatial)))
                                                           (* ch (int spatial)) sp)]
                                                (aset dbeta ch (+ (aget dbeta ch) (aget dy idx)))))))
                                        dbeta)))

;; ----------------------------------------------------------------
;; Batch-norm backward: dx, dgamma, dbeta
;; ----------------------------------------------------------------

(deftm batch-norm-backward-dx (All [T] [dy :- (Array T) x :- (Array T)
                                        gamma :- (Array T)
                                        batch :- Long features :- Long eps :- Double]
                                   :- (Array T)
                                   (let [dx (alloc-like dy (* batch features))]
                                     (dotimes [j features]
                                       (let [mean (loop [i 0 s 0.0]
                                                    (if (< i batch)
                                                      (recur (inc i) (+ s (aget x (+ (* i (int features)) j))))
                                                      (/ s batch)))
                                             var (loop [i 0 s 0.0]
                                                   (if (< i batch)
                                                     (let [d (- (aget x (+ (* i (int features)) j)) mean)]
                                                       (recur (inc i) (+ s (* d d))))
                                                     (/ s batch)))
                                             inv-std (/ 1.0 (n/sqrt (+ var eps)))
                                             s1 (loop [i 0 s 0.0]
                                                  (if (< i batch)
                                                    (let [idx (+ (* i (int features)) j)
                                                          x-hat (* (- (aget x idx) mean) inv-std)
                                                          dg (* (aget dy idx) (aget gamma j))]
                                                      (recur (inc i) (+ s (* dg x-hat))))
                                                    s))
                                             s2 (loop [i 0 s 0.0]
                                                  (if (< i batch)
                                                    (let [idx (+ (* i (int features)) j)]
                                                      (recur (inc i) (+ s (* (aget dy idx) (aget gamma j)))))
                                                    s))
                                             n-inv (/ 1.0 batch)]
                                         (dotimes [i batch]
                                           (let [idx (+ (* i (int features)) j)
                                                 x-hat (* (- (aget x idx) mean) inv-std)
                                                 dyi (aget dy idx)
                                                 gi (aget gamma j)]
                                             (aset dx idx
                                                   (* inv-std (- (* dyi gi)
                                                                 (* n-inv (+ s2 (* s1 x-hat))))))))))
                                     dx)))

(deftm batch-norm-backward-dgamma (All [T] [dy :- (Array T) x :- (Array T)
                                            batch :- Long features :- Long eps :- Double]
                                       :- (Array T)
                                       (let [dgamma (alloc-like dy features)]
                                         (dotimes [j features]
                                           (let [mean (loop [i 0 s 0.0]
                                                        (if (< i batch)
                                                          (recur (inc i) (+ s (aget x (+ (* i (int features)) j))))
                                                          (/ s batch)))
                                                 var (loop [i 0 s 0.0]
                                                       (if (< i batch)
                                                         (let [d (- (aget x (+ (* i (int features)) j)) mean)]
                                                           (recur (inc i) (+ s (* d d))))
                                                         (/ s batch)))
                                                 inv-std (/ 1.0 (n/sqrt (+ var eps)))]
                                             (dotimes [i batch]
                                               (let [idx (+ (* i (int features)) j)
                                                     x-hat (* (- (aget x idx) mean) inv-std)]
                                                 (aset dgamma j (+ (aget dgamma j) (* (aget dy idx) x-hat)))))))
                                         dgamma)))

(deftm batch-norm-backward-dbeta (All [T] [dy :- (Array T)
                                           batch :- Long features :- Long]
                                      :- (Array T)
                                      (let [dbeta (alloc-like dy features)]
                                        (dotimes [i batch]
                                          (dotimes [j features]
                                            (aset dbeta j (+ (aget dbeta j) (aget dy (+ (* i (int features)) j))))))
                                        dbeta)))

;; ----------------------------------------------------------------
;; Conv1d backward: dx, dW, db
;; ----------------------------------------------------------------

(deftm conv1d-backward-dx (All [T] [dy :- (Array T) x :- (Array T)
                                    W :- (Array T)
                                    batch :- Long c-in :- Long length :- Long
                                    c-out :- Long kernel :- Long stride :- Long
                                    pad :- Long]
                               :- (Array T)
                               (let [l-out (+ 1 (quot (+ length (* 2 pad) (- kernel)) stride))
                                     ck (* c-in kernel)
                                     bl (* batch l-out)
                                     dy-cols (alloc-like dy (* c-out bl))]
                                 (dotimes [co c-out]
                                   (dotimes [bi batch]
                                     (dotimes [li l-out]
                                       (let [src-idx (+ (* bi (int (* c-out l-out)))
                                                        (* co (int l-out)) li)
                                             dst-idx (+ (* co (int bl)) (* bi (int l-out)) li)]
                                         (aset dy-cols dst-idx (aget dy src-idx))))))
                                 (let [d-cols (alloc-like dy (* ck bl))
                                       _ (blas/dgemm-tn! W dy-cols d-cols ck c-out bl (n/oftype dy 1.0) (n/oftype dy 0.0))]
                                   (col2im-1d d-cols batch c-in length kernel stride pad)))))

(deftm conv1d-backward-dW (All [T] [dy :- (Array T) x :- (Array T)
                                    batch :- Long c-in :- Long length :- Long
                                    c-out :- Long kernel :- Long stride :- Long
                                    pad :- Long]
                               :- (Array T)
                               (let [l-out (+ 1 (quot (+ length (* 2 pad) (- kernel)) stride))
                                     ck (* c-in kernel)
                                     bl (* batch l-out)
                                     dy-cols (alloc-like dy (* c-out bl))]
                                 (dotimes [co c-out]
                                   (dotimes [bi batch]
                                     (dotimes [li l-out]
                                       (let [src-idx (+ (* bi (int (* c-out l-out)))
                                                        (* co (int l-out)) li)
                                             dst-idx (+ (* co (int bl)) (* bi (int l-out)) li)]
                                         (aset dy-cols dst-idx (aget dy src-idx))))))
                                 (let [cols (im2col-1d x batch c-in length kernel stride pad)
                                       dW (alloc-like dy (* c-out ck))
                                       _ (blas/dgemm-nt! dy-cols cols dW c-out bl ck (n/oftype dy 1.0) (n/oftype dy 0.0))]
                                   dW))))

(deftm conv1d-backward-db (All [T] [dy :- (Array T)
                                    batch :- Long c-out :- Long
                                    length :- Long kernel :- Long
                                    stride :- Long pad :- Long]
                               :- (Array T)
                               (let [l-out (+ 1 (quot (+ length (* 2 pad) (- kernel)) stride))
                                     db (alloc-like dy c-out)]
                                 (dotimes [bi batch]
                                   (dotimes [co c-out]
                                     (dotimes [li l-out]
                                       (aset db co
                                             (+ (aget db co)
                                                (aget dy (+ (* bi (int (* c-out l-out)))
                                                            (* co (int l-out)) li)))))))
                                 db)))

;; ================================================================
;; Pullback registrations (after backward deftms)
;; ================================================================

;; layer-norm rrule — per-gradient deftms (parametric)
(tmpl/merge-into-template! 'raster.dl.nn/layer-norm
                           {:pullback-factory (fn [_result x gamma _beta batch features eps]
                                                (fn [dy]
                                                  [(layer-norm-backward-dx dy x gamma batch features eps)
                                                   (layer-norm-backward-dgamma dy x batch features eps)
                                                   (layer-norm-backward-dbeta dy batch features)
                                                   nil nil nil]))})

;; rms-norm rrule — runtime pullback + compiled grads-fn (dx, dweight; frozen
;; gain-offset/dims -> nil). The QK-norm and the 6 sandwich norms per Gemma layer
;; all differentiate through this. (rms-norm x weight rows features eps gain-offset)
(tmpl/merge-into-template! 'raster.dl.nn/rms-norm
                           {:pullback-factory (fn [_result x weight rows features eps gain-offset]
                                                (fn [dy]
                                                  [(rms-norm-backward-dx dy x weight rows features eps gain-offset)
                                                   (rms-norm-backward-dweight dy x rows features eps)
                                                   nil nil nil nil]))
                            :params '[x weight rows features eps gain-offset] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x weight rows features eps gain-offset]
                                           _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx") dw (gensym-fn "dw")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/rms-norm-backward-dx
                                                             adjoint-sym x weight rows features eps gain-offset)
                                                    dw (list 'raster.dl.nn/rms-norm-backward-dweight
                                                             adjoint-sym x rows features eps)])
                                           [dx dw nil nil nil nil]]))})

;; hadamard rrule — elementwise product (gated MLPs). d_a = dy⊙b, d_b = dy⊙a.
(tmpl/merge-into-template! 'raster.dl.nn/hadamard
                           {:pullback-factory (fn [_result a b n]
                                                (fn [dy]
                                                  [(hadamard-backward dy b n)
                                                   (hadamard-backward dy a n)
                                                   nil]))
                            :params '[a b n] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [a b n] _result-sym adjoint-sym gensym-fn]
                                        (let [da (gensym-fn "da") db (gensym-fn "db")]
                                          [(update ctx :bindings into
                                                   [da (list 'raster.dl.nn/hadamard-backward adjoint-sym b n)
                                                    db (list 'raster.dl.nn/hadamard-backward adjoint-sym a n)])
                                           [da db nil]]))})

;; linear-nb rrule — bias-free linear (Gemma/Qwen projections). Same as linear
;; minus the db term; reuses linear-dx / linear-dW. (linear-nb x W batch in-f out-f)
(tmpl/merge-into-template! 'raster.dl.nn/linear-nb
                           {:pullback-factory (fn [_result x W batch in-f out-f]
                                                (fn [dy]
                                                  [(linear-dx dy W batch in-f out-f)
                                                   (linear-dW dy x batch in-f out-f)
                                                   nil nil nil]))
                            :params '[x W batch in-f out-f] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x W batch in-f out-f] _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx") dW (gensym-fn "dW")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/linear-dx adjoint-sym W batch in-f out-f)
                                                    dW (list 'raster.dl.nn/linear-dW adjoint-sym x batch in-f out-f)])
                                           [dx dW nil nil nil]]))})

;; group-norm rrule — per-gradient deftms (parametric)
(tmpl/merge-into-template! 'raster.dl.nn/group-norm
                           {:pullback-factory (fn [_result x gamma _beta batch channels spatial groups eps]
                                                (fn [dy]
                                                  [(group-norm-backward-dx dy x gamma batch channels spatial groups eps)
                                                   (group-norm-backward-dgamma dy x batch channels spatial groups eps)
                                                   (group-norm-backward-dbeta dy batch channels spatial)
                                                   nil nil nil nil nil]))})

;; batch-norm rrule — per-gradient deftms (parametric)
(tmpl/merge-into-template! 'raster.dl.nn/batch-norm
                           {:pullback-factory (fn [_result x gamma _beta _running-mean _running-var batch features eps _momentum _training]
                                                (fn [dy]
                                                  [(batch-norm-backward-dx dy x gamma batch features eps)
                                                   (batch-norm-backward-dgamma dy x batch features eps)
                                                   (batch-norm-backward-dbeta dy batch features)
                                                   nil nil nil nil nil nil nil]))})

;; conv1d rrule — per-gradient deftms (parametric)
(tmpl/merge-into-template! 'raster.dl.nn/conv1d
                           {:pullback-factory (fn [_result x W _b batch c-in length c-out kernel stride pad]
                                                (fn [dy]
                                                  [(conv1d-backward-dx dy x W batch c-in length c-out kernel stride pad)
                                                   (conv1d-backward-dW dy x batch c-in length c-out kernel stride pad)
                                                   (conv1d-backward-db dy batch c-out length kernel stride pad)
                                                   nil nil nil nil nil nil]))})

;; ================================================================
;; Template registrations for complex ops
;; ================================================================

;; layer-norm template — direct calls to per-gradient deftms
(tmpl/merge-into-template! 'raster.dl.nn/layer-norm
                           {:params '[x gamma beta batch features eps] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x gamma _beta batch features eps] _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx")
                                              dgamma (gensym-fn "dgamma")
                                              dbeta (gensym-fn "dbeta")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/layer-norm-backward-dx
                                                             adjoint-sym x gamma batch features eps)
                                                    dgamma (list 'raster.dl.nn/layer-norm-backward-dgamma
                                                                 adjoint-sym x batch features eps)
                                                    dbeta (list 'raster.dl.nn/layer-norm-backward-dbeta
                                                                adjoint-sym batch features)])
                                           [dx dgamma dbeta nil nil nil]]))})

;; group-norm template — direct calls to per-gradient deftms
(tmpl/merge-into-template! 'raster.dl.nn/group-norm
                           {:params '[x gamma beta batch channels spatial groups eps] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x gamma _beta batch channels spatial groups eps]
                                           _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx")
                                              dgamma (gensym-fn "dgamma")
                                              dbeta (gensym-fn "dbeta")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/group-norm-backward-dx
                                                             adjoint-sym x gamma batch channels spatial groups eps)
                                                    dgamma (list 'raster.dl.nn/group-norm-backward-dgamma
                                                                 adjoint-sym x batch channels spatial groups eps)
                                                    dbeta (list 'raster.dl.nn/group-norm-backward-dbeta
                                                                adjoint-sym batch channels spatial)])
                                           [dx dgamma dbeta nil nil nil nil nil]]))})

;; batch-norm template — direct calls to per-gradient deftms
(tmpl/merge-into-template! 'raster.dl.nn/batch-norm
                           {:params '[x gamma beta running-mean running-var batch features eps momentum training]
                            :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x gamma _beta _rm _rv batch features eps _momentum _training]
                                           _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx")
                                              dgamma (gensym-fn "dgamma")
                                              dbeta (gensym-fn "dbeta")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/batch-norm-backward-dx
                                                             adjoint-sym x gamma batch features eps)
                                                    dgamma (list 'raster.dl.nn/batch-norm-backward-dgamma
                                                                 adjoint-sym x batch features eps)
                                                    dbeta (list 'raster.dl.nn/batch-norm-backward-dbeta
                                                                adjoint-sym batch features)])
                                           [dx dgamma dbeta nil nil nil nil nil nil nil]]))})

;; conv1d template — direct calls to per-gradient deftms
(tmpl/merge-into-template! 'raster.dl.nn/conv1d
                           {:params '[x W b batch c-in length c-out kernel stride pad]
                            :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x W _b batch c-in length c-out kernel stride pad]
                                           _result-sym adjoint-sym gensym-fn]
                                        (let [dx (gensym-fn "dx")
                                              dW (gensym-fn "dW")
                                              db (gensym-fn "db")]
                                          [(update ctx :bindings into
                                                   [dx (list 'raster.dl.nn/conv1d-backward-dx
                                                             adjoint-sym x W batch c-in length c-out kernel stride pad)
                                                    dW (list 'raster.dl.nn/conv1d-backward-dW
                                                             adjoint-sym x batch c-in length c-out kernel stride pad)
                                                    db (list 'raster.dl.nn/conv1d-backward-db
                                                             adjoint-sym batch c-out length kernel stride pad)])
                                           [dx dW db nil nil nil nil nil nil]]))})
