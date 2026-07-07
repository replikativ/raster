(ns raster.quant.train
  "Differentiable quantized linear for fine-tuning (LoRA / QLoRA base layer).

  Forward:  y[M,out] = X[M,in] @ dequant(W)^T   over a FROZEN row-major Q8_0 weight
  Backward: dx[M,in] = dY[M,out] @ dequant(W)   (the dequant-transpose GEMM)

  There is NO weight gradient — the quantized base is frozen. The point is that
  gradients flow THROUGH the frozen quantized weight to reach the input, so LoRA
  adapters on *lower* layers still train (a transformer layer's x is the previous
  layer's output). This is exactly QLoRA's base-layer treatment (frozen weight
  dequantized on the fly; the trainable low-rank adapters differentiate normally).

  Row-major Q8_0 layout, per-32-block symmetric int8:
    codes  : byte[out*in]   signed int8 code per weight, row-major
    scales : float[out*nb]  per-row per-block scale (nb = in/32)
    W[o,i] = scales[o*nb + i/32] * codes[o,i]

  The forward is a template-backed op: its :grads-fn (below) keeps it symbolic
  through the AD transform, so its body is never differentiated (the derivative
  relationship is the hand-written pullback qlinear-q8-dx, the same contract as
  raster.dl.nn/linear over BLAS dgemm). Both use the identical dequant, so the
  analytic dx is the exact gradient of the forward and a finite-difference check
  matches to float precision. Neither is ^:no-inline: opacity to differentiation
  comes from the template, and dropping the inline barrier lets the post-AD backend
  expansion inline the `par/map-void!` bodies so they lower to RESIDENT GPU kernel
  steps (the int8-resident QLoRA training path), just like the decode quant GEMVs."
  (:require [raster.core :refer [deftm]]
            [raster.ad.templates :as tmpl]))

;; ---------------------------------------------------------------------------
;; Row-major Q8_0 codec (plain helpers — data prep, not on the compute path)
;; ---------------------------------------------------------------------------

(defn q8-quantize
  "Quantize a dense row-major f32 weight W[out,in] to row-major Q8_0:
   {:codes byte[out*in] :scales float[out*nb] :in :out}. Per-row per-32-block
   symmetric int8: d = max|w|/127, code = round(w/d) (clamped [-127,127]).
   `in` must be a multiple of 32."
  [^floats W out in]
  (let [out (int out) in (int in) nb (quot in 32)
        codes (byte-array (* out in))
        scales (float-array (* out nb))]
    (assert (zero? (rem in 32)) (str "in=" in " not a multiple of 32"))
    (dotimes [o out]
      (dotimes [b nb]
        (let [base (+ (* o in) (* b 32))
              mx (loop [k 0 mm 0.0]
                   (if (< k 32)
                     (recur (inc k) (max mm (Math/abs (double (aget W (+ base k))))))
                     mm))
              d (/ mx 127.0)
              id (if (pos? d) (/ 1.0 d) 0.0)]
          (aset scales (+ (* o nb) b) (float d))
          (dotimes [k 32]
            (let [q (Math/round (* (double (aget W (+ base k))) id))]
              (aset codes (+ base k)
                    (unchecked-byte (max -127 (min 127 q)))))))))
    {:codes codes :scales scales :in in :out out}))

(defn dequant-q8-dense
  "Row-major Q8_0 (codes, scales) -> dense f32 W[out*in]. For a reference forward
   and finite-difference checks; the differentiable op dequantizes on the fly."
  ^floats [^bytes codes ^floats scales in out]
  (let [in (int in) out (int out) nb (quot in 32)
        W (float-array (* out in))]
    (dotimes [o out]
      (dotimes [i in]
        (aset W (+ (* o in) i)
              (float (* (aget scales (+ (* o nb) (quot i 32)))
                        (double (aget codes (+ (* o in) i))))))))
    W))

;; ---------------------------------------------------------------------------
;; Differentiable op: forward + backward-data kernels (dequant on the fly)
;; ---------------------------------------------------------------------------

(deftm qlinear-q8
  "Forward: y[M,out] = X[M,in] @ dequant(codes,scales)^T over a frozen row-major
   Q8_0 weight. y[m,o] = Σ_i scales[o,i/32]·codes[o,i]·x[m,i].

   NOT ^:no-inline: the :grads-fn template below keeps this op SYMBOLIC through the
   AD transform (pre-AD lowering skips template-backed ops; interpreted value+grad
   uses the template pullback), so opacity to differentiation is provided by the
   template, not by an inline barrier. Post-AD `expand-for-backends` then inlines
   the body so its `par/map-void!` lowers to a resident GPU :map-void kernel step —
   the memory-fit win of keeping the int8 weight resident. Structurally identical to
   the decode quant GEMV kernels (raster.quant.kernels-k) which lower the same way."
  [x :- (Array float) codes :- (Array byte) scales :- (Array float)
   m :- Long in :- Long out :- Long] :- (Array float)
  (let [y  (float-array (clojure.core/* (long m) (long out)))
        nb (clojure.core/quot (long in) 32)]
    (raster.par/map-void! p (clojure.core/* (long m) (long out))
      (let [mi (clojure.core/quot p (long out))
            o  (clojure.core/rem p (long out))
            xb (clojure.core/* mi (long in))
            wb (clojure.core/* o (long in))
            sb (clojure.core/* o nb)]
        (raster.arrays/aset y p
          (raster.numeric/oftype y
            (loop [i 0 acc 0.0]
              (if (clojure.core/< i (long in))
                (recur (clojure.core/inc i)
                       (raster.numeric/+ acc
                         (raster.numeric/* (raster.numeric/* (raster.arrays/aget scales (clojure.core/+ sb (clojure.core/quot i 32)))
                                                             (raster.numeric/oftype y (raster.arrays/aget codes (clojure.core/+ wb i))))
                                           (raster.arrays/aget x (clojure.core/+ xb i)))))
                acc))))))
    y))

(deftm qlinear-q8-dx
  "Backward-data: dx[M,in] = dY[M,out] @ dequant(codes,scales). The dequant-transpose
   GEMM. dx[m,i] = Σ_o scales[o,i/32]·codes[o,i]·dy[m,o]. No weight gradient (frozen).

   NOT ^:no-inline: this is the backward kernel emitted by qlinear-q8's :grads-fn. It
   only appears in the IR AFTER the AD transform (never differentiated itself), so it
   carries no template — it just needs to inline post-AD so its `par/map-void!` lowers
   to a resident GPU :map-void step, exactly like the forward."
  [dy :- (Array float) codes :- (Array byte) scales :- (Array float)
   m :- Long in :- Long out :- Long] :- (Array float)
  (let [dx (float-array (clojure.core/* (long m) (long in)))
        nb (clojure.core/quot (long in) 32)]
    (raster.par/map-void! p (clojure.core/* (long m) (long in))
      (let [mi (clojure.core/quot p (long in))
            i  (clojure.core/rem p (long in))
            db (clojure.core/* mi (long out))
            si (clojure.core/quot i 32)]
        (raster.arrays/aset dx p
          (raster.numeric/oftype dx
            (loop [o 0 acc 0.0]
              (if (clojure.core/< o (long out))
                (recur (clojure.core/inc o)
                       (raster.numeric/+ acc
                         (raster.numeric/* (raster.numeric/* (raster.arrays/aget scales (clojure.core/+ (clojure.core/* o nb) si))
                                                             (raster.numeric/oftype dx (raster.arrays/aget codes (clojure.core/+ (clojure.core/* o (long in)) i))))
                                           (raster.arrays/aget dy (clojure.core/+ db o)))))
                acc))))))
    dx))

;; ---------------------------------------------------------------------------
;; AD registration: only x gets a gradient; the quantized weight is frozen.
;; ---------------------------------------------------------------------------

;; qlinear-q8 is a ^:no-inline opaque op (like nn/linear over BLAS dgemm): its
;; body is never differentiated. The backward is the hand-written qlinear-q8-dx
;; GEMM, emitted as a flat let-binding via :grads-fn — the same single
;; representation every other op uses, so it fuses and lowers like a template op.
;; Only x gets a gradient; the quantized weight is frozen (nil for the rest).
(tmpl/merge-into-template! 'raster.quant.train/qlinear-q8
  {:params '[x codes scales m in out] :result nil :adjoint 'dy
   :grads-fn (fn [ctx [_x codes scales m in out] _result-sym adjoint-sym gensym-fn]
               (let [dx (gensym-fn "dx")]
                 [(update ctx :bindings into
                          [dx (list 'raster.quant.train/qlinear-q8-dx
                                    adjoint-sym codes scales m in out)])
                  [dx nil nil nil nil nil]]))})
