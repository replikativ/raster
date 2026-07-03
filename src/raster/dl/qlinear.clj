(ns raster.dl.qlinear
  "Quantized int8-MAC linear as a first-class compiler op.

  The one ISA seam — the widening int8 dot (maddubs/dpbusd on x86, dp4a/DPAS on the
  GPU) — promoted from a hand-called kernel into a registered compiler op, so the
  quantized matmul composes with the rest of the deftm block library instead of
  living outside the pipeline:

   - it FUSES in compile-aot: as a ^:no-inline op it stays an opaque CALL in compiled
     code (exactly like raster.dl.nn/matmul over BLAS dgemm!), while the surrounding
     norms / activations / residuals fuse around it — a whole quantized layer compiles
     to one unit with the matmul as a kernel call;
   - it is DEVICE-PLACEABLE: the :placement facet (:jvm | :cpu-quant | :gpu) is the
     MLX-style per-op compute-profile tag the GPU lowering keys off;
   - it is COMPOSABLE for fine-tuning a layer on top: the frozen quantized stack is a
     forward op whose outputs feed an ordinary trainable deftm layer.

  Two variants, mirroring matmul / matmul!:
    qlinear-i8!  — ^:no-inline opaque kernel op: int8-quantizes the activation and
                   streams the spin-pool gemv into a pre-allocated y.
    qlinear-i8   — allocates y[out] then calls qlinear-i8!  (buffer-semantics rewrite).

  Weight = the repack-stream Q4_0 layout (wqi byte[], wsi float[]) produced by
  raster.compiler.backend.cpu.quant/repack-stream; the activation is int8-quantized
  inside the op. The default kernel is the host-best maddubs/VNNI build, compiled once.

  AD-through the frozen weight (dx = dequant(W)^T @ dy) is intentionally not yet
  registered: fine-tuning a layer ON TOP needs only the forward op (the trainable
  layer differentiates; the quantized features are constants). Backprop INTO/THROUGH a
  frozen quantized layer (LoRA-into-lower, full fine-tune) is the follow-on AD rule."
  (:require [raster.core :refer [deftm]]
            [raster.compiler.backend.cpu.quant :as cq]
            [raster.compiler.core.op-descriptor :as descriptor]))

;; One host-best stream-gemv kernel per weight format, compiled once — the
;; maddubs/VNNI int8-MAC seam. Under -march=native the C macro cascade picks
;; dpbusd (AVX-VNNI) or maddubs. Q4_0 = decode default; Q8_0 = the embedder-quality
;; format (measured: 4-bit weights cost ~5%+ embedding cosine, Q8_0 is lossless).
(def ^:private kq4 (delay (cq/compile-qmatmul-stream "qlinear_i8_q4" cq/q4-0 :maddubs 1)))
(def ^:private kq8 (delay (cq/compile-qmatmul-stream "qlinear_i8_q8" cq/q8-0 :maddubs 1)))

(deftm ^:no-inline qlinear-i8!
  "In-place quantized linear: int8-quantize x, stream-gemv into y over a repack-stream
  Q4_0 weight. The opaque int8-MAC op — ^:no-inline keeps it a call in compiled code
  (like dgemm!), so it lowers to the host kernel rather than being inlined/walked."
  [x :- (Array float), wqi :- (Array byte), wsi :- (Array float),
   y :- (Array float), in :- Long, out :- Long] :- (Array float)
  (let [q (cq/quantize-act-i8-par x in)]
    (@kq4 (:xq q) (:xs q) (:xsum q) wqi wsi y in out)
    y))

(deftm qlinear-i8
  "Quantized linear: y = dequant(W_q4) @ x, x:[in] float -> y:[out] float. Allocates y
  then delegates to the in-place op (buffer-semantics rewrites this to qlinear-i8!)."
  [x :- (Array float), wqi :- (Array byte), wsi :- (Array float),
   in :- Long, out :- Long] :- (Array float)
  (let [y (float-array out)]
    (qlinear-i8! x wqi wsi y in out)
    y))

(deftm ^:no-inline qlinear-i8-q8!
  "In-place quantized linear over a Q8_0 repack-stream weight (byte-direct layout,
  `(repack-stream wq ws out in q8-0)`). Same A8 activation quant + dpbusd core as the
  Q4 op — only the weight unpack differs. The embedder-quality op: Q8_0 weights are
  cosine-lossless where 4-bit costs ~5%."
  [x :- (Array float), wqi :- (Array byte), wsi :- (Array float),
   y :- (Array float), in :- Long, out :- Long] :- (Array float)
  (let [q (cq/quantize-act-i8-par x in)]
    (@kq8 (:xq q) (:xs q) (:xsum q) wqi wsi y in out)
    y))

(deftm qlinear-i8-q8
  "Quantized linear over a Q8_0 weight: allocates y then delegates to the in-place op
  (buffer-semantics rewrites this to qlinear-i8-q8!)."
  [x :- (Array float), wqi :- (Array byte), wsi :- (Array float),
   in :- Long, out :- Long] :- (Array float)
  (let [y (float-array out)]
    (qlinear-i8-q8! x wqi wsi y in out)
    y))

;; ---------------------------------------------------------------------------
;; Compiler op-descriptor registration
;; ---------------------------------------------------------------------------

;; Buffer semantics: qlinear-i8 allocates y[out], no in-place input reuse; the
;; into-variant is qlinear-i8! (mirrors matmul -> matmul!).
(descriptor/register-buffer-semantics! 'raster.dl.qlinear/qlinear-i8
  {:allocates? true
   :in-place-arg nil
   :alloc-form (fn [[_x _wqi _wsi _in out] _opts] (list 'float-array out))
   :rewrite-fn (fn [[x wqi wsi in out] buf]
                 (list 'raster.dl.qlinear/qlinear-i8! x wqi wsi buf in out))})

;; qlinear-i8! fully overwrites its y argument (index 3) — safe to reuse without
;; zeroing (the kernel stores, never accumulates into, y).
(descriptor/register-op-descriptor! 'raster.dl.qlinear/qlinear-i8!
  {:effects {:pure? false :mutating? true}})
(descriptor/register-buffer-write! 'raster.dl.qlinear/qlinear-i8! :overwrite 3)

;; Placement: the int8-MAC op runs under the CPU quant profile by default. The GPU
;; profile re-tags it :gpu to select the emit-qmatmul-opencl lowering (GPU Step 2).
(descriptor/register-placement! 'raster.dl.qlinear/qlinear-i8 :cpu-quant)
(descriptor/register-placement! 'raster.dl.qlinear/qlinear-i8! :cpu-quant)

;; The Q8_0 twin: identical semantics/effects/placement, byte-direct weight layout.
(descriptor/register-buffer-semantics! 'raster.dl.qlinear/qlinear-i8-q8
  {:allocates? true
   :in-place-arg nil
   :alloc-form (fn [[_x _wqi _wsi _in out] _opts] (list 'float-array out))
   :rewrite-fn (fn [[x wqi wsi in out] buf]
                 (list 'raster.dl.qlinear/qlinear-i8-q8! x wqi wsi buf in out))})
(descriptor/register-op-descriptor! 'raster.dl.qlinear/qlinear-i8-q8!
  {:effects {:pure? false :mutating? true}})
(descriptor/register-buffer-write! 'raster.dl.qlinear/qlinear-i8-q8! :overwrite 3)
(descriptor/register-placement! 'raster.dl.qlinear/qlinear-i8-q8 :cpu-quant)
(descriptor/register-placement! 'raster.dl.qlinear/qlinear-i8-q8! :cpu-quant)
