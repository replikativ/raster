(ns raster.dl.qlinear-composable
  "Composable-IR expression of the Q4_0 quantized GEMV — the path to retire the hand-
  written string emitter (raster...cpu.quant/emit-qmatmul-*) in favour of one expression
  the compiler lowers to every backend.

  The int8-MAC is the ONE irreducible seam, expressed as the `wi8-dot-q4` primitive: a
  portable scalar deftm here, to be LOWERED per backend to the hardware op (dpbusd/VNNI on
  x86, dp4a/DPAS on the GPU) in the next step. The GEMV itself — loop over output rows,
  per-block scale, zero-point fold, float reduce — is ordinary composable deftm IR, so it
  compiles to SIMD-C / OpenCL / future backends from this single source and fuses with
  neighbouring norms.

  Held to the bench/raster/quant_faithfulness gate: correct + within noise of the hand
  kernel. VALIDATED: via `compile-aot :target :c` this composable source is bit-faithful
  AND roughly per-thread competitive with the handwritten kernel (single-thread C ≈ hand
  per-thread; -O3 -march=native auto-vectorizes the int8 loop about as well as the hand
  VNNI). The remaining gap to the hand kernel is THREADING (the hand kernel's 4-thread
  spin-pool over output-row slices), not codegen. On the JVM backend it is 37x slower —
  the JVM has no int8-MAC — so :target :c is the correct lowering for quantized math.
  Next: parallelize the row loop (par/map! → OpenMP in the C backend, or pool-drive the C
  fn over row slices); optionally lower wi8-dot-q4 → the _i8dot dpbusd intrinsic for the
  shapes where auto-vec underperforms. Then retire the hand-written string emitter.

  Layout: row-major Q4_0 {wq[out*in/2], ws[out*nb]} (the natural layout, same as the
  scalar reference). The permute-free interleaved layout (repack-stream) is a later
  optimisation, once the primitive vectorises."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.compiler.backend.cpu.quant :as cq]))

(deftm wi8-dot-q4
  "The int8-MAC seam over ONE Q4_0 block: unsigned-nibble x signed-int8 dot. Reads 16
  packed weight bytes at wq[woff..] (32 nibbles, lo=elem k, hi=elem k+16) and 32 int8
  activations at xq[xoff..], returns the raw int32 dot (the caller folds the zero-point
  via -ZP*sum(xq)). Portable scalar reference; the per-backend lowering replaces this body
  with dpbusd / dp4a / DPAS."
  [wq :- (Array byte), woff :- Long, xq :- (Array byte), xoff :- Long] :- Long
  (loop [k 0 s 0]
    (if (< k 16)
      (let [bv (bit-and (long (ra/aget wq (+ woff k))) 0xFF)]
        (recur (inc k)
               (+ s (* (long (ra/aget xq (+ xoff k))) (bit-and bv 0xF))
                    (* (long (ra/aget xq (+ xoff k 16))) (bit-shift-right bv 4)))))
      s)))

(deftm qmatmul-q4-composable
  "Composable Q4_0 GEMV: y[o] = sum_b (xs[b]*ws[o,b]) * (wi8-dot-q4(o,b) - 8*xsum[b]).
  Activation pre-quantized (xq int8, xs scale, xsum block-sum). Row-major weight."
  [xq :- (Array byte), xs :- (Array float), xsum :- (Array int),
   wq :- (Array byte), ws :- (Array float), in :- Long, out :- Long] :- (Array float)
  (let [y (float-array out)
        nb (quot (long in) 32)
        half (quot (long in) 2)]
    (dotimes [o (long out)]
      (let [wrow (* (long o) half)
            wsrow (* (long o) nb)
            acc (loop [b 0 a 0.0]
                  (if (< b nb)
                    (let [dot (wi8-dot-q4 wq (+ wrow (* (long b) 16)) xq (* (long b) 32))
                          folded (- dot (* 8 (long (ra/aget xsum b))))
                          scale (* (double (ra/aget xs b)) (double (ra/aget ws (+ wsrow b))))]
                      (recur (inc b) (+ a (* scale (double folded)))))
                    a))]
        (ra/aset y o (float acc))))
    y))

(deftm qmatmul-q4-composable!
  "Sliceable in-place Q4_0 GEMV: same compute as qmatmul-q4-composable, but writes y[o]
  for o in [o-start, o-start+o-count) into a SHARED pre-allocated y. Disjoint row slices
  -> the persistent spin-pool drives them concurrently (the threading the hand kernel gets,
  composably — NOT OpenMP-per-call, which is measured worse than serial for tiny decode
  matmuls). Compile via compile-aot :target :c, then pool-drive with make-composable-c-gemv."
  [xq :- (Array byte), xs :- (Array float), xsum :- (Array int),
   wq :- (Array byte), ws :- (Array float), y :- (Array float),
   in :- Long, out :- Long, o-start :- Long, o-count :- Long] :- (Array float)
  (let [nb (quot (long in) 32)
        half (quot (long in) 2)]
    (dotimes [oi (long o-count)]
      (let [o (+ (long o-start) oi)
            wrow (* o half)
            wsrow (* o nb)
            acc (loop [b 0 a 0.0]
                  (if (< b nb)
                    (let [dot (wi8-dot-q4 wq (+ wrow (* (long b) 16)) xq (* (long b) 32))
                          folded (- dot (* 8 (long (ra/aget xsum b))))
                          scale (* (double (ra/aget xs b)) (double (ra/aget ws (+ wsrow b))))]
                      (recur (inc b) (+ a (* scale (double folded)))))
                    a))]
        (ra/aset y o (float acc))))
    y))

(defn make-composable-c-gemv
  "Compile qmatmul-q4-composable! to one C slice-kernel (compile-aot :target :c) once, and
  return a driver (fn [xq xs xsum wq ws in out] -> y) that allocates y and pool-drives the
  C kernel over disjoint output-row slices via the persistent spin-pool — the same threading
  model as the handwritten kernel, from composable deftm source. defn: device/runtime glue."
  []
  (let [cfn ((requiring-resolve 'raster.compiler.pipeline/compile-aot)
             #'qmatmul-q4-composable! :target :c)]
    (fn [xq xs xsum wq ws in out]
      (let [out (long out)
            y (float-array out)]
        (cq/run-par-fn!
         (fn [wid n]
           (let [chunk (quot (+ out (dec (long n))) (long n))
                 o0 (* (long wid) chunk)
                 cnt (max 0 (min chunk (- out o0)))]
             (when (pos? cnt)
               (cfn xq xs xsum wq ws y in out o0 cnt)))))
        y))))
