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
            [raster.compiler.backend.cpu.quant :as cq]
            [raster.compiler.core.op-descriptor :as descriptor]))

(deftm ^:no-inline wi8-dot-q4
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

(deftm ^:no-inline wi8-dot-q4-x8
  "The 8-column int8-MAC TILE over one Q4_0 block in the repack-stream interleaved
  layout — the tile-level tensorize seam. Reads the 128 interleaved weight bytes for
  an 8-column group at wqi[woff..woff+128) and 32 int8 activations at xq[xoff..+32),
  writes the 8 raw int32 column dots to out[0..8). Column L -> lane L, no horizontal
  reduce (the interleave puts each column in its own lane). Portable scalar reference
  that exactly mirrors the persistent-accumulator microkernel its per-backend lowering
  emits: dpbusd/VNNI on x86, dp4a/DPAS on the GPU. The surrounding GEMV (block loop,
  per-block scale, zero-point fold, store) stays composable deftm IR."
  [wqi :- (Array byte), woff :- Long, xq :- (Array byte), xoff :- Long,
   out :- (Array int)] :- (Array int)
  (dotimes [L 8]
    (let [s (loop [ig 0 acc 0]
              (if (< ig 8)
                (recur (inc ig)
                       (loop [k 0 a acc]
                         (if (< k 4)
                           (let [nib-idx (+ (* L 4) k)
                                 bidx (+ woff (* ig 16)
                                         (if (< nib-idx 16) nib-idx (- nib-idx 16)))
                                 bv (bit-and (long (ra/aget wqi bidx)) 0xFF)
                                 nib (if (< nib-idx 16) (bit-and bv 0xF) (bit-shift-right bv 4))
                                 act (long (ra/aget xq (+ xoff (* ig 4) k)))]
                             (recur (inc k) (+ a (* act nib))))
                           a)))
                acc))]
      (ra/aset out L (int s))))
  out)

;; The ISA seam: wi8-dot-q4's optimal C lowering is the AVX2 maddubs block-dot (the
;; widening unsigned-nibble x signed-int8 MAC), not its translated scalar body. Registered
;; as a :c-helper override the CPU-C AOT backend emits for the ^:no-inline call. (Halide
;; treats this as a one-row intrinsic table; here it's one registered op helper. The
;; surrounding loop/accumulate is still the composable GEMV — see qmatmul-q4-composable.)
(descriptor/register-c-helper! 'raster.dl.qlinear-composable/wi8-dot-q4
  {:includes "#include <immintrin.h>\n"
   :gen (fn [c-name]
          (str "static inline long " c-name
               "(const signed char* wq, long woff, const signed char* xq, long xoff) {\n"
               "  __m128i raw = _mm_loadu_si128((const __m128i*)(wq + woff));\n"
               "  __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw,4), raw), _mm256_set1_epi8(0x0F));\n"
               "  __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + xoff));\n"
               "  __m256i p = _mm256_maddubs_epi16(nib, xv);\n"
               "  __m256i s = _mm256_madd_epi16(p, _mm256_set1_epi16(1));\n"
               "  __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s,1));\n"
               "  lo = _mm_hadd_epi32(lo, lo); lo = _mm_hadd_epi32(lo, lo);\n"
               "  return (long)_mm_cvtsi128_si32(lo);\n}\n"))})

;; The TILE's optimal C lowering: 8 dpbusd into a PERSISTENT __m256i accumulator over
;; the block's 8 input-groups (no per-block hsum — the gap the measurement localized),
;; then store 8 int32 column dots. Column L -> lane L (the interleaved layout puts it
;; there). Under -march=native the cascade picks vpdpbusd (AVX-VNNI) or maddubs+madd.
;; This is the irreducible intrinsic; the GEMV's scale/fold/accumulate stays composable.
(descriptor/register-c-helper! 'raster.dl.qlinear-composable/wi8-dot-q4-x8
  {:includes "#include <immintrin.h>\n"
   :gen (fn [c-name]
          (str "static inline int* " c-name
               "(const signed char* wqi, long woff, const signed char* xq, long xoff, int* out) {\n"
               "  const unsigned char* wb = (const unsigned char*)(wqi + woff);\n"
               "  const signed char* xb = xq + xoff;\n"
               "  __m256i ia = _mm256_setzero_si256();\n"
               "  const __m256i m4 = _mm256_set1_epi8(0x0F);\n"
               "  for (int ig = 0; ig < 8; ig++) {\n"
               "    __m128i raw = _mm_loadu_si128((const __m128i*)(wb + ig*16));\n"
               "    __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw,4), raw), m4);\n"
               "    __m256i ab = _mm256_set1_epi32(*(const int*)(xb + ig*4));\n"
               "#if defined(__AVX512VNNI__) && defined(__AVX512VL__)\n"
               "    ia = _mm256_dpbusd_epi32(ia, nib, ab);\n"
               "#elif defined(__AVXVNNI__)\n"
               "    ia = _mm256_dpbusd_avx_epi32(ia, nib, ab);\n"
               "#else\n"
               "    ia = _mm256_add_epi32(ia, _mm256_madd_epi16(_mm256_maddubs_epi16(nib, ab), _mm256_set1_epi16(1)));\n"
               "#endif\n"
               "  }\n"
               "  _mm256_storeu_si256((__m256i*)out, ia);\n"
               "  return out;\n}\n"))})

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

;; ---------------------------------------------------------------------------
;; Tiled GEMV — the 8-column-in-lanes structure the measurement showed is the gap.
;; ---------------------------------------------------------------------------

;; The tile call WRITES out8 (arg 4) and is read after — declare the mutation so DCE
;; keeps the call and the effects analysis sequences it correctly (mirrors qlinear-i8!).
(descriptor/register-op-descriptor! 'raster.dl.qlinear-composable/wi8-dot-q4-x8
  {:effects {:pure? false :mutating? true}})
(descriptor/register-buffer-write! 'raster.dl.qlinear-composable/wi8-dot-q4-x8 :overwrite 4)

(deftm qmatmul-q4-x8!
  "Tiled composable Q4_0 GEMV over the INTERLEAVED (repack-stream) layout: processes 8
  output columns per group via the wi8-dot-q4-x8 dpbusd tile (column L -> lane L), folds
  the per-block zero-point + scale, accumulates 8 floats, stores 8 — the hand kernel's
  structure with only the dpbusd tile as a kernel call, the rest composable IR. Sliceable
  over GROUPS [g-start, g-start+g-count) of 8 columns; pool-driven like the hand kernel."
  [xq :- (Array byte), xs :- (Array float), xsum :- (Array int),
   wqi :- (Array byte), wsi :- (Array float), y :- (Array float),
   in :- Long, out :- Long, g-start :- Long, g-count :- Long] :- (Array float)
  (let [nb (quot (long in) 32)
        out8 (int-array 8)
        acc8 (float-array 8)]
    (dotimes [gi (long g-count)]
      (let [g (+ (long g-start) gi)
            wbase (* g nb 128)
            sbase (* g nb 8)]
        (dotimes [L 8] (ra/aset acc8 L (float 0.0)))
        (dotimes [b (long nb)]
          ;; fill out8 with the 8 column dots (side-effecting tile; mutation registered)
          (wi8-dot-q4-x8 wqi (+ wbase (* (long b) 128)) xq (* (long b) 32) out8)
          (let [xsb (double (ra/aget xs b))
                fold (* 8 (long (ra/aget xsum b)))]
            (dotimes [L 8]
              (let [folded (- (long (ra/aget out8 L)) fold)
                    scale (* xsb (double (ra/aget wsi (+ sbase (* (long b) 8) L))))]
                (ra/aset acc8 L (float (+ (double (ra/aget acc8 L)) (* scale (double folded)))))))))
        (dotimes [L 8] (ra/aset y (+ (* g 8) L) (ra/aget acc8 L)))))
    y))

(defn make-x8-c-gemv
  "Compile qmatmul-q4-x8! to one C slice-kernel (compile-aot :target :c) once; return a
  driver (fn [xq xs xsum wqi wsi in out] -> y) that pool-drives it over disjoint COLUMN-
  GROUP slices. Weight must be the interleaved repack-stream layout (wqi/wsi). The tile-
  tensorize candidate measured against the hand kernel. defn: device/runtime glue."
  []
  (let [cfn ((requiring-resolve 'raster.compiler.pipeline/compile-aot)
             #'qmatmul-q4-x8! :target :c)]
    (fn [xq xs xsum wqi wsi in out]
      (let [out (long out)
            ng (quot out 8)
            y (float-array out)]
        (cq/run-par-fn!
         (fn [wid n]
           (let [chunk (quot (+ ng (dec (long n))) (long n))
                 g0 (* (long wid) chunk)
                 cnt (max 0 (min chunk (- ng g0)))]
             (when (pos? cnt)
               (cfn xq xs xsum wqi wsi y in out g0 cnt)))))
        y))))
