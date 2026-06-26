(ns raster.compiler.backend.cpu.quant
  "Quantized matmul codegen for the native CPU backend — the principled,
  no-hardcoded-kernel design distilled from Futhark / MLIR / ggml / XLA:

    qmatmul = par/map rows · par/reduce blocks ·
                scale · widening-i8-dot(act_i8, unpack(W, FORMAT-DESCRIPTOR))

  Three composable parts, NOT N hand-written kernels:
   - FORMAT = DATA: a layout descriptor (block size, weight bits, packing,
     zero-point, symmetric?) drives the unpack codegen. Q4_0 is one data row.
   - The MAC = ONE primitive `widening-i8-dot`, lowered per ISA (`:maddubs` AVX2 /
     `:scalar` fallback; `:vnni`/`:dpas` later). The single irreducible seam.
   - SCHEDULE: deferred per-block scale, float-vector accumulation, one reduce/row.

  The emitter assembles these → the same optimal C llama.cpp hand-writes per
  format×ISA, from one descriptor + one primitive."
  (:require [clojure.string :as str]
            [raster.compiler.backend.cpu.codegen :as cpu]))

;; ---- encode side: quantizers matching the q4-0 kernel (interleaved pack) ----
(defn quantize-weight-q4
  "f32 weight (len % 32 = 0) → {:wq byte[len/2] :ws float[len/32]}, interleaved
  Q4_0 pack (byte k of a block holds w[k] low / w[k+16] high), symmetric d=max/-8."
  [^floats w]
  (let [len (alength w) nb (quot len 32)
        wq (byte-array (quot len 2)) ws (float-array nb)]
    (dotimes [b nb]
      (let [base (* b 32)
            mx (loop [j 0 a 0.0 v 0.0]
                 (if (< j 32)
                   (let [x (aget w (+ base j)) ax (Math/abs x)]
                     (if (> ax a) (recur (inc j) ax x) (recur (inc j) a v)))
                   v))
            d (/ mx -8.0) id (if (zero? d) 0.0 (/ 1.0 d))]
        (aset ws b (float d))
        (dotimes [k 16]
          (let [q0 (max 0 (min 15 (+ (Math/round (* (aget w (+ base k)) id)) 8)))
                q1 (max 0 (min 15 (+ (Math/round (* (aget w (+ base k 16)) id)) 8)))]
            (aset wq (+ (* b 16) k) (unchecked-byte (bit-or q0 (bit-shift-left q1 4))))))))
    {:wq wq :ws ws}))

(defn- quantize-act-blocks!
  "Quantize the 32-element blocks [r0, r0+cnt) of activation x into xq/xs/xsum
  (per-block symmetric int8, d=max/127, with the per-block sum for the -8 fold).
  Blocks are independent — this is the parallel work unit."
  [^floats x ^bytes xq ^floats xs ^ints xsum r0 cnt]
  (dotimes [i (long cnt)]
    (let [r (+ (long r0) i) base (* r 32)
          mx (loop [j 0 a 0.0] (if (< j 32) (recur (inc j) (max a (Math/abs (aget x (+ base j))))) a))
          d (/ mx 127.0) id (if (zero? d) 0.0 (/ 1.0 d))]
      (aset xs r (float d))
      (aset xsum r (int (loop [k 0 s 0]
                          (if (< k 32)
                            (let [q (max -127 (min 127 (Math/round (* (aget x (+ base k)) id))))]
                              (aset xq (+ base k) (unchecked-byte q)) (recur (inc k) (+ s (int q))))
                            s)))))))

(defn quantize-act-i8
  "f32 activation row(s) [M*in] → {:xq byte[M*in] :xs float[M*nb] :xsum int[M*nb]}
  per-block (32) symmetric int8, d=max/127, with per-block sum for the -8 correction."
  [^floats x M in]
  (let [nb (quot in 32)
        xq (byte-array (alength x)) xs (float-array (* M nb)) xsum (int-array (* M nb))]
    (quantize-act-blocks! x xq xs xsum 0 (* M nb))
    {:xq xq :xs xs :xsum xsum}))

;; ---- FORMAT DESCRIPTORS (data) ----
(def q4-0
  "Symmetric 4-bit, block 32, two nibbles/byte, interleaved (lo=w[0..15],
  hi=w[16..31]), zero-point 8. (= GGUF Q4_0 quant level.)"
  {:name "q4_0" :block 32 :weight-bits 4 :pack :nibble-interleaved
   :zero-point 8 :symmetric true :act-type :i8})

;; ---- per-ISA pieces of the widening-i8-dot primitive (the one seam) ----
;; :maddubs names the x86 SIMD lowering, but the emitted `_i8dot` is a PORTABLE
;; macro cascade: under -march=native the host's best int8-MAC is selected at
;; compile time — AVX-VNNI `dpbusd` (1 instr) where available, baseline AVX2
;; `maddubs+madd` otherwise. No runtime CPUID needed for the JIT-on-host model.
;; (Other ISAs — NEON sdot, SVE — are added as more #elif arms; same descriptor,
;; same schedule, different MAC spelling. A shipped fat binary would multi-version.)
(def ^:private i8dot-helper
  (str "#include <immintrin.h>\n"
       "// widening int8 dot, ACCUMULATING: acc += sum_4(u8 w * s8 x) per int32 lane.\n"
       "// Host-selected MAC via -march=native macros (no runtime CPUID).\n"
       "static inline __m256i _i8dot_acc(__m256i acc, __m256i wu8, __m256i xs8){\n"
       "#if defined(__AVX512VNNI__) && defined(__AVX512VL__)\n"
       "  return _mm256_dpbusd_epi32(acc, wu8, xs8);\n"
       "#elif defined(__AVXVNNI__)\n"
       "  return _mm256_dpbusd_avx_epi32(acc, wu8, xs8);\n"
       "#else\n"
       "  return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wu8, xs8), _mm256_set1_epi16(1)));\n"
       "#endif\n}\n"
       "static inline __m256i _i8dot(__m256i wu8, __m256i xs8){ return _i8dot_acc(_mm256_setzero_si256(), wu8, xs8); }\n"))
(def ^:private mac-includes {:maddubs i8dot-helper})
(def ^:private mac-consts
  {:maddubs "  const __m256i _m4 = _mm256_set1_epi8(0x0F);\n"})

(defn- emit-row
  "Emit one output row's contribution for the v5 schedule: unpack the weight block
  (from the FORMAT descriptor), widening-i8-dot with the shared activation `xv`,
  FMA the scaled int32 partials into accumulator `facc-i`, accrue the zero-point
  correction `corr-i`. i = row offset within the register block."
  [{:keys [weight-bits pack]} mac i]
  (assert (and (= weight-bits 4) (= pack :nibble-interleaved) (= mac :maddubs)))
  (let [w (str "w" i) s (str "s" i) f (str "f" i) c (str "c" i)]
    (str "        { __m128i _wl = _mm_loadu_si128((const __m128i*)(" w " + b*16));\n"
         "          __m256i _wq = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(_wl,4), _wl), _m4);\n"
         "          __m256i _s32 = _i8dot(_wq, _xv);\n"
         "          float _sc = _xsb*" s "[b]; " f " = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_s32), _mm256_set1_ps(_sc), " f ");\n"
         "          " c " += _sc*_xsm; }\n")))

;; ---- compose the v5 (4-row register-blocked) kernel from descriptor + primitive ----
(def ^:private BR 4) ;; register block: 4 output rows = 4 independent FMA chains
(defn emit-qmatmul
  "Emit the v5 quantized matmul C kernel for FORMAT + MAC. Register-blocks BR output
  rows (independent accumulators hide FMA latency) + loads the activation block once
  per BR rows. Signature: void NAME(xq,xs,xsum,wq,ws,y,M,in,out,o_start,o_count)."
  [name format mac]
  (let [{:keys [block zero-point]} format zp (str zero-point ".f")
        rows (range BR)]
    (str "#include <stdint.h>\n" (mac-includes mac)
         "static inline float _hsum8(__m256 v){__m128 x=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));x=_mm_hadd_ps(x,x);x=_mm_hadd_ps(x,x);return _mm_cvtss_f32(x);}\n"
         "void " name "(const int8_t* xq, const float* xs, const int* xsum,\n"
         "    const uint8_t* wq, const float* ws, float* y, int M, int in, int out,\n"
         "    int o_start, int o_count) {\n"
         "  int nb = in/" block ";\n" (mac-consts mac)
         "  for (int mm = 0; mm < M; mm++) {\n"
         "    const int8_t* xr = xq + (long)mm*in;\n"
         "    const float* xsr = xs + (long)mm*nb; const int* xsm = xsum + (long)mm*nb;\n"
         "    int oend = o_start + o_count, o = o_start;\n"
         ;; --- main: BR rows at a time ---
         "    for (; o + " BR " <= oend; o += " BR ") {\n"
         (apply str (for [i rows] (str "      __m256 f" i "=_mm256_setzero_ps(); float c" i "=0;\n")))
         (apply str (for [i rows] (str "      const uint8_t* w" i "=wq+(long)(o+" i ")*in/2; const float* s" i "=ws+(long)(o+" i ")*nb;\n")))
         "      for (int b = 0; b < nb; b++) {\n"
         "        __m256i _xv = _mm256_loadu_si256((const __m256i*)(xr + b*32));\n"
         "        float _xsb = xsr[b]; int _xsm = xsm[b];\n"
         (apply str (for [i rows] (emit-row format mac i)))
         "      }\n"
         (apply str (for [i rows] (str "      y[(long)mm*out+o+" i "] = _hsum8(f" i ") - " zp "*c" i ";\n")))
         "    }\n"
         ;; --- tail: remaining (<BR) rows, one at a time ---
         "    for (; o < oend; o++) {\n"
         "      __m256 f0=_mm256_setzero_ps(); float c0=0;\n"
         "      const uint8_t* w0=wq+(long)o*in/2; const float* s0=ws+(long)o*nb;\n"
         "      for (int b = 0; b < nb; b++) {\n"
         "        __m256i _xv = _mm256_loadu_si256((const __m256i*)(xr + b*32)); float _xsb=xsr[b]; int _xsm=xsm[b];\n"
         (emit-row format mac 0)
         "      }\n"
         "      y[(long)mm*out+o] = _hsum8(f0) - " zp "*c0;\n"
         "    }\n  }\n}\n")))

(defonce ^:private pool
  (delay (java.util.concurrent.Executors/newFixedThreadPool
          (.availableProcessors (Runtime/getRuntime)))))

(defn compile-qmatmul
  "Emit + compile + load a quantized matmul. Returns a fn (xq xs xsum wq ws y M in
  out). With n-threads>1, big matmuls (out>=1024) are split over output-row ranges
  across a JVM thread pool — each native call writes a disjoint y slice (no OpenMP
  dep; raster scales where memory-bound llama.cpp saturates)."
  ([name format mac] (compile-qmatmul name format mac 1))
  ([name format mac n-threads]
   (let [k (cpu/load-kernel (cpu/compile-source! (emit-qmatmul name format mac)) name 6 5)]
     (if (<= n-threads 1)
       (fn [xq xs xsum wq ws y M in out] (k xq xs xsum wq ws y M in out 0 out))
       (let [p @pool]
         (fn [xq xs xsum wq ws y M in out]
           (if (< (int out) 1024)
             (k xq xs xsum wq ws y M in out 0 out)
             (let [chunk (quot (+ (int out) (dec n-threads)) n-threads)
                   tasks (for [t (range n-threads)
                               :let [o0 (* t chunk) cnt (min chunk (- (int out) o0))]
                               :when (pos? cnt)]
                           (reify java.util.concurrent.Callable
                             (call [_] (k xq xs xsum wq ws y M in out o0 cnt) nil)))]
               (doseq [f (.invokeAll p tasks)] (.get f))))))))))

;; ============================================================================
;; SCHEDULES — regime-appropriate kernel strategies (schedule-as-data).
;; A schedule names a {layout, kernel, parallelization} strategy chosen by the
;; roofline regime. We measured the decode GEMV to be MEMORY-bandwidth bound (84MB
;; lm_head streamed at 9.6/11.2 GB/s = 86% of the single-core memory ceiling), so
;; its optimal schedule is to STREAM the compressed weights contiguously — there is
;; nothing for a compute schedule to do. The :stream-gemv schedule below is that.
;; Compute-bound :tile-gemm (prefill M>1, register-block tile) and GPU variants slot
;; into the same {descriptor × primitive × schedule} skeleton as more entries.
;; ============================================================================
(def ^:private NC 8) ;; column tile = AVX2 f32 lane count (16 for AVX-512, 4 for NEON)

(defn repack-stream
  "Row-major Q4_0 {wq[out*in/2], ws[out*nb]} → the :stream-gemv layout, where output
  column i lands in accumulator lane i (no shuffles, no final permute). Per (group of
  NC cols, block, input-group of 4 elems), byte j (0..15) packs col(j/4) elem(j%4) in
  the low nibble and col(4+j/4) elem(j%4) in the high nibble — so a [lo|hi] unpack
  yields the dpbusd operand directly. wsi = NC column scales per (group, block). out
  must be a multiple of NC (caller pads). nb = in/32. One-time weight-load cost."
  [^bytes wq ^floats ws out in]
  (let [nb (quot in 32) ng (quot out NC) half (quot in 2)
        wqi (byte-array (* ng nb 128)) wsi (float-array (* ng nb NC))
        ;; nibble of (output row, block b, element e in 0..31) from interleaved Q4_0
        src-nib (fn [row b e]
                  (let [bv (bit-and (aget wq (+ (* row half) (* b 16) (if (< e 16) e (- e 16)))) 0xFF)]
                    (if (< e 16) (bit-and bv 0xF) (bit-shift-right bv 4))))]
    (dotimes [gi ng]
      (dotimes [b nb]
        (dotimes [ig 8]
          (dotimes [j 16]
            (let [base (+ (* gi NC) (quot j 4)) e (+ (* ig 4) (mod j 4))
                  nlo (src-nib base b e) nhi (src-nib (+ base 4) b e)]
              (aset wqi (+ (* gi nb 128) (* b 128) (* ig 16) j)
                    (unchecked-byte (bit-or nlo (bit-shift-left nhi 4)))))))
        (dotimes [r NC]
          (aset wsi (+ (* gi nb NC) (* b NC) r) (aget ws (+ (* (+ (* gi NC) r) nb) b))))))
    {:wqi wqi :wsi wsi}))

(defn emit-qmatmul-stream
  "Emit the :stream-gemv decode kernel over a repack-stream weight. One contiguous
  weight stream; per block an int32 lane-packed accumulator (8 dpbusd, no shuffle/
  blend), the -ZP fold via the activation block-sum, one fmadd of (col×act scale).
  Column i → lane i, so the store needs no permute. Signature:
  void NAME(xq,xs,xsum,wqi,wsi,y,in,out)."
  [name {:keys [zero-point]} mac]
  (let [zp (str zero-point)]
    (str "#include <stdint.h>\n" i8dot-helper
         "void " name "(const int8_t* xq, const float* xs, const int* xsum,\n"
         "    const uint8_t* restrict wqi, const float* restrict wsi, float* y, int in, int out,\n"
         "    int o_start, int o_count) {\n"
         "  int nb = in/32; const __m256i _m4 = _mm256_set1_epi8(0x0F); int oend = o_start + o_count;\n"
         "  for (int g = o_start; g + 8 <= oend; g += 8) {\n"
         "    long gi = g/8;\n"
         "    const uint8_t* restrict wg = wqi + gi*(long)nb*128;\n"
         "    const float* restrict sg = wsi + gi*(long)nb*8;\n"
         "    __m256 acc = _mm256_setzero_ps();\n"
         "    for (int b = 0; b < nb; b++) {\n"
         "      const uint8_t* wb = wg + b*128; const int8_t* xb = xq + b*32;\n"
         "      __m256i ia = _mm256_setzero_si256();\n"
         "      for (int ig = 0; ig < 8; ig++) {\n"
         "        __m128i raw = _mm_loadu_si128((const __m128i*)(wb + ig*16));\n"
         "        __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw,4), raw), _m4);\n"
         "        __m256i ab = _mm256_set1_epi32(*(const int*)(xb + ig*4));\n"
         "        ia = _i8dot_acc(ia, nib, ab);\n"
         "      }\n"
         "      ia = _mm256_sub_epi32(ia, _mm256_set1_epi32(" zp " * xsum[b]));\n"
         "      __m256 cs = _mm256_loadu_ps(sg + b*8);\n"
         "      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia), _mm256_mul_ps(cs, _mm256_set1_ps(xs[b])), acc);\n"
         "    }\n"
         "    _mm256_storeu_ps(y + g, acc);\n"
         "  }\n}\n")))

(defn classify-regime
  "Roofline regime classifier (stub). Quant matmul reuses each weight ~M times: decode
  (M=1) is memory-bound → :stream-gemv. Prefill (M>1) is compute-bound → :tile-gemm
  (register-block tile, not yet implemented). The default schedule per op lives here."
  [M]
  (if (<= (long M) 1) :stream-gemv :stream-gemv))

(def schedules
  "schedule-key → {:repack :emit :n-ptrs :n-ints}. Only the memory-bound :stream-gemv
  (decode GEMV) is implemented; :tile-gemm and GPU variants extend this map."
  {:stream-gemv {:repack repack-stream :emit emit-qmatmul-stream :n-ptrs 6 :n-ints 4}})

;; ---- persistent spin-barrier pool (the ggml model, in the JVM) -----------------
;; Decode issues ~127 tiny matmuls/token. Forking an Executor per call (invokeAll)
;; or entering an OpenMP region per call costs more than the work — measured WORSE
;; than serial. llama.cpp instead keeps a PERSISTENT thread team that spins on an
;; atomic between ops; kickoff is one atomic store, sync is a spin-barrier. We do the
;; same: N-1 workers spin on `gen`; each matmul posts a task + bumps `gen`, every
;; worker runs its output-row slice via a concurrent critical() call (these scale
;; ~3.16x/4t over shared heap arrays — verified), then a spin-barrier on `arrive`.
(definterface IParTask (^void runSlice [^int wid ^int n]))

(def ^:private pool-threads
  (max 1 (or (some-> (System/getenv "RASTER_DECODE_THREADS") Integer/parseInt) 1)))

(defonce ^:private the-pool
  (delay
    (let [n     pool-threads
          gen   (java.util.concurrent.atomic.AtomicLong. 0)
          arrive (java.util.concurrent.atomic.AtomicLong. 0)
          task  (java.util.concurrent.atomic.AtomicReference.)]
      (dotimes [w (dec n)]
        (let [wid (inc w)]
          (doto (Thread.
                 (fn []
                   (let [g (long-array 1)]
                     (loop []
                       (let [cur (.get gen)]
                         (if (== cur (aget g 0))
                           (do (Thread/onSpinWait) (recur))
                           (do (aset g 0 cur)
                               (.runSlice ^IParTask (.get task) wid n)
                               (.incrementAndGet arrive)
                               (recur))))))))
            (.setDaemon true) (.setName (str "raster-decode-" wid)) (.start))))
      {:gen gen :arrive arrive :task task :n n})))

(defn run-par!
  "Run `t` across the persistent pool: worker i of n runs (.runSlice t i n). The
  caller is worker 0; it busy-waits on the spin-barrier until the n-1 workers finish."
  [^IParTask t]
  (let [{:keys [^java.util.concurrent.atomic.AtomicLong gen
                ^java.util.concurrent.atomic.AtomicLong arrive
                ^java.util.concurrent.atomic.AtomicReference task n]} @the-pool]
    (if (== 1 (int n))
      (.runSlice t 0 1)
      (do (.set arrive 0)
          (.set task t)            ; volatile store, visible before the gen bump
          (.incrementAndGet gen)   ; full fence — releases the spinning workers
          (.runSlice t 0 (int n))  ; main = worker 0
          (let [need (dec (long n))]
            (while (< (.get arrive) need) (Thread/onSpinWait)))))))

(defn run-par-fn!
  "Public pooling entry for consumers (e.g. the decode loop's attention): runs
  (f wid n) on every worker of the persistent pool, f writing a disjoint slice.
  Serial at 1 thread."
  [f]
  (if (== 1 (int pool-threads))
    (f 0 1)
    (run-par! (reify IParTask (runSlice [_ wid n] (f wid n))))))

(defn quantize-act-i8-par
  "Pooled M=1 quantize-act-i8: the nb blocks are independent, so split them across
  the SAME persistent pool the matmul uses — the otherwise-idle workers do the
  quantization instead of spinning. Falls back to serial at 1 thread."
  [^floats x in]
  (let [nb (quot (int in) 32)
        xq (byte-array (int in)) xs (float-array nb) xsum (int-array nb)]
    (if (== 1 (int pool-threads))
      (quantize-act-blocks! x xq xs xsum 0 nb)
      (run-par!
       (reify IParTask
         (runSlice [_ wid n]
           (let [chunk (quot (+ nb (dec (int n))) (int n))
                 r0 (* (int wid) chunk)
                 cnt (max 0 (min chunk (- nb r0)))]
             (when (pos? cnt) (quantize-act-blocks! x xq xs xsum r0 cnt)))))))
    {:xq xq :xs xs :xsum xsum}))

(defn- matmul-slice-task
  "An IParTask: worker wid of n streams its disjoint output-row-group slice of one
  :stream-gemv into the shared y (groups are NC=8 cols; o_start/o_count in cols)."
  ^IParTask [k xq xs xsum wqi wsi y in out]
  (let [ng (quot (int out) 8)]
    (reify IParTask
      (runSlice [_ wid n]
        (let [chunk (quot (+ ng (dec (int n))) (int n))
              g0 (* (int wid) chunk)
              cnt (max 0 (min chunk (- ng g0)))]
          (when (pos? cnt)
            (k xq xs xsum wqi wsi y in out (* g0 8) (* cnt 8))))))))

(defn compile-qmatmul-stream
  "Compile the :stream-gemv decode kernel. Returns a fn (xq xs xsum wqi wsi y in out)
  over a repack-stream weight. Each call splits its output-row groups across the
  persistent spin-barrier pool (RASTER_DECODE_THREADS, default 1) — every worker
  streams a disjoint weight slice into a disjoint y slice via a concurrent critical
  call, synchronized by a spin-barrier. The n-threads arg is kept for API symmetry."
  ([name format mac] (compile-qmatmul-stream name format mac 1))
  ([name format mac _n-threads]
   (let [k (cpu/load-kernel (cpu/compile-source! (emit-qmatmul-stream name format mac)) name 6 4)]
     (fn [xq xs xsum wqi wsi y in out]
       (if (== 1 (int pool-threads))
         (k xq xs xsum wqi wsi y in out 0 out)
         (run-par! (matmul-slice-task k xq xs xsum wqi wsi y in out)))))))

;; ---- (4) GPU lowering: the SAME format descriptor → OpenCL for the Arc ----
;; The widening-i8-dot primitive's GPU lowering: one work-item per output row,
;; scalar int8 MAC (SIMT supplies the parallelism — `dot()`/DPAS via
;; cl_khr_integer_dot_product / cl_intel_subgroup_matrix_multiply_accumulate is
;; the peak follow-on, the GPU analogue of maddubs). Same data descriptor drives
;; the unpack; only the scaffold (work-item vs host loop) and MAC spelling differ.
(defn emit-qmatmul-opencl
  "Emit the OpenCL kernel for FORMAT. Compiles for the Arc (ocloc -device lnl)."
  [name {:keys [block zero-point weight-bits]}]
  (assert (= weight-bits 4) "opencl unpack described for 4-bit")
  (str "__kernel void " name "(__global const char* xq, __global const float* xs,\n"
       "    __global const int* xsum, __global const uchar* wq,\n"
       "    __global const float* ws, __global float* y, int in_, int out_) {\n"
       "  int o = get_global_id(0); if (o >= out_) return;\n"
       "  int nb = in_/" block ";\n"
       "  __global const uchar* wrow = wq + (long)o*in_/2;\n"
       "  __global const float* wsr = ws + (long)o*nb;\n"
       "  float acc = 0.f, corr = 0.f;\n"
       "  for (int b = 0; b < nb; b++) {\n"
       "    __global const uchar* wb = wrow + b*16;\n"
       "    __global const char* xb = xq + b*32;\n"
       "    int s = 0;\n"
       "    for (int k = 0; k < 16; k++) {\n"
       "      s += (int)xb[k]    * (((int)(wb[k] & 0xF)) - " zero-point ");\n"
       "      s += (int)xb[k+16] * (((int)(wb[k] >> 4)) - " zero-point ");\n"
       "    }\n"
       "    float scale = xs[b]*wsr[b]; acc += scale*s; corr += scale*xsum[b];\n"
       "  }\n"
       "  y[o] = acc - " zero-point ".f*corr;\n"
       "}\n"))
