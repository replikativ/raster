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

(defn quantize-act-i8
  "f32 activation row(s) [M*in] → {:xq byte[M*in] :xs float[M*nb] :xsum int[M*nb]}
  per-block (32) symmetric int8, d=max/127, with per-block sum for the -8 correction."
  [^floats x M in]
  (let [nb (quot in 32)
        xq (byte-array (alength x)) xs (float-array (* M nb)) xsum (int-array (* M nb))]
    (dotimes [r (* M nb)]
      (let [base (* r 32)
            mx (loop [j 0 a 0.0] (if (< j 32) (recur (inc j) (max a (Math/abs (aget x (+ base j))))) a))
            d (/ mx 127.0) id (if (zero? d) 0.0 (/ 1.0 d))]
        (aset xs r (float d))
        (aset xsum r (int (loop [k 0 s 0]
                            (if (< k 32)
                              (let [q (max -127 (min 127 (Math/round (* (aget x (+ base k)) id))))]
                                (aset xq (+ base k) (unchecked-byte q)) (recur (inc k) (+ s (int q))))
                              s))))))
    {:xq xq :xs xs :xsum xsum}))

;; ---- FORMAT DESCRIPTORS (data) ----
(def q4-0
  "Symmetric 4-bit, block 32, two nibbles/byte, interleaved (lo=w[0..15],
  hi=w[16..31]), zero-point 8. (= GGUF Q4_0 quant level.)"
  {:name "q4_0" :block 32 :weight-bits 4 :pack :nibble-interleaved
   :zero-point 8 :symmetric true :act-type :i8})

;; ---- per-ISA pieces of the widening-i8-dot primitive (the one seam) ----
;; :maddubs = AVX2 maddubs(u8 weight, s8 act). Other ISAs (:vnni dpbusd, NEON
;; :dotprod vdotq, :scalar) are added as more entries — same descriptor, same
;; schedule, different MAC spelling.
(def ^:private mac-includes {:maddubs "#include <immintrin.h>\n"})
(def ^:private mac-consts
  {:maddubs "  const __m256i _ones = _mm256_set1_epi16(1), _m4 = _mm256_set1_epi8(0x0F);\n"})

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
         "          __m256i _s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(_wq, _xv), _ones);\n"
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
