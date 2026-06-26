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

;; ---- FORMAT DESCRIPTORS (data) ----
(def q4-0
  "Symmetric 4-bit, block 32, two nibbles/byte, interleaved (lo=w[0..15],
  hi=w[16..31]), zero-point 8. (= GGUF Q4_0 quant level.)"
  {:name "q4_0" :block 32 :weight-bits 4 :pack :nibble-interleaved
   :zero-point 8 :symmetric true :act-type :i8})

;; ---- unpack snippet generated FROM the descriptor (32 unsigned weights -> wqv) ----
(defn- emit-unpack [{:keys [weight-bits pack]}]
  (case [weight-bits pack]
    [4 :nibble-interleaved]
    (str "      __m128i w  = _mm_loadu_si128((const __m128i*)(wrow + b*16));\n"
         "      __m128i lo = _mm_and_si128(w, _mm_set1_epi8(0x0F));\n"
         "      __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), _mm_set1_epi8(0x0F));\n"
         "      __m256i wqv = _mm256_set_m128i(hi, lo);\n")
    (throw (ex-info "unpack not described for format" {:weight-bits weight-bits :pack pack}))))

;; ---- widening-i8-dot PRIMITIVE: per-ISA lowerings (the one seam) ----
(def ^:private mac-lowerings
  {;; AVX2: maddubs(u8 weight, s8 act) -> int16 pairwise -> int32 (8 lanes)
   :maddubs {:includes "#include <immintrin.h>\n"
             :consts   "  const __m256i _ones = _mm256_set1_epi16(1);\n"
             :dot      "      __m256i xv = _mm256_loadu_si256((const __m256i*)(xrow + b*32));\n"
                       ;; result: __m256i s32 (8 int32 partials of Σ wq*x)
             :accum    (str "      __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(wqv, xv), _ones);\n"
                            "      float scale = xsr[b]*wsr[b];\n"
                            "      facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s32), _mm256_set1_ps(scale), facc);\n")
             :acc-decl "    __m256 facc = _mm256_setzero_ps();\n"
             :reduce   (str "    __m128 v = _mm_add_ps(_mm256_castps256_ps128(facc), _mm256_extractf128_ps(facc,1));\n"
                            "    v = _mm_hadd_ps(v, v); v = _mm_hadd_ps(v, v);\n"
                            "    float dot = _mm_cvtss_f32(v);\n")}})

;; ---- compose the full kernel from FORMAT descriptor + MAC primitive ----
(defn emit-qmatmul
  "Emit a quantized matmul C kernel for FORMAT (a descriptor) and MAC (a primitive
  lowering key). Signature:
    void NAME(const int8_t* xq, const float* xs, const int* xsum,
              const uint8_t* wq, const float* ws, float* y, int M, int in, int out)"
  [name format mac]
  (let [{:keys [block zero-point]} format
        m (mac-lowerings mac)]
    (str "#include <stdint.h>\n" (:includes m)
         "void " name "(const int8_t* xq, const float* xs, const int* xsum,\n"
         "    const uint8_t* wq, const float* ws, float* y, int M, int in, int out) {\n"
         "  int nb = in/" block ";\n" (:consts m)
         "  for (int mm = 0; mm < M; mm++) {\n"
         "    const int8_t* xrow0 = xq + (long)mm*in;\n"
         "    const float* xsr = xs + (long)mm*nb; const int* xsm = xsum + (long)mm*nb;\n"
         "    for (int o = 0; o < out; o++) {\n"
         (:acc-decl m)
         "      float corr = 0.f;\n"
         "      const uint8_t* wrow = wq + (long)o*in/2; const float* wsr = ws + (long)o*nb;\n"
         "      const int8_t* xrow = xrow0;\n"
         "      for (int b = 0; b < nb; b++) {\n"
         (emit-unpack format)
         (:dot m)
         (:accum m)
         "        corr += xsr[b]*wsr[b]*xsm[b];\n"
         "      }\n"
         (:reduce m)
         "      y[(long)mm*out + o] = dot - " zero-point ".f*corr;\n"
         "    }\n  }\n}\n")))

(defn compile-qmatmul
  "Emit + compile + load a quantized matmul. Returns a fn
  (xq xs xsum wq ws y M in out) calling the native kernel."
  [name format mac]
  (let [src (emit-qmatmul name format mac)]
    (cpu/load-kernel (cpu/compile-source! src) name 6 3)))

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
