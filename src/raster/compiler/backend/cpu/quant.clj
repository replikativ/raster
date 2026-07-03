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
            [raster.compiler.core.layout :as layout]
            [raster.compiler.backend.cpu.codegen :as cpu]))

;; ---- FORMAT DESCRIPTORS (data) — one row per quant level ----
;; The unifying trick: encode weights UNSIGNED with a zero-point (q in [0,2^bits),
;; zp = 2^(bits-1)), so the kernel's u8*s8 dpbusd + (-zp*Σact) fold reconstructs
;; dot = d_w*d_x*(dpbusd(q_w,q_x) - zp*Σact) for ANY bit width — 4-bit and 8-bit
;; share one MAC core (only the unpack: nibble vs byte). The legacy _0 family is a
;; clean parametric {bits, pack}; K-quants (Q4_K…) need per-format unpack/scale-decode.
(def q4-0
  "Symmetric 4-bit, block 32, two nibbles/byte interleaved (lo=w[k], hi=w[k+16]),
  zero-point 8. (= GGUF Q4_0 quant level.)"
  {:name "q4_0" :block 32 :weight-bits 4 :pack :nibble-interleaved
   :zero-point 8 :symmetric true :act-type :i8})

(def q8-0
  "Symmetric 8-bit, block 32, one unsigned byte/weight, zero-point 128. raster's
  dpbusd-compatible 8-bit (unsigned-offset, not GGUF's signed Q8_0 — functionally
  equivalent, same accuracy). The clean parametric proof: same kernel as q4-0 modulo
  byte-direct unpack."
  {:name "q8_0" :block 32 :weight-bits 8 :pack :byte-direct
   :zero-point 128 :symmetric true :act-type :i8})

;; ---- encode side: ONE parametric quantizer (Q4_0 is the default data row) ----
(defn quantize-weight
  "f32 weight (len % block = 0) → {:wq byte[] :ws float[]}: unsigned-offset symmetric
  quantization parameterized by the FORMAT (weight-bits, zero-point, pack). Per block,
  d = max/(-zp); the unsigned weight q = clamp(round(w/d)+zp, 0, 2^bits-1). :nibble-
  interleaved packs two weights/byte (w[k] low, w[k+half] high); :byte-direct one/byte.
  The bit width changes only the pack + range; the dpbusd kernel core is shared."
  [^floats w {:keys [block weight-bits zero-point pack]}]
  (let [len (alength w) blk (long block) nb (quot len blk) half (quot blk 2)
        zp (long zero-point) qmax (dec (bit-shift-left 1 (long weight-bits)))
        ws (float-array nb)
        wq (byte-array (if (= pack :nibble-interleaved) (quot len 2) len))]
    (dotimes [b nb]
      (let [base (* b blk)
            mx (loop [j 0 a 0.0 v 0.0]
                 (if (< j blk)
                   (let [x (aget w (+ base j)) ax (Math/abs x)]
                     (if (> ax a) (recur (inc j) ax x) (recur (inc j) a v)))
                   v))
            d (/ mx (double (- zp))) id (if (zero? d) 0.0 (/ 1.0 d))
            q (fn [x] (max 0 (min qmax (+ (Math/round (* (double x) id)) zp))))]
        (aset ws b (float d))
        (if (= pack :nibble-interleaved)
          (dotimes [k half]
            (aset wq (+ (* b half) k)
                  (unchecked-byte (bit-or (q (aget w (+ base k)))
                                          (bit-shift-left (q (aget w (+ base k half))) 4)))))
          (dotimes [k blk]
            (aset wq (+ base k) (unchecked-byte (q (aget w (+ base k)))))))))
    {:wq wq :ws ws}))

(defn quantize-weight-q4
  "f32 weight → Q4_0 {:wq byte[len/2] :ws float[len/32]}. Thin wrapper over the
  parametric quantize-weight with the q4-0 format."
  [^floats w]
  (quantize-weight w q4-0))

;; ---- K-quants: super-block format (the registry case the legacy params don't cover) ----
;; A K-quant is a SUPER-BLOCK of 256 split into sub-blocks (32 here), each with its OWN
;; scale + min, those per-sub-block scales/mins themselves quantized to 6-bit via two
;; super-block fp scales. Asymmetric: x ≈ (d·aq_j)·q + (dmin·bq_j). This is the same
;; block→scale→int-MAC skeleton, but the unpack + scale-decode are format-specific (the
;; "registry" not the flat {bits,pack} params). raster's own layout (not GGUF-byte-exact;
;; GGUF compat is a loader concern) — proves the pipeline generalizes to super-blocks.
(def q4-K
  "Asymmetric 4-bit super-block: 256 elems / 8 sub-blocks of 32; per-sub-block 6-bit
  scale (aq, unsigned) + 6-bit min (bq, signed); two fp super-scales d (of scales) and
  dmin (of mins). Reconstruct x = (d·aq)·q + (dmin·bq)."
  {:name "q4_K" :super-block 256 :subblock 32 :n-subblocks 8
   :weight-bits 4 :scale-bits 6 :has-dmin true :pack :nibble-interleaved :act-type :i8-K})

(defn quantize-weight-q4k
  "f32 weight (len % super-block = 0) → the q4-K super-block format:
   {:wq byte[len/2] (nibbles, w[k] low / w[k+16] high per sub-block)
    :da float[nsb] :db float[nsb]   (super-block scale-of-scales / scale-of-mins)
    :aq byte[nsub] :bq byte[nsub]}  (per-sub-block 6-bit scale 0..63 / min -32..31).
   Weights are quantized against the QUANTIZED sub-scales so encode/decode are consistent."
  [^floats w {:keys [super-block subblock]}]
  (let [len (alength w) sbk (long super-block) sub (long subblock)
        subs-per (quot sbk sub) nsb (quot len sbk) nsub (quot len sub) half (quot sub 2)
        wq (byte-array (quot len 2))
        da (float-array nsb) db (float-array nsb)
        aq (byte-array nsub) bq (byte-array nsub)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk)
            as (double-array subs-per) bs (double-array subs-per)]
        ;; 1. per sub-block scale a = (max-min)/15 and min b
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub))
                lo (loop [i 1 m (aget w base)] (if (< i sub) (recur (inc i) (min m (aget w (+ base i)))) m))
                hi (loop [i 1 m (aget w base)] (if (< i sub) (recur (inc i) (max m (aget w (+ base i)))) m))]
            (aset as j (/ (- hi lo) 15.0)) (aset bs j lo)))
        ;; 2. super-block scales-of-scales, quantize sub scales/mins to 6-bit
        (let [amax (areduce as i m 0.0 (max m (aget as i)))
              bmax (areduce bs i m 0.0 (max m (Math/abs (aget bs i))))
              dav (/ amax 63.0) dbv (/ bmax 31.0)
              ida (if (zero? dav) 0.0 (/ 1.0 dav))
              idb (if (zero? dbv) 0.0 (/ 1.0 dbv))]
          (aset da sb (float dav)) (aset db sb (float dbv))
          (dotimes [j subs-per]
            (let [aqj (max 0 (min 63 (Math/round (* (aget as j) ida))))
                  bqj (max -32 (min 31 (Math/round (* (aget bs j) idb))))
                  a' (* dav aqj) b' (* dbv bqj)
                  ia' (if (zero? a') 0.0 (/ 1.0 a'))
                  base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)]
              (aset aq sidx (unchecked-byte aqj))
              (aset bq sidx (unchecked-byte bqj))
              ;; 3. quantize weights against the QUANTIZED sub scale/min (consistency)
              (dotimes [k half]
                (let [q0 (max 0 (min 15 (Math/round (* (- (aget w (+ base k)) b') ia'))))
                      q1 (max 0 (min 15 (Math/round (* (- (aget w (+ base k half)) b') ia'))))]
                  (aset wq (+ (quot base 2) k)
                        (unchecked-byte (bit-or q0 (bit-shift-left q1 4)))))))))))
    {:wq wq :da da :db db :aq aq :bq bq}))

(defn dequant-q4k
  "q4-K super-block → f32[len]: x = (da·aq)·q + (db·bq) per sub-block. The inverse of
  quantize-weight-q4k (raster's own layout); the reference the fused kernel matches."
  [{:keys [wq da db aq bq]} {:keys [super-block subblock]} len]
  (let [out (float-array len) sbk (long super-block) sub (long subblock)
        subs-per (quot sbk sub) nsb (quot len sbk) half (quot sub 2)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk) dav (aget da sb) dbv (aget db sb)]
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                a' (* dav (aget aq sidx))   ;; aq in [0,63] (positive byte)
                b' (* dbv (aget bq sidx))]  ;; bq signed
            (dotimes [k half]
              (let [bv (bit-and (long (aget wq (+ (quot base 2) k))) 0xFF)]
                (aset out (+ base k)      (float (+ (* a' (bit-and bv 0xF)) b')))
                (aset out (+ base k half) (float (+ (* a' (bit-shift-right bv 4)) b')))))))))
    out))

(defn quantize-act-q8k
  "f32 activation [in] (in % super-block = 0) → q8_K for the K-quant dot: per super-block
  int8 quants + scale d=max/127, plus bsums[nsub] = per-sub-block int sum of the quants
  (the K min-term needs Σ activation per sub-block, precomputed)."
  [^floats x in {:keys [super-block subblock]}]
  (let [sbk (long super-block) sub (long subblock)
        nsb (quot (long in) sbk) nsub (quot (long in) sub) subs-per (quot sbk sub)
        xq (byte-array in) xs (float-array nsb) bsums (int-array nsub)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk)
            mx (loop [i 0 m 0.0] (if (< i sbk) (recur (inc i) (max m (Math/abs (aget x (+ sbase i))))) m))
            d (/ mx 127.0) id (if (zero? d) 0.0 (/ 1.0 d))]
        (aset xs sb (float d))
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)]
            (aset bsums sidx (int (loop [k 0 s 0]
                                    (if (< k sub)
                                      (let [qv (max -127 (min 127 (Math/round (* (aget x (+ base k)) id))))]
                                        (aset xq (+ base k) (unchecked-byte qv)) (recur (inc k) (+ s (int qv))))
                                      s))))))))
    {:xq xq :xs xs :bsums bsums}))

(defn q4k-q8k-dot
  "Reference dot of ONE Q4_K weight row · a q8_K activation — the K-quant compute the
  fused kernel will SIMD-ize. Per super-block: d_act·[Σ_sub (d·aq)·dpbusd + (dmin·bq)·bsum],
  where dpbusd = Σ_i q_w·q_act (the int8-MAC, dpbusd-able) and bsum is the precomputed
  per-sub-block activation sum. Equals dot(dequant weight, dequant act) exactly."
  [{:keys [wq da db aq bq]} {:keys [xq xs bsums]} in {:keys [super-block subblock]}]
  (let [sbk (long super-block) sub (long subblock) half (quot sub 2)
        nsb (quot (long in) sbk) subs-per (quot sbk sub)]
    (loop [sb 0 acc 0.0]
      (if (< sb nsb)
        (let [sbase (* sb sbk) dav (aget da sb) dbv (aget db sb) dact (double (aget xs sb))
              sbsum (loop [j 0 s 0.0]
                      (if (< j subs-per)
                        (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                              a' (* dav (aget aq sidx)) b' (* dbv (aget bq sidx))
                              dp (loop [k 0 d 0]
                                   (if (< k half)
                                     (let [bv (bit-and (long (aget wq (+ (quot base 2) k))) 0xFF)]
                                       (recur (inc k)
                                              (+ d (* (bit-and bv 0xF) (long (aget xq (+ base k))))
                                                 (* (bit-shift-right bv 4) (long (aget xq (+ base k half)))))))
                                     d))]
                          (recur (inc j) (+ s (* a' (double dp)) (* b' (double (aget bsums sidx))))))
                        s))]
          (recur (inc sb) (+ acc (* dact sbsum))))
        acc))))

;; ---- Q6_K: the 2nd K-quant — a DIFFERENT variant, to confirm the registry
;; generalizes across K-quants (not just one). Q6_K is SYMMETRIC (d only, no min/dmin),
;; 16×16 sub-blocks (vs Q4_K's 8×32), int8 sub-scales (vs Q4_K's 6-bit-packed), 6-bit
;; weights. Reuses the SAME skeleton + the SAME q8_K activation (parameterized to 16-elem
;; sub-blocks). The signed 6-bit weight rides the unsigned+zero-point trick (zp=32) so the
;; dpbusd core is shared. (raster stores 6-bit/byte; GGUF's 4+2 plane is a packing detail.)
(def q6-K
  "Symmetric 6-bit super-block: 256 / 16 sub-blocks of 16; per-sub-block int8 scale,
  one fp super-scale d, no min. x = (d·sc)·(q-32), q unsigned 6-bit [0,63]."
  {:name "q6_K" :super-block 256 :subblock 16 :n-subblocks 16
   :weight-bits 6 :scale-bits 8 :has-dmin false :zero-point 32
   :pack :byte-direct :act-type :i8-K})

(defn quantize-weight-q6k
  "f32 weight → q6-K: {:wq byte[len] (unsigned 6-bit) :sc byte[nsub] (int8 sub-scale)
   :ds float[nsb] (super-scale)}. Symmetric: per sub-block a=max|x|/32, super d=max(a)/127."
  [^floats w {:keys [super-block subblock zero-point]}]
  (let [len (alength w) sbk (long super-block) sub (long subblock) zp (long zero-point)
        subs-per (quot sbk sub) nsb (quot len sbk) nsub (quot len sub) qmax 63
        wq (byte-array len) sc (byte-array nsub) ds (float-array nsb)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk) as (double-array subs-per)]
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub))
                mx (loop [i 0 m 0.0] (if (< i sub) (recur (inc i) (max m (Math/abs (aget w (+ base i))))) m))]
            (aset as j (/ mx 32.0))))
        (let [amax (areduce as i m 0.0 (max m (aget as i)))
              dv (/ amax 127.0) idv (if (zero? dv) 0.0 (/ 1.0 dv))]
          (aset ds sb (float dv))
          (dotimes [j subs-per]
            (let [scj (max 0 (min 127 (Math/round (* (aget as j) idv))))
                  a' (* dv scj) ia' (if (zero? a') 0.0 (/ 1.0 a'))
                  base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)]
              (aset sc sidx (unchecked-byte scj))
              (dotimes [k sub]
                (aset wq (+ base k)
                      (unchecked-byte (max 0 (min qmax (+ (Math/round (* (aget w (+ base k)) ia')) zp)))))))))))
    {:wq wq :sc sc :ds ds}))

(defn dequant-q6k
  "q6-K → f32[len]: x = (ds·sc)·(q-zp) per sub-block."
  [{:keys [wq sc ds]} {:keys [super-block subblock zero-point]} len]
  (let [out (float-array len) sbk (long super-block) sub (long subblock) zp (long zero-point)
        subs-per (quot sbk sub) nsb (quot len sbk)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk) dv (aget ds sb)]
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                a' (* dv (long (aget sc sidx)))]
            (dotimes [k sub]
              (aset out (+ base k)
                    (float (* a' (- (bit-and (long (aget wq (+ base k))) 0xFF) zp)))))))))
    out))

(defn q6k-q8k-dot
  "Reference dot of ONE Q6_K weight row · q8_K activation: per super-block d·d_act·Σ_sub
  sc·(dpbusd - zp·bsum). Symmetric (no min term); the signed 6-bit weight via the unsigned
  +zp(32) fold reuses the shared dpbusd core. Equals dot(dequant, dequant) exactly."
  [{:keys [wq sc ds]} {:keys [xq xs bsums]} in {:keys [super-block subblock zero-point]}]
  (let [sbk (long super-block) sub (long subblock) zp (long zero-point)
        nsb (quot (long in) sbk) subs-per (quot sbk sub)]
    (loop [sb 0 acc 0.0]
      (if (< sb nsb)
        (let [sbase (* sb sbk) dv (double (aget ds sb)) dact (double (aget xs sb))
              ssum (loop [j 0 s 0.0]
                     (if (< j subs-per)
                       (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                             scj (long (aget sc sidx))
                             dp (loop [k 0 d 0]
                                  (if (< k sub)
                                    (recur (inc k) (+ d (* (bit-and (long (aget wq (+ base k))) 0xFF)
                                                           (long (aget xq (+ base k))))))
                                    d))
                             folded (- dp (* zp (long (aget bsums sidx))))]
                         (recur (inc j) (+ s (* scj (double folded)))))
                       s))]
          (recur (inc sb) (+ acc (* dv dact ssum))))
        acc))))

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

(def ^:private stream-gemv-layout
  (layout/quant-stream-layout {:vector-bits 256} q4-0)) ; AVX2 NC=8, format-derived

(defn repack-stream
  "Row-major quantized {wq, ws[out*nb]} → the :stream-gemv interleaved layout,
  where output column i lands in accumulator lane i (no shuffles, no permute). Single
  source: delegates to the descriptor-driven core.layout/repack, which the kernel's
  gather (emit-qmatmul-stream) mirrors. out must be a multiple of NC=8. One-time cost.
  Default format q4-0; pass q8-0 for the byte-direct 8-bit layout (embedder quality)."
  ([^bytes wq ^floats ws out in]
   (layout/repack stream-gemv-layout wq ws out in))
  ([^bytes wq ^floats ws out in format]
   (layout/repack (layout/quant-stream-layout {:vector-bits 256} format) wq ws out in)))

(defn emit-qmatmul-stream
  "Emit the :stream-gemv decode kernel — the descriptor-driven SCHEDULE for the Q4
  interleaved layout. The structure (column tile NC, bytes/input-group, input-groups,
  k-group, block) is sourced from the layout descriptor (`stream-gemv-layout`), so the
  kernel's gather provably matches `repack-stream`'s scatter — both read ONE descriptor,
  no drifting convention. Register-resident: per block an int32 lane-packed accumulator
  (NC dpbusd, no shuffle/blend), the -ZP fold via the activation block-sum, one fmadd of
  (col×act scale). Column i → lane i, so the store needs no permute. NC=8 is the AVX2
  path (the f32 lane count); AVX-512 (NC=16) / NEON (NC=4) are intrinsic-family variants
  keyed off the same descriptor. Signature: void NAME(xq,xs,xsum,wqi,wsi,y,in,out)."
  [name {:keys [zero-point] :as format} mac]
  (let [{:keys [tile block bytes-per-igroup igroups k-group pack]}
        (if format (layout/quant-stream-layout {:vector-bits 256} format) stream-gemv-layout)
        nc tile
        bpi bytes-per-igroup
        blk (* igroups bpi)        ;; bytes per (group, block): 128 for Q4/NC=8, 256 for Q8
        byte-direct? (= pack :byte-direct)
        zp (str zero-point)]
    (assert (= nc 8)
            (str "emit-qmatmul-stream: only NC=8 (AVX2 __m256) implemented; NC=" nc
                 " (AVX-512/NEON) is an intrinsic-family variant — TODO"))
    (str "#include <stdint.h>\n" i8dot-helper
         "void " name "(const int8_t* xq, const float* xs, const int* xsum,\n"
         "    const uint8_t* restrict wqi, const float* restrict wsi, float* y, int in, int out,\n"
         "    int o_start, int o_count) {\n"
         "  int nb = in/" block ";"
         (when-not byte-direct? " const __m256i _m4 = _mm256_set1_epi8(0x0F);")
         " int oend = o_start + o_count;\n"
         "  for (int g = o_start; g + " nc " <= oend; g += " nc ") {\n"
         "    long gi = g/" nc ";\n"
         "    const uint8_t* restrict wg = wqi + gi*(long)nb*" blk ";\n"
         "    const float* restrict sg = wsi + gi*(long)nb*" nc ";\n"
         "    __m256 acc = _mm256_setzero_ps();\n"
         "    for (int b = 0; b < nb; b++) {\n"
         "      const uint8_t* wb = wg + b*" blk "; const int8_t* xb = xq + b*" block ";\n"
         "      __m256i ia = _mm256_setzero_si256();\n"
         "      for (int ig = 0; ig < " igroups "; ig++) {\n"
         (if byte-direct?
           ;; 8-bit: one unsigned byte per weight — direct 32-byte load, no unpack
           (str "        __m256i wv = _mm256_loadu_si256((const __m256i*)(wb + ig*" bpi "));\n")
           (str "        __m128i raw = _mm_loadu_si128((const __m128i*)(wb + ig*" bpi "));\n"
                "        __m256i wv = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw,4), raw), _m4);\n"))
         "        __m256i ab = _mm256_set1_epi32(*(const int*)(xb + ig*" k-group "));\n"
         "        ia = _i8dot_acc(ia, wv, ab);\n"
         "      }\n"
         "      ia = _mm256_sub_epi32(ia, _mm256_set1_epi32(" zp " * xsum[b]));\n"
         "      __m256 cs = _mm256_loadu_ps(sg + b*" nc ");\n"
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

;; Workers spin-poll for new work this many iterations, THEN park (sleep). During
;; active decode, dispatches arrive faster than this window, so workers stay spinning
;; (low latency); when decode stops/idle they park — no idle core-burn, and robust
;; under contention (a busy machine doesn't get 4 cores pinned by a dormant pool).
(def ^:private spin-limit 100000)

(defonce ^:private the-pool
  (delay
    (let [n     pool-threads
          gen   (java.util.concurrent.atomic.AtomicLong. 0)
          arrive (java.util.concurrent.atomic.AtomicLong. 0)
          task  (java.util.concurrent.atomic.AtomicReference.)
          threads (object-array (dec n))]   ;; worker handles, for unpark
      (dotimes [w (dec n)]
        (let [wid (inc w)
              th (Thread.
                  (fn []
                    (let [g (long-array 1)]
                      (loop []
                        (let [cur (.get gen)]
                          (if (== cur (aget g 0))
                            ;; no work: spin a bounded window, then park until unparked.
                            ;; park()/unpark() permits make this lost-wakeup-safe (an
                            ;; unpark before park returns immediately).
                            (do (loop [s 0]
                                  (when (and (< s spin-limit) (== (.get gen) (aget g 0)))
                                    (Thread/onSpinWait) (recur (inc s))))
                                (when (== (.get gen) (aget g 0))
                                  (java.util.concurrent.locks.LockSupport/park))
                                (recur))
                            (do (aset g 0 cur)
                                (.runSlice ^IParTask (.get task) wid n)
                                (.incrementAndGet arrive)
                                (recur))))))))]
          (aset threads w th)
          (doto th (.setDaemon true) (.setName (str "raster-decode-" wid)) (.start))))
      {:gen gen :arrive arrive :task task :n n :threads threads})))

(defn run-par!
  "Run `t` across the persistent pool: worker i of n runs (.runSlice t i n). The
  caller is worker 0; it busy-waits on the spin-barrier until the n-1 workers finish."
  [^IParTask t]
  (let [{:keys [^java.util.concurrent.atomic.AtomicLong gen
                ^java.util.concurrent.atomic.AtomicLong arrive
                ^java.util.concurrent.atomic.AtomicReference task n
                ^objects threads]} @the-pool]
    (if (== 1 (int n))
      (.runSlice t 0 1)
      (do (.set arrive 0)
          (.set task t)            ; volatile store, visible before the gen bump
          (.incrementAndGet gen)   ; full fence — releases the spinning workers
          (dotimes [w (dec (int n))]   ; wake any parked workers
            (java.util.concurrent.locks.LockSupport/unpark (aget threads w)))
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
       "      // raw unsigned nibble * signed activation (the dp4a/dpbusd convention);\n"
       "      // the zero-point is folded ONCE below via corr = -ZP*sum(xq).\n"
       "      s += (int)xb[k]    * ((int)(wb[k] & 0xF));\n"
       "      s += (int)xb[k+16] * ((int)(wb[k] >> 4));\n"
       "    }\n"
       "    float scale = xs[b]*wsr[b]; acc += scale*s; corr += scale*xsum[b];\n"
       "  }\n"
       "  y[o] = acc - " zero-point ".f*corr;\n"
       "}\n"))
