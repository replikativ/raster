(ns raster.compiler.backend.cpu.csimd
  "Explicit CPU-C SIMD emitter — the C analog of jvm/segop_simd.

   Consumes the SAME SegRed/SegMap records that par_simd builds on-demand from
   par-forms (par-form->soac->lower-soac), and emits AVX2 __m256 intrinsic C via
   the intrinsics VECTOR FACET (raster.compiler.backend.intrinsics/simd-*). It
   REUSES segop_simd's backend-neutral lane recognizers (aget-form?, simd-able?,
   collect-load-sites, value-position-arrays) rather than re-deriving them — the
   recognizer is shared, only the emit target differs (C intrinsics vs JVM Vector
   API). This is the CPU-C half of #27; the recognizer's eventual extraction to a
   backend-neutral ns (retiring the require on segop_simd) is the later cleanup.

   ISA is the hardware knob (:avx2 today; :avx512/:neon slot in as more facet
   rows). Only the reduction op `+` with a multiply/identity lane body (dot,
   sum-of-squares, plain sum) is vectorized here — the dot/quant-fold money case;
   anything else returns nil so the caller keeps the scalar loop (never a silent
   miscompile)."
  (:require [raster.compiler.backend.jvm.segop-simd :as ss]
            [raster.compiler.backend.intrinsics :as in]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [clojure.string :as str]))

;; element keyword → intrinsics vector-facet element key
(defn- vt-of [dtype] (case dtype :double :f64 :float :f32 :long :i32 :int :i32 nil))

(def ^:private default-n-acc
  "Independent vector accumulators to break the reduction dependency chain (hides
   FMA latency). 4 matches the JVM segop_simd default; the hardware-descriptor
   wiring (core.hardware/reduction-accumulators) replaces this constant later."
  4)

;; ---------------------------------------------------------------------------
;; Horizontal-reduce helpers (emitted once per compiled unit that uses them).
;; ---------------------------------------------------------------------------
(def simd-includes "#include <immintrin.h>\n")

(def simd-helpers
  (str
   "static inline double raster_hsum256_pd(__m256d v){\n"
   "  __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);\n"
   "  __m128d s=_mm_add_pd(lo,hi), sh=_mm_unpackhi_pd(s,s);\n"
   "  return _mm_cvtsd_f64(_mm_add_sd(s,sh));\n}\n"
   "static inline float raster_hsum256_ps(__m256 v){\n"
   "  __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);\n"
   "  __m128 s=_mm_add_ps(lo,hi); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s);\n"
   "  return _mm_cvtss_f32(s);\n}\n"))

(defn- hsum-fn [elem] (case elem :f64 "raster_hsum256_pd" :f32 "raster_hsum256_ps"))

;; ---------------------------------------------------------------------------
;; Reduction lane-body analysis (reuse the shared recognizers).
;; ---------------------------------------------------------------------------

(defn- reduction-plan
  "Extract the neutral vector plan from a SegRed's reduce-op, or nil if not a
   `+`-reduction with a vectorizable multiply/identity lane body. Mirrors the
   recognition core of segop_simd/compile-segred but backend-neutrally.
   Returns {:idx :bound :acc :init :elem-expr :factors [f1 f2]|nil :fma? bool}."
  [segred]
  (let [idx   (ss/seg-idx segred)
        bound (ss/seg-bound segred)
        {:keys [acc init lambda]} (:reduce-op segred)
        lambda (when lambda (ss/normalize-invk (ss/clean-dead-bindings lambda)))]
    (when (and acc init idx bound (seq? lambda) (>= (count lambda) 3))
      (let [op   (first lambda)
            ;; acc must appear as one addend; the other is the lane element.
            elem (cond (= (nth lambda 1) acc) (nth lambda 2)
                       (= (nth lambda 2) acc) (nth lambda 1)
                       :else nil)]
        (when (and (= op '+) elem
                   (ss/simd-able? elem idx)
                   (empty? (ss/value-position-arrays elem idx)))
          (let [factors (when (and (seq? elem) (= '* (first elem)) (= 3 (count elem))
                                   (ss/aget-form? (nth elem 1) idx)
                                   (ss/aget-form? (nth elem 2) idx))
                          [(nth elem 1) (nth elem 2)])]
            {:idx idx :bound bound :acc acc :init init
             :elem-expr elem :factors factors :fma? (boolean factors)}))))))

;; ---------------------------------------------------------------------------
;; C emission.
;; ---------------------------------------------------------------------------

(defn- scalar-aget
  "Scalar C for (aget arr affine-idx) at loop counter `jv`: arr[base + jv]."
  [arr base jv array-syms]
  (let [arrs (ce/c-symbol arr)]
    (str arrs "[" (if base
                    (str "(" (ce/emit-expr base nil array-syms "idx") ") + " jv)
                    jv) "]")))

(defn- vec-load
  "AVX2 vector load for (aget arr affine-idx) at counter `jv` + lane-block offset
   `blk` (a C int expr): loadu(&arr[base + jv + blk])."
  [ti arr base jv blk array-syms]
  (let [arrs (ce/c-symbol arr)
        off  (cond-> jv
               (not= blk "0") (as-> o (str o " + " blk))
               base (as-> o (str "(" (ce/emit-expr base nil array-syms "idx") ") + " o)))]
    (str (:loadu ti) "(&" arrs "[" off "])")))

(defn compile-segred-c
  "SegRed → a C statement block (string) that assigns the reduction result to the
   caller-declared variable `target` (a symbol; defaults to the reduce-op's acc).
   Returns {:includes :helpers :block} or nil if not vectorizable (caller keeps
   scalar). The block is self-contained (own `{ }` scope) except for the outer
   target var. `array-syms` is the set of array symbols in scope (for c-emit of
   index exprs)."
  ([segred isa array-syms] (compile-segred-c segred isa array-syms nil))
  ([segred isa array-syms target]
   (when-let [{:keys [idx bound acc init elem-expr factors]} (reduction-plan segred)]
    (let [acc (ce/c-symbol (or target acc))
          elem (vt-of (:dtype segred))
          ti   (in/simd-type-info isa elem)
          vadd (in/simd-op isa :+ elem)
          vfma (in/simd-op isa :fma elem)]
      (when (and ti vadd (or (nil? factors) vfma))
        (let [vt    (:vtype ti)
              lanes (:lanes ti)
              nacc  default-n-acc
              stride (* nacc lanes)
              accs  (mapv #(str "va" %) (range nacc))
              n-c   (ce/emit-expr bound nil array-syms "idx")
              hsum  (hsum-fn elem)
              ;; per-accumulator lane-block offset: k*lanes (k=0 → "0")
              blk   (fn [k] (str (* k lanes)))
              ;; main loop body: nacc independent FMA/mul-add accumulations
              step  (str/join
                     "\n    "
                     (for [k (range nacc)]
                       (if factors
                         (let [[f1 f2] factors
                               [a1 b1] (ss/aget-form? f1 idx)
                               [a2 b2] (ss/aget-form? f2 idx)]
                           (str (accs k) " = " vfma "("
                                (vec-load ti a1 b1 "j" (blk k) array-syms) ", "
                                (vec-load ti a2 b2 "j" (blk k) array-syms) ", "
                                (accs k) ");"))
                         ;; non-fused: elem is a single affine load → add
                         (let [[a b] (ss/aget-form? elem-expr idx)]
                           (str (accs k) " = " vadd "(" (accs k) ", "
                                (vec-load ti a b "j" (blk k) array-syms) ");")))))
              ;; combine accumulators pairwise via vadd
              combine (reduce (fn [a b] (str vadd "(" a ", " b ")")) accs)
              ;; scalar tail element at counter j
              tail-elem (if factors
                          (let [[f1 f2] factors
                                [a1 b1] (ss/aget-form? f1 idx)
                                [a2 b2] (ss/aget-form? f2 idx)]
                            (str (scalar-aget a1 b1 "j" array-syms) " * "
                                 (scalar-aget a2 b2 "j" array-syms)))
                          (let [[a b] (ss/aget-form? elem-expr idx)]
                            (scalar-aget a b "j" array-syms)))]
          {:includes simd-includes
           :helpers  simd-helpers
           :block
           (str "{\n"
                "  const int _n = " n-c ";\n"
                (apply str (for [a accs] (str "  " vt " " a " = " (:setzero ti) "();\n")))
                "  int j = 0;\n"
                "  for (; j + " stride " <= _n; j += " stride ") {\n    "
                step "\n  }\n"
                "  " vt " _vc = " combine ";\n"
                "  " acc " = (" init ") + " hsum "(_vc);\n"
                "  for (; j < _n; j++) " acc " = " acc " + (" tail-elem ");\n"
                "}")}))))))
