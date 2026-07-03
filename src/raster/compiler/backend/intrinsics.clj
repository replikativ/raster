(ns raster.compiler.backend.intrinsics
  "Single source of truth for how primitive numeric operators + intrinsics lower
   to each code-generation backend (wasm, C/OpenCL, GLSL, WGSL).

   Per the compiler design rule \"Centralize operator classification\": code-gen
   backends must not each carry their own hardcoded `#{…}` / `case` of operator
   symbols. They are thin dispatchers over this one table — looking up the
   canonical op and reading the lowering for the current element type. Adding a
   new primitive op = one row here, and the row states exactly which backends
   support it (a nil/absent facet = unsupported on that backend, surfaced as a
   clear error rather than a silent miscompile).

   Canonical key: a keyword naming the operation (:+ :< :sqrt :mod …). Callers
   normalize whatever they hold — a `raster.numeric/+` symbol, a bare `+`, a
   `Math/sqrt`, a comparison name, or a mangled devirtualized impl prefix
   (`_plus__m_double_double`) — to the canonical key via `canonical`.

   Lowering facets per row:
     :arity   1 or 2
     :kind    :infix (binary operator)  | :cmp (binary, bool result)
              :fn    (function call)    | :special (custom per-backend sequence)
     :wasm    {vt → encoder-opcode-keyword}  (vt ∈ #{:f64 :f32 :i32}); absent = unsupported
     :c       C/OpenCL form — infix string, or {:fn name} (GLSL fn override via :glsl)
     :wgsl    WGSL form — infix string or {:fn name}"
  (:require [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; The table — canonical op key → lowering facets
;; ---------------------------------------------------------------------------

(defn- vt3 [f64 f32 i32] (cond-> {} f64 (assoc :f64 f64) f32 (assoc :f32 f32) i32 (assoc :i32 i32)))
(defn- math1 [f64 f32 cfn] {:arity 1 :kind :fn :wasm (vt3 f64 f32 nil) :c {:fn cfn} :wgsl {:fn cfn}})

(def table
  {;; binary arithmetic
   :+ {:arity 2 :kind :infix :wasm (vt3 :f64.add :f32.add :i32.add) :c "+" :wgsl "+"}
   :- {:arity 2 :kind :infix :wasm (vt3 :f64.sub :f32.sub :i32.sub) :c "-" :wgsl "-"}
   :* {:arity 2 :kind :infix :wasm (vt3 :f64.mul :f32.mul :i32.mul) :c "*" :wgsl "*"}
   :div {:arity 2 :kind :infix :wasm (vt3 :f64.div :f32.div :i32.div_s) :c "/" :wgsl "/"}
   ;; comparisons (bool result)
   :lt {:arity 2 :kind :cmp :wasm (vt3 :f64.lt :f32.lt :i32.lt_s) :c "<" :wgsl "<"}
   :gt {:arity 2 :kind :cmp :wasm (vt3 :f64.gt :f32.gt :i32.gt_s) :c ">" :wgsl ">"}
   :le {:arity 2 :kind :cmp :wasm (vt3 :f64.le :f32.le :i32.le_s) :c "<=" :wgsl "<="}
   :ge {:arity 2 :kind :cmp :wasm (vt3 :f64.ge :f32.ge :i32.ge_s) :c ">=" :wgsl ">="}
   :eq {:arity 2 :kind :cmp :wasm (vt3 :f64.eq :f32.eq :i32.eq) :c "==" :wgsl "=="}
   :ne {:arity 2 :kind :cmp :wasm (vt3 :f64.ne :f32.ne :i32.ne) :c "!=" :wgsl "!="}
   ;; integer bitwise + shifts (i32/i64); used by hashing / noise / bit twiddling
   :bit-and {:arity 2 :kind :infix :wasm {:i32 :i32.and :i64 :i64.and} :c "&" :wgsl "&"}
   :bit-or  {:arity 2 :kind :infix :wasm {:i32 :i32.or :i64 :i64.or} :c "|" :wgsl "|"}
   :bit-xor {:arity 2 :kind :infix :wasm {:i32 :i32.xor :i64 :i64.xor} :c "^" :wgsl "^"}
   :shl     {:arity 2 :kind :infix :wasm {:i32 :i32.shl :i64 :i64.shl} :c "<<" :wgsl "<<"}
   :shr     {:arity 2 :kind :infix :wasm {:i32 :i32.shr_s :i64 :i64.shr_s} :c ">>" :wgsl ">>"}
   :ushr    {:arity 2 :kind :infix :wasm {:i32 :i32.shr_u :i64 :i64.shr_u} :c ">>" :wgsl ">>"}
   ;; integer remainder / modulo / quotient
   :rem  {:arity 2 :kind :special :wasm {:i32 :i32.rem_s} :c "%" :wgsl "%"}
   :mod  {:arity 2 :kind :special :c :floored-mod :wgsl :floored-mod}
   :quot {:arity 2 :kind :infix :wasm {:i32 :i32.div_s} :c "/" :wgsl "/"}
   ;; math — unary
   :sqrt  (math1 :f64.sqrt :f32.sqrt "sqrt")
   :abs   {:arity 1 :kind :fn :wasm (vt3 :f64.abs :f32.abs nil) :c {:fn "fabs" :glsl "abs"} :wgsl {:fn "abs"}}
   :floor (math1 :f64.floor :f32.floor "floor")
   :ceil  {:arity 1 :kind :fn :wasm :poly :c {:fn "ceil"} :wgsl {:fn "ceil"}}   ; -floor(-x)
   :trunc {:arity 1 :kind :fn :wasm (vt3 :f64.trunc :f32.trunc nil) :c {:fn "trunc"} :wgsl {:fn "trunc"}}
   :round {:arity 1 :kind :fn :c {:fn "round"} :wgsl {:fn "round"}}    ; returns int — no wasm
   :neg   {:arity 1 :kind :fn :wasm (vt3 :f64.neg :f32.neg nil) :c {:prefix "-"} :wgsl {:prefix "-"}}
   ;; math — binary
   :min {:arity 2 :kind :fn :wasm (vt3 :f64.min :f32.min nil) :c {:fn "fmin" :glsl "min"} :wgsl {:fn "min"}}
   :max {:arity 2 :kind :fn :wasm (vt3 :f64.max :f32.max nil) :c {:fn "fmax" :glsl "max"} :wgsl {:fn "max"}}
   ;; transcendentals — wasm has no opcode; all lower to an inline polynomial
   ;; (:wasm :poly, see backend.wasm.transcendental): sin/cos/tan + exp via
   ;; squaring, log via sqrt-reduction, pow=exp(y·log x), fma=a·b+c. No bit ops.
   :sin {:arity 1 :kind :fn :wasm :poly :c {:fn "sin"} :wgsl {:fn "sin"}}
   :cos {:arity 1 :kind :fn :wasm :poly :c {:fn "cos"} :wgsl {:fn "cos"}}
   :tan {:arity 1 :kind :fn :wasm :poly :c {:fn "tan"} :wgsl {:fn "tan"}}
   :exp {:arity 1 :kind :fn :wasm :poly :c {:fn "exp"} :wgsl {:fn "exp"}}
   :log {:arity 1 :kind :fn :wasm :poly :c {:fn "log"} :wgsl {:fn "log"}}
   :pow {:arity 2 :kind :fn :wasm :poly :c {:fn "pow"} :wgsl {:fn "pow"}}
   :fma {:arity 3 :kind :fn :wasm :poly :c {:fn "fma"} :wgsl {:fn "fma"}}
   ;; widening int8 multiply-accumulate: int8×int8 → int32 lane accumulate. The
   ;; ONE canonical quant primitive; scalar :c is a plain int-widening mul-add,
   ;; the vector lowering is a fixed intrinsic SEQUENCE per ISA (AVX2
   ;; maddubs→madd→add, wasm i32x4.dot_i16x8_s, GPU dp4a) — see simd-widen below.
   :wi8-dot {:arity 2 :kind :widening-mac :wasm {:i32 :i32x4.dot-i16x8-s}}
   ;; broader elementary set — all wasm via composition/polynomial (see
   ;; backend.wasm.transcendental); GPU facet set only where it's a builtin (else
   ;; omitted → GPU keeps its generated-helper fallback, no regression).
   :asin  {:arity 1 :kind :fn :wasm :poly :c {:fn "asin"} :wgsl {:fn "asin"}}
   :acos  {:arity 1 :kind :fn :wasm :poly :c {:fn "acos"} :wgsl {:fn "acos"}}
   :atan  {:arity 1 :kind :fn :wasm :poly :c {:fn "atan"} :wgsl {:fn "atan"}}
   :atan2 {:arity 2 :kind :fn :wasm :poly :c {:fn "atan2"} :wgsl {:fn "atan2"}}
   :sinh  {:arity 1 :kind :fn :wasm :poly :c {:fn "sinh"} :wgsl {:fn "sinh"}}
   :cosh  {:arity 1 :kind :fn :wasm :poly :c {:fn "cosh"} :wgsl {:fn "cosh"}}
   :tanh  {:arity 1 :kind :fn :wasm :poly :c {:fn "tanh"} :wgsl {:fn "tanh"}}
   :asinh {:arity 1 :kind :fn :wasm :poly :c {:fn "asinh"} :wgsl {:fn "asinh"}}
   :acosh {:arity 1 :kind :fn :wasm :poly :c {:fn "acosh"} :wgsl {:fn "acosh"}}
   :atanh {:arity 1 :kind :fn :wasm :poly :c {:fn "atanh"} :wgsl {:fn "atanh"}}
   :cbrt  {:arity 1 :kind :fn :wasm :poly :c {:fn "cbrt"}}
   :log2  {:arity 1 :kind :fn :wasm :poly :c {:fn "log2"} :wgsl {:fn "log2"}}
   :log10 {:arity 1 :kind :fn :wasm :poly :c {:fn "log10"}}
   :exp2  {:arity 1 :kind :fn :wasm :poly :c {:fn "exp2"} :wgsl {:fn "exp2"}}
   :exp10 {:arity 1 :kind :fn :wasm :poly}
   :expm1 {:arity 1 :kind :fn :wasm :poly :c {:fn "expm1"}}
   :log1p {:arity 1 :kind :fn :wasm :poly :c {:fn "log1p"}}
   :hypot {:arity 2 :kind :fn :wasm :poly :c {:fn "hypot"}}
   :deg2rad {:arity 1 :kind :fn :wasm :poly :wgsl {:fn "radians"}}
   :rad2deg {:arity 1 :kind :fn :wasm :poly :wgsl {:fn "degrees"}}
   :clamp {:arity 3 :kind :fn :wasm :poly :wgsl {:fn "clamp"}}
   :signum {:arity 1 :kind :fn :wasm :poly :wgsl {:fn "sign"}}
   :copysign {:arity 2 :kind :fn :wasm :poly :c {:fn "copysign"}}
   :flipsign {:arity 2 :kind :fn :wasm :poly}})

;; ---------------------------------------------------------------------------
;; Normalization: any op form → canonical key
;; ---------------------------------------------------------------------------

;; local-name → canonical key (covers raster.numeric/Math/clojure.core syms and
;; bare names; arithmetic + comparison spellings both routed here).
(def ^:private name->key
  {"+" :+ "-" :- "*" :* "/" :div
   "<" :lt ">" :gt "<=" :le ">=" :ge "==" :eq "=" :eq "not=" :ne "!=" :ne
   "rem" :rem "unchecked-remainder-int" :rem "mod" :mod "quot" :quot
   "bit-and" :bit-and "bit-or" :bit-or "bit-xor" :bit-xor
   "bit-shift-left" :shl "bit-shift-right" :shr "unsigned-bit-shift-right" :ushr
   "sqrt" :sqrt "abs" :abs "floor" :floor "ceil" :ceil "round" :round "trunc" :trunc
   "min" :min "max" :max "sin" :sin "cos" :cos "tan" :tan "exp" :exp "log" :log
   "pow" :pow "fma" :fma
   "asin" :asin "acos" :acos "atan" :atan "atan2" :atan2
   "sinh" :sinh "cosh" :cosh "tanh" :tanh "asinh" :asinh "acosh" :acosh "atanh" :atanh
   "cbrt" :cbrt "log2" :log2 "log10" :log10 "exp2" :exp2 "exp10" :exp10
   "expm1" :expm1 "log1p" :log1p "hypot" :hypot "deg2rad" :deg2rad "rad2deg" :rad2deg
   "clamp" :clamp "signum" :signum "copysign" :copysign "copySign" :copysign
   "flipsign" :flipsign "wi8-dot" :wi8-dot})

;; mangled devirtualized prefix → canonical key (e.g. _plus__m_double_double)
(def ^:private mangled-prefix->key
  {"_plus_" :+ "_minus_" :- "_star_" :* "_div_" :div
   "_lt_" :lt "_gt_" :gt "_lteq_" :le "_gteq_" :ge "_eq_" :eq})

(defn canonical
  "Normalize an op form to its canonical key, or nil if not a known intrinsic.
   Accepts a symbol (qualified/bare/Math), a string name, or a keyword."
  [op]
  (cond
    (keyword? op) (when (contains? table op) op)
    (string? op)  (name->key op)
    (symbol? op)  (or (name->key (name op))
                      (when-let [i (clojure.string/index-of (name op) "_m_")]
                        (mangled-prefix->key (subs (name op) 0 i))))
    :else nil))

(defn descriptor [op] (get table (canonical op)))

;; ---------------------------------------------------------------------------
;; wasm accessor — canonical key + element vt → encoder opcode keyword
;; ---------------------------------------------------------------------------
(defn wasm-op
  "Encoder opcode keyword for op at element valtype vt, or nil if the op has no
   wasm lowering for that vt (caller surfaces a clear error)."
  [op vt]
  (let [w (get-in table [(canonical op) :wasm])]
    (when (map? w) (get w vt))))

(defn wasm-poly?
  "True when op lowers to an inline polynomial on wasm (transcendentals with no
   native opcode — see backend.wasm.transcendental)."
  [op]
  (= :poly (get-in table [(canonical op) :wasm])))

(defn kind [op] (:kind (descriptor op)))
(defn arity [op] (:arity (descriptor op)))

;; ---------------------------------------------------------------------------
;; C / OpenCL / GLSL accessor — for the GPU c-emit backend.
;; Resolves a *mangled* devirtualized impl name (prefix before _m_) and returns
;; the c-emit consumption shape {:kind :infix|:fn|:floored-mod :op str}, or nil
;; (→ caller falls back to a generated helper call). glsl? picks GLSL fn names.
;; ---------------------------------------------------------------------------
(defn- desc->c-shape
  "Descriptor map → the c-emit consumption shape {:kind :infix/:fn/:floored-mod :op}."
  [d glsl?]
  (when d
    (let [c (:c d)]
      (cond
        (= :floored-mod c) {:kind :floored-mod}
        (string? c)        {:kind :infix :op c}
        (and (map? c) (:fn c)) {:kind :fn :op (if (and glsl? (:glsl c)) (:glsl c) (:fn c))}
        :else nil))))

(defn op->c-lowering
  "C lowering for a SEMANTIC op (symbol/keyword, e.g. raster.numeric/* or :*) — the
  clean metadata-driven path, read from a form's :raster.op/original. nil if not an
  intrinsic."
  [op glsl?]
  (desc->c-shape (descriptor op) glsl?))

(defn c-lowering
  "C lowering from a MANGLED devirtualized impl name — the fallback when no
  :raster.op/original metadata is present. Prefer op->c-lowering."
  [mangled-name glsl?]
  (when-let [i (str/index-of mangled-name "_m_")]
    ;; prefix before `_m_` is an operator prefix (_star_, via mangled-prefix->key) or a
    ;; function name (sqrt, via name->key in descriptor).
    (let [prefix (subs mangled-name 0 i)]
      (desc->c-shape (descriptor (or (mangled-prefix->key prefix) prefix)) glsl?))))

;; ---------------------------------------------------------------------------
;; Vector (SIMD) facet — the lane-parallel analog of the :c facet, for the
;; explicit CPU-C SIMD emitter (emit-c-vexpr). Two parts, mirroring the JVM
;; segop_simd split (per-type scaffolding vs per-op VectorOperators):
;;   simd-isa    per-ELEMENT-TYPE scaffolding (vector ctype, lane count, and the
;;               broadcast/load/store/zero intrinsics) keyed by ISA → elem.
;;   simd-ops    per-OP lanewise intrinsic spelling keyed by ISA → op → elem.
;;   simd-widen  the widening int8-MAC intrinsic SEQUENCE (int8×int8→int32),
;;               keyed by ISA — the :wi8-dot canonical op's vector lowering.
;; ISA is the target-aware knob (:avx2 8×f32 / :avx512 16×f32 / :neon 4×f32),
;; selected from core.hardware's HardwareDescriptor — never hardcoded at a call
;; site. An absent facet = op/type not vectorizable on that ISA, so the caller
;; keeps the scalar loop (a clear gap, never a silent miscompile).
;;
;; NOTE: op_descriptor's simd-*-ops (the JVM :jvm-vector spelling) are NOT folded
;; in here yet — that convergence retires the duplicate JVM/GPU/wasm recognizers
;; and is the LAST cleanup step (#27 step 5'); keeping this purely additive now.
;; ---------------------------------------------------------------------------

(def simd-isa
  {:avx2
   {:f64 {:vtype "__m256d" :lanes 4 :set1 "_mm256_set1_pd" :loadu "_mm256_loadu_pd"
          :storeu "_mm256_storeu_pd" :setzero "_mm256_setzero_pd"}
    :f32 {:vtype "__m256"  :lanes 8 :set1 "_mm256_set1_ps" :loadu "_mm256_loadu_ps"
          :storeu "_mm256_storeu_ps" :setzero "_mm256_setzero_ps"
          ;; widening conversion i32×8 → f32×8 (the int8-MAC/quant fold boundary)
          :from-i32 "_mm256_cvtepi32_ps"}
    ;; int32 lanes — the accumulator type for the widening int8-MAC
    :i32 {:vtype "__m256i" :lanes 8 :set1 "_mm256_set1_epi32" :loadu "_mm256_loadu_si256"
          :storeu "_mm256_storeu_si256" :setzero "_mm256_setzero_si256"}}})

(def ^:private simd-ops
  {:avx2
   {:+   {:f64 "_mm256_add_pd" :f32 "_mm256_add_ps" :i32 "_mm256_add_epi32"}
    :-   {:f64 "_mm256_sub_pd" :f32 "_mm256_sub_ps" :i32 "_mm256_sub_epi32"}
    :*   {:f64 "_mm256_mul_pd" :f32 "_mm256_mul_ps"}
    :div {:f64 "_mm256_div_pd" :f32 "_mm256_div_ps"}
    :min {:f64 "_mm256_min_pd" :f32 "_mm256_min_ps"}
    :max {:f64 "_mm256_max_pd" :f32 "_mm256_max_ps"}
    :sqrt {:f64 "_mm256_sqrt_pd" :f32 "_mm256_sqrt_ps"}
    ;; a·b+c fused; only where the ISA has a true FMA (AVX2 implies FMA3)
    :fma {:f64 "_mm256_fmadd_pd" :f32 "_mm256_fmadd_ps"}}})

(def ^:private simd-widen
  "Widening int8-MAC (:wi8-dot) vector lowering per ISA. On AVX2 it is the
   maddubs→madd→add sequence: uint8×int8 pairwise → int16, adjacent int16 → int32,
   accumulate into the int32 lane accumulator (:acc)."
  {:avx2 {:mul "_mm256_maddubs_epi16" :widen "_mm256_madd_epi16" :acc "_mm256_add_epi32"
          :ones "_mm256_set1_epi16"}})

(defn simd-type-info
  "Per-element-type SIMD scaffolding for an ISA — the vector ctype, lane count,
   and set1/loadu/storeu/setzero intrinsic names — or nil if unsupported."
  [isa elem]
  (get-in simd-isa [isa elem]))

(defn simd-lanes
  "SIMD lane count for element type elem on isa (8 for f32/AVX2, 4 for f64), or nil."
  [isa elem]
  (:lanes (simd-type-info isa elem)))

(defn simd-op
  "Lanewise vector intrinsic name for op at element type elem on isa, or nil if the
   op has no vector lowering there (caller keeps the scalar loop). op is any form
   accepted by `canonical` (symbol/string/keyword)."
  [isa op elem]
  (get-in simd-ops [isa (canonical op) elem]))

(defn simd-widen-seq
  "The widening int8-MAC intrinsic sequence for isa (:mul/:widen/:acc/:ones), or nil."
  [isa]
  (get simd-widen isa))
