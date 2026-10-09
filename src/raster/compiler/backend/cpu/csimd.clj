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
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [clojure.string :as str]
            [clojure.walk :as walk]))

(defn- inline-lets
  "Beta-reduce a lane lambda's let bindings into its body so the lane expression is a single form
   (no manual kernel-source inlining needed). nil if that is not possible, so the caller keeps the
   scalar loop.

   The rewrite is `util/inline-pure-lets`, shared with the GPU emitter and the SOAC lowerer. Two
   things the local copy this replaces got wrong, both silent:

     • it had NO purity gate, resting on the unstated claim that `SIMD value-lambdas are effect-free`
       — an effectful binding was inlined and its effect duplicated per lane;
     • it kept only `(last body)`, DISCARDING the earlier forms of a multi-statement body. Those
       forms are exactly the ones held for their effects, so a store could vanish from the
       vectorized loop while the scalar loop performed it.

   Both are now refusals: nil ⇒ scalar loop, which is correct by construction."
  [expr]
  (try
    (let [r (util/inline-pure-lets expr)]
      ;; a `do` means the body had several forms; a lane expression cannot carry statements
      (when-not (and (seq? r) (= 'do (first r))) r))
    (catch clojure.lang.ExceptionInfo e
      (when-not (= :impure-binding (:reason (ex-data e))) (throw e)))))

;; element keyword → intrinsics vector-facet element key
(defn- vt-of [dtype] (case dtype :double :f64 :float :f32 :long :i32 :int :i32 nil))

(defn active-isa
  "The SIMD ISA facet key for the active hardware — the target-aware knob call sites
   use instead of a literal :avx2. Returns :avx512 only when the descriptor is ≥512-bit
   AND the intrinsics facet actually has that row; else :avx2 (the facet's populated ISA)."
  []
  (if (and (>= (long (:vector-bits (hw/active-descriptor) 256)) 512)
           (get-in in/simd-isa [:avx512 :f32]))
    :avx512 :avx2))

(def ^:dynamic *array-types*
  "Retained array storage element types supplied by AOT binding. Storage width is
   independent of arithmetic precision. Only an entirely empty environment retains
   the legacy direct-emitter interface; partial environments fail closed."
  {})

(defn- storage-load-compatible? [isa elem arr int-widen?]
  (let [storage (get *array-types* arr)]
    (or (empty? *array-types*)
        (= storage (case elem :f64 :double :f32 :float :i32 :int nil))
        (and (= elem :f64) (= storage :float)
             (:from-f32 (in/simd-type-info isa elem))
             (:from-f32-load (in/simd-type-info isa elem)))
        ;; Existing eight-lane i32 widening into Float arithmetic.
        (and int-widen? (= elem :f32) (= storage :int)))))

(defn- storage-loads-compatible? [isa elem expr idx int-widen?]
  (every? #(storage-load-compatible? isa elem (first %) int-widen?)
          (ss/collect-load-sites expr idx)))

(defn- complete-double-operation-evidence? [expression]
  (cond
    ;; Storage and address calculations have their own admission obligations.
    (and (seq? expression) (descriptor/aget-op? (descriptor/semantic-op expression))) true
    (seq? expression)
    (and (or (descriptor/cast-op? (first expression))
             (= :double (dtype/dtype-for-scalar-tag (:raster.type/tag (meta expression)))))
         (every? complete-double-operation-evidence? (rest expression)))
    (coll? expression) (every? complete-double-operation-evidence? expression)
    :else true))

(defn- n-accumulators
  "Independent vector accumulators to hide FMA latency — the register-blocking policy
   from core.hardware/reduction-accumulators (the ONE place it lives, shared with the
   JVM segop_simd reducer), not a hard-coded constant."
  [elem]
  (try (hw/reduction-accumulators (hw/active-descriptor)
                                  (case elem :f64 :double :f32 :float :double))
       (catch Throwable _ 4)))

;; ---------------------------------------------------------------------------
;; Horizontal-reduce helpers (emitted once per compiled unit that uses them).
;; ---------------------------------------------------------------------------
(def simd-includes
  (str "#include <immintrin.h>\n" (in/simd-helper-sources :avx2)))

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
   Returns {:idx :bound :acc :init :elem-expr :factors [f1 f2]|nil}.
   Recognizing a product does not grant contraction of its rounding boundary."
  [segred]
  (let [idx   (ss/seg-idx segred)
        bound (ss/seg-bound segred)
        {:keys [acc init lambda]} (segop/scalar-reduce-op segred)
        lambda (when lambda (ss/normalize-invk (ss/clean-dead-bindings lambda)))]
    (when (and acc init idx bound (seq? lambda) (>= (count lambda) 3))
      (let [op   (first lambda)
            ;; acc must appear as one addend; the other is the lane element.
            elem (cond (= (nth lambda 1) acc) (nth lambda 2)
                       (= (nth lambda 2) acc) (nth lambda 1)
                       :else nil)]
        (when (and (= op '+) elem
                   (ss/retained-floating-precision-compatible? lambda (:dtype segred))
                   (ss/simd-able? elem idx)
                   (empty? (ss/value-position-arrays elem idx)))
          (let [factors (when (and (seq? elem) (= '* (first elem)) (= 3 (count elem))
                                   (ss/aget-form? (nth elem 1) idx)
                                   (ss/aget-form? (nth elem 2) idx))
                          [(nth elem 1) (nth elem 2)])]
            {:idx idx :bound bound :acc acc :init init
             :elem-expr elem :factors factors}))))))

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
    (if (and (= :float (get *array-types* arr)) (:from-f32 ti))
      (str (:from-f32 ti) "(" (:from-f32-load ti) "(&" arrs "[" off "]))")
      (str (:loadu ti) "(&" arrs "[" off "])"))))

(defn- vidx
  "C index expr for an aget at affine base + loop counter jv."
  [base jv array-syms]
  (if base (str "(" (ce/emit-expr base nil array-syms "idx") ") + " jv) jv))

(defn- dom-of
  "Vector domain of an array/scalar element kind."
  [elem-kw]
  (case elem-kw (:int :long :byte) :int (:float :double) :float :float))

(defn emit-c-vexpr
  "Emit a lane-parallel expression as C, coerced to `target-dom` (:float | :int).
   DOMAIN-AWARE (the C analog of segop_simd/emit-simd): an internal `go` returns
   [c-str domain] with domain ∈ {:int :float :poly}. :poly = a number/scalar with
   no fixed domain (broadcast into whatever its use needs). int-typed arrays (per
   *array-types*) load as __m256i and use epi32 ops; a `(float/double <int-expr>)`
   boundary emits _mm256_cvtepi32_ps. `felem` is the float element type (:f32/:f64)
   for float ops. Float storage into Double arithmetic uses a four-element converting
   load, not a Double pointer load. Callers certify supported storage widths as well
   as syntax before emission. Throws on anything unvectorizable."
  [expr idx isa felem jv array-syms target-dom]
  (let [vector-op (fn [op elem]
                    (or (in/simd-op isa op elem)
                        (throw (ex-info "csimd: no vector lowering for operation"
                                        {:op op :isa isa :elem elem}))))
        fti (in/simd-type-info isa felem)
        iti (in/simd-type-info isa :i32)
        to-float (fn [[s d]] (case d
                               :float s
                               :int   (str (:from-i32 fti) "(" s ")")
                               :poly  (str (:set1 fti) "(" s ")")))
        to-int   (fn [[s d]] (case d
                               :int  s
                               :poly (str (:set1 iti) "(" s ")")
                               :float (throw (ex-info "csimd: float→int in vector body unsupported" {:s s}))))
        go (fn go [e]
             (cond
               (number? e) [(str e) :poly]
               (symbol? e) [(ce/c-symbol e) :poly]           ; loop-invariant scalar → poly broadcast
               (ss/aget-form? e idx)
               (let [[arr base] (ss/aget-form? e idx)
                     off (vidx base jv array-syms)]
                 (if (= :int (dom-of (get *array-types* arr :float)))
                   [(str (:loadu iti) "((const __m256i*)&" (ce/c-symbol arr) "[" off "])") :int]
                   [(vec-load fti arr base jv "0" array-syms) :float]))
               ;; (long/int x) → int domain: identity on an int value, broadcast on a poly.
               (and (seq? e) (contains? '#{long int clojure.core/long clojure.core/int} (first e)))
               (let [[s d] (go (second e))]
                 (case d :int [s :int] :poly [s :poly]
                       :float (throw (ex-info "csimd: float→int cast in vector body" {:e e}))))
               ;; (float/double x) → float domain, converting an int subexpr here.
               (and (seq? e) (contains? '#{float double clojure.core/float clojure.core/double} (first e)))
               [(to-float (go (second e))) :float]
               (seq? e)
               (let [op (first e) as (map go (rest e))]
                 (cond
                   (some #(= :float (second %)) as)     ; any float operand → float op, coerce rest
                   [(str (vector-op op felem) "(" (str/join ", " (map to-float as)) ")") :float]
                   (some #(= :int (second %)) as)        ; int op (widening MAC subtract/add)
                   (if (= :int (dtype/dtype-for-scalar-tag (:raster.type/tag (meta e))))
                     (if-let [iop (in/simd-op isa op :i32)]
                       [(str iop "(" (str/join ", " (map to-int as)) ")") :int]
                       (throw (ex-info "csimd: no int vector op" {:op op})))
                     (throw (ex-info "csimd: integer lane arithmetic requires retained i32 evidence"
                                     {:op op :reason :unsupported-integer-lane-width})))
                   :else                                 ; all poly → default float
                   [(str (vector-op op felem) "(" (str/join ", " (map to-float as)) ")") :float]))
               :else (throw (ex-info "csimd/emit-c-vexpr: cannot vectorize" {:expr e}))))]
    (case target-dom
      :int (to-int (go expr))
      (to-float (go expr)))))

(defn- scalar-tail
  "Scalar C for the remainder loop of a SegMap: emit the NORMALIZED lambda (bare
   ops + clojure.core/aget) with idx→jv through the SHARED c_emit expression path
   (which already lowers bare arithmetic / aget / casts), then apply the store
   cast. Delegating to c_emit rather than a bespoke scalar emitter keeps ONE C
   emitter (the earlier `.invk`-literal problem came only from feeding c_emit the
   raw devirtualized form; the normalized form lowers cleanly)."
  [lambda idx jv cast array-syms]
  (let [norm-j (clojure.walk/postwalk-replace {idx (symbol jv)} lambda)
        s (binding [ce/*int-vars* (conj ce/*int-vars* (symbol jv))]
            (ce/emit-expr norm-j nil array-syms "idx"))]
    (if cast (str "(" (name cast) ")(" s ")") s)))

(defn compile-segmap-c
  "SegMap → a C statement block (string) that writes the vectorized element-wise map
   into out-sym, plus a scalar tail. Returns {:includes :block} or nil if not
   vectorizable (caller keeps the scalar loop). Supports pure value-lets, i32→Float
   widening and Float-storage→Double arithmetic loads. Uniform Double arithmetic
   can end in a declared Float store; intervening mixed arithmetic still declines."
  [segmap isa array-syms]
  (let [idx    (ss/seg-idx segmap)
        bound  (ss/seg-bound segmap)
        ;; Canonical TypedSOAC conversions are semantic middle-end terms, not target intrinsics.
        ;; Project them through their retained source spelling at this emitter boundary.  The
        ;; conversion attributes have already been certified, and both the vector expression and
        ;; scalar tail must see the same projection.
        source (some-> (:lambda segmap) ss/clean-dead-bindings)
        out    (:out-sym segmap)
        source-cast (:cast-fn segmap)
        terminal-conversion (conversion/canonical-parts source)
        terminal-double-to-float?
        (= {:source-dtype :double :target-dtype :float
            :rounding :nearest-even :overflow :ieee}
           (select-keys (:attributes terminal-conversion)
                        [:source-dtype :target-dtype :rounding :overflow]))
        double-source (if terminal-double-to-float? (:operand terminal-conversion) source)
        storage-elem (vt-of (:dtype segmap))
        same-precision? (ss/retained-floating-precision-compatible? source (:dtype segmap))
        double-info (in/simd-type-info isa :f64)
        ;; The lambda's retained operations determine arithmetic precision; the output
        ;; dtype only determines storage. Do not erase intervening Float roundings.
        double-to-float? (and (not same-precision?) (= :float (:dtype segmap))
                              (or terminal-double-to-float?
                                  (= 'float (descriptor/cast-result-tag source-cast)))
                              (seq *array-types*)
                              (:to-f32 double-info) (:to-f32-store double-info)
                              (ss/retained-floating-precision-compatible? double-source :double))
        ;; Move only a certified terminal narrowing to the store, never an interior cast.
        compute-source (if double-to-float? double-source source)
        raw (some-> compute-source soac-dialect/scalar-converts->source)
        lambda (when raw (inline-lets (ss/normalize-invk raw)))
        cast (if double-to-float? 'float source-cast)
        elem   (cond same-precision? storage-elem double-to-float? :f64)
        ti     (in/simd-type-info isa elem)]
    (when (and idx bound out ti (seq? lambda)
               ;; Compatibility with missing stamps is not proof for changing species.
               (or (not double-to-float?) (complete-double-operation-evidence? lambda))
               ;; Check before projecting conversions, while their declared source precision
               ;; is still present. Syntax admission below remains emitter-specific.
               (storage-loads-compatible? isa elem lambda idx true)
               (or (empty? *array-types*)
                   (= (get *array-types* out)
                      (case storage-elem :f64 :double :f32 :float :i32 :int nil)))
               (ss/simd-able? lambda idx)
               (empty? (ss/value-position-arrays lambda idx)))
      (let [lanes (:lanes ti)
            n-c   (ce/emit-expr bound nil array-syms "idx")
            outc  (ce/c-symbol out)
            vexpr (try (emit-c-vexpr lambda idx isa (if (#{:f32 :f64} elem) elem :f32) "j"
                                     array-syms (if (= elem :i32) :int :float))
                       (catch clojure.lang.ExceptionInfo _ nil))
            ;; scalar tail via the shared c_emit path (idx→"j", cast applied)
            tail (try (scalar-tail lambda idx "j" cast array-syms)
                      (catch Throwable _ nil))]
        (when (and vexpr tail)
          {:includes simd-includes
           :block
           (str "{\n"
                "  const int _n = " n-c ";\n"
                "  int j = 0;\n"
                "  for (; j + " lanes " <= _n; j += " lanes ") {\n"
                "    " (if double-to-float? (:to-f32-store ti) (:storeu ti))
                "(&" outc "[j], "
                (if double-to-float? (str (:to-f32 ti) "(" vexpr ")") vexpr) ");\n"
                "  }\n"
                "  for (; j < _n; j++) " outc "[j] = " tail ";\n"
                "}")})))))

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
           vmul (in/simd-op isa :* elem)]
       (when (and ti vadd (or (nil? factors) vmul)
                  (storage-loads-compatible? isa elem elem-expr idx false))
         (let [vt    (:vtype ti)
               lanes (:lanes ti)
               nacc  (n-accumulators elem)
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
                            ;; The monoid certificate permits changing the addition tree,
                            ;; not fusing the separately rounded element product into it.
                            (str (accs k) " = " vadd "(" (accs k) ", " vmul "("
                                 (vec-load ti a1 b1 "j" (blk k) array-syms) ", "
                                 (vec-load ti a2 b2 "j" (blk k) array-syms) "));"))
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
                             (let [ct (if (= elem :f64) "double" "float")]
                               (str "(" ct ")(" (scalar-aget a1 b1 "j" array-syms) ") * "
                                    "(" ct ")(" (scalar-aget a2 b2 "j" array-syms) ")")))
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
