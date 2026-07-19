(ns raster.compiler.backend.gpu.c-emit
  "Shared C-family expression emitter for GPU kernel backends.

  Emits S-expressions as C/OpenCL/GLSL/HIP expression strings.
  Backend differences (cast syntax, struct constructors, atomics) are
  parameterized via a config map bound to *emit-config*.

  All kernel backends (par_opencl, par_vulkan, par_hip) delegate
  expression/statement emission to this module and only implement
  their own kernel scaffold (declarations, index computation, barriers).

  Usage:
    (binding [*emit-config* opencl-config
              *scalar-type* \"float\"]
      (emit-expr body idx-sym array-syms \"idx\"))"
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.form :as form]
            [raster.compiler.core.types :as types]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.core :as rcore]))

;; ================================================================
;; Backend configuration
;; ================================================================

(def ^:dynamic *emit-config*
  "Backend-specific emission configuration.
  Keys:
    :cast-style       - :c for (float)x, :glsl for float(x)
    :atomic-add-int   - function name for int atomic add
    :atomic-add-float - function name or :cas-helper for CAS loop
    :float-abs        - \"fabs\" or \"abs\"
    :float-max        - \"fmax\" or \"max\"
    :float-min        - \"fmin\" or \"min\"
    :float-suffix?    - append 'f' to float literals"
  {:cast-style      :c
   :atomic-add-int  "atomic_add"
   :atomic-add-float :cas-helper
   :float-abs       "fabs"
   :float-max       "fmax"
   :float-min       "fmin"
   :float-suffix?   true})

(def opencl-config
  {:cast-style      :c
   :atomic-add-int  "atomic_add"
   :atomic-add-float :cas-helper
   :float-abs       "fabs"
   :float-max       "fmax"
   :float-min       "fmin"
   :float-suffix?   true
   ;; Affine-index vectorization emits OpenCL vloadV/vstoreV. HIP/CUDA/GLSL configs
   ;; omit this until their vector load/store spelling is added (they fall back to
   ;; the scalar loop — always correct).
   :vectorize?      true})

(def glsl-config
  {:cast-style      :glsl
   :atomic-add-int  "atomicAdd"
   :atomic-add-float "atomicAdd"
   :float-abs       "abs"
   :float-max       "max"
   :float-min       "min"
   :float-suffix?   true
   :supports-stmt-expr? false       ;; GLSL has no GCC ({ }) statement expressions
   :type-remap {"long" "int" "long long" "int"}  ;; GLSL has no 64-bit int
   :atan2-name "atan"})

(def hip-config
  ;; CUDA/HIP device code is C++: fabs/fmax/fmin have BOTH float and double overloads, so the
  ;; unsuffixed names are correct for either dtype. The -f-suffixed forms (fabsf/…) would compile
  ;; on both vendors for a DOUBLE kernel via implicit narrowing and silently discard ~29 mantissa
  ;; bits — a "compiles on both, computes differently from OpenCL" miscompile. Match opencl-config.
  {:cast-style      :c
   :atomic-add-int  "atomicAdd"
   :atomic-add-float "atomicAdd"
   :float-abs       "fabs"
   :float-max       "fmax"
   :float-min       "fmin"
   :float-suffix?   true})

;; ================================================================
;; Dynamic state
;; ================================================================

(def ^:dynamic *scalar-type* "double")

(def ^:dynamic *loop-result-var*
  "When a loop is emitted as a statement-expression value (reduction), the C
  name of the properly-typed result variable its terminal value is assigned to.
  nil in statement context, where the loop's value is discarded and the terminal
  falls back to the first loop var (legacy behavior)."
  nil)

(def ^:dynamic *idx-sym*
  "The par loop index symbol (always uint/int). Used by infer-c-type."
  nil)

(def ^:dynamic *int-vars*
  "Set of in-scope symbols known to be integer-typed (loop induction/counter
  variables and int-inferred let bindings). infer-c-type returns \"int\" for these,
  so index arithmetic bound to a name (e.g. `base = b*32`) is typed int rather than
  defaulting to *scalar-type*. Loop and let* emitters seed it for their bodies."
  #{})

(def ^:dynamic *vec-width*
  "When non-nil (2 or 4), the affine-index vectorizer is active and *scalar-type*
  refers to the ELEMENT type of the V-wide vectors being emitted. Only set inside
  the vectorized fast-path of an elementwise kernel loop (see the affine
  vectorization section). Drives the ::vload / vector-cast emission in emit-expr."
  nil)

(defn- contains-vload?
  "True if the vectorizer's ::vload marker occurs anywhere in expr (i.e. the expr is
  vector-valued). Used to drop element-type casts the walker inserted around what is
  now a vector — `(double)(double4)` is illegal C but semantically identity here."
  [expr]
  (let [found (volatile! false)]
    (walk/postwalk (fn [f] (when (and (seq? f) (= ::vload (first f))) (vreset! found true)) f) expr)
    @found))

;; ================================================================
;; Op mapping (valid across all C-family backends)
;; ================================================================

(def op-map
  "Maps Clojure/Java op symbols to C equivalents.
  Backend-specific overrides (fabs vs abs) are handled via *emit-config*.
  java.lang.Math/X aliases are auto-added (see below) so inliner-qualified Java
  method symbols resolve identically to the short Math/X reader forms."
  (let [base
        {'Math/sin   "sin"
         'Math/cos   "cos"
         'Math/tan   "tan"
         'Math/exp   "exp"
         'Math/log   "log"
         'Math/sqrt  "sqrt"
         'Math/abs   "fabs"    ;; overridden per-backend at emit time
         'Math/pow   "pow"
         'Math/max   "fmax"    ;; overridden per-backend at emit time
         'Math/min   "fmin"    ;; overridden per-backend at emit time
         'Math/floor "floor"
         'Math/ceil  "ceil"
         'Math/round "round"
         'Math/atan2 "atan2"
         'Math/fma   "fma"
         '+          "+"
         '-          "-"
         '*          "*"
         '/          "/"
         'clojure.core/+  "+"
         'clojure.core/-  "-"
         'clojure.core/*  "*"
         'clojure.core//  "/"
         'clojure.core/quot "/"
         'clojure.core/rem  "%"
         'quot "/"
         'rem  "%"
         'clojure.core/unchecked-add "+"
         'clojure.core/unchecked-multiply "*"
         'clojure.core/unchecked-subtract "-"
         'clojure.core/unchecked-inc "+"
         'clojure.core/unchecked-dec "-"
         'clojure.core/< "<"
         'clojure.core/> ">"
         'clojure.core/<= "<="
         'clojure.core/>= ">="
         'clojure.core/== "=="
         'unchecked-add-int "+"
         'unchecked-subtract-int "-"
         'unchecked-multiply-int "*"
         'clojure.core/unchecked-add-int "+"
         'clojure.core/unchecked-subtract-int "-"
         'clojure.core/unchecked-multiply-int "*"
         'bit-and "&"
         'clojure.core/bit-and "&"
         'bit-or "|"
         'clojure.core/bit-or "|"
         'bit-xor "^"
         'clojure.core/bit-xor "^"
         'bit-shift-left "<<"
         'clojure.core/bit-shift-left "<<"
         'bit-shift-right ">>"
         'clojure.core/bit-shift-right ">>"
         'unsigned-bit-shift-right ">>"
         'clojure.core/unsigned-bit-shift-right ">>"
         '< "<"
         '> ">"
         '<= "<="
         '>= ">="
         '== "=="
         '!= "!="
   ;; GPU math builtins (GLSL/OpenCL/HIP)
         'fract "fract"
         'clamp "clamp"
         'mix   "mix"
         'step  "step"
         'smoothstep "smoothstep"
         'abs   "abs"
         'sign  "sign"
         'mod   "mod"
         'length "length"
         'distance "distance"
         'dot   "dot"
         'cross "cross"
         'normalize "normalize"
         'min   "min"
         'max   "max"
         'sqrt  "sqrt"
         'sin   "sin"
         'cos   "cos"
         'tan   "tan"
         'exp   "exp"
         'log   "log"
         'pow   "pow"
         'floor "floor"
         'ceil  "ceil"
         'round "round"
         'fma   "fma"
   ;; int8 4-way dot-accumulate → portable OpenCL/C helper (pattern-matched to
   ;; a hardware dp4a). Spelling sourced from the intrinsics registry (:dp4a
   ;; canonical entry) — CUDA/HIP spellings will be sibling rows there, not here.
         'dp4a            (get-in intrinsics/table [:dp4a :c :fn])
         'par/dp4a        (get-in intrinsics/table [:dp4a :c :fn])
         'raster.par/dp4a (get-in intrinsics/table [:dp4a :c :fn])}]
    ;; auto-alias java.lang.Math/X -> the short Math/X entry (the inliner emits
    ;; fully-qualified Java method symbols; both must lower the same).
    (merge base
           (into {} (for [[k v] base
                          :when (and (qualified-symbol? k) (= "Math" (namespace k)))]
                      [(symbol "java.lang.Math" (name k)) v])))))

;; A2 drift guard: op-map is a PARALLEL table to backend.intrinsics (the full
;; fold — per-dialect :c facets — is the D4 end state, tracked in
;; .internal/emitter_review/). Until then: for the unambiguous infix/cmp/bitwise
;; subset, the two tables must agree at load time. Fn-style ops (fabs vs abs,
;; fmax vs max) are per-backend override territory and excluded.
(let [checkable (fn [k] (and (string? (get op-map k))
                             (re-matches #"[+\-*/%<>=!&|^]+" (get op-map k))))]
  (doseq [k (keys op-map)
          :when (checkable k)
          :let [canon (intrinsics/canonical k)
                ix-c  (when canon (get-in intrinsics/table [canon :c]))]
          :when (string? ix-c)]
    (when (not= ix-c (get op-map k))
      (throw (ex-info (str "c_emit op-map drifted from intrinsics for " k
                           ": op-map=" (get op-map k) " intrinsics=" ix-c)
                      {:op k})))))

;; ================================================================
;; Devirtualized arithmetic → C operator/function mapping
;; ================================================================

(defn- mangled-name->c-op
  "C/OpenCL/GLSL lowering for a mangled devirtualized impl name (e.g.
  '_plus__m_double_double' or 'sqrt__m_double'), via the shared intrinsics
  registry. Returns {:kind :infix/:fn/:floored-mod :op \"...\"} or nil.
  GLSL fn-name overrides (abs vs fabs, min vs max) keyed off the emit config."
  [sym-name]
  (intrinsics/c-lowering sym-name (= :glsl (:cast-style *emit-config*))))

;; ================================================================
;; Type mappings
;; ================================================================

(def type-map
  "Dtype keyword → C type. Derived from the single faceted dtype/native-types."
  (dtype/backend-types :c))

(defn scalar-native-type
  "Native type for a kernel SCALAR param, for a kernel whose element type is
   `ctype`. A DECLARED type wins authoritatively — float/double/half → `ctype`
   (the kernel element type), int/long/byte → \"int\"; ONLY when the declared
   type is unknown does the name-regex / :tag heuristic apply. Loud-over-silent:
   a declared float named `step-size`/`max-len` must NOT be truncated to int by
   the counter-name regex. Single source for the OpenCL/GLSL backends (was a
   7-way-duplicated cond where only an explicit :int short-circuited)."
  [sym scalar-types ctype]
  (let [sname (name sym)
        explicit (get scalar-types sym (get scalar-types (symbol sname)))
        m (meta sym)
        ;; Compiler-canonical tag first: :raster.type/tag is stamped for EVERY typed
        ;; binding incl. primitives (a `long` local never gets Clojure :tag —
        ;; compute-binding-hint drops bare primitives). Clojure :tag as a fallback
        ;; for non-primitive hints. This is the reliable signal; the name regex is a
        ;; fragile last-resort heuristic for untagged locals (e.g. a scalar whose
        ;; binder the inliner alpha-renamed `nb`→`nb_α_7`, which the regex misses).
        mtag (or (:raster.type/tag m) (:tag m))]
    (cond
      (contains? #{:float :double :half} explicit) ctype
      (contains? #{:int :long :byte} explicit) "int"
      ;; Stamped tag is authoritative — it OUTRANKS the name regex so a float named
      ;; `len`/`count` isn't truncated to int and an int renamed away from an int-ish
      ;; name still declares int.
      (contains? #{'long 'int 'byte} mtag) "int"
      (contains? #{'float 'double 'half} mtag) ctype
      (re-find #"(?i)n[-_]|size|count|len|idx|offset" sname) "int"
      :else ctype)))

(defn fn-style-reduction-op?
  "True if a reduction op emits as a C function call `f(a,b)` (fmax/fmin) rather
   than an infix operator `(a op b)`. Single source for the GPU reduction
   combine-shape classification (was duplicated across the OpenCL backends)."
  [op]
  (contains? #{'Math/max 'Math/min} op))

(defn combine-fn
  "Return a (fn [a b] -> C string) that combines two operands with reduction
   operator string `c-op`, as `c-op(a, b)` when `fn-style?` else `(a c-op b)`."
  [c-op fn-style?]
  (if fn-style?
    (fn [a b] (str c-op "(" a ", " b ")"))
    (fn [a b] (str "(" a " " c-op " " b ")"))))

(def tag->ctype
  "Map walker :tag metadata to C type strings."
  {'ints "int" 'floats "float" 'doubles "double" 'longs "long" 'bytes "int8_t"
   'int "int" 'float "float" 'double "double" 'long "long" 'byte "int8_t"})

(def element-tag->c
  "Map primitive element-tag to C type string."
  {'double "double" 'float "float" 'long "long" 'int "int" 'byte "int8_t"})

;; ================================================================
;; Symbol mangling
;; ================================================================

(def ^:private c-reserved-words
  #{"int" "float" "double" "long" "void" "char" "short" "unsigned" "signed"
    "for" "while" "do" "if" "else" "return" "break" "continue" "switch" "case"
    "struct" "union" "enum" "typedef" "const" "static" "extern" "auto" "register"
    "sizeof" "goto" "volatile"
    ;; OpenCL specific (incl. scalar type keywords that are common variable names —
    ;; `half` is the OpenCL fp16 type, so a deftm binding named `half` collides)
    "kernel" "global" "local" "constant" "private" "restrict"
    "half" "bool" "uchar" "ushort" "uint" "ulong" "size_t" "ptrdiff_t"
    ;; GLSL specific
    "attribute" "uniform" "varying" "layout" "buffer" "shared"
    "in" "out" "inout" "precision" "lowp" "mediump" "highp"
    "vec2" "vec3" "vec4" "mat2" "mat3" "mat4" "sampler2D" "samplerCube"
    "gl_FragColor" "gl_Position" "gl_VertexID" "gl_InstanceID" "gl_GlobalInvocationID"
    "gl_LocalInvocationID" "gl_WorkGroupID" "gl_NumWorkGroups" "gl_WorkGroupSize"
    "barrier" "memoryBarrier" "memoryBarrierBuffer" "memoryBarrierShared"
    "discard" "flat" "smooth" "noperspective"
    ;; Other reserved
    "sample" "patch" "coherent" "readonly" "writeonly"
    "centroid" "invariant" "main" "true" "false"})

(defn c-symbol
  "Convert a Clojure symbol name to a valid C identifier.
  Appends _ suffix for C/GLSL/OpenCL reserved word collisions."
  [sym]
  (let [s (-> (name sym)
              (str/replace "-" "_")
              (str/replace "*" "_star_")
              (str/replace "/" "_slash_")
              (str/replace "?" "_p")
              (str/replace "!" "_b")
              (str/replace "'" "_prime"))]   ; Clojure allows ' in symbols (a', x'); C does not
    (if (contains? c-reserved-words s)
      (str s "_")
      s)))

;; ================================================================
;; Identity value normalization
;; ================================================================

(defn normalize-identity-val
  "Normalize identity values for C source (infinities, NaN)."
  [v]
  (let [s (str v)]
    (cond
      (or (= s "-Infinity") (= s "##-Inf")) "-INFINITY"
      (or (= s "Infinity") (= s "##Inf"))   "INFINITY"
      (= s "NaN")                            "NAN"
      :else s)))

;; ================================================================
;; Type inference
;; ================================================================

;; quot/rem/mod + bitwise ops produce integers (over integer operands). Without these
;; in the structural fallback they default to *scalar-type* (double), which silently
;; widens integer counters/indices and — critically — the Q4 nibble unpack (bit-and/
;; shift) INSIDE the int8-MAC loop, injecting int<->double conversions that block the
;; C compiler from lowering the loop to VNNI/maddubs. (Matches the +/-/* clause below.)
(def ^:private integer-result-ops
  '#{quot rem mod clojure.core/quot clojure.core/rem clojure.core/mod
     bit-and bit-or bit-xor bit-shift-left bit-shift-right unsigned-bit-shift-right
     clojure.core/bit-and clojure.core/bit-or clojure.core/bit-xor
     clojure.core/bit-shift-left clojure.core/bit-shift-right
     clojure.core/unsigned-bit-shift-right})

(def ^:private scalar-default-seen (atom #{}))

(defn- warn-scalar-default!
  "A2 instrumentation: infer-c-type fell through to *scalar-type* — the silent
   default that miscompiles when the expression isn't actually the kernel element
   type. Warn ONCE per distinct head so hot compile paths don't flood stderr.
   Goal: zero warnings across the suite, then this default becomes a throw
   (same fail-loud policy as the wasm emitter's *lenient-types?*)."
  [expr]
  (let [k (if (seq? expr) (first expr) (class expr))]
    (when-not (contains? @scalar-default-seen k)
      (swap! scalar-default-seen conj k)
      (binding [*out* *err*]
        (println (str "WARNING: infer-c-type scalar-type default for head " (pr-str k)
                      " — expr: " (pr-str (if (seq? expr) (take 4 expr) expr))))))))

(defn infer-c-type
  "Infer the C type of an expression. Prefers the walker-stamped :raster.type/tag
  result-type metadata (the single source of truth — no mangled-name parsing); falls
  back to structural inference (casts, aget element type, comparisons) then
  *scalar-type*."
  [expr]
  (or
   ;; Metadata-first: the walker stamps every typed form (incl. devirtualized .invk
   ;; arithmetic) with its result element type. Use it directly instead of parsing
   ;; impl names or re-deriving from operand structure.
   (when (instance? clojure.lang.IObj expr)
     (when-let [t (:raster.type/tag (meta expr))]
       (get tag->ctype t)))
   (cond
     (and (seq? expr) (= 2 (count expr))
          (contains? #{'int 'float 'double 'long} (first expr)))
     (get tag->ctype (first expr) *scalar-type*)

     (and (seq? expr) (descriptor/aget-op? (first expr))
          (>= (count expr) 3))
     (let [arr (second expr)
           tag (when (symbol? arr) (or (:raster.type/tag (meta arr)) (:tag (meta arr))))]
       (get tag->ctype tag *scalar-type*))

     (and (seq? expr)
          (contains? #{'unchecked-add-int 'unchecked-subtract-int
                       'unchecked-multiply-int} (first expr)))
     "int"

     (and (seq? expr)
          (contains? #{'raster.par/atomic-add! 'par/atomic-add!} (first expr)))
     (let [arr (second expr)
           tag (when (symbol? arr) (or (:raster.type/tag (meta arr)) (:tag (meta arr))))]
       (get tag->ctype tag "int"))

     (and (seq? expr) (contains? #{'inc 'dec 'clojure.core/inc 'clojure.core/dec} (first expr)))
     (let [inner-type (infer-c-type (second expr))]
       (if (= inner-type "int") "int" *scalar-type*))

    ;; Comparison operators return bool
     (and (seq? expr)
          (contains? #{'< '> '<= '>= '== '!=
                       'clojure.core/< 'clojure.core/> 'clojure.core/<=
                       'clojure.core/>= 'clojure.core/==
                       'and 'or 'clojure.core/and 'clojure.core/or} (first expr)))
     "bool"

    ;; Par loop index variable is always uint/int
     (and (symbol? expr) *idx-sym* (= (name expr) (name *idx-sym*)))
     "uint"

    ;; In-scope integer variable (loop counter / int-inferred binding)
     (and (symbol? expr) (contains? *int-vars* expr))
     "int"

     (integer? expr) "int"
     (float? expr) *scalar-type*

    ;; Integer index/counter arithmetic: +/-/* over integer operands stays integer.
    ;; long is integer too (the walker casts index operands to long, e.g.
    ;; (* (long b) (long 32))) — widen to long if any operand is long.
     (and (seq? expr)
          (let [op (first expr)]
            (or (descriptor/addition-op? op) (descriptor/subtraction-op? op) (descriptor/multiplication-op? op)))
          (let [types (map infer-c-type (rest expr))]
            (every? #(contains? #{"int" "uint" "long"} %) types)))
     (let [types (map infer-c-type (rest expr))]
       (cond (some #(= "long" %) types) "long"
             (some #(= "uint" %) types) "uint"
             :else "int"))

    ;; quot/rem/mod + bitwise over integer operands stay integer (widen to long).
     (and (seq? expr)
          (contains? integer-result-ops (first expr))
          (let [types (map infer-c-type (rest expr))]
            (every? #(contains? #{"int" "uint" "long"} %) types)))
     (let [types (map infer-c-type (rest expr))]
       (if (some #(= "long" %) types) "long" "int"))

     :else (do (warn-scalar-default! expr) *scalar-type*))))

;; ================================================================
;; Side effect detection
;; ================================================================

(defn void-form?
  "True if the form itself is a void (statement-only) operation.
   Checks walker-stamped :raster.type/tag first — if present, the form has
   a return type and is not void. Falls back to op-descriptor registry for
   compiler primitives (aset, collect!) that pass through the walker as
   opaque macros and are handled directly by GPU codegen."
  [expr]
  (and (seq? expr)
       (not (:raster.type/tag (meta expr)))
       (descriptor/void-op? (first expr))))

(defn has-side-effects?
  "Check if an expression tree contains void (side-effect) operations — including
  DEVIRTUALIZED array writes (.invk aset-impl …), recognized via the walker-stamped
  :raster.op/original op (an intermediate's aset materializes to .invk, which the
  surface-op void-form? check would otherwise miss → a side-effecting binding would be
  wrongly inlined and re-run)."
  [expr]
  (cond
    (not (seq? expr)) false
    (void-form? expr) true
    (and (= '.invk (first expr))
         (descriptor/aset-op? (:raster.op/original (meta expr)))) true
    :else (some has-side-effects? (rest expr))))

;; ================================================================
;; Backend-specific helpers
;; ================================================================

(defn- remap-type
  "Remap a C type name according to backend config (e.g. long -> int for GLSL)."
  [ctype]
  (get (:type-remap *emit-config*) ctype ctype))

(defn decl-type
  "C declaration type for a binding init: the TC-stamped :raster.type/tag (read via
  infer-c-type's metadata-first path) when present, else the structural fallback —
  then backend-remapped. The single type-resolution entry point for backends that
  declare locals (e.g. the CPU-C AOT outer-let* bindings); they must NOT hardcode a
  type or re-derive one, only read what the walker stamped."
  [init]
  (remap-type (infer-c-type init)))

(defn- supports-stmt-expr?
  "Whether the current backend supports GCC statement expressions ({ ... })."
  []
  (get *emit-config* :supports-stmt-expr? true))

(defn- emit-cast
  "Emit a type cast expression according to backend config."
  [type-name inner-str]
  (case (:cast-style *emit-config*)
    :glsl (str type-name "(" inner-str ")")
    ;; :c and :cpp both use C-style casts
    (str "(" type-name ")(" inner-str ")")))

(defn- resolve-op
  "Resolve an op symbol to a C function name, applying backend overrides."
  [op]
  (let [base (get op-map op)]
    (case base
      "fabs" (:float-abs *emit-config* "fabs")
      "fmax" (:float-max *emit-config* "fmax")
      "fmin" (:float-min *emit-config* "fmin")
      "atan2" (or (:atan2-name *emit-config*) "atan2")
      (or base (c-symbol op)))))

;; ================================================================
;; Expression & statement emission
;; ================================================================

(declare emit-expr emit-stmt emit-stmts-with-result
         emit-loop-body emit-loop-expr try-inline-deftm
         resolve-gpu-inlinable-var)

(defn emit-stmt
  "Emit an S-expression as a C statement (with trailing semicolon)."
  [expr idx-sym array-syms opencl-idx]
  (cond
    ;; aset -> array write (SoA stores are scalar-replaced upstream by soa-lower)
    (and (seq? expr)
         (descriptor/aset-op? (first expr))
         (>= (count expr) 4))
    (let [[_ arr idx-e val-e] expr
          val-str (emit-expr val-e idx-sym array-syms opencl-idx)
          idx-str (emit-expr idx-e idx-sym array-syms opencl-idx)
          ;; GLSL: cast value to target array type
          arr-ctype (when (and (not (supports-stmt-expr?)) (symbol? arr))
                      (or (get element-tag->c (or (:raster.type/tag (meta arr)) (:tag (meta arr))))
                          *scalar-type*))
          val-str (if arr-ctype
                    (str arr-ctype "(" val-str ")")
                    val-str)]
      (str (c-symbol arr) "[" idx-str "] = " val-str ";"))

    ;; when -> void if
    (and (seq? expr) (= 'when (first expr)))
    (let [[_ cond-expr & body] expr
          cond-str (emit-expr cond-expr idx-sym array-syms opencl-idx)]
      (str "if (" cond-str ") { "
           (str/join " " (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx)) body))
           " }"))

    ;; if (void context)
    (and (seq? expr) (= 'if (first expr)))
    (let [args (rest expr)
          cond-expr (first args)
          then-expr (second args)
          else-expr (when (>= (count args) 3) (nth args 2))
          cond-str (emit-expr cond-expr idx-sym array-syms opencl-idx)]
      (if else-expr
        (str "if (" cond-str ") { "
             (emit-stmt then-expr idx-sym array-syms opencl-idx)
             " } else { "
             (emit-stmt else-expr idx-sym array-syms opencl-idx)
             " }")
        (str "if (" cond-str ") { "
             (emit-stmt then-expr idx-sym array-syms opencl-idx)
             " }")))

    ;; do block -> emit all as statements
    (and (seq? expr) (= 'do (first expr)))
    (str/join " " (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx))
                       (rest expr)))

    ;; let in statement context
    (and (seq? expr) (contains? #{'let 'let*} (first expr)))
    (let [[_ bindings & body] expr
          pairs-vec (vec (partition 2 bindings))
          count-uses (fn [sym forms]
                       (let [cnt (atom 0)]
                         (walk/postwalk
                          (fn [f] (when (= f sym) (swap! cnt inc)) f)
                          forms)
                         @cnt))
          use-counts (into {} (map-indexed
                               (fn [i [sym _]]
                                 (let [rest-forms (concat
                                                   (mapcat (fn [[_ v]] [v])
                                                           (subvec pairs-vec (inc i)))
                                                   body)]
                                   [sym (count-uses sym rest-forms)]))
                               pairs-vec))
          multi-use? (fn [sym] (> (get use-counts sym 0) 1))
          {:keys [env locals loop-stmts ints]}
          (reduce
           (fn [{:keys [env locals loop-stmts seen-names ints]} [sym val]]
             ;; Type each binding under the integer vars accumulated so far, so a
             ;; dependent index binding (e.g. sbb = o*nsb after nsb = in/256) infers
             ;; integer rather than defaulting to *scalar-type* (float) — which would
             ;; emit a float array subscript and fail to compile.
             (binding [*int-vars* (into *int-vars* ints)]
               (let [val-subst (walk/postwalk
                                (fn [f] (if (and (symbol? f) (contains? env f))
                                          (get env f) f))
                                val)]
                  ;; Detect loop RHS in GLSL (needs statement hoisting)
                 (if (and (not (supports-stmt-expr?))
                          (seq? val-subst) (contains? #{'loop 'loop*} (first val-subst)))
                    ;; Hoist loop: emit as statements, alias sym → first loop var
                   (let [loop-var (first (take-nth 2 (second val-subst)))
                         loop-c-name (c-symbol loop-var)
                         loop-code (emit-expr val-subst idx-sym array-syms opencl-idx)]
                     {:env (assoc env sym (symbol loop-c-name))
                      :locals locals
                      :loop-stmts (conj (or loop-stmts []) loop-code)
                      :seen-names (or seen-names {})
                      :ints ints})
                   ;; Force-declare loop-valued bindings even when single-use: inlining a
                   ;; reduction into its use site recomputes it per use AND duplicates the
                   ;; loop's induction var into the surrounding scope (a shadowing bug the
                   ;; OpenCL compiler miscompiles). A reduction must be computed once.
                   (if (or (multi-use? sym)
                           (and (seq? val-subst) (contains? #{'loop 'loop*} (first val-subst))))
                     (let [base-name (c-symbol sym)
                           prev-count (get seen-names base-name 0)
                           c-name (if (> prev-count 0) (str base-name "_" prev-count) base-name)
                           c-expr (emit-expr val-subst idx-sym array-syms opencl-idx)
                           c-type (infer-c-type val-subst)]
                       {:env (assoc env sym (symbol c-name))
                        :locals (conj locals [c-name c-expr c-type])
                        :loop-stmts loop-stmts
                        :seen-names (assoc seen-names base-name (inc prev-count))
                        :ints (if (contains? #{"int" "uint" "long"} c-type)
                                (conj ints (symbol c-name)) ints)})
                     {:env (assoc env sym val-subst)
                      :locals locals
                      :loop-stmts loop-stmts
                      :seen-names (or seen-names {})
                      :ints ints})))))
           {:env {} :locals [] :loop-stmts [] :seen-names {} :ints #{}}
           pairs-vec)
          body-subst (map (fn [form]
                            (walk/postwalk
                             (fn [f] (if (and (symbol? f) (contains? env f))
                                       (get env f) f))
                             form))
                          body)
          loop-prefix (str/join " " loop-stmts)
          local-decls (str/join " "
                                (map (fn [[n e t]] (str (remap-type t) " " n " = " e ";")) locals))
          ;; Emit the body under the int vars this scope's bindings introduced, so
          ;; references to int-typed index locals type as integer (not float).
          body-stmts (binding [*int-vars* (into *int-vars* ints)]
                       (str/join " "
                                 (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx))
                                      body-subst)))]
      (str "{ " (when (seq loop-prefix) (str loop-prefix " "))
           (when (seq locals) (str local-decls " "))
           body-stmts " }"))

    ;; loop in statement context -> while loop (no ({ }) wrapper)
    (and (seq? expr) (contains? #{'loop 'loop*} (first expr)))
    (let [[_ bindings & body] expr
          pairs (partition 2 bindings)
          var-names (map first pairs)
          var-inits (map second pairs)
          var-types (map (fn [[_ init]] (remap-type (infer-c-type init))) pairs)
          decls (str/join " "
                          (map (fn [sym typ init]
                                 (str typ " " (c-symbol sym) " = "
                                      (emit-expr init idx-sym array-syms opencl-idx) ";"))
                               var-names var-types var-inits))
          int-loop-vars (set (keep (fn [[sym typ]]
                                     (when (contains? #{"int" "uint" "long"} typ) sym))
                                   (map vector var-names var-types)))
          loop-body (binding [*int-vars* (into *int-vars* int-loop-vars)]
                      (emit-loop-body body var-names var-types idx-sym array-syms opencl-idx))]
      (str "{ " decls " while (true) { " loop-body " } }"))

    ;; atomic-add! as statement
    (and (seq? expr) (contains? #{'raster.par/atomic-add! 'par/atomic-add!} (first expr)))
    (str (emit-expr expr idx-sym array-syms opencl-idx) ";")

    ;; par/collect! -> atomic slot claim + writes
    (and (seq? expr) (contains? #{'raster.par/collect! 'par/collect!} (first expr)))
    (let [[_ count-arr & pairs] expr
          slot-var (str "slot_" (c-symbol count-arr))
          atomic-fn (:atomic-add-int *emit-config* "atomic_add")]
      (str "{ int " slot-var " = " atomic-fn "(&" (c-symbol count-arr) "[0], 1); "
           (str/join " "
                     (map (fn [[arr val]]
                            (str (c-symbol arr) "[" slot-var "] = "
                                 (emit-expr val idx-sym array-syms opencl-idx) ";"))
                          (partition 2 pairs)))
           " }"))

    ;; case in void context -> switch
    (and (seq? expr) (= 'case (first expr)))
    (let [[_ test-expr & clauses] expr
          test-str (emit-expr test-expr idx-sym array-syms opencl-idx)
          pairs (partition 2 clauses)
          default (when (odd? (count clauses)) (last clauses))]
      (str "switch ((int)(" test-str ")) { "
           (str/join " "
                     (map (fn [[val body]]
                            (str "case " val ": { "
                                 (emit-stmt body idx-sym array-syms opencl-idx)
                                 " break; }"))
                          pairs))
           (when default
             (str " default: { "
                  (emit-stmt default idx-sym array-syms opencl-idx)
                  " break; }"))
           " }"))

    ;; Default: expression statement
    :else
    (str (emit-expr expr idx-sym array-syms opencl-idx) ";")))

(defn emit-stmts-with-result
  "Emit a body that may contain side effects, assigning final value to result-var."
  [expr idx-sym array-syms opencl-idx result-var]
  (cond
    (and (seq? expr) (= 'do (first expr)))
    (let [stmts (rest expr)
          leading (butlast stmts)
          last-e (last stmts)]
      (str (str/join " " (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx)) leading))
           " " result-var " = " (emit-expr last-e idx-sym array-syms opencl-idx) ";"))

    (and (seq? expr) (contains? #{'let 'let*} (first expr)))
    (str result-var " = " (emit-expr expr idx-sym array-syms opencl-idx) ";")

    :else
    (str result-var " = " (emit-expr expr idx-sym array-syms opencl-idx) ";")))

(defn- loop-terminal
  "Terminal (non-recur) value expression of a loop BODY (a seq of body forms) —
  the value the loop evaluates to when it exits, via form/terminal-value-expr
  over the implicit do. A VOID terminal (a body ending in a statement-only op
  like aset) yields nil, so callers fall back to *scalar-type*; void-form? is a
  GPU-emission concept, so that check lives here, not in ir/form."
  [body]
  (let [t (form/terminal-value-expr (if (= 1 (count body)) (first body) (cons 'do body)))]
    (when-not (void-form? t) t)))

(defn- emit-loop-body
  "Emit the body of a loop as C while-body statements. var-types parallels var-names — the
  DECLARED C type of each loop variable, used to type recur temps (a recur value feeds back
  into its loop var, so the temp must match the var's type — re-inferring it can produce a
  float type for an int counter, which breaks the OpenCL loop vectorizer)."
  [body var-names var-types idx-sym array-syms opencl-idx]
  (let [body-expr (if (= 1 (count body)) (first body) (cons 'do body))]
    (emit-loop-expr body-expr var-names var-types idx-sym array-syms opencl-idx)))

(defn- emit-loop-expr
  "Recursively emit loop body, translating recur to assignments+continue."
  [expr var-names var-types idx-sym array-syms opencl-idx]
  (cond
    ;; recur -> parallel assignment via temps, then assign back + continue
    (and (seq? expr) (= 'recur (first expr)))
    (let [new-vals (rest expr)
          indexed (map-indexed vector (map vector var-names new-vals))
          ;; Emit temp declarations: type _t0 = expr_a; type _t1 = expr_b; ...
          ;; Temp type = the loop var's DECLARED type (var-types), not infer-c-type of the
          ;; recur value — the temp is assigned back into the loop var, so they must match.
          temps (str/join " "
                          (map (fn [[i [sym val]]]
                                 (let [typ (or (nth var-types i nil) (remap-type (infer-c-type val)))
                                       val-str (emit-expr val idx-sym array-syms opencl-idx)]
                                   (str typ " _t" i " = " val-str ";")))
                               indexed))
          ;; Assign temps to loop vars: a = _t0; b = _t1; ...
          assigns (str/join " "
                            (map (fn [[i [sym _val]]]
                                   (str (c-symbol sym) " = _t" i ";"))
                                 indexed))]
      (str temps " " assigns " continue;"))

    ;; if in loop body
    (and (seq? expr) (= 'if (first expr)))
    (let [args (rest expr)
          cond-expr (first args)
          then-expr (second args)
          else-expr (when (>= (count args) 3) (nth args 2))
          cond-str (emit-expr cond-expr idx-sym array-syms opencl-idx)]
      (if else-expr
        (str "if (" cond-str ") { "
             (emit-loop-expr then-expr var-names var-types idx-sym array-syms opencl-idx)
             " } else { "
             (emit-loop-expr else-expr var-names var-types idx-sym array-syms opencl-idx)
             " }")
        (str "if (" cond-str ") { "
             (emit-loop-expr then-expr var-names var-types idx-sym array-syms opencl-idx)
             " } else { break; }")))

    ;; do block
    (and (seq? expr) (= 'do (first expr)))
    (let [stmts (rest expr)
          leading (butlast stmts)
          last-e (last stmts)]
      (str (str/join " " (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx)) leading))
           " " (emit-loop-expr last-e var-names var-types idx-sym array-syms opencl-idx)))

    ;; let in loop body
    (and (seq? expr) (contains? #{'let 'let*} (first expr)))
    (let [[_ bindings & body] expr
          pairs (partition 2 bindings)
          ;; Thread integer-typed bindings into *int-vars* so a dependent index
          ;; binding (e.g. wj = wsb + j*4) infers integer rather than float.
          {:keys [decl-strs ints]}
          (reduce
           (fn [{:keys [decl-strs seen ints]} [sym val]]
             (binding [*int-vars* (into *int-vars* ints)]
               (let [base (c-symbol sym)
                     prev (get seen base 0)
                     c-name (if (> prev 0) (str base "_" prev) base)
                     c-type (infer-c-type val)]
                 {:decl-strs (conj decl-strs
                                   (str (remap-type c-type) " " c-name " = "
                                        (emit-expr val idx-sym array-syms opencl-idx) ";"))
                  :seen (assoc seen base (inc prev))
                  :ints (if (contains? #{"int" "uint" "long"} c-type)
                          (conj ints sym) ints)})))
           {:decl-strs [] :seen {} :ints #{}}
           pairs)
          decls (str/join " " decl-strs)
          body-strs (binding [*int-vars* (into *int-vars* ints)]
                      (emit-loop-body body var-names var-types idx-sym array-syms opencl-idx))]
      (str decls " " body-strs))

    ;; when in loop
    (and (seq? expr) (= 'when (first expr)))
    (let [[_ cond-expr & body] expr
          cond-str (emit-expr cond-expr idx-sym array-syms opencl-idx)]
      (str "if (" cond-str ") { "
           (str/join " " (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx)) (butlast body)))
           " " (emit-loop-expr (last body) var-names var-types idx-sym array-syms opencl-idx)
           " } else { break; }"))

    ;; Void terminal (aset, collect!, atomic-add!) -> emit as stmt then break
    (void-form? expr)
    (str (emit-stmt expr idx-sym array-syms opencl-idx) " break;")

    ;; Terminal expression -> assign to the typed result var (reduction value)
    ;; or, in statement context, the first loop var (value discarded).
    :else
    (str (or *loop-result-var* (c-symbol (first var-names))) " = "
         (emit-expr expr idx-sym array-syms opencl-idx) "; break;")))

(defn emit-expr
  "Convert an S-expression body to a C-family expression string.
  idx-sym: the iteration variable from the walker
  array-syms: set of symbols that are arrays (accessed via [idx])
  opencl-idx: the C variable name to use for indexing (default \"idx\")"
  ([expr idx-sym array-syms] (emit-expr expr idx-sym array-syms "idx"))
  ([expr idx-sym array-syms opencl-idx]
   (cond
     (nil? expr) "0"

     (number? expr)
     (let [s (str expr)]
       (if (and (= *scalar-type* "float")
                (:float-suffix? *emit-config* true)
                (or (instance? Double expr) (instance? Float expr))
                (not (str/ends-with? s "f")))
         (str s "f")
         s))

     (symbol? expr)
     (let [n (c-symbol expr)]
       (if (contains? array-syms (symbol (name expr)))
         (str n "[" opencl-idx "]")
         (if (= expr idx-sym) opencl-idx n)))

     ;; aget -> array read. Handles surface forms (aget arr idx) AND devirtualized
     ;; interface calls (.invk <impl> arr idx) via the walker-stamped :raster.op/original
     ;; op: typed params stay primitive clojure.core/aget, but INTERMEDIATE arrays are
     ;; devirtualized to .invk by the materialize pass — both must emit the same C array
     ;; subscript. (Reading the carried op, not the mangled impl name — see CLAUDE.md.)
     ;; (SoA reads/field projection are scalar-replaced upstream by soa-lower, so only
     ;; plain per-field arrays reach here.)
     (or (and (seq? expr)
              (descriptor/aget-op? (first expr))
              (>= (count expr) 3))
         (and (seq? expr) (= '.invk (first expr)) (>= (count expr) 4)
              (descriptor/aget-op? (:raster.op/original (meta expr)))))
     (let [invk?    (= '.invk (first expr))
           arr      (if invk? (nth expr 2) (second expr))
           idx-expr (if invk? (nth expr 3) (nth expr 2))]
       (str (c-symbol arr) "["
            (emit-expr idx-expr idx-sym array-syms opencl-idx) "]"))

     ;; Vectorizer marker: (::vload arr off-expr) → a V-wide contiguous vector load
     ;; vloadV(0, arr + off). Emitted only inside the vectorized fast-path of an
     ;; elementwise kernel (see the affine vectorization section), where *vec-width*
     ;; is bound and the runtime divisibility guard guarantees off is V-aligned and
     ;; the block stays within one row (so a rowrel `off = idx % C` load is contiguous).
     (and (seq? expr) (= ::vload (first expr)))
     (let [[_ arr off] expr]
       (str "vload" *vec-width* "(0, " (c-symbol arr) " + "
            (emit-expr off idx-sym array-syms opencl-idx) ")"))

     ;; Primitive cast
     (and (seq? expr)
          (contains? #{'double 'float 'long 'int} (first expr))
          (= 2 (count expr)))
     ;; Vectorizing: a cast the walker inserted around a now-vector expr. An element-type
     ;; cast MATCHING the element type (e.g. `(double x)` where x is a double vload) is a
     ;; no-op → drop it (`(double)(double4)` is illegal C). A cast to a DIFFERENT element
     ;; type over a vector is a genuine per-lane precision change (a `^double` helper in a
     ;; float kernel) with no clean vector lowering → bail this kernel to the scalar loop.
     (if (and *vec-width* (contains-vload? (second expr)))
       (if (= (name (first expr)) *scalar-type*)
         (emit-expr (second expr) idx-sym array-syms opencl-idx)
         (throw (ex-info "vector precision cast" {:raster.compiler.backend.gpu.c-emit/bail true})))
       (emit-cast (remap-type (name (first expr)))
                  (emit-expr (second expr) idx-sym array-syms opencl-idx)))

     ;; let/let* -> local variables with CSE
     (and (seq? expr) (contains? #{'let 'let*} (first expr)))
     (let [[_ bindings & body] expr
           pairs-vec (vec (partition 2 bindings))
           ;; In GLSL mode (no stmt-expr support), always inline all bindings.
           ;; This may duplicate expressions but produces valid GLSL.
           force-inline? (not (supports-stmt-expr?))
           count-uses (fn [sym forms]
                        (let [cnt (atom 0)]
                          (walk/postwalk
                           (fn [f] (when (= f sym) (swap! cnt inc)) f)
                           forms)
                          @cnt))
           use-counts (into {} (map-indexed
                                (fn [i [sym _]]
                                  (let [rest-forms (concat
                                                    (mapcat (fn [[_ v]] [v])
                                                            (subvec pairs-vec (inc i)))
                                                    body)]
                                    [sym (count-uses sym rest-forms)]))
                                pairs-vec))
           multi-use? (fn [sym] (and (not force-inline?)
                                     (> (get use-counts sym 0) 1)))
           {:keys [env locals ints]}
           (reduce
            (fn [{:keys [env locals seen-names ints]} [sym val]]
              ;; Type bindings under the integer vars accumulated so far, so a
              ;; dependent index binding (e.g. k = base*2 after base = b*32) infers
              ;; int rather than defaulting to *scalar-type*.
              (binding [*int-vars* (into *int-vars* ints)]
                (let [val-subst (walk/postwalk
                                 (fn [f] (if (and (symbol? f) (contains? env f))
                                           (get env f) f))
                                 val)]
                  ;; Force-declare when: multi-use; loop-valued (a reduction must be computed
                  ;; once, not inlined+recomputed per use); or SIDE-EFFECTING (e.g. a loop that
                  ;; writes an array, as in softmax's exp-and-sum pass) — inlining a single-use
                  ;; side-effecting binding into a later loop body re-runs its writes every
                  ;; iteration (silent miscompile).
                  (if (or (multi-use? sym)
                          (has-side-effects? val)
                          (and (seq? val-subst) (contains? #{'loop 'loop*} (first val-subst))))
                    (let [base-name (c-symbol sym)
                          prev-count (get seen-names base-name 0)
                          c-name (if (> prev-count 0)
                                   (str base-name "_" prev-count)
                                   base-name)
                          c-expr (emit-expr val-subst idx-sym array-syms opencl-idx)
                          c-type (remap-type (infer-c-type val-subst))]
                      {:env (assoc env sym (symbol c-name))
                       :locals (conj locals [c-name c-expr c-type])
                       :seen-names (assoc seen-names base-name (inc prev-count))
                       :ints (if (contains? #{"int" "uint" "long"} c-type)
                               (conj ints (symbol c-name)) ints)})
                    {:env (assoc env sym val-subst)
                     :locals locals
                     :seen-names (or seen-names {})
                     :ints ints}))))
            {:env {} :locals [] :seen-names {} :ints #{}}
            pairs-vec)
           subst-form (fn [form]
                        (walk/postwalk
                         (fn [f] (if (and (symbol? f) (contains? env f))
                                   (get env f) f))
                         form))
           all-body (mapv subst-form body)
           leading-body (butlast all-body)
           tail-body (last all-body)
           has-leading? (and (seq leading-body)
                             (some has-side-effects? leading-body))]
       ;; Emit leading statements + body under the int vars bound by this scope's
       ;; bindings, so references to int-typed locals (index math) type as int.
       (binding [*int-vars* (into *int-vars* ints)]
         (let [body-str (emit-expr tail-body idx-sym array-syms opencl-idx)]
           (cond
         ;; Multiple body forms with side effects — need statement expression
             (and has-leading? (supports-stmt-expr?))
             (let [leading-stmts (str/join " "
                                           (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx))
                                                leading-body))]
               (str "({ "
                    (str/join " "
                              (map (fn [[n e t]] (str t " " n " = " e ";")) locals))
                    " " leading-stmts
                    " " body-str "; })"))

             (and has-leading? (not (supports-stmt-expr?)))
             (throw (ex-info "GLSL: let* body has multiple forms with side effects but statement expressions are not supported"
                             {:expr expr}))

         ;; Single body or no side effects in leading forms — original path
             (seq locals)
             (str "({ "
                  (str/join " "
                            (map (fn [[n e t]] (str t " " n " = " e ";")) locals))
                  " " body-str "; })")

             :else body-str))))

     ;; do block
     (and (seq? expr) (= 'do (first expr)))
     (let [stmts (rest expr)]
       (if (= 1 (count stmts))
         (emit-expr (first stmts) idx-sym array-syms opencl-idx)
         (if (supports-stmt-expr?)
           (let [leading (butlast stmts)
                 last-expr (last stmts)]
             (str "({ "
                  (str/join " " (map (fn [s]
                                       (emit-stmt s idx-sym array-syms opencl-idx))
                                     leading))
                  " " (emit-expr last-expr idx-sym array-syms opencl-idx) "; })"))
           ;; GLSL: no statement expressions
           (let [leading (butlast stmts)]
             (if (some has-side-effects? leading)
               (throw (ex-info "GLSL: do block has non-tail forms with side effects but statement expressions are not supported"
                               {:expr expr}))
               (emit-expr (last stmts) idx-sym array-syms opencl-idx))))))

     ;; if -> ternary (pure) or compound statement (side-effectful)
     (and (seq? expr) (= 'if (first expr)))
     (let [args (rest expr)
           cond-expr (first args)
           then-expr (second args)
           else-expr (when (>= (count args) 3) (nth args 2))
           cond-str (emit-expr cond-expr idx-sym array-syms opencl-idx)]
       (if (nil? else-expr)
         ;; void if — should only appear in statement context, but handle gracefully
         (if (supports-stmt-expr?)
           (str "({ if (" cond-str ") { "
                (emit-stmt then-expr idx-sym array-syms opencl-idx) " } })")
           ;; GLSL: emit as ternary with 0 fallback
           (str "((" cond-str ") ? ("
                (emit-expr then-expr idx-sym array-syms opencl-idx) ") : 0)"))
         (let [side-effects? (or (has-side-effects? then-expr)
                                 (has-side-effects? else-expr))]
           (if (and side-effects? (supports-stmt-expr?))
             (str "({ " *scalar-type* " _r; if (" cond-str ") { "
                  (emit-stmts-with-result then-expr idx-sym array-syms opencl-idx "_r")
                  " } else { "
                  (emit-stmts-with-result else-expr idx-sym array-syms opencl-idx "_r")
                  " } _r; })")
             (do
               (when (and side-effects? (not (supports-stmt-expr?)))
                 (throw (ex-info "GLSL: if branches contain side effects but statement expressions are not supported"
                                 {:expr expr})))
               ;; Pure ternary
               (str "((" cond-str ") ? ("
                    (emit-expr then-expr idx-sym array-syms opencl-idx)
                    ") : ("
                    (emit-expr else-expr idx-sym array-syms opencl-idx) "))"))))))

     ;; when -> void if (only meaningful in statement context)
     (and (seq? expr) (= 'when (first expr)))
     (let [[_ cond-expr & body] expr
           cond-str (emit-expr cond-expr idx-sym array-syms opencl-idx)]
       (if (supports-stmt-expr?)
         (str "({ if (" cond-str ") { "
              (str/join " " (map (fn [s]
                                   (emit-stmt s idx-sym array-syms opencl-idx))
                                 body))
              " } })")
         ;; GLSL: 'when' in expr context emits 0 (should use stmt context instead)
         (str "/* when */ 0")))

     ;; loop/recur -> while loop
     (and (seq? expr) (contains? #{'loop 'loop*} (first expr)))
     (let [[_ bindings & body] expr
           pairs (partition 2 bindings)
           var-names (map first pairs)
           var-inits (map second pairs)
           var-types (map (fn [[_ init]] (remap-type (infer-c-type init))) pairs)
           decls (str/join " "
                           (map (fn [sym typ init]
                                  (str typ " " (c-symbol sym) " = "
                                       (emit-expr init idx-sym array-syms opencl-idx) ";"))
                                var-names var-types var-inits))
           ;; Integer-typed loop induction vars (counters) seed *int-vars* for the
           ;; loop body, so index math over them (e.g. base = j*4) infers integer
           ;; instead of *scalar-type* (float) — which would emit a float subscript.
           int-loop-vars (set (keep (fn [[sym typ]]
                                      (when (contains? #{"int" "uint" "long"} typ) sym))
                                    (map vector var-names var-types)))
           terminal (loop-terminal body)]
       (binding [*int-vars* (into *int-vars* int-loop-vars)]
         (if (supports-stmt-expr?)
           ;; Reduction result: a dedicated var typed from the loop's terminal
           ;; value, not the first loop var (usually the int counter — assigning a
           ;; double accumulator into it would truncate).
           (let [result-var (str (c-symbol (first var-names)) "_res")
                 result-type (remap-type (if terminal (infer-c-type terminal) *scalar-type*))
                 loop-body (binding [*loop-result-var* result-var]
                             (emit-loop-body body var-names var-types idx-sym array-syms opencl-idx))]
             (str "({ " decls " " result-type " " result-var "; while (1) { " loop-body " } " result-var "; })"))
           ;; GLSL: no statement-expressions; result stays in the first loop var.
           (let [loop-body (emit-loop-body body var-names var-types idx-sym array-syms opencl-idx)]
             (str decls " while (true) { " loop-body " }")))))

     ;; case -> switch
     (and (seq? expr) (= 'case (first expr)))
     (if (supports-stmt-expr?)
       (let [[_ test-expr & clauses] expr
             test-str (emit-expr test-expr idx-sym array-syms opencl-idx)
             pairs (partition 2 clauses)
             default (when (odd? (count clauses)) (last clauses))]
         (str "({ " *scalar-type* " _r; switch ((int)(" test-str ")) { "
              (str/join " "
                        (map (fn [[val body]]
                               (str "case " val ": { _r = "
                                    (emit-expr body idx-sym array-syms opencl-idx)
                                    "; break; }"))
                             pairs))
              (when default
                (str " default: { _r = "
                     (emit-expr default idx-sym array-syms opencl-idx)
                     "; break; }"))
              " } _r; })"))
       (throw (ex-info "GLSL does not support case/switch in expression context."
                       {:expr expr})))

     ;; aset -> array write
     (and (seq? expr)
          (descriptor/aset-op? (first expr))
          (>= (count expr) 4))
     (let [[_ arr idx-e val-e] expr
           val-str (emit-expr val-e idx-sym array-syms opencl-idx)
           idx-str (emit-expr idx-e idx-sym array-syms opencl-idx)
           ;; GLSL: always cast value to target array type (no implicit conversions)
           arr-ctype (when (and (not (supports-stmt-expr?)) (symbol? arr))
                       (or (get element-tag->c (or (:raster.type/tag (meta arr)) (:tag (meta arr))))
                           *scalar-type*))
           val-str (if arr-ctype
                     (str arr-ctype "(" val-str ")")
                     val-str)]
       (str "(" (c-symbol arr) "[" idx-str "] = " val-str ")"))

     ;; raster.par/atomic-add! -> backend-specific atomic
     (and (seq? expr)
          (contains? #{'raster.par/atomic-add! 'par/atomic-add!} (first expr)))
     (let [[_ arr idx-e val-e] expr
           arr-str (c-symbol arr)
           idx-str (emit-expr idx-e idx-sym array-syms opencl-idx)
           val-str (emit-expr val-e idx-sym array-syms opencl-idx)
           tag (when (symbol? arr) (or (:raster.type/tag (meta arr)) (:tag (meta arr))))
           is-float? (contains? #{'floats 'float} tag)
           atomic-fn (if is-float?
                       (let [f (:atomic-add-float *emit-config*)]
                         (if (= f :cas-helper) "atomic_add_float" f))
                       (:atomic-add-int *emit-config* "atomic_add"))]
       (str atomic-fn "(" arr-str " + " idx-str ", " val-str ")"))

     ;; inc/dec
     (and (seq? expr) (= 2 (count expr))
          (contains? #{'inc 'clojure.core/inc} (first expr)))
     (str "((" (emit-expr (second expr) idx-sym array-syms opencl-idx) ") + 1)")

     (and (seq? expr) (= 2 (count expr))
          (contains? #{'dec 'clojure.core/dec} (first expr)))
     (str "((" (emit-expr (second expr) idx-sym array-syms opencl-idx) ") - 1)")

     ;; unchecked-inc / unchecked-dec as (x + 1) / (x - 1), NOT ++/--
     (and (seq? expr) (= 2 (count expr))
          (contains? #{'clojure.core/unchecked-inc 'unchecked-inc} (first expr)))
     (str "((" (emit-expr (second expr) idx-sym array-syms opencl-idx) ") + 1)")

     (and (seq? expr) (= 2 (count expr))
          (contains? #{'clojure.core/unchecked-dec 'unchecked-dec} (first expr)))
     (str "((" (emit-expr (second expr) idx-sym array-syms opencl-idx) ") - 1)")

     ;; unchecked-*-int as binary infix
     (and (seq? expr) (= 3 (count expr))
          (contains? #{'unchecked-add-int 'clojure.core/unchecked-add-int
                       'unchecked-subtract-int 'clojure.core/unchecked-subtract-int
                       'unchecked-multiply-int 'clojure.core/unchecked-multiply-int} (first expr)))
     (let [[op a b] expr]
       (str "(" (emit-expr a idx-sym array-syms opencl-idx)
            " " (get op-map op (name op)) " "
            (emit-expr b idx-sym array-syms opencl-idx) ")"))

     ;; Bitwise ops: bit-and, bit-or, bit-xor, bit-shift-left/right, unsigned-bit-shift-right
     (and (seq? expr) (= 3 (count expr))
          (contains? #{'bit-and 'clojure.core/bit-and
                       'bit-or 'clojure.core/bit-or
                       'bit-xor 'clojure.core/bit-xor
                       'bit-shift-left 'clojure.core/bit-shift-left
                       'bit-shift-right 'clojure.core/bit-shift-right
                       'unsigned-bit-shift-right 'clojure.core/unsigned-bit-shift-right}
                     (first expr)))
     (let [[op a b] expr
           ;; unsigned-bit-shift-right needs unsigned cast for correct semantics
           a-str (emit-expr a idx-sym array-syms opencl-idx)
           a-str (if (contains? #{'unsigned-bit-shift-right
                                  'clojure.core/unsigned-bit-shift-right} op)
                   (str "(unsigned long)(" a-str ")")
                   a-str)]
       (str "(" a-str
            " " (get op-map op) " "
            (emit-expr b idx-sym array-syms opencl-idx) ")"))

     ;; N-ary +/*/bit-or/bit-and
     (and (seq? expr)
          (let [op (first expr)]
            (or (descriptor/addition-op? op) (descriptor/multiplication-op? op)
                (contains? #{'unchecked-add 'clojure.core/unchecked-add
                             'unchecked-multiply 'clojure.core/unchecked-multiply
                             'bit-or 'clojure.core/bit-or
                             'bit-and 'clojure.core/bit-and} op)))
          (> (count expr) 3))
     (let [op-str (get op-map (first expr) (name (first expr)))
           args (rest expr)]
       (str "(" (str/join (str " " op-str " ")
                          (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")"))

     ;; not -> !
     (and (seq? expr) (= 2 (count expr))
          (contains? #{'not 'clojure.core/not} (first expr)))
     (str "(!" (emit-expr (second expr) idx-sym array-syms opencl-idx) ")")

     ;; and/or -> && / ||
     (and (seq? expr) (contains? #{'and 'clojure.core/and 'or 'clojure.core/or} (first expr)))
     (let [op-str (if (contains? #{'and 'clojure.core/and} (first expr)) "&&" "||")
           args (rest expr)]
       (str "(" (str/join (str " " op-str " ")
                          (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")"))

     ;; Binary infix
     (and (seq? expr) (= 3 (count expr))
          (let [op (first expr)]
            (or (descriptor/arithmetic-op? op)
                (contains? #{'< '> '<= '>= '== '!=
                             'clojure.core/< 'clojure.core/> 'clojure.core/<= 'clojure.core/>=
                             'clojure.core/== 'clojure.core/quot 'clojure.core/rem
                             'quot 'rem
                             'unchecked-add 'clojure.core/unchecked-add
                             'unchecked-subtract 'clojure.core/unchecked-subtract
                             'unchecked-multiply 'clojure.core/unchecked-multiply} op))))
     (let [[op a b] expr]
       (str "(" (emit-expr a idx-sym array-syms opencl-idx)
            " " (get op-map op (name op)) " "
            (emit-expr b idx-sym array-syms opencl-idx) ")"))

     ;; deftm helper call -> check for devirtualized arithmetic, else C helper
     (and (seq? expr) (symbol? (first expr))
          (resolve-gpu-inlinable-var (first expr)))
     (let [sym (first expr)
           args (rest expr)
           c-op (mangled-name->c-op (name sym))]
       (if c-op
         ;; Devirtualized arithmetic op -> emit native C operator/function
         (case (:kind c-op)
           :infix (let [es (map #(emit-expr % idx-sym array-syms opencl-idx) args)]
                    (if (= 1 (count es))
                      ;; unary operator (e.g. negation `_minus__m_double`) — prefix
                      ;; the operator; a str/join over one arg would drop it entirely.
                      (str "(" (:op c-op) (first es) ")")
                      (str "(" (str/join (str " " (:op c-op) " ") es) ")")))
           :fn    (str (:op c-op) "("
                       (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")")
           :floored-mod (let [ea (emit-expr (first args) idx-sym array-syms opencl-idx)
                              eb (emit-expr (second args) idx-sym array-syms opencl-idx)]
                          (str "((" ea " % " eb " + " eb ") % " eb ")")))
         ;; Non-arithmetic deftm helper -> C helper invocation. Array-sym args are passed
         ;; as the bare pointer (the helper takes a pointer + offset), not element access.
         (let [c-name (str "gpufn_" (c-symbol sym))
               emit-arg (fn [a] (if (and (symbol? a) (contains? array-syms a))
                                  (c-symbol a)
                                  (emit-expr a idx-sym array-syms opencl-idx)))]
           (str c-name "(" (str/join ", " (map emit-arg args)) ")"))))

     ;; oftype: (.invk oftype-impl ref val) coerces `val` to `ref`'s type (parametric
     ;; materialization inserts it, e.g. to type a reduction seed against a float array). In a
     ;; GPU kernel the value type is the kernel scalar type, so emit a plain cast of the value
     ;; arg — the ref only names the target type. Read the carried op, not the mangled impl name.
     (and (seq? expr) (= '.invk (first expr)) (>= (count expr) 4)
          (= 'raster.numeric/oftype (:raster.op/original (meta expr))))
     (str "(" *scalar-type* ")(" (emit-expr (nth expr 3) idx-sym array-syms opencl-idx) ")")

     ;; .invk typed dispatch on deftm helper -> check for devirtualized arithmetic
     (and (seq? expr) (= '.invk (first expr))
          (>= (count expr) 3)
          (symbol? (second expr))
          (resolve-gpu-inlinable-var (second expr)))
     (let [resolved-var (resolve-gpu-inlinable-var (second expr))
           var-name (str (:name (meta resolved-var)))
           args (nnext expr)
           ;; Metadata-first: the walker stamps the .invk form with its semantic op
           ;; (:raster.op/original). Use it directly; fall back to mangled-name parsing.
           c-op (or (when-let [op (:raster.op/original (meta expr))]
                      (intrinsics/op->c-lowering op (= :glsl (:cast-style *emit-config*))))
                    (mangled-name->c-op var-name))]
       (if c-op
         ;; Devirtualized arithmetic op -> emit native C operator/function
         (case (:kind c-op)
           :infix (let [es (map #(emit-expr % idx-sym array-syms opencl-idx) args)]
                    (if (= 1 (count es))
                      ;; unary operator (e.g. negation `_minus__m_double`) — prefix
                      ;; the operator; a str/join over one arg would drop it entirely.
                      (str "(" (:op c-op) (first es) ")")
                      (str "(" (str/join (str " " (:op c-op) " ") es) ")")))
           :fn    (str (:op c-op) "("
                       (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")")
           :floored-mod (let [ea (emit-expr (first args) idx-sym array-syms opencl-idx)
                              eb (emit-expr (second args) idx-sym array-syms opencl-idx)]
                          (str "((" ea " % " eb " + " eb ") % " eb ")")))
         ;; Non-arithmetic deftm helper -> C helper invocation (array-sym args as bare ptr).
         (let [base-sym (symbol (str (.-ns ^clojure.lang.Var resolved-var))
                                (str (:name (meta resolved-var))))
               c-name (str "gpufn_" (c-symbol base-sym))
               emit-arg (fn [a] (if (and (symbol? a) (contains? array-syms a))
                                  (c-symbol a)
                                  (emit-expr a idx-sym array-syms opencl-idx)))]
           (str c-name "(" (str/join ", " (map emit-arg args)) ")"))))

     ;; deftm inlining: try to inline before generic function fallback
     (and (seq? expr) (symbol? (first expr)))
     (or (try-inline-deftm expr idx-sym array-syms opencl-idx)

         ;; Unary function
         (when (= 2 (count expr))
           (let [[op arg] expr
                 arg-str (emit-expr arg idx-sym array-syms opencl-idx)]
             (if (= "round" (get op-map op))
               ;; Java Math.round / raster.math/round are half-UP, defined as
               ;; floor(x + 0.5). C/OpenCL/WGSL round() round half AWAY FROM ZERO,
               ;; which diverges on negative .5 values (round(-63.5) = -64 vs Java
               ;; -63). Emit floor((x) + 0.5) for cross-backend-consistent half-up.
               (str (resolve-op 'Math/floor) "((" arg-str ") + 0.5)")
               (str (resolve-op op) "(" arg-str ")"))))

         ;; Binary function
         (when (= 3 (count expr))
           (let [[op a b] expr]
             (str (resolve-op op) "(" (emit-expr a idx-sym array-syms opencl-idx)
                  ", " (emit-expr b idx-sym array-syms opencl-idx) ")")))

         ;; Ternary function (fma)
         (when (= 4 (count expr))
           (let [[op a b c] expr]
             (str (resolve-op op) "(" (emit-expr a idx-sym array-syms opencl-idx)
                  ", " (emit-expr b idx-sym array-syms opencl-idx)
                  ", " (emit-expr c idx-sym array-syms opencl-idx) ")")))

         ;; N-ary function fallback
         (let [[op & fargs] expr]
           (str (resolve-op op) "("
                (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) fargs))
                ")")))

     :else
     ;; Loud over silently corrupt: an unhandled IR form pr-str'd into the C/OpenCL
     ;; source produces garbage (a cryptic clang/ocloc error, or a call to a
     ;; nonexistent function). Surfacing this as a WARNING (was fully silent) flagged
     ;; two real backend gaps: `case*` branch-maps (abm/firms, blelloch-scan,
     ;; autotuner) and a SIMD-fold vector `[L 8]` (x8-simd-fold) — both stringified
     ;; into invalid C. TODO: lower case* → cond and handle the SIMD vector, THEN
     ;; flip this to `throw` (the true loud form). Tracked in compiler_consolidation.
     (do (binding [*out* *err*]
           (println (str "WARNING: c-emit unhandled IR form (" (type expr)
                         ") — emitting as text, likely invalid. Lower it upstream. Form: "
                         (pr-str expr))))
         (str expr)))))

;; ================================================================
;; deftm inlining support
;; ================================================================

(defn infer-arg-tag
  "Infer the type tag of an expression in the current body context."
  [expr]
  (cond
    ;; Vectorizer marker: a ::vload carries V element-type lanes — tag it as the element
    ;; type so the deftm inliner accepts it and inlines the helper body with the vector
    ;; flowing through (OpenCL vector arithmetic uses the scalar operators/builtins).
    (and (seq? expr) (= ::vload (first expr)))
    (when *scalar-type*
      (get {"float" 'float "double" 'double "int" 'int "long" 'long} *scalar-type*))

    (symbol? expr)
    (or (:tag (meta expr))
        ;; Fall back to scalar type for untagged symbols (push constants, locals)
        (when *scalar-type*
          (get {"float" 'float "double" 'double "int" 'int "long" 'long}
               *scalar-type*)))

    (and (seq? expr) (descriptor/aget-op? (first expr))
         (>= (count expr) 3))
    (let [arr (second expr)]
      (or (when (symbol? arr)
            (get {'doubles 'double 'floats 'float 'longs 'long 'ints 'int}
                 (or (:raster.type/tag (meta arr)) (:tag (meta arr)))))
          ;; Fall back to scalar type for untagged arrays
          (when *scalar-type*
            (get {"float" 'float "double" 'double "int" 'int "long" 'long}
                 *scalar-type*))))

    (and (seq? expr) (contains? #{'. 'clojure.core/.} (first expr)))
    ;; Field access on a struct — infer scalar type
    (when *scalar-type*
      (get {"float" 'float "double" 'double} *scalar-type*))

    ;; Field access (.x v) form
    (and (seq? expr) (= 2 (count expr))
         (let [n (name (first expr))]
           (.startsWith n ".")))
    (when *scalar-type*
      (get {"float" 'float "double" 'double} *scalar-type*))

    (double? expr) 'double
    (float? expr) 'float
    (int? expr) 'long

    :else nil))

(defn- try-inline-deftm
  "Try to inline a call to a deftm generic function as a C expression.
  Returns nil if inlining is not possible."
  [expr idx-sym array-syms opencl-idx]
  (when (and (seq? expr) (symbol? (first expr)) (>= (count expr) 2))
    (when-let [v (try (resolve (first expr)) (catch Exception _ nil))]
      (let [m (meta v)]
        (when-let [table-atom (get m :raster.core/dispatch-table)]
          (let [args     (rest expr)
                arg-tags (mapv infer-arg-tag args)]
            (when (every? some? arg-tags)
              (let [arity    (count args)
                    methods  (get @table-atom arity)]
                (when methods
                  (when-let [matched-tags
                             (some (fn [entry]
                                     (let [checks (:check-classes entry)]
                                       (when (every? true?
                                                     (map (fn [t c]
                                                            (or (nil? c)
                                                                (if (class? c)
                                                                  (let [cls (or (get {'double Double 'float Float
                                                                                      'long Long 'int Integer} t)
                                                                                (try (resolve t) (catch Exception _ nil)))]
                                                                    (and (class? cls)
                                                                         (or (.isAssignableFrom ^Class c ^Class cls)
                                                                       ;; GPU numeric widening: float→double, int→long
                                                                             (and (= c Double) (= cls Float))
                                                                             (and (= c Long) (= cls Integer)))))
                                                                  true)))
                                                          arg-tags checks))
                                         (:tags entry))))
                                   methods)]
                    (let [fn-simple (symbol (name (:name m)))
                          mangled   (try (types/mangle fn-simple matched-tags)
                                         (catch Exception _ nil))]
                      (when mangled
                        (let [fn-ns     (:ns m)
                              impl-v    (try (ns-resolve fn-ns mangled)
                                             (catch Exception _ nil))]
                          (when impl-v
                            (when-let [walked-body (or (get (meta impl-v) :raster.core/deftm-walked-body)
                                                       (rcore/ensure-walked-body! impl-v))]
                              (let [params (get (meta impl-v) :raster.core/deftm-params)]
                                (when (= (count params) (count args))
                                  (let [subst      (zipmap params (vec args))
                                        body-expr  (if (= 1 (count walked-body))
                                                     (first walked-body)
                                                     (cons 'do walked-body))
                                        subst-body (walk/postwalk-replace subst body-expr)]
                                    (emit-expr subst-body idx-sym array-syms opencl-idx)))))))))))))))))))

;; ================================================================
;; Body analysis utilities
;; ================================================================

(defn normalize-array-prims
  "Rewrite DEVIRTUALIZED array-primitive .invk forms back to their canonical aget/aset/alength
   head, using the walker's :raster.op/original metadata (the semantic identity it stamps —
   CLAUDE.md: 'no pass recovers meaning from mangled names'). A forward-pass devirtualizes the
   polymorphic raster.arrays/aget into `(.invk aget_m_T-impl arr idx)`; the GPU emitter's
   aget-op?/aset-op? matchers (array detection, element typing, emit) key on the HEAD, so without
   this the array args are missed (classified scalar) and aget is mis-emitted as a C helper call
   subscripting a scalar. One normalization here makes every downstream c_emit site recognize
   them — preserves the form's metadata so :raster.type/tag survives for element typing."
  [body]
  (walk/postwalk
   (fn [form]
     (if (and (seq? form) (= '.invk (first form)) (>= (count form) 3))
       (let [op (:raster.op/original (meta form))]
         (if (or (descriptor/aget-op? op) (descriptor/aset-op? op) (descriptor/alength-op? op))
           (with-meta (list* op (nnext form)) (meta form))
           form))
       form))
   body))

(defn collect-arrays-in-body
  "Collect symbols used as arrays in a par body expression."
  [body]
  (let [arrays (atom #{})]
    (walk/postwalk
     (fn [form]
       (when (seq? form)
         (let [head (first form)]
           (when (and (descriptor/aget-op? head)
                      (>= (count form) 3))
             (swap! arrays conj (second form)))
           (when (and (descriptor/aset-op? head)
                      (>= (count form) 4))
             (swap! arrays conj (second form)))
           (when (and (contains? #{'raster.par/atomic-add! 'par/atomic-add!} head)
                      (>= (count form) 4))
             (swap! arrays conj (second form)))
           (when (and (contains? #{'raster.par/collect! 'par/collect!} head)
                      (>= (count form) 2))
             (let [[_ count-arr & pairs] form]
               (swap! arrays conj count-arr)
               (doseq [[arr _] (partition 2 pairs)]
                 (swap! arrays conj arr))))))
       form)
     body)
    @arrays))

(defn collect-let-bound-syms
  "Collect all symbols bound by let/let*/loop forms in body."
  [body]
  (let [bound (atom #{})]
    (walk/postwalk
     (fn [form]
       (when (and (seq? form)
                  ;; loop* (not loop) is what survives macroexpand/materialize into the kernel
                  ;; body — omitting it leaks loop induction/accumulator vars (i, s) into the
                  ;; kernel's scalar params (→ unresolved-symbol when arg-fns are built).
                  (contains? #{'let 'let* 'loop 'loop*} (first form))
                  (>= (count form) 3))
         (let [bindings (second form)]
           (doseq [i (range 0 (count bindings) 2)]
             (swap! bound conj (nth bindings i)))))
       form)
     body)
    @bound))

(defn collect-scalars-in-body
  "Collect symbols used as scalars (not arrays, not the idx) in body."
  [body idx-sym array-syms]
  (let [scalars (atom #{})
        let-bound (collect-let-bound-syms body)
        fn-heads  (let [heads (atom #{})]
                    (walk/postwalk
                     (fn [f]
                       (when (seq? f)
                          ;; Regular function calls: (sym ...)
                         (when (symbol? (first f))
                           (swap! heads conj (first f)))
                          ;; .invk typed dispatch: (.invk impl-var args...)
                          ;; The impl-var is a dispatch target, not a scalar
                         (when (and (= '.invk (first f))
                                    (>= (count f) 2)
                                    (symbol? (second f)))
                           (swap! heads conj (second f))))
                       f)
                     body)
                    @heads)]
    (walk/postwalk
     (fn [form]
       (when (and (symbol? form)
                  (not= form idx-sym)
                  (not (contains? array-syms form))
                  (not (contains? let-bound form))
                  (not (contains? fn-heads form))
                  (not (str/starts-with? (name form) "->"))
                  (not (str/starts-with? (name form) "."))
                  (not (contains? #{'aget 'aset 'alength 'let 'let* 'do
                                    'clojure.core/aget 'clojure.core/aset
                                    'clojure.core/alength
                                    'if 'when 'cond 'case 'case* 'loop 'loop* 'recur
                                    'fn 'fn* 'quote 'new 'throw 'try
                                    'double 'float 'long 'int
                                    'Math/sin 'Math/cos 'Math/exp 'Math/log
                                    'Math/sqrt 'Math/abs 'Math/pow 'Math/max
                                    'Math/min 'Math/fma 'Math/atan2
                                    'Math/floor 'Math/ceil 'Math/round
                                    '+ '- '* '/
                                    'clojure.core/+ 'clojure.core/-
                                    'clojure.core/* 'clojure.core//
                                    'clojure.core/< 'clojure.core/> 'clojure.core/<=
                                    'clojure.core/>= 'clojure.core/==
                                    '< '> '<= '>= '== '!=
                                    'inc 'dec 'clojure.core/inc 'clojure.core/dec
                                    'unchecked-add-int 'unchecked-subtract-int
                                    'unchecked-multiply-int
                                    'clojure.core/unchecked-add-int
                                    'clojure.core/unchecked-subtract-int
                                    'clojure.core/unchecked-multiply-int
                                    'bit-and 'clojure.core/bit-and
                                    'bit-or 'clojure.core/bit-or
                                    'bit-xor 'clojure.core/bit-xor
                                    'bit-shift-left 'clojure.core/bit-shift-left
                                    'bit-shift-right 'clojure.core/bit-shift-right
                                    'unsigned-bit-shift-right
                                    'clojure.core/unsigned-bit-shift-right
                                    'raster.par/atomic-add!
                                    'par/atomic-add!
                                    'raster.par/collect!
                                    'par/collect!
                                    'not 'clojure.core/not
                                    'and 'or 'clojure.core/and 'clojure.core/or
                                     ;; GPU builtins
                                    'fract 'clamp 'mix 'step 'smoothstep
                                    'abs 'sign 'mod 'length 'distance
                                    'dot 'cross 'normalize 'min 'max
                                    'sqrt 'sin 'cos 'tan 'exp 'log
                                    'pow 'floor 'ceil 'round 'fma
                                    'nil} form)))
         (swap! scalars conj form))
       form)
     body)
    @scalars))

(defn adapt-casts-for-dtype
  "Rewrite double/float cast forms in body to match target dtype."
  [body dtype]
  (if (= dtype :double)
    body
    (walk/postwalk
     (fn [form]
       (if (and (seq? form) (= 'double (first form)) (= 2 (count form)))
         (list 'float (second form))
         form))
     body)))

(defn collect-array-types-from-meta
  "Extract array types from walker :tag metadata on symbols in the body."
  [body]
  (let [types (atom {})]
    (walk/postwalk
     (fn [f]
       (when (and (symbol? f) (:tag (meta f)))
         (let [tag (:tag (meta f))]
           (when-let [kw (get {'ints :int 'floats :float 'doubles :double 'longs :long} tag)]
             (swap! types assoc (symbol (name f)) kw))))
       f)
     body)
    @types))

(defn collect-written-arrays
  "Collect array symbols that are written to via aset or atomic-add!."
  [body]
  (let [written (atom #{})]
    (walk/postwalk
     (fn [form]
       (when (seq? form)
         (let [head (first form)]
           (when (and (descriptor/aset-op? head)
                      (>= (count form) 4))
             (swap! written conj (second form)))
           (when (and (contains? #{'raster.par/atomic-add! 'par/atomic-add!} head)
                      (>= (count form) 4))
             (swap! written conj (second form)))
           (when (and (contains? #{'raster.par/collect! 'par/collect!} head)
                      (>= (count form) 2))
             (let [[_ count-arr & pairs] form]
               (swap! written conj count-arr)
               (doseq [[arr _] (partition 2 pairs)]
                 (swap! written conj arr))))))
       form)
     body)
    @written))

;; ================================================================
;; GPU function helpers — auto-inlining of deftm scalar helpers
;; ================================================================

(def tag->ctype-helper
  {'Double "double" 'Float "float" 'Long "long" 'Integer "int"
   'int "int" 'long "long" 'float "float" 'double "double"})

(defn- resolve-gpu-inlinable-var
  "Resolve a symbol to a GPU-inlinable deftm var. Any deftm with source-body
  metadata is inlinable as a static C helper in GPU kernels. Handles both
  direct vars and -impl vars (for .invk typed dispatch patterns)."
  [sym]
  (when-let [v (try (resolve sym) (catch Exception _ nil))]
    (let [m (meta v)]
      (if (:raster.core/deftm-source-body m)
        v
        ;; Try stripping -impl suffix for deftype singleton vars
        (let [sname (name sym)]
          (when (str/ends-with? sname "-impl")
            (let [base-sym (symbol (namespace sym) (subs sname 0 (- (count sname) 5)))]
              (when-let [base-v (try (resolve base-sym) (catch Exception _ nil))]
                (when (:raster.core/deftm-source-body (meta base-v))
                  base-v)))))))))

(defn- try-add-gpu-helper
  "Check if a var is a GPU-inlinable deftm and add it to result if not seen.
  Always stores the resolved base var's qualified name as :sym for consistent c-name generation."
  [sym seen result]
  (when-not (contains? @seen sym)
    (when-let [v (resolve-gpu-inlinable-var sym)]
      (let [m (meta v)
            ;; Use the resolved var's qualified name, not the original sym
            ;; This ensures -impl vars map to the same c-name as direct calls
            base-sym (symbol (str (.-ns ^clojure.lang.Var v)) (str (:name m)))]
        (swap! seen conj sym)
        (swap! result conj
               {:sym base-sym :var v
                :source-body (:raster.core/deftm-source-body m)
                :params (:raster.core/deftm-params m)
                :tags (:raster.core/deftm-tags m)})))))

(defn collect-gpu-fn-calls
  "Scan body for calls to GPU-inlinable deftm functions.
  Recognizes both direct calls (fn-sym args...) and typed dispatch (.invk impl-var args...)."
  [body]
  (let [seen (atom #{})
        result (atom [])]
    (walk/postwalk
     (fn [form]
       (when (seq? form)
          ;; Direct calls: (fn-sym args...)
         (when (symbol? (first form))
           (try-add-gpu-helper (first form) seen result))
          ;; Typed dispatch: (.invk impl-var args...)
         (when (and (= '.invk (first form))
                    (>= (count form) 2)
                    (symbol? (second form)))
           (try-add-gpu-helper (second form) seen result)))
       form)
     body)
    @result))

(defn generate-c-helper
  "Generate a static C helper function from a GPU-inlinable deftm."
  [{:keys [sym var params tags source-body]}]
  (let [c-name (str "gpufn_" (c-symbol sym))
        ;; Return type from var metadata, not from tags (which are param-only dispatch tags)
        ret-tag (or (:raster.core/return-tag (meta var)) (last tags))
        ret-type (get tag->ctype-helper ret-tag "double")
        ;; All tags are param dispatch tags
        param-types (map #(get tag->ctype-helper % "double") tags)
        param-strs (str/join ", "
                             (map (fn [p t] (str t " " (c-symbol p)))
                                  params param-types))
        body-expr (if (= 1 (count source-body))
                    (first source-body)
                    (cons 'do source-body))
        body-str (binding [*scalar-type* ret-type]
                   (emit-expr body-expr nil #{} "idx"))]
    {:c-name c-name
     :source (str "static " ret-type " " c-name "(" param-strs ") {\n"
                  "    return " body-str ";\n"
                  "}\n")}))

;; ================================================================
;; Affine-index vectorization (shared, opt-in-by-provability)
;; ================================================================
;; Rewrites a straight-line elementwise kernel body so each work item processes
;; V consecutive elements via vector loads/stores (vloadV / vstoreV). This is the
;; #1 GPU bandwidth lever for the indexed elementwise family (chunked-rms-norm
;; apply and friends): the currency for these streaming kernels is outstanding
;; bytes per lane, and float4 loads/stores recover ~40% of the bandwidth the
;; scalar path leaves on the table.
;;
;; Subscript classification (relative to the work-item index `idx`):
;;   identity  a[idx]        -> contiguous  vloadV(0, a + idx)
;;   a[idx/C]  (broadcast)   -> CONSTANT across the V lanes of one block -> scalar
;;                             load a[idx/C], broadcast into the vector arithmetic
;;   a[idx%C]  (row-relative)-> contiguous within a row -> vloadV(0, a + (idx%C))
;;   free of idx (invariant) -> scalar broadcast
;;   anything else           -> NOT vectorizable; fall back to the scalar loop.
;;
;; Legality (non-straddling) is enforced at RUNTIME, not guessed: the vectorized
;; fast-path runs only when the element count and EVERY divisor C are multiples of
;; V. Given the loop hands each work item a V-aligned block base (vb*V), C % V == 0
;; guarantees a V-block never straddles a `/C` or `%C` row boundary — so idx/C is
;; constant across the block and idx%C is contiguous. Otherwise the scalar loop
;; runs. The transform is therefore always correct; it only ever CAPTURES the win
;; when the dims cooperate (gemma d=640, hd=256 are multiples of 4).
;;
;; The classification is pure S-expression analysis (the JVM/CPU affine matchers in
;; segop_simd are backend-coupled, so the small pure equivalents live here). Living
;; in c_emit, every C-family backend that sets :vectorize? in its emit-config
;; inherits it; the vloadV/vstoreV spelling is OpenCL-specific and gated on that
;; flag, so HIP/CUDA enable it later by adding their spelling, not by rewriting.

(defn- vec-bail!
  "Abort vectorization of the current kernel (caught at the top level → scalar loop)."
  [] (throw (ex-info "not-vectorizable" {::bail true})))

(def ^:private quot-ops #{'quot 'clojure.core/quot})
(def ^:private rem-ops  #{'rem 'clojure.core/rem 'mod 'clojure.core/mod})
(def ^:private prim-cast-ops #{'double 'float 'long 'int
                               'clojure.core/double 'clojure.core/float
                               'clojure.core/long 'clojure.core/int})
(def ^:private idx-cast-ops #{'long 'int 'clojure.core/long 'clojure.core/int})

(defn- sym-occurs?
  "True if symbol s (matched by name) occurs anywhere in expr."
  [expr s]
  (let [found (atom false)]
    (walk/postwalk (fn [f] (when (and (symbol? f) s (= (name f) (name s)))
                             (reset! found true))
                     f)
                   expr)
    @found))

(defn- strip-idx-cast
  "Unwrap a single (long idx)/(int idx) cast."
  [e]
  (if (and (seq? e) (= 2 (count e)) (contains? idx-cast-ops (first e)))
    (second e) e))

(defn- classify-subscript
  "Classify an aget/aset index expression E relative to idx-sym. Returns one of
  {:kind :contiguous} / {:kind :broadcast :divisor C} / {:kind :rowrel :divisor C}
  / {:kind :invariant} / nil (non-affine → not vectorizable)."
  [e idx-sym]
  (let [e (strip-idx-cast e)
        idx? (fn [x] (let [x (strip-idx-cast x)]
                       (and (symbol? x) (= (name x) (name idx-sym)))))]
    (cond
      (idx? e) {:kind :contiguous}
      (and (seq? e) (= 3 (count e)) (contains? quot-ops (first e))
           (idx? (nth e 1)) (not (sym-occurs? (nth e 2) idx-sym)))
      {:kind :broadcast :divisor (nth e 2)}
      (and (seq? e) (= 3 (count e)) (contains? rem-ops (first e))
           (idx? (nth e 1)) (not (sym-occurs? (nth e 2) idx-sym)))
      {:kind :rowrel :divisor (nth e 2)}
      (not (sym-occurs? e idx-sym)) {:kind :invariant}
      :else nil)))

(defn- inline-pure-lets
  "Deep-inline EVERY pure let/let* (no side effects, not loops) anywhere in the form,
  so subscript index expressions are exposed and no scalar local is declared for a
  value that becomes a vector (a nested let inside a store value would otherwise emit
  `float x = <float4>` — a type error). A vectorizable straight-line body has only
  pure bindings; an impure/loop-valued binding throws vec-bail! (→ scalar loop)."
  [expr]
  (walk/prewalk
   (fn [f]
     (if (and (seq? f) (contains? #{'let 'let*} (first f)))
       (let [[_ bindings & body] f
             pairs (partition 2 bindings)]
         (when (some (fn [[_ v]] (or (has-side-effects? v)
                                     (and (seq? v) (contains? #{'loop 'loop*} (first v)))))
                     pairs)
           (vec-bail!))
         (let [env (reduce (fn [env [s v]]
                             (assoc env s (walk/postwalk-replace env v)))
                           {} pairs)
               body' (mapv #(walk/postwalk-replace env %) body)]
           (if (= 1 (count body')) (first body') (cons 'do body'))))
       f))
   expr))

(defn- aget-form?* [f] (and (seq? f) (>= (count f) 3) (descriptor/aget-op? (first f))))
(defn- aset-form?* [f] (and (seq? f) (>= (count f) 4) (descriptor/aset-op? (first f))))

(defn- collect-stores
  "Return a vector of (aset ...) forms from a straight-line core (an aset or a do
  of asets). nil if the shape is not straight-line stores."
  [core]
  (cond
    (aset-form?* core) [core]
    (and (seq? core) (= 'do (first core)))
    (let [ss (rest core)]
      (when (every? aset-form?* ss) (vec ss)))
    :else nil))

(def ^:private cast-target
  {'float "float" 'double "double" 'int "int" 'long "long"
   'clojure.core/float "float" 'clojure.core/double "double"
   'clojure.core/int "int" 'clojure.core/long "long"})

(def ^:private value-has-vload?
  (fn [expr]
    (let [found (atom false)]
      (walk/postwalk (fn [f] (when (and (seq? f) (= ::vload (first f))) (reset! found true)) f) expr)
      @found)))

(defn- preprocess-value
  "Rewrite a value expression into its vectorized form: contiguous/row-relative loads
  become ::vload markers (emit-expr renders vloadV); broadcast/invariant loads and bare
  scalars stay scalar (OpenCL broadcasts them into the vector arithmetic). Element-type
  casts (float/double) wrapping a vector are no-ops and are dropped — a `(float)(float4)`
  is illegal C but semantically identity here. A deftm helper call keeps its shape; a
  ::vload arg makes emit-expr's inliner (via infer-arg-tag) inline the helper body with
  the vector flowing through — so no scalar-per-lane helper call survives. Collects the
  divisor of every broadcast/rowrel subscript into `divisors!` (an atom). Throws vec-bail!
  on anything that cannot be proven vector-safe (non-affine subscript, gather,
  integer-narrowing over a vector, non-integer divisor)."
  [expr idx-sym array-syms divisors!]
  (letfn [(record-div! [d]
            (when-not (contains? #{"int" "uint" "long"} (infer-c-type d)) (vec-bail!))
            (swap! divisors! conj d))
          (go [f]
              (cond
              ;; explicit aget → classify its index
                (aget-form?* f)
                (let [arr (nth f 1) idx-e (nth f 2)
                      c (classify-subscript idx-e idx-sym)]
                  (case (:kind c)
                    :contiguous (list ::vload arr idx-sym)
                    :rowrel     (do (record-div! (:divisor c)) (list ::vload arr idx-e))
                    :broadcast  (do (record-div! (:divisor c)) f) ;; scalar broadcast load
                    :invariant  f                                 ;; scalar, free of idx
                    (vec-bail!)))                                 ;; nil → non-affine/gather
              ;; primitive cast: element-type cast over a vector is a no-op → drop it
                (and (seq? f) (= 2 (count f)) (contains? prim-cast-ops (first f)))
                (let [inner (go (second f))
                      target (get cast-target (first f))]
                  (if (value-has-vload? inner)
                    (if (= target *scalar-type*) inner (vec-bail!))
                    (list (first f) inner)))
              ;; bare array symbol used as a value → contiguous load
                (and (symbol? f) (contains? array-syms (symbol (name f))))
                (list ::vload f idx-sym)
                (seq? f)    (doall (map go f))
                (vector? f) (mapv go f)
                :else f))]
    (go expr)))

(defn- idx-leaks?
  "True if idx-sym occurs anywhere OUTSIDE an aget/aset index position — i.e. it
  would feed per-lane arithmetic, which the block-scalar `idx` can't represent."
  [core idx-sym]
  (let [blanked (walk/postwalk
                 (fn [f]
                   (cond
                     (aset-form?* f) (list (first f) (nth f 1) 0 (nth f 3))
                     (aget-form?* f) (list (first f) (nth f 1) 0)
                     :else f))
                 core)]
    (sym-occurs? blanked idx-sym)))

(defn analyze-elementwise-vectorizable
  "Analyze an elementwise kernel body for affine-index vectorization. Returns nil
  (fall back to scalar) or a map {:stores [{:arr :val'} ...] :divisors [C-expr ...]}
  where val' has contiguous/rowrel loads rewritten to ::vload markers. Width-independent
  — the runtime guard is emitted per width by emit-vectorized-elementwise-loop.

  Bails (nil) unless the body is straight-line elementwise stores over contiguous
  positions, every array subscript classifies, idx never leaks into per-lane
  arithmetic, and there is no control flow. Real GPU-helper calls (gpufn_*) are caught
  downstream in the emitter (their vector overloads are a separate concern)."
  [body idx-sym array-syms]
  (when (contains? #{"float" "double"} *scalar-type*)
    (try
      (let [core (inline-pure-lets body)
            stores (or (collect-stores core) (vec-bail!))]
        (when (idx-leaks? core idx-sym) (vec-bail!))
        ;; no control flow in a vectorizable straight-line body
        (walk/postwalk
         (fn [f] (when (and (seq? f)
                            (contains? #{'if 'when 'case 'cond 'loop 'loop* 'and 'or} (first f)))
                   (vec-bail!)) f)
         core)
        (let [divisors! (atom [])
              store-recs
              (mapv (fn [st]
                      (let [[_ arr idx-e val] st]
                        ;; a vstore writes contiguous lanes — the store index must be idx
                        (when-not (= :contiguous (:kind (classify-subscript idx-e idx-sym)))
                          (vec-bail!))
                        ;; a nested store inside the value (horizontally-fused multi-output
                        ;; map) can't be a lane vector store — bail to scalar
                        (walk/postwalk (fn [f] (when (aset-form?* f) (vec-bail!)) f) val)
                        {:arr arr :val (preprocess-value val idx-sym array-syms divisors!)}))
                    stores)]
          {:stores store-recs :divisors (vec (distinct @divisors!))}))
      (catch clojure.lang.ExceptionInfo e
        (if (::bail (ex-data e)) nil (throw e))))))

(defn- emit-vec-store
  "Emit one vstoreV statement for a store {:arr :val} at width V (element type =
  *scalar-type*, bound by the caller). store-name overrides the target C name (used by
  the SegMap path whose output param is the literal `out`, not c-symbol-mangled)."
  [{:keys [arr val]} idx-sym array-syms opencl-idx V store-name]
  (let [val-str (emit-expr val idx-sym array-syms opencl-idx)
        vtype   (str *scalar-type* V)
        ;; a pure-broadcast value (no vector load) must be widened to the vector type
        data    (if (value-has-vload? val) val-str (str "(" vtype ")(" val-str ")"))]
    (str "vstore" V "(" data ", 0, " (or store-name (c-symbol arr)) " + " opencl-idx ");")))

(defn emit-vectorized-elementwise-loop
  "Given an elementwise kernel body and the scalar loop body already emitted by the
  backend (scalar-body-str), return the FULL loop region: a runtime divisibility
  guard choosing a float4 / float2 vectorized grid-stride loop, falling back to the
  scalar grid-stride loop. Returns nil if the body is not provably vectorizable, in
  which case the caller emits its usual scalar loop.

  opts: {:n-bound \"_n_bound\"   (the C variable holding the element count)
         :store-name \"out\"}     (optional literal target name for a single store)"
  [body idx-sym array-syms opencl-idx scalar-body-str
   {:keys [n-bound store-name] :or {n-bound "_n_bound"}}]
  (when (:vectorize? *emit-config*)
    (when-let [{:keys [stores divisors]} (analyze-elementwise-vectorizable body idx-sym array-syms)]
      (try
        (let [scalar-loop (str "for (int " opencl-idx " = get_global_id(0); "
                               opencl-idx " < " n-bound "; "
                               opencl-idx " += get_global_size(0)) {\n"
                               "        " scalar-body-str "\n"
                               "    }")
              emit-vec-loop
              (fn [V]
                (let [store-strs (str/join " "
                                           (map #(emit-vec-store % idx-sym array-syms opencl-idx V store-name) stores))]
                  (str "for (int _vb = get_global_id(0); _vb < (" n-bound " / " V "); "
                       "_vb += get_global_size(0)) {\n"
                       "        int " opencl-idx " = _vb * " V ";\n"
                       "        " store-strs "\n"
                       "    }")))
              guard (fn [V] (str/join " && "
                                      (cons (str "((" n-bound " % " V ") == 0)")
                                            (map (fn [d]
                                                   (str "((" (emit-expr d idx-sym array-syms opencl-idx)
                                                        " % " V ") == 0)"))
                                                 divisors))))
              v4 (binding [*vec-width* 4] (emit-vec-loop 4))
              v2 (binding [*vec-width* 2] (emit-vec-loop 2))]
          ;; A gpufn_ helper call surviving in the vectorized body would pass a vector to a
          ;; scalar helper — bail to scalar. (Deftm helpers inline via emit-expr; numeric
          ;; ops resolve to C operators — both pass. A non-inlinable helper leaves gpufn_.)
          (when-not (or (str/includes? v4 "gpufn_") (str/includes? v2 "gpufn_"))
            (str "if (" (guard 4) ") {\n    " v4 "\n    } else if (" (guard 2) ") {\n    "
                 v2 "\n    } else {\n    " scalar-loop "\n    }")))
        ;; A precision-changing vector cast (or other vector-unsafe form) surfaced during
        ;; emission → fall back to the scalar loop (caller emits it).
        (catch clojure.lang.ExceptionInfo e
          (if (::bail (ex-data e)) nil (throw e)))))))
