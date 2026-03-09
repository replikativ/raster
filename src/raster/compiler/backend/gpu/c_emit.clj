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
              *scalar-type* \"float\"
              *soa-expansions* {}]
      (emit-expr body idx-sym array-syms \"idx\"))"
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]))

;; ================================================================
;; Backend configuration
;; ================================================================

(def ^:dynamic *emit-config*
  "Backend-specific emission configuration.
  Keys:
    :cast-style       - :c for (float)x, :glsl for float(x)
    :struct-ctor      - :c99 for (T){.x=a}, :glsl for T(a,b), :cpp for T{a,b}
    :struct-typedef   - :c99 for 'typedef struct{...}T;', :glsl/:cpp for 'struct T{...};'
    :atomic-add-int   - function name for int atomic add
    :atomic-add-float - function name or :cas-helper for CAS loop
    :float-abs        - \"fabs\" or \"abs\"
    :float-max        - \"fmax\" or \"max\"
    :float-min        - \"fmin\" or \"min\"
    :float-suffix?    - append 'f' to float literals"
  {:cast-style      :c
   :struct-ctor     :c99
   :struct-typedef  :c99
   :atomic-add-int  "atomic_add"
   :atomic-add-float :cas-helper
   :float-abs       "fabs"
   :float-max       "fmax"
   :float-min       "fmin"
   :float-suffix?   true})

(def opencl-config
  {:cast-style      :c
   :struct-ctor     :c99
   :struct-typedef  :c99
   :atomic-add-int  "atomic_add"
   :atomic-add-float :cas-helper
   :float-abs       "fabs"
   :float-max       "fmax"
   :float-min       "fmin"
   :float-suffix?   true})

(def glsl-config
  {:cast-style      :glsl
   :struct-ctor     :glsl
   :struct-typedef  :glsl
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
  {:cast-style      :c
   :struct-ctor     :cpp
   :struct-typedef  :cpp
   :atomic-add-int  "atomicAdd"
   :atomic-add-float "atomicAdd"
   :float-abs       "fabsf"
   :float-max       "fmaxf"
   :float-min       "fminf"
   :float-suffix?   true})

;; ================================================================
;; Dynamic state
;; ================================================================

(def ^:dynamic *scalar-type* "double")

(def ^:dynamic *soa-expansions*
  "Maps symbol-name-as-symbol -> {:scalar-tag :soa-tag :fields [{:name :element-tag :array-tag}...]}"
  {})

(def ^:dynamic *idx-sym*
  "The par loop index symbol (always uint/int). Used by infer-c-type."
  nil)

;; ================================================================
;; Op mapping (valid across all C-family backends)
;; ================================================================

(def op-map
  "Maps Clojure/Java op symbols to C equivalents.
  Backend-specific overrides (fabs vs abs) are handled via *emit-config*."
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
   'fma   "fma"})

;; ================================================================
;; Devirtualized arithmetic → C operator/function mapping
;; ================================================================

(def ^:private mangled-prefix->c-op
  "Maps mangled deftm name prefixes (from types/mangle) back to C operators or functions.
  These prefixes appear after devirtualization, e.g. _plus__m_double_double for (+ a b).
  Binary infix ops map to their C operator string; unary/math functions map to C function names."
  {"_plus_"  {:kind :infix  :op "+"}
   "_minus_" {:kind :infix  :op "-"}
   "_star_"  {:kind :infix  :op "*"}
   "_div_"   {:kind :infix  :op "/"}
   "_lt_"    {:kind :infix  :op "<"}
   "_gt_"    {:kind :infix  :op ">"}
   "_lteq_"  {:kind :infix  :op "<="}
   "_gteq_"  {:kind :infix  :op ">="}
   "mod"     {:kind :floored-mod}
   "rem"     {:kind :infix  :op "%"}
   "quot"    {:kind :infix  :op "/"}
   "sin"     {:kind :fn     :op "sin"}
   "cos"     {:kind :fn     :op "cos"}
   "tan"     {:kind :fn     :op "tan"}
   "exp"     {:kind :fn     :op "exp"}
   "log"     {:kind :fn     :op "log"}
   "sqrt"    {:kind :fn     :op "sqrt"}
   "abs"     {:kind :fn     :op "fabs"}
   "pow"     {:kind :fn     :op "pow"}
   "floor"   {:kind :fn     :op "floor"}
   "ceil"    {:kind :fn     :op "ceil"}
   "round"   {:kind :fn     :op "round"}
   "fma"     {:kind :fn     :op "fma"}})

(defn- mangled-name->c-op
  "Given a mangled deftm symbol name like '_plus__m_double_double', extract the
  base op prefix and look up the corresponding C operator/function.
  Returns {:kind :infix/:fn, :op \"...\"} or nil."
  [sym-name]
  (when-let [idx (str/index-of sym-name "_m_")]
    (let [prefix (subs sym-name 0 idx)]
      (get mangled-prefix->c-op prefix))))

;; ================================================================
;; Type mappings
;; ================================================================

(def type-map
  {:double "double"
   :float  "float"
   :int    "int"
   :long   "long long"})

(def tag->ctype
  "Map walker :tag metadata to C type strings."
  {'ints "int" 'floats "float" 'doubles "double" 'longs "long"
   'int "int" 'float "float" 'double "double" 'long "long"})

(def element-tag->c
  "Map primitive element-tag to C type string."
  {'double "double" 'float "float" 'long "long" 'int "int"})

;; ================================================================
;; Symbol mangling
;; ================================================================

(def ^:private c-reserved-words
  #{"int" "float" "double" "long" "void" "char" "short" "unsigned" "signed"
    "for" "while" "do" "if" "else" "return" "break" "continue" "switch" "case"
    "struct" "union" "enum" "typedef" "const" "static" "extern" "auto" "register"
    "sizeof" "goto" "volatile"
    ;; OpenCL specific
    "kernel" "global" "local" "constant" "private" "restrict"
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
              (str/replace "!" "_b"))]
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

(defn infer-c-type
  "Infer the C type of an expression from walker :tag metadata on symbols,
  primitive casts, and aget on typed arrays. Falls back to *scalar-type*."
  [expr]
  (cond
    (and (seq? expr) (= 2 (count expr))
         (contains? #{'int 'float 'double 'long} (first expr)))
    (get tag->ctype (first expr) *scalar-type*)

    (and (seq? expr) (descriptor/aget-op? (first expr))
         (>= (count expr) 3))
    (let [arr     (second expr)
          arr-sym (when (symbol? arr) (symbol (name arr)))]
      (if-let [soa-info (get *soa-expansions* arr-sym)]
        (name (:scalar-tag soa-info))
        (let [tag (when (symbol? arr) (or (:raster.type/tag (meta arr)) (:tag (meta arr))))]
          (get tag->ctype tag *scalar-type*))))

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

    (integer? expr) "int"
    (float? expr) *scalar-type*

    (and (seq? expr)
         (let [op (first expr)]
           (or (descriptor/addition-op? op) (descriptor/subtraction-op? op) (descriptor/multiplication-op? op)))
         (let [types (map infer-c-type (rest expr))]
           (every? #(contains? #{"int" "uint"} %) types)))
    (let [types (map infer-c-type (rest expr))]
      (if (some #(= "uint" %) types) "uint" "int"))

    :else *scalar-type*))

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
  "Check if an expression tree contains void (side-effect) operations."
  [expr]
  (cond
    (not (seq? expr)) false
    (void-form? expr) true
    :else (some has-side-effects? (rest expr))))

;; ================================================================
;; Backend-specific helpers
;; ================================================================

(defn- remap-type
  "Remap a C type name according to backend config (e.g. long -> int for GLSL)."
  [ctype]
  (get (:type-remap *emit-config*) ctype ctype))

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

(defn- emit-struct-ctor
  "Emit a struct constructor according to backend config."
  [type-name fields-with-values]
  (case (:struct-ctor *emit-config*)
    :c99  (str "(" type-name "){ "
               (str/join ", " (map (fn [[fname val]] (str "." fname "=" val))
                                   fields-with-values))
               " }")
    :glsl (str type-name "("
               (str/join ", " (map second fields-with-values))
               ")")
    :cpp  (str type-name "{"
               (str/join ", " (map second fields-with-values))
               "}")))

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
    ;; aset on SoA array -> field-by-field writes via temp struct
    (and (seq? expr)
         (descriptor/aset-op? (first expr))
         (>= (count expr) 4))
    (let [[_ arr idx-e val-e] expr
          arr-sym (when (symbol? arr) (symbol (name arr)))]
      (if-let [soa-info (get *soa-expansions* arr-sym)]
        (let [scalar-name (name (:scalar-tag soa-info))
              idx-str (emit-expr idx-e idx-sym array-syms opencl-idx)
              val-str (emit-expr val-e idx-sym array-syms opencl-idx)
              fields  (:fields soa-info)
              arr-c   (c-symbol arr)
              writes  (str/join " "
                                (map (fn [{:keys [name]}]
                                       (str arr-c "_" name "[" idx-str "] = _soa_tmp." name ";"))
                                     fields))]
          (str "{ " scalar-name " _soa_tmp = " val-str "; " writes " }"))
        (let [val-str (emit-expr val-e idx-sym array-syms opencl-idx)
              idx-str (emit-expr idx-e idx-sym array-syms opencl-idx)
              ;; GLSL: cast value to target array type
              arr-ctype (when (and (not (supports-stmt-expr?)) (symbol? arr))
                          (or (get element-tag->c (or (:raster.type/tag (meta arr)) (:tag (meta arr))))
                              *scalar-type*))
              val-str (if arr-ctype
                        (str arr-ctype "(" val-str ")")
                        val-str)]
          (str (c-symbol arr) "[" idx-str "] = " val-str ";"))))

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
          {:keys [env locals loop-stmts]}
          (reduce
           (fn [{:keys [env locals loop-stmts seen-names]} [sym val]]
             (let [val-subst (walk/postwalk
                              (fn [f] (if (and (symbol? f) (contains? env f))
                                        (get env f) f))
                              val)]
                ;; Detect loop RHS in GLSL (needs statement hoisting)
               (if (and (not (supports-stmt-expr?))
                        (seq? val-subst) (= 'loop (first val-subst)))
                  ;; Hoist loop: emit as statements, alias sym → first loop var
                 (let [loop-var (first (take-nth 2 (second val-subst)))
                       loop-c-name (c-symbol loop-var)
                       loop-code (emit-expr val-subst idx-sym array-syms opencl-idx)]
                   {:env (assoc env sym (symbol loop-c-name))
                    :locals locals
                    :loop-stmts (conj (or loop-stmts []) loop-code)
                    :seen-names (or seen-names {})})
                 (if (multi-use? sym)
                   (let [base-name (c-symbol sym)
                         prev-count (get seen-names base-name 0)
                         c-name (if (> prev-count 0) (str base-name "_" prev-count) base-name)
                         c-expr (emit-expr val-subst idx-sym array-syms opencl-idx)
                         c-type (infer-c-type val-subst)]
                     {:env (assoc env sym (symbol c-name))
                      :locals (conj locals [c-name c-expr c-type])
                      :loop-stmts loop-stmts
                      :seen-names (assoc seen-names base-name (inc prev-count))})
                   {:env (assoc env sym val-subst)
                    :locals locals
                    :loop-stmts loop-stmts
                    :seen-names (or seen-names {})}))))
           {:env {} :locals [] :loop-stmts [] :seen-names {}}
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
          body-stmts (str/join " "
                               (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx))
                                    body-subst))]
      (str "{ " (when (seq loop-prefix) (str loop-prefix " "))
           (when (seq locals) (str local-decls " "))
           body-stmts " }"))

    ;; loop in statement context -> while loop (no ({ }) wrapper)
    (and (seq? expr) (= 'loop (first expr)))
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
          loop-body (emit-loop-body body var-names idx-sym array-syms opencl-idx)]
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

(defn- emit-loop-body
  "Emit the body of a loop as C while-body statements."
  [body var-names idx-sym array-syms opencl-idx]
  (let [body-expr (if (= 1 (count body)) (first body) (cons 'do body))]
    (emit-loop-expr body-expr var-names idx-sym array-syms opencl-idx)))

(defn- emit-loop-expr
  "Recursively emit loop body, translating recur to assignments+continue."
  [expr var-names idx-sym array-syms opencl-idx]
  (cond
    ;; recur -> parallel assignment via temps, then assign back + continue
    (and (seq? expr) (= 'recur (first expr)))
    (let [new-vals (rest expr)
          indexed (map-indexed vector (map vector var-names new-vals))
          ;; Emit temp declarations: type _t0 = expr_a; type _t1 = expr_b; ...
          temps (str/join " "
                          (map (fn [[i [sym val]]]
                                 (let [typ (remap-type (infer-c-type val))
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
             (emit-loop-expr then-expr var-names idx-sym array-syms opencl-idx)
             " } else { "
             (emit-loop-expr else-expr var-names idx-sym array-syms opencl-idx)
             " }")
        (str "if (" cond-str ") { "
             (emit-loop-expr then-expr var-names idx-sym array-syms opencl-idx)
             " } else { break; }")))

    ;; do block
    (and (seq? expr) (= 'do (first expr)))
    (let [stmts (rest expr)
          leading (butlast stmts)
          last-e (last stmts)]
      (str (str/join " " (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx)) leading))
           " " (emit-loop-expr last-e var-names idx-sym array-syms opencl-idx)))

    ;; let in loop body
    (and (seq? expr) (contains? #{'let 'let*} (first expr)))
    (let [[_ bindings & body] expr
          pairs (partition 2 bindings)
          decls (let [seen (atom {})]
                  (str/join " "
                            (map (fn [[sym val]]
                                   (let [base (c-symbol sym)
                                         prev (get @seen base 0)
                                         c-name (if (> prev 0) (str base "_" prev) base)]
                                     (swap! seen assoc base (inc prev))
                                     (str (remap-type (infer-c-type val)) " " c-name " = "
                                          (emit-expr val idx-sym array-syms opencl-idx) ";")))
                                 pairs)))
          body-strs (emit-loop-body body var-names idx-sym array-syms opencl-idx)]
      (str decls " " body-strs))

    ;; when in loop
    (and (seq? expr) (= 'when (first expr)))
    (let [[_ cond-expr & body] expr
          cond-str (emit-expr cond-expr idx-sym array-syms opencl-idx)]
      (str "if (" cond-str ") { "
           (str/join " " (map (fn [s] (emit-stmt s idx-sym array-syms opencl-idx)) (butlast body)))
           " " (emit-loop-expr (last body) var-names idx-sym array-syms opencl-idx)
           " } else { break; }"))

    ;; Void terminal (aset, collect!, atomic-add!) -> emit as stmt then break
    (void-form? expr)
    (str (emit-stmt expr idx-sym array-syms opencl-idx) " break;")

    ;; Terminal expression -> assign to first var and break
    :else
    (str (c-symbol (first var-names)) " = "
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

     ;; (.field expr) -> struct field access
     (and (seq? expr) (= 2 (count expr))
          (symbol? (first expr))
          (str/starts-with? (name (first expr)) "."))
     (let [field-name (subs (name (first expr)) 1)
           obj-str    (emit-expr (second expr) idx-sym array-syms opencl-idx)]
       (str "(" obj-str ")." field-name))

     ;; (->Type arg...) -> struct constructor
     (and (seq? expr) (symbol? (first expr))
          (str/starts-with? (name (first expr)) "->"))
     (let [type-name  (subs (name (first expr)) 2)
           scalar-tag (symbol type-name)
           soa-reg    @types/soa-registry
           soa-info   (get soa-reg scalar-tag)
           fields     (:fields soa-info)
           args       (rest expr)]
       (if (and soa-info (= (count fields) (count args)))
         (emit-struct-ctor type-name
                           (map (fn [{:keys [name]} arg]
                                  [name (emit-expr arg idx-sym array-syms opencl-idx)])
                                fields args))
         ;; Fallback: function-call style
         (str type-name "("
              (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) args))
              ")")))

     ;; aget on SoA array -> struct literal from flat reads
     (and (seq? expr)
          (descriptor/aget-op? (first expr))
          (>= (count expr) 3))
     (let [arr     (second expr)
           arr-sym (when (symbol? arr) (symbol (name arr)))
           idx-expr (nth expr 2)]
       (if-let [soa-info (get *soa-expansions* arr-sym)]
         (let [scalar-name (name (:scalar-tag soa-info))
               idx-str     (emit-expr idx-expr idx-sym array-syms opencl-idx)
               arr-c       (c-symbol arr)
               fields      (:fields soa-info)]
           (emit-struct-ctor scalar-name
                             (map (fn [{:keys [name]}]
                                    [name (str arr-c "_" name "[" idx-str "]")])
                                  fields)))
         ;; plain array aget
         (str (c-symbol arr) "["
              (emit-expr idx-expr idx-sym array-syms opencl-idx) "]")))

     ;; Primitive cast
     (and (seq? expr)
          (contains? #{'double 'float 'long 'int} (first expr))
          (= 2 (count expr)))
     (emit-cast (remap-type (name (first expr)))
                (emit-expr (second expr) idx-sym array-syms opencl-idx))

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
           {:keys [env locals]}
           (reduce
            (fn [{:keys [env locals seen-names]} [sym val]]
              (let [val-subst (walk/postwalk
                               (fn [f] (if (and (symbol? f) (contains? env f))
                                         (get env f) f))
                               val)]
                (if (multi-use? sym)
                  (let [base-name (c-symbol sym)
                        prev-count (get seen-names base-name 0)
                        c-name (if (> prev-count 0)
                                 (str base-name "_" prev-count)
                                 base-name)
                        c-expr (emit-expr val-subst idx-sym array-syms opencl-idx)
                        c-type (remap-type (infer-c-type val-subst))]
                    {:env (assoc env sym (symbol c-name))
                     :locals (conj locals [c-name c-expr c-type])
                     :seen-names (assoc seen-names base-name (inc prev-count))})
                  {:env (assoc env sym val-subst)
                   :locals locals
                   :seen-names (or seen-names {})})))
            {:env {} :locals [] :seen-names {}}
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
                             (some has-side-effects? leading-body))
           body-str (emit-expr tail-body idx-sym array-syms opencl-idx)]
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

         :else body-str))

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
     (and (seq? expr) (= 'loop (first expr)))
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
           result-var (c-symbol (first var-names))
           loop-body (emit-loop-body body var-names idx-sym array-syms opencl-idx)]
       (if (supports-stmt-expr?)
         (str "({ " decls " while (1) { " loop-body " } " result-var "; })")
         ;; GLSL: no statement-expressions; emit inline declarations + while
         ;; Result is in the first loop variable after break.
         (str decls " while (true) { " loop-body " }")))

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

     ;; Bitwise ops: bit-and, bit-or, bit-xor, bit-shift-left, unsigned-bit-shift-right
     (and (seq? expr) (= 3 (count expr))
          (contains? #{'bit-and 'clojure.core/bit-and
                       'bit-or 'clojure.core/bit-or
                       'bit-xor 'clojure.core/bit-xor
                       'bit-shift-left 'clojure.core/bit-shift-left
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
           :infix (str "(" (str/join (str " " (:op c-op) " ")
                                     (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")")
           :fn    (str (:op c-op) "("
                       (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")")
           :floored-mod (let [ea (emit-expr (first args) idx-sym array-syms opencl-idx)
                              eb (emit-expr (second args) idx-sym array-syms opencl-idx)]
                          (str "((" ea " % " eb " + " eb ") % " eb ")")))
         ;; Non-arithmetic deftm helper -> C helper invocation
         (let [c-name (str "gpufn_" (c-symbol sym))]
           (str c-name "("
                (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) args))
                ")"))))

     ;; .invk typed dispatch on deftm helper -> check for devirtualized arithmetic
     (and (seq? expr) (= '.invk (first expr))
          (>= (count expr) 3)
          (symbol? (second expr))
          (resolve-gpu-inlinable-var (second expr)))
     (let [resolved-var (resolve-gpu-inlinable-var (second expr))
           var-name (str (:name (meta resolved-var)))
           args (nnext expr)
           c-op (mangled-name->c-op var-name)]
       (if c-op
         ;; Devirtualized arithmetic op -> emit native C operator/function
         (case (:kind c-op)
           :infix (str "(" (str/join (str " " (:op c-op) " ")
                                     (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")")
           :fn    (str (:op c-op) "("
                       (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) args)) ")")
           :floored-mod (let [ea (emit-expr (first args) idx-sym array-syms opencl-idx)
                              eb (emit-expr (second args) idx-sym array-syms opencl-idx)]
                          (str "((" ea " % " eb " + " eb ") % " eb ")")))
         ;; Non-arithmetic deftm helper -> C helper invocation
         (let [base-sym (symbol (str (.-ns ^clojure.lang.Var resolved-var))
                                (str (:name (meta resolved-var))))
               c-name (str "gpufn_" (c-symbol base-sym))]
           (str c-name "("
                (str/join ", " (map #(emit-expr % idx-sym array-syms opencl-idx) args))
                ")"))))

     ;; deftm inlining: try to inline before generic function fallback
     (and (seq? expr) (symbol? (first expr)))
     (or (try-inline-deftm expr idx-sym array-syms opencl-idx)

         ;; Unary function
         (when (= 2 (count expr))
           (let [[op arg] expr]
             (str (resolve-op op) "(" (emit-expr arg idx-sym array-syms opencl-idx) ")")))

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
     (str expr))))

;; ================================================================
;; deftm inlining support
;; ================================================================

(defn infer-arg-tag
  "Infer the type tag of an expression in the current body context."
  [expr]
  (cond
    (symbol? expr)
    (or (:tag (meta expr))
        ;; Fall back to scalar type for untagged symbols (push constants, locals)
        (when *scalar-type*
          (get {"float" 'float "double" 'double "int" 'int "long" 'long}
               *scalar-type*)))

    (and (seq? expr) (descriptor/aget-op? (first expr))
         (>= (count expr) 3))
    (let [arr      (second expr)
          arr-sym  (when (symbol? arr) (symbol (name arr)))
          soa-info (when arr-sym (get *soa-expansions* arr-sym))]
      (if soa-info
        (:scalar-tag soa-info)
        (or (when (symbol? arr)
              (get {'doubles 'double 'floats 'float 'longs 'long 'ints 'int}
                   (or (:raster.type/tag (meta arr)) (:tag (meta arr)))))
            ;; Fall back to scalar type for untagged arrays
            (when *scalar-type*
              (get {"float" 'float "double" 'double "int" 'int "long" 'long}
                   *scalar-type*)))))

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
                            (when-let [walked-body (get (meta impl-v) :raster.core/deftm-walked-body)]
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
                  (contains? #{'let 'let* 'loop} (first form))
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
                                    'if 'when 'cond 'case 'loop 'recur
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

(defn collect-soa-expansions
  "Collect SoA expansion info for symbols in body that have SoA type tags."
  [body]
  (let [result  (atom {})
        soa-rev (try @types/soa-reverse-registry (catch Exception _ {}))
        soa-reg (try @types/soa-registry (catch Exception _ {}))]
    (walk/postwalk
     (fn [f]
       (when (and (symbol? f) (:tag (meta f)))
         (let [tag        (:tag (meta f))
               scalar-tag (get soa-rev tag)]
           (when scalar-tag
             (let [soa-info (get soa-reg scalar-tag)]
               (when soa-info
                 (swap! result assoc (symbol (name f))
                        {:scalar-tag scalar-tag
                         :soa-tag    tag
                         :fields     (:fields soa-info)}))))))
       f)
     body)
    @result))

(defn emit-struct-typedefs
  "Emit C struct type declarations for all scalar defvalue types in soa-expansions."
  [soa-expansions]
  (when (seq soa-expansions)
    (let [seen (atom #{})]
      (->> (vals soa-expansions)
           (keep (fn [{:keys [scalar-tag fields]}]
                   (when-not (@seen scalar-tag)
                     (swap! seen conj scalar-tag)
                     (let [field-decls (str/join " "
                                                 (map (fn [{:keys [name element-tag]}]
                                                        (str (get element-tag->c element-tag "float")
                                                             " " name ";"))
                                                      fields))
                           tname (clojure.core/name scalar-tag)]
                       (case (:struct-typedef *emit-config*)
                         :c99  (str "typedef struct { " field-decls " } " tname ";")
                         :glsl (str "struct " tname " { " field-decls " };")
                         :cpp  (str "struct " tname " { " field-decls " };"))))))
           (str/join "\n")))))

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
