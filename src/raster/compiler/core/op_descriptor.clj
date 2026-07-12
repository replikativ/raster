(ns raster.compiler.core.op-descriptor
  "Unified compiler operation descriptor registry.

  Descriptors merge multiple compiler facets under one lookup point, e.g.:
    :buffer {:allocates? ... :in-place-arg ... :alloc-form ... :rewrite-fn ...}
    :effects {:pure? ... :mutating? ... :primitive-mutation? ...}
    :device {:rule ...}
    :shape  {:dim-rule ...}

  Compiler passes and op libraries should register and query metadata directly
  through this registry."
  (:require [raster.compiler.support.mangled :as mangled]
            [raster.compiler.core.dtype :as dtype]))

(defonce ^:private descriptor-registry (atom {}))

(defn register-op-descriptor!
  "Merge descriptor facets for op-sym into the unified registry."
  [op-sym descriptor]
  (swap! descriptor-registry update op-sym #(merge % descriptor))
  nil)

(defn get-op-descriptor
  "Look up the unified descriptor for op-sym. Returns nil if none registered."
  [op-sym]
  (get @descriptor-registry op-sym))

(defn get-buffer-semantics
  "Return the buffer facet for op-sym, if registered."
  [op-sym]
  (get-in (get-op-descriptor op-sym) [:buffer]))

(declare resolve-op-descriptor)

(defonce ^:private auto-buffer-cache (atom {}))

(defn- array-alloc-size
  "If form is a pure array allocation, return the size expression. Nil otherwise.
   Does NOT match aclone (copy, not pure allocation)."
  [form]
  (when (and (seq? form) (symbol? (first form)))
    (let [n (name (first form))]
      (when (or (.endsWith n "-array") (.startsWith n "zeros-like"))
        (if (= 2 (count form)) (second form) (last form))))))

(defn- param-derived?
  "True if expr depends only on param-set symbols (transitively through bindings)."
  [expr param-set binding-map]
  (cond
    (contains? param-set expr) true
    (symbol? expr) (when-let [init (get binding-map expr)]
                     (param-derived? init param-set binding-map))
    (number? expr) true
    (seq? expr) (let [head (first expr)
                      args (if (= '.invk head) (drop 2 expr) (rest expr))]
                  (and (symbol? head)
                       (every? #(param-derived? % param-set binding-map) args)))
    :else false))

(defn- detect-auto-buffer-semantics
  "Analyze a deftm var's walked body for the single-alloc-return pattern.
   Returns a buffer-semantics map or nil."
  [v]
  (let [m (meta v)
        walked-body (or (:raster.core/deftm-walked-body m)
                        ((requiring-resolve 'raster.core/ensure-walked-body!) v))
        params (:raster.core/deftm-params m)]
    (when (and walked-body params (= 1 (count walked-body)))
      (let [body (first walked-body)]
        ;; Case 1: body is a direct pure alloc call (zeros-like, double-array)
        ;; Returns: just the pre-allocated buffer (no compute body)
        ;; (walked body is closed-core: let -> let*)
        (if (and (seq? body) (not (contains? #{'let 'let*} (first body))) (array-alloc-size body))
          (let [param-set (set params)
                size (array-alloc-size body)]
            (when (param-derived? size param-set {})
              (let [param-idx (into {} (map-indexed (fn [i p] [p i]) params))
                    resolve-to-args
                    (fn resolve-to-args [expr args]
                      (cond
                        (and (symbol? expr) (contains? param-idx expr))
                        (nth args (get param-idx expr))
                        (seq? expr)
                        (apply list (first expr) (map #(resolve-to-args % args) (rest expr)))
                        :else expr))
                    alloc-ctor (first body)]
                {:allocates? true
                 :in-place-arg nil
                 :alloc-form (fn [args _opts]
                               (let [resolved-size (resolve-to-args size args)]
                                 (if (> (count body) 2)
                                   (list alloc-ctor
                                         (resolve-to-args (second body) args)
                                         resolved-size)
                                   (list alloc-ctor resolved-size))))
                 :rewrite-fn (fn [_args buf-sym] buf-sym)})))
          ;; Case 2: body is a let form with alloc-return pattern
          ;; (walked body is closed-core: let -> let*)
          (when (and (seq? body) (contains? #{'let 'let*} (first body)))
            (let [[_ bindings & body-exprs] body
                  pairs (vec (partition 2 bindings))
                  binding-map (into {} (map vec pairs))
                  param-set (set params)
                  return-expr (last body-exprs)]
            ;; Find the alloc binding: either returned directly (symbol) or
            ;; passed to a mutating call as last expr (BLAS pattern)
              (let [alloc-target (cond
                                 ;; Direct return: (let [... out (alloc ...) ...] ... out)
                                   (symbol? return-expr) return-expr
                                 ;; Mutating-call return: (let [... buf (alloc ...) ...]
                                 ;;                        (blas-op! ... buf ...) )
                                 ;; The alloc'd buffer is one of the args in the call
                                   (and (seq? return-expr) (symbol? (first return-expr)))
                                   (let [alloc-syms (set (for [[sym init] pairs
                                                               :when (array-alloc-size init)]
                                                           sym))]
                                     (first (filter alloc-syms (rest return-expr))))
                                   :else nil)
                    match (when alloc-target
                            (first (keep-indexed
                                    (fn [i [sym init]]
                                      (when (= sym alloc-target)
                                        (when-let [size (array-alloc-size init)]
                                          (when (param-derived? size param-set binding-map)
                                            {:idx i :sym sym :init init :size size
                                             :ctor (first init)
                                             :multi-arg? (> (count init) 2)
                                             :ref-arg (when (> (count init) 2) (second init))}))))
                                    pairs)))]
                (when match
                  (let [param-idx (into {} (map-indexed (fn [i p] [p i]) params))
                        ;; Resolve size expr to use call args
                        resolve-to-args
                        (fn resolve-to-args [expr args]
                          (cond
                            (and (symbol? expr) (contains? param-idx expr))
                            (nth args (get param-idx expr))
                            (and (symbol? expr) (get binding-map expr))
                            (resolve-to-args (get binding-map expr) args)
                            (seq? expr)
                            (apply list (first expr) (map #(resolve-to-args % args) (rest expr)))
                            :else expr))]
                    ;; Build into-variant: body with alloc binding → buf param
                    (let [alloc-sym (:sym match)
                          alloc-idx (:idx match)
                          other-pairs (vec (concat (take alloc-idx pairs)
                                                   (drop (inc alloc-idx) pairs)))
                          sub (fn sub [form sym rep]
                                (cond (= form sym) rep
                                      (seq? form) (with-meta (apply list (map #(sub % sym rep) form)) (meta form))
                                      (vector? form) (mapv #(sub % sym rep) form)
                                      :else form))
                          into-binds (vec (mapcat (fn [[s e]] [s (sub e alloc-sym 'buf__auto)]) other-pairs))
                          into-exprs (map #(sub % alloc-sym 'buf__auto) body-exprs)
                          into-template (list* 'let into-binds into-exprs)]
                      {:allocates? true
                       :in-place-arg nil
                       :alloc-form (fn [args _opts]
                                     (let [resolved-size (resolve-to-args (:size match) args)]
                                       (if (:multi-arg? match)
                                         (list (:ctor match)
                                               (resolve-to-args (:ref-arg match) args)
                                               resolved-size)
                                         (list (:ctor match) resolved-size))))
                       ;; Rewrite: inline body with alloc replaced by buf
                       :rewrite-fn (fn [args buf-sym]
                                     (let [smap (merge (zipmap params args) {'buf__auto buf-sym})]
                                       (clojure.walk/postwalk
                                        (fn [f] (if (and (symbol? f) (contains? smap f)) (get smap f) f))
                                        into-template)))})))))))))))

(defn resolve-buffer-semantics
  "Resolve the buffer facet, handling mangled/unqualified names.
  Falls back to auto-detection from the deftm walked body when no
  manual registration exists.
  Returns [buffer-facet base-op] or nil."
  [op-sym]
  (or
   ;; 1. Manual registration
   (when-let [[descriptor base-op] (resolve-op-descriptor op-sym)]
     (when-let [buffer (:buffer descriptor)]
       [buffer base-op]))
   ;; 2. Auto-detection from walked body (cached)
   (when (and (symbol? op-sym) (namespace op-sym))
     (let [cached (get @auto-buffer-cache op-sym ::miss)]
       (if (not= cached ::miss)
         cached
         (let [result (try
                        (when-let [v (resolve op-sym)]
                          (when (and (var? v) (:raster.core/deftm (meta v)))
                            (when-let [bs (detect-auto-buffer-semantics v)]
                              [bs op-sym])))
                        (catch Exception _ nil))]
           (swap! auto-buffer-cache assoc op-sym result)
           result))))))

(defn registered-op-descriptors
  "Return the set of ops with any registered descriptor facets."
  []
  (set (keys @descriptor-registry)))

(defn extract-base-op
  "Extract base op from a mangled deftm symbol when possible."
  [sym]
  (mangled/extract-deftm-base sym))

(defn resolve-op-descriptor
  "Resolve a unified descriptor, handling mangled and unqualified names.
  Returns [descriptor base-op] or nil."
  [op-sym]
  (or
   (when-let [d (get-op-descriptor op-sym)]
     [d op-sym])
   (when-let [base (extract-base-op op-sym)]
     (when-let [d (get-op-descriptor base)]
       [d base]))
   (when (symbol? op-sym)
     (let [n (name op-sym)
           base-n (mangled/unqualified-base-name op-sym)]
       (when (not= base-n n)
         (some (fn [[k v]]
                 (when (= (name k) base-n)
                   [v k]))
               @descriptor-registry))))))

(defn register-buffer-semantics!
  "Register the buffer/effects facets exposed by op-semantics."
  [op-sym entry]
  (register-op-descriptor!
   op-sym
   {:buffer (select-keys entry [:allocates? :in-place-arg :alloc-form :rewrite-fn])
    :effects (select-keys entry [:pure? :mutating? :primitive-mutation?])}))

(defn register-buffer-write!
  "Register how an into-variant op writes to its buffer argument.
   mode is :overwrite (full write, safe to reuse without zeroing)
   or :accumulate (adds to existing content, buffer must be zeroed).
   buf-arg-idx is the 0-based index of the buffer argument."
  [op-sym mode buf-arg-idx]
  (register-op-descriptor! op-sym {:buffer-write {:mode mode :buf-arg-idx buf-arg-idx}}))

(defn get-buffer-write-mode
  "Return the buffer-write descriptor for op-sym, or nil."
  [op-sym]
  (:buffer-write (get-op-descriptor op-sym)))

(defn register-device-rule!
  "Register the device inference facet for an op."
  [op-sym rule-fn]
  (register-op-descriptor! op-sym {:device {:rule rule-fn}}))

(defn register-dim-rule!
  "Register the shape/dimension inference facet for an op."
  [op-sym rule-fn]
  (register-op-descriptor! op-sym {:shape {:dim-rule rule-fn}}))

(defn get-device-rule [op-sym]
  (get-in (get-op-descriptor op-sym) [:device :rule]))

(defn register-c-helper!
  "Register a hand-written C implementation for an op (the :c-helper facet) — used by the
  CPU-C AOT backend for a ^:no-inline op whose optimal lowering is a hardware intrinsic
  (e.g. the int8-MAC seam -> maddubs/dpbusd) rather than its translated scalar body.
  `entry` is {:includes <C #include lines> :gen (fn [c-name] -> full C function string)}.
  The op keeps its portable deftm body (JVM + correctness); the C backend uses this instead."
  [op-sym entry]
  (register-op-descriptor! op-sym {:c-helper entry}))

(defn register-placement!
  "Register the device-placement profile for an op — the MLX-style per-op compute
  tag. `profile` is a device TYPE / compute class — :jvm | :cpu-quant | :gpu — NOT a
  concrete device instance. It names WHICH class of lowering runs the op (JVM bytecode,
  the CPU int8-MAC kernel, or a GPU backend), not where its data lives (data never
  moves; the profile selects the lowering).

  Concrete device SELECTION is orthogonal and lives at the runtime/session layer: a
  :gpu op binds to a specific physical device (:ze:0, :ze:1, :ocl:0, :cuda:0 — there
  can be several, e.g. multiple GPUs) when a session places it. Same op, one profile,
  any concrete device of that class. A descriptor facet (not a separate registry) so
  it composes with buffer/effects/AD."
  [op-sym profile]
  (register-op-descriptor! op-sym {:placement {:profile profile}}))

(defn get-placement
  "The compute-profile (device-type) tag for op-sym (:jvm/:cpu-quant/:gpu), or nil —
  NOT a concrete device id (that is a session-layer binding). Resolves mangled/
  devirtualized names so passes can query with a clean semantic-op symbol."
  [op-sym]
  (when-let [[descriptor _] (resolve-op-descriptor op-sym)]
    (get-in descriptor [:placement :profile])))

(defn get-dim-rule [op-sym]
  (get-in (get-op-descriptor op-sym) [:shape :dim-rule]))

(defn mutating-op?
  "True if op-sym is a mutating operation.
   Checks op-descriptor registry first, then falls back to !-suffix convention.
   Handles mangled names: sgd-step!_m_doubles_double -> checks base name sgd-step!"
  [op-sym]
  (boolean
   (or (when-let [[descriptor _] (resolve-op-descriptor op-sym)]
         (get-in descriptor [:effects :mutating?]))
       (when (symbol? op-sym)
         (let [n (name op-sym)
               idx (.indexOf ^String n "_m_")
               base (if (pos? idx) (subs n 0 idx) n)]
           (.endsWith ^String base "!"))))))

(defn primitive-mutation-op? [op-sym]
  (when-let [[descriptor _] (resolve-op-descriptor op-sym)]
    (get-in descriptor [:effects :primitive-mutation?])))

;; ================================================================
;; Centralized operator classification
;; ================================================================

;; --- Array allocation ---

(def alloc-ops
  "Array allocation ops recognized by the compiler."
  #{'double-array 'float-array 'long-array 'int-array
    'byte-array 'short-array 'char-array 'boolean-array
    'object-array 'make-array
    'clojure.core/double-array 'clojure.core/float-array
    'clojure.core/long-array 'clojure.core/int-array
    'clojure.core/byte-array 'clojure.core/short-array
    'clojure.core/char-array 'clojure.core/boolean-array
    'clojure.core/object-array 'clojure.core/make-array})

(def alloc-sym->array-tag
  "Array constructor symbol → array type tag for type inference.
   Includes both unqualified and clojure.core-qualified forms."
  (let [base {'double-array  'doubles
              'float-array   'floats
              'long-array    'longs
              'int-array     'ints
              'byte-array    'bytes
              'short-array   'shorts
              'char-array    'chars
              'boolean-array 'booleans
              'object-array  'objects}]
    (merge base
           (into {} (map (fn [[k v]] [(symbol "clojure.core" (name k)) v]) base)))))

(def ^:private known-allocator-base-names
  "Base names (before _m_ mangling) of deftm functions that are pure allocators.
   Includes aclone — the hoister fills via arraycopy instead of zero-fill."
  #{"zeros-like" "alloc-like" "similar" "aclone"})

(defn alloc-op?
  "True if sym is an array allocation operation.
   Recognizes standard constructors (double-array etc.) and deftm allocators
   (zeros-like, aclone) including their mangled variants."
  [sym]
  (and (symbol? sym)
       (let [n (name sym)]
         (or (contains? alloc-ops sym)
             (contains? alloc-sym->array-tag sym)
             (contains? alloc-sym->array-tag (symbol n))
             ;; Check base name for known deftm allocators
             (let [base (if-let [idx (clojure.string/index-of n "_m_")]
                          (subs n 0 idx)
                          n)]
               (contains? known-allocator-base-names base))))))

;; --- Primitive casts ---

(def cast-ops
  "Primitive cast symbols and their result types."
  {'long 'long, 'int 'int, 'double 'double, 'float 'float
   'clojure.core/long 'long, 'clojure.core/int 'int
   'clojure.core/double 'double, 'clojure.core/float 'float})

(defn cast-op?
  "True if sym is a primitive cast operation."
  [sym]
  (contains? cast-ops sym))

(defn cast-result-tag
  "Return the result type tag for a cast op, or nil."
  [sym]
  (get cast-ops sym))

;; --- Scalar ops (type-preserving, pure) ---

(def known-scalar-ops
  "Operations known to produce scalar results."
  #{'clojure.core/+ 'clojure.core/- 'clojure.core/* 'clojure.core//
    'clojure.core/quot 'clojure.core/rem 'clojure.core/mod
    'clojure.core/inc 'clojure.core/dec
    'clojure.core/max 'clojure.core/min
    'raster.numeric/+ 'raster.numeric/- 'raster.numeric/* 'raster.numeric//
    'raster.numeric/sqrt 'raster.numeric/pow
    'raster.arrays/aget 'raster.arrays/alength
    'clojure.core/alength
    'Math/sin 'Math/cos 'Math/exp 'Math/log 'Math/sqrt 'Math/pow
    'Math/abs 'Math/tan 'Math/asin 'Math/acos 'Math/atan 'Math/atan2
    'Math/min 'Math/max 'Math/fma 'Math/signum 'Math/floor 'Math/ceil
    'double 'float 'long 'int
    'clojure.core/double 'clojure.core/float 'clojure.core/long 'clojure.core/int})

(defn scalar-op?
  "True if sym is a known scalar-producing operation."
  [sym]
  (contains? known-scalar-ops sym))

;; --- SIMD capability ---

(def simd-unary-ops
  "Unary ops with Vector API method equivalents."
  {'Math/abs        '.abs
   'Math/sqrt       '.sqrt
   'Math/neg        '.neg
   '-               '.neg
   'clojure.core/-  '.neg})

(def simd-lanewise-unary-ops
  "Unary ops using lanewise VectorOperators."
  {'Math/exp  'jdk.incubator.vector.VectorOperators/EXP
   'Math/log  'jdk.incubator.vector.VectorOperators/LOG
   'Math/sin  'jdk.incubator.vector.VectorOperators/SIN
   'Math/cos  'jdk.incubator.vector.VectorOperators/COS
   'Math/tan  'jdk.incubator.vector.VectorOperators/TAN})

(def simd-binary-ops
  "Binary ops with Vector API method equivalents."
  {'+ '.add  '- '.sub  '* '.mul  '/ '.div
   'clojure.core/+ '.add  'clojure.core/- '.sub
   'clojure.core/* '.mul  'clojure.core// '.div
   'Math/max '.max  'Math/min '.min})

(def simd-ternary-ops
  "Ternary ops (fma)."
  {'Math/fma '.fma})

(def simd-lanewise-binary-ops
  "Binary ops using lanewise VectorOperators — base.lanewise(OP, argVec).
   POW is lowered through Intel SVML (x64) / SLEEF (ARM) by the JVM, giving a
   fully vectorized accurate pow. Both raster.numeric/pow and fast-pow map here:
   in vector code SVML POW is both accurate and fast, so the scalar fast-pow
   polynomial isn't needed."
  {'Math/pow  'jdk.incubator.vector.VectorOperators/POW
   'pow       'jdk.incubator.vector.VectorOperators/POW
   'fast-pow  'jdk.incubator.vector.VectorOperators/POW})

(defn simd-capable?
  "True if sym has a SIMD equivalent."
  [sym]
  (or (contains? simd-unary-ops sym)
      (contains? simd-lanewise-unary-ops sym)
      (contains? simd-binary-ops sym)
      (contains? simd-ternary-ops sym)
      (contains? simd-lanewise-binary-ops sym)))

;; --- Array access ---

(def aget-ops
  "Array read operations."
  #{'aget 'clojure.core/aget 'raster.arrays/aget})

(def aset-ops
  "Array write operations."
  #{'aset 'clojure.core/aset 'raster.arrays/aset})

(def alength-ops
  "Array length operations."
  #{'alength 'clojure.core/alength 'raster.arrays/alength})

(def array-mutating-base-names
  "Unqualified short names of array-mutating ops.
   Used by effects.clj to classify ns-publics (which have unqualified keys)."
  #{'aset})

(def void-ops
  "Operations that produce no value (void return type).
   Includes array writes and parallel side-effect primitives.
   Note: atomic-add! returns the old value (not void) — it's excluded here."
  (into aset-ops
        #{'raster.par/collect! 'par/collect!}))

(defn void-op?
  "True if sym is a void (statement-only) operation."
  [sym]
  (contains? void-ops sym))

(defn aget-op?
  "True if sym is an array read operation."
  [sym]
  (contains? aget-ops sym))

(defn aset-op?
  "True if sym is an array write operation."
  [sym]
  (contains? aset-ops sym))

(defn alength-op?
  "True if sym is an array length operation."
  [sym]
  (contains? alength-ops sym))

;; --- Comparison ops (non-differentiable) ---

(def comparison-ops
  "Comparison/boolean operations — zero gradient in AD."
  #{'< '> '<= '>= '== '!= 'not
    'clojure.core/< 'clojure.core/> 'clojure.core/<= 'clojure.core/>=
    'clojure.core/== 'clojure.core/not= 'clojure.core/not
    'clojure.core/zero? 'clojure.core/pos? 'clojure.core/neg?})

(defn comparison-op?
  "True if sym is a comparison/boolean op (non-differentiable)."
  [sym]
  (contains? comparison-ops sym))

;; --- Arithmetic operator classification ---

(def addition-ops
  "All symbol variants recognized as addition."
  #{'+ 'clojure.core/+ 'raster.numeric/+})

(def subtraction-ops
  "All symbol variants recognized as subtraction/negation."
  #{'- 'clojure.core/- 'raster.numeric/-})

(def multiplication-ops
  "All symbol variants recognized as multiplication."
  #{'* 'clojure.core/* 'raster.numeric/*})

(def division-ops
  "All symbol variants recognized as division."
  #{'/ 'clojure.core// 'raster.numeric//})

(def power-ops
  "All symbol variants recognized as exponentiation."
  #{'Math/pow 'raster.numeric/pow})

(defn addition-op? [sym] (contains? addition-ops sym))
(defn subtraction-op? [sym] (contains? subtraction-ops sym))
(defn multiplication-op? [sym] (contains? multiplication-ops sym))
(defn division-op? [sym] (contains? division-ops sym))
(defn power-op? [sym] (contains? power-ops sym))

(defn arithmetic-op?
  "True if sym is any basic arithmetic operation (+, -, *, /)."
  [sym]
  (or (addition-op? sym) (subtraction-op? sym)
      (multiplication-op? sym) (division-op? sym)))

;; --- Increment/decrement ---

(def increment-ops
  "All symbol variants recognized as increment."
  #{'inc 'unchecked-inc 'clojure.core/inc 'clojure.core/unchecked-inc})

(defn increment-op?
  "True if sym is an increment operation."
  [sym]
  (contains? increment-ops sym))

(defn semantic-op
  "The semantic operator of a call form. For a walker-devirtualized
   (.invk impl args...) form, returns the original op from :raster.op/original
   metadata (the sanctioned way to recover meaning — never parse mangled names).
   Otherwise returns (first form). nil if form is not a call/seq."
  [form]
  (when (seq? form)
    (if (= '.invk (first form))
      (:raster.op/original (meta form))
      (first form))))

(defn call-args
  "The semantic arguments of a call form, skipping the impl receiver for
   devirtualized (.invk impl args...) forms. Returns a seq (possibly empty)."
  [form]
  (when (seq? form)
    (if (= '.invk (first form)) (nnext form) (rest form))))

(defn- unwrap-array-arg
  "Unwrap a cast wrapper around an array argument — (double arr) → arr,
   (float arr) → arr — and strip metadata/namespace from a bare symbol so it
   matches binding names. Non-symbol args (rare) pass through unwrapped."
  [arr-sym]
  (let [unwrapped (if (and (seq? arr-sym)
                           (contains? #{'double 'float 'int 'long
                                        'clojure.core/double 'clojure.core/float}
                                      (first arr-sym))
                           (= 2 (count arr-sym)))
                    (second arr-sym)
                    arr-sym)]
    (if (symbol? unwrapped) (symbol (name unwrapped)) unwrapped)))

(defn aget-call?
  "True if `form` is an array-read call — either a bare (aget arr idx ...) or a
   walker-devirtualized (.invk aget-impl arr idx ...). Recovers the semantic op
   via :raster.op/original metadata (semantic-op) so the SOAC/fusion matchers do
   not go blind once raster.arrays/aget is devirtualized to .invk."
  [form]
  (and (seq? form)
       (aget-op? (semantic-op form))
       (>= (count (call-args form)) 2)))

(defn aget-array-sym
  "The array symbol read by an aget call (bare or devirtualized), with cast
   wrappers unwrapped and metadata stripped. nil when `form` is not an aget call."
  [form]
  (when (aget-call? form)
    (unwrap-array-arg (first (call-args form)))))

(defn aset-call?
  "True if `form` is an array-write call — bare (aset arr idx val) or
   walker-devirtualized (.invk aset-impl arr idx val)."
  [form]
  (and (seq? form)
       (aset-op? (semantic-op form))
       (>= (count (call-args form)) 3)))

(defn aset-array-sym
  "The array symbol written by an aset call (bare or devirtualized), with cast
   wrappers unwrapped and metadata stripped. nil when `form` is not an aset call."
  [form]
  (when (aset-call? form)
    (unwrap-array-arg (first (call-args form)))))

(defn- unwrap-int-cast
  "Unwrap an integer cast around an induction-variable reference —
   (long i) / (int i) → i. The walked/AD-prep dialect wraps loop indices in
   long casts ((inc (long i))), so step matching must see through them just
   like idx-matches?/test-index+bound do. Non-cast forms pass through."
  [expr]
  (if (and (seq? expr)
           (contains? #{'long 'int 'clojure.core/long 'clojure.core/int}
                      (first expr))
           (= 2 (count expr)))
    (second expr)
    expr))

(defn affine-step
  "If `expr` is an affine step of the induction variable `idx-sym` — i.e.
   (inc idx), (unchecked-inc idx), or (+ idx c) / (+ c idx) with a constant
   integer c — return the constant stride as a long. Otherwise nil.
   Works on both bare ops and walker-devirtualized (.invk impl ...) forms via
   semantic-op/call-args, so no form rewriting is needed for matching. The
   idx reference may be cast-wrapped ((inc (long i))) — walked-dialect loops
   emit that shape."
  [expr idx-sym]
  (when (seq? expr)
    (let [op   (semantic-op expr)
          args (vec (map unwrap-int-cast (call-args expr)))]
      (cond
        (and (increment-op? op)
             (= 1 (count args))
             (= idx-sym (first args)))
        1

        (and (addition-op? op)
             (= 2 (count args)))
        (let [[a b] args]
          (cond
            (and (= a idx-sym) (integer? b)) (long b)
            (and (= b idx-sym) (integer? a)) (long a)
            :else nil))))))

;; --- Relational comparisons (:comparison facet) ---

(defn register-comparison!
  "Register the relational kind of a binary comparison operator as the
   :comparison facet: {:kind :lt|:le|:gt|:ge|:eq|:ne}. Like register-algebra!,
   this lives in the unified registry — one entry per op, extensible by other
   namespaces — rather than a bespoke set.

   (This is distinct from comparison-op?, which is the broader AD notion of a
   non-differentiable boolean op, including unary predicates like zero?/pos?.)"
  [op-sym kind]
  (register-op-descriptor! op-sym {:comparison {:kind kind}}))

(defn comparison-kind
  "The relational :kind of op-sym (:lt/:le/:gt/:ge/:eq/:ne), or nil."
  [op-sym]
  (:kind (:comparison (get-op-descriptor op-sym))))

(defn less-than-op?
  "True iff op-sym is a strict less-than. A loop guarded by (< i n) iterates
   the contiguous range [0,n), the only bound shape loop-lift currently rewrites
   to a [0,bound) SOAC. Other relational kinds are recognized by the registry
   but loop-lift conservatively bails on them (e.g. <= would need a [0,n+1)
   bound); adding them later is a matcher change, not a new classification set."
  [op-sym]
  (= :lt (comparison-kind op-sym)))

;; Register relational kinds for every surface variant
;; (bare / clojure.core / raster.numeric).
(doseq [[base kind] {'<  :lt  '<=  :le  '>  :gt  '>=  :ge  '==  :eq  'not=  :ne}
        v    [base
              (symbol "clojure.core" (name base))
              (symbol "raster.numeric" (name base))]]
  (register-comparison! v kind))

;; --- Algebraic properties (:algebra facet; loop-lift reduction eligibility) ---

(defn register-algebra!
  "Register the algebraic properties of a binary operator as the :algebra facet:
     {:associative? bool :commutative? bool :identity expr}
   Merged into the unified op-descriptor registry, so adding a new reducible op
   is a single entry — no central set to edit, and other namespaces can declare
   their own ops the same way."
  [op-sym algebra]
  (register-op-descriptor! op-sym {:algebra algebra}))

(defn algebra-facet
  "The :algebra facet for op-sym, or nil. Looks up the exact (qualified) symbol;
   loop-lift queries with descriptor/semantic-op, which yields clean op symbols."
  [op-sym]
  (:algebra (get-op-descriptor op-sym)))

(defn commutative-monoid-op?
  "True iff op-sym is a registered commutative monoid — ASSOCIATIVE *and*
   COMMUTATIVE. This is the precise precondition for lowering a loop/recur
   reduction into a reordered, multi-accumulator SIMD reduce: that reduce both
   regroups (needs associativity) and interleaves operands across lanes (needs
   commutativity). - and / qualify as neither (acc-e = init-Σe is a sum in
   disguise; acc/e a product) and are correctly absent → they bail to a scalar
   loop. Unregistered ops also bail (conservative; never a miscompile).

   Note on floats: +/* are associative only in exact arithmetic; lifting accepts
   the last-ULP reorder, the same global trade par/reduce! already makes (min,
   max, and the integer bitwise monoids are exact)."
  [op-sym]
  (let [a (algebra-facet op-sym)]
    (boolean (and a (:associative? a) (:commutative? a)))))

;; The numeric commutative monoids and the integer bitwise monoids, registered
;; for every surface variant (bare / clojure.core / raster.numeric).
(doseq [[base algebra]
        {'+       {:associative? true :commutative? true :identity 0}
         '*       {:associative? true :commutative? true :identity 1}
         'min     {:associative? true :commutative? true :identity nil}
         'max     {:associative? true :commutative? true :identity nil}
         'bit-and {:associative? true :commutative? true :identity -1}
         'bit-or  {:associative? true :commutative? true :identity 0}
         'bit-xor {:associative? true :commutative? true :identity 0}}
        v    (cond-> [base
                      (symbol "clojure.core" (name base))
                      (symbol "raster.numeric" (name base))]
               (contains? #{'min 'max} base) (conj (symbol "Math" (name base))))]
  (register-algebra! v algebra))

(defn typed-reduce-identity
  "The identity element for a reduction op, TYPED for the element dtype — the
   single source emitters use to seed accumulators (segop-simd, GPU segred).
   Derived from the :algebra facet's abstract identity; min/max (whose abstract
   identity is nil — no identity over unbounded domains) become the dtype's
   ∓Infinity, which IS their identity over IEEE floats. Emits a LITERAL or a
   symbol suitable for splicing into generated code. Throws for ops without a
   registered monoid — an unregistered reduction must fail loud, not seed 0."
  [op-sym dtype]
  (let [base (symbol (name op-sym))                 ; spelling-collapse
        a    (algebra-facet base)]
    (when-not a
      (throw (ex-info (str "no :algebra facet for reduction op " op-sym
                           " — register it (op-descriptor) before reducing with it")
                      {:op op-sym :dtype dtype})))
    (let [float? (= dtype :float)]
      (case base
        (min Math/min) (if float? 'Float/POSITIVE_INFINITY 'Double/POSITIVE_INFINITY)
        (max Math/max) (if float? 'Float/NEGATIVE_INFINITY 'Double/NEGATIVE_INFINITY)
        ;; numeric monoids: type the abstract identity for the dtype
        (let [id (:identity a)]
          (cond
            (nil? id) (throw (ex-info (str "op " op-sym " has no identity") {:op op-sym}))
            (contains? #{:int :long :byte} dtype) id
            float?    (list 'float (double id))
            :else     (double id)))))))

;; --- Result-type inference (:result-type facet) ---

(def ^:private array-tag->element-tag
  "Primitive-array dispatch tag → element scalar tag ('floats → 'float), derived
   from the dtype registry (dtype/dtype-info) — the single dtype source of truth,
   NOT another hardcoded array-tag map. Dtypes without a JVM-primitive scalar
   are absent (:half → 'shorts has :scalar-tag nil)."
  (into {}
        (keep (fn [[_dt {:keys [array-tag scalar-tag]}]]
                (when (and array-tag scalar-tag)
                  [array-tag scalar-tag])))
        dtype/dtype-info))

(def ^:private element-tags
  "The scalar element tags (values of array-tag->element-tag) — accepted as
   passthrough by :element-of-first-arg (the ref may itself be a scalar)."
  (set (vals array-tag->element-tag)))

(defn register-result-type!
  "Register the :result-type facet for an op — the rule deriving the op's result
   TYPE TAG from its argument tags. `rule` is one of
     :element-of-first-arg — element tag of the first arg's array tag
                             ('floats → 'float); an already-scalar first-arg
                             tag passes through ('float → 'float)
     :same-as-first-arg    — the first arg's tag, unchanged
     :first-typed-arg      — the first non-nil arg tag (nil-safe accumulators)
     [:arg n]              — the tag of the (0-based) nth arg
   or a fixed result tag symbol (e.g. 'long)."
  [op-sym rule]
  (register-op-descriptor! op-sym {:result-type {:rule rule}}))

(defn result-type-rule
  "The registered :result-type rule for op-sym, or nil. Exact-symbol lookup:
   callers query with the SEMANTIC op (:raster.op/original / form/effective-op),
   never a mangled impl name."
  [op-sym]
  (get-in (get-op-descriptor op-sym) [:result-type :rule]))

(defn result-tag
  "Result type tag of `op-sym` applied to arguments whose inferred tags are
   `arg-tags` (a seq of dispatch-tag symbols or nils; may be lazy — rules force
   only the prefix they need), per the :result-type facet. nil when no rule is
   registered or the rule yields no tag."
  [op-sym arg-tags]
  (when-let [rule (result-type-rule op-sym)]
    (cond
      (= :element-of-first-arg rule) (let [t (first arg-tags)]
                                       (or (get array-tag->element-tag t)
                                           (when (contains? element-tags t) t)))
      (= :same-as-first-arg rule)    (first arg-tags)
      (= :first-typed-arg rule)      (some identity arg-tags)
      (and (vector? rule)
           (= :arg (first rule)))    (nth arg-tags (second rule) nil)
      ;; a fixed result tag symbol
      (symbol? rule)                 rule)))

;; DELETED (plan D2 assert-then-delete audit, 2026-07-11):
;;   - 'raster.numeric/oftype :element-of-first-arg — oftype-seeded scalars
;;     (softmax neg-inf seeds, attention scales) are now tagged without the
;;     facet: every resident attention/gqa/rms-norm/rope value+grad compile
;;     passes COLD (fresh JVM) with the entry removed, as does the full suite.
;;     (Historically the walker's oftype :element arm needed it; emission-time
;;     tagging + TC stamps cover those sites today.)
;;   - 'raster.ad.reverse/grad-acc :first-typed-arg — grad-acc adjoint bindings
;;     are tagged at their mint site (Π tagged gensym) and the GPU fan-out path
;;     reroutes grad-acc to residual-add before the fixpoint edge; resblk/hfuse/
;;     attn-block cold + ODE/laws canaries + full suite green without it.
;; The fixpoint-edge census throw (pipeline/record-fixpoint-census!) guards the
;; invariant: if either becomes load-bearing again it resurfaces as a loud GPU
;; compile error naming the untagged binding, not silent narrowing.
;; NB raster.dl.nn/residual-add(+backward) :same-as-first-arg SURVIVED the same
;; audit as LOAD-BEARING — see the comment at its registration in raster.dl.nn.

;; NOTE: ops defined in higher layers (raster.dl.*, raster.ad.*) register their own
;; :result-type facet from THEIR namespace (e.g. residual-add in raster.dl.nn), so
;; compiler core does not depend on those namespaces. Only compiler-owned ops belong
;; here.

;; par/reduce returns its accumulator type — arg 1, the init value. The CANONICAL
;; spelling `raster.par/reduce` is typed independently by form-info's
;; :return-type-arg (:par forms → arg 1; ir/form.clj) — the walker types it via its
;; dedicated `:par-reduce` handler and late inference falls through to the form-info
;; arm — so no op-descriptor entry is needed for it. Only the DEFENSIVE bare alias
;; `par/reduce` needs one: form-info's raster.par-namespace matcher (namespace
;; "raster.par…") does not classify the bare "par" namespace, so it has no
;; :return-type-arg of its own.
(register-result-type! 'par/reduce [:arg 1])

;; --- Devirtualization detection ---

(defn undevirtualized-call?
  "True if sym is a qualified call that might be an undevirtualized deftm.
   Returns true for any qualified symbol that:
   - Has no _m_ mangling (already devirtualized calls have it)
   - Is not a known Java/Clojure/special form
   The walker will attempt devirtualization and skip non-deftm calls,
   so false positives just trigger an extra (harmless) rewalk."
  [sym]
  (and (symbol? sym)
       (when-let [ns-str (namespace sym)]
         (and (not (.contains ^String (name sym) "_m_"))
              (not (.startsWith ^String ns-str "clojure."))
              (not (.startsWith ^String ns-str "java."))
              (not= ns-str "Math")))))

