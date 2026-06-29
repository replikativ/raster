(ns raster.compiler.core.op-descriptor
  "Unified compiler operation descriptor registry.

  Descriptors merge multiple compiler facets under one lookup point, e.g.:
    :buffer {:allocates? ... :in-place-arg ... :alloc-form ... :rewrite-fn ...}
    :effects {:pure? ... :mutating? ... :primitive-mutation? ...}
    :device {:rule ...}
    :shape  {:dim-rule ...}

  Compiler passes and op libraries should register and query metadata directly
  through this registry."
  (:require [raster.compiler.support.mangled :as mangled]))

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

(defn affine-step
  "If `expr` is an affine step of the induction variable `idx-sym` — i.e.
   (inc idx), (unchecked-inc idx), or (+ idx c) / (+ c idx) with a constant
   integer c — return the constant stride as a long. Otherwise nil.
   Works on both bare ops and walker-devirtualized (.invk impl ...) forms via
   semantic-op/call-args, so no form rewriting is needed for matching."
  [expr idx-sym]
  (when (seq? expr)
    (let [op   (semantic-op expr)
          args (vec (call-args expr))]
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
        v    [base
              (symbol "clojure.core" (name base))
              (symbol "raster.numeric" (name base))]]
  (register-algebra! v algebra))

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

