(ns raster.compiler.backend.jvm.segop-simd
  "SIMD codegen from SegOp IR.

   Translates SegMap and SegRed records into Java Vector API S-expressions.
   This is the unified backend entry point — all par form analysis has
   already been done by the SegOp lowering pass.

   Responsibilities:
   - Check if lambda is expressible in SIMD (simd-able?)
   - Emit Vector API code with scalar tail
   - Multi-accumulator reduction for SegRed

   Does NOT:
   - Parse par forms (SegOp already has inputs/outputs/scalars/lambda)
   - Decide phase decomposition (SegOp already decided)
   - Collect arrays/scalars (SegOp already has them)"
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.backend.jvm.bytecode :as bc]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.segop :as segop]
            [raster.runtime.hardware :as hw]
            [clojure.set]
            [clojure.walk]))

;; ================================================================
;; Shared SIMD infrastructure (from op-descriptor)
;; ================================================================

(def ^:private simd-unary-ops descriptor/simd-unary-ops)
(def ^:private simd-lanewise-unary-ops descriptor/simd-lanewise-unary-ops)
(def ^:private simd-binary-ops descriptor/simd-binary-ops)
(def ^:private simd-lanewise-binary-ops descriptor/simd-lanewise-binary-ops)
(def ^:private simd-ternary-ops descriptor/simd-ternary-ops)

(def ^:private simd-compare-ops
  "Comparison ops mapped to VectorOperators constants for .compare + .blend."
  {'>  'jdk.incubator.vector.VectorOperators/GT
   '<  'jdk.incubator.vector.VectorOperators/LT
   '>= 'jdk.incubator.vector.VectorOperators/GE
   '<= 'jdk.incubator.vector.VectorOperators/LE
   '== 'jdk.incubator.vector.VectorOperators/EQ})

(def simd-type-info
  {:double {:species-expr  '(jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
            :from-array    'jdk.incubator.vector.DoubleVector/fromArray
            :broadcast     'jdk.incubator.vector.DoubleVector/broadcast
            :cast-fn       'double}
   :float  {:species-expr  '(jdk.incubator.vector.FloatVector/SPECIES_PREFERRED)
            :from-array    'jdk.incubator.vector.FloatVector/fromArray
            :broadcast     'jdk.incubator.vector.FloatVector/broadcast
            :cast-fn       'float}})

(def ^:private reduce-identity-double
  {'+ 0.0, '- 0.0, '* 1.0,
   'Math/max 'Double/NEGATIVE_INFINITY,
   'Math/min 'Double/POSITIVE_INFINITY})

(def ^:private reduce-identity-float
  {'+ '(float 0.0), '- '(float 0.0), '* '(float 1.0),
   'Math/max 'Float/NEGATIVE_INFINITY,
   'Math/min 'Float/POSITIVE_INFINITY})

(defn- reduce-identity
  "Look up the identity element for a reduction op, typed by elem-type."
  [op elem-type]
  (let [table (if (= elem-type :float) reduce-identity-float reduce-identity-double)]
    (or (get table op)
        (throw (ex-info (str "segop-simd: no reduction identity for op `" op
                             "` (elem-type=" elem-type "). Register in reduce-identity-"
                             (name elem-type) ".")
                        {:op op :elem-type elem-type})))))

(defn- normalize-reduce-op
  "clojure.core/+ → +, etc."
  [op]
  (get {'clojure.core/+ '+, 'clojure.core/- '-, 'clojure.core/* '*, 'clojure.core// '/} op op))

;; ================================================================
;; SegOp field accessors
;; ================================================================

(defn seg-idx [segop] (-> segop :space :dims first :name))
(defn seg-bound [segop] (-> segop :space :dims first :bound))

;; ================================================================
;; Normalize .invk forms to canonical ops for SIMD
;; ================================================================

(def ^:private op->canonical
  "Map :raster.op/original metadata symbols from walker devirtualization to canonical SIMD ops.
   The walker attaches {:raster.op/original raster.numeric/+} etc. to (.invk impl ...) forms."
  {'raster.numeric/+   '+,  'raster.numeric/-  '-,
   'raster.numeric/*   '*,  'raster.numeric//  '/,
   'raster.numeric/max 'Math/max, 'raster.numeric/min 'Math/min,
   'raster.numeric/>   '>,  'raster.numeric/<  '<,
   'raster.numeric/>=  '>=, 'raster.numeric/<= '<=,
   'raster.numeric/==  '==,
   'raster.math/exp    'Math/exp,  'raster.math/log   'Math/log,
   'raster.math/sin    'Math/sin,  'raster.math/cos   'Math/cos,
   'raster.math/tan    'Math/tan,
   'raster.math/sqrt   'Math/sqrt, 'raster.math/abs   'Math/abs,
   'raster.numeric/neg 'Math/neg,
   'raster.math/fma    'Math/fma})

(defn- infer-canonical-op
  "Infer canonical SIMD op from :raster.op/original metadata or mangled impl name.
   Uses util/impl->op for general demangling — no hardcoded name patterns."
  [form]
  (or (get op->canonical (:raster.op/original (meta form)))
      ;; Recover op from mangled impl name via general demangler
      (when (and (seq? form) (= '.invk (first form)) (symbol? (second form)))
        (when-let [op (util/impl->op (second form))]
          (get op->canonical op)))))

(defn- normalize-invk
  "Rewrite (.invk impl args...) and qualified raster.*/raster.math/* forms
   to canonical (op args...) for SIMD.
   Uses :op metadata first, falls back to parsing mangled impl names.
   Recursively normalizes sub-expressions."
  [form]
  (cond
    (not (seq? form)) form
    ;; .invk form — try to resolve canonical op
    (= '.invk (first form))
    (let [canonical (infer-canonical-op form)]
      (if canonical
        (with-meta (apply list canonical (map normalize-invk (drop 2 form)))
          (meta form))
        ;; Unknown op — still recurse into args
        (with-meta (apply list '.invk (second form) (map normalize-invk (drop 2 form)))
          (meta form))))
    ;; Direct qualified raster.numeric/* or raster.math/* calls — walker didn't devirtualize
    (and (symbol? (first form)) (get op->canonical (first form)))
    (let [canonical (get op->canonical (first form))]
      (with-meta (apply list canonical (map normalize-invk (rest form)))
        (meta form)))
    ;; let/let* — normalize bindings and body
    (contains? #{'let 'let*} (first form))
    (let [[h bindings & body] form
          norm-bindings (vec (mapcat (fn [[s v]] [s (normalize-invk v)])
                                     (partition 2 bindings)))]
      (with-meta (apply list h norm-bindings (map normalize-invk body))
        (meta form)))
    ;; if — normalize condition and branches
    (= 'if (first form))
    (let [[_ cond then else] form]
      (with-meta (list 'if (normalize-invk cond) (normalize-invk then) (normalize-invk else))
        (meta form)))
    ;; Other seq — normalize all elements
    :else (with-meta (apply list (map normalize-invk form)) (meta form))))

;; ================================================================
;; SIMD-ability check
;; ================================================================

;; --- Integer index/offset/bound/stride arithmetic constructors ----------------
;; SIMD scaffolding (loop bounds, per-lane load offsets, strides, counters) is
;; integer index arithmetic and MUST be emitted as clojure.core primitives
;; (iadd/ladd/...), never bare symbols. These forms are synthesized AFTER the
;; walker runs, so a bare `+` resolves to raster.numeric/+ in the kernel ns
;; (which excludes clojure.core arithmetic); the bytecode emitter then compiles
;; it as a reflective, boxing RT.var -> IFn.invoke call on every iteration. Per
;; CLAUDE.md, integer index arithmetic stays clojure.core.
(defn- ix+ [a b] (list 'clojure.core/+ a b))
(defn- ix* [a b] (list 'clojure.core/* a b))
(defn- ix<= [a b] (list 'clojure.core/<= a b))
(defn- ix< [a b] (list 'clojure.core/< a b))

(defn- expr-free-of?
  "True if expr does not contain sym anywhere in its tree."
  [expr sym]
  (cond
    (= expr sym) false
    (symbol? expr) true
    (number? expr) true
    (seq? expr) (every? #(expr-free-of? % sym) (rest expr))
    :else true))

(defn- extract-affine-base
  "Decompose an additive index expression to find idx-sym and the base offset.
  Returns the base offset expression (everything except idx-sym), or nil if
  idx-sym is not a direct addend. Handles nested + chains.

  Examples (with idx = j):
    j                    → nil  (no base offset)
    (+ (* i cols) j)     → (* i cols)
    (+ j 5)              → 5
    (+ 3 (+ (* i n) j))  → (+ 3 (* i n))"
  [idx-expr idx-sym]
  (when (and (seq? idx-expr) (= '+ (first idx-expr)) (= 3 (count idx-expr)))
    (let [a (second idx-expr)
          b (nth idx-expr 2)]
      (cond
        ;; (+ base j) where base is free of j
        (and (= b idx-sym) (expr-free-of? a idx-sym))
        a
        ;; (+ j base) where base is free of j
        (and (= a idx-sym) (expr-free-of? b idx-sym))
        b
        ;; (+ base (long j)) / (+ (long j) base)
        (and (seq? b) (contains? #{'long 'int} (first b)) (= idx-sym (second b))
             (expr-free-of? a idx-sym))
        a
        (and (seq? a) (contains? #{'long 'int} (first a)) (= idx-sym (second a))
             (expr-free-of? b idx-sym))
        b
        ;; (+ base (+ ... j ...)) — recurse into nested +
        (and (expr-free-of? a idx-sym) (seq? b))
        (when-let [inner-base (extract-affine-base b idx-sym)]
          (ix+ a inner-base))
        ;; (+ (+ ... j ...) base) — recurse into nested +
        (and (expr-free-of? b idx-sym) (seq? a))
        (when-let [inner-base (extract-affine-base a idx-sym)]
          (ix+ inner-base b))
        :else nil))))

(defn aget-form?
  "Check if form is (aget arr idx-expr) where idx-expr is affine in idx-sym.
  Returns [arr-sym, base-offset-expr] on match, nil otherwise.
  base-offset-expr is nil for simple (aget arr j) access."
  [form idx-sym]
  (when (and (seq? form)
             (descriptor/aget-op? (first form))
             (= 3 (count form)))
    (let [arr (second form)
          idx-expr (nth form 2)]
      (cond
        ;; (aget arr j) — direct index
        (= idx-sym idx-expr)
        [arr nil]
        ;; (aget arr (long/int j)) — cast-wrapped
        (and (seq? idx-expr)
             (contains? #{'long 'int 'clojure.core/long 'clojure.core/int} (first idx-expr))
             (= 2 (count idx-expr))
             (= idx-sym (second idx-expr)))
        [arr nil]
        ;; (aget arr (+ base j)) — affine with base offset
        (seq? idx-expr)
        (when-let [base (extract-affine-base idx-expr idx-sym)]
          [arr base])
        :else nil))))

(defn- invk-to-simd-op
  "Map a .invk form's :raster.op/original metadata to a SIMD-recognized op symbol.
  raster.math/exp → Math/exp, raster.numeric/+ → clojure.core/+, etc.
  Returns [arity op-sym] or nil if not SIMD-mappable."
  [invk-form]
  (when-let [original (:raster.op/original (meta invk-form))]
    (let [ns-str (namespace original)
          op-name (name original)]
      (cond
        ;; raster.math/* → Math/*
        (= ns-str "raster.math")
        (let [math-sym (symbol "Math" op-name)]
          (cond
            (contains? simd-unary-ops math-sym) [1 math-sym]
            (contains? simd-lanewise-unary-ops math-sym) [1 math-sym]
            (contains? simd-binary-ops math-sym) [2 math-sym]
            (contains? simd-lanewise-binary-ops math-sym) [2 math-sym]
            (contains? simd-ternary-ops math-sym) [3 math-sym]))
        ;; raster.numeric/* → try both clojure.core/* and bare *
        (= ns-str "raster.numeric")
        (let [bare-sym (symbol op-name)
              core-sym (symbol "clojure.core" op-name)]
          (cond
            (contains? simd-unary-ops bare-sym) [1 bare-sym]
            (contains? simd-unary-ops core-sym) [1 core-sym]
            (contains? simd-binary-ops bare-sym) [2 bare-sym]
            (contains? simd-binary-ops core-sym) [2 core-sym]
            (contains? simd-lanewise-binary-ops bare-sym) [2 bare-sym]
            (contains? simd-ternary-ops bare-sym) [3 bare-sym]
            (contains? simd-ternary-ops core-sym) [3 core-sym]))
        :else nil))))

(defn simd-able?
  "Check if an expression can be fully compiled to SIMD ops."
  [expr idx-sym]
  (cond
    (number? expr) true
    (symbol? expr) true
    (some? (aget-form? expr idx-sym)) true
    ;; Scalar array access: (aget arr non-idx-expr) — loop-invariant, broadcast
    (and (seq? expr) (descriptor/aget-op? (first expr)) (= 3 (count expr))
         (expr-free-of? (nth expr 2) idx-sym))
    true
    (and (seq? expr)
         (contains? #{'double 'float 'long 'int} (first expr))
         (= 2 (count expr)))
    (simd-able? (second expr) idx-sym)
    (and (seq? expr) (= 2 (count expr))
         (contains? simd-unary-ops (first expr)))
    (simd-able? (second expr) idx-sym)
    (and (seq? expr) (= 2 (count expr))
         (contains? simd-lanewise-unary-ops (first expr)))
    (simd-able? (second expr) idx-sym)
    (and (seq? expr) (= 3 (count expr))
         (contains? simd-binary-ops (first expr)))
    (and (simd-able? (nth expr 1) idx-sym)
         (simd-able? (nth expr 2) idx-sym))
    (and (seq? expr) (= 3 (count expr))
         (contains? simd-lanewise-binary-ops (first expr)))
    (and (simd-able? (nth expr 1) idx-sym)
         (simd-able? (nth expr 2) idx-sym))
    (and (seq? expr) (= 4 (count expr))
         (contains? simd-ternary-ops (first expr)))
    (and (simd-able? (nth expr 1) idx-sym)
         (simd-able? (nth expr 2) idx-sym)
         (simd-able? (nth expr 3) idx-sym))
    (and (seq? expr) (> (count expr) 3)
         (contains? #{'+ '*} (first expr)))
    (every? #(simd-able? % idx-sym) (rest expr))
    (and (seq? expr) (contains? #{'let 'let*} (first expr)))
    (let [[_ bindings & body] expr
          pairs (partition 2 bindings)]
      (and (every? (fn [[_ v]] (simd-able? v idx-sym)) pairs)
           (every? #(simd-able? % idx-sym) body)))
    ;; if with comparison → vector compare + blend
    (and (seq? expr) (= 'if (first expr)) (= 4 (count expr)))
    (let [cond-expr (nth expr 1)]
      (and (seq? cond-expr)
           (= 3 (count cond-expr))
           (contains? simd-compare-ops (first cond-expr))
           (simd-able? (nth cond-expr 1) idx-sym)
           (simd-able? (nth cond-expr 2) idx-sym)
           (simd-able? (nth expr 2) idx-sym)
           (simd-able? (nth expr 3) idx-sym)))
    ;; .invk — devirtualized method calls: check :raster.op/original metadata
    (and (seq? expr) (= '.invk (first expr)))
    (when-let [[arity op-sym] (invk-to-simd-op expr)]
      (let [args (drop 2 expr)]
        (case (int arity)
          1 (and (= 1 (count args)) (simd-able? (first args) idx-sym))
          2 (and (= 2 (count args)) (every? #(simd-able? % idx-sym) args))
          3 (and (= 3 (count args)) (every? #(simd-able? % idx-sym) args))
          false)))
    :else false))

;; ================================================================
;; Body analysis helpers (shared with par_simd stencil/fusion)
;; ================================================================

(defn collect-arrays
  "Collect array symbols and their offsets from aget patterns in body.
  Returns {arr-sym → offset-expr} where offset-expr is nil for simple (aget arr j)."
  [body-expr idx-sym]
  (cond
    (some? (aget-form? body-expr idx-sym))
    (let [[arr offset] (aget-form? body-expr idx-sym)]
      {arr offset})
    (and (seq? body-expr) (contains? #{'let 'let*} (first body-expr)))
    (let [[_ bindings & body] body-expr
          pairs (partition 2 bindings)]
      (apply merge {}
             (concat (map (fn [[_ v]] (collect-arrays v idx-sym)) pairs)
                     (map #(collect-arrays % idx-sym) body))))
    (seq? body-expr)
    (apply merge {} (map #(collect-arrays % idx-sym) (rest body-expr)))
    :else {}))

(defn collect-load-sites
  "Distinct SIMD load sites in body, in first-seen order: a vector of
  [arr-sym base] pairs where base is the affine base offset (nil for a direct
  (aget arr j)). Unlike collect-arrays (keyed by array symbol, last-base-wins),
  this distinguishes the SAME array read at DIFFERENT affine bases — e.g.
  (aget data (+ ab d)) and (aget data (+ bb d)) yield two sites [data ab] and
  [data bb], so each becomes its own vector load. Keying loads by site (not by
  array symbol) is what makes row-a-minus-row-b style kernels vectorize
  correctly instead of collapsing to (.sub v v) = 0."
  [body-expr idx-sym]
  (letfn [(unwrap [arr]
            (if (and (seq? arr)
                     (contains? #{'double 'float 'int 'long
                                  'clojure.core/double 'clojure.core/float} (first arr)))
              (second arr) arr))
          (go [e]
              (cond
                (some? (aget-form? e idx-sym))
                (let [[arr base] (aget-form? e idx-sym)
                      a (unwrap arr)]
                  [[(if (symbol? a) (symbol (name a)) a) base]])
                (and (seq? e) (contains? #{'let 'let*} (first e)))
                (let [[_ bindings & body] e]
                  (concat (mapcat (fn [[_ v]] (go v)) (partition 2 bindings))
                          (mapcat go body)))
                (seq? e) (mapcat go (rest e))
                :else nil))]
    (vec (distinct (go body-expr)))))

(defn- collect-all-aget-arrays
  "Collect ALL array symbols from aget patterns in body, regardless of index.
  Returns a set of symbols. Used to prevent outer-loop-indexed arrays from being
  misclassified as scalars (they are handled in-place by emit-simd).
  Handles both (aget arr expr) and (aget (cast arr) expr) forms."
  [body-expr]
  (cond
    (and (seq? body-expr) (descriptor/aget-op? (first body-expr)) (= 3 (count body-expr)))
    (let [arr-arg (second body-expr)]
      (cond
        (symbol? arr-arg) #{arr-arg}
        ;; (aget (float arr) expr) — cast-wrapped
        (and (seq? arr-arg)
             (contains? #{'double 'float 'int 'long
                          'clojure.core/double 'clojure.core/float} (first arr-arg))
             (symbol? (second arr-arg)))
        #{(second arr-arg)}
        :else #{}))
    (and (seq? body-expr) (contains? #{'let 'let*} (first body-expr)))
    (let [[_ bindings & body] body-expr
          pairs (partition 2 bindings)]
      (apply clojure.set/union #{}
             (concat (map (fn [[_ v]] (collect-all-aget-arrays v)) pairs)
                     (map collect-all-aget-arrays body))))
    (seq? body-expr)
    (apply clojure.set/union #{} (map collect-all-aget-arrays (rest body-expr)))
    :else #{}))

(defn collect-free-syms
  "Collect free scalar symbols from body (non-array, non-index, non-let-bound)."
  [body-expr idx-sym]
  (cond
    (and (symbol? body-expr) (not= body-expr idx-sym)) #{body-expr}
    (and (seq? body-expr) (contains? #{'let 'let*} (first body-expr)))
    (let [[_ bindings & body] body-expr
          pairs (partition 2 bindings)
          bound-syms (set (map first pairs))
          binding-frees (apply clojure.set/union #{}
                               (map (fn [[_ v]] (collect-free-syms v idx-sym)) pairs))
          body-frees (apply clojure.set/union #{}
                            (map #(collect-free-syms % idx-sym) body))]
      (clojure.set/difference (clojure.set/union binding-frees body-frees) bound-syms))
    (seq? body-expr)
    (if (some? (aget-form? body-expr idx-sym))
      ;; aget form — collect free syms from the offset expression (if any)
      (let [[_arr offset] (aget-form? body-expr idx-sym)]
        (if offset
          (collect-free-syms offset idx-sym)
          #{}))
      (apply clojure.set/union #{} (map #(collect-free-syms % idx-sym) (rest body-expr))))
    :else #{}))

(defn collect-value-free-syms
  "Like collect-free-syms but only symbols used in VALUE position — does NOT
  descend into an affine aget's index/offset expression. Used to avoid
  broadcasting integer index bases (e.g. ab, bb in (aget data (+ ab d))) into
  dead scalar vectors: those symbols are consumed as primitive offsets by the
  load, never as vector values."
  [body-expr idx-sym]
  (cond
    (and (symbol? body-expr) (not= body-expr idx-sym)) #{body-expr}
    (and (seq? body-expr) (contains? #{'let 'let*} (first body-expr)))
    (let [[_ bindings & body] body-expr
          pairs (partition 2 bindings)
          bound-syms (set (map first pairs))
          binding-frees (apply clojure.set/union #{}
                               (map (fn [[_ v]] (collect-value-free-syms v idx-sym)) pairs))
          body-frees (apply clojure.set/union #{}
                            (map #(collect-value-free-syms % idx-sym) body))]
      (clojure.set/difference (clojure.set/union binding-frees body-frees) bound-syms))
    (seq? body-expr)
    (if (some? (aget-form? body-expr idx-sym))
      #{}   ;; affine aget: the offset is an index, not a value — skip it
      (apply clojure.set/union #{} (map #(collect-value-free-syms % idx-sym) (rest body-expr))))
    :else #{}))

(defn collect-free-syms-scoped
  "Collect free symbols from an S-expression tree, respecting bindings.
   For JIT isolation (wrap-in-fn). Delegates to util/free-syms."
  ([form] (util/free-syms form))
  ([form bound] (util/free-syms form bound)))

;; ================================================================
;; SIMD body emission (expr → Vector API S-expr)
;; ================================================================

(defn emit-simd
  "Compile an S-expression to Vector API operations.
   vec-env: {symbol → vector-sym} for arrays/scalars.
   All emitted list forms are tagged with the vector class so the bytecode
   compiler can resolve method calls when they appear as receivers."
  [expr idx-sym vec-env species-sym elem-type]
  (let [{:keys [broadcast cast-fn]} (get simd-type-info elem-type)
        vec-tag (case elem-type
                  :double 'jdk.incubator.vector.DoubleVector
                  :float  'jdk.incubator.vector.FloatVector)
        tag-result (fn [form]
                     (if (seq? form)
                       (with-meta (apply list form) {:tag vec-tag})
                       form))
        recur-fn #(tag-result (emit-simd % idx-sym vec-env species-sym elem-type))]
    (cond
      (number? expr)
      (list broadcast species-sym (list cast-fn expr))

      (symbol? expr)
      (or (get vec-env (symbol (name expr)))
          (get vec-env expr)
          (list broadcast species-sym (list cast-fn expr)))

      (some? (aget-form? expr idx-sym))
      (let [[arr-raw base] (aget-form? expr idx-sym)
            arr (if (and (seq? arr-raw)
                         (contains? #{'double 'float 'int 'long
                                      'clojure.core/double 'clojure.core/float} (first arr-raw)))
                  (second arr-raw) arr-raw)
            arr-name (if (symbol? arr) (symbol (name arr)) arr)]
        ;; Resolve to the vector for THIS load site [arr base]; a single array read
        ;; at multiple bases has one vector per site (see collect-load-sites). Fall
        ;; back to array-symbol keys only for legacy single-load callers.
        (or (get vec-env [arr-name base])
            (get vec-env arr-name)
            (get vec-env arr)))

      ;; Scalar array access: (aget arr non-idx-expr) — broadcast as scalar
      (and (seq? expr) (descriptor/aget-op? (first expr)) (= 3 (count expr))
           (expr-free-of? (nth expr 2) idx-sym))
      (list broadcast species-sym (list cast-fn expr))

      (and (seq? expr) (contains? #{'double 'float 'long 'int} (first expr))
           (= 2 (count expr)))
      (recur-fn (second expr))

      (and (seq? expr) (= 2 (count expr)) (contains? simd-unary-ops (first expr)))
      (list (simd-unary-ops (first expr)) (recur-fn (second expr)))

      (and (seq? expr) (= 2 (count expr)) (contains? simd-lanewise-unary-ops (first expr)))
      (list '.lanewise (recur-fn (second expr)) (simd-lanewise-unary-ops (first expr)))

      (and (seq? expr) (= 3 (count expr)) (contains? simd-binary-ops (first expr)))
      (list (simd-binary-ops (first expr)) (recur-fn (nth expr 1)) (recur-fn (nth expr 2)))

      ;; binary lanewise (pow via SVML POW): base.lanewise(OP, argVec)
      (and (seq? expr) (= 3 (count expr)) (contains? simd-lanewise-binary-ops (first expr)))
      (list '.lanewise (recur-fn (nth expr 1)) (simd-lanewise-binary-ops (first expr))
            (recur-fn (nth expr 2)))

      (and (seq? expr) (= 4 (count expr)) (contains? simd-ternary-ops (first expr)))
      (list (simd-ternary-ops (first expr))
            (recur-fn (nth expr 1)) (recur-fn (nth expr 2)) (recur-fn (nth expr 3)))

      (and (seq? expr) (> (count expr) 3) (contains? #{'+ '*} (first expr)))
      (reduce (fn [a b] (list (simd-binary-ops (first expr)) a b))
              (map recur-fn (rest expr)))

      (and (seq? expr) (contains? #{'let 'let*} (first expr)))
      (let [[_ bindings & body] expr
            pairs (partition 2 bindings)
            [ext-env let-binds]
            (reduce
             (fn [[env acc] [sym val-expr]]
               (let [vec-tmp (vary-meta (gensym (str "vlet_" (name sym) "_"))
                                        assoc :tag vec-tag)
                     simd-val (emit-simd val-expr idx-sym env species-sym elem-type)]
                 (if simd-val
                   [(assoc env (symbol (name sym)) vec-tmp)
                    (conj acc [vec-tmp simd-val])]
                   (reduced [nil nil]))))
             [vec-env []] pairs)]
        (when ext-env
          (let [simd-body (emit-simd (last body) idx-sym ext-env species-sym elem-type)]
            (when simd-body
              (if (seq let-binds)
                (list 'let* (vec (mapcat identity let-binds)) simd-body)
                simd-body)))))

      ;; if with comparison → vector compare + blend
      ;; (if (> a b) then else) → else_vec.blend(then_vec, a_vec.compare(GT, b_vec))
      (and (seq? expr) (= 'if (first expr)) (= 4 (count expr))
           (let [c (nth expr 1)] (and (seq? c) (contains? simd-compare-ops (first c)))))
      (let [[_ cond-expr then-expr else-expr] expr
            cmp-op (simd-compare-ops (first cond-expr))
            a-vec (recur-fn (nth cond-expr 1))
            b-vec (recur-fn (nth cond-expr 2))
            mask-sym (vary-meta (gensym "mask_") assoc :tag 'jdk.incubator.vector.VectorMask)
            then-vec (recur-fn then-expr)
            else-vec (recur-fn else-expr)]
        (tag-result
         (list 'let* [mask-sym (list '.compare a-vec cmp-op b-vec)]
               (list '.blend else-vec then-vec mask-sym))))

      ;; .invk — devirtualized method calls: resolve to SIMD ops
      (and (seq? expr) (= '.invk (first expr)))
      (when-let [[arity op-sym] (invk-to-simd-op expr)]
        (let [args (vec (drop 2 expr))]  ;; skip .invk impl-sym
          (case (int arity)
            1 (cond
                (contains? simd-unary-ops op-sym)
                (list (simd-unary-ops op-sym) (recur-fn (first args)))
                (contains? simd-lanewise-unary-ops op-sym)
                (list '.lanewise (recur-fn (first args)) (simd-lanewise-unary-ops op-sym)))
            2 (cond
                (contains? simd-binary-ops op-sym)
                (list (simd-binary-ops op-sym)
                      (recur-fn (first args)) (recur-fn (second args)))
                (contains? simd-lanewise-binary-ops op-sym)
                (list '.lanewise (recur-fn (first args))
                      (simd-lanewise-binary-ops op-sym) (recur-fn (second args))))
            3 (when (contains? simd-ternary-ops op-sym)
                (list (simd-ternary-ops op-sym)
                      (recur-fn (first args))
                      (recur-fn (second args))
                      (recur-fn (nth args 2))))
            nil)))

      :else nil)))

;; ================================================================
;; Hardware queries
;; ================================================================

(defn- effective-n-accumulators
  ([] (effective-n-accumulators :double))
  ([elem-type]
   (try (hw/init!) (let [lanes (int (hw/simd-lanes :cpu:0 (or elem-type :double)))]
                     (if (>= lanes 8) 8 4))
        (catch Exception _ 4))))

;; ================================================================
;; SegMap → SIMD
;; ================================================================

(defn compile-segmap
  "Compile a SegMap to SIMD S-expression.
   out-sym: the output array symbol (from the par form context, not from SegOp).
   cast: cast function symbol for aset (from par form context).
   store-offset: optional offset expression for output stores (for offset par/map!).
     When provided, stores write at out[offset+j] instead of out[j].
   Returns SIMD S-expression or nil if body can't be SIMDified."
  [segmap out-sym cast & {:keys [store-offset]}]
  (let [idx (seg-idx segmap)
        bound (seg-bound segmap)
        raw-body (:lambda segmap)
        ;; Desugar .invk for the scalar tail (bytecode compiler needs standard calls)
        desugared-body (bc/desugar-invk raw-body)
        body (clojure.walk/postwalk
              (fn [x] (if (and (seq? x) (not (list? x))) (apply list x) x))
              (normalize-invk raw-body))
        out out-sym
        elem-type (or (:dtype segmap)
                      (when (= cast 'float) :float)
                      (throw (ex-info "SegMap missing :dtype — type metadata lost in pipeline"
                                      {:segmap-sym out-sym :cast cast})))
        cast-fn cast]
    (when (and (simd-able? body idx)
               ;; Guard: reject if the scalar tail body contains undevirtualized
               ;; dispatch calls — these generate invalid bytecode in SIMD helpers.
               ;; Better to fall back to the plain scalar expansion.
               (not (some (fn check-undev [f]
                            (and (seq? f) (symbol? (first f))
                                 (let [ns-str (namespace (first f))]
                                   (and ns-str (.startsWith ^String ns-str "raster.")
                                        (not= ns-str "raster.par")
                                        (not (.contains ^String (name (first f)) "_m_"))))))
                          (tree-seq seq? seq desugared-body))))
      (let [{:keys [species-expr from-array broadcast] cf :cast-fn} (get simd-type-info elem-type)
            species-sym (vary-meta (gensym "sp__") assoc :tag 'jdk.incubator.vector.VectorSpecies)
            lanes-sym (gensym "ln__")
            n-sym (gensym "n__")
            j-sym (gensym "j__")
            ;; Collect per-array offsets from body aget patterns (only loop-indexed arrays)
            arr-offsets (collect-arrays body idx)
            ;; Build vec-env from SegOp pre-computed sets
            ;; Only arrays in arr-offsets are SIMD-loaded; others are scalar-accessed
            ;; (e.g., (aget W i) in an outer loop) and go through scalar broadcast
            all-inputs (set (filter #(and (symbol? %) (not= '.invk %)) (:inputs segmap)))
            arr-syms (set (filter #(contains? arr-offsets %) all-inputs))
            ;; Arrays accessed by non-SIMD-idx (outer loop vars) must NOT be
            ;; broadcast as scalars — emit-simd handles them in-place via the
            ;; "Scalar array access" branch: (broadcast species (cast (aget arr expr)))
            outer-arr-syms (clojure.set/difference (collect-all-aget-arrays body) arr-syms)
            ;; keep only scalars used as VALUES (drop dead integer index bases)
            scalar-syms (clojure.set/intersection
                         (into (set (filter #(and (symbol? %) (not= '.invk %)) (:scalars segmap)))
                               (clojure.set/difference all-inputs arr-syms outer-arr-syms))
                         (collect-value-free-syms body idx))
            ;; Type tag for vector symbols — enables bytecode compiler to resolve methods
            vec-tag (case elem-type
                      :double 'jdk.incubator.vector.DoubleVector
                      :float  'jdk.incubator.vector.FloatVector)
            tag-vec (fn [sym] (vary-meta sym assoc :tag vec-tag))
            ;; Distinct load sites [arr base] — one vector load per site, so a single
            ;; array read at two bases does NOT collapse to (.op v v) (see collect-load-sites).
            load-sites (filterv (fn [[a _]] (contains? arr-syms a))
                                (collect-load-sites body idx))
            arr-vec-syms (into {} (map (fn [[arr _base :as site]]
                                         [site (tag-vec (gensym (str "v_" (name arr) "_")))])
                                       load-sites))
            scalar-vec-syms (into {} (map (fn [s] [(symbol (name s)) (tag-vec (gensym (str "sv_" (name s) "_")))])
                                          scalar-syms))
            vec-env (merge arr-vec-syms scalar-vec-syms)
            simd-result-sym (tag-vec (gensym "vr__"))
            simd-body-expr (let [e (emit-simd body idx vec-env species-sym elem-type)]
                             (if (seq? e) (with-meta (apply list e) {:tag vec-tag}) e))
            ;; Build per-site offset loads: (fromArray species arr (+ base j))
            arr-loads (vec (mapcat (fn [[[arr base] v]]
                                     (let [offset (if base (ix+ base j-sym) j-sym)]
                                       [v (list from-array species-sym arr offset)]))
                                   arr-vec-syms))
            scalar-bcasts (vec (mapcat (fn [[s sv]] [sv (list broadcast species-sym (list cf s))])
                                       scalar-vec-syms))
            ;; Store index: j or (+ offset j) for offset par/map!
            store-idx (fn [j] (if store-offset (ix+ store-offset j) j))
            ;; Scalar tail — uses desugared body (.invk → mangled fn calls)
            ;; so the bytecode compiler can handle it
            scalar-body (clojure.walk/postwalk (fn [f] (if (= f idx) j-sym f)) desugared-body)
            scalar-aset (if cast-fn
                          (list 'clojure.core/aset out (store-idx j-sym) (list cast-fn scalar-body))
                          (list 'clojure.core/aset out (store-idx j-sym) scalar-body))
            ;; SIMD loop + tail
            simd-loop
            (list 'loop* [j-sym '(int 0)]
                  (list 'if (ix<= (ix+ j-sym lanes-sym) n-sym)
                        (list 'let* (vec (concat arr-loads [simd-result-sym simd-body-expr]))
                              (list '.intoArray simd-result-sym out (store-idx j-sym))
                              (list 'recur (ix+ j-sym lanes-sym)))
                        (list 'loop* [j-sym j-sym]
                              (list 'if (ix< j-sym n-sym)
                                    (list 'do scalar-aset (list 'recur (list 'inc j-sym)))
                                    out))))]
        (when simd-body-expr
          (list 'let* [species-sym species-expr
                       lanes-sym (list '.length species-sym)
                       n-sym (list 'int bound)]
                (if (seq scalar-bcasts)
                  (list 'let* (vec scalar-bcasts) simd-loop)
                  simd-loop)))))))

;; ================================================================
;; par/gather → SIMD hardware vgather
;; ================================================================

(defn compile-par-gather
  "Compile (raster.par/gather out src index n) to a SIMD vgather loop:
     out[j..j+L) = src[index[j..j+L)]
   via DoubleVector.fromArray(species, src, 0, index, j) (hardware vgather),
   stored contiguously into out, with a scalar tail. Flat (non-strided) form
   only; returns SIMD S-expr or nil. elem-type ∈ {:double :float}."
  [out src index n elem-type]
  (let [{:keys [species-expr from-array]} (get simd-type-info elem-type)
        vec-tag     (case elem-type :double 'jdk.incubator.vector.DoubleVector
                          :float 'jdk.incubator.vector.FloatVector)
        species-sym (vary-meta (gensym "sp__") assoc :tag 'jdk.incubator.vector.VectorSpecies)
        lanes-sym   (vary-meta (gensym "lanes__") assoc :tag 'long)
        n-sym       (vary-meta (gensym "n__") assoc :tag 'long)
        j-sym       (vary-meta (gensym "j__") assoc :tag 'long)
        gv-sym      (vary-meta (gensym "gv__") assoc :tag vec-tag)]
    (when species-expr
      (list 'let* [species-sym species-expr
                   lanes-sym (list '.length species-sym)
                   n-sym (list 'int n)]
            (list 'loop* [j-sym '(int 0)]
                  (list 'if (ix<= (ix+ j-sym lanes-sym) n-sym)
                        ;; vgather: src[index[j..j+L)] → contiguous out[j..j+L)
                        (list 'let* [gv-sym (list from-array species-sym src 0 index j-sym)]
                              (list '.intoArray gv-sym out j-sym)
                              (list 'recur (ix+ j-sym lanes-sym)))
                        ;; scalar tail
                        (list 'loop* [j-sym j-sym]
                              (list 'if (ix< j-sym n-sym)
                                    (list 'do
                                          (list 'clojure.core/aset out j-sym
                                                (list 'clojure.core/aget src
                                                      (list 'clojure.core/aget index j-sym)))
                                          (list 'recur (list 'inc j-sym)))
                                    out))))))))

;; ================================================================
;; SegRed → SIMD (multi-accumulator)
;; ================================================================

(defn compile-segred
  "Compile a SegRed to SIMD S-expression with multi-accumulator unrolling.
   Returns SIMD S-expression or nil if unsupported."
  [segred]
  (let [idx (seg-idx segred)
        bound (seg-bound segred)
        {:keys [acc init lambda]} (:reduce-op segred)
        lambda (when lambda (normalize-invk lambda))
        elem-type (or (:dtype segred)
                      (throw (ex-info "SegRed missing :dtype — type metadata lost in pipeline"
                                      {:acc acc :init init})))
        _ (when (System/getProperty "raster.debug.segred-dtype")
            (binding [*out* *err*]
              (println "[segred] dtype=" elem-type
                       " acc=" acc " init=" init
                       " inputs=" (vec (:inputs segred)))))
        {:keys [species-expr from-array broadcast] cf :cast-fn} (get simd-type-info elem-type)]
    (when (and acc init idx bound lambda)
      ;; Unwrap let* in lambda
      (let [[let-bindings inner-body]
            (if (and (seq? lambda) (contains? #{'let* 'let} (first lambda)))
              (let [[_ binds & bdy] lambda]
                [(vec (partition 2 binds)) (last bdy)])
              [[] lambda])]
        (when (and (seq? inner-body) (>= (count inner-body) 3))
          (let [raw-op (first inner-body)
                [op method fma?]
                (cond
                  (contains? simd-binary-ops raw-op)
                  [(normalize-reduce-op raw-op) (simd-binary-ops raw-op) false]
                  (and (= 'Math/fma raw-op) (= 4 (count inner-body))
                       (let [a3 (nth inner-body 3)]
                         (or (= a3 acc)
                             (and (seq? a3) (= 'double (first a3)) (= acc (second a3))))))
                  ['+ '.add true]
                  :else [nil nil false])]
            (when op
              (let [[acc-pos elem-raw]
                    (cond
                      fma? [:right (list '* (nth inner-body 1) (nth inner-body 2))]
                      (= (nth inner-body 1) acc) [:left (nth inner-body 2)]
                      (= (nth inner-body 2) acc) [:right (nth inner-body 1)]
                      (and (seq? (nth inner-body 1)) (= 'double (first (nth inner-body 1)))
                           (= acc (second (nth inner-body 1))))
                      [:left (nth inner-body 2)]
                      :else [nil nil])
                    elem-expr (if (and acc-pos (seq let-bindings))
                                (list 'let* (vec (mapcat identity let-bindings)) elem-raw)
                                elem-raw)]
                (when (and acc-pos (simd-able? elem-expr idx))
                  (let [all-inputs (set (filter #(and (symbol? %) (not= '.invk %)) (:inputs segred)))
                        ;; Collect per-array base offsets from aget patterns
                        arr-offsets (collect-arrays elem-expr idx)
                        ;; Only SIMD-indexed arrays get vector loads; outer-loop arrays
                        ;; are handled in-place by emit-simd's scalar array access branch
                        outer-arr-syms (clojure.set/difference (collect-all-aget-arrays elem-expr)
                                                               (set (keys arr-offsets)))
                        arr-syms (clojure.set/difference all-inputs outer-arr-syms)
                        ;; Distinct load sites [arr base] — one vector load per site, so a
                        ;; single array read at two bases (row-a vs row-b) does NOT collapse.
                        load-sites (filterv (fn [[a _]] (contains? arr-syms a))
                                            (collect-load-sites elem-expr idx))
                        ;; keep only scalars used as VALUES; integer index bases
                        ;; (ab, bb) appear solely in aget offsets and must not be
                        ;; broadcast into dead vectors.
                        scalar-syms (clojure.set/intersection
                                     (clojure.set/difference
                                      (set (filter #(and (symbol? %) (not= '.invk %)) (:scalars segred)))
                                      outer-arr-syms)
                                     (collect-value-free-syms elem-expr idx))
                        vec-tag (case elem-type
                                  :double 'jdk.incubator.vector.DoubleVector
                                  :float  'jdk.incubator.vector.FloatVector)
                        tag-vec (fn [sym] (vary-meta sym assoc :tag vec-tag))
                        species-sym (vary-meta (gensym "sp__") assoc :tag 'jdk.incubator.vector.VectorSpecies)
                        lanes-sym (gensym "ln__")
                        stride-sym (gensym "st__")
                        n-sym (gensym "n__")
                        j-sym (gensym "j__")
                        nacc (effective-n-accumulators elem-type)
                        vacc-syms (vec (for [k (range nacc)] (tag-vec (gensym (str "va" k "__")))))
                        ;; one vec sym per [site, accumulator]; key by site [arr base]
                        arr-groups (vec (for [k (range nacc)]
                                          (into {} (map (fn [[arr base :as site]]
                                                          [site (tag-vec (gensym (str "v" k "_" (name arr) "_")))])
                                                        load-sites))))
                        scalar-vec-syms (into {} (map (fn [s] [(symbol (name s)) (tag-vec (gensym (str "sv_" (name s) "_")))])
                                                      scalar-syms))
                        vec-envs (vec (for [k (range nacc)]
                                        (merge (nth arr-groups k) scalar-vec-syms)))
                        simd-elems (vec (for [k (range nacc)]
                                          (let [e (emit-simd elem-expr idx (nth vec-envs k) species-sym elem-type)]
                                            (if (seq? e) (with-meta (apply list e) {:tag vec-tag}) e))))
                        ;; FMA fusion: acc + a[i]*b[i] → DoubleVector.fma(b,c)=this*b+c → one
                        ;; vfmadd231pd. Crucially we inline the fromArray loads INTO the .fma so
                        ;; C2 folds the memory load into the FMA instruction (vfmadd ymm,ymm,[mem])
                        ;; and unrolls deeply — matching hand-tuned Java. Pre-binding loads to vec
                        ;; syms yields register-only FMA + separate vmovupd (slower, shallow unroll).
                        fma-f1 (when (and (= op '+) (contains? #{:double :float} elem-type)
                                          (seq? elem-expr) (= 3 (count elem-expr))
                                          (contains? #{'* 'clojure.core/* 'raster.numeric/*} (first elem-expr)))
                                 (nth elem-expr 1))
                        fma-f2 (when fma-f1 (nth elem-expr 2))
                        fuse-fma? (and fma-f1 (aget-form? fma-f1 idx) (aget-form? fma-f2 idx))
                        fma-load (fn [factor k]   ;; inline (fromArray species arr (base + j + k*lanes))
                                   (let [[arr0 base] (aget-form? factor idx)
                                         arr (if (and (seq? arr0)
                                                      (contains? #{'double 'float 'clojure.core/double 'clojure.core/float}
                                                                 (first arr0)))
                                               (second arr0) arr0)
                                         k-offset (if (zero? k) j-sym (ix+ j-sym (ix* (int k) lanes-sym)))
                                         off (if base (ix+ base k-offset) k-offset)]
                                     (with-meta (list from-array species-sym arr off) {:tag vec-tag})))
                        identity-val (reduce-identity op elem-type)
                        ;; Build per-site, per-accumulator offset loads.
                        ;; For each site [arr base], combine: base + j + k*lanes
                        chunk-loads
                        (vec (for [k (range nacc)]
                               (let [k-offset (if (zero? k) j-sym
                                                  (ix+ j-sym (ix* (int k) lanes-sym)))]
                                 (vec (mapcat (fn [[[arr base] v]]
                                                (let [offset (if base (ix+ base k-offset) k-offset)]
                                                  [v (list from-array species-sym arr offset)]))
                                              (nth arr-groups k))))))
                        elem-syms (vec (for [k (range nacc)] (tag-vec (gensym (str "el" k "__")))))
                        inner-binds (if fuse-fma?
                                      []   ;; loads are inlined into the .fma below
                                      (vec (mapcat (fn [k]
                                                     (concat (nth chunk-loads k)
                                                             [(nth elem-syms k) (nth simd-elems k)]))
                                                   (range nacc))))
                        recur-args (into [(ix+ j-sym stride-sym)]
                                         (for [k (range nacc)]
                                           (if fuse-fma?
                                             ;; vacc_k = a[..]*b[..] + vacc_k, loads inline → memory-operand FMA
                                             (with-meta (list '.fma (fma-load fma-f1 k) (fma-load fma-f2 k) (nth vacc-syms k))
                                               {:tag vec-tag})
                                             (with-meta (list method (nth vacc-syms k) (nth elem-syms k))
                                               {:tag vec-tag}))))
                        combine (reduce (fn [a b] (with-meta (list method a b) {:tag vec-tag}))
                                        (map (fn [k] (nth vacc-syms k)) (range nacc)))
                        scalar-bcasts (vec (mapcat (fn [[s sv]] [sv (list broadcast species-sym (list cf s))])
                                                   scalar-vec-syms))
                        ;; Scalar tail: desugar .invk for bytecode compiler
                        raw-lambda (:lambda (:reduce-op segred))
                        desugared-lambda (bc/desugar-invk raw-lambda)
                        scalar-body (clojure.walk/postwalk (fn [f] (if (= f idx) j-sym f)) desugared-lambda)
                        lanes-op (condp = op
                                   '+ 'jdk.incubator.vector.VectorOperators/ADD
                                   '* 'jdk.incubator.vector.VectorOperators/MUL
                                   'Math/max 'jdk.incubator.vector.VectorOperators/MAX
                                   'Math/min 'jdk.incubator.vector.VectorOperators/MIN
                                   nil)
                        build-loop
                        (fn []
                          (let [vinit (list broadcast species-sym (list cf identity-val))
                                lbinds (vec (concat [j-sym '(int 0)]
                                                    (mapcat (fn [vs] [vs vinit]) vacc-syms)))
                                csym (gensym "vc__")]
                            (list 'loop* lbinds
                                  (list 'if (ix<= (ix+ j-sym stride-sym) n-sym)
                                        (list 'let* inner-binds (list* 'recur recur-args))
                                        (list 'let* [csym combine
                                                     ;; final scalar combine: monomorphic double/float
                                                     ;; in the SIMD path → clojure.core primitive (no box)
                                                     acc (list (cond (= op '+) 'clojure.core/+
                                                                     (= op '*) 'clojure.core/*
                                                                     :else op)
                                                               (list cf init)
                                                               (list '.reduceLanes csym lanes-op))]
                                              (list 'loop* [j-sym j-sym acc acc]
                                                    (list 'if (ix< j-sym n-sym)
                                                          (list 'recur (list 'inc j-sym) scalar-body)
                                                          acc)))))))]
                    (when (and (every? some? simd-elems) lanes-op)
                      (list 'let* [species-sym species-expr
                                   lanes-sym (list '.length species-sym)
                                   stride-sym (ix* (int nacc) lanes-sym)
                                   n-sym (list 'int bound)]
                            (if (seq scalar-bcasts)
                              (list 'let* (vec scalar-bcasts) (build-loop))
                              (build-loop))))))))))))))
