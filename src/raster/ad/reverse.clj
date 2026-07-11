(ns raster.ad.reverse
  "Reverse-mode automatic differentiation via IR transformation.

  Transforms walked S-expressions (from deftm bodies) into augmented
  code that computes both the primal value and a pullback function.
  The pullback maps output adjoints to input adjoints.

  Architecture:
    walked S-expr → normalize → forward pass → reverse pass → [primal, pullback]

  For straight-line code, uses closures-as-tape (Myia insight):
  the pullback is a closure capturing forward intermediates.
  No explicit tape allocation needed.

  Usage:
    (gradient f x)            ;; ∇f(x) : R^n → R
    (vjp f x)                 ;; [f(x), pullback-fn]
    (value+gradient f x)      ;; [f(x), ∇f(x)]"
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.ad.forward :as fwd]
            [raster.numeric :as numeric]
            [raster.ad.templates :as tmpl]
            [raster.compiler.ad.bind-ctx :as bind-ctx]
            [raster.compiler.ad.flatten :as ad-flatten]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.inference :as inf]
            [raster.ad.reverse.normalize :as anf]
            [raster.ad.tangent :as tangent]
            [raster.compiler.ir.form :as form]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.parallel.materialize :as materialize]
            [raster.compiler.passes.parallel.loop-lift :as loop-lift]
            [raster.compiler.passes.parallel.patterns :as patterns]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.support.mangled :as mangled]
            [raster.core :as rcore]
            [raster.arrays :as arrays]))

;; ================================================================
;; Inline callbacks — breaks inline↔reverse cycle.
;; inline.clj registers these at load time.
;; ================================================================

;; ================================================================
;; Gensym utilities for generated code
;; ================================================================

(def ^:private ^:dynamic *gensym-counter*
  "Monotonic counter for AD-generated symbols.
  nil means uninitialized — callers should use `with-ad-gensym`."
  nil)

(defn- ad-gensym
  "Generate a unique symbol for AD-generated code.
  Optional type-tag stamps :raster.type/tag metadata so downstream
  passes can read types without re-inference."
  ([prefix]
   (symbol (str prefix "__" (swap! *gensym-counter* inc))))
  ([prefix type-tag]
   (let [sym (symbol (str prefix "__" (swap! *gensym-counter* inc)))]
     (if type-tag
       (vary-meta sym assoc :raster.type/tag type-tag)
       sym))))

;; ----------------------------------------------------------------
;; Lost-tag instrumentation (diagnostic-first; task #58 phase-1b).
;; The raw-loop reverse-AD codegen defaults an untagged cotangent's shadow
;; array/scalar to 'doubles/'double (silently narrowing f32 → f64: the [F/[D
;; ClassCast + F5 GPU under-devirtualization root). Until the enumeration of
;; lost-tag sites is clean these sites WARN (dedup'd to *err*, keyed by
;; [site binding]) instead of throwing, so the full corpus can be collected
;; without behavior change. Flip to fail-loud (tangent/zero-expr) only once the
;; corpus is clean.
(def ^:private default-tag-warn-seen (atom #{}))

(defn- warn-default-tag!
  "Emit a dedup'd LOST-TAG warning to *err* when a reverse-AD codegen site falls
  back to `default-tag` because the primal binding carries no :raster.type/tag.
  Keyed by [site binding]. Returns `default-tag` unchanged (no behavior change)."
  [site binding-hint default-tag]
  (let [k [site (str binding-hint)]]
    (when-not (contains? @default-tag-warn-seen k)
      (swap! default-tag-warn-seen conj k)
      (binding [*out* *err*]
        (println (str "[ad.reverse LOST-TAG default] site=" site
                      " binding=" binding-hint " -> default " default-tag
                      " (primal has no :raster.type/tag)"))))
    default-tag))

(defmacro ^:private with-ad-gensym
  "Ensure *gensym-counter* is bound. If already bound (non-nil), inherit it;
  otherwise bind a fresh counter starting at 0. This allows composed AD passes
  to share one monotonic counter, avoiding name collisions."
  [& body]
  `(if *gensym-counter*
     (do ~@body)
     (binding [*gensym-counter* (atom 0)]
       ~@body)))

(defn ^:no-doc call-with-shared-ad-gensym
  "Run `thunk` under ONE shared, monotonic *gensym-counter* scope (see
  `with-ad-gensym`). Cross-namespace AD composers (e.g. `raster.ad.jvp/hvp`,
  which chains `ad-prepare` + `reify-pullback`) call this so both phases share
  ONE counter instead of each resetting to 0 and re-minting colliding `anf__`
  temps — the same two-phase-counter hazard fixed inline in
  `build-grad-walked-body`. Function form (not the macro) so it is callable
  from other namespaces without exposing the private counter var."
  [thunk]
  (with-ad-gensym (thunk)))

;; ================================================================
;; Nil-safe gradient accumulation
;; ================================================================

;; Polymorphic array access via raster.arrays — works for any registered
;; array type (double[], float[], and future types like complex[]).

(defn grad-acc
  "Nil-safe gradient accumulation. nil means zero/no gradient.
  Returns nil only when both inputs are nil.
  Handles mixed scalar+array: scalar is broadcast-added to array.
  When both are arrays, uses element-wise addition."
  [a b]
  (cond
    (nil? a) b
    (nil? b) a
    ;; scalar + array: broadcast scalar into array element-wise add
    (and (number? a) (.isArray (class b)))
    (let [n (long (arrays/alength b))
          result (arrays/aclone b)]
      (dotimes [i n]
        (arrays/aset result i (clojure.core/+ (double a) (double (arrays/aget result i)))))
      result)
    (and (number? b) (.isArray (class a)))
    (let [n (long (arrays/alength a))
          result (arrays/aclone a)]
      (dotimes [i n]
        (arrays/aset result i (clojure.core/+ (double (arrays/aget result i)) (double b))))
      result)
    :else (numeric/+ a b)))

(defn aget-grad
  "Gradient of array element read: y = arr[i].
  Returns a zero array of same size as arr, with dy at position i.
  d_arr[i] = dy, d_arr[j] = 0 for j != i."
  [arr i dy]
  (let [n (long (arrays/alength arr))
        g (arrays/alloc-like arr n)]
    (arrays/aset g (int i) dy)
    g))

;; ================================================================
;; Normalization: flatten form for uniform AD processing
;; ================================================================
;; After normalization:
;; - The body is a single symbol
;; - All if-expressions in binding inits have their branches
;;   evaluated into separate bindings (both branches always evaluated)
;; - A branch-flag binding stores the condition result
;; - The if-binding selects via (if flag then-sym else-sym)

(def ^:private trivial-expr? anf/trivial-expr?)

(defn- anf-normalize-bindings [bindings]
  (anf/anf-normalize-bindings bindings ad-gensym))

(defn- normalize-for-ad [bindings body-exprs]
  (anf/normalize-for-ad bindings body-exprs ad-gensym))

(defn- anf-wrap-bare-body
  "Normalize a BARE (non-binding-form) walked body into a `let*` whose bindings
  put every nested call in binding position, BEFORE `lower-to-ad-primitives`.

  Lowering's `inline-deftm-calls` only descends into binding forms
  (`form/binding-form?`), so a deftm whose body is a bare nested expression —
  e.g. `(n/* 2.0 (ad-inner x))` → `(.invk _star 2.0 (.invk ad-inner x))` —
  never gets its composite sub-call inlined. The un-inlined composite then
  reaches `transform-body` with no AD template → nil pullback (the LoRA-wrapper
  null-pullback bug). ANF-lifting the nested call into a binding lets the
  inliner fire and recurse.

  No-op on already-binding-form bodies: the existing (working) let*-bodied path
  is unchanged."
  [body]
  (if (form/binding-form? body)
    body
    (with-ad-gensym
      ;; Propagate the body form's type tag onto the wrapper sym so the
      ;; tangent protocol can emit a statically-typed zero for it.
      (let [r-sym (ad-gensym "adbody" (:raster.type/tag (meta body)))
            anf-b (anf-normalize-bindings [r-sym body])]
        (list 'let* anf-b r-sym)))))

;; ================================================================
;; Core reverse-mode AD transform
;; ================================================================

(declare vjp)

(def ^:private differentiable-tag?
  "Param tags that carry gradient. Floating-point arrays and scalars only —
  Long/long, longs (indices), Boolean, Object, etc. are never differentiated.
  Delegates to the tangent-type protocol (Π) — the type-level ⊥ test."
  tangent/differentiable?)

(defn- auto-make-deftm-rule
  "Create a transient AD rule for a deftm op that has no explicit template.
  Returns [{:pullback-factory rule-fn} canonical-op] or nil.
  Does NOT register in the global template registry — this is a local
  fallback for runtime AD, not a compile-time directive."
  [op]
  (when (qualified-symbol? op)
    (try
      (let [v (resolve op)
            fn-var (or (when (and v (:raster.core/deftm (meta v))) v)
                       (when (and v (.endsWith ^String (name op) "-impl"))
                         (let [base-name (subs (name op) 0 (clojure.core/- (count (name op)) 5))
                               base-sym (symbol (namespace op) base-name)]
                           (when-let [bv (resolve base-sym)]
                             (when (:raster.core/deftm (meta bv)) bv)))))
            canonical-sym (when fn-var (symbol (str (.ns fn-var)) (str (.sym fn-var))))
            existing-closure (when canonical-sym (tmpl/get-pullback-factory canonical-sym))]
        (when fn-var
          (let [rule-fn (or existing-closure
                            (fn [_result & args]
                              (let [[_ pb] (apply vjp fn-var args)]
                                pb)))]
            [{:pullback-factory rule-fn} op])))
      (catch Exception _ nil))))

(def ^:private comparison-op? op/comparison-op?)

(def ^:private numeric-op->qualified
  "Map from unqualified/core arithmetic and math ops to Dual-compatible qualified symbols.
  Used in forward pass so Dual numbers flow through for forward-over-reverse."
  {'+ 'raster.numeric/+, '- 'raster.numeric/-, '* 'raster.numeric/*, '/ 'raster.numeric//,
   'clojure.core/+ 'raster.numeric/+, 'clojure.core/- 'raster.numeric/-,
   'clojure.core/* 'raster.numeric/*, 'clojure.core// 'raster.numeric//,
   'Math/sin 'raster.math/sin, 'Math/cos 'raster.math/cos,
   'Math/exp 'raster.math/exp, 'Math/log 'raster.math/log,
   'Math/sqrt 'raster.numeric/sqrt, 'Math/pow 'raster.numeric/pow,
   'Math/tan 'raster.math/tan, 'Math/abs 'raster.numeric/abs})

(defn- qualify-fwd-expr
  "Qualify arithmetic ops in a forward-pass expression for Dual compatibility."
  [expr]
  (if (seq? expr)
    (let [head (first expr)
          qualified-head (get numeric-op->qualified head head)]
      (cons qualified-head (rest expr)))
    expr))

(defn- free-syms-excluding
  "Collect free symbols in an expression, excluding those in bound-set.
  Does not count seq heads as free (they are operators)."
  [expr bound-set]
  (cond
    (symbol? expr)
    (if (contains? bound-set expr) #{} #{expr})

    (seq? expr)
    (reduce into #{} (map #(free-syms-excluding % bound-set) (rest expr)))

    (vector? expr)
    (reduce into #{} (map #(free-syms-excluding % bound-set) expr))

    :else #{}))

;; Forward declaration for circular dependency: gen-reverse-let <-> gen-reverse-dotimes/par
(declare ^:private gen-reverse-dotimes)
(declare ^:private gen-reverse-par-map)
(declare ^:private gen-reverse-par-reduce)
(declare ^:private gen-reverse-par-scan)

(defn- init-active?
  "Decide whether a binding init carries gradient (is active), given the current
  `activity` map {sym -> bool}. Shared by the AD binding engine and all reverse
  orchestrators (gen-reverse-let / gen-reverse-loop-with-let / reify-pullback)."
  [init-expr activity]
  (cond
    ;; Literal: never active (a symbol alias is active iff its source is)
    (trivial-expr? init-expr)
    (if (symbol? init-expr) (get activity init-expr false) false)

    (seq? init-expr)
    (let [head (first init-expr)]
      (cond
        ;; Comparison: returns a boolean, never active
        (comparison-op? head) false

        ;; If: active iff a branch symbol is active
        (= 'if head)
        (let [[_ _test then else] init-expr]
          (boolean (or (and (symbol? then) (get activity then false))
                       (and (symbol? else) (get activity else false)))))

        ;; Par forms: active iff any free var in the body is active
        (= 'raster.par/map! head)
        (let [[_ _out _idx _bound _cast body] init-expr]
          (boolean (some #(get activity % false) (free-syms-excluding body #{_idx}))))

        (= 'raster.par/reduce head)
        (let [[_ _acc _init _idx _bound body] init-expr]
          (boolean (some #(get activity % false) (free-syms-excluding body #{_idx _acc}))))

        (= 'raster.par/scan head)
        (let [[_ _out _acc init _idx _bound _cast body] init-expr]
          (boolean (some #(get activity % false)
                         (into (free-syms-excluding body #{_idx _acc})
                               (free-syms-excluding init #{})))))

        ;; Loop: active iff any init or body references an active symbol
        (contains? #{'loop 'loop*} head)
        (let [[_ bindings-vec & body-forms] init-expr
              pairs (partition 2 bindings-vec)
              loop-vars (set (map first pairs))
              init-free (reduce into #{} (map #(free-syms-excluding % #{}) (map second pairs)))
              body-free (reduce into #{} (map #(free-syms-excluding % loop-vars) body-forms))]
          (boolean (some #(get activity % false) (into init-free body-free))))

        ;; Normal call / .invk: active iff any arg is active
        :else
        (let [args (if (= '.invk head) (nnext init-expr) (rest init-expr))]
          (boolean (some #(and (symbol? %) (get activity % false)) args)))))

    :else false))

;; ================================================================
;; Forward-pass record creation — one multimethod, kind-dispatched
;; (mirrors form/scope-info's :kind dispatch). A new differentiable SOAC is one
;; new `ad-form-kind` clause + one `ad-record` defmethod — the engine is untouched.
;; ================================================================

(defn- ad-form-kind
  "Classify a binding init into a reverse-record kind. The three array SOACs are
  recognized BEFORE the activity gate (they manage their own activity via the
  active-set); everything else is gated on `active?`. `sym` is the binding
  symbol, used only for a fail-loud message on unsupported control-flow forms."
  [init-expr active? sym]
  (cond
    (and (seq? init-expr) (= 'raster.par/map!   (first init-expr))) :par-map
    (and (seq? init-expr) (= 'raster.par/reduce (first init-expr))) :par-reduce
    (and (seq? init-expr) (= 'raster.par/scan   (first init-expr))) :par-scan
    (and (seq? init-expr) (= 'dotimes           (first init-expr))) :dotimes
    (not active?)                                                   :inactive
    (and (seq? init-expr) (= 'if (first init-expr)))                :if
    (symbol? init-expr)                                             :alias
    (and (seq? init-expr) (contains? #{'loop 'loop*} (first init-expr))) :loop
    ;; Fail loud on control-flow / interop forms carrying an ACTIVE value: the
    ;; `:call` fallthrough would wrap them in a bogus deftm rule → silently
    ;; wrong/dropped gradient. AD through these is unsupported — throw a clear
    ;; message instead of miscompiling. (Inactive occurrences already routed to
    ;; :inactive above, so constant control flow is untouched.)
    (and (seq? init-expr)
         (contains? '#{case case* try letfn letfn* fn fn* new
                       monitor-enter monitor-exit}
                    (first init-expr)))
    (throw (ex-info (str "AD through `" (first init-expr)
                         "` is not supported (form bound to `" sym "`). "
                         "Rewrite using a differentiable primitive "
                         "(e.g. `if`, `par/reduce`, or a deftm with an AD template).")
                    {:form-head (first init-expr) :sym sym :init init-expr}))
    (seq? init-expr)                                                :call
    :else                                                          :inactive))

(defmulti ^:private ad-record
  "Create the reverse-pass record for one binding. Returns
   {:record <map-or-nil> :fwd-patch <fn-or-nil>}, where :fwd-patch (when present)
   is applied to the forward-binding vector via swap! (the SOACs rewrite their own
   forward binding to extract a pullback tape)."
  (fn [kind _sym _init-expr _activity] kind))

(defmethod ad-record :inactive [_ _sym _init _activity] {:record nil})

(defmethod ad-record :alias [_ sym init-expr _activity]
  {:record {:type :alias :sym sym :source init-expr}})

(defmethod ad-record :if [_ sym init-expr _activity]
  (let [[_ branch-expr then-expr else-expr] init-expr]
    {:record {:type :if :sym sym :branch branch-expr :then then-expr :else else-expr}}))

(defmethod ad-record :loop [_ sym _init-expr _activity]
  (throw (ex-info
          (str "Cannot differentiate through raw `loop` form bound to `" sym "`. "
               "Canonical loops ((< i n) test, +1 step, single carry) are lifted "
               "to SOACs automatically; this one declined the soundness gates. "
               "Use `par/reduce` for differentiable reductions, `par/scan` for "
               "differentiable recurrences (a chained carry whose per-step values "
               "you keep — out[i] = acc_i is its own tape), a fixed-point solve "
               "for data-dependent trip counts (while/convergence loops — the "
               "adjoint-of-fixed-point rule differentiates the converged state), "
               "or wrap the loop in a deftm with an AD template.")
          {:sym sym :form-head 'loop})))

(defmethod ad-record :par-map [_ sym init-expr activity]
  (let [active-set (vec (keys (filter val activity)))
        pm-info (assoc (gen-reverse-par-map init-expr active-set) :sym sym)
        tape-sym (:tape-sym pm-info)
        ;; Bind the result sym to the OUTPUT BUFFER (the forward-code populated it),
        ;; NOT nil — so a downstream consumer (`(aget y i)` / `(alength y)`) sees the
        ;; actual array. The result is then an alias of the buffer; its adjoint
        ;; (adj-env[sym]) is combined with adj-env[out-arr] in the backward.
        out-buf (first (:written-arrs pm-info))]
    {:record pm-info
     :fwd-patch (fn [bs]
                  (let [without-last (vec (drop-last 2 bs))]
                    (vec (concat without-last [tape-sym (:forward-code pm-info) sym out-buf]))))}))

(defmethod ad-record :par-reduce [_ sym init-expr activity]
  (let [active-set (vec (keys (filter val activity)))
        pr-info (assoc (gen-reverse-par-reduce init-expr active-set) :sym sym)
        tape-sym (:tape-sym pr-info)
        pair-sym (:result-pair-sym pr-info)]
    {:record pr-info
     :fwd-patch (fn [bs]
                  (let [without-last (vec (drop-last 2 bs))]
                    (vec (concat without-last
                                 [pair-sym (:forward-code pr-info)
                                  tape-sym (list 'aget pair-sym 0)
                                  sym (list 'aget pair-sym 1)]))))}))

(defmethod ad-record :par-scan [_ sym init-expr activity]
  ;; NO :fwd-patch — this is the point of scan-as-recurrence: out[i] = acc_i,
  ;; so THE OUTPUT ARRAY IS THE CARRY TAPE. The forward binding runs completely
  ;; unpatched (the scan macro returns `out`, so `sym` IS the buffer), and the
  ;; backward reconstructs every step's inputs from out[] — no closure tape,
  ;; no ArrayList, no separate residual stack (Griewank store-the-carry
  ;; checkpointing at every step; zero recompute depth).
  (let [active-set (vec (keys (filter val activity)))]
    {:record (assoc (gen-reverse-par-scan init-expr active-set) :sym sym)}))

(defmethod ad-record :dotimes [_ sym init-expr activity]
  (let [active-set (vec (keys (filter val activity)))
        dt-info (assoc (gen-reverse-dotimes init-expr active-set) :sym sym)
        tape-sym (:tape-sym dt-info)
        ;; Bind the result sym to the output buffer (populated by forward-code),
        ;; not nil — so a downstream consumer sees the array (see :par-map).
        out-buf (first (:written-arrs dt-info))]
    {:record dt-info
     :fwd-patch (fn [bs]
                  (let [without-last (vec (drop-last 2 bs))]
                    (vec (concat without-last [tape-sym (:forward-code dt-info) sym out-buf]))))}))

(defmethod ad-record :call [_ sym init-expr activity]
  (let [head (first init-expr)
        [op args invk?] (if (= '.invk head)
                          [(second init-expr) (vec (nnext init-expr)) true]
                          [head (vec (rest init-expr)) false])
        resolved (or (tmpl/resolve-template op)
                     (auto-make-deftm-rule op))
        has-active-args? (some #(and (symbol? %) (get activity % false)) args)]
    (when (and (not resolved) has-active-args?)
      (throw (ex-info (str "No AD template for `" op
                           "` which has active (differentiable) inputs. "
                           "Register an AD template or mark inputs as constant.")
                      {:op op :args args :sym sym
                       :active (filterv #(and (symbol? %) (get activity % false)) args)})))
    {:record
     (when resolved
       (let [[_ base-op] resolved
             arg-tags (let [n (name (if invk? op head))
                            idx (.indexOf ^String n "_m_")]
                        (when (pos? idx)
                          (let [tag-str (subs n (clojure.core/+ idx 3))
                                tag-str (if (.endsWith ^String tag-str "-impl")
                                          (subs tag-str 0 (clojure.core/- (count tag-str) 5))
                                          tag-str)]
                            (mapv symbol (.split ^String tag-str "_")))))]
         {:type :call :sym sym :base-op base-op :args args :arg-tags arg-tags
          :active-args (set (filter #(and (symbol? %) (get activity % false)) args))}))}))

;; ================================================================
;; The reverse-AD binding ENGINE — pure, no mutable state.
;; Three phases, each a `reduce` threading an immutable accumulator:
;;   forward-pass : bindings  -> {:activity :fwd-bindings :records}
;;   reverse-pass : records   -> {:adj-env :rev-ctx}      (emit-backward per record)
;;   collect-param-adjoints   -> {:rev-ctx :param-adj-syms}
;; gen-reverse-let / gen-reverse-loop-with-let / reify-pullback all call
;; `process-bindings`; only their final output ASSEMBLY differs.
;; ================================================================

(defn- sum-contribs
  "Combine a seq of adjoint contribution exprs: empty → `default`, one → itself,
  many → folded with grad-acc."
  [contribs default]
  (cond (empty? contribs)      default
        (= 1 (count contribs)) (first contribs)
        :else (reduce (fn [a b] (list 'raster.ad.reverse/grad-acc a b)) contribs)))

(defn- forward-pass
  "Phase 1 (pure): thread {:activity :fwd-bindings :records} across the bindings."
  [norm-bindings active-params]
  (reduce
   (fn [{:keys [activity fwd-bindings records]} [sym init-expr]]
     (let [active?  (init-active? init-expr activity)
           activity (assoc activity sym active?)
           fwd      (conj fwd-bindings sym (qualify-fwd-expr init-expr))
           {:keys [record fwd-patch]} (ad-record (ad-form-kind init-expr active? sym)
                                                 sym init-expr activity)
           ;; A SOAC whose body carries gradient makes its WRITTEN buffers
           ;; gradient-carrying too: a downstream DIRECT read (aget out k)
           ;; must scatter into adj-env[out] (emit-backward already sums the
           ;; buffer's contributions into d_out alongside the result sym's).
           ;; Without this, statement-position map!/scan followed by a direct
           ;; buffer read silently drops the adjoint (zero gradients).
           activity (if (and active? (seq (:written-arrs record)))
                      (reduce #(assoc %1 %2 true) activity (:written-arrs record))
                      activity)]
       {:activity activity
        :fwd-bindings (if fwd-patch (fwd-patch fwd) fwd)
        :records (if record (conj records record) records)}))
   {:activity (into {} (map (fn [p] [p true]) active-params))
    :fwd-bindings [] :records []}
   (partition 2 norm-bindings)))

(defn- array-zero-of-shape
  "Form for a typed zero adjoint shaped like `arr-sym`, derived from its
  :raster.type/tag (Π). doubles/floats → typed array ctor; double/float →
  typed scalar zero. FAIL-LOUD on untagged/⊥ tags: the old scalar-0.0
  fallback could silently mis-shape an array slot. Callers must only
  materialize this zero when it is actually consumed (compute lazily)."
  [arr-sym]
  (tangent/zero-expr (:raster.type/tag (meta arr-sym))
                     (list 'raster.arrays/alength arr-sym)))

(defn- wire-read-adjoints
  "Append the per-read-array and per-scalar gradient syms into adj-env (the SOAC
  backward shadow arrays / reduce accumulators flow back to their source syms)."
  [adj-env read-arrs d-read-arr-syms active-free d-scalar-syms]
  (as-> adj-env ae
    (reduce (fn [ae [arr d]] (update ae arr (fnil conj []) d)) ae (map vector read-arrs d-read-arr-syms))
    (reduce (fn [ae [p d]]   (update ae p   (fnil conj []) d)) ae (map vector active-free d-scalar-syms))))

(defn- bindings-into
  "Append a flat [sym init ...] vector onto a rev-ctx's :bindings (pure)."
  [rev-ctx kvs]
  (update rev-ctx :bindings into kvs))

(def ^:private nil-possible-heads
  "Adjoint-IR operator heads whose result MAY be the dynamic 0̄ (nil): the
  nil-safe `grad-acc` fold (nil iff both inputs nil), a pullback-vector slot
  `nth` (nil for an inactive/zero component), and `if` (an untaken branch may be
  a nil zero). Anything else in adjoint position is an allocated array/number
  kernel output, hence non-nil."
  '#{raster.ad.reverse/grad-acc grad-acc
     clojure.core/nth nth
     if if*})

(defn- provably-non-nil-contrib?
  "Conservative static test: can `contrib` NEVER evaluate to nil (the dynamic 0̄)?
  True only for a symbol bound (in `binding-map`, the rev-ctx bindings gathered so
  far) to a plain op-call whose head is not a nil-producing form — i.e. a real
  backward-kernel/allocation output (`linear-dx`, `aget-grad`, `residual-add`,
  `float-array`, …). A literal nil, a `(nth grads i)` pullback slot, a nil-safe
  `grad-acc`, an `(if …)`, or a symbol we can't vouch for → false. When in doubt,
  false: the caller then keeps the nil-safe `grad-acc` fold. This is what makes the
  resident `residual-add` reroute SOUND — `residual-add` has no nil check, so it may
  only see cotangents that are provably present."
  [binding-map contrib]
  (letfn [(non-nil-expr? [e]
            (and (some? e) (seq? e) (symbol? (first e))
                 (not (contains? nil-possible-heads (first e)))))]
    (cond
      (nil? contrib)    false
      (symbol? contrib) (if-let [b (find binding-map contrib)]
                          (non-nil-expr? (val b))
                          false)
      (seq? contrib)    (non-nil-expr? contrib)
      :else             false)))

(defn- sum-contribs-into
  "Bind `target-sym` to the sum of `contribs` (default when empty) inside
  `rev-ctx`, choosing a GPU-RESIDENT lowering for the multi-contribution ARRAY
  fan-out case.

  For a SCALAR (or empty / single-contribution) target this is exactly the old
  `(bindings-into rev-ctx [target-sym (sum-contribs contribs default)])`: the
  scalar fold stays `grad-acc` (its nil-safety is load-bearing on the interpreted
  CPU path, and a scalar grad-acc lowers as a host scalar-let anyway).

  For an ARRAY target with ≥2 contributions that are ALL PROVABLY NON-NIL — a
  value/param that FANS OUT to several ACTIVE downstream consumers (e.g. `xn`
  feeding q/k/v, whose cotangents are `linear-dx` kernel outputs) — `grad-acc`'s
  nil-safe `defn` has no GPU kernel, so the array accumulation hits
  `:unlowered-array-op` and defeats residency. Here we instead SEED the
  accumulator with the first contribution directly (numerically identical to
  `(reduce grad-acc contribs)`, which also starts from the first element) and fold
  the rest with the already RESIDENT element-wise `raster.dl.nn/residual-add` (a
  `broadcast +` → `:map` kernel). Each intermediate accumulator is stamped with
  the target's array `tag` so it devirtualizes to the cotangent element dtype
  (float here, not a silently-widened double). The per-add element count is
  `(alength <seed>)`, which `resolve-alength` traces to the seed buffer's
  allocation size host-side.

  CRITICAL soundness gate: `residual-add` has NO nil check, but a cotangent
  contribution can be nil at RUNTIME (a statically-nil template gradient, a
  pullback slot for an inactive arg, an untaken `if` branch). grad-acc's
  `(nil? b) → a` silently drops those; residual-add would NPE. So the reroute
  fires ONLY when every contribution is `provably-non-nil-contrib?`; otherwise we
  keep the nil-safe grad-acc fold (correct on CPU, and grad-acc still lowers as a
  host scalar-let for scalars / is DCE'd when dead). This is exactly the
  attention-block fan-out (all q/k/v cotangents are active kernel outputs) while a
  dot-product / train-step d_W accumulation — which includes a nil template
  gradient — falls back to grad-acc unchanged."
  [rev-ctx target-sym contribs default tag]
  (let [binding-map (into {} (map vec (partition 2 (:bindings rev-ctx))))]
    (if (and (>= (count contribs) 2)
             (= :array (:kind (tangent/tangent-kind tag)))
             (every? #(provably-non-nil-contrib? binding-map %) contribs))
      (let [[seed & more] contribs
            n-sym   (ad-gensym "gacc_n")
            rev-ctx (bindings-into rev-ctx [n-sym (list 'raster.arrays/alength seed)])
            ;; Fold the remaining contributions with the resident add; the LAST add
            ;; binds `target-sym` directly (no trailing alias for the extractor to
            ;; copy-propagate). Intermediates carry `tag` → float-typed buffers.
            n     (count more)
            kvs   (loop [acc seed, cs more, i 0, out []]
                    (if (empty? cs)
                      out
                      (let [dst (if (= i (dec n)) target-sym (ad-gensym "gacc" tag))]
                        (recur dst (rest cs) (inc i)
                               (into out [dst (list 'raster.dl.nn/residual-add
                                                    acc (first cs) n-sym)])))))]
        (bindings-into rev-ctx kvs))
      (bindings-into rev-ctx [target-sym (sum-contribs contribs default)]))))

(defmulti ^:private emit-backward
  "Reverse-pass per-record emission (PURE): returns the updated
  {:adj-env :rev-ctx}. `adj-sym` is this record's gensym'd adjoint (nil for the
  array SOACs, which carry their own d_out/d_acc syms in the record)."
  (fn [record _adj-sym _activity _state] (:type record)))

(defmethod emit-backward :call
  [{:keys [sym base-op args active-args]} adj-sym _activity {:keys [adj-env rev-ctx]}]
  (let [[template _] (tmpl/resolve-template base-op)
        has-template? (and template (or (:grads template) (:grads-fn template)))]
    (if has-template?
      ;; Template path: ctx-aware instantiation (already pure — returns new ctx).
      (let [[new-ctx grad-syms] (tmpl/instantiate-template-ctx template args sym adj-sym rev-ctx)]
        {:rev-ctx new-ctx
         :adj-env (reduce (fn [ae [i grad-sym]]
                            (if (and (some? grad-sym) (< i (count args))
                                     (symbol? (nth args i)) (contains? active-args (nth args i)))
                              (update ae (nth args i) (fnil conj []) grad-sym)
                              ae))
                          adj-env (map-indexed vector grad-syms))})
      ;; Closure fallback — nil pullback entries propagate as nil.
      (let [pb-sym (ad-gensym "pb") grads-sym (ad-gensym "grads")]
        {:rev-ctx (bindings-into rev-ctx
                                 [pb-sym (list* (list 'raster.ad.templates/get-pullback-factory
                                                      (list 'quote base-op)) sym args)
                                  grads-sym (list pb-sym adj-sym)])
         :adj-env (reduce (fn [ae [i arg]]
                            (if (and (symbol? arg) (contains? active-args arg))
                              (update ae arg (fnil conj []) (list 'nth grads-sym i))
                              ae))
                          adj-env (map-indexed vector args))}))))

(defmethod emit-backward :if
  [{:keys [sym branch then else]} adj-sym activity {:keys [adj-env rev-ctx]}]
  ;; Route the adjoint to the taken branch; the other branch gets a SHAPE-matched
  ;; zero (a scalar 0.0 where an array is expected crashes array-add-backward).
  ;; Tagged syms get a STATIC typed zero (fail-loud for known-⊥ tags); untagged
  ;; syms fall back to the runtime `zero-like` on the primal — dynamically
  ;; typed Π, which unlike the old scalar 0.0 cannot mis-shape an array slot.
  ;; `zero` is a delay so the fail-loud path only fires when a branch
  ;; actually materializes the zero.
  (let [zero (delay (if (some? (:raster.type/tag (meta sym)))
                      (array-zero-of-shape sym)
                      (list 'raster.ad.tangent/zero-like sym)))
        adj-env (cond-> adj-env
                  (and (symbol? then) (get activity then false))
                  (update then (fnil conj []) (list 'if branch adj-sym @zero))
                  (and else (symbol? else) (get activity else false))
                  (update else (fnil conj []) (list 'if branch @zero adj-sym)))]
    {:adj-env adj-env :rev-ctx rev-ctx}))

(defmethod emit-backward :alias
  [{:keys [source]} adj-sym activity {:keys [adj-env rev-ctx]}]
  {:adj-env (cond-> adj-env
              (get activity source false) (update source (fnil conj []) adj-sym))
   :rev-ctx rev-ctx})

(defmethod emit-backward :par-map
  [{:keys [sym written-arrs read-arrs d-read-arr-syms d-scalar-syms active-free
           shadow-allocs backward-maps backward-reduces d-out-sym]}
   _adj-sym _activity {:keys [adj-env rev-ctx]}]
  (let [rev-ctx (bindings-into rev-ctx (vec shadow-allocs))
        ;; d_out = adjoint of the result array. The result sym `y` aliases the
        ;; output buffer, so a downstream consumer (`(aget y i)`) scatters into
        ;; adj-env[sym]; a direct buffer write lands in adj-env[out-arr]. SUM both
        ;; (grad-acc element-wise) — dropping either disconnects the gradient.
        rev-ctx (if-let [out-arr (first written-arrs)]
                  (let [contribs (concat (get adj-env sym) (get adj-env out-arr))
                        ;; Materialize the typed zero (fail-loud on unknown
                        ;; tangent space) only when there are no contributions.
                        default (when (empty? contribs) (array-zero-of-shape out-arr))]
                    (bindings-into rev-ctx [d-out-sym (sum-contribs contribs default)]))
                  rev-ctx)
        rev-ctx (reduce bindings-into rev-ctx backward-maps)
        rev-ctx (reduce bindings-into rev-ctx backward-reduces)]
    {:rev-ctx rev-ctx
     :adj-env (wire-read-adjoints adj-env read-arrs d-read-arr-syms active-free d-scalar-syms)}))

(defmethod emit-backward :par-reduce
  [{:keys [sym read-arrs d-read-arr-syms d-scalar-syms active-free
           shadow-allocs backward-maps backward-reduces d-acc-sym]}
   _adj-sym _activity {:keys [adj-env rev-ctx]}]
  (let [rev-ctx (bindings-into rev-ctx (vec shadow-allocs))
        ;; par/reduce returns a SCALAR — a typed scalar zero (Π on the reduce
        ;; result's tag) for the empty case; untagged stays the double 0.0
        ;; (scalar slots cannot mis-shape, only mis-dtype).
        acc-tag (:raster.type/tag (meta sym))
        acc-zero (if (= :scalar (:kind (tangent/tangent-kind acc-tag)))
                   (tangent/zero-expr acc-tag nil)
                   0.0)
        rev-ctx (bindings-into rev-ctx [d-acc-sym (sum-contribs (get adj-env sym) acc-zero)])
        rev-ctx (reduce bindings-into rev-ctx backward-maps)
        rev-ctx (reduce bindings-into rev-ctx backward-reduces)]
    {:rev-ctx rev-ctx
     :adj-env (wire-read-adjoints adj-env read-arrs d-read-arr-syms active-free d-scalar-syms)}))

(defmethod emit-backward :par-scan
  [{:keys [sym written-arrs read-arrs active-free shadow-allocs backward-loop
           d-out-sym n-bwd-sym bound-expr init-expr]}
   _adj-sym activity {:keys [adj-env rev-ctx]}]
  ;; out-as-tape: NO tape unpack — the forward scan ran unpatched; out[] holds
  ;; every carry. The result sym aliases the out buffer, so sum consumer
  ;; (adj-env[sym]) and direct buffer (adj-env[out-arr]) contributions
  ;; (see :par-map) into d_out, the per-step δout[i] source.
  (let [rev-ctx (bindings-into rev-ctx (vec shadow-allocs))
        out-arr (first written-arrs)
        contribs (concat (get adj-env sym) (get adj-env out-arr))
        default (when (empty? contribs) (array-zero-of-shape out-arr))
        rev-ctx (bindings-into rev-ctx [d-out-sym (sum-contribs contribs default)])
        rev-ctx (bindings-into rev-ctx [n-bwd-sym bound-expr])
        bwd-result-sym (ad-gensym "scan_bwd")
        rev-ctx (bindings-into rev-ctx [bwd-result-sym backward-loop])
        ;; backward-loop returns [d_read_arr_0 ... d_scalar_0 ... d_init-carry]
        n-reads (count read-arrs)
        n-scalars (count active-free)
        adj-env (reduce (fn [ae [i arr]]
                          (update ae arr (fnil conj []) (list 'nth bwd-result-sym i)))
                        adj-env (map-indexed vector read-arrs))
        adj-env (reduce (fn [ae [i p]]
                          (update ae p (fnil conj [])
                                  (list 'nth bwd-result-sym (clojure.core/+ n-reads i))))
                        adj-env (map-indexed vector active-free))
        ;; δinit: the carry cotangent remaining after step 0 is ∂loss/∂init —
        ;; wire it to the init symbol when it is active (h0-style learnable
        ;; initial states). Compound active inits are rejected loudly at
        ;; record-construction time (gen-reverse-par-scan).
        adj-env (if (and (symbol? init-expr) (get activity init-expr false))
                  (update adj-env init-expr (fnil conj [])
                          (list 'nth bwd-result-sym (clojure.core/+ n-reads n-scalars)))
                  adj-env)]
    {:adj-env adj-env :rev-ctx rev-ctx}))

(defmethod emit-backward :dotimes
  [{:keys [sym written-arrs read-arrs active-free shadow-allocs backward-loop
           d-out-sym n-bwd-sym bound-expr]}
   _adj-sym _activity {:keys [adj-env rev-ctx]}]
  (let [rev-ctx (bindings-into rev-ctx (vec shadow-allocs))
        ;; result sym aliases the buffer — sum consumer (adj-env[sym]) and direct
        ;; buffer (adj-env[out-arr]) contributions (see :par-map).
        rev-ctx (if-let [out-arr (first written-arrs)]
                  (let [contribs (concat (get adj-env sym) (get adj-env out-arr))
                        default (when (empty? contribs) (array-zero-of-shape out-arr))]
                    (bindings-into rev-ctx [d-out-sym (sum-contribs contribs default)]))
                  rev-ctx)
        rev-ctx (bindings-into rev-ctx [n-bwd-sym bound-expr])
        bwd-result-sym (ad-gensym "dt_bwd")
        rev-ctx (bindings-into rev-ctx [bwd-result-sym backward-loop])
        ;; backward-loop returns [d_read_arr_0 ... d_scalar_0 ...]
        n-reads (count read-arrs)
        adj-env (reduce (fn [ae [i arr]] (update ae arr (fnil conj []) (list 'nth bwd-result-sym i)))
                        adj-env (map-indexed vector read-arrs))
        adj-env (reduce (fn [ae [i p]] (update ae p (fnil conj [])
                                               (list 'nth bwd-result-sym (clojure.core/+ n-reads i))))
                        adj-env (map-indexed vector active-free))]
    {:adj-env adj-env :rev-ctx rev-ctx}))

(defn- reverse-pass
  "Phase 2 (pure): thread {:adj-env :rev-ctx} backward across records, seeded by
  `seed-adj-env` (gen-reverse-let seeds {body-sym [dy]}; loop-with-let seeds the
  loop pullback's grads)."
  [records activity seed-adj-env]
  (reduce
   (fn [{:keys [adj-env rev-ctx]} {:keys [type sym] :as record}]
     (let [array-type? (contains? #{:dotimes :par-map :par-reduce :par-scan} type)
           adj-contribs (when-not array-type? (get adj-env sym))
           sym-tag (:raster.type/tag (meta sym))
           adj-sym (when-not array-type?
                     (ad-gensym (str "d_" (name sym)) sym-tag))
           rev-ctx (if adj-sym
                     ;; Fan-out array cotangent (e.g. d_xn from q/k/v) folds via
                     ;; the resident residual-add; scalars keep the nil-safe
                     ;; grad-acc fold. See sum-contribs-into.
                     (sum-contribs-into rev-ctx adj-sym adj-contribs nil sym-tag)
                     rev-ctx)]
       (if (or array-type? adj-contribs)
         (emit-backward record adj-sym activity {:adj-env adj-env :rev-ctx rev-ctx})
         {:adj-env adj-env :rev-ctx rev-ctx})))
   {:adj-env seed-adj-env :rev-ctx (bind-ctx/make-ctx ad-gensym)}
   (reverse records)))

(defn- param-zero-expr
  "Typed zero default for a param's adjoint slot (Π): array params get a
  shape-matched typed zero (zeros-like keeps the exemplar's dtype), scalar
  params a typed scalar zero. Untagged params keep the scalar 0.0 double
  default (the HVP path differentiates untagged double scalars)."
  [p]
  (let [tag (:raster.type/tag (meta p))]
    (case (:kind (tangent/tangent-kind tag))
      :array  (list 'raster.arrays/zeros-like p (list 'raster.arrays/alength p))
      :scalar (tangent/zero-expr tag nil)
      0.0)))

(defn- collect-param-adjoints
  "Phase 3 (pure): bind a d_<param> for each active param (shape-matched zero when
  no contributions) into rev-ctx; return {:rev-ctx :param-adj-syms}.

  Π at the boundary: each accumulated scalar adjoint is PROJECTED into the
  primal param's tangent space — float params always (double contamination
  via promotion is the default in raster.numeric), double params only when
  float params are present (provably-double-only functions emit no cast).
  Array adjoints are anchored by their typed shadow buffers — no cast."
  [active-params adj-env rev-ctx]
  (let [param-dtype #(-> % meta :raster.type/tag tangent/tangent-kind :dtype)
        mixed-float? (boolean (some #(= :float (param-dtype %)) active-params))]
    (reduce
     (fn [{:keys [rev-ctx param-adj-syms]} p]
       (let [tag (:raster.type/tag (meta p))
             {:keys [kind dtype]} (tangent/tangent-kind tag)
             contribs (get adj-env p)
             adj-sym (ad-gensym (str "d_" (name p)) tag)
             ;; ARRAY param fan-out (a weight consumed by ≥2 ops) accumulates via
             ;; the resident residual-add (no scalar projection applies to arrays).
             rev-ctx
             (if (and (= kind :array) (>= (count contribs) 2))
               (sum-contribs-into rev-ctx adj-sym contribs (param-zero-expr p) tag)
               (let [raw (sum-contribs contribs (param-zero-expr p))
                     expr (if (and (= kind :scalar)
                                   (or (= dtype :float)
                                       (and (= dtype :double) mixed-float?)))
                            (tangent/project-expr tag raw)
                            raw)]
                 (bindings-into rev-ctx [adj-sym expr])))]
         {:rev-ctx rev-ctx
          :param-adj-syms (conj param-adj-syms adj-sym)}))
     {:rev-ctx rev-ctx :param-adj-syms []}
     active-params)))

(defn- process-bindings
  "The shared engine: run Phases 1-3 over a NORMALIZED binding vector and return
  the raw pieces. Pure — no atoms. Wrappers assemble their own output shape.
  opts: {:seed-adj-env <map sym->[adj-expr]>}."
  [norm-bindings active-params {:keys [seed-adj-env]}]
  (let [{:keys [activity fwd-bindings records]} (forward-pass norm-bindings active-params)
        {:keys [adj-env rev-ctx]} (reverse-pass records activity seed-adj-env)
        {:keys [rev-ctx param-adj-syms]} (collect-param-adjoints active-params adj-env rev-ctx)]
    {:fwd-bindings fwd-bindings :records records :adj-env adj-env :activity activity
     :rev-ctx rev-ctx :param-adj-syms param-adj-syms}))

(declare gen-reverse-loop-with-let lift-loop-to-tail)

(defn- loop-init? [init]
  (and (seq? init) (contains? #{'loop 'loop*} (first init))))

(defn- gen-reverse-let-core
  "Generate reverse-mode AD code for a let* form: normalize → process-bindings →
  assemble [primal, pullback-fn]."
  [bindings body-exprs active-params]
  (let [[norm-bindings body-sym] (normalize-for-ad bindings body-exprs)
        dy-sym 'dy__rad
        {:keys [fwd-bindings rev-ctx param-adj-syms]}
        (process-bindings norm-bindings active-params {:seed-adj-env {body-sym [dy-sym]}})
        pullback-body (list 'let* (vec (:bindings rev-ctx)) (vec param-adj-syms))
        pullback-fn   (list 'fn* [dy-sym] pullback-body)
        result-sym    (ad-gensym "result")
        augmented-bindings (vec (concat fwd-bindings [result-sym body-sym]))]
    (list 'let* augmented-bindings (vector result-sym pullback-fn))))

(defn- gen-reverse-let
  "Generate reverse-mode AD code for a let* form.

  A raw `loop` bound to a let* binding is not recordable by the per-binding
  engine (`ad-record :loop` fails loud): only a loop in TAIL position is
  differentiable (via `lift-loop-to-tail` → `gen-reverse-loop-with-let`).
  `transform-body` applies that lift once at the top level, but a SECOND raw
  loop composed in series lands as a mid-let binding inside the FIRST loop's
  result-branch continuation (gen-reverse-loop*'s compound-result path funnels
  that continuation here) — where the tail-lift was never re-applied.

  So when a binding init is an ACTIVE raw loop, lift the first such loop to
  tail (inlining the following bindings + body into its result branch) and
  delegate to `gen-reverse-loop-with-let` — exactly the routing `transform-body`
  uses for a tail loop. The loop-with-let continuation recurses back through
  here, so any number of raw-loop ops compose in series (e.g. N attention
  blocks). Inactive loops and loop-free lets take the core path unchanged."
  [bindings body-exprs active-params]
  (let [pairs (vec (partition 2 bindings))]
    (if-not (some (fn [[_ init]] (loop-init? init)) pairs)
      (gen-reverse-let-core bindings body-exprs active-params)
      ;; Thread activity across the bindings (init-active?, same rule the engine
      ;; uses) WITHOUT running ad-record (which would throw on the active loop).
      ;; Only route when the FIRST loop binding — the one lift-loop-to-tail
      ;; targets — is active; otherwise preserve the core path (and its
      ;; fail-loud behavior for a genuinely un-liftable active loop).
      (let [activity (reduce (fn [act [s init]] (assoc act s (init-active? init act)))
                             (into {} (map (fn [p] [p true]) active-params))
                             pairs)
            first-loop-sym (ffirst (filter (fn [[_ init]] (loop-init? init)) pairs))]
        (if-not (get activity first-loop-sym)
          (gen-reverse-let-core bindings body-exprs active-params)
          (let [lifted (lift-loop-to-tail (apply list 'let* (vec bindings) body-exprs))]
            (cond
              ;; lift-loop-to-tail → bare tail loop (no bindings preceded it)
              (and (seq? lifted) (contains? #{'loop 'loop*} (first lifted)))
              (gen-reverse-loop-with-let [] lifted active-params)

              ;; lift-loop-to-tail → (let* [pre...] (loop ...))
              (and (seq? lifted) (contains? #{'let 'let*} (first lifted))
                   (let [tail (drop 2 lifted)]
                     (and (= 1 (count tail)) (loop-init? (first tail)))))
              (let [[_ pre-bindings loop-form] lifted]
                (gen-reverse-loop-with-let (vec pre-bindings) loop-form active-params))

              ;; lift declined (non-canonical loop body) — preserve fail-loud.
              :else (gen-reverse-let-core bindings body-exprs active-params))))))))

;; ================================================================
;; Loop/recur reverse-mode AD
;; ================================================================

(defn- parse-loop-form
  "Parse a loop* form into its components.
  Returns {:bindings [[sym init] ...], :test, :recur-args, :result-expr}
  Expects loop body shape: (if test (recur args...) result) or vice versa."
  [loop-form]
  (let [[_ raw-bindings & body] loop-form
        bindings (vec (partition 2 raw-bindings))
        loop-syms (mapv first bindings)
        body-expr (if (= 1 (count body)) (first body) (cons 'do body))]
    ;; Normalize: extract if/recur/result
    (when-not (and (seq? body-expr) (= 'if (first body-expr)))
      (throw (ex-info "Loop body must be (if test (recur ...) result)"
                      {:body body-expr})))
    (let [[_ test-expr then-expr else-expr] body-expr
          ;; Extract recur from a branch that may be wrapped in let
          extract-recur
          (fn [expr]
            (cond
              (and (seq? expr) (= 'recur (first expr)))
              {:recur-expr expr :recur-bindings []}
              (and (seq? expr) (#{'let 'let*} (first expr)))
              (let [inner-bindings (vec (partition 2 (second expr)))
                    inner-body (drop 2 expr)
                    last-form (last inner-body)]
                (when (and (seq? last-form) (= 'recur (first last-form)))
                  {:recur-expr last-form :recur-bindings inner-bindings}))
              :else nil))
          then-recur (extract-recur then-expr)
          else-recur (extract-recur else-expr)
          [{:keys [recur-expr recur-bindings]} result-expr recur-in-then?]
          (cond
            then-recur [then-recur else-expr true]
            else-recur [else-recur then-expr false]
            :else (throw (ex-info "Loop body must have recur in one branch"
                                  {:then then-expr :else else-expr})))
          recur-args (vec (rest recur-expr))]
      {:bindings bindings
       :loop-syms loop-syms
       :test-expr test-expr
       :recur-args recur-args
       :result-expr result-expr
       :recur-in-then? recur-in-then?
       :recur-bindings recur-bindings})))

(defn- loop-var-activity
  "Determine which loop vars are active.
  A loop var is active if its init expression OR recur update expression
  references any active param or another active loop var.
  Also traces through recur-bindings (let bindings lifted from the recur branch).
  Uses fixed-point iteration."
  [bindings recur-args active-params & [recur-bindings]]
  (let [loop-syms (mapv first bindings)
        active-set (atom (set active-params))
        ;; Collect free symbols in an expression
        free-syms (fn free-syms [expr]
                    (cond
                      (symbol? expr) #{expr}
                      (seq? expr) (reduce into #{} (map free-syms (rest expr)))
                      :else #{}))
        ;; Build a dependency map for recur-bindings: {sym -> #{dep-syms}}
        rb-deps (into {} (map (fn [[sym expr]] [sym (free-syms expr)])
                              (or recur-bindings [])))
        ;; Expand a set of symbols through recur-binding dependencies
        expand-through-rb (fn [syms]
                            (loop [expanded syms]
                              (let [new-syms (reduce (fn [acc [rb-sym rb-deps]]
                                                       (if (contains? acc rb-sym)
                                                         (into acc rb-deps)
                                                         acc))
                                                     expanded rb-deps)]
                                (if (= new-syms expanded) expanded (recur new-syms)))))
        changed? (atom true)]
    ;; Also mark recur-binding symbols as active if their deps are active
    (doseq [[rb-sym rb-dep-syms] rb-deps]
      (when (some @active-set rb-dep-syms)
        (swap! active-set conj rb-sym)))
    ;; Fixed-point: repeat until no new active vars found
    (while @changed?
      (reset! changed? false)
      (doseq [[sym init] bindings
              :let [idx (.indexOf loop-syms sym)
                    update-expr (nth recur-args idx)
                    init-syms (free-syms init)
                    update-syms (expand-through-rb (free-syms update-expr))
                    all-dep-syms (into init-syms update-syms)]]
        (when (and (not (contains? @active-set sym))
                   (some @active-set all-dep-syms))
          (swap! active-set conj sym)
          (reset! changed? true))))
    (filterv @active-set loop-syms)))

(declare gen-reverse-loop-with-let)
(declare gen-reverse-loop*)

(defn- gen-reverse-loop
  "Generate reverse-mode AD code for a loop* form.

  Strategy (Myia's closures-as-tape):
  - Forward: run loop, at each iteration compute per-iteration pullback
    closure via gen-reverse-let, store pullback in ArrayList
  - Backward: iterate stored pullbacks in reverse, accumulate adjoints
  - Active loop var adjoints chain across iterations
  - Free param adjoints accumulate across all iterations"
  [loop-form active-params]
  ;; CC2 fix: if any init expressions are non-trivial (compound expressions
  ;; that reference active params), lift them into let bindings and delegate
  ;; to gen-reverse-loop-with-let so adjoints propagate through the init.
  ;; E.g. (loop [a (* x 2.0) ...] ...) → (let* [a_init (* x 2.0)] (loop [a a_init ...] ...))
  ;; Only lift inits that reference active params — (long 0) etc. stay in place.
  (let [raw-bindings (vec (partition 2 (second loop-form)))
        active-set (set active-params)
        references-active? (fn refs? [expr]
                             (cond
                               (symbol? expr) (contains? active-set expr)
                               (seq? expr) (some refs? (rest expr))
                               :else false))
        active-nontrivial (filterv (fn [[_ init]]
                                     (and (seq? init) (references-active? init)))
                                   raw-bindings)]
    (if (seq active-nontrivial)
      (let [lifted (mapv (fn [[sym init]]
                           (let [init-sym (ad-gensym (str "init_" (name sym)))]
                             {:sym sym :init init :init-sym init-sym}))
                         active-nontrivial)
            lifted-set (set (map :sym lifted))
            let-bindings (vec (mapcat (fn [{:keys [init-sym init]}]
                                        [init-sym init])
                                      lifted))
            new-loop-bindings (vec (mapcat (fn [[sym init]]
                                             (if (contains? lifted-set sym)
                                               (let [entry (first (filter #(= sym (:sym %)) lifted))]
                                                 [sym (:init-sym entry)])
                                               [sym init]))
                                           raw-bindings))
            new-loop-form (list* 'loop* new-loop-bindings (drop 2 loop-form))]
        (gen-reverse-loop-with-let let-bindings new-loop-form active-params))
      (gen-reverse-loop* loop-form active-params))))

(defn- gen-reverse-loop*
  "Core reverse-mode loop transform (all init expressions are trivial)."
  [loop-form active-params]
  (let [{:keys [bindings loop-syms test-expr recur-args
                result-expr recur-in-then? recur-bindings]} (parse-loop-form loop-form)
        recur-bindings (or recur-bindings [])

        ;; Determine active loop vars (include recur-binding dependencies)
        active-loop-vars (loop-var-activity bindings recur-args active-params recur-bindings)

        ;; All active symbols: active params + active loop vars
        all-active (vec (distinct (concat active-params active-loop-vars)))

        ;; Determine which loop var is the result
        ;; (result-expr should be a loop var symbol)
        result-var (when (symbol? result-expr) result-expr)

        ;; === Per active-loop-var transform data ===
        ;; For each active loop var, gen-reverse-let transforms its update expr.
        ;; iter-active: only active loop vars + active params (gradients returned for these).
        ;; All loop vars are passed to gen-reverse-let as active-params so expressions
        ;; in recur-bindings (e.g. aget W i) can reference loop counters like i.
        ;; Non-active loop vars get nil gradients which are harmlessly discarded.
        iter-active (vec (distinct (concat active-loop-vars active-params)))

        per-var-data
        (mapv (fn [lv]
                (let [idx (.indexOf loop-syms lv)
                      update-expr (nth recur-args idx)
                      nv-sym (ad-gensym (str "nv_" (name lv)))
                      vpb-sym (ad-gensym (str "vpb_" (name lv)))
                      val-sym (ad-gensym (str "val_" (name lv)))
                      pb-sym (ad-gensym (str "pb_" (name lv)))
                      ;; Include recur-branch let bindings so the AD transform
                      ;; differentiates through aget/math inside the loop body.
                      all-bindings (vec (concat (mapcat identity recur-bindings)
                                                [nv-sym update-expr]))]
                  {:lv lv
                   :idx idx
                   :nv-sym nv-sym
                   :vpb-sym vpb-sym
                   :val-sym val-sym
                   :pb-sym pb-sym
                   :transform (gen-reverse-let
                               all-bindings
                               [nv-sym]
                               iter-active)}))
              active-loop-vars)

        n-alv (count active-loop-vars)

        ;; Gensyms for generated code
        pullbacks-sym (ad-gensym "pbs")
        loop-result-sym (ad-gensym "loop_r")
        n-sym (ad-gensym "n")
        j-sym (ad-gensym "j")

        ;; === Forward loop body ===
        ;; At each iteration, for each active loop var:
        ;; 1. Evaluate its transform → [new_value, pullback]
        ;; 2. Store pullback(s)
        ;; 3. Recur with new values
        fwd-loop-body
        (let [transform-bindings
              (vec (mapcat
                    (fn [{:keys [vpb-sym val-sym pb-sym transform]}]
                      [vpb-sym transform
                       val-sym (list 'nth vpb-sym 0)
                       pb-sym (list 'nth vpb-sym 1)])
                    per-var-data))

              store-expr
              (if (= 1 n-alv)
                ;; Single var: store pullback directly
                (list '.add pullbacks-sym (:pb-sym (first per-var-data)))
                ;; Multi var: store vector of pullbacks
                (list '.add pullbacks-sym
                      (cons 'vector (mapv :pb-sym per-var-data))))

              idx->val (into {} (map (fn [{:keys [idx val-sym]}] [idx val-sym])
                                     per-var-data))
              new-recur-args (mapv (fn [i arg] (get idx->val i arg))
                                   (range) recur-args)

              body (list 'let* transform-bindings
                         (list 'do store-expr (cons 'recur new-recur-args)))
              ;; For compound result expressions, return all loop vars + result
              ;; so the pullback can differentiate through the result expression.
              terminal-expr (if result-var
                              result-expr
                              (cons 'vector (concat loop-syms [result-expr])))]
          (if recur-in-then?
            (list 'if test-expr body terminal-expr)
            (list 'if test-expr terminal-expr body)))

        loop-init-bindings (vec (mapcat (fn [[sym init]] [sym init]) bindings))

        ;; The forward loop body. pullbacks-sym (ArrayList) is bound at the
        ;; OUTER scope so both forward and backward code can reference it
        ;; without shadowing — the inline alpha-rename sees one binding.
        forward-loop-form
        (cons 'loop* (cons loop-init-bindings (list fwd-loop-body)))

        ;; === Backward loop ===
        ;; Maintains per-loop-var adjoints (d_lv_i) and per-param adjoints (d_p_k).
        ;; Each iteration (in reverse):
        ;;   For each active loop var i: grads_i = pb_i(d_lv_i)
        ;;   Cross-accumulate: new_d_lv_j = sum_i grads_i[j]
        ;;   Param accumulate: d_p_k += sum_i grads_i[N+k]
        d-lv-syms (mapv (fn [lv] (ad-gensym (str "d_" (name lv)))) active-loop-vars)
        d-param-syms (mapv (fn [p] (ad-gensym (str "d_" (name p)))) active-params)

        backward-body
        (if (= 1 n-alv)
          ;; === Single active var: optimized path (no cross-accumulation) ===
          (let [d-lv-sym (first d-lv-syms)
                iter-pb-sym (ad-gensym "ipb")
                grads-sym (ad-gensym "grads")
                grads-bindings [iter-pb-sym (list '.get pullbacks-sym j-sym)
                                grads-sym (list iter-pb-sym d-lv-sym)]
                new-d-lv (list 'nth grads-sym 0)
                new-d-params (mapv (fn [i p-sym]
                                     (list 'raster.ad.reverse/grad-acc p-sym
                                           (list 'nth grads-sym
                                                 (clojure.core/+ 1 i))))
                                   (range) d-param-syms)
                backward-recur (cons 'recur
                                     (cons (list 'clojure.core/- j-sym 1)
                                           (cons new-d-lv new-d-params)))]
            (list 'let* (vec grads-bindings) backward-recur))

          ;; === Multi active vars: cross-accumulate ===
          (let [pbs-vec-sym (ad-gensym "pbs_v")
                ;; Per-var: extract pullback, call with that var's adjoint
                per-var-grads
                (mapv (fn [i lv]
                        (let [ipb-sym (ad-gensym (str "ipb_" (name lv)))
                              ig-sym (ad-gensym (str "ig_" (name lv)))]
                          {:ipb-sym ipb-sym
                           :ig-sym ig-sym
                           :bindings [ipb-sym (list 'nth pbs-vec-sym i)
                                      ig-sym (list ipb-sym (nth d-lv-syms i))]}))
                      (range) active-loop-vars)

                ;; Cross-accumulate: new_d_lv_j = sum_i grads_i[j]
                new-d-lv-exprs
                (mapv (fn [j]
                        (reduce (fn [acc pvg]
                                  (list 'raster.ad.reverse/grad-acc acc
                                        (list 'nth (:ig-sym pvg) j)))
                                (list 'nth (:ig-sym (first per-var-grads)) j)
                                (rest per-var-grads)))
                      (range n-alv))

                ;; Param accumulate: d_p_k += sum_i grads_i[N+k]
                new-d-param-exprs
                (mapv (fn [k d-p-sym]
                        (let [grad-idx (clojure.core/+ n-alv k)]
                          (reduce (fn [acc pvg]
                                    (list 'raster.ad.reverse/grad-acc acc
                                          (list 'nth (:ig-sym pvg) grad-idx)))
                                  (list 'raster.ad.reverse/grad-acc d-p-sym
                                        (list 'nth (:ig-sym (first per-var-grads))
                                              grad-idx))
                                  (rest per-var-grads))))
                      (range) d-param-syms)

                all-bindings (vec (concat
                                   [pbs-vec-sym (list '.get pullbacks-sym j-sym)]
                                   (mapcat :bindings per-var-grads)))

                recur-args (cons (list 'clojure.core/- j-sym 1)
                                 (concat new-d-lv-exprs new-d-param-exprs))]
            (list 'let* all-bindings (cons 'recur recur-args))))

        ;; === Compound result expression differentiation ===
        ;; When result-expr is not a simple loop var, the forward loop returns
        ;; [lv0 lv1 ... result-value]. The pullback must:
        ;; 1. Bind final loop var values from the forward return vector
        ;; 2. Differentiate result-expr (with final loop vars substituted)
        ;; 3. Use the resulting adjoints to seed the backward loop
        n-loop-syms (count loop-syms)

        result-seed
        (when-not result-var
          (let [;; Gensyms for the final loop var values (accessible in pullback)
                final-lv-syms (mapv (fn [lv] (ad-gensym (str "final_" (name lv))))
                                    loop-syms)
                ;; Substitute loop-var names in result-expr with final-lv gensyms.
                ;; CAPTURE-AVOIDING (util/subst-syms, not walk/postwalk-replace):
                ;; the result branch may contain a NESTED loop that rebinds the
                ;; same loop-var names (two inlined copies of the same deftm — e.g.
                ;; two attention blocks — share `[hq acc]`). A blind postwalk would
                ;; rewrite the inner loop's own binders/refs → variable capture →
                ;; corrupt backward (IndexOutOfBounds / wrong grads). subst-syms
                ;; respects shadowing: the inner rebinding stops the substitution.
                lv-subst (zipmap loop-syms final-lv-syms)
                subst-result (util/subst-syms lv-subst result-expr)
                ;; Differentiate the substituted result-expr w.r.t. iter-active
                ;; (active loop vars + active params), using final-lv names
                subst-iter-active (mapv #(get lv-subst % %) iter-active)
                res-sym (ad-gensym "res")
                ;; Lift any nested let* bindings out of subst-result so AD can
                ;; process each binding as a separate record. If we leave the
                ;; let* opaque, gen-reverse-let's Phase 1 doesn't know how to
                ;; record it (the call dispatch falls through with no template
                ;; for `let*`), Phase 2 has no records, no contributions flow,
                ;; and Phase 3 fires every param adjoint with the no-contribs
                ;; default — the gradient w.r.t. the loop's final value never
                ;; gets computed. Recurse through nested lets to handle deeply
                ;; nested cases produced by inlining.
                [lifted-bindings final-result]
                (loop [expr subst-result acc []]
                  (if (and (seq? expr) (#{'let 'let*} (first expr)))
                    (let [[_ binds & body] expr
                          last-body (last body)]
                      (recur last-body (into acc binds)))
                    [acc expr]))
                res-bindings (vec (concat lifted-bindings [res-sym final-result]))
                res-transform (gen-reverse-let res-bindings [res-sym] subst-iter-active)
                ;; At pullback time: call the result pullback with dy__rad
                rvpb-sym (ad-gensym "rvpb")
                rpb-sym (ad-gensym "rpb")
                rg-sym (ad-gensym "rgrads")
                ;; Per-symbol seed gensyms for the backward loop init
                lv-seed-syms (mapv (fn [lv] (ad-gensym (str "seed_" (name lv))))
                                   active-loop-vars)
                param-seed-syms (mapv (fn [p] (ad-gensym (str "seed_" (name p))))
                                      active-params)]
            {:final-lv-syms final-lv-syms
             :pullback-bindings
             (vec (concat
                   ;; Bind final loop var values from the forward return vector
                   (mapcat (fn [i s] [s (list 'nth loop-result-sym i)])
                           (range) final-lv-syms)
                   ;; Differentiate result expr and call pullback
                   [rvpb-sym res-transform
                    rpb-sym (list 'nth rvpb-sym 1)
                    rg-sym (list rpb-sym 'dy__rad)]
                   ;; Extract per-symbol seeds
                   (mapcat (fn [i s] [s (list 'nth rg-sym i)])
                           (range) lv-seed-syms)
                   (mapcat (fn [i s] [s (list 'nth rg-sym (clojure.core/+ n-alv i))])
                           (range) param-seed-syms)))
             :lv-seed-syms lv-seed-syms
             :param-seed-syms param-seed-syms}))

        ;; Backward loop init: seed adjoints with dy or computed seeds.
        backward-init
        (vec (concat
              [j-sym (list 'clojure.core/- n-sym 1)]
              (if result-var
                ;; Simple result: seed the matching loop var with dy, others nil
                (mapcat (fn [lv d-sym]
                          [d-sym (if (= lv result-var) 'dy__rad nil)])
                        active-loop-vars d-lv-syms)
                ;; Compound result: use pre-computed seed symbols
                (mapcat (fn [d-sym seed-sym] [d-sym seed-sym])
                        d-lv-syms (:lv-seed-syms result-seed)))
              (if result-var
                (mapcat (fn [s] [s nil]) d-param-syms)
                (mapcat (fn [d-sym seed-sym] [d-sym seed-sym])
                        d-param-syms (:param-seed-syms result-seed)))))

        backward-test (list 'clojure.core/>= j-sym 0)

        ;; Backward loop returns all adjoints: [d_lv_0..N-1 d_p_0..M-1]
        backward-return (vec (concat d-lv-syms d-param-syms))

        backward-loop (list 'loop* backward-init
                            (list 'if backward-test
                                  backward-body
                                  backward-return))

        ;; === Init-expr adjoint flow ===
        ;; After backward loop, residual d_lv values flow through init exprs
        ;; back to params. E.g. (loop [a x ...] ...) means d_x += d_a_residual.
        bw-sym (ad-gensym "bw")

        final-param-exprs
        (mapv (fn [k p]
                (let [base-dp (list 'nth bw-sym (clojure.core/+ n-alv k))
                      ;; Contributions from active loop var inits that reference this param
                      init-contribs
                      (keep-indexed
                       (fn [i lv]
                         (let [bind-idx (.indexOf loop-syms lv)
                               [_ init-expr] (nth bindings bind-idx)]
                           (when (and (symbol? init-expr) (= init-expr p))
                             (list 'nth bw-sym i))))
                       active-loop-vars)]
                  (if (empty? init-contribs)
                    base-dp
                    (reduce (fn [acc c] (list 'raster.ad.reverse/grad-acc acc c))
                            base-dp
                            init-contribs))))
              (range) active-params)

        ;; The pullback function.
        ;; For compound result expressions, the seed-computing bindings go
        ;; in the pullback's let*, before the backward loop.
        pullback-bindings
        (vec (concat
              (when result-seed (:pullback-bindings result-seed))
              [n-sym (list '.size pullbacks-sym)
               bw-sym backward-loop]))

        pullback-code
        (list 'fn* ['dy__rad]
              (list 'let* pullback-bindings
                    (vec final-param-exprs)))

        ;; Final assembly: pbs is bound once at the outer scope, shared by
        ;; both forward (which .add's to it) and backward (which .get/.size's it).
        ;; No shadowing, no destructuring — alpha-rename sees one binding.
        ;; For compound result expressions, the forward loop returns a vector
        ;; [lv0 lv1 ... result-value]; extract the result as the last element.
        result-value-expr (if result-var
                            loop-result-sym
                            (list 'nth loop-result-sym n-loop-syms))]
    (list 'let* [pullbacks-sym (list 'java.util.ArrayList.)
                 loop-result-sym forward-loop-form]
          (vector result-value-expr pullback-code))))

(defn- gen-reverse-loop-with-let
  "Handle (let* [bindings...] (loop ...)) by wrapping the let bindings
  around the loop and composing pullbacks.
  The let bindings compute values used by the loop, and the loop
  pullback returns adjoints for those values, which then chain
  backward through the let bindings."
  [let-bindings loop-form active-params]
  (let [anf-bindings (anf-normalize-bindings let-bindings)
        ;; PURE forward pass — now handles par/dotimes/if (the capability the old
        ;; hand-rolled copy lacked). This is what lets an array-valued INTERMEDIATE
        ;; produced in the let-prefix (e.g. a broadcast) and consumed by the loop
        ;; differentiate: its par/map! backward is now generated here.
        {:keys [activity fwd-bindings records]} (forward-pass anf-bindings active-params)
        let-active-syms (filterv #(get activity % false) (map first (partition 2 anf-bindings)))
        ;; Extended active set the loop sees: original params + active let-syms.
        loop-active (vec (distinct (concat active-params let-active-syms)))
        loop-transform (gen-reverse-loop loop-form loop-active)
        dy-sym 'dy__rad
        loop-vpb-sym (ad-gensym "lvpb")
        loop-val-sym (ad-gensym "lval")
        loop-pb-sym  (ad-gensym "lpb")
        loop-grads-sym (ad-gensym "lgrads")
        ;; Seed the let-prefix reverse pass with the loop pullback's adjoints
        ;; (bound at pullback-call time as `loop-grads-sym`).
        seed-adj-env (into {} (map-indexed (fn [i s] [s [(list 'nth loop-grads-sym i)]])
                                           loop-active))
        {:keys [adj-env rev-ctx]} (reverse-pass records activity seed-adj-env)
        {:keys [rev-ctx param-adj-syms]} (collect-param-adjoints active-params adj-env rev-ctx)]
    ;; Assemble: forward let-prefix + loop + composed pullback.
    (list 'let* (vec (concat fwd-bindings
                             [loop-vpb-sym loop-transform
                              loop-val-sym (list 'nth loop-vpb-sym 0)
                              loop-pb-sym  (list 'nth loop-vpb-sym 1)]))
          (vector loop-val-sym
                  (list 'fn* [dy-sym]
                        (list 'let* (vec (concat [loop-grads-sym (list loop-pb-sym dy-sym)]
                                                 (:bindings rev-ctx)))
                              (vec param-adj-syms)))))))

;; ================================================================
;; Dotimes reverse-mode AD (array mutation patterns)
;; ================================================================

(defn- parse-dotimes-form
  "Parse a dotimes form into its components.
  Returns {:counter-sym, :bound-expr, :body}."
  [form]
  (let [[_ [counter-sym bound-expr] & body] form]
    {:counter-sym counter-sym
     :bound-expr bound-expr
     :body (if (= 1 (count body)) (first body) (cons 'do body))}))

(defn- contains-nested-loop?
  "True iff form contains a nested dotimes/loop/loop* (excluding the surface
  let* bindings whose inits we already visit). Used to detect patterns that
  the simple analyzer can't handle."
  [form]
  (cond
    (and (seq? form) (#{'dotimes 'loop 'loop*} (first form))) true
    (and (seq? form) (= 'let* (first form)))
    (or (some contains-nested-loop? (map second (partition 2 (second form))))
        (some contains-nested-loop? (drop 2 form)))
    (seq? form) (some contains-nested-loop? form)
    :else false))

(defn- analyze-dotimes-body
  "Analyze a dotimes body for aset/aget operations.
  Returns {:scalar-bindings [[sym init] ...], :asets [...], :agets [...],
           :free-syms #{...}}.

  Limited shape: handles a flat let* (or no let) whose body-result is the
  final aset. Does NOT recurse into nested dotimes/loops — those would
  reference the outer counter via complex index expressions that this
  template can't safely back-prop through. If a nested loop is detected,
  throws with a pointer to register an explicit AD template."
  [body counter-sym]
  (let [;; Extract let* bindings if present
        [bindings body-result]
        (if (and (seq? body) (#{'let 'let*} (first body)))
          [(partition 2 (second body)) (last (drop 2 body))]
          [[] body])

        ;; Detect unsupported shapes EARLY — silently producing wrong code
        ;; was the d_out__N bug. Surface a clear error instead.
        _ (when (or (some (fn [[_ init]] (contains-nested-loop? init)) bindings)
                    (contains-nested-loop? body-result))
            (throw (ex-info
                    (str "AD reverse-mode dotimes analyzer can't handle nested "
                         "dotimes/loop forms inside the body. The simple flat-loop "
                         "template doesn't apply. Register an explicit AD pullback "
                         "template for the enclosing op via "
                         "raster.ad.templates/merge-into-template!, OR refactor the "
                         "function to use templated primitives (raster.par/map!, "
                         "scaled-dot-product-attn, etc.) that have AD support.")
                    {:counter counter-sym :body body})))

        asets (atom [])
        agets (atom [])
        scalar-bindings (atom [])

        _ (doseq [[sym init] bindings]
            (cond
              ;; aget: record and keep as binding
              (and (seq? init)
                   (op/aget-op?
                    (first init)))
              (do (swap! agets conj {:sym sym :arr (nth init 1) :idx (nth init 2)})
                  (swap! scalar-bindings conj [sym init]))

              ;; aset: record but don't keep as scalar binding
              (and (seq? init)
                   (op/aset-op?
                    (first init)))
              (swap! asets conj {:arr (nth init 1) :idx (nth init 2) :val (nth init 3)})

              ;; Normal binding: keep
              :else
              (swap! scalar-bindings conj [sym init])))

        ;; Check if body-result itself is an aset
        _ (when (and (seq? body-result)
                     (op/aset-op?
                      (first body-result)))
            (swap! asets conj {:arr (nth body-result 1)
                               :idx (nth body-result 2)
                               :val (nth body-result 3)}))

        ;; Collect free symbols (referenced but not bound in body)
        bound-syms (set (cons counter-sym (map first @scalar-bindings)))
        collect-free (fn collect-free [expr]
                       (cond
                         (symbol? expr)
                         (when-not (contains? bound-syms expr) #{expr})

                         (seq? expr)
                         (reduce into #{} (map collect-free (rest expr)))

                         :else #{}))]
    {:scalar-bindings @scalar-bindings
     :asets @asets
     :agets @agets
     :body-result body-result
     :free-syms (reduce into #{}
                        (concat
                         (map (fn [[_ init]] (collect-free init)) @scalar-bindings)
                         (map (fn [{:keys [val]}] (collect-free val)) @asets)))}))

(defn- gen-reverse-dotimes
  "Generate reverse-mode AD code for a dotimes form with array mutation.

  Strategy (closures-as-tape):
  - Forward: run dotimes, at each iteration build pullback closure via
    gen-reverse-let on the scalar computation, store in ArrayList tape.
    Also perform the aset operations normally.
  - Backward: iterate tape in reverse. For each iteration:
    - Read d_out[i] from shadow output array
    - Call pullback with d_out[i] → gradients for inputs
    - Write input gradients to shadow arrays (for aget'd arrays)
    - Accumulate scalar param gradients

  The result is [nil, pullback-fn] where pullback takes a map of
  {arr-sym → d_arr} and returns adjoints for active-params.

  For use within gen-reverse-let: the dotimes record stores enough info
  to generate the backward pass inline."
  [dotimes-form active-params]
  (let [{:keys [counter-sym bound-expr body]} (parse-dotimes-form dotimes-form)
        {:keys [scalar-bindings asets agets free-syms]} (analyze-dotimes-body body counter-sym)

        ;; Determine which arrays are written and read
        written-arrs (vec (distinct (map :arr asets)))
        read-arrs (vec (distinct (map :arr agets)))

        ;; Active free symbols (scalars, not arrays, not counter)
        active-free (filterv (fn [p] (contains? free-syms p)) active-params)

        ;; For each iteration, the scalar computation produces the value(s)
        ;; that get aset'd. We transform these via gen-reverse-let.
        ;; The "iter-active" set includes aget-result symbols + active free params
        aget-syms (mapv :sym agets)
        iter-active (vec (distinct (concat aget-syms active-free)))

        ;; For each aset, get the value being written
        ;; Build the scalar computation to differentiate:
        ;; The bindings are the scalar-bindings (minus aget bindings),
        ;; and the "result" is the value being written.
        pure-scalar-bindings (vec (mapcat identity
                                          (remove (fn [[sym _]]
                                                    (contains? (set aget-syms) sym))
                                                  scalar-bindings)))

        ;; If we have asets, use the first aset's val as the differentiation target
        ;; (multi-aset dotimes would need extension)
        aset-info (first asets)
        val-sym (when aset-info (:val aset-info))

        ;; Gensyms
        tape-sym (ad-gensym "tape")
        vpb-sym (ad-gensym "vpb")
        val-result-sym (ad-gensym "val")
        pb-sym (ad-gensym "pb")
        j-sym (ad-gensym "j")
        n-bwd-sym (ad-gensym "n_bwd")

        ;; Per-iteration scalar transform
        ;; We need val-sym to be the "result" of the scalar computation
        scalar-transform
        (when (and val-sym (seq iter-active))
          (if (seq pure-scalar-bindings)
            (gen-reverse-let
             (vec (concat pure-scalar-bindings [val-result-sym val-sym]))
             [val-result-sym]
             iter-active)
            ;; val-sym is directly an active symbol (e.g., just aget result)
            (gen-reverse-let
             [val-result-sym val-sym]
             [val-result-sym]
             iter-active)))

        ;; === Forward code ===
        ;; (let [tape (ArrayList.)]
        ;;   (dotimes [i n]
        ;;     (let [xi (aget x i)    ;; original aget bindings
        ;;           vpb (transform)   ;; [val, pullback]
        ;;           val (nth vpb 0)
        ;;           pb  (nth vpb 1)]
        ;;       (.add tape pb)
        ;;       (aset out i val)))
        ;;   tape)
        fwd-dotimes-body
        (when scalar-transform
          (let [aget-bindings (vec (mapcat (fn [{:keys [sym arr idx]}]
                                             [sym (list 'aget arr idx)])
                                           agets))
                transform-bindings [vpb-sym scalar-transform
                                    val-result-sym (list 'nth vpb-sym 0)
                                    pb-sym (list 'nth vpb-sym 1)]
                store-and-aset [(ad-gensym "_store") (list '.add tape-sym pb-sym)
                                (ad-gensym "_aset") (list 'aset (:arr aset-info)
                                                          (:idx aset-info)
                                                          val-result-sym)]
                all-bindings (vec (concat aget-bindings transform-bindings store-and-aset))]
            (list 'let* all-bindings nil)))

        forward-code
        (when fwd-dotimes-body
          (list 'let* [tape-sym (list 'java.util.ArrayList.)]
                (list 'do
                      (list 'dotimes [counter-sym bound-expr] fwd-dotimes-body)
                      tape-sym)))

        ;; === Backward code ===
        ;; For each read array: shadow array d_arr — tag matches source array's
        ;; element type (inherits :raster.type/tag if present; defaults to doubles).
        d-read-arr-syms (mapv (fn [arr]
                                (ad-gensym (str "d_" (name arr))
                                           (or (:raster.type/tag (meta arr))
                                               (warn-default-tag! :dotimes-read-arr (name arr) 'doubles))))
                              read-arrs)
        d-scalar-syms (mapv (fn [p]
                              (ad-gensym (str "d_" (name p) "_acc")
                                         (or (:raster.type/tag (meta p))
                                             (warn-default-tag! :dotimes-scalar (name p) 'double))))
                            active-free)

        ;; Backward loop: reads d_out, calls pullbacks, accumulates
        ;; d-out-sym is the gradient of the output array — bound by gen-reverse-let
        ;; from adj-env contributions. Tag inherits from the output array type
        ;; (defaults to doubles, the dominant case for raster training).
        out-array-tag (or (some-> written-arrs first meta :raster.type/tag)
                          (warn-default-tag! :dotimes-out-arr (some-> written-arrs first name) 'doubles))
        out-elem-tag (case out-array-tag
                       doubles 'double, floats 'float, longs 'long, ints 'int,
                       'double)
        d-out-sym (ad-gensym "d_out" out-array-tag)
        iter-pb-sym (ad-gensym "iter_pb")
        d-val-sym (ad-gensym "d_val" out-elem-tag)
        grads-iter-sym (ad-gensym "grads_iter")

        bwd-body-bindings
        (vec (concat
              [iter-pb-sym (list '.get tape-sym j-sym)
               d-val-sym (list 'aget d-out-sym j-sym)
               grads-iter-sym (list iter-pb-sym d-val-sym)]
              ;; aset into shadow arrays for aget'd inputs
              (mapcat (fn [i aget-info d-arr-sym]
                        (let [scatter-sym (ad-gensym "_scatter")]
                          [scatter-sym
                           (list 'aset d-arr-sym j-sym
                                 (list 'nth grads-iter-sym i))]))
                      (range) agets d-read-arr-syms)))

        n-agets (count agets)
        bwd-recur-args
        (cons (list 'clojure.core/- j-sym 1)
              (map-indexed (fn [i d-s]
                             (list 'raster.ad.reverse/grad-acc
                                   d-s
                                   (list 'nth grads-iter-sym (clojure.core/+ n-agets i))))
                           d-scalar-syms))

        bwd-loop-body (list 'let* bwd-body-bindings (cons 'recur bwd-recur-args))

        bwd-loop-init
        (vec (concat
              [j-sym (list 'clojure.core/- n-bwd-sym 1)]
              (mapcat (fn [s] [s nil]) d-scalar-syms)))

        bwd-return (vec (concat d-read-arr-syms d-scalar-syms))

        backward-loop
        (list 'loop* bwd-loop-init
              (list 'if (list 'clojure.core/>= j-sym 0)
                    bwd-loop-body
                    bwd-return))

        ;; Shadow array allocations
        shadow-allocs (vec (mapcat (fn [d-arr-sym arr-sym]
                                     [d-arr-sym (list 'raster.arrays/zeros-like arr-sym
                                                      (list 'clojure.core/alength arr-sym))])
                                   d-read-arr-syms read-arrs))]

    ;; Return: tape + metadata for the backward pass
    ;; The backward will be assembled by the caller (gen-reverse-let)
    ;; when it processes the :dotimes record in reverse
    {:type :dotimes
     :forward-code forward-code
     :tape-sym tape-sym
     :d-out-sym d-out-sym
     :n-bwd-sym n-bwd-sym
     :written-arrs written-arrs
     :read-arrs read-arrs
     :d-read-arr-syms d-read-arr-syms
     :d-scalar-syms d-scalar-syms
     :active-free active-free
     :shadow-allocs shadow-allocs
     :backward-loop backward-loop
     :bound-expr bound-expr}))

;; ================================================================
;; Par map reverse-mode AD (preserves parallel structure)
;; ================================================================

(defn- analyze-par-map-body
  "Analyze a par/map! body for aget references and free scalars.
  Returns {:agets [{:sym :arr :idx} ...], :scalar-bindings [[sym init] ...],
           :body-result expr, :free-syms #{...}}.

  Two sources of aget reads — both routed to the SCATTER (per-element array grad)
  path, following Futhark's vjpMap: a free var read at the map index gets a
  per-element scattered adjoint, NOT a scalar reduce:
   (1) let*-bound agets `[a (aget arr i)]` (explicit ANF), and
   (2) INLINE agets `(aget arr i)` sitting directly in the body result /
       scalar-binding inits — the shape `broadcast`/elementwise bodies produce.
       Each distinct inline `(aget arr <map-idx>)` is lifted to a synthetic aget
       binding (fresh sym) and the body rewritten to reference it, so it joins the
       array-input scatter machinery instead of being mistaken for a free scalar
       (which would reduce → `No promotion rule for + with Double and double[]`).
  Only agets indexed BY THE MAP INDEX scatter; other indexing stays inline."
  [body-expr idx-sym]
  (let [agets (atom [])
        scalar-bindings (atom [])
        ;; Extract let* bindings if present
        [bindings body-result0]
        (if (and (seq? body-expr) (#{'let 'let*} (first body-expr)))
          [(partition 2 (second body-expr)) (last (drop 2 body-expr))]
          [[] body-expr])]
    ;; (1) let*-bound agets
    (doseq [[sym init] bindings]
      (if (and (seq? init)
               (op/aget-op? (first init)))
        (swap! agets conj {:sym sym :arr (nth init 1) :idx (nth init 2)})
        (swap! scalar-bindings conj [sym init])))
    ;; (2) lift inline (aget arr idx-sym) reads to synthetic aget bindings (dedup by
    ;; array — idx is fixed = the map index — so a repeated read shares one sym)
    (let [seen (atom {})
          lift (fn lift [form]
                 (cond
                   (and (seq? form) (op/aget-op? (first form))
                        (= 3 (count form)) (= idx-sym (nth form 2)))
                   (let [arr (nth form 1)
                         k (if (symbol? arr) arr form)]
                     (or (get @seen k)
                         (let [s (ad-gensym (str "ag_" (if (symbol? arr) (name arr) "arr")))]
                           (swap! seen assoc k s)
                           (swap! agets conj {:sym s :arr arr :idx idx-sym})
                           s)))
                   (seq? form) (apply list (map lift form))
                   (vector? form) (mapv lift form)
                   :else form))
          body-result (lift body-result0)
          scalar-bindings* (mapv (fn [[s init]] [s (lift init)]) @scalar-bindings)]
      (let [aget-syms (set (map :sym @agets))
            bound-syms (set (cons idx-sym (concat (map first scalar-bindings*) aget-syms)))
            free-syms (free-syms-excluding body-result bound-syms)
            ;; Also collect free syms from scalar binding inits
            all-free (reduce into free-syms
                             (map (fn [[_ init]]
                                    (free-syms-excluding init bound-syms))
                                  scalar-bindings*))]
        {:agets @agets
         :scalar-bindings scalar-bindings*
         :body-result body-result
         ;; Remove arrays that are aget'd (they're inputs, not scalars)
         :free-syms (disj all-free idx-sym)}))))

(defn- gen-reverse-par-map
  "Generate reverse-mode AD code for a raster.par/map! form.

  Forward form: (raster.par/map! out idx bound cast body-expr)

  Strategy (closures-as-tape, same as dotimes):
  - Forward: run the par/map! as-is, also store per-element pullback
    via a sequential dotimes into an object-array tape (pullbacks are closures).
  - Backward: for array inputs, emit par/map! applying pullback[i] * d_out[i]
    for scalar inputs, emit par/reduce summing pullback contributions.

  Returns a record map for gen-reverse-let to use."
  [par-map-form active-params]
  (let [[_ out-sym idx-sym bound-expr cast-fn body-expr] par-map-form
        {:keys [agets scalar-bindings body-result free-syms]}
        (analyze-par-map-body body-expr idx-sym)

        ;; Determine which arrays are written and read
        written-arrs [out-sym]
        read-arrs (vec (distinct (map :arr agets)))

        ;; Active free symbols (scalars, not arrays, not counter)
        active-free (filterv (fn [p] (contains? free-syms p)) active-params)

        ;; For each iteration, differentiate the scalar body
        aget-syms (mapv :sym agets)
        iter-active (vec (distinct (concat aget-syms active-free)))

        ;; Scalar bindings minus aget bindings
        pure-scalar-bindings (vec (mapcat identity
                                          (remove (fn [[sym _]]
                                                    (contains? (set aget-syms) sym))
                                                  scalar-bindings)))

        ;; Gensyms
        tape-sym (ad-gensym "par_tape")
        vpb-sym (ad-gensym "vpb")
        val-result-sym (ad-gensym "val")
        pb-sym (ad-gensym "pb")

        ;; Per-iteration scalar transform (gen-reverse-let on the scalar body)
        scalar-transform
        (when (seq iter-active)
          (let [binds (if (seq pure-scalar-bindings)
                        (vec (concat pure-scalar-bindings [val-result-sym body-result]))
                        [val-result-sym body-result])]
            (gen-reverse-let binds [val-result-sym] iter-active)))

        ;; === Forward code ===
        ;; Run original par/map! AND build tape of pullbacks
        fwd-dotimes-body
        (when scalar-transform
          (let [aget-bindings (vec (mapcat (fn [{:keys [sym arr idx]}]
                                             [sym (list 'aget arr idx)])
                                           agets))
                transform-bindings [vpb-sym scalar-transform
                                    val-result-sym (list 'nth vpb-sym 0)
                                    pb-sym (list 'nth vpb-sym 1)]
                store-and-aset [(ad-gensym "_store") (list 'aset tape-sym idx-sym pb-sym)
                                (ad-gensym "_aset") (list 'aset out-sym idx-sym
                                                          (if cast-fn
                                                            (list cast-fn val-result-sym)
                                                            val-result-sym))]
                all-bindings (vec (concat aget-bindings transform-bindings store-and-aset))]
            (list 'let* all-bindings nil)))

        forward-code
        (when fwd-dotimes-body
          (list 'let* [tape-sym (list 'object-array bound-expr)]
                (list 'do
                      (list 'dotimes [idx-sym bound-expr] fwd-dotimes-body)
                      tape-sym)))

        ;; === Backward code ===
        ;; For array inputs: par/map! that applies pullback[i] to d_out[i]
        ;; For scalar inputs: par/reduce that accumulates scalar gradients
        ;; Type-tag the gradient symbols so downstream type inference is complete.
        d-read-arr-syms (mapv (fn [arr]
                                (ad-gensym (str "d_" (name arr))
                                           (or (:raster.type/tag (meta arr))
                                               (warn-default-tag! :parmap-read-arr (name arr) 'doubles))))
                              read-arrs)
        d-scalar-syms (mapv (fn [p]
                              (ad-gensym (str "d_" (name p) "_acc")
                                         (or (:raster.type/tag (meta p))
                                             (warn-default-tag! :parmap-scalar (name p) 'double))))
                            active-free)
        out-array-tag (or (some-> out-sym meta :raster.type/tag)
                          (warn-default-tag! :parmap-out-arr (some-> out-sym name) 'doubles))
        out-elem-tag (case out-array-tag
                       doubles 'double, floats 'float, longs 'long, ints 'int,
                       'double)
        d-out-sym (ad-gensym "d_out" out-array-tag)
        bwd-idx-sym (ad-gensym "bi")
        iter-pb-sym (ad-gensym "iter_pb")
        grads-iter-sym (ad-gensym "grads_iter")
        d-val-sym (ad-gensym "d_val" out-elem-tag)

        n-agets (count agets)

        ;; Shadow array allocations
        shadow-allocs (vec (mapcat (fn [d-arr-sym arr-sym]
                                     [d-arr-sym (list 'raster.arrays/zeros-like arr-sym
                                                      (list 'clojure.core/alength arr-sym))])
                                   d-read-arr-syms read-arrs))

        ;; Backward par/map! for each read array:
        ;; (raster.par/map! d_arr bwd_i bound nil
        ;;   (let* [pb (aget tape bwd_i) dv (aget d_out bwd_i) grads (pb dv)]
        ;;     (nth grads aget-index)))
        backward-maps
        (vec (map-indexed
              (fn [aget-idx [d-arr-sym _arr-sym]]
                (let [bwd-body (list 'let* [iter-pb-sym (list 'aget tape-sym bwd-idx-sym)
                                            d-val-sym (list 'aget d-out-sym bwd-idx-sym)
                                            grads-iter-sym (list iter-pb-sym d-val-sym)]
                                     (list 'nth grads-iter-sym aget-idx))]
                  [d-arr-sym (list 'raster.par/map! d-arr-sym bwd-idx-sym bound-expr nil bwd-body)]))
              (map vector d-read-arr-syms read-arrs)))

        ;; Backward par/reduce for each scalar param:
        ;; (raster.par/reduce d_s_acc 0.0 bwd_i bound
        ;;   (let* [pb (aget tape bwd_i) dv (aget d_out bwd_i) grads (pb dv)]
        ;;     (raster.numeric/+ d_s_acc (nth grads scalar-index))))
        d-s-acc-sym (ad-gensym "d_s_acc")
        backward-reduces
        (vec (map-indexed
              (fn [scalar-idx d-scalar-sym]
                (let [grad-idx (clojure.core/+ n-agets scalar-idx)
                      bwd-body (list 'let* [iter-pb-sym (list 'aget tape-sym bwd-idx-sym)
                                            d-val-sym (list 'aget d-out-sym bwd-idx-sym)
                                            grads-iter-sym (list iter-pb-sym d-val-sym)]
                                     (list 'raster.numeric/+ d-s-acc-sym
                                           (list 'nth grads-iter-sym grad-idx)))]
                  [d-scalar-sym (with-meta
                                  (list 'raster.par/reduce d-s-acc-sym 0.0 bwd-idx-sym bound-expr bwd-body)
                                  {:tag 'double})]))
              d-scalar-syms))]

    {:type :par-map
     :forward-code forward-code
     :tape-sym tape-sym
     :d-out-sym d-out-sym
     :written-arrs written-arrs
     :read-arrs read-arrs
     :d-read-arr-syms d-read-arr-syms
     :d-scalar-syms d-scalar-syms
     :active-free active-free
     :shadow-allocs shadow-allocs
     :backward-maps backward-maps
     :backward-reduces backward-reduces
     :bound-expr bound-expr}))

;; ================================================================
;; Par reduce reverse-mode AD (preserves parallel structure)
;; ================================================================

(defn- analyze-par-reduce-body
  "Analyze a par/reduce body for aget references and free scalars.
  Similar to analyze-par-map-body but for reduce body context — including
  its INLINE-aget lift (source (2) there): a bare `(aget arr i)` sitting in
  the body result (the shape the §15.2 loop canonicalization emits, since a
  tail-accumulation loop body is one expression) is lifted to a synthetic
  aget binding so it joins the array-input machinery. Without the lift, the
  aget would be re-emitted inside the forward loop, which alpha-renames the
  index (fi__N) — leaving the original index symbol free/unresolved."
  [body-expr idx-sym acc-sym]
  (let [agets (atom [])
        scalar-bindings (atom [])
        [bindings body-result0]
        (if (and (seq? body-expr) (#{'let 'let*} (first body-expr)))
          [(partition 2 (second body-expr)) (last (drop 2 body-expr))]
          [[] body-expr])]
    (doseq [[sym init] bindings]
      (if (and (seq? init)
               (op/aget-op? (first init)))
        (swap! agets conj {:sym sym :arr (nth init 1) :idx (nth init 2)})
        (swap! scalar-bindings conj [sym init])))
    ;; Lift inline (aget arr idx-sym) reads to synthetic aget bindings (dedup
    ;; by array — idx is fixed = the reduce index, so a repeated read shares
    ;; one sym; e.g. sum-of-squares reads xs[i] twice).
    (let [seen (atom {})
          lift (fn lift [form]
                 (cond
                   (and (seq? form) (op/aget-op? (first form))
                        (= 3 (count form)) (= idx-sym (nth form 2)))
                   (let [arr (nth form 1)
                         k (if (symbol? arr) arr form)]
                     (or (get @seen k)
                         (let [s (ad-gensym (str "ag_" (if (symbol? arr) (name arr) "arr")))]
                           (swap! seen assoc k s)
                           (swap! agets conj {:sym s :arr arr :idx idx-sym})
                           s)))
                   (seq? form) (with-meta (apply list (map lift form)) (meta form))
                   (vector? form) (mapv lift form)
                   :else form))
          body-result (lift body-result0)
          scalar-bindings* (mapv (fn [[s init]] [s (lift init)]) @scalar-bindings)]
      (let [aget-syms (set (map :sym @agets))
            bound-syms (set (list* idx-sym acc-sym (concat (map first scalar-bindings*) aget-syms)))
            free-syms (free-syms-excluding body-result bound-syms)
            all-free (reduce into free-syms
                             (map (fn [[_ init]]
                                    (free-syms-excluding init bound-syms))
                                  scalar-bindings*))]
        {:agets @agets
         :scalar-bindings scalar-bindings*
         :body-result body-result
         :free-syms (disj all-free idx-sym acc-sym)}))))

(defn- gen-reverse-par-reduce
  "Generate reverse-mode AD code for a raster.par/reduce form.

  Forward form: (raster.par/reduce acc init idx bound body-expr)

  Strategy (closures-as-tape):
  - Forward: run the reduce as-is (sequential), also store per-element
    pullback closures in an object-array tape.
  - Backward: for array inputs, emit par/map! applying pullback[i] * d_acc
    for scalar inputs, emit par/reduce summing pullback contributions.

  For simple reductions (acc + f(x[i])):
  d_x[i] = d_acc * df/d(x[i]) → par/map!

  Returns a record map for gen-reverse-let to use."
  [par-reduce-form active-params]
  (let [[_ acc-sym init-expr idx-sym bound-expr body-expr] par-reduce-form
        {:keys [agets scalar-bindings body-result free-syms]}
        (analyze-par-reduce-body body-expr idx-sym acc-sym)

        ;; Read arrays (via aget in body)
        read-arrs (vec (distinct (map :arr agets)))

        ;; Active free symbols
        active-free (filterv (fn [p] (contains? free-syms p)) active-params)

        ;; For each iteration, differentiate the body w.r.t. its inputs
        ;; The body of a reduce uses acc + some per-element function
        ;; We differentiate the body w.r.t. aget results and free scalars
        aget-syms (mapv :sym agets)
        ;; Include acc-sym as an active input (its adjoint chains through iterations)
        iter-active (vec (distinct (concat [acc-sym] aget-syms active-free)))

        ;; Scalar bindings minus aget bindings
        pure-scalar-bindings (vec (mapcat identity
                                          (remove (fn [[sym _]]
                                                    (contains? (set aget-syms) sym))
                                                  scalar-bindings)))

        ;; Gensyms
        tape-sym (ad-gensym "red_tape")
        vpb-sym (ad-gensym "vpb")
        val-result-sym (ad-gensym "val")
        pb-sym (ad-gensym "pb")
        fwd-idx-sym (ad-gensym "fi")

        ;; Per-iteration scalar transform
        scalar-transform
        (when (seq iter-active)
          (let [binds (if (seq pure-scalar-bindings)
                        (vec (concat pure-scalar-bindings [val-result-sym body-result]))
                        [val-result-sym body-result])]
            (gen-reverse-let binds [val-result-sym] iter-active)))

        ;; === Forward code ===
        ;; Run the reduce sequentially, storing pullbacks in tape
        fwd-loop-body
        (when scalar-transform
          (let [aget-bindings (vec (mapcat (fn [{:keys [sym arr idx]}]
                                             [sym (list 'aget arr idx)])
                                           agets))
                transform-bindings [vpb-sym scalar-transform
                                    val-result-sym (list 'nth vpb-sym 0)
                                    pb-sym (list 'nth vpb-sym 1)]
                store-binding [(ad-gensym "_store") (list 'aset tape-sym fwd-idx-sym pb-sym)]
                all-bindings (vec (concat aget-bindings transform-bindings store-binding))]
            ;; The forward loop binds the ALPHA-RENAMED index (fwd-idx-sym) —
            ;; rewrite any surviving reference to the original reduce index
            ;; (aget-binding index exprs, index refs in scalar-binding inits)
            ;; onto it, capture-avoidingly. Without this the emitted loop body
            ;; references an unbound symbol.
            (util/subst-syms {idx-sym fwd-idx-sym}
                             (list 'let* all-bindings val-result-sym))))

        ;; Forward code packs [tape, reduce-result] in an object-array pair.
        ;; The caller (gen-reverse-let) unpacks tape and result from the pair.
        reduce-result-sym (ad-gensym "red_val")
        result-pair-sym (ad-gensym "red_pair")

        forward-code
        (when fwd-loop-body
          (let [pair-arr-sym (ad-gensym "pair")]
            (list 'let* [tape-sym (list 'object-array bound-expr)
                         reduce-result-sym
                         (list 'loop* [fwd-idx-sym 0 acc-sym init-expr]
                               (list 'if (list 'clojure.core/< fwd-idx-sym bound-expr)
                                     (list 'recur (list 'clojure.core/+ fwd-idx-sym 1) fwd-loop-body)
                                     acc-sym))
                         pair-arr-sym (list 'object-array 2)
                         (ad-gensym "_s0") (list 'aset pair-arr-sym 0 tape-sym)
                         (ad-gensym "_s1") (list 'aset pair-arr-sym 1 reduce-result-sym)]
                  pair-arr-sym)))

        ;; === Backward code ===
        d-read-arr-syms (mapv (fn [arr] (ad-gensym (str "d_" (name arr)))) read-arrs)
        d-scalar-syms (mapv (fn [p] (ad-gensym (str "d_" (name p) "_acc"))) active-free)
        d-acc-sym (ad-gensym "d_acc")
        bwd-idx-sym (ad-gensym "bi")
        iter-pb-sym (ad-gensym "iter_pb")
        grads-iter-sym (ad-gensym "grads_iter")

        n-agets (count agets)
        ;; acc-sym is iter-active[0], so its grad is at index 0
        ;; aget-syms are iter-active[1..n-agets], so their grads are at indices 1..n-agets
        ;; active-free are iter-active[n-agets+1..], so grads at n-agets+1..

        ;; Shadow array allocations
        shadow-allocs (vec (mapcat (fn [d-arr-sym arr-sym]
                                     [d-arr-sym (list 'raster.arrays/zeros-like arr-sym
                                                      (list 'clojure.core/alength arr-sym))])
                                   d-read-arr-syms read-arrs))

        ;; For reduce backward: d_acc flows backward through iterations.
        ;; At each iteration i: pullback[i] maps d_acc_out to [d_acc_in, d_aget_0, ..., d_scalar_0, ...]
        ;; Since iterations are chained, we need reverse sequential traversal for d_acc.
        ;; BUT for the aget arrays and scalars, each iteration's contribution is independent.

        ;; The adjoint of a reduce is: for each i from n-1 to 0:
        ;;   d_in_i = pullback_i(d_acc)[aget_index]
        ;;   d_acc = pullback_i(d_acc)[0]  (chain rule through accumulator)
        ;;
        ;; Since d_acc chains, the array gradients depend on the chained d_acc.
        ;; For a simple sum reduction (acc + x[i]):
        ;;   pullback[i](d_acc) = [d_acc, d_acc]  (both partials are 1)
        ;;   So d_x[i] = d_acc for all i (broadcast)
        ;;
        ;; In general, we need a sequential backward loop (like dotimes).
        ;; BUT we can still emit par/map! for the independent parts IF
        ;; the accumulator adjoint is constant (doesn't change across iterations).
        ;;
        ;; For most reductions (sum, weighted sum), d_acc is constant.
        ;; For now, use a sequential backward loop (same approach as dotimes)
        ;; but output the gradient arrays as par forms after the loop.

        ;; Sequential backward loop: iterate from n-1 to 0
        j-sym (ad-gensym "j")
        d-val-sym (ad-gensym "d_val")
        d-acc-chain-sym (ad-gensym "d_acc_chain")

        bwd-body-bindings
        (vec (concat
              [iter-pb-sym (list 'aget tape-sym j-sym)
               grads-iter-sym (list iter-pb-sym d-acc-chain-sym)]
              ;; aset into shadow arrays for aget'd inputs
              ;; acc is iter-active[0], aget-syms are [1..n-agets]
              (mapcat (fn [i _aget-info d-arr-sym]
                        (let [scatter-sym (ad-gensym "_scatter")]
                          [scatter-sym
                           (list 'aset d-arr-sym j-sym
                                 (list 'nth grads-iter-sym (clojure.core/+ 1 i)))]))
                      (range) agets d-read-arr-syms)))

        ;; Recur: update j, chain d_acc through, accumulate scalar grads
        bwd-recur-args
        (cons (list 'clojure.core/- j-sym 1)
              (cons ;; d_acc chains through iteration: pullback[i](d_acc)[0]
               (list 'nth grads-iter-sym 0)
               (map-indexed (fn [i d-s]
                              (list 'raster.ad.reverse/grad-acc
                                    d-s
                                    (list 'nth grads-iter-sym
                                          (clojure.core/+ 1 n-agets i))))
                            d-scalar-syms)))

        n-bwd-sym (ad-gensym "n_bwd")

        bwd-loop-body (list 'let* bwd-body-bindings (cons 'recur bwd-recur-args))

        bwd-loop-init
        (vec (concat
              [j-sym (list 'clojure.core/- n-bwd-sym 1)
               d-acc-chain-sym d-acc-sym]
              (mapcat (fn [s] [s nil]) d-scalar-syms)))

        bwd-return (vec (concat d-read-arr-syms d-scalar-syms))

        backward-loop
        (list 'let* [n-bwd-sym bound-expr]
              (list 'loop* bwd-loop-init
                    (list 'if (list 'clojure.core/>= j-sym 0)
                          bwd-loop-body
                          bwd-return)))

        ;; For par-reduce, the backward uses a sequential loop (due to acc chaining),
        ;; so we represent it as backward-maps (shadow arrays) from the loop result.
        ;; The backward-reduces accumulate the scalar gradients.
        bwd-result-sym (ad-gensym "red_bwd")]

    {:type :par-reduce
     :forward-code forward-code
     :tape-sym tape-sym
     :d-acc-sym d-acc-sym
     :written-arrs []
     :read-arrs read-arrs
     :d-read-arr-syms d-read-arr-syms
     :d-scalar-syms d-scalar-syms
     :active-free active-free
     :shadow-allocs shadow-allocs
     ;; For par-reduce backward, we use a single backward loop
     ;; and distribute results via nth (like dotimes).
     ;; backward-maps: first entry runs the loop, subsequent entries extract.
     :backward-maps
     (vec (concat
           [[bwd-result-sym backward-loop]]
           (map-indexed (fn [i d-arr-sym]
                          [d-arr-sym (list 'nth bwd-result-sym i)])
                        d-read-arr-syms)))
     :backward-reduces
     (vec (map-indexed
           (fn [i d-scalar-sym]
             [d-scalar-sym (list 'nth bwd-result-sym (clojure.core/+ (count read-arrs) i))])
           d-scalar-syms))
     :bound-expr bound-expr
     :reduce-result-sym reduce-result-sym
     :result-pair-sym result-pair-sym}))

;; ================================================================
;; Par scan reverse-mode AD — the differentiable RECURRENCE primitive
;; ================================================================

(defn- gen-reverse-par-scan
  "Generate reverse-mode AD code for a raster.par/scan form — the sanctioned
  differentiable recurrence (the carry chains across steps AND every per-step
  value is kept).

  Forward form: (raster.par/scan out acc init idx bound cast body)
    acc = init; for idx in 0..bound: acc = body; out[idx] = acc; return out.

  KEY INSIGHT (out-as-tape): out[i] = acc_i, so THE OUTPUT ARRAY IS THE CARRY
  TAPE — no closure tape, no ArrayList, no separate residual stack. This is
  Griewank's store-the-carry checkpointing at EVERY step (zero recompute
  depth): the forward binding runs completely unpatched, and the backward
  reconstructs step i's body inputs from the tape — acc_{i-1} = out[i-1]
  (init for i=0) — plus the original array reads at the scan index.
  Body-internal values re-derive from the carry (recompute-from-carry; cheap
  for scalar-carry bodies). Flat-buffer/GPU-ready by construction.

  Backward: one reversed loop, i from bound-1 downto 0, threading the running
  carry cotangent δacc:
    total_i = δout[i] ⊕ δacc          (the out[i] store + step i+1's body)
    grads   = pullback_i(total_i)      (pullback evaluated at acc_{i-1}, x[i])
    δacc    = grads[∂body/∂acc]        (chains to step i-1)
    d_x[·] ⊕= grads[∂body/∂aget_r]     (scatter-ADD at each read's index)
    d_w_s  ⊕= grads[∂body/∂w_s]        (accumulate across steps, loop-carried)
  The carry remaining after step 0 is δinit.

  Cast note: the forward chains the UNCAST acc while out stores cast(acc);
  reconstruction reads the stored value — exact identity for the `double`
  cast, standard checkpoint rounding for narrowing casts (float).

  Returns a record map for the reverse-pass engine (emit-backward :par-scan)."
  [par-scan-form active-params]
  (let [[_ out-sym acc-sym init-expr idx-sym bound-expr _cast-fn body-expr] par-scan-form
        ;; Same body analysis as par/map!: let*-bound agets + inline agets at
        ;; the scan index join the scatter path. acc-sym lands in :free-syms
        ;; but is never an active param, so it cannot leak into active-free.
        {:keys [agets scalar-bindings body-result free-syms]}
        (analyze-par-map-body body-expr idx-sym)

        read-arrs (vec (distinct (map :arr agets)))
        active-free (filterv (fn [p] (contains? free-syms p)) active-params)

        ;; Fail loud: an ACTIVE array referenced outside the recognized aget
        ;; reads would be mis-treated as a free SCALAR (mis-shaped gradient —
        ;; the par/map `No promotion rule` class of bug). Never miscompile.
        _ (doseq [p active-free]
            (when (= :array (:kind (tangent/tangent-kind (:raster.type/tag (meta p)))))
              (throw (ex-info
                      (str "par/scan AD: active array `" p "` is referenced in the "
                           "scan body outside an `(aget " p " <idx>)` read. Only "
                           "aget reads have a scatter gradient path — bind the "
                           "needed element to a let sym inside the body.")
                      {:array p :body body-expr}))))
        ;; Fail loud: a compound init that DEPENDS on active params would
        ;; reconstruct the value correctly but silently drop δinit (the final
        ;; carry only wires to a SYMBOL). Bind it outside the scan first.
        _ (when (and (seq? init-expr)
                     (some (set active-params) (free-syms-excluding init-expr #{})))
            (throw (ex-info
                    (str "par/scan AD: the scan init `" (pr-str init-expr)
                         "` depends on active params. Bind it to a let symbol "
                         "before the scan so its gradient (δinit, the final "
                         "carry cotangent) has a symbol to flow to.")
                    {:init init-expr})))

        aget-syms (mapv :sym agets)
        ;; Pullback grads layout: [∂body/∂acc, ∂body/∂aget..., ∂body/∂free...]
        iter-active (vec (distinct (concat [acc-sym] aget-syms active-free)))

        pure-scalar-bindings (vec (mapcat identity
                                          (remove (fn [[sym _]]
                                                    (contains? (set aget-syms) sym))
                                                  scalar-bindings)))

        ;; Per-step scalar transform (primal recompute + pullback closure) —
        ;; evaluated in the BACKWARD loop at the reconstructed inputs. This is
        ;; recompute-from-carry: no forward-time tape stores it.
        val-result-sym (ad-gensym "val")
        scalar-transform
        (let [binds (if (seq pure-scalar-bindings)
                      (vec (concat pure-scalar-bindings [val-result-sym body-result]))
                      [val-result-sym body-result])]
          (gen-reverse-let binds [val-result-sym] iter-active))

        ;; === Backward code ===
        out-array-tag (or (some-> out-sym meta :raster.type/tag)
                          (warn-default-tag! :carry-out-arr (some-> out-sym name) 'doubles))
        d-out-sym (ad-gensym "d_out" out-array-tag)
        d-read-arr-syms (mapv (fn [arr]
                                (ad-gensym (str "d_" (name arr))
                                           (or (:raster.type/tag (meta arr))
                                               (warn-default-tag! :carry-read-arr (name arr) 'doubles))))
                              read-arrs)
        arr->d-sym (zipmap read-arrs d-read-arr-syms)
        d-scalar-syms (mapv (fn [p] (ad-gensym (str "d_" (name p) "_acc"))) active-free)
        n-bwd-sym (ad-gensym "n_bwd")
        d-carry-sym (ad-gensym "d_carry")
        vpb-sym (ad-gensym "vpb")
        pb-sym (ad-gensym "pb")
        grads-sym (ad-gensym "grads")
        total-sym (ad-gensym "d_total")
        n-agets (count agets)

        shadow-allocs (vec (mapcat (fn [d-arr-sym arr-sym]
                                     [d-arr-sym (list 'raster.arrays/zeros-like arr-sym
                                                      (list 'clojure.core/alength arr-sym))])
                                   d-read-arr-syms read-arrs))

        ;; Step-i reconstruction + pullback. The backward loop counter IS
        ;; idx-sym, so any body sub-expression that still references the scan
        ;; index (index arithmetic, non-lifted reads) sees the right value.
        bwd-body-bindings
        (vec (concat
              ;; acc_{i-1} from the carry tape: out[i-1], init at i=0.
              [acc-sym (list 'if (list 'clojure.core/== idx-sym 0)
                             init-expr
                             (list 'aget out-sym (list 'clojure.core/- idx-sym 1)))]
              ;; Re-bind the body's array reads at their ORIGINAL indices.
              (mapcat (fn [{:keys [sym arr idx]}] [sym (list 'aget arr idx)]) agets)
              [total-sym (list 'raster.ad.reverse/grad-acc
                               (list 'aget d-out-sym idx-sym) d-carry-sym)
               vpb-sym scalar-transform
               pb-sym (list 'nth vpb-sym 1)
               grads-sym (list pb-sym total-sym)]
              ;; Scatter-ADD each read's cotangent at the read's own index
              ;; (read-modify-write: correct for repeated reads of one array
              ;; and for shifted indices like (aget x (- idx 1))).
              (mapcat (fn [k {:keys [arr idx]}]
                        (let [d-arr (arr->d-sym arr)]
                          [(ad-gensym "_scatter")
                           (list 'aset d-arr idx
                                 (list 'raster.ad.reverse/grad-acc
                                       (list 'aget d-arr idx)
                                       (list 'nth grads-sym (clojure.core/+ 1 k))))]))
                      (range) agets)))

        bwd-recur-args
        (list* (list 'clojure.core/- idx-sym 1)
               ;; the carry cotangent chains: δacc_{i-1} = grads[∂body/∂acc]
               (list 'nth grads-sym 0)
               (map-indexed (fn [i d-s]
                              (list 'raster.ad.reverse/grad-acc d-s
                                    (list 'nth grads-sym
                                          (clojure.core/+ 1 n-agets i))))
                            d-scalar-syms))

        bwd-loop-init
        (vec (concat [idx-sym (list 'clojure.core/- n-bwd-sym 1)
                      d-carry-sym 0.0]
                     (mapcat (fn [s] [s nil]) d-scalar-syms)))

        ;; Loop value: [d_read_arr_0 ... d_scalar_0 ... d_init-carry]
        bwd-return (conj (vec (concat d-read-arr-syms d-scalar-syms)) d-carry-sym)

        backward-loop
        (list 'loop* bwd-loop-init
              (list 'if (list 'clojure.core/>= idx-sym 0)
                    (list 'let* bwd-body-bindings (cons 'recur bwd-recur-args))
                    bwd-return))]

    {:type :par-scan
     ;; no :forward-code / :tape-sym — the forward binding IS the tape build.
     :written-arrs [out-sym]
     :read-arrs read-arrs
     :d-read-arr-syms d-read-arr-syms
     :d-scalar-syms d-scalar-syms
     :active-free active-free
     :shadow-allocs shadow-allocs
     :backward-loop backward-loop
     :d-out-sym d-out-sym
     :n-bwd-sym n-bwd-sym
     :bound-expr bound-expr
     :init-expr init-expr}))

;; ================================================================
;; Top-level transform
;; ================================================================

(defn- extract-let-parts
  "Extract bindings and body from a walked form.
  Handles bare let* and nested forms."
  [form]
  (cond
    (and (seq? form) (symbol? (first form))
         (#{'let 'let*} (first form)))
    [(second form) (nnext form)]

    :else
    [[] [form]]))

(declare ^:private lift-loop-to-tail)

(defn transform-body
  "Transform a single walked body expression for reverse-mode AD.
  Returns an S-expression that evaluates to [primal, pullback-fn].

  active-params: ordered sequence of parameter symbols that need gradients.
  form: the walked S-expression body.

  The pullback-fn takes a scalar dy and returns a vector of
  adjoints [d_param1 d_param2 ...] in the same order as active-params."
  [form active-params]
  (with-ad-gensym
    (let [;; If a loop ended up bound to a let-var (common after inlining a
          ;; deftm whose body's tail is a loop), lift it to tail position so
          ;; gen-reverse-loop-with-let can handle it.
          form (lift-loop-to-tail form)
          [bindings body-exprs] (extract-let-parts form)
          ;; Check if the let body is a loop
          body-expr-1 (when (= 1 (count body-exprs)) (first body-exprs))
          body-is-loop? (and body-expr-1 (seq? body-expr-1)
                             (#{'loop 'loop*} (first body-expr-1)))]
      (cond
        ;; let* with loop body: wrap let bindings around the loop
        ;; and pass to gen-reverse-loop-with-let
        (and (seq bindings) body-is-loop?)
        (gen-reverse-loop-with-let bindings body-expr-1 (vec active-params))

        ;; Has let bindings, non-loop body — use the full transform
        (seq bindings)
        (gen-reverse-let bindings body-exprs (vec active-params))

        ;; No let — single expression
        :else
        (let [body-expr (first body-exprs)]
          (cond
            ;; Literal — zero gradient (typed per param tag: a float-array
            ;; param's slot must be a float[] zero, not a scalar 0.0)
            (or (number? body-expr) (nil? body-expr))
            (vector body-expr
                    (list 'fn* ['dy__rad]
                          (mapv param-zero-expr active-params)))

            ;; Symbol — gradient is 1 for that param, typed 0 for others
            (symbol? body-expr)
            (vector body-expr
                    (list 'fn* ['dy__rad]
                          (mapv (fn [p] (if (= p body-expr) 'dy__rad (param-zero-expr p)))
                                active-params)))

            ;; If form — route through gen-reverse-let (normalization handles it)
            (and (seq? body-expr) (= 'if (first body-expr)))
            (gen-reverse-let [] [body-expr] (vec active-params))

            ;; Loop form
            (and (seq? body-expr) (#{'loop 'loop*} (first body-expr)))
            (gen-reverse-loop body-expr (vec active-params))

            ;; Dotimes form — wrap in a let binding for gen-reverse-let to handle
            (and (seq? body-expr) (= 'dotimes (first body-expr)))
            (let [result-sym (ad-gensym "dt")]
              (gen-reverse-let [result-sym body-expr] [result-sym]
                               (vec active-params)))

            ;; Single function call — wrap in a binding
            (seq? body-expr)
            (let [result-sym (ad-gensym "r")]
              (gen-reverse-let [result-sym body-expr] [result-sym]
                               (vec active-params)))

            :else
            (throw (ex-info "Cannot differentiate form" {:form body-expr}))))))))

;; ================================================================
;; Public API
;; ================================================================

(defn ^:no-doc resolve-deftm-var
  "Resolve a deftm dispatch var to its backing mangled var for value+grad. Thin
  wrapper over the canonical raster.core/resolve-deftm-var with the AD policy:
  also accept vars pointing to a value+grad/grad IFn result (:deref-value?),
  throw on ambiguous dispatch, and return f-var unchanged when it is not a
  dispatch var (:on-miss :self)."
  [f-var]
  (rcore/resolve-deftm-var
   f-var {:deref-value? true
          :on-miss :self
          :ambiguity-hint "Use the mangled var directly, e.g. #'ns/name_m_<tags>."}))

(def ^:private vjp-cache (atom {}))

(defn- get-vjp-fn
  "Get or compile the AD-transformed function for a deftm var.
  Cached by [var-identity walked-body] to avoid recompiling on every call.

  Filters active params by tag (same rule as build-grad-walked-body): only
  doubles/floats are seeded as active. The pullback's output vector still has
  one slot per ORIGINAL param (with nil for non-differentiable slots) so
  positional consumers don't shift."
  [resolved]
  (let [m (meta resolved)
        params (or (:raster.core/deftm-params m)
                   (throw (ex-info "vjp requires a deftm var"
                                   {:var resolved})))
        walked-body (or (rcore/ensure-walked-body! resolved)
                        (throw (ex-info "No walked body on var" {:var resolved})))
        cache-key [resolved walked-body]
        cached (get @vjp-cache cache-key)]
    (or cached
        (let [all-params (vec (map #(with-meta (if (symbol? %) % (symbol (name %))) nil) params))
              tags (or (:raster.core/deftm-tags m)
                       (vec (repeat (count params) 'double)))
              diff-active-params (vec (keep-indexed
                                       (fn [i p]
                                         (when (differentiable-tag? (nth tags i nil)) p))
                                       all-params))
              transformed (transform-body (first walked-body) diff-active-params)
              ;; transformed is (let* bindings [primal pullback-fn])
              ;; The pullback returns gradients for diff-active-params only;
              ;; wrap it so it returns gradients aligned with all-params (nil
              ;; in non-differentiable slots).
              pullback-sym (gensym "pb_")
              dy-sym       (gensym "dy_")
              raw-grads-sym (gensym "raw_grads_")
              diff->idx (zipmap diff-active-params (range))
              padded-grads-form
              (mapv (fn [p]
                      (if-let [j (diff->idx p)]
                        (list 'clojure.core/nth raw-grads-sym j)
                        nil))
                    all-params)
              ;; Build (fn [args] (let [tx <transformed> primal (nth tx 0)
              ;;                        pb (nth tx 1)]
              ;;                    [primal (fn [dy] [<padded grads>])]))
              tx-sym (gensym "tx_")
              primal-sym (gensym "primal_")
              fn-form
              (list 'fn (vec all-params)
                    (list 'let* [tx-sym transformed
                                 primal-sym (list 'clojure.core/nth tx-sym 0)
                                 pullback-sym (list 'clojure.core/nth tx-sym 1)]
                          (list 'clojure.core/vector
                                primal-sym
                                (list 'fn [dy-sym]
                                      (list 'let* [raw-grads-sym
                                                   (list pullback-sym dy-sym)]
                                            padded-grads-form)))))
              compiled-fn (eval fn-form)]
          (swap! vjp-cache assoc cache-key compiled-fn)
          compiled-fn))))

(defn vjp
  "Value-and-pullback for a deftm function.
  f-var: a deftm var (dispatch or mangled).
  args: the arguments to evaluate at.

  Returns [f(args...), pullback-fn] where pullback-fn: dy -> [dx1 dx2 ...]."
  [f-var & args]
  (let [resolved (resolve-deftm-var f-var)
        compiled-fn (get-vjp-fn resolved)]
    (apply compiled-fn args)))

;; ================================================================
;; S-expression level API (for testing without deftm)
;; ================================================================

(declare hoist-nested-lets)

(defn- lower-composites
  "ANF-normalize + inline composite (un-templated) deftm calls into AD
  primitives, iterating to a fixpoint.

  `lower-to-ad-primitives` (`inline-deftm-calls`) only descends into binding
  forms, so a BARE-expression body (e.g. `(.invk _star 2.0 (.invk ad-inner x))`)
  is never inlined → the composite reaches `transform-body` with no template →
  nil pullback. `anf-wrap-bare-body` fixes the bare case by lifting nested calls
  into binding position. But a single lowering pass splices callee bodies
  verbatim, so a callee's OWN nested composite lands back in argument position
  (multi-level composite); re-ANF (`relift`) lifts it into binding position for
  the next lowering pass. Each pass strictly reduces the composite-call count,
  so the fixpoint terminates.

  No-op (beyond behavior-preserving ANF) on already-primitive forms: `relift`
  leaves a flat ANF form unchanged, so the loop converges on the first pass."
  [body]
  (with-ad-gensym
    (letfn [(lower1 [f]
              (binding [inline/*ad-transform-body-fn* transform-body]
                (inline/lower-to-ad-primitives f)))
            (relift [f]
              (if (form/binding-form? f)
                (let [[bindings body-exprs] (extract-let-parts f)]
                  (apply list 'let* (anf-normalize-bindings bindings) body-exprs))
                f))]
      (loop [f (anf-wrap-bare-body body) i 0]
        (let [lowered   (lower1 f)
              relifted  (relift lowered)]
          (cond
            (= relifted lowered) lowered           ;; fixpoint: nothing left to inline
            (>= i 32) (throw (ex-info "AD composite lowering did not converge"
                                      {:body body}))
            :else (recur relifted (inc i))))))))

;; ================================================================
;; Loop canonicalization (framework §15.2) — carry→scan materializer
;;
;; The SIMD loop-lift (`lift-parallel-forms`) recovers dotimes+aset → par/map!,
;; tail-accumulation → par/reduce, and carry+aset-returning-out → par/scan.
;; What it cannot lift is the PURE-carry recurrence (no per-step store) and the
;; carry+aset loop that returns the FINAL CARRY instead of the out array —
;; par/scan needs an out buffer, and the loop's value is a scalar. This AD-only
;; materializer closes that gap: it synthesizes the out buffer (which IS the
;; tape — explicit Griewank store-the-carry checkpointing at every step, §15.2)
;; and rebinds the loop's value as the guarded last element. It runs only in
;; ad-prepare because materializing the carry history is pure waste for a
;; non-differentiated primal.
;; ================================================================

(defn- carry-scan-dtype
  "The scalar dtype (:double/:float) of a carry expression, proven from a
  cast wrapper, a typed literal, a tag-stamped symbol, or a tag-stamped call
  form (the walker stamps :raster.type/tag on typed calls — emitters/passes
  READ stamps, never re-infer). nil when it cannot be proven → the
  materializer declines (the loop falls through to the fail-loud :loop
  ladder)."
  [expr]
  (cond
    (and (seq? expr) (contains? #{'float 'clojure.core/float} (first expr))
         (= 2 (count expr))) :float
    (and (seq? expr) (contains? #{'double 'clojure.core/double} (first expr))
         (= 2 (count expr))) :double
    (instance? Double expr) :double
    (instance? Float expr) :float
    (or (symbol? expr) (seq? expr))
    (let [{:keys [kind dtype]} (tangent/tangent-kind
                                (or (:raster.type/tag (meta expr))
                                    (:tag (meta expr))))]
      (when (= kind :scalar) dtype))
    :else nil))

(defn- emit-carry-scan
  "Emit the canonical scan form for a carry loop. Returns a let* expression
  whose value is the loop's value (the final carry).

  RESULT-BINDING CHOICE: par/scan RETURNS the out array (and the :par-scan
  ad-record depends on that — `sym` aliases the buffer, out-as-tape), so the
  final carry must be READ BACK as (aget out (dec n)), exactly the shape the
  o9 scan laws differentiate. The n=0 guard cannot be a lazy branch around the
  aget — flat ANF evaluates binding inits eagerly — so instead BOTH the read
  index and (for the synthesized buffer) the allocation length are clamped to
  keep the eager aget in bounds, and the RESULT is selected by an :if record
  over two symbols: (if (< 0 n) last init). Values and adjoints are exact in
  both cases: for n=0 the aget reads a clamped dummy slot whose adjoint is
  gated to zero by the same branch, and δinit flows through the :if route;
  for n>0 the else-route contributes the branch-gated zero and δinit flows
  through the scan's carry chain."
  [{:keys [out dtype cast acc idx bound init body]}]
  (with-ad-gensym
    (let [scalar-tag (case dtype :float 'float :double 'double nil)
          array-tag  (case dtype :float 'floats :double 'doubles nil)
          alloc-fn   (case dtype :float 'clojure.core/float-array
                           :double 'clojure.core/double-array nil)
          init-sym   (ad-gensym "carry_init" scalar-tag)
          n-sym      (ad-gensym "carry_n" 'long)
          pos-sym    (ad-gensym "carry_pos")
          out-sym    (or out (ad-gensym "carry_out" array-tag))
          scan-sym   (ad-gensym "carry_scan" array-tag)
          li-sym     (ad-gensym "carry_lastidx" 'long)
          last-sym   (ad-gensym "carry_last" scalar-tag)
          bindings   (-> []
                         (into [init-sym init
                                n-sym bound
                                pos-sym (list 'clojure.core/< 0 n-sym)])
                         (into (if out
                                 []   ;; loop already owns the out buffer
                                 [out-sym (list alloc-fn
                                                (list 'if pos-sym n-sym 1))]))
                         (into [scan-sym (list 'raster.par/scan out-sym acc
                                               init-sym idx n-sym cast body)
                                li-sym (list 'if pos-sym
                                             (list 'clojure.core/dec n-sym) 0)
                                last-sym (list 'raster.arrays/aget
                                               scan-sym li-sym)]))]
      (list 'let* (vec bindings)
            (list 'if pos-sym last-sym init-sym)))))

(defn- carry-dtype-consistent?
  "The carry dtype must agree with every KNOWN element type of the arrays the
  step body reads: a mixed-precision scan (double carry over float[] reads)
  would scatter carry-dtype adjoints into read-dtype shadow buffers in the
  backward (typed-aset dispatch CCE). Mixed recurrences decline — they fall
  through to the pre-canonicalization paths, exactly as before."
  [dtype body]
  (every? #(= dtype %) (patterns/read-array-elem-types body)))

(defn- carry-loop->scan
  "Rewrite a carry loop into its canonical par/scan form, or nil when the
  soundness gates decline (see patterns/match-carry-loop). Two shapes:

  (a) carry + per-step aset, loop returns the FINAL CARRY (a scan the lift
      declines because its value is the carry, not out): reuse the loop's
      own out buffer. The n=0 read-guard requires a non-empty out — which
      the loop's own aset contract already implies for n>0; for n=0 an
      empty out throws (degenerate: a zero-trip loop over an empty buffer).

  (b) PURE carry (no store): synthesize the out buffer — the synthesized
      out IS the tape (explicit store-the-carry checkpointing, §15.2)."
  [loop-form]
  (or
   ;; (a) carry + aset returning the carry
   (when-let [{:keys [out-sym acc-sym acc-init index-sym bound-expr cast-fn
                      acc-next-expr else-expr]}
              (patterns/match-scan-loop loop-form)]
     (let [dtype (or (carry-scan-dtype acc-init)
                     (carry-scan-dtype acc-next-expr))]
       (when (and (patterns/acc-ref? else-expr acc-sym)
                  (not (patterns/contains-sym? bound-expr index-sym))
                  (not (patterns/contains-sym? bound-expr acc-sym))
                  (some? dtype)
                  (carry-dtype-consistent? dtype acc-next-expr))
         (emit-carry-scan {:out out-sym
                           :dtype dtype
                           :cast cast-fn
                           :acc acc-sym :idx index-sym
                           :bound bound-expr :init acc-init
                           :body acc-next-expr}))))
   ;; (b) pure carry — dtype from the init or (fallback) the tag-stamped
   ;; step body: the scan stores cast(body), so the body's stamp IS the
   ;; out-buffer dtype.
   (when-let [{:keys [acc-sym acc-init index-sym bound-expr body]}
              (patterns/match-carry-loop loop-form)]
     (let [dtype (or (carry-scan-dtype acc-init)
                     (carry-scan-dtype body))]
       (when (and (some? dtype)
                  (carry-dtype-consistent? dtype body))
         (emit-carry-scan {:out nil
                           :dtype dtype
                           :cast (case dtype :float 'float :double 'double)
                           :acc acc-sym :idx index-sym
                           :bound bound-expr :init acc-init
                           :body body}))))))

(defn- canonicalize-carry-loops
  "Walk a (post-lift) prep form and materialize surviving carry loops as
  par/scan (carry-loop->scan). Descends let* binding inits and body exprs —
  the positions loops occupy after lower-composites' ANF — and leaves par
  forms and unmatched loops untouched (those fall through to the existing
  paths: gen-reverse-loop-with-let on the inline path, the fail-loud ladder
  on the runtime path)."
  [form]
  (cond
    (and (seq? form) (contains? #{'loop 'loop*} (first form)))
    (or (carry-loop->scan form) form)

    (and (seq? form) (form/binding-form? form))
    (let [[head bindings & body] form
          new-bindings (into []
                             (mapcat (fn [[s e]]
                                       [s (canonicalize-carry-loops e)]))
                             (partition 2 bindings))
          new-body (map canonicalize-carry-loops body)]
      (with-meta (apply list head new-bindings new-body) (meta form)))

    :else form))

(defn ^:no-doc ad-prepare
  "The ONE shared pre-AD preparation, used by the reverse path
  (`build-grad-walked-body`, `grad-expr`) AND the forward/JVP path
  (`raster.ad.jvp/jvp`):

    anf-wrap-bare-body → lower-composites (inline un-templated composites to
    a fixpoint) → lift-parallel-forms (loop → SOAC canonicalization, §15.2)
    → canonicalize-carry-loops (AD-only carry → scan materializer) →
    materialize-pass (pure par/map → alloc + par/map!) →
    hoist-nested-lets (flat ANF).

  The loop canonicalization runs AFTER lower-composites (inlining exposes
  callee loops) and BEFORE materialize/hoist. It is a PHASE-ORDERING fix:
  the pipeline's :loop-lift pass runs too late (post-fixpoint) for AD and
  never on the runtime path, so ad-prepare runs the same pure form→form fn
  here. Loops the gates decline are left untouched for the existing paths."
  [body]
  (let [;; ANF-normalize + inline composite deftm calls to a fixpoint so BARE
        ;; and multi-level composites fully lower before AD (lower-composites
        ;; runs anf-wrap-bare-body itself).
        lowered (lower-composites body)
        ;; §15.2 (1): recover SOAC structure from raw loop syntax — dotimes →
        ;; par/map!, tail-accumulation → par/reduce, carry+aset → par/scan —
        ;; with the lift's proven-soundness gates (stats are discarded here;
        ;; matched loops simply become differentiable SOACs).
        lifted (:form (loop-lift/lift-parallel-forms lowered))
        ;; §15.2 (2): materialize surviving pure-carry recurrences as par/scan
        ;; (the synthesized out IS the tape — store-the-carry checkpointing).
        canonical (canonicalize-carry-loops lifted)
        ;; Materialize pure par/map (broadcast → par/pmap) into alloc + par/map!
        ;; before AD — the pure pmap has no reverse rule; the mutating map! does.
        materialized (:form (materialize/materialize-pass canonical nil))]
    ;; Hoist nested lets out of call args into flat ANF for AD.
    (hoist-nested-lets materialized)))

(defn grad-expr
  "Transform a quoted S-expression for reverse-mode AD.
  Returns the transformed S-expression that evaluates to
  [primal, pullback-fn].

  Recurses into user `deftm` calls that have no AD template by inlining them to
  template-bearing primitives first (the Zygote 'recurse-into-IR unless a rule
  exists' model), then hoists nested lets into flat ANF — mirroring the
  var-driven path (`build-grad-walked-body`). Without this, a raw form
  containing a user-deftm call throws \"No AD template\" instead of recursing,
  because the differentiator core only knows template-bearing primitives. Ops
  WITH a template stay symbolic for the transform."
  [form active-params]
  ;; ONE shared gensym counter across both lowering phases (same fix as
  ;; build-grad-walked-body): ad-prepare (lower-composites) and transform-body
  ;; each open their OWN with-ad-gensym; without this enclosing scope each
  ;; RESETS *gensym-counter* to 0 and can re-mint colliding anf__ temps across
  ;; the phase boundary → silent gradient miscompile.
  (with-ad-gensym
    (transform-body (ad-prepare form) (vec active-params))))

(defn numerical-gradient
  "Compute gradient by finite differences (for testing).
  f: a Clojure function of doubles.
  args: vector of double arguments.
  Returns vector of partial derivatives."
  [f args & {:keys [eps] :or {eps 1e-7}}]
  (let [n (count args)
        f0 (apply f args)]
    (vec (for [i (range n)]
           (let [args+ (update args i clojure.core/+ eps)
                 f+ (apply f args+)]
             (clojure.core// (clojure.core/- f+ f0) eps))))))

;; ================================================================
;; Compositional AD: Reified pullbacks for reverse-over-reverse
;; ================================================================
;;
;; Key idea: instead of emitting pullback as (fn* [dy] ...) closure,
;; emit it as a flat let* form with captured forward values as explicit
;; parameters. This makes the pullback itself differentiable by AD.
;;
;; Output structure:
;;   {:primal-form   (let* [...fwd...] body-sym)
;;    :pullback-form (let* [...rev...] [d_p1 d_p2 ...])
;;    :pullback-params [dy captured1 captured2 ...]
;;    :active-params [p1 p2 ...]
;;    :captured-syms [c1 c2 ...]}

(defn- collect-free-syms
  "Collect unqualified, non-namespaced free symbols from an S-expression."
  [e]
  (cond (and (symbol? e) (not (namespace e))) #{e}
        (seq? e)    (apply clojure.set/union #{} (map collect-free-syms e))
        (vector? e) (apply clojure.set/union #{} (map collect-free-syms e))
        :else       #{}))

(defn- collect-bound-syms
  "Collect symbols bound in a let* bindings vector."
  [bindings-vec]
  (set (take-nth 2 bindings-vec)))

(defn ^:no-doc reify-pullback
  "Transform a walked body into a reified pullback suitable for composition.
  Public (^:no-doc) consumer: raster.ad.jvp/hvp jvp-folds the reified
  gradient program (forward-over-reverse, §13 A4).

  Instead of the standard [primal, (fn* [dy] ...)] output, returns a map:
    {:primal-form     S-expr that computes the primal value
     :pullback-form   S-expr that computes gradients (a let* form)
     :pullback-params [dy-sym captured1 captured2 ...]
     :active-params   [p1 p2 ...] (original function params)
     :captured-syms   [c1 c2 ...] (forward values needed by pullback)
     :fwd-bindings    [...] (forward pass bindings vector)}

  The pullback-form can be transformed by AD for reverse-over-reverse."
  [form active-params]
  (with-ad-gensym
    (let [[bindings body-exprs] (extract-let-parts form)
          [norm-bindings body-sym] (normalize-for-ad bindings body-exprs)

          ;; === Phases 1-3: the shared pure engine (no atoms) ===
          dy-sym 'dy__rad
          {:keys [fwd-bindings rev-ctx param-adj-syms]}
          (process-bindings norm-bindings active-params {:seed-adj-env {body-sym [dy-sym]}})

          ;; === Phase 4: Reify pullback ===
          rev-bindings (vec (:bindings rev-ctx))
          pullback-body (list 'let* rev-bindings (vec param-adj-syms))

          ;; Determine captured variables:
          ;; Free symbols in the reverse bindings that are NOT:
          ;; - defined within the reverse bindings themselves
          ;; - the dy symbol
          ;; - the active params (these flow separately)
          rev-bound (collect-bound-syms rev-bindings)
          rev-free  (collect-free-syms rev-bindings)
          fwd-bound (collect-bound-syms (vec fwd-bindings))
          ;; Captured = free in reverse bindings ∩ bound in forward bindings
          captured  (clojure.set/intersection rev-free fwd-bound)
          ;; Maintain deterministic order: use forward binding order
          captured-ordered (filterv captured (take-nth 2 fwd-bindings))

          ;; Pullback params: [dy, captured1, captured2, ...]
          pullback-params (vec (cons dy-sym captured-ordered))

          ;; Standard output for backward compatibility
          result-sym (ad-gensym "result")
          augmented-bindings (vec (concat fwd-bindings [result-sym body-sym]))]

      {:primal-form     (list 'let* augmented-bindings result-sym)
       :pullback-form   pullback-body
       :pullback-params pullback-params
       :active-params   (vec active-params)
       :captured-syms   captured-ordered
       :fwd-bindings    (vec fwd-bindings)
       :result-sym      result-sym
       :body-sym        body-sym
       ;; Standard [primal, pullback] for backward compat
       :standard-form
       (list 'let* augmented-bindings
             (vector result-sym
                     (list 'fn* [dy-sym] pullback-body)))})))

(defn- transform-pullback
  "Apply AD to a reified pullback form, producing second-order derivatives.

  Takes a reified pullback (from reify-pullback) and differentiates the
  pullback function itself. This enables reverse-over-reverse (RoR) AD.

  diff-params: which pullback params to differentiate w.r.t.
               Typically the original function params (subset of captured-syms).
               The dy param is always active (it's the 'input' to the pullback).

  Returns a new reified pullback map for the second-order computation."
  [reified-pb diff-params]
  (let [{:keys [pullback-form pullback-params]} reified-pb
        ;; The pullback form is a let* — extract bindings and body
        [_ bindings-vec & body-exprs] pullback-form
        ;; diff-params should be a subset of pullback-params
        active (set diff-params)
        active-ordered (filterv active pullback-params)]
    (with-ad-gensym
      (let [;; Treat the pullback-form as a function body with pullback-params
            ;; Build a let* wrapping for the transform
            transformed (gen-reverse-let bindings-vec body-exprs active-ordered)]
        {:second-order-form transformed
         :second-order-params active-ordered
         :original reified-pb}))))

(defn- vjp-compositional
  "Value-and-pullback with reified pullback for composition.

  Like vjp, but returns a map with both the standard [value, pullback-fn]
  and the reified pullback form for optional second-order differentiation.

  Returns {:value     f(args)
           :pullback  (fn [dy] -> [d_p1 d_p2 ...])
           :reified   reified-pullback-map}"
  [f-var & args]
  (let [resolved (resolve-deftm-var f-var)
        m (meta resolved)
        params (or (:raster.core/deftm-params m)
                   (throw (ex-info "vjp-compositional requires a deftm var"
                                   {:var f-var})))
        walked-body (or (rcore/ensure-walked-body! resolved)
                        (throw (ex-info "No walked body" {:var f-var})))
        active-params (vec (map #(with-meta (if (symbol? %) % (symbol (name %))) nil) params))
        reified (reify-pullback (first walked-body) active-params)
        param-bindings (vec (mapcat vector active-params args))
        ;; Eval the standard form for the actual computation
        [value pullback] (eval (list 'let* param-bindings (:standard-form reified)))]
    {:value value
     :pullback pullback
     :reified reified}))

;; ================================================================
;; Reified gradient program as a first-class deftm-style fn (#72)
;; ================================================================

(declare ^:private make-runtime-value+grad-fn)

(defonce ^:private ror-rename-counter (atom 0))

(defn- alpha-rename-bound
  "α-rename every let*-bound symbol in a FLAT binding vector to a globally
  unique name (\"__gp<N>\" suffix, monotonic defonce counter), preserving
  each binding sym's :raster.type/tag on all occurrences.

  Why: a reified gradient program FREEZES symbols minted under one
  `with-ad-gensym` scope (counter starts at 0). A LATER AD pass over the
  frozen program starts a fresh counter, and its ANF lifts mint the same
  names — silently REBINDING inner syms mid-program (e.g. a float[] anf__2
  rebound to a Double ANF temp → ClassCastException, or worse a silent
  wrong gradient). Globally-unique renames make the frozen program safe to
  re-enter with any gensym state."
  [bindings tail]
  (let [bound (vec (take-nth 2 bindings))
        smap (into {}
                   (map (fn [s]
                          [s (with-meta
                               (symbol (str (name s) "__gp" (swap! ror-rename-counter inc)))
                               (meta s))]))
                   bound)
        rename (fn [form] (walk/postwalk-replace smap form))]
    [(vec (mapcat (fn [[s init]] [(get smap s s) (rename init)])
                  (partition 2 bindings)))
     (rename tail)]))

(defn ^clojure.lang.IFn reified-grad
  "Reify ∂f/∂param of a scalar deftm as a deftm-style meta-fn — the
  double-backward (reverse-over-reverse) composition surface (#72).

  The result's walked body IS the flat gradient program (reify-pullback
  assembled with the unit adjoint, as hvp does), so `value+grad` of the
  result differentiates THROUGH the gradient computation — grad-of-grad,
  L2(∇f) regularization, MAML-class meta-objectives — using the backward
  kernels' own derived reverse rules (raster.ad.templates op-adjoints).

  param: symbol naming one differentiable param of f.
  - SCALAR param p → (fn [args…] -> ∂f/∂p), a scalar-valued program.
  - ARRAY  param p → (fn [args… c] -> ⟨∂f/∂p, c⟩) with a cotangent array
    `c` appended after f's params (value+grad w.r.t. an array-valued
    gradient needs a scalar contraction; ∇ₓ⟨∇ₓf, c⟩ = Hₓₓ·c).

  NOTE value+grad-of-value+grad does NOT compose: value+grad's walked body
  ends in the VECTOR [primal grads…]; seeding a vector-valued tail silently
  yields zero gradients. This assembly (scalar tail) is the shape that
  composes."
  [f-var param]
  (let [resolved (resolve-deftm-var f-var)
        m (meta resolved)
        params (or (:raster.core/deftm-params m)
                   (throw (ex-info "reified-grad requires a deftm var" {:var f-var})))
        walked-body (or (rcore/ensure-walked-body! resolved)
                        (throw (ex-info "No walked body on var" {:var f-var})))
        tags (or (:raster.core/deftm-tags m) (vec (repeat (count params) 'double)))
        all-params (vec (map-indexed
                         (fn [i p]
                           (let [base (if (symbol? p) p (symbol (name p)))
                                 tag (nth tags i nil)]
                             (if tag
                               (with-meta base {:raster.type/tag tag})
                               (with-meta base nil))))
                         params))
        diff-params (vec (filter #(differentiable-tag? (:raster.type/tag (meta %)))
                                 all-params))
        param-sym (symbol (name param))
        slot-idx (or (first (keep-indexed
                             (fn [i p] (when (= p param-sym) i)) diff-params))
                     (throw (ex-info (str "reified-grad: `" param "` is not a "
                                          "differentiable param of " f-var)
                                     {:param param :diff-params diff-params})))
        param-tag (:raster.type/tag (meta (nth diff-params slot-idx)))
        array-param? (= :array (:kind (tangent/tangent-kind param-tag)))
        ;; The gradient program, assembled exactly as hvp does. ad-prepare
        ;; (lower-composites) and reify-pullback each open their OWN
        ;; with-ad-gensym; run BOTH under one shared counter so reify-pullback
        ;; continues from ad-prepare's high-water mark instead of resetting to 0
        ;; and re-minting colliding anf__ temps. The alpha-rename-bound below
        ;; assumes its input bindings are already unique — it collapses (not
        ;; separates) any duplicate bound name — so the collision must be
        ;; prevented HERE, upstream of the rename.
        {:keys [fwd-bindings result-sym body-sym pullback-form]}
        (call-with-shared-ad-gensym
         (fn [] (reify-pullback (ad-prepare (first walked-body)) diff-params)))
        [_ rev-bindings grad-slots] pullback-form
        grad-slot (nth (vec grad-slots) slot-idx)
        c-sym (when array-param?
                (with-meta (symbol (str "c__ror" (swap! ror-rename-counter inc)))
                  {:raster.type/tag param-tag}))
        tail-bindings (if array-param?
                        [(symbol (str "gdot__ror" (swap! ror-rename-counter inc)))
                         (list 'raster.par/dot-product grad-slot c-sym)]
                        [])
        tail (if array-param? (first tail-bindings) grad-slot)
        bindings (vec (concat fwd-bindings
                              [result-sym body-sym]
                              ['dy__rad 1.0]
                              rev-bindings
                              tail-bindings))
        ;; α-rename: the frozen program must survive a fresh-countered
        ;; second AD pass (see alpha-rename-bound).
        [bindings' tail'] (alpha-rename-bound bindings tail)
        form (list 'let* bindings' tail')
        fn-params (cond-> all-params array-param? (conj c-sym))
        source-ns (or (:ns m) *ns*)
        qualified (inf/qualify-body-symbols form source-ns (set fn-params))
        runtime-fn (make-runtime-value+grad-fn [qualified] fn-params)
        out-tags (cond-> (vec (take (count params) (concat tags (repeat nil))))
                   array-param? (conj param-tag))]
    (with-meta (fn [& args] (apply runtime-fn args))
      {::reified-grad true
       :raster.core/deftm true
       :raster.core/deftm-walked-body [qualified]
       :raster.core/deftm-params fn-params
       :raster.core/deftm-tags out-tags})))

(defn compile-hvp-fn
  "Compile a Hessian-vector product function for a scalar deftm var.

  Returns (fn [args-vec v-vec] -> Hv) where Hv is the Hessian-vector product.
  Uses reverse-over-reverse: differentiates the pullback itself.

  The compiled function computes: H*v = d/dx [v^T * grad(f)(x)]
  where H is the Hessian of f at x.

  This is the key primitive for:
  - Newton's method (solve H*d = -g for direction d)
  - Conjugate gradient on the Hessian
  - Trust region methods"
  [f-var]
  ;; Bind a shared counter for all AD passes (reify-pullback + transform-body)
  (binding [*gensym-counter* (atom 0)]
    (let [resolved (resolve-deftm-var f-var)
          m (meta resolved)
          params (or (:raster.core/deftm-params m)
                     (throw (ex-info "compile-hvp-fn requires a deftm var" {:var f-var})))
          walked-body (or (rcore/ensure-walked-body! resolved)
                          (throw (ex-info "No walked body" {:var f-var})))
          tags (or (:raster.core/deftm-tags m)
                   (vec (repeat (count params) 'double)))
          ;; KEEP each param's :raster.type/tag (§13 A4 fix): the tangent
          ;; protocol reads it to type seeds/zeros — stripping it forced
          ;; untyped zeros downstream.
          all-params (vec (map-indexed
                           (fn [i p]
                             (let [base (if (symbol? p) p (symbol (name p)))
                                   tag (nth tags i nil)]
                               (if tag
                                 (with-meta base {:raster.type/tag tag})
                                 (with-meta base nil))))
                           params))
          ;; Differentiate only w.r.t. params WITH a tangent space (⊥ slots —
          ;; Long dims etc. — get no v slot and no Hv row). All-double scalar
          ;; fns are unchanged: every param stays active.
          active-params (vec (filter #(differentiable-tag?
                                       (:raster.type/tag (meta %)))
                                     all-params))
          n (count active-params)

        ;; Phase 1: Reify the pullback
          reified (reify-pullback (first walked-body) active-params)
          {:keys [pullback-form pullback-params captured-syms
                  fwd-bindings result-sym body-sym]} reified

        ;; Phase 2: Differentiate the pullback w.r.t. captured params
        ;; The pullback computes g(dy, captured...) = [d_p1 ... d_pN]
        ;; We want d/d(captured) of v^T * g(1.0, captured...)
        ;; = d/d(captured) of sum_i v_i * d_p_i
        ;;
        ;; This is the same as running the pullback's own reverse AD
        ;; with adjoint seed = v (one per output component).
        ;;
        ;; But the pullback outputs a VECTOR [d_p1 ... d_pN], so we need
        ;; to form the dot product v^T * [d_p1 ... d_pN] first, THEN
        ;; differentiate that scalar w.r.t. the original params.
        ;;
        ;; Strategy: Build a form that computes v^T * grad(f)(x),
        ;; then differentiate THAT w.r.t. x using standard AD.

        ;; Build the "v-dot-grad" form:
        ;; (let* [...fwd... dy__rad 1.0 ...rev...
        ;;        vdg (+ (* v0 d_p1) (* v1 d_p2) ...)]
        ;;   vdg)
          v-syms (mapv #(with-meta (symbol (str "v__" (name %))) (meta %))
                       active-params)
          [_ rev-bindings-vec rev-body] pullback-form
        ;; rev-body is the [d_p1 d_p2 ...] vector
          grad-syms (vec rev-body)

        ;; Build v·grad — PER-PARAM contraction (§13 A4 fix): scalar params
        ;; contract with numeric/*, ARRAY params with par/dot-product; after
        ;; contraction every term is a scalar, folded with numeric/+.
          array-param? (fn [p]
                         (= :array (:kind (tangent/tangent-kind
                                           (:raster.type/tag (meta p))))))
          _ (when (some array-param? active-params)
              ;; par/dot-product must be loadable when the compiled form
              ;; evals (requiring lazily avoids a reverse→par ns cycle).
              (require 'raster.par))
          dot-terms (mapv (fn [vi gi]
                            (if (array-param? vi)
                              (list 'raster.par/dot-product vi gi)
                              (list 'raster.numeric/* vi gi)))
                          v-syms grad-syms)
          dot-expr (if (= 1 (count dot-terms))
                     (first dot-terms)
                     (reduce (fn [a b] (list 'raster.numeric/+ a b))
                             dot-terms))
          dot-sym (symbol "vdg__hvp")

        ;; Combined form: fwd + dy=1.0 + rev + dot product
          combined-bindings (vec (concat fwd-bindings
                                         [result-sym body-sym]
                                         ['dy__rad 1.0]
                                         rev-bindings-vec
                                         [dot-sym dot-expr]))
          combined-form (list 'let* combined-bindings dot-sym)

        ;; Phase 3: Differentiate the combined form w.r.t. active-params
        ;; This gives us d(v^T * grad(f))/d(x) = H*v
        ;; Counter is shared — reify-pullback already consumed some,
        ;; transform-body continues from there, no collisions.
          hvp-transformed (transform-body combined-form active-params)

        ;; Phase 4: Build the compiled function
        ;; Parameters: [p1 p2 ... v_a v_b ...] — ALL original params (⊥ dims
        ;; are still needed to evaluate) + one v per differentiable param.
          fn-params (vec (concat all-params v-syms))
          compiled-fn (eval (list 'fn fn-params hvp-transformed))]

    ;; Return a function that takes [args-vec v-vec] -> [value Hv-vec]
    ;; (args-vec: all params; v-vec: one entry per DIFFERENTIABLE param;
    ;;  Hv rows align with the differentiable params.)
      (fn [args-vec v-vec]
        (let [all-args (vec (concat args-vec v-vec))
              [_vdg_value pullback] (apply compiled-fn all-args)
            ;; pullback(1.0) gives us [d/dx1, d/dx2, ..., d/dv1, d/dv2, ...]
            ;; We only want the first n entries (derivatives w.r.t. x, not v)
              all-grads (pullback 1.0)]
          (vec (take n all-grads)))))))

(defn compile-hessian-fn
  "Compile a full Hessian matrix function for a scalar deftm var.

  Returns (fn [args-vec] -> [[H11 H12...] [H21 H22...] ...])
  Computes Hessian by n Hessian-vector products with standard basis vectors.

  For large n, prefer compile-hvp-fn (O(n) vs O(n^2) cost)."
  [f-var]
  (let [resolved (resolve-deftm-var f-var)
        params (:raster.core/deftm-params (meta resolved))
        n (count params)
        hvp-fn (compile-hvp-fn f-var)]
    (fn [args-vec]
      (vec (for [i (range n)]
             (let [e-i (vec (concat (repeat i 0.0) [1.0] (repeat (clojure.core/- n i 1) 0.0)))]
               (hvp-fn args-vec e-i)))))))

;; ================================================================
;; Composable AD operators: grad, value+grad
;; ================================================================
;;
;; These return objects that work in two modes:
;; 1. Runtime: call directly as IFn (evaluates AD-transformed code)
;; 2. Compiled: carries ::deftm-walked-body metadata so the compiler
;;    pipeline can inline the AD-transformed body at call sites.
;;
;; This makes AD composable with normal deftm code:
;;   (deftm train-step [W x y lr] :- Double
;;     (let [[loss d_W _ _] ((value+grad my-loss) W x y)]
;;       (axpy! W W (- lr) d_W)
;;       loss))
;;
;; When compiled via compile-aot, the pipeline inlines the
;; AD-transformed body, giving full visibility for optimization.

(defn- undevirtualize
  "Replace .invk calls with their generic dispatch equivalents using :raster.op/original metadata.
  (.invk mangled-impl a b) → (raster.numeric/+ a b)
  This allows Dual numbers to propagate through the walked body."
  [form]
  (cond
    (and (seq? form) (= '.invk (first form)))
    (let [op-sym (or (:raster.op/original (meta form)) (:op (meta form)))]
      (if op-sym
        (apply list op-sym (map undevirtualize (drop 2 form)))
        ;; No :op metadata — keep as-is (e.g., fn-param .invk calls)
        (apply list (map undevirtualize form))))

    (seq? form)
    (with-meta (apply list (map undevirtualize form)) (meta form))

    (vector? form)
    (mapv undevirtualize form)

    :else form))

(defn- strip-primitive-casts
  "Remove (double x), (float x) etc. casts that would fail on Dual numbers.
  Dual dispatch handles type propagation instead."
  [form]
  (cond
    (and (seq? form) (= 2 (count form))
         (contains? #{'double 'float 'long 'int
                      'clojure.core/double 'clojure.core/float
                      'clojure.core/long 'clojure.core/int} (first form)))
    (strip-primitive-casts (second form))

    (seq? form)
    (with-meta (apply list (map strip-primitive-casts form)) (meta form))

    (vector? form)
    (mapv strip-primitive-casts form)

    :else form))

;; alpha-conversion (hygienic rename of every bound var before splicing a form
;; alongside structurally identical siblings — unrolled fold iterations) is the
;; unified `util/alpha-convert`, generic over form/scope-info. Unlike the former
;; private copy (let*/loop*/dotimes/fn* only), it ALSO freshens binders nested in
;; par/ftm/letfn*/reify*/catch bodies — closing the sibling-conflation hole the
;; old version silently passed through.

(defn- hoist-nested-lets
  "Hoist let expressions out of call arguments into a wrapping let*.
  Transforms: (f (let [a e1] a) (let [b e2] b)) → (let* [a e1 b e2] (f a b))
  Recursively processes sub-expressions. Required for AD which expects flat ANF."
  [form]
  (cond
    (not (seq? form)) form

    ;; Already a let/let* — hoist in bindings and body. A binding whose init is
    ;; itself a let (e.g. acc ← (let [...] array-result), as an unrolled fold
    ;; accumulator produces) must be FLATTENED into this let*: splice the inner
    ;; bindings, then bind the sym to the inner let's result. Otherwise the
    ;; nested-let-bound value is mis-typed by the AD (the flat-ANF invariant the
    ;; AD requires covers binding inits, not just call args).
    (and (seq? form) (contains? #{'let 'let*} (first form)))
    (let [[let-sym bindings & body] form
          flat-bindings
          (reduce (fn [acc [sym expr]]
                    (let [he (hoist-nested-lets expr)]
                      (if (and (seq? he) (contains? #{'let 'let*} (first he)))
                        ;; HYGIENE: alpha-convert the inner let (fresh bound names,
                        ;; scope-respecting) BEFORE splicing it into this let*.
                        ;; Without this, sibling flattened lets (e.g. repeated
                        ;; unrolled fold iterations, which reuse the same local
                        ;; names) collide and the reverse AD conflates them —
                        ;; wrong gradients.
                        (let [[_ inner-binds & inner-body] (util/alpha-convert he)]
                          (-> acc (into (vec inner-binds)) (conj sym (last inner-body))))
                        (conj acc sym he))))
                  []
                  (partition 2 bindings))
          hoisted-body (map hoist-nested-lets body)]
      ;; A SINGLE let*-tail merges into this let* (same flat-ANF invariant as
      ;; binding inits): (let* B1 (let* B2 e)) ≡ (let* B1+B2 e) — sequential
      ;; binding semantics are preserved exactly. Loop canonicalization
      ;; produces this shape when a TAIL-position loop rewrites to a let*-
      ;; wrapped par/scan; without the merge the inner let* reaches
      ;; gen-reverse-let as an opaque tail expression (dropped adjoints).
      ;; Only the single-expr case merges: with statement exprs before the
      ;; tail, splicing would move the inner bindings BEFORE the statements'
      ;; effects (e.g. an aset the bindings read).
      (if (and (= 1 (count hoisted-body))
               (seq? (first hoisted-body))
               (contains? #{'let 'let*} (ffirst hoisted-body)))
        (let [[_ inner-binds & inner-body] (util/alpha-convert (first hoisted-body))]
          (apply list let-sym (into flat-bindings (vec inner-binds)) inner-body))
        (apply list let-sym flat-bindings hoisted-body)))

    ;; Scope-introducing forms — recurse but don't hoist across scope boundaries.
    ;; This includes every par/* SOAC (form/introduces-scope?): their bodies
    ;; close over the loop index/accumulator, so splicing a body let* out into
    ;; the wrapping let would free those scoped vars (unresolvable `i` after
    ;; alpha-conversion) — and the SOAC reverse analyzers expect the body's
    ;; let*-bound agets IN PLACE (analyze-par-map-body).
    (and (seq? form) (or (contains? #{'if 'loop 'loop* 'fn* 'do 'when 'recur 'dotimes} (first form))
                         (form/introduces-scope? form)))
    (with-meta (apply list (first form) (map hoist-nested-lets (rest form))) (meta form))

    ;; Function call — check for let expressions in arguments
    (seq? form)
    (let [f (first form)
          args (rest form)
          hoisted-args (map hoist-nested-lets args)
          ;; Collect let bindings from args and replace with their body
          collected (reduce
                     (fn [{:keys [bindings clean-args]} arg]
                       (if (and (seq? arg) (contains? #{'let 'let*} (first arg)))
                         ;; HYGIENE: alpha-convert the inner let (fresh bound
                         ;; names, scope-respecting) BEFORE splicing its bindings
                         ;; into the shared outer let* — same rule as the
                         ;; binding-init and let-tail branches above. Two sibling
                         ;; let-args that reuse the same local name (e.g. repeated
                         ;; unrolled fold iterations) would otherwise collide in
                         ;; the outer let* and the reverse AD would conflate them
                         ;; (later binding shadows the earlier → wrong gradients).
                         (let [[_ inner-bindings & inner-body] (util/alpha-convert arg)]
                           {:bindings (into bindings (vec inner-bindings))
                            :clean-args (conj clean-args (last inner-body))})
                         {:bindings bindings
                          :clean-args (conj clean-args arg)}))
                     {:bindings [] :clean-args []}
                     hoisted-args)]
      (if (seq (:bindings collected))
        (list 'let* (vec (:bindings collected))
              (apply list f (:clean-args collected)))
        (apply list f hoisted-args)))

    :else form))

(defn- lift-loop-to-tail
  "Restructure (let* [pre... loop-sym (loop ...) post...] body) so the loop
  becomes the let's tail expression — the canonical shape gen-reverse-loop-with-let
  expects. Inlines the post-bindings + body into the loop's result branch.

  When called on something that isn't a let* with a loop in binding position,
  returns the form unchanged. Handles only the FIRST loop binding found —
  recurse from the caller if multiple loops in series need lifting.

  Loop body shape required: (if test (recur ...) result-expr) or
                            (if test result-expr (recur ...))."
  [form]
  (if-not (and (seq? form) (#{'let 'let*} (first form)))
    form
    (let [[let-sym bindings & body-exprs] form
          pairs (partition 2 bindings)
          loop-binding? (fn [[_ init]]
                          (and (seq? init) (#{'loop 'loop*} (first init))))
          [pre-pairs post] (split-with (complement loop-binding?) pairs)
          [loop-pair & post-pairs] post]
      (if-not loop-pair
        form
        (let [[loop-sym loop-form] loop-pair
              [_ loop-bindings & loop-body] loop-form
              loop-body-expr (if (= 1 (count loop-body))
                               (first loop-body)
                               (cons 'do loop-body))]
          (if-not (and (seq? loop-body-expr) (= 'if (first loop-body-expr)))
            form    ;; Non-canonical loop body — leave alone
            (let [[_ test-expr then-expr else-expr] loop-body-expr
                  recur? (fn [e]
                           (and (seq? e) (= 'recur (first e))))
                  letted-recur? (fn [e]
                                  (and (seq? e) (#{'let 'let*} (first e))
                                       (recur? (last e))))
                  has-recur? (fn [e] (or (recur? e) (letted-recur? e)))
                  [recur-branch result-branch recur-in-then?]
                  (cond (has-recur? then-expr) [then-expr else-expr true]
                        (has-recur? else-expr) [else-expr then-expr false]
                        :else                  [nil nil nil])]
              (if-not recur-branch
                form    ;; Couldn't identify the recur branch
                (let [;; The loop's "value" gets bound to loop-sym, then
                      ;; post-pairs and body run with that binding visible.
                      cont-bindings (vec (concat
                                          [loop-sym result-branch]
                                          (mapcat identity post-pairs)))
                      cont (apply list 'let* cont-bindings body-exprs)
                      new-loop-body (if recur-in-then?
                                      (list 'if test-expr recur-branch cont)
                                      (list 'if test-expr cont recur-branch))
                      new-loop-form (apply list 'loop* loop-bindings [new-loop-body])]
                  (if (seq pre-pairs)
                    (apply list let-sym
                           (vec (mapcat identity pre-pairs))
                           [new-loop-form])
                    new-loop-form))))))))))

(defn- body-tail-tag
  "The :raster.type/tag of a walked body's TAIL expression — the type of the
  actual primal the differentiated graph produces (which may be narrower than
  the deftm's declared return type, e.g. a `:- Double` fn whose body is a
  parametric T=float chain). Prefers the form's own tag, else descends
  through binding forms to the final body expression."
  [form]
  (or (:raster.type/tag (meta form))
      (when (and (seq? form) (form/binding-form? form) (seq (drop 2 form)))
        (body-tail-tag (last form)))))

(defn- anf-value-temp?
  "True iff `sym` is an ANF *value* temp: a name of the exact form `anf__<N>`
  minted by the ANF normalizer (`(gensym-fn \"anf\")`, reverse/normalize.clj).
  These name single-assignment SSA subexpression values and MUST be globally
  unique by construction.

  Deliberately NARROW — the duplicate check is scoped to these only:
    - USER-level names (which `let*` may legitimately shadow — two nested
      `(let [t ...] ...)` flatten to two `t` bindings) are excluded.
    - ADJOINT accumulators (`d_x__N`, `dg_x__N`, `d_out__N`, `d_anf__3__N`,
      …) are LEGITIMATELY rebound by reverse-mode gradient accumulation
      (`d ← (grad-acc d contribution)`) and are excluded by the `^anf__`
      anchor (`d_anf__3__7` starts with `d_`, not `anf__`)."
  [sym]
  (boolean (re-matches #"anf__\d+" (name sym))))

(defn- assert-unique-ad-minted-binds!
  "Fail LOUD if any ANF value temp (`anf__N`) is bound twice in the flat `let*`
  binding vector. Such a duplicate can only happen if two reverse lowering
  phases reset *gensym-counter* to 0 and re-minted the same `anf__N` for two
  UNRELATED subexpressions — the flat `let*` then silently SHADOWS the first
  binding, so a template backward reads the wrong value/dimension and the
  gradient is quietly miscompiled (the class of bug fixed by wrapping the whole
  reverse lowering in ONE `with-ad-gensym`). Scoped to `anf__` value temps,
  which are single-assignment by construction; legitimate user-level shadowing
  and adjoint re-accumulation are left alone."
  [bindings f-var]
  (let [dups (->> (take-nth 2 bindings)
                  (filter anf-value-temp?)
                  frequencies
                  (keep (fn [[s n]] (when (> n 1) s)))
                  vec)]
    (when (seq dups)
      (throw (ex-info
              (str "Reverse-AD lowering produced duplicate gensym-minted bound "
                   "name(s) " dups " for " f-var " — this indicates a "
                   "*gensym-counter* collision: two lowering phases (e.g. "
                   "ad-prepare/lower-composites and transform-body) each reset "
                   "the counter to 0 and re-minted the same anf__/adjoint temp. "
                   "The flat let* silently shadows the first binding and "
                   "miscompiles the gradient. The whole reverse lowering must run "
                   "under ONE with-ad-gensym scope so every generated name is "
                   "globally unique.")
              {:var f-var :duplicates dups :kind :ad-gensym-counter-collision})))))

(defn- build-grad-walked-body
  "Build the AD-transformed walked body for a deftm var.
  Returns {:walked-body [form], :params [sym ...], :source-ns ns}
  where form is a flat let* returning [primal grad1 ... gradN].
  Returns nil if AD transform fails."
  [f-var]
  ;; ONE shared gensym counter across the whole reverse lowering. ad-prepare
  ;; (lower-composites) and transform-body each open their OWN with-ad-gensym;
  ;; without this enclosing scope each RESETS *gensym-counter* to 0 and re-emits
  ;; anf__1/anf__2/anf__3…, so a binding-position ANF temp (e.g. linear-nb's
  ;; `(* nq hd)` out arg) and a tail-position ANF temp collide on the SAME name
  ;; in the flat let* → the second SHADOWS the first → a template backward reads
  ;; the wrong dimension (gqa dW came out [hd,in] not [nq*hd,in]). Sharing one
  ;; monotonic counter keeps every generated name globally unique.
  (with-ad-gensym
    (let [resolved (resolve-deftm-var f-var)
          m (meta resolved)
          params (or (:raster.core/deftm-params m)
                     (throw (ex-info "grad requires a deftm var" {:var f-var})))
          walked-body (or (rcore/ensure-walked-body! resolved)
                          (throw (ex-info "No walked body on var" {:var f-var})))
        ;; TUPLE-VALUED OUTPUT GUARD (#74, framework §14): grad/value+grad is
        ;; only defined for SCALAR-valued functions — a vector tail (e.g. the
        ;; [primal grads...] of a value+grad result, or a tuple-returning
        ;; deftm) has a PRODUCT cotangent space with no canonical seed, and
        ;; the reverse fold would silently return zero gradients. Fail loud.
        ;; The transparent-composition idiom is to USE the components in a
        ;; scalar program: (let [vg ((value+grad #'f) x)] ...scalar of (nth vg i)...).
          _ (let [wb-form (first walked-body)
                  tail (if (and (seq? wb-form)
                                (contains? #{'let 'let*} (first wb-form)))
                         (last wb-form)
                         wb-form)]
              (when (vector? tail)
                (throw (ex-info
                        (str "Cannot differentiate the tuple-valued function `" f-var
                             "` — its output is a vector, which has no canonical "
                             "cotangent seed (this is what value+grad returns; "
                             "value+grad of value+grad is ill-typed). Compose "
                             "transparently instead: call the inner gradient in a "
                             "scalar deftm and differentiate that, e.g. "
                             "(let [vg ((value+grad #'f) x)] ...scalar use of (nth vg i)...). "
                             "For 1-param scalar fns, (grad (grad f)) composes directly.")
                        {:var f-var :tail-shape :vector :reason :tuple-valued-output}))))
          tags (or (:raster.core/deftm-tags m) (vec (repeat (count params) 'double)))
          all-params (vec (map-indexed
                           (fn [i p]
                             (let [tag (nth tags i nil)
                                   base (if (symbol? p) p (symbol (name p)))]
                               (if tag
                                 (with-meta base {:raster.type/tag tag})
                                 (with-meta base nil))))
                           params))
        ;; Only differentiable-tag params seed activity analysis. Non-diff
        ;; params (Long/longs scalars, indices, etc.) are constants for AD —
        ;; expressions involving only them stay inactive and don't need AD
        ;; rules. Their grad slots are still emitted in the output vector
        ;; (via all-params) so positional consumers don't shift.
          diff-active-params (vec (keep-indexed
                                   (fn [i p]
                                     (when (differentiable-tag? (nth tags i nil)) p))
                                   all-params))
          source-ns (or (:ns m) *ns*)
        ;; Π: the SEED is the cotangent of the RESULT of the differentiated
        ;; GRAPH, so its dtype derives from the walked body's TAIL tag (the
        ;; actual primal flowing) — not from a global binary any-param-is-float
        ;; switch, and NOT from the declared return tag: a fn declared
        ;; `:- Double` whose body is a parametric float chain (T=float
        ;; cross-entropy) needs a FLOAT seed for its pullback; the Double is
        ;; only the .invk interface boundary. The legacy binary inference
        ;; remains as fallback for untagged tails.
          dtype (case (body-tail-tag (first walked-body))
                  float  :float
                  double :double
                  (if (some #{'floats 'float} tags) :float :double))
        ;; Shared pre-AD preparation (lower composites → materialize → hoist
        ;; into flat ANF) — see ad-prepare, shared with the JVP path.
          hoisted (ad-prepare (first walked-body))
        ;; transform-body itself will lift any binding-position loop into
        ;; tail position via lift-loop-to-tail.
          ad-form (transform-body hoisted diff-active-params)
          flat (binding [ad-flatten/*flatten-dtype* dtype]
                 (ad-flatten/flatten-for-gradient ad-form))
        ;; Pad the gradient output vector so positional consumers see one slot
        ;; per ORIGINAL param (loss + N grads), with nil where the param was
        ;; non-differentiable. flat's inner form is (let* bindings [primal d1
        ;; d2 ... dN_diff]); we rewrite it to [primal slot1 ... slotN_all].
          padded-form
          (when flat
            (let [diff->idx (zipmap diff-active-params (range))
                  grad-slots (mapv (fn [p]
                                     (if-let [j (diff->idx p)]
                                       (nth (:param-adj-syms flat) j)
                                       nil))
                                   all-params)
                  output-vec (vec (cons (:result-sym flat) grad-slots))
                  [op bindings _] (:form flat)]
            ;; FAIL-LOUD backstop: the flat let* must have globally-unique
            ;; gensym-minted bound names. A duplicate means a *gensym-counter*
            ;; collision slipped through (the enclosing with-ad-gensym above is
            ;; what prevents it) — would silently miscompile the gradient.
              (assert-unique-ad-minted-binds! bindings f-var)
              (list op bindings output-vec)))]
      (when flat
        {:walked-body [padded-form]
         :params all-params
         :tags tags
         :source-ns source-ns
         :result-sym (:result-sym flat)
         :param-adj-syms (:param-adj-syms flat)}))))

(defn- make-runtime-value+grad-fn
  "Build a runtime IFn from the AD-transformed walked body.
  Uses (& args) + indexed unpack so deftms with >20 params (Clojure's fn
  positional-arg limit) still work. The unpacking lets are trivially
  eliminated by the JVM JIT for the common ≤20-param case."
  [walked-body params]
  ;; The body is eval'd WITHOUT type hints, so `clojure.core/alength` on an
  ;; untyped INTERMEDIATE array (e.g. a broadcast result `y`) reflects ambiguously
  ;; ("more than one matching method: alength"). Rewrite to the polymorphic
  ;; `raster.arrays/alength`, which dispatches on the runtime array type — no
  ;; reflection, and params (already typed) keep working.
  (let [body (walk/postwalk-replace {'clojure.core/alength 'raster.arrays/alength}
                                    (first walked-body))]
    (if (<= (count params) 20)
      (eval (list 'fn (vec params) body))
      (let [args-sym (gensym "args__")
            unpack (vec (mapcat (fn [p i] [p `(nth ~args-sym ~i)])
                                params (range)))]
        (eval (list 'fn ['& args-sym]
                    (list 'let unpack body)))))))

;; ================================================================
;; Forward-mode admissibility (framework §11): mode selection is
;; constrained by CARRIER COVERAGE — an interpretation is admissible
;; only when every op that would execute on the carrier has a lift in
;; that carrier. The coverage matrix is a queryable artifact derived
;; from the dispatch registry (the single source of truth for what a
;; carrier accepts) — never documentation, never a hardcoded op list.
;; ================================================================

(def ^:private forward-neutral-namespaces
  "Op namespaces that never execute on the Dual carrier: clojure.core is
  integer index/counter arithmetic by convention (see CLAUDE.md), and
  raster.arrays is array plumbing (forward mode is scalar-only — array
  params are rejected separately)."
  #{"clojure.core" "raster.arrays"})

(def ^:private forward-interop-namespaces
  "JVM static-method namespaces: primitive interop calls cannot accept a
  Dual, so their presence in a body makes forward mode inadmissible."
  #{"Math" "java.lang.Math" "StrictMath" "java.lang.StrictMath"})

(defn- collect-op-heads
  "All semantic op heads in a walked-body form. For devirtualized
  (.invk impl args...) calls this reads the walk's stamped original-op
  metadata via op/semantic-op (never parses mangled names); a bare .invk
  without metadata (typed fn-param call) contributes nothing."
  [form]
  (let [acc (volatile! #{})]
    (letfn [(go [f]
                (cond
                  (seq? f)
                  (do (when-let [h (if (= '.invk (first f))
                                   ;; same recovery order as undevirtualize
                                     (or (:raster.op/original (meta f)) (:op (meta f)))
                                     (op/semantic-op f))]
                        (when (symbol? h) (vswap! acc conj h)))
                      (doseq [x f] (go x)))
                  (vector? f) (doseq [x f] (go x))
                  (map? f) (doseq [[k v] f] (go k) (go v))))]
      (go form))
    @acc))

(defn- dual-tag?
  "True if a dispatch tag denotes the Dual carrier (short `Dual` for explicit
  overloads, fully qualified `raster.ad.forward.Dual` for auto-specialized
  parametric entries). Dual__Sym etc. do NOT count — they are other carriers."
  [tag]
  (cond
    (symbol? tag) (= "Dual" (peek (str/split (name tag) #"\.")))
    (class? tag) (= "Dual" (.getSimpleName ^Class tag))
    :else false))

(defn- dual-lift?
  "True when a deftm generic var can accept a Dual argument: an explicit
  Dual overload in its dispatch table, or a parametric (All [T]) template
  with a bare type-var param (auto-specializes for Dual on first dispatch,
  e.g. raster.sci.special/sinc)."
  [v qualified-op]
  (or (when-let [dt (:raster.core/dispatch-table (meta v))]
        (boolean (some (fn [entry] (some dual-tag? (:tags entry)))
                       (mapcat val @dt))))
      (boolean (some (fn [entry]
                       (some #(contains? (:type-vars entry) %)
                             (:annotations entry)))
                     (get @dispatch/parametric-registry qualified-op)))))

(defn- forward-op-status
  "Classify one op head for execution on the Dual carrier.
  Returns :covered, :neutral (never carries a Dual), or :uncovered."
  [op-sym]
  (let [ns-str (namespace op-sym)
        nm (name op-sym)]
    (cond
      ;; Unqualified heads: special forms, binders, locals, constructors,
      ;; interop field/method access (.v/.partials/...) — structural, not
      ;; Dual-carrying calls (the walk qualifies every deftm call head).
      (nil? ns-str) :neutral

      ;; JVM static interop (Math/tan ...) — no Dual lift is possible.
      (contains? forward-interop-namespaces ns-str) :uncovered

      (contains? forward-neutral-namespaces ns-str) :neutral

      :else
      (let [base (mangled/extract-deftm-base
                  (symbol ns-str (mangled/strip-impl-suffix nm)))
            v (try (resolve base) (catch Exception _ nil))]
        (cond
          (nil? v) :neutral
          ;; deftm generic (dispatch table or parametric template): the
          ;; registry decides whether the Dual carrier is accepted.
          (or (:raster.core/dispatch-table (meta v))
              (contains? @dispatch/parametric-registry base))
          (if (dual-lift? v base) :covered :uncovered)
          ;; Plain var (defn helper, constant) — not carrier-dispatched.
          :else :neutral)))))

(def ^:private array-param-tags
  '#{doubles floats ints longs shorts bytes booleans chars objects})

(defn forward-coverage
  "Queryable Dual-carrier coverage for a deftm var (framework §4a/§11).
  Walks the deftm's walked body, collects semantic op heads, and checks
  each op that would execute on the Dual carrier against the dispatch
  registry for a Dual lift.

  Returns {:admissible?   bool
           :uncovered-ops sorted vector of op symbols without a Dual lift
           :array-params  params whose type is an array (forward mode is
                          scalar-only — arrays have no Dual seeding)}"
  [f-var]
  (let [resolved (resolve-deftm-var f-var)
        m (meta resolved)
        walked-body (or (rcore/ensure-walked-body! resolved)
                        (throw (ex-info "No walked body on var" {:var f-var})))
        params (or (:raster.core/deftm-params m) [])
        tags (or (:raster.core/deftm-tags m) [])
        array-params (vec (remove nil?
                                  (map (fn [p tag]
                                         (when (or (contains? array-param-tags tag)
                                                   (and (seq? tag) (= 'Array (first tag))))
                                           p))
                                       params (concat tags (repeat nil)))))
        uncovered (->> (collect-op-heads (first walked-body))
                       (filter #(= :uncovered (forward-op-status %)))
                       sort
                       vec)]
    {:admissible? (and (empty? uncovered) (empty? array-params))
     :uncovered-ops uncovered
     :array-params array-params}))

(defn forward-admissible?
  "True when forward mode (the Dual carrier) is admissible for f-var:
  every op in the walked body that would execute on the Dual carrier has
  a Dual lift in the dispatch registry, and no param is array-typed."
  [f-var]
  (:admissible? (forward-coverage f-var)))

(defn ^clojure.lang.IFn value+grad
  "Composable value+gradient operator.

  (value+grad f) returns a function that computes [value, grad1, ..., gradN].
  Works in two modes:
  - Runtime: callable directly as (fn [args...] -> [value grads...])
  - Compiled: carries deftm metadata so the pipeline can inline the
    AD-transformed body when called inside a compiled deftm.

  Options:
    :mode  - :reverse (default), :forward, or :auto

  Usage:
    ;; Runtime
    ((value+grad #'my-loss) 3.0 4.0)  ;; => [25.0 6.0 8.0]

    ;; Inside compiled deftm
    (deftm train-step [W x y lr] :- Double
      (let [[loss d_W _ _] ((value+grad #'my-loss) W x y)]
        (axpy! W W (- lr) d_W)
        loss))"
  ([f-var] (value+grad f-var :mode :reverse))
  ([f-var & {:keys [mode] :or {mode :reverse}}]
   (case mode
     :reverse
     (let [bgw (build-grad-walked-body f-var)
           ;; Fail loud, never return a broken IFn: flatten-for-gradient returns
           ;; nil when the AD form isn't the flat [primal (fn* [dy] ...)] shape —
           ;; today that's a body that IS (or tail-lifts to) a raw loop, whose
           ;; loop-with-let output the runtime assembler can't flatten yet
           ;; (laws-suite finding 2026-07-04; the compiled/inline path handles it).
           _ (when (nil? bgw)
               (throw (ex-info
                       (str "value+grad: cannot assemble a runtime gradient for `"
                            f-var "` — the body reduces to a loop form the runtime "
                            "flattener does not support (canonical fixed-trip loops "
                            "are lifted to SOACs automatically; this one declined "
                            "the soundness gates, e.g. a data-dependent trip count "
                            "or a multi-carry body). Express the reduction via "
                            "`par/reduce`, the recurrence via `par/scan` (the "
                            "sanctioned differentiable recurrence — out[i] = acc_i "
                            "is its own tape), a data-dependent/convergence loop via "
                            "a fixed-point solve (adjoint-of-fixed-point rule), or a "
                            "registered op (e.g. a loss deftm with an AD template), "
                            "or use the compiled path (compile-aot of a deftm "
                            "calling value+grad).")
                       {:var f-var :reason :flatten-failed})))
           {:keys [walked-body params tags source-ns]} bgw
           runtime-fn (make-runtime-value+grad-fn walked-body params)
           ;; Qualify symbols in walked body for inlining from other namespaces
           inline-ns inf/qualify-body-symbols
           param-set (set params)
           qualified-wb (mapv #(inline-ns % source-ns param-set) walked-body)]
       ;; Return an IFn that also carries deftm metadata for the compiler
       (let [result-fn (fn [& args] (apply runtime-fn args))]
         (with-meta result-fn
           {::value+grad true
            :raster.core/deftm true
            :raster.core/deftm-walked-body qualified-wb
            :raster.core/deftm-params params
            :raster.core/deftm-tags tags})))

     :forward
     ;; Forward-mode gradient via Dual numbers on the walked body.
     ;; Un-devirtualize .invk → generic dispatch so Dual numbers propagate.
     ;; Admissibility is checked at CONSTRUCTION time (framework §11): fail
     ;; with the uncovered ops named, never a No-matching-method at call time.
     (let [cov (forward-coverage f-var)
           _ (when-not (:admissible? cov)
               (throw (ex-info
                       (str "value+grad :mode :forward is not admissible for `"
                            f-var "` — "
                            (when (seq (:uncovered-ops cov))
                              (str "ops without a Dual lift: "
                                   (str/join ", " (:uncovered-ops cov)) ". "))
                            (when (seq (:array-params cov))
                              (str "array-typed params (forward mode is scalar-only): "
                                   (str/join ", " (:array-params cov)) ". "))
                            "Mode selection is constrained by carrier coverage "
                            "(framework §11): use :mode :reverse, or add the "
                            "missing Dual overloads in raster.ad.forward.")
                       {:var f-var
                        :uncovered-ops (:uncovered-ops cov)
                        :array-params (:array-params cov)})))
           resolved (resolve-deftm-var f-var)
           m (meta resolved)
           params (or (:raster.core/deftm-params m)
                      (throw (ex-info "grad requires a deftm var" {:var f-var})))
           walked-body (or (rcore/ensure-walked-body! resolved)
                           (throw (ex-info "No walked body on var" {:var f-var})))
           tags (or (:raster.core/deftm-tags m) (vec (repeat (count params) 'double)))
           ;; Build a generic-dispatch version of the walked body
           generic-body (undevirtualize (first walked-body))
           ;; Strip primitive casts — Dual numbers aren't primitives
           clean-body (strip-primitive-casts generic-body)
           active-params (vec (map #(with-meta (if (symbol? %) % (symbol (name %))) nil) params))
           generic-fn (eval (list 'fn active-params clean-body))
           n (count params)
           make-dual fwd/->Dual
           dual-class (Class/forName "raster.ad.forward.Dual")
           get-v (fn [d] (.get (.getField dual-class "v") d))
           get-partials (fn [d] (.get (.getField dual-class "partials") d))]
       (fn [& args]
         (let [arg-vec (vec args)
               value (apply generic-fn arg-vec)
               grads (double-array n)]
           ;; One forward pass per parameter, seeding with Dual(val, 1.0)
           (dotimes [i n]
             (let [dual-args (into []
                                   (map-indexed
                                    (fn [j a]
                                      (if (== j i)
                                        (make-dual (double a) (double-array [1.0]))
                                        (make-dual (double a) (double-array [0.0]))))
                                    arg-vec))
                   result (apply generic-fn dual-args)
                   deriv (if (.isInstance dual-class result)
                           (clojure.core/aget ^doubles (get-partials result) 0)
                           0.0)]
               (clojure.core/aset grads i deriv)))
           (vec (cons (if (.isInstance dual-class value) (get-v value) value)
                      (seq grads))))))

     :auto
     ;; Mode selection = argmin cost over ADMISSIBLE interpretations
     ;; (framework §7 + §11). The Griewank dimension test (forward when
     ;; n_inputs ≤ n_outputs; reverse otherwise — for n = 1 forward avoids
     ;; tape overhead) is the cost heuristic; carrier coverage is the HARD
     ;; filter: forward is only selectable when every op in the body has a
     ;; Dual lift. Reverse is always admissible for deftm bodies (templates
     ;; + composite inlining), so it is the fallback.
     (let [resolved (resolve-deftm-var f-var)
           n-params (count (or (:raster.core/deftm-params (meta resolved)) []))
           selected (if (and (<= n-params 1) (forward-admissible? f-var))
                      :forward
                      :reverse)]
       (value+grad f-var :mode selected)))))

(defn grad
  "Composable gradient operator.

  (grad f) returns a function that computes [grad1, ..., gradN] — or, for a
  SINGLE-param scalar function, the bare scalar derivative g (JAX semantics).
  The 1-param scalar tail is what makes the textbook composition
  (grad (grad f)) = f'' well-typed: grad of f : R -> R IS f' : R -> R, so it
  can be differentiated again directly (framework §14). Multi-param grads
  remain vector-valued; re-differentiating those requires using the components
  in a scalar program (the transparent-composition idiom).

  Options:
    :mode  - :reverse (default), :forward, or :auto

  Usage:
    ((grad #'my-loss) 3.0 4.0)   ;; => [6.0 8.0]
    ((grad #'square) 3.0)        ;; => 6.0        (1-param: bare scalar)
    ((grad (grad #'cube)) 2.0)   ;; => 12.0       (f'' composes directly)"
  ([f-var] (grad f-var :mode :reverse))
  ([f-var & {:keys [mode] :or {mode :reverse}}]
   (let [vg-fn (value+grad f-var :mode mode)
         vg-meta (meta vg-fn)]
     ;; Build a grad-specific walked body that drops the primal
     (if-let [wb (:raster.core/deftm-walked-body vg-meta)]
       (let [params (:raster.core/deftm-params vg-meta)
             tags (:raster.core/deftm-tags vg-meta)
             ;; Rewrite walked body: [primal g1 g2 ...] → [g1 g2 ...], and for
             ;; the 1-param case → the bare scalar g1 (scalar tail: composable).
             grad-wb (mapv (fn [form]
                             (if (and (seq? form) (#{'let 'let*} (first form)))
                               (let [[let-sym bindings & body-exprs] form
                                     last-body (last body-exprs)]
                                 (if (vector? last-body)
                                   (let [grads (vec (rest last-body))
                                         tail (if (= 1 (count grads))
                                                (first grads)
                                                grads)]
                                     (list* let-sym bindings
                                            (concat (butlast body-exprs) [tail])))
                                   form))
                               form))
                           wb)
             runtime-fn (make-runtime-value+grad-fn grad-wb params)]
         (with-meta
           (fn [& args] (apply runtime-fn args))
           {::grad true
            :raster.core/deftm true
            :raster.core/deftm-walked-body grad-wb
            :raster.core/deftm-params params
            :raster.core/deftm-tags tags}))
       ;; Fallback: no walked body (forward mode)
       (fn [& args]
         (vec (rest (apply vg-fn args))))))))
