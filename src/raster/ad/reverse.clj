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
            [raster.ad.purity :as purity]
            [raster.numeric :as numeric]
            [raster.ad.templates :as tmpl]
            [raster.ad.product-ports :as product-ports]
            [raster.compiler.ad.bind-ctx :as bind-ctx]
            [raster.compiler.ad.flatten :as ad-flatten]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.ordered-call :as ordered-call]
            [raster.compiler.core.value-completion :as value-completion]
            [raster.compiler.core.value-product :as value-product]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.call-boundary :as call-boundary]
            [raster.compiler.backend.jvm.source-owner :as source-owner]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.ad.reverse.normalize :as anf]
            [raster.ad.tangent :as tangent]
            [raster.analysis.memory :as memory]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.scalar.soa-lower :as soa-lower]
            [raster.compiler.passes.scalar.local-products :as local-products]
            [raster.compiler.passes.parallel.materialize :as materialize]
            [raster.compiler.passes.parallel.loop-lift :as loop-lift]
            [raster.compiler.passes.parallel.patterns :as patterns]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.support.mangled :as mangled]
            [raster.core :as rcore]
            [raster.arrays :as arrays]
            ;; Generated pullbacks can call residual-add and other layer helpers.
            ;; Their namespace must exist when the generated function is eval'd.
            [raster.dl.nn]))

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

(def ^:private ^:dynamic *constant-gradient-arrays* #{})

(def ^:private ^:dynamic *source-memory-context*
  "Lexically visible values and retained declarations at the current AD scope.
   Child fact catalogs do not extend visibility or prove freshness."
  {:values [] :value-facts {} :replayed {} :unprojectable {} :aliases {}})

(def ^:private ^:dynamic *generated-product-context* nil)

(defn- gradient-bearing-array-read?
  "Only reads from differentiated arrays need scatter cotangents. Reads from
  explicitly constant parameter arrays stay ordinary scalar expressions."
  [form]
  (and (op/aget-call? form)
       (= 2 (count (op/call-args form)))
       (not (contains? *constant-gradient-arrays* (first (op/call-args form))))))

(defn- ad-gensym
  "Generate a unique symbol for AD-generated code.
  Optional type-tag stamps :raster.type/tag metadata so downstream
  passes can read types without re-inference."
  ([prefix]
   (loop [id (symbol (str prefix "__" (swap! *gensym-counter* inc)))]
     (if-let [used (:used-identifiers *generated-product-context*)]
       (if (contains? @used id)
         (recur (symbol (str prefix "__" (swap! *gensym-counter* inc))))
         (do (swap! used conj id) id))
       id)))
  ([prefix type-tag]
   (let [sym (ad-gensym prefix)]
     (if type-tag
       (vary-meta sym assoc :raster.type/tag type-tag)
       sym))))

;; ----------------------------------------------------------------
;; Lost-tag enforcement (plan Phase 2 / D2, .internal/ad_typed_emission_plan.md).
;; These sites used to default an untagged cotangent's shadow array/scalar to
;; 'doubles/'double (silently narrowing f32 → f64: the [F/[D ClassCast + F5 GPU
;; under-devirtualization root), warn-first while the lost-tag corpus was
;; collected. The instrumented full-suite run (2026-07-11, 1804 tests /
;; 31230 assertions, fresh JVM incl. all resident GPU compiles) measured
;; ZERO activations at every site — under the Phase-1 Π emission invariant
;; the primal tag is always present, so the defaults are provably dead and
;; now FAIL LOUD instead: a fire means an upstream forward-path tag loss that
;; must be fixed at its source (walker/inference), never papered over in AD.
(defn- lost-tag!
  "Fail-loud replacement for the retired warn-default-tag! 'doubles/'double
  default sites: a reverse-AD codegen site needed the primal binding's
  :raster.type/tag and it is absent."
  [site binding-hint]
  (throw (ex-info (str "reverse-AD lost tag at " site " (binding " binding-hint
                       "): primal carries no :raster.type/tag. This default site "
                       "was retired (Π emission invariant, plan D2) — fix the "
                       "upstream forward-path tag loss; do not re-add a silent "
                       "'doubles/'double default (f32→f64 narrowing hazard).")
                  {:site site :binding (str binding-hint)})))

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
    ;; two doubles: the common case of a scalar adjoint, added directly — the
    ;; generic `numeric/+` below dispatches at runtime, which dominated the
    ;; cost of every scalar accumulation in a backward pass
    (and (instance? Double a) (instance? Double b)) (clojure.core/+ (double a) (double b))
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

(def ^:private approved-grad-acc-definition
  ;; Capture the library's own definition, never whichever root first prepares an AD owner.
  ;; Ordinary def deliberately refreshes this premise together with a legitimate source reload.
  {:var #'grad-acc :root (.getRawRoot ^clojure.lang.Var #'grad-acc) :metadata (meta #'grad-acc)})

(defn ^:no-doc approved-grad-acc-definition?
  "Whether the exact library-owned helper definition still authorizes scalar addition projection.
   This is a local definition premise, not a registry or protection against arbitrary source edits."
  []
  (let [v (ns-resolve 'raster.ad.reverse 'grad-acc)]
    (boolean
     (and (identical? v (:var approved-grad-acc-definition))
          (.hasRoot ^clojure.lang.Var v)
          (identical? (:root approved-grad-acc-definition) (.getRawRoot ^clojure.lang.Var v))
          (= (:metadata approved-grad-acc-definition) (meta v))))))

;; A common retained tag describes the payload carrier, not its presence, and
;; mixed scalar/array operands broadcast to the array, so no single arg decides.
(op/register-result-type! 'raster.ad.reverse/grad-acc :common-arg-tag)

(defn aget-grad
  "Gradient of array element read: y = arr[i].
  Returns a zero array of same size as arr, with dy at position i.
  d_arr[i] = dy, d_arr[j] = 0 for j != i."
  [arr i dy]
  (let [n (long (arrays/alength arr))
        g (arrays/alloc-like arr n)]
    (arrays/aset g (int i) dy)
    g))

;; The cotangent buffer has the primal array's type; without this facet every
;; downstream read of it is an untyped (reflective) aget.
(op/register-result-type! 'raster.ad.reverse/aget-grad :same-as-first-arg)

;; ================================================================
;; Normalization: flatten form for uniform AD processing
;; ================================================================
;; After normalization:
;; - The body is a single symbol
;; - Conditional arms remain lexical regions; only the selected arm executes
;; - A branch-flag binding stores a nontrivial condition result
;; - Reverse records retain the selected arm's forward values, never replay its primal

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

(defn ^:no-doc tangent-parameter-plan
  "Classify an ordered parameter boundary without treating missing type evidence as NoTangent.
   Default selection includes every parameter, so any Unknown refuses before transformation.
   Explicit :wrt exclusion is authoritative; explicitly selecting an Unknown or NoTangent slot
   also refuses before transformation."
  [params tags wrt]
  (let [parameter-count (count params)
        _ (when-not (= parameter-count (count tags))
            (throw (ex-info "Parameter tags must align exactly with parameters"
                            {:reason :ad-parameter-tags
                             :parameter-count parameter-count :tags tags})))
        _ (when-not (or (nil? wrt) (sequential? wrt))
            (throw (ex-info ":wrt must be a sequence of distinct zero-based parameter indices"
                            {:reason :ad-wrt :wrt wrt :parameter-count parameter-count})))
        requested (when (some? wrt) (set wrt))
        valid (set (range parameter-count))]
    (when-not (or (nil? wrt)
                  (and (= (count wrt) (count requested))
                       (every? valid requested)))
      (throw (ex-info ":wrt must be a sequence of distinct zero-based parameter indices"
                      {:reason :ad-wrt :wrt wrt :parameter-count parameter-count})))
    (let [spaces (mapv tangent/tangent-kind tags)]
      (when (nil? requested)
        (when-let [index (first (keep-indexed #(when (= :unknown (:kind %2)) %1)
                                              spaces))]
          (throw (ex-info "A selected parameter's tangent space is unknown"
                          {:reason :ad-unknown-tangent-space :index index
                           :tag (nth tags index)}))))
      (when requested
        (doseq [index requested]
          (case (:kind (nth spaces index))
            :none (throw (ex-info ":wrt selects a parameter with no tangent space"
                                  {:reason :ad-no-tangent-parameter :index index
                                   :tag (nth tags index)}))
            :unknown (throw (ex-info ":wrt selects a parameter whose tangent space is unknown"
                                     {:reason :ad-unknown-tangent-space :index index
                                      :tag (nth tags index)}))
            nil)))
      (let [selected? #(or (nil? requested) (contains? requested %))
            candidate-indices
            (vec (keep-indexed (fn [index {:keys [kind]}]
                                 (when (and (selected? index)
                                            (contains? #{:scalar :array :unknown} kind))
                                   index))
                               spaces))]
        {:spaces spaces
         :candidate-indices candidate-indices
         :candidate-params (mapv params candidate-indices)
         :no-tangent-indices (vec (keep-indexed #(when (= :none (:kind %2)) %1) spaces))}))))

(defn- unowned-deftm-source
  "Identify a deftm without a registered reverse rule for a precise refusal.
   A source definition is not an owned pullback: deriving VJP in the backward
   pass would repeat its primal and can observe effects or changed storage."
  [op]
  (when (qualified-symbol? op)
    (try
      (let [v (resolve op)
            fn-var (or (when (and v (:raster.core/deftm (meta v))) v)
                       (when (and v (.endsWith ^String (name op) "-impl"))
                         (let [base-name (subs (name op) 0 (clojure.core/- (count (name op)) 5))
                               base-sym (symbol (namespace op) base-name)]
                           (when-let [bv (resolve base-sym)]
                             (when (:raster.core/deftm (meta bv)) bv)))))]
        fn-var)
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
          qualified-head (if (op/semantic-op expr)
                           (get numeric-op->qualified head head)
                           head)]
      (with-meta (cons qualified-head (rest expr)) (meta expr)))
    expr))

;; Forward declaration for circular dependency: gen-reverse-let <-> gen-reverse-dotimes/par
(declare ^:private gen-reverse-dotimes)
(declare ^:private replayable-gather-index?)
(declare ^:private gen-reverse-par-map)
(declare ^:private gen-reverse-par-reduce)
(declare ^:private gen-reverse-ordered-reduce)
(declare ^:private gen-reverse-par-scan)
(declare ^:private let-gradient-pieces)
(declare ^:private extract-let-parts)
(declare ^:private product-expression)
(declare ^:private generated-helper-record)

(defn- structural-completion?
  "Object input completion needs no generated compilation policy. Primitive
   inputs and checked references still belong to their explicit source owner."
  [expression]
  (when-let [{:keys [contract]} (value-completion/canonical-parts expression)]
    (or (= value-completion/object-input-contract contract)
        (some? *generated-product-context*))))

(defn- init-active?
  "Decide whether a binding init carries gradient (is active), given the current
  `activity` map {sym -> bool}. Shared by the AD binding engine and all reverse
  orchestrators (gen-reverse-let / gen-reverse-loop-with-let / reify-pullback)."
  [init-expr activity]
  (cond
    (vector? init-expr)
    (boolean (some #(get activity % false) (util/free-syms init-expr)))

    (structural-completion? init-expr)
    (init-active? (:operand (value-completion/canonical-parts init-expr)) activity)

    (product-ports/projection-plan 'activity init-expr (:value-facts *source-memory-context*))
    (boolean (seq (:active-paths
                   (product-ports/projection-plan 'activity init-expr (:value-facts *source-memory-context*)))))

    ;; Literal: never active (a symbol alias is active iff its source is)
    (trivial-expr? init-expr)
    (if (symbol? init-expr) (get activity init-expr false) false)

    (seq? init-expr)
    (let [head (first init-expr)]
      (cond
        ;; Comparison: returns a boolean, never active
        (or (comparison-op? (op/semantic-op init-expr))
            (op/comparison-kind (op/semantic-op init-expr))) false

        ;; An array's length is an integer shape, never a gradient carrier
        (or (contains? '#{alength clojure.core/alength raster.arrays/alength}
                       (op/semantic-op init-expr))
            (and (= '.invk head) (not (:lexical-callee (op/call-description init-expr)))
                 (symbol? (second init-expr))
                 (.startsWith (name (second init-expr)) "alength_m_"))) false

        ;; The predicate is discrete; activity comes from either lexical arm.
        (= 'if head)
        (let [[_ _test then else] init-expr]
          (boolean (some #(get activity % false)
                         (set/union (util/free-syms then)
                                    (util/free-syms else)))))

        ;; A loop or a parallel SOAC is active iff a variable free in it is:
        ;; the compiler's scope information supplies its binders (index,
        ;; accumulator), so no form needs its own rule here.
        (or (contains? '#{loop loop* dotimes} head)
            (= :par (:kind (form/form-info init-expr))))
        (binding [util/*shadowing-locals* (into util/*shadowing-locals* (keys activity))]
          (boolean (some #(get activity % false) (util/free-syms init-expr))))

        ;; Surviving lexical regions carry their free dependencies. They are
        ;; not ordinary calls with only symbol-valued arguments; preparation
        ;; normally flattens these before the reverse-record classifier.
        (or (form/binding-form? init-expr)
            (= :do (:kind (form/form-info init-expr))))
        (binding [util/*shadowing-locals* (into util/*shadowing-locals* (keys activity))]
          (boolean (some #(get activity % false) (util/free-syms init-expr))))

        ;; An opaque closure carries its free captures. The scope grammar
        ;; excludes formal binders; transparent local lambdas were beta-reduced.
        (= :lambda (:kind (form/form-info init-expr)))
        (binding [util/*shadowing-locals* (into util/*shadowing-locals* (keys activity))]
          (boolean (some #(get activity % false) (util/free-syms init-expr))))

        ;; Lexical receiver dependence is not a global operation identity.
        :else
        (let [{:keys [arguments lexical-callee]} (op/call-description init-expr)]
          (boolean (some #(and (symbol? %) (get activity % false))
                         (cond-> arguments lexical-callee (conj lexical-callee)))))))

    :else false))

;; The SOAC heads reverse mode has rules for, by record kind. Each kind's rule is
;; its ad-record and emit-backward methods; a record states the buffers it
;; writes (:written-arrs) and the values its pullback reads again (:replays),
;; and the engine derives dispatch and overwrite checks from those.
(def ^:private soac-record-kinds
  '{raster.par/map!   :par-map
    raster.par/reduce :par-reduce
    raster.par/scan   :par-scan
    raster.par/gather :par-gather
    dotimes           :dotimes})

;; ================================================================
;; Forward-pass record creation — one multimethod, kind-dispatched
;; (mirrors form/scope-info's :kind dispatch). A new differentiable SOAC is one
;; new `ad-form-kind` clause + one `ad-record` defmethod — the engine is untouched.
;; ================================================================

(defn ^:no-doc assert-source-scan-checkpoint!
  "Enforce a source checkpoint obligation only after the existing activity
   authority proves that this region needs a derivative."
  [expression active?]
  (when (and active? (::inexact-source-scan-checkpoint (meta expression)))
    (throw (ex-info "AD stored-carry loop requires an exact carry checkpoint"
                    {:reason :ad-scan-carry-precision :loop expression}))))

(defn ^:no-doc assert-unhandled-parallel-region!
  "A parallel region reaching a mode's ordinary-call fallback has no scoped
   derivative handler. Active captures cannot be proved constant from the
   call's top-level arguments. Both modes use their existing scoped activity
   authority and invoke this only after their supported handlers. Inactive
   regions stay primal; this is refusal, not support for parallel effects."
  [expression active? sym]
  (when (and active? (= :par (:kind (form/form-info expression))))
    (throw (ex-info "Active parallel region has no scoped AD rule"
                    {:reason :ad-unsupported-active-parallel-region
                     :form-head (first expression) :sym sym :init expression}))))

(defn- ad-form-kind
  "Classify a binding init into a reverse-record kind. SOACs (soac-record-kinds)
  are recognized BEFORE the activity gate (they manage their own activity via
  the active-set). Actual constructor/projection structural interfaces survive
  inactivity; other scalar records are gated on `active?`. `sym` is the binding
  symbol, used only for a fail-loud message on unsupported control-flow forms."
  [init-expr active? sym]
  (cond
    (vector? init-expr) :product
    (and (symbol? init-expr)
         (get-in *source-memory-context* [:value-facts init-expr :product])) :product-alias
    (structural-completion? init-expr) :completion
    (product-ports/projection-plan sym init-expr (:value-facts *source-memory-context*)) :product-projection
    ;; Structural interfaces survive inactive choices too. Otherwise a later
    ;; active join forgets the shape of an already known constant product.
    (and (seq? init-expr) (= 'if (first init-expr))) :if
    (and active? (seq? init-expr)
         (some #(get-in *source-memory-context* [:value-facts % :product])
               (:arguments (op/call-description init-expr))))
    (throw (ex-info "active product has no closed structural consumer"
                    {:reason :ad-unsupported-product-use :init init-expr :sym sym}))
    (and (seq? init-expr) (soac-record-kinds (first init-expr)))   (soac-record-kinds (first init-expr))
    (not active?)                                                   :inactive
    (and (seq? init-expr) (= 'if (first init-expr)))                :if
    (symbol? init-expr)                                             :alias
    (and (= :par (:kind (form/form-info init-expr)))
         ;; Declared call rules still own parallel deftm primitives such as
         ;; dot-product/scale. Only unhandled scoped regions refuse here;
         ;; admitted templates must reach the existing :call record below.
         (not (tmpl/resolve-template (:operation (op/call-description init-expr)))))
    (do (assert-unhandled-parallel-region! init-expr true sym) :inactive)
    (and (seq? init-expr) (contains? #{'loop 'loop*} (first init-expr)))
    (do (assert-source-scan-checkpoint! init-expr true) :loop)
    ;; Do not treat an unlinearized region as an inactive ordinary call. Until
    ;; shared preparation flattens it, it has no reverse record contract.
    (or (form/binding-form? init-expr)
        (= :do (:kind (form/form-info init-expr))))
    (throw (ex-info "Active lexical region survived AD normalization"
                    {:reason :ad-active-unlinearized-region
                     :form-head (first init-expr) :sym sym :init init-expr}))
    ;; Fail loud on control-flow / interop forms carrying an ACTIVE value: the
    ;; `:call` fallthrough would wrap them in a bogus deftm rule → silently
    ;; wrong/dropped gradient. AD through these is unsupported — throw a clear
    ;; message instead of miscompiling. (Inactive occurrences already routed to
    ;; :inactive above, so constant control flow is untouched.)
    (and (seq? init-expr)
         (or (= :lambda (:kind (form/form-info init-expr)))
             (contains? '#{case case* try letfn letfn* new
                           monitor-enter monitor-exit}
                        (first init-expr))))
    (throw (ex-info (str "AD through `" (first init-expr)
                         "` is not supported (form bound to `" sym "`). "
                         "Rewrite using a differentiable primitive "
                         "(e.g. `if`, `par/reduce`, or a deftm with an AD template).")
                    {:reason (when (= :lambda (:kind (form/form-info init-expr))) :ad-active-closure)
                     :form-head (first init-expr) :sym sym :init init-expr}))
    (seq? init-expr)                                                :call
    :else                                                          :inactive))

(defmulti ^:private ad-record
  "Create the reverse-pass record for one binding. Returns
   {:record <map-or-nil> :fwd-patch <fn-or-nil>}, where :fwd-patch (when present)
   is applied to the forward-binding vector via swap! (the SOACs rewrite their own
   forward binding to extract a pullback tape)."
  (fn [kind _sym _init-expr _activity] kind))

(defmethod ad-record :inactive [_ _sym _init _activity] {:record nil})

(defn- product-record [sym plan]
  (let [structured? (= :product (get-in plan [:schema :kind]))]
    {:record {:type :product-edges :sym sym :edges (:edges plan)
              :cotangent-ports (mapv (fn [[path leaf]]
                                       {:key (if structured? (product-ports/field-port sym path) sym)
                                        :tag (:tag leaf)})
                                     (filter (fn [[path _]] (contains? (:active-paths plan) path))
                                             (product-ports/leaf-paths (:schema plan))))
              :value-facts (when structured?
                             {sym {:product (select-keys plan [:schema :active-paths])}})}}))

(defmethod ad-record :product [_ sym init activity]
  (product-record sym (product-ports/literal-plan sym init (:value-facts *source-memory-context*) activity)))

(defmethod ad-record :product-projection [_ sym init _activity]
  (product-record sym (product-ports/projection-plan sym init (:value-facts *source-memory-context*))))

(defmethod ad-record :product-alias [_ sym source _activity]
  (let [{:keys [schema active-paths]} (get-in *source-memory-context* [:value-facts source :product])]
    (product-record sym {:schema schema :active-paths active-paths
                         :edges (mapv (fn [[path leaf]]
                                        {:port (product-ports/field-port sym path)
                                         :source (product-ports/field-port source path)
                                         :tag (:tag leaf) :active? (contains? active-paths path)})
                                      (product-ports/leaf-paths schema))})))

(declare ^:private exact-replay-root)

(defn- floating-completion-tangent-plan
  "Transpose one owner-validated floating conversion edge. Both explicit
   completion records and paired helper inputs consume this same contract."
  [contract operand source-tag result-tag source-policy]
  (value-completion/project-to-jvm-source
   (value-completion/complete-value contract operand) source-policy)
  (when-not (and (= :jvm-primitive-input (:source-context contract))
                 (= :scalar (:kind (tangent/tangent-kind source-tag)))
                 (= :scalar (:kind (tangent/tangent-kind result-tag)))
                 (identical? (:target-carrier contract) (get types/tag->java-class result-tag)))
    (throw (ex-info "active primitive completion requires matching floating tangent declarations"
                    {:reason :ad-unclosed-completion-transpose
                     :source-tag source-tag :result-tag result-tag})))
  {:source-tag source-tag})

(defmethod ad-record :completion [_ sym init activity]
  (let [{:keys [contract operand]} (value-completion/canonical-parts init)]
   (if (and (= value-completion/object-input-contract contract)
            (not (trivial-expr? operand)))
    (do
      (when-not (and (anf/closed-call? operand)
                     (= :scalar (:kind (tangent/tangent-kind (types/sym-type-tag sym))))
                     (= :pure (effects/analyze-effect operand)))
        (throw (ex-info "completion operand has no closed structural transpose"
                        {:reason :ad-unclosed-completion-transpose :init init})))
      ;; Derive only the tangent through the existing binding engine. The
      ;; forward binding remains completion of the ORIGINAL invocation, not a
      ;; raw call followed by completion of an ANF temporary. Region-producing
      ;; helper patches need an explicit result-context owner before admission.
      (let [result (ad-record (ad-form-kind operand (init-active? operand activity) sym)
                              sym operand activity)]
        (when (:fwd-patch result)
          (throw (ex-info "completion cannot discard a forward region patch"
                          {:reason :ad-unclosed-completion-transpose :init init})))
        result))
    (do
    (when-not (or (symbol? operand)
                  (and (= value-completion/object-input-contract contract)
                       (trivial-expr? operand)))
      (throw (ex-info "completion operand has no closed structural transpose"
                      {:reason :ad-unclosed-completion-transpose :init init})))
    (case (:source-context contract)
      (:jvm-object-input :jvm-checkcast)
      ;; The transpose transports the tangent unchanged, not the physical
      ;; primal value. Both contexts consume an Object and may first box a
      ;; primitive operand. Preserve possible borrowed-reference aliases, but
      ;; do not publish an exact replay-root substitution from a logical dtype.
      (if (symbol? operand)
        {:record {:type :alias :sym sym :source operand :result-aliases #{operand}}}
        {:record nil})

      :jvm-primitive-input
      (do
        ;; Validate through the source projection authority. This is the actual
        ;; GENERATED source owner's policy, never an inferred deftm policy.
        (value-completion/project-to-jvm-source init (:source-policy *generated-product-context*))
        (if-not (get activity operand)
          {:record {:type :primal :sym sym}}
          (let [source-tag (types/sym-type-tag
                            (exact-replay-root operand (:value-facts *source-memory-context*)))
                result-tag (types/sym-type-tag sym)]
            {:record (merge {:type :floating-completion :sym sym :source operand}
                            (floating-completion-tangent-plan
                             contract operand source-tag result-tag
                             (:source-policy *generated-product-context*)))}))))))))

;; out[e] = src[index[e]] (strided: blocks of `stride`). The forward runs as
;; written; the pullback is its transpose, a scatter-add into d_src.
(defmethod ad-record :par-gather [_ sym init-expr activity]
  (let [[_ out src index n stride] init-expr]
    {:record {:type :par-gather :sym sym :written-arrs [out]
              :result-alias out
              :replays (when (get activity src false) (util/free-syms index))
              :out out :src src :index index :n n :stride stride
              :d-out-sym (ad-gensym "d_gather_out" (:raster.type/tag (meta out)))
              :d-src-sym (ad-gensym (str "d_" (if (symbol? src) (name src) "src"))
                                    (:raster.type/tag (meta src)))
              :d-src-zero-sym (ad-gensym "d_src_zero" (:raster.type/tag (meta src)))}}))

(defmethod ad-record :alias [_ sym init-expr _activity]
  {:record {:type :alias :sym sym :source init-expr :result-alias init-expr}})

(defmethod ad-record :if [_ sym init-expr activity]
  (let [[_ branch then else] init-expr
        then-product (get-in *source-memory-context* [:value-facts then :product])
        else-product (get-in *source-memory-context* [:value-facts else :product])]
    (when-not (trivial-expr? branch)
      (throw (ex-info "Reverse AD conditional test survived predicate normalization"
                      {:reason :ad-region-unclosed-test :sym sym :test branch})))
    (cond
      (and (every? trivial-expr? [then else]) (or then-product else-product))
      (product-record sym (product-ports/choice-plan sym branch then then-product else else-product))

      (every? trivial-expr? [then else])
      {:record {:type :if :sym sym :branch branch :then then :else else}}

      :else
      (let [free (set/union (util/free-syms then) (util/free-syms else))
            params (vec (sort-by str (filter #(and (get activity %) (contains? free %))
                                            (keys activity))))
            dy (ad-gensym "branch_dy" (:raster.type/tag (meta sym)))
            pieces (mapv (fn [arm]
                           (let [[bindings body] (extract-let-parts (util/alpha-convert arm))]
                             (binding [*source-memory-context* (assoc *source-memory-context* :internal-region? true)
                                       *generated-product-context*
                                       (when *generated-product-context*
                                         (assoc *generated-product-context* :internal-region? true))]
                               (let-gradient-pieces bindings body params dy))))
                         [then else])
            result-products (mapv :result-product pieces)
            result-product (when (some some? result-products)
                             (apply product-ports/join-facts result-products))
            tape (vary-meta (ad-gensym "branch_tape") assoc
                            :raster.ad/residual
                            {:kind :conditional :representation :persistent-vector
                             :source-form init-expr})
            replay-roots (vec (sort-by str (into #{} (mapcat (comp keys :pullback-replay-conditions)) pieces)))
            paired (mapv (fn [{:keys [fwd-bindings body-sym pullback-replay-conditions]}]
                           (let [saved (ad-gensym "branch_saved")
                                 conditions (mapv (fn [root]
                                                    [(ad-gensym "branch_replay")
                                                     (get pullback-replay-conditions root false)])
                                                  replay-roots)]
                             (list 'let* (vec (concat fwd-bindings
                                                     [saved (vec (take-nth 2 fwd-bindings))]
                                                     (mapcat identity conditions)))
                                   ;; Export evaluated obligations through the existing
                                   ;; residual, never leak a child's private selector.
                                   ;; Keep every residual element in ANF so that a
                                   ;; subsequent linearization can traverse this tape.
                                   (into [body-sym saved] (map first conditions)))))
                         pieces)]
        {:record (cond-> {:type :if-region :sym sym :branch branch :params params
                  :parameter-products (into {} (keep (fn [parameter]
                                                         (when-let [product (get-in *source-memory-context*
                                                                                  [:value-facts parameter :product])]
                                                           [parameter (assoc product :active-paths
                                                                             (into #{} (mapcat #(get-in % [:parameter-product-demands parameter])) pieces))]))) params)
                  :dy dy :tape tape :pieces pieces
                  :replay-conditions
                  (into {} (map-indexed (fn [index root]
                                         [root (list 'clojure.core/nth tape (clojure.core/+ 2 index))]) replay-roots))
                  :effects {:kind :choice :regions pieces}}
                   result-product
                   (merge {:result-product result-product
                           :cotangent-ports (:cotangent-ports (:record (product-record sym result-product)))
                           :value-facts {sym {:product result-product}}}))
         :fwd-patch (fn [bindings]
                      (into (vec (drop-last 2 bindings))
                            [tape (list 'if branch (first paired) (second paired))
                             sym (list 'clojure.core/nth tape 0)]))}))))

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
    {:record (assoc pm-info :result-alias out-buf)
     ;; No per-element pullback means no tape: the original binding still
     ;; performs the map's writes.
     :fwd-patch (fn [bs]
                  (if-let [forward-code (:forward-code pm-info)]
                    (let [without-last (vec (drop-last 2 bs))]
                      (vec (concat without-last [tape-sym forward-code sym out-buf])))
                    bs))}))

(defmethod ad-record :par-reduce [_ sym init-expr activity]
  (let [active-set (vec (keys (filter val activity)))
        pr-info (assoc (gen-reverse-par-reduce init-expr active-set) :sym sym)
        reduce-result-sym (:reduce-result-sym pr-info)]
    {:record pr-info
     :fwd-patch (fn [bs]
                  (let [without-last (vec (drop-last 2 bs))]
                    (vec (concat without-last
                                 (:forward-bindings pr-info)
                                 [sym reduce-result-sym]))))}))

(defmethod ad-record :par-scan [_ sym init-expr activity]
  ;; NO :fwd-patch — this is the point of scan-as-recurrence: out[i] = acc_i,
  ;; so THE OUTPUT ARRAY IS THE CARRY TAPE. The forward binding runs completely
  ;; unpatched (the scan macro returns `out`, so `sym` IS the buffer), and the
  ;; backward reconstructs every step's inputs from out[] — no closure tape,
  ;; no ArrayList, no separate residual stack (Griewank store-the-carry
  ;; checkpointing at every step; zero recompute depth).
  (if (init-active? init-expr activity)
    (let [active-set (vec (keys (filter val activity)))]
      {:record (assoc (gen-reverse-par-scan init-expr active-set)
                      :sym sym :result-alias (second init-expr))})
    ;; The primal still executes and forward-pass still checks its source
    ;; writes/exit aliases through shared memory analysis. No active input
    ;; means no reverse recurrence or carry reconstruction is required.
    {:record {:type :primal :sym sym :written-arrs [(second init-expr)]
              :result-alias (second init-expr)}}))

(defmethod ad-record :dotimes [_ sym init-expr activity]
  (let [active-set (vec (keys (filter val activity)))
        dt-info (assoc (gen-reverse-dotimes init-expr active-set) :sym sym)
        tape-sym (:tape-sym dt-info)
        ;; Bind the result sym to the output buffer (populated by forward-code),
        ;; not nil — so a downstream consumer sees the array (see :par-map).
        out-buf (first (:written-arrs dt-info))]
    {:record (cond-> dt-info (:forward-code dt-info) (assoc :result-alias out-buf))
     :fwd-patch (fn [bs]
                  (if-let [forward-code (:forward-code dt-info)]
                    (let [without-last (vec (drop-last 2 bs))]
                      (vec (concat without-last [tape-sym forward-code sym out-buf])))
                    bs))}))

(def ^:private mangled-arg-stamp-tags
  "Dispatch tags from a devirtualized forward call's mangled name that are
  precise enough to CARRY onto its arg symbols as :raster.type/tag for the
  backward emission (ad-record :call): concrete primitive scalars/arrays
  only. 'Object and value-type tags are deliberately excluded — the runtime
  value may be more specific than the boxed dispatch slot, and stamping
  them would lock the emitted backward helpers onto generic overloads."
  '#{double float long int doubles floats longs ints})

(defmethod ad-record :call [_ sym init-expr activity]
  (let [{op :operation args :arguments implementation :implementation-op lexical :lexical-callee}
        (op/call-description init-expr)
        resolved (when-not lexical
                   (let [candidate (tmpl/resolve-template op)]
                     (when (tmpl/reverse-rule? (first candidate)) candidate)))
        has-active-args? (some #(and (symbol? %) (get activity % false))
                               (cond-> args lexical (conj lexical)))]
    (if-let [owned (when (and *generated-product-context* (not resolved))
                     (generated-helper-record sym init-expr activity))]
      owned
      (do
    (when (and (not resolved) has-active-args?)
      (when-let [source (when-not lexical (unowned-deftm-source implementation))]
        (throw (ex-info "Derived helper AD requires a forward-owned pullback and completed call inputs"
                        {:reason :ad-unowned-derived-pullback
                         :op op :implementation-op implementation
                         :source (symbol (str (.ns source)) (str (.sym source)))
                         :sym sym :form init-expr
                         :active (filterv #(and (symbol? %) (get activity % false)) args)})))
      (throw (ex-info (str "No AD template for `" (or lexical op)
                           "` which has active (differentiable) inputs. "
                           "Register an AD template or mark inputs as constant.")
                      {:reason (if lexical :ad-active-lexical-callee :ad-missing-rule)
                       :callee lexical :op op :args args :sym sym
                       :active (filterv #(and (symbol? %) (get activity % false)) args)})))
    {:record
     (when resolved
       (let [[template base-op] resolved
             arg-tags (let [n (name implementation)
                            idx (.indexOf ^String n "_m_")]
                        (when (pos? idx)
                          (let [tag-str (subs n (clojure.core/+ idx 3))
                                tag-str (if (.endsWith ^String tag-str "-impl")
                                          (subs tag-str 0 (clojure.core/- (count tag-str) 5))
                                          tag-str)]
                            (mapv symbol (.split ^String tag-str "_")))))
             ;; Π-emission input (plan 1c): CARRY the devirtualized forward
             ;; call's dispatch tags onto its arg SYMBOLS, so the template
             ;; grads-fn and the emission-time resolver read them as primal
             ;; tags. Only stamps a symbol that has no tag yet, and only
             ;; concrete primitive scalar/array tags — 'Object (and value
             ;; types) are excluded so backward helpers aren't locked onto
             ;; the generic overload. Carried, never guessed.
             args (if (and arg-tags (= (count arg-tags) (count args)))
                    (mapv (fn [a t]
                            (if (and (symbol? a)
                                     (nil? (:raster.type/tag (meta a)))
                                     (contains? mangled-arg-stamp-tags t))
                              (vary-meta a assoc :raster.type/tag t)
                              a))
                          args arg-tags)
                    args)]
         {:type :call :sym sym :base-op base-op :resolved-template template :args args
          :active-args (set (filter #(and (symbol? %) (get activity % false)) args))}))}))))

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


(defn- written-syms
  "Array-write targets, using the retained call identity for bare and typed calls."
  [init-expr]
  (into #{}
        (keep op/aset-array-sym)
        (tree-seq coll? seq init-expr)))

(defn- index-replays
  "Loop-invariant values reread to reconstruct a transposed read's index.
   The forward read value itself can be taped while its index still needs replay."
  [reads counter]
  (reduce into #{} (map #(util/free-syms (:idx %) #{counter}) reads)))

(defn- record-memory-effects
  "Compose already-checked child regions without flattening exclusive paths.
   Choice-arm writes can invalidate an earlier residual; either arm's residual
   must survive later writes. Writes in one arm do not execute in the other."
  [record expression]
  (if-let [{:keys [kind regions]} (:effects record)]
    (do
      (when-not (and (or (= :choice kind) (and (= :region kind) (= 1 (count regions))))
                     (seq regions)
                     (every? #(and (set? (:writes %)) (set? (:replays %))) regions))
        (throw (ex-info "Malformed reverse AD region effect summary"
                        {:reason :ad-region-effects :effects (:effects record)})))
      {:writes (reduce into #{} (map :writes regions))
       :replays (reduce into #{} (map :replays regions))
       :value-facts (merge (reduce merge {} (map :value-facts regions))
                           ;; The region owner may publish its checked joined
                           ;; result interface; child catalogs do not define it.
                           (:value-facts record))
       ;; Child scopes were checked independently by the same forward pass.
       ;; The choice itself has no additional source write outside those scopes.
       :local-writes #{}
       ;; A map still owns its implicit output store AFTER the checked child.
       ;; Child replay facts do not replace source input/output stability.
       :local-reads (if (seq (:written-arrs record))
                      (set (map :sym (op/aget-reads expression))) #{})})
    (let [writes (written-syms expression)]
      {:writes writes :local-writes writes :replays (set (:replays record))
       ;; Stable input storage includes excluded observation arrays too. The
       ;; shared retained-call collector is independent of tangent activity.
       :local-reads (set (map :sym (op/aget-reads expression)))
       :value-facts (:value-facts record)})))

(defn- join-input-replay-conditions [left right]
  (merge-with (fn [a b]
                (cond (or (true? a) (true? b)) true
                      (= a b) a
                      :else (list 'if a true b)))
              left right))

(defn- exact-replay-root [id value-facts]
  (loop [id id seen #{}]
    (when (contains? seen id)
      (throw (ex-info "Reverse AD replay has a cyclic exact alias proof"
                      {:reason :ad-replay-alias-cycle :value id})))
    (let [{:keys [value alias]} (get value-facts id)]
      (if alias (recur alias (conj seen id)) (or value id)))))

(defn- source-memory-scope
  "Project actual forward state into a child construction scope. Visibility is
   not the fact catalog; conditional replay obligations are not the replay set."
  [{:keys [values value-facts aliases activity input-replay-conditions unprojectable]}]
  (when-not (and (vector? values)
                 (every? map? [value-facts aliases activity input-replay-conditions unprojectable]))
    (throw (ex-info "Reverse AD child scope requires complete forward state"
                    {:reason :ad-forward-scope})))
  (assoc *source-memory-context*
         :values values :value-facts value-facts :aliases aliases
         :activity (merge-with #(or %1 %2) (:activity *source-memory-context*) activity)
         :replayed (join-input-replay-conditions (:replayed *source-memory-context*)
                                                input-replay-conditions)
         :unprojectable (join-input-replay-conditions (:unprojectable *source-memory-context*)
                                                      unprojectable)))

(defn- record-replay-conditions
  "Project actual replay roots through exact aliases and retained declarations.
   Choice selectors are already evaluated; private child selectors never escape."
  [record memory-effects value-facts]
  (let [roots (fn [ids]
                (into #{}
                      (keep (fn [id]
                              (let [root (exact-replay-root id value-facts)]
                                (when-not (contains? types/primitive-info
                                                     (:raster.type/tag (meta root)))
                                  root))))
                      ids))]
    (if (= :if-region (:type record))
      (let [conditions (:replay-conditions record)]
        (when-not (map? conditions)
          (throw (ex-info "Conditional reverse region lacks its residual replay projection"
                          {:reason :ad-region-replay-projection :sym (:sym record)})))
        conditions)
      (zipmap (roots (:replays memory-effects)) (repeat true)))))

(defn- replay-write-compatible? [value-facts read write]
  (let [array-class (fn [id]
                      (get types/primitive-array-info
                           (:raster.type/tag (meta (get-in value-facts [id :value] id)))))
        read-class (array-class read) write-class (array-class write)]
    ;; An absent retained declaration is not proof of incompatibility.
    (or (nil? read-class) (nil? write-class) (= read-class write-class))))

(defn- replay-input-guard-bindings
  "Retain identity checks between visible replay objects and write destinations.
   An unknown destination is not fresh; child facts do not grant visibility."
  ([conditions writes]
   (replay-input-guard-bindings conditions writes
                                {:reason :ad-replayed-array-alias
                                 :message "Reverse AD replay input aliases a later write"
                                 :include-same-value? true}))
  ([conditions writes {:keys [reason message include-same-value?]}]
  (if (or (empty? conditions) (empty? writes))
    []
    (do
      (when (some (complement symbol?) writes)
        (throw (ex-info "Reverse AD replay guard requires a normalized write destination"
                        {:reason :ad-unclosed-write-target :destinations writes})))
      (let [values (:values *source-memory-context*)
            value-facts (:value-facts *source-memory-context*)
            pairs (vec (for [read values :when (contains? conditions read)
                             write (sort-by str writes)
                             :when (and (or include-same-value? (not= read write))
                                        (replay-write-compatible? value-facts read write))]
                         [read write (get conditions read)]))
            private-writes (set/difference (set (map second pairs)) (set values))]
        (when (seq private-writes)
          (throw (ex-info "Reverse AD cannot project a write destination into this scope"
                          {:reason :ad-unprojectable-write-root
                           :writes (vec (sort-by str private-writes))})))
        (vec
         (mapcat
          (fn [[read write condition]]
            [(vary-meta (ad-gensym "_replay_alias_guard") assoc :raster.effect/effectful true)
             (let [guard (list 'if (list 'clojure.core/identical? read write)
                               (list 'throw
                                     (list 'clojure.core/ex-info
                                           message
                                           {:reason reason
                                            :input (list 'quote read) :write (list 'quote write)}))
                               nil)]
               (if (true? condition) guard (list 'if condition guard nil)))])
          pairs)))))))

(defn- active-source-storage?
  "Positive prior activity through the same memory alias/root authority."
  [value activity aliases value-facts]
  (some #(or (get activity % false)
             (get activity (exact-replay-root % value-facts) false))
        (memory/transitive-closure #{value} aliases)))

(defn- source-write-disjoint-bindings
  "Independent visible reference values are not differentiated memory versions.
   Differentiated outputs must be separate from every prior visible reference;
   passive writes need separation only from prior active storage. Replay fences
   remain independent. Identity alone cannot synchronize stale activity facts."
  [writes differentiated-writes prior-activity]
  (if (empty? writes)
    []
    (let [{:keys [aliases value-facts activity]} *source-memory-context*
          activity (merge-with #(or %1 %2) activity prior-activity)
          conditions (into {}
                           (keep (fn [value]
                                   (when (and (not (contains? types/primitive-info
                                                             (:raster.type/tag (meta value))))
                                              (or (seq differentiated-writes)
                                                  (active-source-storage? value activity aliases value-facts)))
                                     [value true])))
                           (:values *source-memory-context*))]
      (replay-input-guard-bindings
       conditions writes
       {:reason :ad-unversioned-storage-alias
        :include-same-value? false
        :message "Source reverse AD requires disjoint independently named writable storage"}))))

(defn- assert-source-write-contract!
  "Source records do not version an active array's previous contents. Refuse
   overwrites of that version, and active scalar stores without a store rule.
   Positive alias evidence is sufficient to refuse, never to establish freshness."
  [sym expression writes output-writes local-reads activity]
  (let [{:keys [aliases value-facts]} *source-memory-context*
        activity (merge-with #(or %1 %2) (:activity *source-memory-context*) activity)
        destination (op/aset-array-sym expression)
        active-storage
        (filterv #(active-source-storage? % activity aliases value-facts)
                 (sort-by str writes))]
    (when (seq active-storage)
      (throw (ex-info "Source reverse AD requires a versioned derivative for active storage writes"
                      {:reason :ad-active-storage-write :sym sym :writes active-storage})))
    (when (and destination
               (init-active? (last (op/call-args expression)) activity))
      (throw (ex-info "Source reverse AD has no versioned derivative for an active scalar store"
                      {:reason :ad-unversioned-store :sym sym :write destination})))
    ;; A declared output is differentiated as an independent result, not an
    ;; inter-lane mutable state. Explicit stores inside ordered child programs
    ;; retain their own checked tape/activity contracts; do not conflate those
    ;; effects with an implicit differentiated output version.
    (let [read-write (set/intersection (memory/transitive-closure (set output-writes) aliases)
                                       (memory/transitive-closure local-reads aliases))]
      (when (seq read-write)
        (throw (ex-info "Source reverse AD requires stable input storage during a writing operation"
                        {:reason :ad-unversioned-storage-read-write :sym sym
                         :arrays (vec (sort-by str read-write))}))))))

(defn- forward-pass
  "Phase 1 (pure): thread {:activity :fwd-bindings :records} across the bindings.
  An array a pullback replays must keep its forward contents until the
  backward runs: a later write to it is rejected rather than differentiated at
  the overwritten values. Alias state retains positive may-alias edges through
  the shared memory analysis; an absent edge is not a freshness proof."
  [norm-bindings active-params]
  (reduce
   (fn [{:keys [activity fwd-bindings records replayed writes aliases input-replay-conditions
                values value-facts unprojectable]}
        [sym init-expr]]
     (let [active?  (binding [*source-memory-context* (assoc *source-memory-context* :value-facts value-facts)]
                      (init-active? init-expr activity))
           activity (assoc activity sym active?)
           may-aliases #(memory/transitive-closure (set %) aliases)
           scope (source-memory-scope
                  {:values values :value-facts value-facts :aliases aliases :activity activity
                   :input-replay-conditions input-replay-conditions :unprojectable unprojectable})
           ambient (:replayed scope)
           unknown (:unprojectable scope)
           ;; Inactive choices can still change storage dependencies. Shared
           ;; effect analysis selects the existing scoped adapter conservatively;
           ;; it is not a freshness or derivative certificate.
           kind (binding [*source-memory-context* scope]
                  (if (and (seq? init-expr) (= 'if (first init-expr))
                         (or (and (or (seq ambient) (seq unknown))
                                  (seq (written-syms init-expr)))
                             (and (not active?)
                                  (not= :pure (effects/analyze-effect init-expr)))))
                  :if
                  (ad-form-kind init-expr active? sym)))
           {:keys [record fwd-patch]}
           (binding [*source-memory-context* scope]
             (ad-record kind sym init-expr activity))
           ;; A SOAC whose body carries gradient makes its WRITTEN buffers
           ;; gradient-carrying too: a downstream DIRECT read (aget out k)
           ;; must scatter into adj-env[out] (emit-backward already sums the
           ;; buffer's contributions into d_out alongside the result sym's).
           ;; Without this, statement-position map!/scan followed by a direct
           ;; buffer read silently drops the adjoint (zero gradients).
           prior-activity activity
           activity (if (and active? (seq (:written-arrs record)))
                      (reduce #(assoc %1 %2 true) activity (:written-arrs record))
                      activity)
           memory-effects (record-memory-effects record init-expr)
           value-facts (merge value-facts (:value-facts memory-effects))
           conditions (record-replay-conditions record memory-effects value-facts)
           visible-ids (set values)
           visible-conditions (select-keys conditions visible-ids)
           private-conditions (apply dissoc conditions visible-ids)
           current-replays (may-aliases (:replays memory-effects))
           source-writes (:writes memory-effects)
           unprojectable-overwrite
           (seq (sort-by str
                         (into #{}
                               (for [[roots targets] [[(keys unknown)
                                                      (into source-writes (:written-arrs record))]
                                                     [(keys private-conditions)
                                                      (:local-writes memory-effects)]]
                                     root roots write targets
                                     :when (replay-write-compatible? value-facts root write)]
                                 root))))
           overwritten
           (seq (sort-by str (set/union
                  (set/intersection replayed
                                    (may-aliases (into source-writes (:written-arrs record))))
                  ;; A step may overwrite its own scatter index before the
                  ;; pullback. Check explicit source writes, not the normal
                  ;; output writes declared by scan/map/gather records.
                  (set/intersection current-replays
                                    (may-aliases (:local-writes memory-effects)))
                  (set/intersection (may-aliases (:input-replays record))
                                    (may-aliases (:written-arrs record))))))
           ;; Even an inactive alias can write an earlier residual. Records
           ;; describe result/output identity without another operation registry.
           result-aliases (into (or (:result-aliases record) #{})
                               (if-let [result-alias (:result-alias record)]
                                 #{result-alias}
                                 (:symbols (form/normal-exit-facts init-expr))))
           aliases (:aliases
                    (memory/track-aliases {:aliases aliases :consumed #{}}
                                          ;; Partial positive evidence survives
                                          ;; unknown arms; completeness/freshness
                                          ;; are NOT inferred from missing edges.
                                          sym (or result-aliases #{}) #{}))
           ;; Choice children check their own writes on the selected path.
           declared-writes (into (if-let [destination (op/aset-array-sym init-expr)]
                                   #{destination} #{})
                                 (:written-arrs record))
           guard-bindings (binding [*source-memory-context* scope]
                            (replay-input-guard-bindings ambient declared-writes))
           disjoint-bindings
           (binding [*source-memory-context* scope]
             (source-write-disjoint-bindings
              ;; A scoped choice exports no local writes: only the selected
              ;; child's checks execute. An unclosed non-choice target refuses.
              (into (:local-writes memory-effects) (:written-arrs record))
              (when active? (:written-arrs record)) prior-activity))
           ;; Freeze the integer proof at this binding's source position. Following raw
           ;; initializer aliases later could observe a different lexical rebinding.
           constant-index (value-product/index-value
                           init-expr #(get-in value-facts [% :constant-index]))
           fwd (into (into (into fwd-bindings guard-bindings) disjoint-bindings)
                     [sym (qualify-fwd-expr init-expr)])]
       (when overwritten
         (throw (ex-info
                 (str "Reverse-mode AD: `" sym "` writes " (vec overwritten) ", which an "
                      "operation re-reads in its pullback. Write a fresh "
                      "array instead.")
                 {:reason :replayed-array-overwritten :sym sym :arrays (vec overwritten)})))
       (when unprojectable-overwrite
         (throw (ex-info "Reverse AD cannot project a private replay object across this write"
                         {:reason :ad-unprojectable-replay-root :sym sym
                          :replays (vec unprojectable-overwrite)})))
       (binding [*source-memory-context* scope]
         (assert-source-write-contract! sym init-expr
                                        (into source-writes (:written-arrs record))
                                        (when active? (:written-arrs record))
                                        (:local-reads memory-effects)
                                        prior-activity))
       {:activity activity
        :fwd-bindings (if fwd-patch (fwd-patch fwd) fwd)
        :records (if record (conj records record) records)
        :aliases aliases
        ;; A binding enters visibility only after its initializer. Exact result
        ;; identity comes from the existing record or shared exact-tail authority.
        :values (conj values sym)
        :value-facts (assoc value-facts sym
                            (cond-> {:value sym :alias (or (:result-alias record)
                                                           (form/tail-symbol init-expr))}
                              (some? constant-index) (assoc :constant-index constant-index)
                              (get-in value-facts [sym :product])
                              (assoc :product (get-in value-facts [sym :product]))))
        :unprojectable (join-input-replay-conditions unprojectable private-conditions)
        :input-replay-conditions (join-input-replay-conditions
                                  input-replay-conditions
                                  visible-conditions)
        :replayed (into replayed
                        (memory/transitive-closure (set (:replays memory-effects)) aliases))
        :writes (into writes (may-aliases (into source-writes (:written-arrs record))))}))
   {:activity (into {} (map (fn [p] [p true]) active-params))
    :fwd-bindings [] :records [] :replayed #{} :writes #{}
    :values (:values *source-memory-context*)
    :value-facts (merge (into {} (map (fn [value] [value {:value value}]))
                              (:values *source-memory-context*))
                        (:value-facts *source-memory-context*))
    :unprojectable {}
    :aliases (:aliases *source-memory-context* {})
    :input-replay-conditions {}}
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

(defn- sum-contribs-into
  "Retain ordered optional accumulation. Physical array lowering belongs to
   the shared producer/presence analysis, not a blacklist of source call heads."
  [rev-ctx target-sym contribs default]
  (bindings-into rev-ctx [target-sym (sum-contribs contribs default)]))

(defmulti ^:private emit-backward
  "Reverse-pass per-record emission (PURE): returns the updated
  {:adj-env :rev-ctx}. `adj-sym` is this record's gensym'd adjoint (nil for the
  array SOACs, which carry their own d_out/d_acc syms in the record)."
  (fn [record _adj-sym _activity _state] (:type record)))

(defmethod emit-backward :primal [_ _ _ state]
  ;; Primal-only regions still own their memory footprint and result identity.
  ;; Their checked activity says there is no input tangent to transpose.
  state)

(defmethod emit-backward :product-edges
  [{:keys [edges]} adjoints _activity {:keys [adj-env rev-ctx]}]
  {:rev-ctx rev-ctx
   :adj-env (reduce (fn [environment {:keys [port source active? branch arm]}]
                      (if (and active? (contains? adjoints port))
                        (let [value (get adjoints port)
                              value (case arm
                                      :then (list 'if branch value nil)
                                      :else (list 'if branch nil value)
                                      value)]
                          (update environment source (fnil conj []) value))
                        environment))
                    adj-env edges)})

(defn- lift-pullback-over-absence
  "Lift one checked call pullback to the dynamic absent cotangent. A numeric
   zero remains present and executes the original derivative arithmetic."
  [before after gradients adjoint]
  (let [prefix (vec (:bindings before))
        body (subvec (vec (:bindings after)) (count prefix))
        fresh (:gensym-fn before)
        result (fresh "present_pullback")
        slots (mapv (fn [gradient]
                      (when (some? gradient)
                        (fresh "present_gradient" (:raster.type/tag (meta gradient))))) gradients)]
    [(assoc after :bindings
            (vec (concat prefix
                         [result (list 'if (list 'clojure.core/nil? adjoint)
                                       (vec (repeat (count gradients) nil))
                                       (list 'let* body (vec gradients)))]
                         (mapcat (fn [[index slot]]
                                   (when slot [slot (list 'clojure.core/nth result index)]))
                                 (map-indexed vector slots)))))
     slots]))

(defmethod emit-backward :call
  [{:keys [sym base-op args active-args resolved-template]} adj-sym _activity {:keys [adj-env rev-ctx]}]
  ;; Static emission consumes the SAME selected rule that admitted the call.
  ;; These records are ephemeral derivation data, not persistent source IR.
  ;; Registered opaque factories retain their existing runtime lookup contract;
  ;; this does not claim to capture them or introduce a private closure owner.
  (let [template resolved-template
        has-template? (and template (or (:grads template) (:grads-fn template)))]
    (if has-template?
      ;; Template path: ctx-aware instantiation (already pure — returns new ctx).
      (let [[template-ctx template-gradients] (tmpl/instantiate-template-ctx template args sym adj-sym rev-ctx)
            [new-ctx grad-syms] (lift-pullback-over-absence rev-ctx template-ctx template-gradients adj-sym)]
        {:rev-ctx new-ctx
         :adj-env (reduce (fn [ae [i grad-sym]]
                            (if (and (some? grad-sym) (< i (count args))
                                     (symbol? (nth args i)) (contains? active-args (nth args i)))
                              (update ae (nth args i) (fnil conj []) grad-sym)
                              ae))
                          adj-env (map-indexed vector grad-syms))})
      ;; Closure fallback — nil pullback entries propagate as nil.
      (let [pb-sym (ad-gensym "pb") grads-sym (ad-gensym "grads")
            call-bindings [pb-sym (list* (list 'raster.ad.templates/get-pullback-factory
                                              (list 'quote base-op)) sym args)
                           grads-sym (list pb-sym adj-sym)]
            rev-ctx (bindings-into rev-ctx
                                     [grads-sym (list 'if (list 'clojure.core/nil? adj-sym)
                                                      (vec (repeat (count args) nil))
                                                      (list 'let* call-bindings grads-sym))])]
        {:rev-ctx rev-ctx
         :adj-env (reduce (fn [ae [i arg]]
                            (if (and (symbol? arg) (contains? active-args arg))
                              (update ae arg (fnil conj []) (list 'nth grads-sym i))
                              ae))
                          adj-env (map-indexed vector args))}))))

(defmethod emit-backward :if
  [{:keys [sym branch then else]} adj-sym activity {:keys [adj-env rev-ctx]}]
  ;; The unselected edge is absent, not a present zero. The latter could
  ;; activate a singular upstream pullback (0*infinity) or have the wrong
  ;; array shape. Public gradient materialization belongs to the boundary.
  (let [adj-env (cond-> adj-env
                  (and (symbol? then) (get activity then false))
                  (update then (fnil conj []) (list 'if branch adj-sym nil))
                  (and else (symbol? else) (get activity else false))
                  (update else (fnil conj []) (list 'if branch nil adj-sym)))]
    {:adj-env adj-env :rev-ctx rev-ctx}))

(defmethod emit-backward :if-region
  [{:keys [sym branch params dy tape pieces parameter-products result-product]} adj-sym _activity {:keys [adj-env rev-ctx]}]
  (let [saved (ad-gensym "branch_saved")
        seed (if result-product
               (product-expression (:schema result-product)
                                   (into {} (map (fn [[path _]]
                                                   [path (get adj-sym (product-ports/field-port sym path))]))
                                         (product-ports/leaf-paths (:schema result-product))))
               adj-sym)
        arms (mapv (fn [{:keys [fwd-bindings rev-bindings param-adj-syms]}]
                     (let [restore (mapcat (fn [index value]
                                             [value (list 'clojure.core/nth saved index)])
                                           (range) (take-nth 2 fwd-bindings))]
                       (list 'let* (vec (concat restore [dy seed] rev-bindings))
                             (vec param-adj-syms))))
                   pieces)
        grads (ad-gensym "branch_grads")
        rev-ctx (bindings-into rev-ctx
                               [saved (list 'clojure.core/nth tape 1)
                                grads (list 'if branch (first arms) (second arms))])]
    {:rev-ctx rev-ctx
     :adj-env (reduce (fn [env [index param]]
                        (let [gradient (list 'clojure.core/nth grads index)]
                          (if-let [{:keys [active-paths]} (get parameter-products param)]
                            (reduce (fn [env path]
                                      (update env (product-ports/field-port param path) (fnil conj [])
                                              (reduce (fn [value index] (list 'clojure.core/nth value index))
                                                      gradient path)))
                                    env (sort active-paths))
                            (update env param (fnil conj []) gradient))))
                      adj-env (map-indexed vector params))}))

(defmethod emit-backward :alias
  [{:keys [source]} adj-sym activity {:keys [adj-env rev-ctx]}]
  {:adj-env (cond-> adj-env
              (get activity source false) (update source (fnil conj []) adj-sym))
   :rev-ctx rev-ctx})

(defmethod emit-backward :floating-completion
  [{:keys [source source-tag]} adjoint _activity {:keys [adj-env rev-ctx]}]
  (let [projected (ad-gensym "input_cotangent" (tangent/tangent-tag source-tag))]
    {:rev-ctx (bindings-into rev-ctx [projected (tangent/project-expr source-tag adjoint)])
     :adj-env (update adj-env source (fnil conj []) projected)}))

(defmethod emit-backward :paired-helper
  [{:keys [pullback inputs result-tag]} adjoint _activity {:keys [adj-env rev-ctx]}]
  (let [gradients (ad-gensym "helper_gradients")
        rev-ctx (bindings-into rev-ctx
                               [gradients (list 'if (list 'clojure.core/nil? adjoint)
                                                nil (list pullback (tangent/project-expr result-tag adjoint)))])]
    (reduce
     (fn [{:keys [adj-env rev-ctx]} [index {:keys [source source-tag floating?]}]]
       (let [gradient (ad-gensym "helper_input_cotangent" (tangent/tangent-tag source-tag))
             slot (list 'if (list 'clojure.core/nil? gradients) nil
                        (list 'clojure.core/nth gradients index))
             projected (if floating? (tangent/project-expr source-tag slot) slot)]
         {:rev-ctx (bindings-into rev-ctx [gradient projected])
          :adj-env (update adj-env source (fnil conj []) gradient)}))
     {:rev-ctx rev-ctx :adj-env adj-env} (map-indexed vector inputs))))

(defmethod emit-backward :par-gather
  [{:keys [sym out src index n stride d-out-sym d-src-sym d-src-zero-sym]}
   _adj-sym activity {:keys [adj-env rev-ctx]}]
  (let [contribs (concat (get adj-env sym) (get adj-env out))]
    (if (or (empty? contribs) (not (get activity src false)))
      {:adj-env adj-env :rev-ctx rev-ctx}
      {:rev-ctx (bindings-into rev-ctx
                               [d-out-sym (sum-contribs contribs nil)
                                d-src-zero-sym (list 'raster.arrays/zeros-like src
                                                     (list 'raster.arrays/alength src))
                                d-src-sym (apply list 'raster.par/scatter! d-src-zero-sym d-out-sym
                                                 index n (when stride [stride]))])
       :adj-env (update adj-env src (fnil conj []) d-src-sym)})))

(defmethod emit-backward :par-map
  [{:keys [sym written-arrs read-arrs d-read-arr-syms d-scalar-syms active-free
           shadow-allocs backward-maps backward-reduces d-out-sym]}
   _adj-sym _activity {:keys [adj-env rev-ctx]}]
  (let [;; The output seed belongs outside the optional fragment. Its shadow
        ;; allocations and backward kernels execute only for a present seed,
        ;; just as an ordinary call's pullback does. A numeric-zero buffer is
        ;; present and must not suppress singular derivative arithmetic.
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
        before rev-ctx
        after (reduce bindings-into (bindings-into before (vec shadow-allocs))
                      (concat backward-maps backward-reduces))
        [rev-ctx gradients] (lift-pullback-over-absence
                             before after (vec (concat d-read-arr-syms d-scalar-syms)) d-out-sym)
        array-gradients (subvec gradients 0 (count d-read-arr-syms))
        scalar-gradients (subvec gradients (count d-read-arr-syms))]
    {:rev-ctx rev-ctx
     :adj-env (wire-read-adjoints adj-env read-arrs array-gradients active-free scalar-gradients)}))

(defmethod emit-backward :par-reduce
  [{:keys [sym read-arrs d-read-arr-syms d-scalar-syms active-free
           shadow-allocs backward-maps backward-reduces d-acc-sym
           bwd-result-sym init-expr]}
   _adj-sym activity {:keys [adj-env rev-ctx]}]
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
        rev-ctx (reduce bindings-into rev-ctx backward-reduces)
        adj-env (wire-read-adjoints adj-env read-arrs d-read-arr-syms
                                    active-free d-scalar-syms)
        ;; The carry cotangent after reversing iteration zero belongs to the
        ;; reduction init, including the zero-trip case.
        adj-env (if (and (symbol? init-expr) (get activity init-expr false))
                  (update adj-env init-expr (fnil conj [])
                          (list 'nth bwd-result-sym
                                (clojure.core/+ (count read-arrs)
                                                (count active-free))))
                  adj-env)]
    {:rev-ctx rev-ctx
     :adj-env adj-env}))

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
  ([records activity seed-adj-env]
   (reverse-pass records activity seed-adj-env (:value-facts *source-memory-context*)))
  ([records activity seed-adj-env value-facts]
  (reduce
   (fn [{:keys [adj-env rev-ctx pullback-replays pullback-replay-conditions]} {:keys [type sym] :as record}]
     (let [product? (contains? record :cotangent-ports)
           incoming-ports (when product?
                            (filter #(seq (get adj-env (:key %))) (:cotangent-ports record)))
           product-adjoints (when product?
                              (reduce (fn [{:keys [adjoints rev-ctx]} {:keys [key tag]}]
                                        (let [scalar? (= :scalar (:kind (tangent/tangent-kind tag)))
                                              raw (ad-gensym "field_contribution" (when-not scalar? tag))
                                              accumulated (sum-contribs-into rev-ctx raw (get adj-env key) nil)
                                              value (if scalar? (ad-gensym "field_cotangent" tag) raw)
                                              accumulated (if scalar?
                                                            ;; A primal-space declaration is not evidence
                                                            ;; that incoming arithmetic used that dtype.
                                                            ;; Project before attaching the field tag.
                                                            (bindings-into accumulated
                                                                           [value (tangent/project-expr tag raw)])
                                                            accumulated)]
                                          {:adjoints (assoc adjoints key value)
                                           :rev-ctx accumulated}))
                                      {:adjoints {} :rev-ctx rev-ctx} incoming-ports))
           rev-ctx (if product? (:rev-ctx product-adjoints) rev-ctx)
           ;; Array writers can receive cotangents through either their result
           ;; or any declared output buffer. No incoming edge means no reverse
           ;; execution; an explicit nil/zero-valued seed is still an edge.
           array-type? (boolean (seq (:written-arrs record)))
           output-demand? (when array-type?
                            (some #(seq (get adj-env %))
                                  (cons sym (:written-arrs record))))
           adj-contribs (when-not (or product? array-type?) (get adj-env sym))
           sym-tag (:raster.type/tag (meta sym))
           scalar? (= :scalar (:kind (tangent/tangent-kind sym-tag)))
           ;; A sum of scalar cotangents need not have the primal's physical
           ;; carrier (Float fan-in may accumulate in Double). Only an actual
           ;; projection may establish that tag, as for product fields above.
           accumulated-tag (when-not scalar? sym-tag)
           adj-sym (if product? (:adjoints product-adjoints)
                       (when-not array-type? (ad-gensym (str "d_" (name sym)) accumulated-tag)))
           rev-ctx (if (and (not product?) adj-sym)
                     ;; Presence and physical array lowering are established by
                     ;; the shared producer pass, never from the target tag.
                     (sum-contribs-into rev-ctx adj-sym adj-contribs nil)
                     rev-ctx)
           raw-adjoint adj-sym
           adj-sym (if (and scalar? (not product?) (not array-type?))
                     (ad-gensym "scalar_cotangent" sym-tag) raw-adjoint)
           rev-ctx (if (and scalar? (not product?) (not array-type?))
                     ;; Establish the tangent port BEFORE exposing its tag to
                     ;; rule specialization. This mirrors structural field Π.
                     (bindings-into rev-ctx
                                    [adj-sym (tangent/project-expr
                                              sym-tag raw-adjoint
                                              ;; This is the immediately preceding scalar
                                              ;; binding. Do not rebuild the whole reverse
                                              ;; environment at every node (quadratic).
                                              {:binding-map {raw-adjoint (peek (:bindings rev-ctx))}})])
                     rev-ctx)]
       (if (or (seq incoming-ports) output-demand? adj-contribs)
         (let [memory-effects (record-memory-effects record nil)]
           (assoc (emit-backward record adj-sym activity {:adj-env adj-env :rev-ctx rev-ctx})
                  :pullback-replays (into pullback-replays (:replays memory-effects))
                  :pullback-replay-conditions
                  (join-input-replay-conditions
                   pullback-replay-conditions
                   (record-replay-conditions record memory-effects value-facts))))
         {:adj-env adj-env :rev-ctx rev-ctx :pullback-replays pullback-replays
          :pullback-replay-conditions pullback-replay-conditions})))
   {:adj-env seed-adj-env :rev-ctx (bind-ctx/make-ctx ad-gensym)
    :pullback-replays #{} :pullback-replay-conditions {}}
   (reverse records))))

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

(defn- product-leaf-expression [value path]
  (reduce (fn [value index] (list 'clojure.core/nth value index)) value path))

(defn- product-expression [schema values]
  (letfn [(build [schema path]
            (if (= :product (:kind schema))
              (mapv (fn [index child] (build child (conj path index))) (range) (:fields schema))
              (get values path)))]
    (build schema [])))

(defn- collect-param-adjoints
  "Phase 3 (pure): bind a d_<param> for each active param into rev-ctx;
  return {:rev-ctx :param-adj-syms}. Internal ports preserve absent cotangents;
  the public boundary materializes a typed shape-matched zero.

  Π at the boundary: each accumulated scalar adjoint is PROJECTED into the
  primal param's tangent space. Input dtypes do not prove incoming cotangent
  dtypes: a Double-only function can return Float. The shared projection may
  elide itself only from retained evidence about the accumulated value.
  Array adjoints are anchored by their typed shadow buffers — no cast."
  ([active-params adj-env rev-ctx]
   (collect-param-adjoints active-params adj-env rev-ctx (:value-facts *source-memory-context*)))
  ([active-params adj-env rev-ctx value-facts]
  (let [;; Contributions refer only to the completed reverse context, not to final
        ;; parameter binders appended below. Snapshot once for all scalar projections.
        binding-map (into {} (map vec (partition 2 (:bindings rev-ctx))))]
    (reduce
     (fn [{:keys [rev-ctx param-adj-syms]} p]
       (if-let [{:keys [schema active-paths]} (get-in value-facts [p :product])]
         (let [leaves (filterv (fn [[path _]] (and (contains? active-paths path)
                                                 (seq (get adj-env (product-ports/field-port p path)))))
                               (product-ports/leaf-paths schema))
               parameters (mapv (fn [[_ leaf]] (ad-gensym "field_primal" (:tag leaf))) leaves)
               rev-ctx (bindings-into rev-ctx
                                      (vec (mapcat (fn [parameter [path _]]
                                                     [parameter (product-leaf-expression p path)])
                                                   parameters leaves)))
               field-env (into {} (map (fn [parameter [path _]]
                                         [parameter (get adj-env (product-ports/field-port p path))])
                                       parameters leaves))
               {next-ctx :rev-ctx fields :param-adj-syms}
               (collect-param-adjoints parameters field-env rev-ctx {})
               result (ad-gensym "d_product")
               values (zipmap (map first leaves) fields)]
           {:rev-ctx (bindings-into next-ctx [result (product-expression schema values)])
            :param-adj-syms (conj param-adj-syms result)})
         (let [tag (:raster.type/tag (meta p))
             {:keys [kind]} (tangent/tangent-kind tag)
             contribs (get adj-env p)
             adj-sym (ad-gensym (str "d_" (name p)) tag)
             ;; Internal region ports exchange absence, not public gradient
             ;; materializations. A fabricated zero can activate a singular
             ;; upstream pullback even when this parameter was never demanded.
             default (when-not (or (:internal-region? *generated-product-context*)
                                   (:internal-region? *source-memory-context*))
                       (param-zero-expr p))
             ;; Array and scalar contributions share the same optional sum.
             rev-ctx
             (if (and (= kind :array) (>= (count contribs) 2))
               (sum-contribs-into rev-ctx adj-sym contribs default)
               (let [raw (sum-contribs contribs default)
                     expr (if (= kind :scalar)
                            (tangent/project-expr
                             tag raw {:binding-map binding-map})
                            raw)]
                 (bindings-into rev-ctx [adj-sym expr])))
             public? (not (or (:internal-region? *generated-product-context*)
                              (:internal-region? *source-memory-context*)))
             public-sym (if public? (ad-gensym "public_cotangent" tag) adj-sym)
             rev-ctx (if public?
                       (bindings-into rev-ctx
                                      [public-sym (list 'if (list 'clojure.core/nil? adj-sym)
                                                        default adj-sym)])
                       rev-ctx)]
         {:rev-ctx rev-ctx
          :param-adj-syms (conj param-adj-syms public-sym)})))
     {:rev-ctx rev-ctx :param-adj-syms []}
     active-params))))

(defn- reverse-program-projection
  "Project one output seed environment through an already checked forward
   program. Does not normalize or execute another forward pass. Replay facts
   remain raw here; the owning composition closes them over its alias graph."
  [{:keys [records activity value-facts]} active-params seed-adj-env]
  (let [{:keys [adj-env rev-ctx pullback-replays pullback-replay-conditions]}
        (reverse-pass records activity seed-adj-env value-facts)
        {:keys [rev-ctx param-adj-syms]}
        (collect-param-adjoints active-params adj-env rev-ctx value-facts)]
    {:adj-env adj-env :rev-ctx rev-ctx :param-adj-syms param-adj-syms
     :pullback-replays pullback-replays
     :pullback-replay-conditions pullback-replay-conditions}))

(defn- process-bindings
  "The shared engine: run Phases 1-3 over a NORMALIZED binding vector and return
  the raw pieces. Pure — no atoms. Wrappers assemble their own output shape.
  opts: {:seed-adj-env <map sym->[adj-expr]>}, or a local :seed-projection
  that constructs field seeds from the completed forward interface."
  [norm-bindings active-params {:keys [seed-adj-env seed-projection]}]
  (binding [util/*shadowing-locals* (into (into util/*shadowing-locals* active-params)
                                       (take-nth 2 norm-bindings))]
    (let [{:keys [activity fwd-bindings records replayed writes value-facts] :as forward-state}
          (forward-pass norm-bindings active-params)
          {:keys [adj-env rev-ctx pullback-replays pullback-replay-conditions param-adj-syms]}
          (reverse-program-projection forward-state active-params
                                      (if seed-projection (seed-projection forward-state) seed-adj-env))]
      {:fwd-bindings fwd-bindings :records records :adj-env adj-env :activity activity
       :rev-ctx rev-ctx :param-adj-syms param-adj-syms :replays replayed :writes writes
       :value-facts value-facts :forward-scope (source-memory-scope forward-state)
       :pullback-replay-conditions pullback-replay-conditions
       :pullback-replays (memory/transitive-closure pullback-replays (:aliases forward-state))})))

(declare gen-reverse-loop-with-let gen-reverse-loop-with-let-program lift-loop-to-tail)

(defn- loop-init? [init]
  (and (seq? init) (contains? #{'loop 'loop*} (first init))))

(defn- let-gradient-pieces
  "Run the shared reverse engine once and expose its flat forward and reverse
  bindings. Both a reified pullback and an in-loop step gradient consume this
  representation; the latter must not allocate a closure per iteration."
  [bindings body-exprs active-params dy-sym]
  (let [[norm-bindings body-sym] (normalize-for-ad bindings body-exprs)
        {:keys [fwd-bindings rev-ctx param-adj-syms replays writes value-facts forward-scope adj-env
                pullback-replays pullback-replay-conditions]}
        (process-bindings norm-bindings active-params
                          {:seed-projection
                           (fn [{:keys [value-facts]}]
                             (if-let [{:keys [active-paths]} (get-in value-facts [body-sym :product])]
                               (into {} (map (fn [path]
                                               [(product-ports/field-port body-sym path)
                                                [(product-leaf-expression dy-sym path)]])) active-paths)
                               {body-sym [dy-sym]}))})]
    {:body-sym body-sym :dy dy-sym
     :fwd-bindings fwd-bindings
     :rev-bindings (vec (:bindings rev-ctx))
     :replays replays :writes writes :value-facts value-facts :forward-scope forward-scope
     :result-product (get-in value-facts [body-sym :product])
     :parameter-product-demands
     (into {} (keep (fn [parameter]
                        (when-let [{:keys [active-paths]} (get-in value-facts [parameter :product])]
                          [parameter (into #{} (filter #(seq (get adj-env (product-ports/field-port parameter %)))) active-paths)])))
             active-params)
     :pullback-replays pullback-replays
     :pullback-replay-conditions pullback-replay-conditions
     :param-adj-syms param-adj-syms}))

(defn- reverse-program-forms
  "Project shared reverse pieces into existing primal/closure forms. Memory
   facts belong to the pieces, not to free-symbol inspection of emitted code."
  [{:keys [body-sym dy fwd-bindings rev-bindings param-adj-syms]}]
  (let [pullback-body (list 'let* rev-bindings (vec param-adj-syms))
        result-sym (ad-gensym "result")
        augmented-bindings (vec (concat fwd-bindings [result-sym body-sym]))]
    {:primal-form (list 'let* augmented-bindings result-sym)
     :pullback-form pullback-body
     :result-sym result-sym
     :standard-form (list 'let* augmented-bindings
                          (vector result-sym (list 'fn* [dy] pullback-body)))}))

(defn- gen-reverse-let-program-core
  "Generate reverse-mode AD code for a let* form: normalize → process-bindings →
  assemble [primal, pullback-fn]."
  [bindings body-exprs active-params]
  (let [pieces (let-gradient-pieces bindings body-exprs active-params 'dy__rad)
        aliases (get-in pieces [:forward-scope :aliases])]
    {:form (:standard-form (reverse-program-forms pieces))
     :writes (:writes pieces) :replays (:pullback-replays pieces)
     :value-facts (:value-facts pieces) :aliases aliases
     ;; Positive MAY identity, including private IDs, never exact :alias evidence.
     :result-aliases (memory/transitive-closure #{(:body-sym pieces)} aliases)}))

(defn- gen-inline-step-gradients
  "Emit a closure-free scalar-step pullback inside an enclosing ordered loop.
  The caller has already bound `adjoint`; every primal and reverse binding is
  evaluated in that loop iteration and the result is the ordered gradient
  vector for `active-params`."
  [bindings body-exprs active-params adjoint]
  (let [dy-sym (ad-gensym "step_dy")
        {:keys [fwd-bindings rev-bindings param-adj-syms]}
        (let-gradient-pieces bindings body-exprs active-params dy-sym)]
    (list 'let* (vec (concat fwd-bindings [dy-sym adjoint] rev-bindings))
          (vec param-adj-syms))))

(defn- gen-reverse-let-program
  "Generate reverse-mode AD code for a let* form.

  A raw `loop` bound to a let* binding is not recordable by the per-binding
  engine (`ad-record :loop` fails loud): only a loop in TAIL position is
  differentiable (via `lift-loop-to-tail` → `gen-reverse-loop-with-let`).
  A SECOND raw loop composed in series lands as a mid-let binding inside the FIRST loop's
  result-branch continuation (gen-reverse-loop-program*'s compound-result path funnels
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
      (gen-reverse-let-program-core bindings body-exprs active-params)
      ;; Thread activity across the bindings (init-active?, same rule the engine
      ;; uses) WITHOUT running ad-record (which would throw on the active loop).
      ;; Select the first ACTIVE loop with this activity authority. Earlier
      ;; inactive loops stay in the primal prefix, rather than becoming tapes.
      (let [activity (reduce (fn [act [s init]] (assoc act s (init-active? init act)))
                             (into {} (map (fn [p] [p true]) active-params))
                             pairs)
            first-loop-sym (ffirst (filter (fn [[sym init]]
                                            (and (loop-init? init) (get activity sym))) pairs))]
        (if-not first-loop-sym
          (gen-reverse-let-program-core bindings body-exprs active-params)
          (let [_ (assert-source-scan-checkpoint!
                   (second (first (filter #(= first-loop-sym (first %)) pairs))) true)
                lifted (lift-loop-to-tail (apply list 'let* (vec bindings) body-exprs)
                                          first-loop-sym)]
            (cond
              ;; lift-loop-to-tail → bare tail loop (no bindings preceded it)
              (and (seq? lifted) (contains? #{'loop 'loop*} (first lifted)))
              (gen-reverse-loop-with-let-program [] lifted active-params)

              ;; lift-loop-to-tail → (let* [pre...] (loop ...))
              (and (seq? lifted) (contains? #{'let 'let*} (first lifted))
                   (let [tail (drop 2 lifted)]
                     (and (= 1 (count tail)) (loop-init? (first tail)))))
              (let [[_ pre-bindings loop-form] lifted]
                (gen-reverse-loop-with-let-program (vec pre-bindings) loop-form active-params))

              ;; lift declined (non-canonical loop body) — preserve fail-loud.
              :else (gen-reverse-let-program-core bindings body-exprs active-params))))))))

(defn- gen-reverse-let [bindings body-exprs active-params]
  (:form (gen-reverse-let-program bindings body-exprs active-params)))

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
        lexical-names (into (into (set active-params) loop-syms)
                            (map first recur-bindings))
        dependencies (fn [expression]
                       (binding [util/*shadowing-locals*
                                 (into util/*shadowing-locals* lexical-names)]
                         (util/free-syms expression)))
        ;; Build a dependency map for recur-bindings: {sym -> #{dep-syms}}
        rb-deps (into {} (map (fn [[sym expr]] [sym (dependencies expr)])
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
                    init-syms (dependencies init)
                    update-syms (expand-through-rb (dependencies update-expr))
                    all-dep-syms (into init-syms update-syms)]]
        (when (and (not (contains? @active-set sym))
                   (some @active-set all-dep-syms))
          (swap! active-set conj sym)
          (reset! changed? true))))
    (filterv @active-set loop-syms)))

(declare gen-reverse-loop-with-let)
(declare gen-reverse-loop-program*)
(declare hoist-nested-lets)

(defn- gen-reverse-loop-program
  "Generate reverse-mode AD code for a loop* form.

  Strategy (Myia's closures-as-tape):
  - Forward: run loop, at each iteration compute per-iteration pullback
    closure via gen-reverse-let, store pullback in ArrayList
  - Backward: iterate stored pullbacks in reverse, accumulate adjoints
  - Active loop var adjoints chain across iterations
  - Free param adjoints accumulate across all iterations"
  [loop-form active-params]
  (let [active? (init-active? loop-form (zipmap active-params (repeat true)))]
    (assert-source-scan-checkpoint! loop-form active?)
    ;; The generic tape route is not a convergence/while-solver contract.
    ;; Preserve the canonical materializer's existing invariant-bound gate
    ;; rather than accidentally admitting a data-dependent trip count here.
    (when active?
      (when-let [{:keys [acc-sym bound-expr]} (patterns/match-reduce-loop loop-form)]
        (when (patterns/contains-sym? bound-expr acc-sym)
          (ad-record :loop '_loop loop-form {})))))
  ;; Initializers execute once, in source order, including on zero-trip loops.
  ;; Lift compound inactive initializers too: their effects and aliases must
  ;; pass through the same checked prefix as differentiated initializers.
  ;; E.g. (loop [a (* x 2.0) ...] ...) → (let* [a_init (* x 2.0)] (loop [a a_init ...] ...))
  ;; Once lifting is needed, preserve the entire sequential initializer region;
  ;; inactive origins can trap or be referenced by later active initializers.
  (let [raw-bindings (vec (partition 2 (second loop-form)))]
    (if (some (comp seq? second) raw-bindings)
      ;; Loop initializers are sequential bindings, not independent arguments.
      ;; If any compound initializer needs lifting, retain *all* initializers in
      ;; source order and substitute earlier bindings capture-avoidantly.
      ;; Lifting only active expressions could move a read before an earlier
      ;; trapping origin, or leave its reference to that origin out of scope.
      (let [lifted (:lifted
                    (reduce (fn [{:keys [lifted substitution]} [sym init]]
                              (let [init-sym (ad-gensym (str "init_" (name sym))
                                                        (:raster.type/tag (meta sym)))]
                                {:lifted (conj lifted
                                               {:sym sym :init (util/subst-syms substitution init)
                                                :init-sym init-sym})
                                 :substitution (assoc substitution sym init-sym)}))
                            {:lifted [] :substitution {}} raw-bindings))
            let-bindings (vec (mapcat (fn [{:keys [init-sym init]}]
                                        [init-sym init])
                                      lifted))
            new-loop-bindings (vec (mapcat (fn [{:keys [sym init-sym]}] [sym init-sym])
                                           lifted))
            new-loop-form (with-meta (list* 'loop* new-loop-bindings (drop 2 loop-form))
                            (meta loop-form))]
        (gen-reverse-loop-with-let-program let-bindings new-loop-form active-params))
      (gen-reverse-loop-program* loop-form active-params))))

(defn- gen-reverse-loop [loop-form active-params]
  (:form (gen-reverse-loop-program loop-form active-params)))

(defn- gen-reverse-loop-program*
  "Core reverse-mode loop transform (all init expressions are trivial)."
  [loop-form active-params]
  (let [{:keys [bindings loop-syms test-expr recur-args
                result-expr recur-in-then? recur-bindings]} (parse-loop-form loop-form)
        recur-bindings (or recur-bindings [])

        ;; Determine active loop vars (include recur-binding dependencies)
        active-loop-vars (loop-var-activity bindings recur-args active-params recur-bindings)

        ;; These fragments are not visited by the existing per-active-port
        ;; forward programs. A write summary alone cannot close their lexical
        ;; aliases or establish source-order replay safety. Refuse rather than
        ;; exporting private write IDs as if a child program had checked them.
        test-writes (written-syms test-expr)
        unchecked-recur-writes
        (into #{} (mapcat written-syms)
              (keep-indexed (fn [index expression]
                              (when-not (some #{(nth loop-syms index)} active-loop-vars)
                                expression)) recur-args))
        unchecked-prelude-writes
        (when (empty? active-loop-vars)
          (written-syms (cons 'do (map second recur-bindings))))
        _ (doseq [[phase writes] [[:test test-writes]
                                 [:inactive-recur unchecked-recur-writes]
                                 [:inactive-prelude unchecked-prelude-writes]]
                  :when (seq writes)]
            (throw (ex-info "Reverse loop has a writing fragment outside its checked forward programs"
                            {:reason :unclosed-loop-effect :region :ordered-loop
                             :phase phase :arrays (vec (sort-by str writes))})))

        ;; All active symbols: active params + active loop vars
        all-active (vec (distinct (concat active-params active-loop-vars)))

        ;; Determine which loop var is the result
        ;; (result-expr should be a loop var symbol)
        result-var (when (and (symbol? result-expr)
                              (some #{result-expr} loop-syms))
                     result-expr)

        ;; === Per active-loop-var transform data ===
        ;; For each active loop var, gen-reverse-let transforms its update expr.
        ;; iter-active: only active loop vars + active params (gradients returned for these).
        ;; All loop vars are passed to gen-reverse-let as active-params so expressions
        ;; in recur-bindings (e.g. aget W i) can reference loop counters like i.
        ;; Non-active loop vars get nil gradients which are harmlessly discarded.
        iter-active (vec (distinct (concat active-loop-vars active-params)))

        ;; Linearize the exit where it actually executes. Re-running its primal
        ;; in the pullback would repeat inactive effects and reread mutable data.
        terminal-pair-sym (ad-gensym "terminal_vpb")
        terminal-program
        (when-not result-var
          (let [[bindings body] (extract-let-parts (hoist-nested-lets result-expr))]
            (gen-reverse-let-program bindings body iter-active)))
        terminal-transform (:form terminal-program)

        per-var-data
        (mapv (fn [lv]
                (let [idx (.indexOf loop-syms lv)
                      update-expr (nth recur-args idx)
                      ;; Preserve the update value's retained declaration. The
                      ;; carry name is not an independent type inference rule.
                      nv-sym (ad-gensym (str "nv_" (name lv))
                                       (:raster.type/tag (meta update-expr)))
                      vpb-sym (ad-gensym (str "vpb_" (name lv)))
                      val-sym (ad-gensym (str "val_" (name lv)))
                      pb-sym (ad-gensym (str "pb_" (name lv)))
                      ;; Include recur-branch let bindings so the AD transform
                      ;; differentiates through aget/math inside the loop body.
                      all-bindings (vec (concat (mapcat identity recur-bindings)
                                                [nv-sym update-expr]))]
                  (let [program (gen-reverse-let-program all-bindings [nv-sym] iter-active)]
                  {:lv lv
                   :idx idx
                   :nv-sym nv-sym
                   :vpb-sym vpb-sym
                   :val-sym val-sym
                   :pb-sym pb-sym
                   :program program :transform (:form program)})))
              active-loop-vars)

        ;; These summaries describe the existing emitted transforms. Raw known
        ;; writes add denial evidence only; absence is not an effect certificate.
        step-programs (mapv :program per-var-data)
        programs (cond-> step-programs terminal-program (conj terminal-program))
        child-aliases (reduce #(merge-with into %1 (:aliases %2))
                              (:aliases *source-memory-context*) programs)
        loop-aliases
        (:aliases
         (reduce (fn [state [index [sym init]]]
                   (let [program (:program (some #(when (= sym (:lv %)) %) per-var-data))
                         alternatives (into (:symbols (form/normal-exit-facts init))
                                            (into (:symbols (form/normal-exit-facts (nth recur-args index)))
                                                  (:result-aliases program)))]
                     (memory/track-aliases state sym alternatives #{})))
                 {:aliases child-aliases :consumed #{}} (map-indexed vector bindings)))
        step-writes (memory/transitive-closure
                     (into (written-syms (cons 'do (concat (map second recur-bindings) recur-args)))
                           (mapcat :writes step-programs)) loop-aliases)
        terminal-writes (memory/transitive-closure (set (:writes terminal-program)) loop-aliases)
        step-replays (memory/transitive-closure (into #{} (mapcat :replays step-programs)) loop-aliases)
        terminal-replays (memory/transitive-closure (set (:replays terminal-program)) loop-aliases)
        invalidated (set/intersection step-replays (into step-writes terminal-writes))
        _ (when (seq invalidated)
            (throw (ex-info "Reverse loop iterations invalidate outstanding pullback replay contents"
                            {:reason :replayed-array-overwritten :region :ordered-loop
                             :arrays (vec (sort-by str invalidated))})))
        program-writes (memory/transitive-closure
                        (into (written-syms loop-form) (mapcat :writes programs)) loop-aliases)
        program-facts (reduce #(merge %1 (:value-facts %2))
                              (into (:value-facts *source-memory-context*)
                                    (map (fn [sym] [sym {:value sym}]) loop-syms)) programs)

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
              ;; Keep the exit pullback beside its primal on the existing tape.
              terminal-expr (if result-var
                              result-expr
                              (list 'let* [terminal-pair-sym terminal-transform]
                                    (cons 'vector
                                          (concat loop-syms
                                                  [(list 'nth terminal-pair-sym 0)
                                                   (list 'nth terminal-pair-sym 1)]))))]
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

        ;; Seed the backward recurrence with the saved exit pullback. Its
        ;; closure captures final carry values without substituting source forms.
        n-loop-syms (count loop-syms)

        result-seed
        (when-not result-var
          (let [rpb-sym (ad-gensym "rpb")
                rg-sym (ad-gensym "rgrads")
                lv-seed-syms (mapv (fn [lv] (ad-gensym (str "seed_" (name lv))))
                                   active-loop-vars)
                param-seed-syms (mapv (fn [p] (ad-gensym (str "seed_" (name p))))
                                      active-params)]
            {:pullback-bindings
             (vec (concat
                   [rpb-sym (list 'nth loop-result-sym (inc n-loop-syms))
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
        ;; [lv0 lv1 ... result-value result-pullback]; expose only the primal.
        result-value-expr (if result-var
                            loop-result-sym
                            (list 'nth loop-result-sym n-loop-syms))]
    {:writes program-writes :replays (into step-replays terminal-replays)
     :aliases loop-aliases :value-facts program-facts
     :result-aliases (memory/transitive-closure
                      (if result-var #{result-var} (set (:result-aliases terminal-program))) loop-aliases)
     :form (list 'let* [pullbacks-sym (list 'java.util.ArrayList.)
                 loop-result-sym forward-loop-form]
          (vector result-value-expr pullback-code))}))

(defn- gen-reverse-loop-with-let-program
  "Handle (let* [bindings...] (loop ...)) by wrapping the let bindings
  around the loop and composing pullbacks.
  The let bindings compute values used by the loop, and the loop
  pullback returns adjoints for those values, which then chain
  backward through the let bindings."
  [let-bindings loop-form active-params]
  (let [;; A loop initializer can introduce a nested let after ad-prepare's
        ;; outer hoisting. Flatten that newly lifted prefix through the same
        ;; ordered projection before computing activity and its pullback.
        normalized-prefix (hoist-nested-lets (list 'let* let-bindings nil))
        anf-bindings (anf-normalize-bindings (second normalized-prefix))
        ;; PURE forward pass — now handles par/dotimes/if (the capability the old
        ;; hand-rolled copy lacked). This is what lets an array-valued INTERMEDIATE
        ;; produced in the let-prefix (e.g. a broadcast) and consumed by the loop
        ;; differentiate: its par/map! backward is now generated here.
        {:keys [activity fwd-bindings] :as forward-state}
        (forward-pass anf-bindings active-params)
        let-active-syms (filterv #(get activity % false) (map first (partition 2 anf-bindings)))
        ;; Extended active set the loop sees: original params + active let-syms.
        loop-active (vec (distinct (concat active-params let-active-syms)))
        loop-program (binding [*source-memory-context* (source-memory-scope forward-state)]
                       (gen-reverse-loop-program loop-form loop-active))
        loop-transform (:form loop-program)
        dy-sym 'dy__rad
        loop-vpb-sym (ad-gensym "lvpb")
        loop-val-sym (ad-gensym "lval")
        loop-pb-sym  (ad-gensym "lpb")
        loop-grads-sym (ad-gensym "lgrads")
        ;; Seed the let-prefix reverse pass with the loop pullback's adjoints
        ;; (bound at pullback-call time as `loop-grads-sym`).
        seed-adj-env (into {} (map-indexed (fn [i s] [s [(list 'nth loop-grads-sym i)]])
                                           loop-active))
        {:keys [rev-ctx pullback-replays param-adj-syms]}
        (reverse-program-projection forward-state active-params seed-adj-env)
        aliases (merge-with into (:aliases forward-state) (:aliases loop-program))]
    ;; Assemble: forward let-prefix + loop + composed pullback.
    {:writes (into (:writes forward-state) (:writes loop-program))
     :replays (memory/transitive-closure (into pullback-replays (:replays loop-program)) aliases)
     :aliases aliases :value-facts (merge (:value-facts forward-state) (:value-facts loop-program))
     :result-aliases (memory/transitive-closure (:result-aliases loop-program) aliases)
     :form (list 'let* (vec (concat fwd-bindings
                             [loop-vpb-sym loop-transform
                              loop-val-sym (list 'nth loop-vpb-sym 0)
                              loop-pb-sym  (list 'nth loop-vpb-sym 1)]))
          (vector loop-val-sym
                  (list 'fn* [dy-sym]
                        (list 'let* (vec (concat [loop-grads-sym (list loop-pb-sym dy-sym)]
                                                 (:bindings rev-ctx)))
                              (vec param-adj-syms)))))}))

(defn- gen-reverse-loop-with-let [let-bindings loop-form active-params]
  (:form (gen-reverse-loop-with-let-program let-bindings loop-form active-params)))

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

        ;; The backward reads d_out at the loop counter and differentiates one
        ;; store; it scatter-adds each read's cotangent at the read's index,
        ;; recomputed per step (the counter, a literal or a gather such as
        ;; alpha[group[i]]). Decline anything else rather than transpose it to
        ;; the wrong element.
        locals (set (map first scalar-bindings))
        readable? (fn [{:keys [arr idx]}]
                    (or (not (gradient-bearing-array-read? (list 'aget arr idx)))
                        (= counter-sym idx)
                        (integer? idx)
                        (replayable-gather-index? idx counter-sym locals)))
        _ (when (or (< 1 (count asets))
                    (some #(not= counter-sym (:idx %)) asets)
                    (not-every? readable? agets))
            (throw (ex-info
                    (str "dotimes AD: the loop must store once, at the counter `" counter-sym
                         "`, and read active arrays at the counter, a literal or a gather "
                         "index the backward can recompute.")
                    {:reason :dotimes-unsupported-access :asets asets
                     :reads (mapv #(select-keys % [:arr :idx]) agets)})))
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
        ;; element type (inherits :raster.type/tag; absent tag fails loud — plan D2).
        d-read-arr-syms (mapv (fn [arr]
                                (ad-gensym (str "d_" (name arr))
                                           (or (:raster.type/tag (meta arr))
                                               (lost-tag! :dotimes-read-arr (name arr)))))
                              read-arrs)
        arr->d-sym (zipmap read-arrs d-read-arr-syms)
        d-scalar-syms (mapv (fn [p]
                              (ad-gensym (str "d_" (name p) "_acc")
                                         (or (:raster.type/tag (meta p))
                                             (lost-tag! :dotimes-scalar (name p)))))
                            active-free)

        ;; Backward loop: reads d_out, calls pullbacks, accumulates
        ;; d-out-sym is the gradient of the output array — bound by gen-reverse-let
        ;; from adj-env contributions. Tag inherits from the output array type
        ;; (absent tag fails loud — plan D2).
        out-array-tag (or (some-> written-arrs first meta :raster.type/tag)
                          (lost-tag! :dotimes-out-arr (some-> written-arrs first name)))
        out-elem-tag (case out-array-tag
                       doubles 'double, floats 'float, longs 'long, ints 'int,
                       (lost-tag! :dotimes-out-elem out-array-tag))
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
              ;; one contribution per read, accumulated into its array's shadow
              (mapcat (fn [i {:keys [arr idx]}]
                        (let [d-arr-sym (arr->d-sym arr)
                              at (util/subst-syms {counter-sym j-sym} idx)]
                          [(ad-gensym "_scatter")
                           (list 'raster.arrays/aset d-arr-sym at
                                 (list 'raster.ad.reverse/grad-acc
                                       (list 'aget d-arr-sym at)
                                       (list 'nth grads-iter-sym i)))]))
                      (range) agets)))

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
     :replays (index-replays agets counter-sym)
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

(defn- replayable-gather-index?
  "A sequential scan may scatter at a data-dependent index when the backward
  iteration can recompute that index exactly: a pure expression of the scan
  index and loop-invariant values, reading only constant arrays. Body locals
  are excluded because lifted reads are replayed before the scalar bindings."
  [index idx-sym locals]
  (let [free (util/free-syms index)]
    (and (seq? index)
         (contains? free idx-sym)
         (not-any? locals free)
         (not-any? gradient-bearing-array-read? (tree-seq coll? seq index))
         (= :pure (effects/analyze-effect index)))))

(defn- scan-read-lifter
  "Returns (fn lift [expr scoped? locals]) that replaces each admitted active
  array read with (on-read arr index). Reads at the map index are admitted
  everywhere; sequential scans additionally admit unconditional reads at an
  integer literal or a replayable gather index. Consumers that enable gather
  reads must scatter-add their cotangents; a lane-wise overwrite would race."
  [idx-sym sequential? on-read & [replay-index]]
  (fn lift [expr scoped? locals]
    (let [[array original-index] (when (gradient-bearing-array-read? expr)
                                  (op/call-args expr))
          index (when (gradient-bearing-array-read? expr)
                  (if (and replay-index (not scoped?))
                    (replay-index original-index)
                    original-index))]
     (cond
      (and (gradient-bearing-array-read? expr)
           (or (= idx-sym index)
               (and sequential? (not scoped?)
                    (not (contains? locals array))
                    (or (integer? index)
                        (replayable-gather-index? index idx-sym locals)))))
      (on-read array index)
      (seq? expr)
      (let [{:keys [kind liftable?]} (form/form-info expr)
            nested-scope? (or scoped? (= :binding kind) (not liftable?))]
        (with-meta (apply list (map #(lift % nested-scope? locals) expr)) (meta expr)))
      (vector? expr) (with-meta (mapv #(lift % scoped? locals) expr) (meta expr))
      :else expr))))

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
  Inline reads use the map index by default. Sequential scan may additionally admit literal
  or replayable gather indices in unconditional scalar expressions (see scan-read-lifter);
  its backward loop scatter-adds at those indices."
  ([body-expr idx-sym] (analyze-par-map-body body-expr idx-sym false))
  ([body-expr idx-sym literal-index-reads?]
   (analyze-par-map-body body-expr idx-sym literal-index-reads? nil))
  ([body-expr idx-sym literal-index-reads? replay-forbidden-values]
  (let [agets (atom [])
        ;; Extract let* bindings if present
        [bindings body-result0]
        (if (and (seq? body-expr) (#{'let 'let*} (first body-expr)))
          [(partition 2 (second body-expr)) (last (drop 2 body-expr))]
          [[] body-expr])
        ;; Map index replay may expand only pure sequential aliases independent
        ;; of active captures and the destination being written. Keep the
        ;; original local initializers at their original evaluation sites; these
        ;; expressions are proof/replay data, not a reordered scalar program.
        local-ids (set (map first bindings))
        index-alias-state
        (when (some? replay-forbidden-values)
          (binding [util/*shadowing-locals* (into util/*shadowing-locals*
                                                (conj local-ids idx-sym))]
            (reduce
             (fn [{:keys [aliases snapshots]} [id initializer]]
               (let [expanded (util/subst-syms aliases initializer)
                     free (util/free-syms expanded)]
                 {:snapshots (conj snapshots aliases)
                  :aliases (if (and (not-any? local-ids free)
                                    (not-any? replay-forbidden-values free)
                                    (not-any? gradient-bearing-array-read?
                                              (tree-seq coll? seq expanded))
                                    (= :pure (effects/analyze-effect expanded)))
                             (assoc aliases id expanded)
                             (dissoc aliases id))}))
             {:aliases {} :snapshots []} bindings)))
        index-aliases (:aliases index-alias-state)
        alias-snapshots (or (:snapshots index-alias-state) (repeat nil))]
    ;; (1) let*-bound agets
    (doseq [[[sym init] aliases] (map vector bindings alias-snapshots)]
      (when (gradient-bearing-array-read? init)
        (let [[array index] (op/call-args init)]
          (swap! agets conj {:sym sym :arr array
                            :idx (if aliases
                                   (util/subst-syms aliases index)
                                   index)}))))
    ;; (2) lift admitted inline reads, deduplicating by array AND original index.
    ;; Literal reads in scan stay inside the reverse iteration, not outside a zero-trip loop.
    (let [seen (atom {})
          on-read (fn [arr index]
                  (let [k [arr index]]
                    (or (get @seen k)
                        (let [s (ad-gensym (str "ag_" (if (symbol? arr) (name arr) "arr")))]
                          (swap! seen assoc k s)
                          (swap! agets conj {:sym s :arr arr :idx index})
                          s))))
          lift-for (fn [aliases]
                     (scan-read-lifter idx-sym literal-index-reads? on-read
                                       (when aliases #(util/subst-syms aliases %))))
          lift (lift-for index-aliases)
          ;; These top-level bindings were stripped above. A lifted read is replayed
          ;; before scalar bindings, so it must not capture their aliases/shadows.
          body-result (lift body-result0 false (set (map first bindings)))
          scalar-bindings* (:bindings
                            (reduce (fn [{:keys [locals bindings]} [[s init] aliases]]
                                      {:locals (conj locals s)
                                       :bindings (if (gradient-bearing-array-read? init)
                                                   bindings
                                                   (conj bindings [s ((lift-for aliases) init false locals)]))})
                                    {:locals #{} :bindings []}
                                    (map vector bindings alias-snapshots)))]
      (let [aget-syms (set (map :sym @agets))
            bound-syms (set (cons idx-sym (concat (map first scalar-bindings*) aget-syms)))
            free-syms (util/free-syms body-result bound-syms)
            ;; Also collect free syms from scalar binding inits
            all-free (reduce into free-syms
                             (map (fn [[_ init]]
                                    (util/free-syms init bound-syms))
                                  scalar-bindings*))]
        {:agets @agets
         :scalar-bindings scalar-bindings*
         :body-result body-result
         ;; Remove arrays that are aget'd (they're inputs, not scalars)
         :free-syms (disj all-free idx-sym)})))))

(defn ^:no-doc assert-map-store-callee!
  "An active map's scalar pullback does not differentiate an arbitrary store
   callable. Admit only no callable or a source-resolved floating core cast.
   This preserves existing floating conversion semantics, not a new rounding policy."
  [map-form active?]
  (let [cast (nth map-form 4 nil)
        callee (:raster.op/resolved-callee (meta cast))]
    (when (and active? cast
               (not (and (symbol? cast) (symbol? callee)
                         (= "clojure.core" (namespace callee))
                         (some-> callee op/cast-result-tag dtype/dtype-for-scalar-tag dtype/fp-dtype?))))
      (throw (ex-info "AD map! store callable has no proved floating conversion authority"
                      {:reason :ad-map-store-callee :cast cast :resolved-callee callee
                       :form map-form})))))

(defn- gen-reverse-read-region
  "Lifted reads execute before this scalar child. Their scatter indices must
   survive its effects, unlike a child-owned replay established after a write."
  [bindings body active-params prior-replays]
  (let [prior-replays (set prior-replays)
        conditions (record-replay-conditions nil {:replays prior-replays}
                                             (:value-facts *source-memory-context*))
        visible (set (:values *source-memory-context*))
        program (binding [*source-memory-context*
                          (-> *source-memory-context*
                              (update :replayed join-input-replay-conditions
                                      (select-keys conditions visible))
                              (update :unprojectable join-input-replay-conditions
                                      (apply dissoc conditions visible)))]
                  (gen-reverse-let-program bindings body active-params))
        aliases (:aliases program)
        overwritten (set/intersection
                     (memory/transitive-closure (into prior-replays (keys conditions)) aliases)
                     (memory/transitive-closure (:writes program) aliases))]
    (when (seq overwritten)
      (throw (ex-info "Scalar child overwrites an index needed by its parent's pullback"
                      {:reason :replayed-array-overwritten
                       :arrays (vec (sort-by str overwritten))})))
    program))

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
        _ (binding [util/*shadowing-locals* (into util/*shadowing-locals* active-params)]
            (assert-map-store-callee!
             par-map-form (init-active? par-map-form (zipmap active-params (repeat true)))))
        {:keys [agets scalar-bindings body-result free-syms]}
        (analyze-par-map-body body-expr idx-sym true (conj (set active-params) out-sym))

        ;; Determine which arrays are written and read
        written-arrs [out-sym]
        ;; A read at the map index transposes lane-wise in a backward map. A
        ;; read at a literal or a gathered index (alpha[group[i]]) is a gather:
        ;; lanes may share a destination, so its cotangent is accumulated with
        ;; a scatter-add of per-lane values (an accumulator adjoint).
        locals (set (map first scalar-bindings))
        gather-read? (fn [{:keys [idx]}]
                       (or (integer? idx) (replayable-gather-index? idx idx-sym locals)))
        _ (doseq [{:keys [arr idx] :as read} agets]
            (when-not (or (= idx-sym idx) (gather-read? read))
              (throw (ex-info
                      (str "par/map! AD: active array `" arr "` is read at `" (pr-str idx)
                           "`, neither the map index `" idx-sym "` nor a gather index the "
                           "backward can recompute.")
                      {:reason :par-map-shifted-read :array arr :index idx}))))
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
        scalar-program
        (when (seq iter-active)
          (let [binds (if (seq pure-scalar-bindings)
                        (vec (concat pure-scalar-bindings [val-result-sym body-result]))
                        [val-result-sym body-result])]
            (gen-reverse-read-region binds [val-result-sym] iter-active
                                     (index-replays agets idx-sym))))
        scalar-transform (:form scalar-program)

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
                                (ad-gensym "_aset") (list 'raster.arrays/aset out-sym idx-sym
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
                                               (lost-tag! :parmap-read-arr (name arr)))))
                              read-arrs)
        d-scalar-syms (mapv (fn [p]
                              (ad-gensym (str "d_" (name p) "_acc")
                                         (or (:raster.type/tag (meta p))
                                             (lost-tag! :parmap-scalar (name p)))))
                            active-free)
        out-array-tag (or (some-> out-sym meta :raster.type/tag)
                          (lost-tag! :parmap-out-arr (some-> out-sym name)))
        out-elem-tag (case out-array-tag
                       doubles 'double, floats 'float, longs 'long, ints 'int,
                       (lost-tag! :parmap-out-elem out-array-tag))
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
        ;; Every read of an array contributes: sum its gradient slots (a slot's
        ;; position is the read's position in agets).
        lane-grad (fn [k-slots]
                    (list 'let* [iter-pb-sym (list 'aget tape-sym bwd-idx-sym)
                                 d-val-sym (list 'aget d-out-sym bwd-idx-sym)
                                 grads-iter-sym (list iter-pb-sym d-val-sym)]
                          (reduce (fn [acc k]
                                    (list 'raster.ad.reverse/grad-acc acc
                                          (list 'nth grads-iter-sym k)))
                                  (list 'nth grads-iter-sym (first k-slots))
                                  (rest k-slots))))
        backward-maps
        (mapv (fn [d-arr-sym arr-sym]
                (let [reads (keep-indexed (fn [k read] (when (= arr-sym (:arr read)) [k read])) agets)
                      store-tag (or (some-> arr-sym meta :raster.type/tag
                                            dtype/dtype-for-array-tag dtype/scalar-tag-for-dtype)
                                    (lost-tag! :parmap-adjoint-store (name arr-sym)))
                      store-cast (symbol "clojure.core" (name store-tag))
                      lane-slots (keep (fn [[k {:keys [idx]}]] (when (= idx-sym idx) k)) reads)
                      gathers (remove (fn [[_ {:keys [idx]}]] (= idx-sym idx)) reads)]
                  (vec (concat
                        ;; every lane read of the array, summed, overwrites its lane
                        (when (seq lane-slots)
                          [d-arr-sym (list 'raster.par/map! d-arr-sym bwd-idx-sym bound-expr store-cast
                                           (lane-grad lane-slots))])
                        ;; each gathered read: per-lane cotangents and indices, then
                        ;; their scatter-add into the array's cotangent
                        (mapcat (fn [[k {:keys [idx]}]]
                                  (let [vals-sym (ad-gensym "gather_d" (:raster.type/tag (meta arr-sym)))
                                        idxs-sym (ad-gensym "gather_idx" 'ints)]
                                    [vals-sym (list 'raster.arrays/zeros-like arr-sym bound-expr)
                                     vals-sym (list 'raster.par/map! vals-sym bwd-idx-sym bound-expr store-cast
                                                    (lane-grad [k]))
                                     idxs-sym (list 'clojure.core/int-array bound-expr)
                                     idxs-sym (list 'raster.par/map! idxs-sym bwd-idx-sym bound-expr 'int
                                                    (util/subst-syms {idx-sym bwd-idx-sym} idx))
                                     d-arr-sym (list 'raster.par/scatter! d-arr-sym vals-sym idxs-sym
                                                     bound-expr)]))
                                gathers)))))
              d-read-arr-syms read-arrs)

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
     :replays (when forward-code (into (index-replays agets idx-sym) (:replays scalar-program)))
     :input-replays (when forward-code (into (index-replays agets idx-sym) (:replays scalar-program)))
     :effects (when scalar-program
                {:kind :region
                 :regions [(assoc (select-keys scalar-program [:writes :replays :value-facts])
                                  :writes (into (set written-arrs) (:writes scalar-program))
                                  :replays (into (index-replays agets idx-sym) (:replays scalar-program)))]})
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

(declare ^:private carry-scan-dtype)
(declare ^:private carry-dtype-consistent?)

(defn- replay-pure-reduce-expr?
  "A narrow proof that a scalar reduction step may be evaluated again in reverse.
  Devirtualized calls can lose their original-op metadata in AD preparation, so
  resolve and check the concrete implementation var rather than trusting `.invk`
  or guessing a name from its mangle. Unknown calls keep the closure-tape route."
  [expr]
  (cond
    (seq? expr)
    (let [head (first expr)]
      (cond
        (contains? #{'let 'let*} head)
        (and (every? replay-pure-reduce-expr? (map second (partition 2 (second expr))))
             (every? replay-pure-reduce-expr? (drop 2 expr)))
        (contains? #{'if 'do} head)
        (every? replay-pure-reduce-expr? (rest expr))
        (= '.invk head)
        (let [impl-var (when (symbol? (second expr)) (resolve (second expr)))]
          (and (var? impl-var)
               (= :pure (effects/analyze-var-effect impl-var))
               (every? replay-pure-reduce-expr? (nnext expr))))
        :else
        (and (= :pure (purity/pure-op? head))
             (every? replay-pure-reduce-expr? (rest expr)))))
    (vector? expr) (every? replay-pure-reduce-expr? expr)
    :else true))

(defn- additive-step
  "{:sign ±1 :term t} when `expr` adds (or subtracts) a term to the
  accumulator that does not itself read the accumulator; else nil."
  [expr acc-sym]
  (when (seq? expr)
    (let [[op args] (if (= '.invk (first expr))
                      [(second expr) (nnext expr)]
                      [(first expr) (rest expr)])
          op-name (when (symbol? op) (name op))
          numeric? (and op-name (= "raster.numeric" (namespace op)))
          kind (cond
                 (contains? '#{+ clojure.core/+ raster.numeric/+} op) :plus
                 (contains? '#{- clojure.core/- raster.numeric/-} op) :minus
                 (and numeric? (.startsWith ^String op-name "_plus__m_")) :plus
                 (and numeric? (.startsWith ^String op-name "_minus__m_")) :minus)
          reads-acc? #(contains? (util/free-syms %) acc-sym)]
      (when (and kind (= 2 (count args)))
        (let [[a b] args]
          (cond
            (and (= acc-sym a) (not (reads-acc? b)))
            {:sign (if (= :plus kind) 1 -1) :term b}
            (and (= :plus kind) (= acc-sym b) (not (reads-acc? a)))
            {:sign 1 :term a}))))))

(declare ^:private untemplated-scan-call? lower-composites)

(defn- gen-reverse-additive-reduce
  "Reverse mode for a reduction whose step adds a term independent of the
  accumulator: acc_n = init ± Σ term(i). The accumulator's cotangent is the
  same at every step, so the backward needs no carry tape and no forward
  replay of the accumulator: one loop evaluates each term's gradient under
  that cotangent, scatter-adds array cotangents at each read's own index (the
  loop is sequential, so a gathered read such as alpha[group[i]] may repeat a
  destination) and sums scalar gradients. The forward is the source-order
  loop, unchanged. nil when the step is not additive or reads an active array
  the term's gradient cannot replay."
  [par-reduce-form active-params]
  (let [[_ acc-sym init-expr idx-sym bound-expr body-expr] par-reduce-form
        ;; as for a scan step: inline untemplated deftm calls before
        ;; differentiating the term
        step-expr (if (untemplated-scan-call? body-expr)
                    (lower-composites body-expr)
                    body-expr)
        analyzed (analyze-par-map-body step-expr idx-sym true)
        ;; ANF ends the step in `v = acc ± term; v`: the update is v's init
        {:keys [agets scalar-bindings body-result free-syms]}
        (let [{:keys [scalar-bindings body-result]} analyzed
              [last-sym last-init] (peek scalar-bindings)]
          (if (and (symbol? body-result) (= body-result last-sym))
            (assoc analyzed :scalar-bindings (pop scalar-bindings) :body-result last-init)
            analyzed))
        {:keys [sign term]} (additive-step body-result acc-sym)
        active-free (filterv #(contains? free-syms %) active-params)]
    (when (and term
               ;; The backward evaluates the prelude and term again. Additive
               ;; shape alone does not permit replaying writes or fresh draws;
               ;; use the same purity proof as the ordered carry replay route.
               (every? (comp replay-pure-reduce-expr? second) scalar-bindings)
               (replay-pure-reduce-expr? body-result)
               (not-any? (fn [[_ init]] (contains? (util/free-syms init) acc-sym))
                         scalar-bindings)
               ;; an active array outside the admitted reads would be a free scalar
               (not-any? #(= :array (:kind (tangent/tangent-kind (:raster.type/tag (meta %)))))
                         active-free)
               (not (and (seq? init-expr) (some (set active-params) (util/free-syms init-expr)))))
      (let [aget-syms (mapv :sym agets)
            iter-active (vec (distinct (concat aget-syms active-free)))
            pure-scalar-bindings (vec (mapcat identity
                                              (remove (fn [[sym _]] (contains? (set aget-syms) sym))
                                                      scalar-bindings)))
            d-acc-sym (ad-gensym "d_acc")
            d-term-sym (ad-gensym "d_term")
            term-sym (ad-gensym "term")
            [_ step-bindings step-grads]
            (gen-inline-step-gradients (conj pure-scalar-bindings term-sym term)
                                       [term-sym] iter-active d-term-sym)
            read-arrs (vec (distinct (map :arr agets)))
            d-read-arr-syms (mapv (fn [arr]
                                    (ad-gensym (str "d_" (name arr))
                                               (or (:raster.type/tag (meta arr))
                                                   (lost-tag! :additive-read-arr (name arr)))))
                                  read-arrs)
            arr->d-sym (zipmap read-arrs d-read-arr-syms)
            d-scalar-syms (mapv (fn [p] (ad-gensym (str "d_" (name p) "_acc"))) active-free)
            n-sym (ad-gensym "n_red")
            reduce-result-sym (ad-gensym "red_val")
            bwd-result-sym (ad-gensym "red_bwd")
            n-agets (count agets)
            bwd-body
            (vec (concat
                  (mapcat (fn [{:keys [sym arr idx]}] [sym (list 'aget arr idx)]) agets)
                  step-bindings
                  (mapcat (fn [k {:keys [arr idx]}]
                            (let [d-arr (arr->d-sym arr)]
                              [(ad-gensym "_scatter")
                               (list 'raster.arrays/aset d-arr idx
                                     (list 'raster.ad.reverse/grad-acc
                                           (list 'aget d-arr idx)
                                           (nth step-grads k)))]))
                          (range) agets)))
            backward-loop
            (list 'let* [n-sym (list 'clojure.core/int bound-expr)
                         d-term-sym (if (= 1 sign) d-acc-sym (list 'raster.numeric/- d-acc-sym))]
                  (list 'loop* (vec (concat [idx-sym 0]
                                            (mapcat (fn [s p] [s (param-zero-expr p)])
                                                    d-scalar-syms active-free)))
                        (list 'if (list 'clojure.core/< idx-sym n-sym)
                              (list 'let* bwd-body
                                    (list* 'recur (list 'clojure.core/inc idx-sym)
                                           (map-indexed (fn [i d-s]
                                                          (list 'raster.ad.reverse/grad-acc d-s
                                                                (nth step-grads (clojure.core/+ n-agets i))))
                                                        d-scalar-syms)))
                              ;; [d_read_arrs… d_scalars… d_init]
                              (vec (concat d-read-arr-syms d-scalar-syms [d-acc-sym])))))]
        {:type :par-reduce
         :residual-kind :none
         ;; the backward evaluates each term again
         :replays (util/free-syms par-reduce-form)
         ;; source-order forward: the loop par/reduce stands for
         :forward-bindings
         [reduce-result-sym
          (let [n (ad-gensym "n_fwd")]
            (list 'let* [n (list 'clojure.core/int bound-expr)]
                  (list 'loop* [idx-sym 0 acc-sym init-expr]
                        (list 'if (list 'clojure.core/< idx-sym n)
                              (list 'recur (list 'clojure.core/inc idx-sym) body-expr)
                              acc-sym))))]
         :d-acc-sym d-acc-sym
         :written-arrs []
         :read-arrs read-arrs
         :d-read-arr-syms d-read-arr-syms
         :d-scalar-syms d-scalar-syms
         :active-free active-free
         :shadow-allocs (vec (mapcat (fn [d arr]
                                       [d (list 'raster.arrays/zeros-like arr
                                                (list 'clojure.core/alength arr))])
                                     d-read-arr-syms read-arrs))
         :backward-maps (vec (concat
                              [[bwd-result-sym backward-loop]]
                              (map-indexed (fn [i d] [d (list 'nth bwd-result-sym i)])
                                           d-read-arr-syms)))
         :backward-reduces (vec (map-indexed
                                 (fn [i d] [d (list 'nth bwd-result-sym (clojure.core/+ (count read-arrs) i))])
                                 d-scalar-syms))
         :bound-expr bound-expr
         :reduce-result-sym reduce-result-sym
         :bwd-result-sym bwd-result-sym
         :init-expr init-expr}))))

(defn- gen-reverse-par-reduce
  "Generate reverse-mode AD code for a raster.par/reduce form.

  Forward form: (raster.par/reduce acc init idx bound body-expr)

  An additive step (acc ± term, term independent of acc) needs no carry
  residual (gen-reverse-additive-reduce). Otherwise proven-pure double-carry
  steps use one primitive carry tape and replay the scalar step during the
  ordered reverse sweep, and other supported steps retain their per-step
  closure tape; no route reorders the forward accumulation.

  Returns a record map for gen-reverse-let to use."
  [par-reduce-form active-params]
  (or
   (gen-reverse-additive-reduce par-reduce-form active-params)
   (gen-reverse-ordered-reduce par-reduce-form active-params)))

(defn- gen-reverse-ordered-reduce
  "The general par/reduce reverse: the accumulator's cotangent chains through
  the steps (see gen-reverse-par-reduce)."
  [par-reduce-form active-params]
  (let [[_ acc-sym init-expr idx-sym bound-expr body-expr] par-reduce-form
        ;; as for a scan step: inline untemplated deftm calls first
        body-expr (if (untemplated-scan-call? body-expr)
                    (lower-composites body-expr)
                    body-expr)
        _ (when (and (seq? init-expr)
                     (some (set active-params)
                           (util/free-syms init-expr)))
            (throw (ex-info
                    "par/reduce AD: active compound init must be let-bound before the reduction"
                    {:reason :par-reduce-active-compound-init
                     :init init-expr})))
        {:keys [agets scalar-bindings body-result free-syms]}
        (analyze-par-map-body body-expr idx-sym true)
        typed-double? (and (every? (comp replay-pure-reduce-expr? second)
                                  scalar-bindings)
                           (replay-pure-reduce-expr? body-result)
                           (= :double (or (carry-scan-dtype acc-sym)
                                          (carry-scan-dtype init-expr)
                                          (some (fn [p]
                                                  (when (= p init-expr)
                                                    (carry-scan-dtype p)))
                                                active-params)))
                           (= 'double (or (:raster.type/tag (meta body-result))
                                          (inf/infer-rewritten-tag body-result)))
                           (carry-dtype-consistent? :double body-expr))

        ;; Read arrays (via aget in body)
        read-arrs (vec (distinct (map :arr agets)))

        ;; Active free symbols
        active-free (filterv (fn [p] (contains? free-syms p)) active-params)
        _ (doseq [p active-free]
            (when (= :array (:kind (tangent/tangent-kind
                                    (:raster.type/tag (meta p)))))
              (throw (ex-info
                      "par/reduce AD: active arrays must be read through a recognized aget"
                      {:reason :par-reduce-untracked-array-read
                       :array p :body body-expr}))))
        ;; the ordered backward is sequential: a read's cotangent scatter-adds at
        ;; its index, recomputed per step (index, literal or gather)
        locals (set (map first scalar-bindings))
        _ (doseq [{:keys [idx]} agets]
            (when-not (or (= idx-sym idx) (integer? idx)
                          (replayable-gather-index? idx idx-sym locals))
              (throw (ex-info
                      "par/reduce AD: an array read at an index the backward cannot recompute"
                      {:reason :par-reduce-shifted-read
                       :index idx :reduction-index idx-sym}))))

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
        d-acc-chain-sym (ad-gensym "d_acc_chain")

        ;; Per-iteration scalar transform
        step-bindings (if (seq pure-scalar-bindings)
                        (vec (concat pure-scalar-bindings [val-result-sym body-result]))
                        [val-result-sym body-result])
        scalar-program (when (and (seq iter-active) (not typed-double?))
                         (gen-reverse-read-region step-bindings [val-result-sym] iter-active
                                                  (index-replays agets idx-sym)))
        scalar-transform (when (seq iter-active)
                           (if typed-double?
                             (gen-inline-step-gradients step-bindings [val-result-sym]
                                                        iter-active d-acc-chain-sym)
                             (:form scalar-program)))
        step-replays (into (index-replays agets idx-sym)
                           (if typed-double? (util/free-syms par-reduce-form)
                               (:replays scalar-program)))

        ;; === Forward code ===
        ;; Run the reduce sequentially, storing pullbacks in tape
        fwd-loop-body
        (when scalar-transform
          (if typed-double?
            (util/subst-syms
             {idx-sym fwd-idx-sym}
             (list 'let* [val-result-sym body-expr
                          (ad-gensym "_store")
                          (list 'aset tape-sym fwd-idx-sym val-result-sym)]
                   val-result-sym))
            (let [aget-bindings (vec (mapcat (fn [{:keys [sym arr idx]}]
                                               [sym (list 'aget arr idx)])
                                             agets))
                  transform-bindings [vpb-sym scalar-transform
                                      val-result-sym (list 'nth vpb-sym 0)
                                      pb-sym (list 'nth vpb-sym 1)]
                  store-binding [(ad-gensym "_store") (list 'aset tape-sym fwd-idx-sym pb-sym)]
                  all-bindings (vec (concat aget-bindings transform-bindings store-binding))]
              ;; The forward loop binds the ALPHA-RENAMED index (fwd-idx-sym).
              (util/subst-syms {idx-sym fwd-idx-sym}
                               (list 'let* all-bindings val-result-sym)))))

        ;; Expose the tape and result as flat bindings. The forward and reverse
        ;; engines already consume flat SSA-like bindings; wrapping these in
        ;; an object array only hid the tape's ownership from later passes.
        ;; par/reduce evaluates (int bound) exactly once before its loop. A
        ;; negative bound runs zero steps, so the tape length clamps at zero.
        count-sym (ad-gensym "red_count")
        reduce-result-sym (ad-gensym "red_val")

        forward-bindings
        (when fwd-loop-body
          [count-sym (list 'clojure.core/int bound-expr)
           tape-sym (list (if typed-double? 'double-array 'object-array)
                          (list 'clojure.core/max 0 count-sym))
           reduce-result-sym
           (list 'loop* [fwd-idx-sym 0 acc-sym init-expr]
                 (list 'if (list 'clojure.core/< fwd-idx-sym count-sym)
                       (list 'recur (list 'clojure.core/+ fwd-idx-sym 1) fwd-loop-body)
                       acc-sym))])

        ;; === Backward code ===
        d-read-arr-syms (mapv (fn [arr] (ad-gensym (str "d_" (name arr)))) read-arrs)
        arr->d-sym (zipmap read-arrs d-read-arr-syms)
        d-scalar-syms (mapv (fn [p] (ad-gensym (str "d_" (name p) "_acc"))) active-free)
        d-acc-sym (ad-gensym "d_acc")
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
        bwd-body-bindings
        (vec (concat
              (if typed-double?
                (concat
                 [acc-sym (list 'if (list 'clojure.core/zero? j-sym)
                                init-expr
                                (list 'aget tape-sym (list 'clojure.core/dec j-sym)))]
                 (mapcat (fn [{:keys [sym arr idx]}]
                           [sym (util/subst-syms
                                 {idx-sym j-sym} (list 'aget arr idx))])
                         agets)
                 [grads-iter-sym
                  (util/subst-syms {idx-sym j-sym} scalar-transform)])
                [iter-pb-sym (list 'aget tape-sym j-sym)
                 grads-iter-sym (list iter-pb-sym d-acc-chain-sym)])
              ;; aset into shadow arrays for aget'd inputs
              ;; acc is iter-active[0], aget-syms are [1..n-agets]
              (mapcat (fn [i {:keys [arr idx]}]
                        (let [scatter-sym (ad-gensym "_scatter")
                              d-arr-sym (get arr->d-sym arr)
                              at (util/subst-syms {idx-sym j-sym} idx)]
                          [scatter-sym
                           (list 'raster.arrays/aset d-arr-sym at
                                 (list 'raster.ad.reverse/grad-acc
                                       (list 'aget d-arr-sym at)
                                       (list 'nth grads-iter-sym
                                             (clojure.core/+ 1 i))))]))
                      (range) agets)))

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

        bwd-return (vec (concat d-read-arr-syms d-scalar-syms
                                [d-acc-chain-sym]))

        backward-loop
        (list 'let* [n-bwd-sym (list 'clojure.core/alength tape-sym)]
              (list 'loop* bwd-loop-init
                    (list 'if (list 'clojure.core/>= j-sym 0)
                          bwd-loop-body
                          bwd-return)))

        ;; For par-reduce, the backward uses a sequential loop (due to acc chaining),
        ;; so we represent it as backward-maps (shadow arrays) from the loop result.
        ;; The backward-reduces accumulate the scalar gradients.
        bwd-result-sym (ad-gensym "red_bwd")]

    {:type :par-reduce
     :residual-kind (if typed-double? :double-carry :closure-tape)
     ;; Closures save the read values, but the scatter still reconstructs
     ;; destinations. Primitive carry replay additionally rereads the step.
     :replays step-replays
     :input-replays step-replays
     :effects (when scalar-program
                {:kind :region
                 :regions [(assoc (select-keys scalar-program [:writes :replays :value-facts])
                                  :replays step-replays)]})
     :forward-bindings forward-bindings
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
     :bwd-result-sym bwd-result-sym
     :init-expr init-expr}))

;; ================================================================
;; Par scan reverse-mode AD — the differentiable RECURRENCE primitive
;; ================================================================

(declare lower-composites)

(defn- untemplated-scan-call?
  [body]
  (boolean
   (some (fn [form]
           (and (seq? form) (= '.invk (first form))
                (nil? (tmpl/resolve-template (second form)))))
         (tree-seq coll? seq body))))

(defn ^:no-doc exact-scan-carry-checkpoint?
  "Whether retained scan storage and its resolved store callee preserve a Double carry.
   Spellings are not identity evidence: source walking retains the actual callee
   defining symbol before lexical scope is lost. Narrowing requires an unrounded residual tape."
  [out acc cast]
  (and (= 'double (:raster.type/tag (meta acc)))
       (or (= 'clojure.core/double (:raster.op/resolved-callee (meta cast)))
           (and (nil? cast) (= 'double (:raster.type/tag (meta acc)))))
       (contains? '#{doubles (Array double)} (:raster.type/tag (meta out)))))

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
  reconstruction reads the stored value. Only Double carry/storage with an
  identity store is admitted: rounding a checkpoint can change the point at
  which a nonlinear step's pullback is evaluated. Narrowing needs a separate
  unrounded carry tape, which this rule does not retain.

  Returns a record map for the reverse-pass engine (emit-backward :par-scan)."
  [par-scan-form active-params]
  (let [[_ out-sym acc-sym init-expr idx-sym bound-expr cast-fn body-expr] par-scan-form
        _ (when-not (exact-scan-carry-checkpoint? out-sym acc-sym cast-fn)
            (throw (ex-info "par/scan AD: carry reconstruction requires double storage and an identity cast"
                            {:reason :par-scan-carry-precision
                             :cast cast-fn
                             :carry-tag (:raster.type/tag (meta acc-sym))
                             :out-tag (:raster.type/tag (meta out-sym))
                             :form par-scan-form})))
        ;; The outer AD preparation runs before a carry loop becomes a scan.
        ;; Its body is therefore still a nested expression at that point, and
        ;; untemplated deftm helpers inside it have not been inlined. Prepare
        ;; the per-step scalar program before building its pullback as well.
        body-expr (if (untemplated-scan-call? body-expr)
                    (lower-composites body-expr)
                    body-expr)
        ;; Sequential scatter permits unconditional literal-index reads as well as
        ;; scan-index reads. Parallel map/reduce retain their narrower admission.
        ;; acc-sym lands in :free-syms
        ;; but is never an active param, so it cannot leak into active-free.
        {:keys [agets scalar-bindings body-result free-syms]}
        (analyze-par-map-body body-expr idx-sym true)

        read-arrs (vec (distinct (map :arr agets)))
        active-free (filterv (fn [p] (contains? free-syms p)) active-params)

        ;; Fail loud: an ACTIVE array referenced outside the recognized aget
        ;; reads would be mis-treated as a free SCALAR (mis-shaped gradient —
        ;; the par/map `No promotion rule` class of bug). Never miscompile.
        _ (doseq [p active-free]
            (when (= :array (:kind (tangent/tangent-kind (:raster.type/tag (meta p)))))
              (throw (ex-info
                      (str "par/scan AD: active array `" p "` has no closed scatter read. "
                           "Inline reads must use the scan index or an unconditional integer "
                           "literal; guarded or data-dependent reads need an indexed residual.")
                      {:reason :par-scan-untracked-array-read :array p :body body-expr}))))
        ;; Fail loud: a compound init that DEPENDS on active params would
        ;; reconstruct the value correctly but silently drop δinit (the final
        ;; carry only wires to a SYMBOL). Bind it outside the scan first.
        _ (when (and (seq? init-expr)
                     (some (set active-params) (util/free-syms init-expr)))
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

        ;; Per-step scalar transform (primal recompute + flat gradient program)
        ;; — evaluated in the BACKWARD loop at the reconstructed inputs. This
        ;; is recompute-from-carry: no forward-time tape or closure stores it.
        val-result-sym (ad-gensym "val")
        total-sym (ad-gensym "d_total")
        scalar-transform
        (let [binds (if (seq pure-scalar-bindings)
                      (vec (concat pure-scalar-bindings [val-result-sym body-result]))
                      [val-result-sym body-result])]
          (gen-inline-step-gradients binds [val-result-sym] iter-active total-sym))

        ;; === Backward code ===
        out-array-tag (or (some-> out-sym meta :raster.type/tag)
                          (lost-tag! :carry-out-arr (some-> out-sym name)))
        carry-tag (case out-array-tag doubles 'double floats 'float nil)
        d-out-sym (ad-gensym "d_out" out-array-tag)
        d-read-arr-syms (mapv (fn [arr]
                                (ad-gensym (str "d_" (name arr))
                                           (or (:raster.type/tag (meta arr))
                                               (lost-tag! :carry-read-arr (name arr)))))
                              read-arrs)
        arr->d-sym (zipmap read-arrs d-read-arr-syms)
        d-scalar-syms (mapv (fn [p] (ad-gensym (str "d_" (name p) "_acc"))) active-free)
        n-bwd-sym (ad-gensym "n_bwd")
        d-carry-sym (ad-gensym "d_carry")
        n-agets (count agets)
        ;; Splice the step's gradient program into the backward iteration so
        ;; its typed scalar adjoints reach the accumulators and scatters
        ;; directly, without a per-step gradient vector.
        [_ step-bindings step-grads] scalar-transform

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
                               (list 'aget d-out-sym idx-sym) d-carry-sym)]
              step-bindings
              ;; Scatter-ADD each read's cotangent at the read's own index
              ;; (read-modify-write: correct for repeated reads of one array
              ;; and for shifted indices like (aget x (- idx 1))).
              (mapcat (fn [k {:keys [arr idx]}]
                        (let [d-arr (arr->d-sym arr)]
                          [(ad-gensym "_scatter")
                           (list 'raster.arrays/aset d-arr idx
                                 (list 'raster.ad.reverse/grad-acc
                                       (list 'aget d-arr idx)
                                       (nth step-grads (clojure.core/+ 1 k))))]))
                      (range) agets)))

        bwd-recur-args
        (list* (list 'clojure.core/- idx-sym 1)
               ;; the carry cotangent chains: δacc_{i-1} = grads[∂body/∂acc]
               (nth step-grads 0)
               (map-indexed (fn [i d-s]
                              (list 'raster.ad.reverse/grad-acc d-s
                                    (nth step-grads (clojure.core/+ 1 n-agets i))))
                            d-scalar-syms))

        bwd-loop-init
        (vec (concat [idx-sym (list 'clojure.core/- n-bwd-sym 1)
                      d-carry-sym (if carry-tag (tangent/zero-expr carry-tag nil) 0.0)]
                     ;; Typed zeros keep scalar accumulators primitive.
                     (mapcat (fn [s p]
                               [s (if (= :scalar (:kind (tangent/tangent-kind
                                                         (:raster.type/tag (meta p)))))
                                    (param-zero-expr p)
                                    nil)])
                             d-scalar-syms active-free)))

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
     ;; the carry tape and every value the step reads are read again
     :replays (conj (util/free-syms par-scan-form) out-sym)
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

(declare ^:private lift-loop-to-tail
         ^:private retain-source-scan-checkpoint-obligations)

(defn transform-body
  "Transform a single walked body expression for reverse-mode AD.
  Returns an S-expression that evaluates to [primal, pullback-fn].

  active-params: ordered sequence of parameter symbols that need gradients.
  form: the walked S-expression body.

  The pullback-fn takes a scalar dy and returns a vector of
  adjoints [d_param1 d_param2 ...] in the same order as active-params."
  [form active-params]
  (with-ad-gensym
    (let [;; Generated derivative programs may reenter here without ad-prepare.
          ;; Use the same lexical value normalization as source preparation;
          ;; products are not new differentiable primitives or opaque calls.
          form (-> form retain-source-scan-checkpoint-obligations
                   local-products/lower-local-products)
          ;; Binding-position loops are routed by gen-reverse-let's existing
          ;; activity decision. Unconditionally tail-lifting here rewrites even
          ;; primal-only loops and bypasses that decision/checkpoint obligation.
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

(defn- deftm-params-or-throw
  "The resolved deftm's param vector, or a throw that tells the user HOW to fix
   the call. The most common miss (issue #56, hit on the README example) is
   passing the DEREF'D dispatch fn — `(value+grad rosenbrock …)` — instead of
   the var `#'rosenbrock`: the fn object carries none of the deftm metadata the
   AD transform reads, and the old bare \"grad requires a deftm var\" named
   neither the mistake nor the remedy."
  [f-var resolved]
  (or (:raster.core/deftm-params (meta resolved))
      (throw (ex-info
              (str "grad/value+grad requires a deftm VAR, got "
                   (cond
                     (var? f-var)
                     (str f-var " — a var, but not one defined by deftm/ftm.")
                     (ifn? f-var)
                     (str "a function object. Pass the var instead: "
                          "((value+grad #'my-fn) args…), not (value+grad my-fn) — "
                          "the deftm metadata the AD transform reads lives on the var.")
                     :else (pr-str f-var)))
              {:var f-var}))))

(def ^:private vjp-cache
  ;; A compiled VJP snapshots nested template rules and compiler-visible
  ;; definitions. Retain only the current generations;
  ;; a caller already holding a closure may continue to use that snapshot.
  (atom {:registry-revision nil :definition-revision nil :entries {}}))

(declare ad-prepare)

(defn- get-paired-helper-program
  "Compile an owned bytecode helper's paired forward/pullback program.

   The caller must first match this actual emitter snapshot to its selected
   definition. Historical static dependencies are fenced here, but this function
   does not turn reconstructed source metadata into emission authority. Returned
   facts retain their helper-local identities; the caller owns formal-to-actual
   memory projection and must compose its selected-definition guard with :guard.
   No companion is installed in dispatch or in a global Var."
  [{:keys [backend implementation params tags return-tag body source-ns typed-interface]
    :as emission} active-indices]
  (method-entry/assert-emission-inputs-current! emission)
  (when-not (and (= :jvm-bytecode backend) (some? implementation) (not (var? implementation))
                 (vector? params) (every? #(and (symbol? %) (nil? (namespace %))) params)
                 (= (count params) (count (distinct params)) (count tags))
                 (vector? tags)
                 (every? #(or (contains? types/primitive-info %)
                               (contains? types/primitive-array-info %)) tags)
                 (sequential? body) (seq body)
                 (instance? clojure.lang.Namespace source-ns)
                 (vector? active-indices) (= (count active-indices) (count (distinct active-indices)))
                 (every? #(and (integer? %) (<= 0 %) (< % (count params))) active-indices))
    (throw (ex-info "paired helper requires an actual primitive-carrier bytecode emission"
                    {:reason :ad-paired-helper-emission :active-indices active-indices})))
  ;; The primitive method return and these bytecode cast intrinsics share
  ;; emit-coerce. Retain that completion BEFORE AD, including f32 -> f64 and
  ;; f64 -> f32, rather than boxing the uncompleted body as the paired primal.
  ;; Other return descriptors need their own emitter-owned completion evidence.
  (when-not (contains? '#{float double} return-tag)
    (throw (ex-info "paired helper return completion is not represented"
                    {:reason :ad-paired-helper-return-completion :return-tag return-tag})))
  (doseq [index active-indices]
    (when-not (tangent/differentiable? (nth tags index))
      (throw (ex-info "paired helper active port has no retained tangent space"
                      {:reason :ad-paired-helper-active-space :index index :tag (nth tags index)}))))
  (let [interface-name (if (class? typed-interface)
                         (symbol (.getName ^Class typed-interface)) typed-interface)
        interface (get @types/fn-interface-by-name interface-name)
        ^Class interface-class (:iface-class interface)
        methods (when (class? interface-class)
                  (filter #(= "invk" (.getName ^java.lang.reflect.Method %))
                          (.getDeclaredMethods interface-class)))
        ^java.lang.reflect.Method invocation (when (= 1 (count methods)) (first methods))
        loader (.getClassLoader (class implementation))]
    (when-not (and (class? interface-class)
                   (or (not (class? typed-interface)) (identical? typed-interface interface-class))
                   (.isInstance interface-class implementation)
                   (= (vec tags) (:param-tags interface))
                   invocation (= Object (.getReturnType invocation))
                   (= (mapv types/tag->java-class tags) (vec (.getParameterTypes invocation)))
                   (instance? clojure.lang.DynamicClassLoader loader)
                   (identical? interface-class
                               (Class/forName (.getName interface-class) false loader)))
      (throw (ex-info "paired helper input interface does not match its actual implementation"
                      {:reason :ad-paired-helper-interface :typed-interface typed-interface})))
    (let [compile-pair (requiring-resolve 'raster.compiler.backend.jvm.bytecode/compile-typed-impl!)
          revisions (requiring-resolve 'raster.ad.normalized-scalar/revisions)
          capture (requiring-resolve 'raster.ad.normalized-scalar/execution-guard)
          expected (revisions)
          cache-key [::paired-helper implementation active-indices emission expected]
          cache @vjp-cache
          cached (when (and (= (:template-revision expected) (:registry-revision cache))
                            (= (:compiler-revision expected) (:definition-revision cache)))
                   (get-in cache [:entries cache-key]))]
      (if cached
        (do ((:guard cached)) cached)
        (let [all-params (mapv #(vary-meta %1 assoc :raster.type/tag %2) params tags)
              active-params (mapv all-params active-indices)
              actual-body (if (= 1 (count body)) (first body) (list* 'do body))
              completed-body (with-meta (list (symbol "clojure.core" (name return-tag)) actual-body)
                               {:raster.type/tag return-tag})
              entry-guard (capture source-ns expected [{:parameters all-params :body completed-body}])
              _ (entry-guard)
              proof-inputs (atom {:static [] :values []})
              program
              (binding [*ns* source-ns
                        method-entry/*emission-inputs* proof-inputs
                        *generated-product-context* nil
                        *constant-gradient-arrays* (set (keep-indexed
                                                        (fn [index parameter]
                                                          (when (and (contains? types/primitive-array-info (nth tags index))
                                                                     (not (contains? (set active-indices) index)))
                                                            parameter)) all-params))
                        ;; A paired companion is a compositional pullback,
                        ;; not the final public gradient boundary. An unused
                        ;; input must remain absent across this helper edge.
                        *source-memory-context* {:values all-params :internal-region? true
                                                 :value-facts (into {} (map (fn [p] [p {:value p}])) all-params)
                                                 :activity (zipmap active-params (repeat true))
                                                 :replayed {} :unprojectable {} :aliases {}}
                        util/*shadowing-locals* (set all-params)]
                (with-ad-gensym
                  (let [prepared (ad-prepare completed-body (zipmap all-params tags))
                        [bindings results] (extract-let-parts prepared)]
                    (if (and (= 1 (count results)) (loop-init? (first results)))
                      (gen-reverse-loop-with-let-program bindings (first results) active-params)
                      (gen-reverse-let-program bindings results active-params)))))
              program-guard (capture source-ns expected [{:parameters all-params :body (:form program)}])
              guard (fn []
                      (method-entry/assert-emission-inputs-current! emission)
                      (when-not (identical? source-ns (find-ns (ns-name source-ns)))
                        (throw (ex-info "paired helper source namespace changed"
                                        {:reason :ad-paired-helper-namespace})))
                      (when-not (identical? interface-class
                                            (:iface-class (get @types/fn-interface-by-name interface-name)))
                        (throw (ex-info "paired helper interface registration changed"
                                        {:reason :ad-paired-helper-interface})))
                      (entry-guard)
                      (method-entry/assert-definition-inputs-current! @proof-inputs)
                      (program-guard))
              _ (guard)
              companion (binding [*ns* source-ns
                                  method-entry/*compilation-classloader* loader]
                          (compile-pair (str "raster.compiled.PairedHelper" (gensym))
                                        params tags 'Object [(:form program)] source-ns interface-class))
              _ (when-not (.isInstance interface-class companion)
                  (throw (ex-info "paired helper compiler returned a different input interface"
                                  {:reason :ad-paired-helper-interface})))
              _ (guard)
              result {:companion companion :program program :guard guard
                      :emission emission :parameters all-params :active-indices active-indices
                      :active-params active-params :interface-class interface-class :method invocation
                      :dependency-snapshots (:dependency-snapshots emission)
                      :definition-revision (:definition-revision emission)
                      :return-tag return-tag
                      :writes (:writes program) :replays (:replays program)
                      :value-facts (:value-facts program) :aliases (:aliases program)}]
          ;; Share the existing cache generation; an older derivation must not
          ;; evict entries already published by a newer generation.
          (swap! vjp-cache
                 (fn [current]
                   (cond
                     (and (= (:template-revision expected) (:registry-revision current))
                          (= (:compiler-revision expected) (:definition-revision current)))
                     (assoc-in current [:entries cache-key] result)

                     (or (and (:registry-revision current)
                              (> (:registry-revision current) (:template-revision expected)))
                         (and (:definition-revision current)
                              (> (:definition-revision current) (:compiler-revision expected)))) current

                     :else {:registry-revision (:template-revision expected)
                            :definition-revision (:compiler-revision expected)
                            :entries {cache-key result}})))
          result)))))

(defn- capture-selected-helper-emission
  "Capture an emission through an already resolved global typed call.
   No operand tags select a method here. The original call-site producer must
   prove global lexical correspondence and supply its actual resolved Method.
   Realizing the selected lazy implementation never executes its numeric body.
   Capture is independent of derivative activity: an inactive call still needs
   the same source/receiver ownership and historical-input checks."
  [operation implementation-op argument-count resolved-method]
  (let [select! (fn []
                  (let [signature (call-boundary/selected-typed-signature
                                   operation implementation-op argument-count resolved-method)]
                    (when-not (= :known (:status signature))
                      (throw (ex-info "paired helper has no verified selected typed invocation"
                                      {:reason :ad-paired-helper-selected-definition
                                       :operation operation :implementation-op implementation-op
                                       :decline (:reason signature)})))
                    signature))
        initial (select!)
        _ (rcore/ensure-bytecode-implementation! (get-in initial [:definition :receiver-var]))
        {:keys [definition declaring-class] :as signature} (select!)
        {:keys [source-var receiver-var parameters tags return-tag entry]} definition
        emission (:raster.core/bytecode-emission (meta receiver-var))]
    (when-not (and (= :jvm-bytecode (:backend emission))
                   (identical? (:implementation emission) @receiver-var)
                   (identical? (:implementation emission) (:typed-impl entry))
                   (= (:params emission) parameters) (= (:tags emission) tags)
                   (= (:return-tag emission) return-tag)
                   (identical? (:source-ns emission) (.ns ^clojure.lang.Var source-var))
                   (= (:typed-interface emission) (:typed-iface entry)))
      (throw (ex-info "selected helper does not own its retained bytecode emission"
                      {:reason :ad-paired-helper-emission-correspondence
                       :operation operation :implementation-op implementation-op})))
    (let [observations (mapv method-entry/var-snapshot [source-var receiver-var])
          selection-guard
          (fn []
            (let [current (select!)]
              (when-not (and (every? method-entry/var-snapshot-current? observations)
                             (identical? declaring-class (:declaring-class current))
                             (identical? source-var (get-in current [:definition :source-var]))
                             (identical? receiver-var (get-in current [:definition :receiver-var]))
                             (identical? (:implementation emission)
                                         (get-in current [:definition :entry :typed-impl])))
                (throw (ex-info "selected paired helper definition changed"
                                {:reason :ad-paired-helper-selected-definition
                                 :operation operation :implementation-op implementation-op})))))
          guard (fn [] (selection-guard)
                  (method-entry/assert-emission-inputs-current! emission)
                  nil)
          _ (guard)]
      {:emission emission :guard guard :signature signature :method resolved-method})))

(defn- get-selected-paired-helper-program
  "Derive the paired program from the actual selected emission. Runtime owners
   apply its guard to forward calls and delayed pullbacks; selection alone is
   not authority to change the original lexical call's evaluation order."
  [operation implementation-op argument-count resolved-method active-indices]
  (let [{:keys [emission guard signature]}
        (capture-selected-helper-emission operation implementation-op argument-count resolved-method)
        artifact (get-paired-helper-program emission active-indices)
        paired-guard (fn [] (guard) ((:guard artifact)))]
    (paired-guard)
    (assoc artifact :guard paired-guard :signature signature)))

(defn- paired-helper-memory-region
  "Project a paired scalar helper's retained replay roots onto caller values.
   Primitive inputs are copied values, not borrowed storage. Array roots must
   correspond to actual visible arguments, never private helper identifiers or
   guessed freshness. The ordinary caller engine owns subsequent alias/write
   checks. Mutable or private-storage exports require a richer owned interface."
  [{:keys [parameters emission writes replays value-facts]} arguments visible-values]
  (when-not (and (vector? parameters) (vector? arguments)
                 (= (count parameters) (count arguments))
                 (set? writes) (set? replays) (map? value-facts)
                 (set? visible-values))
    (throw (ex-info "paired helper memory projection requires its actual formal/argument interface"
                    {:reason :ad-paired-helper-memory-interface})))
  (when-not (and (empty? writes)
                 (every? #(= :pure (effects/analyze-effect %)) (:body emission)))
    (throw (ex-info "paired helper effects require an explicit owned memory export interface"
                    {:reason :ad-paired-helper-effects :writes writes})))
  (let [formals (zipmap parameters parameters)
        actuals (zipmap parameters arguments)
        projected
        (into #{}
              (keep (fn [replay]
                      (let [root (exact-replay-root replay value-facts)
                            formal (get formals root)
                            tag (types/sym-type-tag (or formal root))]
                        (when-not (contains? types/primitive-info tag)
                          (let [actual (get actuals root)]
                            (when-not (and formal (contains? types/primitive-array-info tag)
                                           (symbol? actual) (contains? visible-values actual))
                              (throw (ex-info "paired helper has a replay root outside its caller-owned interface"
                                              {:reason :ad-unprojectable-helper-replay
                                               :replay replay :root root :argument actual})))
                            actual))))) replays)]
    {:kind :region :regions [{:writes #{} :replays projected :value-facts {}}]}))

(defn- get-vjp-fn
  "Get or compile the AD-transformed function for a deftm var.
  Cached by [template/definition revisions, var-identity, walked-body] to avoid
  recompiling on every call. A registry mutation conservatively starts a new
  cache generation because transforms can depend on nested template rules or
  inlined deftm definitions, even when the parent's walked body is unchanged.

  Filters active params by tag (same rule as build-grad-walked-body): only
  doubles/floats are seeded as active. The pullback's output vector still has
  one slot per ORIGINAL param (with nil for non-differentiable slots) so
  positional consumers don't shift."
  [resolved]
  (loop []
    (let [registry-revision (tmpl/registry-revision)
          definition-revision (dispatch/compiler-definition-revision)
          m (meta resolved)
          params (or (:raster.core/deftm-params m)
                     (throw (ex-info "vjp requires a deftm var"
                                     {:var resolved})))
          walked-body (or (rcore/ensure-walked-body! resolved)
                          (throw (ex-info "No walked body on var" {:var resolved})))
          ;; Parameter tags are part of the admission proof. They can change while the
          ;; retained body remains identical (for example during deftm redefinition), so
          ;; an old classification must never be reused across that metadata boundary.
          cache-key [resolved walked-body params (:raster.core/deftm-tags m)]
          cache @vjp-cache
          cached (when (and (= registry-revision (:registry-revision cache))
                            (= definition-revision (:definition-revision cache)))
                   (get-in cache [:entries cache-key]))]
      (cond
        ;; Walking or metadata acquisition can itself span a redefinition.
        ;; Do not return an old cache hit after observing that boundary drift.
        (or (not= registry-revision (tmpl/registry-revision))
            (not= definition-revision (dispatch/compiler-definition-revision)))
        (recur)

        cached cached

        :else
        (let [tags (or (:raster.core/deftm-tags m)
                       (vec (repeat (count params) nil)))
              all-params (vec (map-indexed
                               (fn [index parameter]
                                 (let [base (if (symbol? parameter)
                                              parameter (symbol (name parameter)))
                                       tag (nth tags index nil)]
                                   (cond-> base tag (vary-meta assoc :raster.type/tag tag))))
                               params))
              parameter-plan (tangent-parameter-plan all-params tags nil)
              diff-active-params (:candidate-params parameter-plan)
              transformed (binding [*source-memory-context* {:values all-params
                                                            :value-facts {} :unprojectable {}
                                                            :replayed {} :aliases {}}
                                    util/*shadowing-locals* (into util/*shadowing-locals* all-params)]
                            (with-ad-gensym
                              (transform-body
                               (ad-prepare (first walked-body)
                                           (zipmap all-params tags))
                               diff-active-params)))
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
                compiled-fn (eval (value-completion/project-to-jvm-source fn-form))]
            ;; A mutation during transformation may otherwise produce a closure
            ;; assembled from more than one registry snapshot. Do not publish or
            ;; return it: retry against the new monotonic revision instead.
            (if (or (not= registry-revision (tmpl/registry-revision))
                    (not= definition-revision (dispatch/compiler-definition-revision)))
              (recur)
              (do
                (swap! vjp-cache
                       (fn [cache]
                         (let [cached-revision (:registry-revision cache)
                               cached-definition (:definition-revision cache)]
                           (cond
                             (and (= registry-revision cached-revision)
                                  (= definition-revision cached-definition))
                             (assoc-in cache [:entries cache-key] compiled-fn)

                             ;; A compiler from an older generation must not
                             ;; evict entries already published by a newer one.
                             (or (and cached-revision
                                      (> cached-revision registry-revision))
                                 (and cached-definition
                                      (> cached-definition definition-revision)))
                             cache

                             :else
                             {:registry-revision registry-revision
                              :definition-revision definition-revision
                              :entries {cache-key compiled-fn}}))))
                compiled-fn)))))))

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
              (inline/lower-to-ad-primitives f))
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

(defn- lower-recurrence-bodies
  "Inline constructor-bearing code inside a structured scan or reduction,
  including its initial value in the enclosing scope. Rebuild through the
  shared scope descriptor so no binding escapes it."
  [body]
  (letfn [(bind-tail [expr]
            ;; The composite inliner visits binding initializers, not a let's
            ;; bare result expression. Keep the result inside this lexical
            ;; region while giving nested calls an ANF binding to inline into.
            (if (and (form/binding-form? expr) (= 3 (count expr))
                     (not (symbol? (last expr))))
              (let [[head bindings result] expr
                    result-sym (ad-gensym "recurrence_result")]
                (with-meta
                  (list head (into (vec bindings) [result-sym result]) result-sym)
                  (meta expr)))
              expr))
          (step [expr]
            (cond
              (and (seq? expr) (#{'quote 'clojure.core/quote} (first expr))) expr
              (and (seq? expr)
                   (or (par-ir/par-reduce-form? expr)
                       (par-ir/par-scan-form? expr)))
              (let [{:keys [scoped-syms inner-exprs outer-exprs rebuild]}
                    (par-ir/par-scope-info expr)]
                (rebuild scoped-syms
                         (mapv (fn [inner]
                                 (let [inner (step inner)]
                                   (if (soa-lower/contains-local-constructor? inner)
                                     (lower-composites (bind-tail inner))
                                     inner)))
                               inner-exprs)
                         (mapv (fn [outer]
                                 (let [outer (step outer)]
                                   (if (soa-lower/contains-local-constructor? outer)
                                     (lower-composites outer)
                                     outer)))
                               outer-exprs)))
              ;; Loop canonicalization can introduce a let-bound scan initializer
              ;; outside the scan itself. Inline calls that *consume* constructors
              ;; in their own lexical binding; leave direct constructor bindings
              ;; for the field projector, without moving an evaluation across
              ;; the recurrence boundary.
              (and (seq? expr) (form/binding-form? expr))
              (let [[head bindings & body] expr]
                (with-meta
                  (apply list head
                         (into [] (mapcat (fn [[s init]]
                                            (let [init (step init)]
                                              [s (if (and (seq? init)
                                                          (some soa-lower/contains-local-constructor?
                                                                (rest init)))
                                                   (lower-composites init)
                                                   init)])))
                               (partition 2 bindings))
                         (map step body))
                  (meta expr)))
              (seq? expr) (with-meta (apply list (map step expr)) (meta expr))
              (vector? expr) (mapv step expr)
              :else expr))]
    (step body)))

(def ^:private variadic-core-arithmetic
  '#{clojure.core/+ clojure.core/- clojure.core/* clojure.core//
     + - * /})

(defn- postwalk-code
  "Postwalk executable forms, leaving quoted data untouched."
  [f body]
  (letfn [(step [form]
            (if (and (seq? form)
                     (contains? '#{quote clojure.core/quote} (first form)))
              form
              (f (walk/walk step identity form))))]
    (step body)))

(defn- binaryize-core-arithmetic
  "Express core arithmetic's variadic contract using identities, aliases,
  unary calls, and ordered binary folds. This matches the fixed-arity AD
  templates without changing evaluation order or inventing per-arity rules."
  [body]
  (postwalk-code
   (fn [form]
     (if (and (seq? form) (not= '.invk (first form))
              (contains? variadic-core-arithmetic (op/semantic-op form)))
       (let [op (first form)
             args (rest form)
             n (count args)
             name (name op)
             normalized (cond
                          (and (= name "+") (zero? n)) 0
                          (and (= name "*") (zero? n)) 1
                          (and (#{"+" "*"} name) (= n 1)) (first args)
                          (> n 2) (reduce (fn [acc arg] (list op acc arg))
                                          (first args) (rest args))
                          :else form)]
         (if (and (instance? clojure.lang.IObj normalized) (meta form))
           (with-meta normalized (meta form))
           normalized))
       form))
   body))

(defn- inline-qualified-constants
  "Lower qualified ^:const primitive vars in inlined callees. The source
  namespace may keep such vars private, while the generated AD function is
  evaluated in the caller's namespace. Only immutable scalar constants are
  eligible; ordinary Vars remain runtime references."
  [body]
  (postwalk-code
   (fn [form]
     (if (and (symbol? form) (namespace form))
       (if-let [v (when (find-ns (symbol (namespace form)))
                    (find-var form))]
         (if (:const (meta v))
           (let [value @v]
             (if (or (number? value) (string? value) (boolean? value))
               value
               form))
           form)
         form)
       form))
   body))

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
    ;; a scoped step's value is its body tail
    (and (seq? expr) (contains? #{'let 'let*} (first expr)) (< 2 (count expr)))
    (recur (last expr))
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
          ;; Only the synthetic tape's proven Double carry has an identity store.
          ;; No callable conversion may be rebound by the source lexical/namespace scope.
          acc (if (and (nil? out) (= :double dtype))
                (vary-meta acc assoc :raster.type/tag 'double) acc)
          cast (if (and (nil? out) (= :double dtype)) nil cast)
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

(defn- active-reads-match-dtype?
  "Every active array read anywhere in `expr` (including let* binding inits,
  which read-array-elem-types does not descend into) has element dtype
  `dtype`; its cotangent scatters into a shadow of that array. Unstamped
  arrays contribute no evidence, as in read-array-elem-types."
  [dtype expr]
  (every? (fn [read]
            (case (:raster.type/tag (meta (first (op/call-args read))))
              floats (= dtype :float)
              doubles (= dtype :double)
              (nil Object) true
              false))
          (filter gradient-bearing-array-read? (tree-seq coll? seq expr))))

(defn- scan-prelude-reads-replayable?
  "True when every active array read in a carry-loop let prelude is one the
  par/scan pullback can replay and scatter into (scan-read-lifter)."
  [bindings index-sym]
  (let [lift (scan-read-lifter index-sym true (fn [_ _] '_read))]
    (:ok? (reduce (fn [{:keys [locals]} [sym init]]
                    (let [ok? (if (gradient-bearing-array-read? init)
                                (= '_read (lift init false locals))
                                (not-any? gradient-bearing-array-read?
                                          (tree-seq coll? seq (lift init false locals))))]
                      (if ok?
                        {:ok? true :locals (conj locals sym)}
                        (reduced {:ok? false}))))
                  {:ok? true :locals #{}}
                  bindings))))

(defn- emit-carry-recurrence
  "A pure carry recurrence (no stored out buffer) as par/reduce when its step
  is additive — the reduction's pullback needs no carry tape — else as the
  synthesized-out scan. `body` may be a let* prelude ending in the update."
  [{:keys [acc idx bound init body dtype] :as spec}]
  (let [tail (loop [e body] (if (and (seq? e) (contains? #{'let 'let*} (first e))) (recur (last e)) e))
        prelude-reads-acc? (and (seq? body) (contains? #{'let 'let*} (first body))
                                (some (fn [[_ init]] (contains? (util/free-syms init) acc))
                                      (partition 2 (second body))))]
    (if (and (additive-step tail acc) (not prelude-reads-acc?))
      (with-ad-gensym
        (let [init-sym (ad-gensym "carry_init" (case dtype :float 'float :double 'double nil))]
          (list 'let* [init-sym init]
                (list 'raster.par/reduce acc init-sym idx bound body))))
      (emit-carry-scan spec))))

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
         (emit-carry-recurrence {:out nil
                           :dtype dtype
                           :cast (case dtype :float 'float :double nil)
                           :acc acc-sym :idx index-sym
                           :bound bound-expr :init acc-init
                           :body body}))))
   ;; A pure let prelude whose tail is the sole recur remains one ordered
   ;; recurrence step. Retain its lexical bindings inside the scan body;
   ;; reverse AD recomputes that body, so effectful initializers must decline.
   (when-let [{:keys [acc-sym acc-init index-sym index-init bound-expr bound-mode
                      then-branch else-expr scoped-update-expr]}
              (patterns/match-ordered-reduce-loop loop-form)]
     (let [bindings (when (and (form/binding-form? then-branch)
                               (= 3 (count then-branch))
                               (seq? (last then-branch))
                               (= 'recur (first (last then-branch))))
                      (partition 2 (second then-branch)))
           dtype (or (carry-scan-dtype acc-init)
                     (carry-scan-dtype scoped-update-expr))]
       (when (and (seq bindings)
                  (= 0 index-init)
                  (= :exclusive bound-mode)
                  (patterns/acc-ref? else-expr acc-sym)
                  (not (patterns/contains-sym? bound-expr index-sym))
                  (not (patterns/contains-sym? bound-expr acc-sym))
                  (every? (fn [[_ init]] (= :pure (effects/analyze-effect init)))
                          bindings)
                  ;; The scan pullback replays the prelude's active array reads
                  ;; and scatter-adds their cotangents. Reads it cannot replay
                  ;; keep the loop on its existing AD path.
                  (scan-prelude-reads-replayable? bindings index-sym)
                  (some? dtype)
                  (carry-dtype-consistent? dtype scoped-update-expr)
                  (active-reads-match-dtype? dtype scoped-update-expr))
         (emit-carry-recurrence {:out nil
                           :dtype dtype
                           :cast (case dtype :float 'float :double nil)
                           :acc acc-sym :idx index-sym
                           :bound bound-expr :init acc-init
                           :body scoped-update-expr}))))))

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

(defn- retain-source-scan-checkpoint-obligations
  "Check a recognized stored-carry loop before beta/ANF can hide its store
   behind argument bindings. An inexact output is not an unrounded carry tape;
   losing the scan pattern must not silently admit the generic loop fallback.
   Denial is monotone across preparation: an existing key, including malformed
   or legacy payloads, can only retain a refusal obligation, never authorize AD.
   Independently rediscover source unsafety and canonicalize the union to pure
   true. The existing activity decision enforces it; inactive regions remain
   ordinary primal code. This metadata is not authenticated provenance."
  [body]
  (postwalk-code
   (fn [expression]
     (if (seq? expression)
       (let [prior-denial? (contains? (meta expression) ::inexact-source-scan-checkpoint)
             clean (vary-meta expression dissoc ::inexact-source-scan-checkpoint)
             source-denial?
             (when-let [{:keys [out-sym acc-sym cast-fn else-expr]}
                        (patterns/match-scan-loop clean)]
               (and (or (= out-sym else-expr)
                        (patterns/acc-ref? else-expr acc-sym))
                    (not (exact-scan-carry-checkpoint? out-sym acc-sym cast-fn))))]
         (cond-> clean
           (or prior-denial? source-denial?)
           (vary-meta assoc ::inexact-source-scan-checkpoint true)))
       expression))
   body))

(defn ^:no-doc ad-prepare
  "The ONE shared pre-AD preparation, used by the reverse path
  (`build-grad-walked-body`, `grad-expr`) AND the forward/JVP path
  (`raster.ad.jvp/jvp`):

    project constructor-bearing source scan/reduce scopes →
    anf-wrap-bare-body → lower-composites (inline un-templated composites to
    a fixpoint) → project nonescaping value constructors →
    lift-parallel-forms (loop → SOAC canonicalization, §15.2)
    → canonicalize-carry-loops (AD-only carry → scan materializer) →
    inline constructor-bearing scan/reduce bodies and initializers → project their fields →
    materialize-pass (pure par/map → alloc + par/map!) →
    hoist-nested-lets (flat ANF).

  The loop canonicalization runs AFTER lower-composites (inlining exposes
  callee loops) and BEFORE materialize/hoist. It is a PHASE-ORDERING fix:
  the pipeline's :loop-lift pass runs too late (post-fixpoint) for AD and
  never on the runtime path, so ad-prepare runs the same pure form→form fn
  here. Loops the gates decline are left untouched for the existing paths."
  ([body] (ad-prepare body nil))
  ([body param-env]
   (let [body (retain-source-scan-checkpoint-obligations body)
         body (local-products/lower-local-products body)
         body (inline/inline-transparent-lexical-calls body param-env)
         ;; A source-written reduction must project its initializer and step
        ;; before broad ANF is allowed to move either across the binder.
        source-scoped (binding [inline/*param-env* param-env]
                        (-> body lower-recurrence-bodies
                            binaryize-core-arithmetic inline-qualified-constants
                            soa-lower/lower-local-constructors))
        ;; ANF-normalize + inline composite deftm calls to a fixpoint so BARE
        ;; and multi-level composites fully lower before AD (lower-composites
        ;; runs anf-wrap-bare-body itself).
        lowered (binding [inline/*param-env* param-env]
                  (-> source-scoped binaryize-core-arithmetic lower-composites
                      binaryize-core-arithmetic inline-qualified-constants))
        lowered (soa-lower/lower-local-constructors lowered)
        ;; §15.2 (1): recover SOAC structure from raw loop syntax — dotimes →
        ;; par/map!, tail-accumulation → par/reduce, carry+aset → par/scan —
        ;; with the lift's proven-soundness gates (stats are discarded here;
        ;; matched loops simply become differentiable SOACs).
        lifted (:form (loop-lift/lift-parallel-forms lowered))
        ;; §15.2 (2): materialize surviving pure-carry recurrences as par/scan
        ;; (the synthesized out IS the tape — store-the-carry checkpointing).
        canonical (canonicalize-carry-loops lifted)
        scoped (binding [inline/*param-env* param-env]
                 (-> canonical lower-recurrence-bodies
                     binaryize-core-arithmetic inline-qualified-constants
                     soa-lower/lower-local-constructors))
        ;; Materialize pure par/map (broadcast → par/pmap) into alloc + par/map!
        ;; before AD — the pure pmap has no reverse rule; the mutating map! does.
        materialized (:form (materialize/materialize-pass scoped nil))]
    ;; Hoist nested lets out of call args into flat ANF for AD.
    (let [flat (hoist-nested-lets materialized)
          tag-facts
          (reduce-kv
           (fn [facts id tag]
             (cond
               (dtype/dtype-for-array-tag tag)
               (assoc-in facts [:array-types id] (dtype/dtype-for-array-tag tag))
               (dtype/dtype-for-scalar-tag tag)
               (assoc-in facts [:scalar-types id] (dtype/dtype-for-scalar-tag tag))
               :else facts))
           {:array-types {} :scalar-types {}} (or param-env {}))]
      (:form ((requiring-resolve
               'raster.compiler.passes.parallel.typed-soac-frontend/canonicalize-fresh-full-write-maps)
              flat tag-facts))))))

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
    (transform-body
     (ad-prepare form (into {} (keep (fn [p]
                                      (when-let [tag (:raster.type/tag (meta p))]
                                        [p tag])))
                            active-params))
     (vec active-params))))

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
          dy-sym 'dy__rad
          {:keys [fwd-bindings] :as pieces}
          (let-gradient-pieces bindings body-exprs active-params dy-sym)
          {:keys [pullback-form] :as forms} (reverse-program-forms pieces)

          ;; === Phase 4: Reify pullback ===
          ;; Determine captured variables:
          ;; Free symbols in the reverse bindings that are NOT:
          ;; - defined within the reverse bindings themselves
          ;; - the dy symbol
          ;; - the active params (these flow separately)
          fwd-bound (collect-bound-syms (vec fwd-bindings))
          rev-free  (binding [util/*shadowing-locals* (into util/*shadowing-locals* fwd-bound)]
                      (util/free-syms pullback-form #{dy-sym}))
          ;; Captured = free in reverse bindings ∩ bound in forward bindings
          captured  (clojure.set/intersection rev-free fwd-bound)
          ;; Maintain deterministic order: use forward binding order
          captured-ordered (filterv captured (take-nth 2 fwd-bindings))

          ;; Pullback params: [dy, captured1, captured2, ...]
          pullback-params (vec (cons dy-sym captured-ordered))]

      (merge forms
             {:pullback-params pullback-params
              :active-params   (vec active-params)
              :captured-syms   captured-ordered
              :fwd-bindings    (vec fwd-bindings)
              :body-sym        (:body-sym pieces)}))))

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
(declare assert-scalar-loss-result-admissible!)

(defonce ^:private ror-rename-counter (atom 0))

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
        tags (or (:raster.core/deftm-tags m) (vec (repeat (count params) nil)))
        _ (assert-scalar-loss-result-admissible!
           f-var resolved walked-body params tags)
        all-params (vec (map-indexed
                         (fn [i p]
                           (let [base (if (symbol? p) p (symbol (name p)))
                                 tag (nth tags i nil)]
                             (if tag
                               (with-meta base {:raster.type/tag tag})
                               (with-meta base nil))))
                         params))
        param-sym (symbol (name param))
        source-slot (or (first (keep-indexed
                                (fn [i p] (when (= p param-sym) i)) all-params))
                        (throw (ex-info (str "reified-grad: unknown parameter `" param
                                             "` of " f-var)
                                        {:param param :params all-params})))
        parameter-plan (tangent-parameter-plan all-params tags [source-slot])
        diff-params (:candidate-params parameter-plan)
        slot-idx 0
        param-tag (:raster.type/tag (meta (nth diff-params slot-idx)))
        array-param? (= :array (:kind (tangent/tangent-kind param-tag)))
        ;; The gradient program, assembled exactly as hvp does. ad-prepare
        ;; (lower-composites) and reify-pullback each open their OWN
        ;; with-ad-gensym; run BOTH under one shared counter so reify-pullback
        ;; continues from ad-prepare's high-water mark instead of resetting to 0
        ;; and re-minting colliding anf__ temps. (The old local rename ALSO required its input
        ;; bindings to be unique, since it collapsed duplicates; util/alpha-convert does not, so
        ;; this shared counter is now belt-and-braces rather than load-bearing for that reason.)
        {:keys [fwd-bindings result-sym body-sym pullback-form]}
        (call-with-shared-ad-gensym
         (fn [] (reify-pullback
                 (ad-prepare (first walked-body) (zipmap all-params tags))
                 diff-params)))
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
        ;; α-rename so the frozen program survives a fresh-countered second AD pass: a reified
        ;; gradient FREEZES symbols minted under one `with-ad-gensym` scope, and a later pass starts a
        ;; fresh counter that re-mints the same `anf__` names, silently rebinding inner syms.
        ;;
        ;; util/alpha-convert, not a local rename. The previous local version built its map with
        ;; `(into {} …)` over the bound names, so a name bound TWICE collapsed to one entry and the
        ;; first binding's uses read the second's value — a silent wrong gradient. Its docstring and
        ;; this call site both stated "assumes the bindings are already unique"; nothing asserted it.
        ;; alpha-convert has no such precondition (util_test pins the separation), renames nested
        ;; binders too (consistent renaming is semantics-preserving), and carries binder metadata onto
        ;; every reference — which is what preserved :raster.type/tag before.
        form (util/alpha-convert (list 'let* bindings tail))
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
                   (vec (repeat (count params) nil)))
          _ (assert-scalar-loss-result-admissible!
             f-var resolved walked-body params tags)
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
          parameter-plan (tangent-parameter-plan all-params tags nil)
          ;; Proven ⊥ slots (Long dimensions, indices, controls) have no v/Hv row.
          active-params (:candidate-params parameter-plan)
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
          ;; A generated pullback is another compiler input, not necessarily
          ;; flat primitive ANF. Run the shared preparation before re-entering
          ;; AD, including scoped nil-preserving cotangent projections.
          hvp-transformed
          (transform-body
           (ad-prepare combined-form
                       (into {} (map (fn [parameter]
                                       [parameter (types/sym-type-tag parameter)]))
                             (concat all-params v-syms)))
           active-params)

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
    (let [{op-sym :recorded-op args :arguments} (op/call-description form)]
      (if op-sym
        (apply list op-sym (map undevirtualize args))
        ;; No :op metadata — keep as-is (e.g., fn-param .invk calls)
        (apply list (map undevirtualize form))))

    (seq? form)
    (with-meta (apply list (map undevirtualize form)) (meta form))

    (vector? form)
    (mapv undevirtualize form)

    :else form))

;; alpha-conversion (hygienic rename of every bound var before splicing a form
;; alongside structurally identical siblings — unrolled fold iterations) is the
;; unified `util/alpha-convert`, generic over form/scope-info. Unlike the former
;; private copy (let*/loop*/dotimes/fn* only), it ALSO freshens binders nested in
;; par/ftm/letfn*/reify*/catch bodies — closing the sibling-conflation hole the
;; old version silently passed through.

(defn- hoisted-let-value
  "Project a hygienic let into ordered ANF bindings and its final value.
   Every body statement remains evaluated, including unused checked reads."
  [expression]
  (let [[_ bindings & body] (util/alpha-convert expression)
        [extras value] (anf/anf-normalize-expr (cons 'do body) ad-gensym)
        trivial? (anf/trivial-expr? value)
        result (if trivial? value (with-meta (ad-gensym "argument") (meta value)))]
    {:bindings (cond-> (into (vec bindings) extras)
                 (not trivial?) (into [result value]))
     :value result}))

(defn- hoist-nested-lets
  "Hoist let expressions out of call arguments into a wrapping let*.
  Transforms: (f (let [a e1] a) (let [b e2] b)) → (let* [a e1 b e2] (f a b))
  Recursively processes sub-expressions. Required for AD which expects flat ANF."
  [form]
  (cond
    (not (seq? form)) form

    ;; Data and owned input-completion regions are opaque to this hoister,
    ;; just as they are to the shared ANF normalizer. Entering quoted data
    ;; executes it; lifting a completion's operand changes its original JVM
    ;; evaluation context (primitive versus boxed conversion, for example).
    (or (contains? '#{quote clojure.core/quote} (first form))
        (value-completion/canonical-parts form)) form

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
                        (let [{:keys [bindings value]} (hoisted-let-value he)]
                          (-> acc (into bindings) (conj sym value)))
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

    ;; A reduction initializer is evaluated in the enclosing scope, after
    ;; its bound and before the first step. AD needs an active compound init
    ;; as an ordinary let binding so its pullback flows into the initial
    ;; accumulator. The reduction body remains inside its own acc/index scope.
    (par-ir/par-reduce-form? form)
    (let [[head acc init idx bound body] form
          init' (hoist-nested-lets init)
          bound' (hoist-nested-lets bound)
          body' (hoist-nested-lets body)]
      (if (seq? init')
        (let [bound-sym (with-meta (gensym "reduce_bound__") (meta bound'))
              init-sym (with-meta (gensym "reduce_init__") (meta init'))
              reduction (with-meta (list head acc init-sym idx bound-sym body') (meta form))]
          (hoist-nested-lets
           (list 'let* [bound-sym bound' init-sym init'] reduction)))
        (with-meta (list head acc init' idx bound' body') (meta form))))

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
          has-let? (some #(and (seq? %) (contains? #{'let 'let*} (first %))) hoisted-args)
          ;; Once one argument's bindings move before the call, all earlier arguments
          ;; must be evaluated there too. Otherwise a later let initializer can run
          ;; before an earlier checked read or effectful argument.
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
                         (let [projection (hoisted-let-value arg)]
                           {:bindings (into bindings (:bindings projection))
                            :clean-args (conj clean-args (:value projection))})
                         (if has-let?
                           (let [[extras normalized] (anf/anf-normalize-expr arg ad-gensym)
                                 value (if (anf/trivial-expr? normalized) normalized
                                           (with-meta (ad-gensym "argument") (meta arg)))]
                             {:bindings (cond-> (into bindings extras)
                                          (not (anf/trivial-expr? normalized))
                                          (into [value normalized]))
                              :clean-args (conj clean-args value)})
                           {:bindings bindings :clean-args (conj clean-args arg)})))
                     {:bindings [] :clean-args []}
                     hoisted-args)]
      (if (seq (:bindings collected))
        (list 'let* (vec (:bindings collected))
              (with-meta (apply list f (:clean-args collected)) (meta form)))
        (with-meta (apply list f hoisted-args) (meta form))))

    :else form))

(defn- lift-loop-to-tail
  "Restructure (let* [pre... loop-sym (loop ...) post...] body) so the loop
  becomes the let's tail expression — the canonical shape gen-reverse-loop-with-let
  expects. Inlines the post-bindings + body into the loop's result branch.

  When called on something that isn't a let* with a loop in binding position,
  returns the form unchanged. The caller's existing activity analysis selects
  target-loop-sym; earlier inactive loops stay in the primal prefix. Recurse
  from the caller if multiple active loops in series need lifting.

  Loop body shape required: (if test (recur ...) result-expr) or
                            (if test result-expr (recur ...))."
  [form target-loop-sym]
  (if-not (and (seq? form) (#{'let 'let*} (first form)))
    form
    (let [[let-sym bindings & body-exprs] form
          pairs (partition 2 bindings)
          loop-binding? (fn [[sym init]]
                          (and (= target-loop-sym sym)
                               (seq? init) (#{'loop 'loop*} (first init))))
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
                      new-loop-form (with-meta (apply list 'loop* loop-bindings [new-loop-body])
                                      (meta loop-form))]
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

(defn ^:no-doc assert-scalar-loss-result-admissible!
  "Reject positively-known non-scalar results before a public AD constructor
  synthesizes the unit cotangent of a scalar loss.

  `infer-result-tags` is the existing retained-source authority for reachable
  result alternatives.  It intentionally returns nil for result shapes whose
  fixed-point evidence is not retained yet (notably loops), so Unknown remains
  admitted here for compatibility and is NOT treated as proof of scalarity.
  The declared return tag contributes negative evidence only; it never selects
  the seed dtype, which continues to follow the actual walked-body checkpoint."
  [f-var resolved walked-body params tags]
  (let [m (meta resolved)
        body (last walked-body)
        tail (if (and (seq? body)
                      (contains? #{'let 'let*} (first body)))
               (last body)
               body)]
    ;; Preserve the established product-output diagnostic.  A literal vector
    ;; has no JVM primitive tag for infer-result-tags to classify.
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
              {:var f-var :tail-shape :vector :reason :tuple-valued-output})))
    (let [source-ns (or (some-> (:raster.core/deftm-source-ns m) find-ns)
                        (:ns m) *ns*)
          actual-tags (inf/infer-result-tags body (zipmap params tags) source-ns)
          declared-tag (:raster.core/return-tag m)
          evidence (cond-> (mapv (fn [tag] {:source :walked-body :tag tag})
                                 actual-tags)
                     declared-tag (conj {:source :declared-interface
                                         :tag declared-tag}))
          declined (filterv (fn [{:keys [tag]}]
                              (contains? #{:array :none}
                                         (:kind (tangent/tangent-kind tag))))
                            evidence)]
      (when (seq declined)
        (throw (ex-info
                (str "Scalar-loss AD requires a floating-point scalar result for `" f-var
                     "`; retained result evidence contains a known result tag "
                     "without a scalar tangent space.")
                {:reason :ad-nonscalar-loss-result
                 :var f-var
                 :actual-result-tags actual-tags
                 :declared-return-tag declared-tag
                 :declined-evidence declined})))
      {:actual-result-tags actual-tags
       :declared-return-tag declared-tag})))

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
  [f-var wrt]
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
          params (deftm-params-or-throw f-var resolved)
          walked-body (or (rcore/ensure-walked-body! resolved)
                          (throw (ex-info "No walked body on var" {:var f-var})))
          tags (or (:raster.core/deftm-tags m) (vec (repeat (count params) nil)))
          _ (assert-scalar-loss-result-admissible!
             f-var resolved walked-body params tags)
          all-params (vec (map-indexed
                           (fn [i p]
                             (let [tag (nth tags i nil)
                                   base (if (symbol? p) p (symbol (name p)))]
                               (if tag
                                 (with-meta base {:raster.type/tag tag})
                                 (with-meta base nil))))
                           params))
          parameter-plan (tangent-parameter-plan all-params tags wrt)
        ;; Unknown parameters refuse above unless explicit :wrt excludes them. Proven ⊥
        ;; parameters are positional constants and selected known tangent spaces stay ordered.
          diff-active-params (:candidate-params parameter-plan)
          constant-gradient-arrays (set (remove (set diff-active-params)
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
        ;; Entry bodies can still use source-ns aliases (e.g. dist/->Normal).
        ;; Qualify before inference/inlining: infer-arg-tag recognizes the
        ;; registered constructor by its canonical symbol, not a caller alias.
          qualified-source (inf/qualify-body-symbols
                            (first walked-body) source-ns (set params))
        ;; Shared pre-AD preparation (lower composites → materialize → hoist
        ;; into flat ANF) — see ad-prepare, shared with the JVP path.
          hoisted (binding [*ns* source-ns
                            *constant-gradient-arrays* constant-gradient-arrays]
                    (ad-prepare qualified-source
                                (zipmap all-params tags)))
        ;; transform-body delegates binding-position loops to gen-reverse-let's
        ;; activity-aware routing; inactive loops remain in the primal prefix.
          ad-form (binding [*constant-gradient-arrays* constant-gradient-arrays
                            *source-memory-context* {:values all-params
                                                     :value-facts {} :unprojectable {}
                                                     :replayed {} :aliases {}}
                            util/*shadowing-locals* (into util/*shadowing-locals* all-params)]
                    (transform-body hoisted diff-active-params))
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

(defn- runtime-function-form
  [params body]
  (if (<= (count params) 20)
    (list 'fn (vec params) body)
    (let [used (util/free-syms-flat [params body])
          args-sym (loop [id (gensym "args__")]
                     (if (contains? used id) (recur (gensym "args__")) id))
          unpack (vec (mapcat (fn [p i] [p (list 'clojure.core/nth args-sym i)]) params (range)))]
      (list 'fn ['& args-sym] (list 'let unpack body)))))

(defn- generated-helper-owner
  "Resolve only the actual globally retained receiver interface. No argument
   dtype or mangled function name is method-selection authority."
  [expression]
  (let [{:keys [dispatch? lexical-callee operation implementation-op arguments]}
        (op/call-description expression)]
    (when (and dispatch? (not lexical-callee))
      (let [hint (:tag (meta implementation-op))
            interface-name (if (class? hint) (symbol (.getName ^Class hint)) hint)
            interface-class (:iface-class (get @types/fn-interface-by-name interface-name))
            methods (when (class? interface-class)
                      (filter #(= "invk" (.getName ^java.lang.reflect.Method %))
                              (.getDeclaredMethods ^Class interface-class)))
            method (when (= 1 (count methods)) (first methods))]
        (when-not (and (qualified-symbol? operation) (qualified-symbol? implementation-op)
                       method (= (count arguments) (.getParameterCount ^java.lang.reflect.Method method))
                       (or (not (class? hint)) (identical? hint interface-class)))
          (throw (ex-info "generated helper requires its retained unique receiver descriptor"
                          {:reason :ad-generated-helper-descriptor :expression expression})))
        (let [key [operation implementation-op method]
              owners (:helpers *generated-product-context*)]
          (or (get @owners key)
              (let [owner (capture-selected-helper-emission
                           operation implementation-op (count arguments) method)]
                (when-not (every? #(= :pure (effects/analyze-effect %)) (get-in owner [:emission :body]))
                  (throw (ex-info "generated helper requires a read-only emitted body, even when inactive"
                                  {:reason :ad-paired-helper-effects :expression expression})))
                (swap! owners assoc key owner)
                (swap! (:guards *generated-product-context*) conj (:guard owner))
                (swap! (:methods *generated-product-context*) conj method)
                owner)))))))

(defn- generated-helper-record
  [sym expression activity]
  (when-let [{:keys [emission method]} (generated-helper-owner expression)]
    (let [{:keys [arguments]} (op/call-description expression)
          indices (vec (keep-indexed (fn [i argument]
                                      (when (and (symbol? argument) (get activity argument)) i)) arguments))
          artifact (get-paired-helper-program emission indices)
          _ (when-not (= (types/sym-type-tag sym) (:return-tag emission))
              (throw (ex-info "generated helper result must retain its emitted scalar carrier"
                              {:reason :ad-generated-helper-result :sym sym})))
          plan (ordered-call/plan-resolved-call method
                                               (:implementation-op (op/call-description expression))
                                               arguments (:source-policy *generated-product-context*))
          normalized (ordered-call/normalize-resolved-call plan ad-gensym
                                                          @(:used-identifiers *generated-product-context*))
          input-values (:input-values normalized)
          inputs (mapv (fn [index]
                         (let [source (nth arguments index)
                               source-tag (types/sym-type-tag
                                           (exact-replay-root source (:value-facts *source-memory-context*)))
                               target-tag (nth (:tags emission) index)
                               contract (nth (:input-contracts plan) index)]
                           (if (= :jvm-primitive-input (:source-context contract))
                             (assoc (floating-completion-tangent-plan
                                     contract source source-tag target-tag (:source-policy *generated-product-context*))
                                    :source source :floating? true)
                             (do
                               (when-not (and (= source-tag target-tag)
                                              (contains? types/primitive-array-info source-tag))
                                 (throw (ex-info "helper reference tangent requires its retained array carrier"
                                                 {:reason :ad-paired-helper-active-space :source source})))
                               {:source source :source-tag source-tag})))) indices)
          region (paired-helper-memory-region artifact arguments (set (:values *source-memory-context*)))
          companion (ad-gensym "helper_companion")
          tape (ad-gensym "helper_tape")
          pullback (ad-gensym "helper_pullback")]
      (swap! (:captures *generated-product-context*) conj {:port companion :value (:companion artifact)})
      (swap! (:guards *generated-product-context*) conj (:guard artifact))
      {:record {:type :paired-helper :sym sym :pullback pullback :inputs inputs
                :result-tag (:return-tag emission) :effects region}
       :fwd-patch (fn [bindings]
                    (into (vec (drop-last 2 bindings))
                          [tape (list 'let* (:bindings normalized) (list* companion input-values))
                           sym (list 'clojure.core/nth tape 0)
                           pullback (list 'clojure.core/nth tape 1)]))})))

(defn- assert-generated-region!
  "Check the already ordered source region. This does not normalize operands;
   child conditionals retain their own lexical bindings and evaluation order."
  [visible expression]
  (let [completion (value-completion/canonical-parts expression)
        {:keys [operation arguments dispatch?]} (when (seq? expression) (op/call-description expression))
        operator (when (symbol? operation) (resolve operation))
        helper (when (and dispatch? *generated-product-context*) (generated-helper-owner expression))
        fail! #(throw (ex-info "generated product binding has an unclosed input or effect scope"
                               {:reason :ad-generated-product-region :expression expression}))]
    (cond
      (and (seq? expression) (= 'if (first expression)))
      (do
        (when-not (and (= 4 (count expression)) (trivial-expr? (second expression))) (fail!))
        (doseq [part (rest expression)] (assert-generated-region! visible part)))

      (and (seq? expression) (contains? #{'let 'let*} (first expression)))
      (let [[_ bindings result] expression
            names (when (vector? bindings) (take-nth 2 bindings))]
        (when-not (and (= 3 (count expression)) (vector? bindings) (even? (count bindings))
                       (every? #(and (symbol? %) (nil? (namespace %))) names)
                       (= (count names) (count (distinct names)))
                       (empty? (set/intersection (set names) visible))) (fail!))
        (let [scope (reduce (fn [scope [name value]]
                              (assert-generated-region! scope value) (conj scope name))
                            visible (partition 2 bindings))]
          (assert-generated-region! scope result)))

      :else
      (do
        (when (and (seq? expression) (not completion)
                   (not (and (or helper (= :call (:kind (form/form-info expression))))
                              (or helper (not dispatch?)) (qualified-symbol? operation)
                              (var? operator) (not (:macro (meta operator)))
                              (every? trivial-expr? arguments)
                              (not (op/mutating-op? (op/semantic-op expression)))
                              (or helper (= :pure (effects/analyze-effect expression))))))
          (fail!))
        (let [operands (cond completion [(:operand completion)]
                             (seq? expression) arguments
                             (vector? expression) expression
                             (symbol? expression) [expression]
                             :else [])
              identifiers (into #{} (filter symbol?) (tree-seq vector? seq operands))]
          (when-not (every? visible identifiers)
            (throw (ex-info "generated product references a value outside its SSA scope"
                            {:reason :ad-generated-open-value :expression expression
                             :values (set/difference identifiers visible)})))))))
  expression)

(defn- generated-product-form
  [program params]
  (let [{:keys [bindings result active-inputs source-policy]} program
        names (when (vector? bindings) (vec (take-nth 2 bindings)))
        selected-inputs (when (vector? active-inputs) (mapv (zipmap params params) active-inputs))]
    (when-not (and (= #{:kind :bindings :result :active-inputs :source-policy} (set (keys program)))
                   (= :generated-product-bindings (:kind program))
                   (vector? params) (every? #(and (symbol? %) (nil? (namespace %))) params)
                   (= (count params) (count (distinct params)))
                   (value-completion/source-policy? source-policy)
                   (vector? bindings) (even? (count bindings))
                   (every? #(and (symbol? %) (nil? (namespace %))) names)
                   (= (count names) (count (distinct names)))
                   (empty? (set/intersection (set names) (set params)))
                   (symbol? result) (contains? (into (set params) names) result)
                   (vector? active-inputs) (= (count active-inputs) (count (distinct active-inputs)))
                   (every? (set params) active-inputs))
      (throw (ex-info "generated product AD requires a closed SSA binding program"
                      {:reason :ad-generated-product-program})))
    (doseq [parameter params]
      (when-not (or (get types/primitive-info (types/sym-type-tag parameter))
                    (get types/primitive-array-info (types/sym-type-tag parameter)))
        (throw (ex-info "generated product input requires a numeric carrier declaration"
                        {:reason :ad-generated-input-carrier :parameter parameter}))))
    (doseq [parameter selected-inputs]
      (when-not (tangent/differentiable? (types/sym-type-tag parameter))
        (throw (ex-info "generated product input has no retained tangent declaration"
                        {:reason :ad-product-unknown-tangent :parameter parameter}))))
    ;; The structural interpreter is read-only. Lexical branch regions are
    ;; checked before the existing conditional AD producer consumes them.
    (let [used (atom (util/free-syms-flat [params bindings]))
          captures (atom []) guards (atom []) methods (atom []) helpers (atom {})]
      (with-ad-gensym
        (binding [*generated-product-context* {:source-policy source-policy :used-identifiers used
                                              :captures captures :guards guards :methods methods :helpers helpers}
                  *source-memory-context* {:values params
                                          :value-facts (into {} (map (fn [p] [p {:value p}])) params)
                                          :aliases {} :activity (zipmap selected-inputs (repeat true))
                                          :replayed {} :unprojectable {}}
                  util/*shadowing-locals* (into (set params) names)]
          (reduce
           (fn [visible [name expression]]
             (assert-generated-region! visible expression)
             (conj visible name))
           (set params) (partition 2 bindings))
          (let [{:keys [fwd-bindings value-facts] :as forward} (forward-pass bindings selected-inputs)
                _ (when (get-in value-facts [result :product])
                    (throw (ex-info "structured result requires an explicit seed-port interface"
                                    {:reason :ad-generated-product-result})))
                result-tag (or (types/sym-type-tag result)
                               (some-> (get-in value-facts [result :value]) types/sym-type-tag))
                _ (when-not (= :scalar (:kind (tangent/tangent-kind result-tag)))
                    (throw (ex-info "generated product result requires a scalar tangent declaration"
                                    {:reason :ad-generated-product-result :tag result-tag})))
                dy (ad-gensym "generated_seed")
                {:keys [rev-ctx param-adj-syms]} (reverse-program-projection forward selected-inputs {result [dy]})]
            {:result-tag result-tag :captures @captures :guards @guards :methods @methods
             :form (:standard-form (reverse-program-forms
                                    {:body-sym result :dy dy :fwd-bindings fwd-bindings
                                     :rev-bindings (vec (:bindings rev-ctx))
                                     :param-adj-syms param-adj-syms}))}))))))

(defn- make-runtime-value+grad-fn
  "Build a runtime IFn from the AD-transformed walked body.
  Uses (& args) + indexed unpack so deftms with >20 params (Clojure's fn
  positional-arg limit) still work. The unpacking lets are trivially
  eliminated by the JVM JIT for the common ≤20-param case.

  Three-argument INTERNAL entry compiles an explicit generated ordered-call
  program under its OWN supplied policy, capturing actual Classes in private
  factory parameters. It does not derive an original deftm policy or enable
  completion AD rules/source-call normalization on the default path."
  ([walked-body params]
  ;; The body is eval'd WITHOUT type hints, so `clojure.core/alength` on an
  ;; untyped INTERMEDIATE array (e.g. a broadcast result `y`) reflects ambiguously
  ;; ("more than one matching method: alength"). Rewrite to the polymorphic
  ;; `raster.arrays/alength`, which dispatches on the runtime array type — no
  ;; reflection, and params (already typed) keep working.
  (let [body (walk/postwalk-replace {'clojure.core/alength 'raster.arrays/alength}
                                    (first walked-body))]
    ;; Keep canonical terms in retained derivative bodies for re-entry. Only
    ;; the executable source owner projects policy-independent Object inputs;
    ;; primitive/checkcast contexts without an owner still refuse here.
    (eval (value-completion/project-to-jvm-source (runtime-function-form params body)))))
  ([generated-call params {:keys [generated-source-policy] :as context}]
   ;; Explicit internal generated-program entry, not source-facing AD support.
   ;; The old two-argument path above neither discovers calls nor gains rules.
   (when-not (and (= #{:generated-source-policy} (set (keys context)))
                  (value-completion/source-policy? generated-source-policy)
                  (= generated-source-policy (:source-policy generated-call)))
     (throw (ex-info "generated invocation requires its own retained source policy"
                     {:reason :unclosed-generated-call-policy})))
   (let [product? (= :generated-product-bindings (:kind generated-call))
         ;; Capture before template derivation, not just before eval: otherwise
         ;; code derived in one epoch could be blessed with the next epoch.
         expected-revisions (when product?
                              ((requiring-resolve 'raster.ad.normalized-scalar/revisions)))
         proof-inputs (atom {:static [] :values []})
         product-program (when product?
                           (binding [method-entry/*emission-inputs* proof-inputs]
                             (generated-product-form generated-call params)))
         normalized (when-not product? (ordered-call/normalize-resolved-call generated-call gensym (set params)))
         source (if product?
                  (:form product-program)
                  (list 'let* (:bindings normalized) (:invocation normalized)))
         ;; Derive private Class ports from canonical completion terms only.
         ;; Unlike a Class literal these lexical ports retain loader identity.
         classes (volatile! #{})
         _ (when product?
             (util/postwalk-preserving-meta
              (fn [node]
                (when-let [{:keys [contract]} (value-completion/canonical-parts node)]
                  (when (= :jvm-checkcast (:source-context contract))
                    (vswap! classes conj (:target-carrier contract)))) node)
              source))
         input-classes (when product?
                         (mapv (fn [parameter]
                                 (let [tag (types/sym-type-tag parameter)]
                                   [parameter (or (get types/primitive-info tag)
                                                  (get types/primitive-array-info tag))])) params))
         source (if product?
                  (list* 'do
                         (concat
                          (map (fn [[parameter carrier]]
                                 (list 'if (list 'clojure.core/nil? parameter)
                                       (list 'throw (list 'clojure.core/ex-info
                                                          "generated numeric input must be present"
                                                          {:reason :ad-generated-input-carrier}))
                                       (value-completion/complete-value
                                        (value-completion/reference-check-contract carrier) parameter)))
                               input-classes)
                          [source]))
                  source)
         _ (when product? (doseq [[_ carrier] input-classes] (vswap! classes conj carrier)))
         helper-captures (vec (:captures product-program))
         source-identifiers (into (util/free-syms-flat [params source]) (map :port helper-captures))
         fresh-class-port (fn [] (loop [id (gensym "reference_class__")]
                                   (if (contains? source-identifiers id)
                                     (recur (gensym "reference_class__")) id)))
         ports (if product? (mapv (fn [carrier] {:class carrier :port (fresh-class-port)}) @classes)
                   (:class-ports normalized))
         reference-bindings (into {} (map (juxt :class :port)) ports)
         body (value-completion/project-to-jvm-source
               source
               generated-source-policy reference-bindings)]
     (let [captures (into helper-captures
                          (map (fn [{:keys [class port]}] {:port port :value class}) ports))
           guard (when product?
                     (let [capture (requiring-resolve 'raster.ad.normalized-scalar/execution-guard)]
                       (let [program-guard
                             (capture *ns* expected-revisions
                                      [{:parameters (into (vec params) (map :port captures)) :body body}])]
                         (fn []
                           (method-entry/assert-definition-inputs-current! @proof-inputs)
                           (program-guard)
                           (doseq [helper-guard (:guards product-program)] (helper-guard))))))
             execute (source-owner/compile-function
                      params source runtime-function-form
                      {:source-policy generated-source-policy :source-ns *ns*
                       :captures captures
                       :reference-bindings reference-bindings
                       :methods (if normalized [(:method normalized)] (vec (:methods product-program))) :guard guard})]
         (if-not product? execute
           (fn [& arguments]
             (guard)
             (let [[primal pullback] (apply execute arguments)]
               (guard)
               (when-not (.isInstance ^Class (get types/primitive-info (:result-tag product-program)) primal)
                 (throw (ex-info "generated scalar result has the wrong carrier"
                                 {:reason :ad-generated-result-carrier})))
               [primal (fn [seed]
                         (guard)
                         (when-not (.isInstance ^Class (get types/primitive-info (:result-tag product-program)) seed)
                           (throw (ex-info "generated scalar seed has the wrong carrier"
                                           {:reason :ad-generated-seed-carrier})))
                         (let [gradient (pullback seed)] (guard) gradient))])))))))

;; ================================================================
;; Forward-mode admissibility (framework §11): mode selection is
;; constrained by CARRIER COVERAGE — an interpretation is admissible
;; only when every op that would execute on the carrier has a lift in
;; that carrier. The coverage matrix is a queryable artifact derived
;; from the dispatch registry (the single source of truth for what a
;; carrier accepts) — never documentation, never a hardcoded op list.
;; ================================================================

(defn- dual-tag?
  "Use resolved type identity, never a basename shared with a foreign Dual."
  [tag]
  (let [carrier (types/tag->check-class 'raster.ad.forward.Dual)]
    (and (not= Object carrier)
         (or (symbol? tag) (class? tag) (types/compound-tag? tag))
         (= carrier (if (class? tag) tag (types/tag->check-class tag))))))

(def ^:private array-param-tags
  '#{doubles floats ints longs shorts bytes booleans chars objects})

(defn- forward-conversion-plan
  "One lexical Dual preparation: retain conversions and admit potentially active calls.
   Adapt only certified Double identities. Seed dependence is conservative, not proof
   that a dependent result is Dual-valued. Propagate selected result/carrier facts
   separately from primal type tags; unknown and Object results prove nothing."
  ([body params tags source-ns]
   (forward-conversion-plan body params tags source-ns {}))
  ([body params tags source-ns {:keys [validate-only? validation-state]}]
  (let [declines (volatile! []) call-declines (volatile! [])
        validation-state (or validation-state {:visiting (volatile! #{}) :remaining (volatile! 128)})
        initial-types (into {} (map (fn [p tag]
                                     [p {:tag tag
                                         :carriers #{(if (and (not validate-only?)
                                                             (= :scalar (:kind (tangent/tangent-kind tag))))
                                                       'raster.ad.forward.Dual tag)}}]) params tags))
        initial-active (set (keep (fn [[p tag]]
                                   (when (if validate-only? (dual-tag? tag)
                                             (= :scalar (:kind (tangent/tangent-kind tag)))) p))
                                 (map vector params tags)))]
    (letfn [(dependent? [expr active]
              (binding [util/*shadowing-locals* (into util/*shadowing-locals* active)]
                (boolean (seq (set/intersection active (util/free-syms expr))))))
            (tag-of [expr env]
              (if (and (symbol? expr) (contains? env expr))
                (:tag (get env expr))
                (or (types/sym-type-tag expr)
                    (inf/infer-expr-tag expr env source-ns))))
            (join-carriers [facts]
              (let [joined (apply set/union #{} facts)]
                (if (or (empty? joined) (contains? joined nil)) #{nil} joined)))
            (tag-fact
              ([tag] (tag-fact tag source-ns))
              ([tag defining-ns]
               (binding [*ns* defining-ns]
                 (let [cls (when (or (symbol? tag) (class? tag) (types/compound-tag? tag))
                             (if (class? tag) tag (types/tag->check-class tag)))]
                   #{(cond (dual-tag? tag) 'raster.ad.forward.Dual
                           (or (nil? cls) (= Object cls)) nil
                           (types/compound-tag? tag)
                           (apply list (symbol (.getName ^Class cls))
                                  (types/compound-tag-params tag))
                           (or (dtype/dtype-for-scalar-tag tag)
                               (contains? types/primitive-array-tags tag)) tag
                           :else (symbol (.getName ^Class cls)))}))))
            (tuples [facts]
              ;; Bound the product, not just the number of alternatives per operand.
              ;; An unknown or oversized product is a decline, never one guessed tuple.
              (when (and (every? #(and (seq %) (not (contains? % nil))) facts)
                         (<= (reduce *' 1 (map count facts)) 64))
                (reduce (fn [rows choices]
                          (vec (for [row rows choice choices] (conj row choice))))
                        [[]] facts)))
            (signature [operation tuple]
              (when (symbol? operation)
                (binding [*ns* source-ns]
                  (let [query #(dispatch/selected-call-signature
                                operation tuple (inf/registered-call-selection operation tuple))
                        selected (query)
                        selected (if (= :parametric (:selection selected))
                                   (try
                                     ;; A declared signature is applicability, not successful
                                     ;; compilation. Materialize through the existing compiler
                                     ;; callback, without invoking the callee or faking values.
                                     (dispatch/register-parametric-call-tags! operation tuple)
                                     (let [published (query)]
                                       (if (= :registered (:selection published)) published
                                           {:failure {:reason :unpublished-carrier-specialization}}))
                                     (catch clojure.lang.ExceptionInfo e
                                       (util/rethrow-compiler-invariant! e)
                                       {:failure {:message (.getMessage e) :data (ex-data e)}}))
                                   selected)
                        carrier? (some dual-tag? tuple)
                        dual-contract? (some dual-tag? (:tags selected))
                        casts (:promotion-casts selected)]
                    (cond
                      (:failure selected) selected
                      (and selected (or (not carrier?) dual-contract?)
                           (not-any? true? (map (fn [tag cast] (and (dual-tag? tag) (some? cast)))
                                                tuple (or casts (repeat nil)))))
                      (if (and carrier? (:parametric-provenance selected))
                        (let [key [(:mangled-sym selected) tuple]
                              visiting (:visiting validation-state)
                              remaining (:remaining validation-state)]
                          (cond
                            (contains? @visiting key)
                            {:failure {:reason :recursive-carrier-specialization :implementation (first key)}}
                            (not (pos? @remaining))
                            {:failure {:reason :carrier-validation-budget}}
                            :else
                            (do
                              (vswap! remaining dec)
                              (vswap! visiting conj key)
                              (try
                                (let [v (resolve (:mangled-sym selected))
                                      m (meta v)
                                      retained (when v (rcore/ensure-walked-body! v))
                                      ps (:raster.core/deftm-params m)
                                      ts (:raster.core/deftm-tags m)
                                      ns' (some-> (:raster.core/deftm-source-ns m) find-ns)]
                                  (if (and (seq retained) ns' (= (count tuple) (count ps) (count ts)))
                                    (let [plan (forward-conversion-plan
                                                (first retained) ps
                                                (mapv (fn [tag cast] (or cast tag))
                                                      tuple (or casts (repeat nil))) ns'
                                                {:validate-only? true :validation-state validation-state})]
                                      (if (and (empty? (:declines plan)) (empty? (:call-declines plan)))
                                        selected
                                        {:failure {:reason :unsupported-carrier-specialization
                                                   :implementation (:mangled-sym selected)
                                                   :conversion-declines (:declines plan)
                                                   :call-declines (:call-declines plan)}}))
                                    {:failure {:reason :missing-carrier-specialization-body}}))
                                (finally (vswap! visiting disj key))))))
                        selected)

                      :else nil)))))
            (call-facts [expr env active]
              (let [{:keys [operation arguments dispatch? recorded-op]} (op/call-description expr)
                    facts (mapv #(carrier-tags % env active) arguments)
                    rows (tuples facts)
                    signatures (when (and rows (or (not dispatch?) recorded-op))
                                 (mapv #(signature operation %) rows))]
                {:facts facts :tuples rows :signatures signatures
                 :result (if (and (seq signatures)
                                  (every? #(and % (not (:failure %))) signatures))
                           (join-carriers
                            (map (fn [selected]
                                   (tag-fact (:return-tag selected)
                                             (or (some-> (:mangled-sym selected) namespace symbol find-ns)
                                                 source-ns))) signatures))
                           #{nil})}))
            (bind-carriers [env active binders inits sequential?]
              (reduce (fn [[e a] [binder init]]
                        (let [ie (if sequential? e env) ia (if sequential? a active)]
                          [(assoc e binder {:tag (or (types/sym-type-tag binder) (tag-of init ie))
                                            :carriers (carrier-tags init ie ia)})
                           ((if (dependent? init ia) conj disj) a binder)]))
                      [env active] (map vector binders inits)))
            (loop-carriers [expr env active]
              (let [parsed (try (parse-loop-form expr)
                                (catch clojure.lang.ExceptionInfo _ nil))]
                (if parsed
                  (let [{:keys [loop-syms recur-args recur-bindings]} parsed]
                  (loop [e env round 0]
                    (let [[recur-env recur-active]
                          (bind-carriers e active (mapv first recur-bindings)
                                         (mapv second recur-bindings) true)
                          updates (mapv #(carrier-tags % recur-env recur-active) recur-args)
                          e' (reduce (fn [acc [sym fact]]
                                       (assoc-in acc [sym :carriers]
                                                 (join-carriers [(get-in e [sym :carriers] #{nil}) fact])))
                                     e (map vector loop-syms updates))]
                      (cond (= e e') e
                            (< round 8) (recur e' (inc round))
                            :else (reduce #(assoc-in %1 [%2 :carriers] #{nil}) e loop-syms)))))
                  (reduce #(assoc-in %1 [%2 :carriers] #{nil}) env (map first (partition 2 (second expr)))))))
            (carrier-tags [expr env active]
              (cond
                (and (symbol? expr) (contains? env expr)) (get-in env [expr :carriers] #{nil})
                (not (seq? expr)) (tag-fact (tag-of expr env))
                (= 'quote (first expr)) #{nil}
                (= :branch (:kind (form/form-info expr)))
                (join-carriers (map #(carrier-tags % env active) (drop 2 expr)))
                (= 'do (first expr)) (carrier-tags (last expr) env active)
                :else
                (if-let [{:keys [scopes sequential? rec?]} (form/scope-info expr)]
                  (if (or rec? (= :lambda (:kind (form/form-info expr))))
                    #{nil}
                    (join-carriers
                     (map (fn [{:keys [binders inits body]}]
                            (let [[e a] (bind-carriers env active binders inits sequential?)
                                  e (reduce #(assoc %1 %2 {:tag nil :carriers #{nil}})
                                            e (drop (count inits) binders))
                                  e (if (= :scope (:kind (form/form-info expr)))
                                      (loop-carriers expr e a) e)]
                              (if (= :scope (:kind (form/form-info expr)))
                                (if-let [parsed (try (parse-loop-form expr)
                                                     (catch clojure.lang.ExceptionInfo _ nil))]
                                  (carrier-tags (:result-expr parsed) e a)
                                  #{nil})
                                (carrier-tags (last body) e a)))) scopes)))
                  (let [{:keys [semantic-op arguments implementation-op]} (op/call-description expr)]
                    (cond
                      (op/cast-op? semantic-op)
                      (if (and (= 1 (count arguments))
                               (= :double (some-> semantic-op op/cast-result-tag dtype/dtype-for-scalar-tag))
                               (dependent? (first arguments) active))
                        (carrier-tags (first arguments) env active)
                        (tag-fact (op/cast-result-tag semantic-op)))
                      :else
                      (let [{:keys [result facts]} (call-facts expr env active)
                            carrier-call? (or (some #(some dual-tag? %) facts)
                                              (some true? (map (fn [fact arg]
                                                                 (and (contains? fact nil)
                                                                      (dependent? arg active)))
                                                               facts arguments))
                                              (dependent? implementation-op active))]
                        (if (= #{nil} result)
                          ;; Primal source inference is never a result witness for a
                          ;; carrier call. It remains usable for inactive expressions.
                          (if carrier-call? #{nil}
                              (tag-fact (or (tag-of expr env)
                                            (binding [*ns* source-ns]
                                              (inf/infer-rewritten-tag expr nil env)))))
                          result)))))))
            (carrier-dependent? [expr env active]
              (let [facts (carrier-tags expr env active)]
                (or (some dual-tag? facts)
                    (and (contains? facts nil) (dependent? expr active)))))
            (go [expr env active]
              (cond
                (and (seq? expr) (= 'quote (first expr))) expr
                (seq? expr)
                (if-let [{:keys [scopes outer rebuild sequential? rec?]} (form/scope-info expr)]
                  (let [scopes'
                        (mapv
                         (fn [{:keys [binders inits body] :as region}]
                           ;; Recurrences can acquire seed dependence on later iterations.
                           ;; Conservatively retain it for all carries, never assume init-only activity.
                           (let [recurrence? (= :scope (:kind (form/form-info expr)))
                                 [env' active' inits']
                                 (reduce
                                  (fn [[e a transformed] [binder init]]
                                    (let [init-env (if (or sequential? rec?) e env)
                                          init-active (if (or sequential? rec?) a active)
                                          transformed-init (go init init-env init-active)
                                          tag (or (types/sym-type-tag binder)
                                                  (tag-of init init-env))
                                          dep? (dependent? init init-active)]
                                      [(assoc e binder {:tag tag
                                                        :carriers (carrier-tags init init-env init-active)})
                                       ((if dep? conj disj) a binder)
                                       (conj transformed transformed-init)]))
                                  [(if rec? (reduce #(assoc %1 %2 {:tag (types/sym-type-tag %2)}) env binders) env)
                                   (if rec? (into active binders) active) []]
                                  (map vector binders inits))
                                 ;; Uninitialized binders shadow outer symbols. Their carrier
                                 ;; is not established here, so do not certify casts on them.
                                 uninitialized (drop (count inits) binders)
                                 env' (reduce #(assoc %1 %2 {:tag (types/sym-type-tag %2)}) env' uninitialized)
                                 active' (into active' uninitialized)
                                 carry-activity
                                 (when recurrence?
                                   (try
                                     (let [{:keys [bindings recur-args recur-bindings]}
                                           (parse-loop-form expr)]
                                       (loop-var-activity bindings recur-args active' recur-bindings))
                                     (catch clojure.lang.ExceptionInfo _
                                       ;; Noncanonical recurrences lack an update witness here.
                                       ;; Never certify their carries from initialization alone.
                                       (when (or (some active' binders)
                                                 (some #(dependent? % active') body))
                                         binders))))
                                 active' (into active' carry-activity)
                                 ;; A recurrence initializer is not an invariant type witness.
                                 ;; Invalidate it for active carries, including retained symbol
                                 ;; tags, while preserving proven inactive integer counters.
                                 env' (reduce #(assoc-in %1 [%2 :tag] nil) env' carry-activity)
                                 env' (if recurrence? (loop-carriers expr env' active') env')]
                             (assoc region :inits inits'
                                    :body (mapv #(go % env' active') body)))) scopes)]
                    (rebuild scopes' (mapv #(go % env active) outer)))
                  (let [{semantic-op :semantic-op args :arguments
                         operation :operation recorded :recorded-op
                         implementation :implementation-op dispatch? :dispatch?}
                        (op/call-description expr)]
                    (if (and (op/cast-op? semantic-op) (= 1 (count args))
                             (dependent? (first args) active))
                      (let [operand (first args)
                            source (some-> (tag-of operand env) dtype/dtype-for-scalar-tag)
                            target (some-> semantic-op op/cast-result-tag dtype/dtype-for-scalar-tag)
                            policy (when (and source target)
                                     (conversion/policy source target
                                                        (case (op/cast-integral-narrowing semantic-op)
                                                          :wrap :wrap :trap)))
                            adapted-operand (go operand env active)]
                        (if (and (not validate-only?) (= :double source target) (= [:exact :exact] policy))
                          (with-meta (list 'raster.ad.forward/identity-double-conversion adapted-operand)
                            (meta expr))
                          (do (vswap! declines conj
                                      {:reason :unsupported-forward-conversion
                                       :operation semantic-op :source-dtype source :target-dtype target
                                       :policy policy :form expr})
                              (with-meta (if (= '.invk (first expr))
                                           (list '.invk (second expr) adapted-operand)
                                           (list (first expr) adapted-operand))
                                (meta expr)))))
                      (do
                        (when (contains? #{:call :invk} (:kind (form/form-info expr)))
                          (let [active-arguments (vec (keep-indexed
                                                       (fn [i arg]
                                                         (when (carrier-dependent? arg env active) i)) args))
                                active-callee? (dependent? implementation active)
                                evidence (call-facts expr env active)
                                admissible-tuples? (and (seq (:tuples evidence))
                                                        (every? #(and % (not (:failure %)))
                                                                (:signatures evidence)))]
                            (when (and (or (seq active-arguments) active-callee?)
                                       (not (and (not active-callee?)
                                                 (or (not dispatch?) recorded)
                                                 (symbol? operation)
                                                 admissible-tuples?)))
                              (vswap! call-declines conj
                                      {:reason :unsupported-forward-call :operation operation
                                       :implementation implementation
                                       :active-argument-indices active-arguments
                                       :argument-carriers (:facts evidence)
                                       :argument-tuples (:tuples evidence)
                                       :signature-failures (vec (keep :failure (:signatures evidence)))
                                       :active-callee? active-callee? :form expr}))))
                        (with-meta (apply list (map #(go % env active) expr)) (meta expr))))))
                (vector? expr) (with-meta (mapv #(go % env active) expr) (meta expr))
                (map? expr) (with-meta (into (empty expr)
                                            (map (fn [[k v]] [(go k env active) (go v env active)])) expr)
                             (meta expr))
                (set? expr) (with-meta (into (empty expr) (map #(go % env active)) expr) (meta expr))
                :else expr))]
      {:body (go body initial-types initial-active)
       :declines @declines :call-declines @call-declines}))))

(defn- forward-preparation
  "Queryable Dual-carrier coverage for a deftm var (framework §4a/§11).
  The shared lexical preparation checks calls that may receive an active carrier,
  retaining argument and callee dependence instead of classifying heads alone.

  Returns {:admissible?   bool
           :uncovered-ops sorted vector of op symbols without a Dual lift
           :active-indices scalar tangent parameter positions
           :constant-params parameters with no tangent space
           :array-params  params whose type is an array (forward mode is
                          scalar-only — arrays have no Dual seeding)}"
  [f-var]
  (let [resolved (resolve-deftm-var f-var)
        m (meta resolved)
        walked-body (or (rcore/ensure-walked-body! resolved)
                        (throw (ex-info "No walked body on var" {:var f-var})))
        params (or (:raster.core/deftm-params m) [])
        tags (or (:raster.core/deftm-tags m) [])
        conversion-plan (forward-conversion-plan (first walked-body) params tags
                                                 (or (some-> (:raster.core/deftm-source-ns m) the-ns) *ns*))
        array-params (vec (remove nil?
                                  (map (fn [p tag]
                                         (when (or (contains? array-param-tags tag)
                                                   (and (seq? tag) (= 'Array (first tag))))
                                           p))
                                       params (concat tags (repeat nil)))))
        active-indices (vec (keep-indexed
                             (fn [i _]
                               (when (= :scalar (:kind (tangent/tangent-kind (nth tags i nil))))
                                 i)) params))
        constant-params (vec (keep-indexed
                              (fn [i p]
                                (when (= :none (:kind (tangent/tangent-kind (nth tags i nil))))
                                  p)) params))
        unknown-params (vec (keep-indexed
                             (fn [i p]
                               (when (= :unknown (:kind (tangent/tangent-kind (nth tags i nil))))
                                 p)) params))
        uncovered (->> (:call-declines conversion-plan)
                       ;; Unknown operand facts are propagated refusals, not
                       ;; evidence that an otherwise lifted consumer lacks a lift.
                       (filter #(seq (:argument-tuples %)))
                       (keep :operation)
                       (filter symbol?) distinct
                       sort
                       vec)]
    {:resolved resolved :params params :tags tags :walked-body walked-body
     :conversion-plan conversion-plan
     :coverage {:admissible? (and (empty? (:call-declines conversion-plan))
                                 (empty? array-params) (empty? unknown-params)
                                 (empty? (:declines conversion-plan)))
                :conversion-declines (:declines conversion-plan)
                :call-declines (:call-declines conversion-plan)
                :uncovered-ops uncovered
                :active-indices active-indices
                :constant-params constant-params
                :unknown-params unknown-params
                :array-params array-params}}))

(defn forward-coverage
  "Queryable Dual-carrier admission derived from the retained body and compiler
   type/conversion contracts. Includes uncovered ops, conversion declines,
   array parameters, and scalar tangent/constant parameter classification.
   May materialize applicable generic specializations through the compiler;
   never invokes helpers. Derived bodies are validated with actual carrier facts,
   while existing explicit Dual overloads remain trusted declared contracts."
  [f-var]
  (:coverage (forward-preparation f-var)))

(defn forward-admissible?
  "True when forward mode (the Dual carrier) is admissible for f-var:
  every op in the walked body that would execute on the Dual carrier has
  a Dual lift in the dispatch registry, and no param is array-typed."
  [f-var]
  (:admissible? (forward-coverage f-var)))

(def ^:private ^:dynamic *compiling-value+grad?*
  "True while a value+grad wrapper is being compiled: a value+grad made in
  that extent keeps the evaluated runtime body (no nested compilation)."
  false)

(defn- compiled-value+grad-fn
  "The runtime value+grad of `f-var` compiled by the full pipeline: a private
  wrapper deftm with `f`'s own parameters and annotations that binds the
  value+grad call and returns its elements — the pipeline inlines the
  AD-transformed body there and devirtualizes it, as for a value+grad inside
  any compiled deftm — then compile-aot. Returns {:fn compiled} or, when the
  pipeline cannot compile it (the caller keeps the evaluated runtime body),
  {:failure reason}."
  [f-var wrt]
  (when-not *compiling-value+grad?*
    (binding [*compiling-value+grad?* true]
      (try
        (let [resolved (resolve-deftm-var f-var)
              m (meta resolved)
              params (:raster.core/deftm-params m)
              anns (:raster.core/deftm-annotations m)
              source-ns (some-> (:raster.core/deftm-source-ns m) the-ns)
              target (symbol f-var)]
          (if-not (and source-ns (seq params) (= (count params) (count anns)))
            {:failure {:reason :missing-deftm-metadata :var f-var}}
            (let [wrapper (gensym (str (name target) "--value+grad-"))
                  slots (vec (repeatedly (inc (count params)) #(gensym "vg")))
                  form `(raster.core/deftm ~(with-meta wrapper {:private true})
                          [~@(mapcat (fn [p a] [p :- a]) params anns)]
                          (let [~slots ((raster.ad.reverse/value+grad (var ~target) ~@(when (some? wrt) [:wrt wrt]))
                                        ~@params)]
                            ~slots))]
              (binding [*ns* source-ns] (eval form))
              {:fn ((requiring-resolve 'raster.compiler.pipeline/compile-aot) (ns-resolve source-ns wrapper))})))
        (catch Throwable t {:failure t})))))

(defn ^:no-doc prepare-value+grad
  "Prepare the existing typed reverse-gradient program without compiling a runtime wrapper.
   Both runtime construction and compiler inlining consume this retained program/slot boundary."
  [f-var wrt]
  (or (build-grad-walked-body f-var wrt)
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
                   "registered op (e.g. a loss deftm with an AD template). "
                   "Runtime construction and compiled AD calls share this preparation gate.")
              {:var f-var :reason :flatten-failed}))))

(declare value+grad)

(defn- ^clojure.lang.IFn source-value+grad
  "Composable value+gradient operator.

  (value+grad f) returns a function that computes [value, grad1, ..., gradN].
  Works in two modes:
  - Runtime: callable directly as (fn [args...] -> [value grads...])
  - Compiled: carries deftm metadata so the pipeline can inline the
    AD-transformed body when called inside a compiled deftm.

  Options:
    :mode  - :reverse (default), :forward, or :auto
             Forward seeds only scalar tangent parameters. Discrete controls
             retain their primal values and occupy nil gradient slots, as in reverse.
    :wrt   - distinct zero-based parameter indices to differentiate in reverse
             mode; other gradient slots are nil (e.g. :wrt [0 3 4] keeps an
             observation array at index 1 constant)
    :compile? - reverse mode: build the runtime function with the full
             compile-aot pipeline (inlined, devirtualized backward pass — orders
             of magnitude faster on scalar-heavy loops) instead of evaluating
             the AD-transformed body. Pipeline semantics apply: dead checked
             reads may be eliminated and float arithmetic follows the compiled
             dtype. Like any compile-aot fn it may be called concurrently and
             with any shapes.
             Falls back to the evaluated body when the pipeline cannot
             compile `f`; `(:raster.ad.reverse/compiled? (meta vg))` says which and
             `(:raster.ad.reverse/compile-failure (meta vg))` why.

  Usage:
    ;; Runtime
    ((value+grad #'my-loss) 3.0 4.0)  ;; => [25.0 6.0 8.0]

    ;; Inside compiled deftm
    (deftm train-step [W x y lr] :- Double
      (let [[loss d_W _ _] ((value+grad #'my-loss) W x y)]
        (axpy! W W (- lr) d_W)
        loss))"
  ([f-var] (value+grad f-var :mode :reverse))
  ([f-var & {:keys [mode wrt compile?] :or {mode :reverse}}]
   (case mode
     :reverse
     (let [bgw (prepare-value+grad f-var wrt)
           {:keys [walked-body params tags source-ns]} bgw
           {compiled-fn :fn compile-failure :failure} (when compile? (compiled-value+grad-fn f-var wrt))
           runtime-fn (or compiled-fn (make-runtime-value+grad-fn walked-body params))
           ;; Qualify symbols in walked body for inlining from other namespaces
           inline-ns inf/qualify-body-symbols
           param-set (set params)
           qualified-wb (mapv #(inline-ns % source-ns param-set) walked-body)]
       ;; Return an IFn that also carries deftm metadata for the compiler
       (let [result-fn (fn [& args] (apply runtime-fn args))]
         (with-meta result-fn
           (cond-> {::value+grad true
                    ::compiled? (some? compiled-fn)
                    :raster.core/deftm true
                    :raster.core/deftm-walked-body qualified-wb
                    :raster.core/deftm-params params
                    :raster.core/deftm-tags tags}
             ;; why :compile? fell back to the evaluated body
             compile-failure (assoc ::compile-failure compile-failure)))))

     :forward
     (do
       (when (some? wrt)
         (throw (ex-info ":wrt is not yet supported in forward AD"
                         {:mode mode :wrt wrt})))
     ;; Forward-mode gradient via Dual numbers on the walked body.
     ;; Un-devirtualize .invk → generic dispatch so Dual numbers propagate.
     ;; Admissibility is checked at CONSTRUCTION time (framework §11): fail
     ;; with the uncovered ops named, never a No-matching-method at call time.
     (let [preparation (forward-preparation f-var)
           {resolved-for-result :resolved
            result-params :params
            result-body :walked-body
            result-tags :tags} preparation
           _ (assert-scalar-loss-result-admissible!
              f-var resolved-for-result result-body result-params result-tags)
           cov (:coverage preparation)
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
                            (when (seq (:unknown-params cov))
                              (str "params with unknown tangent spaces: "
                                   (str/join ", " (:unknown-params cov)) ". "))
                            (when (seq (:conversion-declines cov))
                              (str "source conversions without a faithful Dual interpretation: "
                                   (pr-str (:conversion-declines cov)) ". "))
                            (when (seq (:call-declines cov))
                              (str "active calls without a proven Dual dispatch: "
                                   (pr-str (:call-declines cov)) ". "))
                            "Mode selection is constrained by carrier coverage "
                            "(framework §11): use :mode :reverse, or add the "
                            "missing Dual overloads in raster.ad.forward.")
                       {:var f-var
                        :uncovered-ops (:uncovered-ops cov)
                        :conversion-declines (:conversion-declines cov)
                        :call-declines (:call-declines cov)
                        :array-params (:array-params cov)
                        :unknown-params (:unknown-params cov)})))
           resolved (:resolved preparation)
           params (deftm-params-or-throw f-var resolved)
           walked-body (:walked-body preparation)
           ;; Build a generic-dispatch version of the walked body
           generic-body (undevirtualize (first walked-body))
           conversion-plan (:conversion-plan preparation)
           active-params (vec (map #(with-meta (if (symbol? %) % (symbol (name %))) nil) params))
           generic-fn (eval (list 'fn active-params generic-body))
           tangent-fn (eval (list 'fn active-params (undevirtualize (:body conversion-plan))))
           n (count params)
           active-indices (:active-indices cov)
           active-set (set active-indices)
           make-dual fwd/->Dual
           dual-class (Class/forName "raster.ad.forward.Dual")
           get-v (fn [d] (.get (.getField dual-class "v") d))
           get-partials (fn [d] (.get (.getField dual-class "partials") d))]
       (fn [& args]
         (let [arg-vec (vec args)
               value (apply generic-fn arg-vec)
               grads (reduce
                      (fn [grads i]
                        (let [dual-args (into []
                                              (map-indexed
                                               (fn [j a]
                                                 ;; A zero tangent is not a Double conversion.
                                                 (if (contains? active-set j)
                                                   (make-dual (double a)
                                                              (double-array [(if (== j i) 1.0 0.0)]))
                                                   a))
                                               arg-vec))
                              result (apply tangent-fn dual-args)
                              deriv (if (.isInstance dual-class result)
                                      (clojure.core/aget ^doubles (get-partials result) 0)
                                      0.0)]
                          (assoc grads i deriv)))
                      (vec (repeat n nil))
                      active-indices)]
           (vec (cons (if (.isInstance dual-class value) (get-v value) value)
                      grads))))))

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
       (value+grad f-var :mode (if (some? wrt) :reverse selected) :wrt wrt)))))

(defn ^clojure.lang.IFn value+grad
  "Construct value and all positional gradient slots. Default :representation :source
   retains existing AD. :normalized-scalar opts into the checked scalar runtime bundle;
   it supports :mode :reverse and :wrt, but not :compile? true yet."
  ([f-var] (source-value+grad f-var))
  ([f-var & {:keys [representation mode wrt compile?]
             :or {representation :source mode :reverse} :as options}]
   (case representation
     :source (apply source-value+grad f-var (mapcat identity (dissoc options :representation)))
     :normalized-scalar
     (do
       (when-not (= :reverse mode)
         (throw (ex-info "normalized scalar value+grad requires reverse mode"
                         {:reason :normalized-scalar-mode :mode mode})))
       (when compile?
         (throw (ex-info "normalized scalar value+grad currently supports runtime construction only"
                         {:reason :normalized-scalar-compile-unsupported})))
       (let [bundle ((requiring-resolve 'raster.ad.normalized-scalar/prepare)
                     (resolve-deftm-var f-var) wrt)
             {:keys [walked-body params tags source-ns]}
             ((requiring-resolve 'raster.ad.normalized-scalar/value+grad-body) bundle)
             runtime-fn ((requiring-resolve 'raster.ad.normalized-scalar/guarded-runtime)
                         bundle params (first walked-body)
                         #(make-runtime-value+grad-fn walked-body params))]
         (with-meta runtime-fn
           {::value+grad true ::compiled? false :raster.core/deftm true
            :raster.core/deftm-walked-body walked-body
            :raster.core/deftm-params params :raster.core/deftm-tags tags
            :raster.core/deftm-source-ns (ns-name source-ns)
            :raster.ad/normalized-bundle bundle})))
     (throw (ex-info "unknown AD representation"
                     {:reason :ad-representation :representation representation})))))

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
    :wrt   - distinct zero-based parameter indices to differentiate in reverse
             mode; other gradient slots are nil

  Usage:
    ((grad #'my-loss) 3.0 4.0)   ;; => [6.0 8.0]
    ((grad #'square) 3.0)        ;; => 6.0        (1-param: bare scalar)
    ((grad (grad #'cube)) 2.0)   ;; => 12.0       (f'' composes directly)"
  ([f-var] (grad f-var :mode :reverse))
  ([f-var & {:keys [mode wrt] :or {mode :reverse}}]
   (let [vg-fn (value+grad f-var :mode mode :wrt wrt)
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
