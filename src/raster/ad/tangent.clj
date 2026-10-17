(ns raster.ad.tangent
  "Π — the tangent-type protocol (framework §5, task #44).

  A COMPILE-TIME protocol: these functions are consulted during the AD
  transform / flatten / template codegen to emit correctly-TYPED seeds,
  zeros, and casts in the generated IR, derived from the PRIMAL's type
  tag. Never runtime wrapper objects — a tagged-object tangent flowing
  through compiled code would destroy the flat fusible backward. The
  tangent algebra acts at STAGE time and emits monomorphic code.

  Runtime `nil` remains the dynamic ⊕-identity (grad-acc handles it);
  ⊥ (no tangent space — primitive integral/Boolean slots) remains STATIC:
  `differentiable?` is the type-level ⊥ test that activity analysis
  abstracts.

  Tag vocabulary is the :raster.type/tag / deftm-tags symbols:
    double float          → scalar tangent spaces
    doubles floats        → array tangent spaces
    long longs int ints boolean … → ⊥ (no tangent space)
    Object, reference tags, and nil     → unknown (not evidence of ⊥)

  Dependency-light BY DESIGN: only retained type vocabulary and canonical
  operation identity are required. Both raster.ad.reverse and
  raster.compiler.ad.flatten (and compiler passes) may require this ns
  without creating cycles."
  (:refer-clojure :exclude [])
  (:require [raster.compiler.core.types :as types]
            [raster.ad.accumulation :as accumulation]
            [raster.compiler.core.op-descriptor :as descriptor]))

(def ^:private scalar-tag->dtype {'double :double 'float :float})
(def ^:private array-tag->dtype  {'doubles :double 'floats :float})

(defn tangent-kind
  "Classify a primal type tag into its tangent space.
  Returns {:kind :scalar|:array, :dtype :double|:float} for tags with a
  tangent space, {:kind :none} for proven ⊥ primitive tags, and
  {:kind :unknown} when the tag supplies no tangent-space evidence."
  [tag]
  (if-let [dt (scalar-tag->dtype tag)]
    {:kind :scalar :dtype dt}
    (if-let [dt (array-tag->dtype tag)]
      {:kind :array :dtype dt}
      (if (or (and (contains? types/primitive-info tag)
                   (not (contains? scalar-tag->dtype tag)))
              (and (contains? types/primitive-array-info tag)
                   (not (contains? array-tag->dtype tag))))
        {:kind :none}
        {:kind :unknown}))))

(defn differentiable?
  "Does `tag` positively denote a tangent space that carries gradient?
  False includes both proven NoTangent and Unknown; admission callers must
  inspect `tangent-kind` rather than treating false as proof of inactivity."
  [tag]
  (contains? #{:scalar :array} (:kind (tangent-kind tag))))

(defn tangent-tag
  "Π — the tangent-type projection on TAGS (.internal/ad_formal_framework.md
  §5, item #1): maps a PRIMAL's :raster.type/tag to the tag of its
  tangent/cotangent space. Identity on the differentiable tags
  (double/float scalars, doubles/floats arrays — exactly the tags
  `differentiable?` accepts), nil for everything else — proven ⊥ or unknown.

  The single seam every AD emission site asks instead of copying the
  primal tag ad hoc or defaulting a dtype: a nil result means the
  emission stays UNTAGGED (never guess, never narrow)."
  [primal-tag]
  (when (differentiable? primal-tag) primal-tag))

(defn zero-expr
  "IR expression for a typed zero cotangent of the tangent space of `tag`.
  Scalars: a typed zero literal. Arrays: a typed array constructor sized
  by `shape-expr` (an IR expression for the element count).

  FAIL-LOUD on ⊥/unknown tags: a materialized zero whose tangent space is
  unknown could silently mis-shape an array slot (scalar 0.0 where an
  array adjoint is expected). Callers that can tolerate absence must use
  nil (the dynamic 0̄) instead of asking for a materialized zero."
  [tag shape-expr]
  (let [{:keys [kind dtype]} (tangent-kind tag)]
    (case kind
      :scalar (case dtype :float (list 'float 0.0) :double 0.0)
      :array  (case dtype
                :float  (list 'float-array shape-expr)
                :double (list 'double-array shape-expr))
      (throw (ex-info (str "No tangent space for tag " (pr-str tag)
                           " — cannot materialize a typed zero cotangent. "
                           "Untagged adjoint slots must stay nil (dynamic 0̄) "
                           "or the primal must carry a :raster.type/tag.")
                      {:tag tag :shape-expr shape-expr})))))

(defn seed-expr
  "IR expression for the typed seed cotangent (1̄) of a SCALAR loss with
  type `tag`. Untagged/unknown and proven NoTangent tags retain the legacy
  scalar-loss Double seed; array tags fail loud. Parameter admission is the
  safety boundary for Unknown in this slice; internal result-tag propagation
  and seed precision are separate work."
  [tag]
  (let [{:keys [kind dtype]} (tangent-kind tag)]
    (case kind
      :scalar (case dtype :float (list 'float 1.0) :double 1.0)
      :none 1.0
      :unknown 1.0
      (throw (ex-info (str "Seed requires a scalar loss; got array tag " (pr-str tag))
                      {:tag tag})))))

;; ================================================================
;; Projection Π_x — runtime helpers + the expr emitter
;; ================================================================

(defn project-double
  "Runtime Π onto a Double primal's tangent space. nil-safe: nil is the
  dynamic 0̄ and passes through."
  [x]
  (when (some? x) (Double/valueOf (double x))))

(defn project-float
  "Runtime Π onto a Float primal's tangent space. nil-safe: nil is the
  dynamic 0̄ and passes through."
  [x]
  (when (some? x) (Float/valueOf (float x))))

(defn zero-like
  "Runtime Π fallback: a zero cotangent shaped like the PRIMAL value `x`,
  deriving dtype+shape dynamically. Used only where no static tag exists
  (e.g. untagged if-result syms in runtime AD) — unlike a bare scalar 0.0
  it can never mis-shape an array slot. nil (0̄) passes through."
  [x]
  (cond
    (nil? x) nil
    (instance? Double x) 0.0
    (instance? Float x) (Float/valueOf (float 0.0))
    (.isArray (class x))
    (java.lang.reflect.Array/newInstance (.getComponentType (class x))
                                         (java.lang.reflect.Array/getLength x))
    (number? x) 0.0
    :else (throw (ex-info (str "No runtime tangent zero for value of class "
                               (class x))
                          {:class (class x)}))))

(defn- explicit-scalar-cast-tag [expression]
  (when (and (seq? expression) (= 1 (count (descriptor/call-args expression))))
    (let [tag (descriptor/cast-result-tag (descriptor/semantic-op expression))]
      (when (contains? scalar-tag->dtype tag) tag))))

(defn- expr-manifest-tag
  "The statically KNOWN scalar tag of an IR expression — carried or manifest,
  never guessed from the target primal: a carried :raster.type/tag stamp,
  explicit cast, literal, or the approved optional sum's retained result facet.
  nil when the expression's type is not statically known."
  ([expr] (expr-manifest-tag expr {}))
  ([expr binding-map]
   (letfn [(known-tag [expr seen]
             (or (cond
                   (or (symbol? expr) (seq? expr))
                   (or (:raster.type/tag (meta expr))
                       (explicit-scalar-cast-tag expr)
                       (when (and (seq? expr) (accumulation/approved-call? expr))
                         (accumulation/nullable-result-tag
                          expr (mapv #(known-tag % seen) (descriptor/call-args expr)))))
                   (float? expr) (types/literal-tag expr)
                   :else nil)
                 ;; SSA aliases retain the source value's type even when an intermediate
                 ;; binder has no tag. Never infer a function's result or the desired target
                 ;; type; missing, cyclic, and dynamically nil-producing chains stay unknown.
                 (when (and (symbol? expr) (not (contains? seen expr))
                            (contains? binding-map expr))
                   (known-tag (get binding-map expr) (conj seen expr)))))]
     (known-tag expr #{}))))

(defn project-expr
  "Wrap `cotangent-expr` with the projection onto the tangent space of the
  primal `tag` (Π_x), when the dtypes could mismatch.

  Emission is TYPED IR whenever the cotangent's type is statically known
  (carried tag / manifest cast / literal): same dtype → identity (no wrap),
  different scalar dtype → a nil-preserving primitive conversion. A carried
  scalar tag describes the present value's dtype, not proof that a nullable
  cotangent is present. Unknown/nullable values use ordinary let/if/cast IR,
  evaluating their source once; no opaque runtime projection call is emitted.
  Arrays and unknown primal tags pass through unchanged (array adjoints are
  anchored by their typed shadow buffers).

  `:materialized? true` is a caller obligation that the expression produces a concrete,
  non-nil scalar numeric value (for example an executed kernel adjoint). It permits a primitive scalar
  projection even when its exact dtype has not survived inlining; it does not guess that dtype
  or weaken the default nil-safe dynamic-cotangent contract.

  `:binding-map` supplies existing SSA bindings for following untagged aliases to
  a carried/manifest source tag. It does not derive types from operator names."
  ([tag cotangent-expr] (project-expr tag cotangent-expr {}))
  ([tag cotangent-expr {:keys [materialized? binding-map]}]
  (let [{:keys [kind dtype]} (tangent-kind tag)]
    (if (= kind :scalar)
      (let [ct (tangent-kind (expr-manifest-tag cotangent-expr binding-map))
            cast-tag (case dtype :float 'float :double 'double)
            cast-op (symbol "clojure.core" (name cast-tag))]
        (cond
          ;; Statically in the primal's tangent space already — Π is the identity.
          (and (= :scalar (:kind ct)) (= dtype (:dtype ct)))
          cotangent-expr
          ;; Explicit source casts and literals are materialized on normal
          ;; return. Their evaluation (including original traps) stays intact.
          (or materialized? (number? cotangent-expr)
              (explicit-scalar-cast-tag cotangent-expr))
          (with-meta (list cast-op cotangent-expr)
            {:raster.type/tag cast-tag})
          ;; Nullable/unknown: retain absence and single source evaluation.
          ;; A tag alone must never turn nil into a primitive-cast exception.
          :else
          (let [input (with-meta (gensym "projection_input__")
                        (select-keys (meta cotangent-expr) [:raster.type/tag]))]
            (with-meta
              (list 'let* [input cotangent-expr]
                    (list 'if (list 'clojure.core/nil? input) nil
                          (with-meta (list cast-op input) {:raster.type/tag cast-tag})))
              {:raster.type/tag cast-tag}))))
      cotangent-expr))))
