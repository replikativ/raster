(ns raster.ad.tangent
  "Π — the tangent-type protocol (framework §5, task #44).

  A COMPILE-TIME protocol: these functions are consulted during the AD
  transform / flatten / template codegen to emit correctly-TYPED seeds,
  zeros, and casts in the generated IR, derived from the PRIMAL's type
  tag. Never runtime wrapper objects — a tagged-object tangent flowing
  through compiled code would destroy the flat fusible backward. The
  tangent algebra acts at STAGE time and emits monomorphic code.

  Runtime `nil` remains the dynamic ⊕-identity (grad-acc handles it);
  ⊥ (no tangent space — Long/Boolean/Object slots) remains STATIC:
  `differentiable?` is the type-level ⊥ test that activity analysis
  abstracts.

  Tag vocabulary is the :raster.type/tag / deftm-tags symbols:
    double float          → scalar tangent spaces
    doubles floats        → array tangent spaces
    long longs int ints boolean Object … (and nil) → ⊥ (no tangent space)

  Dependency-light BY DESIGN: no requires. Both raster.ad.reverse and
  raster.compiler.ad.flatten (and compiler passes) may require this ns
  without creating cycles."
  (:refer-clojure :exclude []))

(def ^:private scalar-tag->dtype {'double :double 'float :float})
(def ^:private array-tag->dtype  {'doubles :double 'floats :float})

(defn tangent-kind
  "Classify a primal type tag into its tangent space.
  Returns {:kind :scalar|:array, :dtype :double|:float} for tags with a
  tangent space, {:kind :none} for ⊥ tags (integers, booleans, Object,
  untagged/nil — no tangent space or statically unknown)."
  [tag]
  (if-let [dt (scalar-tag->dtype tag)]
    {:kind :scalar :dtype dt}
    (if-let [dt (array-tag->dtype tag)]
      {:kind :array :dtype dt}
      {:kind :none})))

(defn differentiable?
  "Type-level ⊥ test: does `tag` denote a tangent space that carries
  gradient? (The single source of truth that reverse.clj's
  differentiable-tag? and the inliner's local copy delegate to.)"
  [tag]
  (not= :none (:kind (tangent-kind tag))))

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
  type `tag`. nil/unknown tags fall back to the double 1.0 (the loss of a
  value+grad is scalar by contract; array tags fail loud)."
  [tag]
  (let [{:keys [kind dtype]} (tangent-kind tag)]
    (case kind
      :scalar (case dtype :float (list 'float 1.0) :double 1.0)
      :none 1.0
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

(defn project-expr
  "Wrap `cotangent-expr` with the projection onto the tangent space of the
  primal `tag` (Π_x), when the dtypes could mismatch. Scalars get a
  nil-safe cast call; arrays and unknown tags pass through unchanged
  (array adjoints are anchored by their typed shadow buffers)."
  [tag cotangent-expr]
  (let [{:keys [kind dtype]} (tangent-kind tag)]
    (if (= kind :scalar)
      (case dtype
        :float  (list 'raster.ad.tangent/project-float cotangent-expr)
        :double (list 'raster.ad.tangent/project-double cotangent-expr))
      cotangent-expr)))
