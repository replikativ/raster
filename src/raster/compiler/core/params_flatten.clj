(ns raster.compiler.core.params-flatten
  "Source-level pre-flatten for HMap / HVec parameter pytrees.

  Rewrites a deftm body so every (:k m) / (get m :k) / (nth v i) on a typed
  pytree arg becomes a reference to a flat positional symbol. The walker, TC,
  bytecode emission, and lazy JIT path see no HMap — they see a flat-arg deftm.
  The wrapper kind (Param / Frozen) is tracked per leaf in the treedef so AD
  and the optimizer can dispatch at runtime."
  (:require [clojure.set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.compiler.core.params :as params]
            [raster.compiler.core.types :as types]))

;; ----------------------------------------------------------------------
;; Spec inspection
;; ----------------------------------------------------------------------

(defn hmap-spec? [s]
  (and (sequential? s) (= 'HMap (first s))))

(defn hvec-spec? [s]
  (and (sequential? s) (= 'HVec (first s))))

(defn pytree-spec? [s]
  (or (hmap-spec? s) (hvec-spec? s)))

(defn params-marker?
  "True iff annotation is the (Params <inner>) marker at the deftm-arg level.
  Params is the EXPLICIT signal that an arg is a structured pytree to be
  flattened at compile time. Inside the marker, the structure uses ordinary
  HMap/HVec/Param/Frozen forms."
  [annotation]
  (and (sequential? annotation) (= 'Params (first annotation))))

(defn unwrap-params
  "Unwrap (Params <inner>) → <inner>. Other annotations pass through."
  [annotation]
  (if (params-marker? annotation)
    (second annotation)
    annotation))

(defn hmap-mandatory
  "Extract :mandatory keys from an HMap spec.
  Supports (HMap :mandatory {:k T ...}) and the shorthand (HMap {:k T ...})."
  [hmap-spec]
  (let [tail (rest hmap-spec)]
    (cond
      ;; (HMap {:k T ...}) shorthand
      (and (= 1 (count tail)) (map? (first tail))) (first tail)
      ;; (HMap :mandatory {...} :optional {...} ...)
      :else (let [opts (apply hash-map tail)]
              (or (:mandatory opts) {})))))

(defn hvec-elems
  "Extract element types from an HVec spec: (HVec [T0 T1 ...]) -> [T0 T1 ...]."
  [hvec-spec]
  (let [v (second hvec-spec)]
    (if (vector? v) v [])))

(defn validate-pytree-spec!
  "Validate that a pytree spec uses only the closed, fully-specified constructs
  the Params machinery can flatten. Throws clojure.lang.ExceptionInfo with a
  pointed message on:

   - (HMap :optional {...})            — optional keys break canonical leaf order
   - (HMap :complete? false)           — open maps mean unknown leaves at compile time
   - (HMap :absent-keys ...)           — semantics ambiguous for closed pytree
   - (HMap <unknown-opt> ...)          — typo-guard for unsupported HMap options
   - (HVec X) where X is not a vector  — HVec must list its element types
   - (HVec [T...] <extras>)            — HVec doesn't take extra options

  Recurses into every nested HMap/HVec. Leaves are unchecked (any non-pytree
  type passes through). Returns the spec on success."
  [spec]
  (letfn [(walk [s]
            (cond
              (hmap-spec? s)
              (let [tail (rest s)
                    shorthand? (and (= 1 (count tail)) (map? (first tail)))]
                (when-not shorthand?
                  (when-not (and (zero? (mod (count tail) 2))
                                 (every? keyword? (take-nth 2 tail)))
                    (throw (ex-info
                             (str "Malformed HMap spec — expected (HMap {...}) "
                                  "or (HMap :mandatory {...} ...). Got: "
                                  (pr-str s))
                             {:spec s})))
                  (let [opts (apply hash-map tail)]
                    (when (contains? opts :optional)
                      (throw (ex-info
                               (str "Params HMap spec must not declare :optional "
                                    "keys (canonical leaf ordering requires a "
                                    "closed shape). Got: " (pr-str s))
                               {:spec s :forbidden :optional})))
                    (when (contains? opts :absent-keys)
                      (throw (ex-info
                               (str "Params HMap spec must not declare "
                                    ":absent-keys. Got: " (pr-str s))
                               {:spec s :forbidden :absent-keys})))
                    (when (and (contains? opts :complete?)
                               (not (true? (:complete? opts))))
                      (throw (ex-info
                               (str "Params HMap spec requires :complete? true "
                                    "(closed map). Got: " (pr-str s))
                               {:spec s :forbidden :complete?-false})))
                    (let [unknown (clojure.set/difference
                                    (set (keys opts))
                                    #{:mandatory :complete?})]
                      (when (seq unknown)
                        (throw (ex-info
                                 (str "Params HMap spec has unsupported options "
                                      (vec unknown)
                                      ". Supported: :mandatory, :complete? true. "
                                      "Got: " (pr-str s))
                                 {:spec s :unknown unknown}))))))
                (doseq [[_ child] (hmap-mandatory s)]
                  (walk child)))

              (hvec-spec? s)
              (let [tail (rest s)]
                (when-not (and (= 1 (count tail)) (vector? (first tail)))
                  (throw (ex-info
                           (str "Params HVec spec must be (HVec [T0 T1 ...]). "
                                "Got: " (pr-str s))
                           {:spec s})))
                (doseq [child (hvec-elems s)]
                  (walk child)))

              :else
              ;; Leaf type — pytree machinery doesn't inspect it further.
              nil))]
    (walk spec)
    spec))

(defn lookup-key
  "Look up :k in an HMap spec, returning the leaf or sub-spec type, or nil."
  [spec k]
  (when (hmap-spec? spec)
    (get (hmap-mandatory spec) k)))

(defn lookup-index
  "Look up index i in an HVec spec, returning the element type, or nil."
  [spec i]
  (when (hvec-spec? spec)
    (let [elems (hvec-elems spec)]
      (when (and (integer? i) (< -1 i (count elems)))
        (nth elems i)))))

;; ----------------------------------------------------------------------
;; Treedef: ordered description of leaves
;; ----------------------------------------------------------------------

(defn flatten-spec
  "Walk a spec depth-first, returning a vector of leaf descriptors in canonical
  order. Each entry: {:path [k|i ...] :type T :kind :param|:frozen|:plain}.

  HMap keys are sorted to give a stable order. HVec entries are in vector index
  order. Sub-pytrees recurse; non-pytree types are leaves."
  [spec]
  (letfn [(walk [s path]
            (cond
              (hmap-spec? s)
              (vec (mapcat (fn [[k t]] (walk t (conj path k)))
                           (sort-by (fn [[k _]] (str k)) (hmap-mandatory s))))

              (hvec-spec? s)
              (vec (mapcat (fn [i t] (walk t (conj path i)))
                           (range) (hvec-elems s)))

              :else
              [{:path  path
                :type  (params/inner-type s)
                :kind  (params/leaf-kind s)}]))]
    (walk spec [])))

(defn path->sym
  "Stable flat-arg symbol for a leaf at given path under root-sym.
  E.g. (path->sym 'w [:layers 0 :Wq]) => 'w__layers__0__Wq"
  [root-sym path]
  (if (empty? path)
    root-sym
    (symbol (str root-sym "__"
                 (str/join "__" (map (fn [p] (if (keyword? p) (name p) (str p)))
                                     path))))))

;; ----------------------------------------------------------------------
;; Access detection
;; ----------------------------------------------------------------------

(def ^:private get-syms #{'get 'clojure.core/get})
(def ^:private nth-syms #{'nth 'clojure.core/nth})

(defn extract-access-step
  "Recognize a single-step pytree access. Returns [target-form step] or nil.
  Supports (:k m), (get m :k), (nth v i-literal) with qualified or unqualified
  get/nth. Step is a keyword or integer."
  [form]
  (when (seq? form)
    (let [n (count form)
          op (first form)]
      (cond
        ;; (:keyword target)
        (and (= 2 n) (keyword? op))
        [(second form) op]
        ;; (get target :keyword)  — literal key only
        (and (= 3 n) (contains? get-syms op) (keyword? (nth form 2)))
        [(second form) (nth form 2)]
        ;; (nth target literal-int)
        (and (= 3 n) (contains? nth-syms op) (integer? (nth form 2)))
        [(second form) (nth form 2)]))))

(defn leaf-spec? [spec]
  (and (some? spec) (not (pytree-spec? spec))))

(defn path->flat-idx
  "Given a pytree spec and a leaf path, return the canonical flat-index of
  that path in the spec's leaves (in sorted-key/ascending-index order).
  Returns nil if path doesn't correspond to a leaf."
  [spec path]
  (let [target (vec path)]
    (some (fn [[idx leaf]] (when (= (:path leaf) target) idx))
          (map-indexed vector (flatten-spec spec)))))

(defn resolve-path
  "Try to interpret form as a static path under one of the typed roots in env.
  env entries are one of:

    Structured-view: {:root R :path P :spec S}
      Path access resolves to a flat positional arg via path->sym.

    Flat-view:       {:flat-view true :flat-source <expr> :flat-spec S
                      :starting-at I :path P :spec S'}
      Path access resolves to (nth flat-source <flat-idx>) where flat-idx
      is computed from path against flat-spec, plus the starting-at offset.

  Returns the env entry extended with the new path step, or nil if form is
  not a static path."
  [form env]
  (cond
    ;; A symbol bound to a known path
    (and (symbol? form) (contains? env form))
    (env form)

    :else
    (when-let [[target step] (extract-access-step form)]
      (when-let [parent (resolve-path target env)]
        (let [parent-spec (:spec parent)
              child-spec  (cond
                            (keyword? step) (lookup-key parent-spec step)
                            (integer? step) (lookup-index parent-spec step))]
          (when child-spec
            (let [new-path (conj (:path parent) step)]
              (if (:flat-view parent)
                (cond-> {:flat-view  true
                         :flat-source (:flat-source parent)
                         :flat-spec   (:flat-spec parent)
                         :starting-at (:starting-at parent 0)
                         :path        new-path
                         :spec        child-spec}
                  (leaf-spec? child-spec)
                  (assoc :flat-idx (+ (:starting-at parent 0)
                                      (or (path->flat-idx (:flat-spec parent) new-path)
                                          (throw (ex-info
                                                  (str "flat-view path " new-path
                                                       " not found in spec leaves")
                                                  {:path new-path
                                                   :spec (:flat-spec parent)}))))))
                {:root (:root parent)
                 :path new-path
                 :spec child-spec}))))))))

;; ----------------------------------------------------------------------
;; raster.tree/flat-view detection
;; ----------------------------------------------------------------------

(def ^:private flat-view-syms
  "Symbols that name the raster.tree/flat-view form (qualified or unqualified)."
  '#{flat-view raster.tree/flat-view})

(defn flat-view-form?
  "True iff form is a (raster.tree/flat-view source-vec & opts) call."
  [form]
  (and (seq? form)
       (>= (count form) 1)
       (contains? flat-view-syms (first form))))

(defn- unquote-form
  "Strip a (quote X) wrapper to get X. Both `:spec (HMap ...)` and `:spec '(HMap ...)`
  are accepted in the user-facing API; the latter wraps in a quote at read time."
  [form]
  (if (and (seq? form) (= 'quote (first form)))
    (second form)
    form))

(defn- parse-flat-view
  "Parse (flat-view source-vec :spec-of var :spec lit :starting-at idx).
  Returns {:source <expr> :spec <spec> :starting-at <int>} or throws.
  The :spec value is unquoted and unwrapped from any (Params ...) marker
  so callers can pass either the inner HMap/HVec directly or wrap with Params.

  Validates that the resolved spec is a well-formed pytree spec (HMap or HVec)
  — catches typos and malformed specs at macro time rather than producing
  silent garbage at runtime."
  [form]
  (let [[_ source & opts] form
        opt-map (apply hash-map opts)
        unknown-opts (clojure.set/difference (set (keys opt-map))
                                              #{:spec :spec-of :starting-at})
        _ (when (seq unknown-opts)
            (throw (ex-info (str "flat-view: unknown options " (vec unknown-opts)
                                 ". Recognized: :spec, :spec-of, :starting-at.")
                            {:form form :unknown unknown-opts})))
        starting-at (let [s (:starting-at opt-map 0)]
                      (when-not (integer? s)
                        (throw (ex-info "flat-view :starting-at must be a literal integer"
                                        {:got s})))
                      s)
        spec (cond
               (:spec opt-map) (unwrap-params (unquote-form (:spec opt-map)))
               (:spec-of opt-map)
               (let [v (:spec-of opt-map)
                     v (cond
                         (var? v) v
                         (and (seq? v) (= 'var (first v))) (resolve (second v))
                         (symbol? v) (resolve v)
                         :else (throw (ex-info "flat-view :spec-of must be a var"
                                               {:got v})))
                     treedefs (-> v meta :raster.params/treedefs)
                     [_ first-td] (first treedefs)]
                 (or (:spec first-td)
                     (throw (ex-info "flat-view :spec-of var has no :raster.params/treedefs metadata"
                                     {:var v}))))
               :else (throw (ex-info "flat-view requires :spec-of or :spec option" {:form form})))]
    (when-not (pytree-spec? spec)
      (throw (ex-info (str "flat-view :spec must be an HMap or HVec form (or wrapped "
                           "in Params). Got: " (pr-str spec))
                      {:form form :spec spec})))
    (validate-pytree-spec! spec)
    (when (zero? (count (flatten-spec spec)))
      (throw (ex-info "flat-view :spec has no leaves — empty pytree?"
                      {:form form :spec spec})))
    {:source source :spec spec :starting-at starting-at}))

;; ----------------------------------------------------------------------
;; Path-access expression builder (used by splicing, walk!, scan-vec)
;; ----------------------------------------------------------------------

(defn access-expr-for-path
  "Build a path-access expression: (:k1 (:k2 ... root)) or (nth ... i) for
  each step. Used by walk!, scan-vec, and cross-deftm splicing."
  [root-form path]
  (reduce (fn [acc step]
            (cond
              (keyword? step) (list step acc)
              (integer? step) (list 'clojure.core/nth acc step)
              :else (throw (ex-info "Bad path step" {:step step}))))
          root-form
          path))

;; ----------------------------------------------------------------------
;; Cross-deftm pytree call splicing
;;
;; When the body contains a call to a deftm whose params include
;; (Params ...), splice the pytree leaves at the call site instead of
;; passing a structured value. Replaces the call with one to the callee's
;; flat-var. Eliminates runtime Map reconstruction at deftm-to-deftm
;; boundaries — the structured wrapper is only used at the outer program
;; boundary (calls from non-pytree user code).
;; ----------------------------------------------------------------------

(defn- safe-to-duplicate?
  "True iff `form` can be evaluated multiple times with the same effect and
  result. Symbols, literals, and chains of (:k ...) / (get ... :k) / (nth ... i)
  rooted in such a form are safe. Anything else may have side effects or be
  non-trivial to recompute, so splicing must lift it to a temp binding."
  [form]
  (loop [f form]
    (cond
      (symbol? f) true
      (or (number? f) (string? f) (keyword? f)
          (true? f) (false? f) (nil? f) (char? f))
      true
      :else
      (if-let [[target _step] (extract-access-step f)]
        (recur target)
        false))))

(defn splice-cross-deftm-call
  "If form is a call to a deftm with Params-typed args, splice the pytree
  leaves at the call site to call the flat-var directly. Returns the
  rewritten form, or nil if no splicing applies.

  If a pytree arg is not safe to duplicate (e.g. a side-effecting expression
  or non-trivial computation), it is lifted into a temp let-binding so the
  evaluation happens exactly once — preserving the contract of the original
  flatten-on-call wrapper, which evaluates each pytree arg once and reads
  multiple keys from the result."
  [form _env]
  (when (and (seq? form) (symbol? (first form)))
    (let [f-sym (first form)
          args (vec (rest form))
          f-var (try (resolve f-sym) (catch Exception _ nil))
          callee-meta (when (var? f-var) (meta f-var))
          original-args (:raster.params/original-args callee-meta)
          treedefs (:raster.params/treedefs callee-meta)
          flat-var (:raster.params/flat-var callee-meta)]
      (when (and flat-var
                 original-args
                 (seq treedefs)
                 (= (count args) (count original-args)))
        (let [;; First pass: lift unsafe pytree args to temp bindings.
              tmp-binds (volatile! [])
              normalized-args
              (mapv (fn [arg-form callee-arg-name]
                      (if (and (get treedefs callee-arg-name)
                               (not (safe-to-duplicate? arg-form)))
                        (let [g (gensym (str "__pytree_arg__"
                                             (name callee-arg-name) "__"))]
                          (vswap! tmp-binds into [g arg-form])
                          g)
                        arg-form))
                    args
                    original-args)
              ;; Second pass: splice each pytree arg into per-leaf path-access
              ;; expressions, leaving non-pytree args untouched.
              spliced-args
              (vec
                (mapcat
                  (fn [arg-form callee-arg-name]
                    (if-let [td (get treedefs callee-arg-name)]
                      (let [leaves (:leaves td)]
                        (mapv (fn [{:keys [path]}]
                                (access-expr-for-path arg-form path))
                              leaves))
                      [arg-form]))
                  normalized-args
                  original-args))
              flat-fn-sym (symbol (str (.name (.ns flat-var)))
                                  (str (.sym flat-var)))
              call (cons flat-fn-sym spliced-args)]
          (if (seq @tmp-binds)
            (list 'let* @tmp-binds call)
            call))))))

;; ----------------------------------------------------------------------
;; raster.tree/walk! detection + expansion
;; ----------------------------------------------------------------------

(def ^:private walk-syms
  "Symbols that name the raster.tree/walk! form."
  '#{walk! raster.tree/walk!})

(defn walk-form?
  "True iff form is a (raster.tree/walk! kind f [pytree-args] & extras) call."
  [form]
  (and (seq? form) (>= (count form) 4) (contains? walk-syms (first form))))

(defn- count-typed-params
  "Count the actual params in a typed param vector like [a :- T b :- U c].
  `:-` and the type-form following it are skipped. Plain (untyped) param
  vectors are counted as-is."
  [param-vec]
  (loop [items (seq param-vec)
         n 0]
    (if-not items
      n
      (let [rest-items (next items)]
        (if (and rest-items (= ':- (first rest-items)))
          (recur (next (next rest-items)) (inc n))
          (recur rest-items (inc n)))))))

(defn- callee-arities
  "Return a set of accepted arities for `f-form` (a symbol). Uses, in order:
   1. (Params) defmodel metadata: count of :raster.params/original-args.
   2. Var :arglists (typed deftm or plain fn).
  Returns nil if no arity information is available — the caller should then
  skip the arity check rather than producing a false positive."
  [f-form]
  (when (symbol? f-form)
    (let [v (try (resolve f-form) (catch Exception _ nil))]
      (when (var? v)
        (let [m (meta v)]
          (cond
            (:raster.params/original-args m)
            #{(count (:raster.params/original-args m))}

            (:arglists m)
            (set (map count-typed-params (:arglists m)))))))))

(defn- expand-walk-form
  "Expand (tree/walk! kind f [pytree-args...] extras...) using compile-time
  spec info from the FIRST pytree-arg. Emits a `do` form with one f call per
  matching leaf.

  Arity-checks `f` against the call shape (one arg per pytree-arg + extras)
  when arity info is available. Catches user errors at macro time rather than
  surfacing as opaque IFn invoke failures at runtime."
  [form env]
  (let [[_ kind f-form pytree-args-vec & extras] form]
    (when-not (vector? pytree-args-vec)
      (throw (ex-info "tree/walk!: pytree-args must be a vector"
                      {:got pytree-args-vec})))
    (let [first-arg (first pytree-args-vec)
          first-resolved (resolve-path first-arg env)
          spec (or (:spec first-resolved)
                   (throw (ex-info (str "tree/walk!: first pytree-arg `"
                                        (pr-str first-arg)
                                        "` must resolve to a known pytree spec")
                                   {:arg first-arg :env-keys (keys env)})))
          leaves (flatten-spec spec)
          matching (filter #(or (= kind :any) (= (:kind %) kind)) leaves)
          expected-arity (+ (count pytree-args-vec) (count extras))
          arities (callee-arities f-form)]
      (when (and arities (not (contains? arities expected-arity)))
        (throw (ex-info
                 (str "tree/walk!: f `" (pr-str f-form) "` has arity "
                      (if (= 1 (count arities))
                        (str (first arities))
                        (str "in " (sort arities)))
                      ", but walk! emits per-leaf calls with arity "
                      expected-arity " ("
                      (count pytree-args-vec) " pytree arg"
                      (when (not= 1 (count pytree-args-vec)) "s")
                      " + " (count extras) " extra"
                      (when (not= 1 (count extras)) "s")
                      "). Make sure f's signature matches "
                      "(f leaf1 ... leafN extra1 ... extraK).")
                 {:f f-form
                  :expected expected-arity
                  :callee-arities arities
                  :pytree-args (count pytree-args-vec)
                  :extras (count extras)})))
      (when (empty? matching)
        (binding [*out* *err*]
          (println (format "WARN: tree/walk! :%s found no matching leaves for spec %s"
                           (name kind) (pr-str spec)))))
      ;; Emit a special splice marker — the per-leaf calls become
      ;; statement-position forms in the parent let body, mirroring the proven
      ;; pattern that compile-gsdm-train-fn uses (raw effects after let
      ;; bindings get lifted by normalize-let-body to _eff_NN bindings with
      ;; effectful metadata). The marker is detected by the let-body walker
      ;; in rewrite-body and spliced; downstream sees a normal multi-body let.
      (let [calls (mapv
                    (fn [{:keys [path]}]
                      (apply list f-form
                             (concat (mapv (fn [arg-form]
                                             (access-expr-for-path arg-form path))
                                           pytree-args-vec)
                                     extras)))
                    matching)]
        (cons ::splice-statements calls)))))

(defn splice-statements?
  "True iff form is a (::splice-statements ...) marker emitted by walk! and
  consumed by the let-body walker (which splices its statements into the
  parent body)."
  [form]
  (and (seq? form) (= ::splice-statements (first form))))

;; ----------------------------------------------------------------------
;; raster.tree/scan-vec detection + expansion
;; ----------------------------------------------------------------------

(def ^:private scan-vec-syms
  "Symbols that name the raster.tree/scan-vec form."
  '#{scan-vec raster.tree/scan-vec})

(defn scan-vec-form?
  "True iff form is a (raster.tree/scan-vec f init xs & extras) call."
  [form]
  (and (seq? form) (>= (count form) 4) (contains? scan-vec-syms (first form))))

(defn- reconstruct-at-path
  "Build expression that, at runtime, reproduces the structured value at
  `xs-form` followed by `path`, for the given `spec`.

  HMap → map literal; HVec → vector literal; leaf → direct path access.

  Used by scan-vec to materialize each iteration's element as a structured
  value the callee can consume. Leaf accesses emitted here flow through the
  pre-flatten walk and resolve to flat-arg symbols or flat-view nths."
  [xs-form path spec]
  (cond
    (hmap-spec? spec)
    (into {}
          (map (fn [[k sub-spec]]
                 [k (reconstruct-at-path xs-form (conj path k) sub-spec)]))
          (hmap-mandatory spec))

    (hvec-spec? spec)
    (mapv (fn [j sub-spec]
            (reconstruct-at-path xs-form (conj path j) sub-spec))
          (range)
          (hvec-elems spec))

    :else
    (access-expr-for-path xs-form path)))

(defn- expand-scan-vec
  "Expand (tree/scan-vec f init xs extras...) using compile-time HVec length.
  Emits a let* chain of N applications of f.

  When f resolves to a Params-typed defmodel, each iteration's call is spliced
  directly to the callee's --flat var with per-leaf path accesses into xs[i]
  — no per-iteration HMap reconstruction. Otherwise falls back to building a
  structured value and passing it to the user-facing wrapper."
  [form env]
  (let [[_ f-form init xs-form & extras] form
        xs-resolved (resolve-path xs-form env)
        spec (or (:spec xs-resolved)
                 (throw (ex-info (str "scan-vec: xs `" (pr-str xs-form)
                                      "` must resolve to a known pytree spec")
                                 {:xs xs-form :env-keys (keys env)})))
        _ (when-not (hvec-spec? spec)
            (throw (ex-info "scan-vec: xs must be an HVec sub-tree"
                            {:xs xs-form :spec spec})))
        elems (hvec-elems spec)
        n (count elems)
        ;; Try to resolve f as a Params-typed defmodel. If its second positional
        ;; arg (the element) is a pytree, splice each iteration directly to the
        ;; callee's --flat var rather than reconstructing the structured elt.
        f-var (when (symbol? f-form)
                (try (resolve f-form) (catch Exception _ nil)))
        f-meta (when (var? f-var) (meta f-var))
        f-original-args (:raster.params/original-args f-meta)
        f-treedefs (:raster.params/treedefs f-meta)
        f-flat-var (:raster.params/flat-var f-meta)
        elt-arg-name (when (and f-original-args (>= (count f-original-args) 2))
                       (nth f-original-args 1))
        elt-treedef (when elt-arg-name (get f-treedefs elt-arg-name))
        splice? (and elt-treedef f-flat-var)
        flat-fn-sym (when splice?
                      (symbol (str (.name (.ns f-flat-var)))
                              (str (.sym f-flat-var))))
        elt-leaves (when splice? (:leaves elt-treedef))
        acc-names (mapv #(symbol (str "scan_acc__" %)) (range (inc n)))
        bindings
        (vec
          (cons (first acc-names)    ;; init binding
                (cons init
                      (mapcat
                        (fn [i elt-spec]
                          (let [prev (nth acc-names i)
                                this (nth acc-names (inc i))
                                call (if splice?
                                       (apply list flat-fn-sym prev
                                              (concat
                                                (map (fn [{:keys [path]}]
                                                       (access-expr-for-path
                                                         xs-form (into [i] path)))
                                                     elt-leaves)
                                                extras))
                                       (let [elt (reconstruct-at-path xs-form [i] elt-spec)]
                                         (apply list f-form prev elt extras)))]
                            [this call]))
                        (range n)
                        elems))))]
    (list 'let* bindings (last acc-names))))

;; ----------------------------------------------------------------------
;; Body rewrite
;; ----------------------------------------------------------------------

(defn- bound-form?
  "Forms that introduce local bindings whose binding-vec we have to walk specially."
  [form]
  (and (seq? form) (#{'let 'let* 'loop 'loop*} (first form))))

(defn- update-let-env
  "Process let bindings left-to-right, returning [new-bindings-vec new-env].
  Each binding value is recursively walked with the running env. If the value
  resolves to a typed path:
   - Leaf path: the binding value becomes the flat-arg reference (alias kept).
   - Sub-tree path: the binding is DROPPED entirely. The original expression
     would dangle (the pytree-arg has been removed from the deftm signature),
     and emitting [name nil] confuses downstream passes that capture the
     binding into helper-method args. The binding name is recorded in env so
     subsequent leaf accesses resolve through it. If a sub-tree alias is used
     as a value (e.g. passed to a function), assert-no-dangling-pytree-refs!
     catches it post-walk."
  [bindings env walk-fn record-leaf!]
  (loop [pairs (partition 2 bindings)
         out []
         e env]
    (if-let [[bsym bval] (first pairs)]
      (let [resolved (resolve-path bval e)]
        (cond
          ;; flat-view declaration: parse opts, register binding as a flat-view
          ;; root in env, replace binding value with just the source expression.
          (flat-view-form? bval)
          (let [{:keys [source spec starting-at]} (parse-flat-view bval)
                new-e (assoc e bsym {:flat-view   true
                                     :flat-source source
                                     :flat-spec   spec
                                     :starting-at starting-at
                                     :path        []
                                     :spec        spec})
                walked-source (walk-fn source e)]
            (recur (rest pairs) (conj out bsym walked-source) new-e))

          ;; Leaf path: keep the binding; value becomes the flat-arg sym
          ;; (structured view) or (nth flat-source idx) (flat view).
          (and resolved (leaf-spec? (:spec resolved)))
          (let [walked-val (walk-fn bval e)
                new-e (if (:flat-view resolved)
                        ;; Flat-view leaf: bsym holds an array reference; no
                        ;; further structured access is meaningful — don't
                        ;; track in env.
                        e
                        (assoc e bsym {:root (:root resolved)
                                       :path (:path resolved)
                                       :spec (:spec resolved)}))]
            (when-not (:flat-view resolved)
              (record-leaf! resolved))
            (recur (rest pairs) (conj out bsym walked-val) new-e))

          ;; Sub-tree path: drop the binding; track in env only
          (and resolved (pytree-spec? (:spec resolved)))
          (recur (rest pairs) out (assoc e bsym resolved))

          :else
          (let [walked-val (walk-fn bval e)]
            (recur (rest pairs) (conj out bsym walked-val) e))))
      [out e])))

(defn rewrite-body
  "Rewrite body: replace static pytree access with flat-arg references.

  Returns {:body body' :leaves [{:root :path :spec :sym} ...]} where leaves
  describes every flat arg referenced by the rewritten body, in first-seen
  order. The caller can use `flatten-spec` over the original arg specs to get
  the *full* canonical order including unreferenced leaves."
  [body init-env]
  (let [leaves   (atom [])
        seen     (atom #{})
        record-leaf!
        (fn [{:keys [root path spec]}]
          (let [k {:root root :path path}]
            (when-not (@seen k)
              (swap! seen conj k)
              (swap! leaves conj {:root root :path path :spec spec
                                  :sym  (path->sym root path)}))))]
    (letfn [(walk [form env]
              (let [;; Compute splice candidate once. Skipped for forms that
                    ;; we know are not plain seq calls (walk!, scan-vec) so
                    ;; the metadata lookup doesn't run on every node.
                    spliced (when (and (seq? form)
                                       (not (walk-form? form))
                                       (not (scan-vec-form? form)))
                              (splice-cross-deftm-call form env))]
                (cond
                  ;; tree/walk! form: expand into per-leaf calls, then walk the result
                  (walk-form? form)
                  (walk (expand-walk-form form env) env)

                  ;; tree/scan-vec form: expand into a let* chain, then walk the result
                  (scan-vec-form? form)
                  (walk (expand-scan-vec form env) env)

                  ;; Cross-deftm pytree splicing: if the call target is a deftm
                  ;; whose original signature had Params args, replace the call
                  ;; with a direct call to its flat-var, with each pytree arg
                  ;; expanded into its leaves at the call site.
                  spliced
                  (walk spliced env)

                  ;; Static path access?
                  (let [r (resolve-path form env)]
                    (when (and r (leaf-spec? (:spec r)))
                      r))
                (let [r (resolve-path form env)]
                  (cond
                    ;; flat-view leaf: emit (nth flat-source flat-idx)
                    (:flat-view r)
                    (list 'clojure.core/nth (:flat-source r) (:flat-idx r))
                    ;; structured-view leaf: emit a flat positional arg sym
                    :else
                    (do (record-leaf! r)
                        (path->sym (:root r) (:path r)))))

                ;; let / let* / loop / loop*
                ;; Splice ::splice-statements forms returned by walk!/scan-vec
                ;; into the parent body so the per-leaf calls become
                ;; statement-position forms — matching the proven
                ;; compile-gsdm-train-fn pattern that survives through the
                ;; pipeline (lifted to _eff_NN bindings by normalize-let-body).
                (bound-form? form)
                (let [[op bindings & body-forms] form
                      [new-bindings new-env]
                      (update-let-env bindings env walk record-leaf!)
                      walked-body (map #(walk % new-env) body-forms)
                      spliced-body (mapcat (fn [f]
                                             (if (splice-statements? f)
                                               (rest f)
                                               [f]))
                                           walked-body)]
                  (apply list op (vec new-bindings) spliced-body))

                ;; Generic seq: recurse
                (seq? form)
                (apply list (map #(walk % env) form))

                (vector? form)
                (mapv #(walk % env) form)

                (map? form)
                (into (empty form) (map (fn [[k v]] [(walk k env) (walk v env)])) form)

                  (set? form)
                  (into (empty form) (map #(walk % env)) form)

                  :else form)))]
      (let [body' (walk body init-env)]
        {:body   body'
         :leaves @leaves}))))

;; ----------------------------------------------------------------------
;; Runtime flatten / unflatten
;; ----------------------------------------------------------------------

(defn flatten-value
  "Walk a runtime pytree value matching spec, returning a vector of leaves
  in canonical order (same as flatten-spec)."
  [spec value]
  (letfn [(walk [s v]
            (cond
              (hmap-spec? s)
              (vec (mapcat (fn [[k t]] (walk t (get v k)))
                           (sort-by (fn [[k _]] (str k)) (hmap-mandatory s))))
              (hvec-spec? s)
              (vec (mapcat (fn [i t] (walk t (nth v i nil)))
                           (range) (hvec-elems s)))
              :else
              [v]))]
    (walk spec value)))

(defn assert-no-identity-collisions!
  "Walk a runtime pytree value matching spec; throw if two leaves share JVM
  identity (the same object placed at two pytree positions, a.k.a. naked weight
  tying). Naked tying causes incorrect gradients: each position gets its own
  gradient buffer, so contributions from different paths don't accumulate.

  Resolution: use a single canonical pytree position for the shared parameter
  and reference it from multiple places in the model body, or wrap with `Tied`
  (Phase 2 — not yet implemented).

  Primitives (numbers, booleans, chars) are exempt — JVM identity is not
  meaningful for autoboxed primitives, and they don't carry gradient state."
  [spec value]
  (let [seen (java.util.IdentityHashMap.)
        ;; Use a volatile so we can break out early on the first collision.
        result (volatile! nil)]
    (letfn [(walk [s v path]
              (when (nil? @result)
                (cond
                  (hmap-spec? s)
                  (doseq [[k sub-spec] (sort-by (fn [[k _]] (str k))
                                                (hmap-mandatory s))]
                    (walk sub-spec (get v k) (conj path k)))

                  (hvec-spec? s)
                  (doseq [[i sub-spec] (map-indexed vector (hvec-elems s))]
                    (walk sub-spec (nth v i nil) (conj path i)))

                  :else
                  (when (and (some? v)
                             (not (number? v))
                             (not (boolean? v))
                             (not (char? v)))
                    (if-let [prev-path (.get seen v)]
                      (vreset! result [prev-path path])
                      (.put seen v path))))))]
      (walk spec value [])
      (when-let [[a b] @result]
        (throw (ex-info
                 (str "Pytree leaves at " (pr-str a) " and " (pr-str b)
                      " share JVM identity (same object at two positions). "
                      "Naked weight sharing causes incorrect gradients — each "
                      "position gets its own gradient buffer; contributions "
                      "from different paths don't accumulate. Use one "
                      "canonical pytree position and reference it from "
                      "multiple places in the model body, or wrap with `Tied` "
                      "(not yet implemented).")
                 {:path-a a :path-b b}))))))

(defn unflatten-value
  "Given spec and a flat vector of leaves in canonical order, reconstruct the
  pytree value. Inverse of flatten-value."
  [spec leaves]
  (letfn [(walk [s i]
            (cond
              (hmap-spec? s)
              (let [pairs (sort-by (fn [[k _]] (str k)) (hmap-mandatory s))]
                (loop [pairs pairs i i acc {}]
                  (if-let [[k t] (first pairs)]
                    (let [[i' v] (walk t i)]
                      (recur (rest pairs) i' (assoc acc k v)))
                    [i acc])))

              (hvec-spec? s)
              (let [elems (hvec-elems s)]
                (loop [elems elems i i acc []]
                  (if-let [t (first elems)]
                    (let [[i' v] (walk t i)]
                      (recur (rest elems) i' (conj acc v)))
                    [i acc])))

              :else
              [(inc i) (nth leaves i)]))]
    (second (walk spec 0))))

(defn filter-by-kind
  "Select runtime leaf values matching a given kind (:param / :frozen / :plain).
  leaf-descriptors comes from flatten-spec; flat-values from flatten-value
  (same length, same order)."
  [kind leaf-descriptors flat-values]
  (->> (map vector leaf-descriptors flat-values)
       (filter (fn [[ld _]] (= kind (:kind ld))))
       (mapv second)))

(defn pytree-shape
  "A small structural fingerprint of a runtime value: the spec walked to leaves,
  with each leaf replaced by [:leaf <class>] for its runtime class. Useful as a
  cache key when validating that a runtime call's pytree matches a treedef."
  [spec value]
  (let [leaves (flatten-value spec value)]
    (mapv (fn [v] [:leaf (some-> v class .getName)]) leaves)))

;; ----------------------------------------------------------------------
;; Top-level: prepare deftm args
;; ----------------------------------------------------------------------

(defn pytree-arg?
  "True iff annotation is the (Params ...) marker — the explicit signal that
  this deftm-arg is a structured pytree to be flattened at compile time.

  HMap/HVec annotations alone do NOT trigger flattening — they remain as
  regular runtime types. Wrap them in (Params ...) to opt into the pytree
  pipeline. This makes the contract syntactically explicit: looking at an
  annotation tells you whether the arg gets flattened."
  [annotation]
  (params-marker? annotation))

(defn- collect-symbols
  "Return the set of all symbols that appear anywhere in form."
  [form]
  (let [acc (volatile! #{})]
    (clojure.walk/postwalk
     (fn [x] (when (symbol? x) (vswap! acc conj x)) x)
     form)
    @acc))

(defn- assert-no-dangling-pytree-refs!
  "Pytree args are removed from the deftm signature by the pre-flatten pass.
  Any reference to a pytree-arg symbol (or a sub-tree alias bound only to a
  path) that survives the rewrite indicates the user used the value
  non-statically — e.g. passed it to a function or stored it. We can't compile
  that under B1; surface a clear error rather than producing dangling code."
  [rewritten-body pytree-arg-syms]
  (let [present (collect-symbols rewritten-body)
        leaks   (filter present pytree-arg-syms)]
    (when (seq leaks)
      (throw (ex-info
              (str "Pytree-typed deftm arg(s) " (vec leaks)
                   " referenced as a value in the body. The pre-flatten pass "
                   "only handles static leaf access via (:k m) / (get m :k) / "
                   "(nth v i). Passing a pytree (or sub-tree) as a value to "
                   "another function is not supported in this version.")
              {:leaked-symbols (vec leaks)})))))

(defn prepare-deftm
  "Pre-flatten a deftm signature. params and annotations come from
  parse-typed-params; body is the deftm body.

  For each pytree-typed param, all leaves become flat positional args
  (canonical sorted order). Non-pytree params pass through unchanged.

  Returns {:params new-params, :annotations new-anns, :body new-body,
           :treedefs {root-sym {:spec ... :leaves [{:path :sym :kind :type}]}}}"
  [params annotations body]
  (let [;; Pytree args are tagged with (Params <inner>). Use the inner spec
        ;; for flattening; the Params wrapper is just the trigger.
        param-inner (fn [ann] (when (pytree-arg? ann) (unwrap-params ann)))
        ;; Reject malformed pytree specs at the deftm boundary so the user
        ;; sees a clear error here instead of silent leaf misordering or a
        ;; cryptic downstream failure.
        _ (doseq [[p ann] (map vector params annotations)
                  :when (pytree-arg? ann)]
            (try (validate-pytree-spec! (unwrap-params ann))
                 (catch clojure.lang.ExceptionInfo e
                   (throw (ex-info (str "Invalid Params spec for arg `" p "`: "
                                        (.getMessage e))
                                   (assoc (ex-data e) :param p)
                                   e)))))
        ;; Build initial env: each pytree param becomes a path root with the
        ;; INNER (unwrapped) structural spec.
        init-env (into {}
                       (keep (fn [[p ann]]
                               (when-let [inner (param-inner ann)]
                                 [p {:root p :path [] :spec inner}]))
                             (map vector params annotations)))
        ;; For each pytree arg, generate canonical flat leaves up-front so
        ;; ordering is stable independent of body access order
        canonical-leaves
        (into {}
              (keep (fn [[p ann]]
                      (when-let [inner (param-inner ann)]
                        [p (mapv (fn [{:keys [path] :as leaf}]
                                   (assoc leaf :sym (path->sym p path)))
                                 (flatten-spec inner))]))
                    (map vector params annotations)))
        pytree-arg-syms (vec (keep (fn [[p ann]] (when (pytree-arg? ann) p))
                                   (map vector params annotations)))
        ;; Rewrite the body
        {body' :body} (rewrite-body body init-env)
        _ (assert-no-dangling-pytree-refs! body' pytree-arg-syms)
        ;; Build new param + annotation vectors
        [new-params new-anns]
        (reduce (fn [[ps as] [p ann]]
                  (if (pytree-arg? ann)
                    (let [leaves (canonical-leaves p)]
                      [(into ps (map :sym) leaves)
                       (into as (map (fn [{:keys [type kind]}]
                                       ;; Keep wrapper kind for consumers downstream
                                       (case kind
                                         :param  (list 'Param type)
                                         :frozen (list 'Frozen type)
                                         type))
                                     leaves))])
                    [(conj ps p) (conj as ann)]))
                [[] []]
                (map vector params annotations))
        treedefs (into {}
                       (keep (fn [[p ann]]
                               (when-let [inner (param-inner ann)]
                                 [p {:spec inner :leaves (canonical-leaves p)}]))
                             (map vector params annotations)))]
    {:params      new-params
     :annotations new-anns
     :body        body'
     :treedefs    treedefs}))
