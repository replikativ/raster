(ns raster.compiler.core.params-flatten
  "Source-level pre-flatten for HMap / HVec parameter trees.

  Rewrites a deftm body so every (:k m) / (get m :k) / (nth v i) on a typed
  tree arg becomes a reference to a flat positional symbol. The walker, TC,
  bytecode emission, and lazy JIT path see no HMap — they see a flat-arg deftm.
  The wrapper kind (Param / Frozen) is tracked per leaf in the treedef so AD
  and the optimizer can dispatch at runtime."
  (:require [clojure.set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.compiler.core.params :as params]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.form :as form]))

;; ----------------------------------------------------------------------
;; Spec inspection
;; ----------------------------------------------------------------------

(defn hmap-spec? [s]
  (and (sequential? s) (= 'HMap (first s))))

(defn hvec-spec? [s]
  (and (sequential? s) (= 'HVec (first s))))

(defn tree-spec? [s]
  (or (hmap-spec? s) (hvec-spec? s)))

(defn params-marker?
  "True iff annotation is the (Params <inner>) marker at the deftm-arg level.
  Params is the EXPLICIT signal that an arg is a structured tree to be
  flattened at compile time. Inside the marker, the structure uses ordinary
  HMap/HVec/Param/Frozen forms."
  [annotation]
  (and (sequential? annotation) (= 'Params (first annotation))))

(defn unwrap-params
  "Unwrap (Params <inner>) → <inner>. Other annotations pass through.
  Bare HMap/HVec annotations are already in the canonical inner form, so
  they pass through unchanged — Params is just a back-compat trigger."
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

(defn validate-tree-spec!
  "Validate that a tree spec uses only the closed, fully-specified constructs
  the Params machinery can flatten. Throws clojure.lang.ExceptionInfo with a
  pointed message on:

   - (HMap :optional {...})            — optional keys break canonical leaf order
   - (HMap :complete? false)           — open maps mean unknown leaves at compile time
   - (HMap :absent-keys ...)           — semantics ambiguous for closed tree
   - (HMap <unknown-opt> ...)          — typo-guard for unsupported HMap options
   - (HVec X) where X is not a vector  — HVec must list its element types
   - (HVec [T...] <extras>)            — HVec doesn't take extra options

  Recurses into every nested HMap/HVec. Leaves are unchecked (any non-tree
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
              ;; Leaf type — tree machinery doesn't inspect it further.
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
  order. Sub-trees recurse; non-tree types are leaves."
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
  "Recognize a single-step tree access. Returns [target-form step] or nil.
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
  (and (some? spec) (not (tree-spec? spec))))

(defn path->flat-idx
  "Given a tree spec and a leaf path, return the canonical flat-index of
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

  Validates that the resolved spec is a well-formed tree spec (HMap or HVec)
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
    (when-not (tree-spec? spec)
      (throw (ex-info (str "flat-view :spec must be an HMap or HVec form (or wrapped "
                           "in Params). Got: " (pr-str spec))
                      {:form form :spec spec})))
    (validate-tree-spec! spec)
    (when (zero? (count (flatten-spec spec)))
      (throw (ex-info "flat-view :spec has no leaves — empty tree?"
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
;; Cross-deftm tree call splicing
;;
;; When the body contains a call to a deftm whose params include
;; (Params ...), splice the tree leaves at the call site instead of
;; passing a structured value. Replaces the call with one to the callee's
;; flat-var. Eliminates runtime Map reconstruction at deftm-to-deftm
;; boundaries — the structured wrapper is only used at the outer program
;; boundary (calls from non-tree user code).
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
  "If form is a call to a deftm with Params-typed args, splice the tree
  leaves at the call site to call the flat-var directly. Returns the
  rewritten form, or nil if no splicing applies.

  If a tree arg is not safe to duplicate (e.g. a side-effecting expression
  or non-trivial computation), it is lifted into a temp let-binding so the
  evaluation happens exactly once — preserving the contract of the original
  flatten-on-call wrapper, which evaluates each tree arg once and reads
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
        (let [;; First pass: lift unsafe tree args to temp bindings.
              tmp-binds (volatile! [])
              normalized-args
              (mapv (fn [arg-form callee-arg-name]
                      (if (and (get treedefs callee-arg-name)
                               (not (safe-to-duplicate? arg-form)))
                        (let [g (gensym (str "__tree_arg__"
                                             (name callee-arg-name) "__"))]
                          (vswap! tmp-binds into [g arg-form])
                          g)
                        arg-form))
                    args
                    original-args)
              ;; Second pass: splice each tree arg into per-leaf path-access
              ;; expressions, leaving non-tree args untouched.
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
  "True iff form is a (raster.tree/walk! kind f [tree-args] & extras) call."
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
  "Expand (tree/walk! kind f [tree-args...] extras...) using compile-time
  spec info from the FIRST tree-arg. Emits a `do` form with one f call per
  matching leaf.

  Arity-checks `f` against the call shape (one arg per tree-arg + extras)
  when arity info is available. Catches user errors at macro time rather than
  surfacing as opaque IFn invoke failures at runtime."
  [form env]
  (let [[_ kind f-form tree-args-vec & extras] form]
    (when-not (vector? tree-args-vec)
      (throw (ex-info "tree/walk!: tree-args must be a vector"
                      {:got tree-args-vec})))
    (let [first-arg (first tree-args-vec)
          first-resolved (resolve-path first-arg env)
          spec (or (:spec first-resolved)
                   (throw (ex-info (str "tree/walk!: first tree-arg `"
                                        (pr-str first-arg)
                                        "` must resolve to a known tree spec")
                                   {:arg first-arg :env-keys (keys env)})))
          leaves (flatten-spec spec)
          matching (filter #(or (= kind :any) (= (:kind %) kind)) leaves)
          expected-arity (+ (count tree-args-vec) (count extras))
          arities (callee-arities f-form)]
      (when (and arities (not (contains? arities expected-arity)))
        (throw (ex-info
                (str "tree/walk!: f `" (pr-str f-form) "` has arity "
                     (if (= 1 (count arities))
                       (str (first arities))
                       (str "in " (sort arities)))
                     ", but walk! emits per-leaf calls with arity "
                     expected-arity " ("
                     (count tree-args-vec) " tree arg"
                     (when (not= 1 (count tree-args-vec)) "s")
                     " + " (count extras) " extra"
                     (when (not= 1 (count extras)) "s")
                     "). Make sure f's signature matches "
                     "(f leaf1 ... leafN extra1 ... extraK).")
                {:f f-form
                 :expected expected-arity
                 :callee-arities arities
                 :tree-args (count tree-args-vec)
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
                                          tree-args-vec)
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
                                      "` must resolve to a known tree spec")
                                 {:xs xs-form :env-keys (keys env)})))
        _ (when-not (hvec-spec? spec)
            (throw (ex-info "scan-vec: xs must be an HVec sub-tree"
                            {:xs xs-form :spec spec})))
        elems (hvec-elems spec)
        n (count elems)
        ;; Try to resolve f as a Params-typed defmodel. If its second positional
        ;; arg (the element) is a tree, splice each iteration directly to the
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
     would dangle (the tree-arg has been removed from the deftm signature),
     and emitting [name nil] confuses downstream passes that capture the
     binding into helper-method args. The binding name is recorded in env so
     subsequent leaf accesses resolve through it. If a sub-tree alias is used
     as a value (e.g. passed to a function), assert-no-dangling-tree-refs!
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
          (and resolved (tree-spec? (:spec resolved)))
          (recur (rest pairs) out (assoc e bsym resolved))

          :else
          (let [walked-val (walk-fn bval e)]
            (recur (rest pairs) (conj out bsym walked-val) e))))
      [out e])))

(defn rewrite-body
  "Rewrite body: replace static tree access with flat-arg references.

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

                  ;; Cross-deftm tree splicing: if the call target is a deftm
                  ;; whose original signature had Params args, replace the call
                  ;; with a direct call to its flat-var, with each tree arg
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
  "Walk a runtime tree value matching spec, returning a vector of leaves
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
  "Walk a runtime tree value matching spec; throw if two leaves share JVM
  identity (the same object placed at two tree positions, a.k.a. naked weight
  tying). Naked tying causes incorrect gradients: each position gets its own
  gradient buffer, so contributions from different paths don't accumulate.

  Resolution: use a single canonical tree position for the shared parameter
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
                (str "Tree leaves at " (pr-str a) " and " (pr-str b)
                     " share JVM identity (same object at two positions). "
                     "Naked weight sharing causes incorrect gradients — each "
                     "position gets its own gradient buffer; contributions "
                     "from different paths don't accumulate. Use one "
                     "canonical tree position and reference it from "
                     "multiple places in the model body, or wrap with `Tied` "
                     "(not yet implemented).")
                {:path-a a :path-b b}))))))

(defn assert-tree-shape!
  "Walk a runtime tree value matching spec; throw if the runtime shape doesn't
  match the spec — wrong HVec length, missing HMap keys, or extra HMap keys.

  Without this check, the wrapper silently extracts only the leaves the spec
  declares and ignores or fails on extras, so a wrong-shape tree at runtime
  produces wrong values (extras dropped) or cryptic IndexOutOfBoundsException
  (missing leaves) deep inside the compiled kernel.

  Primitives are exempt — JVM identity is not meaningful and they don't
  carry shape."
  [spec value]
  (letfn [(walk [s v path]
            (cond
              (hmap-spec? s)
              (let [keys-spec (set (keys (hmap-mandatory s)))
                    keys-val  (when (map? v) (set (keys v)))
                    missing   (clojure.set/difference keys-spec (or keys-val #{}))
                    extra     (clojure.set/difference (or keys-val #{}) keys-spec)]
                (when-not (map? v)
                  (throw (ex-info
                          (str "Tree shape mismatch at " (pr-str path)
                               ": spec declares an HMap but runtime value is "
                               (pr-str (class v)))
                          {:path path :spec s :value v :expected :map})))
                (when (seq missing)
                  (throw (ex-info
                          (str "Tree shape mismatch at " (pr-str path)
                               ": missing HMap keys " (pr-str missing))
                          {:path path :spec s :missing-keys missing})))
                (when (seq extra)
                  (throw (ex-info
                          (str "Tree shape mismatch at " (pr-str path)
                               ": runtime tree has extra HMap keys "
                               (pr-str extra) " not declared in the spec. "
                               "If you want these to flow through, declare "
                               "them in the model spec.")
                          {:path path :spec s :extra-keys extra})))
                (doseq [[k sub-spec] (hmap-mandatory s)]
                  (walk sub-spec (get v k) (conj path k))))

              (hvec-spec? s)
              (let [elems (hvec-elems s)
                    spec-n (count elems)
                    val-n  (when (sequential? v) (count v))]
                (when-not (sequential? v)
                  (throw (ex-info
                          (str "Tree shape mismatch at " (pr-str path)
                               ": spec declares an HVec but runtime value is "
                               (pr-str (class v)))
                          {:path path :spec s :value v :expected :sequential})))
                (when (not= spec-n val-n)
                  (throw (ex-info
                          (str "Tree shape mismatch at " (pr-str path)
                               ": spec declares HVec of length " spec-n
                               " but runtime value has length " val-n ". "
                               "Variable-length HVec is not supported — the "
                               "compiled kernel is monomorphic in shape.")
                          {:path path :spec s
                           :expected-length spec-n :actual-length val-n})))
                (doseq [[i sub-spec] (map-indexed vector elems)]
                  (walk sub-spec (nth v i) (conj path i))))

              ;; leaf: nothing to check structurally
              :else nil))]
    (walk spec value [])))

(defn unflatten-value
  "Given spec and a flat vector of leaves in canonical order, reconstruct the
  tree value. Inverse of flatten-value."
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

(defn tree-shape
  "A small structural fingerprint of a runtime value: the spec walked to leaves,
  with each leaf replaced by [:leaf <class>] for its runtime class. Useful as a
  cache key when validating that a runtime call's tree matches a treedef."
  [spec value]
  (let [leaves (flatten-value spec value)]
    (mapv (fn [v] [:leaf (some-> v class .getName)]) leaves)))

;; ----------------------------------------------------------------------
;; Type-directed flattening over nested Parameters value-classes
;;
;; Params can be authored as nested `defvalue … :implements Parameters`
;; value-classes — each module a small typed struct, composed via value-class
;; fields and (Array SubParams). Flattening is instance-driven: array fields
;; carry their depth only at runtime, so the canonical leaf order (the treedef)
;; is read off the params instance. Both flatten and unflatten walk fields in
;; declaration order, which is exactly the value-class constructor's parameter
;; order — so reconstruction is a single reflective ctor (also bypasses the
;; >20-param factory limit and works for any field types).
;; ----------------------------------------------------------------------

(def ^:private parameters-iface
  (delay (try (Class/forName "raster.core.Parameters") (catch Throwable _ nil))))

(defn parameters-class?
  "True iff Class c is a Parameters value-class (a nested-params module)."
  [c]
  (boolean (when-let [^Class p @parameters-iface]
             (and (class? c) (.isAssignableFrom p ^Class c)))))

(defn- vc-instance? [v]
  (and (some? v) (parameters-class? (class v))))

(defn- params-array? [v]
  (and (some? v)
       (let [^Class c (class v)]
         (and (.isArray c) (parameters-class? (.getComponentType c))))))

(defn- vc-fields
  "Instance fields of a value-class, in declaration (= ctor param) order."
  [^Class c]
  (->> (.getDeclaredFields c)
       (remove #(java.lang.reflect.Modifier/isStatic
                 (.getModifiers ^java.lang.reflect.Field %)))
       vec))

(defn- field-get [^java.lang.reflect.Field f obj]
  (.setAccessible f true)
  (.get f obj))

(defn- vc-construct
  "Reconstruct a value-class of Class c from field values in declaration order,
  via its sole declared constructor."
  [^Class c args]
  (let [^java.lang.reflect.Constructor ctor (first (.getDeclaredConstructors c))]
    (.setAccessible ctor true)
    (.newInstance ctor (object-array args))))

(defn flatten-params
  "Walk a nested Parameters value-class instance depth-first. Returns
  {:descs [{:path :type :kind}] :vals [leaf …]} in canonical (declaration)
  order — the type-directed analogue of flatten-spec + flatten-value.

  Recursion: a Parameters-typed field descends into the sub-struct
  (path += field keyword); an (Array Parameters) field descends per element
  (path += index); every other field (typically a primitive array) is a leaf.
  Leaves are :param by default (freezing is a path-level optimizer concern)."
  [inst]
  (let [descs (transient []) vals (transient [])]
    (letfn [(walk [v path]
              (cond
                (vc-instance? v)
                (doseq [^java.lang.reflect.Field f (vc-fields (class v))]
                  (walk (field-get f v) (conj path (keyword (.getName f)))))

                (params-array? v)
                (dotimes [i (java.lang.reflect.Array/getLength v)]
                  (walk (java.lang.reflect.Array/get v i) (conj path i)))

                :else
                (do (conj! descs {:path path :type (some-> v class) :kind :param})
                    (conj! vals v))))]
      (walk inst []))
    {:descs (persistent! descs) :vals (persistent! vals)}))

(defn params-treedef
  "Canonical leaf descriptors of a nested Parameters instance — the single source
  of leaf order for AD grad assembly and the optimizer."
  [inst]
  (:descs (flatten-params inst)))

;; --- value-class access rewrite (the .-field / aget analogue of the HMap
;; (:k m) -> leaf-sym pre-flatten). After sub-forward inlining + scan-vec
;; unroll, every param use is a concrete deep access chain; we rewrite each
;; leaf chain to its canonical leaf symbol and bind those in a prologue, so the
;; differentiable inner body is in terms of free leaf locals (exactly what
;; container-vg-runtime expects).

;; Structural block-array indexing uses clojure.core/aget (integer index into a
;; (Array Parameters) field); leaf element reads use raster.arrays/aget — only
;; the former is a value-class access step.
(def ^:private struct-aget-syms '#{aget clojure.core/aget})

(defn- lit-int
  "Unwrap an integer literal, including walker-emitted (long N) / (int N)."
  [x]
  (cond
    (integer? x) x
    (and (seq? x) (= 2 (count x)) ('#{long int clojure.core/long clojure.core/int} (first x))
         (integer? (second x)))
    (second x)))

(defn vc-extract-access-step
  "Recognize one value-class access step. Returns [target step] or nil:
  (.-field x) -> [x :field]; (clojure.core/aget x i-literal) -> [x i]."
  [form]
  (when (and (seq? form) (>= (count form) 2))
    (let [op (first form)]
      (cond
        (and (symbol? op) (= 2 (count form)) (.startsWith (name op) ".-"))
        [(second form) (keyword (subs (name op) 2))]

        (and (contains? struct-aget-syms op) (= 3 (count form)) (lit-int (nth form 2)))
        [(second form) (lit-int (nth form 2))]))))

(defn vc-resolve-path
  "Resolve form to [root path] when it is an access chain rooted at one of the
  known value-class param roots (a set of symbols); else nil. Resolution is
  greedy, but a (clojure.core/aget leaf j) element read past a leaf array needs
  no special handling: leaf paths are terminal — flatten-params never emits a
  leaf that is a prefix of another leaf — so an over-extended path can never
  equal a leaf path, and vc-rewrite-body simply descends and rewrites the inner
  leaf access (the index j is left intact as an element read).

  `aliases` maps a let-bound symbol to the [root path-prefix] of the sub-struct it
  was bound to (e.g. `b` ← `(clojure.core/aget (.-blocks p) 0)` gives
  b → [p [:blocks 0]]), so chains rooted at an intermediate struct local resolve."
  ([form roots] (vc-resolve-path form roots {}))
  ([form roots aliases]
   (cond
     (and (symbol? form) (contains? roots form)) [form []]
     (and (symbol? form) (contains? aliases form)) (get aliases form)
     :else (when-let [[target step] (vc-extract-access-step form)]
             (when-let [[root path] (vc-resolve-path target roots aliases)]
               [root (conj path step)])))))

(defn vc-access-expr-for-path
  "Inverse of vc-resolve-path: build the deep .-field / aget access chain for a
  leaf path under root."
  [root path]
  (reduce (fn [acc step]
            (cond
              (keyword? step) (list (symbol (str ".-" (name step))) acc)
              (integer? step) (list 'clojure.core/aget acc step)
              :else (throw (ex-info "Bad value-class path step" {:step step}))))
          root path))

(defn vc-prologue
  "Given a root param symbol and a treedef (descs from params-treedef), return a
  flat let*-style binding vector [leaf-sym access-expr ...] for every leaf."
  [root descs]
  (vec (mapcat (fn [{:keys [path]}]
                 [(path->sym root path) (vc-access-expr-for-path root path)])
               descs)))

(defn vc-rewrite-body
  "Replace every maximal value-class access chain rooted at a known param root
  that resolves to a leaf path with that path's canonical leaf symbol. Non-leaf
  (intermediate struct/array) chains are descended into, not replaced.

  Threads an alias environment through let/let* binding forms: a binding whose
  init resolves to a NON-leaf sub-struct path (e.g. `b ← (aget (.-blocks p) i)`)
  records `b → [root path]`, so later chains rooted at `b` resolve into the tree.
  This handles the natural per-block fold idiom (bind the block, access its
  fields) that both hand-written code and scan-vec/inline output produce."
  [body roots leaf-paths]
  (let [leaf-set (set (map vec leaf-paths))
        leaf? (fn [r] (and r (contains? leaf-set (vec (second r)))))]
    (letfn [(rw [form aliases]
              (let [resolved (vc-resolve-path form roots aliases)]
                (cond
                  ;; leaf access chain → canonical leaf symbol
                  (leaf? resolved)
                  (path->sym (first resolved) (second resolved))

                  ;; let/let* — process bindings left-to-right, accumulating
                  ;; sub-struct aliases, then rewrite the body under them
                  (and (seq? form) (form/binding-form? form))
                  (let [[lsym bvec & bodyforms] form
                        [binds aliases']
                        (reduce
                         (fn [[bs al] [s init]]
                           (let [init' (rw init al)
                                 ir    (vc-resolve-path init roots al)
                                 al'   (if (and ir (not (leaf? ir))) (assoc al s ir) al)]
                             [(conj bs s init') al']))
                         [[] aliases]
                         (partition 2 bvec))]
                    (apply list lsym binds (map #(rw % aliases') bodyforms)))

                  (seq? form)    (apply list (map #(rw % aliases) form))
                  (vector? form) (mapv #(rw % aliases) form)
                  (map? form)    (into (empty form)
                                       (map (fn [[k v]] [(rw k aliases) (rw v aliases)]) form))
                  :else form)))]
      (rw body {}))))

(defn- field-by-name [inst fname]
  (let [^java.lang.reflect.Field f (.getDeclaredField (class inst) fname)]
    (.setAccessible f true)
    (.get f inst)))

(defn nav-path
  "Navigate a nested Parameters instance along a path (field keywords + array
  indices) to the value at that path. (nav-path m [:blocks 0 :attn :W]) etc."
  [inst path]
  (reduce (fn [v step]
            (cond
              (keyword? step) (field-by-name v (name step))
              (integer? step) (java.lang.reflect.Array/get v step)
              :else (throw (ex-info "bad nav step" {:step step}))))
          inst path))

;; --- trace-time block-fold unroll ---
;; (vc-fold [acc blk] init blocks-expr body) folds `body` over each element of
;; the value-class array `blocks-expr`, threading `acc` from `init`. The block
;; count is a runtime property, so the fold is unrolled at trace time (when
;; value+grad-params / compile-aot see a concrete instance) into the flat
;; per-block let* idiom that vc-rewrite-body + grad-expr already handle — the
;; JAX "trace per shape" model.

(def ^:private vc-fold-syms '#{vc-fold raster.params/vc-fold})

(defn vc-fold-form? [f]
  (and (seq? f) (contains? vc-fold-syms (first f))))

(defn has-vc-fold? [body]
  (boolean (some vc-fold-form? (tree-seq seqable? seq body))))

(defn- subst-syms-tree [smap form]
  (walk/postwalk (fn [x] (if (symbol? x) (get smap x x) x)) form))

(defn unroll-vc-folds
  "Expand every (vc-fold [acc blk] init blocks-expr body) in `body` into a flat
  let* of N per-block applications, where N = (count-fn blocks-expr). acc/blk are
  alpha-renamed per iteration; bottom-up so nested folds unroll inside-out."
  [body count-fn]
  (walk/postwalk
   (fn [form]
     (if (vc-fold-form? form)
       (let [[_ [acc blk] init blocks fold-body] form
             n     (long (count-fn blocks))
             accs  (vec (repeatedly (inc n) #(gensym (str (name acc) "_"))))
             binds (into [(accs 0) init]
                         (mapcat (fn [i]
                                   (let [blki (gensym (str (name blk) "_"))]
                                     [blki (list 'clojure.core/aget blocks i)
                                      (accs (inc i))
                                      (subst-syms-tree {acc (accs i) blk blki} fold-body)]))
                                 (range n)))]
         (list 'let* binds (accs n)))
       form))
   body))

(defn unflatten-params
  "Rebuild a nested Parameters structure shaped like `template`, substituting
  leaves from `new-vals` in canonical order. Inverse of flatten-params: structure
  (sub-struct classes, array lengths) comes from the template, leaf payloads from
  new-vals. Returns a fresh top-level value-class."
  [template new-vals]
  (let [i (volatile! 0)]
    (letfn [(rebuild [v]
              (cond
                (vc-instance? v)
                (vc-construct (class v)
                              (mapv (fn [^java.lang.reflect.Field f]
                                      (rebuild (field-get f v)))
                                    (vc-fields (class v))))

                (params-array? v)
                (let [n (java.lang.reflect.Array/getLength v)
                      out (make-array (.getComponentType ^Class (class v)) n)]
                  (dotimes [k n]
                    (java.lang.reflect.Array/set
                     out k (rebuild (java.lang.reflect.Array/get v k))))
                  out)

                :else
                (let [x (nth new-vals @i)] (vswap! i inc) x)))]
      (rebuild template))))

;; ----------------------------------------------------------------------
;; Top-level: prepare deftm args
;; ----------------------------------------------------------------------

(defn tree-arg?
  "True iff annotation describes a structured tree to be flattened at compile
  time. Recognized forms:
    (HMap :mandatory {...})  /  (HMap {...})
    (HVec [T0 T1 ...])
    (Params <inner>)         — back-compat alias; <inner> must be HMap/HVec

  Other annotations (Array, primitives, value classes) pass through as
  ordinary typed deftm args."
  [annotation]
  (or (params-marker? annotation)
      (hmap-spec? annotation)
      (hvec-spec? annotation)))

(defn- collect-symbols
  "Return the set of all symbols that appear anywhere in form."
  [form]
  (let [acc (volatile! #{})]
    (clojure.walk/postwalk
     (fn [x] (when (symbol? x) (vswap! acc conj x)) x)
     form)
    @acc))

(defn- assert-no-dangling-tree-refs!
  "Tree args are removed from the deftm signature by the pre-flatten pass.
  Any reference to a tree-arg symbol (or a sub-tree alias bound only to a
  path) that survives the rewrite indicates the user used the value
  non-statically — e.g. passed it to a function or stored it. We can't compile
  that under B1; surface a clear error rather than producing dangling code."
  [rewritten-body tree-arg-syms]
  (let [present (collect-symbols rewritten-body)
        leaks   (filter present tree-arg-syms)]
    (when (seq leaks)
      (throw (ex-info
              (str "Tree-typed deftm arg(s) " (vec leaks)
                   " referenced as a value in the body. The pre-flatten pass "
                   "only handles static leaf access via (:k m) / (get m :k) / "
                   "(nth v i). Passing a tree (or sub-tree) as a value to "
                   "another function is not supported in this version.")
              {:leaked-symbols (vec leaks)})))))

(defn prepare-deftm-shape
  "Cheap, body-independent half of prepare-deftm. Walks the spec only
  (no body rewrite) to compute:

    :params      flat positional param symbols (canonical leaf order)
    :annotations flat positional annotations matching :params
    :treedefs    {root-sym {:spec ... :leaves [{:path :sym :kind :type}]}}
    :env         {root-sym {:root ... :path [] :spec ...}} for use by
                 prepare-deftm-body to rewrite (:k root) accesses
    :tree-arg-syms vector of tree-typed root symbols (for dangling-ref check)

  Validates spec well-formedness. Body work (rewrite-body) is split out
  into prepare-deftm-body so the deftm macro can defer it to lazy JIT
  time while still emitting the --flat deftm with the right arity at
  macro time."
  [params annotations]
  (let [param-inner (fn [ann] (when (tree-arg? ann) (unwrap-params ann)))
        _ (doseq [[p ann] (map vector params annotations)
                  :when (tree-arg? ann)]
            (try (validate-tree-spec! (unwrap-params ann))
                 (catch clojure.lang.ExceptionInfo e
                   (throw (ex-info (str "Invalid Params spec for arg `" p "`: "
                                        (.getMessage e))
                                   (assoc (ex-data e) :param p)
                                   e)))))
        init-env (into {}
                       (keep (fn [[p ann]]
                               (when-let [inner (param-inner ann)]
                                 [p {:root p :path [] :spec inner}]))
                             (map vector params annotations)))
        canonical-leaves
        (into {}
              (keep (fn [[p ann]]
                      (when-let [inner (param-inner ann)]
                        [p (mapv (fn [{:keys [path] :as leaf}]
                                   (assoc leaf :sym (path->sym p path)))
                                 (flatten-spec inner))]))
                    (map vector params annotations)))
        tree-arg-syms (vec (keep (fn [[p ann]] (when (tree-arg? ann) p))
                                 (map vector params annotations)))
        [new-params new-anns]
        (reduce (fn [[ps as] [p ann]]
                  (if (tree-arg? ann)
                    (let [leaves (canonical-leaves p)]
                      [(into ps (map :sym) leaves)
                       (into as (map (fn [{:keys [type kind]}]
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
    {:params        new-params
     :annotations   new-anns
     :treedefs      treedefs
     :env           init-env
     :tree-arg-syms tree-arg-syms}))

(defn prepare-deftm-body
  "Body half of prepare-deftm. Given env (from prepare-deftm-shape) and
  the original deftm body, run the (:k root) → root__k rewrite and return
  the rewritten body. Throws if any tree-arg root symbol leaks through
  the rewrite (i.e. used as a value rather than a path).

  Pure function — safe to call at macro time, lazy JIT time, or any time
  in between. The defmodel macro defers this to lazy JIT so the walker
  pre-flatten happens alongside the type walker, not at source-read time."
  [body env tree-arg-syms]
  (let [{body' :body} (rewrite-body body env)]
    (assert-no-dangling-tree-refs! body' tree-arg-syms)
    body'))

(defn prepare-deftm
  "Convenience: prepare-deftm-shape + prepare-deftm-body in one call.
  Kept for callers that don't need to defer body rewriting.

  Returns {:params new-params, :annotations new-anns, :body new-body,
           :treedefs {root-sym {:spec ... :leaves [{:path :sym :kind :type}]}}}"
  [params annotations body]
  (let [{:keys [params annotations treedefs env tree-arg-syms]}
        (prepare-deftm-shape params annotations)
        new-body (prepare-deftm-body body env tree-arg-syms)]
    {:params      params
     :annotations annotations
     :body        new-body
     :treedefs    treedefs}))
