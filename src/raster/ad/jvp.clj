(ns raster.ad.jvp
  "Forward-mode AD via source transform — the JVP fold (framework §13,
  Option A, phase A2).

  A SIBLING of the reverse engine's forward-pass: one pure reduce over the
  ANF bindings of the shared-prepped body (raster.ad.reverse/ad-prepare —
  the SAME lower-composites → materialize → hoist pipeline the reverse path
  uses), threading a tangent environment {primal-sym → tangent-sym}. Each
  active binding gets a PAIRED tangent binding emitted from the op's
  :jvp-fn — the forward contraction DERIVED from the one rule source
  (raster.ad.templates :structure / :grads-fn facets, §13 A1). No tape, no
  reversal: structurally simpler than reverse.

  Separate ns by design: reverse.clj is already 3000 lines and slated for
  decomposition; jvp only consumes reverse's shared pieces (ad-prepare,
  resolve-deftm-var, grad-acc as the ⊕ kernel).

  Not yet supported (fail loud, forward loop rules are follow-up work):
  par/map!, par/reduce, dotimes, loop — the differentiable loop forms have
  reverse orchestrators but no forward fold yet."
  (:require [raster.ad.reverse :as rev]
            [raster.ad.templates :as tmpl]
            [raster.ad.tangent :as tangent]
            [raster.ad.reverse.normalize :as anf]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.op-descriptor :as opdesc]
            [raster.core :as rcore]
            [clojure.string :as string]
            [clojure.walk :as walk]
            ;; Kernel namespaces the derived rules emit calls into:
            [raster.par]      ;; dot-product — the scalar-loss frule contraction
            [raster.arrays])) ;; zeros-like/alength — typed zero tangents

;; ================================================================
;; Gensym
;; ================================================================

(def ^:private jvp-counter (atom 0))

(defn- jvp-gensym
  "Unique symbol for JVP-generated code. Optional type-tag stamps
  :raster.type/tag so downstream dispatch/devirtualization reads types
  without re-inference (the tag-carrying-emission hazard, §13 A2)."
  ([prefix] (symbol (str prefix "__jvp" (swap! jvp-counter inc))))
  ([prefix type-tag]
   (cond-> (symbol (str prefix "__jvp" (swap! jvp-counter inc)))
     (some? type-tag) (vary-meta assoc :raster.type/tag type-tag))))

;; ================================================================
;; The jvp-fold
;; ================================================================

(defn- extract-let-parts
  "Bindings + body of a prepared walked form (bare forms → no bindings)."
  [form]
  (if (and (seq? form) (symbol? (first form)) (#{'let 'let*} (first form)))
    [(second form) (nnext form)]
    [[] [form]]))

(defn- any-active?
  "Does `form` reference any symbol with a tangent in tenv? (Over-approximates
  by scanning all symbols incl. op heads — heads are qualified/mangled and
  never tenv keys, so this is safe.)"
  [tenv form]
  (boolean (some #(and (symbol? %) (contains? tenv %))
                 (tree-seq coll? seq (if (seq? form) (vec form) form)))))

(defn- branch-tangent-zero
  "Typed zero tangent for the INACTIVE branch of an if (tangent protocol):
  tagged branch syms get a static typed zero; untagged fall back to the
  runtime zero-like on the primal (dynamically-typed Π — cannot mis-shape)."
  [branch]
  (let [btag (and (symbol? branch) (:raster.type/tag (meta branch)))]
    (if (some? btag)
      (tangent/zero-expr btag (list 'raster.arrays/alength branch))
      (list 'raster.ad.tangent/zero-like branch))))

(def ^:private shape-read-heads
  "Shape/length reads: integer-valued, no tangent space — an active array
  reaching them is a shape REFERENCE, not a differentiable dependence."
  '#{raster.arrays/alength clojure.core/alength alength})

(def ^:private unsupported-forward-heads
  "Differentiable-in-reverse forms with NO forward fold yet (follow-up):
  fail loud when they carry an active value, never silently drop a tangent."
  '#{loop loop* dotimes raster.par/map! raster.par/reduce fn* do case case*
     try letfn letfn* new monitor-enter monitor-exit})

(defn- fold-call
  "Emit the paired tangent bindings for one active :call binding.
  Returns [tenv' extra-bindings]."
  [tenv sym init tag]
  (let [head (first init)
        invk? (= '.invk head)
        args (if invk? (vec (nnext init)) (vec (rest init)))
        ;; same op-recovery order as the reverse engine's ad-record :call
        op (if invk?
             (or (:raster.op/original (meta init)) (:op (meta init)) (second init))
             head)
        tangent-args (mapv #(when (symbol? %) (get tenv %)) args)]
    (if-let [jf (tmpl/op-jvp-fn op)]
      (let [ctx {:bindings [] :gensym-fn jvp-gensym}
            [ctx' t-sym] (jf ctx args tangent-args sym jvp-gensym)
            ;; tangent carries exactly its primal's tag (tangent protocol)
            t-use (cond-> t-sym (some? tag) (vary-meta assoc :raster.type/tag tag))]
        [(assoc tenv sym t-use) (:bindings ctx')])
      (let [[template canonical] (or (tmpl/resolve-template op) [nil op])]
        (throw (ex-info
                (if template
                  (str "jvp: op `" canonical "` has a reverse template but no "
                       ":jvp-fn / :structure facet — hand frule pending (§13 A3). "
                       "Tag its Jacobian structure in raster.ad.templates/op-structures "
                       "or register an explicit :jvp-fn.")
                  (str "jvp: no AD rule for op `" op "` (bound to `" sym
                       "`) with active tangent inputs — un-templated deftm calls "
                       "should have been inlined by ad-prepare."))
                {:op op :canonical canonical :sym sym
                 :active-args (filterv identity tangent-args)}))))))

(defn- jvp-fold
  "A2: ONE pure reduce over the normalized ANF bindings threading
  {:tenv (tangent env, sym → tangent-sym), :bindings (flat primal+tangent)}.
  Per binding: inactive → passthrough; alias → tangent alias; templated call
  → paired tangent bindings via the op's (derived) :jvp-fn; templated-but-no-
  jvp-fn or un-templated-active → FAIL LOUD; `if` → branch-selected tangent
  with a typed zero for the inactive branch; loops/par forms → FAIL LOUD."
  [norm-bindings param-tangents]
  (reduce
   (fn [{:keys [tenv bindings]} [sym init]]
     (let [bindings (conj bindings sym init)
           tag (:raster.type/tag (meta sym))
           done (fn [tenv' extra] {:tenv tenv' :bindings (into bindings extra)})]
       (cond
         ;; literal / constant — no tangent
         (and (anf/trivial-expr? init) (not (symbol? init)))
         (done tenv [])

         ;; alias — tangent alias
         (symbol? init)
         (done (if-let [t (get tenv init)] (assoc tenv sym t) tenv) [])

         (seq? init)
         (let [head (first init)]
           (cond
             ;; if: tangent = (if test Δthen Δelse), typed zero on the
             ;; inactive branch (tangent protocol).
             (= 'if head)
             (let [[_ test then else] init
                   t-then (and (symbol? then) (get tenv then))
                   t-else (and (symbol? else) (get tenv else))]
               (if (or t-then t-else)
                 (let [dt (jvp-gensym (str "dt_" (name sym)) tag)]
                   (done (assoc tenv sym dt)
                         [dt (list 'if test
                                   (or t-then (branch-tangent-zero then))
                                   (or t-else (branch-tangent-zero else)))]))
                 (done tenv [])))

             ;; loops / par forms / other control flow: no forward rules yet
             (contains? unsupported-forward-heads head)
             (if (any-active? tenv init)
               (throw (ex-info
                       (str "jvp: forward-mode rules for `" head "` (bound to `"
                            sym "`) are not implemented yet — forward loop/SOAC "
                            "rules are follow-up work (reverse mode supports "
                            "these; use vjp/value+grad).")
                       {:form-head head :sym sym}))
               (done tenv []))

             ;; call (incl. devirtualized .invk)
             :else
             (let [op-sym (if (= '.invk head)
                            (or (:raster.op/original (meta init))
                                (:op (meta init)) (second init))
                            head)]
               (cond
                 ;; Pure allocations (zeros-like / alloc-like / *-array …):
                 ;; a fresh zero/uninitialized buffer with NO differentiable
                 ;; dependence on any input — active syms reach them only as
                 ;; shape/dtype REFERENCES (e.g. the gradient program's
                 ;; (zeros-like tgt (alength tgt)) typed zeros). No tangent.
                 ;; aclone is a COPY, deliberately NOT covered — it carries a
                 ;; :linear structure tag (tangent = aclone of the tangent).
                 (and (symbol? op-sym)
                      (or (contains? shape-read-heads op-sym)
                          (and (opdesc/alloc-op? op-sym)
                               (not (string/starts-with? (name op-sym) "aclone")))))
                 (done tenv [])

                 (some #(and (symbol? %) (contains? tenv %))
                       (if (= '.invk head) (nnext init) (rest init)))
                 (let [[tenv' extra] (fold-call tenv sym init tag)]
                   (done tenv' extra))

                 :else (done tenv [])))))

         :else (done tenv []))))
   {:tenv param-tangents :bindings []}
   (partition 2 norm-bindings)))

;; ================================================================
;; Entry point
;; ================================================================

(defn- make-runtime-jvp-fn
  "Eval the transformed body into an IFn — the make-runtime-value+grad-fn
  pattern: (& args) + indexed unpack past Clojure's 20-positional limit, and
  clojure.core/alength → raster.arrays/alength so untyped INTERMEDIATE arrays
  don't reflect ambiguously."
  [form params]
  (let [body (walk/postwalk-replace {'clojure.core/alength 'raster.arrays/alength}
                                    form)]
    (if (<= (count params) 20)
      (eval (list 'fn (vec params) body))
      (let [args-sym (gensym "jvp_args__")
            unpack (vec (mapcat (fn [p i] [p (list 'clojure.core/nth args-sym i)])
                                params (range)))]
        (eval (list 'fn ['& args-sym] (list 'let unpack body)))))))

(defn ^clojure.lang.IFn jvp
  "Forward-mode value-and-directional-derivative for a deftm var.

  (jvp #'f) → (fn [p1 … pN Δp_a … Δp_k] -> [primal tangent])

  where Δp_a … Δp_k are tangent arguments for the DIFFERENTIABLE params only
  (⊥ slots — Long/index/etc. — have no tangent slot), in param order. Pass
  typed zero arrays/0.0 for unseeded directions (the JAX jvp contract). The
  tangent output is J·v — exact, one forward sweep, no tape.

  The transformed body/params are exposed deftm-style as metadata
  (:raster.core/deftm-walked-body / -params / -tags) so compile-aot can
  consume the transform later, mirroring value+grad."
  [f-var]
  (let [resolved (rev/resolve-deftm-var f-var)
        m (meta resolved)
        params (or (:raster.core/deftm-params m)
                   (throw (ex-info "jvp requires a deftm var" {:var f-var})))
        walked-body (or (rcore/ensure-walked-body! resolved)
                        (throw (ex-info "No walked body on var" {:var f-var})))
        tags (or (:raster.core/deftm-tags m) (vec (repeat (count params) 'double)))
        all-params (vec (map-indexed
                         (fn [i p]
                           (let [tag (nth tags i nil)
                                 base (if (symbol? p) p (symbol (name p)))]
                             (if tag
                               (with-meta base {:raster.type/tag tag})
                               (with-meta base nil))))
                         params))
        ;; Only differentiable-tag params get tangent slots (⊥ params carry
        ;; no tangent space — same seeding rule as build-grad-walked-body).
        diff-params (vec (keep-indexed
                          (fn [i p]
                            (when (tangent/differentiable? (nth tags i nil)) p))
                          all-params))
        _ (when (empty? diff-params)
            (throw (ex-info (str "jvp: no differentiable params on " f-var
                                 " — every param tag is ⊥ (no tangent space)")
                            {:var f-var :tags tags})))
        ;; Shared pre-AD prep (identical to the reverse path).
        prepared (rev/ad-prepare (first walked-body))
        [bindings body-exprs] (extract-let-parts prepared)
        [norm-bindings body-sym] (anf/normalize-for-ad bindings body-exprs jvp-gensym)
        ;; Tangent params: one per differentiable param, tagged like its primal.
        tangent-params (mapv (fn [p] (with-meta (symbol (str "d" (name p) "__jt"))
                                       (meta p)))
                             diff-params)
        {:keys [tenv bindings]} (jvp-fold norm-bindings
                                          (zipmap diff-params tangent-params))
        tangent-out (or (get tenv body-sym)
                        ;; output independent of every seeded input → typed 0̄
                        (branch-tangent-zero body-sym))
        jvp-form (list 'let* (vec bindings) [body-sym tangent-out])
        fn-params (into all-params tangent-params)
        source-ns (or (:ns m) *ns*)
        qualified (inf/qualify-body-symbols jvp-form source-ns (set fn-params))
        runtime-fn (make-runtime-jvp-fn qualified fn-params)
        out-tags (into (vec (take (count params) (concat tags (repeat nil))))
                       (mapv #(:raster.type/tag (meta %)) tangent-params))]
    (with-meta (fn [& args] (apply runtime-fn args))
      {::jvp true
       :raster.core/deftm true
       :raster.core/deftm-walked-body [qualified]
       :raster.core/deftm-params fn-params
       :raster.core/deftm-tags out-tags})))

;; ================================================================
;; HVP — forward-over-reverse (§13 A4, Pearlmutter 1994)
;; ================================================================

(defn ^clojure.lang.IFn hvp
  "Exact Hessian-vector product for a SCALAR-valued deftm var via
  forward-over-reverse: the jvp-fold applied to the GRADIENT PROGRAM.

  Mechanics: reify-pullback (the shared reverse engine) yields the flat
  forward bindings and the reverse bindings that compute [g_a … g_k]; the
  gradient program is their concatenation assembled with the unit adjoint
  (dy__rad = 1.0) — exactly compile-hvp-fn's combined form WITHOUT the
  v·grad dot. The jvp-fold then runs over THAT program with tangent seeds
  v on the differentiable params: the tangent of each gradient g_i is
  (H·v)_i. The pullback's kernels are linear/bilinear in dy (§13 classes
  b/c — including the backward kernels' own :structure tags), so no
  second-order rrules are needed.

  (hvp #'f) → (fn [p1 … pN Δp_a … Δp_k] -> [grads hv])

  where Δp_a … Δp_k are tangent (v) slots for the DIFFERENTIABLE params
  only (⊥ slots — Long dims etc. — have no tangent slot), in param order;
  `grads` is the gradient vector [∂f/∂p_a …] and `hv` its directional
  tangent [(H·v)_a …], both aligned with the differentiable params."
  [f-var]
  (let [resolved (rev/resolve-deftm-var f-var)
        m (meta resolved)
        params (or (:raster.core/deftm-params m)
                   (throw (ex-info "hvp requires a deftm var" {:var f-var})))
        walked-body (or (rcore/ensure-walked-body! resolved)
                        (throw (ex-info "No walked body on var" {:var f-var})))
        tags (or (:raster.core/deftm-tags m) (vec (repeat (count params) 'double)))
        all-params (vec (map-indexed
                         (fn [i p]
                           (let [tag (nth tags i nil)
                                 base (if (symbol? p) p (symbol (name p)))]
                             (if tag
                               (with-meta base {:raster.type/tag tag})
                               (with-meta base nil))))
                         params))
        diff-params (vec (keep-indexed
                          (fn [i p]
                            (when (tangent/differentiable? (nth tags i nil)) p))
                          all-params))
        _ (when (empty? diff-params)
            (throw (ex-info (str "hvp: no differentiable params on " f-var
                                 " — every param tag is ⊥ (no tangent space)")
                            {:var f-var :tags tags})))
        ;; The gradient program: shared prep → reified pullback → flat
        ;; fwd + (dy = 1.0) + rev bindings, outputs [g_a … g_k].
        prepared (rev/ad-prepare (first walked-body))
        {:keys [fwd-bindings result-sym body-sym pullback-form]}
        (rev/reify-pullback prepared diff-params)
        [_ rev-bindings grad-vec] pullback-form
        grad-slots (vec grad-vec)
        ;; ANF-normalize: the reverse engine's grad-acc chains nest calls in
        ;; argument position; the fold's tangent lookup is symbol-only, so
        ;; un-lifted nested calls would silently DROP tangent contributions.
        grad-bindings (anf/anf-normalize-bindings
                       (vec (concat fwd-bindings
                                    [result-sym body-sym]
                                    ['dy__rad 1.0]
                                    rev-bindings))
                       jvp-gensym)
        ;; Seed tangents: one v per differentiable param, tagged like it.
        tangent-params (mapv (fn [p] (with-meta (symbol (str "d" (name p) "__jt"))
                                       (meta p)))
                             diff-params)
        {:keys [tenv bindings]} (jvp-fold grad-bindings
                                          (zipmap diff-params tangent-params))
        ;; Tangent of each gradient slot = the H·v row. Non-symbol slots
        ;; (nil / materialized typed-zero exprs for rule-frozen params) have
        ;; constant gradients → nil tangent (dynamic 0̄).
        hv-outs (mapv (fn [g]
                        (when (symbol? g)
                          (or (get tenv g) (branch-tangent-zero g))))
                      grad-slots)
        hvp-form (list 'let* (vec bindings)
                       (list 'clojure.core/vector grad-slots hv-outs))
        fn-params (into all-params tangent-params)
        source-ns (or (:ns m) *ns*)
        qualified (inf/qualify-body-symbols hvp-form source-ns (set fn-params))
        runtime-fn (make-runtime-jvp-fn qualified fn-params)]
    (with-meta (fn [& args] (apply runtime-fn args))
      {::hvp true
       :raster.core/deftm true
       :raster.core/deftm-walked-body [qualified]
       :raster.core/deftm-params fn-params
       :raster.core/deftm-tags (into (vec (take (count params)
                                                (concat tags (repeat nil))))
                                     (mapv #(:raster.type/tag (meta %))
                                           tangent-params))})))
