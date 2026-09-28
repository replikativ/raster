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

  Double-carry par/scan and par/reduce and pure maps into fresh buffers have
  forward rules. Other loop/SOAC forms still fail loud when active."
  (:require [raster.ad.reverse :as rev]
            [raster.ad.templates :as tmpl]
            [raster.ad.tangent :as tangent]
            [raster.ad.reverse.normalize :as anf]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.op-descriptor :as opdesc]
            [raster.compiler.passes.scalar.effects :as effects]
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

(def ^:private opaque-forward-scopes
  "These bind local values or carry closure state. A free-variable scan of their
   enclosing ANF binding cannot prove that a tangent is inactive: reverse AD's
   loop tape may contain pullback closures that capture an active parameter."
  '#{loop loop* dotimes raster.par/map! raster.par/reduce raster.par/scan})

(def ^:private unsupported-forward-heads
  "Differentiable-in-reverse forms with NO forward fold yet (follow-up):
  fail loud when they carry an active value, never silently drop a tangent."
  (into opaque-forward-scopes
        '#{fn* do case case* try letfn letfn* new monitor-enter monitor-exit}))

(defn- reject-unlinearized-hvp-scopes!
  "The gradient program can hide active values inside loop-tape pullback
   closures. Until those scopes have forward rules, HVP cannot establish that
   a zero tangent is valid merely because the outer binding appears inactive."
  [bindings]
  (when-let [form (some (fn [form]
                         (when (and (seq? form)
                                    (contains? opaque-forward-scopes (first form)))
                           form))
                       (tree-seq coll? seq bindings))]
    (throw (ex-info
            (str "hvp: forward linearization of `" (first form)
                 "` is not implemented; its pullback may capture active values")
            {:reason :hvp-unlinearized-scope
             :form-head (first form)
             :form form}))))

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

(declare jvp-fold)

(defn- fresh-map-output?
  "A map! tangent may shadow only a private, zero-initialized allocation.
  Reject aliases and buffers with earlier uses: their prior contents could
  contribute to the result even when the map writes fewer than all elements."
  [prior-bindings out]
  (let [pairs (vec (partition 2 prior-bindings))
        alloc-idx (first (keep-indexed
                          (fn [i [s init]]
                            (when (= s out) i)) pairs))]
    (and (some? alloc-idx)
         (let [[_ init] (nth pairs alloc-idx)]
           (and (seq? init)
                (symbol? (first init))
                (= :zero (opdesc/allocation-initialization (first init)))
                (contains? '#{doubles floats (Array double) (Array float)}
                           (:raster.type/tag (meta out)))))
         (not-any? (fn [[_ init]]
                     (some #{out} (tree-seq coll? seq init)))
                   (subvec pairs (inc alloc-idx))))))

(defn- pure-map-step?
  "Ask the shared effect analysis about the source operation, not an opaque
  devirtualized call. AD's existing template resolver is the authority for
  that identity; an unknown .invk stays unknown and therefore declines."
  [body]
  (let [semantic-body
        (walk/postwalk
         (fn [form]
           (if (and (seq? form) (= '.invk (first form)))
             (if-let [[_ canonical] (tmpl/resolve-template (second form))]
               (with-meta (cons canonical (nnext form)) (meta form))
               form)
             form))
         body)]
    (effects/removable-expr? semantic-body)))

(defn- fold-fresh-map!
  "Linearize a pure indexed map into a freshly allocated buffer. The primal
  map has already run; the tangent map evaluates the same scalar step with
  tangent inputs, writing to an independent zero-initialized shadow."
  [tenv sym map-form tag]
  (let [[_ out idx bound & tail] map-form
        [cast body] (when (= 2 (count tail)) tail)]
    (when-not (and (= 2 (count tail)) (symbol? out) (symbol? idx)
                   (contains? '#{nil float double clojure.core/float
                                 clojure.core/double} cast)
                   (pure-map-step? body)
                   (not (some #{out} (tree-seq coll? seq body))))
      (throw (ex-info "jvp: map! needs a floating cast, pure body and non-self-reading output"
                      {:reason :jvp-map-unverified-step :form map-form})))
    (let [[step-bindings step-result]
          (anf/normalize-for-ad [] [body] jvp-gensym)
          {:keys [bindings] step-tenv :tenv}
          (jvp-fold step-bindings (dissoc tenv idx))
          dstep (or (get step-tenv step-result)
                    (branch-tangent-zero step-result))
          dout (jvp-gensym (str "d_" (name out))
                            (or (:raster.type/tag (meta out)) tag))
          map-result (jvp-gensym (str "d_" (name sym)) tag)
          tangent-body (list 'let* (vec bindings) dstep)]
      [dout map-result
       [dout (list 'raster.arrays/zeros-like out
                   (list 'raster.arrays/alength out))
        map-result (list 'raster.par/map! dout idx bound cast tangent-body)]])))

(defn- fold-double-scan
  "Linearize an inclusive double scan using its output as the primal carry
  tape. The tangent is another ordered scan; at step i, the prior primal
  carry is init (i=0) or primal-out[i-1]. Narrowing casts require a separate
  unrounded carry tape and are deliberately not reconstructed this way."
  [tenv sym scan-form tag]
  (let [[_ out acc init idx bound cast body] scan-form]
    (when-not (and (contains? '#{nil double}
                             (:raster.type/tag (meta acc)))
                   (or (contains? '#{double clojure.core/double} cast)
                       (and (nil? cast)
                            (= 'double (:raster.type/tag (meta acc)))))
                   (contains? '#{doubles (Array double)}
                              (:raster.type/tag (meta out))))
      (throw (ex-info "jvp: scan carry reconstruction requires double storage and cast"
                      {:reason :jvp-scan-carry-precision
                       :cast cast
                       :carry-tag (:raster.type/tag (meta acc))
                       :out-tag (:raster.type/tag (meta out))
                       :form scan-form})))
    (when (and (seq? init) (any-active? tenv init))
      (throw (ex-info "jvp: active scan init must be bound before the scan"
                      {:reason :jvp-scan-active-init :init init})))
    (let [dout (jvp-gensym (str "d_" (name sym) "_out") tag)
          dacc (jvp-gensym (str "d_" (name acc) "_carry") 'double)
          step (jvp-gensym "scan_step" 'double)
          [body-bindings body-exprs] (extract-let-parts body)
          body-result (if (= 1 (count body-exprs))
                        (first body-exprs)
                        (cons 'do body-exprs))
          [step-bindings step-result]
          (anf/normalize-for-ad
           (vec (concat body-bindings [step body-result]))
           [step] jvp-gensym)
          {:keys [bindings] step-tenv :tenv}
          (jvp-fold step-bindings (assoc tenv acc dacc))
          dstep (or (get step-tenv step-result)
                    (branch-tangent-zero step-result))
          prior (list 'if (list 'clojure.core/zero? idx)
                      init
                      (list 'clojure.core/aget sym
                            (list 'clojure.core/dec idx)))
          step-body (list 'let* (vec (concat [acc prior] bindings)) dstep)
          dinit (if (symbol? init) (or (get tenv init) 0.0) 0.0)
          alloc (list 'raster.arrays/zeros-like sym
                      (list 'raster.arrays/alength sym))
          scan (list 'raster.par/scan dout dacc dinit idx bound cast step-body)
          result (jvp-gensym (str "d_" (name sym)) tag)]
      [result [dout alloc result scan]])))

(defn- fold-double-reduce
  "Run a reduction's primal and tangent carries together in one ordered loop.
  Each scalar step is ANF-normalized once, so effects and reads are not replayed.
  This is a forward rule for a double carry, not a reassociation of the fold."
  [tenv sym reduce-form tag]
  (let [[_ acc init idx bound body] reduce-form
        init-tag (or (:raster.type/tag (meta init))
                     (some (fn [p]
                             (when (= p init) (:raster.type/tag (meta p))))
                           (keys tenv))
                     (when (instance? Double init) 'double))]
    (when-not (and (= 'double tag)
                   (= 'double init-tag)
                   (= 'double (or (:raster.type/tag (meta body))
                                  (inf/infer-rewritten-tag body))))
      (throw (ex-info "jvp: reduction requires a proven double carry"
                      {:reason :jvp-reduce-carry-precision
                       :result-tag tag :init-tag init-tag
                       :body-tag (inf/infer-rewritten-tag body)
                       :form reduce-form})))
    (when (and (seq? init) (any-active? tenv init))
      (throw (ex-info "jvp: active reduction init must be bound before the fold"
                      {:reason :jvp-reduce-active-init :init init})))
    (let [dacc (jvp-gensym (str "d_" (name acc) "_carry") 'double)
          step (jvp-gensym "reduce_step" 'double)
          [body-bindings body-exprs] (extract-let-parts body)
          body-result (if (= 1 (count body-exprs))
                        (first body-exprs)
                        (cons 'do body-exprs))
          [step-bindings step-result]
          (anf/normalize-for-ad
           (vec (concat body-bindings [step body-result]))
           [step] jvp-gensym)
          {:keys [bindings] step-tenv :tenv}
          (jvp-fold step-bindings (assoc tenv acc dacc))
          dstep (or (get step-tenv step-result)
                    (branch-tangent-zero step-result))
          dinit (if (symbol? init) (or (get tenv init) 0.0) 0.0)
          loop-body (list 'let* (vec bindings)
                          (list 'recur (list 'clojure.core/inc idx)
                                step-result dstep))
          count-sym (jvp-gensym "reduce_count" 'int)
          pair (jvp-gensym "reduce_pair")
          result (jvp-gensym (str "d_" (name sym)) tag)
          pair-expr (list 'loop* [idx 0 acc init dacc dinit]
                          (list 'if (list 'clojure.core/< idx count-sym)
                                loop-body
                                [acc dacc]))]
      [result [count-sym (list 'clojure.core/int bound)
               pair pair-expr
               sym (list 'nth pair 0)
               result (list 'nth pair 1)]])))

(defn- jvp-fold
  "A2: ONE pure reduce over the normalized ANF bindings threading
  {:tenv (tangent env, sym → tangent-sym), :bindings (flat primal+tangent)}.
  Per binding: inactive → passthrough; alias → tangent alias; templated call
  → paired tangent bindings via the op's (derived) :jvp-fn; templated-but-no-
  jvp-fn or un-templated-active → FAIL LOUD; `if` → branch-selected tangent
  with a typed zero for the inactive branch; double scans → a tangent scan;
  double reductions → paired primal/tangent carries;
  pure fresh-buffer maps → an independent tangent map;
  other loops/par forms → FAIL LOUD."
  [norm-bindings param-tangents]
  (reduce
   (fn [{:keys [tenv bindings]} [sym init]]
     (let [prior-bindings bindings
           bindings (conj bindings sym init)
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

             ;; A scan's output carries the prior primal states. Use it as
             ;; the tape for a second scan over the linearized recurrence.
             (= 'raster.par/scan head)
             (let [[_ out acc scan-init idx _bound _cast body] init
                   free-tenv (dissoc tenv acc idx)
                   active? (or (and (symbol? scan-init)
                                    (contains? free-tenv scan-init))
                               (and (seq? scan-init)
                                    (any-active? free-tenv scan-init))
                               (any-active? free-tenv body))]
               (cond
                 (contains? free-tenv out)
                 (throw (ex-info "jvp: scan mutates an active input buffer"
                                 {:reason :jvp-scan-active-output :out out}))

                 active?
                 (let [[dscan extra] (fold-double-scan tenv sym init tag)]
                   (done (assoc tenv sym dscan) extra))

                 :else (done tenv [])))

             (= 'raster.par/reduce head)
             (let [[_ acc reduce-init idx _bound body] init
                   free-tenv (dissoc tenv acc idx)
                   active? (or (and (symbol? reduce-init)
                                    (contains? free-tenv reduce-init))
                               (and (seq? reduce-init)
                                    (any-active? free-tenv reduce-init))
                               (any-active? free-tenv body))]
               (if active?
                 (let [[dreduce extra] (fold-double-reduce tenv sym init tag)]
                   {:tenv (assoc tenv sym dreduce)
                    :bindings (into prior-bindings extra)})
                 (done tenv [])))

             (= 'raster.par/map! head)
             (let [[_ out idx _bound & tail] init
                   body (last tail)
                   active? (any-active? (dissoc tenv idx) body)]
               (cond
                 (contains? tenv out)
                 (throw (ex-info "jvp: map! mutates an active input buffer"
                                 {:reason :jvp-map-active-output :out out}))

                 (and active? (not (fresh-map-output? prior-bindings out)))
                 (throw (ex-info "jvp: map! output is not proven fresh"
                                 {:reason :jvp-map-unproven-output :out out}))

                 active?
                 (let [[dout dmap extra] (fold-fresh-map! tenv sym init tag)]
                   (done (assoc tenv out dout sym dmap) extra))

                 :else (done tenv [])))

             ;; Remaining control flow has no forward rule yet.
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
        prepared (rev/ad-prepare (first walked-body)
                                 (zipmap all-params tags))
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
        ;; ad-prepare (lower-composites) and reify-pullback each open their OWN
        ;; with-ad-gensym; run BOTH under one shared counter (like
        ;; compile-hvp-fn's enclosing binding) so reify-pullback continues from
        ;; ad-prepare's high-water mark instead of resetting *gensym-counter* to
        ;; 0 and re-minting colliding anf__ temps across the phase boundary.
        {:keys [fwd-bindings result-sym body-sym pullback-form]}
        (rev/call-with-shared-ad-gensym
         (fn [] (rev/reify-pullback
                 (rev/ad-prepare (first walked-body)
                                 (zipmap all-params tags))
                 diff-params)))
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
        _ (reject-unlinearized-hvp-scopes! grad-bindings)
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
