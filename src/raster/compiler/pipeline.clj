(ns raster.compiler.pipeline
  "Nanopass-inspired compiler pipeline orchestration.

  Passes are composed declaratively via typed arrows (=> From To).
  Each pass declares its input and output dialect, and the pipeline
  runner validates arrow compatibility at composition time.

  Single unified pipeline (GPU/SIMD):
    [:lower :fixpoint :dce :buffer-fuse :resolve-alength :soac-fuse :compound-detect :backend :mem-merge]

  AD is handled by inlining value+grad calls during the fixpoint pass.
  Training, gradients, and higher-order derivatives all go through the
  same forward pipeline — write a deftm that calls value+grad and any
  optimizer/transform deftm, then compile with compile-aot.

  Key insight: AD handles par forms directly — the backward of a par/map!
  is a par/map!, preserving parallelism through the entire pipeline.

  Terminal step (hoist) runs after all passes, producing the
  compiled function with lazy buffer allocation."
  (:require [clojure.string]
            [clojure.pprint :as pp]
            [raster.compiler.passes.scalar.dce :as dce]
            [raster.compiler.passes.scalar.cse :as cse]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.buffer-fuse :as buffer-fuse]
            [raster.compiler.passes.scalar.normalize :as normalize]
            [raster.compiler.passes.scalar.rewalk :as rewalk]
            [raster.compiler.ir.par :as par]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.backend.wasm.emit :as wasm-emit]
            [raster.compiler.backend.intrinsics :as ix]
            [raster.compiler.backend.gpu.wgsl :as wgsl-emit]
            [raster.compiler.passes.scalar.soa-lower :as soa-lower]
            [raster.compiler.passes.parallel.par-fusion :as par-fusion]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-graph :as soac-graph]
            [raster.compiler.backend.gpu.entry :as gpu-entry]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.passes.parallel.compound-detect :as compound-detect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.loop-lift :as loop-lift]
            [raster.compiler.passes.parallel.write-read-fuse :as write-read-fuse]
            [raster.compiler.passes.parallel.materialize :as materialize]
            [raster.compiler.passes.parallel.gpu-plan :as gpu-plan]
            [raster.compiler.passes.scalar.mem-merge :as mem-merge]
            [raster.compiler.passes.scalar.resolve-alength :as resolve-alength]
            [raster.compiler.core.closure :as closure]
            [raster.compiler.core.hoist :as hoist]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.macroexpand :as mex]
            [raster.ad.purity :as purity]
            [raster.ad.reverse :as ad-reverse]
            [raster.compiler.core.util :as util]
            [raster.compiler.ad.flatten :as flatten]
            [raster.compiler.ad.mode-select :as mode-select]
            [raster.compiler.passes.parallel.device :as device]
            [clojure.walk :as walk]
            [raster.runtime.hardware :as hardware]
            [raster.compiler.ir.dialects :as dialects]
            [raster.compiler.ir.invariants :as invariants]
            [raster.compiler.ir.form :as form]
            [raster.analysis.memory :as mem]
            [clojure.set :as set]
            [raster.compiler.core.method-entry :as me]
            [raster.core :as rcore]))

(defn- check-purity! [walked-body active-params warning-meta]
  (purity/validate-for-ad! walked-body active-params warning-meta))

;; ================================================================
;; Helpers
;; ================================================================

(defn- resolve-deftm-var
  "Resolve a deftm var. If f-var is a dispatch var (generic function),
  find the backing method var. When dtype is specified, selects the
  matching overload (e.g. :float picks float[] methods over double[]).
  Without dtype, picks the single overload or errors if ambiguous."
  ([f-var] (resolve-deftm-var f-var nil))
  ([f-var dtype]
   (if (:raster.core/deftm (meta f-var))
     f-var
     ;; Look up the dispatch table
     (when-let [dt (:raster.core/dispatch-table (meta f-var))]
       (let [all-methods (mapcat val @dt)
             dtype-tag (case dtype :float 'floats :double 'doubles nil)
             method (if dtype-tag
                      (or (first (filter #(some #{dtype-tag} (:tags %)) all-methods))
                          (first all-methods))
                      ;; No dtype: use single overload, or error if ambiguous
                      (if (= 1 (count all-methods))
                        (first all-methods)
                        (throw (ex-info
                                (str "Ambiguous overload for compile-aot on "
                                     (:name (meta f-var)) ". Specify :dtype to disambiguate.\n"
                                     "Available: " (mapv :tags all-methods))
                                {:var f-var :methods (mapv :tags all-methods)}))))
             ns-obj (:ns (meta f-var))
             mangled-name (symbol (str (types/mangle
                                        (:name (meta f-var)) (:tags method))))]
         (ns-resolve ns-obj mangled-name))))))

(defn- infer-dtype
  "Infer dtype from a resolved deftm var's parameter tags.
   Checks array tags first (floats/doubles), then scalar tags (float/double).
   Returns :float, :double, or nil (integer-only functions)."
  [resolved-var]
  (when-let [tags (:raster.core/deftm-tags (meta resolved-var))]
    (cond
      (some #{'floats} tags) :float
      (some #{'doubles} tags) :double
      (some #{'float} tags) :float
      (some #{'double} tags) :double
      :else nil)))

(defn walk-body-with-tc
  "Walk a deftm body with TC type inference. Used at compilation time
  (compile-aot, lazy JIT) when concrete types are known.
  Uses build-param-type-env — single source of truth for type-env."
  [body params tags annotations source-ns]
  (let [type-env (inf/build-param-type-env params tags annotations)
        ;; Run TC for per-binding type inference
        tc-binding-tags (when annotations
                          (try
                            (:binding-tags (inf/tc-analyze-deftm-body '<compile> params annotations body source-ns))
                            (catch Throwable e
                              (binding [*out* *err*]
                                (println (str "WARNING: TC analysis failed at compile time: " (.getMessage e))))
                              nil)))
        walk-opts (cond-> {:type-env type-env :source-ns (or source-ns *ns*)}
                    (seq tc-binding-tags) (assoc :tc-binding-tags tc-binding-tags))]
    (mapv #(walker/walk-body % walk-opts) body)))

(defn get-walked-body [f-var & [dtype]]
  (let [resolved (or (resolve-deftm-var f-var dtype) f-var)
        m (meta resolved)]
    ;; Always re-walk with TC at compilation time for best type inference.
    ;; The definition-time walked body (without TC) is used for interpreted
    ;; dispatch; at compile-aot time we want TC binding tags.
    (or (when-let [raw-body (or (:raster.core/deftm-source-body m)
                                (:raster.core/deftm-raw-body m))]
          (let [params (:raster.core/deftm-params m)
                tags (:raster.core/deftm-tags m)
                annotations (or (:raster.core/deftm-annotations m)
                                ;; Reconstruct annotations from tags for older deftm
                                (when tags (mapv (fn [t] (if (= t 'Object) nil t)) tags)))
                src-ns-sym (:raster.core/deftm-source-ns m)
                src-ns (when src-ns-sym (try (the-ns src-ns-sym) (catch Exception _ *ns*)))
                walked (try (walk-body-with-tc (vec raw-body) params tags annotations src-ns)
                            (catch Throwable t
                              (binding [*out* *err*]
                                (println "WARNING: walker re-walk failed for" f-var
                                         "- falling back to pre-walked body:" (.getMessage t)))
                              nil))]
            walked))
        ;; Fallback: use pre-walked body from definition time or lazy walk
        (:raster.core/deftm-walked-body-typed m)
        (:raster.core/deftm-walked-body m)
        (rcore/ensure-walked-body! resolved)
        (throw (ex-info "Var has no deftm walked body or raw body" {:var f-var})))))

(defn get-params [f-var & [dtype]]
  (let [resolved (or (resolve-deftm-var f-var dtype) f-var)]
    (or (:raster.core/deftm-params (meta resolved))
        (throw (ex-info "Var has no deftm params" {:var f-var})))))

(defn build-param-env
  "Build a {sym -> tag} type environment from deftm parameter tags.
  Maps each param symbol to its walker tag (e.g. 'doubles, 'double, 'longs).
  Falls back to nil (unknown) for params without tags."
  [f-var & [dtype]]
  (let [resolved (or (resolve-deftm-var f-var dtype) f-var)
        params (:raster.core/deftm-params (meta resolved))
        tags   (:raster.core/deftm-tags (meta resolved))]
    (when (and params tags (= (count params) (count tags)))
      (into {}
            (map (fn [p t]
                   [(with-meta (if (symbol? p) p (symbol (name p))) nil)
                    (symbol t)])
                 params tags)))))

(defn clean-params
  "Strip metadata (type hints) from param symbols for eval."
  [params]
  (vec (map #(with-meta (if (symbol? %) % (symbol (name %))) nil)
            params)))

;; ================================================================
;; Dialect validation — formal structural invariants via nanopass
;; ================================================================

(def ^:private ^:dynamic *validate-dialects?*
  "Structural validation at each pass boundary: dialect-shape checkers +
  always-on structural invariants (cheap). Default true."
  true)

(def ^:dynamic *validate-deep?*
  "Deep SEMANTIC invariants (qualify-upfront, type-tag presence, op
  classification) at pass boundaries. Cross-cutting properties a dialect grammar
  cannot express. Currently WARN-only; default off (enable in CI / tests).
  Independent of how a pass is implemented — runs at the same boundary seam
  whether the pass is direct-walking or a pattern `defpass`."
  false)

(defn- validate-dialect!
  "Validate form against dialect at a pass boundary. Throws on a structural
  violation (dialect shape or always-on structural invariant); warns on a deep
  semantic invariant when *validate-deep?* is set. `opts` carries :active-params
  (the deftm params), needed to exclude them from the qualify-upfront check."
  [dialect-key form pass-key opts]
  (when *validate-dialects?*
    (when-let [[valid? validate-fn] (get dialects/dialect-checkers dialect-key)]
      (when-not (valid? form)
        (let [details (validate-fn form)]
          (throw (ex-info (str "Pass :" pass-key " produced invalid " dialect-key " form")
                          {:pass pass-key
                           :dialect dialect-key
                           :validation-error details
                           :form-preview (pr-str (if (seq? form)
                                                   (take 3 form)
                                                   form))})))))
    (invariants/check-structural! dialect-key form pass-key))
  (when *validate-deep?*
    (invariants/check-deep! dialect-key form pass-key (:active-params opts))))

;; ================================================================
;; Pass functions — each takes (form, opts) → form or {:form :stats}
;; ================================================================

(defn- effect-bindings-for
  "Bind each effect statement to a fresh effectful throwaway, preserving order.
   (effect1) (effect2) → [_eff1 (effect1) _eff2 (effect2)]"
  [effects]
  (vec (mapcat (fn [eff]
                 [(with-meta (gensym "_eff_") {:raster.effect/effectful true}) eff])
               effects)))

(defn- normalize-let-body
  "Normalize a body to a single-result let* form, lifting statement effects into
   effectful bindings (so order/effects survive DCE and the let*-only dialects).
   (let [a 1] (effect1) (effect2) result)
   → (let* [a 1 _eff1 (effect1) _eff2 (effect2)] result)
   (do (effect1) result)        → (let* [_eff1 (effect1)] result)
   A multi-statement `do` is the shape a multi-form deftm body walks to (e.g.
   `(dotimes ...) ret`); without this it reaches the :fixpointed dialect as a bare
   `do` and fails the let* check (compile-aot of any multi-form kernel)."
  [form]
  (cond
    ;; (do e1 ... ret) — multi-statement do → let* with effect bindings
    (and (seq? form) (= 'do (first form)))
    (let [body-exprs (rest form)]
      (cond
        (empty? body-exprs) form
        (= 1 (count body-exprs)) (recur (first body-exprs))   ; (do x) → normalize x
        :else (list 'let* (effect-bindings-for (butlast body-exprs)) (last body-exprs))))

    (not (form/binding-form? form)) form

    :else
    (let [[_head bindings & body-exprs] form]
      (if (<= (count body-exprs) 1)
        form ;; Already single-body — leave as-is
        ;; Multi body — lift effects into bindings
        (list 'let* (vec (concat bindings (effect-bindings-for (butlast body-exprs))))
              (last body-exprs))))))

(defn- tag-expr-types
  "Recursively attach :tag metadata to expressions whose type is inferrable.
   Tags let/let* forms (recurse into bindings), par forms (from init/buffer),
   and loop forms. Does not descend into opaque calls."
  [expr env]
  (cond
    (not (seq? expr)) expr
    ;; let/let*: tag binding symbols + recurse into init exprs and body
    (form/binding-form? expr)
    (let [[head bindings-vec & body] expr
          pairs (partition 2 bindings-vec)
          [new-env new-bindings]
          (reduce
           (fn [[env acc] [sym init-expr]]
             (let [tagged-init (tag-expr-types init-expr env)
                   tag (inf/infer-arg-tag tagged-init env)
                   tagged-sym (if tag (vary-meta sym assoc :tag tag :raster.type/tag tag) sym)
                   new-env (if tag (assoc env sym tag) env)]
               [new-env (conj acc tagged-sym tagged-init)]))
           [env []]
           pairs)
          tagged-body (map #(tag-expr-types % new-env) body)
          r (apply list head (vec new-bindings) tagged-body)]
      ;; Preserve original form metadata (source locations, :raster.op/original)
      (if-let [m (meta expr)]
        (with-meta r m)
        r))
    ;; Other forms: tag the form itself if type is inferrable
    :else
    (let [tag (inf/infer-arg-tag expr env)]
      (if (and tag (not (:tag (meta expr))))
        (try (with-meta expr (assoc (or (meta expr) {}) :tag tag))
             (catch Exception _ expr))
        expr))))

(defn- tag-binding-types
  "Walk a form and attach :tag metadata to binding symbols and expressions.
   Entry point that initializes the env from param-env."
  [form param-env]
  (tag-expr-types form (or param-env {})))

(defn- pass-lower
  "Lower compound deftm ops into primitive ops for AD.
  Stops at rrule template boundaries — ops with templates are kept as calls.
  Wraps bare expressions in let* for dialect consistency.
  Converts multi-body let forms into single-body with effect bindings.
  Tags all binding symbols with inferred types via metadata.
  (=> :walked :lowered)"
  [form opts]
  (let [;; Normalize multi-body let FIRST — before inlining can drop effects
        normalized (normalize-let-body form)
        result (if (:inline? opts)
                 (with-bindings {#'flatten/*flatten-dtype* (:dtype opts)
                                 #'inline/*param-env* (:param-env opts)
                                 #'inline/*ad-transform-body-fn* ad-reverse/transform-body}
                   (inline/lower-to-ad-primitives normalized))
                 normalized)
        ;; Ensure result is a let* form (dialect requirement)
        let-form (if (form/binding-form? result)
                   result
                   (list 'let* [] result))]
    ;; Tag binding symbols with inferred types
    (tag-binding-types let-form (:param-env opts))))

(defn- pass-expand
  "Expand template-backed ops for backend optimization.
  Full inlining (no AD-friendliness concerns) — exposes loop bodies
  for buffer-fuse, SOAC fusion, and SIMD/GPU backends.
  Runs after buffer-fuse so rewritten ops (e.g. transpose-2d!) get inlined too.
  (=> :buffer-fused :expanded)"
  [form opts]
  (let [expanded (binding [inline/*ad-transform-body-fn* ad-reverse/transform-body]
                   (inline/expand-for-backends form 3 (:param-env opts)))]
    (if (= expanded form)
      form
      {:form expanded
       :stats {:expanded? true}})))

(defn- has-undevirtualized-calls?
  "Check if a form contains qualified raster.* dispatch calls that could be devirtualized."
  [form]
  (cond
    (seq? form)
    (or (op/undevirtualized-call? (first form))
        (some has-undevirtualized-calls? form))
    (vector? form) (some has-undevirtualized-calls? form)
    :else false))

(defn- ensure-let*-result
  "Normalize a form to a let* so the flat-let* dialects accept it. A multi-form
   deftm body walks to a bare `do`, and a single-expression body (a `loop`, `if`,
   call, …) is also valid IR — but PE collapses the empty `(let* [] …)` wrapper
   pass-lower adds (pe.clj returns the bare body when all bindings are eliminated),
   leaving a non-let* top form that fails validation. Re-wrap here (post-PE, so it
   survives): `do` → effect-binding let*; any other non-binding form → bind the
   result to a gensym. The final emission (compile!) likewise wraps non-let* forms."
  [form]
  (let [norm (normalize-let-body form)]
    (if (form/binding-form? norm)
      norm
      (let [g (gensym "_ret_")] (list 'let* [g norm] g)))))

(defn- pass-fixpoint
  "Fixpoint pass: expand → normalize → rewalk → PE+CSE until stable.

  Handles composable AD operators (value+grad, grad) whose inlined bodies
  contain undevirtualized backward-pass calls. Each iteration:
  1. Expand (inline deftm/value+grad calls)
  2. Normalize (canonicalize scalar ops)
  3. Rewalk (devirtualize arithmetic back to .invk)
  4. PE+CSE (constant fold, eliminate redundancy)

  Converges in 1 iteration for simple AD, N iterations for N-level nesting.
  Max 5 iterations to prevent divergence.
  (=> :* :fixpointed)"
  [form opts]
  (let [max-iters 5
        ;; Force simplify mode for the fixpoint — we always want normalize+rewalk
        opts (assoc opts :simplify? true)]
    (loop [current form
           iter 0
           total-stats {:fixpoint-iterations 0}]
      (if (>= iter max-iters)
        {:form (ensure-let*-result current) :stats total-stats}
        (let [;; Step 1: Expand (inline deftm calls + value+grad AD inlining)
              expanded (binding [inline/*ad-transform-body-fn* ad-reverse/transform-body]
                         (inline/expand-for-backends current 3 (:param-env opts)))
              ;; Step 2: Normalize + Rewalk only if expand introduced NEW
              ;; undevirtualized calls (e.g. from composable AD body inlining).
              ;; Regular expand (compound deftm inlining) produces pre-walked code
              ;; that should NOT be rewalked — rewalk can change arithmetic behavior.
              needs-rewalk? (and (not= expanded current)
                                 (has-undevirtualized-calls? expanded))
              rewalked (if needs-rewalk?
                         ;; Skip full normalize — PE handles .invk normalization
                         ;; internally (normalize-1 + simplify-1). A full recursive
                         ;; normalize here strips typed metadata from .invk calls
                         ;; inside nested structures (dotimes, par bodies, loop*)
                         ;; that the walker can't re-reach.
                         (let [rw-result (rewalk/pe-rewalk expanded opts)]
                           (if (map? rw-result) (:form rw-result) rw-result))
                         expanded)]
          (if (= rewalked current)
            {:form (ensure-let*-result current) :stats (assoc total-stats :fixpoint-iterations iter)}
            (recur rewalked (inc iter)
                   (assoc total-stats :fixpoint-iterations (inc iter)))))))))

(defn- pass-mode-select
  "Analyze Jacobian structure and annotate form with recommended AD mode.
  (=> :walked :walked) — identity dialect, metadata annotation only.
  When analysis succeeds, sets ::ad-mode on form metadata.
  Falls back silently (no annotation) when symbolic tracing fails."
  [form opts]
  (try
    (let [active-params (:active-params opts)
          analysis (mode-select/analyze-from-exprs [form] (mapv symbol active-params))
          mode (:recommended-mode analysis)]
      (if (seq? form)
        (with-meta (doall form) (assoc (meta form) ::ad-mode mode))
        form))
    (catch Exception _e
      ;; Symbolic analysis can fail on loops, opaque calls, etc.
      ;; Fall back gracefully — AD pass defaults to :reverse
      form)))

(defn- pass-dce
  "Dead code elimination. Removes unused bindings via backward reachability.
  Flexible input: accepts any flat let* dialect (:lowered, :simplified, :peepholed).
  Returns {:form :stats}."
  [form opts]
  (let [result (dce/eliminate-dead-bindings form)]
    {:form (:form result) :stats (:stats result)}))

(defn- undevirtualized-dispatch-call?
  "True if sym is a generic dispatch call that should have been devirtualized.
  Excludes par primitives (pmap, map!, reduce, etc.), Java system calls, and
  other compiler IR forms."
  [sym]
  (and (op/undevirtualized-call? sym)
       (symbol? sym)
       (when-let [ns-str (namespace sym)]
         ;; Only flag raster.* namespace calls (actual deftm dispatch)
         (and (.startsWith ^String ns-str "raster.")
              (not= ns-str "raster.par")
              (not= ns-str "raster.ad.reverse")))))

(defn- count-undevirtualized
  "Count remaining undevirtualized dispatch calls in a form."
  [form]
  (cond
    (seq? form)
    (+ (if (undevirtualized-dispatch-call? (first form)) 1 0)
       (reduce + 0 (map count-undevirtualized form)))
    (vector? form) (reduce + 0 (map count-undevirtualized form))
    :else 0))

(defn- pass-late-cleanup
  "Late CSE + DCE + dispatch resolution after buffer fusion and backward inlining.
  Eliminates aliases and dead bindings, re-tags binding types (new bindings
  from fixpoint/buffer-fuse need types), then resolves remaining generic
  dispatch calls to their mangled implementations.
  Returns {:form :stats}."
  [form opts]
  (if-not (form/binding-form? form)
    form
    (let [cse-result (cse/cse-let form)
          dce-result (dce/eliminate-dead-bindings (:form cse-result))
          ;; Re-tag binding types — fixpoint and buffer-fuse may have
          ;; introduced new bindings without type metadata
          tagged (tag-binding-types (:form dce-result) (:param-env opts))
          ;; Resolve any remaining generic dispatch calls
          resolved (inline/resolve-generic-deftm-calls tagged (:param-env opts))
          ;; Warn about surviving undevirtualized calls — these cause massive
          ;; performance degradation (270ns dispatch vs 4ns .invk per call)
          remaining (count-undevirtualized resolved)]
      (when (pos? remaining)
        (binding [*out* *err*]
          (println (str "WARNING: " remaining " undevirtualized dispatch call(s) survive "
                        "late-cleanup — expect ~100x slowdown in hot loops. "
                        "Check type inference in gradient/backward code."))))
      {:form resolved
       :stats {:cse-aliases (:cse-aliases (:stats cse-result))
               :bindings-removed (:bindings-removed (:stats dce-result))
               :undevirtualized-remaining remaining}})))

(defn- pass-buffer-fuse
  "Buffer fusion: rewrite allocating ops to reuse dead buffers.
  Also resolves remaining generic deftm dispatch calls to mangled
  implementations — these appear from buffer-fuse into-variant rewrites
  and from inlined forward pass code (softmax loop with exp, etc.).
  Returns {:form :stats}."
  [form opts]
  (if (form/binding-form? form)
    (let [result (buffer-fuse/fuse-let form :dtype (:dtype opts))
          resolved (inline/resolve-generic-deftm-calls (:form result) (:param-env opts))]
      (assoc result :form resolved))
    {:form form :stats {:fused 0 :fresh-allocs 0 :unchanged 0}}))

(defn- pass-loop-lift
  "Lift plain dotimes/loop forms into raster.par operations.
  Detects map, reduce, and scan patterns in sequential loops and
  rewrites them as par/map!, par/reduce, par/scan for downstream
  SOAC fusion and SIMD/GPU compilation.
  Returns {:form :stats}."
  [form _opts]
  (if (form/binding-form? form)
    (let [{:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      {:form form :stats stats})
    {:form form :stats {:maps-detected 0 :reduces-detected 0 :scans-detected 0}}))

(defn- pass-write-read-fuse
  "Fuse 2D dotimes+par/map! producer → 1D par/map! consumer patterns.
  Eliminates intermediate gradient buffers (e.g. dW+SGD fusion).
  Returns {:form :stats}."
  [form _opts]
  (if (form/binding-form? form)
    (write-read-fuse/fuse-write-read form)
    {:form form :stats {:write-read-fused 0}}))

(defn- pass-soac-fuse
  "SOAC graph-based fusion: vertical (map→map, map→reduce, map→scan),
  horizontal (independent same-bound maps), iterated to fixpoint.
  Falls back to par-fusion for non-let* forms.
  Returns {:form :stats}."
  [form _opts]
  (if (form/binding-form? form)
    (let [[let-sym bindings-vec & body-exprs] form
          pairs (vec (partition 2 bindings-vec))
          nodes (soac/let-bindings->nodes pairs)
          graph (soac-graph/build-fusion-graph nodes)
          [fused-graph stats] (soac-graph/fusion-fixpoint graph)
          new-pairs (soac/nodes->let-bindings (:nodes fused-graph))
          new-bindings (vec (mapcat identity new-pairs))]
      {:form (list* let-sym new-bindings body-exprs)
       :stats stats})
    ;; Not a let* form — fall back to par-fusion
    (par-fusion/par-fusion-pass form)))

(defn- register-gpu-kernels!
  "Register generated GPU kernels eagerly so they're available at eval time."
  [kernels]
  (when (seq kernels)
    (try
      (require 'raster.gpu.ze-runtime)
      (let [register! (resolve 'raster.gpu.ze-runtime/register-kernel!)]
        (doseq [k kernels]
          (register! (:kernel-name k) k)))
      (catch Exception e
        (println "Warning: could not register GPU kernels:" (.getMessage e))))))

(defn- pass-gpu-plan
  "GPU execution plan: rewrite BLAS calls to GPU GEMM kernels,
  plan DeviceBuffer allocations for persistent GPU execution.
  Only runs when target-device is a GPU device (Level Zero, CUDA, ROCm).
  (=> :dce-cleaned :gpu-planned)"
  [form opts]
  (let [target-device (:target-device opts)
        dtype (:dtype opts)]
    (if-not (device/gpu-target? target-device)
      ;; No GPU target — pass through
      form
      (let [result (gpu-plan/gpu-plan-pass form
                                           :dtype dtype
                                           :weight-params (:weight-params opts)
                                           :active-params (:active-params opts)
                                           :target-device target-device)
            kernels (:kernels result)]
        (register-gpu-kernels! kernels)
        (if (pos? (+ (get-in result [:stats :gemm-rewrites] 0)
                     (get-in result [:stats :transpose-rewrites] 0)
                     (get-in result [:stats :map-rewrites] 0)
                     (get-in result [:stats :reduce-rewrites] 0)))
          {:form (:form result)
           :stats (:stats result)
           :kernels kernels
           :gpu-buffers (:gpu-buffers result)}
          ;; Nothing rewritten — pass through original form
          form)))))

(defn- pass-materialize
  "Materialize pure par/map forms into alloc + par/map! for backend consumption.
  Pure par/map forms are value-producing with no output buffer. This pass
  allocates output buffers and converts them to imperative par/map! forms.
  (=> :soac-fused :materialized)"
  [form opts]
  (materialize/materialize-pass form opts))

(defn- pass-segop-lower
  "Lower par forms to SegOp records via SOAC intermediate.
  SegOp records are attached as metadata on binding symbols.
  (=> :materialized :segop-lowered)"
  [form opts]
  (segop-lower/segop-lower-pass form opts))

(defn- pass-compound-detect
  "Compound kernel detection: finds dotimes loops with ≥2 par forms
  and wraps them in raster.compiler/compound-kernel markers.
  Returns {:form :stats}."
  [form opts]
  (compound-detect/compound-detect-pass form opts))

(defn- strip-compound-markers
  "Replace compound-kernel markers with their original dotimes form.
  Used by non-GPU backends (SIMD, scalar) that don't process compound markers."
  [form]
  (walk/postwalk
   (fn [f]
     (if (and (seq? f) (= 'raster.compiler/compound-kernel (first f)))
       (nth f 2)  ;; (compound-kernel metadata original-dotimes) → original-dotimes
       f))
   form))

(defn- pass-backend
  "Backend-specific optimization. Dispatches to SIMD, CUDA, OpenCL, or scalar.
  Returns {:form :stats :cuda-result :kernels}."
  [form opts]
  (let [target-device (:target-device opts)
        simd? (:simd? opts true)]
    (if (:keep-par-forms? opts)
      ;; CPU-C SIMD path: leave par/map!/par/reduce forms INTACT (neither scalarize
      ;; nor JVM-Vector-API lower) so the monolithic C backend can consume the same
      ;; SegRed/SegMap the JVM path builds, but emit __m256 intrinsics instead.
      {:form (strip-compound-markers form) :stats nil :backend :par-preserve}
      (case (device/select-runtime-backend target-device simd? nil)
      :cuda
      (throw (ex-info "CUDA backend not yet reimplemented (use :opencl)" {:target target-device}))

      :opencl
      ;; Attach the declared scalar/array types (from opts) onto the form so opencl-pass's
      ;; generators read declared types instead of guessing. The materialized form preserves
      ;; the original param symbols, so a name-keyed type map still applies.
      (let [form (cond-> form
                   (or (:scalar-types opts) (:array-types opts))
                   (vary-meta assoc :scalar-types (:scalar-types opts) :array-types (:array-types opts)))
            result (opencl-pass/opencl-pass form
                                            :device-id target-device
                                            :dtype (:dtype opts))]
        (register-gpu-kernels! (:kernels result))
        {:form (:form result) :stats (:stats result) :kernels (:kernels result) :backend :opencl})

      :simd
      (let [clean-form (strip-compound-markers form)
            {:keys [form stats]} (par-simd/simd-pass clean-form)]
        {:form form :stats stats :backend :simd})

      ;; :scalar
      {:form (par/expand-par-forms (strip-compound-markers form)) :stats nil :backend :scalar}))))

(defn- pass-resolve-alength
  "Resolve (alength hoistable-buf) to original allocation size.
  Decouples iteration counts from physical buffer sizes so mem-merge
  can freely merge different-size buffers.
  Returns {:form :stats}."
  [form _opts]
  (resolve-alength/resolve-alength-pass form))

(defn- pass-mem-merge
  "Memory block merging: Futhark-style interference graph coloring.
  Returns {:form :stats}."
  [form opts]
  (let [device-env (:device-env opts)
        params-set (set (:active-params opts))
        ;; Auto-infer :hoistable on allocations whose sizes are param-derived
        form (if (seq params-set)
               (hoist/infer-hoistable form params-set)
               form)]
    (mem-merge/merge-memory-blocks form :device-env device-env)))

;; ================================================================
;; Pass registry with typed arrows (=> From To)
;; ================================================================

(def ^:private pass-specs
  "Registry of pass specifications. Each pass declares its input and output
  dialect via :from/:to arrows, enabling typed pipeline composition.

  Some passes are flexible in their input dialect (e.g. :dce accepts any
  flat let* form). The :from tag is set to the most common input."
  {;; Core pipeline passes (used in forward-passes)
   :lower            {:from :walked            :to :lowered           :fn pass-lower}
   :fixpoint         {:from :lowered           :to :fixpointed        :fn pass-fixpoint}
   :dce              {:from :*                 :to :dce-cleaned       :fn pass-dce}
   :buffer-fuse      {:from :dce-cleaned       :to :buffer-fused      :fn pass-buffer-fuse}
   :loop-lift        {:from :late-cleaned      :to :loop-lifted       :fn pass-loop-lift}
   :write-read-fuse  {:from :loop-lifted       :to :write-read-fused  :fn pass-write-read-fuse}
   :soac-fuse        {:from :write-read-fused  :to :soac-fused        :fn pass-soac-fuse}
   :materialize      {:from :soac-fused        :to :materialized      :fn pass-materialize}
   :segop-lower      {:from :materialized      :to :segop-lowered     :fn pass-segop-lower}
   :compound-detect  {:from :segop-lowered     :to :compound-detected :fn pass-compound-detect}
   :backend          {:from :compound-detected :to :backend-applied   :fn pass-backend}
   :resolve-alength  {:from :backend-applied   :to :alength-resolved  :fn pass-resolve-alength}
   :mem-merge        {:from :alength-resolved  :to :mem-merged        :fn pass-mem-merge}
   ;; Building blocks for custom/GPU pipelines
   :expand           {:from :*                 :to :expanded          :fn pass-expand}
   :mode-select      {:from :walked            :to :walked            :fn pass-mode-select}
   :gpu-plan         {:from :dce-cleaned       :to :gpu-planned       :fn pass-gpu-plan}
   :late-cleanup     {:from :*                 :to :late-cleaned      :fn pass-late-cleanup}})

;; ================================================================
;; Pipeline runner — validates arrow types between passes
;; ================================================================

(defn- pass-result-form
  "Extract the form from a pass result (plain form or {:form ...} map)."
  [result]
  (if (map? result) (:form result) result))

(defn run-passes
  "Reduce a form through a sequence of passes, validating arrow types.
  Passes with :from :* accept any input dialect (flexible).
  Validates output against target dialect when *validate-dialects?* is true.
  Returns the final transformed form.

  `start-dialect` (default :walked) lets a caller resume a split pipeline — e.g.
  the resident GPU path runs the front half to :materialized, applies soa-lower
  (which is not a registered pass because it also rewrites the param signature),
  then resumes the back half from :materialized."
  ([form passes opts] (run-passes form passes opts :walked))
  ([form passes opts start-dialect]
   (first
    (reduce (fn [[f dialect] pass-key]
              (let [{:keys [from to]} (get pass-specs pass-key)
                    pass-fn (:fn (get pass-specs pass-key))
                    _ (assert (or (= :* from) (= dialect from))
                              (str "Pass :" pass-key " expects :" from " but pipeline is at :" dialect
                                   ". Check pass ordering."))
                    result (pass-fn f opts)
                    f' (pass-result-form result)]
                ;; Validate output against target dialect (fails hard)
                (validate-dialect! to f' pass-key opts)
                [f' to]))
            [form start-dialect] passes))))

;; ================================================================
;; Mode configurations (declarative pass vectors)
;; ================================================================

(def forward-passes
  "Forward: lower → fixpoint → DCE → buffer-fuse → late-cleanup → loop-lift → write-read-fuse → soac-fuse → segop-lower → compound-detect → backend → resolve-alength → mem-merge.
   fixpoint loops expand+normalize+rewalk until stable, handling composable AD inlining.
   loop-lift recovers par structure from dotimes/loop forms (e.g. SGD inlined to dotimes).
   write-read-fuse eliminates intermediate buffers by fusing 2D producers into 1D consumers (dW+SGD).
   segop-lower converts par forms to SegOp IR for unified backend consumption."
  [:lower :fixpoint :dce :buffer-fuse :late-cleanup :loop-lift :write-read-fuse :soac-fuse :materialize :segop-lower :compound-detect :backend :resolve-alength :mem-merge])

(def gpu-resident-pre-soa-passes
  "forward-passes up to (and including) :materialize. The resident GPU path splits here so
   soa-lower can explode value-type (Params container) params into per-field arrays at the
   :materialized boundary — BEFORE the :backend pass emits kernels (the JVM bytecode backend
   keeps Valhalla value-classes native, so soa-lower is per-backend, not a shared pass)."
  [:lower :fixpoint :dce :buffer-fuse :late-cleanup :loop-lift :write-read-fuse :soac-fuse :materialize])

(def gpu-resident-post-soa-passes
  "forward-passes from :segop-lower onward — resumed (from :materialized) after soa-lower."
  [:segop-lower :compound-detect :backend :resolve-alength :mem-merge])

;; ================================================================
;; Diagnostic runner for show-pipeline
;; ================================================================

(defn- run-passes-diagnostic
  "Run passes with full intermediate recording for diagnostics.
  Returns {:form final-form :stages {stage-key form} :stats {stage-key stats}}."
  [form passes opts]
  (reduce (fn [acc pass-key]
            (let [{:keys [from to]} (get pass-specs pass-key)
                  pass-fn (:fn (get pass-specs pass-key))
                  current-dialect (:dialect acc)
                  _ (assert (or (= :* from) (= current-dialect from))
                            (str "Pass :" pass-key " expects :" from " but pipeline is at :"
                                 current-dialect))
                  result (pass-fn (:form acc) opts)
                  new-form (pass-result-form result)]
              (validate-dialect! to new-form pass-key opts)
              (-> acc
                  (assoc :form new-form :dialect to)
                  (assoc-in [:stages to] new-form)
                  (cond-> (and (map? result) (:stats result))
                    (assoc-in [:stats to] (:stats result)))
                  ;; Capture backend-specific data
                  (cond-> (and (map? result) (:backend result))
                    (assoc-in [:stages :backend-type] (:backend result)))
                  (cond-> (and (map? result) (:kernels result))
                    (assoc-in [:stages :kernels] (:kernels result)))
                  (cond-> (and (map? result) (:cuda-result result))
                    (assoc-in [:stages :cuda-result] (:cuda-result result))))))
          {:form form :stages {} :stats {} :dialect :walked} passes))

;; ================================================================
;; Pipeline: forward-only (buffer fusion, no AD)
;; ================================================================

(declare compile-aot-jvm)

(defn compile-aot
  "Compile a deftm var into an optimized forward function with buffer fusion.

  Pipeline: walker → inline → DCE → buffer-fuse → par-fuse → backend
            → mem-merge → hoist → eval

  Par forms from broadcast!/reduce! are preserved through the pipeline,
  then fused (map→reduce, horizontal) by par-fusion, then optimized to
  SIMD, CUDA, or OpenCL by the selected backend. Remaining par forms
  fall back to sequential loops.

  Returns (fn [& args] -> result) with lazy buffer allocation.

  Options:
    :inline?         - inline deftm calls (default true)
    :simd?           - apply SIMD optimization to parallel forms (default true)
    :dtype           - numeric dtype (:double or :float, selects overload)
    :target-device   - device-id for backend selection (e.g. :cuda:0 or :ze:0)
    :target          - :c routes to the native CPU-C backend (one fused C function
                       per deftm via Panama FFM, no per-op JVM dispatch). Bypasses
                       the bytecode/GPU path."
  [f-var & {:keys [inline? simd? dtype target-device target]
            :or {inline? true simd? true}}]
  (if (= target :c)
    ;; Native CPU-C backend: emit the whole fused body as one C function (no
    ;; per-op JVM dispatch) + Panama FFM. Bypasses the bytecode/hoist path.
    ((requiring-resolve 'raster.compiler.backend.cpu.aot/compile-aot-c)
     f-var (or dtype :double))
    (compile-aot-jvm f-var :inline? inline? :simd? simd? :dtype dtype
                     :target-device target-device)))

(defn- compile-aot-jvm
  "JVM/GPU compile-aot: bytecode (SIMD) or GPU (target-device) backend."
  [f-var & {:keys [inline? simd? dtype target-device]
            :or {inline? true simd? true}}]
  (let [;; Use the classloader that defined the target function's defrecord types.
        ;; Creating a child DCL causes class identity issues: defrecord classes
        ;; loaded through a child resolve to DIFFERENT Class objects than those
        ;; in the dispatch table, because Clojure DCL branches can't see each
        ;; other's defineClass'd classes. Using the same DCL avoids this.
        ;; The cost is compiled classes can't be GC'd separately — acceptable
        ;; since users compile a handful of functions, not thousands.
        parent-cl (let [tcl (.getContextClassLoader (Thread/currentThread))]
                    (if (instance? clojure.lang.DynamicClassLoader tcl)
                      tcl
                      ;; Find a DCL from the f-var's namespace
                      (let [root (try (.getRawRoot ^clojure.lang.Var f-var)
                                      (catch Exception _ nil))
                            var-cl (when root (.getClassLoader (class root)))]
                        (if (instance? clojure.lang.DynamicClassLoader var-cl)
                          var-cl
                          tcl))))]
    (binding [me/*compilation-classloader* parent-cl]
      (let [params (get-params f-var dtype)
            walked-body (get-walked-body f-var dtype)
            active-params (clean-params params)
            param-env (build-param-env f-var dtype)
          ;; Get the declared return type from the deftm var
            resolved-var (or (resolve-deftm-var f-var dtype) f-var)
            return-tag (:raster.core/return-tag (meta resolved-var))
            effective-dtype (or dtype (infer-dtype resolved-var)
                              ;; Scalar-only functions (no array params) default to
                              ;; double precision — the JVM's native floating-point type.
                                :double)
            ;; The deftm's defining namespace — threaded into the passes so any
            ;; re-walk (pe-rewalk) recognizes deftm calls and devirtualizes them.
            ;; Without it the re-walk runs under *ns* and silently leaves deftm
            ;; calls un-devirtualized (and bare in lifted closure helpers).
            source-ns (let [m (meta resolved-var)]
                        (or (when-let [s (:raster.core/deftm-source-ns m)]
                              (try (the-ns s) (catch Exception _ nil)))
                            (when (var? resolved-var) (.ns ^clojure.lang.Var resolved-var))))
            ;; Param annotations ({sym -> typedclojure annotation}), so the PE
            ;; re-walk reconstructs the SAME type-env as walk-1 (array :element,
            ;; (Fn ...) :fn-info) instead of a tag-only env that under-devirtualizes.
            param-annotations (let [m (meta resolved-var)
                                    ps (:raster.core/deftm-params m)
                                    anns (or (:raster.core/deftm-annotations m)
                                             (when-let [tags (:raster.core/deftm-tags m)]
                                               (mapv (fn [t] (if (= t 'Object) nil t)) tags)))]
                                (when (and ps anns (= (count ps) (count anns)))
                                  (zipmap (map #(if (symbol? %) % (symbol (name %))) ps) anns)))
            ;; Declared GPU param types from the SAME shared derivation Path A uses, so the
            ;; pipeline's opencl backend gets the deftm's declared scalar/array types (not the
            ;; emitter's name-regex guess). ONE type source across both compile entries.
            gpu-param-types (when target-device
                              (opencl-pass/derive-param-types
                               (:raster.core/deftm-params (meta resolved-var))
                               (:raster.core/deftm-tags (meta resolved-var))
                               effective-dtype))
            opts (cond-> {:inline? inline? :simd? simd? :target-device target-device
                          :active-params active-params :dtype effective-dtype}
                   param-env (assoc :param-env param-env)
                   source-ns (assoc :source-ns source-ns)
                   param-annotations (assoc :param-annotations param-annotations)
                   gpu-param-types (assoc :scalar-types (:scalar-types gpu-param-types)
                                          :array-types (:array-types gpu-param-types)))
            compile! (fn []
                       (let [raw-form (if (= 1 (count walked-body))
                                        (first walked-body)
                                        (list* 'do walked-body))
                             form (run-passes raw-form forward-passes opts)
                             hoist-opts (cond-> {:dtype effective-dtype}
                                          param-env (assoc :param-env param-env)
                                          return-tag (assoc :return-tag return-tag)
                                          ;; Thread the deftm's defining namespace so the bytecode
                                          ;; backend can resolve source-ns-local classes — e.g. a
                                          ;; value-class param container, whose `(.leaf container)`
                                          ;; field access must resolve the container's class. Without
                                          ;; it the backend falls back to *ns* and fails with
                                          ;; "Cannot resolve class for (. container leaf)".
                                          source-ns (assoc :source-ns source-ns))
                             result (if (form/binding-form? form)
                                      (closure/hoist-and-compile form active-params active-params
                                                                 hoist-opts)
                                    ;; Non-let* result: wrap in trivial let* and compile normally.
                                      (do (when (System/getProperty "raster.debug")
                                            (println "WARN: pipeline produced non-let* form, wrapping:"
                                                     (if (seq? form) (first form) (type form))))
                                          (closure/hoist-and-compile
                                           (list 'let* [] form)
                                           active-params active-params hoist-opts)))]
                       ;; Store diagnostics on the compiled function's metadata
                         (alter-meta! f-var assoc
                                      ::compiled-diagnostics
                                      (dissoc result :fn))
                         (:fn result)))]
        (if target-device
          ((requiring-resolve 'raster.gpu.ze-runtime/make-gpu-fn) compile!)
          (compile!))))))

;; ================================================================
;; Resident GPU program extraction (Option A: bound-dispatch path)
;;
;; make-gpu-fn returns a per-call STAGING fn (copy JVM arrays in, launch, copy out — every
;; call). For a straight-line fused sequence (a decoder layer: norm→quant→matmul→…) the kernel
;; order + buffer dataflow is fixed, so it can instead be bound ONCE to a session (resident
;; buffers, prepare! each kernel, record-graph! the sequence) and replayed. extract-gpu-program
;; turns the fused IR into the flat descriptor the session bound path consumes; the bound
;; machinery (bind-kernel!/launch-bound!/record-graph!) is already convention-agnostic, so map!
;; and map-void! kernels bind identically — the output is just another resident buffer.
;; ================================================================

(def ^:private gpu-invoke-heads
  '#{raster.gpu.ze-runtime/invoke-registered-map-void-kernel
     raster.gpu.ze-runtime/invoke-registered-kernel
     raster.gpu.ze-runtime/invoke-reduction-kernel})

(def ^:private gpu-array-alloc-heads
  '#{double-array float-array int-array long-array byte-array
     clojure.core/double-array clojure.core/float-array
     clojure.core/int-array clojure.core/long-array clojure.core/byte-array})

;; --- Stage A: functional reduce → resident-buffer fusion ------------------------------------
;; A par/reduce result is a SCALAR in the materialized IR, but on the resident GPU path it must
;; live in device memory (a 1-element buffer) so the per-token graph never syncs to host. This
;; pass (Futhark's reduce→broadcast→map fusion) realizes each non-escaping reduce result as a
;; 1-element buffer written by `par/reduce-into`, and FORWARD-INLINES the pure scalar chain that
;; depends on it (e.g. rms-norm's `ms`/`inv`) into the consuming kernel bodies, with the reduce
;; result read back as `(aget buf 0)`. The existing SegMap aget-array path (#37) then consumes
;; the buffer with NO new machinery. Escape analysis is the accept predicate: a reduce result
;; that reaches the host result (or any non-kernel binding — e.g. a buffer alloc size) is NOT
;; realizable, so the whole form is left unchanged (host-roundtrip fallback path).

(defn- pure-scalar-init?
  "init is a pure scalar expr — a candidate to forward-inline into a consuming kernel body —
   i.e. not a par form, a buffer alloc, or a kernel invoke."
  [init]
  (not (and (seq? init)
            (or (par/par-form? init)
                (and (symbol? (first init))
                     (or (contains? gpu-array-alloc-heads (first init))
                         (contains? gpu-invoke-heads (first init))))))))

(defn- reduce-buffer-alloc-head [dtype]
  (if (= dtype :double) 'clojure.core/double-array 'clojure.core/float-array))

(defn fuse-reduce-results
  "See block comment above. Returns the fused let* form, or the input form UNCHANGED when there
   is no realizable reduce (no par/reduce, or a reduce result genuinely escapes to host). `dtype`
   selects the 1-element buffer element type. Operates on the materialized IR (par/reduce +
   par/map! still present), so it must run before the SegOp/backend lowering."
  [form dtype]
  (if-not (and (seq? form) (#{'let* 'let} (first form)))
    form
    (let [pairs      (vec (partition 2 (second form)))
          body-forms (drop 2 form)
          body-expr  (if (= 1 (count body-forms)) (first body-forms) (cons 'do body-forms))]
      (loop [ps (seq pairs), subst {}, realized #{}, rbufs #{}, out []]
        (if-not ps
          (let [body' (util/subst-syms subst body-expr)
                ;; Soundness guard: an (aget rbuf 0) may appear ONLY inside par-form bodies
                ;; (device kernels). If a reduce buffer leaks into the host result or any
                ;; non-par kept binding (e.g. an alloc size), the reduce result genuinely
                ;; escapes — bail entirely (no fusion), preserving the host-roundtrip path.
                leaks-at-host? (fn [e] (boolean (some #(contains? (mem/sexp-free-vars e) %) rbufs)))
                leaked? (or (leaks-at-host? body')
                            (some (fn [[_ i]] (and (not (par/par-form? i)) (leaks-at-host? i))) out))]
            (if (or (empty? rbufs) leaked?)
              form
              (apply list (first form) (vec (mapcat identity out)) [body'])))
          (let [[sym init] (first ps)
                init' (util/subst-syms subst init)]
            (cond
              ;; reduce result → realize as a 1-element device buffer + a reduce-into step
              (par/par-reduce-form? init')
              (let [[_ acc init0 idx bound rbody] init'
                    buf     (gensym (str (name sym) "__rbuf"))
                    alloc-b [buf (list (reduce-buffer-alloc-head dtype) 1)]
                    red-b   [(gensym "_red__")
                             (list 'raster.par/reduce-into buf acc init0 idx bound rbody)]]
                (recur (next ps)
                       (assoc subst sym (list 'clojure.core/aget buf 0))
                       (conj realized sym) (conj rbufs buf)
                       (conj out alloc-b red-b)))
              ;; pure scalar depending on a realized value → forward-inline (drop the binding)
              (and (pure-scalar-init? init')
                   (seq (set/intersection (mem/sexp-free-vars init) realized)))
              (recur (next ps) (assoc subst sym init') (conj realized sym) rbufs out)
              ;; everything else (allocs, par/map! consumers, kernels): keep, subst applied
              :else
              (recur (next ps) subst realized rbufs (conj out [sym init'])))))))))

(defn- parse-gpu-step
  "One invoke-registered-* binding → a step record. :arrays is the full resident-buffer arg
   list in the kernel's signature order (for map! the separate output is appended, so on the
   resident path it is just another buffer). nil for an unrecognized head."
  [sym expr]
  (let [head (first expr)]
    (cond
      (= head 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel)
      (let [[_ kname arrays scalars n] expr]
        {:kernel-name kname :arrays (vec arrays) :scalars (vec scalars) :n-expr n
         :convention :map-void :returns sym})
      (= head 'raster.gpu.ze-runtime/invoke-registered-kernel)
      (let [[_ kname inputs out scalars n] expr]
        {:kernel-name kname :arrays (conj (vec inputs) out) :scalars (vec scalars) :n-expr n
         :convention :map :returns sym})
      (= head 'raster.gpu.ze-runtime/invoke-reduction-kernel)
      ;; 3-arg legacy (host-scalar return) vs 4-arg resident (writes out-buf, stays on device).
      (if (= 5 (count expr))
        (let [[_ kname inputs out-buf n] expr]
          {:kernel-name kname :arrays (vec inputs) :output out-buf :scalars [] :n-expr n
           :convention :reduce :returns sym})
        (let [[_ kname inputs n] expr]
          {:kernel-name kname :arrays (vec inputs) :scalars [] :n-expr n
           :convention :reduce :returns sym}))
      :else nil)))

(defn- contains-gpu-invoke?
  "Deep check: does form contain a gpu-invoke head anywhere? (A scalar binding whose value is
   itself computed from a kernel result is NOT a straight-line scalar let.)"
  [form]
  (let [found (volatile! false)]
    (clojure.walk/postwalk
     (fn [x] (when (and (seq? x) (symbol? (first x)) (contains? gpu-invoke-heads (first x)))
               (vreset! found true)) x)
     form)
    @found))

(defn extract-gpu-program
  "Walk a straight-line fused GPU IR into a flat program: ordered kernel steps, intermediate
   buffer allocs, and the pure scalar-let bindings that feed their sizes/counts (e.g.
   `n (* rows feat)`). Returns nil when a binding has control flow or a kernel result feeding
   scalar compute — the caller then falls back to the staging fn.

   :scalar-lets is the ordered [sym expr ...] of pure scalar bindings; size/count/scalar exprs
   are evaluated in an env that binds the deftm params AND these lets (so an intermediate sized
   `(* rows feat)` resolves)."
  [form]
  (when (and (seq? form) (#{'let* 'let} (first form)))
    (let [bindings (partition 2 (second form))
          body (last form)
          allocs (volatile! [])
          steps (volatile! [])
          scalar-lets (volatile! [])
          ok (volatile! true)]
      (doseq [[sym expr] bindings]
        (cond
          (and (seq? expr) (contains? gpu-array-alloc-heads (first expr)))
          (vswap! allocs conj {:sym sym :size-expr (second expr)})
          (and (seq? expr) (symbol? (first expr)) (contains? gpu-invoke-heads (first expr)))
          (if-let [s (parse-gpu-step sym expr)]
            (vswap! steps conj s)
            (vreset! ok false))
          ;; A pure scalar binding (no nested kernel) feeds sizes/counts — keep it.
          (not (contains-gpu-invoke? expr))
          (vswap! scalar-lets into [sym expr])
          ;; control flow / kernel-result-into-scalar → not straight-line.
          :else (vreset! ok false)))
      (when (and @ok (seq @steps))
        {:allocs @allocs :steps @steps :scalar-lets @scalar-lets :result body}))))

(defn- strip-meta
  "Drop a symbol's metadata — the walker stamps :tag, and a primitive-initialized local can't
   carry a type hint (`Can't type hint a local with a primitive initializer`), which the eval'd
   arg-fn would otherwise hit on e.g. `^long n (* (long rows) (long feat))`."
  [x]
  (if (symbol? x) (with-meta x nil) x))

(defn- expr->arg-fn
  "Compile a size/count/scalar expr into (fn [arg-vector] value), evaluating it in an env that
   binds the deftm params AND the program's pure scalar-lets (so `(* rows feat)` resolves).
   Built once at compile time; called per bind with the runtime arg values."
  [param-syms scalar-lets expr]
  (let [clean-params (mapv strip-meta param-syms)
        clean-lets (vec (map-indexed (fn [i x] (if (even? i) (strip-meta x) x)) scalar-lets))]
    (eval (list 'fn [clean-params] (list* 'let* clean-lets [expr])))))

(defn compile-gpu-program
  "Compile f-var through the SAME fused GPU pipeline as compile-aot :target-device, but return a
   RESIDENT program descriptor for the session bound-dispatch path instead of make-gpu-fn's
   per-call staging fn. Kernels are registered (globally) as a side effect of the backend pass.
   Returns nil when the fused IR is not straight-line (caller uses compile-aot staging fn).

   Descriptor:
     {:dtype :float
      :all-params  [a out n]        ; deftm param order (the resident fn's arg order)
      :array-params [a out]         ; array params (resident buffers, up/downloaded)
      :scalar-params [n]
      :allocs [{:sym b :size-fn (fn [args] int)}]   ; intermediate scratch buffers
      :steps  [{:kernel-name str :arrays [syms] :n-fn (fn [args] int)
                :scalar-fns [(fn [args] v) ...] :convention :map-void|:map :phase kw}]
      :result-sym sym}"
  [f-var device-id & {:keys [dtype]}]
  (let [resolved-var (or (resolve-deftm-var f-var dtype) f-var)
        params       (get-params f-var dtype)
        walked-body  (get-walked-body f-var dtype)
        active-params (clean-params params)
        param-env    (build-param-env f-var dtype)
        effective-dtype (or dtype (infer-dtype resolved-var) :double)
        source-ns    (let [m (meta resolved-var)]
                       (or (when-let [s (:raster.core/deftm-source-ns m)]
                             (try (the-ns s) (catch Exception _ nil)))
                           (when (var? resolved-var) (.ns ^clojure.lang.Var resolved-var))))
        ;; --- soa-lower (resident GPU value-type explosion) ----------------------------------
        ;; Run the front half to :materialized, explode any Params-container / value-type param
        ;; into per-field array params, then resume to the backend. Gated on a value-type param
        ;; so flat-param deftms are untouched (their eff-param-specs == param-specs). The
        ;; descriptor's params + the backend's array types are derived from the EXPLODED leaves.
        d-params*   (:raster.core/deftm-params (meta resolved-var))
        d-tags*     (:raster.core/deftm-tags (meta resolved-var))
        param-specs (mapv (fn [p t] {:sym (if (symbol? p) (with-meta p nil) (symbol (name p)))
                                     :tag (when t (symbol t))})
                          d-params* d-tags*)
        value-reg   @types/soa-registry
        value-fn?   (boolean (some #(contains? value-reg (:tag %)) param-specs))
        pre-opts (cond-> {:inline? true :simd? false :target-device device-id
                          :active-params active-params :dtype effective-dtype}
                   param-env (assoc :param-env param-env)
                   source-ns (assoc :source-ns source-ns))
        raw-form (if (= 1 (count walked-body)) (first walked-body) (cons 'do walked-body))
        form-mat (run-passes raw-form gpu-resident-pre-soa-passes pre-opts)
        {form-soa :body eff-param-specs :params}
        (if value-fn?
          (soa-lower/soa-lower form-mat param-specs)
          {:body form-mat :params param-specs})
        all-params (mapv :sym eff-param-specs)
        gpu-param-types (opencl-pass/derive-param-types
                         (mapv :sym eff-param-specs) (mapv :tag eff-param-specs) effective-dtype)
        array-param-set (set (keys (:array-types gpu-param-types)))
        array-params (filterv #(contains? array-param-set %) all-params)
        scalar-params (filterv #(not (contains? array-param-set %)) all-params)
        post-opts (cond-> {:inline? true :simd? false :target-device device-id
                           :active-params active-params :dtype effective-dtype}
                    param-env (assoc :param-env param-env)
                    source-ns (assoc :source-ns source-ns)
                    gpu-param-types (assoc :scalar-types (:scalar-types gpu-param-types)
                                           :array-types (:array-types gpu-param-types)))
        ;; Stage A: realize non-escaping par/reduce results as device-resident 1-elem buffers and
        ;; fuse their dependent scalar chains into consuming kernels (before SegOp/backend lowering).
        _pre-fuse form-soa
        form-soa (fuse-reduce-results form-soa effective-dtype)
        _ (when (System/getProperty "raster.debug.fuse")
            (binding [*out* *err*]
              (println "=== MATERIALIZED (pre-fuse) ===") (println (pr-str _pre-fuse))
              (println "=== POST-FUSE ===") (println (pr-str form-soa))
              (println "=== END-FUSE-DUMP ===")))
        form (run-passes form-soa gpu-resident-post-soa-passes post-opts :materialized)
        prog (extract-gpu-program form)
        scalar-lets (:scalar-lets prog)
        ;; Default buffer roles for the residency layer: an array param WRITTEN by any kernel
        ;; (its name is in some kernel's :written-arrays) defaults to :output (downloaded after
        ;; replay); a read-only param defaults to :input (re-uploaded per call). Intermediates are
        ;; :scratch (never moved). The CALLER overrides read-only params that are fixed across
        ;; calls to :constant (weights — uploaded once at bind) and persistent written state to
        ;; :state (KV cache — never downloaded): that cross-call axis isn't a single-program
        ;; property (escape analysis can't see it), so it's declared, not derived.
        reg-entry (when prog (requiring-resolve 'raster.gpu.ze-runtime/kernel-registry-entry))
        written-params (when prog
                         (reduce (fn [acc s]
                                   (let [ki (reg-entry (:kernel-name s))]
                                     (into acc (map (comp symbol name) (:written-arrays ki)))))
                                 #{} (:steps prog)))
        array-roles (when prog
                      (into {} (map (fn [p] [p (if (contains? written-params p) :output :input)])
                                    array-params)))]
    (when prog
      {:dtype effective-dtype
       :all-params all-params
       :array-params array-params
       :array-roles array-roles
       :scalar-params scalar-params
       :allocs (mapv (fn [{:keys [sym size-expr]}]
                       {:sym sym :size-fn (expr->arg-fn all-params scalar-lets size-expr)})
                     (:allocs prog))
       :steps (mapv (fn [i {:keys [kernel-name arrays scalars n-expr convention output]}]
                      {:kernel-name kernel-name
                       :arrays arrays
                       ;; :reduce steps carry the resident 1-elem output buffer (sym keyword) so
                       ;; bind-program! wires it like a map output (it lives in :allocs as scratch).
                       :output (when output (keyword (name output)))
                       :n-fn (expr->arg-fn all-params scalar-lets n-expr)
                       ;; Each scalar typed by the deftm's DECLARED type (the invoke scalars are
                       ;; the kernel's scalar-param syms = deftm params), so an int param (e.g.
                       ;; `features`) is bound as :int, not mis-typed float (→ launch error).
                       :scalar-specs (mapv (fn [s]
                                             {:type (or (and (symbol? s)
                                                             (get (:scalar-types gpu-param-types) s))
                                                        (if (= effective-dtype :double) :double :float))
                                              :value-fn (expr->arg-fn all-params scalar-lets s)})
                                           scalars)
                       :convention convention
                       :phase (keyword (str "gpu-step-" i))})
                    (range) (:steps prog))
       :result-sym (:result prog)})))

(defn- wasm-kernel-spec
  "Front-half for one deftm var → {:name :params :ir :simd?} ready for the wasm
   emitter: walker → forward-passes → soa-lower (value-type SoA → per-field arrays)."
  [f-var dtype name wasm-simd?]
  (let [resolved      (or (resolve-deftm-var f-var dtype) f-var)
        walked-body   (get-walked-body f-var dtype)
        active-params (clean-params (get-params f-var dtype))
        param-env     (build-param-env f-var dtype)
        source-ns     (let [m (meta resolved)]
                        (or (when-let [s (:raster.core/deftm-source-ns m)]
                              (try (the-ns s) (catch Exception _ nil)))
                            (when (var? resolved) (.ns ^clojure.lang.Var resolved))))
        opts (cond-> {:inline? true :simd? false :dtype dtype :active-params active-params}
               param-env (assoc :param-env param-env)
               source-ns (assoc :source-ns source-ns))
        raw-form (if (= 1 (count walked-body)) (first walked-body) (list* 'do walked-body))
        ;; wasm inline budget: keep large straight-line callees as wasm functions
        ;; (deftm->function) rather than inlining. 1200 nodes is generous — above
        ;; every current kernel (biggest straight-line callee perlin3d ≈ 368) and
        ;; aligned with V8's ~1250-node wasm inline-eligibility, so only genuinely
        ;; large callees outline; fusable (loop/par) callees always inline. JVM/GPU
        ;; leave the limit nil (inline all + split.clj). See .internal notes.
        form (binding [inline/*inline-size-limit* (or inline/*inline-size-limit* 1200)]
               (run-passes raw-form forward-passes opts))
        d-params (:raster.core/deftm-params (meta resolved))
        d-tags   (:raster.core/deftm-tags (meta resolved))
        return-tag (:raster.core/return-tag (meta resolved))
        param-specs (mapv (fn [p t] {:sym (with-meta (if (symbol? p) p (symbol (name p))) nil)
                                     :tag (symbol t)})
                          d-params d-tags)
        value-reg @raster.compiler.core.types/soa-registry
        ;; scalar value-type kernel (value-type param or return) vs SoA-array kernel
        value-fn? (or (some #(contains? value-reg (:tag %)) param-specs)
                      (contains? value-reg return-tag))
        {lowered-body :body lowered-params :params}
        (if value-fn?
          (soa-lower/lower-value-fn form param-specs return-tag)
          (soa-lower/soa-lower form param-specs))
        ;; Macroexpand at the last moment (mirrors the JVM bytecode backend): the
        ;; walker leaves high-level forms (let/loop/case/cond/when/or/and) for the
        ;; passes to match; the emitter only needs the ~12 special forms. Bind *ns*
        ;; to the kernel's source ns so source-local macros resolve. Meta is
        ;; preserved (:raster.type/tag, :raster.op/original).
        ;; Keep `dotimes` un-expanded: the emitter's SIMD vectorizer matches its
        ;; counted-loop shape directly (generic loop* expansion would defeat it).
        wasm-skip-head? (fn [h] (= "dotimes" (clojure.core/name h)))
        expanded-body (binding [*ns* (or source-ns *ns*)]
                        (mex/resugar-interop
                         (mex/macroexpand-all-preserving lowered-body wasm-skip-head?)))]
    {:name (or name (str (:name (meta resolved))))
     :params lowered-params :ir expanded-body :simd? wasm-simd?
     ;; the ORIGINAL (pre-lowering) semantic signature — for the cljs marshaling
     ;; codegen: which params/return are value types (and their field order).
     :sig {:param-names (mapv :sym param-specs)   ; already plain symbols
           :param-tags  (mapv :tag param-specs)
           :return-tag  return-tag}}))

(defn- intrinsic-invk?
  "True if an .invk node calls a registered intrinsic (raster.numeric/Math/…) —
   these are emitted inline by the wasm backend, never as a `call`."
  [node]
  (and (seq? node) (= '.invk (first node))
       (let [op (:raster.op/original (meta node))]
         (and (symbol? op) (ix/canonical op)))))

(defn- discover-callees
  "Set of impl-syms of non-intrinsic deftm callees that survive (un-inlined) in ir.
   Each becomes a sibling wasm function invoked via `call` (recursion today;
   size-driven outlining later)."
  [ir]
  (let [out (atom #{})]
    (walk/postwalk
     (fn [x]
       (when (and (seq? x) (= '.invk (first x)) (not (intrinsic-invk? x)))
         (swap! out conj (second x)))
       x)
     ir)
    @out))

(defn compile-wasm
  "Compile a deftm var to a WebAssembly module (Track A — browser/WASI target).
   Reuses compile-aot's front-half (walker → forward-passes) to produce the same
   post-pass IR, then emits a .wasm via backend/wasm/emit instead of JVM bytecode.

   Returns {:bytes byte[] :name str :param-types [valtype-kw...] :params [{:sym :tag}]}.
   The exported fn takes the params in order; array params are (ptr) byte-offsets
   into the exported `memory`, with element data the caller writes/reads through a
   typed-array view (see cljs-sandbox for the cljs side).

   v1: scalar, single-function, param-only kernels (no hoisted intermediate
   buffers); `:simd? false` is forced (the JVM Vector-API SIMD lowering is
   JVM-only — a wasm-SIMD128 path is a later increment)."
  [f-var & {:keys [dtype name wasm-simd?] :or {dtype :double}}]
  (let [root  (wasm-kernel-spec f-var dtype name wasm-simd?)
        rv    (or (resolve-deftm-var f-var dtype) f-var)
        root-rkey (symbol (str (.ns ^clojure.lang.Var rv)) (clojure.core/name (.sym ^clojure.lang.Var rv)))
        ;; transitive discovery of non-inlined (non-intrinsic) deftm callees that
        ;; survive in the IR — recursion guard keeps recursive calls, and (later)
        ;; the outline pass will introduce others. Each becomes a sibling wasm fn.
        result (loop [pending (vec (discover-callees (:ir root)))
                      by-key  {}            ; callee impl-sym → kernel spec
                      root-ck nil]          ; root's own :call-key if self-recursive
                 (if (empty? pending)
                   {:by-key by-key :root-ck root-ck}
                   (let [impl (peek pending), more (pop pending)
                         rkey (inline/recursion-key impl)]
                     (cond
                       (= rkey root-rkey)            ; self-call → resolves to the root fn
                       (recur more by-key impl)
                       (contains? by-key impl)       ; already compiled
                       (recur more by-key root-ck)
                       :else
                       (let [cv   (resolve rkey)
                             spec (assoc (wasm-kernel-spec cv dtype (str (clojure.core/name (inline/recursion-key impl))) wasm-simd?)
                                         :call-key impl :export? false)]
                         (recur (into more (discover-callees (:ir spec)))
                                (assoc by-key impl spec) root-ck))))))
        callees (vals (:by-key result))
        root*   (cond-> (assoc root :export? true)
                  (:root-ck result) (assoc :call-key (:root-ck result)))
        module  (wasm-emit/compile-module (into [root*] callees))
        entry   (first (:exports module))]
    {:bytes (:bytes module)
     :name (:name entry) :param-types (:param-types entry)
     :result-types (:result-types entry) :sig (:sig root)}))

(defn compile-wasm-module
  "Compile several deftm vars into ONE wasm module that shares a single linear
   memory — so a program's data lives in one buffer addressed by all the exported
   kernels (vs compile-wasm's one-module-per-kernel). Each spec is either a var or
   {:var v :name str :dtype kw :wasm-simd? bool}.
   Returns {:bytes byte[] :exports [{:name :param-types :result-types :sig} …]}
   where :sig is the original semantic signature (for cljs marshaling codegen)."
  [specs]
  (let [kspecs (mapv (fn [s]
                       (let [s (if (map? s) s {:var s})]
                         (wasm-kernel-spec (:var s) (or (:dtype s) :double) (:name s) (:wasm-simd? s))))
                     specs)
        sig-by-name (into {} (map (juxt :name :sig) kspecs))
        module (wasm-emit/compile-module kspecs)]
    (update module :exports
            (fn [exs] (mapv #(assoc % :sig (sig-by-name (:name %))) exs)))))

(defn compile-wgsl
  "Compile a deftm var to a WGSL compute shader (Track C — WebGPU compute).
   WebGPU has no f64, so the front-half runs at :float (f32) and the emitter
   targets f32 storage buffers. Returns {:wgsl str :array-params :scalar-params
   :n-sym :workgroup} — the host binds arrays as f32 storage buffers in order,
   then a uniform struct (scalars + _n). v1: elementwise maps."
  [f-var & {:keys [name]}]
  (wgsl-emit/compile-kernel (wasm-kernel-spec f-var :float name false)))

;; ================================================================
;; Typed gradient helpers (ftm-based, primitive fast-path)
;; ================================================================

(defn- get-resolved-param-tags
  "Get parameter tags from a deftm var. Returns vector of tags like ['double 'double ...]."
  [f-var]
  (let [resolved (or (resolve-deftm-var f-var) f-var)]
    (:raster.core/deftm-tags (meta resolved))))

(defn- all-primitive-params?
  "True if all param tags are JVM primitives (suitable for typed gradient)."
  [tags]
  (and (seq tags)
       (every? #{'double 'long 'float 'int 'Double 'Long 'Float 'Integer} tags)))

;; ================================================================
;; Pipeline: value + gradient (vjp-style, returns typed fn)
;; ================================================================

;; ================================================================
;; Diagnostics
;; ================================================================

(defn show-pipeline
  "Show intermediate forms at each pipeline stage.

  Returns a map with :walked and all stage forms/stats from the unified
  pass vector. Stages are keyed by each pass's declared output dialect.

  Options:
    :simd?           - apply SIMD optimization (default true)
    :target-device   - device-id for backend (e.g. :cuda:0)
    :inline?         - inline deftm calls (default true)"
  [f-var & {:keys [mode inline?
                   simd? target-device dtype]
            :or {mode :forward
                 simd? true}}]
  (let [inline? (if (nil? inline?) true inline?)
        params (get-params f-var dtype)
        active-params (clean-params params)
        walked-body (get-walked-body f-var dtype)
        raw-form (if (= 1 (count walked-body))
                   (first walked-body)
                   (list* 'do walked-body))
        passes forward-passes
        resolved-var (or (resolve-deftm-var f-var dtype) f-var)
        effective-dtype (or dtype (infer-dtype resolved-var)
                            ;; Scalar-only functions default to double precision
                            :double)
        param-env (build-param-env f-var dtype)
        opts (cond-> {:inline? inline?
                      :active-params active-params
                      :simd? simd? :target-device target-device
                      :dtype effective-dtype}
               param-env (assoc :param-env param-env))
        {:keys [stages stats]} (run-passes-diagnostic raw-form passes opts)
        backend-type (get-in stages [:backend-type])]
    (-> {:walked raw-form
         :backend (or backend-type (if simd? :simd :scalar))}
        (merge (dissoc stages :backend-type :cuda-result :kernels))
        (merge (into {} (map (fn [[k v]] [(keyword (str (name k) "-stats")) v]) stats)))
        (cond->
         (get-in stages [:kernels]) (assoc :kernels (get-in stages [:kernels]))
         (get-in stages [:cuda-result]) (assoc :cuda-result (get-in stages [:cuda-result]))))))

;; ================================================================
;; explain-pipeline — human-readable stage-by-stage summary
;; ================================================================

(defn- pprint-form
  "Pretty-print a form to a string, truncating at max-lines."
  [form max-lines]
  (let [s (with-out-str (binding [pp/*print-right-margin* 80]
                          (pp/pprint form)))
        lines (clojure.string/split-lines s)]
    (if (<= (count lines) max-lines)
      s
      (str (clojure.string/join "\n" (take max-lines lines))
           "\n  [... " (- (count lines) max-lines) " more lines]\n"))))

(defn- print-stage
  "Print a single pipeline stage. Returns true if printed (not unchanged)."
  [n label form prev-form & {:keys [stats always?]}]
  (println (str "\n--- Stage " n ": " label " ---"))
  (if (and (not always?) prev-form (= form prev-form))
    (do (println "  [unchanged]") false)
    (do
      (when stats
        (println (str "  Stats: " (pr-str stats))))
      (print (pprint-form form 30))
      true)))

(defn explain-pipeline
  "Print human-readable stage-by-stage compilation explanation.
  Shows what each pass does, diffs between stages, and statistics.
  Same options as show-pipeline.

  Returns the stage map (same as show-pipeline) for programmatic use."
  [f-var & {:keys [mode] :or {mode :forward} :as opts}]
  (let [all-opts (merge {:mode mode} opts)
        result (apply show-pipeline f-var (mapcat identity all-opts))
        resolved (or (resolve-deftm-var f-var (:dtype opts)) f-var)
        var-name (str (ns-name (:ns (meta resolved))) "/" (:name (meta resolved)))
        source-body (:raster.core/deftm-source-body (meta resolved))
        backend (:backend result)]
    ;; Header
    (println (str "\n=== Pipeline: " var-name
                  " (" (name mode)
                  (case backend
                    :simd    ", SIMD"
                    :cuda    ", CUDA"
                    :opencl  ", OpenCL/L0"
                    :scalar  ", scalar"
                    "")
                  ") ==="))
    ;; Hardware info
    (hardware/init!)
    (let [dev (hardware/default-device)]
      (when dev
        (println (str "Hardware: " (:name dev)
                      (when-let [lanes (hardware/simd-lanes (:id dev) :double)]
                        (str ", " lanes " double lanes"))
                      (when-let [flanes (hardware/simd-lanes (:id dev) :float)]
                        (str ", " flanes " float lanes"))))))

    ;; Source (if available)
    (when source-body
      (println "\n--- Stage 0: Source ---")
      (print (pprint-form (if (= 1 (count source-body))
                            (first source-body)
                            (cons 'do source-body))
                          30)))

    ;; Unified stage printing — walks the pass vector
    (let [passes forward-passes
          stage-labels {:lower "Lower" :fixpoint "Fixpoint"
                        :dce "DCE" :buffer-fuse "Buffer Fusion"
                        :soac-fuse "SOAC Fusion"
                        :materialize "Materialize"
                        :segop-lower "SegOp Lower"
                        :compound-detect "Compound Detect"
                        :backend "Backend"
                        :resolve-alength "Resolve Alength"
                        :mem-merge "Memory Merge"
                        ;; Building blocks (for custom pipelines)
                        :expand "Expand" :mode-select "Mode Select"
                        :gpu-plan "GPU Plan"}
          walked (:walked result)
          prev (atom walked)]
      (print-stage 0 "Walker" walked nil :always? true)
      (doseq [[n pass-key] (map-indexed vector passes)]
        (let [stage-key (:to (get pass-specs pass-key))
              form (get result stage-key)
              label (get stage-labels pass-key (name pass-key))
              stats-key (keyword (str (name stage-key) "-stats"))
              stats (get result stats-key)]
          (when form
            (print-stage (inc n) label form @prev :stats stats)
            (reset! prev form))))
      ;; Kernel info
      (when-let [kernels (:kernels result)]
        (doseq [k kernels]
          (when-let [occ (:occupancy-estimate k)]
            (println (str "  Kernel " (:kernel-name k)
                          ": occupancy=" (format "%.0f%%" (* 100.0 (:occupancy occ)))
                          " blocks/SM=" (:blocks-per-sm occ)
                          " limited-by=" (name (:limiting-factor occ))))))))

    (println)
    result))

;; ================================================================
;; Type diagnostics
;; ================================================================

(defn explain-types
  "Show type information for all bindings in a deftm's walked body.
  Highlights which bindings have fn-info (typed function parameters),
  which calls were devirtualized (.invk), and which fell back to
  plain dispatch.

  Usage:
    (pipeline/explain-types #'my-fn)
    (pipeline/explain-types #'ode/solve :dtype [DP5 ODEProblem double])"
  [f-var & {:keys [dtype]}]
  (let [resolved (or (when dtype (resolve-deftm-var f-var dtype)) f-var)
        m (meta resolved)
        walked (or (:raster.core/deftm-walked-body-typed m)
                   (:raster.core/deftm-walked-body m)
                   (rcore/ensure-walked-body! resolved))]
    (when-not walked
      (println "No walked body found for" f-var))
    (println (str "\n=== Type Analysis: " (:name m) " ===\n"))
    ;; Collect all let bindings with their type metadata
    (let [bindings (atom [])
          invk-calls (atom [])
          plain-calls (atom [])]
      (clojure.walk/postwalk
       (fn [form]
          ;; Track let bindings
         (when (form/binding-form? form)
           (doseq [[sym init] (partition 2 (second form))]
             (when (symbol? sym)
               (swap! bindings conj
                      {:sym sym
                       :tag (or (:raster.type/tag (meta sym)) (:tag (meta sym)))
                       :fn-info (:raster.type/fn-info (meta sym))
                       :source (:raster.type/source (meta sym))}))))
          ;; Track .invk calls (devirtualized)
         (when (and (seq? form) (= '.invk (first form)))
           (swap! invk-calls conj
                  {:impl (second form)
                   :op (or (:raster.op/original (meta form)) (:op (meta form)))
                   :arity (- (count form) 2)}))
          ;; Track plain deftm calls (not devirtualized)
         (when (and (seq? form) (symbol? (first form))
                    (namespace (first form))
                    (not= '.invk (first form))
                    (not (contains? #{'clojure.core/aget 'clojure.core/aset
                                      'clojure.core/nth 'clojure.core/+
                                      'clojure.core/* 'clojure.core/- 'clojure.core//}
                                    (first form))))
           (swap! plain-calls conj
                  {:fn (first form) :arity (dec (count form))}))
         form)
       walked)
      ;; Print bindings
      (println "Bindings:")
      (doseq [{:keys [sym tag fn-info source]} @bindings]
        (println (format "  %-30s tag: %-20s fn-info: %-5s source: %s"
                         sym (or tag "-") (boolean fn-info) (or source "-"))))
      ;; Print devirtualized calls
      (println (str "\nDevirtualized calls (" (count @invk-calls) "):"))
      (doseq [{:keys [op impl arity]} @invk-calls]
        (println (format "  .invk %-40s op: %s (arity %d)" impl (or op "?") arity)))
      ;; Print non-devirtualized calls
      (when (seq @plain-calls)
        (println (str "\nPlain dispatch calls (" (count @plain-calls) ") — potential optimization targets:"))
        (doseq [{:keys [fn arity]} @plain-calls]
          (println (format "  (%s ...) arity %d" fn arity))))
      (println))))
