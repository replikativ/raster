(ns raster.compiler.passes.scalar.inline
  "Inlining and lowering helpers for scalar compiler passes.

	This namespace owns cross-function inlining, pre-AD lowering, and
	post-AD backend expansion. It does not perform buffer reuse."
  (:require [clojure.string :as str]
            [raster.ad.templates :as rt]
            [raster.compiler.ad.flatten :as flatten]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]
            [raster.compiler.passes.scalar.cse :as cse]))

(def ^:private call-head util/call-head)
(def ^:private call-args util/call-args)

(defn- inlinable-body?
  "Check if a walked body form is suitable for inlining.
   Only forms with decomposable structure (let*, loop, dotimes, do,
   .invk, par) are inlinable. Bare function calls are not — they
   need let* wrapping for the inliner to decompose."
  [body]
  (and (seq? body)
       (contains? #{:binding :scope :do :invk :par}
                  (:kind (form/form-info body)))))

(def ^:private prim-or-array-tags
  "Tags that are safe for inlining even when body has scoped forms.
   Reference-typed params (Dual, Complex, records) may have host interop
   (.v, .partials) in the body that the bytecoder can't resolve after
   argument substitution."
  #{'double 'long 'float 'int 'doubles 'floats 'longs 'ints 'objects})

(def ^:private trivial-cast?
  "Cast/coercion calls that are cheap to duplicate — single arg, no allocation."
  #{'long 'double 'float 'int 'byte 'short 'char
    'unchecked-long 'unchecked-double 'unchecked-float 'unchecked-int
    'clojure.core/long 'clojure.core/double 'clojure.core/float 'clojure.core/int
    'clojure.core/byte 'clojure.core/short 'clojure.core/char})

(defn- needs-arg-lift?
  "True if arg is a non-trivial expression that should be lifted to a let binding
   before parameter substitution. Prevents duplication when callee params appear
   multiple times (e.g. inside par/pmap bodies where CSE can't reach)."
  [arg]
  (and (seq? arg)
       (let [head (first arg)]
         (not (trivial-cast? head)))))

(defn- safe-to-inline?
  "Check if a resolved deftm body is safe to inline.
   Previously rejected bodies with scopes + non-primitive params, but
   ANF argument lifting now creates typed bindings for non-primitive args,
   preserving type info through substitution. Always returns true."
  [_deftm-info]
  true)

(defn- try-resolve-walked-body
  "Try to extract walked body from a var. Returns {:params :walked-body :tags} or nil.
  Checks both var metadata (normal deftm) and the deref'd value's metadata
  (for value+grad/grad results stored in vars).
  :tags is the vector of deftm parameter type tags (e.g. ['doubles 'double ...])."
  [v]
  (let [metadata (meta v)
        ;; Also check the fn object's metadata (for value+grad results)
        val-metadata (when (instance? clojure.lang.Var v)
                       (try (meta @v) (catch Exception _ nil)))
        walked-body (or (:raster.core/deftm-walked-body-typed metadata)
                        (:raster.core/deftm-walked-body metadata)
                        (:raster.core/deftm-walked-body-typed val-metadata)
                        (:raster.core/deftm-walked-body val-metadata))
        params (or (:raster.core/deftm-params metadata)
                   (:raster.core/deftm-params val-metadata))
        tags (or (:raster.core/deftm-tags metadata)
                 (:raster.core/deftm-tags val-metadata))
        return-tag (or (:raster.core/return-tag metadata)
                       (:raster.core/return-tag val-metadata))]
    (when (and walked-body params (not (:no-inline metadata)))
      (let [effective-wb (if (> (count walked-body) 1)
                           [(apply list 'do walked-body)]
                           walked-body)]
        (when (inlinable-body? (first effective-wb))
          (let [source-ns-sym (symbol (str (.name (.ns v))))
                source-ns (try (the-ns source-ns-sym) (catch Exception _ nil))
                param-set (set (map #(if (symbol? %) % (symbol (name %))) params))
                ;; Qualify non-primitive tags to fully-qualified class names.
                ;; e.g. DP5Cache → raster.ode.core.DP5Cache so the bytecode
                ;; backend can resolve them in any namespace context.
                prim-tags #{'double 'long 'float 'int 'Double 'Long 'Float 'Integer
                            'doubles 'floats 'longs 'ints 'objects}
                qualified-tags
                (when (and tags source-ns)
                  (mapv (fn [t]
                          (if (or (nil? t) (prim-tags t)
                                  (.contains (str t) "."))
                            t ;; already qualified or primitive
                            (if-let [cls (try (ns-resolve source-ns (symbol (str t)))
                                              (catch Exception _ nil))]
                              (if (class? cls)
                                (symbol (.getName ^Class cls))
                                t)
                              t)))
                        tags))]
            (if source-ns
              {:params params
               :tags (or qualified-tags tags)
               :return-tag return-tag
               :walked-body (mapv #(inf/qualify-body-symbols % source-ns param-set)
                                  effective-wb)}
              {:params params :tags tags :return-tag return-tag :walked-body effective-wb})))))))

(defn- try-resolve-deftm
  "Try to resolve a symbol to a deftm var with an inlinable walked body.
   When arg-tags are provided, selects the matching overload from multi-method
   dispatch tables. Without arg-tags, only inlines single-method dispatch."
  ([sym] (try-resolve-deftm sym nil))
  ([sym arg-tags]
   (when (qualified-symbol? sym)
     (let [name-str (name sym)
           base-sym (if (.endsWith ^String name-str "-impl")
                      (symbol (namespace sym) (subs name-str 0 (- (count name-str) 5)))
                      sym)
           dispatch-no-inline?
           (let [base-name (name base-sym)]
             (when-let [idx (str/index-of base-name "_m_")]
               (let [dispatch-sym (symbol (namespace base-sym) (subs base-name 0 idx))]
                 (when-let [dispatch-var (try (resolve dispatch-sym) (catch Exception _ nil))]
                   (:no-inline (meta dispatch-var))))))]
       (when-not dispatch-no-inline?
         (when-let [v (try (resolve base-sym) (catch Exception _ nil))]
           (when-not (:no-inline (meta v))
             (or
              (try-resolve-walked-body v)
              (when-let [dispatch-table (:raster.core/dispatch-table (meta v))]
                (let [all-methods (mapcat identity (vals @dispatch-table))
                      has-parametric? (when-let [pr (requiring-resolve
                                                     'raster.compiler.core.dispatch/parametric-registry)]
                                        (contains? @@pr (symbol (namespace base-sym) (name base-sym))))
                      ;; Type-safe overload selection with parametric specialization.
                      ;; Single-method fallback only when function is NOT parametric
                      ;; (i.e., the one method IS the only possible overload).
                      match (or (when (seq arg-tags)
                                  (or (first (filter #(= (vec arg-tags) (vec (:tags %))) all-methods))
                                      ;; No match — try parametric specialization
                                      (when-let [try-ps (requiring-resolve
                                                         'raster.compiler.core.inference/try-parametric-specialize!)]
                                        (when (try-ps (symbol (namespace base-sym) (name base-sym)) arg-tags)
                                          (let [new-methods (mapcat identity
                                                                    (vals @(:raster.core/dispatch-table (meta v))))]
                                            (first (filter #(= (vec arg-tags) (vec (:tags %))) new-methods)))))))
                                ;; Single method fallback: safe when no other overloads exist
                                ;; AND function is not parametric. Parametric functions
                                ;; may need a different specialization that hasn't been
                                ;; generated yet — inlining the only existing overload
                                ;; (e.g. doubles) into a float pipeline causes VerifyErrors.
                                (when (and (= 1 (count all-methods))
                                           (not has-parametric?))
                                  (first all-methods)))]
                  (when match
                    (let [backing-sym (symbol (namespace base-sym)
                                              (str (types/mangle (name base-sym)
                                                                 (:tags match))))
                          backing-var (try (resolve backing-sym) (catch Exception _ nil))]
                      (when backing-var
                        (try-resolve-walked-body backing-var))))))))))))))

(def ^:private alloc-sym->array-tag op/alloc-sym->array-tag)

(def ^:private devirt->dispatch
  {'clojure.core/aget 'raster.arrays/aget
   'clojure.core/aset 'raster.arrays/aset
   'clojure.core/alength 'raster.arrays/alength})

(defn- subst-sym-leaf
  "Substitute a single symbol according to smap, preserving metadata."
  [smap expr]
  (let [replacement (get smap expr expr)]
    (if (and (not (identical? replacement expr))
             (symbol? replacement)
             (meta expr))
      (vary-meta replacement merge (meta expr))
      replacement)))

(defn- smap-captures?
  "Check if any value in smap is equal to the given scope variable symbol.
   This detects when substitution would introduce a captured reference,
   e.g., smap {gamma → g} entering (dotimes [g ...]) would make
   'gamma' references resolve to the loop var instead of the outer 'g'."
  [smap scope-var]
  (some #(= scope-var %) (vals smap)))

(defn- alpha-rename-scope-var
  "When a scope variable collides with a substitution target value,
   alpha-rename it: generate a fresh name and add the renaming to smap.
   The smap values that point to scope-var are NOT changed — they refer to
   the outer scope's binding (which is what we want).
   Returns [new-var updated-smap]."
  [smap scope-var]
  (if (smap-captures? smap scope-var)
    (let [fresh (with-meta (gensym (str (name scope-var) "_α_"))
                  (meta scope-var))
          ;; Don't change existing smap values — they reference the OUTER scope-var.
          ;; Just dissoc scope-var as key (it's rebound in this scope) and add
          ;; the renaming so references to the loop/let var use the fresh name.
          inner-smap (assoc (dissoc smap scope-var) scope-var fresh)]
      [fresh inner-smap])
    [scope-var (dissoc smap scope-var)]))

(defn- subst-syms
  "Recursively substitute symbols in expr according to smap.
   Scope-aware: when entering a scope form (dotimes, loop, let, fn, par),
   handles two kinds of variable capture:
   1. Key capture: scope var is a KEY in smap → dissoc it (don't substitute
      what the scope rebinds).
   2. Value capture: scope var appears as a VALUE in smap → alpha-rename
      the scope var to prevent outer substitution targets from being
      shadowed by the scope var."
  [smap expr]
  (if (empty? smap)
    expr
    (cond
      (symbol? expr)
      (subst-sym-leaf smap expr)

      (seq? expr)
      (let [head (first expr)
            rebuild (fn [children]
                      (if-let [m (meta expr)]
                        (with-meta (apply list children) m)
                        (apply list children)))]
        (cond
          ;; dotimes: (dotimes [var bound] body...)
          (= 'dotimes head)
          (let [[_ [var bound] & body] expr
                new-bound (subst-syms smap bound)
                [new-var inner-smap] (alpha-rename-scope-var smap var)]
            (rebuild (cons 'dotimes
                           (cons [new-var new-bound]
                                 (map (partial subst-syms inner-smap) body)))))

          ;; loop/loop*: (loop [v1 init1 v2 init2 ...] body...)
          (contains? #{'loop 'loop*} head)
          (let [[_ bindings-vec & body] expr
                pairs (partition 2 bindings-vec)
                ;; init exprs see outer scope (not loop vars)
                new-inits (map (fn [[_ init]] (subst-syms smap init)) pairs)
                ;; Alpha-rename loop vars that collide with smap values
                loop-vars (map first pairs)
                [renamed-vars inner-smap]
                (reduce (fn [[vars sm] v]
                          (let [[new-v sm'] (alpha-rename-scope-var sm v)]
                            [(conj vars new-v) sm']))
                        [[] smap]
                        loop-vars)
                new-bindings (vec (mapcat vector renamed-vars new-inits))]
            (rebuild (cons head (cons new-bindings
                                      (map (partial subst-syms inner-smap) body)))))

          ;; let/let*: sequential scoping
          (contains? #{'let 'let*} head)
          (let [[_ bindings-vec & body] expr
                pairs (partition 2 bindings-vec)
                [new-pairs final-smap]
                (reduce (fn [[acc sm] [v init]]
                          (let [substed-init (subst-syms sm init)
                                [new-v sm'] (alpha-rename-scope-var sm v)]
                            [(conj acc [new-v substed-init]) sm']))
                        [[] smap]
                        pairs)
                new-bindings (vec (mapcat identity new-pairs))]
            (rebuild (cons head (cons new-bindings
                                      (map (partial subst-syms final-smap) body)))))

          ;; fn/fn*: (fn* [params...] body...)
          (contains? #{'fn 'fn*} head)
          (let [[_ params & body] expr
                param-vars (filter symbol? (if (vector? params) params []))
                [renamed-params inner-smap]
                (reduce (fn [[ps sm] v]
                          (let [[new-v sm'] (alpha-rename-scope-var sm v)]
                            [(conj ps new-v) sm']))
                        [[] smap]
                        param-vars)
                new-params (if (vector? params)
                             (vec (concat renamed-params
                                          (remove symbol? params)))
                             params)]
            (rebuild (cons head (cons new-params
                                      (map (partial subst-syms inner-smap) body)))))

          ;; par/pmap: (raster.par/pmap idx bound cast body)
          (and (symbol? head) (= "raster.par" (namespace head))
               (= "pmap" (name head)))
          (let [[_ idx bound cast body] expr
                [new-idx inner-smap] (alpha-rename-scope-var smap idx)]
            (rebuild (list head new-idx
                           (subst-syms smap bound)
                           (subst-syms smap cast)
                           (subst-syms inner-smap body))))

          ;; par/map: (raster.par/map idx bound cast body)
          (and (symbol? head) (= "raster.par" (namespace head))
               (= "map" (name head)))
          (let [[_ idx bound cast body] expr
                [new-idx inner-smap] (alpha-rename-scope-var smap idx)]
            (rebuild (list head new-idx
                           (subst-syms smap bound)
                           (subst-syms smap cast)
                           (subst-syms inner-smap body))))

          ;; par/map!: (raster.par/map! out idx bound cast body)
          ;;   offset: (raster.par/map! out idx bound :offset base cast body)
          (and (symbol? head) (= "raster.par" (namespace head))
               (= "map!" (name head)))
          (if (and (>= (count expr) 8) (= :offset (nth expr 4)))
            ;; Offset variant
            (let [[_ out idx bound _kw base cast body] expr
                  [new-idx inner-smap] (alpha-rename-scope-var smap idx)]
              (rebuild (list head
                             (subst-syms smap out)
                             new-idx
                             (subst-syms smap bound)
                             :offset
                             (subst-syms smap base)
                             (subst-syms smap cast)
                             (subst-syms inner-smap body))))
            ;; Standard variant
            (let [[_ out idx bound cast body] expr
                  [new-idx inner-smap] (alpha-rename-scope-var smap idx)]
              (rebuild (list head
                             (subst-syms smap out)
                             new-idx
                             (subst-syms smap bound)
                             (subst-syms smap cast)
                             (subst-syms inner-smap body)))))

          ;; par/reduce: (raster.par/reduce acc init idx bound body)
          (and (symbol? head) (= "raster.par" (namespace head))
               (= "reduce" (name head)))
          (let [[_ acc init idx bound body] expr
                [new-acc smap1] (alpha-rename-scope-var smap acc)
                [new-idx inner-smap] (alpha-rename-scope-var smap1 idx)]
            (rebuild (list head new-acc
                           (subst-syms smap init)
                           new-idx
                           (subst-syms smap bound)
                           (subst-syms inner-smap body))))

          ;; par/reduce!: (raster.par/reduce! out acc init idx bound cast body)
          (and (symbol? head) (= "raster.par" (namespace head))
               (= "reduce!" (name head)))
          (let [[_ out acc init idx bound cast body] expr
                [new-acc smap1] (alpha-rename-scope-var smap acc)
                [new-idx inner-smap] (alpha-rename-scope-var smap1 idx)]
            (rebuild (list head
                           (subst-syms smap out)
                           new-acc
                           (subst-syms smap init)
                           new-idx
                           (subst-syms smap bound)
                           (subst-syms smap cast)
                           (subst-syms inner-smap body))))

          ;; par/scan!: (raster.par/scan! out idx bound init cast body)
          (and (symbol? head) (= "raster.par" (namespace head))
               (= "scan!" (name head)))
          (let [[_ out idx bound init cast body] expr
                [new-idx inner-smap] (alpha-rename-scope-var smap idx)]
            (rebuild (list head
                           (subst-syms smap out)
                           new-idx
                           (subst-syms smap bound)
                           (subst-syms smap init)
                           (subst-syms smap cast)
                           (subst-syms inner-smap body))))

          ;; Default: recurse into all children
          :else
          (rebuild (map (partial subst-syms smap) expr))))

      (vector? expr) (mapv (partial subst-syms smap) expr)
      :else expr)))

(defn- subst-syms-safe
  "Like subst-syms but also rewrites devirtualized core array ops to
	dispatched raster.arrays versions."
  [smap expr]
  ;; Merge devirt->dispatch as base, smap overrides — then use
  ;; the scope-aware subst-syms which handles scoping correctly.
  (subst-syms (merge devirt->dispatch smap) expr))

;; ================================================================
;; value+grad / grad inlining
;; ================================================================

(def ^:dynamic *ad-transform-body-fn*
  "The AD transform function (reverse/transform-body).
  Bound by the pipeline when AD inlining is needed.
  Breaks the inline↔reverse circular dependency."
  nil)

(declare inline-deftm-calls inline-invk flatten-nested-lets)

(defn- value+grad-call?
  "Detect ((value+grad <var>) args...) or ((grad <var>) args...) call pattern.
   Returns {:var-sym, :args, :mode (:value+grad or :grad)} or nil."
  [init]
  (when (and (seq? init) (seq? (first init)) (>= (count init) 2))
    (let [callee (first init)
          args (vec (rest init))]
      (when (and (= 2 (count callee))
                 (seq? (second callee)))
        (let [op (first callee)
              var-form (second callee)]
          (when (and (or (= op 'raster.ad.reverse/value+grad)
                         (= op 'raster.ad.reverse/grad))
                     (seq? var-form)
                     (= 'var (first var-form)))
            (let [var-sym (second var-form)]
              {:var-sym var-sym
               :args args
               :mode (if (= op 'raster.ad.reverse/value+grad)
                       :value+grad :grad)})))))))

(defn- parse-ftm-form
  "Parse an ftm form: (ftm [a :- Type, b :- Type] :- RetType body).
   Handles both typed [a :- T, b :- T] and untyped [a b] and mixed [a :- T, b] params.
   Returns {:params [a b] :body body} or nil."
  [form]
  (when (and (seq? form) (= 'ftm (first form)))
    (let [args (rest form)
          params-vec (first args)
          ;; Extract param names, handling both typed [a :- T] and untyped [a] params
          param-syms (loop [items (seq params-vec) result []]
                       (if-not items
                         result
                         (let [item (first items)]
                           (if (and (symbol? item) (not= item :-))
                             (if (and (next items) (= :- (second items)))
                               ;; typed param: sym :- Type — skip :- and type
                               (recur (nnext (next items)) (conj result item))
                               ;; untyped param: just sym
                               (recur (next items) (conj result item)))
                             ;; skip non-symbol tokens (stray :- or types)
                             (recur (next items) result)))))
          ;; Find body: after params-vec, skip :- RetType if present
          remaining (rest args)
          remaining (if (= :- (first remaining))
                      (drop 2 remaining)  ;; skip :- RetType
                      remaining)
          body (if (= 1 (count remaining))
                 (first remaining)
                 (cons 'do remaining))]
      (when (and (seq param-syms) body)
        {:params param-syms :body body}))))

(defn- unwrap-D
  "Unwrap nested D applications: (D (D (var f))) → {:var-form (var f) :order 2}.
   Handles raster.sym.fn-algebra/D and raster.core/D."
  [form]
  (if (and (seq? form) (= 'var (first form)))
    {:var-form form :order 0}
    (when (and (seq? form) (= 2 (count form))
               (contains? #{'raster.sym.fn-algebra/D 'raster.core/D} (first form)))
      (when-let [{:keys [var-form order]} (unwrap-D (second form))]
        {:var-form var-form :order (inc order)}))))

(defn- ensure-derivative-deftm!
  "For higher-order D^k, dynamically create intermediate deftm vars.
   D²f needs a deftm for f', D³f needs deftms for f' and f'', etc.
   Returns the var for the (k-1)th derivative deftm."
  [base-var-sym ^long order]
  (loop [k 1
         current-var-sym base-var-sym]
    (if (>= k order)
      current-var-sym
      ;; Create a deftm for the k-th derivative
      (let [v (resolve current-var-sym)
            m (meta v)
            ;; Get param info from the base or intermediate var
            params (or (:raster.core/deftm-params m)
                       (let [dt (:raster.core/dispatch-table m)]
                         (when dt
                           (let [entry (first (mapcat val @dt))
                                 mn (symbol (str (types/mangle (:name m) (:tags entry))))
                                 mv (ns-resolve (:ns m) mn)]
                             (:raster.core/deftm-params (meta mv))))))
            tags (or (:raster.core/deftm-tags m)
                     (let [dt (:raster.core/dispatch-table m)]
                       (when dt (:tags (first (mapcat val @dt))))))
            src-ns (or (:ns m) *ns*)
            ;; Build the derivative deftm name
            deriv-name (symbol (str (name current-var-sym) "__D" k))
            deriv-sym (symbol (str src-ns) (str deriv-name))
            ;; Build annotated params with :- type annotations.
            ;; Convert walker tags (double, doubles) to TC types (Double, (Array double))
            tag->tc {'double 'Double 'float 'Float 'long 'Long 'int 'Int
                     'doubles '(Array double) 'floats '(Array float)
                     'longs '(Array long) 'ints '(Array int)}
            _ (when-not tags
                (throw (ex-info (str "ensure-derivative-deftm!: cannot synthesize derivative for `"
                                     current-var-sym "` — no parameter type tags. "
                                     "All deftm functions must have :- Type annotations.")
                                {:var current-var-sym :order k})))
            annotated-params (vec (mapcat (fn [p t] [p :- (or (get tag->tc t)
                                                              (throw (ex-info (str "Unknown tag `" t "` for param `" p "`")
                                                                              {:param p :tag t})))])
                                          params tags))
            ;; Body: (let [vg ((value+grad (var current)) args...)] (nth vg 1))
            ;; Must use let binding so inline-one-pass detects the value+grad call
            vg-sym (gensym "vg_")
            body (list 'let
                       [vg-sym (apply list
                                      (list 'raster.ad.reverse/value+grad
                                            (list 'var current-var-sym))
                                      params)]
                       (list 'nth vg-sym 1))]
        (binding [*ns* src-ns]
          (eval (list (requiring-resolve 'raster.core/deftm)
                      deriv-sym annotated-params :- 'Double body)))
        (recur (inc k) deriv-sym)))))

(defn- rewrite-D-values
  "Rewrite D-as-value expressions anywhere in a form.
   (D (var f)) appearing as an argument (not applied) gets replaced with
   (var auto-D-f) where auto-D-f is a dynamically-created deftm.
   This enables function algebra like (n/+ (D #'f) (D #'g)) to compile."
  [form]
  (cond
    ;; (D (var f)) — D producing a function value
    (and (seq? form) (= 2 (count form)))
    (when-let [{:keys [var-form order]} (unwrap-D form)]
      (when (pos? order)
        (let [var-sym (second var-form)
              ;; ensure-derivative-deftm! creates up to order-1 intermediates
              ;; and returns the (order-1)th var. For D-as-value, we need
              ;; the derivative itself — create one MORE level.
              target-sym (ensure-derivative-deftm! var-sym (inc order))]
          (list 'var target-sym))))
    :else nil))

(defn- rewrite-D-values-in-form
  "Walk a form and replace all D-as-value subexpressions with var references."
  [form]
  (cond
    (not (seq? form)) form
    ;; Check if this form itself is a D value
    (rewrite-D-values form) (rewrite-D-values form)
    ;; Recurse into subforms
    :else (let [rewritten (map rewrite-D-values-in-form form)
                changed? (not= (seq rewritten) (seq form))]
            (if changed?
              (let [r (apply list rewritten)]
                (if-let [m (meta form)] (with-meta r m) r))
              form))))

(defn- fn-valued?
  "True if form is a D, partial-d, or var expression (produces a function value)."
  [form]
  (and (seq? form)
       (or (unwrap-D form)                        ;; (D ...) or (D (D ...))
           (and (= 'var (first form)))             ;; (var f)
           (and (seq? (first form))                ;; ((partial-d i) (var f))
                (let [op (first (first form))]
                  (contains? #{'raster.sym.fn-algebra/partial-d
                               'raster.core/partial-d} op))))))

(defn- distribute-fn-application
  "Distribute function application over function algebra.
   ((op f g) x y) → (op (f x y) (g x y)) when op is a pointwise function op
   and at least one argument is function-valued (D, var, partial-d).
   Scalar arguments (numbers, bare symbols) are passed through unchanged.
   Returns rewritten form or nil."
  [init]
  (when (and (seq? init) (seq? (first init)) (>= (count init) 2))
    (let [callee (first init)
          app-args (rest init)]
      (when (and (seq? callee) (>= (count callee) 2)
                 (symbol? (first callee))
                 (namespace (first callee))
                 (op/arithmetic-op? (first callee)))
        (let [op (first callee)
              fn-args (rest callee)]
          ;; Only distribute if at least one arg is function-valued
          (when (some fn-valued? fn-args)
            (apply list op
                   (map (fn [fa]
                          (if (fn-valued? fa)
                            (apply list fa app-args)
                            fa))
                        fn-args))))))))

(defn- D-call?
  "Detect D operator call patterns in walked bodies and rewrite to value+grad.

   Patterns detected:
     ((D (var f)) args...)                    → value+grad call + nth 1
     ((D (D (var f))) args...)               → creates intermediate deftm, then value+grad
     (((partial-d i) (var f)) args...)        → value+grad call + nth (inc i)
     ((op (D f) (D g)) args...)               → distributes application: (op ((D f) args) ((D g) args))

   Returns {:vg-call form, :nth-idx index} for decomposition into two bindings,
   or a rewritten form (for distribution), or nil."
  [init]
  (when (and (seq? init) (seq? (first init)) (>= (count init) 2))
    (let [callee (first init)
          args (rest init)]
      (or
       ;; ((op f g) x) — distribute application over function algebra
       ;; Returns a plain rewritten form (not {:vg-call ...})
       (when-let [distributed (distribute-fn-application init)]
         {:distributed distributed})

       ;; ((D ...) args...) — single or nested D
       (when-let [{:keys [var-form order]} (unwrap-D callee)]
         (when (pos? order)
           (let [var-sym (second var-form)
                 ;; For order>1, create intermediate deftms and use the last one
                 target-sym (if (= order 1)
                              var-sym
                              (ensure-derivative-deftm! var-sym order))]
             {:vg-call (apply list (list 'raster.ad.reverse/value+grad
                                         (list 'var target-sym)) args)
              :nth-idx 1})))

       ;; (((partial-d i) (var f)) args...) — partial derivative
       (when (and (seq? callee) (= 2 (count callee))
                  (seq? (first callee)) (= 2 (count (first callee))))
         (let [pd-form (first callee)
               pd-op (first pd-form)
               pd-arg (second pd-form)
               target (second callee)]
           (when (and (contains? #{'raster.sym.fn-algebra/partial-d 'raster.core/partial-d} pd-op)
                      (seq? target) (= 'var (first target)))
             (let [i (if (and (seq? pd-arg) (= 'long (first pd-arg)))
                       (second pd-arg)
                       pd-arg)]
               {:vg-call (apply list (list 'raster.ad.reverse/value+grad target) args)
                :nth-idx (list 'clojure.core/inc i)}))))))))

(defn- rewrite-D-calls-in-form
  "Walk a form and rewrite D-call subexpressions to value+grad.
   Returns {:form rewritten-form :bindings [[sym expr]...]} with any
   lifted vg bindings that need to be prepended."
  [form]
  (cond
    (not (seq? form))
    {:form form :bindings []}

    :else
    ;; Check if this form is a D-call at the top level
    (if-let [d-result (D-call? form)]
      (if-let [distributed (:distributed d-result)]
        ;; Distributed: recurse into the distributed form
        (rewrite-D-calls-in-form distributed)
        ;; D/partial-d: create vg binding + nth
        (let [{:keys [vg-call nth-idx]} d-result
              vg-sym (gensym "vg_D_")]
          {:form (list 'clojure.core/nth vg-sym nth-idx)
           :bindings [[vg-sym vg-call]]}))
      ;; Not a D-call — recurse into subforms
      (let [results (map rewrite-D-calls-in-form form)
            all-bindings (vec (mapcat :bindings results))
            new-children (map :form results)
            changed? (or (seq all-bindings) (not= (map :form results) (seq form)))]
        (if changed?
          {:form (let [r (apply list new-children)]
                   (if-let [m (meta form)] (with-meta r m) r))
           :bindings all-bindings}
          {:form form :bindings []})))))

(defn- resolve-deftm-for-ad
  "Resolve a var symbol to its walked body and params for AD transform.
   Handles dispatch table resolution for generic deftm functions.
   When arg-tags are provided, selects the matching overload.
   Returns {:walked-body, :params, :var} or nil."
  ([var-sym] (resolve-deftm-for-ad var-sym nil))
  ([var-sym arg-tags]
   (let [v (or (try (resolve var-sym) (catch Exception _ nil))
               ;; For unqualified symbols from inlined code, try all loaded namespaces
               (when-not (namespace var-sym)
                 (some (fn [ns-obj]
                         (try
                           (let [v (ns-resolve ns-obj var-sym)]
                             (when (and v (var? v)
                                        (or (:raster.core/deftm-walked-body (meta v))
                                            (:raster.core/deftm-walked-body (try (meta @v) (catch Exception _ nil)))
                                            (:raster.core/dispatch-table (meta v))))
                               v))
                           (catch Exception _ nil)))
                       (all-ns))))]
     (when v
       (let [;; Try direct metadata, deref'd value metadata, then dispatch table
             val-meta (when (var? v) (try (meta @v) (catch Exception _ nil)))
             resolved (cond
                        (:raster.core/deftm-walked-body (meta v)) v
                        (:raster.core/deftm-walked-body val-meta) @v
                        :else (when-let [dt (:raster.core/dispatch-table (meta v))]
                                (let [all-methods (mapcat val @dt)
                                      has-parametric? (when-let [pr (requiring-resolve
                                                                     'raster.compiler.core.dispatch/parametric-registry)]
                                                        (contains? @@pr (symbol (namespace var-sym) (name var-sym))))
                                ;; Type-safe overload selection + parametric specialization
                                      method (or (when (seq arg-tags)
                                                   (or (first (filter #(= (vec arg-tags) (vec (:tags %))) all-methods))
                                                       (when-let [try-ps (requiring-resolve
                                                                          'raster.compiler.core.inference/try-parametric-specialize!)]
                                                         (when (try-ps (symbol (namespace var-sym) (name var-sym)) arg-tags)
                                                           (let [new-methods (mapcat val @dt)]
                                                             (first (filter #(= (vec arg-tags) (vec (:tags %)))
                                                                            new-methods)))))))
                                           ;; Non-parametric single method: safe (no other overloads possible)
                                                 (when (and (= 1 (count all-methods)) (not has-parametric?))
                                                   (first all-methods)))
                                      ns-obj (:ns (meta v))
                                      mangled (when method
                                                (symbol (str (types/mangle (:name (meta v))
                                                                           (:tags method)))))]
                                  (when mangled
                                    (try (ns-resolve ns-obj mangled) (catch Exception _ nil))))))]
         (when resolved
           (let [m (meta resolved)
                 walked-body (:raster.core/deftm-walked-body m)
                 params (:raster.core/deftm-params m)]
             (when (and walked-body params)
               {:walked-body walked-body :params params :var resolved}))))))))

(def ^:dynamic *expand-for-ad-trace* false)
(def ^:dynamic *param-env* nil)

(defn- expand-for-ad
  "Fixpoint expansion for AD: inline all non-template deftm calls and
   value+grad calls until stable. Template-backed calls stay symbolic
   so the AD transform can use their templates.

   Each iteration:
   1. Normalize to let* form
   2. Inline non-template deftm calls in bindings (incl. value+grad)
   3. Inline non-template .invk calls in body/binding expressions
   4. Flatten nested lets
   5. CSE to fold (nth vector-literal N)

   Converges when no more expansions are possible.
   Set *expand-for-ad-trace* to true for diagnostic output."
  [form max-iters]
  (let [cse-let cse/cse-let
        trace? *expand-for-ad-trace*
        binding-count (fn [f] (when (form/binding-form? f)
                                (/ (count (second f)) 2)))]
    (loop [current form, iter 0]
      (if (>= iter max-iters)
        (do (when trace? (println "  [expand-for-ad] max iterations reached"))
            current)
        (let [;; Step 1: Normalize to let* and flatten nested lets
              wrapped (if (form/binding-form? current)
                        current
                        (list 'let* [] current))
              ;; Step 2: Inline deftm calls in bindings
              ;; With auto-rrules removed, only ops with explicit templates
              ;; or explicit rrule closures are protected from inlining.
              ;; User deftm functions (one-step, two-steps) get inlined.
              after-deftm (inline-deftm-calls wrapped 3 false false)
              _ (when (and trace? (not= after-deftm wrapped))
                  (println (str "  [expand-for-ad] iter " iter " step 2 (inline): "
                                (binding-count wrapped) " → " (binding-count after-deftm) " bindings")))
              ;; Step 3: Inline .invk calls with simple bodies (no loops/fn)
              after-invk (inline-invk after-deftm {:preserve-templates? true})
              ;; Step 4: Flatten nested lets (with ANF-lift renaming)
              after-flat (flatten-nested-lets after-invk)
              _ (when (and trace? (not= (binding-count after-flat) (binding-count after-invk)))
                  (println (str "  [expand-for-ad] iter " iter " step 4 (flatten): "
                                (binding-count after-invk) " → " (binding-count after-flat) " bindings")))
              ;; Step 5: CSE (fold nth on vector literals, propagate aliases)
              expanded (:form (cse-let after-flat))
              _ (when (and trace? (not= expanded after-flat))
                  (println (str "  [expand-for-ad] iter " iter " step 5 (CSE): changed")))
              ;; Convergence: no step changed the form
              changed? (or (not= after-deftm wrapped)
                           (not= after-invk after-deftm))]
          (if changed?
            (recur expanded (inc iter))
            (do (when trace?
                  (println (str "  [expand-for-ad] converged at iter " iter
                                ", " (binding-count expanded) " bindings")))
                expanded)))))))

(defn inline-value+grad-call
  "Inline a ((value+grad #'f) args...) call into flat let* bindings.

   Applies reverse-mode AD to f's walked body, flattens the result,
   and produces bindings for the loss value and gradient symbols.

   transform-body-fn: the AD transform function (reverse/transform-body).
   Passed as parameter to avoid inline↔reverse circular dependency.

   Returns {:bindings [[sym1 expr1] [sym2 expr2] ...]
            :result-sym   — symbol bound to the [loss, d_p1, d_p2, ...] vector
            :loss-sym     — symbol bound to the loss
            :grad-syms    — [d_p1, d_p2, ...] gradient symbols}
   or nil if the call cannot be inlined."
  [vg-info & {:keys [param-env]}]
  (let [{:keys [var-sym args mode]} vg-info
        ;; Infer arg types from param-env to select the right overload
        arg-tags (when (and param-env (every? symbol? args))
                   (let [tags (mapv #(get param-env %) args)]
                     (when (every? some? tags) tags)))
        deftm-info (resolve-deftm-for-ad var-sym arg-tags)]
    (when deftm-info
      (let [{:keys [walked-body params]} deftm-info
            transform-body (or *ad-transform-body-fn*
                               (throw (ex-info "*ad-transform-body-fn* not bound — pipeline must bind it for AD inlining" {})))
            body-form (first walked-body)
            ;; Fixpoint expansion: inline all non-template deftm calls and
            ;; value+grad calls until no more expansions are possible.
            ;; Template-backed ops (matmul, mse-loss, etc.) stay symbolic
            ;; so the AD transform can use their templates.
            body-form (expand-for-ad body-form 10)
            active-params (mapv #(if (symbol? %) % (symbol (name %))) params)
            ad-form (transform-body body-form active-params)
            ;; Canonicalize and flatten
            canonical (flatten/canonicalize-ad-form ad-form)]
        (when canonical
          (let [flat-result (flatten/flatten-ad-form canonical)]
            (when flat-result
              (let [{:keys [form result-sym param-adj-syms]} flat-result
                    ;; form is (let* [...fwd... dy=1.0 ...rev...] primal)
                    ;; Extract bindings
                    [_ bindings-vec body-expr] form
                    pairs (partition 2 bindings-vec)
                    ;; Substitute actual args for formal params
                    param-names (mapv #(with-meta (if (symbol? %) % (symbol (name %))) nil)
                                      params)
                    param-subst (zipmap param-names args)
                    ;; Rename internal symbols to avoid conflicts, and substitute params
                    renamed-pairs
                    (let [subst (atom param-subst)]
                      (mapv (fn [[sym expr]]
                              (let [base (if (contains? (set param-names) sym)
                                           sym ;; don't rename param bindings
                                           (gensym (str (name sym) "_")))
                                    ;; Preserve metadata (AD type tags etc.)
                                    new-sym (if (and (not (identical? base sym)) (meta sym))
                                              (with-meta base (meta sym))
                                              base)
                                    substed-expr (subst-syms @subst expr)]
                                (when-not (= sym new-sym)
                                  (swap! subst assoc sym new-sym))
                                [new-sym substed-expr]))
                            pairs))
                    ;; The loss symbol after substitution
                    final-subst (reduce (fn [m [old [new _]]]
                                          (if (= old new) m (assoc m old new)))
                                        param-subst
                                        (map vector
                                             (map first pairs)
                                             renamed-pairs))
                    loss-sym (get final-subst result-sym result-sym)
                    grad-syms (mapv #(get final-subst % %) param-adj-syms)
                    ;; Tag gradient syms with expected types from param tags.
                    ;; Array params (doubles, floats, longs) have array gradients.
                    ;; This ensures downstream type inference knows the gradient types
                    ;; even when AD initializes accumulators to scalar 0.0.
                    ;; Use arg-tags (from param-env) first — these are always correct.
                    ;; Fall back to deftm var metadata for functions with <5 params.
                    param-tags (or arg-tags
                                   (:raster.core/deftm-tags (meta (resolve var-sym)))
                                   (vec (repeat (count grad-syms) 'double)))
                    grad-syms (mapv (fn [gs tag]
                                      (if (and tag (not= tag 'double) (not= tag 'long))
                                        (with-meta gs {:tag tag})
                                        gs))
                                    grad-syms param-tags)
                    ;; Elements: loss + all gradient syms (DCE handles dead ones)
                    elements (if (= mode :value+grad)
                               (vec (cons loss-sym grad-syms))
                               (vec grad-syms))]
                {:bindings (vec renamed-pairs)
                 :result-sym nil
                 :loss-sym loss-sym
                 :grad-syms grad-syms
                 :elements elements}))))))))

(defn- inline-one-pass
  "Single pass of deftm inlining over a let* form's bindings."
  ([form] (inline-one-pass form subst-syms false))
  ([form subst-fn] (inline-one-pass form subst-fn false))
  ([form subst-fn skip-ad-rule-check?]
   (let [[_ bindings-vec & body-exprs-raw] form
         ;; Lift body-position calls into bindings so they get inlined too.
         ;; Without this, a tail call like (dense W2 a b2) stays opaque because
         ;; inline-one-pass only processes let* binding pairs.
         [extra-body-pairs body-exprs]
         (if (and (= 1 (count body-exprs-raw))
                  (seq? (first body-exprs-raw))
                  (let [head (first (first body-exprs-raw))]
                    (and (symbol? head) (qualified-symbol? head)
                         (not (form/binding-form? (first body-exprs-raw))))))
           (let [result-sym (with-meta (gensym "body_result_")
                              (meta (first body-exprs-raw)))]
             [[[result-sym (first body-exprs-raw)]] [result-sym]])
           [nil body-exprs-raw])
         pairs (concat (partition 2 bindings-vec) extra-body-pairs)
         result-pairs (atom [])
         any-inlined? (atom false)
         ;; Type env for dispatch resolution when inlining multi-overload calls
         type-env (atom (or *param-env* {}))
         ;; Map from value+grad result sym → elements vector for nth resolution
         vg-elements (atom {})
         ;; Map from sym → {:var-form, :mode} for indirect value+grad calls
         ;; e.g. vg-fn = (value+grad f) ... result = (vg-fn args...)
         vg-fns (atom {})
         ;; Map from sym → {:params [syms...] :body form} for ftm-valued bindings.
         ;; Enables beta reduction: (let [f (ftm [a] body)] (f x)) → body[a:=x]
         ftm-bindings (atom {})
         ;; Pre-compute symbol use counts for safe beta reduction.
         ;; Only beta-reduce ftms that are single-use OR ^:inline.
         sym-uses (let [all-forms (concat (map second pairs) body-exprs)]
                    (frequencies
                     (mapcat (fn count-syms [f]
                               (cond (symbol? f) [f]
                                     (seq? f) (mapcat count-syms f)
                                     (vector? f) (mapcat count-syms f)
                                     :else []))
                             all-forms)))]
     (doseq [[sym init-raw] pairs]
       (let [;; D-value rewriting is deferred — D-call? and distribution run first.
             ;; Only rewrite D-as-value when D appears as an argument to non-pointwise ops.
             init init-raw
             ;; Detect (value+grad f) or (grad f) partial application
             vg-partial (when (and (seq? init) (= 2 (count init)))
                          (let [op (first init)]
                            (when (or (= op 'raster.ad.reverse/value+grad)
                                      (= op 'raster.ad.reverse/grad))
                              (let [arg (second init)]
                                (when (or (and (seq? arg) (= 'var (first arg)))
                                          (symbol? arg))
                                  {:var-or-form arg
                                   :mode (if (= op 'raster.ad.reverse/value+grad)
                                           :value+grad :grad)})))))
             ;; Check for indirect call: (vg-fn args...) where vg-fn was (value+grad f)
             indirect-vg (when (and (not vg-partial) (seq? init) (symbol? (first init)))
                           (when-let [vg-entry (get @vg-fns (first init))]
                             (let [{:keys [var-or-form mode]} vg-entry
                                   args (vec (rest init))
                                   var-form (if (symbol? var-or-form)
                                              (list 'var var-or-form)
                                              var-or-form)]
                               {:var-sym (second var-form)
                                :args args
                                :mode mode})))
             ;; Rewrite D/partial-d/fn-algebra calls recursively in init
             {:keys [form bindings] :as d-result}
             (when-not vg-partial (rewrite-D-calls-in-form init))
             d-rewritten? (and d-result (seq bindings))
             init (if d-rewritten? form init)
             ;; Check for direct ((value+grad f) args...) pattern
             vg-info (or indirect-vg (value+grad-call? init))]
         (if d-rewritten?
           ;; D/partial-d/fn-algebra → emit lifted bindings + rewritten init
           (do (doseq [b bindings] (swap! result-pairs conj b))
               (swap! result-pairs conj [sym form])
               (reset! any-inlined? true))
           (if vg-partial
           ;; Partial application: track for later indirect call resolution
             (do (swap! vg-fns assoc sym vg-partial)
                 (swap! result-pairs conj [sym init])
                 (reset! any-inlined? true))
             (if vg-info
           ;; value+grad call → inline AD transform
               (if-let [vg-result (inline-value+grad-call vg-info :param-env *param-env*)]
                 (do
                   (doseq [[bsym bexpr] (:bindings vg-result)]
                     (swap! result-pairs conj [bsym bexpr]))
               ;; Track elements for resolve-vg-nth (handles nested nth refs)
               ;; For <=20 elements, also emit vector literal (CSE folds top-level nth)
                   (do (swap! vg-elements assoc sym (:elements vg-result))
                       (if (<= (count (:elements vg-result)) 20)
                         (swap! result-pairs conj [sym (vec (:elements vg-result))])
                         nil)) ;; >20: no vector literal, resolve-vg-nth handles all
                   (reset! any-inlined? true))
             ;; AD inlining failed, keep as-is
                 (swap! result-pairs conj [sym init]))
       ;; Beta reduction: (sym args...) where sym is bound to an ftm.
       ;; Only inline if the ftm is ^:inline OR used exactly once
       ;; (prevents code blowup from multi-use beta reduction).
               (let [beta-result
                     (when (and (seq? init) (symbol? (first init))
                                (contains? @ftm-bindings (first init)))
                       (let [ftm-sym (first init)
                             {:keys [params body inline?]} (get @ftm-bindings ftm-sym)
                             args (vec (rest init))
                             single-use? (<= (get sym-uses ftm-sym 0) 1)]
                         (when (and (= (count params) (count args))
                                    (or inline? single-use?))
                           (subst-syms (zipmap params args) body))))]
                 (if beta-result
                   (do (swap! result-pairs conj [sym beta-result])
                       (reset! any-inlined? true))
       ;; Check for (nth <vg-sym> N) — resolve to element symbol
                   (let [nth-resolved
                         (when (and (seq? init) (= 'clojure.core/nth (first init)) (= 3 (count init)))
                           (let [target (second init)
                                 idx-expr (nth init 2)
                                 idx (cond (integer? idx-expr) idx-expr
                                           (and (seq? idx-expr) (= 'long (first idx-expr))) (second idx-expr)
                                           :else nil)]
                             (when (and (symbol? target) idx (contains? @vg-elements target))
                               (let [elems (get @vg-elements target)]
                                 (when (< idx (count elems)) (clojure.core/nth elems idx))))))]
                     (if nth-resolved
                       (do (swap! result-pairs conj [sym nth-resolved])
                           (reset! any-inlined? true))
                       (let [head (call-head init)
             ;; Check for explicit AD gradient rules (templates with :grads/:grads-fn).
             ;; Ops with AD rules stay symbolic so the AD transform can use them.
                             has-ad-rule? (when (and head (not skip-ad-rule-check?))
                                            (or (rt/has-ad-rule? head)
                              ;; Also check via mangled-name resolution
                                                (when-let [[tmpl _] (rt/resolve-template head)]
                                                  (or (:grads tmpl) (:grads-fn tmpl)))))
                             arg-tags (when (and head (not has-ad-rule?) (seq? init))
                                        (let [tags (mapv #(inf/infer-arg-tag % @type-env) (call-args init))]
                                          (when (every? some? tags) tags)))
                             deftm-info (when (and head (not has-ad-rule?)) (try-resolve-deftm head arg-tags))]
                         (if (and deftm-info (safe-to-inline? deftm-info))
                           (let [{:keys [params walked-body tags]} deftm-info
                                 args (call-args init)
                 ;; Handle multi-form bodies: wrap in (do ...) if more than one form
                                 body-form (if (> (count walked-body) 1)
                                             (list* 'do walked-body)
                                             (first walked-body))
                                 callee-params (mapv #(with-meta (if (symbol? %) % (symbol (name %))) nil)
                                                     params)
                 ;; ANF argument lifting + typed bindings for callee params.
                 ;; Two cases require lifting an argument to a preceding let binding:
                 ;; 1. Record/value-class params: preserves type info for bytecode backend
                 ;; 2. Non-trivial call expressions: prevents duplication when the param
                 ;;    appears multiple times in the callee body (e.g. inside par/pmap
                 ;;    where CSE can't extract the duplicate afterward)
                                 skip-tags #{'double 'long 'float 'int
                                             'doubles 'floats 'longs 'ints 'objects}
                                 param-subst
                                 (into {}
                                       (map-indexed
                                        (fn [i p]
                                          (let [a (nth args i)
                                                t (when tags (nth tags i nil))]
                                            (cond
                              ;; Record/value type — emit typed binding
                                              (and t (not (skip-tags t))
                                                   (not (and t (.contains (str t) "IFn__")))
                                                   (symbol? a))
                                              (let [typed-sym (with-meta (gensym (str (name p) "_"))
                                                                {:tag t})]
                                                (swap! result-pairs conj [typed-sym a])
                                                [p typed-sym])

                              ;; Non-trivial call arg — lift to prevent duplication
                                              (needs-arg-lift? a)
                                              (let [lifted-sym (with-meta (gensym (str "arg_" (name p) "_"))
                                                                 (meta a))]
                                                (swap! result-pairs conj [lifted-sym a])
                                                [p lifted-sym])

                              ;; Simple arg (symbol, constant, trivial cast)
                                              :else
                                              [p a]))))
                                       callee-params)
                                 body-head (first body-form)]
                             (if (form/binding-form? body-form)
                               (let [[_ inner-bindings & inner-body] body-form
                                     inner-pairs (partition 2 inner-bindings)
                                     current-subst (atom param-subst)]
                                 (doseq [[inner-sym inner-init] inner-pairs]
                                   (let [base-renamed (gensym (str (name inner-sym) "_"))
                                         substed (subst-fn @current-subst inner-init)
                                         renamed (if (and (seq? substed)
                                                          (symbol? (first substed))
                                                          (op/alloc-op? (first substed)))
                                                   (with-meta base-renamed
                                                     {:raster.buffer/hoistable true
                                                      :tag (get alloc-sym->array-tag (first substed))})
                                                   (if (meta inner-sym)
                                                     (vary-meta base-renamed merge (meta inner-sym))
                                                     base-renamed))]
                                     (swap! result-pairs conj [renamed substed])
                                     (swap! current-subst assoc inner-sym renamed)))
                                 (let [substed-body (mapv #(subst-fn @current-subst %) inner-body)]
                                   (when (> (count substed-body) 1)
                                     (doseq [effect (butlast substed-body)]
                                       (swap! result-pairs conj [(with-meta (gensym "effect_") {:raster.effect/effectful true}) effect])))
                                   (swap! result-pairs conj [sym (last substed-body)])))
               ;; Handle do blocks: decompose into effect bindings + result
                               (if (and (seq? body-form) (= 'do (first body-form)))
                                 (let [do-exprs (rest body-form)
                                       substed-exprs (mapv #(subst-fn param-subst %) do-exprs)]
                                   (doseq [effect (butlast substed-exprs)]
                                     (swap! result-pairs conj [(with-meta (gensym "effect_") {:raster.effect/effectful true}) effect]))
                                   (swap! result-pairs conj [sym (last substed-exprs)]))
                                 (swap! result-pairs conj [sym (subst-fn param-subst body-form)])))
                             (reset! any-inlined? true)
            ;; Record the inlined function's return type so subsequent
            ;; backward calls can infer arg-tags through the binding chain
                             (when-let [ret-tag (:return-tag deftm-info)]
                               (swap! type-env assoc sym ret-tag)))
                           (do (swap! result-pairs conj [sym init])
               ;; Track ftm-valued bindings for beta reduction.
               ;; ^:inline on the sym or the ftm form enables multi-use inlining.
                               (when-let [ftm-info (parse-ftm-form init)]
                                 (swap! ftm-bindings assoc sym
                                        (assoc ftm-info :inline?
                                               (or (:inline (meta sym))
                                                   (:inline (meta init))))))
                               (when-let [tag (inf/infer-arg-tag init @type-env)]
                                 (swap! type-env assoc sym tag)))))))))))))
     ;; Rebuild type-env from all result-pairs (covers inlined bindings)
       (doseq [[s e] @result-pairs]
         (when-not (contains? @type-env s)
           (when-let [tag (inf/infer-arg-tag e @type-env)]
             (swap! type-env assoc s tag)))))
     ;; Resolve (nth <vg-sym> N) in body expressions for >20 param cases
     (let [resolve-vg-nth (fn resolve-vg-nth [expr]
                            (cond
                              (and (seq? expr) (= 'clojure.core/nth (first expr)) (= 3 (count expr)))
                              (let [target (second expr)
                                    idx-expr (clojure.core/nth expr 2)
                                    idx (cond (integer? idx-expr) idx-expr
                                              (and (seq? idx-expr) (= 'long (first idx-expr))) (second idx-expr)
                                              :else nil)]
                                (if (and (symbol? target) idx (contains? @vg-elements target))
                                  (let [elems (get @vg-elements target)]
                                    (if (< idx (count elems)) (clojure.core/nth elems idx) expr))
                                  expr))
                              (seq? expr) (let [r (apply list (map resolve-vg-nth expr))]
                                            (if-let [m (meta expr)] (with-meta r m) r))
                              (vector? expr) (mapv resolve-vg-nth expr)
                              :else expr))
           ;; Resolve nth in BOTH binding values and body expressions
           resolved-pairs (if (seq @vg-elements)
                            (mapv (fn [[s e]] [s (resolve-vg-nth e)]) @result-pairs)
                            @result-pairs)
           ;; Rewrite D-as-value and D-call patterns in body expressions.
           extra-bindings (atom [])
           d-rewritten-body (mapv (fn [expr]
                                    (let [{:keys [form bindings]} (rewrite-D-calls-in-form expr)]
                                      (if (seq bindings)
                                        (do (swap! extra-bindings into bindings)
                                            (reset! any-inlined? true)
                                            form)
                                        expr)))
                                  body-exprs)
           resolved-body (if (seq @vg-elements)
                           (map resolve-vg-nth d-rewritten-body)
                           d-rewritten-body)]
       ;; Append extra bindings from D-call body rewrites
       (let [all-pairs (if (seq @extra-bindings)
                         (into (vec resolved-pairs) @extra-bindings)
                         resolved-pairs)]
         {:form (list* 'let* (vec (mapcat identity all-pairs)) resolved-body)
          :inlined? @any-inlined?})))))

(defn inline-deftm-calls
  "Inline deftm function bodies at call sites in let* forms."
  ([form] (inline-deftm-calls form 3))
  ([form max-depth] (inline-deftm-calls form max-depth false))
  ([form max-depth safe?] (inline-deftm-calls form max-depth safe? false))
  ([form max-depth safe? skip-ad-rule-check?]
   (if (not (form/binding-form? form))
     form
     (let [subst-fn (if safe? subst-syms-safe subst-syms)]
       (loop [current-form form
              depth max-depth]
         (if (<= depth 0)
           current-form
           (let [{:keys [form inlined?]} (inline-one-pass current-form subst-fn skip-ad-rule-check?)]
             (if inlined?
               (recur form (dec depth))
               form))))))))

(defn lower-to-ad-primitives
  "Pre-AD lowering: expand compound deftm ops into primitive ops."
  ([form] (lower-to-ad-primitives form 3))
  ([form max-depth]
   (inline-deftm-calls form max-depth false)))

(defn- flatten-binding-expr
  "If expr is a nested let/do, lift inner bindings to the parent level."
  [result-sym expr]
  (cond
    (form/binding-form? expr)
    (let [[_ inner-bindings & inner-body] expr
          inner-pairs (vec (partition 2 inner-bindings))
          body-exprs (vec inner-body)]
      (if (= 1 (count body-exprs))
        (let [lifted (mapv vec inner-pairs)
              body-lifted (flatten-binding-expr result-sym (first body-exprs))]
          (into lifted body-lifted))
        (let [lifted (mapv vec inner-pairs)
              effects (mapv (fn [effect-expr] [(with-meta (gensym "_eff_") {:raster.effect/effectful true}) effect-expr]) (butlast body-exprs))
              final (flatten-binding-expr result-sym (last body-exprs))]
          (into (into lifted effects) final))))

    (and (seq? expr) (= 'do (first expr)))
    (let [exprs (vec (rest expr))]
      (if (= 1 (count exprs))
        (flatten-binding-expr result-sym (first exprs))
        (let [effects (mapv (fn [effect-expr] [(with-meta (gensym "_eff_") {:raster.effect/effectful true}) effect-expr]) (butlast exprs))
              final (flatten-binding-expr result-sym (last exprs))]
          (into effects final))))

    :else
    [[result-sym expr]]))

(defn- inline-invk
  "Inline .invk calls to deftm implementations.
   Policy controls behavior:
   - :preserve-templates? true  -> skip template-backed ops (for pre-AD expansion)
   - :preserve-templates? false -> inline everything (for post-AD backend expansion)"
  ([form] (inline-invk form {}))
  ([form {:keys [preserve-templates?] :as policy}]
   (cond
     (and (seq? form) (= '.invk (first form)) (>= (count form) 3))
     (let [impl-sym (second form)
           has-rule? (and preserve-templates? (first (rt/resolve-template impl-sym)))
           deftm-info (when-not has-rule? (try-resolve-deftm impl-sym))]
       (if deftm-info
         (let [{:keys [params walked-body]} deftm-info
               body-form (first walked-body)
               safe? (safe-to-inline? deftm-info)]
           (if (not safe?)
             ;; Complex body -- keep as .invk, recurse into arguments only
             (let [args (drop 2 form)
                   new-args (map #(inline-invk % policy) args)]
               (util/make-invk impl-sym new-args (meta form)))
             ;; Simple body -- safe to inline
             (let [args (vec (drop 2 form))
                   {:keys [tags]} deftm-info
                   callee-params (mapv #(with-meta (if (symbol? %) % (symbol (name %))) nil) params)
                   skip-tags #{'double 'long 'float 'int
                               'doubles 'floats 'longs 'ints 'objects}
                   ;; ANF lifting + typed bindings — same logic as inline-one-pass
                   lifted-bindings (atom [])
                   param-subst
                   (into {}
                         (map-indexed
                          (fn [i p]
                            (let [a (nth args i nil)
                                  t (when tags (nth tags i nil))]
                              (cond
                                (and t (not (skip-tags t))
                                     (not (and t (.contains (str t) "IFn__")))
                                     (symbol? a))
                                (let [typed-sym (with-meta (gensym (str (name p) "_"))
                                                  {:tag t})]
                                  (swap! lifted-bindings conj typed-sym a)
                                  [p typed-sym])

                                (needs-arg-lift? a)
                                (let [lifted-sym (with-meta (gensym (str "arg_" (name p) "_"))
                                                   (meta a))]
                                  (swap! lifted-bindings conj lifted-sym a)
                                  [p lifted-sym])

                                :else
                                [p a]))))
                         callee-params)
                   inlined (subst-syms param-subst body-form)
                   result (inline-invk inlined policy)]
               ;; Wrap in let* with lifted bindings if any were emitted
               (if (seq @lifted-bindings)
                 (list 'let* (vec @lifted-bindings) result)
                 result))))
         ;; Has template or can't resolve: keep call, recurse into arguments
         (let [args (drop 2 form)
               new-args (map #(inline-invk % policy) args)]
           (util/make-invk impl-sym new-args (meta form)))))

     ;; Binding forms -- recurse into bindings and body
     (and (seq? form) (form/binding-form? form))
     (let [[head bindings & body] form
           new-bindings (vec (mapcat (fn [[sym value]] [sym (inline-invk value policy)])
                                     (partition 2 bindings)))
           new-body (map #(inline-invk % policy) body)
           r (apply list head new-bindings new-body)]
       (if-let [m (meta form)] (with-meta r m) r))

     ;; Scope-introducing forms -- recurse into body only (not binding vector)
     (and (seq? form) (form/scope-form? form))
     (let [[head binding & body] form
           r (apply list head binding (map #(inline-invk % policy) body))]
       (if-let [m (meta form)] (with-meta r m) r))

     ;; do block -- recurse into all expressions
     (and (seq? form) (= 'do (first form)))
     (let [r (apply list 'do (map #(inline-invk % policy) (rest form)))]
       (if-let [m (meta form)] (with-meta r m) r))

     ;; Any other seq -- recurse into all elements
     (seq? form)
     (let [r (apply list (map #(inline-invk % policy) form))]
       (if-let [m (meta form)] (with-meta r m) r))

     (vector? form) (mapv #(inline-invk % policy) form)
     :else form)))

(defn- anf-lift
  "Lift nested let/let* forms out of expression position into sibling bindings.
   Returns [lifted-bindings result-expr] where lifted-bindings is a vector
   of [sym expr] pairs and result-expr is the expression with lets replaced
   by their body symbols. Renames symbols to avoid collisions when multiple
   nested lets are lifted into the same scope."
  [expr]
  (cond
    ;; Binding form in expression position → lift bindings out with renaming
    (form/binding-form? expr)
    (let [[_ inner-bindings & inner-body] expr
          inner-pairs (vec (map vec (partition 2 inner-bindings)))
          ;; Rename all symbols to avoid collisions
          rename-map (atom {})
          renamed-pairs (mapv (fn [[sym init]]
                                (if-not (symbol? sym)
                                  ;; Destructuring or non-symbol binding — keep as-is
                                  [sym (subst-syms @rename-map init)]
                                  (let [base (gensym (str (name sym) "_"))
                                        new-sym (if (meta sym)
                                                  (with-meta base (meta sym))
                                                  base)]
                                    (swap! rename-map assoc sym new-sym)
                                    [new-sym (subst-syms @rename-map init)])))
                              inner-pairs)
          ;; Recursively ANF-lift each renamed binding value
          lifted-inner (mapcat (fn [[sym init]]
                                 (let [[lifted cleaned] (anf-lift init)]
                                   (concat lifted [[sym cleaned]])))
                               renamed-pairs)
          ;; ANF-lift the body (with renaming applied)
          body-expr (subst-syms @rename-map (last inner-body))
          [body-lifted body-cleaned] (anf-lift body-expr)]
      [(vec (concat lifted-inner body-lifted)) body-cleaned])

    ;; Non-liftable forms — do NOT recurse (would lift bindings
    ;; past scope boundaries or conditional branches)
    (and (seq? expr) (not (:liftable? (form/form-info expr))))
    [[] expr]

    ;; seq (function call) → recurse into arguments
    (seq? expr)
    (let [results (map anf-lift expr)
          all-lifted (vec (mapcat first results))
          cleaned (let [r (apply list (map second results))]
                    (if-let [m (meta expr)] (with-meta r m) r))]
      (if (empty? all-lifted)
        [[] expr]
        [all-lifted cleaned]))

    ;; vector → recurse
    (vector? expr)
    (let [results (map anf-lift expr)
          all-lifted (vec (mapcat first results))
          cleaned (mapv second results)]
      (if (empty? all-lifted)
        [[] expr]
        [all-lifted cleaned]))

    ;; leaf
    :else [[] expr]))

(defn flatten-nested-lets
  "Flatten all nested let/let* forms into a single top-level let*.
   Handles lets in binding values, body expressions, and argument positions."
  [form]
  (if-not (form/binding-form? form)
    form
    (let [[_ bindings-vec & body-exprs] form
          pairs (partition 2 bindings-vec)
          ;; ANF-lift each binding value
          new-pairs (vec (mapcat (fn [[sym init]]
                                   (let [[lifted cleaned] (anf-lift init)]
                                     (concat lifted [[sym cleaned]])))
                                 pairs))
          ;; ANF-lift body expressions (only lift from the last one)
          [body-lifted final-body-expr]
          (if (= 1 (count body-exprs))
            (let [[lifted cleaned] (anf-lift (first body-exprs))]
              [lifted [cleaned]])
            [[] body-exprs])
          all-pairs (vec (concat new-pairs body-lifted))
          new-bindings (vec (mapcat identity all-pairs))]
      (if (= new-bindings (vec bindings-vec))
        form
        (list* 'let* new-bindings final-body-expr)))))

(defn- try-resolve-dispatch
  "Try to resolve a generic deftm call using arg types from env.
   Returns resolved form or nil."
  [expr env]
  (when (and (seq? expr) (symbol? (first expr)) (namespace (first expr))
             (not (.contains ^String (name (first expr)) "_m_")))
    (when-let [v (try (resolve (first expr)) (catch Exception _ nil))]
      (when-let [dt-atom (:raster.core/dispatch-table (meta v))]
        (let [dt @dt-atom
              args (rest expr)
              arg-tags (mapv #(inf/infer-arg-tag % env) args)
              entries (get dt (count args))]
          (when (seq entries)
            ;; Only resolve when ALL arg types are known and concrete
            (let [all-known? (every? some? arg-tags)
                  match (when all-known?
                          (or (first (filter (fn [e]
                                               (= (vec arg-tags) (vec (:tags e))))
                                             entries))
                              ;; No exact match — try parametric specialization for (All [T]) functions
                              (when-let [try-ps (requiring-resolve
                                                 'raster.compiler.core.inference/try-parametric-specialize!)]
                                (when (try-ps (symbol (namespace (first expr)) (name (first expr))) arg-tags)
                                  (let [new-entries (get @dt-atom (count args))]
                                    (first (filter (fn [e]
                                                     (= (vec arg-tags) (vec (:tags e))))
                                                   new-entries)))))))]
              (when match
                (let [mn (str (types/mangle (symbol (name (first expr))) (:tags match)))
                      ms (symbol (str (:mangled-ns match)) mn)
                      ms-impl (symbol (str (:mangled-ns match)) (str mn "-impl"))]
                  ;; Emit .invk form for typed dispatch (like the walker does)
                  ;; This avoids IFn boxing — calls the typed interface directly
                  (when (try (resolve ms) (catch Exception _ nil))
                    (let [impl-v (try (resolve ms-impl) (catch Exception _ nil))
                          typed-iface (:typed-iface match)
                          ret-tag (:raster.core/return-tag
                                   (try (meta (resolve ms)) (catch Exception _ nil)))
                          primitive-ret? (contains? #{'double 'float 'long 'int 'boolean} ret-tag)
                          ;; Bytecode-compiled deftm IFn stubs throw — must use .invk.
                          ;; Exclude alloc ops: hoisting/buffer-fusion pattern-match on
                          ;; mangled alloc calls, not .invk forms.
                          bc-compiled? (when (and impl-v (not primitive-ret?)
                                                  (not (op/alloc-op? (first expr))))
                                         (try (not (fn? @(resolve ms-impl)))
                                              (catch Exception _ false)))]
                      (if (and impl-v typed-iface (or primitive-ret? bc-compiled?))
                        ;; Full devirtualization: .invk impl-sym with typed interface tag
                        (util/make-invk (vary-meta ms-impl assoc :tag typed-iface) args (meta expr))
                        ;; Mangled call — still resolved, just not .invk
                        (let [r (cons ms args)]
                          (if-let [m (meta expr)] (with-meta r m) r))))))))))))))

(defn- resolve-dispatch-walk
  "Recursively resolve generic deftm calls in an expression.
   Tracks types through let/loop bindings for inner resolution.
   Also resolves constant var references (e.g. raster.numeric/neg-inf)."
  [expr env]
  (cond
    ;; Resolve constant var references (plain values like neg-inf, pi)
    ;; Skip: functions, generic dispatch vars, deftm impl singletons
    (and (symbol? expr) (namespace expr) (not (get env expr)))
    (or (try (when-let [v (resolve expr)]
               (let [val @v]
                 (when (and (var? v)
                            (not (fn? val))
                            (not (:raster.core/generic-function (meta v)))
                            ;; Only resolve primitive constants (numbers, strings, etc.)
                            ;; Don't resolve Java objects (impl singletons, arrays, etc.)
                            (or (number? val) (string? val) (boolean? val) (nil? val)))
                   val)))
             (catch Exception _ nil))
        expr)
    (not (seq? expr)) expr
    ;; Loop/loop*: propagate init types for loop vars into body
    (contains? #{'loop 'loop*} (first expr))
    (let [[head bindings-vec & body] expr
          pairs (partition 2 bindings-vec)
          loop-env (reduce (fn [e [sym init]]
                             (let [resolved (resolve-dispatch-walk init e)
                                   tag (inf/infer-arg-tag resolved e)]
                               (if tag (assoc e sym tag) e)))
                           env pairs)
          new-bindings (vec (mapcat (fn [[sym init]]
                                      [sym (resolve-dispatch-walk init env)])
                                    pairs))
          new-body (map #(resolve-dispatch-walk % loop-env) body)]
      (let [r (apply list head new-bindings new-body)]
        (if-let [m (meta expr)] (with-meta r m) r)))
    ;; Let/let*: propagate binding types
    (contains? #{'let 'let*} (first expr))
    (let [[head bindings-vec & body] expr
          pairs (partition 2 bindings-vec)
          [let-env new-bindings]
          (reduce (fn [[e acc] [sym init]]
                    (let [resolved (resolve-dispatch-walk init e)
                          tag (inf/infer-arg-tag resolved e)]
                      [(if tag (assoc e sym tag) e)
                       (conj acc sym resolved)]))
                  [env []] pairs)
          new-body (map #(resolve-dispatch-walk % let-env) body)
          r (apply list head (vec new-bindings) new-body)]
      (if-let [m (meta expr)] (with-meta r m) r))
    ;; dotimes: propagate loop counter type (long) into body
    (= 'dotimes (first expr))
    (let [[_ [sym bound] & body] expr
          resolved-bound (resolve-dispatch-walk bound env)
          loop-env (assoc env sym 'long)
          new-body (map #(resolve-dispatch-walk % loop-env) body)
          r (apply list 'dotimes [sym resolved-bound] new-body)]
      (if-let [m (meta expr)] (with-meta r m) r))
    ;; par/pmap: (raster.par/pmap idx bound cast body) — idx is long
    (and (symbol? (first expr))
         (contains? #{'raster.par/pmap 'par/pmap} (first expr)))
    (let [[head idx bound cast & body-forms] expr
          par-env (assoc env idx 'long)
          resolved-bound (resolve-dispatch-walk bound env)
          new-body (map #(resolve-dispatch-walk % par-env) body-forms)
          r (apply list head idx resolved-bound cast new-body)]
      (if-let [m (meta expr)] (with-meta r m) r))
    ;; par/map!: (raster.par/map! out idx bound [:offset off] cast body) — idx is long
    (and (symbol? (first expr))
         (contains? #{'raster.par/map! 'par/map!} (first expr)))
    (let [args (rest expr)
          ;; Parse: out idx bound [:offset off] cast body
          ;; Find idx — second symbol after head
          [out idx bound & rest-args] args
          par-env (assoc env idx 'long)
          ;; Walk all non-idx/non-out arguments
          resolved-args (map #(resolve-dispatch-walk % par-env) rest-args)
          resolved-bound (resolve-dispatch-walk bound env)
          r (apply list (first expr) out idx resolved-bound resolved-args)]
      (if-let [m (meta expr)] (with-meta r m) r))
    ;; par/reduce: (raster.par/reduce acc init idx bound body) — acc type from init, idx is long
    (and (symbol? (first expr))
         (contains? #{'raster.par/reduce 'par/reduce} (first expr)))
    (let [[head acc init idx bound & body-forms] expr
          acc-tag (inf/infer-arg-tag init env)
          par-env (cond-> (assoc env idx 'long)
                    acc-tag (assoc acc acc-tag))
          resolved-init (resolve-dispatch-walk init env)
          resolved-bound (resolve-dispatch-walk bound env)
          new-body (map #(resolve-dispatch-walk % par-env) body-forms)
          r (apply list head acc resolved-init idx resolved-bound new-body)]
      (if-let [m (meta expr)] (with-meta r m) r))
    ;; Qualified symbol call: try dispatch resolution then recurse into args
    (and (symbol? (first expr)) (namespace (first expr)))
    (let [;; First recurse into args so nested calls get resolved
          walked-args (map #(resolve-dispatch-walk % env) (rest expr))
          walked-expr (let [r (apply list (first expr) walked-args)]
                        (if-let [m (meta expr)] (with-meta r m) r))]
      ;; Then try dispatch resolution on the form with resolved args
      (or (try-resolve-dispatch walked-expr env)
          walked-expr))
    :else (let [r (map #(resolve-dispatch-walk % env) expr)]
            (if (= (seq r) (seq expr))
              expr
              (let [new-form (apply list r)]
                (if-let [m (meta expr)] (with-meta new-form m) new-form))))))

(defn resolve-generic-deftm-calls
  "Resolve all generic deftm dispatch calls in a flat let* to mangled
   implementations. Infers argument types from the binding context
   and optional param-env (function parameter types).
   Late pipeline step — no AD template concerns at this point."
  ([form] (resolve-generic-deftm-calls form {}))
  ([form param-env]
   (if-not (form/binding-form? form)
     form
     (let [[head bindings-vec & body] form
           pairs (vec (partition 2 bindings-vec))
          ;; Build type env incrementally from bindings, seeded with params
           [env new-pairs]
           (reduce (fn [[env acc] [sym expr]]
                     (let [resolved (resolve-dispatch-walk expr env)
                           tag (or (:tag (meta sym))
                                   (inf/infer-arg-tag resolved env))
                          ;; Stamp inferred tag on binding symbol so downstream
                          ;; passes (closure extraction, bytecode) see it
                           sym (if (and tag (not (:tag (meta sym))))
                                 (vary-meta sym assoc :tag tag :raster.type/tag tag)
                                 sym)]
                       [(if tag (assoc env sym tag) env)
                        (conj acc [sym resolved])]))
                   [(or param-env {}) []] pairs)
           new-body (map #(resolve-dispatch-walk % env) body)
           r (list* head (vec (mapcat identity new-pairs)) new-body)]
       (if-let [m (meta form)] (with-meta r m) r)))))

(defn expand-for-backends
  "Post-AD expansion: fully inline template-backed ops for backend optimization.
   param-env seeds the type environment with function parameter types."
  ([form] (expand-for-backends form 3 nil))
  ([form max-depth] (expand-for-backends form max-depth nil))
  ([form max-depth param-env]
   (-> (inline-deftm-calls form max-depth false true)
       inline-invk
       flatten-nested-lets)))

