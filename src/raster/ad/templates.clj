(ns raster.ad.templates
  "Data-based rrule templates for flat AD codegen.

  Each template describes how to compute gradients for an operation as
  quoted S-expressions (data), not closures. This enables the AD transform
  to splice gradient code directly into the backward pass as flat let*
  bindings, making all allocations visible to buffer fusion.

  Templates are the single gradient representation: :grads (auto-compiled to
  :grads-fn) or :grads-fn directly. Both the compiler AD transform and runtime
  value+grad emit backward code from the same :grads-fn. Ops without a template
  are differentiated by inlining their body; the only per-op runtime pullback
  path (get-pullback-factory / closure fallback) is a generic net for
  un-templated opaque calls and holds no registrations by default.

  Template structure:
    {:params  [x y]        ;; formal parameter symbols
     :result  nil or sym   ;; symbol bound to forward result (nil if unused)
     :adjoint dy           ;; symbol for incoming adjoint
     :grads   [expr1 ...]  ;; gradient expressions, one per param
     :grads-fn (optional)  ;; (fn [ctx params result-sym adjoint-sym gensym-fn]
                           ;;   -> [updated-ctx [grad-sym ...]]) for flat bindings}"
  (:require [raster.compiler.support.mangled :as mangled]
            [raster.compiler.core.util :as util]
            [raster.ad.tangent :as tangent]))

;; ================================================================
;; Template registry
;; ================================================================

(defonce ^:private template-registry (atom {}))

(declare ^:private compile-grads-to-fn)

(defn register-template!
  "Register an AD gradient rule for an operation.

   op: symbol identifying the operation (e.g. '+, 'Math/sin)
   template: map with keys:
     :params    — formal parameter symbols (documentation; also used for static :grads compilation)
     :result    — symbol for forward result (nil if unused)
     :adjoint   — symbol for incoming adjoint (dy)
     :grads     — static gradient expressions (one per param), auto-compiled into :grads-fn
     :grads-fn  — dynamic code generator (fn [ctx params result adjoint gensym-fn] -> [ctx grad-syms])
     :pullback-factory — runtime pullback (fn [result & args] -> (fn [dy] -> [grad1 ...]))

   If :grads is provided without :grads-fn, it is automatically compiled into a
   :grads-fn at registration time. At instantiation, only :grads-fn is used."
  [op template]
  (when-let [grads (:grads template)]
    (when-let [params (:params template)]
      (when (not= (count grads) (count params))
        (throw (ex-info (str "AD template for " op ": :grads count (" (count grads)
                             ") doesn't match :params count (" (count params) ")")
                        {:op op :params params :grads grads})))))
  ;; Auto-compile :grads into :grads-fn when no :grads-fn is provided
  (let [template (if (and (:grads template) (not (:grads-fn template)))
                   (assoc template :grads-fn (compile-grads-to-fn template))
                   template)]
    (swap! template-registry assoc op template))
  nil)

(defn get-template
  "Look up the template for an operation. Returns nil if none registered."
  [op]
  (get @template-registry op))

(defn has-template?
  "True if op has a registered template."
  [op]
  (contains? @template-registry op))

(defn registered-ops
  "Return the set of all ops with registered templates."
  []
  (set (keys @template-registry)))

(defn all-templates
  "Return the full template registry map."
  []
  @template-registry)

(defn merge-into-template!
  "Merge additional keys into an existing template, or create a new entry.
  Validates :params count against the function's arity at registration time
  if the var is already defined."
  [op kvs]
  (when-let [declared-params (:params kvs)]
    (when-let [v (try (resolve op) (catch Exception _ nil))]
      (when-let [arglists (:arglists (meta v))]
        (let [expected (count declared-params)
              actual-arities (set (map count arglists))]
          (when (and (seq actual-arities)
                     (not (contains? actual-arities expected)))
            (binding [*out* *err*]
              (println (str "WARNING: AD template for " op " declares " expected
                            " params " (vec declared-params) " but function has arities "
                            actual-arities ". Template params may be stale."))))))))
  (swap! template-registry update op
         (fn [existing] (merge (or existing {}) kvs)))
  nil)

(defn has-ad-rule?
  "True if op has an explicit AD gradient rule (:grads or :grads-fn).
   This is the check the compiler uses to decide whether to inline a call
   or keep it symbolic for the AD transform."
  [op]
  (when-let [t (get @template-registry op)]
    (or (:grads t) (:grads-fn t))))

(defn get-pullback-factory
  "Get the runtime pullback factory for an op, if registered.
   Returns (fn [result & args] -> (fn [dy] -> [grad1 ...])) or nil."
  [op]
  (or (:pullback-factory (get @template-registry op))
      ;; Legacy: check :closure key for backward compatibility
      (:closure (get @template-registry op))))

;; ================================================================
;; Template instantiation
;; ================================================================

;; Gradient-template substitution uses the unified capture-avoiding
;; `util/subst-syms` (generic over form/scope-info). Template :grads are
;; hand-written gradient S-exprs with no closures/par binders, and the smap
;; values are plain symbols (param→actual, result→sym, adjoint→sym), so this is
;; a strict superset of the old hand-rolled let*/loop*/dotimes substitutor.

(defn- compile-grads-to-fn
  "Compile static :grads data into a :grads-fn closure.
  This unifies the two template paths so instantiation always uses :grads-fn."
  [{:keys [params result adjoint grads]}]
  (fn [ctx actual-params result-sym adjoint-sym gensym-fn]
    (let [param-subst (zipmap params actual-params)
          result-subst (if result {result result-sym} {})
          adj-subst {adjoint adjoint-sym}
          smap (merge param-subst result-subst adj-subst)
          grad-bindings (mapv (fn [i grad-expr]
                                (let [grad-sym (gensym-fn (str "dg_" (name (nth params i))))
                                      substituted (util/subst-syms smap grad-expr)]
                                  [grad-sym substituted]))
                              (range) grads)]
      [(reduce (fn [c [sym expr]] (update c :bindings into [sym expr]))
               ctx grad-bindings)
       (mapv first grad-bindings)])))

(defn instantiate-template-ctx
  "Instantiate a template via :grads-fn with BindCtx threading.

  All templates have a :grads-fn (static :grads are auto-compiled at registration).
  The :grads-fn receives actual call-site params and produces gradient bindings.

  Returns [updated-ctx, [grad-sym ...]] where grad-syms are the symbols
  bound to gradient values for each parameter."
  [template actual-params result-sym adjoint-sym ctx]
  (let [grads-fn (or (:grads-fn template)
                     (throw (ex-info "AD template has no :grads-fn (and no :grads to compile)"
                                     {:template (select-keys template [:params :result :adjoint])})))
        [ctx' grad-syms :as result] (grads-fn ctx actual-params result-sym adjoint-sym (:gensym-fn ctx))]
    ;; Validate gradient count matches actual parameter count
    (when (and grad-syms (not= (count grad-syms) (count actual-params)))
      (throw (ex-info (str "AD template gradient count mismatch: " (count grad-syms)
                           " gradients returned for " (count actual-params)
                           " params. The :grads-fn return vector must have one entry per param.")
                      {:grad-count (count grad-syms)
                       :param-count (count actual-params)
                       :actual-params (vec actual-params)
                       :declared-params (:params template)})))
    result))

;; ================================================================
;; Resolve generic deftm calls emitted by grads-fn
;; ================================================================

(defn- try-resolve-call-with-tags
  "Resolve a generic deftm call to its mangled implementation using arg-tags
   from the forward call as type context. Returns resolved form or nil."
  [expr forward-tags]
  (when (and (seq? expr) (symbol? (first expr)) (namespace (first expr))
             (not (.contains ^String (name (first expr)) "_m_")))
    (try
      (when-let [v (resolve (first expr))]
        (when-let [dt-atom (:raster.core/dispatch-table (meta v))]
          (let [dt @dt-atom
                arity (dec (count expr))
                entries (get dt arity)]
            (when (seq entries)
              ;; Build expected tags: backward helpers typically use the same
              ;; element types as the forward call (doubles→doubles, etc.)
              ;; Use the forward tags as a pool of known types
              (let [dominant-tag (first forward-tags) ;; e.g., doubles
                    ;; Match: find entry whose tags are all the dominant type
                    match (or (first (filter (fn [entry]
                                               (every? #(= % dominant-tag) (:tags entry)))
                                             entries))
                             ;; Or single-method fallback
                              (when (= 1 (count entries)) (first entries)))]
                (when match
                  (let [mangled-name (str (name (first expr)) "_m_"
                                          (clojure.string/join "_" (map name (:tags match))))
                        mangled (symbol (str (:mangled-ns match)) mangled-name)]
                    (when (resolve mangled)
                      (cons mangled (rest expr))))))))))
      (catch Exception _ nil))))

(defn- resolve-in-expr
  "Recursively resolve generic deftm calls in an expression."
  [expr forward-tags]
  (cond
    (not (seq? expr)) expr
    (and (symbol? (first expr)) (namespace (first expr)))
    (or (try-resolve-call-with-tags expr forward-tags)
        (let [resolved (map #(resolve-in-expr % forward-tags) expr)]
          (if (= (seq resolved) (seq expr)) expr (apply list resolved))))
    :else (let [resolved (map #(resolve-in-expr % forward-tags) expr)]
            (if (= (seq resolved) (seq expr)) expr (apply list resolved)))))

(defn resolve-emitted-calls
  "Resolve generic deftm calls in template-emitted bindings using the
   forward call's arg-tags as type context. This devirtualizes backward
   helper calls at AD transform time, avoiding runtime dispatch."
  [bindings forward-tags]
  (vec (mapcat (fn [[sym expr]]
                 [sym (resolve-in-expr expr forward-tags)])
               (partition 2 bindings))))

;; ================================================================
;; Qualified numeric op → base op reverse mapping
;; ================================================================

(def ^:private qualified->base-op
  "Reverse mapping from Dual-compatible qualified ops to base registry keys.
  The forward pass qualifies ops (e.g. * → raster.numeric/*) for Dual flow;
  this map reverses that for template lookup in second-order AD."
  {'raster.numeric/+    '+,    'raster.numeric/-    '-,
   'raster.numeric/*    '*,    'raster.numeric//    '/,
   'raster.math/sin     'Math/sin,  'raster.math/cos     'Math/cos,
   'raster.math/exp     'Math/exp,  'raster.math/log     'Math/log,
   'raster.numeric/sqrt 'Math/sqrt, 'raster.numeric/pow  'Math/pow,
   'raster.math/tan     'Math/tan,  'raster.numeric/abs  'Math/abs,
   'raster.math/asin    'Math/asin, 'raster.math/acos    'Math/acos,
   'raster.math/atan    'Math/atan, 'raster.math/atan2   'Math/atan2,
   ;; raster.arrays/aget is the polymorphic-dispatch spelling of the array read;
   ;; it differentiates exactly like clojure.core/aget (scatter dy into arr[i]).
   'raster.arrays/aget  'clojure.core/aget})

;; ================================================================
;; Mangled name resolution
;; ================================================================

(defn- extract-base-op
  "Extract the base operation name from a mangled symbol."
  [sym]
  (when (symbol? sym)
    (let [n (name sym)]
      (cond
        (.startsWith n "_plus_")  '+
        (.startsWith n "_minus_") '-
        (.startsWith n "_star_")  '*
        (.startsWith n "_div_")   '/
        (.startsWith n "sin_")    'Math/sin
        (.startsWith n "cos_")    'Math/cos
        (.startsWith n "exp_")    'Math/exp
        (.startsWith n "log_")    'Math/log
        (.startsWith n "sqrt_")   'Math/sqrt
        (.startsWith n "pow_")    'Math/pow
        (.startsWith n "abs_")    'Math/abs
        (.startsWith n "tan_")    'Math/tan
        (.startsWith n "asin_")   'Math/asin
        (.startsWith n "acos_")   'Math/acos
        (.startsWith n "atan2_")  'Math/atan2
        (.startsWith n "atan_")   'Math/atan
        (.startsWith n "min_")    'Math/min
        (.startsWith n "max_")    'Math/max
        (.startsWith n "fma_")    'Math/fma
        :else nil))))

(defn- extract-deftm-base
  "Extract the base qualified symbol from a deftm-mangled name.
  E.g. raster.nn/dense_m_Object_Object_Object → raster.nn/dense"
  [op]
  (mangled/extract-deftm-base op))

(defn- normalize-op
  "Reduce any op symbol to its canonical template key.
  Handles: -impl suffix stripping, qualified→base mapping,
  mangled arithmetic names, deftm _m_ mangled names."
  [op]
  (when (symbol? op)
    (let [n (name op)
          ns-str (namespace op)
          ;; Strip -impl suffix first (from .invk calls)
          [n ns-str] (if (.endsWith ^String n "-impl")
                       [(mangled/strip-impl-suffix n) ns-str]
                       [n ns-str])
          reconstructed (if ns-str (symbol ns-str n) (symbol n))]
      ;; 1. Direct lookup key
      (or (when (get-template reconstructed) reconstructed)
          ;; 2. Qualified numeric op → base (raster.numeric/* → *)
          (get qualified->base-op reconstructed)
          ;; 3. Mangled arithmetic/math name (_plus_m_... → '+, sin_m_... → 'Math/sin)
          (extract-base-op (symbol n))
          ;; 4. deftm mangled name (ns/foo_m_double_double → ns/foo)
          (extract-deftm-base reconstructed)
          ;; 5. As-is (already canonical)
          reconstructed))))

(defn resolve-template
  "Look up a template by normalizing the op to its canonical form.
  Returns [template canonical-op] or nil."
  [op]
  (when-let [canonical (normalize-op op)]
    (when-let [t (get-template canonical)]
      [t canonical])))

(defn template-pullback
  "Return a runtime pullback for an op from its single available gradient
  representation. When the op has a :grads-fn, derive the pullback by
  instantiating the SAME backward code the compiler emits and eval'ing it as a
  closure (so interpreted AD and gradient tests exercise the one source of truth,
  not a separately-maintained duplicate). When the op has only a hand-written
  :pullback-factory (opaque/implicit ops with no static grads-fn — einsum,
  rearrange, solve-effort, pack/unpack-heads), fall back to it. Returns
  (fn [result & args] -> (fn [adjoint] -> [grad ...])) or nil if no rule exists."
  [op]
  (if-let [[template _] (resolve-template op)]
    (if (:grads-fn template)
      (fn [result & args]
        (fn [adjoint]
          (let [param-syms (mapv (fn [i] (gensym (str "tp_p" i "_"))) (range (count args)))
                result-sym (gensym "tp_res_")
                adj-sym    (gensym "tp_adj_")
                ctx        {:bindings [] :gensym-fn (fn [p] (gensym (str p "_")))}
                [ctx' grad-syms] (instantiate-template-ctx template param-syms result-sym adj-sym ctx)
                f (eval (list 'fn (vec (concat param-syms [result-sym adj-sym]))
                              (list 'let* (vec (:bindings ctx'))
                                    (vec grad-syms))))]
            (apply f (concat args [result adjoint])))))
      (get-pullback-factory op))
    (get-pullback-factory op)))

;; ================================================================
;; Arithmetic templates
;; ================================================================

(register-template! '+
                    {:params '[x y] :result nil :adjoint 'dy
                     :grads '[dy dy]})

(register-template! 'clojure.core/+ (get-template '+))
(register-template! 'raster.numeric/+ (get-template '+))

;; grad-acc is nil-safe +, same gradient semantics
(register-template! 'raster.ad.reverse/grad-acc
                    {:params '[a b] :result nil :adjoint 'dy
                     :grads '[dy dy]})

(register-template! '-
                    {:params '[x y] :result nil :adjoint 'dy
                     :grads '[dy (raster.numeric/- dy)]
                     :grads-fn (fn [ctx actual-params _result-sym adjoint-sym gensym-fn]
                                 (if (= 1 (count actual-params))
                 ;; Unary negation: grad = -dy
                                   (let [g (gensym-fn "dg_x")]
                                     [(update ctx :bindings into [g (list 'raster.numeric/- adjoint-sym)])
                                      [g]])
                 ;; Binary subtraction: grad_x = dy, grad_y = -dy
                                   (let [gx (gensym-fn "dg_x")
                                         gy (gensym-fn "dg_y")]
                                     [(update ctx :bindings into [gx adjoint-sym
                                                                  gy (list 'raster.numeric/- adjoint-sym)])
                                      [gx gy]])))})

(register-template! 'clojure.core/- (get-template '-))

(register-template! '*
                    {:params '[x y] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy y) (raster.numeric/* dy x)]})

(register-template! 'clojure.core/* (get-template '*))

(register-template! '/
                    {:params '[x y] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// dy y)
                              (raster.numeric/- (raster.numeric// (raster.numeric/* dy x)
                                                                  (raster.numeric/* y y)))]})

(register-template! 'clojure.core// (get-template '/))

;; Qualified raster.numeric variants
(register-template! 'raster.numeric/- (get-template '-))
(register-template! 'raster.numeric/* (get-template '*))
(register-template! 'raster.numeric// (get-template '/))

;; ================================================================
;; Math templates
;; ================================================================

;; sin: d/dx sin(x) = cos(x) * dy
(register-template! 'Math/sin
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.math/cos x))]})

;; cos: d/dx cos(x) = -sin(x) * dy
(register-template! 'Math/cos
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/- (raster.math/sin x)))]})

;; exp: d/dx exp(x) = exp(x) * dy  (uses result for efficiency)
(register-template! 'Math/exp
                    {:params '[x] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy r)]})

;; log: d/dx log(x) = dy/x
(register-template! 'Math/log
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// dy x)]})

;; sqrt: d/dx sqrt(x) = dy / (2*sqrt(x))
(register-template! 'Math/sqrt
                    {:params '[x] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric// dy (raster.numeric/* 2.0 r))]})

;; pow: d/dx x^n = n*x^(n-1)*dy, d/dn x^n = x^n*log(x)*dy
(register-template! 'Math/pow
                    {:params '[x n] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/* n (raster.numeric/pow x (raster.numeric/- n 1.0))))
                              (raster.numeric/* dy (raster.numeric/* r (raster.math/log x)))]})

;; abs: d/dx |x| = sign(x) * dy
(register-template! 'Math/abs
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (Math/signum (double x)))]})

;; tan: d/dx tan(x) = (1 + tan²(x)) * dy
(register-template! 'Math/tan
                    {:params '[x] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/+ 1.0 (raster.numeric/* r r)))]})

;; asin: d/dx asin(x) = dy / sqrt(1 - x²)
(register-template! 'Math/asin
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// dy (raster.numeric/sqrt (raster.numeric/- 1.0 (raster.numeric/* x x))))]})

;; acos: d/dx acos(x) = -dy / sqrt(1 - x²)
(register-template! 'Math/acos
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/- (raster.numeric// dy (raster.numeric/sqrt (raster.numeric/- 1.0 (raster.numeric/* x x)))))]})

;; atan: d/dx atan(x) = dy / (1 + x²)
(register-template! 'Math/atan
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// dy (raster.numeric/+ 1.0 (raster.numeric/* x x)))]})

;; atan2: d/dy atan2(y,x) = x/(x²+y²)*dy, d/dx atan2(y,x) = -y/(x²+y²)*dy
(register-template! 'Math/atan2
                    {:params '[y x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// (raster.numeric/* dy x)
                                                (raster.numeric/+ (raster.numeric/* x x) (raster.numeric/* y y)))
                              (raster.numeric/- (raster.numeric// (raster.numeric/* dy y)
                                                                  (raster.numeric/+ (raster.numeric/* x x) (raster.numeric/* y y))))]})

;; min: d/dx min(x,y) = dy if x <= y, 0 otherwise (subgradient)
(register-template! 'Math/min
                    {:params '[x y] :result nil :adjoint 'dy
                     :grads-fn (fn [ctx [x y] _result-sym adjoint-sym gensym-fn]
                                 (let [gx (gensym-fn "dg_x")
                                       gy (gensym-fn "dg_y")]
                                   [(update ctx :bindings into
                                            [gx (list 'if (list 'clojure.core/<= (list 'double x) (list 'double y))
                                                      adjoint-sym 0.0)
                                             gy (list 'if (list 'clojure.core/<= (list 'double x) (list 'double y))
                                                      0.0 adjoint-sym)])
                                    [gx gy]]))})

;; max: d/dx max(x,y) = dy if x >= y, 0 otherwise (subgradient)
(register-template! 'Math/max
                    {:params '[x y] :result nil :adjoint 'dy
                     :grads-fn (fn [ctx [x y] _result-sym adjoint-sym gensym-fn]
                                 (let [gx (gensym-fn "dg_x")
                                       gy (gensym-fn "dg_y")]
                                   [(update ctx :bindings into
                                            [gx (list 'if (list 'clojure.core/>= (list 'double x) (list 'double y))
                                                      adjoint-sym 0.0)
                                             gy (list 'if (list 'clojure.core/>= (list 'double x) (list 'double y))
                                                      0.0 adjoint-sym)])
                                    [gx gy]]))})

;; fma: d/da fma(a,b,c) = b*dy, d/db = a*dy, d/dc = dy
(register-template! 'Math/fma
                    {:params '[a b c] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy b) (raster.numeric/* dy a) dy]})

;; sinh: d/dx sinh(x) = cosh(x) * dy
(register-template! 'Math/sinh
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.math/cosh x))]})

;; cosh: d/dx cosh(x) = sinh(x) * dy
(register-template! 'Math/cosh
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.math/sinh x))]})

;; tanh: d/dx tanh(x) = (1 - tanh²(x)) * dy
(register-template! 'Math/tanh
                    {:params '[x] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/- 1.0 (raster.numeric/* r r)))]})

;; log1p: d/dx log(1+x) = dy / (1+x)
(register-template! 'Math/log1p
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// dy (raster.numeric/+ 1.0 x))]})

;; expm1: d/dx (exp(x)-1) = exp(x) * dy = (expm1(x)+1) * dy
(register-template! 'Math/expm1
                    {:params '[x] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/+ r 1.0))]})

;; cbrt: d/dx cbrt(x) = dy / (3 * cbrt(x)²)
(register-template! 'Math/cbrt
                    {:params '[x] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric// dy (raster.numeric/* 3.0 (raster.numeric/* r r)))]})

;; log10: d/dx log10(x) = dy / (x * ln(10))
(register-template! 'Math/log10
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// dy (raster.numeric/* x 2.302585092994046))]})

;; hypot: d/dx hypot(x,y) = x/hypot(x,y)*dy, d/dy = y/hypot(x,y)*dy
(register-template! 'Math/hypot
                    {:params '[x y] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric// x r))
                              (raster.numeric/* dy (raster.numeric// y r))]})

;; Qualified raster.math/raster.numeric variants (eagerly registered)
(register-template! 'raster.math/sin  (get-template 'Math/sin))
(register-template! 'raster.math/cos  (get-template 'Math/cos))
(register-template! 'raster.math/exp  (get-template 'Math/exp))
(register-template! 'raster.math/log  (get-template 'Math/log))
(register-template! 'raster.numeric/sqrt (get-template 'Math/sqrt))
(register-template! 'raster.numeric/pow  (get-template 'Math/pow))
(register-template! 'raster.numeric/abs  (get-template 'Math/abs))
(register-template! 'raster.math/tan  (get-template 'Math/tan))
(register-template! 'raster.math/asin (get-template 'Math/asin))
(register-template! 'raster.math/acos (get-template 'Math/acos))
(register-template! 'raster.math/atan (get-template 'Math/atan))
(register-template! 'raster.math/atan2 (get-template 'Math/atan2))
(register-template! 'raster.math/sinh (get-template 'Math/sinh))
(register-template! 'raster.math/cosh (get-template 'Math/cosh))
(register-template! 'raster.math/tanh (get-template 'Math/tanh))
(register-template! 'raster.math/log1p (get-template 'Math/log1p))
(register-template! 'raster.math/expm1 (get-template 'Math/expm1))
(register-template! 'raster.math/cbrt (get-template 'Math/cbrt))
(register-template! 'raster.math/log10 (get-template 'Math/log10))
(register-template! 'raster.math/hypot (get-template 'Math/hypot))

;; ================================================================
;; Numeric cast templates (identity passthrough)
;; ================================================================

(register-template! 'double
                    {:params '[x] :result nil :adjoint 'dy :grads '[dy]})

(register-template! 'clojure.core/double (get-template 'double))

(register-template! 'float
                    {:params '[x] :result nil :adjoint 'dy :grads '[dy]})

(register-template! 'clojure.core/float (get-template 'float))

;; Integer casts: zero gradient (non-differentiable)
(register-template! 'long
                    {:params '[x] :result nil :adjoint 'dy :grads '[0.0]})

(register-template! 'clojure.core/long (get-template 'long))

(register-template! 'int
                    {:params '[x] :result nil :adjoint 'dy :grads '[0.0]})

(register-template! 'clojure.core/int (get-template 'int))

;; ================================================================
;; Special function templates
;; ================================================================

;; lgamma: d/dx lgamma(x) = digamma(x) * dy
(register-template! 'raster.sci.special/lgamma
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.sci.special/digamma x))]})

;; digamma: d/dx digamma(x) = trigamma(x) * dy
(register-template! 'raster.sci.special/digamma
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.sci.special/trigamma x))]})

;; erf: d/dx erf(x) = 2/sqrt(pi) * exp(-x^2) * dy
(register-template! 'raster.sci.special/erf
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/* (/ 2.0 (Math/sqrt Math/PI))
                                                                     (raster.math/exp (raster.numeric/- (raster.numeric/* x x)))))]})

;; erfinv: d/dx erfinv(y) = sqrt(pi)/2 * exp(erfinv(y)^2) * dy
(register-template! 'raster.sci.special/erfinv
                    {:params '[y] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/* (/ (Math/sqrt Math/PI) 2.0)
                                                                     (raster.math/exp (raster.numeric/* r r))))]})

;; expit (sigmoid): d/dx sigma(x) = sigma(x) * (1 - sigma(x)) * dy
(register-template! 'raster.sci.special/expit
                    {:params '[x] :result 'r :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/* r (raster.numeric/- 1.0 r)))]})

;; logit: d/dx logit(x) = 1/(x*(1-x)) * dy
(register-template! 'raster.sci.special/logit
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric// dy (raster.numeric/* x (raster.numeric/- 1.0 x)))]})

;; log1pexp: d/dx log(1+exp(x)) = sigmoid(x) * dy
(register-template! 'raster.sci.special/log1pexp
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric// 1.0 (raster.numeric/+ 1.0 (raster.math/exp (raster.numeric/- x)))))]})

;; besselj0: d/dx J0(x) = -J1(x) * dy
(register-template! 'raster.sci.special/besselj0
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/- (raster.sci.special/besselj1 x)))]})

;; besselj1: d/dx J1(x) = (J0(x) - J1(x)/x) * dy
(register-template! 'raster.sci.special/besselj1
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[(raster.numeric/* dy (raster.numeric/- (raster.sci.special/besselj0 x)
                                                                     (raster.numeric// (raster.sci.special/besselj1 x) x)))]})

;; betainc: partial derivatives w.r.t. x only (a,b treated as constants)
;; d/dx I_x(a,b) = x^(a-1) * (1-x)^(b-1) / B(a,b) * dy
(register-template! 'raster.sci.special/betainc
                    {:params '[a b x] :result nil :adjoint 'dy
                     :grads-fn (fn [ctx [a b x] _result-sym adjoint-sym gensym-fn]
                                 (let [dx (gensym-fn "dg_x")]
                                   [(update ctx :bindings into
                                            [dx (list 'raster.numeric/*
                                                      adjoint-sym
                                                      (list 'raster.numeric//
                                                            (list 'raster.numeric/*
                                                                  (list 'Math/pow x (list 'raster.numeric/- a 1.0))
                                                                  (list 'Math/pow (list 'raster.numeric/- 1.0 x) (list 'raster.numeric/- b 1.0)))
                                                            (list 'raster.sci.special/beta a b)))])
                                    [nil nil dx]]))})

;; ================================================================
;; Array element access
;; ================================================================

;; Array infrastructure: these operations touch active arrays but produce
;; non-differentiable results (lengths, allocations, mutations).

;; alength: y = len(arr) → no gradient (integer result, not differentiable)
(register-template! 'clojure.core/alength
                    {:params '[arr] :result nil :adjoint 'dy
                     :grads '[nil]})

;; aset: arr[i] = v → no gradient through aset itself (mutation side-effect).
;; The gradient for mutation flows through the stored value's downstream reads.
(register-template! 'clojure.core/aset
                    {:params '[arr i v] :result nil :adjoint 'dy
                     :grads '[nil nil nil]})

;; Array allocation: creates new arrays, no gradient flow
(register-template! 'clojure.core/double-array
                    {:params '[n] :result nil :adjoint 'dy
                     :grads '[nil]})
(register-template! 'clojure.core/float-array
                    {:params '[n] :result nil :adjoint 'dy
                     :grads '[nil]})
(register-template! 'clojure.core/long-array
                    {:params '[n] :result nil :adjoint 'dy
                     :grads '[nil]})
(register-template! 'clojure.core/int-array
                    {:params '[n] :result nil :adjoint 'dy
                     :grads '[nil]})

;; aget: y = arr[i] → d_arr = scatter(dy, i, len(arr)), d_i = nil
;; The gradient w.r.t. the array is a one-hot: zeros except dy at position i.
(register-template! 'clojure.core/aget
                    {:params '[arr i] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [arr i] _result-sym adjoint-sym gensym-fn]
                       (let [d-arr (gensym-fn "d_arr")]
                         [(update ctx :bindings into
                                  [d-arr (list 'raster.ad.reverse/aget-grad arr i adjoint-sym)])
                          [d-arr nil]]))})

;; ================================================================
;; Gradient control templates
;; ================================================================

;; stop-gradient: identity forward, zero gradient backward
(register-template! 'raster.ad.forward/stop-gradient
                    {:params '[x] :result nil :adjoint 'dy
                     :grads '[0.0]})

;; straight-through: applies f to x forward, passes gradient through as identity
(register-template! 'raster.ad.forward/straight-through
                    {:params '[f x] :result nil :adjoint 'dy
                     :grads '[0.0 dy]})

;; ================================================================
;; Deep learning array operation templates
;;
;; These use :grads-fn to emit flat let* bindings with BLAS calls,
;; making all allocations and computations visible to the compiler
;; for buffer fusion, SIMD, and GPU optimization.
;; ================================================================

;; mse-loss: loss = mean((pred - target)^2)
;; d_pred[i] = 2*(pred[i] - target[i])/n * dy
(register-template! 'raster.dl.loss/mse-loss
                    {:params '[pred target n] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [pred target n] _result-sym adjoint-sym gensym-fn]
                       (let [d-pred (gensym-fn "d_pred")
           ;; Emit: d_pred = (let [out (double-array n)]
           ;;                  (dotimes [i n]
           ;;                    (aset out i (* (/ (* 2.0 dy) n) (- pred[i] target[i]))))
           ;;                  out)
           ;; But as flat S-expressions the compiler can see through
                             scale (gensym-fn "mse_scale")
                             n-long (gensym-fn "mse_n")
                             ctx1 (update ctx :bindings into
                                          [n-long (list 'long n)
                                           scale (list 'raster.numeric// (list 'raster.numeric/* 2.0 adjoint-sym) (list 'double n-long))
                                           d-pred (list 'raster.linalg.blas/daxpy-diff! pred target scale n-long)])]
                         [ctx1 [d-pred nil nil]]))})

;; transpose-2d: out[j*rows+i] = a[i*cols+j]
;; d_a[i*cols+j] = d_out[j*rows+i]  i.e. d_a = transpose(d_out, cols, rows)
(register-template! 'raster.dl.nn/transpose-2d
                    {:params '[a rows cols] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [a rows cols] _result-sym adjoint-sym gensym-fn]
                       (let [d-a (gensym-fn "d_a")]
                         [(update ctx :bindings into
                                  [d-a (list 'raster.dl.nn/transpose-2d adjoint-sym cols rows)])
                          [d-a nil nil]]))})

;; matmul: C = A @ B, A:[M,K] B:[K,N] -> C:[M,N]
;; dA = dC @ B^T (dgemm-nt: dC[M,N] @ B[K,N]^T -> dA[M,K])
;; dB = A^T @ dC (dgemm-tn: A[M,K]^T @ dC[M,N] -> dB[K,N])
(register-template! 'raster.dl.nn/matmul
                    {:params '[A B m k n] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [A B m k n] _result-sym adjoint-sym gensym-fn]
                       (let [dA (gensym-fn "dA")
                             dB (gensym-fn "dB")]
                         [(update ctx :bindings into
                                  [dA (list 'raster.dl.nn/matmul-dA adjoint-sym B m k n)
                                   dB (list 'raster.dl.nn/matmul-dB A adjoint-sym m k n)])
                          [dA dB nil nil nil]]))})

;; linear: y = x @ W^T + b, x:[batch,in], W:[out,in], b:[out] -> y:[batch,out]
;; dx = dy @ W         (dgemm: dy[B,out] @ W[out,in] -> dx[B,in])
;; dW = dy^T @ x       (dgemm-tn: dy[B,out]^T @ x[B,in] -> dW[out,in])
;; db = sum(dy, dim=0)
(register-template! 'raster.dl.nn/linear
                    {:params '[x W b batch in-f out-f] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [x W b batch in-f out-f] _result-sym adjoint-sym gensym-fn]
                       (let [dx (gensym-fn "dx")
                             dW (gensym-fn "dW")
                             db (gensym-fn "db")]
                         [(update ctx :bindings into
                                  [dx (list 'raster.dl.nn/linear-dx adjoint-sym W batch in-f out-f)
                                   dW (list 'raster.dl.nn/linear-dW adjoint-sym x batch in-f out-f)
                                   db (list 'raster.dl.nn/linear-db adjoint-sym batch out-f)])
                          [dx dW db nil nil nil]]))})

;; silu: y = x * sigmoid(x)
;; dx[i] = dy[i] * sigma(x[i]) * (1 + x[i]*(1-sigma(x[i])))
(register-template! 'raster.dl.nn/silu
                    {:params '[x n] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [x n] _result-sym adjoint-sym gensym-fn]
                       (let [dx (gensym-fn "dx")]
                         [(update ctx :bindings into
                                  [dx (list 'raster.dl.nn/silu-backward adjoint-sym x n)])
                          [dx nil]]))})

;; embed-timestep: sinusoidal embedding
;; d_t via sinusoidal-embed-backward
;; embed-timestep: (All [T]) [ref :- (Array T), t :- Double, dim :- Long]
;; The ref param is for type-directed allocation — no gradient.
(register-template! 'raster.dl.gsdm/embed-timestep
                    {:params '[ref t dim] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [ref t dim] _result-sym adjoint-sym gensym-fn]
                       (let [dt (gensym-fn "dt")]
                         [(update ctx :bindings into
                                  [dt (list 'raster.dl.gsdm/sinusoidal-embed-backward
                                            adjoint-sym t dim)])
                          [nil dt nil]]))})

;; ================================================================
;; raster.nn MLP primitives
;; ================================================================

;; dense: y = W*x + b
;; dW = dy outer x^T, dx = W^T*dy, db = clone(dy)
;; Emits explicit allocations so buffer-fuse / mem-merge can reuse them,
;; then calls ^:no-inline into-variants that take the pre-allocated buffer.
;; The into-variants take only array params (no primitives) to avoid
;; typed interface generation which would cause .invk → reflection at eval.
(register-template! 'raster.nn/dense
                    {:params '[W x b] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [W x b] _result-sym adjoint-sym gensym-fn]
                       (let [;; dimension bindings — reference forward params directly for hoistability
                             dW-rows (gensym-fn "dW_rows")
                             dW-cols (gensym-fn "dW_cols")
                             dW-buf  (gensym-fn "dW_buf")
                             dW      (gensym-fn "dW")
                             dx-buf  (gensym-fn "dx_buf")
                             dx      (gensym-fn "dx")
                             db-buf  (gensym-fn "db_buf")
                             db      (gensym-fn "db")]
                         [(update ctx :bindings into
                                  [;; compute dimensions from forward params (b, x) for hoistability
                                   dW-rows (list 'clojure.core/alength b)
                                   dW-cols (list 'clojure.core/alength x)
           ;; alloc dW[rows*cols] — zeros-like dispatches on W's type (double[]/float[])
                                   dW-buf  (list 'raster.arrays/zeros-like W (list 'clojure.core/* dW-rows dW-cols))
                                   dW      (list 'raster.nn/dense-backward-dW-into!
                                                 adjoint-sym x dW-buf)
           ;; alloc dx[cols] — same type as x
                                   dx-buf  (list 'raster.arrays/zeros-like x dW-cols)
                                   dx      (list 'raster.nn/dense-backward-dx-into!
                                                 adjoint-sym W dx-buf)
           ;; db: copy dy into pre-allocated buffer — same type as b
                                   db-buf  (list 'raster.arrays/zeros-like b dW-rows)
                                   db      (list 'raster.nn/dense-backward-db-into!
                                                 adjoint-sym db-buf)])
                          [dW dx db]]))})

;; relu: y = max(0, x)
;; dx = dy * (x > 0 ? 1 : 0)
(register-template! 'raster.nn/relu
                    {:params '[x] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [x] _result-sym adjoint-sym gensym-fn]
                       (let [dx (gensym-fn "dx")]
                         [(update ctx :bindings into
                                  [dx (list 'raster.nn/relu-backward adjoint-sym x)])
                          [dx]]))})

;; softmax: s = softmax(x)
;; dx = s * (dy - dot(s, dy))
(register-template! 'raster.nn/softmax
                    {:params '[x] :result 's :adjoint 'dy
                     :grads-fn
                     (fn [ctx [x] result-sym adjoint-sym gensym-fn]
                       (let [dx (gensym-fn "dx")]
                         [(update ctx :bindings into
                                  [dx (list 'raster.nn/softmax-backward adjoint-sym result-sym)])
                          [dx]]))})

;; cross-entropy: L = -sum(t * log(p))
;; dp = -t/p * dy (dt not needed for training)
(register-template! 'raster.nn/cross-entropy
                    {:params '[p t] :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [p t] _result-sym adjoint-sym gensym-fn]
                       (let [dp (gensym-fn "dp")]
                         [(update ctx :bindings into
                                  [dp (list 'raster.nn/cross-entropy-backward-dp adjoint-sym p t)])
                          [dp nil]]))})

;; ================================================================
;; raster.dl.nn conv2d/maxpool2d templates
;; ================================================================

;; conv2d: y = conv2d(x, W, b, batch, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw)
;; Forward uses im2col → matmul → rearrange+bias
;; Backward: dW via GEMM-NT, dx via GEMM-TN + col2im, db via spatial sum
(register-template! 'raster.dl.nn/conv2d
                    {:params '[x W b batch c-in h w c-out kh kw stride-h stride-w pad-h pad-w]
                     :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [x W b batch c-in h w c-out kh kw stride-h stride-w pad-h pad-w]
                          _result-sym adjoint-sym gensym-fn]
                       (let [;; dimension bindings
                             h-out  (gensym-fn "h_out")
                             w-out  (gensym-fn "w_out")
                             ckk    (gensym-fn "ckk")
                             bhw    (gensym-fn "bhw")
           ;; buffers
                             dy-cols-buf (gensym-fn "dy_cols_buf")
                             dy-cols     (gensym-fn "dy_cols")
                             cols-buf    (gensym-fn "cols_buf")
                             cols        (gensym-fn "cols")
                             dW-buf      (gensym-fn "dW_buf")
                             dW          (gensym-fn "dW")
                             d-cols-buf  (gensym-fn "d_cols_buf")
                             d-cols      (gensym-fn "d_cols")
                             dx-buf      (with-meta (gensym-fn "dx_buf") {:hoistable true})
                             dx          (gensym-fn "dx")
                             db-buf      (with-meta (gensym-fn "db_buf") {:hoistable true})
                             db          (gensym-fn "db")]
                         [(update ctx :bindings into
                                  [;; compute output dims
                                   h-out (list 'clojure.core/+ 1
                                               (list 'clojure.core/quot
                                                     (list 'clojure.core/+ (list 'clojure.core/+ h (list 'clojure.core/* 2 pad-h))
                                                           (list 'clojure.core/- kh))
                                                     stride-h))
                                   w-out (list 'clojure.core/+ 1
                                               (list 'clojure.core/quot
                                                     (list 'clojure.core/+ (list 'clojure.core/+ w (list 'clojure.core/* 2 pad-w))
                                                           (list 'clojure.core/- kw))
                                                     stride-w))
                                   ckk   (list 'long (list 'clojure.core/* c-in (list 'clojure.core/* kh kw)))
                                   bhw   (list 'long (list 'clojure.core/* batch (list 'clojure.core/* h-out w-out)))
           ;; rearrange dy -> dy_cols[c_out, bhw]
                                   (with-meta dy-cols-buf {:hoistable true})
                                   (list 'raster.arrays/zeros-like W (list 'clojure.core/* c-out bhw))
                                   dy-cols     (list 'raster.dl.nn/conv2d-rearrange-dy!
                                                     adjoint-sym dy-cols-buf batch c-out h-out w-out)
           ;; im2col for forward recomputation
                                   (with-meta cols-buf {:hoistable true})
                                   (list 'raster.arrays/zeros-like x (list 'clojure.core/* ckk bhw))
                                   cols        (list 'raster.dl.nn/im2col-2d!
                                                     x cols-buf batch c-in h w kh kw stride-h stride-w pad-h pad-w)
           ;; dW = dy_cols @ cols^T
                                   (with-meta dW-buf {:hoistable true})
                                   (list 'raster.arrays/zeros-like W (list 'clojure.core/* c-out ckk))
                                   dW          (list 'raster.dl.nn/conv2d-backward-dW-into!
                                                     dy-cols cols dW-buf c-out bhw ckk)
           ;; d_cols = W^T @ dy_cols
                                   (with-meta d-cols-buf {:hoistable true})
                                   (list 'raster.arrays/zeros-like x (list 'clojure.core/* ckk bhw))
                                   d-cols      (list 'raster.dl.nn/conv2d-backward-dcols-into!
                                                     W dy-cols d-cols-buf ckk c-out bhw)
           ;; dx = col2im(d_cols) into pre-allocated buffer
                                   dx-buf      (list 'raster.arrays/zeros-like x (list 'clojure.core/* (list 'clojure.core/* batch c-in) (list 'clojure.core/* h w)))
                                   dx          (list 'raster.dl.nn/col2im-2d!
                                                     d-cols dx-buf
                                                     batch c-in h w kh kw stride-h stride-w pad-h pad-w)
           ;; db = sum over spatial dims
                                   db-buf      (list 'raster.arrays/zeros-like b c-out)
                                   db          (list 'raster.dl.nn/conv2d-backward-db-into!
                                                     adjoint-sym db-buf batch c-out h-out w-out)])
        ;; grads: [dx dW db nil*11] — nil for all Long dimension params
        ;; params: [x W b batch c-in h w c-out kh kw stride-h stride-w pad-h pad-w]
                          [dx dW db nil nil nil nil nil nil nil nil nil nil nil]]))})

;; maxpool2d: y = maxpool2d(x, batch, channels, h, w, kh, kw)
;; Forward saves argmax, backward scatters gradient to argmax positions
(register-template! 'raster.dl.nn/maxpool2d
                    {:params '[x batch channels h w kh kw]
                     :result nil :adjoint 'dy
                     :grads-fn
                     (fn [ctx [x batch channels h w kh kw] _result-sym adjoint-sym gensym-fn]
                       (let [;; dimension expressions — all Long call-site params, will be constant-folded
                             h-out-expr  (list 'clojure.core/quot h kh)
                             w-out-expr  (list 'clojure.core/quot w kw)
                             n-out-expr  (list 'clojure.core/* (list 'clojure.core/* batch channels)
                                               (list 'clojure.core/* h-out-expr w-out-expr))
                             n-in-expr   (list 'clojure.core/* (list 'clojure.core/* batch channels)
                                               (list 'clojure.core/* h w))
           ;; bound syms
                             n-out    (gensym-fn "n_out")
                             n-in     (gensym-fn "n_in")
                             out-buf  (with-meta (gensym-fn "mp_out_buf") {:hoistable true})
                             argmax   (with-meta (gensym-fn "argmax") {:hoistable true})
                             fwd      (gensym-fn "mp_fwd")
                             dx-buf   (with-meta (gensym-fn "dx_buf") {:hoistable true})
                             dx       (gensym-fn "dx")]
                         [(update ctx :bindings into
                                  [n-out  n-out-expr
                                   n-in   n-in-expr
           ;; forward with argmax tracking — inline size exprs for hoistability
                                   out-buf (list 'raster.arrays/zeros-like x n-out-expr)
                                   argmax  (list 'long-array n-out-expr)
                                   fwd     (list 'raster.dl.nn/maxpool2d! x out-buf argmax
                                                 batch channels h w kh kw)
           ;; backward: scatter dy to argmax positions
                                   dx-buf  (list 'raster.arrays/zeros-like x n-in-expr)
                                   dx      (list 'raster.dl.nn/maxpool2d-bwd!
                                                 adjoint-sym argmax dx-buf n-out n-in)])
        ;; grads: [dx nil nil nil nil nil nil]
                          [dx nil nil nil nil nil nil]]))})

;; ================================================================
;; Linear algebra templates
;; ================================================================

;; solve: Ax = b, backward via A^{-T} and outer product
(merge-into-template! 'raster.linalg.core/solve
                      {:params '[A b n] :result 'x :adjoint 'dy
                       :grads-fn (fn [ctx [A b n] result-sym adjoint-sym gensym-fn]
                                   (let [grads-arr (gensym-fn "solve_grads")
                                         dA (gensym-fn "dA")
                                         db (gensym-fn "db")]
                                     [(update ctx :bindings into
                                              [grads-arr (list 'raster.linalg.core/solve-backward
                                                               adjoint-sym A result-sym n)
                                               dA (list 'clojure.core/aget grads-arr 0)
                                               db (list 'clojure.core/aget grads-arr 1)])
                                      [dA db nil]]))})

;; array-det: det(A), backward is det(A) * (A^{-1})^T * dy
(merge-into-template! 'raster.linalg.core/array-det
                      {:params '[A n] :result 'det-val :adjoint 'dy
                       :grads-fn (fn [ctx [A n] result-sym adjoint-sym gensym-fn]
                                   (let [dA (gensym-fn "dA")]
                                     [(update ctx :bindings into
                                              [dA (list 'raster.linalg.core/array-det-backward
                                                        adjoint-sym result-sym A n)])
                                      [dA nil]]))})

;; ================================================================
;; §13 A1 — structure tags + DERIVED forward rules (JVP), one rule source
;;
;; The :structure facet is a presentation of the op's Jacobian
;; (framework §2: ONE presentation, two contractions). The reverse
;; contraction is the existing :grads/:grads-fn; the forward contraction
;; (:jvp-fn) is DERIVED from the structure class — never a second
;; hand-written table (the Julia two-forward-tables mistake):
;;
;;   :elementwise  diagonal J = Jᵀ      → frule(v) = grads-fn with dy:=v
;;                 (also every SCALAR op: output is 1-dim, so grads[i] is
;;                  Jᵢ·dy, linear in dy ⇒ frule = Σᵢ grads[i](dy:=Δxᵢ) —
;;                  the @scalar_rule mechanics, product rule falls out)
;;   :symmetric    J = Jᵀ (softmax)     → same diagonal reuse
;;   :linear       jointly-linear in :in → frule = the op itself on the
;;                 tangents (inactive :in slots get typed zeros — NOT the
;;                 primal: f(Δa, b) ≠ df for additive ops)
;;   :bilinear     frule = op(Δx,W,0ₑ) ⊕ op(x,ΔW,0ₑ), affine :linear-extra
;;                 slots (biases) zeroed in every term except the first,
;;                 which routes the extras' own tangents
;;   :scalar-loss  frule = ⟨grads-fn(dy:=1̄), v⟩ via a generic dot kernel
;;
;; Emission hazard (RoR recon): every emitted call is FULLY QUALIFIED and
;; tangent syms carry their primal's :raster.type/tag (tangent protocol) so
;; parametric dispatch resolves and devirtualization can work downstream.
;;
;; ⊕ for tangent contributions is raster.ad.reverse/grad-acc — the SAME
;; polymorphic (scalar/array, nil-safe) accumulator the reverse pass uses.
;; The dot kernel is raster.par/dot-product ((All [T]) → Double), reused —
;; no new kernel needed.
;; ================================================================

(def ^:private op-structures
  "Jacobian structure classes per §13-RESOLVED inventory (indices verified
  against each op's registered :params / deftm param list).
  NOTE inventory corrections:
   - class (b) linear counts 16 ops here (the doc's `14` undercounts the
     pack/unpack and slice/scatter pairs it lists as single entries);
   - raster.nn/dense arg order is [W x b] (weights first, unlike linear);
   - raster.nn/softmax-cross-entropy returns a TUPLE [loss s]; its derived
     JVP is the tangent of the LOSS component (mirrors the reverse rule,
     whose scalar adjoint dy also addresses only the loss slot);
   - forward-noise is JOINTLY linear in {x0, noise} (alpha-t rule-frozen);
   - maxpool2d, einsum, dgemm! deliberately untagged (A3/kernel gaps)."
  {;; --- (a) elementwise: diagonal Jacobian, frule = backward(tangent) ---
   'raster.nn/relu                       {:class :elementwise :in #{0}} ;; [x]
   'raster.dl.nn/gelu                    {:class :elementwise :in #{0}} ;; [x n]
   'raster.dl.nn/silu                    {:class :elementwise :in #{0}} ;; [x n]
   'raster.dl.nn/sigmoid                 {:class :elementwise :in #{0}} ;; [x n]
   'raster.dl.nn/tanh-act                {:class :elementwise :in #{0}} ;; [x n]
   'raster.dl.nn/leaky-relu              {:class :elementwise :in #{0}} ;; [x n alpha]
   'raster.dl.nn/dropout                 {:class :elementwise :in #{0}} ;; [x mask n], mask frozen
   'raster.dl.array-ops/scale-clamp-exp  {:class :elementwise :in #{0}} ;; [x scale clamp-bound n], scale rule-frozen
   ;; --- (a′) symmetric: J = diag(s) − ssᵀ = Jᵀ ---
   'raster.nn/softmax                    {:class :symmetric :in #{0}}   ;; [x]
   'raster.dl.nn/softmax-1d              {:class :symmetric :in #{0}}   ;; [x n]
   ;; --- (b) linear: frule = the op on the tangent(s) ---
   'raster.dl.nn/residual-add            {:class :linear :in #{0 1}}    ;; [a b n]
   'raster.dl.array-ops/array-add        {:class :linear :in #{0 1}}    ;; [a b n]
   'raster.dl.array-ops/broadcast-add    {:class :linear :in #{0 1}}    ;; [h t batch dim]
   'raster.dl.nn/transpose-2d            {:class :linear :in #{0}}      ;; [a rows cols]
   'raster.dl.array-ops/pack-heads       {:class :linear :in #{0}}      ;; [x seq-len n-heads head-dim]
   'raster.dl.array-ops/unpack-heads     {:class :linear :in #{0}}      ;; [x seq-len n-heads head-dim]
   'raster.dl.array-ops/slice-strided-2d {:class :linear :in #{0}}      ;; [src rows row-stride col-offset n-cols]
   'raster.dl.array-ops/scatter-strided-2d {:class :linear :in #{0}}    ;; [packed rows row-stride col-offset n-cols]
   'raster.dl.array-ops/gather           {:class :linear :in #{0}}      ;; [src indices n-src n-pairs stride]
   'raster.dl.array-ops/scatter-add      {:class :linear :in #{0}}      ;; [vals indices n-dst n-pairs stride]
   'raster.dl.array-ops/reduce-axis      {:class :linear :in #{0}}      ;; [src n-rows n-cols]
   'raster.dl.nn/embedding               {:class :linear :in #{0}}      ;; [table indices n vocab dim]
   'raster.dl.attention/rope             {:class :linear :in #{0}}      ;; [x seq-len heads head-dim theta] — orthogonal rotation
   'raster.dl.einsum/rearrange           {:class :linear :in #{0}}      ;; [tensor pattern axis-lengths?] — permutation
   'raster.dl.diffusion/forward-noise    {:class :linear :in #{0 1}}    ;; [x0 noise alpha-t n], alpha-t rule-frozen
   'raster.quant.train/qlinear-q8        {:class :linear :in #{0}}      ;; [x codes scales m in out] — codes frozen
   'clojure.core/aget                    {:class :linear :in #{0}}      ;; [arr i] — one-hot read
   ;; --- (c) bilinear: frule = op(Δx,·) ⊕ op(·,ΔW) ---
   'raster.dl.nn/matmul          {:class :bilinear :args [0 1]}                   ;; [A B m k n]
   'raster.dl.nn/linear          {:class :bilinear :args [0 1] :linear-extra #{2}} ;; [x W b batch in-f out-f]
   'raster.dl.nn/linear-nb       {:class :bilinear :args [0 1]}                   ;; [x W batch in-f out-f]
   'raster.nn/dense              {:class :bilinear :args [0 1] :linear-extra #{2}} ;; [W x b]
   'raster.dl.nn/hadamard        {:class :bilinear :args [0 1]}                   ;; [a b n]
   'raster.dl.array-ops/dot-rows {:class :bilinear :args [0 1] :linear-extra #{2}} ;; [h W bias n-rows dim]
   'raster.dl.array-ops/indexed-dot     {:class :bilinear :args [0 1]}            ;; [A B idx-a idx-b ...]
   'raster.dl.array-ops/scatter-mul-add {:class :bilinear :args [0 1]}            ;; [coeffs src ...]
   'raster.dl.nn/conv1d          {:class :bilinear :args [0 1] :linear-extra #{2}} ;; [x W b batch ...]
   'raster.dl.nn/conv2d          {:class :bilinear :args [0 1] :linear-extra #{2}} ;; [x W b batch ...]
   ;; --- (d) scalar losses: frule = ⟨grads-fn(1̄), v⟩ ---
   'raster.dl.loss/mse-loss             {:class :scalar-loss} ;; [pred target n], target rule-frozen
   'raster.dl.loss/l1-loss              {:class :scalar-loss} ;; [pred target n]
   'raster.dl.loss/huber-loss           {:class :scalar-loss} ;; [pred target n delta]
   'raster.dl.array-ops/masked-mse-loss {:class :scalar-loss} ;; [pred target states n]
   'raster.dl.loss/cross-entropy-loss   {:class :scalar-loss} ;; [logits target batch classes]
   'raster.nn/cross-entropy             {:class :scalar-loss} ;; [p t], t rule-frozen
   'raster.nn/softmax-cross-entropy     {:class :scalar-loss} ;; [logits t] → [loss s] tuple (see note)
   ;; --- scalar primitives: frule = Σᵢ grads(dy:=Δxᵢ) (mechanical, §2) ---
   '+          {:class :elementwise :in #{0 1}}
   '-          {:class :elementwise :in #{0 1}}
   '*          {:class :elementwise :in #{0 1}}
   '/          {:class :elementwise :in #{0 1}}
   'raster.ad.reverse/grad-acc {:class :elementwise :in #{0 1}}
   'Math/sin   {:class :elementwise :in #{0}}
   'Math/cos   {:class :elementwise :in #{0}}
   'Math/exp   {:class :elementwise :in #{0}}
   'Math/log   {:class :elementwise :in #{0}}
   'Math/sqrt  {:class :elementwise :in #{0}}
   'Math/pow   {:class :elementwise :in #{0 1}}
   'Math/abs   {:class :elementwise :in #{0}}
   'Math/tan   {:class :elementwise :in #{0}}
   'Math/asin  {:class :elementwise :in #{0}}
   'Math/acos  {:class :elementwise :in #{0}}
   'Math/atan  {:class :elementwise :in #{0}}
   'Math/atan2 {:class :elementwise :in #{0 1}}
   'Math/min   {:class :elementwise :in #{0 1}}
   'Math/max   {:class :elementwise :in #{0 1}}
   'Math/fma   {:class :elementwise :in #{0 1 2}}
   'Math/sinh  {:class :elementwise :in #{0}}
   'Math/cosh  {:class :elementwise :in #{0}}
   'Math/tanh  {:class :elementwise :in #{0}}
   'Math/log1p {:class :elementwise :in #{0}}
   'Math/expm1 {:class :elementwise :in #{0}}
   'Math/cbrt  {:class :elementwise :in #{0}}
   'Math/log10 {:class :elementwise :in #{0}}
   'Math/hypot {:class :elementwise :in #{0 1}}
   'double     {:class :elementwise :in #{0}}
   'float      {:class :elementwise :in #{0}}
   'long       {:class :elementwise :in #{0}}   ;; grads 0.0 → zero tangent (cast kills derivative)
   'int        {:class :elementwise :in #{0}}
   'raster.sci.special/lgamma   {:class :elementwise :in #{0}}
   'raster.sci.special/digamma  {:class :elementwise :in #{0}}
   'raster.sci.special/erf      {:class :elementwise :in #{0}}
   'raster.sci.special/erfinv   {:class :elementwise :in #{0}}
   'raster.sci.special/expit    {:class :elementwise :in #{0}}
   'raster.sci.special/logit    {:class :elementwise :in #{0}}
   'raster.sci.special/log1pexp {:class :elementwise :in #{0}}
   'raster.sci.special/besselj0 {:class :elementwise :in #{0}}
   'raster.sci.special/besselj1 {:class :elementwise :in #{0}}
   'raster.sci.special/betainc  {:class :elementwise :in #{2}}})

;; Merge structure facets into the registry. merge-into-template! is
;; order-independent: ops whose :grads-fn is registered by a later-loaded ns
;; (raster.dl.*, raster.nn, …) get a partial {:structure …} entry here that
;; the later merge fills in. Ops registered by register-template! ABOVE in
;; this file are merged after their registration (this block is at the end).
(doseq [[op st] op-structures]
  (merge-into-template! op {:structure st}))

(defn- jvp-zero-like
  "Typed zero tangent shaped/dtyped like the PRIMAL arg (tangent protocol):
  used for :in / :linear-extra slots whose tangent is absent in a term where
  substituting the primal would be WRONG (linear ⇒ f(Δa,b) ≠ df)."
  [primal-arg]
  (list 'raster.arrays/zeros-like primal-arg
        (list 'raster.arrays/alength primal-arg)))

(defn- sum-tangent-contribs
  "Bind the ⊕-sum of tangent contribution exprs; returns [ctx' sym].
  ⊕ is raster.ad.reverse/grad-acc — polymorphic scalar/array, the same
  accumulator the reverse pass uses."
  [ctx contribs gensym-fn]
  (let [t-sym (gensym-fn "jt")
        expr (reduce (fn [a b] (list 'raster.ad.reverse/grad-acc a b)) contribs)]
    [(update ctx :bindings into [t-sym expr]) t-sym]))

(defn- derive-jvp-fn
  "Synthesize the :jvp-fn facet from the op's :structure tag + its EXISTING
  gradient representation (one rule source). Returns
    (fn [ctx args tangent-args result-sym gensym-fn] -> [ctx' tangent-sym])
  emitting FLAT bindings (BindCtx style, like grads-fn). tangent-args is
  aligned with args; nil = no tangent flowing into that slot."
  [canonical-op template {:keys [class in args linear-extra]}]
  (case class
    ;; Diagonal / symmetric / scalar Jacobian: reuse grads-fn with dy:=Δxᵢ.
    (:elementwise :symmetric)
    (let [in-idxs (vec (sort (or in #{0})))]
      (when-not (:grads-fn template)
        (throw (ex-info (str "derive-jvp-fn: " canonical-op " tagged " class
                             " but has no :grads-fn to reuse")
                        {:op canonical-op :class class})))
      (fn [ctx call-args tangent-args result-sym gensym-fn]
        (let [[ctx contribs]
              (reduce
               (fn [[ctx contribs] i]
                 (if-let [tv (and (< i (count call-args)) (nth tangent-args i nil))]
                   (let [[ctx' gsyms] (instantiate-template-ctx
                                       template (vec call-args) result-sym tv ctx)
                         g (nth gsyms i nil)]
                     (if (some? g) [ctx' (conj contribs g)] [ctx' contribs]))
                   [ctx contribs]))
               [ctx []] in-idxs)]
          (when (empty? contribs)
            (throw (ex-info (str "derive-jvp-fn: no active tangent reached " canonical-op)
                            {:op canonical-op :tangent-args tangent-args})))
          (sum-tangent-contribs ctx contribs gensym-fn))))

    ;; Jointly-linear: run the op itself with ALL :in slots replaced by their
    ;; tangents (typed zero where a slot's tangent is absent — linearity makes
    ;; the simultaneous substitution exactly the pushforward; a one-at-a-time
    ;; sum with primal in the other slot would be wrong for additive ops).
    :linear
    (let [in-set (set in)]
      (fn [ctx call-args tangent-args _result-sym gensym-fn]
        (when-not (some #(nth tangent-args % nil) in-set)
          (throw (ex-info (str "derive-jvp-fn: no active tangent reached " canonical-op)
                          {:op canonical-op})))
        (let [args' (map-indexed
                     (fn [i a]
                       (if (contains? in-set i)
                         (or (nth tangent-args i nil) (jvp-zero-like a))
                         a))
                     call-args)
              t-sym (gensym-fn "jt")]
          [(update ctx :bindings into [t-sym (cons canonical-op args')]) t-sym])))

    ;; Bilinear(i,j) + affine extras: op(Δxᵢ, xⱼ, extras…) ⊕ op(xᵢ, Δxⱼ, 0ₑ).
    ;; Extras (biases) are AFFINE offsets: they must be ZEROED in every term,
    ;; except the first emitted term routes the extras' own tangents (linear).
    :bilinear
    (let [[bi bj] args
          extras (set (or linear-extra #{}))]
      (fn [ctx call-args tangent-args _result-sym gensym-fn]
        (let [ti (nth tangent-args bi nil)
              tj (nth tangent-args bj nil)
              extra-active? (some #(nth tangent-args % nil) extras)
              subst (fn [swap-idx zero-i? route-extras?]
                      (map-indexed
                       (fn [k a]
                         (cond
                           (= k swap-idx) (nth tangent-args swap-idx)
                           (contains? extras k)
                           (if route-extras?
                             (or (nth tangent-args k nil) (jvp-zero-like a))
                             (jvp-zero-like a))
                           (and zero-i? (= k bi)) (jvp-zero-like a)
                           :else a))
                       call-args))
              terms (cond-> []
                      ti (conj (cons canonical-op (subst bi false true)))
                      tj (conj (cons canonical-op (subst bj false (not ti))))
                      ;; extras-only tangent: op(0ᵢ, xⱼ, Δextras) = Δextras' image
                      (and (not ti) (not tj) extra-active?)
                      (conj (cons canonical-op
                                  (map-indexed
                                   (fn [k a]
                                     (cond
                                       (= k bi) (jvp-zero-like a)
                                       (contains? extras k)
                                       (or (nth tangent-args k nil) (jvp-zero-like a))
                                       :else a))
                                   call-args))))]
          (when (empty? terms)
            (throw (ex-info (str "derive-jvp-fn: no active tangent reached " canonical-op)
                            {:op canonical-op})))
          (let [[ctx term-syms]
                (reduce (fn [[ctx syms] term]
                          (let [s (gensym-fn "jbt")]
                            [(update ctx :bindings into [s term]) (conj syms s)]))
                        [ctx []] terms)]
            (sum-tangent-contribs ctx term-syms gensym-fn)))))

    ;; Scalar loss: frule = ⟨grads-fn(dy := typed 1̄), Δactive⟩ per active arg,
    ;; summed. Dot kernel: raster.par/dot-product ((All [T]) → Double), reused.
    :scalar-loss
    (do
      (when-not (:grads-fn template)
        (throw (ex-info (str "derive-jvp-fn: " canonical-op
                             " tagged :scalar-loss but has no :grads-fn to reuse")
                        {:op canonical-op})))
      (fn [ctx call-args tangent-args result-sym gensym-fn]
        (let [seed-sym (gensym-fn "jseed")
              ;; typed unit adjoint from the RESULT's tag (tangent protocol);
              ;; untagged falls back to 1.0 (loss is scalar by contract).
              seed (tangent/seed-expr (:raster.type/tag (meta result-sym)))
              ctx (update ctx :bindings into [seed-sym seed])
              [ctx gsyms] (instantiate-template-ctx
                           template (vec call-args) result-sym seed-sym ctx)
              contribs (vec (keep-indexed
                             (fn [i g]
                               (when-let [tv (and (some? g) (nth tangent-args i nil))]
                                 (list 'raster.par/dot-product g tv)))
                             gsyms))]
          (if (empty? contribs)
            ;; every tangent-bearing arg is rule-frozen (e.g. mse target):
            ;; the rule's pushforward is exactly 0 — mirror the reverse rule.
            (let [t-sym (gensym-fn "jt")]
              [(update ctx :bindings into [t-sym 0.0]) t-sym])
            (sum-tangent-contribs ctx contribs gensym-fn)))))

    (throw (ex-info (str "derive-jvp-fn: unknown :structure class " class
                         " on " canonical-op)
                    {:op canonical-op :class class}))))

(defn op-jvp-fn
  "The op's forward-mode (JVP) rule: an explicit :jvp-fn facet, or one derived
  (and cached) from its :structure tag. Returns nil when the op has neither —
  the transform (raster.ad.jvp) fails loud naming the op in that case.

  Alias chasing: qualified alias entries (raster.numeric/+, clojure.core/sin
  variants, …) were value-copied at registration BEFORE structure tagging, so
  a structureless alias falls back to its canonical base key."
  [op]
  (when-let [[template canonical] (resolve-template op)]
    (or (:jvp-fn template)
        (when-let [st (:structure template)]
          (let [f (derive-jvp-fn canonical template st)]
            (swap! template-registry assoc-in [canonical :jvp-fn] f)
            f))
        ;; alias → base fallbacks
        (when-let [base (get qualified->base-op canonical)]
          (op-jvp-fn base))
        (when (and (qualified-symbol? canonical)
                   (= "raster.math" (namespace canonical)))
          (let [interop (symbol "Math" (name canonical))]
            (when (get-template interop) (op-jvp-fn interop))))
        (when (and (qualified-symbol? canonical)
                   (= "clojure.core" (namespace canonical))
                   (not= canonical (symbol (name canonical))))
          (let [bare (symbol (name canonical))]
            (when (get-template bare) (op-jvp-fn bare)))))))
