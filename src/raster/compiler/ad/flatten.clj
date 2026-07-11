(ns raster.compiler.ad.flatten
  "Closure-flattening pass for training function compilation.

   Respects *flatten-dtype* for type-appropriate AD seed values.

  The AD transform produces:
    (let* [... fwd-bindings ...
           result__N body-sym]
      [result__N (fn* [dy__rad] (let* [... rev-bindings ...]
                                       [d_param1 d_param2 ...]))])

  For peephole/fusion to work, we need everything in one flat let*:
    (let* [... fwd-bindings ...
           result__N body-sym
           dy__rad 1.0
           ... rev-bindings ...
           d_param1 expr1
           d_param2 expr2
           ... SGD updates ...]
      result__N)

  This pass:
  1. Detects the [primal, (fn* [dy] ...)] return pattern
  2. Extracts and inlines the reverse-pass bindings with dy=1.0
  3. Names the gradient outputs (one per active param)
  4. Appends SGD weight update calls: (nc/axpy! (- lr) d_Wi Wi)"
  (:require [raster.ad.tangent :as tangent]
            [raster.compiler.ir.dialects :as dialects]
            [raster.compiler.ir.form :as form]))

(def ^:dynamic *flatten-dtype*
  "Pipeline dtype for type-appropriate AD seed values.
   :float → (float 1.0), :double or nil → 1.0"
  nil)

;; ================================================================
;; Structure detection
;; ================================================================

(def ^:private let-form?
  "Alias for form/binding-form? — detects let/let* forms."
  form/binding-form?)

(defn- canonical-body-expr
  "Return a single body expression for a let-like form body.
  Multiple body expressions are wrapped in an explicit do."
  [body-exprs]
  (cond
    (= 1 (count body-exprs)) (first body-exprs)
    (empty? body-exprs) nil
    :else (cons 'do body-exprs)))

(defn- make-flat-let*
  "Build the canonical flat let* form used by flattening passes."
  [bindings body-expr]
  (list 'let* (vec bindings) body-expr))

(declare decompose-flat-let*)

(defn canonicalize-ad-form
  "Normalize AD output to the compiler's canonical let* shape.

  Accepts either:
    - (let/let* [...] [primal pullback])
    - [primal pullback]

  Returns a canonical let* form or nil when the input is not recognized
  as AD output. Every caller passes what it believes is transform-body
  output and treats nil as \"leave the call unexpanded\" — a silent
  degradation — so a nil result warns with the ADTransformed dialect
  validator's diagnosis."
  [form]
  (let [canonical
        (cond
          (let-form? form)
          (when-let [{:keys [bindings body]} (decompose-flat-let* form)]
            (when (and (vector? body)
                       (= 2 (count body))
                       (seq? (second body))
                       (= 'fn* (first (second body))))
              (with-meta (make-flat-let* bindings body) (meta form))))

          (and (vector? form)
               (= 2 (count form))
               (seq? (second form))
               (= 'fn* (first (second form))))
          (with-meta (make-flat-let* [] form) (meta form))

          :else nil)]
    (when (and (nil? canonical) (some? form))
      (binding [*out* *err*]
        (println (str "WARNING: AD output failed canonicalization — the value+grad "
                      "call stays unexpanded (silent fallback). Dialect check: "
                      (pr-str (dialects/validate-ad-transformed form))))))
    canonical))

(defn- decompose-flat-let*
  "Decompose a let/let* form and normalize it to the canonical flat-let view.
  Returns {:bindings [...], :body expr} or nil."
  [form]
  (when (let-form? form)
    (let [[_ bindings-vec & body-exprs] form]
      {:bindings (vec bindings-vec)
       :body (canonical-body-expr body-exprs)})))

(defn- update-flat-let*
  "Apply a binding/body update to a let-like form and rebuild it canonically
  as let*."
  [form update-fn]
  (when-let [{:keys [bindings body]} (decompose-flat-let* form)]
    (let [{new-bindings :bindings
           new-body :body} (update-fn {:bindings bindings :body body})]
      (make-flat-let* new-bindings new-body))))

(defn- extract-ad-structure
  "Parse the AD output into its components.

  Input: (let* [... fwd ... result-sym body-sym]
           [result-sym (fn* [dy-sym] pullback-body)])

  Returns:
    {:fwd-bindings [sym1 expr1 sym2 expr2 ...]
     :result-sym   symbol
     :body-sym     symbol (aliases result in fwd)
     :dy-sym       symbol
     :pullback-body the pullback body form
     :param-adj-syms [d_p1 d_p2 ...] from the pullback return vector}
  or nil if form doesn't match the AD pattern."
  [form]
  (when-let [{:keys [bindings body]} (some-> form canonicalize-ad-form decompose-flat-let*)]
    (let [bindings-vec bindings]
      ;; Body should be [result-sym pullback-fn]
      (when (vector? body)
        (let [[result-sym pullback-fn] body]
          (when (and (symbol? result-sym)
                     (seq? pullback-fn)
                     (= 'fn* (first pullback-fn)))
            ;; pullback-fn: (fn* [dy-sym] pullback-body)
            (let [[_ params-vec pullback-body] pullback-fn
                  dy-sym (first params-vec)]
              ;; pullback-body: (let* [...rev-bindings...] [d_p1 d_p2 ...])
              (when-let [{:keys [bindings body]} (decompose-flat-let* pullback-body)]
                (let [rev-bindings-vec bindings
                      pb-body body]
                  ;; pb-body should be [d_p1 d_p2 ...]
                  (when (vector? pb-body)
                    ;; The last binding in fwd is [result-sym body-sym]
                    (let [fwd-pairs (partition 2 bindings-vec)
                          [last-sym last-expr] (last fwd-pairs)
                          ;; Verify the result-sym is bound in fwd
                          body-sym (when (= last-sym result-sym) last-expr)]
                      {:fwd-bindings bindings-vec
                       :result-sym result-sym
                       :body-sym body-sym
                       :dy-sym dy-sym
                       :rev-bindings rev-bindings-vec
                       :param-adj-syms (vec pb-body)})))))))))))

;; ================================================================
;; Flattening
;; ================================================================

(defn flatten-ad-form
  "Flatten an AD-transformed form into a single let* block.

  Takes the AD output (let* [...fwd...] [primal, (fn* [dy] ...)])
  and produces (let* [...fwd..., dy=1.0, ...bwd...] primal).

  Returns {:form flattened-form
           :result-sym symbol bound to the primal value
           :param-adj-syms [d_p1 d_p2 ...] gradient symbols}
  or nil if form doesn't match AD pattern."
  [form]
  (when-let [{:keys [fwd-bindings result-sym body-sym dy-sym
                     rev-bindings param-adj-syms]} (extract-ad-structure form)]
    (let [;; Seed value: 1.0 typed by the RESULT's tangent space (Π): the
          ;; seed is the cotangent of the loss, so its dtype derives from
          ;; the result/body sym's :raster.type/tag when present. Falls
          ;; back to the pipeline-wide *flatten-dtype* for untagged results.
          result-tag (or (some-> result-sym meta :raster.type/tag)
                         (some-> body-sym meta :raster.type/tag))
          seed (if (= :scalar (:kind (tangent/tangent-kind result-tag)))
                 (tangent/seed-expr result-tag)
                 (case *flatten-dtype* :float (list 'float 1.0) 1.0))
          ;; Build flat bindings: fwd + [dy seed] + rev
          flat-bindings (vec (concat fwd-bindings
                                     [dy-sym seed]
                                     rev-bindings))
          flat-form (make-flat-let* flat-bindings result-sym)]
      {:form flat-form
       :result-sym result-sym
       :param-adj-syms param-adj-syms})))

(defn flatten-for-gradient
  "Flatten AD output for gradient computation (no SGD updates).

  Takes the AD output (let* [...] [primal, (fn* [dy] ...)]) and produces
  a flat let* that computes both the primal value and parameter gradients:
    (let* [...fwd... dy=1.0 ...bwd... d_p1=expr ...] [primal d_p1 ... d_pN])

  Returns {:form    flat let* returning [primal grad1 ... gradN]
           :result-sym   symbol bound to primal value
           :param-adj-syms [d_p1 d_p2 ...] gradient symbols
           :stats   {:bindings-flattened N}}
  or nil if form doesn't match AD pattern."
  [ad-form]
  (when-let [{:keys [form result-sym param-adj-syms]} (flatten-ad-form ad-form)]
    (let [{:keys [bindings]} (decompose-flat-let* form)]
      {:form (make-flat-let* bindings
                             (vec (cons result-sym param-adj-syms)))
       :result-sym result-sym
       :param-adj-syms param-adj-syms
       :stats {:bindings-flattened (count (partition 2 bindings))}})))

