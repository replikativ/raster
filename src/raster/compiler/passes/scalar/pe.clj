(ns raster.compiler.passes.scalar.pe
  "Partial evaluator for walked S-expressions.

  Performs:
  - Constant propagation: known-value bindings substituted into body
  - Dead branch elimination: (if true x y) => x
  - Let inlining: single-use bindings substituted directly
  - Algebraic simplification via raster.compiler.passes.scalar.simplify
  - Fixpoint iteration (configurable, default 10 rounds)

  Dead binding elimination is NOT done here — it belongs in the standalone
  DCE pass which sees the final form. Combining DCE with constant propagation
  in one left-to-right scan is unsound for AD-generated code.

  Operates on the same S-expression format that the walker produces
  and the bytecode compiler consumes. Does NOT modify the form structure
  beyond the transformations above — preserves metadata, qualified symbols, etc."
  (:require [raster.compiler.ir.form :as form]
            [raster.compiler.passes.scalar.normalize :as normalize]
            [raster.compiler.passes.scalar.simplify :as simp]
            [raster.core :as rcore]))

;; ================================================================
;; Value analysis
;; ================================================================

(defn- constant?
  "True if form is a compile-time constant (no free variables).
  Numeric literals, strings, keywords, booleans, nil."
  [form]
  (or (number? form)
      (string? form)
      (keyword? form)
      (boolean? form)
      (nil? form)
      (char? form)))

;; ================================================================
;; Use counting
;; ================================================================

(defn- count-uses
  "Count how many times sym appears free in form."
  [sym form]
  (cond
    (= form sym) 1
    (seq? form)
    (let [head (first form)]
      (cond
        ;; let/let*/loop — sym might be shadowed
        (and (contains? #{:binding :scope} (:kind (form/form-info form)))
             (sequential? (second form)))
        (let [bindings (partition 2 (second form))
              ;; Count uses in init exprs (before potential shadowing)
              init-count (reduce
                          (fn [cnt [bsym init]]
                            (let [cnt (+ cnt (count-uses sym init))]
                              (if (= bsym sym)
                                (reduced cnt) ;; shadowed from here
                                cnt)))
                          0 bindings)
              ;; Check if shadowed in body
              shadowed? (some #(= sym (first %)) bindings)]
          (if shadowed?
            init-count
            (+ init-count (reduce + 0 (map #(count-uses sym %) (nnext form))))))

        ;; fn — sym might be shadowed by params
        (#{'fn 'fn*} head)
        0 ;; Conservative: don't inline into closures

        ;; General form
        :else
        (reduce + 0 (map #(count-uses sym %) form))))

    (vector? form) (reduce + 0 (map #(count-uses sym %) form))
    (map? form) (reduce + 0 (map #(+ (count-uses sym (key %))
                                     (count-uses sym (val %))) form))
    (set? form) (reduce + 0 (map #(count-uses sym %) form))
    :else 0))

(defn- trivial?
  "True if form is trivial enough to inline regardless of use count.
  Symbols, literals, small constants."
  [form]
  (or (symbol? form)
      (constant? form)))

;; ================================================================
;; Partial evaluation - single pass
;; ================================================================

(defn- pe-let
  "Partially evaluate a let/let*/loop form.
  - Propagate constants into body
  - Inline single-use bindings of trivial expressions
  - For loop: keep all bindings (recur needs stable locals),
    only PE the init exprs and body.

  Dead binding elimination is NOT done here — it belongs in the standalone
  DCE pass (pass-dce) which sees the final stable form. Combining DCE with
  constant propagation in one left-to-right scan is unsound: const-prop
  changes which symbols appear in the body via env substitution, but DCE's
  count-uses checks the ORIGINAL body, creating a window where bindings
  appear dead but are actually needed after substitution."
  [form const-env pe-fn]
  (let [[let-sym bindings & body] form
        is-loop? (#{'loop 'loop* 'dotimes} let-sym)
        binding-pairs (when (sequential? bindings) (partition 2 bindings))]
    (when-not (sequential? bindings)
      ;; Malformed let — PE sub-forms generically
      (throw (ex-info (str "pe-let: non-sequential bindings in " let-sym " form")
                      {:form form :bindings bindings :type (type bindings)})))
    (if is-loop?
      ;; Loop: PE init exprs but keep all bindings. Don't propagate
      ;; constants into body because recur will change them.
      (let [loop-syms (set (map first binding-pairs))
            body-env (apply dissoc const-env loop-syms)
            pe-bindings (vec (mapcat (fn [[sym init]]
                                       [sym (pe-fn init const-env)])
                                     binding-pairs))
            pe-body (map #(pe-fn % body-env) body)]
        (let [r (apply list let-sym pe-bindings pe-body)]
          (if-let [m (meta form)] (with-meta r m) r)))
      ;; Let/let*: constant propagation + trivial inlining
      (loop [pairs (seq binding-pairs)
             new-bindings []
             env const-env]
        (if-not pairs
          ;; All bindings processed — PE the body
          (let [pe-body (map #(pe-fn % env) body)]
            (if (empty? new-bindings)
              ;; All bindings eliminated — just return the body
              (if (= 1 (count pe-body))
                (first pe-body)
                (let [r (cons 'do pe-body)]
                  (if-let [m (meta form)] (with-meta r m) r)))
              ;; Reconstruct let with surviving bindings
              (let [r (apply list let-sym (vec (apply concat new-bindings)) pe-body)]
                (if-let [m (meta form)] (with-meta r m) r))))

          (let [[sym init] (first pairs)
                pe-init (pe-fn init env)]
            (cond
              ;; Constant — propagate into env, skip binding
              (constant? pe-init)
              (recur (next pairs) new-bindings (assoc env sym pe-init))

              ;; Single-use trivial expression — inline
              (and (trivial? pe-init)
                   (let [uses (reduce + 0 (map #(count-uses sym %) body))]
                     (<= uses 1)))
              (recur (next pairs) new-bindings (assoc env sym pe-init))

              ;; Keep the binding
              :else
              (recur (next pairs)
                     (conj new-bindings [sym pe-init])
                     env))))))))

(defn- pe-if
  "Partially evaluate an if form.
  - Dead branch elimination when test is constant
  - Simplify when both branches are equal"
  [form const-env pe-fn]
  (let [[_ test then else] form
        pe-test (pe-fn test const-env)]
    (cond
      ;; Constant true — take then branch
      (and (constant? pe-test) pe-test (not (and (number? pe-test) (zero? pe-test))))
      (pe-fn then const-env)

      ;; Constant false/nil — take else branch
      (or (nil? pe-test) (false? pe-test)
          (and (number? pe-test) (zero? pe-test) (not (number? pe-test))))
      ;; Actually only nil and false are falsy in Clojure
      (if (or (nil? pe-test) (false? pe-test))
        (if else (pe-fn else const-env) nil)
        ;; Non-nil, non-false constants are truthy
        (pe-fn then const-env))

      ;; Both branches same — just use one (still eval test for side effects)
      ;; Skip this if test might have side effects (conservative)

      :else
      (let [pe-then (pe-fn then const-env)
            pe-else (when else (pe-fn else const-env))
            r (if (some? pe-else)
                (list 'if pe-test pe-then pe-else)
                (list 'if pe-test pe-then))]
        (if-let [m (meta form)] (with-meta r m) r)))))

;; ================================================================
;; Main PE pass
;; ================================================================

(defn pe-pass
  "Single pass of partial evaluation.
  const-env: {symbol -> constant-value} for known bindings."
  [form const-env]
  (cond
    ;; Symbol — substitute if known constant
    (symbol? form)
    (if (contains? const-env form)
      (get const-env form)
      form)

    ;; Not a collection — return as-is
    (not (coll? form)) form

    ;; List (call or special form)
    (seq? form)
    (let [head (first form)]
      (cond
        ;; let/let*/loop
        (and (symbol? head) (contains? #{:binding :scope} (:kind (form/form-info form))))
        (pe-let form const-env pe-pass)

        ;; if
        (= head 'if)
        (pe-if form const-env pe-pass)

        ;; do — PE each form, eliminate constant statements except last
        (= head 'do)
        (let [pe-forms (map #(pe-pass % const-env) (rest form))
              ;; Keep non-constant forms (might have side effects) and last form
              kept (if (<= (count pe-forms) 1)
                     pe-forms
                     (let [init (butlast pe-forms)
                           last-form (last pe-forms)
                           ;; Keep non-trivial init forms (potential side effects)
                           kept-init (remove constant? init)]
                       (if (seq kept-init)
                         (concat kept-init [last-form])
                         [last-form])))]
          (case (count kept)
            0 nil
            1 (first kept)
            (cons 'do kept)))

        ;; fn/fn* — don't PE inside (different scope)
        (and (symbol? head) (#{'fn 'fn*} head))
        form

        ;; ftm — leave alone
        (= head 'ftm)
        form

        ;; quote — leave alone
        (= head 'quote)
        form

        ;; recur — PE the args but keep recur
        (= head 'recur)
        (let [r (cons 'recur (map #(pe-pass % const-env) (rest form)))]
          (if-let [m (meta form)] (with-meta r m) r))

        ;; throw — PE the expression
        (= head 'throw)
        (let [r (list 'throw (pe-pass (second form) const-env))]
          (if-let [m (meta form)] (with-meta r m) r))

        ;; try/catch — PE bodies
        (= head 'try)
        form ;; Conservative: don't PE inside try/catch

        ;; .invk — PE the args, then try simplify. If simplification doesn't
        ;; change anything, preserve the .invk form (with its typed interface
        ;; metadata) instead of returning a bare dispatch call. Normalizing to
        ;; bare ops is only useful when algebraic rules can actually simplify.
        (= head '.invk)
        (let [impl (second form)
              pe-args (map #(pe-pass % const-env) (nnext form))
              invk-form (apply list '.invk impl pe-args)
              invk-form (if-let [m (meta form)] (with-meta invk-form m) invk-form)
              normalized (normalize/normalize-1 invk-form)
              simplified (simp/simplify-1 normalized)]
          (if (= simplified normalized)
            ;; Simplification didn't help — keep the .invk form with metadata
            invk-form
            ;; Simplification reduced something — use simplified result
            (if-let [m (meta form)]
              (if (instance? clojure.lang.IMeta simplified) (with-meta simplified m) simplified)
              simplified)))

        ;; new — PE the constructor args
        (= head 'new)
        (let [class-sym (second form)
              pe-args (map #(pe-pass % const-env) (nnext form))
              r (apply list 'new class-sym pe-args)]
          (if-let [m (meta form)] (with-meta r m) r))

        ;; set! — PE value, keep target
        (= head 'set!)
        (let [[_ target val-form] form
              r (list 'set! target (pe-pass val-form const-env))]
          (if-let [m (meta form)] (with-meta r m) r))

        ;; case* — preserve structure, PE the test expression
        (= head 'case*)
        form

        ;; var — leave alone
        (= head 'var)
        form

        ;; General call — PE args, then try constant folding
        :else
        (let [pe-children (map #(pe-pass % const-env) form)
              r (simp/simplify-1 (apply list pe-children))]
          (if-let [m (meta form)]
            (if (instance? clojure.lang.IMeta r) (with-meta r m) r)
            r))))

    ;; Vector
    (vector? form)
    (mapv #(pe-pass % const-env) form)

    ;; Map
    (map? form)
    (into {} (map (fn [[k v]] [(pe-pass k const-env) (pe-pass v const-env)])) form)

    ;; Set
    (set? form)
    (into #{} (map #(pe-pass % const-env)) form)

    :else form))

;; ================================================================
;; Fixpoint PE
;; ================================================================

(defn pe
  "Partial evaluation with algebraic simplification.
  Runs PE + simplification to fixpoint (or max-rounds).

  const-env: {symbol -> value} for known constants (e.g. function parameters
  that are known at compile time).

  Returns the simplified form."
  ([form] (pe form {}))
  ([form const-env] (pe form const-env 10))
  ([form const-env max-rounds]
   (loop [f form
          n 0]
     (if (>= n max-rounds)
       f
       (let [pe-d (pe-pass f const-env)
             simplified (simp/simplify pe-d)]
         (if (= simplified f)
           simplified
           (recur simplified (inc n))))))))

;; ================================================================
;; Convenience: PE a deftm walked body
;; ================================================================

(defn pe-walked-body
  "Partially evaluate a walked deftm body with known parameter values.

  params: vector of parameter symbols
  values: map of {param -> known-value} (nil for unknown)
  body: the walked body forms (as from ::deftm-walked-body)

  Returns the simplified body forms."
  [params values body]
  (let [const-env (reduce (fn [env p]
                            (if-let [v (get values p)]
                              (assoc env p v)
                              env))
                          {} params)]
    (mapv #(pe % const-env) body)))

;; ================================================================
;; High-level specialization API
;; ================================================================

(defn specialize-var
  "Specialize a deftm var with known parameter values.

  var-or-sym: a Var or symbol naming a deftm function
  known-values: map of {param-symbol -> constant-value}

  Returns {:params [...] :body [...] :tags [...] :return-tag tag
           :original-body [...] :const-env {...}}
  where :body is the PE'd and simplified walked body."
  [var-or-sym known-values]
  (let [v (if (var? var-or-sym) var-or-sym (resolve var-or-sym))
        _ (when-not v (throw (ex-info "Cannot resolve var" {:sym var-or-sym})))
        m (meta v)
        params (or (:raster.core/deftm-params m)
                   (throw (ex-info "Not a deftm var" {:var v})))
        tags (:raster.core/deftm-tags m)
        walked-body (or (:raster.core/deftm-walked-body m)
                        (rcore/ensure-walked-body! v))
        return-tag (:raster.core/return-tag m)
        _ (when-not walked-body
            (throw (ex-info "No walked body on deftm var" {:var v})))
        specialized (pe-walked-body params known-values walked-body)]
    {:params params
     :tags tags
     :return-tag return-tag
     :body specialized
     :original-body walked-body
     :const-env known-values}))

(defn specialize-body
  "Convenience: specialize a walked body with known values.
  Returns just the simplified body forms.

  Usage:
    (specialize-body #'my-fn {'sigma 10.0 'rho 28.0})"
  [var-or-sym known-values]
  (:body (specialize-var var-or-sym known-values)))

(defn specialize-dimensions
  "PE body substituting dimension queries with constants.
  known-dims: {dim-expr -> int}, e.g. {(nc/dim x) -> 784, (nc/mrows W) -> 128}

  This replaces dimension queries in the body with their known integer values,
  making all buffer allocations pre-sized and hoistable."
  [walked-body known-dims]
  (let [;; Normalize forms to plain lists (strip metadata, coerce lazy seqs)
        ;; so structural equality works reliably as map keys
        normalize (fn normalize [form]
                    (cond (seq? form) (apply list (map normalize form))
                          (vector? form) (mapv normalize form)
                          :else form))
        expr->val (into {} (map (fn [[k v]] [(normalize k) v]) known-dims))
        replace-dims (fn replace-dims [form]
                       (cond
                         ;; Check if this form matches a known dim expression
                         (and (seq? form)
                              (contains? expr->val (normalize form)))
                         (get expr->val (normalize form))

                         ;; Recurse into sequences
                         (seq? form)
                         (with-meta (apply list (map replace-dims form))
                           (meta form))

                         ;; Recurse into vectors
                         (vector? form)
                         (with-meta (mapv replace-dims form)
                           (meta form))

                         :else form))]
    (if (vector? walked-body)
      (mapv replace-dims walked-body)
      (replace-dims walked-body))))
