(ns raster.compiler.passes.scalar.cse
  "Common Subexpression Elimination on flat let* forms.

   Walks forward through bindings, caching expr → first-sym.
   Duplicate expressions become aliases; DCE cleans dead bindings."
  (:require [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.ir.form :as form]))

(defn- normalize-expr
  "Normalize an expression for structural equality.
   Strips metadata from symbols (type hints vary but semantics don't)."
  [expr]
  (cond
    (symbol? expr) (let [m (meta expr)]
                     (with-meta expr (select-keys m [:raster.type/tag :raster.type/fn-info :raster.type/element :raster.type/source])))
    (seq? expr) (apply list (map normalize-expr expr))
    (vector? expr) (mapv normalize-expr expr)
    :else expr))

(defn- try-fold-nth
  "If expr is (clojure.core/nth <sym> <const-idx>) and sym is bound to a
   vector literal, return the element at that index. Otherwise nil."
  [expr vec-bindings]
  (when (and (seq? expr) (= 'clojure.core/nth (first expr)) (= 3 (count expr)))
    (let [target (second expr)
          idx-expr (nth expr 2)
          idx (cond (integer? idx-expr) idx-expr
                    (and (seq? idx-expr) (= 'long (first idx-expr)))
                    (second idx-expr)
                    :else nil)]
      (when (and (symbol? target) (integer? idx))
        (when-let [v (get vec-bindings target)]
          (when (and (>= idx 0) (< idx (count v)))
            (clojure.core/nth v idx)))))))

(defn- subst-syms-in
  "Substitute symbols in expr using smap."
  [smap expr]
  (cond
    (symbol? expr) (get smap expr expr)
    (seq? expr) (let [new-children (map (partial subst-syms-in smap) expr)]
                  (if-let [m (meta expr)]
                    (with-meta (apply list new-children) m)
                    (apply list new-children)))
    (vector? expr) (mapv (partial subst-syms-in smap) expr)
    :else expr))

(defn cse-let
  "Apply CSE and constant propagation to a flat let* form.
   Folds (nth vector-literal N) → element, propagates aliases.
   Returns {:form transformed-form :stats {:cse-aliases N}}."
  [form]
  (if-not (form/binding-form? form)
    {:form form :stats {:cse-aliases 0}}
    (let [[let-sym bindings-vec & body-exprs] form
          pairs (partition 2 bindings-vec)
          {:keys [new-pairs cache aliases vec-bindings sym-subst]}
          (reduce
           (fn [{:keys [new-pairs cache aliases vec-bindings sym-subst]} [sym expr]]
             (let [;; Apply pending substitutions to this binding's expr
                   expr (if (seq sym-subst) (subst-syms-in sym-subst expr) expr)
                   ;; Note: alias chains (dp__3→dp) are resolved by the SOAC fusion
                   ;; graph builder, not by CSE. Copy propagation here breaks the
                   ;; closure builder's binding expectations.
                   ;; Try vector-literal nth folding
                   folded (try-fold-nth expr vec-bindings)
                   expr (or folded expr)
                   ;; Track vector-literal bindings for nth folding
                   vec-bindings (if (vector? expr)
                                  (assoc vec-bindings sym expr)
                                  vec-bindings)
                   ;; If folded to a symbol, register as substitution
                   sym-subst (if (and folded (symbol? folded))
                               (assoc sym-subst sym folded)
                               sym-subst)
                   ;; CSE
                   norm (when (effects/cse-safe-expr? expr) (normalize-expr expr))
                   cached (when norm (get cache norm))]
               (if cached
                 {:new-pairs (conj new-pairs [sym cached])
                  :cache cache :aliases (inc aliases)
                  :vec-bindings vec-bindings :sym-subst (assoc sym-subst sym cached)}
                 {:new-pairs (conj new-pairs [sym expr])
                  :cache (if norm (assoc cache norm sym) cache)
                  :aliases aliases
                  :vec-bindings vec-bindings :sym-subst sym-subst})))
           {:new-pairs [] :cache {} :aliases 0 :vec-bindings {} :sym-subst {}}
           pairs)
          ;; Apply substitutions and nth folding to body expressions
          resolve-body-expr (fn resolve-body-expr [expr]
                              (cond
                                (and (seq? expr) (= 'clojure.core/nth (first expr)))
                                (let [resolved (try-fold-nth expr vec-bindings)]
                                  (if resolved
                                    (if (symbol? resolved) (get sym-subst resolved resolved) resolved)
                                    (let [r (apply list (map resolve-body-expr expr))]
                                      (if-let [m (meta expr)] (with-meta r m) r))))
                                (symbol? expr) (get sym-subst expr expr)
                                (seq? expr) (let [r (apply list (map resolve-body-expr expr))]
                                              (if-let [m (meta expr)] (with-meta r m) r))
                                (vector? expr) (mapv resolve-body-expr expr)
                                :else expr))
          resolved-body (map resolve-body-expr body-exprs)
          new-bindings (vec (mapcat identity new-pairs))]
      {:form (let [r (list* let-sym new-bindings resolved-body)]
               (if-let [m (meta form)] (with-meta r m) r))
       :stats {:cse-aliases aliases}})))
