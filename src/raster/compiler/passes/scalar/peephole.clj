(ns raster.compiler.passes.scalar.peephole
  "Extensible peephole optimization registry.

  Pattern-based rewrite rules that operate on flat (let* [bindings...] body)
  S-expressions. Rules are registered via register-peephole-rule! and applied
  iteratively until no more fusions are found.

  Includes infrastructure for binding manipulation, alias inlining,
  and nil-binding removal."
  (:require [clojure.walk]
            [raster.compiler.ir.form :as form]))

;; ================================================================
;; Binding manipulation helpers
;; ================================================================

(defn- bindings->pairs
  "Convert flat bindings vector [s1 e1 s2 e2 ...] to [[s1 e1] [s2 e2] ...]."
  [bindings-vec]
  (vec (partition 2 bindings-vec)))

(defn- pairs->bindings
  "Convert [[s1 e1] [s2 e2] ...] back to flat [s1 e1 s2 e2 ...]."
  [pairs]
  (vec (mapcat identity pairs)))

(defn- remove-nil-bindings
  "Remove binding pairs where init is nil (fused away)."
  [pairs]
  (vec (remove (fn [[_s e]] (nil? e)) pairs)))

(defn- inline-aliases
  "Resolve trivial alias bindings (sym = other-sym) by substituting
  the alias target throughout subsequent bindings, then nil out the
  alias binding (it's dead after substitution).
  This is needed because AD generates alias chains like:
    dg_W__8 = (some-op ...)
    d_W2__19 = dg_W__8          ;; alias
    _ = (update-op! neg_lr d_W2__19 W2)
  Without inlining, peephole can't see that d_W2__19 IS dg_W__8.
  Returns {:pairs updated-pairs :aliases alias-map}."
  [pairs]
  (let [n (count pairs)]
    (loop [i 0
           result pairs
           alias-map {}]
      (if (>= i n)
        {:pairs result :aliases alias-map}
        (let [[sym init] (nth result i)]
          (if (and (symbol? init) (not (:raster.effect/effectful (meta sym))))
            ;; This is an alias: sym = init (or transitively resolved)
            ;; Nil it out — all downstream uses are substituted
            ;; But preserve effectful bindings
            (let [resolved (get alias-map init init)
                  new-map (assoc alias-map sym resolved)]
              (recur (inc i) (assoc result i [sym nil]) new-map))
            ;; Not an alias — substitute any known aliases in the expression
            (let [substituted (if (and (seq? init) (seq alias-map))
                                (clojure.walk/postwalk
                                 (fn [x] (if (and (symbol? x) (contains? alias-map x))
                                           (get alias-map x)
                                           x))
                                 init)
                                init)]
              (recur (inc i) (assoc result i [sym substituted]) alias-map))))))))

;; ================================================================
;; Peephole pattern registry
;; ================================================================

;; Registry of peephole rewrite rules. Each entry:
;;   {:name string, :fn (fn [pairs] -> {:pairs updated :count N :stats-key kw} or nil)}
(defonce ^:private peephole-registry (atom []))

(defn register-peephole-rule!
  "Register a peephole optimization rule.
  name:    descriptive name for diagnostics
  rule-fn: (fn [pairs] -> {:pairs updated-pairs :count N :stats-key keyword} or nil)
           stats-key is the key used in the returned stats map."
  [name rule-fn]
  (swap! peephole-registry conj {:name name :fn rule-fn})
  nil)

;; ================================================================
;; Public API
;; ================================================================

(defn apply-rules
  "Apply all registered peephole optimizations to a let* form.
  Iterates until no more rewrites are found.
  Returns {:form optimized-form :stats {...}}."
  [let-form]
  (if-not (and (seq? let-form) (form/binding-form? let-form))
    {:form let-form :stats {:rk-fused 0 :copy-fused 0}}
    (let [[let-sym bindings-vec & body-exprs] let-form
          {:keys [pairs aliases]} (inline-aliases (bindings->pairs bindings-vec))
          ;; Apply alias substitutions to body expressions too
          body-exprs (if (seq aliases)
                       (map (fn [expr]
                              (clojure.walk/postwalk
                               (fn [x] (if (and (symbol? x) (contains? aliases x))
                                         (get aliases x)
                                         x))
                               expr))
                            body-exprs)
                       body-exprs)
          rules @peephole-registry]
      (loop [pairs pairs
             totals {}]
        ;; Apply all registered rules in one pass
        (let [{new-pairs :pairs
               round-totals :totals
               any-fused? :any-fused?}
              (reduce (fn [acc rule]
                        (let [rule-fn (:fn rule)
                              result (rule-fn (:pairs acc))]
                          (if result
                            (-> acc
                                (assoc :pairs (:pairs result))
                                (update :totals
                                        (fn [t] (update t (:stats-key result)
                                                        (fnil + 0) (:count result))))
                                (assoc :any-fused? true))
                            acc)))
                      {:pairs pairs :totals {} :any-fused? false}
                      rules)]
          (if-not any-fused?
            ;; No more fusions — done
            (let [clean-pairs (remove-nil-bindings pairs)
                  new-bindings (pairs->bindings clean-pairs)
                  ;; Ensure default stats keys are present
                  final-stats (merge {:rk-fused 0 :copy-fused 0} totals)]
              {:form (list* let-sym new-bindings body-exprs)
               :stats final-stats})
            ;; At least one fusion — iterate with accumulated totals
            (recur new-pairs (merge-with + totals round-totals))))))))
