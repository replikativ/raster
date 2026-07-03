(ns raster.compiler.passes.scalar.peephole
  "Extensible peephole optimization registry.

  Pattern-based rewrite rules that operate on flat (let* [bindings...] body)
  S-expressions. Rules are registered via register-peephole-rule! and applied
  iteratively until no more fusions are found.

  Includes infrastructure for binding manipulation, alias inlining,
  and nil-binding removal."
  (:require [clojure.walk]
            [raster.compiler.ir.form :as form]
            [raster.compiler.core.util :as util]))

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

(defn- subst-aliases
  "Substitute known aliases (alias-map sym -> target) throughout an expression."
  [alias-map e]
  (if (and (seq? e) (seq alias-map))
    (clojure.walk/postwalk
     (fn [x] (if (and (symbol? x) (contains? alias-map x)) (get alias-map x) x)) e)
    e))

(defn- inline-aliases
  "Copy-propagation over flat let* bindings. Resolves two alias shapes by recording
  sym -> target and substituting the target downstream:
   - DIRECT symbol alias (sym = other-sym, non-effectful) — nil the binding (dead).
   - EFFECTFUL alias whose init's tail position is a symbol (e.g. AD chains, or
     r = (let* [..writes b..] b) from inlining a buffer-producing op) — KEEP the
     binding for its writes; DCE removes it later if it turns out dead+pure.
  This lets peephole/DCE see that d_W2__19 IS dg_W__8, and that r IS b.
  Returns {:pairs updated-pairs :aliases alias-map}."
  [pairs]
  (let [n (count pairs)]
    (loop [i 0 result pairs alias-map {}]
      (if (>= i n)
        {:pairs result :aliases alias-map}
        (let [[sym init] (nth result i)
              init* (subst-aliases alias-map init)
              tail (form/tail-symbol init*)]
          (cond
            ;; direct symbol alias — record + nil out (dead after substitution)
            (and (symbol? init*) (not (:raster.effect/effectful (meta sym))))
            (recur (inc i) (assoc result i [sym nil])
                   (assoc alias-map sym (get alias-map init* init*)))

            ;; effectful binding whose tail is a (different) symbol — record the alias
            ;; for downstream substitution but KEEP the binding for its effects.
            ;; SOUNDNESS: only when `tail` is FREE in init* — i.e. it names an
            ;; enclosing-scope value (the r=(let* [..writes b..] b) shape). A tail
            ;; symbol BOUND inside init* (a loop*/let* accumulator, e.g. the `acc` of
            ;; (loop* [acc ..] .. acc)) does NOT exist in the outer scope; aliasing
            ;; s→acc there leaks a scope-local name (undeclared-identifier in C /
            ;; miscompile). Reductions bound to a name + used downstream hit this.
            (and (symbol? tail) (not= tail sym) (not (symbol? init*))
                 (contains? (util/free-syms init*) tail))
            (recur (inc i) (assoc result i [sym init*])
                   (assoc alias-map sym (get alias-map tail tail)))

            :else
            (recur (inc i) (assoc result i [sym init*]) alias-map)))))))

(defn copy-propagate-let
  "Copy-propagation on a (let* [bindings] body...) form: resolve alias bindings (see
  inline-aliases) by substituting their target throughout subsequent bindings AND the
  body, dropping nil'd (dead pure) alias bindings. Returns the rewritten let* (or the
  form unchanged if not a let*). The single, shared copy-prop — backends should use
  this rather than re-deriving aliasing."
  [form]
  (if-not (and (seq? form) (contains? #{'let* 'let} (first form)))
    form
    (let [[hd binds & body] form
          {:keys [pairs aliases]} (inline-aliases (bindings->pairs binds))]
      (cons hd (cons (pairs->bindings (remove-nil-bindings pairs))
                     (map #(subst-aliases aliases %) body))))))

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
