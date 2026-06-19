(ns raster.compiler.passes.scalar.dce
  "Dead code elimination on flat let* forms.

	Uses backward liveness from body roots, effect descriptors from
	raster.compiler.passes.scalar.effects, and mutation-target identity
	from op-semantics."
  (:require [clojure.set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.passes.scalar.effects :as effects]))

(def ^:private free-syms
  util/free-syms-flat)

(defn- par-mutation-form?
  "True if expr is a par form that mutates an output buffer.
   par/map!, par/scan, par/scan-exclusive, par/stencil!, par/map-void!,
   par/scatter!, par/gather, par/butterfly!, par/rng-fill!, par/collect!,
   par/active-ids! all write to an output buffer (arg 0). par/reduce and
   par/map (pure) are not mutations."
  [expr]
  (and (par/par-form? expr)
       (not (par/par-reduce-form? expr))
       (not (par/par-reduce-by-key-form? expr))
       (not (par/par-map-pure-form? expr))))

(defn- extract-mutation-targets
  "Find buffer symbols mutated by mutation ops anywhere in expr."
  [expr]
  (cond
    ;; Par forms: only the first arg (output buffer) is mutated
    (par-mutation-form? expr)
    (let [buf-arg (second expr)]
      (if (symbol? buf-arg)
        #{buf-arg}
        (extract-mutation-targets buf-arg)))

    ;; Array writes (aset / clojure.core/aset / raster.arrays/aset) mutate their
    ;; FIRST arg (the array). Without this, a `_`-bound array-fill loop whose value
    ;; is unused is wrongly eliminated as dead — leaving the array unwritten.
    (and (seq? expr)
         (symbol? (first expr))
         (>= (count expr) 2)
         (descriptor/aset-op? (first expr)))
    (if (symbol? (second expr))
      #{(second expr)}
      (extract-mutation-targets (second expr)))

    (and (seq? expr)
         (symbol? (first expr))
         (>= (count expr) 2)
         (or (descriptor/mutating-op? (first expr))
             (descriptor/primitive-mutation-op? (first expr))
             (:mutating (meta expr))))
    ;; All symbol args are potential mutation targets for mutating ops
    (apply clojure.set/union #{}
           (map (fn [arg]
                  (if (symbol? arg)
                    #{arg}
                    (extract-mutation-targets arg)))
                (rest expr)))

    (and (seq? expr)
         (= '.invk (first expr))
         (>= (count expr) 3)
         (symbol? (second expr))
         (let [impl-sym (second expr)]
           (or (descriptor/mutating-op? impl-sym)
               (when-let [base (descriptor/extract-base-op impl-sym)]
                 (descriptor/mutating-op? base)))))
    (let [args (nnext expr)]
      (apply clojure.set/union #{}
             (map (fn [arg]
                    (if (symbol? arg)
                      #{arg}
                      (extract-mutation-targets arg)))
                  args)))

    (and (seq? expr)
         (= 'dotimes (first expr)))
    (let [[_ _bindings & body-forms] expr]
      (apply clojure.set/union #{} (map extract-mutation-targets body-forms)))

    (seq? expr)
    (apply clojure.set/union #{} (map extract-mutation-targets expr))

    (vector? expr)
    (apply clojure.set/union #{} (map extract-mutation-targets expr))

    :else #{}))

(defn- locally-pure-form?
  "True if expr is a form that has no externally visible side effects,
   even though beichte might classify it as non-pure. Covers:
   - Array allocations (double-array, zeros-like, similar, etc.)
   - .invk method calls where the original var is known-pure via beichte
   - par/reduce (pure reduction, no buffer mutation)"
  [expr]
  (and (seq? expr)
       (let [head (first expr)
             op (form/effective-op expr)]
         (or ;; Allocations (handles both direct and .invk forms)
          (and op (descriptor/alloc-op? op))
             ;; .invk: look up the original var's effect via beichte
          (and (= '.invk head)
               (let [impl-sym (second expr)
                     original-op (:raster.op/original (meta expr))]
                 (cond
                      ;; Known mutating ops are never pure
                   (descriptor/mutating-op? impl-sym) false
                      ;; If we have the original var, ask beichte
                   original-op
                   (when-let [v (try (resolve original-op) (catch Exception _ nil))]
                     (when (var? v)
                       (= :pure (effects/analyze-var-effect v))))
                      ;; No original op metadata — conservative: not pure
                   :else false)))
             ;; par/reduce is pure (no buffer mutation)
          (par/par-reduce-form? expr)))))

(defn- keep-unconditionally?
  "True if expr must be kept even if its result is unused.
   Uses beichte: anything not proven pure is kept.
   Locally-pure forms (allocations, .invk calls, par/reduce) are excluded —
   they only need to survive if their result is referenced or they mutate
   a live buffer."
  [expr]
  (and (not (locally-pure-form? expr))
       (let [{:keys [effect flags]} (effects/descriptor expr)]
         (or (not= :pure effect)
             (seq flags)))))

(defn eliminate-dead-bindings
  "Remove dead bindings from a flat let* form.
   Effectful bindings are preserved only when their side effects are
   externally visible: either they mutate a live buffer, or they have
   no identifiable mutation targets (opaque effects like IO).

   Seeds `live` with body free-syms PLUS all free symbols anywhere in the
   let* that aren't bound by it (function parameters and outer-scope refs).
   Without this, a binding that mutates a function parameter could be
   incorrectly DCE'd because backward iteration processes the mutation
   before the parameter becomes live via later bindings — but mutations of
   parameters are externally observable and must always be preserved."
  [let-form & {:keys [root-pred] :or {root-pred (constantly false)}}]
  (let [[let-sym bindings-vec & body-exprs] let-form
        pairs (vec (partition 2 bindings-vec))
        n (count pairs)
        bound-syms (set (map first pairs))
        body-free (apply clojure.set/union #{} (map free-syms body-exprs))
        binding-free (apply clojure.set/union #{}
                            (map #(free-syms (second %)) pairs))
        let-free (clojure.set/difference
                   (clojure.set/union body-free binding-free)
                   bound-syms)
        live (atom (clojure.set/union body-free let-free))
        _ (doseq [[sym expr] pairs]
            (when (or (root-pred sym)
                      ;; Only seed opaque effects (no identifiable mutation targets).
                      ;; Mutation effects are handled by mutates-live? in the backward pass.
                      (and (keep-unconditionally? expr)
                           (empty? (extract-mutation-targets expr))))
              (swap! live conj sym)))
        live-indices (atom #{})]
    (doseq [i (range (dec n) -1 -1)]
      (let [[sym expr] (nth pairs i)
            sym-meta (meta sym)
            referenced? (contains? @live sym)
            effect-binding? (:raster.effect/effectful sym-meta)
            mutation-targets (extract-mutation-targets expr)
            has-mutation-targets? (seq mutation-targets)
            ;; For ops with known mutation targets (par/map!, etc.): only keep
            ;; if the mutation target buffer is live (someone reads it).
            ;; For opaque effects (IO, unknown mutations): keep if any free var
            ;; is live (conservative fallback).
            mutates-live? (and has-mutation-targets?
                               (some @live mutation-targets))
            opaque-effect-refs-live? (and effect-binding?
                                          (not has-mutation-targets?)
                                          (some @live (free-syms expr)))]
        (when (or referenced? opaque-effect-refs-live? mutates-live?)
          (swap! live-indices conj i)
          (swap! live into (free-syms expr)))))
    (let [live-pairs (keep-indexed (fn [i pair]
                                     (when (contains? @live-indices i) pair))
                                   pairs)
          n-removed (- n (count (seq live-pairs)))
          live-bindings (vec (mapcat identity live-pairs))]
      {:form (let [r (list* let-sym live-bindings body-exprs)]
               (if-let [m (meta let-form)] (with-meta r m) r))
       :stats {:bindings-removed n-removed}})))