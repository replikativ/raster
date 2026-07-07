(ns raster.compiler.passes.scalar.effects
  "Effect classification for compiler passes via Beichte.

   Beichte is the sole authority for effect analysis. No syntactic fallbacks.
   When beichte cannot analyze an expression, it is conservatively assumed
   effectful (:io). Only expressions beichte proves pure get :pure.

   The raster context pre-registers raster.numeric and raster.math vars
   as :pure so beichte doesn't need to analyze their source each time."
  (:require [beichte.core :as b]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

(defonce ^:private raster-context
  (delay
    (let [;; Register all raster.numeric and raster.math vars as :pure
          pure-entries
          (into {}
                (mapcat (fn [ns-sym]
                          (try
                            (require ns-sym)
                            (map (fn [[_ v]] [v :pure])
                                 (ns-publics (the-ns ns-sym)))
                            (catch Exception e
                              (binding [*out* *err*]
                                (println "WARNING: effects.clj failed to load" ns-sym "-" (.getMessage e)))
                              nil)))
                        '[raster.numeric raster.math]))
          ;; Register raster.arrays: aget/alength → :pure, aset → :local
          array-entries
          (try
            (require 'raster.arrays)
            (let [publics (ns-publics (the-ns 'raster.arrays))
                  mutating descriptor/array-mutating-base-names]
              (into {} (map (fn [[sym v]]
                              [v (if (contains? mutating sym) :local :pure)])
                            publics)))
            (catch Exception e
              (binding [*out* *err*]
                (println "WARNING: effects.clj failed to load raster.arrays -" (.getMessage e)))
              {}))
          ;; Register the pure FORWARD loss reductions as :pure. Each is a
          ;; reduce over pred/target with no IO or buffer mutation, but beichte
          ;; cannot see through the (All [T]) dispatch var to its reduce! body,
          ;; so it conservatively defaults them to :io — which wrongly pins the
          ;; DEAD value+grad primal (e.g. the mse-loss forward emitted alongside
          ;; the gradients but unused when the caller computes its own loss) so
          ;; DCE can't drop it, and it then leaks a resident forward array into a
          ;; host scalar closure on the GPU-resident path. Only the forward
          ;; reductions are registered — the *-backward helpers allocate + aset
          ;; (:local), so they are intentionally excluded.
          loss-entries
          (try
            (require 'raster.dl.loss)
            (let [publics (ns-publics (the-ns 'raster.dl.loss))]
              (into {} (keep (fn [[sym v]]
                               (when (contains? '#{mse-loss cross-entropy-loss huber-loss l1-loss} sym)
                                 [v :pure]))
                             publics)))
            (catch Exception e
              (binding [*out* *err*]
                (println "WARNING: effects.clj failed to load raster.dl.loss -" (.getMessage e)))
              {}))
          raster-entries (merge pure-entries array-entries loss-entries)
          reg (b/extend-registry (b/default-registry) raster-entries)]
      (b/make-context {:registry reg}))))

(defn init-raster-context!
  "Force initialization of the shared Raster Beichte context."
  []
  @raster-context
  nil)

(defn get-raster-context
  "Return the shared Raster Beichte context."
  []
  @raster-context)

(defn- collect-locals
  "Collect free non-operator symbols from an expression for beichte's :locals.
   Excludes call-position symbols (they should resolve as vars, not locals)."
  [expr]
  (cond
    (symbol? expr)
    (when (and (not (namespace expr))
               (not (contains? form/known-form-heads expr)))
      #{expr})
    (seq? expr)
    ;; Skip head (operator position) — only collect from args
    (apply clojure.set/union #{} (map collect-locals (rest expr)))
    (vector? expr)
    (apply clojure.set/union #{} (map collect-locals expr))
    :else #{}))

(declare descriptor)

(defn analyze-effect
  "Analyze an expression and return its effect level (:pure, :local, :mutation, :io)."
  [expr]
  (:effect (descriptor expr)))

(defn analyze-var-effect
  "Analyze a var and return its effect level."
  [v]
  (b/analyze-var v @raster-context))

(defn descriptor
  "Return the effect descriptor for a compiler IR expression.

   Uses beichte with :locals (free symbols treated as locally-bound)
   so it can analyze walked IR without var resolution failures.

   Conservative default: if beichte fails, returns {:effect :io :flags #{}}
   (assumed effectful). Only proven-pure expressions get :pure."
  [expr]
  (if (not (seq? expr))
    {:effect :pure :flags #{}}
    (try
      (let [locals (collect-locals expr)
            result (b/analyze-full expr @raster-context locals)]
        (if (map? result)
          (update result :flags #(or % #{}))
          {:effect (or result :io) :flags #{}}))
      (catch Exception _
        ;; Conservative: unknown = effectful
        {:effect :io :flags #{}})
      (catch StackOverflowError _
        ;; Deep expressions overflow tools.analyzer — assume effectful
        {:effect :io :flags #{}}))))

(defn removable-expr?
  "True if an unused expression can be dropped without changing semantics."
  [expr]
  (let [{:keys [effect flags]} (descriptor expr)]
    (and (= :pure effect)
         (empty? flags))))

(defn cse-safe-expr?
  "True if an expression is safe to common-subexpression eliminate."
  [expr]
  (and (seq? expr)
       (symbol? (first expr))
       (removable-expr? expr)))
