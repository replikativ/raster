(ns raster.compiler.passes.scalar.rewalk
  "PE+CSE fixpoint plus re-walk pass.

  Cleans up flattened scalar code via PE+CSE, then re-runs the walker
  with TC type inference to devirtualize scalar arithmetic back into
  .invk calls. Used after AD inlining in the gradient/training pipelines."
  (:require [raster.compiler.passes.scalar.cse :as cse]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.ir.form :as form]))

(defn- pe-cse-fixpoint
  "Run PE + CSE in a fixpoint loop until convergence or max rounds."
  [form max-rounds]
  (loop [current form
         rounds 0]
    (if (>= rounds max-rounds)
      current
      (let [pe-form (pe/pe current)
            cse-form (:form (cse/cse-let pe-form))
            pe-form-2 (pe/pe cse-form)]
        (if (= pe-form-2 current)
          current
          (recur pe-form-2 (inc rounds)))))))

(defn pe-rewalk
  "PE+CSE fixpoint plus re-walk to devirtualize scalar arithmetic.

  Returns {:form :stats} when enabled, or the original form when simplify is
  disabled in opts, matching the surrounding pipeline pass convention."
  [form opts]
  (if-not (:simplify? opts)
    form
    (let [optimized (pe-cse-fixpoint form 5)
          active-params (:active-params opts)
          current-dtype (:dtype opts)
          scalar-tag (dtype/scalar-tag-for-dtype current-dtype)
          param-env (if-let [env (:param-env opts)]
                      (dtype/remap-env env current-dtype)
                      (if active-params
                        (into {}
                              (map (fn [param]
                                     [(with-meta (if (symbol? param) param (symbol (name param))) nil)
                                      scalar-tag])
                                   active-params))
                        {}))
          ;; Include training-injected params (lr, optimizer state) in env
          param-env (if-let [lr (:lr-sym opts)]
                      (assoc param-env (with-meta (if (symbol? lr) lr (symbol (name lr))) nil) scalar-tag)
                      param-env)
          ;; Type-env from param-env. Uses build-param-type-env for consistency
          ;; with definition-time and AOT-time walks.
          type-env (inf/build-param-type-env
                    (vec (keys param-env))
                    (vec (vals param-env)))
          ;; TC binding tags: run TC once on the post-AD form for systematic
          ;; type inference of all let bindings.
          tc-binding-tags (inf/tc-infer-binding-tags param-env optimized)
          rewalked (try
                     (walker/walk-body optimized
                                       (cond-> {:type-env type-env}
                                         (:source-ns opts) (assoc :source-ns (:source-ns opts))
                                         (seq tc-binding-tags) (assoc :tc-binding-tags tc-binding-tags)))
                     (catch Exception e
                       (binding [*out* *err*]
                         (println (str "WARNING: rewalk failed, passing through: " (.getMessage e))))
                       optimized))
          orig-bindings (when (form/binding-form? form)
                          (count (partition 2 (second form))))
          final-bindings (when (form/binding-form? rewalked)
                           (count (partition 2 (second rewalked))))]
      {:form rewalked
       :stats {:pe-rewalk-bindings-before (or orig-bindings 0)
               :pe-rewalk-bindings-after (or final-bindings 0)}})))