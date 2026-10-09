(ns raster.compiler.passes.parallel.mixed-matrix-validation
  "Exact typed-law and stage reconstruction checks for mixed matrix schedules.

   The caller must independently derive the source graph. A narrow direct-matrix write-domain
   projection and closed operational numerical model are included. Finite error bounds,
   caller permission and hardware/dispatch admission remain separate."
  (:require [raster.compiler.ir.contraction-closure :as closure]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.matrix-stage :as matrix-stage]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.matrix-body-plan :as matrix-plan]
            [raster.compiler.passes.parallel.mixed-matrix-body :as body]
            [raster.compiler.passes.parallel.mixed-matrix-schedule :as schedule]
            [raster.compiler.passes.parallel.typed-contraction-context :as context]))

(defn- full-k-write-domains
  [algorithm source facts planned stage-bodies]
  (let [out (:out facts)
        value (get-in (soac/facts algorithm) [:values out])
        expected (:graph planned)
        terminal (last (:nodes expected))
        operation (:operation terminal)
        writers (filterv (fn [node]
                           (some #(and (= out (:buffer %))
                                       (contains? #{:write :read-write} (:access %)))
                                 (:uses node)))
                         (:nodes expected))
        dimensions (mapv second (:free-axes facts))]
    (when (and (= :float (:dtype facts)) (= :float (:dtype value))
               (closure/plain-storage? value)
               (contains? #{2 3} (count dimensions))
               (= [out] (mapv :id (:outputs source)))
               (not-any? #(= out (:id %)) (:inputs source))
               (= [terminal] writers)
               (matrix-stage/matrix-stage? operation)
               (= :full (get-in operation [:reduction :kind]))
               (= out (:result operation))
               (= dimensions (:result-shape operation))
               (= dimensions
                  (get ((if (= 3 (count dimensions))
                          matrix-plan/leading-batch-result-write-domain
                          matrix-plan/dense-result-write-domain)
                        (:body (last stage-bodies))) out)))
      {out (apply launch/product dimensions)})))

(defn- operational-numerical-model
  [planned stage-bodies]
  {:kind :mixed-matrix-operational-model :version 1 :mode :approximate-model
   ;; This records permitted operations, not a universal finite error bound: half overflow
   ;; and target instruction exceptional values cannot be bounded for arbitrary inputs.
   :contract (get-in planned [:refinement :numerics])
   :exceptional-values {:operand-conversion :ieee
                        :matrix-instruction :target-defined
                        :scalar-result :target-scalar-contract}
   :stages
   (mapv (fn [certificate]
           (cond-> {:numerics (:numerics certificate)
                    :storage (mapv #(select-keys % [:kind :dtype :shape :role])
                                   (get-in certificate [:body :parameters]))}
             (matrix-stage/matrix-stage? (:source certificate))
             (assoc :matrix
                    {:schedule (get-in certificate [:body :schedule])
                     :dimensions (get-in certificate [:body :attributes :dimension-values])
                     :iteration-range (get-in certificate [:body :attributes :iteration-range])
                     :batching (get-in certificate [:source :batching])
                     :input-value-regions (get-in certificate [:source :input-value-regions])
                     :result-transform (get-in certificate [:body :attributes :epilogue])})))
         stage-bodies)})

(defn validate-reconstruction!
  "Check the candidate against the retained algorithm and independently supplied source graph.

   Returns the reference plan and ordered stage bodies. Never obtain `source` from the candidate
   witness at an admission boundary: structural refinement validation cannot supply that proof."
  ([algorithm source refinement]
   (validate-reconstruction! algorithm source refinement {}))
  ([algorithm source refinement caller-options]
  (let [source (graph/validate! source)
        _ (when-not (= 1 (count (:nodes source)))
            (throw (ex-info "mixed matrix validation requires one semantic contraction"
                            {:reason :mixed-matrix-reconstruction-source})))
        {:keys [facts]} (context/validate-semantic! algorithm (get-in source [:nodes 0 :operation]))
        planned (schedule/reconstruct-refinement algorithm source refinement)
        expected (:graph planned)]
    (when-not (fingerprint/equivalent? expected (:graph refinement))
      (throw (ex-info "mixed matrix stage graph differs from independent reconstruction"
                      {:reason :mixed-matrix-reconstruction-graph})))
    (let [stage-bodies (mapv #(body/schedule-for-node % expected caller-options) (:nodes expected))]
      (assoc planned :stage-bodies stage-bodies
             :numerical-model (operational-numerical-model planned stage-bodies)
             :complete-write-domains
             (full-k-write-domains algorithm source facts planned stage-bodies))))))
