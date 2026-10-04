(ns raster.compiler.passes.parallel.mixed-matrix-validation
  "Exact typed-law and stage reconstruction checks for mixed matrix schedules.

   The caller must independently derive the source graph. A narrow direct-matrix write-domain
   projection is included; composed numerical error bounds and dispatch admission remain separate."
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

(defn- direct-write-domains
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
               (= 2 (count dimensions))
               (= [out] (mapv :id (:outputs source)))
               (not-any? #(= out (:id %)) (:inputs source))
               (= [terminal] writers)
               (matrix-stage/matrix-stage? operation)
               (= :full (get-in operation [:reduction :kind]))
               (= out (:result operation))
               (= dimensions (:result-shape operation))
               (= dimensions
                  (get (matrix-plan/dense-result-write-domain (:body (last stage-bodies))) out)))
      {out (apply launch/product dimensions)})))

(defn validate-reconstruction!
  "Check the candidate against the retained algorithm and independently supplied source graph.

   Returns the reference plan and ordered stage bodies. Never obtain `source` from the candidate
   witness at an admission boundary: structural refinement validation cannot supply that proof."
  [algorithm source refinement]
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
    (let [stage-bodies (mapv #(body/schedule-for-node % expected) (:nodes expected))]
      (assoc planned :stage-bodies stage-bodies
             :complete-write-domains
             (direct-write-domains algorithm source facts planned stage-bodies)))))
