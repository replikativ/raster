(ns raster.compiler.passes.parallel.mixed-matrix-validation
  "Exact typed-law and stage reconstruction checks for mixed matrix schedules.

   The caller must independently derive the source graph. These checks do not establish a
   terminal complete-write domain, a composed numerical error bound, or dispatch admission."
  (:require [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.passes.parallel.mixed-matrix-body :as body]
            [raster.compiler.passes.parallel.mixed-matrix-schedule :as schedule]
            [raster.compiler.passes.parallel.typed-contraction-context :as context]))

(defn validate-reconstruction!
  "Check the candidate against the retained algorithm and independently supplied source graph.

   Returns the reference plan and ordered stage bodies. Never obtain `source` from the candidate
   witness at an admission boundary: structural refinement validation cannot supply that proof."
  [algorithm source refinement]
  (let [source (graph/validate! source)
        _ (when-not (= 1 (count (:nodes source)))
            (throw (ex-info "mixed matrix validation requires one semantic contraction"
                            {:reason :mixed-matrix-reconstruction-source})))
        _ (context/validate-semantic! algorithm (get-in source [:nodes 0 :operation]))
        planned (schedule/reconstruct-refinement algorithm source refinement)
        expected (:graph planned)]
    (when-not (fingerprint/equivalent? expected (:graph refinement))
      (throw (ex-info "mixed matrix stage graph differs from independent reconstruction"
                      {:reason :mixed-matrix-reconstruction-graph})))
    (assoc planned :stage-bodies
           (mapv #(body/schedule-for-node % expected) (:nodes expected)))))
