(ns raster.compiler.passes.parallel.typed-contraction-test-support
  "Test construction through the production TypedSOAC contraction-result fusion route."
  (:require [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]))

(defn fuse-result-map
  "Build `contract` followed by one flat result map, run certified fusion, and project the
   validated equation to the temporary contraction leaf vocabulary. Returns nil when legality
   keeps the two operations separate."
  [contract destination extent map-body & {:keys [array-types scalar-types]
                                           :or {array-types {} scalar-types {}}}]
  (let [source (list 'let*
                     ['contract-step contract
                      'map-step
                      (list 'raster.par/map! destination 't extent nil map-body)]
                     'map-step)
        program (frontend/form->program source
                                        {:dtype :float
                                         :array-types array-types
                                         :scalar-types scalar-types})
        [fused stats] (fusion/fusion-fixpoint program)]
    (when (and (= 1 (:vertical stats))
               (= 1 (count (dialect/equations fused))))
      (let [equation (first (dialect/equations fused))
            form (projection/segmented-reduce-contract-form fused equation)]
        {:program fused
         :equation equation
         :form form
         :epilogue (:epilogue (apply hash-map (drop 5 form)))
         :stats stats}))))
