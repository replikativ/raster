(ns raster.compiler.passes.parallel.typed-contraction-context
  "Shared typed equation/scheduled contraction boundary, independent of target emission."
  (:require [raster.compiler.ir.contraction-facts :as cf]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.passes.parallel.typed-soac-projection :as typed-projection]))

(defn validate!
  "Validate that one scheduled SegRed is the physical schedule for its TypedSOAC equation.

   This is the typed semantic/schedule seam. The immutable equation id joins the two dialects;
   dtype, iteration space, physical operands and result storage are checked here so target routing
   cannot accept a SegRed with a different physical boundary. This checks boundary consistency,
   not equivalence of an arbitrary modified reduction body; scheduled equation/body certificates
   remain responsible for that stronger claim."
  [program operation]
  (let [program (soac-dialect/validate! program)
        _ (when-not (segop/seg-red? operation)
            (throw (ex-info "typed contraction boundary requires a scheduled SegRed"
                            {:reason :typed-contraction-operation :operation operation})))
        operation-id (:id operation)
        equation (or (some #(when (= operation-id (second %)) %)
                           (soac-dialect/equations program))
                     (throw (ex-info "typed contraction program lacks its scheduled equation"
                                     {:reason :typed-contraction-equation
                                      :operation operation-id})))
        components (typed-projection/segmented-reduce-contract-components program equation)
        facts (cf/from-components components)
        equation-dtype (:dtype components)
        expected-space (conj (:free-axes facts) (:flat-contract-axis facts))
        actual-space (mapv (juxt :name :bound) (get-in operation [:space :dims]))
        transform-inputs (set (map :sym (get-in facts [:epilogue :operands])))
        expected-inputs (into transform-inputs (map :sym) (:operands facts))
        expected-outputs #{(:out facts)}
        schedule (reduction/validate-schedule! (:schedule operation))]
    (when-not (= :contraction (:phase operation))
      (throw (ex-info "typed contraction SegRed has the wrong phase"
                      {:reason :typed-contraction-phase
                       :operation operation-id :phase (:phase operation)})))
    (when-not (= equation-dtype (:dtype operation))
      (throw (ex-info "typed contraction SegRed dtype disagrees with its equation"
                      {:reason :typed-contraction-operation-dtype
                       :operation operation-id :equation-dtype equation-dtype
                       :operation-dtype (:dtype operation)})))
    (when-not (= expected-space actual-space)
      (throw (ex-info "typed contraction SegRed iteration space disagrees with its equation"
                      {:reason :typed-contraction-space
                       :operation operation-id :expected expected-space :actual actual-space})))
    (when-not (and (= expected-inputs (segop/operation-inputs operation))
                   (= expected-outputs (segop/operation-outputs operation)))
      (throw (ex-info "typed contraction SegRed storage boundary disagrees with its equation"
                      {:reason :typed-contraction-storage
                       :operation operation-id
                       :expected-inputs expected-inputs
                       :actual-inputs (segop/operation-inputs operation)
                       :expected-outputs expected-outputs
                       :actual-outputs (segop/operation-outputs operation)})))
    {:program program
     :operation operation
     :operation-id operation-id
     :equation equation
     :components components
     :facts facts
     :dtype equation-dtype
     :schedule schedule}))
