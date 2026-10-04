(ns raster.compiler.passes.parallel.typed-contraction-context
  "Shared typed equation/scheduled contraction boundary, independent of target emission."
  (:require [raster.compiler.ir.contraction-facts :as cf]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.passes.parallel.typed-soac-projection :as typed-projection]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]))

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

(defn validate-semantic!
  "Check the exact canonical typed contraction law through its original semantic constructor.

   Unlike validate!, this also compares reduction state, algebra and scope. Grid and schedule
   remain physical policy, checked separately by schedule reconstruction. This admission is
   deliberately restricted to lambda-local-free contractions: the existing segmented lowerer
   does not yet retain arbitrary lambda locals as an ordered reduction region."
  [program operation]
  (let [{:keys [program equation dtype] :as context} (validate! program operation)
        lambda (:lambda (soac-dialect/operation-parts equation))
        _ (when (seq (:locals (soac-dialect/lambda-parts lambda)))
            (throw (ex-info "canonical contraction semantic proof requires a closed scalar body"
                            {:reason :typed-contraction-semantic-locals
                             :operation (:id operation)})))
        expected (:operation
                  (soac-lower/project-typed-segmented-reduce
                   program :dtype dtype :flat-idx (get-in operation [:space :flat-idx])))
        semantic-view #(dissoc % :grid :schedule)]
    (when-not (and expected
                   (fingerprint/equivalent? (semantic-view expected) (semantic-view operation)))
      (throw (ex-info "scheduled contraction changed its exact typed reduction semantics"
                      {:reason :typed-contraction-semantic-operation
                       :operation (:id operation)})))
    context))
