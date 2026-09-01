(ns raster.compiler.passes.parallel.structured-control-lower
  "Certified scheduling of typed sequential control.

   The loop remains a sequential fixpoint while its closed TypedSOAC body takes the ordinary
   TypedSOAC -> ParallelProgram -> SegOp -> KernelGraph vertical.  This pass deliberately does not
   emit a persistent kernel or reconstruct a source-shaped compound program."
  (:require [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as control-schedule]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]))

(def scheduled-loop? control-schedule/scheduled-loop?)

(def validate! control-schedule/validate!)

(defn schedule
  "Schedule a typed structured loop without changing its sequential association.

   The returned KernelGraph describes one iteration. A runtime host-repetition binder can replay
   it with the loop's ordered invariant/carry bindings; persistent device execution is a distinct
   future schedule, not an emitter fallback."
  [algorithm opts]
  (let [algorithm (control/validate! algorithm)
        envelope (typed-route/program-envelope (control/body algorithm))
        scheduled (:form (segop-lower/segop-lower-pass envelope opts))
        graph (control-schedule/iteration-graph algorithm scheduled)]
    (control-schedule/make
     {:algorithm algorithm
      :body scheduled
      :graph graph
      :strategy {:kind :host-repetition :association :sequential}
      :effects (:effects (control/facts algorithm))
      :provenance {:source-dialect :typed-structured-control
                   :pass :structured-control-lower}
      :attributes {:body-dialect :typed-soac
                   :schedule-dialect :segop
                   :graph-dialect :kernel-graph}})))
