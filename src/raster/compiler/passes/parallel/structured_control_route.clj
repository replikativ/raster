(ns raster.compiler.passes.parallel.structured-control-route
  "Place a typed sequential fixpoint in Raster's common ParallelProgram envelope.

   TypedSOAC and TypedStructuredControl are two algorithm variants, not two program containers.
   This route gives structured control the same ordered equation/value spine used by loop-free
   algorithms. Scheduling replaces the equation's typed control operation with one checked
   ScheduledStructuredLoop; it does not reconstruct or recognize a source-shaped compound loop."
  (:require [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]))

(defn- ordered-distinct
  [values]
  (vec (distinct values)))

(defn- algorithm-boundary?
  [equation algorithm]
  (and (control/loop-program? algorithm)
       (= algorithm (control/validate! algorithm))
       (= (:operands equation) (ordered-distinct (control/outer-operands algorithm)))
       (= (:results equation) (control/outer-results algorithm))))

(defn program-envelope
  "Wrap one certified frontend decomposition in the shared typed program envelope."
  [{:keys [loop source loop-binding] :as decomposition}]
  (when-not (and (map? decomposition) loop)
    (throw (ex-info "structured-control routing requires a certified frontend decomposition"
                    {:reason :structured-control-decomposition
                     :decomposition decomposition})))
  (let [loop (control/validate! loop)
        facts (control/facts loop)
        operands (ordered-distinct (control/outer-operands loop))
        results (control/outer-results loop)
        equation
        (program/->ProgramEquation
         (:id facts) [:binding loop-binding] nil operands results loop [loop]
         (:effects facts)
         (assoc (:provenance facts) :pass :structured-control-route)
         (assoc (:attributes facts) :algorithm-dialect :typed-structured-control))]
    (program/make
     {:dialect :typed-parallel
      :source source
      :values (control/outer-values loop)
      :inputs operands
      :equations [equation]
      :outputs results
      :effects (:effects facts)
      :provenance {:source-dialect :typed-structured-control
                   :pass :structured-control-route}
      :attributes {:host-control :typed-structured-control}
      :operation? control/loop-program?
      :algorithm? algorithm-boundary?})))

(defn schedule-program
  "Schedule every typed-control equation through the ordinary loop-body SOAC vertical."
  [parallel-program opts]
  (let [parallel-program
        (program/validate! parallel-program control/loop-program? algorithm-boundary?)
        equations
        (mapv (fn [equation]
                (let [scheduled (lower/schedule (:algorithm equation) opts)]
                  (-> equation
                      (assoc :operations [scheduled])
                      (update :provenance assoc :target-dialect :structured-control-schedule)
                      (update :attributes assoc
                              :schedule-dialect :segop
                              :graph-dialect :kernel-graph))))
              (:equations parallel-program))]
    (program/make
     {:dialect :structured-control-schedule
      :source (:source parallel-program)
      :values (:values parallel-program)
      :inputs (:inputs parallel-program)
      :equations equations
      :outputs (:outputs parallel-program)
      :effects (:effects parallel-program)
      :diagnostics (:diagnostics parallel-program)
      :provenance (assoc (:provenance parallel-program)
                         :target-dialect :structured-control-schedule)
      :attributes (:attributes parallel-program)
      :operation? schedule/scheduled-loop?
      :algorithm? algorithm-boundary?})))
