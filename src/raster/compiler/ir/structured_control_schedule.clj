(ns raster.compiler.ir.structured-control-schedule
  "Checked schedule value for a typed sequential fixpoint."
  (:require [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]))

(defrecord ScheduledStructuredLoop [algorithm body graph strategy effects provenance attributes])

(defn scheduled-loop?
  [value]
  (and value
       (= "raster.compiler.ir.structured_control_schedule.ScheduledStructuredLoop"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :structured-control-schedule))))

(defn validate!
  [scheduled-loop]
  (when-not (scheduled-loop? scheduled-loop)
    (fail! :structured-loop-schedule-type "expected a ScheduledStructuredLoop"
           {:actual (type scheduled-loop)}))
  (let [{:keys [algorithm body graph strategy effects provenance attributes]} scheduled-loop
        algorithm (control/validate! algorithm)
        body (parallel-program/validate!
              body segop/segop-node?
              (fn [equation typed-algorithm]
                (and (= typed-algorithm (soac/validate! typed-algorithm))
                     (= (:operands equation) (:inputs (soac/facts typed-algorithm)))
                     (= (:results equation) (soac/outputs typed-algorithm)))))
        graph (graph/validate! graph)
        typed-body (control/body algorithm)
        retained-equations (vec (mapcat (comp soac/equations :algorithm) (:equations body)))]
    (when-not (= :segop (:dialect body))
      (fail! :structured-loop-body-dialect "structured loop body must be fully scheduled SegOp"
             {:dialect (:dialect body)}))
    (when-not (and (= (soac/equations typed-body) retained-equations)
                   (= (:inputs (soac/facts typed-body)) (:inputs body))
                   (= (soac/outputs typed-body) (:outputs body)))
      (fail! :structured-loop-algorithm
             "structured loop body boundary or equations changed during scheduling"
             {:algorithm-inputs (:inputs (soac/facts typed-body))
              :scheduled-inputs (:inputs body)
              :algorithm-outputs (soac/outputs typed-body)
              :scheduled-outputs (:outputs body)}))
    (when-not (= {:kind :host-repetition :association :sequential} strategy)
      (fail! :structured-loop-strategy "first structured loop schedule must be host repetition"
             {:strategy strategy}))
    (when-not (= effects (:effects (control/facts algorithm)))
      (fail! :structured-loop-effects "scheduled loop effects differ from its algorithm"
             {:scheduled effects :algorithm (:effects (control/facts algorithm))}))
    (doseq [[field value] [[:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :structured-loop-description "scheduled loop descriptions must be maps"
               {:field field :value value})))
    (when-not (= (vec (mapcat :operations (:equations body)))
                 (mapv :operation (:nodes graph)))
      (fail! :structured-loop-graph "KernelGraph operations differ from the scheduled body" {}))
    scheduled-loop))

(defn make
  [fields]
  (validate! (map->ScheduledStructuredLoop fields)))
