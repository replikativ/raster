(ns raster.compiler.ir.emitted-structured-loop
  "Checked target emission of one ScheduledStructuredLoop.

   The value pairs the exact semantic/scheduled loop with an emitted KernelGraph. It is the
   operation carried by an equation-first target program before runtime buffers and scalar values
   are bound into a StructuredLoopCall."
  (:require [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.structured-control-schedule :as schedule]))

(defrecord EmittedStructuredLoop [schedule graph provenance attributes])

(defn emitted-loop?
  [value]
  (and value
       (= "raster.compiler.ir.emitted_structured_loop.EmittedStructuredLoop"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :emitted-structured-loop))))

(defn validate!
  [emitted-loop]
  (when-not (emitted-loop? emitted-loop)
    (fail! :emitted-structured-loop-type
           "expected an EmittedStructuredLoop"
           {:actual (type emitted-loop)}))
  (let [{scheduled :schedule emitted :graph provenance :provenance attributes :attributes}
        emitted-loop
        scheduled (schedule/validate! scheduled)
        emitted (graph/validate! emitted)]
    (when-not (every? (comp artifact/kernel-artifact? :operation) (:nodes emitted))
      (fail! :emitted-structured-loop-artifact
             "emitted structured-loop graph requires only KernelArtifact nodes" {}))
    (when-not (graph/dataflow-equivalent? (:graph scheduled) emitted)
      (fail! :emitted-structured-loop-dataflow
             "target emission changed the scheduled loop dataflow"
             {:scheduled (graph/dataflow-contract (:graph scheduled))
              :emitted (graph/dataflow-contract emitted)}))
    (when-not (every? true?
                      (map (fn [scheduled-node emitted-node]
                             (= (:operation scheduled-node)
                                (get-in emitted-node
                                        [:operation :provenance :scheduled-operation])))
                           (:nodes (:graph scheduled)) (:nodes emitted)))
      (fail! :emitted-structured-loop-operation
             "target emission changed a scheduled loop operation certificate" {}))
    (doseq [[field value] [[:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :emitted-structured-loop-description
               "emitted structured-loop descriptions must be maps"
               {:field field :value value})))
    emitted-loop))

(defn make
  ([scheduled emitted]
   (make scheduled emitted {}))
  ([scheduled emitted {:keys [provenance attributes]
                       :or {provenance {} attributes {}}}]
   (validate!
    (->EmittedStructuredLoop scheduled emitted provenance attributes))))
