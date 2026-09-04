(ns raster.compiler.ir.emitted-parallel-equation
  "Checked target emission of one scheduled TypedSOAC equation."
  (:require [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]))

(defrecord EmittedParallelEquation [algorithm body refinement graph provenance attributes])

(defn emitted-equation?
  [value]
  (and value
       (= "raster.compiler.ir.emitted_parallel_equation.EmittedParallelEquation"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :emitted-parallel-equation))))

(defn- algorithm-boundary?
  [equation algorithm]
  (and (= algorithm (soac/validate! algorithm))
       (= (:operands equation) (:inputs (soac/facts algorithm)))
       (= (:results equation) (soac/outputs algorithm))))

(defn validate!
  [emitted-equation]
  (when-not (emitted-equation? emitted-equation)
    (fail! :emitted-parallel-equation-type
           "expected an EmittedParallelEquation"
           {:actual (type emitted-equation)}))
  (let [{:keys [algorithm body refinement graph provenance attributes]} emitted-equation
        algorithm (soac/validate! algorithm)
        body (program/validate! body segop/segop-node? algorithm-boundary?)
        expected (equation-graph/make algorithm body)
        refinement (when refinement (refinement/validate-against! refinement expected))
        scheduled (if refinement (refinement/scheduled-graph refinement) expected)
        emitted (-> graph graph/validate! executable/validate!)]
    (when-not (every? (comp artifact/kernel-artifact? :operation) (:nodes emitted))
      (fail! :emitted-parallel-equation-artifact
             "emitted equation graph requires only KernelArtifact nodes" {}))
    (when-not (graph/dataflow-equivalent? scheduled emitted)
      (fail! :emitted-parallel-equation-dataflow
             "target emission changed scheduled equation dataflow"
             {:scheduled (graph/dataflow-contract scheduled)
              :emitted (graph/dataflow-contract emitted)}))
    (when-not (every? true?
                      (map (fn [scheduled-node emitted-node]
                             (let [certificate (get-in emitted-node
                                                       [:operation :provenance
                                                        :scheduled-operation])]
                               (if (scheduled-body/scheduled-kernel-body? certificate)
                                 (do (scheduled-body/validate-against-node!
                                      certificate scheduled-node scheduled)
                                     (scheduled-body/validate-artifact-projection!
                                      certificate (:operation emitted-node))
                                     true)
                                 (= (:operation scheduled-node) certificate))))
                           (:nodes scheduled) (:nodes emitted)))
      (fail! :emitted-parallel-equation-operation
             "target emission changed a scheduled equation operation certificate" {}))
    (doseq [[field value] [[:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :emitted-parallel-equation-description
               "emitted equation descriptions must be maps"
               {:field field :value value})))
    emitted-equation))

(defn make
  ([algorithm body emitted]
   (make algorithm body emitted {}))
  ([algorithm body emitted {:keys [refinement provenance attributes]
                            :or {provenance {} attributes {}}}]
   (validate!
    (->EmittedParallelEquation algorithm body refinement emitted provenance attributes))))
