(ns raster.compiler.ir.emitted-parallel-equation
  "Checked target emission of one scheduled TypedSOAC equation."
  (:require [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.indexed-weighted-reduction-body :as indexed-body]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.mixed-matrix-validation :as mixed-validation]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.passes.parallel.segscan-body :as segscan-body]
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

(defn- expected-graph
  [algorithm body caller-options]
  (if (swr/plan? algorithm)
    (let [numerical (filterv #(not (true? (get-in % [:attributes :host-only])))
                             (:equations body))
          equation (first numerical)]
      (when-not (and (= 1 (count numerical)) (= algorithm (:algorithm equation))
                     (= 1 (count (:operations equation)))
                     (scheduled-body/scheduled-kernel-body? (first (:operations equation))))
        (fail! :emitted-parallel-equation-algorithm
               "emitted reduction body requires its exact single scheduled semantic equation" {}))
      (let [graph (:graph (equation-graph/make-for-plan-equation body equation))
            certificate (first (:operations equation))
            width (get-in certificate [:body :launch :workgroup-size 0])
            subgroup? (= :indexed-segmented-reduction-subgroup-score-reuse
                         (get-in certificate [:attributes :strategy]))
            expected ((if subgroup? indexed-body/schedule-score-reuse-for-node
                                      indexed-body/schedule-reference-for-node)
                      algorithm (first (:nodes graph)) graph
                      ;; Reconstruct the leaf, not target discovery: original compilation owns
                      ;; target admission. Every body, binding, numerical and launch obligation
                      ;; must still equal the generated candidate below.
                      (cond-> {:subgroup-size width :max-workgroup-size width}
                        subgroup? (assoc :device-type :gpu :vendor "Intel"
                                         :subgroup-sizes #{width}))
                      caller-options)]
        (when-not (semantic-fingerprint/equivalent? expected certificate)
          (fail! (if subgroup? :emitted-reduction-subgroup-refinement
                               :emitted-reduction-reference-refinement)
                 "protected reduction requires its exact generated schedule" {}))
        (scheduled-body/validate-against-node!
         certificate (first (:nodes graph)) graph)
        graph))
    ;; The canonical graph constructor validates the complete algorithm and scheduled body,
    ;; including the same operand/result boundary. Do not repeat that proof immediately here.
    ;; Each independent emitted-equation validation still reconstructs its graph from scratch.
    (equation-graph/make algorithm body)))

(defn- validation-report
  "Check the complete boundary and retain its source graph only for synchronous projection.
   Public validators still derive fresh reports; no graph or validation authority is cached."
  [emitted-equation caller-options]
  (when-not (emitted-equation? emitted-equation)
    (fail! :emitted-parallel-equation-type
           "expected an EmittedParallelEquation"
           {:actual (type emitted-equation)}))
  (let [{:keys [algorithm body refinement graph provenance attributes]} emitted-equation
        policy (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
        expected (expected-graph algorithm body caller-options)
        mixed? (= :mixed-precision-contraction (get-in refinement [:schedule :kind]))
        mixed (when mixed?
                (mixed-validation/validate-reconstruction! algorithm expected refinement caller-options))
        refinement (when refinement
                     (if mixed? refinement (refinement/validate-against! refinement expected)))
        scheduled (or (:graph mixed)
                      (if refinement (refinement/scheduled-graph refinement) expected))
        ;; Executable validation already validates this exact graph before checking its ABI,
        ;; scalar dependencies and artifacts. Do not immediately repeat the graph proof.
        emitted (executable/validate! graph)]
    (when-not (every? (comp artifact/kernel-artifact? :operation) (:nodes emitted))
      (fail! :emitted-parallel-equation-artifact
             "emitted equation graph requires only KernelArtifact nodes" {}))
    (when-not (graph/dataflow-equivalent? scheduled emitted)
      (fail! :emitted-parallel-equation-dataflow
             "target emission changed scheduled equation dataflow"
             {:scheduled (graph/dataflow-contract scheduled)
              :emitted (graph/dataflow-contract emitted)}))
    (when-not (every? true?
                      (map (fn [scheduled-node emitted-node generated-body]
                             (let [certificate (get-in emitted-node
                                                       [:operation :provenance
                                                        :scheduled-operation])]
                               (if (scheduled-body/scheduled-kernel-body? certificate)
                                 (do (scheduled-body/validate-against-math-policy! certificate policy)
                                     (cond
                                       (some #(segop/seg-scan? (:operation %)) (:nodes scheduled))
                                       (segscan-body/validate-against-node!
                                        certificate scheduled-node scheduled caller-options)
                                       (and (segop/seg-red? (:operation scheduled-node))
                                              (contains? #{:single :block-local :cross-block}
                                                         (:phase (:operation scheduled-node)))
                                              (empty? (segop/seg-space-segment-dims
                                                       (:space (:operation scheduled-node)))))
                                       (segred-body/validate-against-node!
                                        certificate scheduled-node scheduled algorithm body caller-options)
                                       :else
                                       (scheduled-body/validate-against-node!
                                        certificate scheduled-node scheduled))
                                     (when (and generated-body
                                                (not (semantic-fingerprint/equivalent?
                                                      generated-body certificate)))
                                       (fail! :emitted-mixed-matrix-artifact-refinement
                                              "mixed matrix artifact changed its exact generated stage body" {}))
                                     (when (and (swr/plan? algorithm)
                                                (not (semantic-fingerprint/equivalent?
                                                      certificate
                                                      (first (:operations (last (:equations body)))))))
                                       (fail! :emitted-reduction-artifact-refinement
                                              "emitted reduction artifact changed its exact schedule" {}))
                                     (scheduled-body/validate-artifact-projection!
                                      certificate (:operation emitted-node))
                                     true)
                                 (if generated-body
                                   (fail! :emitted-mixed-matrix-artifact-refinement
                                          "mixed matrix artifact requires its generated scheduled-body certificate" {})
                                   (= (:operation scheduled-node) certificate)))))
                           (:nodes scheduled) (:nodes emitted)
                           (or (:stage-bodies mixed) (repeat nil))))
      (fail! :emitted-parallel-equation-operation
             "target emission changed a scheduled equation operation certificate" {}))
    (doseq [[field value] [[:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :emitted-parallel-equation-description
               "emitted equation descriptions must be maps"
               {:field field :value value})))
    {:boundary emitted-equation :source-graph expected :mixed-reconstruction mixed}))

(defn validate!
  ([emitted-equation] (validate! emitted-equation {}))
  ([emitted-equation caller-options]
   (:boundary (validation-report emitted-equation caller-options))))

(defn- complete-write-domains-for-validated-boundary
  [{:keys [algorithm]}]
  {(get-in algorithm [:output :id])
   (swr/descriptor-launch-elements (:output algorithm))})

(defn complete-write-domains
  "Exact output domains proved by schedule rederivation, not ABI write permissions.
   The reference leaf covers the output grid. The subgroup leaf covers each head's components,
   then its sole head-zero/tile-zero owner writes the remaining row tail with disjoint lane strides.
   Both write empty destinations and invalid-edge results. Exact rederivation above is required;
   this is not a general must-write analysis for arbitrary KernelBody or future schedules."
  ([emitted] (complete-write-domains emitted {}))
  ([emitted caller-options]
  (when (swr/plan? (:algorithm emitted))
    (complete-write-domains-for-validated-boundary (validate! emitted caller-options)))))

(defn- contraction-write-domains-for-validated-boundary
  [boundary source-graph mixed caller-options]
  (if mixed
    (:complete-write-domains mixed)
    (let [{:keys [algorithm refinement graph]} boundary]
      (when (and (soac/program-form? algorithm)
                 (= 1 (count (soac/equations algorithm)))
                 (contains? '#{contract segmented-reduce}
                            (soac/operation-kind (first (soac/equations algorithm))))
                 (nil? refinement) (= 1 (count (:nodes graph))))
        (let [source source-graph
              node (first (:nodes source))
              certificate (get-in graph [:nodes 0 :operation :provenance :scheduled-operation])]
          (when (and (= 1 (count (:nodes source)))
                     (= :contraction (get-in node [:operation :phase]))
                     (= :float (get-in node [:operation :dtype]))
                     (scheduled-body/scheduled-kernel-body? certificate))
            (contraction-schedule/complete-write-domain algorithm node source certificate caller-options)))))))

(defn contraction-write-domains
  "Candidate-specific coverage for plain FP32 contraction results.

   This stronger query is for dispatch certification, not a replacement for ordinary SOAC
   initialization analysis. It reconstructs from the retained algorithm and source graph;
   Direct and full-K leading-batch mixed graphs share the generated matrix topology proof; split-K and other
   storage/schedule families still decline. Numerical and target admission remain separate."
  ([emitted] (contraction-write-domains emitted {}))
  ([emitted caller-options]
   (let [{:keys [boundary source-graph mixed-reconstruction]} (validation-report emitted caller-options)]
     (contraction-write-domains-for-validated-boundary boundary source-graph mixed-reconstruction caller-options))))

(defn- physical-results-for-validated-boundary
  [{:keys [algorithm body]}]
  (if (swr/plan? algorithm)
    (let [equation (last (:equations body))]
      (zipmap (:results equation)
              (map :destination (get-in equation [:attributes :result-storage]))))
    (soac/physical-result-map algorithm)))

(declare validate-with-result-contracts)

(defn ^:no-doc validate-with-physical-results
  "Validate an equation and return its exact boundary with the derived storage projection.
   This report is data, not authority to accept a later call without checking its bindings."
  ([emitted] (validate-with-physical-results emitted {}))
  ([emitted caller-options]
   (select-keys (validate-with-result-contracts emitted caller-options)
                [:boundary :physical-results :complete-write-domains])))

(defn ^:no-doc validate-with-result-contracts
  "Independently check one candidate and derive storage, writes and any mixed operational model.
   The returned data cannot authorize a later validation; no checked source graph is retained."
  ([emitted] (validate-with-result-contracts emitted {}))
  ([emitted caller-options]
  (let [{:keys [boundary source-graph mixed-reconstruction]} (validation-report emitted caller-options)
        algorithm (:algorithm boundary)]
    {:boundary boundary
     :physical-results (physical-results-for-validated-boundary boundary)
     :numerical-model (:numerical-model mixed-reconstruction)
     :complete-write-domains
     (if (swr/plan? algorithm)
       (complete-write-domains-for-validated-boundary boundary)
       (contraction-write-domains-for-validated-boundary boundary source-graph mixed-reconstruction caller-options))})))

(defn physical-results
  "Project logical results to physical storage from the retained, validated semantic equation."
  ([emitted] (physical-results emitted {}))
  ([emitted caller-options]
   (:physical-results (validate-with-physical-results emitted caller-options))))

(defn make
  ([algorithm body emitted]
   (make algorithm body emitted {}))
  ([algorithm body emitted metadata]
   (make algorithm body emitted metadata {}))
  ([algorithm body emitted {:keys [refinement provenance attributes]
                            :or {provenance {} attributes {}}} caller-options]
   (validate!
    (->EmittedParallelEquation algorithm body refinement emitted provenance attributes) caller-options)))
