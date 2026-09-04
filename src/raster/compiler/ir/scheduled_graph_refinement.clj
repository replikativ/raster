(ns raster.compiler.ir.scheduled-graph-refinement
  "A checked one-semantic-node to many-scheduled-kernels refinement.

   The witness preserves the graph's exact public boundary while allowing a schedule to introduce
   private temporaries and an ordered kernel DAG.  This IR establishes structural identity; the
   producing schedule pass remains responsible for proving its algorithm-specific numerical or
   effect law before constructing the witness.  Target emission subsequently remains a strict
   one-node-to-one-artifact projection of the refined graph."
  (:require [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.numerical-contract :as numerical-contract]))

(defrecord ScheduledGraphRefinement [source graph schedule numerics provenance attributes])

(declare fail!)

(defn scheduled-graph-refinement?
  "Recognize refinement witnesses across Typed Clojure child classloaders."
  [value]
  (and value
       (= "raster.compiler.ir.scheduled_graph_refinement.ScheduledGraphRefinement"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message
                  (assoc data :reason reason :ir :scheduled-graph-refinement))))

(defn validate!
  "Validate and return a scheduled graph-refinement witness."
  [refinement]
  (when-not (scheduled-graph-refinement? refinement)
    (fail! :scheduled-graph-refinement-type
           "expected a ScheduledGraphRefinement"
           {:actual (type refinement)}))
  (let [{source :source refined :graph
         :keys [schedule numerics provenance attributes]} refinement
        source (graph/validate! source)
        refined (graph/validate! refined)]
    (when-not (= 1 (count (:nodes source)))
      (fail! :scheduled-graph-refinement-source-cardinality
             "a graph refinement currently requires exactly one semantic source node"
             {:nodes (mapv :id (:nodes source))}))
    (when-not (seq (:nodes refined))
      (fail! :scheduled-graph-refinement-empty
             "a scheduled graph refinement cannot erase its semantic source node"
             {}))
    (when-not (and (some? (:scalars source)) (some? (:scalars refined)))
      (fail! :scheduled-graph-refinement-scalar-interface
             "a graph refinement requires an explicit target-neutral scalar interface"
             {:source-scalars (:scalars source) :refined-scalars (:scalars refined)}))
    (when-not (= (graph/boundary-contract source)
                 (graph/boundary-contract refined))
      (fail! :scheduled-graph-refinement-boundary
             "scheduled graph refinement changed the public graph boundary"
             {:source (graph/boundary-contract source)
              :refined (graph/boundary-contract refined)}))
    (when-not (and (map? schedule) (keyword? (:kind schedule)))
      (fail! :scheduled-graph-refinement-description
             "scheduled graph refinement requires an identified schedule"
             {:field :schedule :value schedule}))
    (numerical-contract/validate!
     numerics {:reason :scheduled-graph-refinement-numerics
               :ir :scheduled-graph-refinement})
    (doseq [[field value] [[:schedule schedule]
                           [:numerics numerics]
                           [:provenance provenance]
                           [:attributes attributes]]]
      (when-not (map? value)
        (fail! :scheduled-graph-refinement-description
               "scheduled graph refinement descriptions must be maps"
               {:field field :value value})))
    refinement))

(defn make
  "Construct a refinement from explicit source and refined graphs.

   `schedule` is deliberately explicit: a caller must name the verified scheduling decision that
   justifies the one-to-many rewrite rather than relying on matching boundaries alone."
  [{:keys [source graph schedule numerics provenance attributes]
    :or {provenance {} attributes {}}}]
  (validate!
   (->ScheduledGraphRefinement source graph schedule numerics provenance attributes)))

(defn source-operation
  "Return the exact semantic operation refined by this witness."
  [refinement]
  (-> refinement validate! :source :nodes first :operation))

(defn scheduled-graph
  "Return the checked many-kernel schedule graph."
  [refinement]
  (:graph (validate! refinement)))

(defn validate-against!
  "Require a refinement witness to name the exact rederived source schedule."
  [refinement expected-source]
  (let [refinement (validate! refinement)
        expected-source (graph/validate! expected-source)]
    (when-not (= (graph/schedule-contract expected-source)
                 (graph/schedule-contract (:source refinement)))
      (fail! :scheduled-graph-refinement-source
             "scheduled graph refinement does not refine the expected source schedule"
             {:expected (graph/schedule-contract expected-source)
              :source (graph/schedule-contract (:source refinement))}))
    refinement))
