(ns raster.compiler.passes.parallel.product-consumer-route
  "Certified graph schedule for a product tree followed by an ordered scalar consumer.

   The semantic source remains the exact two-node graph.  The refined graph has one compound
   schedule node whose private workgroup allocation replaces the source graph temporary."
  (:require [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.passes.parallel.product-consumer-body :as body]))

(defn- compound-source [plan]
  {:kind :product-ordered-consumer
   :equations (:equations plan)
   :operations [(:producer plan) (:consumer plan)]})

(defn- numerical-contract [plan]
  {:mode :reassociated
   :policy :declared-integral-tree-with-ordered-outer-fold
   :accumulators
   (mapv (fn [ordinal dtype]
           {:value [:inner-product ordinal]
            :dtype dtype :rounding :exact :policy :declared-product-tree})
         (range) (get-in plan [:numerics :inner :dtypes]))
   :ordered-consumer (get-in plan [:numerics :outer])})

(defn schedule
  "Build one ScheduledKernelBody and graph refinement from an admitted region plan."
  [plan]
  (let [{kernel-body :kernel-body arguments :arguments} (body/lower plan)
        source (compound-source plan)
        scheduled
        (scheduled/make
         {:source source :body kernel-body :arguments arguments
          :scalar-bindings (scheduled/derive-scalar-bindings kernel-body arguments)
          :effects {:kind :product-ordered-consumer
                    :uses (scheduled/derive-uses kernel-body arguments)}
          :legality {:kind :certified-product-consumer-region
                     :equations (:equations plan)
                     :axis-partition (:axes plan)
                     :intermediate-loads (:intermediate-loads plan)}
          :numerics (numerical-contract plan)
          :provenance {:source-dialect :segop
                       :source-operations (get-in plan [:provenance :source-operations])}
          :attributes {:candidate-only true}})
        source-graph (get-in plan [:source :graph])
        uses (mapv (fn [{:keys [value access]}] (graph/->ValueUse value access))
                   (get-in scheduled [:effects :uses]))
        scalar-uses (into #{} (map :value) (:scalar-bindings scheduled))
        refined
        (graph/make
         {:inputs (:inputs source-graph) :outputs (:outputs source-graph) :temporaries []
          :scalars (:scalars source-graph)
          :nodes [(graph/->ScheduledKernel
                   [:product-ordered-consumer (:equations plan)] source uses scalar-uses [])]
          :effects (:effects source-graph)
          :provenance {:source-dialect :segop :schedule :product-ordered-consumer}
          :attributes {:scheduled-kernel-body scheduled}})
        _ (scheduled/validate-against-node! scheduled (first (:nodes refined)) refined)
        witness
        (refinement/make
         {:source source-graph :graph refined
          :schedule {:kind :product-tree-ordered-consumer
                     :workgroup-size (:workgroup-size plan)
                     :axis-partition (:axes plan)}
          :numerics (numerical-contract plan)
          :provenance {:source-operations (get-in plan [:provenance :source-operations])}
          :attributes {:private-intermediate (:intermediate plan)}})]
    {:scheduled scheduled :graph refined :refinement witness :plan plan}))

(defn emit
  "Project a scheduled product-consumer region to one executable C-family graph."
  [kernel-name {:keys [scheduled graph refinement] :as routed} target-dialect]
  (let [artifact (target/emit-artifact kernel-name scheduled target-dialect)
        public-interface (graph/public-interface (:abi artifact) (:arguments artifact))
        pairs (mapv vector (:abi public-interface) (:arguments public-interface))
        pointers (filterv (fn [[slot _]] (not= :scalar (:kind slot))) pairs)
        scalar-by-value (into {} (map (fn [[slot value]] [value [slot value]]))
                              (filter (fn [[slot _]] (= :scalar (:kind slot))) pairs))
        ordered-scalars (mapv #(get scalar-by-value (:id %)) (:scalars graph))
        interface-pairs (into pointers ordered-scalars)
        emitted (-> graph
                    ;; A node body may choose a target-convenient scalar parameter order. The
                    ;; executable graph retains its independently checked public scalar order;
                    ;; node binding is by stable argument identity, never by this position.
                    (assoc :abi (mapv first interface-pairs)
                           :arguments (mapv second interface-pairs))
                    (assoc-in [:nodes 0 :operation] artifact)
                    (assoc-in [:attributes :scheduled-graph-refinement] refinement)
                    executable/validate!)]
    (when-not (graph/dataflow-equivalent? graph emitted)
      (throw (ex-info "product-consumer target emission changed scheduled dataflow"
                      {:reason :product-consumer-emission-dataflow})))
    (scheduled/validate-artifact-projection! scheduled artifact)
    (assoc routed :emitted emitted :artifact artifact)))
