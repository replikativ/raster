(ns raster.compiler.passes.parallel.indexed-weighted-reduction-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.segop-opencl :as graph-emission]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.indexed-attention-recognize :as recognize]
            [raster.compiler.passes.parallel.indexed-weighted-reduction-body :as indexed-body]))

(defn- chain
  []
  '(let* [raw (raster.dl.array-ops/indexed-dot
               Q K dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          weights (raster.dl.array-ops/scale-clamp-exp
                   raw (raster.numeric// 1.0 (raster.numeric/sqrt dk))
                   5.0 (clojure.core/* n-edges n-heads))
          denominator (raster.dl.array-ops/scatter-add
                       weights dst n-nodes n-edges n-heads)
          weighted (raster.dl.array-ops/scatter-mul-add
                    weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          normalized (raster.dl.array-ops/segment-div
                      weighted denominator n-nodes emb-dim n-heads 1.0e-6)]
     normalized))

(defn- plan
  []
  (first (recognize/recognize (chain) :dtype :float :accumulator-dtype :float)))

(defn- source-graph
  ([plan] (source-graph plan :long))
  ([plan scalar-dtype]
   (let [fields (indexed-body/dynamic-fields plan)
         scalar-ids (vec (distinct (mapcat (comp launch/expression-references :value) fields)))
         inputs (mapv (fn [{:keys [id dtype] :as descriptor}]
                        (graph/buffer id dtype (swr/descriptor-launch-elements descriptor) :device :input))
                      (:operands plan))
         output-description (:output plan)
         output (graph/buffer (:id output-description) (:dtype output-description)
                              (swr/descriptor-launch-elements output-description) :device :output)
         uses (into (mapv #(graph/->ValueUse (:id %) :read) inputs)
                    [(graph/->ValueUse (:id output) :write)])
         node (graph/->ScheduledKernel
               [:indexed-reference (:id plan)] plan uses (set scalar-ids) [])]
     (graph/make
      {:inputs inputs
       :outputs [output]
       :scalars (mapv #(graph/scalar % scalar-dtype) scalar-ids)
       :nodes [node]
       :effects {:kind :segmented-weighted-reduction}
       :provenance {:pass :indexed-reference-certificate}}))))

(defn- reason
  [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception
         (:reason (ex-data exception)))))

(defn- subgroup-descriptor [width]
  {:device-type :gpu :vendor "Intel" :subgroup-size width
   :subgroup-sizes #{width} :max-workgroup-size 256})

(deftest subgroup-certificate-shares-reference-storage-and-scalar-interface
  (let [plan (plan)
        source (source-graph plan)
        node (first (:nodes source))
        reference (indexed-body/schedule-reference-for-node plan node source {})
        scheduled (indexed-body/schedule-score-reuse-for-node plan node source (subgroup-descriptor 16))]
    (is (= (:arguments reference) (:arguments scheduled)))
    (is (= (:scalar-bindings reference) (:scalar-bindings scheduled)))
    (is (= (get-in reference [:effects :uses]) (get-in scheduled [:effects :uses])))
    (is (= :reassociated (get-in scheduled [:numerics :mode])))
    (is (= :subgroup-dot-ordered-edge-accumulation (get-in scheduled [:numerics :policy])))
    (is (= :ordered-edge-list (get-in scheduled [:body :schedule :membership-traversal])))
    (is (= 16 (get-in scheduled [:legality :required-subgroup-size])))
    (is (= [16 1 1] (get-in (scheduled-body/realized-launch scheduled) [:workgroup-size])))
    (is (= [(launch/ceil-div 'dk 16) (launch/runtime-value 'n-heads)
            (launch/runtime-value 'n-nodes)]
           (get-in (scheduled-body/realized-launch scheduled) [:group-count])))
    (doseq [[dialect width] [[:opencl-intel 16] [:cuda 32] [:hip 64]]]
      (let [scheduled (indexed-body/schedule-score-reuse-for-node
                       plan node source (subgroup-descriptor width))
            emitted (target/emit-artifact "indexed_subgroup" scheduled dialect)]
        (is (= :reassociated (get-in emitted [:attributes :numerics :mode])))
        (is (empty? (:temporaries emitted)))
        (is (= emitted (scheduled-body/validate-artifact-projection! scheduled emitted)))))))

(deftest subgroup-certificate-bounds-all-three-group-coordinates
  (let [plan (plan)
        source (source-graph plan)
        scheduled (indexed-body/schedule-score-reuse-for-node
                   plan (first (:nodes source)) source (subgroup-descriptor 16))
        values {'n_entities 3 'n_edges 4 'total_dim 5 'n_heads 2
                'n_components 2 'output_elements 15}
        check #(precondition/check! (:preconditions scheduled) %)]
    (is (true? (check values)))
    (doseq [changed [{'n_entities 2147483649 'output_elements 10737418245}
                     {'n_entities 1 'n_heads 2147483649 'n_components 1
                      'total_dim 2147483649 'output_elements 2147483649}
                     {'n_entities 1 'n_heads 1 'n_components 34359738369
                      'total_dim 34359738369 'output_elements 34359738369}
                     {'n_edges -1} {'output_elements 14}]]
      (is (= :kernel-precondition-failed (reason #(check (merge values changed))))))
    (is (true? (check {'n_entities 1 'n_edges 0 'n_heads 1 'n_components 34359738368
                      'total_dim 34359738368 'output_elements 34359738368}))
        "the final legal group coordinate is INT_MAX; component arithmetic is int64")
    (is (thrown? ArithmeticException
                 (check (assoc values 'n_heads Long/MAX_VALUE))))))

(deftest subgroup-certificate-rejects-unproved-targets-and-interfaces
  (let [plan (plan)
        source (source-graph plan)
        node (first (:nodes source))
        descriptor (subgroup-descriptor 16)
        schedule #(indexed-body/schedule-score-reuse-for-node plan node source %)]
    (doseq [[changed expected]
            [[{:vendor "NVIDIA"} :score-reuse-requires-intel-subgroup-dialect]
             [{:device-type :cpu} :score-reuse-requires-gpu]
             [{:subgroup-sizes #{8}} :score-reuse-subgroup-width-unsupported]
             [{:max-workgroup-size 8} :score-reuse-invalid-subgroup-geometry]]]
      (is (= expected (reason #(schedule (merge descriptor changed))))))
    (let [source (source-graph plan :int)]
      (is (= :indexed-reference-public-scalar-dtype
             (reason #(indexed-body/schedule-score-reuse-for-node
                       plan (first (:nodes source)) source descriptor)))))
    (let [source (assoc-in source [:inputs 0 :elements] 1)]
      (is (= :indexed-reference-graph-storage
             (reason #(indexed-body/schedule-score-reuse-for-node
                       plan (first (:nodes source)) source descriptor)))))))

(deftest reference-and-subgroup-graphs-share-one-dispatch-interface
  (let [plan (plan)
        source (source-graph plan)
        node (first (:nodes source))
        descriptor (subgroup-descriptor 16)
        emit (fn [certificate]
               (graph-emission/generate-kernel-graph
                source :target-dialect :opencl-intel
                :scheduled-bodies {(:id node) certificate}))
        reference (emit (indexed-body/schedule-reference-for-node
                         plan node source descriptor))
        subgroup (emit (indexed-body/schedule-score-reuse-for-node
                        plan node source descriptor))
        reference-strategy :indexed-segmented-reduction-reference
        subgroup-strategy :indexed-segmented-reduction-subgroup-score-reuse
        selection (dispatch/make
                   {:id "indexed-score-reuse-probe"
                    :alternatives [reference subgroup]
                    :default-strategy reference-strategy
                    :selector {:kind :fixed-strategy :strategy subgroup-strategy
                               :fallback :none}})
        scalars {'n-nodes {:type :long :value 3}
                 'n-edges {:type :long :value 0}
                 'emb-dim {:type :long :value 5}
                 'n-heads {:type :long :value 2}
                 'dk {:type :long :value 2}}
        arguments (mapv (fn [slot value]
                          (if (= :scalar (:kind slot)) (get scalars value) value))
                        (executable/abi reference) (executable/arguments reference))]
    (is (= [reference-strategy subgroup-strategy]
           (mapv executable/strategy (:alternatives selection))))
    (is (= (executable/common-view reference) (executable/common-view subgroup)))
    (is (= :exact (get-in reference [:nodes 0 :operation :attributes :numerics :mode])))
    (is (= :reassociated (get-in subgroup [:nodes 0 :operation :attributes :numerics :mode])))
    (is (= subgroup (dispatch/select-alternative selection arguments)))
    (is (= subgroup
           (:executable
            (dispatch/admit-alternative
             selection arguments
             (fn [candidate]
               (graph-call/preflight! candidate scalars)
               [])))))
    (is (= :kernel-dispatch-inapplicable
           (reason #(dispatch/admit-alternative
                     selection arguments
                     (fn [candidate]
                       (if (= subgroup-strategy (executable/strategy candidate))
                         [{:reason :declined-for-probe}]
                         []))))))
    (is (= :kernel-graph-certified-strategy
           (reason #(graph-emission/generate-kernel-graph
                     (assoc-in source [:attributes :strategy] :different-strategy)
                     :target-dialect :opencl-intel
                     :scheduled-bodies
                     {(:id node) (indexed-body/schedule-reference-for-node
                                  plan node source descriptor)}))))))

(deftest reference-schedule-certifies-an-independent-semantic-graph
  (let [plan (plan)
        source (source-graph plan)
        node (first (:nodes source))
        scheduled (indexed-body/schedule-reference-for-node
                   plan node source
                   {:subgroup-size 16 :max-workgroup-size 256})]
    (is (= plan (:source scheduled)))
    (is (= '[Q K V dst src normalized
             n-nodes n-edges emb-dim n-heads dk]
           (vec (butlast (:arguments scheduled)))))
    (is (= (launch/product 'n-nodes 'emb-dim)
           (last (:arguments scheduled))))
    (is (= {'Q :read, 'K :read, 'V :read, 'dst :read, 'src :read,
            'normalized :write}
           (into {} (map (juxt :value :access)) (get-in scheduled [:effects :uses]))))
    (is (= :same-typed-ssa-evaluation-order (get-in scheduled [:numerics :policy])))
    (is (= :indexed-edge-list-reference (get-in scheduled [:legality :kind])))
    (is (= [16 1] (get-in (scheduled-body/realized-launch scheduled) [:workgroup-size])))
    (is (= [(launch/ceil-div (launch/runtime-value 'emb-dim) 16)
            (launch/runtime-value 'n-nodes)]
           (get-in (scheduled-body/realized-launch scheduled) [:group-count])))
    (is (= 12 (count (:preconditions scheduled))))
    (is (true? (precondition/check!
                (:preconditions scheduled)
                {'n_entities 3, 'n_edges 4, 'total_dim 5, 'n_heads 2,
                 'n_components 2, 'output_elements 15})))
    (is (= :kernel-precondition-failed
           (reason #(precondition/check!
                     (:preconditions scheduled)
                     {'n_entities 3, 'n_edges 4, 'total_dim 3, 'n_heads 2,
                      'n_components 2, 'output_elements 9}))))
    (is (thrown? ArithmeticException
                 (precondition/check!
                  (:preconditions scheduled)
                  {'n_entities 3, 'n_edges 4, 'total_dim 5, 'n_heads Long/MAX_VALUE,
                   'n_components 2, 'output_elements 15})))))

(deftest reference-launch-bounds-cover-masked-tail-coordinates
  (let [plan (plan)
        source (source-graph plan)
        scheduled (indexed-body/schedule-reference-for-node
                   plan (first (:nodes source)) source {:subgroup-size 3})
        values {'n_entities 1 'n_edges 0 'n_heads 1 'n_components 1}
        check #(precondition/check! (:preconditions scheduled)
                                    (assoc values 'total_dim % 'output_elements %))]
    (is (true? (check 2147483646)))
    (is (= :kernel-precondition-failed (reason #(check Integer/MAX_VALUE)))
        "the logical bound fits int, but its padded group would evaluate INT_MAX+1")
    (is (= :kernel-precondition-failed
           (reason #(precondition/check!
                     (:preconditions scheduled)
                     (assoc values 'n_entities 2147483649
                            'total_dim 1 'output_elements 2147483649)))))))

(deftest static-dimensions-use-explicit-private-int64-bindings
  (let [source-form (walk/postwalk-replace
                     {'n-nodes 3 'n-edges 4 'dk 2 'emb-dim 5 'n-heads 2} (chain))
        plan (first (recognize/recognize source-form :dtype :float :accumulator-dtype :float))
        source (source-graph plan)
        scheduled (indexed-body/schedule-reference-for-node
                   plan (first (:nodes source)) source {})]
    (is (empty? (:scalars source)))
    (is (= [3 4 5 2 2 15]
           (mapv #(launch/resolve-expression (constantly nil) %)
                 (drop 6 (:arguments scheduled)))))
    (is (every? #(= :long (:dtype %)) (:scalar-bindings scheduled)))
    (is (some? (target/emit-artifact "static_indexed_reference" scheduled :opencl-portable)))))

(deftest reference-certificate-survives-common-target-emission
  (let [plan (plan)
        source (source-graph plan)
        scheduled (indexed-body/schedule-reference-for-node
                   plan (first (:nodes source)) source {})]
    (doseq [[dialect module] [[:opencl-portable :opencl-c] [:cuda :cuda-c] [:hip :hip-cpp]]]
      (let [artifact (target/emit-artifact "indexed_reference" scheduled dialect)]
        (is (= module (:target artifact)))
        (is (= scheduled (get-in artifact [:provenance :scheduled-operation])))
        (is (= (:arguments scheduled) (:arguments artifact)))
        (is (= (:preconditions scheduled) (:preconditions artifact)))
        (is (= artifact (scheduled-body/validate-artifact-projection! scheduled artifact)))
        (is (= :scheduled-kernel-body-artifact-projection
               (reason #(scheduled-body/validate-artifact-projection!
                         scheduled (assoc-in artifact [:arguments 0] 'different-input)))))))))

(deftest semantic-reference-graph-emits-through-the-common-certified-route
  (let [plan (plan)
        source (source-graph plan)
        node (first (:nodes source))
        scheduled (indexed-body/schedule-reference-for-node plan node source {})
        certificates {(:id node) scheduled}
        values {'n-nodes 3 'n-edges 0 'emb-dim 5 'n-heads 2 'dk 2}
        runtime-values (into {} (map (fn [[id value]] [id {:type :long :value value}])) values)]
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (let [emitted (graph-emission/generate-kernel-graph
                     source :target-dialect dialect :scheduled-bodies certificates)]
        (is (graph/dataflow-equivalent? source emitted))
        (is (= scheduled (get-in emitted [:nodes 0 :operation :provenance :scheduled-operation])))
        (is (= emitted (graph-call/preflight! emitted runtime-values)))
        (is (= :kernel-precondition-failed
               (reason #(graph-call/preflight!
                         emitted (assoc-in runtime-values ['dk :value] 8)))))))
    (is (= :kernel-graph-scheduled-body-coverage
           (reason #(graph-emission/generate-kernel-graph source :scheduled-bodies {}))))))

(deftest reference-schedule-rejects-unproved-representations-and-graph-drift
  (let [plan (plan)
        source (source-graph plan)
        node (first (:nodes source))
        schedule #(indexed-body/schedule-reference-for-node
                   plan %1 %2 {:subgroup-size 16 :max-workgroup-size 256})]
    (testing "the current KernelBody does not silently widen int public dimensions"
      (let [int-source (source-graph plan :int)]
        (is (= :indexed-reference-public-scalar-dtype
               (reason #(schedule (first (:nodes int-source)) int-source))))))
    (testing "the certificate is for the exact plan stored in the graph node"
      (let [changed-plan (assoc-in plan [:provenance :semantic-op] :changed)
            changed-source (source-graph changed-plan)]
        (is (= :scheduled-kernel-body-source
               (reason #(schedule (first (:nodes changed-source)) changed-source))))))
    (testing "graph effects cannot weaken or change the body's storage contract"
      (let [changed-node (assoc-in node [:uses 0 :access] :write)
            changed-source (graph/validate! (assoc source :nodes [changed-node]))]
        (is (= :scheduled-kernel-body-node-effects
               (reason #(schedule changed-node changed-source))))))
    (testing "graph storage dtypes remain authoritative"
      (let [changed-input (assoc (first (:inputs source)) :dtype :double)
            changed-source (graph/validate!
                            (assoc source :inputs
                                   (assoc (:inputs source) 0 changed-input)))
            changed-node (first (:nodes changed-source))]
        (is (= :indexed-reference-graph-storage
               (reason #(schedule changed-node changed-source))))))))

(deftest reference-schedule-rejects-a-general-plan-with-different-algebra
  (let [original (plan)
        changed (assoc-in original [:weight :body] 'score)
        source (source-graph original)
        changed-node (assoc (first (:nodes source)) :operation changed)
        source (graph/validate! (assoc source :nodes [changed-node]))]
    (is (= :indexed-segmented-reduction-plan-unsupported
           (reason #(indexed-body/schedule-reference-for-node
                     changed changed-node source
                     {:subgroup-size 16 :max-workgroup-size 256}))))))
