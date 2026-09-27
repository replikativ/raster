(ns raster.compiler.passes.parallel.indexed-weighted-reduction-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
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
         inputs (mapv (fn [{:keys [id dtype elements]}]
                        (graph/buffer id dtype elements :device :input))
                      (:operands plan))
         output-description (:output plan)
         output (graph/buffer (:id output-description) (:dtype output-description)
                              (:elements output-description) :device :output)
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
    (is (= 8 (count (:preconditions scheduled))))
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
                  {'n_entities Long/MAX_VALUE, 'n_edges 4, 'total_dim 2, 'n_heads 1,
                   'n_components 1, 'output_elements Long/MAX_VALUE})))))

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
