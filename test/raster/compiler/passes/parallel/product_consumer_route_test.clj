(ns raster.compiler.passes.parallel.product-consumer-route-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.passes.parallel.product-consumer-region :as region]
            [raster.compiler.passes.parallel.product-consumer-region-test :as fixtures]
            [raster.compiler.passes.parallel.product-consumer-route :as route]
            [raster.quant.ggml-kernels :as ggml-kernels]
            [raster.runtime.hardware :as runtime-hardware]))

(def ^:private subgroup-device
  {:device-type :gpu
   :execution {:preferred-subgroup-size 16
               :subgroup-sizes #{16 32}
               :max-workgroup-size 256}})

(defn- routed-region
  ([] (routed-region nil))
  ([target-device]
   (let [program (#'fixtures/scheduled-product-consumer)]
     (route/schedule
      (region/analyze program (#'fixtures/numerical-equations program))
      target-device))))

(defn- record-name [value]
  (some-> value class .getSimpleName))

(deftest exact-two-node-region-refines-to-one-cooperative-node
  (let [{scheduled-body :scheduled scheduled-graph :graph witness :refinement}
        (routed-region)
        source (:source witness)]
    (is (= 2 (count (:nodes source))))
    (is (= 1 (count (:nodes scheduled-graph))))
    (is (= ['partials] (mapv :id (:temporaries source))))
    (is (empty? (:temporaries scheduled-graph))
        "the graph-private intermediate is now workgroup-local storage")
    (is (= (graph/boundary-contract source)
           (graph/boundary-contract scheduled-graph)))
    (is (= (mapv :operation (:nodes source))
           (refinement/source-operations witness))
        "the certificate retains both exact semantic operations")
    (is (= (mapv :operation (:nodes source))
           (get-in scheduled-body [:source :operations])))
    (is (identical? scheduled-body (scheduled/validate! scheduled-body)))
    (is (identical? witness (refinement/validate! witness)))))

(deftest certified-region-projects-through-one-portable-body
  (let [routed (routed-region)]
    (doseq [[dialect target marker]
            [[:opencl-portable :opencl-c "__kernel void product_consumer"]
             [:cuda :cuda-c "extern \"C\" __global__ void product_consumer"]
             [:hip :hip-cpp "extern \"C\" __global__ void product_consumer"]]]
      (testing (name dialect)
        (let [{:keys [artifact emitted]} (route/emit "product_consumer" routed dialect)]
          (is (= target (:target artifact)))
          (is (str/includes? (:source artifact) marker))
          (is (= 1 (count (:nodes emitted))))
          (is (empty? (:temporaries emitted)))
          (is (= ['input 'weights 'output 'rows] (:arguments emitted)))
          (is (= artifact (first (executable/artifacts emitted))))
          (is (= (:refinement routed)
                 (get-in emitted [:attributes :scheduled-graph-refinement]))))))))

(deftest certified-additive-product-selects-an-all-register-subgroup-body
  (let [{:keys [scheduled refinement] :as routed} (routed-region subgroup-device)
        kernel-body (:body scheduled)
        operations (tree-seq coll? seq (:operations kernel-body))]
    (is (= :subgroup-product-ordered-consumer
           (get-in kernel-body [:schedule :strategy])))
    (is (= 16 (get-in kernel-body [:schedule :subgroup-size])))
    (is (empty? (:allocations kernel-body)))
    (is (= 1 (count (filter #(= "Collective" (record-name %)) operations))))
    (is (not-any? #(= "WorkgroupBarrier" (record-name %)) operations))
    (is (not-any? #(= 'partials (:buffer %)) operations))
    (is (= :subgroup-product-ordered-consumer
           (get-in refinement [:schedule :strategy])))
    (is (= :proved-local-offset
           (get-in refinement [:schedule :intermediate-substitution])))
    (doseq [[dialect reduction]
            [[:opencl-portable "sub_group_reduce_add"]
             [:cuda "__shfl_down_sync"]
             [:hip "__shfl_down"]]]
      (let [source (get-in (route/emit "product_consumer_subgroup" routed dialect)
                           [:artifact :source])]
        (is (str/includes? source reduction) (name dialect))
        (is (not (str/includes? source "barrier(")) (name dialect))
        (is (not (str/includes? source "__syncthreads")) (name dialect))))))

(deftest certified-product-valued-reduction-emits-one-collective-per-component
  (let [program (#'fixtures/scheduled-product-pair-consumer)
        routed (route/schedule
                (region/analyze program (#'fixtures/numerical-equations program))
                subgroup-device)
        kernel-body (get-in routed [:scheduled :body])
        operations (tree-seq coll? seq (:operations kernel-body))]
    (is (= :subgroup-product-ordered-consumer
           (get-in kernel-body [:schedule :strategy])))
    (is (empty? (:allocations kernel-body)))
    (is (= 2 (count (filter #(= "Collective" (record-name %)) operations))))
    (is (not-any? #(= "WorkgroupBarrier" (record-name %)) operations))))

(deftest subgroup-selection-falls-back-when-the-target-cannot-prove-a-legal-width
  (doseq [[label target reason]
          [[:missing nil :hardware-descriptor-unavailable]
           [:too-narrow (assoc-in subgroup-device [:execution :subgroup-sizes] #{4})
            :subgroup-width-unavailable]
           [:workgroup-limit (assoc-in subgroup-device [:execution :max-workgroup-size] 8)
            :subgroup-infeasible]]]
    (let [{:keys [scheduled plan]} (routed-region target)]
      (is (= :product-tree-ordered-consumer
             (get-in scheduled [:body :schedule :strategy]))
          (name label))
      (is (= reason (get-in plan [:physical-schedule :fallback-reason]))
          (name label)))))

(deftest equation-first-emission-selects-the-certified-region
  (let [scheduled (assoc (#'fixtures/scheduled-product-consumer)
                         :dialect :scheduled-parallel)
        {:keys [program kernels stats]}
        (c-family/emit-program scheduled {:target-dialect :opencl-portable})
        fused (peek (:equations program))
        operation (first (:operations fused))]
    (is (= [0 [:product-ordered-consumer [2 3]]]
           (mapv :id (:equations program))))
    (is (= [2 3] (get-in fused [:attributes :emitted-source-equations])))
    (is (emitted-equation/emitted-equation? operation))
    (is (= 2 (count (get-in operation [:refinement :source :nodes]))))
    (is (= 1 (count (get-in operation [:graph :nodes]))))
    (is (empty? (get-in operation [:graph :temporaries])))
    (is (= 1 (count kernels)))
    (is (= 1 (:product-consumer-regions-emitted stats)))
    (is (= {:kernel-body 1} (:emission-routes stats)))
    (is (= ['input 'weights 'output 'rows] (:arguments (first kernels))))))

(deftest real-q6-product-dot-emits-one-allocation-free-kernel
  (let [target :ze:q6-subgroup-compile-test
        _ (runtime-hardware/register-target-device!
           target {:type :ze
                   :name "Synthetic Intel subgroup product compile test"
                   :capabilities {:subgroup-sizes [16 32]
                                  :simd-width 16
                                  :max-workgroup-size 256
                                  :shared-local-memory 65536}})
        compilation
        (equation-first/compile #'ggml-kernels/qdot-q6-K-rows!
                                {:target target :dtype :float})
        fused (peek (get-in compilation [:emitted :equations]))
        emitted-equation (first (:operations fused))
        artifact (first (:kernels compilation))
        in 256
        out 1
        nrows 1
        plan (equation-first/lower
              compilation
              [(int-array 64) (float-array 1)
               (int-array 64) (float-array 1) (int-array 16)
               (float-array 1) (long in) (long out) (long nrows)])]
    (is (= 1 (get-in compilation [:stats :emission
                                  :product-consumer-regions-emitted])))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (not (some #{'partials} (get-in compilation [:emitted :inputs])))
        "the refined emitted program boundary no longer advertises eliminated storage")
    (is (= 1 (count (:kernels compilation))))
    (is (empty? (:temporaries artifact)))
    (is (= :subgroup-product-ordered-consumer
           (get-in artifact [:attributes :kernel-body :schedule :strategy])))
    (is (= 16 (count (filter #(= "Collective" (record-name %))
                             (tree-seq coll? seq
                                       (get-in artifact
                                               [:attributes :kernel-body :operations]))))))
    (is (= [3 5] (get-in fused [:attributes :emitted-source-equations])))
    (is (= '[wd wq wsc xd xq y in nrows out nb rstr_extent_3]
           (get-in emitted-equation [:graph :arguments]))
        "the public graph keeps source scalar order")
    (is (= '[wd wq wsc xd xq y in nb nrows out rstr_extent_3]
           (:arguments artifact))
        "the target body may independently order its scalar parameters")
    (is (link-plan/link-plan? plan))
    (is (= 6 (count (:nodes plan)))
        "the eliminated host allocation never becomes a resident LinkPlan node")
    (is (not-any? #(some #{'partials} (tree-seq coll? seq (:id %)))
                  (vals (:nodes plan))))))

(deftest real-q4-product-dot-reuses-the-product-valued-subgroup-schedule
  (let [target :ze:q4-subgroup-compile-test
        _ (runtime-hardware/register-target-device!
           target {:type :ze
                   :name "Synthetic Intel Q4 subgroup product compile test"
                   :capabilities {:subgroup-sizes [16 32]
                                  :simd-width 16
                                  :max-workgroup-size 256
                                  :shared-local-memory 65536}})
        compilation (equation-first/compile #'ggml-kernels/qdot-q4-K-rows!
                                            {:target target :dtype :float})
        artifact (first (:kernels compilation))
        operations (tree-seq coll? seq
                             (get-in artifact [:attributes :kernel-body :operations]))]
    (is (= 1 (get-in compilation [:stats :emission :product-consumer-regions-emitted])))
    (is (= 1 (count (:kernels compilation))))
    (is (empty? (:temporaries artifact)))
    (is (= :subgroup-product-ordered-consumer
           (get-in artifact [:attributes :kernel-body :schedule :strategy])))
    (is (= 10 (count (filter #(= "Collective" (record-name %)) operations))))
    (is (= 2 (count (re-seq #"xbs\[" (:source artifact))))
        "only the two live minimum-correction lanes retain q8 block-sum loads")
    (is (= 8 (count (re-seq #"wq\[" (:source artifact))))
        "only the eight live dot lanes retain packed-weight loads")
    (is (not-any? #(= "WorkgroupBarrier" (record-name %)) operations))))
