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
            [raster.quant.ggml-kernels :as ggml-kernels]))

(defn- routed-region []
  (let [program (#'fixtures/scheduled-product-consumer)]
    (route/schedule
     (region/analyze program (#'fixtures/numerical-equations program)))))

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
  (let [compilation
        (equation-first/compile #'ggml-kernels/qdot-q6-K-product-rows!
                                {:target :ze:0 :dtype :float})
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
