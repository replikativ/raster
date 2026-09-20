(ns raster.compiler.passes.parallel.product-consumer-route-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.passes.parallel.product-consumer-region :as region]
            [raster.compiler.passes.parallel.product-consumer-region-test :as fixtures]
            [raster.compiler.passes.parallel.product-consumer-route :as route]))

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
