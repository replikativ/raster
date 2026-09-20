(ns raster.compiler.passes.parallel.product-consumer-body-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.passes.parallel.product-consumer-body :as candidate]
            [raster.compiler.passes.parallel.product-consumer-region-test :as fixtures]))

(defn- candidate-body []
  (let [program (#'fixtures/scheduled-product-consumer)]
    (:kernel-body
     (candidate/from-program program (#'fixtures/numerical-equations program)))))

(defn- pair-candidate-body []
  (let [program (#'fixtures/scheduled-product-pair-consumer)]
    (:kernel-body
     (candidate/from-program program (#'fixtures/numerical-equations program)))))

(defn- operation-kinds [kernel-body]
  (->> (:operations kernel-body)
       (tree-seq coll? seq)
       (keep #(some-> % class .getSimpleName))
       set))

(deftest product-and-ordered-consumer-share-one-validated-body
  (let [kernel-body (candidate-body)]
    (is (identical? kernel-body (body/validate! kernel-body)))
    (is (= :product-tree-ordered-consumer (get-in kernel-body [:schedule :strategy])))
    (is (= [8] (get-in kernel-body [:launch :workgroup-size])))
    (is (= ['input 'weights 'output 'rows] (mapv :id (:parameters kernel-body))))
    (is (= ['partials] (mapv :id (:allocations kernel-body)))
        "the graph-private intermediate becomes workgroup storage, not a public allocation")
    (is (every? (operation-kinds kernel-body)
                ["ForLoop" "WorkgroupBarrier" "ScalarLoad" "ScalarStore"]))
    (is (= :implementation-defined
           (get-in kernel-body [:attributes :inner-numerics :association])))
    (is (= :ordered (get-in kernel-body [:attributes :outer-numerics :association])))))

(deftest one-body-projects-to-portable-opencl-cuda-and-hip
  (let [kernel-body (candidate-body)]
    (doseq [[dialect marker]
            [[:opencl-portable ["__kernel void product_consumer" "barrier"]]
             [:cuda ["extern \"C\" __global__ void product_consumer" "__syncthreads"]]
             [:hip ["extern \"C\" __global__ void product_consumer" "__syncthreads"]]]]
      (testing (name dialect)
        (let [module (emitter/emit-scalar-module
                      "product_consumer" kernel-body {:target-dialect dialect})]
          (is (every? #(str/includes? (:source module) %) marker))
          (is (not (str/includes? (:source module) "partials,"))
              "private scratch must not leak into the public ABI"))))))

(deftest product-valued-workgroup-fallback-keeps-independent-typed-scratch
  (let [kernel-body (pair-candidate-body)]
    (is (identical? kernel-body (body/validate! kernel-body)))
    (is (= '[left-partials right-partials] (mapv :id (:allocations kernel-body))))
    (is (= [:int :int] (mapv :dtype (:allocations kernel-body))))
    (is (= 64 (get-in kernel-body [:launch :shared-memory-bytes])))))
