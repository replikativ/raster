(ns raster.compiler.backend.gpu.gemm-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as launch]))

(defn- emitted
  [variant]
  (gemm/emit-executable
   {:id (str "gemm-test-" (name variant))
    :a 'a :b 'b :c 'c :m 'm :n 'n :k 'k
    :variant variant :precision :f16-xmx
    :tile (hardware/derive-gemm-tile {})
    :fill-workgroups 32}))

(defn- arguments
  [m n k]
  [:a-buffer :b-buffer :c-buffer
   {:type :int :value m} {:type :int :value n} {:type :int :value k}])

(deftest hardware-aware-gemm-selection-is-checked-data
  (let [scheduled (emitted :nn)
        select #(dispatch/select-alternative scheduled (apply arguments %))]
    (is (= [:f32-scalar :xmx-direct :xmx-split-k]
           (mapv executable/strategy (:alternatives scheduled))))
    (testing "the matrix-instruction pitch gate is part of the selector, not a runtime binder"
      (is (= :f32-scalar (executable/strategy (select [32 4 4096]))))
      (is (= :f32-scalar (executable/strategy (select [32 128 4])))))
    (testing "a machine-filling shape stays direct"
      (is (= :xmx-direct (executable/strategy (select [512 512 512])))))
    (testing "a low-output-occupancy, deep-K shape selects a graph-private split"
      (is (= :xmx-split-k (executable/strategy (select [13 640 262144])))))))

(deftest split-k-storage-and-launch-use-the-selector-expression
  (let [scheduled (emitted :nn)
        runtime-arguments (arguments 13 640 262144)
        graph (dispatch/select-alternative scheduled runtime-arguments)
        {:keys [buffers scalar-values]} (executable/graph-bindings graph runtime-arguments)
        temporary-specs (graph-call/temporary-specs graph scalar-values)
        partial-spec (some (fn [[id spec]] (when (= :partials (last id)) spec))
                           temporary-specs)
        contract (some #(when (= :matrix-contract (get-in % [:operation :attributes :strategy]))
                          (:operation %))
                       (:nodes graph))]
    (is (= :xmx-split-k (executable/strategy graph)))
    (is (= #{'a 'b 'c} (set (keys buffers))))
    (is (= [:float (* 26 13 640) nil] partial-spec))
    (is (= [5 1 26]
           (:group-count
            (launch/realize (:launch contract)
                            #(graph-call/resolve-integer scalar-values %)))))
    (is (= 4 (count (:nodes graph))))))

(deftest layout-variants-are-graph-topology-not-runtime-conventions
  (doseq [[variant expected-node-count transpose-phase]
          [[:nn 3 nil] [:nt 4 :transpose-b] [:tn 4 :transpose-a]]]
    (let [graph (dispatch/alternative (emitted variant) :xmx-direct)
          phases (mapv #(get-in % [:operation :attributes :strategy]) (:nodes graph))]
      (is (= expected-node-count (count (:nodes graph))) (name variant))
      (when transpose-phase
        (is (some #{transpose-phase} phases) (name variant))))))

(deftest every-layout-schedule-realizes-to-kernel-calls
  (let [runtime-arguments (arguments 13 640 262144)]
    (doseq [variant [:nn :nt :tn]
            strategy [:xmx-direct :xmx-split-k]]
      (let [graph (dispatch/alternative (emitted variant) strategy)
            {:keys [buffers scalar-values]} (executable/graph-bindings graph runtime-arguments)
            temporary-specs (graph-call/temporary-specs graph scalar-values)
            temporary-buffers (zipmap (keys temporary-specs) (repeat :temporary-buffer))
            call (graph-call/make graph (merge buffers temporary-buffers) scalar-values)]
        (is (graph-call/kernel-graph-call? call) (str (name variant) "/" (name strategy)))
        (is (= (count (:nodes graph)) (count (:nodes call))))))))
