(ns raster.compiler.backend.gpu.gemm-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.opencl-codegen :as opencl-codegen]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as launch]))

(defn- matrix-contract
  [graph]
  (some #(when (= :matrix-contract (get-in % [:operation :attributes :strategy]))
           (:operation %))
        (:nodes graph)))

(defn- emitted
  ([variant] (emitted variant {}))
  ([variant options]
   (gemm/emit-executable
    (merge
     {:id (str "gemm-test-" (name variant))
      :a 'a :b 'b :c 'c :m :m :n :n :k :k
      :variant variant :precision :mixed-f16-f32
      :tile (hardware/derive-gemm-tile {})
      :fill-workgroups 32}
     options))))

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
      (is (= :f32-scalar (executable/strategy (select [32 128 4]))))
      (is (= :f32-scalar (executable/strategy (select [32 126 4096]))))
      (is (= :f32-scalar (executable/strategy (select [32 128 4094])))))
    (testing "a machine-filling shape stays direct"
      (is (= :xmx-direct (executable/strategy (select [512 512 512])))))
    (testing "a low-output-occupancy, deep-K shape selects a graph-private split"
      (is (= :xmx-split-k (executable/strategy (select [13 640 262144])))))))

(deftest scalar-layout-fallbacks-are-portable-typed-contractions
  (doseq [variant [:nn :nt :tn :tt]]
    (let [graph (dispatch/alternative (emitted variant) :f32-scalar)
          operation (get-in graph [:nodes 0 :operation])
          kernel-body (artifact/attribute operation :kernel-body)
          runtime-arguments (arguments 3 2 4)
          {:keys [buffers scalar-values]} (executable/graph-bindings graph runtime-arguments)
          call (graph-call/make graph buffers scalar-values)]
      (is (body/kernel-body? kernel-body) (name variant))
      (is (= :contraction (artifact/attribute operation :semantic-op)))
      (is (= [256] (get-in kernel-body [:launch :workgroup-size])))
      (is (graph-call/kernel-graph-call? call)))))

(deftest explicit-split-factors-are-finite-schedule-alternatives
  (let [scheduled (emitted :nn {:split-factors [2 8 32]})
        by-strategy (into {} (map (juxt executable/strategy identity))
                          (:alternatives scheduled))]
    (is (= #{:f32-scalar :xmx-direct :xmx-split-k
             :xmx-split-k-2 :xmx-split-k-8 :xmx-split-k-32}
           (set (keys by-strategy))))
    (doseq [factor [2 8 32]]
      (let [strategy (gemm/split-factor-strategy factor)
            graph (get by-strategy strategy)]
        (is (= factor (get-in graph [:attributes :requested-splits])))
        (is (= factor (get-in scheduled
                              [:attributes :split-factor-schedules strategy])))))
    (is (= :xmx-split-k
           (executable/strategy
            (dispatch/select-alternative scheduled (arguments 13 640 262144))))
        "explicit tuning candidates do not replace the analytic default selector")))

(deftest split-k-storage-and-launch-use-the-selector-expression
  (let [scheduled (emitted :nn)
        runtime-arguments (arguments 13 640 262144)
        graph (dispatch/select-alternative scheduled runtime-arguments)
        {:keys [buffers scalar-values]} (executable/graph-bindings graph runtime-arguments)
        temporary-specs (graph-call/temporary-specs graph scalar-values)
        partial-spec (some (fn [[id spec]] (when (= :partials (last id)) spec))
                           temporary-specs)
        contract (matrix-contract graph)
        combine (some #(when (= :split-k-combine
                                (get-in % [:operation :attributes :strategy]))
                         (:operation %))
                      (:nodes graph))
        kernel-body (artifact/attribute contract :kernel-body)
        combine-body (artifact/attribute combine :kernel-body)
        outer-loop (first (filter #(instance? raster.compiler.ir.kernel_body.Loop %)
                                  (get-in kernel-body [:operations 0 :operations])))]
    (is (= :xmx-split-k (executable/strategy graph)))
    (is (= #{'a 'b 'c} (set (keys buffers))))
    (is (= [:float (* 26 13 640) nil] partial-spec))
    (is (= [5 1 26]
           (:group-count
            (launch/realize (:launch contract)
                            #(graph-call/resolve-integer scalar-values %)))))
    (is (= 4 (count (:nodes graph))))
    (is (body/kernel-body? kernel-body))
    (is (body/kernel-body? combine-body)
        "split-K combination is the portable typed contraction schedule")
    (is (= :contraction (artifact/attribute combine :semantic-op)))
    (is (= 1 (count (:views kernel-body))))
    (is (= [:splits :m :n] (:shape (first (filter #(= :result (:role %))
                                                  (:parameters kernel-body))))))
    (is (= 3 (count (get-in kernel-body [:launch :group-count]))))
    (is (not= 0 (:lower outer-loop)) "the K partition is an explicit loop bound")))

(deftest direct-xmx-graphs-carry-the-shared-scheduled-body
  (let [tile (hardware/derive-gemm-tile {})
        graph (dispatch/alternative (emitted :nn) :xmx-direct)
        contract (matrix-contract graph)
        kernel-body (artifact/attribute contract :kernel-body)
        dimensions (filter #(= :dimension (:role %)) (:parameters kernel-body))
        result (first (filter #(= :result (:role %)) (:parameters kernel-body)))
        oracle (apply opencl-codegen/emit-gemm-tiled (:kernel-name contract)
                      :c-dtype :float :prefetch (:num-stages tile)
                      (mapcat identity
                              (select-keys tile
                                           [:block-m :block-n :sg-m :sg-n :block-k :matrix])))]
    (is (body/kernel-body? kernel-body))
    (is (= [:m :n :k] (mapv :id dimensions))
        "the body retains graph ABI identities instead of a parallel M/N/K convention")
    (is (= :float (:dtype result)))
    (is (= (:source contract) oracle)
        "direct KernelBody lowering preserves the proven f32 GEMM source exactly")))

(deftest shared-direct-emission-does-not-call-the-legacy-template
  (with-redefs [opencl-codegen/emit-gemm-tiled
                (fn [& _]
                  (throw (ex-info "legacy template was called" {:reason :test/failure})))]
    (let [emitted (gemm/emit-scheduled-matrix-kernel
                   {:kernel-name "body_direct"
                    :a 'a :b 'b :c 'c :m 'm :n 'n :k 'k
                    :tile (hardware/derive-gemm-tile {})
                    :result-dtype :float})]
      (is (body/kernel-body? (:kernel-body emitted)))
      (is (re-find #"__global float\* restrict C" (:source emitted)))
      (is (= (get-in emitted [:kernel-body :launch :workgroup-size])
             (:workgroup-size emitted))))))

(deftest typed-epilogue-is-lowered-from-the-shared-body
  (with-redefs [opencl-codegen/emit-gemm-tiled
                (fn [& _]
                  (throw (ex-info "legacy template was called" {:reason :test/failure})))]
    (let [emitted (gemm/emit-scheduled-matrix-kernel
                   {:kernel-name "body_epilogue"
                    :a 'a :b 'b :c 'c :m 'm :n 'n :k 'k
                    :tile (hardware/derive-gemm-tile {})
                    :result-dtype :float
                    :epilogue {:acc 'acc
                               :expr '(raster.numeric/*
                                       (raster.numeric/+ acc (aget bias j)) scale)
                               :operands [{:sym 'bias
                                           :map (axis-map/of-axes [['j 'n]])
                                           :dtype :half}]
                               :scalars [{:sym 'scale :dtype :float}]}})
          kernel-body (:kernel-body emitted)
          epilogue-parameters (filterv #(= :epilogue (:role %)) (:parameters kernel-body))
          region (:value-region
                  (first (filter #(instance? raster.compiler.ir.kernel_body.TileStore %)
                                 (tree-seq coll? seq kernel-body))))]
      (is (= [['bias :input :half] ['scale :scalar :float]]
             (mapv (juxt :id :kind :dtype) epilogue-parameters)))
      (is (= ['acc 'bias 'scale] (:parameters region)))
      (is (re-find #"restrict bias, float scale" (:source emitted)))
      (is (re-find #"bias\[.*col.*\].*scale" (:source emitted))))))

(deftest unresolved-source-helper-calls-are-not-scalar-ir
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"typed portable scalar lowering"
       (gemm/emit-scheduled-matrix-kernel
        {:kernel-name "body_undefined_helper"
         :a 'a :b 'b :c 'c :m 'm :n 'n :k 'k
         :tile (hardware/derive-gemm-tile {})
         :result-dtype :float
         :epilogue {:acc 'acc :expr '(source_helper acc)}}))))

(deftest grid-z-matrix-emission-does-not-call-the-legacy-template
  (let [tile (hardware/derive-gemm-tile {})]
    (with-redefs [opencl-codegen/emit-gemm-tiled
                  (fn [& _]
                    (throw (ex-info "legacy template was called" {:reason :test/failure})))]
      (let [split (gemm/emit-scheduled-split-k-kernel
                   {:kernel-name "body_split"
                    :a 'a :b 'b :c 'partials :m 'm :n 'n :k 'k
                    :kc 'kc :splits 'splits :tile tile})
            batched (gemm/emit-scheduled-batched-matrix-kernel
                     {:kernel-name "body_batched"
                      :a 'a :b 'b :c 'c :m 'm :n 'n :k 'k
                      :batch 'batch :tile tile})]
        (is (body/kernel-body? (:kernel-body split)))
        (is (= 1 (count (get-in split [:kernel-body :views]))))
        (is (re-find #"int KC, int splits" (:source split)))
        (is (re-find #"int k_begin" (:source split)))
        (is (= 3 (count (get-in batched [:kernel-body :views]))))
        (is (re-find #"int batch" (:source batched)))
        (is (every? #(re-find (re-pattern (str % " \\+= ")) (:source batched))
                    ["A" "B" "C"]))))))

(deftest layout-variants-are-graph-topology-not-runtime-conventions
  (doseq [[variant expected-node-count transpose-phase]
          [[:nn 3 nil] [:nt 4 :transpose-b] [:tn 4 :transpose-a] [:tt 5 :transpose-a]]]
    (let [graph (dispatch/alternative (emitted variant) :xmx-direct)
          phases (mapv #(get-in % [:operation :attributes :strategy]) (:nodes graph))]
      (is (= expected-node-count (count (:nodes graph))) (name variant))
      (when transpose-phase
        (is (some #{transpose-phase} phases) (name variant))))))

(deftest every-layout-schedule-realizes-to-kernel-calls
  (let [runtime-arguments (arguments 13 640 262144)]
    (doseq [variant [:nn :nt :tn :tt]
            strategy [:xmx-direct :xmx-split-k]]
      (let [graph (dispatch/alternative (emitted variant) strategy)
            {:keys [buffers scalar-values]} (executable/graph-bindings graph runtime-arguments)
            temporary-specs (graph-call/temporary-specs graph scalar-values)
            temporary-buffers (zipmap (keys temporary-specs) (repeat :temporary-buffer))
            call (graph-call/make graph (merge buffers temporary-buffers) scalar-values)]
        (is (graph-call/kernel-graph-call? call) (str (name variant) "/" (name strategy)))
        (is (= (count (:nodes graph)) (count (:nodes call))))))))
