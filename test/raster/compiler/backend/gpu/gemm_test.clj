(ns raster.compiler.backend.gpu.gemm-test
  (:require [raster.compiler.reference.gemm-opencl :as gemm-oracle]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.opencl-codegen :as opencl-codegen]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.intel-block-io :as block-io]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.ir.layout-stage :as layout-stage]
            [raster.compiler.ir.matrix-stage :as matrix-stage]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]))

(deftest opencl-backend-aliases-share-mixed-matrix-admission
  (let [desc {:device-type :gpu :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}
              :subgroup-size 16 :execution {:subgroup-sizes #{16 32} :max-workgroup-size 1024}
              :grf-bytes-per-lane 256 :machine-lanes 8192 :shared-local-memory 131072}
        schedules (mapv #(gemm/mixed-dpas-schedule (assoc desc :backend %) nil)
                        [:ze :opencl :ocl])]
    (is (every? some? schedules))
    (is (apply = schedules))
    (doseq [unsupported [(assoc desc :backend :cuda)
                         (assoc desc :backend :hip)
                         (assoc-in (assoc desc :backend :ocl) [:matrix :family] :mma)
                         (assoc-in (assoc desc :backend :ocl) [:execution :subgroup-sizes] #{32})]]
      (is (nil? (gemm/mixed-dpas-schedule unsupported nil))))))

(deftest handwritten-gemm-entry-points-are-test-only
  (is (nil? (ns-resolve 'raster.compiler.backend.gpu.opencl-codegen 'emit-gemm-tiled)))
  (is (nil? (ns-resolve 'raster.compiler.backend.gpu.segop-opencl
                        'generate-dpas-contraction-kernel))))

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
    (is (= [:f32-scalar :xmx-direct :xmx-split-k :xmx-direct-lhs-tile-cast]
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

(deftest literal-dimensions-are-private-specialization-facts
  (let [scheduled (gemm/emit-executable
                   {:id "literal-gemm" :a 'a :b 'b :c 'c
                    :m 3 :n 2 :k 4 :variant :nn :precision :mixed-f16-f32
                    :tile (hardware/derive-gemm-tile {}) :fill-workgroups 32})
        graph (dispatch/select-alternative scheduled [:a-buffer :b-buffer :c-buffer])]
    (is (= '[a b c] (:arguments graph)))
    (is (= [] (:scalars graph)))
    (is (every? empty? (map :scalar-uses (:nodes graph))))))

(deftest explicit-split-factors-are-finite-schedule-alternatives
  (let [scheduled (emitted :nn {:split-factors [2 8 32]})
        by-strategy (into {} (map (juxt executable/strategy identity))
                          (:alternatives scheduled))]
    (is (= #{:f32-scalar :xmx-direct :xmx-split-k
             :xmx-split-k-2 :xmx-split-k-8 :xmx-split-k-32 :xmx-direct-lhs-tile-cast}
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
        scheduled-contract (artifact/attribute contract :scheduled-kernel-body)
        stage (:source scheduled-contract)
        combine-body (artifact/attribute combine :kernel-body)
        outer-loop (first (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
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
    (is (scheduled-body/scheduled-kernel-body? scheduled-contract))
    (is (matrix-stage/matrix-stage? stage))
    (is (= :split-k (get-in stage [:reduction :kind])))
    (is (= [(get-in stage [:reduction :partitions]) :m :n]
           (:result-shape stage)))
    (is (= (:arguments scheduled-contract) (:arguments contract)))
    (is (= (:effects scheduled-contract) (:effects contract)))
    (is (= (scheduled-body/realized-launch scheduled-contract) (:launch contract)))
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
        scheduled (artifact/attribute contract :scheduled-kernel-body)
        stage (:source scheduled)
        dimensions (filter #(= :dimension (:role %)) (:parameters kernel-body))
        result (first (filter #(= :result (:role %)) (:parameters kernel-body)))
        oracle (apply gemm-oracle/emit-gemm-tiled (:kernel-name contract)
                      :c-dtype :float :prefetch (:num-stages tile)
                      (mapcat identity
                              (select-keys tile
                                           [:block-m :block-n :sg-m :sg-n :block-k :matrix])))]
    (is (nil? (get-in graph [:attributes :scheduled-graph-refinement]))
        "the standalone compatibility API does not invent a semantic-source witness")
    (is (body/kernel-body? kernel-body))
    (is (scheduled-body/scheduled-kernel-body? scheduled))
    (is (matrix-stage/matrix-stage? stage))
    (is (= {:kind :full :range [0 :k]} (:reduction stage)))
    (is (= [:m :n] (:result-shape stage)))
    (is (= (:arguments scheduled) (:arguments contract)))
    (is (= (:effects scheduled) (:effects contract)))
    (is (= [:m :n :k] (mapv :id dimensions))
        "the body retains graph ABI identities instead of a parallel M/N/K convention")
    (is (= :float (:dtype result)))
    (is (= (:source contract) (-> oracle
                                  (str/replace "int k =" "long k =")
                                  (str/replace "int pk =" "long pk =")
                                  (str/replace "C[row*N+col]" "C[(long)row*(long)N+(long)col]")))
        "direct lowering preserves the oracle except for widened K and output arithmetic")))

(deftest production-xmx-epilogue-is-part-of-the-certified-stage
  (let [tile (hardware/derive-gemm-tile {})
        epilogue {:acc 'acc
                  :expr '(raster.numeric/*
                          (raster.numeric/+ acc (aget bias j)) scale)
                  :operands [{:sym 'bias :map (axis-map/of-axes [['j 'n]]) :dtype :half}]
                  :scalars [{:sym 'scale :dtype :float}]}
        {:keys [alternatives]}
        (gemm/emit-matrix-alternatives
         {:id "gemm-epilogue" :a 'a :b 'b :c 'c :m :m :n :n :k :k
          :variant :nn :precision :mixed-f16-f32 :tile tile :fill-workgroups 32
          :vector-width 4 :epilogue epilogue})
        graph (first alternatives)
        contract (matrix-contract graph)
        scheduled (artifact/attribute contract :scheduled-kernel-body)
        stage (:source scheduled)
        runtime-arguments [:a-buffer :b-buffer :c-buffer
                           {:type :int :value 16} {:type :int :value 32}
                           {:type :int :value 32} :bias-buffer
                           {:type :float :value 0.5}]
        {:keys [buffers scalar-values]} (executable/graph-bindings graph runtime-arguments)
        temporary-specs (graph-call/temporary-specs graph scalar-values)
        temporaries (into {} (map (fn [id] [id {:id [:temporary-buffer id] :alignment 64}]))
                          (keys temporary-specs))
        call (graph-call/make graph (merge buffers temporaries) scalar-values)]
    (is (= [:xmx-direct :xmx-direct-lhs-tile-cast] (mapv executable/strategy alternatives))
        "the result transform composes with input fusion but still disables split-K")
    (is (= epilogue (:epilogue stage)))
    (is (= '[bias scale]
           (mapv :name (filter #(= :epilogue (:role %)) (:abi contract)))))
    (is (= '[bias scale] (take-last 2 (:arguments contract))))
    (is (= (:effects scheduled) (:effects contract)))
    (is (= {:kind :typed-scalar-region
            :policy :same-typed-ssa-evaluation-order
            :input-dtype :float
            :result-dtype :float}
           (get-in scheduled [:numerics :result-transform])))
    (is (= #{:no-write-alias}
           (set (keep :aliasing (filter #(= :input (:kind %)) (:abi contract))))))
    (is (graph-call/kernel-graph-call? call))))

(deftest shared-direct-emission-does-not-call-the-legacy-template
  (with-redefs [gemm-oracle/emit-gemm-tiled
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
  (with-redefs [gemm-oracle/emit-gemm-tiled
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
    (with-redefs [gemm-oracle/emit-gemm-tiled
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
        (is (re-find #"long k_begin" (:source split)))
        (is (str/includes? (:source split)
                           "long k_begin = ((long)(k_slice) * (long)(KC));"))
        (is (str/includes? (:source split)
                           "long k_end = min((((long)(k_slice) * (long)(KC)) + (long)(KC)), (long)(K));")
            "emission uses widened typed bounds, not late casts of semantic metadata")
        (is (= 3 (count (get-in batched [:kernel-body :views]))))
        (is (re-find #"int batch" (:source batched)))
        (is (every? #(re-find (re-pattern (str % " \\+= ")) (:source batched))
                    ["A" "B" "C"]))))))

(deftest direct-split-matrix-requires-whole-positive-k-fragments
  (let [emitted (gemm/emit-scheduled-split-k-kernel
                 {:kernel-name "partition_contract" :a 'a :b 'b :c 'partials
                  :m 'm :n 'n :k 'k :kc 'chunk :splits 'partitions
                  :tile (hardware/derive-gemm-tile {})})
        conditions (:preconditions emitted)
        values {'m 8 'n 32 'k 64 'partitions 2}]
    (doseq [chunk [16 32 64]]
      (is (precondition/check! conditions (assoc values 'chunk chunk))))
    (doseq [chunk [-16 0 1 17 31]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"precondition failed"
                            (precondition/check! conditions (assoc values 'chunk chunk)))))))

(deftest partition-contract-binds-literal-limits-and-rejects-unknown-ranges
  (let [emitted (gemm/emit-scheduled-split-k-kernel
                 {:kernel-name "literal_partition" :a 'a :b 'b :c 'partials
                  :m 8 :n 32 :k 64 :kc 'chunk :splits 'partitions
                  :tile (hardware/derive-gemm-tile {})})
        kernel (:kernel-body emitted)
        {:keys [m n k]} (get-in kernel [:attributes :dimension-parameters])
        values {m 8 n 32 k 64 'chunk 32 'partitions 2}]
    (is (precondition/check! (:preconditions emitted) values))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"precondition failed"
                          (precondition/check! (:preconditions emitted) (assoc values k 32))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"canonical uniform grid-Z"
                          (block-io/body-requirements
                           (assoc-in kernel [:attributes :iteration-range :k] [1 64]))))))

(deftest layout-variants-are-graph-topology-not-runtime-conventions
  (doseq [[variant expected-node-count transpose-phase]
          [[:nn 3 nil] [:nt 4 :transpose-b] [:tn 4 :transpose-a] [:tt 5 :transpose-a]]]
    (let [graph (dispatch/alternative (emitted variant) :xmx-direct)
          phases (mapv #(get-in % [:operation :attributes :strategy]) (:nodes graph))]
      (is (= expected-node-count (count (:nodes graph))) (name variant))
      (when transpose-phase
        (is (some #{transpose-phase} phases) (name variant))))))

(deftest every-production-gemm-graph-stage-has-one-certified-target-projection
  (let [scheduled (emitted :tt)
        graphs (mapv #(dispatch/alternative scheduled %)
                     [:f32-scalar :xmx-direct :xmx-split-k])]
    (doseq [graph graphs
            node (:nodes graph)]
      (let [artifact (:operation node)
            refinement (artifact/attribute artifact :scheduled-kernel-body)]
        (is (scheduled-body/scheduled-kernel-body? refinement)
            (str (executable/strategy graph) " / " (:id node)))
        (is (= (:arguments refinement) (:arguments artifact)))
        (is (= (:effects refinement) (:effects artifact)))
        (is (= (scheduled-body/realized-launch refinement) (:launch artifact)))))
    (let [layout-sources
          (for [graph (rest graphs)
                node (:nodes graph)
                :let [source (get-in node [:operation :attributes
                                           :scheduled-kernel-body :source])]
                :when (layout-stage/layout-stage? source)]
            source)]
      (is (seq layout-sources))
      (is (every? #(contains? #{:cast :transpose} (:operation %)) layout-sources)))))

(deftest scheduled-stage-identities-include-the-candidate-strategy
  (let [graphs (mapv #(dispatch/alternative (emitted :tt {:split-factors [2 8]}) %)
                     [:f32-scalar :xmx-direct :xmx-split-k
                      :xmx-split-k-2 :xmx-split-k-8])
        stages (for [graph graphs
                     node (:nodes graph)
                     :let [source (get-in node [:operation :attributes
                                                :scheduled-kernel-body :source])]]
                 [(:id node) (:id source)])
        stage-ids (mapv second stages)]
    (is (every? (fn [[node-id stage-id]] (= node-id stage-id)) stages)
        "the certificate names the exact scheduled graph node")
    (is (= (count stage-ids) (count (distinct stage-ids)))
        "a stage identity denotes one exact alternative, not merely a phase name")))

(deftest batched-production-graph-stages-use-the-common-scheduled-body-projection
  (let [{graph :graph}
        (gemm/emit-batched-matrix-alternative
         {:id :batched-certificate
          :a 'a :b 'b :c 'c :batch 'batch :m 'm :n 'n :k 'k
          :variant :nn :tile (hardware/derive-gemm-tile {})
          :batching {:row true :col false}})]
    (doseq [node (:nodes graph)]
      (let [artifact (:operation node)
            certificate (artifact/attribute artifact :scheduled-kernel-body)]
        (is (scheduled-body/scheduled-kernel-body? certificate))
        (is (= (:arguments certificate) (:arguments artifact)))
        (is (= (:effects certificate) (:effects artifact)))
        (is (= (scheduled-body/realized-launch certificate) (:launch artifact)))))))

(deftest every-layout-schedule-realizes-to-kernel-calls
  (let [runtime-arguments (arguments 13 640 262144)]
    (doseq [variant [:nn :nt :tn :tt]
            strategy [:xmx-direct :xmx-split-k]]
      (let [graph (dispatch/alternative (emitted variant) strategy)
            {:keys [buffers scalar-values]} (executable/graph-bindings graph runtime-arguments)
            temporary-specs (graph-call/temporary-specs graph scalar-values)
            temporary-buffers (into {} (map (fn [id] [id {:id [:temporary-buffer id] :alignment 64}]))
                                    (keys temporary-specs))
            call (graph-call/make graph (merge buffers temporary-buffers) scalar-values)]
        (is (graph-call/kernel-graph-call? call) (str (name variant) "/" (name strategy)))
        (is (= (count (:nodes graph)) (count (:nodes call))))))))

(deftest batched-input-slices-preserve-block-io-alignment-before-allocation
  (doseq [row-batched? [true false]]
    (let [{:keys [graph selector]}
          (gemm/emit-batched-matrix-alternative
           {:id [:slice-alignment row-batched?]
            :a 'a :b 'b :c 'c :batch 'batch :m 'm :n 'n :k 'k
            :variant :nn :tile (hardware/derive-gemm-tile {})
            :batching {:row row-batched? :col false}})
          values {'batch {:type :int :value 2} 'm {:type :int :value 3}
                  'n {:type :int :value 40} 'k {:type :int :value 48}}
          scalar-values (into {} (map (fn [[id value]] [id (:value value)])) values)
          fallback? (some (fn [{:keys [expression op value]}]
                            (precondition/compare-value?
                             op (launch/resolve-expression scalar-values expression) value))
                          (:cases selector))]
      (is (= row-batched? (boolean fallback?)))
      (if row-batched?
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"precondition failed"
                              (graph-call/temporary-specs graph values)))
        (is (map? (graph-call/temporary-specs graph values)))))))

(deftest matrix-stages-derive-logical-widths-from-their-graph
  (doseq [variant [:nn :nt :tn :tt]
          widths [[:long :long :long] [:long :int :long] [:int :int :long]]]
    (let [interface (executable/common-view (dispatch/default-alternative (emitted variant)))
          by-argument (zipmap [:m :n :k] widths)
          interface (update interface :abi
                            (fn [slots]
                              (mapv (fn [slot argument]
                                      (if-let [dtype (get by-argument argument)]
                                        (assoc slot :dtype dtype :kernel-dtype dtype) slot))
                                    slots (:arguments interface))))
          {:keys [alternatives]}
          (gemm/emit-matrix-alternatives
           {:id [:graph-widths variant widths] :a 'a :b 'b :c 'c :m :m :n :n :k :k
            :variant variant :tile (hardware/derive-gemm-tile {}) :fill-workgroups 32
            :external-interface interface})
          values (zipmap [:m :n :k] (mapv (fn [dtype value] {:type dtype :value value})
                                         widths [3 32 1024]))]
      (doseq [graph alternatives]
        (is (map? (graph-call/temporary-specs graph values)))
        (doseq [node (:nodes graph)
                binding (get-in node [:operation :attributes :scheduled-kernel-body :scalar-bindings])]
          (is (= (launch/typed-expression-dtype (:value binding) by-argument) (:dtype binding)))
          (is (= (if (= (:kernel-dtype binding) (:dtype binding)) :identity :checked-range)
                 (:conversion binding))))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"physical ABI range"
                              (graph-call/temporary-specs graph
                                                         (assoc-in values [:m :value] 2147483648))))))))

(deftest batched-matrix-stages-retain-the-public-long-environment
  (let [spec {:id :long-batch :a 'a :b 'b :c 'c :batch 'batch :m 'm :n 'n :k 'k
              :variant :nn :tile (hardware/derive-gemm-tile {})
              :batching {:row true :col false}}
        base (:graph (gemm/emit-batched-matrix-alternative spec))
        interface (update (executable/common-view base) :abi
                          #(mapv (fn [slot] (if (= :scalar (:kind slot))
                                             (assoc slot :dtype :long :kernel-dtype :long) slot)) %))
        {graph :graph selector :selector}
        (gemm/emit-batched-matrix-alternative (assoc spec :external-interface interface))
        values (zipmap '[batch m n k] (mapv #(hash-map :type :long :value %) [2 8 32 32]))
        bindings (mapcat #(get-in % [:operation :attributes :scheduled-kernel-body :scalar-bindings])
                         (:nodes graph))]
    (is (every? #(= :long (:dtype %)) bindings))
    (is (= #{:identity :checked-range} (set (map :conversion bindings))))
    (is (map? (graph-call/temporary-specs graph values)))
    (doseq [[batch fallback?] [[2 false] [2147483648 true]]]
      (let [scalars (assoc (into {} (map (fn [[id value]] [id (:value value)])) values)
                           'batch batch)]
        (is (= fallback?
               (boolean
                (some (fn [{:keys [expression op value]}]
                        (precondition/compare-value?
                         op (launch/resolve-expression scalars expression) value))
                      (:cases selector)))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"physical ABI range"
                          (graph-call/temporary-specs graph (assoc-in values ['batch :value] 2147483648))))))

(deftest long-layout-and-combine-extents-do-not-narrow-shape-products
  (let [interface (update (executable/common-view (dispatch/default-alternative (emitted :nn)))
                          :abi #(mapv (fn [slot] (if (= :scalar (:kind slot))
                                                 (assoc slot :dtype :long :kernel-dtype :long) slot)) %))
        {:keys [alternatives]}
        (gemm/emit-matrix-alternatives
         {:id :wide-products :a 'a :b 'b :c 'c :m :m :n :n :k :k :variant :nn
          :tile (hardware/derive-gemm-tile {}) :fill-workgroups 32 :split-factors [2]
          :external-interface interface})
        direct (first alternatives)
        split (some #(when (= :xmx-split-k-2 (executable/strategy %)) %) alternatives)
        values (fn [m n k] (zipmap [:m :n :k] (mapv #(hash-map :type :long :value %) [m n k])))
        input-specs (graph-call/temporary-specs direct (values 65536 32 65536))
        output-specs (graph-call/temporary-specs split (values 65536 65536 64))
        combine (:operation (last (:nodes split)))]
    (is (some #{4294967296} (map second (vals input-specs))))
    (is (some #{8589934592} (map second (vals output-specs))))
    (is (every? #(= :long (:kernel-dtype %)) (filter #(= :scalar (:kind %)) (:abi combine))))
    (is (re-find #"long mn" (:source combine)))
    (is (every? #(= :identity (:conversion %))
                (get-in combine [:attributes :scheduled-kernel-body :scalar-bindings])))))
