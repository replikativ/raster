(ns raster.compiler.backend.gpu.attention
  "FP16-KV leaves for logical attention over dense or CSR paged KV routes.

   The direct one-work-item/component leaf remains the executable semantic oracle.  A separately
   validated SegmentedWeightedReductionSchedule either emits one subgroup/query-head or partitions
   bounded membership into partial online states and a deterministic merge graph. Both schedules
  share each QK score across lane-strided value accumulators without changing the semantic plan,
  ordered external ABI, storage ownership or graph effects."
  (:require [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-opencl :as body-opencl]
            [raster.compiler.backend.gpu.target :as gpu-target]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as swr-schedule]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-body :as swr-body]))

(defn- attention-problem
  [plan]
  ;; Provenance is diagnostic, not a legality token. The leaf is selected from the plan's algebra,
  ;; storage and membership descriptors, then independently proves the complete routed-row shape
  ;; below. A non-attention frontend may therefore construct the same legal reduction without
  ;; forging an :attention origin label.
  (attention/validate! (:source-operation (swr/validate! plan))))

(defn- reference-plan!
  [plan]
  (let [plan (swr/validate! plan)
        {:keys [id query route output q-heads kv-heads qk-head-dim value-head-dim
                q-dtype k-dtype v-dtype output-dtype accumulator-dtype scale visibility]
         :as problem} (attention-problem plan)
        packed? (attention/dense-packed-route? route)
        expected-segments [{:name :query-token :extent (:total-tokens query)}
                           {:name :query-head :extent q-heads}]
        expected-score {:kind :dot
                        :axis {:name :qk-component :extent qk-head-dim}
                        :head-map {:kind :grouped-query
                                   :query-heads q-heads :kv-heads kv-heads}
                        :left {:kind :packed-query :buffer (:values query) :dtype q-dtype}
                        :right {:kind (if packed? :packed-key :routed-key)
                                :buffer (:k-pages problem) :dtype k-dtype}}
        expected-membership {:kind :logical-attention-visibility
                             :visibility-kind (attention/visibility-kind visibility)
                             :position-filter (into {} (attention/position-filter visibility))
                             :duplicate-policy (when (attention/csr-visibility? visibility)
                                                 (:duplicate-policy visibility))
                             :buffers (attention/visibility-buffer-ids visibility)}]
    (when-not (and (swr/online-softmax-algebra? plan)
                   (= [:segmented-weighted-reduction id] (:id plan))
                   (= expected-segments (:segment-axes plan))
                   (= expected-membership (:membership plan))
                   (= (if packed? :dense-packed-kv :routed-paged-kv)
                      (get-in plan [:storage :kind]))
                   (= (attention/route-kind route) (get-in plan [:storage :route-kind]))
                   (= route (get-in plan [:storage :route]))
                   (= (attention/route-buffer-ids route) (get-in plan [:storage :buffers]))
                   (= expected-score (select-keys (:score plan)
                                                  [:kind :axis :head-map :left :right]))
                   (= (list 'raster.numeric/* 'dot (double scale))
                      (get-in plan [:score :finalize :body]))
                   (= {:kind (if packed? :packed-value :routed-value)
                       :buffer (:v-pages problem)
                       :dtype v-dtype :components value-head-dim}
                      (:value plan))
                   (= accumulator-dtype (:accumulator-dtype plan))
                   (= (attention/ordered-input-buffer-ids problem)
                      (swr/ordered-input-ids plan))
                   (= {:id output :dtype output-dtype
                       :shape [(:total-tokens query) q-heads value-head-dim]
                       :elements (* (:total-tokens query) q-heads value-head-dim)}
                      (:output plan)))
      (throw (ex-info "FP16 reference leaf cannot preserve this reduction plan exactly"
                      {:reason :attention-reference-plan-unsupported
                       :operation-id id :plan-id (:id plan)})))
    plan))

(defn reference-workgroup-x
  "Choose the reference leaf's x workgroup from output width and hardware resources."
  [plan desc]
  (let [value-head-dim (:value-head-dim (attention-problem plan))
        subgroup (long (or (:subgroup-size desc) 16))
        maximum (long (or (:max-workgroup-size desc) 256))]
    (long (max 1 (min value-head-dim subgroup maximum)))))

(defn- kernel-name
  ([problem] (kernel-name problem :reference))
  ([problem schedule-identity]
   (let [identity (assoc (select-keys problem
                                      [:batch-size :q-heads :kv-heads :qk-head-dim :value-head-dim
                                       :page-size :physical-pages :scale :k-format :v-format
                                       :k-layout :v-layout :q-dtype :k-dtype :v-dtype
                                       :output-dtype])
                         :schedule schedule-identity)
         visibility (:visibility problem)
         identity (assoc identity
                         :route-kind (attention/route-kind (:route problem))
                         :route-shape (select-keys (:route problem)
                                                   [:pages-per-sequence
                                                    :page-index-capacity :total-tokens])
                         :visibility-kind (attention/visibility-kind visibility)
                         :visibility-shape
                         (cond-> {:position-filter (into {} (attention/position-filter visibility))}
                           (attention/csr-visibility? visibility)
                           (assoc :key-index-capacity (:key-index-capacity visibility)))
                         :total-query-tokens (get-in problem [:query :total-tokens]))]
     (format (if (attention/dense-packed-route? (:route problem))
               "raster_attention_dense_packed_%08x"
               "raster_attention_fp16_%08x")
             (bit-and 0xffffffff (long (hash identity)))))))

(defn- cooperative-plan!
  [plan schedule]
  (let [plan (reference-plan! plan)
        schedule (swr-schedule/validate! schedule)
        problem (attention-problem plan)
        components (:value-head-dim problem)]
    (when-not (and (= components (get-in schedule [:value-mapping :components]))
                   (= (get-in plan [:score :axis :name])
                      (get-in schedule [:score-reduction :axis]))
                   (= (:accumulator-dtype plan)
                      (get-in schedule [:numerical-mode :score-accumulate]))
                   (= (:accumulator-dtype plan)
                      (get-in schedule [:numerical-mode :state-accumulate]))
                   (= (if (attention/dense-packed-route? (:route problem))
                        :dense-packed-kv :routed-paged-kv)
                      (get-in schedule [:attributes :storage-kind]))
                   (= (attention/route-kind (:route problem))
                      (get-in schedule [:attributes :route-kind]))
                   (= (attention/visibility-kind (:visibility problem))
                      (get-in schedule [:attributes :visibility-kind]))
                   (= (if (attention/csr-visibility? (:visibility problem))
                        :csr-row
                        :contiguous-interval)
                      (:membership-traversal schedule)))
      (throw (ex-info "cooperative attention schedule does not describe this reduction plan"
                      {:reason :attention-cooperative-schedule-plan-mismatch
                       :schedule schedule :plan-id (:id plan)})))
    [plan schedule problem]))

(defn- ordered-inputs
  [plan]
  (swr/ordered-input-ids plan))

(defn- ordered-abi
  [problem]
  (let [{:keys [query k-pages v-pages route visibility output q-dtype k-dtype v-dtype
                output-dtype]} problem
        packed? (attention/dense-packed-route? route)
        common [(kabi/slot (:values query) :input q-dtype :c-name "q" :role :query)
                (kabi/slot (:row-offsets query) :input :int
                           :c-name "q_row_offsets" :role :query-rows)
                (kabi/slot (:positions query) :input :int
                           :c-name "q_positions" :role :query-positions)
                (kabi/slot k-pages :input k-dtype
                           :c-name (if packed? "k_values" "k_pages") :role :key-cache)
                (kabi/slot v-pages :input v-dtype
                           :c-name (if packed? "v_values" "v_pages") :role :value-cache)]
        route-slots
        (cond
          packed?
          [(kabi/slot (:row-offsets route) :input :int
                      :c-name "kv_row_offsets" :role :kv-row-offsets)
           (kabi/slot (:start-positions route) :input :int
                      :c-name "kv_start_positions" :role :kv-start-positions)]

          (attention/dense-paged-route? route)
          [(kabi/slot (:page-table route) :input :int
                      :c-name "page_table" :role :page-routing)
           (kabi/slot (:lengths route) :input :int
                      :c-name "kv_lengths" :role :kv-lengths)
           (kabi/slot (:start-positions route) :input :int
                      :c-name "kv_start_positions" :role :kv-start-positions)]
          :else
          [(kabi/slot (:page-offsets route) :input :int
                      :c-name "page_offsets" :role :page-row-offsets)
           (kabi/slot (:page-indices route) :input :int
                      :c-name "page_indices" :role :page-routing)
           (kabi/slot (:last-page-lengths route) :input :int
                      :c-name "last_page_lengths" :role :last-page-lengths)
           (kabi/slot (:start-positions route) :input :int
                      :c-name "kv_start_positions" :role :kv-start-positions)])
        visibility-slots
        (when (attention/csr-visibility? visibility)
          [(kabi/slot (:row-offsets visibility) :input :int
                      :c-name "attention_row_offsets" :role :attention-row-offsets)
           (kabi/slot (:key-indices visibility) :input :int
                      :c-name "attention_key_indices" :role :attention-key-indices)])]
    (kabi/validate!
     (mapv #(if (= :input (:kind %))
              (assoc % :aliasing :no-write-alias)
              %)
           (conj (into (into common route-slots) visibility-slots)
                 (kabi/slot output :output output-dtype
                            :c-name "output" :role :result))))))

(defn emit-fp16-reference
  "Emit the portable correctness schedule from target-neutral KernelBody.

   This deliberately retains the slower one-work-item-per-output-component mapping, but no target
   source template. It is therefore the common semantic baseline for OpenCL, CUDA and HIP."
  [plan desc]
  (let [plan (reference-plan! plan)
        {:keys [output route] :as problem}
        (attention-problem plan)
        name (kernel-name problem)
        workgroup-x (reference-workgroup-x plan desc)
        inputs (ordered-inputs plan)
        arguments (conj inputs output)
        target-dialect (or (gpu-target/kernel-body-c-dialect desc) :opencl-portable)
        dialect (c-dialect/resolve! target-dialect)
        kernel-body (swr-body/lower-routed-paged-reference plan workgroup-x)
        base-abi (ordered-abi problem)
        parameter-names (into {} (map (juxt :name :c-name)) base-abi)
        abi (body-abi/project-contracts base-abi kernel-body)]
    (kart/make
     {:kernel-name name
      :target (c-dialect/target dialect)
      :source (body-opencl/emit-scalar-kernel
               name kernel-body
               {:parameter-names parameter-names :target-dialect target-dialect})
      :abi abi
      :arguments arguments
      :launch (:launch kernel-body)
      :effects {:kind :attention :reads inputs :writes [output]}
      :provenance {:operation-id (:id problem) :semantic-op :attention
                   :algebra-plan-id (:id plan)
                   :lowering (if (attention/dense-packed-route? route)
                               :dense-packed-reference :fp16-reference)}
      :attributes {:strategy (if (attention/dense-packed-route? route)
                               :dense-packed-reference :fp16-reference)
                   :optimization-tier :reference
                   :algebra :segmented-weighted-reduction
                   :algebra-key (swr/algebra-key plan)
                   :storage-dtype (when (= (:k-dtype problem) (:v-dtype problem))
                                    (:k-dtype problem))
                   :storage-dtypes [(:k-dtype problem) (:v-dtype problem)]
                   :q-dtype (:q-dtype problem)
                   :output-dtype (:output-dtype problem) :accumulator-dtype :float
                   :route-kind (attention/route-kind route)
                   :visibility-kind (attention/visibility-kind (:visibility problem))
                   :k-layout (:k-layout problem) :v-layout (:v-layout problem)
                   :visibility (:visibility problem)
                   :layout (attention/layouts problem)
                   :kernel-body kernel-body
                   :target-dialect target-dialect
                   :materialized-intermediates []
                   :complexity :quadratic-in-qk-head-dim}})))

(defn emit-fp16-cooperative
  "Emit one subgroup per query segment for dense-packed or routed-paged K/V attention.

   The artifact preserves the reference leaf's complete ordered ABI and logical effects.  Only
   its target-neutral SegmentedWeightedReductionSchedule, launch mapping and target body differ.
   The scheduled body is independently verified before target-only lowering selects a C-family
   spelling. The two-argument form retains the production Intel OpenCL route."
  ([plan schedule]
   (emit-fp16-cooperative plan schedule :opencl-intel))
  ([plan schedule target-dialect]
   (let [dialect (c-dialect/resolve! target-dialect)
         [plan schedule {:keys [output route] :as problem}]
         (cooperative-plan! plan schedule)
         name (kernel-name problem schedule)
         inputs (ordered-inputs plan)
         arguments (conj inputs output)
         kernel-body (swr-body/lower-routed-paged plan schedule)
         base-abi (ordered-abi problem)
         parameter-names (into {} (map (juxt :name :c-name)) base-abi)
         abi (body-abi/project-contracts base-abi kernel-body)]
     (kart/make
      {:kernel-name name
       :target (c-dialect/target dialect)
       :source (body-opencl/emit-scalar-kernel
                name kernel-body
                {:parameter-names parameter-names :target-dialect target-dialect})
       :abi abi
       :arguments arguments
       :launch (:launch kernel-body)
       :effects {:kind :attention :reads inputs :writes [output]}
       :provenance {:operation-id (:id problem) :semantic-op :attention
                    :algebra-plan-id (:id plan)
                    :lowering :subgroup-online-score-reuse}
       :attributes {:strategy (if (attention/dense-packed-route? route)
                                :dense-packed-subgroup-online-score-reuse
                                :routed-paged-subgroup-online-score-reuse)
                    :optimization-tier :subgroup
                    :algebra :segmented-weighted-reduction
                    :algebra-key (swr/algebra-key plan)
                    :segmented-weighted-reduction-schedule schedule
                    :storage-dtype (when (= (:k-dtype problem) (:v-dtype problem))
                                     (:k-dtype problem))
                    :storage-dtypes [(:k-dtype problem) (:v-dtype problem)]
                    :q-dtype (:q-dtype problem)
                    :output-dtype (:output-dtype problem) :accumulator-dtype :float
                    :route-kind (attention/route-kind route)
                    :visibility-kind (attention/visibility-kind (:visibility problem))
                    :k-layout (:k-layout problem) :v-layout (:v-layout problem)
                    :visibility (:visibility problem)
                    :layout (attention/layouts problem)
                    :kernel-body kernel-body
                    :target-dialect target-dialect
                    :target-collective-association
                    (c-dialect/collective-association dialect)
                    :materialized-intermediates []
                    :complexity :query-head-token-dot-plus-value}}))))

(defn emit-fp16-pipelined
  "Emit the double-buffered membership-row schedule for routed FP16 K/V reduction.

   Scheduling and KernelBody lowering remain target-neutral.  The target dialect only chooses
   the verified asynchronous-copy spelling (including capability-gated CUDA cp.async)."
  ([plan schedule]
   (emit-fp16-pipelined plan schedule :opencl-intel {}))
  ([plan schedule target-dialect]
   (emit-fp16-pipelined plan schedule target-dialect {}))
  ([plan schedule target-dialect target-features]
   (let [dialect (c-dialect/resolve! target-dialect)
         [plan schedule {:keys [output route] :as problem}]
         (cooperative-plan! plan schedule)
         layout-swizzle (get-in schedule [:staging :layout-swizzle])
         strategy (if (= :identity layout-swizzle)
                    :routed-paged-subgroup-online-pipelined-history
                    (keyword (str "routed-paged-subgroup-online-pipelined-history-"
                                  (name layout-swizzle))))
         name (kernel-name problem schedule)
         inputs (ordered-inputs plan)
         arguments (conj inputs output)
         kernel-body (swr-body/lower-routed-paged-pipelined plan schedule)
         base-abi (ordered-abi problem)
         parameter-names (into {} (map (juxt :name :c-name)) base-abi)
         abi (body-abi/project-contracts base-abi kernel-body)]
     (kart/make
      {:kernel-name name
       :target (c-dialect/target dialect)
       :source (body-opencl/emit-scalar-kernel
                name kernel-body
                {:parameter-names parameter-names
                 :target-dialect target-dialect
                 :target-features target-features})
       :abi abi
       :arguments arguments
       :launch (:launch kernel-body)
       :effects {:kind :attention :reads inputs :writes [output]}
       :provenance {:operation-id (:id problem) :semantic-op :attention
                    :algebra-plan-id (:id plan)
                    :lowering :subgroup-online-pipelined-history}
       :attributes {:strategy strategy
                    :optimization-tier :subgroup-pipelined
                    :algebra :segmented-weighted-reduction
                    :algebra-key (swr/algebra-key plan)
                    :segmented-weighted-reduction-schedule schedule
                    :storage-dtype :half :q-dtype (:q-dtype problem)
                    :output-dtype (:output-dtype problem) :accumulator-dtype :float
                    :route-kind (attention/route-kind route)
                    :visibility-kind (attention/visibility-kind (:visibility problem))
                    :k-layout (:k-layout problem) :v-layout (:v-layout problem)
                    :visibility (:visibility problem)
                    :layout (attention/layouts problem)
                    :kernel-body kernel-body
                    :target-dialect target-dialect
                    :target-features target-features
                    :target-collective-association
                    (c-dialect/collective-association dialect)
                    :materialized-intermediates []
                    :complexity :pipelined-query-head-token-dot-plus-value}}))))

(def ^:private partial-c-names
  {:partial-valid "partial_valid"
   :partial-maximum "partial_maximum"
   :partial-denominator "partial_denominator"
   :partial-weighted-values "partial_weighted_values"})

(defn- private-state-slots
  [kind partial-specs]
  (mapv (fn [{:keys [id dtype role]}]
          (kabi/slot id kind dtype :c-name (get partial-c-names role) :role role))
        partial-specs))

(defn- partial-abi
  [problem partial-specs kernel-body]
  (let [external-inputs (vec (butlast (ordered-abi problem)))
        private-outputs (private-state-slots :output partial-specs)]
    (body-abi/project-contracts (into external-inputs private-outputs) kernel-body)))

(defn- merge-abi
  [problem partial-specs kernel-body]
  (let [private-inputs (private-state-slots :input partial-specs)
        output-slot (peek (ordered-abi problem))]
    (body-abi/project-contracts (conj private-inputs output-slot) kernel-body)))

(defn- emitted-body-artifact
  [problem plan scheduled phase kernel-body abi arguments reads writes target-dialect]
  (let [dialect (c-dialect/resolve! target-dialect)
        entry-name (str (kernel-name problem [scheduled phase]) "_" (name phase))
        parameter-names (into {} (map (juxt :name :c-name)) abi)]
    (kart/make
     {:kernel-name entry-name
      :target (c-dialect/target dialect)
      :source (body-opencl/emit-scalar-kernel
               entry-name kernel-body
               {:parameter-names parameter-names :target-dialect target-dialect})
      :abi abi
      :arguments arguments
      :launch (:launch kernel-body)
      :effects {:kind :segmented-weighted-reduction-phase
                :phase phase :reads reads :writes writes}
      :provenance {:operation-id (:id problem)
                   :semantic-op :attention
                   :algebra-plan-id (:id plan)
                   :lowering (keyword (str "subgroup-online-tiled-history-" (name phase)))}
      :attributes {:strategy :routed-paged-subgroup-online-tiled-history
                   :optimization-tier :subgroup-tiled
                   :phase phase
                   :target-dialect target-dialect
                   :target-collective-association
                   (c-dialect/collective-association dialect)
                   :segmented-weighted-reduction-schedule scheduled
                   :kernel-body kernel-body}})))

(defn emit-fp16-tiled-history
  "Emit a two-kernel tiled-history graph with graph-owned mergeable online state.

  The external ABI is exactly the original routed attention ABI. Partial states are typed graph
  temporaries, so allocation, dependencies and ownership remain compiler-managed and invisible
  to cache/page scheduling above this layer."
  ([plan scheduled]
   (emit-fp16-tiled-history plan scheduled :opencl-intel))
  ([plan scheduled target-dialect]
   (let [[plan scheduled {:keys [output id route] :as problem}]
         (cooperative-plan! plan scheduled)
         _ (when-not (swr-schedule/tiled? scheduled)
             (throw (ex-info "tiled-history emission requires a tiled schedule"
                             {:reason :attention-tiled-history-schedule
                              :schedule scheduled})))
         inputs (ordered-inputs plan)
         partial-specs (swr-body/partial-buffer-specs plan scheduled)
         partial-ids (mapv :id partial-specs)
         partial-body (swr-body/lower-routed-paged-partial plan scheduled)
         merge-body (swr-body/lower-routed-paged-merge plan scheduled)
         partial-arguments (into inputs partial-ids)
         merge-arguments (conj partial-ids output)
         partial-artifact (emitted-body-artifact
                           problem plan scheduled :partial partial-body
                           (partial-abi problem partial-specs partial-body) partial-arguments
                           inputs partial-ids target-dialect)
         merge-artifact (emitted-body-artifact
                         problem plan scheduled :merge merge-body
                         (merge-abi problem partial-specs merge-body) merge-arguments
                         partial-ids [output] target-dialect)
         specs (attention/buffer-specs problem)
         graph-buffer (fn [buffer-id]
                        (let [{:keys [dtype elements role]} (get specs buffer-id)]
                          (kgraph/buffer buffer-id dtype elements :device role)))
         partial-node [:attention id :tiled-history :partial]
         merge-node [:attention id :tiled-history :merge]
         external-abi (ordered-abi problem)]
     (kgraph/make
      {:inputs (mapv graph-buffer inputs)
       :outputs [(graph-buffer output)]
       :temporaries (mapv (fn [{:keys [id dtype elements]}]
                            (kgraph/buffer id dtype elements :device :temporary))
                          partial-specs)
       :abi external-abi
       :arguments (conj inputs output)
       :scalars []
       :nodes [(kgraph/->ScheduledKernel
                partial-node partial-artifact
                (vec (concat (map #(kgraph/->ValueUse % :read) inputs)
                             (map #(kgraph/->ValueUse % :write) partial-ids)))
                #{}
                [])
               (kgraph/->ScheduledKernel
                merge-node merge-artifact
                (vec (concat (map #(kgraph/->ValueUse % :read) partial-ids)
                             [(kgraph/->ValueUse output :write)]))
                #{}
                [partial-node])]
       :effects {:kind :attention :logical-visibility true :ordered-page-routing true}
       :provenance {:operation-id id :semantic-op :attention
                    :algebra-plan-id (:id plan)
                    :lowering :subgroup-online-tiled-history}
       :attributes {:strategy :routed-paged-subgroup-online-tiled-history
                    :reference? false
                    :optimization-tier :subgroup-tiled
                    :algebra :segmented-weighted-reduction
                    :algebra-key (swr/algebra-key plan)
                    :storage-dtype :half
                    :q-dtype (:q-dtype problem)
                    :output-dtype (:output-dtype problem)
                    :accumulator-dtype :float
                    :route-kind (attention/route-kind route)
                    :visibility-kind (attention/visibility-kind (:visibility problem))
                    :k-layout (:k-layout problem)
                    :v-layout (:v-layout problem)
                    :visibility (:visibility problem)
                    :layout (attention/layouts problem)
                    :segmented-weighted-reduction-schedule scheduled
                    :materialized-intermediates partial-ids
                    :complexity :parallel-history-tiles-plus-online-state-merge
                    :private-online-state (mapv #(select-keys % [:id :dtype :shape :role])
                                                partial-specs)}}))))

(defn kernel-graph
  "Wrap either verified attention leaf in an explicit one-node scheduled graph."
  [plan artifact]
  (let [plan (reference-plan! plan)
        {:keys [output id] :as problem} (attention-problem plan)
        artifact (kart/validate! artifact)
        specs (attention/buffer-specs problem)
        inputs (ordered-inputs plan)
        graph-buffer (fn [buffer-id]
                       (let [{:keys [dtype elements role]} (get specs buffer-id)]
                         (kgraph/buffer buffer-id dtype elements :device role)))
        uses (vec (concat (map #(kgraph/->ValueUse % :read) inputs)
                          [(kgraph/->ValueUse output :write)]))
        strategy (get-in artifact [:attributes :strategy])
        reference? (= :reference (get-in artifact [:attributes :optimization-tier]))
        node-id [:attention id strategy]]
    (kgraph/make
     {:inputs (mapv graph-buffer inputs)
      :outputs [(graph-buffer output)]
      :abi (:abi artifact)
      :arguments (:arguments artifact)
      :scalars []
      :nodes [(kgraph/->ScheduledKernel node-id artifact uses #{} [])]
      :effects {:kind :attention :logical-visibility true :ordered-page-routing true}
      :provenance {:operation-id id :semantic-op :attention}
      :attributes {:strategy strategy :reference? reference?
                   :route-kind (attention/route-kind (:route problem))}})))
