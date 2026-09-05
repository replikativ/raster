(ns raster.compiler.passes.parallel.attention-route-test
  (:require [clojure.string :as str]
            [clojure.set :as set]
            [clojure.java.shell :as shell]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.attention :as attention-emit]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-body :as kbody]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as swr-schedule]
            [raster.compiler.passes.parallel.attention-route :as route]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-body :as swr-body]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-schedule
             :as schedule-pass]))

(defn- query
  []
  (attention/packed-query-batch
   {:values 'q :row-offsets 'q-row-offsets :positions 'q-positions :total-tokens 4}))

(defn- dense-route
  []
  (attention/dense-paged-route
   {:page-table 'page-table :lengths 'kv-lengths
    :start-positions 'kv-start-positions :pages-per-sequence 3}))

(defn- csr-route
  []
  (attention/csr-paged-route
   {:page-offsets 'page-offsets :page-indices 'page-indices
    :last-page-lengths 'last-page-lengths :start-positions 'kv-start-positions
    :page-index-capacity 6}))

(defn- csr-visibility
  []
  (attention/csr-visibility
   {:row-offsets 'attention-row-offsets :key-indices 'attention-key-indices
    :key-index-capacity 8 :duplicate-policy :multiset
    :position-filter (attention/visibility
                      {:causal? true :window-left 2 :window-right 0})}))

(defn- problem
  [& overrides]
  (attention/make
   (merge {:id :attention :query (query) :k-pages 'k-pages :v-pages 'v-pages
           :route (dense-route) :output 'output
           :batch-size 3 :q-heads 4 :kv-heads 2
           :qk-head-dim 8 :value-head-dim 6 :page-size 2 :physical-pages 7
           :scale 0.25 :k-layout :kv-head-major :v-layout :page-major
           :visibility (attention/visibility
                        {:causal? true :window-left 2 :window-right 0})}
          (apply hash-map overrides))))

(def ^:private intel-desc
  {:device-type :gpu :vendor "Intel" :subgroup-size 16
   :max-workgroup-size 256})

(defn- operation-tree
  [operations]
  (letfn [(children [operation]
            (vec (concat (:operations operation)
                         (:then-operations operation)
                         (:else-operations operation))))]
    (mapcat #(tree-seq (comp seq children) children %) operations)))

(deftest dense-reference-is-a-complete-packed-attention-compiler-value
  (let [{:keys [strategy reference? declines plan artifact graph schedule]}
        (route/route! (problem)
                      {:device-type :gpu :subgroup-size 16 :max-workgroup-size 256
                       :segmented-weighted-reduction-schedule :reference})]
    (is (= :fp16-reference strategy))
    (is reference?)
    (is (empty? declines))
    (is (swr/plan? plan))
    (is (kart/kernel-artifact? artifact))
    (is (kgraph/kernel-graph? graph))
    (is (= '[q q-row-offsets q-positions k-pages v-pages
             page-table kv-lengths kv-start-positions output]
           (:arguments artifact)))
    (is (= ["q" "q_row_offsets" "q_positions" "k_pages" "v_pages"
            "page_table" "kv_lengths" "kv_start_positions" "output"]
           (mapv :c-name (:abi artifact))))
    (is (= [:half :int :int :half :half :int :int :int :half]
           (mapv :dtype (:abi artifact))))
    (is (= {:workgroup-size [6 1 1] :group-count [1 4 4]} schedule))
    (is (= 3 (count (:workgroup-size (:launch artifact)))))
    (is (= 1 (count (:nodes graph))))
    (is (= :dense-paged (get-in artifact [:attributes :route-kind])))
    (is (str/includes? (:source artifact) "q_row_offsets"))
    (is (str/includes? (:source artifact) "kv_position <= (long)q_position"))
    (is (str/includes? (:source artifact) "kv_position >= (long)q_position - 2L"))
    (is (str/includes? (:source artifact) "physical_page = page_table"))
    (is (str/includes? (:source artifact) "const long v_base"))))

(deftest dense-interval-attention-uses-the-shared-score-online-schedule
  (let [desc {:device-type :gpu :vendor "Intel" :subgroup-size 16
              :max-workgroup-size 256}
        optimized (route/route! (problem) desc)
        reference (route/route! (problem)
                                (assoc desc :segmented-weighted-reduction-schedule :reference))
        {:keys [strategy reference? declines artifact graph schedule]} optimized
        swr-schedule (get-in artifact [:attributes :segmented-weighted-reduction-schedule])
        kernel-body (get-in artifact [:attributes :kernel-body])
        source (:source artifact)]
    (is (= :routed-paged-subgroup-online-score-reuse strategy))
    (is (false? reference?))
    (is (empty? declines))
    (is (swr-schedule/schedule? swr-schedule))
    (is (= :one-workgroup-per-segment (:segment-mapping swr-schedule)))
    (is (= :contiguous-interval (:membership-traversal swr-schedule)))
    (is (= {:kind :lane-strided :components 6 :components-per-lane 1}
           (:value-mapping swr-schedule)))
    (is (= {:workgroup-size [16 1] :group-count [4 4]} schedule))
    (is (= strategy (get-in graph [:attributes :strategy])))
    (is (false? (get-in graph [:attributes :reference?])))
    (is (= (kexec/common-view (:artifact reference))
           (kexec/common-view artifact)))
    (is (not= (:kernel-name (:artifact reference)) (:kernel-name artifact)))
    (is (kbody/kernel-body? kernel-body))
    (is (= :scheduled-kernel-body (get-in kernel-body [:provenance :lowering])))
    (is (str/includes? source "intel_reqd_sub_group_size(16)"))
    (is (str/includes? source
                       "float rstr_dot = sub_group_reduce_add(rstr_partial_dot)"))
    (is (not (str/includes? source "sub_group_broadcast"))
        "identical online state is computed by every lane")
    (is (str/includes? source "int rstr_kv_head = (rstr_query_head / 2)"))
    (is (str/includes? source
                       "for (long rstr_membership_token = rstr_attention_begin"))
    (is (str/includes? source "float rstr_final_maximum"))
    (is (str/includes? source "float rstr_weighted_value_next_0"))))

(deftest aligned-dense-history-selects-a-verified-double-buffered-schedule
  (let [problem (problem :value-head-dim 8)
        descriptor (assoc intel-desc
                          :segmented-weighted-reduction-schedule
                          :subgroup-online-pipelined-history)
        reference (route/route!
                   problem (assoc intel-desc
                                  :segmented-weighted-reduction-schedule :reference))
        {:keys [strategy reference? declines artifact]} (route/route! problem descriptor)
        scheduled (get-in artifact [:attributes :segmented-weighted-reduction-schedule])
        kernel-body (get-in artifact [:attributes :kernel-body])
        operations (operation-tree (:operations kernel-body))
        source (:source artifact)]
    (is (= :routed-paged-subgroup-online-pipelined-history strategy))
    (is (false? reference?))
    (is (empty? declines))
    (is (= (kexec/common-view (:artifact reference))
           (kexec/common-view artifact)))
    (is (= {:kind :double-buffered-membership-rows
            :stages 2 :members-per-iteration 2 :element-dtype :half
            :key-elements 8 :value-elements 8 :transfer-bytes 16
            :overlap :preferred :layout-swizzle :identity
            :tail-policy :separate-epilogue
            :shared-memory-bytes 64}
           (:staging scheduled)))
    (is (= 2 (count (:allocations kernel-body))))
    (is (= [[2 8] [2 8]] (mapv :shape (:allocations kernel-body))))
    (is (every? #(= {:kind :shared-memory :swizzle :identity}
                    (select-keys (:layout %) [:kind :swizzle]))
                (:allocations kernel-body)))
    (is (= 64 (get-in kernel-body [:launch :shared-memory-bytes])))
    (is (= :scheduled-pipelined-kernel-body
           (get-in kernel-body [:provenance :lowering])))
    (is (some #(instance? raster.compiler.ir.kernel_body.PipelinedFor %) operations))
    (is (some #(instance? raster.compiler.ir.kernel_body.AsyncWorkgroupCopy %) operations))
    (is (str/includes? source "async_work_group_copy"))
    (is (str/includes? source "rstr_pipeline_has_odd"))
    (is (str/includes? source "rstr_pipeline_membership_count >= 2"))
    (is (= :routed-paged-subgroup-online-score-reuse
           (:strategy (route/route! problem intel-desc)))
        "unmeasured double buffering remains opt-in")))

(deftest staged-layout-axis-emits-every-verified-row-swizzle
  (let [problem (problem :value-head-dim 8)
        plan (:plan (route/route!
                     problem
                     (assoc intel-desc :segmented-weighted-reduction-schedule :reference)))
        candidates (schedule-pass/measured-candidates plan intel-desc)]
    (is (= [{:stage-count 0 :layout-swizzle :identity}
            {:stage-count 2 :layout-swizzle :identity}
            {:stage-count 2 :layout-swizzle :xor-2}
            {:stage-count 2 :layout-swizzle :xor-4}
            {:stage-count 2 :layout-swizzle :xor-8}]
           (mapv #(select-keys % [:stage-count :layout-swizzle]) candidates)))
    (let [descriptor (assoc intel-desc
                            :segmented-weighted-reduction-schedule
                            :subgroup-online-pipelined-history
                            :segmented-weighted-reduction-layout-swizzle :xor-8)
          artifact (:artifact (route/route! problem descriptor))
          body (get-in artifact [:attributes :kernel-body])
          source (:source artifact)]
      (is (= [:xor-8 :xor-8]
             (mapv #(get-in % [:layout :swizzle]) (:allocations body))))
      (is (str/includes? source "^ ((long)(0) & 7)"))
      (is (str/includes? source "get_local_id(0)"))
      (is (not (str/includes? source "async_work_group_copy"))
          "a swizzled logical row copy uses the honest cooperative scatter fallback"))))

(deftest measured-attention-dispatch-carries-static-stage-and-layout-alternatives
  (let [dispatch (route/measured-dispatch (problem :value-head-dim 8) intel-desc)
        strategies (mapv kdispatch/alternative-strategy (:alternatives dispatch))]
    (is (kdispatch/kernel-dispatch? dispatch))
    (is (= :fp16-reference (:default-strategy dispatch)))
    (is (= {:kind :fixed-strategy :strategy :fp16-reference} (:selector dispatch)))
    (is (= [:fp16-reference
            :routed-paged-subgroup-online-score-reuse
            :routed-paged-subgroup-online-pipelined-history
            :routed-paged-subgroup-online-pipelined-history-xor-2
            :routed-paged-subgroup-online-pipelined-history-xor-4
            :routed-paged-subgroup-online-pipelined-history-xor-8]
           strategies))
    (is (every? #(= (kexec/common-view (first (:alternatives dispatch)))
                    (kexec/common-view %))
                (rest (:alternatives dispatch))))
    (is (= :offline-device-measurement
           (get-in dispatch [:attributes :selection])))))

(deftest routed-row-lowering-proves-structure-instead-of-requiring-attention-provenance
  (let [plan (:plan (route/route!
                     (problem :value-head-dim 8)
                     (assoc intel-desc :segmented-weighted-reduction-schedule :reference)))
        generic-plan (assoc plan :provenance {:semantic-op :functional-program
                                              :operation-id :generic-weighted-reduction
                                              :lowering :recognized-algebra})
        scheduled (:schedule
                   (schedule-pass/plan-subgroup-online-pipelined generic-plan intel-desc))
        artifact (attention-emit/emit-fp16-pipelined generic-plan scheduled)]
    (is (= :routed-paged-subgroup-online-pipelined-history
           (get-in artifact [:attributes :strategy])))
    (is (= (:id generic-plan)
           (get-in artifact [:provenance :algebra-plan-id])))))

(deftest pipelined-history-legality-is-storage-and-resource-explicit
  (let [aligned-problem (problem :value-head-dim 8)
        plan (:plan (route/route!
                     aligned-problem
                     (assoc intel-desc
                            :segmented-weighted-reduction-schedule :reference)))]
    (is (= :pipelined-history-requires-dense-paged-route
           (:reason
            (schedule-pass/plan-subgroup-online-pipelined
             (:plan (route/route!
                     (problem :route (csr-route) :value-head-dim 8)
                     (assoc intel-desc
                            :segmented-weighted-reduction-schedule :reference)))
             intel-desc))))
    (is (= :pipelined-history-requires-interval-membership
           (:reason
            (schedule-pass/plan-subgroup-online-pipelined
             (:plan (route/route!
                     (problem :visibility (csr-visibility) :value-head-dim 8)
                     (assoc intel-desc
                            :segmented-weighted-reduction-schedule :reference)))
             intel-desc))))
    (is (= :pipelined-history-row-transfer-width-unsupported
           (:reason (schedule-pass/plan-subgroup-online-pipelined
                     (:plan (route/route!
                             (problem)
                             (assoc intel-desc
                                    :segmented-weighted-reduction-schedule :reference)))
                     intel-desc))))
    (is (= :pipelined-history-shared-memory-exceeded
           (:reason (schedule-pass/plan-subgroup-online-pipelined
                     plan (assoc-in intel-desc [:cache :slm] 32)))))))

(deftest cooperative-attention-source-is-valid-opencl
  (let [clang? (zero? (:exit (shell/sh "sh" "-c" "command -v clang")))]
    (if-not clang?
      (is true "clang unavailable")
      (doseq [[physical-route visibility]
              [[(dense-route) (attention/visibility)]
               [(dense-route) (csr-visibility)]
               [(csr-route) (attention/visibility)]
               [(csr-route) (csr-visibility)]]]
        (let [source (get-in
                      (route/route!
                       (problem :route physical-route :visibility visibility
                                :qk-head-dim 256 :value-head-dim 256)
                       {:device-type :gpu :vendor "Intel"
                        :subgroup-size 16 :max-workgroup-size 256})
                      [:artifact :source])
              result (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                               "-fsyntax-only" "-" :in source)]
          (is (str/includes? source
                             "int rstr_value_component_15 = (rstr_lane + 240);"))
          (is (str/includes? source
                             "float rstr_weighted_value_result_15 = 0.0f;"))
          (is (not (str/includes? source "rstr_value_component_16")))
          (is (zero? (:exit result)) (:err result)))))))

(deftest scheduled-attention-artifacts-preserve-the-abi-across-c-family-targets
  (let [descriptor {:device-type :gpu :subgroup-size 32 :max-workgroup-size 1024}
        plan (:plan (route/route!
                     (problem)
                     (assoc descriptor :segmented-weighted-reduction-schedule :reference)))
        scheduled (:schedule (schedule-pass/plan-subgroup-online plan descriptor))
        opencl (attention-emit/emit-fp16-cooperative
                plan scheduled :opencl-portable)]
    (doseq [[dialect target]
            [[:cuda :cuda-c] [:hip :hip-cpp]]]
      (let [artifact (attention-emit/emit-fp16-cooperative plan scheduled dialect)]
        (is (= target (:target artifact)))
        (is (= (:abi opencl) (:abi artifact)))
        (is (= (:arguments opencl) (:arguments artifact)))
        (is (= (:effects opencl) (:effects artifact)))
        (is (= dialect (get-in artifact [:attributes :target-dialect])))
        (is (= :explicit-shuffle-down-tree
               (get-in artifact [:attributes :target-collective-association])))
        (is (str/includes? (:source artifact) "extern \"C\" __global__ void"))
        (is (str/includes? (:source artifact) "__shfl_down"))))))

(deftest pipelined-attention-preserves-one-body-and-abi-across-c-family-targets
  (let [descriptor {:device-type :gpu :subgroup-size 32 :max-workgroup-size 1024}
        plan (:plan (route/route!
                     (problem :value-head-dim 8)
                     (assoc descriptor :segmented-weighted-reduction-schedule :reference)))
        scheduled (:schedule
                   (schedule-pass/plan-subgroup-online-pipelined plan descriptor))
        opencl (attention-emit/emit-fp16-pipelined
                plan scheduled :opencl-portable)
        cuda (attention-emit/emit-fp16-pipelined
              plan scheduled :cuda {:compute-capability [8 0]})
        hip (attention-emit/emit-fp16-pipelined plan scheduled :hip)]
    (doseq [artifact [cuda hip]]
      (is (= (:abi opencl) (:abi artifact)))
      (is (= (:arguments opencl) (:arguments artifact)))
      (is (= (:effects opencl) (:effects artifact)))
      (is (= (select-keys (get-in opencl [:attributes :kernel-body])
                          [:id :parameters :allocations :indices :schedule :launch
                           :provenance :attributes])
             (select-keys (get-in artifact [:attributes :kernel-body])
                          [:id :parameters :allocations :indices :schedule :launch
                           :provenance :attributes]))))
    (is (= :cuda-c (:target cuda)))
    (is (str/includes? (:source cuda) "cp.async.ca.shared.global"))
    (is (str/includes? (:source cuda) "cp.async.wait_group 1"))
    (is (= :hip-cpp (:target hip)))
    (is (str/includes? (:source hip) "synchronous cooperative copies are complete"))
    (is (not (str/includes? (:source hip) "cp.async")))))

(deftest tiled-history-is-a-two-stage-kernel-graph-with-a-stable-external-abi
  (let [desc (assoc intel-desc
                    :segmented-weighted-reduction-schedule
                    :subgroup-online-tiled-history
                    :segmented-weighted-reduction-history-tile-size 2)
        reference (route/route!
                   (problem) (assoc intel-desc
                                    :segmented-weighted-reduction-schedule :reference))
        {:keys [strategy reference? artifact graph executable schedule]}
        (route/route! (problem) desc)
        other-graph (:graph (route/route! (problem :id :other-attention) desc))
        scheduled (get-in graph [:attributes :segmented-weighted-reduction-schedule])
        [partial merge] (:nodes graph)
        partial-body (get-in partial [:operation :attributes :kernel-body])
        merge-body (get-in merge [:operation :attributes :kernel-body])]
    (is (= :routed-paged-subgroup-online-tiled-history strategy))
    (is (false? reference?))
    (is (nil? artifact) "a multi-kernel executable is not disguised as one artifact")
    (is (identical? graph executable))
    (is (= (kexec/common-view (:graph reference)) (kexec/common-view graph)))
    (is (= {:kind :static-contiguous-tiles :tile-size 2 :tile-count 3
            :membership-capacity 6 :merge-order :increasing-membership-tile}
           (:membership-tiling scheduled)))
    (is (= {:kind :maximum-rescale-sum
            :order :increasing-membership-tile
            :nan-policy :propagate :empty-policy :identity}
           (get-in scheduled [:state :merge])))
    (is (= :increasing-members-within-tile-then-increasing-tile-left-fold
           (get-in scheduled [:numerical-mode :online-state-order])))
    (is (= 2 (count (:nodes graph))))
    (is (= 4 (count (:temporaries graph))))
    (is (= (mapv :id (:temporaries graph))
           (get-in graph [:attributes :materialized-intermediates])))
    (is (empty? (set/intersection
                 (set (map :id (:temporaries graph)))
                 (set (map :id (:temporaries other-graph)))))
        "linked tiled reductions retain disjoint graph-private identities")
    (is (= #{:partial-valid :partial-maximum :partial-denominator
             :partial-weighted-values}
           (set (map :role (get-in graph [:attributes :private-online-state])))))
    (is (= [4 4 3] (get-in partial [:operation :launch :group-count])))
    (is (= [4 4] (get-in merge [:operation :launch :group-count])))
    (is (= [(:id partial)] (:dependencies merge)))
    (is (= {:stages
            [{:id (:id partial) :workgroup-size [16 1 1] :group-count [4 4 3]}
             {:id (:id merge) :workgroup-size [16 1] :group-count [4 4]}]}
           schedule))
    (is (kbody/kernel-body? partial-body))
    (is (kbody/kernel-body? merge-body))
    (is (= :scheduled-partial-kernel-body (get-in partial-body [:provenance :lowering])))
    (is (= :scheduled-merge-kernel-body (get-in merge-body [:provenance :lowering])))
    (is (= :partial (get-in partial-body [:schedule :phase])))
    (is (= :merge (get-in merge-body [:schedule :phase])))
    (is (every? #(str/includes? (get-in % [:operation :source])
                                "intel_reqd_sub_group_size(16)")
                [partial merge])
        "lane-mapped merge kernels retain their required subgroup geometry without a collective")
    (is (str/includes? (get-in partial [:operation :source])
                       "rstr_tile_membership_begin"))
    (is (str/includes? (get-in merge [:operation :source])
                       "for (int rstr_merge_tile = 0; rstr_merge_tile < 3"))
    (let [nvidia (route/route
                  (problem)
                  {:device-type :gpu :vendor "NVIDIA" :subgroup-size 32
                   :max-workgroup-size 1024
                   :segmented-weighted-reduction-schedule
                   :subgroup-online-tiled-history})]
      (is (nil? (:strategy nvidia)))
      (is (= :routed-paged-subgroup-online-tiled-history
             (get-in nvidia [:declines 0 :leaf])))
      (is (= :score-reuse-requires-intel-subgroup-dialect
             (get-in nvidia [:declines 0 :reason]))))
    (is (= :score-reuse-invalid-history-tile-size
           (get-in (route/route
                    (problem)
                    (assoc desc :segmented-weighted-reduction-history-tile-size 0))
                   [:declines 0 :reason])))
    (testing "a self-consistent tile plan cannot forge the source membership capacity"
      (let [plan (:plan reference)
            lower! (fn [candidate]
                     (try
                       (swr-body/lower-routed-paged-partial plan candidate)
                       :accepted
                       (catch clojure.lang.ExceptionInfo exception
                         (:reason (ex-data exception)))))
            under (assoc scheduled :membership-tiling
                         {:kind :static-contiguous-tiles :tile-size 2 :tile-count 2
                          :membership-capacity 4 :merge-order :increasing-membership-tile})
            over (assoc scheduled :membership-tiling
                        {:kind :static-contiguous-tiles :tile-size 2 :tile-count 4
                         :membership-capacity 8 :merge-order :increasing-membership-tile})
            oversized-capacity (inc (long Integer/MAX_VALUE))
            oversized (assoc scheduled :membership-tiling
                            {:kind :static-contiguous-tiles :tile-size 2
                             :tile-count (quot (+ oversized-capacity 1) 2)
                             :membership-capacity oversized-capacity
                             :merge-order :increasing-membership-tile})]
        (is (= :segmented-weighted-reduction-body-membership-capacity (lower! under)))
        (is (= :segmented-weighted-reduction-body-membership-capacity (lower! over)))
        (is (= :segmented-weighted-reduction-membership-tiling (lower! oversized))
            "values that become int literals are rejected before lowering")))))

(deftest tiled-history-sources-cover-route-and-visibility-products
  (let [clang? (zero? (:exit (shell/sh "sh" "-c" "command -v clang")))
        desc (assoc intel-desc
                    :segmented-weighted-reduction-schedule
                    :subgroup-online-tiled-history
                    :segmented-weighted-reduction-history-tile-size 2)]
    (if-not clang?
      (is true "clang unavailable")
      (doseq [[physical-route visibility]
              [[(dense-route) (attention/visibility)]
               [(dense-route) (csr-visibility)]
               [(csr-route) (attention/visibility)]
               [(csr-route) (csr-visibility)]]
              node (get-in (route/route!
                            (problem :route physical-route :visibility visibility)
                            desc)
                           [:graph :nodes])]
        (let [source (get-in node [:operation :source])
              result (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                               "-fsyntax-only" "-" :in source)]
          (is (zero? (:exit result)) (:err result)))))))

(deftest cooperative-schedule-is-validated-and-target-legality-is-explicit
  (let [desc {:device-type :gpu :vendor "Intel"
              :subgroup-size 16 :max-workgroup-size 256}
        plan (:plan (route/route!
                     (problem) (assoc desc :segmented-weighted-reduction-schedule :reference)))
        {:keys [schedule]} (schedule-pass/plan-subgroup-online
                            plan desc)
        too-wide (route/route! (problem :value-head-dim 513) desc)]
    (is (swr-schedule/schedule? schedule))
    (is (= {:kind :subgroup :width 16 :axis :qk-component}
           (:score-reduction schedule)))
    (is (= {:score-accumulate :float :state-accumulate :float
            :dot-order :implementation-defined
            :online-state-order :increasing-membership
            :online-rescale? true}
           (:numerical-mode schedule)))
    (is (= :segmented-weighted-reduction-value-mapping
           (try
             (swr-schedule/validate!
              (assoc-in schedule [:value-mapping :components-per-lane] 2))
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :segmented-weighted-reduction-membership-traversal
           (try
             (swr-schedule/validate!
              (assoc schedule :membership-traversal :sequential))
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :segmented-weighted-reduction-partial-schedule
           (try
             (swr-body/partial-buffer-specs plan schedule)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (let [tiled (:schedule (schedule-pass/plan-subgroup-online-tiled
                            plan (assoc desc
                                        :segmented-weighted-reduction-history-tile-size 2)))]
      (is (= :segmented-weighted-reduction-membership-tiling
             (try
               (swr-schedule/validate!
                (assoc-in tiled [:membership-tiling :tile-count] 2))
               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (= :attention-cooperative-schedule-plan-mismatch
           (try
             (attention-emit/emit-fp16-cooperative
              plan (assoc-in schedule [:score-reduction :axis] :wrong-axis))
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :segmented-weighted-reduction-body-plan-mismatch
           (try
             (swr-body/lower-routed-paged
              plan (assoc-in schedule [:score-reduction :axis] :wrong-axis))
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (let [nvidia-desc {:device-type :gpu :vendor "NVIDIA" :subgroup-size 32
                       :max-workgroup-size 1024}
          nvidia-schedule (schedule-pass/plan-subgroup-online plan nvidia-desc)
          nvidia-route (route/route (problem) nvidia-desc)]
      (is (:ok nvidia-schedule)
          "the schedule is portable even while the current optimized emitter is not")
      (is (= 32 (get-in nvidia-schedule [:schedule :workgroup-size])))
      (is (:reference? nvidia-route))
      (is (= :score-reuse-requires-intel-subgroup-dialect
             (get-in nvidia-route [:declines 0 :reason]))))
    (let [spoofed-intel {:device-type :gpu :backend :cuda :vendor "Intel"
                         :subgroup-size 32 :max-workgroup-size 1024
                         :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 32}}
          routed (route/route (problem) spoofed-intel)]
      (is (:reference? routed))
      (is (= :score-reuse-requires-intel-subgroup-dialect
             (get-in routed [:declines 0 :reason]))))
    (is (= :score-reuse-missing-execution-capability
           (:reason (schedule-pass/plan-subgroup-online plan nil)))
        "an absent descriptor cannot prove a cooperative schedule")
    (is (= :score-reuse-missing-execution-capability
           (:reason
            (schedule-pass/plan-subgroup-online
             plan {:device-type :gpu :vendor "Intel"
                   :execution {:subgroup-sizes #{}
                               :preferred-subgroup-size nil
                               :max-workgroup-size 256}}))))
    (is (= :score-reuse-subgroup-width-unsupported
           (:reason
            (schedule-pass/plan-subgroup-online
             plan {:device-type :gpu :vendor "Intel"
                   :execution {:subgroup-sizes #{32}
                               :preferred-subgroup-size 16
                               :max-workgroup-size 256}}))))
    (is (:reference? too-wide))
    (is (= :score-reuse-register-state-too-wide
           (get-in too-wide [:declines 0 :reason])))))

(deftest csr-route-has-native-compact-page-abi-and-cooperative-schedule
  (let [{:keys [artifact reference? declines]}
        (route/route! (problem :route (csr-route)) intel-desc)]
    (is (false? reference?))
    (is (empty? declines))
    (is (= :csr-paged (get-in artifact [:attributes :route-kind])))
    (is (= '[q q-row-offsets q-positions k-pages v-pages page-offsets page-indices
             last-page-lengths kv-start-positions output]
           (:arguments artifact)))
    (is (= ["page_offsets" "page_indices" "last_page_lengths"]
           (subvec (mapv :c-name (:abi artifact)) 5 8)))
    (is (= :contiguous-interval
           (get-in artifact [:attributes :segmented-weighted-reduction-schedule
                             :membership-traversal])))
    (is (str/includes? (:source artifact) "page_indices["))
    (is (str/includes? (:source artifact)
                       "rstr_empty_last_page_valid = (rstr_final_page_length == 0)"))))

(deftest csr-route-sanitizes-untrusted-page-metadata-before-proved-arithmetic
  (let [{:keys [plan artifact]} (route/route! (problem :route (csr-route)) intel-desc)
        schedule (get-in artifact [:attributes :segmented-weighted-reduction-schedule])
        lowered (swr-body/lower-routed-paged plan schedule)
        operations (operation-tree (:operations lowered))
        compute-by-id (into {}
                            (keep (fn [operation]
                                    (when-let [result (:result operation)]
                                      [(:id result) operation])))
                            operations)
        raw-routed-count (get compute-by-id 'raw-routed-page-count)
        routed-count (get compute-by-id 'routed-page-count)
        computed-length (get compute-by-id 'computed-kv-length)]
    ;; Raw device page offsets and last-page lengths have no overflow contract.  The only marked
    ;; arithmetic consumes clamped SSA values, while invalid raw metadata remains in route-valid
    ;; and therefore masks all writes.
    (is (= :no-overflow (get-in raw-routed-count [:expression :options :overflow])))
    (is (= ['safe-page-end 'safe-page-begin]
           (get-in raw-routed-count [:expression :arguments])))
    (is (= ['raw-routed-page-count (kbody/literal 0 :int)]
           (get-in routed-count [:expression :arguments])))
    (is (= :no-overflow (get-in computed-length [:expression :options :overflow])))
    (is (= 'safe-final-page-length
           (last (get-in computed-length [:expression :arguments]))))
    (is (some? (get compute-by-id 'route-valid)))))

(deftest csr-route-clamps-an-inverted-row-before-certified-length-arithmetic
  (let [route (attention/csr-paged-route
               {:page-offsets 'page-offsets :page-indices 'page-indices
                :last-page-lengths 'last-page-lengths :start-positions 'kv-start-positions
                :page-index-capacity 1})
        problem (problem :route route :page-size Integer/MAX_VALUE :physical-pages 1)
        {:keys [plan artifact]} (route/route! problem intel-desc)
        schedule (get-in artifact [:attributes :segmented-weighted-reduction-schedule])
        lowered (swr-body/lower-routed-paged plan schedule)
        compute-by-id (into {}
                            (keep (fn [operation]
                                    (when-let [result (:result operation)]
                                      [(:id result) operation])))
                            (operation-tree (:operations lowered)))
        count-minus-one (get compute-by-id 'computed-kv-length)]
    ;; An invalid row with begin=1/end=0 has a raw count of -1.  It is clamped to zero before
    ;; the certified `count - 1` product, even with the largest legal page size.
    (is (= 'safe-final-page-length
           (last (get-in count-minus-one [:expression :arguments]))))
    (is (= 'routed-page-count
           (get-in count-minus-one [:expression :arguments 0 :arguments 0 :arguments 0])))
    (is (= :no-overflow (get-in count-minus-one [:expression :options :overflow])))))

(deftest logical-csr-visibility-composes-with-physical-route-as-distinct-abi-slots
  (let [{:keys [artifact reference? declines]}
        (route/route! (problem :visibility (csr-visibility)) intel-desc)]
    (is (false? reference?))
    (is (empty? declines))
    (is (= :dense-paged (get-in artifact [:attributes :route-kind])))
    (is (= :csr (get-in artifact [:attributes :visibility-kind])))
    (is (= '[q q-row-offsets q-positions k-pages v-pages
             page-table kv-lengths kv-start-positions
             attention-row-offsets attention-key-indices output]
           (:arguments artifact)))
    (is (= ["attention_row_offsets" "attention_key_indices"]
           (subvec (mapv :c-name (:abi artifact)) 8 10)))
    (is (= :csr-row
           (get-in artifact [:attributes :segmented-weighted-reduction-schedule
                             :membership-traversal])))
    (is (str/includes? (:source artifact)
                       "for (int rstr_membership_edge = rstr_attention_begin"))
    (is (str/includes? (:source artifact)
                       "rstr_logical_token = attention_key_indices["))
    (is (str/includes? (:source artifact) "rstr_logical_token_nonnegative"))
    (is (str/includes? (:source artifact)
                       "rstr_kv_position <= rstr_query_position_long"))))

(deftest independent-k-and-v-layouts-are-lowered-without-repacking
  (let [source (:source (:artifact (route/route! (problem))))]
    (is (str/includes? source
                       "((long)kv_head * 7 + physical_page) * 2 + page_token) * 8"))
    (is (str/includes? source
                       "((long)physical_page * 2 + page_token) * 2 + kv_head) * 6"))))

(deftest fp32-query-and-output-compose-with-fp16-kv-storage
  (let [fp16-artifact (:artifact (route/route! (problem) intel-desc))
        artifact (:artifact (route/route!
                             (problem :q-dtype :float :output-dtype :float)
                             intel-desc))
        source (:source artifact)]
    (is (not= (:kernel-name fp16-artifact) (:kernel-name artifact)))
    (is (= [:float :int :int :half :half :int :int :int :float]
           (mapv :dtype (:abi artifact))))
    (is (= :float (get-in artifact [:attributes :q-dtype])))
    (is (= :float (get-in artifact [:attributes :output-dtype])))
    (is (str/includes? source "__global const float* restrict q"))
    (is (str/includes? source "__global float* output"))
    (is (str/includes? source
                       "float rstr_qk_product = (rstr_query_float * rstr_key_float)"))
    (is (str/includes? source
                       "rstr_valid_output_value_0 = (rstr_final_valid"))
    (is (str/includes? source "output["))))

(deftest pinned-cooperative-policy-selects-csr-membership-without-changing-semantics
  (let [result (route/route
                (problem :visibility (csr-visibility))
                {:device-type :gpu :vendor "Intel" :subgroup-size 16
                 :max-workgroup-size 256
                 :segmented-weighted-reduction-schedule :subgroup-score-reuse})]
    (is (= :routed-paged-subgroup-online-score-reuse (:strategy result)))
    (is (empty? (:declines result)))
    (is (= :csr-row
           (get-in result [:artifact :attributes :segmented-weighted-reduction-schedule
                           :membership-traversal])))))

(deftest physical-routing-and-logical-membership-vary-independently
  (doseq [[physical-route visibility traversal]
          [[(dense-route) (attention/visibility) :contiguous-interval]
           [(dense-route) (csr-visibility) :csr-row]
           [(csr-route) (attention/visibility) :contiguous-interval]
           [(csr-route) (csr-visibility) :csr-row]]]
    (let [{:keys [strategy reference? declines artifact]}
          (route/route! (problem :route physical-route :visibility visibility) intel-desc)]
      (is (= :routed-paged-subgroup-online-score-reuse strategy))
      (is (false? reference?))
      (is (empty? declines))
      (is (= traversal
             (get-in artifact [:attributes :segmented-weighted-reduction-schedule
                               :membership-traversal])))
      (let [kernel-body (get-in artifact [:attributes :kernel-body])]
        (is (kbody/kernel-body? kernel-body))
        (is (= (:arguments artifact) (mapv :id (:parameters kernel-body))))
        (is (= (vec (butlast (:arguments artifact)))
               (mapv :buffer (:stable-reads kernel-body))))
        (is (every? #(or (= :output (:kind %))
                         (= :no-write-alias (:aliasing %)))
                    (:abi artifact)))))))

(deftest unsupported-representations-return-machine-readable-declines
  (testing "quantization declines before generic dtype routing"
    (let [r (route/route
             (problem :k-dtype :byte
                      :k-format {:dtype :byte :quantization :int8 :group-size 32}))]
      (is (nil? (:strategy r)))
      (is (= :attention-quantized-kv-abi-unimplemented
             (get-in r [:declines 0 :reason])))))
  (testing "non-FP16 K/V storage declines rather than being reinterpreted"
    (let [r (route/route (problem :k-dtype :float))]
      (is (= :attention-reference-storage-unsupported
             (get-in r [:declines 0 :reason])))))
  (testing "a CPU target cannot receive GPU reference scheduling"
    (is (= :attention-requires-gpu
           (get-in (route/route (problem) {:device-type :cpu})
                   [:declines 0 :reason]))))
  (is (= :attention-no-kernel-route
         (try
           (route/route! (problem) {:device-type :cpu})
           (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
