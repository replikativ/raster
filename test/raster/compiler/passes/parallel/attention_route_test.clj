(ns raster.compiler.passes.parallel.attention-route-test
  (:require [clojure.string :as str]
            [clojure.java.shell :as shell]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.attention :as attention-emit]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as swr-schedule]
            [raster.compiler.passes.parallel.attention-route :as route]
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
    (is (str/includes? source "intel_reqd_sub_group_size(16)"))
    (is (str/includes? source "const float dot = sub_group_reduce_add(partial_dot)"))
    (is (str/includes? source "old_weight = sub_group_broadcast(old_weight, 0)"))
    (is (str/includes? source "const int kv_head = q_head / 2"))
    (is (str/includes? source "if (attention_begin == attention_end)"))
    (is (str/includes? source
                       "for (int token = (int)attention_begin; token < (int)attention_end"))
    (is (str/includes? source "float maximum"))
    (is (str/includes? source "accumulator0 = accumulator0 * old_weight"))))

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
          (is (str/includes? source "const int d15 = (int)lane + 240;"))
          (is (str/includes? source "float accumulator15 = 0.0f;"))
          (is (not (str/includes? source "accumulator16")))
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
            :dot-order :subgroup-tree :online-rescale? true}
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
    (is (= :attention-cooperative-schedule-plan-mismatch
           (try
             (attention-emit/emit-fp16-cooperative
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
    (is (str/includes? (:source artifact) "page_indices[page_begin + logical_page]"))
    (is (str/includes? (:source artifact) "routed_page_count == 0"))))

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
                       "for (int edge = attention_begin; edge < attention_end; ++edge)"))
    (is (str/includes? (:source artifact)
                       "const int token = attention_key_indices[edge]"))
    (is (str/includes? (:source artifact) "token < 0 || token >= length"))
    (is (str/includes? (:source artifact) "kv_position <= (long)q_position"))))

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
    (is (str/includes? source "__global const float* q"))
    (is (str/includes? source "__global float* output"))
    (is (str/includes? source "dot += q[q_base + x] * convert_float(k_pages"))
    (is (str/includes? source "output[out_base + d0] = NAN;"))
    (is (str/includes? source
                       "output[out_base + d0] = denominator == 0.0f ? 0.0f"))))

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
                               :membership-traversal]))))))

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
