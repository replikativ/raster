(ns raster.compiler.backend.gpu.attention
  "FP16-KV leaves for logical attention over dense or CSR paged KV routes.

   The direct one-work-item/component leaf remains the executable semantic oracle.  A separately
   validated SegmentedWeightedReductionSchedule emits one subgroup/query-head across dense/CSR
   routes and interval/CSR membership, sharing each QK score across lane-strided value
   accumulators without changing the semantic plan, ordered ABI, storage ownership or graph
   effects."
  (:require [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as swr-schedule]))

(defn- attention-problem
  [plan]
  (let [{:keys [source-operation provenance]} (swr/validate! plan)]
    (when-not (and (= :attention (:semantic-op provenance))
                   (= :canonical-segmented-weighted-reduction (:lowering provenance)))
      (throw (ex-info "FP16 attention emission requires an attention-derived reduction plan"
                      {:reason :attention-emitter-wrong-plan-provenance
                       :provenance provenance})))
    (attention/validate! source-operation)))

(defn- reference-plan!
  [plan]
  (let [plan (swr/validate! plan)
        {:keys [id query route output q-heads kv-heads qk-head-dim value-head-dim
                q-dtype k-dtype v-dtype output-dtype accumulator-dtype scale visibility]
         :as problem} (attention-problem plan)
        expected-segments [{:name :query-token :extent (:total-tokens query)}
                           {:name :query-head :extent q-heads}]
        expected-score {:kind :dot
                        :axis {:name :qk-component :extent qk-head-dim}
                        :head-map {:kind :grouped-query
                                   :query-heads q-heads :kv-heads kv-heads}
                        :left {:kind :packed-query :buffer (:values query) :dtype q-dtype}
                        :right {:kind :routed-key :buffer (:k-pages problem) :dtype k-dtype}}
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
                   (= (attention/route-kind route) (get-in plan [:storage :route-kind]))
                   (= route (get-in plan [:storage :route]))
                   (= (attention/route-buffer-ids route) (get-in plan [:storage :buffers]))
                   (= expected-score (select-keys (:score plan)
                                                  [:kind :axis :head-map :left :right]))
                   (= (list 'raster.numeric/* 'dot (double scale))
                      (get-in plan [:score :finalize :body]))
                   (= {:kind :routed-value :buffer (:v-pages problem)
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
                                       :k-layout :v-layout :q-dtype :output-dtype])
                         :schedule schedule-identity)
         visibility (:visibility problem)
         identity (assoc identity
                         :route-kind (attention/route-kind (:route problem))
                         :route-shape (select-keys (:route problem)
                                                   [:pages-per-sequence
                                                    :page-index-capacity])
                         :visibility-kind (attention/visibility-kind visibility)
                         :visibility-shape
                         (cond-> {:position-filter (into {} (attention/position-filter visibility))}
                           (attention/csr-visibility? visibility)
                           (assoc :key-index-capacity (:key-index-capacity visibility)))
                         :total-query-tokens (get-in problem [:query :total-tokens]))]
     (format "raster_attention_fp16_%08x" (bit-and 0xffffffff (long (hash identity)))))))

(defn- opencl-type
  [dtype]
  (case dtype
    :half "half"
    :float "float"))

(defn- load-float
  [dtype expression]
  (case dtype
    :half (str "convert_float(" expression ")")
    :float expression))

(defn- store-float
  [dtype expression]
  (case dtype
    :half (str "convert_half_rte(" expression ")")
    :float expression))

(defn- cache-base
  [{:keys [physical-pages page-size kv-heads]} layout dim]
  (case layout
    :kv-head-major
    (str "((((long)kv_head * " physical-pages " + physical_page) * "
         page-size " + page_token) * " dim ")")
    :page-major
    (str "((((long)physical_page * " page-size " + page_token) * "
         kv-heads " + kv_head) * " dim ")")))

(defn- route-signature
  [route]
  (if (attention/dense-paged-route? route)
    (str "    __global const int* page_table,\n"
         "    __global const int* kv_lengths,\n"
         "    __global const int* kv_start_positions,\n")
    (str "    __global const int* page_offsets,\n"
         "    __global const int* page_indices,\n"
         "    __global const int* last_page_lengths,\n"
         "    __global const int* kv_start_positions,\n")))

(defn- visibility-signature
  [visibility]
  (when (attention/csr-visibility? visibility)
    (str "    __global const int* attention_row_offsets,\n"
         "    __global const int* attention_key_indices,\n")))

(defn- route-initialization
  [{:keys [route page-size]} invalid-result]
  (if (attention/dense-paged-route? route)
    (let [capacity (* (:pages-per-sequence route) page-size)]
      (str "  const int length = kv_lengths[batch];\n"
           "  const int kv_start_position = kv_start_positions[batch];\n"
           "  if (length < 0 || length > " capacity
           " || kv_start_position < 0\n"
           "      || (long)kv_start_position + (long)length > 2147483648L) {\n"
           invalid-result
           "  }\n"))
    (let [capacity (:page-index-capacity route)]
      (str "  const int page_begin = page_offsets[batch];\n"
           "  const int page_end = page_offsets[batch + 1];\n"
           "  const int final_page_length = last_page_lengths[batch];\n"
           "  const int kv_start_position = kv_start_positions[batch];\n"
           "  if (page_offsets[0] != 0 || page_begin < 0 || page_end < page_begin\n"
           "      || page_end > " capacity " || kv_start_position < 0) {\n"
           invalid-result
           "  }\n"
           "  const int routed_page_count = page_end - page_begin;\n"
           "  if ((routed_page_count == 0 && final_page_length != 0)\n"
           "      || (routed_page_count > 0\n"
           "          && (final_page_length < 1 || final_page_length > " page-size "))) {\n"
           invalid-result
           "  }\n"
           "  const int length = routed_page_count == 0 ? 0\n"
           "      : (routed_page_count - 1) * " page-size " + final_page_length;\n"
           "  if ((long)kv_start_position + (long)length > 2147483648L) {\n"
           invalid-result
           "  }\n"))))

(defn- physical-page-expression
  [route]
  (if (attention/dense-paged-route? route)
    (str "page_table[(long)batch * " (:pages-per-sequence route) " + logical_page]")
    "page_indices[page_begin + logical_page]"))

(defn- visibility-statements
  [{:keys [causal? window-left window-right]}]
  (str (when causal?
         "    visible = visible && (kv_position <= (long)q_position);\n")
       (when (some? window-left)
         (str "    visible = visible && (kv_position >= (long)q_position - "
              window-left "L);\n"))
       (when (some? window-right)
         (str "    visible = visible && (kv_position <= (long)q_position + "
              window-right "L);\n"))))

(defn- visibility-initialization
  [visibility invalid-result]
  (if (attention/csr-visibility? visibility)
    (str "  const int attention_begin = attention_row_offsets[q_token];\n"
         "  const int attention_end = attention_row_offsets[q_token + 1];\n"
         "  if (attention_row_offsets[0] != 0 || attention_begin < 0\n"
         "      || attention_end < attention_begin || attention_end > "
         (:key-index-capacity visibility) ") {\n"
         invalid-result
         "  }\n")
    (let [{:keys [causal? window-left window-right]} visibility]
      (str "  long attention_begin = 0L;\n"
           "  long attention_end = (long)length;\n"
           (when (some? window-left)
             (str "  attention_begin = max(0L, (long)q_position - " window-left
                  "L - (long)kv_start_position);\n"
                  "  attention_begin = min(attention_begin, (long)length);\n"))
           (cond
             causal?
             (str "  attention_end = min(attention_end, (long)q_position"
                  " - (long)kv_start_position + 1L);\n"
                  "  attention_end = max(attention_end, 0L);\n")

             (some? window-right)
             (str "  attention_end = min(attention_end, (long)q_position + " window-right
                  "L - (long)kv_start_position + 1L);\n"
                  "  attention_end = max(attention_end, 0L);\n"))))))

(defn- visibility-loop-start
  [visibility invalid-result]
  (if (attention/csr-visibility? visibility)
    (str "  for (int edge = attention_begin; edge < attention_end; ++edge) {\n"
         "    const int token = attention_key_indices[edge];\n"
         "    if (token < 0 || token >= length) {\n"
         invalid-result
         "    }\n")
    "  for (int token = (int)attention_begin; token < (int)attention_end; ++token) {\n"))

(defn- reference-source
  [problem name]
  (let [{:keys [query route batch-size q-heads kv-heads qk-head-dim value-head-dim
                page-size physical-pages scale k-layout v-layout visibility
                q-dtype output-dtype]} problem
        total-query-tokens (:total-tokens query)
        gqa-ratio (quot q-heads kv-heads)
        k-base (cache-base problem k-layout qk-head-dim)
        v-base (cache-base problem v-layout value-head-dim)
        invalid-result (str "    output[out_index] = " (store-float output-dtype "NAN") ";\n"
                            "    return;\n")]
    (str "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "__kernel void " name "(\n"
         "    __global const " (opencl-type q-dtype) "* q,\n"
         "    __global const int* q_row_offsets,\n"
         "    __global const int* q_positions,\n"
         "    __global const half* k_pages,\n"
         "    __global const half* v_pages,\n"
         (route-signature route)
         (visibility-signature visibility)
         "    __global " (opencl-type output-dtype) "* output) {\n"
         "  const int d = (int)get_global_id(0);\n"
         "  const int q_head = (int)get_global_id(1);\n"
         "  const int q_token = (int)get_global_id(2);\n"
         "  if (d >= " value-head-dim " || q_head >= " q-heads
         " || q_token >= " total-query-tokens ") return;\n"
         "  const long out_index = ((long)q_token * " q-heads
         " + q_head) * " value-head-dim " + d;\n"
         "  int query_metadata_valid = q_row_offsets[0] == 0\n"
         "      && q_row_offsets[" batch-size "] == " total-query-tokens ";\n"
         "  int batch = -1;\n"
         "  for (int b = 0; b < " batch-size "; ++b) {\n"
         "    const int row_start = q_row_offsets[b];\n"
         "    const int row_end = q_row_offsets[b + 1];\n"
         "    query_metadata_valid = query_metadata_valid && row_start >= 0\n"
         "        && row_end >= row_start && row_end <= " total-query-tokens ";\n"
         "    if (q_token >= row_start && q_token < row_end) batch = b;\n"
         "  }\n"
         "  const int q_position = q_positions[q_token];\n"
         "  if (!query_metadata_valid || batch < 0 || q_position < 0) {\n"
         "    output[out_index] = " (store-float output-dtype "NAN") ";\n"
         "    return;\n"
         "  }\n"
         (route-initialization problem invalid-result)
         (visibility-initialization visibility invalid-result)
         "  const int kv_head = q_head / " gqa-ratio ";\n"
         "  const long q_base = ((long)q_token * " q-heads
         " + q_head) * " qk-head-dim ";\n"
         "  float maximum = -3.402823466e+38f;\n"
         "  float denominator = 0.0f;\n"
         "  float accumulator = 0.0f;\n"
         (visibility-loop-start visibility invalid-result)
         "    const long kv_position = (long)kv_start_position + token;\n"
         "    int visible = 1;\n"
         (visibility-statements (attention/position-filter visibility))
         "    if (!visible) continue;\n"
         "    const int logical_page = token / " page-size ";\n"
         "    const int physical_page = " (physical-page-expression route) ";\n"
         "    if (physical_page < 0 || physical_page >= " physical-pages ") {\n"
         "      output[out_index] = " (store-float output-dtype "NAN") ";\n"
         "      return;\n"
         "    }\n"
         "    const int page_token = token - logical_page * " page-size ";\n"
         "    const long k_base = " k-base ";\n"
         "    const long v_base = " v-base ";\n"
         "    float dot = 0.0f;\n"
         "    for (int x = 0; x < " qk-head-dim "; ++x) {\n"
         "      dot += " (load-float q-dtype "q[q_base + x]")
         " * convert_float(k_pages[k_base + x]);\n"
         "    }\n"
         "    const float logit = dot * " (Float/toString (float scale)) "f;\n"
         "    const float next_maximum = fmax(maximum, logit);\n"
         "    const float old_weight = exp(maximum - next_maximum);\n"
         "    const float new_weight = exp(logit - next_maximum);\n"
         "    accumulator = accumulator * old_weight\n"
         "                  + convert_float(v_pages[v_base + d]) * new_weight;\n"
         "    denominator = denominator * old_weight + new_weight;\n"
         "    maximum = next_maximum;\n"
         "  }\n"
         "  output[out_index] = denominator == 0.0f ? "
         (store-float output-dtype "0.0f") "\n"
         "      : " (store-float output-dtype "accumulator / denominator") ";\n"
         "}\n")))

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
                   (= :routed-paged-kv (get-in schedule [:attributes :storage-kind]))
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

(defn- component-code
  [schedule f]
  (apply str (map f (range (get-in schedule [:value-mapping :components-per-lane])))))

(defn- cooperative-component-declarations
  [schedule]
  (let [subgroup-size (:workgroup-size schedule)]
    (component-code
     schedule
     (fn [slot]
       (str "  const int d" slot " = (int)lane + " (* slot subgroup-size) ";\n"
            "  float accumulator" slot " = 0.0f;\n")))))

(defn- cooperative-component-write
  [problem schedule expression]
  (component-code
   schedule
   (fn [slot]
     (str "    if (d" slot " < " (:value-head-dim problem) ")\n"
          "      output[out_base + d" slot "] = "
          (store-float (:output-dtype problem) (expression slot)) ";\n"))))

(defn- cooperative-component-update
  [problem schedule]
  (component-code
   schedule
   (fn [slot]
     (str "    if (d" slot " < " (:value-head-dim problem) ")\n"
          "      accumulator" slot " = accumulator" slot " * old_weight\n"
          "          + convert_float(v_pages[v_base + d" slot "]) * new_weight;\n"))))

(defn- cooperative-source
  [problem schedule name]
  (let [{:keys [query route batch-size q-heads kv-heads qk-head-dim value-head-dim
                page-size physical-pages scale k-layout v-layout visibility
                q-dtype output-dtype]} problem
        total-query-tokens (:total-tokens query)
        gqa-ratio (quot q-heads kv-heads)
        subgroup-size (:workgroup-size schedule)
        k-base (cache-base problem k-layout qk-head-dim)
        v-base (cache-base problem v-layout value-head-dim)
        invalid-result
        (str (cooperative-component-write problem schedule (constantly "NAN"))
             "    return;\n")
        empty-result
        (str (cooperative-component-write problem schedule (constantly "0.0f"))
             "    return;\n")]
    (str "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "__attribute__((intel_reqd_sub_group_size(" subgroup-size ")))\n"
         "__kernel void " name "(\n"
         "    __global const " (opencl-type q-dtype) "* q,\n"
         "    __global const int* q_row_offsets,\n"
         "    __global const int* q_positions,\n"
         "    __global const half* k_pages,\n"
         "    __global const half* v_pages,\n"
         (route-signature route)
         (visibility-signature visibility)
         "    __global " (opencl-type output-dtype) "* output) {\n"
         "  const long lane = (long)get_sub_group_local_id();\n"
         "  const int q_head = (int)get_group_id(0);\n"
         "  const int q_token = (int)get_group_id(1);\n"
         "  const long out_base = ((long)q_token * " q-heads
         " + q_head) * " value-head-dim ";\n"
         (cooperative-component-declarations schedule)
         "  int query_metadata_valid = q_row_offsets[0] == 0\n"
         "      && q_row_offsets[" batch-size "] == " total-query-tokens ";\n"
         "  int batch = -1;\n"
         "  for (int b = 0; b < " batch-size "; ++b) {\n"
         "    const int row_start = q_row_offsets[b];\n"
         "    const int row_end = q_row_offsets[b + 1];\n"
         "    query_metadata_valid = query_metadata_valid && row_start >= 0\n"
         "        && row_end >= row_start && row_end <= " total-query-tokens ";\n"
         "    if (q_token >= row_start && q_token < row_end) batch = b;\n"
         "  }\n"
         "  const int q_position = q_positions[q_token];\n"
         "  if (!query_metadata_valid || batch < 0 || q_position < 0) {\n"
         invalid-result
         "  }\n"
         (route-initialization problem invalid-result)
         (visibility-initialization visibility invalid-result)
         (when (attention/interval-visibility? visibility)
           (str "  if (attention_begin == attention_end) {\n"
                empty-result
                "  }\n"))
         "  const int kv_head = q_head / " gqa-ratio ";\n"
         "  const long q_base = ((long)q_token * " q-heads
         " + q_head) * " qk-head-dim ";\n"
         "  float maximum = -3.402823466e+38f;\n"
         "  float denominator = 0.0f;\n"
         (visibility-loop-start visibility invalid-result)
         "    const long kv_position = (long)kv_start_position + token;\n"
         "    int visible = 1;\n"
         (visibility-statements (attention/position-filter visibility))
         "    if (!visible) continue;\n"
         "    const int logical_page = token / " page-size ";\n"
         "    const int physical_page = " (physical-page-expression route) ";\n"
         "    if (physical_page < 0 || physical_page >= " physical-pages ") {\n"
         invalid-result
         "    }\n"
         "    const int page_token = token - logical_page * " page-size ";\n"
         "    const long k_base = " k-base ";\n"
         "    const long v_base = " v-base ";\n"
         "    float partial_dot = 0.0f;\n"
         "    for (int x = (int)lane; x < " qk-head-dim "; x += " subgroup-size ")\n"
         "      partial_dot += " (load-float q-dtype "q[q_base + x]")
         " * convert_float(k_pages[k_base + x]);\n"
         "    const float dot = sub_group_reduce_add(partial_dot);\n"
         "    const float logit = dot * " (Float/toString (float scale)) "f;\n"
         "    float old_weight = 0.0f;\n"
         "    float new_weight = 0.0f;\n"
         "    if (lane == 0L) {\n"
         "      if (isnan(maximum) || isnan(logit)) {\n"
         "        maximum = NAN;\n"
         "        old_weight = NAN;\n"
         "        new_weight = NAN;\n"
         "      } else {\n"
         "        const float next_maximum = fmax(maximum, logit);\n"
         "        old_weight = exp(maximum - next_maximum);\n"
         "        new_weight = exp(logit - next_maximum);\n"
         "        maximum = next_maximum;\n"
         "      }\n"
         "      denominator = denominator * old_weight + new_weight;\n"
         "    }\n"
         "    old_weight = sub_group_broadcast(old_weight, 0);\n"
         "    new_weight = sub_group_broadcast(new_weight, 0);\n"
         (cooperative-component-update problem schedule)
         "  }\n"
         "  denominator = sub_group_broadcast(denominator, 0);\n"
         (cooperative-component-write
          problem schedule
          (fn [slot]
            (str "denominator == 0.0f ? 0.0f : accumulator" slot " / denominator")))
         "}\n")))

(defn- ordered-inputs
  [plan]
  (swr/ordered-input-ids plan))

(defn- ordered-abi
  [problem]
  (let [{:keys [query k-pages v-pages route visibility output q-dtype output-dtype]} problem
        common [(kabi/slot (:values query) :input q-dtype :c-name "q" :role :query)
                (kabi/slot (:row-offsets query) :input :int
                           :c-name "q_row_offsets" :role :query-rows)
                (kabi/slot (:positions query) :input :int
                           :c-name "q_positions" :role :query-positions)
                (kabi/slot k-pages :input :half :c-name "k_pages" :role :key-cache)
                (kabi/slot v-pages :input :half :c-name "v_pages" :role :value-cache)]
        route-slots
        (if (attention/dense-paged-route? route)
          [(kabi/slot (:page-table route) :input :int
                      :c-name "page_table" :role :page-routing)
           (kabi/slot (:lengths route) :input :int
                      :c-name "kv_lengths" :role :kv-lengths)
           (kabi/slot (:start-positions route) :input :int
                      :c-name "kv_start_positions" :role :kv-start-positions)]
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
     (conj (into (into common route-slots) visibility-slots)
           (kabi/slot output :output output-dtype :c-name "output" :role :result)))))

(defn emit-fp16-reference
  "Emit route-specialized attention with FP16 K/V storage as a verified KernelArtifact."
  [plan desc]
  (let [plan (reference-plan! plan)
        {:keys [query output q-heads value-head-dim route] :as problem}
        (attention-problem plan)
        name (kernel-name problem)
        workgroup-x (reference-workgroup-x plan desc)
        inputs (ordered-inputs plan)
        arguments (conj inputs output)
        launch (klaunch/spec
                {:workgroup-size [workgroup-x 1 1]
                 :group-count [(long (quot (+ value-head-dim (dec workgroup-x)) workgroup-x))
                               q-heads (:total-tokens query)]})]
    (kart/make
     {:kernel-name name
      :source (reference-source problem name)
      :abi (ordered-abi problem)
      :arguments arguments
      :launch launch
      :effects {:kind :attention :reads inputs :writes [output]}
      :provenance {:operation-id (:id problem) :semantic-op :attention
                   :algebra-plan-id (:id plan) :lowering :fp16-reference}
      :attributes {:strategy :fp16-reference :optimization-tier :reference
                   :algebra :segmented-weighted-reduction
                   :algebra-key (swr/algebra-key plan)
                   :storage-dtype :half :q-dtype (:q-dtype problem)
                   :output-dtype (:output-dtype problem) :accumulator-dtype :float
                   :route-kind (attention/route-kind route)
                   :visibility-kind (attention/visibility-kind (:visibility problem))
                   :k-layout (:k-layout problem) :v-layout (:v-layout problem)
                   :visibility (:visibility problem)
                   :layout (attention/layouts problem)
                   :complexity :quadratic-in-qk-head-dim}})))

(defn emit-fp16-cooperative
  "Emit one subgroup per query segment for routed FP16 K/V attention.

   The artifact preserves the reference leaf's complete ordered ABI and logical effects.  Only
   its target-neutral SegmentedWeightedReductionSchedule, launch mapping and target body differ."
  [plan schedule]
  (let [[plan schedule {:keys [query output q-heads route] :as problem}]
        (cooperative-plan! plan schedule)
        subgroup-size (:workgroup-size schedule)
        name (kernel-name problem schedule)
        inputs (ordered-inputs plan)
        arguments (conj inputs output)]
    (kart/make
     {:kernel-name name
      :source (cooperative-source problem schedule name)
      :abi (ordered-abi problem)
      :arguments arguments
      :launch (klaunch/spec
               {:workgroup-size [subgroup-size 1]
                :group-count [q-heads (:total-tokens query)]})
      :effects {:kind :attention :reads inputs :writes [output]}
      :provenance {:operation-id (:id problem) :semantic-op :attention
                   :algebra-plan-id (:id plan)
                   :lowering :subgroup-online-score-reuse}
      :attributes {:strategy :routed-paged-subgroup-online-score-reuse
                   :optimization-tier :subgroup
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
                   :materialized-intermediates []
                   :complexity :query-head-token-dot-plus-value}})))

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
      :nodes [(kgraph/->ScheduledKernel node-id artifact uses [])]
      :effects {:kind :attention :logical-visibility true :ordered-page-routing true}
      :provenance {:operation-id id :semantic-op :attention}
      :attributes {:strategy strategy :reference? reference?
                   :route-kind (attention/route-kind (:route problem))}})))
