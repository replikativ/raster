(ns raster.compiler.backend.gpu.attention
  "Portable FP16 reference lowering for logical attention over dense or CSR paged KV routes.

   One work-item owns one output component and recomputes QK. This intentionally slow schedule is
   the executable semantic oracle for packed queries, GQA, independent K/V layouts and dimensions,
   logical interval/CSR visibility, and physical page routing."
  (:require [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]))

(defn reference-workgroup-x
  "Choose the reference leaf's x workgroup from output width and hardware resources."
  [problem desc]
  (let [value-head-dim (:value-head-dim (attention/validate! problem))
        subgroup (long (or (:subgroup-size desc) 16))
        maximum (long (or (:max-workgroup-size desc) 256))]
    (long (max 1 (min value-head-dim subgroup maximum)))))

(defn- kernel-name
  [problem]
  (let [identity (select-keys problem
                              [:batch-size :q-heads :kv-heads :qk-head-dim :value-head-dim
                               :page-size :physical-pages :scale :k-format :v-format
                               :k-layout :v-layout])
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
    (format "raster_attention_fp16_%08x" (bit-and 0xffffffff (long (hash identity))))))

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
  [{:keys [route page-size]}]
  (if (attention/dense-paged-route? route)
    (let [capacity (* (:pages-per-sequence route) page-size)]
      (str "  const int length = kv_lengths[batch];\n"
           "  const int kv_start_position = kv_start_positions[batch];\n"
           "  if (length < 0 || length > " capacity
           " || kv_start_position < 0\n"
           "      || (long)kv_start_position + (long)length > 2147483648L) {\n"
           "    output[out_index] = convert_half_rte(NAN);\n"
           "    return;\n"
           "  }\n"))
    (let [capacity (:page-index-capacity route)]
      (str "  const int page_begin = page_offsets[batch];\n"
           "  const int page_end = page_offsets[batch + 1];\n"
           "  const int final_page_length = last_page_lengths[batch];\n"
           "  const int kv_start_position = kv_start_positions[batch];\n"
           "  if (page_offsets[0] != 0 || page_begin < 0 || page_end < page_begin\n"
           "      || page_end > " capacity " || kv_start_position < 0) {\n"
           "    output[out_index] = convert_half_rte(NAN);\n"
           "    return;\n"
           "  }\n"
           "  const int routed_page_count = page_end - page_begin;\n"
           "  if ((routed_page_count == 0 && final_page_length != 0)\n"
           "      || (routed_page_count > 0\n"
           "          && (final_page_length < 1 || final_page_length > " page-size "))) {\n"
           "    output[out_index] = convert_half_rte(NAN);\n"
           "    return;\n"
           "  }\n"
           "  const int length = routed_page_count == 0 ? 0\n"
           "      : (routed_page_count - 1) * " page-size " + final_page_length;\n"
           "  if ((long)kv_start_position + (long)length > 2147483648L) {\n"
           "    output[out_index] = convert_half_rte(NAN);\n"
           "    return;\n"
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
  [visibility]
  (when (attention/csr-visibility? visibility)
    (str "  const int attention_begin = attention_row_offsets[q_token];\n"
         "  const int attention_end = attention_row_offsets[q_token + 1];\n"
         "  if (attention_row_offsets[0] != 0 || attention_begin < 0\n"
         "      || attention_end < attention_begin || attention_end > "
         (:key-index-capacity visibility) ") {\n"
         "    output[out_index] = convert_half_rte(NAN);\n"
         "    return;\n"
         "  }\n")))

(defn- visibility-loop-start
  [visibility]
  (if (attention/csr-visibility? visibility)
    (str "  for (int edge = attention_begin; edge < attention_end; ++edge) {\n"
         "    const int token = attention_key_indices[edge];\n"
         "    if (token < 0 || token >= length) {\n"
         "      output[out_index] = convert_half_rte(NAN);\n"
         "      return;\n"
         "    }\n")
    "  for (int token = 0; token < length; ++token) {\n"))

(defn- source
  [problem name]
  (let [{:keys [query route batch-size q-heads kv-heads qk-head-dim value-head-dim
                page-size physical-pages scale k-layout v-layout visibility]} problem
        total-query-tokens (:total-tokens query)
        gqa-ratio (quot q-heads kv-heads)
        k-base (cache-base problem k-layout qk-head-dim)
        v-base (cache-base problem v-layout value-head-dim)]
    (str "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "__kernel void " name "(\n"
         "    __global const half* q,\n"
         "    __global const int* q_row_offsets,\n"
         "    __global const int* q_positions,\n"
         "    __global const half* k_pages,\n"
         "    __global const half* v_pages,\n"
         (route-signature route)
         (visibility-signature visibility)
         "    __global half* output) {\n"
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
         "    output[out_index] = convert_half_rte(NAN);\n"
         "    return;\n"
         "  }\n"
         (route-initialization problem)
         (visibility-initialization visibility)
         "  const int kv_head = q_head / " gqa-ratio ";\n"
         "  const long q_base = ((long)q_token * " q-heads
         " + q_head) * " qk-head-dim ";\n"
         "  float maximum = -3.402823466e+38f;\n"
         "  float denominator = 0.0f;\n"
         "  float accumulator = 0.0f;\n"
         (visibility-loop-start visibility)
         "    const long kv_position = (long)kv_start_position + token;\n"
         "    int visible = 1;\n"
         (visibility-statements (attention/position-filter visibility))
         "    if (!visible) continue;\n"
         "    const int logical_page = token / " page-size ";\n"
         "    const int physical_page = " (physical-page-expression route) ";\n"
         "    if (physical_page < 0 || physical_page >= " physical-pages ") {\n"
         "      output[out_index] = convert_half_rte(NAN);\n"
         "      return;\n"
         "    }\n"
         "    const int page_token = token - logical_page * " page-size ";\n"
         "    const long k_base = " k-base ";\n"
         "    const long v_base = " v-base ";\n"
         "    float dot = 0.0f;\n"
         "    for (int x = 0; x < " qk-head-dim "; ++x) {\n"
         "      dot += convert_float(q[q_base + x]) * convert_float(k_pages[k_base + x]);\n"
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
         "  output[out_index] = denominator == 0.0f ? (half)0\n"
         "      : convert_half_rte(accumulator / denominator);\n"
         "}\n")))

(defn- ordered-inputs
  [problem]
  (let [{:keys [query k-pages v-pages route visibility]} problem
        common [(:values query) (:row-offsets query) (:positions query) k-pages v-pages]
        route-inputs (if (attention/dense-paged-route? route)
                       [(:page-table route) (:lengths route) (:start-positions route)]
                       [(:page-offsets route) (:page-indices route) (:last-page-lengths route)
                        (:start-positions route)])]
    (into (into common route-inputs) (attention/visibility-buffer-ids visibility))))

(defn- ordered-abi
  [problem]
  (let [{:keys [query k-pages v-pages route visibility output]} problem
        common [(kabi/slot (:values query) :input :half :c-name "q" :role :query)
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
           (kabi/slot output :output :half :c-name "output" :role :result)))))

(defn emit-fp16-reference
  "Emit a route-specialized plain-FP16 attention reference as a verified KernelArtifact."
  [problem desc]
  (let [{:keys [query output q-heads value-head-dim route] :as problem}
        (attention/validate! problem)
        name (kernel-name problem)
        workgroup-x (reference-workgroup-x problem desc)
        inputs (ordered-inputs problem)
        arguments (conj inputs output)
        launch (klaunch/spec
                {:workgroup-size [workgroup-x 1 1]
                 :group-count [(long (quot (+ value-head-dim (dec workgroup-x)) workgroup-x))
                               q-heads (:total-tokens query)]})]
    (kart/make
     {:kernel-name name
      :source (source problem name)
      :abi (ordered-abi problem)
      :arguments arguments
      :launch launch
      :effects {:kind :attention :reads inputs :writes [output]}
      :provenance {:operation-id (:id problem) :semantic-op :attention
                   :lowering :fp16-reference}
      :attributes {:strategy :fp16-reference :optimization-tier :reference
                   :storage-dtype :half :accumulator-dtype :float
                   :route-kind (attention/route-kind route)
                   :visibility-kind (attention/visibility-kind (:visibility problem))
                   :k-layout (:k-layout problem) :v-layout (:v-layout problem)
                   :visibility (:visibility problem)
                   :layout (attention/layouts problem)
                   :complexity :quadratic-in-qk-head-dim}})))

(defn kernel-graph
  "Wrap the reference artifact in an explicit one-node scheduled graph."
  [problem artifact]
  (let [{:keys [output id] :as problem} (attention/validate! problem)
        artifact (kart/validate! artifact)
        specs (attention/buffer-specs problem)
        inputs (ordered-inputs problem)
        graph-buffer (fn [buffer-id]
                       (let [{:keys [dtype elements role]} (get specs buffer-id)]
                         (kgraph/buffer buffer-id dtype elements :device role)))
        uses (vec (concat (map #(kgraph/->ValueUse % :read) inputs)
                          [(kgraph/->ValueUse output :write)]))
        node-id [:attention id :fp16-reference]]
    (kgraph/make
     {:inputs (mapv graph-buffer inputs)
      :outputs [(graph-buffer output)]
      :nodes [(kgraph/->ScheduledKernel node-id artifact uses [])]
      :effects {:kind :attention :logical-visibility true :ordered-page-routing true}
      :provenance {:operation-id id :semantic-op :attention}
      :attributes {:strategy :fp16-reference :reference? true
                   :route-kind (attention/route-kind (:route problem))}})))
