(ns raster.compiler.backend.gpu.paged-attention
  "Portable reference lowering for decode-time paged attention.

   This kernel is an executable semantic oracle, not the eventual fast leaf: one work-item owns
   one output element and recomputes the query/key dot product for that element. That deliberately
   simple O(B*Hq*L*D^2) schedule makes page routing, GQA, FP16 storage, FP32 online softmax and the
   ordered ABI testable through every runtime before flash/persistent schedules are introduced."
  (:require [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.paged-attention :as paged]))

(defn reference-workgroup-x
  "Choose the reference leaf's x workgroup from the hardware resource descriptor."
  [operation desc]
  (let [head-dim (:head-dim (paged/validate! operation))
        subgroup (long (or (:subgroup-size desc) 16))
        maximum (long (or (:max-workgroup-size desc) 256))]
    (long (max 1 (min head-dim subgroup maximum)))))

(defn- kernel-name
  [operation]
  (let [identity (select-keys operation
                              [:batch-size :q-heads :kv-heads :head-dim :page-size
                               :physical-pages :pages-per-sequence :scale :cache-format
                               :cache-layout])]
    (format "raster_paged_attention_fp16_%08x" (bit-and 0xffffffff (long (hash identity))))))

(defn- source
  [operation name]
  (let [{:keys [batch-size q-heads kv-heads head-dim page-size physical-pages
                pages-per-sequence scale cache-layout]} operation
        gqa-ratio (quot q-heads kv-heads)
        cache-base (case cache-layout
                     :kv-head-major
                     (str "(((kv_head * " physical-pages " + physical_page) * "
                          page-size " + page_token) * " head-dim ")")
                     :page-major
                     (str "(((physical_page * " page-size " + page_token) * "
                          kv-heads " + kv_head) * " head-dim ")"))]
    (str "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "__kernel void " name "(\n"
         "    __global const half* q,\n"
         "    __global const half* k_pages,\n"
         "    __global const half* v_pages,\n"
         "    __global const int* page_table,\n"
         "    __global const int* lengths,\n"
         "    __global half* output) {\n"
         "  const int d = (int)get_global_id(0);\n"
         "  const int q_head = (int)get_global_id(1);\n"
         "  const int batch = (int)get_global_id(2);\n"
         "  if (d >= " head-dim " || q_head >= " q-heads " || batch >= " batch-size ") return;\n"
         "  const int out_index = (batch * " q-heads " + q_head) * " head-dim " + d;\n"
         "  const int length = lengths[batch];\n"
         "  if (length < 0 || length > " (* pages-per-sequence page-size) ") {\n"
         "    output[out_index] = convert_half_rte(NAN);\n"
         "    return;\n"
         "  }\n"
         "  if (length == 0) { output[out_index] = (half)0; return; }\n"
         "  const int kv_head = q_head / " gqa-ratio ";\n"
         "  const int q_base = (batch * " q-heads " + q_head) * " head-dim ";\n"
         "  float maximum = -3.402823466e+38f;\n"
         "  float denominator = 0.0f;\n"
         "  float accumulator = 0.0f;\n"
         "  for (int token = 0; token < length; ++token) {\n"
         "    const int logical_page = token / " page-size ";\n"
         "    const int physical_page = page_table[batch * " pages-per-sequence
         " + logical_page];\n"
         "    if (physical_page < 0 || physical_page >= " physical-pages ") {\n"
         "      output[out_index] = convert_half_rte(NAN);\n"
         "      return;\n"
         "    }\n"
         "    const int page_token = token - logical_page * " page-size ";\n"
         "    const int cache_base = " cache-base ";\n"
         "    float dot = 0.0f;\n"
         "    for (int x = 0; x < " head-dim "; ++x) {\n"
         "      dot += convert_float(q[q_base + x]) * convert_float(k_pages[cache_base + x]);\n"
         "    }\n"
         "    const float logit = dot * " (Float/toString (float scale)) "f;\n"
         "    const float next_maximum = fmax(maximum, logit);\n"
         "    const float old_weight = exp(maximum - next_maximum);\n"
         "    const float new_weight = exp(logit - next_maximum);\n"
         "    accumulator = accumulator * old_weight\n"
         "                  + convert_float(v_pages[cache_base + d]) * new_weight;\n"
         "    denominator = denominator * old_weight + new_weight;\n"
         "    maximum = next_maximum;\n"
         "  }\n"
         "  output[out_index] = convert_half_rte(accumulator / denominator);\n"
         "}\n")))

(defn emit-fp16-reference
  "Emit the plain-FP16 cache reference leaf as one verified KernelArtifact."
  [operation desc]
  (let [{:keys [q k-pages v-pages page-table lengths output batch-size q-heads head-dim]
         :as operation} (paged/validate! operation)
        name (kernel-name operation)
        workgroup-x (reference-workgroup-x operation desc)
        abi (kabi/validate!
             [(kabi/slot q :input :half :c-name "q" :role :query)
              (kabi/slot k-pages :input :half :c-name "k_pages" :role :key-cache)
              (kabi/slot v-pages :input :half :c-name "v_pages" :role :value-cache)
              (kabi/slot page-table :input :int :c-name "page_table" :role :page-routing)
              (kabi/slot lengths :input :int :c-name "lengths" :role :sequence-length)
              (kabi/slot output :output :half :c-name "output" :role :result)])
        launch (klaunch/spec
                {:workgroup-size [workgroup-x 1 1]
                 :group-count [(long (quot (+ head-dim (dec workgroup-x)) workgroup-x))
                               q-heads batch-size]})]
    (kart/make
     {:kernel-name name
      :source (source operation name)
      :abi abi
      :arguments [q k-pages v-pages page-table lengths output]
      :launch launch
      :effects {:kind :paged-attention :reads [q k-pages v-pages page-table lengths]
                :writes [output]}
      :provenance {:operation-id (:id operation) :semantic-op :paged-attention
                   :lowering :fp16-reference}
      :attributes {:strategy :fp16-reference :optimization-tier :reference
                   :storage-dtype :half :accumulator-dtype :float
                   :cache-layout (:cache-layout operation)
                   :layout (paged/layouts operation)
                   :complexity :quadratic-in-head-dim}})))

(defn kernel-graph
  "Wrap the reference artifact in an explicit one-node scheduled graph."
  [operation artifact]
  (let [{:keys [q k-pages v-pages page-table lengths output id]} (paged/validate! operation)
        artifact (kart/validate! artifact)
        specs (paged/buffer-specs operation)
        graph-buffer (fn [buffer-id]
                       (let [{:keys [dtype elements role]} (get specs buffer-id)]
                         (kgraph/buffer buffer-id dtype elements :device role)))
        inputs [q k-pages v-pages page-table lengths]
        uses (vec (concat (map #(kgraph/->ValueUse % :read) inputs)
                          [(kgraph/->ValueUse output :write)]))
        node-id [:paged-attention id :fp16-reference]]
    (kgraph/make
     {:inputs (mapv graph-buffer inputs)
      :outputs [(graph-buffer output)]
      :nodes [(kgraph/->ScheduledKernel node-id artifact uses [])]
      :effects {:kind :paged-attention :ordered-page-routing true}
      :provenance {:operation-id id :semantic-op :paged-attention}
      :attributes {:strategy :fp16-reference :reference? true}})))
