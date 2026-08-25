(ns raster.compiler.backend.gpu.paged-kv-append
  "Portable OpenCL-C reference lowering for paged FP32-to-FP16 K/V assignment."
  (:require [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.paged-kv-append :as append]))

(defn- kernel-name
  [problem]
  (let [identity (select-keys problem
                              [:batch-size :key-elements-per-token
                               :value-elements-per-token :page-size :physical-pages
                               :key-input-dtype :value-input-dtype
                               :key-storage-dtype :value-storage-dtype :rounding-mode])]
    (format "raster_paged_kv_append_%08x"
            (bit-and 0xffffffff (long (hash identity))))))

(defn reference-workgroup-x
  "Choose the reference kernel's x workgroup from row width and device limits."
  [problem desc]
  (let [{:keys [key-elements-per-token value-elements-per-token]}
        (append/validate! problem)
        width (max key-elements-per-token value-elements-per-token)
        subgroup (long (or (:subgroup-size desc) 16))
        maximum (long (or (:max-workgroup-size desc) 256))]
    (long (max 1 (min width subgroup maximum)))))

(defn- source
  [problem name]
  (let [{:keys [batch-size key-elements-per-token value-elements-per-token]}
        (append/validate! problem)
        slots (append/physical-slots problem)]
    (str "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "__kernel void " name "(\n"
         "    __global const float* key_rows,\n"
         "    __global const float* value_rows,\n"
         "    __global const int* slot_mapping,\n"
         "    __global half* key_pages,\n"
         "    __global half* value_pages) {\n"
         "  const int component = (int)get_global_id(0);\n"
         "  const int lane = (int)get_global_id(1);\n"
         "  if (lane >= " batch-size ") return;\n"
         "  const int slot = slot_mapping[lane];\n"
         "  if (slot < 0 || (long)slot >= " slots "L) return;\n"
         "  if (component < " key-elements-per-token ") {\n"
         "    const long src = (long)lane * " key-elements-per-token " + component;\n"
         "    const long dst = (long)slot * " key-elements-per-token " + component;\n"
         "    key_pages[dst] = convert_half_rte(key_rows[src]);\n"
         "  }\n"
         "  if (component < " value-elements-per-token ") {\n"
         "    const long src = (long)lane * " value-elements-per-token " + component;\n"
         "    const long dst = (long)slot * " value-elements-per-token " + component;\n"
         "    value_pages[dst] = convert_half_rte(value_rows[src]);\n"
         "  }\n"
         "}\n")))

(defn- ordered-abi
  [problem]
  (let [{:keys [key-rows value-rows slot-mapping key-pages value-pages]}
        (append/validate! problem)]
    (kabi/validate!
     [(kabi/slot key-rows :input :float :c-name "key_rows" :role :key-rows)
      (kabi/slot value-rows :input :float :c-name "value_rows" :role :value-rows)
      (kabi/slot slot-mapping :input :int :c-name "slot_mapping" :role :slot-mapping)
      (kabi/slot key-pages :output :half :c-name "key_pages" :role :key-pages)
      (kabi/slot value-pages :output :half :c-name "value_pages" :role :value-pages)])))

(defn emit-fp32-to-fp16-reference
  "Emit the portable assignment kernel as a verified KernelArtifact."
  [problem desc]
  (let [{:keys [id batch-size key-elements-per-token value-elements-per-token]
         :as problem} (append/validate! problem)
        name (kernel-name problem)
        workgroup-x (reference-workgroup-x problem desc)
        width (max key-elements-per-token value-elements-per-token)
        inputs (append/ordered-input-buffer-ids problem)
        outputs (append/ordered-output-buffer-ids problem)]
    (artifact/make
     {:kernel-name name
      :source (source problem name)
      :abi (ordered-abi problem)
      :arguments (into inputs outputs)
      :launch (launch/spec
               {:workgroup-size [workgroup-x 1]
                :group-count [(long (quot (+ width (dec workgroup-x)) workgroup-x))
                              batch-size]})
      :effects {:kind :paged-kv-append :reads inputs :writes outputs}
      :provenance {:operation-id id :semantic-op :paged-kv-append
                   :lowering :fp32-to-fp16-reference}
      :attributes {:strategy :fp32-to-fp16-reference
                   :optimization-tier :reference
                   :assignment :unique-slot
                   :rounding-mode :round-to-nearest-even
                   :input-dtype :float :storage-dtype :half}})))

(defn kernel-graph
  "Wrap one append artifact in a verified graph with explicit in-place page effects."
  [problem emitted]
  (let [{:keys [id] :as problem} (append/validate! problem)
        emitted (artifact/validate! emitted)
        specs (append/buffer-specs problem)
        inputs (append/ordered-input-buffer-ids problem)
        outputs (append/ordered-output-buffer-ids problem)
        buffer (fn [buffer-id]
                 (let [{:keys [dtype elements role]} (get specs buffer-id)]
                   (graph/buffer buffer-id dtype elements :device role)))
        input-buffers (mapv buffer inputs)
        output-buffers (mapv buffer outputs)
        uses (vec (concat (map #(graph/->ValueUse % :read) inputs)
                          (map #(graph/->ValueUse % :write) outputs)))]
    (graph/make
     {:inputs (into input-buffers output-buffers)
      :outputs output-buffers
      :abi (:abi emitted)
      :arguments (:arguments emitted)
      :nodes [(graph/->ScheduledKernel
               [:paged-kv-append id :fp32-to-fp16-reference]
               emitted uses [])]
      :effects {:kind :paged-kv-append :assignment :unique-slot}
      :provenance {:operation-id id :semantic-op :paged-kv-append}
      :attributes {:strategy :fp32-to-fp16-reference :reference? true}})))
