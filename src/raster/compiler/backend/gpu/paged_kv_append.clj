(ns raster.compiler.backend.gpu.paged-kv-append
  "C-family target lowering for paged FP32-to-FP16 K/V assignment."
  (:require [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.paged-kv-append :as append]
            [raster.compiler.passes.parallel.paged-kv-append-body :as append-body]))

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

(defn- ordered-abi
  [problem]
  (let [{:keys [key-rows value-rows slot-mapping key-pages value-pages]}
        (append/validate! problem)]
    (kabi/validate!
     [(kabi/slot key-rows :input :float :c-name "key_rows" :role :key-rows)
      (kabi/slot value-rows :input :float :c-name "value_rows" :role :value-rows)
      (kabi/slot slot-mapping :input :int :c-name "slot_mapping" :role :slot-mapping)
      (kabi/slot key-pages :inout :half :c-name "key_pages" :role :key-pages)
      (kabi/slot value-pages :inout :half :c-name "value_pages" :role :value-pages)])))

(defn emit-fp32-to-fp16-reference
  "Schedule the portable assignment as KernelBody and emit one C-family artifact."
  ([problem desc]
   (emit-fp32-to-fp16-reference problem desc :opencl-intel))
  ([problem desc target-dialect]
   (let [{:keys [id] :as problem} (append/validate! problem)
         name (kernel-name problem)
         workgroup-x (reference-workgroup-x problem desc)
         inputs (append/ordered-input-buffer-ids problem)
         outputs (append/ordered-output-buffer-ids problem)
         kernel-body (append-body/lower problem workgroup-x)
         base-abi (ordered-abi problem)
         abi (body-abi/project-contracts base-abi kernel-body)
         dialect (c-dialect/resolve! target-dialect)
         parameter-names (into {} (map (juxt :name :c-name)) base-abi)]
     (artifact/make
      {:kernel-name name
       :target (c-dialect/target dialect)
       :source (emitter/emit-scalar-kernel
                name kernel-body
                {:target-dialect target-dialect :parameter-names parameter-names})
       :abi abi
       :arguments (into inputs outputs)
       :launch (:launch kernel-body)
       :effects {:kind :paged-kv-append :reads inputs :writes outputs}
       :provenance {:operation-id id :semantic-op :paged-kv-append
                    :lowering :kernel-body-fp32-to-fp16}
       :attributes {:strategy :fp32-to-fp16-reference
                    :optimization-tier :reference
                    :assignment :unique-slot
                    :rounding-mode :round-to-nearest-even
                    :input-dtype :float :storage-dtype :half
                    :kernel-body kernel-body
                    :target-dialect target-dialect}}))))

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
                          (map #(graph/->ValueUse % :read-write) outputs)))]
    (graph/make
     {:inputs (into input-buffers output-buffers)
      :outputs output-buffers
      :abi (:abi emitted)
      :arguments (:arguments emitted)
      :scalars []
      :nodes [(graph/->ScheduledKernel
               [:paged-kv-append id :fp32-to-fp16-reference]
               emitted uses #{} [])]
      :effects {:kind :paged-kv-append :assignment :unique-slot}
      :provenance {:operation-id id :semantic-op :paged-kv-append}
      :attributes {:strategy :fp32-to-fp16-reference :reference? true}})))
