(ns raster.vk.compute
  "Vulkan compute pipeline creation and dispatch."
  (:require [raster.vk.shader :as shader])
  (:import
   [org.lwjgl.vulkan
    VK13 VkDevice
    VkComputePipelineCreateInfo VkPipelineShaderStageCreateInfo
    VkPipelineLayoutCreateInfo VkDescriptorSetLayoutBinding
    VkDescriptorSetLayoutCreateInfo VkDescriptorPoolCreateInfo VkDescriptorPoolSize
    VkDescriptorSetAllocateInfo VkWriteDescriptorSet VkDescriptorBufferInfo
    VkCommandBuffer VkPushConstantRange]
   [org.lwjgl.system MemoryStack MemoryUtil]
   [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Descriptor set layout (for SSBO bindings)
;; ================================================================

(defn create-storage-descriptor-layout
  "Create a descriptor set layout with n storage buffer bindings.
  Returns layout handle (long)."
  ^long [^VkDevice device ^long n-bindings]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [bindings (VkDescriptorSetLayoutBinding/calloc (int n-bindings) stack)
            _ (dotimes [i n-bindings]
                (doto (.get bindings (int i))
                  (.binding (int i))
                  (.descriptorType VK13/VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                  (.descriptorCount 1)
                  (.stageFlags VK13/VK_SHADER_STAGE_COMPUTE_BIT)))
            ci (doto (VkDescriptorSetLayoutCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                 (.pBindings bindings))
            p-layout (.mallocLong stack 1)
            result (VK13/vkCreateDescriptorSetLayout device ci nil p-layout)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create descriptor set layout" {:result result})))
        (.get p-layout 0))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Descriptor pool + set
;; ================================================================

(defn create-descriptor-pool
  "Create a descriptor pool for storage buffers."
  ^long [^VkDevice device ^long max-sets ^long max-descriptors]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [pool-sizes (doto (VkDescriptorPoolSize/calloc 1 stack)
                         (-> (.get 0)
                             (.type VK13/VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                             (.descriptorCount (int max-descriptors))))
            ci (doto (VkDescriptorPoolCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                 (.maxSets (int max-sets))
                 (.pPoolSizes pool-sizes))
            p-pool (.mallocLong stack 1)
            result (VK13/vkCreateDescriptorPool device ci nil p-pool)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create descriptor pool" {:result result})))
        (.get p-pool 0))
      (finally
        (MemoryStack/stackPop)))))

(defn allocate-descriptor-set
  "Allocate a descriptor set from pool with the given layout."
  ^long [^VkDevice device ^long pool ^long layout]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ai (doto (VkDescriptorSetAllocateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                 (.descriptorPool pool)
                 (.pSetLayouts (.longs stack layout)))
            p-set (.mallocLong stack 1)
            result (VK13/vkAllocateDescriptorSets device ai p-set)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to allocate descriptor set" {:result result})))
        (.get p-set 0))
      (finally
        (MemoryStack/stackPop)))))

(defn update-descriptor-set!
  "Update descriptor set bindings to point to the given buffers.
  buffers: vector of {:buffer long :size long} maps, one per binding."
  [^VkDevice device ^long descriptor-set buffers]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [n (count buffers)
            writes (VkWriteDescriptorSet/calloc n stack)]
        (dotimes [i n]
          (let [{:keys [buffer size]} (nth buffers i)
                buf-info (doto (VkDescriptorBufferInfo/calloc 1 stack)
                           (-> (.get 0)
                               (.buffer (long buffer))
                               (.offset 0)
                               (.range (long size))))]
            (doto (.get writes (int i))
              (.sType VK13/VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
              (.dstSet descriptor-set)
              (.dstBinding (int i))
              (.descriptorCount 1)
              (.descriptorType VK13/VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
              (.pBufferInfo buf-info))))
        (VK13/vkUpdateDescriptorSets device writes nil))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Compute pipeline
;; ================================================================

(defn create-pipeline-layout
  "Create a pipeline layout with one descriptor set layout and optional push constants.
  push-constant-size: bytes for push constants (0 to disable)."
  [^VkDevice device desc-set-layout & {:keys [push-constant-size]
                                       :or {push-constant-size 0}}]
  (let [desc-set-layout (long desc-set-layout)
        stack (MemoryStack/stackPush)]
    (try
      (let [ci (doto (VkPipelineLayoutCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                 (.pSetLayouts (.longs stack desc-set-layout)))
            _ (when (pos? push-constant-size)
                (let [pc-range (VkPushConstantRange/calloc 1 stack)]
                  (doto (.get pc-range 0)
                    (.stageFlags VK13/VK_SHADER_STAGE_COMPUTE_BIT)
                    (.offset 0)
                    (.size (int push-constant-size)))
                  (.pPushConstantRanges ci pc-range)))
            p-layout (.mallocLong stack 1)
            result (VK13/vkCreatePipelineLayout device ci nil p-layout)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create pipeline layout" {:result result})))
        (.get p-layout 0))
      (finally
        (MemoryStack/stackPop)))))

(defn create-compute-pipeline
  "Create a compute pipeline from GLSL source.
  Returns {:pipeline long :layout long :desc-layout long :shader-module long
           :spirv ByteBuffer :push-constant-size int}.
  push-constant-size: bytes for push constants (default 0)."
  [^VkDevice device ^String glsl-source n-bindings
   & {:keys [push-constant-size] :or {push-constant-size 0}}]
  (let [n-bindings (long n-bindings)
        stack (MemoryStack/stackPush)]
    (try
      (let [{:keys [module spirv]} (shader/compile-shader-module device glsl-source :compute)
            desc-layout (create-storage-descriptor-layout device n-bindings)
            layout (create-pipeline-layout device desc-layout
                                           :push-constant-size push-constant-size)
            stage-ci (doto (VkPipelineShaderStageCreateInfo/calloc stack)
                       (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                       (.stage VK13/VK_SHADER_STAGE_COMPUTE_BIT)
                       (.module module)
                       (.pName (.UTF8 stack "main")))
            ci (doto (VkComputePipelineCreateInfo/calloc 1 stack)
                 (-> (.get 0)
                     (.sType VK13/VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                     (.stage stage-ci)
                     (.layout layout)))
            p-pipeline (.mallocLong stack 1)
            result (VK13/vkCreateComputePipelines device 0 ci nil p-pipeline)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create compute pipeline" {:result result})))
        {:pipeline (.get p-pipeline 0)
         :layout layout
         :desc-layout desc-layout
         :shader-module module
         :spirv spirv
         :push-constant-size push-constant-size})
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Push constants
;; ================================================================

(defn cmd-push-constants!
  "Record push constant update. data is a direct ByteBuffer."
  [^VkCommandBuffer cb ^long layout ^ByteBuffer data]
  (VK13/vkCmdPushConstants cb layout
                           VK13/VK_SHADER_STAGE_COMPUTE_BIT 0 data))

;; ================================================================
;; Dispatch
;; ================================================================

(defn cmd-dispatch!
  "Record a compute dispatch into a command buffer.
  push-constants: optional direct ByteBuffer of push constant data."
  [^VkCommandBuffer cb pipeline-info descriptor-set group-counts
   & {:keys [push-constants]}]
  (let [pipeline (long (:pipeline pipeline-info))
        layout (long (:layout pipeline-info))
        descriptor-set (long descriptor-set)
        gx (int (nth group-counts 0))
        gy (int (nth group-counts 1 1))
        gz (int (nth group-counts 2 1))]
    (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_COMPUTE pipeline)
    (let [stack (MemoryStack/stackPush)]
      (try
        (VK13/vkCmdBindDescriptorSets cb VK13/VK_PIPELINE_BIND_POINT_COMPUTE
                                      layout 0 (.longs stack descriptor-set) nil)
        (finally
          (MemoryStack/stackPop))))
    (when push-constants
      (cmd-push-constants! cb layout push-constants))
    (VK13/vkCmdDispatch cb gx gy gz)))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-compute-pipeline!
  [^VkDevice device pipeline-info]
  (VK13/vkDestroyPipeline device (:pipeline pipeline-info) nil)
  (VK13/vkDestroyPipelineLayout device (:layout pipeline-info) nil)
  (VK13/vkDestroyDescriptorSetLayout device (:desc-layout pipeline-info) nil)
  (shader/destroy-shader-module! device (:shader-module pipeline-info))
  (when-let [spirv (:spirv pipeline-info)]
    (MemoryUtil/memFree spirv)))

(defn destroy-descriptor-pool! [^VkDevice device ^long pool]
  (VK13/vkDestroyDescriptorPool device pool nil))

;; ================================================================
;; Kernel Registry — cache compiled pipelines by source hash
;; ================================================================

(defonce ^:private kernel-registry (atom {}))

(defn get-or-create-pipeline!
  "Get a cached compute pipeline or compile and cache a new one.
  kernel-gen: map from generate-compute-map-kernel / generate-compute-map-void-kernel.
  Returns pipeline-info map."
  [^VkDevice device kernel-gen]
  (let [source (:source kernel-gen)
        cache-key (hash source)]
    (or (get @kernel-registry cache-key)
        (let [pipeline (create-compute-pipeline
                        device source
                        (:n-bindings kernel-gen)
                        :push-constant-size (:push-constant-size kernel-gen))]
          (swap! kernel-registry assoc cache-key
                 (merge pipeline (select-keys kernel-gen
                                              [:array-params :scalar-params :soa-expansions
                                               :written-arrays :out-binding :dtype :workgroup-size])))
          (get @kernel-registry cache-key)))))

(defn clear-kernel-registry!
  "Destroy all cached pipelines and clear the registry."
  [^VkDevice device]
  (doseq [[_ info] @kernel-registry]
    (destroy-compute-pipeline! device info))
  (reset! kernel-registry {}))
