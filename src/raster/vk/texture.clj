(ns raster.vk.texture
  "Texture loading, GPU upload, and sampling for Vulkan.
  Supports loading images via STB, creating VkImage + VkImageView + VkSampler."
  (:require
   [raster.vk.core :as vk]
   [raster.vk.mem :as mem])
  (:import
   [org.lwjgl.stb STBImage]
   [org.lwjgl.vulkan
    VK13 VkDevice VkCommandBuffer
    VkImageCreateInfo VkImageViewCreateInfo
    VkSamplerCreateInfo VkBufferImageCopy
    VkDescriptorSetLayoutBinding VkDescriptorPoolSize
    VkDescriptorImageInfo VkWriteDescriptorSet]
   [org.lwjgl.util.vma Vma VmaAllocationCreateInfo]
   [org.lwjgl.system MemoryStack MemoryUtil]
   [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Image loading (STB)
;; ================================================================

(defn load-image
  "Load an image from disk via STB. Returns {:pixels ByteBuffer :width int :height int :channels int}.
  Caller must free pixels with STBImage/stbi_image_free."
  [^String path]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [pw (.mallocInt stack 1)
            ph (.mallocInt stack 1)
            pc (.mallocInt stack 1)
            pixels (STBImage/stbi_load path pw ph pc 4)] ;; force RGBA
        (when-not pixels
          (throw (ex-info (str "Failed to load image: " (STBImage/stbi_failure_reason))
                          {:path path})))
        {:pixels pixels
         :width (.get pw 0)
         :height (.get ph 0)
         :channels 4})
      (finally
        (MemoryStack/stackPop)))))

(defn load-image-from-memory
  "Load an image from a ByteBuffer (e.g. embedded resource). Returns same as load-image."
  [^ByteBuffer data]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [pw (.mallocInt stack 1)
            ph (.mallocInt stack 1)
            pc (.mallocInt stack 1)
            pixels (STBImage/stbi_load_from_memory data pw ph pc 4)]
        (when-not pixels
          (throw (ex-info (str "Failed to load image from memory: " (STBImage/stbi_failure_reason)) {})))
        {:pixels pixels
         :width (.get pw 0)
         :height (.get ph 0)
         :channels 4})
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; GPU texture creation
;; ================================================================

(defrecord Texture [image allocation view sampler width height])

(defn- cmd-image-barrier-color!
  "Image layout transition for color images."
  [^VkCommandBuffer cb image old-layout new-layout src-access dst-access src-stage dst-stage]
  (let [image (long image)
        old-layout (int old-layout)
        new-layout (int new-layout)
        src-access (int src-access)
        dst-access (int dst-access)
        src-stage (int src-stage)
        dst-stage (int dst-stage)
        stack (MemoryStack/stackPush)]
    (try
      (let [^org.lwjgl.vulkan.VkImageMemoryBarrier b
            (.get (org.lwjgl.vulkan.VkImageMemoryBarrier/calloc 1 stack) 0)
            _ (doto b
                (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                (.oldLayout old-layout)
                (.newLayout new-layout)
                (.srcQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
                (.dstQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
                (.image image)
                (.srcAccessMask src-access)
                (.dstAccessMask dst-access))
            _ (doto (.subresourceRange b)
                (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
                (.baseMipLevel 0) (.levelCount 1)
                (.baseArrayLayer 0) (.layerCount 1))
            barrier (org.lwjgl.vulkan.VkImageMemoryBarrier/create (.address b) 1)]
        (VK13/vkCmdPipelineBarrier cb src-stage dst-stage 0 nil nil barrier))
      (finally
        (MemoryStack/stackPop)))))

(defn create-texture
  "Create a GPU texture from pixel data. Uploads via staging buffer + one-shot command buffer.
  ctx: GpuContext (or map with :device :allocator :queue :command-pool).
  image-data: {:pixels ByteBuffer :width int :height int} (from load-image).
  filter-mode: :nearest or :linear (default :linear).
  address-mode: :repeat (default), :clamp, :mirror.
  Returns Texture record."
  [ctx image-data & {:keys [filter-mode address-mode]
                     :or {filter-mode :linear address-mode :repeat}}]
  (let [{:keys [^ByteBuffer pixels width height]} image-data
        width (int width)
        height (int height)
        image-size (* width height 4)
        device ^VkDevice (:device ctx)
        allocator (long (:allocator ctx))
        queue (:queue ctx)
        cmd-pool (long (:command-pool ctx))

        ;; Create staging buffer
        staging (mem/create-buffer allocator image-size :staging :cpu-to-gpu)
        _ (mem/with-mapped-buffer [bb allocator staging]
            (let [src (.duplicate pixels)]
              (.put ^java.nio.ByteBuffer bb src)))

        ;; Create GPU image
        stack (MemoryStack/stackPush)
        img-ci (doto (VkImageCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                 (.imageType VK13/VK_IMAGE_TYPE_2D)
                 (.format VK13/VK_FORMAT_R8G8B8A8_SRGB)
                 (-> .extent (.width width) (.height height) (.depth 1))
                 (.mipLevels 1) (.arrayLayers 1)
                 (.samples VK13/VK_SAMPLE_COUNT_1_BIT)
                 (.tiling VK13/VK_IMAGE_TILING_OPTIMAL)
                 (.usage (bit-or VK13/VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                 VK13/VK_IMAGE_USAGE_SAMPLED_BIT))
                 (.sharingMode VK13/VK_SHARING_MODE_EXCLUSIVE)
                 (.initialLayout VK13/VK_IMAGE_LAYOUT_UNDEFINED))
        alloc-ci (doto (VmaAllocationCreateInfo/calloc stack)
                   (.usage (int Vma/VMA_MEMORY_USAGE_GPU_ONLY)))
        p-image (.mallocLong stack 1)
        p-alloc (.mallocPointer stack 1)
        r (Vma/vmaCreateImage allocator img-ci alloc-ci p-image p-alloc nil)
        _ (when-not (= r VK13/VK_SUCCESS)
            (throw (ex-info "Failed to create texture image" {:result r})))
        image (.get p-image 0)
        allocation (.get p-alloc 0)]

    (MemoryStack/stackPop)

    ;; Upload via one-shot command buffer
    (let [cb (vk/allocate-command-buffer device cmd-pool)]
      (vk/begin-command-buffer cb)

      ;; Transition: undefined -> transfer dst
      (cmd-image-barrier-color! cb image
                                VK13/VK_IMAGE_LAYOUT_UNDEFINED
                                VK13/VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                0 VK13/VK_ACCESS_TRANSFER_WRITE_BIT
                                VK13/VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                VK13/VK_PIPELINE_STAGE_TRANSFER_BIT)

      ;; Copy buffer -> image
      (let [stack2 (MemoryStack/stackPush)
            ^VkBufferImageCopy region (.get (VkBufferImageCopy/calloc 1 stack2) 0)]
        (doto region
          (.bufferOffset 0) (.bufferRowLength 0) (.bufferImageHeight 0))
        (doto (.imageSubresource region)
          (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
          (.mipLevel 0) (.baseArrayLayer 0) (.layerCount 1))
        (doto (.imageOffset region) (.x 0) (.y 0) (.z 0))
        (doto (.imageExtent region) (.width width) (.height height) (.depth 1))
        (VK13/vkCmdCopyBufferToImage cb (:buffer staging) image
                                     VK13/VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                     (VkBufferImageCopy/create (.address region) 1))
        (MemoryStack/stackPop))

      ;; Transition: transfer dst -> shader read
      (cmd-image-barrier-color! cb image
                                VK13/VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                VK13/VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                VK13/VK_ACCESS_TRANSFER_WRITE_BIT VK13/VK_ACCESS_SHADER_READ_BIT
                                VK13/VK_PIPELINE_STAGE_TRANSFER_BIT
                                VK13/VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)

      (vk/end-command-buffer cb)
      (vk/submit-and-wait! device queue cb))

    ;; Free staging buffer
    (mem/destroy-buffer! allocator staging)

    ;; Create image view
    (let [stack3 (MemoryStack/stackPush)
          iv-ci (doto (VkImageViewCreateInfo/calloc stack3)
                  (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                  (.image image)
                  (.viewType VK13/VK_IMAGE_VIEW_TYPE_2D)
                  (.format VK13/VK_FORMAT_R8G8B8A8_SRGB))
          _ (doto (.subresourceRange iv-ci)
              (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
              (.baseMipLevel 0) (.levelCount 1)
              (.baseArrayLayer 0) (.layerCount 1))
          p-view (.mallocLong stack3 1)
          r2 (VK13/vkCreateImageView device iv-ci nil p-view)
          _ (when-not (= r2 VK13/VK_SUCCESS)
              (throw (ex-info "Failed to create texture image view" {:result r2})))
          view (.get p-view 0)
          _ (MemoryStack/stackPop)

          ;; Create sampler
          stack4 (MemoryStack/stackPush)
          vk-filter (case filter-mode
                      :nearest VK13/VK_FILTER_NEAREST
                      :linear VK13/VK_FILTER_LINEAR)
          vk-address (case address-mode
                       :repeat VK13/VK_SAMPLER_ADDRESS_MODE_REPEAT
                       :clamp VK13/VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                       :mirror VK13/VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT)
          s-ci (doto (VkSamplerCreateInfo/calloc stack4)
                 (.sType VK13/VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                 (.magFilter vk-filter)
                 (.minFilter vk-filter)
                 (.addressModeU vk-address)
                 (.addressModeV vk-address)
                 (.addressModeW vk-address)
                 (.anisotropyEnable false)
                 (.borderColor VK13/VK_BORDER_COLOR_INT_OPAQUE_BLACK)
                 (.unnormalizedCoordinates false)
                 (.compareEnable false)
                 (.mipmapMode VK13/VK_SAMPLER_MIPMAP_MODE_LINEAR)
                 (.mipLodBias (float 0))
                 (.minLod (float 0))
                 (.maxLod (float 0)))
          p-sampler (.mallocLong stack4 1)
          r3 (VK13/vkCreateSampler device s-ci nil p-sampler)
          _ (when-not (= r3 VK13/VK_SUCCESS)
              (throw (ex-info "Failed to create sampler" {:result r3})))
          sampler (.get p-sampler 0)]
      (MemoryStack/stackPop)
      (->Texture image allocation view sampler width height))))

;; ================================================================
;; 2D Array Texture (multiple layers, same resolution per layer)
;; ================================================================

(defrecord TextureArray [image allocation view sampler width height layers layer-map])

(defn create-texture-array
  "Create a 2D array texture from multiple images. All images are padded to the
  max dimensions found. Each image becomes one layer with hardware REPEAT.

  images: seq of {:name string :pixels byte[] :width int :height int}
  Returns TextureArray record with :layer-map {name → {:layer int :scale-u float :scale-v float}}
  where scale-u/v = actual_size / padded_size for UV correction in shader.

  filter-mode: :nearest (default) or :linear"
  [ctx images & {:keys [filter-mode] :or {filter-mode :nearest}}]
  (let [n (count images)
        _ (when (zero? n)
            (throw (ex-info "No images for texture array" {})))
        ;; Find max dimensions
        max-w (int (reduce max (map :width images)))
        max-h (int (reduce max (map :height images)))
        layer-size (* max-w max-h 4)
        total-size (* layer-size n)

        device ^VkDevice (:device ctx)
        allocator (long (:allocator ctx))
        queue (:queue ctx)
        cmd-pool (long (:command-pool ctx))

        ;; Create staging buffer with all layers
        staging (mem/create-buffer allocator total-size :staging :cpu-to-gpu)
        _ (mem/with-mapped-buffer [^ByteBuffer bb allocator staging]
            (dotimes [i n]
              (let [img (nth images i)
                    ^bytes pixels (:pixels img)
                    iw (int (:width img))
                    ih (int (:height img))
                    layer-offset (* i layer-size)]
                ;; Clear layer to transparent black first
                (dotimes [j layer-size]
                  (.put bb (int (+ layer-offset j)) (byte 0)))
                ;; Copy actual pixels row by row (padding rightward + downward)
                (dotimes [row ih]
                  (let [src-offset (* row iw 4)
                        dst-offset (+ layer-offset (* row max-w 4))
                        copy-bytes (* iw 4)]
                    (dotimes [b copy-bytes]
                      (.put bb (int (+ dst-offset b))
                            (aget pixels (+ src-offset b)))))))))

        ;; Create GPU image (2D array)
        stack (MemoryStack/stackPush)
        img-ci (doto (VkImageCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                 (.imageType VK13/VK_IMAGE_TYPE_2D)
                 (.format VK13/VK_FORMAT_R8G8B8A8_SRGB)
                 (-> .extent (.width max-w) (.height max-h) (.depth 1))
                 (.mipLevels 1)
                 (.arrayLayers n)
                 (.samples VK13/VK_SAMPLE_COUNT_1_BIT)
                 (.tiling VK13/VK_IMAGE_TILING_OPTIMAL)
                 (.usage (bit-or VK13/VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                 VK13/VK_IMAGE_USAGE_SAMPLED_BIT))
                 (.sharingMode VK13/VK_SHARING_MODE_EXCLUSIVE)
                 (.initialLayout VK13/VK_IMAGE_LAYOUT_UNDEFINED))
        alloc-ci (doto (VmaAllocationCreateInfo/calloc stack)
                   (.usage (int Vma/VMA_MEMORY_USAGE_GPU_ONLY)))
        p-image (.mallocLong stack 1)
        p-alloc (.mallocPointer stack 1)
        r (Vma/vmaCreateImage allocator img-ci alloc-ci p-image p-alloc nil)
        _ (when-not (= r VK13/VK_SUCCESS)
            (throw (ex-info "Failed to create texture array image" {:result r})))
        image (.get p-image 0)
        allocation (.get p-alloc 0)]

    (MemoryStack/stackPop)

    ;; Upload via one-shot command buffer
    (let [cb (vk/allocate-command-buffer device cmd-pool)]
      (vk/begin-command-buffer cb)

      ;; Transition: undefined → transfer dst (all layers)
      (let [stack2 (MemoryStack/stackPush)
            ^org.lwjgl.vulkan.VkImageMemoryBarrier b
            (.get (org.lwjgl.vulkan.VkImageMemoryBarrier/calloc 1 stack2) 0)]
        (doto b
          (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
          (.oldLayout VK13/VK_IMAGE_LAYOUT_UNDEFINED)
          (.newLayout VK13/VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
          (.srcQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
          (.dstQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
          (.image image)
          (.srcAccessMask 0)
          (.dstAccessMask VK13/VK_ACCESS_TRANSFER_WRITE_BIT))
        (doto (.subresourceRange b)
          (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
          (.baseMipLevel 0) (.levelCount 1)
          (.baseArrayLayer 0) (.layerCount n))
        (VK13/vkCmdPipelineBarrier cb
                                   VK13/VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                   VK13/VK_PIPELINE_STAGE_TRANSFER_BIT
                                   0 nil nil
                                   (org.lwjgl.vulkan.VkImageMemoryBarrier/create (.address b) 1))
        (MemoryStack/stackPop))

      ;; Copy buffer → image (one region per layer)
      (let [stack3 (MemoryStack/stackPush)
            regions (VkBufferImageCopy/calloc n stack3)]
        (dotimes [i n]
          (let [^VkBufferImageCopy r (.get regions i)]
            (doto r
              (.bufferOffset (* i layer-size))
              (.bufferRowLength max-w)
              (.bufferImageHeight max-h))
            (doto (.imageSubresource r)
              (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
              (.mipLevel 0)
              (.baseArrayLayer i)
              (.layerCount 1))
            (doto (.imageOffset r) (.x 0) (.y 0) (.z 0))
            (doto (.imageExtent r) (.width max-w) (.height max-h) (.depth 1))))
        (VK13/vkCmdCopyBufferToImage cb (:buffer staging) image
                                     VK13/VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                     regions)
        (MemoryStack/stackPop))

      ;; Transition: transfer dst → shader read (all layers)
      (let [stack4 (MemoryStack/stackPush)
            ^org.lwjgl.vulkan.VkImageMemoryBarrier b
            (.get (org.lwjgl.vulkan.VkImageMemoryBarrier/calloc 1 stack4) 0)]
        (doto b
          (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
          (.oldLayout VK13/VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
          (.newLayout VK13/VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          (.srcQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
          (.dstQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
          (.image image)
          (.srcAccessMask VK13/VK_ACCESS_TRANSFER_WRITE_BIT)
          (.dstAccessMask VK13/VK_ACCESS_SHADER_READ_BIT))
        (doto (.subresourceRange b)
          (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
          (.baseMipLevel 0) (.levelCount 1)
          (.baseArrayLayer 0) (.layerCount n))
        (VK13/vkCmdPipelineBarrier cb
                                   VK13/VK_PIPELINE_STAGE_TRANSFER_BIT
                                   VK13/VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                   0 nil nil
                                   (org.lwjgl.vulkan.VkImageMemoryBarrier/create (.address b) 1))
        (MemoryStack/stackPop))

      (vk/end-command-buffer cb)
      (vk/submit-and-wait! device queue cb))

    ;; Free staging buffer
    (mem/destroy-buffer! allocator staging)

    ;; Create image view (2D_ARRAY)
    (let [stack5 (MemoryStack/stackPush)
          iv-ci (doto (VkImageViewCreateInfo/calloc stack5)
                  (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                  (.image image)
                  (.viewType VK13/VK_IMAGE_VIEW_TYPE_2D_ARRAY)
                  (.format VK13/VK_FORMAT_R8G8B8A8_SRGB))
          _ (doto (.subresourceRange iv-ci)
              (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
              (.baseMipLevel 0) (.levelCount 1)
              (.baseArrayLayer 0) (.layerCount n))
          p-view (.mallocLong stack5 1)
          r2 (VK13/vkCreateImageView device iv-ci nil p-view)
          _ (when-not (= r2 VK13/VK_SUCCESS)
              (throw (ex-info "Failed to create texture array image view" {:result r2})))
          view (.get p-view 0)
          _ (MemoryStack/stackPop)

          ;; Create sampler (CLAMP_TO_EDGE — shader does fract for repeat)
          stack6 (MemoryStack/stackPush)
          vk-filter (case filter-mode
                      :nearest VK13/VK_FILTER_NEAREST
                      :linear VK13/VK_FILTER_LINEAR)
          s-ci (doto (VkSamplerCreateInfo/calloc stack6)
                 (.sType VK13/VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                 (.magFilter vk-filter)
                 (.minFilter vk-filter)
                 (.addressModeU VK13/VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                 (.addressModeV VK13/VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                 (.addressModeW VK13/VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                 (.anisotropyEnable false)
                 (.borderColor VK13/VK_BORDER_COLOR_INT_OPAQUE_BLACK)
                 (.unnormalizedCoordinates false)
                 (.compareEnable false)
                 (.mipmapMode VK13/VK_SAMPLER_MIPMAP_MODE_NEAREST)
                 (.mipLodBias (float 0))
                 (.minLod (float 0))
                 (.maxLod (float 0)))
          p-sampler (.mallocLong stack6 1)
          r3 (VK13/vkCreateSampler device s-ci nil p-sampler)
          _ (when-not (= r3 VK13/VK_SUCCESS)
              (throw (ex-info "Failed to create texture array sampler" {:result r3})))
          sampler (.get p-sampler 0)]
      (MemoryStack/stackPop)

      ;; Build layer map: name → {:layer idx :scale-u f :scale-v f :width int :height int}
      (let [layer-map (into {}
                            (map-indexed
                             (fn [i img]
                               [(:name img)
                                {:layer i
                                 :width (:width img)
                                 :height (:height img)
                                 :scale-u (/ (double (:width img)) (double max-w))
                                 :scale-v (/ (double (:height img)) (double max-h))}])
                             images))]
        (->TextureArray image allocation view sampler max-w max-h n layer-map)))))

(defn destroy-texture-array!
  "Destroy a texture array (image, view, sampler, allocation)."
  [^VkDevice device ^long allocator ^TextureArray ta]
  (VK13/vkDestroySampler device (:sampler ta) nil)
  (VK13/vkDestroyImageView device (:view ta) nil)
  (Vma/vmaDestroyImage allocator (:image ta) (:allocation ta)))

;; ================================================================
;; Descriptor set for texture binding
;; ================================================================

(defn create-texture-descriptor-layout
  "Create a descriptor set layout with a combined image sampler at binding 0."
  ^long [^VkDevice device]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [binding (org.lwjgl.vulkan.VkDescriptorSetLayoutBinding/calloc 1 stack)
            _ (doto ^VkDescriptorSetLayoutBinding (.get binding 0)
                (.binding 0)
                (.descriptorType VK13/VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                (.descriptorCount 1)
                (.stageFlags VK13/VK_SHADER_STAGE_FRAGMENT_BIT))
            ci (doto (org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                 (.pBindings binding))
            p-layout (.mallocLong stack 1)
            r (VK13/vkCreateDescriptorSetLayout device ci nil p-layout)]
        (when-not (= r VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create texture descriptor layout" {:result r})))
        (.get p-layout 0))
      (finally
        (MemoryStack/stackPop)))))

(defn create-texture-descriptor-pool
  "Create a descriptor pool for combined image samplers."
  ^long [^VkDevice device ^long max-sets]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [pool-sizes (org.lwjgl.vulkan.VkDescriptorPoolSize/calloc 1 stack)
            _ (doto ^VkDescriptorPoolSize (.get pool-sizes 0)
                (.type VK13/VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                (.descriptorCount (int max-sets)))
            ci (doto (org.lwjgl.vulkan.VkDescriptorPoolCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                 (.maxSets (int max-sets))
                 (.pPoolSizes pool-sizes))
            p-pool (.mallocLong stack 1)
            r (VK13/vkCreateDescriptorPool device ci nil p-pool)]
        (when-not (= r VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create texture descriptor pool" {:result r})))
        (.get p-pool 0))
      (finally
        (MemoryStack/stackPop)))))

(defn allocate-texture-descriptor-set
  "Allocate a descriptor set and bind a texture to it."
  ^long [^VkDevice device ^long pool ^long layout ^Texture tex]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ai (doto (org.lwjgl.vulkan.VkDescriptorSetAllocateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                 (.descriptorPool pool)
                 (.pSetLayouts (.longs stack layout)))
            p-set (.mallocLong stack 1)
            r (VK13/vkAllocateDescriptorSets device ai p-set)]
        (when-not (= r VK13/VK_SUCCESS)
          (throw (ex-info "Failed to allocate texture descriptor set" {:result r})))
        (let [desc-set (.get p-set 0)
              img-info (org.lwjgl.vulkan.VkDescriptorImageInfo/calloc 1 stack)
              _ (doto ^VkDescriptorImageInfo (.get img-info 0)
                  (.imageLayout VK13/VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                  (.imageView (:view tex))
                  (.sampler (:sampler tex)))
              write (org.lwjgl.vulkan.VkWriteDescriptorSet/calloc 1 stack)
              _ (doto ^VkWriteDescriptorSet (.get write 0)
                  (.sType VK13/VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                  (.dstSet desc-set)
                  (.dstBinding 0)
                  (.descriptorCount 1)
                  (.descriptorType VK13/VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                  (.pImageInfo img-info))]
          (VK13/vkUpdateDescriptorSets device write nil)
          desc-set))
      (finally
        (MemoryStack/stackPop)))))

(defn allocate-texture-array-descriptor-set
  "Allocate a descriptor set and bind a TextureArray to it (sampler2DArray at binding 0)."
  ^long [^VkDevice device ^long pool ^long layout ^TextureArray ta]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ai (doto (org.lwjgl.vulkan.VkDescriptorSetAllocateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                 (.descriptorPool pool)
                 (.pSetLayouts (.longs stack layout)))
            p-set (.mallocLong stack 1)
            r (VK13/vkAllocateDescriptorSets device ai p-set)]
        (when-not (= r VK13/VK_SUCCESS)
          (throw (ex-info "Failed to allocate texture array descriptor set" {:result r})))
        (let [desc-set (.get p-set 0)
              img-info (org.lwjgl.vulkan.VkDescriptorImageInfo/calloc 1 stack)
              _ (doto ^VkDescriptorImageInfo (.get img-info 0)
                  (.imageLayout VK13/VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                  (.imageView (:view ta))
                  (.sampler (:sampler ta)))
              write (org.lwjgl.vulkan.VkWriteDescriptorSet/calloc 1 stack)
              _ (doto ^VkWriteDescriptorSet (.get write 0)
                  (.sType VK13/VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                  (.dstSet desc-set)
                  (.dstBinding 0)
                  (.descriptorCount 1)
                  (.descriptorType VK13/VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                  (.pImageInfo img-info))]
          (VK13/vkUpdateDescriptorSets device write nil)
          desc-set))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Pipeline layout with texture descriptor set
;; ================================================================

(defn create-textured-pipeline-layout
  "Create a pipeline layout with push constants + one texture descriptor set layout.
  Returns {:layout long :desc-layout long}."
  [^VkDevice device push-constants]
  (let [desc-layout (create-texture-descriptor-layout device)
        stack (MemoryStack/stackPush)]
    (try
      (let [ci (doto (org.lwjgl.vulkan.VkPipelineLayoutCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                 (.pSetLayouts (.longs stack desc-layout)))
            _ (when push-constants
                (let [buf (org.lwjgl.vulkan.VkPushConstantRange/calloc 1 stack)
                      ^org.lwjgl.vulkan.VkPushConstantRange pc (.get buf 0)]
                  (doto pc
                    (.stageFlags (int (:stages push-constants)))
                    (.offset 0)
                    (.size (int (:size push-constants))))
                  (.pPushConstantRanges ci buf)))
            p-layout (.mallocLong stack 1)
            r (VK13/vkCreatePipelineLayout device ci nil p-layout)]
        (when-not (= r VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create textured pipeline layout" {:result r})))
        {:layout (.get p-layout 0) :desc-layout desc-layout})
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-texture!
  "Destroy texture image, view, sampler, and free VMA allocation."
  [^VkDevice device ^long allocator ^Texture tex]
  (VK13/vkDestroySampler device (:sampler tex) nil)
  (VK13/vkDestroyImageView device (:view tex) nil)
  (Vma/vmaDestroyImage allocator (:image tex) (:allocation tex)))

(defn free-image-pixels!
  "Free pixels ByteBuffer returned by load-image."
  [image-data]
  (STBImage/stbi_image_free ^ByteBuffer (:pixels image-data)))
