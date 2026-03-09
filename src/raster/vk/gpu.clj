(ns raster.vk.gpu
  "GPU context: bundles instance + device + queue + allocator + swapchain +
  command pool + depth image into a single record. Created once, passed everywhere."
  (:require
   [raster.vk.core :as vk]
   [raster.vk.mem :as mem]
   [raster.vk.window :as win])
  (:import
   [org.lwjgl.vulkan VK13 VkDevice VkQueue VkCommandBuffer]
   [org.lwjgl.system MemoryStack]))

(set! *warn-on-reflection* true)

(defrecord GpuContext
           [instance physical-device device queue queue-family-idx
            allocator window surface swapchain-info command-pool
            depth-image extent properties])

;; ================================================================
;; Depth image (extracted from demos)
;; ================================================================

(defn- create-depth-image
  [^VkDevice device ^long allocator ^long w ^long h]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [img-ci (doto (org.lwjgl.vulkan.VkImageCreateInfo/calloc stack)
                     (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                     (.imageType VK13/VK_IMAGE_TYPE_2D)
                     (.format VK13/VK_FORMAT_D32_SFLOAT)
                     (-> .extent (.width (int w)) (.height (int h)) (.depth 1))
                     (.mipLevels 1) (.arrayLayers 1)
                     (.samples VK13/VK_SAMPLE_COUNT_1_BIT)
                     (.tiling VK13/VK_IMAGE_TILING_OPTIMAL)
                     (.usage VK13/VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                     (.sharingMode VK13/VK_SHARING_MODE_EXCLUSIVE))
            alloc-ci (doto (org.lwjgl.util.vma.VmaAllocationCreateInfo/calloc stack)
                       (.usage (int org.lwjgl.util.vma.Vma/VMA_MEMORY_USAGE_GPU_ONLY)))
            p-image (.mallocLong stack 1)
            p-alloc (.mallocPointer stack 1)
            r (org.lwjgl.util.vma.Vma/vmaCreateImage allocator img-ci alloc-ci p-image p-alloc nil)]
        (when-not (= r VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create depth image" {:result r})))
        (let [image (.get p-image 0)
              allocation (.get p-alloc 0)
              iv-ci (doto (org.lwjgl.vulkan.VkImageViewCreateInfo/calloc stack)
                      (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                      (.image image)
                      (.viewType VK13/VK_IMAGE_VIEW_TYPE_2D)
                      (.format VK13/VK_FORMAT_D32_SFLOAT))
              _ (doto (.subresourceRange iv-ci)
                  (.aspectMask VK13/VK_IMAGE_ASPECT_DEPTH_BIT)
                  (.baseMipLevel 0) (.levelCount 1)
                  (.baseArrayLayer 0) (.layerCount 1))
              p-view (.mallocLong stack 1)
              r2 (VK13/vkCreateImageView device iv-ci nil p-view)]
          (when-not (= r2 VK13/VK_SUCCESS)
            (throw (ex-info "Failed to create depth image view" {:result r2})))
          {:image image :allocation allocation :view (.get p-view 0)}))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Context creation / destruction
;; ================================================================

(defn create-gpu-context
  "One call to set up everything. Returns GpuContext."
  [& {:keys [width height title debug? depth? features]
      :or {width 1024 height 768 title "Raster" debug? false depth? true features {}}}]
  (win/init-glfw!)
  (let [window (win/create-window width height title)
        exts (win/required-extensions)
        instance (vk/create-instance :extensions exts :debug? debug?)
        surface (win/create-surface instance window)
        sel (vk/select-physical-device instance)
        pd (:physical-device sel)
        qf-idx (:index (first (filter #(and (:graphics? %) (:compute? %))
                                      (:queue-families sel))))
        _ (assert (win/check-surface-support pd qf-idx surface)
                  "Queue family doesn't support presentation")
        dev-info (vk/create-device pd qf-idx :features features)
        device (:device dev-info)
        queue (:graphics-queue dev-info)
        allocator (mem/create-allocator instance pd device)
        sc (win/create-swapchain device pd surface [width height] qf-idx)
        cmd-pool (vk/create-command-pool device qf-idx)
        depth (when depth?
                (create-depth-image device allocator width height))]
    (->GpuContext instance pd device queue qf-idx
                  allocator window surface sc cmd-pool
                  depth [width height] (:properties sel))))

(defn destroy-gpu-context!
  "Reverse teardown of all GPU resources."
  [^GpuContext ctx]
  (let [device (:device ctx)
        allocator (:allocator ctx)]
    (VK13/vkDeviceWaitIdle device)
    (vk/destroy-command-pool! device (:command-pool ctx))
    (when-let [depth (:depth-image ctx)]
      (VK13/vkDestroyImageView device (:view depth) nil)
      (org.lwjgl.util.vma.Vma/vmaDestroyImage allocator (:image depth) (:allocation depth)))
    (win/destroy-swapchain! device (:swapchain-info ctx))
    (mem/destroy-allocator! allocator)
    (win/destroy-surface! (:instance ctx) (:surface ctx))
    (vk/destroy-device! device)
    (vk/destroy-instance! (:instance ctx))
    (win/destroy-window! (:window ctx))))

(defmacro with-gpu-context
  "Bind ctx, execute body, destroy in finally.
  opts is a map literal: (with-gpu-context [ctx {:width 800 :height 600}] ...)"
  [[sym opts] & body]
  (let [kv-pairs (mapcat identity opts)]
    `(let [~sym (create-gpu-context ~@kv-pairs)]
       (try
         ~@body
         (finally
           (destroy-gpu-context! ~sym))))))

(defn one-shot-submit!
  "Allocate a transient command buffer, call (f cb), submit-and-wait."
  [^GpuContext ctx f]
  (let [cb (vk/allocate-command-buffer (:device ctx) (:command-pool ctx))]
    (vk/begin-command-buffer cb)
    (f cb)
    (vk/end-command-buffer cb)
    (vk/submit-and-wait! (:device ctx) (:queue ctx) cb)))
