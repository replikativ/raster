(ns raster.vk.core
  "Thin Vulkan wrapper via LWJGL.
  Data-oriented: all GPU state lives in plain maps/records.
  Manages instance, device, queues, and command pools."
  (:import
   [org.lwjgl.system MemoryStack MemoryUtil]
   [org.lwjgl.vulkan
    VK13 KHRSurface KHRSwapchain
    VkInstance VkInstanceCreateInfo VkApplicationInfo
    VkPhysicalDevice VkDevice VkDeviceCreateInfo VkDeviceQueueCreateInfo
    VkQueue VkQueueFamilyProperties
    VkCommandPoolCreateInfo
    VkCommandBuffer VkCommandBufferAllocateInfo VkCommandBufferBeginInfo
    VkSubmitInfo VkFenceCreateInfo
    VkPhysicalDeviceFeatures VkPhysicalDeviceProperties
    VkExtensionProperties VkLayerProperties]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Instance
;; ================================================================

(defn create-instance
  "Create a VkInstance with optional validation layers and extensions.
  Returns VkInstance."
  [& {:keys [app-name engine-name extensions layers debug?]
      :or {app-name "Raster" engine-name "Raster Engine"
           extensions [] layers [] debug? false}}]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [app-info (doto (VkApplicationInfo/calloc stack)
                       (.sType VK13/VK_STRUCTURE_TYPE_APPLICATION_INFO)
                       (.pApplicationName (.UTF8 stack app-name))
                       (.applicationVersion (VK13/VK_MAKE_API_VERSION 0 1 0 0))
                       (.pEngineName (.UTF8 stack engine-name))
                       (.engineVersion (VK13/VK_MAKE_API_VERSION 0 1 0 0))
                       (.apiVersion (VK13/VK_API_VERSION_1_3)))
            all-layers (if debug?
                         (into ["VK_LAYER_KHRONOS_validation"] layers)
                         layers)
            pp-layers (when (seq all-layers)
                        (let [buf (.mallocPointer stack (count all-layers))]
                          (doseq [^String l all-layers]
                            (.put buf (.UTF8 stack l)))
                          (.flip buf)))
            pp-exts (when (seq extensions)
                      (let [buf (.mallocPointer stack (count extensions))]
                        (doseq [^String e extensions]
                          (.put buf (.UTF8 stack e)))
                        (.flip buf)))
            create-info (doto (VkInstanceCreateInfo/calloc stack)
                          (.sType VK13/VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                          (.pApplicationInfo app-info))
            _ (when pp-layers (.ppEnabledLayerNames create-info pp-layers))
            _ (when pp-exts (.ppEnabledExtensionNames create-info pp-exts))
            p-instance (.mallocPointer stack 1)
            result (VK13/vkCreateInstance create-info nil p-instance)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create Vulkan instance" {:result result})))
        (VkInstance. (.get p-instance 0) create-info))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Physical device selection
;; ================================================================

(defn enumerate-physical-devices
  "Return a vector of VkPhysicalDevice handles."
  [^VkInstance instance]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [p-count (.mallocInt stack 1)
            _ (VK13/vkEnumeratePhysicalDevices instance p-count nil)
            n (.get p-count 0)
            p-devices (.mallocPointer stack n)
            _ (VK13/vkEnumeratePhysicalDevices instance p-count p-devices)]
        (mapv #(VkPhysicalDevice. (.get p-devices (int %)) instance) (range n)))
      (finally
        (MemoryStack/stackPop)))))

(defn device-properties
  "Get physical device properties as a map."
  [^VkPhysicalDevice pd]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [props (VkPhysicalDeviceProperties/calloc stack)
            _ (VK13/vkGetPhysicalDeviceProperties pd props)]
        {:device-name (.deviceNameString props)
         :device-type (.deviceType props)
         :api-version (.apiVersion props)
         :vendor-id (.vendorID props)
         :device-id (.deviceID props)})
      (finally
        (MemoryStack/stackPop)))))

(defn find-queue-families
  "Return vector of queue family properties maps."
  [^VkPhysicalDevice pd]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [p-count (.mallocInt stack 1)
            _ (VK13/vkGetPhysicalDeviceQueueFamilyProperties pd p-count nil)
            n (.get p-count 0)
            props (VkQueueFamilyProperties/calloc n stack)
            _ (VK13/vkGetPhysicalDeviceQueueFamilyProperties pd p-count props)]
        (mapv (fn [i]
                (let [^VkQueueFamilyProperties p (.get props (int i))]
                  {:index i
                   :queue-count (.queueCount p)
                   :flags (.queueFlags p)
                   :graphics? (pos? (bit-and (.queueFlags p) VK13/VK_QUEUE_GRAPHICS_BIT))
                   :compute? (pos? (bit-and (.queueFlags p) VK13/VK_QUEUE_COMPUTE_BIT))
                   :transfer? (pos? (bit-and (.queueFlags p) VK13/VK_QUEUE_TRANSFER_BIT))}))
              (range n)))
      (finally
        (MemoryStack/stackPop)))))

(defn select-physical-device
  "Select the best physical device (discrete GPU preferred, then integrated).
  Returns {:physical-device pd :properties props :queue-families families}."
  [^VkInstance instance]
  (let [devices (enumerate-physical-devices instance)
        scored (mapv (fn [pd]
                       (let [props (device-properties pd)
                             families (find-queue-families pd)
                             score (cond
                                     (= (:device-type props) VK13/VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) 100
                                     (= (:device-type props) VK13/VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) 50
                                     :else 10)]
                         {:physical-device pd :properties props
                          :queue-families families :score score}))
                     devices)]
    (first (sort-by :score > scored))))

;; ================================================================
;; Logical device + queues
;; ================================================================

(defn create-device
  "Create a logical VkDevice with one graphics+compute queue.
  features: map of VkPhysicalDeviceFeatures to enable, e.g. {:large-points true}.
  Returns {:device VkDevice :graphics-queue VkQueue :queue-family-index int}."
  [^VkPhysicalDevice pd queue-family-index & {:keys [extensions features]
                                              :or {extensions [] features {}}}]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [priorities (.floats stack (float 1.0))
            queue-ci-buf (VkDeviceQueueCreateInfo/calloc 1 stack)
            ^VkDeviceQueueCreateInfo qci (.get queue-ci-buf 0)
            _ (doto qci
                (.sType VK13/VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                (.queueFamilyIndex (int (long queue-family-index)))
                (.pQueuePriorities priorities))
            ;; Enable swapchain extension for presentation
            all-exts (into [KHRSwapchain/VK_KHR_SWAPCHAIN_EXTENSION_NAME] extensions)
            pp-exts (let [buf (.mallocPointer stack (count all-exts))]
                      (doseq [^String e all-exts]
                        (.put buf (.UTF8 stack e)))
                      (.flip buf))
            phys-features (doto (VkPhysicalDeviceFeatures/calloc stack)
                            (.largePoints (boolean (:large-points features))))
            ;; Vulkan 1.3 features: dynamic rendering + shader demote to helper
            vk13-features (doto (org.lwjgl.vulkan.VkPhysicalDeviceVulkan13Features/calloc stack)
                            (.sType VK13/VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
                            (.dynamicRendering true)
                            (.synchronization2 true)
                            (.shaderDemoteToHelperInvocation true))
            device-ci (doto (VkDeviceCreateInfo/calloc stack)
                        (.sType VK13/VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                        (.pNext (.address vk13-features))
                        (.pQueueCreateInfos queue-ci-buf)
                        (.ppEnabledExtensionNames pp-exts)
                        (.pEnabledFeatures phys-features))
            p-device (.mallocPointer stack 1)
            result (VK13/vkCreateDevice pd device-ci nil p-device)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create Vulkan device" {:result result})))
        (let [device (VkDevice. (.get p-device 0) pd device-ci)
              p-queue (.mallocPointer stack 1)
              _ (VK13/vkGetDeviceQueue device (int queue-family-index) 0 p-queue)
              queue (VkQueue. (.get p-queue 0) device)]
          {:device device
           :graphics-queue queue
           :queue-family-index queue-family-index}))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Command pool + buffers
;; ================================================================

(defn create-command-pool
  "Create a command pool for the given queue family."
  [^VkDevice device ^long queue-family-index]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ci (doto (VkCommandPoolCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                 (.flags VK13/VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                 (.queueFamilyIndex (int queue-family-index)))
            p-pool (.mallocLong stack 1)
            result (VK13/vkCreateCommandPool device ci nil p-pool)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create command pool" {:result result})))
        (.get p-pool 0))
      (finally
        (MemoryStack/stackPop)))))

(defn allocate-command-buffer
  "Allocate a single primary command buffer from pool."
  [^VkDevice device ^long command-pool]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ai (doto (VkCommandBufferAllocateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                 (.commandPool command-pool)
                 (.level VK13/VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                 (.commandBufferCount 1))
            p-cb (.mallocPointer stack 1)
            result (VK13/vkAllocateCommandBuffers device ai p-cb)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to allocate command buffer" {:result result})))
        (VkCommandBuffer. (.get p-cb 0) device))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Fence
;; ================================================================

(defn create-fence
  "Create a fence, optionally signaled."
  [^VkDevice device & {:keys [signaled?] :or {signaled? false}}]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ci (doto (VkFenceCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
                 (.flags (if signaled? VK13/VK_FENCE_CREATE_SIGNALED_BIT 0)))
            p-fence (.mallocLong stack 1)
            result (VK13/vkCreateFence device ci nil p-fence)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create fence" {:result result})))
        (.get p-fence 0))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Command recording helpers
;; ================================================================

(defn begin-command-buffer
  "Begin recording into a command buffer (one-time submit)."
  [^VkCommandBuffer cb]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [bi (doto (VkCommandBufferBeginInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                 (.flags VK13/VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))]
        (VK13/vkBeginCommandBuffer cb bi))
      (finally
        (MemoryStack/stackPop)))))

(defn end-command-buffer
  [^VkCommandBuffer cb]
  (VK13/vkEndCommandBuffer cb))

(defn submit-and-wait!
  "Submit command buffer to queue and wait for fence."
  [^VkDevice device ^VkQueue queue ^VkCommandBuffer cb]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [fence (create-fence device)
            p-cb (.mallocPointer stack 1)
            _ (.put p-cb 0 cb)
            si (doto (VkSubmitInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_SUBMIT_INFO)
                 (.pCommandBuffers p-cb))
            result (VK13/vkQueueSubmit queue si (long fence))]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Queue submit failed" {:result result})))
        (VK13/vkWaitForFences device (long fence) true Long/MAX_VALUE)
        (VK13/vkDestroyFence device fence nil))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Semaphores
;; ================================================================

(defn create-semaphore
  "Create a Vulkan semaphore."
  ^long [^VkDevice device]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ci (doto (org.lwjgl.vulkan.VkSemaphoreCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO))
            p-sem (.mallocLong stack 1)
            result (VK13/vkCreateSemaphore device ci nil p-sem)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create semaphore" {:result result})))
        (.get p-sem 0))
      (finally
        (MemoryStack/stackPop)))))

(defn destroy-semaphore! [^VkDevice device ^long sem]
  (VK13/vkDestroySemaphore device sem nil))

;; ================================================================
;; Submit with semaphore sync
;; ================================================================

(defn submit-with-sync!
  "Submit command buffer with wait/signal semaphores.
  sync: [wait-semaphore signal-semaphore fence]"
  [^VkDevice device ^VkQueue queue ^VkCommandBuffer cb sync]
  (let [wait-sem (long (nth sync 0))
        signal-sem (long (nth sync 1))
        fence (long (nth sync 2))
        stack (MemoryStack/stackPush)]
    (try
      (let [p-cb (.mallocPointer stack 1)
            _ (.put p-cb 0 cb)
            si (doto (VkSubmitInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_SUBMIT_INFO)
                 (.pWaitSemaphores (.longs stack wait-sem))
                 (.pWaitDstStageMask (.ints stack VK13/VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT))
                 (.pCommandBuffers p-cb)
                 (.pSignalSemaphores (.longs stack signal-sem)))
            result (VK13/vkQueueSubmit queue si fence)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Queue submit failed" {:result result}))))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Image layout transitions
;; ================================================================

(defn cmd-image-barrier!
  "Record an image memory barrier for layout transition.
  layouts: [old-layout new-layout], stages: [src-stage dst-stage], access: [src-access dst-access]"
  [^VkCommandBuffer cb image layouts stages access]
  (let [old-layout (int (nth layouts 0))
        new-layout (int (nth layouts 1))
        src-stage (int (nth stages 0))
        dst-stage (int (nth stages 1))
        src-access (int (nth access 0))
        dst-access (int (nth access 1))
        stack (MemoryStack/stackPush)]
    (try
      (let [^org.lwjgl.vulkan.VkImageMemoryBarrier b (.get (org.lwjgl.vulkan.VkImageMemoryBarrier/calloc 1 stack) 0)
            _ (doto b
                (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                (.oldLayout old-layout)
                (.newLayout new-layout)
                (.srcQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
                (.dstQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
                (.image (long image))
                (.srcAccessMask src-access)
                (.dstAccessMask dst-access))
            _ (doto (.subresourceRange b)
                (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
                (.baseMipLevel 0)
                (.levelCount 1)
                (.baseArrayLayer 0)
                (.layerCount 1))
            barrier (org.lwjgl.vulkan.VkImageMemoryBarrier/create (.address b) 1)]
        (VK13/vkCmdPipelineBarrier cb
                                   src-stage dst-stage
                                   0 nil nil barrier))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Buffer memory barriers (compute <-> render sync)
;; ================================================================

(defn cmd-buffer-barrier!
  "Record a buffer memory barrier for synchronization between pipeline stages.
  Typical use: compute shader write -> vertex input read.
  stages: [src-stage dst-stage], access: [src-access dst-access]"
  [^VkCommandBuffer cb buffer size stages access]
  (let [buffer (long buffer)
        size (long size)
        src-stage (int (nth stages 0))
        dst-stage (int (nth stages 1))
        src-access (int (nth access 0))
        dst-access (int (nth access 1))
        stack (MemoryStack/stackPush)]
    (try
      (let [^org.lwjgl.vulkan.VkBufferMemoryBarrier b
            (.get (org.lwjgl.vulkan.VkBufferMemoryBarrier/calloc 1 stack) 0)
            _ (doto b
                (.sType VK13/VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER)
                (.srcAccessMask src-access)
                (.dstAccessMask dst-access)
                (.srcQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
                (.dstQueueFamilyIndex VK13/VK_QUEUE_FAMILY_IGNORED)
                (.buffer buffer)
                (.offset 0)
                (.size size))
            barrier (org.lwjgl.vulkan.VkBufferMemoryBarrier/create (.address b) 1)]
        (VK13/vkCmdPipelineBarrier cb
                                   src-stage dst-stage
                                   0 nil barrier nil))
      (finally
        (MemoryStack/stackPop)))))

(defn cmd-compute-to-vertex-barrier!
  "Convenience: barrier from compute shader write to vertex attribute read."
  [^VkCommandBuffer cb buffer size]
  (cmd-buffer-barrier! cb buffer size
                       [VK13/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                        VK13/VK_PIPELINE_STAGE_VERTEX_INPUT_BIT]
                       [VK13/VK_ACCESS_SHADER_WRITE_BIT
                        VK13/VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT]))

(defn cmd-vertex-to-compute-barrier!
  "Convenience: barrier from vertex read back to compute shader write."
  [^VkCommandBuffer cb buffer size]
  (cmd-buffer-barrier! cb buffer size
                       [VK13/VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
                        VK13/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT]
                       [VK13/VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
                        VK13/VK_ACCESS_SHADER_WRITE_BIT]))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-command-pool! [^VkDevice device ^long pool]
  (VK13/vkDestroyCommandPool device pool nil))

(defn destroy-device! [^VkDevice device]
  (VK13/vkDestroyDevice device nil))

(defn destroy-instance! [^VkInstance instance]
  (VK13/vkDestroyInstance instance nil))
