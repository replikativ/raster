(ns raster.vk.window
  "GLFW window + Vulkan surface + swapchain management."
  (:import
   [org.lwjgl.glfw GLFW GLFWVulkan GLFWKeyCallback]
   [org.lwjgl.vulkan
    VK13 KHRSurface KHRSwapchain
    VkInstance VkPhysicalDevice VkDevice VkQueue
    VkSurfaceCapabilitiesKHR VkSurfaceFormatKHR VkPresentInfoKHR
    VkSwapchainCreateInfoKHR VkImageViewCreateInfo]
   [org.lwjgl.system MemoryStack MemoryUtil]))

(set! *warn-on-reflection* true)

;; ================================================================
;; GLFW init + window
;; ================================================================

(defn init-glfw!
  "Initialize GLFW for Vulkan (no OpenGL context)."
  []
  ;; On Wayland, GLFW uses libdecor for window decorations.
  ;; libdecor-gtk.so often fails to load, stripping title bar / close button.
  ;; Force X11 (via XWayland) for reliable decorations.
  (when (System/getenv "WAYLAND_DISPLAY")
    (GLFW/glfwInitHint GLFW/GLFW_PLATFORM GLFW/GLFW_PLATFORM_X11))
  (when-not (GLFW/glfwInit)
    (throw (ex-info "Failed to initialize GLFW" {})))
  (GLFW/glfwWindowHint GLFW/GLFW_CLIENT_API GLFW/GLFW_NO_API)
  (GLFW/glfwWindowHint GLFW/GLFW_RESIZABLE GLFW/GLFW_FALSE))

(defn create-window
  "Create a GLFW window. Returns window handle (long)."
  ^long [^long width ^long height ^String title]
  (let [window (GLFW/glfwCreateWindow (int width) (int height) title 0 0)]
    (when (zero? window)
      (throw (ex-info "Failed to create GLFW window" {:width width :height height})))
    window))

(defn required-extensions
  "Return the Vulkan instance extensions required by GLFW for surface creation."
  []
  (let [exts (GLFWVulkan/glfwGetRequiredInstanceExtensions)]
    (when-not exts
      (throw (ex-info "Vulkan not supported by GLFW" {})))
    (mapv #(.getStringUTF8 exts (int %)) (range (.remaining exts)))))

;; ================================================================
;; Surface
;; ================================================================

(defn create-surface
  "Create a Vulkan surface for the GLFW window.
  Returns surface handle (long)."
  ^long [^VkInstance instance ^long window]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [p-surface (.mallocLong stack 1)
            result (GLFWVulkan/glfwCreateWindowSurface instance window nil p-surface)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create window surface" {:result result})))
        (.get p-surface 0))
      (finally
        (MemoryStack/stackPop)))))

(defn check-surface-support
  "Check if a queue family supports presentation to a surface."
  [^VkPhysicalDevice pd ^long queue-family-index ^long surface]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [p-supported (.mallocInt stack 1)
            _ (KHRSurface/vkGetPhysicalDeviceSurfaceSupportKHR
               pd (int queue-family-index) surface p-supported)]
        (= (.get p-supported 0) VK13/VK_TRUE))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Swapchain
;; ================================================================

(defn create-swapchain
  "Create a swapchain for the given surface.
  Returns {:swapchain long :format int :extent [w h] :images [long...] :image-views [long...]}."
  [^VkDevice device ^VkPhysicalDevice pd surface
   extent queue-family-index]
  (let [surface (long surface)
        queue-family-index (long queue-family-index)
        width (long (nth extent 0))
        height (long (nth extent 1))
        stack (MemoryStack/stackPush)]
    (try
      (let [caps (VkSurfaceCapabilitiesKHR/calloc stack)
            _ (KHRSurface/vkGetPhysicalDeviceSurfaceCapabilitiesKHR pd surface caps)

            ;; Pick format
            p-count (.mallocInt stack 1)
            _ (KHRSurface/vkGetPhysicalDeviceSurfaceFormatsKHR pd surface p-count nil)
            formats (VkSurfaceFormatKHR/calloc (.get p-count 0) stack)
            _ (KHRSurface/vkGetPhysicalDeviceSurfaceFormatsKHR pd surface p-count formats)

            ;; Prefer B8G8R8A8_SRGB
            fmt (or (some (fn [i]
                            (let [f (.get formats (int i))]
                              (when (= (.format f) VK13/VK_FORMAT_B8G8R8A8_SRGB)
                                {:format (.format f) :color-space (.colorSpace f)})))
                          (range (.remaining formats)))
                    (let [f (.get formats 0)]
                      {:format (.format f) :color-space (.colorSpace f)}))

            image-count (max (inc (.minImageCount caps))
                             (if (pos? (.maxImageCount caps))
                               (min (inc (.minImageCount caps)) (.maxImageCount caps))
                               (inc (.minImageCount caps))))

            ci (doto (VkSwapchainCreateInfoKHR/calloc stack)
                 (.sType KHRSwapchain/VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR)
                 (.surface surface)
                 (.minImageCount (int image-count))
                 (.imageFormat (int (:format fmt)))
                 (.imageColorSpace (int (:color-space fmt)))
                 (-> .imageExtent (.width (int width)) (.height (int height)))
                 (.imageArrayLayers 1)
                 (.imageUsage (bit-or VK13/VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                      VK13/VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
                 (.imageSharingMode VK13/VK_SHARING_MODE_EXCLUSIVE)
                 (.preTransform (.currentTransform caps))
                 (.compositeAlpha KHRSurface/VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                 (.presentMode KHRSurface/VK_PRESENT_MODE_FIFO_KHR)  ;; vsync (guaranteed supported)
                 (.clipped true))

            p-swapchain (.mallocLong stack 1)
            result (KHRSwapchain/vkCreateSwapchainKHR device ci nil p-swapchain)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create swapchain" {:result result})))

        (let [swapchain (.get p-swapchain 0)
              ;; Get swapchain images
              _ (KHRSwapchain/vkGetSwapchainImagesKHR device swapchain p-count nil)
              n-images (.get p-count 0)
              p-images (.mallocLong stack n-images)
              _ (KHRSwapchain/vkGetSwapchainImagesKHR device swapchain p-count p-images)
              images (mapv #(.get p-images (int %)) (range n-images))

              ;; Create image views
              image-views
              (mapv (fn [img]
                      (let [iv-ci (doto (VkImageViewCreateInfo/calloc stack)
                                    (.sType VK13/VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                                    (.image img)
                                    (.viewType VK13/VK_IMAGE_VIEW_TYPE_2D)
                                    (.format (int (:format fmt)))
                                    (-> .components
                                        (.r VK13/VK_COMPONENT_SWIZZLE_IDENTITY)
                                        (.g VK13/VK_COMPONENT_SWIZZLE_IDENTITY)
                                        (.b VK13/VK_COMPONENT_SWIZZLE_IDENTITY)
                                        (.a VK13/VK_COMPONENT_SWIZZLE_IDENTITY))
                                    (-> .subresourceRange
                                        (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
                                        (.baseMipLevel 0)
                                        (.levelCount 1)
                                        (.baseArrayLayer 0)
                                        (.layerCount 1)))
                            p-view (.mallocLong stack 1)
                            r (VK13/vkCreateImageView device iv-ci nil p-view)]
                        (when-not (= r VK13/VK_SUCCESS)
                          (throw (ex-info "Failed to create image view" {:result r})))
                        (.get p-view 0)))
                    images)]
          {:swapchain swapchain
           :format (:format fmt)
           :extent [width height]
           :images images
           :image-views image-views}))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Present
;; ================================================================

(defn acquire-next-image
  "Acquire the next swapchain image. Returns image index."
  ^long [^VkDevice device ^long swapchain ^long semaphore]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [p-idx (.mallocInt stack 1)
            ;; 1-second timeout instead of infinite — prevents hang when window not visible
            result (KHRSwapchain/vkAcquireNextImageKHR
                    device swapchain 1000000000 semaphore 0 p-idx)]
        (cond
          (or (= result VK13/VK_SUCCESS)
              (= result KHRSwapchain/VK_SUBOPTIMAL_KHR))
          (.get p-idx 0)
          ;; Timeout or not ready — return -1 to signal skip frame
          (= result VK13/VK_TIMEOUT)
          -1
          (= result VK13/VK_NOT_READY)
          -1
          :else
          (throw (ex-info "Failed to acquire swapchain image" {:result result}))))
      (finally
        (MemoryStack/stackPop)))))

(defn present!
  "Present a swapchain image to the screen."
  [^VkQueue queue ^long swapchain ^long image-index ^long wait-semaphore]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [pi (doto (VkPresentInfoKHR/calloc stack)
                 (.sType KHRSwapchain/VK_STRUCTURE_TYPE_PRESENT_INFO_KHR)
                 (.pWaitSemaphores (.longs stack wait-semaphore))
                 (.swapchainCount 1)
                 (.pSwapchains (.longs stack swapchain))
                 (.pImageIndices (.ints stack (int image-index))))]
        (KHRSwapchain/vkQueuePresentKHR queue pi))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-swapchain!
  [^VkDevice device swapchain-info]
  (doseq [iv (:image-views swapchain-info)]
    (VK13/vkDestroyImageView device (long iv) nil))
  (KHRSwapchain/vkDestroySwapchainKHR device (:swapchain swapchain-info) nil))

(defn destroy-surface! [^VkInstance instance ^long surface]
  (KHRSurface/vkDestroySurfaceKHR instance surface nil))

(defn destroy-window! [^long window]
  (GLFW/glfwDestroyWindow window)
  (GLFW/glfwTerminate))

(defn window-should-close? [^long window]
  (GLFW/glfwWindowShouldClose window))

(defn poll-events! []
  (GLFW/glfwPollEvents))
