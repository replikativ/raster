(ns raster.vk.screenshot
  "Screenshot capture — copy swapchain image to CPU buffer and write PNG.
  Call (request-screenshot! path) from REPL, screenshot happens next frame."
  (:require
   [raster.vk.core :as vk]
   [raster.vk.mem :as mem])
  (:import
   [org.lwjgl.vulkan VK13 KHRSwapchain VkCommandBuffer VkBufferImageCopy]
   [org.lwjgl.system MemoryStack]
   [java.awt.image BufferedImage]
   [javax.imageio ImageIO]
   [java.io File]))

(set! *warn-on-reflection* true)

;; Pending screenshot request (atom: nil or {:path "..."})
(defonce pending (atom nil))

;; Last error for debugging
(defonce last-error (atom nil))

(defn request-screenshot!
  "Request a screenshot to be captured on the next frame."
  [^String path]
  (reset! pending {:path path})
  (println "Screenshot requested:" path))

(defn capture-if-requested!
  "Call from frame loop. If a screenshot was requested,
  copies the swapchain image to a staging buffer and writes a PNG."
  [ctx ^long image ^long w ^long h]
  (when-let [req @pending]
    (reset! pending nil)
    (let [path (:path req)]
      (try
        (let [device (:device ctx)
              allocator (long (:allocator ctx))
              buf-size (* w h 4)
              staging (mem/create-buffer allocator buf-size :staging :gpu-to-cpu)]
          (try
            ;; Wait for GPU to finish
            (VK13/vkDeviceWaitIdle device)
            ;; Copy image → buffer
            (let [cb (vk/allocate-command-buffer device (:command-pool ctx))]
              (vk/begin-command-buffer cb)
              ;; Transition image to TRANSFER_SRC
              (vk/cmd-image-barrier! cb image
                                     [KHRSwapchain/VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                      VK13/VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL]
                                     [VK13/VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                                      VK13/VK_PIPELINE_STAGE_TRANSFER_BIT]
                                     [0 VK13/VK_ACCESS_TRANSFER_READ_BIT])
              ;; Copy command
              (let [stack (MemoryStack/stackPush)]
                (try
                  (let [region (VkBufferImageCopy/calloc 1 stack)
                        ^VkBufferImageCopy r (.get region 0)]
                    (.bufferOffset r 0)
                    (.bufferRowLength r 0)
                    (.bufferImageHeight r 0)
                    (doto (.imageSubresource r)
                      (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
                      (.mipLevel 0)
                      (.baseArrayLayer 0)
                      (.layerCount 1))
                    (doto (.imageOffset r)
                      (.set 0 0 0))
                    (doto (.imageExtent r)
                      (.set (int w) (int h) 1))
                    (VK13/vkCmdCopyImageToBuffer cb image
                                                 VK13/VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                                 (long (:buffer staging)) region))
                  (finally
                    (MemoryStack/stackPop))))
              ;; Transition back
              (vk/cmd-image-barrier! cb image
                                     [VK13/VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                      KHRSwapchain/VK_IMAGE_LAYOUT_PRESENT_SRC_KHR]
                                     [VK13/VK_PIPELINE_STAGE_TRANSFER_BIT
                                      VK13/VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT]
                                     [VK13/VK_ACCESS_TRANSFER_READ_BIT 0])
              (vk/end-command-buffer cb)
              (vk/submit-and-wait! device (:queue ctx) cb))
            ;; Read pixels
            (let [^java.nio.ByteBuffer bb (mem/map-buffer allocator staging)
                  img (BufferedImage. (int w) (int h) BufferedImage/TYPE_INT_ARGB)]
              (dotimes [y h]
                (dotimes [x w]
                  (let [offset (int (* (+ (* y w) x) 4))
                        b (bit-and (.get bb offset) 0xFF)
                        g (bit-and (.get bb (+ offset 1)) 0xFF)
                        r (bit-and (.get bb (+ offset 2)) 0xFF)
                        a (bit-and (.get bb (+ offset 3)) 0xFF)
                        argb (unchecked-int
                              (bit-or (bit-shift-left a 24)
                                      (bit-shift-left r 16)
                                      (bit-shift-left g 8)
                                      b))]
                    (.setRGB img (int x) (int y) argb))))
              (mem/unmap-buffer allocator staging)
              (ImageIO/write img "PNG" (File. ^String path))
              ;; Write confirmation to a debug file too
              (spit "/tmp/doom-screenshot-status.txt" (str "OK: " path "\n")))
            (finally
              (mem/destroy-buffer! allocator staging))))
        (catch Throwable e
          (reset! last-error e)
          (spit "/tmp/doom-screenshot-status.txt"
                (str "ERROR: " (.getMessage e) "\n"
                     (with-out-str (.printStackTrace e)))))))))
