(ns raster.vk.frame
  "Frame loop: fixed-timestep game loop with compute->render phases.
  Handles swapchain acquire/present, barriers, sync objects.

  Synchronization follows the official Vulkan swapchain semaphore reuse pattern:
  - acquire_semaphores[frame_index]  — per frame slot (fence wait guarantees prior consume)
  - submit_semaphores[image_index]   — per swapchain image (acquire guarantees prior present done)
  - fences[frame_index]              — per frame slot
  - command_buffers[frame_index]     — per frame slot

  Key insight: vkQueuePresentKHR cannot signal fences, so a submit fence does NOT
  guarantee the previous present released its wait semaphore. The only guarantee
  that a previous present for image N completed is successfully acquiring image N again.
  Therefore present-wait (submit-signal) semaphores must be indexed by image, not frame."
  (:require
   [raster.vk.core :as vk]
   [raster.vk.window :as win]
   [raster.vk.pipeline :as pipeline]
   [raster.vk.input :as input]
   [raster.vk.screenshot :as screenshot]
   [raster.vk.stream :as stream])
  (:import
   [org.lwjgl.system MemoryStack]
   [org.lwjgl.vulkan VK13 KHRSwapchain VkDevice VkCommandBuffer
    VkSubmitInfo2 VkSemaphoreSubmitInfo VkCommandBufferSubmitInfo]
   [org.lwjgl.glfw GLFW]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Frames-in-flight sync objects
;; ================================================================

(def ^:const MAX-FRAMES-IN-FLIGHT 2)

(defrecord FrameSync [acquire-sems       ;; per frame slot — signaled by vkAcquireNextImageKHR
                      submit-sems        ;; per swapchain image — signaled by submit, waited by present
                      in-flight-fences   ;; per frame slot — signaled by submit
                      command-buffers])  ;; per frame slot — pre-allocated, reset each frame

(defn create-frame-sync
  "Create sync objects following the Vulkan swapchain semaphore reuse pattern.
  acquire-sems:     MAX-FRAMES-IN-FLIGHT (indexed by frame slot)
  submit-sems:      swapchain-image-count (indexed by acquired image index)
  in-flight-fences: MAX-FRAMES-IN-FLIGHT (indexed by frame slot, created signaled)
  command-buffers:  MAX-FRAMES-IN-FLIGHT (indexed by frame slot)"
  [ctx]
  (let [device (:device ctx)
        n MAX-FRAMES-IN-FLIGHT
        sc-image-count (count (get-in ctx [:swapchain-info :images]))]
    (->FrameSync
     (long-array (repeatedly n #(vk/create-semaphore device)))
     (long-array (repeatedly sc-image-count #(vk/create-semaphore device)))
     (long-array (repeatedly n #(vk/create-fence device :signaled? true)))
     (object-array (repeatedly n #(vk/allocate-command-buffer device (:command-pool ctx)))))))

(defn destroy-frame-sync!
  [ctx ^FrameSync sync]
  (let [device (:device ctx)]
    (doseq [^longs arr [(:acquire-sems sync) (:submit-sems sync)]]
      (dotimes [i (alength arr)]
        (vk/destroy-semaphore! device (aget arr i))))
    (let [^longs fences (:in-flight-fences sync)]
      (dotimes [i (alength fences)]
        (VK13/vkDestroyFence device (aget fences i) nil)))))

;; ================================================================
;; Game loop
;; ================================================================

;; Atom holding current game state. Set during game loop for REPL access.
;; Read: @frame/game-state
;; Write: (reset! frame/game-state new-state) — takes effect next frame.
(defonce game-state (atom nil))

(defn- resolve-fn
  "Resolve a var or return the function directly."
  [x]
  (cond
    (var? x) (deref x)
    (qualified-symbol? x) (deref (requiring-resolve x))
    :else x))

(defn run-game-loop!
  "Main game loop. Fixed timestep update, variable render.

  Synchronization (per the Vulkan swapchain semaphore reuse guide):
    1. vkWaitForFences(fence[frame])           — CPU waits for this slot's prior submit
    2. vkResetFences(fence[frame])             — ready for reuse
    3. vkAcquireNextImageKHR(acq_sem[frame])   — signals acquire semaphore
       Acquiring image N guarantees prior present of image N completed,
       so submit_sem[N] is safe to signal again.
    4. Record commands into cmd_buf[frame]
    5. vkQueueSubmit: wait acq_sem[frame], signal submit_sem[image], fence[frame]
    6. vkQueuePresentKHR: wait submit_sem[image]

  Callbacks (fn or var — vars are deref'd each frame for hot-reload):
    update-fn:  (fn [state input dt] new-state)   -- fixed timestep, pure
    compute-fn: (fn [state cb ctx] nil)            -- record compute dispatches (optional)
    render-fn:  (fn [state cb ctx frame-info] nil) -- record draw commands

  State is exposed via game-state atom for REPL inspection and modification.
  frame-info: {:image-index :image :image-view :extent :depth-view :t :dt}"
  [ctx {:keys [frame-sync init-state update-fn compute-fn render-fn
               input-system clear-color fixed-dt title state-atom]
        :or {clear-color [0.02 0.02 0.05 1.0] fixed-dt (/ 1.0 60)}}]
  (let [^VkDevice device (:device ctx)
        ^org.lwjgl.vulkan.VkQueue queue (:queue ctx)
        ^long window (:window ctx)
        sc (:swapchain-info ctx)
        [w h] (:extent ctx)
        depth-view (when-let [d (:depth-image ctx)] (:view d))

        ^longs acquire-sems (:acquire-sems frame-sync)
        ^longs submit-sems (:submit-sems frame-sync)
        ^longs in-flight-fences (:in-flight-fences frame-sync)
        ^objects cmd-bufs (:command-buffers frame-sync)

        fixed-dt (double fixed-dt)
        base-title (or title "Raster")
        start-time (System/nanoTime)
        frame-count (atom 0)
        fps-time (atom (System/nanoTime))
        sa (or state-atom game-state)]
    (reset! sa init-state)
    (try
      (loop [state init-state
             last-time (System/nanoTime)
             accumulator 0.0
             current-frame 0]
        (when-not (or (win/window-should-close? window)
                      (:quit? state))
          (win/poll-events!)

          ;; Check for external state changes via atom
          (let [state (let [ext @sa] (if (identical? ext state) state ext))

                ;; Resolve vars each frame for hot-reload
                upd-f (when update-fn (resolve-fn update-fn))
                cmp-f (when compute-fn (resolve-fn compute-fn))
                rnd-f (when render-fn (resolve-fn render-fn))

                ;; Inject browser input events (if streaming)
                _ (when input-system
                    (stream/inject-browser-events! input-system))

                ;; Poll input
                inp (when input-system (input/poll-input! input-system))

                ;; Fixed timestep accumulation
                now (System/nanoTime)
                frame-dt (min (/ (double (- now last-time)) 1e9) 0.1)
                acc (+ accumulator frame-dt)
                t (/ (double (- now start-time)) 1e9)

                ;; Run update steps
                [state' acc']
                (loop [s state a acc]
                  (if (>= a fixed-dt)
                    (let [s' (try
                               (if upd-f (upd-f s inp fixed-dt) s)
                               (catch Exception e
                                 (println "Update error:" (.getMessage e))
                                 s))]
                      (recur s' (- a fixed-dt)))
                    [s a]))

                ;; Per-frame-slot sync objects
                fi (int current-frame)
                frame-fence (aget in-flight-fences fi)
                acquire-sem (aget acquire-sems fi)
                ^VkCommandBuffer cb (aget cmd-bufs fi)]

            ;; Publish state to atom
            (reset! sa state')

            ;; 1. Wait for this frame slot's prior submit to complete
            ;;    This guarantees acquire-sem[fi] was consumed by that submit.
            ;;    1-second timeout prevents hang when window is not visible.
            (VK13/vkWaitForFences device frame-fence true 1000000000)
            (VK13/vkResetFences device frame-fence)

            ;; 2. Acquire swapchain image — signals acquire-sem[fi]
            ;;    Acquiring image N guarantees the prior present of image N completed,
            ;;    which means submit-sems[N] is now safe to signal again.
            (let [img-idx (win/acquire-next-image device (:swapchain sc) acquire-sem)]
              (if (neg? img-idx)
              ;; Acquire timed out — skip frame, poll events, continue
                (do (org.lwjgl.glfw.GLFW/glfwPollEvents)
                    (Thread/sleep 16))
                (let [submit-sem (aget submit-sems (int img-idx))
                      sc-image (nth (:images sc) img-idx)
                      sc-view (nth (:image-views sc) img-idx)]

              ;; 3. Reset and record commands into pre-allocated command buffer
                  (VK13/vkResetCommandBuffer cb 0)
                  (vk/begin-command-buffer cb)

              ;; Compute phase (optional)
                  (when cmp-f
                    (try (cmp-f state' cb ctx)
                         (catch Exception e (println "Compute error:" (.getMessage e)))))

              ;; Image barrier: undefined -> color attachment
                  (vk/cmd-image-barrier! cb sc-image
                                         [VK13/VK_IMAGE_LAYOUT_UNDEFINED
                                          VK13/VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL]
                                         [VK13/VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                          VK13/VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
                                         [0 VK13/VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT])

              ;; Begin dynamic rendering
                  (pipeline/cmd-begin-rendering! cb
                                                 {:color-view sc-view :depth-view depth-view}
                                                 [w h] clear-color)

                  (pipeline/cmd-set-viewport! cb w h)
                  (pipeline/cmd-set-scissor! cb w h)

              ;; Render phase
                  (let [frame-info {:image-index img-idx
                                    :image sc-image
                                    :image-view sc-view
                                    :extent [w h]
                                    :depth-view depth-view
                                    :t t
                                    :dt frame-dt}]
                    (when rnd-f
                      (try (rnd-f state' cb ctx frame-info)
                           (catch Exception e (println "Render error:" (.getMessage e))))))

                  (pipeline/cmd-end-rendering! cb)

              ;; Image barrier: color attachment -> present
                  (vk/cmd-image-barrier! cb sc-image
                                         [VK13/VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                          KHRSwapchain/VK_IMAGE_LAYOUT_PRESENT_SRC_KHR]
                                         [VK13/VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                          VK13/VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT]
                                         [VK13/VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT 0])

                  (vk/end-command-buffer cb)

              ;; 4. Submit using VK 1.3 vkQueueSubmit2 — explicit semaphore stage info
                  (let [stk (MemoryStack/stackPush)]
                    (try
                      (let [;; Wait: acquire-sem at color attachment output stage
                            ^VkSemaphoreSubmitInfo wait-info
                            (doto ^VkSemaphoreSubmitInfo (.get (VkSemaphoreSubmitInfo/calloc 1 stk) 0)
                              (.sType VK13/VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO)
                              (.semaphore acquire-sem)
                              (.stageMask VK13/VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT))
                        ;; Signal: submit-sem at all commands stage
                            ^VkSemaphoreSubmitInfo signal-info
                            (doto ^VkSemaphoreSubmitInfo (.get (VkSemaphoreSubmitInfo/calloc 1 stk) 0)
                              (.sType VK13/VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO)
                              (.semaphore submit-sem)
                              (.stageMask VK13/VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
                        ;; Command buffer wrapper
                            ^VkCommandBufferSubmitInfo cb-info
                            (doto ^VkCommandBufferSubmitInfo (.get (VkCommandBufferSubmitInfo/calloc 1 stk) 0)
                              (.sType VK13/VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO)
                              (.commandBuffer cb))
                        ;; Submit info 2
                            ^VkSubmitInfo2 si2
                            (doto ^VkSubmitInfo2 (.get (VkSubmitInfo2/calloc 1 stk) 0)
                              (.sType VK13/VK_STRUCTURE_TYPE_SUBMIT_INFO_2)
                              (.pWaitSemaphoreInfos
                               (VkSemaphoreSubmitInfo/create (.address wait-info) 1))
                              (.pCommandBufferInfos
                               (VkCommandBufferSubmitInfo/create (.address cb-info) 1))
                              (.pSignalSemaphoreInfos
                               (VkSemaphoreSubmitInfo/create (.address signal-info) 1)))
                            result (VK13/vkQueueSubmit2 queue
                                                        (VkSubmitInfo2/create (.address si2) 1)
                                                        frame-fence)]
                        (when-not (= result VK13/VK_SUCCESS)
                          (throw (ex-info "Queue submit2 failed" {:result result}))))
                      (finally
                        (MemoryStack/stackPop))))

              ;; Screenshot capture (if requested via REPL)
                  (try (screenshot/capture-if-requested! ctx sc-image w h)
                       (catch Throwable e
                         (binding [*out* *err*]
                           (println "Screenshot error:" (.getMessage e))
                           (.printStackTrace e))))

              ;; Stream capture (if streaming active)
                  (when-let [hook @stream/post-frame-hook]
                    (try (hook ctx sc-image w h)
                         (catch Throwable e
                           (println "Stream capture error:" (.getMessage e)))))

              ;; 5. Present: wait submit-sem[image]
                  (win/present! queue (:swapchain sc) img-idx submit-sem))))  ;; close let + if

            ;; FPS counter
            (let [fc (swap! frame-count inc)]
              (when (zero? (mod fc 60))
                (let [now2 (System/nanoTime)
                      elapsed (/ (double (- now2 @fps-time)) 1e9)
                      fps (/ 60.0 (max elapsed 0.001))
                      title (or (:title state') base-title)]
                  (reset! fps-time now2)
                  (GLFW/glfwSetWindowTitle window
                                           (format "%s -- %.0f fps" title fps)))))

            ;; Advance to next frame slot
            (recur state' now (double acc')
                   (int (mod (inc current-frame) MAX-FRAMES-IN-FLIGHT))))))
      (finally
      ;; Ensure GPU is idle before returning — prevents stale semaphore state
        (try (VK13/vkDeviceWaitIdle device) (catch Exception _))))))
