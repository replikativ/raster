(ns raster.render.vulkan
  "Vulkan/JVM backend for the portable renderer protocol (raster.render).

   Wraps raster.vk.* (LWJGL Vulkan 1.3, dynamic rendering, single queue). The
   neutral pipeline spec → a real graphics pipeline; a neutral dynamic mesh → a
   cpu-to-gpu vertex buffer; uniforms → push-constants; the run loop → the vk
   frame loop with a GLFW input-system mapped to the game's key-action set."
  (:require [raster.render :as r]
            [raster.render.shader :as shader]
            [raster.vk.gpu :as gpu]
            [raster.vk.frame :as frame]
            [raster.vk.pipeline :as pipeline]
            [raster.vk.mem :as mem]
            [raster.vk.input :as input])
  (:import [org.lwjgl.vulkan VK13 VkCommandBuffer]
           [org.lwjgl.system MemoryStack]
           [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

(defn- vk-format [fmt]
  (case fmt
    :float2 VK13/VK_FORMAT_R32G32_SFLOAT
    :float3 VK13/VK_FORMAT_R32G32B32_SFLOAT
    :float4 VK13/VK_FORMAT_R32G32B32A32_SFLOAT))

(defn- vk-topology [topo]
  (case topo :lines :lines :triangles :triangles :triangles))

(defn- vk-stage-flags [stages]
  (reduce (fn [acc s] (bit-or acc (case s
                                    :vertex VK13/VK_SHADER_STAGE_VERTEX_BIT
                                    :fragment VK13/VK_SHADER_STAGE_FRAGMENT_BIT)))
          0 stages))

;; --- per-frame command recorder ----------------------------------------------
(defrecord VkFrame [^VkCommandBuffer cb]
  r/Frame
  (-bind-pipeline! [_ pipeline]
    (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                            (long (:pipeline pipeline))))
  (-set-uniform! [_ pipeline floats]
    (let [stack (MemoryStack/stackPush)]
      (try
        (let [arr (float-array floats)
              fb  (.mallocFloat stack (alength arr))]
          (.put fb arr) (.flip fb)
          (VK13/vkCmdPushConstants cb (long (:layout pipeline))
                                   (int (:push-stages pipeline)) 0 fb))
        (finally (MemoryStack/stackPop)))))
  (-bind-mesh! [_ mesh]
    (let [stack (MemoryStack/stackPush)]
      (try
        (VK13/vkCmdBindVertexBuffers cb 0
                                     (.longs stack (long (:buffer (:buf mesh))))
                                     (.longs stack 0))
        (finally (MemoryStack/stackPop)))))
  (-draw! [_ vertex-count first-vertex]
    (VK13/vkCmdDraw cb (int vertex-count) 1 (int first-vertex) 0)))

;; --- the renderer ------------------------------------------------------------
(defrecord VulkanRenderer [ctx]
  r/Renderer
  (-make-pipeline [_ spec]
    (let [{:keys [shader topology vertex blend?]} spec
          {:keys [vert frag]} (shader/->glsl shader)
          stages (shader/uniform-stages shader)
          ubytes (shader/uniform-bytes shader)
          push-flags (vk-stage-flags stages)
          info (pipeline/create-graphics-pipeline
                (:device ctx)
                {:vert-glsl vert :frag-glsl frag
                 :topology (vk-topology topology)
                 :color-format (:format (:swapchain-info ctx))
                 :depth-format 0
                 :blend? (boolean blend?)
                 :push-constants (when (and (pos? ubytes) (seq stages))
                                   {:size ubytes :stages push-flags})
                 :vertex-bindings [{:binding 0 :stride (:stride vertex)}]
                 :vertex-attributes (mapv (fn [{:keys [location format offset]}]
                                            {:location location :binding 0
                                             :format (vk-format format) :offset offset})
                                          (:attributes vertex))})]
      ;; expose :pipeline + :layout + :push-stages for the Frame command verbs
      {:pipeline (:pipeline info) :layout (:layout info)
       :push-stages push-flags :info info}))
  (-make-mesh [_ max-verts stride]
    (let [buf (mem/create-buffer (:allocator ctx) (* (long max-verts) (long stride))
                                 :vertex :cpu-to-gpu)]
      {:buf buf :stride stride :max-verts max-verts :vertex-count 0}))
  (-upload-mesh! [_ mesh verts]
    (let [arr (float-array verts)
          floats-per-vert (quot (long (:stride mesh)) 4)]
      (mem/with-mapped-buffer [bb (:allocator ctx) (:buf mesh)]
        (let [^ByteBuffer bb bb
              fb (.asFloatBuffer bb)]
          (.put fb arr 0 (alength arr))
          (.flip fb)))
      (assoc mesh :vertex-count (quot (alength arr) floats-per-vert))))
  (-run! [_ {:keys [init-state update render clear-color title]
             :or {clear-color [0.0 0.0 0.0 1.0] title "Raster"}}]
    (let [fsync (frame/create-frame-sync ctx)
          inp-sys (input/create-input-system (:window ctx))
          ;; map GLFW key-down state -> the game's action set
          key->action {:up :up :left :left :right :right :space :fire}
          ->actions (fn [polled]
                      (persistent!
                       (reduce-kv (fn [s k a]
                                    (if (input/key-down? polled k) (conj! s a) s))
                                  (transient #{}) key->action)))]
      (frame/run-game-loop!
       ctx
       {:frame-sync fsync
        :init-state init-state
        :input-system inp-sys
        :clear-color clear-color
        :title title
        :update-fn (fn [state polled dt] (update state (->actions polled) dt))
        :render-fn (fn [state cb _ctx _frame-info] (render state (->VkFrame cb)))}))))

(defn make-renderer
  "Create a Vulkan renderer (window + device + swapchain). opts forwarded to
   raster.vk.gpu/create-gpu-context (e.g. :width :height :title)."
  [& {:as opts}]
  (->VulkanRenderer (apply gpu/create-gpu-context (mapcat identity opts))))

(defn destroy-renderer! [^VulkanRenderer r]
  (gpu/destroy-gpu-context! (:ctx r)))
