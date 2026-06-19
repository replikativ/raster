(ns raster.render.vulkan
  "Vulkan/JVM backend for the portable renderer protocol (raster.render).

   Wraps raster.vk.* (LWJGL Vulkan 1.3, dynamic rendering, single queue). The
   neutral pipeline spec → a real graphics pipeline; neutral meshes → cpu-to-gpu
   (dynamic) or staged indexed (static) buffers; a neutral texture → a vk texture
   + combined-image-sampler descriptor set; uniforms → push-constants; the run
   loop → the vk frame loop with a GLFW input-system mapped to the key-action set.
   Depth is handled by the frame loop (it creates + attaches the depth image); a
   pipeline opts in via :depth?."
  (:require [raster.render :as r]
            [raster.render.shader :as shader]
            [raster.vk.gpu :as gpu]
            [raster.vk.frame :as frame]
            [raster.vk.pipeline :as pipeline]
            [raster.vk.mem :as mem]
            [raster.vk.mesh :as vkmesh]
            [raster.vk.texture :as texture]
            [raster.vk.input :as input])
  (:import [org.lwjgl.vulkan VK13 VkCommandBuffer]
           [org.lwjgl.system MemoryStack]
           [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

(defn- vk-format [fmt]
  (case fmt
    :float  VK13/VK_FORMAT_R32_SFLOAT
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

(defn- ensure-tex!
  "Lazily create the shared combined-image-sampler descriptor layout + pool. Sets
   allocated against it are compatible with the textured pipeline layout (identical
   binding)."
  [ctx tex-atom]
  (or @tex-atom
      (let [d (:device ctx)]
        (reset! tex-atom {:layout (texture/create-texture-descriptor-layout d)
                          :pool   (texture/create-texture-descriptor-pool d 64)}))))

(defn- pixels->bb ^ByteBuffer [pixels]
  (let [arr (byte-array (map unchecked-byte pixels))
        bb  (java.nio.ByteBuffer/allocateDirect (alength arr))]
    (.put bb arr) (.flip bb) bb))

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
        ;; dynamic mesh has :buf, static (indexed) mesh has :vb + :ib (vk.mesh.Mesh)
        (let [vb (long (:buffer (or (:vb mesh) (:buf mesh))))]
          (VK13/vkCmdBindVertexBuffers cb 0 (.longs stack vb) (.longs stack 0))
          (when-let [ib (:ib mesh)]
            (VK13/vkCmdBindIndexBuffer cb (long (:buffer ib)) 0 VK13/VK_INDEX_TYPE_UINT32)))
        (finally (MemoryStack/stackPop)))))
  (-bind-texture! [_ pipeline texture]
    (let [stack (MemoryStack/stackPush)]
      (try
        (VK13/vkCmdBindDescriptorSets cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                      (long (:layout pipeline)) 0
                                      (.longs stack (long (:desc-set texture))) nil)
        (finally (MemoryStack/stackPop)))))
  (-draw! [_ vertex-count first-vertex]
    (VK13/vkCmdDraw cb (int vertex-count) 1 (int first-vertex) 0))
  (-draw-indexed! [_ index-count first-index]
    (VK13/vkCmdDrawIndexed cb (int index-count) 1 (int first-index) 0 0)))

;; --- the renderer ------------------------------------------------------------
(defrecord VulkanRenderer [ctx tex]
  r/Renderer
  (-make-pipeline [_ spec]
    (let [{:keys [shader topology vertex blend? depth? cull]} spec
          {:keys [vert frag]} (shader/->glsl shader)
          stages (shader/uniform-stages shader)
          ubytes (shader/uniform-bytes shader)
          push-flags (vk-stage-flags stages)
          push-c (when (and (pos? ubytes) (seq stages)) {:size ubytes :stages push-flags})
          textured? (or (seq (:textures shader)) (seq (:texture-arrays shader)))
          lay (when textured? (:layout (texture/create-textured-pipeline-layout (:device ctx) push-c)))
          info (pipeline/create-graphics-pipeline
                (:device ctx)
                {:vert-glsl vert :frag-glsl frag
                 :topology (vk-topology topology)
                 :color-format (:format (:swapchain-info ctx))
                 :depth-format (if depth? VK13/VK_FORMAT_D32_SFLOAT 0)
                 :blend? (boolean blend?)
                 :cull-mode (or cull :none)
                 :push-constants push-c
                 :layout lay                       ; nil → created from push-constants
                 :vertex-bindings [{:binding 0 :stride (:stride vertex)}]
                 :vertex-attributes (mapv (fn [{:keys [location format offset]}]
                                            {:location location :binding 0
                                             :format (vk-format format) :offset offset})
                                          (:attributes vertex))})]
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
  (-make-static-mesh [_ verts indices stride]
    (vkmesh/create-static-mesh ctx (float-array verts) (int-array indices) stride))
  (-make-texture [_ {:keys [width height pixels filter]}]
    (let [{:keys [layout pool]} (ensure-tex! ctx tex)
          tex-rec (texture/create-texture ctx {:pixels (pixels->bb pixels) :width width :height height}
                                          :filter-mode (or filter :linear))
          ds (texture/allocate-texture-descriptor-set (:device ctx) pool layout tex-rec)]
      {:texture tex-rec :desc-set ds}))
  (-make-texture-array [_ {:keys [width height layers pixels filter]}]
    (let [{:keys [layout pool]} (ensure-tex! ctx tex)
          pv  (vec pixels)
          per (* (long width) (long height) 4)
          imgs (mapv (fn [li] {:name (str li)
                               :pixels (byte-array (map unchecked-byte (subvec pv (* li per) (* (inc li) per))))
                               :width width :height height})
                     (range layers))
          ta (texture/create-texture-array ctx imgs :filter-mode (or filter :nearest))
          ds (texture/allocate-texture-array-descriptor-set (:device ctx) pool layout ta)]
      {:texture-array ta :desc-set ds}))
  (-run! [_ {:keys [init-state update render clear-color title]
             :or {clear-color [0.0 0.0 0.0 1.0] title "Raster"}}]
    (let [fsync (frame/create-frame-sync ctx)
          inp-sys (input/create-input-system (:window ctx))
          key->action {:up :up :left :left :right :right :space :fire
                       :w :w :a :a :s :s :d :d}
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
  (->VulkanRenderer (apply gpu/create-gpu-context (mapcat identity opts)) (atom nil)))

(defn destroy-renderer! [^VulkanRenderer r]
  (gpu/destroy-gpu-context! (:ctx r)))
