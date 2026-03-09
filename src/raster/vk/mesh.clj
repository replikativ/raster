(ns raster.vk.mesh
  "Static and dynamic mesh management with indexed draw support.
  Provides GPU-resident vertex+index buffers with staging upload."
  (:require
   [raster.vk.mem :as mem]
   [raster.vk.gpu :as gpu]
   [raster.vk.core :as vkcore])
  (:import
   [org.lwjgl.vulkan VK13 VkCommandBuffer VkBufferCopy]
   [org.lwjgl.system MemoryStack]
   [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

(defrecord Mesh [^raster.vk.mem.GpuBuffer vb
                 ^raster.vk.mem.GpuBuffer ib
                 ^long index-count
                 ^long vertex-count
                 ^long vertex-stride])

(defn create-static-mesh
  "Upload vertex + index data to GPU-only buffers via staging.
  vertices: float array, indices: int array, stride: bytes per vertex."
  [ctx ^floats vertices ^ints indices ^long stride]
  (let [allocator (:allocator ctx)
        vb-size (* (alength vertices) 4)
        ib-size (* (alength indices) 4)
        ;; Create GPU-only destination buffers
        vb (mem/create-buffer allocator vb-size :vertex :gpu-only)
        ib (mem/create-buffer allocator ib-size :index :gpu-only)
        ;; Create staging buffers
        vb-staging (mem/create-buffer allocator vb-size :staging :cpu-to-gpu)
        ib-staging (mem/create-buffer allocator ib-size :staging :cpu-to-gpu)]
    ;; Upload vertex data to staging
    (mem/with-mapped-buffer [bb allocator vb-staging]
      (let [^ByteBuffer bb bb
            fb (.asFloatBuffer bb)]
        (.put fb vertices)
        (.flip fb)))
    ;; Upload index data to staging
    (mem/with-mapped-buffer [bb allocator ib-staging]
      (let [^ByteBuffer bb bb
            ib-buf (.asIntBuffer bb)]
        (.put ib-buf indices)
        (.flip ib-buf)))
    ;; Copy staging → GPU via one-shot command buffer
    (gpu/one-shot-submit! ctx
                          (fn [^VkCommandBuffer cb]
                            (let [stack (MemoryStack/stackPush)
                                  region (VkBufferCopy/calloc 1 stack)]
          ;; Copy vertex buffer
                              (-> ^VkBufferCopy (.get region 0)
                                  (.srcOffset 0)
                                  (.dstOffset 0)
                                  (.size vb-size))
                              (VK13/vkCmdCopyBuffer cb (long (:buffer vb-staging)) (long (:buffer vb)) region)
          ;; Copy index buffer
                              (-> ^VkBufferCopy (.get region 0)
                                  (.srcOffset 0)
                                  (.dstOffset 0)
                                  (.size ib-size))
                              (VK13/vkCmdCopyBuffer cb (long (:buffer ib-staging)) (long (:buffer ib)) region)
                              (MemoryStack/stackPop))))
    ;; Clean up staging
    (mem/destroy-buffer! allocator vb-staging)
    (mem/destroy-buffer! allocator ib-staging)
    (->Mesh vb ib (alength indices) (/ vb-size stride) stride)))

(defn create-dynamic-mesh
  "Create CPU-to-GPU mapped buffers for per-frame geometry updates.
  max-verts/max-indices are element counts."
  [ctx ^long max-verts ^long max-indices ^long stride]
  (let [allocator (:allocator ctx)
        vb (mem/create-buffer allocator (* max-verts stride) :vertex :cpu-to-gpu)
        ib (mem/create-buffer allocator (* max-indices 4) :index :cpu-to-gpu)]
    (->Mesh vb ib 0 0 stride)))

(defn update-dynamic-mesh!
  "Update a dynamic mesh's vertex and index data. Returns updated Mesh."
  [ctx ^Mesh mesh ^floats vertices ^ints indices]
  (let [allocator (:allocator ctx)]
    (mem/with-mapped-buffer [bb allocator (:vb mesh)]
      (let [^ByteBuffer bb bb
            fb (.asFloatBuffer bb)]
        (.put fb vertices 0 (alength vertices))
        (.flip fb)))
    (mem/with-mapped-buffer [bb allocator (:ib mesh)]
      (let [^ByteBuffer bb bb
            ib (.asIntBuffer bb)]
        (.put ib indices 0 (alength indices))
        (.flip ib)))
    (assoc mesh
           :index-count (alength indices)
           :vertex-count (long (/ (* (alength vertices) 4) (:vertex-stride mesh))))))

(defn cmd-bind-mesh!
  "Bind vertex and index buffers for a mesh."
  [^VkCommandBuffer cb ^Mesh mesh]
  (let [stack (MemoryStack/stackPush)]
    (VK13/vkCmdBindVertexBuffers cb 0
                                 (.longs stack (long (:buffer (:vb mesh))))
                                 (.longs stack 0))
    (VK13/vkCmdBindIndexBuffer cb (long (:buffer (:ib mesh))) 0 VK13/VK_INDEX_TYPE_UINT32)
    (MemoryStack/stackPop)))

(defn cmd-draw-mesh!
  "Draw entire mesh with indexed draw call."
  [^VkCommandBuffer cb ^Mesh mesh]
  (cmd-bind-mesh! cb mesh)
  (VK13/vkCmdDrawIndexed cb (int (:index-count mesh)) 1 0 0 0))

(defn cmd-draw-mesh-range!
  "Draw a sub-range of a mesh's index buffer."
  [^VkCommandBuffer cb ^Mesh mesh ^long first-index ^long index-count]
  (VK13/vkCmdDrawIndexed cb (int index-count) 1 (int first-index) 0 0))

(defn destroy-mesh!
  "Destroy mesh vertex and index buffers."
  [ctx ^Mesh mesh]
  (let [allocator (:allocator ctx)]
    (mem/destroy-buffer! allocator (:vb mesh))
    (mem/destroy-buffer! allocator (:ib mesh))))
