(ns raster.vk.mem
  "Vulkan Memory Allocator (VMA) wrapper for buffer/image allocation."
  (:import
   [org.lwjgl.util.vma Vma VmaAllocatorCreateInfo VmaAllocationCreateInfo
    VmaAllocationInfo VmaVulkanFunctions]
   [org.lwjgl.vulkan VK13 VkInstance VkPhysicalDevice VkDevice
    VkBufferCreateInfo]
   [org.lwjgl.system MemoryStack MemoryUtil]))

(set! *warn-on-reflection* true)

;; ================================================================
;; VMA allocator
;; ================================================================

(defn create-allocator
  "Create a VMA allocator for the given device.
  Returns the allocator handle (long)."
  ^long [^VkInstance instance ^VkPhysicalDevice pd ^VkDevice device]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [vma-fns (doto (VmaVulkanFunctions/calloc stack)
                      (.set instance device))
            ci (doto (VmaAllocatorCreateInfo/calloc stack)
                 (.instance instance)
                 (.physicalDevice pd)
                 (.device device)
                 (.pVulkanFunctions vma-fns)
                 (.vulkanApiVersion (VK13/VK_API_VERSION_1_3)))
            p-alloc (.mallocPointer stack 1)
            result (Vma/vmaCreateAllocator ci p-alloc)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create VMA allocator" {:result result})))
        (.get p-alloc 0))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Buffer allocation
;; ================================================================

(defrecord GpuBuffer
           [^long buffer     ;; VkBuffer handle
            ^long allocation ;; VMA allocation handle
            ^long size       ;; size in bytes
            usage            ;; keyword
            ])

(defn create-buffer
  "Allocate a buffer via VMA.
  usage-kw: :vertex, :index, :storage, :uniform, :staging
  mem-kw: :gpu-only, :cpu-to-gpu, :gpu-to-cpu
  Returns GpuBuffer."
  [^long allocator size usage-kw mem-kw]
  (let [stack (MemoryStack/stackPush)
        size (long size)]
    (try
      (let [usage-bits (case usage-kw
                         :vertex  (bit-or VK13/VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                                          VK13/VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                         :index   (bit-or VK13/VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                                          VK13/VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                         :storage (bit-or VK13/VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                          VK13/VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                          VK13/VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                         :storage+vertex (bit-or VK13/VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                 VK13/VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                                                 VK13/VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                 VK13/VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                         :uniform VK13/VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                         :staging (bit-or VK13/VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                          VK13/VK_BUFFER_USAGE_TRANSFER_DST_BIT))
            mem-usage (case mem-kw
                        :gpu-only   Vma/VMA_MEMORY_USAGE_GPU_ONLY
                        :cpu-to-gpu Vma/VMA_MEMORY_USAGE_CPU_TO_GPU
                        :gpu-to-cpu Vma/VMA_MEMORY_USAGE_GPU_TO_CPU)
            buf-ci (doto (VkBufferCreateInfo/calloc stack)
                     (.sType VK13/VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                     (.size size)
                     (.usage (int usage-bits))
                     (.sharingMode VK13/VK_SHARING_MODE_EXCLUSIVE))
            alloc-ci (doto (VmaAllocationCreateInfo/calloc stack)
                       (.usage (int mem-usage)))
            p-buffer (.mallocLong stack 1)
            p-alloc (.mallocPointer stack 1)
            result (Vma/vmaCreateBuffer allocator buf-ci alloc-ci p-buffer p-alloc nil)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create buffer" {:result result :size size :usage usage-kw})))
        (->GpuBuffer (.get p-buffer 0) (.get p-alloc 0) size usage-kw))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Buffer mapping (for cpu-to-gpu / gpu-to-cpu)
;; ================================================================

(defn map-buffer
  "Map a VMA-allocated buffer. Returns a direct ByteBuffer.
  Only valid for :cpu-to-gpu or :gpu-to-cpu memory."
  [^long allocator ^GpuBuffer buf]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [pp-data (.mallocPointer stack 1)
            result (Vma/vmaMapMemory allocator (:allocation buf) pp-data)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to map buffer" {:result result})))
        (MemoryUtil/memByteBuffer (.get pp-data 0) (int (:size buf))))
      (finally
        (MemoryStack/stackPop)))))

(defn unmap-buffer
  "Unmap a VMA-allocated buffer."
  [^long allocator ^GpuBuffer buf]
  (Vma/vmaUnmapMemory allocator (:allocation buf)))

(defmacro with-mapped-buffer
  "Map buffer, execute body with bb bound to the ByteBuffer, then unmap.
  Usage: (with-mapped-buffer [bb allocator gpu-buf] (.putFloat bb 0 1.0))"
  [[sym allocator buf] & body]
  `(let [buf# ~buf
         alloc# ~allocator
         ~sym (map-buffer alloc# buf#)]
     (try
       ~@body
       (finally
         (unmap-buffer alloc# buf#)))))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-buffer!
  "Free a VMA-allocated buffer."
  [^long allocator ^GpuBuffer buf]
  (Vma/vmaDestroyBuffer allocator (:buffer buf) (:allocation buf)))

(defn destroy-allocator!
  "Destroy the VMA allocator."
  [^long allocator]
  (Vma/vmaDestroyAllocator allocator))

;; ================================================================
;; GpuSoA — Struct-of-Arrays GPU storage for defvalue types
;; ================================================================

(def ^:private dtype->bytes
  {:float 4 :double 8 :int 4 :long 8})

(defrecord GpuSoA
           [scalar-tag    ;; e.g. 'Particle
            soa-tag       ;; e.g. 'ParticleSoA
            n             ;; number of scalar elements
            field-segs    ;; ordered vec: [{:name "x" :dtype :float :buffer GpuBuffer} ...]
            ])

(defn gpu-array
  "Allocate GPU-resident storage for n elements of a defvalue SoA type.
  Looks up raster.core/soa-registry for field layout.
  usage-kw: :storage (default), :storage+vertex
  mem-kw: :gpu-only (default), :cpu-to-gpu, :gpu-to-cpu
  Returns GpuSoA."
  [allocator scalar-tag n
   & {:keys [usage mem] :or {usage :storage mem :gpu-only}}]
  (let [allocator (long allocator)
        n (long n)
        soa-info (get @(resolve 'raster.compiler.core.types/soa-registry) scalar-tag)]
    (when-not soa-info
      (throw (ex-info (str "No SoA registration for " scalar-tag) {:tag scalar-tag})))
    (let [fields (:fields soa-info)
          field-segs (mapv (fn [{:keys [name element-tag]}]
                             (let [dtype (case element-tag
                                           (float Float) :float
                                           (double Double) :double
                                           (int Integer) :int
                                           (long Long) :long
                                           :float)
                                   bytes (* n (long (get dtype->bytes dtype 4)))
                                   buf (create-buffer allocator bytes usage mem)]
                               {:name name :dtype dtype :buffer buf}))
                           fields)]
      (->GpuSoA scalar-tag (:soa-type-tag soa-info) n field-segs))))

(defn gpu-soa-buffers
  "Return vector of {:buffer handle :size bytes} maps for descriptor set binding,
  in canonical field order (matching kernel parameter order)."
  [^GpuSoA soa]
  (mapv (fn [{:keys [buffer]}]
          {:buffer (:buffer buffer) :size (:size buffer)})
        (:field-segs soa)))

(defn n-elements
  "Return element count for GpuBuffer (size/4 for float) or GpuSoA (n)."
  [buf]
  (cond
    (instance? GpuSoA buf) (:n buf)
    (instance? GpuBuffer buf) (quot (:size buf) 4)
    :else (throw (ex-info "Unknown buffer type" {:type (type buf)}))))

(defn destroy-gpu-soa!
  "Free all field buffers in a GpuSoA."
  [^long allocator ^GpuSoA soa]
  (doseq [{:keys [buffer]} (:field-segs soa)]
    (destroy-buffer! allocator buffer)))

;; ================================================================
;; SoA Transfer API
;; ================================================================

(defn copy-to-gpu!
  "Copy JVM SoA object fields to GPU SoA buffers.
  soa-obj is a defvalue SoA companion (e.g. Vec3SoA with float[] fields).
  Buffers must be :cpu-to-gpu or :gpu-to-cpu memory."
  [^long allocator ^GpuSoA gpu-soa soa-obj]
  (let [n (long (:n gpu-soa))]
    (doseq [{:keys [name dtype buffer]} (:field-segs gpu-soa)]
      (let [field-sym (symbol (str "." name))
            ;; Get the array from the SoA object via reflection
            arr (clojure.lang.Reflector/invokeInstanceMethod soa-obj name (object-array 0))]
        (with-mapped-buffer [bb allocator buffer]
          (case dtype
            :float  (let [^floats fa arr
                          fb (.asFloatBuffer bb)]
                      (.put fb fa 0 (min n (alength fa))))
            :double (let [^doubles da arr
                          db (.asDoubleBuffer bb)]
                      (.put db da 0 (min n (alength da))))
            :int    (let [^ints ia arr
                          ib (.asIntBuffer bb)]
                      (.put ib ia 0 (min n (alength ia))))
            :long   (let [^longs la arr
                          lb (.asLongBuffer bb)]
                      (.put lb la 0 (min n (alength la))))))))))

(defn copy-from-gpu!
  "Copy GPU SoA buffers back to JVM SoA object fields.
  soa-obj is a defvalue SoA companion (e.g. Vec3SoA with float[] fields).
  Buffers must be :gpu-to-cpu or :cpu-to-gpu memory."
  [^long allocator ^GpuSoA gpu-soa soa-obj]
  (let [n (long (:n gpu-soa))]
    (doseq [{:keys [name dtype buffer]} (:field-segs gpu-soa)]
      (let [arr (clojure.lang.Reflector/invokeInstanceMethod soa-obj name (object-array 0))]
        (with-mapped-buffer [bb allocator buffer]
          (case dtype
            :float  (let [^floats fa arr
                          fb (.asFloatBuffer bb)]
                      (.get fb fa 0 (min n (alength fa))))
            :double (let [^doubles da arr
                          db (.asDoubleBuffer bb)]
                      (.get db da 0 (min n (alength da))))
            :int    (let [^ints ia arr
                          ib (.asIntBuffer bb)]
                      (.get ib ia 0 (min n (alength ia))))
            :long   (let [^longs la arr
                          lb (.asLongBuffer bb)]
                      (.get lb la 0 (min n (alength la))))))))))
