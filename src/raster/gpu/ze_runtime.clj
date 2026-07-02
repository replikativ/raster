(ns raster.gpu.ze-runtime
  "Level Zero GPU runtime bindings via Panama FFM (Java 21+).

  Provides JVM access to Intel GPUs through the Level Zero API.
  All native calls use Panama Foreign Function & Memory API for
  zero-overhead interop without JNI.

  Level Zero is Intel's low-level GPU API, similar to Vulkan compute
  but purpose-built for GPGPU. Targets Intel Arc, Data Center Max,
  and integrated Xe graphics.

  Usage:
    (ze/init!)
    (ze/context)    ;; cached context handle
    (ze/device)     ;; default device handle

    ;; Module/kernel management
    (ze/load-module! spv-bytes)
    (ze/create-kernel module \"kernel_name\")

    ;; Memory
    (ze/alloc-shared n-bytes)
    (ze/free! segment)

    ;; Kernel launch
    (ze/launch! kernel group-count-x workgroup-size-x arg-segments)"
  (:refer-clojure :exclude [reset!])
  (:import [java.lang.foreign
            Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout
            AddressLayout]
           [java.lang.invoke MethodHandle]
           [java.nio.file Files Path])
  (:require [raster.compiler.core.types :as types]))

;; ================================================================
;; Library loading
;; ================================================================

(def ^:private ze-lib-paths
  "Search paths for Level Zero loader library."
  ["/usr/lib/x86_64-linux-gnu/libze_loader.so.1"
   "/usr/lib/x86_64-linux-gnu/libze_loader.so"
   "/usr/lib64/libze_loader.so.1"
   "/usr/lib64/libze_loader.so"
   "/usr/lib/libze_loader.so.1"
   "/usr/lib/libze_loader.so"
   "/opt/intel/oneapi/lib/libze_loader.so.1"])

(defn- find-ze-lib ^SymbolLookup []
  (let [loader (SymbolLookup/loaderLookup)]
    (if (.isPresent (.find loader "zeInit"))
      loader
      (or (some (fn [path]
                  (try
                    (let [lib (SymbolLookup/libraryLookup path (Arena/global))]
                      (when (.isPresent (.find lib "zeInit"))
                        lib))
                    (catch Exception _ nil)))
                ze-lib-paths)
          (throw (ex-info "Level Zero loader (libze_loader.so) not found"
                          {:searched ze-lib-paths}))))))

(def ^:private ze-lib (delay (find-ze-lib)))

;; ================================================================
;; Method handle creation
;; ================================================================

(defn- lookup-symbol ^MemorySegment [^String sym-name]
  (let [opt (.find ^SymbolLookup @ze-lib sym-name)]
    (when-not (.isPresent opt)
      (throw (ex-info (str "Level Zero symbol not found: " sym-name)
                      {:symbol sym-name})))
    (.get opt)))

(defn- fd
  "Create a FunctionDescriptor with return type and arg types.
  Java varargs require explicit into-array for Clojure interop."
  ^FunctionDescriptor [ret & args]
  (FunctionDescriptor/of ret (into-array MemoryLayout args)))

(defn- make-handle
  ^MethodHandle [^String symbol-name ^FunctionDescriptor fd]
  (.downcallHandle (Linker/nativeLinker)
                   (lookup-symbol symbol-name)
                   fd
                   (into-array Linker$Option [])))

;; ================================================================
;; Level Zero constants
;; ================================================================

(def ^:private ZE_INIT_FLAG_GPU_ONLY 1)
(def ^:private ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES 0x03)
(def ^:private ZE_STRUCTURE_TYPE_CONTEXT_DESC 0x0d)
(def ^:private ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC 0x0e)
(def ^:private ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC 0x15)
(def ^:private ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC 0x16)
(def ^:private ZE_STRUCTURE_TYPE_MODULE_DESC 0x1b)
(def ^:private ZE_STRUCTURE_TYPE_KERNEL_DESC 0x1d)
(def ^:private ZE_MODULE_FORMAT_IL_SPIRV 0x00)
(def ^:private ZE_MODULE_FORMAT_NATIVE 0x01)
(def ^:private ZE_RESULT_SUCCESS 0)

;; ================================================================
;; Layout helpers
;; ================================================================

(def ^:private PTR ValueLayout/ADDRESS)
(def ^:private I32 ValueLayout/JAVA_INT)
(def ^:private I64 ValueLayout/JAVA_LONG)
(def ^:private F64 ValueLayout/JAVA_DOUBLE)

(defn- ptr-seg
  "Allocate a pointer-sized segment in the given arena."
  ^MemorySegment [^Arena arena]
  (.allocate arena PTR))

(defn- int-seg
  "Allocate an int segment with a value."
  ^MemorySegment [^Arena arena ^long v]
  (let [seg (.allocate arena I32)]
    (.set seg I32 0 (int v))
    seg))

(defn- read-ptr
  "Read a pointer from a segment at offset 0."
  ^MemorySegment [^MemorySegment seg]
  (.get seg PTR 0))

(defn- read-int
  "Read an int from a segment at offset 0."
  ^long [^MemorySegment seg]
  (long (.get seg I32 0)))

;; ================================================================
;; Method handles (lazy, created on first use)
;; ================================================================

(def ^:private h-zeInit
  (delay (make-handle "zeInit" (fd I32 I32))))

(def ^:private h-zeDriverGet
  (delay (make-handle "zeDriverGet" (fd I32 PTR PTR))))

(def ^:private h-zeDeviceGet
  (delay (make-handle "zeDeviceGet" (fd I32 PTR PTR PTR))))

(def ^:private h-zeDeviceGetProperties
  (delay (make-handle "zeDeviceGetProperties" (fd I32 PTR PTR))))

(def ^:private h-zeContextCreate
  (delay (make-handle "zeContextCreate" (fd I32 PTR PTR PTR))))

(def ^:private h-zeCommandListCreateImmediate
  (delay (make-handle "zeCommandListCreateImmediate" (fd I32 PTR PTR PTR PTR))))

(def ^:private h-zeModuleCreate
  (delay (make-handle "zeModuleCreate" (fd I32 PTR PTR PTR PTR PTR))))

(def ^:private h-zeKernelCreate
  (delay (make-handle "zeKernelCreate" (fd I32 PTR PTR PTR))))

(def ^:private h-zeKernelSetGroupSize
  (delay (make-handle "zeKernelSetGroupSize" (fd I32 PTR I32 I32 I32))))

(def ^:private h-zeKernelSetArgumentValue
  (delay (make-handle "zeKernelSetArgumentValue" (fd I32 PTR I32 I64 PTR))))

(def ^:private h-zeMemAllocShared
  (delay (make-handle "zeMemAllocShared" (fd I32 PTR PTR PTR I64 I64 PTR PTR))))

(def ^:private h-zeMemAllocDevice
  (delay (make-handle "zeMemAllocDevice" (fd I32 PTR PTR I64 I64 PTR PTR))))

(def ^:private h-zeMemAllocHost
  (delay (make-handle "zeMemAllocHost" (fd I32 PTR PTR I64 I64 PTR))))

(def ^:private h-zeMemFree
  (delay (make-handle "zeMemFree" (fd I32 PTR PTR))))

(def ^:private h-zeCommandListAppendLaunchKernel
  (delay (make-handle "zeCommandListAppendLaunchKernel" (fd I32 PTR PTR PTR PTR I32 PTR))))

(def ^:private h-zeCommandListAppendMemoryCopy
  (delay (make-handle "zeCommandListAppendMemoryCopy" (fd I32 PTR PTR PTR I64 PTR I32 PTR))))

(def ^:private h-zeCommandListAppendBarrier
  (delay (make-handle "zeCommandListAppendBarrier" (fd I32 PTR PTR I32 PTR))))

;; --- Regular (replayable) command list + queue: enqueue-all-sync-once (command graph) ---
(def ^:private ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC 0x0f)
(def ^:private h-zeCommandQueueCreate
  (delay (make-handle "zeCommandQueueCreate" (fd I32 PTR PTR PTR PTR))))
(def ^:private h-zeCommandListCreate
  (delay (make-handle "zeCommandListCreate" (fd I32 PTR PTR PTR PTR))))
(def ^:private h-zeCommandListClose
  (delay (make-handle "zeCommandListClose" (fd I32 PTR))))
(def ^:private h-zeCommandQueueExecuteCommandLists
  (delay (make-handle "zeCommandQueueExecuteCommandLists" (fd I32 PTR I32 PTR PTR))))
(def ^:private h-zeCommandListDestroy
  (delay (make-handle "zeCommandListDestroy" (fd I32 PTR))))
(def ^:private h-zeCommandQueueDestroy
  (delay (make-handle "zeCommandQueueDestroy" (fd I32 PTR))))
(def ^:private h-zeKernelDestroy
  (delay (make-handle "zeKernelDestroy" (fd I32 PTR))))

(def ^:private h-zeModuleDestroy
  (delay (make-handle "zeModuleDestroy" (fd I32 PTR))))

(def ^:private h-zeKernelDestroy
  (delay (make-handle "zeKernelDestroy" (fd I32 PTR))))

;; ================================================================
;; State
;; ================================================================

(defonce ^:private state
  (atom {:initialized? false
         :driver nil       ;; MemorySegment (ze_driver_handle_t)
         :device nil       ;; MemorySegment (ze_device_handle_t)
         :context nil      ;; MemorySegment (ze_context_handle_t)
         :cmd-list nil     ;; MemorySegment (ze_command_list_handle_t)
         :arena nil        ;; Arena for long-lived allocations
         :modules {}       ;; hash -> module handle
         :kernels {}}))    ;; [module kernel-name] -> kernel handle

;; ================================================================
;; Invocation helper (uses invokeWithArguments for boxing compat)
;; ================================================================

(defn- ze-call!
  "Invoke a Level Zero function and check result.
  Uses invokeWithArguments for Clojure boxing compatibility."
  [^String context ^MethodHandle mh args]
  (let [result (int (.invokeWithArguments mh ^java.util.List (java.util.List/of (object-array args))))]
    (when-not (== result ZE_RESULT_SUCCESS)
      (throw (ex-info (str "Level Zero error in " context ": 0x"
                           (Integer/toHexString result))
                      {:result result :context context})))
    result))

;; ================================================================
;; Initialization
;; ================================================================

(defn init!
  "Initialize Level Zero runtime. Idempotent.
  Finds first GPU device, creates context and immediate command list."
  []
  (when-not (:initialized? @state)
    (let [arena (Arena/ofShared)]
      ;; zeInit
      (ze-call! "zeInit" @h-zeInit [(int ZE_INIT_FLAG_GPU_ONLY)])

      ;; zeDriverGet — get count, then first driver
      (let [count-seg (int-seg arena 0)]
        (ze-call! "zeDriverGet(count)" @h-zeDriverGet [count-seg MemorySegment/NULL])
        (let [n-drivers (read-int count-seg)]
          (when (zero? n-drivers)
            (throw (ex-info "No Level Zero drivers found" {})))

          (let [drivers-seg (.allocate arena (MemoryLayout/sequenceLayout n-drivers PTR))
                _ (.set count-seg I32 0 (int n-drivers))
                _ (ze-call! "zeDriverGet(handles)" @h-zeDriverGet [count-seg drivers-seg])
                driver (.get drivers-seg PTR 0)

                ;; zeDeviceGet — get first GPU device
                dev-count-seg (int-seg arena 0)
                _ (ze-call! "zeDeviceGet(count)" @h-zeDeviceGet
                            [driver dev-count-seg MemorySegment/NULL])
                n-devices (read-int dev-count-seg)]
            (when (zero? n-devices)
              (throw (ex-info "No Level Zero devices found" {})))

            (let [devices-seg (.allocate arena (MemoryLayout/sequenceLayout n-devices PTR))
                  _ (.set dev-count-seg I32 0 (int n-devices))
                  _ (ze-call! "zeDeviceGet(handles)" @h-zeDeviceGet
                              [driver dev-count-seg devices-seg])
                  device (.get devices-seg PTR 0)

                  ;; zeContextCreate
                  ctx-desc (.allocate arena 24)
                  _ (.set ctx-desc I32 0 (int ZE_STRUCTURE_TYPE_CONTEXT_DESC))
                  ctx-out (ptr-seg arena)
                  _ (ze-call! "zeContextCreate" @h-zeContextCreate
                              [driver ctx-desc ctx-out])
                  context (read-ptr ctx-out)

                  ;; zeCommandListCreateImmediate (synchronous mode)
                  cq-desc (.allocate arena 40)
                  _ (.set cq-desc I32 0 (int ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC))
                  _ (.set cq-desc I32 28 (int 1)) ;; ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS
                  cmd-out (ptr-seg arena)
                  _ (ze-call! "zeCommandListCreateImmediate" @h-zeCommandListCreateImmediate
                              [context device cq-desc cmd-out])
                  cmd-list (read-ptr cmd-out)]

              ;; Query device ID for ocloc compilation
              (let [dev-props (.allocate arena 512)
                    _ (.set dev-props I32 0 (int ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES))
                    _ (ze-call! "zeDeviceGetProperties" @h-zeDeviceGetProperties
                                [device dev-props])
                    ;; ze_device_properties_t: stype@0, pNext@8, type@16, vendorId@20, deviceId@24
                    device-id-val (.get dev-props I32 24)]
                (clojure.core/reset! state
                                     {:initialized? true
                                      :driver driver
                                      :device device
                                      :context context
                                      :cmd-list cmd-list
                                      :arena arena
                                      :device-id-hex (format "0x%04x" device-id-val)
                                      :modules {}
                                      :kernels {}}))))))
      nil)))

(defn- ensure-init! []
  (when-not (:initialized? @state)
    (init!)))

;; ================================================================
;; Public accessors
;; ================================================================

(defn context
  "Get the cached Level Zero context handle."
  ^MemorySegment []
  (ensure-init!)
  (:context @state))

(defn device
  "Get the default Level Zero device handle."
  ^MemorySegment []
  (ensure-init!)
  (:device @state))

(defn driver
  "Get the Level Zero driver handle."
  ^MemorySegment []
  (ensure-init!)
  (:driver @state))

;; ================================================================
;; Device queries (for hardware detection)
;; ================================================================

(defn query-devices
  "Query all Level Zero devices and return their properties.
  Returns a vector of device info maps. Used by raster.runtime.hardware."
  []
  (let [arena (Arena/ofConfined)]
    (try
      (ze-call! "zeInit" @h-zeInit [(int ZE_INIT_FLAG_GPU_ONLY)])
      (let [count-seg (int-seg arena 0)]
        (ze-call! "zeDriverGet(count)" @h-zeDriverGet [count-seg MemorySegment/NULL])
        (let [n-drivers (read-int count-seg)]
          (if (zero? n-drivers)
            []
            (let [drivers-seg (.allocate arena (MemoryLayout/sequenceLayout n-drivers PTR))
                  _ (.set count-seg I32 0 (int n-drivers))
                  _ (ze-call! "zeDriverGet(handles)" @h-zeDriverGet [count-seg drivers-seg])]
              (vec
               (mapcat
                (fn [di]
                  (let [drv (.get drivers-seg PTR (* (long di) (.byteSize PTR)))
                        dev-count-seg (int-seg arena 0)
                        _ (ze-call! "zeDeviceGet(count)" @h-zeDeviceGet
                                    [drv dev-count-seg MemorySegment/NULL])
                        n-devs (read-int dev-count-seg)]
                    (when (pos? n-devs)
                      (let [devs-seg (.allocate arena (MemoryLayout/sequenceLayout n-devs PTR))
                            _ (.set dev-count-seg I32 0 (int n-devs))
                            _ (ze-call! "zeDeviceGet(handles)" @h-zeDeviceGet
                                        [drv dev-count-seg devs-seg])]
                        (mapv
                         (fn [dj]
                           (let [dev (.get devs-seg PTR (* (long dj) (.byteSize PTR)))
                                    ;; ze_device_properties_t: allocate generously
                                 props (.allocate arena 512)
                                 _ (.set props I32 0 (int ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES))
                                 _ (ze-call! "zeDeviceGetProperties" @h-zeDeviceGetProperties
                                             [dev props])
                                    ;; Struct offsets (x86_64, 8-byte aligned):
                                    ;; 0: stype(4) + 4: pad(4) + 8: pNext(8) = 16
                                    ;; 16: type(4) + 20: vendorId(4) + 24: deviceId(4) + 28: flags(4) = 32
                                    ;; 32: subdeviceId(4) + 36: coreClockRate(4) = 40
                                    ;; 40: maxMemAllocSize(8) = 48
                                    ;; 48: maxHardwareContexts(4) + 52: maxCommandQueuePriority(4) = 56
                                    ;; 56: numThreadsPerEU(4) + 60: physicalEUSimdWidth(4) = 64
                                    ;; 64: numEUsPerSubslice(4) + 68: numSubslicesPerSlice(4) = 72
                                    ;; 72: numSlices(4) + 76: timerResolution(8, aligned to 80) = 88
                                    ;; 88: timestampValidBits(4) + 92: kernelTimestampValidBits(4) = 96
                                    ;; 96: uuid(16 bytes) = 112
                                    ;; 112: name[ZE_MAX_DEVICE_NAME=256]
                                 device-id-val (.get props I32 24)
                                    ;; flags at offset 28: bit 0 = ZE_DEVICE_PROPERTY_FLAG_INTEGRATED
                                 flags-val (.get props I32 28)
                                 integrated? (not (zero? (bit-and flags-val 1)))
                                 core-clock (.get props I32 36)
                                 threads-per-eu (.get props I32 56)
                                 simd-width (.get props I32 60)
                                 eus-per-subslice (.get props I32 64)
                                 subslices-per-slice (.get props I32 68)
                                 num-slices (.get props I32 72)
                                    ;; Read name at offset 112
                                 name-offset 112
                                 name-bytes (byte-array 256)
                                 name-seg (.asSlice props (long name-offset) 256)
                                 _ (MemorySegment/copy name-seg 0
                                                       (MemorySegment/ofArray name-bytes) 0 256)
                                 dev-name (let [end (or (some #(when (zero? (aget name-bytes (int %))) %)
                                                              (range 256))
                                                        256)]
                                            (String. name-bytes 0 (int end) "UTF-8"))
                                 total-eus (* eus-per-subslice subslices-per-slice num-slices)]
                             {:name (.trim dev-name)
                              :device-id-hex (format "0x%04x" device-id-val)
                              :integrated? integrated?
                              :core-clock-mhz core-clock
                              :total-eus total-eus
                              :threads-per-eu threads-per-eu
                              :simd-width simd-width
                              :eus-per-subslice eus-per-subslice
                              :subslices-per-slice subslices-per-slice
                              :num-slices num-slices}))
                         (range n-devs))))))
                (range n-drivers)))))))
      (finally
        (.close arena)))))

;; ================================================================
;; Module / Kernel management
;; ================================================================

(def ^:private ZE_MODULE_FORMAT_NATIVE 0x01)

(defn load-module!
  "Load a SPIR-V or native module from bytes. Returns the module handle.
  Modules are cached by content hash.
  format: :spirv (default) or :native for pre-compiled ZEBIN."
  (^MemorySegment [^bytes spv-bytes]
   (load-module! spv-bytes :spirv))
  (^MemorySegment [^bytes spv-bytes format]
   (ensure-init!)
   (let [hash (java.util.Arrays/hashCode spv-bytes)]
     (if-let [cached (get-in @state [:modules hash])]
       cached
       (let [arena (:arena @state)
             ctx (:context @state)
             dev (:device @state)
             ;; ze_module_desc_t layout (x86_64):
             ;; 0: stype(4) + 4: pad(4) + 8: pNext(8) + 16: format(4) + 20: pad(4)
             ;; 24: inputSize(8) + 32: pInputModule(8) + 40: pBuildFlags(8) + 48: pConstants(8)
             mod-desc (.allocate arena 64)
             fmt (case format
                   :spirv  ZE_MODULE_FORMAT_IL_SPIRV
                   :native ZE_MODULE_FORMAT_NATIVE)
             _ (.set mod-desc I32 0 (int ZE_STRUCTURE_TYPE_MODULE_DESC))
             _ (.set mod-desc I32 16 (int fmt))
             _ (.set mod-desc I64 24 (long (alength spv-bytes)))
             spv-seg (.allocateFrom arena ValueLayout/JAVA_BYTE spv-bytes)
             _ (.set mod-desc PTR 32 spv-seg)
             mod-out (ptr-seg arena)
             _ (ze-call! "zeModuleCreate" @h-zeModuleCreate
                         [ctx dev mod-desc mod-out MemorySegment/NULL])
             module (read-ptr mod-out)]
         (swap! state assoc-in [:modules hash] module)
         module)))))

(defn create-kernel
  "Create a kernel from a loaded module. Returns the kernel handle.
  Cached by [module, kernel-name]."
  ^MemorySegment [^MemorySegment module ^String kernel-name]
  (ensure-init!)
  (let [cache-key [module kernel-name]]
    (if-let [cached (get-in @state [:kernels cache-key])]
      cached
      (let [arena (:arena @state)
            kern-desc (.allocate arena 32)
            _ (.set kern-desc I32 0 (int ZE_STRUCTURE_TYPE_KERNEL_DESC))
            name-seg (.allocateFrom arena kernel-name)
            _ (.set kern-desc PTR 24 name-seg)
            kern-out (ptr-seg arena)
            _ (ze-call! "zeKernelCreate" @h-zeKernelCreate
                        [module kern-desc kern-out])
            kernel (read-ptr kern-out)]
        (swap! state assoc-in [:kernels cache-key] kernel)
        kernel))))

;; ================================================================
;; Memory allocation
;; ================================================================

(defn alloc-shared
  "Allocate shared (host+device visible) memory. Returns MemorySegment."
  ^MemorySegment [^long n-bytes]
  (ensure-init!)
  (let [arena (:arena @state)
        ctx (:context @state)
        dev (:device @state)
        dev-desc (.allocate arena 24)
        _ (.set dev-desc I32 0 (int ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC))
        host-desc (.allocate arena 24)
        _ (.set host-desc I32 0 (int ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC))
        out-ptr (ptr-seg arena)]
    (ze-call! "zeMemAllocShared" @h-zeMemAllocShared
              [ctx dev-desc host-desc (long n-bytes) (long 64) dev out-ptr])
    (.reinterpret (read-ptr out-ptr) n-bytes)))

(defn alloc-device
  "Allocate device-only memory. Returns MemorySegment."
  ^MemorySegment [^long n-bytes]
  (ensure-init!)
  (let [arena (:arena @state)
        ctx (:context @state)
        dev (:device @state)
        dev-desc (.allocate arena 24)
        _ (.set dev-desc I32 0 (int ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC))
        out-ptr (ptr-seg arena)]
    (ze-call! "zeMemAllocDevice" @h-zeMemAllocDevice
              [ctx dev-desc (long n-bytes) (long 64) dev out-ptr])
    (.reinterpret (read-ptr out-ptr) n-bytes)))

(defn alloc-host
  "Allocate host-pinned memory accessible by GPU via DMA. Returns MemorySegment.
  Use on discrete GPUs: CPU writes here, GPU reads via PCIe without extra copy."
  ^MemorySegment [^long n-bytes]
  (ensure-init!)
  (let [arena (:arena @state)
        ctx   (:context @state)
        host-desc (.allocate arena 24)
        _ (.set host-desc I32 0 (int ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC))
        out-ptr (ptr-seg arena)]
    (ze-call! "zeMemAllocHost" @h-zeMemAllocHost
              [ctx host-desc (long n-bytes) (long 64) out-ptr])
    (.reinterpret (read-ptr out-ptr) n-bytes)))

(defn free!
  "Free a Level Zero memory allocation."
  [^MemorySegment segment]
  (ensure-init!)
  (ze-call! "zeMemFree" @h-zeMemFree [(:context @state) segment]))

;; ================================================================
;; Data transfer
;; ================================================================

(defn copy!
  "Copy n-bytes from src to dst using the command list. Synchronous."
  [^MemorySegment dst ^MemorySegment src ^long n-bytes]
  (ensure-init!)
  (let [cmd (:cmd-list @state)]
    (ze-call! "zeCommandListAppendMemoryCopy" @h-zeCommandListAppendMemoryCopy
              [cmd dst src (long n-bytes) MemorySegment/NULL (int 0) MemorySegment/NULL])
    (ze-call! "zeCommandListAppendBarrier" @h-zeCommandListAppendBarrier
              [cmd MemorySegment/NULL (int 0) MemorySegment/NULL])))

(defn copy-to-device!
  "Copy a JVM double[] to a shared/device MemorySegment."
  [^MemorySegment dst ^doubles src]
  (let [n-bytes (* (alength src) 8)
        src-seg (MemorySegment/ofArray src)]
    (MemorySegment/copy src-seg 0 dst 0 n-bytes)))

(defn copy-from-device!
  "Copy from a shared/device MemorySegment to a JVM double[]."
  [^doubles dst ^MemorySegment src]
  (let [n-bytes (* (alength dst) 8)
        dst-seg (MemorySegment/ofArray dst)]
    (MemorySegment/copy src 0 dst-seg 0 n-bytes)))

;; ================================================================
;; Kernel launch
;; ================================================================

(defn launch!
  "Launch a kernel with the given group count.
  Sets workgroup size and arguments, then dispatches.
  Synchronous (uses immediate command list with barrier).

  kernel: kernel handle from create-kernel
  group-count-x: number of workgroups in X dimension
  workgroup-size-x: threads per workgroup in X
  kernel-args: seq of kernel argument specs, each one of:
    - MemorySegment (treated as pointer arg)
    - {:type :int, :value N} (int scalar)
    - {:type :long, :value N} (long scalar)
    - {:type :float, :value N} (float scalar)
    - {:type :double, :value N} (double scalar)"
  [^MemorySegment kernel ^long group-count-x ^long workgroup-size-x
   kernel-args]
  (ensure-init!)
  (let [cmd (:cmd-list @state)
        arena (:arena @state)]
    ;; Set group size
    (ze-call! "zeKernelSetGroupSize" @h-zeKernelSetGroupSize
              [kernel (int workgroup-size-x) (int 1) (int 1)])

    ;; Set arguments
    (doseq [[idx arg] (map-indexed vector kernel-args)]
      (if (instance? MemorySegment arg)
        ;; Pointer argument: pass pointer-to-pointer with size = sizeof(ptr)
        (let [arg-ptr (ptr-seg arena)]
          (.set arg-ptr PTR 0 ^MemorySegment arg)
          (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                    [kernel (int idx) (long (.byteSize PTR)) arg-ptr]))
        ;; Scalar argument: pass pointer-to-value with correct size
        (let [{:keys [type value]} arg]
          (case type
            :int    (let [s (.allocate ^Arena arena I32)]
                      (.set s I32 0 (int value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :long   (let [s (.allocate ^Arena arena I64)]
                      (.set s I64 0 (long value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))
            :float  (let [s (.allocate ^Arena arena ValueLayout/JAVA_FLOAT)]
                      (.set s ValueLayout/JAVA_FLOAT 0 (float value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :double (let [s (.allocate ^Arena arena F64)]
                      (.set s F64 0 (double value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))))))

    ;; Build ze_group_count_t: { groupCountX(u32), groupCountY(u32), groupCountZ(u32) }
    (let [gc (.allocate arena 12)]
      (.set gc I32 0 (int group-count-x))
      (.set gc I32 4 (int 1))
      (.set gc I32 8 (int 1))

      ;; Launch
      (ze-call! "zeCommandListAppendLaunchKernel" @h-zeCommandListAppendLaunchKernel
                [cmd kernel gc MemorySegment/NULL (int 0) MemorySegment/NULL])

      ;; Barrier for synchronous completion
      (ze-call! "zeCommandListAppendBarrier" @h-zeCommandListAppendBarrier
                [cmd MemorySegment/NULL (int 0) MemorySegment/NULL]))))

;; ================================================================
;; Persistent device buffers
;; ================================================================

(defrecord DeviceBuffer [^MemorySegment segment ^long n-elements ^long byte-size dtype])

(defn device-buffer?
  "Check if x is a DeviceBuffer."
  [x]
  (instance? DeviceBuffer x))

(def ^:private dtype-byte-sizes
  {:double 8 :float 4 :float32 4 :int 4 :long 8 :half 2 :float16 2})

(defn make-buffer
  "Allocate a persistent shared-memory GPU buffer.
  Returns a DeviceBuffer that survives across kernel launches.

  n: number of elements
  dtype: :double, :float, :int, :long, or :half (default :float)"
  ([n] (make-buffer n :float))
  ([n dtype]
   (let [n (long n)
         elem-size (long (get dtype-byte-sizes dtype 4))
         byte-size (* n elem-size)
         seg (alloc-shared byte-size)]
     (->DeviceBuffer seg n byte-size dtype))))

(defn make-buffer-like
  "Allocate a DeviceBuffer with the same shape/dtype as an existing one."
  [^DeviceBuffer buf]
  (make-buffer (:n-elements buf) (:dtype buf)))

(defn free-buffer!
  "Free a DeviceBuffer's GPU memory."
  [^DeviceBuffer buf]
  (free! (:segment buf)))

(defn buffer-as-float-buffer
  "Return a java.nio.FloatBuffer view over a :float DeviceBuffer's shared memory.
  Zero-copy on unified-memory GPUs: reads/writes go directly to GPU-accessible memory.
  The buffer uses native byte order. Valid only while the DeviceBuffer is alive."
  ^java.nio.FloatBuffer [^DeviceBuffer buf]
  (-> (.asByteBuffer ^MemorySegment (:segment buf))
      (.order (java.nio.ByteOrder/nativeOrder))
      (.asFloatBuffer)))

(defn buffer-as-int-buffer
  "Return a java.nio.IntBuffer view over a :int DeviceBuffer's shared memory."
  ^java.nio.IntBuffer [^DeviceBuffer buf]
  (-> (.asByteBuffer ^MemorySegment (:segment buf))
      (.order (java.nio.ByteOrder/nativeOrder))
      (.asIntBuffer)))

(defn buffer-as-long-buffer
  "Return a java.nio.LongBuffer view over a :long DeviceBuffer's shared memory."
  ^java.nio.LongBuffer [^DeviceBuffer buf]
  (-> (.asByteBuffer ^MemorySegment (:segment buf))
      (.order (java.nio.ByteOrder/nativeOrder))
      (.asLongBuffer)))

(defn array->buffer!
  "Copy a JVM array into an existing DeviceBuffer. Returns the buffer."
  [^DeviceBuffer buf arr]
  (let [seg (:segment buf)
        src (MemorySegment/ofArray arr)
        n-bytes (min (:byte-size buf) (.byteSize src))]
    (MemorySegment/copy src 0 seg 0 n-bytes)
    buf))

(defn buffer->array
  "Copy a DeviceBuffer's contents to a new JVM array.
  For :float16/:half, returns a short array of encoded FP16 values.
  Use buffer->float-array or buffer->double-array for decoded values."
  [^DeviceBuffer buf]
  (let [seg (:segment buf)
        dtype (:dtype buf)
        n (:n-elements buf)]
    (case dtype
      :float  (let [out (float-array n)
                    dst (MemorySegment/ofArray out)]
                (MemorySegment/copy seg 0 dst 0 (:byte-size buf))
                out)
      :double (let [out (double-array n)
                    dst (MemorySegment/ofArray out)]
                (MemorySegment/copy seg 0 dst 0 (:byte-size buf))
                out)
      :int    (let [out (int-array n)
                    dst (MemorySegment/ofArray out)]
                (MemorySegment/copy seg 0 dst 0 (:byte-size buf))
                out)
      :long   (let [out (long-array n)
                    dst (MemorySegment/ofArray out)]
                (MemorySegment/copy seg 0 dst 0 (:byte-size buf))
                out)
      (:float16 :half)
      (let [out (short-array n)
            dst (MemorySegment/ofArray out)]
        (MemorySegment/copy seg 0 dst 0 (:byte-size buf))
        out))))

(defn buffer-of-array
  "Create a new DeviceBuffer from a JVM array (allocates + copies).
  dtype is auto-detected from array type if not specified."
  ([arr] (buffer-of-array arr nil))
  ([arr dtype]
   (let [dtype (or dtype
                   (cond (instance? (Class/forName "[F") arr) :float
                         (instance? (Class/forName "[D") arr) :double
                         (instance? (Class/forName "[I") arr) :int
                         (instance? (Class/forName "[J") arr) :long
                         :else :float))
         n (cond (instance? (Class/forName "[F") arr) (alength ^floats arr)
                 (instance? (Class/forName "[D") arr) (alength ^doubles arr)
                 (instance? (Class/forName "[I") arr) (alength ^ints arr)
                 (instance? (Class/forName "[J") arr) (alength ^longs arr)
                 :else (throw (ex-info "Unsupported array type" {:type (type arr)})))
         buf (make-buffer n dtype)]
     (array->buffer! buf arr))))

(defn zero-buffer!
  "Zero out a DeviceBuffer. Returns the buffer."
  [^DeviceBuffer buf]
  (let [seg (:segment buf)]
    (.fill seg (byte 0))
    buf))

(defn buffer-of-floats-as-half
  "Create a :float16 DeviceBuffer from a float array.
  Converts each float32 to float16 using Float/floatToFloat16."
  [^floats arr]
  (let [n (alength arr)
        shorts (short-array n)
        _ (dotimes [i n]
            (aset shorts i (short (Float/floatToFloat16 (aget arr i)))))
        buf (make-buffer n :float16)
        seg (:segment buf)
        src (MemorySegment/ofArray shorts)]
    (MemorySegment/copy src 0 seg 0 (* n 2))
    buf))

(defn buffer-of-doubles-as-half
  "Create a :float16 DeviceBuffer from a double array.
  Converts each float64 → float32 → float16."
  [^doubles arr]
  (let [n (alength arr)
        shorts (short-array n)
        _ (dotimes [i n]
            (aset shorts i (short (Float/floatToFloat16 (float (aget arr i))))))
        buf (make-buffer n :float16)
        seg (:segment buf)
        src (MemorySegment/ofArray shorts)]
    (MemorySegment/copy src 0 seg 0 (* n 2))
    buf))

(defn buffer-of-short-array
  "Create a :float16 DeviceBuffer from a pre-encoded short array.
  Each short is an IEEE 754 float16 encoded value."
  [^shorts arr]
  (let [n (alength arr)
        buf (make-buffer n :float16)
        seg (:segment buf)
        src (MemorySegment/ofArray arr)]
    (MemorySegment/copy src 0 seg 0 (* n 2))
    buf))

(defn buffer->short-array
  "Copy a :float16 DeviceBuffer's raw short values (IEEE 754 encoded)."
  [^DeviceBuffer buf]
  (let [n (:n-elements buf)
        out (short-array n)
        dst (MemorySegment/ofArray out)]
    (MemorySegment/copy (:segment buf) 0 dst 0 (* n 2))
    out))

(defn buffer->double-array
  "Read a DeviceBuffer's contents as a double array.
  Handles all dtypes including :float16 with conversion."
  [^DeviceBuffer buf]
  (let [seg (:segment buf)
        n (:n-elements buf)
        out (double-array n)]
    (case (:dtype buf)
      (:float16 :half)
      (dotimes [i n]
        (aset out i (double (Float/float16ToFloat
                             (.get seg ValueLayout/JAVA_SHORT (long (* i 2)))))))
      :float
      (dotimes [i n]
        (aset out i (double (.get seg ValueLayout/JAVA_FLOAT (long (* i 4))))))
      :double
      (let [dst (MemorySegment/ofArray out)]
        (MemorySegment/copy seg 0 dst 0 (:byte-size buf)))
      :int
      (dotimes [i n]
        (aset out i (double (.get seg ValueLayout/JAVA_INT (long (* i 4))))))
      :long
      (dotimes [i n]
        (aset out i (double (.get seg ValueLayout/JAVA_LONG (long (* i 8)))))))
    out))

(defn copy-doubles-to-fp16!
  "Copy a double array into an existing :float16 DeviceBuffer.
  Converts float64 → float16 in-place. Returns the buffer."
  [^DeviceBuffer buf ^doubles arr]
  (let [n (min (:n-elements buf) (alength arr))
        seg (:segment buf)]
    (dotimes [i n]
      (.set seg ValueLayout/JAVA_SHORT (long (* i 2))
            (short (Float/floatToFloat16 (float (aget arr i))))))
    buf))

(defn copy-fp16-to-doubles!
  "Copy a :float16 DeviceBuffer into an existing double array.
  Converts float16 → float64 in-place. Returns the double array."
  [^DeviceBuffer buf ^doubles arr]
  (let [n (min (:n-elements buf) (alength arr))
        seg (:segment buf)]
    (dotimes [i n]
      (aset arr i (double (Float/float16ToFloat
                           (.get seg ValueLayout/JAVA_SHORT (long (* i 2)))))))
    arr))

(defn buffer->float-array
  "Read a :float16 DeviceBuffer back as a float array."
  [^DeviceBuffer buf]
  (let [n (:n-elements buf)
        seg (:segment buf)
        out (float-array n)]
    (case (:dtype buf)
      (:float16 :half)
      (dotimes [i n]
        (aset out i (Float/float16ToFloat
                     (.get seg ValueLayout/JAVA_SHORT (long (* i 2))))))
      :float
      (let [dst (MemorySegment/ofArray out)]
        (MemorySegment/copy seg 0 dst 0 (:byte-size buf)))
      :double
      (dotimes [i n]
        (aset out i (float (.get seg ValueLayout/JAVA_DOUBLE (long (* i 8)))))))
    out))

(declare launch-2d!)

;; ================================================================
;; GpuSoA — GPU-resident SoA for defvalue types
;; ================================================================

(def ^:private element-tag->bytes-map
  {'double 8 'float 4 'long 8 'int 4})

(def ^:private element-tag->dtype-map
  {'double :double 'float :float 'long :long 'int :int})

(defrecord GpuSoA
           [scalar-tag    ;; Symbol, e.g. 'Particle
            soa-tag       ;; Symbol, e.g. 'ParticleSoA
            n             ;; number of scalar elements (long)
            field-segs    ;; ordered vec: [{:name "x" :dtype :float :seg MemorySegment} ...]
            ])

(defn gpu-soa?
  "Returns true if x is a GpuSoA."
  [x]
  (instance? GpuSoA x))

(defn gpu-array
  "Allocate GPU-resident (shared) storage for n elements of scalar-type.
   scalar-type: the defvalue type symbol (e.g. 'Particle) or Class.
   Returns GpuSoA. Works with any SoA-eligible defvalue type (all-primitive fields)."
  [scalar-type n]
  (let [scalar-tag (cond
                     (symbol? scalar-type) scalar-type
                     (class? scalar-type)  (symbol (.getSimpleName ^Class scalar-type))
                     :else (symbol (str scalar-type)))
        soa-reg @types/soa-registry
        soa-info (get soa-reg scalar-tag)]
    (when-not soa-info
      (throw (ex-info (str "No SoA registered for type: " scalar-tag
                           ". Use defvalue with all-primitive fields.")
                      {:scalar-type scalar-type :registered (keys soa-reg)})))
    (let [fields  (:fields soa-info)
          soa-tag (:soa-type-tag soa-info)
          field-segs (mapv (fn [{:keys [name element-tag]}]
                             (let [dtype      (get element-tag->dtype-map element-tag :float)
                                   elem-bytes (long (get element-tag->bytes-map element-tag 4))
                                   seg        (alloc-shared (* (long n) elem-bytes))]
                               {:name name :dtype dtype :seg seg}))
                           fields)]
      (->GpuSoA scalar-tag soa-tag (long n) field-segs))))

(defn gpu-array-device
  "Allocate device-only GPU storage for n elements of scalar-type.
   Faster than gpu-array for GPU-only data; requires explicit copy via
   ze/copy! for transfers. Returns GpuSoA."
  [scalar-type n]
  (let [scalar-tag (cond
                     (symbol? scalar-type) scalar-type
                     (class? scalar-type)  (symbol (.getSimpleName ^Class scalar-type))
                     :else (symbol (str scalar-type)))
        soa-reg @types/soa-registry
        soa-info (get soa-reg scalar-tag)]
    (when-not soa-info
      (throw (ex-info (str "No SoA registered for type: " scalar-tag) {})))
    (let [fields  (:fields soa-info)
          soa-tag (:soa-type-tag soa-info)
          field-segs (mapv (fn [{:keys [name element-tag]}]
                             (let [dtype      (get element-tag->dtype-map element-tag :float)
                                   elem-bytes (long (get element-tag->bytes-map element-tag 4))
                                   seg        (alloc-device (* (long n) elem-bytes))]
                               {:name name :dtype dtype :seg seg}))
                           fields)]
      (->GpuSoA scalar-tag soa-tag (long n) field-segs))))

(defn n-elements
  "Return the number of elements in a DeviceBuffer or GpuSoA."
  [buf]
  (cond
    (instance? DeviceBuffer buf) (:n-elements ^DeviceBuffer buf)
    (instance? GpuSoA buf)       (:n ^GpuSoA buf)
    :else (throw (ex-info "Not a DeviceBuffer or GpuSoA" {:type (type buf)}))))

(defn- get-soa-field-arr
  "Get array field from a JVM SoA object by field name using reflection."
  [soa-obj ^String field-name]
  (let [cls   (class soa-obj)
        ^java.lang.reflect.Field field
        (doto (.getDeclaredField cls field-name)
          (.setAccessible true))]
    (.get field soa-obj)))

(defn- dtype->elem-bytes ^long [dtype]
  (case dtype :double 8 :float 4 :long 8 :int 4))

(defn copy-to-gpu!
  "Copy a JVM SoA object (defvalue SoA type) into the GpuSoA's field segments.
   Uses zero-copy MemorySegment/copy for shared memory.
   Returns gpu-buf."
  [^GpuSoA gpu-buf soa-obj]
  (let [field-segs (:field-segs gpu-buf)
        n          (long (:n gpu-buf))]
    (doseq [{:keys [name dtype seg]} field-segs]
      (let [jvm-arr (get-soa-field-arr soa-obj name)
            n-bytes (* n (dtype->elem-bytes dtype))
            src-seg (MemorySegment/ofArray jvm-arr)]
        (MemorySegment/copy src-seg 0 seg 0 n-bytes)))
    gpu-buf))

(defn copy-from-gpu!
  "Copy GpuSoA field segments back into a JVM SoA object.
   Returns soa-obj."
  [^GpuSoA gpu-buf soa-obj]
  (let [field-segs (:field-segs gpu-buf)
        n          (long (:n gpu-buf))]
    (doseq [{:keys [name dtype seg]} field-segs]
      (let [jvm-arr (get-soa-field-arr soa-obj name)
            n-bytes (* n (dtype->elem-bytes dtype))
            dst-seg (MemorySegment/ofArray jvm-arr)]
        (MemorySegment/copy seg 0 dst-seg 0 n-bytes)))
    soa-obj))

;; ================================================================
;; GPU GEMM (non-square XMX)
;; ================================================================

(def ^:private gemm-cache
  "Cache for compiled GEMM kernel module. Atom holding {:module :kernel}."
  (atom nil))

(defn- ensure-gemm-kernel!
  "Lazily compile + cache the XMX gemm_nonsquare kernel. Returns {:module :kernel}."
  []
  (ensure-init!)
  (when (nil? @gemm-cache)
    (let [cl-src (do (require 'raster.compiler.backend.gpu.opencl-codegen)
                     ((resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-gemm-nonsquare-kernel)
                      "gemm_nonsquare"))
          device-hex (:device-id-hex @state)
          spv (do (require 'raster.compiler.support.spirv-cache)
                  ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                   cl-src :device device-hex))
          module (load-module! spv)
          kernel (create-kernel module "gemm_nonsquare")]
      (clojure.core/reset! gemm-cache {:module module :kernel kernel})))
  @gemm-cache)

(defn gemm!
  "GPU matrix multiply: C = A × B using XMX DPAS instructions.
  A: FP16 DeviceBuffer [M×K], B: FP16 DeviceBuffer [K×N],
  C: FP16 DeviceBuffer [M×N] (output, will be overwritten).
  All matrices are row-major.

  Returns C."
  [a b c m n k]
  (let [{:keys [kernel]} (ensure-gemm-kernel!)
        gc-m (int (Math/ceil (/ (double m) 128.0)))
        gc-n (int (Math/ceil (/ (double n) 128.0)))
        args [(:segment a) (:segment b) (:segment c)
              {:type :int :value (int m)}
              {:type :int :value (int n)}
              {:type :int :value (int k)}]]
    ;; 256 work-items per workgroup (16 subgroups × 16 lanes)
    (launch-2d! kernel [256 1] [gc-n gc-m] args)
    c))

;; ================================================================
;; GPU weight buffer manager (persistent FP16 across training)
;; ================================================================

(defn make-weight-buffers
  "Create persistent FP16 GPU buffers for a set of weight arrays.
  weights: map of keyword→double-array (e.g., {:W1 W1-arr, :b1 b1-arr}).
  Returns map of keyword→DeviceBuffer, all :float16.

  Usage at model init:
    (def gpu-weights (make-weight-buffers {:W1 W1 :b1 b1 :W2 W2 :b2 b2}))
  Then pass (:W1 gpu-weights) to GPU kernels."
  [weights]
  (into {} (map (fn [[k ^doubles arr]]
                  [k (buffer-of-doubles-as-half arr)])
                weights)))

(defn sync-weights-to-gpu!
  "Copy updated CPU weight arrays to existing GPU FP16 buffers.
  weights: map of keyword→double-array
  gpu-bufs: map of keyword→DeviceBuffer (from make-weight-buffers)
  Returns gpu-bufs."
  [weights gpu-bufs]
  (doseq [[k ^doubles arr] weights]
    (when-let [buf (get gpu-bufs k)]
      (copy-doubles-to-fp16! buf arr)))
  gpu-bufs)

(defn sync-gradients-from-gpu!
  "Copy GPU gradient FP16 buffers back to CPU double arrays.
  grad-bufs: map of keyword→DeviceBuffer (FP16 gradient buffers)
  grad-arrays: map of keyword→double-array (pre-allocated CPU arrays)
  Returns grad-arrays."
  [grad-bufs grad-arrays]
  (doseq [[k ^doubles arr] grad-arrays]
    (when-let [buf (get grad-bufs k)]
      (copy-fp16-to-doubles! buf arr)))
  grad-arrays)

(defn free-weight-buffers!
  "Free all GPU buffers in a weight buffer map."
  [gpu-bufs]
  (doseq [[_ buf] gpu-bufs]
    (free-buffer! buf)))

;; ================================================================
;; High-level convenience
;; ================================================================

(defn invoke-kernel
  "High-level kernel invocation for Raster pipeline.
  Handles array arguments: copies JVM arrays to shared memory,
  launches kernel, copies results back.

  Supports both JVM arrays and DeviceBuffers as inputs/output.
  DeviceBuffers are passed directly (no copy), JVM arrays are
  copied to temp shared memory (allocated+freed per call).

  input-arrays: seq of JVM arrays or DeviceBuffers
  output-array: JVM array or DeviceBuffer to receive results
  scalar-args: seq of {:type :int/:long/:float/:double, :value N}
  n: number of elements
  workgroup-size: threads per workgroup
  dtype-size: bytes per element (8 for double, 4 for float)"
  ([kernel-name module input-arrays output-array scalar-args n workgroup-size]
   (invoke-kernel kernel-name module input-arrays output-array scalar-args n workgroup-size 8))
  ([^String kernel-name ^MemorySegment module
    input-arrays output-array scalar-args
    n workgroup-size dtype-size]
   (let [n (long n)
         workgroup-size (long workgroup-size)
         dtype-size (long dtype-size)
         kernel (create-kernel module kernel-name)
         n-bytes (* n (long dtype-size))
        ;; Resolve inputs: DeviceBuffers pass through, arrays get temp alloc
         temp-segs (atom [])
         dev-inputs (mapv (fn [arr]
                            (if (device-buffer? arr)
                              (:segment ^DeviceBuffer arr)
                              (let [arr-bytes (if (instance? (Class/forName "[D") arr)
                                                (* (alength ^doubles arr) 8)
                                                (* (alength ^floats arr) 4))
                                    seg (alloc-shared arr-bytes)
                                    src (MemorySegment/ofArray arr)]
                                (MemorySegment/copy src 0 seg 0 arr-bytes)
                                (swap! temp-segs conj seg)
                                seg)))
                          input-arrays)
        ;; Resolve output
         output-is-buffer? (device-buffer? output-array)
         dev-output (if output-is-buffer?
                      (:segment ^DeviceBuffer output-array)
                      (let [seg (alloc-shared n-bytes)]
                        (swap! temp-segs conj seg)
                        seg))
        ;; Build arg list: input ptrs, output ptr, scalars, n
         all-args (vec (concat dev-inputs
                               [dev-output]
                               scalar-args
                               [{:type :int :value n}]))
         group-count (long (Math/ceil (/ (double n) workgroup-size)))]

     (launch! kernel group-count workgroup-size all-args)

    ;; Copy results back only for JVM array output
     (when-not output-is-buffer?
       (let [dst-seg (MemorySegment/ofArray output-array)]
         (MemorySegment/copy dev-output 0 dst-seg 0 n-bytes)))

    ;; Free only temporary allocations
     (doseq [seg @temp-segs] (free! seg))

     output-array)))

;; ================================================================
;; 2D kernel launch (for matmul, stencil, etc.)
;; ================================================================

(defn launch-2d!
  "Launch a kernel with 2D group dimensions.
  workgroup-size: [x y] threads per workgroup
  group-count: [x y] number of workgroups
  kernel-args: same format as launch!"
  [^MemorySegment kernel workgroup-size group-count kernel-args]
  (ensure-init!)
  (let [cmd (:cmd-list @state)
        arena (:arena @state)
        [wg-x wg-y] workgroup-size
        [gc-x gc-y] group-count]
    ;; Set group size
    (ze-call! "zeKernelSetGroupSize" @h-zeKernelSetGroupSize
              [kernel (int wg-x) (int wg-y) (int 1)])

    ;; Set arguments (same as launch!)
    (doseq [[idx arg] (map-indexed vector kernel-args)]
      (if (instance? MemorySegment arg)
        (let [arg-ptr (ptr-seg arena)]
          (.set arg-ptr PTR 0 ^MemorySegment arg)
          (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                    [kernel (int idx) (long (.byteSize PTR)) arg-ptr]))
        (let [{:keys [type value]} arg]
          (case type
            :int    (let [s (.allocate ^Arena arena I32)]
                      (.set s I32 0 (int value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :float  (let [s (.allocate ^Arena arena ValueLayout/JAVA_FLOAT)]
                      (.set s ValueLayout/JAVA_FLOAT 0 (float value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :double (let [s (.allocate ^Arena arena F64)]
                      (.set s F64 0 (double value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))
            :long   (let [s (.allocate ^Arena arena I64)]
                      (.set s I64 0 (long value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))))))

    ;; 2D group count
    (let [gc (.allocate arena 12)]
      (.set gc I32 0 (int gc-x))
      (.set gc I32 4 (int gc-y))
      (.set gc I32 8 (int 1))
      (ze-call! "zeCommandListAppendLaunchKernel" @h-zeCommandListAppendLaunchKernel
                [cmd kernel gc MemorySegment/NULL (int 0) MemorySegment/NULL]))))

;; ================================================================
;; Fast repeated launch (pre-bound kernel)
;; ================================================================

(defn bind-kernel!
  "Pre-bind kernel arguments for fast repeated launches.
  Returns a map that can be passed to launch-bound!.

  kernel-args: same format as launch! args
  Binds all arguments once; use launch-bound! for zero-overhead dispatch."
  [^MemorySegment kernel ^long workgroup-size-x kernel-args]
  (ensure-init!)
  (let [arena (:arena @state)]
    ;; Set group size once
    (ze-call! "zeKernelSetGroupSize" @h-zeKernelSetGroupSize
              [kernel (int workgroup-size-x) (int 1) (int 1)])
    ;; Set all arguments once
    (doseq [[idx arg] (map-indexed vector kernel-args)]
      (if (instance? MemorySegment arg)
        (let [arg-ptr (ptr-seg arena)]
          (.set arg-ptr PTR 0 ^MemorySegment arg)
          (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                    [kernel (int idx) (long (.byteSize PTR)) arg-ptr]))
        (let [{:keys [type value]} arg]
          (case type
            :int    (let [s (.allocate ^Arena arena I32)]
                      (.set s I32 0 (int value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :float  (let [s (.allocate ^Arena arena ValueLayout/JAVA_FLOAT)]
                      (.set s ValueLayout/JAVA_FLOAT 0 (float value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :double (let [s (.allocate ^Arena arena F64)]
                      (.set s F64 0 (double value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))
            :long   (let [s (.allocate ^Arena arena I64)]
                      (.set s I64 0 (long value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))))))
    (let [gc-seg (.allocate arena 12)
          cmd (:cmd-list @state)]
      ;; Pre-fill Y=1, Z=1 (only X changes per launch)
      (.set gc-seg I32 4 (int 1))
      (.set gc-seg I32 8 (int 1))
      ;; Pre-allocate launch args array (reused every call)
      (let [launch-args (object-array [cmd kernel gc-seg
                                       MemorySegment/NULL (int 0) MemorySegment/NULL])]
        {:kernel kernel :cmd cmd :gc-seg gc-seg
         :launch-args launch-args
         :h-launch @h-zeCommandListAppendLaunchKernel}))))

(defn launch-bound!
  "Launch a pre-bound kernel. Only dispatches (no arg setup, no barrier).
  Uses synchronous command list — completes before returning.
  Much faster than launch! for repeated calls on the same buffers.

  bound: map from bind-kernel!
  group-count-x: number of workgroups"
  [bound ^long group-count-x]
  (let [^MemorySegment gc (:gc-seg bound)]
    (.set gc I32 0 (int group-count-x))
    ;; Synchronous cmd list: launch completes before return, no barrier needed
    ;; Reuse pre-allocated args array
    (.invokeWithArguments ^MethodHandle (:h-launch bound)
                          ^"[Ljava.lang.Object;" (:launch-args bound))))

(defn bind-kernel-2d!
  "Pre-bind kernel arguments for fast repeated 2D launches.
  Returns a map that can be passed to launch-bound-2d!.

  workgroup-size: [x y] threads per workgroup
  kernel-args: same format as launch! args"
  [^MemorySegment kernel workgroup-size kernel-args]
  (ensure-init!)
  (let [arena (:arena @state)
        [wg-x wg-y] workgroup-size]
    ;; Set group size once
    (ze-call! "zeKernelSetGroupSize" @h-zeKernelSetGroupSize
              [kernel (int wg-x) (int wg-y) (int 1)])
    ;; Set all arguments once
    (doseq [[idx arg] (map-indexed vector kernel-args)]
      (if (instance? MemorySegment arg)
        (let [arg-ptr (ptr-seg arena)]
          (.set arg-ptr PTR 0 ^MemorySegment arg)
          (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                    [kernel (int idx) (long (.byteSize PTR)) arg-ptr]))
        (let [{:keys [type value]} arg]
          (case type
            :int    (let [s (.allocate ^Arena arena I32)]
                      (.set s I32 0 (int value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :float  (let [s (.allocate ^Arena arena ValueLayout/JAVA_FLOAT)]
                      (.set s ValueLayout/JAVA_FLOAT 0 (float value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 4) s]))
            :double (let [s (.allocate ^Arena arena F64)]
                      (.set s F64 0 (double value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))
            :long   (let [s (.allocate ^Arena arena I64)]
                      (.set s I64 0 (long value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 8) s]))))))
    (let [gc-seg (.allocate arena 12)
          cmd (:cmd-list @state)]
      (.set gc-seg I32 8 (int 1))  ;; Z=1
      (let [launch-args (object-array [cmd kernel gc-seg
                                       MemorySegment/NULL (int 0) MemorySegment/NULL])]
        {:kernel kernel :cmd cmd :gc-seg gc-seg
         :launch-args launch-args
         :h-launch @h-zeCommandListAppendLaunchKernel}))))

(defn launch-bound-2d!
  "Launch a pre-bound 2D kernel. Only sets group counts and dispatches.
  bound: map from bind-kernel-2d!
  group-count-x, group-count-y: number of workgroups in each dimension"
  [bound ^long group-count-x ^long group-count-y]
  (let [^MemorySegment gc (:gc-seg bound)]
    (.set gc I32 0 (int group-count-x))
    (.set gc I32 4 (int group-count-y))
    (.invokeWithArguments ^MethodHandle (:h-launch bound)
                          ^"[Ljava.lang.Object;" (:launch-args bound))))

;; ================================================================
;; Command graph: enqueue-all → execute-once → sync-once (OpenVINO's model)
;; The per-op-barrier immediate list pays the ~35-75µs launch floor per kernel;
;; a recorded regular command list pays the host-append cost ONCE and replays the
;; whole sequence with a single queue execute. This is the raster analog of
;; OpenVINO's in-order-enqueue + single-finish (5ms MiniLM vs 36ms per-op).
;; ================================================================

(defn create-kernel-fresh
  "Create a NEW, UNCACHED kernel handle. LZ kernel arguments are mutable state on
  the kernel handle, snapshotted at append; a recorded graph with N launches of
  the same compiled kernel must give each launch its own handle or the last args
  win (the 127-matmul clobber). Pair with destroy-kernel! at teardown."
  ^MemorySegment [^MemorySegment module ^String kernel-name]
  (ensure-init!)
  (let [arena (:arena @state)
        kern-desc (.allocate arena 32)
        _ (.set kern-desc I32 0 (int ZE_STRUCTURE_TYPE_KERNEL_DESC))
        name-seg (.allocateFrom arena kernel-name)
        _ (.set kern-desc PTR 24 name-seg)
        kern-out (ptr-seg arena)
        _ (ze-call! "zeKernelCreate" @h-zeKernelCreate [module kern-desc kern-out])]
    (read-ptr kern-out)))

(defn record-graph!
  "Record an ordered seq of bound kernels into a regular (replayable) command list.
  Each `bound` (from bind-kernel!/bind-kernel-2d!, each carrying its own dedicated
  kernel handle with args + group counts already set into its :gc-seg) is appended
  once, with NO per-op barrier — ordering is implicit in the list. Returns a graph
  {:queue :list :lists-arr} for replay-graph!. Re-record only if the kernel
  sequence or any buffer pointer changes; buffer CONTENTS may change between replays."
  [bounds & {:keys [barrier?] :or {barrier? true}}]
  (ensure-init!)
  (let [{:keys [arena context device]} @state
        cq-desc (.allocate ^Arena arena 40)
        _ (.set cq-desc I32 0 (int ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC))
        _ (.set cq-desc I32 28 (int 1)) ;; SYNCHRONOUS: execute blocks until list completes
        q-out (ptr-seg arena)
        _ (ze-call! "zeCommandQueueCreate" @h-zeCommandQueueCreate [context device cq-desc q-out])
        queue (read-ptr q-out)
        cl-desc (.allocate ^Arena arena 24)
        _ (.set cl-desc I32 0 (int ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC))
        l-out (ptr-seg arena)
        _ (ze-call! "zeCommandListCreate" @h-zeCommandListCreate [context device cl-desc l-out])
        lst (read-ptr l-out)
        h-launch @h-zeCommandListAppendLaunchKernel]
    (doseq [{:keys [kernel gc-seg]} bounds]
      (ze-call! "zeCommandListAppendLaunchKernel" h-launch
                [lst kernel gc-seg MemorySegment/NULL (int 0) MemorySegment/NULL])
      ;; DEVICE-SIDE barrier between kernels enforces RAW/WAW ordering on the GPU
      ;; (a GEMM writes H, the next kernel reads H) with no host round-trip. The
      ;; whole graph still costs ONE host sync (the queue execute). Skip only when
      ;; all kernels are independent (:barrier? false — e.g. a batch of GEMMs to
      ;; distinct outputs).
      (when barrier?
        (ze-call! "zeCommandListAppendBarrier" @h-zeCommandListAppendBarrier
                  [lst MemorySegment/NULL (int 0) MemorySegment/NULL])))
    (ze-call! "zeCommandListClose" @h-zeCommandListClose [lst])
    (let [lists-arr (ptr-seg arena)]
      (.set lists-arr PTR 0 ^MemorySegment lst)
      {:queue queue :list lst :lists-arr lists-arr})))

(defn replay-graph!
  "Execute a recorded command graph once. The SYNCHRONOUS queue blocks until the
  whole recorded sequence completes — one host round-trip for the entire graph."
  [graph]
  (ze-call! "zeCommandQueueExecuteCommandLists" @h-zeCommandQueueExecuteCommandLists
            [(:queue graph) (int 1) (:lists-arr graph) MemorySegment/NULL]))

(defn destroy-graph!
  "Destroy a recorded graph's command list + queue (pairs zeCommandListCreate /
  zeCommandQueueCreate; avoids the driver-object-leak SIGABRT)."
  [graph]
  (when-let [l (:list graph)] (ze-call! "zeCommandListDestroy" @h-zeCommandListDestroy [l]))
  (when-let [q (:queue graph)] (ze-call! "zeCommandQueueDestroy" @h-zeCommandQueueDestroy [q])))

(defn destroy-kernel!
  "Destroy a kernel handle from create-kernel-fresh."
  [^MemorySegment kernel]
  (ze-call! "zeKernelDestroy" @h-zeKernelDestroy [kernel]))

;; ================================================================
;; Kernel registry (pipeline integration)
;; ================================================================

(def kernel-registry
  "Global registry mapping kernel-name → kernel info.
  Populated by pipeline before eval, consumed by invoke-kernel/invoke-reduction-kernel."
  (atom {}))

;; ----------------------------------------------------------------
;; Arena-scoped kernel lifetime management
;; ----------------------------------------------------------------

(def ^:dynamic *current-arena*
  "When set, newly registered kernels are tagged with this arena-id.
  Set by with-gpu-computation."
  nil)

(defn make-kernel-arena!
  "Create a new kernel arena. Returns a unique arena-id (keyword).
  Kernels registered while *current-arena* is bound to this id
  will be freed together when close-kernel-arena! is called."
  []
  (keyword (str "arena-" (gensym ""))))

(defn close-kernel-arena!
  "Free all GPU MemorySegments and remove all kernels registered
  under arena-id from kernel-registry."
  [arena-id]
  (let [reg @kernel-registry
        arena-kernels (filter (fn [[_ info]] (= (:arena-id info) arena-id)) reg)]
    (doseq [[kname info] arena-kernels]
      (doseq [[k v] info]
        (when (instance? MemorySegment v)
          (try (free! v) (catch Exception _)))))
    (swap! kernel-registry #(reduce dissoc % (map first arena-kernels))))
  nil)

(defmacro with-gpu-computation
  "Execute body with a fresh kernel arena bound to arena-sym.
  All kernels registered during body execution (via compile-par-forms!
  or compile-abm-kernels!) are tagged with the arena-id.
  On exit (normal or exceptional), close-kernel-arena! frees their
  cached GPU MemorySegments and removes them from kernel-registry."
  [[arena-sym] & body]
  `(let [~arena-sym (make-kernel-arena!)]
     (binding [*current-arena* ~arena-sym]
       (try
         ~@body
         (finally
           (close-kernel-arena! ~arena-sym))))))

;; ----------------------------------------------------------------
;; GPU function arena management helpers
;; ----------------------------------------------------------------

(defn close!
  "Free GPU MemorySegments for a compiled GPU function.
  Reads :arena-id from the function's metadata and calls close-kernel-arena!.
  Works with any function (ftm, plain fn) that was compiled inside make-gpu-fn."
  [f]
  (when-let [arena-id (:arena-id (meta f))]
    (close-kernel-arena! arena-id))
  nil)

(defn make-gpu-fn
  "Compile f-thunk inside a fresh kernel arena. Returns the compiled function
  (an ftm TypedFn or plain fn, as produced by the pipeline) with :arena-id
  stored in its metadata. Kernels registered during f-thunk are tagged with
  the arena-id and freed by close! or close-kernel-arena!.

  For the TypedFn (hoist-and-eval-typed) path, Closeable is injected directly
  into the reify so with-open works without any wrapper. For the plain fn path,
  call (ze/close! f) when done."
  [compile-thunk]
  (let [arena-id (make-kernel-arena!)]
    (binding [*current-arena* arena-id]
      (let [compiled (compile-thunk)]
        ;; If the pipeline injected Closeable into the reify, it's already there.
        ;; Always store :arena-id in metadata for ze/close! and introspection.
        (with-meta compiled (assoc (meta compiled) :arena-id arena-id))))))

(defn register-kernel!
  "Register a kernel's compilation artifacts (source, spv-bytes, etc.).
  When *current-arena* is bound, tags the kernel with that arena-id
  so it can be freed en-masse by close-kernel-arena!."
  ([kernel-name kernel-info]
   (register-kernel! kernel-name kernel-info *current-arena*))
  ([kernel-name kernel-info arena-id]
   (let [info (cond-> kernel-info
                arena-id (assoc :arena-id arena-id))]
     (swap! kernel-registry assoc kernel-name info))))

(defn- ensure-kernel-loaded!
  "Lazily compile SPIR-V and load module for a registered kernel.
  Returns updated kernel-info with :module and :kernel-handle."
  [kernel-name]
  (ensure-init!)
  (let [info (get @kernel-registry kernel-name)]
    (when-not info
      (throw (ex-info (str "Kernel not registered: " kernel-name)
                      {:kernel-name kernel-name
                       :registered (keys @kernel-registry)})))
    (if (:kernel-handle info)
      info
      (let [;; Compile SPIR-V if not already done
            device-hex (:device-id-hex @state)
            spv-bytes (or (:spv-bytes info)
                          (let [cache (delay
                                        ((requiring-resolve
                                          'raster.compiler.support.spirv-cache/make-cache)))
                                compile-fn (fn [src]
                                             ((requiring-resolve
                                               'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                                              src :device device-hex))
                                get-or-compile (requiring-resolve
                                                'raster.compiler.support.spirv-cache/get-or-compile)]
                            (get-or-compile @cache (:source info) compile-fn device-hex)))
            module (load-module! spv-bytes)
            kernel-handle (create-kernel module kernel-name)
            updated (assoc info
                           :spv-bytes spv-bytes
                           :module module
                           :kernel-handle kernel-handle)]
        (swap! kernel-registry assoc kernel-name updated)
        updated))))

(defn- ensure-seg
  "Return a cached MemorySegment for [kernel-name k], allocating via
  alloc-shared if absent or smaller than n-bytes."
  ^MemorySegment [^String kernel-name k ^long n-bytes]
  (let [info (get @kernel-registry kernel-name)
        cached (get info k)]
    (if (and cached (>= (.byteSize ^MemorySegment cached) n-bytes))
      cached
      (let [seg (alloc-shared n-bytes)]
        (when cached (free! cached))
        (swap! kernel-registry assoc-in [kernel-name k] seg)
        seg))))

(defn- ensure-arr
  "Return a cached short-array for [kernel-name k], allocating if absent
  or smaller than n-elems."
  ^shorts [^String kernel-name k ^long n-elems]
  (let [info (get @kernel-registry kernel-name)
        ^shorts cached (get info k)]
    (if (and cached (>= (alength cached) n-elems))
      cached
      (let [arr (short-array n-elems)]
        (swap! kernel-registry assoc-in [kernel-name k] arr)
        arr))))

(defn invoke-registered-kernel
  "Pipeline-friendly kernel invocation. Looks up kernel from registry.
  input-arrays: vector of JVM arrays (double[] or float[]) or DeviceBuffers
  output-array: JVM array or DeviceBuffer to receive results
  scalar-args: vector of scalar values
  n: element count
  Dtype is read from kernel registry entry (:dtype, default :double)."
  [^String kernel-name input-arrays output-array scalar-args n]
  (let [{:keys [kernel-handle workgroup-size dtype]
         :or {workgroup-size 256 dtype :double}
         :as info} (ensure-kernel-loaded! kernel-name)
        n (long n)
        dtype-size (long (get dtype-byte-sizes dtype 8))
        scalar-type (if (= dtype :float) :float :double)
        n-bytes (* n dtype-size)
        ;; Copy input arrays to cached shared memory
        dev-inputs (mapv (fn [arr idx]
                           (if (device-buffer? arr)
                             (:segment ^DeviceBuffer arr)
                             (let [arr-bytes (if (= dtype :float)
                                               (* (alength ^floats arr) 4)
                                               (* (alength ^doubles arr) 8))
                                   seg (ensure-seg kernel-name (keyword (str "input-seg-" idx)) arr-bytes)
                                   src (MemorySegment/ofArray arr)]
                               (MemorySegment/copy src 0 seg 0 arr-bytes)
                               seg)))
                         input-arrays (range))
        ;; Output buffer
        output-is-buffer? (device-buffer? output-array)
        dev-output (if output-is-buffer?
                     (:segment ^DeviceBuffer output-array)
                     (ensure-seg kernel-name :output-seg n-bytes))
        ;; Build args: inputs, output, scalars, n
        scalar-kernel-args (mapv (fn [v] {:type scalar-type
                                          :value (if (= scalar-type :float)
                                                   (float v) (double v))})
                                 scalar-args)
        all-args (vec (concat dev-inputs
                              [dev-output]
                              scalar-kernel-args
                              [{:type :int :value (int n)}]))
        wg (long (or workgroup-size 256))
        group-count (long (Math/ceil (/ (double n) wg)))]
    (launch! kernel-handle group-count wg all-args)
    ;; Copy results back
    (when-not output-is-buffer?
      (let [dst-seg (MemorySegment/ofArray output-array)]
        (MemorySegment/copy dev-output 0 dst-seg 0 n-bytes)))
    output-array))

(defn invoke-reduction-kernel
  "Pipeline-friendly reduction kernel invocation. Returns scalar double.
  input-arrays: vector of JVM arrays (double[] or float[]) or DeviceBuffers
  n: element count
  Dtype is read from kernel registry entry (:dtype, default :double)."
  [^String kernel-name input-arrays n]
  (let [{:keys [kernel-handle workgroup-size identity-val dtype]
         :or {workgroup-size 256 identity-val 0.0 dtype :double}
         :as info} (ensure-kernel-loaded! kernel-name)
        n (long n)
        wg (long (or workgroup-size 256))
        group-count (long (Math/ceil (/ (double n) wg)))
        dtype-size (long (get dtype-byte-sizes dtype 8))
        float-dtype? (= dtype :float)
        value-layout (if float-dtype? ValueLayout/JAVA_FLOAT ValueLayout/JAVA_DOUBLE)
        ;; Copy input arrays to cached shared memory
        dev-inputs (mapv (fn [arr idx]
                           (if (device-buffer? arr)
                             (:segment ^DeviceBuffer arr)
                             (let [arr-bytes (if float-dtype?
                                               (* (alength ^floats arr) 4)
                                               (* (alength ^doubles arr) 8))
                                   seg (ensure-seg kernel-name (keyword (str "red-input-seg-" idx)) arr-bytes)
                                   src (MemorySegment/ofArray arr)]
                               (MemorySegment/copy src 0 seg 0 arr-bytes)
                               seg)))
                         input-arrays (range))
        ;; Partial sums output (one per workgroup)
        partial-bytes (* group-count dtype-size)
        dev-partial (ensure-seg kernel-name :partial-seg partial-bytes)
        ;; Build args: inputs, output, n
        all-args (vec (concat dev-inputs
                              [dev-partial]
                              [{:type :int :value (int n)}]))]
    (launch! kernel-handle group-count wg all-args)
    ;; Read partial sums and reduce on CPU
    (if (= group-count 1)
      (double (.get dev-partial value-layout 0))
      (loop [i 0 acc (double identity-val)]
        (if (< i group-count)
          (recur (inc i)
                 (+ acc (double (.get dev-partial value-layout
                                      (* i dtype-size)))))
          acc)))))

;; ================================================================
;; Void-map kernel invocation (side-effect-only kernels)
;; ================================================================

(defn kernel-registry-entry
  "Registry info for a kernel-name (source, :array-params, :written-arrays, dtype…)."
  [kernel-name]
  (get @kernel-registry kernel-name))

(defn bind-registered-map-void-kernel
  "Bind a registered void-map kernel over GPU-RESIDENT args, for recording into a
  command graph (whole-offload). `arrays` must be resident (DeviceBuffer /
  MemorySegment) — NO per-call JVM-array staging (residency is the whole point).
  `scalar-args` are scalar values; `n` is the element count. Returns a bound map
  {:kernel :gc-seg …} (from bind-kernel!, with :gc-seg X pre-set to the group
  count) — pass directly to record-graph!. A FRESH kernel handle per binding is
  used (LZ kernel args are mutable handle state → shared handles clobber)."
  [kernel-name arrays scalar-args n]
  (let [{:keys [module workgroup-size dtype]
         :or {workgroup-size 256 dtype :float}} (ensure-kernel-loaded! kernel-name)
        kernel-handle (create-kernel-fresh module kernel-name)
        wg (long workgroup-size)
        n (long n)
        dev-segs (reduce
                  (fn [acc arr]
                    (cond
                      (device-buffer? arr)          (conj acc (:segment ^DeviceBuffer arr))
                      (instance? MemorySegment arr)  (conj acc arr)
                      :else (throw (ex-info "bind-registered-map-void-kernel requires GPU-resident args (DeviceBuffer/MemorySegment); JVM-array staging is not supported on the bound path"
                                            {:arr-type (type arr)}))))
                  [] arrays)
        scalar-type (if (= dtype :float) :float :double)
        scalar-kernel-args (mapv (fn [v]
                                   (if (map? v) v
                                       {:type scalar-type
                                        :value (if (= scalar-type :float) (float v) (double v))}))
                                 scalar-args)
        all-args (vec (concat dev-segs scalar-kernel-args [{:type :int :value (int n)}]))
        group-count (long (Math/ceil (/ (double n) (double wg))))
        bnd (bind-kernel! kernel-handle wg all-args)]
    (.set ^MemorySegment (:gc-seg bnd) I32 0 (int group-count))
    bnd))

(defn bind-registered-gemm!
  "Bind the XMX GEMM kernel (C = A×B) over RESIDENT fp16 DeviceBuffers for recording into a
  command graph — the resident analog of invoke-registered-gemm! (which stages JVM arrays every
  call). A:[m×k] B:[k×n] C:[m×n], all fp16 (:half) resident buffers, row-major. Returns a bound
  {:kernel :gc-seg …} map (128×128 XMX tiles → gc = ceil(n/128) × ceil(m/128)). A fresh kernel
  handle per binding (LZ kernel args are mutable handle state → shared handles clobber)."
  [a b c m n k]
  (let [{:keys [module]} (ensure-gemm-kernel!)
        kh (create-kernel-fresh module "gemm_nonsquare")
        m (long m) n (long n) k (long k)
        args [(:segment a) (:segment b) (:segment c)
              {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]
        bnd (bind-kernel-2d! kh [256 1] args)
        gc ^MemorySegment (:gc-seg bnd)]
    (.set gc I32 0 (int (Math/ceil (/ (double n) 128.0))))   ;; X = gc-n
    (.set gc I32 4 (int (Math/ceil (/ (double m) 128.0))))   ;; Y = gc-m
    (.set gc I32 8 (int 1))
    bnd))

(defn invoke-registered-map-void-kernel
  "Pipeline-friendly void-map kernel invocation. No dedicated output array.
  All arrays are passed as read-write — copies to device before launch,
  copies written arrays back after launch.

  arrays: vector of JVM arrays (float[], int[], double[], long[])
  scalar-args: vector of scalar values
  n: element count
  Array types and write info read from kernel registry entry.

  Options:
    :workgroup-size  Override registry workgroup-size for this call only.
                     Used by the autotuner for wg-sweep benchmarking."
  ([^String kernel-name arrays scalar-args n]
   (invoke-registered-map-void-kernel kernel-name arrays scalar-args n {}))
  ([^String kernel-name arrays scalar-args n opts]
   (let [{:keys [kernel-handle workgroup-size dtype written-arrays array-types]
          :or {workgroup-size 256 dtype :float}
          :as info} (ensure-kernel-loaded! kernel-name)
         workgroup-size (long (get opts :workgroup-size workgroup-size))
         n (long n)
         default-dtype-size (long (get dtype-byte-sizes dtype 4))
        ;; Determine per-array byte size from actual array type
         arr-byte-size (fn [arr]
                         (cond
                           (instance? (Class/forName "[F") arr)
                           (* (alength ^floats arr) 4)
                           (instance? (Class/forName "[I") arr)
                           (* (alength ^ints arr) 4)
                           (instance? (Class/forName "[D") arr)
                           (* (alength ^doubles arr) 8)
                           (instance? (Class/forName "[J") arr)
                           (* (alength ^longs arr) 8)
                           :else
                           (* n default-dtype-size)))
        ;; Build expanded entries: GpuSoA expands to N field segments,
        ;; DeviceBuffer passes through, JVM arrays get copied to shared memory.
        ;; Each entry: {:seg MemorySegment :source JVM-arr-or-nil :ab byte-count-or-nil}
         expanded-entries
         (reduce
          (fn [acc [idx arr]]
            (cond
              ;; GpuSoA: expand to one segment per field (already shared memory)
              (gpu-soa? arr)
              (into acc (mapv (fn [{:keys [seg]}]
                                {:seg seg :source nil :ab nil})
                              (:field-segs ^GpuSoA arr)))
              ;; DeviceBuffer: pass segment through (no copy)
              (device-buffer? arr)
              (conj acc {:seg (:segment ^DeviceBuffer arr) :source nil :ab nil})
              ;; JVM array: copy to cached shared segment
              :else
              (let [ab  (long (arr-byte-size arr))
                    seg (ensure-seg kernel-name (keyword (str "void-arr-" idx)) ab)
                    src (MemorySegment/ofArray arr)]
                (MemorySegment/copy src 0 seg 0 ab)
                (conj acc {:seg seg :source arr :ab ab}))))
          []
          (map-indexed vector arrays))
         dev-segs (mapv :seg expanded-entries)
        ;; Scalar args
         scalar-type (if (= dtype :float) :float :double)
         scalar-kernel-args (mapv (fn [v]
                                    (if (map? v)
                                      v  ;; pre-typed scalar: {:type :int :value 4}
                                      {:type scalar-type
                                       :value (if (= scalar-type :float)
                                                (float v) (double v))}))
                                  scalar-args)
         all-args (vec (concat dev-segs
                               scalar-kernel-args
                               [{:type :int :value (int n)}]))
         wg (long (or workgroup-size 256))
         group-count (long (Math/ceil (/ (double n) wg)))]
     (launch! kernel-handle group-count wg all-args)
    ;; Copy back only JVM arrays (GpuSoA/DeviceBuffer have :source nil)
     (doseq [{:keys [seg source ab]} expanded-entries]
       (when source
         (MemorySegment/copy seg 0 (MemorySegment/ofArray source) 0 (long ab))))
     nil)))

;; ================================================================
;; Exclusive scan kernel invocation (Blelloch algorithm)
;; ================================================================

(defn- invoke-full-gpu-scan!
  "Fully GPU-resident Blelloch exclusive scan. No CPU round-trip.
  Recursively applies the block-scan and propagation kernels at each level.
  All intermediate buffers are temporary shared-memory allocations freed on return.

  Returns the total sum (as a long for int scan, double for double scan).

  block-kernel-handle: compiled kernel handle for block-level scan
  prop-kernel-handle:  compiled kernel handle for block-offset propagation
  input-segs:         vector of MemorySegment for input arrays
  output-seg:         MemorySegment for output (n elements, [0..n-1])
  n:                  number of input elements
  block-size:         work-items processed per workgroup (compile-time param)
  wg:                 workgroup size (= block-size/2 for Blelloch)
  elem-size:          bytes per element (4 for int, 8 for double)
  is-int?:            true → int arithmetic, false → double
  depth:              recursion depth (0 = outermost); limits to 2 levels"
  [block-kernel-handle prop-kernel-handle
   input-segs output-seg n block-size wg elem-size is-int? depth]
  (let [n (long n)
        block-size (long block-size)
        wg (long wg)
        elem-size (long elem-size)
        num-blocks (long (Math/ceil (/ (double n) (double block-size))))
        value-layout (if is-int? ValueLayout/JAVA_INT ValueLayout/JAVA_DOUBLE)
        ;; Phase 1: block-level scan → output-seg + block-sums
        block-sums-bytes (* num-blocks elem-size)
        block-sums-seg (alloc-shared block-sums-bytes)
        block-args (vec (concat input-segs [output-seg block-sums-seg
                                            {:type :int :value (int n)}]))]
    (try
      (launch! block-kernel-handle num-blocks wg block-args)
      (if (<= num-blocks 1)
        ;; Single block: total is in block-sums-seg[0]; no propagation needed
        (if is-int?
          (long (.get block-sums-seg ValueLayout/JAVA_INT 0))
          (double (.get block-sums-seg ValueLayout/JAVA_DOUBLE 0)))
        ;; Multi-block: recurse to scan block-sums → block-offsets
        (let [block-offsets-bytes (* num-blocks elem-size)
              block-offsets-seg (alloc-shared block-offsets-bytes)]
          (try
            (let [block-total
                  (if (> depth 1)
                    ;; Safety fallback for very large n (>256M): CPU combine
                    (let [block-sums-host (if is-int? (int-array num-blocks) (double-array num-blocks))
                          bs-src (MemorySegment/ofArray block-sums-host)]
                      (MemorySegment/copy block-sums-seg 0 bs-src 0 block-sums-bytes)
                      (if is-int?
                        (loop [i 0 acc (int 0)]
                          (if (< i num-blocks)
                            (do (.setAtIndex block-offsets-seg ValueLayout/JAVA_INT i acc)
                                (recur (inc i) (unchecked-add-int acc (clojure.core/aget ^ints block-sums-host i))))
                            (long acc)))
                        (loop [i 0 acc (double 0.0)]
                          (if (< i num-blocks)
                            (do (.setAtIndex block-offsets-seg ValueLayout/JAVA_DOUBLE i acc)
                                (recur (inc i) (+ acc (clojure.core/aget ^doubles block-sums-host i))))
                            acc))))
                    ;; Recursive GPU scan of block-sums
                    (invoke-full-gpu-scan!
                     block-kernel-handle prop-kernel-handle
                     [block-sums-seg] block-offsets-seg num-blocks block-size wg elem-size is-int? (inc depth)))]
              ;; Phase 3: propagate block offsets to output
              (launch! prop-kernel-handle num-blocks wg
                       [output-seg block-offsets-seg {:type :int :value (int n)}])
              block-total)
            (finally
              (free! block-offsets-seg)))))
      (finally
        (free! block-sums-seg)))))

(defn invoke-registered-scan-exclusive-kernel
  "Pipeline-friendly exclusive scan kernel invocation.
  Uses Blelloch algorithm: block-level scan → CPU block-sum scan → propagate.

  block-kernel-name: registered block-scan kernel
  prop-kernel-name:  registered propagation kernel
  input-arrays:      vector of JVM arrays or DeviceBuffers read by element expression
  output-array:      JVM array with n+1 elements (exclusive scan output), or DeviceBuffer.
                     DeviceBuffer path writes directly to GPU memory — no round-trip.
                     Only block-sums (~num_blocks*4 bytes) cross CPU/GPU boundary.
  n:                 number of input elements (output has n+1)"
  [^String block-kernel-name ^String prop-kernel-name input-arrays output-array n]
  (let [{:keys [kernel-handle workgroup-size block-size scan-dtype
                identity-num combine-op]
         :or {workgroup-size 256}
         :as block-info} (ensure-kernel-loaded! block-kernel-name)
        n (long n)
        wg (long (or workgroup-size 256))
        block-size (long (or block-size (* 2 wg)))
        num-blocks (long (Math/ceil (/ (double n) (double block-size))))
        ;; Type handling — prefer scan-dtype from kernel metadata
        out-is-device-buf? (device-buffer? output-array)
        is-int? (or (= scan-dtype :int)
                    (and (not out-is-device-buf?)
                         (instance? (Class/forName "[I") output-array)))
        elem-size (long (if is-int? 4 8))
        value-layout (if is-int? ValueLayout/JAVA_INT ValueLayout/JAVA_DOUBLE)
        ;; Determine per-array byte size from actual array type
        arr-byte-size (fn [arr]
                        (cond
                          (instance? (Class/forName "[I") arr)
                          (* (alength ^ints arr) 4)
                          (instance? (Class/forName "[F") arr)
                          (* (alength ^floats arr) 4)
                          (instance? (Class/forName "[D") arr)
                          (* (alength ^doubles arr) 8)
                          (instance? (Class/forName "[J") arr)
                          (* (alength ^longs arr) 8)
                          :else (* n elem-size)))
        ;; Copy input arrays to device (DeviceBuffers used directly)
        dev-inputs (mapv (fn [arr idx]
                           (if (device-buffer? arr)
                             (:segment ^DeviceBuffer arr)
                             (let [ab (long (arr-byte-size arr))
                                   seg (ensure-seg block-kernel-name
                                                   (keyword (str "scan-input-" idx)) ab)
                                   src (MemorySegment/ofArray arr)]
                               (MemorySegment/copy src 0 seg 0 ab)
                               seg)))
                         input-arrays (range))
        ;; Device output: use DeviceBuffer's segment directly (eliminates round-trip),
        ;; or allocate a temp shared segment (kernel writes to [0..n-1])
        output-bytes (* n elem-size)
        dev-output (if out-is-device-buf?
                     (:segment ^DeviceBuffer output-array)
                     (ensure-seg block-kernel-name :scan-output output-bytes))
        prop-info (ensure-kernel-loaded! prop-kernel-name)]
    ;; Full GPU-resident Blelloch scan — no CPU round-trip for block sums.
    ;; invoke-full-gpu-scan! recursively scans block-sums on GPU, returns total.
    (let [total (invoke-full-gpu-scan!
                 kernel-handle (:kernel-handle prop-info)
                 dev-inputs dev-output n block-size wg elem-size is-int? 0)]
      ;; Write output[n] = total; copy [0..n-1] only for JVM array output
      (if out-is-device-buf?
        ;; DeviceBuffer: result already in GPU segment; write total to element[n]
        (.setAtIndex dev-output value-layout n
                     (if is-int? (int total) (double total)))
        ;; JVM array: copy back and set element[n]
        (let [out-seg (MemorySegment/ofArray output-array)]
          (MemorySegment/copy dev-output 0 out-seg 0 output-bytes)
          (if is-int?
            (clojure.core/aset ^ints output-array (int n) (int total))
            (clojure.core/aset ^doubles output-array (int n) (double total))))))
    output-array))

;; ================================================================
;; RNG fill kernel invocation (splitmix64 parallel seed generation)
;; ================================================================

(defn invoke-registered-rng-fill-kernel
  "Invoke a compiled rng-fill kernel to fill a seeds buffer on-device.
  Generates n splitmix64 seeds from base-seed + i*golden_ratio in parallel.

  kernel-name: registered rng-fill kernel
  seeds-buf:   DeviceBuffer (long, n elements) or JVM long-array
  n:           number of seeds to generate
  base-seed:   scalar long seed from which per-element seeds are derived"
  [^String kernel-name seeds-buf n ^long base-seed]
  (let [{:keys [kernel-handle workgroup-size]} (ensure-kernel-loaded! kernel-name)
        n (long n)
        wg (long (or workgroup-size 256))
        groups (long (Math/ceil (/ (double n) (double wg))))
        seeds-seg (if (device-buffer? seeds-buf)
                    (:segment ^DeviceBuffer seeds-buf)
                    (let [ab (* n 8)
                          seg (ensure-seg kernel-name :seeds-buf ab)
                          src (MemorySegment/ofArray seeds-buf)]
                      (MemorySegment/copy src 0 seg 0 ab)
                      seg))]
    (launch! kernel-handle groups wg
             [seeds-seg {:type :int :value (int n)} {:type :long :value base-seed}])
    seeds-buf))

;; ================================================================
;; Active-ids kernel invocation (splitmix64 random index generation)
;; ================================================================

(defn invoke-registered-active-ids-kernel
  "Invoke a compiled active-ids kernel to fill an ids buffer on-device.
  Generates n-active random int indices in [0, n-agents) from base-seed + i*golden_ratio.

  kernel-name: registered active-ids kernel
  ids-buf:     DeviceBuffer (int, n-active elements) or JVM int-array
  n-active:    number of active agents to sample
  n-agents:    total agent count (modulus for index generation)
  base-seed:   scalar long seed from which per-element indices are derived"
  [^String kernel-name ids-buf n-active n-agents base-seed]
  (let [{:keys [kernel-handle workgroup-size]} (ensure-kernel-loaded! kernel-name)
        n-active (long n-active)
        n-agents (long n-agents)
        base-seed (long base-seed)
        wg (long (or workgroup-size 256))
        groups (long (Math/ceil (/ (double n-active) (double wg))))
        ids-seg (if (device-buffer? ids-buf)
                  (:segment ^DeviceBuffer ids-buf)
                  (let [ab (* n-active 4)
                        seg (ensure-seg kernel-name :ids-buf ab)
                        src (MemorySegment/ofArray ids-buf)]
                    (MemorySegment/copy src 0 seg 0 ab)
                    seg))]
    (launch! kernel-handle groups wg
             [ids-seg {:type :int :value (int n-active)}
              {:type :long :value n-agents}
              {:type :long :value base-seed}])
    ids-buf))

;; ================================================================
;; Compound kernel invocation (PDE solver fusion)
;; ================================================================

(defn invoke-compound-kernel
  "Launch a compound kernel that fuses multiple phases into a single
  __local-memory kernel. Used for PDE solvers with small n.

  kernel-name: registered compound kernel name
  arrays:      vector of JVM arrays [inputs... outputs...]
  scalar-args: vector of scalar values
  nsteps:      number of time steps (outer loop inside kernel)
  n:           element count (= workgroup size for :local strategy)

  Copies arrays to/from device. The kernel runs the full time-stepping
  loop on-device with __local scratch arrays."
  [^String kernel-name arrays scalar-args nsteps n]
  (let [{:keys [kernel-handle dtype]
         :or {dtype :double}
         :as info} (ensure-kernel-loaded! kernel-name)
        n (long n)
        nsteps (long nsteps)
        dtype-size (long (get dtype-byte-sizes dtype 8))
        scalar-type (if (= dtype :float) :float :double)
        n-bytes (* n dtype-size)
        ;; Copy all arrays to device memory
        dev-arrays (mapv (fn [arr idx]
                           (if (device-buffer? arr)
                             (:segment ^DeviceBuffer arr)
                             (let [arr-bytes (if (= dtype :float)
                                               (* (alength ^floats arr) 4)
                                               (* (alength ^doubles arr) 8))
                                   seg (ensure-seg kernel-name (keyword (str "compound-arr-" idx)) arr-bytes)
                                   src (MemorySegment/ofArray arr)]
                               (MemorySegment/copy src 0 seg 0 arr-bytes)
                               seg)))
                         arrays (range))
        ;; Build args: arrays, scalars, nsteps, n
        scalar-kernel-args (mapv (fn [v] {:type scalar-type
                                          :value (if (= scalar-type :float)
                                                   (float v) (double v))})
                                 scalar-args)
        all-args (vec (concat dev-arrays
                              scalar-kernel-args
                              [{:type :int :value (int nsteps)}
                               {:type :int :value (int n)}]))]
    ;; Launch: 1 workgroup of size n (all work in __local memory)
    (launch! kernel-handle 1 n all-args)
    ;; Copy output arrays back (outputs are the later arrays in the vector)
    ;; For now, copy ALL arrays back since we don't know which are outputs
    (doseq [[arr dev-seg idx] (map vector arrays dev-arrays (range))]
      (when-not (device-buffer? arr)
        (let [dst-seg (MemorySegment/ofArray arr)
              arr-bytes (if (= dtype :float)
                          (* (alength ^floats arr) 4)
                          (* (alength ^doubles arr) 8))]
          (MemorySegment/copy dev-seg 0 dst-seg 0 arr-bytes))))
    ;; Return first output array (convention: outputs follow inputs)
    (first arrays)))

;; ================================================================
;; Zero-copy DeviceBuffer kernel invocation (GPU plan pass)
;; ================================================================

(defn invoke-gpu-kernel
  "Launch kernel on DeviceBuffers. Zero CPU↔GPU copies.
  All data stays on-device between kernel launches.

  kernel-name:  registered kernel name
  input-bufs:   vector of DeviceBuffer inputs
  output-buf:   DeviceBuffer for output
  scalar-args:  vector of {:type kw :value v} scalar args
  n:            element count

  Backend-agnostic: works with Level Zero today, CUDA/ROCm later."
  [^String kernel-name input-bufs output-buf scalar-args n]
  (let [{:keys [kernel-handle workgroup-size dtype]
         :or {workgroup-size 256 dtype :float}
         :as info} (ensure-kernel-loaded! kernel-name)
        n (long n)
        ;; Extract MemorySegments from DeviceBuffers
        dev-inputs (mapv (fn [^DeviceBuffer buf] (:segment buf)) input-bufs)
        dev-output (:segment ^DeviceBuffer output-buf)
        all-args (vec (concat dev-inputs
                              [dev-output]
                              scalar-args
                              [{:type :int :value (int n)}]))
        wg (long (or workgroup-size 256))
        group-count (long (Math/ceil (/ (double n) wg)))]
    (launch! kernel-handle group-count wg all-args)
    output-buf))

(defn invoke-gpu-reduction-kernel
  "Reduce DeviceBuffer to scalar. Only reads back partial sums.
  Returns scalar double (always promoted to double for Clojure interop).

  kernel-name:  registered kernel name
  input-bufs:   vector of DeviceBuffer inputs
  n:            element count

  Backend-agnostic: works with Level Zero today, CUDA/ROCm later."
  [^String kernel-name input-bufs n]
  (let [{:keys [kernel-handle workgroup-size identity-val dtype]
         :or {workgroup-size 256 identity-val 0.0 dtype :float}
         :as info} (ensure-kernel-loaded! kernel-name)
        n (long n)
        wg (long (or workgroup-size 256))
        group-count (long (Math/ceil (/ (double n) wg)))
        dtype-size (long (get dtype-byte-sizes dtype 4))
        float-dtype? (contains? #{:float :float16 :half} dtype)
        value-layout (if float-dtype? ValueLayout/JAVA_FLOAT ValueLayout/JAVA_DOUBLE)
        ;; Extract MemorySegments from DeviceBuffers
        dev-inputs (mapv (fn [^DeviceBuffer buf] (:segment buf)) input-bufs)
        ;; Partial sums output (one per workgroup) — cached shared memory for readback
        partial-bytes (* group-count dtype-size)
        dev-partial (ensure-seg kernel-name :gpu-partial-seg partial-bytes)
        all-args (vec (concat dev-inputs
                              [dev-partial]
                              [{:type :int :value (int n)}]))]
    (launch! kernel-handle group-count wg all-args)
    ;; Read partial sums and reduce on CPU (only data leaving GPU)
    (if (= group-count 1)
      (double (.get dev-partial value-layout 0))
      (loop [i 0 acc (double identity-val)]
        (if (< i group-count)
          (recur (inc i)
                 (+ acc (double (.get dev-partial value-layout
                                      (* i dtype-size)))))
          acc)))))

(defn invoke-gpu-axpy!
  "In-place axpy on DeviceBuffers: y += alpha * x.
  Uses registered axpy kernel. Zero CPU↔GPU copies."
  [^String kernel-name ^DeviceBuffer y-buf ^DeviceBuffer x-buf alpha n]
  (let [{:keys [kernel-handle workgroup-size dtype]
         :or {workgroup-size 256 dtype :float}} (ensure-kernel-loaded! kernel-name)
        n (long n)
        scalar-type (if (= dtype :float) :float :double)
        all-args [(:segment y-buf)
                  (:segment x-buf)
                  {:type scalar-type :value (if (= scalar-type :float)
                                              (float alpha) (double alpha))}
                  {:type :int :value (int n)}]
        wg (long (or workgroup-size 256))
        group-count (long (Math/ceil (/ (double n) wg)))]
    (launch! kernel-handle group-count wg all-args)
    y-buf))

(defn invoke-registered-transpose!
  "Pipeline-friendly transpose: out = in^T (JVM float/double arrays).
  Copies to shared memory, launches kernel, copies back."
  [^String kernel-name in-arr out-arr rows cols]
  (let [{:keys [kernel-handle workgroup-size dtype]
         :or {workgroup-size 256 dtype :float}} (ensure-kernel-loaded! kernel-name)
        total (long (* (long rows) (long cols)))
        dtype-size (long (get dtype-byte-sizes dtype 4))
        n-bytes (* total dtype-size)
        ;; Copy input to cached shared memory
        in-seg (ensure-seg kernel-name :transpose-in-seg n-bytes)
        _ (MemorySegment/copy (MemorySegment/ofArray in-arr) 0 in-seg 0 n-bytes)
        ;; Cached output shared memory
        out-seg (ensure-seg kernel-name :transpose-out-seg n-bytes)
        all-args [in-seg out-seg
                  {:type :int :value (int rows)}
                  {:type :int :value (int cols)}]
        wg (long workgroup-size)
        gc (long (Math/ceil (/ (double total) wg)))]
    (launch! kernel-handle gc wg all-args)
    ;; Copy back
    (MemorySegment/copy out-seg 0 (MemorySegment/ofArray out-arr) 0 n-bytes)
    out-arr))

(defn invoke-registered-gemm!
  "Pipeline-friendly GEMM: C = A*B.
  Accepts float[] or double[] arrays for A, B, C.
  Internally converts to FP16 shared memory for XMX DPAS.
  C[M*N] output is FP32 shared memory, copied back to the output array.
  Uses 2D launch for XMX tiled GEMM."
  [^String kernel-name A B C m n k]
  (let [{:keys [kernel-handle workgroup-size]
         :or {workgroup-size 256}} (ensure-kernel-loaded! kernel-name)
        m (long m) n (long n) k (long k)
        a-float? (instance? (Class/forName "[F") A)
        b-float? (instance? (Class/forName "[F") B)
        c-float? (instance? (Class/forName "[F") C)
        ;; Convert A (M*K) to FP16 shared memory
        a-elems (* m k)
        a-shorts (ensure-arr kernel-name :gemm-a-shorts a-elems)
        _ (if a-float?
            (let [^floats af A]
              (dotimes [i a-elems]
                (aset a-shorts i (short (Float/floatToFloat16 (aget af i))))))
            (let [^doubles ad A]
              (dotimes [i a-elems]
                (aset a-shorts i (short (Float/floatToFloat16 (float (aget ad i))))))))
        a-seg (ensure-seg kernel-name :gemm-a-seg (* a-elems 2))
        _ (MemorySegment/copy (MemorySegment/ofArray a-shorts) 0 a-seg 0 (* a-elems 2))
        ;; Convert B (K*N) to FP16 shared memory
        b-elems (* k n)
        b-shorts (ensure-arr kernel-name :gemm-b-shorts b-elems)
        _ (if b-float?
            (let [^floats bf B]
              (dotimes [i b-elems]
                (aset b-shorts i (short (Float/floatToFloat16 (aget bf i))))))
            (let [^doubles bd B]
              (dotimes [i b-elems]
                (aset b-shorts i (short (Float/floatToFloat16 (float (aget bd i))))))))
        b-seg (ensure-seg kernel-name :gemm-b-seg (* b-elems 2))
        _ (MemorySegment/copy (MemorySegment/ofArray b-shorts) 0 b-seg 0 (* b-elems 2))
        ;; Cached C (M*N) FP32 shared memory
        c-elems (* m n)
        c-seg (ensure-seg kernel-name :gemm-c-seg (* c-elems 4))
        ;; 2D launch config
        gc-m (int (Math/ceil (/ (double m) 128.0)))
        gc-n (int (Math/ceil (/ (double n) 128.0)))
        args [a-seg b-seg c-seg
              {:type :int :value (int m)}
              {:type :int :value (int n)}
              {:type :int :value (int k)}]]
    (launch-2d! kernel-handle [256 1] [gc-n gc-m] args)
    ;; Copy FP32 output back to C array
    (if c-float?
      (do (MemorySegment/copy c-seg 0 (MemorySegment/ofArray ^floats C) 0 (* c-elems 4))
          C)
      ;; For double[] output: read FP32 from GPU, convert to double
      (let [tmp-floats (float-array c-elems)
            ^doubles cd C]
        (MemorySegment/copy c-seg 0 (MemorySegment/ofArray tmp-floats) 0 (* c-elems 4))
        (dotimes [i c-elems]
          (aset cd i (double (aget tmp-floats i))))
        C))))

(defn invoke-gpu-transpose!
  "Transpose matrix on GPU via registered kernel. Zero CPU↔GPU copies.
  in-buf: DeviceBuffer [rows x cols] row-major
  out-buf: DeviceBuffer [cols x rows] row-major (transposed)"
  [^String kernel-name ^DeviceBuffer in-buf ^DeviceBuffer out-buf rows cols]
  (let [{:keys [kernel-handle workgroup-size]
         :or {workgroup-size 256}} (ensure-kernel-loaded! kernel-name)
        total (long (* (long rows) (long cols)))
        all-args [(:segment in-buf)
                  (:segment out-buf)
                  {:type :int :value (int rows)}
                  {:type :int :value (int cols)}]
        wg (long (or workgroup-size 256))
        group-count (long (Math/ceil (/ (double total) wg)))]
    (launch! kernel-handle group-count wg all-args)
    out-buf))

;; ================================================================
;; Scatter kernel invocation
;; ================================================================

(defn invoke-registered-scatter-kernel
  "Pipeline-friendly scatter-add kernel invocation.
  output[index[i]] += src[i] (atomically).

  kernel-name:  registered scatter kernel
  output:       JVM array or DeviceBuffer (accumulation target)
  src:          JVM array or DeviceBuffer (source values)
  index:        JVM int-array or DeviceBuffer (destination indices)
  n:            number of source elements
  stride:       optional stride for strided scatter (nil for unstrided)"
  [^String kernel-name output src index n & [stride]]
  (let [{:keys [kernel-handle workgroup-size dtype]
         :or {workgroup-size 256 dtype :float}} (ensure-kernel-loaded! kernel-name)
        n (long n)
        dtype-size (long (get dtype-byte-sizes dtype 4))
        wg (long (or workgroup-size 256))
        groups (long (Math/ceil (/ (double n) wg)))
        ;; Copy arrays to device
        copy-to (fn [arr tag byte-count]
                  (if (device-buffer? arr)
                    (:segment ^DeviceBuffer arr)
                    (let [seg (ensure-seg kernel-name tag byte-count)
                          s (MemorySegment/ofArray arr)]
                      (MemorySegment/copy s 0 seg 0 byte-count)
                      seg)))
        out-elems (if (device-buffer? output) n
                      (if stride (* n (long stride)) (alength output)))
        out-seg (copy-to output :scatter-out (* out-elems dtype-size))
        src-seg (copy-to src :scatter-src (* n (if stride (* (long stride) dtype-size) dtype-size)))
        idx-seg (copy-to index :scatter-idx (* n 4))
        args (if stride
               [out-seg src-seg idx-seg {:type :int :value (int n)} {:type :int :value (int stride)}]
               [out-seg src-seg idx-seg {:type :int :value (int n)}])]
    (launch! kernel-handle groups wg args)
    ;; Copy back output
    (when-not (device-buffer? output)
      (MemorySegment/copy out-seg 0 (MemorySegment/ofArray output) 0
                          (* (long (alength output)) dtype-size)))
    output))

;; ================================================================
;; Reduce-by-key kernel invocation
;; ================================================================

(defn invoke-registered-reduce-by-key-kernel
  "Pipeline-friendly reduce-by-key kernel invocation.
  output[keys[i]] += vals[i] (atomically).

  kernel-name:  registered reduce-by-key kernel
  output:       JVM array or DeviceBuffer (accumulation target)
  keys:         JVM int-array or DeviceBuffer (key indices)
  vals:         JVM array or DeviceBuffer (values to accumulate)
  n:            number of elements"
  [^String kernel-name output keys vals n]
  (let [{:keys [kernel-handle workgroup-size dtype]
         :or {workgroup-size 256 dtype :float}} (ensure-kernel-loaded! kernel-name)
        n (long n)
        dtype-size (long (get dtype-byte-sizes dtype 4))
        wg (long (or workgroup-size 256))
        groups (long (Math/ceil (/ (double n) wg)))
        copy-to (fn [arr tag byte-count]
                  (if (device-buffer? arr)
                    (:segment ^DeviceBuffer arr)
                    (let [seg (ensure-seg kernel-name tag byte-count)
                          s (MemorySegment/ofArray arr)]
                      (MemorySegment/copy s 0 seg 0 byte-count)
                      seg)))
        out-seg (copy-to output :rbk-out (* (long (if (device-buffer? output) n (alength output))) dtype-size))
        keys-seg (copy-to keys :rbk-keys (* n 4))
        vals-seg (copy-to vals :rbk-vals (* n dtype-size))
        args [out-seg keys-seg vals-seg {:type :int :value (int n)}]]
    (launch! kernel-handle groups wg args)
    ;; Copy back output
    (when-not (device-buffer? output)
      (MemorySegment/copy out-seg 0 (MemorySegment/ofArray output) 0
                          (* (long (alength output)) dtype-size)))
    output))

;; ================================================================
;; Cleanup
;; ================================================================

(defn shutdown!
  "Release all Level Zero resources."
  []
  (when (:initialized? @state)
    ;; Free cached MemorySegments from kernel registry
    (doseq [[_ info] @kernel-registry]
      (doseq [[k v] info]
        (when (instance? MemorySegment v)
          (try (free! v) (catch Exception _)))))
    (clojure.core/reset! kernel-registry {})
    (doseq [[_ k] (:kernels @state)]
      (try (.invokeWithArguments ^MethodHandle @h-zeKernelDestroy
                                 ^java.util.List (java.util.List/of (object-array [k])))
           (catch Exception _)))
    (doseq [[_ m] (:modules @state)]
      (try (.invokeWithArguments ^MethodHandle @h-zeModuleDestroy
                                 ^java.util.List (java.util.List/of (object-array [m])))
           (catch Exception _)))
    (when-let [^Arena arena (:arena @state)]
      (.close arena))
    (clojure.core/reset! state {:initialized? false :driver nil :device nil
                                :context nil :cmd-list nil :arena nil
                                :modules {} :kernels {}}))
  nil)

(defn reset!
  "Full GPU reset: shutdown + reinitialize. Use in the REPL when the GPU
  state is corrupted or after switching simulation scales. Clears all
  kernel handles, modules, and device buffers from the registry."
  []
  (shutdown!)
  (init!)
  nil)

