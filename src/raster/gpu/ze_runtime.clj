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
  (:require [raster.compiler.core.types :as types]
            [raster.compiler.core.dtype :as dt]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.resident-value :as resident-value]))

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
(def ^:private ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC 0x0f)
(def ^:private ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC 0x15)
(def ^:private ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC 0x16)
(def ^:private ZE_STRUCTURE_TYPE_MODULE_DESC 0x1b)
(def ^:private ZE_STRUCTURE_TYPE_KERNEL_DESC 0x1d)
(def ^:private ZE_MODULE_FORMAT_IL_SPIRV 0x00)
(def ^:private ZE_MODULE_FORMAT_NATIVE 0x01)
(def ^:private ZE_RESULT_SUCCESS 0)
(def ^:private ZE_RESULT_NOT_READY 1)
(def ^:private ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS 1)
(def ^:private ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS 2)

;; Device-event kernel timing (profiling)
(def ^:private ZE_STRUCTURE_TYPE_EVENT_POOL_DESC 0x10)
(def ^:private ZE_STRUCTURE_TYPE_EVENT_DESC 0x11)
(def ^:private ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 0x00020006)
(def ^:private ZE_EVENT_POOL_FLAG_HOST_VISIBLE 0x1)
(def ^:private ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP 0x4)
(def ^:private ZE_EVENT_SCOPE_FLAG_HOST 0x4)

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

(def ^:private h-zeCommandListHostSynchronize
  (delay (make-handle "zeCommandListHostSynchronize" (fd I32 PTR I64))))

;; Regular (replayable) command list + queue — for recorded command graphs
(def ^:private h-zeCommandQueueCreate
  (delay (make-handle "zeCommandQueueCreate" (fd I32 PTR PTR PTR PTR))))

(def ^:private h-zeCommandListCreate
  (delay (make-handle "zeCommandListCreate" (fd I32 PTR PTR PTR PTR))))

(def ^:private h-zeCommandListClose
  (delay (make-handle "zeCommandListClose" (fd I32 PTR))))

(def ^:private h-zeCommandQueueExecuteCommandLists
  (delay (make-handle "zeCommandQueueExecuteCommandLists" (fd I32 PTR I32 PTR PTR))))

(def ^:private h-zeCommandQueueSynchronize
  (delay (make-handle "zeCommandQueueSynchronize" (fd I32 PTR I64))))

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

;; --- Device events (kernel timestamp profiling) ---
(def ^:private h-zeEventPoolCreate
  (delay (make-handle "zeEventPoolCreate" (fd I32 PTR PTR I32 PTR PTR))))
(def ^:private h-zeEventPoolDestroy
  (delay (make-handle "zeEventPoolDestroy" (fd I32 PTR))))
(def ^:private h-zeEventCreate
  (delay (make-handle "zeEventCreate" (fd I32 PTR PTR PTR))))
(def ^:private h-zeEventDestroy
  (delay (make-handle "zeEventDestroy" (fd I32 PTR))))
(def ^:private h-zeEventQueryKernelTimestamp
  (delay (make-handle "zeEventQueryKernelTimestamp" (fd I32 PTR PTR))))
(def ^:private h-zeEventHostReset
  (delay (make-handle "zeEventHostReset" (fd I32 PTR))))

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

(def ^:private h-zeCommandListDestroy
  (delay (make-handle "zeCommandListDestroy" (fd I32 PTR))))

(def ^:private h-zeCommandQueueDestroy
  (delay (make-handle "zeCommandQueueDestroy" (fd I32 PTR))))

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
                  _ (.set cq-desc I32 28 (int ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS))
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

(defn async-cmd-list
  "The shared ASYNCHRONOUS immediate command list, created lazily. Unlike the default
  synchronous list (every append blocks until the kernel completes ~100µs), appends to this
  list return immediately and execute in order; the host waits once via synchronize-async!.
  This amortizes the completion-wait across a batch of dispatches (the decode pattern: queue a
  token's ~200 dependent GEMVs, sync once)."
  ^MemorySegment []
  (ensure-init!)
  (or (:cmd-list-async @state)
      (let [{:keys [arena context device]} @state
            cq-desc (.allocate ^Arena arena 40)
            _ (.set cq-desc I32 0 (int ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC))
            _ (.set cq-desc I32 28 (int ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS))
            cmd-out (ptr-seg arena)
            _ (ze-call! "zeCommandListCreateImmediate" @h-zeCommandListCreateImmediate
                        [context device cq-desc cmd-out])
            lst (read-ptr cmd-out)]
        (clojure.core/swap! state assoc :cmd-list-async lst)
        lst)))

(defn synchronize-async!
  "Block until all commands appended to the async command list have completed."
  []
  (ze-call! "zeCommandListHostSynchronize" @h-zeCommandListHostSynchronize
            [(async-cmd-list) (long -1)]))  ;; UINT64_MAX = wait forever

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

(defn device-timer-props
  "Device timer properties for converting kernel-timestamp ticks → nanoseconds.
  Returns {:ns-per-tick double, :kernel-timestamp-valid-bits long, :timer-resolution long}.

  Queried with stype ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, where timerResolution
  is the timer FREQUENCY in cycles/sec (exact — the 1.0 stype reports integer
  nanoseconds-per-tick, which truncates e.g. 52.083 ns → 52). Fallback heuristic for
  old drivers that ignore the 1_2 stype: a value < 1e5 cannot be a frequency and is
  interpreted as ns-per-tick directly. Cached after first query."
  []
  (ensure-init!)
  (or (:timer-props @state)
      (let [{:keys [^Arena arena device]} @state
            props (.allocate arena 512)
            _ (.set props I32 0 (int ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2))
            _ (ze-call! "zeDeviceGetProperties(timer)" @h-zeDeviceGetProperties [device props])
            ;; ze_device_properties_t offsets: timerResolution u64 @80,
            ;; timestampValidBits u32 @88, kernelTimestampValidBits u32 @92 (see query-devices).
            timer-resolution (.get props I64 80)
            kts-bits (long (.get props I32 92))
            ns-per-tick (if (< timer-resolution 100000)
                          (double timer-resolution)              ;; already ns/tick (1.0 semantics)
                          (/ 1.0e9 (double timer-resolution)))   ;; cycles/sec → ns/tick
            tp {:ns-per-tick ns-per-tick
                :kernel-timestamp-valid-bits kts-bits
                :timer-resolution timer-resolution}]
        (swap! state assoc :timer-props tp)
        tp)))

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

(defn create-kernel-fresh
  "Create a NEW kernel handle from a module (never cached). Each handle has independent,
  mutable argument state — use one per pre-bound argument-set so concurrent bindings of the
  same kernel source don't clobber each other (create-kernel shares one cached handle)."
  ^MemorySegment [^MemorySegment module ^String kernel-name]
  (ensure-init!)
  (let [arena (:arena @state)
        kern-desc (.allocate ^Arena arena 32)
        _ (.set kern-desc I32 0 (int ZE_STRUCTURE_TYPE_KERNEL_DESC))
        name-seg (.allocateFrom ^Arena arena kernel-name)
        _ (.set kern-desc PTR 24 name-seg)
        kern-out (ptr-seg arena)
        _ (ze-call! "zeKernelCreate" @h-zeKernelCreate [module kern-desc kern-out])]
    (read-ptr kern-out)))

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
  {:double 8 :float 4 :float32 4 :int 4 :long 8 :half 2 :float16 2 :byte 1 :int8 1})

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

(defn slice-buffer
  "Create a non-owning typed pointer view into a DeviceBuffer. The returned record must never be
   freed independently: its MemorySegment is a slice of the base allocation and exists only for
   ABI binding. Bounds and dtype alignment are established by the backend-neutral BufferView."
  [^DeviceBuffer buf byte-offset byte-length dtype]
  (let [byte-offset (long byte-offset)
        byte-length (long byte-length)
        element-size (get dtype-byte-sizes dtype)]
    (when-not element-size
      (throw (ex-info "cannot slice a Level Zero buffer with an unknown dtype" {:dtype dtype})))
    (let [element-bytes (long element-size)]
      (when (or (neg? byte-offset) (neg? byte-length)
                (> (+ byte-offset byte-length) (:byte-size buf))
                (not (zero? (mod byte-length element-bytes))))
        (throw (ex-info "Level Zero buffer slice is out of bounds or misaligned"
                        {:byte-offset byte-offset :byte-length byte-length
                         :buffer-bytes (:byte-size buf) :dtype dtype})))
      (->DeviceBuffer (.asSlice ^MemorySegment (:segment buf) byte-offset byte-length)
                      (quot byte-length element-bytes) byte-length dtype))))

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

(defn- as-segment
  "A JVM primitive array or a MemorySegment, as a MemorySegment. The one conversion point, so a
   Boring mmap segment and a float[] take the same code path."
  ^MemorySegment [x]
  (if (instance? MemorySegment x) x (MemorySegment/ofArray x)))

(defn- check-range!
  "Element-range bounds for a ranged transfer. Throws rather than clamping: `array->buffer!`
   silently copies `(min buf src)` bytes, which is fine for a whole-buffer transfer but is exactly
   the wrong behaviour for a range — a clamped KV-cache export looks identical to a correct one
   until the restored continuation diverges at the missing tokens."
  ;; no primitive hints: >4 primitive params is a Clojure limit (CLAUDE.md); the body casts
  [what have-elements offset elements elem-size other-bytes]
  (let [have-elements (long have-elements) offset (long offset) elements (long elements)
        elem-size (long elem-size) other-bytes (long other-bytes)]
    (when (or (neg? offset) (neg? elements))
      (throw (ex-info (str what ": negative offset or length") {:offset offset :elements elements})))
    (when (> (+ offset elements) have-elements)
      (throw (ex-info (str what ": range exceeds the buffer")
                      {:offset offset :elements elements :buffer-elements have-elements})))
    (when (> (* elements elem-size) other-bytes)
      (throw (ex-info (str what ": range exceeds the host-side array/segment")
                      {:needed-bytes (* elements elem-size) :available-bytes other-bytes})))))

(defn plan-range
  "Validate one ranged transfer and return its byte-level PLAN, without copying anything:
   `{:buf-off :host-off :n-bytes :host-seg}`. Throws on any out-of-range.

   Split from the copy so a BATCH can validate every entry before executing any. Without that, a
   bad spec in the 30th layer of a 36-layer KV restore would leave the first 29 written and the
   cache half-restored — a partial state that is worse than either all or nothing, and that a
   single-range API could never produce."
  [^DeviceBuffer buf host {:keys [src-element dst-element elements]
                           :or {src-element 0 dst-element 0}} direction]
  (let [es (long (get dtype-byte-sizes (:dtype buf) 4))
        host-seg (as-segment host)
        ;; for :upload the host side is src and the buffer side is dst; :download is the mirror
        [buf-el host-el] (case direction :upload [dst-element src-element] :download [src-element dst-element])
        host-off (* (long host-el) es)]
    (check-range! (name direction) (:n-elements buf) buf-el elements es
                  (- (.byteSize host-seg) host-off))
    {:buf-off (* (long buf-el) es) :host-off host-off :n-bytes (* (long elements) es)
     :host-seg host-seg}))

(defn execute-range!
  "Perform a transfer previously validated by `plan-range`. Session buffers are SHARED
   (host-coherent) allocations, so this is a plain Panama copy — no command list, and a
   MemorySegment host side (e.g. an mmap'd file) is copied directly without materializing a JVM
   array."
  [^DeviceBuffer buf {:keys [buf-off host-off n-bytes host-seg]} direction]
  (case direction
    :upload   (MemorySegment/copy ^MemorySegment host-seg (long host-off) (:segment buf) (long buf-off) (long n-bytes))
    :download (MemorySegment/copy (:segment buf) (long buf-off) ^MemorySegment host-seg (long host-off) (long n-bytes))))

(defn submit-range-batch!
  "Execute a validated range batch against Level Zero shared allocations.

   Shared allocations are host coherent, so there is no device copy command to signal: Panama
   performs the copies before submission returns. The common GPUEvent contract still represents
   this legal inline completion and reports host-monotonic timing rather than claiming a device
  event measurement."
  [entries direction]
  (let [active (filterv (fn [[_ plan]] (pos? (long (:n-bytes plan)))) entries)
        started (System/nanoTime)]
    (doseq [[buffer plan] active]
      (execute-range! buffer plan direction))
    (let [elapsed (- (System/nanoTime) started)
          bytes (reduce + 0 (map (comp long :n-bytes second) entries))]
      {:complete? true
       :completion {:timing-source :host-monotonic
                    :elapsed-ns elapsed
                    :bytes bytes
                    :commands (count active)
                    :direction direction
                    :asynchronous? false}})))

(defn transfer-capabilities
  "Return the Level Zero runtime's physical transfer execution contract."
  []
  {:submission :inline-host-copy
   ;; Transfers are inline host copies into shared device allocations: complete on return, so
   ;; there is no in-flight staging copy, no asynchrony, no peer route, no event to await and no
   ;; physical queue of their own; the host argument may be any JVM array or segment.
   :host-staging :none
   :independent-physical-queue? false
   :queue-ordering :inline
   :async-h2d? false
   :async-d2h? false
   :peer-transfer? false
   :peer-mechanisms []
   :event-semantics :already-complete
   :host-lease-until-await? false
   :host-memory :any
   :physically-serialized? true})

(defn upload-range!
  "Copy `elements` elements from `src` (JVM array or MemorySegment, starting at element
   `src-element`) into `buf` starting at element `dst-element`. Elements, not bytes: the byte
   size comes from the buffer's own dtype, so a caller cannot mis-size a copy. Returns the buffer."
  [^DeviceBuffer buf src spec]
  (execute-range! buf (plan-range buf src spec :upload) :upload)
  buf)

(defn download-range!
  "Copy `elements` elements from `buf` (starting at element `src-element`) into `dst` (JVM array
   or MemorySegment, starting at element `dst-element`). Mirror of `upload-range!`. Returns `dst`."
  [^DeviceBuffer buf dst spec]
  (execute-range! buf (plan-range buf dst spec :download) :download)
  dst)

(defn copy-buffer-range!
  "Synchronously copy an element range between resident Level Zero buffers."
  [^DeviceBuffer src ^DeviceBuffer dst src-element dst-element elements]
  (when-not (= (:dtype src) (:dtype dst))
    (throw (ex-info "Level Zero resident copy requires matching dtypes"
                    {:source-dtype (:dtype src) :destination-dtype (:dtype dst)})))
  (let [element-bytes (long (get dtype-byte-sizes (:dtype src) 4))
        src-byte (* (long src-element) element-bytes)
        dst-byte (* (long dst-element) element-bytes)
        byte-count (* (long elements) element-bytes)]
    (synchronize-async!)
    (MemorySegment/copy (:segment src) src-byte
                        (:segment dst) dst-byte byte-count)
    dst))

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
      (:byte :int8)
      (let [out (byte-array n)
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
        (aset out i (double (.get seg ValueLayout/JAVA_LONG (long (* i 8))))))
      (:byte :int8)
      (dotimes [i n]
        (aset out i (double (.get seg ValueLayout/JAVA_BYTE (long i))))))
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

(defn- physical-pointer-dtypes
  "Storage dtypes after expanding each logical map/map-void pointer binding."
  [arrays]
  (reduce (fn [dtypes arr]
            (cond
              (gpu-soa? arr) (into dtypes (mapv :dtype (:field-segs ^GpuSoA arr)))
              (device-buffer? arr) (conj dtypes (:dtype ^DeviceBuffer arr))
              (instance? MemorySegment arr) (conj dtypes :opaque)
              :else (conj dtypes (dt/dtype-for-jvm-array arr))))
          [] arrays))

(defn expand-pointer-binding
  "Expand one logical artifact pointer binding into Level Zero's physical resident values.
   A ResidentComposite or legacy GpuSoA supplies one checked value per ABI field; ordinary resident
   buffers supply one slot. This function is driver-free and runs before KernelCall construction."
  [{:keys [binding slots] :as group} value]
  (cond
    (resident-value/resident-composite? value)
    (resident-value/expand group value)

    (gpu-soa? value)
    (let [fields (:field-segs ^GpuSoA value)]
      (when-not (= (count slots) (count fields))
        (throw (ex-info "GpuSoA field count differs from its artifact binding"
                        {:binding binding :expected (count slots) :actual (count fields)
                         :slots slots :fields (mapv :name fields)})))
      (doseq [[slot field] (map vector slots fields)]
        (let [expected-name (symbol (str (name binding) "_" (name (:name field))))]
          (when-not (= expected-name (:name slot))
            (throw (ex-info "GpuSoA field order/name differs from its physical ABI slot"
                            {:binding binding :slot slot :field (:name field)
                             :expected expected-name :actual (:name slot)}))))
        (when-not (= (:dtype slot) (dt/canon (:dtype field)))
          (throw (ex-info "GpuSoA field dtype differs from its physical ABI slot"
                          {:binding binding :slot slot :field (:name field)
                           :expected (:dtype slot) :actual (dt/canon (:dtype field))}))))
      (mapv :seg fields))

    (not= 1 (count slots))
    (throw (ex-info "multi-slot logical pointer requires a resident composite or GpuSoA value"
                    {:binding binding :slots slots :value-type (type value)}))

    (or (device-buffer? value) (instance? MemorySegment value))
    [value]

    :else
    (throw (ex-info "Level Zero logical pointer requires DeviceBuffer/GpuSoA/MemorySegment"
                    {:binding binding :slot (first slots) :value-type (type value)}))))

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
  (case dtype :double 8 :float 4 :long 8 :int 4 (:byte :int8) 1))

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

(defn- emit-scheduled-gemm
  "Ask the compiler for the shared scheduled matrix body and its direct OpenCL lowering."
  [kernel-name c-dtype tile epilogue]
  ((requiring-resolve 'raster.compiler.backend.gpu.gemm/emit-scheduled-matrix-kernel)
   {:kernel-name kernel-name
    :id [:resident-gemm kernel-name]
    :a 'A :b 'B :c 'C :m 'M :n 'N :k 'K
    :tile tile :result-dtype c-dtype :epilogue epilogue
    :provenance {:dialect :resident-runtime}}))

(defn- emit-scheduled-split-k-gemm
  "Ask the compiler for a grid-Z sliced K reduction over resident buffers."
  [kernel-name tile]
  ((requiring-resolve 'raster.compiler.backend.gpu.gemm/emit-scheduled-split-k-kernel)
   {:kernel-name kernel-name
    :id [:resident-gemm kernel-name]
    :a 'A :b 'B :c 'C :m 'M :n 'N :k 'K :kc 'KC :splits 'splits
    :tile tile
    :provenance {:dialect :resident-runtime}}))

(defn- emit-scheduled-batched-gemm
  "Ask the compiler for grid-Z-selected independent resident matrix views."
  [kernel-name tile]
  ((requiring-resolve 'raster.compiler.backend.gpu.gemm/emit-scheduled-batched-matrix-kernel)
   {:kernel-name kernel-name
    :id [:resident-gemm kernel-name]
    :a 'A :b 'B :c 'C :m 'M :n 'N :k 'K :batch 'batch
    :tile tile
    :provenance {:dialect :resident-runtime}}))

(declare gemm-tile)

(def ^:private gemm-cache
  "Cache for compiled GEMM kernels, keyed by C-output dtype (:half | :float).
   Each entry is {:module :kernel :kernel-name}. (A/B are always fp16 in.)"
  (atom {}))

(defn- ensure-gemm-kernel!
  "Lazily compile + cache the XMX gemm_nonsquare kernel for a given C-output dtype
   (:half or :float — A/B always fp16, fp32 accumulate). Returns {:module :kernel :kernel-name}."
  [c-dtype]
  (ensure-init!)
  (or (get @gemm-cache c-dtype)
      (let [kname (str "gemm_nonsquare_" (name c-dtype))
            tile (gemm-tile)
            emitted (emit-scheduled-gemm kname c-dtype tile nil)
            cl-src (:source emitted)
            device-hex (:device-id-hex @state)
            spv (do (require 'raster.compiler.support.spirv-cache)
                    ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                     cl-src :device device-hex))
            module (load-module! spv)
            kernel (create-kernel module kname)
            entry {:module module :kernel kernel :kernel-name kname
                   :tile tile :kernel-body (:kernel-body emitted)
                   :workgroup (:workgroup-size emitted)}]
        (swap! gemm-cache assoc c-dtype entry)
        entry)))

(defn- gemm-tile
  "The GEMM tile for this device, from the ONE source (compiler.core.hardware/gemm-tile-for).
   Launch geometry MUST be derived from the same tile the kernel was emitted with — the `/128.0`
   literals this replaces were a second, independent spelling of block-m/block-n, so any tile change
   would have silently mismatched kernel and grid."
  []
  (let [f (requiring-resolve 'raster.compiler.core.hardware/gemm-tile-for)
        d (try ((requiring-resolve 'raster.compiler.core.hardware/descriptor-for) (:device-id @state))
               (catch Throwable _ nil))]
    (f d)))

(defn gemm!
  "GPU matrix multiply: C = A × B using XMX DPAS instructions.
  A: FP16 DeviceBuffer [M×K], B: FP16 DeviceBuffer [K×N],
  C: FP16 DeviceBuffer [M×N] (output, will be overwritten).
  All matrices are row-major.

  Returns C."
  [a b c m n k]
  (let [{:keys [kernel tile workgroup]} (ensure-gemm-kernel! :half)
        {:keys [block-m block-n]} tile
        gc-m (int (Math/ceil (/ (double m) (double block-m))))
        gc-n (int (Math/ceil (/ (double n) (double block-n))))
        args [(:segment a) (:segment b) (:segment c)
              {:type :int :value (int m)}
              {:type :int :value (int n)}
              {:type :int :value (int k)}]]
    (launch-2d! kernel workgroup [gc-n gc-m] args)
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
                                [kernel (int idx) (long 8) s]))
            :half   (let [s (.allocate ^Arena arena ValueLayout/JAVA_SHORT)]
                      (.set s ValueLayout/JAVA_SHORT 0
                            (short (Float/floatToFloat16 (float value))))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 2) s]))
            :byte   (let [s (.allocate ^Arena arena ValueLayout/JAVA_BYTE)]
                      (.set s ValueLayout/JAVA_BYTE 0 (byte value))
                      (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                [kernel (int idx) (long 1) s]))))))

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
  "Pre-bind kernel arguments for fast repeated 1-3D launches.
  Returns a map that can be passed to launch-bound!.

  kernel-args: same format as launch! args
  Binds all arguments once; use launch-bound! for zero-overhead dispatch.
  Optional cmd-list selects which immediate command list launch-bound! appends to —
  default is the synchronous list (each dispatch blocks); pass (async-cmd-list) for batched
  dispatch (appends return immediately; sync once via synchronize-async!)."
  ([^MemorySegment kernel workgroup-size kernel-args]
   (bind-kernel! kernel workgroup-size kernel-args (:cmd-list @state)))
  ([^MemorySegment kernel workgroup-size kernel-args ^MemorySegment cmd-list]
   (ensure-init!)
   (let [arena (:arena @state)
         workgroup-size (if (vector? workgroup-size)
                          workgroup-size
                          [(long workgroup-size)])
         _ (when-not (and (<= 1 (count workgroup-size) 3)
                          (every? #(and (integer? %) (pos? %)) workgroup-size))
             (throw (ex-info "Level Zero binding requires a positive 1-3D workgroup vector"
                             {:workgroup-size workgroup-size})))
         [wg-x wg-y wg-z] (take 3 (concat workgroup-size [1 1]))]
    ;; Set group size once
     (ze-call! "zeKernelSetGroupSize" @h-zeKernelSetGroupSize
               [kernel (int wg-x) (int wg-y) (int wg-z)])
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
                                 [kernel (int idx) (long 8) s]))
             :half   (let [s (.allocate ^Arena arena ValueLayout/JAVA_SHORT)]
                       (.set s ValueLayout/JAVA_SHORT 0
                             (short (Float/floatToFloat16 (float value))))
                       (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                 [kernel (int idx) (long 2) s]))
             :byte   (let [s (.allocate ^Arena arena ValueLayout/JAVA_BYTE)]
                       (.set s ValueLayout/JAVA_BYTE 0 (byte value))
                       (ze-call! (str "zeKernelSetArgumentValue[" idx "]") @h-zeKernelSetArgumentValue
                                 [kernel (int idx) (long 1) s]))))))
     (let [gc-seg (.allocate arena 12)
           cmd cmd-list]
      ;; Pre-fill a legal 1-D launch; KernelCall binding overwrites every active dimension.
       (.set gc-seg I32 0 (int 1))
       (.set gc-seg I32 4 (int 1))
       (.set gc-seg I32 8 (int 1))
      ;; Pre-allocate launch args array (reused every call)
       (let [launch-args (object-array [cmd kernel gc-seg
                                        MemorySegment/NULL (int 0) MemorySegment/NULL])]
         {:kernel kernel :cmd cmd :gc-seg gc-seg
          :launch-args launch-args
          :h-launch @h-zeCommandListAppendLaunchKernel})))))

(defn launch-bound!
  "Launch a pre-bound 1-3D kernel. Only dispatches (no arg setup, no barrier).
  Uses synchronous command list — completes before returning.
  Much faster than launch! for repeated calls on the same buffers.

  bound: map from bind-kernel!
  `group-count` is a positive one-to-three-dimensional vector or a legacy 1-D scalar."
  [bound group-count]
  (let [^MemorySegment gc (:gc-seg bound)
        group-count (if (vector? group-count) group-count [(long group-count)])]
    (when-not (and (<= 1 (count group-count) 3)
                   (every? #(and (integer? %) (pos? %)) group-count))
      (throw (ex-info "Level Zero launch requires a positive 1-3D group-count vector"
                      {:group-count group-count})))
    (doseq [[axis count] (map-indexed vector (take 3 (concat group-count [1 1])))]
      (.set gc I32 (long (* axis 4)) (int count)))
    ;; Synchronous cmd list: launch completes before return, no barrier needed
    ;; Reuse pre-allocated args array
    (.invokeWithArguments ^MethodHandle (:h-launch bound)
                          ^"[Ljava.lang.Object;" (:launch-args bound))))

(defn launch-geometry!
  "Bind arguments and launch one checked 1-3D geometry. Artifact-backed staging uses this instead
   of the legacy dimension-specific launchers so no valid KernelCall axis can be discarded."
  [kernel workgroup-size group-count kernel-args]
  (let [bound (bind-kernel! kernel workgroup-size kernel-args)]
    (launch-bound! bound group-count)))

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
  Each `bound` (from the uniform bind-kernel!, carrying its own dedicated
  kernel handle with args + group counts already set into its :gc-seg) is appended
  once, with NO per-op barrier — ordering is implicit in the list. Returns a graph
  {:queue :list :lists-arr} for replay-graph!. Re-record only if the kernel
  sequence or any buffer pointer changes; buffer CONTENTS may change between replays."
  [bounds & {:keys [barrier?] :or {barrier? true}}]
  (ensure-init!)
  (let [{:keys [arena context device]} @state
        cq-desc (.allocate ^Arena arena 40)
        _ (.set cq-desc I32 0 (int ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC))
        _ (.set cq-desc I32 28 (int ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS))
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
  Populated by the pipeline before eval and consumed by the registered ABI binders."
  (atom {}))

(def kernel-dispatch-registry
  "Pure compiler dispatch values keyed independently of single-entry native kernels."
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
    (swap! kernel-registry #(reduce dissoc % (map first arena-kernels)))
    (swap! kernel-dispatch-registry
           (fn [dispatches]
             (into {} (remove (fn [[_ dispatch]] (= arena-id (:arena-id dispatch))))
                   dispatches))))
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
   (let [_ (when (kart/kernel-artifact? kernel-info) (kart/validate! kernel-info))
         info (cond-> kernel-info
                arena-id (assoc :arena-id arena-id))]
     (swap! kernel-registry assoc kernel-name info))))

(defn kernel-registry-entry
  "Public read of a registered kernel's info map (source, :array-params, :scalar-params, dtype,
   …). Used by the resident GPU-program binder to map kernel params → resident buffers."
  [kernel-name]
  (get @kernel-registry kernel-name))

(defn register-kernel-dispatch!
  ([dispatch] (register-kernel-dispatch! dispatch *current-arena*))
  ([dispatch arena-id]
   (let [dispatch (cond-> (kdispatch/validate! dispatch)
                    arena-id (assoc :arena-id arena-id))]
     (swap! kernel-dispatch-registry assoc (:id dispatch) dispatch)
     dispatch)))

(defn kernel-dispatch-registry-entry
  [dispatch-id]
  (get @kernel-dispatch-registry dispatch-id))

(defn- registered-1d-workgroup-size
  "Read a 1-D workgroup from the canonical artifact launch contract, or from the remaining plain
   registry entry while specialized emitters migrate. Never silently flatten a 2-D/3-D launch."
  [kernel-info]
  (if-let [launch (:launch kernel-info)]
    (let [workgroup (klaunch/static-workgroup-size launch)]
      (when-not (= 1 (count workgroup))
        (throw (ex-info "1-D kernel path received a multidimensional launch contract"
                        {:kernel-name (:kernel-name kernel-info) :launch launch})))
      (first workgroup))
    (long (or (:workgroup-size kernel-info) 256))))

(defn- kernel-info-value
  "Read a compiler-owned emitter attribute from an artifact or a remaining specialized entry."
  [kernel-info k default]
  (if (kart/kernel-artifact? kernel-info)
    (get (:attributes kernel-info) k default)
    (get kernel-info k default)))

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
  "Pipeline-friendly map invocation. Looks up the emitter-authored ordered ABI from the registry
  and stages/binds every pointer and scalar in that order.
  input-arrays: vector of JVM primitive arrays or DeviceBuffers
  output-array: JVM array or DeviceBuffer to receive results
  scalar-args: vector of scalar values
  n: element count
  The split call shape is retained for compiled-code compatibility; it is structurally checked
  against :abi before the driver is touched."
  [^String kernel-name input-arrays output-array scalar-args n]
  (let [registered (or (get @kernel-registry kernel-name)
                       (throw (ex-info (str "Kernel not registered: " kernel-name)
                                       {:kernel-name kernel-name
                                        :registered (keys @kernel-registry)})))
        abi (or (:abi registered)
                (throw (ex-info "registered map kernel has no ordered ABI"
                                {:kernel-name kernel-name :registry-entry registered
                                 :fallback :none})))
        abi (kabi/validate! abi)
        arguments (vec (concat input-arrays [output-array] scalar-args [n]))
        _ (kabi/validate-arguments! abi arguments)
        pairs (mapv vector abi arguments)
        pointer-count (count (kabi/pointer-slots abi))
        scalar-user-count (count (remove #(= :bound (:role %)) (kabi/scalar-slots abi)))
        _ (when-not (= pointer-count (inc (count input-arrays)))
            (throw (ex-info "map kernel ABI pointer count does not match marker"
                            {:kernel-name kernel-name :expected pointer-count
                             :actual (inc (count input-arrays)) :abi abi})))
        _ (when-not (= scalar-user-count (count scalar-args))
            (throw (ex-info "map kernel ABI scalar count does not match marker"
                            {:kernel-name kernel-name :expected scalar-user-count
                             :actual (count scalar-args) :abi abi})))
        _ (doseq [[slot value] pairs :when (not= :scalar (:kind slot))]
            (let [actual (if (device-buffer? value)
                           (dt/canon (:dtype ^DeviceBuffer value))
                           (dt/dtype-for-jvm-array value))]
              (when-not actual
                (throw (ex-info "map kernel ABI pointer value is not a supported buffer/array"
                                {:kernel-name kernel-name :slot slot :value-type (type value)})))
              (when-not (= (:dtype slot) actual)
                (throw (ex-info "map kernel ABI storage dtype mismatch"
                                {:kernel-name kernel-name :slot slot
                                 :expected (:dtype slot) :actual actual})))))
        ;; Loading/compilation may touch the native driver. Every ABI check above deliberately
        ;; runs first, so a malformed marker fails deterministically even on a machine without
        ;; the target device.
        {:keys [kernel-handle] :as loaded} (ensure-kernel-loaded! kernel-name)
        staged (mapv
                (fn [idx [slot value]]
                  (if (= :scalar (:kind slot))
                    (let [t (:kernel-dtype slot)]
                      {:arg {:type t
                             :value (case t
                                      :int (int value)
                                      :long (long value)
                                      :float (float value)
                                      :double (double value)
                                      value)}})
                    (if (device-buffer? value)
                      {:arg (:segment ^DeviceBuffer value) :value value :slot slot}
                      (let [host (MemorySegment/ofArray value)
                            n-bytes (.byteSize host)
                            seg (ensure-seg kernel-name (keyword (str "abi-arg-" idx)) n-bytes)]
                        (when (kabi/readable? slot)
                          (MemorySegment/copy host 0 seg 0 n-bytes))
                        {:arg seg :value value :host host :n-bytes n-bytes :slot slot}))))
                (range) pairs)
        all-args (mapv :arg staged)
        bound-pair (first (filter #(= :bound (:role (first %))) pairs))
        _ (when-not bound-pair
            (throw (ex-info "map kernel ABI has no :bound scalar" {:kernel-name kernel-name :abi abi})))
        n (long (second bound-pair))
        wg (registered-1d-workgroup-size loaded)
        group-count (long (Math/ceil (/ (double n) wg)))]
    (launch! kernel-handle group-count wg all-args)
    ;; Writable ABI kinds are the single source of copy-back truth. DeviceBuffers are already
    ;; resident and therefore need no host copy.
    (doseq [{:keys [arg host n-bytes slot]} staged
            :when (and host (kabi/writable? slot))]
      (MemorySegment/copy ^MemorySegment arg 0 ^MemorySegment host 0 (long n-bytes)))
    (or (some (fn [[slot value]] (when (= :result (:role slot)) value)) pairs)
        output-array)))

(defn invoke-registered-reduction-kernel
  "Run a SegRed host-scalar reduction from one complete ordered ABI value vector. The single
   :result value must be nil: staging owns and inserts the workgroup-partial buffer at exactly
   that slot. Captured scalars retain their declared kernel dtypes, and every structural/storage
   check runs before kernel loading touches the driver. Returns a scalar double after combining
   workgroup partials on the host."
  [^String kernel-name arguments]
  (let [registered (or (get @kernel-registry kernel-name)
                       (throw (ex-info (str "Kernel not registered: " kernel-name)
                                       {:kernel-name kernel-name
                                        :registered (keys @kernel-registry)})))
        abi (kabi/validate! (:abi registered))
        {:keys [pairs pointer-pairs scalar-pairs result-pair bound-pair]}
        (kabi/validate-reduction-arguments! abi arguments)
        _ (when-not (nil? (second result-pair))
            (throw (ex-info "staging reduction ABI :result value must be nil (runtime-owned partial buffer)"
                            {:kernel-name kernel-name :result-pair result-pair})))
        non-result-pointers (filterv #(not= :result (:role (first %))) pointer-pairs)
        _ (doseq [[slot value] non-result-pointers]
            (when-not (= :input (:kind slot))
              (throw (ex-info "staging reduction supports only input pointers besides its :result"
                              {:kernel-name kernel-name :slot slot})))
            (let [actual (if (device-buffer? value)
                           (dt/canon (:dtype ^DeviceBuffer value))
                           (dt/dtype-for-jvm-array value))]
              (when-not actual
                (throw (ex-info "reduction ABI pointer value is not a supported buffer/array"
                                {:kernel-name kernel-name :slot slot :value-type (type value)})))
              (when-not (= (:dtype slot) actual)
                (throw (ex-info "reduction ABI storage dtype mismatch"
                                {:kernel-name kernel-name :slot slot
                                 :expected (:dtype slot) :actual actual})))))
        _ (doseq [[slot value] scalar-pairs]
            (when (map? value)
              (when-not (= (:kernel-dtype slot) (:type value))
                (throw (ex-info "reduction ABI scalar binding has the wrong kernel dtype"
                                {:kernel-name kernel-name :slot slot
                                 :expected (:kernel-dtype slot) :actual (:type value)})))))
        [bound-slot bound-value] bound-pair
        _ (when (and (map? bound-value)
                     (not= (:kernel-dtype bound-slot) (:type bound-value)))
            (throw (ex-info "reduction ABI bound binding has the wrong kernel dtype"
                            {:kernel-name kernel-name :slot bound-slot
                             :expected (:kernel-dtype bound-slot) :actual (:type bound-value)})))
        n (long (if (map? bound-value) (:value bound-value) bound-value))
        result-dtype (:dtype (first result-pair))
        _ (when-not (#{:float :double} result-dtype)
            (throw (ex-info "host partial combine currently supports only float/double SegRed results"
                            {:kernel-name kernel-name :dtype result-dtype})))
        ;; Native driver/loading begins only after every ABI/value check above.
        loaded (ensure-kernel-loaded! kernel-name)
        kernel-handle (:kernel-handle loaded)
        workgroup-size (registered-1d-workgroup-size loaded)
        identity-val (or (get-in loaded [:attributes :identity-val])
                         (:identity-val loaded) 0.0)
        c-op (or (get-in loaded [:attributes :c-op]) (:c-op loaded) "+")
        combine (case c-op "fmax" (fn ^double [^double a ^double b] (Math/max a b))
                      "fmin" (fn ^double [^double a ^double b] (Math/min a b))
                      "*"    (fn ^double [^double a ^double b] (* a b))
                      (fn ^double [^double a ^double b] (+ a b)))
        wg (long (or workgroup-size 256))
        group-count (max 1 (long (Math/ceil (/ (double n) wg))))
        dtype-size (long (get dtype-byte-sizes result-dtype))
        value-layout (if (= result-dtype :float) ValueLayout/JAVA_FLOAT ValueLayout/JAVA_DOUBLE)
        staged-inputs
        (into {}
              (map-indexed
               (fn [idx [slot value]]
                 [slot
                  (if (device-buffer? value)
                    (:segment ^DeviceBuffer value)
                    (let [host (MemorySegment/ofArray value)
                          n-bytes (.byteSize host)
                          seg (ensure-seg kernel-name (keyword (str "red-abi-input-" idx)) n-bytes)]
                      (MemorySegment/copy host 0 seg 0 n-bytes)
                      seg))])
               non-result-pointers))
        partial-bytes (* group-count dtype-size)
        dev-partial (ensure-seg kernel-name :partial-seg partial-bytes)
        scalar-arg (fn [slot value]
                     (if (map? value)
                       value
                       (let [t (:kernel-dtype slot)]
                         {:type t
                          :value (case t
                                   :int (int value)
                                   :long (long value)
                                   :float (float value)
                                   :double (double value)
                                   value)})))
        all-args
        (mapv (fn [[slot value]]
                (cond
                  (= :result (:role slot)) dev-partial
                  (= :scalar (:kind slot)) (scalar-arg slot value)
                  :else (get staged-inputs slot)))
              pairs)]
    (launch! kernel-handle group-count wg all-args)
    (if (= group-count 1)
      (double (.get dev-partial value-layout 0))
      (loop [i 0 acc (double identity-val)]
        (if (< i group-count)
          (recur (inc i)
                 (double (combine acc (double (.get dev-partial value-layout
                                                    (* i dtype-size))))))
          acc)))))

;; ================================================================
;; Void-map kernel invocation (side-effect-only kernels)
;; ================================================================

(defn kernel-registry-entry
  "Registry info for a kernel-name (source, :array-params, :written-arrays, dtype…)."
  [kernel-name]
  (get @kernel-registry kernel-name))

(defn bind-registered-gemm!
  "Bind the XMX GEMM kernel (C = A×B) over RESIDENT fp16 DeviceBuffers for recording into a
  command graph — the resident analog of invoke-registered-gemm! (which stages JVM arrays every
  call). A:[m×k] B:[k×n] C:[m×n], all fp16 (:half) resident buffers, row-major. Returns a bound
  {:kernel :gc-seg …} map (128×128 XMX tiles → gc = ceil(n/128) × ceil(m/128)). A fresh kernel
  handle per binding (LZ kernel args are mutable handle state → shared handles clobber)."
  ([a b c m n k] (bind-registered-gemm! a b c m n k :half))
  ([a b c m n k c-dtype]
   ;; Fail loud on an output-buffer dtype mismatch: a :half kernel writing 2-byte halfs into a
   ;; :float (4-byte) buffer reads back as silent zeros/garbage — the exact silent-miscompile the
   ;; compiler is built to prevent. The kernel's output dtype IS c-dtype; the buffer must agree.
   (when-let [bd (:dtype c)]
     (when (not= bd c-dtype)
       (throw (ex-info (str "bind-registered-gemm!: output buffer dtype " bd " ≠ kernel output dtype " c-dtype
                            " — a mismatched write reads back as garbage. Allocate C as " c-dtype
                            " or pass the matching c-dtype.")
                       {:buffer-dtype bd :kernel-c-dtype c-dtype}))))
   (let [{:keys [module kernel-name tile workgroup]} (ensure-gemm-kernel! c-dtype)
         kh (create-kernel-fresh module kernel-name)
         m (long m) n (long n) k (long k)
         args [(:segment a) (:segment b) (:segment c)
               {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]
         bnd (bind-kernel! kh workgroup args)
         gc ^MemorySegment (:gc-seg bnd)]
     (.set gc I32 0 (int (Math/ceil (/ (double n) (double (:block-n tile))))))   ;; X = gc-n
     (.set gc I32 4 (int (Math/ceil (/ (double m) (double (:block-m tile))))))   ;; Y = gc-m
     (.set gc I32 8 (int 1))
     bnd)))

;; ── TILE-PARAMETRIC GEMM (autotune-facing) ─────────────────────────────────────
;; The default bind-registered-gemm! above schedules the device-derived default tile. This path
;; takes an EXPLICIT tile map (from schedule/derive-gemm-tile or an autotune candidate) through the
;; same KernelBody scheduler and derives launch geometry from that body. At the default tile it
;; produces identical source and launch geometry.

(def ^:private gemm-tiled-cache
  "Compiled tile-parametric GEMM kernels, keyed by [c-dtype tile-map]. Distinct tiles are distinct
   kernels (distinct __kernel names)."
  (atom {}))

(defn- tile-signature [tile]
  (let [{:keys [block-m block-n sg-m sg-n block-k num-stages]} tile]
    (str block-m "x" block-n "_" sg-m "x" sg-n "_k" block-k "_s" (or num-stages 3))))

(defn- ensure-gemm-kernel-tiled!
  "Compile + cache the GEMM kernel for [c-dtype × tile]. Returns {:module :kernel-name :tile}."
  [c-dtype tile]
  (ensure-init!)
  (or (get @gemm-tiled-cache [c-dtype tile])
      (let [kname (str "gemm_tiled_" (name c-dtype) "_" (tile-signature tile))
            emitted (emit-scheduled-gemm kname c-dtype tile nil)
            cl-src (:source emitted)
            spv (do (require 'raster.compiler.support.spirv-cache)
                    ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                     cl-src :device (:device-id-hex @state)))
            module (load-module! spv)
            entry {:module module :kernel-name kname :tile tile
                   :kernel-body (:kernel-body emitted)
                   :workgroup (:workgroup-size emitted)}]
        (swap! gemm-tiled-cache assoc [c-dtype tile] entry)
        entry)))

(defn bind-registered-gemm-tiled!
  "Bind the GEMM kernel for an EXPLICIT tile over resident fp16 buffers, DERIVING the launch
   geometry from the tile. `tile` is a derive-gemm-tile map {:block-m :block-n :sg-m :sg-n :block-k
   :matrix}. Fail-loud on an output-dtype mismatch, like bind-registered-gemm!."
  [a b c m n k c-dtype tile]
  (when-let [bd (:dtype c)]
    (when (not= bd c-dtype)
      (throw (ex-info (str "bind-registered-gemm-tiled!: output buffer dtype " bd " ≠ kernel dtype " c-dtype)
                      {:buffer-dtype bd :kernel-c-dtype c-dtype}))))
  (let [{:keys [module kernel-name workgroup]} (ensure-gemm-kernel-tiled! c-dtype tile)
        {:keys [block-m block-n]} tile
        kh   (create-kernel-fresh module kernel-name)
        args [(:segment a) (:segment b) (:segment c)
              {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]
        bnd  (bind-kernel! kh workgroup args)
        gc ^MemorySegment (:gc-seg bnd)]
    (.set gc I32 0 (int (Math/ceil (/ (double n) (double block-n)))))   ;; X = gc-n
    (.set gc I32 4 (int (Math/ceil (/ (double m) (double block-m)))))   ;; Y = gc-m
    (.set gc I32 8 (int 1))
    bnd))

;; ── FUSED-EPILOGUE GEMM (C = ScalarRegion(A·B, row, col)) ──────────────────────
;; One typed KernelBody for GEMM + a same-position elementwise consumer (bias/act/residual), the
;; training-M win (~15-18% at M≥512, measured). Runtime bindings remain separate from the scalar
;; program so cache identity and ABI order come from compiler data rather than source strings.
(def ^:private gemm-epilogue-cache (atom {}))

(defn- resident-epilogue-program
  [epilogue]
  (let [allowed #{:acc :expr :operands :scalars :dtype :bindings}
        unsupported (seq (remove allowed (keys epilogue)))]
    (when unsupported
      (throw (ex-info "resident GEMM epilogue contains unsupported fields"
                      {:unsupported (vec unsupported) :allowed allowed})))
    (when-not (map? (:bindings epilogue))
      (throw (ex-info "resident GEMM epilogue requires explicit runtime bindings"
                      {:epilogue epilogue})))
    (dissoc epilogue :bindings)))

(defn- ensure-gemm-epilogue-kernel!
  "Compile and cache a typed ScalarRegion GEMM by its structural program, output dtype and tile."
  [c-dtype tile epilogue]
  (ensure-init!)
  (let [program (resident-epilogue-program epilogue)
        cache-key [c-dtype tile program]]
    (or (get @gemm-epilogue-cache cache-key)
        (let [program-id (Integer/toUnsignedString (hash program) 16)
              kname (str "gemm_epi_" program-id "_" (name c-dtype) "_" (tile-signature tile))
              emitted (emit-scheduled-gemm kname c-dtype tile program)
              cl-src (:source emitted)
              spv (do (require 'raster.compiler.support.spirv-cache)
                      ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                       cl-src :device (:device-id-hex @state)))
              module (load-module! spv)
              entry {:module module :kernel-name kname :tile tile :program program
                     :kernel-body (:kernel-body emitted)
                     :workgroup (:workgroup-size emitted)}]
          (swap! gemm-epilogue-cache assoc cache-key entry)
          entry))))

(defn- resident-epilogue-argument
  [parameter value]
  (case (:kind parameter)
    :input
    (do
      (when-not (instance? DeviceBuffer value)
        (throw (ex-info "resident epilogue buffer binding is not a DeviceBuffer"
                        {:parameter parameter :actual (type value)})))
      (when-not (= (dt/canon (:dtype parameter)) (dt/canon (:dtype value)))
        (throw (ex-info "resident epilogue buffer dtype differs from its KernelBody ABI"
                        {:parameter parameter :actual (:dtype value)})))
      (:segment value))

    :scalar
    {:type (case (dt/canon (:dtype parameter))
             :int8 :byte
             :byte :byte
             :half :half
             :float :float
             :double :double
             :int :int
             :long :long
             (throw (ex-info "resident epilogue scalar dtype is unsupported"
                             {:parameter parameter})))
     :value value}

    (throw (ex-info "resident epilogue ABI slot must be input or scalar"
                    {:parameter parameter}))))

(defn bind-registered-gemm-epilogue!
  "Bind C = ScalarRegion(A·B) using the same typed epilogue descriptor as the compiler.

   `epilogue` contains :acc, :expr, ordered :operands/:scalars, and a :bindings map from each
   operand/scalar identity to its resident buffer or scalar value. KernelBody owns the physical
   ABI and launch geometry; target source callbacks and declaration strings are not accepted."
  [a b c m n k c-dtype tile epilogue]
  (when-let [bd (:dtype c)]
    (when (not= bd c-dtype)
      (throw (ex-info (str "bind-registered-gemm-epilogue!: output buffer dtype " bd " ≠ kernel dtype " c-dtype) {}))))
  (let [{:keys [module kernel-name kernel-body workgroup]}
        (ensure-gemm-epilogue-kernel! c-dtype tile epilogue)
        {:keys [block-m block-n]} tile
        parameters (filterv #(= :epilogue (:role %)) (:parameters kernel-body))
        expected (mapv :id parameters)
        bindings (:bindings epilogue)
        provided (set (keys bindings))
        _ (when-not (= (set expected) provided)
            (throw (ex-info "resident epilogue bindings differ from the KernelBody ABI"
                            {:expected expected :provided (vec (keys bindings))})))
        kh   (create-kernel-fresh module kernel-name)
        args (vec (concat [(:segment a) (:segment b) (:segment c)
                           {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]
                          (map #(resident-epilogue-argument % (get bindings (:id %))) parameters)))
        bnd  (bind-kernel! kh workgroup args)
        gc ^MemorySegment (:gc-seg bnd)]
    (.set gc I32 0 (int (Math/ceil (/ (double n) (double block-n)))))
    (.set gc I32 4 (int (Math/ceil (/ (double m) (double block-m)))))
    (.set gc I32 8 (int 1))
    bnd))

;; ── SPLIT-K GEMM (the low-occupancy-shape schedule) ────────────────────────────
;; A GEMM whose (M,N) tiling yields fewer workgroups than fill the machine —
;; ceil(N/128)·ceil(M/128) — cannot be rescued by a better inner loop: the machine
;; is idle. Splitting the K reduction across a third grid dimension multiplies the
;; workgroup count at CONSTANT DRAM traffic (each (n-tile, k-chunk) block of B is
;; still read exactly once), then a second kernel sums the per-chunk partials.
;; Measured lever: the tied-embedding backward dx[13,640] = dlogits[13,262144] ·
;; E[262144,640] launches 5 workgroups of the ~32 that fill this iGPU.

(def ^:private gemm-splitk-cache (atom nil))
(def ^:private splitk-reduce-cache (atom nil))

(defn- ensure-gemm-splitk-kernel!
  "Lazily compile + cache the split-k XMX gemm (f16 A/B in, f32 PARTIALS out).
   Returns {:module :kernel :kernel-name}."
  []
  (ensure-init!)
  (when (nil? @gemm-splitk-cache)
    (let [kname "gemm_nonsquare_splitk"
          tile (gemm-tile)
          emitted (emit-scheduled-split-k-gemm kname tile)
          cl-src (:source emitted)
          spv (do (require 'raster.compiler.support.spirv-cache)
                  ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                   cl-src :device (:device-id-hex @state)))
          module (load-module! spv)
          kernel (create-kernel module kname)]
      (clojure.core/reset! gemm-splitk-cache
                           {:module module :kernel kernel :kernel-name kname :tile tile
                            :kernel-body (:kernel-body emitted)
                            :workgroup (:workgroup-size emitted)})))
  @gemm-splitk-cache)

(defn- ensure-splitk-reduce-kernel!
  "Lazily compile + cache the split-k partials-combine kernel."
  []
  (ensure-init!)
  (when (nil? @splitk-reduce-cache)
    (let [kname "gemm_splitk_reduce"
          src (do (require 'raster.compiler.backend.gpu.opencl-codegen)
                  ((resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-gemm-splitk-reduce-kernel)
                   kname))
          spv (do (require 'raster.compiler.support.spirv-cache)
                  ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                   src :device (:device-id-hex @state)))
          module (load-module! spv)
          kernel (create-kernel module kname)]
      (clojure.core/reset! splitk-reduce-cache
                           {:module module :kernel kernel :kernel-name kname})))
  @splitk-reduce-cache)

(defn bind-registered-gemm-splitk!
  "Bind the SPLIT-K XMX GEMM: A[m×k]·B[k×n] → `partials` [splits, m, n] f32.
  Grid is 3D — X = ceil(n/128), Y = ceil(m/128), Z = splits — so the launched
  workgroup count is `splits` times the plain GEMM's. `kc` is the k-chunk each
  z-slice reduces (must be a multiple of 32 so no interior chunk hits the k-remainder
  path; the LAST chunk clamps to k). Pair with bind-registered-splitk-reduce!."
  [a b partials m n k kc splits]
  (let [{:keys [module kernel-name tile workgroup]} (ensure-gemm-splitk-kernel!)
        {:keys [block-m block-n]} tile
        kh (create-kernel-fresh module kernel-name)
        m (long m) n (long n) k (long k) kc (long kc) splits (long splits)
        args [(:segment a) (:segment b) (:segment partials)
              {:type :int :value (int m)} {:type :int :value (int n)}
              {:type :int :value (int k)} {:type :int :value (int kc)}
              {:type :int :value (int splits)}]
        bnd (bind-kernel! kh workgroup args)
        gc ^MemorySegment (:gc-seg bnd)]
    (.set gc I32 0 (int (Math/ceil (/ (double n) (double block-n))))) ;; X = gc-n
    (.set gc I32 4 (int (Math/ceil (/ (double m) (double block-m))))) ;; Y = gc-m
    (.set gc I32 8 (int splits))                             ;; Z = k-chunks
    bnd))

(defn bind-registered-splitk-reduce!
  "Bind the split-k combine: C[i] = Σ_s partials[s·mn + i], one work-item per output
  element of C (mn = m·n)."
  [partials c mn splits]
  (let [{:keys [module kernel-name]} (ensure-splitk-reduce-kernel!)
        kh (create-kernel-fresh module kernel-name)
        mn (long mn) splits (long splits)
        args [(:segment partials) (:segment c)
              {:type :int :value (int mn)} {:type :int :value (int splits)}]
        bnd (bind-kernel! kh 256 args)]
    (.set ^MemorySegment (:gc-seg bnd) I32 0 (int (Math/ceil (/ (double mn) 256.0))))
    bnd))

;; ── BATCHED XMX GEMM (bmm) ───────────────────────────────────────────────────
;; A per-slab GEMM whose (M,N) tiling launches too few workgroups to fill the
;; machine is occupancy-bound (see split-k). When the deficit is instead resolved
;; by MANY independent slabs (attention over heads: dV[b]=W[b]ᵀ·dO[b], 64 slabs of
;; m=k=64,n=256), a single batched kernel over a 3D grid (z = slab) launches
;; slabs × per-slab-tiles workgroups — feeding the DPAS array across all slabs where
;; one slab starves it. A/B/C are contiguous [batch,M,K]/[batch,K,N]/[batch,M,N]
;; f16/f16/f32 buffers; each workgroup offsets its base by its slab. This is the
;; batched-nn primitive; the -tn/-nt layouts are handled by staging A/B (convert +
;; batched transpose) exactly as the single GEMM's resident path does.

(def ^:private gemm-batched-cache (atom nil))

(defn- ensure-gemm-batched-kernel!
  "Lazily compile + cache the BATCHED XMX gemm (f16 A/B in, f32 C out, grid-z=slab).
   Returns {:module :kernel :kernel-name}."
  []
  (ensure-init!)
  (when (nil? @gemm-batched-cache)
    (let [kname "gemm_nonsquare_batched"
          tile (gemm-tile)
          emitted (emit-scheduled-batched-gemm kname tile)
          cl-src (:source emitted)
          spv (do (require 'raster.compiler.support.spirv-cache)
                  ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                   cl-src :device (:device-id-hex @state)))
          module (load-module! spv)
          kernel (create-kernel module kname)]
      (clojure.core/reset! gemm-batched-cache
                           {:module module :kernel kernel :kernel-name kname :tile tile
                            :kernel-body (:kernel-body emitted)
                            :workgroup (:workgroup-size emitted)})))
  @gemm-batched-cache)

(defn bind-registered-gemm-batched!
  "Bind the BATCHED XMX GEMM: C[b] = A[b]·B[b] (nn) for b in 0..batch-1, over a 3D
  grid — X = ceil(n/128), Y = ceil(m/128), Z = batch — so the launched workgroup
  count is `batch`× the plain GEMM's. A[batch,m,k] & B[batch,k,n] f16, C[batch,m,n]
  f32, all contiguous. Fresh kernel handle per bind (LZ kernel args are mutable
  handle state)."
  [a b c m n k batch]
  (let [{:keys [module kernel-name tile workgroup]} (ensure-gemm-batched-kernel!)
        {:keys [block-m block-n]} tile
        kh (create-kernel-fresh module kernel-name)
        m (long m) n (long n) k (long k) batch (long batch)
        args [(:segment a) (:segment b) (:segment c)
              {:type :int :value (int m)} {:type :int :value (int n)}
              {:type :int :value (int k)} {:type :int :value (int batch)}]
        bnd (bind-kernel! kh workgroup args)
        gc ^MemorySegment (:gc-seg bnd)]
    (.set gc I32 0 (int (Math/ceil (/ (double n) (double block-n))))) ;; X = gc-n
    (.set gc I32 4 (int (Math/ceil (/ (double m) (double block-m))))) ;; Y = gc-m
    (.set gc I32 8 (int batch))                              ;; Z = slabs
    bnd))

(def ^:private gemm-scalar-cache
  "Cache for compiled scalar (non-XMX) GEMM kernels, keyed by variant (:nn|:nt|:tn).
   Each entry is {:module :kernel :kernel-name}. f32 in/out — the small-N fallback."
  (atom {}))

(defn- ensure-gemm-scalar-kernel!
  "Lazily compile + cache the plain scalar f32 GEMM kernel for a layout variant
   (:nn | :nt | :tn). Returns {:module :kernel :kernel-name}. Used when the output
   column dim N is too small for the XMX 2D-block path (2D-block IO needs a
   >=16-byte pitch → N>=8 at fp16; N<8 reads garbage)."
  [variant]
  (ensure-init!)
  (or (get @gemm-scalar-cache variant)
      (let [kname (str "gemm_scalar_" (name variant))
            cl-src (do (require 'raster.compiler.backend.gpu.opencl-codegen)
                       ((resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-gemm-scalar-kernel)
                        kname :variant variant))
            device-hex (:device-id-hex @state)
            spv (do (require 'raster.compiler.support.spirv-cache)
                    ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                     cl-src :device device-hex))
            module (load-module! spv)
            kernel (create-kernel module kname)
            entry {:module module :kernel kernel :kernel-name kname}]
        (swap! gemm-scalar-cache assoc variant entry)
        entry)))

(defn bind-registered-gemm-scalar!
  "Bind the plain scalar (non-XMX) f32 GEMM kernel over RESIDENT f32 DeviceBuffers for
  recording into a command graph—the small-N fallback for linked :gemm steps
  (XMX's 2D-block B read violates the 16-byte minimum pitch when N<8 and produces
  garbage). Reads/writes the f32 buffers directly: no f16 convert or transpose expansion.
  variant :nn (C=A·B), :nt (C=A·Bᵀ, B stored [n,k]), :tn (C=Aᵀ·B, A stored [k,m]).
  Returns a bound {:kernel :gc-seg …} map (1D grid over m·n). A fresh kernel handle per
  binding (LZ kernel args are mutable handle state → shared handles clobber)."
  [a b c m n k variant]
  (let [{:keys [module kernel-name]} (ensure-gemm-scalar-kernel! variant)
        kh (create-kernel-fresh module kernel-name)
        m (long m) n (long n) k (long k)
        total (* m n)
        args [(:segment a) (:segment b) (:segment c)
              {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]
        bnd (bind-kernel! kh 256 args)]
    (.set ^MemorySegment (:gc-seg bnd) I32 0 (int (Math/ceil (/ (double total) 256.0))))
    bnd))

(def ^:private convert-cache (atom {}))

(defn- convert-source
  "Compatibility substrate for raw benchmark callers. Production graph schedules consume the
   same compiler-owned conversion emitter directly."
  [w]
  ((requiring-resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-f32-to-f16-kernel)
   "f32_to_f16" w))

(defn- ensure-convert-kernel!
  "Lazily compile + cache the f32→f16 convert kernel for vector width `w`. {:module :kernel}."
  [w]
  (ensure-init!)
  (or (get @convert-cache w)
      (let [spv (do (require 'raster.compiler.support.spirv-cache)
                    ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                     (convert-source w) :device (:device-id-hex @state)))
            module (load-module! spv)
            entry {:module module :kernel (create-kernel module "f32_to_f16")}]
        (swap! convert-cache assoc w entry)
        entry)))

(defn bind-registered-convert!
  "Bind an f32→f16 element-wise convert kernel over RESIDENT buffers for recording into a
  command graph (the Option-B activation cast before an fp16 XMX GEMM). in: f32 DeviceBuffer,
  out: f16 (:half) DeviceBuffer, n elements. Returns a bound {:kernel :gc-seg} map. Fresh
  kernel handle per binding.

  `w` = elements per work-item (the vector width the CALLER scheduled from the hardware
  descriptor); default 1 = the scalar kernel."
  ([in out n] (bind-registered-convert! in out n 1))
  ([in out n w]
   (let [{:keys [module]} (ensure-convert-kernel! w)
         kh (create-kernel-fresh module "f32_to_f16")
         n (long n) w (long w)
         args [(:segment in) (:segment out) {:type :int :value (int n)}]
         bnd (bind-kernel! kh 256 args)
         ;; one work-item per w elements. Floor at one group so an n < w program still
         ;; launches the work-items its tail loop needs.
         gc (max 1 (long (Math/ceil (/ (double (quot n w)) 256.0))))]
     (.set ^MemorySegment (:gc-seg bnd) I32 0 (int gc))
     bnd)))

(def ^:private transpose-cache (atom {}))

(defn- ensure-transpose-kernel!
  "Lazily compile + cache a 2D transpose kernel for a dtype (:half | :float | :byte). Generic
   on element width via opencl-type-map (:byte → char). Returns {:module :kernel :kernel-name}."
  [dtype]
  (ensure-init!)
  (or (get @transpose-cache dtype)
      (let [kname (str "transpose_" (name dtype))
            src (do (require 'raster.compiler.backend.gpu.opencl-codegen)
                    ((resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-transpose-kernel)
                     kname :dtype dtype))
            spv (do (require 'raster.compiler.support.spirv-cache)
                    ((resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
                     src :device (:device-id-hex @state)))
            module (load-module! spv)
            kernel (create-kernel module kname)
            entry {:module module :kernel kernel :kernel-name kname}]
        (swap! transpose-cache assoc dtype entry)
        entry)))

(defn bind-registered-transpose!
  "Bind a 2D transpose kernel (in[rows,cols] → out[cols,rows], row-major) over RESIDENT buffers
  for recording — used to realize dgemm-nt!/-tn! by transposing an operand before the :nn XMX
  GEMM, and to prepare an int8 :nt operand for the dp4a peak leaf (B3-insert). dtype :half
  (default) | :float | :byte (int8). The kernel transposes at ELEMENT granularity for its dtype:
  for int8 operands it MUST be bound :byte — transposing at :int (packed int32) granularity
  would permute packed words and scramble dp4a's K-packing (silent garbage). Returns a bound
  {:kernel :gc-seg} map."
  ([in out rows cols] (bind-registered-transpose! in out rows cols :half))
  ([in out rows cols dtype]
   (let [{:keys [module kernel-name]} (ensure-transpose-kernel! dtype)
         kh (create-kernel-fresh module kernel-name)
         total (* (long rows) (long cols))
         args [(:segment in) (:segment out) {:type :int :value (int rows)} {:type :int :value (int cols)}]
         bnd (bind-kernel! kh 256 args)]
     (.set ^MemorySegment (:gc-seg bnd) I32 0 (int (Math/ceil (/ (double total) 256.0))))
     bnd)))

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
   (let [abi (:abi (get @kernel-registry kernel-name))
         _ (when abi
             (kabi/validate-split-binding! abi arrays scalar-args)
             (kabi/validate-physical-pointer-dtypes! abi (physical-pointer-dtypes arrays)))
         {:keys [kernel-handle] :as info} (ensure-kernel-loaded! kernel-name)
         dtype (kernel-info-value info :dtype :float)
         workgroup-size (long (get opts :workgroup-size
                                   (registered-1d-workgroup-size info)))
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
                           (instance? (Class/forName "[S") arr)
                           (* (alength ^shorts arr) 2)
                           (instance? (Class/forName "[B") arr)
                           (alength ^bytes arr)
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
         expanded-entries
         (if abi
           (mapv (fn [entry slot]
                   (assoc entry :write? (kabi/writable? slot)))
                 expanded-entries (kabi/pointer-slots abi))
           expanded-entries)
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
     (doseq [{:keys [seg source ab write?]} expanded-entries]
       (when (and source (or (nil? abi) write?))
         (MemorySegment/copy seg 0 (MemorySegment/ofArray source) 0 (long ab))))
     nil)))

(defn bind-kernel-call
  "Bind a backend-neutral KernelCall over Level Zero resident buffers. ABI order and complete
   1-3D geometry come exclusively from the call; no map/reduction convention is interpreted."
  [call]
  (let [{:keys [kernel-name abi pairs pointer-pairs workgroup-size group-count] :as plan}
        (kcall/binding-plan call)
        registered (or (get @kernel-registry kernel-name)
                       (throw (ex-info (str "Kernel not registered: " kernel-name)
                                       {:kernel-name kernel-name
                                        :registered (keys @kernel-registry)})))
        _ (kcall/validate-registered! call registered)
        pointer-values (mapv second pointer-pairs)
        _ (doseq [[slot value] pointer-pairs]
            (when-not (or (device-buffer? value) (instance? MemorySegment value))
              (throw (ex-info "Level Zero KernelCall requires DeviceBuffer/MemorySegment pointers"
                              {:kernel-name kernel-name :slot slot :value-type (type value)}))))
        _ (kabi/validate-physical-pointer-dtypes!
           abi (physical-pointer-dtypes pointer-values))
        _ (when (= :pure-reduction (get-in registered [:effects :kind]))
            (doseq [[slot value] pointer-pairs
                    :when (= :result (:role slot))]
              (when (and (device-buffer? value) (< (:n-elements ^DeviceBuffer value) 1))
                (throw (ex-info "resident reduction result buffer must hold at least one element"
                                {:kernel-name kernel-name :slot slot :buffer-elements 0})))))
        _ (when (= :tensor-contraction (get-in registered [:effects :kind]))
            (let [extent-expr (kart/attribute registered :out-elems)
                  out-elems (long (if (number? extent-expr)
                                    extent-expr
                                    (kcall/resolve-value call extent-expr)))]
              (when (neg? out-elems)
                (throw (ex-info "resident contraction output extent must be non-negative"
                                {:kernel-name kernel-name :out-elems out-elems})))
              (doseq [[slot value] pointer-pairs :when (= :result (:role slot))]
                (let [capacity (cond
                                 (device-buffer? value) (:n-elements ^DeviceBuffer value)
                                 (instance? MemorySegment value)
                                 (quot (.byteSize ^MemorySegment value) (dt/bytes-of (:dtype slot))))]
                  (when (< (long capacity) out-elems)
                    (throw (ex-info "contraction output buffer is smaller than its artifact extent"
                                    {:kernel-name kernel-name :slot slot :out-elems out-elems
                                     :buffer-elements capacity})))))))
        ;; Driver contact begins only after call/artifact/ABI/value/geometry validation.
        {:keys [module]} (ensure-kernel-loaded! kernel-name)
        kernel-handle (create-kernel-fresh module kernel-name)
        native-args (mapv (fn [[slot value]]
                            (if (= :scalar (:kind slot))
                              value
                              (if (device-buffer? value)
                                (:segment ^DeviceBuffer value)
                                value)))
                          pairs)
        bound (bind-kernel! kernel-handle workgroup-size native-args)
        ^MemorySegment gc (:gc-seg bound)]
    (doseq [[axis count] (map-indexed vector (take 3 (concat group-count [1 1])))]
      (.set gc I32 (long (* axis 4)) (int count)))
    {:bound bound
     ;; Geometry is already baked into gc-seg. record-graph! must not reinterpret X specially.
     :group-count nil
     :kernel-name kernel-name
     :kernel-call call
     :binding-plan plan}))

(defn bind-registered-map-void-kernel
  "Pre-bind a registered void-map kernel's arguments ONCE for fast repeated dispatch.
  All array args must be GPU-resident (DeviceBuffer / GpuSoA / MemorySegment) — no per-call
  JVM-array staging copy is allowed, since the whole point is to skip per-launch arg setup.
  Buffer CONTENTS may change between launches (the bound pointers are stable); only re-bind
  if a buffer is reallocated or n changes. Returns a map for launch-registered-bound!.

  This is the dispatch-overhead fix: launch! re-sets every arg + appends a barrier each call
  (~450µs measured); a pre-bound kernel dispatches via launch-bound! (no arg setup, no barrier)."
  ([^String kernel-name arrays scalar-args n]
   (bind-registered-map-void-kernel kernel-name arrays scalar-args n {}))
  ([^String kernel-name arrays scalar-args n opts]
   (let [_ (when-let [abi (:abi (get @kernel-registry kernel-name))]
             (kabi/validate-split-binding! abi arrays scalar-args)
             (kabi/validate-physical-pointer-dtypes! abi (physical-pointer-dtypes arrays)))
         {:keys [module] :as loaded} (ensure-kernel-loaded! kernel-name)
         dtype (kernel-info-value loaded :dtype :float)
         ;; CRITICAL: create a DEDICATED kernel handle per binding. Level Zero kernel args are
         ;; mutable state ON the kernel handle, so reusing the registry's shared handle would
         ;; make every binding of the same kernel clobber the others (the last prepare! wins).
         ;; A fresh handle per binding gives each its own arg state — required for the decode
         ;; pattern (N matmuls share one kernel SOURCE but need N independent arg sets).
         kernel-handle (create-kernel-fresh module kernel-name)
         workgroup-size (long (get opts :workgroup-size
                                   (registered-1d-workgroup-size loaded)))
         n (long n)
         dev-segs (reduce
                   (fn [acc arr]
                     (cond
                       (gpu-soa? arr)        (into acc (mapv :seg (:field-segs ^GpuSoA arr)))
                       (device-buffer? arr)  (conj acc (:segment ^DeviceBuffer arr))
                       (instance? MemorySegment arr) (conj acc arr)
                       :else (throw (ex-info "bind-registered-map-void-kernel requires GPU-resident args (DeviceBuffer/GpuSoA); JVM-array staging is not supported on the bound path"
                                             {:arr-type (type arr)}))))
                   [] arrays)
         scalar-type (if (= dtype :float) :float :double)
         scalar-kernel-args (mapv (fn [v]
                                    (if (map? v) v
                                        {:type scalar-type
                                         :value (if (= scalar-type :float) (float v) (double v))}))
                                  scalar-args)
         all-args (vec (concat dev-segs scalar-kernel-args [{:type :int :value (int n)}]))
         wg (long workgroup-size)
         ;; A reduction binds via this same path (arrays..., scalars..., _n_bound) but launches a
         ;; SINGLE workgroup so the kernel's grid-stride loop covers all n and writes output[0] —
         ;; the result stays device-resident with no host cross-group combine. Callers pass
         ;; {:group-count 1}; map/map-void leave it nil and get the full ceil(n/wg) grid.
         group-count (long (or (get opts :group-count) (Math/ceil (/ (double n) wg))))
         async? (boolean (get opts :async?))
         cmd-list (if async? (async-cmd-list) (:cmd-list @state))]
     {:bound (bind-kernel! kernel-handle wg all-args cmd-list)
      :group-count group-count
      :kernel-name kernel-name
      :async? async?})))

(defn launch-registered-bound!
  "Dispatch a pre-bound kernel. A KernelCall has its complete geometry baked into :gc-seg;
   compatibility bindings may still supply a scalar :group-count."
  [prepared]
  (if-some [group-count (:group-count prepared)]
    (launch-bound! (:bound prepared) group-count)
    (let [bound (:bound prepared)]
      (.invokeWithArguments ^MethodHandle (:h-launch bound)
                            ^"[Ljava.lang.Object;" (:launch-args bound)))))

(defn ensure-zero-fill-kernel!
  "Register (idempotently) a zero-fill kernel `out[i]=0` for the given dtype and
   return its name. Used by the resident scatter step to re-zero the accumulator
   buffer each replay (scatter-add's output has zeros-like semantics; the atomic
   `+=` kernel must start from a cleared buffer). C-sig: (out, int n) — matches the
   (array, _n_bound) layout bind-registered-map-void-kernel expects, so the zero-fill
   binds through that same path. Idempotency is keyed on the actual kernel-registry
   (not a side atom) so it survives ns reload."
  [dtype]
  (let [dtype (if (= dtype :double) :double :float)
        ctype (if (= dtype :double) "double" "float")
        kname (str "zero_fill_" (name dtype))]
    (when-not (get @kernel-registry kname)
      (register-kernel! kname
                        {:source (str "__kernel void " kname
                                      "(__global " ctype "* out, int n) {\n"
                                      "  for (int i = get_global_id(0); i < n; i += get_global_size(0))\n"
                                      "    out[i] = 0;\n}\n")
                         :array-params ['out]
                         :scalar-params []
                         :written-arrays ['out]
                         :dtype dtype}))
    kname))

(defn bind-registered-scatter-kernel!
  "Pre-bind a registered scatter-add kernel over RESIDENT buffers for the replayable
   program graph. arrays = [out src index] (DeviceBuffer/MemorySegment); the index
   buffer is int32. Kernel C-sig: (out, src, index, int n [, int stride]) — n precedes
   stride (NOT the arrays,scalars,n order of a map-void). One work-item per source pair,
   grid = ceil(n/wg); the atomic `+=` fans overlapping indices in safely. `out` must be
   pre-zeroed (the resident binder prepends a zero-fill). Returns {:bound :group-count}."
  ([^String kernel-name arrays n] (bind-registered-scatter-kernel! kernel-name arrays n nil))
  ([^String kernel-name arrays n stride]
   (let [{:keys [module workgroup-size]
          :or {workgroup-size 256}} (ensure-kernel-loaded! kernel-name)
         kernel-handle (create-kernel-fresh module kernel-name)
         wg (long workgroup-size)
         n (long n)
         seg-of (fn [arr]
                  (cond
                    (device-buffer? arr) (:segment ^DeviceBuffer arr)
                    (instance? MemorySegment arr) arr
                    :else (throw (ex-info "bind-registered-scatter-kernel! requires resident buffers (DeviceBuffer/MemorySegment)"
                                          {:arr-type (type arr)}))))
         dev-segs (mapv seg-of arrays)
         all-args (vec (concat dev-segs
                               [{:type :int :value (int n)}]
                               (when stride [{:type :int :value (int stride)}])))
         group-count (long (Math/ceil (/ (double n) wg)))]
     {:bound (bind-kernel! kernel-handle wg all-args (:cmd-list @state))
      :group-count group-count
      :kernel-name kernel-name})))

(defn record-graph!
  "Record a FIXED ordered sequence of pre-bound kernels into a regular (replayable) command
  list — a 'command graph'. The per-launch host-append cost (FFI + driver, ~75µs each) is paid
  ONCE here; replay-graph! then dispatches the whole sequence with a single queue execute. This
  is the AOT decode-graph: the kernel sequence, group sizes, and buffer POINTERS are baked in at
  record time; buffer CONTENTS may change freely between replays (the decode pattern — each
  token rewrites the activation buffers, the recorded matmuls read whatever is there).

  prepareds: ordered seq from bind-kernel-call or a specialized compatibility binder (each
  carries its own dedicated kernel handle with args already set). Returns a graph handle for replay-graph!.
  Re-record only if the kernel sequence or any buffer is reallocated.

  Profiling (:profile? true): one kernel-timestamp device event per launch is created from a
  dedicated event pool and passed as the launch's signal event; read-graph-timestamps! after a
  replay returns per-kernel device times in execution order (names/phases read from each
  prepared's :kernel-name/:phase). Opt-in ONLY: with :profile? false (default) the recorded
  command list is exactly today's fast path (NULL signal events, no pool, no extra keys on the
  returned graph map). A profiled graph must have its events reset between replays —
  read-graph-timestamps! does so after reading; reset-graph-events! does it without reading."
  ([prepareds] (record-graph! prepareds {:barriers? true}))
  ([prepareds {:keys [barriers? profile?] :or {barriers? true profile? false}}]
   (ensure-init!)
   (let [{:keys [arena context device]} @state
         cq-desc (.allocate ^Arena arena 40)
         _ (.set cq-desc I32 0 (int ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC))
         ;; The public replay path still waits, but submit-graph! can now return immediately and
         ;; expose queue completion through the backend-neutral runtime event contract.
         _ (.set cq-desc I32 28 (int ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS))
         q-out (ptr-seg arena)
         _ (ze-call! "zeCommandQueueCreate" @h-zeCommandQueueCreate
                     [context device cq-desc q-out])
         queue (read-ptr q-out)
         cl-desc (.allocate ^Arena arena 24)
         _ (.set cl-desc I32 0 (int ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC))
         l-out (ptr-seg arena)
         _ (ze-call! "zeCommandListCreate" @h-zeCommandListCreate
                     [context device cl-desc l-out])
         lst (read-ptr l-out)
         h-launch @h-zeCommandListAppendLaunchKernel
         h-barrier @h-zeCommandListAppendBarrier
         n-kernels (count prepareds)
         ;; Profiling event pool + one timestamp event per launch. HOST_VISIBLE so the host
         ;; can query/reset; KERNEL_TIMESTAMP so zeEventQueryKernelTimestamp is valid.
         event-pool (when (and profile? (pos? n-kernels))
                      (let [pool-desc (.allocate ^Arena arena 24)
                            _ (.set pool-desc I32 0 (int ZE_STRUCTURE_TYPE_EVENT_POOL_DESC))
                            _ (.set pool-desc I32 16 (int (bit-or ZE_EVENT_POOL_FLAG_HOST_VISIBLE
                                                                  ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP)))
                            _ (.set pool-desc I32 20 (int n-kernels))
                            p-out (ptr-seg arena)]
                        (ze-call! "zeEventPoolCreate" @h-zeEventPoolCreate
                                  [context pool-desc (int 0) MemorySegment/NULL p-out])
                        (read-ptr p-out)))
         events (when event-pool
                  (mapv (fn [i]
                          (let [ev-desc (.allocate ^Arena arena 32)
                                _ (.set ev-desc I32 0 (int ZE_STRUCTURE_TYPE_EVENT_DESC))
                                _ (.set ev-desc I32 16 (int i))  ;; index
                                ;; signal scope 0 (in-command-list only), NOT HOST: a HOST-scope
                                ;; signal forces a cache-hierarchy flush after EVERY kernel, which
                                ;; measured 3-5x slower on the gemma layer graphs (fwd 32→170 ms)
                                ;; — a profiler must not perturb what it measures. The pool is
                                ;; HOST_VISIBLE, so event status + kernel timestamps are readable
                                ;; from the host after the synchronous queue-execute returns
                                ;; (validated vs host wall time on iGPU; re-validate on dGPU).
                                _ (.set ev-desc I32 20 (int 0))  ;; signal scope
                                _ (.set ev-desc I32 24 (int 0))  ;; wait scope
                                e-out (ptr-seg arena)]
                            (ze-call! "zeEventCreate" @h-zeEventCreate
                                      [event-pool ev-desc e-out])
                            (read-ptr e-out)))
                        (range n-kernels)))]
    ;; append each bound kernel's launch (args already live on its dedicated handle; the
    ;; group-count value is captured into the recorded command at append time). A barrier
    ;; after each launch enforces ordering so a kernel sees the previous one's writes — the
    ;; decode chain is dependent (norm→quant→matmul→…). Independent ops still serialize, but
    ;; each is memory-bound so overlap buys little; correctness first. (Measured: an IN_ORDER
    ;; command list with no barriers is the same replay time — the ~50µs/kernel floor on this
    ;; iGPU is per-dispatch overhead, not the barrier. Fewer/bigger kernels is the lever.)
     (doseq [[i {:keys [bound group-count]}] (map-indexed vector prepareds)]
       (let [^MemorySegment gc (:gc-seg bound)
             signal-ev (if events (nth events i) MemorySegment/NULL)]
        ;; A 1D map/map-void carries its X-grid as group-count (Y=Z=1 pre-filled). A GEMM /
        ;; convert / transpose expansion pre-bakes its FULL (2D) gc-seg at bind time and passes
        ;; group-count = nil — leave its X/Y/Z untouched (overwriting X would break the 2D grid).
         (when (some? group-count) (.set gc I32 0 (int group-count)))
         (ze-call! "zeCommandListAppendLaunchKernel" h-launch
                   [lst (:kernel bound) gc signal-ev (int 0) MemorySegment/NULL])
         (when barriers?
           (ze-call! "zeCommandListAppendBarrier" h-barrier
                     [lst MemorySegment/NULL (int 0) MemorySegment/NULL]))))
     (ze-call! "zeCommandListClose" @h-zeCommandListClose [lst])
     (let [lists-arr (ptr-seg arena)]
       (.set lists-arr PTR 0 ^MemorySegment lst)
       (cond-> {:queue queue :list lst :lists-arr lists-arr}
         events (assoc :events events
                       :event-pool event-pool
                       :kernel-names (mapv #(or (:kernel-name %) "unknown") prepareds)
                       :phases (mapv :phase prepareds)))))))

(defn submit-graph!
  "Submit a recorded graph without waiting. Returns a runtime-private completion token.
   The token contains a Level Zero queue handle, but it never enters compiler IR or the public
   gpu.core event value. A graph permits one in-flight submission at a time at the core layer."
  [graph]
  (ze-call! "zeCommandQueueExecuteCommandLists" @h-zeCommandQueueExecuteCommandLists
            [(:queue graph) (int 1) (:lists-arr graph) MemorySegment/NULL])
  {:queue (:queue graph)})

(defn await-event!
  "Wait for a runtime-private graph completion token."
  [{:keys [complete? completion queue]}]
  (if complete?
    completion
    (do
      (ze-call! "zeCommandQueueSynchronize" @h-zeCommandQueueSynchronize
                [queue (long -1)])
      nil)))

(defn event-complete?
  "Nonblocking query of a runtime-private graph completion token."
  [{:keys [complete? queue]}]
  (if complete?
    true
    (let [result (int (.invokeWithArguments
                       ^MethodHandle @h-zeCommandQueueSynchronize
                       ^java.util.List (java.util.List/of
                                        (object-array [queue (long 0)]))))]
      (cond
        (= result ZE_RESULT_SUCCESS) true
        (= result ZE_RESULT_NOT_READY) false
        :else (throw (ex-info (str "Level Zero error querying graph event: 0x"
                                   (Integer/toHexString result))
                              {:result result :context "zeCommandQueueSynchronize"}))))))

(defn release-event!
  "Release a runtime-private graph completion token. Queue ownership remains with the graph."
  [_event]
  nil)

(defn replay-graph!
  "Execute a recorded command graph once and wait for completion."
  [graph]
  (let [event (submit-graph! graph)]
    (try
      (await-event! event)
      (finally
        (release-event! event)))))

(defn- destroy-handle!
  [^MethodHandle mh ^MemorySegment seg]
  (when (and seg (not (.equals MemorySegment/NULL seg)))
    (try (.invokeWithArguments mh ^java.util.List (java.util.List/of (object-array [seg])))
         (catch Exception _))))

(defn destroy-prepared!
  "Destroy the DEDICATED kernel handle a prepared binding owns (create-kernel-fresh allocates one
  per bind; it is NOT in the registry's cache, so shutdown!/free-arena-kernels! never reach it).
  Without this every prepare!/bind leaks a zeKernel driver object → the driver's kernel table
  fills → zeKernelCreate / launch eventually fail (SIGABRT). Idempotent; safe on partial maps."
  [prepared]
  (destroy-handle! @h-zeKernelDestroy (get-in prepared [:bound :kernel])))

(defn destroy-graph!
  "Destroy a recorded command graph's queue + list (record-graph! creates one of each per graph
  and never frees them), plus any profiling events + event pool (:profile? graphs). Without this
  every record-graph! leaks a zeCommandQueue + zeCommandList."
  [graph]
  (destroy-handle! @h-zeCommandListDestroy (:list graph))
  (destroy-handle! @h-zeCommandQueueDestroy (:queue graph))
  (doseq [ev (:events graph)]
    (destroy-handle! @h-zeEventDestroy ev))
  (when-let [pool (:event-pool graph)]
    (destroy-handle! @h-zeEventPoolDestroy pool)))

(defn reset-graph-events!
  "Host-reset all profiling events of a graph recorded with :profile? true, WITHOUT reading
  them — required between replays (re-signaling an already-signaled event is invalid). No-op
  for non-profiled graphs. read-graph-timestamps! resets as part of reading; call this instead
  when a replay's timestamps are not wanted (e.g. warmup)."
  [graph]
  (doseq [ev (:events graph)]
    (ze-call! "zeEventHostReset" @h-zeEventHostReset [ev]))
  nil)

(defn read-graph-timestamps!
  "Read per-kernel device timestamps from a graph recorded with :profile? true, AFTER a
  replay-graph! has completed (the synchronous queue guarantees all events are signaled).
  Resets the events afterwards so the graph can be replayed again.

  Returns {:kernels [{:kernel-name str :phase kw|nil :ms double :context-ms double
                      :start-ticks long :end-ticks long} …]   ;; execution order
           :wall-ms double|nil    ;; device span first-kernel-start → last-kernel-end
           :ns-per-tick double}

  :ms is the GLOBAL (wall-clock domain) kernel duration; :context-ms counts only cycles while
  the kernel was actively executing (excludes preemption) — normally equal on this runtime's
  single in-order queue. Tick→ns uses device-timer-props; per-kernel wraparound is corrected
  with kernelTimestampValidBits, but :wall-ms is nil if the span itself wrapped (rare: ~223 s
  at 32 valid bits / 52 ns per tick — report honestly rather than alias)."
  [graph]
  (let [{:keys [events kernel-names phases]} graph]
    (when-not events
      (throw (ex-info "read-graph-timestamps!: graph was not recorded with :profile? true" {})))
    (let [{:keys [ns-per-tick kernel-timestamp-valid-bits]} (device-timer-props)
          ns-per-tick (double ns-per-tick)
          bits (long kernel-timestamp-valid-bits)
          wrap (if (< bits 64) (bit-shift-left 1 bits) 0)
          dur-ticks (fn ^long [^long s ^long e]
                      (if (>= e s) (- e s) (+ (- wrap s) e)))
          rows (let [tmp-arena (Arena/ofConfined)]
                 (try
                   (let [res (.allocate tmp-arena 32)]
                     (mapv (fn [i ev]
                             (ze-call! "zeEventQueryKernelTimestamp" @h-zeEventQueryKernelTimestamp
                                       [ev res])
                             (let [g-start (.get res I64 0)  g-end (.get res I64 8)
                                   c-start (.get res I64 16) c-end (.get res I64 24)]
                               {:kernel-name (nth kernel-names i)
                                :phase (nth phases i)
                                :ms (/ (* ns-per-tick (dur-ticks g-start g-end)) 1.0e6)
                                :context-ms (/ (* ns-per-tick (dur-ticks c-start c-end)) 1.0e6)
                                :start-ticks g-start
                                :end-ticks g-end}))
                           (range (count events)) events))
                   (finally (.close tmp-arena))))
          _ (doseq [ev events]
              (ze-call! "zeEventHostReset" @h-zeEventHostReset [ev]))
          span (when (seq rows)
                 (- (long (reduce max (map :end-ticks rows)))
                    (long (reduce min (map :start-ticks rows)))))]
      {:kernels rows
       :wall-ms (when (and span (>= span 0)) (/ (* ns-per-tick span) 1.0e6))
       :ns-per-tick ns-per-tick})))

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
  (let [{:keys [kernel-handle workgroup-size identity-val c-op dtype]
         :or {workgroup-size 256 identity-val 0.0 c-op "+" dtype :float}
         :as info} (ensure-kernel-loaded! kernel-name)
        combine (case c-op "fmax" (fn ^double [^double a ^double b] (Math/max a b))
                      "fmin" (fn ^double [^double a ^double b] (Math/min a b))
                      "*"    (fn ^double [^double a ^double b] (* a b))
                      (fn ^double [^double a ^double b] (+ a b)))
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
                 (double (combine acc (double (.get dev-partial value-layout
                                                    (* i dtype-size))))))
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

(defn- stage-operand!
  "Copy host array `arr` (double[]/float[]) of `nel` elements into cached shared memory for
  [kernel-name k], converting to f16 when `half?`. Returns the MemorySegment."
  ^MemorySegment [^String kernel-name k arr nel half? esize]
  (let [nel (long nel) esize (long esize)]
    (if half?
      (let [shorts (ensure-arr kernel-name (keyword (str (name k) "-sh")) nel)
            seg (ensure-seg kernel-name k (* nel 2))]
        (if (instance? (Class/forName "[F") arr)
          (let [^floats af arr] (dotimes [i nel] (aset shorts i (short (Float/floatToFloat16 (aget af i))))))
          (let [^doubles ad arr] (dotimes [i nel] (aset shorts i (short (Float/floatToFloat16 (float (aget ad i))))))))
        (MemorySegment/copy (MemorySegment/ofArray shorts) 0 seg 0 (* nel 2))
        seg)
      (let [seg (ensure-seg kernel-name k (* nel esize))]
        (MemorySegment/copy (MemorySegment/ofArray arr) 0 seg 0 (* nel esize))
        seg))))

(defn- readback-operand!
  "Copy `nel` elements from shared segment `seg` back into host array `out`, converting from
  f16 when `half?`."
  [^MemorySegment seg out nel half? esize]
  (let [nel (long nel) esize (long esize)]
    (if half?
      (if (instance? (Class/forName "[F") out)
        (let [^floats of out] (dotimes [i nel] (aset of i (Float/float16ToFloat (.get seg ValueLayout/JAVA_SHORT (long (* i 2)))))))
        (let [^doubles od out] (dotimes [i nel] (aset od i (double (Float/float16ToFloat (.get seg ValueLayout/JAVA_SHORT (long (* i 2)))))))))
      (MemorySegment/copy seg 0 (MemorySegment/ofArray out) 0 (* nel esize)))
    out))

(defn invoke-registered-contraction!
  "Launch a routed contraction by its registered executable artifact.

   `arguments` contains evaluated values in exact ABI order.  The ABI decides which values are
   buffers or scalars, how host buffers are staged, and the scalar type passed to Level Zero.
   Packed kernels therefore retain byte storage while declaring a distinct int32 kernel view.
   Output extent and 1-3D geometry are resolved from the artifact, never marker positions."
  [^String kernel-name arguments]
  (let [registered (get @kernel-registry kernel-name)
        _ (when-not registered
            (throw (ex-info (str "Kernel not registered: " kernel-name)
                            {:kernel-name kernel-name :registered (keys @kernel-registry)})))
        artifact (kart/validate! registered)
        abi (:abi artifact)
        typed-arguments (kexec/typed-runtime-arguments artifact arguments)
        call (kcall/make artifact typed-arguments)
        out-elems-expr (kart/attribute artifact :out-elems)
        out-elems (long (if (number? out-elems-expr)
                          out-elems-expr
                          (kcall/resolve-value call out-elems-expr)))
        {:keys [workgroup-size group-count]} (kcall/binding-plan call)
        input-index (volatile! -1)
        output (volatile! nil)
        output-seg (volatile! nil)
        all-args
        (mapv (fn [{:keys [kind dtype kernel-dtype] :as slot} value]
                (case kind
                  :input
                  (let [idx (vswap! input-index inc)
                        half? (boolean (#{:half :float16} dtype))
                        esize (long (get dtype-byte-sizes dtype 8))]
                    (if (device-buffer? value)
                      (:segment ^DeviceBuffer value)
                      (stage-operand! kernel-name (keyword (str "c-in-" idx))
                                      value (java.lang.reflect.Array/getLength value) half? esize)))

                  :output
                  (let [half? (boolean (#{:half :float16} dtype))
                        esize (long (get dtype-byte-sizes dtype 8))
                        seg (if (device-buffer? value)
                              (:segment ^DeviceBuffer value)
                              (ensure-seg kernel-name :c-out (* out-elems esize)))]
                    (vreset! output value)
                    (vreset! output-seg [seg half? esize])
                    seg)

                  ;; Read-write storage: the kernel reads the destination it overwrites (an
                  ;; accumulating contraction), so a host array is staged in and read back.
                  :inout
                  (let [half? (boolean (#{:half :float16} dtype))
                        esize (long (get dtype-byte-sizes dtype 8))
                        seg (if (device-buffer? value)
                              (:segment ^DeviceBuffer value)
                              (stage-operand! kernel-name :c-out value out-elems half? esize))]
                    (vreset! output value)
                    (vreset! output-seg [seg half? esize])
                    seg)

                  :scalar
                  value

                  (throw (ex-info "unsupported contraction ABI slot kind"
                                  {:kernel-name kernel-name :slot slot}))))
              abi typed-arguments)
        {:keys [kernel-handle]} (ensure-kernel-loaded! kernel-name)
        out @output
        [out-seg out-half? out-esize] @output-seg]
    (launch-geometry! kernel-handle workgroup-size group-count all-args)
    (when-not (device-buffer? out)
      (readback-operand! out-seg out out-elems out-half? out-esize))
    out))

(defn invoke-registered-contraction-dispatch!
  "Select one ABI-compatible contraction artifact from concrete scalar values, then use the
   ordinary artifact staging path. `default-kernel-name` is compiler-carried redundancy checked
   against the registered dispatch; it lets resident extraction inspect the common ABI without
   interpreting dispatch internals."
  [^String dispatch-id ^String default-kernel-name arguments]
  (let [dispatch (or (get @kernel-dispatch-registry dispatch-id)
                     (throw (ex-info "Kernel dispatch not registered"
                                     {:dispatch-id dispatch-id
                                      :registered (keys @kernel-dispatch-registry)})))
        dispatch (kdispatch/validate! dispatch)
        expected-default (:kernel-name (kdispatch/default-alternative dispatch))
        _ (when-not (= expected-default default-kernel-name)
            (throw (ex-info "contraction dispatch marker default differs from its registry"
                            {:dispatch-id dispatch-id
                             :expected expected-default
                             :actual default-kernel-name})))
        selected (kdispatch/select-alternative dispatch arguments)]
    (invoke-registered-contraction! (:kernel-name selected) arguments)))

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
        ;; Output size is the DESTINATION buffer length (n-dst*stride), NOT the
        ;; source pair count. For a JVM array the array length is authoritative;
        ;; the old (* n stride) sized it as n-pairs*stride and overran the output
        ;; whenever n-pairs > n-dst (the overlapping-index case). For a device
        ;; buffer copy-to ignores byte-count (it returns the segment directly),
        ;; so out-elems is unused there.
        out-elems (if (device-buffer? output) n (alength output))
        ;; GPU index buffer is int32. scatter-add/gather use (Array long) indices
        ;; on the CPU; narrow a long-array to int here so the byte copy is coherent
        ;; (copying n*4 bytes of an 8-byte-per-element long[] reinterprets pairs of
        ;; longs as ints and silently corrupts the indices).
        index (if (and (not (device-buffer? index))
                       (instance? (Class/forName "[J") index))
                (let [li ^longs index m (alength li) ia (int-array m)]
                  (dotimes [k m] (aset ia k (int (aget li k))))
                  ia)
                index)
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
    (clojure.core/reset! kernel-dispatch-registry {})
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
