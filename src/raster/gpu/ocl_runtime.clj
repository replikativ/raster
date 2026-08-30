(ns raster.gpu.ocl-runtime
  "OpenCL ICD runtime bindings via Panama FFM (Java 21+).

  Portable GPU runtime that works across NVIDIA, AMD, and Intel GPUs
  through the OpenCL Installable Client Driver (ICD) loader.
  Compiles kernels from OpenCL C source directly (no SPIR-V step needed).

  Mirrors the raster.gpu.ze-runtime API surface so raster.gpu.core can dispatch
  between Level Zero and OpenCL backends transparently.

  Usage:
    (ocl/init!)
    (ocl/query-devices)
    (ocl/register-kernel! \"my_kernel\" {:source \"...\" :kernel-name \"my_kernel\" ...})
    (ocl/invoke-registered-map-void-kernel \"my_kernel\" bufs scalars n)"
  (:refer-clojure :exclude [reset!])
  (:import [java.lang.foreign
            Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout
            AddressLayout]
           [java.lang.invoke MethodHandle])
  (:require [raster.compiler.core.dtype :as dt]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.resident-value :as resident-value]))

;; ================================================================
;; Library loading
;; ================================================================

(def ^:private ocl-lib-paths
  "Search paths for OpenCL ICD loader library."
  ["/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"
   "/usr/lib/x86_64-linux-gnu/libOpenCL.so"
   "/usr/lib64/libOpenCL.so.1"
   "/usr/lib64/libOpenCL.so"
   "/usr/lib/libOpenCL.so.1"
   "/usr/lib/libOpenCL.so"
   "/opt/intel/oneapi/lib/libOpenCL.so"
   ;; macOS
   "/System/Library/Frameworks/OpenCL.framework/OpenCL"])

(defn- find-ocl-lib ^SymbolLookup []
  (let [loader (SymbolLookup/loaderLookup)]
    (if (.isPresent (.find loader "clGetPlatformIDs"))
      loader
      (or (some (fn [path]
                  (try
                    (let [lib (SymbolLookup/libraryLookup path (Arena/global))]
                      (when (.isPresent (.find lib "clGetPlatformIDs"))
                        lib))
                    (catch Exception _ nil)))
                ocl-lib-paths)
          (throw (ex-info "OpenCL ICD loader (libOpenCL.so) not found"
                          {:searched ocl-lib-paths}))))))

(def ^:private ocl-lib (delay (find-ocl-lib)))

;; ================================================================
;; Method handle creation
;; ================================================================

(defn- lookup-symbol ^MemorySegment [^String sym-name]
  (let [opt (.find ^SymbolLookup @ocl-lib sym-name)]
    (when-not (.isPresent opt)
      (throw (ex-info (str "OpenCL symbol not found: " sym-name)
                      {:symbol sym-name})))
    (.get opt)))

(defn- fd
  "Create a FunctionDescriptor with return type and arg types."
  ^FunctionDescriptor [ret & args]
  (FunctionDescriptor/of ret (into-array MemoryLayout args)))

(defn- make-handle
  ^MethodHandle [^String symbol-name ^FunctionDescriptor fd]
  (.downcallHandle (Linker/nativeLinker)
                   (lookup-symbol symbol-name)
                   fd
                   (into-array Linker$Option [])))

;; ================================================================
;; OpenCL constants
;; ================================================================

(def ^:private CL_SUCCESS 0)
(def ^:private CL_DEVICE_NOT_FOUND -1)
(def ^:private CL_PLATFORM_NOT_FOUND_KHR -1001)
(def ^:private CL_DEVICE_TYPE_GPU (long 4))
(def ^:private CL_DEVICE_TYPE_CPU (long 2))
(def ^:private CL_DEVICE_TYPE_ALL (long 0xFFFFFFFF))

(defn- requested-device-type
  "CL device type from RASTER_OCL_DEVICE_TYPE (cpu|all; default gpu) — lets the
  same runtime bind POCL/Intel-CPU OpenCL for vendor-portability validation."
  ^long []
  (case (some-> (System/getenv "RASTER_OCL_DEVICE_TYPE") .toLowerCase)
    "cpu" CL_DEVICE_TYPE_CPU
    "all" CL_DEVICE_TYPE_ALL
    CL_DEVICE_TYPE_GPU))
(def ^:private CL_MEM_READ_WRITE (long 1))
(def ^:private CL_MEM_COPY_HOST_PTR (long 32))
(def ^:private CL_FALSE (int 0))
(def ^:private CL_TRUE (int 1))
(def ^:private CL_COMPLETE 0)
(def ^:private CL_QUEUE_PROFILING_ENABLE (long 2))
(def ^:private CL_EVENT_COMMAND_EXECUTION_STATUS 0x11D3)
(def ^:private CL_PROFILING_COMMAND_START 0x1282)
(def ^:private CL_PROFILING_COMMAND_END 0x1283)

;; clGetDeviceInfo param_name constants
(def ^:private CL_DEVICE_MAX_COMPUTE_UNITS 0x1002)
(def ^:private CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004)
(def ^:private CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C)
(def ^:private CL_DEVICE_GLOBAL_MEM_SIZE 0x101F)
(def ^:private CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035)
(def ^:private CL_DEVICE_MEM_BASE_ADDR_ALIGN 0x1019)
(def ^:private CL_DEVICE_NAME 0x102B)
(def ^:private CL_DEVICE_VENDOR 0x102C)
(def ^:private CL_DEVICE_EXTENSIONS 0x1030)
(def ^:private CL_DEVICE_VERSION 0x102F)

;; clGetProgramBuildInfo param_name
(def ^:private CL_PROGRAM_BUILD_LOG 0x1183)

;; cl_buffer_create_type
(def ^:private CL_BUFFER_CREATE_TYPE_REGION 0x1220)

;; ================================================================
;; Layout helpers
;; ================================================================

(def ^:private PTR ValueLayout/ADDRESS)
(def ^:private I32 ValueLayout/JAVA_INT)
(def ^:private I64 ValueLayout/JAVA_LONG)
(def ^:private F32 ValueLayout/JAVA_FLOAT)
(def ^:private F64 ValueLayout/JAVA_DOUBLE)

(defn- read-int
  "Read an int from a segment at offset 0."
  ^long [^MemorySegment seg]
  (long (.get seg I32 0)))

(def ^:private dtype-byte-sizes
  {:float 4, :double 8, :int 4, :long 8, :float16 2, :half 2, :byte 1, :int8 1})

;; ================================================================
;; Method handles (lazy, created on first use)
;; ================================================================

;; Platform/Device
(def ^:private h-clGetPlatformIDs
  (delay (make-handle "clGetPlatformIDs" (fd I32 I32 PTR PTR))))
(def ^:private h-clGetDeviceIDs
  (delay (make-handle "clGetDeviceIDs" (fd I32 PTR I64 I32 PTR PTR))))
(def ^:private h-clGetDeviceInfo
  (delay (make-handle "clGetDeviceInfo" (fd I32 PTR I32 I64 PTR PTR))))
;; Context/Queue
(def ^:private h-clCreateContext
  (delay (make-handle "clCreateContext" (fd PTR PTR I32 PTR PTR PTR PTR))))
(def ^:private h-clCreateCommandQueue
  (delay (make-handle "clCreateCommandQueue" (fd PTR PTR PTR I64 PTR))))

;; Program/Kernel
(def ^:private h-clCreateProgramWithSource
  (delay (make-handle "clCreateProgramWithSource" (fd PTR PTR I32 PTR PTR PTR))))
(def ^:private h-clBuildProgram
  (delay (make-handle "clBuildProgram" (fd I32 PTR I32 PTR PTR PTR PTR))))
(def ^:private h-clGetProgramBuildInfo
  (delay (make-handle "clGetProgramBuildInfo" (fd I32 PTR PTR I32 I64 PTR PTR))))
(def ^:private h-clCreateKernel
  (delay (make-handle "clCreateKernel" (fd PTR PTR PTR PTR))))
(def ^:private h-clSetKernelArg
  (delay (make-handle "clSetKernelArg" (fd I32 PTR I32 I64 PTR))))

;; Memory
(def ^:private h-clCreateBuffer
  (delay (make-handle "clCreateBuffer" (fd PTR PTR I64 I64 PTR PTR))))
(def ^:private h-clCreateSubBuffer
  (delay (make-handle "clCreateSubBuffer" (fd PTR PTR I64 I32 PTR PTR))))
(def ^:private h-clEnqueueWriteBuffer
  (delay (make-handle "clEnqueueWriteBuffer" (fd I32 PTR PTR I32 I64 I64 PTR I32 PTR PTR))))
(def ^:private h-clEnqueueReadBuffer
  (delay (make-handle "clEnqueueReadBuffer" (fd I32 PTR PTR I32 I64 I64 PTR I32 PTR PTR))))
(def ^:private h-clEnqueueCopyBuffer
  (delay (make-handle "clEnqueueCopyBuffer" (fd I32 PTR PTR PTR I64 I64 I64 I32 PTR PTR))))
(def ^:private h-clEnqueueFillBuffer
  (delay (make-handle "clEnqueueFillBuffer" (fd I32 PTR PTR PTR I64 I64 I64 I32 PTR PTR))))
(def ^:private h-clReleaseMemObject
  (delay (make-handle "clReleaseMemObject" (fd I32 PTR))))

;; Execution
(def ^:private h-clEnqueueNDRangeKernel
  (delay (make-handle "clEnqueueNDRangeKernel" (fd I32 PTR PTR I32 PTR PTR PTR I32 PTR PTR))))
(def ^:private h-clFinish
  (delay (make-handle "clFinish" (fd I32 PTR))))
(def ^:private h-clFlush
  (delay (make-handle "clFlush" (fd I32 PTR))))
(def ^:private h-clWaitForEvents
  (delay (make-handle "clWaitForEvents" (fd I32 I32 PTR))))
(def ^:private h-clGetEventInfo
  (delay (make-handle "clGetEventInfo" (fd I32 PTR I32 I64 PTR PTR))))
(def ^:private h-clGetEventProfilingInfo
  (delay (make-handle "clGetEventProfilingInfo" (fd I32 PTR I32 I64 PTR PTR))))
(def ^:private h-clReleaseEvent
  (delay (make-handle "clReleaseEvent" (fd I32 PTR))))

;; Cleanup
(def ^:private h-clReleaseKernel
  (delay (make-handle "clReleaseKernel" (fd I32 PTR))))
(def ^:private h-clReleaseProgram
  (delay (make-handle "clReleaseProgram" (fd I32 PTR))))
(def ^:private h-clReleaseCommandQueue
  (delay (make-handle "clReleaseCommandQueue" (fd I32 PTR))))
(def ^:private h-clReleaseContext
  (delay (make-handle "clReleaseContext" (fd I32 PTR))))

;; ================================================================
;; Call helpers
;; ================================================================

(defn- cl-call!
  "Invoke an OpenCL function and check for errors."
  [^String name ^MethodHandle handle args]
  (let [ret (int (.invokeWithArguments handle (into-array Object args)))]
    (when (not= CL_SUCCESS ret)
      (throw (ex-info (str "OpenCL " name " failed with error " ret)
                      {:function name :error ret})))
    ret))

(defn- check-cl-result!
  [^String name ret]
  (when (not= CL_SUCCESS ret)
    (throw (ex-info (str "OpenCL " name " failed with error " ret)
                    {:function name :error ret})))
  ret)

;; ================================================================
;; State management
;; ================================================================

(defonce ^:private state
  (atom {:initialized? false
         :platform nil       ;; MemorySegment (cl_platform_id)
         :device nil         ;; MemorySegment (cl_device_id)
         :context nil        ;; MemorySegment (cl_context)
         :queue nil          ;; MemorySegment (cl_command_queue)
         :arena nil          ;; Arena for long-lived allocations
         :device-name nil
         :device-info nil
         :unified-memory? false
         :buffer-offset-alignment nil ;; bytes; CL_DEVICE_MEM_BASE_ADDR_ALIGN is reported in bits
         :programs {}        ;; source-hash -> cl_program handle
         :kernels {}}))      ;; [program kernel-name] -> cl_kernel handle

(def kernel-registry
  "Global registry mapping kernel-name → kernel info.
  Same structure as raster.gpu.ze-runtime/kernel-registry."
  (atom {}))

(def kernel-dispatch-registry
  "Pure compiler dispatch values keyed independently of single-entry native kernels."
  (atom {}))

(def ^:dynamic *current-arena*
  "When set, newly registered kernels are tagged with this arena-id."
  nil)

;; ================================================================
;; Device query helpers
;; ================================================================

(defn- query-device-info-string
  "Query a string device info parameter."
  ^String [^MemorySegment device ^long param-name]
  (let [arena (Arena/ofConfined)
        size-ret (.allocate arena I64)]
    (try
      (cl-call! "clGetDeviceInfo"
                @h-clGetDeviceInfo
                [device (int param-name) (long 0) MemorySegment/NULL size-ret])
      (let [size (long (.get size-ret I64 0))
            buf (.allocate arena size)]
        (cl-call! "clGetDeviceInfo"
                  @h-clGetDeviceInfo
                  [device (int param-name) size buf size-ret])
        (let [s (.getString buf 0)]
          (.trim s)))
      (finally (.close arena)))))

(defn- query-device-info-uint
  "Query a cl_uint device info parameter."
  ^long [^MemorySegment device ^long param-name]
  (let [arena (Arena/ofConfined)
        buf (.allocate arena I32)]
    (try
      (cl-call! "clGetDeviceInfo"
                @h-clGetDeviceInfo
                [device (int param-name) (long 4) buf MemorySegment/NULL])
      (long (.get buf I32 0))
      (finally (.close arena)))))

(defn- query-device-info-ulong
  "Query a cl_ulong device info parameter."
  ^long [^MemorySegment device ^long param-name]
  (let [arena (Arena/ofConfined)
        buf (.allocate arena I64)]
    (try
      (cl-call! "clGetDeviceInfo"
                @h-clGetDeviceInfo
                [device (int param-name) (long 8) buf MemorySegment/NULL])
      (long (.get buf I64 0))
      (finally (.close arena)))))

(defn- query-device-info-size-t
  "Query a size_t device info parameter."
  ^long [^MemorySegment device ^long param-name]
  ;; size_t is 8 bytes on 64-bit
  (query-device-info-ulong device param-name))

(defn- device-buffer-offset-alignment
  "Return the device's legal clCreateSubBuffer origin alignment in bytes. OpenCL reports this
   capability in bits and requires it to be a power of two."
  [device]
  (let [bits (query-device-info-uint device CL_DEVICE_MEM_BASE_ADDR_ALIGN)]
    (when-not (and (pos? bits)
                   (zero? (bit-and bits (dec bits)))
                   (zero? (mod bits 8)))
      (throw (ex-info "OpenCL reported an invalid buffer base-address alignment"
                      {:alignment-bits bits})))
    (quot bits 8)))

(defn- device-info
  [device]
  {:name (query-device-info-string device CL_DEVICE_NAME)
   :vendor (query-device-info-string device CL_DEVICE_VENDOR)
   :version (query-device-info-string device CL_DEVICE_VERSION)
   :max-compute-units (query-device-info-uint device CL_DEVICE_MAX_COMPUTE_UNITS)
   :max-work-group-size (query-device-info-size-t device CL_DEVICE_MAX_WORK_GROUP_SIZE)
   :max-clock-mhz (query-device-info-uint device CL_DEVICE_MAX_CLOCK_FREQUENCY)
   :global-mem-bytes (query-device-info-ulong device CL_DEVICE_GLOBAL_MEM_SIZE)
   :buffer-offset-alignment (device-buffer-offset-alignment device)
   :extensions (query-device-info-string device CL_DEVICE_EXTENSIONS)
   :integrated? (try
                  (== 1 (query-device-info-uint device CL_DEVICE_HOST_UNIFIED_MEMORY))
                  (catch Exception _ false))})

;; ================================================================
;; Initialization
;; ================================================================

(declare init!)

(defn- ensure-init! []
  (when-not (:initialized? @state) (init!)))

(defn init!
  "Initialize OpenCL runtime. Idempotent.
  Finds the first platform with a device of the requested type (default GPU;
  RASTER_OCL_DEVICE_TYPE=cpu|all selects CPU/any device — POCL, Intel CPU
  runtime, vendor-portability testing), creates context and in-order queue."
  []
  (when-not (:initialized? @state)
    (let [arena (Arena/ofShared)
          ;; Get platforms
          num-plat-seg (.allocate arena I32)
          _ (cl-call! "clGetPlatformIDs" @h-clGetPlatformIDs
                      [(int 0) MemorySegment/NULL num-plat-seg])
          num-plat (read-int num-plat-seg)
          _ (when (zero? num-plat)
              (throw (ex-info "No OpenCL platforms found" {})))
          plat-buf (.allocate arena (* num-plat 8))
          _ (cl-call! "clGetPlatformIDs" @h-clGetPlatformIDs
                      [(int num-plat) plat-buf num-plat-seg])

          ;; Find first platform with a GPU device
          [platform device]
          (or (some
               (fn [plat-idx]
                 (let [plat (.get plat-buf PTR (* plat-idx 8))
                       num-dev-seg (.allocate arena I32)
                       ret (int (.invokeWithArguments ^MethodHandle @h-clGetDeviceIDs
                                                      (into-array Object [plat (long (requested-device-type))
                                                                          (int 0) MemorySegment/NULL num-dev-seg])))]
                   (cond
                     (= CL_DEVICE_NOT_FOUND ret) nil
                     (not= CL_SUCCESS ret) (check-cl-result! "clGetDeviceIDs" ret)
                     :else
                     (let [num-dev (read-int num-dev-seg)]
                       (when (> num-dev 0)
                         (let [dev-buf (.allocate arena 8)
                               _ (cl-call! "clGetDeviceIDs" @h-clGetDeviceIDs
                                           [plat (long (requested-device-type)) (int 1) dev-buf num-dev-seg])
                               dev (.get dev-buf PTR 0)]
                           [plat dev]))))))
               (range num-plat))
              (throw (ex-info "No OpenCL GPU devices found" {:num-platforms num-plat})))

          ;; Complete all fallible capability queries before allocating a context/queue.  A broken
          ;; clGetDeviceInfo must not leave native execution resources unreachable.
          device-info (device-info device)
          dev-name (:name device-info)
          unified? (:integrated? device-info)
          buffer-offset-alignment (:buffer-offset-alignment device-info)

          ;; Create context
          err-seg (.allocate arena I32)
          ctx (.invokeWithArguments ^MethodHandle @h-clCreateContext
                                    (into-array Object [MemorySegment/NULL (int 1)
                                                        (.allocateFrom arena PTR device)
                                                        MemorySegment/NULL MemorySegment/NULL err-seg]))
          _ (when (not= CL_SUCCESS (read-int err-seg))
              (throw (ex-info "clCreateContext failed" {:error (read-int err-seg)})))

          ;; One physical in-order queue initially realizes the backend-neutral compute and
          ;; transfer queue classes. Profiling is enabled so transfer GPUEvents have device
          ;; timestamps without introducing unsynchronized cross-queue execution.
          queue (.invokeWithArguments ^MethodHandle @h-clCreateCommandQueue
                                      (into-array Object
                                                  [ctx device CL_QUEUE_PROFILING_ENABLE err-seg]))
          _ (when (not= CL_SUCCESS (read-int err-seg))
              (throw (ex-info "clCreateCommandQueue failed" {:error (read-int err-seg)})))]

      (swap! state assoc
             :initialized? true
             :platform platform
             :device device
             :context ctx
             :queue queue
             :arena arena
             :device-name dev-name
             :device-info device-info
             :unified-memory? unified?
             :buffer-offset-alignment buffer-offset-alignment)
      (println (str "[ocl-runtime] Initialized: " dev-name
                    (when unified? " (unified memory)"))))))

(defn query-devices
  "Query all requested OpenCL devices across all platforms.

   A clean CL_PLATFORM_NOT_FOUND_KHR or per-platform CL_DEVICE_NOT_FOUND is represented by an
   empty vector. Loader, FFM, driver, and device-info failures propagate: callers must not confuse
   a broken runtime with an absent device."
  []
  @ocl-lib
  (let [arena (Arena/ofConfined)]
    (try
      (let [num-plat-seg (.allocate arena I32)
            platform-ret
            (int (.invokeWithArguments ^MethodHandle @h-clGetPlatformIDs
                                       (into-array Object [(int 0) MemorySegment/NULL num-plat-seg])))]
        (if (= CL_PLATFORM_NOT_FOUND_KHR platform-ret)
          []
          (let [_ (check-cl-result! "clGetPlatformIDs" platform-ret)
                num-plat (read-int num-plat-seg)]
            (if (zero? num-plat)
              []
              (let [plat-buf (.allocate arena (* num-plat 8))
                    _ (cl-call! "clGetPlatformIDs" @h-clGetPlatformIDs
                                [(int num-plat) plat-buf num-plat-seg])]
                (vec
                 (mapcat
                  (fn [plat-idx]
                    (let [plat (.get plat-buf PTR (* plat-idx 8))
                          num-dev-seg (.allocate arena I32)
                          ret (int (.invokeWithArguments ^MethodHandle @h-clGetDeviceIDs
                                                         (into-array
                                                          Object
                                                          [plat (long (requested-device-type))
                                                           (int 0) MemorySegment/NULL num-dev-seg])))]
                      (cond
                        (= CL_DEVICE_NOT_FOUND ret) []
                        (not= CL_SUCCESS ret) (check-cl-result! "clGetDeviceIDs" ret)
                        :else
                        (let [num-dev (read-int num-dev-seg)]
                          (if (zero? num-dev)
                            []
                            (let [dev-buf (.allocate arena (* num-dev 8))
                                  _ (cl-call! "clGetDeviceIDs" @h-clGetDeviceIDs
                                              [plat (long (requested-device-type))
                                               (int num-dev) dev-buf num-dev-seg])]
                              (mapv (fn [dev-idx]
                                      (device-info (.get dev-buf PTR (* dev-idx 8))))
                                    (range num-dev))))))))
                  (range num-plat))))))))
      (finally (.close arena)))))

(defn selected-device-info
  "Return the exact OpenCL device bound by this singleton runtime.

   Capability gates must use this value rather than searching `query-devices`: the latter may
   contain devices with capabilities that the initialized context and queue do not have."
  []
  (ensure-init!)
  (:device-info @state))

;; ================================================================
;; DeviceBuffer (OpenCL cl_mem backed)
;; ================================================================

(defrecord OclBuffer [^MemorySegment segment    ;; host staging MemorySegment
                      ^MemorySegment cl-mem      ;; cl_mem handle
                      ^long n-elements
                      ^long byte-size
                      dtype
                      ^long alignment])          ;; legal kernel-view origin alignment in bytes

(defn device-buffer? [x]
  (instance? OclBuffer x))

(defn- physical-pointer-dtypes [arrays]
  (mapv #(cond
           (device-buffer? %) (:dtype ^OclBuffer %)
           (instance? MemorySegment %) :opaque
           :else (dt/dtype-for-jvm-array %))
        arrays))

(defn expand-pointer-binding
  "Expand one logical artifact pointer binding into OpenCL physical resident values. Driver state
   is not touched; composite validation is shared with the Level Zero backend."
  [{:keys [binding slots] :as group} value]
  (if (resident-value/resident-composite? value)
    (resident-value/expand group value)
    (do
      (when-not (= 1 (count slots))
        (throw (ex-info "multi-slot logical pointer requires a resident composite value"
                        {:binding binding :slots slots :value-type (type value)})))
      (when-not (or (device-buffer? value) (instance? MemorySegment value))
        (throw (ex-info "OpenCL logical pointer requires OclBuffer/MemorySegment"
                        {:binding binding :slot (first slots) :value-type (type value)})))
      [value])))

(defn- kernel-info-value
  "Read a compiler-owned emitter attribute from an artifact or a remaining specialized entry."
  [kernel-info k default]
  (if (kart/kernel-artifact? kernel-info)
    (get (:attributes kernel-info) k default)
    (get kernel-info k default)))

(declare buffer-offset-alignment)

(defn make-buffer
  "Allocate a persistent GPU buffer via clCreateBuffer."
  ([n] (make-buffer n :float))
  ([n dtype]
   (ensure-init!)
   (let [{:keys [context arena]} @state
         elem-size (long (get dtype-byte-sizes dtype 4))
         byte-size (long (* n elem-size))
         err-seg (.allocate ^Arena arena I32)
         cl-mem (.invokeWithArguments ^MethodHandle @h-clCreateBuffer
                                      (into-array Object [context (long CL_MEM_READ_WRITE)
                                                          (long byte-size) MemorySegment/NULL err-seg]))
         _ (when (not= CL_SUCCESS (read-int err-seg))
             (throw (ex-info "clCreateBuffer failed" {:error (read-int err-seg) :size byte-size})))
         ;; Host staging buffer for data transfer
         host-seg (.allocate ^Arena arena byte-size)]
     (->OclBuffer host-seg cl-mem (long n) byte-size dtype
                  (long (buffer-offset-alignment))))))

(defn buffer-offset-alignment
  "Minimum byte alignment for a clCreateSubBuffer origin on the selected OpenCL device."
  []
  (ensure-init!)
  (:buffer-offset-alignment @state))

(defn slice-buffer
  "Create an independently reference-counted cl_mem sub-buffer over an exact byte range.

   Unlike Level Zero's non-owning sliced pointer, the returned OclBuffer owns one native
   cl_mem reference and MUST be passed to free-buffer!. `byte-offset` is relative to the root
   buffer and must satisfy CL_DEVICE_MEM_BASE_ADDR_ALIGN; dtype reinterpretation is forbidden."
  [^OclBuffer buf byte-offset byte-length dtype]
  (ensure-init!)
  (let [byte-offset (long byte-offset)
        byte-length (long byte-length)
        dtype (dt/canon dtype)
        raw-dtype (dt/canon (:dtype buf))
        element-bytes (long (dt/bytes-of dtype))
        alignment (long (buffer-offset-alignment))]
    (when-not (= raw-dtype dtype)
      (throw (ex-info "OpenCL sub-buffer cannot reinterpret its parent dtype"
                      {:buffer-dtype raw-dtype :dtype dtype})))
    (when (or (neg? byte-offset) (not (pos? byte-length))
              (> (+ byte-offset byte-length) (:byte-size buf))
              (not (zero? (mod byte-offset element-bytes)))
              (not (zero? (mod byte-length element-bytes))))
      (throw (ex-info "OpenCL sub-buffer range is invalid"
                      {:byte-offset byte-offset :byte-length byte-length
                       :buffer-bytes (:byte-size buf) :dtype dtype})))
    (when-not (zero? (mod byte-offset alignment))
      (throw (ex-info "OpenCL sub-buffer origin violates the device alignment requirement"
                      {:byte-offset byte-offset :required-alignment alignment})))
    (let [arena (Arena/ofConfined)]
      (try
        (let [region (.allocate arena (long 16) (long 8))
              err-seg (.allocate arena I32)
              _ (.set region I64 0 byte-offset)
              _ (.set region I64 8 byte-length)
              cl-mem (.invokeWithArguments
                      ^MethodHandle @h-clCreateSubBuffer
                      (into-array Object [(:cl-mem buf) (long CL_MEM_READ_WRITE)
                                          (int CL_BUFFER_CREATE_TYPE_REGION) region err-seg]))
              error (read-int err-seg)]
          (when-not (= CL_SUCCESS error)
            (throw (ex-info "clCreateSubBuffer failed"
                            {:error error :byte-offset byte-offset :byte-length byte-length
                             :required-alignment alignment})))
          (->OclBuffer (.asSlice ^MemorySegment (:segment buf) byte-offset byte-length)
                       cl-mem (quot byte-length element-bytes) byte-length dtype alignment))
        (finally
          (.close arena))))))

(defn free-buffer!
  "Free an OclBuffer's cl_mem."
  [^OclBuffer buf]
  (cl-call! "clReleaseMemObject" @h-clReleaseMemObject [(:cl-mem buf)]))

(defn buffer-as-float-buffer
  "Return a java.nio.FloatBuffer view over the host staging segment."
  [^OclBuffer buf]
  (.asSlice (:segment buf) 0 (:byte-size buf))
  (-> (.asByteBuffer (:segment buf))
      (.order (java.nio.ByteOrder/nativeOrder))
      (.asFloatBuffer)))

(defn buffer-as-int-buffer
  "Return a java.nio.IntBuffer view over the host staging segment."
  [^OclBuffer buf]
  (-> (.asByteBuffer (:segment buf))
      (.order (java.nio.ByteOrder/nativeOrder))
      (.asIntBuffer)))

(defn array->buffer!
  "Copy a JVM array into an OclBuffer (host → device). Returns the buffer."
  [^OclBuffer buf arr]
  (ensure-init!)
  (let [{:keys [queue]} @state
        src-seg (MemorySegment/ofArray arr)
        byte-size (:byte-size buf)]
    ;; Copy to host staging
    (MemorySegment/copy src-seg 0 (:segment buf) 0 byte-size)
    ;; Upload to device
    (cl-call! "clEnqueueWriteBuffer" @h-clEnqueueWriteBuffer
              [queue (:cl-mem buf) (int CL_TRUE) (long 0) (long byte-size)
               (:segment buf) (int 0) MemorySegment/NULL MemorySegment/NULL])
    buf))

(defn- as-segment
  ^MemorySegment [x]
  (if (instance? MemorySegment x) x (MemorySegment/ofArray x)))

(defn- check-range!
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
  "Validate one ranged transfer and return its byte-level plan without copying. Same contract as
   the ze runtime's `plan-range`; split from execution so a batch validates everything first."
  [^OclBuffer buf host {:keys [src-element dst-element elements]
                        :or {src-element 0 dst-element 0}} direction]
  (let [es (long (get dtype-byte-sizes (:dtype buf) 4))
        host-seg (as-segment host)
        [buf-el host-el] (case direction :upload [dst-element src-element] :download [src-element dst-element])
        host-off (* (long host-el) es)]
    (check-range! (name direction) (:n-elements buf) buf-el elements es
                  (- (.byteSize host-seg) host-off))
    {:buf-off (* (long buf-el) es) :host-off host-off :n-bytes (* (long elements) es)
     :host-seg host-seg}))

(defn execute-range!
  "Perform a validated transfer. OpenCL buffers are not host-coherent, so the range goes through
   the host staging segment and a clEnqueueWrite/ReadBuffer at the byte offset — only the RANGE
   crosses the bus."
  [^OclBuffer buf {:keys [buf-off host-off n-bytes host-seg]} direction]
  (ensure-init!)
  (let [{:keys [queue]} @state
        staging (.asSlice ^MemorySegment (:segment buf) (long buf-off))]
    (case direction
      :upload
      (do (MemorySegment/copy ^MemorySegment host-seg (long host-off) (:segment buf) (long buf-off) (long n-bytes))
          (cl-call! "clEnqueueWriteBuffer" @h-clEnqueueWriteBuffer
                    [queue (:cl-mem buf) (int CL_TRUE) (long buf-off) (long n-bytes)
                     staging (int 0) MemorySegment/NULL MemorySegment/NULL]))
      :download
      (do (cl-call! "clEnqueueReadBuffer" @h-clEnqueueReadBuffer
                    [queue (:cl-mem buf) (int CL_TRUE) (long buf-off) (long n-bytes)
                     staging (int 0) MemorySegment/NULL MemorySegment/NULL])
          (MemorySegment/copy (:segment buf) (long buf-off) ^MemorySegment host-seg (long host-off) (long n-bytes))))))

(defn upload-range!
  "Ranged host → device copy; returns the buffer."
  [^OclBuffer buf src spec]
  (execute-range! buf (plan-range buf src spec :upload) :upload)
  buf)

(defn download-range!
  "Ranged device → host copy; returns `dst`."
  [^OclBuffer buf dst spec]
  (execute-range! buf (plan-range buf dst spec :download) :download)
  dst)

(defn copy-buffer-range!
  "Synchronously enqueue a device-resident OpenCL buffer range copy."
  [^OclBuffer src ^OclBuffer dst src-element dst-element elements]
  (when-not (= (:dtype src) (:dtype dst))
    (throw (ex-info "OpenCL resident copy requires matching dtypes"
                    {:source-dtype (:dtype src) :destination-dtype (:dtype dst)})))
  (let [element-bytes (long (get dtype-byte-sizes (:dtype src) 4))
        src-byte (* (long src-element) element-bytes)
        dst-byte (* (long dst-element) element-bytes)
        byte-count (* (long elements) element-bytes)
        queue (:queue @state)]
    (cl-call! "clEnqueueCopyBuffer" @h-clEnqueueCopyBuffer
              [queue (:cl-mem src) (:cl-mem dst)
               src-byte dst-byte byte-count
               (int 0) MemorySegment/NULL MemorySegment/NULL])
    (cl-call! "clFinish" @h-clFinish [queue])
    dst))

(defn buffer->array
  "Copy an OclBuffer's contents to a new JVM array (device → host)."
  [^OclBuffer buf]
  (ensure-init!)
  (let [{:keys [queue]} @state
        byte-size (:byte-size buf)
        n (:n-elements buf)]
    ;; Download from device
    (cl-call! "clEnqueueReadBuffer" @h-clEnqueueReadBuffer
              [queue (:cl-mem buf) (int CL_TRUE) (long 0) (long byte-size)
               (:segment buf) (int 0) MemorySegment/NULL MemorySegment/NULL])
    ;; Copy to JVM array
    (case (:dtype buf)
      :float (let [arr (float-array n)]
               (MemorySegment/copy (:segment buf) ValueLayout/JAVA_FLOAT 0
                                   arr 0 (int n))
               arr)
      :int   (let [arr (int-array n)]
               (MemorySegment/copy (:segment buf) ValueLayout/JAVA_INT 0
                                   arr 0 (int n))
               arr)
      :long  (let [arr (long-array n)]
               (MemorySegment/copy (:segment buf) ValueLayout/JAVA_LONG 0
                                   arr 0 (int n))
               arr)
      :double (let [arr (double-array n)]
                (MemorySegment/copy (:segment buf) ValueLayout/JAVA_DOUBLE 0
                                    arr 0 (int n))
                arr)
      (:float16 :half) (let [arr (short-array n)]
                         (MemorySegment/copy (:segment buf) ValueLayout/JAVA_SHORT 0
                                             arr 0 (int n))
                         arr)
      (:byte :int8) (let [arr (byte-array n)]
                      (MemorySegment/copy (:segment buf) ValueLayout/JAVA_BYTE 0
                                          arr 0 (int n))
                      arr))))

(defn buffer-of-array
  "Create a new OclBuffer from a JVM array (allocates + copies)."
  ([arr] (buffer-of-array arr nil))
  ([arr dtype]
   (let [dt (or dtype
                (cond (instance? (Class/forName "[F") arr) :float
                      (instance? (Class/forName "[I") arr) :int
                      (instance? (Class/forName "[J") arr) :long
                      (instance? (Class/forName "[D") arr) :double
                      :else :float))
         n (java.lang.reflect.Array/getLength arr)
         buf (make-buffer n dt)]
     (array->buffer! buf arr)
     buf)))

(defn zero-buffer!
  "Zero out an OclBuffer. Returns the buffer."
  [^OclBuffer buf]
  (ensure-init!)
  (let [{:keys [queue arena]} @state
        zero-pattern (.allocate ^Arena arena 4)]
    (.set zero-pattern I32 0 (int 0))
    (cl-call! "clEnqueueFillBuffer" @h-clEnqueueFillBuffer
              [queue (:cl-mem buf) zero-pattern (long 4) (long 0) (long (:byte-size buf))
               (int 0) MemorySegment/NULL MemorySegment/NULL])
    (cl-call! "clFinish" @h-clFinish [queue])
    buf))

;; ================================================================
;; Arena management
;; ================================================================

(defn make-kernel-arena!
  "Create a new kernel arena. Returns a unique arena-id keyword."
  []
  (keyword (str "arena-" (gensym ""))))

(defn close-kernel-arena!
  "Free all kernels registered under arena-id from kernel-registry."
  [arena-id]
  (let [reg @kernel-registry
        arena-kernels (filter (fn [[_ info]] (= (:arena-id info) arena-id)) reg)]
    ;; Release OpenCL kernel/program handles
    (doseq [[_ info] arena-kernels]
      (when-let [kh (:kernel-handle info)]
        (try (.invokeWithArguments ^MethodHandle @h-clReleaseKernel
                                   (into-array Object [kh]))
             (catch Exception _)))
      (when-let [prog (:program info)]
        (try (.invokeWithArguments ^MethodHandle @h-clReleaseProgram
                                   (into-array Object [prog]))
             (catch Exception _)))
      ;; Free cached host segments
      (doseq [[k v] info]
        (when (and (instance? MemorySegment v)
                   (not= k :kernel-handle)
                   (not= k :program)
                   (not= k :cl-mem))
          ;; Host segments are arena-allocated, no explicit free needed
          nil)))
    ;; Remove from registry
    (swap! kernel-registry #(reduce dissoc % (map first arena-kernels)))
    (swap! kernel-dispatch-registry
           (fn [dispatches]
             (into {} (remove (fn [[_ dispatch]] (= arena-id (:arena-id dispatch))))
                   dispatches)))))

;; ================================================================
;; Kernel registration and compilation
;; ================================================================

(defn register-kernel!
  "Register a kernel's compilation artifacts (source, params, etc.).
  When *current-arena* is bound, tags with that arena-id."
  ([kernel-name kernel-info]
   (register-kernel! kernel-name kernel-info *current-arena*))
  ([kernel-name kernel-info arena-id]
   (let [_ (when (kart/kernel-artifact? kernel-info) (kart/validate! kernel-info))
         info (cond-> kernel-info
                arena-id (assoc :arena-id arena-id))]
     (swap! kernel-registry assoc kernel-name info))))

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

(defn- compile-program!
  "Compile OpenCL C source to a cl_program. Returns the program handle."
  ^MemorySegment [^String source]
  (ensure-init!)
  (let [{:keys [context device arena]} @state
        err-seg (.allocate ^Arena arena I32)
        ;; Create string pointer
        src-seg (.allocateFrom ^Arena arena source)
        src-ptr-seg (.allocate ^Arena arena PTR)
        _ (.set src-ptr-seg PTR 0 src-seg)
        ;; clCreateProgramWithSource
        program (.invokeWithArguments ^MethodHandle @h-clCreateProgramWithSource
                                      (into-array Object [context (int 1) src-ptr-seg
                                                          MemorySegment/NULL err-seg]))
        _ (when (not= CL_SUCCESS (read-int err-seg))
            (throw (ex-info "clCreateProgramWithSource failed"
                            {:error (read-int err-seg)})))
        ;; clBuildProgram
        dev-seg (.allocateFrom ^Arena arena PTR device)
        ret (int (.invokeWithArguments ^MethodHandle @h-clBuildProgram
                                       (into-array Object [program (int 1) dev-seg
                                                           MemorySegment/NULL MemorySegment/NULL MemorySegment/NULL])))]
    (when (not= CL_SUCCESS ret)
      ;; Get build log for diagnostics
      (let [log-size-seg (.allocate ^Arena arena I64)
            _ (.invokeWithArguments ^MethodHandle @h-clGetProgramBuildInfo
                                    (into-array Object [program device (int CL_PROGRAM_BUILD_LOG)
                                                        (long 0) MemorySegment/NULL log-size-seg]))
            log-size (long (.get log-size-seg I64 0))
            log-buf (.allocate ^Arena arena log-size)
            _ (.invokeWithArguments ^MethodHandle @h-clGetProgramBuildInfo
                                    (into-array Object [program device (int CL_PROGRAM_BUILD_LOG)
                                                        log-size log-buf log-size-seg]))
            build-log (.getString log-buf 0)]
        (throw (ex-info (str "clBuildProgram failed: " build-log)
                        {:error ret :build-log build-log}))))
    program))

(defn- ensure-kernel-loaded!
  "Lazily compile source and create kernel for a registered kernel.
  Returns updated kernel-info with :program and :kernel-handle."
  [kernel-name]
  (ensure-init!)
  (let [info (get @kernel-registry kernel-name)]
    (when-not info
      (throw (ex-info (str "Kernel not registered: " kernel-name)
                      {:kernel-name kernel-name
                       :registered (keys @kernel-registry)})))
    (if (:kernel-handle info)
      info
      (let [{:keys [arena]} @state
            source (:source info)
            _ (when-not source
                (throw (ex-info "Kernel has no :source for OpenCL compilation"
                                {:kernel-name kernel-name})))
            program (compile-program! source)
            err-seg (.allocate ^Arena arena I32)
            kname-seg (.allocateFrom ^Arena arena ^String kernel-name)
            kernel-handle (.invokeWithArguments ^MethodHandle @h-clCreateKernel
                                                (into-array Object [program kname-seg err-seg]))
            _ (when (not= CL_SUCCESS (read-int err-seg))
                (throw (ex-info (str "clCreateKernel failed for " kernel-name)
                                {:error (read-int err-seg)})))
            updated (assoc info
                           :program program
                           :kernel-handle kernel-handle)]
        (swap! kernel-registry assoc kernel-name updated)
        updated))))

;; ================================================================
;; Kernel argument setup
;; ================================================================

(defn- set-kernel-arg-buffer!
  "Set a cl_mem buffer as a kernel argument."
  [^MemorySegment kernel ^long arg-idx ^MemorySegment cl-mem]
  (let [arena (Arena/ofConfined)]
    (try
      (let [arg-seg (.allocate arena PTR)]
        (.set arg-seg PTR 0 cl-mem)
        (cl-call! "clSetKernelArg" @h-clSetKernelArg
                  [kernel (int arg-idx) (long 8) arg-seg]))
      (finally (.close arena)))))

(defn- set-kernel-arg-scalar!
  "Set a scalar value as a kernel argument."
  [^MemorySegment kernel ^long arg-idx {:keys [type value]}]
  (let [arena (Arena/ofConfined)]
    (try
      (case type
        :int    (let [seg (.allocate arena I32)]
                  (.set seg I32 0 (int value))
                  (cl-call! "clSetKernelArg" @h-clSetKernelArg
                            [kernel (int arg-idx) (long 4) seg]))
        :long   (let [seg (.allocate arena I64)]
                  (.set seg I64 0 (long value))
                  (cl-call! "clSetKernelArg" @h-clSetKernelArg
                            [kernel (int arg-idx) (long 8) seg]))
        :float  (let [seg (.allocate arena F32)]
                  (.set seg F32 0 (float value))
                  (cl-call! "clSetKernelArg" @h-clSetKernelArg
                            [kernel (int arg-idx) (long 4) seg]))
        :double (let [seg (.allocate arena F64)]
                  (.set seg F64 0 (double value))
                  (cl-call! "clSetKernelArg" @h-clSetKernelArg
                            [kernel (int arg-idx) (long 8) seg]))
        :half   (let [seg (.allocate arena ValueLayout/JAVA_SHORT)]
                  (.set seg ValueLayout/JAVA_SHORT 0
                        (short (Float/floatToFloat16 (float value))))
                  (cl-call! "clSetKernelArg" @h-clSetKernelArg
                            [kernel (int arg-idx) (long 2) seg]))
        :byte   (let [seg (.allocate arena ValueLayout/JAVA_BYTE)]
                  (.set seg ValueLayout/JAVA_BYTE 0 (byte value))
                  (cl-call! "clSetKernelArg" @h-clSetKernelArg
                            [kernel (int arg-idx) (long 1) seg]))
        (throw (ex-info (str "Unknown scalar type: " type) {:type type})))
      (finally (.close arena)))))

;; ================================================================
;; Host segment caching (matches ze_runtime ensure-seg pattern)
;; ================================================================

(defn- ensure-host-seg
  "Return a cached host MemorySegment for [kernel-name k],
  allocating if absent or smaller than n-bytes."
  ^MemorySegment [^String kernel-name k ^long n-bytes]
  (let [info (get @kernel-registry kernel-name)
        ^MemorySegment cached (get info k)]
    (if (and cached (>= (.byteSize cached) n-bytes))
      cached
      (let [{:keys [arena]} @state
            seg (.allocate ^Arena arena n-bytes)]
        (swap! kernel-registry assoc-in [kernel-name k] seg)
        seg))))

;; ================================================================
;; Kernel invocation
;; ================================================================

(defn invoke-registered-map-void-kernel
  "Invoke a compiled map-void kernel. Mirrors ze-runtime API.
  arrays: vector of OclBuffers or JVM arrays
  scalar-args: vector of {:type :int/:float/:long :value v}
  n: number of work items"
  ([^String kernel-name arrays scalar-args n]
   (invoke-registered-map-void-kernel kernel-name arrays scalar-args n {}))
  ([^String kernel-name arrays scalar-args n opts]
   (let [abi (:abi (get @kernel-registry kernel-name))
         _ (when abi
             (kabi/validate-split-binding! abi arrays scalar-args)
             (kabi/validate-physical-pointer-dtypes! abi (physical-pointer-dtypes arrays)))
         {:keys [kernel-handle] :as info} (ensure-kernel-loaded! kernel-name)
         {:keys [queue]} @state
         dtype (kernel-info-value info :dtype :float)
         workgroup-size (long (get opts :workgroup-size
                                   (if (:launch info)
                                     (first (klaunch/static-workgroup-size (:launch info)))
                                     (long (or (:workgroup-size info) 256)))))
         n (long n)
         default-elem-size (long (get dtype-byte-sizes dtype 4))

         ;; Expand arrays: OclBuffer passes through, JVM arrays get staged
         expanded-entries
         (reduce
          (fn [acc [idx arr]]
            (cond
              (device-buffer? arr)
              (conj acc {:cl-mem (:cl-mem arr) :source nil :byte-size nil :host-seg nil})

              :else  ;; JVM array
              (let [byte-size (long (* (java.lang.reflect.Array/getLength arr)
                                       (cond
                                         (instance? (Class/forName "[F") arr) 4
                                         (instance? (Class/forName "[I") arr) 4
                                         (instance? (Class/forName "[D") arr) 8
                                         (instance? (Class/forName "[J") arr) 8
                                         (instance? (Class/forName "[S") arr) 2
                                         (instance? (Class/forName "[B") arr) 1
                                         :else default-elem-size)))
                     ;; Create temp cl_mem and upload
                    {:keys [context arena]} @state
                    err-seg (.allocate ^Arena arena I32)
                    host-seg (ensure-host-seg kernel-name
                                              (keyword (str "void-arr-" idx)) byte-size)
                    _ (MemorySegment/copy (MemorySegment/ofArray arr) 0
                                          host-seg 0 byte-size)
                    cl-mem-handle (.invokeWithArguments ^MethodHandle @h-clCreateBuffer
                                                        (into-array Object [context (long (bit-or CL_MEM_READ_WRITE CL_MEM_COPY_HOST_PTR))
                                                                            (long byte-size) host-seg err-seg]))]
                (conj acc {:cl-mem cl-mem-handle :source arr :byte-size byte-size
                           :host-seg host-seg :temp? true}))))
          []
          (map-indexed vector arrays))

         expanded-entries
         (if abi
           (mapv (fn [entry slot]
                   (assoc entry :write? (kabi/writable? slot)))
                 expanded-entries (kabi/pointer-slots abi))
           expanded-entries)

         ;; Set kernel args: buffers first, then scalars, then n
         arg-idx (atom 0)]

     ;; Buffer args
     (doseq [{:keys [cl-mem]} expanded-entries]
       (set-kernel-arg-buffer! kernel-handle @arg-idx cl-mem)
       (swap! arg-idx inc))

     ;; Scalar args
     (doseq [scalar scalar-args]
       (let [s (if (map? scalar) scalar {:type (if (= dtype :float) :float :double) :value scalar})]
         (set-kernel-arg-scalar! kernel-handle @arg-idx s)
         (swap! arg-idx inc)))

     ;; n argument (always int, always last)
     (set-kernel-arg-scalar! kernel-handle @arg-idx {:type :int :value (int n)})
     (swap! arg-idx inc)

     ;; Enqueue NDRange
     (let [global-size (* workgroup-size (long (Math/ceil (/ (double n) workgroup-size))))
           gs-arena (Arena/ofConfined)]
       (try
         (let [global-seg (.allocate gs-arena I64)
               local-seg (.allocate gs-arena I64)]
           (.set global-seg I64 0 (long global-size))
           (.set local-seg I64 0 (long workgroup-size))
           (cl-call! "clEnqueueNDRangeKernel" @h-clEnqueueNDRangeKernel
                     [queue kernel-handle (int 1) MemorySegment/NULL
                      global-seg local-seg (int 0) MemorySegment/NULL MemorySegment/NULL])
           (cl-call! "clFinish" @h-clFinish [queue]))
         (finally (.close gs-arena))))

     ;; Copy back JVM arrays and free temp cl_mems
     (doseq [{:keys [cl-mem source byte-size host-seg temp? write?]} expanded-entries]
       (when (and source (or (nil? abi) write?))
         ;; Read back from device
         (cl-call! "clEnqueueReadBuffer" @h-clEnqueueReadBuffer
                   [queue cl-mem (int CL_TRUE) (long 0) (long byte-size)
                    host-seg (int 0) MemorySegment/NULL MemorySegment/NULL])
         (MemorySegment/copy host-seg 0 (MemorySegment/ofArray source) 0 (long byte-size)))
       (when temp?
         (.invokeWithArguments ^MethodHandle @h-clReleaseMemObject
                               (into-array Object [cl-mem]))))

     nil)))

(defn invoke-registered-kernel
  "Pipeline-friendly value-returning map invocation for OpenCL.

   The emitter-authored ABI is still authoritative; this wrapper only projects the compatibility
   `(inputs, result, scalars, bound)` marker into the existing ordered OpenCL staging path and
   returns the declared result value."
  [^String kernel-name input-arrays output-array scalar-args n]
  (let [registered (or (get @kernel-registry kernel-name)
                       (throw (ex-info (str "Kernel not registered: " kernel-name)
                                       {:kernel-name kernel-name
                                        :registered (keys @kernel-registry)})))
        abi (or (:abi registered)
                (throw (ex-info "registered map kernel has no ordered ABI"
                                {:kernel-name kernel-name :registry-entry registered
                                 :fallback :none})))
        arguments (vec (concat input-arrays [output-array] scalar-args [n]))
        pairs (mapv vector (kabi/validate! abi)
                    (kabi/validate-arguments! abi arguments))
        result-pairs (filterv #(= :result (:role (first %))) pairs)]
    (when-not (= 1 (count result-pairs))
      (throw (ex-info "map kernel ABI must identify exactly one result"
                      {:kernel-name kernel-name :result-slots (mapv first result-pairs)})))
    (invoke-registered-map-void-kernel kernel-name
                                       (vec (concat input-arrays [output-array]))
                                       scalar-args n)
    (second (first result-pairs))))

(defn invoke-registered-scan-exclusive-kernel
  "Invoke Blelloch exclusive scan. Same multi-pass algorithm as ze_runtime.
  block-kernel-name: block-level scan kernel
  prop-kernel-name: propagation kernel
  input-arrays: vector of OclBuffers
  output-array: OclBuffer for output
  n: number of elements"
  [^String block-kernel-name ^String prop-kernel-name
   input-arrays output-array n]
  (let [{:keys [kernel-handle workgroup-size block-size scan-dtype]
         :or {workgroup-size 256}} (ensure-kernel-loaded! block-kernel-name)
        n (long n)
        wg (long (or workgroup-size 256))
        block-size (long (or block-size (* 2 wg)))
        num-blocks (long (Math/ceil (/ (double n) (double block-size))))
        is-int? (or (= scan-dtype :int) (= (:dtype output-array) :int))
        elem-size (long (if is-int? 4 8))
        value-layout (if is-int? ValueLayout/JAVA_INT ValueLayout/JAVA_DOUBLE)]

    (letfn [(scan-recursive! [input-bufs output-buf n depth]
              (let [num-blocks (long (Math/ceil (/ (double n) (double block-size))))
                    block-sums-buf (make-buffer num-blocks (if is-int? :int :double))]
                (try
                  ;; Phase 1: block-level scan
                  (invoke-registered-map-void-kernel
                   block-kernel-name
                   (vec (concat input-bufs [output-buf block-sums-buf]))
                   [{:type :int :value (int n)}]
                   num-blocks
                   {:workgroup-size wg})

                  (if (<= num-blocks 1)
                    ;; Single block: read total from block-sums
                    (let [arr (buffer->array block-sums-buf)]
                      (if is-int?
                        (long (clojure.core/aget ^ints arr 0))
                        (double (clojure.core/aget ^doubles arr 0))))

                    ;; Multi-block: recursive scan of block sums
                    (let [block-offsets-buf (make-buffer num-blocks (if is-int? :int :double))]
                      (try
                        (let [total (if (> depth 1)
                                     ;; CPU fallback for very deep recursion
                                      (let [sums (buffer->array block-sums-buf)
                                            offsets (if is-int? (int-array num-blocks) (double-array num-blocks))]
                                        (if is-int?
                                          (loop [i 0 acc (int 0)]
                                            (if (< i num-blocks)
                                              (do (clojure.core/aset ^ints offsets i acc)
                                                  (recur (inc i) (unchecked-add-int acc (clojure.core/aget ^ints sums i))))
                                              (do (array->buffer! block-offsets-buf offsets)
                                                  (long acc))))
                                          (loop [i 0 acc 0.0]
                                            (if (< i num-blocks)
                                              (do (clojure.core/aset ^doubles offsets i acc)
                                                  (recur (inc i) (+ acc (clojure.core/aget ^doubles sums i))))
                                              (do (array->buffer! block-offsets-buf offsets)
                                                  acc)))))
                                     ;; Recursive GPU scan
                                      (scan-recursive! [block-sums-buf] block-offsets-buf
                                                       num-blocks (inc depth)))]
                          ;; Phase 3: propagate offsets
                          (let [prop-info (ensure-kernel-loaded! prop-kernel-name)]
                            (invoke-registered-map-void-kernel
                             prop-kernel-name
                             [output-buf block-offsets-buf]
                             [{:type :int :value (int n)}]
                             num-blocks
                             {:workgroup-size wg}))
                          total)
                        (finally (free-buffer! block-offsets-buf)))))
                  (finally (free-buffer! block-sums-buf)))))]

      (let [total (scan-recursive! input-arrays output-array n 0)]
        ;; Write total to output[n]
        (let [arr (buffer->array output-array)
              full-arr (if is-int?
                         (let [a (int-array (inc n))]
                           (System/arraycopy arr 0 a 0 (int n))
                           (clojure.core/aset a (int n) (int total))
                           a)
                         (let [a (double-array (inc n))]
                           (System/arraycopy arr 0 a 0 (int n))
                           (clojure.core/aset a (int n) (double total))
                           a))]
          full-arr)))))

(defn invoke-registered-rng-fill-kernel
  "Invoke a compiled rng-fill kernel. Same interface as ze_runtime."
  [^String kernel-name seeds-buf n ^long base-seed]
  (ensure-init!)
  (let [{:keys [kernel-handle workgroup-size]
         :or {workgroup-size 256}} (ensure-kernel-loaded! kernel-name)
        {:keys [queue]} @state
        n (long n)
        wg (long workgroup-size)
        global-size (* wg (long (Math/ceil (/ (double n) wg))))]
    ;; Set args: ids-buf, n_active (reused as n), base_seed, n_agents (reused as n)
    (set-kernel-arg-buffer! kernel-handle 0 (:cl-mem seeds-buf))
    (set-kernel-arg-scalar! kernel-handle 1 {:type :int :value (int n)})
    (set-kernel-arg-scalar! kernel-handle 2 {:type :long :value base-seed})

    (let [arena (Arena/ofConfined)]
      (try
        (let [global-seg (.allocate arena I64)
              local-seg (.allocate arena I64)]
          (.set global-seg I64 0 (long global-size))
          (.set local-seg I64 0 (long wg))
          (cl-call! "clEnqueueNDRangeKernel" @h-clEnqueueNDRangeKernel
                    [queue kernel-handle (int 1) MemorySegment/NULL
                     global-seg local-seg (int 0) MemorySegment/NULL MemorySegment/NULL])
          (cl-call! "clFinish" @h-clFinish [queue]))
        (finally (.close arena))))
    (buffer->array seeds-buf)))

(defn invoke-registered-active-ids-kernel
  "Invoke a compiled active-ids kernel. Same interface as ze_runtime."
  [^String kernel-name ids-buf n-active n-agents base-seed]
  (ensure-init!)
  (let [{:keys [kernel-handle workgroup-size]
         :or {workgroup-size 256}} (ensure-kernel-loaded! kernel-name)
        {:keys [queue]} @state
        n-active (long n-active)
        wg (long workgroup-size)
        global-size (* wg (long (Math/ceil (/ (double n-active) wg))))]
    ;; Set args: ids, n_active, base_seed, n_agents
    (set-kernel-arg-buffer! kernel-handle 0 (:cl-mem ids-buf))
    (set-kernel-arg-scalar! kernel-handle 1 {:type :int :value (int n-active)})
    (set-kernel-arg-scalar! kernel-handle 2 {:type :long :value (long base-seed)})
    (set-kernel-arg-scalar! kernel-handle 3 {:type :long :value (long n-agents)})

    (let [arena (Arena/ofConfined)]
      (try
        (let [global-seg (.allocate arena I64)
              local-seg (.allocate arena I64)]
          (.set global-seg I64 0 (long global-size))
          (.set local-seg I64 0 (long wg))
          (cl-call! "clEnqueueNDRangeKernel" @h-clEnqueueNDRangeKernel
                    [queue kernel-handle (int 1) MemorySegment/NULL
                     global-seg local-seg (int 0) MemorySegment/NULL MemorySegment/NULL])
          (cl-call! "clFinish" @h-clFinish [queue]))
        (finally (.close arena))))
    (buffer->array ids-buf)))

(defn invoke-registered-scatter-kernel
  "Invoke a compiled scatter-add kernel. Same interface as ze_runtime."
  [^String kernel-name output src index n & [stride]]
  (invoke-registered-map-void-kernel kernel-name
                                     (if stride [output src index] [output src index])
                                     (if stride
                                       [{:type :int :value (int stride)}]
                                       [])
                                     n))

(defn invoke-registered-reduce-by-key-kernel
  "Invoke a compiled reduce-by-key kernel. Same interface as ze_runtime."
  [^String kernel-name output keys vals n]
  (invoke-registered-map-void-kernel kernel-name [output keys vals] [] n))

;; ================================================================
;; Lifecycle
;; ================================================================

;; ================================================================
;; Resident session layer (ports of the ze_runtime bound/graph API)
;; OpenCL has no command graphs in core: record-graph! stores the ordered
;; launches; replay-graph! enqueues them on the in-order queue (which gives
;; the ze barrier semantics) and clFinish-es once. The ~us/enqueue host cost
;; is paid per replay rather than once — correct now, cl_khr_command_buffer
;; later if it matters.
;; ================================================================

(defn kernel-registry-entry
  "Public read of a registered kernel's info map (source, :array-params,
  :scalar-params, dtype, ...). Same contract as ze-runtime."
  [kernel-name]
  (get @kernel-registry kernel-name))

(defn- registered-1d-workgroup-size
  "Read the canonical artifact workgroup or the remaining plain registry entry. A 1-D binder must
   reject rather than flatten multidimensional launch geometry."
  [kernel-info]
  (if-let [launch (:launch kernel-info)]
    (let [workgroup (klaunch/static-workgroup-size launch)]
      (when-not (= 1 (count workgroup))
        (throw (ex-info "1-D kernel path received a multidimensional launch contract"
                        {:kernel-name (:kernel-name kernel-info) :launch launch})))
      (first workgroup))
    (long (or (:workgroup-size kernel-info) 256))))

(defn- create-kernel-fresh
  "A DEDICATED cl_kernel per binding — kernel args are mutable state on the
  kernel object (same clobbering hazard as Level Zero shared handles)."
  ^MemorySegment [^MemorySegment program ^String kernel-name]
  (let [{:keys [arena]} @state
        err-seg (.allocate ^Arena arena I32)
        kname-seg (.allocateFrom ^Arena arena kernel-name)
        kh (.invokeWithArguments ^MethodHandle @h-clCreateKernel
                                 (into-array Object [program kname-seg err-seg]))]
    (when (not= CL_SUCCESS (read-int err-seg))
      (throw (ex-info (str "clCreateKernel (fresh) failed for " kernel-name)
                      {:error (read-int err-seg)})))
    kh))

(defn- device-mem-of
  "cl_mem of a resident arg (OclBuffer or raw cl_mem MemorySegment)."
  ^MemorySegment [arr]
  (cond
    (device-buffer? arr) (:cl-mem arr)
    (instance? MemorySegment arr) arr
    :else (throw (ex-info "bound path requires GPU-resident args (OclBuffer); JVM-array staging is not supported here"
                          {:arr-type (type arr)}))))

(defn bind-kernel-call
  "Bind a backend-neutral KernelCall over OpenCL resident buffers. ABI order and complete 1-3D
   geometry come exclusively from the call; no map/reduction convention is interpreted."
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
              (throw (ex-info "OpenCL KernelCall requires OclBuffer/MemorySegment pointers"
                              {:kernel-name kernel-name :slot slot :value-type (type value)}))))
        _ (kabi/validate-physical-pointer-dtypes!
           abi (physical-pointer-dtypes pointer-values))
        _ (when (= :pure-reduction (get-in registered [:effects :kind]))
            (doseq [[slot value] pointer-pairs
                    :when (= :result (:role slot))]
              (when (and (device-buffer? value) (< (:n-elements ^OclBuffer value) 1))
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
                                 (device-buffer? value) (:n-elements ^OclBuffer value)
                                 (instance? MemorySegment value)
                                 (quot (.byteSize ^MemorySegment value) (dt/bytes-of (:dtype slot))))]
                  (when (< (long capacity) out-elems)
                    (throw (ex-info "contraction output buffer is smaller than its artifact extent"
                                    {:kernel-name kernel-name :slot slot :out-elems out-elems
                                     :buffer-elements capacity})))))))
        ;; Driver contact begins only after call/artifact/ABI/value/geometry validation.
        {:keys [program]} (ensure-kernel-loaded! kernel-name)
        kh (create-kernel-fresh program kernel-name)]
    (doseq [[idx [slot value]] (map-indexed vector pairs)]
      (if (= :scalar (:kind slot))
        (set-kernel-arg-scalar! kh idx value)
        (set-kernel-arg-buffer! kh idx (device-mem-of value))))
    {:bound {:kernel kh :wg workgroup-size}
     :group-count group-count
     :kernel-name kernel-name
     :kernel-call call
     :binding-plan plan}))

(defn bind-registered-map-void-kernel
  "Pre-bind a registered void-map kernel's args ONCE over RESIDENT OclBuffers.
  Same contract as ze-runtime/bind-registered-map-void-kernel: buffer CONTENTS
  may change between launches; re-bind only on reallocation or n change."
  ([^String kernel-name arrays scalar-args n]
   (bind-registered-map-void-kernel kernel-name arrays scalar-args n {}))
  ([^String kernel-name arrays scalar-args n opts]
   (let [_ (when-let [abi (:abi (get @kernel-registry kernel-name))]
             (kabi/validate-split-binding! abi arrays scalar-args)
             (kabi/validate-physical-pointer-dtypes! abi (physical-pointer-dtypes arrays)))
         {:keys [program] :as loaded} (ensure-kernel-loaded! kernel-name)
         dtype (kernel-info-value loaded :dtype :float)
         kh (create-kernel-fresh program kernel-name)
         wg (long (get opts :workgroup-size (registered-1d-workgroup-size loaded)))
         n (long n)
         scalar-type (if (= dtype :float) :float :double)
         idx (atom -1)
         next-idx! #(swap! idx inc)]
     (doseq [arr arrays]
       (set-kernel-arg-buffer! kh (next-idx!) (device-mem-of arr)))
     (doseq [v scalar-args]
       (set-kernel-arg-scalar! kh (next-idx!)
                               (if (map? v) v
                                   {:type scalar-type
                                    :value (if (= scalar-type :float) (float v) (double v))})))
     (set-kernel-arg-scalar! kh (next-idx!) {:type :int :value (int n)})
     {:bound {:kernel kh :wg wg}
      :group-count (long (or (get opts :group-count) (Math/ceil (/ (double n) wg))))
      :kernel-name kernel-name
      :async? (boolean (get opts :async?))})))

(defn- enqueue-bound!
  "Enqueue one pre-bound kernel (no finish)."
  ([bound group-count]
   (enqueue-bound! bound group-count MemorySegment/NULL))
  ([bound group-count event-out]
   (enqueue-bound! bound group-count event-out (:queue @state)))
  ([{:keys [^MemorySegment kernel wg]} group-count event-out queue]
   (let [arena (Arena/ofConfined)
         wg (if (vector? wg) wg [(long wg)])
         group-count (if (vector? group-count) group-count [(long group-count)])
         work-dim (count wg)]
     (when-not (= work-dim (count group-count))
       (throw (ex-info "bound OpenCL workgroup and grid dimensionality differ"
                       {:workgroup wg :grid group-count})))
     (try
       (let [g (.allocate ^Arena arena (* 8 work-dim))
             l (.allocate ^Arena arena (* 8 work-dim))]
         (doseq [i (range work-dim)]
           (.set g I64 (long (* 8 i)) (* (long (nth group-count i)) (long (nth wg i))))
           (.set l I64 (long (* 8 i)) (long (nth wg i))))
         (cl-call! "clEnqueueNDRangeKernel" @h-clEnqueueNDRangeKernel
                   [queue kernel (int work-dim) MemorySegment/NULL g l
                    (int 0) MemorySegment/NULL event-out]))
       (finally (.close arena))))))

(defn launch-registered-bound!
  "Dispatch a pre-bound kernel. Synchronous."
  [prepared]
  (enqueue-bound! (:bound prepared) (:group-count prepared))
  (cl-call! "clFinish" @h-clFinish [(:queue @state)]))

(defn record-graph!
  "OpenCL 'graph': capture the ordered pre-bound launches for replay. The
  in-order queue serializes them (= ze per-kernel barriers). Profiling is opt-in: a dedicated
  CL_QUEUE_PROFILING_ENABLE queue and per-replay events exist only on a profiled graph."
  ([prepareds] (record-graph! prepareds {}))
  ([prepareds {:keys [profile?] :or {profile? false}}]
   (ensure-init!)
   (let [launches (mapv (fn [{:keys [bound group-count]}]
                          {:bound bound :group-count group-count})
                        prepareds)]
     (if-not profile?
       {:launches launches}
       (let [{:keys [context device]} @state
             arena (Arena/ofConfined)]
         (try
           (let [err-seg (.allocate arena I32)
                 queue (.invokeWithArguments ^MethodHandle @h-clCreateCommandQueue
                                             (into-array Object
                                                         [context device
                                                          CL_QUEUE_PROFILING_ENABLE err-seg]))]
             (when-not (= CL_SUCCESS (read-int err-seg))
               (throw (ex-info "clCreateCommandQueue(profile) failed"
                               {:error (read-int err-seg)})))
             {:launches launches
              :profile? true
              :profile-queue queue
              :profile-state (atom nil)
              :kernel-names (mapv #(or (:kernel-name %) "unknown") prepareds)
              :phases (mapv :phase prepareds)})
           (finally (.close arena))))))))

(defn- release-native-event!
  [event]
  (when (and event (not (.equals ^MemorySegment event MemorySegment/NULL)))
    (cl-call! "clReleaseEvent" @h-clReleaseEvent [event])))

(defn submit-range-batch!
  "Submit a validated range batch on the profiling-enabled OpenCL queue without waiting.

   Every command owns a private native staging slice until the returned completion token is
   awaited and released. Upload sources are copied into immutable staging before return; download
   destinations are populated only by await-event!, after device completion establishes host
   visibility."
  [entries direction]
  (let [active (filterv (fn [[_ plan]] (pos? (long (:n-bytes plan)))) entries)
        total-bytes (reduce + 0 (map (comp long :n-bytes second) entries))]
    (if (empty? active)
      {:complete? true
       :completion {:timing-source :host-monotonic
                    :elapsed-ns 0 :bytes total-bytes :commands 0
                    :direction direction :asynchronous? false}}
      (let [queue (:queue @state)
            arena (Arena/ofShared)
            event-outs (.allocate arena (* 8 (count active)))
            status-out (.allocate arena I32)
            enqueued (volatile! 0)]
        (try
          (let [copies
                (mapv
                 (fn [[index [^OclBuffer buffer
                              {:keys [buf-off host-off n-bytes host-seg]}]]]
                   (let [n-bytes (long n-bytes)
                         staging (.allocate arena n-bytes 64)
                         event-out (.asSlice event-outs (* 8 index) 8)]
                     (when (= :upload direction)
                       (MemorySegment/copy ^MemorySegment host-seg (long host-off)
                                           staging 0 n-bytes))
                     (cl-call! (if (= :upload direction)
                                 "clEnqueueWriteBuffer" "clEnqueueReadBuffer")
                               (if (= :upload direction)
                                 @h-clEnqueueWriteBuffer @h-clEnqueueReadBuffer)
                               [queue (:cl-mem buffer) CL_FALSE (long buf-off) n-bytes
                                staging (int 0) MemorySegment/NULL event-out])
                     (vswap! enqueued inc)
                     {:staging staging :host-seg host-seg :host-off (long host-off)
                      :n-bytes n-bytes}))
                 (map-indexed vector active))]
            (cl-call! "clFlush" @h-clFlush [queue])
            (let [events (mapv #(.get event-outs PTR (* 8 %)) (range (count active)))
                  final-offset (* 8 (dec (count active)))]
              {:transfer? true
               :queue queue
               :direction direction
               :bytes total-bytes
               :copies copies
               :events events
               :event (peek events)
               :event-array (.asSlice event-outs final-offset 8)
               :status-out status-out
               :arena arena}))
          (catch Exception error
            ;; A failed enqueue can leave earlier nonblocking commands live. Keep their staging
            ;; valid until the queue is drained, then release every event written so far.
            (try (cl-call! "clFinish" @h-clFinish [queue]) (catch Exception _))
            (doseq [index (range @enqueued)]
              (try (release-native-event! (.get event-outs PTR (* 8 index)))
                   (catch Exception _)))
            (.close arena)
            (throw error)))))))

(defn- await-transfer!
  [{:keys [direction bytes copies events]}]
  (when (= :download direction)
    (doseq [{:keys [staging host-seg host-off n-bytes]} copies]
      (MemorySegment/copy ^MemorySegment staging 0 ^MemorySegment host-seg
                          (long host-off) (long n-bytes))))
  (let [arena (Arena/ofConfined)]
    (try
      (let [value (.allocate arena I64)
            read-ts (fn [event parameter]
                      (cl-call! "clGetEventProfilingInfo" @h-clGetEventProfilingInfo
                                [event (int parameter) (long 8) value MemorySegment/NULL])
                      (.get value I64 0))
            start (read-ts (first events) CL_PROFILING_COMMAND_START)
            end (read-ts (peek events) CL_PROFILING_COMMAND_END)]
        {:timing-source :device-event
         :elapsed-ns (- end start)
         :bytes bytes
         :commands (count events)
         :direction direction
         :asynchronous? true})
      (finally
        (.close arena)))))

(defn- clear-profile-state!
  [graph]
  (when-let [profile-state (:profile-state graph)]
    (when-let [{:keys [events ^Arena arena]} @profile-state]
      (try
        (doseq [event events] (release-native-event! event))
        (finally
          (.close arena)
          (clojure.core/reset! profile-state nil))))))

(defn submit-graph!
  "Submit an OpenCL graph without waiting. The in-order queue preserves the recorded order and
   the final launch signals a native event held only by this runtime-private token."
  [graph]
  (let [launches (:launches graph)
        profile? (:profile? graph)
        queue (or (:profile-queue graph) (:queue @state))]
    (if (empty? launches)
      {:complete? true}
      (let [arena (Arena/ofShared)
            event-outs (.allocate arena (* 8 (if profile? (count launches) 1)))
            status-out (.allocate arena I32)]
        (try
          (when (and profile? @(:profile-state graph))
            (throw (ex-info "profiled OpenCL graph events must be read or reset before replay"
                            {})))
          (doseq [[index {:keys [bound group-count]}] (map-indexed vector launches)]
            (enqueue-bound! bound group-count
                            (if (or profile? (= index (dec (count launches))))
                              (.asSlice event-outs (if profile? (* 8 index) 0) 8)
                              MemorySegment/NULL)
                            queue))
          (cl-call! "clFlush" @h-clFlush [queue])
          (let [events (when profile?
                         (mapv #(.get event-outs PTR (* 8 %)) (range (count launches))))
                final-offset (if profile? (* 8 (dec (count launches))) 0)
                token {:event (.get event-outs PTR final-offset)
                       :event-array (.asSlice event-outs final-offset 8)
                       :status-out status-out
                       :arena arena
                       :profile? (boolean profile?)}]
            (when profile?
              (clojure.core/reset! (:profile-state graph) {:events events :arena arena}))
            token)
          (catch Exception e
            (.close arena)
            (throw e)))))))

(defn await-event!
  "Wait for a runtime-private OpenCL completion token."
  [{:keys [complete? completion transfer? event-array] :as token}]
  (when-not complete?
    (cl-call! "clWaitForEvents" @h-clWaitForEvents [(int 1) event-array]))
  (cond
    complete? completion
    transfer? (await-transfer! token)
    :else nil))

(defn event-complete?
  "Nonblocking query of a runtime-private OpenCL completion token."
  [{:keys [complete? event status-out]}]
  (if complete?
    true
    (do
      (cl-call! "clGetEventInfo" @h-clGetEventInfo
                [event (int CL_EVENT_COMMAND_EXECUTION_STATUS) (long 4)
                 status-out MemorySegment/NULL])
      (= CL_COMPLETE (.get ^MemorySegment status-out I32 0)))))

(defn release-event!
  "Release a runtime-private OpenCL completion token. The caller must establish completion."
  [{:keys [complete? profile? transfer? events event ^Arena arena]}]
  (when-not complete?
    ;; A profiled graph retains every event until read-graph-timestamps! (or reset) consumes the
    ;; sample. The graph owns the shared arena in that interval. Ordinary completion tokens keep
    ;; the previous immediate-release behavior.
    (when-not profile?
      (try
        (if transfer?
          (doseq [native-event events] (release-native-event! native-event))
          (release-native-event! event))
        (finally
          (.close arena)))))
  nil)

(defn replay-graph!
  "Enqueue every recorded launch and wait for completion."
  [graph]
  (let [event (submit-graph! graph)]
    (try
      (await-event! event)
      (finally
        (release-event! event)))))

(defn synchronize-async!
  "Block until all enqueued work completes."
  []
  (cl-call! "clFinish" @h-clFinish [(:queue @state)]))

(defn reset-graph-events!
  "Discard the most recent OpenCL profiling sample and release its native events."
  [graph]
  (clear-profile-state! graph)
  nil)

(defn read-graph-timestamps!
  "Read and consume one profiled OpenCL graph replay.

   OpenCL event timestamps are nanoseconds in the device time domain. The return shape matches
   Level Zero's `read-graph-timestamps!`, allowing gpu.core and autotuning to stay backend-neutral."
  [graph]
  (when-not (:profile? graph)
    (throw (ex-info "read-graph-timestamps!: graph was not recorded with :profile? true" {})))
  (let [profile-state (:profile-state graph)
        {:keys [events]} @profile-state]
    (when-not (seq events)
      (throw (ex-info "read-graph-timestamps!: profiled graph has no completed replay" {})))
    (let [arena (Arena/ofConfined)]
      (try
        (let [value (.allocate arena I64)
              read-timestamp
              (fn [event parameter]
                (cl-call! "clGetEventProfilingInfo" @h-clGetEventProfilingInfo
                          [event (int parameter) (long 8) value MemorySegment/NULL])
                (.get value I64 0))
              rows (mapv (fn [index event]
                           (let [start (read-timestamp event CL_PROFILING_COMMAND_START)
                                 end (read-timestamp event CL_PROFILING_COMMAND_END)
                                 ms (/ (double (- end start)) 1.0e6)]
                             {:kernel-name (nth (:kernel-names graph) index)
                              :phase (nth (:phases graph) index)
                              :ms ms
                              :context-ms ms
                              :start-ticks start
                              :end-ticks end}))
                         (range (count events)) events)
              span (when (seq rows)
                     (- (reduce max (map :end-ticks rows))
                        (reduce min (map :start-ticks rows))))]
          {:kernels rows
           :wall-ms (when span (/ (double span) 1.0e6))
           :ns-per-tick 1.0})
        (finally
          (.close arena)
          (clear-profile-state! graph))))))

(defn destroy-prepared!
  "Release the dedicated cl_kernel a binding owns."
  [prepared]
  (when-let [^MemorySegment kh (get-in prepared [:bound :kernel])]
    (try (.invokeWithArguments ^MethodHandle @h-clReleaseKernel
                               (into-array Object [kh]))
         (catch Exception _))))

(defn destroy-graph!
  "Release profiling-only OpenCL graph resources. Ordinary graphs own no driver objects."
  [graph]
  (clear-profile-state! graph)
  (when-let [queue (:profile-queue graph)]
    (cl-call! "clReleaseCommandQueue" @h-clReleaseCommandQueue [queue]))
  nil)

(defn bind-registered-gemm!
  "Intel-XMX fp16 GEMM is not available on generic OpenCL devices."
  [& _]
  (throw (ex-info "bind-registered-gemm! (XMX fp16 tile GEMM) is Level-Zero/Intel-only; generic OpenCL GEMM fallback not yet implemented" {})))

(defn bind-registered-convert!
  [& _]
  (throw (ex-info "bind-registered-convert! (fp16 cast for XMX) is Level-Zero/Intel-only" {})))

(defn bind-registered-transpose!
  [& _]
  (throw (ex-info "bind-registered-transpose! (XMX operand transpose) is Level-Zero/Intel-only" {})))

(defn shutdown!
  "Shutdown OpenCL runtime, releasing all handles."
  []
  (let [{:keys [queue context arena initialized?]} @state]
    (when initialized?
      (when queue
        (let [ret (int (.invokeWithArguments ^MethodHandle @h-clReleaseCommandQueue
                                             (into-array Object [queue])))]
          (when-not (zero? ret)
            (println (str "[ocl-runtime] WARNING: clReleaseCommandQueue failed with error " ret)))))
      (when context
        (let [ret (int (.invokeWithArguments ^MethodHandle @h-clReleaseContext
                                             (into-array Object [context])))]
          (when-not (zero? ret)
            (println (str "[ocl-runtime] WARNING: clReleaseContext failed with error " ret)))))
      (when arena
        (.close ^Arena arena))
      (clojure.core/reset! state
                           {:initialized? false :platform nil :device nil :context nil
                            :queue nil :arena nil :device-name nil :device-info nil
                            :unified-memory? false
                            :buffer-offset-alignment nil
                            :programs {} :kernels {}}))))

(defn reset!
  "Reset the state atom. Use with caution."
  []
  (let [{:keys [arena]} @state]
    (when arena
      (.close ^Arena arena)))
  (clojure.core/reset! state
                       {:initialized? false :platform nil :device nil :context nil
                        :queue nil :arena nil :device-name nil :device-info nil
                        :unified-memory? false
                        :buffer-offset-alignment nil
                        :programs {} :kernels {}})
  (clojure.core/reset! kernel-registry {})
  (clojure.core/reset! kernel-dispatch-registry {}))
