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
           [java.lang.invoke MethodHandle]))

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
(def ^:private CL_DEVICE_TYPE_GPU (long 4))
(def ^:private CL_MEM_READ_WRITE (long 1))
(def ^:private CL_MEM_COPY_HOST_PTR (long 32))
(def ^:private CL_TRUE (int 1))

;; clGetDeviceInfo param_name constants
(def ^:private CL_DEVICE_MAX_COMPUTE_UNITS 0x1002)
(def ^:private CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1003)
(def ^:private CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C)
(def ^:private CL_DEVICE_GLOBAL_MEM_SIZE 0x101F)
(def ^:private CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035)
(def ^:private CL_DEVICE_NAME 0x102B)
(def ^:private CL_DEVICE_VENDOR 0x102C)
(def ^:private CL_DEVICE_EXTENSIONS 0x1030)
(def ^:private CL_DEVICE_VERSION 0x102F)

;; clGetProgramBuildInfo param_name
(def ^:private CL_PROGRAM_BUILD_LOG 0x1183)

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
  {:float 4, :double 8, :int 4, :long 8, :float16 2, :half 2})

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
(def ^:private h-clEnqueueWriteBuffer
  (delay (make-handle "clEnqueueWriteBuffer" (fd I32 PTR PTR I32 I64 I64 PTR I32 PTR PTR))))
(def ^:private h-clEnqueueReadBuffer
  (delay (make-handle "clEnqueueReadBuffer" (fd I32 PTR PTR I32 I64 I64 PTR I32 PTR PTR))))
(def ^:private h-clEnqueueFillBuffer
  (delay (make-handle "clEnqueueFillBuffer" (fd I32 PTR PTR PTR I64 I64 I64 I32 PTR PTR))))
(def ^:private h-clReleaseMemObject
  (delay (make-handle "clReleaseMemObject" (fd I32 PTR))))

;; Execution
(def ^:private h-clEnqueueNDRangeKernel
  (delay (make-handle "clEnqueueNDRangeKernel" (fd I32 PTR PTR I32 PTR PTR PTR I32 PTR PTR))))
(def ^:private h-clFinish
  (delay (make-handle "clFinish" (fd I32 PTR))))

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
         :unified-memory? false
         :programs {}        ;; source-hash -> cl_program handle
         :kernels {}}))      ;; [program kernel-name] -> cl_kernel handle

(def kernel-registry
  "Global registry mapping kernel-name → kernel info.
  Same structure as raster.gpu.ze-runtime/kernel-registry."
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
      (.invokeWithArguments ^MethodHandle @h-clGetDeviceInfo
                            (into-array Object [device (int param-name) (long 0)
                                                MemorySegment/NULL size-ret]))
      (let [size (long (.get size-ret I64 0))
            buf (.allocate arena size)]
        (.invokeWithArguments ^MethodHandle @h-clGetDeviceInfo
                              (into-array Object [device (int param-name) size buf size-ret]))
        (let [s (.getString buf 0)]
          (.trim s)))
      (finally (.close arena)))))

(defn- query-device-info-uint
  "Query a cl_uint device info parameter."
  ^long [^MemorySegment device ^long param-name]
  (let [arena (Arena/ofConfined)
        buf (.allocate arena I32)]
    (try
      (.invokeWithArguments ^MethodHandle @h-clGetDeviceInfo
                            (into-array Object [device (int param-name) (long 4) buf MemorySegment/NULL]))
      (long (.get buf I32 0))
      (finally (.close arena)))))

(defn- query-device-info-ulong
  "Query a cl_ulong device info parameter."
  ^long [^MemorySegment device ^long param-name]
  (let [arena (Arena/ofConfined)
        buf (.allocate arena I64)]
    (try
      (.invokeWithArguments ^MethodHandle @h-clGetDeviceInfo
                            (into-array Object [device (int param-name) (long 8) buf MemorySegment/NULL]))
      (long (.get buf I64 0))
      (finally (.close arena)))))

(defn- query-device-info-size-t
  "Query a size_t device info parameter."
  ^long [^MemorySegment device ^long param-name]
  ;; size_t is 8 bytes on 64-bit
  (query-device-info-ulong device param-name))

;; ================================================================
;; Initialization
;; ================================================================

(declare init!)

(defn- ensure-init! []
  (when-not (:initialized? @state) (init!)))

(defn init!
  "Initialize OpenCL runtime. Idempotent.
  Finds first GPU platform/device, creates context and in-order command queue."
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
                                                      (into-array Object [plat (long CL_DEVICE_TYPE_GPU)
                                                                          (int 0) MemorySegment/NULL num-dev-seg])))]
                   (when (== CL_SUCCESS ret)
                     (let [num-dev (read-int num-dev-seg)]
                       (when (> num-dev 0)
                         (let [dev-buf (.allocate arena 8)
                               _ (cl-call! "clGetDeviceIDs" @h-clGetDeviceIDs
                                           [plat (long CL_DEVICE_TYPE_GPU) (int 1) dev-buf num-dev-seg])
                               dev (.get dev-buf PTR 0)]
                           [plat dev]))))))
               (range num-plat))
              (throw (ex-info "No OpenCL GPU devices found" {:num-platforms num-plat})))

          ;; Create context
          err-seg (.allocate arena I32)
          ctx (.invokeWithArguments ^MethodHandle @h-clCreateContext
                                    (into-array Object [MemorySegment/NULL (int 1)
                                                        (.allocateFrom arena PTR device)
                                                        MemorySegment/NULL MemorySegment/NULL err-seg]))
          _ (when (not= CL_SUCCESS (read-int err-seg))
              (throw (ex-info "clCreateContext failed" {:error (read-int err-seg)})))

          ;; Create in-order command queue (CL 1.x compatible)
          queue (.invokeWithArguments ^MethodHandle @h-clCreateCommandQueue
                                      (into-array Object [ctx device (long 0) err-seg]))
          _ (when (not= CL_SUCCESS (read-int err-seg))
              (throw (ex-info "clCreateCommandQueue failed" {:error (read-int err-seg)})))

          dev-name (query-device-info-string device CL_DEVICE_NAME)
          unified? (try (== 1 (query-device-info-uint device CL_DEVICE_HOST_UNIFIED_MEMORY))
                        (catch Exception _ false))]

      (swap! state assoc
             :initialized? true
             :platform platform
             :device device
             :context ctx
             :queue queue
             :arena arena
             :device-name dev-name
             :unified-memory? unified?)
      (println (str "[ocl-runtime] Initialized: " dev-name
                    (when unified? " (unified memory)"))))))

(defn query-devices
  "Query all OpenCL GPU devices across all platforms.
  Returns a vector of device info maps."
  []
  (try
    @ocl-lib ;; Force library load
    (let [arena (Arena/ofConfined)]
      (try
        (let [num-plat-seg (.allocate arena I32)
              _ (cl-call! "clGetPlatformIDs" @h-clGetPlatformIDs
                          [(int 0) MemorySegment/NULL num-plat-seg])
              num-plat (read-int num-plat-seg)
              plat-buf (.allocate arena (* num-plat 8))
              _ (cl-call! "clGetPlatformIDs" @h-clGetPlatformIDs
                          [(int num-plat) plat-buf num-plat-seg])]
          (vec
           (mapcat
            (fn [plat-idx]
              (let [plat (.get plat-buf PTR (* plat-idx 8))
                    num-dev-seg (.allocate arena I32)
                    ret (int (.invokeWithArguments ^MethodHandle @h-clGetDeviceIDs
                                                   (into-array Object [plat (long CL_DEVICE_TYPE_GPU)
                                                                       (int 0) MemorySegment/NULL num-dev-seg])))]
                (when (== CL_SUCCESS ret)
                  (let [num-dev (read-int num-dev-seg)
                        dev-buf (.allocate arena (* num-dev 8))
                        _ (cl-call! "clGetDeviceIDs" @h-clGetDeviceIDs
                                    [plat (long CL_DEVICE_TYPE_GPU) (int num-dev) dev-buf num-dev-seg])]
                    (mapv
                     (fn [dev-idx]
                       (let [dev (.get dev-buf PTR (* dev-idx 8))]
                         {:name (query-device-info-string dev CL_DEVICE_NAME)
                          :vendor (query-device-info-string dev CL_DEVICE_VENDOR)
                          :version (query-device-info-string dev CL_DEVICE_VERSION)
                          :max-compute-units (query-device-info-uint dev CL_DEVICE_MAX_COMPUTE_UNITS)
                          :max-work-group-size (query-device-info-size-t dev CL_DEVICE_MAX_WORK_GROUP_SIZE)
                          :max-clock-mhz (query-device-info-uint dev CL_DEVICE_MAX_CLOCK_FREQUENCY)
                          :global-mem-bytes (query-device-info-ulong dev CL_DEVICE_GLOBAL_MEM_SIZE)
                          :extensions (query-device-info-string dev CL_DEVICE_EXTENSIONS)
                          :integrated? (try (== 1 (query-device-info-uint dev CL_DEVICE_HOST_UNIFIED_MEMORY))
                                            (catch Exception _ false))}))
                     (range num-dev))))))
            (range num-plat))))
        (finally (.close arena))))
    (catch Exception _
      [])))

;; ================================================================
;; DeviceBuffer (OpenCL cl_mem backed)
;; ================================================================

(defrecord OclBuffer [^MemorySegment segment    ;; host staging MemorySegment
                      ^MemorySegment cl-mem      ;; cl_mem handle
                      ^long n-elements
                      ^long byte-size
                      dtype])

(defn device-buffer? [x]
  (instance? OclBuffer x))

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
     (->OclBuffer host-seg cl-mem (long n) byte-size dtype))))

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
    (swap! kernel-registry #(reduce dissoc % (map first arena-kernels)))))

;; ================================================================
;; Kernel registration and compilation
;; ================================================================

(defn register-kernel!
  "Register a kernel's compilation artifacts (source, params, etc.).
  When *current-arena* is bound, tags with that arena-id."
  ([kernel-name kernel-info]
   (register-kernel! kernel-name kernel-info *current-arena*))
  ([kernel-name kernel-info arena-id]
   (let [info (cond-> kernel-info
                arena-id (assoc :arena-id arena-id))]
     (swap! kernel-registry assoc kernel-name info))))

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
   (let [{:keys [kernel-handle workgroup-size dtype]
          :or {workgroup-size 256 dtype :float}
          :as info} (ensure-kernel-loaded! kernel-name)
         {:keys [queue]} @state
         workgroup-size (long (get opts :workgroup-size workgroup-size))
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
     (doseq [{:keys [cl-mem source byte-size host-seg temp?]} expanded-entries]
       (when source
         ;; Read back from device
         (cl-call! "clEnqueueReadBuffer" @h-clEnqueueReadBuffer
                   [queue cl-mem (int CL_TRUE) (long 0) (long byte-size)
                    host-seg (int 0) MemorySegment/NULL MemorySegment/NULL])
         (MemorySegment/copy host-seg 0 (MemorySegment/ofArray source) 0 (long byte-size)))
       (when temp?
         (.invokeWithArguments ^MethodHandle @h-clReleaseMemObject
                               (into-array Object [cl-mem]))))

     nil)))

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
                            :queue nil :arena nil :device-name nil :unified-memory? false
                            :programs {} :kernels {}}))))

(defn reset!
  "Reset the state atom. Use with caution."
  []
  (let [{:keys [arena]} @state]
    (when arena
      (.close ^Arena arena)))
  (clojure.core/reset! state
                       {:initialized? false :platform nil :device nil :context nil
                        :queue nil :arena nil :device-name nil :unified-memory? false
                        :programs {} :kernels {}})
  (clojure.core/reset! kernel-registry {}))
