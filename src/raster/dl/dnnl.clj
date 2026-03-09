(ns raster.dl.dnnl
  "Panama FFI bindings to oneDNN for optimized f32 convolution.

  Pure Panama — calls oneDNN directly via JDK 22+ Foreign Function API.
  Primitives are cached per-shape for amortized overhead. Native memory
  buffers are allocated per-call (oneDNN stores pointers across create→execute).

  Only f32 is supported on CPU — oneDNN does not support f64 convolution
  on the CPU engine."
  (:import [java.lang.foreign
            Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Library loading
;; ================================================================

(def ^:private lib-paths
  ["/lib/x86_64-linux-gnu/libdnnl.so"
   "/usr/lib/x86_64-linux-gnu/libdnnl.so"
   "/usr/lib64/libdnnl.so"
   "/opt/homebrew/lib/libdnnl.dylib"])

(defn- find-lib []
  (some (fn [path]
          (try
            (let [lib (SymbolLookup/libraryLookup path (Arena/global))]
              (when (.isPresent (.find lib "dnnl_engine_create"))
                lib))
            (catch Exception _ nil)))
        lib-paths))

(def ^:private dnnl-lib (delay (find-lib)))

(defn available?
  "Returns true if oneDNN is available on this system."
  []
  (boolean @dnnl-lib))

;; ================================================================
;; Panama helpers
;; ================================================================

(def ^:private A ValueLayout/ADDRESS)
(def ^:private I32 ValueLayout/JAVA_INT)
(def ^:private I64 ValueLayout/JAVA_LONG)

(defn- fd [ret & params]
  (FunctionDescriptor/of ret (into-array MemoryLayout params)))

(defn- make-handle [^String sym fd]
  (.downcallHandle (Linker/nativeLinker)
                   (.get (.find ^SymbolLookup @dnnl-lib sym))
                   ^FunctionDescriptor fd
                   (into-array Linker$Option [])))

(defn- invoke [^java.lang.invoke.MethodHandle mh & args]
  (.invokeWithArguments mh ^java.util.List (vec args)))

;; ================================================================
;; oneDNN constants (from dnnl_common_types.h / dnnl_types.h)
;; ================================================================

(def ^:const ^:private DNNL_F32 3)      ;; f16=1, bf16=2, f32=3
(def ^:const ^:private DNNL_ABCD 5)     ;; nchw = oihw
(def ^:const ^:private DNNL_A 2)        ;; 1D
(def ^:const ^:private DNNL_FWD_INF 96) ;; forward_inference
(def ^:const ^:private DNNL_FWD_TRN 64) ;; forward_training (for backward)
(def ^:const ^:private DNNL_CONV_AUTO 3)

(def ^:const ^:private ARG_SRC 1)
(def ^:const ^:private ARG_DST 17)
(def ^:const ^:private ARG_WEIGHTS 33)
(def ^:const ^:private ARG_BIAS 41)
(def ^:const ^:private ARG_DIFF_SRC 129)
(def ^:const ^:private ARG_DIFF_DST 145)
(def ^:const ^:private ARG_DIFF_WEIGHTS 161)
(def ^:const ^:private ARG_DIFF_BIAS 169)

;; ================================================================
;; Method handles (lazy)
;; ================================================================

(def ^:private h-engine-create (delay (make-handle "dnnl_engine_create" (fd I32 A I32 I32))))
(def ^:private h-stream-create (delay (make-handle "dnnl_stream_create" (fd I32 A A I32))))
(def ^:private h-md-create     (delay (make-handle "dnnl_memory_desc_create_with_tag" (fd I32 A I32 A I32 I32))))
(def ^:private h-mem-create    (delay (make-handle "dnnl_memory_create" (fd I32 A A A A))))
(def ^:private h-mem-set-handle (delay (make-handle "dnnl_memory_set_data_handle" (fd I32 A A))))
(def ^:private h-conv-fwd-pd   (delay (make-handle "dnnl_convolution_forward_primitive_desc_create" (fd I32 A A I32 I32 A A A A A A A A A))))
(def ^:private h-conv-bwd-d-pd (delay (make-handle "dnnl_convolution_backward_data_primitive_desc_create" (fd I32 A A I32 A A A A A A A A))))
(def ^:private h-conv-bwd-w-pd (delay (make-handle "dnnl_convolution_backward_weights_primitive_desc_create" (fd I32 A A I32 A A A A A A A A A))))
(def ^:private h-prim-create   (delay (make-handle "dnnl_primitive_create" (fd I32 A A))))
(def ^:private h-prim-execute  (delay (make-handle "dnnl_primitive_execute" (fd I32 A A I32 A))))
(def ^:private h-stream-wait   (delay (make-handle "dnnl_stream_wait" (fd I32 A))))
(def ^:private h-prim-destroy  (delay (make-handle "dnnl_primitive_destroy" (fd I32 A))))
(def ^:private h-pd-destroy    (delay (make-handle "dnnl_primitive_desc_destroy" (fd I32 A))))
(def ^:private h-mem-destroy   (delay (make-handle "dnnl_memory_destroy" (fd I32 A))))
(def ^:private h-md-destroy    (delay (make-handle "dnnl_memory_desc_destroy" (fd I32 A))))

;; ================================================================
;; Engine + stream (singleton)
;; ================================================================

(defn- check [^long status ^String op]
  (when-not (zero? status)
    (throw (ex-info (str "oneDNN error in " op ": status=" status) {:status status :op op}))))

(def ^:private engine+stream
  (delay
    (when @dnnl-lib
      (let [arena (Arena/global)
            pe (.allocate arena A)
            _ (check (invoke @h-engine-create pe (int 1) (int 0)) "engine_create")
            engine (.get pe A 0)
            ps (.allocate arena A)
            _ (check (invoke @h-stream-create ps engine (int 1)) "stream_create")]
        {:engine engine :stream (.get ps A 0)}))))

;; ================================================================
;; Memory descriptor + data helpers
;; ================================================================

(defn- dnnl-dims ^MemorySegment [^Arena arena vals]
  (let [arr (long-array 12)
        _ (dotimes [i (count vals)] (aset arr i (long (nth vals i))))
        seg (.allocate arena 96)]
    (MemorySegment/copy arr 0 seg I64 0 12)
    seg))

(defn- native-floats ^MemorySegment [^Arena arena ^floats arr]
  (let [seg (.allocate arena (* 4 (alength arr)))]
    (MemorySegment/copy arr 0 seg ValueLayout/JAVA_FLOAT 0 (alength arr))
    seg))

(defn- make-md ^MemorySegment [^Arena arena ndims dims]
  (let [p (.allocate arena A)]
    (check (invoke @h-md-create p (int ndims) (dnnl-dims arena dims) (int DNNL_F32) (int (if (= ndims 1) DNNL_A DNNL_ABCD)))
           "md_create")
    (.get p A 0)))

(defn- make-mem ^MemorySegment [^Arena arena md engine data-seg]
  (let [p (.allocate arena A)]
    (check (invoke @h-mem-create p md engine data-seg) "mem_create")
    (.get p A 0)))

(defn- make-prim ^MemorySegment [^Arena arena pd]
  (let [p (.allocate arena A)]
    (check (invoke @h-prim-create p pd) "prim_create")
    (.get p A 0)))

(defn- exec-args
  "Build dnnl_exec_arg_t array. entries is [[arg-id mem-handle] ...]"
  ^MemorySegment [^Arena arena entries]
  (let [n (count entries)
        args (.allocate arena (* n 16))]
    (dotimes [i n]
      (let [[arg-id mem] (nth entries i)
            base (* i 16)]
        (.set args I32 (long base) (int arg-id))
        (.set args A (long (+ base 8)) mem)))
    args))

(defn- execute! [prim stream args-seg n]
  (check (invoke @h-prim-execute prim stream (int n) args-seg) "execute")
  (check (invoke @h-stream-wait stream) "stream_wait"))

;; ================================================================
;; Cached conv2d primitive pool
;; ================================================================

(def ^:private conv-cache (atom {}))

(defn- get-or-create-conv-fwd!
  "Get cached forward conv primitive for shape, or create one."
  [n c h w oc kh kw sh sw ph pw]
  (let [key [n c h w oc kh kw sh sw ph pw]]
    (or (get @conv-cache key)
        (let [{:keys [engine]} @engine+stream
              arena (Arena/global)  ;; long-lived
              oh (+ 1 (quot (+ h (* 2 ph) (- kh)) sh))
              ow (+ 1 (quot (+ w (* 2 pw) (- kw)) sw))
              src-md (make-md arena 4 [n c h w])
              wt-md (make-md arena 4 [oc c kh kw])
              bias-md (make-md arena 1 [oc])
              dst-md (make-md arena 4 [n oc oh ow])
              p (.allocate arena A)
              _ (check (invoke @h-conv-fwd-pd p engine (int DNNL_FWD_INF) (int DNNL_CONV_AUTO)
                               src-md wt-md bias-md dst-md
                               (dnnl-dims arena [sh sw]) (dnnl-dims arena [0 0])
                               (dnnl-dims arena [ph pw]) (dnnl-dims arena [ph pw])
                               MemorySegment/NULL) "conv_fwd_pd")
              pd (.get p A 0)
              prim (make-prim arena pd)
              entry {:prim prim :pd pd
                     :src-md src-md :wt-md wt-md :bias-md bias-md :dst-md dst-md
                     :oh oh :ow ow}]
          (swap! conv-cache assoc key entry)
          entry))))

;; ================================================================
;; Public API: conv2d forward f32
;; ================================================================

(defn conv2d-forward-f32!
  "Compute f32 conv2d forward using oneDNN with cached primitives.
  All arrays are NCHW/OIHW layout. Writes result into dst."
  [^floats src ^floats weights ^floats bias ^floats dst
   n c h w oc kh kw sh sw ph pw]
  (let [n (long n) c (long c) h (long h) w (long w)
        oc (long oc) kh (long kh) kw (long kw)
        sh (long sh) sw (long sw) ph (long ph) pw (long pw)
        {:keys [engine stream]} @engine+stream
        {:keys [prim src-md wt-md bias-md dst-md]} (get-or-create-conv-fwd! n c h w oc kh kw sh sw ph pw)
        arena (Arena/ofConfined)]
    (try
      (let [src-seg (native-floats arena src)
            wt-seg (native-floats arena weights)
            bias-seg (native-floats arena bias)
            dst-seg (.allocate arena (* 4 (alength dst)))
            src-mem (make-mem arena src-md engine src-seg)
            wt-mem (make-mem arena wt-md engine wt-seg)
            bias-mem (make-mem arena bias-md engine bias-seg)
            dst-mem (make-mem arena dst-md engine dst-seg)]
        (execute! prim stream
                  (exec-args arena [[ARG_SRC src-mem] [ARG_WEIGHTS wt-mem]
                                    [ARG_BIAS bias-mem] [ARG_DST dst-mem]])
                  4)
        (MemorySegment/copy dst-seg ValueLayout/JAVA_FLOAT 0 dst 0 (alength dst))
        ;; Destroy per-call memory objects (not the cached primitive/md)
        (invoke @h-mem-destroy src-mem)
        (invoke @h-mem-destroy wt-mem)
        (invoke @h-mem-destroy bias-mem)
        (invoke @h-mem-destroy dst-mem))
      (finally
        (.close arena))))
  dst)
