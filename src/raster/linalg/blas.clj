(ns raster.linalg.blas
  "Panama FFI bindings to CBLAS (MKL preferred, OpenBLAS fallback).

  Uses JDK 22+ Foreign Function & Memory API with Linker.Option.critical(true)
  to pin heap arrays during native calls — no copying into off-heap buffers.

  Key functions:
    dgemm!  — general matrix multiply:  C = alpha * A @ B + beta * C
    dgemv!  — matrix-vector multiply:   y = alpha * A @ x + beta * y

  All arrays are row-major flat double[]. The BLAS column-major convention is
  handled internally (row-major A@B = col-major B^T@A^T with transposed args).

  Library loading is lazy — BLAS is only loaded on first call. Tries Intel MKL
  first (optimized for the host CPU), falls back to OpenBLAS. Avoids conflicts
  with Neanderthal's bundled JNI OpenBLAS by checking loader lookup first."
  (:require [raster.core :refer [deftm]]
            [raster.ad.templates :as tmpl]
            [raster.compiler.core.op-descriptor :as descriptor])
  (:import [java.lang.foreign
            Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout]))

;; ================================================================
;; Lazy library loading (MKL preferred, OpenBLAS fallback)
;; ================================================================

(def ^:private mkl-paths
  "Search paths for Intel MKL LP64 interface library.
  We load libmkl_intel_lp64.so directly (not libmkl_rt.so) to avoid the
  runtime dispatcher's OpenMP dependency (libmkl_intel_thread.so)."
  ["/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_lp64.so"
   "/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_lp64.so.2"
   "/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so"
   "/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so"])

(def ^:private openblas-paths
  "Common OpenBLAS library paths by platform."
  ["/usr/lib/x86_64-linux-gnu/libopenblas.so"  ;; Ubuntu/Debian
   "/lib/x86_64-linux-gnu/libopenblas.so"       ;; Ubuntu (legacy)
   "/usr/lib64/libopenblas.so"                  ;; Fedora/RHEL
   "/usr/lib/libopenblas.so"                    ;; Arch
   "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"  ;; macOS ARM
   "/usr/local/opt/openblas/lib/libopenblas.dylib"])   ;; macOS Intel

(defn- try-load-lib
  "Attempt to load a shared library from a list of paths.
  Returns SymbolLookup or nil."
  [paths]
  (some (fn [path]
          (try
            (SymbolLookup/libraryLookup path (Arena/global))
            (catch Exception _ nil)))
        paths))

(def ^:private mkl-core-paths
  ["/opt/intel/oneapi/mkl/latest/lib/libmkl_core.so"
   "/opt/intel/mkl/lib/intel64/libmkl_core.so"
   "/usr/lib/x86_64-linux-gnu/libmkl_core.so"])

(def ^:private mkl-sequential-paths
  ["/opt/intel/oneapi/mkl/latest/lib/libmkl_sequential.so"
   "/opt/intel/mkl/lib/intel64/libmkl_sequential.so"
   "/usr/lib/x86_64-linux-gnu/libmkl_sequential.so"])

;; Threaded MKL layer (OpenMP). Preferred over sequential: the intel_thread layer
;; multithreads GEMM (~3x on this machine at 4 threads) and works with raster's
;; critical/zero-copy heap-segment downcalls. Requires libiomp5. Set
;; RASTER_MKL_SEQUENTIAL=1 to force the sequential layer, or MKL_NUM_THREADS=N to
;; cap threads (4 is best on hybrid P/E-core CPUs; unset defaults to all cores).
(def ^:private mkl-iomp5-paths
  ["/opt/intel/oneapi/compiler/latest/lib/libiomp5.so"
   "/opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so"
   "/usr/lib/x86_64-linux-gnu/libiomp5.so"])

(def ^:private mkl-thread-paths
  ["/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_thread.so"
   "/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so"
   "/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so"])

(defn- dlopen-global
  "Load a shared library with RTLD_LAZY | RTLD_GLOBAL via Panama's native linker.
  This makes the library's symbols visible to subsequently loaded libraries,
  which is required for MKL's component libraries to find each other."
  [^String path]
  (let [linker (Linker/nativeLinker)
        dlopen-addr (.get (.find (.defaultLookup linker) "dlopen"))
        dlopen-fd (FunctionDescriptor/of ValueLayout/ADDRESS
                                         (into-array MemoryLayout
                                                     [ValueLayout/ADDRESS ValueLayout/JAVA_INT]))
        dlopen-mh (.downcallHandle linker dlopen-addr dlopen-fd
                                   (into-array Linker$Option []))
        cstr (.allocateFrom (Arena/global) path)
        rtld-global (if (.contains (System/getProperty "os.name") "Mac") 0x8 0x100)
        handle ^MemorySegment (.invokeWithArguments dlopen-mh
                                                    [cstr (int (bit-or 1 rtld-global))])]
    (not= (.address handle) 0)))

(defn- try-load-mkl
  "Load MKL with sequential threading (no OpenMP dependency).
  Uses dlopen(RTLD_GLOBAL) to load core → sequential → lp64 so the
  component libraries can resolve each other's symbols. Then gets a
  stable SymbolLookup handle via libraryLookup."
  []
  (try
    (let [force-seq? (= "1" (System/getenv "RASTER_MKL_SEQUENTIAL"))
          ;; Prefer the THREADED layer (iomp5 → core → intel_thread → lp64) for
          ;; multithreaded GEMM; fall back to sequential (core → sequential →
          ;; lp64) if iomp5/intel_thread are unavailable or forced off.
          threaded? (and (not force-seq?)
                         (some dlopen-global mkl-iomp5-paths)
                         (some dlopen-global mkl-core-paths)
                         (some dlopen-global mkl-thread-paths)
                         (some dlopen-global mkl-paths))
          ok (or threaded?
                 (and (some dlopen-global mkl-core-paths)
                      (some dlopen-global mkl-sequential-paths)
                      (some dlopen-global mkl-paths)))]
      (when ok
        ;; Get a SymbolLookup handle for the already-loaded library
        (when-let [lib (try-load-lib mkl-paths)]
          (when (.isPresent (.find lib "cblas_dgemm"))
            [lib (if threaded? :mkl-threaded :mkl)]))))
    (catch Exception _ nil)))

(defn- try-load-openblas []
  (when-let [lib (try-load-lib openblas-paths)]
    (when (.isPresent (.find lib "cblas_dgemm"))
      [lib :openblas])))

(defn- find-blas []
  ;; First check if cblas symbols are already available (e.g. loaded by
  ;; Neanderthal's JNI OpenBLAS). This avoids loading a second library
  ;; which causes LAPACKE symbol conflicts.
  (let [loader (SymbolLookup/loaderLookup)]
    (if (.isPresent (.find loader "cblas_dgemm"))
      [loader :preloaded]
      ;; Try MKL first (with sequential threading), then OpenBLAS
      (or (try-load-mkl)
          (try-load-openblas)))))

;; Lazy — only loads library on first deref
(def ^:private blas-state
  "Delay returning [SymbolLookup, backend-keyword]."
  (delay (find-blas)))

(defn backend
  "Returns the active BLAS backend: :mkl, :openblas, :preloaded, or nil if unavailable."
  []
  (second @blas-state))

(defn- require-blas! []
  (when-not @blas-state
    (throw (ex-info "No BLAS library found. Install Intel MKL or OpenBLAS."
                    {:searched-mkl mkl-paths
                     :searched-openblas openblas-paths}))))

;; ================================================================
;; MethodHandle creation with critical(true)
;; ================================================================

(def ^:private ^"[Ljava.lang.foreign.Linker$Option;"
  critical-opts
  (into-array Linker$Option [(Linker$Option/critical true)]))

(defn- lookup-symbol ^MemorySegment [^String name]
  (require-blas!)
  (let [^SymbolLookup lib (first @blas-state)
        opt (.find lib name)]
    (when-not (.isPresent opt)
      (throw (ex-info (str "BLAS symbol not found: " name
                           " (backend: " (second @blas-state) ")")
                      {:symbol name :backend (second @blas-state)})))
    (.get opt)))

(defn- make-handle
  "Create a downcall MethodHandle with critical(true) for zero-copy heap access."
  [^String symbol-name ^FunctionDescriptor fd]
  (.downcallHandle (Linker/nativeLinker)
                   (lookup-symbol symbol-name)
                   fd
                   critical-opts))

(defn- make-optional-handle
  "Create a zero-copy downcall when the active BLAS exposes an optional extension.
  nil keeps callers on their portable CBLAS fallback."
  [^String symbol-name ^FunctionDescriptor fd]
  (require-blas!)
  (let [opt (.find ^SymbolLookup (first @blas-state) symbol-name)]
    (when (.isPresent opt)
      (.downcallHandle (Linker/nativeLinker) (.get opt) fd critical-opts))))

;; ================================================================
;; CBLAS constants
;; ================================================================

(def ^:private ^:const CBLAS_ROW_MAJOR (int 101))
(def ^:private ^:const CBLAS_NO_TRANS  (int 111))
(def ^:private ^:const CBLAS_TRANS     (int 112))
(def ^:private ^:const CBLAS_PACKED    (int 151))
(def ^:private ^:const CBLAS_B_MATRIX  (int 162))

;; ================================================================
;; cblas_dgemm — general matrix multiply
;; ================================================================

(def ^:private dgemm-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; Order
                ValueLayout/JAVA_INT    ;; TransA
                ValueLayout/JAVA_INT    ;; TransB
                ValueLayout/JAVA_INT    ;; M
                ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_INT    ;; K
                ValueLayout/JAVA_DOUBLE ;; alpha
                ValueLayout/ADDRESS     ;; A
                ValueLayout/JAVA_INT    ;; lda
                ValueLayout/ADDRESS     ;; B
                ValueLayout/JAVA_INT    ;; ldb
                ValueLayout/JAVA_DOUBLE ;; beta
                ValueLayout/ADDRESS     ;; C
                ValueLayout/JAVA_INT])))  ;; ldc

;; Lazy — MethodHandle created on first deref
(def ^:private dgemm-mh (delay (make-handle "cblas_dgemm" dgemm-fd)))

(defn- scale-output-double!
  "BLAS semantics for a GEMM whose contraction extent is zero: `C := beta·C` over the `m·n`
   output elements (zero fill for `beta = 0`). The library call itself is skipped for `k = 0`
   because MKL rejects a zero leading dimension, but the output contract must not change."
  [^doubles C ^long n ^double beta]
  (if (zero? beta)
    (java.util.Arrays/fill C 0 (int n) 0.0)
    (dotimes [i n] (aset C i (* beta (aget C i))))))

(defn- scale-output-float!
  [^floats C ^long n ^double beta]
  (if (zero? beta)
    (java.util.Arrays/fill C 0 (int n) (float 0.0))
    (dotimes [i n] (aset C i (float (* beta (aget C i)))))))

(deftm ^:no-inline dgemm!
  [A :- (Array double) B :- (Array double) C :- (Array double)
   m :- Long k :- Long n :- Long alpha :- Double beta :- Double]
  :- (Array double)
  ;; Skip degenerate (empty) gemm — see the f32 dgemm-nt! note (MKL lda=0 guard).
  (if (and (pos? m) (pos? n) (zero? k))
    (scale-output-double! C (* m n) beta)
    (when (and (pos? m) (pos? n) (pos? k))
      (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgemm-mh
                            [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS
                             (int m) (int n) (int k)
                             alpha
                             (MemorySegment/ofArray A) (int k)
                             (MemorySegment/ofArray B) (int n)
                             beta
                             (MemorySegment/ofArray C) (int n)])))
  C)

(deftm ^:no-inline dgemm-tn!
  [A :- (Array double) B :- (Array double) C :- (Array double)
   m :- Long k :- Long n :- Long alpha :- Double beta :- Double]
  :- (Array double)
  (if (and (pos? m) (pos? n) (zero? k))
    (scale-output-double! C (* m n) beta)
    (when (and (pos? m) (pos? n) (pos? k))
      (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgemm-mh
                            [CBLAS_ROW_MAJOR CBLAS_TRANS CBLAS_NO_TRANS
                             (int m) (int n) (int k)
                             alpha
                             (MemorySegment/ofArray A) (int m)   ;; lda = m (A is [k,m])
                             (MemorySegment/ofArray B) (int n)
                             beta
                             (MemorySegment/ofArray C) (int n)])))
  C)

(deftm ^:no-inline dgemm-nt!
  [A :- (Array double) B :- (Array double) C :- (Array double)
   m :- Long k :- Long n :- Long alpha :- Double beta :- Double]
  :- (Array double)
  (if (and (pos? m) (pos? n) (zero? k))
    (scale-output-double! C (* m n) beta)
    (when (and (pos? m) (pos? n) (pos? k))
      (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgemm-mh
                            [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS
                             (int m) (int n) (int k)
                             alpha
                             (MemorySegment/ofArray A) (int k)
                             (MemorySegment/ofArray B) (int k)   ;; ldb = k (B is [n,k])
                             beta
                             (MemorySegment/ofArray C) (int n)])))
  C)

;; ================================================================
;; cblas_sgemm — single-precision general matrix multiply
;; ================================================================

(def ^:private sgemm-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; Order
                ValueLayout/JAVA_INT    ;; TransA
                ValueLayout/JAVA_INT    ;; TransB
                ValueLayout/JAVA_INT    ;; M
                ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_INT    ;; K
                ValueLayout/JAVA_FLOAT  ;; alpha
                ValueLayout/ADDRESS     ;; A
                ValueLayout/JAVA_INT    ;; lda
                ValueLayout/ADDRESS     ;; B
                ValueLayout/JAVA_INT    ;; ldb
                ValueLayout/JAVA_FLOAT  ;; beta
                ValueLayout/ADDRESS     ;; C
                ValueLayout/JAVA_INT])))  ;; ldc

(def ^:private sgemm-mh (delay (make-handle "cblas_sgemm" sgemm-fd)))

;; oneMKL's optional packed-GEMM extension. Packed storage is deliberately an
;; opaque, closeable object: its format is backend-private, depends on M/N/K,
;; and must remain at the address where MKL created its internal metadata.
(def ^:private sgemm-pack-get-size-fd
  (FunctionDescriptor/of
   ValueLayout/JAVA_LONG
   (into-array MemoryLayout
               [ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT ValueLayout/JAVA_INT])))

(def ^:private sgemm-pack-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_FLOAT ValueLayout/ADDRESS ValueLayout/JAVA_INT
                ValueLayout/ADDRESS])))

(def ^:private sgemm-compute-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/ADDRESS ValueLayout/JAVA_INT
                ValueLayout/ADDRESS ValueLayout/JAVA_INT
                ValueLayout/JAVA_FLOAT ValueLayout/ADDRESS ValueLayout/JAVA_INT])))

(def ^:private sgemm-pack-get-size-mh
  (delay (make-optional-handle "cblas_sgemm_pack_get_size" sgemm-pack-get-size-fd)))
(def ^:private sgemm-pack-mh
  (delay (make-optional-handle "cblas_sgemm_pack" sgemm-pack-fd)))
(def ^:private sgemm-compute-mh
  (delay (make-optional-handle "cblas_sgemm_compute" sgemm-compute-fd)))

(declare dgemm-nt!)

(defrecord PackedSgemmNtB [raw packed arena m k n alpha bytes backend closed]
  java.lang.AutoCloseable
  (close [_]
    (when (compare-and-set! closed false true)
      (when (and arena (.isAlive (.scope ^Arena arena)))
        (.close ^Arena arena)))))

(defn pack-sgemm-nt-b
  "Prepare a fixed-shape F32 B operand for repeated `A @ B^T` calls.

  oneMKL receives aligned, address-stable native storage and bakes `alpha` plus
  the exact M/N/K shape into its private format. Callers must close the returned
  value (prefer `with-open`). Backends without the optional MKL extension return
  the same closeable contract backed by the original array; compute then falls
  back to ordinary CBLAS. Packed objects are not portable across shapes."
  [^floats B m k n alpha]
  (let [m (long m) k (long k) n (long n) alpha (float alpha)
        get-size @sgemm-pack-get-size-mh
        pack @sgemm-pack-mh
        compute @sgemm-compute-mh]
    (when-not (= (alength B) (* n k))
      (throw (ex-info "SGEMM B size does not match n*k"
                      {:length (alength B) :m m :k k :n n})))
    (if (and get-size pack compute (pos? m) (pos? k) (pos? n))
      (let [bytes (long (.invokeWithArguments ^java.lang.invoke.MethodHandle get-size
                                              [CBLAS_B_MATRIX (int m) (int n) (int k)]))
            arena (Arena/ofShared)]
        (try
          (let [storage (.allocate arena bytes 64)]
            (.invokeWithArguments ^java.lang.invoke.MethodHandle pack
                                  [CBLAS_ROW_MAJOR CBLAS_B_MATRIX CBLAS_TRANS
                                   (int m) (int n) (int k) alpha
                                   (MemorySegment/ofArray B) (int k) storage])
            (->PackedSgemmNtB nil storage arena m k n alpha bytes :mkl-packed
                              (atom false)))
          (catch Throwable error
            (.close arena)
            (throw error))))
      (->PackedSgemmNtB B nil nil m k n alpha (* 4 (alength B)) :portable
                        (atom false)))))

(defn sgemm-nt-prepacked!
  "Compute `C := packed-alpha * A @ B^T + beta*C` with a value returned by
  `pack-sgemm-nt-b`. Shape mismatches fail rather than silently repacking."
  [^floats A ^PackedSgemmNtB prepared ^floats C m k n beta]
  (let [m (long m) k (long k) n (long n) beta (float beta)]
    (when @(:closed prepared)
      (throw (ex-info "prepacked SGEMM storage is closed" {:shape [m k n]})))
    (when-not (= [m k n] [(:m prepared) (:k prepared) (:n prepared)])
      (throw (ex-info "prepacked SGEMM shape mismatch"
                      {:requested [m k n]
                       :prepared [(:m prepared) (:k prepared) (:n prepared)]})))
    (when-not (= (alength A) (* m k))
      (throw (ex-info "SGEMM A size does not match m*k"
                      {:length (alength A) :m m :k k :n n})))
    (when-not (= (alength C) (* m n))
      (throw (ex-info "SGEMM C size does not match m*n"
                      {:length (alength C) :m m :k k :n n})))
    (if-let [^MemorySegment storage (:packed prepared)]
      (do
        (when-not (.isAlive (.scope storage))
          (throw (ex-info "prepacked SGEMM storage is closed" {:shape [m k n]})))
        (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-compute-mh
                              [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_PACKED
                               (int m) (int n) (int k)
                               (MemorySegment/ofArray A) (int k)
                               storage (int 0) beta
                               (MemorySegment/ofArray C) (int n)]))
      (dgemm-nt! A (:raw prepared) C m k n (:alpha prepared) beta))
    C))

(deftm ^:no-inline dgemm!
  [A :- (Array float) B :- (Array float) C :- (Array float)
   m :- Long k :- Long n :- Long alpha :- Float beta :- Float]
  :- (Array float)
  (if (and (pos? m) (pos? n) (zero? k))
    (scale-output-float! C (* m n) beta)
    (when (and (pos? m) (pos? n) (pos? k))
      (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-mh
                            [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS
                             (int m) (int n) (int k)
                             alpha
                             (MemorySegment/ofArray A) (int k)
                             (MemorySegment/ofArray B) (int n)
                             beta
                             (MemorySegment/ofArray C) (int n)])))
  C)

(deftm ^:no-inline dgemm-tn!
  [A :- (Array float) B :- (Array float) C :- (Array float)
   m :- Long k :- Long n :- Long alpha :- Float beta :- Float]
  :- (Array float)
  (if (and (pos? m) (pos? n) (zero? k))
    (scale-output-float! C (* m n) beta)
    (when (and (pos? m) (pos? n) (pos? k))
      (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-mh
                            [CBLAS_ROW_MAJOR CBLAS_TRANS CBLAS_NO_TRANS
                             (int m) (int n) (int k)
                             alpha
                             (MemorySegment/ofArray A) (int m)
                             (MemorySegment/ofArray B) (int n)
                             beta
                             (MemorySegment/ofArray C) (int n)])))
  C)

(deftm ^:no-inline dgemm-nt!
  [A :- (Array float) B :- (Array float) C :- (Array float)
   m :- Long k :- Long n :- Long alpha :- Float beta :- Float]
  :- (Array float)
  ;; A degenerate (empty) gemm is a no-op; skip the FFI call. MKL rejects
  ;; lda=0 (Parameter 9) while OpenBLAS tolerates it. The all-zero case comes
  ;; from raster's parametric specialization invoking the freshly-compiled
  ;; float `linear` impl with the triggering call's args (core.clj parametric-
  ;; specialize!). Guarding here keeps both backends happy and is BLAS-correct.
  (if (and (pos? m) (pos? n) (zero? k))
    (scale-output-float! C (* m n) beta)
    (when (and (pos? m) (pos? n) (pos? k))
      (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-mh
                            [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS
                             (int m) (int n) (int k)
                             alpha
                             (MemorySegment/ofArray A) (int k)
                             (MemorySegment/ofArray B) (int k)
                             beta
                             (MemorySegment/ofArray C) (int n)])))
  C)

;; ── which argument a GEMM WRITES (the compiler's effect model) ─────────────────
;; A `!` name makes these `mutating-op?` by convention, and a mutating op with no
;; registered output index is treated as possibly writing EVERY array argument. That
;; over-approximation is what kept DEAD gemms alive: DCE keeps a mutating binding
;; whose mutation target is live, and `A`/`B` (the READ-ONLY operands) are of course
;; live — so a `C` nobody ever reads still got computed. The concrete case is
;; frozen-weight training: the backward of a `linear-nb` against a :constant weight
;; produces a dW (a `:tn` gemm into a fresh buffer) whose value+grad slot the caller
;; never `nth`s. It was computed, written, and discarded — 7 GEMMs + their f16
;; converts/transposes per gemma LoRA layer.
;;
;; C (arg 2) is the ONLY buffer a cblas gemm writes. Declaring it makes the effect
;; explicit, so DCE can see that a dead C means a dead GEMM. :allocates? false — these
;; write into a caller-provided buffer, so buffer_fuse leaves them alone (the same
;; shape as the conv2d-backward-*-into! registrations). Mode is :accumulate because
;; beta is a RUNTIME arg (beta≠0 reads C first); every current call site passes 0.
(doseq [op '[raster.linalg.blas/dgemm! raster.linalg.blas/dgemm-tn!
             raster.linalg.blas/dgemm-nt!]]
  (descriptor/register-buffer-semantics! op {:allocates? false :in-place-arg 2
                                             :mutating? true})
  (descriptor/register-buffer-write! op :accumulate 2))

;; ================================================================
;; Batched GEMM — one contiguous [batch,·,·] buffer per operand. Use a single
;; strided-batch call when the BLAS exposes it (oneMKL), otherwise loop over head
;; slabs via asSlice offsets without allocating. Used by batched multi-head
;; attention: QK^T (nt) and scores@V (nn). alpha folds the 1/sqrt(dk) scale.
;; float+double arms (dispatch), mirroring dgemm!. beta is 0 (overwrite C).
;; ================================================================

(def ^:private sgemm-batch-strided-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_FLOAT
                ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_FLOAT
                ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT])))

(def ^:private dgemm-batch-strided-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_DOUBLE
                ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_DOUBLE
                ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT])))

(def ^:private sgemm-batch-strided-mh
  (delay (make-optional-handle "cblas_sgemm_batch_strided" sgemm-batch-strided-fd)))
(def ^:private dgemm-batch-strided-mh
  (delay (make-optional-handle "cblas_dgemm_batch_strided" dgemm-batch-strided-fd)))

(deftm ^:no-inline batched-gemm-nt-layout!
  "C_h = alpha*A_h@B_h^T for equally strided row-major matrix views.

  Offsets, leading dimensions, and batch strides are in elements. Unlike
  `batched-gemm-nt!`, matrices need not occupy contiguous slabs; this admits
  head views embedded in `[row,head,dim]` tensors without packing."
  [A :- (Array float) B :- (Array float) C :- (Array float)
   batch :- Long m :- Long k :- Long n :- Long alpha :- Float
   a-offset :- Long lda :- Long stride-a :- Long
   b-offset :- Long ldb :- Long stride-b :- Long
   c-offset :- Long ldc :- Long stride-c :- Long] :- (Array float)
  (when (and (pos? batch) (pos? m) (pos? n) (pos? k))
    (let [^MemorySegment sa (.asSlice (MemorySegment/ofArray A) (* (long a-offset) 4))
          ^MemorySegment sb (.asSlice (MemorySegment/ofArray B) (* (long b-offset) 4))
          ^MemorySegment sc (.asSlice (MemorySegment/ofArray C) (* (long c-offset) 4))]
      (if-let [mh ^java.lang.invoke.MethodHandle
               ;; oneMKL rejects an output batch stride smaller than the
               ;; padded matrix span, even when the logical elements of
               ;; interleaved matrix views are disjoint.  Independent GEMMs
               ;; preserve that useful layout without a pack/unpack pass.
               (when (or (== batch 1) (>= stride-c (* m ldc)))
                 @sgemm-batch-strided-mh)]
        (.invokeWithArguments mh
                              [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS
                               (int m) (int n) (int k) alpha
                               sa (int lda) (Math/toIntExact (long stride-a))
                               sb (int ldb) (Math/toIntExact (long stride-b))
                               (float 0.0) sc (int ldc) (Math/toIntExact (long stride-c))
                               (Math/toIntExact (long batch))])
        (dotimes [h batch]
          (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-mh
                                [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS
                                 (int m) (int n) (int k) alpha
                                 (.asSlice sa (* (long h) stride-a 4)) (int lda)
                                 (.asSlice sb (* (long h) stride-b 4)) (int ldb)
                                 (float 0.0)
                                 (.asSlice sc (* (long h) stride-c 4)) (int ldc)])))))
  C)

(deftm ^:no-inline batched-gemm-nn-layout!
  "C_h = alpha*A_h@B_h for equally strided row-major matrix views.

  Offsets, leading dimensions, and batch strides are in elements. Input and
  output heads may be interleaved in a wider row layout."
  [A :- (Array float) B :- (Array float) C :- (Array float)
   batch :- Long m :- Long k :- Long n :- Long alpha :- Float
   a-offset :- Long lda :- Long stride-a :- Long
   b-offset :- Long ldb :- Long stride-b :- Long
   c-offset :- Long ldc :- Long stride-c :- Long] :- (Array float)
  (when (and (pos? batch) (pos? m) (pos? n) (pos? k))
    (let [^MemorySegment sa (.asSlice (MemorySegment/ofArray A) (* (long a-offset) 4))
          ^MemorySegment sb (.asSlice (MemorySegment/ofArray B) (* (long b-offset) 4))
          ^MemorySegment sc (.asSlice (MemorySegment/ofArray C) (* (long c-offset) 4))]
      (if-let [mh ^java.lang.invoke.MethodHandle
               (when (or (== batch 1) (>= stride-c (* m ldc)))
                 @sgemm-batch-strided-mh)]
        (.invokeWithArguments mh
                              [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS
                               (int m) (int n) (int k) alpha
                               sa (int lda) (Math/toIntExact (long stride-a))
                               sb (int ldb) (Math/toIntExact (long stride-b))
                               (float 0.0) sc (int ldc) (Math/toIntExact (long stride-c))
                               (Math/toIntExact (long batch))])
        (dotimes [h batch]
          (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-mh
                                [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS
                                 (int m) (int n) (int k) alpha
                                 (.asSlice sa (* (long h) stride-a 4)) (int lda)
                                 (.asSlice sb (* (long h) stride-b 4)) (int ldb)
                                 (float 0.0)
                                 (.asSlice sc (* (long h) stride-c 4)) (int ldc)])))))
  C)

(deftm ^:no-inline batched-gemm-nt!
  "C_h = alpha * A_h @ B_h^T for h in 0..batch. A:[batch,m,k] B:[batch,n,k]
  C:[batch,m,n], all row-major contiguous."
  [A :- (Array float) B :- (Array float) C :- (Array float)
   batch :- Long m :- Long k :- Long n :- Long alpha :- Float] :- (Array float)
  (when (and (pos? batch) (pos? m) (pos? n) (pos? k))
    (let [sa (MemorySegment/ofArray A) sb (MemorySegment/ofArray B) sc (MemorySegment/ofArray C)
          as (* m k) bs (* n k) cs (* m n)]
      (if-let [mh ^java.lang.invoke.MethodHandle @sgemm-batch-strided-mh]
        (.invokeWithArguments mh
                              [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS
                               (int m) (int n) (int k) alpha
                               sa (int k) (Math/toIntExact (long as))
                               sb (int k) (Math/toIntExact (long bs))
                               (float 0.0) sc (int n) (Math/toIntExact (long cs))
                               (Math/toIntExact (long batch))])
        (dotimes [h batch]
          (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-mh
                                [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS (int m) (int n) (int k) alpha
                                 (.asSlice sa (* (long h) as 4)) (int k)
                                 (.asSlice sb (* (long h) bs 4)) (int k)
                                 (float 0.0) (.asSlice sc (* (long h) cs 4)) (int n)])))))
  C)

(deftm ^:no-inline batched-gemm-nt!
  [A :- (Array double) B :- (Array double) C :- (Array double)
   batch :- Long m :- Long k :- Long n :- Long alpha :- Double] :- (Array double)
  (when (and (pos? batch) (pos? m) (pos? n) (pos? k))
    (let [sa (MemorySegment/ofArray A) sb (MemorySegment/ofArray B) sc (MemorySegment/ofArray C)
          as (* m k) bs (* n k) cs (* m n)]
      (if-let [mh ^java.lang.invoke.MethodHandle @dgemm-batch-strided-mh]
        (.invokeWithArguments mh
                              [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS
                               (int m) (int n) (int k) alpha
                               sa (int k) (Math/toIntExact (long as))
                               sb (int k) (Math/toIntExact (long bs))
                               0.0 sc (int n) (Math/toIntExact (long cs))
                               (Math/toIntExact (long batch))])
        (dotimes [h batch]
          (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgemm-mh
                                [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_TRANS (int m) (int n) (int k) alpha
                                 (.asSlice sa (* (long h) as 8)) (int k)
                                 (.asSlice sb (* (long h) bs 8)) (int k)
                                 0.0 (.asSlice sc (* (long h) cs 8)) (int n)])))))
  C)

(deftm ^:no-inline batched-gemm-nn!
  "C_h = alpha * A_h @ B_h for h in 0..batch. A:[batch,m,k] B:[batch,k,n]
  C:[batch,m,n], all row-major contiguous."
  [A :- (Array float) B :- (Array float) C :- (Array float)
   batch :- Long m :- Long k :- Long n :- Long alpha :- Float] :- (Array float)
  (when (and (pos? batch) (pos? m) (pos? n) (pos? k))
    (let [sa (MemorySegment/ofArray A) sb (MemorySegment/ofArray B) sc (MemorySegment/ofArray C)
          as (* m k) bs (* k n) cs (* m n)]
      (if-let [mh ^java.lang.invoke.MethodHandle @sgemm-batch-strided-mh]
        (.invokeWithArguments mh
                              [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS
                               (int m) (int n) (int k) alpha
                               sa (int k) (Math/toIntExact (long as))
                               sb (int n) (Math/toIntExact (long bs))
                               (float 0.0) sc (int n) (Math/toIntExact (long cs))
                               (Math/toIntExact (long batch))])
        (dotimes [h batch]
          (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemm-mh
                                [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS (int m) (int n) (int k) alpha
                                 (.asSlice sa (* (long h) as 4)) (int k)
                                 (.asSlice sb (* (long h) bs 4)) (int n)
                                 (float 0.0) (.asSlice sc (* (long h) cs 4)) (int n)])))))
  C)

(deftm ^:no-inline batched-gemm-nn!
  [A :- (Array double) B :- (Array double) C :- (Array double)
   batch :- Long m :- Long k :- Long n :- Long alpha :- Double] :- (Array double)
  (when (and (pos? batch) (pos? m) (pos? n) (pos? k))
    (let [sa (MemorySegment/ofArray A) sb (MemorySegment/ofArray B) sc (MemorySegment/ofArray C)
          as (* m k) bs (* k n) cs (* m n)]
      (if-let [mh ^java.lang.invoke.MethodHandle @dgemm-batch-strided-mh]
        (.invokeWithArguments mh
                              [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS
                               (int m) (int n) (int k) alpha
                               sa (int k) (Math/toIntExact (long as))
                               sb (int n) (Math/toIntExact (long bs))
                               0.0 sc (int n) (Math/toIntExact (long cs))
                               (Math/toIntExact (long batch))])
        (dotimes [h batch]
          (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgemm-mh
                                [CBLAS_ROW_MAJOR CBLAS_NO_TRANS CBLAS_NO_TRANS (int m) (int n) (int k) alpha
                                 (.asSlice sa (* (long h) as 8)) (int k)
                                 (.asSlice sb (* (long h) bs 8)) (int n)
                                 0.0 (.asSlice sc (* (long h) cs 8)) (int n)])))))
  C)

;; Every batched contraction overwrites only C (argument 2).  Make that
;; explicit so calls embedded in a larger compiled graph participate in DCE
;; and buffer planning just like the scalar GEMM wrappers above.
(doseq [op '[raster.linalg.blas/batched-gemm-nt-layout!
             raster.linalg.blas/batched-gemm-nn-layout!
             raster.linalg.blas/batched-gemm-nt!
             raster.linalg.blas/batched-gemm-nn!]]
  (descriptor/register-buffer-semantics! op {:allocates? false :in-place-arg 2
                                             :mutating? true})
  (descriptor/register-buffer-write! op :overwrite 2))

;; ================================================================
;; cblas_dgemv — matrix-vector multiply
;; ================================================================

(def ^:private dgemv-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; Order
                ValueLayout/JAVA_INT    ;; Trans
                ValueLayout/JAVA_INT    ;; M
                ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_DOUBLE ;; alpha
                ValueLayout/ADDRESS     ;; A
                ValueLayout/JAVA_INT    ;; lda
                ValueLayout/ADDRESS     ;; x
                ValueLayout/JAVA_INT    ;; incx
                ValueLayout/JAVA_DOUBLE ;; beta
                ValueLayout/ADDRESS     ;; y
                ValueLayout/JAVA_INT])))  ;; incy

;; Lazy — MethodHandle created on first deref
(def ^:private dgemv-mh (delay (make-handle "cblas_dgemv" dgemv-fd)))

(deftm ^:no-inline dgemv!
  [A :- (Array double) x :- (Array double) y :- (Array double)
   m :- Long n :- Long alpha :- Double beta :- Double]
  :- (Array double)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgemv-mh
                        [CBLAS_ROW_MAJOR CBLAS_NO_TRANS
                         (int m) (int n)
                         alpha
                         (MemorySegment/ofArray A) (int n)
                         (MemorySegment/ofArray x) (int 1)
                         beta
                         (MemorySegment/ofArray y) (int 1)])
  y)

;; ================================================================
;; cblas_dgemv with transpose — matrix-vector multiply
;; ================================================================

(deftm ^:no-inline dgemv-t!
  "y = alpha * A^T @ x + beta * y.  A is [m,n] row-major."
  [A :- (Array double) x :- (Array double) y :- (Array double)
   m :- Long n :- Long alpha :- Double beta :- Double]
  :- (Array double)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgemv-mh
                        [CBLAS_ROW_MAJOR CBLAS_TRANS
                         (int m) (int n)
                         alpha
                         (MemorySegment/ofArray A) (int n)
                         (MemorySegment/ofArray x) (int 1)
                         beta
                         (MemorySegment/ofArray y) (int 1)])
  y)

;; ================================================================
;; cblas_dger — rank-1 update: A += alpha * x * y^T
;; ================================================================

(def ^:private dger-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; Order
                ValueLayout/JAVA_INT    ;; M
                ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_DOUBLE ;; alpha
                ValueLayout/ADDRESS     ;; x
                ValueLayout/JAVA_INT    ;; incx
                ValueLayout/ADDRESS     ;; y
                ValueLayout/JAVA_INT    ;; incy
                ValueLayout/ADDRESS     ;; A
                ValueLayout/JAVA_INT])))  ;; lda

(def ^:private dger-mh (delay (make-handle "cblas_dger" dger-fd)))

(deftm ^:no-inline dger!
  "A += alpha * x * y^T.  A is [m,n] row-major, x is [m], y is [n]."
  [x :- (Array double) y :- (Array double) A :- (Array double)
   m :- Long n :- Long alpha :- Double]
  :- (Array double)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @dger-mh
                        [CBLAS_ROW_MAJOR
                         (int m) (int n)
                         alpha
                         (MemorySegment/ofArray x) (int 1)
                         (MemorySegment/ofArray y) (int 1)
                         (MemorySegment/ofArray A) (int n)])
  A)

;; ================================================================
;; cblas_sgemv — single-precision matrix-vector multiply
;; ================================================================

(def ^:private sgemv-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; Order
                ValueLayout/JAVA_INT    ;; Trans
                ValueLayout/JAVA_INT    ;; M
                ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_FLOAT  ;; alpha
                ValueLayout/ADDRESS     ;; A
                ValueLayout/JAVA_INT    ;; lda
                ValueLayout/ADDRESS     ;; x
                ValueLayout/JAVA_INT    ;; incx
                ValueLayout/JAVA_FLOAT  ;; beta
                ValueLayout/ADDRESS     ;; y
                ValueLayout/JAVA_INT])))  ;; incy

(def ^:private sgemv-mh (delay (make-handle "cblas_sgemv" sgemv-fd)))

(deftm ^:no-inline dgemv!
  [A :- (Array float) x :- (Array float) y :- (Array float)
   m :- Long n :- Long alpha :- Float beta :- Float]
  :- (Array float)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemv-mh
                        [CBLAS_ROW_MAJOR CBLAS_NO_TRANS
                         (int m) (int n)
                         alpha
                         (MemorySegment/ofArray A) (int n)
                         (MemorySegment/ofArray x) (int 1)
                         beta
                         (MemorySegment/ofArray y) (int 1)])
  y)

(deftm ^:no-inline dgemv-t!
  "y = alpha * A^T @ x + beta * y.  A is [m,n] row-major, float[]."
  [A :- (Array float) x :- (Array float) y :- (Array float)
   m :- Long n :- Long alpha :- Float beta :- Float]
  :- (Array float)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @sgemv-mh
                        [CBLAS_ROW_MAJOR CBLAS_TRANS
                         (int m) (int n)
                         alpha
                         (MemorySegment/ofArray A) (int n)
                         (MemorySegment/ofArray x) (int 1)
                         beta
                         (MemorySegment/ofArray y) (int 1)])
  y)

;; ================================================================
;; cblas_sger — single-precision rank-1 update: A += alpha * x * y^T
;; ================================================================

(def ^:private sger-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; Order
                ValueLayout/JAVA_INT    ;; M
                ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_FLOAT  ;; alpha
                ValueLayout/ADDRESS     ;; x
                ValueLayout/JAVA_INT    ;; incx
                ValueLayout/ADDRESS     ;; y
                ValueLayout/JAVA_INT    ;; incy
                ValueLayout/ADDRESS     ;; A
                ValueLayout/JAVA_INT])))  ;; lda

(def ^:private sger-mh (delay (make-handle "cblas_sger" sger-fd)))

(deftm ^:no-inline dger!
  "A += alpha * x * y^T.  A is [m,n] row-major, x is [m], y is [n]. float[]."
  [x :- (Array float) y :- (Array float) A :- (Array float)
   m :- Long n :- Long alpha :- Float]
  :- (Array float)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @sger-mh
                        [CBLAS_ROW_MAJOR
                         (int m) (int n)
                         alpha
                         (MemorySegment/ofArray x) (int 1)
                         (MemorySegment/ofArray y) (int 1)
                         (MemorySegment/ofArray A) (int n)])
  A)

;; ================================================================
;; cblas_saxpy — single-precision y += alpha * x
;; ================================================================

(def ^:private saxpy-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_FLOAT  ;; alpha
                ValueLayout/ADDRESS     ;; x
                ValueLayout/JAVA_INT    ;; incx
                ValueLayout/ADDRESS     ;; y
                ValueLayout/JAVA_INT])))  ;; incy

(def ^:private saxpy-mh (delay (make-handle "cblas_saxpy" saxpy-fd)))

(deftm ^:no-inline daxpy!
  "y += alpha * x.  Both float[] vectors have n elements."
  [x :- (Array float) y :- (Array float) n :- Long alpha :- Float]
  :- (Array float)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @saxpy-mh
                        [(int n) alpha
                         (MemorySegment/ofArray x) (int 1)
                         (MemorySegment/ofArray y) (int 1)])
  y)

;; ================================================================
;; cblas_daxpy — y += alpha * x
;; ================================================================

(def ^:private daxpy-fd
  (FunctionDescriptor/ofVoid
   (into-array java.lang.foreign.MemoryLayout
               [ValueLayout/JAVA_INT    ;; N
                ValueLayout/JAVA_DOUBLE ;; alpha
                ValueLayout/ADDRESS     ;; x
                ValueLayout/JAVA_INT    ;; incx
                ValueLayout/ADDRESS     ;; y
                ValueLayout/JAVA_INT])))  ;; incy

(def ^:private daxpy-mh (delay (make-handle "cblas_daxpy" daxpy-fd)))

(deftm ^:no-inline daxpy!
  "y += alpha * x.  Both vectors have n elements."
  [x :- (Array double) y :- (Array double) n :- Long alpha :- Double]
  :- (Array double)
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @daxpy-mh
                        [(int n) alpha
                         (MemorySegment/ofArray x) (int 1)
                         (MemorySegment/ofArray y) (int 1)])
  y)

;; ================================================================
;; Scaled difference: out[i] = scale * (a[i] - b[i])
;; Used by MSE loss backward template.
;; ================================================================

(deftm ^:no-inline daxpy-diff!
  "out[i] = scale * (a[i] - b[i]) for all i in [0, n)."
  [a :- (Array double) b :- (Array double) scale :- Double n :- Long]
  :- (Array double)
  (let [out (double-array n)]
    (dotimes [i n]
      (clojure.core/aset out i (* scale (- (clojure.core/aget a i)
                                           (clojure.core/aget b i)))))
    out))

(deftm ^:no-inline daxpy-diff-into!
  "out[i] = scale * (a[i] - b[i]), writing into pre-allocated out."
  [a :- (Array double) b :- (Array double) scale :- Double n :- Long
   out :- (Array double)]
  :- (Array double)
  (dotimes [i n]
    (clojure.core/aset out i (* scale (- (clojure.core/aget a i)
                                         (clojure.core/aget b i)))))
  out)

;; Float overloads — the MSE-loss backward runs over whatever dtype the loss
;; was computed in, so float training (e.g. LoRA over an f32/quantized base)
;; needs these or reverse-AD's gradient step has no matching method.
(deftm ^:no-inline daxpy-diff!
  [a :- (Array float) b :- (Array float) scale :- Double n :- Long]
  :- (Array float)
  (let [out (float-array n)]
    (dotimes [i n]
      (clojure.core/aset out i (float (* scale (- (clojure.core/aget a i)
                                                  (clojure.core/aget b i))))))
    out))

(deftm ^:no-inline daxpy-diff-into!
  [a :- (Array float) b :- (Array float) scale :- Double n :- Long
   out :- (Array float)]
  :- (Array float)
  (dotimes [i n]
    (clojure.core/aset out i (float (* scale (- (clojure.core/aget a i)
                                                (clojure.core/aget b i))))))
  out)

;; ================================================================
;; Availability check
;; ================================================================

(defn available?
  "Check if BLAS is loaded and functional. Returns backend keyword or false."
  []
  (try
    (let [a (double-array [1 0 0 1])
          b (double-array [2 3])
          c (double-array 2)]
      (dgemv! a b c 2 2 1.0 0.0)
      (when (and (== 2.0 (aget c 0)) (== 3.0 (aget c 1)))
        (backend)))
    (catch Exception _ false)))

;; ================================================================
;; Backward helper for dgemm! (template-based AD codegen)
;; C = alpha * A @ B + beta * C
;; dA = alpha * dC @ B^T, dB = alpha * A^T @ dC
;; ================================================================

(deftm dgemm-backward [dC :- (Array double)
                       A :- (Array double) B :- (Array double)
                       m :- Long k :- Long n :- Long
                       alpha :- Double beta :- Double]
  :- (Array Object)
  (let [;; dA = alpha * dC @ B^T -> [m, k]
        dA (double-array (* m k))
        _ (dgemm-nt! dC B dA m n k alpha 0.0)
        ;; dB = alpha * A^T @ dC -> [k, n]
        dB (double-array (* k n))
        _ (dgemm-tn! A dC dB k m n alpha 0.0)]
    (object-array [dA dB])))

;; ================================================================
;; Template registration for dgemm! — C = alpha * A @ B + beta * C
;; ================================================================

(tmpl/merge-into-template! 'raster.linalg.blas/dgemm!
                           {:params '[A B C m k n alpha beta] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [A B C m k n alpha beta] _result-sym adjoint-sym gensym-fn]
                                        ;; grads-arr is the Object[] bundle from
                                        ;; dgemm-backward — no differentiable tag
                                        ;; (Π ⊥), stays untagged by design.
                                        (let [grads-arr (gensym-fn "dgemm_grads")
                                              dA (gensym-fn "dA" (tmpl/grad-tag A))
                                              dB (gensym-fn "dB" (tmpl/grad-tag B))]
                                          [(update ctx :bindings into
                                                   [grads-arr (list 'raster.linalg.blas/dgemm-backward
                                                                    adjoint-sym A B m k n alpha beta)
                                                    dA (list 'clojure.core/aget grads-arr 0)
                                                    dB (list 'clojure.core/aget grads-arr 1)])
                                           [dA dB nil nil nil nil nil nil]]))})
