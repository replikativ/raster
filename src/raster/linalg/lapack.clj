(ns raster.linalg.lapack
  "Panama FFI bindings to LAPACK via OpenBLAS.

  SVD and eigendecomposition use Fortran LAPACK direct (dgesdd_, dsyevd_).
  QR uses LAPACKE C wrapper (small matrices in randomized SVD).

  For SVD, row-major input is transposed to column-major before calling
  dgesdd_ with original (m,n) dimensions. This ensures LAPACK's optimized
  m>=n code path is used for tall matrices. Results are transposed back.
  For symmetric eigendecomposition, row-major = col-major, so no transpose.

  Set raster.openblas.path system property to override the library location.

  All public functions accept flat double[] in row-major order.
  All native calls use critical(true) for zero-copy heap array pinning."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset]]
            [raster.math :as m]
            [raster.numeric :as n])
  (:import [java.lang.foreign
            Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout]))

;; ================================================================
;; Lazy library loading
;; ================================================================

(def ^:private lib-paths
  (let [custom (System/getProperty "raster.openblas.path")
        defaults ["/usr/lib/x86_64-linux-gnu/libopenblas.so"
                  "/lib/x86_64-linux-gnu/libopenblas.so"
                  "/usr/lib64/libopenblas.so"
                  "/usr/lib/libopenblas.so"
                  "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
                  "/usr/local/opt/openblas/lib/libopenblas.dylib"]]
    (if custom (into [custom] defaults) defaults)))

(def ^:private lapacke-lib-paths
  ["/usr/lib/x86_64-linux-gnu/liblapacke.so"
   "/usr/lib64/liblapacke.so"
   "/usr/lib/liblapacke.so"])

(defn- find-lib [^String symbol paths]
  (let [loader (SymbolLookup/loaderLookup)]
    (if (.isPresent (.find loader symbol))
      loader
      (some (fn [path]
              (try
                (let [lib (SymbolLookup/libraryLookup path (Arena/global))]
                  (when (.isPresent (.find lib symbol))
                    lib))
                (catch Exception _ nil)))
            paths))))

(def ^:private openblas (delay (find-lib "dgesdd_" lib-paths)))
(def ^:private lapacke  (delay (find-lib "LAPACKE_dgeqrf" (concat lapacke-lib-paths lib-paths))))

(defn available?
  "Returns true if LAPACK (via OpenBLAS) is loaded and functional."
  []
  (boolean @openblas))

(defn- require-lapack! []
  (when-not @openblas
    (throw (ex-info "No LAPACK library found. Install OpenBLAS (apt-get install libopenblas-dev)."
                    {:searched lib-paths}))))

;; ================================================================
;; MethodHandle creation with critical(true)
;; ================================================================

(def ^:private ^"[Ljava.lang.foreign.Linker$Option;"
  critical-opts
  (into-array Linker$Option [(Linker$Option/critical true)]))

(defn- lookup-symbol ^MemorySegment [^SymbolLookup lib ^String name]
  (let [opt (.find lib name)]
    (when-not (.isPresent opt)
      (throw (ex-info (str "Symbol not found: " name) {:symbol name})))
    (.get opt)))

(defn- make-handle
  [^SymbolLookup lib ^String symbol-name ^FunctionDescriptor fd]
  (when-not lib
    (throw (ex-info (str "No LAPACK library found for " symbol-name ". Install OpenBLAS.")
                    {:symbol symbol-name :searched lib-paths})))
  (.downcallHandle (Linker/nativeLinker)
                   (lookup-symbol lib symbol-name)
                   fd
                   critical-opts))

;; ================================================================
;; Helpers for Fortran pass-by-reference
;; ================================================================

(defn- int-seg ^MemorySegment [^long v]
  (MemorySegment/ofArray (int-array [(int v)])))

(defn- byte-seg ^MemorySegment [^long v]
  (MemorySegment/ofArray (byte-array [(byte v)])))

;; ================================================================
;; Row-major <-> Column-major transpose
;; ================================================================

(deftm transpose-to-col!
  "Transpose row-major A[m,n] to column-major A-col[m*n].
  A-col[i + j*m] = A[i*n + j]."
  [A :- (Array double) A-col :- (Array double) m :- Long n :- Long]
  :- (Array double)
  (dotimes [i m]
    (dotimes [j n]
      (aset A-col (+ i (* j m)) (aget A (+ (* i n) j)))))
  A-col)

(deftm transpose-to-row!
  "Transpose column-major A-col[m,n] (stored as m*n flat, col-major)
  to row-major A-row[m,n]. A-row[i*n + j] = A-col[i + j*m]."
  [A-col :- (Array double) A-row :- (Array double) m :- Long n :- Long]
  :- (Array double)
  (dotimes [i m]
    (dotimes [j n]
      (aset A-row (+ (* i n) j) (aget A-col (+ i (* j m))))))
  A-row)

;; ================================================================
;; Thread control — auto-optimize on first LAPACK call
;; ================================================================

(def ^:private set-threads-fd
  (FunctionDescriptor/ofVoid (into-array MemoryLayout [ValueLayout/JAVA_INT])))

(def ^:private get-threads-fd
  (FunctionDescriptor/of ValueLayout/JAVA_INT (into-array MemoryLayout [])))

(defn- make-handle-noncritical
  [^SymbolLookup lib ^String symbol-name ^FunctionDescriptor fd]
  (.downcallHandle (Linker/nativeLinker)
                   (lookup-symbol lib symbol-name)
                   fd
                   (into-array Linker$Option [])))

(def ^:private set-threads-mh
  (delay (make-handle-noncritical @openblas "openblas_set_num_threads" set-threads-fd)))

(def ^:private get-threads-mh
  (delay (make-handle-noncritical @openblas "openblas_get_num_threads" get-threads-fd)))

(defn get-num-threads
  "Get current OpenBLAS thread count."
  ^long []
  (long (.invokeWithArguments ^java.lang.invoke.MethodHandle @get-threads-mh [])))

(defn set-num-threads!
  "Set OpenBLAS thread count. For LAPACK on modern CPUs,
  half the physical cores often outperforms using all cores."
  [^long n]
  (.invokeWithArguments ^java.lang.invoke.MethodHandle @set-threads-mh [(int n)]))

(defn optimize-threads!
  "Set OpenBLAS threads to half the available processors (min 1, max 4).
  Multi-threaded LAPACK often underperforms on modern hybrid CPUs
  when using all cores due to thread synchronization overhead."
  []
  (let [n (max 1 (min 4 (quot (.availableProcessors (Runtime/getRuntime)) 2)))]
    (set-num-threads! n)
    n))

(def ^:private _init
  (delay
    (let [n (optimize-threads!)]
      (when (System/getProperty "raster.debug")
        (println "raster.lapack: set OpenBLAS threads to" n))
      n)))

;; ================================================================
;; dgesdd_ — Divide-and-conquer SVD (Fortran direct)
;;
;; Strategy: transpose row-major A[m,n] to column-major, call dgesdd_
;; with original (m,n). This hits LAPACK's optimized m>=n path for
;; tall matrices. Results (U, Vt in col-major) are transposed back
;; to row-major.
;; ================================================================

(def ^:private dgesdd-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                ValueLayout/ADDRESS ValueLayout/ADDRESS])))

(def ^:private dgesdd-mh (delay (make-handle @openblas "dgesdd_" dgesdd-fd)))

(deftm dgesdd!
  "Thin SVD via divide-and-conquer. A is [m,n] row-major, overwritten.
  U is [m,k] row-major, S is [k], Vt is [k,n] row-major, k=min(m,n).
  Returns LAPACK info (0 = success).

  Transposes to column-major for LAPACK, then transposes results back.
  This ensures the optimized m>=n path is used for tall matrices."
  [A :- (Array double) U :- (Array double) S :- (Array double)
   Vt :- (Array double) m :- Long n :- Long] :- Long
  (let [_ @_init
        k (min m n)
        ;; Transpose A to column-major
        A-col (double-array (* m n))
        _ (transpose-to-col! A A-col m n)
        ;; Column-major output buffers
        U-col  (double-array (* m k))
        Vt-col (double-array (* k n))
        work-query (double-array 1)
        iwork (int-array (* 8 k))
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dgesdd-mh]
    ;; Workspace query
    (.invokeWithArguments mh
                          [(byte-seg (int \S))
                           (int-seg m) (int-seg n)                               ;; original dims
                           (MemorySegment/ofArray A-col) (int-seg m)             ;; lda = m (col-major)
                           (MemorySegment/ofArray S)
                           (MemorySegment/ofArray U-col) (int-seg m)             ;; ldu = m
                           (MemorySegment/ofArray Vt-col) (int-seg k)            ;; ldvt = k
                           (MemorySegment/ofArray work-query) (int-seg -1)
                           (MemorySegment/ofArray iwork)
                           (MemorySegment/ofArray info-out)])
    (let [lwork (long (m/ceil (aget work-query 0)))
          work (double-array lwork)]
      (.invokeWithArguments mh
                            [(byte-seg (int \S))
                             (int-seg m) (int-seg n)
                             (MemorySegment/ofArray A-col) (int-seg m)
                             (MemorySegment/ofArray S)
                             (MemorySegment/ofArray U-col) (int-seg m)
                             (MemorySegment/ofArray Vt-col) (int-seg k)
                             (MemorySegment/ofArray work) (int-seg lwork)
                             (MemorySegment/ofArray iwork)
                             (MemorySegment/ofArray info-out)])
      ;; Transpose results back to row-major
      ;; U-col is m×k col-major -> U is m×k row-major
      (transpose-to-row! U-col U m k)
      ;; Vt-col is k×n col-major -> Vt is k×n row-major
      (transpose-to-row! Vt-col Vt k n)
      (long (clojure.core/aget info-out 0)))))

(deftm dgesdd-values!
  "SVD singular values only (no U/Vt). A is [m,n] row-major, overwritten.
  S is [min(m,n)] filled with singular values in descending order.
  Returns LAPACK info (0 = success).

  Uses jobz='N' — skips U/Vt computation and all transposes."
  [A :- (Array double) S :- (Array double)
   m :- Long n :- Long] :- Long
  (let [_ @_init
        k (min m n)
        ;; Transpose A to column-major
        A-col (double-array (* m n))
        _ (transpose-to-col! A A-col m n)
        ;; Dummy U/Vt — LAPACK requires non-null but jobz='N' ignores them
        dummy (double-array 1)
        work-query (double-array 1)
        iwork (int-array (* 8 k))
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dgesdd-mh]
    ;; Workspace query
    (.invokeWithArguments mh
                          [(byte-seg (int \N))
                           (int-seg m) (int-seg n)
                           (MemorySegment/ofArray A-col) (int-seg m)
                           (MemorySegment/ofArray S)
                           (MemorySegment/ofArray dummy) (int-seg 1)
                           (MemorySegment/ofArray dummy) (int-seg 1)
                           (MemorySegment/ofArray work-query) (int-seg -1)
                           (MemorySegment/ofArray iwork)
                           (MemorySegment/ofArray info-out)])
    (let [lwork (long (m/ceil (aget work-query 0)))
          work (double-array lwork)]
      (.invokeWithArguments mh
                            [(byte-seg (int \N))
                             (int-seg m) (int-seg n)
                             (MemorySegment/ofArray A-col) (int-seg m)
                             (MemorySegment/ofArray S)
                             (MemorySegment/ofArray dummy) (int-seg 1)
                             (MemorySegment/ofArray dummy) (int-seg 1)
                             (MemorySegment/ofArray work) (int-seg lwork)
                             (MemorySegment/ofArray iwork)
                             (MemorySegment/ofArray info-out)])
      (long (clojure.core/aget info-out 0)))))

;; ================================================================
;; dsyevd_ — Symmetric eigendecomposition (Fortran direct)
;; ================================================================

(def ^:private dsyevd-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                ValueLayout/ADDRESS ValueLayout/ADDRESS])))

(def ^:private dsyevd-mh (delay (make-handle @openblas "dsyevd_" dsyevd-fd)))

(deftm dsyevd!
  "Symmetric eigendecomposition. A is [n,n] row-major symmetric,
  overwritten with eigenvectors. eigenvalues[n] filled ascending.
  Returns LAPACK info (0 = success).

  Symmetric matrices have identical row-major and column-major layout,
  so zero transposition overhead."
  [A :- (Array double) eigenvalues :- (Array double) n :- Long] :- Long
  (let [_ @_init
        info-out (int-array 1)
        work-query (double-array 1)
        iwork-query (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dsyevd-mh]
    ;; Workspace query
    (.invokeWithArguments mh
                          [(byte-seg (int \V)) (byte-seg (int \U))
                           (int-seg n)
                           (MemorySegment/ofArray A) (int-seg n)
                           (MemorySegment/ofArray eigenvalues)
                           (MemorySegment/ofArray work-query)  (int-seg -1)
                           (MemorySegment/ofArray iwork-query) (int-seg -1)
                           (MemorySegment/ofArray info-out)])
    (let [lwork (long (m/ceil (aget work-query 0)))
          liwork (long (clojure.core/aget iwork-query 0))
          work (double-array lwork)
          iwork (int-array liwork)]
      (.invokeWithArguments mh
                            [(byte-seg (int \V)) (byte-seg (int \U))
                             (int-seg n)
                             (MemorySegment/ofArray A) (int-seg n)
                             (MemorySegment/ofArray eigenvalues)
                             (MemorySegment/ofArray work) (int-seg lwork)
                             (MemorySegment/ofArray iwork) (int-seg liwork)
                             (MemorySegment/ofArray info-out)])
      (long (clojure.core/aget info-out 0)))))

;; ================================================================
;; LAPACKE_dgeqrf / LAPACKE_dorgqr — QR factorization (C wrapper)
;; Used for randomized SVD power iterations where matrices are small.
;; ================================================================

(def ^:private ^:const LAPACK_ROW_MAJOR (int 101))

(def ^:private dgeqrf-fd
  (FunctionDescriptor/of
   ValueLayout/JAVA_INT
   (into-array MemoryLayout
               [ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/ADDRESS  ValueLayout/JAVA_INT ValueLayout/ADDRESS])))

(def ^:private dgeqrf-mh (delay (make-handle @lapacke "LAPACKE_dgeqrf" dgeqrf-fd)))

(deftm dgeqrf!
  "QR factorization via LAPACKE (row-major). A[m,n] overwritten with
  R (upper) and Householder reflectors (lower). tau[min(m,n)]."
  [A :- (Array double) tau :- (Array double) m :- Long n :- Long] :- Long
  (long (.invokeWithArguments ^java.lang.invoke.MethodHandle @dgeqrf-mh
                              [LAPACK_ROW_MAJOR (int m) (int n)
                               (MemorySegment/ofArray A) (int n)
                               (MemorySegment/ofArray tau)])))

(def ^:private dorgqr-fd
  (FunctionDescriptor/of
   ValueLayout/JAVA_INT
   (into-array MemoryLayout
               [ValueLayout/JAVA_INT ValueLayout/JAVA_INT ValueLayout/JAVA_INT
                ValueLayout/JAVA_INT ValueLayout/ADDRESS  ValueLayout/JAVA_INT
                ValueLayout/ADDRESS])))

(def ^:private dorgqr-mh (delay (make-handle @lapacke "LAPACKE_dorgqr" dorgqr-fd)))

(deftm dorgqr!
  "Generate thin Q[m,n] from Householder reflectors via LAPACKE (row-major)."
  [A :- (Array double) tau :- (Array double)
   m :- Long n :- Long k :- Long] :- Long
  (long (.invokeWithArguments ^java.lang.invoke.MethodHandle @dorgqr-mh
                              [LAPACK_ROW_MAJOR (int m) (int n) (int k)
                               (MemorySegment/ofArray A) (int n)
                               (MemorySegment/ofArray tau)])))

;; ================================================================
;; Thread & availability API
;; ================================================================

;; ================================================================
;; dgesv_ — General linear solve Ax=b (Fortran direct)
;;
;; Solves by LU factorization with partial pivoting.
;; General matrices need transpose to column-major.
;; ================================================================

(def ^:private dgesv-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; N, NRHS
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; A, LDA
                ValueLayout/ADDRESS                      ;; IPIV
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; B, LDB
                ValueLayout/ADDRESS])))                  ;; INFO

(def ^:private dgesv-mh (delay (make-handle @openblas "dgesv_" dgesv-fd)))

(deftm dgesv!
  "Solve Ax=B via LU factorization. A[n,n] and B[n,nrhs] row-major,
  both overwritten. ipiv[n] receives pivot indices.
  Returns LAPACK info (0 = success)."
  [A :- (Array double) B :- (Array double) ipiv :- (Array int)
   n :- Long nrhs :- Long] :- Long
  (let [_ @_init
        A-col (double-array (* n n))
        _ (transpose-to-col! A A-col n n)
        B-col (double-array (* n nrhs))
        _ (transpose-to-col! B B-col n nrhs)
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dgesv-mh]
    (.invokeWithArguments mh
                          [(int-seg n) (int-seg nrhs)
                           (MemorySegment/ofArray A-col) (int-seg n)
                           (MemorySegment/ofArray ipiv)
                           (MemorySegment/ofArray B-col) (int-seg n)
                           (MemorySegment/ofArray info-out)])
    (transpose-to-row! B-col B n nrhs)
    (long (clojure.core/aget info-out 0))))

;; ================================================================
;; dpotrf_ — Cholesky factorization (Fortran direct)
;;
;; Symmetric positive definite — row-major = col-major for symmetric.
;; ================================================================

(def ^:private dpotrf-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; UPLO, N
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; A, LDA
                ValueLayout/ADDRESS])))                  ;; INFO

(def ^:private dpotrf-mh (delay (make-handle @openblas "dpotrf_" dpotrf-fd)))

(deftm dpotrf!
  "Cholesky factorization of symmetric positive definite matrix.
  A[n,n] row-major symmetric, overwritten with lower triangular factor L.
  Returns LAPACK info (0 = success, >0 = not positive definite)."
  [A :- (Array double) n :- Long] :- Long
  (let [_ @_init
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dpotrf-mh]
    (.invokeWithArguments mh
                          [(byte-seg (int \L)) (int-seg n)
                           (MemorySegment/ofArray A) (int-seg n)
                           (MemorySegment/ofArray info-out)])
    (long (clojure.core/aget info-out 0))))

;; ================================================================
;; dpotrs_ — Cholesky solve (Fortran direct)
;;
;; Solves Ax=B given Cholesky factor L from dpotrf_.
;; Symmetric — no transpose needed.
;; ================================================================

(def ^:private dpotrs-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; UPLO, N
                ValueLayout/ADDRESS                      ;; NRHS
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; A, LDA
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; B, LDB
                ValueLayout/ADDRESS])))                  ;; INFO

(def ^:private dpotrs-mh (delay (make-handle @openblas "dpotrs_" dpotrs-fd)))

(deftm dpotrs!
  "Solve Ax=B using Cholesky factor L (from dpotrf!).
  A[n,n] contains L (lower), B[n,nrhs] row-major overwritten with solution.
  Returns LAPACK info (0 = success)."
  [A :- (Array double) B :- (Array double)
   n :- Long nrhs :- Long] :- Long
  (let [_ @_init
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dpotrs-mh]
    (if (== nrhs 1)
      ;; nrhs=1: column vector is identical in row/col-major, skip transpose
      (do (.invokeWithArguments mh
                                [(byte-seg (int \L)) (int-seg n)
                                 (int-seg 1)
                                 (MemorySegment/ofArray A) (int-seg n)
                                 (MemorySegment/ofArray B) (int-seg n)
                                 (MemorySegment/ofArray info-out)])
          (long (clojure.core/aget info-out 0)))
      ;; nrhs>1: need transpose for column-major layout
      (let [B-col (double-array (* n nrhs))
            _ (transpose-to-col! B B-col n nrhs)]
        (.invokeWithArguments mh
                              [(byte-seg (int \L)) (int-seg n)
                               (int-seg nrhs)
                               (MemorySegment/ofArray A) (int-seg n)
                               (MemorySegment/ofArray B-col) (int-seg n)
                               (MemorySegment/ofArray info-out)])
        (transpose-to-row! B-col B n nrhs)
        (long (clojure.core/aget info-out 0))))))

(deftm dpotrs-upper!
  "Solve Ax=B using upper Cholesky factor (column-major upper = row-major lower).
  A[n,n] contains U (upper in col-major), B[n,nrhs] overwritten with solution.
  Returns LAPACK info (0 = success)."
  [A :- (Array double) B :- (Array double)
   n :- Long nrhs :- Long] :- Long
  (let [_ @_init
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dpotrs-mh]
    (if (== nrhs 1)
      ;; nrhs=1: column vector is identical in row/col-major, skip transpose
      (do (.invokeWithArguments mh
                                [(byte-seg (int \U)) (int-seg n)
                                 (int-seg 1)
                                 (MemorySegment/ofArray A) (int-seg n)
                                 (MemorySegment/ofArray B) (int-seg n)
                                 (MemorySegment/ofArray info-out)])
          (long (clojure.core/aget info-out 0)))
      ;; nrhs>1: need transpose for column-major layout
      (let [B-col (double-array (* n nrhs))
            _ (transpose-to-col! B B-col n nrhs)]
        (.invokeWithArguments mh
                              [(byte-seg (int \U)) (int-seg n)
                               (int-seg nrhs)
                               (MemorySegment/ofArray A) (int-seg n)
                               (MemorySegment/ofArray B-col) (int-seg n)
                               (MemorySegment/ofArray info-out)])
        (transpose-to-row! B-col B n nrhs)
        (long (clojure.core/aget info-out 0))))))

;; ================================================================
;; dgetrf_ — LU factorization (Fortran direct)
;;
;; General matrix — needs transpose to column-major.
;; ================================================================

(def ^:private dgetrf-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; M, N
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; A, LDA
                ValueLayout/ADDRESS                      ;; IPIV
                ValueLayout/ADDRESS])))                  ;; INFO

(def ^:private dgetrf-mh (delay (make-handle @openblas "dgetrf_" dgetrf-fd)))

(deftm dgetrf!
  "LU factorization with partial pivoting. A[m,n] row-major,
  overwritten with L and U factors. ipiv[min(m,n)] receives pivots.
  Returns LAPACK info (0 = success)."
  [A :- (Array double) ipiv :- (Array int)
   m :- Long n :- Long] :- Long
  (let [_ @_init
        A-col (double-array (* m n))
        _ (transpose-to-col! A A-col m n)
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dgetrf-mh]
    (.invokeWithArguments mh
                          [(int-seg m) (int-seg n)
                           (MemorySegment/ofArray A-col) (int-seg m)
                           (MemorySegment/ofArray ipiv)
                           (MemorySegment/ofArray info-out)])
    (transpose-to-row! A-col A m n)
    (long (clojure.core/aget info-out 0))))

;; ================================================================
;; dgetrs_ — LU solve (Fortran direct)
;;
;; Solves Ax=B given LU factors from dgetrf_.
;; General — needs transpose.
;; ================================================================

(def ^:private dgetrs-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; TRANS, N
                ValueLayout/ADDRESS                      ;; NRHS
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; A, LDA
                ValueLayout/ADDRESS                      ;; IPIV
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; B, LDB
                ValueLayout/ADDRESS])))                  ;; INFO

(def ^:private dgetrs-mh (delay (make-handle @openblas "dgetrs_" dgetrs-fd)))

(deftm dgetrs!
  "Solve Ax=B using LU factors (from dgetrf!).
  A[n,n] contains LU, B[n,nrhs] row-major overwritten with solution.
  Returns LAPACK info (0 = success)."
  [A :- (Array double) ipiv :- (Array int) B :- (Array double)
   n :- Long nrhs :- Long] :- Long
  (let [_ @_init
        ;; A is already in row-major LU form — need to transpose for Fortran
        A-col (double-array (* n n))
        _ (transpose-to-col! A A-col n n)
        B-col (double-array (* n nrhs))
        _ (transpose-to-col! B B-col n nrhs)
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dgetrs-mh]
    (.invokeWithArguments mh
                          [(byte-seg (int \N)) (int-seg n)
                           (int-seg nrhs)
                           (MemorySegment/ofArray A-col) (int-seg n)
                           (MemorySegment/ofArray ipiv)
                           (MemorySegment/ofArray B-col) (int-seg n)
                           (MemorySegment/ofArray info-out)])
    (transpose-to-row! B-col B n nrhs)
    (long (clojure.core/aget info-out 0))))

;; ================================================================
;; dgetri_ — Matrix inverse from LU (Fortran direct)
;;
;; Computes inverse using LU from dgetrf_.
;; General — needs transpose.
;; ================================================================

(def ^:private dgetri-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS                      ;; N
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; A, LDA
                ValueLayout/ADDRESS                      ;; IPIV
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; WORK, LWORK
                ValueLayout/ADDRESS])))                  ;; INFO

(def ^:private dgetri-mh (delay (make-handle @openblas "dgetri_" dgetri-fd)))

(deftm dgetri!
  "Compute matrix inverse from LU factors (from dgetrf!).
  A[n,n] row-major overwritten with inverse.
  Returns LAPACK info (0 = success)."
  [A :- (Array double) ipiv :- (Array int) n :- Long] :- Long
  (let [_ @_init
        A-col (double-array (* n n))
        _ (transpose-to-col! A A-col n n)
        work-query (double-array 1)
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dgetri-mh]
    ;; Workspace query
    (.invokeWithArguments mh
                          [(int-seg n)
                           (MemorySegment/ofArray A-col) (int-seg n)
                           (MemorySegment/ofArray ipiv)
                           (MemorySegment/ofArray work-query) (int-seg -1)
                           (MemorySegment/ofArray info-out)])
    (let [lwork (long (m/ceil (aget work-query 0)))
          work (double-array lwork)]
      (.invokeWithArguments mh
                            [(int-seg n)
                             (MemorySegment/ofArray A-col) (int-seg n)
                             (MemorySegment/ofArray ipiv)
                             (MemorySegment/ofArray work) (int-seg lwork)
                             (MemorySegment/ofArray info-out)])
      (transpose-to-row! A-col A n n)
      (long (clojure.core/aget info-out 0)))))

;; ================================================================
;; dgels_ — Least squares solve (Fortran direct)
;;
;; Solves overdetermined/underdetermined least squares.
;; General — needs transpose.
;; ================================================================

(def ^:private dgels-fd
  (FunctionDescriptor/ofVoid
   (into-array MemoryLayout
               [ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; TRANS, M
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; N, NRHS
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; A, LDA
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; B, LDB
                ValueLayout/ADDRESS ValueLayout/ADDRESS  ;; WORK, LWORK
                ValueLayout/ADDRESS])))                  ;; INFO

(def ^:private dgels-mh (delay (make-handle @openblas "dgels_" dgels-fd)))

(deftm dgels!
  "Least squares solve via QR/LQ. A[m,n] row-major, B[max(m,n),nrhs].
  A overwritten with factorization, B overwritten with solution.
  Returns LAPACK info (0 = success)."
  [A :- (Array double) B :- (Array double)
   m :- Long n :- Long nrhs :- Long] :- Long
  (let [_ @_init
        ldb (max m n)
        A-col (double-array (* m n))
        _ (transpose-to-col! A A-col m n)
        B-col (double-array (* ldb nrhs))
        _ (transpose-to-col! B B-col ldb nrhs)
        work-query (double-array 1)
        info-out (int-array 1)
        ^java.lang.invoke.MethodHandle mh @dgels-mh]
    ;; Workspace query
    (.invokeWithArguments mh
                          [(byte-seg (int \N)) (int-seg m)
                           (int-seg n) (int-seg nrhs)
                           (MemorySegment/ofArray A-col) (int-seg m)
                           (MemorySegment/ofArray B-col) (int-seg ldb)
                           (MemorySegment/ofArray work-query) (int-seg -1)
                           (MemorySegment/ofArray info-out)])
    (let [lwork (long (m/ceil (aget work-query 0)))
          work (double-array lwork)]
      (.invokeWithArguments mh
                            [(byte-seg (int \N)) (int-seg m)
                             (int-seg n) (int-seg nrhs)
                             (MemorySegment/ofArray A-col) (int-seg m)
                             (MemorySegment/ofArray B-col) (int-seg ldb)
                             (MemorySegment/ofArray work) (int-seg lwork)
                             (MemorySegment/ofArray info-out)])
      (transpose-to-row! B-col B ldb nrhs)
      (long (clojure.core/aget info-out 0)))))

;; ================================================================
;; Thread & availability API
;; ================================================================

(defn available?
  "Check if OpenBLAS LAPACK is loaded and functional."
  []
  (try
    @_init
    (let [a (double-array [2 1 1 2])
          w (double-array 2)]
      (== 0 (dsyevd! a w 2)))
    (catch Exception _ false)))
