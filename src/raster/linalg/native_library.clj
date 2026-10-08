(ns raster.linalg.native-library
  "Admission checks shared by the LP64 BLAS and LAPACK Panama bindings.
   Absence of provider metadata is not evidence of a verified integer ABI."
  (:import [java.lang.foreign FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout]))

(defn openblas-config
  "Read optional OpenBLAS build metadata without invoking a numerical routine.
   Returns nil for libraries that do not expose openblas_get_config."
  [^SymbolLookup library]
  (when library
    (let [symbol (.find library "openblas_get_config")]
      (when (.isPresent symbol)
        (let [descriptor (FunctionDescriptor/of ValueLayout/ADDRESS
                                                (into-array MemoryLayout []))
              handle (.downcallHandle (Linker/nativeLinker) (.get symbol) descriptor
                                       (into-array Linker$Option []))
              address ^MemorySegment (.invokeWithArguments handle [])]
          (when (zero? (.address address))
            (throw (ex-info "OpenBLAS returned null build metadata"
                            {:reason :invalid-native-library-metadata})))
          ;; OpenBLAS owns this static, NUL-terminated string. Bound the view;
          ;; never close/free provider-owned memory.
          (.getString (.reinterpret address 4096) 0))))))

(defn require-lp64!
  "Reject known ILP64 OpenBLAS before calling bindings with JAVA_INT slots.
   Return the original lookup. Unknown providers retain existing admission;
   this check does not certify their ABI or resolve symbol interposition."
  [library]
  (when-let [config (openblas-config library)]
    (when (re-find #"(?:^|\s)USE64BITINT(?:\s|$)" config)
      (throw (ex-info "Raster BLAS/LAPACK bindings require LP64 (32-bit integers); OpenBLAS is ILP64"
                      {:reason :native-integer-abi-mismatch
                       :expected :lp64 :actual :ilp64
                       :provider :openblas :configuration config}))))
  library)
