(ns raster.linalg.native-library
  "Admission checks shared by the LP64 BLAS and LAPACK Panama bindings.
   Absence of provider metadata is not evidence of a verified integer ABI."
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout]))

(def openblas-paths
  "Conventional discovery paths; an explicit path never falls back to these."
  ["/usr/lib/x86_64-linux-gnu/libopenblas.so"
   "/lib/x86_64-linux-gnu/libopenblas.so"
   "/usr/lib64/libopenblas.so"
   "/usr/lib/libopenblas.so"
   "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
   "/usr/local/opt/openblas/lib/libopenblas.dylib"])

(defn explicit-path [] (System/getProperty "raster.openblas.path"))

(defn- load-library [^String path]
  (SymbolLookup/libraryLookup path (Arena/global)))

(defn- loader-lookup [] (SymbolLookup/loaderLookup))

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

(defn- check-lp64-config! [config]
  (when (and config (re-find #"(?:^|\s)USE64BITINT(?:\s|$)" config))
    (throw (ex-info "Raster BLAS/LAPACK bindings require LP64 (32-bit integers); OpenBLAS is ILP64"
                    {:reason :native-integer-abi-mismatch
                     :expected :lp64 :actual :ilp64
                     :provider :openblas :configuration config}))))

(defn require-lp64!
  "Reject known ILP64 OpenBLAS before calling bindings with JAVA_INT slots.
   Return the original lookup. Unknown providers retain existing admission;
   this check does not certify their ABI or resolve symbol interposition."
  [library]
  (check-lp64-config! (openblas-config library))
  library)

(defn find-library
  "Resolve a required numerical symbol. An explicit OpenBLAS path is a pin:
   load or capability errors never fall back to preloaded/default providers.
   Read the property lazily, at the caller's first delayed library selection.
   The three-argument form uses an already captured path (nil means discovery).
   Default discovery preserves compatibility with metadata-absent providers."
  ([required-symbol paths]
   (find-library required-symbol paths (explicit-path)))
  ([^String required-symbol paths selected-path]
   (if-some [path selected-path]
     (let [library (try (load-library path)
                        (catch Exception e
                          (throw (ex-info "Cannot load explicitly selected OpenBLAS"
                                          {:reason :explicit-native-library-unavailable
                                           :path path :symbol required-symbol} e))))]
       (when-not (.isPresent (.find ^SymbolLookup library required-symbol))
         (throw (ex-info "Explicit OpenBLAS lacks the required numerical symbol"
                         {:reason :explicit-native-symbol-unavailable
                          :path path :symbol required-symbol})))
       (let [config (openblas-config library)]
         (when-not (seq config)
           (throw (ex-info "Explicit OpenBLAS has no build metadata for integer ABI admission"
                           {:reason :explicit-native-provider-unverified
                            :path path :symbol required-symbol})))
         (check-lp64-config! config))
       library)
     (let [loader (loader-lookup)]
       (if (.isPresent (.find ^SymbolLookup loader required-symbol))
         loader
         (some (fn [path]
                 (when-let [library (try (load-library path) (catch Exception _ nil))]
                   (when (.isPresent (.find ^SymbolLookup library required-symbol)) library)))
               paths))))))
