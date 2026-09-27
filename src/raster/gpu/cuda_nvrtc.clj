(ns raster.gpu.cuda-nvrtc
  "Hardware-free CUDA C++ to PTX compilation through NVRTC. This is a transport
   for generated KernelArtifact source, not a resident CUDA session or a second emitter."
  (:require [raster.compiler.ir.kernel-artifact :as artifact])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]
           [java.nio.charset StandardCharsets]))

(def ^:private library-names
  ["libnvrtc.so" "libnvrtc.so.12" "libnvrtc.so.13"])

(defn- find-library
  []
  (or (some (fn [library]
              (try
                (let [lookup (SymbolLookup/libraryLookup library (Arena/global))]
                  (when (.isPresent (.find lookup "nvrtcVersion")) lookup))
                (catch Exception _ nil)))
            library-names)
      (throw (ex-info "NVRTC library is unavailable"
                      {:reason :nvrtc-unavailable :searched library-names}))))

(def ^:private library (delay (find-library)))

(defn available?
  "Whether the NVRTC shared library can be loaded; no CUDA device is required."
  []
  (try @library true
       (catch Exception _ false)))

(defn- downcall
  ^MethodHandle [symbol return arguments]
  (let [found (.find ^SymbolLookup @library symbol)]
    (when-not (.isPresent found)
      (throw (ex-info "NVRTC symbol is unavailable"
                      {:reason :nvrtc-symbol-unavailable :symbol symbol})))
    (.downcallHandle (Linker/nativeLinker)
                     (.get found)
                     (FunctionDescriptor/of return (into-array MemoryLayout arguments))
                     (into-array Linker$Option []))))

(def ^:private handles
  (delay
    {:version (downcall "nvrtcVersion" ValueLayout/JAVA_INT
                        [ValueLayout/ADDRESS ValueLayout/ADDRESS])
     :create (downcall "nvrtcCreateProgram" ValueLayout/JAVA_INT
                       [ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/ADDRESS
                        ValueLayout/JAVA_INT ValueLayout/ADDRESS ValueLayout/ADDRESS])
     :compile (downcall "nvrtcCompileProgram" ValueLayout/JAVA_INT
                        [ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/ADDRESS])
     :destroy (downcall "nvrtcDestroyProgram" ValueLayout/JAVA_INT
                        [ValueLayout/ADDRESS])
     :ptx-size (downcall "nvrtcGetPTXSize" ValueLayout/JAVA_INT
                         [ValueLayout/ADDRESS ValueLayout/ADDRESS])
     :ptx (downcall "nvrtcGetPTX" ValueLayout/JAVA_INT
                    [ValueLayout/ADDRESS ValueLayout/ADDRESS])
     :log-size (downcall "nvrtcGetProgramLogSize" ValueLayout/JAVA_INT
                         [ValueLayout/ADDRESS ValueLayout/ADDRESS])
     :log (downcall "nvrtcGetProgramLog" ValueLayout/JAVA_INT
                    [ValueLayout/ADDRESS ValueLayout/ADDRESS])}))

(def ^:private c-headers
  ;; NVRTC does not inherit nvcc's host C include search path. Raster's CUDA
  ;; preamble uses these two C headers, but feeding glibc headers to NVRTC
  ;; drags in host ABI declarations that its device compiler cannot parse.
  ;; CUDA math declarations come from cuda_runtime.h; the integer aliases are
  ;; the fixed-width C99 types used by the emitted CUDA dialect.
  [["stdint.h" (str "#ifndef RSTR_NVRTC_STDINT_H\n#define RSTR_NVRTC_STDINT_H\n"
                     "typedef signed char int8_t; typedef unsigned char uint8_t;\n"
                     "typedef short int16_t; typedef unsigned short uint16_t;\n"
                     "typedef int int32_t; typedef unsigned int uint32_t;\n"
                     "typedef long int64_t; typedef unsigned long uint64_t;\n"
                     "#endif\n")]
   ["math.h" "/* CUDA math declarations are supplied by cuda_runtime.h. */\n"]])

(def virtual-header-version
  "Cache identity for Raster's NVRTC C-header adaptation."
  2)

(defn- pointer-array
  ^MemorySegment [^Arena arena strings]
  (let [pointers (.allocate arena (* (count strings) (.byteSize ValueLayout/ADDRESS))
                            (.byteAlignment ValueLayout/ADDRESS))]
    (doseq [[index value] (map-indexed vector strings)]
      (.set pointers ValueLayout/ADDRESS
            (* index (.byteSize ValueLayout/ADDRESS))
            (.allocateFrom arena ^String value)))
    pointers))

(defn- invoke
  [^MethodHandle handle & arguments]
  (.invokeWithArguments handle (into-array Object arguments)))

(defn- check!
  [operation result]
  (when-not (zero? (int result))
    (throw (ex-info "NVRTC operation failed"
                    {:reason :nvrtc-error :operation operation :error-code (int result)}))))

(defn version
  "Return the installed NVRTC major/minor without probing a GPU."
  []
  (with-open [arena (Arena/ofConfined)]
    (let [major (.allocate arena ValueLayout/JAVA_INT)
          minor (.allocate arena ValueLayout/JAVA_INT)]
      (check! :version (invoke (:version @handles) major minor))
      [(.get major ValueLayout/JAVA_INT 0)
       (.get minor ValueLayout/JAVA_INT 0)])))

(defn- bytes-from-program
  [^Arena arena program size-handle read-handle operation]
  (let [size-pointer (.allocate arena ValueLayout/JAVA_LONG)]
    (check! (keyword (str (name operation) "-size"))
            (invoke size-handle program size-pointer))
    (let [size (.get size-pointer ValueLayout/JAVA_LONG 0)]
      (when (or (neg? size) (> size Integer/MAX_VALUE))
        (throw (ex-info "NVRTC output exceeds Java array capacity"
                        {:operation operation :bytes size})))
      (let [segment (.allocate arena (max 1 size))
            bytes (byte-array (int size))]
        (check! operation (invoke read-handle program segment))
        (.get (.asByteBuffer segment) bytes)
        bytes))))

(defn- log-string
  [arena program]
  (let [bytes (bytes-from-program arena program (:log-size @handles)
                                  (:log @handles) :log)
        content (if (and (pos? (alength bytes)) (zero? (aget bytes (dec (alength bytes)))))
                  (java.util.Arrays/copyOf bytes (dec (alength bytes)))
                  bytes)]
    (String. ^bytes content StandardCharsets/UTF_8)))

(defn compile-ptx
  "Compile generated CUDA C++ source for a virtual architecture such as `compute_80`.
   Returns PTX bytes and the compiler log. A real device is not needed.
   `:include-paths` are explicit CUDA toolkit header directories, not inferred from a
   device descriptor; they participate in the caller's compilation cache key."
  [source kernel-name virtual-architecture & {:keys [include-paths] :or {include-paths []}}]
  (when-not (and (string? source) (seq source)
                 (string? kernel-name) (seq kernel-name)
                 (string? virtual-architecture)
                 (re-matches #"compute_[0-9]+" virtual-architecture)
                 (vector? include-paths)
                 (every? #(and (string? %) (seq %)) include-paths))
    (throw (ex-info "invalid NVRTC compilation request"
                    {:kernel-name kernel-name :virtual-architecture virtual-architecture})))
  (with-open [arena (Arena/ofConfined)]
    (let [program-pointer (.allocate arena ValueLayout/ADDRESS)
          src (.allocateFrom arena source)
          name (.allocateFrom arena kernel-name)
          options (into [(str "--gpu-architecture=" virtual-architecture) "--std=c++17"]
                        (map #(str "--include-path=" %) include-paths))
          option-pointer (pointer-array arena options)
          header-pointer (pointer-array arena (mapv second c-headers))
          header-name-pointer (pointer-array arena (mapv first c-headers))]
      (check! :create (invoke (:create @handles) program-pointer src name
                              (int (count c-headers)) header-pointer header-name-pointer))
      (try
        (let [program (.get program-pointer ValueLayout/ADDRESS 0)
              result (invoke (:compile @handles) program (int (count options)) option-pointer)
              log (log-string arena program)]
          (when-not (zero? (int result))
            (throw (ex-info "NVRTC compilation failed"
                            {:reason :nvrtc-compilation :kernel-name kernel-name
                             :virtual-architecture virtual-architecture
                             :error-code (int result) :log log})))
          {:ptx (bytes-from-program arena program (:ptx-size @handles)
                                    (:ptx @handles) :ptx)
           :log log
           :virtual-architecture virtual-architecture
           :include-paths include-paths
           :virtual-header-version virtual-header-version
           :nvrtc-version (version)})
        (finally
          (check! :destroy (invoke (:destroy @handles) program-pointer)))))))

(defn compile-artifact-ptx
  "Compile a verified CUDA KernelArtifact through the NVRTC transport."
  [kernel-artifact virtual-architecture & {:keys [include-paths] :or {include-paths []}}]
  (let [kernel-artifact (artifact/validate! kernel-artifact)]
    (when-not (= :cuda-c (:target kernel-artifact))
      (throw (ex-info "NVRTC requires a CUDA C++ KernelArtifact"
                      {:reason :nvrtc-artifact-target
                       :kernel-name (:kernel-name kernel-artifact)
                       :target (:target kernel-artifact)})))
    (assoc (compile-ptx (:source kernel-artifact) (:kernel-name kernel-artifact)
                        virtual-architecture :include-paths include-paths)
           :kernel-name (:kernel-name kernel-artifact)
           :target (:target kernel-artifact))))
