(ns raster.gpu.cuda-driver-probe
  "Read-only CUDA Driver API discovery. This does not create a context, allocate
   memory, register a Raster backend, or imply that a device can run kernels."
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]))

(def ^:private library-names ["libcuda.so.1" "libcuda.so"])

(defn- find-library
  []
  (or (some (fn [library]
              (try
                (let [lookup (SymbolLookup/libraryLookup library (Arena/global))]
                  (when (.isPresent (.find lookup "cuInit")) lookup))
                (catch Exception _ nil)))
            library-names)
      (throw (ex-info "CUDA Driver API library is unavailable"
                      {:reason :cuda-driver-unavailable :searched library-names}))))

(def ^:private library (delay (find-library)))

(defn available?
  "Whether the CUDA driver library can be loaded; this is not a device probe."
  []
  (try @library true
       (catch Exception _ false)))

(defn- downcall
  ^MethodHandle [symbol return arguments]
  (let [found (.find ^SymbolLookup @library symbol)]
    (when-not (.isPresent found)
      (throw (ex-info "CUDA Driver API symbol is unavailable"
                      {:reason :cuda-driver-symbol-unavailable :symbol symbol})))
    (.downcallHandle (Linker/nativeLinker) (.get found)
                     (FunctionDescriptor/of return (into-array MemoryLayout arguments))
                     (into-array Linker$Option []))))

(def ^:private handles
  (delay
    {:init (downcall "cuInit" ValueLayout/JAVA_INT [ValueLayout/JAVA_INT])
     :driver-version (downcall "cuDriverGetVersion" ValueLayout/JAVA_INT
                               [ValueLayout/ADDRESS])
     :count (downcall "cuDeviceGetCount" ValueLayout/JAVA_INT [ValueLayout/ADDRESS])
     :device (downcall "cuDeviceGet" ValueLayout/JAVA_INT
                       [ValueLayout/ADDRESS ValueLayout/JAVA_INT])
     :name (downcall "cuDeviceGetName" ValueLayout/JAVA_INT
                     [ValueLayout/ADDRESS ValueLayout/JAVA_INT ValueLayout/JAVA_INT])
     :capability (downcall "cuDeviceComputeCapability" ValueLayout/JAVA_INT
                           [ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/JAVA_INT])}))

(defn- invoke
  [^MethodHandle handle & arguments]
  (.invokeWithArguments handle (into-array Object arguments)))

(defn- check!
  [operation code]
  (when-not (zero? (int code))
    (throw (ex-info "CUDA Driver API probe failed"
                    {:reason :cuda-driver-error :operation operation :error-code (int code)}))))

(defn- device-info
  [^Arena arena ordinal]
  (let [device-pointer (.allocate arena ValueLayout/JAVA_INT)
        name-pointer (.allocate arena 256)
        major-pointer (.allocate arena ValueLayout/JAVA_INT)
        minor-pointer (.allocate arena ValueLayout/JAVA_INT)]
    (check! :device (invoke (:device @handles) device-pointer (int ordinal)))
    (let [device (.get device-pointer ValueLayout/JAVA_INT 0)]
      (check! :name (invoke (:name @handles) name-pointer (int 256) (int device)))
      (check! :compute-capability
              (invoke (:capability @handles) major-pointer minor-pointer (int device)))
      {:ordinal ordinal
       :name (.getString name-pointer 0)
       :compute-capability [(.get major-pointer ValueLayout/JAVA_INT 0)
                            (.get minor-pointer ValueLayout/JAVA_INT 0)]})))

(defn probe
  "Return driver version and physical CUDA devices without creating a context.
   An installed driver library with no usable GPU reports `:no-device`; other
   native failures are explicit errors and must not register a target."
  []
  (if-not (available?)
    {:status :unavailable :devices []}
    (let [init-code (int (invoke (:init @handles) (int 0)))]
      (cond
        (= 100 init-code) {:status :no-device :devices [] :init-code init-code}
        (not (zero? init-code)) {:status :driver-error :devices [] :init-code init-code}
        :else
        (with-open [arena (Arena/ofConfined)]
          (let [version-pointer (.allocate arena ValueLayout/JAVA_INT)
                count-pointer (.allocate arena ValueLayout/JAVA_INT)]
            (check! :driver-version (invoke (:driver-version @handles) version-pointer))
            (check! :device-count (invoke (:count @handles) count-pointer))
            (let [count (.get count-pointer ValueLayout/JAVA_INT 0)]
              (when (neg? count)
                (throw (ex-info "CUDA driver reported a negative device count"
                                {:reason :cuda-driver-device-count :count count})))
              {:status (if (zero? count) :no-device :ready)
               :driver-api-version (.get version-pointer ValueLayout/JAVA_INT 0)
               :devices (mapv #(device-info arena %) (range count))})))))))
