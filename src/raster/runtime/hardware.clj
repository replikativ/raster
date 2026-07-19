(ns raster.runtime.hardware
  "Hardware registry with auto-detection and capability queries.

  Auto-detects CPU capabilities on startup (cores, SIMD width, cache sizes).
  GPU detection is conditional on ClojureCUDA availability.
  Predefined catalogue fills gaps for cross-compilation scenarios.

  Device data model:
    {:id :cpu:0
     :type :cpu
     :index 0
     :name \"AMD Ryzen 9 7950X\"
     :capabilities {:cores 16 :simd-width 4 ...}
     :source {:simd-width :detected ...}}

  Usage:
    (hardware/init!)
    (hardware/devices)
    (hardware/device :cpu:0)
    (hardware/simd-lanes :cpu:0 :double)
    (hardware/register-target-device! :cuda:0 {...})"
  (:require [clojure.string :as str]
            [clojure.java.io :as io]
            [raster.runtime.hardware-catalogue :as catalogue])
  (:import [java.io File]))

;; ================================================================
;; Device registry (mutable, filled on init!)
;; ================================================================

(defonce ^:private device-registry (atom {}))
(defonce ^:private initialized? (atom false))

;; `device` is defined below (with the query API) but the calibration disk-cache above it
;; references it — forward-declare so this file compiles top-down from a cold classloader.
(declare device)

;; The :measured layer — per-device microbench results (raster.runtime.microbench/calibrate-*!)
;; that OVERRIDE probed/catalogue values in descriptor-for. Kept here (the probe/measure domain)
;; so core.hardware reads it without a cycle. (Disk persistence is a later refinement.)
(defonce ^:private measured-registry (atom {}))

(defn set-measured!
  "Store a device's measured hardware map (from a microbench calibration)."
  [device-id measured]
  (swap! measured-registry assoc device-id measured))

(defn measured-for
  "The stored :measured map for a device, or nil if it hasn't been calibrated."
  [device-id]
  (get @measured-registry device-id))

;; --- calibration disk cache (measure-once, keyed by device identity × version) ---------------

(def calibration-version
  "Bump when the microbench methodology changes, invalidating on-disk calibrations."
  1)

(defn device-signature
  "A stable identity string for a device: name + the caps that affect measured performance. Two
   machines with the same signature share a calibration; a different CPU/GPU/version gets a fresh
   one (the disk-cache key discipline from XLA/Inductor)."
  [device-id]
  (let [d (device device-id)
        caps (:capabilities d)]
    (pr-str [(:name d)
             (select-keys caps [:cores :arch :simd-width :total-eus :threads-per-eu
                                :sm-count :compute-capability :global-memory-bytes])
             calibration-version])))

(defn- calibration-file ^File [device-id]
  (io/file (System/getProperty "user.home") ".raster" "calibration"
           (str (format "%08x" (hash (device-signature device-id))) ".edn")))

(defn save-calibration!
  "Persist a device's measured map to disk (atomic: temp file → rename), keyed by its signature."
  [device-id measured]
  (let [f (calibration-file device-id)]
    (io/make-parents f)
    (let [tmp (File/createTempFile "cal" ".edn" (.getParentFile f))]
      (spit tmp (pr-str {:signature (device-signature device-id) :measured measured}))
      (.renameTo tmp f))
    measured))

(defn load-calibration!
  "Load a device's on-disk calibration into the measured-registry IF the signature matches (same
   machine + calibration version). Returns the measured map or nil. Safe to call on startup."
  [device-id]
  (let [f (calibration-file device-id)]
    (when (.exists f)
      (try
        (let [{:keys [signature measured]} (read-string (slurp f))]
          (when (= signature (device-signature device-id))
            (set-measured! device-id measured)
            measured))
        (catch Exception _ nil)))))

;; ================================================================
;; CPU detection (pure JVM)
;; ================================================================

(defn- read-proc-lines
  "Read a /proc file as a vector of lines. MUST NOT use `slurp` — /proc files report no size and
   `FileInputStream.available()` errors on them (kernel-dependent); a BufferedReader/line-seq reads
   fine. Returns nil on any failure (non-Linux, permission)."
  [path]
  (try
    (with-open [r (java.io.BufferedReader. (java.io.FileReader. ^String path))]
      (vec (line-seq r)))
    (catch Exception _ nil)))

(defn- parse-cpuinfo-linux
  "Parse /proc/cpuinfo for model name (Linux only)."
  []
  (let [lines (read-proc-lines "/proc/cpuinfo")
        model-line (first (filter #(str/starts-with? % "model name") lines))
        model-name (when model-line
                     (str/trim (second (str/split model-line #":\s*" 2))))]
    (cond-> {} model-name (assoc :model-name model-name))))

(def ^:private cpuinfo-feature-map
  "Kernel /proc/cpuinfo flag → raster SIMD-feature keyword. Only the flags the compiler's
   hardware model cares about (dtype dot-reduce + vector width tier)."
  {"avx"          :avx     "avx2"        :avx2      "fma"        :fma
   "avx512f"      :avx512f "sse4_2"      :sse4-2
   "avx_vnni"     :avx-vnni "avx512_vnni" :avx512-vnni "amx_int8" :amx-int8 "amx_tile" :amx-tile
   "f16c"         :f16c    "avx512bf16"  :avx512-bf16})

(defn- parse-cpu-features-linux
  "Parse the `flags` line of /proc/cpuinfo into a set of raster SIMD-feature keywords (Linux only).
   Drives :has-native-dot-reduce (VNNI) and dtype legality — the coarse arch-string heuristic
   claims int-dot-reduce for ALL x86, which is wrong for pre-VNNI parts. Empty set on non-Linux."
  []
  (let [lines (read-proc-lines "/proc/cpuinfo")
        flag-line (first (filter #(str/starts-with? % "flags") lines))
        flags (when flag-line (set (str/split (str/trim (second (str/split flag-line #":\s*" 2))) #"\s+")))]
    (into #{} (keep cpuinfo-feature-map) (or flags #{}))))

(defn- parse-cache-sizes-linux
  "Read cache sizes from /sys/devices/system/cpu (Linux only)."
  []
  (try
    (let [read-cache (fn [level]
                       (let [path (str "/sys/devices/system/cpu/cpu0/cache/index" level "/size")
                             f (File. ^String path)]
                         (when (.exists f)
                           (let [s (str/trim (slurp path))
                                 n (Long/parseLong (re-find #"\d+" s))
                                 unit (re-find #"[KMG]" s)]
                             (case unit
                               "K" (* n 1024)
                               "M" (* n 1024 1024)
                               "G" (* n 1024 1024 1024)
                               n)))))]
      {:cache-l1 (read-cache 1)       ;; index1 = L1d (index0 = L1i)
       :cache-l2 (read-cache 2)
       :cache-l3 (read-cache 3)})
    (catch Exception _ {})))

(defn- detect-simd-width
  "Detect SIMD width via Vector API (JDK 16+).
  Returns {:simd-width int :simd-width-float int :source :detected}
  or catalogue defaults."
  []
  (try
    (let [dv-cls (Class/forName "jdk.incubator.vector.DoubleVector")
          sp-field (.getField dv-cls "SPECIES_PREFERRED")
          species (.get sp-field nil)
          length-method (.getMethod (class species) "length" (into-array Class []))
          double-lanes (.invoke length-method species (into-array Object []))
          fv-cls (Class/forName "jdk.incubator.vector.FloatVector")
          fsp-field (.getField fv-cls "SPECIES_PREFERRED")
          fspecies (.get fsp-field nil)
          flength-method (.getMethod (class fspecies) "length" (into-array Class []))
          float-lanes (.invoke flength-method fspecies (into-array Object []))]
      {:simd-width (int double-lanes)
       :simd-width-float (int float-lanes)
       :source :detected})
    (catch Exception _
      (let [arch (System/getProperty "os.arch")
            defaults (catalogue/cpu-defaults-for-arch arch)]
        (assoc defaults :source :catalogue)))))

(defn- detect-cpu
  "Auto-detect CPU capabilities. Returns a device map."
  []
  (let [cores (.availableProcessors (Runtime/getRuntime))
        arch (System/getProperty "os.arch")
        os-name (System/getProperty "os.name")
        linux? (and os-name (str/starts-with? (str/lower-case os-name) "linux"))
        cpuinfo (if linux? (parse-cpuinfo-linux) {})
        caches (if linux? (parse-cache-sizes-linux) {})
        features (if linux? (parse-cpu-features-linux) #{})
        simd (detect-simd-width)
        model-name (or (:model-name cpuinfo) (str arch " CPU"))]
    {:id :cpu:0
     :type :cpu
     :index 0
     :name model-name
     :capabilities (merge
                    {:cores cores
                     :simd-width (:simd-width simd)
                     :simd-width-float (:simd-width-float simd)
                     :arch arch}
                    (when (seq features) {:simd-features features})
                    (when (:cache-l1 caches) {:cache-l1 (:cache-l1 caches)})
                    (when (:cache-l2 caches) {:cache-l2 (:cache-l2 caches)})
                    (when (:cache-l3 caches) {:cache-l3 (:cache-l3 caches)}))
     :source {:simd-width (:source simd)
              :cores :detected
              :features (if (seq features) :detected :unavailable)
              :cache (if (seq caches) :detected :unavailable)}}))

;; ================================================================
;; GPU detection (conditional on ClojureCUDA)
;; ================================================================

(defn- clojurecuda-available?
  "Check if CUDA device detection is available.
  Currently returns false — CUDA support will be reimplemented
  via Level Zero or native CUDA driver API when targeted."
  []
  false)

(defn- detect-cuda-devices
  "Detect CUDA devices. Currently a stub — CUDA support will be
  reimplemented via native CUDA driver API when targeted."
  []
  [])

;; ================================================================
;; Level Zero device detection
;; ================================================================

(defn- detect-level-zero-devices
  "Detect Level Zero GPU devices via Panama FFM.
  Uses ze_runtime if available, otherwise shells out to ze_query tool.
  Returns seq of device maps or empty seq."
  []
  (try
    ;; Try using raster.gpu.ze-runtime for detection
    (require 'raster.gpu.ze-runtime)
    (let [query-fn (resolve 'raster.gpu.ze-runtime/query-devices)]
      (when query-fn
        (let [devices (query-fn)]
          (mapv (fn [idx dev-info]
                  (let [dev-name (:name dev-info)
                        catalogue-spec (catalogue/find-gpu-spec dev-name)
                        caps (merge
                              (when catalogue-spec catalogue-spec)
                              (select-keys dev-info
                                           [:total-eus :threads-per-eu :simd-width
                                            :subgroup-sizes :max-workgroup-size
                                            :shared-local-memory :global-memory-bytes
                                            :memory-bandwidth-gb-s :core-clock-mhz
                                            :device-id-hex :integrated?]))]
                    {:id (keyword (str "ze:" idx))
                     :type :ze
                     :index idx
                     :name dev-name
                     :capabilities caps
                     :source {:all :detected}}))
                (range) devices))))
    (catch Exception e
      (when (System/getenv "ROMEO_DEBUG")
        (println "Level Zero detection failed:" (.getMessage e)))
      [])))

;; ================================================================
;; OpenCL device detection
;; ================================================================

(defn- detect-opencl-devices
  "Detect OpenCL GPU devices via raster.gpu.ocl-runtime.
  Returns seq of device maps or empty seq."
  []
  (try
    (require 'raster.gpu.ocl-runtime)
    (let [query-fn (resolve 'raster.gpu.ocl-runtime/query-devices)]
      (when query-fn
        (let [devices (query-fn)]
          (mapv (fn [idx dev-info]
                  (let [dev-name (:name dev-info)
                        catalogue-spec (catalogue/find-gpu-spec dev-name)
                        caps (merge
                              (when catalogue-spec catalogue-spec)
                              (select-keys dev-info
                                           [:max-compute-units :max-work-group-size
                                            :global-mem-bytes :max-clock-mhz
                                            :extensions :vendor :version
                                            :integrated?]))]
                    {:id (keyword (str "ocl:" idx))
                     :type :ocl
                     :index idx
                     :name dev-name
                     :capabilities caps
                     :source {:all :detected}}))
                (range) devices))))
    (catch Exception e
      (when (System/getenv "ROMEO_DEBUG")
        (println "OpenCL detection failed:" (.getMessage e)))
      [])))

;; ================================================================
;; Initialization
;; ================================================================

(defn init!
  "Initialize the hardware registry. Called lazily on first use.
  Detects CPU capabilities and, if available, GPU devices (CUDA and Level Zero).
  Safe to call multiple times (idempotent)."
  []
  (when (compare-and-set! initialized? false true)
    (let [cpu (detect-cpu)
          cuda-gpus (if (clojurecuda-available?)
                      (detect-cuda-devices)
                      [])
          ze-gpus (detect-level-zero-devices)
          ocl-gpus (detect-opencl-devices)]
      (swap! device-registry merge
             {(:id cpu) cpu}
             (into {} (map (fn [g] [(:id g) g]) cuda-gpus))
             (into {} (map (fn [g] [(:id g) g]) ze-gpus))
             (into {} (map (fn [g] [(:id g) g]) ocl-gpus)))))
  nil)

(defn- ensure-init! []
  (when-not @initialized? (init!)))

;; ================================================================
;; Public API
;; ================================================================

(defn devices
  "Return all detected devices as {device-id -> device-map}."
  []
  (ensure-init!)
  @device-registry)

(defn device
  "Get a specific device by id (e.g. :cpu:0, :cuda:0).
  Returns nil if device not found."
  [device-id]
  (ensure-init!)
  (get @device-registry device-id))

(defn default-device
  "Return the default compute device (usually :cpu:0)."
  []
  (ensure-init!)
  (or (get @device-registry :cpu:0)
      (first (vals @device-registry))))

(defn device-type
  "Extract device type (:cpu, :cuda, :ze, or :ocl) from a device-id keyword.
  E.g. :cpu:0 -> :cpu, :cuda:1 -> :cuda, :ze:0 -> :ze, :ocl:0 -> :ocl"
  [device-id]
  (when device-id
    (let [s (name device-id)]
      (cond
        (str/starts-with? s "cpu")  :cpu
        (str/starts-with? s "cuda") :cuda
        (str/starts-with? s "ze")   :ze
        (str/starts-with? s "ocl")  :ocl
        :else (keyword (first (str/split s #":")))))))

(defn register-target-device!
  "Register a target device for cross-compilation.
  Used when developing CUDA code on a CPU-only machine.

  device-id: keyword like :cuda:0
  spec: map with at minimum {:name \"GPU Name\" :capabilities {...}}

  Capabilities are merged with catalogue data if available."
  [device-id spec]
  (ensure-init!)
  (let [dev-name (:name spec)
        catalogue-spec (when dev-name (catalogue/find-gpu-spec dev-name))
        idx (or (:index spec)
                (when-let [m (re-find #":(\d+)$" (name device-id))]
                  (Long/parseLong (second m)))
                0)
        ;; Merge capabilities: catalogue provides defaults, user spec overrides
        merged-caps (merge (or catalogue-spec {})
                           (or (:capabilities spec) {}))
        device (merge
                {:id device-id
                 :type (device-type device-id)
                 :index idx
                 :source {:all :user}}
                 ;; User spec (without :capabilities, which we merge separately)
                (dissoc spec :capabilities)
                 ;; Merged capabilities
                {:capabilities merged-caps})]
    (swap! device-registry assoc device-id device)
    device))

;; ================================================================
;; Capability queries
;; ================================================================

(defn simd-lanes
  "Get SIMD lane count for a device and element type.
  Returns int (e.g. 4 for AVX2 doubles, 8 for AVX-512 doubles)."
  [device-id element-type]
  (ensure-init!)
  (let [dev (device device-id)
        caps (:capabilities dev)]
    (case element-type
      :double (or (:simd-width caps) 4)
      :float  (or (:simd-width-float caps) 8)
      :long   (or (:simd-width caps) 4)       ;; same as double (64-bit)
      :int    (or (:simd-width-float caps) 8)  ;; same as float (32-bit)
      4)))

(defn warp-size
  "Get warp/wavefront size for a CUDA device. Returns 32 by default."
  [device-id]
  (ensure-init!)
  (let [dev (device device-id)]
    (or (get-in dev [:capabilities :warp-size]) 32)))

(defn sm-count
  "Get streaming multiprocessor count for a CUDA device."
  [device-id]
  (ensure-init!)
  (get-in (device device-id) [:capabilities :sm-count]))

(defn max-threads-per-block
  "Get max threads per block for a CUDA device."
  [device-id]
  (ensure-init!)
  (or (get-in (device device-id) [:capabilities :max-threads-per-block]) 1024))

(defn shared-memory-per-block
  "Get shared memory per block in bytes for a CUDA device."
  [device-id]
  (ensure-init!)
  (or (get-in (device device-id) [:capabilities :shared-memory-per-block]) 49152))

(defn compute-capability
  "Get compute capability [major minor] for a CUDA device."
  [device-id]
  (ensure-init!)
  (get-in (device device-id) [:capabilities :compute-capability]))

(defn compute-capability-str
  "Get compute capability as 'smXY' string (e.g. 'sm90' for [9 0])."
  [device-id]
  (when-let [[major minor] (compute-capability device-id)]
    (str "sm" major minor)))

;; ================================================================
;; Level Zero capability queries
;; ================================================================

(defn subgroup-size
  "Get preferred subgroup (SIMD) size for a Level Zero device.
  Analogous to warp-size for CUDA."
  [device-id]
  (ensure-init!)
  (let [dev (device device-id)
        caps (:capabilities dev)]
    (or (:simd-width caps) 16)))

(defn total-eus
  "Get total Execution Units for a Level Zero device."
  [device-id]
  (ensure-init!)
  (get-in (device device-id) [:capabilities :total-eus]))

(defn shared-local-memory
  "Get Shared Local Memory (SLM) size in bytes for a Level Zero device."
  [device-id]
  (ensure-init!)
  (or (get-in (device device-id) [:capabilities :shared-local-memory]) 65536))

;; ================================================================
;; Memory topology
;; ================================================================

(defn memory-topology
  "Return the memory topology for a device — how CPU and GPU share memory.

  Returns a map:
    :model      — :unified (CPU+GPU share one pool, zero-copy)
                  :discrete (separate VRAM, PCIe transfers)
    :integrated? — true if GPU is integrated into the SoC/package
    :pcie-gb-s  — PCIe bandwidth in GB/s (nil for :unified)
    :global-memory-gb — total device-visible memory in GB

  Used by gpu.clj to choose between zero-copy buffer views (:unified)
  and explicit upload (:discrete). CUDA devices always :discrete.
  Future NVLink/CXL devices may add :nvlink or :cxl models."
  [device-id]
  (ensure-init!)
  (let [dev  (device device-id)
        caps (:capabilities dev)
        dtype (:type dev)
        ;; Primary: Level Zero ZE_DEVICE_PROPERTY_FLAG_INTEGRATED (bit 0 of flags)
        ;; Fallback: name-based — "Intel Arc Graphics" with no model suffix is
        ;; the canonical Lunar Lake / Meteor Lake iGPU naming convention.
        ;; Some LZ driver versions don't set the flag reliably.
        dev-name    (or (:name dev) "")
        name-integrated? (and (re-find #"(?i)intel.*arc.*graphics$" dev-name)
                              (not (re-find #"(?i)A[0-9]" dev-name)))
        integrated? (or (boolean (:integrated? caps))
                        name-integrated?)
        global-gb   (when-let [b (:global-memory-bytes caps)]
                      (/ b 1e9))
        model (cond
                ;; CUDA devices are always discrete (for now)
                (= dtype :cuda) :discrete
                ;; OpenCL: same heuristic as Level Zero
                ;; Level Zero: flag or name heuristic determines topology
                integrated?     :unified
                :else           :discrete)
        pcie-gb-s (when (= model :discrete)
                    ;; Default PCIe 4.0 x16 throughput; catalogue may override
                    (or (:pcie-bandwidth-gb-s caps) 16.0))]
    {:model        model
     :integrated?  integrated?
     :pcie-gb-s    pcie-gb-s
     :global-memory-gb global-gb}))

;; ================================================================
;; Reset (for testing)
;; ================================================================

(defn reset-hardware!
  "Reset the hardware registry (for testing)."
  []
  (clojure.core/reset! device-registry {})
  (clojure.core/reset! initialized? false)
  nil)
