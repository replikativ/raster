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

;; ================================================================
;; CPU detection (pure JVM)
;; ================================================================

(defn- parse-cpuinfo-linux
  "Parse /proc/cpuinfo for model name and cache sizes (Linux only)."
  []
  (try
    (let [content (slurp "/proc/cpuinfo")
          lines (str/split-lines content)
          model-line (first (filter #(str/starts-with? % "model name") lines))
          model-name (when model-line
                       (str/trim (second (str/split model-line #":\s*" 2))))]
      {:model-name model-name})
    (catch Exception _ {})))

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
                    (when (:cache-l1 caches) {:cache-l1 (:cache-l1 caches)})
                    (when (:cache-l2 caches) {:cache-l2 (:cache-l2 caches)})
                    (when (:cache-l3 caches) {:cache-l3 (:cache-l3 caches)}))
     :source {:simd-width (:source simd)
              :cores :detected
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
;; Optimal launch config (hardware-aware)
;; ================================================================

(defn- enumerate-block-sizes
  "Generate candidate block sizes as multiples of warp-size up to max-tpb.
  Includes powers of 2 and powers-of-2 * 3 (Halide-inspired tiling enumeration)."
  [ws max-tpb]
  (let [pow2s (take-while #(<= % max-tpb) (iterate #(* 2 %) ws))
        pow2x3 (filter #(<= % max-tpb) (map #(* 3 %) pow2s))
        all (sort (distinct (concat pow2s pow2x3)))]
    (filter #(zero? (rem % ws)) all)))

(defn optimal-block-size
  "Compute optimal CUDA block size for n elements on a device.
  Uses tiling enumeration (Halide-inspired): generates candidate block sizes
  as powers of 2 (+ factor 3), scores by estimated occupancy.
  Falls back to 256/512 when no device info available."
  [device-id n & {:keys [reduction?] :or {reduction? false}}]
  (ensure-init!)
  (let [ws (warp-size device-id)
        max-tpb (max-threads-per-block device-id)
        dev (device device-id)
        caps (:capabilities dev)
        max-warps-per-sm (or (:max-warps-per-sm caps) 64)
        max-blocks-per-sm (or (:max-blocks-per-sm caps) 32)
        candidates (enumerate-block-sizes ws max-tpb)]
    (if (and dev (> (count candidates) 1))
      ;; Score each candidate by occupancy and pick best
      (let [scored (map (fn [bs]
                          (let [warps-per-block (quot bs ws)
                                blocks-by-warps (quot max-warps-per-sm warps-per-block)
                                blocks-per-sm (min max-blocks-per-sm blocks-by-warps)
                                active-warps (* blocks-per-sm warps-per-block)
                                occupancy (/ (double active-warps) max-warps-per-sm)
                                ;; Prefer larger blocks for reductions (fewer partial results)
                                score (if reduction?
                                        (* occupancy (Math/log (double bs)))
                                        occupancy)]
                            {:block-size bs :occupancy occupancy :score score}))
                        candidates)
            best (apply max-key :score scored)]
        (:block-size best))
      ;; Fallback
      (let [preferred (if reduction? 512 256)]
        (* ws (quot (min max-tpb (max ws preferred)) ws))))))

(defn optimal-grid-size
  "Compute optimal CUDA grid size for n elements and block-size.
  Uses occupancy-aware calculation: blocks-per-SM * SM-count * waves.
  With grid-stride loops, the grid cap is a performance knob, not correctness."
  [device-id n block-size & {:keys [reduction?] :or {reduction? false}}]
  (ensure-init!)
  (let [sm (or (sm-count device-id) 108)
        ws (warp-size device-id)
        max-warps-per-sm (or (get-in (device device-id) [:capabilities :max-warps-per-sm]) 64)
        max-blocks-per-sm (or (get-in (device device-id) [:capabilities :max-blocks-per-sm]) 32)
        warps-per-block (quot block-size ws)
        blocks-per-sm (min max-blocks-per-sm (quot max-warps-per-sm warps-per-block))
        waves (if reduction? 1 2)
        target-grid (* blocks-per-sm sm waves)
        needed (int (Math/ceil (/ (double n) block-size)))]
    (min needed target-grid)))

(defn optimal-launch-config
  "Compute hardware-optimal CUDA launch config for n elements.
  Returns {:block-size int :grid-size int :shared-mem int} or nil
  if n is too small for GPU."
  [device-id n & {:keys [reduction? min-elements]
                  :or {reduction? false min-elements 1024}}]
  (when (>= n min-elements)
    (let [block-size (optimal-block-size device-id n :reduction? reduction?)
          grid-size (optimal-grid-size device-id n block-size :reduction? reduction?)
          shared-mem (if reduction?
                       ;; Reduction: shared mem = block-size * sizeof(element)
                       (* block-size 8)  ;; 8 bytes for double
                       0)]
      {:block-size block-size
       :grid-size grid-size
       :shared-mem shared-mem})))

;; ================================================================
;; Occupancy estimation
;; ================================================================

(defn estimate-occupancy
  "Estimate theoretical GPU occupancy from resource usage.
  device-id: CUDA device keyword
  resource-usage: {:registers-per-thread int :shared-mem int :block-size int}

  Returns {:occupancy float :blocks-per-sm int :limiting-factor kw}"
  [device-id {:keys [registers-per-thread shared-mem block-size]}]
  (ensure-init!)
  (let [dev (device device-id)
        caps (:capabilities dev)
        max-tpb (or (:max-threads-per-block caps) 1024)
        warp-sz (or (:warp-size caps) 32)
        ;; These are typical limits for modern NVIDIA GPUs
        max-warps-per-sm (or (:max-warps-per-sm caps) 64)
        max-blocks-per-sm (or (:max-blocks-per-sm caps) 32)
        shared-mem-per-sm (or (:shared-memory-per-sm caps)
                              (* 4 (or (:shared-memory-per-block caps) 49152)))
        registers-per-sm (or (:registers-per-sm caps) 65536)
        ;; Blocks limited by block size
        warps-per-block (int (Math/ceil (/ (double block-size) warp-sz)))
        blocks-by-warps (quot max-warps-per-sm warps-per-block)
        ;; Blocks limited by shared memory
        blocks-by-smem (if (and shared-mem (pos? shared-mem))
                         (quot shared-mem-per-sm shared-mem)
                         max-blocks-per-sm)
        ;; Blocks limited by registers
        regs-per-block (* registers-per-thread block-size)
        blocks-by-regs (if (and registers-per-thread (pos? registers-per-thread))
                         (quot registers-per-sm regs-per-block)
                         max-blocks-per-sm)
        ;; Actual blocks per SM
        blocks-per-sm (min max-blocks-per-sm blocks-by-warps blocks-by-smem blocks-by-regs)
        ;; Occupancy
        active-warps (* blocks-per-sm warps-per-block)
        occupancy (/ (double active-warps) max-warps-per-sm)
        ;; Limiting factor
        limiting (cond
                   (= blocks-per-sm blocks-by-smem) :shared-memory
                   (= blocks-per-sm blocks-by-regs) :registers
                   (= blocks-per-sm blocks-by-warps) :warps
                   :else :block-limit)]
    {:occupancy (min 1.0 occupancy)
     :blocks-per-sm blocks-per-sm
     :limiting-factor limiting}))

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

(defn optimal-workgroup-size
  "Compute optimal workgroup size for n elements on a Level Zero device.
  Uses EU count + subgroup sizes for occupancy-based selection.

  Level Zero model: EUs × threads-per-EU ÷ subgroup-size = max concurrent subgroups.
  Each workgroup is dispatched to one or more EUs."
  [device-id n & {:keys [reduction?] :or {reduction? false}}]
  (ensure-init!)
  (let [dev (device device-id)
        caps (:capabilities dev)
        sg-size (or (:simd-width caps) 16)
        max-wg (or (:max-workgroup-size caps) 1024)
        eus (or (:total-eus caps) 64)
        threads-per-eu (or (:threads-per-eu caps) 8)
        ;; Candidate workgroup sizes: multiples of subgroup size
        candidates (take-while #(<= % max-wg)
                               (iterate #(* 2 %) sg-size))
        ;; Score by occupancy: prefer sizes that fill EUs well
        scored (map (fn [wg]
                      (let [subgroups-per-wg (quot wg sg-size)
                            max-concurrent-sgs (* eus threads-per-eu (/ 1 sg-size))
                            wgs-per-eu (max 1 (quot threads-per-eu subgroups-per-wg))
                            occupancy (min 1.0 (/ (* wgs-per-eu subgroups-per-wg sg-size)
                                                  (* threads-per-eu 1.0)))
                            ;; Prefer larger groups for reductions
                            score (if reduction?
                                    (* occupancy (Math/log (double wg)))
                                    occupancy)]
                        {:workgroup-size wg :occupancy occupancy :score score}))
                    candidates)
        best (apply max-key :score scored)]
    (:workgroup-size best)))

(defn optimal-group-count
  "Compute optimal group count (grid size) for n elements and workgroup size.
  With grid-stride loops, caps at EU-count × waves."
  [device-id n workgroup-size & {:keys [reduction?] :or {reduction? false}}]
  (ensure-init!)
  (let [eus (or (total-eus device-id) 64)
        waves (if reduction? 1 2)
        target-groups (* eus waves)
        needed (int (Math/ceil (/ (double n) workgroup-size)))]
    (min needed target-groups)))

(defn kernel-launch-worthwhile?
  "Check if GPU kernel launch is worthwhile for n elements.
  Based on kernel launch overhead (~10µs) vs compute time.
  Returns true if GPU is expected to be faster than CPU."
  [device-id n]
  (ensure-init!)
  (let [dev (device device-id)
        caps (:capabilities dev)
        eus (or (:total-eus caps) 64)
        ;; Empirical: need ~100 elements per EU to amortize launch overhead
        min-elements (* eus 100)]
    (>= n min-elements)))

(defn roofline-bound
  "Determine if a kernel is compute-bound or memory-bound using the roofline model.
  ops: total floating-point operations
  bytes: total memory bytes accessed
  Returns {:bound :compute-bound|:memory-bound
           :expected-throughput-gflops double
           :arithmetic-intensity double}"
  [device-id ops bytes]
  (ensure-init!)
  (let [dev (device device-id)
        caps (:capabilities dev)
        peak-flops (or (:peak-flops-dp caps) 249.6e9)
        bandwidth (or (:memory-bandwidth-gb-s caps) 89.6)
        bandwidth-bytes (* bandwidth 1e9)
        ai (if (pos? bytes) (/ (double ops) bytes) Double/POSITIVE_INFINITY)
        ridge-point (/ peak-flops bandwidth-bytes)
        bound (if (< ai ridge-point) :memory-bound :compute-bound)
        expected (if (= bound :memory-bound)
                   (* ai bandwidth)          ;; memory-limited: AI × BW
                   (/ peak-flops 1e9))]       ;; compute-limited: peak
    {:bound bound
     :expected-throughput-gflops expected
     :arithmetic-intensity ai
     :ridge-point ridge-point}))

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
