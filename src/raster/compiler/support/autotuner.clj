(ns raster.compiler.support.autotuner
  "General-purpose GPU kernel autotuner for raster compiled kernels.

  Three levels of tuning:
    Level 0 (always): workgroup-size sweep — no recompile needed.
      Workgroup size is set at zeKernelSetGroupSize launch time, not baked into SPIR-V.
    Level 1 (opt-in): source rewrites (vector-width, unroll) — requires recompile.
      Only worthwhile for compute-bound kernels; ABM kernels are memory-bound.
    Level 2 (opt-in): scan block-size sweep — requires recompile, scan kernels only.

  Usage:
    ;; After compile-abm-kernels!:
    (autotune-kernel! \"my_kernel\" ... :device-id :ze:0)

    ;; Save/load tuning cache:
    (save-tuning-cache! :ze:0 tuning-map)
    (load-tuning-cache :ze:0)

    ;; Apply cached tuning to registry:
    (apply-cached-tuning! \"my_kernel\" :ze:0)"
  (:require [raster.gpu.ze-runtime :as ze]
            [clojure.edn :as edn]
            [clojure.java.io :as io])
  (:import [java.io File]))

;; ================================================================
;; KernelTuning result record
;; ================================================================

(defrecord KernelTuning
           [kernel-name      ;; best kernel name (may differ for rewrite variants)
            workgroup-size   ;; int: best workgroup size found
            best-ms          ;; double: mean ms over timed-iters
            all-results])    ;; {config → ms} for introspection

;; ================================================================
;; Timing primitives
;; ================================================================

(defn time-kernel-runs!
  "Benchmark kernel at given workgroup-size. Returns mean ms over timed-iters.
  Uses System/nanoTime — accurate since immediate command lists sync on completion.

  kernel-name:   registered kernel name
  arrays:        vector of DeviceBuffers (passed as array args)
  scalar-args:   vector of {:type :int/:long/:float/:double :value N}
  n:             number of elements (loop bound)
  workgroup-size: workgroup size to use for this measurement
  warmup:        number of warmup launches (not timed)
  timed:         number of timed launches"
  [kernel-name arrays scalar-args n workgroup-size warmup timed]
  (let [invoke! (requiring-resolve 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel)]
    ;; Warmup
    (dotimes [_ warmup]
      (invoke! kernel-name arrays scalar-args n {:workgroup-size workgroup-size}))
    ;; Timed
    (let [t0 (System/nanoTime)]
      (dotimes [_ timed]
        (invoke! kernel-name arrays scalar-args n {:workgroup-size workgroup-size}))
      (/ (double (- (System/nanoTime) t0)) (* (double timed) 1e6)))))

;; ================================================================
;; Level 0: Workgroup-size sweep (no recompile)
;; ================================================================

(defn autotune-wg-sweep!
  "Level 0: try workgroup sizes, no recompile needed.
  Temporarily sets :workgroup-size in registry during each timed run.
  Returns KernelTuning with best workgroup-size and per-config ms map.

  kernel-name:  registered kernel name
  arrays:       vector of DeviceBuffers matching kernel's array-params
  scalar-args:  vector of pre-typed scalars {:type :int :value N}
  n:            element count (loop bound)
  device-id:    ignored (workgroup size is launch-time, not baked)
  Options:
    :wg-sizes  [64 128 256 512]  sizes to try
    :warmup    3                 warmup launches per size
    :timed     5                 timed launches per size"
  [kernel-name arrays scalar-args n device-id
   & {:keys [wg-sizes warmup timed]
      :or {wg-sizes [64 128 256 512] warmup 3 timed 5}}]
  (let [results (into {}
                      (map (fn [wg]
                             (let [ms (try
                                        (time-kernel-runs! kernel-name arrays scalar-args n wg warmup timed)
                                        (catch Exception e
                                          (println "  [autotune] wg" wg "failed:" (.getMessage e))
                                          Double/MAX_VALUE))]
                               (println (format "  [autotune] %s wg=%d → %.3fms" kernel-name (int wg) ms))
                               [wg ms]))
                           wg-sizes))
        best-wg (apply min-key results (keys results))
        best-ms (get results best-wg)]
    (->KernelTuning kernel-name best-wg best-ms {:wg-sizes results})))

;; ================================================================
;; Full autotuner (level 0; level 1 if roofline suggests)
;; ================================================================

(defn autotune-kernel!
  "Full autotuning: always runs level 0 (wg sweep, no recompile).
  Level 1 (source rewrites) is opt-in via :with-rewrites? true and
  only applied when roofline analysis suggests compute-bound behavior.
  Stores best KernelTuning into registry entry under :tuning key.
  Returns KernelTuning.

  kernel-name:    registered kernel name
  arrays:         vector of DeviceBuffers for array args
  scalar-args:    pre-typed scalars [{:type :int :value N} ...]
  n:              element count
  device-id:      device keyword (:ze:0 etc.)
  Options: see autotune-wg-sweep!"
  [kernel-name arrays scalar-args n device-id
   & {:keys [wg-sizes warmup timed with-rewrites?]
      :or {wg-sizes [64 128 256 512] warmup 3 timed 5 with-rewrites? false}}]
  (println (str "[autotune] Tuning " kernel-name " (n=" n ")"))
  ;; Level 0: wg-size sweep (always)
  (let [tuning (autotune-wg-sweep! kernel-name arrays scalar-args n device-id
                                   :wg-sizes wg-sizes :warmup warmup :timed timed)]
    (println (format "[autotune] Best: wg=%d → %.3fms" (:workgroup-size tuning) (:best-ms tuning)))
    ;; Patch registry with best wg-size
    (swap! ze/kernel-registry assoc-in [kernel-name :workgroup-size] (:workgroup-size tuning))
    (swap! ze/kernel-registry assoc-in [kernel-name :tuning] tuning)
    tuning))

;; ================================================================
;; Cache persistence
;; ================================================================

(defn- cache-dir
  "Return ~/.raster/tuning/ directory, creating it if needed."
  ^File []
  (let [d (io/file (System/getProperty "user.home") ".raster" "tuning")]
    (.mkdirs d)
    d))

(defn- cache-file
  "Return the EDN cache file for a device-id."
  ^File [device-id]
  (io/file (cache-dir) (str (name device-id) ".edn")))

(defn save-tuning-cache!
  "Save a map of {kernel-name → KernelTuning} to device-specific EDN cache.
  Only saves tuning data (workgroup-size, best-ms) — not MemorySegments."
  [device-id tuning-map]
  (let [f (cache-file device-id)
        serializable (into {}
                           (map (fn [[k v]]
                                  [k {:workgroup-size (:workgroup-size v)
                                      :best-ms (:best-ms v)
                                      :all-results (:all-results v)}])
                                tuning-map))]
    (spit f (pr-str serializable))
    (println (str "[autotune] Cache saved to " (.getPath f)))
    f))

(defn load-tuning-cache
  "Load tuning cache for device-id. Returns {} if not found."
  [device-id]
  (let [f (cache-file device-id)]
    (if (.exists f)
      (try
        (edn/read-string (slurp f))
        (catch Exception e
          (println "[autotune] Failed to read cache:" (.getMessage e))
          {}))
      {})))

(defn apply-cached-tuning!
  "Apply cached tuning to a kernel in the registry if a cache hit exists.
  Patches :workgroup-size in the registry entry.
  Returns true if cache hit, false if not found."
  [kernel-name device-id]
  (let [cache (load-tuning-cache device-id)
        entry (get cache kernel-name)]
    (if entry
      (do
        (swap! ze/kernel-registry assoc-in [kernel-name :workgroup-size]
               (:workgroup-size entry))
        (swap! ze/kernel-registry assoc-in [kernel-name :tuning]
               (map->KernelTuning (assoc entry :kernel-name kernel-name)))
        true)
      false)))
