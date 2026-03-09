(ns raster.compiler.support.spirv-cache
  "SPIR-V compilation cache for OpenCL kernels.

  Caches compiled SPIR-V bytecode on disk at ~/.raster/spirv-cache/{hash}.spv.
  Offline compilation via `ocloc` is ~100-500ms first time; cached SPIR-V
  loads in <1ms.

  Supports per-device subdirectories for device-specific native binaries.

  Usage:
    (def cache (make-cache))
    (get-or-compile cache cl-source compile-fn)
    (get-or-compile cache cl-source compile-fn \"0x64a0\")"
  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.security MessageDigest]
           [java.io File]
           [java.nio.file Files Path Paths]
           [java.nio.file.attribute FileAttribute]))

;; ================================================================
;; Hashing
;; ================================================================

(defn- sha256
  "Compute SHA-256 hash of a string, return hex."
  [^String s]
  (let [md (MessageDigest/getInstance "SHA-256")
        bytes (.digest md (.getBytes s "UTF-8"))]
    (str/join (map #(format "%02x" %) bytes))))

;; ================================================================
;; Cache directory
;; ================================================================

(def ^:private default-cache-dir
  "Default SPIR-V cache directory."
  (str (System/getProperty "user.home") "/.raster/spirv-cache"))

(defn- ensure-cache-dir!
  "Create cache directory if it doesn't exist."
  [^String dir]
  (let [f (File. dir)]
    (when-not (.exists f)
      (.mkdirs f))
    dir))

;; ================================================================
;; Cache operations
;; ================================================================

(defn make-cache
  "Create a SPIR-V cache instance.
  Options:
    :dir - cache directory (default ~/.raster/spirv-cache)"
  [& {:keys [dir] :or {dir default-cache-dir}}]
  {:dir (ensure-cache-dir! dir)
   :stats (atom {:hits 0 :misses 0 :compiles 0})})

(defn cache-path
  "Get the file path for a cached SPIR-V entry.
  When device-id is specified, uses per-device subdirectory."
  ([cache source-hash]
   (str (:dir cache) "/" source-hash ".spv"))
  ([cache source-hash device-id]
   (if device-id
     (let [dev-dir (str (:dir cache) "/" device-id)]
       (ensure-cache-dir! dev-dir)
       (str dev-dir "/" source-hash ".spv"))
     (cache-path cache source-hash))))

(defn cached?
  "Check if SPIR-V for the given source is cached.
  Optional device-id string (e.g. '0x64a0') for per-device caching."
  ([cache ^String cl-source]
   (cached? cache cl-source nil))
  ([cache ^String cl-source device-id]
   (let [hash (sha256 (if device-id
                        (str cl-source "_" device-id)
                        cl-source))
         path (cache-path cache hash device-id)]
     (.exists (File. ^String path)))))

(defn get-cached
  "Get cached SPIR-V bytes, or nil if not cached.
  Optional device-id string for per-device caching."
  ([cache ^String cl-source]
   (get-cached cache cl-source nil))
  ([cache ^String cl-source device-id]
   (let [hash (sha256 (if device-id
                        (str cl-source "_" device-id)
                        cl-source))
         path (cache-path cache hash device-id)
         f (File. ^String path)]
     (when (.exists f)
       (swap! (:stats cache) update :hits inc)
       (Files/readAllBytes (.toPath f))))))

(defn put-cache!
  "Store SPIR-V bytes in cache.
  Optional device-id string for per-device caching."
  ([cache ^String cl-source ^bytes spv-bytes]
   (put-cache! cache cl-source spv-bytes nil))
  ([cache ^String cl-source ^bytes spv-bytes device-id]
   (let [hash (sha256 (if device-id
                        (str cl-source "_" device-id)
                        cl-source))
         path (cache-path cache hash device-id)]
     (Files/write (Paths/get path (into-array String []))
                  spv-bytes
                  (into-array java.nio.file.OpenOption []))
     (swap! (:stats cache) update :compiles inc)
     spv-bytes)))

(defn get-or-compile
  "Get cached SPIR-V bytes or compile with the provided compile-fn.
  compile-fn: (fn [cl-source] -> byte-array of SPIR-V)
  Optional device-id string for per-device caching.

  Returns SPIR-V byte array."
  ([cache ^String cl-source compile-fn]
   (get-or-compile cache cl-source compile-fn nil))
  ([cache ^String cl-source compile-fn device-id]
   (or (get-cached cache cl-source device-id)
       (do
         (swap! (:stats cache) update :misses inc)
         (let [spv (compile-fn cl-source)]
           (put-cache! cache cl-source spv device-id)
           spv)))))

;; ================================================================
;; ocloc compilation
;; ================================================================

(defn compile-opencl-to-spirv
  "Compile OpenCL C source to SPIR-V using Intel's ocloc offline compiler.
  Returns SPIR-V byte array.

  Options:
    :device - target device architecture string (e.g. \"lnl\" for Lunar Lake)
    :options - additional ocloc options string"
  [^String cl-source & {:keys [device options]}]
  (let [tmp-dir (str (System/getProperty "java.io.tmpdir") "/raster-spirv-" (System/nanoTime))
        tmp-cl (str tmp-dir "/kernel.cl")
        tmp-spv (str tmp-dir "/kernel.spv")]
    (try
      (.mkdirs (File. ^String tmp-dir))
      (spit tmp-cl cl-source)
      (let [cmd (cond-> ["ocloc" "compile"
                         "-file" tmp-cl
                         "-spv_only"
                         "-output" (str tmp-dir "/kernel")]
                  device  (into ["-device" device])
                  options (into (str/split options #"\s+")))
            proc (-> (ProcessBuilder. ^java.util.List (vec cmd))
                     (.redirectErrorStream true)
                     (.start))
            output (slurp (.getInputStream proc))
            exit (.waitFor proc)]
        (when-not (zero? exit)
          (throw (ex-info (str "ocloc compilation failed:\n" output)
                          {:exit-code exit :command cmd :source cl-source})))
        ;; ocloc appends "_.spv" to the -output base name
        (let [files (.listFiles (File. ^String tmp-dir))
              spv (first (filter #(.endsWith (.getName ^File %) ".spv") files))]
          (if spv
            (Files/readAllBytes (.toPath ^File spv))
            (throw (ex-info "ocloc produced no SPIR-V output"
                            {:output output :dir tmp-dir :files (mapv str files)})))))
      (finally
        ;; Cleanup temp files
        (doseq [^File f (.listFiles (File. ^String tmp-dir))]
          (.delete f))
        (.delete (File. ^String tmp-dir))))))

;; ================================================================
;; Stats and cleanup
;; ================================================================

(defn cache-stats
  "Return cache hit/miss/compile statistics."
  [cache]
  @(:stats cache))

(defn clear-cache!
  "Remove all cached SPIR-V files."
  [cache]
  (letfn [(delete-recursive [^File f]
            (when (.isDirectory f)
              (doseq [^File child (.listFiles f)]
                (delete-recursive child)))
            (when (.getName f)
              (when (or (.endsWith (.getName f) ".spv")
                        (.endsWith (.getName f) ".bin"))
                (.delete f))))]
    (delete-recursive (File. ^String (:dir cache))))
  (reset! (:stats cache) {:hits 0 :misses 0 :compiles 0})
  nil)
