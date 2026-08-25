(ns raster.gpu.tuning-cache
  "Shared atomic disk storage for offline GPU tuning results.

   Search algorithms own their cache keys and values; this namespace owns exactly one durable
   mechanism. Entries are guarded by the full unhashed key inside the file, so filename hash
   collisions become misses rather than wrong schedules."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io])
  (:import [java.nio.charset StandardCharsets]
           [java.nio.file AtomicMoveNotSupportedException CopyOption Files StandardCopyOption]
           [java.security MessageDigest]))

(def ^:dynamic *cache-root*
  "Root of the shared GPU autotune cache. Bind to a temporary directory in tests."
  (io/file (System/getProperty "user.home") ".raster" "autotune"))

(defn- sha256
  [value]
  (let [digest (.digest (MessageDigest/getInstance "SHA-256")
                        (.getBytes (str value) StandardCharsets/UTF_8))]
    (apply str (map #(format "%02x" (bit-and 0xff (int %))) digest))))

(defn entry-file
  [key]
  (io/file *cache-root* (str (sha256 key) ".edn")))

(defn read-entry
  "Return the cached value for exact `key`, or nil for a miss/corrupt entry."
  [key]
  (let [file (entry-file key)]
    (when (.exists file)
      (try
        (let [entry (edn/read-string (slurp file))]
          (when (= key (:key entry)) (:value entry)))
        (catch Exception _ nil)))))

(defn write-entry!
  "Atomically store `value` under exact `key` and return value."
  [key value]
  (let [directory (.toPath (io/file *cache-root*))
        target (.toPath (entry-file key))]
    (Files/createDirectories directory (make-array java.nio.file.attribute.FileAttribute 0))
    (let [temporary (Files/createTempFile directory "gpu-tuning-" ".edn"
                                          (make-array java.nio.file.attribute.FileAttribute 0))]
      (try
        (spit (.toFile temporary) (pr-str {:key key :value value}))
        (try
          (Files/move temporary target
                      (into-array CopyOption [StandardCopyOption/ATOMIC_MOVE
                                              StandardCopyOption/REPLACE_EXISTING]))
          (catch AtomicMoveNotSupportedException _
            (Files/move temporary target
                        (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING]))))
        (finally
          (Files/deleteIfExists temporary))))
    value))
