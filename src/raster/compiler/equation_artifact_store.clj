(ns raster.compiler.equation-artifact-store
  "Atomic bounded storage for authenticated equation-first compiler artifacts.

   The semantic request fingerprint selects a file; the full unhashed identity remains inside the
   envelope and is checked on every load. Corrupt, stale, missing and I/O-failed entries are cache
   misses, never compiler failures or negative results."
  (:require [clojure.java.io :as io]
            [raster.compiler.equation-artifact :as artifact])
  (:import [java.nio.channels FileChannel]
           [java.nio.file AtomicMoveNotSupportedException CopyOption Files OpenOption Path
            StandardCopyOption StandardOpenOption]
           [java.nio.file.attribute FileAttribute FileTime]
           [java.security MessageDigest]))

(def ^:private default-max-entries 128)
(def ^:private default-max-bytes (* 2 1024 1024 1024))

(defn make-store
  "Create a store description without touching the filesystem."
  ([] (make-store {}))
  ([{:keys [root max-entries max-bytes]
     :or {root (io/file (System/getProperty "user.home") ".raster" "compiler-artifacts" "v1")
          max-entries default-max-entries
          max-bytes default-max-bytes}}]
   (when-not (and (integer? max-entries) (pos? max-entries)
                  (integer? max-bytes) (pos? max-bytes))
     (throw (ex-info "artifact store bounds must be positive integers"
                     {:reason :equation-artifact-store-bounds
                      :max-entries max-entries :max-bytes max-bytes})))
   {:root (io/file root) :max-entries max-entries :max-bytes max-bytes}))

(defn- sha256 [value]
  (let [digest (.digest (MessageDigest/getInstance "SHA-256")
                        (.getBytes ^String value java.nio.charset.StandardCharsets/UTF_8))]
    (apply str (map #(format "%02x" (bit-and 0xff %)) digest))))

(defn entry-file
  "Return the opaque file for a semantic request fingerprint."
  [store semantic-request-fingerprint]
  (when-not (and (string? semantic-request-fingerprint)
                 (not-empty semantic-request-fingerprint))
    (throw (ex-info "artifact store key must be a non-empty semantic fingerprint"
                    {:reason :equation-artifact-store-key})))
  (io/file (:root store) (str (sha256 semantic-request-fingerprint) ".cbor")))

(defn- artifact-files [store]
  (let [root ^java.io.File (:root store)]
    (if (.isDirectory root)
      (vec (filter #(and (.isFile ^java.io.File %)
                         (.endsWith (.getName ^java.io.File %) ".cbor"))
                   (.listFiles root)))
      [])))

(defn- trim! [store protected]
  (loop [files (sort-by #(.lastModified ^java.io.File %) (artifact-files store))
         total (reduce + 0 (map #(.length ^java.io.File %) (artifact-files store)))]
    (when (or (< (:max-entries store) (count files))
              (< (:max-bytes store) total))
      (when-let [victim (first (remove #(= (.getCanonicalPath ^java.io.File %)
                                           (.getCanonicalPath ^java.io.File protected))
                                       files))]
        (let [size (.length ^java.io.File victim)]
          (Files/deleteIfExists (.toPath ^java.io.File victim))
          (recur (vec (remove #(= victim %) files)) (- total size)))))))

(defn load-artifact
  "Load and authenticate one exact artifact.

   Returns `{:status :hit :value compilation}` or a source-free miss report."
  [store semantic-request-fingerprint expected-identity]
  (let [file (entry-file store semantic-request-fingerprint)]
    (if-not (.isFile file)
      {:status :miss :reason :not-found}
      (try
        (let [envelope (artifact/decode (Files/readAllBytes (.toPath file)))
              compilation (artifact/open expected-identity envelope)]
          ;; A read-only cache remains usable; inability to refresh approximate LRU state must not
          ;; turn a valid artifact into a compiler miss.
          (try
            (Files/setLastModifiedTime (.toPath file)
                                       (FileTime/fromMillis (System/currentTimeMillis)))
            (catch Exception _ nil))
          {:status :hit :value compilation :bytes (.length file)})
        (catch Exception error
          {:status :miss :reason :invalid-entry
           :error-class (.getName (class error))
           :artifact-reason (when (instance? clojure.lang.ExceptionInfo error)
                              (:reason (ex-data error)))})))))

(defn store-artifact!
  "Atomically publish and retain a bounded artifact entry. Returns a compact write report."
  [store semantic-request-fingerprint identity compilation]
  (when-not (= semantic-request-fingerprint (:semantic-request-fingerprint identity))
    (throw (ex-info "artifact store key must equal the sealed semantic request identity"
                    {:reason :equation-artifact-store-key-identity})))
  (let [directory (.toPath ^java.io.File (:root store))
        target (.toPath (entry-file store semantic-request-fingerprint))
        bytes (artifact/encode (artifact/seal identity compilation))]
    (when (< (:max-bytes store) (alength ^bytes bytes))
      (throw (ex-info "equation artifact exceeds the complete store byte budget"
                      {:reason :equation-artifact-store-entry-too-large
                       :bytes (alength ^bytes bytes) :max-bytes (:max-bytes store)})))
    (Files/createDirectories directory (make-array FileAttribute 0))
    (let [temporary (Files/createTempFile directory "equation-artifact-" ".tmp"
                                          (make-array FileAttribute 0))]
      (try
        (Files/write temporary bytes
                     (into-array OpenOption [StandardOpenOption/WRITE
                                             StandardOpenOption/TRUNCATE_EXISTING]))
        (with-open [channel (FileChannel/open temporary
                                              (into-array OpenOption [StandardOpenOption/WRITE]))]
          (.force channel true))
        (try
          (Files/move temporary target
                      (into-array CopyOption [StandardCopyOption/ATOMIC_MOVE
                                              StandardCopyOption/REPLACE_EXISTING]))
          (catch AtomicMoveNotSupportedException _
            (Files/move temporary target
                        (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING]))))
        (finally
          (Files/deleteIfExists temporary))))
    (trim! store (.toFile target))
    {:status :stored :bytes (alength ^bytes bytes) :file (.getName (.toFile target))}))
