(ns raster.test-support.numerical-checkpoint
  "Local-file realization fixtures, not a production persistence provider."
  (:require [raster.compiler.ir.numerical-state :as state]
            [raster.runtime.numerical-content :as content]
            [raster.gpu.core :as gpu])
  (:import [java.lang.foreign Arena]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Files OpenOption StandardOpenOption]
           [java.nio.file.attribute FileAttribute]
           [java.security MessageDigest]
           [java.util HexFormat]))

(defn temp-files [fields]
  (let [created (atom {})]
    (try
      (doseq [field fields]
        (swap! created assoc field
               (Files/createTempFile "raster-numerical-" ".bin" (make-array FileAttribute 0))))
      @created
      (catch Throwable error
        (doseq [path (vals @created)]
          (try (Files/deleteIfExists path)
               (catch Throwable cleanup (.addSuppressed error cleanup))))
        (throw error)))))

(defn payload-address [^ByteBuffer bytes]
  (let [digest (MessageDigest/getInstance "SHA-256")]
    (.update digest (.duplicate bytes))
    (state/content-address :sha-256 (.formatHex (HexFormat/of) (.digest digest)))))

(defn capture-f64! [session resident path elements]
  (when-not (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
    (throw (ex-info "raw f64 checkpoint transfer needs endian conversion" {})))
  (let [byte-count (Math/multiplyExact (long elements) (long Double/BYTES))]
    (with-open [arena (Arena/ofConfined)
                channel (FileChannel/open path (into-array OpenOption
                                                          [StandardOpenOption/READ StandardOpenOption/WRITE]))]
      (.truncate channel 0)
      (.position channel (dec byte-count))
      (.write channel (ByteBuffer/wrap (byte-array [0])))
      (let [segment (.map channel FileChannel$MapMode/READ_WRITE 0 byte-count arena)]
        (gpu/download-range! session resident segment {:elements elements})
        (.force segment)
        {:content (payload-address (.asByteBuffer segment)) :bytes byte-count}))))

(defn open-chunk-lease [path chunk]
  (let [address (:content chunk)
        arena (Arena/ofConfined)]
    (try
      (with-open [channel (FileChannel/open path (into-array OpenOption [StandardOpenOption/READ]))]
        (when-not (= (:stored-byte-length chunk) (.size channel))
          (throw (ex-info "checkpoint payload extent differs" {:reason :checkpoint-size})))
        (let [segment (.map channel FileChannel$MapMode/READ_ONLY 0 (.size channel) arena)]
          (when-not (= address (payload-address (.asByteBuffer segment)))
            (throw (ex-info "checkpoint payload does not match its content address"
                            {:reason :checkpoint-digest})))
          (content/local-content-lease
           {:content address
            :placement (content/content-placement {:provider-id :local-test :tier-id :file :content address})
            :segment segment :byte-length (.byteSize segment) :release-fn #(.close arena)})))
      (catch Throwable error (.close arena) (throw error)))))
