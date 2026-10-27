(ns raster.compiler.ir.semantic-fingerprint
  "Canonical, process-independent fingerprints for pure compiler values.

   This is deliberately narrower than Clojure printing or equality. Values are encoded with
   explicit type and collection tags; maps and sets are ordered by their encoded contents; source
   locations are excluded while other metadata is retained. Unsupported runtime objects fail
   rather than contributing identity hashes or printable representations."
  (:import [java.io ByteArrayOutputStream DataOutputStream]
           [java.math BigDecimal BigInteger]
           [java.nio.charset StandardCharsets]
           [java.security MessageDigest]
           [java.util IdentityHashMap UUID]))

(def schema :raster.semantic-fingerprint/v1)

(def ^:private source-location-keys
  #{:line :column :end-line :end-column :file})

(defn- fail! [value]
  (throw (ex-info "semantic fingerprints require pure supported compiler data"
                  {:reason :semantic-fingerprint-unsupported
                   :class (some-> value class .getName)})))

(defn- write-bytes! [^DataOutputStream out ^bytes bytes]
  (.writeInt out (alength bytes))
  (.write out bytes 0 (alength bytes)))

(defn- write-text! [^DataOutputStream out value]
  (write-bytes! out (.getBytes (str value) StandardCharsets/UTF_8)))

(defn- unsigned-byte-compare [^bytes left ^bytes right]
  (let [common (min (alength left) (alength right))]
    (loop [index 0]
      (if (= index common)
        (compare (alength left) (alength right))
        (let [l (bit-and 0xff (aget left index))
              r (bit-and 0xff (aget right index))]
          (if (= l r) (recur (inc index)) (compare l r)))))))

(declare ^:private encode-bytes)

(defn- sorted-encodings [context values]
  (sort unsigned-byte-compare (map #(encode-bytes context %) values)))

(defn- semantic-metadata [value]
  (when (instance? clojure.lang.IObj value)
    (let [metadata (apply dissoc (or (meta value) {}) source-location-keys)]
      (when (seq metadata) metadata))))

(declare encode-value!)

(defn- encode-map! [^DataOutputStream out context entries]
  (let [encoded
        (sort (fn [[left-key left-value] [right-key right-value]]
                (let [key-order (unsigned-byte-compare left-key right-key)]
                  (if (zero? key-order)
                    (unsigned-byte-compare left-value right-value)
                    key-order)))
              (map (fn [[key value]] [(encode-bytes context key) (encode-bytes context value)]) entries))]
    (.writeInt out (count encoded))
    (doseq [[key value] encoded]
      (write-bytes! out key)
      (write-bytes! out value))))

(defn- encode-sequential! [^DataOutputStream out context values]
  (.writeInt out (count values))
  (doseq [value values]
    (write-bytes! out (encode-bytes context value))))

(defn- encode-raw! [^DataOutputStream out context value]
  (cond
    (nil? value) (.writeByte out 0)
    (false? value) (.writeByte out 1)
    (true? value) (.writeByte out 2)

    (string? value)
    (do (.writeByte out 3) (write-text! out value))

    (keyword? value)
    (do (.writeByte out 4)
        (write-text! out (or (namespace value) ""))
        (write-text! out (name value)))

    (symbol? value)
    (do (.writeByte out 5)
        (write-text! out (or (namespace value) ""))
        (write-text! out (name value)))

    (char? value)
    (do (.writeByte out 6) (.writeInt out (int value)))

    (instance? Byte value)
    (do (.writeByte out 10) (.writeByte out (byte value)))

    (instance? Short value)
    (do (.writeByte out 11) (.writeShort out (short value)))

    (instance? Integer value)
    (do (.writeByte out 12) (.writeInt out (int value)))

    (instance? Long value)
    (do (.writeByte out 13) (.writeLong out (long value)))

    (instance? clojure.lang.BigInt value)
    (do (.writeByte out 14) (write-text! out value))

    (instance? BigInteger value)
    (do (.writeByte out 15) (write-bytes! out (.toByteArray ^BigInteger value)))

    (instance? Float value)
    (do (.writeByte out 16) (.writeInt out (Float/floatToRawIntBits (float value))))

    (instance? Double value)
    (do (.writeByte out 17) (.writeLong out (Double/doubleToRawLongBits (double value))))

    (instance? BigDecimal value)
    (do (.writeByte out 18)
        (.writeInt out (.scale ^BigDecimal value))
        (write-bytes! out (.toByteArray (.unscaledValue ^BigDecimal value))))

    (ratio? value)
    (do (.writeByte out 19)
        (write-bytes! out (encode-bytes context (numerator value)))
        (write-bytes! out (encode-bytes context (denominator value))))

    (instance? UUID value)
    (do (.writeByte out 20)
        (.writeLong out (.getMostSignificantBits ^UUID value))
        (.writeLong out (.getLeastSignificantBits ^UUID value)))

    ;; Only the private pair-local source comparison supplies this context.
    ;; No Class name, identity hash or portable handle representation is used.
    (and context (class? value))
    (let [^IdentityHashMap identities context
          ordinal (if (.containsKey identities value)
                    (.get identities value)
                    (let [ordinal (.size identities)]
                      (.put identities value ordinal)
                      ordinal))]
      (.writeByte out 21)
      (.writeInt out (int ordinal)))

    (record? value)
    (do (.writeByte out 30)
        (write-text! out (.getName (class value)))
        (encode-map! out context value))

    (map? value)
    (do (.writeByte out 31) (encode-map! out context value))

    (vector? value)
    (do (.writeByte out 32) (encode-sequential! out context value))

    (list? value)
    (do (.writeByte out 33) (encode-sequential! out context value))

    (set? value)
    (do (.writeByte out 34)
        (let [values (sorted-encodings context value)]
          (.writeInt out (count values))
          (doseq [encoded values] (write-bytes! out encoded))))

    (sequential? value)
    (do (.writeByte out 35) (encode-sequential! out context value))

    :else (fail! value)))

(defn- encode-value! [^DataOutputStream out context value]
  (encode-raw! out context value)
  (if-let [metadata (semantic-metadata value)]
    (do (.writeByte out 1) (write-bytes! out (encode-bytes context metadata)))
    (.writeByte out 0)))

(defn- encode-bytes [context value]
  (let [buffer (ByteArrayOutputStream.)]
    (with-open [out (DataOutputStream. buffer)]
      (encode-value! out context value))
    (.toByteArray buffer)))

(defn canonical-bytes
  "Return the canonical byte encoding of a supported pure compiler value.

   Source-location metadata is intentionally ignored. Other metadata, including type tags, is
   semantic and participates. Functions, Vars, namespaces, arrays and runtime handles fail loud."
  [value]
  (encode-bytes nil value))

(defn retained-source-equivalent?
  "Exact process-local comparison of retained source, additionally allowing Class identity.
   All ordinary semantic encoding rules still apply. Classes compare by actual
   identity within this ONE pair, including inside metadata/maps/sets. Distinct
   loaders' same-named Classes are different. No local bytes or context escape;
   this boolean is not a portable fingerprint, ownership proof or cache key.
   Other runtime objects remain unsupported even when identical on both sides."
  [left right]
  (let [context (IdentityHashMap.)]
    (java.util.Arrays/equals ^bytes (encode-bytes context left)
                            ^bytes (encode-bytes context right))))

(defn equivalent?
  "Exact canonical semantic equality, including floating-point bits and type metadata.
   Unlike Clojure numeric equality, identical NaN payloads compare equal and signed zeros differ."
  [left right]
  (java.util.Arrays/equals ^bytes (canonical-bytes left) ^bytes (canonical-bytes right)))

(defn fingerprint
  "Return a schema-qualified SHA-256 fingerprint of a supported pure compiler value."
  [value]
  (let [digest (.digest (MessageDigest/getInstance "SHA-256") (canonical-bytes value))]
    (str (subs (str schema) 1) ":"
         (apply str (map #(format "%02x" (bit-and 0xff %)) digest)))))
