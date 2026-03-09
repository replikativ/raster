(ns raster.dl.safetensors
  "SafeTensors file format loader and writer.

  SafeTensors is a simple, safe tensor serialization format used by HuggingFace.
  Layout: [8-byte LE header_size] [JSON header] [raw tensor data...]

  The JSON header maps tensor names to {\"dtype\", \"shape\", \"data_offsets\"}.
  Tensor data is stored contiguously after the header, accessed via mmap.

  Usage:
    ;; Load
    (def model (load-safetensors \"/path/to/model.safetensors\"))
    (keys model)          ;; tensor names
    (:dtype (model \"embeddings.weight\"))  ;; \"F32\", \"F16\", \"BF16\", etc.
    (:data  (model \"embeddings.weight\"))  ;; float[] (native F32 storage)

    ;; Save
    (save-safetensors {\"weight\" {:shape [3 4] :data (double-array 12)}} \"/tmp/out.safetensors\")"
  (:require [clojure.data.json :as json])
  (:import [java.io RandomAccessFile FileOutputStream]
           [java.nio ByteBuffer ByteOrder MappedByteBuffer]
           [java.nio.channels FileChannel FileChannel$MapMode]))

(defn- read-header
  "Read SafeTensors header: 8-byte LE length + JSON string.
  Returns [header-map data-offset]."
  [^FileChannel channel]
  (let [len-buf (doto (ByteBuffer/allocate 8) (.order ByteOrder/LITTLE_ENDIAN))
        _ (.read channel len-buf)
        _ (.flip len-buf)
        header-size (.getLong len-buf)
        header-buf (ByteBuffer/allocate (int header-size))
        _ (.read channel header-buf)
        _ (.flip header-buf)
        header-str (String. (.array header-buf) "UTF-8")
        header (json/read-str header-str)]
    [header (+ 8 header-size)]))

(defn- read-tensor-floats
  "Read raw tensor data from mmap'd buffer, converting to float[].
  F32 is a direct bulk copy; other dtypes convert element-wise."
  [^MappedByteBuffer mmap offset end-offset dtype numel]
  (let [offset (long offset) end-offset (long end-offset) numel (long numel)
        buf (.slice mmap (int offset) (int (- end-offset offset)))
        _ (.order buf ByteOrder/LITTLE_ENDIAN)
        out (float-array numel)]
    (case dtype
      "F32" (let [fb (.asFloatBuffer buf)]
              (.get fb out))
      "F64" (dotimes [i numel] (aset out i (float (.getDouble buf))))
      "F16" (dotimes [i numel]
              (aset out i (Float/float16ToFloat (.getShort buf))))
      "BF16" (dotimes [i numel]
               (let [bits (int (bit-and (int (.getShort buf)) 0xFFFF))]
                 (aset out i (Float/intBitsToFloat
                              (bit-shift-left bits 16)))))
      "I64" (dotimes [i numel] (aset out i (float (.getLong buf))))
      "I32" (dotimes [i numel] (aset out i (float (.getInt buf))))
      "I16" (dotimes [i numel] (aset out i (float (.getShort buf))))
      "I8"  (dotimes [i numel] (aset out i (float (.get buf))))
      "U8"  (dotimes [i numel] (aset out i (float (bit-and (int (.get buf)) 0xFF))))
      (throw (ex-info (str "Unsupported dtype: " dtype) {:dtype dtype})))
    out))

(defn load-safetensors
  "Load a SafeTensors file. Returns a map of tensor-name to
  {:dtype string, :shape vector, :data float-array}.

  All numeric types are converted to float[] for uniform handling.
  F32 tensors use bulk FloatBuffer copy (fast)."
  [^String path]
  (with-open [raf (RandomAccessFile. path "r")
              channel (.getChannel raf)]
    (let [[header data-offset] (read-header channel)
          file-size (.size channel)
          mmap (.map channel FileChannel$MapMode/READ_ONLY
                     (long data-offset) (- file-size (long data-offset)))]
      (persistent!
       (reduce-kv
        (fn [acc tensor-name info]
          (if (= tensor-name "__metadata__")
            acc
            (let [dtype (get info "dtype")
                  shape (mapv long (get info "shape"))
                  [offset end-offset] (get info "data_offsets")
                  numel (reduce * 1 shape)
                  data (read-tensor-floats mmap (long offset) (long end-offset)
                                           dtype numel)]
              (assoc! acc tensor-name {:dtype dtype :shape shape :data data}))))
        (transient {})
        header)))))

(defn load-safetensors-lazy
  "Load SafeTensors header only. Returns a map with :path, :data-offset,
  and :tensors {name → {:dtype :shape :offset :end-offset}}.
  Use `read-tensor` to load individual tensors on demand."
  [^String path]
  (with-open [raf (RandomAccessFile. path "r")
              channel (.getChannel raf)]
    (let [[header data-offset] (read-header channel)]
      {:path path
       :data-offset data-offset
       :tensors
       (persistent!
        (reduce-kv
         (fn [acc tensor-name info]
           (if (= tensor-name "__metadata__")
             acc
             (let [dtype (get info "dtype")
                   shape (mapv long (get info "shape"))
                   [offset end-offset] (get info "data_offsets")]
               (assoc! acc tensor-name
                       {:dtype dtype :shape shape
                        :offset (long offset)
                        :end-offset (long end-offset)}))))
         (transient {})
         header))})))

(defn read-tensor
  "Read a single tensor from a lazy-loaded SafeTensors file."
  [lazy-st tensor-name]
  (let [{:keys [path data-offset tensors]} lazy-st
        {:keys [dtype shape offset end-offset]} (get tensors tensor-name)
        numel (reduce * 1 shape)]
    (when-not dtype
      (throw (ex-info (str "Tensor not found: " tensor-name)
                      {:available (keys tensors)})))
    (with-open [raf (RandomAccessFile. ^String path "r")
                channel (.getChannel raf)]
      (let [mmap (.map channel FileChannel$MapMode/READ_ONLY
                       (long data-offset)
                       (- (.size channel) (long data-offset)))]
        {:dtype dtype :shape shape
         :data (read-tensor-floats mmap offset end-offset dtype numel)}))))

(defn tensor-summary
  "Print a summary of tensors in a SafeTensors file."
  [path]
  (let [lazy (load-safetensors-lazy path)
        tensors (:tensors lazy)]
    (println (str (count tensors) " tensors in " path))
    (doseq [[name {:keys [dtype shape]}] (sort-by key tensors)]
      (println (format "  %-60s %-5s %s" name dtype (str shape))))))

;; ================================================================
;; Writer
;; ================================================================

(defn- dtype-for-array
  "Infer SafeTensors dtype string from JVM array type."
  ^String [arr]
  (let [cls (class arr)]
    (condp = cls
      (Class/forName "[D") "F64"
      (Class/forName "[F") "F32"
      (Class/forName "[J") "I64"
      (Class/forName "[I") "I32"
      (throw (ex-info (str "Unsupported array type for SafeTensors: " cls)
                      {:type cls})))))

(defn- bytes-per-element
  "Bytes per element for a dtype string."
  ^long [^String dtype]
  (case dtype "F64" 8 "F32" 4 "I64" 8 "I32" 4 "F16" 2 "BF16" 2
        "I16" 2 "I8" 1 "U8" 1))

(defn- write-array-bytes!
  "Write array data to a ByteBuffer in little-endian order."
  [^ByteBuffer buf arr ^String dtype]
  (case dtype
    "F64" (let [db (.asDoubleBuffer buf)]
            (.put db ^doubles arr)
            (.position buf (+ (.position buf) (* 8 (alength ^doubles arr)))))
    "F32" (let [fb (.asFloatBuffer buf)]
            (.put fb ^floats arr)
            (.position buf (+ (.position buf) (* 4 (alength ^floats arr)))))
    "I64" (let [lb (.asLongBuffer buf)]
            (.put lb ^longs arr)
            (.position buf (+ (.position buf) (* 8 (alength ^longs arr)))))
    "I32" (let [ib (.asIntBuffer buf)]
            (.put ib ^ints arr)
            (.position buf (+ (.position buf) (* 4 (alength ^ints arr)))))))

(defn save-safetensors
  "Save tensors to a SafeTensors file.

  tensors: map of tensor-name (string) to {:shape [dims], :data array}
           Dtype is inferred from the array type (double[]→F64, float[]→F32, etc.)
  path: output file path

  Example:
    (save-safetensors {\"weight\" {:shape [3 4] :data (double-array 12)}
                       \"bias\"   {:shape [4]   :data (float-array 4)}}
                      \"/tmp/model.safetensors\")"
  [tensors ^String path]
  (let [;; Sort by name for deterministic output
        sorted-names (sort (keys tensors))
        ;; Compute dtype and byte sizes, build data_offsets
        tensor-infos
        (loop [names sorted-names offset 0 infos []]
          (if-let [name (first names)]
            (let [{:keys [shape data]} (get tensors name)
                  dtype (dtype-for-array data)
                  numel (reduce * 1 shape)
                  byte-size (* numel (bytes-per-element dtype))
                  end (+ offset byte-size)]
              (recur (rest names) end
                     (conj infos {:name name :dtype dtype :shape shape
                                  :data data :offset offset :end end})))
            infos))
        total-data-bytes (if (seq tensor-infos)
                           (:end (peek tensor-infos))
                           0)
        ;; Build JSON header
        header-map (reduce (fn [m {:keys [name dtype shape offset end]}]
                             (assoc m name {"dtype" dtype
                                            "shape" shape
                                            "data_offsets" [offset end]}))
                           {} tensor-infos)
        header-json (json/write-str header-map)
        header-bytes (.getBytes ^String header-json "UTF-8")
        ;; Pad header to 8-byte alignment
        header-len (count header-bytes)
        padded-len (* 8 (long (Math/ceil (/ header-len 8.0))))
        padding (- padded-len header-len)
        ;; Allocate and write
        total-size (+ 8 padded-len total-data-bytes)
        buf (doto (ByteBuffer/allocate total-size)
              (.order ByteOrder/LITTLE_ENDIAN))]
    ;; Write header size (8 bytes LE)
    (.putLong buf (long padded-len))
    ;; Write header JSON + padding
    (.put buf header-bytes)
    (dotimes [_ padding] (.put buf (byte 0x20))) ;; space padding
    ;; Write tensor data
    (doseq [{:keys [data dtype]} tensor-infos]
      (write-array-bytes! buf data dtype))
    ;; Flush to file
    (.flip buf)
    (with-open [fos (FileOutputStream. path)
                channel (.getChannel fos)]
      (.write channel buf))))

(defn save-model-params
  "Save model parameters to SafeTensors format.
  params: map of name (string) → {:data array, :shape vector}
  Infers dtype from array type."
  [params ^String path]
  (save-safetensors params path))
