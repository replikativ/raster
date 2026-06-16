(ns npy
  "Robust .npy reader for the umap comparison harnesses.

  Unlike the old ad-hoc readers, this PARSES the header dict (descr,
  fortran_order, shape) and returns data in C-order (row-major) flat arrays.
  Column-major (fortran_order:True) files are transposed on load — silently
  reading those row-major was the root cause of the phantom 'large-n recall
  collapse' (see memory nndescent_collapse_was_fortran_order)."
  (:require [clojure.string :as str]))

(defn- parse-header
  "Returns {:descr \"<f8\" :fortran bool :shape [..] :data-offset int}."
  [^bytes bs]
  (let [bb (doto (java.nio.ByteBuffer/wrap bs) (.order java.nio.ByteOrder/LITTLE_ENDIAN))
        ver (aget bs 6)                                   ; major version byte
        [hlen hstart] (if (= ver 1)
                        [(bit-and (int (.getShort bb 8)) 0xFFFF) 10]
                        [(int (.getInt bb 8)) 12])         ; v2: 4-byte header len
        hdr (String. bs (int hstart) (int hlen) java.nio.charset.StandardCharsets/US_ASCII)
        descr (second (re-find #"'descr':\s*'([^']+)'" hdr))
        fortran (= "True" (second (re-find #"'fortran_order':\s*(True|False)" hdr)))
        shape (->> (re-find #"'shape':\s*\(([^)]*)\)" hdr) second
                   (#(str/split % #",")) (map str/trim) (remove str/blank?)
                   (mapv #(Long/parseLong %)))]
    {:descr descr :fortran fortran :shape shape :data-offset (+ hstart hlen)}))

(defn- elt-bytes [descr] (Long/parseLong (subs descr (dec (count descr)))))

;; Fast type-hinted 2D fortran->C transpose: C[i*cols+j] = F[j*rows+i].
;; (X is always f8/f4; labels/indices are 1D so never transposed.)
(defn- transpose-f64 ^doubles [^doubles raw ^long rows ^long cols]
  (let [out (double-array (* rows cols))]
    (dotimes [i rows] (dotimes [j cols] (aset out (+ (* i cols) j) (aget raw (+ (* j rows) i))))) out))
(defn- transpose-f32 ^floats [^floats raw ^long rows ^long cols]
  (let [out (float-array (* rows cols))]
    (dotimes [i rows] (dotimes [j cols] (aset out (+ (* i cols) j) (aget raw (+ (* j rows) i))))) out))

(defn- transpose-needed? [{:keys [fortran shape]}] (and fortran (= 2 (count shape))))

(defn load-npy
  "Load a .npy file. Returns {:shape [..] :descr s :data <primitive C-order flat array>}.
  Supported descr: <f8 <f4 <i4 <i8. Fortran-order 2D arrays are transposed to C-order."
  [path]
  (let [bs (java.nio.file.Files/readAllBytes (java.nio.file.Path/of path (make-array String 0)))
        {:keys [descr shape data-offset] :as h} (parse-header bs)
        bb (doto (java.nio.ByteBuffer/wrap bs) (.order java.nio.ByteOrder/LITTLE_ENDIAN)
                 (.position (int data-offset)))
        n (long (reduce * 1 shape))
        eb (elt-bytes descr)
        ;; read raw in stored order
        raw (case (int eb)
              8 (if (str/starts-with? descr "<f")
                  (let [a (double-array n)] (.get (.asDoubleBuffer bb) a) a)
                  (let [a (long-array n)]   (.get (.asLongBuffer bb) a) a))
              4 (if (str/starts-with? descr "<f")
                  (let [a (float-array n)]  (.get (.asFloatBuffer bb) a) a)
                  (let [a (int-array n)]    (.get (.asIntBuffer bb) a) a)))
        data (if (transpose-needed? h)
               (let [[rows cols] shape]
                 (cond
                   (instance? (Class/forName "[D") raw) (transpose-f64 raw rows cols)
                   (instance? (Class/forName "[F") raw) (transpose-f32 raw rows cols)
                   :else (throw (ex-info "fortran_order transpose only implemented for f8/f4 2D arrays"
                                         {:descr descr :shape shape}))))
               raw)]
    {:shape shape :descr descr :fortran (:fortran h) :data data}))

(defn save-f64-2d
  "Write a flat C-order double[rows*cols] as a v1.0 .npy file readable by numpy."
  [path ^doubles data ^long rows ^long cols]
  (let [dict (str "{'descr': '<f8', 'fortran_order': False, 'shape': (" rows ", " cols "), }")
        ;; header (magic 6 + ver 2 + hlen 2 + dict) padded to 64-byte multiple, dict ends with \n
        base (+ 10 (count dict))
        pad (mod (- 64 (mod base 64)) 64)
        hdr (str dict (apply str (repeat pad \space)) "\n")
        hlen (count hdr)
        bb (doto (java.nio.ByteBuffer/allocate (+ 10 hlen (* rows cols 8)))
             (.order java.nio.ByteOrder/LITTLE_ENDIAN))]
    (.put bb (byte-array [(unchecked-byte 0x93) 78 85 77 80 89]))   ; \x93NUMPY
    (.put bb (byte 1)) (.put bb (byte 0))                           ; version 1.0
    (.putShort bb (short hlen))
    (.put bb (.getBytes ^String hdr java.nio.charset.StandardCharsets/US_ASCII))
    (let [db (.asDoubleBuffer bb)] (.put db data))
    (java.nio.file.Files/write (java.nio.file.Path/of path (make-array String 0)) (.array bb)
                               (make-array java.nio.file.OpenOption 0))
    path))

;; ---- typed convenience: always return C-order flat primitive arrays ----
(defn read-f64 ^doubles [path]
  (let [{:keys [data descr]} (load-npy path)]
    (if (instance? (Class/forName "[D") data) data
        (let [a (double-array (alength ^floats data))] (dotimes [i (alength ^floats data)] (aset a i (double (aget ^floats data i)))) a))))
(defn read-f32 ^floats [path]
  (let [{:keys [data]} (load-npy path)]
    (if (instance? (Class/forName "[F") data) data
        (let [n (alength ^doubles data) a (float-array n)] (dotimes [i n] (aset a i (float (aget ^doubles data i)))) a))))
(defn read-i32 ^ints [path]
  (let [{:keys [data]} (load-npy path)]
    (if (instance? (Class/forName "[I") data) data
        (let [n (alength ^longs data) a (int-array n)] (dotimes [i n] (aset a i (int (aget ^longs data i)))) a))))
