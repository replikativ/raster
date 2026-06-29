(ns raster.mnist-data
  "Shared MNIST loading + LeNet/MLP weight initialization for the bench
  namespaces. Both raster.lenet-bench and raster.mnist-bench load the
  same IDX files and init the same architectures — keeping one copy
  here means changes propagate."
  (:require [raster.nn :as nn])
  (:import [java.io DataInputStream]
           [java.util Random]
           [java.util.zip GZIPInputStream]))

;; ================================================================
;; IDX file parsing (gzipped, big-endian magic + dims + bytes)
;; ================================================================

(defn parse-idx-images
  "Read MNIST images (IDX format, gzipped). dtype ∈ {:double :float}.
  Returns a vector of n typed arrays of length rows*cols, each pixel
  scaled to [0, 1]."
  [^String path dtype]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [magic (.readInt dis)
          _     (assert (= magic 2051) (str "bad image magic " magic))
          n     (.readInt dis)
          rows  (.readInt dis)
          cols  (.readInt dis)
          dim   (* rows cols)]
      (mapv (fn [_]
              (if (= dtype :float)
                (let [arr (float-array dim)]
                  (dotimes [j dim]
                    (aset arr j (float (/ (double (Byte/toUnsignedInt (.readByte dis))) 255.0))))
                  arr)
                (let [arr (double-array dim)]
                  (dotimes [j dim]
                    (aset arr j (/ (double (Byte/toUnsignedInt (.readByte dis))) 255.0)))
                  arr)))
            (range n)))))

(defn parse-idx-labels
  "Read MNIST labels (IDX format, gzipped) as one-hot vectors of length 10.
  dtype ∈ {:double :float}."
  [^String path dtype]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [magic (.readInt dis)
          _     (assert (= magic 2049) (str "bad label magic " magic))
          n     (.readInt dis)]
      (mapv (fn [_]
              (let [lbl (Byte/toUnsignedInt (.readByte dis))]
                (if (= dtype :float)
                  (let [arr (float-array 10)] (aset arr lbl (float 1.0)) arr)
                  (let [arr (double-array 10)] (aset arr lbl 1.0) arr))))
            (range n)))))

(defn parse-idx-labels-int
  "Read MNIST labels as a flat int-array of class indices [0..9]."
  [^String path]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [_ (.readInt dis) n (.readInt dis) arr (int-array n)]
      (dotimes [i n] (aset arr i (Byte/toUnsignedInt (.readByte dis))))
      arr)))

;; ================================================================
;; One-shot loader: returns the full benchmark dataset.
;; ================================================================

(def ^:private default-dir "data/mnist")

(defn data-dir
  "Resolve the MNIST data directory. Honors -Draster.bench.mnist-dir, then
  the MNIST_DIR env var, then defaults to data/mnist relative to CWD."
  []
  (or (System/getProperty "raster.bench.mnist-dir")
      (System/getenv "MNIST_DIR")
      default-dir))

(defn data-available?
  "True iff all four MNIST IDX files are present in (data-dir)."
  []
  (let [d (data-dir)]
    (every? #(.exists (java.io.File. (str d "/" %)))
            ["train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz"
             "t10k-images-idx3-ubyte.gz"  "t10k-labels-idx1-ubyte.gz"])))

(defn load-mnist
  "Load MNIST. dtype ∈ {:double :float} controls train arrays. Test images
  are always loaded as :double; the eval helpers convert per-sample if needed.

  Returns {:train-imgs vec, :train-lbls vec, :test-imgs vec, :test-lbls-int int[]}."
  [dtype]
  (let [d (data-dir)]
    (when-not (data-available?)
      (throw (ex-info (str "MNIST data not found in " d
                           ". Set raster.bench.mnist-dir or MNIST_DIR, or "
                           "place the four .gz files there.")
                      {:dir d})))
    {:train-imgs    (parse-idx-images (str d "/train-images-idx3-ubyte.gz") dtype)
     :train-lbls    (parse-idx-labels (str d "/train-labels-idx1-ubyte.gz") dtype)
     :test-imgs     (parse-idx-images (str d "/t10k-images-idx3-ubyte.gz")  :double)
     :test-lbls-int (parse-idx-labels-int (str d "/t10k-labels-idx1-ubyte.gz"))}))

;; ================================================================
;; Model initialization
;; ================================================================

(defn init-mlp
  "MLP: 784 → hidden → 10. Xavier-uniform init scaled by Math/sqrt(6/(in+out))."
  [^long hidden ^Random rng dtype]
  (if (= dtype :float)
    (let [l1 (Math/sqrt (/ 6.0 (+ 784.0 (double hidden))))
          l2 (Math/sqrt (/ 6.0 (+ (double hidden) 10.0)))
          W1 (float-array (* 784 hidden))
          W2 (float-array (* hidden 10))]
      (dotimes [i (* 784 hidden)] (aset W1 i (float (- (* (.nextDouble rng) 2.0 l1) l1))))
      (dotimes [i (* hidden 10)] (aset W2 i (float (- (* (.nextDouble rng) 2.0 l2) l2))))
      {:W1 W1 :b1 (float-array hidden) :W2 W2 :b2 (float-array 10)})
    {:W1 (nn/xavier-init! rng 784 hidden (double-array (* 784 hidden)))
     :b1 (double-array hidden)
     :W2 (nn/xavier-init! rng hidden 10 (double-array (* hidden 10)))
     :b2 (double-array 10)}))

(defn init-lenet
  "LeNet-5: conv(1→6,5×5) → pool → conv(6→16,5×5) → pool → fc(256→120) → fc(120→10).
  Kaiming-normal init for weights, zero biases."
  [^Random rng dtype]
  (if (= dtype :float)
    (let [kaiming (fn [^long fan-in ^long n]
                    (let [s (Math/sqrt (/ 2.0 (double fan-in)))
                          a (float-array n)]
                      (dotimes [i n] (aset a i (float (* (.nextGaussian rng) s))))
                      a))]
      {:conv1-W (kaiming 25 (* 6 1 5 5))   :conv1-b (float-array 6)
       :conv2-W (kaiming 150 (* 16 6 5 5)) :conv2-b (float-array 16)
       :fc1-W   (kaiming 256 (* 120 256))  :fc1-b   (float-array 120)
       :fc2-W   (kaiming 120 (* 10 120))   :fc2-b   (float-array 10)})
    {:conv1-W (nn/kaiming-init! rng (* 1 5 5) (double-array (* 6 1 5 5)))
     :conv1-b (double-array 6)
     :conv2-W (nn/kaiming-init! rng (* 6 5 5) (double-array (* 16 6 5 5)))
     :conv2-b (double-array 16)
     :fc1-W   (nn/kaiming-init! rng 256 (double-array (* 120 256)))
     :fc1-b   (double-array 120)
     :fc2-W   (nn/kaiming-init! rng 120 (double-array (* 10 120)))
     :fc2-b   (double-array 10)}))
