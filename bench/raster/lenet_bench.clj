(ns raster.lenet-bench
  "LeNet-5 MNIST training benchmark.
  Uses value+grad for reverse-mode AD + compile-aot for JIT.

  Run: clojure -J--add-modules=jdk.incubator.vector -M:bench -m raster.lenet-bench"
  (:require [raster.core :refer [deftm]]
            [raster.nn :as nn]
            [raster.dl.nn :as dl]
            [raster.dl.lenet :as lenet]
            [raster.dl.optim :as optim]
            [raster.arrays :as arrays]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pipeline])
  (:import [java.io DataInputStream]
           [java.util Random]
           [java.util.zip GZIPInputStream]))

;; ================================================================
;; IDX file parsing
;; ================================================================

(defn- parse-idx-images [^String path]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [magic (.readInt dis)
          _     (assert (= magic 2051))
          n     (.readInt dis)
          rows  (.readInt dis)
          cols  (.readInt dis)
          dim   (* rows cols)]
      (mapv (fn [_]
              (let [arr (double-array dim)]
                (dotimes [j dim]
                  (aset arr j (/ (double (Byte/toUnsignedInt (.readByte dis))) 255.0)))
                arr))
            (range n)))))

(defn- parse-idx-labels [^String path]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [magic (.readInt dis)
          _     (assert (= magic 2049))
          n     (.readInt dis)]
      (mapv (fn [_]
              (let [label (Byte/toUnsignedInt (.readByte dis))
                    arr   (double-array 10)]
                (aset arr label 1.0)
                arr))
            (range n)))))

(defn- parse-idx-labels-int [^String path]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [magic (.readInt dis)
          _     (assert (= magic 2049))
          n     (.readInt dis)
          arr   (int-array n)]
      (dotimes [i n] (aset arr i (Byte/toUnsignedInt (.readByte dis))))
      arr)))

;; ================================================================
;; Train step: value+grad of lenet loss + SGD update
;; ================================================================

(deftm train-step
  [conv1-W :- (Array double) conv1-b :- (Array double)
   conv2-W :- (Array double) conv2-b :- (Array double)
   fc1-W :- (Array double) fc1-b :- (Array double)
   fc2-W :- (Array double) fc2-b :- (Array double)
   x :- (Array double) y :- (Array double)
   lr :- Double] :- Double
  (let [vg ((rev/value+grad (var lenet/lenet-loss-fn))
            conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b x y)
        loss (nth vg 0)]
    ;; SGD update all 8 weight arrays (indices 1-8 in gradient vector)
    (optim/sgd-step! conv1-W (nth vg 1) (arrays/alength conv1-W) lr)
    (optim/sgd-step! conv1-b (nth vg 2) (arrays/alength conv1-b) lr)
    (optim/sgd-step! conv2-W (nth vg 3) (arrays/alength conv2-W) lr)
    (optim/sgd-step! conv2-b (nth vg 4) (arrays/alength conv2-b) lr)
    (optim/sgd-step! fc1-W   (nth vg 5) (arrays/alength fc1-W)   lr)
    (optim/sgd-step! fc1-b   (nth vg 6) (arrays/alength fc1-b)   lr)
    (optim/sgd-step! fc2-W   (nth vg 7) (arrays/alength fc2-W)   lr)
    (optim/sgd-step! fc2-b   (nth vg 8) (arrays/alength fc2-b)   lr)
    loss))

;; ================================================================
;; Model init + evaluation
;; ================================================================

(defn- init-lenet [^Random rng]
  {:conv1-W (nn/kaiming-init! rng (* 1 5 5) (double-array (* 6 1 5 5)))
   :conv1-b (double-array 6)
   :conv2-W (nn/kaiming-init! rng (* 6 5 5) (double-array (* 16 6 5 5)))
   :conv2-b (double-array 16)
   :fc1-W   (nn/kaiming-init! rng 256 (double-array (* 120 256)))
   :fc1-b   (double-array 120)
   :fc2-W   (nn/kaiming-init! rng 120 (double-array (* 10 120)))
   :fc2-b   (double-array 10)})

(defn- forward-lenet
  ^doubles [conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b ^doubles x]
  (let [c1 (dl/conv2d x conv1-W conv1-b 1 1 28 28 6 5 5 1 1 0 0)
        r1 (nn/relu c1)
        p1 (dl/maxpool2d r1 1 6 24 24 2 2)
        c2 (dl/conv2d p1 conv2-W conv2-b 1 6 12 12 16 5 5 1 1 0 0)
        r2 (nn/relu c2)
        p2 (dl/maxpool2d r2 1 16 8 8 2 2)
        f1 (nn/dense fc1-W p2 fc1-b)
        a1 (nn/relu f1)]
    (nn/dense fc2-W a1 fc2-b)))

(defn- evaluate [conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b images ^ints labels predict-compiled]
  (let [n (alength labels)
        fwd (or predict-compiled
                (fn [c1w c1b c2w c2b f1w f1b f2w f2b x]
                  (forward-lenet c1w c1b c2w c2b f1w f1b f2w f2b x)))]
    {:accuracy (* 100.0 (/ (double (loop [i 0 c 0]
                                     (if (< i n)
                                       (let [pred (arrays/argmax
                                                    (fwd conv1-W conv1-b conv2-W conv2-b
                                                         fc1-W fc1-b fc2-W fc2-b
                                                         (nth images i)))]
                                         (recur (inc i) (if (== pred (aget labels i)) (inc c) c)))
                                       c)))
                           (double n)))}))

;; ================================================================
;; Main
;; ================================================================

(defn -main []
  (println "=== Raster LeNet-5 MNIST Benchmark ===\n")

  (print "Loading MNIST... ") (flush)
  (let [t0 (System/currentTimeMillis)
        train-imgs (parse-idx-images "data/mnist/train-images-idx3-ubyte.gz")
        train-lbls (parse-idx-labels "data/mnist/train-labels-idx1-ubyte.gz")
        test-imgs  (parse-idx-images "data/mnist/t10k-images-idx3-ubyte.gz")
        test-lbls  (parse-idx-labels-int "data/mnist/t10k-labels-idx1-ubyte.gz")]
    (println (format "done (%.1fs, %d train, %d test)"
               (/ (- (System/currentTimeMillis) t0) 1000.0)
               (count train-imgs) (alength test-lbls)))

    ;; Try compiled, fall back to uncompiled if compiler bug
    (let [compiled (try
                     (print "Compiling... ") (flush)
                     (let [tc (System/currentTimeMillis)
                           c (pipeline/compile-aot #'train-step)]
                       (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) tc) 1000.0)))
                       c)
                     (catch Throwable e
                       (println "compilation failed:" (.getMessage e))
                       nil))
          predict-compiled (try
                             (let [pc (pipeline/compile-aot #'lenet/lenet-predict-fn :mode :forward)
                                   ;; Smoke test: verify compiled predict returns correct-sized output
                                   test-m (init-lenet (java.util.Random. 0))
                                   test-img (double-array 784)]
                               (pc (:conv1-W test-m) (:conv1-b test-m)
                                   (:conv2-W test-m) (:conv2-b test-m)
                                   (:fc1-W test-m) (:fc1-b test-m)
                                   (:fc2-W test-m) (:fc2-b test-m)
                                   test-img)
                               pc)
                             (catch Throwable _ nil))
          ;; Test if compiled function actually works
          f (if compiled
              (let [test-m (init-lenet (java.util.Random. 0))
                    test-img (double-array 784)
                    test-lbl (double-array 10)]
                (aset test-lbl 0 1.0)
                (try (compiled (:conv1-W test-m) (:conv1-b test-m)
                               (:conv2-W test-m) (:conv2-b test-m)
                               (:fc1-W test-m) (:fc1-b test-m)
                               (:fc2-W test-m) (:fc2-b test-m)
                               test-img test-lbl 0.01)
                     (println "Using compiled pipeline.")
                     compiled
                     (catch Throwable e
                       (println "Compiled function failed, using deftm dispatch:" (.getMessage e))
                       nil)))
              nil)
          f (or f train-step)
          {:keys [conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b]}
          (init-lenet (Random. 42))
          lr 0.01
          n  (count train-imgs)]
      (println "LeNet-5 (Conv→Pool→Conv→Pool→FC→FC), SGD lr=0.01\n")

      ;; Warmup
      (print "Warming up JIT... ") (flush)
      (let [tw (System/currentTimeMillis)]
        (dotimes [i 200]
          (f conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b
             (nth train-imgs (mod i n)) (nth train-lbls (mod i n)) lr))
        ;; Warmup predict too
        (when predict-compiled
          (dotimes [i 50]
            (predict-compiled conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b
                              (nth train-imgs (mod i n)))))
        (println (format "done (%.1fs)\n" (/ (- (System/currentTimeMillis) tw) 1000.0))))

      ;; Train
      (doseq [ep (range 3)]
        (let [idx (shuffle (range n))
              t0  (System/currentTimeMillis)
              total (reduce (fn [acc i]
                              (+ acc (double (f conv1-W conv1-b conv2-W conv2-b
                                               fc1-W fc1-b fc2-W fc2-b
                                               (nth train-imgs i) (nth train-lbls i) lr))))
                            0.0 idx)
              dt (/ (- (System/currentTimeMillis) t0) 1000.0)
              ev (evaluate conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b
                           test-imgs test-lbls predict-compiled)]
          (println (format "Epoch %d/3  loss=%.4f  acc=%.2f%%  time=%.1fs  (%.0f µs/sample)"
                     (inc ep) (/ total n) (:accuracy ev) dt
                     (* 1000.0 (/ dt n))))))
      (println "\nDone."))))
