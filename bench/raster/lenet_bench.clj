(ns raster.lenet-bench
  "LeNet-5 MNIST training benchmark.

  Uses raster.params/compile-train-step: a defmodel wraps the loss function
  with a structured weight tree, and compile-train-step fuses
  value+grad + grad-clip + Adam into one bytecoded train kernel.

  Run: clojure -J--add-modules=jdk.incubator.vector -M:bench -m raster.lenet-bench"
  (:require [raster.nn :as nn]
            [raster.dl.nn :as dl]
            [raster.dl.lenet :as lenet]
            [raster.arrays :as arrays]
            [raster.params :as rp])
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
;; Loss model — structured weights, fused train step via compile-train-step
;; ================================================================

(rp/defmodel lenet-loss
  [w :- (Params (HMap :mandatory
                      {:conv1-W (Param (Array double)) :conv1-b (Param (Array double))
                       :conv2-W (Param (Array double)) :conv2-b (Param (Array double))
                       :fc1-W   (Param (Array double)) :fc1-b   (Param (Array double))
                       :fc2-W   (Param (Array double)) :fc2-b   (Param (Array double))}))
   x :- (Array double) y :- (Array double)] :- Double
  (lenet/lenet-loss-fn (:conv1-W w) (:conv1-b w) (:conv2-W w) (:conv2-b w)
                       (:fc1-W w) (:fc1-b w) (:fc2-W w) (:fc2-b w)
                       x y))

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

(defn- evaluate [w images ^ints labels predict-compiled]
  (let [n (alength labels)
        fwd (or predict-compiled
                (fn [w x] (forward-lenet (:conv1-W w) (:conv1-b w)
                                          (:conv2-W w) (:conv2-b w)
                                          (:fc1-W w) (:fc1-b w)
                                          (:fc2-W w) (:fc2-b w) x)))]
    {:accuracy (* 100.0 (/ (double (loop [i 0 c 0]
                                     (if (< i n)
                                       (let [pred (arrays/argmax (fwd w (nth images i)))]
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

    (print "Compiling fused train step (value+grad + grad-clip + Adam)... ") (flush)
    (let [tc (System/currentTimeMillis)
          {:keys [train-fn init-state]} (rp/compile-train-step #'lenet-loss)
          _ (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) tc) 1000.0)))
          predict-compiled (try
                             (let [{:keys [train-fn]} (rp/compile-train-step #'lenet-loss)]
                               ;; predict reuses lenet-predict-fn directly via lazy JIT
                               (fn [w x] (lenet/lenet-predict-fn
                                           (:conv1-W w) (:conv1-b w)
                                           (:conv2-W w) (:conv2-b w)
                                           (:fc1-W w) (:fc1-b w)
                                           (:fc2-W w) (:fc2-b w)
                                           x)))
                             (catch Throwable _ nil))
          w     (init-lenet (Random. 42))
          state (init-state w)
          m     (:m state)
          v     (:v state)
          ;; Adam hyperparameters (defaults match raster.params/compile-train-step)
          lr 0.01  beta1 0.9  beta2 0.999  eps 1e-8  max-grad-norm 1.0
          n  (count train-imgs)]
      (println "LeNet-5 (Conv→Pool→Conv→Pool→FC→FC), Adam lr=0.01\n")

      ;; Warmup: 200 fused train calls
      (print "Warming up JIT... ") (flush)
      (let [tw (System/currentTimeMillis)
            t-counter (atom 0)]
        (dotimes [i 200]
          (train-fn w m v (nth train-imgs (mod i n)) (nth train-lbls (mod i n))
                    max-grad-norm lr beta1 beta2 eps (swap! t-counter inc)))
        (println (format "done (%.1fs)\n" (/ (- (System/currentTimeMillis) tw) 1000.0)))

        ;; Train 3 epochs
        (doseq [ep (range 3)]
          (let [idx (shuffle (range n))
                t0  (System/currentTimeMillis)
                total (reduce (fn [acc i]
                                (+ acc (double (train-fn w m v
                                                         (nth train-imgs i) (nth train-lbls i)
                                                         max-grad-norm lr beta1 beta2 eps
                                                         (swap! t-counter inc)))))
                              0.0 idx)
                dt (/ (- (System/currentTimeMillis) t0) 1000.0)
                ev (evaluate w test-imgs test-lbls predict-compiled)]
            (println (format "Epoch %d/3  loss=%.4f  acc=%.2f%%  time=%.1fs  (%.0f µs/sample)"
                       (inc ep) (/ total n) (:accuracy ev) dt
                       (* 1000.0 (/ dt n))))))
        (println "\nDone.")))))
