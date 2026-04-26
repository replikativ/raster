(ns raster.mnist-bench
  "MNIST training benchmark: MLP and LeNet-5, f64 and f32.

  Each train step is one rp/compile-train-step call: defmodel wraps the
  loss with a structured weight tree, value+grad + grad-clip + Adam are
  fused into a single bytecoded kernel.

  Run with Valhalla JDK:
    source valhalla-env.sh
    OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \\
      -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \\
      -J--enable-native-access=ALL-UNNAMED \\
      -J--add-modules=jdk.incubator.vector \\
      -J-XX:ReservedCodeCacheSize=512m -J-Xmx8g \\
      -M:bench:valhalla -m raster.mnist-bench"
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

(defn- parse-idx-images [^String path dtype]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [_ (.readInt dis) n (.readInt dis) _ (.readInt dis) _ (.readInt dis)]
      (mapv (fn [_]
              (if (= dtype :float)
                (let [arr (float-array 784)]
                  (dotimes [j 784] (aset arr j (float (/ (double (Byte/toUnsignedInt (.readByte dis))) 255.0))))
                  arr)
                (let [arr (double-array 784)]
                  (dotimes [j 784] (aset arr j (/ (double (Byte/toUnsignedInt (.readByte dis))) 255.0)))
                  arr)))
            (range n)))))

(defn- parse-idx-labels [^String path dtype]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [_ (.readInt dis) n (.readInt dis)]
      (mapv (fn [_]
              (let [lbl (Byte/toUnsignedInt (.readByte dis))]
                (if (= dtype :float)
                  (let [arr (float-array 10)] (aset arr lbl (float 1.0)) arr)
                  (let [arr (double-array 10)] (aset arr lbl 1.0) arr))))
            (range n)))))

(defn- parse-idx-labels-int [^String path]
  (with-open [dis (DataInputStream. (GZIPInputStream. (java.io.FileInputStream. path)))]
    (let [_ (.readInt dis) n (.readInt dis) arr (int-array n)]
      (dotimes [i n] (aset arr i (Byte/toUnsignedInt (.readByte dis)))) arr)))

;; ================================================================
;; MLP loss models (one per dtype — defmodel surface is monomorphic)
;; ================================================================

(rp/defmodel mlp-loss-f64
  [w :- (Params (HMap :mandatory {:W1 (Param (Array double)) :b1 (Param (Array double))
                                  :W2 (Param (Array double)) :b2 (Param (Array double))}))
   x :- (Array double) y :- (Array double)] :- Double
  (nn/loss-fn (:W1 w) (:b1 w) (:W2 w) (:b2 w) x y))

(rp/defmodel mlp-loss-f32
  [w :- (Params (HMap :mandatory {:W1 (Param (Array float)) :b1 (Param (Array float))
                                  :W2 (Param (Array float)) :b2 (Param (Array float))}))
   x :- (Array float) y :- (Array float)] :- Float
  (nn/loss-fn (:W1 w) (:b1 w) (:W2 w) (:b2 w) x y))

;; ================================================================
;; LeNet loss models (one per dtype)
;; ================================================================

(rp/defmodel lenet-loss-f64
  [w :- (Params (HMap :mandatory
                      {:conv1-W (Param (Array double)) :conv1-b (Param (Array double))
                       :conv2-W (Param (Array double)) :conv2-b (Param (Array double))
                       :fc1-W   (Param (Array double)) :fc1-b   (Param (Array double))
                       :fc2-W   (Param (Array double)) :fc2-b   (Param (Array double))}))
   x :- (Array double) y :- (Array double)] :- Double
  (lenet/lenet-loss-fn (:conv1-W w) (:conv1-b w) (:conv2-W w) (:conv2-b w)
                       (:fc1-W w) (:fc1-b w) (:fc2-W w) (:fc2-b w)
                       x y))

(rp/defmodel lenet-loss-f32
  [w :- (Params (HMap :mandatory
                      {:conv1-W (Param (Array float)) :conv1-b (Param (Array float))
                       :conv2-W (Param (Array float)) :conv2-b (Param (Array float))
                       :fc1-W   (Param (Array float)) :fc1-b   (Param (Array float))
                       :fc2-W   (Param (Array float)) :fc2-b   (Param (Array float))}))
   x :- (Array float) y :- (Array float)] :- Float
  (lenet/lenet-loss-fn (:conv1-W w) (:conv1-b w) (:conv2-W w) (:conv2-b w)
                       (:fc1-W w) (:fc1-b w) (:fc2-W w) (:fc2-b w)
                       x y))

;; ================================================================
;; Model initialization
;; ================================================================

(defn- init-mlp [^long hidden ^Random rng dtype]
  (if (= dtype :float)
    (let [l1 (Math/sqrt (/ 6.0 (+ 784.0 hidden)))
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

(defn- init-lenet [^Random rng dtype]
  (if (= dtype :float)
    (let [kaiming (fn [^long fan-in ^long n]
                    (let [s (Math/sqrt (/ 2.0 (double fan-in)))
                          a (float-array n)]
                      (dotimes [i n] (aset a i (float (* (.nextGaussian rng) s))))
                      a))]
      {:conv1-W (kaiming 25 (* 6 1 5 5))
       :conv1-b (float-array 6)
       :conv2-W (kaiming 150 (* 16 6 5 5))
       :conv2-b (float-array 16)
       :fc1-W   (kaiming 256 (* 120 256))
       :fc1-b   (float-array 120)
       :fc2-W   (kaiming 120 (* 10 120))
       :fc2-b   (float-array 10)})
    {:conv1-W (nn/kaiming-init! rng (* 1 5 5) (double-array (* 6 1 5 5)))
     :conv1-b (double-array 6)
     :conv2-W (nn/kaiming-init! rng (* 6 5 5) (double-array (* 16 6 5 5)))
     :conv2-b (double-array 16)
     :fc1-W   (nn/kaiming-init! rng 256 (double-array (* 120 256)))
     :fc1-b   (double-array 120)
     :fc2-W   (nn/kaiming-init! rng 120 (double-array (* 10 120)))
     :fc2-b   (double-array 10)}))

;; ================================================================
;; Evaluation
;; ================================================================

(defn- mlp-evaluate [w images ^ints labels dtype]
  (let [{:keys [W1 b1 W2 b2]} w
        n (alength labels)
        float? (= dtype :float)
        fwd (fn [x] (nn/dense W2 (nn/relu (nn/dense W1 x b1)) b2))]
    (* 100.0 (/ (double (loop [i 0 c 0]
                           (if (< i n)
                             (let [img (nth images i)
                                   ;; eval images are always f64; convert if needed
                                   img (if float?
                                         (let [^doubles img img
                                               f (float-array (clojure.core/alength img))]
                                           (dotimes [j (clojure.core/alength img)] (aset f j (float (aget img j)))) f)
                                         img)
                                   pred (arrays/argmax (fwd img))]
                               (recur (inc i) (if (== pred (aget labels i)) (inc c) c)))
                             c)))
               (double n)))))

(defn- lenet-evaluate [w images ^ints labels dtype]
  (let [{:keys [conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b]} w
        n (alength labels)
        float? (= dtype :float)
        fwd (fn [x]
              (let [c1 (dl/conv2d x conv1-W conv1-b 1 1 28 28 6 5 5 1 1 0 0)
                    r1 (nn/relu c1)
                    p1 (dl/maxpool2d r1 1 6 24 24 2 2)
                    c2 (dl/conv2d p1 conv2-W conv2-b 1 6 12 12 16 5 5 1 1 0 0)
                    r2 (nn/relu c2)
                    p2 (dl/maxpool2d r2 1 16 8 8 2 2)
                    f1 (nn/dense fc1-W p2 fc1-b)
                    a1 (nn/relu f1)]
                (nn/dense fc2-W a1 fc2-b)))]
    (* 100.0 (/ (double (loop [i 0 c 0]
                           (if (< i n)
                             (let [img (nth images i)
                                   img (if float?
                                         (let [^doubles img img
                                               f (float-array (clojure.core/alength img))]
                                           (dotimes [j (clojure.core/alength img)] (aset f j (float (aget img j)))) f)
                                         img)
                                   pred (arrays/argmax (fwd img))]
                               (recur (inc i) (if (== pred (aget labels i)) (inc c) c)))
                             c)))
               (double n)))))

;; ================================================================
;; Micro-benchmark harness (median of N runs on single samples)
;; ================================================================

(defn- micro-bench
  "Time f over n-steps calls, return median µs/call."
  [f n-warmup n-steps]
  (dotimes [_ n-warmup] (f))
  (System/gc) (Thread/sleep 200)
  (let [times (long-array n-steps)]
    (dotimes [i n-steps]
      (let [t0 (System/nanoTime)] (f) (aset times i (- (System/nanoTime) t0))))
    (java.util.Arrays/sort times)
    (/ (double (aget times (int (* n-steps 0.5)))) 1e3)))

;; ================================================================
;; Run benchmarks
;; ================================================================

;; Adam hyperparameters (defaults match raster.params/compile-train-step)
(def ^:private adam-lr 0.01)
(def ^:private adam-beta1 0.9)
(def ^:private adam-beta2 0.999)
(def ^:private adam-eps 1e-8)
(def ^:private adam-clip 1.0)

(defn- run-mlp-bench [dtype train-imgs train-lbls test-imgs test-lbls-int]
  (let [dtype-str  (if (= dtype :float) "f32" "f64")
        model-var  (if (= dtype :float) #'mlp-loss-f32 #'mlp-loss-f64)
        n (count train-imgs)
        t-counter (atom 0)]
    (println (format "\n── MLP 784-128-10 %s ──" dtype-str))
    (print "  Compiling fused train step... ") (flush)
    (let [tc (System/currentTimeMillis)
          {:keys [train-fn init-state]} (rp/compile-train-step model-var)
          _ (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) tc) 1000.0)))
          w (init-mlp 128 (Random. 42) dtype)
          {:keys [m v]} (init-state w)
          step! (fn [^doubles x ^doubles y]
                  (train-fn w m v x y adam-clip adam-lr adam-beta1 adam-beta2 adam-eps
                            (swap! t-counter inc)))]
      ;; Micro-benchmark: median µs/step
      (let [n-warmup (Integer/parseInt (System/getProperty "raster.bench.warmup" "3000"))
            n-timed  (Integer/parseInt (System/getProperty "raster.bench.timed" "5000"))]
        (printf "  Micro-bench (%d warmup, %d timed)... " n-warmup n-timed) (flush)
        (let [us (micro-bench #(step! (nth train-imgs (rand-int n))
                                      (nth train-lbls (rand-int n)))
                              n-warmup n-timed)]
          (println (format "%.0f µs/step" us))))
      ;; Full epoch (skip with -Draster.bench.skip-epoch=true)
      (when-not (= "true" (System/getProperty "raster.bench.skip-epoch"))
        (let [w (init-mlp 128 (Random. 42) dtype)
              {:keys [m v]} (init-state w)
              step! (fn [^doubles x ^doubles y]
                      (train-fn w m v x y adam-clip adam-lr adam-beta1 adam-beta2 adam-eps
                                (swap! t-counter inc)))]
          (print "  Warming up... ") (flush)
          (dotimes [i 1000] (step! (nth train-imgs (mod i n)) (nth train-lbls (mod i n))))
          (println "done")
          (let [idx (shuffle (range n))
                t0 (System/currentTimeMillis)
                total (reduce (fn [acc i]
                                (+ acc (double (step! (nth train-imgs i) (nth train-lbls i)))))
                              0.0 idx)
                dt (/ (- (System/currentTimeMillis) t0) 1000.0)
                acc (mlp-evaluate w test-imgs test-lbls-int dtype)]
            (println (format "  Epoch 1  loss=%.4f  acc=%.2f%%  time=%.1fs  (%.0f µs/step)"
                       (/ total n) acc dt (* 1e6 (/ dt n))))))))))

(defn- run-lenet-bench [dtype train-imgs train-lbls test-imgs test-lbls-int]
  (let [dtype-str (if (= dtype :float) "f32" "f64")
        model-var (if (= dtype :float) #'lenet-loss-f32 #'lenet-loss-f64)
        n (count train-imgs)
        t-counter (atom 0)]
    (println (format "\n── LeNet-5 %s ──" dtype-str))
    (print "  Compiling fused train step... ") (flush)
    (let [tc (System/currentTimeMillis)
          compiled (try (rp/compile-train-step model-var)
                        (catch Throwable e
                          (println (format "failed: %s" (.getMessage e))) nil))]
      (when compiled
        (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) tc) 1000.0)))
        (let [{:keys [train-fn init-state]} compiled
              w (init-lenet (Random. 42) dtype)
              {:keys [m v]} (init-state w)
              step! (fn [^doubles x ^doubles y]
                      (train-fn w m v x y adam-clip adam-lr adam-beta1 adam-beta2 adam-eps
                                (swap! t-counter inc)))]
          ;; Micro-benchmark
          (let [n-warmup (Integer/parseInt (System/getProperty "raster.bench.lenet.warmup" "500"))
                n-timed  (Integer/parseInt (System/getProperty "raster.bench.lenet.timed" "2000"))]
            (printf "  Micro-bench (%d warmup, %d timed)... " n-warmup n-timed) (flush)
            (let [us (micro-bench #(step! (nth train-imgs (rand-int n))
                                          (nth train-lbls (rand-int n)))
                                  n-warmup n-timed)]
              (println (format "%.0f µs/step" us))))
          ;; Full epoch (skip with -Draster.bench.skip-epoch=true)
          (when-not (= "true" (System/getProperty "raster.bench.skip-epoch"))
            (let [w (init-lenet (Random. 42) dtype)
                  {:keys [m v]} (init-state w)
                  step! (fn [^doubles x ^doubles y]
                          (train-fn w m v x y adam-clip adam-lr adam-beta1 adam-beta2 adam-eps
                                    (swap! t-counter inc)))]
              (print "  Warming up... ") (flush)
              (dotimes [i 200] (step! (nth train-imgs (mod i n)) (nth train-lbls (mod i n))))
              (println "done")
              (let [idx (shuffle (range n))
                    t0 (System/currentTimeMillis)
                    total (reduce (fn [acc i]
                                    (+ acc (double (step! (nth train-imgs i) (nth train-lbls i)))))
                                  0.0 idx)
                    dt (/ (- (System/currentTimeMillis) t0) 1000.0)
                    acc (lenet-evaluate w test-imgs test-lbls-int dtype)]
                (println (format "  Epoch 1  loss=%.4f  acc=%.2f%%  time=%.1fs  (%.0f µs/step)"
                           (/ total n) acc dt (* 1e6 (/ dt n))))))))))))

(defn -main [& args]
  (println "=== Raster MNIST DL Benchmark (compiled pipeline + reverse-mode AD) ===")
  (println (format "    JDK: %s" (System/getProperty "java.version")))

  (print "Loading MNIST... ") (flush)
  (let [t0 (System/currentTimeMillis)
        train-imgs-f64 (parse-idx-images "data/mnist/train-images-idx3-ubyte.gz" :double)
        train-lbls-f64 (parse-idx-labels "data/mnist/train-labels-idx1-ubyte.gz" :double)
        train-imgs-f32 (parse-idx-images "data/mnist/train-images-idx3-ubyte.gz" :float)
        train-lbls-f32 (parse-idx-labels "data/mnist/train-labels-idx1-ubyte.gz" :float)
        ;; Test images always f64 (eval converts if needed)
        test-imgs (parse-idx-images "data/mnist/t10k-images-idx3-ubyte.gz" :double)
        test-lbls-int (parse-idx-labels-int "data/mnist/t10k-labels-idx1-ubyte.gz")]
    (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) t0) 1000.0)))

    ;; MLP benchmarks
    (run-mlp-bench :double train-imgs-f64 train-lbls-f64 test-imgs test-lbls-int)
    (run-mlp-bench :float  train-imgs-f32 train-lbls-f32 test-imgs test-lbls-int)

    ;; LeNet benchmarks
    (run-lenet-bench :double train-imgs-f64 train-lbls-f64 test-imgs test-lbls-int)
    (run-lenet-bench :float  train-imgs-f32 train-lbls-f32 test-imgs test-lbls-int)

    (println "\n── Reference: JAX CPU (same machine) ──")
    (println "  MLP f64:   86 µs/step")
    (println "  MLP f32:   50 µs/step")
    (println "  LeNet f64: 370 µs/step")
    (println "  LeNet f32: 356 µs/step")
    (println "\nDone."))
  (shutdown-agents))
