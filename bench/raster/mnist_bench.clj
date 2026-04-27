(ns raster.mnist-bench
  "MNIST training benchmark: MLP and LeNet-5, f64 and f32 — single-sample SGD.

  Each train step is one rp/compile-train-step call: defmodel wraps the
  loss with a structured weight tree, value+grad + grad-clip + the per-leaf
  SGD update are fused into a single bytecoded kernel. Per-dtype defmodels
  exist because the defmodel surface is monomorphic.

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
            [raster.params :as rp]
            [raster.mnist-data :as data])
  (:import [java.util Random]))

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

(def ^:private sgd-lr 0.01)
(def ^:private sgd-clip 1.0)

(defn- run-mlp-bench [dtype train-imgs train-lbls test-imgs test-lbls-int]
  (let [dtype-str  (if (= dtype :float) "f32" "f64")
        model-var  (if (= dtype :float) #'mlp-loss-f32 #'mlp-loss-f64)
        n (count train-imgs)]
    (println (format "\n── MLP 784-128-10 %s ──" dtype-str))
    (print "  Compiling fused train step (SGD)... ") (flush)
    (let [tc (System/currentTimeMillis)
          {:keys [train-fn]} (rp/compile-train-step model-var {:optimizer :sgd})
          _ (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) tc) 1000.0)))
          w (data/init-mlp 128 (Random. 42) dtype)
          step! (fn [x y] (train-fn w x y sgd-clip sgd-lr))]
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
        (let [w (data/init-mlp 128 (Random. 42) dtype)
              step! (fn [x y] (train-fn w x y sgd-clip sgd-lr))]
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
        n (count train-imgs)]
    (println (format "\n── LeNet-5 %s ──" dtype-str))
    (print "  Compiling fused train step (SGD)... ") (flush)
    (let [tc (System/currentTimeMillis)
          compiled (try (rp/compile-train-step model-var {:optimizer :sgd})
                        (catch Throwable e
                          (println (format "failed: %s" (.getMessage e))) nil))]
      (when compiled
        (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) tc) 1000.0)))
        (let [{:keys [train-fn]} compiled
              w (data/init-lenet (Random. 42) dtype)
              step! (fn [x y] (train-fn w x y sgd-clip sgd-lr))]
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
            (let [w (data/init-lenet (Random. 42) dtype)
                  step! (fn [x y] (train-fn w x y sgd-clip sgd-lr))]
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
        f64 (data/load-mnist :double)
        f32 (data/load-mnist :float)
        test-imgs     (:test-imgs f64)
        test-lbls-int (:test-lbls-int f64)]
    (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) t0) 1000.0)))

    ;; MLP benchmarks
    (run-mlp-bench :double (:train-imgs f64) (:train-lbls f64) test-imgs test-lbls-int)
    (run-mlp-bench :float  (:train-imgs f32) (:train-lbls f32) test-imgs test-lbls-int)

    ;; LeNet benchmarks
    (run-lenet-bench :double (:train-imgs f64) (:train-lbls f64) test-imgs test-lbls-int)
    (run-lenet-bench :float  (:train-imgs f32) (:train-lbls f32) test-imgs test-lbls-int)

    (println "\n── Reference: JAX CPU (same machine) ──")
    (println "  MLP f64:   86 µs/step")
    (println "  MLP f32:   50 µs/step")
    (println "  LeNet f64: 370 µs/step")
    (println "  LeNet f32: 356 µs/step")
    (println "\nDone."))
  (shutdown-agents))
