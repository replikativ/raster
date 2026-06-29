(ns raster.lenet-bench
  "LeNet-5 MNIST training benchmark — single-sample SGD, f64.

  Uses raster.params/compile-train-step with :optimizer :sgd. A deftm with
  HMap-typed weight args expands at compile time into a flat-arg deftm;
  compile-train-step fuses value+grad + grad-clip + the per-leaf SGD
  update into one bytecoded train kernel.

  Run: clojure -J--add-modules=jdk.incubator.vector -M:bench -m raster.lenet-bench"
  (:require [raster.core :as core]
            [raster.nn :as nn]
            [raster.dl.nn :as dl]
            [raster.dl.lenet :as lenet]
            [raster.arrays :as arrays]
            [raster.params :as rp]
            [raster.mnist-data :as data])
  (:import [java.util Random]))

;; ================================================================
;; Loss model — structured weights, fused train step via compile-train-step
;; ================================================================

(core/deftm lenet-loss
  [w :- (HMap :mandatory
              {:conv1-W (Param (Array double)) :conv1-b (Param (Array double))
               :conv2-W (Param (Array double)) :conv2-b (Param (Array double))
               :fc1-W   (Param (Array double)) :fc1-b   (Param (Array double))
               :fc2-W   (Param (Array double)) :fc2-b   (Param (Array double))})
   x :- (Array double) y :- (Array double)] :- Double
  (lenet/lenet-loss-fn (:conv1-W w) (:conv1-b w) (:conv2-W w) (:conv2-b w)
                       (:fc1-W w) (:fc1-b w) (:fc2-W w) (:fc2-b w)
                       x y))

;; ================================================================
;; Forward path for evaluation (lazy JIT — only used to predict accuracy
;; off the test set, never on the hot training loop)
;; ================================================================

(defn- forward-lenet
  ^doubles [w ^doubles x]
  (let [{:keys [conv1-W conv1-b conv2-W conv2-b fc1-W fc1-b fc2-W fc2-b]} w
        c1 (dl/conv2d x conv1-W conv1-b 1 1 28 28 6 5 5 1 1 0 0)
        r1 (nn/relu c1)
        p1 (dl/maxpool2d r1 1 6 24 24 2 2)
        c2 (dl/conv2d p1 conv2-W conv2-b 1 6 12 12 16 5 5 1 1 0 0)
        r2 (nn/relu c2)
        p2 (dl/maxpool2d r2 1 16 8 8 2 2)
        f1 (nn/dense fc1-W p2 fc1-b)
        a1 (nn/relu f1)]
    (nn/dense fc2-W a1 fc2-b)))

(defn- evaluate [w images ^ints labels]
  (let [n (alength labels)]
    {:accuracy (* 100.0 (/ (double (loop [i 0 c 0]
                                     (if (< i n)
                                       (let [pred (arrays/argmax (forward-lenet w (nth images i)))]
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
        {:keys [train-imgs train-lbls test-imgs test-lbls-int]} (data/load-mnist :double)]
    (println (format "done (%.1fs, %d train, %d test)"
               (/ (- (System/currentTimeMillis) t0) 1000.0)
               (count train-imgs) (alength test-lbls-int)))

    (print "Compiling fused train step (value+grad + grad-clip + SGD)... ") (flush)
    (let [tc (System/currentTimeMillis)
          {:keys [train-fn]} (rp/compile-train-step #'lenet-loss {:optimizer :sgd})
          _ (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) tc) 1000.0)))
          w     (data/init-lenet (Random. 42) :double)
          lr 0.01  max-grad-norm 1.0
          n  (count train-imgs)]
      (println "LeNet-5 (Conv→Pool→Conv→Pool→FC→FC), SGD lr=0.01\n")

      ;; Warmup: 200 fused train calls
      (print "Warming up JIT... ") (flush)
      (let [tw (System/currentTimeMillis)]
        (dotimes [i 200]
          (train-fn w (nth train-imgs (mod i n)) (nth train-lbls (mod i n))
                    max-grad-norm lr))
        (println (format "done (%.1fs)\n" (/ (- (System/currentTimeMillis) tw) 1000.0)))

        ;; Train 3 epochs
        (doseq [ep (range 3)]
          (let [idx (shuffle (range n))
                t0  (System/currentTimeMillis)
                total (reduce (fn [acc i]
                                (+ acc (double (train-fn w (nth train-imgs i) (nth train-lbls i)
                                                         max-grad-norm lr))))
                              0.0 idx)
                dt (/ (- (System/currentTimeMillis) t0) 1000.0)
                ev (evaluate w test-imgs test-lbls-int)]
            (println (format "Epoch %d/3  loss=%.4f  acc=%.2f%%  time=%.1fs  (%.0f µs/sample)"
                       (inc ep) (/ total n) (:accuracy ev) dt
                       (* 1000.0 (/ dt n))))))
        (println "\nDone.")))))
