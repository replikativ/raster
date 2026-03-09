(ns raster.dl.compiled-test
  "Compiled DL correctness tests: compile-aot on conv2d, maxpool2d,
  dense with AD backward pass. Verifies gradients against finite differences."
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.core :refer [deftm]]
            [raster.nn :as nn]
            [raster.dl.nn :as dl]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pipeline]
            [raster.linalg.blas :as blas]))

(use-fixtures :once
  (fn [f] (if (blas/available?) (f) (println "[SKIP] No BLAS library"))))

(defn- finite-diff-grad
  "Compute gradient of scalar fn f w.r.t. array param at index i."
  [f ^doubles params ^long idx ^double eps]
  (let [orig (aget params idx)]
    (aset params idx (+ orig eps))
    (let [f+ (double (f))]
      (aset params idx (- orig eps))
      (let [f- (double (f))]
        (aset params idx orig)
        (/ (- f+ f-) (* 2.0 eps))))))

;; ================================================================
;; Test functions (top-level deftm for var resolution)
;; ================================================================

(deftm mlp-loss
  [W :- (Array double) b :- (Array double)
   x :- (Array double) y :- (Array double)] :- Double
  (let [h (nn/dense W x b) r (nn/relu h) p (nn/softmax r)]
    (nn/cross-entropy p y)))

(deftm mlp-vg
  [W :- (Array double) b :- (Array double)
   x :- (Array double) y :- (Array double)] :- Object
  ((rev/value+grad (var mlp-loss)) W b x y))

(deftm conv-loss
  [x :- (Array double) W :- (Array double) b :- (Array double)] :- Double
  (let [out (dl/conv2d x W b 1 1 4 4 2 3 3 1 1 0 0)
        n (clojure.core/alength out)]
    (loop [i 0 s 0.0]
      (if (< i n) (recur (inc i) (+ s (clojure.core/aget out i))) s))))

(deftm conv-vg
  [x :- (Array double) W :- (Array double) b :- (Array double)] :- Object
  ((rev/value+grad (var conv-loss)) x W b))

(deftm maxpool-loss
  [x :- (Array double)] :- Double
  (let [pooled (dl/maxpool2d x 1 1 4 4 2 2)]
    (+ (clojure.core/aget pooled 0) (* 2.0 (clojure.core/aget pooled 1)))))

(deftm maxpool-vg
  [x :- (Array double)] :- Object
  ((rev/value+grad (var maxpool-loss)) x))

;; ================================================================
;; Tests
;; ================================================================

(deftest compiled-mlp-gradient-test
  (testing "Compiled MLP gradient matches finite differences"
    (let [W (double-array [0.1 0.2 0.3 0.4 0.5 0.6])
          b (double-array [0.01 0.02])
          x (double-array [1.0 0.5 0.3])
          y (double-array [1.0 0.0])
          compiled (pipeline/compile-aot #'mlp-vg)
          result (compiled W b x y)
          ^doubles ad-dW (nth result 1)
          eps 1e-5]
      (dotimes [i 6]
        (let [fd (finite-diff-grad #(mlp-loss (aclone W) b x y) W i eps)
              ad (aget ad-dW i)]
          (is (< (Math/abs (- ad fd)) 1e-4)
              (format "dW[%d]: AD=%.6f FD=%.6f" i ad fd)))))))

(deftest compiled-conv2d-gradient-test
  (testing "Compiled conv2d gradient matches finite differences"
    (let [x (double-array (range 16))
          W (double-array (map #(* 0.1 %) (range 18)))
          b (double-array [0.1 0.2])
          compiled (pipeline/compile-aot #'conv-vg)
          result (compiled x W b)
          ^doubles ad-dW (nth result 2)
          eps 1e-5]
      (dotimes [i (min 6 (alength W))]
        (let [fd (finite-diff-grad #(conv-loss x (aclone W) b) W i eps)
              ad (aget ad-dW i)]
          (is (< (Math/abs (- ad fd)) 1e-3)
              (format "dW[%d]: AD=%.6f FD=%.6f" i ad fd)))))))

(deftest compiled-maxpool-gradient-test
  (testing "Compiled maxpool2d gradient has correct sparse pattern"
    (let [compiled (pipeline/compile-aot #'maxpool-vg)
          x (double-array [1 3 2 4  5 7 6 8  9 11 10 12  13 15 14 16])
          result (compiled x)
          ^doubles grad (nth result 1)]
      (is (== 1.0 (aget grad 5)) "grad at max pos 0")
      (is (== 2.0 (aget grad 7)) "grad at max pos 1")
      (is (== 0.0 (aget grad 0)) "non-max = 0")
      (is (== 0.0 (aget grad 1)) "non-max = 0"))))

;; ================================================================
;; Float32 compiled DL tests
;; ================================================================

(deftm mlp-loss-f32
  [W :- (Array float) b :- (Array float)
   x :- (Array float) y :- (Array float)] :- Double
  (let [h (nn/dense W x b) r (nn/relu h) p (nn/softmax r)]
    (nn/cross-entropy p y)))

(deftm mlp-vg-f32
  [W :- (Array float) b :- (Array float)
   x :- (Array float) y :- (Array float)] :- Object
  ((rev/value+grad (var mlp-loss-f32)) W b x y))

(deftest compiled-mlp-gradient-f32-test
  (testing "Compiled f32 MLP gradient matches finite differences"
    (let [W (float-array [0.1 0.2 0.3 0.4 0.5 0.6])
          b (float-array [0.01 0.02])
          x (float-array [1.0 0.5 0.3])
          y (float-array [1.0 0.0])
          compiled (pipeline/compile-aot #'mlp-vg-f32 :dtype :float)
          result (compiled W b x y)
          ^floats ad-dW (nth result 1)
          eps 1e-3]
      (dotimes [i 6]
        (let [fd (let [orig (aget W i)]
                   (aset W i (float (+ orig eps)))
                   (let [f+ (double (mlp-loss-f32 W b x y))]
                     (aset W i (float (- orig eps)))
                     (let [f- (double (mlp-loss-f32 W b x y))]
                       (aset W i orig)
                       (/ (- f+ f-) (* 2.0 eps)))))
              ad (aget ad-dW i)]
          (is (< (Math/abs (- ad fd)) 0.05)
              (format "f32 dW[%d]: AD=%.6f FD=%.6f" i ad fd)))))))
