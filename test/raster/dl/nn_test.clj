(ns raster.dl.nn-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.dl.nn :as nn]
            [raster.dl.tensor :as t]
            [raster.ad.templates :as tmpl]
            [raster.linalg.blas :as blas]))

;; ================================================================
;; Helpers
;; ================================================================

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

(defn- arr-approx=
  ([a b] (arr-approx= a b 1e-5))
  ([^doubles a ^doubles b eps]
   (and (= (alength a) (alength b))
        (every? true?
                (for [i (range (alength a))]
                  (< (Math/abs (- (aget a i) (aget b i))) (double eps)))))))

(defn- numerical-grad-array
  "Finite difference gradient of scalar function f w.r.t. array a."
  [f ^doubles a eps]
  (let [n (alength a)
        grad (double-array n)
        f0 (double (f))]
    (dotimes [i n]
      (let [orig (aget a i)]
        (aset a i (+ orig eps))
        (let [f+ (double (f))]
          (aset a i orig)
          (aset grad i (/ (- f+ f0) eps)))))
    grad))

;; ================================================================
;; Matmul tests
;; ================================================================

(deftest matmul-test
  (when-not (blas/available?) (println "  [SKIP] No BLAS library"))
  (when (blas/available?)
    (testing "2x3 @ 3x2 matrix multiply"
      ;; A = [[1 2 3], [4 5 6]]
      ;; B = [[7 8], [9 10], [11 12]]
      ;; C = [[58 64], [139 154]]
      (let [A (double-array [1 2 3 4 5 6])
            B (double-array [7 8 9 10 11 12])
            C (nn/matmul A B 2 3 2)]
        (is (approx= 58.0 (aget C 0)))
        (is (approx= 64.0 (aget C 1)))
        (is (approx= 139.0 (aget C 2)))
        (is (approx= 154.0 (aget C 3)))))

    (testing "identity matrix multiply"
      (let [A (double-array [1 0 0 1])
            B (double-array [5 6 7 8])
            C (nn/matmul A B 2 2 2)]
        (is (arr-approx= B C))))))

(deftest matmul-gradient-test
  (when-not (blas/available?) (println "  [SKIP] No BLAS library"))
  (when (blas/available?)
    (testing "matmul gradients vs finite diff"
      (let [A (double-array [0.1 0.2 0.3 0.4 0.5 0.6])
            B (double-array [0.7 0.8 0.9 1.0 1.1 1.2])
            rrfn (tmpl/template-pullback 'raster.dl.nn/matmul)
            C (nn/matmul A B 2 3 2)
            pb (rrfn C A B 2 3 2)
            ;; Use dC = ones as upstream gradient
            dC (double-array [1 1 1 1])
            [dA dB _ _ _] (pb dC)
            ;; sum(C) as scalar loss
            num-dA (numerical-grad-array
                    #(let [c (nn/matmul A B 2 3 2)]
                       (loop [i 0 s 0.0]
                         (if (< i 4) (recur (inc i) (+ s (aget c i))) s)))
                    A 1e-5)
            num-dB (numerical-grad-array
                    #(let [c (nn/matmul A B 2 3 2)]
                       (loop [i 0 s 0.0]
                         (if (< i 4) (recur (inc i) (+ s (aget c i))) s)))
                    B 1e-5)]
        (is (arr-approx= dA num-dA 1e-3))
        (is (arr-approx= dB num-dB 1e-3))))))

;; ================================================================
;; Linear layer tests
;; ================================================================

(deftest linear-test
  (when-not (blas/available?) (println "  [SKIP] No BLAS library"))
  (when (blas/available?)
    (testing "linear forward y = x @ W^T + b"
      ;; x:[1,3] = [[1 2 3]]
      ;; W:[2,3] = [[1 0 0], [0 1 0]]
      ;; b:[2] = [0.1, 0.2]
      ;; y = x @ W^T + b = [[1, 2]] + [[0.1, 0.2]] = [[1.1, 2.2]]
      (let [x (double-array [1 2 3])
            W (double-array [1 0 0 0 1 0])
            b (double-array [0.1 0.2])
            y (nn/linear x W b 1 3 2)]
        (is (approx= 1.1 (aget y 0)))
        (is (approx= 2.2 (aget y 1)))))

    (testing "linear batched"
      (let [x (double-array [1 2 3 4 5 6])  ;; [2, 3]
            W (double-array [1 0 0 0 1 0])  ;; [2, 3]
            b (double-array [0 0])
            y (nn/linear x W b 2 3 2)]
        ;; batch 0: [1, 2], batch 1: [4, 5]
        (is (approx= 1.0 (aget y 0)))
        (is (approx= 2.0 (aget y 1)))
        (is (approx= 4.0 (aget y 2)))
        (is (approx= 5.0 (aget y 3)))))))

(deftest linear-gradient-test
  (when-not (blas/available?) (println "  [SKIP] No BLAS library"))
  (when (blas/available?)
    (testing "linear gradients vs finite diff"
      (let [x (double-array [0.1 0.2 0.3])
            W (double-array [0.4 0.5 0.6 0.7 0.8 0.9])
            b (double-array [0.01 0.02])
            rrfn (tmpl/template-pullback 'raster.dl.nn/linear)
            y (nn/linear x W b 1 3 2)
            pb (rrfn y x W b 1 3 2)
            dy (double-array [1 1])
            [dx dW db _ _ _] (pb dy)
            num-dW (numerical-grad-array
                    #(let [y (nn/linear x W b 1 3 2)]
                       (+ (aget y 0) (aget y 1)))
                    W 1e-5)
            num-dx (numerical-grad-array
                    #(let [y (nn/linear x W b 1 3 2)]
                       (+ (aget y 0) (aget y 1)))
                    x 1e-5)]
        (is (arr-approx= dW num-dW 1e-3))
        (is (arr-approx= dx num-dx 1e-3))
        (is (arr-approx= db dy 1e-6))))))

;; ================================================================
;; Activation tests
;; ================================================================

(deftest silu-test
  (testing "silu forward"
    (let [x (double-array [-2 -1 0 1 2])
          y (nn/silu x 5)]
      ;; silu(0) = 0 * 0.5 = 0
      (is (approx= 0.0 (aget y 2)))
      ;; silu is monotonically increasing for positive x
      (is (> (aget y 4) (aget y 3)))))

  (testing "silu gradient"
    (let [x (double-array [0.5 1.0 -0.5])
          rrfn (tmpl/template-pullback 'raster.dl.nn/silu)
          y (nn/silu x 3)
          pb (rrfn y x 3)
          dy (double-array [1 1 1])
          [dx _] (pb dy)
          num-dx (numerical-grad-array
                  #(let [y (nn/silu x 3)]
                     (+ (aget y 0) (aget y 1) (aget y 2)))
                  x 1e-5)]
      (is (arr-approx= dx num-dx 1e-4)))))

(deftest gelu-test
  (testing "gelu forward"
    (let [x (double-array [0 1 -1])
          y (nn/gelu x 3)]
      ;; gelu(0) = 0
      (is (approx= 0.0 (aget y 0) 1e-4))
      ;; gelu(1) ≈ 0.841
      (is (approx= 0.841 (aget y 1) 0.01))))

  (testing "gelu gradient"
    (let [x (double-array [0.5 1.0 -0.5])
          rrfn (tmpl/template-pullback 'raster.dl.nn/gelu)
          y (nn/gelu x 3)
          pb (rrfn y x 3)
          dy (double-array [1 1 1])
          [dx _] (pb dy)
          num-dx (numerical-grad-array
                  #(let [y (nn/gelu x 3)]
                     (+ (aget y 0) (aget y 1) (aget y 2)))
                  x 1e-5)]
      (is (arr-approx= dx num-dx 1e-4)))))

(deftest sigmoid-test
  (testing "sigmoid forward"
    (let [x (double-array [0 -100 100])
          y (nn/sigmoid x 3)]
      (is (approx= 0.5 (aget y 0)))
      (is (approx= 0.0 (aget y 1) 1e-10))
      (is (approx= 1.0 (aget y 2) 1e-10))))

  (testing "sigmoid gradient"
    (let [x (double-array [0.5 -0.5 1.0])
          rrfn (tmpl/template-pullback 'raster.dl.nn/sigmoid)
          y (nn/sigmoid x 3)
          pb (rrfn y x 3)
          dy (double-array [1 1 1])
          [dx _] (pb dy)
          num-dx (numerical-grad-array
                  #(let [y (nn/sigmoid x 3)]
                     (+ (aget y 0) (aget y 1) (aget y 2)))
                  x 1e-5)]
      (is (arr-approx= dx num-dx 1e-4)))))

(deftest tanh-act-test
  (testing "tanh forward"
    (let [x (double-array [0 1 -1])
          y (nn/tanh-act x 3)]
      (is (approx= 0.0 (aget y 0)))
      (is (approx= (Math/tanh 1.0) (aget y 1)))))

  (testing "tanh gradient"
    (let [x (double-array [0.5 -0.5 1.0])
          rrfn (tmpl/template-pullback 'raster.dl.nn/tanh-act)
          y (nn/tanh-act x 3)
          pb (rrfn y x 3)
          dy (double-array [1 1 1])
          [dx _] (pb dy)
          num-dx (numerical-grad-array
                  #(let [y (nn/tanh-act x 3)]
                     (+ (aget y 0) (aget y 1) (aget y 2)))
                  x 1e-5)]
      (is (arr-approx= dx num-dx 1e-4)))))

(deftest leaky-relu-test
  (testing "leaky-relu forward"
    (let [x (double-array [-2 -1 0 1 2])
          y (nn/leaky-relu x 5 0.01)]
      (is (approx= -0.02 (aget y 0)))
      (is (approx= 0.0 (aget y 2)))
      (is (approx= 2.0 (aget y 4)))))

  (testing "leaky-relu gradient"
    (let [x (double-array [0.5 -0.5 1.0])
          rrfn (tmpl/template-pullback 'raster.dl.nn/leaky-relu)
          y (nn/leaky-relu x 3 0.01)
          pb (rrfn y x 3 0.01)
          dy (double-array [1 1 1])
          [dx _ _] (pb dy)]
      (is (approx= 1.0 (aget dx 0)))
      (is (approx= 0.01 (aget dx 1)))
      (is (approx= 1.0 (aget dx 2))))))

;; ================================================================
;; Layer Norm tests
;; ================================================================

(deftest layer-norm-test
  (testing "layer-norm forward normalizes"
    (let [x (double-array [1 2 3 4 5 6])  ;; [2, 3]
          gamma (double-array [1 1 1])
          beta (double-array [0 0 0])
          y (nn/layer-norm x gamma beta 2 3 1e-5)]
      ;; Each row should have ~zero mean and ~unit variance
      (let [mean (/ (+ (aget y 0) (aget y 1) (aget y 2)) 3.0)]
        (is (approx= 0.0 mean 1e-4)))))

  (testing "layer-norm gradient"
    (let [x (double-array [0.1 0.2 0.3 0.4 0.5 0.6])
          gamma (double-array [1.0 1.0 1.0])
          beta (double-array [0.0 0.0 0.0])
          rrfn (tmpl/template-pullback 'raster.dl.nn/layer-norm)
          y (nn/layer-norm x gamma beta 2 3 1e-5)
          pb (rrfn y x gamma beta 2 3 1e-5)
          dy (double-array [1 0 0 0 1 0])
          [dx dgamma dbeta _ _ _] (pb dy)
          num-dx (numerical-grad-array
                  #(let [y (nn/layer-norm x gamma beta 2 3 1e-5)]
                     (+ (aget y 0) (aget y 4)))
                  x 1e-5)]
      (is (arr-approx= dx num-dx 1e-3)))))

;; ================================================================
;; Group Norm tests
;; ================================================================

(deftest group-norm-test
  (testing "group-norm with 1 group = layer-norm-like"
    (let [;; [1, 2, 3] batch=1, channels=2, spatial=3
          x (double-array [1 2 3 4 5 6])
          gamma (double-array [1 1])
          beta (double-array [0 0])
          y (nn/group-norm x gamma beta 1 2 3 1 1e-5)]
      ;; With 1 group, all 6 elements normalized together
      (let [sum (loop [i 0 s 0.0]
                  (if (< i 6) (recur (inc i) (+ s (aget y i))) s))]
        (is (approx= 0.0 sum 1e-3))))))

;; ================================================================
;; Batch Norm tests
;; ================================================================

(deftest batch-norm-test
  (testing "batch-norm training mode"
    (let [x (double-array [1 2 3 4])  ;; [2, 2]
          gamma (double-array [1 1])
          beta (double-array [0 0])
          rm (double-array [0 0])
          rv (double-array [1 1])
          y (nn/batch-norm x gamma beta rm rv 2 2 1e-5 0.1 1)]
      ;; After normalization, each feature should have ~zero mean
      (is (approx= 0.0 (+ (aget y 0) (aget y 2)) 1e-4))
      (is (approx= 0.0 (+ (aget y 1) (aget y 3)) 1e-4)))))

;; ================================================================
;; Conv1d tests
;; ================================================================

(deftest conv1d-test
  (when-not (blas/available?) (println "  [SKIP] No BLAS library"))
  (when (blas/available?)
    (testing "conv1d forward"
      ;; x:[1,1,5] = [1,2,3,4,5]
      ;; W:[1,1,3] = [1,1,1]
      ;; b:[1] = [0]
      ;; y = [6, 9, 12] (sum of sliding window, no pad, stride=1)
      (let [x (double-array [1 2 3 4 5])
            W (double-array [1 1 1])
            b (double-array [0])
            y (nn/conv1d x W b 1 1 5 1 3 1 0)]
        (is (= 3 (alength y)))
        (is (approx= 6.0 (aget y 0)))
        (is (approx= 9.0 (aget y 1)))
        (is (approx= 12.0 (aget y 2)))))))

(deftest conv1d-gradient-test
  (when-not (blas/available?) (println "  [SKIP] No BLAS library"))
  (when (blas/available?)
    (testing "conv1d gradients vs finite diff"
      (let [x (double-array [0.1 0.2 0.3 0.4 0.5])
            W (double-array [0.1 0.2 0.3])
            b (double-array [0.01])
            rrfn (tmpl/template-pullback 'raster.dl.nn/conv1d)
            y (nn/conv1d x W b 1 1 5 1 3 1 0)
            pb (rrfn y x W b 1 1 5 1 3 1 0)
            dy (double-array (repeat (alength y) 1.0))
            [dx dW db _ _ _ _ _ _] (pb dy)
            num-dW (numerical-grad-array
                    #(let [y (nn/conv1d x W b 1 1 5 1 3 1 0)]
                       (loop [i 0 s 0.0]
                         (if (< i (alength y)) (recur (inc i) (+ s (aget y i))) s)))
                    W 1e-5)
            num-dx (numerical-grad-array
                    #(let [y (nn/conv1d x W b 1 1 5 1 3 1 0)]
                       (loop [i 0 s 0.0]
                         (if (< i (alength y)) (recur (inc i) (+ s (aget y i))) s)))
                    x 1e-5)]
        (is (arr-approx= dW num-dW 1e-3))
        (is (arr-approx= dx num-dx 1e-3))))))

;; ================================================================
;; MaxPool2d tests
;; ================================================================

(deftest maxpool2d-test
  (testing "maxpool2d forward 1x1x4x4 with 2x2 kernel"
    ;; x:[1,1,4,4] - single batch, single channel
    ;; Pool with 2x2 -> output [1,1,2,2]
    (let [x (double-array [1  2  3  4
                           5  6  7  8
                           9 10 11 12
                           13 14 15 16])
          y (nn/maxpool2d x 1 1 4 4 2 2)]
      (is (= 4 (alength y)))
      ;; Window (0,0)-(1,1): max(1,2,5,6) = 6
      (is (approx= 6.0 (aget y 0)))
      ;; Window (0,2)-(1,3): max(3,4,7,8) = 8
      (is (approx= 8.0 (aget y 1)))
      ;; Window (2,0)-(3,1): max(9,10,13,14) = 14
      (is (approx= 14.0 (aget y 2)))
      ;; Window (2,2)-(3,3): max(11,12,15,16) = 16
      (is (approx= 16.0 (aget y 3)))))

  (testing "maxpool2d forward multi-channel"
    ;; x:[1,2,2,2] - 1 batch, 2 channels, 2x2 spatial
    ;; Pool with 2x2 -> output [1,2,1,1]
    (let [x (double-array [1 2 3 4   ;; channel 0: max = 4
                           5 6 7 8])  ;; channel 1: max = 8
          y (nn/maxpool2d x 1 2 2 2 2 2)]
      (is (= 2 (alength y)))
      (is (approx= 4.0 (aget y 0)))
      (is (approx= 8.0 (aget y 1)))))

  (testing "maxpool2d forward multi-batch"
    ;; x:[2,1,2,2] - 2 batches, 1 channel, 2x2 spatial
    ;; Pool with 2x2 -> output [2,1,1,1]
    (let [x (double-array [1 2 3 4   ;; batch 0: max = 4
                           5 6 7 8])  ;; batch 1: max = 8
          y (nn/maxpool2d x 2 1 2 2 2 2)]
      (is (= 2 (alength y)))
      (is (approx= 4.0 (aget y 0)))
      (is (approx= 8.0 (aget y 1))))))

(deftest maxpool2d-gradient-test
  (testing "maxpool2d gradient scatters to argmax"
    ;; x:[1,1,4,4], kh=2, kw=2 -> y:[1,1,2,2]
    (let [x (double-array [1  2  3  4
                           5  6  7  8
                           9 10 11 12
                           13 14 15 16])
          rrfn (tmpl/template-pullback 'raster.dl.nn/maxpool2d)
          y (nn/maxpool2d x 1 1 4 4 2 2)
          pb (rrfn y x 1 1 4 4 2 2)
          dy (double-array [1.0 2.0 3.0 4.0])
          [dx _ _ _ _ _ _] (pb dy)]
      ;; Gradient should only appear at argmax positions:
      ;; (1,1)=6 -> 1.0, (1,3)=8 -> 2.0, (3,1)=14 -> 3.0, (3,3)=16 -> 4.0
      ;; Position 5 = (1,1) -> 1.0
      (is (approx= 1.0 (aget dx 5)))
      ;; Position 7 = (1,3) -> 2.0
      (is (approx= 2.0 (aget dx 7)))
      ;; Position 13 = (3,1) -> 3.0
      (is (approx= 3.0 (aget dx 13)))
      ;; Position 15 = (3,3) -> 4.0
      (is (approx= 4.0 (aget dx 15)))
      ;; All other positions should be zero
      (is (approx= 0.0 (aget dx 0)))
      (is (approx= 0.0 (aget dx 1)))
      (is (approx= 0.0 (aget dx 6)))
      (is (approx= 0.0 (aget dx 10)))))

  (testing "maxpool2d gradient vs finite diff"
    (let [x (double-array [0.1 0.9 0.3 0.7
                           0.5 0.2 0.8 0.4
                           0.6 0.3 0.1 0.5
                           0.4 0.8 0.7 0.2])
          rrfn (tmpl/template-pullback 'raster.dl.nn/maxpool2d)
          y (nn/maxpool2d x 1 1 4 4 2 2)
          pb (rrfn y x 1 1 4 4 2 2)
          dy (double-array [1 1 1 1])
          [dx _ _ _ _ _ _] (pb dy)
          num-dx (numerical-grad-array
                  #(let [y (nn/maxpool2d x 1 1 4 4 2 2)]
                     (loop [i 0 s 0.0]
                       (if (< i (alength y)) (recur (inc i) (+ s (aget y i))) s)))
                  x 1e-7)]
      (is (arr-approx= dx num-dx 1e-3)))))

;; ================================================================
;; Embedding tests
;; ================================================================

(deftest embedding-test
  (testing "embedding lookup"
    (let [table (double-array [0.1 0.2 0.3  ;; word 0
                               0.4 0.5 0.6  ;; word 1
                               0.7 0.8 0.9]) ;; word 2
          indices (long-array [2 0 1])
          out (nn/embedding table indices 3 3 3)]
      ;; First row should be word 2
      (is (approx= 0.7 (aget out 0)))
      (is (approx= 0.8 (aget out 1)))
      (is (approx= 0.9 (aget out 2)))
      ;; Second row should be word 0
      (is (approx= 0.1 (aget out 3)))))

  (testing "embedding gradient"
    (let [table (double-array [0.1 0.2  0.3 0.4  0.5 0.6])
          indices (long-array [1 0])
          rrfn (tmpl/template-pullback 'raster.dl.nn/embedding)
          out (nn/embedding table indices 2 3 2)
          pb (rrfn out table indices 2 3 2)
          dy (double-array [1 0 0 1])
          [d-table _ _ _ _] (pb dy)]
      ;; d_table[0,:] gets dy[1,:] = [0 1] (index 0 was looked up at position 1)
      (is (approx= 0.0 (aget d-table 0)))
      (is (approx= 1.0 (aget d-table 1)))
      ;; d_table[1,:] gets dy[0,:] = [1 0] (index 1 was looked up at position 0)
      (is (approx= 1.0 (aget d-table 2)))
      (is (approx= 0.0 (aget d-table 3))))))

;; ================================================================
;; Dropout tests
;; ================================================================

(deftest dropout-test
  (testing "dropout with all-keep mask"
    (let [x (double-array [1 2 3 4 5])
          mask (double-array [1 1 1 1 1])  ;; keep all (p=0)
          y (nn/dropout x mask 5)]
      (is (arr-approx= x y))))

  (testing "dropout with all-drop mask"
    (let [x (double-array [1 2 3 4 5])
          mask (double-array [0 0 0 0 0])
          y (nn/dropout x mask 5)]
      (is (every? zero? (seq y))))))

;; ================================================================
;; Softmax-1d tests
;; ================================================================

(deftest softmax-1d-test
  (testing "softmax sums to 1"
    (let [x (double-array [1 2 3])
          s (nn/softmax-1d x 3)
          sum (+ (aget s 0) (aget s 1) (aget s 2))]
      (is (approx= 1.0 sum))))

  (testing "softmax gradient"
    (let [x (double-array [1.0 2.0 3.0])
          rrfn (tmpl/template-pullback 'raster.dl.nn/softmax-1d)
          s (nn/softmax-1d x 3)
          pb (rrfn s x 3)
          dy (double-array [1 0 0])
          [dx _] (pb dy)
          num-dx (numerical-grad-array
                  #(aget (nn/softmax-1d x 3) 0)
                  x 1e-5)]
      (is (arr-approx= dx num-dx 1e-4)))))

;; ================================================================
;; Transpose tests
;; ================================================================

(deftest transpose-2d-test
  (testing "transpose"
    (let [a (double-array [1 2 3 4 5 6])  ;; [2,3]
          b (nn/transpose-2d a 2 3)]      ;; [3,2]
      (is (= 1.0 (aget b 0)))  ;; (0,0)
      (is (= 4.0 (aget b 1)))  ;; (0,1)
      (is (= 2.0 (aget b 2)))  ;; (1,0)
      (is (= 5.0 (aget b 3)))  ;; (1,1)
      (is (= 3.0 (aget b 4)))  ;; (2,0)
      (is (= 6.0 (aget b 5)))))) ;; (2,1)

;; ================================================================
;; Weight init tests
;; ================================================================

(deftest he-init-test
  (testing "he-init creates correct size"
    (let [w (nn/he-init 3 5)]
      (is (= 15 (alength w))))))

(deftest xavier-init-test
  (testing "xavier-init creates correct size"
    (let [w (nn/xavier-init 3 5)]
      (is (= 15 (alength w))))))
