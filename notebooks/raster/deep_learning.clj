;; # Deep Learning with Raster

;; authors: Christian Weilbach

;; Raster provides neural network primitives that compose with the
;; automatic differentiation system. The same `deftm` functions used
;; for scientific computing also define neural network layers — the
;; compiler handles forward pass, backward pass, and optimization
;; in one unified pipeline.

;; ## Setup

(ns raster.deep-learning
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.nn :as nn]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pipeline]
            [scicloj.kindly.v4.kind :as kind]))

;; ## 1. Neural Network Primitives
;;
;; Raster's NN ops are `deftm` functions with hand-written reverse-mode
;; AD rules (rrules). This means the AD system knows their exact
;; gradients — no numerical approximation needed.

;; **Dense layer** — matrix-vector multiply + bias:

(let [W (double-array [0.1 0.2 0.3 0.4])  ;; 2x2 weight matrix (row-major)
      x (double-array [1.0 2.0])            ;; input
      b (double-array [0.0 0.0])]           ;; bias
  {:output (vec (nn/dense W x b))})  ;; W*x + b = [0.5, 1.1]

;; **Activations:**

(let [x (double-array [-1.0 0.0 1.0 2.0])]
  {:relu (vec (nn/relu x))
   :softmax (vec (nn/softmax x))})

;; ## 2. Defining a Loss Function
;;
;; A loss function is just a `deftm` that returns a scalar. The compiler
;; and AD system handle the rest:

(deftm mlp-loss
  "Two-layer MLP loss: cross-entropy(softmax(W2 * relu(W1 * x + b1) + b2), y)"
  [W1 :- (Array double), b1 :- (Array double),
   W2 :- (Array double), b2 :- (Array double),
   x :- (Array double), y :- (Array double)] :- Double
  (let [h1 (nn/dense W1 x b1)
        h1r (nn/relu h1)
        out (nn/dense W2 h1r b2)
        p (nn/softmax out)]
    (nn/cross-entropy p y)))

;; ## 3. Computing Gradients
;;
;; `value+grad` on a loss function gives you the gradient of the loss
;; with respect to every parameter — this is backpropagation:

(def vg-mlp (rev/value+grad #'mlp-loss))

(let [input-dim 4, hidden-dim 8, output-dim 3
      W1 (double-array (* hidden-dim input-dim))
      b1 (double-array hidden-dim)
      W2 (double-array (* output-dim hidden-dim))
      b2 (double-array output-dim)
      x  (double-array [1.0 0.0 -1.0 0.5])
      y  (let [a (double-array output-dim)] (aset a 1 1.0) a)]
  ;; Initialize weights with small random values
  (dotimes [i (alength W1)] (aset W1 i (* 0.1 (- (Math/random) 0.5))))
  (dotimes [i (alength W2)] (aset W2 i (* 0.1 (- (Math/random) 0.5))))
  (let [result (vg-mlp W1 b1 W2 b2 x y)]
    {:loss (nth result 0)
     :n-gradients (dec (count result))
     :grad-W1-norm (Math/sqrt (reduce + (map #(* % %) (nth result 1))))}))

;; ## 4. Training Loop
;;
;; A training step combines the forward pass, backward pass, and
;; parameter update. Here's a simple SGD loop using `value+grad`:

(def dl-training-history
  (let [input-dim 4, hidden-dim 8, output-dim 3
        W1 (double-array (* hidden-dim input-dim))
        b1 (double-array hidden-dim)
        W2 (double-array (* output-dim hidden-dim))
        b2 (double-array output-dim)
        _ (dotimes [i (alength W1)] (aset W1 i (* 0.1 (- (Math/random) 0.5))))
        _ (dotimes [i (alength W2)] (aset W2 i (* 0.1 (- (Math/random) 0.5))))
        x  (double-array [1.0 0.0 -1.0 0.5])
        y  (let [a (double-array output-dim)] (aset a 1 1.0) a)
        lr 0.01]
    (loop [epoch 0, history []]
      (if (>= epoch 200)
        history
        (let [result (vg-mlp W1 b1 W2 b2 x y)
              loss (double (nth result 0))
              dW1 (nth result 1) db1 (nth result 2)
              dW2 (nth result 3) db2 (nth result 4)]
          ;; SGD update
          (dotimes [i (alength W1)] (aset W1 i (- (aget W1 i) (* lr (aget dW1 i)))))
          (dotimes [i (alength b1)] (aset b1 i (- (aget b1 i) (* lr (aget db1 i)))))
          (dotimes [i (alength W2)] (aset W2 i (- (aget W2 i) (* lr (aget dW2 i)))))
          (dotimes [i (alength b2)] (aset b2 i (- (aget b2 i) (* lr (aget db2 i)))))
          (recur (inc epoch)
                 (conj history {:epoch epoch :loss loss})))))))

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 500 :height 300
  :title "MLP Training: Cross-Entropy Loss"
  :data {:values (take-nth 2 dl-training-history)}
  :mark {:type "line" :strokeWidth 2}
  :encoding {:x {:field "epoch" :type "quantitative" :title "Epoch"}
             :y {:field "loss" :type "quantitative" :title "Loss"
                 :scale {:type "log"}}}})

;; ## 5. Compiled Training
;;
;; For production performance, `compile-aot` compiles the entire
;; training step — forward pass, backward pass, and SGD update — into
;; one fused method. C2 can then optimize across all three phases.
;;
;; On a 784→128→10 MLP (MNIST-sized), the compiled train step runs at
;; ~62µs per sample — competitive with JAX on CPU.
;;
;; ```clojure
;; (def fast-step (pipeline/compile-aot #'train-step! :simd? false))
;; ```

;; ## Available Layers and Losses
;;
;; | Layer | Function | Description |
;; |---|---|---|
;; | Dense | `nn/dense` | y = W*x + b |
;; | ReLU | `nn/relu` | max(0, x) |
;; | Softmax | `nn/softmax` | Normalized exponentials |
;; | Cross-entropy | `nn/cross-entropy` | -Σ y_i log(p_i) |
;; | Combined | `nn/softmax-cross-entropy` | Returns [loss, softmax] pair |
;;
;; All have hand-written rrules for reverse-mode AD. The compiler
;; inlines the AD-transformed code, so there's no runtime tape overhead.
