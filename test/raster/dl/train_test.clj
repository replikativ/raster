(ns raster.dl.train-test
  "Integration tests for end-to-end training loops.
  Tests that layers, loss, backward passes, and optimizers work together."
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.optim :as optim]
            [raster.dl.train :as train]
            [raster.ad.templates :as tmpl]
            [raster.linalg.blas :as blas]))

(use-fixtures :once
  (fn [f] (if (blas/available?) (f) (println "[SKIP] No BLAS library"))))

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

(defn- arr-copy ^doubles [^doubles a]
  (let [out (double-array (alength a))]
    (System/arraycopy a 0 out 0 (alength a))
    out))

(defn- arr-sum ^double [^doubles a]
  (loop [i 0 s 0.0]
    (if (< i (alength a))
      (recur (inc i) (+ s (aget a i)))
      s)))

;; ================================================================
;; Forward pass shape tests
;; ================================================================

(deftest forward-shape-test
  (testing "2-layer MLP produces correct output shape"
    ;; Input: [batch=4, in=2]
    ;; Layer 1: linear(2 -> 8), leaky-relu
    ;; Layer 2: linear(8 -> 1)
    (let [batch 4
          in-dim 2 hidden 8 out-dim 1
          W1 (nn/he-init hidden in-dim)   ;; [8, 2]
          b1 (double-array hidden)
          W2 (nn/he-init out-dim hidden)  ;; [1, 8]
          b2 (double-array out-dim)
          x (double-array (repeatedly (* batch in-dim) #(- (rand) 0.5)))
          ;; Forward
          h (nn/linear x W1 b1 batch in-dim hidden)
          a (nn/leaky-relu h (* batch hidden) 0.01)
          y (nn/linear a W2 b2 batch hidden out-dim)]
      (is (= (* batch hidden) (alength h))
          "hidden layer output should be batch*hidden")
      (is (= (* batch hidden) (alength a))
          "activation output should preserve shape")
      (is (= (* batch out-dim) (alength y))
          "output layer should be batch*out_dim")))

  (testing "single sample forward"
    (let [W1 (nn/he-init 4 3)
          b1 (double-array 4)
          x (double-array [1.0 2.0 3.0])
          h (nn/linear x W1 b1 1 3 4)]
      (is (= 4 (alength h))))))

;; ================================================================
;; Optimizer step direction test
;; ================================================================

(deftest optimizer-step-direction-test
  (testing "SGD moves weights in negative gradient direction"
    (let [param (double-array [1.0 -2.0 3.0])
          param-before (arr-copy param)
          grad (double-array [0.5 -0.3 0.8])
          lr 0.1
          _ (optim/sgd-step! param grad 3 lr)]
      ;; param_new = param_old - lr * grad
      (dotimes [i 3]
        (is (approx= (- (aget param-before i) (* lr (aget grad i)))
                     (aget param i))
            (str "SGD step direction at index " i)))))

  (testing "Adam step moves toward minimum on simple loss"
    ;; param starts at 5.0, loss = param^2, grad = 2*param
    ;; After one step, param should decrease (move toward 0)
    (let [param (double-array [5.0])
          m (double-array [0.0])
          v (double-array [0.0])
          param-before (aget param 0)
          grad (double-array [(* 2.0 (aget param 0))])
          _ (optim/adam-step! param grad m v 1 0.01 0.9 0.999 1e-8 1)]
      (is (< (Math/abs (aget param 0)) (Math/abs param-before))
          "Adam should move param closer to zero"))))

;; ================================================================
;; Gradient accumulation test
;; ================================================================

(deftest gradient-accumulation-test
  (testing "multiple backward passes on linear layer accumulate gradients"
    (let [;; Setup: linear y = x @ W^T + b, batch=1, in=2, out=2
          x1 (double-array [1.0 0.0])
          x2 (double-array [0.0 1.0])
          W (double-array [0.5 0.3 0.2 0.7])  ;; [2, 2]
          b (double-array [0.0 0.0])
          linear-rrule (tmpl/get-pullback-factory 'raster.dl.nn/linear)

          ;; Forward + backward for x1
          y1 (nn/linear x1 W b 1 2 2)
          pb1 (linear-rrule y1 x1 W b 1 2 2)
          dy (double-array [1.0 1.0])
          [_dx1 dW1 db1 _ _ _] (pb1 dy)

          ;; Forward + backward for x2
          y2 (nn/linear x2 W b 1 2 2)
          pb2 (linear-rrule y2 x2 W b 1 2 2)
          [_dx2 dW2 db2 _ _ _] (pb2 dy)

          ;; Manually accumulate gradients
          acc-dW (double-array (alength dW1))
          acc-db (double-array (alength db1))
          _ (dotimes [i (alength acc-dW)]
              (aset acc-dW i (+ (aget dW1 i) (aget dW2 i))))
          _ (dotimes [i (alength acc-db)]
              (aset acc-db i (+ (aget db1 i) (aget db2 i))))

          ;; Compare with batch forward: x=[x1; x2] batch=2
          x-batch (double-array [1.0 0.0 0.0 1.0])
          y-batch (nn/linear x-batch W b 2 2 2)
          pb-batch (linear-rrule y-batch x-batch W b 2 2 2)
          dy-batch (double-array [1.0 1.0 1.0 1.0])
          [_dx-batch dW-batch db-batch _ _ _] (pb-batch dy-batch)]

      (is (arr-approx= acc-dW dW-batch 1e-10)
          "accumulated weight grads should match batched grad")
      (is (arr-approx= acc-db db-batch 1e-10)
          "accumulated bias grads should match batched grad"))))

;; ================================================================
;; End-to-end training loop: sine regression
;; ================================================================

(deftest sine-regression-training-test
  (testing "MLP learns sine function via training loop"
    ;; Dataset: y = sin(x) for x in [-pi, pi], 200 samples
    (let [n-samples 200
          in-dim 1 hidden 16 out-dim 1
          ;; Generate dataset
          dataset (vec
                   (for [i (range n-samples)]
                     (let [x (* (- (* 2.0 (/ (double i) n-samples)) 1.0) Math/PI)
                           y (Math/sin x)]
                       [(double-array [x]) (double-array [y])])))
          ;; Initialize weights
          W1 (nn/he-init hidden in-dim)
          b1 (double-array hidden)
          W2 (nn/he-init out-dim hidden)
          b2 (double-array out-dim)
          ;; Optimizer state
          adam-state {:W1 {:m (double-array (* hidden in-dim))
                           :v (double-array (* hidden in-dim))
                           :t (atom 0)}
                      :b1 {:m (double-array hidden)
                           :v (double-array hidden)
                           :t (atom 0)}
                      :W2 {:m (double-array (* out-dim hidden))
                           :v (double-array (* out-dim hidden))
                           :t (atom 0)}
                      :b2 {:m (double-array out-dim)
                           :v (double-array out-dim)
                           :t (atom 0)}}
          lr 0.01
          ;; rrules
          linear-rrule (tmpl/get-pullback-factory 'raster.dl.nn/linear)
          relu-rrule (tmpl/get-pullback-factory 'raster.dl.nn/leaky-relu)
          mse-rrule (tmpl/get-pullback-factory 'raster.dl.loss/mse-loss)
          ;; Model function: forward + backward + update, returns loss
          model-fn
          (fn [batch]
            (let [xs (train/collate-doubles (:xs batch))
                  ys (train/collate-doubles (:ys batch))
                  bs (:batch-size batch)
                  ;; Forward
                  h (nn/linear xs W1 b1 bs in-dim hidden)
                  a (nn/leaky-relu h (* bs hidden) 0.01)
                  pred (nn/linear a W2 b2 bs hidden out-dim)
                  loss-val (loss/mse-loss pred ys (* bs out-dim))
                  ;; Backward: loss -> pred -> a -> h -> inputs
                  loss-pb (mse-rrule loss-val pred ys (* bs out-dim))
                  [d-pred _ _] (loss-pb 1.0)
                  pred-pb (linear-rrule pred a W2 b2 bs hidden out-dim)
                  [d-a dW2-g db2-g _ _ _] (pred-pb d-pred)
                  relu-pb (relu-rrule a h (* bs hidden) 0.01)
                  [d-h _ _] (relu-pb d-a)
                  h-pb (linear-rrule h xs W1 b1 bs in-dim hidden)
                  [_d-xs dW1-g db1-g _ _ _] (h-pb d-h)]
              ;; Update parameters with Adam
              (let [step-W1 (swap! (:t (:W1 adam-state)) inc)]
                (optim/adam-step! W1 dW1-g (:m (:W1 adam-state)) (:v (:W1 adam-state))
                                  (* hidden in-dim) lr 0.9 0.999 1e-8 step-W1))
              (let [step-b1 (swap! (:t (:b1 adam-state)) inc)]
                (optim/adam-step! b1 db1-g (:m (:b1 adam-state)) (:v (:b1 adam-state))
                                  hidden lr 0.9 0.999 1e-8 step-b1))
              (let [step-W2 (swap! (:t (:W2 adam-state)) inc)]
                (optim/adam-step! W2 dW2-g (:m (:W2 adam-state)) (:v (:W2 adam-state))
                                  (* out-dim hidden) lr 0.9 0.999 1e-8 step-W2))
              (let [step-b2 (swap! (:t (:b2 adam-state)) inc)]
                (optim/adam-step! b2 db2-g (:m (:b2 adam-state)) (:v (:b2 adam-state))
                                  out-dim lr 0.9 0.999 1e-8 step-b2))
              loss-val))
          ;; Train for 100 epochs, record first and last epoch loss
          batch-size 32
          first-epoch-loss (train/train-epoch! model-fn dataset batch-size
                                               :shuffle? true)
          _ (dotimes [_ 98]
              (train/train-epoch! model-fn dataset batch-size :shuffle? true))
          last-epoch-loss (train/train-epoch! model-fn dataset batch-size
                                              :shuffle? true)]
      (is (> first-epoch-loss last-epoch-loss)
          (str "loss should decrease: first=" first-epoch-loss
               " last=" last-epoch-loss))
      (is (< last-epoch-loss (* 0.5 first-epoch-loss))
          "loss should decrease by at least 50%")
      (is (< last-epoch-loss 0.1)
          (str "final loss should be small, got " last-epoch-loss)))))

;; ================================================================
;; End-to-end training: XOR classification
;; ================================================================

(deftest xor-training-test
  (testing "MLP learns XOR pattern"
    (let [;; XOR dataset: 4 samples, repeated for enough data
          xor-data [[0.0 0.0 0.0]
                    [0.0 1.0 1.0]
                    [1.0 0.0 1.0]
                    [1.0 1.0 0.0]]
          dataset (vec
                   (for [_ (range 50)  ;; repeat for mini-batching
                         [x1 x2 y] xor-data]
                     [(double-array [x1 x2]) (double-array [y])]))
          in-dim 2 hidden 8 out-dim 1
          ;; Initialize weights
          W1 (nn/he-init hidden in-dim)
          b1 (double-array hidden)
          W2 (nn/he-init out-dim hidden)
          b2 (double-array out-dim)
          ;; Optimizer state
          adam-state {:W1 {:m (double-array (* hidden in-dim))
                           :v (double-array (* hidden in-dim))
                           :t (atom 0)}
                      :b1 {:m (double-array hidden)
                           :v (double-array hidden)
                           :t (atom 0)}
                      :W2 {:m (double-array (* out-dim hidden))
                           :v (double-array (* out-dim hidden))
                           :t (atom 0)}
                      :b2 {:m (double-array out-dim)
                           :v (double-array out-dim)
                           :t (atom 0)}}
          lr 0.01
          linear-rrule (tmpl/get-pullback-factory 'raster.dl.nn/linear)
          relu-rrule (tmpl/get-pullback-factory 'raster.dl.nn/leaky-relu)
          mse-rrule (tmpl/get-pullback-factory 'raster.dl.loss/mse-loss)
          model-fn
          (fn [batch]
            (let [xs (train/collate-doubles (:xs batch))
                  ys (train/collate-doubles (:ys batch))
                  bs (:batch-size batch)
                  h (nn/linear xs W1 b1 bs in-dim hidden)
                  a (nn/leaky-relu h (* bs hidden) 0.01)
                  pred (nn/linear a W2 b2 bs hidden out-dim)
                  loss-val (loss/mse-loss pred ys (* bs out-dim))
                  ;; Backward
                  [d-pred _ _] ((mse-rrule loss-val pred ys (* bs out-dim)) 1.0)
                  [d-a dW2-g db2-g _ _ _] ((linear-rrule pred a W2 b2 bs hidden out-dim) d-pred)
                  [d-h _ _] ((relu-rrule a h (* bs hidden) 0.01) d-a)
                  [_ dW1-g db1-g _ _ _] ((linear-rrule h xs W1 b1 bs in-dim hidden) d-h)]
              ;; Adam updates
              (doseq [[param grad state-key n-elem]
                      [[W1 dW1-g :W1 (* hidden in-dim)]
                       [b1 db1-g :b1 hidden]
                       [W2 dW2-g :W2 (* out-dim hidden)]
                       [b2 db2-g :b2 out-dim]]]
                (let [st (get adam-state state-key)
                      step (swap! (:t st) inc)]
                  (optim/adam-step! param grad (:m st) (:v st)
                                    n-elem lr 0.9 0.999 1e-8 step)))
              loss-val))
          batch-size 16
          ;; Record loss at epoch 1
          first-loss (train/train-epoch! model-fn dataset batch-size :shuffle? true)
          ;; Train more epochs
          _ (dotimes [_ 198]
              (train/train-epoch! model-fn dataset batch-size :shuffle? true))
          last-loss (train/train-epoch! model-fn dataset batch-size :shuffle? true)
          ;; Evaluate on all 4 XOR inputs
          predict (fn [x1 x2]
                    (let [x (double-array [x1 x2])
                          h (nn/linear x W1 b1 1 in-dim hidden)
                          a (nn/leaky-relu h hidden 0.01)
                          y (nn/linear a W2 b2 1 hidden out-dim)]
                      (aget y 0)))]
      (is (> first-loss last-loss)
          "loss should decrease during training")
      (is (< last-loss 0.05)
          (str "XOR loss should be small, got " last-loss))
      ;; Check XOR outputs are in the right ballpark
      (is (< (predict 0.0 0.0) 0.3) "XOR(0,0) should be near 0")
      (is (> (predict 0.0 1.0) 0.7) "XOR(0,1) should be near 1")
      (is (> (predict 1.0 0.0) 0.7) "XOR(1,0) should be near 1")
      (is (< (predict 1.0 1.0) 0.3) "XOR(1,1) should be near 0"))))

;; ================================================================
;; Data loader test
;; ================================================================

(deftest data-loader-test
  (testing "data-loader batches dataset correctly"
    (let [dataset (vec (for [i (range 10)]
                         [(double-array [(double i)]) (double-array [(* 2.0 i)])]))
          batches (vec (train/data-loader dataset 3 :shuffle? false :drop-last? false))]
      (is (= 4 (count batches)) "10 samples / batch 3 = 4 batches (last incomplete)")
      (is (= 3 (:batch-size (first batches))))
      (is (= 1 (:batch-size (last batches))))))

  (testing "collate-doubles stacks into flat array"
    (let [items [(double-array [1.0 2.0]) (double-array [3.0 4.0]) (double-array [5.0 6.0])]
          flat (train/collate-doubles items)]
      (is (= 6 (alength flat)))
      (is (approx= 1.0 (aget flat 0)))
      (is (approx= 4.0 (aget flat 3)))
      (is (approx= 6.0 (aget flat 5))))))
