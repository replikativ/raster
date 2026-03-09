(ns raster.dl.optim-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.dl.optim :as optim]))

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

;; ================================================================
;; SGD
;; ================================================================

(deftest sgd-step-test
  (testing "SGD updates parameters"
    (let [param (double-array [1.0 2.0 3.0])
          grad (double-array [0.1 0.2 0.3])
          _ (optim/sgd-step! param grad 3 0.1)]
      ;; param -= lr * grad
      (is (approx= 0.99 (aget param 0)))
      (is (approx= 1.98 (aget param 1)))
      (is (approx= 2.97 (aget param 2))))))

;; ================================================================
;; Adam
;; ================================================================

(deftest adam-step-test
  (testing "Adam converges on simple quadratic"
    ;; Minimize f(x) = x^2, gradient = 2x
    (let [param (double-array [5.0])
          m (double-array [0.0])
          v (double-array [0.0])]
      (dotimes [t 1000]
        (let [grad (double-array [(* 2.0 (aget param 0))])]
          (optim/adam-step! param grad m v 1 0.01 0.9 0.999 1e-8 (inc t))))
      (is (approx= 0.0 (aget param 0) 0.2)))))

(deftest adamw-step-test
  (testing "AdamW with weight decay"
    (let [param (double-array [5.0])
          m (double-array [0.0])
          v (double-array [0.0])]
      (dotimes [t 1000]
        (let [grad (double-array [(* 2.0 (aget param 0))])]
          (optim/adamw-step! param grad m v 1 0.01 0.9 0.999 1e-8 0.01 (inc t))))
      ;; Should converge near 0 (Adam is slow on simple quadratic)
      (is (approx= 0.0 (aget param 0) 0.2)))))

;; ================================================================
;; Gradient clipping
;; ================================================================

(deftest clip-grad-norm-test
  (testing "clips when norm exceeds max"
    (let [grads (double-array [3.0 4.0])  ;; norm = 5
          _ (optim/clip-grad-norm! grads 2 2.0)]
      ;; norm should be 2.0 after clipping
      (let [norm (Math/sqrt (+ (* (aget grads 0) (aget grads 0))
                               (* (aget grads 1) (aget grads 1))))]
        (is (approx= 2.0 norm 1e-6)))))

  (testing "no-op when norm below max"
    (let [grads (double-array [0.1 0.2])
          orig (aclone grads)
          _ (optim/clip-grad-norm! grads 2 10.0)]
      (is (= (aget orig 0) (aget grads 0)))
      (is (= (aget orig 1) (aget grads 1))))))

;; ================================================================
;; EMA
;; ================================================================

(deftest ema-update-test
  (testing "EMA moves towards param"
    (let [shadow (double-array [1.0])
          param (double-array [2.0])
          _ (optim/ema-update! shadow param 1 0.9)]
      ;; shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
      (is (approx= 1.1 (aget shadow 0))))))

;; ================================================================
;; LR Schedulers
;; ================================================================

(deftest cosine-lr-test
  (testing "cosine starts at base-lr"
    (is (approx= 0.1 (optim/cosine-lr 0.1 0 100))))
  (testing "cosine ends near 0"
    (is (approx= 0.0 (optim/cosine-lr 0.1 100 100) 1e-6)))
  (testing "cosine at midpoint = base/2"
    (is (approx= 0.05 (optim/cosine-lr 0.1 50 100) 1e-6))))

(deftest warmup-cosine-lr-test
  (testing "warmup phase is linear"
    (is (approx= 0.0 (optim/warmup-cosine-lr 0.1 0 100 10)))
    (is (approx= 0.05 (optim/warmup-cosine-lr 0.1 5 100 10)))))

(deftest step-lr-test
  (testing "step decay"
    (is (approx= 0.1 (optim/step-lr 0.1 0 10 0.1)))
    (is (approx= 0.01 (optim/step-lr 0.1 10 10 0.1)))))

;; ================================================================
;; Adam state management
;; ================================================================

(deftest make-adam-state-test
  (testing "creates momentum and variance buffers"
    (let [params {"W" (double-array [1 2 3])}
          state (optim/make-adam-state params)]
      (is (= 3 (alength ^doubles (:m (get state "W")))))
      (is (= 3 (alength ^doubles (:v (get state "W")))))
      (is (= 0 @(:t (get state "W")))))))
