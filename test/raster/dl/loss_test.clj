(ns raster.dl.loss-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.dl.loss :as loss]
            [raster.ad.templates :as tmpl]))

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
;; MSE Loss
;; ================================================================

(deftest mse-loss-test
  (testing "mse-loss zero when equal"
    (let [pred (double-array [1 2 3])
          target (double-array [1 2 3])]
      (is (approx= 0.0 (loss/mse-loss pred target 3)))))

  (testing "mse-loss positive"
    (let [pred (double-array [1 2 3])
          target (double-array [2 3 4])]
      ;; mean((1-2)^2 + (2-3)^2 + (3-4)^2) = 3/3 = 1.0
      (is (approx= 1.0 (loss/mse-loss pred target 3)))))

  (testing "mse-loss gradient"
    (let [pred (double-array [0.5 1.5 2.5])
          target (double-array [1.0 2.0 3.0])
          rrfn (tmpl/get-pullback-factory 'raster.dl.loss/mse-loss)
          l (loss/mse-loss pred target 3)
          pb (rrfn l pred target 3)
          [d-pred _ _] (pb 1.0)
          num-d (numerical-grad-array
                 #(loss/mse-loss pred target 3) pred 1e-5)]
      (is (arr-approx= d-pred num-d 1e-4)))))

;; ================================================================
;; Cross-entropy Loss
;; ================================================================

(deftest cross-entropy-loss-test
  (testing "cross-entropy-loss with clear prediction"
    (let [logits (double-array [10 0 0  0 10 0])  ;; [2, 3]
          target (long-array [0 1])]
      (let [l (loss/cross-entropy-loss logits target 2 3)]
        ;; Very confident predictions → near 0 loss
        (is (< l 0.001)))))

  (testing "cross-entropy-loss gradient"
    (let [logits (double-array [1.0 2.0 3.0])  ;; [1, 3]
          target (long-array [2])
          rrfn (tmpl/get-pullback-factory 'raster.dl.loss/cross-entropy-loss)
          l (loss/cross-entropy-loss logits target 1 3)
          pb (rrfn l logits target 1 3)
          [d-logits _ _ _] (pb 1.0)
          num-d (numerical-grad-array
                 #(loss/cross-entropy-loss logits target 1 3) logits 1e-5)]
      (is (arr-approx= d-logits num-d 1e-4)))))

;; ================================================================
;; Huber Loss
;; ================================================================

(deftest huber-loss-test
  (testing "huber-loss small residuals = MSE"
    (let [pred (double-array [1.0 2.0])
          target (double-array [1.1 2.1])
          h (loss/huber-loss pred target 2 1.0)
          m (loss/mse-loss pred target 2)]
      ;; For small residuals, huber ≈ 0.5 * mse (because huber = 0.5*d^2)
      (is (approx= (* 0.5 m) h 1e-6))))

  (testing "huber-loss gradient"
    (let [pred (double-array [0.5 3.0])
          target (double-array [1.0 0.0])
          rrfn (tmpl/get-pullback-factory 'raster.dl.loss/huber-loss)
          l (loss/huber-loss pred target 2 1.0)
          pb (rrfn l pred target 2 1.0)
          [d-pred _ _ _] (pb 1.0)
          num-d (numerical-grad-array
                 #(loss/huber-loss pred target 2 1.0) pred 1e-5)]
      (is (arr-approx= d-pred num-d 1e-4)))))

;; ================================================================
;; L1 Loss
;; ================================================================

(deftest l1-loss-test
  (testing "l1-loss"
    (let [pred (double-array [1 2 3])
          target (double-array [2 3 4])]
      ;; mean(|1-2| + |2-3| + |3-4|) = 3/3 = 1.0
      (is (approx= 1.0 (loss/l1-loss pred target 3))))))
