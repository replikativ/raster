(ns raster.dl.diffusion-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.dl.diffusion :as diff]
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

;; ================================================================
;; Noise schedules
;; ================================================================

(deftest linear-beta-schedule-test
  (testing "linear schedule has correct endpoints"
    (let [betas (diff/linear-beta-schedule 100 0.0001 0.02)]
      (is (= 100 (alength betas)))
      (is (approx= 0.0001 (aget betas 0)))
      (is (approx= 0.02 (aget betas 99)))))

  (testing "linear schedule is monotonically increasing"
    (let [betas (diff/linear-beta-schedule 100 0.0001 0.02)]
      (is (every? true?
                  (for [i (range 99)]
                    (<= (aget betas i) (aget betas (inc i)))))))))

(deftest cosine-beta-schedule-test
  (testing "cosine schedule has correct length"
    (let [betas (diff/cosine-beta-schedule 100)]
      (is (= 100 (alength betas)))))

  (testing "cosine schedule values are valid"
    (let [betas (diff/cosine-beta-schedule 100)]
      (is (every? true?
                  (for [i (range 100)]
                    (and (> (aget betas i) 0)
                         (< (aget betas i) 1))))))))

(deftest compute-alphas-cumprod-test
  (testing "alphas cumprod is decreasing"
    (let [betas (diff/linear-beta-schedule 100 0.0001 0.02)
          alphas (diff/compute-alphas-cumprod betas)]
      (is (= 100 (alength alphas)))
      ;; First alpha should be close to 1
      (is (> (aget alphas 0) 0.99))
      ;; Last alpha should be small
      (is (< (aget alphas 99) 0.5))
      ;; Should be monotonically decreasing
      (is (every? true?
                  (for [i (range 99)]
                    (>= (aget alphas i) (aget alphas (inc i)))))))))

;; ================================================================
;; Forward noise process
;; ================================================================

(deftest forward-noise-test
  (testing "forward noise with alpha=1 returns x0"
    (let [x0 (double-array [1 2 3])
          noise (double-array [10 20 30])
          xt (diff/forward-noise x0 noise 1.0 3)]
      (is (arr-approx= x0 xt))))

  (testing "forward noise with alpha=0 returns noise"
    (let [x0 (double-array [1 2 3])
          noise (double-array [10 20 30])
          xt (diff/forward-noise x0 noise 0.0 3)]
      (is (arr-approx= noise xt))))

  (testing "forward noise gradient"
    (let [x0 (double-array [1.0 2.0 3.0])
          noise (double-array [0.1 0.2 0.3])
          rrfn (tmpl/template-pullback 'raster.dl.diffusion/forward-noise)
          xt (diff/forward-noise x0 noise 0.5 3)
          pb (rrfn xt x0 noise 0.5 3)
          dy (double-array [1 1 1])
          [dx0 d-noise _ _] (pb dy)]
      ;; dx0 = dy * sqrt(alpha)
      (let [sa (Math/sqrt 0.5)]
        (is (approx= sa (aget dx0 0)))
        (is (approx= sa (aget dx0 1)))))))

;; ================================================================
;; Predict x0
;; ================================================================

(deftest predict-x0-test
  (testing "predict-x0 recovers x0 from forward process"
    (let [x0 (double-array [1 2 3])
          noise (double-array [0.5 0.5 0.5])
          alpha 0.8
          xt (diff/forward-noise x0 noise alpha 3)
          x0-pred (diff/predict-x0 xt noise alpha 3)]
      (is (arr-approx= x0 x0-pred 1e-10)))))

;; ================================================================
;; DDPM sample step
;; ================================================================

(deftest ddpm-sample-step-test
  (testing "ddpm-sample-step runs without error"
    (let [xt (double-array [1 2 3])
          eps-pred (double-array [0.1 0.2 0.3])
          noise (double-array [0.01 0.02 0.03])
          beta-t 0.01
          alpha-t 0.99
          alpha-bar-t 0.8
          result (diff/ddpm-sample-step xt eps-pred noise
                                        beta-t alpha-t alpha-bar-t 3)]
      (is (= 3 (alength result))))))
