(ns raster.ode.sde-test
  (:refer-clojure :exclude [aget aset alength aclone reduce])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ode.sde :as sde]
            [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.par :as par]))

;; ================================================================
;; Euler-Maruyama step test
;; ================================================================

(deftest em-step-test
  (testing "Euler-Maruyama step with known values"
    (let [u (double-array [1.0 2.0 3.0])
          mu (double-array [0.1 0.2 0.3])
          sigma (double-array [0.5 0.5 0.5])
          dW (double-array [0.1 -0.1 0.2])
          dt 0.01
          sqrt-dt (Math/sqrt dt)
          result (sde/em-step u mu sigma dW dt sqrt-dt)]
      ;; u_new = u + mu*dt + sigma*sqrt(dt)*dW
      ;; u[0] = 1.0 + 0.1*0.01 + 0.5*sqrt(0.01)*0.1 = 1.0 + 0.001 + 0.005 = 1.006
      (is (< (Math/abs (- (aget result 0)
                          (+ 1.0 (* 0.1 0.01) (* 0.5 sqrt-dt 0.1))))
             1e-12)))))

;; ================================================================
;; GBM test
;; ================================================================

(deftest gbm-drift-test
  (testing "GBM drift: mu * S"
    (let [u (double-array [100.0 200.0 300.0])
          du (sde/gbm-drift u 0.05)]
      (is (< (Math/abs (- (aget du 0) 5.0)) 1e-12))
      (is (< (Math/abs (- (aget du 1) 10.0)) 1e-12))
      (is (< (Math/abs (- (aget du 2) 15.0)) 1e-12)))))

(deftest gbm-diffusion-test
  (testing "GBM diffusion: sigma * S"
    (let [u (double-array [100.0 200.0 300.0])
          du (sde/gbm-diffusion u 0.2)]
      (is (< (Math/abs (- (aget du 0) 20.0)) 1e-12))
      (is (< (Math/abs (- (aget du 1) 40.0)) 1e-12))
      (is (< (Math/abs (- (aget du 2) 60.0)) 1e-12)))))

;; ================================================================
;; SDE solver test
;; ================================================================

(deftest solve-sde-basic
  (testing "SDE solver runs without error"
    (let [;; Simple Brownian motion: dX = dt + dW
          drift (ftm [du :- (Array double), _u :- (Array double), _t :- Double]
                     (par/fill du 0.0))
          diffusion (ftm [du :- (Array double), _u :- (Array double), _t :- Double]
                         (par/fill du 1.0))
          prob (sde/sde-problem drift diffusion [0.0] 0.0 1.0)
          alg (sde/->EulerMaruyama)
          sol (sde/solve alg prob 0.01 (java.util.Random. 42))]
      (is (pos? (:steps sol)))
      (is (= (inc (:steps sol)) (count (:ts sol))))
      (is (= (inc (:steps sol)) (count (:us sol)))))))

(deftest solve-gbm
  (testing "GBM solver produces reasonable results"
    (let [prob (sde/geometric-brownian-motion 0.05 0.2 [100.0] 0.0 1.0)
          alg (sde/->EulerMaruyama)
          rng (java.util.Random. 42)
          sol (sde/solve alg prob 0.001 rng)
          final-price (first (last (:us sol)))]
      ;; GBM with mu=0.05, sigma=0.2 over 1 year starting at 100
      ;; Should be positive and in a reasonable range
      (is (pos? final-price) "GBM price should be positive")
      (is (< 10.0 final-price 1000.0) "GBM price should be in reasonable range"))))

(deftest solve-sde-multidim
  (testing "SDE solver works with multi-dimensional problems"
    (let [d 10
          prob (sde/geometric-brownian-motion 0.05 0.2
                                              (vec (repeat d 100.0)) 0.0 0.1)
          alg (sde/->EulerMaruyama)
          sol (sde/solve alg prob 0.01 (java.util.Random. 42))]
      (is (= d (count (last (:us sol)))))
      (is (every? pos? (last (:us sol)))
          "All GBM prices should be positive"))))
