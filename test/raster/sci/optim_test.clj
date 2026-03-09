(ns raster.sci.optim-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as n]
            [raster.sci.optim :refer [optimize]]))

;; ================================================================
;; Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
;; Minimum at (1, 1) with value 0
;; ================================================================

(deftm rosenbrock [x :- (Array double)] :- Double
  (let [a (n/- (arrays/aget x 0) 1.0)
        b (n/- (arrays/aget x 1) (n/* (arrays/aget x 0) (arrays/aget x 0)))]
    (n/+ (n/* a a) (n/* 100.0 (n/* b b)))))

;; ================================================================
;; Quadratic: f(x) = x^2 + y^2, minimum at (0,0)
;; ================================================================

(deftm quadratic [x :- (Array double)] :- Double
  (let [nn (arrays/alength x)]
    (loop [i 0 acc 0.0]
      (if (>= i nn) acc
          (recur (inc i) (n/+ acc (n/* (arrays/aget x i) (arrays/aget x i))))))))

;; ================================================================
;; Nelder-Mead tests
;; ================================================================

(deftest nelder-mead-quadratic-test
  (testing "Nelder-Mead minimizes quadratic"
    (let [result (optimize quadratic [5.0 -3.0]
                           {:algorithm :nelder-mead :tol 1e-10})]
      (is (:converged? result))
      (is (< (Math/abs ((:minimizer result) 0)) 1e-4))
      (is (< (Math/abs ((:minimizer result) 1)) 1e-4))
      (is (< (:minimum result) 1e-8)))))

(deftest nelder-mead-rosenbrock-test
  (testing "Nelder-Mead minimizes Rosenbrock (may need many iters)"
    (let [result (optimize rosenbrock [0.0 0.0]
                           {:algorithm :nelder-mead :maxiter 5000 :tol 1e-10})]
      (is (< (:minimum result) 1e-4))
      (is (< (Math/abs (- ((:minimizer result) 0) 1.0)) 0.05))
      (is (< (Math/abs (- ((:minimizer result) 1) 1.0)) 0.05)))))

;; ================================================================
;; L-BFGS tests
;; ================================================================

(deftest lbfgs-quadratic-test
  (testing "L-BFGS minimizes quadratic"
    (let [result (optimize quadratic [5.0 -3.0]
                           {:algorithm :lbfgs :tol 1e-10})]
      (is (:converged? result))
      (is (< (Math/abs ((:minimizer result) 0)) 1e-6))
      (is (< (Math/abs ((:minimizer result) 1)) 1e-6)))))

(deftest lbfgs-rosenbrock-test
  (testing "L-BFGS minimizes Rosenbrock"
    (let [result (optimize rosenbrock [-1.0 1.0]
                           {:algorithm :lbfgs :maxiter 500 :tol 1e-8})]
      (is (< (:minimum result) 1e-4))
      (is (< (Math/abs (- ((:minimizer result) 0) 1.0)) 0.05)))))

;; ================================================================
;; Gradient Descent tests
;; ================================================================

(deftest gradient-descent-quadratic-test
  (testing "Gradient descent minimizes quadratic"
    (let [result (optimize quadratic [5.0 -3.0]
                           {:algorithm :gradient-descent :tol 1e-8})]
      (is (:converged? result))
      (is (< (Math/abs ((:minimizer result) 0)) 1e-4))
      (is (< (Math/abs ((:minimizer result) 1)) 1e-4)))))

;; ================================================================
;; Newton tests
;; ================================================================

(deftest newton-quadratic-test
  (testing "Newton minimizes quadratic"
    (let [result (optimize quadratic [5.0 -3.0]
                           {:algorithm :newton :tol 1e-10})]
      (is (:converged? result))
      (is (< (Math/abs ((:minimizer result) 0)) 1e-5))
      (is (< (Math/abs ((:minimizer result) 1)) 1e-5)))))
