(ns raster.sci.optim-extended-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.optim :refer [levenberg-marquardt projected-gradient
                                      differential-evolution curve-fit]]
            [raster.core :refer [ftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as n]))

(def ^:const EPS 1e-3)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Levenberg-Marquardt: fit f(x,p) = p[0]*exp(-p[1]*x)
;; ================================================================

(deftest levenberg-marquardt-exp-fit-test
  (testing "LM fits exponential decay"
    ;; True params: p0=2.0, p1=0.5
    ;; Generate data: f(x) = 2*exp(-0.5*x) for x = 0,1,...,9
    (let [n-data 10
          n-params 2
          residual-fn (ftm [params :- (Array double), residuals :- (Array double)] :- (Array double)
                           (dotimes [i n-data]
                             (let [xi (double i)
                                   model-val (n/* (arrays/aget params 0)
                                                  (Math/exp (n/- (n/* (arrays/aget params 1) xi))))
                                   yi (n/* 2.0 (Math/exp (n/- (n/* 0.5 xi))))]
                               (arrays/aset residuals i (n/- model-val yi))))
                           residuals)
          p0 (double-array [1.0 1.0])
          result (levenberg-marquardt residual-fn p0 n-data n-params 1e-10 200)]
      (is (approx= 2.0 ((:minimizer result) 0) 0.1)
          (str "p0 should be ~2.0, got " ((:minimizer result) 0)))
      (is (approx= 0.5 ((:minimizer result) 1) 0.1)
          (str "p1 should be ~0.5, got " ((:minimizer result) 1))))))

;; ================================================================
;; Projected gradient: minimize x^2 with bounds [1, 10]
;; ================================================================

(deftest projected-gradient-bounded-test
  (testing "projected gradient: min x^2 s.t. x in [1, 10] => x=1"
    (let [f (ftm [x :- (Array double)] :- Double
                 (n/* (arrays/aget x 0) (arrays/aget x 0)))
          grad-fn (ftm [x :- (Array double), g :- (Array double)]
                       (arrays/aset g 0 (n/* 2.0 (arrays/aget x 0))))
          x0 (double-array [5.0])
          lower (double-array [1.0])
          upper (double-array [10.0])
          result (projected-gradient f grad-fn x0 lower upper 1 1e-8 1000)]
      (is (approx= 1.0 ((:minimizer result) 0) 0.01)
          (str "Should converge to lower bound 1.0, got " ((:minimizer result) 0))))))

;; ================================================================
;; Differential evolution: minimize Rosenbrock
;; ================================================================

(deftest differential-evolution-rosenbrock-test
  (testing "DE minimizes Rosenbrock in [-5,5]^2"
    (let [rosenbrock (ftm [x :- (Array double)] :- Double
                          (let [a (n/- (arrays/aget x 0) 1.0)
                                b (n/- (arrays/aget x 1) (n/* (arrays/aget x 0) (arrays/aget x 0)))]
                            (n/+ (n/* a a) (n/* 100.0 (n/* b b)))))
          lower (double-array [-5.0 -5.0])
          upper (double-array [5.0 5.0])
          result (differential-evolution rosenbrock lower upper 2 50 500 1e-6)]
      (is (< (:minimum result) 0.1)
          (str "Minimum should be near 0, got " (:minimum result)))
      (is (approx= 1.0 ((:minimizer result) 0) 0.5)
          (str "x should be near 1.0, got " ((:minimizer result) 0))))))

;; ================================================================
;; Curve fit convenience
;; ================================================================

(deftest curve-fit-linear-test
  (testing "curve-fit fits a linear model y = p0 + p1*x"
    (let [model (ftm [params :- (Array double), x :- Double] :- Double
                     (n/+ (arrays/aget params 0) (n/* (arrays/aget params 1) x)))
          xdata (double-array [0.0 1.0 2.0 3.0 4.0])
          ;; y = 1 + 2*x
          ydata (double-array [1.0 3.0 5.0 7.0 9.0])
          p0 (double-array [0.0 0.0])
          result (curve-fit model xdata ydata p0 5 2)]
      (is (approx= 1.0 ((:minimizer result) 0) 0.1)
          (str "intercept should be ~1.0, got " ((:minimizer result) 0)))
      (is (approx= 2.0 ((:minimizer result) 1) 0.1)
          (str "slope should be ~2.0, got " ((:minimizer result) 1))))))
