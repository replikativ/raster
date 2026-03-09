(ns raster.ode.sensitivity-test
  "ODE sensitivity tests: forward-mode AD through generic ODE solvers.
  Verifies that Dual numbers propagate correctly through solve(RK4, ...)
  using the standard GenericODEProblem API."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.numeric :refer [+ - * / real-value]]
            [raster.ode.core :as ode]
            [raster.ad.forward :as ad]))

(defn- approx=
  [^double a ^double b ^double tol]
  (< (Math/abs (clojure.core/- a b)) tol))

;; ================================================================
;; 1. Scalar exponential decay: du/dt = -u
;; ================================================================

(deftest scalar-dual-exponential-decay
  (testing "RK4 with scalar Dual: value and derivative match e^-1"
    (let [f (fn [u t] (- u))
          u0 (ad/dual 1.0 1.0)
          prob (ode/generic-ode-problem f u0 0.0 1.0)
          sol (ode/solve (ode/->RK4) prob 0.001)
          uf (last (:us sol))
          exact (Math/exp -1.0)]
      (is (approx= exact (:v uf) 1e-8)
          (str "u(1) = " (:v uf) ", expected " exact))
      (is (approx= exact (aget ^doubles (:partials uf) 0) 1e-8)
          (str "du/du0 = " (aget ^doubles (:partials uf) 0) ", expected " exact)))))

;; ================================================================
;; 2. Vector Lotka-Volterra: du/du0 sensitivity
;; ================================================================

(deftest vector-dual-lotka-volterra
  (testing "RK4 with vector-of-Duals through Lotka-Volterra"
    (let [f (fn [state t]
              (let [u (nth state 0) v (nth state 1)]
                [(- (* 1.5 u) (* u v))
                 (- (* u v) (* 3.0 v))]))
          ;; Seed derivative on u0 (prey initial population)
          u0 [(ad/dual 1.0 1.0) (ad/dual 1.0 0.0)]
          prob (ode/generic-ode-problem f u0 0.0 1.0)
          sol (ode/solve (ode/->RK4) prob 0.001)
          final (last (:us sol))
          u-val (:v (nth final 0))
          v-val (:v (nth final 1))
          du-du0 (aget ^doubles (:partials (nth final 0)) 0)
          dv-du0 (aget ^doubles (:partials (nth final 1)) 0)]
      ;; Values should be positive and bounded
      (is (> u-val 0.0) "Prey positive")
      (is (> v-val 0.0) "Predator positive")
      (is (< u-val 100.0) "Prey bounded")
      ;; Derivatives should be non-zero (u0 affects both species)
      (is (not (zero? du-du0)) "du/du0 should be non-zero")
      (is (not (zero? dv-du0)) "dv/du0 should be non-zero"))))

;; ================================================================
;; 3. Parameter sensitivity: du/dt = -p*u, d/dp u(T) = -T*e^(-pT)
;; ================================================================

(deftest parameter-sensitivity-exponential
  (testing "AD gradient of u(T) w.r.t. decay rate p"
    (let [T 1.0
          p-val 1.0
          ;; State = [u, p], dp/dt = 0 (parameter is constant)
          f (fn [state t]
              (let [u (nth state 0) p (nth state 1)]
                [(- (* p u))   ;; du/dt = -p*u
                 (* p 0.0)]))  ;; dp/dt = 0
          ;; Seed derivative on p (index 1)
          u0 [(ad/dual 1.0 0.0) (ad/dual p-val 1.0)]
          prob (ode/generic-ode-problem f u0 0.0 T)
          sol (ode/solve (ode/->RK4) prob 0.001)
          final (last (:us sol))
          u-final (nth final 0)
          ad-deriv (aget ^doubles (:partials u-final) 0)
          ;; Exact: d/dp exp(-pT) = -T*exp(-pT)
          exact (clojure.core/* (clojure.core/- T) (Math/exp (clojure.core/- (clojure.core/* p-val T))))]
      (is (approx= exact ad-deriv 1e-6)
          (str "AD=" ad-deriv " exact=" exact)))))

;; ================================================================
;; 4. Dual arithmetic sanity
;; ================================================================

(deftest dual-arithmetic-test
  (testing "Dual + produces correct result"
    (let [a (ad/->Dual 3.0 (double-array [1.0 0.0]))
          b (ad/->Dual 5.0 (double-array [0.0 1.0]))
          r ^raster.ad.forward.Dual (+ a b)]
      (is (== 8.0 (.v r)))
      (is (== 1.0 (aget (.partials r) 0)))
      (is (== 1.0 (aget (.partials r) 1)))))

  (testing "Dual * produces correct derivatives"
    (let [a (ad/->Dual 3.0 (double-array [1.0 0.0]))
          b (ad/->Dual 5.0 (double-array [0.0 1.0]))
          r ^raster.ad.forward.Dual (* a b)]
      (is (== 15.0 (.v r)))
      (is (== 5.0 (aget (.partials r) 0)))
      (is (== 3.0 (aget (.partials r) 1)))))

  (testing "Dual mixed Number ops"
    (let [a (ad/->Dual 3.0 (double-array [1.0]))
          r ^raster.ad.forward.Dual (+ a 10.0)]
      (is (== 13.0 (.v r)))
      (is (== 1.0 (aget (.partials r) 0))))))
