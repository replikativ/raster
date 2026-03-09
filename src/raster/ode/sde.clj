(ns raster.ode.sde
  "Stochastic Differential Equation (SDE) solvers using parallel primitives.

  Implements the Euler-Maruyama method for SDEs of the form:
    dX_t = mu(X_t, t) dt + sigma(X_t, t) dW_t

  The SDE update step is naturally element-wise (parallel broadcast),
  enabling automatic SIMD/CUDA optimization through raster.par.

  Usage:
    (def prob (sde-problem drift diffusion (double-array [1.0]) 0.0 1.0))
    (def sol (solve (->EulerMaruyama) prob 0.01 (java.util.Random. 42)))"
  (:refer-clojure :exclude [aget aset alength aclone reduce])
  (:require [raster.core :refer [deftm ftm defabstract defval defvalue broadcast]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.numeric :as n]
            [raster.par]))

;; ================================================================
;; SDE Problem
;; ================================================================

(defrecord SDEProblem [drift diffusion u0 tspan opts])

(defn sde-problem
  "Create an SDE problem: dX = drift(du, u, t) dt + diffusion(du, u, t) dW.
  drift/diffusion: (fn [du u t]) — writes into du in-place.
  u0: initial state (double-array).
  t0, tf: time span."
  ([drift diffusion u0 t0 tf]
   (sde-problem drift diffusion u0 t0 tf {}))
  ([drift diffusion u0 t0 tf opts]
   (->SDEProblem drift diffusion (double-array u0) [t0 tf] opts)))

;; ================================================================
;; Algorithm types
;; ================================================================

(defabstract SDEAlgorithm)
(defabstract SDECache)

(defval EulerMaruyama :implements SDEAlgorithm)

;; ================================================================
;; Cache
;; ================================================================

(defvalue EMCache (All [T]) [mu :- (Array T), sigma :- (Array T),
                             dW :- (Array T)]
  :implements SDECache)

(deftm make-em-cache (All [T]
                          [alg :- EulerMaruyama, ref :- (Array T)] :- EMCache
                          (->EMCache (n/similar ref) (n/similar ref) (n/similar ref))))

;; ================================================================
;; Euler-Maruyama step
;; ================================================================

(deftm em-step (All [T]
                    [u :- (Array T), mu :- (Array T),
                     sigma :- (Array T), dW :- (Array T),
                     dt :- Double, sqrt-dt :- Double] :- (Array T)
                    (broadcast [u mu sigma dW]
                               (+ u (+ (* mu dt) (* sigma (* sqrt-dt dW)))))))

(deftm fill-gaussian! [arr :- (Array double), rng :- Object] :- (Array double)
  (let [n (alength arr)]
    (dotimes [i n]
      (aset arr i (.nextGaussian ^java.util.Random rng)))
    arr))

;; ================================================================
;; SDE solver
;; ================================================================

(deftm solve [alg :- EulerMaruyama, prob :- SDEProblem,
              dt :- Double, rng :- Object]
  (let [drift (:drift prob)
        diffusion (:diffusion prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt (double dt)
        n (alength u0)
        cache (make-em-cache alg u0)
        u (aclone u0)
        sqrt-dt (n/sqrt dt)
        save? (get (:opts prob) :save-everystep true)]
    (acopy! u0 0 u 0 n)
    (loop [t t0
           ts (transient [t0])
           us (transient [(aclone u0)])
           steps 0]
      (if (>= t tf)
        {:ts (persistent! ts) :us (persistent! us) :steps steps}
        (let [mu-arr (.mu cache)
              sigma-arr (.sigma cache)
              dW-arr (.dW cache)]
          (drift mu-arr u t)
          (diffusion sigma-arr u t)
          (fill-gaussian! dW-arr rng)
          (acopy! (em-step u mu-arr sigma-arr dW-arr dt sqrt-dt) 0 u 0 (alength u))
          (let [t-new (+ t dt)]
            (recur t-new
                   (if save? (conj! ts t-new) ts)
                   (if save? (conj! us (aclone u)) us)
                   (inc steps))))))))

;; ================================================================
;; Convenience: geometric Brownian motion
;; ================================================================

(deftm gbm-drift (All [T]
                      [u :- (Array T),
                       mu-val :- Double] :- (Array T)
                      (broadcast [u] (* mu-val u))))

(deftm gbm-diffusion (All [T]
                          [u :- (Array T),
                           sigma-val :- Double] :- (Array T)
                          (broadcast [u] (* sigma-val u))))

(defn geometric-brownian-motion
  "Create an SDE problem for geometric Brownian motion:
    dS = mu*S*dt + sigma*S*dW

  mu: drift rate
  sigma: volatility
  S0: initial price(s) (double-array or seq)"
  [mu sigma S0 t0 tf]
  (sde-problem
   (ftm [du :- (Array double), u :- (Array double), _t :- Double]
        (acopy! (gbm-drift u mu) 0 du 0 (alength du))
        du)
   (ftm [du :- (Array double), u :- (Array double), _t :- Double]
        (acopy! (gbm-diffusion u sigma) 0 du 0 (alength du))
        du)
   (if (sequential? S0) (double-array S0) S0)
   t0 tf))
