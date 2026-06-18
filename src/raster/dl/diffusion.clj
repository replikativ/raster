(ns raster.dl.diffusion
  "Diffusion model utilities for the Raster deep learning framework.

  Provides:
    linear-beta-schedule    - linear noise schedule
    cosine-beta-schedule    - cosine noise schedule
    compute-alphas-cumprod  - cumulative alpha products
    forward-noise           - q(x_t | x_0) forward noise process
    predict-x0              - predict x_0 from noise prediction
    ddpm-sample-step        - single DDPM reverse sampling step"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm broadcast]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; Noise schedules (pure functions)
;; ================================================================

(deftm linear-beta-schedule (All [T] [num-steps :- Long beta-start :- Double beta-end :- Double]
                                 :- (Array double)
                                 (par/map [i num-steps]
                                          (n/+ beta-start (n/* (n// (double i) (double (- num-steps 1))) (n/- beta-end beta-start))))))

(deftm cosine-beta-schedule (All [T] [num-steps :- Long s :- Double] :- (Array double)
                                 (par/map [i num-steps]
                                          (let [t1 (n// (double i) (double num-steps))
                                                t2 (n// (double (+ i 1)) (double num-steps))
                                                a1 (m/cos (n/* (n// (n/+ t1 s) (n/+ 1.0 s)) (n/* 0.5 n/pi)))
                                                a2 (m/cos (n/* (n// (n/+ t2 s) (n/+ 1.0 s)) (n/* 0.5 n/pi)))
                                                b (n/min 0.999 (n/- 1.0 (n// (n/* a2 a2) (n/* a1 a1))))]
                                            (n/max 0.0001 b)))))

(deftm cosine-beta-schedule (All [T] [num-steps :- Long] :- (Array double)
                                 (cosine-beta-schedule num-steps 0.008)))

(deftm compute-alphas-cumprod (All [T] [betas :- (Array T)] :- (Array T)
                                   (let [n (alength betas)
                                         alphas-cumprod (n/similar betas)]
                                     (loop [i 0 prod 1.0]
                                       (when (< i n)
                                         (let [a (n/* prod (n/- 1.0 (aget betas i)))]
                                           (aset alphas-cumprod i a)
                                           (recur (inc i) a))))
                                     alphas-cumprod)))

;; ================================================================
;; Forward noise process: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
;; ================================================================

(deftm forward-noise (All [T] [x0 :- (Array T) noise :- (Array T)
                               alpha-t :- Double n :- Long] :- (Array T)
                          (let [sqrt-alpha (n/sqrt alpha-t)
                                sqrt-1ma (n/sqrt (n/- 1.0 alpha-t))]
                            (broadcast [x0 noise]
                                       (n/+ (n/* sqrt-alpha x0) (n/* sqrt-1ma noise))))))

;; forward-noise rrule
(tmpl/merge-into-template! 'raster.dl.diffusion/forward-noise
                           {:pullback-factory (fn [_result x0 noise alpha-t n]
                                                (fn [dy]
                                                  (let [n (long n)
                                                        alpha-t (double alpha-t)
                                                        sqrt-alpha (n/sqrt alpha-t)
                                                        sqrt-1ma (n/sqrt (- 1.0 alpha-t))
                                                        dx0 (double-array n)
                                                        d-noise (double-array n)]
                                                    (dotimes [i n]
                                                      (let [dyi (aget dy i)]
                                                        (aset dx0 i (* dyi sqrt-alpha))
                                                        (aset d-noise i (* dyi sqrt-1ma))))
                                                    [dx0 d-noise nil nil])))})

;; ================================================================
;; Predict x0 from noise prediction
;; ================================================================

(deftm predict-x0 (All [T] [xt :- (Array T) eps-pred :- (Array T)
                            alpha-t :- Double n :- Long] :- (Array T)
                       (let [sqrt-alpha (n/sqrt alpha-t)
                             sqrt-1ma (n/sqrt (n/- 1.0 alpha-t))
                             inv-sqrt-alpha (n// 1.0 sqrt-alpha)]
                         (broadcast [xt eps-pred]
                                    (n/* inv-sqrt-alpha (n/- xt (n/* sqrt-1ma eps-pred)))))))

;; ================================================================
;; DDPM reverse sampling step
;; ================================================================

(deftm ddpm-sample-step (All [T] [xt :- (Array T) eps-pred :- (Array T)
                                  noise :- (Array T)
                                  beta-t :- Double alpha-t :- Double
                                  alpha-bar-t :- Double n :- Long]
                             :- (Array T)
                             (let [;; mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
                                   inv-sqrt-alpha (n// 1.0 (n/sqrt alpha-t))
                                   coeff (n// beta-t (n/sqrt (n/- 1.0 alpha-bar-t)))
                                   sigma (n/sqrt beta-t)]
                               (broadcast [xt eps-pred noise]
                                          (n/+ (n/* inv-sqrt-alpha (n/- xt (n/* coeff eps-pred)))
                                               (n/* sigma noise))))))

;; ================================================================
;; Backward helper for forward-noise
;; ================================================================

;; dx0[i] = dy[i] * sqrt(alpha_t), d_noise[i] = dy[i] * sqrt(1 - alpha_t)
(deftm forward-noise-backward (All [T] [dy :- (Array T)
                                        alpha-t :- Double n :- Long]
                                   :- (Array Object)
                                   (let [sqrt-alpha (n/sqrt alpha-t)
                                         sqrt-1ma (n/sqrt (n/- 1.0 alpha-t))
                                         dx0 (n/similar dy)
                                         d-noise (n/similar dy)]
                                     (dotimes [i n]
                                       (let [dyi (aget dy i)]
                                         (aset dx0 i (n/* dyi sqrt-alpha))
                                         (aset d-noise i (n/* dyi sqrt-1ma))))
                                     (object-array [dx0 d-noise]))))

;; ================================================================
;; Template registration for forward-noise
;; ================================================================

(tmpl/merge-into-template! 'raster.dl.diffusion/forward-noise
                           {:params '[x0 noise alpha-t n] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [x0 noise alpha-t n] _result-sym adjoint-sym gensym-fn]
                                        (let [grads-arr (gensym-fn "fn_grads")
                                              dx0 (gensym-fn "dx0")
                                              d-noise (gensym-fn "d_noise")]
                                          [(update ctx :bindings into
                                                   [grads-arr (list 'raster.dl.diffusion/forward-noise-backward
                                                                    adjoint-sym alpha-t n)
                                                    dx0 (list 'clojure.core/aget grads-arr 0)
                                                    d-noise (list 'clojure.core/aget grads-arr 1)])
                                           [dx0 d-noise nil nil]]))})
