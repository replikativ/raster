(ns raster.ode.core
  "ODE integrators with Julia-style typed multiple dispatch.
   Demonstrates raster's type-driven specialization on a real
   numerical computing problem.

   Implements Euler, RK4, and Dormand-Prince (DP5) methods with
   in-place array operations for performance.

   Usage:
     (def prob (ode-problem exponential-decay
                            (double-array [1.0])
                            0.0 10.0))
     (def sol (solve (->Euler) prob 0.01))
     (:us sol)  ;; solution states"
  (:refer-clojure :exclude [aget aset alength aclone reduce])
  (:require [raster.core :as mc :refer [deftm ftm defvalue defval defabstract                                         broadcast reduce! muladd]]
            [raster.arrays :refer [aget aset alength aclone acopy! alloc-like]]
            [raster.par]
            [raster.math :as m]
            [raster.numeric :as n]
            [clojure.core.typed :as t]))

;; ================================================================
;; Abstract type hierarchy (like Julia)
;; ================================================================

(defabstract ODEAlgorithm)
(defabstract FixedStepAlgorithm :extends ODEAlgorithm)
(defabstract AdaptiveAlgorithm :extends ODEAlgorithm)
(defabstract ImplicitAlgorithm :extends ODEAlgorithm)

(defabstract ODECache)
(defabstract FixedStepCache :extends ODECache)
(defabstract AdaptiveCache :extends ODECache)
(defabstract ImplicitCache :extends ODECache)

;; ================================================================
;; ODE Problem
;; ================================================================

(defvalue ODEProblem
  [f :- Object,
   u0 :- Object,
   tspan :- Object,
   opts :- Object])

(defn ode-problem
  "Create an ODE problem: du/dt = f(u, p, t).
   f: (fn [du u t]) — writes into du (in-place).
   u0: initial state (any type supporting aget/aset/alength/aclone).
   t0, tf: time span.
   opts: optional map, e.g. {:save-everystep false}."
  ([f u0 t0 tf] (ode-problem f u0 t0 tf {}))
  ([f u0 t0 tf opts]
   (->ODEProblem f (if (sequential? u0) (double-array u0) (aclone u0)) [t0 tf] opts)))

;; ================================================================
;; Algorithm types
;; ================================================================

(defval Euler :implements FixedStepAlgorithm)
(defval RK4 :implements FixedStepAlgorithm)
(defvalue DP5 [atol :- Double, rtol :- Double] :implements AdaptiveAlgorithm)
(defvalue Tsit5 [atol :- Double, rtol :- Double] :implements AdaptiveAlgorithm)

(t/ann-record DP5 [atol :- Double rtol :- Double])
(t/ann-record Tsit5 [atol :- Double rtol :- Double])

(defn dp5
  "Create a Dormand-Prince 5(4) adaptive solver."
  ([] (->DP5 1e-6 1e-3))
  ([atol rtol] (->DP5 atol rtol)))

(defn tsit5
  "Create a Tsitouras 5(4) adaptive solver."
  ([] (->Tsit5 1e-6 1e-3))
  ([atol rtol] (->Tsit5 atol rtol)))

;; ================================================================
;; Cache types (defvalue auto-generates defrecord + t/ann-record)
;; ================================================================

(defvalue EulerCache (All [T]) [k :- (Array T)]
  :implements FixedStepCache)

(defvalue RK4Cache (All [T]) [k1 :- (Array T), k2 :- (Array T),
                              k3 :- (Array T), k4 :- (Array T),
                              tmp :- (Array T)]
  :implements FixedStepCache)

(defvalue DP5Cache (All [T]) [k1 :- (Array T), k2 :- (Array T),
                              k3 :- (Array T), k4 :- (Array T),
                              k5 :- (Array T), k6 :- (Array T),
                              k7 :- (Array T), tmp :- (Array T),
                              utilde :- (Array T),
                              atol :- Double, rtol :- Double]
  :implements AdaptiveCache)

(defvalue Tsit5Cache (All [T]) [k1 :- (Array T), k2 :- (Array T),
                                k3 :- (Array T), k4 :- (Array T),
                                k5 :- (Array T), k6 :- (Array T),
                                k7 :- (Array T), tmp :- (Array T),
                                utilde :- (Array T),
                                atol :- Double, rtol :- Double]
  :implements AdaptiveCache)

;; ================================================================
;; Cache factory (dispatch on algorithm type)
;; ================================================================

(deftm make-cache
  "Allocate solver workspace arrays for the given algorithm and initial state."
  (All [T] [alg :- Euler, u0 :- (Array T)] :- EulerCache
       (->EulerCache (n/similar u0))))

(deftm make-cache (All [T] [alg :- RK4, u0 :- (Array T)] :- RK4Cache
                       (->RK4Cache (n/similar u0) (n/similar u0) (n/similar u0)
                                   (n/similar u0) (n/similar u0))))

(deftm make-cache [alg :- DP5, u0] :- DP5Cache
  (->DP5Cache (n/similar u0) (n/similar u0) (n/similar u0)
              (n/similar u0) (n/similar u0) (n/similar u0)
              (n/similar u0) (n/similar u0) (n/similar u0)
              (:atol alg) (:rtol alg)))

(deftm make-cache [alg :- Tsit5, u0] :- Tsit5Cache
  (->Tsit5Cache (n/similar u0) (n/similar u0) (n/similar u0)
                (n/similar u0) (n/similar u0) (n/similar u0)
                (n/similar u0) (n/similar u0) (n/similar u0)
                (:atol alg) (:rtol alg)))

;; ================================================================
;; perform-step! — core dispatch on cache type.
;; ================================================================

;; --- Euler: u = u + dt * f(u, t) ---

(deftm perform-step!
  "Advance state u by one step of size dt using the given cache and RHS f."
  (All [T] [cache :- EulerCache, f :- (Fn [(Array T) (Array T) Double] Double), u :- (Array T), t :- Double, dt :- Double]
       (let [t (double t) dt (double dt)
             k (:k cache)]
         (f k u t)
         (let [n (alength u)]
           (dotimes [i n]
             (aset u i (+ (aget u i) (* dt (aget k i)))))))))

;; --- RK4: classic 4th-order Runge-Kutta ---

(deftm perform-step! (All [T] [cache :- RK4Cache, f :- (Fn [(Array T) (Array T) Double] Double), u :- (Array T), t :- Double, dt :- Double]
                          (let [t (double t) dt (double dt)
                                k1 (:k1 cache) k2 (:k2 cache) k3 (:k3 cache) k4 (:k4 cache)
                                tmp (:tmp cache)
                                n (alength u)
                                half-dt (* 0.5 dt)]
                            (f k1 u t)
                            (dotimes [i n] (aset tmp i (+ (aget u i) (* half-dt (aget k1 i)))))
                            (f k2 tmp (+ t half-dt))
                            (dotimes [i n] (aset tmp i (+ (aget u i) (* half-dt (aget k2 i)))))
                            (f k3 tmp (+ t half-dt))
                            (dotimes [i n] (aset tmp i (+ (aget u i) (* dt (aget k3 i)))))
                            (f k4 tmp (+ t dt))
                            (let [dt6 (/ dt 6.0)]
                              (dotimes [i n]
                                (aset u i (+ (aget u i) (* dt6 (+ (aget k1 i) (* 2.0 (aget k2 i))
                                                                  (* 2.0 (aget k3 i)) (aget k4 i)))))))
                            u)))

;; --- Dormand-Prince 5(4): adaptive step-size control ---
;; Butcher tableau coefficients

(def ^:private ^:const dp5-a21 (/ 1.0 5.0))
(def ^:private ^:const dp5-a31 (/ 3.0 40.0))
(def ^:private ^:const dp5-a32 (/ 9.0 40.0))
(def ^:private ^:const dp5-a41 (/ 44.0 45.0))
(def ^:private ^:const dp5-a42 (/ -56.0 15.0))
(def ^:private ^:const dp5-a43 (/ 32.0 9.0))
(def ^:private ^:const dp5-a51 (/ 19372.0 6561.0))
(def ^:private ^:const dp5-a52 (/ -25360.0 2187.0))
(def ^:private ^:const dp5-a53 (/ 64448.0 6561.0))
(def ^:private ^:const dp5-a54 (/ -212.0 729.0))
(def ^:private ^:const dp5-a61 (/ 9017.0 3168.0))
(def ^:private ^:const dp5-a62 (/ -355.0 33.0))
(def ^:private ^:const dp5-a63 (/ 46732.0 5247.0))
(def ^:private ^:const dp5-a64 (/ 49.0 176.0))
(def ^:private ^:const dp5-a65 (/ -5103.0 18656.0))

;; 5th-order weights
(def ^:private ^:const dp5-b1 (/ 35.0 384.0))
(def ^:private ^:const dp5-b3 (/ 500.0 1113.0))
(def ^:private ^:const dp5-b4 (/ 125.0 192.0))
(def ^:private ^:const dp5-b5 (/ -2187.0 6784.0))
(def ^:private ^:const dp5-b6 (/ 11.0 84.0))

;; Error estimation: b - b* (difference between 5th and 4th order)
(def ^:private ^:const dp5-e1 (- (/ 35.0 384.0) (/ 5179.0 57600.0)))
(def ^:private ^:const dp5-e3 (- (/ 500.0 1113.0) (/ 7571.0 16695.0)))
(def ^:private ^:const dp5-e4 (- (/ 125.0 192.0) (/ 393.0 640.0)))
(def ^:private ^:const dp5-e5 (- (/ -2187.0 6784.0) (/ -92097.0 339200.0)))
(def ^:private ^:const dp5-e6 (- (/ 11.0 84.0) (/ 187.0 2100.0)))
(def ^:private ^:const dp5-e7 (/ -1.0 40.0))

;; Time fractions
(def ^:private ^:const dp5-c2 (/ 1.0 5.0))
(def ^:private ^:const dp5-c3 (/ 3.0 10.0))
(def ^:private ^:const dp5-c4 (/ 4.0 5.0))
(def ^:private ^:const dp5-c5 1.0)
(def ^:private ^:const dp5-c6 (/ 8.0 9.0))

;; dp5-step! — one Dormand-Prince step with muladd for FMA
(deftm dp5-step!
  "One Dormand-Prince step. Returns error estimate."
  [cache :- DP5Cache, f :- (Fn [(Array double) (Array double) Double] Double), u :- (Array double), t :- Double, dt :- Double]
  :- Double
  (let [t (double t) dt (double dt)
        k1 (:k1 cache) k2 (:k2 cache) k3 (:k3 cache)
        k4 (:k4 cache) k5 (:k5 cache) k6 (:k6 cache)
        k7 (:k7 cache) utilde (:utilde cache)
        n (alength u)]
    ;; k1 = f(u, t)
    (f k1 u t)
    ;; Stage 2: u + dt*a21*k1
    (f k2 (muladd (broadcast [u k1] (+ u (* dt dp5-a21 k1)))) (+ t (* dp5-c2 dt)))
    ;; Stage 3
    (f k3 (muladd (broadcast [u k1 k2]
                             (+ u (* dt (+ (* dp5-a31 k1) (* dp5-a32 k2)))))) (+ t (* dp5-c3 dt)))
    ;; Stage 4
    (f k4 (muladd (broadcast [u k1 k2 k3]
                             (+ u (* dt (+ (* dp5-a41 k1) (* dp5-a42 k2) (* dp5-a43 k3)))))) (+ t (* dp5-c4 dt)))
    ;; Stage 5
    (f k5 (muladd (broadcast [u k1 k2 k3 k4]
                             (+ u (* dt (+ (* dp5-a51 k1) (* dp5-a52 k2) (* dp5-a53 k3) (* dp5-a54 k4)))))) (+ t (* dp5-c5 dt)))
    ;; Stage 6
    (f k6 (muladd (broadcast [u k1 k2 k3 k4 k5]
                             (+ u (* dt (+ (* dp5-a61 k1) (* dp5-a62 k2) (* dp5-a63 k3)
                                           (* dp5-a64 k4) (* dp5-a65 k5)))))) (+ t (* dp5-c6 dt)))
    ;; 5th-order solution -> utilde
    (acopy! (muladd (broadcast [u k1 k3 k4 k5 k6]
                               (+ u (* dt (+ (* dp5-b1 k1) (* dp5-b3 k3) (* dp5-b4 k4)
                                             (* dp5-b5 k5) (* dp5-b6 k6))))))
            0 utilde 0 n)
    ;; k7 = f(utilde, t+dt) — for FSAL and error
    (f k7 utilde (+ t dt))
    ;; Error estimate
    (let [atol (double (:atol cache))
          rtol (double (:rtol cache))
          err (muladd (reduce! [acc 0.0] [k1 k3 k4 k5 k6 k7 u utilde]
                               (let [ei (* dt (+ (* dp5-e1 k1) (* dp5-e3 k3) (* dp5-e4 k4)
                                                 (* dp5-e5 k5) (* dp5-e6 k6) (* dp5-e7 k7)))
                                     sc (+ atol (* rtol (n/max (n/abs u) (n/abs utilde))))
                                     r (/ ei sc)]
                                 (+ acc (* r r)))))]
      (n/sqrt (/ err n)))))

(deftm perform-step! [cache :- DP5Cache, f :- (Fn [(Array double) (Array double) Double] Double), u :- (Array double), t :- Double, dt :- Double]
  (let [t (double t) dt (double dt)
        err (dp5-step! cache f u t dt)
        utilde (:utilde cache)]
    (acopy! utilde 0 u 0 (alength u))
    err))

;; ================================================================
;; Tsit5: Tsitouras 5(4) — matching Julia's OrdinaryDiffEq.jl
;; Coefficients from Ch. Tsitouras, "Runge–Kutta pairs of order 5(4)
;; satisfying only the first column simplifying assumption"
;; ================================================================

;; Butcher tableau (from Julia's tsit_tableaus.jl, Float64 branch)
(def ^:private ^:const tsit5-c1 0.161)
(def ^:private ^:const tsit5-c2 0.327)
(def ^:private ^:const tsit5-c3 0.9)
(def ^:private ^:const tsit5-c4 0.9800255409045097)

(def ^:private ^:const tsit5-a21 0.161)
(def ^:private ^:const tsit5-a31 -0.008480655492356989)
(def ^:private ^:const tsit5-a32 0.335480655492357)
(def ^:private ^:const tsit5-a41 2.8971530571054935)
(def ^:private ^:const tsit5-a42 -6.359448489975075)
(def ^:private ^:const tsit5-a43 4.3622954328695815)
(def ^:private ^:const tsit5-a51 5.325864828439257)
(def ^:private ^:const tsit5-a52 -11.748883564062828)
(def ^:private ^:const tsit5-a53 7.4955393428898365)
(def ^:private ^:const tsit5-a54 -0.09249506636175525)
(def ^:private ^:const tsit5-a61 5.86145544294642)
(def ^:private ^:const tsit5-a62 -12.92096931784711)
(def ^:private ^:const tsit5-a63 8.159367898576159)
(def ^:private ^:const tsit5-a64 -0.071584973281401)
(def ^:private ^:const tsit5-a65 -0.028269050394068383)
(def ^:private ^:const tsit5-a71 0.09646076681806523)
(def ^:private ^:const tsit5-a72 0.01)
(def ^:private ^:const tsit5-a73 0.4798896504144996)
(def ^:private ^:const tsit5-a74 1.379008574103742)
(def ^:private ^:const tsit5-a75 -3.290069515436081)
(def ^:private ^:const tsit5-a76 2.324710524099774)

;; Error coefficients btilde = b - b* (5th order - 4th order)
(def ^:private ^:const tsit5-btilde1 -0.00178001105222577714)
(def ^:private ^:const tsit5-btilde2 -0.0008164344596567469)
(def ^:private ^:const tsit5-btilde3 0.007880878010261995)
(def ^:private ^:const tsit5-btilde4 -0.1447110071732629)
(def ^:private ^:const tsit5-btilde5 0.5823571654525552)
(def ^:private ^:const tsit5-btilde6 -0.45808210592918697)
(def ^:private ^:const tsit5-btilde7 0.015151515151515152)

;; Dense output interpolation coefficients for Tsit5
;; From Tsitouras 2011, continuous extension of the 5(4) pair.
;; b1(θ) = θ*(r11 + θ*(r12 + θ*(r13 + θ*r14)))
;; b_i(θ) = θ²*(r_i2 + θ*(r_i3 + θ*r_i4))  for i=2..7
(def ^:private ^:const tsit5-r11  1.0)
(def ^:private ^:const tsit5-r12 -2.763706197274826)
(def ^:private ^:const tsit5-r13  2.9132554618219126)
(def ^:private ^:const tsit5-r14 -1.0530884977290216)

(def ^:private ^:const tsit5-r22  0.13169999999999998)
(def ^:private ^:const tsit5-r23 -0.2234)
(def ^:private ^:const tsit5-r24  0.1017)

(def ^:private ^:const tsit5-r32  3.9302962368947516)
(def ^:private ^:const tsit5-r33 -5.941033872131505)
(def ^:private ^:const tsit5-r34  2.490627285651253)

(def ^:private ^:const tsit5-r42 -12.411077166933676)
(def ^:private ^:const tsit5-r43  30.33818863028232)
(def ^:private ^:const tsit5-r44 -16.548102889244902)

(def ^:private ^:const tsit5-r52  37.50931341651104)
(def ^:private ^:const tsit5-r53 -88.1789048947664)
(def ^:private ^:const tsit5-r54  47.37952196281928)

(def ^:private ^:const tsit5-r62 -27.896526289197286)
(def ^:private ^:const tsit5-r63  65.09189467479366)
(def ^:private ^:const tsit5-r64 -34.87065786149661)

(def ^:private ^:const tsit5-r72  1.5)
(def ^:private ^:const tsit5-r73 -4.0)
(def ^:private ^:const tsit5-r74  2.5)

(deftm tsit5-step!
  "One Tsit5 step. Assumes k1 already contains f(uprev, t) (FSAL).
  Writes 5th-order solution to utilde, k7 = f(utilde, t+dt) for FSAL.
  Returns error estimate (RMS norm)."
  [cache :- Tsit5Cache, f :- (Fn [(Array double) (Array double) Double] Double), u :- (Array double), t :- Double, dt :- Double]
  :- Double
  (let [t (double t) dt (double dt)
        k1 (:k1 cache) k2 (:k2 cache) k3 (:k3 cache)
        k4 (:k4 cache) k5 (:k5 cache) k6 (:k6 cache)
        k7 (:k7 cache) utilde (:utilde cache)
        n (alength u)]
    ;; k1 already set (FSAL from previous step or initialization)
    ;; Stage 2: u + dt*a21*k1
    (f k2 (muladd (broadcast [u k1] (+ u (* dt tsit5-a21 k1)))) (+ t (* tsit5-c1 dt)))
    ;; Stage 3
    (f k3 (muladd (broadcast [u k1 k2]
                             (+ u (* dt (+ (* tsit5-a31 k1) (* tsit5-a32 k2)))))) (+ t (* tsit5-c2 dt)))
    ;; Stage 4
    (f k4 (muladd (broadcast [u k1 k2 k3]
                             (+ u (* dt (+ (* tsit5-a41 k1) (* tsit5-a42 k2) (* tsit5-a43 k3)))))) (+ t (* tsit5-c3 dt)))
    ;; Stage 5
    (f k5 (muladd (broadcast [u k1 k2 k3 k4]
                             (+ u (* dt (+ (* tsit5-a51 k1) (* tsit5-a52 k2) (* tsit5-a53 k3) (* tsit5-a54 k4)))))) (+ t (* tsit5-c4 dt)))
    ;; Stage 6
    (f k6 (muladd (broadcast [u k1 k2 k3 k4 k5]
                             (+ u (* dt (+ (* tsit5-a61 k1) (* tsit5-a62 k2) (* tsit5-a63 k3)
                                           (* tsit5-a64 k4) (* tsit5-a65 k5)))))) (+ t dt))
    ;; 5th-order solution -> utilde  (a71..a76 weights)
    (acopy! (muladd (broadcast [u k1 k2 k3 k4 k5 k6]
                               (+ u (* dt (+ (* tsit5-a71 k1) (* tsit5-a72 k2) (* tsit5-a73 k3)
                                             (* tsit5-a74 k4) (* tsit5-a75 k5) (* tsit5-a76 k6))))))
            0 utilde 0 n)
    ;; k7 = f(utilde, t+dt) — FSAL: becomes k1 of next step
    (f k7 utilde (+ t dt))
    ;; Error estimate: RMS of (dt * btilde_i * k_i) / (atol + rtol * max(|uprev|, |u|))
    (let [atol (double (:atol cache))
          rtol (double (:rtol cache))
          err (muladd (reduce! [acc 0.0] [k1 k2 k3 k4 k5 k6 k7 u utilde]
                               (let [ei (* dt (+ (* tsit5-btilde1 k1) (* tsit5-btilde2 k2) (* tsit5-btilde3 k3)
                                                 (* tsit5-btilde4 k4) (* tsit5-btilde5 k5) (* tsit5-btilde6 k6)
                                                 (* tsit5-btilde7 k7)))
                      ;; Julia's calculate_residuals: max(|uprev|, |u_new|)
                                     sc (+ atol (* rtol (n/max (n/abs u) (n/abs utilde))))
                                     r (/ ei sc)]
                                 (+ acc (* r r)))))]
      (n/sqrt (/ err n)))))

(deftm perform-step! [cache :- Tsit5Cache, f :- (Fn [(Array double) (Array double) Double] Double), u :- (Array double), t :- Double, dt :- Double]
  (let [t (double t) dt (double dt)
        err (tsit5-step! cache f u t dt)
        utilde (:utilde cache)]
    (acopy! utilde 0 u 0 (alength u))
    err))

;; ================================================================
;; Implicit Euler — for stiff systems
;; ================================================================

(defvalue ImplicitEuler [tol :- Double, max-iter :- Long] :implements ImplicitAlgorithm)

(defn implicit-euler
  "Create an implicit Euler solver for stiff problems."
  ([] (->ImplicitEuler 1e-10 20))
  ([tol max-iter] (->ImplicitEuler tol (long max-iter))))

;; Cache includes u-old for Newton iteration
(defvalue ImplicitEulerCache (All [T])
  [k :- (Array T), J :- (Array T), pivot :- (Array int),
   delta :- (Array T), work :- (Array T),
   u-old :- (Array T), n :- Long]
  :implements ImplicitCache)

(deftm make-cache [alg :- ImplicitEuler, n] :- ImplicitEulerCache
  (let [n (int n)]
    (->ImplicitEulerCache
     (double-array n) (double-array (* n n)) (int-array n)
     (double-array n) (double-array n) (double-array n) n)))

;; --- LU decomposition with partial pivoting (in-place) ---

(deftm lu-decompose!
  "LU decomposition with partial pivoting. Overwrites J in-place.
  Returns true if successful, false if singular."
  [J :- (Array double), pivot :- (Array int), n :- Long]
  (let [n (long n)]
    (dotimes [i n] (aset pivot i (int i)))
    (loop [k 0]
      (if (>= k n)
        true
      ;; Find pivot row
        (let [pivot-row (loop [i (inc k) best-i k
                               best-v (n/abs (aget J (+ (* k n) k)))]
                          (if (>= i n)
                            best-i
                            (let [v (n/abs (aget J (+ (* i n) k)))]
                              (if (> v best-v)
                                (recur (inc i) i v)
                                (recur (inc i) best-i best-v)))))]
          (when (not= pivot-row k)
          ;; Swap rows
            (let [tmp (aget pivot k)]
              (aset pivot k (aget pivot pivot-row))
              (aset pivot pivot-row (int tmp)))
            (dotimes [j n]
              (let [ik (+ (* k n) j) ip (+ (* pivot-row n) j)
                    tmp (aget J ik)]
                (aset J ik (aget J ip))
                (aset J ip tmp))))
          (let [diag (aget J (+ (* k n) k))]
            (if (< (n/abs diag) 1e-30)
              false
              (do
                (loop [i (inc k)]
                  (when (< i n)
                    (let [factor (/ (aget J (+ (* i n) k)) diag)]
                      (aset J (+ (* i n) k) factor)
                      (loop [j (inc k)]
                        (when (< j n)
                          (aset J (+ (* i n) j)
                                (- (aget J (+ (* i n) j))
                                   (* factor (aget J (+ (* k n) j)))))
                          (recur (inc j)))))
                    (recur (inc i))))
                (recur (inc k))))))))))

(deftm lu-solve!
  "Solve Jx = b using pre-computed LU decomposition. Overwrites b with solution."
  [J :- (Array double), pivot :- (Array int), b :- (Array double), n :- Long]
  ;; Apply permutation and forward substitution (Ly = Pb)
  (let [n (long n) y (double-array n)]
    (dotimes [i n]
      (aset y i (aget b (aget pivot i))))
    (loop [i 1]
      (when (< i n)
        (loop [j 0]
          (when (< j i)
            (aset y i (- (aget y i) (* (aget J (+ (* i n) j)) (aget y j))))
            (recur (inc j))))
        (recur (inc i))))
    ;; Back substitution (Ux = y)
    (loop [i (dec n)]
      (when (>= i 0)
        (let [sum (loop [j (inc i) s 0.0]
                    (if (>= j n) s
                        (recur (inc j) (+ s (* (aget J (+ (* i n) j)) (aget b j))))))]
          (aset b i (/ (- (aget y i) sum) (aget J (+ (* i n) i)))))
        (recur (dec i))))))

;; --- Numerical Jacobian via forward finite differences ---

(deftm compute-jacobian!
  "Compute J_f = df/du via forward finite differences.
  J is n×n row-major. f0 = f(u,t) already computed."
  [f :- (Fn [(Array double) (Array double) Double] Double), J :- (Array double), u :- (Array double), t :- Double, f0 :- (Array double), work :- (Array double), n :- Long]
  (let [t (double t) n (long n) eps 1e-8]
    (dotimes [j n]
      (let [uj (aget u j)
            h (* eps (n/max 1.0 (n/abs uj)))]
        (aset u j (+ uj h))
        (f work u t)
        (aset u j uj)
        (dotimes [i n]
          (aset J (+ (* i n) j)
                (/ (- (aget work i) (aget f0 i)) h)))))))

;; --- Implicit Euler step ---
;; Solve: u_{n+1} = u_n + dt * f(u_{n+1}, t_{n+1})
;; Newton on G(x) = x - u_n - dt*f(x, t_{n+1}) = 0
;; J_G = I - dt*J_f
;; Update: x_{k+1} = x_k - J_G^{-1} * G(x_k)

(deftm perform-step!
  [cache :- ImplicitEulerCache, f :- (Fn [(Array double) (Array double) Double] Double), u :- (Array double), t :- Double, dt :- Double]
  (let [t (double t) dt (double dt)
        n     (long (:n cache))
        k     (:k cache)
        J-buf (:J cache)
        pivot (:pivot cache)
        delta (:delta cache)
        work  (:work cache)
        u-old (:u-old cache)
        t-new (+ t dt)]
    ;; Save u_old
    (acopy! u 0 u-old 0 n)
    ;; Initial guess: explicit Euler
    (f k u t)
    (dotimes [i n]
      (aset u i (+ (aget u-old i) (* dt (aget k i)))))
    ;; Newton iteration
    (loop [iter 0]
      (if (>= iter 20)
        u
        (do
          ;; f(u_k, t_{n+1})
          (f k u t-new)
          ;; Residual G(u_k) = u_k - u_old - dt*f(u_k)
          (dotimes [i n]
            (aset delta i (- (aget u i) (aget u-old i) (* dt (aget k i)))))
          ;; Check convergence: |G| < tol
          (let [norm (loop [i 0 s 0.0]
                       (if (>= i n) (n/sqrt s)
                           (let [d (aget delta i)]
                             (recur (inc i) (+ s (* d d))))))]
            (if (< norm 1e-10)
              u
              (do
                ;; Build J_G = I - dt*J_f
                (compute-jacobian! f J-buf u t-new k work n)
                (dotimes [i n]
                  (dotimes [j n]
                    (let [idx (+ (* i n) j)]
                      (aset J-buf idx (- (if (= i j) 1.0 0.0)
                                         (* dt (aget J-buf idx)))))))
                ;; Solve J_G * correction = G  (delta already holds G)
                (lu-decompose! J-buf pivot n)
                (lu-solve! J-buf pivot delta n)
                ;; u_{k+1} = u_k - correction
                (dotimes [i n]
                  (aset u i (- (aget u i) (aget delta i))))
                (recur (inc iter))))))))
    u))

;; ================================================================
;; Solve options
;; ================================================================

(defrecord SolveOpts [save-everystep])

;; Public solve interface — dispatch on algorithm type

;; solve-fixed-step-impl: parametric inner function that specializes on state type T.
;; The walker re-walks this for each concrete T, resolving perform-step!/make-cache
;; to the right overloads (Julia-style whole-function specialization).
(deftm solve-fixed-step-impl
  "Inner fixed-step solver loop, parametric over state element type T."
  (All [T]
       [alg :- Object, f :- Object, u0 :- (Array T),
        t0 :- Double, tf :- Double, dt :- Double]
       (let [t0 (double t0) tf (double tf) dt (double dt)
             n (alength u0)
             cache (make-cache alg u0)
             u (n/similar u0)]
         (acopy! u0 0 u 0 n)
         (loop [t t0
                ts (transient [t0])
                us (transient [(aclone u0)])]
           (if (>= t tf)
             (let [ts-out (persistent! ts)
                   us-out (persistent! us)]
               {:ts ts-out :us us-out :iterations (dec (count ts-out))})
             (let [dt-actual (n/min dt (- tf t))]
               (perform-step! cache f u t dt-actual)
               (let [t-new (+ t dt-actual)]
                 (recur t-new
                        (conj! ts t-new)
                        (conj! us (aclone u))))))))))

;; Thin wrappers that unpack ODEProblem and delegate to the parametric impl.
;; T is inferred from u0's runtime type.

(deftm solve
  "Integrate an ODE problem using the given algorithm and step size."
  [alg :- Euler, prob :- ODEProblem, dt :- Double]
  (let [u0 (:u0 prob)
        tspan (:tspan prob)]
    (solve-fixed-step-impl alg (:f prob) u0
                           (double (nth tspan 0)) (double (nth tspan 1)) dt)))

(deftm solve [alg :- RK4, prob :- ODEProblem, dt :- Double]
  (let [u0 (:u0 prob)
        tspan (:tspan prob)]
    (solve-fixed-step-impl alg (:f prob) u0
                           (double (nth tspan 0)) (double (nth tspan 1)) dt)))

(deftm solve [alg :- DP5, prob :- ODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        save? (get (:opts prob) :save-everystep true)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt0 (double dt)
        n (alength u0)
        cache (make-cache alg u0)
        u (n/similar u0)
        u-save (n/similar u0)
        utilde (:utilde cache)]
    (acopy! u0 0 u 0 n)
    (if save?
      ;; Full trajectory storage
      (loop [t t0
             dt-loop dt0
             ts (transient [t0])
             us (transient [(aclone u0)])
             nreject 0]
        (if (>= t tf)
          (let [ts-out (persistent! ts)
                us-out (persistent! us)]
            {:ts ts-out :us us-out
             :iterations (dec (count ts-out)) :nreject nreject})
          (let [dt-actual (n/min dt-loop (- tf t))
                _ (acopy! u 0 u-save 0 n)
                err (double (dp5-step! cache f u t dt-actual))]
            (if (<= err 1.0)
              (let [_ (acopy! utilde 0 u 0 n)
                    t-new (+ t dt-actual)
                    factor (n/min 5.0
                                  (n/max 0.2
                                         (* 0.9 (n/pow err -0.2))))]
                (recur t-new
                       (* dt-actual factor)
                       (conj! ts t-new)
                       (conj! us (aclone u))
                       nreject))
              (let [factor (n/max 0.2 (* 0.9 (n/pow err -0.2)))
                    _ (acopy! u-save 0 u 0 n)]
                (recur t
                       (* dt-actual factor)
                       ts us
                       (inc nreject)))))))
      ;; No-save path: return final state only
      (loop [t t0
             dt-loop dt0
             nreject 0]
        (if (>= t tf)
          {:u (aclone u) :t t :nreject nreject}
          (let [dt-actual (n/min dt-loop (- tf t))
                _ (acopy! u 0 u-save 0 n)
                err (double (dp5-step! cache f u t dt-actual))]
            (if (<= err 1.0)
              (let [_ (acopy! utilde 0 u 0 n)
                    t-new (+ t dt-actual)
                    factor (n/min 5.0
                                  (n/max 0.2
                                         (* 0.9 (n/pow err -0.2))))]
                (recur t-new
                       (* dt-actual factor)
                       nreject))
              (let [factor (n/max 0.2 (* 0.9 (n/pow err -0.2)))
                    _ (acopy! u-save 0 u 0 n)]
                (recur t
                       (* dt-actual factor)
                       (inc nreject))))))))))

(deftm solve [alg :- Tsit5, prob :- ODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        save? (get (:opts prob) :save-everystep true)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt0 (double dt)
        n (alength u0)
        cache (make-cache alg u0)
        u (n/similar u0)
        u-save (n/similar u0)
        k1 (:k1 cache)
        k2 (:k2 cache)
        k3 (:k3 cache)
        k4 (:k4 cache)
        k5 (:k5 cache)
        k6 (:k6 cache)
        k7 (:k7 cache)
        utilde (:utilde cache)
        beta1 (/ 7.0 50.0)
        beta2 (/ 2.0 25.0)
        safety 0.9]
    (acopy! u0 0 u 0 n)
    ;; Initialize FSAL: k1 = f(u0, t0)
    (f k1 u t0)
    (if save?
      ;; Full trajectory storage
      (loop [t t0
             dt-loop dt0
             EEst-prev 1.0
             ts (transient [t0])
             us (transient [(aclone u0)])
             ks (transient [])
             nreject 0
             nf 1]
        (if (>= t tf)
          (let [ts-out (persistent! ts)
                us-out (persistent! us)]
            {:ts ts-out :us us-out :ks (persistent! ks)
             :iterations (dec (count ts-out)) :nreject nreject :nf nf})
          (let [dt-actual (n/min dt-loop (- tf t))
                _ (acopy! u 0 u-save 0 n)
                err (double (tsit5-step! cache f u t dt-actual))
                nf-new (+ nf 6)]
            (if (<= err 1.0)
              (let [;; Clone stages BEFORE FSAL copy for dense output
                    step-ks [(aclone k1) (aclone k2) (aclone k3)
                             (aclone k4) (aclone k5) (aclone k6) (aclone k7)]
                    _ (acopy! utilde 0 u 0 n)
                    _ (acopy! k7 0 k1 0 n)
                    t-new (+ t dt-actual)
                    factor (n/min 5.0
                                  (n/max 0.2
                                         (* safety
                                            (n/pow err (- beta1))
                                            (n/pow EEst-prev beta2))))]
                (recur t-new
                       (* dt-actual factor)
                       err
                       (conj! ts t-new)
                       (conj! us (aclone u))
                       (conj! ks step-ks)
                       nreject
                       nf-new))
              (let [factor (n/max 0.2 (* safety (n/pow err (- beta1))))
                    _ (acopy! u-save 0 u 0 n)]
                (recur t
                       (* dt-actual factor)
                       EEst-prev
                       ts us
                       ks
                       (inc nreject)
                       nf-new))))))
      ;; No-save path: return final state only
      (loop [t t0
             dt-loop dt0
             EEst-prev 1.0
             nreject 0
             nf 1]
        (if (>= t tf)
          {:u (aclone u) :t t :nreject nreject :nf nf}
          (let [dt-actual (n/min dt-loop (- tf t))
                _ (acopy! u 0 u-save 0 n)
                err (double (tsit5-step! cache f u t dt-actual))
                nf-new (+ nf 6)]
            (if (<= err 1.0)
              (let [_ (acopy! utilde 0 u 0 n)
                    _ (acopy! k7 0 k1 0 n)
                    t-new (+ t dt-actual)
                    factor (n/min 5.0
                                  (n/max 0.2
                                         (* safety
                                            (n/pow err (- beta1))
                                            (n/pow EEst-prev beta2))))]
                (recur t-new
                       (* dt-actual factor)
                       err
                       nreject
                       nf-new))
              (let [factor (n/max 0.2 (* safety (n/pow err (- beta1))))
                    _ (acopy! u-save 0 u 0 n)]
                (recur t
                       (* dt-actual factor)
                       EEst-prev
                       (inc nreject)
                       nf-new)))))))))

(deftm solve [alg :- ImplicitEuler, prob :- ODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt (double dt)
        n (alength u0)
        cache (make-cache alg n)
        u (n/similar u0)]
    (acopy! u0 0 u 0 n)
    (loop [t t0
           ts (transient [t0])
           us (transient [(aclone u0)])]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out :us us-out :iterations (dec (count ts-out))})
        (let [dt-actual (n/min dt (- tf t))]
          (perform-step! cache f u t dt-actual)
          (let [t-new (+ t dt-actual)]
            (recur t-new
                   (conj! ts t-new)
                   (conj! us (aclone u)))))))))

;; ================================================================
;; Rosenbrock23 (Shampine & Reichelt, 1997) — L-stable, order 2(3)
;; Semi-implicit Rosenbrock-W method for stiff problems.
;; One Jacobian + LU factorization per step, 3 back-solves,
;; 3 function evaluations, embedded error estimation.
;; ================================================================

(defvalue Rosenbrock23 [atol :- Double, rtol :- Double]
  :implements ImplicitAlgorithm)

(defn rosenbrock23
  "Create a Rosenbrock23 solver for stiff problems."
  ([] (->Rosenbrock23 1e-8 1e-6))
  ([atol rtol] (->Rosenbrock23 atol rtol)))

(defvalue Rosenbrock23Cache (All [T])
  [f0     :- (Array T)    ;; f(u_n) — FSAL
   f1     :- (Array T)    ;; f at midpoint
   k1     :- (Array T)    ;; stage 1
   k2     :- (Array T)    ;; stage 2
   linrhs :- (Array T)    ;; RHS for linear solve (overwritten with solution)
   J      :- (Array T)    ;; Jacobian → W matrix (n×n row-major)
   pivot  :- (Array int)       ;; LU pivoting
   work   :- (Array T)    ;; Jacobian FD scratch
   utilde :- (Array T)    ;; solution candidate u_{n+1}
   atol   :- Double
   rtol   :- Double
   n      :- Long]
  :implements ImplicitCache)

(deftm make-cache [alg :- Rosenbrock23, n] :- Rosenbrock23Cache
  (let [n (int n)]
    (->Rosenbrock23Cache
     (double-array n) (double-array n) (double-array n) (double-array n)
     (double-array n) (double-array (* n n)) (int-array n) (double-array n)
     (double-array n) (.atol ^Rosenbrock23 alg) (.rtol ^Rosenbrock23 alg) n)))

;; Rosenbrock23 coefficients (Shampine & Reichelt 1997)
(def ^:private ^:const ros23-d (/ 1.0 (+ 2.0 (n/sqrt 2.0))))    ;; gamma = 1/(2+√2) ≈ 0.2929
(def ^:private ^:const ros23-c32 (+ 6.0 (n/sqrt 2.0)))           ;; c_32 = 6+√2 ≈ 7.4142

(deftm rosenbrock23-step!
  [cache :- Rosenbrock23Cache,
   f :- (Fn [(Array double) (Array double) Double] Double),
   u :- (Array double), t :- Double, dt :- Double]
  :- Double
  (let [t     (double t)
        dt    (double dt)
        n     (long (:n cache))
        f0    (:f0 cache)
        f1    (:f1 cache)
        k1    (:k1 cache)
        k2    (:k2 cache)
        linrhs (:linrhs cache)
        J-buf (:J cache)
        pivot (:pivot cache)
        work  (:work cache)
        utilde (:utilde cache)
        d     (double ros23-d)
        c32   (double ros23-c32)
        dtgamma      (* dt d)
        neginvdtgamma (/ -1.0 dtgamma)
        invdtgamma   (/ 1.0 dtgamma)]

    ;; f0 = f(u, t) — already set if FSAL, else compute
    ;; (caller must ensure f0 is set for FSAL; on first step we compute it)

    ;; Compute Jacobian J = df/du at (u, t)
    (compute-jacobian! f J-buf u t f0 work n)

    ;; Form W = J - I/(dtgamma) in-place (Julia convention)
    (dotimes [i n]
      (dotimes [j n]
        (let [idx (+ (* i n) j)]
          (aset J-buf idx (- (aget J-buf idx)
                             (if (= i j) invdtgamma 0.0))))))

    ;; LU decompose W
    (lu-decompose! J-buf pivot n)

    ;; --- Stage 1: W * u1 = f0, then k1 = u1 * neginvdtgamma ---
    (acopy! f0 0 linrhs 0 n)
    (lu-solve! J-buf pivot linrhs n)
    ;; k1 = linrhs * neginvdtgamma
    (dotimes [i n]
      (aset k1 i (* (aget linrhs i) neginvdtgamma)))

    ;; --- Stage 2: g2 = u + (dt/2)*k1, f1 = f(g2), solve for k2 ---
    ;; g2 stored in utilde temporarily
    (dotimes [i n]
      (aset utilde i (+ (aget u i) (* (* 0.5 dt) (aget k1 i)))))
    (f f1 utilde (+ t (* 0.5 dt)))
    ;; RHS = f1 - k1
    (dotimes [i n]
      (aset linrhs i (- (aget f1 i) (aget k1 i))))
    (lu-solve! J-buf pivot linrhs n)
    ;; k2 = linrhs * neginvdtgamma + k1
    (dotimes [i n]
      (aset k2 i (+ (* (aget linrhs i) neginvdtgamma) (aget k1 i))))

    ;; --- Solution: utilde = u + dt * k2 ---
    (dotimes [i n]
      (aset utilde i (+ (aget u i) (* dt (aget k2 i)))))

    ;; --- Stage 3: error estimation ---
    ;; f2 = f(utilde, t+dt) — store in work temporarily
    (f work utilde (+ t dt))
    ;; RHS = f2 - c32*(k2 - f1) - 2*(k1 - f0)
    (dotimes [i n]
      (aset linrhs i (- (aget work i)
                        (* c32 (- (aget k2 i) (aget f1 i)))
                        (* 2.0 (- (aget k1 i) (aget f0 i))))))
    (lu-solve! J-buf pivot linrhs n)
    ;; k3 = linrhs * neginvdtgamma (stored in linrhs itself for error calc)

    ;; --- Error: err = (dt/6) * (k1 - 2*k2 + k3), normalized RMS ---
    (let [atol (double (:atol cache))
          rtol (double (:rtol cache))
          dt6 (/ dt 6.0)]
      (loop [i 0, acc 0.0]
        (if (>= i n)
          (n/sqrt (/ acc (double n)))
          (let [k3i (* (aget linrhs i) neginvdtgamma)
                ei (* dt6 (+ (aget k1 i) (* -2.0 (aget k2 i)) k3i))
                sc (+ atol (* rtol (n/max (n/abs (aget u i))
                                          (n/abs (aget utilde i)))))
                r (/ ei sc)]
            (recur (inc i) (+ acc (* r r)))))))))

(deftm perform-step!
  [cache :- Rosenbrock23Cache,
   f :- (Fn [(Array double) (Array double) Double] Double),
   u :- (Array double), t :- Double, dt :- Double]
  ;; Compute f0 for initial call (non-FSAL interface)
  (let [f0 (:f0 cache)]
    (f f0 u t)
    (let [err (rosenbrock23-step! cache f u t dt)]
      ;; Copy solution to u
      (acopy! (:utilde cache) 0 u 0 (long (:n cache)))
      err)))

(deftm solve [alg :- Rosenbrock23, prob :- ODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt0 (double dt)
        n (alength u0)
        cache (make-cache alg n)
        u (n/similar u0)
        u-save (n/similar u0)
        utilde (:utilde cache)
        f0 (:f0 cache)]
    (acopy! u0 0 u 0 n)
    ;; Initial f evaluation (FSAL start)
    (f f0 u t0)
    (loop [t t0
           dt-loop dt0
           ts (transient [t0])
           us (transient [(aclone u0)])
           nreject 0
           nf 1]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out
           :us us-out
           :iterations (dec (count ts-out))
           :nreject nreject
           :nf nf})
        (let [dt-actual (n/min dt-loop (- tf t))
              _ (acopy! u 0 u-save 0 n)
              err (double (rosenbrock23-step! cache f u t dt-actual))]
          (if (<= err 1.0)
            ;; Accept: copy utilde→u, FSAL: f(utilde) already in work
            (let [_ (acopy! utilde 0 u 0 n)
                  ;; FSAL: f(u_{n+1}) was computed as stage 3's f2 in work
                  _ (acopy! (:work cache) 0 f0 0 n)
                  t-new (+ t dt-actual)
                  ;; I-controller for order 2 method: exponent = -1/3
                  factor (n/min 5.0
                                (n/max 0.2
                                       (* 0.9 (n/pow err (/ -1.0 3.0)))))]
              (recur t-new
                     (* dt-actual factor)
                     (conj! ts t-new)
                     (conj! us (aclone u))
                     nreject
                     (+ nf 3)))
            ;; Reject: restore u, shrink dt
            (let [factor (n/max 0.2 (* 0.9 (n/pow err (/ -1.0 3.0))))
                  _ (acopy! u-save 0 u 0 n)]
              (recur t
                     (* dt-actual factor)
                     ts us
                     (inc nreject)
                     (+ nf 3)))))))))

;; ================================================================
;; SDIRK4: Singly Diagonally Implicit Runge-Kutta, 4th order
;; 5-stage, L-stable. Hairer & Wanner, "Solving Ordinary Differential
;; Equations II", Table IV.6.5 (the SDIRK4 method).
;;
;; Key property: all diagonal entries gamma are equal, so only ONE
;; LU factorization per step (of W = I - dt*gamma*J) reused for
;; all 5 stage solves. Embedded 3rd-order method for error estimation.
;; ================================================================

(defvalue SDIRK4 [atol :- Double, rtol :- Double]
  :implements ImplicitAlgorithm)

(defn sdirk4
  "Create a SDIRK4 solver for stiff problems."
  ([] (->SDIRK4 1e-8 1e-6))
  ([atol rtol] (->SDIRK4 atol rtol)))

(defvalue SDIRK4Cache (All [T])
  [k1 :- (Array T)     ;; stage vectors
   k2 :- (Array T)
   k3 :- (Array T)
   k4 :- (Array T)
   k5 :- (Array T)
   linrhs :- (Array T) ;; RHS for linear solve
   J  :- (Array T)     ;; Jacobian / W matrix (n×n row-major)
   W  :- (Array T)     ;; W = I - dt*gamma*J (separate copy for reuse)
   pivot :- (Array int)
   work :- (Array T)   ;; scratch for Jacobian FD
   ftmp :- (Array T)   ;; scratch for f evaluations
   utilde :- (Array T) ;; solution candidate
   atol :- Double
   rtol :- Double
   n :- Long]
  :implements ImplicitCache)

(deftm make-cache [alg :- SDIRK4, n] :- SDIRK4Cache
  (let [n (int n)]
    (->SDIRK4Cache
     (double-array n) (double-array n) (double-array n)
     (double-array n) (double-array n)
     (double-array n) (double-array (* n n)) (double-array (* n n))
     (int-array n) (double-array n) (double-array n) (double-array n)
     (.atol ^SDIRK4 alg) (.rtol ^SDIRK4 alg) n)))

;; SDIRK4 Butcher tableau (Hairer & Wanner, Table IV.6.5)
;; gamma = 1/4
;;
;; c | A
;; ------+------------------------------------------
;; 1/4   | 1/4
;; 3/4   | 1/2       1/4
;; 11/20 | 17/50     -1/25     1/4
;; 1/2   | 371/1360  -137/2720 15/544    1/4
;; 1     | 25/24     -49/48    125/16    -85/12    1/4
;; ------+------------------------------------------
;;   b   | 25/24     -49/48    125/16    -85/12    1/4
;;   b̂   | 59/48     -17/96    225/32    -85/12    0
;;
;; Note: b = last row of A (stiffly accurate), so u_{n+1} = last stage.

(def ^:private ^:const sdirk4-gamma 0.25)

;; a_{i,j} coefficients (lower-triangular, excluding diagonal gamma)
(def ^:private ^:const sdirk4-a21 0.5)
(def ^:private ^:const sdirk4-a31 (/ 17.0 50.0))
(def ^:private ^:const sdirk4-a32 (/ -1.0 25.0))
(def ^:private ^:const sdirk4-a41 (/ 371.0 1360.0))
(def ^:private ^:const sdirk4-a42 (/ -137.0 2720.0))
(def ^:private ^:const sdirk4-a43 (/ 15.0 544.0))
(def ^:private ^:const sdirk4-a51 (/ 25.0 24.0))
(def ^:private ^:const sdirk4-a52 (/ -49.0 48.0))
(def ^:private ^:const sdirk4-a53 (/ 125.0 16.0))
(def ^:private ^:const sdirk4-a54 (/ -85.0 12.0))

;; c_i values
(def ^:private ^:const sdirk4-c2 0.75)
(def ^:private ^:const sdirk4-c3 (/ 11.0 20.0))
(def ^:private ^:const sdirk4-c4 0.5)
(def ^:private ^:const sdirk4-c5 1.0)

;; Embedded 3rd-order method weights (for error estimation)
(def ^:private ^:const sdirk4-bhat1 (/ 59.0 48.0))
(def ^:private ^:const sdirk4-bhat2 (/ -17.0 96.0))
(def ^:private ^:const sdirk4-bhat3 (/ 225.0 32.0))
(def ^:private ^:const sdirk4-bhat4 (/ -85.0 12.0))
(def ^:private ^:const sdirk4-bhat5 0.0)

;; b_i weights = last row of A (stiffly accurate)
(def ^:private ^:const sdirk4-b1 sdirk4-a51)
(def ^:private ^:const sdirk4-b2 sdirk4-a52)
(def ^:private ^:const sdirk4-b3 sdirk4-a53)
(def ^:private ^:const sdirk4-b4 sdirk4-a54)
(def ^:private ^:const sdirk4-b5 sdirk4-gamma)

(deftm sdirk4-step!
  "One SDIRK4 step. Returns normalized error estimate.
   Writes 4th-order solution to utilde."
  [cache :- SDIRK4Cache,
   f :- (Fn [(Array double) (Array double) Double] Double),
   u :- (Array double), t :- Double, dt :- Double]
  :- Double
  (let [t     (double t)
        dt    (double dt)
        n     (long (:n cache))
        k1    (:k1 cache)
        k2    (:k2 cache)
        k3    (:k3 cache)
        k4    (:k4 cache)
        k5    (:k5 cache)
        linrhs (:linrhs cache)
        J-buf (:J cache)
        W-buf (:W cache)
        pivot (:pivot cache)
        work  (:work cache)
        ftmp  (:ftmp cache)
        utilde (:utilde cache)
        gamma (double sdirk4-gamma)
        dtg   (* dt gamma)]

    ;; --- Compute Jacobian at (u, t) ---
    ;; We need f(u, t) first for Jacobian FD
    (f ftmp u t)
    (compute-jacobian! f J-buf u t ftmp work n)

    ;; --- Form W = I - dt*gamma*J, then LU-factorize ---
    ;; Copy J into W, then modify W
    (acopy! J-buf 0 W-buf 0 (* n n))
    (dotimes [i n]
      (dotimes [j n]
        (let [idx (+ (* i n) j)]
          (aset W-buf idx (- (if (= i j) 1.0 0.0)
                             (* dtg (aget W-buf idx)))))))
    (lu-decompose! W-buf pivot n)

    ;; --- Stage 1: y1 = u + dt*gamma*k1 ---
    ;; Solve: W * k1 = f(u, t)  (initial guess y1 = u, f(u,t) = ftmp)
    ;; Actually for SDIRK: solve (I - dt*gamma*J) * k1 = f(u + dt*gamma*k1, t + c1*dt)
    ;; Simplified Newton: use f(u, t) as RHS, iterate once
    ;; Standard approach: k1 = W^{-1} * f(u, t)
    (acopy! ftmp 0 k1 0 n)
    (lu-solve! W-buf pivot k1 n)

    ;; --- Stage 2: y2 = u + dt*(a21*k1 + gamma*k2) ---
    ;; RHS: f(u + dt*a21*k1, t + c2*dt)
    (dotimes [i n]
      (aset utilde i (+ (aget u i) (* dt sdirk4-a21 (aget k1 i)))))
    (f linrhs utilde (+ t (* sdirk4-c2 dt)))
    (acopy! linrhs 0 k2 0 n)
    (lu-solve! W-buf pivot k2 n)

    ;; --- Stage 3: y3 = u + dt*(a31*k1 + a32*k2 + gamma*k3) ---
    (dotimes [i n]
      (aset utilde i (+ (aget u i)
                        (* dt (+ (* sdirk4-a31 (aget k1 i))
                                 (* sdirk4-a32 (aget k2 i)))))))
    (f linrhs utilde (+ t (* sdirk4-c3 dt)))
    (acopy! linrhs 0 k3 0 n)
    (lu-solve! W-buf pivot k3 n)

    ;; --- Stage 4: y4 = u + dt*(a41*k1 + a42*k2 + a43*k3 + gamma*k4) ---
    (dotimes [i n]
      (aset utilde i (+ (aget u i)
                        (* dt (+ (* sdirk4-a41 (aget k1 i))
                                 (* sdirk4-a42 (aget k2 i))
                                 (* sdirk4-a43 (aget k3 i)))))))
    (f linrhs utilde (+ t (* sdirk4-c4 dt)))
    (acopy! linrhs 0 k4 0 n)
    (lu-solve! W-buf pivot k4 n)

    ;; --- Stage 5: y5 = u + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4 + gamma*k5) ---
    (dotimes [i n]
      (aset utilde i (+ (aget u i)
                        (* dt (+ (* sdirk4-a51 (aget k1 i))
                                 (* sdirk4-a52 (aget k2 i))
                                 (* sdirk4-a53 (aget k3 i))
                                 (* sdirk4-a54 (aget k4 i)))))))
    (f linrhs utilde (+ t (* sdirk4-c5 dt)))
    (acopy! linrhs 0 k5 0 n)
    (lu-solve! W-buf pivot k5 n)

    ;; --- Solution: u_{n+1} = u + dt*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5) ---
    ;; Stiffly accurate: b = last row of A, so this equals y5 + dt*gamma*k5
    (dotimes [i n]
      (aset utilde i (+ (aget u i)
                        (* dt (+ (* sdirk4-b1 (aget k1 i))
                                 (* sdirk4-b2 (aget k2 i))
                                 (* sdirk4-b3 (aget k3 i))
                                 (* sdirk4-b4 (aget k4 i))
                                 (* sdirk4-b5 (aget k5 i)))))))

    ;; --- Error estimation: e = dt * sum((b_i - bhat_i) * k_i) ---
    (let [atol (double (:atol cache))
          rtol (double (:rtol cache))
          db1 (- sdirk4-b1 sdirk4-bhat1)
          db2 (- sdirk4-b2 sdirk4-bhat2)
          db3 (- sdirk4-b3 sdirk4-bhat3)
          db4 (- sdirk4-b4 sdirk4-bhat4)
          db5 (- sdirk4-b5 sdirk4-bhat5)]
      (loop [i 0, acc 0.0]
        (if (>= i n)
          (n/sqrt (/ acc (double n)))
          (let [ei (* dt (+ (* db1 (aget k1 i))
                            (* db2 (aget k2 i))
                            (* db3 (aget k3 i))
                            (* db4 (aget k4 i))
                            (* db5 (aget k5 i))))
                sc (+ atol (* rtol (n/max (n/abs (aget u i))
                                          (n/abs (aget utilde i)))))
                r (/ ei sc)]
            (recur (inc i) (+ acc (* r r)))))))))

(deftm perform-step!
  [cache :- SDIRK4Cache,
   f :- (Fn [(Array double) (Array double) Double] Double),
   u :- (Array double), t :- Double, dt :- Double]
  (let [err (sdirk4-step! cache f u t dt)]
    ;; Copy solution to u
    (acopy! (:utilde cache) 0 u 0 (long (:n cache)))
    err))

(deftm solve [alg :- SDIRK4, prob :- ODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt0 (double dt)
        n (alength u0)
        cache (make-cache alg n)
        u (n/similar u0)
        u-save (n/similar u0)
        utilde (:utilde cache)]
    (acopy! u0 0 u 0 n)
    (loop [t t0
           dt-loop dt0
           ts (transient [t0])
           us (transient [(aclone u0)])
           nreject 0
           nf 0]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out
           :us us-out
           :iterations (dec (count ts-out))
           :nreject nreject
           :nf nf})
        (let [dt-actual (n/min dt-loop (- tf t))
              _ (acopy! u 0 u-save 0 n)
              err (double (sdirk4-step! cache f u t dt-actual))]
          (if (<= err 1.0)
            ;; Accept: copy utilde→u
            (let [_ (acopy! utilde 0 u 0 n)
                  t-new (+ t dt-actual)
                  ;; I-controller for order 4 method: exponent = -1/5
                  factor (n/min 5.0
                                (n/max 0.2
                                       (* 0.9 (n/pow err -0.2))))]
              (recur t-new
                     (* dt-actual factor)
                     (conj! ts t-new)
                     (conj! us (aclone u))
                     nreject
                     (+ nf 6)))  ;; ~6 f evaluations per step (5 stages + Jacobian)
            ;; Reject: restore u, shrink dt
            (let [factor (n/max 0.2 (* 0.9 (n/pow err -0.2)))
                  _ (acopy! u-save 0 u 0 n)]
              (recur t
                     (* dt-actual factor)
                     ts us
                     (inc nreject)
                     (+ nf 6)))))))))

;; ================================================================
;; Vern7: Verner 7(6) — 10-stage, 7th-order explicit method
;; J.H. Verner, "Explicit Runge-Kutta pairs with lower stage-order"
;; SIAM J. Numer. Anal., Vol. 50, No. 4, pp. 1890-1909, 2012.
;; Coefficients from Julia's OrdinaryDiffEq.jl (verner_tableaus.jl)
;; ================================================================

(defvalue Vern7 [atol :- Double, rtol :- Double] :implements AdaptiveAlgorithm)

(defn vern7
  "Create a Verner 7(6) adaptive solver. High accuracy for smooth problems."
  ([] (->Vern7 1e-10 1e-8))
  ([atol rtol] (->Vern7 atol rtol)))

(defvalue Vern7Cache (All [T])
  [k1  :- (Array T), k2  :- (Array T), k3  :- (Array T),
   k4  :- (Array T), k5  :- (Array T), k6  :- (Array T),
   k7  :- (Array T), k8  :- (Array T), k9  :- (Array T),
   k10 :- (Array T), tmp :- (Array T), utilde :- (Array T),
   atol :- Double, rtol :- Double]
  :implements AdaptiveCache)

(deftm make-cache [alg :- Vern7, u0] :- Vern7Cache
  (->Vern7Cache (n/similar u0) (n/similar u0) (n/similar u0)
                (n/similar u0) (n/similar u0) (n/similar u0)
                (n/similar u0) (n/similar u0) (n/similar u0)
                (n/similar u0) (n/similar u0) (n/similar u0)
                (:atol alg) (:rtol alg)))

;; Time fractions (c1=0, c9=c10=1 implicit)
(def ^:private ^:const vern7-c2 0.005)
(def ^:private ^:const vern7-c3 0.10888888888888888)
(def ^:private ^:const vern7-c4 0.16333333333333333)
(def ^:private ^:const vern7-c5 0.4555)
(def ^:private ^:const vern7-c6 0.6095094489978381)
(def ^:private ^:const vern7-c7 0.884)
(def ^:private ^:const vern7-c8 0.925)

;; Stage 2: a21*k1
(def ^:private ^:const vern7-a21 0.005)
;; Stage 3: a31*k1 + a32*k2
(def ^:private ^:const vern7-a31 -1.07679012345679)
(def ^:private ^:const vern7-a32  1.185679012345679)
;; Stage 4: a41*k1 + a43*k3
(def ^:private ^:const vern7-a41 0.04083333333333333)
(def ^:private ^:const vern7-a43 0.1225)
;; Stage 5: a51*k1 + a53*k3 + a54*k4
(def ^:private ^:const vern7-a51  0.6389139236255726)
(def ^:private ^:const vern7-a53 -2.455672638223657)
(def ^:private ^:const vern7-a54  2.272258714598084)
;; Stage 6: a61*k1 + a63*k3 + a64*k4 + a65*k5
(def ^:private ^:const vern7-a61 -2.6615773750187572)
(def ^:private ^:const vern7-a63  10.804513886456137)
(def ^:private ^:const vern7-a64 -8.3539146573962)
(def ^:private ^:const vern7-a65  0.820487594956657)
;; Stage 7: a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6
(def ^:private ^:const vern7-a71  6.067741434696772)
(def ^:private ^:const vern7-a73 -24.711273635911088)
(def ^:private ^:const vern7-a74  20.427517930788895)
(def ^:private ^:const vern7-a75 -1.9061579788166472)
(def ^:private ^:const vern7-a76  1.006172249242068)
;; Stage 8: a81*k1 + a83*k3 + a84*k4 + a85*k5 + a86*k6 + a87*k7
(def ^:private ^:const vern7-a81  12.054670076253203)
(def ^:private ^:const vern7-a83 -49.75478495046899)
(def ^:private ^:const vern7-a84  41.142888638604674)
(def ^:private ^:const vern7-a85 -4.461760149974004)
(def ^:private ^:const vern7-a86  2.042334822239175)
(def ^:private ^:const vern7-a87 -0.09834843665406107)
;; Stage 9: a91*k1 + a93*k3 + a94*k4 + a95*k5 + a96*k6 + a97*k7 + a98*k8
(def ^:private ^:const vern7-a91  10.138146522881808)
(def ^:private ^:const vern7-a93 -42.6411360317175)
(def ^:private ^:const vern7-a94  35.76384003992257)
(def ^:private ^:const vern7-a95 -4.3480228403929075)
(def ^:private ^:const vern7-a96  2.0098622683770357)
(def ^:private ^:const vern7-a97  0.3487490460338272)
(def ^:private ^:const vern7-a98 -0.27143900510483127)
;; Stage 10: a101*k1 + a103*k3 + a104*k4 + a105*k5 + a106*k6 + a107*k7
(def ^:private ^:const vern7-a101 -45.030072034298676)
(def ^:private ^:const vern7-a103  187.3272437654589)
(def ^:private ^:const vern7-a104 -154.02882369350186)
(def ^:private ^:const vern7-a105  18.56465306347536)
(def ^:private ^:const vern7-a106 -7.141809679295079)
(def ^:private ^:const vern7-a107  1.3088085781613787)

;; 7th-order solution weights (b2=b3=b10=0)
(def ^:private ^:const vern7-b1  0.04715561848627222)
(def ^:private ^:const vern7-b4  0.25750564298434153)
(def ^:private ^:const vern7-b5  0.26216653977412624)
(def ^:private ^:const vern7-b6  0.15216092656738558)
(def ^:private ^:const vern7-b7  0.4939969170032485)
(def ^:private ^:const vern7-b8 -0.29430311714032503)
(def ^:private ^:const vern7-b9  0.08131747232495111)

;; Error coefficients: btilde = b - bhat (7th minus 6th order)
(def ^:private ^:const vern7-bt1   0.002547011879931045)
(def ^:private ^:const vern7-bt4  -0.00965839487279575)
(def ^:private ^:const vern7-bt5   0.04206470975639691)
(def ^:private ^:const vern7-bt6  -0.0666822437469301)
(def ^:private ^:const vern7-bt7   0.2650097464621281)
(def ^:private ^:const vern7-bt8  -0.29430311714032503)
(def ^:private ^:const vern7-bt9   0.08131747232495111)
(def ^:private ^:const vern7-bt10 -0.02029518466335628)

(deftm vern7-step!
  "One Vern7 step. Computes 10 stages, writes 7th-order solution to utilde.
  Returns RMS error estimate."
  [cache :- Vern7Cache, f :- (Fn [(Array double) (Array double) Double] Double),
   u :- (Array double), t :- Double, dt :- Double]
  :- Double
  (let [t (double t) dt (double dt)
        k1  (:k1 cache) k2  (:k2 cache) k3  (:k3 cache)
        k4  (:k4 cache) k5  (:k5 cache) k6  (:k6 cache)
        k7  (:k7 cache) k8  (:k8 cache) k9  (:k9 cache)
        k10 (:k10 cache) utilde (:utilde cache)
        n (alength u)]
    ;; Stage 1: k1 = f(u, t)
    (f k1 u t)
    ;; Stage 2: u + dt*a21*k1
    (f k2 (muladd (broadcast [u k1] (+ u (* dt vern7-a21 k1)))) (+ t (* vern7-c2 dt)))
    ;; Stage 3: u + dt*(a31*k1 + a32*k2)
    (f k3 (muladd (broadcast [u k1 k2]
                             (+ u (* dt (+ (* vern7-a31 k1) (* vern7-a32 k2)))))) (+ t (* vern7-c3 dt)))
    ;; Stage 4: u + dt*(a41*k1 + a43*k3) [no k2]
    (f k4 (muladd (broadcast [u k1 k3]
                             (+ u (* dt (+ (* vern7-a41 k1) (* vern7-a43 k3)))))) (+ t (* vern7-c4 dt)))
    ;; Stage 5
    (f k5 (muladd (broadcast [u k1 k3 k4]
                             (+ u (* dt (+ (* vern7-a51 k1) (* vern7-a53 k3) (* vern7-a54 k4)))))) (+ t (* vern7-c5 dt)))
    ;; Stage 6
    (f k6 (muladd (broadcast [u k1 k3 k4 k5]
                             (+ u (* dt (+ (* vern7-a61 k1) (* vern7-a63 k3) (* vern7-a64 k4)
                                           (* vern7-a65 k5)))))) (+ t (* vern7-c6 dt)))
    ;; Stage 7
    (f k7 (muladd (broadcast [u k1 k3 k4 k5 k6]
                             (+ u (* dt (+ (* vern7-a71 k1) (* vern7-a73 k3) (* vern7-a74 k4)
                                           (* vern7-a75 k5) (* vern7-a76 k6)))))) (+ t (* vern7-c7 dt)))
    ;; Stage 8
    (f k8 (muladd (broadcast [u k1 k3 k4 k5 k6 k7]
                             (+ u (* dt (+ (* vern7-a81 k1) (* vern7-a83 k3) (* vern7-a84 k4)
                                           (* vern7-a85 k5) (* vern7-a86 k6) (* vern7-a87 k7)))))) (+ t (* vern7-c8 dt)))
    ;; Stage 9 (at c=1)
    (f k9 (muladd (broadcast [u k1 k3 k4 k5 k6 k7 k8]
                             (+ u (* dt (+ (* vern7-a91 k1) (* vern7-a93 k3) (* vern7-a94 k4)
                                           (* vern7-a95 k5) (* vern7-a96 k6) (* vern7-a97 k7)
                                           (* vern7-a98 k8)))))) (+ t dt))
    ;; 7th-order solution -> utilde (uses k1, k4..k9; b2=b3=b10=0)
    (acopy! (muladd (broadcast [u k1 k4 k5 k6 k7 k8 k9]
                               (+ u (* dt (+ (* vern7-b1 k1) (* vern7-b4 k4) (* vern7-b5 k5)
                                             (* vern7-b6 k6) (* vern7-b7 k7) (* vern7-b8 k8)
                                             (* vern7-b9 k9))))))
            0 utilde 0 n)
    ;; Stage 10 (at c=1, different weights; for error estimate only)
    (f k10 (muladd (broadcast [u k1 k3 k4 k5 k6 k7]
                              (+ u (* dt (+ (* vern7-a101 k1) (* vern7-a103 k3) (* vern7-a104 k4)
                                            (* vern7-a105 k5) (* vern7-a106 k6) (* vern7-a107 k7)))))) (+ t dt))
    ;; Error estimate: RMS of btilde-weighted stages
    (let [atol (double (:atol cache))
          rtol (double (:rtol cache))
          err (muladd (reduce! [acc 0.0] [k1 k4 k5 k6 k7 k8 k9 k10 u utilde]
                               (let [ei (* dt (+ (* vern7-bt1 k1) (* vern7-bt4 k4) (* vern7-bt5 k5)
                                                 (* vern7-bt6 k6) (* vern7-bt7 k7) (* vern7-bt8 k8)
                                                 (* vern7-bt9 k9) (* vern7-bt10 k10)))
                                     sc (+ atol (* rtol (n/max (n/abs u) (n/abs utilde))))
                                     r (/ ei sc)]
                                 (+ acc (* r r)))))]
      (n/sqrt (/ err n)))))

(deftm perform-step! [cache :- Vern7Cache,
                      f :- (Fn [(Array double) (Array double) Double] Double),
                      u :- (Array double), t :- Double, dt :- Double]
  (let [t (double t) dt (double dt)
        err (vern7-step! cache f u t dt)
        utilde (:utilde cache)]
    (acopy! utilde 0 u 0 (alength u))
    err))

(deftm solve [alg :- Vern7, prob :- ODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt0 (double dt)
        n (alength u0)
        cache (make-cache alg u0)
        u (n/similar u0)
        u-save (n/similar u0)
        utilde (:utilde cache)]
    (acopy! u0 0 u 0 n)
    (loop [t t0
           dt-loop dt0
           ts (transient [t0])
           us (transient [(aclone u0)])
           nreject 0
           nf 0]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out :us us-out
           :iterations (dec (count ts-out)) :nreject nreject :nf nf})
        (let [dt-actual (n/min dt-loop (- tf t))
              _ (acopy! u 0 u-save 0 n)
              err (double (vern7-step! cache f u t dt-actual))
              nf-new (+ nf 10)]
          (if (<= err 1.0)
            (let [_ (acopy! utilde 0 u 0 n)
                  t-new (+ t dt-actual)
                  ;; I-controller: exponent -1/(p+1) = -1/8 for 7th order
                  factor (n/min 5.0
                                (n/max 0.2
                                       (* 0.9 (n/pow err -0.125))))]
              (recur t-new
                     (* dt-actual factor)
                     (conj! ts t-new)
                     (conj! us (aclone u))
                     nreject
                     nf-new))
            (let [factor (n/max 0.2 (* 0.9 (n/pow err -0.125)))
                  _ (acopy! u-save 0 u 0 n)]
              (recur t
                     (* dt-actual factor)
                     ts us
                     (inc nreject)
                     nf-new))))))))

;; ================================================================
;; PID step-size controller
;; ================================================================

(deftm pid-factor
  "PID step-size controller. Returns factor to multiply dt by.
  err/err1/err2: current, previous, and 2-steps-ago normalized errors.
  Use err1=err2=1.0 for first steps. order: method order."
  [err :- Double, err1 :- Double, err2 :- Double, order :- Long] :- Double
  (let [safety 0.9
        qmin 0.2
        qmax 5.0
        p1 (/ 1.0 (+ (double order) 1.0))
        beta1 (* 0.7 p1)
        beta2 (* 0.4 p1)
        beta3 (* 0.1 p1)
        factor (* safety
                  (n/pow err (- beta1))
                  (n/pow err1 beta2)
                  (n/pow err2 (- beta3)))]
    (n/min qmax (n/max qmin factor))))

;; ================================================================
;; AutoSwitch: automatic stiffness-based algorithm switching
;; Starts with Tsit5 (explicit), switches to Rosenbrock23 (implicit)
;; when stiffness is detected (consecutive rejections), and probes
;; back to explicit after a configurable interval.
;; ================================================================

(defvalue AutoSwitch [atol :- Double, rtol :- Double,
                      max-stiff-rejects :- Long, probe-interval :- Long]
  :implements ODEAlgorithm)

(deftm auto-switch
  "Create an AutoSwitch solver (0-arg default)."
  [] :- AutoSwitch (->AutoSwitch 1e-8 1e-6 10 20))

(deftm auto-switch
  "Create an AutoSwitch solver with tolerances."
  [atol :- Double, rtol :- Double] :- AutoSwitch
  (->AutoSwitch atol rtol 10 20))

(deftm auto-switch
  "Create an AutoSwitch solver with all parameters."
  [atol :- Double, rtol :- Double, max-rejects :- Long, probe :- Long] :- AutoSwitch
  (->AutoSwitch atol rtol max-rejects probe))

(deftm solve [alg :- AutoSwitch, prob :- ODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        n (alength u0)
        atol (:atol alg)
        rtol (:rtol alg)
        max-stiff (:max-stiff-rejects alg)
        probe-int (:probe-interval alg)
        ;; Create both caches
        tsit-cache (make-cache (tsit5 atol rtol) u0)
        ros-cache (make-cache (rosenbrock23 atol rtol) n)  ;; implicit solver: stays with n (needs Jacobian matrix)
        ;; State arrays
        u (n/similar u0)
        u-save (n/similar u0)
        ;; Tsit5 arrays
        tk1 (:k1 tsit-cache)
        tk7 (:k7 tsit-cache)
        t-utilde (:utilde tsit-cache)
        ;; Rosenbrock23 arrays
        rf0 (:f0 ros-cache)
        r-utilde (:utilde ros-cache)
        r-work (:work ros-cache)
        ;; PI controller gains (Tsit5, order 5)
        beta1 (/ 7.0 50.0)
        beta2 (/ 2.0 25.0)
        safety 0.9]
    (acopy! u0 0 u 0 n)
    ;; Init Tsit5 FSAL
    (f tk1 u t0)
    (loop [t t0
           dt-loop dt
           mode :explicit
           err-prev 1.0
           consec-rej 0
           impl-steps 0
           ts (transient [t0])
           us (transient [(aclone u0)])
           nreject 0
           nswitch 0
           nf 1]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out :us us-out
           :iterations (dec (count ts-out)) :nreject nreject
           :nswitch nswitch :nf nf :mode mode})
        (let [dt-actual (n/min dt-loop (- tf t))]
          (if (= mode :explicit)
            ;; === Explicit (Tsit5) ===
            (let [_ (acopy! u 0 u-save 0 n)
                  err (tsit5-step! tsit-cache f u t dt-actual)
                  nf-new (+ nf 6)]
              (if (<= err 1.0)
                ;; Accept
                (let [_ (acopy! t-utilde 0 u 0 n)
                      _ (acopy! tk7 0 tk1 0 n)
                      t-new (+ t dt-actual)
                      factor (n/min 5.0
                                    (n/max 0.2
                                           (* safety
                                              (n/pow err (- beta1))
                                              (n/pow err-prev beta2))))]
                  (recur t-new (* dt-actual factor) :explicit
                         err 0 0
                         (conj! ts t-new) (conj! us (aclone u))
                         nreject nswitch nf-new))
                ;; Reject
                (let [factor (n/max 0.2 (* safety (n/pow err (- beta1))))
                      _ (acopy! u-save 0 u 0 n)
                      new-rej (inc consec-rej)]
                  (if (>= new-rej max-stiff)
                    ;; Switch to implicit
                    (do (f rf0 u t) ;; init Rosenbrock23 FSAL
                        (recur t (* dt-actual factor) :implicit
                               err-prev 0 0
                               ts us (inc nreject) (inc nswitch) (inc nf-new)))
                    ;; Stay explicit
                    (recur t (* dt-actual factor) :explicit
                           err-prev new-rej 0
                           ts us (inc nreject) nswitch nf-new)))))
            ;; === Implicit (Rosenbrock23) ===
            (let [err (rosenbrock23-step! ros-cache f u t dt-actual)
                  nf-new (+ nf 3)]
              (if (<= err 1.0)
                ;; Accept
                (let [_ (acopy! r-utilde 0 u 0 n)
                      _ (acopy! r-work 0 rf0 0 n)
                      t-new (+ t dt-actual)
                      factor (n/min 5.0
                                    (n/max 0.2
                                           (* 0.9 (n/pow err (/ -1.0 3.0)))))
                      new-impl (inc impl-steps)]
                  (if (>= new-impl probe-int)
                    ;; Probe: switch back to explicit
                    (do (f tk1 u t-new)
                        (recur t-new (* dt-actual factor) :explicit
                               err 0 0
                               (conj! ts t-new) (conj! us (aclone u))
                               nreject (inc nswitch) (inc nf-new)))
                    ;; Stay implicit
                    (recur t-new (* dt-actual factor) :implicit
                           err 0 new-impl
                           (conj! ts t-new) (conj! us (aclone u))
                           nreject nswitch nf-new)))
                ;; Reject
                (recur t
                       (* dt-actual (n/max 0.2 (* 0.9 (n/pow err (/ -1.0 3.0)))))
                       :implicit err-prev 0 0
                       ts us (inc nreject) nswitch nf-new)))))))))

;; ================================================================
;; Event handling (zero-crossing detection)
;; ================================================================

(defrecord ODEEvent
           [condition   ;; (fn [^doubles u ^double t]) -> double; event at zero-crossing
            affect!     ;; (fn [^doubles u ^double t]) -> :continue or :terminate
            direction]) ;; :rising, :falling, or :both

(defn ode-event
  "Create an ODE event.
  condition: (fn [u t]) -> double — event triggers on zero-crossing.
  affect!:   (fn [u t]) -> :continue or :terminate — modify state or stop.
  direction: :rising (- to +), :falling (+ to -), or :both (default)."
  ([condition affect!] (->ODEEvent condition affect! :both))
  ([condition affect! direction] (->ODEEvent condition affect! direction)))

(deftm sign-change?
  "Check if g0 -> g1 represents a zero-crossing in the given direction."
  [g0 :- Double, g1 :- Double, direction :- Object]
  (case direction
    :rising  (and (neg? g0) (pos? g1))
    :falling (and (pos? g0) (neg? g1))
    :both    (not= (pos? g0) (pos? g1))))

(deftm find-event-root
  "Bisection to find t* where condition(u(t*)) ≈ 0.
  Integrates from t0 with state u0, using f and cache.
  Returns [t-event u-event]."
  [f :- Object, u0 :- (Array double), t0 :- Double, t1 :- Double,
   condition :- Object, cache :- Object, max-iter :- Long]
  (let [t0 (double t0) t1 (double t1) max-iter (long max-iter)
        n (alength u0)
        u-lo (n/similar u0)
        u-hi (n/similar u0)
        u-mid (n/similar u0)]
    (acopy! u0 0 u-lo 0 n)
    ;; u-hi = integrate from t0 to t1 (full step already done, u0 is at t0)
    ;; We need to re-integrate to get u at intermediate points
    ;; Simpler: bisect on time, re-integrate from u0 each time
    (loop [lo t0 hi t1 iter 0]
      (if (or (>= iter max-iter) (< (- hi lo) 1e-12))
        (let [t-mid (* 0.5 (+ lo hi))]
          ;; Integrate from u0 at t0 to t-mid
          (acopy! u0 0 u-mid 0 n)
          (perform-step! cache f u-mid t0 (- t-mid t0))
          [t-mid u-mid])
        (let [t-mid (* 0.5 (+ lo hi))]
          ;; Integrate from u0 at t0 to t-mid
          (acopy! u0 0 u-mid 0 n)
          (perform-step! cache f u-mid t0 (- t-mid t0))
          (let [g-mid (double (condition u-mid t-mid))
                ;; g at lo
                _ (acopy! u0 0 u-lo 0 n)
                g-lo (double (condition u-lo t0))]
            (if (sign-change? g-lo g-mid :both)
              (recur lo t-mid (inc iter))
              (recur t-mid hi (inc iter)))))))))

(defn solve-with-events
  "Solve an ODE problem with event detection.
  alg: solver algorithm, prob: ODE problem, dt: step size,
  events: seq of ODEEvent records.
  Returns solution map with :event-log [{:t :event-idx :action} ...]."
  [alg prob dt events]
  (let [{:keys [f u0 tspan]} prob
        [t0 tf] tspan
        t0 (double t0) tf (double tf) dt (double dt)
        n (alength u0)
        u (n/similar u0)
        u-prev (n/similar u0)
        cache (make-cache alg u0)
        events (vec events)]
    (acopy! u0 0 u 0 n)
    (loop [t t0
           ts (transient [t0])
           us (transient [(aclone u0)])
           event-log (transient [])]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out
           :us us-out
           :iterations (dec (count ts-out))
           :event-log (persistent! event-log)})
        (let [dt-actual (n/min dt (- tf t))
              ;; Save state before step
              _ (acopy! u 0 u-prev 0 n)
              ;; Evaluate conditions before step
              g-before (mapv (fn [^ODEEvent ev] (double ((.condition ev) u t))) events)
              ;; Perform step
              _ (perform-step! cache f u t dt-actual)
              t-new (+ t dt-actual)
              ;; Evaluate conditions after step
              g-after (mapv (fn [^ODEEvent ev] (double ((.condition ev) u t-new))) events)
              ;; Check for events
              [t-final u-final event-log terminated?]
              (loop [ei 0 t-cur t-new event-log event-log terminated? false]
                (if (or terminated? (>= ei (count events)))
                  [t-cur u event-log terminated?]
                  (let [^ODEEvent ev (nth events ei)
                        g0 (double (nth g-before ei))
                        g1 (double (nth g-after ei))]
                    (if (and (not (zero? g0)) (sign-change? g0 g1 (.direction ev)))
                      ;; Event detected — find root
                      (let [[t-event u-event] (find-event-root f u-prev t t-new
                                                               (.condition ev) cache 50)]
                        ;; Copy event state into u
                        (acopy! u-event 0 u 0 n)
                        ;; Call affect!
                        (let [action ((.affect! ev) u t-event)]
                          (recur (inc ei) t-event
                                 (conj! event-log {:t t-event :event-idx ei :action action})
                                 (= action :terminate))))
                      (recur (inc ei) t-cur event-log terminated?)))))]
          (if terminated?
            (let [ts-out (persistent! (conj! ts t-final))
                  us-out (persistent! (conj! us (aclone u)))]
              {:ts ts-out
               :us us-out
               :iterations (dec (count ts-out))
               :event-log (persistent! event-log)})
            (recur t-final
                   (conj! ts t-final)
                   (conj! us (aclone u))
                   event-log)))))))

;; ================================================================
;; Dense output / interpolation
;; ================================================================

(deftm hermite-cubic!
  "Hermite cubic interpolation: write u(t0 + θ*h) into out.
  Uses the standard basis: H(θ) = h00*u0 + h10*f0 + h01*u1 + h11*f1
  where h00..h11 are Hermite basis polynomials in θ = (t - t0)/h."
  [out :- (Array double), u0 :- (Array double), u1 :- (Array double), f0 :- (Array double), f1 :- (Array double), h :- Double, theta :- Double]
  (let [h (double h) theta (double theta)
        n (alength u0)
        theta2 (* theta theta)
        theta3 (* theta2 theta)
        ;; Hermite basis functions
        h00 (+ 1.0 (* -3.0 theta2) (* 2.0 theta3))    ;; 2θ³ - 3θ² + 1
        h10 (* h (+ theta (* -2.0 theta2) theta3))      ;; h*(θ³ - 2θ² + θ)
        h01 (+ (* 3.0 theta2) (* -2.0 theta3))          ;; -2θ³ + 3θ²
        h11 (* h (+ (* -1.0 theta2) theta3))]            ;; h*(θ³ - θ²)
    (dotimes [i n]
      (aset out i (+ (* h00 (aget u0 i))
                     (* h10 (aget f0 i))
                     (* h01 (aget u1 i))
                     (* h11 (aget f1 i)))))
    out))

(defn- binary-search-interval
  "Find index i such that ts[i] <= t < ts[i+1] using binary search."
  [ts ^double t]
  (loop [lo (int 0) hi (int (- (count ts) 2))]
    (if (<= (- hi lo) 1)
      (if (<= (double (nth ts hi)) t)
        hi
        lo)
      (let [mid (unchecked-add-int lo (bit-shift-right (unchecked-subtract-int hi lo) 1))]
        (if (<= (double (nth ts mid)) t)
          (recur mid hi)
          (recur lo mid))))))

(deftm tsit5-interp!
  "Native Tsit5 4th-order interpolation at θ ∈ [0,1].
  Writes u(t_i + θ*h) into out.
  ks is a vector of 7 double-arrays [k1 k2 k3 k4 k5 k6 k7]."
  [out :- (Array double), u0 :- (Array double), ks, h :- Double, theta :- Double]
  (let [h (double h) theta (double theta)
        n (alength u0)
        ^doubles k1 (nth ks 0) ^doubles k2 (nth ks 1) ^doubles k3 (nth ks 2)
        ^doubles k4 (nth ks 3) ^doubles k5 (nth ks 4) ^doubles k6 (nth ks 5)
        ^doubles k7 (nth ks 6)
        th2 (* theta theta)
        ;; b1(θ) = θ*(r11 + θ*(r12 + θ*(r13 + θ*r14)))
        b1 (* theta (+ tsit5-r11 (* theta (+ tsit5-r12 (* theta (+ tsit5-r13 (* theta tsit5-r14)))))))
        ;; b_i(θ) = θ²*(r_i2 + θ*(r_i3 + θ*r_i4))
        b2 (* th2 (+ tsit5-r22 (* theta (+ tsit5-r23 (* theta tsit5-r24)))))
        b3 (* th2 (+ tsit5-r32 (* theta (+ tsit5-r33 (* theta tsit5-r34)))))
        b4 (* th2 (+ tsit5-r42 (* theta (+ tsit5-r43 (* theta tsit5-r44)))))
        b5 (* th2 (+ tsit5-r52 (* theta (+ tsit5-r53 (* theta tsit5-r54)))))
        b6 (* th2 (+ tsit5-r62 (* theta (+ tsit5-r63 (* theta tsit5-r64)))))
        b7 (* th2 (+ tsit5-r72 (* theta (+ tsit5-r73 (* theta tsit5-r74)))))]
    (dotimes [i n]
      (aset out i (+ (aget u0 i)
                     (* h (+ (* b1 (aget k1 i)) (* b2 (aget k2 i)) (* b3 (aget k3 i))
                             (* b4 (aget k4 i)) (* b5 (aget k5 i)) (* b6 (aget k6 i))
                             (* b7 (aget k7 i)))))))
    out))

(defn dense-output
  "Create a continuous interpolant from a discrete ODE solution.
   f: the ODE right-hand side (fn [du u t])
   sol: solution map from solve (contains :ts, :us, optionally :ks)
   Returns a function (fn [t] -> double-array).

   When :ks (stage derivatives) are present (Tsit5), uses native 4th-order
   polynomial interpolation. Otherwise falls back to Hermite cubic.
   The returned function clones its output, so each call returns a fresh array."
  [f sol]
  (let [ts (:ts sol)
        us (:us sol)
        n (alength (first us))
        buf (n/similar (first us))]
    (if-let [ks (:ks sol)]
      ;; Native Tsit5 interpolation using stored stages
      (ftm [t :- Double] :- (Array double)
           (let [t0 (double (nth ts 0))
                 tf (double (peek ts))]
             (cond
               (<= t t0) (aclone (first us))
               (>= t tf) (aclone (peek us))
               :else
               (let [i (binary-search-interval ts t)
                     ti  (double (nth ts i))
                     ti1 (double (nth ts (inc i)))
                     h   (- ti1 ti)
                     theta (/ (- t ti) h)]
                 (tsit5-interp! buf ^doubles (nth us i) (nth ks i) h theta)
                 (aclone buf)))))
      ;; Hermite cubic fallback
      (let [fs (mapv (fn [u t]
                       (let [du (n/similar u)]
                         (f du u (double t))
                         du))
                     us ts)]
        (ftm [t :- Double] :- (Array double)
             (let [t0 (double (nth ts 0))
                   tf (double (peek ts))]
               (cond
                 (<= t t0) (aclone (first us))
                 (>= t tf) (aclone (peek us))
                 :else
                 (let [i (binary-search-interval ts t)
                       ti  (double (nth ts i))
                       ti1 (double (nth ts (inc i)))
                       h   (- ti1 ti)
                       theta (/ (- t ti) h)]
                   (hermite-cubic! buf
                                   ^doubles (nth us i) ^doubles (nth us (inc i))
                                   ^doubles (nth fs i) ^doubles (nth fs (inc i))
                                   h theta)
                   (aclone buf)))))))))

(defn saveat
  "Evaluate a continuous interpolant at specified time points.
   interp: interpolation function from dense-output
   ts-out: collection of time points to evaluate at
   Returns {:ts ts-out, :us us-out}."
  [interp ts-out]
  (let [ts-out (vec ts-out)
        us-out (mapv (fn [t] (interp (double t))) ts-out)]
    {:ts ts-out :us us-out}))

;; ================================================================
;; Example problems
;; ================================================================

(deftm exponential-decay
  "du/dt = -u. Exact solution: u(t) = u0 * exp(-t)."
  [du :- (Array double), u :- (Array double), t :- Double]
  (let [n (alength u)]
    (dotimes [i n]
      (aset du i (- (aget u i))))))

(deftm lorenz!
  "Lorenz attractor: sigma=10, rho=28, beta=8/3."
  [du :- (Array double), u :- (Array double), t :- Double]
  (let [x (aget u 0) y (aget u 1) z (aget u 2)
        sigma 10.0 rho 28.0 beta (/ 8.0 3.0)]
    (aset du 0 (* sigma (- y x)))
    (aset du 1 (- (* x (- rho z)) y))
    (aset du 2 (- (* x y) (* beta z)))))

(deftm harmonic-oscillator!
  "Simple harmonic oscillator: u = [x, v], du/dt = [v, -x]."
  [du :- (Array double), u :- (Array double), t :- Double]
  (aset du 0 (aget u 1))
  (aset du 1 (- (aget u 0))))

(deftm bouncing-ball!
  "Bouncing ball under gravity: u = [height, velocity], du/dt = [v, -9.81]."
  [du :- (Array double), u :- (Array double), t :- Double]
  (aset du 0 (aget u 1))
  (aset du 1 -9.81))

(deftm stiff-decay!
  "Stiff decay: du/dt = -1000*u. Exact: u(t) = u0 * exp(-1000*t).
  Eigenvalue = -1000, so explicit Euler needs dt < 0.002 for stability."
  [du :- (Array double), u :- (Array double), t :- Double]
  (let [n (alength u)]
    (dotimes [i n]
      (aset du i (* -1000.0 (aget u i))))))

(deftm van-der-pol!
  "Van der Pol oscillator: u = [x, v], du/dt = [v, mu*(1-x^2)*v - x].
  Becomes stiff for large mu. Standard test problem from Hairer & Wanner."
  [du :- (Array double), u :- (Array double), t :- Double, mu :- Double]
  (let [x (aget u 0)
        v (aget u 1)]
    (aset du 0 v)
    (aset du 1 (- (* mu (- 1.0 (* x x)) v) x))))

(deftm van-der-pol!
  [du :- (Array double), u :- (Array double), t :- Double]
  (van-der-pol! du u t 1000.0))

;; ================================================================
;; Generic out-of-place solvers
;; Work with any state type that supports raster.numeric/+, raster.numeric/*
;; f: (fn [u t]) -> du/dt  (returns derivative, out-of-place)
;; ================================================================

(defrecord GenericODEProblem [f u0 tspan opts])

(defn generic-ode-problem
  "Create an ODE problem with generic state type (out-of-place).
   f: (fn [u t]) -> du/dt (returns derivative).
   u0: initial state (any type with raster.numeric/+, * defined)."
  ([f u0 t0 tf] (->GenericODEProblem f u0 [t0 tf] {}))
  ([f u0 t0 tf opts] (->GenericODEProblem f u0 [t0 tf] opts)))

(deftm solve [alg :- Euler, prob :- GenericODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt (double dt)]
    (loop [t t0
           u u0
           ts (transient [t0])
           us (transient [u0])]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out :us us-out :iterations (dec (count ts-out))})
        (let [dt-actual (n/min dt (- tf t))
              k (f u t)
              u-new (n/+ u (n/* dt-actual k))
              t-new (+ t dt-actual)]
          (recur t-new u-new
                 (conj! ts t-new)
                 (conj! us u-new)))))))

(deftm solve [alg :- RK4, prob :- GenericODEProblem, dt :- Double]
  (let [f (:f prob)
        u0 (:u0 prob)
        tspan (:tspan prob)
        t0 (double (nth tspan 0))
        tf (double (nth tspan 1))
        dt (double dt)]
    (loop [t t0
           u u0
           ts (transient [t0])
           us (transient [u0])]
      (if (>= t tf)
        (let [ts-out (persistent! ts)
              us-out (persistent! us)]
          {:ts ts-out :us us-out :iterations (dec (count ts-out))})
        (let [dt-actual (n/min dt (- tf t))
              half-dt (* 0.5 dt-actual)
              k1 (f u t)
              k2 (f (n/+ u (n/* half-dt k1)) (+ t half-dt))
              k3 (f (n/+ u (n/* half-dt k2)) (+ t half-dt))
              k4 (f (n/+ u (n/* dt-actual k3)) (+ t dt-actual))
              u-new (n/+ u (n/* (/ dt-actual 6.0)
                                (n/+ (n/+ k1 (n/* 2.0 k2))
                                     (n/+ (n/* 2.0 k3) k4))))
              t-new (+ t dt-actual)]
          (recur t-new u-new
                 (conj! ts t-new)
                 (conj! us u-new)))))))

;; Aliases for backward compatibility — the deftm versions above
;; implement IFn__ and work as typed function parameters directly.
(def ^{:doc "Alias for exponential-decay (backward compatibility)."} exponential-decay-ftm exponential-decay)
(def ^{:doc "Alias for lorenz! (backward compatibility)."} lorenz-ftm lorenz!)
