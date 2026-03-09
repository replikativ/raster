(ns raster.ode.pde
  "PDE solvers via method-of-lines (spatial discretization -> ODE in time).

  Converts PDEs to systems of ODEs by discretizing spatial derivatives
  with finite-difference stencils, then solves with raster.ode solvers.

  Supported equations:
    - 1D heat equation: du/dt = alpha d^2u/dx^2
    - 2D heat equation: du/dt = alpha (d^2u/dx^2 + d^2u/dy^2)
    - 1D wave equation: d^2u/dt^2 = c^2 d^2u/dx^2"
  (:refer-clojure :exclude [aget aset alength aclone reduce])
  (:require [raster.core :refer [deftm ftm broadcast reduce! stencil!]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.par :as par]
            [raster.numeric :as n]
            [raster.ode.core :as ode]))

;; ================================================================
;; 1D Heat Equation
;; ================================================================

(deftm heat-rhs-1d! (All [T]
                         [du :- (Array T), u :- (Array T),
                          alpha :- Double, inv-dx2 :- Double]
                         (stencil! [du u] :radius 1 :boundary :dirichlet
                                   (* alpha (* inv-dx2 (+ (aget u (- i 1))
                                                          (* -2.0 (aget u i))
                                                          (aget u (+ i 1))))))))

(deftm solve-fixed-step
  "Simple fixed-step ODE solver loop. Returns {:u final-state :iterations N}.
  f: (fn [du u t]) writes into du in-place."
  [f :- (Fn [(Array double) (Array double) Double] Double), u0 :- (Array double), t0 :- Double, tf :- Double, dt :- Double, cache]
  (let [t0 (double t0) tf (double tf) dt (double dt)
        n (int (alength u0))
        u (aclone u0)]
    (loop [t t0 nsteps 0]
      (if (>= t tf)
        {:u u :iterations nsteps}
        (let [dt-actual (n/min dt (- tf t))]
          (ode/perform-step! cache f u t dt-actual)
          (recur (+ t dt-actual) (inc nsteps)))))))

(defn heat-1d-solve
  "Solve 1D heat equation via method-of-lines.
  u0: initial condition (double-array, includes boundary values)
  alpha: thermal diffusivity
  dx: spatial step
  tspan: [t0 tf]
  alg: ODE algorithm (e.g. (ode/->RK4))
  dt: time step

  Returns {:u final-state :iterations step-count}."
  [u0 alpha dx tspan alg dt]
  (let [inv-dx2 (/ 1.0 (* dx dx))
        n (int (alength u0))
        f (ftm [du :- (Array double), u :- (Array double), t :- Double]
               (heat-rhs-1d! du u alpha inv-dx2))
        cache (ode/make-cache alg u0)]
    (solve-fixed-step f u0 (first tspan) (second tspan) dt cache)))

;; ================================================================
;; 2D Heat Equation
;; ================================================================

(deftm heat-rhs-2d!
  "Compute RHS of 2D heat equation: du/dt = alpha * (d^2u/dx^2 + d^2u/dy^2).
  5-point stencil on row-major flat array with Dirichlet boundary conditions."
  (All [T] [du :- (Array T), u :- (Array T),
            nx :- Long, ny :- Long,
            alpha :- Double, inv-dx2 :- Double, inv-dy2 :- Double]
       (let [nx (int nx)
             ny (int ny)]
    ;; Zero all boundaries first
         (dotimes [i nx]
           (aset du (+ (* i ny) 0) 0.0)
           (aset du (+ (* i ny) (- ny 1)) 0.0))
         (dotimes [j ny]
           (aset du j 0.0)
           (aset du (+ (* (- nx 1) ny) j) 0.0))
    ;; Interior points — nested dotimes with row-major aset
    ;; loop-lift will flatten this nested row-major dotimes into 1D par/map!
         (dotimes [ii (- nx 2)]
           (let [i (int (+ ii 1))]
             (dotimes [jj (- ny 2)]
               (let [j (int (+ jj 1))
                     idx (int (+ (* i ny) j))
                     uij (aget u idx)
                     laplacian (+ (* inv-dx2 (+ (aget u (- idx ny)) (* -2.0 uij) (aget u (+ idx ny))))
                                  (* inv-dy2 (+ (aget u (- idx 1)) (* -2.0 uij) (aget u (+ idx 1)))))]
                 (aset du idx (* alpha laplacian))))))
         du)))

(defn heat-2d-solve
  "Solve 2D heat equation via method-of-lines.
  u0: initial condition (flat row-major double-array, nx*ny)
  nx, ny: grid dimensions
  alpha: thermal diffusivity
  dx, dy: spatial steps
  tspan: [t0 tf]
  alg: ODE algorithm
  dt: time step

  Returns {:u final-state :iterations step-count}."
  [u0 nx ny alpha dx dy tspan alg dt]
  (let [inv-dx2 (/ 1.0 (* dx dx))
        inv-dy2 (/ 1.0 (* dy dy))
        n (int (* nx ny))
        f (ftm [du :- (Array double), u :- (Array double), t :- Double]
               (heat-rhs-2d! du u nx ny alpha inv-dx2 inv-dy2))
        cache (ode/make-cache alg u0)]
    (solve-fixed-step f u0 (first tspan) (second tspan) dt cache)))

;; ================================================================
;; 1D Wave Equation
;; ================================================================

(deftm wave-rhs-1d!
  "Compute RHS of 1D wave equation: d^2u/dt^2 = c^2 * d^2u/dx^2.
  State vector is [u0..u_{n-1} v0..v_{n-1}] where u is displacement, v is velocity.
  du/dt = v, dv/dt = c^2 * d^2u/dx^2. Fixed-endpoint Dirichlet boundaries."
  (All [T] [duv :- (Array T), uv :- (Array T),
            n-pts :- Long, c2-inv-dx2 :- Double]
       (let [n (int n-pts)]
    ;; du/dt = v (copy velocity to displacement derivative)
         (par/map! duv i n nil (aget uv (+ n i)))
    ;; dv/dt = c^2 * d^2u/dx^2 with Dirichlet boundaries
    ;; Boundaries: fixed endpoints
         (aset duv n 0.0)
         (aset duv (+ n (- n 1)) 0.0)
    ;; Interior: dv[i]/dt = c^2 * (u[i-1] - 2u[i] + u[i+1]) / dx^2
         (dotimes [j (- n 2)]
           (let [i (int (+ j 1))]
             (aset duv (+ n i) (* c2-inv-dx2
                                  (+ (aget uv (- i 1))
                                     (* -2.0 (aget uv i))
                                     (aget uv (+ i 1)))))))
         duv)))

(defn wave-1d-solve
  "Solve 1D wave equation via method-of-lines.
  u0: initial displacement (double-array, includes boundary values)
  v0: initial velocity (double-array, same size)
  c: wave speed
  dx: spatial step
  tspan: [t0 tf]
  alg: ODE algorithm
  dt: time step

  Returns {:u final-state :iterations step-count}.
  The state vector is [u | v] where u is displacement and v is velocity."
  [u0 v0 c dx tspan alg dt]
  (let [n (int (alength u0))
        c2-inv-dx2 (/ (* c c) (* dx dx))
        state0 (double-array (* 2 n))]
    (acopy! u0 0 state0 0 n)
    (acopy! v0 0 state0 n n)
    (let [f (ftm [duv :- (Array double), uv :- (Array double), t :- Double]
                 (wave-rhs-1d! duv uv n c2-inv-dx2))
          cache (ode/make-cache alg state0)]
      (solve-fixed-step f state0 (first tspan) (second tspan) dt cache))))

;; ================================================================
;; Compiled Heat Equation Loss (for reverse-mode AD pipeline)
;; ================================================================
;; Self-contained deftm that inlines RK4 time-stepping with heat RHS.
;; Compiles to optimized forward evaluation via:
;;   (pipeline/compile-aot #'raster.ode.pde/heat-loss-rk4)
;;
;; NOTE: Reverse-mode AD through the time-stepping loop is not yet supported
;; by gen-reverse-dotimes (requires loop-carried adjoint support). Use
;; raster.ode.sensitivity/estimate-heat-alpha for forward-mode AD gradient.
;;
;; Design: no function pointers (f param), no closures — everything
;; is explicit deftm calls that the walker can inline.

(deftm heat-loss-rk4
  "MSE loss of 1D heat equation integrated with RK4 vs target state.
  Fully self-contained — inlines RK4 + heat RHS for pipeline compilation.
  Uses broadcast/reduce! typed macros for GPU/SIMD-ready inner loops.

  Parameters:
    u0      — initial temperature field (double-array, n points)
    target  — observed final temperature field (double-array, n points)
    alpha   — thermal diffusivity (the parameter to optimize)
    inv-dx2 — 1/dx^2 (precomputed grid constant)
    dt      — time step
    nsteps  — number of RK4 steps

  Returns: MSE loss (scalar double).

  Usage with pipeline:
    (pipeline/compile-aot #'raster.ode.pde/heat-loss-rk4)"
  (All [T] [u0 :- (Array T), target :- (Array T),
            alpha :- Double, inv-dx2 :- Double,
            dt :- Double, nsteps :- Long]
       (let [n (int (alength u0))
        ;; Working state — clone u0 so we don't mutate the input
             u (aclone u0)
        ;; RK4 scratch arrays
             k1 (n/similar u0)
             k2 (n/similar u0)
             k3 (n/similar u0)
             k4 (n/similar u0)
             half-dt (* 0.5 dt)
             dt6 (/ dt 6.0)
             nsteps (int nsteps)]
    ;; Time-stepping loop (sequential — each step depends on previous)
         (dotimes [_step nsteps]
      ;; k1 = f(u)
           (heat-rhs-1d! k1 u alpha inv-dx2)
      ;; tmp = u + 0.5*dt*k1
           (let [tmp (broadcast [u k1] (+ u (* half-dt k1)))]
        ;; k2 = f(tmp)
             (heat-rhs-1d! k2 tmp alpha inv-dx2))
      ;; tmp = u + 0.5*dt*k2
           (let [tmp (broadcast [u k2] (+ u (* half-dt k2)))]
        ;; k3 = f(tmp)
             (heat-rhs-1d! k3 tmp alpha inv-dx2))
      ;; tmp = u + dt*k3
           (let [tmp (broadcast [u k3] (+ u (* dt k3)))]
        ;; k4 = f(tmp)
             (heat-rhs-1d! k4 tmp alpha inv-dx2))
      ;; u = u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
           (acopy! (broadcast [u k1 k2 k3 k4]
                              (+ u (* dt6 (+ k1 (* 2.0 k2) (* 2.0 k3) k4))))
                   0 u 0 n))
    ;; MSE loss: (1/n) * sum((u[i] - target[i])^2)
         (let [inv-n (/ 1.0 (double n))
               sum-sq (reduce! [acc 0.0] [u target]
                               (let [diff (- u target)]
                                 (+ acc (* diff diff))))]
           (* sum-sq inv-n)))))
