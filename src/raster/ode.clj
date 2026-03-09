(ns raster.ode
  "Differential equations — public entry point.

  Provides ODE solvers (Euler, RK4, DP5, Tsit5, implicit methods),
  PDE method-of-lines, and stochastic DEs.

  Usage:
    (require '[raster.ode :as ode])
    (def prob (ode/ode-problem lorenz! u0 [0.0 100.0]))
    (def sol  (ode/solve (ode/tsit5) prob))

  Sub-namespaces for advanced use:
    raster.ode.core        — ODE problem setup, solvers, events, dense output
    raster.ode.pde         — method-of-lines PDE solvers
    raster.ode.sde         — stochastic DEs (Euler-Maruyama)
    raster.ode.sensitivity — sensitivity analysis via forward-mode AD"
  (:require [raster.ode.core :as core]
            [raster.ode.pde :as pde]
            [raster.ode.sde :as sde]
            [raster.support :refer [import-vars]]))

;; ================================================================
;; ODE problem setup and algorithms
;; ================================================================

(import-vars raster.ode.core
             ode-problem
             dp5 tsit5 implicit-euler rosenbrock23 sdirk4 vern7
             solve
             ode-event solve-with-events
             dense-output saveat
             auto-switch)

;; ================================================================
;; PDE solvers
;; ================================================================

(import-vars raster.ode.pde
             heat-1d-solve heat-2d-solve wave-1d-solve)

;; ================================================================
;; Stochastic DEs
;; ================================================================

(import-vars raster.ode.sde
             sde-problem)
