# Automatic Differentiation

Raster provides both forward-mode and reverse-mode AD, integrated with the
compiler pipeline so gradients compile to the same optimized code as forward
passes.

## Forward Mode

Forward-mode AD uses Dual numbers — values paired with their derivatives.
Raster's `Dual` is a parametric value type `(All [T])` that works with
any numeric type, including Float32 for GPU workloads.

```clojure
(require '[raster.ad :refer [value+grad grad]])

(deftm f [x :- Double] :- Double
  (* x (sin x)))

;; Evaluate and differentiate in one pass
(value+grad f 1.0)  ;; => [value, gradient]
```

Forward mode is efficient for functions with few inputs (gradients, Jacobian
columns). It composes naturally through ODE solvers for sensitivity analysis.

## Reverse Mode

Reverse-mode AD operates at the IR level: the compiler transforms the function
body into primal + pullback closures. No runtime tape is allocated.

```clojure
;; Gradient of a scalar function (reverse mode, efficient for many inputs)
(grad rosenbrock [1.0 1.0])
```

### rrules

Custom reverse-mode rules are registered alongside `deftm` definitions via
`raster.ad.rrule`. The compiler checks for explicit AD templates to decide
whether a call should stay symbolic (for the AD transform) or be inlined.

## Composable Operators

`value+grad` and `grad` are first-class functions that carry `deftm` metadata.
The compiler can inline through them — a training step that combines forward
pass, AD, and optimizer update compiles into a single fused method via
`compile-aot`.

## Sensitivity Analysis

Forward-mode AD through ODE integration enables gradient-based parameter
estimation for dynamical systems. Dual numbers propagate through the ODE
solver's step function, giving exact derivatives of the solution with respect
to parameters.

```clojure
(require '[raster.ode :as ode])

;; Solve with Dual parameters — derivatives flow through the solver
(ode/solve (ode/rk4 0.01)
           (ode/ode-problem rhs-with-duals u0 0.0 10.0) 0.01)
```

This is handled through the standard `GenericODEProblem` API in `raster.ode.core`.
