# Documentation

Detailed documentation for Raster's subsystems. For a quick overview, see
the [main README](../README.md).

## Guides

| Document | Description |
|----------|-------------|
| [Compiler Pipeline](compiler.md) | Nanopass architecture, pass sequence, `compile-aot`, diagnostics |
| [Automatic Differentiation](autodiff.md) | Forward-mode, reverse-mode, rrules, sensitivity analysis |
| [GPU Computing](gpu.md) | Parallel primitives, session API, backends, SoA layout |
| [Deep Learning](deep-learning.md) | Layers, loss, optimizers, compiled training, einsum |

## Interactive Notebooks

The [notebooks/raster/](../notebooks/raster/) directory contains Kindly/Clay
notebooks that walk through features with runnable code:

| Notebook | Topics |
|----------|--------|
| [Getting Started](../notebooks/raster/getting_started.clj) | `deftm`, typed dispatch, value types |
| [Automatic Differentiation](../notebooks/raster/autodiff.clj) | Forward-mode, reverse-mode, `value+grad` |
| [ODE Solvers](../notebooks/raster/ode_solvers.clj) | Lorenz attractor, adaptive solvers, events |
| [Linear Algebra](../notebooks/raster/linear_algebra.clj) | Vec/Mat types, LU, Cholesky, SVD |
| [Optimization](../notebooks/raster/optimization.clj) | L-BFGS, Nelder-Mead, Newton's method |
| [Deep Learning](../notebooks/raster/deep_learning.clj) | MLP training with compiled AD |

## Architecture

The [design/](../design/) directory contains architecture documents for
ongoing and planned work:

| Document | Topic |
|----------|-------|
| [Compiler](../design/compiler.md) | Compiler architecture overview |
| [Type System](../design/type-system.md) | Typed multiple dispatch design |
| [Autodiff](../design/autodiff.md) | AD design and implementation |
| [SOAC Fusion](../design/soac-fusion.md) | Loop fusion design |
| [Loop Vectorization](../design/loop-vectorization.md) | Auto-vectorization design |
| [Functional Combinators](../design/functional-combinators.md) | Pure parallel form design |
