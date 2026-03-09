# Third-Party Notices

Raster is released under the MIT License. This file records third-party projects
whose ideas informed specific modules and their licenses.

## Design Inspirations

### Futhark (ISC License)

The compiler pipeline's SOAC (Second-Order Array Combinators) classification, fusion
via dependency graphs, SegOp GPU lowering, and uniqueness-based memory analysis are
informed by the Futhark compiler (https://futhark-lang.org). Raster's implementation
is independent (different language, data structures, and algorithms) but cites Futhark
as the conceptual source throughout the compiler module comments.

Modules influenced: `raster.compiler.ir.soac`, `raster.compiler.passes.parallel.soac-graph`,
`raster.compiler.ir.segop`, `raster.analysis.memory`.

### OrdinaryDiffEq.jl (MIT License)

ODE solver Butcher tableau coefficients (DP5, Tsit5), dense output interpolation
coefficients, and adaptive step-size control formulas.
https://github.com/SciML/OrdinaryDiffEq.jl

Module: `raster.ode.core`.

## Dependencies

### EPL-1.0 Licensed

The following dependencies are licensed under the Eclipse Public License 1.0.
EPL-1.0 is compatible with MIT for distribution but EPL components retain their
own license terms.

- **Clojure** (org.clojure/clojure) — https://clojure.org
- **data.json** (org.clojure/data.json) — https://github.com/clojure/data.json
- **Typed Clojure** (org.typedclojure/typed.clj.checker) — https://typedclojure.org
- **Pattern** (pangloss/pattern) — https://github.com/pangloss/pattern
- **Beichte** (org.replikativ/beichte) — https://github.com/replikativ/beichte
