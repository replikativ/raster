# Scientific extension validation

Source-review checkpoint: 2026-09-13, Raster `fe3943a1` (implementation tree based on
`c0393fb1`). This is an experiment plan, not execution or performance evidence. It supplements
the [eight-item campaign](compiler-unification-campaign.md), without replacing its compiler,
training, or distributed acceptance gates.

## Architectural decision

Keep scientific operators as ordinary typed library programs composed through the existing
SOAC algebra. Domain, discretization, boundary, convergence, and approximation semantics belong
above that algebra. Storage, communication, scheduling, and target instructions remain explicit
below it. Do not introduce a solver-name registry or one compiler operation per physical law.

The comparison suggests three complementary reference ideas:

- Julia/SciML: ordinary method extension, reusable problem/algorithm/integrator interfaces, and
  dependency-aware specialization. [SciML's solver hooks](https://github.com/SciML/SciMLBase.jl/blob/573bc6c305c4321b7095da74f92f19ce034c16ee/src/SciMLBase.jl#L85-L127)
  are a library extension boundary, not numerical methods in the compiler.
- Devito/AMReX/PETSc: retain numerical and communication obligations long enough to optimize
  without confusing a cheaper approximation with an equivalent implementation.
- libCEED/UFL/MFEM: factor local operators into connectivity/restriction, basis operations,
  user pointwise physics, and transpose accumulation. [libCEED's public interfaces](https://github.com/CEED/libCEED/blob/3381e6d3ca5750f9dad1bcf0a0bec6afee7db4aa/include/ceed/ceed.h)
  make a useful small decomposition; a monolithic finite-element compiler IR is not required.

These are design inferences from source, not claims of feature or speed parity.

## Preserve the existing scientific library

Raster already has adaptive and implicit ODE algorithms, algorithm/cache/step methods, events,
and generic out-of-place states in `src/raster/ode/core.clj`. It also has CG, PCG, GMRES,
BiCGSTAB, Lanczos, and Jacobi preconditioning in `src/raster/linalg/iterative.clj`; PCG accepts
separate operator and preconditioner functions. These are not missing numerical algorithms.

The unresolved question is how uniformly these library programs specialize, retain state on the
device, compose control and reductions, and execute through checked distributed plans. Existing
methods are evidence of library capability, not proof of those end-to-end accelerator routes.
Conversely, limitations in a new distributed fixture do not imply the general library lacks a
solver. Audit existing APIs before proposing replacement records or semantic nodes.

Current narrower boundaries include radius-one flattened Dirichlet `SegStencil` scheduling,
whole-patch dyadic FP64 coarse/fine providers, synchronous distributed execution, and no verified
adaptive regrid/reflux lifecycle. General indexed programs may express more than the dedicated
schedule admits; measure public admission before widening an IR.

## Acceptance experiments, in campaign order

| Experiment | Required evidence | Design change only if exposed |
|---|---|---|
| Late-loaded typed method or AD-rule extension | Compile a caller and gradient, add/replace a relevant method or rule, then compare fresh and cached execution. Test unrelated cache reuse separately. | Dependency stamps/invalidation for semantic decisions, using existing method, descriptor and AD authorities. Current JVM callsite invalidation does not by itself prove transitive artifact invalidation. |
| External strided-array package | Same generic map/reduction on JVM and accelerator, nonzero offset/stride, alias and out-of-bounds rejection, no compiler edits in the package. | A checked extension protocol projecting retained type/layout/ownership facts into existing contracts; no per-consumer type inference. |
| Devito-matched 2-D heat evolution | Identical mesh, FP64 discretization, boundaries, stable timestep and physical time; scalar reference plus manufactured-solution mesh convergence. | Generalized neighborhood/access obligations only if ordinary typed indexed composition loses required facts. |
| PETSc-matched matrix-free Poisson | Reuse existing CG/PCG library logic; identical operator, Jacobi preconditioner and stopping policy; independently recomputed true residual and solution error. | Reusable resident iteration/control composition and explicit termination reasons, not a hardcoded CG kernel. |
| AMReX-matched conservative two-level transport | Constant preservation, mass balance, interface flux accounting, one deterministic hierarchy change, uninterrupted versus restored evolution. | Explicit combine/reflux and hierarchy-version transitions when needed. Prolong/restrict roundtrips alone do not satisfy this gate. |
| libCEED/MFEM-matched Q1 operator and adjoint | Restriction → basis contraction → ordinary user quadrature function → transpose contraction → additive scatter; assembled CPU oracle and adjoint identity. | Library-level mesh/basis data and checked indirect/oriented access if existing gather/scatter cannot express it. No FE-specific KernelBody opcode. |

Do not implement all proposed interfaces in advance. The first experiment is a small correctness
probe; the remaining experiments are external workload validation alongside the existing campaign.
Direct training/compatibility migrations remain the immediate implementation priority.

For the FE experiment, repeated destination indices are essential: testing only unique gathers
does not establish transpose scatter correctness. Follow with signed interior-face flux and
nonlinear pointwise AD only after the basic composition works. Existing `aget` adjoints do not
establish arbitrary gather/scatter AD end to end.

For scientific comparisons, report compile/setup separately from repeated execution, resident
allocations/peak bytes, transfers, reductions, and time to fixed solution accuracy. A residual
tolerance is not a discretization-error bound. Numerical order, timestep, preconditioner, reflux,
or coarsening changes are algorithm choices with acceptance contracts, not automatically legal
compiler rewrites. Analytical topology costs are not measured network/overlap performance.

## Reproducibility

Source reviews used fetched revisions without building dependencies or changing sibling source:

| Reference | Reviewed revision |
|---|---|
| Julia | `ca626376c0204796b57d06bb6c4517332f1542f3` |
| SciMLBase | `573bc6c305c4321b7095da74f92f19ce034c16ee` |
| OrdinaryDiffEq | `4a5db15cd59b89a3e5c7c57729d70e9c98110f81` |
| Devito | `3d35a2745cb193df4c05c9bb00e55b02210a4440` |
| AMReX | `66028f892b7d0e0902ffe821e10fcc0915ab1cda` |
| PETSc | `042ad87f3ecf1a25e780dea8472e79f63b840661` |
| MFEM | `10b0b596dbc26dca384b9f25f23026b1a6403592` |
| UFL | `f3290cd330545e4ce0232a005c525cc4aa0e0a2a` |
| libCEED | `3381e6d3ca5750f9dad1bcf0a0bec6afee7db4aa` |

The source audits do not establish an installed, runnable benchmark environment. Pin each future
experiment's actual dependency, compiler, runtime, device and numerical settings independently.
Keep installation and performance runs outside laptop test loops and required CI gates.
