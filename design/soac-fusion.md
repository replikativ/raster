# TypedSOAC fusion

Status: production architecture, reconciled with the implementation on 2026-09-01.

Raster performs functional fusion over `soac-dialect/Program`, not over the historical record
graph or walked Clojure forms. The production route is:

```text
closed typed Clojure
  -> TypedSOAC equations + AbstractValue/effect/alias facts
  -> certified fusion fixpoint
  -> scheduled SegOps / KernelGraph
  -> verified KernelBody
  -> target module
```

Source outside the direct frontend remains byte-for-byte unfused and enters explicit compatibility
lowering. A coverage gap may therefore retain a materialization, but it cannot select a weaker
fusion legality implementation. The former dependency-graph fusion implementation has been
deleted.

## Current algebra

TypedSOAC currently has functional equations for pointwise maps and tuple maps, unique-index
scatter, stencils, reductions, product/segmented reductions, segmented fold-map, scans,
contractions represented as segmented reductions, and scalar equations needed to retain host
shape/value dependencies.

Fusion iterates three general transformations to a fixpoint:

- Vertical producer/consumer composition for maps into maps, reductions, and scans.
- Reduction-result scalar and segmented-reduction result-map composition.
- Horizontal composition of independent maps with equivalent extents.

The rules rewrite typed lambda regions. They preserve explicit local scalar SSA, result dtypes,
ordered multi-results, physical result-storage contracts, effects, aliases, and constituent
provenance. Contraction result transforms are represented as typed store regions rather than as a
hard-coded attention or linear-algebra fusion pass.

Nested source scopes are alpha-renamed into the same ordered local-SSA region. Finite closed-core
integer `case*` forms become exact conditional scalar expressions at the TypedSOAC boundary, so
Clojure dispatch tables and keyword metadata are never part of scheduled or emitted kernel IR.
Guarded dense updates remain ordinary maps only when all reads of an inout result use the lane's
same logical index; nonlocal reads require a different certified schedule.

## Legality boundary

A transformation is legal only when the TypedSOAC value/effect facts prove it. In particular:

- Equation motion cannot cross a conflicting physical read/write/atomic footprint or a host-work
  barrier.
- Caller-owned destinations remain distinct from functional result identities.
- Destination aliasing, sibling store/read ordering, and unproved scatter conflicts decline.
- Reduction or scan reassociation requires an algebra certificate; ordinary recurrences do not
  borrow that proof.
- Multi-consumer recomputation requires a target-neutral cost witness. Otherwise the intermediate
  remains materialized.
- Every rewrite reconstructs and validates a complete typed program. A contradiction is a compiler
  error, not a reason to fall through to compatibility code.

These checks are the semantic certificate. Scheduling and autotuning may choose among legal
implementations but may not weaken them.

## Shape and extent identity

Launch extents are semantic scalar values, not emitter strings. Scalar bindings and known tensor
shapes normalize equivalent expressions such as `rows` and `(alength producer)` to one value before
fusion. Compound pure extents receive stable scalar SSA identities. This lets dense-result to
elementwise chains fuse without relying on mangled implementation names or syntactic coincidence.

## Profitability

Fusion legality is target independent; profitability may use an `AbstractMachine`. The current
fan-out rule compares recomputation against the dtype-specific machine ridge point and records its
decision and inputs in `:fusion/placements`. With no target facts Raster conservatively preserves
the materialization boundary.

This is the initial form of the adaptive/JIT contract: observed device facts and calibrated cost
models can refine a choice, while the typed program and numerical oracle constrain the search.

## Remaining work

1. Replace remaining compatibility operation coverage (active-ID narrowing, collect/atomics, and
   specialized block movement) with direct typed equations and schedules. RNG fill is now an
   ordinary typed map with wrapping SplitMix64 scalar SSA.
2. Finish explicit typed scalar SSA for every region; do not recover scalar types in emitters or
   beta-reduce away type contracts.
3. Finish explicit integer arithmetic semantics. Unchecked source add/subtract/multiply now retain
   `:wrap` in scalar `KernelBody` SSA and C-family targets execute them through the corresponding
   unsigned representation; add `trap` and proved-no-overflow contracts before admitting ordinary
   integral scalar algebra for indexing, RNG, and quantization.
4. Generalize `AxisMap` and layout facts for multidimensional views, strided gather/scatter, and
   block movement without turning layouts into semantic tensor operators.
5. Add independent numerical/property tests across aliases, mutation, fan-out, empty extents,
   floating-point edge cases, AD graphs, contractions, scans, and scientific stencils.
6. Extend schedule search and measured caches across GPU families while keeping the SOAC algebra
   and KernelBody target neutral.

## Relationship to reference systems

- Futhark supplies the model of a small functional array algebra with aggressive, legality-driven
  fusion and explicit uniqueness/alias reasoning.
- JAX/XLA supply staging, shape specialization, graph-level AD composition, and profile-guided
  executable selection.
- MLIR supplies the discipline of explicit dialect conversions, verified interfaces, and target
  lowering boundaries.
- Triton and Halide supply programmable schedules and hardware-aware tiling/locality choices.

Raster keeps Clojure/TypedClojure as the source language, its own functional semantic IR, and
inspectable schedule/artifact values. Attention, quantization formats, paged storage, and cluster
placement are consumers of this stack or typed representation/layout facts; they are not alternate
fusion authorities.

## References

- `design/compiler-north-star.md`
- `src/raster/compiler/ir/soac_dialect.clj`
- `src/raster/compiler/passes/parallel/typed_soac_frontend.clj`
- `src/raster/compiler/passes/parallel/typed_soac_fusion.clj`
- Futhark fusion: `../futhark/src/Futhark/Optimise/Fusion/GraphRep.hs`
