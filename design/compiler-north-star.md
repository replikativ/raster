# Raster compiler north star

Status: architectural direction, reconciled with the implementation on 2026-09-01. Objective,
fidelity/time schedule axes, irregular primitives, composition, and the inference track were added
on 2026-09-03 after the simulator survey (twenty rubric surveys of production simulators,
training and inference systems; internal working note).

Raster should become a compiler in which a typed Clojure program, its parallel
algorithm, its schedule, its device placement, and its executable artifact are
all inspectable values. The compiler may search and learn schedules, but it must
never learn around the type system, effect system, legality checks, or numerical
oracle.

The shortest description of the destination is:

```text
typed Clojure
    → semantic program IR
    → functional parallel IR (SOAC)
    → scheduled kernel graph
    → verified kernel IR
    → target module
    → composable Compiled artifact
```

This is an evolution of Raster's existing architecture, not a replacement with
Futhark, Triton, JAX, or MLIR. Raster keeps its typed multiple-dispatch frontend,
functional Clojure programming model, explicit AD rules, portable hardware
model, and vendor-JIT backends. The reference systems supply specific missing
ideas at specific layers.

## 0. Objective: what the compiler is for

Raster is the foundation for compiler-driven algebraic simulation and for inference through
simulators. The quantity to optimize is not kernel time. It is **grounding quality per unit of
compute**: posterior or decision quality obtained from a simulation-backed model for a given
budget of time, memory, energy and communication. Simulation-based inference that treats the
simulator as a black box is the worst case the system must always support. The intended case is
that inference uses the simulator's internals, through the same certified transformations the
compiler already owns: automatic differentiation, local linearization, coarse-grained or
marginalized variants, multi-level estimators, and emulators trained on internal state.

This changes what a unit of value is. An AI agent can write a bespoke SPH or shallow-water code
in days. What it cannot cheaply reproduce is a verified program transformation with a certificate,
a shared cost and measurement history across workloads, and composition of plans with a database
and provenance. Production simulators confirm the alternative: hand-instantiated kernel families,
several hand-written transports of one halo, two hand schedules of one physics, whole-loop taping
without checkpoint schedules. The unit of value in Raster is therefore **a verified transformation
of a program that keeps its semantic identity**, not a kernel. The acceptance test for the whole
project is the agent baseline in §9: if a bespoke reimplementation plus its certification is
cheaper than expressing the workload in Raster, the direction is wrong.

Three consequences are structural:

1. **The cost vector gains approximation error and information value.** Beyond latency,
   throughput, peak memory, bytes, tuning budget, energy and floating-point error, a plan carries
   the approximation error introduced by every fidelity, coarse-graining, marginalization or
   surrogate decision, under an explicit error model, and where a decision or inference task is
   known, the expected information gain or decision utility that the compute buys.
2. **Fidelity is a schedule axis.** Level of refinement, coarse-graining map, marginalized
   sub-model, multi-level estimator, surrogate substitution and precision are schedule decisions
   with legality rules (conservation, symmetry, positivity, coupling contracts) and an error
   model, exactly as tile size is a schedule decision with a resource rule. Coarse-grained models
   are proposals with an error bound, never silent substitutes.
3. **Every transformation carries an error model.** The minimal contract is not that every solver
   is a probabilistic-numerics solver; it is that every solver and every transformation exposes
   an error model the certificate can check and the cost vector can consume: an a-posteriori
   estimate, a multi-level difference, a perturbation ensemble, or a posterior variance.

Probabilistic numerics is the lens that unifies these. Discretization error is epistemic
uncertainty; a multi-fidelity hierarchy is a model over fidelities; coarse-graining is
marginalization with an error bound; a solver is an inference procedure whose posterior variance is
the numerical-error entry of the cost vector; optimal grounding per compute is experimental design
under a budget. The lens is a framing for the contracts above, not a requirement on users.

## 1. What is already strong

Raster is not starting from a toy compiler:

- `deftm`/`ftm`, typed dispatch, monomorphization, AD expansion, and qualified
  semantic operator identities form a capable language frontend.
- The nanopass pipeline has explicit pass ordering, diagnostics, and declared
  dialect arrows.
- SOAC fusion already handles useful vertical and horizontal fusion, aliases,
  multiple consumers, and profitability vetoes.
- Contractions carry explicit free and contracted axes, checked index maps,
  contraction facts, layouts, staged accumulators, and routes for scalar,
  register-tiled, DPAS, DP4A, and block-quant execution.
- Hardware descriptors centralize launch geometry, matrix-instruction shape,
  register budgets, cache and bandwidth data, and measured calibration
  provenance.
- Schedules are data, have a compile-time resource feasibility gate, and can be
  searched and cached by hardware signature.
- `Compiled` and `DeviceArray` establish device-resident values, ownership,
  donation, multi-output projection, and functional invocation.
- The backend surface already includes JVM, C/SIMD, OpenCL, Level Zero, CUDA,
  HIP, Vulkan, WGSL, and WASM compile gates.
- The quantized path already covers Q4_0, Q8_0, Q4_K, and Q6_K, including
  staged contraction and DP4A-related work.

These pieces are the foundation. The SegOp boundary now uses a first-class typed
`ParallelProgram`; its equations can retain a validated functional algorithm beside the ordered
scheduled operations. Remaining gaps are concentrated in migrating the full earlier SOAC program,
the remaining scheduled kernel-body representation, duplicated lowering, and string-template seams.

## 2. The load-bearing gaps

### 2.1 The middle end is not yet a sequence of real dialects

`segop-lower-pass` now produces a `ParallelProgram` with ordered equations,
SSA-like result IDs, `AbstractValue` contracts, effects, diagnostics, and provenance.
GPU and JVM backends consume the recorded operations; the source expression is retained only to
reconstruct scalar host control around them. The `:segop-lowered` validator performs a full
operation/type legality check, and source equality invalidates an equation after a backend-local
rewrite. Direct calls to a backend may still request explicit, counted compatibility scheduling,
but that request now crosses the shared SegOp middle-end boundary; JVM and OpenCL emitters no
longer construct legacy SOAC nodes themselves.

The typed map/reduction vertical is now production-routed for supported programs whether or not a
fusion fires. Unsupported source remains deliberately unfused for compatibility lowering; it can
no longer re-enter an untyped fusion authority. A
`ProgramEquation` owns the alpha-renamed TypedSOAC program and the SegRed mechanically derived from
its explicit lambda parameters, operands, extent, result, dtype and algebra facts. JVM SIMD consumes
that SegRed. The GPU schedules eligible scalar regions as one verified workgroup-tree `KernelBody`
with typed scalar SSA, stable reads, workgroup allocation, masks, barriers and an ordered result ABI;
the same body emits as OpenCL, CUDA and HIP and is compiled by the hardware-free vendor CI gates.
Unsupported scalar regions decline explicitly to the established verified SegRed OpenCL emitter.
The public `equation-first/compile` boundary now selects that C-family emitter directly from a
synthetic or physical OpenCL, CUDA, or HIP target descriptor. Its first hardware-free public
vertical covers portable SegRed and SegFoldMap KernelBodies, preserves `:fallback :none`, validates
that every artifact target agrees with the emitted program dialect, and sends the resulting CUDA
and HIP reduction phases through nvcc/hipcc in CI. Graph families that have not yet acquired a
portable KernelBody fail at this boundary instead of embedding OpenCL source in a CUDA/HIP
program. Typed one-dimensional SegMap now shares one scalar-expression lowering with SegFoldMap:
stable loads, retained dtypes, scalar SSA, branches, and horizontally fused stores emit from the
same KernelBody to all three C-family targets. Canonical loop/recur scalar carries become ordered
KernelBody `ForLoop` regions rather than being reassociated as reductions. This is sufficient for
the public seven-stage GQA composition to emit entirely as CUDA/HIP and pass both vendor compilers;
certified stencils and scans now lower their
single- or three-phase schedule to target-neutral KernelBody scalar SSA, workgroup storage,
barriers, ordered block carries, and inclusive/exclusive result placement. The same bodies compile
with OpenCL, nvcc, and hipcc; the former handwritten OpenCL scan and stencil source paths have
been deleted.

The map/scalar/full-reduction/certified-scan front end now constructs TypedSOAC directly from closed analyzed
source and retained walker type metadata. It establishes stable equation/value identity, types,
effects, aliases and placement provenance before fusion; no `ir.soac` record or record-to-dialect
adapter participates in this production route. The old dependency-graph fusion path has been
removed. Scan has a distinct
equation with an explicit `:inclusive` or `:exclusive` result mode and a checked `AssociativeScan`
certificate; its caller-owned destination and output layout are facts, not part of the functional
algebra. Both modes lower directly to the same one- or three-node SegScan `KernelGraph` without
constructing a legacy `SoacScan`. Exclusive mode retains logical traversal extent `n`, records its
result as `n+1` through checked symbolic integer-expression IR, and specializes scheduled stores to
materialize the identity at zero and prefixes at `idx+1`. Scan uses the same checked
logical-result/physical-`result-storage` relation as every other destination-writing TypedSOAC;
the destination is neither a functional capture nor a scan-specific backend convention. The GPU
backend now target-lowers that
scheduled value through one fail-loud graph-emission boundary, wraps it in the same
`KernelDispatch` value used by other selectable
schedules, and resident descriptor extraction retains it as one executable step. LinkPlan and the
runtime binder allocate its private temporary and flatten its nodes without reconstructing scan or
ABI semantics. The ordinary per-call compiled function now invokes that same registered
`KernelDispatch` through a backend-neutral staged `KernelExecutable` runner: its ordered ABI types
scalars, preserves pointer identity, stages JVM arrays (or borrows resident buffers), owns graph
temporaries, and copies back only declared writes. No duplicate sequential source implementation
is embedded in the emitted form. The public GPU-session compiler now crosses the same direct
TypedSOAC and scheduled SegOp boundary instead of asking the OpenCL emitter to reconstruct generic
maps and reductions from walked source. Destination-writing maps retain explicit destination,
alias, effect and return facts. General recurrences remain outside this typed operation and must
not borrow its reassociation proof. The obsolete raw exclusive-scan GPU generator is no longer a
backend fallback: unscheduled source fails loudly. Remaining compatibility operations must join the
direct typed front end before their source-reparsing lowerers can be deleted; coverage debt may
retain a materialization boundary, but it cannot weaken fusion legality.

Pointwise maps that read and write their caller-owned destination stay on this same route. The
typed equation records read/write destination access, and GPU lowering emits that storage exactly
once as a `KernelABI` `:inout` pointer with semantic role `:result`. Physical access and functional
identity are therefore orthogonal: resident and staged binders derive transfers from ABI kind,
while composition identifies the returned value by role. The portable schedule admits an inout
map only when every destination read uses the same logical element index as its write; shifted or
indirect reads still require a stencil, scatter, or ordered schedule. OpenCL and Level Zero staged
maps also use target-specific invocation markers selected by the emitter, rather than leaking a
Level Zero runtime call into an OpenCL program.

Effect-only pointwise maps now use the same orthogonal contract for several outputs. `map2!` and
independent multi-store `map-void!` bodies become one tuple-valued TypedSOAC `map`: fresh logical
result IDs remain SSA values, while an ordered checked `:result-storage` vector maps each result to
its caller-owned destination, access mode, and host-return contract. The effect binder therefore
retains its real `nil` host semantics instead of masquerading as a tensor result. The scheduled
SegMap and its ordered ABI are consumed by staged and resident compilation; unsupported raw
map-void bodies retain the compatibility generator. The front end refuses duplicate destinations,
uncontracted scatter indices, sibling write/read ordering, and atomics. A destination wrapped in
`par/unique-index` carries an explicit caller proof of conflict freedom into a TypedSOAC `scatter`;
the marker is consumed before scalar lowering, while the checked read/write storage contract and
guarded indexed store remain visible to scheduling and ABI construction. Shared scalar work now
remains one explicit ordered local-SSA spine inside the tuple map's scalar region; nested lexical
scopes are alpha-renamed into that spine, each definition has a dtype retained from
walker/TypedClojure facts, and local binders never become fake program inputs. Closed-core integer
`case*` dispatch is projected into exact typed conditional expressions before scheduling; hash
dispatch maps and switch metadata never reach KernelBody or a target emitter.
Validation proves definition order and lexical closure, and the front end expands locals only for
I/O discovery. A direct sibling-destination read still preserves imperative store ordering and
declines the functional tuple map, while a typed local may snapshot any destination before the
store sequence and safely feed several results. JVM and GPU lowering both
materialize the same typed spine once around all tuple results. Vertical and horizontal fusion
alpha-rename and compose these regions while preserving definition order; a typed producer-result
bridge is introduced only when reuse requires it, so fusion does not duplicate scalar work. Flat
resident `deftm` signatures provide their declared per-buffer dtypes before fusion, so a
mixed FP32/int8-input, FP32/int32-output map does not inherit a global default dtype.

Mixed kernels whose lane performs both unique writes and certified reductions use a distinct
typed `effect-map`, rather than disguising ordered mutation as a tuple-valued functional map. Its
ordered destinations, predicates, local SSA and per-destination conflict certificates schedule
directly to one `SegMap`; unique effects lower to `ScalarStore`, reductions lower to `AtomicRMW`,
and guarded effects retain structured `IfRegion` control in the common `KernelBody`. The same body
and ordered ABI emit as portable OpenCL, CUDA, and HIP. Fusion treats this operation as a
conservative boundary until an effect-aware rule proves that moving or combining it preserves the
declared memory order. Source selection never infers indirect-write uniqueness from a workload or
variable name. Closed typed source forms parallel effect maps only from explicit uniqueness or
reduction certificates; unproved ordinary conflicts instead carry `:ordered` and
`:iteration-order :sequential`. That schedule keeps the semantic extent and ABI unchanged, launches
one work item, and emits an explicit source-order `KernelBody.ForLoop` on OpenCL/CUDA/HIP. JVM
execution consumes the same ordered region. Typed locals are pre-effect snapshots, while direct
reads after sibling writes are legal only in the sequential schedule.

This proof boundary exposed a pre-existing firms-ABM race rather than hiding it: active IDs are
sampled with replacement, so two decision lanes may write the same agent state and cannot carry a
`unique-index` certificate. The semantics-preserving production route is a resident ordered queue
loop (or, later, grouping by agent followed by one ordered lane per group). The target-neutral
ordered-iteration schedule now represents that route; parallel `effect-map` remains unavailable
without a real conflict-freedom certificate.

Adam and AdamW use the functional tuple-map storage contract directly: gradient is a read-only operand, parameter
and moment buffers are typed inout storage, and the bias-corrected moment snapshots form one local
SSA spine evaluated before all three writes. Each optimizer lowers as one certified resident
kernel with zero compiler allocation or compatibility fallback. The C-family vector fast path
currently declines multi-store lexical regions rather than inlining a snapshot past a state write;
vector KernelBody lowering of local SSA is the remaining schedule improvement, not a second
optimizer-specific semantic path.

Boundary-aware scientific stencils now enter that same direct vertical as a distinct functional
`stencil` equation. The equation keeps whole neighborhood tensors as stable captures, proves every
load to be a constant affine offset within its declared radius, and separates the logical result
from caller-owned output storage. It lowers mechanically to `SegStencil`; JVM SIMD consumes the
scheduled value without reparsing source, while OpenCL emits a typed ordered ABI whose guarded
Dirichlet branch dominates all neighborhood reads. The first SIMD schedule supports radius one;
wider neighborhoods and multidimensional tiling remain schedule extensions, not new semantic
operators or emitter-side pattern registries.

The first architectural correction is therefore:

> SOAC, scheduled kernel graph, and kernel IR must be first-class program values,
> and every conversion must have a legality target that fails loudly.

### 2.2 Algorithm, schedule, and emitted kernel are still entangled

Raster has schedule data and hardware-aware contraction routes, but there is no
single stage that applies a schedule to an algorithm and produces the kernel IR
consumed by every emitter. Several scheduling axes are costed or gated more
broadly than they are emitted. Resident BLAS GEMM calls now close over compiler-emitted
scalar/direct/split executable graphs, but their semantic front door still sits outside the
general SegOp contraction route and the XMX leaf remains a target-specific emitter template.

The required separation is:

- **algorithm** says what values are computed;
- **schedule** says where, in what order, with what tiling/layout/staging;
- **kernel IR** records the selected execution without losing the algorithm's
  semantic identity;
- **emission** is a target spelling of the verified kernel IR.

No schedule key is complete until it is declared, checked, costed, emitted, and
measured. A key that intentionally models feasibility before emission must be
labelled as such in the schema and excluded from claims of executable coverage.

The first general `KernelBody` vocabulary now extends the matrix-fragment path with typed scalar
SSA, explicit literals and conversions, rank-checked scalar loads/stores, structured `if` and
loop-carried regions, and full-participation subgroup reductions/broadcasts. Branch and loop yields
are typed products; use-before-definition, identity collisions, divergent collectives, implicit
masked-load values, conversion policy gaps, and launch/subgroup mismatches fail verification. Body
launches use the shared `LaunchSpec`, keeping host launch expressions separate from in-kernel index
arithmetic. Group, 1–3D local-thread, subgroup, and subgroup-lane indices are distinct hardware
sources. Uniform memory facts require an explicit body-level stable-read contract; the checked
KernelBody-to-ABI seam projects it to a no-write-alias precondition, artifacts reject equal compiler
bindings, direct calls check resident ranges, and graph/link binding retains its stronger
overlapping-write rejection. Floating narrowing distinguishes IEEE overflow from integral
wrap/saturate/trap policies. Unchecked integral add/subtract/multiply retain an explicit wrapping
policy in scalar SSA, and the shared C-family lowering performs those operations in the
same-width unsigned representation rather than relying on undefined signed overflow. Scalar SSA
also distinguishes a compiler-certified `:no-overflow` operation from a semantic `:trap`; C-family
targets emit the former as signed arithmetic. CUDA and HIP lower the latter through checked
same-width unsigned predicates followed by PTX/LLVM target traps; OpenCL targets explicitly decline
it because OpenCL C has no standard trap primitive. Schedule-local scan extent subtraction
is the first proved operation. Source-derived address algebra is not blanket-certified: it still
awaits AxisMap/range proofs. Whole-kernel
workgroup allocations now have static typed shapes, named
layouts, explicit 1–16-byte alignment, and one deterministic packed-memory plan; launch shared-byte
accounting must match that plan exactly. Full-participation acquire/release workgroup barriers are
rejected outside workgroup-uniform control flow and lower from the same operation to OpenCL,
CUDA, and HIP. Workgroup-cooperative contiguous global-to-workgroup copies now carry explicit
source/destination coordinates, element and transfer widths, cache intent, full participation, and
preferred-versus-required overlap. Named commits close the exact copies issued since the previous
commit; waits consume an oldest group prefix and state the remaining pipeline depth. The verifier
rejects unstable sources, divergent base addresses, live destination reuse, lifetimes crossing a
structured-region boundary, and staged-memory consumption before a separate workgroup barrier.
Required overlap also proves coordinate alignment, projects its base-address alignment through the
ordered ABI, and direct calls validate resident ranges before launch; preferred CUDA overlap keeps
a uniform runtime-alignment fallback.
OpenCL lowers the dependency graph to events, CUDA targets with compute capability 8.0 or newer to
`cp.async` groups, and older/unknown CUDA or HIP targets report an honest synchronous cooperative
lowering; those targets reject a schedule that requires overlap. Masked collectives
remain absent until a concrete schedule supplies verifiable participation. An implemented OpenCL
target spelling now lowers this general scalar/control vocabulary directly: dense ranked loads and
stores (including contiguous leading-slice views), SSA expressions, structured branch/loop yields,
explicit numerical casts, and full-subgroup reductions/broadcasts, without recovering an algorithm
or schedule. Unsupported target semantics fail at this boundary instead of silently changing a
cast, layout, collective, or reduction-association contract.

The first production route now constructs
`SegmentedWeightedReductionSchedule -> KernelBody -> OpenCL` and projects the body's stable reads
onto the invariant ordered ABI. Attention contributes only the routed-paged storage descriptor used
by this first lowering row; neither KernelBody nor the target emitter contains an attention
operation. Dense/CSR physical page routing combines independently with bounded contiguous intervals
or bounded CSR membership. The generated leaf has replaced the handwritten cooperative source and
is checked against the reference algebra on a real device. Its subgroup builtin is recorded honestly
as implementation-defined association rather than claiming a fixed tree. A second production
schedule now partitions bounded membership into static contiguous tiles, emits one partial
KernelBody per segment/tile, and deterministically merges the explicit maximum/denominator/weighted
value state in increasing tile order. The four typed partial buffers are operation-derived,
graph-private temporaries; the source operation and external ABI are unchanged. This two-kernel
schedule is opt-in until measured selection can account for dynamic history length and temporary
traffic. Scalar/control KernelBody now has one C-family lowering core with thin portable OpenCL,
CUDA and HIP spelling descriptors. The same scheduled cooperative and tiled-history bodies retain
one ordered ABI while OpenCL uses subgroup builtins and CUDA/HIP select explicit shuffle-down trees;
the emitted artifacts record that target association. CUDA is compiled to PTX and HIP to an RDNA3
code object in mandatory hardware-free CI jobs, while real Intel execution remains the numerical
oracle. The compiler fixtures now stage values through both synchronous workgroup memory and the
verified async issue/commit/wait contract; the exact async body is compiled to CUDA sm_80 PTX and
an RDNA3 HIP code object in hardware-free CI. CUDA/HIP runtime registration, launch and on-device
numerical coverage remain deliberately separate from source legality. The verifier still forbids
pending asynchronous state and live staged storage from escaping an ordinary structured region,
but `PipelinedFor` now carries a complete ordered async queue across loop iterations, verifies each
rotating stage's layout and lifetime, and requires an explicit exact or separate-epilogue tail
policy. OpenCL lowers the queue through native event variables; CUDA maps it to `cp.async` groups;
HIP preserves the same verified schedule with an honest synchronous fallback. A production
double-buffered weighted-reduction schedule now stages two dense-paged FP16 K/V membership rows,
overlaps each refill with consumption of the other row, and handles histories of length zero or one
through the prior sequential body and even/odd tails through an explicit drain/epilogue. It retains
the semantic plan, ordered ABI and interval bounds; CSR membership, CSR physical routing and
unaligned row widths decline to their prior verified schedules. The exact body runs against the
reference algebra on Intel and compiles through nvcc and hipcc without target hardware.
Shared-memory layout is now an explicit verified KernelBody facet. Its finite family consists of
identity and named 2/4/8/16/32-period row-XOR permutations; the verifier proves static shape,
in-bounds bijection and exact byte charge, and the shared C-family emitter lowers the same physical
address transform to OpenCL, CUDA and HIP. Native contiguous async copy accepts only the identity
member. Hardware descriptors normalize explicit
bank topology, derive documented CUDA/AMD values and Intel Level Zero's device-dependent current
default, and abstain for generic OpenCL. The resource model reports allocation feasibility,
broadcasts and per-bank transaction conflict degree. The production row pipeline now represents
its rotating storage as one typed two-dimensional stage ring. Stage depth zero and depth two crossed
with every shape-legal finite swizzle form one emitted, ABI-compatible `KernelDispatch` candidate
set. Identity retains native event/`cp.async` lowering; non-contiguous XOR rows use an honest
cooperative layout-aware scatter until a native tensor-copy operation exists. Explicit offline
tuning validates every candidate against an oracle, measures device events, caches the complete
device/source/numerical/layout identity, and returns a fixed selector without adding a synthetic
runtime scalar to the ABI. The old schedule-map coordinate-descent tuner is gone. Routed-row
lowering proves algebra, membership, storage, shapes and dtypes instead of accepting an attention
provenance label as a legality token. Automatic production selection remains deliberately separate
from compilation: an untuned program keeps its analytic or explicit policy, while measured
selectors are immutable artifacts that can be reapplied only after identity verification.

### 2.3 Executable steps and resident artifact values compose

Descriptor instances share one executable-step binder: a step may select a
single `KernelArtifact` or a multi-kernel `KernelGraph`, own its private
temporaries, and flatten with other instances into one resident replay graph.
This closes the mechanism gap for GEMM-containing layer composition.

`LinkPlan` lifts that mechanism into a public immutable compiler value. Stable
typed/shaped nodes replace name-decoding binders; ordered descriptor instances
carry scalar and schedule environments; pure validation proves views, ranges,
ownership, aliases and producer/consumer effects before runtime allocation.
Instantiation allocates internal nodes once and returns an owned executable
value over one replay graph. It can also attach to a caller session and import
borrowed/external allocations without taking ownership, which is the boundary
needed by paged cache managers.

`ProgramStage` supplies the corresponding replacement seam for external runtime
effects. A frontend declares stable state/output anchors; Raster derives the
unique minimal descriptor interval from checked executable ABIs, proves its
live boundary, and projects ordinary before/selected/after descriptors. Paged
attention is the first consumer, not a special compiler pass or linker mode.

The pretrained decoder now composes through declared `ProgramStage` boundaries and `LinkPlan`
node identities rather than handwritten buffer-name inspection. An ordinary resident descriptor
has a pure certifying conversion into a one-instance plan: the checkable witness preserves
parameter order, pointer/value identity, scalar specialization, roles, realized view contracts,
schedule, aliases, target, and outputs. `Compiled` owns that witness and invokes only its
`LinkedExecutable`; compatible device inputs use an exact-view no-op or a device-to-device copy,
never an implicit host round trip. The runtime preserves the certified allocation identities and
profiles the same stable recorded-graph boundary. Each instance also retains the complete ordered
specialization environment required by descriptor shape closures, while runtime pointers still
come only from certified node bindings. Independently lowered `Prepared` artifacts now compose
before allocation through semantic input/output keys. The composition certificate namespaces
component identities, unifies explicit dataflow and shared constant/input nodes, and re-derives one
validated plan; one later instantiation therefore removes even the device copy. The remaining
value-layer cleanup is ranged-view composition and explicit cross-component ownership transfer;
the duplicate legacy whole-program binding API is retired.

Pretrained-rstr's batch boundary is the current concrete shape test: weights bind once to shared
constant nodes, while residual, scratch, position, logits and token storage are lane-local nodes or
disjoint views. Packed Q/K/V and attention views already fit LinkPlan. The first projection slice now
makes Q8_K activation quantization and Q4_K DP4A projection uniformly row-capable: row count is an
ordinary outer shape/launch extent, activation leaves are row-major, and packed weight leaves remain
shared. B=1 is no longer a separate kernel contract. The remaining performance work is to derive
these projection, post-attention and head programs from the scheduled contraction/SOAC path rather
than treating the Q4_K source kernel as the final abstraction; batching is not a linker convention
or an attention special case.

Buffered RoPE follows the same rule: one kernel consumes row-major `[B,heads,head-dim]` values and
the routed attention `int positions[B]` buffer, so B=1 and B>1 neither select different kernels nor
upload duplicate position representations. Dense semantic rows whose logical width differs from a
quantized contraction's padded width use a generated padded-row quantization adapter. It synthesizes
zeroes while packing and therefore avoids a materialized padding buffer; the branch-free equal-width
kernel remains available for already padded layouts. Padding is layout, not quantization-specific
memory ownership.

Decoder selection is likewise an ordinary indexed row reduction, not an attention or quantization
primitive. TypedSOAC represents it as `product-reduce`: ordered independently typed components,
identities and algebra accompany distinct element and closed binary-combine regions. A component
may participate in the algebra without becoming a materialized SSA result; an explicit ordinal map
connects only retained components to caller-owned result storage. This projects mechanically to the
canonical `ProductReduction` used by scheduling rather than recovering either region from source.
The first portable `ReductionSchedule` maps a segmented product to
strided lane folds, a fixed workgroup-local tree and segment stores. Its workgroup is constrained by
the target thread limit and the sum of every component's local-memory width, while numerical mode
and tuning candidates remain inspectable data. `argmax-rows!` therefore emits one mixed-type
`(value,index)` workgroup tree per row with no compiler-visible global scratch. The semantic ABI
contains only values, output indices, row count and width; ties select the lowest index, and the first
NaN outranks numeric values so corruption is visible and deterministic. Its public compile report
now proves direct analyzed-source-to-TypedSOAC routing, typed schedule reuse and zero compatibility
fallback. Subgroup/shuffle and multi-workgroup alternatives remain schedule candidates, not new
semantic primitives.

TypedSOAC now also names the general `segmented-reduce` algebra directly: ordered parallel segment
axes, one innermost reduction axis, typed accumulator products, dense element operands and stable
arbitrarily indexed tensor captures. Its extents participate in the typed program boundary and
alpha-remapping, and general equations lower mechanically to canonical `SegRed` without
constructing the former `SoacContract` record. Ordinary scalar `raster.par/contract` source now
enters this equation directly; the operation is intentionally not a matmul node, so scientific
contractions, batched reductions and attention-derived reductions share the same semantic
vocabulary. Target selection consumes the validated equation/facts together with its already
scheduled `SegRed`; matrix, register-tiled and portable families do not reconstruct a source form
or a second reduction operation. Ordinary typed contractions
lower to a canonical `SegRed` whose `ReductionSchedule` records the hardware candidate families,
workgroup search space and numerical constraints; the validated TypedSOAC equation remains attached
to its `ProgramEquation`. Candidate families are executable constraints rather than diagnostics:
pinning `:matrix`, `:register-tiled` or `:portable` disables the other leaves, and compilation fails
loudly when no enabled family can implement the equation. GPU routing derives its transient verified
contraction facts from that equation behind a typed routing API and never reparses the walked source
form; the OpenCL pipeline
therefore knows only the typed program, stable operation ID and certified candidate schedule, not
the projection used by compatibility leaves. Host materialization and the still-form-backed
staged/quantized compatibility leaves request an explicit surface spelling locally; it is not a
common routing invariant. A segmented reduction may now carry one typed scalar
result-transform: its accumulator, expression, result dtype, tensor operands with axis maps/dtypes,
and uniform scalar captures with dtypes are all verified at the equation boundary. Contraction
epilogues expressed by that closed contract therefore stay on TypedSOAC and lower through the same
matrix store region and ordered ABI as the proven compatibility oracle. Typed fusion now discovers
an adjacent, single-use pointwise map over a caller-owned segmented-reduction result as this same
result-transform. It transfers the map's logical result, physical storage, alias/effect and host
return boundary to the reduction; removes the dead intermediate; proves each residual/broadcast
operand's map over the segment axes; and leaves the fold lambda byte-for-byte unchanged. Ambiguous
indices, observable intermediates, read-write stores and dtype changes decline rather than guessing.
Staged quantization, decode lambdas, declared physical operand maps and output conversions remain on
the certified compatibility front door, and may still use `SegContract`, until the typed equation
has explicit facts for them; admitting them while dropping those contracts would be a miscompile,
not migration progress.

The typed route can also enumerate every legal enabled contraction family without benchmarking or
choosing among them. Each result retains its pinned `ReductionSchedule`, concrete strategy,
validated executable artifact and family-qualified decline trail, so offline tuning measures the
same compiler products that ordinary execution uses. These alternatives may still have different
physical ABIs and therefore are not falsely packaged as a runtime `KernelDispatch`; ABI
normalization or graph-private adapters must precede runtime selection through one dispatch.
For static shapes, one-node candidate graphs now provide exactly that normalization: the logical
operand/result pointers and typed result-transform scalars form the shared external ABI, while
leaf-only `M/N/K` or flattened segment bounds remain checked graph-private integer scalars.
Runtime-dependent private dimensions are
refused until they are represented in a shared public scalar ABI, preventing an offline sample from
being mistaken for a valid dynamic specialization.
The production OpenCL pass registers those static graphs as one fixed-selector dispatch and invokes
it through the generic scheduled-executable runner. It emits every legal leaf once, preserves the
current analytic winner as the default, and records dynamic-shape or special-protocol declines
instead of hiding a fallback. Each dispatch carries the generic tuning contract that maps a
validated measured fixed selector into `[:typed-contraction :measured-selectors <dispatch-id>]` in
the resolved schedule. Its ID and entry points are derived deterministically from canonical emitted
candidates, so recompilation reproduces the executable identity and can consume a cache hit without
benchmarking. Measurement remains an explicit runtime action guarded by the existing oracle,
device-event, stationarity, numerical-mode and layout checks.

Tensor contraction uses that same semantic operator rather than reconstructing `init`, `combine`
and a fold body in `contract-lower`. The single contraction-facts derivation now owns its canonical
one-component `ProductReduction`, normalized reduced coordinate and flattened semantic body;
ordinary `SegRed` scheduling and peak routing consume those facts unchanged. The compatibility
`SegContract` projection is now limited to contractions that are not yet in the typed subset. Free
axes and typed result transforms are part of the functional equation; operand axis maps,
decode/storage precision and staged accumulator structure remain orthogonal contraction facts and
schedule choices. Contraction is therefore an ordinary segmented
TypedSOAC reduction, not a new hard-coded GEMM algebra; matrix, register-tiled and portable kernels
are competing schedules for that algebra.

Indexed storage movement remains a separate generic operation. Allocation-free `gather-blocks!`
and `scatter-blocks!` move contiguous typed blocks between dense staging and routed resident
storage through the direct TypedSOAC→SegMap vertical: gather is a stable indexed read in a dense
`map`, while scatter is an explicit guarded `scatter` with unique destinations. Compound launch
extents are hoisted into typed scalar SSA before scheduling. Routing is an `int[nblocks]` buffer; it does not
encode pages, attention, cache policy, or quantization. Gather permits repeated sources, while the
non-reducing scatter contract requires unique destinations. Host validation proves active extents,
index bounds, non-aliasing and the current 32-bit device-index limit; emitted kernels retain bounds
guards. FP16 is a bit-preserving short-array storage overload whose physical ABI is `half*`, while
index arithmetic remains in the ordinary scalar compiler domain. `gather-rows!` remains the
row-oriented compatibility spelling. Consequently greedy decode composes indexed reduction and
row gather as ordinary graph dataflow, while chunked paged prefill and cache persistence can compose
bulk host/device staging with block scatter/gather. Neither path introduces a decoder, attention,
page-manager, or quantization primitive into the compiler.

Repeated-destination updates are not smuggled through the unique-write contract. Flat
`scatter!` and additive `reduce-by-key` now canonicalize to the same proof-carrying TypedSOAC
scatter: its conflict value records the operator, identity, dtype and checked commutative-monoid
certificate, while each scalar-region write denotes a contribution. The scalar JVM schedule
executes exact ordered read/modify/write updates; the portable OpenCL schedule selects typed atomic
addition and retains an `:inout` ABI. Thus histogram/scatter algebra is independent from the key,
value and destination layouts. Strided contributions, non-additive atomics, privatized/tiled
histograms and deterministic floating-point accumulation remain schedule and numerical-policy work.

Artifact linking and value rebinding are not runtime conveniences. They are
compiler primitives required for competitive model execution.

### 2.4 Shape, layout, effects, ownership, and sharding are separate facts

The current metadata and descriptor registries contain much of this information,
but there is no one JAX-like abstract value threaded through transformations.
Consequently, shape polymorphism, layout propagation, aliasing, device
placement, and future distributed sharding do not share a single checked
contract.

The first value-layer slice now makes that separation concrete. `AbstractValue` records logical
dtype/shape/layout, numerical representation, memory space, placement, sharding, ownership and
effects without containing a buffer or backend handle. `LinkValue` records a later physical layout
and an ordered set of named `LinkNode` leaves. Kernel ABI field identities certify the flattening,
and the common resident binder expands the same composite contract for Level Zero and OpenCL. A
Q4_K value can therefore be represented by packed blocks, scales and sums while allocation remains
generic byte storage; the allocator does not branch on Q4_K. Existing dense plans receive implicit
one-leaf values. The follow-up slice makes resident parameter and allocation descriptors speak the
same contract: composite parameters have field initializers, composite scratch has independent
leaf size closures, and allocation still sees only typed byte ranges. Certified composition now
unifies and shares whole logical values atomically while retaining both logical and physical
certificate mappings; public composite outputs must flatten every leaf in field order. The
remaining value-layer boundary is ranged/subview composition with explicit ownership transfer.

### 2.5 Workload coverage is broad at the backend edge, not yet through one door

Triton's practical strength is not merely its set of operations. Masks, block
pointers, layouts, reductions, scans, atomics, gathers/scatters, tensor-core
forms, and software pipelines travel through one staged compilation path.
Raster has many of the leaves, but production reachability through the same
verified IR is the important coverage metric.

## 3. The settled architecture

### 3.1 Semantic program IR

The output of tracing/lowering should be an immutable, SSA-like program with
unique value identities, nested regions, source locations, and explicit
constants. S-expression syntax may remain the readable frontend form, but
passes that need semantic facts should operate on program values rather than
recovering them from spelling or symbol metadata.

Every value has an `AbstractValue`:

```clojure
{:dtype        :f16
 :shape        [batch seq hidden] ; dimensions may be symbolic
 :logical-layout {:order [0 1 2]}
 :memory-space :device
 :placement    {:device :ze:0}
 :sharding     nil
 :ownership    :owned
 :effects      #{}}
```

The program envelope and fact tables may be records or validated maps, while operation and region
syntax may remain compact S-expressions. The invariant matters more than the container: all facets
describe the same explicit value identity, and a pass cannot update one without revalidation.

Operations use one descriptor/interface system. In addition to the existing
buffer, device, shape, placement, algebra, comparison, and result facets, an
operation can define:

- abstract evaluation;
- effect and alias behavior;
- JVP, VJP/transpose, and batching rules;
- layout and sharding constraints;
- canonicalization and fusion traits;
- legal lowerings for each next dialect;
- cost features and measurement identity.

This borrows JAX's useful core—abstract values, primitives, explicit equations,
transformation rules, effects, and pytrees—without borrowing Python tracing
semantics or making retracing the programming model.

### 3.2 Functional parallel IR

The portable algorithm dialect should contain nested, typed parallel
operations. Its minimum useful family is:

- map;
- reduce and scan, including multiple results;
- fused map/fold compositions;
- histogram;
- stream/chunked fold;
- gather and scatter;
- stencil;
- contraction;
- structured `if`, `while`, and call regions;
- explicit effect/token operations where functional dataflow is insufficient.

Futhark is the semantic reference here. In particular, Raster should adopt its
discipline of typed SOAC inputs/results, nested parallel distribution,
regularity and balance legality, and typed kernel results. Raster need not copy
Futhark's Haskell representation or uniqueness type system wholesale.

This dialect is hardware-independent. Hardware facts may be supplied to a
costing or scheduling pass, but must not change the meaning of an operation.

#### 3.2.1 Representation and Pattern integration

Raster keeps the small functional, Lisp-native character of the SOAC IR. The intended form is a
typed, dialect-validated S-expression program rather than a tools.analyzer-style map AST or a graph
reconstructed from arbitrary source forms. Conceptually:

```clojure
(soac-program
  {:inputs [%x]
   :values {%x x-value %y y-value %z z-value}}
  [(= map-0 [%y]
      (map {:index %i :extent %n} [%x] []
           (lambda [%xi]
             (region [(let-value %squared :float (* %xi %xi))]
                     [%squared]))))
   (= reduce-0 [%z]
      (reduce {:index %i :extent %n
               :accumulators [%acc] :identities [0.0]
               :dtypes [:float] :algebra [{:associative? true}]}
              [%y] []
              (lambda [%acc %yi]
                (region [] [(+ %acc %yi)]))))]
  [%z])
```

`../pattern` remains the rewrite engine. Its nanopass dialects declare the accepted S-expression
grammar and pass arrows; fusion rules match the compact SOAC equations directly. The surrounding
`ParallelProgram`-style envelope supplies stable value/equation IDs, `AbstractValue`s, effects,
aliases, consumption, results, diagnostics and provenance. Essential facts are explicit fields or
table entries keyed by IDs, not Clojure metadata that ordinary list reconstruction can drop.

The first Pattern-declared dialect now makes this boundary executable for map and scalar/full
reduce equations. It verifies ordered SSA definitions, exact typed input/output/effect boundaries,
rank/extent/result contracts, explicit lexical capture binding (including compound stable value
IDs), tuple-valued maps, and fact-preserving functional fusion. Differential tests compare
map→map, map→reduce and horizontal-map transformations with the former graph over their overlap,
but the graph is no longer a production certificate. Supported maps and scalar/full reductions
route through a `:typed-soac` ParallelProgram even when no fusion fires; host-visible intermediates
remain materialized typed equations. Horizontal fusion lowers tuple-valued maps to one SegMap whose
declared outputs and ABI include every write. The route mechanically materializes host control from
the typed equations, then derives SegMap/SegRed directly from the retained algorithm. JVM SIMD and
OpenCL therefore consume the same facts without backend re-analysis. Unknown/effectful host scalar
equations and unsupported parallel forms still decline explicitly at the front-end boundary.
OpenCL emits the tuple-valued SegMap as one multi-output kernel; the current JVM SIMD leaf consumes
the same scheduled equation without compatibility re-lowering but scalarizes the secondary store.

Pure host scalar dependencies now use the same ordered SSA spine through a typed `scalar` equation.
The first production subset covers statically typed literals/aliases and semantic array-length
queries, value-numbers `alength` of a produced tensor to its certified extent, and materializes the
host spelling mechanically. Map/reduce captures also distinguish pointwise element operands,
stable tensor reads and true scalar parameters. This lets the ordinary two-layer `predict-fn`
retain typed dense→ReLU fusion without conflating weights or inner-reduction activations with the
outer map shape, and preserves buffer versus scalar ABI roles through JVM/OpenCL lowering. Scalar
result dtypes come from retained walker/TypedClojure tags; `AbstractValue` transports those dtypes,
while the scheduled operation—not tensor rank—declares whether an ABI value is a scalar or buffer.

This representation is shared before backend selection. SOAC fusion changes the program consumed
by both JVM and accelerator lowerings; SegOp and schedule conversion then specialize it. The scalar
spelling for the migrated subset is generated from TypedSOAC rather than retained source, and JVM
SIMD/GPU reuse its certified SegOps. Non-escaping reduction results remain logical rank-zero values
whose `:resident-scalar-buffer` representation drives one-element device allocation and stable
consumer loads; dependent scalar equations are inlined as a typed transform. The former raw-source
resident rewrite and typed-route opt-out are deleted. The analyzed front end now constructs this
TypedSOAC subset directly. Certified inclusive and exclusive scans also cross the same typed
algorithm and scheduled-program boundary, retaining explicit destination, result-layout, and
graph-owned temporary-storage contracts. Migration is complete when hardware-costed multi-consumer
fusion, the remaining parallel forms, and covered backend compatibility re-lowerings are deleted.
Nested scalar folds inside a retained map are now explicit typed scalar-region terms. Monoid-shaped
folds carry an `AssociativeScan` certificate and an implementation-defined association contract, so
the JVM may select its multi-accumulator SIMD reduction. General recurrences carry an ordered
association contract and lower to an exact scalar loop. Portable C-family emission consumes the same
term as a sequential fold; future subgroup/workgroup schedules must be selected from its certificate
rather than rediscovering a reduction from source spelling.

Sequential control around a parallel algorithm is represented separately as the Pattern-declared
`TypedStructuredControl` dialect. Its canonical loop is a typed fixpoint around one closed
`TypedSOAC` body: it binds an integer induction value, ordered invariants, and ordered loop-carried
scalars or tensors, and gives each final result a distinct outer SSA value. The minimal body input
boundary is an ordered subset of those declared binders, so an unused induction value or invariant
does not require fake scalar work. The body boundary,
effects, and `AbstractValue` contracts are checked modulo the explicit outer-to-inner alpha-renaming;
zero iterations therefore retain the initial value contract without emitter inference. Values
defined and consumed only inside the body are ordinary SSA scratch, not entries in another memory
registry. The enclosing AbstractValue environment is retained in the loop facts, so scheduling and
runtime-call construction revalidate the same boundary instead of trusting a transient front-end
check. Repeated outer operands are ordinary SSA uses; loop results alone must be fresh definitions.
The association contract is explicitly sequential.

This does not add a time-loop SOAC or a PDE-specific primitive. SOAC fusion continues to reason
algebraically about the parallel body. The first lowering must preserve the fixpoint as explicit
host repetition of a target-neutral scheduled `KernelGraph`; a persistent workgroup loop is a later
schedule selected only when its participation, memory, and synchronization proofs hold. Once this
vertical consumes scientific time loops, the source-shaped compound detector and handwritten local
emitter are deleted rather than retained as a compatibility route.

The one-iteration graph is re-derived during validation from the retained TypedSOAC algorithm and
scheduled SegOps; buffers, storage contracts, uses, hazards, and dependencies cannot be supplied as
independent claims. Target emission may replace only each scheduled operation. Every resulting
`KernelArtifact` retains the complete immutable operation it implements, while graph validation
requires the pre/post-emission dataflow contracts to be identical. A copied operation ID is not an
emission certificate.

Scalar ABI dtypes are likewise semantic facts, not C-spelling inference. One canonical projection
from retained dtype metadata drives OpenCL, CUDA/HIP, and host binding; target code generation maps
that dtype to native syntax only afterward. In particular, a 64-bit loop induction value remains
64-bit while an explicitly 32-bit array bound remains 32-bit, and runtime call construction checks
both against the retained structured-loop AbstractValues.

Functionalizing a mutating host loop is deliberately proof-gated. A relational host
`AbstractValue` refinement records shape equalities exposed by typed length queries, clones,
same-shaped allocation, and pure maps. A primitive copy becomes a loop-carried state transition
only when that analysis proves zero offsets, the complete rank-one extent, and equal logical
dtypes; partial or offset copies remain effects and decline this route. The refinement consumes the
canonical dtype facets and typed `AbstractValue` seeds rather than introducing another operator or
type registry. This is the initial rank-one proof needed by the scientific-loop vertical, not the
eventual N-dimensional shape solver.

### 3.3 Schedule and transform IR

A schedule is an immutable transformation program over the functional parallel
IR. It is more than a bag of kernel kwargs. It can express:

- split, tile, reorder, fuse, compute placement, and materialization;
- thread/block/subgroup mapping;
- vectorization and matrix-instruction selection;
- operand layout and explicit layout conversion;
- staging space, copies, pipeline depth, and barriers;
- reduction decomposition and split-K;
- buffer donation, residency, and graph capture;
- device/mesh placement and resharding policy;
- time integration scheme, operator splitting, subcycling ratios, task ordering and
  iteration budgets of data-dependent loops;
- fidelity level, coarse-graining map, marginalized sub-models, multi-level estimator
  selection, surrogate substitution, and checkpoint/recompute schedule across time steps.

Time and fidelity are scheduled axes (§3.8). Their legality rules are conservation, symmetry,
coupling and error-model contracts rather than resource limits, but they enter the same
legality → feasibility → analytic rank → measured rank sequence, and a failed legality check never
changes the semantic program.

Halide supplies the algorithm/schedule separation and structural search model.
MLIR's Transform dialect supplies a useful safety model: typed handles,
preconditions, declared effects, and recoverable versus irrecoverable transform
failure. Raster can implement those properties in EDN and Clojure records.

Applying a schedule produces a candidate scheduled kernel graph. Candidate
generation, legality, feasibility, profitability, and measurement remain
separate:

```text
rewrite legality → hardware feasibility → analytic rank → measured rank
```

A cost model may abstain. A failed legality or resource check is never converted
to a different semantic program.

Runtime selection and offline tuning operate on an emitted executable
alternative, not necessarily one kernel. An alternative is either a single
`KernelArtifact` or an emitted `KernelGraph`. All alternatives in one
`KernelDispatch` share one ordered external ABI, compiler argument order,
target, and logical effects; graph topology, entry points, launch geometry,
derived scalars, and private temporaries are schedule-owned and may differ.
This is what permits a direct contraction and a split-K partial-plus-combine
graph to compete without making split-K storage part of the user-visible call.

The first resident GEMM slice uses this boundary directly. Compilation emits scalar f32, direct
mixed-precision XMX, and split-K graphs with one `(A B C M N K)` ABI. Checked integer-expression
IR derives the pitch gate, occupancy decision, private conversion/transpose/partials storage,
`KC`, and 1–3D launches from shape and schedule data. The resident binder supplies ABI values and
binds ordinary graph calls; it does not assemble a GEMM algorithm. Constant-only layout/conversion
nodes are hoisted by a graph-generic cacheable-transform rule. Remaining work is to originate the
same graphs from the canonical typed contraction route, add vendor matrix-instruction leaves, and
make measured shape tables override the analytic selector.

### 3.4 Scheduled kernel graph and kernel IR

The scheduled graph contains calls between kernels and explicit resident
buffers. Each kernel contains typed operations and regions sufficient to
represent:

- N-dimensional program and thread indices;
- named register, shared/local, cache-resident, and global layouts;
- masked loads/stores and boundary values;
- block views/pointers, strides, offsets, gathers, and scatters;
- scalar, subgroup, block, and grid reductions/scans;
- matrix/dot operands and accumulator fragments;
- shared-memory allocation and swizzles;
- asynchronous copy, commit, wait, and barrier;
- atomics;
- structured control flow;
- ordered kernel inputs, outputs, scalars, aliases, and temporary buffers.

SegMap, SegRed, and SegScan should migrate into this real dialect. Histograms and
the missing result forms should be added by workload demand.

Layouts should initially be a finite, named family with stride, permutation,
tile, vector, and swizzle parameters, plus explicit legal conversions. This is
enough for DPAS, WMMA/WGMMA, and MFMA families while Raster delegates register
allocation and final instruction scheduling to vendor compilers. A general
Triton GF(2) linear-layout algebra is not a prerequisite.

The kernel ABI is one ordered, typed slot vector:

```clojure
[{:name 'x   :kind :input  :dtype :f16 :shape [m k] :layout x-layout}
 {:name 'w   :kind :input  :dtype :q4-k :shape [n k] :layout w-layout}
 {:name 'out :kind :output :dtype :f32 :shape [m n] :layout out-layout}]
```

The signature renderer, binder, verifier, cache key, profiler, and debugger all
consume this same vector. Unordered input/output symbol sets and independently
maintained argument counts must disappear.

### 3.5 Dialect conversion and verification

Raster should adopt MLIR's conversion discipline without requiring MLIR as a
runtime dependency:

- a conversion target declares legal, dynamically legal, and illegal
  operations;
- full conversion succeeds only when no illegal operation remains;
- partial conversion is named explicitly and cannot masquerade as a completed
  dialect;
- type/layout conversions insert explicit adapters;
- each operation supplies interfaces/traits, avoiding repeated exact-op
  conditionals;
- every pass declares preserved analyses and invalidates the rest;
- pass diagnostics retain source and semantic value identities.

The immediate implication is that SegOp lowering may no longer warn and return
`nil`. Unsupported forms produce a structured diagnostic naming the operation,
source, missing rule, target dialect, and possible legal fallback. A fallback
is an explicit conversion rule, not exception handling.

Native MLIR remains an optional future interchange or target boundary. Raster
should not take a mandatory LLVM/MLIR/Triton build dependency merely to gain
dialect concepts it can express in its existing nanopass system.

### 3.6 Target lowering

Target backends select instruction families and render source/binaries from the
same verified kernel IR. Vendor differences that change the structure of a
kernel—DPAS versus WMMA/WGMMA versus MFMA—fork in target lowering. Dtype
spellings and instruction variants within a family are data tables.

The vendor compiler continues to own register allocation, low-level instruction
scheduling, and binary generation until measurements demonstrate that this
boundary is the limiting factor.

A direct PTX route, when justified by those measurements, is a late NVIDIA target dialect beneath
`KernelBody`, not a second scheduling IR and never a shortcut from SOAC. Full conversion first
legalizes a verified body to NVIDIA address spaces, predicates, cache policies, vector memory
operations, `cp.async`/`mbarrier`/TMA dependencies, exact MMA/WGMMA operand tuples and resource
directives; a separate renderer produces PTX and `ptxas` produces cubin. CUDA C remains a broad
differential backend. PTX gives Raster tighter instruction selection and memory-ordering control,
but physical registers, spills and final SASS scheduling remain vendor-compiler responsibilities.
The artifact records `ptxas` register, spill, stack and shared-memory reports so feasibility and
autotuning can use actual resource outcomes.

### 3.7 Executable artifact and runtime

`Compiled` evolves from a whole-program wrapper into a composable executable
graph:

- target module and kernel cache;
- typed input/output tree specifications;
- device values with shape, dtype, layout, placement, and ownership;
- ordered kernel ABI bindings;
- explicit aliases and donation;
- captured constants and parameters;
- events and asynchronous dependencies;
- schedule, hardware signature, compiler version, and measurement provenance;
- graph instantiation/linking without recompiling shared kernels;
- serialization of all non-live-resource state.

Execution scheduling is a distinct backend-neutral lowering. An `ExecutionPlan`
assigns operations to logical queue classes and connects them with logical
wait/completion events. Compiler events are stable identities, never OpenCL,
Level Zero, MPI, or vendor-library handles. A backend may initially realize the
plan on one in-order compute queue, but it must retain the dependency DAG so
later compute/transfer/collective overlap is a scheduling change rather than an
ABI rewrite.

The runtime event contract is submit, nonblocking status, host wait, and safe
release. Completion owns the lifetime of every referenced graph, kernel,
allocation, and view; destruction must first establish completion. Status
observation alone is not assumed to establish host memory visibility.

The first transfer realization uses that same public event contract for
bounds-checked batches of host/resident ranges. A completed transfer reports
bytes, command count, elapsed time, and timing provenance. OpenCL uses owned
native staging and device-event timestamps; a caller may reuse upload sources
after submission, while download destinations become visible only after the
host wait. Level Zero's current shared allocations complete their Panama copy
inline and report host-monotonic timing explicitly rather than pretending that
a device copy event exists. Logical compute and transfer queues initially map
to distinct physical in-order OpenCL queues for explicit range submissions.
Callers await a transfer event before the same buffer changes queue roles;
unrelated immutable transfers may overlap compute. Automatic cross-queue wait
edges for a mixed execution plan remain future lowering work. Immediate-wait
measurements are the calibration path; `host-wall-ns` otherwise includes any
intentional host work between submission and wait.

Device-to-device binding is the normal path. Host transfer is an explicit graph
edge with a reason and byte count. The compiler's memory plan owns temporary
liveness, reuse, alignment, and peak-memory accounting.

A running cluster is driven from a REPL. Changing a fidelity level, a decomposition, a coupling
period or a tile must be a plan hot-swap on live artifacts, not a whole-program rebuild, so
compilation must be incremental at the granularity of plan nodes, with unchanged kernels and
artifacts reused by identity. A forkable execution context, as in Spindel, is the mechanism for
trying a plan change on a branch and committing or discarding it with its measurements.

JAX pytrees are the model for separating user structure from flat leaves, but
Raster trees should retain stable keyed paths and type/shape information. JAX
donation and sharding contracts are also useful references; Raster's existing
ownership checks should remain fail-loud.

### 3.8 Time, composition and rate scheduling

Algorithm/schedule separation applies to the time axis. The semantic layer states a dynamical
system: right-hand sides, conservation laws, symmetries, constraints, coupling ports, and the
observables that ground it. The schedule layer states how it is integrated: explicit, IMEX or
implicit tableau; operator splitting; leapfrog or multistep history; subcycled nesting with
per-level time ratios; task ordering and retry; iteration budgets and reduced exit conditions of
data-dependent loops; checkpoint and recompute placement for adjoints. Production codes show
these as three schedules of one semantics: a per-block task list with dependency masks, a
recursive nesting scheduler with per-level Δt, and a tableau-driven stepper with an implicit
column sub-solve. Raster does not fix an integrator. It fixes that the integrator is schedule data
with legality rules and an error model.

Composition of models with different rates and different grids is a scheduling problem, not a
scene graph. The pattern is shared by coupled Earth-system components, whole-cell models that
partition one state among ODE, stochastic and linear-programming sub-models each step, and
molecular dynamics with a mesh solver on its own decomposition. The plan therefore needs:

- a rate scheduler over systems with declared read/write sets and periods, in the sense of an
  entity-component schedule, expressed as SOACs over entity-set shards with explicit effects;
- a coupler node: several meshes with owner maps, remapping weights as a content-addressed sparse
  contraction with a pinned reduction order, conservative fractions, accumulation between coupling
  intervals, and coupling periods;
- a partitioned-shared-state primitive: request, allocate, run, merge, with an exact-sum
  conservation contract for contended quantities;
- data-dependent loops with a reduced exit condition and deterministic ordered reductions, so
  iterative solvers lower and their iteration counts are priced.

Open dynamical systems composed along ports are the specification of what composition must
preserve. Raster uses that algebra to state legality and certificates; it does not require users
to write categorical syntax, and it does not adopt a scene hierarchy or an entity DSL.

### 3.9 Irregular values and communication

The simulator survey found that dense stencils and contractions are two of ten computational
patterns in production simulation; the others are unstructured-neighbour stencils, particles and
populations, event-driven sparse updates, spectral transposes, multi-rate composition, coupled
grids, adaptive refinement and differentiable rollouts. Five primitives cover them without
prescribing any simulator:

1. **A star-forest communication node.** Root and leaf index sets, a combiner monoid with a
   deterministic-order attribute, split issue and wait phases, and periodicity as a per-pair
   coordinate shift. It subsumes N-D halos with per-axis periodicity and widths, corner policy
   and staggered centering; accumulating reverse halos (direct stiffness summation, deposition,
   force return); redistribution and pencil transposes; all-to-all-v; population migration; and
   delayed delivery, which is a halo along the time axis.
2. **Irregular values.** Ragged arrays and CSR with offsets as first-class values; connectivity
   tables with source dimension, local dimension, maximum arity and skip value; shards as
   entity-id sets with an ownership function or table. The SOACs over them are gather-contract
   (a gather fused into a weighted reduction), scatter-reduce with a declared conflict algebra,
   segmented irregular reduce over runtime segment sets, and compaction. A pass that tiles a
   sparse neighbour structure into small dense contractions with masks, as cluster-pair
   molecular-dynamics kernels do, is the highest-value single lowering.
3. **Access facts on edges.** Each dependency carries subset, volume and write-conflict facts
   that propagate through scopes, so halo widths are derived from operator access, fusion
   legality is subset intersection, transfer bytes are exact, and in-place reuse is proved rather
   than inferred from spelling.
4. **A schedule for time** (§3.8), whose simulator prices live memory under a recompute policy,
   pipeline bubbles, subcycle ratios and iteration counts, and ingests measured per-task costs.
5. **Contracts as certificates.** Adjoint prolongation/restriction pairs or flux registers for
   refinement, ordered-reduction attributes that survive lowering, invariance of results under
   re-layout, index rotation and restart, tolerance-envelope oracles from perturbation ensembles,
   non-field state in the durable manifest, and stateless counter-based random number generation
   so checkpoints carry no generator state.

Population and particle values are irregular values with a pure `position → region → owner`
function, so migration is derived, plus a spatial index as a plan node with static-shape overflow.
Region trees are IR values.

## 4. Quantized computation

Quantization should be represented as contraction semantics plus an operand
storage layout, not as one opaque hand-written kernel per format. A format
descriptor must define:

- block shape and byte layout;
- scale, offset, and auxiliary metadata encoding;
- logical element-to-byte index map;
- decode computation and signedness;
- permitted activation and accumulator dtypes;
- reference pack/unpack implementation;
- legal instruction families;
- numerical tolerance and golden vectors.

Q4_0, Q8_0, Q4_K, and Q6_K establish the pattern. The next formats should be
selected from actual pretrained and finetuning model requirements and measured
quality/performance—not from a goal of mirroring every `llama.cpp` enum. Likely
candidates include high-use IQ variants and current low-bit floating formats
such as MXFP4/NVFP4, subject to model evidence and hardware support.

`llama.cpp-new` is the interoperability and CPU-reference oracle. Every added
format needs pack/unpack parity, randomized boundary cases, odd-tail coverage,
and end-to-end logits or loss comparisons. A format is not complete when only
its isolated emitter test passes.

## 5. Hardware-aware compilation and autotuning

Raster's existing hardware descriptor and schedule cache form the right base.
Runtime probes and catalogues may retain backend-native names, but their compiler projection
must normalize the execution hierarchy. In particular, the descriptor records a set of supported
subgroup/warp/wave widths separately from its preferred width, maximum workgroup geometry, and
per-field provenance. A schedule chooses one supported width; it must never inherit another
vendor's default. Matrix-operation scope remains an independent capability because a device may
support several subgroup widths while one DPAS, MMA, or MFMA operation requires exactly one.

The closed loop is:

```text
program + abstract values + hardware descriptor
    → legal schedules
    → resource-feasible schedules
    → analytic shortlist
    → compile and device-time benchmark
    → stationary winner
    → versioned cache and Compiled artifact
```

This is a staged JIT/runtime contract, not only an ahead-of-time compiler pipeline. Static analysis
may leave symbolic shapes, strides, placement choices and schedule alternatives in a partially
specialized artifact. At binding or execution time, Raster may propagate newly known shapes,
residency, allocation/topology, backend capabilities and calibrated measurements back into
specialization, candidate generation and tuning. Adaptation always returns or selects a newly
verified immutable artifact; it never mutates a running kernel around type, effect, ownership,
numerical or resource checks. Logical events, measurements, hardware descriptors, `Compiled`,
`LinkPlan` and `ExecutionPlan` are the compiler/runtime seam to OpenCL, Level Zero, CUDA/HIP and
future collective or cluster runtimes.

The search state should include both kernel and graph choices. Per-kernel axes
include tile, vector width, instruction family, workgroup geometry, layout,
staging, pipeline depth, and reduction strategy. Graph axes include fusion,
materialization, parameter/activation residency, recomputation, capture, and
placement.

The benchmark protocol must record:

- warmup and cache-state policy;
- device-event time, not host submission time;
- minimum and distribution summary;
- stationarity/variance result;
- compilation time and binary size;
- peak memory and temporary bytes;
- transfers and bytes moved;
- hardware, driver, compiler, program, schedule, and numerical-mode hashes.

Analytic models seed and prune; measurement decides among survivors. Learned
cost models may later replace parts of ranking, but the raw observations remain
the source of truth.

## 6. Distributed and resource-aware programming model

Distribution belongs in the abstract value and scheduled graph, not in a
separate orchestration wrapper. Shards are entity-id sets with an ownership function or table,
of which rectangular one-axis partitions are the simplest case; shard boundaries are plan state
that a re-plan may move on measured cost, as dynamic load balancing and regridding require.
Halos carry a combiner and are instances of the star-forest node of §3.9. The program model needs:

- a device mesh and topology descriptor;
- sharding annotations on logical dimensions;
- sharding propagation rules per operation;
- explicit reshard, all-reduce, all-gather, reduce-scatter, broadcast, and
  point-to-point operations;
- compute/communication dependency events;
- memory capacity and bandwidth constraints per device/link;
- heterogeneous placement and legal dtype/layout capabilities;
- mesh-axis to tensor-axis bindings so several shardings coexist on one value;
- pipeline stage and microbatch coordinates on steps;
- a recompute and offload policy with live-range memory accounting;
- redistribution and all-to-all-v as first-class steps.

Compilation minimizes a cost vector rather than a single kernel time:

```text
latency, throughput, peak memory, transferred bytes,
compile/tune budget, energy when measurable, floating-point error,
approximation error under an explicit error model, and information value
```

The analytic simulator is a seed. It must price live memory under the recompute policy, pipeline
bubbles, subcycle ratios and iteration counts, and it must ingest measured per-task costs keyed
by plan node and device signature so that re-planning on measurement is an ordinary transform.

The compiler hierarchy extends upward without making the single-device `LinkPlan` a cluster object:

```text
semantic program and AD
    → verified mesh, topology, resource envelope and allocation
    → distributed schedule with explicit sharding and communication
    → certified DistributedPlan with shard-local programs and cross-device events
    → one LinkPlan and ExecutionPlan per device
    → KernelGraph / KernelBody / target artifacts
```

`DistributedPlan` is the missing bridge. It owns global-to-shard value mappings, collective groups,
point-to-point routes, resource claims, per-device plans and a witness for shard coverage,
replication, ownership, capabilities and collective agreement. Native communicators and event
handles remain runtime resources. An outer `WorkloadPlan` may handle admission, continuous batching,
deadlines, retries, checkpoints and safe reallocation for dynamic services; it selects certified
compiled variants rather than injecting service policy into SOAC or kernel IR.

The first executable part of that bridge now exists for point-to-point schedules. A rectangular
`DeviceMesh` is embedded in a directed topology whose devices declare memory capacity and whose
links declare bandwidth and latency. Concrete `AbstractValue` shapes carry either exact replication
or one-axis partitioning; certification proves full coverage, absence of gaps/overlaps, ownership,
and placement in the mesh. Ordered compute and transfer steps form a fail-loud dependency DAG.
Its analytic simulator serializes compute per device and transfers per directed link, while allowing
the two resource classes to overlap, and reports makespan, per-device peak memory, link bytes, and
total transferred bytes. The re-derived certificate contains those costs and shard/route witnesses.
This is deliberately a planning model: measured costs may replace analytic durations, and Datahike
may retain its immutable plan/measurement history, but driver buffers, communicators, and events do
not enter the IR. Collective agreement, halo regions, and an initial block-structured AMR layer are
described below.

Collective agreement is now an explicit extension of the same plan. `all-reduce`, `all-gather`,
`reduce-scatter`, and `broadcast` retain their semantic kind, group, value, and root/reduction
contracts; reducing collectives require the existing certified associative algebra rather than an
operator name guessed by a communicator backend. A separate `CollectiveSchedule` selects an
algorithm and ordered communication rounds. Each round expands mechanically to routed transfer
steps, parallel legs cannot claim the same directed link, and the certificate retains both the
semantic operation and its schedule. This makes ring/tree/native implementations replaceable while
the dependency DAG and topology simulator account for their actual bytes and routes.

Axis-partitioned scientific values now use the same machinery for nonperiodic halo exchange. A
semantic `HaloExchange` names the value, partition axis, halo width, and boundary policy. Scheduling
derives adjacent owned shard faces, exact source rectangles, dtype-sized byte counts, and both
directed routes; those facts are retained on transfer steps and in the distributed certificate.
The first contract intentionally supports nonperiodic one-axis partitions. Periodic boundaries,
multidimensional decomposition, and overlap lifetimes still need explicit semantics rather than
being smuggled through generic transfer attributes.

Block-structured adaptive meshes now have an explicit outer `AMRWorkloadPlan`. A certified
`RefinementHierarchy` retains rectangular patch identity, level coordinates, per-axis ratios,
alignment, non-overlap, and a proper-nesting margin. Each patch binds one durable
`NumericalStateManifest` field to one fully owned distributed value. Semantic prolongation and
restriction retain exact source/target rectangles, compatible storage contracts, and declared
operator requirements; planning mechanically expands cross-device cases to dtype-sized routed
transfers followed by target-device compute with explicit read/write roles. The outer certificate
incorporates the complete durable-state and `DistributedPlan` certificates, not merely their
user-selected identities. This composition is intentional: mesh semantics do not become
opaque transfer attributes, while storage placement and fabric resources do not leak into the
hierarchy.

This first AMR schema is not yet a claim of executable numerical operators or conservative,
subcycled AMR execution. It is cell-centred, joins adjacent levels, and has explicit
`:hierarchy-only` and complete-patch `:transfer-cycle` modes. A conservative
hyperbolic/PDE vertical must next represent level time ratios, flux-register contributions,
reflux ordering, and average-down dependencies, then validate mass/lake-at-rest and restart
oracles on a non-trivial 2-D workload. Version 1 also deliberately requires the full proper-nesting
margin to fit one parent patch; it has no physical-boundary exemption.

The first distributed target should be data-parallel training with explicit gradient all-reduce,
followed by tensor/sequence sharding for a transformer block and a scientific halo-exchange case.
Pipeline and expert parallelism should wait until the value, sharding, communication and failure
contracts can state them without runtime side protocols. MPI, NCCL/RCCL, oneCCL, UCX and eventually
GPU-initiated transports such as NVSHMEM are interchangeable execution backends, not Raster's
semantic memory model. Datahike may retain durable topology, plan, measurement and decision history;
it does not replace hot-path allocation or communication.

Durable numerical state follows the same separation. A certified storage-neutral
`NumericalStateManifest` binds logical `AbstractValue` fields to immutable content-addressed chunks,
lineage, numerical compatibility and provenance. Store placement remains a runtime decision: a
local mmap/Konserve or LMDB frontend may be backed by S3, Ceph, a parallel filesystem or an
institutional archive. Datahike publishes the semantic state only after required durability
receipts; direct fabrics still move hot halos, gradients and activations. See
[`durable-numerical-state.md`](durable-numerical-state.md).

## 7. Reflection, inference through simulator internals, and learning

This section is a track, not an appendix. Inference is the driver of §0, and inference that uses
simulator internals is exactly reflection: the program is data, so automatic differentiation,
linearization, coarse-graining, marginalization, multi-level estimators and emulators of internal
state are certified transformations that an inference procedure may request. The track's
demonstrator is fixed: one simulation-backed inference task solved by black-box sequential Monte
Carlo or approximate Bayesian computation on the simulator, and again by procedures that use its
internals through Raster transformations, with calibration checked by simulation-based
calibration and the cost vector reported for both. Learned proposals, including amortized
proposals from pretrained models, compete in shadow mode under the same calibration gate.

Raster can make reflection unusually powerful because Clojure data is a natural
representation for programs and schedules. The safe version has four rules:

1. Source operations, semantic values, SOAC nodes, kernel nodes, and measurements
   retain stable identities across lowering.
2. IRs, transforms, schedules, hardware descriptions, and measurements are
   immutable, content-addressed, inspectable data.
3. Every structural change is a declared transform with preconditions,
   provenance, and a verifier; rejected changes leave the payload unchanged.
4. Learned proposals compete in shadow mode before they may be cached or shipped,
   and no proposal bypasses legality, resource, or numerical gates.

This permits learning across representations: a model can associate source
structure, SOAC structure, chosen schedules, emitted kernel features, hardware,
and measurements. It does not require unrestricted mutation of compiler code or
live executables. Rewrite policies and cost models are replaceable artifacts;
the trusted verifier stays small.

## 8. Landing order

Each increment should be a thin vertical slice with a production-path test.

The immediate continuation after the verified double-buffered weighted-reduction route is:

1. **Landed:** define a typed SOAC S-expression dialect with `../pattern` and differential-test
   map→map, map→reduce and horizontal-map fusion against the current graph.
2. **Landed:** carry that dialect in the existing program/value envelope and route one ordinary
   fused reduction through JVM SIMD and GPU `KernelBody` without reconstructing compiler facts from
   source spelling.
3. **Landed:** add a finite verified shared-memory swizzle family and bank-conflict/resource model;
   keep the current 1-D attention stages on its identity member until measured 2-D staging lands.
4. **Landed:** make stage count and swizzle finite measured schedule axes, retire the legacy tuner,
   and remove the attention-provenance gate from routed weighted-reduction lowering.
5. **In progress:** finish typed parallel coverage and scalar JVM scheduling, then remove compatibility
   re-lowering. Supported map/reduce programs, including unfused and host-visible intermediates,
   horizontal tuple maps, typed scalar/shape equations, stable tensor captures, and resident
   reduction results and certified inclusive/exclusive scan now share the direct
   analyzed-source→TypedSOAC→SegOp production route. Both scan modes additionally reach an emitted
   graph-backed resident executable through ordinary dispatch, LinkPlan and binding, and a generic
   per-call staging graph runner executes the same registered dispatch. Pointwise read/write map
   destinations lower to one physical `:inout` ABI result and execute through both resident and
   staged binding without an aliased compatibility input. Independent effect-only tuple maps also
   carry an ordered logical-result-to-physical-storage contract and compile through their scheduled
   SegMap on resident OpenCL execution. Their shared typed scalar computations use one explicit
   ordered local-SSA region that lowers once through JVM and GPU paths; vertical and horizontal
   fusion now alpha-rename and compose those regions without duplicating shared scalar work.
   Explicit typed segmented-reduction result transforms now reach contraction store regions without
   compatibility re-lowering; the typed fusion pass discovers adjacent single-use result maps and
   removes their physical intermediates without rewriting the reduction fold. Both the matrix
   schedule and the portable sequential-segment schedule lower that closed transform to typed
   KernelBody scalar SSA, including capture loads and explicit precision conversions. Matrix
   storage parameters retain dense physical layouts independently from matrix-fragment lane
   layouts; a transform that reuses A/B refers to the existing compiler identity instead of
   duplicating its pointer. The matrix target ABI is projected from the verified KernelBody and its
   stable-read contracts, and its store emitter consumes only typed SSA—no source expression or
   target-local type inference remains on that path. The portable path
   emits unchanged through OpenCL, CUDA and HIP and is compiled by the hardware-free CUDA/HIP CI
   gates. The register-tiled family now also consumes verified contraction facts directly and
   expresses cooperative dense staging, workgroup barriers, its K loops and register microtile as
   scalar/control KernelBody. Every microtile store receives an alpha-renamed instance of the same
   typed result transform, while the body-projected ABI deduplicates shared captures. The former
   handwritten OpenCL register-tiled generator is deleted; its device numerical oracle now invokes
   the scheduled body, and hardware-free CI compiles that body with both nvcc and hipcc. A finite
   descending tile family is filtered by the target workgroup and shared-memory limits before
   emission; an explicit infeasible tile or a target with no legal candidate declines honestly.
   Hardware-costed multi-consumer placement now uses one target-neutral roofline policy shared by
   the TypedSOAC and compatibility graph routes. Typed fan-out values carry an explicit,
   serializable recompute/materialize witness (fanout, dtype, registered scalar work, abstract-
   machine ridge, threshold and reason); cheap producers may be cloned into consumers while their
   equation remains live, whereas a proven-expensive producer stays materialized. A host-visible
   value remains an observable result even when dependency removal subsequently permits one
   tuple-valued horizontal kernel. Ordinary compilation and direct session scheduling both pass
   the same device-stripped Abstract Machine into this decision, and a missing machine retains the
   typed materialization boundary rather than speculating about duplicated work.
   Stable indexed gathers and explicitly unique guarded scatters also use this typed scheduled-map
   route, including resident block transfers. The public flat `par/gather` spelling now canonicalizes
   to that ordinary typed map: JVM scheduling recognizes an exact stable indirect read and selects
   hardware `vgather`, while the portable scalar/control KernelBody emits the admitted scheduled
   SegMap subset across OpenCL, CUDA and HIP. Canonical one-index/one-carry scalar loops preserve
   their ordered semantics as KernelBody `ForLoop`; non-canonical control remains a checked decline.
   Kernel-local C names are fresh with respect to the typed ABI, so an index buffer cannot shadow
   the launch coordinate. Flat
   additive reducing scatters also carry a checked conflict algebra through this boundary and select
   exact JVM or atomic OpenCL schedules. Strided gather and additive scatter flatten into the same
   algebra with one hoisted product extent; gather remains a map, while generic effect analysis
   proves the scatter update before target lowering selects sequential JVM updates or GPU atomics.
   Direct flat gather, scatter-add and reduce-by-key calls now use that same singleton typed
   scheduling boundary instead of their handwritten OpenCL generators. Indirect tensor indices
   lower as explicit scalar SSA loads feeding later loads/stores, so the scheduled map remains
   portable across OpenCL, CUDA and HIP; hardware-free vendor gates compile public gather and
   scatter workloads. Direct strided gather and scatter now consume the complete typed
   mini-program: normalization's hoisted product extent remains an ordered host-scalar equation,
   and the backend consumes the following scheduled SegMap without reconstructing either fact.
   Direct maps with compound extents use the same complete mini-program route on GPU and JVM. The
   remaining singleton adapter accepts only a closed equation and fails with
   `:direct-operation-requires-program` if a caller would discard a host-scalar prefix; it never
   silently re-enters legacy SOAC to erase that dependency.
   A raw binding program handed directly to the GPU backend likewise enters this whole-program
   adapter once, preserving cross-equation typed facts instead of independently scheduling each
   child operation. The JVM SIMD direct entry does the same when its caller supplies an
   authoritative dtype; untyped compatibility calls remain counted rather than guessing types in
   the backend.
   Monolithic C/SIMD now preserves the `ParallelProgram` envelope through host-only length and
   memory-reuse passes. Direct map/reduction sites consume their already scheduled SegMap/SegRed;
   C ABI length legalization is a target expression transform, and the former C-only legacy-SOAC
   reconstruction is deleted. Direct JVM and GPU compatibility entry points use one middle-end
   singleton adapter that attempts analyzed-source→TypedSOAC first and falls back only for an exact
   structured source-admission decline. Scheduled SegMap storage selects the runtime marker from
   the emitted ABI: dense single-result maps remain value-producing, while unique scatters and
   fused multi-result maps use the general effect marker without reconstructing destination facts
   from source. Additional atomic monoids, privatized histogram schedules, nested
   structured C/SIMD sites, the remaining parallel forms, calibrated whole-graph placement costs,
   and deletion of the remaining graph/backend fallbacks remain.
6. Add a differential PTX target dialect/module boundary. Start topology and sharding values as a
   read-only distributed track without interrupting the kernel and typed-middle-end verticals.

The integrated roadmap has six cooperating tracks rather than one backend-only sequence:

| Track | Near-term completion gate | Unlock |
|---|---|---|
| Verified device scheduling | Add verified swizzles and measured stage depth to the landed double-buffered reduction | General staged reductions and FlashAttention-like schedules |
| Typed functional middle end | Pattern-declared typed SOAC program; map/reduce fusion reaches JVM and GPU from one fact set | Reliable fusion, batching, AD and reflection across representations |
| Portable peak kernels | Verified swizzles/layouts, unified matrix fragments and quant storage descriptors | Competitive GEMM, attention and scientific tiles across Intel/NVIDIA/AMD |
| Measurement and selection | One tuner over coupled graph/kernel axes with numerical and resource gates | Hardware-aware schedule selection and selective autotuning |
| Precise target lowering | Format-neutral target modules, differential PTX, later AMD-specific lowering | Exact async/matrix/cache control where portable source is insufficient |
| Distributed/workload planning | Typed mesh/topology/sharding values, then certified `DistributedPlan` | End-to-end node, cluster and data-center optimization |
| Simulation and inference vertical | One end-to-end workload with a fidelity transform, restart/branch and a parameter inference, early and on toy kernels | The objective of §0 becomes measurable before the backend is complete |

The table below is the durable architectural dependency ledger, not a claim that every increment is
unstarted. ABI, artifact composition and much of the kernel-IR foundation have landed; their stated
completion gates remain useful for finding legacy paths that have not joined the common route.

| Increment | Work | Completion gate |
|---|---|---|
| 1. Real SegOp boundary | Introduce a first-class program container for SOAC/SegOps; make lowering a full, fail-loud conversion; remove metadata-only and backend re-lowering for the migrated forms; validate operations and types, not only outer `let*`. | Map, reduction, scan, and one contraction travel through the same recorded dialect path on CPU and GPU; unsupported forms cannot silently bypass it. |
| 2. Kernel ABI | Replace unordered symbol sets and hand-maintained signatures/binders with one ordered typed ABI; derive emission, binding, cache identity, and diagnostics from it. | Every production GPU invocation compares the emitted and bound ABI structurally; syntax/compile checks cover every emitted kernel. |
| 3. Abstract values and operation interfaces | Add canonical shape/dtype/layout/placement/ownership/effect facts and operation rules; flatten/unflatten structured arguments with stable paths. | Passes no longer infer these facts from spellings or independent maps; differential type/effect tests cover nested and polymorphic shapes. |
| 4. Scheduled contraction slice | Apply an explicit schedule to typed segmented contraction; bring one GEMM family through kernel IR with masks, named operand layouts, and fusible epilogue. | The same algorithm selects scalar, tiled, DPAS/DP4A, and quant leaves by legal schedule; emitted results match the flat semantic oracle. |
| 5. Composable artifacts | Lift executable/session ownership, add foreign `DeviceArray` rebinding, graph linking/instantiation, parameter capture, events, and N-D trees. | Two compiled transformer subgraphs compose with zero host transfers; a full forward/VJP/update step is one resident artifact. |
| 6. Closed tuning loop | Connect real device timing to schedule candidates and graph choices; make cache/provenance complete; wire or explicitly classify every schedule axis. | A cold compile tunes once, a warm compile reproduces the winner, numerical validation precedes timing, and the artifact explains why it won. |
| 7. Workload-driven coverage | Add stream/hist/gather/scatter, masking/block views, atomics, layouts/staging, and quant formats as demanded by model/scientific kernels. | Coverage is measured through production lowering on a published workload corpus, with differential and device compile/run gates. |
| 8. Distributed IR | Add mesh/sharding abstract values, collective nodes, communication cost, and placement scheduling. | A transformer training step runs data-parallel on multiple devices without hidden host copies and reports compute, communication, and memory costs. |

The tracks interleave. The simulation and inference vertical does not wait for increments 7 and 8;
it lands on toy kernels as soon as increment 1 is closed, so that fidelity axes, error models and
calibration gates are exercised while the backend matures. Its ladder, each rung one
production-lowered kernel, one certified plan and one oracle, with the reference kernel named in the
survey:

| Rung | Demonstrator | Primitives | Oracle |
|---|---|---|---|
| D1 | 2-D shallow water, 2-D decomposition, periodic and wall halos, coarse-graining transform, restart and branch, one parameter inference | star forest, access facts, contracts | mass, lake-at-rest, convergence, bitwise invariance under re-layout/rotate/restart, calibration |
| D2 | Icosahedral divergence or spectral-element gradient with direct stiffness summation | irregular values, combiner halo | reference values, rank-invariant ordered sums |
| D3 | Smoothed-particle pair density or Lennard-Jones over cell lists with ghost exchange | populations, star forest | brute-force oracle with per-field tolerance table |
| D4 | Spiking-network spike delivery with delays: ragged send table, ring-buffer scatter, all-to-all-v | irregular values, time halo | exact-integration neuron reference |
| D5 | Pencil transpose and FFT inside a distributed pressure solve with a reduced exit | redistribution, data-dependent loop | reference solver |
| D6 | One transformer layer under tensor and data parallelism with sharded optimizer state, certified collective sequence, simulated with recompute memory and a pipeline bubble | time schedule, access facts | collective trace and measured two-device run |
| D7 | Whole-cell-style partition/merge of ODE, stochastic and linear-programming sub-models under a versioned outer loop | rate scheduler, coupler, contracts | exact-sum conservation, queryable plan lineage |
| D8 | Key/value continuation state: prefix index over the content chain and a cross-node transfer plan | star forest | hit rate and transfer bytes against serving systems |

D2 to D4 are the test of the claim that one substrate covers simulation, because they are the
three patterns that are not dense. D6 is the honest ceiling for training on measurable hardware.

The first two increments are correctness infrastructure and should precede
additional backend breadth. Increment 5 is the first major model-execution
unlock. Increment 6 makes the hardware-aware work a closed system instead of a
collection of useful components.

## 9. Scientific and model acceptance ladder

Every tier compares semantics first and performance second:

1. generated small map/reduce/scan/stencil programs across interpreter, JVM,
   C/SIMD, and available GPU backends;
2. reductions with nontrivial identities, ragged sizes, aliases, multiple
   outputs, and exceptional values;
3. contractions over permutations, batching, split-K, transposes, layouts,
   epilogues, and supported quant formats;
4. FFT/PDE/sparse or irregular scientific kernels that force the non-GEMM
   coverage;
5. one transformer layer forward, VJP, optimizer update, and mixed precision;
6. `pretrained-rstr` prefill and decode with KV residency and logits parity;
7. `finetune-rstr` batched SFT/LoRA/QLoRA with a wholly resident train step;
8. full model execution and training, then multi-device versions;
9. transformation legality: a fidelity switch, coarse-graining or marginalization preserves the
   declared conservation and coupling contracts and stays inside its error model;
10. calibration: simulation-based calibration of inference across fidelities, with the cost
    vector reported per fidelity;
11. invariance: identical results under re-layout, index rotation, restart and reproducible mode,
    as a test matrix rather than a single case;
12. the agent baseline: time an agent writing and certifying a bespoke version of the same
    workload; if that is cheaper than the Raster expression including its certificate, record it
    as a failure of direction, not of the agent.

For each tier record numerical error, kernel count, allocations, peak memory,
host/device and device/device bytes, compile time, tuning time, steady-state
time, and achieved bandwidth/compute. Comparisons with JAX/XLA, Triton,
`llama.cpp-new`, and vendor libraries must use matched dtypes, shapes, warmup,
transfer boundaries, and numerical modes.

## 10. Non-goals and anti-drift rules

- Do not port an entire reference compiler.
- Do not adopt native MLIR merely to obtain terminology or pass structure.
- Do not add an optimization knob without its schema, legality, resource,
  emission, measurement, and explanation path.
- Do not count isolated emitter tests as feature coverage.
- Do not infer semantics from mangled names, source spelling, or exception
  fallbacks.
- Do not create a second operation-property registry.
- Do not let tuning change numerical mode without making that mode part of the
  schedule, cache key, validation oracle, and returned artifact.
- Do not add distributed side protocols that are invisible to the value and
  effect model.
- Do not let self-modifying or learned components bypass the trusted verifier.
- Do not adopt a scene graph, entity hierarchy or integrator DSL; integrators, rates and couplings
  are schedule data over the semantic dynamical system.
- Do not require users to write categorical syntax; the algebra of open systems is the
  specification the certificate checks.
- Do not add a fidelity, coarse-graining, marginalization or surrogate transform without its
  error model, legality rule, calibration oracle and cost-vector entry.
- Do not require every solver to be a probabilistic-numerics solver; require every solver to
  expose an error model.

## 11. Reference mapping and study snapshot

| Reference | Borrow | Do not borrow |
|---|---|---|
| Futhark | typed SOAC semantics, nested distribution, balance/regularity legality, first-class SegOps and results | Haskell representation or full uniqueness system |
| Halide | algorithm/schedule separation, explicit schedule transformations, structural search states and hard resource filters | a separate DSL or every scheduling primitive before a workload needs it |
| Triton | staged high/kernel/target IR, explicit layouts and masks, broad kernel primitives, autotune protocol and failure handling | Python block-program frontend or a mandatory Triton backend dependency |
| `llama.cpp-new` | quant layout oracle, golden formats, model-driven kernel cases | one bespoke compiler architecture per quant format |
| JAX/XLA | abstract values, primitive transformation rules, effects, pytrees, donation, sharding contracts, layout constraints | Python retracing behavior or outsourcing Raster's programming model to XLA |
| MLIR | conversion legality, operation interfaces/traits, pass invalidation, Transform and DataLayout discipline | compulsory native MLIR integration before a concrete backend requires it |
| Mojo/MAX | one language spanning host and device, parametric low-level GPU libraries, explicit layouts/pipelines, heterogeneous cross-compilation | coupling Raster's semantics to a closed compiler stack or requiring imperative kernels as the main abstraction |
| TVM Relax/Disco | graph-level tensor distribution, SPMD worker sessions, explicit collective runtime boundary | making a controller/runtime protocol the distributed semantic IR |
| Legion/DaCe | correctness-independent mapping, topology-aware task/data placement, explicit data movement and transformations | adopting a second user programming model before Raster's value and schedule IRs are complete |
| PETSc star forest, DaCe memlets | one communication primitive under halos, gather-scatter and collectives; subset/volume facts on edges with propagation | a solver-library object model |
| Oceananigans, Devito, GROMACS | halo width and pairlist lifetime derived from operator access and an error tolerance; decomposition priced by an analytic communication model | hand-packed tags, per-layout kernel families |
| AMReX, Trixi.jl | flux registers or adjoint prolong/restrict as the conservation contract of refinement | a fixed refinement data structure |
| ICON, MOM6, E3SM | order-insensitive sums, invariance test matrices, perturbation-ensemble tolerance oracles, two schedules of one physics | Fortran-era global state |
| Athena++, SWIFT | per-block task lists with dependencies and retry, measured task costs driving re-decomposition, stateless random numbers | bulk-synchronous stepping |
| NEST, FLAME GPU 2, whole-cell models | ragged delivery with delays, populations with birth/death and spatial messaging, partition/merge of contended state across sub-models | a fixed neuron, agent or cell model |
| Alpa, DeepSpeed | plans and pipeline schedules as replayable data with a cost model and memory constraint | mutable process-group globals |
| vLLM, Mooncake | prefix-hash identity for continuation state, topology-aware transfer | scheduler state that cannot be replayed |
| Probabilistic numerics, AlgebraicDynamics | error as uncertainty, solvers as inference, open systems composed along ports as the composition specification | a mandatory probabilistic solver or a categorical user syntax |

Local source revisions studied for this direction:

```text
futhark          f05e40c0f
Halide           2fad88f4c
triton           434aecbe9
llama.cpp-new    5c7c22c
jax              d36f6b3b7
xla              2aa87bb
mlir-reference   edff73e1
pretrained-rstr  3caf8a1
finetune-rstr    9e9ba5d
```

The durable decision is not any one revision's class hierarchy. It is the
separation of semantic program, transform/schedule, verified kernel, target
lowering, and composable artifact—with stable identities and measurements
connecting them.

## 12. Iteration procedure

The document is the specification; the survey corpus and its reference kernels are the
regression reference; the demonstrator ladder in §8 is the schedule. Each rung goes through one
cycle:

1. **Design note** (internal working note, not this document): the rung, the reference kernel by
   file and line in the surveyed system, the primitives of §3.8–§3.9 it touches, the oracle,
   the error model, and the agent baseline to be timed.
2. **REPL spike** on toy sizes, single device, using the existing typed route; no new pass until
   the spike shows which primitive is actually missing.
3. **Land the primitive** with its certificate, a production-path test through the ordinary
   compile entry, `explain-pipeline` visibility, and no metadata-only transport.
4. **Run the gates**: the rung's oracle, the invariance matrix of §9, the calibration check where
   inference is involved, and the agent baseline.
5. **Record** measurements and the certified plan with provenance, so re-planning on measurement
   and comparison across rungs are queries rather than notebooks.
6. **Write back** to this document: every architectural claim carries one of the labels
   *designed*, *landed* or *measured*, and a claim may not be promoted without the gate that
   promotes it.
7. **Drift audit** before the next rung: an independent rubric-style reading of the landed code
   against §10, in the manner of the alignment notes, so that landed-but-unwired pieces are
   caught while they are small.

One rung is one pull request per coherent phase, with checkpoint commits on the branch. A rung
that fails its agent baseline twice is a direction finding, not an implementation finding, and
returns to §0.
