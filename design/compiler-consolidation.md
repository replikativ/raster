# Local compiler consolidation

Status: agreed landing order, 2026-09-27. This is the near-term completion ledger for
the compiler north star, not another architecture or a replacement for its longer roadmap.

## Four-step consolidation acceptance campaign — 2026-10-01

The original eight-item campaign remains authoritative. Execute these bounded steps locally
before treating additional backend coverage as compiler completion:

1. Reconcile the support/evidence ledger and retire only demonstrably redundant paths. Public
   tree invocation (#969), batch write preflight (#970), and resident conservative CSR transfer
   (#971) are merged with all seven gates green and published through 0.2.1147. These extend
   declared representation and workload acceptance; they do not establish general language
   completeness, aggregate AD or competitive performance.
2. Close shared correctness seams in normalization, AD and storage/lifetime reasoning. Review
   the existing source/JVM lifetime pass independently of LinkPlan reuse. A narrower GPU proof
   cannot justify deleting an analysis covering different execution paths. Every fix needs a
   counterexample plus unchanged valid-case acceptance; uncertain legality must fail closed.
3. Establish reproducible resident performance evidence using the existing measurement and
   production-canary infrastructure, not a second benchmarking framework. Report preparation,
   allocations, transfers and execution separately. Match timing scopes and numerical policies;
   varying background load permits diagnostic samples, not schedule promotion.
4. Improve general schedules only from validated matched measurements. Preserve independent
   numerical oracles and explicit reassociation policies. Record rejected/inapplicable routes;
   source compilation and native execution remain separate CUDA/HIP evidence.

Pivotal review gates are shared proof/normalization changes and schedule promotion. Focused
tests run in the capped reusable REPL; full suites and vendor compiler checks run in CI.
The outstanding AD, external-model, scientific and distributed milestones stay on the original
campaign rather than becoming implicit claims of this consolidation checkpoint.

The first proof audit reproduced a source/JVM memory-reuse miscompile: `selected = (if c a b)`
could leave the first arm's allocation apparently dead, so a later temporary overwrote it
before the selected read (11.0 became 33.0). Until control-aware may-alias/escape facts are
available, the source memory pass excludes allocations transitively feeding any conditional
initializer/body from reuse. It includes whole enclosing let/loop initializers, not just the
branch-local names. Both selections, nested wrappers, unknown call returns, transitive aliases
and escaped array sizes have original/optimized JVM comparisons; unrelated straight-line reuse
remains enabled. The conservative scan may suppress safe optimization for scalar reads,
shadowing or quoted forms. This is not complete alias analysis and does not justify deleting
the pass or weakening the independent LinkPlan proofs. A suspected separate peephole issue did
not reproduce and is not claimed as a defect here.

The next covered duplication is synchronous KernelCall admission. Scalar-precondition checks
now have one private helper for locally checked artifact/argument values; direct call validation,
launch realization and public scalar preflight retain fresh independent checks at their entry.
Construction still performs its own preflight and final call validation. Scalar range/literal
specialization, ordered guards, alias/alignment and geometry checks remain intact. No retained
proof, cache or new representation is introduced. Count regressions distinguish one checked
boundary from construction's two, and malformed arguments still fail public validation.

The AD normalization audit reproduced a public semantic failure: an unused checked array read
inside a nested let initializer throws in the primal, but `value+grad` previously erased it and
returned a value. Nested-let projection now uses the existing ordered ANF normalizer, retaining
all body statements and capturing nontrivial argument terminals before later sibling bindings.
The old binder-blind loop dependency scans now use shared lexical free-symbol analysis with
declared parameter/carry/recur-local names retained, including names shadowing core functions.
The same projection also normalizes newly lifted loop-initializer prefixes before their activity
and pullback are computed: the quoted-expression API had silently omitted the derivative of a
nested-let initializer while the equivalent `deftm` path worked. Both APIs now share numerical
and finite-difference oracles for this shape.
This closes specific normalization seams, not arbitrary-loop AD or aggregate residual support.

Matched prebound-extent GEMM/ReLU replay confirms the existing general fusion rule emits one
stage, just like the explicit epilogue, on OpenCL and Level Zero. Numerical oracles pass on every
replay. Its timing and the cooperative RMSNorm canary remain nonstationary under background load.
Step 4 therefore retains current schedule defaults: no noisy measurement licenses promotion or
weakening the checked-extent ordering rule. See `local-compiler-evidence.md` for timing scope and
the remaining acceptance gaps.

### Equation-first preparation reuse: next structural boundary

The measured warm Q4 preparation still spends about two seconds constructing its LinkPlan even
when structural compiler-template resolution is a hit. Extend the existing resident-plan
template service rather than introducing another compiler cache or bypassing public proofs.
The exact-scalar specialization prerequisite is PR #978; it is not equation-first plan reuse.

The implementation order is:

1. Separate fresh invocation materialization and host-equation evaluation from reusable call
   construction. Materialization runs on every preparation, retaining lifted reads, checked
   conversions, aggregate projection, generated initializers and exceptions. Every host equation
   runs exactly once per preparation, including cache hits; no evaluator closure is retained.
2. Define a source-free binding witness over the complete staged scalar environment, resolved
   storage dtype/shape/range, initialization contract and storage alias partition. Canonical bits
   distinguish signed zero and NaN payloads. Compile epoch, target facts and numerical/schedule
   policy remain part of the enclosing compiler-template identity. Array contents are excluded;
   scalar values derived from those contents are evaluated freshly before selecting an entry.
3. Generalize the existing single-flight resident-template cache for the two existing certified
   lowering kinds. Strip all caller and generated/clone initializer sources. Rebind those sources
   from the current materialization, check the complete witness independently of its digest,
   and reconstruct public defaults/input-output projections from current arguments.
4. Retain fresh final projection/effect certification for roles, donations, outputs and taps.
   A cached proof must be sealed to the exact immutable structure it proves. Public revalidation
   must still reject tampered calls/plans/certificates; a boolean cache-hit flag grants no authority.

First acceptance includes changed arrays and lifted scalars, failed reads on hits, host evaluation
counts, raw floating bits, changed dtype/extents/ranges, shared versus distinct storage, initializer
freshness, changed roles/outputs, compiler redefinition and certificate tampering. Reuse numerical
Q4 composition and existing tree/record oracles; compare fresh and cached preparation. Bypass
template reuse for an unsupported staging form without narrowing accepted compiler source forms.
The measured Q4 host prefixes must be covered before claiming this fixes that preparation cliff.

The prerequisite audit also reproduced an equation-first host-result consistency defect:
ordinary equality admitted supplied +0.0 versus evaluated -0.0, but rejected equal Double NaN
payloads. The existing canonical comparison now checks that conflict boundary without changing
host execution order or scalar validation. The structured-control/program-call suite passes
44 tests / 280 assertions; this remains a correctness change, not template reuse.

The first structural prerequisite now separates private `stage-inputs` from
`construct-staged-call`. Public `make` composes them synchronously with unchanged arities and
validation order. The evaluator is not retained; physical-result projections and construction
identity maps stay invocation-local. Two dependent host equations run once, in order, on each
preparation; invalid inputs and host exceptions cannot enter structural construction. The
combined structured-control and actual local composition acceptance passes 54 tests / 378
assertions. No template lookup or retained public staging authority is introduced by this split.

A warm public Q4 preparation profile after this split resolves the compiler template in about
10 microseconds, but LinkPlan construction still takes about 1.69 seconds. Fresh staging includes
about 434 ms of program validation; structural call construction takes only about 2.2 ms. These
shared-load observations rule out caching constructed calls as the principal fix. Target static
proof reuse and complete source-free plan binding instead; inclusive profiling is not a speedup.

The next prerequisite makes emitted-program physical-result evidence read-only and seals it to
the exact in-process program/evidence objects. Copied/modified values and replacement metadata
callbacks fail the owner check. Public program/call validators still independently validate;
their mutable synchronous projection scopes are fresh copies, not the sealed index. No global
cache or persistent artifact field is added, and no validation is skipped by this prerequisite.
The seal is an internal structural-tamper contract, not security isolation against JVM reflection
or private-Var mutation. Cache integration must additionally check compiler epoch and pipeline
identity; object identity alone is not semantic reload invalidation.
The focused structured-control and local device-composition suites pass 55 tests / 392 assertions,
including mutation rejection through the projection map and its entry views.

The existing compilation-template entry now owns a delayed, process-local static proof, derived
only after ordinary artifact resolution/load/store. Reuse requires the exact live cache entry,
compilation and emitted-program identities, plus current compiler epoch, pipeline root and
validator root. Guards are checked again after forcing the delay. Drift uses independent
validation; failed proof derivation evicts only its own entry and permits retry. This is not a
transactional reload guarantee. No second cache or serialized proof field is introduced.

Only repeated static emitted-program analysis is skipped. Invocation materialization, scalar and
buffer validation, result views, host equations, final LinkPlan validation and public verification
remain fresh. The focused owner/failure/drift suite passes 4 tests / 42 assertions; affected
composition/control/local-device suites pass 75 tests / 522 assertions. Review found no blocker.

One public Q4 preparation run on the shared-load, power-save laptop records cold total 8.78 s
(7.38 s compiler-template resolution) and warm total 0.891 s (25 microseconds template resolution,
0.875 s LinkPlan construction). Earlier warm construction was about 1.69 s, but these are unmatched
diagnostic observations, not a controlled speedup. Cold proof derivation is included in aggregate
equation lowering, outside its materialization/construction subphase timers. Complete source-free
plan binding and matched performance evidence remain outstanding.

Storage realization now consumes that same sealed physical-result index for exact plain-equation
boundary identities. Missing entries, dispatch boundaries and structured loops retain independent
projection validation. Concrete shape, prefix view, backing range and initialization realization
are unchanged. Independent versus retained lowering agrees on values, outputs and aliases;
focused tests pass 3 / 23 assertions and affected suites pass 75 / 522. Review found no blocker.
After this additional static deduplication, a warm public Q4 diagnostic records 0.435 s total,
including 0.431 s LinkPlan construction. Fresh final validation remains the dominant measured
cost. These shared-load observations still do not establish a controlled execution speedup or
complete whole-plan binding reuse.

Then resume matched performance evidence, external training acceptance and the distributed/PDE
milestones. This optimization does not complete the wider campaign or authorize schedule promotion.

Finish the general compiler work on OpenCL and Level Zero first. Continue CUDA/HIP
hardware-free compile gates; native vendor runtimes and hardware acceptance follow the
local milestone. FPGA spatial scheduling and distributed optimization are later tracks.

### Parallel CUDA ownership — 2026-09-30

Native CUDA backend implementation and device acceptance are delegated to a separate
collaborator. They do not gate the local consolidation campaign. Work here remains the shared
TypedSOAC/control and AD semantics, generated scheduling vocabulary, ABI/LinkPlan ownership,
preparation-cost deduplication and OpenCL/Level Zero workload acceptance. Keep the hardware-free
vendor compile gates running to catch changes to that shared vocabulary.

Backend-neutral does not mean hardware-oblivious: target descriptors constrain legal schedules,
instruction shapes, memory spaces, subgroup widths and synchronization. CUDA may specialize
those existing contracts; missing hardware operations should be proposed as explicit shared
IR capabilities, not a second binding, ownership or compiler-cache convention. Local numerical
oracles and vendor device execution remain separate acceptance evidence. Shared IR/ABI changes
should include their compatibility impact for the collaborator.

Verified dense f16 contractions now produce the same scheduled matrix-body vocabulary for
the exact DPAS, CUDA MMA, and CDNA MFMA instruction shapes. Authoritative device facts must
agree with the requested instruction and subgroup width. Direct CUDA/HIP schedules require
static tile-aligned extents and FP32 output because their current fragment emitters cannot
preserve masked edges or half-result stores. The CI source fixtures for both vendors start
at verified contraction facts; HIP remains an explicitly source-only, pinned rocWMMA candidate.
The public flat contraction route still admits only DPAS emission, and the typed program
route's mixed-precision matrix alternatives are Intel-specific. Those are the next routing
obligations, not evidence that a public CUDA/HIP GEMM is already available.
The portable scalar contraction now separates its reduction dtype from a declared result
storage dtype. A typed post-reduction cast supplies the fallback ABI for widened results;
the register-tiled leaf declines this case until it can preserve the same contract. This is
needed before vendor matrix alternatives can join one semantic dispatch rather than each
inventing an incompatible result buffer.

## Completion order

1. **Control and effects.** Preserve bindings, branch results, ordered effects and loop
   carries through the existing typed regions. Reuse scalar/product Fold and KernelBody
   control instead of adding recognizers with separate semantics. General effect-loop
   carry tuples and pure branch-local exits/recurrences are implemented. Counted effect loops
   admit multiple recurrence sites when every arm yields the full carry tuple and advances
   the same verified unit step; early exits in effectful loops remain open. Accepted kernels must
   retain all effects; unknown legality produces a source-located decline.
2. **One complete memory proof.** Follow `memory-planning-campaign.md`: join logical values,
   storage views, initialization, selected replay order, escape/AD retention and completion
   evidence; enable one justified reuse and remove any analysis it actually supersedes. Shadow
   proposals alone do not count as enabled reuse.
3. **Pretrained's generated execution paths.** Validate quantized projections, cooperative
   reductions, routed attention and prefill through public compilation and linking. Serving
   policy, page allocation and persistence policy remain in pretrained-rstr. Raster owns
   reusable numerical, storage and execution contracts. The masked prefill softmax now has a
   verified row-ownership proof and emits one TypedSOAC/KernelBody kernel: an unrelated remainder
   local no longer invalidates its row-major address certificate, while a remainder used in an
   address still declines. Its mixed float-load/double-sentinel join is explicitly typed in the
   source. Public equation-first execution now matches the JVM across four window shapes on
   local OpenCL and Level Zero, retaining one emitted kernel and no fallback in each case.
   The older Level Zero sandbox probe failure no longer describes this local evidence. New
   corpus compilation errors fail the ratchet instead of being invisible until a baseline
   refresh.

   The public Q8_K cooperative padded-row quantizer and Q4_K product projection now have a
   composed B=2, width-640/padded-768 execution oracle on local Level Zero and OpenCL. The
   three packed activation leaves are connected as resident LinkPlan nodes with no host
   source or inter-program transfer; packed weights and metadata remain shared constants.
   This proves one generated quantized projection chain and CPU-reference numerical parity,
   not the full Gemma decoder, its scheduling quality, or a performance win.
   Emitted `KernelGraph`s can now enter `LinkPlan` directly with exact LinkValue and typed-scalar
   bindings, and run in order beside equation-first program instances. This removes the need to
   fabricate a resident descriptor for a graph at that boundary; the local indexed-attention
   reference graph is the first dual-backend device oracle. Mixing legacy descriptor instances
   into that prepared sequence is still an explicit runtime decline. Graph output ABI writes are
   not by themselves evidence of full-element coverage; complete-write/initialization proofs
   remain a separate compiler obligation before private reuse or an uninitialized partial output.
4. **Local consolidation gate.** Publish a support/evidence matrix, retire covered duplicate
   paths, and record reproducible correctness, compilation, allocation, transfer and execution
   baselines on both local backends. Retain city irregular kernels, one PDE/stencil and a
   resident AD training step alongside pretrained. Existing tests are starting points, not
   evidence that every target or shape was executed.
5. **Cross-vendor acceptance.** Add native CUDA, then HIP execution through the same ABI,
   artifact and event contracts. Rent hardware after the acceptance runner is ready, or
   earlier if a concrete hardware question blocks the design. Compilation, execution and
   competitive performance are separate claims.

### ASR resident-capture acceptance order

Keep this inside item 3, not a replacement campaign or another attention ABI. The allocating
`gqa-decode-attention-weights!` is now an explicit host-only allocating wrapper. Its mandatory
host oracles are separate from BLAS availability. The caller-owned
`gqa-decode-attention-weights-resident!` change passes local acceptance with two generated
KernelBody kernels, without a new attention ABI or emitter. All seven CI gates passed and
the slice merged as #924; external ASR migration remains open. Its contract is:

1. Retain physical score/output/sink storage precision independently of scalar computation
   and normalization scratch precision. Use the shared declared-storage policy, including
   source specialization and cache identities; do not reconstruct types in an emitter.
2. Expose caller-owned output and scratch through ordinary public invocation/LinkPlan
   contracts. Reuse the library's numerical stages rather than copying another target kernel.
   Preserve score and exponential materialization, the sum's evaluation order, and the
   inverse's precision. Float source reductions/exponentials/reciprocals materialize at Float;
   Double scratch stores the rounded reciprocal exactly, and its reload restores source T.
   These source boundaries are explicit rather than dependent on inference context.
   Reusing a normalized-probability buffer introduces another rounding
   boundary and is not silently equivalent to the existing routine.
3. Map independent sink positions in parallel and retain ordered, per-head typed sink stores
   inside each work-item. The existing ordered effect-loop dialect can express this; neither
   atomics nor a widened head sum followed by one store preserves Float rounding. A warm-REPL
   public equation-first device test preserves the 8388608 + three separately materialized
   thirds counterexample and an untouched sink tail on Level Zero and OpenCL.
4. Validate Float/Double, MHA/GQA, empty history, existing sink contents, untouched tails,
   scratch ownership and repeated resident replay on both local backends against the host
   frozen sequential weight-capture oracle. The mandatory host/preflight suite passes 95
   assertions, both local device tests pass 124 assertions, and the workload ledger pins
   zero driver allocations during public lowering, seven caller-owned arrays and two outputs.
   Host wrapper sink/input identity aliases now fail before writes; resident graph preflight
   rejects writable aliases, while read-only q/k/v sharing remains legal. This is an explicit
   surface-contract tightening needed by the staged algorithm, not a hidden numerical change.
   Then migrate the external ASR consumer and measure preparation/execution separately.

AD consolidation and external training validation remain separate obligations. None of these
inference-only checks establishes differentiability, general loop tapes, or higher-order AD.

### Parametric source reload contract

The ASR precision investigation exposed a separate warm-REPL correctness gap: after redefining
an `All [T]` function, its default Double method changed but a previously materialized Float
method remained callable through both generic dispatch and compiler resolution. Derived method
metadata now records its template annotation signature and concrete type-variable bindings.
Replacing that exact template eagerly rebuilds its already materialized bindings through the
existing registration-only callback. Explicit concrete overloads have no such provenance and
remain intact; derived rebuilding does not advance the semantic source epoch again.

This is a direct-template replacement contract, not transactional reload or global dependency
invalidation. Next, audit stale/orphaned mangled Vars, eager Double provenance and transitive
inlined callees against the existing definition epoch and source-dependency manifest. Existing
compiled artifacts or captured dispatch objects are not retroactively rewritten. Keep those
obligations distinct from the artifact cache's existing epoch invalidation.

### Generated numerical continuation acceptance

The resident full-domain refinement workload composes four existing periodic heat steps and
restriction in one caller-owned LinkPlan, replayed without intermediate host transfers. Inlining
the same numerical operator exposed repeated checked dimension bindings after device effects.
Normalization now reuses the existing dominating checked-value evidence for scalar bindings too:
an identical successful conversion over immutable scalar SSA operands may name the earlier value.
The first check remains at its source position. A new check after an effect still declines this
route, and array-backed checks remain separate evaluations. This extends the existing proof; it
does not make checked casts generally removable, add a PDE opcode or hoist possible exceptions.

Local OpenCL and Level Zero execute uninterrupted six-checkpoint evolution and a three-step
producer/checkpoint/fresh-restore/three-step continuation. Coarse and fine chunks name hashes of
actual mmap bytes; the producer closes before read mappings open and synchronous restoration
finishes before leases close. Same-backend final fields agree bit-for-bit; generated fields also
match the JVM and retain volume-weighted mass. The frontend and new continuation suites pass
114 tests / 575 assertions locally, followed by all seven green CI gates and merge #926.
The snapshot's producer string is
a test-fixture label, not a production compiler-build identity. Production manifest publication,
partial-patch interface fluxes, actual distributed transport, subcycling and reflux stay open.

### Conservative partial-interface acceptance — 2026-10-01

The next local numerical slice adds ordinary Double-storage/Int-index typed face-flux and
CSR-divergence programs in `raster.ode.finite-volume`. Each face flux is materialized once;
incident cells gather it with opposite signs and their inverse cell volume. The compiler still
uses the existing maps, scalar loops, arrays and LinkPlan stage/ownership contracts. There is
no finite-volume or AMR compiler opcode, target kernel, mesh allocator or extra cache.

The fixture replaces only a central coarse patch: twelve coarse cells, sixteen fine cells and
sixty shared faces on a periodic unit square. It checks complete face coverage, positive volume,
the explicit convex-combination timestep bound, constant preservation, nontrivial evolution and
volume-weighted mass. An independent face-scatter oracle checks the production CSR-gather path.
Four generated stages implement two timesteps per resident replay; three replays on each local
backend agree with the oracle after every replay. Scratch/flux start poisoned, lowering allocates
no driver buffers, and only the donated evolved field escapes.

This is a synchronous mixed-resolution numerical acceptance, not a completed AMR hierarchy
projection or a discretization-accuracy comparison. Caller-supplied CSR bounds and paired signs
are contracts, not a mesh certificate. Deriving this incidence/geometry from AMRPlan, reconnecting
coarse/fine fields during regridding, conservative transfer, time interpolation, subcycling/flux
registers/reflux, restart across a hierarchy change and external AMReX comparison remain open.
Keep the existing full-domain mapped-byte continuation tests: this fixture does not replace them.

The follow-up acceptance derives active cells from the existing validated
`RefinementHierarchy`, excluding covered coarse cells before constructing shared faces.
Four bounded layouts include moved/narrower and disjoint fine patches; coverage and paired
incidences are checked independently. Central and disjoint layouts also exercise the same
generated programs on both local backends. This is a test-only, two-level 2D projection:
its small pairwise face enumeration is not a production connectivity algorithm, and no new
public mesh API or compiler IR is introduced. A general hierarchy-to-field/ownership mapping
and scalable connectivity construction remain open.

### Conservative layout remapping acceptance — 2026-10-01

The bounded partial-patch fixture now projects overlaps between its existing active-cell
layouts into ordinary CSRMatrix data and calls the existing `sparse/spmv` through public
equation-first compilation. No remap/AMR semantic opcode, target kernel or production mesh
builder is added. Sixteen source/target layout pairs check positive weights, constant preservation,
volume-weighted mass and every source column's extensive contribution. A separate finest-tile
lookup/average oracle does not consult the CSR entries or their overlap builder.

Resident acceptance composes four generated heat stages with the generated CSR stage. The
evolving field is an explicit donated owner with a read-only remap borrower, initialized from
the same host object; the borrower cannot refresh that old initializer on replay. Central-to-moved
and moved-to-disjoint layouts run three replays on actual OpenCL and Level Zero, matching
independent face-scatter evolution and tile-remap oracles and retaining mass at every checkpoint.
Component lowering reports zero driver allocations, and composition is tested with allocation
forbidden. The shared field resolves to one physical node; no host bridge is introduced.

This is conservative transfer of a cell-average field to another layout, not a completed adaptive
simulator. The overlap enumeration is small/test-only. Production connectivity, migration and
lineage publication, subcycling/reflux and
discretization/error-estimator or external simulator comparisons remain separate acceptance work.

### Resident post-transfer evolution — 2026-10-02

The next acceptance composes source evolution, conservative CSR transfer, then two steps of
evolution on the target layout: nine generated stages with no inter-program host bridge.
The original field remains a donated mutable owner with a read-only remap borrower. Target
evolution reads the remapped field and writes a distinct owned result; this is ordinary functional
dataflow, not transfer into another mutable owner. The linker still rejects connections to state
consumers and donations without explicit mutable-owner bindings; those separate limitations are
not silently bypassed or generalized by this fixture.

This workload reproduced a general composition decline: connecting a producer output discarded
its source initializer, even when the producer reads that output (CSR `spmv` retains `beta*y`).
Composition now preserves producer initialization unless existing certified effect evidence proves
a complete overwrite with no reads anywhere in that component. Missing or read/write evidence
retains the initializer; existing write-only generated intermediates still avoid uploads. The
replaced consumer node's source disappears. Fresh initialization/effect proofs remain mandatory;
a zero numerical coefficient is not an initializer-elimination proof.
A hardware-free read/write producer regression checks source identity, disappearance of the
consumer source, independent verification, and rejection if the initializer is removed.

Central-to-moved and moved-to-disjoint layouts run three resident replays on each local backend.
Independent face-scatter evolution and finest-tile transfer oracles agree at every replay; target
evolution is nontrivial and retains volume-weighted mass. Target input/output and scratch begin
poisoned, while `spmv`'s old output is finite because `0*NaN` is not zero. The link-composition and
full partial-patch suites pass 19 tests / 324 assertions. Production connectivity/field mapping,
durable restart across layout changes, lineage publication, subcycling/reflux and external accuracy
acceptance remain outstanding. No AMR compiler opcode or new runtime/cache is introduced.
The combined public composition, native replay, linker and scientific suites pass 48 tests / 535
assertions, including the existing no-upload generated-intermediate assertions; both local device
availability gates were explicitly true. Pivotal review found no blocker in the conservative
initializer proof, with the existing sealed-Prepared prevalidated boundary retained.
The existing hardware-free public C-family composition test additionally checks certified
write-only initializer removal (1 test / 10 assertions), so this positive optimization boundary
is protected even when native device execution is skipped.

### Changed-layout mapped-byte continuation — 2026-10-02

The partial-patch fixture now compares an uninterrupted resident source-evolve → transfer →
six-target-step chain (17 generated stages) with a nine-stage midpoint captured after two target
steps, then restored into a fresh target-layout executable for the remaining four steps. Actual
source and destination mapped bytes are content-addressed; producer sessions and writable maps
close before read leases open. Parent/child manifests retain the distinct active-cell order,
layout fingerprint and actual chunk content identity, including equal-length changed layouts.
Read leases are verified and closed after synchronous initialization, before restored replay.

Both OpenCL and Level Zero match uninterrupted final fields bit-for-bit, independent face-scatter
and finest-tile-transfer oracles within tolerance, and volume-weighted mass. Reusable preparation
shares target geometry constants across the uninterrupted chain rather than uploading duplicate
copies. The affected partial-patch, whole-domain continuation and manifest suites pass 21 tests /
351 assertions; both native availability gates were explicitly true. The existing whole-domain
checkpoint oracle is preserved, not replaced.

The mandatory host test rejects coordinate-space tampering of a certified state despite unchanged
field length. This proves certificate integrity, not automatic rejection of an independently valid
but wrong-layout checkpoint during runtime binding. The fixture supplies the matching geometry;
production hierarchy-to-field reconstruction, semantic restore compatibility and lineage/store
publication remain open, as do subcycling/reflux and external accuracy comparisons. Provenance is
explicitly a fixture label, not a production compiler-build identity. Pivotal review found no
lifetime, resource-cleanup or numerical-oracle blocker. No new compiler operation or runtime API.

### Aggregate invocation projection audit — 2026-10-01

Before inventing a conservative-remap primitive, probe the existing CSR `spmv`. Its JVM
result is correct, but public equation-first compilation declines before scheduling.
The walked binders already retain `m: long`, `rp: ints`, `ci: ints`, and `vs: doubles`.
This is **not missing type inference**: an array-valued field binding reaches the scalar
equation builder without its physical storage projection. The frontend now distinguishes
that failure as `:unsupported-buffer-projection`, retaining the tag, expression and source
coordinates; genuinely untyped scalar bindings keep their original diagnostic.

There are two representation boundaries to consolidate. `soa-lower` already projects
registered SoA companions and all-array records in the resident pipeline, but equation-first
compilation does not invoke that parameter-rewriting step. CSR additionally mixes arrays
with scalar dimensions, outside the all-array admission. Physical KernelABI scalar slots
currently reject logical pointer bindings/fields; removing that invariant alone would not
make public invocation materialize a record correctly.

Keep this work inside campaign item 3, as a prerequisite for reusing existing sparse library
operators in item 8; it does not replace training, memory, emitter or benchmark acceptance.
The implementation order is:

1. Factor one checked parameter-representation projection from the existing aggregate pass.
   Derive ordered leaves exclusively from declared field types/order, preserving each leaf's
   scalar or array kind and storage type. Do not add a CSR, solver or backend registry.
2. Apply that representation before TypedSOAC construction in the public path. Retain the
   logical caller parameter and field path separately from physical leaf symbols; share the
   producer with the existing resident path instead of copying its source rewrite.
3. Carry the checked logical-to-physical parameter mapping into invocation materialization.
   Public calls still supply the declared record, not hidden extra dimensions or field buffers.
   Array leaves retain source identity and per-leaf ownership/alias proofs; scalar leaves use
   existing scalar type checks. Do not evaluate arbitrary field expressions in the linker.
4. Keep kernel signatures flat. Decide explicitly whether logical scalar provenance belongs
   only to invocation projection or also to KernelABI; do not conflate a scalar dimension with
   a resident pointer. Any shared ABI change needs coordination with the vendor backend work.
5. Validate generated SoA, all-array containers and mixed CSR records through public lowering
   and native JVM/device parity, plus malformed fields, wrong scalar types, aliases and warm
   specialization/dependency invalidation. Then reuse CSR for conservative hierarchy remapping.

Record projection does not prove CSR row-pointer monotonicity, indirect-index bounds or
consistency of scalar dimensions with storage capacities. Preserve those as explicit library
input contracts unless checked independently; never infer a complete write or safe read from
the record class name alone.

Nested aggregates, returning records, aggregate donation and AD over sparse indices must be
specified and tested independently; none follows from accepting read-only CSR parameters.

### Mixed primitive record invocation — 2026-10-01

The equation-first representation environment now admits numeric scalar fields alongside
primitive-array fields. The same declared field registry/order supplies body projection,
physical parameter order and InvocationPlan validation. Scalars are rank-zero numeric values;
arrays retain their declared storage contracts. Scalar KernelABI slots do not acquire pointer
`:binding`/`:field` provenance: logical field access happens at invocation materialization.
The old resident descriptor binder does not opt into the mixed representation.

Raster's existing `raster.linalg.sparse/spmv` is the first acceptance workload, not a rewritten
flat equivalent or CSR-specific compiler operation. Its original CSRMatrix, x, y, alpha and beta
arguments lower to one generated KernelBody kernel. Actual OpenCL and Level Zero runs agree
exactly with the JVM for the nontrivial rectangular CSR fixture, including alpha/beta and replay
with new matrix buffers. Lowering performs zero driver allocations; y is the donated state.

Prepared replay captures record scalar fields. Replacing the logical record checks its nominal
class and rejects changed scalar fields **before uploads or kernel replay** with
`:compiled-aggregate-scalar-change`; callers must prepare again to change them. Tests change
each CSR scalar field and then verify untouched device state through the next successful call.
All fields are conservatively captured, including fields not currently used by the emitted leaf.
This is not a general mutable aggregate ownership API or automatic scalar respecialization.

Nominal CSR shape is not a proof of monotonic row pointers, column bounds or consistency between
nnz and array contents. Those remain numerical input preconditions, not newly claimed compiler
verification. Nested records, mixed-record AD reconstruction
and writable aggregate donation remain open. Continue the shared declared path projection for
map/record roots rather than adding a second binder or kernel ABI.

### Annotated map parameter bindings — 2026-10-01

Issue #966's annotations inside `:keys` are not Clojure/Typed Clojure binding syntax.
The supported surface annotates the enclosing binding:

```clojure
(deftm squared [{:keys [x]} :- (HMap {:x Double})] :- Double (* x x))
```

This is source normalization into the existing closed typed-tree path, not a new map
dispatch system. `:keys`, qualified keyword keys, explicit keyword renaming and nested map
bindings obtain types from the declared HMap. Canonical leaf order and runtime shape/alias
checks remain shared with `raster.params`. Missing/extra keys still decline; `:or`, `:as`,
string/symbol keys and optional/open maps are not admitted by this slice. Inline annotations
inside `:keys` now fail at macro expansion, not deferred bytecode compilation.

The tree rewriter now respects ordinary local shadowing and changing scalar/array loop carries.
Other binding forms use the existing `form/scope-info` authority; quoted data stays opaque.
Only unused generated pure parameter aliases are removed, proved by shared scoped free-variable
analysis. This avoids making another loop recognizer or type/function registry.

Local evidence covers structured JVM/JIT/AOT calls, reconstructed reverse gradients and the
generated flat method through target-neutral TypedSOAC. OpenCL/Level Zero device tests compare
that flat method against the uncompiled structured call, including replay and replacement
array values. CUDA evidence here is hardware-free source compilation, not native execution.

**Public invocation integration:** equation-first `gpu.compiled` now consumes logical closed
HMap/HVec wrappers directly. It resolves the existing flat source specialization for numerical
compilation, while InvocationPlan retains the caller's declared tree paths and canonical leaf
order. The shared aggregate selector handles record fields and tree paths during materialization
and replay; no second binder, cache or physical kernel ABI is introduced. Declared numeric scalar
and primitive-array leaves are supported. Empty trees, nonnumeric leaves and nested record leaves
remain outside this GPU contract; closed-tree flattening alone does not prove them representable.

Named roots expose semantic keys such as `[:model :buffers 0]`; anonymous destructured parameters
have stable ordinal labels (`arg0`, etc.), with underscores appended to avoid named-argument
collisions. Internal source roots remain hygienic gensyms. The prepared descriptor and in-tree
expose the actual labels, not inferred runtime map keys. Existing explicit HMap syntax is unchanged.
Whole-root input replacement validates tree shape and identity restrictions before projection,
and rejects changed captured scalar leaves before uploads/replay. Changing scalar dimensions or
parameters requires preparing a new invocation. Constant roots cannot be replaced; whole-root
donation and aggregate writes remain explicit declines. Composition retains qualified leaf inputs
and strips unqualified root shorthand just as for records.

Dynamic maps inside kernels,
whole-tree carries and general optional/default semantics remain distinct language capabilities.
This slice belongs to direct TypedSOAC/compatibility consolidation; the eight-item campaign,
training/AD consolidation and distributed numerical acceptance remain open.

### Invocation input batch preflight — 2026-10-01

A reproduced nested-tree invocation exposed a general artifact boundary gap: a wrong second
buffer dtype was rejected only after uploading the first buffer. The fix belongs to the shared
LinkNode/DeviceArray input contract, not map or record compilation. `link/validate-write!` checks
host storage and device liveness, target, layout, shape and range overlap without transfers,
registrations or readiness mutation. Artifact invocation preflights every dynamic input before
the first write; profiling and measurement likewise preflight all captured inputs before refresh.
Public writes still validate independently. No retained proof cache or backend convention is added.

Structural/scalar aggregate checks and donation ownership checks remain separate, preceding batch
input writes. Native OpenCL and Level Zero oracles verify zero writes on a bad later tree leaf and
unchanged earlier resident input contents. Host tests also preserve output handles, readiness and
pending-input state after invalid dtype, length or device-target inputs. This is validation-failure
atomicity, not rollback of driver failures or a transaction over concurrent caller mutations.

### Public all-array aggregate invocation — 2026-10-01

The first implementation shares `soa-lower` with equation-first compilation, before TypedSOAC
construction. One captured representation environment supplies both the source rewrite and
ordered physical parameters. The invocation plan retains a checked mapping from declared public
parameters to physical leaves. Materialization checks the nominal record class, projects only
its declared fields and delegates each leaf's storage/shape checks to the existing buffer
materializer. Kernels remain flat typed signatures; KernelABI pointer/scalar rules are unchanged.

The compiled facade accepts the original record in its ordered arguments and in replay input
maps, e.g. `{:state new-record}`. Semantic leaf keys are `[:state :positions]`, not generated ABI
names. Capturing a record as `:constant` captures its admitted array leaves. Supplying both a
record and one of its fields in a replay is rejected. Source-record shorthand is deliberately
not propagated through composition: composed artifacts keep explicit component/leaf references
such as `[:component [:state :positions]]`, avoiding collisions between same-named source records.
The initial facade rejects writes through aggregate fields, donation and whole-record outputs;
role overrides cannot disguise a write as a read. This does not establish mutable aggregate or
aggregate-AD semantics. The lower-level compiler still represents array leaves, not record-valued
outputs or a runtime-owned record.

Declared field order, storage tags and nominal class participate in template identity. Changing
a relevant declaration changes that identity; unrelated field metadata leaves the projection
unchanged. A checked synchronous construction reuses already materialized leaf sources for facade
defaults rather than projecting caller fields again. Flat parameter behavior is preserved.

Native acceptance covers an all-array record containing Float and Int arrays, record replacement,
captured constants, malformed classes/storage, role/output declines and a generated SoA input with
an unused Long-array field. The same programs agree with their JVM/independent oracles on OpenCL
and Level Zero and lower without driver allocations. CUDA source compilation is checked separately,
not presented as CUDA device execution. Existing resident SoA and invocation/composition tests
remain required. The optional WASM runtime is absent from the lean local REPL; its full tests stay
in CI, whose test alias includes Chicory.

Mixed scalar/array records (CSR), nested aggregates, returned records and per-field mutable
ownership remain next steps. This oracle uses let-bound primitive-array field reads: a direct
nested core `aget` of a record field exposed an existing JVM Object-array cast gap during the
probe and is retained as a separate front-end language-coverage follow-up, not claimed fixed.

### Public training boundary follow-up

The full existing tiny Gemma/LoRA forward/reverse-AD/SGD program was probed through
`compiled/lower` with the equation-first compiler, adapter donation and frozen constants.
It exposed a contraction whose verified scalar boundary included `seq`, but portable body
discovery excluded that name as `clojure.core/seq`. Contraction lowering now binds the shared
lexical-local set from its physical boundary and supplied type environments. The reported leaf
then retains `[d r seq]`, all with their declared Long widths, without relaxing index validation.
The focused contraction suite passes 20 tests / 171 assertions. Full public training execution
remains a distinct acceptance gate, not established by a successful leaf. Direct/synthetic
contraction callers still have the older default-int policy for unspecified scalar dtypes;
the production probe supplies explicit types, and tightening that fallback remains separate.

The same public probe also exposed a missing capture in the typed invocation prefix: `(long seq)`
was emitted as a closed scalar region with no operands. Invocation operand discovery now uses
its explicit lexical environment; validation also retains the public/prefix symbol boundary so
a corrupt region cannot hide a missing core-named local. Focused tests execute `seq`, `count`
and `first` as typed Long operands and reject their deliberately omitted captures. This changes
neither source semantics nor scalar dtype inference, and introduces no separate operation registry.

The next public-training gate reached initialization certification. A fresh RMSNorm-gradient
buffer is completely written by a rectangular effect map, but its address uses a retained Long
SSA row-offset local. The existing coverage proof now expands integral SSA locals in source
order, preserving each declared cast; typed interval checking still precedes mixed-radix
injection and exact-capacity checks. Narrowing, overflow, guards and non-injective addresses
remain declines. This extends proof evidence, not ABI write permission or numerical inference.
The full tiny Gemma/LoRA program then lowers with zero driver allocations and runs through the
public compiled API on the local Arc. After two SGD steps, GPU/JVM losses were
2.40469313/2.40469337 and maximum adapter absolute error was 2.98e-8. The existing 25-step
FP32 trajectory test is migrated from descriptor-fixture invocation to this public API; its
mixed-precision counterpart still uses the fixture and remains a separate migration item.
An allocation-free public probe at CFG-MP with only `:precision :mixed-f16-f32` lowers and binds
122 generated kernels, but its 50 contraction leaves retain sequential-segment schedules.
This is expected under equation-first `:typed-contraction :auto`, which deliberately remains
ordered portable; a numerical precision permission is not an optimized schedule selection.
The legacy mixed-precision test must remain until per-equation matrix admission and selection
reach the same public program boundary. A global explicit matrix request cannot replace that
work: a training program also contains non-matrix reductions. Non-empty measured contraction
selectors now fail before frontend lowering on equation-first instead of being silently ignored;
the descriptor route still owns their existing validated consumption. Empty selector maps and
the current portable default are unchanged. Next, reuse the existing typed candidate/refinement
and selector machinery per equation rather than adding a Gemma or AD-specific kernel route.
Local validation: the migrated 25-step device/JVM trajectory passes 43 assertions, with both
losses falling from 2.800152 to 0.256915; the focused coverage test passes 25 assertions and the
initialization suite passes 21 tests / 129 assertions. These are local Arc results, not CUDA/HIP
runtime or external full-model training evidence.

The adapter-only follow-up uses reverse AD's existing `:wrt` boundary rather than deriving
activity from model names or compiler donation roles. Static option-bearing direct and
let-bound `value+grad`/`grad` calls now consume `prepare-value+grad`, the same typed preparation
used by the runtime constructor, without compiling an unused runtime wrapper. Index validation,
constant-data treatment, tangent types, seeds and original parameter/nil-slot order stay owned
by that preparation. Compiled single-parameter `grad` now returns the public API's scalar,
not the inliner's former one-element vector; multi-parameter gradients remain padded vectors.
Dynamic option expressions and non-reverse compiled modes decline explicitly. The optionless
inline transformer now consumes the same preparation too: its duplicate activity selection,
gradient flattening and default scalar-type reconstruction are removed, together with the
compiler's AD-transform callback and unused AD-only expansion helper. The inliner retains
overload resolution and metadata-preserving argument substitution, not a second AD algebra.
This does not establish general loop tapes, conditional residual transposition or complete
higher-order array AD.
The final shared-preparation path passes the public 25-step adapter-only Gemma trajectory
(43 assertions), the inliner/AOT suite (18 tests / 81 assertions), four existing compiled AD
checks (11 assertions), and the density/observation bridge suite (8 tests / 64 assertions).
The subsequent optionless consolidation passes the inliner/AOT suite (18 tests / 81 assertions),
the selected first/second/third derivative checks plus public Gemma trajectory (5 tests / 54
assertions), loop/helper/HVP regressions (4 tests / 38 assertions), Q8 resident LoRA tests
(3 tests / 18 assertions), and the density/observation bridge suite (8 tests / 64 assertions).
The selected higher-order run no longer prints the prior late undevirtualized-dispatch warnings;
that bounded observation is not a claim that all higher-order scalar/array typing or performance
is consolidated. Runtime and compiled raw-loop failures now describe the same preparation gate,
rather than recommending compilation as an escape from a shared unsupported construct.

The equation-first public result boundary now includes explicit outputs, donations and taps in
the certified LinkPlan outputs, not only in runtime wrappers. This joins escape declarations to
the existing initialization and storage analysis; it does not enable reuse through inspectable
resident handles or promise versioned AD tape snapshots. The ordinary `Compiled/invoke-leased`
boundary now pins one completed owned resident result across callers and invalidates its external
wrappers on release; private temporary reuse still needs opaque composition and completion/AD
retention evidence.

## Current control/effect slice

Register-tiled contraction replacement now requires the existing dense matrix view's exact
two-operand product proof, not just matching operand indices. Additional factors or additive
terms in the reduction body decline this schedule instead of being dropped. A separate retained
epilogue remains supported. This legality check precedes extending physical operand orientations.

Pure counted loops with terminal effects use the existing scalar or product Fold followed by
an ordered exit region. This replaces the single-carry/single-terminal-store restriction.
All carry projections precede the exit; terminal reads and writes retain their source order.
Zero trips still execute the exit using the initial carries. Induction-dependent exits and
effectful recurrence bodies remain outside this extraction. Ordinary ownership analysis, not
the extraction, decides whether an exit can run independently across map items.

Dense but dependent stores enter the existing ordered-effect path rather than being mistaken
for independent map results. A conditional whose predicate reads a destination written by its
branches retains one lexical decision snapshot; branch flattening must not change the decision
after the first store. The snapshot uses the existing typed scalar and effect-region grammar.

Validation: source/JVM comparisons cover moments, simultaneous swapped carries, terminal
read-after-write and reads preceding a terminal overwrite. Generated moments contain one
loop, emit through the OpenCL/CUDA/HIP target dialects, and match JVM results on local Arc
OpenCL and Level Zero for zero, one and seven trips. These are correctness checks, not
performance measurements. Cross-row destination reads retain sequential ordering.

Effect-loop carries now use one ordered `:carries` vector across the canonical dialect, lexical
scope/rebinding, validation, ownership, JVM projection and KernelBody lowering. Counted
effect-only source loops accept multiple carried values; the result-valued source recognizer
still admits a single returned carry. Pure search loops now project multiple exits/recurrence
sites through the existing ordered while-fold. The projection preserves lexical lets and lazy
branches; it does not reinterpret effectful loops as pure Fold terms. Counted effect loops now
accept branch-local recurrence sites when only the induction index advances: both arms must
prove the same unit step, and their complete lexical effect regions stay in source order.
Each stripped arm must contain recognized effects; empty arms still decline rather than
dropping a potentially observable predicate evaluation.
Guard presence is structural: an absent predicate is unconditional, while an explicit false
or nil guard suppresses the region. Projection must not replace this distinction with host
truthiness. Constant-arm elimination uses the existing closed-core branch selector; dynamic
predicates remain KernelBody control. Ownership proofs include entry-guard reads in their
enclosing scope, before branch locals, so cross-row predicates cannot acquire a false
independence certificate merely from row-local stores.
Multi-store branch scopes use the existing guarded regions, including entry snapshots when
their stores can change the predicate. Source/JVM and public OpenCL/Level Zero replay oracles
cover both arms, zero trips, and a mutable predicate; no new kernel node or emitter is needed.

Several source recurrence sites with carried values now use the canonical effect dialect's
typed result-bearing branch, rather than conditional updates reconstructed after
the effects. Each arm yields the same typed tuple after its effects; only fresh merged results
escape. Lexical rebinding, validation, ownership reads, JVM projection and scheduling consume
this contract. KernelBody uses its existing multi-result IfRegion/Yield, with no new emitter
or control node. Branch complete-write proofs conservatively decline; they do not union arm
writes without a coverage proof. Source recognition retains lexical `let`, sequential prefixes
and binary branches until arm effects and the full recurrence tuple can be projected together.
Every terminal recurrence proves the same unit step and arity. An arm may yield values without
writing; the loop as a whole still requires recognized effects. Lexical
regions now have explicit typed results, so a prefix effect followed by local reads can yield the carry
tuple without hoisting those reads across effects. This is an unguarded region with fresh exported
binders, not an implicit escape of ordinary region locals or a synthetic always-true branch.
Its host continuation and target SSA projections must preserve the same evaluation order.
Early exit/recurrence mixtures remain a separate obligation.
Canonical scope/validation and source tests pass (31 tests, 298 assertions); carried-branch
replay, including the post-store local case, passes on OpenCL and Level Zero (96 assertions).
The two duplicate source-order region projectors now share one field-preserving projector.
The shared JVM effect builder accepts
an explicit continuation and loop-body callback: recurrence stays inside the scope of exported
effect results. Both host materialization routes use that one contract; a returned atomic value
feeding the same loop's carry is covered without changing surface syntax or numeric policy.
Canonical tests cover old-tuple swaps, branch-local loads, name collisions, predicate reads
changed by stores, zero trips, and actual OpenCL execution. CUDA/HIP source emission is checked;
native execution is not claimed. Source/JVM and public OpenCL/Level Zero replay compare
branch-local carried updates, a mutable predicate, an effect-empty arm, and zero trips.

Source initialization retains a typed sequential local spine at the loop's effect position.
All recurrence updates see the old tuple after the body effects and yield simultaneously.
No new KernelBody node or target emitter was introduced. The single-carry spelling and the
separate zero/one-carry counted-loop recognizers were removed. Scope transforms must reject,
not truncate and repair, mismatched carry declarations/parameters/initializers.

Validation includes three-carry source/JVM/OpenCL/Level Zero parity, dependent initialization,
zero trips, cyclic simultaneous updates, second-result capture avoidance, malformed arities,
the original city kernels and the existing single-carry/CSR/prefill regression cases. Native
CUDA/HIP execution remains unclaimed; generated sources use their existing compile gates.

The multi-exit search oracle covers the city candidate walk on OpenCL and Level Zero against
the JVM, including immediate bound exit. Pure projection tests exercise distinct exit values,
several recurrence sites and zero-length arrays (untaken reads must not execute). Effects in
guards, exits, lexical initializers or updates still decline scalar while-fold admission.
The shared purity predicate now traverses binding vectors and evaluated collection literals;
quoted data stays opaque. This also prevents beta-reduction from overlooking nested stores or
value-returning atomics. Subtrees with only exits or only recurrences evaluate their local work
in that projection, rather than duplicating it just to compute a constant continuation flag.
Counted product loops and data-dependent while loops share this tail-decision projection.
The old per-carry recursive projector is removed; counted-loop admission still separately proves
unit induction steps and rejects early exits, rather than borrowing the while-loop legality rule.

The shared lexical scope authority also covers existing result-producing atomics: their
inputs use the preceding scope and their result binds only subsequent effects. Substitution
and alpha normalization preserve declarations and static conflict contracts. This closes a
scope inconsistency before generalizing effect-loop result tuples; it adds no new dialect.

## Working rhythm

Public boundary consolidation is still in progress. The city three-carry effect oracle and
the complete linear/MSE/AD/SGD oracle now replay through `gpu.compiled/lower` with the
equation-first compiler on both local backends. Explicit effect outputs retain state across
replays; donated weights consume the previous value handle, while constant training inputs
are not reuploaded. Closing the executable invalidates its result handles. The RK4 scalar
loss also crosses this public boundary. Its `u0` clone is currently staged, not a bindable
public device input; only `target` is exposed as such. This test does not establish replay
with a replacement initial condition.

Do not switch the public default or delete the descriptor orchestration yet. Remaining gates
include retained binding-time admission/tuning evidence and attention dispatch with scratch
through the same public API. Both entry points now share precision-policy validation and the
pre-emission feasibility gate; equation-first retains the resolved schedule instead of silently
stripping the deprecated precision option. This is not candidate parity: its portable contraction
graph still lacks the descriptor route's alternative-selection machinery. The public
`resident-structured-reduction-probe` now retains its recognized reduction plan through equation-first
compilation, fixed-reference scheduling, checked C-family emission and public linking. The reference
path executes on local OpenCL and Level Zero; optimized graph/dispatch selection remains open.
Keep those gaps explicit rather than equating schedule metadata
with implemented optimization. Equation-first compilation captures its target description once
for fusion costs, schedule admission, launch planning and C-family projection; direct low-level
callers may still resolve a descriptor when none is supplied. The equation-first prepared-template
cache fingerprints that same capture and passes it into compilation; each subsequent preparation
captures current facts, so changed calibration creates a new specialization. Unknown/non-data
descriptors fail rather than collapsing into a shared nil cache identity. This snapshot is not an
admission lease against later runtime device changes. Both public preparation paths now use
the shared schedule-override normalizer before template identity: the existing nested
`:gemm-precision` alias shares its template with `:precision`, while distinct precision policies
and default-versus-pinned provenance remain distinct. Conflicts fail before lookup, unknown
keys are retained for validation, and invalid top-level sugar is not hidden by normalization.
This is spelling canonicalization, not equivalence inferred from coincidentally identical kernels.
Quantized Q4_K already has a public compiled
artifact oracle. Reuse these workload tests instead of adding another acceptance framework.

### Remaining reduction boundary, in landing order

The indexed dot/weight/scatter/normalize recognizer already proves a generic
`SegmentedWeightedReductionPlan`. Retain that exact plan in the common `ParallelProgram`
equation spine; do not add an attention-specific semantic API or recover its mathematics from
an emitted ABI. The logical result is distinct from the physical output buffer and is joined
through the existing `:result-storage` contract.

1. Validate the internal algorithm boundary: ordered operands, scalar dependencies, exact
   buffer/result types and shapes, read/write effects, and logical-to-physical output binding.
   An admitted but unscheduled plan must decline explicitly with its equation and source site.
2. Admit protected source markers through the existing sequential description builder. Ordinary
   SOAC runs retain their fusion pipeline; interleave the existing plans without a second source
   walker. Preserve allocation/scalar dependencies and invocation contracts. Internal boundary
   support alone does **not** close public equation-first coverage.
3. Certify fixed-reference `ScheduledKernelBody` emission against the source plan and exercise
   public resident execution on OpenCL and Level Zero, including empty/unequal segments.
   The indexed leaf's exact algebra/layout admission check now lives in the shared schedule
   lowering, used by static reference, dynamic reference, subgroup and backend entry points.
   Direct-lowering tests reject valid general plans with different score, weight or normalization
   algebra. Generic plan validity alone cannot authorize a specialized implementation.
   The reference schedule now constructs a `ScheduledKernelBody` against an independently
   supplied exact-plan graph: ordered pointer/scalar bindings, descriptor footprints, effects,
   int-coordinate limits (including masked tails), checked shape products and byte capacities.
   Literal dimensions are explicitly widened; public int32 shape leaves still decline rather
   than acquiring an implicit conversion. The semantic equation/value validator is shared by
   compilation stages, and C-family emission reads the verified graph scalar interface instead
   of reconstructing it from operation families. Semantic graph construction and equation-envelope
   integration now preserve exact outer values/allocation contracts. Reference completeness is
   proved by rederiving the exact generated schedule (including empty-segment and row-tail stores),
   not from an ABI write flag; the emitted artifact must project that same certificate. Public
   replay matches the independent plan oracle on OpenCL and Level Zero. Arbitrary schedules still
   need their own coverage proof. Entirely empty edge lists now execute on both local backends:
   zero-length logical index buffers receive a bindable native pointer without a visible element.
   Explicit `:schedule {:segmented-weighted-reduction {:strategy :subgroup-score-reuse}}`
   now selects the existing generated subgroup leaf on admitted Intel targets. This authorizes
   its declared Q·K reassociation; edge folds remain ordered. Exact schedule rederivation also
   proves this leaf's active-component writes plus its disjoint head-zero/tile-zero tail writes.
   Artifact serialization and public replay retain that proof. `:auto` still chooses reference
   in the equation-first path; it does not silently promise the compatibility path's dispatch.
4. Generalize the existing executable slot to graph-or-dispatch. Each alternative must retain
   its semantic refinement proof; ABI agreement alone does not establish equivalence. Bind the
   common arguments once and allocate only the selected alternative's scratch.

   A single certified KernelBody graph now carries its verified schedule strategy into the emitted
   executable. The reference and subgroup indexed reduction graphs pass the existing KernelDispatch
   common-interface and pure admission checks with different, explicit numerical contracts.
   This establishes candidate compatibility; public equation calls still bind one fixed graph.
   The remaining step is to carry both certified alternatives through the equation/program/call
   boundary, select and preflight one before allocation, and keep the numerical policy in tuning
   identity. A graph strategy label by itself does not authorize a numerical substitution.

   The reduction-specific `EmittedEquationDispatch` now groups independently validated emitted
   equations behind one `KernelDispatch`. It requires the same semantic plan, physical result
   mapping and complete-write proof for every candidate, an exact default, and an explicit set
   of permitted numerical modes. The emitted program validator also checks every alternative's
   enclosing host-scalar prefix and target, not just the selected candidate. The pure equation
   call now selects and preflights a certified graph from concrete ABI scalars before LinkPlan
   instantiation, and the LinkPlan binds only that selected graph's private storage. An explicit
   numerical policy is still required. The local OpenCL device replay covers both nonempty and
   empty indexed edge lists. The public compiler now offers the explicit
   `:schedule {:segmented-weighted-reduction {:strategy :dispatch-reassociated}}` mode. It
   retains one semantic TypedSOAC program, schedules/emits exact and subgroup alternatives
   independently, and joins only matching reduction equations. Existing `:auto` stays exact.
   Its analytic selector chooses the exact reference below the descriptor-derived component-width
   crossover and the admitted subgroup candidate above it. A selector measured elsewhere can
   override that choice under the stable dispatch ID; its tuning contract retains the allowed
   numerical modes and physical layout. This is selection machinery, not measured performance
   evidence. A public equation-first tuning manifest/benchmark path and emission from one
   multi-schedule program remain subsequent optimizations.
   Unsupported targets decline explicitly. The older resident-descriptor compiler entry also
   declines this equation-only policy rather than silently interpreting it as a pinned schedule.

Invocation shape realization now reuses the graph's checked dimension-to-launch projection.
The shared index algebra lives under compiler IR (the former pass namespace is removed), so
materialization and linking do not interpret arbitrary source or guess types. Shape products are
checked for overflow and exact backing length. Bound scalar preconditions run before session setup
or allocation. Canonical semantic equality preserves NaN bits and numeric widths when rederiving a
certificate; printing is not an equality operation. No surface-language change is introduced.

The source frontend now projects each ordinary numerical equation once, shared by dependency
selection and final construction, and uses one numerical-description predicate for admission
and storage analysis. This removes duplication before mixed-plan admission; it does not change
the supported source language. The plan's older private scalar vocabulary is separate cleanup
debt: reuse its validator now, then converge it on the canonical typed scalar authority rather
than enlarging or copying the whitelist.

Steps 1–2 now have internal/source coverage. The same description/value builder yields maximal
ordinary SOAC runs interleaved with the exact protected plans; fusion, resident scalar handling,
initialization and ownership use one shared component optimizer. Source-facing JVM arrays retain
flat `[elements]` AbstractValues while the plan retains mathematical axes. Only an exact plain
contiguous flattening is admitted; neither arbitrary reshapes nor missing scalar types are inferred.
Initialization belongs to the first component that accesses each allocation, not every subsequent
component. A scratch read-modify-write before and after a plan has one initialization fill.
When the first access is the protected plan itself, its allocation obligation stays in the common
program facts; executable scheduling must realize or explicitly decline it before permitting reads.

The common invocation attachment now handles both ordinary and mixed loop-free programs.
Pipeline admission validates the mixed union before treating it as accepted. The internal
`:segmented-plans?` gate is enabled by equation-first compilation only until executable scheduling
lands; existing descriptor compilation is unchanged, not replaced with an unscheduled path.
There is no fallback after a mixed plan is admitted. Remove this staged gate as part of retiring
descriptor orchestration, after reference execution and certified dispatch are covered.

Both orchestration paths now use one typed contraction boundary validator before emission.
It joins the equation identity and checks dtype, iteration space and physical input/output storage;
the older router's private implementation and the weaker equation-first projection are removed.
This boundary check is not a proof that an arbitrary modified reduction body is equivalent.
ScheduledKernelBody and equation/graph certificates remain necessary for optimized schedules.
The next optimized equation-first slice must preserve those certificates and explicit numerical
policy rather than copying the descriptor route or silently enabling FP16/FMA changes.

Equation-first portable contractions now bind their existing ordered KernelBody to the exact
graph node as a ScheduledKernelBody and use the common target emitter. The graph path no longer
builds a separate portable artifact and then attaches only its source operation. Ordered arguments,
derived scalar conversions, memory uses and realized launch are checked before target projection.
The numerical witness preserves the existing typed SSA evaluation schedule; it is not a promise
of bitwise equality between target compilers. The older descriptor body's source/ABI adapter and
direct low-level artifact tests still exist; removing those requires migrating their callers.

The register-tiled schedule has an explicit graph-certified candidate constructor, exercised
through equation-first emission and public replay for ragged NN and shared-weight NT contractions.
The existing dense matrix view proves the orientation; NT stages physical `[N,K]` storage through
contiguous K loads into the same canonical shared `[K,N]` tile. The source does not need a
materialized transpose or a special projection kernel. The certified variant records this choice.
It reuses the
existing cooperative KernelBody (local storage, barriers and per-thread accumulators); no kernel
source template is added. Admission requires positive FP32 shapes, int-sized storage capacities
and a numerical policy that permits target contraction. Literal and scalar-bound extents share
the same body; runtime positivity, capacity and padded-coordinate obligations are retained in its
certificate and checked before allocation or launch. Mixed static/dynamic shapes still reject
oversized static products at compile time. Strict arithmetic, TN/TT orientations and leading
batched matrix slabs still decline this candidate. `linear-nb` now uses NT directly, including
batch-one and ragged batches with shared weights. The public opt-in is
`:schedule {:typed-contraction {:strategy :register-tiled}}`; the resolved numerical policy must
permit contraction. Declines are explicit, with no portable fallback. Automatic selection is
unchanged, and explicit `:portable` selects the same ordered body as `:auto`. All selections use
the same ScheduledKernelBody certificate and target emitter, without test-only schedule injection.
Do not advertise optimized equation-first defaults or a speedup from candidate construction alone.

Before per-equation contraction dispatch can use that candidate, every alternative needs its own
complete-write evidence. The internal contraction coverage query now rederives the exact generated
portable/register body from the authoritative TypedSOAC algorithm and graph storage/scalar types.
It also checks ordered compiler arguments, scalar bindings, output representation, launch and
register admission guards. Same ABI, output-size labels or candidate-provided widths do not prove
coverage. Static and Long-dimension regressions reject dropped stores, shortened launches, rebound
segment counts and omitted guards. The focused suite passes 21 tests / 191 assertions. This query
is a prerequisite, not a new dispatch or initialization-elision path; compute coverage once when
certifying alternatives, and keep numerical/target admission independent of must-write evidence.

The certified equation-dispatch boundary now also accepts one plain FP32 contraction. Portable
and register-tiled candidates must be emitted from the same retained semantic spine, independently
prove equal complete-write domains, and preserve the external graph and physical result mapping.
The register schedule requires explicit reassociation permission; the default remains ordered
portable. Invocation initialization uses the common proof across all alternatives, never the
default candidate's coverage alone. A re-emitted artifact with a dropped executable store retains
a valid projection but is rejected as dispatch coverage evidence. This is an IR composition
facility, not automatic public candidate generation or an optimized default; per-equation public
selection remains the next slice. Coverage access reuses its validation report within each query
rather than immediately rebuilding the first candidate again; cross-query caching is not added.
The focused dispatch suite passes 7 tests / 35 assertions; the final contraction-only numerical
forgery and initialization checks pass 1 test / 13 assertions. Existing SWR round-trip validation
passes after refreshing the warm REPL's stale compiler-record decoder registry.

Equation-first now offers `:typed-contraction {:strategy :dispatch-register-tiled}` as an explicit
candidate-family opt-in. The frontend and reference schedule stay portable. Each admitted plain
FP32 equation gets its independently certified register alternative from the same retained
algorithm/body and captured target description. Other equations remain unchanged; screening and
actual candidate-admission declines have separate counters. Strict precision does not authorize
the candidate. Binding-time preconditions may choose the exact portable fallback (including empty
dimensions), using the existing dispatch preflight rather than a new runtime convention. The
compatibility compiler rejects this equation-only mode. `:auto` is unchanged.

The explicit public dispatch now declares the existing generic offline tuning contract and
accepts exported selector preferences under `[:typed-contraction :measured-selectors dispatch-id]`.
Every supplied ID must be consumed by an admitted equation; extra/stale IDs and requests for
numerically declined candidates fail loudly. Generic KernelDispatch validation owns strategy,
scalar-expression and ABI checks. The exact portable default and both independently certified
alternatives remain present; `:fallback :none` pins decline until public pruning is implemented.
Reference frontend scheduling receives no measured-selector request, while the original option
is retained in compiler/cache identity. Default/`:auto` selector requests still decline early.

This adds no benchmark side effect to compilation, no new cache or binder, and no numeric rule.
The existing tuning service validates device/artifact/ABI/numerical/layout identities before
exporting a selector. A bare map supplied directly to the compiler is only a legal preference,
not portable measurement evidence; reports label it `:supplied-selector`. Synthetic stationary
durations test tune/export/recompile plumbing without claiming real measured performance.

Mixed reduction/contraction emission revealed that independently re-emitting an unrelated
contraction may generate different private SSA identities. The reduction join now requires exact
enclosing semantic equation equality and retains the reference physical emission; it does not
equate, select, or certify the discarded unrelated schedule. Kernel enumeration includes only
retained graphs and certified alternatives. The mixed public oracle deliberately retains an
ordered double-product/float-result contraction: the register gate must not erase its conversion
terms just to select a matrix leaf. Canonical typed conversion/specialization remains a follow-up;
two simultaneously admitted numerical dispatch families need a separate workload oracle.

The mixed attention/projection fixture now shares one JVM-callable source across admission
and device tests. Its array helpers declare Double methods, so the source inputs are Double;
the previous Float-input synthetic fixture could compile but could not be evaluated on the
JVM. The device invocation explicitly binds FP32 copies under `:dtype :float`, while the
reference invokes the original Double arrays and materializes the Float result. Two resident
replays on each local backend compare all output elements with a 1e-6 absolute tolerance and
require exact zeros for the destination with no edges. The initial exploratory local run observed
maximum error 1.49e-8 on both backends. This validates a composed numerical workload, not two
admitted candidate families: the conversion-bearing projection remains conservatively ordered.

Focused checks for the public candidate slice: static/dynamic admission, exact empty fallback and
strict precision 1 test / 11 assertions; OpenCL/CUDA/HIP source/projection checks 1 / 15; real
Level Zero and OpenCL ragged 65x67x17 replay versus JVM 1 / 4; schedule feasibility 1 / 43. These
are correctness checks, not calibrated throughput or a claim of competitive GEMM performance.
An exploratory warm-REPL variant sharing an allocation extent across protected and ordinary
components reported unavailable initialization extent; reproduce in a clean session before
classifying it, then address host-prefix sharing as a bounded correctness slice if confirmed.
The final clean-session dispatch namespace passes 9 tests / 59 assertions. A bounded read-only
review found no blocker in the semantic-spine join or retained-kernel enumeration. Top-level
emission-route statistics count retained executable artifacts; nested alternative-emission
statistics describe the independent emission attempt and may include subsequently discarded
artifacts. A real device-throughput claim still requires resident, event-timed benchmark evidence.
The final retained-statistics assertions pass 2 tests / 22 assertions.

The shared-extent initialization probe is now confirmed in a fresh REPL. Mixed component slicing
inherited the complete source-binding list, so an extent computed before the protected reduction
was incorrectly classified as a local host binding still waiting to execute. Component scoping
now removes only actual prior-prefix definitions from that local-binding list. Current and later
bindings stay unavailable, and exact TypedSOAC input closure is unchanged. Initializer generation
introduces the genuine incoming scalar capture when needed; no arbitrary metadata-only input or
guessed product equality is admitted. The tiny regression proves both prior availability and
future-definition rejection. The initialization suite passes 22 tests / 137 assertions. The
original mixed probe compiles to three kernels and lowers to four steps with zero driver
allocations. Conservative initialization is retained when an opaque incoming extent cannot
prove the numerical domain product; cross-component product witnesses remain a separate proof
optimization, not part of this correctness fix.

The shared ScheduledKernelBody graph check now requires structural node membership and exact
canonical pointer-storage dtypes. Previously the executable boundary caught a dtype mismatch
after emission; the earlier refinement certificate now rejects it too. The scalar-reduction and
fold-map-specific copies of this dtype check are removed, while their distinct extent and source
equivalence checks remain. Explicit matrix/packed-storage conversions still use separate typed
buffers or value conversions; pointer reinterpretation is not an implicit exception to this rule.

The flat half-input contraction route now preserves an explicitly declared FP32 result through
its matrix KernelBody, OpenCL pointer ABI and descriptor. A half-typed register/portable fallback
may not claim that result: it declines before allocation when the matrix leaf cannot lower the
requested output type. This closes a silent half-output miscompile; it does not yet make the
production matrix scheduler admit CUDA WMMA or HIP MFMA. Those target rows must preserve this
same result contract and carry their own numerical/admission evidence.

ScheduledKernelBody also retains ordered scalar preconditions in the existing checked
launch/storage expression algebra. They refer to physical integral scalar parameters, including
derived bindings, and precede target-specific requirements in the emitted artifact. The projection
certificate rejects dropped or changed conditions. Existing graph/call preflight checks them
before allocation or launch; no second guard interpreter is introduced. This supplies the missing
admission boundary for explicitly requested dynamic tiles, not automatic tile selection. The
legacy descriptor selector derives inverse fallback guards from the same condition builder;
buffer capacity validation remains separate from these bounded-index proofs.

Equation-first execution reporting now observes actual fixed graph bindings through the public
compiled API. It shares compact executable descriptions with descriptor admission reports and
does not infer precision from the requested policy. Loop carry variants are reported once per
distinct binding, not once per iteration; this report is not replay order, timing, or completion
evidence. Dynamic alternative admission and tuning still require carrying those alternatives
through the equation path; an empty admission list with `:selection :fixed` does not prove them.

The current executed workload matrix and bounded projection diagnostic are recorded in
[local compiler evidence](local-compiler-evidence.md). This separates exact/parity gates from
nonstationary timing samples and from still-unexecuted native CUDA/HIP claims.

Memory proof progress: straight-line equation-first replay order now joins real typed complete
writes. An opt-in host-result-only execution scope realizes one full-allocation temporary pair
and checks actual bound order against the selected graph order before launch. Selected order is
projected from validated equation calls before allocation, so this boundary binds once rather
than allocating and binding an otherwise-unused baseline. Runtime observation shares the same
source-step composition. The four-layer OpenCL/Level Zero oracle saves
one allocation and retains two-replay JVM parity. General resident reuse/escape contracts,
structured-loop order and AD tape retention remain open. No existing lifetime pass is
superseded by this narrow GPU proof; keep their distinct JVM/source coverage.

The private host-result boundary now has a reusable callable executor: one binding remains
resident across serialized invocations, with fully checked host input/state updates and detached
output snapshots. `evaluate!` delegates to it. No internal view, resident output lease or AD tape
handle escapes; ordinary inspectable `LinkedExecutable` and `Compiled` behavior is unchanged.
This removes repeated construction within that confined scope, not host output copies or the
remaining resident-composition ownership obligations.

### Interleaved AD consolidation checkpoint (2026-09-29)

Keep this as a bounded interleaved track in the completion order above, not a new prerequisite for
Q4 preparation, local execution evidence, or the remaining pretrained routes. #880–#882 and
#892–#893 close the observed Gaussian bridge shapes: selected differentiable parameters,
interleaved constant observations, constructed priors in let-bound and direct loop initializers,
both public array-read spellings, and a typed compound reduction initializer. They do not prove
that arbitrary source loops or constructed values differentiate.

The current conditional-AD slice corrects a separate semantic defect: shared normalization
must not eagerly evaluate both arms of an `if`. Untaken square roots and checked array reads
are not safe to speculate merely because they do not mutate memory. Reverse AD saves the
selected arm's primal residual; JVP linearizes inside the selected lexical region. The
pullback must consume saved reads, not replay them against potentially changed arrays.
Focused tests cover analytic/finite-difference gradients, JVP/HVP, lexical locals, inactive
reads and mutation between primal evaluation and pullback.

This is not yet a completed compiled conditional-AD vertical. Its selected residual is a host
vector. GPU compilation explicitly declines it with `:ad-conditional-residual-not-lowered`
until scalar replacement or continuation sinking gives it a verified primitive representation.
Do not exempt the vector from typedness checks or guess a scalar tag. Reverse-over-reverse
also needs structural tuple rules or tuple elimination; forward-over-reverse HVP evidence does
not establish that capability. This accepted-domain change replaces unsafe speculation with
correct JVM evaluation and an explicit GPU boundary, not a new GPU performance claim.

The local resident RMSNorm gradient gate exposed an independent compilation failure:
`rms-norm-value+grad-resident-parity` rejects an allocation operand with
`:invocation-prefix-free-value`. It reproduces with unchanged HEAD (`541f6a41`) loaded into
the same bounded REPL, as well as with this conditional-AD slice. No device launch or parity
claim is possible for that gate yet. Keep this in the original workload/ownership campaign:
repair retained allocation expressions across hygienic renaming, with a hardware-free
compile regression and the real resident gradient oracle. Do not weaken lexical validation
or classify it as an unavailable-device skip.

The allocation follow-up retains recognized allocation extents in the existing source-order
dimension table and keeps established known contracts when a later consumer proves an equivalent
shape. Unknown shapes can still refine to known contracts; incompatible shapes still decline.
This prevents equality evidence from introducing a forward/circular scheduling dependency.
Hardware-free regressions cover the SOAC producer/consumer boundary and the public RMSNorm
gradient wrapper on OpenCL/Level Zero targets. The existing Level Zero resident oracle now
executes both gradients, with relative errors about `1.17e-7` (input) and `8.05e-8` (weight).
These are correctness measurements, not a speedup or other-vendor execution claim.

AD free-value discovery now delegates to the compiler's scope-aware `util/free-syms` grammar:
the reverse pass's duplicate recursive symbol collector and JVP's binder-blind tree scan are
removed. Local shadowing and quoted data do not create outer dependence; genuine captured
values still do. This consolidates dependency discovery, not derivative rules or support for
arbitrary effects/loops. Existing unsupported active tape/closure checks remain mandatory.

The scalar CSE body path now uses the same canonical substitution safety check as its
binding initializers. Alias replacement and vector projection expansion must not capture
inner binders or rewrite quoted data. Unsafe body expressions use hygienic alias substitution
and retain vector projections conservatively. Regressions compare original and optimized JVM
values under ordinary/core-name shadowing and nested function bindings. Generated gradient
composition and the resident RMSNorm oracle still pass; this is a scope-correctness fix, not
a numerical reassociation or performance-policy change.

Tuple projection folding also requires an already-evaluated element, not merely a vector
source form. Calls, collection construction and global reads must retain their saved value;
copying their source into an `nth` use can replay mutation or observe later state. CSE now
folds only scalar literals and source-order-proven lexical locals, excluding rebound element
and vector names and retaining other projections. Differential
JVM regressions cover effects in body/initializer projections, array mutation and dynamic Var
rebinding; evaluated local projections still fold. This is the same evaluate-once requirement
as conditional residuals, without an AD-specific rule or effect-based speculation.

1. Converge AD preparation on the same canonical typed scalar/control facts as the compiler:
   one result dtype and effect classification per operation, explicit lexical scope and
   constructor-field projection, and one retained scan/reduction algebra. Remove duplicate
   form-spelling/type recovery only when a workload proves the replacement preserves the
   frontend's dispatch, numeric policy, and failure boundary. Do not add density-specific AD
   dispatch or a second function/type registry.
2. Make the supported-domain boundary explicit: counted carry loops may become ordered scans;
   other loops need a recorded residual/tape or a distinct implicit rule. Unsupported active
   constructors, data-dependent loops, effects, and changing shapes must fail loudly rather
   than return a zero or silently drop a contribution.
3. Gate each new AD shape with primal JVM parity, analytic or finite-difference gradients,
   zero-trip and nonzero-trip cases, `:wrt`/constant observations, and a composed compiled
   caller. Keep device and training-step oracles distinct from a scalar AD unit test. This is
   correctness evidence, not a claim of formal completeness; later proof work can build on
   the typed scope and residual contracts.

### Public generated-contraction event canary

`raster.perf.production-canary/equation-gemm!` (CLI `:case :equation-gemm`) exercises
the public equation-first FP32 path with explicit `:portable`, `:register-tiled`, or
`:dispatch-register-tiled` schedules. It uses the existing resident Compiled/LinkPlan event
measurement service, not a second binder or a host-call timer. Compilation, binding, uploads,
and exact independent pre/post numerical validation stay outside samples. Positive shapes are
bounded to 16M products, 64 MiB logical buffers and K <= 4096; the dyadic input recipe keeps
the reference sums exactly representable in FP32 over this bounded domain.

A local 64x64x64 smoke comparison executed both portable and dispatch schedules on OpenCL
and Level Zero with exact oracle agreement and one launched kernel. Three of four 50 ms
sample sets were nonstationary; this is execution evidence, **not** a calibrated performance
baseline or a speedup claim. Ordinary CI checks the canary lifecycle/failure boundary without
timing assertions. External BLAS/Triton comparisons, representative shape ladders and
controlled interleaved measurements remain needed before claiming competitive throughput.

Return to the original local-first order now: finish public workload coverage and the
support/evidence matrix, audit the still-expensive Q4 host preparation with exact proof
boundaries, then perform matched execution/performance gates before vendor-device claims.

- One bounded JVM/REPL; focused affected tests locally. Full suites run in CI.
- One reviewer at structural boundaries or before landing a risky change; no permanent agent swarm.
- Each slice states which existing path it replaces and what remains unsupported. Tests that
  preserve distinct numerical oracles are not duplicates merely because they cover similar ops.
- Require all expected CI jobs, not only currently posted checks, before squash merging.
- Every claim is labeled designed, implemented, executed or measured. Reconcile stale design
  claims with executable tests before starting another subsystem.
- Changes in accepted source forms are documented; no silent changes to numeric association,
  array length, mutation, ownership or exception policies.

### Final-construction static proof reuse — 2026-10-02

A warm public Q4 preparation probe separates final validation from storage realization:
`program-call/validate!` accounts for about 436 ms, while allocation/alias validation,
bound access facts and initialization analysis together take less than 1 ms. Nine SOAC
physical-result lookups total about 0.11 ms; caching those lookups would not address this cliff.
Inclusive instrumentation overlaps and must not be summed.

Final synchronous construction now accepts the same exact-owner emitted-program proof already
used for staging/storage. A private identity scope is built only after the role/output projection
callback returns. It names the exact program, not a fingerprint or structurally equal copy.
Other programs validate independently. The internal call validator copies sealed physical
projections into a fresh local scope and rechecks all existing concrete step/binding obligations,
with no constructor exemptions. Final role, range, alias and ordered-initialization/effect checks
remain fresh. No additional cache or retained call/plan field is introduced. One-argument public
validation always rederives the complete program, including inside a retained construction scope.

Focused owner/callback/mixed-program/tampering checks pass 3 tests / 53 assertions. Affected
control, LinkPlan, composition/cache and native composition suites pass 103 / 696; both local
OpenCL and Level Zero availability gates were true. Review found no blocker. A subsequent warm
public Q4 diagnostic reports 11.69 ms total, 8.73 ms LinkPlan construction and a 17 microsecond
compiler-template hit. These unmatched shared-load, power-save observations establish removal
of repeated static derivation, not a controlled end-to-end model or kernel speedup.
The public generated/serial Q4 canary at `[1,256,64]` also completes on each local backend,
checking raw float-bit ggml agreement before and after two measured resident replays. Those
minimal diagnostic samples do not establish stationarity or select a tuning winner.

This preserves the current validation algebra; it does not strengthen every preexisting
structured-loop carry-plan or top-level staging-metadata invariant. Whole source-free plan
binding, persisted compiler-build/dependency evidence, external training migration and matched
performance acceptance remain separate campaign obligations. No source semantics changed.

### Source-ordered emitted-call binding consistency — 2026-10-02

The next audit reproduced an independent correctness gap: public call validation accepted a loop
with resolved trip count two while its retained outer `steps` scalar was still three. The loop
graph alone cannot prove repetition, because that scalar may not be consumed by any kernel.
Reconstruction from final buffer bindings was also insufficient: a zero-trip loop aliases its
result to the initial carry and loses the originally requested destination in that final map.

`EmittedParallelProgramCall` now carries explicit `entry-buffers` execution data alongside final
bindings. Validation threads the buffer environment in source order, reconstructs each loop's
canonical binding using the existing `make`, and checks its trip count, invariants, scalar ports,
scratch/rotation and outputs. Numerical steps share one logical ABI/result-binding projection
with construction. Validation checks the retained certified dispatch alternative without selecting
again. Final bindings and exported values must agree with that source-ordered environment.
Scalar comparisons preserve floating bits; buffer tokens retain ordinary equality rather than
being serialized or compared by numerical contents. Storage enumeration and renaming include
entry bindings, including unused zero-trip destinations.

The constructor's synchronous exact-step identity map retains checked post-environments, avoiding
reconstruction of its newly made numerical steps. Independent and retained-static public-call
checks receive no such exemptions. Host callbacks are not rerun. No additional proof cache,
operator registry or source-pattern matcher is introduced. This is an internal call-IR schema
change, not a change to surface Clojure semantics or the supported public compilation API; raw
calls without entry bindings now fail validation instead of being treated as certified execution.

Acceptance includes zero/one/odd/even trips; two consecutive loops including a zero-trip first
loop; distinct multiple carries; numerical consumption of aliased loop outputs; binding/export
tampering; entry-map renaming; no dispatch reselection; and signed-zero/NaN invariant bits.
The affected control, dispatch, LinkPlan and native composition suites pass 120 tests / 889
assertions; the six focused cache-owner/final-proof checks pass 55 assertions. The persistent
REPL required refreshing the artifact namespace's imported record class after an IR reload;
the artifact round-trip and the complete affected rerun then passed. No source workaround was
introduced for that stale-session failure.
This closes the contextual binding gap, not comprehensive host-step operand/evaluation proofs,
general dynamic-loop AD, distributed readiness, or all remaining campaign requirements.

### Hierarchy-derived conservative connectivity — 2026-10-02

The partial-patch acceptance previously built its own small 4x4/dyadic mesh. The next production
seam is `raster.ode.amr-geometry/project-hierarchy`: a pure host-side projection of the existing
validated hierarchy into immutable packed active cells, patch/field/device/local-row provenance,
exact finest-lattice rectangles, geometric faces and CSR incidences. It supports 2D rectangular
domains, anisotropic integer refinement, multiple levels/patches, periodic and homogeneous
no-flux boundaries, positive physical lengths and uniform non-negative diffusivity. Other ranks
and boundary/material contracts decline explicitly rather than changing the AMR provider ABI.

Covered parent rows are indexed under the declared patch-row budget, not tested against every
fine patch. Faces are matched by axis, plane and interval through opposing-side merges. Unlike
the old cell-pair aggregation, distinct periodic faces and self-neighbours retain both signed
incidences. Patch-row mappings explicitly describe packing; they do not transfer ownership or
force host copies. `materialize-connectivity` converts trusted constructor output into fresh
caller-owned arrays for the unchanged finite-volume kernels; it is not independent certification
of an arbitrary caller-supplied map. No compiler opcode, provider registry or handwritten kernel
is added. The dyadic whole-patch multilevel provider remains separate and unchanged.

The native evolution/remap/restart fixtures now use this production projection while retaining
their old bounded all-pairs mesh as an independent geometry oracle. Hardware-free acceptance
covers nonsquare domains, three levels, [2 3] refinement, complete/disjoint refinement, tiny
periodic dimensions, exact tile/side coverage, opposite incidences, patch-order permutations,
nonconstant packing, fresh arrays, no-flux conservation and constant preservation. Six tests
pass 966 assertions, including the final rank/physical-capacity checks. Both native availability
gates are true. The combined geometry, native partial-patch evolution/remap/restart and hierarchy
run passes 28 tests / 1303 assertions; the subsequent focused rerun adds those two capacity checks.

This advances the hierarchy-to-field/connectivity seam, not a complete adaptive simulator.
Existing hierarchy validation still compares patch pairs: the row budget is not a bound on total
validation time. Spatial order at tangentially displaced coarse/fine centres, timestep stability,
device-side patch packing, general conservative remap construction, subcycling/reflux, manifest
publication and real distributed execution remain separate acceptance obligations. Shared face
fluxes and paired incidences establish the conservative structure, not those broader claims.

### Production cell-average layout transfer — 2026-10-02

`raster.ode.amr-transfer/matrix` replaces the fixture-only all-pairs remap builder. It normalizes
source/target cell rectangles to exact rational coordinates, sweeps x events and queries disjoint
active y intervals to visit actual overlaps once. Touching-only cells are excluded; packed row and
column order remain those of the supplied layouts. Both layouts' total area, non-overlap and
per-cell coverage are checked exactly, as are target row-weight sums. Different finest lattices
are supported on the same physical domain. The existing FP64 CSR operator performs execution.

Nonzero budgets reject impossible coverage before event preparation and stop streamed overlap
discovery before storing excess entries. Int capacity includes the extra target row-offset entry;
positive coefficient representability is checked before fresh primitive-array materialization.
Independent small all-pairs rational and finest-tile oracles remain. The affected transfer and
native evolution/remap/restart suite passes 16 tests / 375 assertions; both local backends are
available. Review found no remaining blocker after budget/capacity hardening.

Geometric overlap conservation is exact before FP64 rounding; constant preservation and mass
balance in execution are tolerance claims, not bitwise conservation or high-order reconstruction.
This adds no compiler operation, ownership transfer, manifest compatibility certificate or
publication protocol. Device patch packing, adaptive/subcycled/refluxed evolution, semantic
restore compatibility and external training/distributed acceptance remain on the campaign.

### Semantic restore boundary — 2026-10-02

The strict declared-target gate and bit-preserving numerical certificate policy are documented
once in [durable-numerical-state.md](durable-numerical-state.md#restore-compatibility). Affected
native restart, state, storage lease and AMR checks pass 40 tests / 454 assertions. Correct
snapshot bytes cannot bypass target cell-order/phase/program-policy checks. Compiler-derived
producer evidence, codecs, publication, explicit migrations and external training/distributed
acceptance remain separate; no surface numerical semantics changed.

### Failure-safe synchronous input replacement — 2026-10-02

Linked writes distinguish cold pending inputs from storage tainted by failed backend transfers.
Failure invalidates all overlapping nodes; a no-copy self-write or a copy reading tainted storage
cannot claim recovery. Successful full writes repair the explicitly written node. Borrowed D2D
cleanup preserves the primary error and attaches cleanup failures as suppressed exceptions.

Compiled invocation, profile and measure share one preflight/retirement/write boundary. Strict
object-ancestry private borrowed reads preserve recurrent previous-output inputs under the existing
lifetime lock; public wrappers are retired before mutation. Donations commit after full preflight
but before transfers, including when backend execution subsequently fails. This intentional failure
contract avoids live aliases to partially mutated storage; pure validation failures retain handles.

The affected native link, composition and lease suite passes 42 tests / 303 assertions, with no
remaining pivotal-review blocker. The full seven CI gates remain the merge requirement. This
prerequisite does not implement completed byte receipts, general failed-kernel rollback, cross-owner
mutation tracking, asynchronous input lineage or the remaining external/distributed acceptance.

### Unified synchronous replay state — 2026-10-03

Run, profile and measurement now use one completion/failure boundary; measurement warmups and
probes no longer execute outside replay accounting. Owner-local epochs invalidate continuity on
input replacement, replay/restoration and flush. Failed execution or measurement callbacks poison
the executable while preserving the original cause and allowing close; pure preflight declines
remain nonmutating. Failed transfers retain the narrower explicit reinitialization contract.
These epochs are diagnostic invalidation, not portable producer fingerprints or byte receipts.

Affected link/composition/lease/measurement/ordered-program tests pass 65 tests / 495 assertions,
including native replay, recurrent outputs, all measurement phases, callback failures, unavailable
device timing and idempotent poisoned cleanup. Full CI and pivotal review remain required before
merge. Actual resident-byte producer evidence and the broader eight-item campaign remain open.
Pivotal review found no blocker in this scope. Before producer evidence is admitted, the linked
autotuning bridge must stop launching candidate kernels directly through the session outside
the lifetime lock, leases and epoch/poison boundary. Raw-session mutation remains out of contract;
Raster's own linked tuning must use a tracked exclusive mutation scope rather than inherit that
exemption. No continuity claim is made for tuned instances yet.
The tuning review also exposed reentrant poisoning: a restore callback can catch an inner
failed replay. Poison is now first-failure-wins, retaining that original backend cause through
the outer already-poisoned wrapper. A focused nested-restore regression includes safe close.

### Linked autotuning mutation boundary — 2026-10-03

The linked bridge now enters one existing-lock exclusive mutation scope for candidate callbacks
and launches. Live output leases reject before tuning begins. Scope entry and successful exit
invalidate continuity; no candidate receives full-plan completion credit. Callback failure
retains the original cause and poisons the owner. A callback may replay for restoration but
cannot export an intermediate output lease. Even a tuning-cache hit conservatively invalidates
readiness: callers replay the plan before exposing a result. This closes Raster-owned tuning's
raw-session bypass, not arbitrary direct session mutation or producer-byte certification.
Focused lease/dispatch-benchmark/measurement/program-tuning tests pass 42 tests / 308 assertions in the retained
capped REPL. Review and full CI remain prerequisites; completed byte evidence remains next.
The caught-inner-replay regression verifies first-failure identity, removal of the temporary scope
marker, and idempotent poisoned close. The parent replay fix is shared, not a tuning-only exception.

### Retained instantiated artifact owner — 2026-10-03

`Compiled` now retains the original `Prepared` and reuses its exact-object seal mechanism for
explicit structural inspection. There is no new cache, eager identity hash or byte-attestation
claim; copied owners still fail closed. The contract is recorded once in
[durable-numerical-state.md](durable-numerical-state.md). Artifact/composition/lease tests pass
41 tests / 323 assertions in the existing REPL; pivotal review found no blocker. Actual resident
byte snapshots, completed producer receipts and publication remain on the original campaign.

### Offline resident-byte producer evidence — 2026-10-03

The next explicit boundary snapshots actual post-transfer inputs and post-replay outputs under
the existing ownership lock and output lease. It reuses bounded SHA-256 content hashing and the
retained artifact identity; no second cache or ordinary-invocation readback is introduced. Mutable
state chains only across uninterrupted same-owner epochs with matching actual bytes. Admission
and fault tests cover refreshed defaults, device copies, NaN payloads, bounded ranges, externally
owned sessions, asynchronous events, prologues and failed transfer/readback cleanup. Pivotal review
required removing an unsupported host-byte-order label; raw receipts now explicitly retain opaque
device-native representation. Focused content/artifact/composition/lease/native tests pass 51 tests
/ 541 assertions in the retained capped REPL; full CI remains a landing prerequisite. Publication
and cross-device codec/restore matching remain open.

### Verified raw-array byte codec — 2026-10-03

Production content runtime now decodes verified `:raw-array` chunks into caller-owned segments,
with explicit target dtype/byte order, exact shape-derived extents and source/destination alias
protection. It uses the existing dtype table and content verifier, not another representation
registry. Bounded endian reversal preserves raw element bits and does not infer GPU storage
semantics. Runtime tests pass 13 tests / 218 assertions, including both orders, NaN/signed-zero/half
patterns, staging boundaries, digest failures, overflow, partial overlap and closed destinations.
Pivotal review found no blocker. Full CI remains required; publication and independent semantic
restore matching are not established by this codec alone.

### Owner-bound resident storage evidence — 2026-10-03

The new measurement API uses the existing exact-artifact seal and LinkedExecutable exclusive
mutation guard, not another session convention or cache. An original Compiled owns generated
probe scratch/binding/readback; success leaves no scratch graph/buffer/event, invalidates old
output readiness/wrappers and does not credit a program replay. Runtime emitter selection comes
from the existing resident-backend descriptor. Actual selected device/driver facts are queried
directly on OpenCL/Level Zero, not inferred from catalogue scheduling estimates.

Completed storage descriptions require a live original receipt and matching original dtype facts
from the same exact executable/session/session-id and program. Portable data labels the already
addressed physical leaf bytes with explicit raw-array order; it is not a new authority token,
logical quantization descriptor, state commit or universal codec proof. Pure admission failures
do not mutate; surfaced in-scope native/cleanup faults poison the owner and preserve primary errors
while attempting both graph/scratch API releases. Existing shared destructors can swallow native
destruction failures, so strict native destruction/failed-resource retention is still explicit
runtime cleanup debt, not a reclamation proof from these tests. Identical ZE registration retains
cached module/kernel/staging; native repeated measurement checks that the registry handle survives.
Registration-time target/source/compiler/arena identity distinguishes source-compiled cache bytes
from explicitly supplied SPIR-V. Explicit payloads are cloned and fingerprinted on registration;
source-only and explicit modes cannot alias after lazy loading. Six hardware-free registry tests
cover both transitions, target/arena/source/payload changes, unknown provenance and caller mutation.
Focused owner, fault, native, output-lease and backend suites pass
20 tests / 289 assertions; actual OpenCL and Level Zero integration has no local native skip.
Semantic field selection, bounded provider staging and producer-derived manifest construction
remain the next seam. General device-side canonical packing remains a separate IR vertical.

### Generated storage representation probes — 2026-10-03

The resident producer path hashes device-native bytes and deliberately supplies no byte order.
The next typed-manifest seam now has generated probe machinery: two asymmetric finite sentinels
per canonical dtype, verified KernelBody stores, common OpenCL/CUDA/HIP artifact emission with
compilation requirements retained, and independent explicit byte-pattern classification. All six
types have real local OpenCL and Level Zero byte readbacks and resource cleanup checks. Focused
oracles: 6 tests / 207 assertions, zero failures/errors; no native skip on this machine. The new
native namespace is selected by the existing OpenCL CI gate; all six sources join CUDA/HIP fixtures.
Optional Level Zero FP16/FP64 tests query the exact live device's core module flags; absent
capabilities are reported, while query and compilation failures remain failures. Strict classifier
admission rejects nonintegral/out-of-range observations before signed-byte normalization.
The CI raw-skip debt ratchet caught an unaccounted optional-capability print site. It now uses the
shared Level Zero skip reporter and ledger, with missing/true/unknown facts and non-live-device
skip attempts rejected. The affected storage/CI suites pass 8 tests / 363 assertions.

This is not a new cache, dtype registry, handwritten kernel or producer certificate. Pure probe
observations grant no authority. Next: seal measured facts to the exact live execution owner and
probe artifact, then project completed bytes into typed manifests. Device-side canonical packing
and general bitcast semantics remain a separate compiler vertical; do not infer them from finite
sentinels or host endianness. Existing ordinary invoke and compilation paths stay unchanged.

### Ordered numerical-state availability finalization — 2026-10-03

The content runtime now verifies/localizes/promotes each certified manifest chunk in bounded
sequence and invokes an external metadata callback only after matching durable placements. It
reuses the manifest/content/provider contracts rather than adding a store or compiler cache.
Rejected provider-event handoffs drain through the originating provider; accepted-event field
validation is shared with consumption. Source leases close before promotion, and callback or
validation errors retain their identity with cleanup errors suppressed. The provider's safe-drain
obligation explicitly includes never-awaited handoffs. Combined publication/content/state tests
pass 33 tests / 618 assertions in the existing REPL. Pivotal review required accepted-event cleanup
to bypass fresh descriptor validation: an unavailable or changed descriptor must not prevent the
originating provider's raw safe drain. Counterexamples cover pre-await failure and post-await drift,
including primary/suppressed cleanup errors. The final review found no remaining blocker; full CI
remains a landing gate. Multi-field/repeated-content tests preserve order and keep opaque runtime
receipt metadata out of the certified compiler state.
This finalizes provider-declared availability, not producer authentication, codec verification,
parent existence or transactional metadata publication. Orphan blobs and acknowledgment loss are
explicit failure cases; receipt-to-manifest integration remains on the eight-item campaign.

### Bounded source ingestion into the existing content lifecycle — 2026-10-03

The missing producer-to-provider handoff is an optional ContentIngestor capability alongside
ContentProvider, not a replacement store/session or a required method on existing providers.
Explicit target-tier preflight precedes provider/source contact. Provider-owned writable windows
are at most 64 KiB; source callbacks consume a contiguous complete stream synchronously on the
submitting thread. Callback failures are sticky, including caught/retried and reentrant failures.
The final callback verifies SHA-256 before returning success; empty-source digest validation is
preflight. Every submission exit expires the callback and clears its source-reader reference.
Provider windows are exclusive borrows during the callback; neither side may retain the other's
source/window beyond that scope. These are trusted implementation obligations, not inferred
escape proofs.

The existing StorageEvent represents pending placement, not a source borrow or independent
durability proof. Rejected returned events drain through the originating provider with cleanup
errors suppressed onto the primary. Submission failure without an event leaves native/provider
cleanup internal to that provider. The synchronous ingest-content! helper checks exact
provider/tier/content placement and drains before returning. A source arena can close before
placement await. Ingestion attests the supplied stream only; the existing reopen/extent/SHA
verification remains required before durable availability and metadata publication.

Pivotal review found that a provider could catch a reader/digest error then replace it with its
own submission error. The wrapper now captures both outcomes, preserving the exact first callback
fault and suppressing a distinct provider error without inventing an event handoff. Expired reader
closures also clear fault data so retained callbacks cannot keep source objects through an error.
Focused ingestion/content/publication tests pass 35 tests / 698 assertions in the capped REPL.
This adds no producer provenance, numerical field selection, transactional metadata publication
or completed-receipt-to-manifest vertical; those remain the next integration work. The original
eight-item campaign, external training acceptance and AMR/subcycling requirements remain open.

### Completed resident fields to numerical manifests — 2026-10-03

`raster.runtime.resident-state/capture!` connects original completed execution receipts and
owner-bound measured storage facts to the existing manifest and content-provider lifecycle.
It introduces no compiler cache, session convention or persistence implementation. All selected
fields are validated and the manifest certified before provider writes. An extra existing Link
output lease pins the entire synchronous capture, including placement await, even if a provider
callback closes the caller receipt. Historical input frontiers are not current state sources.

Version 1 accepts exact positive plain contiguous tensor leaves and emits one complete raw-array
chunk per field. Provider byte windows may be unaligned: downloads align outward within the
selected leaf and bounded scratch storage, copying only the requested bytes to the provider.
Byte order comes from owner-bound generated probe evidence, not the host. Semantic names,
coordinates, parent state IDs and numerical policy are application declarations; completed
producer fingerprints, bound schedules, representation facts and ordered field-to-node/content
bindings are derived. Placement events stay outside the certified numerical manifest.

The lifetime monitor covers receipt validation, pure plan certification and lease acquisition
only; provider submit/await runs outside it under the private lease. A cross-thread provider
callback can therefore acquire the monitor and promptly receive the lease-active decline rather
than deadlock with capture. Known failure releases the private lease; unknown cleanup retains explicit cleanup authority.
Later-field failure may leave earlier verified orphan blobs and returns no state. Availability
finalization still independently reopens, hashes and promotes the chunks before invoking external
metadata publication. Neither capture nor finalization is transactional publication, codec
verification, mathematical equivalence, or a proof of parent existence.

Focused hardware-free and actual OpenCL/Level Zero tests pass 10 tests / 117 assertions. Generated
periodic heat evolution is captured, the producer session is closed, and a fresh execution resumes
from the stored bytes through the public verified raw-array decoder after availability finalization.
Same-backend continuation matches uninterrupted execution exactly; JVM
comparison uses the declared FP64 tolerance. These native tests use an explicitly synthetic
packaged build identity and a bounded test provider, not release-build or production-store
validation. The existing multilevel/AMR operators remain unchanged. General device packing,
partial-chunk codecs, scalable AMR/subcycling/reflux and external training acceptance remain open;
this slice does not complete the eight-item campaign.

### Coarse/fine restart consumes completed producer evidence — 2026-10-03

The existing full-domain refined-heat continuation fixture no longer handwrites a producer string
or assumes little-endian storage when constructing its manifest. It uses the same public completed
receipt, measured representation and resident-state capture boundary for the coarse output and
evolving fine post-state. Its original generated four-fine-step/restriction program, JVM reference,
six-step uninterrupted baseline, actual mapped-file leases and mass checks remain independent.

Each of three replays captures both fields and finalizes availability before a small test metadata
callback records the manifest. That callback checks that declared state parents already exist;
the child also checks its execution-parent fingerprint against the preceding captured replay.
Numerical state parents and execution replay parents stay distinct. Runtime tensor shape is the
actual dense one-dimensional buffer; declared grid geometry is retained in field coordinates,
not silently reinterpreted as a compiler-certified reshape. All producer sessions close before
the final files are mapped for fresh continuation. Native OpenCL/Level Zero and host checks pass
3 tests / 64 assertions, including same-backend exact coarse/fine restoration and conservation.

These are synthetic packaged-build and bounded provider/metadata fixtures, not release-build
authentication, transactional distributed lineage or a production persistent store. Mapped-file
realization independently verifies content before upload. Partial-patch producer integration,
device packing, scalable hierarchy validation, temporal interpolation/subcycling/reflux and
external numerical comparisons remain open.

### Captured partial-patch restore boundary — 2026-10-03

The restore integration keeps the generic four-facet exact-provenance verifier unchanged. A
separate fixed composition first verifies independently declared complete field geometry, phase
and numerical policy, then resolves ordered source field keys through the original sealed
producer Prepared. Source nodes and identity come from the existing certified byte frontier,
not the continuation or an incoming-provenance subset. Current output/post-state exports remain
distinct from historical inputs. Aliased public ports decline because node-only v1 capture
cannot distinguish their keys; real composition already rejects duplicate output nodes.

Each selected field must be one complete plain raw-array chunk with the exact producer dtype,
shape and content reference. Its representation kind, dtype, program and normalized byte order
must agree internally. Dynamic completed fingerprints, schedules and measured observations are
audit data, not target predictions or serialized authentication. Numerical-state parents and
execution-replay parents remain separate; no new store, session or cache convention is added.

The partial-patch remap/restart fixture now captures its original composed midpoint producer,
finalizes content availability, closes the source session and restores from independently
verified mapped bytes. Independent geometry/active ordering, wrong-same-extent rejection,
host evolution/remap, mass and exact uninterrupted 17-stage continuation checks remain. The
capped REPL passes 82 affected tests / 1,404 assertions, including OpenCL and Level Zero and
rejection against a real foreign sealed source program. Packaged build identities and bounded
providers remain explicitly synthetic. Final independent review and all seven exact-head CI
gates are still required before merge. Device packing, scalable hierarchy validation,
subcycling/reflux, external training acceptance and real fabric execution remain open.

### Actual local AD to collective to SGD boundary — 2026-10-09

`distributed-plan/refinement-plan` assembles the existing checked projection with caller-owned
local producer/consumer LinkPlans and generated combine evidence. It derives no identity input
or export kernels. Terminal refinement copies target consumer input storage directly; the root
combine and its consumer share the same exact dense physical realization. Same-worker staging
would remain ordinary local compute, not a fabricated self-link. Producer order follows the
retained participant vector, and incoming declarations cannot shadow derived contribution SSA.

The differentiated fixture uses ordinary public `value+grad` over a linear prediction and MSE
loss, with parameters selected by `:wrt [0]`. Local mean gradients are multiplied by their local
sample count before the declared FP32 sum tree; SGD divides the learning rate by the global
sample count. Unequal batches `[1 3]` and `[1 2 4]` prevent equal-size averaging from standing in
for this contract. Public equation-first lowerings are namespaced through existing certified
LinkComposition before ordinary, unsealed LinkPlan endpoint rebinding and validation. There is
no separate training IR, execution loop, cache or GPU session convention, and no host gradient
computation or download/re-upload between the actual device producers and updates.
This is compiler acceptance evidence using checked LinkPlan rebinding, not yet a released
public distributed trainer interface.

This is one complete synchronous update with initially identical parameter replicas. It does
not prove multi-step state reuse: the distributed owner is deliberately one-shot. Repeated
training needs explicit canonical parameter state and a checked unrolled/replay contract, not
calling `run!` again with stale initialization evidence. Co-located logical workers do not
demonstrate multi-host/fabric performance; timings in this fixture are analytical declarations.
The held original real-model gradient gate, external transformer training acceptance and the
rest of the eight-item campaign remain open.

AD type-propagation debt found by the JVM reference: a local `nth` projection from the generated
`value+grad` tuple, immediately consumed by `broadcast`, can emit an `Object[]` access for an
actual `float[]` and throw `ClassCastException`. The explicitly Float fixture retains `^floats`
on that binding. Its GPU lowering succeeded even without the tag; this is not evidence that the
JVM boundary was sound. Follow up by retaining the selected gradient slot's primitive-array
type through the ordinary typed tuple projection, with unannotated JVM/native/GPU regression
coverage. Do not add an AD-specific array/function registry or silently remove the CPU oracle.

### Sound erased-array JVM boundary — 2026-10-09

The AD fixture's unannotated tuple projection exposed a general JVM emission defect, not a
special meaning of `nth`: unknown array storage was assumed to be `Object[]` before `aaload`
or `aastore`. A primitive array cannot satisfy that cast. The JVM emitter now delegates erased
reads/writes to Clojure's existing runtime array dispatch, using the existing boxed-call emitter
for once-only, source-ordered operand evaluation and standard coercion/exception behavior.
Retained concrete array types keep their direct JVM load/store instructions; their checkcasts
reuse the existing JVM type-descriptor authority instead of duplicated array-class tables.
Nested array operations likewise retain all arguments rather than using a one-dimensional
instruction on a variadic call.

The same distributed AD/collective/SGD fixture now omits `^floats`; its JVM analytic oracle and
actual device updates remain unchanged. Separate JIT/AOT checks compare all primitive-array
kinds and reference arrays against the ordinary Clojure runtime, with independent pre-store
arrays, raw NaN/signed-zero bits, invalid accesses, source-ordered side effects, nested indices
and direct-typed-path dispatch tripwires. This fixes erased JVM execution correctness. It does
not pretend that the AD tuple's missing static slot type has been recovered: retaining that
information remains an optimization/type-propagation follow-up, and GPU/native layouts still
require their existing static admission proofs. No numerical policy or AD-specific registry is
introduced, and the original actual-model training gate remains held.

### Narrow lazy-JIT AD normalization — 2026-10-09

Direct AD applications in a walked straight-line `let` now consume the same prepared typed
reverse program and known tuple projections as AOT. This occurs after TC walking and before
optional SIMD, without rewalking the retained conversions, expanding ordinary helper or ftm
boundaries, or introducing a result-type registry. Earlier local types participate in overload
selection. Argument evaluation uses the shared call-by-value lifting contract; projection
rewriting respects the canonical lexical scope description and quoted data.

The admitted lazy-JIT slice does not hoist nested applications across branches, scopes or
other operands, expand loop bodies, or perform D-algebra rewriting. Those calls retain runtime
behavior. Indirect applications may retain their explicit constructor binding because this
step performs no dead-code elimination; it does not claim construction-free execution for
that shape. General non-inlined aggregate result typing remains open. New scalar and Float-array
fixtures use runtime-constructor and boxed-array-read tripwires, while mixed Float/Double
checks retain raw-bit conversion oracles. This is not real-model gradient acceptance or a
numerical-policy change, and the original training gate remains held.

### Complete public results versus typed island live-outs — 2026-10-09

A fresh public linear-prediction/MSE probe exposed a silent equation-first miscompile: the
function declares a scalar loss, but its emitted program returned the prediction vector and
omitted the target input. Internal TypedSOAC island extraction correctly retains values needed
by an opaque host consumer; those live-outs are not a certificate for the complete function.

Whole-program promotion now checks the retained source return against the logical outputs and
their existing result-storage relation. Host-controlled bindings must have an invocation or
equation executor, or the existing effect analysis must prove them removable. Diagnostics retain
the unsupported return/binding and source metadata. Island extraction and its host materialization
contract are unchanged; no loss-name recognition or host download/continuation is introduced.

The unsupported scalar objective therefore fails before allocation instead of returning a wrong
array. This is a correctness prerequisite, not completed loss support. Next, lower the complete
objective through the ordinary typed reduction/scalar algebra and restore positive JVM/device
loss parity. Keep the public compiler-default switch blocked until whole-source coverage and
the remaining measured dispatch/workload contracts are established. The original model gate
and the full campaign remain open.
