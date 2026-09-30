# Local compiler consolidation

Status: agreed landing order, 2026-09-27. This is the near-term completion ledger for
the compiler north star, not another architecture or a replacement for its longer roadmap.

Finish the general compiler work on OpenCL and Level Zero first. Continue CUDA/HIP
hardware-free compile gates; native vendor runtimes and hardware acceptance follow the
local milestone. FPGA spatial scheduling and distributed optimization are later tracks.

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
KernelBody kernels, without a new attention ABI or emitter. CI and external ASR migration
remain open. Its contract is:

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
114 tests / 574 assertions locally; full CI remains required. The snapshot's producer string is
a test-fixture label, not a production compiler-build identity. Production manifest publication,
partial-patch interface fluxes, actual distributed transport, subcycling and reflux stay open.

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
admission lease against later runtime device changes. Cache canonicalization of equivalent policy spellings
remains a follow-up. Quantized Q4_K already has a public compiled
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
