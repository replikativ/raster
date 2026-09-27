# Local compiler consolidation

Status: agreed landing order, 2026-09-27. This is the near-term completion ledger for
the compiler north star, not another architecture or a replacement for its longer roadmap.

Finish the general compiler work on OpenCL and Level Zero first. Continue CUDA/HIP
hardware-free compile gates; native vendor runtimes and hardware acceptance follow the
local milestone. FPGA spatial scheduling and distributed optimization are later tracks.

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
   reusable numerical, storage and execution contracts.
4. **Local consolidation gate.** Publish a support/evidence matrix, retire covered duplicate
   paths, and record reproducible correctness, compilation, allocation, transfer and execution
   baselines on both local backends. Retain city irregular kernels, one PDE/stencil and a
   resident AD training step alongside pretrained. Existing tests are starting points, not
   evidence that every target or shape was executed.
5. **Cross-vendor acceptance.** Add native CUDA, then HIP execution through the same ABI,
   artifact and event contracts. Rent hardware after the acceptance runner is ready, or
   earlier if a concrete hardware question blocks the design. Compilation, execution and
   competitive performance are separate claims.

The equation-first public result boundary now includes explicit outputs, donations and taps in
the certified LinkPlan outputs, not only in runtime wrappers. This joins escape declarations to
the existing initialization and storage analysis; it does not enable reuse through inspectable
resident handles or promise versioned AD tape snapshots.

## Current control/effect slice

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
writing; the loop as a whole still requires recognized effects. A prefix effect followed by
result-bearing locals that would escape an ordinary region currently declines, rather than
hoisting those locals across effects. Early exit/recurrence mixtures also decline.
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
graph still lacks the descriptor route's alternative-selection machinery. A direct probe of
`resident-structured-reduction-probe` also declines equation-first semantic coverage, while the
descriptor route accepts it. Keep those gaps explicit rather than equating schedule metadata
with implemented optimization. Equation-first compilation captures its target description once
for fusion costs, schedule admission, launch planning and C-family projection; direct low-level
callers may still resolve a descriptor when none is supplied. The equation-first prepared-template
cache fingerprints that same capture and passes it into compilation; each subsequent preparation
captures current facts, so changed calibration creates a new specialization. Unknown/non-data
descriptors fail rather than collapsing into a shared nil cache identity. This snapshot is not an
admission lease against later runtime device changes. Cache canonicalization of equivalent policy spellings
remains a follow-up. Quantized Q4_K already has a public compiled
artifact oracle. Reuse these workload tests instead of adding another acceptance framework.

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
through equation-first emission and public replay for a ragged NN contraction. It reuses the
existing cooperative KernelBody (local storage, barriers and per-thread accumulators); no kernel
source template is added. Admission requires positive FP32 shapes, int-sized storage capacities
and a numerical policy that permits target contraction. Literal and scalar-bound extents share
the same body; runtime positivity, capacity and padded-coordinate obligations are retained in its
certificate and checked before allocation or launch. Mixed static/dynamic shapes still reject
oversized static products at compile time. Strict arithmetic and unsupported layouts (including
the transposed weights of `linear-nb`) decline this candidate. The public opt-in is
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

- One bounded JVM/REPL; focused affected tests locally. Full suites run in CI.
- One reviewer at structural boundaries or before landing a risky change; no permanent agent swarm.
- Each slice states which existing path it replaces and what remains unsupported. Tests that
  preserve distinct numerical oracles are not duplicates merely because they cover similar ops.
- Require all expected CI jobs, not only currently posted checks, before squash merging.
- Every claim is labeled designed, implemented, executed or measured. Reconcile stale design
  claims with executable tests before starting another subsystem.
- Changes in accepted source forms are documented; no silent changes to numeric association,
  array length, mutation, ownership or exception policies.
