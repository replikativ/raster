# Compiler unification campaign

### October 9 physical compute lanes follow explicit worker placement

A reproduced simulator gap gave independent compute lanes to logical workers
explicitly placed on the same physical target. Compute steps and transfer
`:serialized-on` claims now share one nonrecursive worker-to-target projection.
Claims resolving to the same lane are deduplicated. Unplaced workers and
topology-only relay identities retain their original lanes; distinct physical
targets remain independent. Route-resource certificate witnesses use the same
projection and reject placement changes even when makespan is unchanged.

Logical compute totals, transfer bytes and per-worker peak-memory reports remain
unchanged. `:physical-compute-ns` additionally reports accumulated compute service
by physical target, not transfer-inclusive busy time. Physical memory-capacity
aggregation, link-sharing discovery and actual asynchronous distributed execution
remain separate obligations. This changes planning resource accounting, not the
surface numerical program, kernel emission or source arithmetic.

### October 9 explicit empirical route-cost simulation

The existing simulator now accepts optional live route context, supplied profile
reports and an explicit admission policy. Matching includes physical endpoint
placement, device/hardware signatures, directed route facts, bytes, realized
layout (not allocation identity), transport and staging budget. Evidence must
have consistent before/after snapshots, fresh endpoint sessions, finite positive
whole-step durations and bounded Unix-epoch age. The shared Measurement summary
supplies CV/spread and median costs. Insufficient, inconsistent, stale, repeated
or noisy evidence leaves an explicit per-route analytical fallback.
The policy's `:cold-warm` label is caller-declared summary metadata; these
observations do not independently establish sample warmness.

This is a pure diagnostic projection, not a new compiler cache or authority:
the original one-argument simulator and all plan certificates remain analytical.
Supplied reports are not authenticated, synchronous timings include observation
overhead, and admitted durations neither prove overlap nor model congestion.
Native fresh-owner OpenCL/Level Zero controls compare projected durations with
the actual captured samples using a deliberately permissive spread policy for
nonflaky plumbing coverage, not a production calibration claim. Production
stationarity, load-regime validation and overlap-conditioned route costs remain
open, along with the unchanged real-model training acceptance.

Focused simulator, compute-binding, AMR-plan and profile regression suites pass
80 tests / 625 assertions. The native route-admission control passes on both
local backends (1 test / 34 assertions); it checks projection of observed sample
medians, not held-out predictive accuracy. Reports can be supplied to
`simulate` with `:route-context` obtained from `raster.gpu.distributed/cost-context`
on a fresh ready owner and an explicit `:route-policy` containing `:now-ms` and
`:cold-warm`. Existing certificates never inherit these simulation options.

### October 9 distributed profiling hardware-evidence context

Distributed profiling now retains the existing HardwareDescriptor evidence
signature for each physical session, alongside live backend device/driver and
session identity. Queries use the session device, not a logical worker label.
Before/after calibration-version or admitted bandwidth drift invalidates the
observation; normal unprofiled execution is unchanged. Focused hardware-free
and real OpenCL/Level Zero context tests cover this boundary. These signatures
are necessary context for measured route-cost admission, not measurements of
the fabric, automatic topology discovery or an adaptive cost overlay. Exact
route/byte/transport/staging matching and noisy/stale sample admission remain
the next distributed calibration slice; analytical simulation remains a seed.

### October 9 shared physical service lanes

Directed links may now explicitly declare `:attributes :serialization-domains`,
a vector of distinct keyword identities. A transfer occupies its route links,
the union of their shared domains, and any existing endpoint `:serialized-on`
compute claims. One resource-occupancy update drives all these lanes; byte
accounting remains per route link and logical transfer, not per shared domain.
No domains means the existing independent-link behavior. This is conservative
exclusive serialization for declared physical sharing, not automatic discovery,
bandwidth sharing, switch congestion, or a calibrated fabric prediction.

Certificates retain each transfer's derived resource claims as well as costs:
renaming a shared domain invalidates an old witness even when numeric makespan
is unchanged. Warm-REPL distributed-plan, compute-binding and AMR-plan suites
pass 67 tests / 424 assertions, covering opposite-direction routes, independent
domains, empty/default declarations, malformed declarations and multi-hop claim
deduplication. Measured route service curves and production distributed execution
remain open alongside the original training and adaptive AMR obligations.

### October 9 physical topology admission

Reconstruction of a topology now reruns the existing device/link constructors,
not just their record-type and endpoint checks. Before this fix a modified link
with NaN bandwidth simulated and freshly certified with a 700 ns makespan in
the data-parallel fixture. Nonfinite/nonpositive bandwidth, invalid latency,
memory capacity, descriptor and attributes now fail through the existing field
contracts. Modified topology indexes must also match their contained record
identities and retain actual device/link maps. Constructor, simulation, fresh
certification and verification regression cases cover these boundaries. Pre-plan
route-cost queries also validate each visited link and its index identity, without
rescanning the entire topology for each transfer.

The affected hardware-free plan, compute-binding and AMR-plan suites pass
65 tests / 405 assertions. No valid constructor API or numerical policy changes;
this rejects malformed physical claims, not measured-but-inaccurate costs.
Explicit shared-link serialization domains, calibrated fabric costs, the held
training gate and the remaining distributed/AMR workload obligations remain open.

### October 9 distributed capture and physical-model reconciliation

PR #1136 is merged as `4fb7ec5b` and #1137 as `9f39cd6e`, each after independent
exact-head review and all seven registered CI gates. Original distributed owners
now retain physical allocation budgets and owner/session-bound measured storage
facts. Completed bounded AMR cycles capture both whole fields through the existing
manifest/provider contracts in a pinned read scope; every download and the final
provider callback boundary revalidate the live representation snapshot. Coordinates
are deliberately closed to nonnegative synchronized step advancement, not arbitrary
caller time labels. Final capture controls pass 5 tests / 40 assertions; the actual
OpenCL/Level Zero capture/close/decode/fresh-context oracle passes 1 test / 14
assertions without capability skips.

PR #1138's independently reviewed rebased head is
`843e04aab393a6b7e458aa1ff69a530f3fc242cc`; its fresh CI remains pending.
Source-compiler/target-semantic compatibility is checked before opening content,
then bounded fixture files are independently verified through read-only mmap leases
and decoded before fresh-context continuation. Pure and native tests pass 3 tests /
25 assertions without skips. The provider and explicit file materialization remain
test fixtures: this does not prove production durability, metadata atomicity or
durable parent existence. Use the existing availability finalizer before an
application-owned publication transaction.

The distributed physical model is not wholly overlap-blind: declared backend
transfer facts distinguish submission, physical queue independence/serialization,
event completion, staging/host lease requirements and peer mechanisms. An explicit
transfer `:serialized-on` claim occupies the relevant compute lane in simulation.
Directed-link bandwidth/latency remain analytical inputs, not discovered/measured
fabric facts. Shared NIC/PCIe/switch contention domains, message-size curves,
load-conditioned calibration and actual asynchronous distributed execution remain
open. Never infer physical overlap merely from asynchronous API shape.

Device-level calibration already feeds the single HardwareDescriptor and its
evidence signature; it is not yet distributed route calibration. The CPU
microbench retains nonstationary bandwidth as `:measured-noisy` for inspection.
Since #1139, those observations do not replace planning estimates: the existing
descriptor authority admits only positive finite fields tagged `:measured`,
rejects explicitly nonstationary bandwidth, and requires complete, alias-consistent
peak-FLOPS families. This admission does not establish an expiry policy or immunity
to systematic bias. Power/load regime, calibration validity and fabric contention
still need explicit evidence before treating estimates as reliable production-selection
facts. Reuse existing Link device-event profiling when
instrumenting distributed execution, rather than measuring repeated mutations of
a completed one-shot owner or inventing a second timing/cache convention.

The original campaign remains intact. The next compiler/training acceptance work
must retain the held real-weight oracle and shared typed realization contracts;
these successful scientific state boundaries neither close that gate nor establish
vendor-device performance. Local sibling dependencies are still old (`pretrained-rstr`
0.2.545, `finetune-rstr` 0.2.287), and the latter still calls retired
`bind-program!`; committed isolated acceptance sources, local checkout state and
published consumer migration must remain separate evidence.

### October 9 verified nonlinear boundary checkpoint

The shared elementary-math realization contract covers 23 C-family library operations while
preserving source precision. A captured real-model layer-0 diagnostic now verifies its taps
against the unchanged JVM pullback bit-for-bit. Supplying identical tanh values makes CPU/GPU
pullback arithmetic identical; target Float tanh differences reproduce strong derivative
cancellation sensitivity on both systems. This is local boundary evidence, not closure of the
held real-weight gate. See `local-compiler-evidence.md` for controls and the three-coordinate
distinction from the original GPU kernel. All eight campaign items retain their original
workload acceptance obligations; continue the general compiler, training and distributed
verticals without replacing them with isolated numerical regressions.

### October 9 typed arithmetic consolidation checkpoint

#1098–#1100 are merged: C-family scalar product boundaries are protected separately
from explicit FMA, transposed-left products use the shared register body, and JVM
helper extraction retains typed calls and their operand conversions. #1101 extends
that preservation to scalar simplification and partial evaluation; its follow-up
checks retained operation precision during JVM scheduled SegMap admission with an
active species. An executed mixed-width
surface map exposed different rounding in vector lanes and the scalar tail; the
correct scalar schedule is retained until mixed-species vector lowering exists.
These are shared semantic fixes, not replacement model kernels or a new type registry.
The follow-up threads active-species precision obligations into stencil and C-SIMD
consumers and checks complete reduction recurrences. C integer-widening syntax
admission remains emitter-specific; canonical conversion precision is checked before
source projection. A further reproduced Boolean comparison miscompile requires
retaining selected interface parameter tags through SIMD canonicalization. Declared
comparison domains are checked before compare-and-blend; missing signatures decline
rather than being inferred. Untyped compatibility and missing operation stamps still
need separate evidence, and mixed-species vector execution remains unimplemented.

The C-SIMD guard also exposes a quant-fold performance residual: Q4 x8 explicitly
computes Double scale products into Float storage, while its old vector assertion
required an all-Float implementation. Bit-sensitive changed-input JVM/native tests
now retain the source precision. The integer-dot override and homogeneous Float
widening tests remain; mixed-precision load/compute/store vector schedules are still
needed to recover the fold optimization without narrowing arithmetic.

The fresh unchanged real-weight two-layer training gate after #1100 passes loss and
input-gradient checks but still fails 12 of 28 adapter-gradient checks.
After the subsequent retained-precision fixes, a cold unchanged run fails eleven
of 28 adapter checks; a public strict-FP32 matrix-policy diagnostic produces the
identical metrics. This rules out that policy switch as a remedy for this case.
Acceptance therefore remains open; neither these narrow regressions nor synthetic
training acceptance closes campaign item 5. Continue precision-boundary consolidation
and identical-input matrix/VJP localization without loosening the pinned oracle.
The October 9 diagnostic below preserves that gate: an independent CPU model
using the default GPU's decomposed Float dot order reproduces the same eleven
failing adapter keys and the GPU loss exactly, but four adapter comparisons
still exceed the original coordinate threshold against that counterfactual.
Whole-tensor relative L2 errors against the unchanged native reference are much
smaller (at most 2.45e-5); this is evidence of sensitivity, not acceptance or
permission to relax the original oracle. See `local-compiler-evidence.md`.
CUDA/HIP runtime performance and multi-host execution likewise remain separate from
hardware-free vendor compilation and local logical-worker acceptance.

### October 8 collective execution and precision-boundary follow-up

The storage boundary fix is merged as #1095 after all seven required CI gates passed.
The next isolated checkpoint diagnostic uses identical layer-0 Q/K/V and a synthetic cotangent,
not a full-model replay. Raw causal scores and DW dots match the JVM exactly on local OpenCL;
the D reduction matches exactly for both CPU and GPU supplied intermediates. Same-input softmax
weights differ by at most one Float ULP. Selectively spelling exponential evaluation as
`float(Math.exp(double(delta)))` restores exact agreement for this sixteen-weight diagnostic
without changing Float max/sum carries, storage or the reference. This does not close the held
adapter-gradient gate or prove universal libm bitwise equality.

Native `:exp` realization is now stated separately from evaluation-order evidence in KernelBody.
Its target-library accuracy is implementation-defined, not a correctly-rounded or bounded-error
claim. Float calls remain Float; explicit Double calls followed by Float conversion retain that
typed sequence. Other transcendental realization contracts and a user-selected reproducible
algorithm remain follow-ups; do not globally widen math or introduce an attention-specific emitter.

The contribution refinement, ordinary generated-combine bindings and complete realization
certificate are merged (#1090–1094), with all required CI gates green. Two and three logical
workers execute complete all-reduces on the local OpenCL and Level Zero devices. Fractional
products and FP32/FP64 cancellation are checked against independently rounded evaluation trees;
optional FP64 skips retain all FP32 cases. This closes the local numerical collective seam,
not multi-host transport, distributed transformer training or performance acceptance.

Training diagnosis exposed a separate declared-storage inconsistency: the preservation option
reached parameter typing but was dropped before the TypedSOAC frontend, and resolved local
polymorphic allocations still followed ambient precision. Retaining their array tags also exposes
two existing restrictions: fold-map results assumed a single ambient storage dtype, and a canonical
scalar Fold required its consumer precision to equal its carry precision. The follow-up forwards
one existing policy, derives each destination's storage dtype and converts only completed Fold
results through the shared scalar conversion authority. Default allocation policy is unchanged;
the explicitly requested preservation option now applies to resolved locals as well as parameters.
Host fold-map projection spells the same per-destination completed-result conversions explicitly,
including mixed Float/Double outputs. Its executed oracle compares the projected form with typed
source execution; the shared mixed-storage fixture also enters CUDA/HIP compile gates. No fold
carry is narrowed merely because its consumer uses narrower storage.

An isolated real-checkpoint attention probe executes Double computation with Float storage after
these changes, with small nonzero differences from the existing CPU reference. The global `:dtype`
still selects generic source overloads; this probe fixes the Float source overload explicitly.
It is not a new independent precision-selection API and does not close the held real-model gate.
The next acceptance work must distinguish source specialization, intermediate arithmetic and
storage, without rewriting the pinned model or loosening its original gradient checks.

### October 8 native reference-environment admission

`blas/library-info` and `lapack/library-info` now report retained selection facts,
with no numerical smoke test or thread reconfiguration. Discovery retains its loader
input path and optional OpenBLAS configuration once; public diagnostics omit the foreign
lookup. Fortran LAPACK and LAPACKE report separately, including absent/rejected/error
outcomes, rather than inferring QR capability from a Fortran smoke test. Selected means
selected, not functional or fast. The requested loader path is not a binary hash, and
dependency-exported OpenBLAS metadata only advertises that provider's ABI, not a wrapper's
certified interface. MKL reports the LP64 interface filename as declared, not as an
OpenBLAS build fact. No second compiler cache or numerical descriptor registry is added.

Local retained evidence is threaded MKL for BLAS and OpenBLAS 0.3.30 (including
`NO_LAPACKE`) for Fortran LAPACK; the separately loaded LAPACKE wrapper resolves the
same dependency metadata. This differs from the external OpenBLAS 0.3.32 environment
and is not a preferred-version claim or crash diagnosis. Integer overflow probes confirm
existing core `int`, LAPACK `int-seg` and BLAS GEMM reject `INT_MAX + 1`; a hardware-free
GEMM-handle regression preserves that contract instead of adding a redundant conversion layer.

Explicit selection now shares conventional paths and one resolver across BLAS/LAPACK.
`raster.openblas.path` is read at first delayed selection rather than namespace load;
the captured pin bypasses preloaded/MKL/default paths. A missing file, required symbol,
or OpenBLAS build metadata rejects the pin instead of silently changing providers.
This intentionally tightens LAPACK's former preferred-path semantics and makes BLAS
honor the same setting. Pinned QR requires LAPACKE symbols in that selected library;
it does not silently bind a separate LAPACKE library with unknown provider linkage.
Without a pin, existing discovery order and metadata-absent compatibility remain.
The pin is not a guarantee against dynamic-loader symbol interposition. Provider/status
diagnostics and argument-range checks remain separate native admission work.

An external CUDA developer reports 4,602 tests / 57,807 assertions at `f7cbf12`,
with 16 failures and 45 errors. Their initial categories are missing BLAS/LAPACK
(37 checks), exact floating-point differences (10), unsupported portable trapping
arithmetic (11), attention dispatch (two), and a missing Clang case subsequently
resolved. These are reported baseline evidence, not reproduced diagnoses or a count
of independent root causes. A standalone C batched-GEMM crash with OpenBLAS 0.3.32
still requires the arguments, headers and linked-library integer ABI to be checked.
CUDA compiler fixtures and two changed-input device checks with clean memory-sanitizer
results are preliminary execution evidence, not completed runtime or performance gates.

Both Panama numerical libraries declare 32-bit integer slots. A shared optional
`openblas_get_config` probe now rejects advertised `USE64BITINT` libraries before
BLAS or LAPACK numerical downcalls. The rejection is intentional: changing integer
width requires different descriptors, not a tolerance adjustment. Unknown providers
without metadata retain existing admission and are not certified LP64 by this check.
Hardware-free native stubs test the pointer-returning metadata ABI, absent symbols,
null metadata and incompatible build flags. This does not explain the reported C crash.
Admission errors propagate through availability queries rather than being converted into
whole-namespace skips. Ordinary missing-library discovery retains the existing optional behavior.
The CI test job declares `RASTER_EXPECT_NATIVE_LIBRARIES=1`; an unconditional test
namespace then requires functional BLAS and LAPACK, without forcing either onto laptop
or compiler-only jobs. This preflight does not replace per-capability coverage accounting.

Actual binary identity/version and live thread metadata,
ABI treatment for unknown/preloaded providers/wrappers, and a pinned native reference environment
remain CI/compatibility debt. The shared pin resolves the former BLAS/LAPACK property
discrepancy; it does not provide a complete reproducibility manifest. Keep the remaining work
on items 1–2 without replacing the external training, distributed or PDE gates.

### October 8 resumed acceptance and AD review

The bounded JVM Float-seed correction (#1070) and position-sensitive helper capture (#1071)
are squash-merged after all seven registered CI gates passed. The unchanged real-checkpoint
gate after #1070 still declined: loss and input-gradient checks passed, but 12 of 28 adapter
checks exceeded the original componentwise threshold. Same-input raw gradients and a secondary
sequential-FP32 matrix diagnostic isolated an arithmetic-realization contribution without closing
the residual or replacing the pinned CPU oracle. The external training PR remains held.

Upstream #1072–1077 subsequently changed allocation defaults, buffer ownership, additive
reduction AD and gathered-read transposes. Recheck current main before carrying forward the old
numerical diagnosis. A code review flags two distinct replay obligations for focused reproduction:
additive replay must satisfy the existing purity proof rather than repeat effectful preludes;
storage replay/overwrite checks must follow aliases and include gathered index dependencies.
Symbolic aliases alone cannot prove distinct caller-provided arrays physically disjoint.
These correctness checks belong to the existing consolidation campaign, not a new compiler or
another AD rule registry. The distributed, multilevel/PDE and broader workload gates remain open.

The current-main checkpoint recheck reproduces the same failed numerical gate, without changing
the source pins or tolerances. Independent deterministic probes confirm the replay concerns:
an alias write changes a scalar gradient from 3 to 102, and an effectful additive prelude executes
twice (gradient 12 and scratch 6 instead of 3 and 3). Additive replay now reuses the ordered
route's existing purity proof; impure steps retain their closure tape and execute effects once.
This preserves support for effectful reductions rather than rejecting their additive shape.
Storage-alias and gathered-index replay protection remain the next distinct slice.

That slice derives direct storage representatives from ordinary aliases and each operation
record's result/output identity. The shared forward pass checks writes against those representatives;
gather, mapped and separately analyzed dotimes transposes declare their reread index dependencies.
Ordered closure reductions also declare scatter-index dependencies even though their primal read
values are taped. Explicit source writes inside the same operation are checked against its own
replay claims; normal record-declared output writes are not mistaken for such source effects.
Captured dotimes index values remain valid after source-index mutation, rather than inheriting
another representation's replay restriction. Focused source/AD oracles cover transitive inactive
aliases, map-result/output aliases, index mutation, captured indices and fresh-buffer writes.
This is not a physical no-alias proof for distinct caller arguments or a general may-alias
analysis of arbitrary conditional/opaque returned arrays. Keep those ownership obligations
explicit in the remaining consolidation work; no numerical gate or residual representation changes.

### October 6 contextual initializer precision follow-up

Real-weight boundary isolation exposed a remaining JVM/GPU arithmetic discrepancy in attention
scores: the GPU uses sequential FP32 products and sums, while the JVM's large-method helper
extraction retained Float products but accidentally accumulated in Double. Small unsplit methods
hid this defect because their typed Float calls rounded the accumulator on every iteration.
A four-map cancellation fixture reproduces the split-path failure independently of checkpoint
data: `[1e8, 1, -1e8]` incorrectly sums to one rather than zero.

The JVM loop emitter now realizes the retained contextual Float seed before forming its slot
type, only for a bare Double literal with both a Float binder stamp and Float recurrence width.
This adds no inference registry or attention rule. Explicit Double/nonliteral initializers and
genuinely wider recurrences remain unchanged. Canonical AD/SOAC forms are unchanged: an initial
attempt to insert casts in shared walking broke Double-storage AD islands and GQA proof
reconstruction in cold CI, so that broader change was withdrawn. Split/unsplit JVM tests, an
actual causal-weight cancellation oracle, and public OpenCL/Level Zero replay check the narrower
boundary; the affected AD and GQA cases are explicit non-regressions. Fresh checkpoint attention
weights match the independent sequential FP32 reference; full real-weight gradient acceptance
must be rerun and is not claimed by this correction. The original eight-item campaign stays open.

The split regression also exposed a separate lexical dependency defect: naming its length
parameter `count` can cause an extracted helper to read `clojure.core/count` instead. Keep that
follow-up distinct from initializer precision and cover core-shadowing parameters at helper
extraction; renaming a fixture to `cnt` isolates this slice, not a language restriction or fix.

The helper-capture follow-up uses the existing shared lexical free-symbol analysis with visible
parent locals declared, rather than allowing global core resolution to erase parameters. It also
prevents a numeric global constant from replacing a same-named local parameter. Visibility is
position-sensitive: each extracted `let` initializer sees only parameters and prior binders;
the body sees all completed bindings. True globals remain eligible for the existing constant
projection, and nested binders stay owned by the canonical scope analysis. Core-name/constant
shadowing, nested shadowing and early-global/later-local regressions cover this JVM boundary.
This does not introduce a new scanner, change surface signatures or close arbitrary-loop AD.

Current scalar AD consolidation: final parameter projection follows retained SSA alias bindings
to carried source types, keeping unknown/dynamically absent cotangents nil-safe. This removes the
cold projection-helper typedness gap without adding an intrinsic or adjoint registry. The
placement follow-up lets a direct Float/Double result conversion cross pure SSA equations and
replace only a same-dtype manifest identity transform. Effects, aliases, opaque host boundaries,
nonidentity transforms and general mixed-width arithmetic retain their existing barriers. The
mixed scalar-energy gradient now executes with exact JVM parity on both local backends, for
empty, small and two-phase reductions. This admitted reduction epilogue is not arbitrary scalar
consumer placement. Continue with external training acceptance and generic composition; do not
treat this slice as completion of the eight-item campaign.

Authorized scope is the eight-item campaign below. The original four-stage emitter sequence and
subsequent chronological notes remain the detailed history, not a reduction of that scope. Each
production migration must retain numerical, ABI, ownership and resource contracts; isolated emitter
coverage is not a completed vertical. The north star remains the architectural specification.

The current local-first landing order and completion gates are in
[compiler-consolidation.md](compiler-consolidation.md). The checkpoint below and chronological
notes retain the wider campaign and its implementation history.

## Current eight-item checkpoint — 2026-10-02

### October 4 public mixed-matrix admission slice

The public schedule and uniform Int/Long graph-carrier slices are now in review. The
hardware-free Gemma training compile is the next integration gate, before migrating its existing
mixed-precision device trajectory off the descriptor fixture. It exposed a reconstruction-context
gap: declared shape operands named like core functions (for example `seq`) disappeared from
standalone scalar-dependency analysis. The pure semantic constructor now derives its lexical
context from retained TypedSOAC values, just as production lowering does. Core-shadowing shape
regressions and the wider TypedSOAC route suite pass; exact source-law comparison is unchanged.

The same Gemma compile subsequently exposes an unresolved storage relation: the matrix axis
uses a retained scalar alias (`nqh`), while its output capacity expands to `nq * hd`. Do not assert
these equal from their spelling or relax capacity validation. The follow-up derives the dense
output-prefix minimum from the canonical reduction axes and enforces a checked allocation-capacity
guard at the independently reconstructed graph boundary. This proves storage safety without
asserting that independently supplied scalar values are equal. Unsupported physical layouts and
noncanonical folds remain outside this projection; declared capacities are not replaced.
Public mixed training acceptance, native numerical trajectories and performance remain open.

With the output-prefix guard, the hardware-free Gemma training compilation succeeds: 189
scheduled equations, 123 emitted tensor equations, 66 host scalar equations and 50 contraction
dispatches, with no candidate declines or legacy fallback. The other 73 emitted equations are
not single plain FP32 contractions; this is not evidence that every operation is matrix-tiled.
This measurement uses the synthetic Intel descriptor and establishes compilation only, not
native dispatch selection, training parity or throughput.

### October 5 invocation-local dispatch proof consolidation

The per-invocation boundary scan completes pure Gemma planning in 157 seconds after a
182-second fresh compile under variable background load: 123 tensor calls, all 50 mixed
`xmx-direct` selections (21 NT, 15 NN, 14 TN), and zero driver allocations. An inclusive
profile on the same compilation takes 136 seconds and records 905 boundary-equation queries
accounting for 68 seconds; nested timings are not additive and these are not matched benchmarks.

The next bounded consolidation extends the existing sealed program evidence, not the cache or
certificate hierarchy. One read-only operation-identity map retains the independently checked
fallback equation, common physical results and certified candidates. Every dispatch candidate,
numerical permission/model, common storage and exact fallback is checked during that proof.
Storage lowering and synchronous call construction reuse only those exact immutable facts.
Absent external evidence, lowering derives fresh invocation-local evidence itself. Copied or
modified owners/evidence reject; a missing scoped operation projection fails closed. Public
validation remains independent; runtime admission, scalar/range checks, result views, aliases,
host callbacks and final LinkPlan obligations remain fresh. No proof is attached to the plan,
and callbacks do not inherit the projection scope. Native Gemma trajectory acceptance and
the original external training/distributed/multilevel campaign gates remain open.

The first post-consolidation large check compiles in 118 seconds and lowers in 51 seconds,
again retaining 123 tensor calls and all 50 intended mixed selections without driver allocation.
An inclusive repeat takes 42 seconds: boundary queries fall from 905 to 290, KernelBody
validations from 70,122 to 20,883, and call construction from 32 seconds to 2 seconds. Timing
conditions are not controlled; call counts, unlike wall-time ratios, directly identify removed
duplicate work. Remaining boundary queries account for 18 seconds and justify auditing the
existing complete-write consumers next, before adding any further retained facts or changing
native/default schedules. No native trajectory or end-to-end training speedup is established.

The symbolic-write follow-up retains only already proved domains in that same operation report,
not a resolved complete-write flag. Invocation initialization and final LinkPlan effect analysis
resolve current scalar extents, capacities and views afresh. Ordinary SOAC/contractions keep
their prior semantic, alias-aware coverage path even when a report contains a stronger certified
contraction extent; sharing evidence does not promote another initialization-coverage family.
Protected SWR and dispatch admission retain their existing rules. Public LinkPlan validation
reconstructs independently, while retained validation checks the exact program seal before each
operation projection. Source-order, partial/prefix writes and structured loops are unchanged.
Fresh-versus-retained effect witnesses, varied capacities/extents, alias rejection and callback
isolation remain the oracles. Equal copied operations can be independently checked as equations,
but cannot substitute for a whole program's exact owned step; whole-call constructors should use
the owning program rather than manually replacing a step. No new cache or certificate is added.
Reusing the existing Gemma compilation, the large public plan lowers in 14.1 seconds and retains
123 tensor calls, all 50 mixed selections and zero driver allocations. This is a local diagnostic
observation under changing background/JIT conditions, not native or controlled performance data.

The existing mixed Gemma trajectory fixture is being migrated to public lowering and replay,
with donated adapters, shared constant weights and fixed-graph binding evidence. All 50 eligible
dispatch equations must select mixed execution; filtering only already-mixed selections would
hide fallback regressions. The initial pure public lowering exceeded a 590-second local check
budget; stack samples showed repeated semantic fingerprint/source-ABI validation and equation
reconstruction under `invocation-link/lower`'s buffer-by-equation boundary scans. After consolidating
those scans within one invocation, a fresh capped REPL compiled in 182 seconds and lowered in
157 seconds under varying background load. The resulting plan has 123 tensor calls and selects
all 50 mixed `:xmx-direct` dispatches (21 NT, 15 NN, 14 TN), with zero driver allocations.
This establishes public planning and selection on the synthetic Intel descriptor, not a matched
performance comparison or native training acceptance. Lowering remains too expensive; profile
the remaining repeated validation before extending the existing exact-owner evidence. Runtime
admission, independent public validation and native 25-step numerical acceptance remain required.

The subsequent native recheck passes the tiny public register-tiled replay on OpenCL and
Level Zero (six assertions), followed by a small Level Zero `:xmx-direct` matrix case with
relative L1 error `1.5e-8`. The migrated public mixed Gemma fixture then passes its complete
25-step trajectory: 35 assertions, zero failures/errors, all 50 mixed dispatch selections
(21 NT, 15 NN, 14 TN), zero planning allocations, and planned-versus-bound graph agreement.
FP32 loss decreases from 2.281172 to 1.261994; mixed loss ends at 1.262064. Every step remains
within the existing `5e-3` relative trajectory tolerance. This supersedes the unresolved local
native acceptance for this small fixture, not the historical queue errors or other workloads.
The capped reusable REPL completes the check in 248 seconds including compilation, planning,
binding, replay and host loss evaluation; this aggregate is not kernel timing or a performance
comparison. Full-model training and matched performance acceptance remain open. The fixture now
uses public equation-first lowering/replay rather than the resident descriptor lifecycle.

The next equation-first candidate constructor derives a full-K mixed schedule from the retained
typed algorithm and freshly reconstructed public graph. A pure planner owns target admission;
the existing matrix emitter consumes the resulting stage bodies. The shared target-schedule
owner checks descriptor/instruction agreement, divisible positive geometry and normalized
workgroup limits. The first row admits Intel DPAS with Intel OpenCL only, Int dimensions,
materialized dense storage or NN/NT tile-input fusion, and fused leading-batch slices. Unsupported
batched materialization, targets, layouts and precision policies return explicit declines.

This constructor is not numerical permission or public runtime selection. Every emitted artifact
still passes reconstruction, ordered ABI and complete-write checks, and physical scalar guards
remain enforced by graph preflight before allocation. Explicit public dispatch consent, retained
selection evidence and the Gemma training trajectory remain the next gates. Automatic defaults,
split-K coverage, external training and distributed/multilevel acceptance are not completed here.

The public opt-in is `:schedule {:precision :mixed-f16-f32 :typed-contraction
{:strategy :dispatch-mixed-matrix}}`. Both keys must be explicitly requested; inherited mixed
precision defaults do not grant consent. The existing equation dispatch owns one reconstructed
mixed candidate and an exact portable fallback, shares measured-selector consumption, and keeps
the full operational model in its numerical/tuning identity. Unsupported target/layout admission
retains the exact path with explicit decline statistics. Unconsumed selectors, fallback pruning,
tile-space expansion and split-K requests fail rather than being silently ignored. Automatic
defaults and vendor matrix schedules remain unchanged. This is public compiler/lowering and
artifact-roundtrip acceptance; native trajectory/performance acceptance is still pending.

Uniform Long shapes reuse the existing checked per-stage matrix specialization. The public
graph's integral carrier follows its declared GraphScalar width rather than the first consuming
node's physical width; exact and mixed alternatives therefore share one stable public ABI.
Individual kernel parameters and their checked int/long conversions are unchanged. Direct
physical range conditions are projected once by the existing owner and checked before node
conversion/scratch sizing, making a valid public Long outside an int specialization an explicit
schedule-admission failure. Invalid public representations remain caller errors; computed
private expressions still require their own checked preflight. Mixed-width dimensions remain
declined. This intentionally changes serialized declared-integral graph ABI; compiler/build
identity must invalidate older artifacts, and no compatibility wrapper is introduced.

### October 3 runtime/state integration update

This update supersedes the producer-integration gaps below only for the specifically validated
plain dense fields; it does not change the other seven campaign items or claim campaign completion.
PR #1027 connects original completed receipts and exact owner-bound storage measurements to the
existing numerical manifest/content-provider lifecycle. Selected output/post-state fields stream
through bounded aligned downloads under a private output lease. Physical producer, schedule,
representation and field/content bindings are derived; semantic coordinates/policy remain declared.
Independent review fixed a cross-thread provider deadlock by restricting the lifetime monitor to
validation/plan/lease acquisition. Affected content/capture/publication/skip tests pass 46 tests /
962 assertions, including actual OpenCL and Level Zero generated heat capture and public-codec
restored continuation. Exact-HEAD review and all seven CI gates preceded squash merge.

PR #1028 migrates the existing full-domain coarse/fine heat restart oracle away from handwritten
producer/byte-order metadata. Three witnessed replays capture both fields, verify availability,
record numerical-state parents and separately check execution-replay parents. The independent
JVM evolution, uninterrupted six-step device baseline, mapped-file restore and conservation checks
remain. Affected multilevel/state/capture tests pass 27 tests / 257 assertions; exact rebased-HEAD
review and all seven gates preceded squash merge. Both fixtures explicitly use synthetic packaged
build evidence and test providers/metadata, not release authentication or a production store.

#### Item-8 integration in review: captured partial-patch restore

The current strict `verify-restore!` compares complete producer provenance. Completed capture
retains dynamic execution fingerprints, schedules and representation observations; predicting
those from a target would either become circular or confuse producer and consumer programs.
Keep the existing strict four-facet API unchanged. Do not copy incoming provenance into expected
contracts, introduce arbitrary subset matching, or substitute the continuation artifact for its
source producer.

Use a fixed composed boundary instead: independently specified complete field/geometry, phase
and numerical-policy semantics, plus an expectation derived from the original sealed source
Prepared and ordered semantic output/post-state keys. Resolve field/node bindings through the
retained source interface/effect evidence; share the existing frontier calculation rather than
adding a second role/donation heuristic. Check exact program identity and ordered source/node
bindings, and require each captured field-producer content reference to match its full-field chunk.
Reject ambiguous public aliases because v1 capture retains nodes, not public keys. Check the
selected dtype's measured representation kind, dtype, source program and normalized byte order
against its stored chunk; these are internal consistency checks, not predictions of target facts.
Keep dynamic replay and numerical-state parent evidence distinct and auditable. Matching these
contracts remains consistency/compatibility, not authentication of an untrusted serialized claim.

The branch migrates the existing partial-patch remap/restart fixture while preserving its independent
layout/active-cell ordering, phase, wrong-same-extent rejection, host remap/evolution, mass and
same-backend exact continuation oracles. Add negative source-artifact, field-binding and content
counterexamples before any lease/upload. This slice remains unmerged pending final independent
review and all seven exact-head CI gates. Device packing, scalable hierarchy validation,
subcycling/reflux, external training migration and real fabric/device measurements remain open.

#### Resident patch packing acceptance

The next slice exercises production hierarchy-derived local/packed indices through the existing
typed gather/scatter source operations. A public compiled program packs a partially covered
coarse patch and a fine patch into active-cell order, then unpacks into caller-owned patch fields.
Independent host indexing and JVM execution agree bit-for-bit with OpenCL and Level Zero,
including signed zero and preservation of unselected covered coarse cells. The affected geometry,
packing and CI selection/skip ratchets pass 14 tests / 1,162 assertions in the capped REPL.
This adds no AMR-specific kernel, owner, codec or session convention. It is bounded resident
packing acceptance, not scalable hierarchy performance, general patch checkpoint materialization,
temporal interpolation, subcycling or reflux. Independent review and CI remain required.

#### Bounded subcycled diffusion and synchronized restart — October 4

PR #1031 is merged after independent review and all seven exact-head CI gates. It adds
temporal boundary interpolation and signed transport accumulation as ordinary source operations,
and retains physical capacities independently of checked traversal minima. Fusion still proves
the consumed output AxisMap; capacity guards do not substitute for source/index proofs.

The next bounded numerical vertical composes those operations with existing finite-volume,
CSR, gather/scatter and restriction operators: a complete coarse prediction, two ratio-2 fine
substeps, signed reflux, then average-down. Pure projection admits one strictly interior aligned
2D fine patch and periodic boundaries, checks enumeration budgets before materialization, and
retains physical face-integrated rate units. Projection does not certify timestep stability,
runtime ownership or a general adaptive cycle; synchronized inputs and a stable timestep remain
explicit caller obligations. No AMR-specific emitter, cache, allocation or session is added.

Independent coordinate-loop oracles cover square and anisotropic rectangular geometry,
conservation, transport signs, coarse-boundary interpolation and equilibrium scratch replay.
Completed capture stores both synchronized fields using original producer and owner-bound
representation evidence. The source closes before mapped-byte restoration into a fresh execution;
continuation matches uninterrupted resident feedback bit-for-bit on both OpenCL and Level Zero.
A different declared phase is rejected. The affected suites pass 12 tests / 316 assertions locally,
with both native backends available. Synthetic build/provider fixtures are not release provenance
or a production store. Independent review and CI remain required before merge. General hierarchy
scaling, adaptive/multi-patch or mid-cycle restart, external accuracy/performance comparisons,
external training migration and real distributed fabrics remain open.

#### Public mixed-precision training integration — October 4

The existing mixed Gemma training oracle still uses the compatibility descriptor fixture and
manually observes candidate selection. Do not replace it with a weaker public test: a reproduced
public `compiled/lower` at its `CFG-MP` shape with `:precision :mixed-f16-f32` binds portable
contractions, because `:auto` deliberately remains portable. An explicit `:matrix` request
correctly declines the FP32 storage with `:dtype-not-dpas`; that strategy is currently the direct
half-storage instruction request, not the packing-plus-matrix schedule. Policy permission does
not establish which instructions were emitted. This is a real integration gap, not an AD bug or
evidence that mixed training already executes through the equation-first path.

The reporting prerequisite adds ordered `:kernels` labels to the common executable description,
which public execution reports already forward. Each leaf retains only its own artifact labels;
missing precision remains nil rather than inheriting graph/request policy. Graph-level mixed
precision and a leaf's numerical contract are different facts. Labels describe bound artifacts,
not replay events, timing, or new compiler certification authority.

Continue in this order, reusing the existing typed mixed-matrix builder and graph refinement:

The reconstruction prerequisite is a target-neutral `mixed-matrix-schedule` pass. The existing
materialized, tile-input-fused, dynamic-LHS, split-K and batched NN/NT stage construction moves
there rather than being copied into another emitter. Target emission consumes the planned graph;
hardware admission and runtime selectors remain outside the pass. This extraction does not yet
prove a refined graph's complete-write or bounded-error law, admit it to equation dispatch, or
migrate the mixed Gemma gate. The next certification step must independently reconstruct from the
retained equation/source boundary and verify the terminal writer, dependencies and emitted bodies.

The reconstruction prerequisite now retains a closed versioned physical recipe, including input
fusion and batching decisions. Reference planning derives semantic operands, dimensions, layouts,
epilogues and the public boundary from the retained typed algorithm and an independently supplied
source graph. Split-K compiler-owned accumulator and flat work-item identities are capture-checked
by their constructors, allowing exact complete-graph comparison without printed-form equality or
a permissive alpha-equivalence check. This remains reference reconstruction, not certification of
the candidate graph, emitted stage bodies or its complete-write/numerical law. Those checks must
land before admitting public mixed-matrix equation dispatch.

The following extraction moves existing matrix-body and scheduled-certificate construction into
`mixed-matrix-body`, shared by stage emission and independent reference projection. Split-K and
batched views, dimension parameter allocation, epilogues and fused input regions retain the same
constructor path. Target emitters retain parameter spelling and artifact emission. This is still
not whole-graph certification. The following slice moves layout and canonical split-K combine
certificates through the same pass and adds reference projection for every stage node. Production
emission consumes these shared constructors; tests compare complete scheduled certificates, not
only emitted labels or ABI shapes. Before using this projection as algorithm validation, require
exact independent typed stage-graph reconstruction, including canonical range and slice geometry.
Terminal complete-write checks and graph-derived composed numerical policy remain required before
public mixed-matrix admission.

Local native validation is currently unresolved: the matrix suite reports queue/upload failures,
and an unchanged pre-extraction register-tiled baseline reproduces Level Zero queue error
`0x7ffffffe` and OpenCL upload error `-5`. Hardware-free checks and CI are separate evidence;
do not claim local device acceptance or mixed training progress from the extraction alone.

The source-law prerequisite shares the existing TypedSOAC segmented-reduction constructor between
ordinary lowering and exact semantic rederivation. The stronger contraction query compares the
reduction state, neutral values, recurrence, algebra, axes, scopes and retained semantic metadata;
only grid and schedule remain separately checked physical policy. It does not accept the
refinement's own source as independent evidence. Admission is deliberately restricted to canonical
lambda-local-free contractions; retaining arbitrary ordered lambda locals in the general reduction
region remains language-coverage work. This query is not yet wired into mixed dispatch admission:
fresh semantic graph derivation, stage reconstruction, body/projection checks and complete-write /
composed-numerics checks must be joined before that migration.

Mixed-matrix reconstruction validation now joins the exact typed reduction-law query with
complete stage-graph reconstruction and ordered stage-body projection. Semantic fingerprints
retain type metadata that ordinary collection equality ignores. A matching candidate/source pair
with a changed reduction neutral is rejected against the retained algorithm. This is deliberately
not complete-write or composed-error certification, and is not wired into public admission yet.
The remaining boundary integration must preserve the independently rederived public graph's
capacity expressions, scalar order, preconditions and optional inferred ABI, rather than impose
the compatibility descriptor's logical operand extents and explicit argument order.

The public graph's missing ordinary-fold read minima are addressed through the existing verified
operand AxisMaps, not a GEMM-specific recognizer or a second recurrence decomposition. Core reads
retain every distinct map, including repeated accesses to one buffer; unsupported shifted/gather
reads decline this optional query. Undeclared-map search is bounded. The graph owner's existing
contract remains: unresolved storage gets a proved minimum checked at binding, while known storage
capacities are preserved and guarded when insufficient. These logical minima do not authorize
physical address projection, rearrange the callable ABI, prove epilogue/neutral reads or admit a
mixed schedule. The mixed planner still needs to preserve this independently derived boundary.

The mixed planner now preserves an independently supplied graph boundary verbatim, including
inferred ABI, scalar order and capacity guards. Public input/output identities, storage roles and
dtypes must match its locally derived matrix interface; every logical read/write extent must be
covered by a capacity contract or explicit guard. Private layout/conversion/split buffers retain
their logical stage extents. Extra public scalars needed solely by capacity guards are preserved.
The compatibility/standalone interface construction remains a fallback, not authority for a
public equation graph. Reconstruction tests use a fresh public equation graph and reject matching
source/candidate undersized or wrong-dtype boundaries. This still does not admit public dispatch:
terminal complete-write and composed numerical policy are next.

EmittedParallelEquation now reconstructs mixed refinements from its independently derived public
source graph and compares every artifact's scheduled-body certificate with the pure stage-body
projection. Matching boundary or stage source alone cannot authorize a changed generated body.
Artifact graphs with an inferred public ABI use the same ordered ABI finalizer as ordinary SegOp
emission; compatibility graphs retain their explicit interface. This is artifact reconstruction,
not complete-write coverage, numerical composition, target admission or performance evidence.

The existing target-neutral matrix topology analyzer now belongs to the scheduling passes, not
the GPU backend. All three target-emission consumers reuse that owner. This mechanical move
prepares complete-write validation to share the same fragment/store/mask analysis without a
proof-to-backend dependency; it introduces no new accepted schedules or numerical claims.

The first mixed complete-write projection is deliberately direct, nonbatched and full-K only.
It reuses the shared matrix topology analyzer, checks the exact two-dimensional group count,
and joins the generated terminal/unique writer to the authoritative plain FP32 result. Its
domain is the retained free-axis product; public buffer overprovisioning does not enlarge it.
Changed stores/masks/launches cannot borrow a matching stage source's proof. Split-K combination,
batched result views, composed numerical permission and public mixed dispatch remain separate
follow-ups. Static empty KernelBody storage still rejects under its existing shape contract.

Full-K leading-batch results now compose the same per-slice store partition over the verified
contiguous `[B,M,N]` parent/view contract and exact group-axis-2 launch. NN/NT and either shared
input reuse this proof; split-K remains a reduction-composition follow-up, not an independent
batch. The coverage fixtures also caught and corrected the shared transposed RHS's internal
logical shape: `K×N` plus the permutation describes physical `N×K`, rather than applying the
transpose twice. This changes no surface semantics. Numerical permission remains separate.

Mixed numerical permission now consumes an operational model derived from the independent
reconstruction and matched to every emitted body, including fused conversions. It records stage
storage/numerics, actual matrix instruction/tiling, traversal, batching and result transforms.
Dispatch requires both `:approximate-model` permission and an exact canonical match in the
ordered `:permitted-models` vector; the default remains the exact portable candidate. An attached
candidate claim cannot replace rederivation. This is explicit precision-change consent, not a
universal finite error bound: IEEE half overflow and target-defined instruction exceptional
values remain visible. Existing refinement metadata's historical `:bounded-error` name does not
assert a proven finite bound. Public schedule selection and target admission remain next.

1. Extend equation dispatch and complete-write validation to consume independently validated
   mixed graph refinements. Preserve the graph's explicit bounded-error contract; inspecting only
   the first packing kernel is not a numerical proof. Output ABI is not a complete-write proof.
2. Share the existing FP32-to-FP16 packing/tile-local fusion, matrix, epilogue and split-K candidate
   construction with equation-first emission, bound to the retained typed equation, source graph,
   external ABI/effects and target descriptor. Add no training-specific emitter.
3. Introduce an explicit mixed-matrix dispatch request alongside the exact portable default.
   Do not redefine direct half-storage `:matrix` or change `:auto` before measurements justify it.
4. Preserve actual selection/admission evidence at the equation binding boundary, separately from
   its later fixed graph binding. Do not claim runtime candidate attempts from a requested policy.
5. Migrate the existing 25-step Gemma trajectory test only when the public path retains its
   selected matrix graph and leaf evidence. Keep loss reduction and FP32-versus-mixed trajectory
   oracles intact. The existing mixed builder is initially Intel/DPAS gated; vendor matrix rows
   must reuse this seam without claiming device acceptance from compiler-only CI.

This interleaves with item 5 and the emitter consolidation; it does not replace the original
eight-item campaign, external real-weight training, benchmarks or distributed execution work.

This section supersedes historical "next" and "still required" statements below where the cited
implementation now exists. It records evidence, not a declaration that the campaign is complete.
The detailed local execution and preparation evidence is in
[local-compiler-evidence.md](local-compiler-evidence.md). The wider campaign remains open;
do not infer a published release from a merge, or device performance from source compilation.

| Campaign item | Current evidence | Remaining acceptance |
|---|---|---|
| 1. CI feedback and coverage | `.circleci/config.yml` uses sixteen deterministic test shards plus public CUDA/HIP compile and OpenCL CPU gates. Targeted laptop checks use one capped REPL. The OpenCL corpus gate now writes its complete same-compile report under `test-results/coverage`, which CircleCI already retains. | Continue measuring shard balance and skip reasons; preserve whole-suite coverage while reducing feedback latency. Green compile jobs alone do not permit merge. |
| 2. Compatibility-debt ledger | `test/raster/compiler/compatibility_ledger.edn` has executable signature checks; `test/resources/coverage/gpu-corpus.edn` retains every corpus row, including errors. The baseline was refreshed through the real CPU OpenCL entry on 2026-09-21. | Keep the committed portable baseline and full target-specific CI report distinct; distinguish public invocation coverage from compatibility-entry census results. Never delete failed rows to improve the count. |
| 3. Direct TypedSOAC and retirement | The portable baseline now contains 252 source functions: 239 flat TypedSOAC, 1 typed structured-control, 5 scalar value helpers and 7 explicit host-only rows. The ASR resident-capture change advances only its two independently recompiled rows: the allocating wrapper becomes explicitly host-only and the caller-owned routine emits two independent KernelBody maps. Its local host/device acceptance is described below; #924 merged after all seven CI gates passed, including the same-compile corpus ratchet. Generated matrix/input-fusion, reductions, counted stores, AD/SGD, ABM effects and numerical operators use retained typed programs and fail closed instead of recovering through the source-shaped OpenCL emitter. Disabled schedule-family policy is now distinct from a real lowering decline in coverage reports. City two-exit/multiple-recur probes have exact local parity; P20f retains an explicit unbounded unchecked-int induction decline. | Finish external ASR consumer migration; keep source-language support broader than this fixed corpus visible through external regression cases. Preserve numerical, effect and overflow semantics rather than forcing unsupported loops through a fallback. Source coverage is not competitive scheduling or vendor device performance; matched accelerator execution and roofline evidence remain required. |
| 4. CUDA/HIP public-source verticals | Public numerical, attention and matrix fixtures compile in hardware-free vendor gates; allocation-free LinkPlan lowering is tested separately from source emission. | Vendor compilation is not vendor device correctness or performance. Keep matched execution/benchmark campaigns outside the laptop/CI dependency chain. |
| 5. Reusable training and external models | Direct equation-first AD/SGD and small resident Gemma/LoRA tests check state progress, numerical trajectories and admitted routes. Isolated committed external sources have LoRA/QLoRA 30-step parity and tiny Gemma forward/VJP/update replays against the JVM on both local backends, with shared mutable adapters (#950). Public typed scalar objectives now compose through ordinary rank-zero distributed domains (#1124–1125). No sibling checkout is changed. | The unchanged real-weight gate remains held: eleven of 28 adapter checks fail the pinned componentwise threshold despite passing loss/input-gradient checks. Local arithmetic/realization diagnostics are not acceptance. External adapter migration, save/load replay, longer training and preparation-cost acceptance remain; do not revive retired resident-descriptor APIs. Ranged sharing and asynchronous lifetime proofs remain separate. |
| 6. Hardware-free distributed simulation | `distributed_plan_test.clj` covers topology, dependency/resource overlap, directed routes, capacity gates, collectives, halos and certificate drift. | Calibrate against real execution and broaden workload projections without presenting analytic costs as measured performance. |
| 7. Data-parallel/halo execution | Checked DAGs and co-located logical workers use shared physical budgets; unequal heat partitions execute resident halo copies and match the monolithic reference. Collective refinements bind actual public reverse AD, row-weighted FP32 all-reduce and resident SGD for unequal batches on both local backends (#1117–1118). #1126 extends this to three explicitly composed updates with one parameter allocation per worker, fresh per-step scratch and checked dependencies; independent rounded oracles check trajectories and replica equality. | Real multi-device/fabric execution, asynchronous overlap and sustained/dynamic trainer execution remain. The finite three-step DAG is not replay of stale startup evidence or a released trainer. Co-location validates ownership, arithmetic and scheduling, not network throughput. |
| 8. Durable numerical state and multilevel/AMR | Heat and changed-layout coarse/fine evolution resume from addressed mapped bytes on both local backends, matching uninterrupted chains bit-for-bit (#926/#988–990). Bounded plain-patch device packing exists (#1030). Generated ratio-2 diffusion subcycling/reflux and synchronized restart have independent numerical oracles (#1032). #1127–1131 prepare the committed resident cycle through public composition and bind retained temporal stages to its exact Prepared, hierarchy-only workload, whole fields and private register lifecycle. The device follow-up executes two fresh-context cycles through the ordinary distributed owner on both local backends. | Generalized packing and scalable multi-patch/adaptive temporal planning, completion-backed distributed durable publication/lineage, general-rank/hyperbolic providers and mid-cycle restart remain. Structural certificates and bounded numerical results are not an arbitrary PDE theorem, external AMR accuracy baseline or production store. |

### Item-7 training reconciliation (2026-10-09)

`distributed_training_device_test.clj` now joins real local loss gradients to the collective
refinement and SGD consumer through allocation-free public lowering and checked LinkPlans.
Local batch sizes `[1 3]` and `[1 2 4]` exercise unequal contribution weights; each gradient is
scaled by its local row count before the declared FP32 sum tree, and SGD scales by the total
row count. Independent coordinate loops reproduce the leaf rounding, binary tree rounding
and final update. Both local backends pass, and final parameter replicas agree bit-for-bit.
The affected fixture passed four tests / 70 assertions, including hardware-free compilation
with session, allocation and instantiation tripwires.

This is one numerical update with co-located logical workers and synchronous resident copies.
It does not establish multi-step parameter-state reuse, asynchronous overlap, fabric throughput
or real-model training acceptance. The runtime deliberately rejects replay of a completed
one-shot owner; future training steps need an explicit checked parameter-state transition,
not a reset that silently reuses stale initializers. The independent real-weight Gemma gradient
gate remains open, and its original model, provider, numerical policy and tolerance are unchanged.

Item 8 now also has a production 2D hierarchy projection in `raster.ode.amr-geometry`: rectangular,
anisotropic, multilevel active rows retain patch provenance and match geometric sides into faces
and CSR incidence. Native evolution/remap/restart fixtures consume it, retaining independent
geometry and numerical oracles (28 tests / 1303 assertions). Fresh array materialization trusts
constructor output and adds no certification or ownership authority. Production cell-average
remap now uses exact normalized geometry, an overlap sweep and the existing FP64 CSR operator;
different finest lattices are covered by independent rational-overlap oracles. The affected
transfer/native evolution/remap/restart suite passes 16 tests / 375 assertions. Device patch packing,
scalable hierarchy validation, durable publication/lineage and general-rank numerical providers
remain open; the campaign is not complete. The reconciliation below supersedes historical
blanket statements about missing producer identity and subcycling/reflux.

### Item-8 reconciliation against implementation (2026-10-08)

PR #1032 already supplies `raster.ode.amr-subcycle`: a two-level ratio-2 diffusion cycle
with coarse prediction, temporal ghost interpolation, two fine steps, signed coarse/fine
flux-register accumulation, conservative reflux and average-down. It uses existing typed
finite-volume, CSR, gather/scatter and multilevel operations, not a domain-specific emitter.
`amr_subcycle_test.clj` compares complete fields and registers with independent coordinate
loops, including rectangular anisotropic geometry and incorrect-sign/frozen-boundary controls.
`amr_subcycle_device_test.clj` contains OpenCL/Level Zero parity and synchronized restart
checks, including rejection of a mismatched restart phase. Test presence is not fresh device
execution evidence; report each backend's actual result and capability skip separately.

The admitted numerical scope is one strictly interior fine patch, two levels, ratio 2,
periodic boundaries and caller-supplied stable timestep/synchronized initial state. Projection,
materialization and input preparation certify neither timestep stability nor initial coarse/fine
synchronization; both remain caller obligations. Adaptation, multiple fine patches,
additional levels, general-rank/hyperbolic providers and mid-cycle restart remain open.

Producer identity is also no longer wholly absent: `resident_state.clj` captures completed
program/storage/content facts from execution evidence and independently derives source-port
bindings/program identity from the prepared executable supplied at restore;
`build_manifest.clj` supplies packaged compiler identity. An unmanifested source checkout
does not gain persistent-cache eligibility. These facts do not establish durable publication,
distributed lineage realization or mathematical correctness of an arbitrary numerical operator.

The actual planning gap is narrower but important: `AMRPlan` still has only prolongation and
restriction operations and hierarchy-only/transfer-cycle modes. It does not retain the cycle's
temporal interpolation, substep dependencies, register lifetime or reflux/average-down ordering.
Integrate the existing numerical cycle into the compositional planning contracts before
claiming end-to-end adaptive AMR scheduling. Do not implement another diffusion/reflux kernel
vertical to compensate for this missing outer representation.

Fresh focused host validation on `b853cc5a` plus this documentation reconciliation:
`amr-subcycle-test`, `amr-geometry-test` and `amr-transfer-test` pass 14 tests / 1233
assertions in the capped REPL. This does not refresh the device/restart measurements or
close the real-checkpoint training gate.

Fresh local refresh on `5e3a7af6` exercises the five AMR geometry, transfer, subcycle,
device-packing and device-subcycle namespaces: 20 tests / 1359 assertions pass without
capability skips, including OpenCL and Level Zero packing and synchronized restart.
The bounded plain-patch packing vertical already exists in #1030; historical statements
that *all* device patch packing is missing are superseded by that evidence. Generalized
packing, temporal outer-plan integration and adaptive/multi-patch acceptance remain open.

### Rank-zero distributed compute composition (2026-10-09)

Explicit local compute domains now admit shape `[]` under the same dense single-leaf,
dtype, equal-volume, containment and ownership checks as positive-rank tensors. It projects
exactly one physical element; an absent shape, extra element or rank-mismatched offset
does not gain admission. Structural regressions preserve the original ABI leaf and allocation.
A complete public prediction/MSE objective also runs through an ordinary one-worker
DistributedPlan on OpenCL and Level Zero and matches the JVM scalar loss. No scalar-loss
opcode, trainer API, special transport or runtime replay convention is introduced.
This closes local scalar-output composition, not multi-step/fabric training or the held
real-weight numerical gate.

### External training publication readiness (2026-10-08)

The local `../finetune-rstr` repository at `9e9ba5d` has no configured Git remote;
the expected GitHub repository is not resolvable by the current account and the Clojars
artifact API returns 404. Its README badges are not release evidence. Raster's public
north-star names this workload, but that does not make its source available to collaborators.
The local README/lora GPU file and decision-training subtree have uncommitted work;
preserve that work and obtain a reviewed source snapshot rather than silently publishing it.

The public `pretrained-rstr` repository supplies model loading/inference and numerical-memory
orchestration. Its local source/test tree has no LoRA/QLoRA training implementation; the
local-private-model roadmap lists a reproducible LoRA fixture as future work. Keep checkpoint
loading there and adapters, objectives and training orchestration in finetune-rstr.

Before presenting finetune-rstr as an installable training package:

1. Migrate its existing head/layer/adapter programs from retired `bind-program!`/`run-program!`
   to the public Compiled/LinkPlan contract; do not revive a compatibility runtime.
2. Replace the old default Raster `0.2.287` dependency with a validated released version,
   and make clean-clone tests independent of sibling checkouts and a local Valhalla JDK.
3. Preserve the held real-checkpoint gradient gate and add an end-to-end adapter save/load
   and inference replay fixture with unchanged base weights. Record numerical/performance
   limits honestly; small synthetic parity is not real-model training acceptance.
4. Review uncommitted sources, license, build/release setup and model-data handling before
   separately authorizing GitHub/Clojars publication. No publication is implied by this ledger.

CUDA work can continue against Raster's committed external-source acceptance fixtures and
public pretrained-rstr while that migration proceeds. It must not require an unpublished
sibling package to execute Raster's hardware-free compiler gates.

### Distributed algebra witness consolidation (2026-10-08)

Review found that the analytic collective witness retained routes/algorithm names but omitted
the reduction algebra, broadcast root and numerical mode. Accumulating halo witnesses retained
only combine/identity/dtype, dropping the algebra's numerical facets. Retained records also
bypassed their constructors, so record identity could masquerade as validation after mutation.

The existing associative scalar authority now independently rederives retained certificates.
Distributed scheduling revalidates operations, schedules and communication legs even when they
are records. Collective certificates retain the reduction/root/numerical mode, and halo
certificates retain the complete associative certificate. Valid semantic changes require new
distributed evidence; missing/forged algebra facts reject before certification. Old witnesses
must be regenerated. This adds no operator registry, domain kernel emitter or runtime transport.

The execution boundary remains narrower than the analytic planner: readiness accepts only copy
halos, and refuses reducing collectives/accumulating halos before sessions or allocations.
Arithmetic execution must eventually transfer into explicit scratch and run a generated typed
combine as an ordinary bound compute step. The analytic witness is not a collective numerical
equivalence proof, a real-fabric measurement or completion of campaign items 6/7.

Focused scan/distributed-plan/physical-binding/AMR-plan validation passes 70 tests / 370
assertions in the capped REPL. Independent review found no blockers; its extra qualified-operator
and retained-leg mutation checks are included. Full cold CI remains a merge requirement.

### Distributed physical assignments before arithmetic lowering (2026-10-08)

The next seam binds ordinary copies to exact existing local graph regions through
`DistributedPlan.copy-bindings`; see [distributed-materialization.md](distributed-materialization.md).
It reuses BufferView projection, shared copy extent/alias checks and the existing readiness/runtime
authorities. Source storage must realize the transferred value and stay within its owned region;
constant destinations, unproven ghost sources, malformed or strided endpoints and stale witnesses
reject. Explicit copies cannot override semantic collective or halo legs. The assignment may
target a different mutable logical value: it is a physical copy contract, not value-version or
collective algebra equivalence evidence.

Focused scan/distributed-plan/compute/AMR-plan tests pass 74 tests / 392 assertions. A generated
`acopy!` → resident scratch copy → generated `axpy!` fixture passes on the actual Level Zero device.
The complete affected numerical namespace passes 13 tests / 61 assertions after separately fixing
independent owned LinkPlan instantiations' runtime root identity collision. That lifetime fix keeps
compiler and explicitly borrowed identities unchanged; it does not relax root retention guards.
Independent physical-copy review found no blockers. Exact-head cold CI remains required; semantic
collective lowering, packing, real fabric/overlap and full-model training acceptance remain open.

The strict semantic restore boundary now checks the independently supplied target field/layout,
logical phase, numerical policy and provenance before leases/uploads; bit-preserving certificate
verification also rejects opaque metadata and signed-zero drift. Affected state/native restart,
lease and AMR checks pass 40 tests / 454 assertions. This matches declared contracts, not omitted
application semantics or mathematical equivalence; durable provider/publication and distributed
lineage acceptance remain open, despite the producer/source evidence now available.

External source audit refreshed on 2026-09-30: authoritative pretrained-rstr main
`98bad4d3517ec05a3c495f019c80d8c553db02af` declares Raster 0.2.922 and uses the public
compiled boundary for Laya/ModernBERT. The older local `f4c2bb4` checkout (0.2.545) and
`pretrained-rstr-main` checkout (0.2.457) do not describe current upstream. An isolated
current-main snapshot is used for focused acceptance; no sibling code or dependency was
modified. The local finetune-rstr `9e9ba5d` checkout still declares 0.2.287 and its layer
forward/backward code names the retired binder, with local edits that must be preserved.
Existing in-repository training or external inference evidence must not be substituted for
the remaining external training gates.

### Execution order from this checkpoint

1. Finish the local public workload/support matrix and host-preparation audit before adding
   vendor-device claims. Keep complete-write evidence distinct from conditional initialized
   postconditions; fresh prefix results must remain legal without crediting untouched parent
   tails. Scalar reductions retain their exact one-element resident representation. Invalid
   shape diagnostics propagate; unsupported proofs may decline. Public validations remain
   independent; redundant proofs may be shared only inside an exact synchronous construction scope.
   Exact-owner static program evidence now reaches final synchronous construction, after the
   projection callback. Concrete call and memory/effect checks remain fresh; public verification
   stays independent. The October 2 affected suites pass 103 tests / 696 assertions, with both
   local native backends available. Warm Q4 preparation is about 11.7 ms in one diagnostic,
   not a controlled performance baseline. Continue auditing structured-call invariant completeness
   separately; this deduplication does not certify arbitrary forged carry-plan/staging metadata.
   The subsequent source-ordered call audit now retains entry buffer bindings and checks loop
   trip counts, carry rotations, numerical argument projection, final bindings and exports against
   each step's actual entry environment. It does not reselect dispatch alternatives or rerun host
   callbacks. Its affected suites pass 120 tests / 889 assertions, with six cache-owner checks
   passing another 55 assertions. Comprehensive host-step evaluation evidence remains separate.
2. Select the next compiler migration from a reproduced public workload, using the compatibility
   ledger to distinguish stale entry-point accounting from an actual semantic/emission gap. Preserve
   the corresponding JVM path and independent numerical tests while retiring the migrated fallback.
3. Coordinate the external training adapter boundary using public compiled artifacts/LinkPlans and
   semantic outputs. Validate chained gradients and real model state before claiming that item done.
4. Continue matched public GEMM/projection measurements under the protocol in
   `bench/comparison/generated-kernel-protocol.md`. Current noisy laptop samples do not promote a
   winner or establish SOTA parity; a cloud accelerator is optional later evidence, never a CI gate.
5. Extend numerical execution to a conservative PDE evolution with coarse/fine transfers and an
   uninterrupted-versus-restored oracle. Reuse existing field/lease/plan contracts. Introduce any
   additional semantic operation only when that workload exposes a concrete missing obligation.

The completed-reduction conversion follow-up removes the numerical-region projection gap for
one explicit Float/Double cast of the completed accumulator. Partial storage stays at the reduction
dtype. The scalar cotangent alias projection in #948 also removes the reproduced cold dynamic-helper
typedness gap; #949 admits placement across unrelated pure equations without crossing effects.
General mixed-width epilogue arithmetic and public Float-accumulator widening remain separate.

The immediate emphasis returns to unresolved compiler/training workload coverage after the durable
transfer acceptance; the distributed and scientific work does not replace the emitter agenda.

### Checked scalar conversion checkpoint

#565–#568 landed runtime VJP-cache invalidation, resident scalar composition, shape-only array
captures, and KernelBody signed narrowing with explicit target trap contracts. Source conversion
integration keeps checked casts distinct from unchecked wrapping, preserves ordered lexical
initializers and unused checked prefixes, and separates one-shot extent evaluation from CSE.
Beichte's external-effect classification remains separate from descriptor-derived exceptional
control obligations; only retained exact-conversion or checked constant evidence removes them.

The integration follow-up keeps implicit result/storage conversions out of the source-cast tree,
so explicit casts inside maps and stores retain their original checking and rounding behavior.
Padded map lanes guard the complete scalar region, not just memory operations. Retained branch
and loop-entry intervals may prove an integral narrowing exact; an unrestricted long-width argmax
still requires a trapping target or an explicit bound proof. Required exceptional prefix shapes
have one evaluator in the typed InvocationPlan, with source-order evidence retained across
non-equation host bindings. These changes are under PR #569 validation, not a green release claim.

The declared result cast of `par/map!` is itself source semantics, distinct from an implicit
array-store coercion. The integration uncovered a missing boundary when the body contains no
explicit inner cast. A typed scalar-conversion term is being validated through fusion and target
projection; it must retain materialization rounding and exceptional ordering, not merely the
destination dtype. Once-per-element evaluation alone does not justify moving a potentially
trapping producer across a consumer that writes observable storage. JVM AOT also now threads
declared scalar/array facts into the shared frontend, and explicit primitive long-to-int bytecode
uses checked narrowing. Focused JIT/AOT boundary tests pass; the full PR remains unmerged pending
conversion/fusion and CI regressions. Boxed numeric and native-C exceptional conversions remain
separate audit gaps, not covered by the primitive JVM result.

Typed scalar conversions also expose a JVM SIMD capability boundary: an exact float identity
conversion can disappear only inside a float vector computation. It remains a rounding boundary
when embedded in a double reduction. The shared typed scalar route preserves mixed-width programs;
all-float map/reduction cases must still vectorize. General mixed-width Vector API conversion and
lane-shape handling remain an explicit performance task, with numerical materialization-boundary
oracles required before admission is broadened. A scalar fallback here is not SIMD parity.
The historical raw source-cast SIMD path still lacks authoritative operand dtype evidence;
matching its destination to the vector species alone does not certify arbitrary nested casts.

Host-only scalar steps currently execute during preparation. Checked scalars after device work
therefore decline rather than moving a potential failure ahead of preceding writes. General
interleaved host/device exceptional control remains future work. Portable OpenCL still declines
unproven trapping device casts; CUDA/HIP/Intel termination is not a JVM exception-delivery promise.
This work does not certify the scalar simplifier's historical floating-point algebra under all
NaN/signed-zero cases, nor complete the remaining counted-map migration or compatibility census.

### External scientific comparisons

The [source-reviewed extension experiments](scientific-extension-validation.md) add Julia/SciML
and matrix-free references while preserving the existing ODE and iterative-solver library. They
distinguish library capabilities from verified resident/distributed compilation; do not infer
missing numerical algorithms from gaps in the new accelerator fixtures.

Use numerical frameworks as well as kernel compilers, with separate kernel and time-to-solution
measurements. Initial comparisons should match precision, discretization order, boundary conditions,
mesh, timestep/stability constraints, stopping tolerance and transfers. Also compare time to a fixed
solution accuracy, rather than treating a cheaper discretization as a compiler speedup.

| Reference | First Raster workload and question |
|---|---|
| [Devito](https://www.devitoproject.org/examples/userapi/01_dsl.html) | Heat/acoustic stencils: symbolic numerical programs, fusion, scheduling, bandwidth and halo work. |
| [AMReX](https://amrex-codes.github.io/amrex/docs_html/) | Block-structured coarse/fine transport: ghost exchange, conservation, refinement interfaces and placement. |
| [PETSc](https://petsc.org/release/manual/dmbase/) | Poisson/implicit evolution: compose operator application, residuals, preconditioners and convergence control; measure time to the same tolerance. |
| [MFEM](https://mfem.org/howto/assembly_levels/) | Later matrix-free finite elements: element/face gather, basis/quadrature contractions and assembly/scatter layouts. |
| [UFL](https://docs.fenicsproject.org/ufl/main/manual/form_language.html) | A semantic reference for domain-aware mathematical expressions and differentiation, not a standalone runtime benchmark. |

Start with Devito, then AMReX and PETSc; use MFEM/UFL to challenge regular-grid assumptions later.
Keep these comparisons out of the hot test loop. Record dependency revisions, numerical settings,
hardware and compiler/runtime revisions with each result. No installation, cloud service or sibling
source modification is implied by listing these baselines.

The architectural test is whether numerical problems lower through the same small functional core:
for example, gather states → compute face fluxes → signed accumulation → update cells. Mesh entity
relationships, geometric factors, boundary conditions, solver convergence and numerical error
contracts belong above that core when needed; scheduling, storage and communication remain explicit
below it. Changing resolution is a numerical approximation with a projection/error contract, not
automatically an equivalent compiler rewrite. Current rectangular cell-centred AMR and explicit
distributed DAGs do not yet establish arbitrary mesh, solver or adaptive-control generality.

## 1. Close product reduction production lowering — landed for the dense row subset

- Landed #382: shared typed multi-result scalar-region lowering.
- Landed #383: candidate row-product KernelBody, CPU OpenCL differential execution, CUDA/HIP
  compilation. At that stage production still used the retained source emitter.
- Landed #384: preserve TypedSOAC local dtypes through SegRed and subsequent
  scalar/index lowering; compare both direct and local-address candidates on CPU OpenCL.
- Landed #385: align the retained product KernelGrid with the segment-count launch,
  validate scratch/workgroup agreement, and replay the body refinement against the retained source.
- Certify the exact semantic source, storage boundaries and public bindings against the graph.
- Landed #386: derive product pointer contracts from the exact retained
  scheduled-equation projection, shared with fold-map. Unknown capacities and unlowered storage
  representations decline. This does not prove arbitrary indexed accesses in bounds or enable
  production selection; those obligations remain explicit.
- Exercise ordinary public compilation, resident execution and hidden/multiple tuple results.
- Cover required axis/bound variants and zero-row planning; reject unsupported cases explicitly.
- Empty-domain kernel follow-up uses one physical group and a workgroup-uniform guard enclosing
  every reduction operation, barrier and store. CPU differential tests execute this path and verify
  untouched outputs. This does not implement zero-byte device allocation or graph-node elision.
- Long-bound follow-up preserves independent row/column int or long facts through the public
  scalar ABI and widened induction. CPU differential execution includes Long dimensions and
  retained local addresses; CUDA/HIP fixtures compile the same candidate. This is still not the
  production route; source access requirements remain open, as does eventual zero-row elision.
- Before widening public dimension support, bind concrete launch geometry to KernelBody's int
  hardware-index representation, including scheduler overrides and compatibility staging. Logical
  Long dimensions must not silently overflow group/local indices; target resource limits remain
  a separate, potentially stricter contract.
- Select the certified body and delete the old product algorithm emitter once its coverage is
  replaced. Keep necessary target instruction lowering.
- Typed access follow-up extends the existing AxisMap relation with a bounded conditional
  intermediate-range proof over KernelBody indices. Exact polynomial coefficients prevent host
  overflow; every add/multiply subtree is checked before affine normalization. The proof assumes
  active positive axis domains and checked resident capacity; deriving those access requirements
  from scalar lowering and enforcing them in graph binding remain production obligations.
- Read-requirement follow-up shares product scalar-region lowering with scheduling and derives
  flat minimum capacities from actual typed dense loads, without invented input shapes. The
  graph preserves larger declared capacities and combines shared-storage requirements; existing
  checked resident/staged binding enforces graph capacities. Gathers, broadcasts and combine
  reads remain unsupported by this initial access proof. Production selection still requires
  successful access admission in addition to exact source/body/graph correspondence.
- Production follow-up admits dense row products through that combined check and the common
  C-family graph emitter. Public resident extraction uses the existing executable convention;
  effect-only staging has an explicit nil result policy, including multiple written outputs.
  Direct descriptor graph binding now enforces the same capacity premise as session/LinkPlan
  binding. Hidden/multiple outputs are checked across all three target dialects; public argmax
  has resident and staged CPU OpenCL numerical coverage.
- Retirement follow-up removes the handwritten product emitter from production sources and
  preserves its algorithm in `test/raster/compiler/reference/product_opencl.clj`. Existing source
  contract tests and CPU OpenCL differential execution retain that independent oracle. This does
  not promote its broader source-only cases (computed reduction bounds or multiple segment axes)
  into supported production coverage; the initial certified route remains dense and rank-one.

Exit: public product workloads use the typed route without reconstructed facts or source assembly.

## 2. Retire the remaining compatibility paths — active

The public staged Byte→Int32→Float contraction is now a separate compatibility-ledger
workload. Unlike the explicit-map Q4 row projection (already KernelBody), it declines
direct TypedSOAC admission and reaches retained `:staged-segred` emission. Its current
compatibility path requires explicit `:dtype :byte`; default Float policy emits Float
operand pointers despite declared Byte inputs, which the resident binder correctly
rejects. Do not count fixture-only staged KernelBody emission as this public vertical.

The next admission must retain the existing `SoacContract`/`ContractionFacts` payload,
including nested stage boundaries, typed lifts and storage dependencies, through the
typed program and projection. Flat ProductReduction components are simultaneous
accumulators, not nested stages. Check operand/lift/output types independently; do not
relax the frontend option gate or flatten stages into a scalar fold. Existing fusion
rules must decline staged nodes until their legality preserves those boundaries.
Acceptance includes exact fact transport without reparsing, public mixed-storage
execution, source return/effect preservation, and checked graph capacities/aliasing.

The dependency prerequisite projects core, stage-lift and epilogue storage plus scalar
captures once from ContractionFacts. SOAC and SegOp accessors share it; stage/epilogue
binders have local scope, and declared maps replace placeholder indices before capture
analysis. Decode expressions reading undeclared storage are explicitly rejected. A direct
lowering test checks equation operands/effects and Byte input, Float scale/output/result,
and integral scalar types. Floating declarations now specialize only under floating
kernel policies, and contraction result aliases inherit the destination type. This
repairs the existing dependency envelope, not direct TypedSOAC staged admission.

The next dialect slice represents a `contract` equation as a lexical closure over the
same ContractionFacts, with ordered external array/capture value IDs. SSA renames those
IDs, not the nested stage expressions or axis binders. Validation checks independent
storage dtypes/capacities, scalar closure, output shape and explicit read/write storage
contracts. Existing flat fusion rules recognize but do not rewrite this operation.
The binding projection retains the original facts for the generated staged body and
maps its arguments onto graph value IDs. This is a prerequisite: the public frontend
still needs admission and production routing before its compatibility ledger can close.
Closure validation is not an independent arithmetic type checker: it checks declared
storage/scalar types and lexical scope. The existing scheduling and scalar-body lowerers
must still prove instruction types and numerical legality. Unlowered representations,
layouts and sharding decline, and outer lifts cannot refer to inner-stage coordinates.

Production graph lowering now transports that closure as a bound SegContract, with
canonical dependency accessors used by graph construction. Before generated staged
emission, the existing graph projection certificate and an exact operation/binding
comparison both run; the resulting ScheduledKernelBody is checked against that node.
This route uses the common C-family graph emitter and resident graph binder. A small
Arc device test exercises vector SSA IDs and distinct row/column outputs. Public
frontend admission and the compatibility-ledger transition remain the next step.

Read-only inventory at #383 identifies these priorities; dynamic reachability must be measured
before treating a definition as live or dead:

| Priority | Boundary | Representative acceptance |
|---|---|---|
| 1 | General contraction fallback in `contract_route` / `segop_opencl` | General/scientific contractions, exact decline inventory |
| 2 | Unscheduled effect maps in `opencl_pass` / `par_opencl` | Resident mutation, aliasing, nil/buffer returns |
| 3 | Strided scatter/gather source emission | Block transfers, collision and view contracts |
| 4 | Seeded active-ID generation | ABM firms, stable order and count/capacity; not general predicate compaction |
| 5 | Compound-local algorithm emission | Structured loop dependencies, storage and barriers |
| 6 | Staged/quantized contractions | Decode/lift algebra, typed accumulators and overflow |

Audit apparent unused wrappers separately. A source oracle is not a production fallback; removing
one must not erase its independent numerical evidence. Track JVM and C/SIMD compatibility as well
as GPU paths, using the same production corpus and retained TypedSOAC facts.

A repository caller audit found four orphan NN source generators (row softmax, group norm,
vector scatter-reduce and scalar scatter-reduce): their only source callers were unreferenced
adapter wrappers, while tests inspected generated strings. The wrappers are removed and the
unchanged generator bodies live in a test-only reference namespace. Existing source assertions
remain, with scalar-scatter coverage added; these are source-shape oracles, not numerical or
accelerator performance evidence. Shared dtype/extension helpers remain production infrastructure.
This removes dead algorithms from production, not a live workload's fallback or an optimization.

The handwritten tiled GEMM generator and legacy direct DPAS source entry are now test-only
oracles. Production DPAS artifact construction has only the verified KernelBody branch, preserving
its source, ABI, epilogue metadata and launch geometry. Independent source comparisons and
historical benchmark rows still use the relocated generator; those rows are explicitly not public
compiler performance evidence. The legacy orientation gate remains shared with oracle tests for
now; live staged quantized contractions still require the epilogue-splice helper. This retirement
does not remove their source emission or the general gathered-contraction fallback.

The zero-reduction contraction router now attempts the shared portable contraction KernelBody
lowerer, using a semantic SegMap projected directly from verified facts. The map branch emits
scalar operations and one store without a synthetic reduction, and obtains each input extent from
its own proven AxisMap (not the output count). Initial admission is deliberately limited to
positive static free-axis extents whose product fits the int launch ABI, no options/result transforms or destination reads, and one
index expression per input array. Unsupported maps retain an explicit compatibility decline.
Focused CPU OpenCL execution uses exact input lengths and a partial final workgroup; the same
routed body is included in CUDA/HIP compile fixtures. This is a contraction-router migration, not
yet the public equation-first frontend: that frontend still excludes zero-reduction contractions.
Admitting them through retained map equations requires preserving independent input capacities;
the generic map lowerer's output-sized input shapes must not be reused as an outer-product proof.

The first generic map-capacity step preserves known positive static GraphBuffer capacities in
each KernelBody pointer shape/layout, including larger retained input capacities. An independent
check compares those static pointer contracts to the exact source graph node before target
emission. The typed-map fixture carries AbstractValues a[4], b[3], C[12] through ordinary typed
scheduling and graph construction; CPU execution uses exact buffers and rejects undersized inputs
and outputs. Source, launch and arguments are unchanged by enlarging a retained capacity. This is
storage-contract correspondence, not proof of arbitrary indexed access safety. Symbolic, unknown
and zero capacities still retain the compatibility behavior; no shape-only scalar ABI is added.
Public zero-reduction contraction admission remains a follow-up, as does access-derived capacity
refinement where the frontend has no declared shape.

CI also exposed target-registry leakage from test fixtures: matrix-capable synthetic targets can
change another test's automatic precision route. The direct contraction ABI test now requests its
FP32 policy explicitly. The isolation follow-up gives the hardware registry tests and the three
resetting cross-compilation fixtures fresh device, initialization and calibration atoms, restoring
the original identities even on exceptions. This is a serial-JVM fixture; parallelism remains
across CI processes. Other registration sites still need an ownership audit.

The first measured elementwise gap is unary subtraction in `scale-clamp-exp`: its typed source
reached the verified SegMap source fallback solely because the scalar lowerer required binary
subtraction. The follow-up uses the existing negation intrinsic and shared prefix spelling for
floating operands (preserving signed zero); integral operands retain checked subtraction and
its explicit portable-OpenCL trap limitation. Public route, CPU OpenCL IEEE cases and vendor
compile fixtures cover this increment. It does not retire the remaining ordered-loop fallback.

The next measured gap is `sum-kv-heads`: its ordered carry starts at one after loading the first
element. The origin follow-up retains nonnegative literal starts in the existing scalar ForLoop;
zero-origin SOAC recognition is unchanged. New nonzero coverage requires direct recursion, exact
binding-slot updates, one loop body and no narrowing induction test. Public GQA fan-in, group-one
signed-zero preservation and cancellation-sensitive accumulation are checked on CPU OpenCL.
This does not authorize reassociation or arbitrary symbolic/negative loop origins.

Coverage reporting now retains the normalized emission routes and decline details from each
existing corpus compilation. TypedSOAC frontend coverage alone cannot distinguish generated
KernelBody from a compatibility emitter. Aggregate counts explicitly describe emitted artifacts,
including dispatch alternatives, not executed launches. The portable baseline remains unchanged;
target-specific route evidence stays in the report and CI output. No extra compile is required.

Static caller audit found the old `par-hip` elementwise source emitter referenced only by its
historical compile-gate test, not by any production routing path. It is moved to the test-only
`reference.elementwise-cuda` namespace with the algorithm unchanged. Historical ABI/math compile
checks remain; mandatory CUDA/HIP CI continues to compile public equation-first KernelBody
fixtures. This removes a misleading second production emitter, not a supported runtime target.

The next caller audit distinguishes a remaining source edge from demonstrated runtime reachability:

| Boundary | Retained owner / consumer | Current evidence |
|---|---|---|
| General contractions | `generate-segmented-reduce-kernel` → `contract_route` | Nested gather selects the source fallback after `:operand-layout` decline. |
| Unscheduled effects | Typed mini-program → KernelBody or fail-loud decline | Raw/nested plain-array and logical-SoA effects schedule once; unsupported bodies cannot re-enter source emission. |
| Strided transfers | Typed mini-program → KernelBody | Bare and host-wrapped leaves share typed scheduling; source generators moved to test-only oracles. |
| Active IDs | `generate-par-active-ids-kernel` → `opencl_pass` | ABM firms uses seeded agent indices, not arbitrary visibility compaction. |
| Compound local | `generate-compound-local-kernel` → `opencl_pass` | Compatibility markers remain; structured typed programs bypass detection. |
| Staged/quantized contractions | `generate-staged-contraction-kernel` → `contract_route` | Device-tested staged/lift/quantized source assembly remains. |
| JVM SIMD / scalar | `pipeline` → `par_simd` or source expansion | Bound SegOps coexist with traversal of the retained source projection. |
| CPU C/SIMD | `cpu/aot` → `cpu/csimd` | Bound SegOps preferred; compatibility reconstruction and distinct vector emission remain. |

Before migrating staged contractions, their numerical admission must preserve the claimed
flat-equivalence law: stage initializers are proven zero using the shared checked-constant
semantics, and packed signed-byte accumulation requires every prefix to fit the DP4A instruction's
Int32 result, even when the surrounding accumulator is Long. The reusable interval-prefix proof
uses unbounded host arithmetic. This closes those admission gaps, not the remaining source
emitter's general integer overflow semantics or the typed staged lowering itself. Finite-precision
equivalence still needs a per-stage conversion/rounding contract; exact-arithmetic distribution
is not a proof of bitwise equivalence.

DP4A staged admission now belongs to `passes.parallel.staged-contraction-schedule`, not the
OpenCL emitter. Both routing and the retained source generator consume the same checked packed
maps; the old emitter-owned admission entry is removed. The next production slice is typed
packed-stage loops and lifts, using existing ScalarLoad/ScalarCompute/ForLoop operations and
shared scalar lowering. DP4A instruction acceptance itself proves no accumulation bounds: retain
the schedule's prefix gate. General scalar integer stages additionally need a checked recurrence
proof (including zero trips and cross-carry dependencies), not weakened KernelBody validation.

A REPL packing probe lowers four byte reads and existing bit operations into verified KernelBody
without reinterpreting the public byte-buffer ABI. Its target emission exposed an adjacent shift
contract gap: C-family word shifts now explicitly mask counts to the retained Int/Long width,
shift unsigned words, and restore arithmetic-right sign bits. Byte shifts decline centrally until
source lowering supplies an explicit promotion/result contract. This does not authorize narrowing
Long source shifts to Int. The shared fixture covers sign-bit packing and signed/count extrema on
OpenCL, with CUDA/HIP source compilation in CI. It is a portability prerequisite, not evidence that
four byte loads match the throughput of an aligned packed-word load or that staged emission is
already unified.

Staged route construction now reads the body and output dtype from the same verified contraction
facts as its axes and stages. It no longer reconstructs or reparses a compatibility form just to
build the existing staged emitter's spec. Regression tests disable both source entry points and
compare ordered ABI, scalar bindings, launch geometry and stage metadata for scalar and packed paths,
including a non-default output dtype. The staged algorithm is still source-emitted; this retires
an input-representation duplication, not the remaining target emission path.

The first typed packed-stage **candidate** now constructs ScheduledKernelBody directly from those
facts. Two-level Int32-dot/Float-lift reductions use byte loads, explicit word packing, DP4A, nested
typed carries and masked stores through the common target projector. Static AxisMap products bound
required storage and index arithmetic; semantic decode declarations, unproved prefixes, arbitrary
lift regions and unsupported domains decline. The required shapes are not proof that an arbitrary
raw backend caller supplied enough storage. Production admission must retain the checked resident
capacity boundary; the low-level OpenCL test binds known correctly sized buffers.

The generic `emit-static-dense-graph` projection now places a ScheduledKernelBody behind the
existing KernelGraph binding boundary. It derives every external buffer's required capacity from
positive static dense shapes, preserving logical storage dtypes, exact source/effects, public
scalar dependencies and target-private scalar expressions. Unknown/strided storage is rejected;
this is neither a new memory ABI nor an access proof for arbitrary producers. Tests exercise
operand, lift and output shortages through both session and descriptor binding, writable aliases,
and device replay with poisoned output. The staged candidate can use this boundary without
changing production schedule selection; performance admission remains outstanding.

Candidate validation covers signed-byte extremes, block widths 4/32/64, multiple blocks, distinct
row scales and masked launch tails on OpenCL; CUDA/HIP compile fixtures exercise the same typed
schedule. The old staged source emitter remains production-selected. Before promotion, compare
old scalar/packed and typed candidates on the same workload, including finite-precision rounding
and throughput, then migrate descriptor construction and remove the superseded emission. Four byte
loads are not an aligned packed-word throughput claim. More general stages still need the checked
recurrence and typed lift-region coverage described above.

The follow-up device comparison binds the same logical Byte ABI for typed, retained scalar and
retained packed emission, with independently poisoned outputs. Dyadic scale cases compare exactly;
non-dyadic cancellation compares against explicitly stage-rounded host arithmetic with tolerance
relative to contribution magnitudes. An all-zero result must fail that tolerance. Prepared kernel
handles are released after synchronous execution.

A small shared-laptop Arc probe (OpenCL profiling events, 3 warmup rounds and 12 interleaved timed
samples per candidate, 64 work-items/group, uploads/compilation excluded) gave the following median
milliseconds for `[rows outputs blocks block-width]`: `[1 128 32 32]` typed 0.052708, retained scalar
0.039166, retained packed 0.032708; `[4 128 32 32]` typed 0.041770, scalar 0.039062, packed 0.031458.
All outputs matched the independent reference. Samples were noisy (e.g. typed second-case range
0.040104–0.072187 ms); this is a directional probe, not a reproducible performance baseline or
regression in the unchanged production route. Do not promote the byte-load candidate on this
evidence. Investigate a generic, explicitly typed/aligned packed storage load (with byte fallback),
then repeat resident comparisons before retiring the production source emitter. Do not introduce
a quantization-specific memory ABI or bypass the common body verifier to obtain that load.

One general target-cleanup slice exposes canonical counted loops when their literal bounds prove
the final induction increment representable. The shared exact range helper includes that exit
increment, not just indices observed inside the loop. Ordinary and asynchronous pipelined loops
share one syntax helper; unknown bounds and explicit IndexCasts (including retained Long bounds)
keep the existing unsigned-distance guard. Scalar yields and event rotation stay before advance.
This does not strengthen recurrence facts or alter accumulation order, and the noisy laptop probes
do not establish a throughput improvement. Device tests cover zero/nonunit/reversed loop counts,
staged reductions and the existing asynchronous workgroup pipeline.

Public matmul/dA/dB probes on CPU OpenCL select generated portable contractions, not the
handwritten gather fallback. Their artifact adapter previously reported semantic `:segcontract`
provenance as the emission route. The follow-up propagates the actual emitter's route through
the adapter; semantic provenance remains separate. No schedule or precision changes.

Strided transfer retirement also closes the raw host-wrapper entry: `do`, conditional and
body-position leaves schedule their scalar extent beside the invocation, without moving it out
of the branch. Supplied typed programs are consumed without re-analysis and reject missing
equations. Existing source-oracle assertions remain test-only. CPU OpenCL checks inactive branches,
output identity, gather overwrite and repeated-index scatter accumulation into nonzero storage;
this is correctness evidence, not an accelerator performance measurement.

Active-ID retirement uncovered a shared source/ABI type bug: `derive-param-types` narrowed every
declared Long to int before TypedSOAC, including random seeds. The prerequisite now retains
declared integral widths at every caller of that common derivation. This changes affected scalar
ABIs; consumers must bind the emitted ABI, not assume an int slot for a Long parameter. Hardware
group/local index types are unchanged; schedules requiring int parameters still need explicit
checked narrowing. Public equation-first, resident and session paths are tested with 64-bit state
and wrapping arithmetic. The active-ID prototype remains unshipped until its checked int count
conversion is also retained; this prerequisite alone does not retire that kernel.

The next prerequisite distinguishes shape-value equality from unchecked cast erasure. Source
normalization removes integral identity/widening casts only when retained types prove them safe;
narrowing and unknown conversions remain scalar equations. Relational extent canonicalization
must not subsequently erase the same check. A public checked count is evaluated by the resident
descriptor before driver contact; horizontal fusion retains the scalar equation ahead of its
combined launch, and inactive host branches do not evaluate their local count. This does not yet
repair convenience operations that unconditionally unwrap counts or prove every compiler pass
preserves checked conversions.

Exit: classify all remaining routes; retire duplicated paths for covered workloads with explicit
production coverage and no silent legacy re-entry. Report any remaining gaps rather than declaring
the entire compiler unified from one successful vertical.

Scalar-emission audit removes the map/fold lowerer's private copy of primitive cast spellings in
favor of the shared operation descriptor and dtype vocabulary. This does not change narrowing or
rounding policies. Before sharing that lowerer with contraction elements, reconcile its explicit
device-wrap narrowing policy with reduction lowering's checked-cast rejection; do not widen
reduction admission accidentally. Indirect contraction loads also need independent storage
capacities and a data-dependent index-range contract. A dense AxisMap or an index array's integral
dtype does not prove its contents are in bounds. Keep the production operand-layout gate until
those obligations are represented and enforced, reusing existing typed loads and graph capacities.

The scalar conversion prerequisite now derives common rounding/overflow choices from dtype facets
in one pure helper, used by map/fold expression lowering and reduction element lowering. Integral
narrowing defaults to rejection; the existing map/fold device-wrap behavior is an explicit owner
opt-in, not an inferred equivalence to checked Clojure casts. Reduction's source checked-cast gate
is unchanged. Exhaustive primitive conversion pairs and owner-admission checks protect this seam.
This does not unify the full scalar lowerers, change epilogue conversion labeling, or admit new
indirect accesses; those remaining differences must be closed separately.

Retained scalar precision is a prerequisite for further sharing: a nested expression's declared
floating type must govern its arithmetic before a wider consumer converts its result. A direct
map/fold lowerer regression exposed early promotion of a Float-tagged addition inside a Double
addition. Lowering now respects retained expression metadata, including unary/variadic
normalization and branch results; missing metadata still uses the owner's contextual type.
The lowerer explicitly converts the completed floating result back to the owner's declared
dtype, including narrowing before stores and loop yields; preserving inner precision must not
violate those boundaries. Public softmax and heat-equation corpus checks exposed this obligation.
Integral contexts retain their existing behavior: the public Q4 projection has widened loop carries
around narrower dp4a result metadata. Reconciling that boundary is a separate prerequisite, not
permission to silently change accumulator widths in this floating-precision increment.
This is lowerer-level correctness coverage, not a demonstrated public-model miscompile or a
new inference rule. Full source reachability and remaining owner-policy differences remain open.

Reduction element lowering now delegates accepted loads, casts and arithmetic over typed child
values to the shared scalar-expression SSA builder. Its admission adapter still requires retained
nested arithmetic types, preserves literal dtypes, rejects checked integral casts/arithmetic and
comparisons, proves load coordinates through the existing owner callback, and checks the final
accumulator dtype. The shared builder accepts explicit owner conversion and masked-load policies;
map/fold defaults are unchanged. This serves existing scalar reductions and portable contractions,
without adding indirect-access admission or retiring their remaining source fallbacks. Result
transform admission still has its own explicit precision and overflow policy.

The shared scalar builder now reserves source, parameter and owner-scope identities before fresh
SSA allocation, including future local binders. Map and reduction fragment callers seed their
complete source region; generated coordinate names are reserved when the fragment arrives.
Reduction fragments pass only their typed child environment instead of all earlier SSA values,
and seeded calls do not repeatedly scan growing environments. Names remain deterministic, and
KernelBody's duplicate-identity validator remains intact. This closes the demonstrated scalar
capture cases; it is not a claim about every other schedule's allocator or target identifier mangling.

Result transforms now use the same typed load, conversion and scalar-compute builder as reduction
elements and map/fold source lowering. Reduction and result-transform adapters no longer rebuild
source fragments to emit accepted operations. The typed entries take approved coordinate vectors,
converted operands and explicit options; they do not infer source semantics or new range proofs.
Result-transform operand roles (including destination reads), multidimensional coordinates,
conversion labels and trapping integer arithmetic remain unchanged. Their separate admission
rules are not duplicate kernel emitters, and harmonizing those rules remains a semantic decision.

The active-ID prototype is deliberately **not shipped**. Review found that its remainder-based
body matches source `mod` only over positive populations, and its output conversion needs a proved
int range. It also inherited unconditional cast erasure on seed/population captures. Passing
ordinary ABM differential tests does not establish those preconditions. Keep the production route
until shared capture normalization and enforced scalar-domain admission preserve empty/inactive
execution as well as overflow. The local `compiler/active-indices-typed-map` branch retains the
prototype and count-check tests for that follow-up; no retirement is claimed.

The next prerequisite closes the existing RNG route's scalar inputs: source-signature count and
seed conversions become ordered host scalar equations, rather than unwrapping a captured checked
cast. Typed scalar materialization reapplies its retained result conversion, preserving
`long(int(seed))` even after value-equality simplification. Raw inactive host branches keep these
checks local; missing certified leaves fail once instead of recursively rescheduling. Public
descriptor preflight, CPU differential execution and vendor compile fixtures cover this boundary.
This does not admit the active-ID population domain or retire its production source kernel.

Raw effect-map follow-up uses the same one-attempt typed mini-program boundary as RNG and strided
transfers. Conditional leaves keep checked extents beside their invocation. Plain mixed-storage
maps and pre-store lexical snapshots have CPU OpenCL differential checks; tiny host-only bodies
still use the existing scalar fallback. A supplied typed program cannot re-enter source lowering.
Unscheduled raw bodies now fail with the retained typed scheduling diagnostics and `:fallback
:none`; they cannot enter the independent source emitter. Logical SoA values are scalar-replaced
before TypedSOAC construction, and bare, conditional and let-bound probes retain per-field dtypes,
stable field identities and one grouped logical binding through KernelABI projection. Tiny literal
host-only bodies still use the explicit scalar host fallback below the configured GPU threshold.
This retires the production compatibility edge; moving the orphaned generator and its historical
source tests out of production is the following mechanical cleanup.

## 3. General fusion and bounded specialization — price admission active; broader work queued

- Validate shared scalar-region composition, coupled reductions and post-reduction transforms.
- Audit recompute/materialize choices on fan-out, effects, aliases and observable results.
- Apply retained static structural facts before fusion, shape/layout/target facts before scheduling,
  and typed constant/range simplification afterward.
- Preserve ordered arithmetic and AD/effect boundaries. Use guards and bounded specialization
  caches; do not unroll large numerical iteration spaces merely because their sizes are known.

The first generic hardening rejects nonnumeric, nonpositive and nonfinite machine ridge prices,
and overflow of their derived recompute threshold. These prices cannot authorize duplicating a
fan-out producer. Sole-consumer elimination still needs no cost estimate. The witness policy is
versioned; production typed-route tests retain both materialized consumers with an explicit reason.
This is price admission, not a new fusion legality proof or a bounded specialization cache.

An observable-result audit found a separate compatibility-boundary bug: horizontally fused stored
maps submit through an effect-only kernel convention, but their source bindings still return
buffers. Returning the submission's nil value lost the first host alias despite correct device
writes. Typed materialization now emits one effect binding followed by every source buffer alias
as a flat binding; resident extraction needs no new nested-call convention. Effect-only source
maps remain nil. Tests check the public resident descriptor, buffer identity and CPU OpenCL values,
alongside retained checked-count rejection. Kernel ABI, launch count and fusion legality are unchanged.

Automatic Object-array element-class specialization now has a separate, explicit admission bound
(default 32 attempts per generic function, configurable down to zero), with generic dispatch at
capacity. One atomic claim owns each pending compilation, and clearing the cache revokes its old
completion token. Failed/pending entries consume the budget too. This bounds this existing host
specialization cache, not the number of functions, manually generated JVM classes, GPU shape
variants or global compilation memory. It is not yet the shared cluster/resource admission policy.

Exit: representative scientific and model expressions compile with explained fusion/placement
choices, differential correctness, and measured kernel/allocation/compile-cost changes.

## 4. Measure generated production kernels and selection — GEMM evidence active; broader ladder queued

The [generated-kernel comparison protocol](../bench/comparison/generated-kernel-protocol.md)
defines baseline selection, per-shape records and measurement boundaries. Baseline adapters
and external accelerator measurements remain follow-up implementation work.

Begin with existing production canaries, not independent handwritten builders:

- `test/raster/perf/production_canary.clj`: public AOT sumsq and resident generated GEMM;
  GPU replay currently measures host-synchronized latency, not pure device throughput.
- `bench/soac_contract_bench.clj`: contraction ladder; verify its entry point before reuse.
- `bench/resident_gemm_cold_bench.clj`: useful cold/warm methodology, but its source-oracle builder
  must be replaced before it counts as generated-production evidence.

Measure GEMM, quantized projections, reductions and attention with matched shape, dtype, numerical
policy, warmup and transfer boundaries. Record compiler/tuning time, kernel count, allocations,
peak storage, transferred bytes and execution time. External comparisons and heavier experiments
stay outside the local hot loop. Use device events where available and label CPU OpenCL results as
CPU results. No SOTA claim follows from successful vendor compilation alone.

The first scientific-stencil canary now exercises the public periodic heat `deftm` with an
independent oracle and resident Level Zero event timing. It exposed a one-work-item serial loop
at `128×128`; spelling independent cells as a SOAC map with the store indexed by the map lane
made uniqueness provable and changed the launch to 256 lanes/workgroup. An intermediate
stationary run measured 19.8 µs; the final portable source measured 15.4 µs median but its
sample series was nonstationary. Every run matched the oracle. This is a measured schedule
correction, not a stable speedup claim or vendor-library comparison. Raw samples are retained in
the generated-kernel comparison protocol.
The first attempted map stored through a derived `(int lane)` local and still scheduled one
work item. That is a real proof-coverage gap: a narrowing cast is injective only when its
runtime domain is proven to fit, so the next general ownership slice should carry an explicit
range fact through the local and certify the derived address. Do not strip the cast or trust a
`unique-index` claim without that proof. The current source uses the original lane for the store.

Use the measurements to shortlist legal schedules and validate selective autotuning/cache replay.
If comparable accelerator hardware is unavailable, complete harnesses and correctness gates but
leave accelerator competitiveness explicitly unmeasured.

The GEMM canary records candidate strategy, source/ABI signature, emission routes and entry-point
counts from its exact Prepared value, with no second compilation. These replace the parallel
strategy/signature arrays in the opt-in report. Candidate counts are not executed launches;
descriptor scratch counts exclude graph-owned temporaries and do not claim peak memory. The
register-tiled emitter now reports its actual KernelBody route through the artifact adapter,
separate from semantic contraction provenance. Host-synchronized timing remains labeled as such;
no device-event timing or accelerator comparison is inferred from this evidence.

Exit: reproducible production measurements identify wins/regressions and explain schedule choices;
claims are limited to tested hardware and workloads. Discuss the results before stages 5–6.

## Working loop

Dense workload follow-up: public parameterized GEMM and its explicit typed ReLU epilogue have
precision-selectable correctness canaries. An ordinary contraction-return alias followed by an
in-place map exposed two boundaries: lexical return-type propagation (now uses exact primitive
operand-return semantics, not suffix guessing), then stable-read alias normalization (still
rejected at binding). Closing the latter and measuring automatic fusion remains required; an
explicit epilogue is not evidence that source composition fuses automatically.

Future mechanized proofs may use the sibling Lean project. Start with small existing contracts
(bounded integer range transfer, access/capacity refinement and alias legality), replaying the
same counterexamples as executable tests. No proof assistant integration or whole-compiler
correctness claim is implied, and this does not block workload-driven retirement.

The returned-buffer follow-up separates same-type facts from exact identity facts. Known
destination-return aliases are normalized before access analysis; producers and public return
bindings remain intact. Scope-aware substitution and alpha-renaming of incoming free identities
prevent local shadowing from redirecting a physical buffer. The composed GEMM/ReLU canary now
executes safely through two generated stages. Automatic epilogue fusion remains the next gap.

Static same-destination contraction→map now uses the existing segmented-reduction result-transform
rule. Its sole lane-owned read becomes the completed accumulator; the fused destination is write-only.
Other destination operands, neighbor reads and externally live intermediates remain excluded.
The public static 3×4×5 GEMM/ReLU case is one generated stage and device-checked on CPU OpenCL.
Dynamic composed GEMM remains two stages: its normalized extent scalar separates the equations,
and symbolic extent equivalence plus legal scalar placement are not yet proved by this rule.

Real Intel Arc checks also cover the static composed case and awkward 127×65×33 explicit
GEMM/ReLU under both precision policies. These are numerical checks, not timing claims.
The public equation-first RK4 and symbolic GEMM device checks exposed a runtime boundary:
program-wide shape bookkeeping must remain available for capacity checks, but only the graph's
declared scalar arguments enter KernelGraphCall. The binder now projects that interface without
weakening exact call validation; both device cases execute after the fix.

The next real-device model probes exposed a distinct schedule legality gap: public linear
projection dimensions retain Long, while mixed-DPAS layout/matrix stages currently bind int
extents. That candidate now declines explicitly as `:mixed-dpas-index-width-not-lowered`,
leaving generated portable contraction schedules available. Tiny forward, input-gradient and
weight-gradient executions match CPU references, and the full AD train-step compiles resident.
The existing train-step gate also executes a 2×3×2 AD/SGD update on an available GPU, checking
output extent, numerical agreement with CPU AD and a nontrivial update from the saved input.
This is not an optimized-training result. Restore that optimization with width-preserving
matrix/layout schedules or guarded specialization whose range proof covers dimensions, products
and indexing; do not narrow retained types implicitly or relax ScheduledKernelBody validation.

The first width-preserving prerequisite gives shared layout cast and transpose schedules explicit
int/long extent parameters, with independent row/column widths and unchanged int defaults.
Exact widened address arithmetic and checked hardware launch limits remain unchanged. Small
OpenCL device tests exercise every width combination and poisoned tails; non-allocating checks
cover a logical extent above int range and overflowing shape products. CUDA/HIP fixtures include
long cast and mixed-width transpose. The mixed-DPAS Long admission gate remains closed until its
matrix, split-K and batched stages also retain widths consistently; this prerequisite alone does
not restore optimized Long-dimension GEMM.

The next prerequisite retires the last legacy KernelBody `Loop` record: matrix K traversal now
uses the same typed `ForLoop`/`Yield` vocabulary as scalar/control kernels. Matrix induction and
prefetch lookahead use long arithmetic with exact widening of leaves before computation, including
split-K bounds. The matrix-plan boundary rejects narrow induction and late casts of arithmetic;
it does not erase casts as an unchecked algebraic equivalence. Intel and CUDA emission retain
the induction width. Focused Arc execution covers direct, split-K, shared-weight batched GEMM and
fused epilogues. Public matrix dimensions and Intel block-I/O coordinates still have int contracts;
pitch, output-product and hardware-coordinate bounds must be addressed before admitting retained
Long dimensions. This change is correctness/unification evidence, not a performance claim.

Contiguous leading-slice BufferViews now carry exact long leaf products in KernelBody itself.
The shared validator checks scalar types and preserves parent-shape/launch correspondence;
late casts, narrow factors and non-integral extents are rejected. Scalar and matrix emission
consume that expression without a separate matrix-only widening helper. Offset correspondence
is not a standalone overflow proof: binding must check parent-capacity products, including every
intermediate multiplication (a trailing zero must not conceal an overflowing prefix). Focused
tests cover these non-allocating limits plus the existing real-device split/batched matrix routes.

The opt-in [matrix-width canary](matrix-width-canary.md) compares the generated DPAS leaf with
the test-only historical source using poisoned output and device events. The initial small Arc
run is numerically correct but partly nonstationary; retain raw samples rather than declaring
parity. Public projection performance remains a separate open gate: matrix pitch/coordinate and
linear-address bounds must be completed before restoring the retained-Long optimized route.

Static zero-reduction-axis public contractions now enter the existing typed SegMap path.
For unresolved plain input capacities, a bounded proof over the actual lowered loads derives
minimum storage independently of output size. Every integer arithmetic prefix must fit its
retained dtype; negative accesses, indirect coordinates and unknown domains do not establish
capacities. Larger declared storage remains intact. This is not general gather bounds checking
or symbolic shape inference. Exact-size CPU OpenCL execution is correctness evidence only.

Executable scalar preconditions now use the existing checked launch/storage expression algebra,
referencing physical ABI slot names. Binding validates scalar ranges and literal specializations;
graph preflight checks every node before scratch allocation. Registry and tuning identities retain
these constraints, and forced/cached selection cannot bypass them. Focused validation: 65 tests,
374 assertions. This supplies enforcement, not proof that target requirements have been supplied.

Next: derive Intel matrix restrictions from the verified schedule and share them between selection
and artifact binding. The [Intel 2D block-I/O specification](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html)
requires at least 64-byte row widths and 64-byte base alignment; the existing tiny FP16 N/K=16
device cases do not establish legality despite passing this driver. Validate surface maxima,
pitch and view-slice alignment as well, widen linear output arithmetic, and keep the retained-Long
DPAS admission gate closed until the complete contract is enforced.

The Intel surface-contract slice derives checked bounds and pitch/fragment divisibility once for
static admission, runtime fallback and artifact binding. A/B bases require 64-byte alignment;
leading input views additionally require 32-half-element slice strides, while shared operands and
scalar C stores are not over-aligned. Projection verification rejects stripped requirements.
Intel block-I/O now admits subgroup 16 only, and C linear stores widen before multiplication.
Focused Arc direct, split-K, batched/shared-weight and fused-SSA-epilogue execution passes with
legal surface widths. Non-allocating tests cover maximum surfaces and rejected odd-height/K48
batched slices. This does not admit retained Long dimensions.

Executable matrix stores now require ScalarSSARegion. The semantic ScalarRegion descriptor remains
an input to scheduling, but cannot bypass typed lowering into KernelBody. The old source-expression
store emitter, tag reconstruction and op-map-derived call whitelist are removed. Ordered epilogue
ABI checks remain shared; an explicit rejection test prevents reintroducing unlowered store regions.
The existing typed SSA storage-address emitter serves matrix epilogues on both Intel and CUDA.
Fresh capped-REPL validation: 68 tests, 688 assertions including Arc matrix execution. This removes
a duplicate lowering path; it is not a new tensor IR or a performance claim.

Mixed-precision graph emission now derives each node's logical scalar widths from the original
GraphScalar environment through the existing checked expression algebra. Matrix, cast, transpose,
split-combine and batched stages state identity or checked Long-to-int bindings before target
projection; emitted ABIs are not rewritten afterward. Mixed-width/all-Long graph tests cover all
four orientations, direct/split execution plans, shared-weight batches and pre-allocation overflow
rejection. Focused validation: 83 tests, 1084 assertions. The public Long-DPAS admission gate remains
closed. Existing int-sized layout/combine slots still impose capacity limits; complete their wide
lowering and selector admission before promising a fallback for every oversized specialization.

The following width slice preserves graph-derived physical extents in cast, transpose and generic
split-combine bodies. The segment-count ABI follows its own shape expression, independently of
wide coordinate uses; mixed Int-output/Long-reduction cases remain valid. Allocation-free checks
cover multi-billion-element layout and combine plans. Focused validation: 53 tests, 909 assertions,
plus three Arc matrix tests with eight assertions. Public Long-DPAS admission remains closed.

Intel split-K artifacts additionally require positive, K16-aligned chunks, derived from canonical
scheduled partition bounds. Literal terminal K bounds are tied to the physical K parameter;
unknown sliced forms fail closed. This protects direct artifact binding, not only graph-generated
chunks. Focused validation: 32 tests, 745 assertions including Arc execution. It does not prove that
an arbitrary caller supplied enough partitions to cover K, nor add CUDA partition constraints.

The first guarded Long admission is unbatched dense contraction with all three dimension
expressions retained as Long. Layout and combine extents stay wide; matrix coordinates are
checked specializations, and surface selectors retain the portable fallback. Batched and mixed
widths remain declined pending their separate admission proofs. All four graph orientations
execute on Arc through direct and explicit split schedules; ordinary public Long contraction
executes direct and dynamic split schedules. Selector fallback and forced-alternative rejection
are both tested. The affected route suite passed 85 tests / 1272 assertions before adding the
extra mixed-width rejection cases; forward/input-gradient/weight-gradient and a complete tiny
AD/SGD update also passed (two tests, 12 assertions). These are correctness results, not measured
projection throughput or a claim that every oversized workload has an executable fallback.

Graph admission can now reuse direct public/node scalar ABI ranges from the existing scalar-range
facts. The batched matrix selector consumes these guards, including its private Int grid-Z count,
before surface arithmetic. This helper deliberately does not evaluate computed arguments or
replace artifact preconditions and allocation preflight. Oversized Long batches select fallback;
forced graph binding still rejects them. Focused validation: 32 tests / 741 assertions, followed
by a six-assertion graph-only API check. Public batch admission remains a separate follow-up.

One small memory-capped REPL; focused affected tests locally. Full suites and hardware-free vendor
compilers run on CircleCI. Review candidate/certificate boundaries; squash only exact reviewed heads
with all required checks green. Never alter the concurrently edited main-checkout north-star file.

### Public staged contractions and result-prefix storage

The bounded Byte/Int32/Float staged contraction now enters the retained TypedSOAC closure from
ordinary public `par/contract` syntax, including canonical derivation of operand maps. Declared
Byte inputs and Float scale/output arrays remain independent under Float compilation policy.
Frontend admission and KernelBody lowering share a non-emitting schedule-domain check; unsupported
staged semantics are not silently flattened or accepted beyond the generated schedule's coverage.
The compatibility ledger records this workload as typed and KernelBody-emitted. This does not
retire all staged fallback schedules, generalize every precision/stage combination, or prove SOTA
performance. Those remain explicit follow-ups to the common generated schedule vertical.

Storage access requirements are lower bounds, aggregated across contractions. Known larger
capacities are preserved. A map reading a smaller static prefix uses the existing indexed capture
representation, not a weakened element-tensor shape. Logical prefix results get LinkPlan views
over their physical destination allocation, with no additional copy or initializer. Emitted calls
retain the semantic result/destination relation; planning and runtime independently check actual
view identity, type, extent and prefix containment. Runtime view resolution is mandatory before
binding such calls. A partial write establishes only its proven logical prefix, not the tail.

Validation includes hardware-free CUDA/HIP public compilation, short-buffer and forged-view
rejection, retained-call mutation checks, fresh-prefix initialization, and Arc execution of
contraction alone, in-place prefix map, and disjoint-destination map with unchanged tail values.
Known capacity is supplied at compilation for these larger-buffer cases. General runtime capacity
adaptation and dynamic view specialization still need their own invocation/view contracts; exact
invocation tensor-shape checks remain intact. Indexed captures can conservatively limit fusion;
recovering pointwise fusion through explicit views belongs with the broader typed fusion work.

### Generic floating staged reductions

Static floating staged contractions now use the same retained closure and resident graph route
as packed contractions. Their arbitrary-depth reduction loops and per-stage conversions are
constructed with the shared scalar SSA builder and emitted through the common target emitters;
there is no floating-stage source template. The packed schedule remains preferred when its
legality proof applies. This ordered scalar schedule establishes a general correctness baseline,
not a claim of competitive tiling or throughput.

Frontend capability analysis constructs the same typed stage plan as lowering, without target
emission or device work. Each buffer retains one declared storage dtype; captures require
authoritative types. Explicit and implicit compound scalar conversion boundaries require retained
type evidence, preserving checked integer arithmetic before floating conversion. This stricter
contract is opt-in for the new route; older packed scalar owners still need separate migration.
Portable OpenCL rejects checked trapping arithmetic, while Intel OpenCL, CUDA and HIP provide
the corresponding target helpers. Target-neutral admission does not imply universal target support.

Validation covers three floating stages, scalar captures, cancellation and signed-zero identities,
common resident binding, public compilation/execution, all four source dialects, unsupported-domain
declines, and retained checked integer arithmetic. Broader integer stage schedules, dynamic domains,
cooperative scheduling, target capability-driven alternatives and external performance comparisons
remain follow-ups; the older staged fallback is not yet fully retired.

### Staged scalar result transforms

Floating staged contractions can now execute their scalar epilogue after the entire nested
reduction, through the same strict scalar builder. The retained closure checks accumulator scope
and excludes closed reduction indices; shared storage requirements include declared epilogue
operand maps over free axes. Explicit scalar declarations must agree with authoritative capture
types, both at schedule admission and after SSA value binding. No stage/epilogue source template
or operator registry is added.

The generated path retains the current output dtype and casts the completed scalar transform
explicitly. Destination-reading transforms still require an inout storage proof and decline before
frontend admission. Decoded operands and broader integer stage schedules remain separate work.
Public validation includes post-reduction placement on Arc, all common source dialects, short
epilogue-buffer rejection before allocation, and CUDA/HIP vendor fixtures using the same public
workload. Whole staged-emitter retirement remains contingent on the uncovered numerical contracts.

### Shared load-transform semantics

The host contraction macro and retained staged emitter now share the same capture-avoiding
per-operand load-transform substitution. The host previously ignored `:decode`, so it was not a
valid numerical oracle for decoded kernels. Decode applies to the core summand before stage lifts
and the final result transform. Summand locals are alpha-renamed before substitution so they cannot
capture decode free variables or inherit a shadowed outer array's transformation.

This correctness prerequisite does not yet admit decoded loads to the generated staged schedule.
That migration still requires retained types for the raw-load binder, decoded scalar expressions,
and any integer widening/overflow semantics; no target-side type inference is introduced here.

### Contraction lexical typing (in progress)

The source walker now supplies contraction-local type contexts for axes, decoded raw loads,
child-stage accumulators and completed-result accumulators. It uses existing inference rather
than teaching target emitters to reconstruct these types. Explicitly unknown lexical bindings
mask stale reference metadata; absent bindings still retain metadata-based rewalk evidence.

Compiler-generated scalar casts are qualified so stage linearity can distinguish genuine
identity conversions from caller-local functions. Flattening only removes a cast when its
canonical identity and child accumulator dtype prove it redundant. Decoded GPU admission is
still gated separately; this prerequisite does not retire uncovered staged fallback cases.

### Decoded loads enter the ordinary typed scalar region

The source walker now normalizes load transforms before inferring contraction arithmetic.
Source aliases and macros are resolved before the shared capture-avoiding substitution; the
result is ordinary scalar arithmetic with explicit casts, not a second decoded KernelBody path.
Typing the transformed expression from the start is essential: substituting a Double decode
into an already Float-typed product would preserve an incorrect early rounding point.

Qualified primitive casts use the existing operator descriptor for their result types in shared
inference. The public decoded staged workloads use the generated staged-scalar schedule, with
zero fallback and zero driver allocations during compilation/lowering. Validation includes
alias/macro reads, an actual-device widening-rounding differential, and the same public-source
fixtures for CUDA/HIP compilation. The compatibility ledger records the generated route.

This does not claim mixed-storage or integer-overflow coverage, whole staged-emitter retirement,
or competitive kernel throughput. Unnormalized direct typed facts remain explicitly gated;
remaining precision/storage cases and measured schedule performance are still campaign work.

### Recursive floating stages around a proved packed inner fold

The generated staged schedule composes the existing verified byte-product Int32 fold beneath
multiple Float/Double stages. Its shared packed fragment is also used by the original two-stage
schedule; packing remains four byte loads and typed word operations, with no pointer reinterpretation.
The existing packed admission proves exact-product replacement, contiguous declared maps,
four-divisible extent, zero identity and every Int32 accumulation prefix. It is not a general
integer-loop invariant or a new quantization-specific rule.

Every outer stage explicitly converts its lifted term to its declared accumulator dtype, including
identity lifts. Public three-stage compilation and resident Arc execution use the generated route;
an independent reference rounds each Float stage with non-power-of-two scales. Hardware-free tests
retain mixed Float/Double outer dtypes and reject unproved integer, layout and lexical cases.
The public workload is included in vendor compile fixtures and the compatibility ledger.

Nonpacked integral folds, unproved overflow, decoded byte products and arbitrary mixed storage
remain separate work. This closes a recursive composition gap, not whole staged source-emitter
retirement or a throughput claim; vectorized packing and cooperative schedules still need measurement.

### Independently checked additive carry ranges

KernelBody validation can derive a static, single integral carry updated by `carry + term`.
Only carry-independent scalar compute/load prefixes participate. The verifier derives term ranges
through existing SSA validation, proves every prefix fits, and distinguishes body-entry prefixes
(N−1 updates) from result prefixes (N updates). Ordinary body validation is replayed with that
evidence; overflow policy is never changed by the proof.

External range provenance is deliberately discarded during this optional analysis, and
IndexCompute prefixes are excluded: mathematical index intervals alone do not prove intermediate
machine-width safety. If this weaker analysis fails, ordinary validation with original facts still
runs. Nested/control-flow/multi-carry recurrences retain conservative ranges. Tests cover fitting
and overflowing prefixes, nonzero initialization, zero-trip preservation, carry dependence,
post-loop facts, unsafe index-derived terms and safe fallback after speculative failure.

This is a verifier prerequisite, not yet broader integer-stage admission or general index-arithmetic
safety. Nonpacked/decoded byte reductions still need their source-to-schedule validation and device
comparisons before a migrated source emitter can be retired.

### Nonpacked integral stages through the shared verifier

Static Int/Long inner folds now use the recursive staged scalar schedule when the shared
KernelBody verifier proves every accumulation prefix fits. Four-byte packing is an optional
schedule, not the semantic admission boundary: widths 1, 3 and 5 and normalized byte decodes
are covered. Admission validates a normalized body specification through the same checks used
when materializing KernelBody, without allocating a record or invoking a target emitter.

Strict typed scalar lowering preserves retained integral operation widths before stage casts.
Checked or wrapping arithmetic retains a mathematical range only when the entire range fits its
machine dtype; this does not rewrite overflow policy. A public Long multiply/divide decode tests
the distinction from premature Int arithmetic. Independent Arc references cover byte extrema and
Float rounding, and both public decoded workloads enter the CUDA/HIP/OpenCL compile fixtures.
Possible overflowing folds still decline before schedule admission. This is correctness and
coverage work, not a throughput claim or proof that all staged source fallbacks can be removed.

### Separate accumulation, result-transform and storage precision

The recursive staged schedule converts its completed result to the declared output storage dtype
instead of requiring storage to equal the outer accumulator dtype. The shared conversion policy
rejects unsupported floating-to-integral conversions and unrequested integral narrowing.
An epilogue first converts to its own declared dtype, including a bare identity expression, then
converts to storage dtype. Neither conversion changes the preceding accumulator types or rounding.

Independent Arc cases distinguish Float accumulation then Double storage (0 rather than 1) and
Double accumulation then a Float identity epilogue then Double storage (16777216 rather than
16777217). Both are public vendor compile fixtures. Single-stage closure admission and the legacy
quant router's implicit accumulator semantics remain separate retirement work.

### Explicit single-stage contractions share the same closure

A nonempty stage list now admits its one-stage base case through the same lexical, storage,
legality and shared KernelBody proofs as deeper nests. No special kernel or quantization rule is
introduced. Public decoded width-three byte products use a proved Int fold followed by Float
storage; exact width-four products use the existing packed fragment through that same recursive
base case. Both have independent Arc references and public vendor compile fixtures. Extent
mismatch and seeded stages remain errors; unproved Int prefix bounds remain capability declines.
This admits explicit single-stage source contracts; the legacy router's implicit stage synthesis
and layout-retargeting still need normalization before the old staged emitter can be removed.

### Explicit accumulator policy enters the shared stage path

For explicit `:acc-dtype`, a single-axis additive reduction with a proved zero identity and
nonempty output axes can expose its canonical accumulator as a one-stage schedule. The helper
reads dtype, combine and neutral from the existing ProductReduction view; it does not infer
accumulation precision from Byte storage. Output dtype is retained independently. Integral zero
is spelled at the accumulator width after stage legality proves the identity is zero.

Source epilogue typing also retains `:acc-dtype` before output conversion. Public Arc cases cover
decoded Byte/Int accumulation and a Double accumulator with Float storage and a rounding-sensitive
epilogue. Both enter vendor compile fixtures. Rank-zero, multiple reduced axes, non-additive folds
and undeclared accumulator precision retain their existing paths. Implicit legacy Byte policy still
requires a numeric equivalence proof; this is not a blanket Byte-to-Int rewrite or emitter retirement.

### Input storage does not choose accumulation precision

Single-axis sums with input storage differing from the output can use the same canonical-stage
projection without changing accumulator dtype. The frontend selects it for mixed storage or
explicit accumulator policy; homogeneous contractions keep their existing schedule selection.
This still requires one common declared dtype among core input arrays, not arbitrary heterogeneous
operand storage. Existing rank/domain, layout, capture and numerical legality gates remain.

For an innermost floating stage over integral storage, untyped compound roots now decline rather
than borrowing arithmetic width from the buffer dtype. Retained source types govern operations;
typed loads/casts use their existing scalar contracts. Tests inspect Long byte-product arithmetic
separately from Float scale arithmetic. Public Arc execution of 1024 products of -128 by -128,
then two products of 1 by 1, gives ordered Float result16777216 rather than Int-then-Float16777218.
The public workload enters vendor compile fixtures. This is correctness/generalization work;
packed replacement of a floating fold still requires an independent equivalence proof.

### CUDA/HIP baseline findings integrated into the existing campaign (2026-09-12)

The reference review used local JAX/Pallas, Triton, CUTLASS `147295a3`, Composable Kernel
`704ff575`, and rocWMMA `97562d3`. It does not introduce a separate IR-redesign campaign.
Raster already schedules workgroup reduction/scan trees and typed matrix store epilogues.
The next portability increments belong to the current emitter-retirement and measurement work:

1. Finish indexed subgroup KernelBody emission and retain its numerical/dispatch tests.
2. Lower existing coordinate-free FP32 TileStore regions on CUDA WMMA, initially uniform
   scalar transforms such as scaling and ReLU. Logical row/column indices and tensor operands
   must decline: WMMA fragment slot mappings are opaque.
3. Add a direct HIP matrix row from the existing instruction/body contract when its selected
   instruction and wave geometry can be compiled and validated. rocWMMA is a reference for
   fragment load/MMA/store, not a mandatory semantic dependency.
4. Admit staged matrix mainloops and indexed epilogues only with schedule-visible storage,
   coordinate ownership, barriers, and resource accounting. CuTe's mapped output partitions
   and rocWMMA's double-buffered LDS GEMM are the concrete references.

Each row needs public compiler-path acceptance before being called a completed vertical;
target-only compile fixtures are prerequisites. Cold performance comparisons stay outside the
laptop hot loop. Resident training/model validation, distributed simulation and execution,
durable manifests, and numerical PDE/AMR acceptance retain their existing completion gates.
Fused GEMM plus auxiliary reductions (CK's mean/mean-square example), warp specialization,
and TMA remain workload-driven follow-ups rather than vocabulary added in anticipation.

### Shared fragment-emitter prerequisite

The CUDA direct-fragment mainloop spelling is separated from target admission. Ordered ABI,
instruction legality, aligned dimensions, view/slice declines and uniform store-region lowering
remain checked by the CUDA entry point. The internal renderer consumes the analyzed body plan;
it does not introduce a second schedule, an allocation, or source-level type inference. CUDA is
its only admitted dialect in this slice. This prepares HIP without copying the GEMM algorithm.

Hardware-free reference probes with rocWMMA `b5a884dc` (ROCm 5.7.1) and the laptop HIP 5.7 /
Clang 21 toolchain compiled f16×f16→f32 fragments for both gfx90a and gfx1100. Unbundling the
code objects and disassembling confirmed `v_mfma_f32_16x16x16f16` and
`v_wmma_f32_16x16x16_f16`, respectively. These are reference-library feasibility results, not
Raster-generated kernels, numerical validation or performance evidence. The newer reference
checkout requires ROCm 6.4 headers; it cannot be assumed compatible with the laptop installation.
The first Raster HIP matrix row should use the existing CDNA MFMA/wave64 witness, with a
separate architecture compile gate. RDNA WMMA/wave32 requires its own explicit capability;
the current gfx1100 scalar compile gate must not be mistaken for MFMA coverage. A pinned
header/toolchain contract, generated-body compilation, physical ABI requirements and public
route tests remain prerequisites before enabling either route.

### HIP MFMA source candidate shares CUDA's admission and fragment lowering

The direct-fragment backend now checks ordered pointer/scalar roles, matrix instruction,
aligned static dimensions, whole-K traversal and unsupported views once for CUDA and the HIP
candidate. CUDA's existing entry delegates to it. The HIP candidate consumes a verified
MFMA 16×16×16 / wave64 body and the same coordinate-free Float store region, including
scaling/ReLU. It does not copy a rocWMMA GEMM: only fragment load, multiply-accumulate and
store operations use the pinned library's target spelling. Prefetch remains a compiler hint,
not an asynchronous-copy or completion guarantee.

Mandatory HIP CI compiles that generated body for gfx90a, requires the expected MFMA in its
disassembly and rejects gfx1100 compilation. The header archive is pinned by revision and
SHA-256. The scalar RDNA3 gate stays separate. Local validation can use the same script and
archive without downloading or accessing a GPU.

This remains deliberately source-only. The common matrix-target boundary still declines HIP:
the current artifact compilation contract admits OpenCL language/extension requirements, not
external C++ header dependencies. Before production admission, represent the resolved header
and compiler/architecture identity in artifact compilation and caching, prove physical pointer
alignment/launch requirements, and preserve those through binding. Then require public-route
acceptance and AMD numerical validation. No throughput or CUDA/AMD numerical parity claim is
made from source compilation or disassembly.

### Workload-first continuation: public GEMM and targeted schedule work

The chosen next emphasis is public performance verticals (option B), with schedule/IR work only
where those workloads expose a need (targeted C). HIP production admission and PTX breadth do not
displace this work. The bounded composed-versus-explicit GEMM probe in the comparison protocol
keeps source/ABI evidence, dispatch declines, poisoned-output checks and rotating raw replay samples.
It labels host-synchronized timing explicitly and optionally uses existing descriptor-backed graph
device events. The earlier profiling blocker applied to `PreparedParallelProgram`, not this probe.
Observed replay events are retained separately from available dispatch alternatives.

The first Arc run exposed a real admission bug: the public descriptor's `:ocl` backend was excluded
from the existing mixed-DPAS schedule's `:opencl`/`:ze` check. Accepting the alias restores matrix
candidates under unchanged instruction, subgroup, precision and binding requirements. Focused
tests retain non-DPAS, CUDA/HIP and unsupported-wave declines. Before/after numerical checks pass;
both short timing runs are nonstationary, so no speedup, regression or tuning winner is asserted.

Next priorities are dynamic GEMM→activation fusion through symbolic extent equivalence and legal
scalar placement; correlating observed events with executable evidence; then the projection shape
ladder and a resident forward/VJP/update. The dynamic extent's checked multiplication cannot simply
be moved across an output write: fusion must preserve failure behavior as well as values.
The first dynamic slice handles an extent product already computed before the contraction. Shared
typed index lowering checks retained widths, and shared axis algebra establishes shape equality;
the fused result transform retains the extent as a dependency. This permits ordinary prebound
contraction→map source to reach one generated step without an explicit epilogue API. The original
post-contraction extent remains a barrier; no speculative motion or inferred range is introduced.
The same comparison probe now retains separate prebound-source identities for single-row and
multi-row projection observations. Both reach one generated XMX replay entry with exact output;
external baselines and stationary measurements remain required before a competitiveness claim.
The next measured boundary alternates activations while retaining constant weights. It validates
fresh results but reveals a separate generated A-conversion launch before each XMX contraction,
with substantial inter-kernel event-span gaps on OpenCL. A graph with one semantic contraction is
not automatically one physical launch. Investigate explicit typed tile-local representation
conversion and replay submission costs; do not relax TileLoad's dtype equality or conceal an
unrepresented conversion in a target emitter. This is a workload-driven extension of the existing
KernelBody/graph pipeline, not a new GEMM semantic API.

The same changing-activation workload also passes on Arc Level Zero. Its measured event spans
are much closer to kernel-duration sums than OpenCL's; both runs are nonstationary and retained
in the comparison protocol. This separates two investigations: runtime replay gaps and explicit
tile-input conversion. Do not attribute the gap to arithmetic or replace the current graph without
measurement. A tile-local conversion candidate must preserve RNE/IEEE semantics, masked zeros,
physical fragment layout, and public FP32 inputs; CUDA/HIP staging must be represented and charged
in the schedule, not silently invented by an emitter. Keep the existing global conversion as a
candidate because conversion/reloads repeated across output-column tiles can be more expensive.
An opt-in Arc oracle now validates the exact Intel 8r16 A-load mapping: lane `l`/component `r`
holds `[r,l]`, allowing eight FP32 loads with explicit RNE half conversion to construct `short8`
without shared staging for that shape. All 2,048 tagged-coordinate/rounding checks pass; test-only
source and reference identities are retained in the comparison protocol. This is a prerequisite,
not a production conversion route or performance admission. Next introduce a pure typed input
value region, preserve it in the structural matrix plan, lower only the proved mapping, and
test every other emitter's explicit decline before enabling a measured graph-fusion candidate.
The first contract slice adds `TileLoad.value-region` using the existing `ScalarSSARegion`, not
a new numerical dialect. It accepts only closed, pure, one-element regions, checks source/storage
and result/fragment dtypes, and runs shared SSA typing with no external initial values. Existing
loads retain nil regions and strict storage/fragment equality. Matrix targets currently reject
every nonnil input region explicitly: this contract alone does not enable conversion fusion.
Store-epilogue collection now excludes load regions. Production enablement must lower the
matching FP32 prefetch, retain Intel byte-pitch/extent constraints, and validate masked zeros.

The narrow Intel lowering now accepts an explicit lhs FP32→FP16 nearest-even/IEEE cast region
from a scheduled matrix body. Shared scalar lowering spells the conversion, while the tested
lane/component mapping packs the fragment. Warm-up and steady prefetches both use FP32 storage;
the physical contract caps K for four-byte surface widths. Row-tail loads are guarded and fill
zero. RHS transformations, views, partitioned K, and non-Intel targets remain explicit declines.
A tiny generated `[13 32 32]` Arc kernel matches all 416 outputs exactly with no temporary buffer.
This is target-lowering validation, not automatic public GEMM graph fusion or a speed claim.
Next: broaden numerical/schedule cases, add the eligible conversion→matrix graph rewrite as a
candidate preserving ABI/effects, then compare changing-activation workloads against the existing
global conversion strategy with stationary measurements and external baselines.

An additional host boundary probe exposed JVM AOT overflow debt: `compile-aot` of the prebound
canary with `m=Long/MAX_VALUE, n=2, k=0` previously returned an unchanged sentinel buffer instead
of throwing on the source long product. The bytecode emitter used `lmul`; the resident descriptor
binder evaluates scalar lets through Clojure and checked launch algebra. The focused follow-up
restores checked long add/subtract/multiply/negate/inc/dec with Math exact operations, preserving
the separate explicit unchecked emitters. Direct source-versus-bytecode boundary tests and the
public sentinel test retain the distinction. Int arithmetic, narrowing casts, and other targets
are not certified by this fix; backend-wide long arithmetic may pay overflow-check costs, so
cross-target semantic parity and performance remain separate acceptance work.
Shared-memory pipelines or indexed matrix epilogues should be introduced when those measurements
or workload requirements justify them. Existing scientific/distributed acceptance gates remain.

The input-conversion follow-up now connects that leaf to the ordinary typed contraction vertical:
the checked private-cast rewrite preserves the semantic source, scheduled graph and per-node body
certificates; eligible Intel NN/NT dispatches enumerate the candidate without changing the analytic
selector. Concrete graph dependencies and public/node ABI alias contracts govern admission before
graph-private allocation. Automatic or cached preferences may fall back only to the declared
default; explicit incompatible requests fail. Offline tuning retains source-bound alias rejection
rows without executing or timing them, excludes them from winners, and requires an applicable
default. This is alias admission, not a claim of complete scalar/alignment/target applicability.

The public prebound-extent GEMM→ReLU canary exercises lowering, schedule-based recompilation,
LinkPlan instantiation, constant B and changing A through this route. On the laptop Level Zero
device, two tiny `[13 32 32]` replays matched 832 outputs exactly and each profiled one generated
contract kernel; B conversion stayed in the one-time prologue. The non-prebound extent still
preserves its post-write checked-arithmetic barrier. These are correctness and replay-topology
results, not stationary throughput or a comparison with an external implementation. Next measure
the public route across the shape ladder against numerically matched external baselines, retain
the global-conversion alternative, and continue the resident forward/VJP/update and scientific
demonstrators. No source-form restriction should become a permanent semantic primitive merely
to make a benchmark fuse; safe scalar placement needs a proof or guarded schedule.

Training coverage now checks analytic dispatch selection rather than copying hardware pitch
predicates. The tiny Gemma LoRA forward/reverse-AD/SGD fixture uses dimensions admitting the
current matrix schedules: all 56 matrix choices select mixed precision across NN/NT/TN. On Arc,
the focused 25-update comparison passed 31 assertions; loss went from 2.281172 to 1.261994 for
FP32 and 1.262058 for mixed precision, within the existing per-step relative tolerance. This
validates the tiny block and selected preference, not a bound-route trace, stationary throughput,
or real-weight model convergence. Concrete alias admission remains a separate runtime check.

Resident binding now retains a compact admitted-executable report: strategy, precision, entry
points and admission attempt reason codes, without retaining live arguments or executable source.
The public compiled/linked `execution-info` query reads that evidence rather than running the
selector again. The alias-fallback fixture distinguishes preferred and bound strategies; the
public Arc fusion canary confirms bound fused selection separately from its measured replay.
Entry points include any prologue and must not be interpreted as replay events. Equation-first
program reporting remains explicitly unsupported; manual bindings without evidence return nil.

The training acceptance also compares the successfully replayed program's bound phase reports
against each selected matrix step, so runtime fallback cannot silently satisfy mixed coverage.
The numerical trajectory and tolerances are unchanged.

External training remains a separate acceptance item. A source audit of finetune-rstr
`9e9ba5d62f3822f056e01c37231d7eaa7c84947c` found that `finetune.train/bind-layer!`,
`forward-pass!` and `backward-pass!` still depend on the retired `gpu/bind-program!` and
`gpu/run-program!` APIs. Its released Raster dependency is 0.2.287, with a sibling override for
development. Migrate shared forward/VJP adapter state and output views onto public compiled
artifacts/LinkPlans before rerunning the two-layer chained-gradient and real-weight stack gates.
Do not restore the old binder or use the tiny Raster block as evidence those external gates pass.
The pretrained-rstr checkout inspected alongside it was `3b13ad42e7bf4c98e348cb779c28096848931ba0`;
no sibling source or dependency was changed by this audit.

2026-09-27 external-source probe, with both sibling checkouts left untouched: the locally cached
pretrained-rstr `origin/main` at `7553e2c9365548ce7593dc87d353198a0cfe475f` was archived into
temporary storage and loaded with Raster main `002abbf902f2ca8acce5bd3560e77aff3d884a51`
as its local dependency. The real descriptor generator's Gemma-3-270m-shaped B=2 layer
(`640×2048`, `:map-void`) compiled for `:ocl:0` to 29 resident `:map-void` stages and zero
compiler allocations, without weights or device allocation. This verifies source acceptance for
that exact external snapshot and the **descriptor** route only; it is not direct equation-first
lowering, numerical replay, or performance evidence. The active pretrained working branch instead
references `gqa-decode-attention-heads!`, presently found only in the dirty Raster root checkout,
so testing that pair against released Raster main fails at namespace load before compilation.
Separately, loading active `finetune.head-gpu` against this Raster main fails at
`finetune/head_gpu.clj:308` because it still refers to removed `gpu/bind-program!`. Next external
acceptance must migrate that adapter to public compiled artifacts/LinkPlans, then run its
real-weight forward/VJP/update gates; do not revive the retired binder to make the import pass.

The existing small `mse ∘ linear-nb` AD/SGD acceptance now executes through the direct equation-first
compile/lower/LinkPlan boundary. Its legacy descriptor compilation assertion remains a compatibility
check, but numerical validation uses two resident updates with CPU parity, per-replay state progress,
and an unchanged host initialization array. Direct lowering reports zero driver allocations; device
allocation starts only at LinkPlan instantiation. This replaces the old numerical descriptor run,
rather than adding another model-sized training loop. It is a reusable compiler capability check,
not external real-model training validation or a performance result.

The retired scatter invocation boundary is removed: its pipeline marker/accumulator metadata,
Level Zero-only resident convention, runtime positional binders and handwritten zero-fill source
had no remaining compiler producer or caller in the source/test/dev/benchmark inventory or the
inspected pretrained/finetune sources. Current scalar/strided scatter uses the existing typed
conflict algebra, scheduled KernelBody and ordinary executable binder. Existing tests remain:
7 typed-scatter tests (55 assertions), plus 3 retained-source/control and OpenCL device tests
(128 assertions), pass with retired vars unmapped from the REPL. In particular, collision updates
preserve nonzero destination contents rather than inheriting the old binder's implicit zero-fill.
This retires an orphan route, not scatter semantics or an independently exercised source oracle.
The retirement includes LinkPlan/ProgramStage certification: old positional descriptors are
rejected for lacking an executable interface, rather than being certified for a missing binder.
All retained descriptor effects use the executable ABI; the dead generic extraction tail and its
emitter-side scalar-type inference import are removed. The focused LinkPlan and ProgramStage
namespaces pass 9 tests/41 assertions, including rejection on both ZE and OpenCL. A follow-up must
validate internal scatter accumulator initialization end to end through ordinary typed fill IR;
do not reintroduce implicit zeroing in a special runtime convention.

## Distributed numerical acceptance checkpoint (2026-09-13)

The first numerical halo acceptance now uses unequal two-row/four-row owned partitions of a
2D heat field, each with explicit ghost rows. It interprets the existing ScheduledHalo source
and target-local rectangles into resident range copies, executes a composed `heat-rhs-2d!` plus
`numeric/axpy!` update through equation-first LinkPlans, and compares four time steps with the
unpartitioned calculation. A no-exchange negative control diverges. Both workers are emulated
on the same laptop GPU/session; this is not multi-host execution, transport overlap, or a bandwidth
benchmark. Intermediate state and halo exchange stay resident; only final numerical verification
downloads results. A hardware-free compile gate covers the composed update independently of
device availability. The composition exposed and fixed ANF result-type loss: existing typed
initializer metadata now survives onto new bindings, including structurally equal rewrites.
No function/type inference registry or census exemption was added.

Local checkpoint continuation is now exercised with actual files and scoped mmap leases. A
certified manifest carries an SHA-256 address and explicit little-endian raw-f64 format; reopening
checks payload length and digest before consuming bytes, and a corruption negative control fails.
The device acceptance advances two compiled steps, downloads directly into the mapped file, closes
the original session and mapping, opens a read-only lease, uploads directly into a fresh session,
closes the lease after synchronous transfer, and advances two more steps to the unpartitioned
reference. Temporary payloads are 448 bytes and deleted in `finally`. This is a local-file provider
fixture, not a production storage adapter, serialized manifest migration test, crash-consistent
publication protocol, asynchronous restore, or distributed checkpoint.
A read-only mapping is not an immutable snapshot against another process writing the backing file;
a production provider must pin an immutable version or hold an appropriate storage snapshot lease.

Still required: bind padded ghost storage explicitly to the global owned-shard contract, realize
the distributed compute/transfer DAG with device-scoped allocation and ownership, then carry the
local checkpoint acceptance through distributed publication and restore. The two-worker acceptance
must not be counted as evidence those cross-entry and multi-device runtime obligations are done.

`BufferView/rectangular-subview` now provides the checked physical region operation used by the
halo acceptance. It preserves dtype, allocation and element strides, proves per-axis containment,
and handles empty regions without extending the backing range. Global/ghost coordinates must be
translated explicitly to the base-local frame; fitting an allocation's byte span is not a proof
of logical rectangle containment. The next binding contract must distinguish owned cells, local
replicas of neighbor cells, and boundary-initialized ghost cells. Merely assigning the padded
buffer to the owned shard would hide the kernel's halo reads and would not prove transfer ordering.

Items 6–8 have checked planning components, but not yet multi-device numerical execution:
`distributed-plan` simulates topology, dependencies, collective/halo schedules and analytic costs;
`numerical-state` certifies chunk coverage and durable field identity; `numerical-content` tests
scoped leases and transfer ownership using fake providers; `amr-plan` composes hierarchy, fields,
regions and routes, but its coarse/fine operator requirements are declarations rather than bodies.
The distributed certificate now retains the optional structural local LinkPlan/ExecutionPlan
contracts instead of only their IDs and operation counts. Equally sized replacements with changed
operations, scalars, event dependencies, outputs or views invalidate the certificate. Persistent
compiler values are shared, not serialized or copied; this is not a content hash or snapshot of
mutable host/device buffers. The next structural binding relates named local entries and global
shards without introducing another ABI or memory representation:

```clojure
{:device-plans
 {:gpu-0
  {:entries {:stencil {:link-plan local-plan}}
   :steps {:advance-0
           {:entry :stencil
            :bindings {:tile-in  {:value :u-now  :shard :tile-0}
                       :tile-out {:value :u-next :shard :tile-0}}}}}}}
```

`distributed-plan/compute-bindings` returns validated entries, logical bindings, actual ABI-derived
accesses and physical leaf views, plus explicit unbound analytical compute steps. Shard identities
are qualified by their global value; exact local shape and the existing AbstractValue storage
contract must match. Public inputs/state/results and unsourced constants cannot disappear from
the binding. Sourced constants and private scratch may remain local. Repeated bindings of the
same resident shard must agree on allocation identity, range and ordered field packing.
Distinct shards cannot overlap physical ranges on one device without an explicit relation;
disjoint subviews remain legal. Declared global memory-space constraints must hold for every
physical leaf. Entry access facts are derived once per report, not once per invocation.
Bound leaves cannot alias private entry storage until an alias-aware access proof exists.
Private-private storage and unused entries remain entry-scoped; only bound shard storage gains
device-scoped identity. Unknown device-local keys are rejected rather than silently ignored.

This is not yet distributed execution: device-scoped allocation sharing, transfer realization,
submission/completion and ownership still need a numerical vertical. Separately instantiated
LinkPlans currently namespace their owned allocations per executable. Matching structural views
does not change that behavior or prove cross-entry zero-copy replay. Whole-shard binding also
does not stand in for a future explicit halo-subregion mapping. Optional flat ExecutionPlans remain
analytical; arbitrary event DAGs cannot be asserted as the realization of a named LinkPlan entry.

The next sequence is workload-driven:

1. Admit the existing `raster.ode.pde/heat-rhs-2d!` through direct TypedSOAC scheduling and emission,
   preserving its multi-store boundaries and offset interior domain. This local admission now
   passes CUDA/HIP compile/link and two-grid ZE numerical parity, through counted-store
   normalization described below. Parallel schedule quality remains unmeasured. Existing 1-D
   RK4 and the direct AD/SGD probe also succeed; the PDE source was not rewritten for admission.
2. Bind distributed compute to exact local program entries and shard/view contracts; make numerical
   program drift load-bearing in certification rather than comparing only IDs and counts.
3. Execute a hardware-free two-shard halo step and compare stitched values with the monolithic
   numerical oracle, alongside the certified region/byte/cost checks.
4. Checkpoint actual addressed bytes through a local immutable file/mmap provider, close/reopen,
   restore, continue, and compare with uninterrupted execution; test branch reuse of unchanged chunks.
5. Grow that seam into conservative 2-D shallow water and typed prolongation/restriction, then
   subcycling/reflux. Existing Dirichlet heat decay tests are not mass conservation, lake-at-rest,
   convergence-rate oracles, or numerical AMR acceptance.

The first numerical evolution oracle now uses an ordinary `deftm` periodic, cell-centred heat
step with full-domain ratio-two prolongation/restriction. It checks volume-weighted mass,
resolution convergence against an analytic eigenmode, and continuation from detached fine-field
bytes against uninterrupted evolution; direct TypedSOAC lowering is checked too. This is a
whole-domain refinement baseline, not partial-patch AMR, halo execution, a durable manifest,
subcycling across levels, or reflux. Device parity is tested only when OpenCL is available.

This keeps cloud hardware, native fabrics and object storage out of the development hot loop.
It does not replace the external training acceptance or measured kernel-comparison work.

The first loop-admission prerequisite fixes an existing unsound dense-map match: a multi-form
`dotimes` body beginning with `do` could select that `do`'s last store and drop its earlier stores
and all sibling forms. Dense matching now requires one accounted-for body; unproved ordered
effects remain intact. The numerical regression checks every destination. This closes an effect
preservation bug, not the broader 2-D heat admission gap; that still needs a complete typed
iteration-domain and cross-iteration write/read proof.

The next counted-domain prerequisite aligns `map-void!` and its compiler CPU expansion with
their documented `dotimes` semantics: evaluate the bound once as Long, execute no iterations for
nonpositive counts, and return nil. Explicit user narrowing casts remain executable. The former
implicit Int coercion could not faithfully represent a source Long-count loop. This changes no
kernel ABI and does not admit GPU launches beyond a target's index limits; unsupported domains
must still fail target checks rather than narrow. Raw store-loop normalization can subsequently
use one typed Long scalar extent `max(0, long(bound))` and the existing effect-region analysis,
with sequential order retained until a dependence proof permits parallel scheduling.

The source normalization now uses that existing effect-map boundary: a closed raw `dotimes`
store region keeps every body form, performs the Long conversion at its original site, and
clamps negative counts through an ordinary typed scalar conditional. Unknown/effectful or
mutable-array-derived bounds remain outside this normalization. No new operation registry or
semantic loop dialect is introduced. An explicit returned destination observes its latest write
even when the loop itself returns nil. Effect result shapes project the physical destination's
extent, and that projection participates in ordinary SSA remapping.

The existing `heat-rhs-2d!` now compiles and links through the direct CUDA/HIP boundary without
driver allocation. A local Level Zero numerical regression compares all cells against the CPU
for 2×3 (empty interior) and 5×7 grids with nonzero initial destination contents. This establishes
local execution correctness, not a competitive parallel stencil schedule, distributed execution,
or a performance result. The next distributed seam remains exact local-program/shard binding,
followed by numerical halo execution and real-byte durable restore.

The counted-loop migration also exposes `softmax-rows!` as an end-to-end scalar-sharing
regression. Pure, destination, and offset maps now project a complete typed lexical `let`
spine into ordered TypedSOAC locals instead of embedding it in one expression for later
substitution. Missing local type evidence declines admission rather than inventing a consumer
type. This preserves each intermediate once through SegMap and KernelBody. On the existing
softmax exponential polynomial, the generated kernel dropped from 25,599 scalar computes and
3,379,887 source bytes to 24 computes and 2,259 bytes. CUDA/HIP source-size ratchets and local
Level Zero numerical parity cover this case; these are not throughput or SOTA benchmark claims.
The shared inliner likewise retains actual argument types and once-only evaluation (including
unused checked conversions), and never flattens initializer bindings into loop/recur parameters.

Coverage accounting distinguishes newly admitted sequential store work from formerly parallel
work becoming serialized. The migration's new sequential regions include host loops previously
outside typed coverage; they do not establish a competitive schedule. The serialization ratchet
remains in force, with deliberate baseline changes requiring an audit of the previous workload
route and preservation of existing independent regions.
The refreshed 236-var baseline admits 23 such regions: 19 previously scalar programs,
previously compatible `softmax-rows!`, and three partially typed programs (`maxpool2d-bwd!`,
`mean-pool`, `dense-into!`) whose old typed kernels covered initialization/copy/final scaling
while the counted loops remained host work. Their existing independent kernels are retained.
The resulting route counts are 132 typed, 47 scalar, 14 compatible, and 43 errors; the 43 errors
remain visible debt rather than being excluded from the corpus.

Rectangular axis permutations now retain derived dimensions as opaque mixed-radix factors while
exposing their product spines to the ownership algebra. `im2col-1d` therefore lowers as a proved
unique traversal rather than an ordered compatibility effect. Its reverse, `col2im-1d`, exercises
the colliding case: walked typed addition is recognized through an identity result cast, and a
branch-local address region is represented as a guarded lexical effect rather than speculated out
of its source branch. That region lowers to ordinary KernelBody `IfRegion` plus `AtomicRMW` on the
shared OpenCL/CUDA/HIP path. A local Arc comparison covers the public equation-first LinkPlan and
the exact CPU result. This is general guarded reducing-scatter support; convolution names do not
participate in admission or lowering.

Fixpoint typedness now distinguishes statement bindings from value bindings through the existing
`:raster.effect/effectful` binder contract. Normalized loops, SOACs and effectful conditionals no
longer require an RHS-head whitelist merely to cross the typedness gate; an unmarked conditional
or parallel call still requires a retained result type and fails closed. The pure-map materializer
now attaches the same effect contract to its generated write step. This only admits statements to
the next conversion boundary: whole-program conditional branches still require explicit typed
program control and may not be mistaken for a scalar expression or a `KernelDispatch` selector.

Guarded lexical effect regions now admit counted store loops. The existing `effect-when` term
remains the only conditional effect scope; its body may contain the existing `effect-loop`, and
portable KernelBody lowering preserves the nesting as `IfRegion` around `ForLoop`. This closes an
obsolete frontend rejection without adding a loop or branch dialect. Integer values are not
implicitly predicates: Clojure treats zero as truthy, unlike C-family targets, so source must use
an explicit comparison until a first-class boolean ABI/value contract is carried end to end.

One-arm source conditionals whose active branch is a removable counted store loop are predicated
into that same effect-map contract. Pure branch-local scalar bindings remain typed effect-region
locals; ownership proof, rather than the rewrite, decides whether the traversal is independent.
A single walker-typed scalar reduction nested under a removable scalar expression is first exposed
as SSA, then handled by the existing reduction-to-scalar fusion rule. Thus gradient clipping lowers
`sqrt(reduce(...))` as a typed reduction result-transform followed by a guarded effect-map, without
a host scalar round trip or an optimizer-specific rule. Resident realization rewrites only actual
consumers of that rank-zero value, preserves effect-map destinations, and carries it physically as
a one-element device buffer. Full resident compilation emits two scheduled GPU stages and no
fallback; this is functional coverage, not a throughput claim.

Resident realization is now the default for non-escaping reduction scalars on every GPU compiler
entry, including diagnostic/staging compilation. Escaping loss values remain host-visible. This
removes the former policy split in which the resident-program API compiled RMSNorm, softmax
backward, and gradient clipping while `show-pipeline` reported contradictory scalar/buffer ABI
roles for the same typed programs.

Ordinary scalar reductions now delegate their complete element expression to the same strict
typed-SSA lowerer as maps, scans, stencils, contractions, and ordered fold-maps. The reduction
adapter retains only its algebra, coordinate, storage, and currently executable overflow gates;
it no longer reparses literals, loads, casts, and arithmetic through a second recursive lowerer.
Authoritative lexical binder types survive the shared pure-let normalization when an initializer
has no result stamp, without overwriting an initializer's own type. Consequently mixed-precision
value conditionals such as public Huber loss lower to KernelBody `IfRegion` plus explicit branch
conversions on OpenCL, CUDA, and HIP, while untyped compound arithmetic and unchecked coordinate
claims still decline.

The shared scalar language now normalizes Clojure's unary numeric predicates to typed comparison
SSA and expands `signum` into comparisons and selects. The expansion preserves signed zero and
returns NaN unchanged instead of relying on target builtins with differing edge semantics.
Compile-time Clojure truthiness is resolved before KernelBody boolean control, including the
literal `:else` arm produced by `cond`; runtime integers are still never treated as predicates.
Public Huber and L1 gradients therefore use the same KernelBody route on OpenCL, CUDA, and HIP
without source fallback or target-specific intrinsic spelling.

Effectful source loops now acquire the fixpoint statement contract only when they have no terminal
value and their enclosing binder is unused; a consumed nil or a value-carrying loop remains subject
to typed-value validation. TypedSOAC composes such an effect-only counted store loop with its later
lane-local continuation through the existing ordered effect-region algebra. This moves the public
GQA decode loop from an early untyped-fixpoint error to the explicit
`sequential-effect-continuation` production boundary. Crossing that boundary still requires a
read/write ownership proof for scratch written by the loop and read by later scalar folds; the
compiler does not claim parallel safety merely because the source came from an attention workload.

That ownership boundary now admits multiple writable destinations when every destination's complete
read/write set has one proved injective outer-item slice. Read-only tensor captures are irrelevant
unless they flow into an address, where the index algebra still declines data-dependent forms.
Dependency closure through typed locals determines which nested sequential loop digits actually
participate in each address; an address may be invariant in a surrounding inner loop without losing
outer-item ownership. Scalar loops embedded directly in effect-store values are canonicalized to
`Fold` before this proof, just like local initializers and carries. KernelGraph binding remains the
physical precondition and rejects overlapping writable views. With these pieces, public GQA decode
proves independent head slices over scratch and output and emits one ordinary KernelBody kernel;
the proof is general mixed-radix ownership, not attention recognition.

## Fresh-storage initialization through the typed vertical

The analogous OpenCL/Level Zero `invoke-registered-reduce-by-key-kernel` shortcuts are also
retired after an exact-symbol and dynamic-dispatch reachability audit. Production reduce-by-key
already enters the typed reducing-scatter conflict contract, scheduled SegMap/KernelBody, and
generic map-void invocation; the public combinator and CPU semantics are unchanged. The test-only
handwritten OpenCL source generator is retired separately: it accepted arbitrary operators but
always emitted addition. Its source assertions now exercise the existing public typed dispatch
test, which also asserts KernelBody emission and no longer skips on namespace-loading failures.
The retained Vulkan scaffold is unchanged; its analogous operator-validation gap must be closed
through the typed conflict contract before any future production activation.

Resident allocation does not itself implement Clojure's fresh-array zero semantics, and allocating
once is not enough for repeated execution. The frontend now retains allocation initialization,
element type, extent, and source order in TypedSOAC facts using the shared allocator descriptor.
GPU scheduling materializes required zeros as ordinary typed maps after fusion and before
ownership certification. Native fresh-array execution keeps its native initialization provider.
Explicit compound allocation lengths use the same scalar SSA normalization as launch extents,
at the original allocation site; the public invocation plan therefore sees a scalar dimension,
not a reconstructed arithmetic expression. Value remapping preserves the allocation contracts.

Overwrite elision uses complete logical result shapes for functional maps, stencils, contractions,
segmented/product reductions, segmented fold-maps, and scans, with no aliased destination read.
The shared extent proof follows retained integral scalar SSA and checked Long products, allowing
equal dimensions with reordered factors without erasing narrowing, floating, or wrapping
arithmetic. Allocation size must equal result volume: padding still requires initialization.
Coverage witnesses are consumed in equation order: only incoming scalars are initially available,
and scalar definitions extend the proof environment after execution. Allocation extent definitions
must precede their allocation; a later definition cannot justify an earlier fill or elision.
Write-only permission alone proves nothing for conditional effect stores or sparse updates.
Allocation cardinality is separate from a destination AbstractValue's logical consumer shape.
For example, an AD gradient may be allocated with `in*out` elements and later consumed over
`alength(weights)`. These remain independent symbolic identities: the initializer covers the
constructor extent, while concrete invocation/LinkPlan validation checks logical demand against
capacity. Known undersized allocations fail statically; neither symbolic equality nor extra
capacity is a license to elide a fill. Plain representation and absent logical layout remain
mandatory for this initialization route.
Unsupported extent, dtype, placement, alias, and observable-host-use contracts fail closed.
General effect-map dense-image proofs remain work; this does not change the matrix selector.
Storage first consumed by retained host bindings keeps the native allocation provider and host
writes; GPU staging uploads that state. Such host buffer accesses still disqualify straight-line
resident extraction. Observations between fused constituents cannot use this exemption.
The native-provider obligation is retained as semantic data and survives value remapping.
Source-independent SOAC promotion and invocation linking reject it until an explicit content
provider exists; an ABI `:write` role cannot silently discharge that obligation.

Validation covers ordinary and strided public scatter, holes, collisions, changed inputs, and
two replays on Arc; both stages are ordinary KernelBody kernels. CUDA/HIP checks cover public
source compilation **and allocation-free LinkPlan lowering**, not emission alone. Focused pure
checks cover full/partial overwrite, copy/unspecified storage, malformed facts, aliases, source
placement, generated names, and value remapping. Existing numerical tests are retained.

Compiler-generated scalar matrix and split-K combine algorithms now enter through semantic
contraction facts, without building and reparsing surface contraction forms. The typed-route
source-reparse guard remains intact; this is independent of initialization scheduling.

Invocation linking also consumes the retained typed functional write domain. A graph output's
`:write` role is insufficient to discard zero initialization: the domain must exactly cover
concrete materialized capacity, have plain layout, and have no graph input sharing its physical
storage token. The current proof admits single-equation algorithms only; multi-equation graphs
need an ordered first-touch proof. Unavailable dimensions and unsupported shape expressions
decline elision, while invalid negative dimensions still fail. This shares the functional-domain
classification with initialization scheduling, rather than introducing a backend operation list.

### 2026-09-14 — ordered product-valued scalar folds

The typed scalar language now retains exact counted recurrences with two or more carries as one
product-valued `Fold`. The source matcher preserves binding order, origins, inclusive/exclusive
bounds and the shared lexical update region without asserting reassociation. Static `nth`
projections are scalar SSA uses of that product region: KernelBody lowering memoizes the region and
emits one multi-carry `ForLoop`, while JVM projection binds the returned tuple once. Focused tests
cover frontend validation, one-loop OpenCL/CUDA/HIP emission, JVM numerical execution, and rejection
of changed exits and non-unit induction. This is general loop-language coverage, not a layer-norm
operation rule; algebra certification and parallel schedules remain separate follow-ups.

### 2026-09-20 — certified segmented fold-map association

The general per-segment fold-then-dense-map algebra now carries association per fold. Source must
explicitly request implementation-defined association; the frontend derives a typed monoid
certificate and the validator checks the serialized certificate against the retained scalar
region. Host projection keeps the request but preserves sequential interpretation. Ordered folds
and ordinary loop recurrences are unchanged. The next schedule may therefore assign a workgroup to
one segment and parallelize only certified reduction axes without recognizing normalization or
another library operation.

The first such schedule is now implemented for a single certified additive or multiplicative
fold. One workgroup owns a segment, lanes traverse the fold axis with a strided private partial,
and a workgroup-memory tree combines those partials before lanes traverse the dense result map.
KernelBody owns the allocation, barriers, scalar SSA and launch geometry, so the identical body
emits through OpenCL, CUDA and HIP. Unsupported fold shapes decline rather than silently becoming
cooperative. The full compiler uses the generic executable ABI for this schedule: unlike the old
map marker, it does not invent one distinguished element-count scalar and therefore preserves the
compiler-selected one-group-per-segment `LaunchSpec`. A public Intel OpenCL device test covers
non-power-of-two and greater-than-workgroup extents; nvcc and hipcc compile the same emitted body in
the hardware-free fixture gate. RMSNorm and quantized dots remain workload acceptance steps, not
special cases in this lowering.

The first workload acceptance expresses float RMSNorm itself as the same general SegFoldMap. The
multi-row operation and its one-row compatibility entry both compile to one cooperative executable
with no compiler allocation; host interpretation remains ordered. Intel Arc device-event
measurement at one row and width 640 observed a 169.8 microsecond ordered median versus a 3.1
microsecond cooperative median. The run was non-stationary and is recorded as directional local
evidence, not a portable throughput claim. Correctness covers widths 1, 17 and 513 plus a three-row
case, and the actual library kernel is part of the nvcc/hipcc fixture corpus. This validates the
algebra/schedule seam on a real decode bottleneck without introducing an RMSNorm compiler rule.

### 2026-09-25 — city language-coverage follow-up (not a current upgrade gate)

The city-rstr GPU day kernels now compile and agree with their JVM counterparts on Raster
0.2.951, including the multi-carry effectful episode loop. On current Raster main at #907,
the external P10 two-exit search and P14 multiple-`recur` probes also compile on Level Zero
and match the JVM exactly; Raster's city fixture independently covers a two-exit search and
branched effectful recurrences. P20f, which nests a seeded SplitMix draw inside a multi-carry
effectful episode walk, still declines with `:no-lowering-rule`. Direct source-region probing
isolated its `unchecked-add-int` induction against a `Long` bound: replacing only that step
with widened addition makes the whole region recognizable, while the other unchecked SplitMix
arithmetic stays unchanged. This is a legality gap, not evidence that SplitMix cannot be emitted;
accepting the int step without a bound proof would risk wraparound. Its full shape now has a
device parity-or-explicit-decline regression. This is not an upgrade gate for city, which
uses supported spellings, nor proof that every P13 variant is admitted. When revisiting the
remaining gap, retain a typed control-flow region and its effect/SSA ownership facts, then
check GPU/JVM results for the full city kernels. Do not reintroduce source-level fallback or
interpret a successful compile as numerical validation.

### 2026-09-25 — runtime-shaped cooperative register tiling

The public typed contraction keeps symbolic `m/n/k` through one semantic ABI while the
register-tiled KernelBody now consumes those scalar bounds directly. Its tile-local loads,
stores, barriers and accumulation are still compiler-generated and target-neutral; there is no
new GEMM source template. The router preserves source `int`/`long` scalar dtypes, widens local
K offsets explicitly when necessary, derives the output count as checked launch IR, and retains
the portable reduction as another executable alternative. Candidate graph composition now
matches logical buffer identities independent of physical A/B argument order. A flat read of a
2-D destination declines the tiled leaf rather than constructing an invalid rank-one load.
The selector sends products beyond int-sized resident capacity to the portable path, whose own
binding still checks capacity, so a Long-valued extent cannot silently overflow tiled address
arithmetic. General symbolic bound expressions remain a documented decline.

The opt-in public Arc oracle passes batch-one, multi-row, square and awkward-tail shapes, and
the generated symbolic body compiles to CUDA PTX and HIP syntax without hardware. The square
shape's local device-event median moved from approximately 393 to 112 microseconds, with
non-stationary early replays; see the raw comparison protocol for the measurement limits.

### 2026-09-29 — whole-plan replay measurement boundary

The opt-in production canary now composes three independently lowered typed effect maps through
the public `Compiled/compose` API and checks one recorded graph, three profiled kernels, and an
independent numerical reference. It reports component compilation, composition, instantiation,
and warm resident device-event replay separately. Ordinary CI tests only the canary contract;
no timing threshold enters the hot loop. One local Arc OpenCL run at width 1024 was numerically
exact, but replay timing was nonstationary (median 93 microseconds, CV 1.34); it cannot establish
a performance improvement. This is a mechanism canary, not a pretrained decoder substitute.

Next, run the same phase accounting on pretrained-rstr's resident decode and chunked prefill,
recording kernel/event attribution and transfer counts without changing its cache policy. The
committed GPU corpus reports 238 typed SOAC, one typed structured-control, six explicit host-only,
and six scalar routes out of 251 functions; it has no compatibility route. That is a coverage
ratchet for the selected corpus, not proof that the compatibility pipeline is dead for external
programs such as city-rstr. Remove compatibility lowering only after a production-path reachability
audit and explicit fail-loud replacements for its remaining admitted forms.

### 2026-10-03 — durable producer/storage authority seam

Items 1–8 remain the campaign, not just the currently active storage slice. Recent producer work
retains original compiler-owned Prepared/Compiled identity, tracks synchronous replay completion,
pins byte receipts and hashes actual resident initialization/output/post-state rather than caller
defaults. Verified raw-array decoding uses explicit dtype/byte order and bounded staging. Provider
availability finalization closes verified local leases and awaits matching durable placements
before invoking external metadata publication; provider/store realization is still separate.

Generated per-dtype KernelBody probes now establish measured storage order on local OpenCL and
Level Zero; all six sources also pass the exact-head CUDA/HIP compile gates (#1001). Owner-bound
measurement shares the artifact seal, lifetime guard, emitter and binder rather than adding a
cache/session convention. A live completed receipt can join only original same-owner/session/dtype
facts, then expose addressed physical leaf bytes with explicit raw-array storage. This is not
logical quantization inference, universal arithmetic/codec equivalence or a durable state commit.

Pivotal review caught loose byte coercion, dishonest optional-capability coverage, repeated ZE
registry-kernel leakage and source-only/explicit-SPIR-V cache aliasing. These have independent
regressions; focused storage/owner/registry/native/CI-ratchet suites pass 36 tests / 767 assertions.
Both local native backends execute actual retained compiler programs, not only emitted-source
tests. No timing claim is made under varying load and power-save settings.

Next acceptance order within item 8:

1. Make shared native graph/prepared destruction report failures and retain unresolved resources
   honestly. Current older destructors swallow some native errors; surfaced API cleanup tests do
   not prove native reclamation. Do not invent a separate checkpoint-only lifetime convention.
2. Bind semantic field/coordinate selection to the addressed producer leaves, stage/verify bounded
   immutable chunks and construct producer-derived manifests before real provider publication.
3. Extend scalable hierarchy/patch packing and temporal interpolation/subcycling/reflux validation.

Canonical external training/package migration, performance/selection evidence, real fabric and
multi-device collectives and production storage/AMR acceptance remain open as listed above. The
local evidence and compile-only vendor gates do not establish those broader end states.

#### Shared native teardown consolidation: acceptance slices

The resource-cleanup mechanism is owner-local runtime state, not compiler IR or a second authority
registry. Ordered destruction dependencies retain failed and blocked resources; successful releases
are removed exactly once. A thrown error defaults to an indeterminate outcome, so another close
reports it without repeating the native call. Only failures explicitly classified retry-safe by
the internal callback contract retry; this ordinary error marker is not authority or evidence.
No native OpenCL/Level Zero error is classified as retry-safe merely because it is nonzero.

Land this consolidation in dependency order, keeping items 1–8 above intact:

1. Validate the small owner-local mechanism with dependency, suppression, concurrency, recursive
   teardown and indeterminate-outcome tests. This alone makes no native reclamation claim.
2. Capture ownership immediately after each backend acquisition, including partial constructors;
   check native release statuses. Retain event arenas until all dependent events are released.
3. Integrate KernelGraph lifecycle under the session lock. Handle identity must include the session
   and generation, avoiding cross-session/key-replacement ABA. Retain provisional/retired owners on
   failed construction or replacement; drain events first; remove ownership only after success.
4. Extend the same ownership to descriptor prepared/recorded graphs and their exact dependencies,
   then root buffers, kernel arenas and LinkedExecutable/parallel-program close. Do not mark a
   wrapper closed before failed teardown resources have an honest retained disposition.

Pivotal review identified constructor rollback, stale graph handles and early wrapper-close flags
as correctness requirements, not optional cleanup. Native dependency ordering is graph recordings
before prepared kernels, kernels before private views/buffers, profiling events before pools and
pointer arenas, and session children before root storage. Independent siblings may continue after
failure only when that independence is actually established. Use of the owner must share the
surrounding session lock with release; a standalone live-status check does not pin resources.

Regression acceptance includes every acquisition/release fault, suppressed cleanup errors retaining
the primary failure, old graphs surviving failed new construction, stale-handle rejection, no double
release, and failed dependent destruction preventing buffer free. Performance acceptance remains
separate from correctness under laptop power-save/background-load conditions.

#### Native KernelCall acquisition and common graph ownership

Both resident backends now reserve a cleanup owner before creating each production KernelCall's
dedicated native kernel, then use checked native destruction. Argument-binding failures roll back
immediately; if destruction is unresolved, an explicit synchronous adoption callback transfers the
original owner into the surrounding session. Suppressed errors are diagnostics, never the lifetime
channel. The raw one-argument backend API preserves unresolved ownership in ExceptionInfo data and
keeps the original failure as its cause; it cannot promise identical top-level exception identity.

Common KernelGraph entries use the shared dependency plan for recordings, individual prepared
kernels, views and temporary buffers. Failed construction retains the same resident root views and
footprints, preventing root release. Successful cleanup is retired incrementally; session close
attempts all independent siblings in a layer, then stops before dependent roots/arena on failure.
Its sticky :releasing lifecycle marks :closed? true to reject further use, without pretending native
disposal succeeded. Repeated close remains available for retained cleanup. Modern bind/run sequences
are serialized by the session lock. Graph handles now carry session and generation; a stale or foreign
handle cannot drain or destroy a replacement's resources. A failed old-generation destruction keeps
that non-runnable owner, and does not publish the newly constructed replacement.

The migrated KernelGraph swallowing destructor is removed; a missing owner fails loudly. Backend
production prepared maps likewise cannot fall through the compatibility destructor if their owner is
missing. A duplicate shadowed ZE fresh-kernel factory is removed. Numeric surface semantics and IR
are unchanged; handle identity, close lifecycle and raw-backend exception handling become stricter.

#### Level Zero recording acquisition ownership

Regular Level Zero recordings reserve one cleanup plan before creating the queue, command list,
profiling pool or timestamp events. A create/readback exception is indeterminate, not proof that
no native allocation occurred: it retains the acquisition slot without guessing a handle to free.
Successful acquisitions are private callback captures, not mutable graph-map fields. Command-list
release precedes event release; all events precede pool release. An unsubmitted constructor's queue
is independent and is still attempted after another resource fails. Never-created dependent slots
may remain blocked plan entries; this does not assert those resources were allocated.

Recording rollback uses the same explicit adoption sink as KernelCall construction. Prepared kernels
also depend on adopted construction debts, preventing destruction while an unresolved recording may
still reference them. Fault tests cover every create/append/close point, independent destruction,
unknown-outcome exact-once behavior, and an append failure combined with failed list destruction.
Live profiled/unprofiled graph replay remains numerically checked on both local backends. Shadowed
older Level Zero record/replay/destroy definitions are removed. Direct backend users must establish
submission completion before destroying a recording; common session teardown drains tracked events.

The Level Zero slice alone does not close OpenCL per-replay ownership, descriptor/root-owner
consolidation or establish performance under varying laptop power/background load.

#### OpenCL recording and per-replay ownership

OpenCL recordings reserve their profiling queue and one retained submission dependency. Each replay,
ordinary or profiled, installs a nested owner before enqueue. Its dependency plan proves queue
completion before event release, and releases every event before closing the pointer arena. A
successful wait/query records completion so subsequent cleanup does not add an unnecessary finish.
If enqueue/flush fails before a public completion token exists, the recording still owns the partial
submission. An unknown drain blocks every dependent resource. Independent event releases may proceed
after successful drain; failed event release retains its arena and profiling queue, and the common
recording dependency keeps prepared kernels and resident roots alive. No unknown native operation is
automatically retried.

Timestamp reads validate the nested owner as well as the recording: a retained vector must not
permit queries on events already released by a partially failed cleanup. Nonempty graph tokens
cannot fall back to legacy release if their owner is missing. Transfers and empty-completion tokens
retain their existing contracts. Timestamp/await errors stay primary when cleanup also fails.
Raw backend callers still must serialize graph use/destruction; common sessions use their lock.

Hardware-free production fault tests cover partial enqueue, flush, unknown drain, queue creation,
independent event release, retained arenas, missing-owner tokens and timestamp access after partial
destruction. A common-session regression proves failed submission without a public event cannot free
the graph's prepared kernels or roots. Live profiled/unprofiled replay and storage/transfer oracles
remain separate checks. Ownership adds replay bookkeeping; performance acceptance remains pending
stable power/load measurements, not inferred from these correctness tests.

This is not complete native reclamation. Legacy descriptor bindings, backend view/event
partial constructors and destruction, asynchronous-event failure cleanup, root-buffer/arena failure
retention, and linked-wrapper close remain acceptance work. In particular native graph destructors
still contain swallowed-error debt, and general legacy concurrent operations are not certified by
the modern graph locking tests. No native error is automatically classified retry-safe.

#### Recorded wrappers and borrowed source retention

The common recorded wrapper reserves prologue and replay ownership before construction, adopts
unresolved backend acquisition debt synchronously, and publishes replacement recordings only after
old-generation teardown succeeds. Failed teardown keeps the registration non-live; independent
successful releases are not repeated. Failed provisional construction retains a private registration
when rollback cannot finish, preserving the exact primary exception and its borrowed sources.

Prepared phases and emitted graph handles cannot be released or rebound while any recorded wrapper
borrows them. Recording, replay, profiling and complete measurement serialize under the session
lock. The context-chain convenience API releases its previous recording before rebinding phases.
Attached cleanup stops before source teardown if recording teardown fails, and before root teardown
if any prepared source fails.

Surface lifecycle tightening: `free-buffer!` rejects releases overlapping a recorded wrapper's
retained source footprint, including failed provisional wrappers. Low-level preparation captures
resolved root objects and their registrations; descriptor materialization captures every resolved
leaf root, including views and composite arguments. A wrapper unions footprints only when every
source has one. Legacy/manual bindings without a footprint retain the conservative **all-buffer**
guard; missing is never interpreted as empty. Owned-root release checks keys, allocation identities
and exact resident-buffer object identity, including aliases registered after binding. Distinct
wrappers of the same native pointer require a shared allocation identity; no arbitrary pointer
equivalence is inferred. Detaching a new
borrowed/external alias checks its registration key because it cannot destroy the native root.
Unrelated staging buffers may therefore be released without special-case key exemptions.
Explicitly release recordings before freeing their buffers or replacing bindings; this does
not change numerical surface semantics. Backend prepared factories, sticky linked/parallel close,
and root/view/event owners remain unfinished; these wrapper checks do not certify their reclamation.

Fault oracles cover multiple recordings borrowing one source, unknown teardown with successful
independent sibling release, adopted constructor debt, failed replacement, and failed prologue replay.
These are lifecycle/correctness checks, not performance or distributed-ownership evidence.

#### Raw layout-binder retirement

The runtime-only convert/transpose binders and their separate Level Zero module/kernel caches
have no production callers. Their three native test consumers now construct target-neutral layout
bodies, project checked ScheduledKernelBody artifacts through the common target emitter, and bind
them through public KernelCall/session APIs. Byte-granularity transpose and all 36 nearest-even
FP16 conversion cases (ragged tails and extents below the unroll width) remain numerical oracles;
the int8 contraction route still checks its inserted transpose against the CPU reference.

The raw binders and OpenCL rejection stubs are removed rather than assigned another ownership
system. Tests release their modern graph bindings; migrated transpose/conversion storage follows
the session lifetime. This removes a duplicate acquisition/cache path, not an emitter capability.
Other legacy routing-test allocation/launch helpers and the raw map preparation binder remain
separate consolidation work. No performance or cross-vendor execution claim follows from this
Intel-native migration; hardware-free target compilation remains its own gate.

#### Plain prepared KernelCall ownership

The public `prepare!` convenience API now projects its split pointer/scalar/bound arguments into
the checked artifact's positional KernelCall ABI and full 1–3D launch geometry. It no longer uses
the raw map preparation binders, which are removed from both runtimes. Argument names are not
assumed unique: distinct physical positions survive projection, and scalar conversion uses the
shared runtime scalar contract. This tightens admission to checked artifacts without changing
the high-level argument convention or numerical surface semantics.

Plain prepared bindings require a live cleanup owner before invocation or recording. Replacement
builds the candidate before releasing the old generation, publishes only after successful release,
and retains failed old registrations as non-live. Failed construction adopts unresolved debt into
a private prepared registration, preserving the primary exception and pinning its resident roots.
Unrecorded prepared bindings now pin their captured root footprint too. Invocation serializes with
release under the session lock. Level Zero async KernelCall teardown drains the exact captured
command list before destroying its kernel; this is not a new completion-event API.

This slice does **not** certify all reclamation: legacy multi-child BoundExecutableStep teardown,
backend pre-reserved acquisition, modern graph alias guards, root/view/event ownership and sticky
linked/parallel close remain explicit follow-ups. Unknown cleanup outcomes are never retried just
because the public entry remains registered. Fault tests cover missing owners, positional duplicate
names, 3D launch projection, failed acquisition debt, replacement failure and launch/release ordering.
Quantized replay tests observe actual intermediate uploads rather than conflating an allocation's
one-time initialization source with hot-path transfers; CPU/device poison and replay oracles remain.

#### Composite prepared-step ownership

Descriptor-selected KernelArtifact and KernelGraph bindings now use the same checked composite
resource owner as modern graph bindings. BoundExecutableStep is only the ordered-child container,
not a teardown exception. Admission, recording and destruction require its live outer owner.
The former per-child swallowed destruction/free errors and publish-before-old-destruction ordering
are removed. A single outer construction accumulator receives every returned child binding,
backend-adopted debt, graph-private temporary and owned descriptor sub-buffer. Rollback reuses a
completed candidate's exact owner rather than wrapping its children in a second ownership plan.

Descriptor-private temporary and owned OpenCL sub-buffer constructors reserve acquisition slots
before native contact. Successful acquisition owners move atomically from conservative debt to
their post-kernel dependency layer. A throwing constructor remains indeterminate; no retry or
“nothing allocated” inference is allowed. Failed construction and failed replacement remain hidden
prepared registrations when cleanup cannot finish. Root footprints pin borrowed session storage,
and session close stops before root teardown if any prepared owner remains unresolved.

The dependency order is adopted debt → kernels → owned views → temporaries. Descriptor-owned view
owner/buffer vectors form matching ordered prefixes, with only a possible owner-only final tail if
bookkeeping fails; other graph paths provide no private view owners. Do not introduce a mixed raw
legacy prefix and migrated-owner suffix into this representation. Constructor tests cover first
and second allocation/slice failures, root/session retention, successful multi-node teardown, and
a later child-bind failure with unknown earlier child destruction. Conservative debt may retain
otherwise independent private storage; it does not justify freeing possibly referenced storage.

Scope remains explicit: public/root allocation constructors, modern graph external materialization,
global registries, root/view/event owners and sticky linked/parallel close are separate follow-ups.
This closes the descriptor-specific lifetime exception, not the entire memory-management campaign.

#### Retained linked/parallel close

Completed prepared programs, program sequences, direct prepared graphs and LinkedExecutables
retain the common cleanup owner across close attempts. Marking a value unusable no longer hides
failed cleanup on later calls. Successful sibling releases are not repeated; only failures
explicitly declaring retry safety are retried. Unknown outcomes preserve their original exception
and ownership. Attached linked teardown is recording → independent phase/program siblings → all
dependent allocation registrations. An owned linked executable has one session-close resource,
not a second competing resource plan for the same native children.

Prepared run/profile/report operations serialize with release, and active-use depth rejects
same-thread reentrant destruction before the closed flag changes. Linked execution scopes use
the same principle under their existing lifetime lock; live operations require a live cleanup
owner. Private, non-watchable active-use counters are installed before observable atom mutations
and balanced even when a watch throws; they do not alter numerical execution-state projections.
Returned raw handle reports remain **borrowed**, not lifetime leases: keeping an owner alive or
using callback-under-lock run/profile APIs remains the caller's responsibility.

Compiled close now performs pure close admission under the linked lifetime lock, invalidates
projected output wrappers, and always delegates to retained linked cleanup. It no longer mistakes
teardown for an active numerical execution scope or silently skips cleanup when the unusable flag
is already set. Output leases, active callbacks and missing owners decline before invalidation or
close-state mutation. This intentionally changes failed-close behavior from a later silent no-op
to a repeated report of retained ownership; numerical surface semantics are unchanged.

Fault oracles cover suppression and independent sibling attempts, nested sequences, explicitly
retry-safe cleanup, unknown outcomes without retry, concurrent use/release, reentrant callbacks,
owned/attached link dependency order, lease preflight and compiled output invalidation. Hand-built
host test fixtures share an ownership-complete helper; it does not validate plans or manufacture
native evidence. Native linked composition remains an affected numerical oracle.

Generic program/sequence staging now reserves the common
cleanup plan before binding and retains failed rollback via executor `:adopt-cleanup!` or the
existing `::resource-cleanup/unresolved` exception contract. Only successfully returned handles
enter that plan; an executor must retain indeterminate native acquisition that throws before
returning. Direct-graph sequence executors are admitted before acquisition, and their handle
slots/owners are reserved before binding. Completed values retain the exact construction owner.

Link program cleanup debt is adopted into the existing session prepared layer with a conservative
registered-root footprint. Original and late alias roots remain pinned. Session teardown treats
prepared programs and kernel graphs as distinct dependency layers, so parent failure cannot
trigger a second child retry in the same close. An adopted program owner contains binding
destruction only: session root allocations stay session-owned and are not freed twice against
the session's close snapshot. Owned Link construction reserves one session owner before target
and external-binding admission; unresolved close keeps that owner (and session) reachable on
the ordinary unresolved-cleanup exception. Successful construction retains that same owner.

Fault oracles cover independent siblings, sticky child failures, direct-graph wrapper failure,
owned admission/close failure and a real typed equation-program Link binding failure with both
attached and owned sessions. The last uses production lowering/validation/staging/views/adoption
and substitutes only native allocation/upload/bind/release. These checks do not certify all
public/root/view/event acquisition, output-value destruction, provider-backed durability,
distributed fabric execution or external model training, which remain explicit follow-ups.

The canonical backend-buffer ownership slice is implemented; final CI gates remain pending.
OpenCL/Level Zero root constructors
reserve a canonical backend buffer owner and can publish it through `:retain-owner!` before
native contact. Unresolved rollback uses `:adopt-cleanup!` (or the retention callback); containers
deduplicate those notifications by owner identity. Array initialization rolls back through that
same owner. OpenCL sub-buffers own a separate native reference; root and slice staging has
per-buffer shared arenas independent of the global runtime arena and parent staging segment.
Level Zero pointer slices remain non-owning and retain the root lifetime reference. Level Zero
captures the pointer before reinterpretation/record setup and destroys with its allocation
context; a retired context is rejected before native free. Transfer/binding admission checks
owner liveness and context identity. These are not concurrent-use leases or automatic reset
recovery. Slice extent arithmetic is checked for overflow.

Modern graph-private allocation/view construction now retains that exact backend owner rather
than creating a second native acquisition/destruction authority. Public session allocation
retains exact root owners before contact, publishes values/contracts after initialization, and
rolls back through those owners on any Throwable. Unknown outcomes remain session-owned and
make the session non-live. Close attempts independent root siblings and removes successful roots
incrementally, before kernel-arena teardown. Root lifecycle operations reject reentrant close,
free, registration and ordinary session use from synchronous atom watches.

Graph/call external materialization uses the same construction accumulator and canonical owners
as modern graph-private staging. The old temporary allocator, successful-prefix-only rollback,
raw session-root free and swallowed view-rollback helpers are removed. Fault doubles share one
ownership-complete constructor, not production inference from a partially returned buffer.

Synchronous session transfers hold the session monitor through resolution, validation and
execution. A scoped-use guard prevents a callback from reentrantly destroying roots. Common
`resource-cleanup/lifetime-owner` distinguishes destruction authority from borrowed lifetime
identity; registration validates the backend buffer, and free compares exact owner identity as
well as registration/object/allocation identity. A registered Level Zero slice or copied root
cannot hide the native alias behind a different allocation ID. Published root/table owner
mismatch declines before native free.

Asynchronous transfer submission uses the same scoped-use guard through event publication,
including retained host resources. Binding replacement uses one shared publication transaction
for prepare, step, graph and call: after successful old-generation teardown, a rejected or
throwing publication removes the exact retired/candidate registration before candidate rollback.
Successful publication also revalidates candidate identity, so a watch that releases the new
binding cannot return an already-dead handle. Replacement publication failure may leave the key
unbound; it cannot restore a destroyed old generation. A failed old-generation destruction still
retains that generation's cleanup debt. General atom validators that reject recovery mutations
can prevent deregistration; the primary error retains that recovery failure as a suppressed error.

The combined affected ownership, graph, transfer, Link composition, program-memory and adjacent
ABI/binding/native-recording checks pass (160 tests, 1572 assertions) in the existing REPL.
Oracles cover actual backend API calls after free/context
retirement, staging independence, setup/upload rollback, uncertain creation/free, independent
siblings, repeated close, reentrant/throwing watches and blocked transfer versus concurrent
free/close. Linked numerical replays retain their CPU parity anchors.

Surface lifetime behavior is intentionally stricter: directly fabricated backend records have
no native destruction authority; borrowed aliases must be detached before freeing their owned
root. `deftm` numerical semantics are unchanged. Raw backend callers, opaque native segments and
returned NIO views still require external lifetime/concurrency discipline; these checks do not
grant a lease or make concurrent runtime reset safe. Kernel-arena destruction itself still needs
retained outcome ownership; this slice only prevents an arena-close failure from repeating
successful root frees. Final review/CI remain pending. No performance claim follows from these
correctness tests, and the full distributed/training campaign remains incomplete.

Review-driven regressions cover retained-event publication with reentrant close, exact registry
removal on throwing watches/validator rejection, preserved unrelated generations and failed
old-generation destruction, plus public prepare rollback after a watch releases its candidate.
The next native lifetime consolidation is kernel-arena teardown: both backend arena closers still
swallow native release errors, and Level Zero scans arbitrary MemorySegment metadata rather than
declared owned resources. Replace these with retained outcome owners before claiming complete
native lifetime coverage; do not infer destructor authority from a value's representation.

The first #1013 full CI run exposed four remaining legacy fixture failures in dispatch, GEMM
Link topology and recorded-root ordering. Their mock constructors now use the shared canonical
acquisition/retention fixture, and the owned staging root carries its exact owner in the session.
All 45 tests / 256 assertions in those three namespaces pass in the existing REPL. No production
ownership check was weakened; the rerun must still pass all seven exact-head CI gates.

#### Kernel-arena ownership: next coherent vertical

The entry-generation publication transaction moves from session-private code into the existing
`resource-cleanup` namespace, with a path-based registry interface. Prepared steps, graphs and
calls use it directly; no adapter or second transaction remains. It checks the exact old entry
after teardown and inside the publication CAS, and checks the exact candidate after atom watches
return. An unrelated generation installed during teardown/publication is never overwritten or
removed. Root and nested registry paths share the same recovery rule. This prerequisite passes
106 tests / 1019 assertions across the affected publication, ownership and topology namespaces.
Admission rejects nil candidates and the exact old generation before any teardown; recovery
deduplicates secondary error identities. Publishing metadata for an identical registration must
preserve its lifetime explicitly, not invoke the retiring replacement transaction.

The rest of this vertical remains implementation work, not an achieved ownership claim:

1. Reserve one owner at registration, before lazy native contact. OpenCL owns its cached kernel,
   program and per-entry shared staging arena; Level Zero owns its fresh base kernel and declared
   cached staging buffers. Destruction orders kernel before program/staging. Dedicated bound
   KernelCall handles remain separately owned.
2. Acquire/load through reserved slots; retain unknown load/rollback outcomes in the exact
   registration. Publish only into that generation. Identical registration preserves the exact
   owner, handles and staging; incompatible replacement uses the shared transaction.
3. Level Zero `ensure-seg` uses canonical byte-buffer constructors/owners, not naked allocation
   and free. Growth publishes an acquired candidate only after safe old teardown, preserving or
   adopting every failure. OpenCL staging is per-entry, not runtime-global.
4. Close independent entries, remove only successfully released exact generations, and retain
   failed/blocked entries. Remove arbitrary MemorySegment scans and representation-based cache
   preservation. Audit/remove the unused ZE cached-kernel/invoke pair and duplicate registry
   accessor only after confirming there are no consumers.
5. Prove acquisition/build/release faults, staging growth, shared-module arena isolation, repeated
   close and registration replacement, plus a real-device close/re-register smoke check.

Level Zero modules are content-shared global-cache resources, *borrowed* by registrations. They
must not be freed by an arena. Their canonical cache ownership and shutdown-before-context order
are the immediately following runtime-root slice; the arena slice must not claim reset recovery
or silently drop those debts. These two slices precede the still-open training/model and durable
distributed acceptance items rather than replacing them.

The unused ZE `create-kernel`/`invoke-kernel` pair and its global mutable-kernel cache are removed,
along with the duplicate registry accessor and stale marker documentation. Source/test/dev/bench
search plus read-only audits of pretrained-rstr, finetune-rstr, umap-rstr, evoc-rstr, city and
spindel found no consumers of those two Raster functions. Registered ABI invocation, fresh bound
kernels and the shared module cache remain. The observable change is removal of those legacy
raw APIs; high-level `deftm`/prepared/Link semantics are unchanged. Adjacent emission and native
binding checks pass (32 tests / 386 assertions), with the removed vars unmapped in the REPL.
The old arena and global-module shutdown implementations still require the ownership migration
above; dead-cache removal alone is not that proof.

#### Registered native generations: implementation checkpoint (2026-10-03)

The shared publication prerequisite is merged as #1014. Native arena ownership is still WIP,
not released or accepted as complete. Registrations now reserve canonical cleanup before lazy
loading. OpenCL owns kernel, program and a per-registration shared staging Arena; Level Zero
owns its base kernel and canonical byte-buffer children while borrowing cached modules.
Staging caches (including ZE host short arrays) live only in registration-owned state, not in
compiler metadata. Identical compiler registration retains that exact state; importing native
handles or cleanup authority through registration metadata is rejected before contact.

Pivotal review identified and prompted tests for reentrant acquisition teardown, lost-generation
cleanup retention, forged metadata and watch reinsertion of released entries. A shared native-use
scope rejects same-thread public register/close callbacks before they can retire an acquiring
slot. If a registry watch loses a generation whose rollback is unresolved, its exact owner and
arena remain reachable through a hidden failed-registration entry. No uncertain native release
is retried. Independent arena entries still close even if one retains failure debt.

Persistent-REPL checks: 26 registration/load/arena tests with 228 assertions, plus 48 cleanup and
native-buffer/kernel tests with 654 assertions, all passing. New regressions cover exact-parent
loss on internal cache/array watches and same-owner refresh, including successful child rollback,
and reentrant register/close during destruction/publication. Pruning only proven-released ZE
children bounds historical owner retention; its final parent-generation postcheck has its own
remove/replace watch regression. The pivotal reviewer approved this scoped semantic change.

Actual OpenCL and Level Zero registration/load/staging/close/re-register smoke checks passed on
the local device. Two OpenCL mixed-storage invocation/identical-refresh tests also pass (six
assertions). These are ownership checks, not performance or reset-recovery claims. All seven
exact-head CI gates remain required before merge.
Global cached-module/runtime shutdown ownership remains the immediately following slice; neither
this checkpoint nor green ownership tests close the training/distributed campaign requirements.

The first OpenCL CPU CI run exposed a production boundary leak: the resident compiler extracts
artifacts via `kernel-registry-entry`, which was returning enriched runtime registrations. Those
descriptors imported the new owner/registration fields and were correctly rejected on binding.
The fix preserves the original admitted compiler artifact before runtime enrichment (and the
cloned explicit ZE payload), separately from the owning registration. Public registry reads now
return only that compiler artifact, never arena identity, loaded handles or cleanup authority.
Backend loading/binding stays on private runtime entries. Caller-supplied `:arena-id` is rejected;
arena selection belongs to the explicit registration argument/dynamic scope. Native-cache tests
inspect private state to verify handle reuse rather than depending on this compiler API leaking
handles. High-level compiler/Link semantics do not change. This is a boundary correction, not a
relaxation of ownership admission. Both-backend resident block-transfer and OpenCL compiled
composition/mixed-storage public compilation regressions pass locally after the correction; the
full exact-head CI gates must be rerun.

The next runtime-root survey confirms that arena ownership alone cannot justify safe shutdown.
ZE still caches raw module handles, can race module/async-list creation, and never destroys its
global immediate lists/context during shutdown. OCL reset still drops native authority. Root
teardown must not scan private session children or force-close live consumers. Landing order:

1. Canonical exact-generation ZE module cache, input-byte snapshots, acquisition/publication
   rollback and checked destruction; keep modules borrowed by kernel registrations. Migrate raw
   module/fresh-kernel benchmark consumers to existing Artifact/KernelCall execution boundaries.
2. Reserve retained backend root construction owners before context/queue/list contact. Unknown
   initialization outcome stays reachable; independent siblings close but context/Arena wait.
3. Generation-qualified session/resource leases and balanced synchronous-use tokens. A live or
   failed child pins the root. Shutdown preflight must decline without mutation/native contact
   while leases remain, following session -> registry -> state -> child lock order.
4. Cover or retire direct ownerless allocations/SoA/display/scan helpers, and adopt unreturned
   OpenCL async-transfer cleanup debt. Do not infer ownership from exposed MemorySegments.
5. Only after those gates replace shutdown/reset with retained root release and new-generation
   initialization. Native fault, lease, concurrency and differential workload tests precede use.

These are the runtime prerequisites of the existing training/distributed campaign, not a new
planner or a substitute completion target. The module-cache slice does not enable eviction or
claim safe global reset while outstanding kernel/buffer/event borrowers remain.

### Module-cache ownership continuation (2026-10-03, merged)

PR #1015 passed all seven exact-head gates and merged as `35a3a1af`. PR #1016 also
passed all seven exact-head gates and merged as `6b872966`. It replaces raw ZE module-cache handles with exact owning
cache references and gives each fresh kernel a `kernel -> module-borrow` cleanup
dependency. A copied/stale handle is not authority. Payload hashing and native
compilation consume the same snapshot; reservation/publication rollback uses the
common cleanup transactions, and hidden failure debt retains the content key.
Live or indeterminate kernel borrowers make module-cache teardown decline before
any module destruction. Checked destruction retains failure debt without retrying
unknown native outcomes. Shutdown admission holds registry -> state -> module-cache.

Low-level API change: `load-module!` returns a cache reference, and
`create-kernel-fresh` returns an owned kernel value. `destroy-kernel!` accepts that
owned value, not a raw pointer. Compiler/session/KernelCall callers retain their
existing public API. Both production kernel constructors are migrated. The two
low-level reference benchmarks share a construction/rollback helper rather than
duplicating success-only teardown. Moving these comparator harnesses entirely to
the public Artifact/KernelCall surface remains a cleanup opportunity, not a claimed
completed migration.

Current focused evidence: 57 hardware-free tests, 568 assertions pass, including
content snapshots, concurrent identical loads, creation/readback/destruction faults,
shared borrowers, hidden-debt identity, reservation watches/validators, and stale
cache/context references, loaded-publication rollback, destruction reentry, and
benchmark prefix/recording rollback. A fresh JVM confirmed 27 registration/boundary
tests (250 assertions), 38 ownership/benchmark fault tests (355 assertions), and
seven actual-device boundary tests (84 assertions) against the final exact-reference
representation. The pivotal reviewer approved the code subject to device/CI gates;
the local device and exact-head CI gates are met. This slice does not certify
root context, command-list, buffer or reset lifetime safety; those are the following
steps, and the full eight-item compiler campaign remains open.

### Root-construction ownership (2026-10-03, local implementation under review)

The shared runtime-root boundary reserves a cleanup DAG and exact generation before
OpenCL context/queue or Level Zero context/list acquisition. Native status, NULL,
readback, publication and destruction failures retain exact unresolved authority;
independent siblings may close, while their context and host Arena remain pinned.
Lazy ZE async-list acquisition uses the same root's pre-reserved slot, serialized
with initialization. Invalid construction callbacks fail before publication.

This slice deliberately changes live `shutdown!`/`reset!` to fail with
`:runtime-root-leases-incomplete`, without destroying or dropping resources.
Only failed, never-live initialization can be cleaned by this boundary. Compiler-only
development reload remains available; native runtime reload requires a fresh process.
This is a temporary explicit restriction, not completed root lifetime support.

Local root/module evidence is 34 tests and 381 assertions, including production
initializers under hardware-free native fault injection. The combined ownership
suite passes 84 tests and 890 assertions. Pivotal review found no remaining R1
correctness blocker, conditional on fresh-process device checks. A fresh capped
REPL now passes 27 registration/boundary tests (250 assertions), seven actual-device
execution tests (84 assertions), and six live-root/reset assertions across OpenCL
and Level Zero. PR #1017 passed all seven exact-head CI gates and merged as `66ace972`.
Next: generation-qualified session/buffer/registration/recording/event leases and
direct allocation migration, followed by checked live teardown/reset. Do not let
root-internal cache ownership masquerade as an external lease that prevents all
teardown. The training, distributed simulator/exchange and durable PDE/AMR campaign
requirements remain unchanged and incomplete.

### Resource-lease follow-up (R2, in progress)

PR #1017 landed R1; PR #1018 passed all seven exact-head CI gates and merged as
`d4cb8c17`. Its first hardware-free gate exercises
generation-qualified lease admission/retirement, concurrent independent leases,
failed lazy-child acquisition, and lost-generation cleanup debt. No live teardown
is enabled by introducing this primitive. Canonical OCL/ZE buffers now acquire a
lease before native contact; independently owned OCL sub-buffers own separate leases,
while ZE pointer views share their root allocation's owner. The lease is the final
dependency in the same canonical child cleanup DAG. Known cleanup retires it; unknown
creation/destruction retains the composite child owner and keeps the root pinned.

Shared `cleanup/build!` now marks the exact owner as constructing, without holding
its monitor across callbacks. Concurrent or reentrant release declines before mutation
with `:owner-construction-in-progress` and explicit retry-safe metadata. Marker install
and retirement check exact token identity after watches; mismatches retain unresolved
authority rather than clearing another generation. Rollback clears its own marker
before native cleanup. This closes a publish-before-contact race in the common owner
construction boundary, not a buffer-specific dispatch convention.

Current focused evidence: 169 tests, 1,770 assertions pass; seven real-device boundary
checks (84 assertions), six live-root/reset checks, and two real-device buffer/view
lease checks (eight assertions) pass on Intel Arc OpenCL and Level Zero. No timing or full live
root teardown claim follows from these tests.

The next slice, `runtime/registration-root-leases`, keeps metadata-only registrations
lazy. Loading acquires one exact root lease before native program/module/kernel contact;
the registration cleanup DAG releases it only after kernel, program and staging cleanup.
Identical registration reuses that authority. Known rollback retires the pin, while
unknown native acquisition/destruction retains the composite owner and blocks retirement.
Pure root-admission rejection leaves its reserved slot fresh; publication failure uses
the common owner transaction rather than pretending a native acquisition occurred.
Pre-lock lifecycle guards also reject native callbacks entering the registration registry,
including ZE module-cache callbacks, before lock-order inversion can occur.

Registration-slice acceptance: 176 focused tests / 1,844 assertions pass in the persistent
REPL; seven actual-device boundary checks / 84 assertions, six live-root/reset assertions,
and two buffer/view tests / eight assertions pass on the local OpenCL/Level Zero device.
PR #1019 passed all seven exact-head CI gates and merged as `8268509f`.
Prepared calls, recordings/events and raw allocation consumers are still outstanding.

The prepared-resource follow-up starts with ZE's public `create-kernel-fresh`: every
fresh kernel now has an independent root pin in its existing kernel/module-borrow
cleanup DAG. The module cache remains a root-internal resource, while a fresh kernel
is a lifetime that may survive the base registration. Kernel destruction must finish
before the module borrow retires, and both precede root-pin retirement. Unknown create,
NULL readback or destruction leaves both borrow and pin retained. Hardware-free tests
exercise two independent kernels and all three uncertain outcomes; 178 focused tests /
1,860 assertions and seven actual-device boundary tests / 84 assertions pass.
OpenCL's public binder now constructs its independent kernel through the canonical
root-child owner, with a reserved native acquisition slot and exact root Arena projection.
The registration lock covers loading through fresh creation; logical native-use admission
is scoped separately so loading does not recursively enter the same guarded registry.
Unknown create/readback or destruction retains the composite kernel/root owner; known
argument-binding failure retires both. Root-child construction shares the transactional
lease-slot publication helper, with before/after-write failure tests proving no native
contact or stranded root pin.

Final local acceptance: 180 focused tests / 1,878 assertions, seven actual-device boundary
tests / 84 assertions, and four buffer/view/prepared device tests / 16 assertions pass.
On both backends a prepared generated kernel survives base-registration retirement,
executes correctly, and balances its independent pin. This does not establish arbitrary
concurrent root reset: recordings/events, use admission and raw consumers remain next.
PR #1020 passed all seven exact-head CI gates and merged as `ebe849bc`.

Recording follow-up: both `record-graph!` implementations construct through the canonical
root-child owner. The OpenCL submission and optional profiling queue precede root release;
ordinary graphs pin their borrowed compute queue's root too. ZE derives its existing native
list/event/pool/independent-queue DAG as a resource plan, then uses the same root construction
transaction rather than building a second owner. Native descriptor allocations and queue
creation consume the exact admitted root projection. Uncertain acquisition or drain retains
the graph's composite owner and its root pin.

Acceptance so far: 182 focused tests / 1,898 assertions, seven actual boundary tests /
84 assertions, and six buffer/view/prepared/recording device tests / 36 assertions pass.
The latter replay ordinary and profiled graphs on OpenCL and ZE after retiring their base
registrations, and verify independent root-pin balance through graph/prepared/buffer teardown.
PR #1021 passed all seven exact-head CI gates and merged as `427d238f`. These pins do not automatically borrow
every child kernel or buffer: existing session ownership still controls those dependencies.
Standalone asynchronous range transfers remain debt, especially OpenCL's old exception path
that swallows drain/event-release failures before closing staging. Migrate that path to retained
completion cleanup before enabling any live reset; synchronous-use and raw-allocation debt also
remain. No throughput or complete distributed-runtime ownership claim follows from these gates.

Standalone transfer follow-up: OpenCL transfers now share the submission resource plan
(drain → event acquisition slots → staging Arena) and the canonical root-child transaction.
Unknown enqueue/readback/drain/event destruction retains the exact composite owner, staging
and root pin, without native retry. Successful cleanup releases events and staging before the
root. Nonempty OpenCL completion tokens require an owner; the raw-handle/finally-close fallback
is removed. ZE shared-allocation transfers remain inline and do not invent an in-flight lease.

The session event table is the sole outer authority: its exact pending event owner and resident
footprint are published before backend contact. Private slots retain returned tokens or adopted
backend cleanup debt, eliminating a post-contact Atom publication. One cleanup DAG orders backend
completion before independently owned host leases. Host ownership is armed only after the backend
returns successfully, preserving the public caller-owns-on-failed-submission contract. Failed
unknown submission keeps that pending event/footprint, so buffer aliases/free/session close cannot
drop dependencies. Failed cleanup is not reported as a successful data operation. The backend
submission contract adds a uniform optional third argument with `:adopt-cleanup!`; direct callers
can still use two arguments. Public GPUEvent and retained-range APIs keep their signatures.

Current acceptance: 182 focused ownership tests / 1,898 assertions and 48 affected range/content/
fault/device tests / 519 assertions pass. New native fault tests cover unknown enqueue, NULL
readback, drain/event-release failure, strict token ownership, caller host ownership, resident
alias blocking, session close and pre-contact publication watch rejection. Actual transfer tests
check OpenCL root-pin balance, staged download visibility and honest ZE inline completion; existing
ordinary/profiled recording and prepared/buffer cases still pass. Pivotal final review and all
seven exact-head CI gates remain required. Synchronous-use admission, raw allocation consumers and
general live reset remain incomplete; these ownership tests do not establish performance.

Review follow-up separates operation failure from destruction failure: an await/profiling error
is recorded once, backend drain/release is still attempted, and host leases retire only after known
backend retirement. A destruction failure retains the canonical debt without repeating await or
an indeterminate native release; the original operation error is preserved with cleanup failure
suppressed. Known retirement consumes a released event even when its operation failed. Session
teardown uses the same private consumption path without treating an already retired operation
error as remaining native debt. Public poll/await/release now admit through scoped session use,
preventing callback/watch reentry into buffer or session lifetime mutation.

The review-fix fault suite passes 12 tests / 166 assertions, including caught/uncaught native and
cleanup-publication watch reentry and known-await-failure session teardown. An uncaught cleanup
publication watch retains its exact owner and root pin before backend contact; this is fail-closed
retention, not a general recovery guarantee for arbitrary throwing watches. The 182-test /
1,898-assertion ownership suite also passes after these fixes. Final review and exact-head CI are
still required; live reset admission remains deliberately unchanged.

Host-lease handoff follows the entire successful construction transaction, not merely native
submission return. A final construction-marker watch can still reject submission after backend
success: rollback retires the backend token but leaves host leases caller-owned, removes known
retired event debt, and preserves the exact primary error. No fallible publication or callback
follows the private host-ownership handoff. The dedicated regression exercises this boundary.

PR #1022 passed all seven exact-head CI gates and merged as `4fdcf4c1`. The affected ownership
suite passes 182 tests / 1,898 assertions, and transfer/range/distributed/local-native lease
acceptance passes 39 tests / 320 assertions. The existing tiny resident Gemma LoRA workload also
passes 43 assertions: 25 Level Zero updates reduce loss from 2.800152 to 0.256915 while tracking
the independent JVM trajectory. This is the existing model twin, not external real-weight
training, throughput or live-reset evidence.

Integration must retain the authoritative child cleanup as well as its root pin;
a count or raw pointer alone is insufficient. Session construction currently creates
only a kernel-arena identifier and must remain lazy (no GPU initialization merely to
construct a session in hardware-free CI). Canonical native buffer owners, independently
owned OpenCL sub-buffers, kernel/program registrations, prepared bindings, graph
recordings and asynchronous events must retire their root lease only after all native
children are released. ZE non-owning views inherit the allocation's lifetime rather
than fabricating a separate owner. Root-owned cached modules are internal children,
not permanent external leases.

Direct allocation debt found in the production inventory: `runtime/display.clj`
resolves raw ZE `alloc-shared`; ZE SoA construction uses `alloc-shared`/`alloc-device`;
the obsolete scan helper allocated raw block-sum/offset scratch. The retirement below removes
that unreachable path rather than introducing another owner for it. Migrate the remaining
display/SoA consumers to canonical
owners or retire the paths, including their fault and view coverage, before changing
the fail-closed live reset gate. Buffer/registration/session lock ordering must be
reviewed against concurrent acquisition and teardown, not inferred from single-thread
test success.

### Retire backend-local scan runtime duplication — 2026-10-03

The production scan path already lowers TypedSOAC to SegScan, then a verified KernelGraph /
KernelDispatch behind the common executable ABI. The public `raster.par/scan`,
`raster.par/scan-exclusive` and `raster.gpu/invoke-scan!` surfaces retain their contracts. Raw
exclusive scan entering the source-shaped backend still fails closed with
`:exclusive-scan-requires-typed-schedule`; it cannot select a backend-local algorithm.

An independent reference/dispatch audit found no production, test, dev or benchmark caller,
generated marker or dynamic selector for either backend's `invoke-registered-scan-exclusive-kernel`.
The only ZE recursive-helper references were its definition/self-call and that obsolete entry.
Both entries and private ZE `invoke-full-gpu-scan!` are removed (235 lines), including raw
block-sum/offset allocation and finally-free behavior that could lose native acquisition debt or
mask a primary error. This is a break for direct callers of those internal runtime Vars, not a
change to supported source or session scan semantics. No raw allocation API is removed in this
slice: display and GpuSoA still use those APIs and require separate ownership migration.

Architectural ratchets reject these markers in program extraction and require the retired runtime
Vars to be absent. Existing typed graph/ABI, raw-source rejection and compatibility-ledger tests
remain the semantic oracles. The capped REPL passes 28 boundary/ledger tests / 150 assertions and
17 selected typed route, JVM, staged/resident and actual OpenCL/Level Zero scan tests / 88 assertions.
Persistent native namespaces are not reloaded: only the three deleted Vars were explicitly
unmapped. Cold CI remains necessary to validate source loading; final review and all seven
exact-head gates are required before merge. This retirement narrows ownership debt, not the
original training, distributed or AMR completion requirements.

### Display allocation uses canonical ownership — 2026-10-03

The live `runtime/display.clj` GPU constructor previously resolved raw ZE `alloc-shared`, exposed
an allocation without a release API, and called nonexistent JDK 25 `MemorySegment.copyInto`.
It now allocates through the existing canonical `make-buffer` transaction and retains that exact
child cleanup before native contact. The enclosing RenderBuffer cleanup delegates to the child
owner; it creates no second native destruction authority. Construction and initialization failure
retire known resources or return explicit unresolved cleanup containing the exact child debt.
Repeated retain/adopt callbacks must name the identical canonical child owner; a distinct
generation is rejected without overwriting the first. The constructor's unresolved-error chain
retains any rejected second generation, while the enclosing cleanup retires or retains the first.
Pixels are initialized to zero before publication; each axis and pixel count are checked, so
invalid/overflowing dimensions, including an oversized axis paired with zero, fail before
driver contact. The record's shape is unchanged and native namespaces/records are not reloaded.

The surface lifetime addition is `close-render-buffer!`: GPU callers must establish completion
and end all pointer borrows before closing. `:device-buffer` exposes the canonical allocation;
`:seg` remains a borrowed pointer, not ownership authority. CPU close is a no-op. Synchronization
checks both the render owner and allocation owner and copies completed pixels into the existing
AWT int array. It is not an implicit wait or a zero-copy GPU-to-AWT view; CPU-to-AWT remains
zero-copy. This does not establish arbitrary concurrent kernel borrowing or safe live reset.

Six hardware-free fault tests plus nine actual root-lease device tests pass 15 tests / 92
assertions, including real ZE render allocation/synchronization and balanced root-pin retirement.
The existing display compatibility cases pass 3 tests / 11 assertions and the affected canonical
ownership suite passes 182 tests / 1,898 assertions. Exact-head review and all seven CI gates are
required before merge. The GpuSoA review recommends retirement in favor of the already working
ResidentComposite/session-owned field path; do not extend raw composite allocation as a second
ownership framework. External training and distributed/AMR completion remain on the campaign.

### Retire ZE-only raw SoA ownership — 2026-10-03

The reference audit found no production/dev/benchmark caller of ZE `GpuSoA`, its raw shared/device
allocators, or its reflection-based copy helpers. The supported `defvalue` SoA vertical already
uses typed field projection, ordered `:binding`/`:field` ABI slots, session-owned DeviceBuffers
and non-owning ResidentComposite values. Retire the separate ZE record, constructors, predicate,
copy helpers, `n-elements` helper and binding arms rather than adding another composite owner.
Direct callers of these internal ZE Vars must migrate; supported SoA source semantics are unchanged.
Vulkan's separate GpuSoA implementation and general raw allocation APIs are untouched.

Five leaking/exception-swallowing raw-allocation tests are replaced by a cold namespace absence
ratchet. Ordered/subset composite binding checks now exercise both backends; actual public
aggregate device tests remain. New hardware-free tests exercise second-field create/upload
failure through canonical session allocation, including uncertain native destruction: no partial
value is published, known prefixes retire, only the exact uncertain child retains its root pin,
and subsequent close never repeats uncertain native destruction. A freed composite field is
rejected by both production binders before driver loading. This is not a new concurrent-borrow
or live-reset guarantee.

The capped REPL passes 74 affected tests / 749 assertions, including public aggregate execution
on OpenCL and Level Zero. Deleted source and test Vars were explicitly unmapped without reloading
native records; cold CI remains required. Exact-head independent review and all seven CI gates
remain mandatory before merge. The eight-item campaign, including external training acceptance,
distributed execution and multilevel/AMR validation, is not complete.

### Wide boxed JVM calls — 2026-10-06

Real-checkpoint training isolation exposed a separate JVM boundary defect: a dynamic
38-argument VJP call emitted a nonexistent `IFn.invoke` overload. Positional `IFn.invoke`
ends at twenty arguments. The three boxed `emit-fn-call` paths (current Var root, local
function and expression head) now share one emitter: up to twenty arguments retain
positional invocation; wider calls build an argument array and use `RT.seq`/`IFn.applyTo`.
The head and each argument are evaluated once, in source order. Typed static calls and
surface arity semantics are unchanged; this is not new GPU variadic support.

Regressions cover 20/21/38 arguments in all three paths, current Var-root replacement,
mixed primitive boxing, effect order, and a real compiled 38-argument typed callee's
`applyTo`. The capped REPL passes 39 JVM tests / 102 assertions and the combined wide-call
and compiled-AD tests pass 8 tests / 30 assertions. Independent review found no blockers;
cold CI remains required. Other guarded interop/wrapper fallback emitters remain separate
debt, not a claim that every boxed call path has been consolidated.

The real-weight acceptance gate remains held at its original tolerance. Its raw Ak
gradient already differs before SGD. On identical captured operands, the final Ak
contraction and preceding low-rank contraction match explicit sequential FP32 evaluation;
the normalized input also matches the CPU exactly. These checks narrow the investigation
to upstream cotangents, but do not establish full training acceptance or SOTA performance.

### Typed JVM loop carriers and lexical scope — 2026-10-06

The real-weight investigation exposed a JVM precision inconsistency, not a new GPU
schedule requirement. Ordinary lazy-JIT `sum-kv-heads` accumulated Float inputs in
Double, whereas the typed GPU/AOT path used Float. The cancellation probe
`[1e8, 1, -1e8]` returned 1 on the former path and 0 on the latter. Recurrence
analysis lacked the loop's own binding environment and the element type of expanded
`clojure.core/aget` reads. Both are now retained; dependent carry widths propagate
to a fixed point, and Long/Float joins use the existing Double promotion rule.

General structural let inference, recurrence scanning and loop seeding share a
source-ordered lexical binding environment. Unknown locals explicitly shadow outer
types rather than inheriting them, including lets inside recurrence expressions.
Array load emission and inference share their retained operand-type translation;
bare `aget` names are not treated as canonical intrinsics without source context.
Numeric recurrence branch joins do not change general boxed-if emission.

The numerical surface consequence is intentional: affected ordinary Float loops
now honor their declared per-add Float precision. Explicit Double loops and genuine
recurrence widening remain supported. This does not change AD rules, GPU reduction
association, BLAS arithmetic policy or acceptance tolerances. The independent review
found no remaining blockers; affected JVM suites pass 79 tests / 270 assertions in
the capped REPL. Warm emitter reloads required fresh anonymous class names in that
diagnostic process; no production counter/cache policy was changed. Cold CI is still
required. Rebuild the real-checkpoint CPU oracle before comparing again: previously
compiled ordinary functions retain the old arithmetic. The external real-weight
gate remains held, and the eight-item campaign remains incomplete.

Cold CI then exposed three `matrix-norm` verifier errors: improved lexical inference
made `case*` predict a uniform primitive result, but a loop arm still emitted a boxed
value. Case emission now reconciles each actual reaching branch, including its default,
with the chosen stack merge type. Unsupported coercions and void-to-primitive predictions
fail before verification instead of silently claiming a conversion. Named/default loop
arms, empty/nonempty inputs and a deliberately corrupt Boolean prediction are covered.
The final affected JVM plus dense-linear-algebra run passes 95 tests / 312 assertions;
the replacement exact-head cold CI run remains required before merge.

A fresh capped process rebuilding the pinned real-checkpoint oracle still declines:
loss error is 0.009375, input-gradient relative error 0.005526, and 11/28 adapter
coordinate-relative checks exceed the unchanged 0.02 threshold (maximum 1.06532).
The loop precision discrepancy is therefore fixed independently of the remaining
training mismatch. Continue identical-operand BLAS/sequential-FP32/public-GPU
triangulation and attention-cotangent isolation; do not silently change the oracle,
relax tolerance or promote schedules on this evidence.

### Same-operand contraction numerical evidence — 2026-10-06

NN, NT and TN BLAS projections now have a bounded public-device regression against
an independent sequential FP32 oracle, rounding every product and addition. It covers
reduction widths 3, 17 and 640 with nondyadic inputs, plus `[1e8, 1, -1e8]`
cancellation in each physical layout. The affected public matrix replay selection
passes 4 tests / 78 assertions on local OpenCL and Level Zero, with no native skips.
This checks the current portable evaluation order, not CPU BLAS bitwise equivalence
or an optimized matrix-family throughput claim.

A separate same-operand CPU/OpenCL diagnostic uses threaded MKL (thread environment
variables unset). GPU output matches explicit sequential FP32 exactly in all nine
smooth-input cases; MKL differs by up to 9.06e-6 absolute. The pipeline documentation
therefore no longer promises that `:f32-scalar` bit-tracks CPU BLAS. Exactness refers
to the retained typed evaluation order, and excluding FP16 does not guarantee a
full-gradient coordinate-relative bound near cancellation.

Remaining design obligation: state the arithmetic permission of projected BLAS
operations at their existing semantic boundary and preserve it through TypedSOAC,
scheduling, executable/tuning identity and reference selection. Ordered source folds,
abstract BLAS products and explicitly reassociated schedules must not silently share
an exactness claim. Do not introduce a second registry or weaken the real-checkpoint
gate to paper over this distinction. Continue same-input attention pullback isolation
before deciding whether that gate exposes a miscompile or an inadequate arithmetic
equivalence specification. External full-model training acceptance is still open.

### Source-arithmetic provenance through the existing contraction spine — 2026-10-06

The existing BLAS projection table now marks abstract product provenance. The concrete
Float/Double contract is constructed only after contextual specialization resolves the
contraction dtype; raw overloaded method spelling and stale call tags are not a second dtype
oracle. Ordinary and staged contractions retain their typed SSA semantics without fabricating
one global precision for a mixed-component fold. The closed descriptive schema distinguishes
operand identity conversion, the product-reduction precision floor, implementation-dependent
association/rounding inside that product, and the unchanged typed result transform.

The facet is explicit in TypedSOAC, projected contraction facts, both SegRed lowering paths,
scheduled numerics and the existing emitted certificate. A changed or omitted scheduled facet
rejects, as does a BLAS compatibility assertion on a source lacking that provenance. Existing
semantic identity therefore changes; old contraction cache artifacts rebuild. No new IR,
operation registry, schedule permission, selector or reference tolerance was introduced. Missing
facets in older typed equations conservatively project to retained SSA, never inferred BLAS.

Focused frontend/source tests pass 120 tests / 646 assertions; affected contraction, dialect
and certificate suites pass 84 tests / 636 assertions in the capped warm REPL. Review found
and corrected premature dtype resolution and lost producer-local provenance. The joined
hardware-free test covers the full semantic-to-artifact chain and rejects certificate mutation.
Affected public portable/register replay passes 4 tests / 78 assertions on local OpenCL and
Level Zero, with no skips. Full cold CI remains the merge gate.

Cold emitter-fixture CI identified an omitted facet in the staged packed/scalar candidate
constructors. Both now explicitly retain the same source facet before their graph rebinding;
the shared scheduled validator remains strict rather than hiding the omission with a default.
The same audit found split-K combine and parent mixed-matrix refinement propagation. The combine
retains its own ordered-SSA facet; physical cast/transpose/matrix stages do not acquire the parent
BLAS contract. Attention JVP additionally specializes an already staged closure after homogeneous
operand dtypes resolve: BLAS provenance is re-instantiated at that existing boundary, while
ordinary staged component semantics and the constructor mismatch guard remain unchanged.
Its focused suite passes 5 tests / 21 assertions. The exact public emitter-fixture path emits
157 artifacts each for synthetic CUDA and HIP; compiler-toolchain CI is still required.
The reviewed replacement also passes 53 GEMM/staged tests / 1338 assertions and 34 source-facet/
GEMM tests / 1243 assertions, including explicit local split-K and parent refinement provenance.

This is provenance, not a new equivalence theorem or schedule admission rule. Follow up with
restrictive refinement checks and reference-selection policy, preserving existing explicit
reassociation/mixed-precision consent. Before provenance could ever grant permission, distinguish
recognized compiler BLAS origins from user-supplied metadata assertions; the marker currently
authenticates neither. Initial numerical compatibility evidence remains alpha=1/beta=0; nontrivial
BLAS alpha/beta rounding is not proved by the product facet. The real-checkpoint gate (#1060)
remains held with unchanged tolerances and the original eight-item campaign remains incomplete.

### Declared product-accumulation floor — 2026-10-06

The shared numerical validator now applies a reject-only check when a contract carries abstract
BLAS source arithmetic and declares an accumulator dtype: a Float source permits Float/Double,
and a Double source permits Double. All single and component declarations are checked, including
when both fields appear. Naming an exact mode does not bypass a below-floor declaration. Ordered
typed SSA with no global accumulator declaration remains unchanged; ordinary staged and mixed
component folds do not acquire a fabricated BLAS floor. Existing mixed-matrix models still retain
Float accumulation and separately attest their explicit operand conversion.

This checks declared accumulator dtypes, not every instruction or operand conversion. Exact
source/body certificates and paired operational models remain their proof boundaries. No new
schedule permission, selector, error tolerance or numerical oracle is introduced. Independent
review found no blocker; focused source/numerical/GEMM suites pass 49 tests / 1390 assertions.
Native portable BLAS and explicitly mixed attention/projection checks
pass 2 tests / 72 assertions on OpenCL and Level Zero. The real-checkpoint gate remains held.

### Explicit register-product realization — 2026-10-08

The real-checkpoint diagnosis is recorded in `local-compiler-evidence.md`: generated
forward matrix products and matrix pullbacks match ordered Float multiply/add
references, while native BLAS uses different reduction/rounding realizations.
This is not sufficient to close the unchanged model-gradient gate.

The existing register-tiled schedule now exposes `:typed-contraction :multiply-add`
as `:decomposed` (default) or `:fused`. The latter produces canonical typed `:fma`
SSA without a new operator registry, emitter or model-specific compiler rule.
Only explicitly permissive register-tiled/fused-register dispatch requests admit
it; other strategies and restrictive precision reject the request. The resolved
schedule, body, legality, numerical certificate, tuning metadata and exact
complete-write rederivation retain the choice. Unknown nested schedule fields
fail closed, including misspelled arithmetic choices.

`Decomposed` describes the typed IR, not proof that a vendor compiler will refrain
from contraction. Numerical rounding remains implementation-defined. Explicit
FMA is currently a C-family capability; the existing WASM polynomial facet is not
a single-round FMA implementation. Before cross-target exact rounding can be
claimed, introduce checked target non-contraction controls or explicit rounded
operations and verify the resulting device/compiler instructions. This is a
correctness follow-up, not permission to weaken an ordered source reduction.

Bit-sensitive NN/NT projection checks on local OpenCL and Level Zero compare raw
Float bits against independent ordered oracles, replace inputs and compile
decomposed/fused/decomposed to exercise cache separation. Fused CUDA sm80 PTX and
HIP gfx1100 code compile locally; inspected PTX contains `fma.rn.f32`. No NVIDIA
or AMD execution/performance acceptance is implied. The real checkpoint's fused
Q projection matches the independent ordered FMA oracle exactly, but still differs
from native BLAS. Full-model gradient parity, sustained training and the original
distributed/AMR campaign residuals remain open.

### Finite collective composition and parameter transitions — 2026-10-09

`refinement-plan-fields` separates existing collective field construction from the
ordinary whole-plan validation boundary. Fields are neither a certificate nor an
executable. `refinement-plan` retains its existing single-request behavior.
`compose-refinement-plans` merges explicitly scoped requests: equal values, shards,
groups and local entries may be shared, but calls, copy bindings and refinements
must remain disjoint. The containing context supplies public outputs and metadata;
nonempty fragment attributes are rejected instead of silently losing evidence.
There is no inferred alias, ID renaming, parameter update or dependency.

The finite training regression composes three AD → all-reduce → SGD steps, with
unequal local batch counts. One parameter value/shard/allocation and startup source
per worker persists across steps; contributions and scratch are epoch-local. Each
new producer depends on the preceding update and only final updates are exported.
Existing readiness and runtime authorities still require one initialization phase
and reject replay of a completed owner. Numerical comparison uses an independently
rounded scalar oracle at the existing tolerance; replica equality is separately
bitwise. This is a co-located resident-copy fixture, not multi-node fabric,
dynamic epoch execution, a trainer API or acceptance of the held real-model gate.
Private allocations deliberately remain distinct throughout this finite DAG;
steady-state training and lifetime-proved scratch recycling remain follow-ups.

The affected refinement/training suites pass 20 tests / 371 assertions on the
capped warm REPL, including actual OpenCL and Level Zero execution with no
capability skips reported. Whole-repository fresh-process and CUDA/HIP compiler
checks remain CI obligations; this does not report a training throughput result.

### Whole AMR cycle as a local numerical provider — 2026-10-09

The existing ratio-2 diffusion cycle is lowered equation-first as one local
program. An ordinary certified LinkComposition connects its coarse/fine outputs
to the existing `arrays/acopy!` operation, committing both fields to initialized
state owners. The cycle borrows those owners read-only. Topology is constant and
intermediate storage is private scratch; the public outputs are the two state
fields. Preparation does not open a device session or allocate device storage.
Semantic input/output ports identify the connections, rather than inspection of
generated ABI names. No diffusion, interpolation, reflux or restriction kernel
is reimplemented by this provider. Public `compiled/compose` retains the prepared
artifact, component preparation reports and donated state boundary for subsequent
execution-identity checks; the provider does not implement another composition API.

Two consecutive invocations of the same local executable compare both complete
fields with the independent coordinate oracle at the existing 1e-11 tolerance.
They also check composite mass and exact average-down. The second invocation
uses the resident committed state without a host upload; original host arrays
remain unchanged. Focused tests pass 4 tests / 47 assertions on actual OpenCL and
Level Zero with no capability skips reported. PR #1127 landed after exact-head
independent approval and all seven fresh-process/compiler CI contexts succeeded.

This boundary is not yet an AMR temporal certificate or a distributed cycle.
The next planner slice must explicitly bind the exact existing program and its
ordered memory/effect evidence to coarse prediction, two fine substeps, temporal
boundary interpolation, register reset/accumulation/consumption, reflux and
average-down. The distributed plan should see one complete local compute call,
not duplicate its internal scratch scheduling. Keep the current schema-1
hierarchy/transfer modes unchanged; introduce temporal facts as a validated,
certificate-bound facet rather than unchecked attributes. Initial synchronization,
CFL stability and the mathematical meaning of the pinned numerical producer
remain separate obligations. No adaptive hierarchy, multi-patch cycle, mid-cycle
restart, storage recycling or asynchronous completion proof is claimed.

#### Temporal binding investigation

The current synthetic Intel preparation retains 26 source equations in the cycle
program: three host-only shape equations and 23 device submissions, followed by
the two state-commit submissions. These counts are observations of this retained
artifact, not a fixed schedule contract or a numerical correctness criterion.
Retained equation result-storage metadata and the validated ordered memory report
provide the storage/effect boundary; source equations must remain authoritative
if fusion later changes the physical submission count. Matching generated kernel
names, destination names or counts does not establish interpolation/reflux math.

No new patch-reshape machinery is needed for the next distributed binding.
`distributed-compute` already admits explicit local domains with one plain dense,
equal-volume ABI leaf, matching dtype and row-major coordinates. A flat numerical
ABI can therefore bind a rectangular patch through existing `:local-shape` and
owned `:placements`, with bounds and coverage checked by the same authority.
This does not permit arbitrary strides, quantized representations or aliasing.

The temporal facet must retain the hierarchy/state/distributed certificates and
exact prepared-program identity, distinguish the two state owners, and bind
register reset/update/consumption evidence to the complete-cycle compute call.
It must preserve the existing source-read/target-write transfer witness unchanged.
Source-level temporal stage meaning is explicit producer evidence; ordering,
storage coverage and exact artifact correspondence are independently checked.
The facet must reject changed geometry, timestep, stage coverage, field mappings,
completion call or retained program instead of accepting descriptive fingerprints
as a certificate. Publication still needs real completion and durable-byte
evidence; ordered memory accesses cannot authorize release or scratch recycling.

The planning review favors a separate certified temporal execution facet over a
schema-2 extension at this stage. Attach it to a verified schema-1
`:hierarchy-only` workload and retain the original sealed Prepared at the ODE /
execution layer, avoiding a compiler-IR dependency on `gpu.compiled`. Neither
the existing transfer-cycle mode nor its two-role implementation witness changes.
Start with pure stage/effect attestation and tamper verification, then bind the
actual provider and project one ordinary distributed compute entry. Generalized
multi-patch/adaptive temporal planning is the point to reconsider schema 2.

Exact program identity deliberately excludes initializer bytes. A same-shaped
topology/coefficient change need not alter it. For the first temporal facet,
geometry and numerical-stage meaning must remain explicitly trusted producer
attestations, bound into the certificate but not advertised as topology-byte or
PDE correctness proofs. A later content-addressed constant projection can make
the geometry-to-initializer correspondence independently checkable.

`emitted-parallel-program/retained-numerical-equations` now supplies the shared
top-level semantic projection needed by stage coverage. It validates the emitted
program, excludes host-only setup and physical operation sequences, and expands
the retained source equations of a compound emission. Dispatch alternatives
must agree under the existing canonical fingerprint relation, including NaN and
signed-zero semantics; expanded equation IDs must remain unique. Structured
loops retain their nested control under one outer equation. Physical fusion
does not erase the numerical spine or authorize an application interpretation.
Host-prefix validation and projection use one numerical-body extraction helper.
Affected provenance/fusion/dispatch/structured-control suites pass 72 tests /
685 assertions on the capped warm REPL; fresh CI remains a landing requirement.

The first ODE temporal projection now checks the bounded producer's nine-stage
attestation against that shared semantic spine: each numerical equation occurs
once in source order; every stage retains its predecessor, time-fraction roles
and register transition. Register evidence preserves declared result-storage
access roles as well as operand dependencies, so a read/write update cannot be
misclassified as a write-only reset. Its last semantic use is derived from those
events. The register must be a nonempty plain FP64 tensor, not an input scalar or
packed representation. These are structural checks over a numerical producer
attestation, not inference of its mathematical meaning from destination names.

The stage layer deliberately leaves full-write reset coverage, completion and
release unproven. It does not yet bind a Prepared, workload, geometry bytes or
distributed call. Those remain the next execution-facet obligations; no new
planner/runtime or alternate diffusion kernel is introduced. Tests compile the
actual numerical cycle without device allocation and include isolated access-role
and storage-facet controls. The focused stage suite passes 5 tests / 25 assertions
on the warm REPL; exact-head review and fresh CI remain landing requirements.
Together with the existing numerical/device/provider suites, the affected AMR
check passes 18 tests / 374 assertions, including actual OpenCL and Level Zero
cycles and synchronized restart with no capability skips reported.

The next execution-binding investigation confirms that an ordinary one-worker,
one-compute DistributedPlan can bind both resident cycle state owners through
existing explicit `[4,4]` local domains over their flat `[16]` ABI leaves. Normal
`compute-bindings` derives `:read-write` for both fields and normal readiness
accepts the plan's initialization/dataflow. Values still require an explicit
sharding facet; no layout or sharding inference was added to bypass that rule.
The probe allocates no driver resources and is not yet a certified AMR execution,
runtime demonstration, durable publication or measured cost result. Next bind
this exact local program and these whole-patch domains to a verified
hierarchy-only workload and the temporal producer evidence.

The bounded temporal execution facet now independently checks that workload,
its sole distributed compute binding, and the original sealed Prepared against
the producer's stage/timestep/field evidence. Both whole state owners require
shared-authority complete-write coverage. Physical register evidence requires
a complete private reset, coarse transport, fine transport and final reflux
consumption in order; release and observed completion remain unproven.

The new certificate does not retain mutable initializer arrays: its verified
workload witness omits the ordinary live distributed certificate, while its
source-free distributed projection replaces the single local LinkPlan with the
existing exact execution identity. The original workload and Prepared remain
owners outside this structural witness. Readiness still independently checks
initializer source identity, but the retained obligations exclude source objects.
Geometry fingerprints and synchronized input state remain explicit producer
attestations, not verified bytes, CFL or conservation proofs.

This investigation exposed missing Long result metadata in synthesized region-copy
address additions. Retaining the known counted ordinal width lets the existing
range/coverage authority prove full-field commits; no coverage predicate was
weakened. Whole copies with nonzero source offsets pass, while shifted/short
destination copies remain partial. The prerequisite is PR #1130. The focused
warm-REPL copy/coverage/execution suites pass 14 tests / 58 assertions with no
failures or errors. The temporal facet is still under review and has not yet
demonstrated distributed runtime execution or durable publication.

### Bounded temporal cycle through distributed ownership

PR #1130 is merged as `7609bedb`; PR #1131 is merged as `5228aa99`.
Both landed only after exact-head independent review and all seven registered
CI contexts succeeded. The stage/execution facet is now an available structural
boundary, not a theorem of PDE correctness or a device-completion receipt.

The follow-up executes that exact certified cycle through the existing one-shot
distributed runtime on both OpenCL and Level Zero. One logical worker is mapped
explicitly to each physical target with a declared aggregate memory budget.
Two cycles run in separately prepared plans and fresh contexts, with synchronous
readback between them. Both match the independent coordinate-loop oracle within
the existing `1e-11` bound, preserve composite mass, and retain exact average-down.
Outputs are rejected before completion and a completed execution cannot replay
stale startup evidence. This is local co-location, not a measured network/fabric
result; copied readback inputs are not durable manifest publication.

The runtime probe found a shared borrowing-boundary gap: a source-backed private
temporary lost its caller initialization obligation when its source was removed.
The existing projection now expresses that obligation using the existing `:state`
role and `:requires` authority. Original initialization sources/views remain with
the enclosing owner and are uploaded before any borrowed local call. Source-free
private temporaries still require ordered producers; ownership alone never proves
bytes initialized. Tests cover owner-view coverage, no spurious caller requirement
after a first local overwrite, idempotence and missing-source rejection.

The borrowing/device run passes 19 tests / 137 assertions with both FP64 probes
available and no skips. The expanded borrowing controls separately pass 18 tests /
103 assertions. This follow-up still requires exact-head review and fresh CI.
General multi-patch/adaptive temporal planning, distributed completion-backed
publication, physical cost calibration and the held real-model numerical gate
remain separate open campaign obligations.

### One actual storage probe, distinct owner capabilities

`gpu.storage-representation/observe!` extracts the existing generated sentinel
probe's session execution and resource cleanup from `Compiled`. It canonicalizes
the dtype, checks actual live storage capability and quiescence, emits for the
session dialect, measures exact bytes, and rejects session/device identity drift.
Temporary graph/buffer cleanup preserves the first failure and attempts every
release. The result is unsealed historical observation data, never a completion
receipt, ownership capability or a fact transferable to another owner.

Its synchronous policy callback lets the existing compiled wrapper retain
exclusive mutation, output invalidation/poisoning, and its original sealed
same-owner representation evidence. Capability rejection precedes that mutation
callback. The observation thunk is same-thread, one-shot and cannot be retained
past that callback; quiescence and identity are rechecked before allocation.
No new store, cache, type registry or handwritten kernel is introduced.
Distributed ownership/evidence admission remains a follow-up, not an implied
consequence of observing bytes on a session.

The existing ownership/fault oracles plus raw-observation controls pass 10 tests /
175 assertions. Existing actual owner-bound evidence checks pass on OpenCL and
Level Zero (2 tests / 48 assertions), including foreign facts, live output leases,
cached module reuse and output retirement. Independent exact-head review and
fresh CI are still required.

### Captured AMR restore consistency before mapped bytes

The follow-up restore verifier composes the existing fixed numerical-state semantic
boundary with independently revalidated source CertifiedCycleExecution evidence.
Re-certifying a changed manifest cannot replace exact producer/temporal/completion
identity, ordered source fields, synchronized next-step lineage, numerical policy or
the complete plain FP64 storage encoding. Historical representation snapshots are
not fresh-target evidence or authentication of serialized producer claims.

The hardware-free verifier controls pass 2 tests / 11 assertions. The native
continuation oracle is extended to check restore compatibility before opening bytes,
materialize the bounded captured chunks as local files, independently verify their
content through real read-only mmap leases, decode into fresh host startup inputs,
then execute a separately instantiated next cycle. This is explicit fixture
materialization, not a new production storage provider or a power-loss durability
receipt. Production provider availability finalization and metadata publication
remain application-owned, using the existing generic finalizer.

### Completion-backed bounded distributed AMR capture

The next item-8 slice connects the existing CertifiedCycleExecution to an original
completed distributed owner and the existing NumericalStateManifest/ContentProvider
contracts. It revalidates the exact compiler plan, pins its own output scope, and
captures the two complete fields with bounded element-aligned downloads and SHA-256.
The verified local domain explicitly relates a flat physical array to its dense
patch shape; identical element counts and byte extent do not license arbitrary
striding, packing or partial coverage. Storage evidence is checked before every
download and after the final provider callback.

The captured manifest retains field geometry, numerical policy and producer-attested
meaning, with an advanced synchronized coordinate and the input state's identity as
its parent. All manifest fields are validated before ingestion. Ingestion failure
returns no state but may leave orphan blobs. The existing availability finalizer
must independently verify/promote content before application-owned metadata
publication. Neither observation nor hashing proves the producer's PDE semantics,
initial geometry bytes or parent existence in a durable catalog.

Hardware-free fault controls and an actual OpenCL/Level Zero capture/close/decode/
fresh-context continuation oracle are under validation. The native test's bounded
in-memory provider is explicitly not evidence of real durable storage. A mapped
provider realization and publication/restore compatibility controls remain next,
alongside the broader multi-patch, adaptive and physical-calibration obligations.

### Distributed representation evidence and explicit probe budgets

The distributed owner now uses the shared in-process identity-witness factory,
with a private issuer distinct from Compiled. Copies and facts from another
owner domain do not authenticate. This removes duplicated sealing mechanics,
not the independent compiler or lifetime checks required by each owner.

After successful execution, the existing generated storage probe can run in
the exclusive output scope. Its private allocation must fit the retained
physical-target resident-buffer budget. Admission declines preserve completion;
failure after probe entry marks the owner failed before releasing the scope.
Evidence binds the exact owner, session identity, target, dtype and observed
device snapshot. Consumption rechecks those facts inside the actual read scope.
Historical dereference alone grants no live-byte or durable-publication authority.

Focused identity, ownership, budget, cleanup and failure controls pass 14 tests /
270 assertions. The actual OpenCL and Level Zero representation/AMR oracles pass
3 tests / 104 assertions without skips. Independent review and fresh CI remain
required. Completion-backed distributed checkpoint capture is the next boundary;
this slice does not fabricate a local CompletedEvidence or certify program identity.

### Distributed output lifetime before durable capture

PR #1132 is merged as `4c7dc400` after independent exact-head review and all
seven registered CI gates. Its two-cycle local distributed execution is not
yet a durable checkpoint publication boundary.

The next prerequisite is `distributed/with-output-values!`: a synchronous read
scope on the existing enclosing owner. Successful completion is required;
close from any thread is refused, and callback failure releases the scope.
User code runs without the owner monitor held, avoiding deadlock when a provider
worker attempts close while the capture callback awaits that worker.
Asynchronous transfers must finish inside the callback and only
copied data may escape. Direct session mutation is outside this contract. This
does not manufacture a sealed compiler receipt or verify stored content.

The hardware-free lifecycle checks pass 4 tests / 30 assertions. The existing
AMR device oracle now copies its field bytes inside this scope; device validation
and exact-head review are still in progress.

Distributed hardware precision remains a distinct campaign obligation. The
topology represents device descriptors, memory capacities and directed links
with bandwidth and latency. The simulator accounts for dependencies and shared
route links, plus explicitly stated transfer/compute serialization. Its route
cost is an optimistic cut-through estimate, not measured fabric performance.
Local device calibration can refine kernel descriptors, but measured cluster
routes, shared physical bottlenecks, contention/load validity and realization
of predicted asynchronous overlap remain open. Keep capability legality and
ownership proofs separate from uncertain performance estimates.

### Shared bounded element-to-byte capture adapter

`numerical-content/element-byte-reader` extracts the existing local resident
capture's alignment adapter for reuse by distributed owners. An explicit
synchronous element downloader fills bounded scratch; arbitrary provider byte
windows receive only their requested raw bits. Extents must contain whole
elements, each byte window is at most 64 KiB, and subtraction-based admission
rejects overflow-sized requests before any source contact. Zero-byte windows
perform no download. Caller lifetime/immutability/representation obligations
remain explicit; this does not infer endianness or authenticate a producer.

The existing local capture uses this shared implementation instead of maintaining
its own alignment arithmetic. Focused adapter, local capture and storage suites
pass 27 tests / 522 assertions, including exact bytes at unaligned windows,
non-power-of-two element widths, overflow/range rejection, download failure,
content hashes and provider cleanup. Reloading the protocol owner required
reloading its test provider implementations in the warm REPL; the initial stale
provider failures were not numerical failures. Actual local capture/restart
checks pass on OpenCL and Level Zero (2 tests / 34 assertions). Independent
exact-head review and fresh CI remain pending.

### One dtype authority for hardware storage widths

Hardware planning now uses `compiler.core.dtype/bytes-of`, rather than a
second private width table. Canonical aliases and Half storage therefore have
the same widths in planning and lowering. Width alone does not grant native
arithmetic support. The planning helper now rejects `:short` and `:i16`: those
are not compiler scalar dtypes, even though Half uses JVM short-array storage.
No surface dtype was added or removed. The focused hardware, roofline and
calibration-admission suites pass 18 tests / 154 assertions in the warm REPL;
full CI and exact-head review remain separate gates.

### Shared physical resident-root accounting

`distributed-plan/resident-storage-plan` now owns the finite distributed runtime's
existing root-pool projection. Runtime allocation and optional simulator capacity
reporting consume the same independently validated bindings, exact root contracts
and overflow-safe physical budgets. Repeated calls share identical allocation
roots; co-located workers retain separate shard buffers and can share private
weights. Remapped workers require an explicit aggregate target budget. Distinct
shards still cannot silently alias storage; a test fixture attempting that was
rejected and corrected without weakening admission.

This models owned LinkPlan roots retained until owner close, not total physical
peak memory. KernelGraph scratch, backend temporaries, host staging, driver
allocations, available VRAM and lifetime reuse remain outside its scope. Default
analytical simulation and certificates are unchanged. The helper may retain
source-bearing LinkPlans and is not a portable report or allocation authority.

Focused planner/storage/cost/compute checks pass 71 tests / 552 assertions.
Existing actual AD -> all-reduce -> SGD checks pass on OpenCL and Level Zero
(2 tests / 154 assertions), including runtime-versus-simulator budget equality.
These are warm-REPL results; fresh CI and independent exact-head review remain
separate gates. The original real-model training numerical acceptance remains
open and is not replaced by these small distributed fixtures.

### Canonical graph-owned temporary storage requirements

`kernel-graph-call/temporary-storage-plan` now projects declared graph scratch
through the existing graph/scalar preflight and extent algebra. Runtime
`temporary-specs` delegates to this projection, rather than having a separate
planning evaluator. Each byte extent is checked against signed 64-bit storage
before allocation; aggregate requirements use exact arithmetic. Existing
out-of-range element values still fail in the canonical extent resolver.

The model counts all declared temporaries until graph unbind. It excludes
external roots, backend temporaries, host staging, alignment/driver overhead and
available device memory. It does not infer live-range reuse or grant allocation
authority. Distributed root budgets do not yet include these requirements;
binding/dispatch lifetime-aware aggregation is the remaining integration work.

Focused graph-call, GEMM and SegOp checks pass 87 tests / 1,605 assertions.
Existing native inclusive-scan tests pass on OpenCL and Level Zero (2 tests /
14 assertions), comparing planned allocation identities/byte sizes/totals with
actual private buffers as well as numerical output. These are warm-REPL checks;
fresh CI and independent exact-head review remain separate gates.

### One compiler-owned stage-once binding plan

The bounded graph/carry-variant calculation now lives in
`emitted-parallel-program-call/preparation-plan`; the runtime delegates to it.
This is the existing staging algorithm, not a new loop interpretation: zero-trip
loops bind no body variant, preserved initial carries plus parity rotation need
at most three body variants, and changing induction scalars still explicitly
decline stage-once preparation. Program-wide shape scalars and local physical
scalar overrides retain their existing merge order. Pure planning exposes the
same graph/scalar/buffer tuples that runtime binding consumes.

Runtime preparation validates the call once through this entry before acquiring
resources. Stage-once declines now precede invalid resolver/executor diagnostics;
accepted program arithmetic and surface semantics are unchanged. Existing
cleanup ownership and reverse rollback remain runtime responsibilities.

This supplies a shared input to upcoming memory accounting: local prepared
variants coexist, whereas the current synchronous distributed runner releases
each local LinkPlan after its compute action. A physical scratch peak must not
sum all distributed actions or count only a default dispatch alternative.
No distributed scratch-capacity admission is enabled by this extraction.

Focused cleanup and staging checks pass 35 tests / 233 assertions, including
compiler-plan versus runtime-bind comparisons and unchanged rollback behavior.
Actual co-located AD/all-reduce/SGD checks pass on OpenCL and Level Zero (2 tests /
154 assertions). Warm-REPL checks do not replace fresh CI or the original
real-model training numerical acceptance, which remains open.

### Producer-local fusion legality analysis

A live real-model VJP preparation sampled in scalar-effect descriptor-key
construction exposed repeated producer analysis during vertical candidate
enumeration. The exceptional-conversion check depends only on the producer and
the current immutable program, but previously ran again for each consumer.
It now uses one lazy result per producer in one enumeration. The original guard
order is unchanged, and subsequent enumerations/fixpoint iterations rederive it;
there is no persistent cache, retained certificate or new admission authority.

The four-map regression reproduces six checks on the previous implementation
and three on the replacement, with unchanged candidate pairs. A fresh all-trapping
enumeration rejects all candidates, and a producer without prospective consumers
never forces its check. Existing checked-conversion completion-boundary tests
remain. Focused fusion, placement, matrix-input and epilogue suites pass 65 tests /
417 assertions in the capped REPL. This is reduced analysis-call evidence, not a
measured end-to-end compile-time or device-performance improvement.

The new real-model layer-1 isolation reproduces all 14 original CPU adapter
gradients exactly with the original forward input and loss cotangent. Its GPU
preparation did not complete: the warm process had retained an eight-argument
math-policy invocation caller while a later staging reload restored the
seven-argument program-call constructor. This is a mixed-revision diagnostic
environment, not evidence of a production miscompile or a new numerical result.
Rebuild a coherent compiler source set before continuing that comparison. The
original componentwise real-model acceptance remains failed and unchanged.
