# Local compiler evidence — 2026-09-27

## SIMD retained-operation precision — 2026-10-09

A surface `deftm` mapping a Float array with a Double scalar and the expression
`(+ (aget src i) (float (* gain gain)))` exposed a vector/scalar-tail disagreement.
At `gain=1.00000006`, zero inputs produced `1.0000001` in scalar execution and
the vector tail, but `1.0000002` in the vector lanes. The original walked multiply
is stamped Double; canonical SIMD arithmetic retained that stamp but ignored it,
broadcasting the operands into the Float species before multiplying.

JVM scheduled SegMap admission with an active species now checks retained floating operation result stamps against the
active species, using the existing dtype projection rather than reconstructing
types from operator names. Mixed-width computations retain scalar execution until
an explicit mixed-species lane-shape schedule exists. Homogeneous typed operations
remain admitted. The executable regression uses two lengths and changed scalar
inputs, with an independent Double-then-Float reference; admission controls cover
both narrowing and widening boundaries. This does not establish real-model training
acceptance or complete mixed-width vector support.
At that checkpoint, JVM stencil and C-SIMD callers without an active species,
reduction-root operation precision, and typed comparison domains remained follow-ups. A
public native-C execution of this fixture also returns the correct result, but does
not establish that its vector admission applied this guard.

The follow-up separates retained floating precision obligations from emitter syntax
admission. Stencil now carries the active species, while C map and reduction schedules
use the same precision predicate without losing their supported integer-widening path.
C map checks canonical conversion facts before projecting them back into source casts.
Both reduction emitters check the complete retained recurrence, not only the extracted
element. Positive and negative hardware-free admission controls cover these consumers;
the affected suites, including executed native C integer widening, pass 53 tests and
207 assertions locally. Missing result stamps are not inferred, and Boolean-result
comparison operand domains remain unproven. This is not complete mixed-width SIMD
lowering, a general numerical equivalence proof, or actual-weight training acceptance.

A subsequent public Float-map probe comparing Double scalars `1.00000006` and
`1.00000007` returned 1 in scalar code and the tail, but 2 in all 64 vector lanes.
The two scalars collapse to one Float; a Boolean result stamp alone cannot certify
the operand domain. Canonical SIMD calls now retain parameter tags from the selected
declared interface, using existing function-info metadata and the shared interface
index, not parsing implementation names or adding a registry. Compare-and-blend
admission checks that signature as well as the operands. Missing typed signatures,
arity mismatches and mismatched integer/floating comparison domains retain scalar
execution. Untyped compatibility syntax remains explicitly outside that proof.
Changed-input execution and positive/negative admission controls bring the affected
suite to 55 tests / 218 assertions locally. Explicit mixed-species lowering and
type-complete removal of untyped compatibility remain follow-ups.

## Scalar non-contraction investigation — 2026-10-08

Separate ScalarCompute multiply/add nodes are not sufficient evidence of two machine-level
roundings. The generated decomposed register-contraction fixture, compiled with the local
`nvcc -ptx -arch=sm_80` defaults, contains 256 FMA instructions. Explicit fused realization
remains a separate compiler policy; neither realization is accepted as a replacement for
the unchanged real-weight training oracle.

A small independent HIP compilation probe tests three operations over resident pointers:
ordinary multiply followed by add, `__fmul_rn` followed by `__fadd_rn`, and explicit `fmaf`.
The translation unit sets `#pragma clang fp contract(off)` after the HIP runtime header.
With local HIP 5.7.31921-9999 / Ubuntu Clang 21.1.2, `--offload-arch=gfx1100
--cuda-device-only -S -O2` produces separate `v_mul_f32_e64` and `v_dual_add_f32` for
ordinary arithmetic, but `v_fmac_f32_e64` for both named-RN and explicit-fused cases.
The installed HIP header defines the named multiplication as plain `x * y`; its previously
parsed helper body retains LLVM `contract` flags despite the later pragma. Merely borrowing
CUDA's intrinsic names would therefore not preserve the same contract on this HIP toolchain.

Device LLVM output (`-S -emit-llvm`) also shows this distinction, both with defaults and
with an explicit `-ffp-contract=fast`: ordinary arithmetic has unflagged `fmul`/`fadd`,
whereas the named helpers retain `contract`. That local pragma behavior must not be promoted
to a portable guarantee: [Clang documents that `fast` can override controlling pragmas](https://clang.llvm.org/docs/LanguageExtensions.html).
[NVIDIA documents non-contraction for its rounded multiply](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rn.html);
[OpenCL provides its own FP_CONTRACT control](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/FP_CONTRACT.html).

The follow-up protects Float/Double scalar products in the shared target lowering: CUDA's
rounded multiply and AMD HIP's empty read/write VGPR asm fence preserve the intermediate
product; OpenCL uses FP_CONTRACT OFF. The HIP fence changes no product bits, has no memory
clobber or volatile spill, and allows unused results to be eliminated. It requires AMD HIP,
not a claim about other HIP platforms. Helper discovery is demand-driven and shared with
matrix scalar epilogues. Explicit canonical FMA and matrix multiply-accumulate instructions
remain explicit. Controls are entirely in the emitted source, already part of artifact/cache
identity; there are no additional flags to lose or a second numerical-policy registry.

The new shared Float/Double nested product/add and explicit-FMA fixtures compile through
both hardware-free vendor instruction checks. CUDA emits separate rounded multiplication
for the decomposed cases and FMA for explicit cases. HIP emits separate multiplication/add
even with `-ffp-contract=fast`, and explicit FMA remains fused. These checks are added to
the existing vendor CI gates. Both multiplication and addition must be present for decomposed
cases. Review removed a redundant global HIP contraction pragma: only Float/Double products
are protected, avoiding unvalidated global changes to unrelated HIP arithmetic. The affected
scalar/matrix source suites pass 34 tests / 474 assertions. The existing bit-sensitive public
register replay oracle passes 1 test / 36 assertions on OpenCL and Level Zero, including
changed inputs and both product policies.

This is bounded Float/Double product non-contraction evidence, not proof of all floating-point
properties, CUDA/HIP execution, instruction throughput, or real-model training acceptance.
Half scalar arithmetic, arbitrary fast-math reassociation/denormal settings, and target-library
transcendental accuracy are not made exact by this change. The unchanged real-weight training
gate still needs its matrix association and full cotangent differences localized.

## Public training and resident GEMM checkpoint — 2026-10-06

PR #1054 (`e57ba203`) migrates the existing mixed Gemma fixture onto public equation-first
lowering, instantiation, replay and close, removing its descriptor/session helpers. The local
small FP32 fixture passes 43 assertions with matching independent CPU/GPU trajectories. The
25-step mixed fixture passes 35 assertions: all 50 eligible contractions select mixed matrix
execution (21 NT, 15 NN, 14 TN), planned and bound graphs agree, and preparation allocates no
driver buffers. FP32 loss decreases from 2.281172 to 1.261994; mixed ends at 1.262064, within
the existing per-step relative tolerance. The combined two tests have no failures/errors.
This is small-block training acceptance, not external real-weight training or throughput.
All seven exact-head CI gates passed before #1054 and its proof-consolidation prerequisite #1057
were squash-merged. Focused proof-consolidation checks pass 77 tests / 620 assertions.

The existing `raster.perf.production-canary/equation-gemm!` then checks `[64 64 64]` with
explicit portable and register-tiled FP32 schedules on both local backends. Every run passes
its independent exact dyadic oracle before and after measurement and emits one kernel.
Each uses a 300 ms device-event measurement budget; compile, bind, transfers and validation
are outside the samples. Environment label: `local-arc-shared-load-power-save`; source tree
was the reviewed `a436a891` tree, identical to the rebased #1054 tree.

| Backend | Requested schedule | Median ns | CV | Stationary heuristic | Samples |
|---|---|---:|---:|---|---:|
| OpenCL | portable | 48,333 | 0.102 | no | 2,920 |
| OpenCL | register-tiled | 29,687 | 0.061 | yes | 7,164 |
| Level Zero | portable | 54,271 | 0.033 | yes | 5,581 |
| Level Zero | register-tiled | 33,125 | 0.039 | yes | 10,000 |

These are separate short runs on one shared-load device, not alternating-round measurements,
an external BLAS baseline or general performance acceptance. In particular, the OpenCL portable
series does not support a stationary comparison. No baseline or schedule default is promoted.
Reproduce with the existing canary, holding shape, numerical policy and environment fixed;
keep raw chronological samples and use the matched-round comparison protocol for promotion.
External package migration, real-weight chained training, larger shape ladders and competitive
cross-system measurements remain required by campaign items 3–5.

The next unchanged external two-layer forward/loss-seed/VJP graph prepares through public
composition in a clean `9e9ba5d` finetune snapshot. A cold thread sample inside emission exposes
wrapping-collective helper discovery traversing all nested record collections, including retained
proof/source attributes. Discovery now reuses the existing executable-region walker and receives
operation roots once, rather than recursively revisiting an already flattened list. Both If arms,
ForLoop, PipelinedFor, Guard and both While regions are covered; collectives are not legal inside
ScalarExpr operands or matrix store scalar regions. The before-fix regression reaches irrelevant
proof data; after the fix the full scalar emitter suite passes 24 tests / 352 assertions.
This is traversal deduplication with unchanged helper/emission coverage, not a measured compiler
speedup. Scalar-expression discovery and independent KernelBody validation remain unchanged.

### External two-layer resident loss-seed/VJP acceptance

A clean local finetune snapshot at `9e9ba5d62f3822f056e01c37231d7eaa7c84947c` supplies unchanged
`gblock-fwd!`, `gblock-vjp-step!` and the original monolithic `ref-loss2` oracle. Only numerical
declarations and pure data/argument helpers are loaded; the retired session adapter is excluded.
The dirty sibling checkout is untouched. This is an isolated adapter probe, not full package
migration or upstream evidence: that local finetune repository has no configured remote.

Public composition orders two forwards, existing `mse-grad`, then two VJP/update programs in
reverse order. Its numerical call counts are `[60 60 1 157 157]`; 28 donated adapter owners share
live state with the corresponding forward borrowers. Activation fanout, loss seed and chained
input cotangents are internal connections, and their five consumer inputs are absent from the
public refresh interface. Frozen weights/norms and the initial input use ordinary shares.
The external reduced-width configuration and original random-data seeds are retained. The
resident loss seed is computed before either update, not supplied from the host.

Two consecutive Level Zero replays compare loss, all 28 recovered adapter gradients (`lr=1`)
and the final input cotangent against the original CPU `value+grad(ref-loss2)` at each pre-step
state. Host downloads are exclusively oracle reads, not component-to-component bridges.

| Replay | GPU / CPU loss | Absolute loss error | Worst relative dx error | Largest adapter worst-relative error |
|---|---|---:|---:|---:|
| 0 | 4.602434635 / 4.602435112 | 4.77e-7 | 4.10e-4 | 1.85e-3 |
| 1 | 4.237384319 / 4.237384319 | 0 | 1.65e-3 | 2.32e-3 |

The next forward therefore observes updated resident adapters, not the captured initial host
arrays. The existing external `2e-2` worst-relative and loss tolerances are unchanged; previous
output handles invalidate on replay. Cold preparation was 238 seconds in the capped REPL,
including compilation and composition; this is a preparation-cost debt, not device throughput.
The same full chain also executes on OpenCL with identical reported errors across both
replays. Preparation took 230 seconds and binding 86 seconds under shared background load;
these are observational costs, not a controlled performance comparison.

The opt-in runner lives in `test/raster/acceptance/finetune_chain.clj`. In an isolated JVM,
with a clean trusted checkout containing the pinned Git object, run:

```sh
clojure -J-Xmx1800m -J-Xss8m -M:dev -m raster.acceptance.finetune-chain \
  '{:source-root "/path/to/finetune-rstr" :targets [:ocl:0 :ze:0]}'
```

It compares source files against the exact pinned object with Git replacement objects disabled,
disables reader evaluation, preserves source line metadata, rejects existing external namespaces,
and retains the original numerical and oracle definitions. Array lengths are checked before the
external relative-error helper, which otherwise truncates comparisons. Ordinary CI runs only
the hardware-independent loader/option/shape checks (7 tests / 20 assertions); it does not need
the external checkout or treat native acceptance as a conditional pass.
Real-weight/head training and the canonical external adapter migration remain open.

## Consolidation diagnostic — 2026-10-01

The existing public equation-first GEMM canary executed `[32 32 32]`, strict FP32,
`:portable`, on local `:ocl:0`. Independent dyadic-reference comparisons before and after
measurement passed; one generated kernel was retained. The explicit revision label was
`d96a29e1+branch-lifetime-candidate`, with varying background load recorded in the environment
identity. Preparation was 1.524 s and binding 1.780 s (host monotonic clocks). The device-event
sample median was 38,229 ns over 91 samples, CV 0.439: **nonstationary**. Transfers and numerical
validation were outside the timed span. This is an executed correctness/measurement-path
diagnostic, not a baseline, throughput claim, comparison against BLAS or schedule promotion.

The proof audit separately reproduced a source/JVM branch-alias reuse failure (original 11.0,
optimized 33.0). The conservative follow-up tests both branches, nested let/loop/do results,
unknown-call returns, transitive aliases, escaped array sizes and unrelated reuse. GPU memory
reuse evidence does not supersede this distinct source/JVM analysis.

The KernelCall admission consolidation passes 32 tests / 144 assertions across direct and graph
call validation, plus 27 tests / 154 assertions across dispatch benchmarking and public compiled
composition. Each public call/precondition/launch entry validates its artifact afresh; its
locally checked precondition helper no longer repeats that artifact/argument validation or
scalar representation checks. Construction still checks before geometry realization and again
at final call admission. Malformed vectors/counts, scalar ranges/specializations, guards and
pointer/geometry contracts retain their existing failure tests. This is a synchronous duplicate
check removal, not a retained proof cache or measured preparation speedup.
The existing native public composition suite adds 9 tests / 88 assertions on actual OpenCL
and Level Zero, including replay and shared forward/update state. Both availability gates were
explicitly checked true after the run; passing a conditional suite alone is not execution proof.

A paired changing-activation GEMM/ReLU diagnostic uses the existing public comparison runner,
strict FP32 `[8 64 64]`, twelve rotating measurement rounds and four warmup rounds. Both candidates
pass the exact independent reference after every poisoned-output replay on both local backends.
Ordinary composed source emits two resident stages; explicit epilogue source emits one. The
OpenCL medians were 35,521 ns and 23,750 ns with CV 0.305 and 0.492 (both nonstationary). Level Zero
medians were 34,166.7 ns and 31,979.2 ns with CV 0.00428 and 0.00335. The latter satisfy the CV
heuristic only; a short sample on the same physical Arc under varying load is not broad performance
acceptance. No baseline or automatic schedule is promoted. The composed form defines a checked
extent after the contraction's writes; existing fusion tests deliberately preserve that boundary.
The prebound-extent form was subsequently tested with the same rotating replay protocol. Both
ordinary composition and explicit epilogue emit one stage on both backends and pass the exact
oracle on every replay. OpenCL medians were 40,104 / 40,104 ns (CV 0.376 / 0.200); Level Zero
medians were 37,083 / 54,479 ns (CV 0.443 / 2.27). All four series are nonstationary. This validates
the existing legally dominated fusion case, not a speedup or permission to hoist checked extents
across writes.

The existing cooperative RMSNorm canary `[1 640]` also passes its independent numerical oracle
on both backends. OpenCL preparation/binding were 1.386 / 0.514 s and device median 3,229 ns
(342 samples, CV 0.585); Level Zero preparation/binding were 0.592 / 0.0468 s and median 12,812.5 ns
(391 samples, CV 0.168). Transfers and validation remain outside device-event timing. Both
series are nonstationary despite passing the deliberately loose advisory roofline cliff bound;
neither is stable performance acceptance or grounds for schedule promotion.

The next normalization audit found a public AD semantic mismatch: a nested let's unused checked
array read threw in the primal but vanished from `value+grad`, which returned `[4.0 4.0 nil]`.
The regression now requires both paths to throw on an empty observation array and compares the
valid primal/gradient with constant observations. Ordered body-statement, sibling-argument and
first-exception oracles separately cover the hoister; lexical activity tests cover captured and
shadowed names. These preserve evaluation semantics without promising differentiation of effects
or expanding general-loop support.
The quoted `grad-expr` API separately returned gradient 3 instead of 7 for a loop seeded by
`(let [local x] (* local local))` at x=2; `deftm` gave 7. Newly lifted initializer prefixes now
pass through that same ordered projection before activity/pullback generation. Surface and
macroexpanded quoted forms match `deftm`, analytic derivatives and finite differences for both
zero-trip and three-trip loops. The focused AD suites pass 100 tests / 376 assertions, and
typed emission passes 9 / 81. Actual Level Zero RMSNorm resident gradients also pass 1 / 5,
with relative errors 1.17e-7 for x and 8.05e-8 for weights; its availability gate was true.

## Current pretrained consumer and native batch layout — 2026-09-30

An isolated snapshot of pretrained-rstr main `98bad4d3517ec05a3c495f019c80d8c553db02af`
pins Raster 0.2.922. Its Laya CPU-reference test exposed a native oneMKL boundary bug:
the strided-batch output-span requirement applies even when batch=1. The old exemption
sent one-head attention output to a rejected FFI call (oneMKL parameter 17), leaving the
destination unchanged and producing a head mismatch of 0.002937. Independent GEMM
already supports the same view. Removing the exemption selects that existing fallback;
there is no new algorithm, numerical policy or GPU specialization.

The corrected local source passes `raster.linalg.blas-test` (17 tests/67 assertions),
the external decoder composition tests (2/8), and
`pretrained.laya-gpu-test/resident-head-preserves-cpu-laya-semantics` (1/1). The regression
checks exact NN/NT results, nonzero offsets and untouched sentinels. A throwing delayed
optional handle also proves that the invalid descriptor never reaches the extension on
machines without MKL. Existing valid batch layouts retain their extension path.

The actual external `laya-resident-first-head` fixture (2 tokens, width 64, FFN 256,
one head) lowers through the public compiled API, instantiates, replays twice, downloads
its sole semantic output and closes on both local OpenCL and Level Zero. Each replay's
128 elements match the uncompiled JVM source with maximum absolute error
2.814456820487976e-6 (fixture tolerance 1e-5). This is small inference correctness evidence,
not real-weight validation, a timing comparison, full encoder composition or training acceptance.

The validation REPL uses an explicit `:deps` local-root for Raster in `-Sdeps`, not a
top-level `:override-deps` entry (which does not override this consumer dependency).
`clojure.java.io/resource` confirms the candidate's source path before acceptance.
Initial runs accidentally used the released artifact; they established the released
failure only and are excluded from candidate evidence. No sibling checkout was changed.

The larger external synthetic composition also passes on both local backends: one
ModernBERT encoder block → final norm → two Laya decision heads, instantiated through
`compile-decision-hidden` and replayed twice through `decision-hidden`. Its 128 elements
match the chained uncompiled JVM source with maximum absolute error
8.64267349243164e-6. The external ModernBERT/Laya focused suites pass 8 tests/80 assertions.
Weights are small synthetic tensors; model-loading, large contexts, performance and real
checkpoint acceptance remain separate.

## Certified shared forward/update state — 2026-09-30

`compiled/compose` now accepts explicit `:mutable-shares`: one donated owner, read-only
input borrowers and exactly one selected final owner output. The same source object initializes
one full owned allocation. Ordinary shares/connections cannot claim those endpoints, constants
cannot become mutable borrowers, and escaped borrowed/old outputs are rejected. The composition
certificate retains these obligations and revalidates remapped effect evidence. The composite
removes borrower refresh slots and retains the owner's donation map; it uses the existing linked
replay, output invalidation and leases, not a new runtime or cache. Omitted donation handles retain
the existing internally persistent-state behavior.

The focused low-level, semantic/preflight and native composition suites pass 35 tests / 243
assertions. Both local backends execute a generated forward/read → SGD/update graph repeatedly
with exact JVM parity; a live output lease rejects replay before mutation. Invalid requests fail
before allocation. Missing/duplicate owners, competing immutable/dataflow claims, different
initializer objects, hidden final outputs, constant borrowers and escaped old aliases are covered.

The isolated unchanged finetune-rstr `9e9ba5d` tiny Gemma declarations also compose through this
public boundary: 60 forward stages followed by 157 VJP/SGD stages, 14 mutable adapter bindings,
shared frozen weights/norms and a shared input. On OpenCL and Level Zero, two replays match the JVM forward
and independently differentiated pseudo-loss/SGD reference. Maximum errors are 1.20e-6 for
forward, 2.87e-6 for input cotangent and 2.99e-8 for adapters. The next forward sees updated
adapters, not their captured initial host arrays. The fixed external output cotangent is supplied
separately; this is not a fused loss-seed graph. Downloads serve only numerical comparison.
Both backends produced the same reported errors. All seven CI gates passed and the slice
squash-merged as #950 (`6533eed1`).

Host-monotonic preparation samples under background load were 70.2 s forward, 287.2 s VJP and
89.6 s composition. These are an explicit preparation-cost debt, not a speedup or throughput
measurement. No sibling source/dependency is modified. Full package migration, real-weight or
longer training, ranged/cross-executable state sharing and asynchronous lifetime proofs remain
separate gates.

The preparation follow-up instruments a fresh composition of the retained exact Gemma components,
without recompiling source or allocating device buffers. Total time is 78.35 s; inclusive
`namespace-instance` time is 50.03 s and final LinkPlan construction is 27.97 s, of which plan
structure validation is 27.92 s. Mutable ownership normalization, access/initialization checks
and alias validation together account for less than 0.1 s in this sample. Inclusive nested times
must not be added. The next narrow deduplication retains independently validated immutable
programs inside buffer remapping, then rechecks all remapped steps/bindings through the existing
call validator. A later public validation and later construction remain independent; no proof
cache, kernel-body change or ownership relaxation is introduced. The input already receives
independent complete validation through `buffer-identities`; removing the additional entry and
unchanged-output program checks reduces root-program validation from three to one per remap.
The focused test checks that count, fresh later validation/construction and rejection of an invalid
source program or malformed source call before any mapper call. Existing totality, alias/collision
and loop-buffer tests remain. The affected suites pass 73 tests / 484 assertions, including local
OpenCL/Level Zero composition replay, and focused review found no blocking proof-boundary issue.

The same retained-component profile after this change reports 45.75 s total: 28.90 s remapping
and 16.49 s final plan construction, with an exactly equal resulting plan (247 nodes / 2 instances).
Background load, warm-up and the unchanged final validator's different timing prevent treating
78.35 → 45.75 s as a controlled speedup. The proven improvement is removal of two redundant root
program checks, not a kernel/runtime improvement or a general compilation-latency guarantee.

The next remapping follow-up reuses only the exact immutable equation boundary's already-checked
physical-result projection between source and final call validation. One fresh identity context is
bound separately around those two synchronous phases; mapper callbacks and rename construction
cannot inherit it. No context is retained in a call or shared with later public validation. Copies,
metadata changes and changed bodies miss the identity proof; every call/ABI/graph/scalar/result-view
check still runs. Focused review found no blocker. The affected suites pass 75 tests / 493 assertions,
including fresh-validation, mapper isolation, copied/changed boundary and native replay checks.

The same actual Gemma composition retains an exactly equal 247-node / 2-instance plan. All 651
equation-call checks still run, while physical-result queries fall from 651 to 434. An instrumented
sample reports 29.17 s total, 14.08 s remapping and 14.84 s final plan construction. This is not a
controlled speedup comparison; independent final plan validation remains intact. The count reduction
and unchanged plan are the proof, not the sampled timing. All seven CI gates passed; this slice
squash-merged as #952 (`c7d54f84`).

### Executable graph validation deduplication — 2026-09-30

The next shared preparation cleanup removes the immediate `KernelGraph/validate!` call
before `KernelExecutable/validate!` in emitted-equation validation. The latter already calls
the former, then checks the external ABI, scalar dependencies, artifact bindings and targets.
The two independent dataflow-contract projections remain. A regression counts three exact
emitted-graph checks instead of four per equation validation, repeats the public validation
to prove freshness, and checks malformed graph rejection. No retained proof or cache is added.

The affected equation/semantic-composition/public-composition suites pass 35 tests / 180
assertions. Native composition tests add 9 tests / 88 assertions on both OpenCL and Level Zero;
availability is explicitly asserted afterward, not inferred from a passing suite. All checks
reuse the existing source-verified REPL. The retained external Gemma profiling attempt exceeded
its 60-second diagnostic limit under background load and was interrupted; it supplies no timing
or speedup evidence for this change. Full CI acceptance remains separate.

That slice passed all seven CI gates and merged as #953 (`6fc1c79c`). The graph-call follow-up
continues the same consolidation: executable validation derives target membership from the
node artifacts it just validated rather than calling the independently validating public
artifact accessor; scalar-range projection delegates graph validation once; alias preflight
uses a private hazard projector after its own executable check. Public alias entry points
still validate independently, incomplete bindings and invalid overlap predicates still fail,
and single artifacts remain rejected by the graph-only projections. No proof context or cache
is added. Each affected public graph entry now checks its exact graph once; emitted-equation
validation additionally retains its independent dataflow comparison.

The affected graph-call/equation/composition suites pass 55 tests / 315 assertions, including
native OpenCL and Level Zero composition. The same retained external Gemma components still
compose to an exactly equal 247-node / 2-instance plan: 651 equation-call checks and 434
physical-result queries remain. A 46.15-second background-loaded instrumented sample reports
24.44 seconds in program validation and 20.19 seconds in physical-result derivation (inclusive
times overlap). It identifies semantic reconstruction as the remaining preparation cost;
it is not a speedup comparison. The REPL stopped accepting connections before a subsequent
full external replay could start; that attempt contributes no new numerical evidence. Full
CI acceptance of this follow-up remains pending.

The graph-call follow-up passed all seven gates and merged as #954 (`9396876e`). A clean
2.2 GiB-capped REPL repeated the 55-test / 315-assertion focused suites with both device
availability gates asserted; graph ordering/runtime/refinement added 21 tests / 119 assertions.

### Exact plain-equation facts across program and call validation

The next slice factors physical-result derivation into the equation validator's returned report.
The enclosing program validator retains those facts in a fresh exact-object identity index while
performing all ordinary semantic, host-prefix and target-module checks. Call validation consumes
that local index rather than reconstructing each plain equation a second time. Every step, ABI,
scalar, result view and binding is still checked. A later public call starts with a fresh complete
program validation and never borrows an inherited index. Only successful validation may publish
its derived projections into a synchronous enclosing rename. No index is stored in the resulting
call or compilation template; no persistent proof cache, type inference or AD rule is introduced.

Dispatch candidates and structured loops keep their existing validation paths. Plain call
construction itself still has its earlier validation sequence; the remaining constructor
duplication must be addressed separately without conveying facts to host-evaluator callbacks.
This slice changes no kernel arithmetic, schedule, ownership or runtime event behavior.

The mixed structured-loop/plain-equation fixture now checks its exact plain boundary once rather
than twice during source validation plus remapping, with one fresh check on each later public call.
Regression cases retain mapper isolation, equal-but-distinct boundaries, changed bodies and failure
before remapping; a failed call publishes no facts, and a public call replaces rather than trusts
an inherited projection. Focused control/equation/composition/native suites pass 85 tests / 527
assertions, dispatch/persistence/numerical-policy checks add 10 / 72, and selected public
mixed-target/exceptional-host-prefix checks add 3 / 12. Full CI and new external model/timing
acceptance remain separate; earlier Gemma timing samples are not measurements of this slice.

All seven CI gates passed and this program/call slice merged as #955 (`308bec35`).

The constructor follow-up consumes that same fresh report before any host evaluation, then passes
the identity index explicitly to private numerical-step preparation. It never binds the index
around the host-evaluator callback. Exact plain boundaries therefore avoid their second derivation
in construction as well; dispatch boundaries keep their previous independent projection. Runtime
scalars, preconditions, result views and every generated step still receive their existing checks.
The mixed fixture proves one plain-boundary check in construction, unchanged buffers and outputs,
no retained projection field and a fresh later public check. The host-scalar fixture checks that
the callback has no constructor projection context and that an invalid program fails before any
callback runs. The affected control/equation/composition/native suites pass 86 tests / 537
assertions. Dispatch/persistence checks add 10 tests / 72 assertions, and selected public
mixed-target/exceptional-host-prefix checks add 3 / 12. Both local device availability gates
remain asserted. No new external Gemma timing or full-package migration is claimed; full CI
acceptance remains pending for this constructor follow-up.

The first constructor CI run passed the six non-test gates and found four stale count assertions
in the existing staged CUDA/HIP contraction regression: each constructor now checks its exact
plain boundary once, not twice. The test retains result-view, independent step-validation and
forged-binding coverage, and now explicitly checks that later public call validation performs a
fresh boundary check. This is an assertion update for the intended preparation change, not a
relaxation of numerical or binding contracts.
The repaired staged-contract test passes 1 test / 64 assertions, followed by all seven green
final-head gates. The constructor slice squash-merged as #956 (`4efdaab6`).

### Fresh external preparation checkpoint

With the constructor follow-up and its CI assertion repair (`1993c9bb`), the same isolated
committed Gemma source is freshly lowered for the registered OpenCL target: 60 forward stages and
157 VJP/update stages, with 14 shared mutable adapters. On this shared-load laptop, host preparation
reports 38.13 s forward, 189.12 s VJP and 14.70 s composition. A separately instrumented
recomposition of those retained components takes 19.15 s and asserts exact resulting-plan equality
(247 nodes / 2 instances). These are single warm/noisy samples, not controlled speedup evidence.

The trace retains 651 independent step/binding validations and 434 emitted-boundary validations,
all through fresh program validation reports. Artifact validation totals 3,906 calls / 0.353 s;
emitted-boundary validation accounts for 18.00 s inclusive. Nested timings overlap and must not be
added. This directs subsequent work toward semantic reconstruction, not device kernels or alias
checks, and does not justify weakening any public validation boundary.

The freshly prepared OpenCL composition also replays twice against independent JVM forward,
input-gradient and adapter-update oracles. Maximum errors are respectively 1.193e-6, 2.862e-6 and
2.981e-8. This renews tiny-model correctness acceptance on the constructor head; it is not a
fresh Level Zero external replay, real-weight training, throughput or a full package migration.

The next bounded cleanup projects plain/dispatch artifact target tags after their executable
validator has already checked those exact artifacts. Structured loops retain the independent
artifact check: their graph validator alone does not check arbitrary artifact descriptions,
especially the older operation-certificate form. Later public program validation remains fresh;
mixed-target and malformed-source rejection remain required acceptance tests.
The focused suites pass 73 tests / 526 assertions: 43 / 275 structured-control checks,
28 / 185 independent equation, dispatch/persistence and public native-composition checks,
and 2 / 66 staged CUDA/HIP and mixed-target checks. Both local device availability gates
are asserted, so native composition acceptance does not come from a skipped test path.

A deeper trace with that cleanup preserves the exact same 247-node plan and reports 14.61 s
recomposition, 651 step validations and 434 emitted-boundary validations. Artifact validation
falls from 3,906 to 3,472 invocations; that count reduction, not the noisy timing change, is the
verified result. The trace exposes 48,740 SOAC validator invocations (9.96 s inclusive) and
11.67 s inclusive expected-graph reconstruction. Multi-arity/self-recursive instrumented entries
can count nested calls; inclusive times may exceed total wall time. The next consolidation audit
is typed semantic projection inside one checked boundary, not removal of independent outer proofs
or a new persistent validation cache. External adapter migration and matched kernel baselines
remain on the existing campaign.

The bounded target-projection cleanup merged as #957 (`41936aa4`) after all seven final-head
gates passed. An exact-identity census then counted 48,740 SOAC validations over 352 distinct
program objects in the same recomposition. Some single-equation scalar-prefix programs were
checked 628 times. This does not make every repetition redundant: separate public graph/program
validations retain independent proofs.

One immediate duplication is wholly private: scheduled graph construction validates its entire
body with `algorithm-boundary?`, then host-prefix projection validates those same algorithms
again. The follow-up removes only that second validation and makes the projector's precondition
explicit in its private name. Scalar result count, dtype, shape and capture checks remain. The
existing scalar-gap fixture checks identical graph output, one exact prefix-algorithm validation
per construction, fresh later construction, malformed semantic inputs and invalid scalar shape.
No validated context, callback convention, persistent cache or new IR is introduced.
Focused acceptance passes 88 tests / 594 assertions: 13 / 58 scheduled-graph checks,
71 / 460 control/equation/dispatch/native-composition checks and 4 / 76 staged vendor,
mixed-target and checked-prefix exception-order checks. Both local device probes are asserted
before native composition tests.
The exact-identity Gemma census after this private cleanup retains the same 247-node / 2-instance
plan and 352 distinct semantic program objects, while SOAC validator invocations fall from
48,740 to 24,804 (23,936 redundant invocations removed). Instrumented recomposition reports
10.74 s under the current background load. The count and identical plan are reproducible
mechanism evidence; the single timing sample is not a controlled latency-speedup claim.

#958 merged as `d7ce1a3d` after all seven final-head gates passed. A refreshed census retains
24,804 validations over 352 exact programs; the most repeated scalar-prefix programs contain
only two or three declared values. This is not evidence that their contexts should be truncated.

### Dispatch result-contract certification

The next consolidation keeps one private boundary-validation report containing its freshly
reconstructed source graph. The public contraction full-write query uses that same graph rather
than deriving it twice. A dispatch-local result-contract report then derives physical result
mapping and independently proved complete-write domains from one checked candidate. The source
graph is not retained in the returned report, call, dispatch or artifact. Public equation validation
keeps its unchanged return value; independent later public queries and dispatch validations derive
fresh facts. No numerical policy, kernel, target admission, ABI, lifetime or persistent cache changes.

This removes repeated candidate certification in dispatch, not proof obligations: truncated stores
still fail full-write admission even when their ABI and ordinary emitted boundary are valid;
register-tiled candidates cannot self-label as exact; exact-only permission still rejects
reassociated schedules. Contraction and protected indexed-reduction tests share one proof-count
helper rather than duplicating fixtures. Existing graph/storage agreement, policy, default and
artifact round-trip checks remain. The fixed-order external Gemma fixture has no such adaptive
dispatch; this slice does not claim to reduce its validator count or device runtime.

With the final report-shape assertions, focused acceptance passes 81 tests / 642 assertions:
71 / 486 equation/dispatch/control/native-composition checks, 4 / 52 actual indexed-dispatch
replays on both local backends, 4 / 76 public staged vendor/mixed-target/exception-order checks,
and 2 / 28 ragged generated GEMM and mixed attention/projection device oracles. Availability and
the actual OpenCL score-reuse capability are asserted; these native results are not skip paths.
The retained fixed-order Gemma recomposition still asserts exact 247-node / 2-instance plan
equality (one shared-load sample: 10.78 s). Reports and count regressions establish removed
proof duplication, not competitive GEMM throughput or a controlled compilation speedup.
The final two affected report fixtures were rerun serially (2 tests / 48 assertions), checking
identity, storage/full-write agreement, source-graph non-retention and fresh public validation.

All seven final-head gates passed; #959 squash-merged as `9a98c2c1`.

### Matched GEMM evidence and benchmark guards — 2026-10-01

The next local acceptance slice reuses the existing strict-FP32 public GEMM canary and
CLBlast comparison rather than adding another compiler or benchmark path. Both oracles
now explicitly reject nonfinite output: a NaN must not escape a maximum-error comparison.
The Raster harness validates shape/work bounds before device initialization, validates
finite positive event durations before integer conversion, and uses the existing measurement
summary rather than introducing a second stationarity rule. Its opt-in host-only checks
pass 3 tests / 26 assertions; the corrected C++ harness builds and executes locally.

The [raw bracket record](../bench/results/strict-f32-local-brackets-20261001.edn) retains
all chronological samples, exact generated source/ABI fingerprints, separate preparation/binding
times and binary provenance. Both `[8,256,256]` and `[256,256,256]` match the CPU oracle
exactly. Five of six bracket series fail the 5% CV diagnostic; this does not establish
a stable ranking, regression or schedule-promotion decision. The generated register-tiled
source/ABI match the September 27 record. The CLBlast library source checkout was removed,
so its source revision is explicitly unverified even though its version and binary hash remain.
This preserves the campaign's correctness/performance distinction; external training migration,
conservative partial-patch PDE interfaces and real CUDA/HIP execution remain open.

The benchmark guards and raw brackets merged as #960 (`511baa8d`) after all seven gates passed.

### Partial-patch conservative transport — 2026-10-01

The ordinary face-flux/CSR-divergence numerical programs lower through the existing public
equation-first compiler. A mixed-resolution periodic mesh with twelve coarse cells, sixteen
fine cells and sixty faces has a genuinely partial refined patch and split interface faces.
Its independent scatter oracle checks the generated gather path; geometry coverage, timestep
stability, constant preservation and volume-weighted mass are mandatory host checks.
Local OpenCL and Level Zero execute four-stage resident programs for three replays without
intermediate transfers, starting from poisoned scratch/flux arrays. Every replay matches two
more reference timesteps, and only the donated evolved field escapes. The new fixture passes
3 tests / 28 assertions; adjacent full-domain refinement and actual mapped-byte continuation
remain green (combined 9 tests / 78 assertions). These are device executions, not availability skips.

This establishes synchronous conservative interface transport, not general adaptive-mesh
planning, AMRPlan-to-incidence projection, subcycling/reflux or external time-to-accuracy parity.
CSR validity and paired incidence signs remain documented caller contracts. No compiler IR,
emitter, numerical reassociation policy, backend binding or memory/cache convention changed.

All seven final-head gates passed; #961 merged as `78cbef90`.

### GEMM timing-envelope audit — 2026-10-01

Source inspection found a benchmark-envelope mismatch: OpenCL Raster profiles the earliest
kernel START through latest kernel END, while the CLBlast harness profiles before/after queue
markers. The marker span may include host submission gaps before the first kernel and between
commands. Shared device clocks and stationary samples alone do not make these the same measure.
The harnesses now name their envelopes explicitly. Historical brackets stay diagnostic; they
do not establish matched-envelope kernel speedups.

The CLBlast harness also records its returned NDRange event START/END, verifies its command
type and containment in the queue span, and labels it last-kernel-only. In inspected CLBlast
1.7.0 source, the direct routine enqueues one kernel; the indirect routine may enqueue
preprocessing and return only the GEMM or final postprocessing kernel event. That diagnostic
must never silently replace whole-operation timing or be inferred from a matrix shape.
No compiler, generated kernel, selection, event-runtime or ownership contract changed.

A separate single-job build of tag `1.7.0` at `ca2fc3cb` completes successfully. Its
library image is byte-identical to the older retained library (SHA-256 `56f9d929…`). This
establishes a verified-source build matching those bytes, not the original checkout's provenance.
The [pinned diagnostic](../bench/results/clblast-pinned-event-envelope-20261001.edn) retains
the source/compiler/build and harness hashes and both timing scopes at the two bounded shapes.
The corrected C++ harness passes both independent CPU oracles and its event-type/containment
checks. Host-only Raster harness regressions remain green (3 tests / 26 assertions).

All seven final-head gates passed; #962 merged as `e4ac43d1`.

### Hierarchy-derived conservative interfaces — 2026-10-01

The partial-patch oracle now starts from `amr-plan/hierarchy`, retaining its nesting,
alignment and non-overlap validation. A bounded test projection removes covered coarse
cells and derives face/CSR arrays for central, narrow, shifted and disjoint fine patches.
Independent finest-grid coverage and per-cell perimeter checks reject missing or duplicate
coverage; each face must have precisely its two expected, oppositely signed incidences.
The unchanged typed numerical programs are checked against face-scatter evolution and
volume-weighted conservation. Central and disjoint geometries exercise the same public
compiled path on OpenCL and Level Zero, including three resident replays each.
The fixture passes 4 tests / 74 assertions with both native capabilities explicitly checked.

This connects the hierarchy geometry to a numerical oracle, not a public scalable AMR
projection. The test helper is explicitly limited to a 4x4 base, two levels, 2:1 refinement
and periodic 2D geometry; quadratic face enumeration is kept out of the library API.
Patch-field ownership, regridding/remapping, subcycling/reflux, mapped restart across a
hierarchy change and external accuracy/performance comparison are still open.

All seven final-head gates passed; #963 merged as `3cd16247`.

### Existing sparse-operator admission audit — 2026-10-01

The existing CSR `spmv` gives `[3.0 4.0]` for a two-row JVM oracle, but public equation-first
lowering declines. Correct-target diagnostic compilation (`:target-device :ocl:0`) locates
the first array-valued field binding `(.-rowptr A)`. The walked binding already carries
`ints`; the other index/storage fields retain `ints`/`doubles`, and the dimension retains
`long`. The scalar-only equation builder lacks a storage projection, not an inferred type.
The new diagnostic distinguishes this from untyped scalar source without changing admission.

An existing generated-SoA test function also declines in public equation-first compilation,
while its resident-pipeline test already exercises aggregate scalar replacement. Source
inspection confirms the public path bypasses the resident parameter-representation pass.
This motivates sharing that representation producer before widening mixed-record support;
it is not a reason to add a remap-specific compiler opcode or handwritten sparse kernel.
The frontend and route suites pass in a clean capped REPL: 188 tests / 1,244 assertions.
An earlier warm run after recursive namespace reload had stale matrix-record identity errors;
it is not counted as passing evidence. No new GPU admission or surface semantics is claimed.

All seven final-head gates passed; #964 merged as `9888c474`.

### Public aggregate projection acceptance — 2026-10-01

Equation-first compilation now invokes the existing aggregate representation pass with one
captured environment. A checked invocation attribute retains the original caller order and
declared array leaves; facade descriptors retain logical parameter names and replay accepts a
new logical record. Generated target signatures are still plain pointers/scalars, not records.
The retained projection also participates in source specialization identity.

Local OpenCL and Level Zero execute an all-array Float/Int record program and a generated SoA
input, including an unused field. JVM/independent comparisons, record replacement, constants,
wrong-class/storage and write/donation declines are checked. Lowering allocates zero driver
buffers. Adjacent invocation, public composition/cache and resident-SoA checks pass together:
32 tests / 210 assertions. The lean REPL does not include Chicory, so optional WASM execution is
delegated to the full CI alias rather than reported as a local pass.

This is a flat read-only aggregate facade boundary, not mixed CSR, nested aggregates, mutable
aggregate ownership, record-valued outputs, arbitrary record AD or vendor device evidence.

## Structured AD invocation and external Gemma — 2026-09-30

Structured `raster.params/value+grad` now uses the same generated fixed-arity flattening
adapter as defmodel forward/AOT invocation. The duplicate rest-argument flattening loop
was silently truncating surplus arguments and supplying nil for missing ports. Tree-shape
and naked identity-alias validation also now run before the flat AD program. Gradient
reconstruction, leaf ordering and underlying differentiation rules are unchanged. This is
an intentional surface correction: malformed structured AD calls fail at the same boundary
as forward calls, rather than proceeding with shifted values or ambiguous tied gradients.
It does not introduce variadic deftm support or a new AD dispatch/type registry.

Focused params/MLP tests pass 13 tests/45 assertions, including valid gradient equivalence
and proof that invalid calls do not execute the flat AD program. A clean isolated snapshot
of committed finetune-rstr `9e9ba5d` passes its FP32-vs-FP64 Gemma gradient test (1/16).
Its synthetic CPU LoRA SFT test reduces loss from 3.8179 to 1.3685 over 150 updates.
The one/two-block Params-tree gradient tests originally passed an obsolete extra batch
argument to their own batchless declarations. Correcting only those four calls in the
temporary snapshot yields 2 tests/32 finite-difference assertions passing across all fourteen
adapter roles. These corrected tests are experimental evidence, not an unchanged consumer
suite passing. The actual sibling checkout and its local work remain untouched.

External GPU training migration, real weights and device optimizer acceptance still remain.
The archived finetune adapter still names retired runtime entry points; this work neither
restores them nor substitutes a Raster-side model twin for that external gate.

The public equation-first resident Gemma train-step also executes after this consolidation:
25 on-device SGD updates pass 43 assertions, and CPU/device losses track from 2.800152
to 0.256915. Device availability is asserted before running the test; a skip is not
counted as execution. This is the existing small Raster model twin, not external real weights.

## Shared GPU scalar-helper frontend — 2026-09-30

A bare pure scalar cast helper exposed entry drift: direct GPU scheduling enabled the
existing hygienic scalar-body inliner, while equation-first and resident representation
passes left the same helper opaque and declined it as lacking a canonical intrinsic.
The shared ordinary and diagnostic pass runners now enable that same policy when
`:target-device` selects GPU compilation, including staged AOT entry points. Bare-tail admission
is limited to scalar, non-intrinsic helpers; array/storage helper and canonical intrinsic
implementation expansion retain their previous representation-pipeline policy. Direct scheduling
separately preserves canonical operators. Checked/call-by-value handling stays in the existing inliner, and JVM
expansion policy is unchanged. No helper-name registry or algorithm-specific rule is added.

The public cast-helper fixture emits one KernelBody map with no fallback and zero driver
allocations through OpenCL/CUDA/HIP equation-first source boundaries. OpenCL descriptor
compilation is checked both with and without diagnostics. The older descriptor entry does
not implement CUDA/HIP `pass-backend`; that preexisting limitation is not a vendor runtime
claim. Exact source-versus-device parity passes two resident replays on each local backend.
The existing inliner suite passes 19 tests/91 assertions, retaining hygiene, unused checked
arguments and call-by-value semantics.

The initial CI run caught a policy interaction missed by the small helper fixture: enabling
bare tails also expanded numeric implementations and array lengths, losing existing softmax/PDE
SOAC shapes. The fix separates admission from direct-scheduler intrinsic preservation and uses
the existing canonical registry plus existing generic-name normalization. Public softmax lowering
is now a regression oracle across all three source families; existing PDE/softmax device cases
are rerun locally. The failed head is not merged and all seven checks must pass on its replacement.
The replacement passes the complete source-emission gates locally (195 files per vendor),
the affected public frontend/PDE/softmax cases (7 tests/61 assertions), and the local coverage
ratchet (1 test/16 assertions): 253 rows, 240 TypedSOAC, one structured-control, five scalar and
seven explicit host-only, with 951 KernelBody artifacts and no lowering declines. These counts
describe this local same-compile report; the committed portable baseline is not silently updated.

An additional local probe redefines only a scalar helper from constant 1 to 2. Fresh public
compilation records a template invalidation and produces `[2,2,2]` on OpenCL, matching JVM
source, after previously producing `[1,1,1]`. This validates that one transitive path; it is
not a complete dependency/provenance or all-overload cache-invalidation audit.

The follow-up oracle makes this path reproducible without a mocked epoch: an unchanged
public map caller invokes a parametric scalar helper, hits its warm compilation template,
then replaces only the helper. The complete dependency manifest changes while the caller's
retained-source fingerprint does not; public lowering misses once and hits the replacement
template on repetition. OpenCL and Level Zero execute the replacement against the JVM
source twice, while the already instantiated old artifact still executes its original
semantics. Source restoration and artifact closure are explicit. No new invalidation layer
or cache is introduced; this closes one concrete transitive test gap, not transactional
reload, all-overload removal, or arbitrary retained-state invalidation.

## Isolated external resident training migration — 2026-09-30

The committed finetune-rstr `9e9ba5d` snapshot is now exercised through a temporary migration
of its actual LoRA and QLoRA train-step declarations. Only the isolated copy changes: its
retired `bind-program!`/`run-program!` wrappers use public `compiled/lower`, `instantiate!`,
adapter donation and semantic outputs. Frozen batch/weights are constants, all updates run
resident, and A/B are downloaded only after 30 steps. The wrappers close their artifacts in
`finally`. Their Double learning-rate port is explicitly materialized as Float at the existing
Float optimizer boundary, making the uncompiled JVM oracle valid too. No sibling checkout or
dependency is modified; this is not an upstream migration or real-weight training claim.

- LoRA, rows=2/in=4/rank=2/out=3: loss 0.08468652765 → 0.04792594910 on JVM, OpenCL and
  Level Zero; final A/B agree exactly.
- QLoRA, rows=2/in=32/rank=2/out=3 with shared frozen row-major INT8 weights: loss
  0.05017762880 → 0.04621510704 on all three paths. Maximum adapter errors are
  2.33e-10 for A and 7.46e-9 for B on each device backend.

These are correctness/state-progress oracles, not throughput measurements. The actual external
Gemma forward numerical declaration is exercised separately from its retired host wrappers, using
its unchanged numerical declarations and data constructors: B=1/sequence=2/d=8/heads=2/KV-heads=1/
head-dim=4/FFN=16/adapter-rank=2. It emits 60 public equation-first stages; both OpenCL and Level
Zero match the uncompiled JVM output within 1.073e-6 over all 16 output elements on two replays.

The admission gap was a storage/iteration distinction, not missing RMS loop syntax: the same
producer acquired incompatible extent names through differently grouped head dimensions and
later array-length reads. Unproved pointwise shape equality now retains the existing indexed
capture instead of retyping producer storage. No product CSE, checked-scalar hoist, dtype rule,
model recognizer, or new shape-equality authority is added. Known symbolic capacities retain the
existing graph precondition; slice-local capacities are checked against the actual linked view.
An overlong prefix is rejected with `:program-link-graph-range` before allocation or launch.
For an external input without an explicit shape/view contract, the first pointwise traversal
establishes only a conservative inferred contract. This does not claim complete symbolic shape
equivalence or admission of every traversal a larger physical caller buffer could support.

This is not a full package migration. The actual external VJP/update evidence below remains
separate from shared forward/VJP state, real-weight acceptance, and cross-vendor device
performance. The local Gemma twin does not substitute for those gates.

### Scalar cotangent and array storage boundary

The unchanged external Gemma VJP/update declaration exposed a JVM transpose failure:
`dot-product` returns Double, but its existing transpose passed that Double cotangent to a
Float array through a `scale` signature requiring one shared type. The shared primitive now
declares independent coefficient and array element types, `All [T S] [S, Array T] → Array T`.
Existing broadcast materialization retains T after the declared scalar arithmetic; the
mathematical adjoints and bilinear operation structure are unchanged. The focused review exposed
a second, pre-existing boundary defect: a derived scalar adjoint's binder received its desired
tangent tag before the expression was projected, so a Float coefficient gradient could remain
a runtime Double. Derived adjoints now resolve the expression's declared result through the
existing emission resolver and apply the shared tangent projection before stamping the binder.
Concrete kernel adjoints use the existing projection algebra's explicit non-nil/materialized
contract; unknown dynamic cotangents retain the default nil-safe helper. When a projection is
needed, its input adjoint and projection are separate flat BindCtx bindings so an inlined SOAC
is not hidden under a cast. Untagged argument types remain unguessed.
The public scalar-gradient reproduction exposed an untagged multi-hop SSA alias chain at final
parameter projection. Looking only at the immediate adjoint binder was insufficient. The shared
tangent projection now follows existing SSA bindings to carried tags or manifest conversions;
it does not infer function result types or assume a cotangent is present. Unknown, cyclic,
conditional and pullback-slot chains retain the nil-safe helper. The focused reproduction now
compiles with a Float semantic result in the freshly loaded REPL. Its final conversion follows
a reduction with an intervening pure array equation. The reduction-fusion rule now places a
direct Float/Double conversion across pure SSA equations, retaining physical effects, aliases
and opaque host barriers. It may replace only an absent epilogue or a same-dtype manifest
identity cast; arbitrary arithmetic and intermediate narrowing are not discarded. The public
scalar-gradient fixture executes empty, three-element and 2049-element cases twice on OpenCL
and Level Zero with exact JVM parity. The selected fusion suite passes 42 tests/226 assertions;
the public boundary and dual-backend numerical selection passes 2 tests/27 assertions. This
closes that reproduction, not arbitrary scalar-device placement or general mixed epilogues.

The completed-conversion follow-up reuses the existing reduction result transform, rather than
adding a scalar AD kernel. A shared predicate permits only one explicit Float/Double conversion
of the completed accumulator when output storage differs. Accumulation, workgroup scratch,
first-phase partial output and terminal-phase input retain the reduction dtype; only the terminal
store takes the converted result dtype. General mixed-width epilogue arithmetic remains outside
this admitted subset, and the existing dialect grammar retains the single-result transform rule.
Public Double-to-Float reduction fixtures execute empty, three-element and 2049-element cases
twice on each local backend, against the JVM. The large case exercises two-phase storage;
the 16777216/1/1 data distinguishes terminal conversion from premature Float accumulation.
Both Float-to-Double and Double-to-Float are checked at the SegRed scheduling boundary, but no
public Float-accumulator widening claim is inferred from that lower-level test. A focused review
found no blocker. The affected SOAC validation/fusion/emission suites pass 86 tests/551 assertions;
the public ABI/gradient-boundary/device selection passes 3 tests/40 assertions. These checks
overlap rather than defining additive coverage; they establish neither performance nor complete scalar AD residency.
This broadens mixed-type scale admission, not implicit dispatch coercion. A rounding
counterexample distinguishes this from rounding a Double coefficient to Float before
multiplication. Public declared-storage compilation and
both local device backends preserve that distinction exactly.

The actual external VJP/update program emits 157 equation-first stages. Two updates on each of
OpenCL and Level Zero match an independent CPU oracle built from the external pseudo-loss,
`value+grad`, and the
existing Float SGD primitive: maximum input-gradient error 2.87e-6, maximum error across
fourteen adapters 2.99e-8. State stays resident between updates; the small validation arrays
are downloaded to compare each step, so this is not a zero-transfer benchmark. The external
Double learning-rate port is explicitly rounded at the oracle's existing Float optimizer
boundary; the compiled numerical declaration itself is unchanged. Shared donated state across
independently prepared forward/backward artifacts is covered by the #950 checkpoint above;
upstream package migration and real-weight training remain separate obligations.

The existing parametric registration keys templates by annotation signature. In a warm process,
this broader declaration adds a template rather than deleting the old narrow template. Its
overlapping arithmetic is unchanged, but this does not establish general all-signature replacement;
restart for a clean one-template state. The earlier exact-signature reload tests do not certify
removal of an older signature.

The new public Float scalar energy-gradient reproduction is retained as an explicit decline:
its reduction and dependent Float projection currently meet an unrepresented device/host gap
in numerical-region selection (`:scheduled-equation-region`). The JVM scalar gradient is Float
and correct; that is not GPU scalar-gradient coverage. An earlier independent-seed variant also
exposed distinct input extents in a reduction; the map storage/capture fix does not establish
complete reduction storage admission. These are follow-up numerical-boundary obligations, not
permission to drop a projection or equate unproved shapes.

## Declared array storage — 2026-09-30

The opt-in `:preserve-declared-array-storage? true` keeps resolved array element tags as
physical pointer storage facts, independently of the kernel's scalar compute precision.
The default precision specialization is unchanged. Both equation-first and resident-descriptor
public lowering use the same parameter derivation; SoA fields follow the same policy. Template
and compilation identities distinguish the policy before reuse. Explicitly typed allocations
remain storage facts; dtype-polymorphic allocations still follow the compute specialization.

Focused checks cover semantic values, pointer ABI/emitted source, default behavior and template
separation. Mixed Float inputs and Double state/output match the JVM across two replays through
both public compiler routes on local OpenCL and Level Zero (eight device assertions).
This is a general mixed-storage contract, not a completed ASR weight-capture schedule, a new
numerical reassociation policy, or NVIDIA/AMD device evidence.

## Acceptance snapshot

Raw source result casts now protect their operand recurrence's retained precision before
canonical Fold construction, just as canonical conversion terms do. A Double recurrence stored
as Float is not a Float fold, and a retained Float carry converted to Double is not widened.
The row-local read/write integration oracle keeps each store in source order and preserves the
parent tail over two resident replays, through both public compilers on OpenCL and Level Zero.
Its first row starts at 16777216 with two unit contributions; an incorrectly Float-accumulated
fold would lose them. No reassociation or new reduction schedule is enabled by this correction.

This is a bounded acceptance snapshot, not a declaration that the compiler campaign is finished.
The numerical workloads below use existing source programs and reference tests; no benchmark-only
kernel implementation was substituted. OpenCL and Level Zero here both mean the same local Intel
Arc device, not independent vendor acceptance.

| Workload | Executed evidence | Boundary still open |
| --- | --- | --- |
| City day kernels, carried effect branches and two-exit search | JVM parity on OpenCL and Level Zero; original city fixture retained; nested recurrence branches, effect-empty arms and post-store local-result scopes replayed | Effectful early exits; broader irregular language coverage |
| Q4_K/Q6_K projection, two activation rows | Exact float-bit parity with ggml reference through public compiled/equation-first paths on both local backends | Large-shape throughput and external end-to-end decoder baseline |
| Q8_K activation quantization → Q4_K projection | Public equation-first B=2 width-640/padded-768 composition on OpenCL and Level Zero; packed activation leaves share resident nodes without a host source, shared weights are captured, and output matches the independent CPU quantized reference within 1e-3 | Pretrained decoder migration and its logits/token anchors; complete-chain device-event performance |
| Full AD linear/MSE/SGD step | Two resident mutable-weight updates match CPU AD on both local backends; no host weight reupload | General tape lifetime/reuse and frontier training scale |
| RK4 heat solver | Public equation-first compile/link, 64 points and three steps, CPU agreement within 1e-10 on both local backends | Distributed halo exchange, large grids and measured solver throughput |
| Routed attention | Existing tests executed: dense F32 and bidirectional packed segments on Level Zero; tiled history with dense/CSR routes and visibility on OpenCL. An indexed reference `KernelGraph` also executes through a direct `GraphLinkInstance` on both local backends, matching the independent numerical oracle. | Full Laya packed-agent benchmark, native cross-vendor execution, and replacing pretrained's synthetic graph descriptor adapter |
| Mixed emitted-program/graph linking | A two-stage program→direct-graph plan replays changed inputs twice through one resident intermediate on both local backends; source order and zero host source/copy for that intermediate are checked. | Mixed legacy descriptor instances still need a unified runtime schedule; graph full-write coverage remains a separate proof |
| Temporary storage reuse | One 16-byte allocation removed in a four-layer program; one private binding reused with refreshed inputs and detached snapshots on both backends | Zero-copy resident output leases, loop order and broader AD retention |
| Explicit generated FP32 register tile | Public equation-first request, static/dynamic NN and shared-weight NT (`linear-nb`); batch-one and ragged/tail shapes replay twice with exact JVM parity on both local backends; scalar preflight rejects invalid/overflowing shapes; CUDA/HIP source certificate checks | TN/TT and leading batched matrix slabs remain unsupported by this schedule; automatic selection stays portable; no competitive throughput claim |

Validation batches in the reusable bounded REPL: initial quant/attention/AD selection 6 tests,
53 assertions; expanded dual-backend quant/AD selection 3 tests, 52 assertions; dual-backend
PDE and precision-gate selection 3 tests, 25 assertions. All passed without device skips.
These overlap intentionally; they are not additive coverage counts.

The ordinary test suite remains untimed. CI discovers generic and precision/subgroup-specific
OpenCL gates; missing optional capabilities produce visible skips rather than claiming execution.
Compilation-only CUDA/HIP gates remain separate from device acceptance.

## Scalar-route census

The committed OpenCL corpus baseline now contains 251 source functions: 238 flat TypedSOAC,
one typed structured-control, six explicitly host-only, and six scalar routes. Five scalar
rows are value helpers rather than standalone GPU programs: `fast-exp` and the four
learning-rate schedules (`cosine-lr`, `linear-warmup-lr`, `step-lr`,
`warmup-cosine-lr`). They may be used inside compiled programs, but a scalar-only source
function has no parallel launch of its own.

The sixth, `gqa-decode-attention-weights!`, is different: it allocates score/output arrays
inside a nested head/token algorithm and accumulates a head-averaged alignment signal. The
Moonshine ASR path in pretrained-rstr calls it. Its scalar route is not evidence of a resident
GPU implementation; a future migration should expose explicit scratch/output ownership and
test the weight-capture result against that external workload. New scalar or compatibility
rows now require an intentional corpus-baseline update rather than entering silently.

The September 30 public-entry probe confirms that both Float and Double requests for this
alignment helper decline with `:equation-first-coverage` before executing sink effects. The
compatibility-debt ledger now records that request explicitly. Its host numerical oracle has
moved out of the BLAS-gated attention namespace: no BLAS or device is needed for bit-exact
output parity, accumulated sink mass, preserved cache/query inputs, untouched sink tails and
empty history. A large-magnitude Float counterexample pins per-head materialization: three
separate `1/3` updates round away, while a single widened aggregate would add one. Resident
capture must retain this ordered storage contract, not replace it with head atomics or an
unqualified reassociation. The current floating kernel policy also specializes declared
floating array storage; mixed-precision normalization scratch needs an explicit retained
contract before being passed between kernels. These are migration obligations, not a landed
resident alignment implementation or an external Moonshine model validation.

The private host-result memory boundary projects order from already selected, validated equation
graphs before allocating; it no longer binds a baseline executable. It still compares the actual
bound order before launch and checks the owned allocation delta. This does not extend reuse to
inspectable resident handles, structured-loop replay, or AD tapes.

## Projection diagnostic, not a promoted baseline

### Strict FP32 countercheck

The [September 27 CLBlast recheck](../bench/results/strict-f32-local-recheck-20260927.edn)
uses the existing public strict-FP32 GEMM canary, not the mixed-FP16 schedules below.
All Raster and CLBlast outputs matched the independent CPU oracle exactly. The generated
register-tiled source and ABI hashes were identical across the two Raster shapes.

| Shape `[M,N,K]` | Raster medians, µs | CLBlast before/after medians, µs |
| --- | --- | --- |
| `[8,256,256]` | 238 / 235 | 151 / 151 |
| `[256,256,256]` | 243 / 271 | 180 / 74 |

These shared-laptop samples do **not** reproduce the favorable September 25 ranking. They also
do not isolate a compiler regression: the CLBlast square bracket itself changes substantially,
and timing protocols differ (Raster kernel event span versus CLBlast queue markers). Raster
has one kernel here, with kernel sum equal to its event span, so inter-kernel launch gaps cannot
explain this case. Keep strict-FP32 schedule performance open; neither promote a selector nor
claim general competitiveness from the earlier snapshot. A frozen-revision, source-fingerprinted,
paired rerun under controlled device load is the next performance gate.

The documented Valhalla installation was unavailable. Device diagnostics used the existing
bounded HotSpot 25 REPL; host performance is not compared with Valhalla baselines. The check also
exposed missing OpenCL driver provenance: native discovery now queries `CL_DRIVER_VERSION`, and
hardware capabilities retain it for the existing calibration signature. Old driver-less identities
no longer match newly discovered devices; no calibration file is deleted or promoted.

### Mixed-FP16/FP32 projection

The existing `bench/public_linear_schedule_probe.clj` was rerun at `[8,256,256]`, with constant
weights, mixed FP16 multiplication/FP32 accumulation, three warmups and eight interleaved samples
per candidate. Both backends passed the exact rounded-input host oracle for all four generated
schedules. [Raw samples](../bench/results/local-projection-20260927.edn) identify the production
revision, scope and timing units. Transfers, validation, binding and the one-time weight transform
are outside the device-event samples. The laptop was shared; the exact driver build was not
recorded. Every series was classified nonstationary, and no tuning policy was promoted.

The materialized `:xmx-direct` schedule again shows a significant OpenCL inter-kernel/event gap:
0.34–0.61 ms total device span versus 0.017–0.028 ms summed kernel work. Level Zero's corresponding
gap is about 1 µs. This agrees with the earlier
[cross-backend diagnostic](../bench/results/public-linear-cross-backend-20260925.md), but does
not establish the cause or justify an unconditional schedule change.

Reproduction in a bench/test REPL:

```clojure
(load-file "bench/public_linear_schedule_probe.clj")
(public-linear-schedule-probe/run!
 {:target :ocl:0 ; repeat with :ze:0
  :shape [8 256 256] :rounds 8 :warmup-rounds 3
  :residency :constant-weights
  :revision "<tested production commit>"
  :environment "<machine/driver/load identity>"})
```

The follow-up isolated a substantial Raster-side contribution: unresolved Java FFM memory-layout
types caused reflective `MemorySegment.get/set` dispatch in launch setup. Instrumented native
enqueue calls took roughly 14–45 µs while complete enqueue wrappers took roughly 158–468 µs in
the captured tail. Supplying the existing concrete native layout types and integral event offsets
removed all reflection warnings in the launch/event hot path. No kernel, queue order, barrier,
precision policy or ownership rule changed.

The [final repeated diagnostic](../bench/results/opencl-native-layouts-20260927.edn), with the same
un-instrumented protocol, retained exact parity for every candidate. The materialized graph's
reported median fell from 506 µs to 44 µs; its kernel-duration sum remained around 18–28 µs.
The materialized series was still nonstationary, and these were not randomized A/B runs. This
supports removing avoidable runtime reflection, not a global speedup claim or selector promotion.
Focused tests cover 1–3D native launch marshalling without a driver, repeated profiled/unprofiled
graphs on both local backends, and transfer/event ownership.

Next: repeat the existing matched strict-FP32 CLBlast protocol
and the external pretrained fixtures under controlled load. Do not compare these mixed-precision
samples to strict-FP32 BLAS, or infer end-to-end Laya latency from this small projection.

## Equation-first indexed reduction reference

The existing `indexed_attention_device_test` now also exercises the public equation-first
compile/lower/LinkPlan path on local Arc OpenCL and Level Zero. The independent plan interpreter
is the oracle for three destinations, width five, two heads and either four or one membership
entries. Both backends passed two replays of both cases (36 assertions): repeated memberships,
unequal/empty destination segments and unused row tails retain their expected values. This is
fixed-reference correctness, not optimized attention performance or cross-vendor execution.

Compilation retains the generic segmented weighted-reduction plan and emits a generated
KernelBody through the common target emitter. Scalar shape preconditions fail during pure
binding, before session setup. Exact reference schedule rederivation proves complete output
writes; changing its stores or enclosing allocation contract is rejected. Shape multiplication
uses the same checked integer algebra in graph construction and invocation realization.

The zero-membership case now executes through public compilation and linking on both local
backends, for both the reference and explicit subgroup schedule. Empty logical index buffers have
zero visible length and retain one native byte only to provide a bindable pointer. Empty uploads,
downloads, fills and resident copies enqueue no device transfer. Zero-length asynchronous range
batches complete immediately under the existing event contract. The independent numerical oracle
and device replays cover this case; they do not imply that a kernel may read a zero-length buffer.

The existing subgroup score-reuse body now has an independent semantic-graph certificate as
well. Reference and subgroup candidates share graph/storage validation, shape leaves and ordered
scalar bindings; target admission is shared with compatibility routing. The subgroup certificate
declares floating-point dot reassociation, ordered membership folds, three-dimensional coordinate
bounds and no scratch. Unit checks project that certificate through OpenCL/CUDA/HIP emitters;
this is source-generation evidence, not NVIDIA/AMD execution or production admission.

A focused warm-REPL check replaced only the existing device test's compatibility graph provider
with the common emitter's certified subgroup graph. The same independent plan oracle passed on
local OpenCL and Level Zero (8 assertions), including empty destinations and row tails. The
equation-first path now also admits explicit subgroup selection, with exact optimized schedule
rederivation before granting complete-write initialization. Public compile/lower/LinkPlan execution
passed both replays of both edge counts on OpenCL and Level Zero (36 assertions), and serialized
artifact roundtrips preserve the selected schedule. `:auto` still selects reference; automatic
dispatch/tuning and paged-storage equation coverage remain open.

The September 29 follow-up exercises the public certified reassociation dispatch at both sides
of its component-width crossover on OpenCL and Level Zero. The same compiled program binds
5-wide and 515-wide inputs, including zero-edge cases, against the independent plan oracle;
the narrow call selects exact reference and the wide call selects subgroup score reuse. The
public `Compiled` equation-first path also executes the existing windowed prefill-softmax source
on both local backends and agrees with its JVM result for all four documented window shapes
(two focused device tests, 40 assertions, no skips). These are route/correctness gates, not a
claim that pretrained-rstr's routed paged-storage graph has migrated to equation-first or that
either attention schedule is fastest for a production decoder.

The public `Compiled` boundary now independently replays the same indexed reduction dispatch
on local OpenCL and Level Zero: 5-wide/nonempty selects the exact reference; 515-wide/empty
selects the subgroup candidate and writes the complete zero result. Both match the plan oracle
and preserve zeroed row tails (two focused device tests, 20 assertions, no skips). This checks
role/output projection and instantiation beyond the lower-level equation/link path; it does not
convert pretrained's physical page-route graph into a source-derived TypedSOAC equation.

The [September 29 Q4_K probe](../bench/results/q4-public-arc-20260929.edn) preserves raw
float-bit ggml parity through public compilation and warm resident replay. Its four-round
generated series is nonstationary, so it is not a tuning decision. A true compiler-template
hit still spent about 25 s in LinkPlan construction and 9 s in invocation certification.
Instrumentation identified repeated emitted-candidate validation inside each program check;
the validator now shares checked candidates only within one exact-object validation call.
Subsequent shared-laptop phase probes ranged widely, including a 4.4 s program-validation
subphase versus 16.5 s before the change, but are not controlled A/B evidence of a latency
improvement. Remaining whole-call/certificate validation must be measured separately.

The call-construction follow-up removes a second validation of the identical emitted program
inside `EmittedParallelProgramCall/make`: the constructor first validates that program, then
checks the constructed call against that same object. Public `validate!` still rederives the
program and rejects a mutated embedded target artifact. A later true template-hit Q4_K probe
spent 19.7 s preparing the same shape, including 14.9 s constructing its LinkPlan and 4.8 s
certifying the invocation. This shared-load observation does not isolate a reliable wall-clock
speedup; the structural reduction is one complete program-validation pass per constructed call.

The [longer September 29 warmup diagnostic](../bench/results/q4-warmup-arc-20260929.edn)
used the merged compiler and the same public Q4_K canary for three interleaved, bit-exact runs.
All six device-event series remained nonstationary. Both candidates shifted speed together after
several measured rounds, even with 16 warmups; none licenses a selector change. Two true compiler
template hits still spent about 15 s constructing the LinkPlan and 4.8 s certifying the
invocation. A scoped proof-pass profile found that each LinkPlan validation checks the same exact
program instance twice: once for structural validity and again while deriving access facts.
The follow-up shares that checked instance only within one LinkPlan validation call, while each
new public validation and each changed call is independently checked. This removes two repeated
program-call validations per Q4 preparation in the observed path (five down to three, including
the constructor), not the two separate LinkPlan proofs around role/output projection. Those
proofs need a single final-boundary lowering design rather than an unchecked persistent cache.

A [later Q4 canary after the final-boundary cleanup](../bench/results/q4-public-arc-20260929-followup.edn)
repeated the public Gemma-shape comparison with twelve rotating rounds. Every resident replay
passed the raw-bit ggml oracle, but both generated and serial event series changed speed state;
neither is stationary or a tuning/performance promotion. The input was already Q8_K. Separate
public B=2 quantizer→projection execution now checks the complete activation conversion seam,
but is a correctness gate, not a timed full-decoder result.

The final-boundary lowering now normalizes storage identities, applies public role and escaped
output projection synchronously, and validates only the projected LinkPlan. The resulting effect
evidence is sealed to that exact plan and reused for its invocation certificate; standalone
`equation-first/lower` still constructs a validated plan and public `certify` still rederives a
proof. A focused test counts one complete LinkPlan proof during `Compiled` preparation and checks
that modifying the certified plan invalidates its retained evidence. This reduces redundant
host-side proof work, not emitted device work; production Q4 timing still needs a controlled run.

The Q8_0 batched head source exposed a related typed-graph gap: its dense read-span proof retained
the map-local uniform block count `(quot in 32)` as if it were a public scalar. The read certificate
now projects only required, pure typed integral locals into checked launch algebra, and independently
rederives that projection from the graph's scalar types before permitting address rewriting. The
hardware-free equation-first route emits one KernelBody without fallback. On local Arc Level Zero,
two activation rows by 257 output channels match both the resident route and ggml's ordered dot
reference raw float bits. This is a small generated-head arithmetic gate, not a complete pretrained
head or real-vocabulary latency measurement.

A later warm, preparation-only Q4_K diagnostic at `[1,1024,640]` on the shared laptop reported
12.45 s for a true process-template hit after final-boundary validation moved into construction;
the certificate wrap itself was about 0.04 ms. This is not a controlled speedup comparison to
the earlier ~20 s samples. Narrow instrumentation attributed about 6.18 s to
`EmittedParallelProgramCall/make` and 2.35 s to the final LinkPlan proof. One exact emitted
equation call was rechecked inside its own constructor; the follow-up now retains only that
constructor-local object identity while leaving public call validation and LinkPlan validation
independent. That is one fewer complete step validation, not a persistent cache or a new tuning
decision. The remaining construction cost still matters for adaptive/JIT workloads.

The subsequent exact-object Q4 preparation trace found zero canonical graph-equivalence
comparisons after selected-graph identity admission, but repeated emitted-program validation
remained. One duplicate had a precise lifetime: the invocation constructed and validated a
temporary program instance, then public-role projection replaced that instance before the final
LinkPlan proof. The internal candidate now stays unvalidated and cannot escape the synchronous
final-boundary constructor; that constructor validates the projected instance once. The public
`program-instance` constructor and each fresh public LinkPlan validation still recheck the
complete call. A C-family test counts this distinction. Instrumented preparation wall time on
the shared laptop is not a controlled speedup or a kernel-performance measurement.

The equation-preparation audit found another synchronous proof duplication: `physical-results`
already validates its emitted boundary, but preparation, result-view checking and final step
checking each rederived that same boundary again. Preparation now obtains the checked physical
projection once and passes it with that exact immutable boundary to private call/result-view
checks. No proof token or validation cache is stored on the resulting record. The focused staged
contraction test counts two boundary validations during construction (the enclosing program and
one equation preparation), with both empty and nonempty result views. A separate public step
validation rederives the boundary once; changed artifacts and invalid result views remain rejected.
This removes redundant host proof work without changing schedules, numerical order or device code;
no preparation latency speedup is claimed from these validation-count tests.

The subsequent Q4 stack trace found the same immediate validation/projection duplication in
invocation storage lowering. That scope also now uses the checked public projection once; its
test counts only storage-local checks, so cold compiler validation cannot change the assertion.
A true template-hit preparation at `[1,1024,640]` before this last removal reported 4.56 s,
then 3.87 s in a stack-attributed repeat, with seven boundary validations. These shared-load
observations are not a paired speedup result. The remaining enclosing invocation, program-call
construction and final public LinkPlan proof remain independent checks.

A subsequent warm preparation-only diagnostic on the same `[1,1024,640]` Q4_K problem
reported about 3.4 s for a process-template hit. Scoped instrumentation attributed the bulk
of a later 3.2 s sample to six emitted-equation reconstructions; structural KernelGraph checks
were only about 19 ms in aggregate. Inclusive nested timings must not be added together.
Ordinary reconstruction validated its typed algorithm and scheduled body immediately before
the canonical graph constructor validated both again with the same boundary predicate.
The follow-up delegates directly to that constructor and removes the duplicate local checker.
The proof-count regression requires one scheduled-body validation per public reconstruction,
fresh validation on the next call, and rejection at `:parallel-program-algorithm` for changed
operands. No retained proof cache, accepted-domain expansion or device-code change is involved;
these shared-load samples do not establish a latency speedup.

The follow-up's public Q4_K canary executes both generated and serial schedules on Level Zero
and OpenCL at `[1,1024,640]`: all ten validation/warmup/measurement replays per backend match
the ggml oracle raw float bits after output poisoning. Two measured rounds are a correctness
check, not a stationary performance series or a schedule-promotion decision.

### Double arithmetic before Float materialization — 2026-10-02

Extending the public Q8_K→Q4_K composition oracle to changed-input replays exposed a
one-ULP scale error on both local backends. For the exact Float input `0.49460068345069885`,
Double division by 127 followed by nearest-even Float conversion should produce
`0.003894493682309985`; native execution produced `0.0038944934494793415`, exactly the
result of multiplication by a rounded Float reciprocal. Packed words and integer sums still
matched, and the existing toleranced projection oracle did not expose the scale discrepancy.

A one-element public kernel reproduces this independently of quantization. The retained
KernelBody source contains Double division and `convert_float_rte`, without fast-math flags.
An OpenCL diagnostic with native optimization temporarily disabled gives the expected result;
a separate resident Double intermediate also gives the expected result on both backends.
These observations isolate the native optimization boundary rather than justify relaxing the
oracle. Production build flags remain unchanged.

The OpenCL emitter now materializes nonliteral Double values in volatile private storage before
nearest-even Float narrowing. The guarded helper is shared by concatenated kernels; its demand
also declares FP64 when only scalar intermediates, rather than buffer storage, use Double.
Literal-only conversions keep their existing emission, and CUDA/HIP keep `__double2float_rn`.
The strengthened chain retains one binding across three inputs, including a zero row, poisons
all connected activation leaves and the output before each replay, and checks packed words,
scales and sums exactly against the independent CPU quantizer. Both actual local backends pass
40 assertions; the final projection retains its existing `1e-3` tolerance. Private volatile
materialization may cost instructions: this is correctness evidence, not measured throughput
or proof of all native arithmetic optimizations.

### Resident alignment capture investigation — 2026-09-30

#920 is merged after all seven final-head CI gates passed. Its source-result cast correction
retains the operand recurrence precision and ordered association, rather than inferring the
fold dtype from the destination store.

The caller-owned ASR alignment prototype subsequently exposed three shared boundaries:
closed numeric constants during typed JVM invocation preparation (#921), empty portable map
launches (#922), and parametric precision selection in the presence of fixed-precision scratch
arguments (#923). These are separate compiler PRs, not attention-specific emitter exceptions.
Their focused checks pass respectively 4 tests/11 assertions, 34 tests/252 assertions and
4 tests/16 assertions. All three have since squash-merged after their seven final-head gates:
#921 `649a00c8`, #922 `b0f02e5c`, #923 `499087f4`. #922's first CI run found an old exact-grid
assertion; its correction checks empty, singleton, full and partial groups (5 tests/42 assertions).

With those corrections loaded in the capped warm REPL, the unlanded resident prototype passes
2 tests/124 assertions across Level Zero and OpenCL: Float and Double storage, MHA/GQA,
empty history, replay accumulation, untouched parent tails and per-head Float materialization.
This is local device correctness evidence, not a performance result or external ASR migration.
The allocating wrapper now has an explicit host-only boundary and rejects input/sink identity
aliases before writes. Resident graph preflight checks reject overlapping writable/input and
writable/writable allocations; read-only q/k/v sharing remains legal. Mandatory host/preflight
tests pass 7 tests/95 assertions, including exact output and sink comparisons with a frozen old
weight-capture algorithm. The public workload ledger passes 1 test/18 assertions and records
two KernelBody steps, retained Float array storage with Double reciprocal scratch, two escaped
results and zero driver allocations during lowering. The baseline advances only these two
independently recompiled ASR rows; CI must still ratchet its complete corpus report.

A fresh-REPL check was essential: moving the arithmetic to a map with fixed Double scratch
exposed contextual widening and one-ULP host differences. Explicit T materialization at the
exponential, reduction updates, reciprocal and scratch reload restored the unchanged old
numerical oracle. The casts are source semantics carried through the existing conversion IR,
not an attention-specific inference or emitter rule. Independent weight/sink comparisons and
poisoned output/scratch device buffers are part of the new acceptance tests. General implicit
materialization/inference auditing and specialization invalidation on namespace reload remain
consolidation work; these explicit boundaries do not prove that every unannotated form is sound.
Annotated closed-HMap argument destructuring has a focused local acceptance slice:
75 tests/204 assertions for shared tree normalization, JVM JIT/AOT execution, reconstructed
reverse gradients, prior structured-model composition and hardware-free CUDA TypedSOAC source
compilation. Its generated flat method also passes 2 tests/8 assertions on actual local OpenCL
and Level Zero, including two replays and changed input-array values against the structured JVM
oracle. These are not evidence of a logical-map public GPU facade or CUDA native execution;
those remain explicit obligations in `compiler-consolidation.md` beside mixed record projection.

The mixed primitive record extension passes 41 tests/261 assertions across CSR projection and
the existing all-array aggregate, SoA, typed-map, composition and invocation/materialization suites. The actual
existing sparse/spmv source compiles to one generated KernelBody route without fallback and
executes on local OpenCL and Level Zero. Device oracles include rectangular CSR data, alpha/beta,
donated y state, new matrix buffers and rejected changes to all three captured scalar fields.
The success after rejected invocations checks that neither resident state nor buffers were
silently modified; preparing again correctly runs a changed three-row launch. This is not generic CSR input validation, mixed-record AD or logical HMap
GPU invocation; the remaining contracts are recorded in `compiler-consolidation.md`.

The broader AD consolidation, external training and distributed numerical acceptance remain
on the eight-item campaign; this vertical does not replace them.

### Declared tree public invocation — 2026-10-01

Logical HMap wrappers now enter public equation-first GPU lowering directly. The anonymous-map
and named nested HMap/HVec oracles execute on both actual local OpenCL and Level Zero, compare
against uncompiled JVM calls, and check replacement buffers, captured scalar rejection and
untouched previous results after structural/scalar preflight failure. Nested paths, canonical
leaf order, declaration-tamper rejection and logical-contract cache identity have host tests.
Existing record, CSR and composition suites remain in the focused acceptance set: 103 tests /
460 assertions pass in the capped warm REPL. This does
not introduce dynamic map dispatch, record/tree kernel values, aggregate donation, a new AD rule
or native CUDA/HIP execution; source compilation and device acceptance remain separate evidence.

The input-preflight follow-up reproduces the malformed-later-buffer issue on actual OpenCL:
one earlier upload succeeded before the bad second dtype declined. Shared LinkNode/DeviceArray
preflight now makes that count zero. Both local backends verify the earlier resident input is
unchanged after rejection. Device preflight tests prohibit registrations and copies and retain
readiness/pending-input state after overlap rejection. The focused link, output-lease,
composition, tree and record acceptance set passes 52 tests / 373 assertions. This verifies
validation-failure atomicity, not rollback after an actual driver transfer fails.

### Resident conservative CSR layout transfer — 2026-10-01

The existing partial-patch acceptance suite now passes 7 tests / 206 assertions, including the
old evolution cases. Sixteen layout-pair host checks compare ordinary CSR spmv with an independent
finest-tile average, preserve constants and mass, and check positive/volume-balanced weights.
Central-to-moved and moved-to-disjoint layouts execute three resident evolution/remap replays on
each local OpenCL/Level Zero backend. Four generated heat stages and one existing CSR stage share
one explicitly owned field allocation; both evolved and transferred fields match independent
oracles, with no component host bridge. This is test-only geometry projection and conservative
cell-average transfer, not regridding control, post-transfer evolution, an AMR accuracy claim or
competitive performance evidence. The original durable-continuation tests remain required.
The combined focused run includes those existing full-domain refinement and actual mapped-byte
continuation oracles: 13 tests / 256 assertions pass with no failures or errors.

### Exact scalar specialization and warm preparation — 2026-10-02

The existing descriptor resident-plan cache admitted a changed signed-zero scalar: ordinary
Clojure equality accepted both a tampered scalar certificate and rebinding a +0.0 template with
-0.0. Cache selection now uses the existing canonical scalar fingerprints; template binding and
certificate verification independently compare full canonical bytes. This preserves scalar
types, signed zero and NaN payloads without putting primitive-array contents in cache keys.
Other certificate fields retain their previous independent checks. This is a correctness
prerequisite for equation-first binding reuse, not that reuse's implementation.
The affected host and native composition suites pass 39 tests / 267 assertions, including
Float/Double signed-zero and NaN cache keys, Float certificate tampering and template rebinding.

An instrumented public Q4 projection preparation ([1,1024,640]) found a warm compiler-template
hit resolving in about 12 microseconds, while LinkPlan construction still took about 2.08 seconds.
Scoped instrumentation attributes most construction time to repeated structural proof work;
inclusive timings overlap and must not be summed. These observations come from a shared-load,
power-save laptop and are bottleneck evidence, not controlled performance comparisons. The next
cache extension must retain fresh invocation materialization and execute host equations once
per preparation, while reusing only source-free certified structure.

The partial-patch heat evolution acceptance was also rechecked on both available local OpenCL
and Level Zero: 2 tests / 36 assertions pass. This does not add subcycling, reflux, production
regridding control or an external AMR accuracy comparison.

### Invocation-local scheduled-body validation — 2026-10-06

An instrumented regression found four validations of the identical KernelBody within one
ScheduledKernelBody validation. Private synchronous projections now consume the exact body
already checked in that invocation, reducing this count to one. Each public helper still
validates independently, including after body mutation; no cache, reusable proof token or
trust flag is introduced. Argument, alias, scalar conversion, effect, numerical and launch
checks remain in place. The affected scheduled-body, emitted-equation and OpenCL emitter
suites pass 46 tests / 503 assertions in the capped warm REPL. This measures removed duplicate
validation, not end-to-end compilation speedup or training numerical acceptance.

### Real-checkpoint softmax precision experiment — 2026-10-08

The held two-layer training gate used the unchanged finetune source/oracle at
`9e9ba5d62f3822f056e01c37231d7eaa7c84947c`, the checkpoint SHA-256
`700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab`,
sequence length 2, batch 1, rank 16 and nonzero adapter B matrices. An isolated
same-input attention diagnostic established that native Float exponential rounding
can change softmax weights by one ULP; explicitly evaluating Double exp and casting
back to Float eliminated that isolated discrepancy on the local OpenCL device.

Applying this explicit precision boundary consistently to both denominator and
numerator in the production materialized softmax did not close the original model
gate. Predicted/reference losses were 17921.275/17921.271875, input-gradient worst
relative error was 0.0040108606627614705, and 11 of 28 adapter gradients exceeded
the unchanged 0.02 worst-relative threshold (maximum 1.1454261263845888).
These are diagnostic results, not accepted training parity or a performance claim.

The source-default experiment and its experiment-specific tests were removed:
extra Double exponentials are not justified by an isolated match when the full
gate still fails. The accepted explicit precision/target-library realization
contracts remain intact. Next diagnosis must compare actual model intermediates
and pullbacks at shared operands to distinguish inherited forward rounding from
local derivative/lowering errors; no oracle rewrite or tolerance relaxation is
authorized by this evidence. Temporary experiment scripts are not release tests.

The subsequent unchanged layer-0 forward-stage diagnostic localized the first
divergence to `linear-nb` inside the Q/K/V LoRA projections. Input RMS normalization
matched exactly. Q/K normalization and RoPE also matched exactly on shared inputs;
their end-to-end differences were inherited. Splitting Q projection showed local
differences in all three matrix products, while residual addition matched exactly
on shared operands. Base Q product maximum absolute difference from native BLAS
was 0.0001068115234375.

An independent dot reference over the same checkpoint operands established that
the generated base Q product matched ordered Float multiply-then-add exactly
(all 2048 outputs). Neither native BLAS nor ordered Float FMA matched that result.
Against a Double accumulated, Float stored diagnostic, native BLAS maximum
absolute error was 0.000030517578125 and generated GPU error was
0.000091552734375. These checks distinguish arithmetic realization from a lost
operand or wrong index in this isolated forward stage; they do not establish
correctness of every backward stage, justify changing the training oracle, or
close full-model gradient parity. Next work must isolate same-input matrix
pullbacks and assess a declared numerical policy for native-BLAS boundaries.

The public `linear-dx` and `linear-dW` helpers were then executed on the same
checkpoint weights/normalized inputs with one identical deterministic synthetic
cotangent, rather than an inherited full-model cotangent. Both generated helpers
matched independent ordered Float multiply-then-add references exactly. Native
BLAS differed (dx maximum absolute difference 0.0000022649765014648438;
dW 0.000030517578125). For the two-row dW reduction, native BLAS matched the
independent ordered Float FMA reference exactly, while the generated helper did
not. This establishes a concrete contraction-realization difference in both
forward and pullback, not a reason to turn on fusion implicitly under an exact
source contract. It is still limited to these operands/helpers: full-model
cotangent propagation and original gradient acceptance remain unverified.

### Transposed-left register staging — 2026-10-08

The unchanged real-checkpoint gate, when explicitly requesting the fused register
schedule, declined at a backward `AᵀB` contraction before model execution. The
register lowerer admitted NN and NT but not TN; the verified dense-matrix view
already represented the missing orientation. This was an execution-capability
gap, not evidence that changing arithmetic policy closes model-gradient parity.

The shared register body now stages physical TN storage `[K,M]` in contiguous
order into the existing canonical `[M,K]` workgroup tile. Its multiply loop,
barriers, scalar ABI, product policy and output mapping are unchanged. TT,
batched products and unsupported layouts remain outside this extension.

Structural tests cover unequal physical dimensions and canonical staging for
NT/TN (14 tests, 111 assertions). Public local OpenCL/Level Zero checks cover
NN/NT/TN decomposed and fused rounding, changed-input replay, and NN/TN ragged
dimensions including `[M,K,N]=[65,17,67]` under fixed and dispatch-selected
schedules, against an independent ordered Float oracle (2 tests, 102 assertions,
no failures/errors). The compile-fixture corpus emits the TN body for portable
OpenCL, CUDA and HIP. The TN fixture also compiles locally to `sm_80` PTX and
`gfx1100` device assembly, with HIP contraction enabled. This is hardware-free
compilation, not vendor execution or throughput evidence.
Full real-checkpoint acceptance remains a separate, unchanged gate.

With TN admitted, the explicit fused-register diagnostic executed the unchanged
two-layer model and reached its numerical gate rather than declining compilation.
It still failed at iteration 0: predicted/reference loss
17921.278125/17921.271875, input-gradient worst-relative error
0.005268926908926802, and 15 of 28 adapter gradients above the unchanged 0.02
threshold (maximum 0.7991052642515084). This rules out merely switching to
ordered FMA as a sufficient fix. The capability extension is independently
validated; the fused policy is not promoted to a default or a training-parity
claim. Actual shared-operand pullbacks and inherited cotangents still need
localization against the monolithic CPU AD oracle.

### Actual shared-cotangent pullbacks — 2026-10-09

Two isolated real-checkpoint block VJPs were run with the native CPU forward
input and native CPU output cotangent supplied identically to CPU and GPU.
The original external backward declaration and explicit fused register schedule
were used. This removes inter-layer input/cotangent drift but still recomputes
each block's internal forward intermediates on its respective target. Layer 1
still exceeds the adapter threshold for Ak and Bg; layer 0 exceeds it for Ak,
Bk, Av, Bg, Au, Bu, Ad and Bd. Thus inter-layer drift is not the only cause.
These are diagnostic comparisons, not a substitute acceptance oracle.

Wrapping the existing Float matrix-pullback typed interfaces captured 42 calls
from the CPU layer-1 VJP without changing any of its 14 adapter gradients
(all coordinates bit-identical). Each adapter gradient is the direct result of
one captured `linear-dW` call. On those exact actual-model operands, all 14
generated fused TN products match both native BLAS and an independent ordered
Float Math/fma dot reference exactly. Their reduction width is two, matching
the unchanged sequence length. This rules out those isolated adapter-matrix
products as the source of the observed layer-1 discrepancy; it does not prove
other widths, the full forward recomputation, input cotangent propagation,
nonlinear adjoints, or the final training gate. Next localization follows the
actual cotangents upstream and checks input-gradient products at shared operands.

All 21 captured layer-1 input-gradient matrix calls were also executed on
identical operands. Generated fused NN products match ordered Float Math/fma
exactly in every coordinate. Native BLAS uses different accumulation results;
the largest observed same-input absolute difference is 0.00067138671875.
Double-dot diagnostics improve some differences and worsen others; they are not
an automatic justification for widening the production graph. The remaining
full-block mismatch must be diagnosed through its recomputed intermediates and
cotangent propagation, rather than attributing it to the isolated adapter-TN
products or replacing the monolithic oracle with a convenient local one.

A separate CPU-only arithmetic simulation replaced the three typed matrix
helpers with independent Double-dot/Float-store implementations, leaving the
model, AD rules, nonlinear operations and native-BLAS monolithic reference
unchanged. Widening all matrix calls still fails 8 of 28 adapter checks (maximum
1.1460127391180726). Widening only input-gradient products leaves two failures
(layer-1 Ak 0.02482720148392874 and Ag 0.03098584773269312). Neither diagnostic
is a GPU execution result or an accepted replacement oracle; no production
widening follows from it. Better local dot accuracy alone is insufficient
evidence that the complete numerical contract is met.

Replacing only the CPU matrix helpers with ordered Float FMA also fails 15
adapter checks (maximum 0.8192590523403551), with predicted loss 17921.278125,
the same loss as the executed fused GPU chain. The unchanged nonlinear CPU
path is retained in this simulation. Thus matrix realization alone can
reproduce the acceptance failure pattern; this is not a proof that every GPU
nonlinear operation or full cotangent agrees with its CPU counterpart.

Captured actual layer-1 normalization calls give additional bounds. All six
chunked RMSNorm forward calls match CPU exactly at shared inputs. All six
backward input-gradient calls have small differences: maximum absolute error
0.00006103515625 and maximum coordinate-relative error
0.00004876049798283915 (these maxima come from different calls). Capture
preserves the original CPU adapter gradients exactly. A proposed partial-state
diagnostic based on evaluating the retained body, including a return-projected
anonymous ftm, did not reproduce the original compiled CPU output exactly.
Its partial states are therefore rejected as an oracle, not used to justify
production changes. A valid partial-state comparison must first establish
observational equivalence through the same specialization and emission path.

The native reference in this warm JVM is the selected MKL threaded LP64
component provider (`libmkl_intel_lp64.so`), not the CUDA agent's separately
reported OpenBLAS batch-extension environment. Provider selection is metadata,
not a numerical proof; the captured dot comparisons above supply the functional
evidence for these operands. Do not conflate this accumulation-order finding
with the independent OpenBLAS interleaved-batch crash.

A bounded CPU-only partitioned-FMA ladder was also checked before adding a
production reduction policy. Strided 4/8/16-chain products each fail 9 adapter
checks; contiguous 4/8/16-chain products fail 10/9/13 respectively. Tiny
reductions (including the two-row adapter products) retain one FMA chain.
The model, nonlinear CPU operations and native monolithic oracle are unchanged.
These finite candidates do not establish that all blocked policies fail, but
none supplies evidence for a numerical fix or a new production default. Keep
parallel-reduction performance work separate from unchanged model acceptance.

### Mixed-precision JVM helper extraction: retained call boundaries

A subsequent partial-state diagnostic used named `deftm` projections with the
original Array-float return annotation and verified that its projected output
was bit-identical to the original captured CPU normalization call. Its partial
dot state differed from the GPU at 35 of 64 coordinates. An independent oracle
identified the CPU realization as Float inner products with Double weighted
terms and a Double carry; the GPU matched narrowing the weighted term before
each Float addition. However, the original retained loop binder and addition
were both stamped Float. This was not evidence for introducing a new GPU
precision mode.

The JVM helper-extraction path normalized typed `.invk` calls into generic
arithmetic, discarding both result stamps and typed-call operand conversions.
That changed a Float addition accepting a Double term into generic Double
arithmetic. A cancelling weighted fold reproduced the difference independently
of the model: the small method returned zero, but the extracted version returned
one. Retaining the original typed calls fixes that regression without a new
type registry or a model-specific lowering. The existing typed JVM call emitter
remains responsible for conversions; helper partitioning is not permission to
change numerical semantics.

With fresh named projections after this change, the actual model's PSS, PC and
k1 states match the GPU exactly, and PC matches the independent Float-carry
oracle exactly. Downstream k2 still differs at one coordinate, and dx differs
at 536 of 1280 coordinates (maximum absolute 0.001953125, maximum coordinate
relative 2.3887326823598837e-7). These diagnostics do not establish acceptance
of the full real-model gradient gate or certify helper-call performance. The
affected JVM loop and bytecode suites pass 50 tests / 139 assertions, including
lazy and AOT changed-input checks across the actual extraction threshold.

Cold CI exposed one test-metric assumption: the chunked Float RMSNorm backward
was compared to a different reduction tree using error divided by the resulting
dx coordinate. The maximum was 0.0010544952502931275 against 0.001 after restored
Float rounding. Because dx subtracts two potentially large terms, that ratio
can amplify harmless reassociation at cancellation coordinates. The test now
uses an independent Double mathematical reference and the sum of the two term
magnitudes as its backward-error scale, for both Float and Double schedules.
The existing numerical bounds and finite-difference checks remain, and both
schedules are checked against the independent reference as well as each other.
Zero scales require exact agreement; no arbitrary absolute-error floor is added.
No production normalization formula or pinned model acceptance metric changes.
The three affected JVM/normalization namespaces pass 55 tests / 201 assertions
in the fresh capped REPL, including the unchanged finite-difference checks.

A fresh capped JVM reran the held real-weight harness at the original source
pins and tolerances with no pre-fix CPU classes. Loss remains 17921.2765625 vs
17921.271875; input-gradient error is 0.006813176206858781. Twelve of 28 adapter
checks still fail, with maximum 1.0649456537233917. This fixes a genuine compiler
partitioning inconsistency but does not close the real-weight acceptance gate.

A subsequent cold run including retained scalar and SIMD precision fixes has
the same loss pair, input-gradient error 0.0019965899080526235, and eleven of 28
adapter checks failing (maximum 0.6251970207953054). An additional diagnostic
using the public `:gemm-precision :f32-scalar` option produces the identical
metrics. Thus this experiment does not attribute the remaining disagreement to
default mixed-precision matrix selection. Source pins, checkpoint, adapter seeds,
reference arithmetic and acceptance tolerances are unchanged. These are still
failed model checks, not a training certificate.

### Quantized native fold: retained arithmetic versus output storage

The wider CI suite exposed an obsolete vectorization assertion in the Q4 x8
test. Its source explicitly converts both scales and the accumulator to Double,
then stores Float after each block. The old all-Float AVX2 lowering instead rounds
the scale product early. A one-block independent fixture distinguishes them:
Float scales 1.0000001 and 0.3, folded integer dot 3, produce 0.90000015 with
retained Double arithmetic but 0.9000001 with early Float rounding. Tests compare
raw Float bits for the JVM source, native scalar and SIMD-requested compilation,
with changed scales and the same compiled functions.

The precision guard therefore declines this mixed-precision fold; the Q8 x8
sibling uses the same Double-compute/Float-store structure and shares this debt.
The integer
dot override remains, and the separate homogeneous Float integer-widening SIMD
execution regression remains required. This is an explicit performance debt,
not evidence of performance parity: restore vectorization through typed
Float-load → Double-compute → Float-store conversions, preserving lane counts,
integer widths and rounding boundaries. Do not change the kernel's arithmetic or
remove the precision guard merely to recover the previous intrinsic spelling.

### Shared scalar partial evaluation: typed operation boundaries

The same conversion-erasure class exists outside JVM helper extraction. Shared
scalar simplification and partial evaluation turned a retained Float addition
of Float 1e8 and Double 1 into Double 100000001; a nested cancelling Float
program changed from zero to one. Direct invocation of the original selected
typed methods supplies an independent executable semantic oracle.

These passes now preserve `.invk` nodes and their selected implementation/type
metadata while simplifying arguments. Generic arithmetic identities and constant
folds do not establish the selected operation's operand conversions, result
rounding or IEEE behavior. Unknown calls are also preserved rather than guessed
from mangled impl names. Direct untyped arithmetic retains its existing rules.
Recursive and fixpoint entrypoints are checked as well as one-step helpers;
regressions include mixed constants, intervening rounding, signed zero, infinity
times zero, metadata and child constant propagation. This is not a new type or
function registry, and it does not add a separate typed scalar evaluator.

The unused recursive scalar normalizer is deleted after checking production and
test references. Its sole test dependency now uses the existing shared implementation-name
decoder strictly as a provenance diagnostic, not as an arithmetic lowering rule.
The positive and deliberately corrupted-call controls preserve that regression.
The SIMD backend has a distinct `normalize-invk` requiring its own proof and
execution regression before a global no-erasure claim. Safe typed scalar
optimizations should derive from existing declared signatures and canonical
conversion/numeric authorities, not reintroduce generic rewriting through a
different namespace. Compile-size and performance effects remain to be measured.
