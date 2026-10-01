# Local compiler evidence — 2026-09-27

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

The broader AD consolidation, external training and distributed numerical acceptance remain
on the eight-item campaign; this vertical does not replace them.
