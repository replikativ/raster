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
`:target-device` selects GPU compilation, including staged AOT entry points. Intrinsics remain
operators, checked/call-by-value argument handling stays in the existing inliner, and JVM
expansion policy is unchanged. No helper-name registry or algorithm-specific rule is added.

The public cast-helper fixture emits one KernelBody map with no fallback and zero driver
allocations through OpenCL/CUDA/HIP equation-first source boundaries. OpenCL descriptor
compilation is checked both with and without diagnostics. The older descriptor entry does
not implement CUDA/HIP `pass-backend`; that preexisting limitation is not a vendor runtime
claim. Exact source-versus-device parity passes two resident replays on each local backend.
The existing inliner suite passes 19 tests/91 assertions, retaining hygiene, unused checked
arguments and call-by-value semantics.

An additional local probe redefines only a scalar helper from constant 1 to 2. Fresh public
compilation records a template invalidation and produces `[2,2,2]` on OpenCL, matching JVM
source, after previously producing `[1,1,1]`. This validates that one transitive path; it is
not a complete dependency/provenance or all-overload cache-invalidation audit.

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
The broader AD consolidation, external training and distributed numerical acceptance remain
on the eight-item campaign; this vertical does not replace them.
