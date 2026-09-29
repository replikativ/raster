# Local compiler evidence — 2026-09-27

This is a bounded acceptance snapshot, not a declaration that the compiler campaign is finished.
The numerical workloads below use existing source programs and reference tests; no benchmark-only
kernel implementation was substituted. OpenCL and Level Zero here both mean the same local Intel
Arc device, not independent vendor acceptance.

| Workload | Executed evidence | Boundary still open |
| --- | --- | --- |
| City day kernels, carried effect branches and two-exit search | JVM parity on OpenCL and Level Zero; original city fixture retained; nested recurrence branches, effect-empty arms and post-store local-result scopes replayed | Effectful early exits; broader irregular language coverage |
| Q4_K/Q6_K projection, two activation rows | Exact float-bit parity with ggml reference through public compiled/equation-first paths on both local backends | Large-shape throughput and external end-to-end decoder baseline |
| Full AD linear/MSE/SGD step | Two resident mutable-weight updates match CPU AD on both local backends; no host weight reupload | General tape lifetime/reuse and frontier training scale |
| RK4 heat solver | Public equation-first compile/link, 64 points and three steps, CPU agreement within 1e-10 on both local backends | Distributed halo exchange, large grids and measured solver throughput |
| Routed attention | Existing tests executed: dense F32 and bidirectional packed segments on Level Zero; tiled history with dense/CSR routes and visibility on OpenCL | Full Laya packed-agent benchmark and independent cross-vendor numerical execution |
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
on both local backends and agrees with its JVM result. These are route/correctness gates, not a
claim that pretrained-rstr's routed paged-storage graph has migrated to equation-first or that
either attention schedule is fastest for a production decoder.

The [September 29 Q4_K probe](../bench/results/q4-public-arc-20260929.edn) preserves raw
float-bit ggml parity through public compilation and warm resident replay. Its four-round
generated series is nonstationary, so it is not a tuning decision. A true compiler-template
hit still spent about 25 s in LinkPlan construction and 9 s in invocation certification.
Instrumentation identified repeated emitted-candidate validation inside each program check;
the validator now shares checked candidates only within one exact-object validation call.
Subsequent shared-laptop phase probes ranged widely, including a 4.4 s program-validation
subphase versus 16.5 s before the change, but are not controlled A/B evidence of a latency
improvement. Remaining whole-call/certificate validation must be measured separately.
