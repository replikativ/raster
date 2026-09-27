# Local compiler evidence — 2026-09-27

This is a bounded acceptance snapshot, not a declaration that the compiler campaign is finished.
The numerical workloads below use existing source programs and reference tests; no benchmark-only
kernel implementation was substituted. OpenCL and Level Zero here both mean the same local Intel
Arc device, not independent vendor acceptance.

| Workload | Executed evidence | Boundary still open |
| --- | --- | --- |
| City day kernels, carried effect branches and two-exit search | JVM parity on OpenCL and Level Zero; original city fixture retained; nested recurrence branches and effect-empty arms replayed | Effectful early exits and post-effect local-result scopes; broader irregular language coverage |
| Q4_K/Q6_K projection, two activation rows | Exact float-bit parity with ggml reference through public compiled/equation-first paths on both local backends | Large-shape throughput and external end-to-end decoder baseline |
| Full AD linear/MSE/SGD step | Two resident mutable-weight updates match CPU AD on both local backends; no host weight reupload | General tape lifetime/reuse and frontier training scale |
| RK4 heat solver | Public equation-first compile/link, 64 points and three steps, CPU agreement within 1e-10 on both local backends | Distributed halo exchange, large grids and measured solver throughput |
| Routed attention | Existing tests executed: dense F32 and bidirectional packed segments on Level Zero; tiled history with dense/CSR routes and visibility on OpenCL | Full Laya packed-agent benchmark and independent cross-vendor numerical execution |
| Temporary storage reuse | One 16-byte allocation removed in a four-layer program; one private binding reused with refreshed inputs and detached snapshots on both backends | Zero-copy resident output leases, loop order and broader AD retention |
| Explicit generated FP32 register tile | Public equation-first request, static and dynamic NN extents; batch-one and ragged/tail shapes replay twice with exact JVM parity on both local backends; scalar preflight rejects invalid/overflowing shapes; CUDA/HIP source certificate checks | NN layouts only; automatic selection stays portable; no competitive throughput claim |

Validation batches in the reusable bounded REPL: initial quant/attention/AD selection 6 tests,
53 assertions; expanded dual-backend quant/AD selection 3 tests, 52 assertions; dual-backend
PDE and precision-gate selection 3 tests, 25 assertions. All passed without device skips.
These overlap intentionally; they are not additive coverage counts.

The ordinary test suite remains untimed. CI discovers generic and precision/subgroup-specific
OpenCL gates; missing optional capabilities produce visible skips rather than claiming execution.
Compilation-only CUDA/HIP gates remain separate from device acceptance.

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
