# Local compiler evidence — 2026-09-27

This is a bounded acceptance snapshot, not a declaration that the compiler campaign is finished.
The numerical workloads below use existing source programs and reference tests; no benchmark-only
kernel implementation was substituted. OpenCL and Level Zero here both mean the same local Intel
Arc device, not independent vendor acceptance.

| Workload | Executed evidence | Boundary still open |
| --- | --- | --- |
| City day kernels and three-carry effects | JVM parity on OpenCL and Level Zero; original city fixture retained | Multiple source exits/recurrence sites; broader irregular language coverage |
| Q4_K/Q6_K projection, two activation rows | Exact float-bit parity with ggml reference through public compiled/equation-first paths on both local backends | Large-shape throughput and external end-to-end decoder baseline |
| Full AD linear/MSE/SGD step | Two resident mutable-weight updates match CPU AD on both local backends; no host weight reupload | General tape lifetime/reuse and frontier training scale |
| RK4 heat solver | Public equation-first compile/link, 64 points and three steps, CPU agreement within 1e-10 on both local backends | Distributed halo exchange, large grids and measured solver throughput |
| Routed attention | Existing tests executed: dense F32 and bidirectional packed segments on Level Zero; tiled history with dense/CSR routes and visibility on OpenCL | Full Laya packed-agent benchmark and independent cross-vendor numerical execution |
| Temporary storage reuse | One 16-byte allocation removed in a four-layer program; two optimized replays match JVM on both backends | Reusable resident nonescape interface, loop order and broader AD retention |

Validation batches in the reusable bounded REPL: initial quant/attention/AD selection 6 tests,
53 assertions; expanded dual-backend quant/AD selection 3 tests, 52 assertions; dual-backend
PDE and precision-gate selection 3 tests, 25 assertions. All passed without device skips.
These overlap intentionally; they are not additive coverage counts.

The ordinary test suite remains untimed. CI discovers generic and precision/subgroup-specific
OpenCL gates; missing optional capabilities produce visible skips rather than claiming execution.
Compilation-only CUDA/HIP gates remain separate from device acceptance.

## Projection diagnostic, not a promoted baseline

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

Next: isolate the OpenCL submission/event gap without weakening dependency barriers; retain whole
graph spans alongside kernel sums. Then repeat the existing matched strict-FP32 CLBlast protocol
and the external pretrained fixtures under controlled load. Do not compare these mixed-precision
samples to strict-FP32 BLAS, or infer end-to-end Laya latency from this small projection.
