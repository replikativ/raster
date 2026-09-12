# Generated-kernel comparison protocol

This is the measurement contract for the compiler-unification campaign, not a report of
accelerator results. External runs are opt-in and stay outside ordinary CI. CPU OpenCL
execution and CUDA/HIP source compilation establish different correctness facts; neither
establishes accelerator competitiveness.

## Landing order

1. Reuse `raster.perf.production-canary/gemm64!` and `prepare-gemm` as the first public
   resident entry point. Preserve its exact prepared-program compilation evidence and
   host-synchronized timing label. Its 64-square shape is a smoke canary, not a throughput claim.
2. Extend the public entry-point ladder to square, rectangular, batch-one projection,
   multi-row projection and awkward-tail shapes. Record candidate selection, including
   fallback. Do not substitute independent source builders for the public compiler path.
3. Add matched external GEMM measurements on available accelerator hardware. Separate
   strict FP32 from reduced-precision multiplication with FP32 accumulation. Keep CPU
   development usable without downloading vendor runtimes or booking cloud machines.
4. Add reductions/fusion, quantized projection, attention and a scientific stencil/solver
   in that order as their generated routes become measurable. Retain regression cases
   even when another schedule is faster on the aggregate.

## Baseline selection

### Public dynamic GEMM/activation probe

In a test or bench REPL, use the same public functions as the existing production canary:

```clojure
(require '[raster.perf.gemm-comparison :as comparison])
(comparison/run! {:shape [8 64 64] :target :ocl:0
                  :gemm-precision :mixed-f16-f32
                  :compiler-revision "<commit plus any dirty changes>"
                  :environment-tag "<machine/driver identity>"
                  :rounds 12 :warmup-rounds 4})
```

The probe compares ordinary dynamic contraction→map composition with an explicit typed ReLU
epilogue, compiling and binding each once. It reuses the existing rotating measurement sampler
with a default `:host-synchronized-replay` clock; those samples are not device throughput.
Set `:timing-source :device-event` to profile this descriptor-backed public graph through
`link/profile!`. Samples use the device event span, not summed kernel durations or host time;
missing timestamps fail without fallback. `:replay-profiles` retains observed kernel events in
execution order, including prevalidation and warmup. One-time conversion prologues are excluded.
The separate `PreparedParallelProgram` runtime still lacks this aggregate profiling API; it is
not the runtime used by this probe.
Every replay starts with NaN output and checks an exact dyadic-input CPU oracle afterward. Uploads
and validation downloads are outside timing but can influence cache/thermal state. Logical buffer
and CPU-reference work budgets do not bound compiler/driver RSS. Fixed even rounds balance order;
they are not a stationarity guarantee. Compile times share the same REPL and compilation order,
so they are not comparable fresh-process cold compilation measurements.

Initial Arc OpenCL observations (driver `26.05.37020.3-0`, Java 25.0.1, shared laptop):
[before](../results/public-gemm-20260912-before.edn) and
[after](../results/public-gemm-20260912-after.edn) the OpenCL backend-alias admission fix.
Both records retain the base compiler revision, dirty-change label, source/ABI signatures and
chronological samples. The initial record predates the added dispatch-decline reporting.

| `[8 64 64]`, mixed precision | Composed median ms | Explicit epilogue median ms | Both stationary |
|---|---:|---:|---|
| Before alias fix | 2.026 | 1.357 | No |
| After alias fix | 1.693 | 1.478 | No |

All poisoned-output checks passed. The useful structural finding is that `:ocl` was rejected by
mixed-DPAS admission, which recognized only `:opencl` and `:ze`. The fix admits the same checked
schedule for all three spellings, without weakening matrix/subgroup requirements. Matrix
alternatives now appear in public OpenCL dispatch, but candidate counts are not selected or
executed launch counts. Composed dynamic source still has two resident steps, explicit source one.
No timing series passes the CV heuristic; these data establish neither a speedup nor a regression.
The subsequent [device-event probe](../results/public-gemm-20260912-device.edn) passes every exact
output check and records two steady replay entries for composed source versus one for the explicit
epilogue, including the generated XMX contraction. Both series remain nonstationary. Event span
includes inter-kernel gaps; kernel-duration sum does not. Neither is interchangeable with the
earlier host-clock measurements, and no winner is promoted from this shared-laptop probe.
Next: automatic symbolic-extent fusion, correlating observed events with executable evidence,
then larger projection shapes and matched external implementations.

### External comparators

| Workload | Comparators to implement | Fairness boundary |
| --- | --- | --- |
| Dense GEMM | Device vendor library; tuned Triton where supported | Same dimensions, transpose/layout, multiplication precision, accumulation and epilogue |
| Fused model expressions | PyTorch eager and `torch.compile`; JAX/XLA | Same graph boundary, outputs and synchronization; compilation separate |
| Functional array/reduction programs | Futhark | Equivalent source semantics, reduction policy and materialized outputs |
| Stencils/layout transformations | Halide; matched JAX implementation | Same boundary conditions, time steps and precision |
| Quantized decode projection | llama.cpp; pretrained-rstr end-to-end | Identical quantization format, padded/logical widths and dequantization semantics |
| Attention | Framework optimized attention; compatible Triton implementation | Same visibility, GQA, KV storage precision and cache layout; account for conversions |

MLIR is a compiler infrastructure reference, not a standalone timed implementation. Baseline
availability is device-dependent. An unavailable comparator must be recorded as unavailable,
not replaced by a weaker result under the same name. These are proposed comparators, not
claims that adapters or measurements already exist.

## One record per implementation, shape and policy

Persist an EDN/JSON record alongside raw samples. Required fields for comparable evidence:

- Identity: workload/version, implementation and source revision, public entry point,
  complete shapes/strides/layout, storage and accumulator dtypes, quantization format,
  numerical policy, seed and exact input-generation recipe.
- Environment: device model, runtime/driver, compiler/library/toolchain revisions,
  thread/device limits and a stable environment tag. Explicitly label CPU versus GPU.
- Correctness: independent reference, tolerance with rationale, maximum absolute/relative
  error, nonfinite/tie behavior where applicable, and validation outcome.
- Compilation: cold compile time, warm cache lookup, tuning time/budget, candidate count,
  chosen schedule, actual emission route and source/ABI signature. Candidate entry points
  are not executed kernel counts. Record unknown counters as unknown, never zero.
- Measurement: warmup, sample count, raw samples, median and dispersion, timing clock,
  synchronization, cache state, transfers/allocation inclusion, and stationarity verdict.
- Resources: measured launches, allocations, peak live device bytes and transferred bytes,
  with measurement method. Descriptor scratch counts are not peak memory.

Device-event kernel timing and host-synchronized replay latency are separate series. Include
an end-to-end series when packing, transfers or page routing are required. Do not hide those
costs behind a resident-only comparison. Include cold compilation/tuning in amortization
reports but never silently mix it into warm execution samples.

For a bounded warm resident A/B/C comparison, bind all candidates with profiling enabled,
validate each independently, then use `gpu/measure-bound-kernel-graphs-interleaved!` with an
ordered vector of `{:id :candidate-name :handle handle :before-sample! restore-fn}`. The optional
restore callback runs before every replay outside device timing. `:rounds` defaults to 12 and
`:warmup-rounds` to 3. Each round rotates the starting candidate; complete cycles balance ordinal
position. Persist both chronological `:samples` and per-candidate `:measurements` with the identities
above. This controls one ordering confound, not cache/thermal drift. The summary's `:stationary?`
is only the existing coefficient-of-variation heuristic; it neither proves stationarity nor
selects a winner. No tuning cache or production selector is changed by this measurement API.

`bench/staged_contraction_probe.clj` provides an opt-in internal three-way comparison using this
API. In a `:bench` REPL, require `staged-contraction-probe` and call `run!` with
`{:shape [1 128 32 32] :revision "<git revision>" :environment "<machine/session identity>"}`.
Its returned plain EDN includes input recipe, selected OpenCL device, executable signatures,
binding times, ordered raw device samples and exact dyadic reference validation. It caps reference
work at eight million products and logical resident storage at 64 MiB; these are not total JVM or
driver memory caps. Positive periodic inputs prevent an all-zero oracle. Broader signed and
cancellation correctness remains covered separately by the staged-contraction device tests.

`:comparison-mode :typed-recursive` compares the specialized two-stage generated schedule with
the recursive generated schedule on the same two-stage facts, ABI storage and dyadic oracle.
This is a schedule-regression probe, not a public-source or external-baseline benchmark, and it
does not measure three-stage throughput. Both paths use the default portable dot implementation.

`bench/results/recursive-stage-f714a871.edn` retains two initial laptop samples, including raw
rotating measurements and executable signatures. Both shapes pass the exact oracle. Median
microseconds for specialized/recursive were 21.458 / 20.312 for `[3 5 3 32]` and
12.291 / 10.937 for `[16 64 8 32]`. Every series failed the CV heuristic; these short, noisy runs
do not establish a speedup or justify schedule promotion. The saved revision identifies the
compiler commit plus the uncommitted probe-mode addition used for the measurement.

The saved internal probe `bench/results/staged-paired-fa7319e4.edn` uses 6 warmup rounds and 24
measured rotating rounds on the Intel Arc laptop. Median microseconds for typed/scalar/packed:
`[1 128 32 32]`: 38.958 / 24.166 / 32.708; `[4 128 32 32]`: 41.041 / 19.687 / 32.500.
All candidates passed their host oracle, but every series failed the default CV heuristic.
These warm resident, shared-host measurements do not justify promotion or an external performance
claim. They suggest examining loop/index code and backend vectorization as well as packed loads;
the retained packed path is not necessarily the fastest baseline. The report identifies the exact
measured revision; later runner serialization changes preserve the same recorded numeric samples.

Pin external revisions before measurement. Retain individual shapes and failures; publish
any aggregate only alongside them. Tune on a declared training set and measure held-out
shapes, with a stated budget for each implementation. Do not update reference baselines
automatically during a run. A faster result with changed precision is a separate policy,
not a regression-test pass.

## Current evidence boundary

The production canary retains the original fixed GEMM and accepts an optional `:shape [m n k]`
for the public scalar-dimension `gemm-mnk!` entry point. `gemm-shapes` lists small suggested
cases. Run one case per options file, output file and explicit baseline, using the existing
canary CLI. For example, the options below select an awkward-tail case (replace revision,
environment and device with the actual run identity):

```clojure
{:case :gemm :shape [127 65 33] :target :ocl:0
 :compiler-revision "<exact revision>" :environment-tag "<stable machine/driver tag>"
 :output "bench/results/gemm-127-65-33.edn"}
```

Without a baseline the CLI reports `:unbaselined` and exits 2, rather than blessing its own
measurement. The fixture uses small dyadic inputs and exact reference comparison; it does
not establish accuracy on arbitrary input distributions. The scalar-dimension and original
fixed-shape workloads have distinct comparison identities. There is no automatic ladder
runner, external adapter, tuning experiment or matched accelerator suite yet. No SOTA or
cross-vendor performance claim is justified until comparisons run on the named devices.

`:variant :relu` selects the public contraction with a typed ReLU result transform. An explicit
`:gemm-precision :f32-scalar` or `:mixed-f16-f32` is forwarded to the compiler; the resolved
policy is recorded in the identity. Policy is not evidence that a particular matrix instruction
executed. The explicit ReLU workload has its own identity and is checked through generated
KernelBody candidates. `:variant :relu-composed` uses an ordinary contraction-return alias
followed by an in-place map. Exact destination-return identities are normalized before storage
contracts, so this now executes through the pointwise inout path without bypassing the ABI
checks. It still has two resident stages, versus one for the explicit epilogue: report that
automatic-fusion gap rather than treating the two implementations as equally fused.

The static public composition regression now fuses to one generated stage. This does not close
the parameterized canary gap: normalized dynamic extent computation still separates its producer
and consumer. Report static and dynamic workloads separately, and do not infer measured speedups
from a reduced stage count.
