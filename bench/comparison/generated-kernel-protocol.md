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
KernelBody candidates. Composing a contraction-return alias with an in-place map is currently
excluded: its stable-read/output alias boundary is rejected by the binder. Retain that case as
a compiler regression until storage normalization admits it safely; do not benchmark a bypass.
