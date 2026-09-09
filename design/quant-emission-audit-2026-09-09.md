# Quantization emission inventory

Measured against main `fb4677aa695cca1be33a1f2133a6fcf8e5a9c27b`, whose tree equals the
tested mixed-storage PR475 tree. This is compile coverage, not throughput or numerical validation.

## Scope and method

Enumerated source `deftm` vars with `coverage/corpus-vars` in `raster.quant.kernels-k`,
`raster.quant.kernels`, and `raster.compiler.fixtures.staged-contracts` (33 entry points).
Called `coverage/report-var` in one capped existing REPL, target `:ocl:0` (Intel Arc).
Used Float policy except the two Double-output staged fixtures, which require Double policy.
The initial uniform Float run correctly rejected those two result/storage mismatches; their
correct-policy reruns are included below.

Wrapped `segop-opencl/generate-staged-contraction-kernel` with a recording delegate while
compiling, preserving its behavior and counting calls even if a compile subsequently failed.
The two Double-policy fixtures were also rerun with this instrumentation.

| Route | Entry points | Emission evidence |
| --- | ---: | --- |
| TypedSOAC | 20 | One KernelBody artifact per entry point |
| Scalar | 9 | No GPU artifact; not a GPU success |
| Compatibility | 2 | Two compatibility-effect-opencl kernels each |
| Compile error | 2 | See failures below |

There were zero legacy staged-emitter calls in this inventory. This does **not** prove the
emitter dead: general/scientific callers, direct compatibility APIs, other compile policies and
shape specializations are outside this measurement. No production emitter is deleted by this audit.

## Actual remaining hot-path debt

`quant-act-q8k-rows-gpu!` and `quant-act-q8k-padded-rows-gpu!` still use compatibility effect
emission. Their public projections (`qmatmul-q4k-dp4a-rows!`, `qmatmul-q6k-dp4a!`,
`qmatmul-q6k-gpu!`, `qmatmul-i8-gemm!`, `i8gemv-dp4a!`) use KernelBody. Improving the activation
packing vertical is therefore more immediately useful than adding more staged projection fixtures.

`quant-act-i8-rows-gpu!` fails with `:segmap-emission-refused`. Its typed scalar lowering attempts
to convert a Double argument of `Math/round` to Long before the operation and declines
`:cast-policy`. The generic intrinsic lowerer currently coerces ordinary function arguments to
the result's expected dtype; Java round has different input and output dtypes. The shared
intrinsic registry also documents a C-backend negative-half-tie deviation. Fix this as a shared
typed numeric operation, not a quantization special case or unchecked Float-to-Long cast.
Validation must include signed ties, adjacent representable values, NaN/infinity and saturation;
native C round and a naive floor(x+0.5) are not assumed interchangeable with Java semantics.

`wi8-dot-q4-x8` fails the typed fixpoint boundary on an untagged conditional binding `nib`.
It needs a separate retained-source-type investigation; do not repair it with an emitter-local
function/type guess.

The nine scalar routes are the composable Q4/Q6 helpers and packed CPU dot variants:
`qmatmul-q4k-composable!`, `qmatmul-q6k-composable!`, `qmatmul-q4-composable`,
`qmatmul-q4-composable!`, `qmatmul-q4-x8!`, `qmatmul-q8-x8!`, `wi8-dot`, `wi8-dot-q4`,
`wi8-dot-q8-x8`. Their intended CPU/C use must be distinguished from missing GPU coverage.

## Next work

1. Correct the shared typed rounding signature/semantics and validate the public I8 quantizer.
2. Trace Q8_K activation packing's effect/control admission; retain row-local writes and ownership.
3. Fix the packed CPU helper's lost conditional type through existing inference.
4. Extend this inventory to remaining scientific/compatibility callers before staged-emitter removal.
5. Measure packing plus projection end to end, separately from resident projection-only timing.

The default CI corpus currently excludes both quant namespaces. Keep this focused inventory in
the cold lane until an explicit workload/policy matrix can ratchet intended GPU entry points
without treating CPU-only helpers or inappropriate dtype specializations as regressions.
