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

## First repair: shared typed Java rounding

The typed scalar lowerer now consumes the existing intrinsic table's Float→Int and Double→Long
source signatures. It expands rounding into ordinary floor, subtraction, comparison, selection,
and saturating conversion operations; no quantization-specific kernel or new IR node is involved.
The overload's result dtype is preserved before conversion to its consumer dtype. Unknown source
overloads decline instead of being guessed from the output type.

OpenCL sanitizes NaN to floating zero before its explicit saturating conversion: the first CI
PoCL run otherwise returned signed MIN_VALUE for NaN despite the Arc oracle passing. The same
edge-case tests remain required on both devices. CUDA/HIP guard NaN and signed range boundaries
before a truncating C++ cast; the upper guard compares against the exact power-of-two boundary,
not a rounded integer maximum. Other unsupported rounding policies still decline.

The targeted I8 packer rerun now reports validated TypedSOAC, one KernelBody kernel, independent
effect iteration, and no emission decline. On Intel Arc, both rounding overloads match Java for
164 inputs each (edge cases plus deterministic random bit patterns). The public packer also
matches JVM packed words and scales across three rows, including half-ties and a zero row.
Both overloads and the public packer join the CUDA/HIP compile fixtures. This is correctness and
emission evidence, **not a throughput measurement**. The original 33-entry table remains the
pre-repair baseline; Q8_K compatibility emission and the packed helper's lost type remain open.

## Q8_K scale ownership

Both row quantizers now publish each super-block scale only from sub-block `j=0`. Previously
eight work-items performed non-atomic writes to the same slot; equal values do not establish
race freedom. The valid-input contract remains complete launches over rows whose physical width
is a multiple of 256. Packed-word and block-sum ownership, scratch size, two-phase scheduling,
and the public ABI are unchanged. CPU row/padding checks, emitted store-guard checks, and real
device quantization-to-Q4_K projection chains pass (five focused tests, 30 assertions).

The second phase still declines TypedSOAC admission. Its store loop also carries a running sum;
the frontend currently admits index-only store loops. The next generalization must preserve
result-carrying effects and their order through the existing effect-region and KernelBody loop
contracts. Do not conceal this with a quantization-specific emitter or a claimed typed route.

## Scheduled carried-effect prerequisite

The shared scheduled scalar region can now express a single typed carry/result on an ordered
store loop. JVM lowering binds the result around the following effects; GPU lowering uses the
existing KernelBody LoopArg/ForLoop/Yield contract. Loop locals and parameters do not escape,
zero-trip loops return their initializer, and stores precede recurrence computation. Lexical
sibling binders are renamed into collision-free KernelBody SSA identities.

This first landing was a **scheduled-region prerequisite**, not new TypedSOAC source admission.
The public Q8_K packers still use the measured compatibility route and unchanged two-phase schedule.

Carried regions require retained compound source types and reject unsupported conversions rather
than inheriting map emission's device-wrap defaults. Checked integer arithmetic retains its width
before a floating carry conversion. Generated FP32 storage conversions use IEEE rounding/overflow
in both JVM and KernelBody; they are distinct from user-written checked casts. Source-shaped
compatibility emission is forbidden for this new scheduled contract. Nested loops inside a carried
loop and multiple carry/result slots remain deliberately unsupported.

Focused validation covers zero/one/eight iterations, continuation scope, sibling reuse, lower-bound
substitution, shadowing of cast names, source overflow order, and fail-closed conversion/type cases.
The OpenCL device oracle checks two independent rows and poisoned output tails. The same scheduled
fixture joins hardware-free CUDA/HIP compiler gates; this does not claim vendor-device execution
or any packing throughput improvement yet.

## Canonical carried-effect projection

The existing `effect-loop` also has an explicitly result-bearing form:

```clojure
(effect-loop {:index k :lower 0
              :carry {:parameter acc :result sum :dtype :float}}
  extent initial
  (lambda [k acc]
    (effect-region [typed-local ...] [ordered-effect ...] update)))
```

Initializer and update are grammar-visible scalar operands, not expressions hidden in attribute
maps. The result-bearing region is legal only as a carried-loop body. Common projection preserves
it into the scheduled region; validation threads result scope through subsequent effects and
checks destination reads against read-write storage and explicit read effects. Existing pure-map
fusion does not absorb these ordered effect maps.

Direct canonical program envelopes and scheduled JVM/KernelBody emission are exercised. The
production source-realization adapter explicitly refuses carried loops until its continuation
binding is implemented; frontend recognition is still closed. A capture whose physical symbol
collides with a carry binder/result currently fails scheduled validation rather than being silently
captured. General hygienic physical-name projection is a follow-up before claiming unrestricted
symbol-ID composability. No throughput claim or public quantizer migration follows from this slice.
