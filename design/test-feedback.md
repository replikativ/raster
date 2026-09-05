# Test feedback and performance evidence

The fast loop is focused tests in a correctly configured reusable JVM. Full correctness tests
run in isolated CI processes. Do not parallelize test vars inside one JVM: tests redefine Vars
and share runtime/dispatch state. Keep local JVM concurrency bounded by available memory.

## Measure before repartitioning

`scripts/ci-test-shard.sh` balances four processes with reviewed load+test timing weights.
On CI run 6675, the test phases took 143, 221, 242 and 566 seconds. This shows an imbalance; it
does not identify which tests are redundant or prove that the compiler itself became slower.
The bootstrap weights come from instrumented CI job 6687 and are committed in
`test/resources/ci_test_timings.tsv`. New test files receive a source-size estimate calibrated
against measured files still present in the checkout. No files are dropped when weights are
missing; a missing default baseline uses the original size policy. An explicitly requested
missing/invalid baseline fails before selection.

Regenerate weights explicitly with `raster.ci.timing-weights OUTPUT SOURCE REPORT...` using
downloaded EDN reports from comparable general-suite CI runs, not the OpenCL lane. Review the
result; no test run updates the baseline automatically. Repeated namespace samples use their
maximum as a conservative seed. This remains a heuristic for placement, not a performance gate.

Both the general shards and OpenCL CPU job now use `raster.ci.timed-runner`, a scoped timing
wrapper around the existing Cognitect runner. Test selection, metadata exclusions, fixtures,
assertion reporting and failure summaries remain unchanged. Each process publishes
`test-results/timings.edn` as a CI artifact, including:

- per-namespace load and test nanoseconds;
- per-test-var execution nanoseconds (nested inside namespace time, **not additive**);
- observed load/test order and JVM version.

Load time includes dependencies first reached by that namespace. Do not treat one sample as an
intrinsic namespace cost: compare multiple runs, preserve process startup overhead separately,
and remeasure after partition changes. Timing instrumentation is diagnostic, not a noisy CI
wall-time regression assertion.

Landing order:

1. Collect load/test timings without dropping coverage.
2. Rebalance processes using measured costs and verify every namespace remains assigned once
   (implemented; remeasure and refresh after observing the new placement).
3. Isolate measured corpus/compiled-training bottlenecks; split indivisible work with an explicit
   completeness check if necessary.
4. Avoid repeating device-free structural tests in the OpenCL lane, while preserving actual
   numerical device execution and explicit capability gates.
5. Share repeated immutable, option-keyed compile fixtures only where tests do not redefine or
   mutate compiler state. Preserve independent numerical and source-code oracles.

## What current performance evidence does not establish

Passing correctness, route/allocation checks and nvcc/hipcc compilation does not establish GPU
speed or parity with vendor GEMM implementations. The current `:perf` canary times a plain
Clojure loop, not a Raster-compiled function. The retained cold GEMM benchmark invokes the
independent source oracle rather than the production TypedSOAC route. Neither is a sufficient
performance regression gate for the current compiler-generated GEMM implementation.

Next performance work must benchmark the public compilation/LinkPlan route with explicit
shape, dtype, selected schedule, target/driver identity, cold compilation, warm device execution,
and transfer/allocation costs. Keep a small stable-machine canary set outside ordinary CI, then
run broader differential workloads against external libraries periodically. Missing measurements
or hardware are missing evidence, not successful performance gates. Never update a baseline
merely to accept a slowdown.

## Opt-in production-route canaries

`clojure -M:test:production-canary /path/to/options.edn` runs either `:case :cpu`
(Raster AOT sum-of-squares) or `:case :gemm` (public compiled/lower, instantiate, resident
LinkPlan replay of a 64×64×64 contraction). Example options:

```clojure
{:case :gemm :target :ocl:0
 :environment-tag "machine-name; driver-version; power-mode"
 :compiler-revision "exact-commit-and-any-local-patch-label"
 :baseline "/path/to/reviewed-gemm-baseline.edn"
 :output "/path/to/new-gemm-measurement.edn"}
```

Both routes validate against independent numerical references before measuring. CI executes
the CPU route and, in the OpenCL lane, the resident GEMM route with timing replaced by one
replay: these are correctness checks, not performance gates. There is no GPU timing claim
when hardware is unavailable.

The GPU metric is **host-synchronized warm resident replay**, including runtime submission
overhead, not device-only kernel latency or peak GEMM throughput. Compilation and binding
are reported separately; upload/download are outside timed replay. This tiny shape catches
route/runtime regressions but does not establish SOTA performance. Device-event profiling of
equation-first prepared programs is not yet supported by `link/measure!`.

The comparison identity includes shape, dtype, numerical policy, device/host/JVM signatures,
timing scope and an explicit machine/driver tag. Compiler revision and emitted executable
signatures are evidence, not comparison keys: a source change must not bypass the old baseline.
Missing baselines, incomparable environments, noisy measurements and invalid measurements
exit nonzero. A stationary median above 115% of baseline also fails. Review repeated initial
measurements and explicitly designate a separate baseline file; the runner never seeds or
updates it. Compilation/binding timings are diagnostic, not yet regression-gated.

The canary directly returns its scheduled contraction: result alias propagation resolves the
common ABI's `:result` buffer without weakening resident-plan validation. This also covers
the public-boundary defect discovered while introducing the canary.
Primitive literal casts now pass through one checked constant boundary shared by scan/reduction
certification, SegRed identity emission and matrix schedule legality. It evaluates supported
Clojure casts rather than stripping them, and compares integer/floating values exactly. Unknown
or failing conversions provide no evidence. The canary uses `(float 0.0)` to exercise this
path; original initializer expressions remain in the retained IR.
