# Test feedback and performance evidence

The fast loop is focused tests in a correctly configured reusable JVM. Full correctness tests
run in isolated CI processes. Do not parallelize test vars inside one JVM: tests redefine Vars
and share runtime/dispatch state. Keep local JVM concurrency bounded by available memory.

## Measure before repartitioning

`scripts/ci-test-shard.sh` currently balances four processes by source bytes, not observed cost.
On CI run 6675, the test phases took 143, 221, 242 and 566 seconds. This shows an imbalance; it
does not identify which tests are redundant or prove that the compiler itself became slower.

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
2. Rebalance processes using measured costs and verify every namespace remains assigned once.
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
