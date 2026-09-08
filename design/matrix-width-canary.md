# Matrix-width regression canary

This opt-in probe compares the generated typed DPAS matrix leaf against the historical test-only
GEMM source, using the same tile, ABI, FP16 inputs and FP32 accumulation/output. It does not time
public compilation, layout conversions, transfers, an entire model, or an external SOTA baseline.
It never runs GPU timings as part of the unit suite or writes an autotuning cache.

From a GPU-accessible test REPL:

```clojure
(require '[raster.perf.matrix-width-canary :as canary])
(canary/run! :ze:0 [[128 128 128] [256 256 256] [32 256 1024]]
             :revision "<git commit; note any dirty compiler changes>"
             :driver-identity "<installed Intel compute driver version>")
```

Record the complete returned EDN, including raw samples, candidate hashes, hardware identity and
stationarity flags. Unknown revision/driver identity is explicit, not filled from an assumption.
Shapes are bounded before any device access. Both candidate orders run; each candidate must
overwrite NaN-poisoned output and match an independent CPU FP16-input reference. Reset, validation,
allocation and compilation are outside the measured device-event interval. Compilation time is
reported separately. The reset also runs before each measured replay.

## Initial Arc observation, 2026-09-08

[Raw protected-output run](measurements/2026-09-08-matrix-width-arc.edn), compiler revision
`3ebeab431b9c4e70b19d8521a55fc6cd08fe11a2+uncommitted-canary`, Java 25.0.1, Level Zero Intel Arc. Driver version was
not recorded; this is an exploratory observation, not an accepted performance baseline.

Each cell reports generated/reference median microseconds, including nonstationary runs:

| M×N×K | Round 0 | Round 1 | Both candidates stationary |
| --- | ---: | ---: | --- |
| 128×128×128 | 13.646 / 13.438 | 13.438 / 13.438 | Round 1 only |
| 256×256×256 | 24.583 / 20.313 | 24.479 / 24.583 | Round 1 only |
| 32×256×1024 | 68.438 / 68.229 | 68.229 / 68.125 | Round 0 only |

All 12 numerical validations passed; relative L1 error was at most 1.24e-7. Five measurements
exceeded the 0.10 coefficient-of-variation threshold. The stationary pairs are close, but selecting
only those pairs would discard contradictory/noisy observations. These data do not establish
performance parity, peak throughput, or SOTA competitiveness. Stabilize and repeat measurements
before treating a change as a performance win or regression.

## Contract/typed-store repeat, 2026-09-08

[Raw repeat](measurements/2026-09-08-matrix-contract-arc.edn), compiler
`b7f53d26dfe3e1fd5944acb3836a662951ee44d5`, Intel Arc Level Zero, installed
`libze-intel-gpu1` and `intel-opencl-icd` version `26.05.37020.3-0`.
The generated candidate now includes the surface contract and wide C addressing; both candidates
retain the same ABI preconditions. The legacy source remains a test-only comparison.

| M×N×K | Round 0 generated/reference µs | Round 1 generated/reference µs |
| --- | ---: | ---: |
| 128×128×128 | 15.521 / 13.542 | 13.229 / 13.438 |
| 256×256×256 | 26.042 / 24.167 | 25.833 / 24.167 |
| 32×256×1024 | 71.458 / 69.688 | 71.458 / 68.750 |

All twelve numerical validations passed (maximum relative L1 1.24e-7), but **all twelve timing
series failed stationarity**. Several medians favor the reference; do not dismiss that signal,
but do not infer a compiler regression from this noisy shared-laptop run. The 128³ ordering
reverses between rounds. Preserve all raw samples, establish a quiet repeat with stable clocks,
then isolate the generated wide-addressing change if the gap persists. Do not relax correct
index arithmetic to recover an unverified timing difference. No autotuning winner was published.
