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
