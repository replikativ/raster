# Equation-first generated NN probe

This opt-in probe compares `:portable` and `:register-tiled` through the public
equation-first compilation, instantiation and resident execution path. It does
not add kernels, change automatic scheduling, or run timing assertions in CI.

In a warm `:dev` REPL, from the repository root:

```clojure
(load-file "bench/equation_tile_probe.clj")
(equation-tile-probe/run!
 {:target :ocl:0 ; repeat with :ze:0
  :shape [8 256 256] ; M N K; also try [256 256 256]
  :revision "<git HEAD>"
  :environment "<device, driver, JVM, other activity>"
  :rounds 12 :warmup-rounds 8})
```

The independent CPU dot-product oracle runs before timing. Deterministic dyadic
FP32 inputs allow exact comparison. Each candidate owns separate resident output
storage. Candidate order rotates between rounds; samples are device-event spans,
excluding compilation, transfers and validation. Preparation and binding times,
cache reports, executable signatures and raw event samples remain in the result.
Resources close on completion or failure. The precision policy permits target
contraction, but both storage and accumulation remain FP32: this is not FP16 GEMM.

The probe caps CPU oracle work at 64 Mi multiply-accumulate iterations and logical
array payload at 4 Mi elements, before allocation. It is not a large-model benchmark.
Preparation/binding times depend on candidate order and cache state; they are diagnostic
metadata, not a controlled comparison of compilation speed. Cleanup attempts every owned
candidate even if one close fails, preserving the primary exception.

Check `:stationary?`, raw samples and driver provenance before drawing conclusions.
A long-lived REPL may retain an old hardware registry; source reload alone does not
refresh detected descriptors. Start a fresh REPL for a published baseline if its
registry no longer reflects current detection. Do not overlap reference runs with
the Raster measurements. External CLBlast measurements use queue markers and a
different floating-point policy, so label those differences explicitly.

## 2026-09-27 diagnostic

Production tree: `c9e1f001`; this probe was added afterward. Shared Intel Arc laptop,
HotSpot 25, OpenCL driver `26.05.37020.3`. Both strategies passed the independent CPU
oracle on OpenCL and Level Zero for `[8 256 256]` and `[256 256 256]`. The original
warm registry omitted driver provenance; those timings are not published baselines.

After refreshing the OpenCL descriptor from actual discovery, `[8 256 256]` still
showed nonstationary timing with eight warmup rounds and twelve measured rounds:

| Strategy | Median ns | CV | Stationary |
|---|---:|---:|---|
| Portable | 372916 | 0.20236 | no |
| Register-tiled | 221666 | 0.31674 | no |

Chronological samples per candidate (rotating interleaved execution):

```clojure
{:portable [387395 372291 372916 378437 373229 373750
            373645 371458 370729 414583 192604 192604]
 :register-tiled [300625 221041 221666 308645 229166 221979
                  222604 221770 220520 88229 147708 90625]}
```

The existing native CLBlast 1.7.0 runner also passed its CPU oracle on both shapes.
These were separate, unbracketed diagnostic runs, not a controlled external ranking.
No schedule is promoted and no competitive-performance claim follows from these
measurements. Repeat on a quiet device and retain full result provenance before
using a result for tuning. NT/transposed weights are outside this NN probe.
