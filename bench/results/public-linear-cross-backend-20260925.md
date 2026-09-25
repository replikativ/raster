# Public linear schedule: Arc backend event diagnostic

Opt-in REPL run on 2026-09-25, compiler branch `compiler/linear-cross-backend`
(based on `ad6992d8`), Intel Arc Graphics, OpenCL 3.0 NEO and Level Zero.
The same public `raster.dl.nn/linear!` source, `[8,256,256]` shape, mixed FP16
multiplication/FP32 accumulation, constant-weight LinkPlan, four generated
candidates, exact host oracle, three warmups and eight interleaved measured
replays were used for each backend. Transfers, binding and the one-time weight
prologue are excluded. Clock: device events. This is a diagnostic, not a
stationary benchmark or a tuning-cache promotion.

The `:xmx-direct` candidate has two steady-state kernels: activation conversion
and contraction. Each pair below is `[event span µs, kernel-duration sum µs]`
in measurement order, taken from `:comparison :replay-profiles`:

```edn
{:ze:0 [[32.188 30.938] [32.188 30.938] [32.083 30.833]
         [32.708 31.458] [32.708 31.458] [32.188 30.938]
         [106.563 105.313] [32.604 31.354]]
 :ocl:0 [[1394.062 22.291] [1406.561 23.541] [2155.312 28.749]
         [894.374 23.228] [888.229 21.249] [886.562 29.791]
         [1267.082 30.624] [1203.750 26.979]]}
```

The OpenCL span exceeds the kernels' measured work by roughly a millisecond;
Level Zero's gap is about 1.25 µs in every replay. This points to a backend
submission/inter-kernel event gap for the materialized graph, not slow matrix
arithmetic. The one-kernel alternatives do not have this gap. Both backend
series were nonstationary, and the device clocks need not be directly
comparable, so these observations do not justify changing the default selector.

Next gate: repeat under a controlled load and capture complete raw profiles
with the probe; compare against a matched vendor implementation at projection
and throughput shapes before promoting any measured schedule.
