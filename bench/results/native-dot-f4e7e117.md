# Native integer-dot candidate comparison

Emitter revision: `f4e7e117dc5e30593c1c6afc8866909487187874`, plus the paired probe
change in this commit. Intel Arc laptop, one capped JVM. This compares two physical
implementations of the **same typed schedule**; both request OpenCL C 3.0. It does
not compare the public compiler against an external baseline.

The probe uses shared Byte inputs, exact positive dyadic reference outputs,
poisoning before every execution, resident binding, and rotating interleaved
device-event measurements. Compilation, transfers, and binding are outside samples.
Both candidates passed the exact oracle at both shapes in both runs.

| Shape `[rows outputs blocks width]` | Warmups / rounds | Emulated median ns | Native median ns | CV emulated / native |
|---|---:|---:|---:|---:|
| `[1 128 32 32]` | 6 / 24 | 37,708 | 14,270 | .555 / .503 |
| `[4 128 32 32]` | 6 / 24 | 40,729 | 13,854 | .087 / .073 |
| `[1 128 32 32]` | 24 / 48 | 37,500 | 13,229 | .038 / .126 |
| `[4 128 32 32]` | 24 / 48 | 40,729 | 13,958 | 1.415 / .046 |

The second run was a predeclared longer-warmup follow-up after observing early
timing changes in the first run. Neither run supplies a pair where both candidates
pass the existing 5% CV heuristic. All samples, including outliers, are retained
in the adjacent paired and long-warmup EDN files, with device information,
executable signatures, input recipe, and chronological measurement order. EDN
round trips were checked against the live results.

Conclusion: native lowering has an observed emulated/native median ratio of
2.64–2.94× at these two tiny shapes, not an established speedup, production
admission, or SOTA claim.
Do not change selectors from these measurements. Next performance evidence needs
larger workload-driven shapes and a more stable machine window; the larger compiler
campaign also still needs external baseline comparisons through public entry points.
