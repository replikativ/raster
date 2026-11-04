# Hoisted helpers above the JIT size limit

Status: fixed for loop bodies and the compute orchestrator on branch `fix/split-huge-hoisted-helpers` (2026-11-04). Follow-ups below.

## Problem

HotSpot never JIT-compiles a method whose bytecode exceeds 8000 bytes (`-XX:+DontCompileHugeMethods`, the default). Closure hoisting (`raster.compiler.core.closure/hoist-and-compile`) extracts every large top-level binding of the compute body into one static helper, with no upper bound on the size of a single binding, and leaves the remaining small bindings in the `compute` orchestrator. Two shapes exceed the limit:

- a loop with a long body, such as the backward sweep of a reduction after reverse-mode AD;
- the orchestrator itself, once a large program leaves hundreds of small bindings in it.

The method then runs in the bytecode interpreter on every call. Nothing reports this: the program is correct, just much slower.

`METHOD_SIZE_BUDGET` (25000) says C2 "aggressively optimizes methods under ~8KB", but above 8000 bytes nothing is compiled at all. `partition-bindings-by-size`, which was apparently meant to bound method sizes, is not called.

## Evidence

Foerster Marketing's media mix model (three channels, 130 weeks, six-week carryover), with and without a temporal lift experiment, Raster 0.2.1719, JDK 25. Gradient time per call, median of interleaved runs:

| Build | Without experiment (µs) | With experiment (µs) |
| --- | ---: | ---: |
| Stock | 79-97 | 1,860-2,110 |
| `-XX:-DontCompileHugeMethods` | 70-100 | ~100 |
| Loop-body outlining | | 73-95 |
| Loop-body and orchestrator outlining (this branch) | 57-69 | 66-82 |

In the stock experiment build, the hot method was `red_bwd` of the experiment reduction: a `loop*` with 25 loop variables whose body is one `let*` of 881 bindings (estimated 26,765 bytes), 100% interpreted in JFR. Its full NUTS fit was stopped after 90 minutes. The model without the experiment had a JIT-compiled hot loop but an interpreted orchestrator. Values and gradients of every outlined build are bitwise identical to the stock build at the fixture points, and a seeded NUTS run makes the same number of gradient calls.

How to check: record JFR during a gradient benchmark, then look at `frames[0].type` of `jdk.ExecutionSample` events in `jfr print --json`. Interpreted frames in `raster.compiled.*` methods mean this problem.

## Fix on this branch

`raster.compiler.core.outline` cuts large binding blocks into chunks of about `OUTLINE_CHUNK_BUDGET` (2500 estimated bytes). Each chunk becomes a sibling static method:

- its parameters are the chunk's free variables, typed from their binding sites;
- it writes its live-outs (values later bindings or the body still read) into per-call frame arrays (`double[]`, `long[]`, `Object[]`) and returns `0.0`;
- at the original position the block calls the chunk and reads the live-outs back with `aget`.

Evaluation order and every operation are unchanged. A value's frame follows its physical tag at the binding site, so a binding tagged Object is carried as Object and never unboxed by the split.

`hoist-and-compile` applies it in two places when the estimate exceeds `OUTLINE_THRESHOLD` (6000):

1. to an extracted helper, for blocks inside loops; only primitive values cross chunk boundaries, otherwise the block is left unchanged;
2. to the orchestrator's straight-line bindings, carrying references through the `Object[]` frame.

`test/raster/compiler/huge_helper_outline_test.clj` covers the transform and compiles a large loop and a large straight-line body end to end against plain Clojure references.

## Follow-ups

1. **Size check in the validator.** `validate-types` reports Object-typed helper parameters but not size. Measure the emitted method code length after class generation, and warn (or fail under `raster.strict.types`) when any method exceeds 8000 bytes. This would turn the next regression into a compile-time signal, and lets the tests assert that outlining fired.
2. **Calibrate the estimator.** `estimate-bytecode-size` is a heuristic. The thresholds above were chosen so the measured cases JIT-compile. Record estimates against actual code lengths for a few programs and adjust. Remove or use `partition-bindings-by-size` and correct the `METHOD_SIZE_BUDGET` comment.
3. **Presence of primitive producers** (branch `perf/presence-primitive-producers`, not part of this PR). Since #1522-#1526, scalars crossing into helpers are passed as Object unless presence is proven, and optional sums stay as runtime `grad-acc` unless both operands are proven present. Three changes prototyped there remove most of it in the model above: nested scalar operands get checked facts (local products), the ordered reduce starts tagged scalar accumulators at their typed zero (as the additive reduce and scan already do), and `normal-return` recognises casts, core arithmetic and `aget` on physically primitive arrays. Surviving `grad-acc` calls fell from 63 to 33 and runtime `grad-acc` disappeared from the profile, but the measured gain was within noise, because the JIT already inlined `grad-acc`'s Double fast path. These touch the deliberately conservative presence design and need their own review and broader tests. Typed `.invk` implementations return through an Object ABI, so proving their presence needs the selected definition, which `normal-return` only sees when emission inputs are bound.
4. **Live-out pressure.** AD backward bodies read many forward values, so chunks can have dozens of live-outs. The frame traffic is cheap next to interpretation, but splitting at points of low liveness would reduce it.
