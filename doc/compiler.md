# Compiler Pipeline

Raster includes a nanopass compiler that transforms `deftm` function bodies
through a sequence of IR dialects, producing optimized JVM bytecode with
zero-allocation inner loops, BLAS dispatch, and optional SIMD vectorization.

## Pass Sequence

```
walked → lowered → fixpointed → DCE → buffer-fused → late-cleanup
       → loop-lifted → write-read-fused → SOAC-fused → materialized
       → segop-lowered → compound-detect → [SIMD | OpenCL | Vulkan]
       → resolve-alength → mem-merge
```

Use `(pipeline/explain-pipeline #'my-fn)` to inspect the transformation at
each stage.

## Key Passes

### Walker Devirtualization

The walker resolves dispatch at definition time. When it encounters a call
like `(raster.numeric/+ x y)` and knows the argument types, it rewrites
the call to a direct `.invk` method on the typed interface — giving
primitive-speed evaluation without boxing.

### Buffer Fusion

The fusion pass (`buffer-fuse`) performs escape analysis on the flattened IR:
if a buffer is dead after an operation, the next allocation reuses it in-place.
This eliminates intermediate array allocations in chains of map/reduce operations.

### SOAC Fusion

The SOAC fusion pass (`soac-graph`) builds a dependency graph of second-order
array combinators (map, reduce, scan) and fuses compatible chains — vertically
(producer-consumer) and horizontally (independent maps over the same input) —
into single kernels, minimizing memory traffic. Inspired by
[Futhark](https://futhark-lang.org).

### SIMD Vectorization

Parallel array operations compile to JDK Vector API intrinsics via the SIMD
backend. The compiler maps element types to vector species and emits
vectorized loops with scalar tails.

### Memory Merge

The final pass identifies buffers with non-overlapping lifetimes and merges
them into shared storage, reducing peak memory usage.

## Compilation Modes

### Lazy JIT

`deftm` supports lazy JVM compilation: normally the first invocation triggers
compilation and subsequent calls use the compiled code. Explicit compiler
preparation can materialize an implementation before numerical execution.

Before emission takes dependency snapshots, a separate preparation step closes
missing source-walk caches of referenced callees from their frozen frontends.
Discovery respects lexical bindings and quoted data; unrelated functions are
not warmed. Earlier observations are not refreshed: a preparation-induced change
requires a whole construction retry. Source preparation and executable-callee
readiness are distinct obligations; neither permits retrying numerical execution
or accepting an old artifact after its retained definitions change.

### AOT via `compile-aot`

`compile-aot` inlines an entire call chain into a single JVM method. This
gives HotSpot C2 full visibility for optimization across function boundaries:

```clojure
(require '[raster.compiler.pipeline :as pipeline])

(def fast-fn (pipeline/compile-aot #'my-fn))
```

Options:

- `:inline?` — inline `deftm` calls (default `true`)
- `:simd?` — apply SIMD optimization (default `true`)
- `:dtype` — numeric dtype (`:double` or `:float`)
- `:target-device` — device for GPU backend (e.g. `:cuda:0`, `:ze:0`)

### Native C and scalar product storage

Use `(pipeline/compile-aot #'my-fn :target :c :dtype :double)` for native C.
Non-escaping scalar product intermediates can be eliminated when the actual
JVM entry, selected helper implementations, and every consumer establish the
same Double value. This removes box storage, not operand evaluation; arithmetic
and source order remain unchanged. A type hint alone is not sufficient evidence.

This storage-elimination path currently requires Double-only parameters and
result, Double emission policy, and a closed scalar body. Identity observations,
checked integral work, arrays, and unclosed AD entry conversions are not admitted
by this path. Other native lowering retains its existing coverage.

The admitted path checks exact call arity, converts every argument in source
order (including unused arguments), then checks retained source/producer
freshness before the native call. A stale compiled function must be recompiled;
the guard does not restamp it. Argument conversion failures retain precedence
over stale-program errors.

### Diagnostics

```clojure
;; Show stage-by-stage compilation with diffs
(pipeline/explain-pipeline #'my-fn)

;; Show type information for all bindings
(pipeline/explain-types #'my-fn)

;; Get intermediate forms as data
(pipeline/show-pipeline #'my-fn)
```

## Purity Analysis

Before AD or GPU compilation, [beichte](https://github.com/replikativ/beichte)
validates function purity on a four-point lattice
(`:pure < :local < :mutation < :io`). Side effects are caught at definition
time rather than producing silently wrong gradients or GPU kernels.

## Compilable `defn`

Raster provides `raster.core/defn`, a drop-in replacement for `clojure.core/defn`
that stores the walker-analyzed function body as var metadata. At runtime it
behaves exactly like a normal Clojure function. But `compile-aot` can see
through it, inlining the body into compiled pipelines.

```clojure
(ns my.app
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]))

(defn my-helper [x y]
  (+ (* x x) (* y y)))
```

Use `raster.core/defn` for functions that don't need typed dispatch but should
be visible to the compiler. Use `deftm` when you need typed dispatch with
`:- Type` annotations.
