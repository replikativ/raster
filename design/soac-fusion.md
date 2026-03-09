# SOAC Loop Fusion Design

Status: **Partially implemented** — vertical and horizontal fusion work via `soac_graph.clj`.
Goal: Fuse AD backward par/map loops to match XLA lax.scan performance.

## Current State

SOAC fusion is implemented in `compiler/passes/parallel/soac_graph.clj` with:
- Vertical fusion (producer→consumer inlining)
- Horizontal fusion (independent same-bound maps merged)
- Fixpoint iteration alternating vertical and horizontal passes
- Pure par/map through the optimizer (materialize pass allocates buffers late)

Current performance (2026-04-10, Valhalla JDK 27):
- MLP f64: 136 µs, MLP f32: 77 µs
- LeNet f64: 222 µs, LeNet f32: 148 µs
- JAX: MLP f64 86 µs, MLP f32 50 µs, LeNet f64 370 µs, LeNet f32 356 µs

## Remaining Gap

The main remaining gap vs XLA is in MLP (1.6-1.5x slower). LeNet is 1.7x faster.

## What XLA Fuses (MLP backward)

```
1. dot_add_fusion:     matmul(W1,x) + bias → single kernel
2. broadcast_max:      relu (max(h, 0)) → fused with matmul output
3. dot_add_fusion:     matmul(W2,a) + bias → single kernel
4. softmax_fusion:     sub_max + exp + sum → single kernel
5. backward_fusion:    cross-entropy grad + softmax backward + SGD → single kernel
6. backward_fusion:    relu backward + weight grad + SGD → single kernel
```

Key: XLA fuses **across operation boundaries** (forward→backward, gradient→SGD).
Our pipeline keeps each par/map! as a separate loop.

## Why Our SOAC Fusion Doesn't Fire

Three blockers identified:

### 1. Bound expression mismatch (FIXED)
`(alength dp)` ≠ `(alength p)` because dp was allocated from p.
**Fix**: resolve-alength runs before soac-fuse (pipeline reorder, done).

### 2. Alias chains hide producer→consumer edges (INVESTIGATED)
AD generates `[dp__3 dp]` `[d_p__4 dp__3]`. The fusion graph's `producer-of`
map only sees the ScalarBinding node for the alias, not the underlying SOAC.

**Options**:
- A: Resolve aliases in the graph builder (tried, works but fused code has
  unresolved symbols — needs alias substitution in fusion output)
- B: Add an alias elimination pass before fusion (CSE copy propagation —
  tried, breaks GSDM test, bytecoder expects bindings to exist)
- C: Use a proper SSA/DAG IR where aliases don't exist

### 3. Multi-consumer producer blocks vertical fusion
`dp` is read by both softmax-backward reduce AND softmax-backward map.
`can-fuse-vertically?` condition #7 requires single consumer.

**XLA approach**: Keeps intermediate buffer for non-fused consumer, or
duplicates cheap computations. We'd need to relax condition #7 with
buffer retention for other consumers.

## IR Design Question

Our let* IR has fundamental limitations for fusion:
- Alias chains (not present in SSA IRs)
- Implicit data flow (dependencies via symbol names, not explicit edges)
- Mixed SOAC/scalar/BLAS bindings in flat sequence
- Mutation-based par/map! (vs functional tensor production in XLA)

**Futhark** uses a similar let-bound IR but runs simplification to
convergence before fusion. Their `InternalRep` has explicit `SubExp`
references and `VName` identifiers with guaranteed uniqueness. Aliases
are tracked via a separate `AliasTable` (transitive closure).

**Our path**: Keep let* IR (maps well to JVM bytecode) but:
1. Add alias elimination before fusion (either in CSE or as a dedicated pass)
2. Build alias-aware producer-of map in fusion graph
3. Support multi-consumer vertical fusion with buffer retention

## Design: Alias-Aware SOAC Fusion

### Phase 1: Alias Elimination (before fusion)
Run a pass that replaces alias bindings `[a b]` with substitution of `a→b`
throughout the body. This is copy propagation but must preserve:
- Hoistable buffer metadata (`:raster.buffer/hoistable`)
- Mutation targets (par/map! output arrays must keep their binding)
- Bindings that the closure builder expects to exist for hoisting

**Approach**: Don't eliminate the binding — instead, build an alias map
and pass it to the fusion graph builder as context. The fusion graph
resolves through aliases when building edges, but the IR stays unchanged.

### Phase 2: Fusion Graph with Alias Resolution
`build-fusion-graph` gets an `alias-map` parameter:
```clojure
(defn build-fusion-graph [soac-nodes alias-map]
  ;; When looking up producer-of for a free sym,
  ;; resolve through alias-map first
  (let [resolve (fn [sym] (get alias-map sym sym))]
    ...))
```

The edges are classified using the resolved symbol (to find the right
SOAC output). The fused code uses the **original** symbols (not resolved),
so no substitution needed in the output.

### Phase 3: Multi-Consumer Vertical Fusion
Relax condition #7 to allow fusion when:
- Producer has other consumers
- Producer output buffer is RETAINED (not eliminated)
- Fused consumer reads from the retained buffer

This is what XLA does — the intermediate buffer stays alive for
the non-fused consumer, but the fused consumer avoids the memory
round-trip.

```clojure
(defn can-fuse-vertically? [graph producer-id consumer-id]
  (and
    ;; ... existing conditions 1-6 ...
    ;; 7. Producer output can be retained if other consumers exist
    (or (empty? (other-consumers graph producer-id consumer-id))
        ;; Other consumers exist: check that producer output is
        ;; written to a buffer that can stay alive
        (some? (soac/soac-output-buffer producer)))))
```

### Phase 4: SGD Fusion
The 4 `sgd-step!` calls at the end are independent par/map! forms on
different arrays (W1, b1, W2, b2). They can't be horizontally fused
(different bounds) but each could be vertically fused with its
gradient producer:

```
backward-dW (par/map!) → sgd-step! W1 (par/map!)  → fuse into single loop
backward-db (par/map!) → sgd-step! b1 (par/map!)  → fuse into single loop
```

This requires sgd-step! to be expanded to par/map! BEFORE fusion runs.
Currently sgd-step! is inlined late (during segop-lower). Moving it
earlier would expose the par form for fusion.

### Phase 5: Matmul Fusion
For small matrices (N < 512), BLAS FFI overhead (~2µs) may exceed the
matmul itself. XLA generates inline matmul for small sizes, fused with
bias addition. We could:
- Skip BLAS below a threshold
- Emit the matmul as a par/map! + par/reduce pattern
- Fuse with bias add and relu

This is lower priority — the main gains come from loop fusion.

## Performance Target

| Operation | Current | After Phase 1-3 | After Phase 4 | XLA |
|-----------|---------|-----------------|---------------|-----|
| Forward pass | 3 loops + 2 BLAS | 1 loop + 2 BLAS | 1 loop + 2 BLAS | 2 fused kernels |
| Backward CE+softmax | 3 loops | 1 fused loop | 1 fused loop | 1 kernel |
| Backward relu+dense | 2 loops + 2 BLAS | 1 loop + 2 BLAS | 1 fused loop + 2 BLAS | 1 kernel |
| SGD (4x) | 4 loops | 4 loops | fused with backward | fused with backward |
| **Total loops** | **9** | **~5** | **~3** | **~3 kernels** |

With phases 1-4, we'd go from 9 separate loops to ~3 fused loops,
matching XLA's fusion level. The remaining gap would be BLAS vs inline
matmul, which is Phase 5.

## References

- XLA HLO dump: `/tmp/xla_dump/module_0035.jit_mlp_step.cpu_after_optimizations.txt`
- Futhark fusion: `../futhark/src/Futhark/Optimise/Fusion/GraphRep.hs`
- Our SOAC graph: `src/raster/compiler/passes/parallel/soac_graph.clj`
- Investigation notes: `memory/f32_parametric_root_cause.md`
