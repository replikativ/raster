# Functional Parallel Combinators

Status: **Step 5 implemented** — pure par/map through IR, materialize pass allocates late.
Goal: Replace mutable par/map!, broadcast!, etc. with functional versions.

## Motivation

Current combinators are mutable: `(par/map! out idx bound cast body)` writes to
a pre-allocated `out` array. This complicates fusion analysis because:
1. The fusion graph needs consumption/anti-dependency edges
2. Alias chains from output→alias→consumer break producer tracking
3. The user must explicitly allocate output arrays
4. Buffer reuse reasoning is entangled with user code

Futhark and XLA use functional models where each op produces a NEW array.
Fusion = expression composition. No consumption edges needed. Buffer
allocation happens at lowering time, not in user code.

## API Change

### broadcast / map
```clojure
;; Old (mutable):
(let [out (n/similar x)]
  (broadcast! [out x y] (+ x y))
  out)

;; New (functional):
(broadcast [x y] (+ x y))   ;; returns new array
```

### reduce (unchanged — already functional)
```clojure
(reduce! [acc 0.0] [x] (+ acc x))  ;; returns scalar — already functional
```

### scan
```clojure
;; Old:
(scan! [out acc 0.0] [x] (+ acc x))

;; New:
(scan [acc 0.0] [x] (+ acc x))   ;; returns new array
```

## Implementation Plan

### Step 1: Add functional macros
Add `broadcast`, `reduce`, `scan` (without `!`) alongside existing `!` versions.
Both expand to the same walker typed-macro, but the functional version:
- Has no output array in the symbol list
- The walker generates the output array name internally
- The output allocation is emitted by the walker expansion

### Step 2: Update all deftm bodies
Replace ~94 `broadcast!` → `broadcast`, ~14 `scan!` → `scan` across:
- nn.clj, dl/nn.clj, dl/optim.clj (neural network ops)
- ode/core.clj, ode/pde.clj, ode/sde.clj (differential equations)
- ad/forward.clj (forward-mode AD)
- sci/*.clj (scientific computing)
- par.clj (standard array ops)

### Step 3: Update walker expansion
`expand-broadcast` in inference.clj generates the output array internally:
```clojure
;; New expansion:
(let [out (n/similar (first inputs))]
  (par/map! out i n cast body)
  out)
```

The `par/map!` form in the IR stays mutable (it's the backend target).
The functional semantics are at the user level only.

### Step 4: Update AD templates
AD templates that emit `broadcast!` calls need to use `broadcast` instead.
This is mechanical — remove the output array from the symbol list.

### Step 5: Functional par/map in IR (DONE)
The IR now uses pure `raster.par/map` through the optimizer pipeline:
- Walker emits `(raster.par/map idx bound cast body)` — no output buffer
- SOAC IR handles pure SoacMap with `:pure? true` flag
- Fusion graph operates on pure forms (no consumption edges needed)
- Materialize pass (`compiler/passes/parallel/materialize.clj`) converts
  pure par/map → alloc + par/map! after fusion, just before backend emission

This enables XLA-style fusion on pure value-producing array forms.

## Migration Strategy

Steps 1-4 are backward compatible:
- Both `broadcast!` and `broadcast` work during migration
- The walker handles both forms
- Tests can be updated incrementally

Steps 1-4 remain as future cleanup work — the critical Step 5 (pure IR)
is done, so fusion benefits are already realized for compiler-generated code.
