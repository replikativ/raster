# Typed Multiple Dispatch

Raster extends Clojure with Julia-style typed multiple dispatch via `deftm`
and `ftm`. This gives Clojure programs three things: polymorphic arithmetic
that works with custom number types (Dual numbers, Jets, complex), primitive
performance through typed interfaces (`.invk`), and compile-time
devirtualization that eliminates dispatch overhead entirely.

## The Problem

Clojure's `defn` produces `IFn` — a single-dispatch, boxed interface.
Calling `(+ x y)` where `x` and `y` are `double` goes through:

```
IFn.invoke(Object, Object) → Object
```

Every argument is boxed, every return is boxed, and there's no way to
dispatch on argument types. This is fine for general programming but kills
numerical performance — a tight loop doing `(+ x y)` on doubles pays ~20ns
per call for boxing alone.

Julia solves this with multiple dispatch on concrete types. Raster brings
the same idea to Clojure.

## `deftm`: Typed Method Definition

```clojure
(deftm add [x :- Double, y :- Double] :- Double
  (raster.numeric/+ x y))
```

This registers a **method entry** in the dispatch table keyed by
`[add, arity=2, tags=[Double Double]]`. Multiple overloads coexist:

```clojure
(deftm add [x :- Double, y :- Double] :- Double ...)
(deftm add [x :- Long, y :- Long] :- Long ...)
(deftm add [x :- (Array double), y :- (Array double)] :- (Array double) ...)
```

At call time, Raster's dispatch engine matches argument types against
registered methods and selects the most specific match.

### What `deftm` Generates

For each method, `deftm` produces two things:

1. **Boxed fallback** — a plain `defn` with a mangled name:
   ```clojure
   (defn add_m_double_double [x y] ...)
   ```

2. **Typed fast-path** — a `deftype` implementing a generated interface:
   ```clojure
   ;; Generated interface (cached globally):
   (gen-interface
     :name raster.fn.IFn__double_double__double
     :methods [[invk [double double] double]])

   ;; Typed implementation:
   (deftype add_m_double_double_Impl []
     raster.fn.IFn__double_double__double
     (invk [_ x y] ...body...))
   ```

The `.invk` method accepts and returns unboxed primitives. Calling it
costs zero boxing — the JVM passes `double` values in registers.

## Runtime Dispatch

When you call a `deftm` function at runtime (without compilation), the
dispatch engine:

1. Looks up the dispatch table for the function name + arity
2. Iterates through method entries, checking `instanceof` for each argument
3. Selects the most specific match (by type specificity ordering)
4. Invokes the method

```clojure
;; This dispatches at runtime:
(raster.numeric/+ 3.0 4.0)  ;; → finds [Double Double] entry, returns 7.0
(raster.numeric/+ 3 4)      ;; → finds [Long Long] entry, returns 7
```

For the Lorenz attractor ODE benchmark, runtime dispatch adds ~2% overhead
vs hand-optimized code — negligible.

## Compile-Time Devirtualization (The Walker)

The real performance win comes from the **walker** — a source-to-source
transformer that resolves dispatch at definition time.

When a `deftm` body is walked, the walker knows the types of all parameters
(from `:- Type` annotations) and can resolve every `raster.numeric/+` call
to a concrete implementation:

```clojure
;; Source:
(deftm f [x :- Double, y :- Double] :- Double
  (raster.numeric/+ x y))

;; After walking:
(let* [result (.invk raster.numeric/_plus__m_double_double-impl x y)]
  result)
```

The polymorphic `raster.numeric/+` is replaced with a direct `.invk` call
on the typed implementation object. The JVM inlines this to raw `dadd`
bytecode after warmup.

### Type Propagation

The walker maintains a type environment `{symbol → tag}` as it walks
bindings:

```clojure
(deftm f [x :- Double, y :- (Array double)] :- Double
  (let [n (raster.arrays/alength y)    ;; walker infers: n is Long
        v (raster.arrays/aget y 0)]    ;; walker infers: v is Double
    (raster.numeric/+ x v)))           ;; resolves to Double+Double
```

Types flow through let bindings, function returns, and array operations.
The walker uses the return type annotations of resolved methods to infer
binding types, enabling cascading resolution.

## Function Parameters: `(Fn [...] ...)`

When a function is passed as an argument, Raster can still avoid boxing
if the parameter is annotated with a function type:

```clojure
(deftm bisect [f :- (Fn [Double] Double),
               a :- Double, b :- Double] :- Double
  (let [fa (f a)]  ;; emits (.invk f a) → primitive double → double
    ...))
```

The walker sees `f` has type `(Fn [Double] Double)` and emits
`(.invk f a)` — calling the typed interface directly. The caller must
pass an `ftm` (typed anonymous function) that implements the same
interface:

```clojure
(bisect (ftm [x :- Double] :- Double (* x x)) 0.0 2.0)
```

This is how ODE solvers achieve primitive performance with user-supplied
right-hand-side functions — the solver is generic over any
`(Fn [(Array double) Double (Array double)] ...)`, but the actual calls
use `.invk` with zero boxing.

## Parametric Polymorphism: `(All [T])`

For functions that work across numeric types:

```clojure
(deftm dot (All [T] [xs :- (Array T), ys :- (Array T)] :- T)
  (par/reduce acc (raster.numeric/zero T) i (raster.arrays/alength xs)
    (raster.numeric/+ acc (raster.numeric/* (raster.arrays/aget xs i)
                                             (raster.arrays/aget ys i)))))
```

`(All [T])` registers a **template**. When first called with concrete types
(e.g., `(Array double)`), Raster auto-generates a concrete specialization
`dot_m_doubles_doubles` by substituting `T → double` in the body and
evaluating the result.

The default specialization (for `double`) is generated at definition time.
Other types (float, Dual, Complex) are specialized on first use via
`try-parametric-dispatch`.

## Performance Tiers

The dispatch system creates a natural performance pyramid:

| Tier | Mechanism | Example | Overhead |
|------|-----------|---------|----------|
| **Generic** | Runtime dispatch, `Object[]` | `(raster.numeric/+ x y)` at REPL | ~20ns/call |
| **Walked** | `.invk` on typed interfaces | `deftm` body after walker | ~0ns (JIT inlines) |
| **Compiled** | Full bytecode pipeline | `compile-aot` | C2 auto-vectorization |

The key insight from the ODE benchmarks: **Tier 2 alone beats Julia**.
The walker's `.invk` devirtualization, without any further compilation,
gives the JVM enough information to produce competitive native code.

## ODE Benchmark: Lorenz Attractor

The Lorenz system is a standard benchmark for ODE solvers:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

with σ=10, ρ=28, β=8/3, integrated from t=0 to t=100 with adaptive
step control (atol=1e-8, rtol=1e-6).

```clojure
(deftm lorenz! [state :- (Array double), t :- Double,
                deriv :- (Array double)] :- (Array double)
  (let [x (aget state 0) y (aget state 1) z (aget state 2)]
    (aset deriv 0 (* 10.0 (- y x)))
    (aset deriv 1 (- (* x (- 28.0 z)) y))
    (aset deriv 2 (- (* x y) (* (/ 8.0 3.0) z)))
    deriv))
```

### Results (JDK 27, 2026-02-20)

| Implementation | Tsit5 | DP5 | Julia |
|---|---|---|---|
| `defn` (boxed IFn) | 504µs | 458µs | — |
| `ftm` (.invk) | 489µs | 452µs | — |
| `deftm` dispatch | 500µs | 461µs | — |
| **Julia DifferentialEquations.jl** | **613µs** | **583µs** | — |

Raster is **~20% faster than Julia** across the board. The `deftm` dispatch
path (which uses the full dispatch table at runtime) adds zero measurable
overhead vs the hand-optimized `defn` path — the JVM's C2 compiler
devirtualizes the interface calls after warmup.

## AD Through ODE Solvers

Because ODE solvers are written with `deftm` and `raster.numeric` operators,
forward-mode AD works transparently — pass Dual numbers and derivatives
propagate through the solver:

```clojure
;; Standard ODE solve:
(ode/solve lorenz! [1.0 1.0 1.0] [0.0 100.0] (ode/->Tsit5Cache ...))

;; Sensitivity via Dual numbers (same solver, no code changes):
(ode/solve lorenz-dual! [Dual(1,∂σ) Dual(1,0) Dual(1,0)]
           [0.0 100.0] (ode/->Tsit5Cache ...))
```

The Lotka-Volterra parameter estimation benchmark shows this in practice:

| Implementation | Single gradient | 200-iter GD | vs Julia |
|---|---|---|---|
| Generic dispatch (Object[]) | 4.2ms | 815ms | 113x slower |
| Devirtualized (.invk) | 15.0µs | 5.8ms | **0.81x** |
| Julia ForwardDiff.jl | 15.9µs | 7.2ms | 1.0x |

The devirtualized path (Tier 2) is 19% faster than Julia's ForwardDiff.jl
for ODE sensitivity — using the same solver code, just with Dual number
inputs.

## How It Ties Into Clojure

Raster's type system is **additive** — it doesn't replace Clojure's
semantics, it extends them:

- `deftm` functions are also `IFn` — callable from normal Clojure code
- `defvalue` types are also `ILookup` — support `(:field obj)` access
- Runtime dispatch handles mixed types gracefully (Long + Double → Double)
- The REPL works exactly as expected — types are checked at call time

The compilation pipeline is opt-in: `compile-aot` takes a `deftm`
var and produces optimized bytecode. Without compilation, `deftm` functions
still work correctly through runtime dispatch — just slower.

This means you can develop at the REPL with full interactivity, then
compile hot paths for production. The same code works in both modes.
