# Raster

Typed multiple dispatch for Clojure — Julia-style polymorphic arithmetic with devirtualization. (Raster & Juliet / Julia pun)

## Project Structure

```
src/raster/
  core.clj                   # Public API: deftm, ftm, defvalue, specialize macros
  numeric.clj                # Polymorphic +, -, *, /, mod, rem, bitwise, comparisons
  math.clj                   # Julia Base math (trig, exp/log, hyperbolic, fma, etc.)
  number_types.clj           # Abstract type hierarchy (Real, AbstractFloat, etc.)
  algebraic_types.clj        # Algebraic lattice (Magma → Ring → Field)
  promote.clj                # Type promotion lattice
  arrays.clj                 # Polymorphic aget/aset/alength
  complex.clj                # Complex number support
  float16.clj                # Float16 value type via Valhalla
  traits.clj                 # Julia-style trait system
  strings.clj                # Julia-style string operations
  random.clj                 # RNG wrappers
  nn.clj                     # NN primitives (dense, relu, softmax) with rrules
  par.clj                    # Parallel primitives (map!/reduce!/scan!/broadcast!)
  tooling/
    inspect.clj              #   REPL tools (bytecode disassembly, JIT diagnostics)
  typed.clj                  # TypedClojure integration

  compiler/                  # Nanopass compiler pipeline
  #   Dependency layering (no cycles allowed):
  #     core/* → pipeline
  #     ir/* defines compiler IR layers
  #     passes/* contains scalar and parallel lowering/optimization
  #     backend/* contains JVM and GPU code generation
  #   requiring-resolve only for genuine cycles (dispatch↔core, bytecode↔inline),
  #   optional deps (TypedClojure, GPU runtimes), or dynamic/computed symbols

  ad/                        # Automatic differentiation
    forward.clj              #   Forward-mode AD (Dual numbers)
    reverse.clj              #   Reverse-mode AD (closures-as-tape)
    rrule.clj                #   Reverse-mode rule registry
    jet.clj                  #   Jet type (truncated Taylor series)
    activity.clj             #   Activity analysis (active vs constant nodes)
    purity.clj               #   Purity analysis (beichte integration)

  linalg/                    # Linear algebra
    core.clj                 #   Dense & fixed-size (Vec2-4, Mat2-4, solve, LU, Cholesky)
    lapack.clj               #   Panama FFI bindings to OpenBLAS
    svd.clj, qr.clj, eigen.clj  # Matrix decompositions
    iterative.clj            #   Krylov methods (CG, GMRES, BiCGSTAB, Lanczos)
    pca.clj                  #   Principal component analysis
    sparse.clj               #   Sparse vectors (sorted compressed format)

  sci/                       # Scientific computing
    special.clj              #   Special functions (gamma, beta, erf, Bessel)
    distributions.clj        #   Probability distributions (Normal, Beta, Gamma, ...)
    stats.clj                #   Statistical tests (t-test, chi-squared, KS)
    optim.clj                #   Optimization (L-BFGS, Nelder-Mead, Newton)
    roots.clj                #   Root finding (bisection, Brent, Newton)
    quadrature.clj           #   Numerical integration (Gauss-Kronrod, Simpson)
    interpolation.clj        #   Splines (linear, cubic, Akima)
    signal.clj               #   Signal processing (windows, PSD, convolution)
    fft.clj                  #   FFT (radix-2 Cooley-Tukey)

  ode/                       # Differential equations
    core.clj                 #   ODE integrators (Euler, RK4, DP5, Tsit5) + GenericODEProblem for sensitivity
    pde.clj                  #   Method-of-lines PDE solver
    sde.clj                  #   Stochastic DEs (Euler-Maruyama)

  sym/                       # Symbolic computation
    core.clj                 #   Symbolic expressions (Sym value type)
    diff.clj                 #   Symbolic differentiation
    analysis.clj             #   Linearity/sparsity analysis
    taylor.clj               #   Taylor series expansion
    fn_algebra.clj           #   Function algebra (D operator, composition)

  ga/                        # Geometric algebra
    core.clj                 #   Cl(p,q,r) multivectors, products, Hodge dual
    compile.clj              #   Valhalla value class generation for GA

  gpu/                       # GPU runtime
    core.clj                 #   Unified session layer (backend dispatch)
    ze_runtime.clj           #   Level Zero (Intel GPUs, Panama FFM)
    ocl_runtime.clj          #   OpenCL ICD (portable, Panama FFM)

  dl/                        # Deep learning
  abm/                     # Agent-based simulation (GPU-compiled)
  vk/                        # Vulkan rendering engine

test/raster/                 # Tests for all modules
bench/                       # Performance comparison benchmarks
design/                      # Architecture documents
```

## Running Tests

### Standard JDK (no Valhalla)
```bash
clojure -M:test
```
Sensitivity tests requiring `Dual4` are guarded by `dual4-available?` and will be skipped.

### Valhalla JDK (full test suite including Dual4)
```bash
JAVA_HOME=~/Development/valhalla-jdk/build/linux-x86_64-server-release/images/jdk \
PATH="$JAVA_HOME/bin:$PATH" \
clojure -J--enable-preview \
        -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
        -J--enable-native-access=ALL-UNNAMED \
        -M:test:valhalla
```

## Valhalla Setup

### JDK Location
The Valhalla JDK (lworld branch, JDK 27-internal) is built at:
```
~/Development/valhalla-jdk/build/linux-x86_64-server-release/images/jdk/
```

### Rebuilding the JDK (if needed)
```bash
cd ~/Development/valhalla-jdk
bash configure --enable-headless-only
make images JOBS=8
```

### deps.edn Alias
The `:valhalla` alias adds `classes/` to the classpath and enables `--enable-preview`:
```clojure
:valhalla {:extra-paths ["classes"]
           :jvm-opts ["--enable-preview"
                      "--add-exports" "java.base/jdk.internal.vm.annotation=ALL-UNNAMED"]}
```

## In-Game Screenshots (Doom / Raster Vulkan)

Use `raster.vk.screenshot` to capture the game window from the REPL:
```clojure
(require 'raster.vk.screenshot)
(raster.vk.screenshot/request-screenshot! "/tmp/screenshot.png")
```
The screenshot is captured on the next frame render. Check `/tmp/doom-screenshot-status.txt` for success/error. The frame loop calls `capture-if-requested!` automatically.

## REPL

### Standard JDK
```bash
clojure -M:nREPL
```

### Valhalla JDK
The cider-nrepl middleware is **incompatible** with JDK 27. Start nREPL without it:
```bash
JAVA_HOME=~/Development/valhalla-jdk/build/linux-x86_64-server-release/images/jdk \
PATH="$JAVA_HOME/bin:$PATH" \
clojure -Sdeps '{:deps {nrepl/nrepl {:mvn/version "1.3.0"}}}' \
  -M:valhalla -m nrepl.cmdline --port <PORT>
```
**Important:** Both `JAVA_HOME` and `PATH` must be set — `JAVA_HOME` alone is not sufficient.

### REPL Reload Caveats
- `MethodEntry` defrecord is in `raster.compiler.core.method-entry` (separate leaf namespace) to survive reloads. Reloading `core.clj` and `dispatch.clj` with `:reload` is safe.
- Avoid `:reload-all` on `raster.core` — it transitively reloads `method-entry.clj` which redefines the record class and causes `ClassCastException`.
- Use `:reload` (without `-all`) when requiring namespaces to pick up source changes.

## Development Policy

Raster is **pre-release software**. Do not add backward compatibility shims, deprecation warnings, or legacy code paths. When changing APIs, update all call sites directly. No v1/v2 suffixes or feature flags for old behavior.

## Key Conventions

### Always use `deftm` and `ftm`, never `defn` or `fn`

**This is the most important design rule in Raster.** All functions should be written
with `deftm` (named) or `ftm` (anonymous), never plain `defn`/`fn`. This applies to:

- All library functions (math, numeric, optimization, root finding, etc.)
- All helper/internal functions
- All function arguments (use `(Fn [ParamTypes...] RetType)` annotation)
- Anonymous functions passed as arguments (use `ftm` instead of `#(...)` or `fn`)

**Exceptions where plain `defn`/`fn` is acceptable:**
- Thin top-level API wrappers that parse keyword args before delegating to `deftm`
- Internal support code: data loading, configuration, test helpers, IO utilities
- Callbacks for registries/frameworks that expect IFn (e.g., rrule pullback closures)
- The key distinction: if it does **numerical computation**, use `deftm`/`ftm`

**Why:** `deftm` enables typed multiple dispatch, walker devirtualization, and bytecode
compilation. When function params are annotated with `(Fn [Double] Double)`, the walker
emits `.invk` calls on the typed interface, giving primitive-speed evaluation without
boxing. Plain `defn`/`fn` bypasses all of this and boxes through `IFn.invoke`.

**Example — function parameter with typed annotation:**
```clojure
(deftm bisect [f :- (Fn [Double] Double),
               a :- Double, b :- Double,
               tol :- Double, maxiter :- Long]
  ;; f is called via .invk → primitive double in, primitive double out
  (let [fa (f a)] ...))
```

**Example — passing typed anonymous functions:**
```clojure
(bisect (ftm [x :- Double] :- Double (- (* x x) 2.0)) 0.0 2.0)
```

**Supported `(Fn ...)` patterns:**
- `(Fn [Double] Double)` — scalar → scalar
- `(Fn [(Array double)] Double)` — double-array → scalar (objectives)
- `(Fn [(Array double) (Array double)])` — mutating gradient (void return)
- `(Fn [(Array double) (Array double) Double])` — ODE right-hand side

### No type hints needed

**Do not use Clojure type hints (`^double`, `^objects`, `^ints`, etc.) in Raster code.**
Raster's `deftm`/`ftm` system handles type dispatch and primitive specialization
automatically via `:- Type` annotations. Type hints are unnecessary and add clutter.
Use `(double ...)` casts inside function bodies when needed for unboxing, but never
use `^`-style metadata type hints on parameters, let bindings, or return values.

### Other conventions

- **Keyword args → dict parameter**: `deftm` doesn't support `& {:keys [...]}`. Pass optional
  args as a map parameter and use `(get opts :key default)` inside the body.
  Thin `defn` wrappers that parse keyword args before delegating to `deftm` are acceptable.
- `deftm` is a macro; use `eval` for dynamic/conditional registration
- Use short class names (e.g., `Dual4` not `raster.Dual4`) in `:- ` type annotations — dots break mangled defn names
- Functions with >4 args cannot have primitive type hints (Clojure limitation) — use `(double ...)` casts inside instead
- ODE sensitivity works through the standard `GenericODEProblem` API in `ode/core.clj` (Dual4 loaded conditionally via `Class/forName`)
- Time variables in ODE solvers always use `clojure.core` arithmetic (plain doubles), only state/params go through `raster.numeric`

### Diagnostics and Transparency

Compiler optimizations must be observable. When adding or changing compiler passes,
always update `show-pipeline` and `explain-pipeline` to reflect the actual pass
sequence. Use `explain-pipeline` during development to verify transformations:

```clojure
(pipeline/explain-pipeline #'par/dot-product)             ;; forward (default)
(pipeline/explain-pipeline #'par/dot-product :mode :training)
```

### Compiler Design Principles

**Preserve semantic identity through the pipeline.** When the walker devirtualizes a
call like `(raster.numeric/+ x y)` into `(.invk impl x y)`, it must attach the original
op as metadata (e.g., `^{:op 'raster.numeric/+}`). No pass should recover semantic meaning
by pattern-matching mangled implementation names. If a pass needs to know "this is addition",
that information must be carried explicitly, not reverse-engineered.

**Passes must be domain-agnostic.** The compiler should not contain domain-specific knowledge
about particular algorithms (SGD, attention, convolution, etc.). It operates on general
constructs: typed calls, parallel primitives, mutation effects, buffer allocation. Domain
knowledge lives in `deftm` definitions, rrule templates, and op descriptors — never in
pass logic. Optimizers, loss functions, and layer operations should all be expressible as
ordinary `deftm` functions composed with par combinators.

**Every pass has a contract.** Each compiler pass must:
- Accept a well-defined input dialect and produce a well-defined output dialect
- Be validated at boundaries (the dialect validator catches violations)
- Be independently testable with unit tests on small forms
- Have its behavior visible via `explain-pipeline`

**Emit qualified symbols.** Code-generating passes (AD templates, rrule grads-fns,
buffer fusion rewrites) must emit fully qualified operator symbols (`raster.numeric/*`,
not bare `*`). The walker resolves qualified symbols; bare symbols bypass dispatch and
cause validator rejections downstream.

**Unify metadata registries.** Operation metadata (buffer semantics, AD templates,
device rules, shape inference, effect classification) should converge on a single
op-descriptor registry rather than separate per-concern registries.

**Use `raster.numeric` operators, not `clojure.core`.** All arithmetic in code that flows
through the compiler (`deftm`/`ftm` bodies, AD templates, rrule grads-fns) must use
`raster.numeric/+`, `raster.numeric/*`, etc. — never bare `+` or `clojure.core/+`. Bare
Clojure arithmetic bypasses typed dispatch and is invisible to the walker. Similarly, use
`raster.arrays/aget`, `raster.arrays/aset!`, `raster.arrays/alength` for array operations.
The only exception is internal compiler emit code (e.g., `clojure.core/aset` in expanded
loop bodies) which intentionally bypasses dispatch for direct JVM primitives.

**Centralize operator classification.** Compiler passes must not maintain independent
hardcoded sets of operator symbols. Operator properties (simplifiable, SIMD-capable,
effectful, type-preserving, allocating) belong in the op-descriptor registry. Passes
query the registry rather than checking `contains?` against local `#{...}` literals.
When adding a new operator, a single registry entry should be sufficient — not updates
to 10+ files.

**Use the unified form classifier (`compiler/ir/form.clj`).** Compiler passes must use
`form/form-info`, `form/binding-form?`, `form/introduces-scope?`, `form/liftable?` etc.
instead of ad-hoc `(contains? #{'let 'let* ...} (first form))` checks. The form classifier
is the single source of truth for form structure and scoping semantics. When adding a new
form type (e.g., a new parallel primitive), update `form-info` in ONE place — all passes
will handle it correctly. Never scatter `contains?` checks with form-type sets across files.

**AD gradient rules are explicit templates, not auto-registered closures.** The compiler
inliner checks for explicit AD templates (`:grads`/`:grads-fn`) to decide whether a deftm
call should stay symbolic (for the AD transform) or be inlined. Runtime `value+grad` creates
transient closures for interpreted AD without registering them globally. Only ops with
explicit templates in the template registry are AD-aware; user deftm functions are always
inlinable by the compiler.

**Effects must be explicit.** Mutation, IO, and allocation must be visible in the IR
through binding metadata or canonical effect nodes — not inferred by scanning for
`aset` calls inside nested forms. DCE and other passes rely on correct effect
classification; implicit effects cause silent miscompilation.

## Performance Baselines (2026-03-24)

### ODE: DP5 Lorenz attractor (t=[0,100], atol=1e-8, rtol=1e-6)

| Implementation | µs | vs Julia |
|---|---|---|
| Raster `compile-aot` | **400** | **0.69x (faster)** |
| Java DP5 (pure, same JVM) | 423 | 0.73x |
| Julia OrdinaryDiffEq.jl | 583 | 1.0x |
| Raster lazy JIT (`deftm`) | 741 | 1.27x |
| Clojure type-hinted | 707 | 1.21x |

`compile-aot` inlines the entire solve loop + RHS + step function into
a single JVM method. Lazy JIT compiles each function individually on first call.

### Sensitivity: Lotka-Volterra parameter estimation (Dual4 forward-mode AD)

RK4 fixed-step, dt=0.02, 20 save points, lr=0.001

| Tier | Implementation | Single gradient | 200-iter GD | vs Julia |
|------|---------------|----------------|-------------|----------|
| 1 | Generic dispatch (`Object[]` + `raster.numeric`) | 4.2ms | 815ms | 113x slower |
| 2 | Specialized/devirtualized (bytecode pipeline) | 15.0µs | 5.8ms | **0.81x** |
| 3 | Java inner loop (`Dual4Solver.java`) | 10.0µs | 1.9ms | **0.27x** |
| — | Julia ForwardDiff.jl | 15.9µs | 7.2ms | 1.0x |

Tier 2 (pure Clojure, devirtualized) is 19% faster than Julia. Tier 3 (Java) is 3.7x faster.

### DL Training (2026-03-26)

Compiled pipeline with AD (forward + backward + SGD), OPENBLAS_NUM_THREADS=1.
Valhalla JDK 27, 3000+ warmup steps, same machine for all measurements.

| Benchmark | Raster compiled f64 | JAX f64 | JAX f32 |
|---|---|---|---|
| MLP 784-128-10 | 143 µs | 86 µs | 50 µs |
| LeNet-5 | **221 µs** | 370 µs | 356 µs |

MLP: JAX 1.7x faster. LeNet: **Raster 1.7x faster.**
Uncompiled deftm: MLP ~2000 µs, LeNet ~3700 µs (14-17x slower).

GraalVM JDK: 3-6x slower than Valhalla on the same compiled code.

Zero heap allocations in hot path — all intermediate buffers hoisted to instance fields.
SIMD vectorization via JDK Vector API for SGD/relu/softmax loops.
Matmul uses OpenBLAS via Panama FFI (`cblas_dgemv`).

### Compilation Latency

| Operation | Time |
|---|---|
| Lazy JIT (simple `deftm`, first call) | ~1.5ms |
| `compile-aot` (simple function) | ~3ms |
| `compile-aot` (MLP train-step + AD) | ~330ms |
| Cold `require` raster.core | ~2.5s |
| Cold `require` raster.numeric | ~5.4s |
| Cold `require` full pipeline | ~11s total |

## Benchmarking

**Always benchmark on Valhalla JDK 27** (HotSpot C2). GraalVM CE has no
auto-vectorization — benchmarks on GraalVM are not representative of
HotSpot performance. Use the wrapper script:

```bash
source valhalla-env.sh  # sets JAVA_HOME + PATH
OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \
  -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
  -J--enable-native-access=ALL-UNNAMED \
  -J--add-modules=jdk.incubator.vector ...
```

Or use `/tmp/start_valhalla_repl.sh` for REPL benchmarks.
