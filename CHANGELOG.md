# Changelog

All notable changes to Raster will be documented in this file.

## 0.1.x (2026-04-10) -- Initial Release

Raster's first public release. Functional scientific computing for Clojure with
typed multiple dispatch, automatic differentiation, and GPU compilation on the JVM.

### Core

- **Typed multiple dispatch** (`deftm`, `ftm`) with Julia-style method specialization
- **Walker devirtualization** resolves dispatch at definition time
- **JDK ClassFile API bytecode emission** for zero-overhead compiled functions
- **Lazy JIT compilation** on first call with low overhead
- **`compile-aot`** for AOT-style whole-program inlining
- **Polymorphic arithmetic** (`raster.numeric`) for all JVM numeric types
- **Math library** (`raster.math`) with trig, hyperbolic, exp/log, rounding, FMA
- **Polymorphic arrays** (`raster.arrays`) with type-dispatched aget/aset/alength
- **Type promotion lattice** with automatic widening
- **Abstract type hierarchy** (Real, AbstractFloat, AbstractInteger, Signed)
- **Algebraic type lattice** (Semigroup, Monoid, Group, Ring, Field)
- **Value types** (`defvalue`) with Valhalla JDK 27 support
- **Parametric types** (`(All [T])`) for generic value types (Dual, SparseVector, etc.)
- **Float16** value type (Valhalla JDK 27)
- **Complex numbers** (`raster.types.complex`)
- **Trait system** (`raster.traits`) for Julia-style trait dispatch

### Compiler Pipeline

- **Nanopass architecture** with well-defined IR dialects and pass contracts
- **Inlining** with safe-to-inline checks and parametric specialization
- **Dead code elimination** with beichte purity analysis
- **Partial evaluation** and constant folding
- **Buffer fusion** via escape analysis (zero heap allocations in hot path)
- **SOAC fusion** (Futhark-style map/reduce/scan chain fusion)
- **SIMD vectorization** via JDK Vector API
- **`explain-pipeline`** for transparent compiler diagnostics

### Automatic Differentiation

- **Forward-mode AD** (`raster.ad.forward`) with parametric Dual numbers
- **Reverse-mode AD** (`raster.ad.reverse`) via IR-level source transformation
- **Composable AD** -- `value+grad`, `grad` as first-class compiler-visible functions
- **rrule registry** for custom reverse-mode rules
- **AD templates** for flat codegen without closure overhead

### GPU Computing

- **Parallel primitives** (`raster.par`) -- map, reduce, scan, stencil, broadcast
- **OpenCL ICD backend** (NVIDIA, AMD, Intel) via Panama FFM
- **Intel Level Zero backend** with unified shared memory
- **Vulkan compute backend** (GLSL 450 -> SPIR-V)
- **SoA memory layout** for GPU-friendly access patterns
- **Unified GPU session API** (`raster.gpu`)

### Scientific Computing

- **ODE solvers** -- Euler, RK4, Dormand-Prince 5, Tsitouras 5, Implicit Euler, Rosenbrock23
- **PDE solvers** -- method-of-lines with finite-difference stencils
- **SDE solvers** -- Euler-Maruyama
- **Sensitivity analysis** via forward-mode AD through ODE integration
- **Optimization** -- L-BFGS, Nelder-Mead, Newton, gradient descent, Adam
- **Root finding** -- bisection, Brent, Newton, fixed-point iteration
- **Numerical integration** -- Gauss-Kronrod, Simpson
- **Interpolation** -- linear, cubic spline, Akima, PCHIP, 2D bilinear/bicubic
- **Signal processing** -- FFT, PSD, convolution, windows (Hann, Hamming, Blackman, Kaiser)
- **Special functions** -- gamma, beta, erf, Bessel
- **Distributions** -- Normal, Uniform, Exponential, Gamma, Poisson, Beta, Binomial
- **Statistics** -- t-test, chi-squared, F-test, KS test, correlation, descriptive stats

### Linear Algebra

- **Dense fixed-size** -- Vec2/3/4, Mat2x2/3x3/4x4 with value semantics
- **Dense general** -- LU, Cholesky, solve
- **LAPACK** via Panama FFI (OpenBLAS cblas_dgemv/sgemv)
- **Decompositions** -- SVD, QR, eigendecomposition
- **Iterative solvers** -- CG, GMRES, BiCGSTAB, Lanczos
- **Sparse vectors/matrices** -- SparseVector, CSR, COO (parametric)
- **PCA**

### Deep Learning

- **Layers** -- linear, conv1d, conv2d, maxpool2d, layer-norm, group-norm, batch-norm
- **Activations** -- relu, silu, gelu, sigmoid, tanh, leaky-relu
- **Loss functions** -- MSE, cross-entropy, Huber, L1
- **Optimizers** -- SGD, Adam, AdamW with gradient clipping and LR schedulers
- **Einsum** and einops-style rearrangement
- **Attention** -- scaled dot-product, multi-head, graph attention

### Symbolic Computation

- **Symbolic expressions** (`raster.sym.core`) with Sym value type
- **Symbolic differentiation** and Taylor series expansion
- **Linearity/sparsity analysis**
- **Function algebra** (D operator, composition)

### Geometric Algebra

- **Cl(p,q,r) multivectors** with geometric, outer, inner products
- **Hodge dual**, reversal, grade projection
- **Valhalla value class generation** for compiled GA types

### Other

- **Agent-based simulation** (`raster.abm`) with GPU compilation
- **Vulkan rendering engine** (`raster.vk`) with mesh, texture, audio support
- **Game examples** -- Valley (voxel survival), Asteroids, Doom-style renderer

### Known Limitations

- **JDK 24+ required** for ClassFile API and Panama FFM
- **Valhalla JDK 27 recommended** for value types, Float16, and best performance
- GraalVM JDK gives 3-6x slower results than Valhalla HotSpot C2
- Float16 math functions (sin, cos, exp, etc.) not yet specialized -- computed via Float32
- GPU compound kernels have a known barrier synchronization issue
- `deftm` does not support `& {:keys [...]}` keyword args directly
- Functions with >4 args cannot have primitive type hints (Clojure limitation)
