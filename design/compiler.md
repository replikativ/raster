# Raster Compiler

Raster includes a nanopass compiler that transforms high-level typed function
definitions (`deftm`) into optimized JVM bytecode with zero-allocation inner
loops, BLAS dispatch, and optional SIMD vectorization. This document explains
how a function goes from source to machine code.

## Overview

The pipeline takes a `deftm` body through a series of source-to-source passes,
each accepting and producing a well-defined S-expression dialect. The final
stage emits JVM bytecode via the ClassFile API.

```
Source (deftm)
  |
  v
Walker .............. devirtualize polymorphic calls to concrete .invk
  |
  v
Lower ............... flatten to single-body let*, expand AD if in training mode
  |
  v
Fixpoint ............ iteratively inline + rewalk until stable (handles composable AD)
  |
  v
DCE ................. dead code elimination (liveness + effect analysis)
  |
  v
Buffer Fusion ....... rewrite allocating ops to in-place variants, mark hoistable buffers
  |
  v
SOAC Fusion ......... fuse consecutive par/map!, par/reduce (vertical + horizontal)
  |
  v
SegOp Lower ......... lower par forms to SegOp IR records
  |
  v
Backend ............. expand par/map! to loop* (scalar) or Vector API (SIMD)
  |
  v
Resolve Alength ..... decouple iteration counts from buffer identity
  |
  v
Memory Merge ........ interference-graph coloring to share buffers
  |
  v
Hoist + Compile ..... extract hoistable allocations, emit bytecode
```

Each pass has a contract: declared input/output dialects, validated at
boundaries, independently testable, and visible via `explain-pipeline`.

## The MLP Training Example

All examples below trace a 784-128-10 MLP training step through the pipeline.
This is the same function used in the regression test suite
(`pipeline_quality_test.clj`) and the JAX comparison benchmarks.

```clojure
(deftm train-step
  [W1 :- (Array double), b1 :- (Array double),
   W2 :- (Array double), b2 :- (Array double),
   x :- (Array double), y :- (Array double),
   lr :- Double] :- Double
  (let [vg ((rev/value+grad (var nn/loss-fn)) W1 b1 W2 b2 x y)
        loss (nth vg 0)]
    (optim/sgd-step! W1 (nth vg 1) (arrays/alength W1) lr)
    (optim/sgd-step! b1 (nth vg 2) (arrays/alength b1) lr)
    (optim/sgd-step! W2 (nth vg 3) (arrays/alength W2) lr)
    (optim/sgd-step! b2 (nth vg 4) (arrays/alength b2) lr)
    loss))
```

This 12-line function compiles to ~80 bindings with forward pass, full backward
pass (reverse-mode AD), softmax gradient, two BLAS matrix-vector products, and
four SGD update loops. After compilation it runs at **62 microseconds per step**
on JDK 27 with OpenBLAS, faster than an equivalent JAX/XLA program (80us).

## Stage 0: Walker (Devirtualization)

The walker is a source-to-source transformer that resolves polymorphic
`deftm` calls into concrete, type-specialized implementations. Given the type
context from parameter annotations (`:- (Array double)`, `:- Double`), it:

1. **Infers binding types** from parameter annotations and return types
2. **Resolves dispatch** — `(nn/dense W x b)` becomes a call to the mangled
   concrete implementation `raster.nn/dense_m_doubles_doubles_doubles`
3. **Emits `.invk` calls** for typed function parameters annotated with
   `(Fn [...] ...)`, calling their typed interface directly (no IFn boxing)
4. **Preserves semantic identity** by attaching `:op` metadata to every call

After the walker, the training step looks like:

```clojure
(let [vg ((raster.ad.reverse/value+grad #'raster.nn/loss-fn) W1 b1 W2 b2 x y)
      loss (clojure.core/nth vg (long 0))]
  (.invk raster.dl.optim/sgd-step!_m_doubles_doubles_long_double-impl
         W1 (clojure.core/nth vg (long 1))
         (.invk raster.arrays/alength_m_doubles-impl W1) lr)
  ;; ... same for b1, W2, b2 ...
  loss)
```

Every polymorphic call is now a monomorphic `.invk` on a typed interface. The
`value+grad` call remains symbolic — it will be expanded in the next stage.

### Why `.invk` instead of static calls?

`deftm` functions implement typed interfaces (e.g., `IFn_m_doubles_doubles_doubles`)
with methods that accept and return unboxed primitives. The `.invk` call compiles
to a virtual method call on this interface, which the JVM inlines after warmup.
This is how Raster achieves zero-boxing dispatch for function parameters — a
function argument annotated `(Fn [Double] Double)` calls `.invk` with primitive
`double` in and `double` out, never touching `Object`.

## Stage 1: Lower

Lowering flattens the source into a single `let*` form with one binding per
operation (A-normal form). Multi-expression let bodies become effect bindings:

```clojure
;; Before:
(let [a 1] (side-effect!) result)

;; After:
(let* [a 1
       _eff (side-effect!)]
  result)
```

When running in training mode, lowering also **expands the AD operator**.
`(value+grad #'loss-fn)` is not a runtime closure — the compiler inlines the
entire forward pass and reverse-mode backward pass as flat bindings:

```clojure
(let* [;; Forward pass
       h    (raster.nn/dense_m_doubles_doubles_doubles W1 x b1)
       a    (raster.nn/relu_m_doubles h)
       out  (raster.nn/dense_m_doubles_doubles_doubles W2 a b2)
       p    (raster.nn/softmax_m_doubles out)
       loss (.invk raster.nn/cross-entropy_m_doubles_doubles-impl p y)

       ;; Backward pass (reverse-mode AD, closures-as-tape)
       dy__rad  1.0
       dp       (par/map! ... (* -1 (/ y p)) ...)  ;; cross-entropy grad
       s-dot-dy (par/reduce ... (+ acc (* p dp)) ...)  ;; softmax helper
       dx       (par/map! ... (* p (- dp s-dot-dy)) ...)  ;; softmax grad
       ;; ... dense backward (dger! for dW, dgemv-t! for dx) ...
       ;; ... relu backward (masked multiply) ...
       ;; ... second dense backward ...

       ;; SGD updates
       _eff (sgd-step! W1 dW1 n1 lr)
       _eff (sgd-step! b1 db1 n2 lr)
       _eff (sgd-step! W2 dW2 n3 lr)
       _eff (sgd-step! b2 db2 n4 lr)]
  loss)
```

The AD expansion uses **explicit gradient templates** from the rrule registry.
Operations with registered `:grads` or `:grads-fn` metadata (dense, relu,
softmax, cross-entropy) stay as opaque calls during AD — their gradient rules
are inlined as template code. User functions without explicit rules are
auto-derived via `value+grad` at runtime.

## Stage 2: Fixpoint (Inline + Normalize + Rewalk)

After lowering, the expanded AD code may contain calls that haven't been
devirtualized yet (e.g., backward-pass arithmetic introduced by gradient
templates). The fixpoint pass iterates:

1. **Expand** — inline all remaining `deftm` calls
2. **Normalize** — canonicalize to ANF
3. **Rewalk** — if expansion introduced new undevirtualized calls, run the
   walker again to resolve them
4. Repeat until stable (typically 1-2 iterations)

This handles **composable AD** — `(grad (grad f))` produces nested backward
passes that each need their own devirtualization round.

After fixpoint, `relu` has been inlined to `(par/map! out i n double (max 0.0 (aget h i)))`,
softmax is expanded to its three-phase implementation (max-reduction,
exp-sum-reduction, normalization map), and all arithmetic is devirtualized
`.invk` calls. The training step is now ~79 flat bindings.

## Stage 3: Dead Code Elimination

DCE removes bindings that are neither referenced nor effectful. It walks
bindings backward from the return value, tracking liveness:

- A binding is **live** if the return value or a live binding references it
- A binding is **effectful** if it mutates a live buffer (classified via the
  op-descriptor registry and beichte purity analysis)
- Everything else is dead

For the MLP training step, DCE removes 7 dead bindings — typically intermediate
alias bindings and unused reduction results from the AD expansion.

## Stage 4: Buffer Fusion

Buffer fusion rewrites allocating operations to use pre-allocated buffers,
enabling later hoisting to eliminate all per-call allocations.

**How it works:**

1. For each call like `(dense W x b)` that returns a new array, the op-descriptor
   registry provides an `in-place-arg` index — dense has a 4-argument variant
   `(dense-into! W x b out)` that writes into `out` instead of allocating.

2. The pass checks if any existing buffer can be reused (dead after this point,
   correct size, no aliases). If so, it rewrites to the in-place variant using
   that buffer.

3. If no reusable buffer exists, a new allocation is inserted and marked
   `^{:hoistable true, :write-mode :overwrite}`:

```clojure
;; Before:
[h (raster.nn/dense_m_doubles_doubles_doubles W1 x b1)]

;; After:
[buf_h (raster.arrays/zeros-like_m_doubles_long b1 (alength b1))  ;; ^:hoistable
 h     (raster.nn/dense-into! W1 x b1 buf_h)]
```

The `:write-mode` annotation (`:overwrite` or `:accumulate`) tells the hoisting
stage whether the buffer needs zeroing before each call. Accumulate-mode buffers
(e.g., gradient accumulators used with `dger!`) are zeroed; overwrite-mode
buffers (e.g., dense output, relu output) are not.

For the MLP training step, buffer fusion produces **13 hoistable buffer
allocations** — dense outputs, relu output, softmax intermediates, gradient
buffers for both layers.

## Stages 5-8: Parallel Optimization

### SOAC Fusion

SOAC (Second-Order Array Combinator) fusion merges consecutive parallel
operations that traverse the same data:

- **Vertical:** `map! -> map!` (inline first body into second's loop)
- **Vertical:** `map! -> reduce` (compute map inline during reduction)
- **Horizontal:** independent same-size `map!` ops (shared index loop)

This reduces the number of array traversals and improves cache locality.

### SegOp Lower

Converts fused parallel forms into SegOp IR records (Structured Array
Operations) — an intermediate representation that backends consume.

### Backend Selection

The backend pass decides how to lower each parallel operation:

- **SIMD (CPU):** Java Vector API (`DoubleVector`) with 4-lane `ymm` operations
  and scalar tail handling. Selected when the operation is vectorizable and the
  array is large enough (configurable threshold).
- **GPU:** SPIR-V or CUDA kernel generation via SegOp-to-kernel compilation.
- **Scalar:** `loop*` with `(int 0)` counter and `unchecked-inc-int` — the
  default when SIMD is disabled.

For the scalar backend, `par/map!` expands to:

```clojure
(let* [n (int (alength out))]
  (loop* [i (int 0)]
    (if (< i n)
      (do (aset out i (double (max 0.0 (aget h i))))
          (recur (unchecked-inc-int i)))
      out)))
```

This uses explicit `(int 0)` initialization and `unchecked-inc-int` instead of
Clojure's `dotimes`, which promotes the counter to `long` and prevents the JVM's
C2 compiler from recognizing the loop as a counted loop (required for
auto-vectorization).

## Stage 10: Resolve Alength

Before memory merge can safely share buffers, iteration counts must be
decoupled from buffer identity. This pass replaces `(alength buf)` with the
size expression used when `buf` was allocated:

```clojure
;; Before:
[buf (double-array (alength h))]
[... (loop* [i (int 0)] (if (< i (int (alength buf))) ...))]

;; After:
[buf (double-array (alength b1))]     ;; h was allocated from b1's length
[... (loop* [i (int 0)] (if (< i (int (alength b1))) ...))]
```

Now `buf` can be resized or merged without breaking loop bounds.

## Stage 11: Memory Merge

Memory merge reduces total buffer count via **interference graph coloring**
(inspired by Futhark):

1. **Liveness analysis** — compute `{def-idx, last-use-idx}` for each buffer
2. **Alias tracking** — in-place operations extend the source buffer's lifetime
3. **Interference graph** — two allocations conflict if their live ranges overlap
4. **Greedy coloring** — assign non-interfering allocations to shared buffers,
   taking the max size

For the training step, memory merge reduces 13 buffer allocations to 8 by
sharing non-overlapping gradient buffers. The reduction is conservative — only
provably non-interfering buffers with compatible types are merged.

## Hoist and Compile

The terminal stage splits the form into two parts:

1. **Allocation function** — evaluated once via Clojure `eval`, returns an array
   of pre-allocated buffers. Buffer dimensions that depend only on function
   parameters are resolved at first call and cached.

2. **Compute function** — compiled to JVM bytecode via the ClassFile API, with
   hoisted buffers passed as extra parameters.

### Method Extraction

Non-trivial bindings (loops, nested lets, large expressions — anything above
a heuristic size threshold) are extracted into separate static methods
(helpers). Simple bindings (aliases, literals, small calls) stay in the
orchestrator method. Helper methods are grouped to stay under 25KB of bytecode
each, well within C2's optimization sweet spot (~8KB). The MLP training step
produces **17 helper methods** — one for each dense/relu/softmax phase, each
gradient computation, and each SGD update loop.

### Bytecode Compilation

The bytecode compiler handles Clojure's ~12 special forms plus primitive
operations. Key optimizations:

**Branch folding.** Instead of materializing a boolean and branching on it:
```
;; Naive (7 instructions):
ILOAD i
ILOAD n
IF_ICMPGE false_branch
ICONST_1        ;; push true
GOTO end
false_branch:
ICONST_0        ;; push false
end:
IFEQ else       ;; branch on boolean
```

The compiler emits a direct comparison branch:
```
;; Optimized (1 instruction):
ILOAD i
ILOAD n
IF_ICMPGE else
```

**Integer comparisons.** Loop bounds use `IF_ICMPGE` (int vs int) instead of
`DCMPG` (double comparison). This is critical for C2 counted loop recognition —
C2's SuperWord auto-vectorizer only kicks in for `int`-counted loops with
`IF_ICMPxx` comparisons.

**Checkcast elimination.** When a method parameter has a typed JVM descriptor
(e.g., `[D` for `double[]`), the compiler marks it as `:verified` in the locals
map and skips `CHECKCAST` instructions for array operations on that parameter.
Unverified parameters (received as `Object`) still get the cast.

**Intrinsic recognition.** `unchecked-inc-int` is compiled to `ILOAD; ICONST_1; IADD`
instead of a var lookup + `IFn.invoke` call.

### SGD Helper Bytecode

The SGD update loop `W[i] = W[i] - lr * grad[i]` compiles to:

```
helper_14 ([DD[D)Object
  ;; n = (int (alength W))
  ALOAD_0                    ;; W
  INVOKESTATIC alength       ;; -> long
  L2I                        ;; -> int
  ISTORE 4                   ;; n

  ;; i = 0
  LDC 0
  ISTORE 5

loop:
  ILOAD 5                   ;; i
  ILOAD 4                   ;; n
  IF_ICMPGE end             ;; if i >= n, exit

  ;; W[i] = W[i] - lr * grad[i]
  ALOAD_0                   ;; W (for DASTORE)
  ILOAD 5                   ;; i (for DASTORE)
  ALOAD_0                   ;; W
  ILOAD 5                   ;; i
  DALOAD                    ;; W[i]
  DLOAD_1                   ;; lr
  ALOAD_3                   ;; grad
  ILOAD 5                   ;; i
  DALOAD                    ;; grad[i]
  INVOKESTATIC _star_       ;; lr * grad[i]
  INVOKESTATIC _minus_      ;; W[i] - (lr * grad[i])
  DASTORE                   ;; W[i] = result

  ;; i++
  ILOAD 5
  LDC 1
  IADD
  ISTORE 5
  GOTO loop

end:
  ALOAD_0                   ;; return W
  ARETURN
```

No `CHECKCAST`, no `DCMPG`, no boxing. C2 recognizes this as a counted loop and
auto-vectorizes it to `vmulpd`/`vsubpd` on `ymm` registers (AVX2, 4 doubles per
operation, 7x unrolled).

## C2 Auto-Vectorization

After JIT compilation, C2's SuperWord pass converts the scalar SGD loop into
SIMD instructions. Requirements for SuperWord:

1. **Int-counted loop** with `IF_ICMPxx` — not `long`, not `DCMPG`
2. **Simple loop body** — array loads, arithmetic, array stores
3. **No control flow** in the loop body (no branches, no method calls that
   aren't inlined)
4. **Contiguous memory access** — sequential `DALOAD`/`DASTORE` with unit stride

The `_star_` and `_minus_` static calls are small enough for C2 to inline,
leaving raw `DMUL`/`DSUB` in the loop body. C2 then packs 4 iterations into
`ymm` registers:

```asm
vmovupd ymm0, [rsi + rcx*8]      ;; load 4 doubles from W
vmovupd ymm1, [rdx + rcx*8]      ;; load 4 doubles from grad
vmulpd  ymm1, ymm1, ymm2         ;; ymm1 = grad * lr (lr broadcast)
vsubpd  ymm0, ymm0, ymm1         ;; ymm0 = W - lr*grad
vmovupd [rsi + rcx*8], ymm0      ;; store 4 doubles to W
;; ... 7x unrolled (28 doubles per iteration)
```

## BLAS Integration

Matrix-vector operations dispatch to OpenBLAS via Panama FFI. The dispatch
table maps typed signatures to BLAS routines:

```clojure
;; In raster.dl.blas:
(deftm dense-into!
  [W :- (Array double), x :- (Array double),
   b :- (Array double), out :- (Array double)] :- (Array double)
  ;; cblas_dgemv via Panama MethodHandle
  (let [rows (alength b)
        cols (alength x)]
    (acopy! b 0 out 0 rows)
    (blas/dgemv-t! rows cols 1.0 W cols x 1 1.0 out 1))
  out)
```

The `MemorySegment.ofArray` call for passing `double[]` to native code is
JIT-inlined to zero overhead by C2. Each `dgemv` call takes ~11 microseconds
for the 784x128 layer.

### Dispatch Ordering

`deftm` supports parametric polymorphism via `(All [T] ...)`. When a parametric
definition auto-generates a concrete `double` overload, it can **overwrite** a
manually registered BLAS specialization (last-writer-wins in the dispatch table).

The fix is to define parametric fallbacks **before** concrete BLAS
specializations:

```clojure
;; Generic fallback — MUST come BEFORE BLAS specializations
(deftm dense (All [T] [W :- (Array T) ...] ...)
  (dotimes [i rows] ...))  ;; scalar loop

;; BLAS specialization — overwrites the auto-generated double version
(deftm dense-into!
  [W :- (Array double) ...] ...
  (blas/dgemv-t! ...))  ;; OpenBLAS
```

The regression test `blas-dispatch-priority-test` verifies this ordering is
maintained by checking that the walked body of `dense-into!` contains `dgemv`
and not `dotimes`.

## GPU Backend

The same `par/map!` and `par/reduce` forms that compile to scalar loops or SIMD
on the CPU can also target GPUs. The GPU backend replaces the JVM bytecode
backend with OpenCL C code generation, SPIR-V compilation, and Level Zero
kernel dispatch.

### Pipeline

The GPU and CPU paths share the **entire pipeline** through stage 8
(compound detect). They diverge at **stage 9 (backend)**, where `pass-backend`
dispatches on the target device:

```
Stages 0-8: identical (walker → lower → fixpoint → DCE → buffer-fuse
            → SOAC-fuse → segop-lower → compound-detect)
                              |
              Stage 9: pass-backend dispatches
              /               |               \
         :scalar           :simd           :opencl
    par/expand-par     par-simd/pass     opencl-pass/pass
    (loop* + int i)    (Vector API)      (OpenCL C → SPIR-V,
                                          replace par forms
                                          with kernel markers)

Stages 10-11: identical (resolve-alength → mem-merge → hoist+compile)
```

The OpenCL backend does **not** dispatch kernels at compile time. It does two
things during stage 9:

1. **Generates and compiles kernels** — each `par/map!` or `par/reduce` form
   produces an OpenCL C source string, which is compiled to SPIR-V (via `ocloc`,
   cached) and registered in a kernel table.
2. **Replaces par forms with invocation markers** — the original `par/map!`
   S-expression is replaced with a call like:
   ```clojure
   (raster.gpu.ze-runtime/invoke-registered-kernel
     "par_map_42" [W x] out [lr] n)
   ```

These markers are ordinary S-expressions that flow through stages 10-11
unchanged. The bytecode compiler emits them as regular static method calls.
At runtime, `invoke-registered-kernel` looks up the pre-compiled SPIR-V kernel
and dispatches it to the GPU via Level Zero (Panama FFI).

After `hoist-and-compile`, the GPU path wraps the result with
`ze-runtime/make-gpu-fn`, which manages device buffer allocation and
host-device memory transfers.

### Kernel Generation

A `par/map!` form like `(par/map! out i n double (max 0.0 (aget h i)))` (ReLU)
generates:

```c
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void par_map_relu(
    __global const double* restrict h,
    __global double* restrict out,
    const long n)
{
    long idx = get_global_id(0);
    if (idx < n) {
        out[idx] = fmax(0.0, h[idx]);
    }
}
```

Reductions use a two-phase approach: a work-group-local reduction with
`barrier(CLK_LOCAL_MEM_FENCE)` and Intel subgroup shuffles
(`intel_sub_group_shuffle_down`), followed by a global atomic accumulation.

### Compound Kernels

PDE solvers and ABM simulations often consist of multiple phases that should
execute as a single GPU dispatch to avoid round-trip latency. The
`compound_detect` pass identifies `dotimes` loops containing 2+ par forms and
wraps them in a `compound-kernel` marker. The GPU backend then generates either:

- **`:local` strategy** — a single kernel with `__local` scratch arrays and
  barrier synchronization between phases. Good for small grids where everything
  fits in local memory.
- **`:global` strategy** — separate kernels per phase, with the time loop on
  the host. Better for large grids where local memory is insufficient.

### ABM on GPU: Firms Model

The firms ABM (`raster.abm.firms`) demonstrates the full GPU compilation path.
The model simulates an economy with agents choosing between firms, production
with Cobb-Douglas-style functions, wage distribution, and firm birth/death
dynamics. Each simulation phase is a `deftm` function using `par/map!`:

```clojure
(deftm produce-output-par!
  [firms :- FirmSoA, n-firms :- Long] :- FirmSoA
  (par/map! (:output firms) i n-firms float
    (let [a (aget (:param-a firms) i)
          e (aget (:total-effort firms) i)
          beta (aget (:param-beta firms) i)]
      (+ (* a e) (* (:param-b firms) (Math/pow e beta)))))
  firms)
```

This compiles to an OpenCL kernel operating on Structure-of-Arrays (SoA) GPU
buffers — each field of `FirmSoA` becomes a separate `__global float*`
parameter. The `gpu.clj` orchestrator maps 16 simulation phases to compiled
kernels and manages buffer allocation on the device.

The GPU SoA layer (`gpu_soa_test.clj`) handles the struct-to-flat-arrays
decomposition: struct typedefs, field-by-field pointer parameters, C99
designated initializers for aget/aset, and copy roundtrips between host and
device memory.

## Source Metadata and Stack Traces

Clojure attaches `:line` and `:column` metadata to source forms during reading.
The compiler preserves this metadata through all passes via the `remake` utility
(`compiler/core/util.clj`), which transfers metadata from the original form to
any rewritten replacement. This means that even after inlining, DCE, buffer
fusion, and backend expansion, each binding retains a reference to the original
source line it came from.

At bytecode emission, the compiler maps this metadata to JVM `LineNumberTable`
entries — one per `let*` binding. It also sets the `SourceFileAttribute` on the
generated class to point back to the original `.clj` file. The result is that
stack traces from compiled code show meaningful source locations:

```
at raster.compiled.CF_232354586/compute(nn.clj:47)
at raster.compiled.CF_232354586/helper_14(nn.clj:72)
```

When source metadata is not available (e.g., for AD-generated bindings that
have no original source location), the compiler falls back to the binding's
sequential index as a synthetic line number. This still gives a monotonic
ordering that helps locate the failing operation in `explain-pipeline` output.

## Diagnostics

The compiler is designed for transparency. Every optimization is observable:

```clojure
;; Show all intermediate forms:
(pipeline/show-pipeline #'train-step :simd? false)

;; Pretty-print stage-by-stage summary:
(pipeline/explain-pipeline #'train-step)

;; Inspect compiled bytecode:
(let [diag (inspect/compiled-diagnostics #'train-step)]
  ;; List all helper methods with signatures
  (inspect/list-helpers #'train-step)
  ;; Disassemble a specific helper
  (inspect/disassemble (:statics-bytes diag) {:method "helper_14"})
  ;; Disassemble main compute method
  (inspect/disassemble-compute #'train-step))
```

## Summary: What Makes It Fast

The MLP training step runs at 62us/step because:

| Optimization | Effect |
|---|---|
| Walker devirtualization | Eliminates runtime dispatch, enables primitive types |
| AD template inlining | No tape allocation, flat let* backward pass |
| Buffer fusion + hoisting | Zero per-call allocations (12 buffers hoisted) |
| Memory merge | 13 buffers -> 8 via liveness analysis |
| Int counted loops | C2 SuperWord auto-vectorization (4x doubles/op) |
| Branch folding | Direct IF_ICMPGE, no boolean materialization |
| Checkcast elimination | No redundant CHECKCAST on typed parameters |
| BLAS dispatch | OpenBLAS dgemv for matrix-vector (11us/call) |
| Method extraction | Small methods -> better C2 inlining decisions |

Each optimization is protected by regression tests in
`test/raster/compiler/pipeline_quality_test.clj`.

The same pipeline targets GPUs by swapping the backend: `par/map!` forms become
OpenCL C kernels dispatched via Level Zero, with SPIR-V caching and SoA buffer
management. The firms ABM uses this path to run 16 economic simulation phases
as compiled GPU kernels.
