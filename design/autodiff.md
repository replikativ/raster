# Automatic Differentiation in Raster

Raster provides forward-mode and reverse-mode automatic differentiation (AD)
integrated with the compiler pipeline. AD operators produce `deftm`-compatible
forms that the compiler can inline, optimize, and compile to bytecode — the
same training step that runs at 62us/step on CPU uses AD for the full backward
pass.

## Two Modes

### Forward-Mode: Dual Numbers

Forward-mode propagates derivatives alongside primal values using Dual numbers:

```
Dual(v, dv) where v = f(x), dv = f'(x)·dx
```

Arithmetic on Duals implements the derivative rules:

```
Dual(a, da) + Dual(b, db) = Dual(a+b, da+db)
Dual(a, da) * Dual(b, db) = Dual(a*b, a·db + b·da)
sin(Dual(a, da))           = Dual(sin(a), cos(a)·da)
```

This works because `deftm` dispatch selects the Dual overload automatically.
A function `f: R^n -> R` written with `raster.numeric/+`, `raster.numeric/*`,
etc. will propagate derivatives when called with Dual inputs — no source
transformation needed.

**Multi-variable gradients** use chunked evaluation: each Dual carries `k`
partial derivatives simultaneously, so computing `n` partials requires
`ceil(n/k)` forward passes instead of `n`.

```clojure
(require '[raster.ad.forward :as fwd])

;; Single derivative
(fwd/derivative (fn [x] (* x x x)) 2.0)  ;=> 12.0  (3x^2 at x=2)

;; Gradient of f: R^n -> R
(fwd/gradient (fn [xs] (+ (* (nth xs 0) (nth xs 0))
                          (* (nth xs 1) (nth xs 1))))
              [3.0 4.0])
;=> [6.0 8.0]  (nabla(x^2 + y^2) = [2x, 2y])
```

**When to use forward-mode:** Few inputs, many outputs (Jacobians), or when
you need derivatives of functions that take non-differentiable control flow
paths (Dual propagation handles branches naturally).

### Reverse-Mode: Closures-as-Tape

Reverse-mode computes gradients by transforming the source IR into a forward
pass + backward pass. Raster uses the **closures-as-tape** approach (from
Myia): the backward pass is a closure that captures forward intermediates,
rather than an explicit tape data structure.

For a function `f: R^n -> R`, reverse-mode computes all `n` partial
derivatives in a single backward pass — O(1) passes regardless of `n`.

The transformation operates on the walked S-expression (flat `let*` form):

**Forward pass:**
```clojure
(let* [a (* x y)       ;; record: pullback_a needs x, y
       b (sin a)        ;; record: pullback_b needs a
       c (+ b 1.0)]     ;; record: pullback_c is trivial
  c)
```

**Backward pass (generated):**
```clojure
;; Start with dc = 1.0 (adjoint of output)
(let* [dc    1.0
       db    dc                    ;; d/db(b + 1.0) = 1
       da    (* db (cos a))        ;; d/da(sin(a)) = cos(a)
       dx    (* da y)              ;; d/dx(x*y) = y
       dy    (* da x)]             ;; d/dy(x*y) = x
  [c dx dy])                       ;; return [primal, gradients...]
```

Each primitive operation has a registered **pullback** — a function that maps
output adjoints to input adjoints. These are stored in the rrule registry.

## Composable Operators: `grad` and `value+grad`

The primary API for differentiation:

```clojure
(require '[raster.ad.reverse :as rev])

;; Returns [f(x,y), df/dx, df/dy]
((rev/value+grad #'my-loss) 3.0 4.0)
;=> [25.0 6.0 8.0]

;; Returns [df/dx, df/dy] only
((rev/grad #'my-loss) 3.0 4.0)
;=> [6.0 8.0]
```

These operators are **composable with the compiler**. The returned function
carries `deftm` metadata (`:raster.core/deftm-walked-body`,
`:raster.core/deftm-params`, `:raster.core/deftm-tags`), so when used
inside a compiled `deftm`, the compiler can inline the AD-expanded form:

```clojure
(deftm train-step
  [W1 :- (Array double), b1 :- (Array double),
   W2 :- (Array double), b2 :- (Array double),
   x :- (Array double), y :- (Array double),
   lr :- Double] :- Double
  (let [vg ((rev/value+grad (var nn/loss-fn)) W1 b1 W2 b2 x y)
        loss (nth vg 0)]
    (optim/sgd-step! W1 (nth vg 1) (arrays/alength W1) lr)
    ;; ... update other params ...
    loss))
```

When `compile-aot` compiles this, the `value+grad` call is expanded
into ~80 flat bindings covering the full forward + backward pass. The compiler
then applies buffer fusion, memory merge, and bytecode emission to the
combined form — producing a single JVM class with zero per-call allocations.

## The rrule Registry

Gradient rules for primitive operations are registered in two forms:

### Runtime Pullbacks (`rrule.clj`)

Used for interpreted AD. Each rule is a factory that takes the forward result
and inputs, returning a pullback function:

```clojure
;; Multiplication: d(x*y) = [dy*y, dy*x]
(register-rrule! 'raster.numeric/*
  (fn [_result x y]
    (fn [dy] [(* dy y) (* dy x)])))

;; sin: d(sin(x)) = cos(x)*dy
(register-rrule! 'raster.math/sin
  (fn [_result x]
    (fn [dy] [(* dy (Math/cos x))])))
```

### Compile-Time Templates (`templates.clj`)

Used when the compiler inlines AD. Templates are S-expression patterns that
the reverse pass splices into the generated code:

```clojure
(register-template! 'raster.math/sin
  {:params '[x]
   :result 'y
   :adjoint 'dy
   :grads [(list 'raster.numeric/* 'dy (list 'raster.math/cos 'x))]})
```

The compiler's inline pass checks for `:grads` / `:grads-fn` metadata to
decide whether a `deftm` call is **AD-opaque** (has explicit gradient rule,
stays symbolic during AD) or **AD-transparent** (no rule, gets inlined and
differentiated through).

For the MLP training step, `dense`, `relu`, `softmax`, and `cross-entropy`
all have explicit gradient templates. Their rules are inlined as flat `let*`
bindings in the backward pass, making buffer fusion and BLAS dispatch
visible to the compiler.

## Higher-Order Differentiation

### Jets: Truncated Taylor Polynomials

For computing k-th order derivatives efficiently, Raster provides Jet
arithmetic — truncated Taylor polynomial rings:

```
Jet(order=k, coeffs=[f(a), f'(a)/1!, f''(a)/2!, ..., f^(k)(a)/k!])
```

Arithmetic on Jets uses truncated Cauchy products:

```
(Jet a) * (Jet b) = Jet c where c[k] = sum_{j=0}^{k} a[j] * b[k-j]
```

This gives all derivatives up to order k in a single pass with O(k^2)
work per arithmetic operation — much better than nested Duals which
require O(2^k) work.

```clojure
(require '[raster.ad.jet :as jet])

;; All derivatives of sin(x) at x=pi up to order 3:
(jet/higher-derivatives (fn [x] (Math/sin x)) Math/PI 3)
;=> [~0, -1.0, ~0, 1.0]  ;; [sin(pi), cos(pi), -sin(pi), -cos(pi)]
```

Math functions on Jets use recurrence relations (Griewank & Walther,
Algorithm 13.1). For example, `exp(f)`:

```
e[0] = exp(f[0])
e[k] = (1/k) * sum_{j=1}^{k} j * f[j] * e[k-j]
```

### Hessian-Vector Products

For second-order optimization, `compile-hvp-fn` computes Hessian-vector
products `H*v` without forming the full Hessian:

```clojure
(let [hvp (rev/compile-hvp-fn #'rosenbrock)]
  (hvp [1.0 1.0] [0.1 0.0]))  ;; H(1,1) * [0.1, 0]
```

This uses reverse-over-forward AD:
1. Compute `g(x) = v^T * grad(f)(x)` (a scalar function of x)
2. Differentiate `g` with respect to x → `H*v`

Cost: O(n) for n parameters, regardless of Hessian size. This enables
Newton-CG and trust-region methods on high-dimensional problems.

## Pipeline Integration: The Fixpoint Pass

When `value+grad` appears inside a compiled `deftm`, the pipeline handles
it through the **fixpoint pass**:

1. **Lower** — expands `(value+grad #'loss-fn)` into the AD-transformed body
   (forward pass + backward pass as flat bindings)
2. **Fixpoint** — the expanded backward pass may contain un-devirtualized
   arithmetic (from gradient templates). The fixpoint pass iterates:
   - **Expand** — inline remaining deftm calls
   - **Rewalk** — devirtualize new `.invk` calls introduced by expansion
   - Repeat until stable (typically 1-2 iterations)

This is what enables composable AD — `grad(grad(f))` works because each
level of differentiation produces a new set of arithmetic operations that
the next fixpoint iteration devirtualizes.

After fixpoint, the MLP training step has ~79 bindings covering:
- Forward: dense (BLAS dgemv), relu (par/map!), softmax, cross-entropy
- Backward: cross-entropy grad, softmax grad (reduce + map), dense backward
  (dger! for dW, dgemv-t! for dx), relu backward (masked multiply)
- SGD: four update loops (par/map! with broadcast)

All fully typed, ready for buffer fusion and bytecode compilation.

## Activity Analysis

Before generating the backward pass, activity analysis determines which
bindings carry gradient information:

- **Active:** depends on a parameter that has a non-zero gradient
- **Constant:** independent of all active parameters (gradient is zero)

A binding is active if any of its inputs is active. Constants skip adjoint
computation entirely, reducing both time and memory. For the MLP backward
pass, constants include the learning rate, array dimensions, and any
precomputed values that don't depend on the parameters.

## AD Through Control Flow

### Loops

Reverse-mode AD supports `loop*/recur` by storing per-iteration pullbacks:

1. **Forward:** run the loop, store each iteration's pullback in a list
2. **Backward:** replay pullbacks in reverse order, accumulating adjoints

This is exact (not truncated) reverse-mode through arbitrary-length loops.

### Parallel Forms

`par/map!` and `par/reduce` preserve their parallel structure through AD:

- **Forward:** the parallel form runs normally, also storing per-element
  pullbacks via a sequential tape pass
- **Backward for array inputs:** emit `par/map!` applying pullbacks
  element-wise
- **Backward for scalar inputs:** emit `par/reduce` summing pullback
  contributions across elements

This means the backward pass of a BLAS-accelerated forward pass still uses
BLAS operations — the compiler sees `dger!` and `dgemv-t!` in the gradient
code, not hand-written loops.

## Differentiable Agent-Based Models

The firms ABM (`raster.abm.firms.differentiable`) demonstrates AD through
complex simulation loops:

### Implicit Function Theorem (IFT)

The ABM's effort solver uses Newton-Raphson to find `e*` satisfying the
first-order condition `F(e*, params) = 0`. Differentiating through
Newton-Raphson naively would require differentiating through all iterations.

Instead, the IFT gives the exact derivative at convergence:

```
de*/dp_i = -(dF/dp_i) / (dF/de)
```

This is registered as an rrule, so reverse-mode AD through `solve-effort`
uses the IFT formula instead of unrolling the Newton iterations. The result
is exact (not approximate) and costs only one extra function evaluation
per parameter.

### Soft Decisions

Hard argmax decisions are non-differentiable. The differentiable variant
replaces them with softmax:

```
p_i = exp(u_i / tau) / sum_j exp(u_j / tau)
output = sum_i p_i * e_i
```

Temperature `tau` controls smoothness: `tau -> 0` recovers argmax,
`tau -> inf` gives uniform weights. This makes the full simulation
loop — decide, work, produce, distribute — differentiable end-to-end,
enabling gradient-based calibration of model parameters against empirical
data.

## Summary

| Feature | Implementation | Use Case |
|---|---|---|
| Forward-mode (Dual) | deftm dispatch on Dual type | Few params, Jacobians |
| Reverse-mode | Source transform, closures-as-tape | Many params, scalar loss |
| Jets | Truncated Taylor polynomials | k-th order derivatives |
| value+grad | Composable operator with deftm metadata | Training loops |
| HVP | Reverse-over-forward | Newton/trust-region methods |
| rrule registry | Pullback factories + code templates | Custom op gradients |
| Activity analysis | Forward propagation of active/const | Skip dead gradients |
| IFT | Registered rrule for implicit functions | Differentiable solvers |
| Loop AD | Per-iteration pullback storage | RNNs, ODE sensitivity |
| Parallel AD | par/map! and par/reduce preserved | BLAS in backward pass |
