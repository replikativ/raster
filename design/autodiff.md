# Automatic Differentiation in Raster

Raster differentiates ordinary `deftm` code. Because every arithmetic op
(`raster.numeric/+`, `raster.math/sin`, `raster.dl.nn/linear-nb`, …) is a typed
multiple-dispatch function, the *same* op definition runs with plain numbers, with
`Dual` numbers (forward mode), with symbolic `Sym` values, or spliced into the IR
and compiled. AD is not a separate interpreter — it is a set of *interpretations*
of the one op signature, and the compiler consumes the result: the backward pass of
a BLAS-accelerated forward pass is itself BLAS, visible to fusion and bytecode
emission. Compiled MLP/LeNet training steps (forward + AD backward + SGD) run at
143 µs / 221 µs on CPU (Valhalla JDK).

Every example is REPL-validated (`;=>` = actual output) and mirrors a law family
(O1–O10) in `test/raster/ad/laws_test.clj`.

```clojure
(require '[raster.ad.reverse :as rev]   '[raster.ad.forward :as fwd]
         '[raster.ad.jvp :as jvp]       '[raster.ad.jet :as jet]
         '[raster.ad.tangent :as tangent] '[raster.ad.templates :as tmpl]
         '[raster.numeric :as n] '[raster.math :as m] '[raster.arrays :as ra]
         '[raster.par :as par] '[raster.core :refer [deftm]]
         '[raster.sym.core :as sym] '[raster.sym.diff :as sdiff]
         '[raster.dl.nn :as nn] '[raster.dl.loss :as loss])

;; deterministic Gaussian float array, used by the array examples below
(defn- fa [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (.nextGaussian r)))) a))
```

## 1. Quick start

`value+grad` returns `[primal grads…]`; `grad` returns just the gradient (a bare
scalar for a 1-param function — JAX semantics — a vector for multiple params).

```clojure
(deftm docqs-poly [x :- Double] :- Double          ;; f = x³ + 2x, f' = 3x² + 2
  (n/+ (n/* x (n/* x x)) (n/* 2.0 x)))

((rev/value+grad #'docqs-poly) 3.0)   ;=> [33.0 29.0]
((rev/grad        #'docqs-poly) 3.0)  ;=> 29.0
```

For an array loss — float mean-squared-error over a bias-free linear layer — the
gradient of each parameter comes back typed in that parameter's own dtype:

```clojure
(deftm docqs-mse [x :- (Array float) W :- (Array float) tgt :- (Array float)
                  batch :- Long in :- Long out :- Long] :- Double
  (loss/mse-loss (nn/linear-nb x W batch in out) tgt (clojure.core/* batch out)))

(let [x (fa 8 1) W (fa 12 2) tgt (fa 6 3)          ;; fa = deterministic float array
      [loss gx gW gtgt] ((rev/value+grad #'docqs-mse) x W tgt 2 4 3)]
  [loss (.getName (class gx)) (vec (take 3 gx))])
;=> [3.3568948109944663 "[F" [0.15883432 0.15100321 0.021924283]]
```

## 2. Mental model

A `deftm` denotes a piecewise-smooth map `f : X × P → Y`: `X` the differentiable
inputs (scalars, dtype-tagged arrays), `P` the *parameters with no tangent space*
(Long indices, shapes, seeds, booleans). Differentiation produces two linear maps
at a point: the pushforward `df_x : X→Y` ("J·v", forward mode applies it to an
input tangent) and the pullback `df*_x : Y→X` ("Jᵀ·w", reverse mode applies it to
an output cotangent).

**Modes are interpretations, not implementations.** The op set is a signature; an
AD mode picks a *carrier* (what flows through the ops) × a *stage* (when):

| carrier ╲ stage | immediate (interpret) | staged (compile) |
|---|---|---|
| ℝ (Double/Float) | plain evaluation | `compile-aot` forward |
| **Dual** = ℝ[ε]/ε² | forward-mode derivative | compiled forward |
| **Jet** = ℝ[ε]/εᵏ⁺¹ | k-th-order (higher derivatives) | — |
| **Sym** (initial algebra) | symbolic trace = the IR itself | symbolic AD |
| (value, cotangent) | runtime `value+grad` | compiled reverse (spliced + fused) |

**A rule is ONE Jacobian presentation with two contractions.** For a primitive
`p`, its per-param `:grads` is the reverse contraction (Jᵀ·w). The forward
contraction (J·v) is *derived* from a small structure tag —
`{:elementwise, :symmetric, :linear, :bilinear, :scalar-loss}` — never a second
hand-written table. Dual/Jet/Sym need no rule table at all: the chain rule is
implicit in evaluation order, so any op with the right overloads just works.
There is no symbolic-vs-runtime split in a rule; the body is ordinary code and the
engine chooses whether to eval it, splice it, or trace it.

## 3. The operators

### value+grad / grad

`grad` peels the primal off `value+grad`. Shape follows arity:

```clojure
(deftm doc-quad2 [x :- Double y :- Double] :- Double (n/+ (n/* x x) (n/* x y)))

((rev/grad #'docqs-poly) 3.0)          ;=> 29.0        ; 1 param → bare scalar
(vec ((rev/grad #'doc-quad2) 3.0 4.0)) ;=> [10.0 3.0]  ; n params → vector
```

Long/shape slots are `⊥` (NoTangent): they get `nil` grads and never seed. In the
array example above, `gtgt` is a typed zero and the `batch/in/out` slots are
`nil` (law O3).

### jvp — directional derivative through array ops

`jvp` builds a function taking the primal args followed by a tangent per
differentiable arg. It threads tangents forward through every array op, deriving
each op's frule from its structure class (elementwise / symmetric / linear /
bilinear derived mechanically; the residue — softmax, norms, attention — is a few
hand-written frules). Output tangents are plain `float[]`/`double[]`, so the whole
thing stays fusible and GPU-ready (law O1b).

```clojure
(let [jf (jvp/jvp #'docqs-mse)
      x (fa 8 1) W (fa 12 2) tgt (fa 6 3)
      v (fa 8 9)                              ;; tangent seeded on x
      [primal tangent] (jf x W tgt 2 4 3 v (float-array 12) (float-array 6))]
  [primal tangent])
;=> [3.3568948109944663 1.0070443153381348]   ; directional FD ≈ 1.0070641835
```

### hvp — Hessian-vector product (forward-over-reverse)

`jvp/hvp` is the JVP fold applied to the *reified gradient program* (Pearlmutter
1994): one reverse sweep to build ∇f, one forward sweep to push a tangent through
it. Cost is O(cost of f) per H·v, no Hessian formed. It returns `[grads Hv]`.

```clojure
(deftm docquad [x :- Double y :- Double] :- Double (n/+ (n/* x x) (n/* x y)))
;; quad = x² + xy, Hessian [[2 1][1 0]]
((jvp/hvp #'docquad) 3.0 2.0 1.0 0.0)   ;=> [[8.0 3.0] [2.0 1.0]]   ; grads, H·e1
((jvp/hvp #'docquad) 3.0 2.0 0.0 1.0)   ;=> [[8.0 3.0] [1.0 0.0]]   ; H·e2
```

It runs through array layers too — an MLP loss, seeding a tangent on `x`, matched
against FD-of-gradient in law O4b:

```clojure
(let [hf (jvp/hvp #'docqs-mse)
      x (fa 8 1) W (fa 12 2) tgt (fa 6 3) v (fa 8 9)
      [grads hv] (hf x W tgt 2 4 3 v (float-array 12) (float-array 6))]
  [(.getName (class (nth hv 0))) (vec (take 3 (nth hv 0)))])
;=> ["[F" [0.86591256 1.34398 -0.076107144]]
```

### Double-backward via transparent composition

The headline: **nesting differentiation happens through ordinary program use.**
The same syntax works nested or not. The idiom is to call an inner gradient inside
a scalar `deftm` and differentiate *that*:

```clojure
(deftm doccube [x :- Double] :- Double (n/* x (n/* x x)))   ;; x³

(deftm docouter [x :- Double] :- Double
  (let [vg ((rev/value+grad #'doccube) x)                   ;; g = 3x²
        g  (double (nth vg 1))]
    (n/* g g)))                                             ;; 9x⁴ → d/dx = 36x³

((rev/value+grad #'docouter) 2.0)   ;=> [144.0 288.0]       ; (3·4)², 36·8
```

For a 1-param scalar function `grad` returns a bare scalar, so the textbook
spelling composes literally, and `reified-grad` scalarizes a gradient program so
`value+grad` can consume it (law O4c/O4d):

```clojure
((rev/grad (rev/grad #'doccube)) 2.0)                    ;=> 12.0   ; f″ = 6x
((rev/value+grad (rev/reified-grad #'doccube 'x)) 2.0)   ;=> [12.0 12.0]  ; [3x², 6x]
```

Applying an operator *directly* to another operator over a tuple output is
ill-typed — a product cotangent space has no canonical seed — and it **fails
loud** (this is a feature: it used to be a silent zero):

```clojure
(try ((rev/value+grad (rev/value+grad #'doccube)) 2.0)
     (catch Exception e (.getMessage e)))
;=> "Cannot differentiate the tuple-valued function `…` — its output is a vector,
;    which has no canonical cotangent seed (this is what value+grad returns;
;    value+grad of value+grad is ill-typed). Compose transparently instead: call
;    the inner gradient in a scalar deftm and differentiate that … For 1-param
;    scalar fns, (grad (grad f)) composes directly."
```

### Loops & recurrences — write loops, they differentiate

The other headline: a raw `(loop [i 0 …] …)` with a static/affine trip count is
**canonicalized to a `par/scan`/`par/reduce` in `ad-prepare`** — before the
reverse transform ever sees it — so the user does not rewrite anything (unlike
JAX, which makes you convert `while`→`scan`). A loop-written linear RNN and its
`par/scan` twin give bit-identical gradients (law O10):

```clojure
(deftm docrnn-loop [w :- Double h0 :- Double x :- (Array double) sn :- Long] :- Double
  (let [out (double-array sn)]
    (loop [i 0 h h0]                                   ;; per-step store, return carry
      (if (< i sn)
        (let [h2 (n/+ (n/* w h) (ra/aget x i))] (ra/aset out i h2) (recur (inc i) h2))
        h))))

(deftm docrnn-scan [w :- Double h0 :- Double x :- (Array double) sn :- Long] :- Double
  (let [out (double-array sn)                          ;; explicit par/scan: out[i]=acc_i
        h (par/scan out acc h0 i sn double (n/+ (n/* w acc) (ra/aget x i)))]
    (ra/aget h (dec sn))))

(let [w 0.6 h0 0.8 x (double-array [0.7 -1.3 0.4 0.9])
      r1 ((rev/value+grad #'docrnn-loop) w h0 x 4)
      r2 ((rev/value+grad #'docrnn-scan) w h0 x 4)]
  [(take 3 r1) (= (nth r1 1) (nth r2 1))])
;=> [(0.92688 0.28719999999999996 0.1296) true]   ; loop ≡ scan, exact
```

`par/scan` is the sanctioned differentiable recurrence: because `out[i] = acc_i`,
the output array *is* the carry tape — no closure tape, no residual stack (Griewank
store-the-carry at every step). Tail-accumulation loops lift to `par/reduce`:

```clojure
(deftm doc-ssq [xs :- (Array double) rn :- Long] :- Double
  (loop [i 0 acc 0.0]
    (if (< i rn) (recur (inc i) (n/+ acc (n/* (ra/aget xs i) (ra/aget xs i)))) acc)))
(let [xs (double-array [1.0 2.0 3.0])
      [v dxs] ((rev/value+grad #'doc-ssq) xs 3)] [v (vec dxs)])
;=> [14.0 [2.0 4.0 6.0]]   ; Σx², grad 2xᵢ exact
```

A **data-dependent** trip count (the bound reads the carry) has no sized residual
stack, so the soundness gates decline and it fails loud, pointing at the modeling
alternatives:

```clojure
(deftm docdatadep [x :- Double] :- Double
  (loop [i 0 acc x] (if (< i (long acc)) (recur (inc i) (n/+ acc x)) acc)))
(try (rev/value+grad #'docdatadep) (catch Exception e (subs (.getMessage e) 0 96)))
;=> "value+grad: cannot assemble a runtime gradient for `#'user/docdatadep` — the body reduces to a l"
;    …full message: express the recurrence via `par/scan`, a convergence loop via a fixed-point solve.
```

### Dual forward mode (scalar)

`:mode :forward` runs the body over `Dual` numbers; agreement with reverse is law
O1. Forward is the right choice for few inputs / many outputs, and its Dual
comparison overloads let it flow through `if` branches (a.e.-correct):

```clojure
(deftm docfwd [x :- Double] :- Double (n/+ (n/* x (m/sin x)) (m/exp x)))
[(nth ((rev/value+grad #'docfwd :mode :forward) 1.3) 1)
 (nth ((rev/value+grad #'docfwd :mode :reverse) 1.3) 1)]
;=> [4.980603330248401 4.9806033302484005]
```

The lower-level `fwd/derivative` / `fwd/gradient` compute single derivatives and
∇ directly over Duals:

```clojure
(fwd/derivative (fn [x] (n/* x (n/* x x))) 2.0)   ;=> 12.0
(vec (fwd/gradient (fn [xs] (n/+ (n/* (nth xs 0) (nth xs 0))
                                 (n/* (nth xs 1) (nth xs 1)))) [3.0 4.0]))
;=> [6.0 8.0]
```

### Jets — higher-order derivatives

A `Jet` is a truncated Taylor ring ℝ[ε]/εᵏ⁺¹; one pass gives all derivatives up to
order k in O(k²) work (vs nested Duals' O(2ᵏ)). Law O4:

```clojure
(vec (jet/higher-derivatives (fn [x] (m/sin x)) Math/PI 3))
;=> [1.2246467991473532E-16 -1.0 -1.2246467991473532E-16 1.0]
;   [sin(π)≈0, cos(π)=-1, -sin(π)≈0, -cos(π)=1]
```

### Symbolic — two routes, one answer

Because `Sym` is the *initial* algebra, symbolic differentiation is free: run any
mode over `Sym` carriers. `sym/differentiate` (the calculus table) and `Dual{Sym}`
(the *same* numeric ops over a symbolic carrier — no table) agree (law O6):

```clojure
(defn- o6f [x] (n/+ (n/* x (m/sin x)) (m/exp x)))
(let [traced  (sym/unwrap (o6f (sym/sym 'x)))
      route-a (sdiff/differentiate traced 'x)            ;; diff.clj calculus
      dx (fwd/make-dual (sym/sym 'x) (object-array [(sym/sym 1)]))
      route-b (sym/unwrap (aget ^objects (.partials (o6f dx)) 0))]  ;; Dual{Sym}
  [route-a route-b])
;=> [(+ (+ (sin x) (* x (cos x))) (exp x))
;    (+ (+ (* x (* 1 (cos x))) (* 1 (sin x))) (* 1 (exp x)))]
;   both = sin(x) + x·cos(x) + eˣ  (route-b before (* 1 ·) simplification)
```

### IFT / fixed-point

`fixed-point-solve` finds z* = g(z*, θ) and carries an rrule that applies the
Implicit Function Theorem — dz*/dθ = (∂g/∂θ)/(1 − ∂g/∂z) — instead of unrolling
the iteration. Exact at convergence, O(1) memory:

```clojure
(require '[raster.ad.fixed-point :as fp] '[raster.core :refer [ftm]])
(deftm docift [theta :- Double] :- Double            ;; g(z,θ)=½z+θ ⇒ z*=2θ, dz*/dθ=2
  (fp/fixed-point-solve (ftm [z :- Double th :- Double] :- Double (n/+ (n/* 0.5 z) th))
                        0.0 theta 1e-12 1000))
((rev/value+grad #'docift) 3.0)   ;=> [5.999999999999318 2.0]
```

### :mode :auto and admissibility

`:auto` selects the cheapest *admissible* mode: cheapest by the Griewank dimension
test, but constrained to carriers whose ops are all covered. `forward-coverage`
names the uncovered ops. `erf` has a Dual lift, so `:auto` may go forward; a raw
JVM `Math/expm1` interop can never have a Dual lift, so `:auto` must stay reverse
(law O5):

```clojure
(deftm docerf   [x :- Double] :- Double (raster.sci.special/erf x))
(deftm docuncov [x :- Double] :- Double (n/* x (Math/expm1 (double x))))
[(rev/forward-admissible? #'docerf)
 (rev/forward-admissible? #'docuncov)
 (:uncovered-ops (rev/forward-coverage #'docuncov))
 (nth ((rev/value+grad #'docuncov :mode :auto) 0.7) 1)]   ;; auto → reverse, correct
;=> [true false [Math/expm1] 0.6912748604105386]
```

Forcing an inadmissible mode fails at *construction* time (never a
No-matching-method at call time):

```clojure
(try (rev/value+grad #'docuncov :mode :forward) (catch Exception e (.getMessage e)))
;=> "value+grad :mode :forward is not admissible for `#'user/docuncov` — ops
;    without a Dual lift: Math/expm1. Mode selection is constrained by carrier
;    coverage (framework §11): use :mode :reverse, or add the missing Dual overloads…"
```

## 4. Adding a differentiable op

Register a rule and the op becomes AD-opaque (stays symbolic for the transform);
without one, a `deftm` is simply inlined and differentiated through (the Zygote
recurse-into-IR default). A rule is:

- **`:grads`** — canonical per-param reverse expressions (Jᵀ·w); auto-compiled to a
  code-generating `:grads-fn` at registration. One source of truth per op.
- **`:grads-fn`** — the escape hatch: emit flat `let*` bindings directly (used for
  array kernels that call BLAS backward routines, keeping them fusion-visible).
- **`:structure`** — a Jacobian tag (`:elementwise` / `:symmetric` / `:linear` /
  `:bilinear` / `:scalar-loss`). From it the forward rule (`:jvp-fn`) and, for
  closure under second-order AD, the derived adjoint are synthesized — no second
  table.

Registering a scalar primitive with both `:grads` and `:structure` makes reverse
*and* forward work through it:

```clojure
(deftm docsq [x :- Double] :- Double (n/* x x))               ;; y = x²
(tmpl/register-template! 'user/docsq
  {:params '[x] :result 'y :adjoint 'dy
   :grads [(list 'raster.numeric/* 'dy (list 'raster.numeric/* 2.0 'x))]})
(tmpl/merge-into-template! 'user/docsq {:structure {:class :elementwise :in #{0}}})

(deftm docusesq [x :- Double] :- Double (docsq (n/* 3.0 x)))  ;; (3x)²=9x², f'=18x
[(nth ((rev/value+grad #'docusesq) 2.0) 1)     ;; reverse: uses :grads
 (nth ((jvp/jvp #'docusesq) 2.0 1.0) 1)]       ;; forward: :jvp-fn derived from :structure
;=> [36.0 36.0]
```

Emitted grad expressions must use fully-qualified `raster.numeric/*` etc. so the
walker resolves them and devirtualization fires downstream.

## 5. The tangent protocol

Cotangents form a commutative monoid `(⊕, 0̄)` with scalar action and a projector
Π. Two distinct "nothings": `0̄` (ZeroTangent — a dynamic zero, still typed and
shaped) and `⊥` (NoTangent — a *static* absence of tangent space, Long/index/bool
slots). ⊕ (`grad-acc`) must be commutative/associative so reverse-emission order
never matters; Π maps a raw cotangent into *its own primal's* tangent space,
deriving dtype/shape from the primal type — this is what makes per-param dtypes
survive without a global switch (framework §5, task #44; law O3b):

```clojure
(deftm docpi [x :- Double y :- Float] :- Double (n/* x y))
(let [[v dx dy] ((rev/value+grad #'docpi) 2.0 (float 3.0))]
  [(class v) (class dx) (class dy)])
;=> [java.lang.Double java.lang.Double java.lang.Float]  ; each grad in its primal's space
```

Zeros are materialized from the primal's tag, and `⊥` has no zero (NoTangent ≠
ZeroTangent) — so it fails loud rather than mis-shaping a slot:

```clojure
[(tangent/tangent-kind 'floats) (tangent/tangent-kind 'long)
 (tangent/zero-expr 'floats 'n) (tangent/zero-expr 'double 'n)
 (try (tangent/zero-expr 'long 'n) (catch Exception _ :NoTangent-has-no-zero))]
;=> [{:kind :array, :dtype :float} {:kind :none}
;    (float-array n) 0.0 :NoTangent-has-no-zero]
```

## 6. Correctness: the laws suite

`test/raster/ad/laws_test.clj` encodes the commuting-diagram obligations as
property-style tests over a `deftm` corpus, with central finite differences as the
external referee where one exists.

| Law | Guarantees |
|---|---|
| O1  | mode agreement: reverse = forward = FD = analytic on the shared scalar domain |
| O1b | array-forward (JVP): directional tangent J·v = directional FD, per param class |
| O3  | tangent algebra: ⊕ fan-out exact; 0̄ identity; ⊥ (Long) slots nil, never seed |
| O3b | Π protocol: per-param dtype preserved (float primal → float grad) |
| O4  | composition: HVP = FD(grad); Jet-k coeffs = k-th derivative |
| O4b | HVP-through-layers (forward-over-reverse) vs FD-of-grad |
| O4c | double-backward (RoR closure): grad-of-grad, Hₓₓ·c through NN blocks |
| O5  | boundaries: unsupported forms throw; :auto admissibility; ⊥ never seeds |
| O6  | initiality: `sym/differentiate` ≡ `Dual{Sym}` ≡ analytic |
| O7  | adjoint identity ⟨J·v, w⟩ = ⟨v, Jᵀ·w⟩ — frule and rrule test each other |
| O8  | rule linearity: every pullback is linear over the tangent algebra |
| O9  | par/scan recurrence vs analytic closed form + FD |
| O10 | loop ≡ scan commuting law: loop-written RNN gradients = scan-written, exact |

The flagship checks are self-refereeing. **O7** — the adjoint dot-product identity
— pits the forward rule against the reverse rule with no FD oracle:

```clojure
(deftm doc-lnb [x :- (Array float) W :- (Array float) b :- Long i :- Long o :- Long]
  :- (Array float) (nn/linear-nb x W b i o))
(deftm doc-dotw [x :- (Array float) W :- (Array float) w :- (Array float)  ;; g = ⟨f(x,W), w⟩
                 b :- Long i :- Long o :- Long] :- Double
  (let [y (nn/linear-nb x W b i o) n* (clojure.core/* b o)]
    (loop [k 0 s 0.0] (if (clojure.core/< k n*)
      (recur (clojure.core/inc k) (+ s (* (ra/aget y k) (ra/aget w k)))) s))))
(defn- fdot [a b] (areduce a k s 0.0 (+ s (* (double (aget a k)) (double (aget b k))))))
(let [x (fa 8 11) W (fa 12 12) v (fa 8 13) w (fa 6 14)
      [_ jv] ((jvp/jvp #'doc-lnb) x W 2 4 3 v (float-array 12))   ;; J·(v,0)
      gx (nth ((rev/value+grad #'doc-dotw) x W w 2 4 3) 1)]        ;; Jᵀ·w over x
  [(fdot jv w) (fdot v gx)])                                       ;; must be equal
;=> [-0.5583058855371057 -0.5583058402333747]   ; ⟨J·v,w⟩ = ⟨v,Jᵀ·w⟩ (float)
```

And the two independent second-order routes — **reverse-over-reverse** (O4c) and
**forward-over-reverse** HVP (O4b) — agree bit-exactly, the commuting square at
machine precision:

```clojure
(let [ror (nth ((rev/value+grad (rev/reified-grad #'doccube 'x)) 2.0) 1)  ;; d/dx(3x²)
      hvp (first (nth ((jvp/hvp #'doccube) 2.0 1.0) 1))]                   ;; f″·1
  [ror hvp (== ror hvp)])   ;=> [12.0 12.0 true]
```

## 7. GPU status (honest)

The AD-generated backward matmuls run on Intel Arc (Level Zero / XMX) and match
CPU BLAS. Validated on the `mse ∘ linear-nb` probe (branch `ad/gpu-validation`):
forward `linear-nb` → resident `:nt` GEMM (relerr 2.55e-4, fp16), backward
`linear-dx` → `:nn` GEMM (2.28e-4), backward `linear-dW` → `:tn` weight-gradient
GEMM (2.60e-4, `:tn` newly wired). The post-SOA IR carries all three GEMM `.invk`
with `:raster.op/original` intact — the AD transform reaches GPU lowering
unchanged.

**Pinned gap:** a *whole* train step is not yet one resident graph. The non-matmul
gradient ops in the composed path (the loss reduction, the `daxpy-diff` elementwise
gradient) have no resident kernel there yet, so `compile-gpu-program` returns nil
and falls back gracefully. The residency / saved-activation policy (store-vs-
recompute for forward intermediates; params + optimizer moments as the donation
set) is *designed* but not built — it converges with the loop-checkpointing policy
(framework §15: one store-vs-recompute pass, two clients). Small-M fp16 DPAS GEMM
has a known boundary bug (rows 1..M-1 wrong for tiny M; multiples of 16 correct),
independent of AD.

## 8. Boundaries & limits

- **Data-dependent loops** (`while`/convergence) are not unrolled — model them as a
  `fixed-point-solve` (adjoint-of-fixed-point / IFT) instead. Fail-loud otherwise.
- **Multi-carry loops** (`loop [i a b]`) are still rejected by the canonicalizer
  (v2: tuple-packed carry); they retain the older ArrayList tail-loop tape on the
  compiled/inline path only.
- **Mutation soundness (R1)** — the tape/IR must capture pre-mutation values;
  raster's typed named buffers make the store-adjoint discipline cheap, but this is
  currently guarded by convention, not analysis (deferred).
- **Vector-tail rule** — `value+grad` of `value+grad` is ill-typed and fails loud
  (§3); compose transparently or use `reified-grad`.
- **Skipped frules** (documented A3 skips — reverse works, forward fails loud
  naming the op): `solve`, `einsum`, `maxpool2d` (needs a gather-at-argmax kernel),
  `array-det`, effectful `dgemm!`. `par/scan` forward mode also fails loud (a
  forward SOAC fold is not landed).
- **Kinks** — differentiation through `if` and through activation kinks (relu) is
  correct almost-everywhere; there is no oracle *at* a nondifferentiable point, so
  the laws sample away from kinks.
- **Known bugs.** #73: the walker's single-method fallback can pick a wrong-dtype
  parametric specialization for a composite when TypedClojure inference fails
  inside it (a `[F→[D` primal miscompile — a devirt hole, not an AD defect; worked
  around with single-method float wrappers). #75: `par/reduce` over `aget` reads has
  a namespace-sensitive lowering hazard (segred aget).

## Summary

| Operator | Mode / mechanism | Use case | Law |
|---|---|---|---|
| `value+grad` / `grad` | reverse source-transform | scalar loss, many params | O1, O3 |
| `:mode :forward` / `fwd/*` | Dual carrier | few inputs, Jacobians, branches | O1 |
| `jvp/jvp` | JVP source-transform (derived frules) | directional deriv through arrays | O1b, O7 |
| `jvp/hvp` | forward-over-reverse | Hessian-vector (Newton-CG, trust region) | O4b |
| `reified-grad` + `value+grad` | reverse-over-reverse | grad-of-grad, ‖∇f‖², MAML | O4c |
| transparent composition | ordinary program use | any nested differentiation | O4d |
| `par/scan` / raw loop | canonicalized recurrence (out-as-tape) | RNNs, sensitivity | O9, O10 |
| `jet` | truncated Taylor ring | k-th-order derivatives | O4 |
| `sym/differentiate`, `Dual{Sym}` | Sym initial algebra | symbolic derivatives | O6 |
| `fixed-point-solve` | IFT rrule | differentiable solvers | — |
| `:mode :auto` | Griewank + admissibility | let the engine pick | O5 |
| `register-template!` | `:grads` + `:structure` | custom differentiable ops | O8 |
| tangent protocol | typed 0̄/⊥, Π projector | per-param dtype correctness | O3b |
