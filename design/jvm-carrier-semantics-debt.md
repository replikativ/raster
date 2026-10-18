# JVM carrier/conversion debt

Numeric declarations, JVM stack categories and source boxing carriers are not
interchangeable. Fixes must preserve conversions, reference identity and input
evaluation order, not merely numerical values on valid inputs.

## Nullable numerical values and boxing

Four properties are independent:

| Property | Authority |
| --- | --- |
| Logical numerical dtype | Retained typing and differentiation rules |
| Presence on normal return | Producer/control-flow facts, not the numerical dtype |
| Emitted JVM carrier | Actual method descriptors and emitted stack values |
| Surviving allocation | Warmed counters and JIT/escape-analysis evidence |

A logically Double cotangent may be absent (`nil`). Helper extraction must not
turn it into a primitive parameter, return or local solely from its logical tag.
Public primitive descriptors and successful primitive-array reads can establish
stronger facts. A primitive-array descriptor admits null; its element read only
establishes scalar presence *on normal completion*, without removing null/bounds
checks. Unknown producers retain reference-capable numerical boundaries.

### Selected entry carriers and primitive evaluation

`method-entry/selected-parameter-carriers` projects the actual registered
`invk` interface under the existing definition owner. Historical source and
receiver observations precede analysis and are checked before publication.
The projection retains real parameter Classes and the Object return; it does
not generate an interface, infer presence from tags, or invoke user code.

`intrinsics/primitive-fp-total?` queries a separate facet of the existing
intrinsic table. Its contract covers completed primitive floating arithmetic
and ordered comparisons, including IEEE zero/NaN/infinity behavior. It does not
certify source operand completion, reference unboxing or a generic invocation:
Object/Object division can throw even with boxed Double operands, while the
primitive Double/Double overload is IEEE. Integer overflow/division and absent
contracts conservatively abstain.

These are proof inputs, not enabled source rewrites. Guarded-loop normalization
must additionally establish exact source-entry/call correspondence, lexical
producer facts and safe movement of the predicate, prefix and count. A reference
array Class is not nonnull evidence; reuse of a shape evaluation requires a
dominating successful check on the same immutable array identity. Zero-trip and
false-guard behavior must survive optimization. No serial one-work-item wrapper
substitutes for proving parallel ownership.

Shared recovery of scalar-cast/read presence is held out until their source
evaluation obligations survive value liveness and DCE. Being present on normal
return does not mean that a null/bounds/conversion failure may be discarded.

Arithmetic emission consumes actual operand carriers, including the return
carrier of each intermediate operation. Reference-carried Numbers must not be
rounded through double merely because a primitive numerical result is desired.
Mixed `Numbers` overloads are semantically significant: Double/Object division
is IEEE, whereas Object/Object division checks zero. A source cast descriptor
alone does not prove that its emitter returns a primitive value.

Direct variadic core arithmetic follows core's nested `nary-inline` operations;
an overflowing inner operation can run before a later argument effect. An
explicit IFn call instead evaluates its arguments before entering the function.
Independent source-oracle tests cover both boundaries rather than assuming they
have the same order. This does not qualify all existing primitive numerical
recipes as complete Clojure semantics: primitive integer division, narrowing
and Character carriers still require their own contracts and qualification.

### Open: large backward-loop materialization

The actual immutable MMM fixture has demonstrated surviving Double and product
allocation. Small closed quadratic controls do not reproduce its row-dependent
allocation, despite Object-returning signatures. A descriptor by itself is not
an allocation measurement. Method size, inlining and escape analysis must be
examined alongside retained loop carries/products and actual boxing sites.

Nullable helper and arithmetic correctness repairs are not an allocation-gain
claim. The small publication slice also does not finish the full MMM zero-trip
case: the ordinary-call walker still inserts inferred primitive operand casts.
Removing those casts exposes existing DCE exceptional-evaluation gaps, including
negative allocation extents. That follow-up must preserve source evaluation
obligations, not weaken exception tests or replace absent gradients with zero.

The intended representation lowering scalar-replaces nonescaping products and
uses separate presence/payload state for proven-safe internal optional scalars.
It materializes reference values at escaping, public or identity-observable
boundaries. Acceptance requires zero-trip and changed-input numerical checks,
identity/effect/exception controls, and matched warmed allocation measurements
on the real model; stable internal port order alone proves no speedup.

## Open: primitive Character completion

Confirmed against commit `c57493ef0ca41056426f22f79fba976d0c5fc88a`, before
the bitwise Long/Object repair, using Clojure 1.12.0:

```clojure
(fn [left right]
  (clojure.core/bit-and (clojure.core/char left)
                        (clojure.core/char right)))
```

At `[7 40]`, independent Clojure source throws `IllegalArgumentException`:
Character is not a supported Numbers bitwise operand. The original bytecode
owner instead returns Integer `0`. The Long-width repair returns Long `0`;
neither is correct Character behavior. Negative inputs additionally expose
the existing unchecked `i2c` conversion rather than checked source conversion.

To reproduce without source rewriting, pass the quoted body above to
`raster.compiler.backend.jvm.bytecode/compile-specialized-class!` in a method
spec with `:name 'run`, `:params '[left right]`, `:return-tag 'Object`,
`:walked-body [body]`, and `:source-ns *ns*`; invoke the returned `run` Method
with `(object-array [7 40])`. Here `body` is the inner bitwise expression, not
the enclosing `fn` form. Compare with `(eval (list 'fn '[left right] body))`.

The producer currently emits `char` as `i2c` with result category `:int`, losing
the distinction between Character and Integer before a consumer sees it.
This belongs in shared physical-carrier/conversion handling, not a bitwise
function-name exception. A fix must qualify checked conversion, actual primitive
char parameter consumers, explicit char-to-int conversion, Object boxing and
argument effect/exception order together. A checked Character-reference result
is a possible representation, but needs compatible consumers and unboxing.

Raw boxed Character rejection is covered by the bitwise differential tests.
Primitive Character construction parity is **not** qualified by those tests or
by restoring the keyed random distributions. Keep this debt open until its
independent source oracle passes; do not restore dtype-only unboxing to hide it.
