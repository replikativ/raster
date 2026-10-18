# JVM carrier/conversion debt

Numeric declarations, JVM stack categories and source boxing carriers are not
interchangeable. Fixes must preserve conversions, reference identity and input
evaluation order, not merely numerical values on valid inputs.

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
