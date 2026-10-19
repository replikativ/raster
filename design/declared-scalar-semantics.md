# Declared scalar precision

Resolved scalar types are independent of tensor storage precision. A declared
Double parameter or hoisted local stays Double in the scalar environment,
KernelBody and kernel ABI even when tensor storage is Float. A genuine Float
scalar stays Float in a Double-storage kernel. Parametric scalar types still
resolve through the existing deftm specialization; integer widths are unchanged.

This replaces implicit floating scalar projection to the kernel element dtype.
It is a numerical and ABI behavior change: `:dtype :float` no longer silently
rounds declared Double scalar values before evaluating their retained Double
operations. Express intended narrowing with a Float declaration or explicit
conversion. Targets must support the retained operations; unsupported precision
must not be repaired by silently narrowing arguments.

Array storage selection and its declared-storage policy are unchanged. No
loss-specific rule or new compile option is introduced. Parameter and binder
derivation plus emitter metadata fallback share the declared-width rule; the
resident chain host binder uses the emitted physical ABI and the existing checked
runtime scalar conversion. Packaged persistent
compiler artifacts remain separated by the existing compiler-build fingerprint,
including the build Git revision. A development REPL must clear in-process
compiler caches when reloading these compiler definitions.

## Logical types are not storage or presence proofs

A logical numerical type selects numerical operations. It does not establish
that a value is non-null, that its producer returned a primitive, or that its
box identity is unobservable. Those facts have separate authorities:

- Logical types come from retained inference and declarations.
- Primitive carriers come from the actual ABI or checked producer lowering.
- Presence describes normal return only; it does not authorize removing an
  evaluation that may throw or moving it past effects.
- Object-input completion retains the original evaluation context and saved
  reference. Ordinary bindings, aliases and ANF-flattened bindings must preserve
  that context independently of their logical numerical tags.

Native scalar replacement may remove a completed lexical alias's box storage
only when its uses are closed over compatible primitive consumers. It must not
recompile the producer under a different overload or replace reference-taking
Clojure calls with primitive conversions. Unknown consumers, identity-sensitive
uses, nullable joins and incompatible widths retain their completion barrier;
native admission must reject an undischarged barrier rather than guess.

Presence simplification tracks evaluation obligations by fresh binding-occurrence
identity. Aliases transport those obligations; shadowed names do not share an
identity. Only when a presence proof is consumed does the pass mark its original
producer with `:raster.effect/preserve-evaluation`. Analysis tokens never enter
portable IR. The shared effects classifier denies removal and CSE, and TypedSOAC
retains source ordering for that marker. Direct expression tests retain the
original evaluation at the test site. This does not establish general arithmetic
totality or permit unrelated dead computations to be removed.

The implementation work validating these obligations is not a claim of complete
AD or native coverage. Qualification includes source/JVM/native comparisons,
reference-identity and exception-order controls, and actual consumer workloads.

The motivating model observation is conditional: a Float-projected loss scale
reproduced the GPU cotangent, while its declared Double value reproduced the CPU
counterfactual on the same prediction. That does not prove this change satisfies
the original real-weight model gate. The original gate and tolerances remain
unchanged and must be rerun separately.
