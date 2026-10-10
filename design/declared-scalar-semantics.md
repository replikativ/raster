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

The motivating model observation is conditional: a Float-projected loss scale
reproduced the GPU cotangent, while its declared Double value reproduced the CPU
counterfactual on the same prediction. That does not prove this change satisfies
the original real-weight model gate. The original gate and tolerances remain
unchanged and must be rerun separately.
