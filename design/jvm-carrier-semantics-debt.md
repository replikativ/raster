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

### Typed entrypoint artifact scope

The typed-entrypoint artifact records the emitted recipe, including the existing
Byte/Short/Char int-stack return boxing as Integer. That is physical provenance,
not qualification of those source boxing semantics. Native scalar materialization
proofs must not reinterpret it as Byte/Short/Character boxing.

The accompanying factory controls also exposed the Object-ABI input bridge
casting Character to Number. The bridge now uses Character.charValue for an
actual `C` parameter; this does not fix the producer `i2c`/boxing or checked-range
debt above. Keep the broader Character obligation open.

## Source contracts are not emitted entrypoints

The frontend retains a same-receiver `:raster.core/source-entry` specification.
It shares the immutable specification with `:raster.core/jit-spec` when JVM JIT
is enabled, but source ownership does not require a JVM compilation request.
The source and receiver identities, signature and retained body must still
agree with registration before this contract can be consumed. It is trusted
compiler-owned process data, not an authenticated user certificate.

An emitted typed-entrypoint artifact instead describes an actual loaded wrapper,
compute Method and boxing recipe. The source contract must not pretend those
Methods or recipes exist before code generation. Both kinds retain historical
definition premises through the existing compilation-input owner.

The first native numeric-region projection supports closed Double value regions.
The source-only proof independently checks a closed computation at actual primitive
invocation input Classes and requires that the computation already returns double;
an annotation alone never proves its result or invents a conversion. Projection
additionally closes every use, preserving source order and rejecting identity,
escape, mutation, unknown calls and unsupported conversions. Unresolved `Math`
aliases are not Class identities and remain refused. Float and structured/loop
regions still need corresponding representation proofs; this is not full native AD
coverage or a measured MMM allocation/performance improvement.

## Exceptional evaluation after presence refinement

The wider qualification gate exposed reliance on qualification-only casts in the
old generic-call walker. Restoring only that old handler makes the checked Long
overflow and negative allocation presence tests pass. Removing those spurious
casts is necessary to preserve ordinary Object inputs, but exposed later
optimization discarding evaluation that can throw.

Do not restore bogus casts or confuse external purity with totality. The compiler
must preserve the original overflow/extent checks through presence simplification
and DCE, while allowing separately proved redundant allocation/materialization to
disappear. The public differential controls are
`checked-primitive-presence-test/compiled-presence-retains-the-original-overflow`
and `normal-return-test/folding-presence-must-preserve-failing-allocation` (JVM and
native). These controls now pass with explicit source evaluation obligations in
the shared effect descriptor: checked integral arithmetic uses the existing
overflow/range policy, and array extent checks follow the actual Number.intValue
constructor narrowing. Proved-safe literal arithmetic and extents remain
removable. Unknown allocating helpers retain their evaluation until a source or
entry-domain proof can discharge their checks. External effect analysis stays
separate and cached; source obligations are recomputed from current policies.

Object-preserving completion inherits its operand's evaluation obligations, but
does not grant CSE permission across separate boxing contexts. Product snapshots
are closed with the existing exception-aware DCE inside the expansion fixpoint,
before GPU typedness enforcement. This removes absent dead cotangents without
fabricating a scalar type. These are denial-only obligations, not a totality
certificate for arbitrary helpers, accesses or loops.

Array-valued product fields now use the independent producer facts already
derived by normal-return analysis. A proven array is already a reference:
retaining its original evaluate-once snapshot preserves identity without an
additional Object-completion context. Later expansion may expose this producer
proof; existing snapshots can discharge only that same Object contract. Mere
type hints and primitive/checkcast contracts cannot use this rule. Differential
controls cover mutation/alias identity, once-only construction and its failure.

Scalar exceptional obligations are shared with TypedSOAC source normalization.
An identical check over immutable operands may reuse a dominating successful
evaluation, but the first check stays ordered. Its successful SSA result may
also inform index proofs without making that evaluation removable. Allocation
checks remain on storage boundaries, not scalar ABI ports.

The affected gate passes 434 tests / 3,839 assertions, including the complete
TypedSOAC frontend, initialization and invocation namespaces. Integral invocation
scalar admission now rejects fractional and out-of-range values, including
retained parameters absent from the final program-input list. That independent
prerequisite landed in PR #1507; it grants no narrower range assumptions.
Training qualification remains
open: the unchanged public LoRA and SFT fixtures now expose checked dimension
arithmetic after device work. Reordering new possible exceptions ahead of prior
effects is not justified by pure arithmetic alone. The generic equation-first
template is compiled before concrete arguments; prepared scalar capture must
not be mistaken for an existing compilation-time range proof. Closing this gap
requires a guarded valid-domain/specialization proof or ordered execution support,
not omission of the new flags or an unchecked lowering. No actual MMM allocation
improvement or end-to-end training performance result is established here.

### Next: guarded scalar domains, not descriptive range metadata

The current invocation type guard establishes only the complete representable
integral dtype domain. A smaller interval must be a real entry precondition,
checked before a numerical evaluator or storage realization consumes it, and
retained by the compilation and invocation boundaries together. A generic
template cannot consume point facts from later materialization. Likewise, a
prepared artifact's captured scalars do not make an already compiled generic
template range-specialized.

Before introducing such specialization, qualify exact correspondence between
the domain consumed by scalar lowering and the domain enforced at invocation,
including copied artifacts, changed dimensions, typed/raw values and rebinding.
Use the existing scalar lowerer's interval results and source overflow policy;
do not grow a second expression/type inference table. Keep generic source checks
ordered unless the admitted domain proves them total. Shape-contract-derived
domains can avoid needless point-template churn, but also require actual guarded
storage/shape premises.

The held draft now implements the bounded contract foundation: a rank-zero
integral AbstractValue may declare a closed `:scalar-domain` interval; invocation
materialization guards it before dependent evaluation, including raw and typed
inputs. The existing scalar lowerer can consume that actual guarded parameter's
interval and prove multiplication nonoverflowing; an otherwise identical generic
parameter retains trapping arithmetic. No second range-transfer implementation
was added. Source-free promotion preserves the contract. Invocation bindings
require the offered domain to imply the consumer's requirement; aliases cannot
manufacture tighter domains, and copied plans cannot weaken the numerical
program's guard. These are semantic constraints, not authenticated certificates.

Explicit caller-supplied public scalar domains now survive ordinary source
compilation and cached preparation through the same invocation guard. Input
constraints are combined by interval intersection with inferred facts, not
discarded because inference knows only the wider dtype. Domain declarations for
non-public, rebound or wrong-dtype source values are rejected. The typed source
route now carries the public parameter signature instead of dropping it. A real
map-helper compilation tests changed valid dimensions and rejection through a
cached template; this is not just a source-free IR control.

The held draft now selects same-sign, power-of-two input domains during modern
prepared compilation, clipped to the declared integral dtype; zero has its own
domain. Explicit caller contracts take precedence. The existing template identity
includes these contracts, and invocation still checks actual scalar values. The
shared scalar lowerer proves totality over the guarded domain and permits safe
dimension calculations to move ahead of device work without replacing scalars by
literals. Unknown helpers, loops without termination proofs and unproved checks
remain ordered. Generic compilation without concrete arguments gains no assumed
range. This is a preparation-policy change, not a change in scalar arithmetic.

The unchanged public LoRA/SFT witnesses now prepare without runtime allocation.
Independent tests cover bucket reuse, changed dimensions, explicit contracts,
malformed scalar rejection and finite-width extrema. This does not establish
MMM allocation savings, real-weight model training or performance acceptance.
Keep exceptional evaluation flags in early scalar passes: optimization uses the
exact scoped SSA region and its guarded input contracts, via the existing scalar
lowerer. Do not replace all parameter reads with literals (which can change
boxing/identity), or use a global name-to-range table that ignores lexical
rebinding. A successful range proof must remain tied to the domain actually
guarded before the dependent program executes.

### Qualification of this bounded vertical (2026-10-18)

The affected compiler/AD gate passed 541 tests and 7,293 assertions. Separate
public-source preparation controls passed 24 assertions, including automatic
domain reuse and explicit-contract rejection. Unchanged public LoRA SGD and SFT
head witnesses replayed three changed batches on both local OpenCL and Level Zero
against their existing independent numerical oracles (262 assertions). These are
small capability witnesses, not real-weight Gemma or frontier-training acceptance.

A fresh, bounded JVM passed eight raw-bit native forward/gradient controls without
realizing the source-only numerical receivers as JVM bytecode. This qualifies the
closed Double source projection, not arbitrary product/loop native gradients.
The MMM profiling report still shows approximately 74 KB/call for scalar AD;
this change has not established a reduction in that allocation or posterior time.
