# Normalized AD design proposal

Status: architecture assessed against Raster main
`642b5f0c0aad7c49f223357a796ce09c900e7c1c`, with an initial opt-in scalar implementation.
This is not a replacement for the compiler campaign or completion of normalized AD. The
unpublished scoped-helper experiment remains held by a checkpoint refusal regression.

The first implementation accepts homogeneous Float/Double scalar literals, aliases, binary
addition/multiplication, unary/binary subtraction and inactive scalar precision casts.
`jvp` and `value+grad` accept `:representation :normalized-scalar`; `:wrt` selects the
active parameter indices. The unchanged source route remains the default. Effects, helpers
without admitted identities, arrays, conditional/loop regions and active precision changes
refuse. Explicit `:compile? true` also refuses; compiler construction inside source deftms
retains its existing closed option gate rather than silently selecting another representation.

The implementation reuses TypedSOAC scalar projection and existing forward/transpose
templates. Coefficient residuals and program interfaces are checked, with compiler/template
revisions and the existing metadata-sensitive semantic fingerprint bound to the bundle.
This is structural producer validation, not a proof of user-supplied derivative rules.

Initial warm validation: 12 focused tests / 74 assertions and 69 affected tests / 525 assertions
pass. Independent review approved the production implementation at `5bf179dd`; the subsequent
test-only control adds subtraction and lexical aliases. Fresh CI remains a separate gate.

The next bounded implementation checks every registered scalar call against the existing
non-invoking dispatch selection. Argument tags and the declared return must agree with the
retained value contracts. A devirtualized receiver must be the actual selected implementation
object; semantic operation metadata alone cannot certify it. Promotions, unprojectable methods
and merely applicable parametric templates refuse in this homogeneous scalar vertical.
Implementation objects are compared transiently, not included in serialized fingerprints.
This tightens admission without creating another method/type registry or broadening coverage.
Focused public/IR controls pass 10 tests / 72 assertions; affected source AD/inlining controls
pass 69 tests / 525 assertions in the existing warm REPL. Independent review found no bounded
source blocker. Receiver aliases, changed receiver roots, direct calls and contradictory
selected signatures are explicit controls. Arbitrary host redefinitions outside compiler
registration are not comprehensively covered by the compiler revision contract.

Raster should differentiate a checked, unscheduled numerical program with explicit value,
control, conversion and effect contracts. Forward and reverse AD should consume the same
normalized program and tangent facts. Hardware scheduling and buffer realization should
remain downstream. This preserves the functional SOAC direction while removing AD's dependence
on repeated recognition of incidental source shapes.

## Current architecture and limits

The current entry points share `raster.ad.reverse/ad-prepare`. It projects constructors,
expands composites, normalizes bindings, lifts recognized loops, materializes maps and
canonicalizes proved fresh full-write regions. Reverse AD then builds records and emits
pullbacks; `raster.ad.jvp` performs a separate tangent fold over the prepared bindings.
Runtime and compiled reverse AD already share template emission. This is useful unification,
but a prepared S-expression is not yet a validated numerical AD program.

Existing authorities should be retained:

| Concern | Existing authority | Required consolidation |
| --- | --- | --- |
| Selected callee and intrinsic identity | `core/op-descriptor.clj`, retained dispatch evidence | Resolve once at the normalized boundary; lexical calls must not acquire a global rule by spelling. |
| Logical type and shape | `ir/abstract-value.clj`, walker and TypedClojure facts | Attach contracts to stable values; no fallback precision or operator-name inference in AD. |
| Functional parallel algorithms | `ir/soac-dialect.clj`, `typed-soac-frontend.clj` | Differentiate typed scalar regions and SOAC operations before scheduling. |
| Ordered equation spine | `ir/parallel-program.clj` | Reuse value/equation IDs and explicit operands/results, including the existing unscheduled TypedSOAC envelope. |
| Tangent typing | `ad/tangent.clj` | Distinguish unknown, excluded, nondifferentiable and typed zero internally. |
| Derivative rules | `ad/templates.clj` and existing structure facets | One rule lookup, with explicit forward, transpose and higher-order capabilities. |
| Activity | Production reverse activity and JVP environments; separate `ad/activity.clj` TypedExpr analysis | One analysis over normalized value IDs; retire migrated classifications rather than reactivate a second analysis authority. |
| Code emission | `compiler/ad/bind_ctx.clj`, `compiler/ad/flatten.clj` | Keep as emission adapters, not additional semantic authorities. |

There are important coverage limits. TypedSOAC does not yet represent every scalar call,
conditional, recurrence or host effect. `ParallelProgram` still retains source for host
reconstruction, but its operations need not be scheduled kernels: `typed-soac-route`
already builds checked unscheduled equations with `:operations [equation]` and a
TypedSOAC algorithm. The TypedSOAC validator rejects local SSA bindings inside a `scalar`
operation, despite the general region grammar; one scalar equation per SSA binding avoids
that restriction. The first AD vertical should reuse both existing representations without
validator extensions. Later region extensions need demonstrated requirements and must not
misuse host-only annotations or bypass existing checks.
The older `ir/core.clj` TypedExpr nodes are useful analysis infrastructure, but currently do
not uniformly retain every call's metadata and are not a complete SSA numerical program.

Two recent findings motivate consolidation. Excluded index arrays lost their constant status
through helper snapshots, hiding an existing gather transpose from the analyzer. A bounded
alias repair passed independent gradient controls. Separately, an inexact stored-carry loop
is correctly refused directly but not through the held helper expansion. A source-preservation
callback did not repair this: the original lexical store-call shape itself was unrecognized.
Neither success in one family nor metadata transport establishes completeness of the next.

## The normalized boundary

Use a semantic phase of the existing value/equation envelope and the existing pattern-friendly
typed scalar/SOAC regions. Add only missing semantic forms and validators. Do not introduce
a parallel type, function, alias or compiler-cache registry.

Normalization establishes:

1. Fresh logical value IDs, block parameters and explicit captured values. Rebinding creates
   a new value; source names are diagnostic information, not identity.
2. Canonical selected operations and typed arguments/results, including explicit conversions.
   The source's retained facts are imported and checked rather than reconstructed by AD.
3. Ordered scalar bindings and explicit conditional/loop regions as coverage is added.
   Argument evaluation, including unused throwing arguments, and inactive primal effects
   retain their original order and occur once. Exception ordering is observable semantics.
4. Logical array reads, writes and aliases. Proved fresh full-write regions may become
   functional results. Other mutation needs explicit state versions or remains unsupported.
5. Shape constraints, numerical policy, source locations and derivative-relevant obligations.
   Each subsequent rewrite must preserve, discharge or explicitly invalidate those facts.

Generated artifacts use the existing semantic fingerprint and callee/template revision
authorities for invalidation. Their identity includes the selected rules, `:wrt`, numerical
policy, normalized contracts and residual schema. No second AD cache or identity registry
should be added. A validator establishes structural properties and declared contracts; it
does not by itself prove that a supplied custom derivative is mathematically correct.

The boundary is independent of `:wrt`. Choosing differentiated inputs changes activity and
generated derivative work, not the primal program or its effects. An unknown type needed
for AD is an error; it must not become `NoTangent`. Opaque inactive work may remain on a
declared primal-only boundary without pretending it is a portable numerical operation.

Existing source-loop recognizers remain frontend importers during migration. They must
produce explicit recurrence facts or a structured coverage refusal. AD must not rerun them
after inlining or materialization to recover semantics lost by an earlier pass.

## Linearization and transposition

The map migration starts at its typed element region. `linearize-map-element` accepts one
validated pure functional map with one homogeneous plain Float/Double array and one result.
The element parameter is active; scalar captures are inactive. Ordered local definitions
retain their declared dtypes and are checked before scalar projection can repair a widening.
The importer uses the existing scalar program entry, derivative rules, bundle and validators;
it does not rewalk source or infer types from output storage. Known empty frontend storage
facets are admitted; effects, aliases, storage-backed outputs, unknown numerical facets,
mixed precision and index dependence refuse. Retained calls must already have canonical
identities, not unresolved source aliases.
The shared scalar call check rejects a callee owned by the current value environment before
global intrinsic/rule lookup, even if no source lexical marker was supplied. A scalar capture
named `double` therefore cannot acquire the global cast rule; a qualified core cast is distinct.

This entry returns a **scalar element bundle**, not a whole-array pullback. The next step is
lifting those ordinary primal/linear/transpose programs into checked map equations. Element
coefficients needed later must be captured into fresh owned residual arrays at primal
execution; a mutable input array is not a snapshot. Inactive scalar captures can remain scalar
residuals. Array transposition needs an explicit cotangent array and must not silently sum an
array output through `value+grad`. This slice does not close the training/model acceptance gate.

The next arithmetic slice admits binary division through the same operation descriptors and
templates. Generated nested rule terms are flattened by the existing AD ANF normalizer;
temporary contracts come from the rule's declared homogeneous scalar space and are checked
against operands and selected signatures, not inferred by another operator table. The
linear program permits division by a coefficient, never by a tangent-dependent denominator.
Transpose generation uses the existing rules and the same checked equation boundary. This
does not introduce reciprocal simplification, StrongZero, reassociation or a new rounding
policy. Public controls include analytic quotients, independent finite differences, JVP/VJP
pairing, excluded inputs and Float storage classes; forged tangent denominators refuse.
Focused public/IR controls pass 12 tests / 110 assertions, including signed-zero, NaN and
explicit-zero-seed exceptional arithmetic; the affected existing gates pass 69 / 525 in the
warm REPL. Independent bounded source review found no blocker. Array/control regions,
compiled normalized AD and the unchanged model-scale gates remain outside this acceptance.

The proposed static compilation result is a bundle of ordinary typed programs:

```text
typed source
    → checked normalized numerical program
    → primal program + residual requirements + linear tangent program
    → optional transpose of the linear program
    → shared SOAC optimization, scheduling and target emission

declared reverse-only rule
    → checked primal + residual requirements + explicit pullback program
    → shared optimization and emission, with no implied JVP or HVP capability
```

Executing the primal program produces `(y, residuals)`. Executing the linear program with
those residuals and an input tangent computes `J·dx`. Its checked transpose computes
`Jᵀ·dy`. The compile-time residual schema is distinct from the values retained at runtime.
For `x*x`, the linear program uses saved primal `x` as a coefficient; it must not multiply
two tangent inputs. Linearity validation tracks dependence on linear inputs separately
from coefficients and residuals.

This does not require immediately replacing every reverse rule. Existing custom pullbacks
can remain reverse-only capabilities. Linear and bilinear structure can derive appropriate
rules where justified. A general reverse rule does not automatically supply a forward rule,
and a first-order rule does not establish higher-order support.

Maps, gathers, contractions and linearized recurrence regions remain ordinary composable
operations. For example, gather transposes to accumulating scatter: repeated indices must
sum contributions. A unique store is legal only with an independently checked uniqueness
fact. Function application exposes a normalized body or an explicit rule boundary, rather
than silently switching to an unvalidated closure fallback.

The local JAX implementation supplies a useful precedent: `linearize_jaxpr` separates
primal/residual production from a tangent program, and transpose lookup refuses missing
rules. Raster should borrow this separation, not copy JAX's registry layout or replace its
SOAC algebra. See `../jax/jax/_src/interpreters/ad.py`, functions `linearize_jaxpr`,
`linearize` and `get_primitive_transpose`.

## Tangent and activity contracts

Internally distinguish four states, even if the public API continues returning `nil` for
excluded gradient slots:

- No tangent space: discrete values such as indices and shapes.
- Excluded input: a differentiable value outside the chosen `:wrt` set.
- Typed zero contribution: an additive identity in a known scalar, array or product space.
- Unsupported or unknown derivative: a construction error, never a zero contribution.

Tangent contracts derive from primal dtype, shape and supported structure. Projection and
accumulation use `ad/tangent.clj` as the starting authority and extend it deliberately for
product or complex spaces. Missing type facts cannot silently select Double. Internally,
known nondifferentiable types must be separated from the unknown tags currently classified
together by `tangent-kind`.

Activity is a checked dataflow analysis over stable IDs, including block/carry fixed points
and output usefulness when those regions are supported. Constant-source aliases share the
source's activity. A zero allocation's initial value may be constant, but later active writes
produce a different array state; allocation freshness does not make that state constant. Primal effects
are not derivative contributions and cannot be deleted just because their result is inactive.

## Residuals and memory

Residual requirements state the producing value or state version, dtype/shape, consumers,
and whether the value may be borrowed, saved, or recomputed. Branch choices and loop iteration
information are explicit residuals when required. Physical buffers, donation, event completion
and release use the existing memory/runtime planning contracts after semantic AD validation.

Allocation identity is not a mutation version. Replaying a read from an array requires either
the same unmodified contents, a saved version, or a checked recomputation. Until versions are
represented, the existing overwritten-replay refusal must remain. Ownership alone cannot
justify replay or erase a write.

For a recurrence, distinguish the carried value, the value sent to the store, and the stored
result. For example:

```text
carry_next : double
store_value = double(float(carry_next))
checkpoint[i] = store_value
```

The checkpoint is not an exact copy of `carry_next`, even though its storage is Double.
The planner must retain the required carry separately, recompute it legally, or refuse the
requested derivative. This decision follows explicit conversion/value identity, not the name
of a lexical function or a dtype check on the destination. A generic loop tape also needs
typed residuals and resource bounds; data-dependent trip counts are a separate capability,
not automatically admitted by this proposal.

## Numerical semantics and higher order

AD differentiates a declared numerical model of the operations, not the literal discontinuous
map implemented by all floating-point bit patterns. Conversion, quantization, exceptional
values and nonsmooth operations need explicit derivative policy. Straight-through estimators
and custom subgradients are opt-in rules, not consequences of integer or packed storage.
An exact copy and a precision-changing checkpoint remain distinct under any such policy.

Mathematical linearity does not authorize changing floating association, overflow behavior,
FMA contraction or library realization. The existing numerical contract must constrain both
primal and derivative schedules. Cotangent addition is mathematically associative, but its
floating implementation is not bitwise associative. Adjoint pairing tests therefore use
declared tolerances and operation policies, not a universal exact-equality claim.

Higher-order AD requires closure of the generated programs under the requested transform.
Residual production must remain differentiable where the outer derivative depends on it;
freezing all residuals would lose Hessian terms. An opaque pullback closure or unrepresented
tape is a capability refusal, not evidence of a zero Hessian-vector product. Keep current
HVP refusals until typed closure is demonstrated.

## Migration and acceptance gates

| Step | Deliverable | Gate and retirement |
| --- | --- | --- |
| 1 | A validated normalized scalar/call boundary using existing IDs and contracts | Direct/helper equivalence, exact-once evaluation, selected-call identity, missing-type refusals and explicit conversions. No source rule is retired yet. |
| 2 | Scalar primal/residual/linear bundle plus checked transpose | Analytic and FD gradients, JVP/VJP pairing, excluded versus zero slots, branch semantics and supported higher-order scalar checks. |
| 3 | Typed maps, gathers, reductions and contractions through the same boundary | Repeated indices, zero extents, mixed storage/computation precision, masked weighted NLL, inactive effects and overwritten-replay refusals. Retire only the migrated source classifications. |
| 4 | Ordered recurrences with explicit carry and residual requirements | Helper/direct checkpoint refusal parity, exact carry witnesses, zero-trip behavior, supported initializer/scan shapes and CPU/device differential tests. Unsupported general loops remain refused. |
| 5 | Shared runtime and compiled AD adapters, followed by cleanup | Public entry points use the same validated bundle; cold CI, cache invalidation, unchanged real-model gradients and native ownership checks. Remove duplicate preparation/activity/tape logic only after equivalent coverage. |

Extend the existing numerical oracles rather than replacing them with expected answers emitted
by the compiler. A law harness can share fixture mechanics while retaining analytic answers,
finite differences, transpose pairing and backend comparisons as independent evidence. Keep
zero-domain negative cases: unsupported code must not become accepted merely because a loop
happens not to execute.

The unchanged real Gemma training case still has 11 failing adapter checks out of 28. The
exact generated SFT objective, Gaussian/logistic density composition, city kernel primal
parity and PDE rollout gradients are complementary gates, not substitutes for that result.
Microtests passing does not establish model-scale correctness or competitive performance.

## Relationship to the compiler campaign

This is consolidation of the semantic numerical layer, not a new backend project. JVM,
native and accelerator compilation consume the same generated numerical programs. AD can
initially precede performance fusion, leaving existing fusion to optimize both primal and
derivative programs. Differentiating an already fused region later requires an explicit
semantic equivalence contract; fusion is not automatically transparent under every numerical
policy.

Memory planning can schedule residual retention and recomputation using the same lifetime
and resource contracts as other values. Distributed placement, communication transposes and
checkpoint costs can follow without teaching basic AD about KV caches or a particular
cluster. Probabilistic inversion remains a separate capability at the spindel boundary;
a shared block/residual description does not equate Bayesian conditioning with transposition.

The proposed first implementation branch is one straight-line scalar vertical: ordered
bindings, selected declared calls, explicit conversions, known scalar tangent spaces,
residual coefficients and checked transpose. It reuses the existing unscheduled envelope
and represents each scalar SSA binding as its own equation; no validator relaxation is
needed for this scope. Conditional/loop
analysis, product/complex spaces, fresh-array helper support and recurrence tapes follow
in later steps.

For that first branch, the held map-helper and checkpoint cases are coverage and refusal
controls, not promises of successful differentiation by a scalar-only route. Known legal
existing paths remain during migration; an unsupported active region must receive an honest
diagnostic rather than a guessed zero or shape-dependent acceptance. Public AD APIs and
declared numerical policies remain unchanged. No cloud hardware is required for the design
or its initial acceptance gates.

### First landing acceptance boundary

The first route admits known homogeneous Float or Double literals, aliases and selected
scalar addition, subtraction and multiplication through the existing operation descriptors
and derivative templates. It produces checked primal, linear and transpose programs with
explicit coefficient residuals. A linear multiplication may depend on tangent inputs on
at most one side. Unknown types, lexical callees without admitted rules, effects and active
narrowing without a declared derivative policy must refuse explicitly. A reverse-only rule
does not acquire forward or higher-order support through this route.

An opt-in route through the existing public `jvp` and `value+grad` APIs must exercise this
bundle; a standalone IR prototype is not a migration gate. The default source route stays
until the corresponding compatibility gates pass. Tests cover analytic derivatives, finite
differences, JVP/VJP pairing, repeated uses, typed zero and excluded slots, rule identity,
cache revisions, forged nonlinear tangent programs and stale residual interfaces. Effectful
programs initially have a refusal gate, not a claim of preserved execution order. The held
helper/checkpoint refusal remains unchanged.

### Reference review: mechanisms to adopt, not new dependencies

The local reference check supports this boundary and adds specific obligations:

- JAX separates primal and tangent traces and derives residual interfaces in
  `jax/_src/interpreters/ad.py` (`_linearize_jaxpr`). Its `bilinear_transpose`
  requires exactly one undefined primal operand. Our linear validator must track
  tangent dependence through aliases and reject tangent-by-tangent multiplication;
  selecting a multiplication intrinsic alone is not a linearity certificate.
- Enzyme's `lib/EnzymeCore/src/rules.jl` makes overwritten arguments explicit in
  reverse configuration and carries an augmented return with a tape. PyTorch's
  `torch/csrc/autograd/saved_variable.cpp` checks saved content versions. Future
  borrow/recompute plans must prove preservation of the needed contents, not just
  preservation of a buffer handle or an arena lifetime.
- Enzyme exposes `StrongZero` as a separate policy in
  `lib/EnzymeCore/src/EnzymeCore.jl`. A typed zero, an absent contribution and
  evaluation of `0 * Inf` are not interchangeable under IEEE arithmetic. The
  normalized route must not introduce strong-zero or fast-math semantics through
  activity analysis or dead-code elimination. Exceptional values and signed-zero
  checks accompany ordinary finite-difference tests.
- Futhark's `src/Futhark/AD/Rev/Map.hs` distinguishes captured-array adjoints;
  `src/Futhark/AD/Rev/Loop.hs` saves actual loop-carried values and copies consumed
  arrays for reversal. Those are models for the later SOAC/tape vertical, not
  evidence that a scalar-first route supports general loops.
- Triton's `python/tutorials/05-layer-norm.py` integrates hand-authored forward
  and backward kernels through an autograd boundary. This is a useful performance
  reference, but not a substitute for general semantic AD in Raster.

These observations refer to the available sibling checkouts, not an assertion about
the latest upstream versions. No reference compiler becomes a runtime dependency.
The unchanged Gemma, SFT and held checkpoint gates remain independent: successful
scalar normalization does not certify them.
