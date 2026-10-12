# Normalized AD design proposal

## Current implementation boundary

Normalized AD remains opt-in; the source AD engine remains the default and independent oracle.
The checked scalar lane supplies derivation evidence to three bounded host families: pure map,
map followed by ordered additive sum, and functional indexed map. Those families now execute
through one canonical program/interface/residual owner. Preparation retains full validation;
hot calls check compiler/rule epochs and captured executable identities without replaying proofs.
Host projection selects from actual TypedSOAC operations and declared operand roles, rather
than AD derivation-family labels. Its bounded map, map→reduce and indexed-store admission
checks remain separate from derivative validation; arbitrary equation chains still refuse.
Indexed pullbacks additionally require private index snapshots and a fresh typed-zero physical
gradient owner on every call. These are sequential lifetime contracts, not concurrent mutation
or asynchronous ownership guarantees.

Checked two-equation pure-spine append now builds the admitted map→ordered-sum programs;
it is not yet a general differentiation rule for arbitrary equation chains.
General equation composition, conditional/loop AD regions, device execution of these normalized
families and default-engine migration remain separate gates. Existing real-weight training
acceptance is still open. The sections below describe the individual construction checkpoints;
an adapter's lack of runtime effects does not mean its checked envelope is unused by the runtime.

## Equation projection and shape facts

`soac/equation-subprogram(s)` keeps executable inputs distinct from shape-only facts. When an
input, logical result or physical destination has a direct symbolic dimension or canonical
`(value id)` dimension, projection retains the already-declared AbstractValue for that ID
verbatim. Thus projecting indexed `data[m]` retains `m` without making it a kernel argument.
Missing facts stay missing; contradictory dtypes or nonscalar facts remain available for the
consuming authority to reject. Projection does not infer types, synthesize scalar facts or
add equations that compute dimensions.

This is a bounded direct projection, not general recursive shape closure. Opaque `extent` and
`unknown-dimension` wrappers and compound shape algebra remain unchanged. General dependency
closure must extend the existing dimension projection authority, not collect symbols from
arbitrary S-expressions or introduce a second shape/type vocabulary. Public surface semantics,
floating-point policy and AD operation coverage do not change.

## Canonical program-envelope pilot

`normalized-program-bundle/adapt-indexed-map` also projects the already-checked functional
indexed-map programs, without changing importer or runtime admission. Its one active data
array has independent capacity from the excluded Int/Long index array and output extent.
The owned index snapshot is identified by its exact primal producer and excluded source-input
origin, not as a scalar derivative coefficient. Shape-only capacity AV facts remain in the
programs without becoming executable inputs.

Unlike pure map/sum envelopes, indexed envelopes have a physical binding: the exact required
owned/exclusive typed-zero gradient contract, retained destination AV and transpose
destination/result alias. This is an unsatisfied allocation/initialization requirement,
**not** a receipt or a claim that the effect equation allocates storage. The cotangent result
interface describes the data-shaped gradient even though the logical host effect binding
is nil; existing host materialization returns the explicitly supplied physical owner.
Sequential ordered updates, original Float checkpoints and source/rule/Namespace validation
are unchanged. Map/sum mixtures refuse. No canonical indexed runtime migration, device
execution, new derivative rule or equation composition is enabled by this adapter.

`normalized-program-bundle/adapt-map` projects exactly one already-admitted homogeneous map
into a canonical envelope: top-level source/primal/linear/transpose fields are actual TypedSOAC
programs, never scalar lane programs with changed field meanings. Ordered retained-ID/AV
interfaces identify the explicit tangent and cotangent spaces. Residual bindings identify their
exact source/scalar origins and primal equation/result producers; direct scalar captures remain
immutable inputs, while coefficient arrays retain owned snapshot AVs. Pure maps have no physical
binding obligations. The existing scalar/map bundle is explicitly nested derivation evidence.

Validation still uses the existing scalar/rule/type/call and fixed-ID map-lifting authority,
then checks exact canonical projection correspondence and a process-independent semantic
certificate. Only the two declared runtime Namespace fields are projected to namespace names,
after their actual Namespace identity has been checked; opaque unsupported values are not
silently sanitized. Existing rule/definition epochs and all physical AV facets remain bound.
Host execution tests use existing materialization; non-host placement still refuses there.
The existing authority checks scalar linearity, residual interfaces and source/lift relations,
and regenerates registered scalar derivative arithmetic using the recorded typed generation
stream described below. This establishes correspondence to the currently registered rules,
not independent proof of custom-rule mathematics or a capability granted by producer certification.

This thin migration adapter adds no runtime mode, parser, rule registry, cache, equation
composition or fusion permission. The adapter itself does not change source admission or public
surface semantics. The nested evidence
and family validators may retire only after a common program validator independently owns the
same source/derivative/residual obligations and the existing numerical/refusal gates agree.
`adapt-map-sum` uses the same canonical projection for an existing admitted map→ordered sum.
The original two-equation source and generated programs stay exact. Tangent input remains an
array, while primal/tangent results and the explicit cotangent seed have the retained scalar
Float/Double space. Residual producer bindings still name the original tuple-map outputs;
physical obligations remain empty. The existing sum authority retains the initializer, actual
step, dtype and operand/iteration order; algebra metadata grants no reassociation permission.
No sum runtime selector, broadcast seed array, fusion or new derivative rule is added. Indexed
canonical migration, further equation composition, exact SFT and the unchanged FAILED Gemma
gate (11/28) remain open.

## Scalar derivation correspondence

Scalar bundles retain an ordered pure generation stream, including tangent inputs, cotangent
seed and every template/ANF temporary. The existing typed fresh owner creates these IDs.
Validation rejects missing evidence, malformed requests or metadata, source collisions,
duplicates and incomplete/excess replay; it never allocates replacement IDs or alpha-normalizes
an untrusted derivative. Existing family and certificate refusals retain their ordering.

After those checks, validation regenerates the linear program from the retained primal and
active interface, then transposes that regenerated linear program using current registered
rules. Full typed programs, tangent/seed IDs and residual ordering must agree semantically.
Rule/definition epochs are checked before and after replay. Existing bundles without generation
evidence refuse rather than receiving a new certificate or inferred proof.

Every explicit validation replays rule emitters; there is no opaque validation cache or safety
skip. Prepared host family execution instead uses the captured lifetime fence described above,
without invoking full validation on each primal or pullback call. Emitters must remain
deterministic under their registered contract, and their original failures propagate. Agreement
with current registered derivative arithmetic is not independent proof of its mathematics,
nor floating reassociation permission or completion of the AD/training migration. Exact SFT
and the unchanged Gemma FAILED gate (11/28) remain independent obligations.

Forward-rule lookup now exposes one snapshot-based owner query shared by JVP derivation.
Structureless copied aliases follow the existing canonical fallback; an alias's explicit
JVP or structure facet stays authoritative. Derived callbacks are cached at that resolved
owner only after a successful registry compare-and-set; replacement forces a fresh lookup.
The query supplies availability and structure, not purity, scalar signature or numerical
proof. Nonlinear scalar admission still needs those independent facts and tangent-linearity
checks.

The homogeneous scalar lane now admits unary elementwise operations when that owner declares
exact unary elementwise structure and supplies reverse rules, the canonical scalar descriptor
recognizes the operation, and existing effect analysis proves purity. Selected method signatures
must agree with retained Float/Double facts. Java Math's Double results cannot acquire Float
semantics merely from caller annotations. There is no second math whitelist or rule registry.
Nonlinear operations may compute frozen coefficients, but may not consume tangents in a linear
program; even `exp(0)` is a coefficient, not an identically zero linear map. Rule replacement
invalidates old bundles and replay uses the new declared rule identity.

Exp/log host-map witnesses check both carriers, independent finite differences, JVP/VJP pairing
and snapshot ownership. This is not general surface math coverage: bodies requiring unsupported
mixed-precision conversions can still decline. It changes neither default source AD, device
execution, reassociation permission nor the unchanged model-scale acceptance gate.

Actual `deftm` Gaussian observation scores (Double) and binary-logistic scores (Float/Double)
now exercise this lane through the frontend's two-equation map→ordered-sum program, canonical
envelope projection, and prepared host execution. The prediction array is the single active
space; observations and the Gaussian scale are excluded. Empty and small inputs check ordinary
primal equality, independent density/probability-domain answers, analytic gradients, finite
differences, JVP/VJP pairing, unchanged input arrays and saved pullbacks after caller mutation.
These are existing-program composition witnesses, not a nonlinear-loss importer or new rules.

The witness does not differentiate a scalar Gaussian location/scale parameter, a vector-latent
regression program or distribution records. Active scalar captures need a separately checked
broadcast tangent and ordered reduction of lane cotangents. Binary logistic likelihood is not
categorical SFT: grouped log-sum-exp, target gathers and the unchanged real model gate remain
open. No family validator or construction evidence retires on the strength of this witness.

## Indexed-map construction checkpoint

`linearize-indexed-map-program` constructs ordinary SOAC programs in the existing certified
bundle's `:gather-programs` facet. The admitted source is a functional indexed map: one excluded
Int/Long pointwise index array and one stable, indirectly read Float/Double data array, with exact
canonical read operands and the actual directly resolved core `aget` Var, independent
data/output extents, unchanged indices and no destination
alias, narrowing or effects. Imperative `par/gather` caller destination storage is not erased.
Retained `.invk` reads are refused until selected executable receiver identity is proved;
semantic operation metadata alone cannot authorize their derivative.
The existing scalar identity linearizer and rule/definition revisions remain authoritative.

Primal explicitly exports a fresh owned index snapshot; linear gathers the explicit data tangent
through it. Transpose is a sequential effect-map with ordered read/add/write checkpoints. Its
`:required-zero-gradient` contract requires a fresh, exclusive, typed-zero data-length physical binding;
the effect equation does **not** allocate or initialize this destination. Construction validates
source/interface/program correspondences, retained physical facts and the owned snapshot.
The opt-in host consumer now reuses shared source realization for these programs. Its distinct
indexed projector admits the mixed integral snapshot and stable array capture without widening
the existing homogeneous pure-map projector. Ordered effects keep their nil logical bindings;
the host result is the validated physical destination supplied separately from semantic inputs.
The runtime checks every original-width index before primal allocation, ties shape-only capacity
to the actual data array length, and saves only a private index snapshot and immutable extent
facts. Every pullback allocates a new typed-zero data-length gradient and uses the sequential
ordered write checkpoints, including empty output extents. Caller arrays cannot be concurrently
mutated during acquisition. Device execution, gather→sum and public-engine integration remain
separate gates. The unchanged real-weight Gemma gate remains FAILED (11/28); host indexed-map
controls are not model or exact SFT acceptance.

Initial assessment checkpoint: Raster main
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
validated pure functional map with one or two homogeneous plain Float/Double arrays and one
result. Exactly one element parameter is active; scalar captures and any other array are
inactive. Two arrays require explicit selection by source value ID. Ordered local definitions
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

`linearize-map-program` adds checked `:map-programs` to that same scalar bundle: the original
map, a tuple-map primal, and explicitly seeded linear/transpose maps. The tuple primal exports
fresh owned arrays for coefficients other than direct scalar captures. Invariant locals are
conservatively snapshotted per lane too; this does not introduce another activity analysis.
Derivative maps consume those arrays rather than recomputing coefficients or rereading the
original input. Direct scalar coefficients keep their existing capture identities.

Validation checks the source-to-scalar relation and reconstructs only the map lifting with
fixed interface IDs. Metadata-sensitive semantic equality checks local definitions, yields,
shape, ownership and coefficient correspondence before the existing bundle certificate. Map
extents retain exact value IDs, including compound IDs, with a retained integral scalar
contract. This is program construction, not yet physical allocation or execution: the runtime
adapter still has to demonstrate fresh buffers, mutation-safe pullbacks and lifecycle cleanup.
This first lifting requires matching input/output memory-space, placement and sharding, and
an unspecified or owned functional result. It preserves those constraints rather than silently
turning an external output into owned storage. Cross-placement derivatives remain a later plan.

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
Actual `deftm` maps enter through the existing frontend. The normalized importer
admits its bookkeeping only when it describes the sole map/result binding and
has no host activity or allocation; unknown/inconsistent facets still refuse.
Homogeneous identity conversions use the shared metadata-preserving canonical
conversion projection and existing ANF normalization. Deterministic local IDs
derived from the scalar output identity let relational validation repeat that
projection without fresh-name or printed-source equality. The original frontend
program stays in the bundle; its facts are not silently stripped.

Host materialization of the lifted maps reuses `typed-soac-route`'s production
allocation, ordered scalar region and tuple-store projection. It returns ordinary
retained source with exact logical input/output IDs transported to fresh source
symbols, not another array interpreter. Float/double JVM execution controls check
fresh coefficient arrays, source mutation after the primal, scalar captures and
empty extents. Device placement, stored/aliased results and external output
ownership are explicit refusals at this host-only boundary. Source projection
alone is not a whole-array public AD API or a device gate.

The opt-in `raster.ad.normalized-map-runtime/prepare` consumer accepts an already
validated map bundle and constructs the existing three host projections once.
Its `:inputs` vector defines the argument order. `:vjp` takes that ordered argument
vector and returns `[output pullback]`; the pullback takes an explicit cotangent
array and returns positional cotangents, with nil for excluded arrays, inactive scalar captures
and extent inputs. `:jvp` takes the argument vector and an explicit tangent array
and returns `[output tangent]`. Array outputs never imply a scalar seed.

Exact primitive storage, scalar capture types, array lengths and representable
nonnegative extents are checked before execution. Coefficient arrays remain
private primal-time snapshots; neither original input array nor the mutable primal
output is read by the pullback. Each derivative result is fresh. The caller must
prevent concurrent mutation during primal acquisition; snapshot ownership does
not promise atomic acquisition. Map and ordered-sum preparation now captures a
private canonical program owner. Full producer/rule validation runs during preparation
(the migration adapter currently validates twice, followed by one captured-callee proof).
Hot primal/JVP/pullback calls use existing compiler/template epochs and effective
executable Var identity checks before and after execution, rather than repeating
full validation or materialization. Dynamic Var bindings are part of effective identity;
Preparation realizes pending typed callees through the existing first-call JIT owner,
without invoking their numerical bodies, before capturing identity. The generated AD
runtime dependency is loaded before bundle epochs are established. The post-capture
proof still checks the realized selected implementation; a later replacement refuses.
persistent alias root replacement therefore refuses even without an epoch change.
This is a sequential fence, not a concurrent raw Var mutation/ABA or callable-internal
mutation guarantee. Indexed preparation uses the same fence, capturing the exact original
read/alias from its checked single-read, empty-local source lambda as well as the three
generated host projections. The scalar identity lane alone is not source dependency evidence.
Full canonical validation after capture binds that fence to the accepted program and physical
contracts. Indexed hot calls perform no proof replay or materialization; exact carrier/bounds,
snapshot and fresh-zero ownership checks remain. Original projection failures stay primary.
Extending indexed source admission beyond this lambda requires revisiting dependency capture.
Its physical typed-zero binding is an obligation checked against the canonical
transpose destination/result alias, not an allocation receipt.
Full scalar validation replays the existing registered derivative rules with the exact
recorded typed IDs. Within each proof, a request-local collector retains the first resolved
operation/receiver Var, effective value and metadata actually used by the selected-call
checks. Later disagreements cannot replace that observation. After the original validation,
certificate and derivation checks succeed, the proof rechecks those identities; earlier
refusals and callback exceptions retain precedence. Execution capture shares the same
first-observation comparison, without a serialized receipt or validation cache.
This strengthens each proof's sequential lifetime only: it does not span separate proof
and projection phases, establish concurrent/ABA safety, or independently prove custom-rule
mathematics. Indexed direct-read admission is not covered by this scalar observation
collector; its original read remains a dependency of the prepared execution guard.
Both adapter proofs and the post-capture proof remain required. This adds no cache,
GPU session or derivative rule owner. General loops, public `deftm` convenience,
higher-order AD of runtime closures and asynchronous residual ownership remain
separate capability gates.

Public `value+grad`/`jvp` runtime closures with `:representation :normalized-scalar`
use the same effective-callee and epoch fence around compilation and invocation.
Full validation stays construction-time, and existing source mode, `:wrt`, carrier
conventions and deftm metadata remain unchanged. Consumers which extract or compile
the exposed walked-body metadata bypass these runtime closures and are **not**
claimed to inherit their guards. The sequential identity/ABA limitations above apply.

The unchanged Gemma, SFT and held checkpoint gates remain independent: successful
scalar normalization does not certify them.

`linearize-map-sum-program` is the next program-construction checkpoint. It accepts
exactly a checked functional map followed by one scalar Float/Double additive sum.
The reduction must read the exact map result over the same extent, have a positive
typed zero initializer, and its actual homogeneous selected step must add precisely
the accumulator and element. Its retained algebra certificate is independently
rederived from that step; it is not reassociation consent. Original addition operand
order, initializer, dtype and the original reduction equation remain in the primal.
The linear program reduces the tangent map with the same step and zero; the transpose
map receives an explicit scalar seed as a capture, without a broadcast array or
implicit cotangent summation. Identity maps may need no residual arrays.

The existing bundle carries these ordinary SOAC programs under `:sum-programs`.
Validation rechecks the original two-equation source, the map prefix relationship,
fresh interfaces and construction correspondence, including semantic metadata and
owned residuals. Known frontend binding bookkeeping is projected only after its
complete two-binding envelope and absent host activity have been checked; unknown
attributes are not stripped. No derivative rules or numerical interpreter are added.
The host consumer adapts admitted legacy evidence once at its preparation boundary. All
three supported families then execute the canonical primal/linear/transpose programs,
ordered input/value interfaces, owned residual bindings and physical requirements. It
does not rebuild a separate runtime interface or execute a legacy program facet. Family
source/derivative validators remain the mathematical authority; they cannot be retired
merely because execution no longer reads their storage representation. Indexed destination
and result IDs remain distinct, and every pullback allocates a fresh typed-zero destination.
No caller array is retained as a residual. This does not add a runtime mode or general AD
composition.

Construction uses `soac/append-pure-spines` instead of a family-local right-biased fact
merge. The consumer→producer boundary is explicit, including identity mappings. Both
inputs are validated single-equation pure programs without aliases or physical result
bindings. Shared AbstractValues and provenance must agree exactly; equation IDs remain
disjoint, remaining program contracts must match, and ordinary validation checks the
combined definition/use scope and caller-supplied output order. This operation neither
infers ownership nor grants reassociation. The AD adapter separately proves its existing
functional source-result projection from unspecified/owned to owned before composition;
borrowed or external results refuse. Source facts remain unchanged. Unknown program
facets cannot be silently dropped from the consumer. Opaque extension contracts are
preserved but cannot contain value IDs requiring remapping: such schemas need an explicit
dialect remapping rule first.

Shared
host materialization assembles the tuple map and original reduction in equation order,
with one exact-ID source transport; it adds no numerical evaluator or reduction loop.
The generated reduction projects only the exact frontend prefix bookkeeping already
checked against the retained original source. Its step, dtype, identity and numerical
facets are unchanged. Physical placement and unknown use-side facets still refuse.
Matching retained Float/Double scalar JVM local hints are omitted only in this host
projection: Clojure's local carrier need not equal the retained Float operation dtype.
Semantic dtype, selected implementation receivers and primitive array hints are preserved;
this neither widens the selected arithmetic nor changes its rounding checkpoints.

For a sum, `:vjp` returns a scalar output and a pullback taking an explicit scalar seed
of exactly the retained Float/Double type; `:jvp` still takes a tangent array but returns
a scalar tangent. Positional excluded cotangents stay nil. The private residual arrays
have the same primal-time snapshot ownership as the map consumer, including required
excluded-array elements. Scalar outputs and seeds are immutable; returned gradients
are fresh and may be mutated without changing later pullbacks. Caller data cannot be
mutated concurrently with primal acquisition. Empty extents return the original typed
zero, and identity transposes use the existing extent/capture map with no fake array.
Independent ordinary-primal finite differences and strict ordered-cancellation checks
are separate from producer certification; no floating reassociation is implied.
Nonzero or
negative-zero initialization, arbitrary folds, transforms, stored results, effects,
mixed precision and reassociated/device execution remain separate gates. The unchanged
real-weight Gemma gate remains FAILED (11/28 adapter components), and exact SFT remains
an independent acceptance obligation.

The bounded map region also admits two same-shape homogeneous arrays when the caller
explicitly selects one source value ID with `{:active-array id}`. The two-argument
one-array entry remains unchanged; two arrays without a selection refuse. Existing
SOAC parameter layout identifies element parameters separately from inactive scalar
captures. An excluded element needed as a coefficient is saved into a fresh residual
array, not borrowed from its mutable source. Runtime validates every source array and
retains none of them in the pullback. Only the selected positional slot has a cotangent;
the excluded array, scalar captures and extent retain nil slots. Multiple active arrays,
reductions and general map composition remain separate capability gates. The unchanged
real Gemma gate remains FAILED (11/28 adapter components); this map witness is not a
replacement objective or model acceptance result.

## Operation-driven primitive host projection

The primitive host projection has one operation-driven entry and one source assembly owner.
Specialized map/map-reduction/indexed APIs keep their existing strict admission contracts.
Shared plain-host AbstractValue checks are factored once; numerical order, fresh outputs,
stable array captures and caller-owned indexed transpose destinations are still checked at
their respective operation boundaries. This does not generalize the accepted topology or
derive AD rules from mere structural source realizability.
