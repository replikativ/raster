# Normalized AD design proposal

## Current implementation boundary

Normalized AD is opt-in. The source AD engine remains the default structural reference;
shared derivative rules mean it is not an independent calculus oracle.
The current public boundary is deliberately smaller than general program differentiation:

| Canonical family | Host execution | Opt-in resident device execution | Preparation and call contract |
| --- | --- | --- | --- |
| Pure map, supported fused homogeneous region, or bounded single-use map spine | Canonical primal/JVP/VJP with retained residuals and ordered multiple active arrays | `normalized-device/prepare-owner` (one boundary array, one active input; multi-node execution refuses) | Concrete array geometry; array cotangent; exact residual dependency leases |
| Map followed by one ordered additive sum | Canonical primal/JVP/VJP with a scalar result and ordered multiple active arrays, or one active scalar capture | `normalized-device/prepare-sum-owner` (one active array) | Concrete array geometry; one exactly typed scalar cotangent fixed at preparation |
| Functional indexed map | Canonical gather primal and sequential ordered transpose | `normalized-device/prepare-indexed-owner` | Static index/cotangent extent; data/gradient extent static or one private `m` fixed from the prepared data array; array cotangent |

All three device surfaces reuse the generic named-entry owner, resident binder, LinkPlan/LinkModule
validation, epoch/callee guards and leases. They do not enable arbitrary equation DAGs,
conditional/loop AD, empty native buffers, per-call geometry changes, concurrent mutation,
host readback, or default-engine activation. Real Gemma acceptance, full GSDM statistical training
and Adam/EMA qualification remain independent gates.

`linearize-map-program`, `linearize-map-chain-program` and `linearize-map-sum-program` accept an ordered, nonempty, distinct
`:active-inputs` vector of exact public source array IDs, mutually exclusive with the existing singleton
activity options. This order determines tangent and internal gradient ports, independently of the
source argument order. Host JVP retains the existing primitive-array argument for one active input;
with multiple active inputs it requires an exact vector of tangent arrays in activity order.
VJP still returns cotangents in source argument order, with nil slots for inactive inputs. Saved
coefficients are privately owned snapshots, so delayed pullbacks do not read subsequently mutated
caller arrays. Supported closed map regions use the existing fusion pass to derive their checked
single-map execution stage; a previously unsupported spine with more than two boundary arrays
instead composes checked equation-local scalar derivatives. Repeated selected external arrays
share one public tangent port and receive explicitly ordered cotangent contributions. No separate
region derivative rule is added. A sum retains the exact
original ordered reduction for its primal and tangent; its scalar seed feeds the existing
per-lane transpose, which returns an ordered gradient array for each selected array. Singleton
scalar-capture differentiation still appends the original ordered reduction to the gradient
contributions. Multiple-active indexed operations and resident device execution remain explicit refusals; this is not general
node-wise DAG or fan-out transposition.

The opt-in canonical interface uses only plural `:tangent-inputs` and `:cotangent-results`
descriptors, related in order to `:active-inputs`; singleton owners use length-one vectors rather
than duplicate singular aliases. This is an internal experimental AD schema migration, not a
change to surface `deftm` semantics or activation of the normalized engine by default.

Host execution consumes each canonical port's own AbstractValue for source arguments,
primal/JVP results, cotangent seeds, gradients and owned residuals. Its current shape boundary
is scalar or rank one, with literal dimensions or direct retained Int/Long dimension values;
this is not a general dynamic-shape evaluator. Nonindexed execution no longer reconstructs
tensor roles or a common buffer extent from map operands. Delayed pullbacks retain only
required immutable residual inputs, direct shape controls and freshly owned residual arrays.
The indexed adapter separately retains its original-width bounds checks, distinct output
extent and data capacity, private index snapshot and exclusive typed-zero gradient owner.
Device adapters share exact canonical port association while keeping their existing family
capability limits and materialized resident geometry.

`linearize-equation-program` delegates pure maps to the existing map entry and additionally
admits one bounded bilinear segmented reduction on the host. Its two external operands and
result are flat plain Float/Double arrays, with positive literal axes and an explicit
order-preserving result regrouping. Exactly one operand is selected with `:active-input`.
The existing dense-product authority proves additive zero, product and layout; normalized
scalar AD differentiates the full accumulator-plus-product step. A checked AxisMap fiber
then determines the JVP/VJP reductions, without a separate derivative rule table. The passive
operand is privately copied for delayed pullbacks. The product may feed a unary, single-use
map spine through the existing checked node composers. Each activation retains its own owned
coefficient snapshots and effective arithmetic policy; shared scalar captures remain inactive.
The original mixed source has an empty program arithmetic scope with an explicit BLAS product
contract. Shared composition preserves every operation's effective policy rather than rewriting
the scope to force admission. Generated derivative phases use existing TypedSOAC operations and
host materialization. Batched/dynamic/empty contractions, maps before the product, fan-out,
multiple products, multiple active operands and resident device execution remain refused; the map-only entry
and default source AD engine are unchanged. The historical internal `:map-programs` facet
holds the checked phases; it is not a second contraction runtime. Float tests qualify host
precision checkpoints, not bitwise agreement with arbitrary BLAS or device schedules.

Compiler-generated pure TypedSOAC programs can acquire an exact source-free invocation boundary
without manufacturing a Clojure `let` or re-inferring parameter types. The ordered IR inputs and
their retained AbstractValues supply a zero-prefix plan; flat public buffers and every retained
shape must resolve directly from public integral inputs or supported public-buffer extents.
The unchanged pure promotion refuses effects, aliases, physical result storage and native
initialization providers. A separate bounded promotion accepts exactly one already validated
sequential ordered effect map and derives its sole read-write destination from retained
result-storage/alias facts. That destination is an unresolved external storage premise, not a
parameter, initializer, allocation, or complete-write claim. Ordinary host materialization refuses
the plan before reading arguments; resident binding and final LinkPlan initialization remain
independent obligations. A separate closed external-dimension premise retains direct private
Int/Long shape symbols for later resident binding without making them logical parameters or
executable values. The existing analyzed-source promotion still independently proves the complete
source return.

Zero-extent ordered effects preserve their exact zero GraphBuffer capacities through scheduling.
Current KernelBody storage parameters require positive extents, so target emission then refuses
explicitly instead of substituting capacity one. Empty-buffer/liveness support is a separate
compiler capability; this prerequisite does not claim it.

A source-free public rank-one buffer `data[m]` can also supply its own private integral shape
dimension through the existing invocation `ShapeProjection`. The invocation parameter alone uses
the checked self-extent view; the original TypedSOAC and program AbstractValues stay unchanged.
Fresh validation proves the exact source parameter, all non-shape facets, the single public
projection source, and the dimension's shape-only uses. Materialization derives typed `m` from
the actual array geometry rather than asking callers for an internal scalar. Shared competing
sources, compound shapes, executable uses, and forged or removed projection markers refuse.
The external-dimension premise can carry the same exact typed value into a separate resident phase
with no public source buffer. This still does not admit empty device storage or arbitrary dynamic
shape programs.

These are generic compiler-frontier premises; only the bounded indexed device family in the table
consumes their single fixed-preparation case. Canonical map/sum
primal, linear and transpose programs promote, schedule and emit through all four C-family
dialects. Generated map lambdas project only a checked homogeneous non-nil Float/Double
`grad-acc` to ordinary typed addition; the original scalar programs remain derivation and
executable-dependency authority. No target-specific AD rule is introduced. `equation-first/compile-typed`
now consumes these programs through the same scheduling and emission tail as deftm compilation.
Its public ABI and homogeneous floating carrier come from retained IR, not caller type overrides.
The ordinary invocation materializer uploads primitive host arrays into owned device storage and
therefore refuses retained borrowed inputs rather than relabeling their ownership. The opt-in
resident surfaces use independently checked resident LinkValues and one same-session named-entry
owner instead. No default-engine migration is implied.

Before source-free equation promotion, the shared reduction-only resident transaction audits the
complete TypedSOAC producer/use/facet graph even when `:resident-outputs` is absent or explicitly
empty. This lets a pristine reduction captured only by a later map cross the component boundary as
one resident buffer and rewrites that consumer to the existing stable singleton load; its producer
and consumer can no longer disagree about buffer-versus-scalar identity. Output-only reductions
remain plain without an explicit demand, host/extent/array escapes still block the transaction,
and terminal ScalarIR realization is not enabled. A changed-input OpenCL/Level Zero map→sum→map
gate checks this internal representation and its original numerical policy. This does not admit a
general TypedSOAC DAG or a dynamic resident scalar input.

Source-free compilation also has one explicit public output-representation demand:
`:resident-outputs` is an ordered vector of exact public TypedSOAC IDs. Each requested ID must be
an output-only, single-result, pristine Float/Double full reduction. The shared resident pass
marks it with the existing one-element device-buffer representation and public reduction
certificate; scheduling, emission and the final compilation boundary independently recheck that
certificate. The demand remains part of public compilation options and identity but is consumed
before scheduling rather than forwarded as a second internal realization request. Multiple
independent direct reductions may be selected, but a multi-result producer,
alias, effect, result-storage relation, forged resident facet or non-reduction output refuses.
Omitting the option preserves output-only scalar behavior, and ordinary deftm compilation
does not accept this source-free ownership request. The option is representation substrate only:
it does not itself make a scalar cotangent input resident or clone singleton storage. The bounded
map-sum device consumer described below adds those separate authorities.

One deterministic map-stage constructor owns fresh interface construction and exact
retained-interface reconstruction. Validation still independently checks scalar derivation before
reconstructing the stage; it neither trusts producer certification nor allocates new residual or
interface identities. The bounded composition below adds an exact execution-source view to the
canonical map owner while retaining the original source as public authority. Existing operation
coverage, residual order and numerical policies are unchanged; arbitrary DAGs remain outside the
boundary, while only the three tabled families have opt-in device consumers.

Checked node-wise map construction, proof replay and segmented-product→activation
composition share one primal/linear/transpose phase assembler. Family admission and
interface validation remain outside it. Constructor accumulation identities and retained
replay identities remain distinct; no node ports, residual copies or arithmetic contracts
are inferred again during assembly.

A checked composition boundary admits a closed region of pure homogeneous maps with one
public result. Intermediates remain private and every source member must satisfy the existing
map-lane and source-envelope premises before fusion. Shared producers and reconverging branches
are allowed only when existing target-neutral fusion derives exactly one supported map with
the same typed public interface and exact retained constituent equation facts. Fusion statistics
and last-merge history are not independent correctness proofs or transitive ancestry.
Scalar projection and differentiation consume that transient fused stage while the bundle
retains the untouched multi-equation source; validation repeats fusion, rule replay and
fixed-interface stage construction. Float conversions and typed locals remain explicit
checkpoints; no source facts or fusion provenance are scrubbed. Composite residual origins
name the ordered source-equation scope and exact fused scalar coefficient, rather than
attributing every coefficient to the first producer. Execution reuses generated primal/JVP/VJP
maps and the canonical owner, not a region runtime, cache or derivative rule. This is the
execution-model contract specified under numerical semantics below, not a proof that AD commutes
with fusion. Escaping intermediates, multiple public results, heterogeneous geometry/precision
and effects remain outside this boundary.

Plain-map derivation evidence now has one checked node-proof schema, rather than both a
top-level scalar bundle and an independently described map lift. Each witness retains exact
source-equation coverage, an execution equation, an existing scalar derivation (or checked
primal-only projection for an inactive node), fixed stage ports and phase identities.
Validation reprojects each scalar lane and replays its existing
derivative rules before reconstructing the executable phases. Removing the proof does not
restore an older plain-map validation path. Map→ordered-sum consumes the same single active
node witness directly; it no longer rehydrates duplicate outer scalar fields through a
transitional adapter. The surrounding sum source must independently prove the original exact
map/reduce relation before scalar-capture activity is admitted. Plain-map scalar-capture
activity still refuses. Indexed derivations now retain their scalar identity witness inside
the indexed proof instead of duplicating its fields at the outer boundary. They do not use
map-node evidence: direct indexed reads, saved indices, ordered scatter-add and the exclusive
typed-zero destination remain independently reconstructed indexed obligations. This is proof
ownership consolidation, not arbitrary indexed composition or additional operation coverage.

Scalar rule correspondence, map reconstruction and ordered-sum reconstruction share one
request-local dependency-observation scope and final sequential identity fence. Bare scalar
public validation retains its own lifecycle. A matching certificate does not replace rule
replay or authorize a different phase program. Sum evidence admits exactly one active map
witness; multi-node map→sum and general reduction AD remain separate capabilities.

All three lifted families and standalone scalar validation reuse the same scalar
correspondence and rule-replay authority. Indexed reconstruction shares its enclosing
dependency-observation scope and final fence, while runtime/device consumers still receive
the same canonical ports, residual bindings and physical initialization obligations. One
schema-owned namespace projection is reused for normalized and canonical fingerprints only
after identity checks. It preserves nested certificates and unknown data; it is not a
validator or permission to normalize arbitrary runtime objects.

Singleton and fused-map proof reconstruction use the same checked node lifecycle as
nodewise maps, without a second scalar projection and phase reconstruction. The outer
envelope must still match the rebuilt node interface exactly, including metadata; generated
ports must remain fresh against both original and fused source values. Scalar-active maps
still require the independently checked ordered-sum source. Removing duplicate replay does
not remove these correspondence premises or broaden operation coverage.

Singletons and regions whose fused execution frontier has at most two arrays retain their
existing one-witness programs and numerical model. Only a previously refused homogeneous
single-use spine with more boundary arrays acquires source-ordered primal/JVP equations and
reverse-ordered transpose equations. Internal active results remain immediate single-use edges;
selected external arrays may occur in several nodes. This is not arbitrary DAG AD or
a claim that these spines cannot be fused by the ordinary compiler.

External tangents share exact identities only in derived execution views; retained node ports
remain fresh and distinct. Reverse contributions accumulate in reverse node order, with each
separate Float/Double addition stored before the next addition. No zero identity or reassociation
is inserted. Each addition retains checked primal-only `grad-acc` evidence, exact contributor
order, fresh equation/result identities and the library helper dependency. Reconstruction derives
the required contributions from source topology and consumes retained identities without
allocation. The first contributor initializes the accumulator. Existing unique-use programs,
small fused-region numerical models and proof schemas remain unchanged.

Activity is derived across nodes. A node outside the selected derivative path retains only
its checked primal, not synthetic tangent ports. Downstream derivatives save required values
as fresh owned residuals, including values produced by such inactive nodes. Residual keys
are qualified by execution equation, and each saved array names its actual producer.
Generated ports are globally fresh. An internal read edge preserves every AbstractValue
facet while relating a fresh owned producer to its checked read-use; composition then uses
the ordinary pure-spine appender. No new evaluator or derivative registry is introduced.
Public `deftm` semantics and the default source AD engine are unchanged.

The shared host compiler materializer separately admits ordered pure homogeneous map graphs,
including tuple results and multiple declared outputs. An intermediate may be consumed by a
later map and also returned as a fresh owned value; this is needed to export saved coefficients.
Equation order, typed checkpoints and declared output order remain unchanged, with no fusion,
pruning or reassociation. Each map still passes the existing host admission checks; the graph
adds closed SSA liveness, exact external-input correspondence and common extent/dtype premises.
Effects, aliases, physical storage, host reads, unknown layout/placement and intermediate
control values refuse. This is a compiler projection capability, not graph differentiation or
authority to classify outputs as primal results, residuals or gradients. Canonical AD must
independently derive those roles, ownership relationships and derivative programs.

The same compiler boundary admits one ordered scalar reduction and pure map graphs ending
in reductions, optionally with fresh segmented products. Each reduction retains its actual
array input, scalar dtype, certified fold and source order through the existing realization;
this grants no reassociation. Mixed graphs require exact live inputs/outputs, no effects,
aliases or physical storage, and owned positive static arrays at cross-operation edges.
Dynamic/empty longer producer graphs, product→sum without a map, and scalar-reduction-to-map edges remain refused;
the existing direct map→reduce path retains its dynamic/empty extent support. This is a
shared host projection capability, not admission of a new AD family or device schedule.

Normalized AD remains opt-in; the source AD engine remains the default structural reference.
The checked scalar lane supplies derivation evidence to three bounded host families: pure map,
map followed by ordered additive sum, and functional indexed map. Those families now execute
through one canonical program/interface/residual owner. Preparation retains full validation;
hot calls check compiler/rule epochs and captured executable identities without replaying proofs.
Host projection selects from actual TypedSOAC operations and declared operand roles, rather
than AD derivation-family labels. Its bounded map, map→reduce and indexed-store admission
checks remain separate from derivative validation; arbitrary equation chains still refuse.
After effective callees are captured, host preparation requests a freshly fully validated,
exhaustive residual plan from the canonical owner. The plan partitions exact existing bindings
into owned arrays and immutable scalars and carries physical obligations through unchanged. It is
preparatory evidence only: it allocates nothing and grants no lifetime, lease or GPU execution.
Indexed pullbacks additionally require private index snapshots and a fresh typed-zero physical
gradient owner on every call. These are sequential lifetime contracts, not concurrent mutation
or asynchronous ownership guarantees.

`normalized-program-bundle/adapt` is the single strict evidence-to-owner boundary. It recognizes
only the exact existing map, map-sum or indexed facet set; missing, nil and mixed facets refuse.
Named adapters delegate to the same construction/validation authority. Runtime `prepare-owner`
accepts and independently validates a canonical owner, while compatibility `prepare` performs
canonical adaptation first. Runtime no longer interprets legacy evidence facets to select an
adapter or admit a family. It still enforces kind-specific numerical/physical requirements.

Both entries use one private consumer of freshly validated owners, retaining independent
post-callee-capture validation. Compatibility preparation keeps its existing three scalar
evidence validations; direct owner preparation performs two. These are preparation-only,
not hot-call replay. This cleanup introduces no proof-skipping flag, public trusted receipt
or validation cache. Invalid evidence passed
to compatibility preparation now reports canonical `:ad-program-bundle-scope`, replacing the
runtime's `:ad-map-runtime-bundle`/`:ad-map-runtime-composition` classifications. Execution and
gradient semantics are unchanged; general equation composition and default AD migration remain open.

The existing public `:normalized-scalar` JVP and reverse entries share one internal runtime
lifecycle. Their distinct body projection, compiler adapters and metadata remain unchanged;
effective-callee capture precedes independent selected-call validation, compilation and every
invocation retain pre/post fences, and original compilation/execution failures propagate.
This removes duplicated lifecycle code, not scalar proof checks or the source/default route.
It is a sequential staleness contract, not protection against concurrent mutation or ABA.

Checked two-equation pure-spine append now builds the admitted map→ordered-sum programs;
it is not yet a general differentiation rule for arbitrary equation chains.
General equation composition, conditional/loop AD regions and default-engine migration remain
separate gates. Device execution is restricted to the tabled families. Existing real-weight
training acceptance is still open. The sections below describe the individual construction checkpoints;
an adapter's lack of runtime effects does not mean its checked envelope is unused by the runtime.

> **Historical checkpoint note:** The sections from “Equation projection and shape facts” through
> “Migration and acceptance gates” preserve the order in which construction and validation
> boundaries were established. Statements about what a checkpoint did not yet enable are historical
> receipts; current availability is governed by the table above and by “Opt-in resident
> normalized-map execution” below. Numerical, ownership and refusal constraints remain current
> unless explicitly superseded.

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
are unchanged. Map/sum mixtures refuse. At this construction checkpoint the adapter did not itself
enable a device runtime, new derivative rule or equation composition; the later resident consumer
adds only the bounded indexed execution listed above.

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

The prediction-array witness does not differentiate a scalar Gaussian parameter, a vector-latent
regression program or distribution records. Binary logistic likelihood is not
categorical SFT: grouped log-sum-exp, target gathers and the unchanged real model gate remain
open. No family validator or construction evidence retires on the strength of this witness.

## One active scalar capture in an ordered map sum

`linearize-map-sum-program` accepts `{:active-input source-value-id}` to select one homogeneous
Float/Double array or scalar capture. Existing `:active-array` remains an array-only option;
the options cannot be combined. Selection names an existing source value, retaining that
member's actual metadata and AV rather than trusting metadata supplied on an option symbol.
These are source selection options, not canonical port aliases. The canonical interface uses
ordered plural `:active-inputs`, `:tangent-inputs` and `:cotangent-results`, including for a
singleton selection.

Derivative operand placement is projected from retained AV shapes through one shared layout:
arrays become pointwise element parameters and scalars become capture parameters. An active
scalar therefore supplies one scalar tangent without a materialized broadcast array. Its
transpose map produces per-lane contributions, then appends the exact original certified
ordered reduction to obtain one scalar cotangent. Its identity, addition, Float checkpoints,
iteration order and numerical attributes stay unchanged. The sum interface identifies the
final `:gradient-results` vector explicitly for both array and scalar activity; the scalar case
is a singleton, not a duplicate singular alias.

Per-lane contributions are never certified as a standalone scalar gradient: the shared private
builder attaches sum evidence before full validation, and standalone scalar-active map program
construction refuses. The existing map-sum family, canonical envelope, host materializer and
hot revision/dependency fence remain the authorities. No new family selector, activity pass,
derivative registry, scalar gradient cache or reassociation permission is introduced.

Gaussian location witnesses cover Float/Double scalar tangent and cotangent carriers, independent
density/analytic/finite-difference checks, zero extent, transpose pairing, exact source reduction
correspondence and mutation before first pullback. Observation arrays and scale remain excluded.
Multiple active values, integral/extent captures, nonzero initializers, device execution,
default-engine migration and general equation composition remain outside this checkpoint.

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

The opt-in normalized map composition boundary differentiates its exact, independently
rederived **execution source**. When several source maps legally fuse to one execution map,
its scalar binding order and precision checkpoints define the primal, JVP and VJP numerical
model. The untouched source retains the public interface, legality premises and provenance;
it is not a claim that node-wise source transposition has identical rounded cotangents.
In particular, substituting a shared producer into several consumers can change derivative
accumulation order even when primal checkpoints agree bit-for-bit. The descriptive retained
source-arithmetic facet grants no additional reassociation permission. No default source-AD
behavior, generic fusion policy or arithmetic permission changes at this opt-in boundary.
Exact source-node graph transposition, including ordered array cotangent accumulation,
remains a separate structural-transform obligation; successful map fusion does not discharge it.

Higher-order AD requires closure of the generated programs under the requested transform.
Residual production must remain differentiable where the outer derivative depends on it;
freezing all residuals would lose Hessian terms. An opaque pullback closure or unrepresented
tape is a capability refusal, not evidence of a zero Hessian-vector product. Keep current
HVP refusals until typed closure is demonstrated.

## Migration and acceptance gates

The bounded canonical Float square-map established the first durable device composition gate: its
checked primal and transpose compile through `compile-typed`, bind existing LinkValues,
and execute through the generic named-entry owner on local OpenCL and Level Zero. An
independent stepwise-Float oracle checks raw bits across changed seeds and input generations;
retained residuals survive caller-input mutation. Rule/callee guards and lease-lifetime
refusals are tested. That test-only assembly preceded the bounded public resident consumers below;
it remains evidence for their shared boundary, not general DAG support or default-engine migration.
Residuals retain their canonical owned contract;
the borrowed cotangent input binds to globally owned storage without retagging its AV.

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

The latest unchanged real Gemma training case has 12 failing adapter checks out of 28 (the earlier
recorded run failed 11). A same-CPU-operand layer control still fails 4 of 14 adapter checks.
The latest selected SGEMM leaf comparison, using the same GPU operands, differs from the actual CPU
SGEMM by at most 2.3841858e-7 absolute with relative L2 error 3.8074463e-8. Upstream input drift
remains; this local diagnostic is not model acceptance.

The October 15 paired-operand diagnostic further isolates the layer-1 key-projection LoRA
gradient. Its final two-term contraction has bit-identical CPU/GPU primal operands; the
incoming cotangent differs by relative L2 1.4330e-5. Independent ordered Float evaluation
matches every GPU output (10,240 elements), while CPU execution matches forward Float FMA
for this particular contraction. At the worst original-metric coordinate, the 6.8274e-6
CPU/GPU difference decomposes into 6.8274e-6 from changed operands and 1.4438e-11 from
final arithmetic realization. Cancellation amplifies the small incoming difference; this
does not justify changing the acceptance criterion.

One contraction upstream, all 4,096 weight elements are identical and independent ordered
Float evaluation matches all 32 GPU results. Both native endpoints fall inside conservative
same-input dot-product rounding bounds, under checked finite/no-overflow/no-observed-subnormal
assumptions. The incoming 512-element cotangent already differs by relative L2 9.0017e-6.
These are local arithmetic/provenance diagnostics, not an end-to-end error bound or proof
that the derivative is mathematically correct. CPU BLAS ordering remains implementation-defined.

An independent seeded-Rademacher finite-difference direction is inconclusive: three small
perturbations leave the Float objective unchanged, and larger steps do not establish a stable
adjacent-scale resolution window. It is neither a derivative failure nor an acceptance pass.
The real-model gate remains open; no tolerance, default math policy or AD engine was changed.

The exact generated SFT objective, Gaussian/logistic density composition, city kernel primal
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
and extent inputs. `:jvp` takes the argument vector and, for one active input, its explicit
tangent array. Multiple active inputs require an exact vector of tangent arrays in declared
active-port order and still return `[output tangent]`. VJP slots remain in source-input order.
Array outputs never imply a scalar seed.

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
The host consumer adapts admitted normalized evidence once at its preparation boundary. All
three supported families then execute the canonical primal/linear/transpose programs,
ordered input/value interfaces, owned residual bindings and physical requirements. It
does not rebuild a separate runtime interface or execute a legacy program facet. Family
source/derivative validators remain the mathematical authority; they cannot be retired
merely because execution no longer reads their storage representation. Indexed destination
and result IDs remain distinct, and every pullback allocates a fresh typed-zero destination.
No caller array is retained as a residual. This does not add a runtime mode or general AD
composition.

Construction uses `soac/append-pure-spines` instead of a family-local right-biased fact
merge. Each consumer→producer boundary is explicit, including identity mappings. The shared
operation accepts a validated nonempty pure producer program and one pure consumer equation;
the normalized-AD caller separately proves that the producer is a linear spine. Neither input may
carry aliases or physical result bindings. Shared AbstractValues and provenance must
agree exactly; equation IDs remain
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
of exactly the retained Float/Double type; `:jvp` takes one tangent for a singleton selection,
or an exact tangent vector in activity order for several active arrays, and returns a scalar
tangent. Positional excluded cotangents stay nil. The private residual arrays
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
retains none of them in the pullback. With a singleton selection only its positional slot has
a cotangent; excluded arrays, scalar captures and extent retain nil slots. Ordered plural
array selection now extends this same ownership contract to maps, finite map chains and one
map followed by an ordered sum. General equation DAGs and chain→sum composition remain separate
gates. The real Gemma gate remains open as described above; this map witness is not a replacement
objective or model acceptance result.

## Operation-driven primitive host projection

The primitive host projection has one operation-driven entry and one source assembly owner.
Specialized map/map-reduction/indexed APIs keep their existing strict admission contracts.
Shared plain-host AbstractValue checks are factored once; numerical order, fresh outputs,
stable array captures and caller-owned indexed transpose destinations are still checked at
their respective operation boundaries. This does not generalize the accepted topology or
derive AD rules from mere structural source realizability.

One internal prerequisite also admits a single source-free functional `segmented-reduce` under
the existing Float/Double `:abstract-blas-product` source contract. It allocates a fresh owned
plain-host result before invoking the unchanged typed contraction projection. Admission requires
static positive dense geometry, one result and accumulator, exact public operands, and no physical
result storage, effects, aliases, transforms, stages, mixed storage or unknown facets. The older
caller-owned projection remains unchanged. This is host realization infrastructure, not an AD
facet or evidence that its implementation-defined reduction association is bit-identical to a
particular BLAS library.

## Opt-in resident normalized-map execution

`raster.ad.normalized-device/prepare-owner` is a bounded lease-oriented device consumer for one
already canonical checked map or finite linear map spine. Preparation fixes the target, numerical
policy, scalar values and concrete array geometry, compiles only the canonical primal and
transpose, and delegates allocation, execution, failure poisoning and lifetime to one ordinary
named-entry owner. It is deliberately not a CPU-compatible AD function: calls upload exact host
input buffers and return opaque resident leases plus logical output IDs, with no host readback.
The target is mandatory; this surface has no hardware-specific default.

Canonical proof reconstruction is preparation-only, including the post-instantiation check.
Resident replay retains the same pre/post compiler-epoch and effective-callee execution fences
as the host normalized runtime. Hot closures capture only those fences, immutable port IDs and
the generic named-entry owner, not canonical evidence or a caller-forgeable validation receipt.
Post-run fence failures still close newly issued leases before publication; cleanup remains
available after invalidation. This removes proof replay, not runtime storage or lifetime checks.

The primal result carries the exact residual-frontier dependency map required by transpose. A
zero-residual transpose receives an empty map and needs no artificial primal lifetime; extra or
stale leases still refuse at the generic named-entry boundary. Seed and gradient storage are fresh
standalone allocations cloned from independently validated concrete geometry. Their logical
contracts come from the canonical cotangent input and transpose output, never from the template
input value. The seed's borrowed consumer contract is projected to owned global storage at this
one boundary; every other AbstractValue facet remains exact, and the emitted consumer remains
borrowed. Fresh resident binding and the final LinkPlan/LinkModule revalidate all storage and
ordered-effect obligations.

`prepare-sum-owner` adds one separately admitted canonical map→ordered-sum family. Preparation
fixes the exactly typed scalar cotangent seed; transpose has no fake rank-zero input buffer and no
per-run scalar override. Primal calls instead upload an exact ID-keyed map covering every canonical
array input, including excluded coefficient arrays. The demanded rank-zero primal loss uses the
shared certified resident-reduction representation, while ordered owned-array residuals remain
private dependency frontiers. Transpose takes only that exact dependency map and publishes a fresh
gradient lease. The same binder, LinkPlan/LinkModule validation, named-entry owner, epoch fences and
cleanup path are used by both families.

`prepare-indexed-owner` adds one positive fixed-preparation-geometry indexed-map family. Its public primal
boundary is the canonical ordered pair of excluded Int/Long indices and active Float/Double data.
The index/cotangent extent remains static. The data/gradient extent may be static or one private
direct integral symbol projected exactly once from the prepared data array; calls cannot change it.
Each call clones the indices, validates their exact dtype, length and bounds against the active
extent, then uploads that same checked clone; the canonical primal owns the resulting device index
snapshot as its sole dependency frontier. The cotangent is an ordinary array input whose borrowed
callee contract is projected to fresh owned module storage. Gradient storage preserves the
canonical owned transpose result and binds both that logical result and its distinct physical
zero-gradient destination to one value.

Transpose is one named entry containing a genuine input-free complete zero map immediately before
the canonical sequential ordered effect. The zero map is derived from the freshly validated single
typed-zero/exclusive physical obligation. Final LinkPlan/LinkModule analysis—not the effect—proves
initialization and complete production; the ordered effect retains its read-write partial-update
contract. No atomic or reassociation permission is introduced. Empty geometry, dynamic index space,
compound/shared dimensions and per-call geometry changes remain outside this slice.

The source-free compiler can nevertheless retain one closed external-dimension premise for this
later boundary: a private Int/Long symbol used directly in resident array shape slots, but absent
from logical inputs, outputs, executable expressions, effects, aliases and storage identities.
The InvocationPlan records the freshly rederived ordered symbol/dtype set without inventing a
parameter, binding, initializer or allocation. Ordinary host materialization refuses such a plan;
only resident binding with the exact typed scalar value can discharge it while rechecking the
concrete buffer ABI. This is compiler infrastructure only and does not make dynamic indexed
geometry generally available; the indexed adapter consumes only the single fixed-preparation case
above.

These opt-in resident slices still accept exactly one active public array and immutable scalar
specialization. They retain no caller array after preparation and do not add a cache, session owner, lease registry,
device readback, dynamic scalar seed, mutable state, general scatter, arbitrary DAG execution, or
default-engine activation. The original campaign gates, real Gemma numerical acceptance and long
GSDM qualification remain independent.

### Approximate primals retain their own derivative semantics

The allocating `raster.dl.nn/gelu` uses a clamped odd rational approximation of tanh
for vectorizable forward arithmetic. Its first and second derivative kernels differentiate
that rational function, not a substituted libm tanh. This corrects gradients without changing
the forward polynomial or its Float/Double arithmetic order. Existing JVP and transpose rules
continue to use those kernels; no additional AD dispatch authority is introduced.

At the raw tanh argument ±9 the clamp uses the saturated-side convention: its derivative
is zero. Outside the clamp, GELU's derivative is the actual constant rational tail slope
(not an assumed exact zero or one), and its second derivative is zero. NaNs propagate.
As elsewhere, AD differentiates the arithmetic structure, not the discontinuous rounding
of a floating-point implementation. Derivatives use dtype-specific base coefficients
(narrowed before differentiation), evaluate the polynomial jets in Double, and narrow at
the derivative array store. This is a smooth extension, not exact differentiation of
every rounded Float intermediate. The separate `gelu!` and `gelu-mul!` operations still
use libm tanh and are not bitwise-interchangeable with this allocating rational primal.

Validation includes frozen forward bits, a structural polynomial-coefficient check,
finite differences of the primal on each smooth side of the clamp, second-order AD laws,
and changed-input device replay. A real Gemma Double diagnostic preserved all sixteen
finite-difference forward outputs and reduced the two inspected coordinate discrepancies
from roughly 2.4e-8/1.6e-8 to 1.2e-10/1.3e-10. Those two coordinates do not establish
full-model Float/GPU numerical acceptance; that gate and its tolerances remain unchanged.
