# Compiler unification campaign

## Reconciled execution plan

This checkpoint preserves the original eight-item campaign. The entries below are completion
obligations, not claims that their already-landed foundations must be implemented again. Historical
receipts remain below; fresh acceptance must distinguish compile-only, hardware-free simulation,
co-located execution and actual multi-device measurements. Normalized AD and memory planning are
cross-cutting parts of this campaign, not competing replacements for it.

### Original campaign obligations

This is the completion checklist. Foundations below are recorded implementation
history, not fresh qualification of their remaining end-to-end gates. Current work
on original source-owned host computations and prepared AD composition advances
items 3 and 5; it does not replace the other six obligations.

| Item | Existing foundation | Remaining completion gate |
|---|---|---|
| 1. CI and coverage | Seven CI gates, vendor source compilation, targeted native/differential tests | Reconcile skipped tests and environment prerequisites; partition by affected compiler layer; retain cold-load, negative-admission and CPU/device controls. |
| 2. Compatibility and debt ledger | Executable compatibility inventory and many bounded retirements | Re-audit reachable production paths and dependent packages; give every remaining adapter an owner and removal gate; remove covered dead code and duplicate tests without losing distinct oracles. |
| 3. Direct TypedSOAC migration | Typed packets, ordered ABI, graph/body emission, retained dtype/effect/storage evidence | Close remaining source/semantic composition boundaries, bound compilation cost and remove migrated re-lowerings; demonstrate the same numerical intent through JVM/C and GPU routes with precise refusals. |
| 4. Portable source compilation | OpenCL/Level Zero execution and CUDA/HIP compile gates | Keep actual public workload fixtures portable and distinguish source compilation from runtime/device evidence; coordinate the external CUDA runtime work against the common contracts. Cloud hardware is not a prerequisite for this local campaign. |
| 5. Reusable training | AD rules, opt-in normalized bundles, artifacts, generated updates, GSDM staged execution | A resident forward/VJP/update composition, normalized AD migration gates, unchanged generated SFT and real-weight numerical acceptance, and measured steady execution. GSDM, Gemma and density/PDE derivatives are complementary controls. |
| 6. Hardware-free distributed planning | Topology, physical service lanes, route evidence, root/scratch budgets, simulation/certificates | Fresh topology/dependency/capacity/collective/halo/drift counterexamples; independently check cost, numerical intent and ownership witnesses. Analytical forecasts must remain distinct from measured fabric performance. |
| 7. Data parallelism and halos | Co-located update/collective and numerical halo demonstrations | Close public compiler-to-plan-to-runtime composition with device-scoped views, completion and cleanup; check nonuniform partitions and exact byte accounting. Do not claim multi-host transport or overlap from one laptop. |
| 8. Durable and multilevel simulation | Manifests, mmap lease tests, checkpoint/restart and AMR provider slices | Fresh-owner restore/publication with producer geometry, time and lineage; numerical restart/branch and declared conservation/convergence/fidelity oracles; reject corrupt, incompatible or prematurely released state. |

### October 21 operand-order repair and the next executable boundary

Whole-draft CI checkpoint, exact remote head `150fd933` (not main): public source
job 33722 attempted 172 compilations, passed 142 and reported 30 refusals. Its 15
refused workloads occur on both CUDA and HIP:

| Shared missing boundary | Public workload set | Next acceptance obligation |
|---|---|---|
| Source-owned scalar completion | `softmax`, `softmax-cross-entropy`, `loss-fn`, `gqa-decode-attention-buf!` | Preserve original scalar and reduction/count semantics through the common frontend, not workload-specific waivers. |
| Checked floating conversion | `batch-norm`, `rms-norm-reassociated!`, `rms-norm-chunked-backward-dx`, three row activation quantizers, `layer-norm-reassociated!`, `rope-prefill-strided!` | Complete generic checked conversion/stage composition or prove totality from independent evidence; do not replace checked conversion with unchecked conversion for a green fixture. |
| Structural read provenance | `rms-norm-chunked` | Transport the actual read certificate through the common transformation, without minting one from equal syntax. |
| Allocation extent initialization | `gqa-causal-mha`, `gqa-causal-mha-jvp` | Retain canonical integral extent and source-position/count completion. |

The failed deterministic unit-job logs (33720) report 305 assertion failures and
273 errors across their failed shard output; these are **not** 578 independent
defects or a complete suite summary. They include canonical liveness/fusion,
normalized AD captures, source ordering, artifact persistence, distributed training
and AMR composition. OpenCL job 33721 also fails. Narrow ownership tests below do
not supersede these results. This checkpoint must remain draft. Fix shared
mechanisms, retain negative-admission tests, and requalify real unchanged workloads
before claiming coverage regained. The independently landed #1558/#1559 each had
all seven green gates; this debt must not be attributed to those main extractions.

Landing update: PR #1558 squash-merged as `94017072` after all seven required
contexts succeeded on reviewed head `e3f56e30`. The focused main-based checks were
57 pipeline/inliner tests / 389 assertions, plus one permanent generated weight
gradient device/JVM/independent-reference test / 12 assertions on OpenCL and Level
Zero with changed inputs. These qualify that repair, not the whole training draft.
The draft has merged this main checkpoint while retaining its additional snapshot
evaluation-denial checks. Its broad CI and unchanged resident training acceptance
remain open.
Fresh post-merge qualification in the single restarted 900 MiB REPL passes 67
construction/import/inliner tests / 555 assertions and 136 source/bytecode/AD
ownership tests / 1,579 assertions. The groups overlap and are not additive;
neither group closes the unimplemented nested-prefix execution relation.

CI feedback extraction: PR #1559, reviewed head `8fd40930`, squash-merged as
`85192b1a` after all seven CI contexts passed. It changes test
infrastructure only. It attempts both CUDA/HIP public fixture groups, accumulates
ordinary refusals and still fails the gate; cancellation/interruption/fatal errors
abort. A same-REPL control against baseline `aa816699` preserved all 172 exact
calls (86 per target), Var identities, options, optional captured descriptors and
ordered artifacts. Eight reporting tests / 45 assertions pass; normal shard
discovery includes the new namespace. These stubbed controls are not source
compiler or device-performance evidence. The draft has merged the extraction,
preserving main's original public Float contraction fixture. This replaces
first-failure-only feedback without waiving any workload.

Independent whole-draft triage separates three shared consumer obligations:
producer-owned source evaluation through the existing activation path; checked
conversion scheduling at its actual semantic position; and in-process execution
ownership versus persistent-artifact eligibility. Keep `execution-identity`'s
portable content-identity contract strict. Local observation, if added, needs an
explicit authenticated process scope and must never populate portable program
fingerprints or reach AMR/resident-state exporters by comparing absent hashes.

Optimization-retention failures are distinct from coverage refusals. A fresh
canonical-liveness probe succeeds with a validated typed program but removes zero
bindings rather than two. Well-formed TypedSOAC alone does not prove successful
original reads or permit erasing their failure obligations. A future removable-
evaluation proof must separately close primitive numerical totality, count/index
completion, original access domain and private/unobserved storage, or retain the
original ordered checks independently of removed numerical work. The raw typed
route fixture is not a valid standalone JVM source oracle; qualify this with an
actual prepared `deftm` and short/null/alias/earlier-effect controls. Do not weaken
effect flags or change expectations merely to make the failed tests green.

Inspecting the actual placement seam uncovered a real pre-existing miscompile in
default `anf-lift`: concatenating work lifted from every call/vector operand moves
a later nested initializer before earlier operand evaluation. A two-element vector
probe changed both its first value and the effect trace from `[:first :second]` to
`[:second :first]`. Binding only earlier source values is insufficient: callee
lookup and typed formal CHECKCAST/coercion can also have required earlier positions.

The independently reviewed repair uses one sequential-wrapper normalizer. Calls
and vectors remain opaque; whole `let`/`do` wrappers still flatten in source order.
The unsafe default/safe-option split and redundant internal switches are removed.
Restoring operand lifting requires an explicit completed-operand/callee contract,
not purity or literal-looking source. This is a focused main-based extraction in
PR #1558, head `d4a2a99e`; draft-side implementation is `91fef595`. No draft-only
snapshot/ownership machinery is included in that PR.

Evidence: exact extracted normalizer functions plus its on-disk tests pass 29
tests / 189 assertions in the reused REPL (supplemental, not a fresh environment).
Draft-side 67 construction/import/inliner tests / 555 assertions and 136 ownership
tests / 1,579 assertions pass; groups overlap. Expanded normalized-AD controls
report five `:ad-normalized-sum-boundary` errors in 20 tests / 121 assertions with
BOTH the repair and exact pre-change `f1b3a2e2` functions in the same REPL. These
remain separate debt, not green qualification. Unchanged `local-loss` still source
compiles to two equations and two invocation steps. Full CI and device/training
acceptance are separate obligations.

The actual final training plan is scalar `rows * 1`, zero allocation, contraction,
Float reduction and a terminal Double quotient. The original MSE count reads its
prediction's length; its lowered use of the allocation extent is not by itself a
source-placement or resident-result correspondence proof. Existing
`project-scalar-metadata`, `owned-long-length-facts`, ShapeProjection and ordered
ScalarCompute machinery can be reused, but must not fill that missing relation.

Next implement a compositional frame-prefix verifier, consumed by SourceOwner:

1. Start at the exact original root call and its checked selected frame. Instantiate
   callee environments from indexed slots; retain every input evaluation/completion,
   including unused inputs, and descend through checked selected ancestry.
2. Derive the complete ordered host prefix up to the first numerical operation.
   Verify sequential placement/alpha-renaming rather than matching generated names.
3. Match that prefix to the final plan's exact source-order entries, ScalarCompute
   objects and result-materialization conversions. The first useful consumer is
   the real training `rows * 1` step, not another standalone trace query.
4. Reuse the original argument outliner/existing scalar executor exactly once.
   Missing/reordered/duplicated work must refuse before allocation. Do not let a
   partial prefix witness grant an unproved later helper interval.
5. Only then connect a completed numerical result to a formal's logical-view
   length. Require the exact live view/producer, not equal shape or backing capacity;
   initially plain contiguous rank-one, Java-array-compatible length. Execute the
   original checked count conversion before seed initialization. Raw signed count
   and nonnegative launch extent remain separate. No download, dummy array or
   original numerical CPU evaluation may supply this evidence.

Acceptance retains unchanged `local-loss`/`local-gradient-sum`, original-versus-
lowered prefix overflow and first-failure/unused-input ordering; changed/zero
lengths, sliced capacity mismatch, inconsistent/oversized views, replaced/stale
results, producer failure and exact release/completion are negative controls.
Native TypedSOAC operations retain their own semantic contracts: importing
original Clojure semantics is not a blanket JVM-emission prerequisite.

Independent consumer review refined the first executable subset. Extend the
existing `source-prefix/begin`/`close` authority for original call roots, retaining
the flat-let case. `original-step` may then identify an authenticated original
argument slot; `source-owner/prepare-scalar-steps` uses the existing
`prepare-frontend-invocation-input!` and stores its executor in the existing
`:steps` table. `evaluate-ordered-invocation-step` remains the sole executor.
Do not add a frame interpreter, a second tracing authority or a model-specific
extent evaluator.

For unchanged `local-loss`, source order is: MSE receiver; prediction's linear
receiver and all five arguments (`x`, `theta`, `rows`, `width`, literal `1`);
linear body entry; allocator receiver; allocator argument zero's reference
CHECKCAST; argument one's checked multiplication and Long completion; only then
the allocator body and allocation. MSE target/denominator arguments occur after
the linear call returns and cannot be hoisted before it. Neither equal carrier
classes nor selected ancestry alone closes constructed-slot correspondence.

First support only unchanged public references whose observed CHECKCAST target
matches the exact array class required by runtime preflight, and primitive Long
public ports/the actual Long literals. Reference completion does not prove
non-nullness; null arrays remain outside the existing exact-array preflight
domain. Computed arrays and Object/Double-to-Long adaptations refuse until their
own completion and placement are integrated. Reject wrong array classes before
the product, changed receiver/callee history, intervening effects, and
missing/reordered/duplicated final prefix entries. Whole-plan preflight must still
refuse later unclosed MSE count/quotient work before any prefix effects run.

The actual selected allocator argument can already be outlined with original
physical ports `[batch out-f]`: zero/changed dimensions agree and checked
`Long/MAX_VALUE * 2` throws. This is execution evidence for the existing owned
argument region, not completed frame-environment or final-placement authority.
The permanent regression extends the existing unchanged-training substitution
case rather than adding a parallel fixture. Its focused case passes 1 test / 73
assertions; the construction/import/inliner group now passes 67 / 561. Independent
review confirmed that outlining remains explicit and grants no prefix replay.

The second review found a missing right-endpoint obligation in the initial slice
plan: the product's executable and final allocation metadata do not prove absence
of intervening original helper work. The actual chain must also close `alloc-like`
delegation to `zeros-like`, its receiver and all formal completions, the exact
original array constructor/count, zero initialization and helper return
completions. `direct-allocation-projection` can certify a selected helper whose
entire original body is a direct constructor; it cannot retrospectively certify
its enclosing delegation interval. `allocation-corresponds?` checks logical
constructor/shape/initialization, while SourceOwner's extent-availability check
does not establish the missing source interval.

Likewise the pre-kernel prefix includes the post-allocation `oftype` arguments,
which still lack their independent source-constant/substitution completion. Do
not admit the product merely because today's `local-loss` later refuses a missing
MSE executor: that incidental failure is not a safety authority. The default next
vertical must finish the complete prefix before granting public execution. A
smaller internal interval is useful only if an independently derived unresolved
enclosing-frame obligation still prevents whole-plan admission. Add controls
with effects/throws between product and constructor, after construction before
helper return, and in a later GEMM argument; each unrepresented operation must
prevent admission. This is a strengthened completion requirement, not a new
function registry or permission to evaluate the numerical body on the CPU.

### October 21 selected ancestry for imported helper calls

Construction direction after independent review: preserve original completed
call slots at the shared inline builder instead of repeatedly recovering erased
input conversions downstream. An authenticated resolved-call plan must retain
receiver/Method, original operand contexts and policy; every argument, including
unused arguments, completes before body entry. Return carrier, boxed payload and
actual return adaptation remain distinct. Unsupported return/completion recipes
must retain/refuse the call. Existing SourcePrefix/SourceOwner consumers remain
the placement/execution authorities; this does not solve resident child results
or helper endpoints by itself.

The first implemented prerequisite is explicit Int/Long bytecode input
completion. It reuses `emit-argument-value`/`emit-coerce`, not Clojure source casts,
logical dtype inference or a new evaluator. Existing FP completion shares that
emitter; source projection refuses unsupported bytecode contexts. Structural AD
explicitly refuses integral completion (including inactive operands) rather than
falling through an unhandled dispatch. Five related namespaces pass 67 tests /
664 assertions in the reused campaign REPL. A focused main-based extraction is
PR #1560, head `d52d2906`: fresh main-based qualification passes 67 tests /
658 assertions across six existing completion/bytecode/AD namespaces. The
semantic discriminator is permanent: original bytecode D2I saturates a primitive
Double overflow, whereas a checked Clojure source cast throws. The exact original
path must be retained. All seven CI gates remain required; no inline expansion or
GPU admission is widened yet.

Imported call origins now feed the existing selected-call contract, rather than
remaining disconnected tracing data. The shared Method/receiver/implementation
checks serve both root and imported calls. An imported frame must derive its exact
caller through one unique recorded parent's independently checked substitution,
then retain that parent's selected ancestry. Identity-cycle detection and a total
search budget derived from the finite recorded builder list bound the search.
Root selection failure cannot fall back to an imported interpretation; stale
ancestors propagate failure, and refused attempts publish no observations.

The unchanged training `local-loss` now connects its actual `alloc-like` call to
the selected `linear-nb` parent. The two imported `oftype` calls still refuse a
frame because their bare cast constants lack an independent source-substitution
proof. Original input conversions are retained; constructed-argument equivalence,
resident result/count metadata, return completion and final execution placement
remain explicitly unclosed. The transport-level imported selection candidate's
`:parent-linkage :unclosed` is candidate-local, not a permission to execute.

Targeted same-REPL evidence: 65 construction/import/inliner tests, 543 assertions;
136 source/bytecode/AD ownership tests, 1,579 assertions; zero failures/errors.
These groups overlap and are not additive. Controls cover competing parents,
cycles, wrong root/ancestor selection, stale ancestor propagation, a wrong imported
call, and transactional refusal. Source-fixture CI on preceding head `65e3250a`
still fails, including softmax's uncertified `inv-sum`; these local checks do not
qualify the broad draft for merge or establish resident training execution.

### October 21 DeepSeek inference reference integration

The recovered full source review is retained locally in
`.internal/deepseek-inference-requirements-20261021.md` (ignored, not a portable
repository artifact). It records source-line evidence and the exact dirty-consumer
boundary; the pinned requirements and acceptance gates below remain the tracked
campaign record. Recovery rechecked pins without builds, model downloads or
changes to pretrained-rstr.

Read-only research verified the official [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
release and its exact dependency pins. Shallow sibling checkouts, without models,
builds or submodule initialization, are available for reproducible source study:

| Sibling directory | Inspected commit |
|---|---|
| `DeepInfer-raster-review` | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` |
| `DeepGEMM-raster-review` | `1fc9c66975ee322d628b3288534ca0e9ce12a64d` |
| `FlashMLA-raster-review` | `2e5429fc5653bab6e081f09477126f731882a6a9` |
| `DeepEP-raster-review` | `f6aa38fe06cd4a0489e7730365d4e0114970a2f3` |

These observations are requirements evidence, not benchmark or execution results.
The pretrained comparison used checkout `f4c2bb476884c36b5e12a8fa934eb2ddf4ca2ead`
with existing decoder/local-server edits; neither it nor Raster's shared root was
modified. It is not a qualification of current released pretrained behavior.

| Requirement / evidence | Existing campaign integration and bounded acceptance |
|---|---|
| Encoded values **and auxiliary scales** have physical layouts, group axes and conversion semantics. DeepInfer `layers/tilelang_ops/cache_common.py::KVCacheFormat` and DeepGEMM `csrc/jit_kernels/heuristics/config.hpp::GemmDesc` keep these concerns explicit. | Items 2/3/4/8: generic byte-storage codecs and views, explicit decode/compute/accumulation and rounding contracts. First compare tiny encoded matvecs with decode→Float contraction, including tails and scale edge cases. Do not implement FP4 as merely a new fractional-byte scalar dtype or add model-specific emitters. |
| Same-shaped inputs can invalidate inspected-content-dependent schedules. Pinned [FlashMLA's interface](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/flash_mla/flash_mla_interface.py#L90) warns that changing valid-length values while reusing scheduler metadata silently retains stale split-KV scheduling. | Items 1/3/4/7/8, immediate correctness priority: distinguish replay inputs from specialized assumptions; retain content/version dependencies and completion-fenced metadata generations through existing owners/events. Test changed lengths with unchanged shapes, stale slots and failure/cancellation across two inflight iterations. Do not add a second attention-specific artifact cache. |
| Sparse selection, ragged histories and expert routing compose several numerical operations. DeepInfer `attention_backend.py`, `models/v4_modules/mlp.py` and pinned grouped-GEMM paths provide concrete cases. | Items 2/3/4/5: qualify existing segmented gather/selection/weighted-reduction and histogram/scan/permute/grouped-contraction/combine composition. Start with empty and skewed groups, duplicate policies and masked invalid values against sequential oracles. Compression rollback remains a model-state contract, not automatic compiler algebra. |
| Bulk matrix pipelines require phase/transaction and result-readiness contracts, beyond an async-copy spelling. DeepGEMM's SM100 implementation uses payload/scale streams, transaction barriers, rings and special storage. | Items 1/3/4/6/7: qualify existing `PipelinedFor`, async-copy/commit/wait, participation barriers, collectives and `MatrixMad` first; report native versus synchronous physical lowering. Later add only workload-backed bulk-copy, async-matrix, special-storage and cluster capabilities. An async-shaped IR or compiling instruction does not prove overlap or speed. |
| Communication consumes compute and memory resources as well as link bandwidth. DeepEP routing handles/events and DeepInfer dispatch/combine SM budgets are concrete evidence. | Items 6/7: segmented exchange/all-to-all-v intent with counts, permutations, replication and combine arithmetic; shared compute/HBM/fabric capacity and completion ownership. Add hardware-free skew/contention/cleanup counterexamples now; real NVLink/RDMA and measured overlap belong to multi-device qualification. |
| Regime-specific specialization includes scale layout, group skew, valid lengths and replay buckets, not merely dense matrix sizes. | Items 1/3/4/5: reuse existing finite schedules, numerical permissions and versioned measurement receipts. Bound search/compile/code-size/memory costs and measure cold preparation, host submission, GPU events and warm replay separately. No independent DeepSeek autotuner or keyword dispatch. |

The release is useful inference implementation evidence, not a complete production
cache/load-balancing/disaggregation stack. These requirements strengthen the same
eight obligations; they do not add a ninth campaign or close source correspondence,
real-weight training, durable publication or actual multi-host acceptance. The
immediate continuation remains the local source/semantic ownership repair, then
content-aware prepared replay and encoded/ragged composition qualification. No
SM100, Ascend, model download or cloud dependency enters the hot development loop.

### October 21 integral call-completion prerequisite and next construction gate

October 22 continuation: the opt-in completed `linear-nb` call is now tested as
part of the **whole CPU training loss**, comparing original and completed JVM
Methods bit-for-bit and against an independent squared-loss oracle for changed
weights and row counts 1/3. Numerical calls run outside retryable compiler proof
transactions; captured definition history is checked before and after execution.
This is not resident training qualification or default producer enablement.

CI also exposed a shared frontend/AD envelope drift: analyzed-source equations
now carry both `:source-prefix?` and `:before-parallel?`, while normalized AD
expected only the former. The common envelope checker now checks both exact
booleans against its admitted no-host-activity equation sequence. It still refuses
unknown attributes, host activity, allocation/evaluation obligations and invalid
provenance. Real frontend Float/Double map-to-sum objectives, their JVP/VJP and
finite-difference oracles, and tampered phase/provenance controls cover this
boundary. An independent read-only review found no blocker. The adjacent resident
scalar allowlist accepts only optional boolean phase facts; it does not use them
as movement authority or weaken totality to admit unfinished source arithmetic.
Targeted AD/whole-loss qualification passes 50 tests / 806 assertions. The new
resident phase-fact test passes 22 assertions; the other 33 resident tests retain
exactly the same failure signatures before/after the allowlist repair (27 failures,
18 errors), including source-decline and unresolved arithmetic obligations. These
are explicitly recorded baseline defects, not a green resident vertical.

Draft #1539 remains unqualified: at head `e8cd519d`, setup/build passed but test,
public fixture emission and OpenCL CPU gates failed. The 16 deterministic test
shards reported 7,535 tests / 88,656 assertions / 317 failures / 274 errors;
these are check totals, not independent defect counts. Public softmax and the
training gradient still show source-arithmetic/evaluation correspondence gaps.
No broad green or merge readiness is claimed by these targeted repairs.

Next reviewed construction obligation: numerical producer → original binding →
scalar consumer. `coverage-decline`'s normal-completion summaries are diagnostic
only: its empty lexical environment can miss a preceding reduction result, and
an Object/present/Double summary cannot establish execution correspondence.
`source-prefix` currently stops before the first parallel form; its success
must not be reused as a post-reduction placement certificate.

Extend the existing SourceConstruction/correspondence/SourceOwner path with a
bounded direct reduction-to-scalar interval. At the actual producer, retain the
original scope and binding ordinals, exact equation/step identities and observed
normalization/construction edges. Join existing reduction emission, binding and
completed positional invocation observations with recurrence/checkpoint and FP
consumer proofs. Keep actual stored carrier distinct from boxed payload. This
relation needs a pre-admission producer seam; neither diagnostics nor a late
prefix check can retroactively grant it. No new global facts, runtime or cache.

First executable witness: an unchanged flat direct interval with checked literal
count, literal initializer and independently certified reads, initially empty and
singleton domains. This is a construction gate toward unchanged softmax/training,
not replacement acceptance. Helper-expanded intervals and general dynamic counts
stay held until their original-position/count/read contracts are connected.
SourceOwner consumes the authenticated interval; resident reconstruction still
checks the final use-closed placement. Required negative controls cover wrong or
copied occurrences, stale history, changed carrier/coercions, intervening effects,
extra uses, unchecked count substitution, insufficient/null storage, changed
recurrence/association and missing read-domain evidence. The arithmetic flags
remain until this complete join—not result typing alone—justifies their discharge.

The next physical prerequisite is now observed across method extraction: the
producer retains original reduction → exact helper body → resolved static
Method/return boundary → actual static argument conversions → exact bound local.
It reuses existing body/invocation catalogs and historical module guards;
generated calls receive no frontend origins and use a distinct `:module-static`
mode. Reduction root and helper body result carriers/contexts must agree. The
body-carrier/Method-return pair describes an actual conversion, not its identity,
nonthrowingness or numerical equivalence. Missing, ambiguous, edited, foreign or
malformed links abstain. No scheduling/admission rule or numerical code changed.

Independent review caught and verified fixes to root/result joining, malformed
record guards and oversized indices. Affected qualification passes 96 tests /
1,393 assertions across seven ownership/input/return/source-construction
namespaces; the new forced-split/unsplit Float/Double boundary and negative controls
account for 3 tests / 67 assertions within that total. Full interval/count/read
correspondence and use-closed consumer placement still remain to be connected.

The source-local observation now joins that producer to an immediately following
selected FP consumer in an unchanged two-binding scope, reusing the existing
private input representation under a distinct entry/query family. Unsplit Object
storage and extracted primitive storage remain different; consumer argument
coercions, receiver, actual return/storage and selected payload computation are
retained separately. Primitive caller shortcuts abstain rather than inheriting
an Object-interface proof. Whole-query history adoption is transactional; a
refusal retains none of its individually successful subqueries. The old prefix
family cannot consume the new capture. Independent review found no blocker for
this conditional scope. Affected qualification: 98 tests / 1,485 assertions,
zero failures/errors; new interval controls account for 2 tests / 92 assertions.
This still supplies no count/read, equation placement or execution certificate.

Next connected execution obligation has existing capacity authorities:
`segred_body`'s input `:minimum-elements`, reconstructed fold-read requirements,
`kernel_call/validate-retained-input-capacities!`, linked-view range validation
and the ordered source cursor in `gpu/link`. These must be reused, not replaced
by another bounds system. They are not yet source read completion: unknown native
capacity remains a caller obligation and capacity failures are not original
null/bounds failures. A general reduction-count outliner may capture arrays or
effects; it cannot be represented by `ScalarCompute`, which requires pure
rank-zero operands. The first closed literal count 0/1 family authenticates the
original cast lookups and evaluates through the shared source constant evaluator,
without inserting a synthetic source binding. General counts still require an
authenticated original-position execution relation. Keep signed source count
separate from nonnegative launch extent.
At the reduction cursor, validate the original read domain against actual public
array/live-view storage. Empty domains require no read or fabricated upload.
Qualify null/empty inputs at counts 0/1, short views over larger roots, unknown
capacity, changed replay, count-before-initializer failure and no launch/result
publication after failed validation before consuming the numerical interval.

The optional closed-count observation now retains the exact original checked
initializer and bound context, original bootstrap cast identities, actual Integer
0/1 value and independent pure/no-flags obligation. It is not a domain/read or
execution certificate. Post-emission cast replacement invalidates the owning
history. Explicit unchecked narrowing can produce 0/1 in the shared evaluator,
but still abstains here while its independent source effect remains unclosed;
value folding does not erase evaluation obligations. No admission rule changed.
The affected seven-namespace gate passes 102 tests / 1,551 assertions. Current
draft head `dbe75388` still fails test33827, fixture33828 and OpenCL33826; setup
and build pass. CUDA/HIP gates have not established readiness. No merge is allowed.

The original empty/singleton execution oracle adds 48 passing checks across
Float/Double and forced split/unsplit emission: count0 does not read null/empty
storage; count1 preserves NullPointerException/ArrayIndexOutOfBoundsException;
changed values 2/4/-2 retain the expected consumer result. These execute original
Methods outside proof transactions, not a device substitute.

Further path review corrects the proposed integration seam: the exact two-binding
fixture returns the scalar consumer, so existing terminal-host suffix construction
already retains it without relaxing scalar admission. A real public equation-first
compile reaches terminal ownership and currently refuses `.invk` with
`:unclosed-terminal-source-projection` ("terminal symbol has no owned Var").
Connect the authenticated original interval to the exact promoted reduction and
existing terminal region; do not add a generic pre-admission bypass for this case.
The generated terminal contract reads the reduction export and evaluates its
nested scalar binding. This still requires original/constructed correspondence,
actual operand/return completion and original read-domain execution contracts.

Independent path review confirms two distinct remaining changes. First, the
terminal projection must consume an authenticated exact interval/contract join
and invoke the captured interface Method/receiver with the original ordered
argument completions. Reuse the existing Long-only original-invocation projector
machinery, but do not resolve `.invk` as a generic Var or rewrite typed Float
division as core `/` (which changes arithmetic width). Second,
`invocation_materialization/begin-materialization` currently validates every
buffer eagerly through `runtime-buffer-shape`; it rejects nil even for count0,
before any reduction cursor. Source-owned deferred array-reference materialization
or verified dead-input projection must preserve public argument positions and
original read ordering/failures. Existing raw arguments/ordered realization are
the reuse point; no independent input cache or fabricated nil-as-empty buffer.
Terminal projection may separately qualify valid-array execution, but must not
be reported as completing zero/null/read semantics. Source-free compile entries
and generic frontend admission remain unchanged. Final affected source gate now
passes 103 tests / 1,599 assertions; this does not resolve the broad draft CI.

The conditional terminal-consumer half is now implemented through the existing
source-prefix correspondence representation and TerminalResultOwner, with no
new runtime, cache or portable IR. The owner joins the original unchanged
two-binding source through its actual normalization/construction phases to the
exact final terminal contract and equation/export endpoint. It invokes the
captured interface Method through shared ordered-call normalization. The actual
Method result and observed consumer storage remain Object. An immediately
returned, single Object alias is elided rather than emitted with an inferred
primitive local hint; both source tags must match the independently observed
boxed payload, and conflicting hints/param-tags refuse. Explicit mismatching
owned-mode input refuses rather than falling back to ordinary projection.

Crucially, the numerical frontend changes accumulator/index/input annotations.
Those are NOT waived or accepted as numerical correspondence: the retained
producer endpoints explicitly carry `:status :unproved`. Tests supply the
original source's completed reduction value to the terminal owner and compare
the resulting whole consumer against original execution. This is conditional
consumer qualification, not GPU producer qualification. The default
equation-first path is deliberately not wired to this capability until the
independent producer/count/read execution proof is connected.

An independent reviewer found no normal-path semantic blocker for this scope;
its malformed-source hardening and explicit-mode refusal fixes are included.
The affected nine-namespace REPL gate passes 141 tests / 2,187 assertions,
including split/unsplit Float/Double, counts0/1, non-exact Float reciprocals,
changed inputs, copied/foreign phase/program/contract boundaries, rollback,
malformed source and conflicting-hint controls. Broad draft CI remains held.

Next input execution work reuses InvocationMaterializationState and ordered raw
arguments: owner-authorized existing buffer parameters may remain pending, never
fabricated MaterializedBuffers, until the reduction cursor checks original entry
CHECKCAST/null/bounds semantics and materializes actual demanded storage. Factor
the existing parameter materializer instead of duplicating shape/alias rules.
The zero case also needs validated empty-domain emission that removes unused
pointer demand: current SegRed still carries its masked input pointer. A nullable
entry CHECKCAST is not a nonnull read proof; deferral alone does not solve count0.

PR #1560 landed as `3bff894e` after all seven exact-head CI gates succeeded.
Its shared bytecode completion emitter now represents actual Int/Long invocation
parameter conversions without substituting Clojure checked casts. The distinction
is observable: the original JVM Double-to-Int path saturates, whereas source
`int` can throw. The campaign integration retains the separate reference-result
context and its stricter AD refusal; merging this prerequisite changes no campaign
execution admission by itself.

PR #1561 landed as `a02ee86b` after all seven exact-head CI gates succeeded.
It is the separately reviewed consumer: original input outlining selects
the contract from the actual Method Class, while the AD producer explicitly
requires a floating completion and matching floating Method return. Targeted
qualification passed 97 tests / 1,221 assertions in the bounded existing REPL;
this is not a fresh-process or whole-draft result. Campaign synchronization of
the squash preserves the independently qualified integration source unchanged.

The next construction change must replace retrospective argument matching with
completed slots at the original-source producer boundaries. The shared inline
builder consumes that construction; it cannot query a source-phase certificate
that does not exist until after it returns. Its acceptance boundary is the whole
selected call interval, not only a successfully evaluated allocation count:

1. Authenticate the actual selected receiver, Method and each original operand
   context. Logical scalar tags do not determine the physical completion recipe.
2. Evaluate receiver, then **every** positional argument and completion once in
   original order, including unused arguments, before entering the callee body.
3. Substitute completed values, not uncompleted operands. Preserve original
   return completion separately; an Object result and a primitive payload are
   not interchangeable. Unrepresented return adaptation retains/refuses the call.
4. Consume this construction through existing prefix/source-owner/scalar-compute
   machinery. Do not introduce a call-frame interpreter or let a product/count
   executor authorize an unclosed enclosing helper interval.
5. Require complete coverage before executing any prefix effect. Controls must
   cover a failing earlier conversion, unused throwing argument, wrong array
   Class before an overflowing count, effects between count and constructor,
   effects after allocation before helper return, later numerical arguments,
   and omitted/reordered/duplicated/copied slot occurrences.

First qualify original public primitive Long/literal inputs and already completed
public arrays with independently checked runtime Class. CHECKCAST alone proves
neither non-nullness nor safe access. Computed array arguments, helper delegation,
return adaptation and resident producer/count correspondence remain explicit
obligations until their complete intervals are represented. The existing two-
equation training loss and softmax's scalar-after-reduction case are acceptance
workloads, not reasons to bypass an unresolved interval.

Fresh coherent campaign integration of the outlining consumer passed nine
completion/bytecode/AD/construction namespaces: 121 tests / 1,570 assertions,
zero failures or errors. The duplicated raw integral outliner branch is removed;
the existing CHECKCAST path, positional completion validation, historical owner
guards and terminal-region implementation remain. Independent integration review
found no blocker. These targeted checks do not supersede the known broad-draft CI
failures or qualify the draft for merging.

A separate REPL construction probe confirms the next gap rather than merely
inferring it from code: an actual typed helper with an unused Long formal throws
on a nil argument; the existing expansion emits no argument bindings and returns
`5`. Constructing three explicit original observed integral completions before
the same scope-aware substitution restores both the nil failure and normal result.
The original retained invocation nevertheless reports an **Object** return, even
for this Long helper. Thus the prototype does not prove a return recipe or grant
production expansion, and cannot justify dropping the return-boundary gate.
The local reproduction is `.internal/probe-unused-formal-completion.clj`; it is
diagnostic evidence, not a permanent passing regression or a shipped inliner fix.

The return-path review also rejects `body :ref` plus invocation Object as a
sufficient recipe. The original static body Method may insert a return CHECKCAST,
and its wrapper is a separate edge. The first reference recipe therefore needs
an actual emitted, unguarded reference pass-through bridge tied to that whole
callee emission. It must replay the body under its saved result context, preserve
the actual Method-return CHECKCAST when non-Object, and preserve wrapper reference
identity. Extend the existing emitted return-bridge producer for that edge;
do not reinterpret its primitive payload proof or add a caller-editable authority.
Primitive boxed helpers retain their calls until their distinct recipe closes.

### October 21 reference-return production and constructor insertion point

PR #1562 landed as `4e15484d` after all seven exact-head CI gates succeeded.
The existing wrapper producer records an actual unguarded reference-static-return
edge, separate from primitive boxing. Shared validation now ties the saved whole
body to its actual generated Method and return descriptor; a same-ABI foreign
Method cannot supply a return CHECKCAST Class. The primitive observer retains its
previous restrictions. Main-based qualification passed 110 tests / 1,396 assertions.

The next campaign implementation constructs an owned whole original reference
call, still **before enabling normalization or GPU admission**. It binds the stable
receiver first, then every actual argument (including unused arguments), using the
original typed-invocation emitter's shared conversion path. A canonical reference
input context joins the existing integral/floating contexts; Object consumption
retains its saved boxing/call-result behavior, while concrete reference consumers
retain their actual CHECKCAST. These are not source casts, AD rules, non-nullness
facts or portable native projection. The imported whole body runs under its saved
result context and actual Method-return check. Historical source ownership and
producer adoption remain shared, not an additional runtime interpreter/cache.

Review identified two further obligations now represented here: substitution
shares the inliner's use-site-metadata policy; actual bytecode body emission
records the existing conservative method-local-callee denial. The constructor
requires explicit closed dependency observations for the body and original
receiver/argument operands. An authentic registered-source regression injects an
owned same-class sibling through the existing producer path, verifies original
execution does not call its colliding global, then checks construction refuses
without publishing speculative dependencies. Other differentials cover null/
array identity, unused argument effects and failure before the body, copied
occurrences and stale ownership. Six whole-call tests pass with 58 assertions,
including unchanged nested delegation, its effects and nullable results. The
preceding affected 17-namespace gate passed 262 tests / 2,702 assertions before
that final delegation test; independent review found no blocker in this unwired
checkpoint. These reused-REPL results are not fresh-process or broad-CI receipts.
The next step is construction-edge integration at both owned normalization
producers, followed by complete helper-interval/result correspondence. No
production normalization fallback is removed by the unwired constructor alone.

The first normalization consumer now accepts the opaque completed-call value
explicitly for its exact original root, not a compiler-option flag or caller map.
The immutable frontend remains unchanged; the normalization edge joins the
constructed root back to its original call and retained constructor. Imported
callee nodes do not acquire caller-source origins. The constructor uses raw body
import, separately from the public helper-import producer, to prevent accidental
recursive construction during later integration. Default production normalization
is unchanged pending full interval admission.

Canonical bytecode input/result contexts are now shared rewrite barriers for
legacy helper expansion, dispatch rewriting, AD application hoisting and D-call
rewriting. This is shape-based protection, not inferred ownership or an AD rule;
source Object completion is not relabelled bytecode-owned. Tests retain the
delegated call and actual argument/return region through backend expansion.
A fresh REPL first passed the constructor/transport gate (27 tests / 241
assertions); the subsequent affected 17-namespace gate passed 265 tests / 2,774
assertions. No whole-draft, numerical-GPU or performance acceptance follows.

The opt-in phase now uses one same-position identity splice in both root
normalization and helper import, rather than a root-only special case. Exactly one
executable original occurrence is replaced; quotes stay opaque, equal-looking
sibling calls are not rewritten, and ancestors retain source origins without
inheriting completed-call authority. Import runs the splice before alpha-renaming
and qualification; the constructor itself deliberately uses raw import.

An actual failing argument-origin differential exposed an important emitter
boundary: `(identity count)` need not emit an invocation observation, and its
emitted operand is a reconstructed object. The constructor now joins the two
existing producer records: exact frontend parent/argument through `InputOrigins`,
then the owned positional argument through `retained-input-source-context` to the
actual operand. Both identity checks are required; no equality/name matching or
new semantic rule for identity is used. The capsule retains optional origin pairs
under the original frontend/module history. Missing edges remain absent and this
join grants no argument equivalence, placement or numerical admission.

Independent review also caught a mixed-AD traversal: when another VG tuple was
tracked, `resolve-vg-nth` could still enter a sealed completion region. It now uses
the same barrier; a regression exercises folding outside while preserving the
original region inside. Lexical-call expansion respects the barrier too. The
latest affected 17-namespace gate passes 270 tests / 2,835 assertions. It covers
source/normalized/imported differentials over branches, sequential rebinding,
earlier failures and duplicate-looking calls; preserved argument origins;
positional/copy/refusal query controls; and copied/foreign/stale import capsules.
Default production normalization/import and numerical admission remain unchanged.
Next is complete interval consumption in the actual training vertical, not a
claim that constructor or origin coverage alone establishes resident execution.

The campaign consumer uses the existing authenticated frontend/module-history
observation boundary, not an editable return map. Cold queries do not compile;
promotion requires a fresh owner, and the old cold history becomes stale. Missing,
copied or foreign modules publish no observations. A real training compilation
qualifies `[F` reference bridges for `linear-nb` and its nested `alloc-like`, while
MSE correctly does not acquire one. These observations still do not close return
substitution or final placement. Twelve integrated namespaces pass 158 tests /
1,995 assertions, zero failures/errors; this is not whole-draft CI acceptance.

Implementation review refined the next insertion point to the two existing owned
source producers: `normalize-frontend-bindings` for the root and
`prepare-inline-body` before callee hygiene/qualification. They have authentic
frontends and exact original call occurrences, unlike the later shared builder.
One shared constructor must authenticate selection, receiver, all original input
contexts, imported callee and reference return bridge; emit every ordered completed
slot; substitute only those values; and wrap the whole body in its saved result
context plus actual Method-return CHECKCAST. Existing hygiene, qualification and
sequential flattening then operate on this construction. Keep the frozen frontend
identity unchanged. Do not manufacture a partial `SourceConstruction`, grow a
retrospective resolver graph or interpret call frames at runtime.

Initially support stable global typed receivers and the represented reference
return recipe. Lexical/guarded receivers and unsupported primitive returns retain
their original calls; later inlining must not erase their still-unclosed boundaries.
Unavailable/cold evidence also retains the original call without silently warming
JIT state. Explicit preparation and whole-owner retry remain producer lifecycle
steps. Native TypedSOAC intrinsics keep their direct semantic contracts and do not
acquire a blanket JVM compilation prerequisite. Acceptance must close one entire
reference-returning helper interval, including unused inputs and delegation, not
only execute its product/count operand.

### October 21 typed reduction identity and rewalk qualification

Focused PR #1554 (`0b839b48`, based on main `5baa8b30`) replaces the
softmax maximum seed with the existing typed `neg-inf-val` helper. The former
Float specialization retained checked `(float ##-Inf)`: ordinary Clojure rejects
that conversion even if an unchecked bytecode conversion masks it. The new source
does not read an element to select its identity, so empty inputs remain valid.
Explicit checked Float narrowing is not weakened.

Inlining this helper exposed an independent information-loss issue: the reduction
walker inferred an accumulator type but retained it only on the form, not its
binder. A subsequent walk after initializer lowering could no longer recover it.
The fix uses the existing symbol metadata helper to retain that same logical type
on the binder. It adds neither static-field inference nor a softmax special case;
logical type retention still cannot stand in for physical-carrier evidence.

Fresh capped clean-main JVM qualification: `raster.nn-test` and
`raster.compiler.core.walker-test`, 53 tests / 356 assertions, zero failures/errors.
The retained-source oracle uses ordinary Clojure evaluation independently of Raster
bytecode coercions. Float/Double tests cover empty and shifted inputs, independent
stable-reference values, primitive array classes, unchanged inputs and changed-data
calls. The integrated broad-branch REPL passes 54 tests / 360 assertions, including
its extra qualified-call reference-preservation regression. Independent review of
the exact four-file patch found no blocker. All seven exact-head CI contexts then
succeeded, and #1554 was squash-merged as `695c24ba` on October 21. The broader
draft has reconciled that main ancestry without extra content changes. These
source/compile gates are not backend-performance or full-draft acceptance claims.

The unchanged-source GPU probe now gets past the reduction-initializer rewalk and
still refuses `softmax/inv-sum` with `:uncertified-host-scalar` and
`:unclosed-source-arithmetic`. The next completion obligation is transporting
actual source-owned producer/checkpoint evidence into scalar FP consumers after
reductions, preserving their original evaluation position. Conditional reduction
value correspondence must not silently prove loop counts, read bounds, argument
evaluation or safe box erasure. No arithmetic flag is waived by this source fix.

### October 21 lexical observations survive method extraction

The next source probe localized a concrete lost producer relation: softmax's
original root scope existed in the bytecode binding catalog, but its binding
observations were empty. `compile-typed-impl!` partitioned the function into
static parallel helpers and rebuilt the main `let` without the root scope's
metadata. The shared extractor now retains that root occurrence only after
checking equal binding-vector size and identical ordered binder objects. It
retains the original lexical head too. This is a producer-owned transformation,
not a syntax search or a new type inference rule.

The observation continues to expose the ACTUAL initializer and physical carrier.
An extracted reduction reports a static helper initializer and may return primitive
Float where the unsplit reduction returned an Object with a Float payload. The
unchanged scalar initializer still identifies its original invocation context;
the generated helper is not an original source invocation. Whole-body vector
identity is deliberately not transferred. Both the original whole-body query and
the production AD entry still refuse the split body without its separate proof.

The regenerated softmax source now has all four binding observations (Float max,
Float-array map output, Float sum, Object scalar-call result), while its GPU scalar
continuation remains honestly unqualified. No count conversion, read-capacity,
association or box-erasure permission follows merely from these observations.

Independent review found no blocker; its requested production-AD negative was
added and passes. Reused capped-REPL qualification covers frontend/reduction,
binding and terminal ownership (42 tests / 386 assertions), plus loop carriers,
whole-bytecode gradients and original source-prefix ownership (58 / 600), all
passing. Copied scopes, equal-but-distinct binders, changed layouts, actual helper
initializers and changed-data Float results are covered. This advances the producer
transport prerequisite of items 3/5; full draft CI and the end-to-end training
acceptance remain outstanding.

### October 21 no-op source reconstruction through region-copy lowering

Actual REPL phase instrumentation found that preserving only type-elaboration
edges was insufficient: region-copy rebuilt every call even when it expanded no
copy. A shared `util/remake-if-changed` now preserves seq/vector occurrences only
when ordered child OBJECTS and count are unchanged. Equal symbols with different
type metadata rebuild; original collection kind and metadata are retained. The
AD application hoister reuses this helper instead of its duplicate implementation.
Region-copy reuses actual control-head symbols and its original binding vector
when unchanged, while rebuilding ancestors of real copy expansions.

The traversal also now treats quote/core-quote as opaque in both value and
statement contexts, and leaves empty `(do)` empty (nil) rather than constructing
`(do do)`. These are source semantics corrections, not source-proof defaults.
No permission for changed copy nodes, array aliases, bounds or metadata-only
equivalence is added.

Three affected namespaces pass 77 tests / 421 assertions; the ten ownership/AD
namespaces separately retain 135 / 1,572, and existing copy complete-write
coverage passes 1 / 8, all in the reused capped REPL. Independent review found
no blocker. A diagnostic construction-edge map at actual type-tagging returns
now reaches the real flat-binding helper builder with authenticated original
caller, authenticated callee import and identical argument occurrences. Without
the no-op region-copy repair the caller chain was missing. This diagnostic map
grants no production admission. The next connected implementation must retain
the actual type-elaboration edge and consume helper intervals in source-prefix
validation and original source execution; it must not infer provenance by names.

The focused main extraction is PR #1557 at `52a05e5d`, with exactly the shared
helper, region-copy traversal, and the no-op/quote-safe AD hoister (main did not
yet have the draft's earlier hoister repair). Independent review of its actual
six-file diff found no blocker or broad-only API dependency. Its exact 67 on-disk
test declarations pass 368 assertions in the reused development process; fresh
all seven exact-head CI contexts then succeeded, and it squash-merged as
`aa816699`. #1556's apparent status delay was verified
through CircleCI's workflow API: setup/build succeeded and downstream jobs were
queued, not absent from configuration. Both exact-head guarded monitors have
completed their merges; these focused receipts do not qualify the broad draft.

### October 21 shared operand outlining retains original method dependencies

A real same-class sibling/global collision reproduced wrong execution in an
owned invocation input outline: original operand `local-value(-2.0)` selected
the original class's sibling and returned -1.0; the detached outline selected an
unchanged same-named global, returned 100.0 and incremented its counter. Historical
Var/module guards and hygienic NEW method names do not prove preservation of the
ORIGINAL class's method environment.

All operand families now capture dependency closure under the actual original
emitter context. Invocation inputs, array indices, allocation counts and terminal
tails share physical-operand-input for lexical ports, full scalar context and
the conservative sibling-dependency facet; reduction counts forward their actual
bound observation to the same outliner guard. Missing/nonboolean evidence refuses.
Same-class sibling candidates refuse until their original executable dependencies
can be transported; they are not re-resolved as globals. The scanner follows the
actual typed (namespace-insensitive name) and ordinary (full spelling) emitter
lookup keys, treats quoted data as opaque and checks nested executable bodies.
It is denial evidence, not a new dispatch registry or proof of purity.

Independent review caught the fifth shared caller, terminal-result outlining;
the expanded affected gate also reported seven terminal errors before its
producer/consumer path was covered. That path now captures its dependency facet
under the original complete terminal emitter context, never defaults it from
the reduced retained context, and keeps original return/Method/history guards.
All five collision fixtures refuse without executing the unrelated global;
positive ordinary-global and terminal controls retain original execution.
No runtime submission or numerical performance improvement is claimed.

The current on-disk affected gate spans ten namespaces, 133 tests / 1,557
assertions, zero failures/errors in the reused capped REPL. The focused main
extraction's exact four test declarations pass 28 assertions in that same
process; this is not fresh isolated main qualification. The independent reviewer
checked the actual extracted source/test diff and found no blocker. Fresh CI
remains mandatory before landing.

The focused main extraction omits terminal/reduction APIs not yet present there;
it retains the same mandatory shared closure for every existing main caller.
This closes a prerequisite for executing original helper argument outlines, not
the ordered argument/result interval transport or full training acceptance.

The focused fix is PR #1556, reconciled head `13e4da6f` after main's #1555
landing. The reconciliation adds only the independently qualified naming fix
and its fixtures; the reviewed dependency-closure diff remains two files. Its
all seven exact-head CI contexts succeeded and it squash-merged as `a17af351`.
#1555 passed all seven exact-head contexts at `3dc8ff4e`
and squash-merged as `0513d222`; the broad draft has merged that main commit
without dropping its separate ownership work. Neither receipt qualifies the
broad draft or closes its still-red workload gates.

After reconciling #1555's additional on-disk fixtures, the affected ten-namespace
gate passes 135 tests / 1,572 assertions in the same capped REPL. This is a
targeted reuse-process receipt, not the broad CI or a fresh-process benchmark.

A subsequent actual-phase probe refines the interval implementation plan: for
unchanged `unused-checked-helper-input`, normalization authenticates its root,
but the first real binding inliner receives a different initializer occurrence.
Equation-first does not default `:inline?` true; `:lower` first elaborates binding
types and rebuilds the call, and the helper is expanded later in `:fixpoint`.
Therefore the normalization-to-type-elaboration construction edge must be
retained before authenticating the inliner interval. Do not assume the inliner
sees an unchanged normalized call, weaken identity queries or infer origins from
equal spellings. This evidence changes the next integration action; it does not
grant completion/execution permission to the elaborated call.

### October 21 ordered unused scalar prerequisite

The actual unchanged helper probe exposed a missing executor before interval
integration: an unused checked argument retained its TypedSOAC equation and
required-evaluation obligation, but no InvocationPlan step. Shape liveness must
not decide whether original checked work executes.

Pre-device required scalars now retain a host-only equation with empty scheduled
operations and exactly one existing ScalarCompute executor. They are not labeled
graph-shape proofs. Prefix dependency pruning roots their existing obligations,
rechecking the nested catalog, transported claim, provenance, exact source
binding and computation flags; removing both catalogs does not prove totality.
Flags come from the retained scalar computation, not a generated result ABI cast.
The frontend's before-parallel relation remains distinct from stageable-prefix.

Focused reused-REPL validation passes the seven invocation/shape/ordering tests
(106 assertions, including the production public-parameter ordering control) and
135 source/ownership tests (1,573 assertions). These are not full draft CI gates.
The helper now has its ordered step but still refuses before allocation with
`:unclosed-ordered-source-executor`: original argument/result interval ownership
is deliberately not inferred from the existence of a scalar evaluator. That
construction-edge integration remains next; no fallback or new interpreter was
introduced. Review also confirmed that rebased source IDs are not globally equal
to retained binding ordinals, so no such new coordinate assumption was added.

### October 21 actual source-construction history

The equation-first pre-SOA phase now scopes one owner-local SourceTransport
collector around the actual pipeline. Type elaboration records its actual
input/output and lexical environment, without rerunning inference. Both shared
inliner entrypoints pass their actual caller before recursive argument rewriting;
the observer records the builder's argument/substitution/result construction,
not a fictitious final AST interval.

An unchanged real helper probe now joins normalization to its caller's actual
type edge and exact imported callee. Identity-only queries reject copied outputs,
ambiguous/fan-in/fan-out/cyclic construction chains, repeated caller expansions,
changed or missing positional arguments, and stale caller/callee source inputs.
The actual recursive path has both an unchanged-argument positive control and a
rewritten-argument negative control. Self identity edges are omitted, and chain
traversal is bounded; queries currently scan retained events, not an indexed cache.

The history is passed directly to the prefix verifier and is not added to compiler
options, AST metadata, target modules or portable compilation artifacts. The
existing bounded prefix/metadata/order verifier remains unchanged after that
phase identity check. An observer trace is trusted producer provenance, NOT
standalone transformation authorization: before expanded intervals can execute,
the consumer still owes independent bounded transformation laws, exact final
placement, formal completion, result adaptation and original source executors.
Incompatible tag provenance deliberately grants no width/completion permission.
The original checked-helper overflow case still refuses before device allocation.
Independent read-only review found no blocker in this provenance-only scope.
The reused capped REPL passes 55 transport/inliner/construction tests (364
assertions) and 136 source/ownership tests (1,579 assertions); these are focused
gates, not fresh-process, whole-draft or performance qualification.

### October 21 original frontend argument execution

Explicit `prepare-frontend-invocation-input!` now follows the authenticated
frontend/input-origin relation into the existing bytecode input outliner. It
preserves the original positional completion, physical ports and emitted method
dependencies. This prepares an argument slot, not a whole helper or numerical
argument; no source interpreter, new dtype inference or execution-order grant
is introduced.

The returned executable freezes both raw frontend identity and full transitive
module history. A module guard alone accepts an equal-but-copied raw source body;
the additional frontend guard correctly refuses it. Guards run before execution
and before returning, with no fresh observations. A local collector prevents
unrelated enclosing dependencies from becoming runtime costs, while historical
adoption still fences the enclosing compilation. The existing input invocation
checks its own region guard; the wrapper does not redundantly repeat that guard.

Independent review found no blocker. Five new tests / 37 assertions cover the
actual unused checked helper argument, overflow, out-of-owner invocation, copied
and foreign inputs, source replacement before/after execution, transitive changes,
no refreshed evidence and isolation from unrelated parent history. The existing
136-test source/ownership gate also passes 1,579 assertions in the capped REPL.
The complete source expansion and unchanged training gates remain unqualified.

The next consumer must use general lexical call frames, not enlarge a root-only
flat-prefix matcher or dispatch on the literal-count fixture. Actual `local-loss`
contains only `mse-loss(linear-nb(...), y, rows)`; allocation products and reduction
counts belong to imported callees. A frame must independently validate the
bounded builder/substitution law, selected callee import, ordered completed formal
slots (including unused arguments), callee-local substitutions, actual return
completion and exact final invocation/equation interval. Construction traces only
locate candidates. Existing ScalarCompute and ordered-source execution remain
the executor model; prepared reverse-AD import closure is a separate obligation.

Numerical arguments must NOT be executed by outlining their whole original CPU
expression. A computed resident prediction needs proven physical-producer/formal
correspondence. MSE's count uses prediction length, not target length or the loss
denominator. Its current original count outliner takes a JVM array: the resident
route requires an authenticated metadata/length projection through existing
allocation/view/stage-completion facts, with null/bounds/domain/failure order
preserved. Downloading or inventing a replacement array is not a valid shortcut.
Acceptance must still rerun unchanged `local-loss` and `local-gradient-sum` with
changed inputs and independent numerical/device oracles; successful scalar frames
alone cannot close campaign items 3 or 5.

The unchanged production `local-loss` probe now independently confirms that
boundary: source compilation succeeds, with 12 actual type-elaboration edges and
five builder observations. `linear-nb` has an authenticated root caller, unchanged
argument occurrences and callee import. `mse-loss` has an authenticated root caller
and callee import, but its numerical prediction argument was actually rewritten,
so the unchanged-argument query correctly abstains. `alloc-like` and two `oftype`
calls have authentic callee imports but no root-normalization origin. These are
hierarchical import/frame and numerical-result correspondence gaps, not evidence
for weakening a flag or adding a root-prefix special case. No device allocation,
numerical execution or training/performance claim follows from this source probe.

### October 21 original reduction count execution and outline hygiene

The shared owned-input outliner now supports an original reduction's WHOLE
observed count initializer, including its checked Int conversion. It reuses the
existing physical-port compilation, context and historical guard machinery; it
does not interpret source or add another numerical IR. Before the bound record
can become executable, the common reduction query checks its sole binding's
exact initializer/binder children and joins its before/after locals to the actual
loop-entry environment. The count outline retains the full operand context and
original element-type context, including an explicit nil. Legacy records missing
those execution inputs refuse rather than acquire defaults.

This count region runs neither accumulator initialization nor iterations. Negative
counts remain negative; out-of-Int Long counts throw as the original checked
conversion does instead of wrapping. Effects in a count run once, and a count
failure precedes the original initializer's effects. Copied/foreign occurrences,
detached initializers/locals, incomplete contexts and stale module history refuse.
Original method-local helper dependencies are conservatively detected from the
actual emitter's sibling tables (including namespace-insensitive typed lookup).
They remain unsupported for this outline rather than silently resolve a same-named
global. This is a denial check, not dispatch, purity or replay permission.

The work also found a shared outliner naming hazard: a fixed generated method
named `input` can capture a source global call of that name. Generated method
names now avoid executable source symbol names, including qualified names and
binders. The final spec and reflected Method use that exact chosen name. This
independent fix is extracted onto main in reviewed PR #1555 (`1ee20648`); its
exact 13 on-disk tests pass with 138 assertions in the reused REPL, and fresh CI
is the required merge gate. It is not a fresh clean-main JVM measurement.

The first fresh-CI run found an adjacent index-sibling fixture hard-coding the
old generated name `input`, rather than consuming its returned method spec.
PR #1555 head `3dc8ff4e` corrects the fixture while preserving direct primitive
INVOKESTATIC, declaring-owner closure and no-reflection/no-boxing assertions.
The corrected index namespace passes 3 tests / 23 assertions in the reused REPL;
the clean-main hygiene declarations still pass 13 / 138. The replacement CI run
is pending; this is not a green-main or merge receipt.

An additional unchanged-stdlib MSE component test emits a fresh diagnostic module
from the original body's exact retained input, rather than restamping an older
REPL implementation. Its count outliner depends only on the actual `pred` array:
lengths 0/1/17 and null behavior pass; neither target nor the mean denominator is
invented as a count dependency. Cross-module source authentication still refuses.
The ownership namespace passes 24 tests / 209 assertions. This is actual count
execution over a public array, NOT full MSE scheduling, resident metadata
projection, or enclosing `local-loss` source correspondence.

The complete broad-branch targeted gate selects current on-disk declarations
across eight ownership/input/AD namespaces, avoiding stale extra test Vars in the
reused REPL: 124 tests / 1,483 assertions, zero failures/errors. Independent review
covered the count/context joins, sibling refusal and shared naming repair.
Memory pressure excluded an additional local JVM; no full-suite or performance
claim follows.

After the real MSE component and index fixture additions, the current on-disk
gate spans nine namespaces: 128 tests / 1,514 assertions, zero failures/errors.
The reused independent reviewer also checked both additions without a blocker;
neither tests nor review establish full MSE scheduling or GPU admission.

Executable count evaluation is NOT yet source-position scheduling or proof that
the chosen launch domain agrees with the original count. The ordered invocation
must execute it once at the original boundary and retain read-capacity,
initialization, association, result-checkpoint and box-use obligations. Public
draft CI at `61b84139` still fails emission job 33524 on the unchanged softmax
`inv-sum` arithmetic-completion gap. This advances items 3/5 without closing that
continuation or the full eight-item campaign.

The reviewed continuation is to reuse InvocationPlan `ScalarCompute`, not add a
count-specific runtime: construct the whole checked count binding before seed
evaluation at the actual reduction transformation; transport the original
occurrence to its count/reduction interval; prepare that exact step through the
owned count executor before generic scalar projection. Keep the signed Int count
distinct from the nonnegative launch extent and qualify read capacity separately.
Do not loosen SourcePrefix's flat-prefix authority to admit later regions.
For unchanged `local-loss`, original helper-call argument/result completion and
substitution intervals must first connect its MSE helper to the final steps:
the original caller contains no reduction. A literal/public-count surrogate or
the component test above cannot close this obligation. Arithmetic denial flags,
Float checkpoints and prepared-AD import checks remain in force.

### October 21 arithmetic normal-completion conservation

A real public helper reproduction exposed a silent source loss: an unused helper
argument containing checked Long multiplication disappeared at `:dce-cleaned`.
The original call throws on overflow, but the resulting program accepted that
input. External purity is not a proof of successful evaluation. The shared
effects descriptor now retains `:unclosed-source-arithmetic` for arithmetic
identified by the existing canonical operation descriptors; DCE, CSE and the
TypedSOAC required-evaluation boundary conserve the obligation. Logical type tags
do not waive it, and quoted arithmetic remains data. This is conservative denial
until a source-owned completion proof is available, not a second type registry.

The repair also makes existing source-prefix correspondence distinguish a shared
scalar ProgramEquation from an invocation-only ScalarCompute. The latter must
retain the same source in both closed region and initializer, or the same central
result materialization in both. Source-order validation independently rebuilds
the entire step. Coherent source/plan mutation still cannot replace the original
source occurrence. The executor uses the actual original Object-returning call
and authenticated non-null Long boxing bridge; a syntactic result `long` wrapper
is not a prerequisite or a replacement for that physical evidence. Exact owner,
Method, implementation, operand, returned-wrapper and historical guards remain.

The existing capped REPL passes six focused namespaces (77 tests / 871 assertions),
the shared inliner namespace (31 / 192), two required-evaluation boundary tests
(2 / 24), and source-owner/invocation-plan namespaces (18 / 83). Original selected-call overflow, changed-input and allocation-order
oracles are retained. The unused-helper argument now survives before allocation
and unsupported lowering refuses rather than silently dropping its exception.
This does not yet provide that imported argument's executable source ownership.
Implicit loop-count conversions, reads, allocations and other pure calls need
their own completion accounting; no general totality or broad CI claim follows.
Draft #1539 at `c41be4d4` still has failed test, OpenCL and public-source emission
contexts, with CUDA/HIP compilation not established. It remains unqualified.

### October 21 Float numerical checkpoint at the terminal host boundary

An unchanged-training recheck after `78a6a4bd` found a missing boundary, not a
reason to drop the new arithmetic obligation: `local-loss`'s retained quotient
needs its numerical Float reduction export, while terminal ports previously
accepted only Double/Long. The four-test diagnostic gate initially reported one
error (`:invocation-terminal-carrier` on that Float result). Its source-emission
oracle is kept unchanged.

Produced terminal ports now additionally support Float. Existing emitted physical
output correspondence, exact dtype/one-element ABI, initialized storage, ordered
completion and source-owner/history checks remain. Download uses a one-element
Float array; exact Float wrapper validation precedes value extraction through
the existing static `RT.uncheckedFloatCast(Object)` adapter. That extraction
does not add a checked narrowing point or permit a Double input. Ordinary core
binary arithmetic over actual Float/Double operands widens Float at the source
call to the existing Numbers Double overload; it never widens the reduction.
Typed scalar Float arithmetic and public Float entry ports remain unsupported
by this rule. Explicit checked source narrowing remains independently checked.

The reused capped REPL passes source-owner, terminal-owner, invocation-plan and
unchanged training diagnostics (47 tests / 476 assertions). These include Float
identity and noncanonical NaN payload, signed zero/infinities, exact wrapper
refusal, checked narrowing overflow, mixed core arithmetic, typed-context refusal
and public Float refusal. Actual OpenCL and Level Zero changed-input replays pass
14 assertions, including exactly one Float-element download and Double host
result comparison with the JVM. The device sums are deliberately association-
exact; they do not claim arbitrary tree/serial Float equivalence. `local-loss`
again emits source; `local-gradient-sum` retains the two existing quotient
completion/count obligations. Resident training and helper-expansion ownership
remain open, and no full-draft green or performance claim follows.

### October 21 generic inlining must preserve dispatch selection

The next helper-origin investigation exposed another silent miscompile. A helper
with only a Long method and a constant Long result correctly rejects a Double
argument in the source program. Ordinary compilation nevertheless selected its
sole body, erased the call, and accepted every tested Double input as an array
program. Unknown arguments and a single registered overload are not evidence of
method applicability.

Ordinary generic inlining now branches on dispatch-table presence before reading
backing metadata. Complete argument tuples use the same selection, priority and
parametric specialization machinery as `try-resolve-call`; the new tag-tuple
facade shares that implementation without fabricating source arguments. Misses,
unknown/partial tuples and unpreserved promotion casts abstain, never fall back
to whichever body exists. Actual selected mangled symbols come from the shared
selector. The actual generic Var and lookup join the existing historical owner;
normal overload publication and namespace-alias remapping invalidate old inputs
even when the imported backing method has not changed. Rejected candidates do
not publish their optional observations.

The duplicated AD resolver is reduced separately. A single-method AD constructor
uses the canonical public `value+grad` method-selection policy BEFORE the returned
gradient receives arguments. That is not the ordinary runtime generic-call rule.
Multi-method AD application selection uses the shared selector; general runtime
parity of that specialization policy is not established here. Argument tags use
the existing inference API rather than a symbol-only lookup. Runtime-versus-
inlined constructor tests preserve actual Float/Double result classes and bits,
argument failures and single evaluation.

The reused capped REPL passes dispatch/inliner/inference tests (60 tests / 328
assertions), canonical dispatch/parametric tests (17 / 130), and source-prefix,
unchanged-training diagnostics and terminal-owner tests (45 / 544). The public
incompatible caller is now explicitly rejected with its undevirtualized callee
still identified; the compatible Long caller compiles and lowers to its expected
seven-element output. This repairs method selection, not formal-completion
proofs: even an applicable overload can require an implicit conversion, and the
already-selected direct-call path still needs the planned expansion transaction.

Exact-head CircleCI at `d86c72ac` remains red. The public emission job #33480
specifically refuses `raster.nn/softmax`'s retained `inv-sum` (`raster.numeric//`,
`:unclosed-source-arithmetic`, normal completion unavailable). That is a real
remaining source-completion obligation exposed by conservation, not an
infrastructure failure and not a reason to waive the flag. Unit and OpenCL gates
also failed; neither CUDA nor HIP compilation is established by that run.

### October 21 dynamic checked-map loop qualification

The checked-map schedule now proves termination/nonoverflow for an actual validated
exclusive Long loop with unit increment and integral leaf bounds. The proof is
local to the exact KernelBody operation occurrence, not copied metadata, inferred
parameter ranges or an arithmetic-expression rewrite. The previous literal-only
proof remains independently available. Negative/reversed bounds are covered by
the header theorem, not claimed as qualified source invocation/read domains; a
finite loop is not necessarily short enough for a device watchdog.

The real dynamic Float-storage/Double-accumulator fixture retains its independent
source read-capacity certificate and conditional private-result initialization on
OpenCL/CUDA/HIP source targets. Both local OpenCL and Level Zero execute zero/one/
three-trip loops and changed-input replays against JVM oracles, including a
cancellation case that distinguishes Double from Float accumulation. An
overflowing final Float conversion prevents publication and poisons execution;
an excessive dynamic count independently fails with `:program-link-graph-range`
before graph submission. Session, graph and event ownership checks remain intact.
The affected three namespaces pass 32 tests / 640 assertions; review found no
blocker and its additional read-capacity control is included.

Unchanged `rms-norm-chunked` now passes its dynamic-loop checked stage, but its
final map remains refused at `:single-output-boundary`: the map domain retains a
lifted scalar extent while output storage projects the checked product of rows
and features. This requires the existing causal extent evidence, not weakened
equality or a norm-specific exception. Internal checked conversions, complete
training/source correspondence, and the broad draft's red CI remain open.

### October 21 causal checked-map volume correspondence

Checked-map emission and independent emitted-equation reconstruction now pass
the retained dependency-closed scheduled body explicitly. When iteration extent
and output elements differ syntactically, the planner reconstructs the complete
source graph, requires its exact match and unique numerical algorithm, then uses
the existing predecessor-only `extent-proof` environment to compare the iteration
extent with the retained physical output shape. It does not manufacture source
arithmetic from KernelLaunch products, trust read-certificate definitions or accept
caller-option assertions. Original scalar evaluations and overflow checks remain.
Callers without the causal body retain exact-equality-only admission.

A product-shaped source map now compiles for OpenCL/CUDA/HIP and executes on both
local backends, with zero-size output, changed-input replay, checked failure and
ownership cleanup. Forged attached definitions, future or malformed predecessor
bodies, changed effects/types and changed output storage remain refused. The
shared extent-proof tests also cover narrowing, wrapping multiplication and
guarded negative domains. The affected four namespaces pass 39 tests / 743
assertions; independent review found no blocker.

Unchanged `rms-norm-chunked` passes the former single-output-boundary refusal and
now reaches a distinct map-body index-scope refusal (`rstr_local_7` undeclared).
That is the next source/lowering investigation, not a qualified RMS execution
claim. Internal checked conversions and the draft's broader red gates remain.

### October 21 preserve the projected checked-map operand

The subsequent index-scope refusal was a concrete cross-pass bug: read-address
projection expanded and pruned certified quotient/remainder locals correctly,
then checked-map planning restored the conversion operand captured before that
projection. The producer now takes both locals and operand from the same
projected region and explicitly requires unchanged outer conversion attributes.
No scalar rule, numerical permission or source correspondence was invented.

The product-shaped map fixture now reconstructs an input coordinate from explicit
quotient/remainder locals. Before the fix it reproduced an undeclared-local
refusal; afterward the existing changed-input/zero-domain/failure-publication
tests execute it on both OpenCL and Level Zero. A mutation control rejects changed
conversion semantics. The five affected address/extent/checked-stage/reporting/
device namespaces pass 55 tests / 800 assertions; review found no blocker.

Unchanged `rms-norm-chunked` now emits three CUDA source kernels. Actual Level Zero
checked instantiation still refuses before device execution at
`:unclosed-ordered-source-executor`, context `:scalar-compute`, for an allocation
extent introduced through `alloc-like`; no original-source correspondence is
available for that imported helper step. This is evidence for the already-open
helper substitution/source-ownership transport, not permission to guess origins
or use a fallback. Full RMS execution and vendor compilation remain unqualified.

### October 21 preserve actual caller occurrences before helper import

The helper-ownership review identified an earlier information loss:
`hoist-nested-vg` reconstructed every ordinary call even when no AD application
was hoisted. It now preserves a source node only when every child is the exact
unchanged occurrence; changed paths retain their existing ordered construction
and metadata. This is not an equality-based origin lookup or execution witness.
The same review found and repaired an existing quote-boundary defect: syntactic
AD applications inside quoted data must never be lifted or executed. Both quote
spellings are opaque at the initializer and nested traversal boundaries.

The inliner, source transport, source-prefix and actual JVM input-completion
namespaces pass 75 tests / 639 assertions. Controls cover unchanged calls/vectors/
scopes, genuine hoists with unchanged siblings, top-level applications and nested
quoted data. Independent review found no blocker.

Next, actual binding expansion must consume the exact caller occurrence, callee
BodyImport and every positional input completion while constructing substitutions
and binding-origin edges together. Existing argument substitution alone is not
formal entry conversion. In particular Object-to-Long completion is not identity;
unused arguments and normal-return completion remain separate obligations. This
prerequisite does not by itself authorize the refused RMS allocation helper.

### October 21 renewed bytecode nominal-interface publication

The broader owned-AD gate exposed a genuine renewal failure: initial emission
published the interface's registry Symbol, but renewal passed its frozen-source
String through unchanged. The actual implementation, Method, parameter/return
signature and namespace checks all passed; only nominal metadata representation
differed. `emit-owned-bytecode!` now publishes a canonical interface Symbol while
leaving source lineage, actual emitter arguments and every physical identity
check unchanged. No AD string-comparison fallback or historical restamping is
introduced. Both nominal input spellings and actual renewal are covered; old
artifacts remain stale. Independent review found no blocker. This repair and the
following operand slice pass the six-namespace affected gate: 72 tests / 1,034
assertions, not a broader draft qualification.

### October 21 retain actual integral call-input completion

Actual RMS allocation-call observations show why an identity-only shortcut is
insufficient: array inputs perform CHECKCAST, direct row counts arrive as Long,
and product count operands return Object before Number-to-Long ABI coercion.
Those are observed emitter operations, not declarations or generated source casts.
The existing owned invocation-input outliner now also supports Int/Long inputs.
It restores physical ports and the complete saved operand context, verifies the
original emitted carrier, then uses the same emitter coercion on method return.
The adaptation classifier is shared with actual typed-call emission. All
positional completion records are checked, including unused other formals;
missing, copied, guarded, malformed or stale evidence remains refused.

Tests distinguish raw Long-to-Int wrapping from explicit checked `int`, retain
Object-to-Long null/wrong-object failures, BigInt and nonfinite Number behavior,
once-only custom Number unboxing and operand-effect order. The existing FP path
is preserved, and its AD consumer now explicitly requires a floating contract:
integral outlining cannot reach floating AD derivation. Review caught and closed
the need to validate the full four-Boolean context rather than inherit outlining
defaults. The six affected lineage/outlining/AD namespaces pass 72 tests / 1,034
assertions; final independent review found no blocker.

This is a JVM-owned operand capability, not authority to move arguments, omit
checks, substitute helper bodies or admit resident execution. The next expansion
transaction still must connect actual caller/callee inputs to its constructed
bindings. Source-owner RMS instantiation remains refused until that closes.
Broad CI at `2bfcda0e` reports 148/172 public fixtures passing and 24 failures
(job 33424), versus the earlier 146/172. Vendor compilation and the other red
unit/OpenCL gates remain unqualified; no failing fixture was skipped.

### October 21 retain actual reference call-input completion

The same owned input outliner now supports an actual recorded CHECKCAST. It
restores the original operand context, verifies its emitted reference carrier,
and uses the typed method return to perform the original reference check. The
outlined method must resolve the identical target Class, not a matching name or
declared dtype. Null and successful reference identity are preserved; wrong array
and object classes fail after the original operand effects, even when the callee
does not use that formal. Complete positional evidence and historical guards
remain required. No floating AD contract is granted to the reference input.

The six affected lineage/outlining/AD namespaces pass 73 tests / 1,055 assertions,
including stale, copied and guarded-evidence controls. Independent review found
no blocker. This closes another prerequisite for ordered helper arguments, not
the helper expansion itself: callee bindings, allocation and return still need
their own source correspondence. Broad draft CI at `e93adb6f` remains red in
test, public source emission and OpenCL; no merge or RMS runtime qualification
is claimed.

### October 21 native replay refusal coverage alignment

The replay-boundary regression now pins the earlier native refusal
`:unclosed-c-family-value-completion`, replacing two stale expectations of the
later unsupported string/throw refusal. It independently requires the retained
completion, original alias-guard message and reason in the refused body, and
checks that native compilation is never invoked. Runtime alias controls remain
unchanged. The namespace passes 9 tests / 65 assertions; independent review found
no blocker. This repairs CI accounting without admitting native replay guards or
claiming a production AD fix. Remaining broad CI failures are still open.

The distinct private-destination guard fixture has the same earlier refusal and
now requires its completion plus original alias message/reason to survive too.
Its existing successful private-write and opaque-alias runtime controls are
unchanged. Both namespaces together pass 14 tests / 85 assertions; review found
no blocker. These four corrected stale assertions are not evidence that the
remaining draft failure clusters below have been resolved.

### October 21 full draft CI failure baseline

The complete test job **33431**, head **e93adb6f**, is substantially broader
evidence than the earlier abbreviated shard inspection: its sixteen failed-shard
summaries total **7,429 tests / 88,348 assertions, 117 FAIL headers and 172 ERROR
headers**. These are that job's logged totals, not a fresh run of later heads.
The reviewed replay-test updates above do not qualify the other failures.

The principal shared boundaries are checked floating conversion in numerical
maps, original source/ordered-host execution, prepared output/donation contracts,
artifact serialization and contextual method ownership. Additional failures
include AD structural completion, counted reductions and stale structural
expectations. Failure locations are diagnostic clusters, not a proof that every
member has the same cause: 43 ERROR headers originate at `segmap_body.clj`, 23 at
`equation_first.clj`, 17 at `method_entry.clj`, 12 at `source_owner.clj`, 12 at
`compiled.clj` and 9 at `equation_artifact.clj`. Keep public workloads and both
positive and negative tests enabled while repairing these mechanisms.

The immediate connected repair remains the actual inliner expansion transaction:
one builder used by binding expansion and `.invk` expansion must consume the
authenticated callee import while constructing argument bindings, substitutions,
callee bindings and results. Retain construction edges there, not after the fact
by generated names. Its caller occurrence and full positional ABI completion
must be authenticated, and source-prefix correspondence must describe ordered
call intervals rather than require an unchanged binding count. Argument-only
evidence authorizes only the original call's ordered argument prefix; callee
allocation/local evaluation and normal return require independent closure before
RMS or training execution is admitted. No weaker fallback or restamped history
is an acceptable route to a green campaign.

### October 21 shared production inline construction

Both binding expansion and recursive `.invk` expansion now use one builder for
ordered argument lifting and callee substitution. The builder retains their
existing flat versus nested placement conventions; sequential local renaming,
allocator metadata and effect/result placement are unchanged. When a BodyImport
is supplied, construction consumes its exact body and authenticated parameters
and tags under its original history. Copied bodies, foreign declarations and
stale inputs fail before substitution. Legacy/generated helpers without imports
do not acquire ownership by passing through this builder.

Review caught and closed two gaps: validating only the body was insufficient
without its declarations, and the recursive path's size/fusability policy needed
the same whole-body convention as construction. The latter also fixes a legacy
multi-form path that previously substituted only the first form. Tests check
both actual entrypoints, once-only ordered arguments, unused checked failure
before later arguments/callee effects, all body effects/results and whole-body
size limits. The five affected inliner/import/prefix/JVM-input namespaces pass
94 tests / 856 assertions; three affected owned-AD namespaces separately pass
33 tests / 524 assertions. Independent final review found no blocker. Unchanged
`rms-norm-chunked` still emits three CUDA source kernels; no vendor compiler or
RMS device execution is established by that source probe.

This intentionally introduces no standalone provenance token. The next connected
slice must retain actual caller/argument/callee construction edges through this
builder, consume them in source-prefix call intervals, and prepare only closed
original argument outlines. Missing callee/allocation/return obligations still
refuse. The broad draft and the original eight completion gates remain open.

### October 21 import-local and checked substitution occurrences

Shared alpha conversion and source qualification now expose optional constructor
observers, separate from resolution-history observation. BodyImport retains only
actual call-node edges in two explicit phase-indexed vectors, using the same
`form/call-form?` classification as bytecode InputOrigins. No lexical environments
are retained. Identity edges remain meaningful within a phase. The guarded query
joins exact qualified → hygienic → original occurrences; copies, repeated originals,
ambiguous outputs and stale source/import history refuse without publishing partial
parent dependencies. Quote opacity, sequential/recursive scope and auxiliary source
bodies remain governed by the existing shared lexical walkers. Normalization uses
the same identity-uniqueness check rather than retaining a duplicate implementation.

The parent-expansion relation is derived, not another passive trace. During the
independent full substitution-law check, the shared substituter internally records
imported-call → replay-call edges. Only after the complete metadata/raw-bit/product
check succeeds does structural-position alignment relate those calls to the exact
recorded builder body. Unique child and replay origins then compose with BodyImport.
Generated names/equal-form searches and caller-supplied edges supply no authority.
Calls introduced solely through argument substitution, including one object used
twice, cannot acquire imported-body origins. The checked parent law/history stays
attached. Locally exhausted substitution environments may still conservatively
abstain; the existing empty-map fast path is not expanded into a second traversal.

The unchanged production `local-loss` now recovers original `alloc-like` and both
`oftype` caller occurrences from the actual `linear-nb` expansion. This closes their
previous missing import-local occurrence relation, not their selected/completed
execution frame. Existing original selected Method/implementation checks, formal
completion, resident prediction/length correspondence and final interval placement
remain required. No host numerical recomputation, download or execution fallback
is introduced.

Independent review found no blocker after ambiguity/staleness controls. Replacing
only the owned capped REPL for the changed private record layout exposed a former
warm-test assumption: the training frontend needed explicit non-evaluating
preparation before its original caller could be compiled. The regression now does
that through the shared service. Reused-process gates after that fresh load pass:
80 lexical/inference/import tests / 463 assertions; 64 transport/inliner/construction
tests / 523 assertions; 136 source/bytecode/AD ownership tests / 1,579 assertions.
The first two gates overlap; these are not additive unique-suite totals.

JVM observations remain an import bridge for faithful existing Clojure semantics,
not a permanent requirement for native TypedSOAC operations with direct semantic
contracts. The next consumer must close and consume actual frames in execution,
then retain compact checked obligations rather than proliferating whole histories.
Preparation, validation and emitted-module costs require separate measurements.
The broad draft CI and all eight campaign completion gates remain open.

### October 21 original selected invocation frames

The checked nested builder now joins an actual original selected call without
executing or outlining any operand. The bytecode emitter's complete positional
conversion validator is shared by the existing input outliner and a new read-only
observer. MethodEntry retains the original whole-module history. SourceTransport
checks actual interface Method, global receiver, registered source/receiver/entry,
implementation identity and parameter Classes against the authenticated callee
import. Matching descriptor spellings or a compatible ABI alone are insufficient.
All composed queries retain optional-owner transaction isolation on refusal.

Each frame slot carries the original source operand, observed evaluation context,
actual emitted carrier/target Class/adaptation and constructed argument/binder.
Constructed-operand correspondence, return adaptation and final placement remain
explicitly unclosed. Array CHECKCAST does not prove a nonnull array or permit
outlining a computed resident prediction. MSE's primitive invokedynamic result is
not confused with its interface's Object result; return correspondence is not
inferred from either carrier.

The unchanged `local-loss` initially had no original invocation contexts because
its CPU implementation was still cold. Explicit non-evaluating preparation of
that original caller, followed by complete source recapture, exposes the actual
`linear-nb`/MSE positional conversions. One loaded-process preparation probe took
15 ms; this is not a cold compiler benchmark or automatic production admission.
The production-source regression checks both frames and MSE's unresolved resident
prediction, while another test forbids input outlining/preparation during queries.
An authentic foreign callee with the same ABI is rejected. Missing/modified unused
completion descriptors also refuse without publishing partial parent dependencies.

Independent review found no blocker in this nonexecuting scope. Reused-REPL gates:
61 transport/inliner/construction tests / 436 assertions; six frontend-input tests /
48 assertions; 136 bytecode/source/AD ownership tests / 1,579 assertions. These
receipts do not qualify the broad draft or complete training execution.

Fresh CI at previous exact head `528d6686` confirms remaining failures rather than
assuming the older baseline unchanged: source job **33648** passes **142/172**
public CUDA/HIP fixtures and reports outstanding checked-floating conversions,
map-address evidence, allocation extents and softmax host scalar closure. Unit job
**33650**, inspected shard zero, has two real `local-gradient-sum` errors in five
tests / 40 assertions, retaining the two known host quotients and original checked
Int reduction-count obligation. This single shard is not an aggregate suite count.
The draft remains non-mergeable; no tests, public fixtures or seven-context gates
are disabled to obtain these targeted results.

### October 21 checked nested helper substitution

The shared builder now retains indexed formal/actual/binder slots for authenticated
nested imports only; ordinary ownerless expansion does not allocate this evidence.
The construction query is factored into exact recorded builder return + imported
body, with the existing root-origin/unchanged-argument query retaining its stricter
requirements. A separate inliner query independently validates each recorded nested
frame's substitution law: exact positional argument identity (including repeats),
ordered once-only lifted initializers, unique noncapturing binders, and replay of
the shared scope-aware substitution over the exact imported body. Significant
metadata and floating bits participate; quoted products retain identity. An arbitrary
builder callback cannot validate itself. No function-name or model registry is added.

Independent review found malformed false-valued binders could hide an unused failing
argument, and wrong-shaped results could throw during comparison. Both now abstain
under explicit shape checks, with regressions for unused checked overflow, repeated
arguments, missing/reordered evaluation, capture, shadowing, quoted products, copied
imports/outputs and faulty callbacks. Complete composed queries use optional-owner
transactions, so a later refusal cannot publish partial imported dependencies.
Identity assertions cover the unchanged parent history on refused root and nested
relations. The three affected source-transport/inliner/construction namespaces pass
60 tests / 413 assertions in the reused capped REPL.

The unchanged production `local-loss` source supplies authentic nested `linear-nb`
and MSE frames that pass this law, even though MSE's rewritten numerical argument
does not satisfy the separate root-only caller relation. The two `oftype` frames
still abstain: their unlifted `(float 1.0)` / `(float 0.0)` arguments use bare cast
names, for which the shared SOURCE constant evaluator supplies no resolution proof.
Canonical constant folding is not substituted for that missing source evidence.

These are local substitution facts, not executable-call certificates. Original
argument/formal completion, selected callee correspondence, resident prediction
length projection, return adaptation, and exact final expanded interval remain
separate obligations. Flat placement remains unqualified. No source execution or
training admission is broadened; draft CI and all eight campaign gates remain open.

### October 21 fresh unchanged training-source checkpoint

Subsequent count-contract investigation found and repaired an actual source-cast
miscompile rather than broadening admission: floating/reference `int` and `long`
must use the Clojure source overload, not raw JVM saturation/ABI coercion. The
independently reviewed clean-main fix is PR #1553 (`a93d6f91`), with 55 tests /
2,056 assertions passing. Draft integration `19d59fe2` preserves the earlier
Float-cast repair; its fresh 14-namespace source-owner, inference/inliner,
native-alias and checked-cast gate passes 198 tests / 2,186 assertions. These
receipts establish source behavior, not a reduction-domain or training proof.

PR #1552 is squash-merged as `c389b458` after all seven contexts passed on exact
head `050e635f`. The draft also adopts main's identical scoped unit APT setup and
bounded OpenCL retries, keeping infrastructure failures separate from compiler
failures. PR #1553 subsequently passed all seven contexts at exact head
`a93d6f91` and squash-merged as `5baa8b30`; this focused source-cast landing
does not qualify the broad draft.

After `ab49dcb6`, a fresh capped REPL compiles the unchanged public `local-loss`
fixture through equation-first to portable OpenCL source, with semantic/scheduled
programs and emitted kernels. This supersedes the earlier first-Long-initializer
refusal as the current next-action evidence. It is not a numerical, device or
complete training acceptance receipt.

The unchanged `local-gradient-sum` still refuses with exactly two retained host
quotients: `clojure.core//` without normal-completion evidence and `raster.numeric//`
with a present Object/known-Double completion. Both retain original source
evaluation. Its Float reduction correspondence remains conditional on the checked
Int source count at its original position, with domain correspondence unproved.
The diagnostics test's stale count of four was observed failing before updating
it to two plus the exact operation set; no production admission was relaxed.
The complete diagnostic namespace, including the new primal source-emission
control, passes four tests / 28 assertions.

Review also identifies the earliest caller-identity seam: immediately after
`retained-frontend-source`, `normalize-source-bindings` rebuilds call nodes, before
`pass-lower`'s additional normalization and inlining. Callee BodyImport alone cannot
authenticate these rewritten caller occurrences. Any subsequent expansion witness
must be produced by those actual reconstruction steps and compose with retained
argument/count/conversion obligations; matching equal forms afterward is not an
alternative. Determine the remaining quotient's exact prerequisite before adding
another transport layer.

The follow-up review recommends closing the bounded reduction contract first,
not starting general caller-occurrence transport merely because that seam exists:
retain the original count conversion and its domain/failure position, the array
read bounds/failure obligations, and the physical Float accumulator payload as
distinct from the Double quotient context. Count evidence alone must not grant
quotient completion. Prepared-AD semantic import closure is a separate obligation;
the body-import witness does not discharge it. Keep both quotient refusals until
their own evidence closes. Resume more general caller transport only where the
current unchanged workload demonstrates that it is the missing prerequisite.

That prerequisite is now explicit from review of the actual count machinery:
InputOrigins relates original frontend input to JVM emission, BodyImport relates
the callee's body transformation, and PrefixCorrespondence covers unchanged flat
prefixes. None recovers the original count site after uniquification, helper
substitution and nested-reduction lifting. The minimal next implementation chain
therefore extends the existing owner-local source-transport machinery:

1. Retain actual occurrence/lexical-scope reconstruction at `util/uniquify*`, for
   both early source normalization and later TypedSOAC normalization.
2. Connect BodyImport to actual helper substitution with all formal completions
   checked independently; initially admit identity completions only.
3. Retain the actual site-to-generated-binding relation in
   `lift-nested-scalar-reductions`. It preserves the reduction node itself, so
   requires no new interpretation or search for equal contents.
4. Reuse ordered scalar-input normalization to bind checked raw count and separate
   nonnegative extent to those exact source occurrences.
5. Compose the retained count/read obligations with physical result context and
   quotient evidence; prepared-AD semantic import closure stays independent.

No global origin registry, generated-name search, post-hoc syntax equality or
metadata safety flag can replace this chain. A diagnostic-only count certificate
would not close the unchanged gradient. The checked-cast repair is complete at
its own source boundary; these reconstruction/composition obligations remain.

The first actual normalization edge is implemented in `bf0e5dae`: the original
uniquifier retains exact constructed occurrences and lexical environments, and
the prefix consumer uses that local witness instead of rematching the normalized
root. Copied/ambiguous occurrences abstain; historical source identity is adopted
by downstream owners. Its 16-namespace gate passed 240 tests / 2,408 assertions,
with two additional artifact portability controls passing nine assertions. This
does not yet connect helper substitution or later reduction lifting.

The next prerequisite records actual typed invocation argument completions in
the existing JVM emitter observation. Three identical emission loops share one
implementation; each observed input retains its exact emitted operand, stack
category, reflected parameter Class and identity/coercion/checkcast branch.
Every formal is represented, including unused ones. Guarded IFn fallbacks supply
no such evidence, and duplicate occurrences remain ambiguous. Identity means
identity AFTER operand emission: explicit casts, effects and Object boxing inside
that operand remain mandatory. This is not Clojure source input conversion and
does not authorize substitution or return conversion. New nested, unused-formal,
physical-port and ambiguity controls pass 12 tests / 109 assertions; the final
six-namespace affected gate passes 62 tests / 481 assertions. The independent
review found no factoring blocker.

Fresh broad-draft CI at `bf0e5dae` is NOT green: emission job 33367 reports 145
of 172 public fixtures passing and 27 failures, including definition-observation
instability (`raster.math/log_m_float-impl`) and portable map/fold-map lowering
refusals. This supersedes describing the draft's remaining vendor gate as a
single known running-variance refusal. No fixture was skipped or permission
relaxed; the full failure inventory and source preparation/lowering repairs
remain required before broad landing.

The preparation failure is now reproduced and repaired at the helper-import
producer, without raising the retry limit. In a fresh capped JVM, three public
softmax-cross-entropy compile attempts exhausted their internal retries across
nine helper receivers. Every captured conflict preserved the executable root and
changed only `:raster.core/jit-spec`: publishing the frozen frontend owner was
invalidating prior observations one helper at a time. Typed-source import now
batches only historically observed, still-unfrozen frontend receivers through
the existing separate-ownership preparation service. Original history is checked
after the entire batch; stale output is discarded and the WHOLE producer retries.
No snapshots are refreshed, numerical bodies invoked, JVM implementations
promoted, or already-frozen frontier callbacks repeated.

The final filtered implementation passes the same cold public reproduction on
its FIRST compile, emitting eight CUDA source kernels; two warm repetitions also
emit eight. This is source-emission evidence, not nvcc or device qualification.
Fresh six-helper fixtures exceed the fixed retry budget and pass in both
observation orders with one rejected producer attempt, unchanged executable
roots, zero numerical calls and bounded warm callback scope. The complete seven
affected namespaces pass 114 tests / 638 assertions. Independent review found no
blocker and required the final fresh-process check, which passed. Portable
map/fold-map and checked-map refusals remain separate open failures.

CI at `244dfb6b` independently confirms the preparation repair: public emission
job 33383 passes 146 of 172 attempts, with 26 remaining refusals; private fixture
generation also refuses a checked conversion. These are not vendor compiler or
device results. Nested scalar-lowering diagnostics now retain only fixed keyword
contract fields (never expressions/bodies), and the existing checked-map gate
identifies its first unmet requirement without changing admission. Representative
unchanged probes distinguish RMS chunked's `:private-result-storage` from the
cooperative Q8 quantizer's `:single-output-boundary`; neither can be fixed by
pretending every checked conversion is one private final store.

Reviewed next conversion order: first a general invariant-scalar motion refinement
exercised by RMSNorm's checked `eps` then `gain-offset` casts. Its guard must mean
the final map actually runs (both rows AND features), with prior reads independently
proved bounded/otherwise-total and conversion order retained. Reuse InvocationPlan
ScalarCompute and the existing owned scalar projector's exact RT.floatCast and
matching-carrier conditionals; do not add a floating launch-expression evaluator.
The derived guard/host step needs its own independently reconstructed source-motion
authority: successful projection alone cannot authorize synthesized work. Test
empty domains with invalid scalars, each/both failures, aliasing, unchanged output
on failure and resident replay. Cooperative lanes must still all reach barriers.
Data-dependent internal casts and batch normalization's multiple running-state
stores remain subsequent general lowering obligations, not exemptions or skips.

One smaller correspondence gap is now closed: a single write-only checked map
may retain the existing `:host-return :effect` representation as well as `:buffer`.
TypedSOAC already validates both against the exact physical destination; the
schedule still requires one closed descriptor, exact write alias, independent
read domains and one final lane-owned store. Returning Void does not discharge
output privacy or status observation. The new private-allocation/helper/return
composition executes through the existing EXPLICIT checked executor on OpenCL
and Level Zero: 94 assertions cover empty domains (including an otherwise invalid
coefficient), changed-input replay, overflow poisoning, no failed publication and
session closure. Ordinary compiled/lower remains deliberately refused. The three
complete affected namespaces pass 29 tests / 534 assertions. This does not close
RMS's subsequent loop/read obligations, multiple destinations, or internal casts.

### October 21 ordered host-integral qualification

Late checked integral computations may now remain opaque ordered host actions
when their dependencies close over explicitly identified public integral scalar
parameters and earlier such actions. Local type hints cannot seed that closure;
every binder shadows the previous identity. Retained expression/result types must
agree, and array reads, lengths, parallel results and unclosed source obligations
do not acquire this admission. Classification is shared with diagnostics. This
does not move checks earlier, establish normal completion, or insert them into
device ScalarIR. The existing source owner and actual-carrier executor still have
to close each action before session capture.

The executor projects checked Int narrowing and integral widening only from
actual primitive Integer/Long carriers. Wrapping `unchecked-int`, unknown Object
ports and floating inputs refuse rather than borrowing checked semantics. Actual
plain Clojure, typed-bytecode and projected oracles agree on supported values and
overflow exception classes with the pinned dependency; no extra backend-origin
parameter or type registry is needed.

Three complete affected namespaces pass 39 tests / 445 assertions, plus the
frontend closure test's 12 assertions. Actual OpenCL and Level Zero checks cover
valid, empty, negative and overflowing dimensions, transitive Int results, prior
resident writes before host failure, changed-input replay, no suffix allocation
after failure, replay poisoning and event/buffer retirement. Public host arrays
are not implicitly copied back. Negative controls cover absent public provenance,
shadowing, forged carriers and missing owners before session/materialization.
This is a bounded step-2 prerequisite, not qualification of the MSE quotient,
selected helper payloads, prepared-AD imports or unchanged training workload.

The subsequent emission triage reproduces one fixture-contract mismatch:
`raster.linalg.contract/contract-mm` declares Double arrays but its vendor gate
requested Float source specialization. That fixture now requests Double; existing
generic Float contractions retain their separate gates. The production mismatch
refusal is unchanged. A retained-frontend namespace gate passes 6 tests / 33
assertions, including both hardware-free CUDA/HIP emissions, all three Double
pointer slots and rejection of the Float request. Vendor compilation and runtime
qualification remain separate; the other aggregate emission refusals remain open.

The next execution repair reuses the original source executor for flat eager prefixes
that already possess an ordered source owner. It checks the entire executor set
before materialization, executes each invocation occurrence once and supplies
shared host equations from those completed results instead of reinterpreting
their source. Missing selected-helper executors refuse before any earlier source
action; there is no typed-interpreter retry. Copied owners and stale histories
also refuse before materialization. Three affected namespaces pass 41 tests /
451 assertions; after adding the historical negative controls, the complete
scalar-source namespace passes 14 tests / 101 assertions. Local device and the
existing ownerless unchecked-prefix controls are included in the former gate.

A wider compatibility gate caught the aggregate/physical-entry mismatch before
landing. Projection tags cannot manufacture the original field-read/unboxing
context. Evaluator-family selection is now explicit before evaluation: validated
aggregate projections keep their existing reference path, with present-owner
historical guards; flat owned plans use their prepared source executors. No failed
executor can switch families. Six affected flat/record/tree namespaces then pass
54 tests / 576 assertions, including CSR on both local GPU backends. Fabricated
projection metadata must still fail exact owner identity before evaluation.
The final complete scalar-source namespace passes 14 tests / 102 assertions after
adding that exact owner-identity control with a structurally valid projection.

That reuse repair alone did not seal eager evaluator selection: removal of an
optional owner could not distinguish an intentionally ownerless reference program
from an edited artifact. The subsequent producer-owned eager authority now seals
both existing evaluator families. It binds exact semantic/emitted identities,
the original ordered owner (including nil), original namespace and collected
definition history. The lowerer uses the sealed mode and namespace, not mutable
policy or namespace metadata. Missing frontend provenance refuses at the producer;
no invocation-time reconstruction or interpreter retry supplies a missing seal.

The requirement is independently derived from validated sourceful host-evaluation
obligations. Removing the seal, ordered owner and namespace together cannot opt
out. Genuine source-free programs keep their separate contract; numerical-only
artifacts do not acquire a process-local owner. Archival explicitly refuses a
supplied eager owner instead of serializing or silently stripping it. The existing
aggregate/reference family remains producer-authorized; this seals selection and
history, not the still-unproved aggregate physical-entry/coercion correspondence.

A fresh bounded REPL passes six complete flat/record/tree/device namespaces (55
tests / 587 assertions), plus three focused numerical-only archival/source-free
controls (23 assertions). Final ownerless-reference, foreign-source-free-owner
and archival controls pass 3 tests / 25 assertions. Tamper cases cover removed or
copied seals, removed ordered owner, changed/nil/removed namespace and fabricated
projection metadata, before materialization. Removing a defrecord basis field
correctly fails the compilation-type guard first. Independent review found no
blocker. The complete archival namespace remains red on process-owner reconstruction
and dependent persistent-evidence expectations; it is not qualified by the focused
controls. Selected dimension helper execution, MSE consumers, prepared AD imports
and unchanged training acceptance remain separate open obligations.

The next reviewed execution boundary is original call-result completion, not
another arithmetic/type registry. A non-emitting REPL query confirms the selected
Long product's actual interface is `Object invk(long,long)`. Exact primitive input
ports therefore do not establish a primitive Long output. The existing invocation
observer can authenticate Method/receiver/ordered input contexts, but the retained
return-bridge proof is FP-only. Reuse exact original binding/input occurrence
evidence and shared value-completion machinery for the result adaptation; do not
insert `.longValue` from a Long annotation or turn an Object payload into its
parent's source carrier. Keep cold queries non-emitting and abstaining when their
original context is unavailable. Foreign receivers/methods, arity/carrier mismatch,
nullable/Object results, checked overflow/order and stale histories are required
negative controls. Do not rewrite helper bodies universally: AD/native consumers
must not silently become warm-only or acquire unsupported reserved terms. This
review identifies the next implementation boundary; it does not qualify execution.

### October 21 shared frontend-to-emission origins

One private input-origin owner now relates original calls, lexical binding scopes
and reductions to their actual emitted occurrences. It seals the exact input and
output bodies, keeps separate immutable identity pairs per family and refuses
duplicate occurrences of the same original object. Equal-but-distinct forms stay
distinct; quotes remain opaque. The reduction-only owner/key are retired, not
retained as a second compatibility path. All three observers delegate to existing
physical bytecode validators. The shared frontend boundary authenticates the
original selection and whole historical emission; editable receiver/definition
maps cannot supply another implementation's evidence. No query emits or executes
code, reparses source, grants an unboxing or proves numerical admission.

The origin gate passes 10 tests / 95 assertions in the bounded REPL, covering
actual Object results despite Long hints, family isolation, exact body identity,
clones, ambiguity, quotes, foreign selections, stale histories and cold queries.
Independent implementation review found no blocker. A wider four-namespace run
first reports one cold selected-host fixture error (`:unclosed-jvm-source-owner`);
the isolated test and warm rerun pass, with the latter reporting 52 tests / 538
assertions. That warm result does not qualify the initial cold failure, which
remains under investigation. Original selected-call execution and unchanged
training acceptance remain open; this is correspondence infrastructure only.

The fresh diagnostic confirms the cold fixture's optional selected Method was
absent while its source policy was valid. That host-execution test no longer
borrows the global numeric operation's dispatch table: it authenticates its own
registered fixture operation, retaining the deliberately misleading nominal
metadata and all original receiver, failure-order and no-redispatch controls.
The same gate order in a fresh 900 MiB REPL then passes the origin namespace
(10 tests / 95 assertions) and all four physical-observation namespaces (52 tests /
539 assertions). No production guard is weakened or extra warm-up lookup added.
The existing ordered-source namespace separately passes 15 tests / 112 assertions.

One producer-owned flat-prefix correspondence transaction is now implemented at
`equation-first/compile`, spanning the authenticated frontend, source normalization,
pre-SOA passes and constructed invocation steps. It validates the complete bounded
evaluated prefix under explicit capture-avoiding renaming, including operand order,
preceding initializers, literal bits and metadata obligations, then binds exact
original scope/ordinal/call and exact constructed `ScalarCompute` objects. Normalized
binding IDs or matching printed expressions alone cannot authenticate an original
bytecode context. Transformed/helper-expanded prefixes, nested scopes, originally
shadowed parameters and aggregate projections abstain. Retained declarations may
elaborate lexical-reference tags; only the existing canonical core allocator table
owns added allocation-result tags. Neither is primitive carrier evidence.

The correspondence stays process-local inside the existing ordered source owner;
it is not another cache, registry, interpreter or portable IR annotation. The first
consumer enriches a missing-executor refusal with its authenticated original source.
Execution remains refused: ownership of the original occurrence does not prove its
Method, operand completion, Object payload, result adaptation or numerical legality.
Existing source-free paths remain independent and no fallback gains admission.

Independent review found a capture counterexample where an earlier initializer
reads a global whose name is a later binder. The verifier now collects free names
sequentially, not by subtracting all binders up front. Exact future-binding and
self-initializer counterexamples, capture/shadowing controls and unchanged-name
positives exercise that repair. Re-review found no remaining blocker. A fresh
900 MiB REPL passes the prefix, ordered-source and original-frontend namespaces
(31 tests / 262 assertions), plus existing checked-source-owner/native-alias gates
(33 tests / 210 assertions) and two focused source-free controls (20 assertions).
The broad draft remains red in CI, including checked floating conversion emission;
these bounded receipts do not qualify it or the unchanged training workload.

The actual small selected-product probe demonstrates two distinct obligations:
the original initializer is an Object-returning `invk`; the normalized scalar
region additionally carries `invocation-source/scalar-equation-source`'s explicit
Long result-materialization conversion. `scalar-result` and all pre-SOA passes
retain the original product computation in this probe; do not blame them for
inventing a widening or collapse the result ABI conversion into original source.
An unchanged training recheck still refuses forward `local-loss` at the missing
ordered executor and `local-gradient-sum` at direct semantic coverage. Neither
workload is qualified by the ownership and physical-observation gates above.

### October 21 checked integral source conversions

Tracing reduction count obligations exposed a JVM source-cast miscompile: an
explicit `int` or `long` of an out-of-range floating operand used a raw JVM
coercion and silently saturated instead of throwing. Source emission now chooses
the Clojure runtime overload from the actual emitted operand carrier. Integral
identity/widening and checked Long-to-Int narrowing keep their existing behavior;
Boolean uses the Object overload, void completes its effects then supplies nil,
and divergent operands remain divergent. Operand evaluation occurs once, including
when the result is discarded. Internal ABI coercions and unchecked casts are
unchanged. No source-execution/projection or GPU admission is broadened.

Independent review found no blocker. The serial affected five-namespace gate in
the capped REPL passes 55 tests / 2,056 assertions, including actual primitive and
Object overloads, reflection carrier/input validation, exceptional values,
public invocation/AOT parity and the existing ownership/effect controls. The
original `par/reduce` oracle confirms checked count conversion before initializer
effects and array reads, with negative/NaN zero trips and out-of-range failure.
The low-level matrix remains independent of full pipeline loading; public AOT
checks reuse the existing checked-casts fixture/integration namespace.

This repairs source behavior needed by the reduction contract; it does not prove
count-to-extent correspondence, read bounds, quotient completion, prepared-AD
import closure or end-to-end training. Those original obligations remain open.
Full exact-head CI, including vendor compile gates, remains required before merge.

### October 21 selective definition preparation

The existing producer preparation service accepts an optional ordered subset of
already-observed receiver Vars. Selection is checked against the original owner
before callbacks; duplicates collapse by Var identity, warm receivers are skipped
and an explicit empty subset prepares nothing. Default all-pending behavior is
unchanged. A separate compilation owner prepares the selected batch; the original
historical guard then rejects a promoted enclosing attempt so the whole producer
retries. No query emits code, no observations are refreshed and no numerical body
executes. This permits callee-before-caller preparation without promoting unrelated
imports. It does not yet enable selected scalar execution or relax any carrier,
source-policy or completion check.

Independent review found no blocker. The exact new function and complete affected
clean-main test namespace pass 9 tests / 36 assertions loaded into the existing
capped campaign REPL. This is bounded loaded-code evidence, not a fresh clean-main
JVM gate; the PR still requires full exact-head CircleCI qualification. Tests cover
requested order, duplicate collapse, warm reuse, no numerical execution, unchanged
unrequested pending inputs, empty selection, pre-callback refusal, nested ownership
and whole-producer retry after promotion.

### October 21 retained physical-observation dependencies

Successful original-frontend call, binding and reduction observations now retain
the exact authenticated emission's historical guard in their enclosing producer.
The existing emission guard owns recursive static dependencies and nested producer
guards; it is not reconstructed from the returned physical observation. The shared
query boundary commits authentication, observation and retention atomically through
the existing optional-proof transaction. Refused queries leave the original parent
collector unchanged. Repeated queries deduplicate by exact emission identity inside
that owner, without a global registry or shape/source-equality cache.

This closes a freshness gap, not a semantic import-closure or execution gap.
Emission maps do not currently publish preparation's source-body/unclosed-import
obligations. Those still require their original preparation owner; a physical
observation does not grant selected scalar execution, primitive unboxing, helper
purity or prepared-AD closure. The training acceptance threshold is unchanged.

Independent review found no blocker. Loaded-code checks in the reused capped REPL
pass 30 tests / 178 assertions for frontend correspondence and definition
preparation, plus 51 tests / 348 assertions for prefix/source owners and native
aliasing. The transitive-history regression mutates a real emitted helper absent
from the parent definition snapshot index and confirms rejection after the query
returns. Existing copied/foreign/cold occurrence controls remain passing. These
are affected loaded-code receipts, not fresh-JVM or full-CI qualification.

### October 21 explicit owned-definition renewal

The existing preparation algorithm now serves two explicit entry points:
pending-only preparation retains its default/all-pending and warm-skip behavior;
owned-definition preparation requires an explicit ordered vector and asks the
existing bytecode owner to validate or renew each selected receiver, including
warm ones. Both validate the entire selection against original observations
before callbacks, compile under separate ownership and check the original owner
afterward. A changed batch requires a fresh whole-producer attempt. Neither API
refreshes observations or acquires physical selection/execution authority.

Actual emitter evidence refines the lifecycle: promoting a callee does not always
invalidate an inlined caller. Current caller emissions remain identical. When a
real retained helper-source dependency changes, the owner emits a new caller
implementation from frozen source and the old emission guard rejects. Preparation
must precede importing occurrence guards; an already-stale enclosing owner is
rejected, not healed. Source-prefix execution remains a subsequent obligation.

The reused capped REPL passes the complete affected 34 tests / 233 assertions,
plus ordered/prefix/checked-source gates (30 tests / 208 assertions). The actual
emitter lifecycle regression uses isolated fresh declarations and also passes two
consecutive reruns without namespace reload (60 assertions), with no numerical
body execution. These are loaded-code receipts, not full-CI qualification. The
selective-preparation prerequisite PR's unit job 33178 failed during apt repository
installation (git-core PPA HTTP 503), before tests; this is an infrastructure
failure and does not qualify the PR for merge.

### October 21 source-prefix producer preparation

Equation-first compilation now explicitly prepares eligible original prefix calls
after correspondence closes and before importing physical occurrence guards.
The existing frontend-selection validator authenticates the caller; signature-only
queries establish primitive Long public ports and exactly registered callees.
This first subset accepts direct original `invk` initializers with those unchanged
public symbols or Long literals. Nested expressions, local/renamed operands,
reference formals, recursive receiver selection and unowned prefixes abstain.
All eligible selections close before the first callback. Preparation runs callees
in original occurrence order, then their actual caller, through the shared owned
renewal service. Changed snapshots escape to the whole-compilation retry.

Correspondence, call and binding queries still perform no preparation. The new
phase grants no selected-call execution, result unboxing or purity: the selected
product fixture still refuses lowering at its missing original-source executor.
Independent review found no blocker. The reused REPL passes 8 prefix tests / 71
assertions and 74 ordered/source-owner/preparation tests / 456 assertions. An
actual nested-argument control retains valid prefix correspondence but produces
zero callbacks and no retained speculative selections. The isolated emitter
lifecycle tests above supply the separate cold/renewal evidence; accepting a warm
or cold preparation count alone does not prove a cold path ran. Fresh-JVM/full-CI
qualification and unchanged real training acceptance remain outstanding.

### October 21 original implementation return observation

Original-frontend return bridges now use the same authenticated whole-emission
observation transaction as calls, bindings and reductions. A successful bridge
query retains the module's full historical dependency guard; a cold, copied,
edited or ownerless query abstains without compilation or speculative retention.
The interface result remains Object, separately from the conditional primitive
payload observed at its actual boxing edge. No unboxing, purity, totality or
selected-call execution is granted.

Independent review found no blocker. A single freshly started 900 MiB REPL passes
the accumulated prefix, preparation, frontend correspondence, ordered/scalar
source-owner and native-alias gates (103 tests / 720 assertions). A subsequent
focused test strengthens the bridge-first/empty-owner control (1 test / 10
assertions). This is fresh local affected-code evidence, not a whole-suite/vendor
CI or real-training acceptance claim.

The reviewed next executor must consume the original direct initializer, its
actual invocation result carrier and matching binding completion, the callee's
owned return bridge and the independently validated Long materialization. An
Object-returning interface alone is insufficient: actual invokedynamic may return
a primitive, and a source binding may unbox. Preserve the actual receiver/Method,
original argument order and returned reference, plus truthful Object/payload facts
and caller/callee historical guards. Unknown inputs or missing correspondence
continue to refuse; prepared-AD import closure remains a separate obligation.

### October 21 original selected-call execution and constructor order

The source owner now consumes that observation for the bounded direct-global
Long-input/boxed-Long-result subset. It validates the original caller invocation
and binding completion against the independently selected callee and its owned
return bridge. Execution calls the captured actual interface Method, preserving
argument order and the returned Object reference; the conditional Long payload
does not relabel the physical result as primitive. Missing, stale or inconsistent
observations, nested operands and unsupported carriers still refuse before any
source computation. No new function/type registry or `compute_static` shortcut is
introduced. This closes a real prefix executor, not the helper-expanded MSE or
prepared-AD import obligations.

Qualification exposed a distinct eager-lowering defect: a source array constructor
can have no invocation-plan step, so iterating steps alone skipped its constructor
count check. Pure specialization now walks retained source order and reuses the
ordered runtime's count/shape validation. Negative counts fail before later scalar
work. Wrapping constructor counts that cannot match retained mathematical extents
refuse rather than producing an incorrect native shape. This creates no storage,
acquisition acknowledgement, initialization or completion evidence. Used host or
realized state is rejected, and terminal materialization cannot execute unconsumed
scalar/shape computations after bypassing their source positions.

Independent review found no remaining blocker in the executor or count repair.
The reused-REPL prefix/shared ordered-runtime/materialization/source-order gate
passes 53 tests / 490 assertions. A freshly started capped REPL then passes all
eleven affected prefix, preparation, source-owner, native-alias and shared ordered
runtime/materialization/source-order namespaces: 148 tests / 1,142 assertions.
Both real local devices are available. The new OpenCL and Level Zero tests pass
2 tests / 36 assertions: changed input sizes 0/1/3, two replays each, independent
JVM outputs, zero source calls while binding, exactly one per replay and closed
sessions. No device test was skipped. These bounded receipts do not qualify the
broad draft's unrelated CI or unchanged-training refusals.

The complete affected ordered-root namespace subsequently passes 4 tests / 462
assertions on these real devices, covering existing source allocation, failure
ordering and replay controls as well as the new selected-prefix tests.

An unchanged training recheck confirms that `local-loss` still refuses its first
helper-expanded scalar occurrence: `rows * 1` imported from `linear-nb`. The final
ScalarCompute contains a logical Long conversion around the selected call, but
has no original-root prefix correspondence. `local-gradient-sum` separately
refuses two retained source evaluations (core division and numeric division),
while retaining its conditional Float reduction correspondence and unproved
source-count-domain obligation. These are failures, not qualified training results.

### October 21 source qualification history prerequisite

The existing body qualifier now accepts an optional observer for its actual
symbol, namespace and alias resolution decisions. Inliner body qualification and
its adjacent nonprimitive formal-tag qualification both retain these decisions
in the existing compiler owner. Resolved Vars reuse definition snapshots at the
original lookup namespace/symbol. Classes, misses and namespace/alias identities
use private immutable producer guards, including detached/replaced namespace
targets. Optional refused queries discard these observations; no second symbol
scan, whole-namespace snapshot or executable registry is introduced.

Review exposed a real qualification defect: an alias can have the same spelling
as an already loaded global namespace. Qualification must resolve in the original
source namespace first. It now preserves that alias target, including a missing
member: the latter must remain unresolved instead of becoming a successful call
in the unrelated global namespace. Independent source-versus-consumer eval tests
cover both cases. Observer failures propagate outside lookup exception handling.

Independent review finds no blocker. Focused controls include Class/static method
owners, structural `new`/`catch`, namespace recreation, changed aliases/imports,
lexical/quoted opacity and actual inliner formal-tag import invalidation. The
completed affected qualifier/inliner, preparation, prefix, retained-frontend,
source-owner and native-alias gate passes 167 tests / 1,060 assertions in the
capped REPL with freshly loaded production classes. This retains lookup history;
it grants no formal argument completion, helper-expansion correspondence or source
execution. Compile-overhead and full exact-head CI remain separate qualification.

PR #1550 is squash-merged as `63a5a97c` after all seven exact-head CircleCI gates
passed. PR #1552, the independent observation-checked explicit preparation API,
now targets main at `1ee2fb0b` and requires its own new seven-gate run. The broad
draft's emission job `33256` still refuses the known checked Double→Float
running-variance update conversion; it remains unqualified and must not merge.

### October 21 retained helper-body import

The inliner now uses one shared source-transport transformation for its existing
multi-form wrapping, alpha-renaming and qualification. An authentic helper frontend
and active compilation owner retain the exact input/output body objects, ordered
metadata-preserving binder renamings and import-local dependency history. Legacy
ownerless transformations remain ownerless; a copied body cannot reconstruct a
witness. An unrelated enclosing dependency does not contaminate an independently
adopted helper import.

The imported history includes a lightweight authenticated source-entry identity
guard, not the intermediate ASTs. This catches equal-but-copied raw body metadata
even when ordinary metadata equality and direct Var snapshots still agree. The
downstream publication test checks that refusal without querying the import again.
Queries never rerun qualification or refresh historical observations.

Independent review finds no blocker. A fresh capped REPL passes the focused
namespace: 12 tests / 47 assertions. Restarting the previous REPL was necessary
because reloaded protocol and implementation classes had different interface
identities; production guards were not relaxed. The complete 12-namespace affected
gate passes 179 tests / 1,107 assertions in that same fresh REPL, covering source
qualification, inference/inlining, preparation, ordered source owners and native
alias execution. Full draft CI and end-to-end training remain separate gates.

This is body-import evidence only. It supplies no formal argument completion,
call substitution, final-step correspondence or permission to execute an imported
body. Unchanged training and the draft's checked floating-conversion refusal
remain open. The next transport below must consume this relation at actual call
expansion rather than reconstructing it later.

PR #1552's previous head passed six CI gates but its unit gate failed before tests
on two shards after Git PPA HTTP 503 retries. Commit `050e635f` scopes unit APT
prerequisites to the image's main source list and makes update failures fatal.
All seven contexts must pass anew on that exact head before merging.

### October 21 actual scoped normalization reconstruction

The shared scope uniquifier now optionally observes its actual reconstruction:
original node, constructed node, visible lexical renaming and bound names. The
callback cannot replace the result; independent invocations do not inherit it,
and both quoted-data spellings remain opaque. Existing normalization arities keep
their output contract and allocate no observation events when tracing is disabled.

Source transport reuses one guarded binder allocator and local-history transaction
for helper import and authenticated frontend normalization. Its owner-local
normalization witness requires unique original/constructed identities. Copied,
ambiguous, ownerless or malformed queries supply no authority. Queries adopt the
original lightweight source identity/history; they do not rerun normalization or
refresh dependencies. This records a transformation, not type inference or
permission to execute, differentiate, move or erase source work.

Equation-first keeps the witness local and the prefix verifier consumes it instead
of re-matching the normalized root structurally. Existing independent represented
source/final-step checks remain. The internal `source-prefix/close` now requires
this witness; there is no old matching fallback. Public `deftm` semantics are
unchanged. Prefix queries into a fresh publication owner also retain exact source
identity. No opaque token is added to options, IR metadata or portable artifacts.

Independent review finds no blocker. The reused capped REPL's complete affected
16-namespace gate passes 240 tests / 2,408 assertions, including unchanged training
diagnostics. Two additional artifact controls pass 9 assertions: numerical-only
compilation still seals/encodes/decodes/opens and lowers equivalently, while
process-local source owners remain explicitly refused. Ambiguity controls exercise
both duplicate output and one original paired with distinct outputs.

This closes the first reconstruction seam only. Helper argument completion and
substitution, later normalization/lifting, original count-to-extent correspondence,
read obligations, quotient execution and prepared-AD closure still require the
remaining chain above. Neither this receipt nor green focused tests qualifies the
broad draft for merge. PR #1553 is now squash-merged as `5baa8b30` after all seven
exact-head contexts passed; the checked source-cast prerequisite is on main.

### Next producer-owned helper-expansion transport

Independent review of the actual inliner locates the missing relation:
`resolve-walked-body-input` owns the callee's authentic frontend transiently, then
returns a hygienic/qualified body without retaining its owner or renaming recipe.
`inline-invk` and the binding inliner apply call-by-value substitution, but that
mechanism preserves actual argument types, not formal argument completion.
Primitive/array symbols may be directly substituted, including unused formals.
Neither generated binder names nor structural expression equality recover the
original call's semantics after this boundary.

The next slice must retain an owner-local expansion transaction at construction:

1. Keep the exact original caller occurrence, selected callee frontend and actual
   hygienic renaming/substitution recipe. Compose nested witnesses through their
   immediate parent rather than globally searching for matching expressions.
2. Reuse shared scope utilities, argument-substitution mechanics and ordered-call
   contracts, while checking argument order, single evaluation and **all** formal
   completions independently. Initially admit identity-completion cases only;
   conversions and unsupported/effectful arguments remain explicit obligations.
3. Validate the constructed expansion before connecting exact final invocation
   steps to original callee occurrences through the existing prefix authority.
   Carry caller/callee historical guards in the existing observation collector;
   do not add an executable registry, mutable public form certificate or hash
   match that can authorize copied source.
4. First qualify diagnostic-only transport with unused failing conversion,
   reordered/duplicated arguments, shadowed locals, copied expansion and stale
   callee counterexamples. Then use the existing source executor/count path for
   execution and rerun unchanged `local-loss` and training composition oracles.

This is one staged relation for ordinary helper composition, not a special
`linear-nb`/MSE rewrite, a permission to simplify checked multiplication by one,
or a relaxation of prepared-AD import closure. Real training acceptance remains
the gate; diagnostic provenance alone does not close campaign items 3 or 5.

PR #1551 (bounded apt acquisition retries, without softening failed prerequisites)
is squash-merged as `4bb40cb1` after all seven exact-head CircleCI contexts passed.
PR #1550 now includes that prerequisite at head `0c0af56f`; its new CI run must be
qualified independently of the old head's six successes and pre-test HTTP failure.

### October 21 explicit owned-definition preparation

The preparation service now shares one observation-checked batch implementation
between pending-only preparation and an explicit owned-receiver API. The latter
accepts an ordered vector of already-observed Vars and calls the supplied
non-evaluating compiler preparation callback for warm receivers too. This lets
the existing bytecode owner decide whether a current emission can be reused or a
new emission is required after a callee promotion; it does not itself renew code.
The original enclosing observations are checked before and after the whole batch,
so changed inputs require a whole-producer retry rather than restamping old code.
No callback runs until the entire selection has been validated.

Independent review finds no broad-draft dependency or pending-only behavior
change. The exact clean-branch functions and tests pass 12 tests / 61 assertions
in an isolated namespace in the existing capped REPL. Simulated callbacks qualify
selection, ordering, empty/warm behavior, ownership and retry; they do not prove
actual bytecode renewal, source correspondence or execution admission. Fresh
clean-branch qualification also passes 12 tests / 61 assertions in a 256 MiB JVM.
Exact-head CI qualification remains a separate gate.

### October 21 primitive boxing-edge correspondence

The existing emitter-owned return bridge now records audited primitive Long
boxing as well as Float/Double boxing. Its source result carrier remains Object;
the observed static return supplies a separate conditional payload carrier. The
query checks actual interface/static Methods, signatures, original body/context
and historical guards. It does not invoke the helper, force the JIT, prove normal
completion or authorize unboxing, source motion, FP recipes or kernel admission.
Missing, duplicate, foreign, legacy and inconsistent observations abstain, as do
hidden Object-to-primitive return adaptations.

Review caught an FP consumer's implicit assumption that every bridge formal was
FP. The shared FP argument-completion predicate now explicitly checks its target
domain. A mixed Float/Long helper with an unused Long formal produces a legitimate
Float boxing observation but cannot acquire an FP recipe by omitting the integral
argument conversion. That regression had two failing assertions before the guard
and passes afterward. Four complete affected namespaces pass 37 tests / 349
assertions in one bounded REPL, including Long checked overflow and unchanged FP,
invocation, binding and input-completion controls. This independent foundation
does not qualify the separate source-ownership draft or its training workload.

PR #1549's first CI head passed setup, build, unit tests and both vendor compiler
gates. The OpenCL gate did not qualify: one shard failed installing dependencies
because the git-core Ubuntu PPA returned HTTP 503. That is infrastructure evidence,
not an OpenCL pass or a compiler regression. A new exact-head qualification is
required before merge; successful contexts on the earlier head are not reused.

The retry at `fbdcf5e1` passed all seven exact-head CircleCI contexts. PR #1549
was squash-merged as `70e643be` and integrated into the campaign branch. The
unrelated larger ownership draft remains held; the foundation's greens are not
its qualification.

### October 21 primitive math completion

Eight audited primitive FP math rows (sqrt, floor, sin, cos, tan, exp, log and
pow) now declare nonthrowing execution after operand completion. This facet does
not prove a source callee, Object unboxing, operand completion, target-library
accuracy or a final checked storage conversion. Descriptor factories stay
unchanged so future math operations do not inherit the grant automatically.
KernelBody checks intrinsic arity before analyses requiring nonempty operands;
malformed empty unary and binary operations receive a structured refusal.

The clean-main 900 MiB REPL passes four complete affected namespaces (107 tests /
1,364 assertions), covering malformed operands, actual physical return payloads,
scalar-region lowering and KernelBody validation. These are local correctness
receipts, not vendor runtime, performance or broader draft qualification. The
separate draft's checked-map device cases additionally exercise final Float
overflow and publication; that machinery is not part of this independent slice.

The independent math slice landed as PR #1548 (`ba675571`) after all seven
CircleCI contexts passed its reviewed head. Fusion robustness landed as PR #1547
(`97ea226b`) under the same exact-head gate. Neither green slice qualifies the
broader source-ownership draft.

### October 21 independent fusion robustness

Optional map-to-reduction/scan fusion now retains the original materialized
program when the exact substituted element lacks a fresh effect proof. Only the
local `*-element-impure-or-unknown` refusal with that same request identity may
abstain; malformed conversion, invalid monoid, stale history and unrelated errors
still propagate. A failed candidate cannot mutate IR or stop later legal fusion;
only successful rewrites contribute to counters and placement decisions.

The bounded clean-main REPL passes four complete affected namespaces (81 tests /
507 assertions). New controls use already-supported Double programs, first prove
normal fusion, then inject unavailable analysis only for the substituted element.
They cover preserved source values, later candidates, scans and strict error
propagation. This is proof-failure robustness, not new IEEE conversion admission
or a kernel-performance result. The separate source-ownership draft's concrete
IEEE conversion case and its wider qualification remain outside this landing.

### October 20 active continuation

The original eight obligations below remain open. Execute bounded, reviewed
verticals and land exact-head green PRs without waiting for another approval;
keep one memory-bounded REPL and delegate full suites/vendor compilation to CI.

#### October 21 qualification-first adjustment

The eight obligations remain unchanged. Broader inspection of draft CI found
multiple shared failure categories, not just the four training-scalar refusals:
checked floating conversion realization, late source actions, process-local owner
archival, normalized AD invocation contexts and source/JIT lifecycle. The review
does not establish that every CI failure belongs to those categories. Freeze new
admission families and repair these production boundaries before extending the
helper-payload projection further.

The fixed-head `fe439ca4` unit report across sixteen failed shards totals 7,306
tests / 87,115 assertions, with 124 failures and 156 errors. Group by the first
structured refusal and its producer, not the generic validator stack line or
workload name. Some old positive fusion fixtures expect IEEE behavior from a
checked source `float`; they require explicit numerical intent, not relaxed
fusion validation. Other categories still require production repairs.

The public ordered route (`90f13070`) now selects the existing owned executor only
for reconstructed late host actions. Prefix-only programs remain concrete; source
owners, executor coverage, role/layout/alias gates and unobserved checked-stage
refusals remain mandatory. Deferred artifacts do not expose a single static
LinkPlan/certificate or become generally composable after execution. Local receipt:
24 tests / 211 assertions, including actual OpenCL and Level Zero replay, changed /
omitted inputs, state/constant capture, output lifetimes and late checked overflow.
An initial test incorrectly expected ordered behavior from a prefix-only fixture;
the corrected fixture has a genuinely late allocation. This is not a performance
or multi-device receipt.

Do not extract that route alone onto main: the attempted cherry-pick exposed
missing original-frontend/source-owner and checked-stage prerequisites. It was
aborted without dropping guards or opening a PR. A safe independent source-ownership
slice is medium-sized and needs its own clean-main workload qualification. Until
then, keep activation pending on the draft; first repair broad qualification,
then checked-result realization, archival reconstruction and helper/AD import
closure, without replacing checked casts by IEEE conversions.

The next reproduced lifecycle repair distinguishes nominal interface spelling
from physical receiver evidence. Frozen source identity canonicalizes a Symbol /
String name, while emitted interface representation remains unchanged for AD
registry consumers. Direct successful bytecode publication also clears the
pending JIT spec with its old stub owner in the same metadata update. Rejection
retains that exact pending spec; neither operation refreshes old observations or
claims transactional concurrent dispatch publication.
The interface-registry review caught and rejected a broader emission-field
normalization: existing AD consumers intentionally retain that representation.
The final fix changes only nominal lineage/comparison and successful pending-spec
retirement. Fresh lifecycle plus paired/owned helper tests pass 51 tests / 503
assertions; the final pure metadata-update form passes the focused lifecycle gate
again (17 / 175). This repairs reproduced failures, not the remaining AD import,
checked-conversion or full-draft qualification categories.

The next reproduced ordered-realization failure was preflight ordering, not a
missing observation capability: `run-ordered-stage!` dereferenced a nil executor
before delegating the checked entry's owner guard. Checked entries now validate
record/state shape and literal-true ownership/observation flags first; actual
historical source ownership still precedes storage acquisition. Ordinary entries
retain their source guard before acquisition. Negative tests forbid acquisition
for malformed executors and for a structurally plausible record missing its source
owner. The affected realization/checked-execution gate passes 12 tests / 271
assertions, including real OpenCL and Level Zero success, failure-order and private
generation checks. This is guard repair, not a new checked-conversion family.

Two raw-result-cast tests still expected the retired raw cast spelling despite
the canonical scalar-conversion term. They now assert its exact source/target
width, source operation, rounding and overflow policy while retaining inner and
nested fold precision checks. Checked Double-to-Float still traps; Float-to-Double
is exact; unchecked-IEEE and lexical-shadowing controls remain unchanged. Review
confirmed no removed precision coverage; the focused gate passes 4 tests / 19
assertions. This test-only correction is not checked-cast execution support.
Two normalized-AD negative tests also expected later refusal sites. Tampered
operation metadata now fails operation validation before certificate comparison;
a wrong-width use-site declaration fails declaration validation before sum-step
validation. Their exact reasons were updated without accepting alternatives or
removing malformed cases. Review confirmed the earlier refusals; both full test
namespaces pass 23 tests / 182 assertions. Production validation is unchanged.

Fresh qualification at `fe439ca4` still fails unit, OpenCL and fixture emission;
setup/build success does not qualify the draft. The public clone fixture exposes
an independent generated-storage mismatch: clone normalization invented a generic
`alloc-like` call with no selected implementation. Generated copy scaffolding now
uses the existing dtype authority's concrete primitive constructor, retaining the
source length evaluation, fresh allocation and identity map. The allocation
validator is unchanged. The affected frontend, public compile/link and actual
OpenCL/Level Zero clone gates pass 140 tests / 876 assertions, including primitive
widths, empty/changed inputs and null failure ordering. Packed-half scalar copying
remains refused. This closes that allocation mismatch, not the checked-exponential
overflow-boundary proof or the remaining broad qualification failures.
The clone repair was independently extracted and fresh-main qualified (133 tests /
819 assertions), then squash-merged as #1546 at `b0cef3df` after all seven exact-head
CI checks passed.

The repeated-dense-map device failure was a contract mismatch in its checkpoint
fixtures. IEEE checkpoint cases now explicitly use `unchecked-float`, with finite
rounding and overflow-to-infinity controls. Separate checked `float` epilogues,
including Double-to-Float-to-Double, refuse before invoking the legacy SegMap
emitter and retain the original portable decline. The existing final epilogue
guard already refused these cases; no returned-artifact miscompile was established.
Independent review found no blocker, and the affected contraction/body plus actual
OpenCL/Level Zero gate passes 29 tests / 451 assertions. Other legacy fallback
decline classes remain outside this bounded guarantee.

Shared positive map/map and horizontal fusion fixtures now use the existing inner
`unchecked-float` pattern, retaining the outer Float storage cast. Executed raw-bit
source/fused comparisons include overflow; checked-cast no-fusion and failure-before-
write controls remain unchanged. The focused gate passes five tests / 27 assertions.
The full fusion namespace remains unqualified: its preceding run had 24 failures
and nine errors, including separate terminal-frontend refusals. Do not replace a
nil frontend program with an invented SOAC value or waive source completion.
The internal `pmap` macro's bare-cast allocator and materializer cast tables also
need later reconciliation with the shared descriptor/dtype authorities: putting
`unchecked-float` directly in that cast slot currently disagrees with runtime
storage. The existing inner-expression pattern avoids extending that inconsistent
surface while qualification is red.

Fresh terminal-roundtrip qualification exposed a missing execution root, not a
reason to weaken owner coverage. A removable scalar reduction extent was retained
as a host-only graph-shape proof equation, but no buffer shape used it and prefix
pruning omitted its evaluator. Prefix construction now roots only results whose
equations already have literal-true `:host-only` and `:graph-shape-definition`.
Canonical source correspondence, dependency pruning and executor checks remain
unchanged. The fresh-process gate passes eight tests / 135 assertions, including
actual OpenCL/Level Zero direct and scaled Double-to-Float-to-Double checkpoints,
zero rows, changed replay, overflow/publication and existing checked-prefix controls.
The complete affected invocation/source-order/terminal-owner namespaces also pass
46 tests / 310 assertions; neither receipt qualifies the whole draft.
The source oracle is JIT-settled before its owner is captured; an earlier warm REPL
also retained an obsolete fixture signature, so only the fresh receipt qualifies.

The next bounded terminal extension now distinguishes numerical-result exports
from public scalar argument ports. Existing `:parameters` remain exactly the
numerical export/output vector; optional `:public-scalar-parameters` carry the
original argument ordinal and declared Double/Long dtype. Semantic revalidation
derives those ports from the actual InvocationPlan parameters, never the contract's
own assertions. At least one numerical dependency remains required; arrays,
unsupported carriers, arbitrary captures and shadowed public captures refuse.

Owner compilation authenticates the retained frontend's exact implementation,
full parameter order and actual primitive body-entry carriers. Independent review
caught an initial reliance on editable frontend `:definition`; the shared
`frontend-body-entry-carriers` helper now binds physical facts back to the opaque
owned implementation and independently selected entry. Same-source/order foreign
entries and a real Object-ABI input with a claimed logical Double type refuse.
No new type registry or runtime carrier inference was introduced.

Projection and invocation share numerical-then-public argument order. Runtime
checks all public carriers before requesting numerical exports, then downloads
only numerical results. Ordered host/numerical calls now share the static path's
declared-program scalar projection: public source-only arguments remain in the
invocation environment, not the emitted ABI. Undeclared-port checks remain intact.

The fresh-process focused gate passes five tests / 186 assertions, including
OpenCL and Level Zero mixed Long/Double arguments across an array slot, zero rows,
changed-input replay, checked overflow and exactly one numerical download per run.
Four complete affected ownership/lineage/invocation namespaces pass 58 tests /
451 assertions, including malformed ordinal/type/symbol and pre-export rejection.
The Object-ABI negative was added in that same fresh process without changing any
existing fixture signature. These receipts do not qualify the full draft or
establish resident training composition; the explicit host-result API and all
ownership/publication guards remain required. Draft CI is still red in broader
unit/emission qualification; do not merge it based on these local receipts.

The existing ordered-prefix scalar preparation now uses that same authenticated
entry projection instead of independently consulting editable frontend definition
fields. Missing physical evidence still abstains after ownership/history checks;
a nonnil foreign entry still throws. This preserves optional capability behavior
without treating stale or contradictory evidence as absence. Required executor
coverage remains the runtime gate; an owner with no prepared scalar executor is
not permission to execute an unsupported source action. Independent review
confirmed the shared boundary and caught the needed absence/contradiction split.
The bounded REPL's four affected lineage/checked-owner/terminal-owner/source-order
namespaces pass 57 tests / 440 assertions, including both required and optional
foreign-selection refusal and missing optional physical evidence.

Fusion qualification now separates nine checked terminal source shapes from the
pure scalar algebra. The checked shapes retain exact ordered source and explicit
declines; Double-only controls retain nonidentity transforms, host boundaries,
aliasing and shared-use obligations. IEEE map fixtures declare that intent in
their body, preserving the outer storage cast. Alias and intervening-write
negatives have matching fusible counterfactuals rather than passing vacuously
because a conversion independently blocks motion. No source cast was globally
reclassified as pure.

This exposed an actual optional-optimization failure: a legal typed IEEE map
feeding a reduction/scan projected its canonical conversion back to source during
algebra recertification, where generic effect analysis could not prove the element.
Fusion now abstains only for that exact local missing-element-effect request and
retains the complete materialized boundary; invalid algebra, malformed conversion,
source/history failures and unrelated exceptions still propagate. Later legal
candidates are attempted and only successful rewrites affect the counters.
Independent review found no blocker. Primitive operand correspondence through
canonical conversion algebra remains a performance capability gap: valid policy
attributes alone are not proof of the operand's physical carrier. Do not solve
that gap by erasing Object/reference completion or relaxing source purity.

A fresh single 900 MiB REPL passes six complete affected namespaces (125 tests /
860 assertions) and four actual OpenCL/Level Zero scalar-observation regression
cases (4 tests / 206 assertions). The complete fusion namespace's preceding
focused run passes 58 tests / 366 assertions. These receipts supersede its earlier
local failures, not the broader draft's red CI. The independent fusion-abstention
repair is being qualified separately on clean main so it need not wait for the
source-ownership draft. Remaining production failures include training scalar
ownership, checked conversion maps and normalized AD/source lifecycle; none is
declared complete by fixture consolidation.

The independent optional-fusion repair landed as #1547 (`97ea226b`) after all
seven exact-head CircleCI contexts passed. Its clean-main tests inject missing
element evidence into otherwise supported Double programs; they do not establish
clean-main IEEE conversion admission. The draft retains its separate concrete
IEEE map/reduction/scan regressions.

The next checked-map repair explicitly marks eight audited primitive FP math
operations (sqrt, floor, sin, cos, tan, exp, log and pow) as nonthrowing only after
operand completion. It does not mark a descriptor factory, infer source carriers,
permit Object unboxing, alter target-library accuracy or discharge final checked
Float storage. KernelBody verifies operand types, exact arity and math realization
independently; the arity check now precedes analyses requiring nonempty operands,
so malformed zero-argument intrinsics produce a structured refusal. Independent
review found no blocker in either boundary.

A fresh 900 MiB REPL passes three complete affected namespaces (28 tests / 459
assertions), including existing physical-return-payload negatives, plus two actual
OpenCL/Level Zero checked-equation cases (2 tests / 102 assertions). The latter
exercise exp with finite values, NaN and overflow, preserve observation before
publication, and check graph/event retirement. Overflow is checked against the
actual target result; implementation-defined library accuracy is not a promise
of identical JVM/device decisions arbitrarily near a conversion threshold. These
bounded receipts do not close the draft's other conversion, training, persistence
or AD failures.

#### Reviewed next vertical: source correspondence through resident training

This is the October 20 planning checkpoint, not a new implementation or qualification
receipt. A read-only independent review studied the actual scalar, AD import, resident
scheduling and training-fixture machinery. It corrected the earlier sequence: closing
the four scalar diagnostics is necessary but insufficient. Prepared reverse AD still
records an unclosed semantic import, which ordered source compilation must reject.

The next vertical must keep these obligations separate:

| Boundary | Required evidence | Evidence that is insufficient |
|---|---|---|
| Source ownership | Original lookup, definition/rule history, parameter plan and source policy | Equal syntax, current tags or a refreshed snapshot |
| Source to typed computation | Corresponding operations, actual input/storage facts and all conversion/checkpoint positions | A successful typed lowerer alone |
| Conditional result | Value domain after original evaluation completes | Proof that reads, unboxing or conversions cannot fail |
| Scheduling | Required-evaluation occurrence/order, legal numerical association and dependency graph | Unused result, read permission or logical dtype |
| Physical realization | Actual JVM carrier or explicit resident payload representation | A logical Float/Double result annotation |
| Lifetime | Exact views, initialized storage, ownership and completion/lease evidence | Allocation existence or nonblocking completion observation |

Implementation/landing sequence:

0. **Establish the gate and baseline.** At the planning checkpoint, exact PR #1539
   head `11852323` had setup/build success and failing unit, emission and OpenCL checks.
   The evolving branch remains draft and unqualified; each later head needs its own
   qualification, and absent vendor successes cannot be assumed. Classify those
   failures and repair affected gates before enabling new admission. Strengthen the
   existing training oracle's paired `map` comparison: unequal lengths must fail,
   and this finite fixture must explicitly reject nonfinite expected/actual values.
   Preserve independent analytic and rounded FP32 oracles and existing tolerances.

1. **Bounded reduction correspondence, without new admission.** Use original owned
   expressions and authoritative primitive array/parameter facts to validate the
   element, sequential lexical locals, initializer, recurrence and exit. Reuse
   `core/source-arithmetic`'s binary/loop plans and the ordinary scalar lowerer;
   do not add a parallel inference engine or carry original bytecode occurrences
   through every AD alpha conversion. Keep `completed-fp-body-carrier`'s existing
   restricted completion contract: array loads must not acquire a nonthrowing proof.
   Retained-type validation checks consistency, not original-source authentication.
   Preserve bound evaluation/conversion, including the surface reduction's Int count,
   and zero-trip initialization. Only already-certified parallel reductions can
   regroup/interleave; per-element arithmetic/checkpoints and ordered folds retain
   their separate contracts. A permuting tree requires commutativity as well as
   associativity. Checkpoint tests are not bitwise equivalence claims for FP trees.

   The follow-up implementation review fixes the query boundary as
   `(owned original reduction, actual operand/storage facts, constructed reduction
   region) -> conditional value/checkpoint correspondence`. It must not invent a
   primitive JVM exit carrier: a Float value can still leave the JVM reduction
   boxed as Object. Physical exit facts remain with the separate actual-emission
   observer. Start with one FP carry, sequential lexical locals, declared primitive
   array reads and authenticated core FP arithmetic/conversions. Compare against
   the actual `ProductReduction` component and lower its region through the existing
   `make-lowerer`; retain source Int-count conversion and zero-trip initialization.
   Unknown storage, nullable scalar inputs, user calls, forged retained widths,
   mismatched narrowing checkpoints, integral/multi-carry/data-dependent loops and
   unsupported count domains abstain initially. Expose successful correspondence
   separately in diagnostics, not as a `:normal-completion :carrier`, nonthrowing
   fact or new admission permission. Reuse maps and the existing dependency owner;
   no new rule/type registry, bytecode materialization or origin propagation is needed.

2. **Scalar consumers through existing representations.** Realize the Float MSE
   result followed by its Double quotient without narrowing Double intermediates.
   Use existing required-evaluation catalogs and singleton resident scalar scheduling;
   do not put device reduction results into host-only scalar preparation. Selected
   Object/Double loss-scale results need the existing owned computation recipe plus
   use-closed payload projection; their physical source carrier stays Object.
   Checked Long dimension products must execute at their original ordered host sites
   when their operands originate there. Late device-dependent exceptional work stays
   unsupported without an explicit completion/transport contract. These may be
   separate focused PRs, but must share the same evidence boundaries.

3. **Close the bounded prepared-AD semantic import.** Reuse its existing generated
   owner and definition collector. Freshness and hygiene alone cannot clear
   `:prepared-reverse-ad`. Validate argument evaluation and entry conversions in their
   original order; the sealed active/inactive parameter plan and padded result slots;
   all input/result conversions, narrowing checkpoints and observable Object uses;
   required evaluations; supported operation correspondence; and retained definition,
   source-policy and AD-rule dependencies. Clear only the exact import whose complete
   bounded projection is checked. Unknown operations and transitive unclosed imports
   remain refusals. The opt-in normalized device subset is not an already-complete
   replacement for this generic prepared path. Do not add another AD rule registry.

4. **Qualify the unchanged public training fixture.** Run source/runtime AD controls,
   hardware-free assembly, then OpenCL and Level Zero forward/gradient/collective/SGD
   execution for unequal batches, changed arrays/parameters and multiple updates.
   Check initialized views, allocation budgets, source fences, lease/cleanup behavior
   and stale preparation rejection. Respect the existing finite unrolled plan's
   single-run boundary; changed-input tests must not assume unsupported replay.
   Co-located execution is not multi-device fabric or overlap evidence.

5. **Consolidate before expanding.** Review the final authority/consumer graph,
   remove only actually superseded implementations/tests, and publish a compact
   supported/refused boundary matrix. Do not weaken numerical tolerances, restore
   source-shaped fallbacks, or count more helper tests as workload closure. Land only
   exact-head green changes; compile-only vendor checks and runtime/performance
   receipts remain distinct. A read-only planning review does not qualify code.

Every implementation boundary needs positive and refusal tests. The cross-boundary
negative set includes forged dtype tags, lexical/namespace shadowing and remapping,
stale AD rules/definitions, nullable or escaping Object results, explicit narrowing,
wider recurrences, checked overflow, empty/negative/count-overflow domains, Float
cancellation checkpoints, and missing/duplicated/reordered required evaluations.
Storage/range safety and operand evaluation remain separately checked even when a
conditional value correspondence succeeds.

Compile latency is an explicit acceptance axis of this vertical, not deferred until
more backends land. Retain one bounded marketing density and the unchanged training
fixture. Separate fresh-process startup/loading, frontend/dispatch, AD preparation,
SOAC/fusion/effect analysis, emission/materialization, executable preparation and JVM
warm-up; record phase counts, allocations and cache hits/misses as well as elapsed
time. Compare fresh, loaded-first, same-signature repeated preparation, changed data
with fixed shape, changed shape and changed-definition/rule cases. These distinguish
unnecessary recompilation from intrinsically excessive analysis and startup cost.
Optimize repeated work before adding cache layers. Reuse immutable owned analysis
results only under their actual source/rule/policy/target dependencies; do not suppress
freshness guards, persist process-local owners or assume a bounded cache bounds
generated-class/native-memory lifetime. Cross-process artifact reuse needs its own
portable identity/validation boundary. Set numerical budgets from measured baselines;
do not report whole-worker or sampling time as pure compiler time. Requalify cold/warm
costs after the vertical closes, independently of execution throughput.

The first execution increment hardens the finite training oracle (3 focused tests /
38 assertions) and diagnoses an actual artifact boundary regression. A real ordinary
map compilation carries an `OrderedSourceOwner`; archival fingerprinting refuses that
process-local object. Sealing now explicitly refuses attached or independently required
ordered owners before serialization, and bounded store diagnostics retain the keyword
reason through the outer exact-evidence error. Dropping an owner cannot manufacture
portability. Focused refusal/diagnostic checks pass 2 tests / 10 assertions; adjacent
source-owner/fingerprint checks pass 13 / 45. This does NOT restore portable evidence
or PDE capture: a producer-owned portable payload plus authenticated source-executor
reconstruction remains required, including exact decoded semantic/emitted identities,
historical dependencies and executor coverage. No opaque-owner fingerprint exemption
or process-local identity fallback is allowed for durable restore.

A warm REPL diagnostic on a tiny synthetic-target map compiled once and recorded
three process-cache hits: loaded-first preparation 348.6 ms; identical, changed-data
and changed-dynamic-size preparations 19.8, 18.1 and 17.0 ms. Hit-path equation lowering
accounted for 18.2, 16.7 and 15.5 ms. This is one loaded-machine probe, not a marketing
profile, cold-start benchmark, stable speedup or device execution receipt. Artifact
publication remained explicitly refused. Keep cache misses and work repeated on hits
as separate profiling targets.

After the source-discovery lifecycle correction, the unchanged local training
gradient diagnostic still refuses at the same four scalar boundaries (Double
loss quotient, Object/Double scale and two checked dimension products). One loaded
REPL observation took 1.019 seconds, with inclusive transform times around 502 ms
in fixpoint, 209 ms in late cleanup, 180 ms in SOAC fusion and 117 ms in DCE. This is
a single coverage-refused analysis under background load, not a successful training
compile, isolated CPU profile or marketing performance measurement. It confirms
that source discovery no longer masks those refusals with an unrelated stale-AD
module error; it does not demonstrate cache sufficiency or remove those refusals.

Use one focused independent reviewer for the authority/admission changes and another
review pass over the integrated workload/deletion diff; reuse reviewers rather than
launching recurring broad surveys. Ask each review to identify duplicated authorities,
unvalidated consumers and counterexamples, not merely approve the written design.
Keep focused REPL feedback local, full suites in CI, and cold-load/lifecycle tests
where they specifically matter. Each PR records its invariant, affected consumers,
negative gate, workload evidence and explicit nonclaims.

The shared pass executor and opt-in phase observer landed as PR #1543 after all
seven exact-head CI contexts succeeded. Production and diagnostic pipelines now
share transform/result extraction/validation; timing separates transform and
validation without clock reads on the unobserved path. This supplies measurement
infrastructure, not evidence that marketing preparation is fast enough.

The independent finite-oracle hardening landed as PR #1544 after all seven
corrected-head contexts succeeded. Its first unit run correctly caught the
exact-membership shard guard still expecting eight training cases; that guard
now explicitly includes the ninth oracle regression and still checks uniqueness.
No tolerance, workload case or required CI context was removed.

The next independently reviewed lifecycle correction separates authenticated
original-frontend discovery from freshness of a previously generated module.
Original source can seed a new preparation after an AD-rule revision; the old
preparation and actual emitted-code consumers still reject stale rules. The
opaque source owner is checked against the actual selected implementation,
including rejection of a same-signature foreign owner. Cold lookup neither runs
the source nor upgrades its bytecode. Focused source/lifecycle tests pass 25 tests /
123 assertions in the existing REPL; the combined affected gate previously passed
36 / 201. This does not establish stale-module renewal, portable owner persistence,
new effect admission, or the complete marketing fresh-worker lifecycle.

Step 1 now has a bounded diagnostic-only first subset in shared scalar machinery:
`check-reduction-fp-correspondence` checks the actual constructor source/product
pair, owned primitive storage facts, one FP carry, sequential locals and qualified
bootstrap-authenticated core FP arithmetic. It reuses the source arithmetic and
initializer plans, then compares two deterministic invocations of the existing
scalar lowerer. Original initializer identity and conversion/checkpoint agreement
are separate checks. Node/depth and child-enumeration limits bound analysis, even
for a lazy unbounded input. Required count conversion remains explicitly named
`:source-count-obligation`, not a proof of successful conversion or bounds.

Independent review caught and removed an attempted boxed-payload/primitive-operand
conflation. The first subset now abstains on all selected calls; supporting a known
Float payload behind Object still needs the planned physical-context projection.
The real selected Object/Float and Object/Double cases have refusal regressions.
Coverage reporting exposes successful results under
`:reduction-value-correspondences`, never as `:normal-completion` or admission
evidence. A synthetic preserved post-reduction quotient remains refused with its
evaluation obligation. This is not closure of the AD-generated training reduction:
its selected calls and other non-core shapes may still abstain. The fresh bounded
REPL's affected scalar-region, physical-return and equation-first diagnostic suites
pass 79 tests / 851 assertions. Full draft qualification remains open.

The next physical-context review confirms that the first real training quotient
still lacks an authenticated result-context relation for its imported/generated
reduction. Audit whether that source occurrence is still directly owned and
observable through `frontend-reduction-emission`; use its actual result fact only
when the exact occurrence relation holds. An AD-generated or alpha-renamed source
needs a relation established by its prepared producer, not syntax equality or a
global guessed reduction-return rule. Preserve Object/Float at the quotient's
source operand and the denominator's actual Double conversion. Only then may
`checked-core-fp-evaluation` justify the enclosing Double arithmetic and existing
singleton resident scheduling realize it. Object payload elimination additionally
needs all uses/aliases closed. Original count/init/read/evaluation obligations and
the larger prepared-AD semantic-import gate remain independent.

The root-context follow-up now records the actual generated reduction root's
result flags and emitted carrier after its body is emitted. It does not substitute
the nested exit's context: a real Float-reduction/core-quotient regression has
`preserve-source-call-results?` false at the root and true at the exit. The query
exposes this observation only alongside the owned expansion and exact single loop
child, matching physical carrier and complete boolean context. Missing root data,
copied regions and malformed contexts abstain; no payload, normal-completion,
source/native projection or new admission follows from this observation alone.
The independently reviewed result-context completion now represents exactly the
saved four flags plus a reference result carrier. Its JVM consumer evaluates once
under that context, requires the actual reference carrier and returns that same
reference without adding conversion or boxing. Primitive metadata cannot change
its physical carrier. Source/native projection and structural AD explicitly
refuse it; scalar analysis gains no presence/payload fact. A joined producer /
consumer regression compares actual observed reduction results, count failures,
bad reads and initializer effect order with the original. This is the mechanism,
not yet the authority bridge: helper projection must still establish exact
pre-hygiene occurrence and historical source ownership before using the contract.
The combined affected gate passes 86 tests / 725 assertions in the bounded REPL;
the independent review found no blocking issue. Draft #1539 at `5b5ece57` still
fails CI: the inspected distributed-training shard reports the same four
preserve-evaluation scalar refusals plus the new conditional Float correspondence.
This checkpoint neither fixes those refusals nor qualifies resident AD training.

The next reviewed implementation boundary is one local helper projection
transaction, not unconditional insertion in `resolve-walked-body-input`. That
resolver also serves pre-AD, source-JVM and native C; the JVM-only completion would
break those consumers and could make an ordinary import fail only after warming.
Before hygiene, use the exact original frontend reduction/quotient and actual
emission history to jointly check root context, conditional FP recurrence,
Object/Float payload, Double denominator conversion, enclosing core arithmetic,
closed numerical uses and required evaluation. Reuse optional definition-input
transactions so abstention does not leak dependencies. Activate only with a
target-neutral recipe that accounts for the wrapper's physical behavior, followed
by existing whole-form hygiene and call-by-value substitution. Do not publish an
unused sidecar, silently force JIT from a query, or infer physical helper ABI from
its declared return tag. Missing evidence preserves ordinary expansion and the
strict coverage refusal. Target precision/capabilities remain checked: a Double
quotient cannot silently become Float arithmetic merely because its eventual
storage is Float. Count/read/initialization and prepared-AD import checks remain
separate from this transaction.

The count audit found that the original MSE bound is an array-length helper call,
whereas the normalized training form already has a bound symbol. ProductReduction
does not store either iteration extent. The diagnostic operator/checkpoint query
therefore now permits an opaque original bound within its existing traversal
budget, and explicitly reports `:domain-correspondence :unproved` alongside checked
Int conversion and original-position obligations. This does not execute or infer
the bound, prove a valid count, transport a hoist, or change admission. Even an
out-of-range literal can have matching operator arithmetic without a successful
source count conversion. The local transaction must separately retain original
bound evaluation exactly once before initialization and match the converted
target extent, including nonpositive domains. Reuse authenticated array-length /
shape projection rather than a new helper-name rule.
Independent review found no admission widening; its test-quality suggestion was
applied using a resolvable count probe with an observable counter. A fresh single
900 MiB REPL passes the nine affected namespaces: 159 tests / 1,510 assertions.
This also joins the real MSE helper's original root observation, Float operator,
Double denominator and Double quotient without asserting source-to-target closure.

Two warm, coverage-refused training observations after this slice took 1.361 and
1.214 seconds (11 observed phases, SOAC transforms 268 and 209 ms). Both retained
the same four scalar refusals and produced no conditional reduction summaries:
the actual selected-call shape is deliberately still outside the first subset.
These loaded-process samples establish neither speedup nor a successful training
compile, and are not marketing cold-worker measurements.

The shared loop-plan/JVM-consumer deduplication was also extracted onto clean main
as PR #1545 for independent qualification. Review found unchanged slot joins and
initializer coercions; arithmetic and unchanged loop-carrier suites pass 27 tests /
634 assertions in the campaign REPL. CI remains the clean-process authority.
PR #1545 subsequently squash-merged as `acfa349d` after all seven exact-head
CircleCI contexts passed.

The next concrete audit refines the diagnosis above. The owned `local-loss` AD
preparation contains a selected MSE helper call, not a generated reduction. That
helper's frozen original frontend still has an exactly observed reduction with
Object/present-Float result. Ordinary helper inlining subsequently renames and
substitutes it. This case therefore needs a local helper-boundary/context
projection, not global reduction-origin tracing through every AD transformation.
Do not attach the original observation to a copied expression by equality.

The actual inlined step was core FP arithmetic with bare `float` casts. Conditional
correspondence now authenticates those casts through their original namespace
lookups, retaining lexical shadowing, remapping and bootstrap inline guards. Its
constructor also moves a `let*` from around the recurrence into its element;
ordered deterministic SSA comparison accepts that administrative difference
without accepting reordered/duplicated/dropped reads. Exact semantic comparison
retains raw FP bits rather than Clojure numeric equality (including signed zero
and NaN payloads). Both retained dtype hints are independently checked before
canonicalization. The unchanged training diagnostic now has one conditional Float
correspondence and still exactly four required-evaluation refusals. The affected
gate passes 82 tests / 871 assertions; no execution admission changed.

One important context observation prevents a shortcut in the next implementation:
the MSE reduction's emitted exit returns a reference, but its original
`:object-result?` and `:object-call-result?` are false. Its reference result comes
from the actual branch/exit boxing path. An Object result fact alone therefore
does not justify relabeling the original evaluation with `object-input-contract`.
Preserve the emitter-owned evaluation/result context, any actual primitive return
conversion, and public helper boxing separately. If existing completion terms do
not express that exact result context, extend the shared completion algebra or
refuse; do not borrow a different input contract merely because both end in Object.

This vertical advances original items 3 and 5 and validates parts of 6 and 7. It does
not replace items 1, 2, 4 or 8, finish general AD migration/lifetimes, establish SOTA
performance, or qualify real multi-host execution. All eight obligations below stay
open until their own acceptance gates close.

CI timing artifacts now also distinguish a completed runner invocation (which may
have failing assertions) from an aborted load/report invocation. The existing
`:outcomes` assertion/skip counters remain complete; each test retains at most
eight small failure diagnostics with source location, exception class and
keyword-only compiler reason/rule fields. Runtime compiler objects, inputs and
formatted expected/actual values are not serialized. These additive fields keep
the timing schema and planner inputs unchanged. Reporter tests cover unchanged
events/counters, a real failing assertion, bounded samples, aborted loads and
output/reporter failures. The process exit/native-crash gate remains authoritative:
a completed diagnostic artifact alone is not a passing CI or coverage receipt.
Exact-head b1d94dc5 still has failing unit and emission jobs; their failures remain
open compiler/fixture obligations, not a new accepted baseline.

The loop-carrier join and contextual Float-literal initializer plan now live in
the shared source-arithmetic authority; JVM loop emission consumes that same plan.
This extracts the existing implementation without changing slot selection or
conversion placement. Unknown/reference initializers remain nonprimitive, explicit
Double expressions are not narrowed from tags, and wider recurrences still widen
the slot. The focused arithmetic/actual JVM loop gate passes 27 tests / 617
assertions after reconnecting to the bounded REPL. This is representation planning,
not a normal-return proof, reassociation permission or new SOAC admission. The next
training obligation is to validate structural reduction arithmetic and checkpoint
correspondence before deriving a conditional numeric value domain for its consumers;
physical carriers and required source evaluations must remain separate facts.

Required invocation-level scalar evaluations now have an explicit TypedSOAC contract:
a program catalog and an exact executor claim per source occurrence. Validation
rejects missing/duplicate claims, missing executors, hidden claims in fusion
constituents and placement across a required evaluation. Whole-program fusion
also compares the retained producer, so deleting both claim and catalog cannot
silently erase the requirement. Source projection/rebasing uses shared dialect
helpers, including fused constituent coordinates; sequential spine composition
requires distinct explicit source occurrences. These constraints do not turn
per-element checked conversions into global host effects.

This is conservation/ordering evidence, not original-source authentication,
totality, initialized storage, or observed runtime completion. Original source
ownership and scheduled-stage completion remain independently required. The explicit
TypedSOAC path now retains unused required scalar roots, projects audited singleton
payload reads, and revalidates the earlier writer and closed uses at scheduling.
Flattened graphs containing required evaluations have an explicit source-order
dependency chain; pure graphs retain their hazard-only DAG. The reviewed affected
gate passes 23 tests / 204 assertions, including OpenCL/Level Zero payload replay,
checked integer/narrowing refusals and real OpenCL invocation failure fencing.
Late *source* admission remains closed until source-to-ScalarIR correspondence is
proved; resident storage and logical dtype alone do not prove an original JVM carrier.

Completion analysis now requires the actual producer namespace for core operations,
including casts. Authenticating the canonical `clojure.core` Var in another namespace
was insufficient: a producer can mask that qualifier. Missing lookup context abstains;
changed definitions and masked qualifiers refuse. The shared Double-anchored arithmetic
summary reuses existing carrier/totality authorities and grants no removability,
argument completion or source motion. Its focused and adjacent gates pass 69 tests /
684 assertions, including independent Clojure IEEE edge cases and AD-generated scope
contexts. A primitive Double operand can also anchor the exact source overload when
the other operand is a proved present Float/Double reference payload; the source
planner still sees that reference, and the producer's physical facts stay unchanged.
Float-only and reference-only calls remain outside this proof. Independent Clojure
Object/Double overload oracles cover both operand orders and IEEE edge cases.
No device performance, complete training vertical or broad CI acceptance
follows from these checks.

The production route now forwards that same producer namespace beyond source
normalization into frontend analysis. It previously dropped this context from the
frontend option map, making otherwise available normal-completion diagnostics
abstain. A before/after regression checks missing versus actual namespace, rejects
a masked `clojure.core` qualifier, and keeps the late required scalar refused in
both ordinary cases. The affected source-completion, scalar-activation and scan
gate passes 28 tests / 839 assertions. This repairs evidence transport; it does not
close reduction checkpoint correspondence or grant operand completion/admission.

Bare core casts now use that same bootstrap authority while retaining their
actual namespace lookup spelling. Successful normal return can establish the
cast's primitive result carrier; missing context, lexical shadowing and replaced
namespace mappings do not borrow a canonical core definition. Historical guards
retain the bare lookup edge, so replacing it invalidates earlier evidence. The
affected completion/payload/activation gate passes 56 tests / 616 assertions.
The unchanged training probe still refuses four late scalar bindings, but now
correctly reports the selected MSE-scale call as an Object with a present Double
payload. This is not primitive storage, totality, box-erasure or admission evidence.

The actual specialized MSE source has also been checked separately from raw Clojure
macro fallback: precision-sensitive inputs retain a Float accumulation checkpoint
before the Double quotient; an empty domain reaches IEEE zero division. This is a
producer regression, not permission to infer that checkpoint from a result tag.
The unchanged hardware-free topology/cost/compute/storage/math-request gate freshly
passes 84 tests / 660 assertions. It does not establish device transport or measured
fabric performance, and items 6–8 retain their remaining integration obligations.

The original bytecode reduction observer now supplies a shared normal-return
`:result-fact` from its authenticated carry-load/exit conversion. Float/Double
carries remain an **Object** result with an exact present builtin payload at the
divergent-if boxing boundary; neither the carry tag nor its slot is a primitive
region-return proof. Owned AD uses this same fact instead of duplicating that
boxing-path test. Initializer/count/recurrence/read/replay obligations are unchanged.
The affected source-owner and runtime/AOT AD gates pass 43 tests / 559 assertions,
including actual Float MSE, empty domains, malformed/copied observations and stale
definitions. This is original-region completion evidence, not permission to copy
it onto alpha-renamed/generated training source or erase a box with arbitrary uses.

An owner-private producer relation now links exact unchanged emitter-input reduction
occurrences to their freshly owned emitted occurrences. The frozen frontend query
requires its existing opaque source owner AND exact emission-map identity in the
already-retained receiver definition observation. An edited emission map cannot
borrow genuine body/occurrence identities to manufacture different completion facts.
Equal-but-distinct reductions stay distinct; one input object reused at multiple
occurrences abstains. Copied bodies, missing legacy evidence and AD/SIMD-replaced
emitter inputs also abstain. Cold queries never force bytecode materialization,
and reduction-free bodies retain no extra input-body owner. The affected source,
completion and frontend/import gates pass 52 tests / 306 assertions, including a
concrete edited-observation attack. This closes the unchanged frontend→emission
edge only: hygienic inlining/substitution, AD forward preservation and strict loss
extraction still owe their own correspondence before training admission can widen.

Review refined the training route: do **not** make bytecode-occurrence tracing
through every alpha-renaming pass the primary AD→TypedSOAC architecture. The shared
AD producer still assembles transformed source forms, and the inliner explicitly
records its prepared reverse-AD import as unclosed. Close that producer once using
authenticated SourceFrontend inputs and its actually consumed historical definitions,
then validate its structural ParIR reductions. Original-bytecode occurrence links
remain useful for original source observations, not a substitute typed AD pipeline.

The first producer correction makes shared reverse preparation consume the frozen
frontend rather than replaceable walked-body caches, under the existing definition
collector transaction. Missing root lineage stays explicitly unclosed. This does
not yet seal generated-program fields, source policy or template revisions for later
reuse; the inliner's unclosed import marker remains. The next owner must bind the
exact generated body, params/tags, namespace, result/gradient slots and normalized
active-parameter selection, plus immutable historical observations and original
source policy. Historical adoption must not refresh snapshots. Retain any unclosed
transitive imports, and distinguish template invalidation from harmless later cache
materialization. Structural-domain and accumulation/checkpoint proofs follow; the
unchanged Float MSE training workload remains the end-to-end acceptance case.
The root-input change passes a fresh-REPL affected gate of 33 tests / 199 assertions
(preparation, source/loop projection, owned array AD, frontend imports and unchanged
training-refusal diagnostics). It rejects poisoned derived walk-cache input and
detects a source change during preparation without materializing a cold numeric
implementation. An earlier warm run hit anonymous-class collisions after backend
reload; restarting only the owned bounded REPL removed those errors. This is not
a whole-suite qualification or a fix for generated-program lifecycle ownership.

The next preparation-only owner now seals the exact generated fields, normalized
parameter plan, original frontend, immutable consumed-definition observations and
AD template epoch. Generation and legacy compilation use the definition's source
policy rather than the caller's policy. Altered fields (including symbol metadata),
stale rules and rules changed during generation refuse. The affected gate passes
39 tests / 220 assertions, including inactive checked conversions and selected
parameter gradient slots. This owner is checked at preparation/compilation/import,
not on every invocation. It does not close transitive imports or authenticate the
legacy `alength` source rewrite as an execution projection. Historical adoption,
benign JIT renewal semantics, structural domains and Float checkpoints remain
separate obligations; the inliner still records prepared reverse AD as unclosed.
At `903c6061`, setup/build passed but unit, emission and OpenCL gates failed.
The preparation gate does not supersede those failures.

Prepared reverse AD now adopts its authenticated indexed definition history into
an enclosing compilation, using the same first-observation merge as ordinary
definition discovery. Adoption never resnapshots, preserves unrelated serialized
inputs, unions lookup/source-import edges and commits once or rejects a competing
collector update. Legacy compilation and inlining both use this handoff. The
reviewed affected gate passes 70 tests / 354 assertions, including stale parent
and child histories, alias removal after handoff and unchanged training refusal.
This is not adoption of static-artifact history or template-epoch authority;
those producer guards and the actual semantic projection must still reach the
consumer before the prepared import can be closed. No fallback was restored.

The existing historical-input validator now also retains process-local producer
dependencies separately from Var snapshots and portable values. Prepared AD's
own frontend/template guard survives adoption, optional transactions and frozen
ordered-source observations. JVM root/static emission payloads and their prepared
reuse guards retain the same dependency. A review found that root AD expansion
previously happened before the emitter's fresh collector; lazy-JIT transformation
and emission now share one historical transaction before publication. Emission
inherits those original observations rather than resnapshotting around transformed
code. An actual first-JIT gradient root, not only injected emitter fixtures, proves
the retained epoch guard. The reviewed affected gate passes 87 tests / 484 assertions.
This connects freshness checks, not general per-call dispatch instrumentation or
the inliner's semantic projection. The prepared import remains unclosed; actual
Float reduction realization, selected loss scaling and checked dimension products
still block unchanged resident training. CI at `39dd672e` has setup/build success
and fixture-emission failure; the other gates were still pending when checked.

CI then exposed an ownership-scope regression: independent root materialization
inherited a caller's unrelated, already-stale analysis collector. Root AD/SIMD
transformation and emission now run in their own historical transaction; outer
observations are neither overwritten nor refreshed and still reject promotion.
The focused real-emission/normal-return gate passes 36 tests / 237 assertions,
including the CI throwing-producer case and an explicitly stale unrelated owner.
This fixes isolation, not a waiver for stale executable reuse.

Whole-call AD projection now lives beside its preparation producer. It checks
the flat program/padded slots against the sealed parameter plan, uses the existing
call-by-value argument planner, then applies shared whole-form alpha conversion
and capture-avoiding substitution. The inliner's independent top-level renaming
map is removed. Rejected shapes/arities do not adopt dependencies; the projection
operation itself records its still-unclosed import. Focused preparation/inliner
checks pass 40 tests / 203 assertions before the extra failure-atomicity controls;
the final preparation namespace passes 16 / 63. The broader 119-test / 771-assertion
gate has zero failures and one resident-reduction error. An A/B run with the
committed versus working inliner gives the same
`:scalar-reduction-requires-typed-graph` refusal; this is not a passing resident
training gate. No projection receipt or public map is granted ownership authority.

Shared scope consolidation follows two reproduced defects: qualification rewrote
quoted symbols and mistook `par/reduce` binders for same-named namespace Vars.
Qualification now uses `form/scope-info` for sequential/recursive/parallel scopes,
leaves quoted data opaque and preserves metadata. Named lambda self-binding is
one enclosing scope across anonymous arities, shared by free-variable analysis,
alpha conversion and substitution. Imported catch/reify/constructor classes use
a separate class-only structural projection; method names stay unchanged. The
reviewed scope/qualification/AD gate passes 164 tests / 973 assertions, including
cross-namespace class execution, recursive capture and existing AD scope controls.
These are closed-core fixes, not arbitrary unexpanded-macro coverage or new
authentication of type-hint metadata. They do not close numerical realization,
entry/return conversions or the prepared import's portable admission obligation.

The JVM scheduled effect-map boundary now uses the existing canonical-conversion
source projection before desugaring calls. Previously a retained conversion term
could execute as its IR constructor, returning a list where a number was needed.
All four reproduced continuation failures are fixed; the reviewed JVM carry/region/
branch gate passes 28 tests / 551 assertions. An overflow control confirms earlier
writes survive and the later store is absent when checked Float narrowing fails.
The wider six-namespace check remains red (65 tests / 886 assertions: one failure,
five errors) on GPU checked narrowing, carried integer-width admission, a bound
diagnostic expectation and source-ordered multilevel composition. Those are retained
debts, not waived by the passing JVM projection gate.

Current integration hold: PR #1539 is draft, not qualified for merge.
At `add22852`, setup/build passed; emission, unit and OpenCL gates failed.
Emission still refuses target-library `exp` inside a checked Float narrowing stage;
exception stability near the overflow boundary remains unproved. That refusal must
not be repaired with a totality flag or numerical tolerance waiver. At `577ca631`,
setup/build passed and emission, unit and OpenCL failed again. These
are not acceptance evidence for CUDA/HIP. This is not a green compiler baseline.
Dependent CUDA/HIP jobs are not acceptance evidence. The following table takes precedence over
historical receipts below; do not infer broad qualification from focused gates.

Public vendor-fixture emission now collects independent compiler refusals across
CUDA and HIP instead of stopping at the first public workload. Each diagnostic
identifies function, target, dtype and requested schedule; empty compiler artifact
lists are failures, not omissions. Any refusal still fails the whole gate. Progress
goes to stderr while successful stdout remains the emitted-file list. Cancellation
and VM failures, including wrapped causes, abort immediately. Five reporting controls
pass 30 assertions; a real six-compilation smoke emits four successful workloads,
reports the checked-exponential refusal on both targets and reaches the later
Long-state workload on each. This improves feedback, not vendor qualification;
private body-fixture generation and actual vendor compilation retain their own gates.
The first aggregate CI run (`e8ef4837`, emission 32706) exposed a phase-order gap:
a private segmented-reduction conversion refusal escaped after CUDA's public group,
hiding its accumulated failures and preventing HIP's public group from running.
Both public groups are now realized before private generation. If private generation
then fails, the public aggregate remains primary and retains the independent private
failure as suppressed evidence; cancellation/VM failures still abort immediately.
Seven reporting controls pass 41 assertions. Holding both artifact vectors is an
explicit CI-memory tradeoff, not permission to omit workloads. At `e8ef4837`, setup
and build passed; emission 32706, unit 32709 and OpenCL 32710 all failed. All eight
campaign obligations and the unchanged training/source-admission work remain open.
The `4203ffcc` emission run (32714) now confirms both public groups were attempted:
172 attempts, 140 passed, 32 refused, plus the separately preserved private
segmented-reduction conversion failure. Repeated target pairs expose missing
target graph lowering (norms, quantizers, RoPE and attention), checked map-stage
conversions, a dense-backward allocation proof and a contraction specialization
owner/dtype mismatch. These are an actionable debt inventory, not successful
vendor-toolchain execution. Unit 32713 also failed; OpenCL 32717 was still running
when this checkpoint was written.

The latest normalization correction replaces recursive reduction discovery and
equality substitution with an occurrence-specific strict evaluation context.
Extraction follows only the first operand of bootstrap-authenticated inline core
calls resolved in the actual producer namespace. It never crosses conditional,
binding, quoted, collection, selected-helper or later-operand contexts. Missing
producer context withholds extraction. This deliberately narrows speculative
helper extraction until its call-context proof exists; no legacy fallback returns.
The focused gate passes 134 tests / 744 assertions. The adjacent route namespace
remains red, with the same failure-event inventory under the committed extractor
and corrected extractor; that comparison is not a passing integration gate.

The follow-up separates normalization from removability. A strict first-operand
reduction can be exposed while its externally pure surrounding scalar retains
all source metadata, checked-completion obligations and preserve-evaluation flags.
The training-shaped unused forward-loss quotient remains an explicit binding;
coverage and the actual route still refuse it after parallel work. The focused
gate passes 135 tests / 754 assertions; the unchanged training diagnostic passes
1 test / 5 assertions. Review confirms fail-closed admission today, but metadata
alone is not a general fusion barrier. Before broadening admission, represent and
enforce the scalar's completion/ordering obligation through TypedSOAC fusion and
the ordered executable boundary. Do not turn preserved evaluations into dead
values or claim that normalization alone activates training.

The next reviewed slice admits checked Double-to-Float after a finite literal-bound
ordered fold, including an exact write-only destination with runtime-proven private
ownership. It preserves the Double accumulator and source failure selection. Helper
allocation correspondence uses the actual prepared original frontend, not an older
walked-body cache; rejected speculative helpers retain no dependency observations.
The affected gate passes 186 tests / 1,852 assertions, including local OpenCL/Level
Zero execution, changed inputs, precision-sensitive folds and failure cleanup.
This is not broad integration acceptance. Source-only public CUDA/HIP fixture probes
progress to `double-exponential-float-storage`, where `exp` lacks descriptor-owned
nonthrowing evidence. Preserve its declared target-library accuracy separately from
exception totality; do not erase checks or claim vendor execution/performance.

Review identified a second requirement before admitting that target-math operand:
nonthrowing `exp` does not prove the downstream checked cast has the same outcome.
An implementation-defined approximation can cross `Float/MAX_VALUE`. The warm
JVM probe around `log(Float/MAX_VALUE)` confirms adjacent Double inputs switch
from successful narrowing to `IllegalArgumentException`. Keep the case rejected
until a range-stability certificate or source-compatible observation policy
justifies exception behavior; simply adding a totality flag is insufficient.
Also separate primitive result-carrier proof from intrinsic nonthrowing evidence:
the current completed-FP helper summary assumes binary arithmetic after that
query and must not accidentally accept unary library operations.

The next compiler-quality slice separates that carrier recipe explicitly from
nonthrowing intrinsic evidence; library functions remain unadmitted. It also
retains the actual semantic-pass decline in equation-first diagnostics instead
of reanalyzing the original compatibility form. Manual normalization already
lifts the nested training loss reduction, but the complete route retains further
evaluation obligations; neither that observation nor better reporting proves
training admission. The unchanged workload still refuses. The focused combined
gate passes 44 tests / 636 assertions (carrier proofs, normalized diagnostics,
checked schedules, source order and actual local device execution). No source
evaluation flag, numerical check or compiler legality rule was relaxed.

Source completion now has an opt-in compositional query alongside the unchanged
presence/array `result-fact`. It threads actual normal-return producer carriers
through expanded sequential bindings and existing selected-method proofs. An
Object result containing Double remains Object; explicit casts retain bootstrap
callee ownership and may still throw while evaluating their operands. Unknown
producers shadow old facts. Bare casts, unexpanded lexical macros, primitive-input
contexts without result-carrier authority and opaque source remain unknown.
In particular, the source primitive-input projection calls `RT.box(T)`: the
conversion's input target does not establish a primitive result carrier.

These facts currently enrich refused-source diagnostics only (source position,
evaluation flags, normal-return carrier/payload); they do not admit kernels,
remove evaluations or supply payload recipes. Selected-call proofs and rejected
dependency transactions are reused rather than duplicating dispatch/type rules.
The six-namespace affected gate passes 175 tests / 988 assertions, plus the
changed nested-compatibility regression (1 / 2). Review found and corrected the
input-context/result-carrier confusion and unauthenticated core-cast assumptions.
Next: connect owned payload computations and closed numerical uses to ordered
scalar stages, with independent source-order coverage and the unchanged training
composition oracle. Cold recipe availability and bare source lookup remain
explicit obligations; a carrier fact alone is not sufficient.

The next tightening authenticates the selected arithmetic implementation's actual
original body, not just its nominal operation name. A canonical FP leaf must use
the same core operation over ordered formals; extra arithmetic, permuted operands
and intermediate rounding refuse. Lowering additionally requires matching actual
formal, operand and payload widths. A conflicting Double result stamp cannot erase
a Float helper's rounding checkpoint. Valid but unproved entry conversions still
refuse; matching lowering widths is not proof of original argument completion.
This does not activate training admission or remove preservation flags. Independent
review found no further concrete blocker in this restricted tightening. The affected
six-namespace gate passes 41 tests / 496 assertions, including checked device execution;
the final payload namespace adds positive Float lowering coverage (10 / 89).

The training investigation also found a producer hygiene defect: built-in AD
generators constructed bare `long`, `double` and `long-array` calls. The MSE,
min/max, convolution-dimension and pool-allocation generators now name their
intended Clojure builtins explicitly, leaving supplied argument forms unchanged.
Hostile-caller execution tests preserve value and exception oracles (including
null/out-of-range casts and negative allocation sizes). The affected three namespaces
pass 42 tests / 346 assertions; the actual JVM training gradient additionally matches
an independent exact-binary oracle on two changed target arrays. Review found no
further concrete issue. Actual training refusal now reports the MSE scale's normal
completion as Object containing Double. It still refuses retained evaluation:
neither qualification nor completion grants motion, erasure or GPU admission.

The next admission prerequisite closes an eager-execution ordering hole. Invocation
materialization used to evaluate all host steps before numerical execution, even
when their source positions follow an earlier kernel. The eager compiler entry now
rederives source order and refuses late host computations/allocations before any
materialization. This includes explicit storage and generated result allocations
whose host steps were pruned; exact aliases remain exempt. Validated source-free
programs retain their separate explicit prefix contract. The ordered executor is
the intended continuation, not permission to hoist or delete these operations.
Automatic Compiled activation and remaining generic entry-path audits stay open.

The affected seven-namespace gate passes 93 tests / 1,106 assertions, including
source-free compilation and actual local OpenCL/Level Zero ordered execution.
Independent JVM oracles write before late overflow/negative-size exceptions;
the eager refusal invokes no materialization callback and changes no output.
Review caught the missing allocation-action family before commit. One initial test
mistakenly assumed an explicit helper allocation had no host step; the actual
missing-step case is now a separate functional-map allocation test. A renamed old
test Var was explicitly removed from the warm REPL before the final passing gate.

The host scalar reference backend also had a selected-call correspondence defect:
it could discard an actual `.invk` receiver and invoke the nominal generic operator.
An explicit retained-receiver fixture independently returns division while the
nominal operator multiplies. This is an internal host-evaluation counterexample,
not evidence that ordinary public dispatch selected the wrong implementation.
Selected calls now refuse without their original source execution context. An
optional execution seam receives the whole expression and raw environment, rather
than already-evaluated arguments; its producer must prepare the executable under
the existing source owner and freeze historical guards. Existing source compilation
preserves the receiver, source policy and original argument completion order.

The focused gate passes 19 tests / 165 assertions: changed-input receiver oracles,
null completion before a later arithmetic failure, nested/lexical refusals, and
explicit region Float rounding/overflow checkpoints without suffix execution or
retry. Independent review found no further required correction. This is not
automatic routing: logical-dtype local/result coercions still require independent
source correspondence, and buffer-shape stand-ins cannot be passed to original
array-taking code without an audited shape projection. The callback itself grants
no ownership authority. Do not catch a partial interpreter refusal and retry it.

Historical source fencing now uses one `OrderedSourceOwner` for both ordinary
and checked ordered execution; the checked-only type/API/field is removed. One
validated occurrence classifier supplies all host actions for ownership and only
their late subset for routing. Prefix actions therefore cannot bypass ownership.
Missing/cross-program owners fail before preparation, device creation or captured
uploads, with the obligation independently rederived when the owner is absent.
Live/source/epoch fences cover each source action, completed stages before
initialization credit, allocation before acquisition, input handoff and publication.
Checked observation and private-generation retirement remain distinct obligations.

The existing strict source-policy/import restriction is not broadened into an
ordinary eager-compilation refusal. Unobserved compilation may explicitly abstain
from acquiring an unsupported ordered capability; frontend/bootstrap/history
failures still throw, and speculative observations roll back. Ordered entry then
independently refuses the absent owner. An unchecked prefix-allocation fixture
still compiles and lowers to a validated eager LinkPlan without touching a device.
No automatic routing or new unchecked source execution is enabled.

Cold frontend tests now create fresh definitions per test instead of assuming a
fresh JVM; two consecutive runs pass in the same REPL. A neighboring call-count
probe also exposed public multi-arity self-delegation being intercepted twice;
both invocation-step arities now share a private implementation. This changes
neither the scalar expression nor its evaluation count. Nested prepared-graph
mutation tests fence the completed outer source action, not each internal graph.
The final affected eleven-namespace gate passes 104 tests / 1,566 assertions,
including actual OpenCL and Level Zero execution and source-free compilation.
Review found no remaining concrete blocker in this bounded consolidation. Earlier
combined rerun failed only the eight cold-state assertions before their fixtures
were made fresh; the final gate retains those assertions unchanged. Independent
ordinary stage/host-action source-mutation probes poison the owner, grant no replay
credit and expose no resident result boundary. This is not broad CI acceptance or
training admission; all eight campaign obligations remain open.

The shared source projection now prepares optional owner-local ScalarCompute
executables using actual selected-entry carrier facts and invocation IDs. This
reuses the terminal projector's walker and existing source compiler, not another
interpreter, function registry or expression-string cache. Source policy, exact
Methods and dependency history are guarded; history freezes after compilation.
Execution requires the exact source step, exact boxed ports/results, and an
independent physical-result/ABI comparison. It performs no typed-region coercions
and no lookup, recompilation or retry. Unknown capabilities remain absent.

This first slice covers checked Long add/subtract/multiply, identity Long casts
and existing compatible Double/cast fragments, including explicit Float checkpoints.
It deliberately refuses selected calls, Long division, unproved casts/joins and
shape-derived carrier assumptions. Ordinary Clojure Long division returns Ratio
where an actual typed entry can return Double; a Method signature or logical dtype
cannot erase this execution-context distinction. Physical Boolean-to-Double also
needs its original context. Warm bytecode outlining already has owned input-region
APIs; cold source lowering must not manufacture those emitter contexts.

The affected fourteen-namespace gate passes 140 tests / 1,874 assertions, including
actual local device/source-free gates. Final focused controls additionally verify
the original typed entry versus ordinary Clojure separately, shadowed invocation
IDs, stale source, nullable Object conversion before later overflow, signed zero,
NaN and Float rounding/overflow (new namespace: 5 / 40). The terminal adapters remain
on the same projector. The final four-namespace gate, rerun after the interruption,
passes 45 tests / 351 assertions in the existing 900 MiB REPL.
Runtime activation is still gated: require complete executor
coverage before any earlier source/device mutation; do not discover missing host
capabilities halfway through an accepted route. No training or performance claim.

The following activation has local qualification, not broad acceptance: ordered
invocation now preflights every source occurrence before session creation/capture,
replay retirement and input handoff. ScalarCompute uses its prepared source
executable directly, including shared host-equation producers; no interpreter
fallback runs after earlier work. Standalone host equations and compound allocation
extents still refuse upfront. Allocations retain their original count conversion,
failure order and existing storage/initialization checks.

Shape carriers need independent source correspondence. A checked Long cast around
an array-length projection is admitted only for a bootstrap core leaf or an actual
selected helper whose authenticated original body computes that projection. A
nominal alength descriptor, interface result or logical shape dtype is insufficient.
Bytecode emission now records qualified core-intrinsic lookups as well as bare
ones; missing historical lookup edges in an old helper remain unsupported, not
recreated by consumers. Plain Clojure body-entry signatures use the owned generated
target class; Object-array hints remain conditional runtime class requirements.
Explicit source-free TypedSOAC projections retain their separately revalidated
closed contract, without inventing a Clojure source owner. Pure numerical routes
with no host source actions do not acquire irrelevant body-entry obligations.
The adjacent twelve-namespace gate passes 120 tests / 1,737 assertions; the final
preparation-fence gate adds 16 tests / 188 assertions after reloading Compiled.
Both local device probes report true. These receipts do not close the integration
hold, training admission, cross-vendor execution or any of the eight obligations.

Training admission will keep two execution modes under the same source-order
owner: host-only checked dimensions as original ordered host actions, and
device-dependent total FP values as resident evaluate-once scalar stages. Required
unused forward loss remains an execution root. Entry carrier evidence must come
from original retained source ownership and the actual selected interface, not
logical dtype options. Scalar use closure, publication and eager/generic binding
gates must be proved before broader frontend admission; none is waived here.

| Boundary | Current evidence | Next acceptance requirement |
|---|---|---|
| Terminal scalar host result | Integrated in `db3b4789`; ownership/invocation 58 tests / 406 assertions, full structured-control route 63 / 614, targeted local OpenCL/Level Zero scalar replay and failure controls pass | Broad integration green; retain explicit host-result API, source-history preflight and poison-before-publication. Array/composite terminal returns and portable persistence still refuse. |
| Checked numerical map result | Source-derived `mixed-scale` compiles across OpenCL/CUDA/HIP. The explicit internal ordered checked executor observes actual local OpenCL/Level Zero results, guards historical source ownership, tracks success-conditioned coverage, and rotates private output generations on replay. Finite literal-bound ordered folds retain Double accumulation and a runtime private-output obligation. Ordinary invocation/composition/persistence still refuse. | Qualify broader integration and automatic Compiled admission; establish descriptor-owned nonthrowing target-math evidence without changing accuracy policy. Preserve checked source diagnostics; never relabel checked casts IEEE to pass. |
| Training scalar source coverage | Actual normalized pass diagnostics now report `local-gradient-sum`'s retained scalar obligations. A second analysis of unnormalized source was misleading about the nested reduction; the complete route still refuses retained divide/multiply evaluations. | Close source-order/carrier evidence without dropping preservation flags or duplicating evaluation; retain the actual distributed-training composition oracle. |
| Existing AD/native/fusion integration | Exact-head unit logs still contain native refusals, closed-product AD gaps and fusion/terminal-conversion failures | Classify actual semantic gaps versus obsolete structural expectations; preserve numerical/failure oracles. Updating tests must not substitute for implementation. |
| Distributed and durable workload acceptance | Original obligations remain below; training composition currently blocked by the scalar row above | Complete their unchanged workload gates after shared compiler fixes. Local co-location/simulation is not multi-host performance evidence. |

The checked-map refinement is deliberately limited to a single-result one-dimensional
map, one outer checked Double-to-Float conversion, independently proved reads and an
otherwise-total primitive operand region. Exact write-only destinations require actual
private-generation ownership; finite literal-bound unit-step folds keep ordered carries.
Dynamic loops and arbitrary mutable maps remain unsupported. Status is overwritten on every active lane;
failed lanes retain their original Double diagnostic and do not store a public result.
Neither those stores nor schedule metadata grant unconditional initialization. The emitted
validator reconstructs from the semantic source even if a witness label is mutated.
Review caught and closed both a label-dispatch bypass and an overbroad conversion trigger.
Structural graph/ABI tests now use Double storage rather than claiming that an arbitrary
fixture-only schedule implements checked narrowing; the real checked-source tests remain.
Compiler reconstruction alone is not production checked-map execution or campaign completion.

The next observation component independently reconstructs the emitted obligation and
interprets completed status snapshots in bounded chunks. It selects the first failing
source lane and reads only that lane's initialized original-Double diagnostic; it reruns
the immutable checked conversion rather than the operand computation. Empty stages do
no reads; malformed protocols refuse. This component returns no initialization proof
and cannot independently enable execution. The confined binder reuses the existing graph
binder privately, retains the actual bound graph generation/temporary cleanup owner through
completion and observation, and releases everything on success and failure. Only exact
artifact KernelCall validation is admitted in an expiring owning-thread scope; ordinary
public graph binding, composition and persistence still refuse. Scope tests cover escaped
bound functions and future inheritance. Actual local device tests cover source result bits,
failure before a consumer, changed inputs, empty extents, checked exceptions and injected
observation-failure cleanup. The remaining integration is historical source ownership,
private output nonescape, ordered cursor advance and success-conditional write coverage.
Review explicitly does not treat the confined seam as ordinary invocation activation.
The five-namespace affected gate passes 76 tests / 563 assertions, including actual
OpenCL and Level Zero execution. The original hardware-free C-family compilation
receipts remain separate; no CUDA/HIP execution or performance claim is implied.

Ordered entry planning now preserves that observation condition explicitly. The existing
semantic write-coverage analysis partitions unconditional writes from checked writes;
ordinary whole-plan analysis refuses unresolved checked conditions. A separate
`PendingObservedEntry` retains a validated storage layout and a rederived entry contract,
not an executable LinkPlan or effect witness. Its `full-produces` excludes checked outputs;
`observations` and `success-produces` remain conditional and cannot satisfy a later read.
The source-ready path returns this pending variant rather than ordinary plan/evidence.
A preceding exact source result-allocation may identify private layout provenance, but
its acquisition or zeroing does not prove map completion or runtime nonescape. The runtime
still refuses this pending entry before acquisition until historical source ownership and
live allocation-ledger checks are integrated. The six affected namespaces pass 86 tests /
697 assertions, including adversarial bindings, condition mutations, ordinary-consumer
refusal, source-ready realization and the existing confined local-device qualification.

Historical source ownership now freezes the original producer's definition inputs,
exact semantic/emitted identities, and policies of actually imported helper bodies.
The inliner consumes snapshot metadata rather than pairing an old body with a later
definition observation. Refused optional imports leave no dependency. Unknown or
unchecked imported policies refuse this checked-source capability; prepared reverse-AD
imports without transitive lineage explicitly refuse it too. Ordinary AD is unchanged.
The eight-namespace focused gate passes 113 tests / 905 assertions, including local
OpenCL/Level Zero checks. This does not enable ordered checked-result execution or
qualify the red integration baseline.

Before activation, close an independently reproduced source-correspondence gap:
a materialized `deftm` keeps its original user-macro behavior, but fresh raw-source
walking can expand a redefined macro differently while the source Var guard still
passes. A REPL oracle returned `3.0` before and after macro replacement; compiler
walking changed from `x + 2.0` to `x + 7.0`. A fresh macro snapshot cannot authenticate
the original expansion. The smaller first route retains the actual typed frontend
body in an opaque producer owner, before backend AD/SIMD, bound to the selected
implementation and original source-entry lineage. Eager/plain producers retain
their actual Clojure frontend too; the context distinguishes parameter precision
from the original literal-walking mode. Consume that typed body without retyping
it, and normalize scopes with the same helper used by fresh source compilation.
Requested precision must correspond to the selected specialization, not silently
retype it. Root lookup uses existing flat-var and mangled-operation authority;
helper imports must use the same original frontend rather than stale walk caches.
Cold/legacy layouts without that evidence receive no checked-execution owner.
The pending JIT and GPU paths now share one frontend-materialization producer
under the existing implementation lock, without numerical invocation or JVM
emission. The immutable semantic frontend is separate from its pending/published
implementation correspondence. Promotion reuses that body but invalidates older
implementation observations; recompilation must obtain fresh correspondence,
not refresh an old execution owner's guards. Cold helper import prepares this
producer before considering legacy caches, avoiding a second frontend walk.
Focused tests cover concurrent requests, macro stability, and specification
changes during preparation. Ordered runtime activation remains separate work.
Acceptance includes changed-macro source oracles, distinct genuine Float/Double
specializations, backed/wrapped source handles, edited metadata refusals and a
prewalk-A/materialize-B helper import. Aliases and parametric cases remain required.
Live allocation-ledger checks, success-conditional initialization and a non-executable
final storage boundary remain required after this source-provenance work.

The non-executable storage projection is now represented separately as
`StorageBoundary`: it shares the existing physical/logical storage, alias and
whole-value output-order validators, but cannot carry instances, effect evidence
or initialization fields. `finish-ordered-storage-boundary` projects completed
source-state storage without rebuilding an executable whole-program call or
replaying initializers. Its construction proves layout only; even a fabricated
completed cursor grants no device completion. Ordinary LinkPlan validation and
instantiation reject its distinct type. Focused storage/LinkPlan/ordered-realization
gate passes 34 tests / 255 assertions. Runtime activation must independently establish exact
source-occurrence/replay-epoch observation accounting, live private-root ownership,
initialized-range checks and failure poisoning before this boundary can publish.

The explicit internal `instantiate-checked-invocation!` vertical now connects
these owners: exact historical source guards precede session creation, source
actions and publication; pending stages reconstruct their contracts and validate
actual private roots against the acquisition ledger and live cleanup owners.
Only successful device completion plus source observation commits initialized
ranges, the source cursor and that replay's occurrence entry together. Final
publication requires all emitted source observation occurrences, including unused
results, for the current epoch, plus live acquired roots and initialized outputs.
Checked replay retires private roots before any source zeroing and acquires them
under fresh physical keys; captured inputs retain their original acquisition
scope. Output leases prohibit that transition. Reallocating under an old key is
not sufficient because old raw resident views could otherwise resolve new bytes.
Focused checked/ordinary ordered execution passes 22 tests / 750 assertions with
both local OpenCL and Level Zero exercised. Review found no further concrete
blocker in this slice. This does not admit checked programs to ordinary binding,
composition or persistence, cover mutable checked folds, prove generated-AD
source lineage, qualify the red integration baseline, or close the eight-item
campaign. Automatic Compiled admission remains separate acceptance work.

The original-frontend slice passes the nine-namespace focused gate: 119 tests /
928 assertions, including actual confined local-device checks. The new source
oracles cover both original eager/plain frontend code and successful JIT frontend
code, original implementation/owner identity, backed lookup, precision mismatch,
and transitive helper-cache drift. Under explicit declared-storage preservation,
source typing remains fixed while the target carrier can differ; Float buffer /
Double scalar ABI checks pass. These receipts do not close cold frontend sharing,
generated-AD provenance, ordered checked execution or the red broad CI baseline.

### Historical integration receipts (not the current acceptance status)

Earlier integration hold at `e3046c82` identified these groups:

- Source memory evaluation: unused owned allocations lose negative-size failures
  in invoked JVM AOT; independent probes also lose unused read bounds failures
  and read-before-allocation exception order. Repair at attested original memory
  producers, not by disabling DCE or marking all generated AD reads effectful.
  The current repair marks both the owned-bytecode producer and the portable
  source-preparation frontier, including imported source helpers before generated
  tape/map storage. Its cold compatibility gate ran 34 tests: 455 assertions
  passed, with 2 failures and 10 errors in the already-identified product/native/
  contraction groups; none was in the allocation/read-order regression namespace.
  The shared evaluation/storage/native-alias gate passes 163 tests, 940 assertions.
  These are bounded repair receipts, not a green PR or campaign completion.
- AD products/completion: conditional product schemas, higher-order completion
  transposes, and native residual Object completions need closure; source guards
  and strict precision refusals must remain observable.
  An adversarial native probe rejected the proposed *general* literal-nil
  completion erasure: source numeric consumption throws NPE, whereas exposed
  native nil could become zero. Keep used/escaping nil behind its completion
  boundary; only a provably unused, flag-free literal-nil Object snapshot may
  be discarded by its lexical binding owner. Passing ordinary callers alone
  was insufficient safety evidence for that broader rewrite.
  The restricted dead-binding repair passes the cold shared gate (165 tests,
  969 assertions) and native AD caller gate (6 tests, 32 assertions), including
  the new source-NPE/native-refusal regression. Alias-chain cleanup remains
  conservative; these receipts do not close higher-order/compositional AD.
- TypedSOAC composition: contraction scalar-contract projection, resident
  training coverage, RMSNorm effect loops, and checked terminal scalar results
  remain integration failures, not permission to restore retired fallbacks.
  Contraction projection now reuses the existing recurrence certificate with
  accumulator-first orientation and lexical/evaluation checks. AD phase geometry
  is checked separately from body-replacing dense-leaf eligibility: full scalar
  elements and conversions remain; hardware replacement still requires the
  original bare-product guard. Cold affected gates pass 76 tests / 1,181
  assertions, including all normalized segmented AD checks. This includes host
  oracles and compile-only scheduled phases, not new device/performance evidence.
- Numerical/control evidence: Q4_K exact Float-bit comparisons differ by an ULP
  on multiple outputs; integer-bound/control regressions also remain. Do not
  relax exact contracts or update expectations without establishing semantics.
  The Q4_K mismatch has now been attributed independently: the compatibility
  consumer's boxed Float subblock result selects Double scale/add followed by
  a Float carry checkpoint; the product consumer accidentally selected Float
  scale/add. Explicit Double scale/add and `clojure.core/unchecked-float` at
  every superblock restore all 33 original outputs exactly in a REPL source
  intervention. Expanded host parity checks cover one, two and three
  superblocks, single/multiple rows and changed data (two selected tests,
  156 assertions pass). General unchecked-float support uses the shared cast
  descriptor, source constant policy and JVM RT overloads; it is not a
  quantization-specific emitter rule. Checked core float remains distinct.
  The cold five-namespace gate passes 103 tests / 1,351 assertions (numeric
  constants, descriptors, JVM input completion, scalar-region lowering and
  the full K-quant kernel namespace). Backend acceptance remains required. This is not
  a device-equivalence or performance receipt. Explicit source FP64 operations
  must not be hidden merely because public storage is Float; the existing
  widened-math capability gate is not a general source-dtype capability proof.

CI on `d3d2d824` also exposed three regressions absent from the prior job: fresh
map/gather AD rejects source reads carrying mandatory-evaluation flags. Two
derivation/canonicalization gates currently conflate external purity with
removability. Keep the flags and source evaluations; repair the gate distinction
under the existing private-output/replay contracts, with primal failure controls.
The reviewed gate correction passes 33 tests / 314 assertions across functional
fresh-write admission, scoped AD effects, owned allocations and the independent
tangent-storage law. Original bounds/negative-size failures remain, while aliases
and unsupported active effects still refuse. This repairs those three regressions;
it does not establish general replay safety or permit DCE to discard reads.
The subsequent actual `e3046c82` jobs (unit 32377, emission 32378, OpenCL CPU
32381) are terminal failures; the three fresh-map/gather errors are absent
from the inspected unit logs. The remaining groups above are still open.
The integer JVM reduction crash was independently isolated to SIMD eligibility:
`compile-segred` attempted Float/Double vector construction for a valid Long
reduction. Eligibility now consults the existing implemented-species table
before constructing vector forms; the caller keeps the original scalar
reduction when no species exists. Direct Int/Long admission controls and source
versus invoked AOT sums cover empty/small inputs and a result larger than Int.
The two affected JVM namespaces pass 63 tests / 180 assertions in the bounded
REPL. This is target eligibility, not a restored GPU source fallback or a claim
of integer vectorization; Object-parameter completeness warnings remain visible.
Product-boundary test consumers now use the existing JVM source projection
instead of evaluating canonical completion terms as runtime namespace calls.
The oracle also checks that a failing, unselected field is still evaluated in
source order. Both AD modes reach the common preparation-time scan precision
guard; checks retain its original inexact-storage witness and independently
contrast the source result (0.25) with a wrongly storage-rounded carry (0.0).
These three reviewed checks pass 13 assertions. The full namespace remains
red (8 tests / 27 assertions, one error): reentrant transformation of an
untyped compound completion still lacks a closed scalar transpose declaration.
That production admission issue is not waived by the test-consumer migration.
The HVP/Hessian failures had a separate concrete consumer defect: the HVP
constructor passed canonical completion IR directly to Clojure `eval`, causing
`ClassNotFoundException` for the reserved IR namespace. It now uses the same
JVM source projection as the other runtime AD consumers. No derivative rule,
compilation policy, reference-Class binding or ownership guard is invented.
Cold qualification passes the full reverse-AD namespace (46 tests / 121
assertions) and the O4 composition law (1 test / 12 assertions), including
analytic Hessians, directional finite differences and definition replacement
guards. Two private, unreferenced pullback prototypes were removed during the
consumer audit; active reification/HVP/Hessian paths remain. This repairs the
observed HVP consumer failure, not all higher-order AD or the marketing lifecycle.

Next normalized product-closure work must preserve **presence, tangent declaration
and physical completion as independent facts**. REPL attribution of the third
derivative shows an actual inactive residual `[nil [nil]]` joining a Double
residual, not permission to treat every untyped leaf as numeric. A prototype
presence-aware schema join progressed to a second refusal: generated `grad-acc`
results whose `present_gradient` operands already lack tangent declarations.
That prototype was reverted, not landed as a partial repair or a guard waiver.

The follow-up order is: shared normal-return facts for canonical completions;
producer-time freezing of lexical facts; retained logical declarations through
rule instantiation, absence lifting, snapshots and projections; existing shared
typing for generated derivative programs before reentrant AD; then checked
nullable-scalar product joins. `BindCtx/genlet` and absence-lifted slots are
specific declaration-transport seams to audit, not grounds for a new AD function
inference registry. Explicit absence may join a declared scalar; unknown Object
must still refuse. Optional arrays stay held until zero construction and ownership
are checked. Original producers, singular pullbacks, source guards and rounding
checkpoints must remain observable. Controls include both branch orientations,
alias shadowing, nested projections, unknown inactive values, differing tangent
declarations, malformed schemas, and absent versus present-zero cotangents.
The shared completion-presence foundation is separately reviewed and qualified:
Object/reference-check contexts forward only already-owned operand facts;
primitive contexts prove presence only on successful normal return, without
payload-width or policy authorization. An original-versus-DCE oracle retains
the failing reference check and its exception order. The cold presence,
local-product, DCE and effects gate passes 83 tests / 482 assertions. No nullable
product or tangent-tag producer changes are included in that foundation.
The next bounded normalization change transports existing `:raster.type/tag`
declarations through lexical references, aliases and the final result. It uses
the shared capture-avoiding scope traversal, publishes bindings only after their
initializers and removes shadowed declarations. It does not infer operation
types, copy physical JVM hints, prove non-nullness or relax product schemas.
Conflicting reference declarations refuse pending an owning refinement proof.
An independent review found qualified-quote traversal; shared substitution and
alpha-renaming now preserve both quote spellings, with direct and nested data
identity controls. Executed source and bytecode controls preserve nil, String
and arbitrary Object identity despite logical numeric declarations. The cold
six-namespace normalization, shared-scope, conditional, initializer and reverse-AD
gate passes 112 tests / 544 assertions. This is not nullable-product closure or
a fix for marketing's separate runtime/AOT lifecycle invalidation.

The subsequent product-closure investigation found two concrete declaration
losses rather than a need for AD-local arithmetic inference: canonical Object
snapshots retained physical `Object` but lost the operand's logical declaration;
and derivative-region normalization started empty even when the actual formal
owner and visible outer value both retained Double declarations. The follow-up
imports those declarations through the same lexical scope engine, never through
private child catalogs. Exact Object completion supplies only a logical value
declaration, with conflict refusals and source/bytecode identity controls.
Presence is frozen at each initializer before rebinding; only the current
record may publish the new result's facts. A recursively checked product join
admits explicit absence with an already-declared scalar tangent, not unknown
leaves, mixed widths, arrays, malformed schemas or active absent ports. The
joined declaration proves neither presence nor a primitive carrier.
The approved optional-cotangent sum's nil identity is projected only for lexical
operands/literal nil, retaining the selected input's Object completion. Compound
operands, changed helper definitions, lexical callees and call-level denial
flags retain the original call. The first-through-third cubic derivative probe
now returns its analytic values in a fresh JVM. The cold ten-namespace gate
passes 182 tests / 981 assertions; the existing first-through-third derivative
tests separately pass 3 tests / 9 assertions. Full CI remains required.
Object-helper completeness warnings remain visible, so
this is not a claim of optimized higher-order AD or complete AD coverage.
Inspection of completed unit job 32391 (`bb27191d`) confirms the Q4_K exact-bit,
Long/SIMD, raw-evaluation test-consumer and outdated scan-reason failures are
gone. Its remaining failures include HVP consumers (repaired in the subsequent
`5c745536`), schema/tag closure, training, native guards and integer/control
proofs. This is a measured delta, not broad green CI or merge acceptance.

Marketing's downloaded 0.2.1690 passes its inline/lagged workflow qualification
(49 tests, 757 assertions plus both fresh workers). Helper migration separately
fails runtime-gradient reuse after AOT with `:ad-normalized-stale-rules`: template
revision remains 238, compiler revision advances 1560 to 1564. Attribute the
definition changes before changing invalidation; a passing warmed small control
does not reproduce or close this lifecycle bug. Cold time/RSS are not improved
by this functional qualification.

The preserved helper fixture now reproduces on campaign source as well: runtime
and AOT both evaluate to `2.0620095064443418`, then retained runtime reuse refuses
with compiler revision 1699 to 1703 (template revision 239 unchanged). Tracing the
four actual changes identifies two generated `deftm` definitions evaluated in
the marketing namespace: each publishes a result descriptor and registers its
method. This is deliberate source publication by the bridge, not a numerical
failure or a mysteriously mutating AOT cache. Raster's own
`compiled-value+grad-fn` has the analogous named-wrapper pattern.

Reviewed continuation: extract one anonymous retained-entry compiler from the
existing JVM AOT path after definition resolution. The Var API remains an
adapter; both use the same source preparation, forward passes, loader and
hoisting/emission. Anonymous entries return diagnostics instead of mutating a
Var and retain actual source namespace/policy, annotations, return contexts,
captures and definition observations. Share typed-body AD preparation rather
than fabricating a Var or introducing another differentiation path. Existing
`ftm` and source-owner helpers are precedents, not substitutes for the complete
AOT owner contract. Keep epoch and changed-definition guards intact: finer
invalidation would require historical selection and negative-lookup dependencies.
Acceptance is an already-retained MMM gradient surviving subsequent anonymous
compilation without namespace/GF publication, with unchanged template/callee/
overload mutation refusals. The local implementation now reuses the existing
metadata-carrier resolver and full JVM AOT path rather than adding another IR
or compiler. It retains actual namespace identity, checked source policy,
annotations and the physical Object return contract. Captured closures,
unchecked arithmetic and anonymous native/GPU targets remain explicitly held.
The generated gradient wrapper uses `ftm`, without interning a definition;
diagnostics attach to the returned anonymous callable. Warm anonymous-entry
qualification passes 10 tests / 55 assertions, including retained-gradient reuse,
unchanged definition revision and Var-held source-context routing. Static review
found no remaining bounded defect. This is not yet released, and the downstream
bridge still needs migration and its actual MMM lifecycle qualification.

The compatibility gate also independently reproduced a silent source-product
regression: lexical substitution/alpha conversion rebuilt vectors without their
runtime metadata, changing the primal branch and derivative. Both vector walkers
now preserve original metadata unchanged; the existing source-product oracle
passes without changing its expectations. A nested-vector control covers both
walkers. A separate product join now forgets absence for an exact unknown leaf
only at a path inactive in both validated arms; it never derives a tangent dtype
or creates a tangent edge there. Nested, reversed, repeated-join and active-unknown
denial controls accompany this repair. Both changes received static review.
The fresh-JVM combined gate passes 69 tests / 423 assertions across anonymous
AOT, lexical utilities, product ports, conditional ports and source products.
This is bounded source/JVM evidence, not a green integration PR or device result.
The additional four-namespace owner/lifecycle/replay/function-algebra gate is not
green: 41 tests / 187 assertions report two failures and one error. The native
replay refusal stops at unclosed value completion before the expected guard
diagnostic, and `D-pipeline-rewrite` still errors. Do not treat these as covered
by the narrower passing product gate; both remain in the integration backlog.

Residual follow-up: the actual failing `D4-compiled` conditional paired a
two-field result with every forward local as its residual. The two forward
arms computed different numbers of locals, although neither reverse arm used
those locals. The producer now retains an ordered free-variable projection of
the reverse bindings and returned parameter adjoints; both saving and restoration
consume that exact projection. All forward expressions and their evaluation
order remain unchanged. No padding, product-schema relaxation or inactive tape
claim is introduced. Truly incompatible live residuals remain unsupported until
the branch owner supplies an explicit heterogeneous residual interface.

This exposes the ordinary empty-vector product in forward-over-reverse. The
existing dynamic tangent-zero fallback now recurses pointwise over vectors,
including the empty product, as the vector JVP rule already does. Unsupported
Object leaves/maps/lists still refuse, and array leaf zeros are freshly allocated.
The warm conditional/replay gate passes 24 tests / 250 assertions, including
selected unused-read failures/effects on runtime and invoked AOT paths. The
existing full `D-pipeline-rewrite` numerical assertions now pass. Four separate
Float cotangent-class law failures remain: template emission currently stamps
the desired input tangent dtype onto produced expressions without materializing
the conversion. Retained lexical declarations expose that producer-contract gap;
do not remove declarations, change expected classes, or weaken precision checks.
Fresh-JVM qualification passes five conditional/product namespaces (46 tests /
436 assertions) plus the dedicated product-zero and existing first-through-fourth
derivative pipeline checks (2 tests / 19 assertions). A subsequent strengthened
forward-failure control explicitly asserts invoked AOT status and passes 1 test /
13 assertions. Integration CI remains outstanding for this follow-up.

Reviewed next repair: close every scalar template output at the shared
`instantiate-template-ctx` boundary, for static and handwritten rules alike.
Keep actual RHS dtype evidence separate from the desired input tangent-space
declaration. The former `stamp-emitted-bindings` created circular evidence by stamping
the desired LHS tag onto the RHS and publishing `(or desired inferred)` as its
actual dtype. Preserve independently established intermediate/array declarations,
but do not feed an unfulfilled scalar output declaration into its own inference.
Evaluate each original RHS once as a flat raw binding, apply the existing
nil-safe `tangent/project-expr` at each returned scalar port, and publish the
target declaration only after that projection. No `:materialized? true` without
independent presence evidence. Ports returning an existing symbol, nil, or one
shared source with conflicting Float/Double targets need separate projections,
not mutation of that shared source's metadata. Keep kernel/SOAC RHSs flat.
Acceptance includes the four existing Float laws, checked overflow/rounding,
nil preservation, shared mixed-target outputs, static/custom rules and nested AD.
This is a producer-contract repair, not per-operation dispatch or a change to
AD algebra. The local implementation replaces that stamping path with
`close-template-output-ports`, sharing the existing scalar projection's retained
value-evidence helper. Dispatch inference may widen conditional branches; it
therefore remains distinct from exact dtype evidence used to elide conversion.
The normalized scalar owner also transports its existing argument declarations
into template calls, with a conflict refusal rather than silent metadata override.
Contradictory shared scalar/array or distinct-array-dtype ports refuse; mixed
Float/Double scalar ports receive independent projections. No global tags or
existing array/intermediate declarations are erased.

Warm qualification passes nine affected namespaces (131 tests / 900 assertions)
and five existing Float/HVP/derivative-pipeline tests (83 assertions). Added
declaration-conflict and incompatible-interface controls pass in a subsequent
four-namespace gate (46 tests / 410 assertions). These are source/JVM results,
not device/performance evidence. Final fresh-JVM qualification passes nine affected
namespaces (133 tests / 906 assertions) and five Float/HVP/derivative-pipeline
tests (83 assertions): 138 tests / 989 assertions total. Static review found no
confirmed remaining bounded defect. Integration CI qualification remains required;
the draft PR is not ready to merge. No source arithmetic contract was relaxed.

### Residency consolidation: ordered host observation

October 19 implementation checkpoint: the generated terminal region now has a
private JVM executable owner, compiled under the numerical producer's previously
observed definition, namespace and arithmetic policy. It freezes historical
dependencies, rejects mismatched program/contract identity and argument carriers
before source effects, and fences successful publication with definition guards.
This owner grants neither physical scalar exports nor completion evidence.

Terminal source projection uses resolved Methods and actual captured IFns rather
than unrestricted macro/inline expansion. Float conversions retain their actual
operand overload and checked/unchecked policy. Unknown helper primitive ABIs,
mixed primitive joins and unaccounted JVM hints refuse. The supported Clojure
1.12 control forms derive carriers from children rather than form tags; those
tags are preserved without becoming evidence. Exact Object hints are admitted
only on already Object-valued locals. Other hint contexts remain unsupported.
Independent source-oracle tests cover these distinctions.

Review exposed a shared source-resolution defect: a namespace alias can redirect
a qualified Class/method spelling to a Var even when the retained Method is
correct. The existing ordered-call verifier now rejects namespace/alias masking
of the declaring Class qualifier. Its regression first demonstrates actual
Clojure compiler redirection, then checks refusal. These are sequential fences,
not a concurrent Namespace mutation/ABA guarantee. Intrinsic projection assumes
standard Clojure definitions at backend bootstrap and guards later replacement.

The warm affected gate passes seven namespaces, 70 tests / 479 assertions. A real
frontend capture of mixed-scale-energy-gradient successfully compiles its owned
terminal expression and returns Float values with matching signed-zero and NaN
bits; Double/MAX_VALUE throws as required. This is a terminal-owner probe, NOT
end-to-end equation-first execution: ordinary compilation still refuses the
unintegrated return boundary. The earlier cold 78-test gate predates the final
hint adjustment and is not fresh-load qualification of that adjustment.

Next acceptance remains the public compile/execute/replay path, not another
standalone helper: attach portable correspondence to InvocationPlan, retain its
private executable with the compilation owner, account for the final source-order
observation, validate actual physical exports, and observe only after producer
completion. Preserve resident-only refusal and complete return coverage until
that entire transition is supported. Audit graph-liveness prefix pruning before
omitting any pure-but-throwing scalar computation. Current exact-head CI remains
red and PR #1539 remains draft; all eight campaign obligations remain open.

Definition-policy prerequisite (October 19): named `deftm`, parametric templates
and their concrete specializations now retain the actual producer's
`*unchecked-math*` context, sharing the existing anonymous-function policy schema.
Frontend facts and equation-first options transport it separately from dtype.
Lazy JIT uses the pending specification's policy; renewal uses the original
lineage; specialization uses its template. Source metadata must correspond before
walking, and a policy-bearing plan must capture a genuine source entry rather
than silently downgrade to unowned emission. Missing, changed and removed policy
metadata are rejection controls; a deterministic mutation during capture checks
that the pending implementation and JIT spec remain untouched on failure.

The affected warm REPL gate passes seven namespaces, 50 tests / 426 assertions
(source policy, source-entry lineage, required JIT walking, parametric dispatch,
specialization returns, anonymous AOT and JVM source-owner compilation). This is
bounded ownership/transport evidence, not device execution or a green integration
gate. The separate Object-array `specialize.clj` path, mixed-policy helper
inlining, general policy-aware target lowering and terminal host observation are
not established by this receipt. Policy-less legacy walking retains its prior
behavior without acquiring source ownership; old pending JIT specs with missing
policy fail closed. Sequential source guards do not claim general concurrent
metadata mutation/ABA protection. All eight campaign obligations remain open.

Terminal correspondence groundwork (October 19): the existing InvocationPlan
namespace now derives and independently rechecks a portable terminal-result
description. It preserves the entire final expression, scoped locals, lazy
branches and canonical completion terms, with deterministic logical Double/Long
ports identified by producer equation/result position. It refuses unrepresented
free values, unsupported carriers, missing policy and nonterminal source shapes.
The actual mixed-gradient candidate yields its one Double dependency without
rewriting its conditional checked Float/Object return. The focused warm plan and
materialization gate passes 23 tests / 177 assertions. Review requested explicit
authority-limit controls: descriptive analysis does not execute an effectful
terminal call, and unchanged producer IDs do not certify changed computations.
Both controls pass. Portable encoding is not purity/callee legality evidence.
This helper is not yet activated in promotion, source-order admission or runtime:
existing refusals remain until actual source ownership, physical scalar exports,
completion and an explicit observation API form the complete vertical.

The independent residency review distinguishes established physical contracts
from an incomplete source-level phase boundary. AbstractValue/LinkValue separate
logical values from physical leaves; LinkPlan checks overlapping writes and
composed initialization/alias contracts; execution events, output leases and
pending-transfer retention express completion and lifetime obligations. These
foundations must be reused, not replaced by another residency runtime.

The current public mixed-storage gradient exposes the missing composition:
a device-produced Double scalar is followed by a conditional checked Float
conversion and Object completion. Rank zero does not mean host availability.
An IEEE device cast does not implement the host conversion's overflow exception,
and a host preparation expression cannot depend on unfinished device work.
Current refusal is correct, but does not satisfy the public compile vertical.
Exact-head `962b054a` emission CI confirms this terminal source-result failure;
unit and OpenCL gates also remain red. No merge qualification follows from the
138-test bounded AD gate.

Implement one typed terminal host-result continuation at the existing invocation
boundary. Its contract must identify the actual producer values, their physical
scalar exports and completion dependencies, the retained closed host computation,
and the complete source result structure. Preserve checked conversions, lazy
branches, Object completion, definition/source-policy ownership and evaluation
order. Execute the host computation once after the required producers complete.
Replace the restricted source-result proof with a proof of this continuation;
do not waive result coverage or reinterpret a destination type as value evidence.

The initial admissible region is terminal: no later device consumer or observable
source operation may be moved across it. Resident-only entry points must refuse
when they cannot return the promised resident result without host observation.
Do not silently download resident chains. General intermediate host/device
crossings require explicit ordered scheduling; deferred device error flags are
not equivalent if later effects can occur before the host notices failure.

Keep logical numerical encoding, physical realization/ABI, placement,
initialization, ownership/escape and completion as separate proof obligations.
The current `:resident-scalar-buffer` representation and pristine-value admission
predicates are consolidation targets: document and remove their coupling only
with replacement proof coverage, not by broadening whitelists. Centralize the
transition contract (prepare, acquire, submit, complete, observe) and state which
semantic/physical/effect/alias/completion changes invalidate each certificate.
This is an execution-layer completion beneath TypedSOAC, not a replacement IR.

Acceptance requires source-versus-execution tests for Float rounding, overflow,
NaNs/infinities/signed zero, selected and unselected nil conversions, failure
ordering around effects, Object boxing, multiple result leaves and repeated
invocations. Add mismatched physical ABI, changed/shadowed source definition and
downstream-device-consumer refusals. Hardware-free contract/emission evidence,
actual OpenCL/Level Zero execution and performance measurements remain separate.
The broader distributed placement/planning obligations below remain open.

Implementation seams confirmed by follow-up review: retain the validated
continuation at InvocationPlan's existing attribute boundary, not its prefix
`:steps`. Construct it during frontend terminal-source analysis and whole-source
promotion; both host coverage and result coverage must consume the same owned
contract. The existing ordered invocation executor already acknowledges numerical
stages after synchronous completion. Extend that owner and its final realization
validation to perform explicit terminal observation. Keep exact producer/export
relations in the contract; rank-zero dtype alone is not export evidence.

`gpu.compiled/invoke-compiled` is a resident API today. Its output tree, resident
view projection and leases must not silently change to return host objects.
Provide a distinct, explicit source-result observation boundary; reject unsupported
resident requests before mutation/replay. The JVM typed-scalar interpreter's
numeric result coercion and restricted expression grammar are not already an
Object/nil completion evaluator. Reuse the existing retained source/bytecode
completion owner when closing the result computation, rather than evaluating
arbitrary source or inventing runtime implementations of canonical IR terms.

The first implementation prerequisite now observes the actual outer lexical tail
at JVM emission. Its final free ports come from emitted local slots/classes,
including shadowing, and its actual result context remains distinct from the
outer method context (Object regions strengthen call-result preservation).
The owned whole-body record also retains the reflected method's actual return
descriptor/carrier. `compile-terminal-result-region` reuses the existing physical
port outliner and JVM emitter, excluding prefix initializers while preserving
checked casts, lazy branches, Object/nil results and reference return checks.
Original occurrence/definition guards remain; copied, stale, ambiguous, altered
or actually split owners refuse. This is compiler-local ownership evidence,
not an authenticated portable certificate or a device scheduling permission.

An initial fresh-JVM owner gate passed 47 tests / 445 assertions, but stronger
independent plain-Clojure oracles then exposed an existing explicit `float` bug:
the JVM emitter used raw narrowing and accepted Double overflow/infinities instead
of throwing. Explicit `float` now shares the existing RT cast-emission path with
`unchecked-float`, choosing `RT.floatCast` rather than raw descriptor coercion.
Implicit argument conversions remain unchanged. Review found the whole-bytecode
AD consumer also rebuilt every cast as raw argument completion; retained cast
evidence now identifies the actual checked source operation, and that consumer
preserves its operation and mandatory evaluation even when its result is unused.
This is a source-semantics repair, not new reassociation or fast-math permission.

The new outliner namespace is discovered by the ordinary CI shard planner.
Independent plain-Clojure oracles cover primitive/Object cast overloads, terminal
conversion, reference return checks and array reads. Retained cast observations
require an explicit lowering kind; missing or unknown provenance refuses rather
than becoming raw conversion evidence. Checked and unchecked overflow paths have
separate actual whole-bytecode AD regressions. Final fresh-JVM qualification of
the combined repair passes six namespaces, 69 tests / 850 assertions. Full
integration CI remains required. The result
continuation's semantic port/export correspondence,
explicit device completion/readback, host-facing invocation entry and resident
preflight refusals are NOT implemented by this prerequisite.

PDE CI triage found an obsolete representation assumption in the rectangular
coverage mutation test: its first step is now a host equation, and the actual
numerical step is a functional map rather than the former effect map. Merely
selecting the numerical step is insufficient: effect mutation then edits nothing.
Retain the public functional-map acceptance and exercise rectangular effect
coverage with an explicit effect-map fixture; do not drop the negative guard,
overflow, narrowing or injectivity checks. The reviewed repair now uses a validated
6-by-10 effect-map fixture with explicit Long address evidence, asserts that each
store mutation actually changes one store, and executes the positive fixture
against sentinel-filled storage. Public prolongation/restriction separately retain
complete-destination initialization assertions and device-versus-JVM checks.
The full affected namespace passes 10 tests / 115 assertions, including local
Level Zero execution. Independent static review found no coverage weakening.
This closes the stale test assumption, not the remaining integration failures.

### Source conversion contracts beneath residency

A subsequent independent REPL check found a second interpretation mismatch:
canonical Double-to-Float conversion accepted `:overflow :ieee` together with
`clojure.core/float`, although projecting that term to source throws on
`Double/MAX_VALUE` while an IEEE narrowing returns infinity. The shared
`scalar-conversion/source-policy` now separates descriptor-owned source casts
from implicit numerical/storage conversion. Checked Double-to-Float retains
`:nearest-even/:trap`; unchecked narrowing retains `:nearest-even/:ieee`.
Canonical validation and frontend construction consume this same policy.

Raw checked Float calls also retain the existing checked-source-cast evaluation
obligation before canonicalization. Totality uses checked source constant evidence
or retained source dtypes, never IEEE constant folding or a destination hint.
The scalar KernelBody lowerer admits a checked narrowing only when its actual
lowered literal proves it cannot throw; other cases refuse rather than silently
emitting IEEE conversion. Existing trapping-conversion fusion barriers therefore
also cover checked Float producers. This does not implement a device exception
protocol or the missing terminal host continuation.

Independent review found no further confirmed soundness blocker in this bounded
repair. The focused conversion/body/effect gate passes 77 tests / 829 assertions;
two further fusion/order controls pass nine assertions. These include overflow,
infinities, NaN, signed zero, lexical callees, nested unchecked Float infinity and
failure before later output writes. Intentionally IEEE test terms now name the
unchecked source operation; checked-source refusal controls remain separate.

Broader qualification is still red: the initial seven-namespace affected gate
reported 35 failures and 11 errors. A subsequent concise inventory, after the
masked-reduction fixture repair, still shows fusion expectation failures,
unclosed terminal scalar programs, the public layer-norm backward checked cast
and missing native vector emission. That inventory suppressed standard assertion
reporting and is diagnostic only, not a passing gate. Do not change source casts,
drop acceptance assertions or relax exceptions to make these green. Finish the
ordered observation/realization boundary and distinguish actual source conversion
from compiler-owned storage conversion using retained producer evidence.

The adjacent source-cast matrix found the same distinction for Float/Double to
Int/Long: ordinary checked casts throw on overflow, whereas unchecked conversion
saturates. `source-policy` now records `:toward-zero/:trap` for those checked
casts, independently of the existing saturating numerical conversion policy.
The shared scalar lowerer applies the same refusal/closed-literal proof boundary.

Review also identified an overload-sensitive proof issue: a boxed Number cast can
truncate before checking an integral bound, unlike the original primitive FP
cast. `numeric-constant/source-value` now selects actual Float/Double primitive
overloads for literal source casts; the separate numerical/IEEE `value` path is
unchanged. Tests compare the proof itself with independently typed source
oracles at fractional Byte/Int boundaries, neighboring Long boundaries,
infinities and NaN, and assert the literal's actual carrier Class. The focused
four-namespace gate passes 88 tests / 1,178 assertions in the reused REPL.
Hardware-free tests also require all three C-family emitters to refuse an
unsupported trapping FP-to-integral KernelBody cast. Independent final static
review found no remaining confirmed blocker in this bounded delta. This is not
a cold integration gate or qualification of the broader currently failing suite.

The real mixed-storage energy-gradient REPL probe still reaches
`:structured-control-source-result`, retaining its conditional Float/Object
completion and one free Double-result port. An explicitly experimental host-only
projection through the existing `source-owner/compile-function` agrees with
ordinary Clojure on Float carrier, rounding, signed zero, NaN and checked
overflow/infinities. This experiment deliberately chose a generated-source
policy and provided no runtime guard; it is NOT production ownership evidence.
The named public deftm metadata inspected here does not retain that policy.
The semantic producer must therefore supply the actual generated source context
and dependency guard, rather than infer them from a return hint, ambient REPL
bindings or an unrelated original bytecode tail.

The existing source-owner seam now has permanent conditional Object/nil-result
and unprojectable bytecode-completion controls (eight tests / 29 assertions).
Next retain the semantic terminal region/context at its producer, derive exact
physical scalar exports, and execute its guarded observation through the existing
ordered executor and an explicit host-result API. None of those integration
steps is established by this host-only experiment.

Exact-head `9ca37eb6` emission CI (32484) additionally reaches a checked
Double-to-Float conversion inside the functional mixed-scale map. Independent
CPU invocation of `par/scale Double/MAX_VALUE` over a Float array throws too;
this is not permission to relabel its conversion IEEE. Terminal result
observation alone will not close that map's execution contract.

After the terminal complete/observe/publish boundary, reuse that transition for
a typed checked-stage result obligation. The first status-buffer realization
may cover only checked Double-to-Float with fresh private non-escaping output,
no aliases/donation, one conversion kind, and otherwise total independent
element computations. Purity alone is insufficient. Preserve the checked
semantic operation; explicit KernelBody predicates/branches and existing
AtomicRMW may realize its validation, with private status storage accounted for
by LinkPlan. Reset before every replay, await actual kernel completion and
observe success BEFORE initialization credit, cursor advancement, consumers or
output publication. Failure must not yield a ready output or output lease.
Do not make status-buffer representation the semantic model or silently add
downloads to resident-only APIs. General effectful/otherwise-throwing maps,
multiple error classes and checked FP-to-integral device realization remain
held until their source-order and carrier-specific contracts are proved.
Independent design review confirmed this bounded seam and requires replay-reset,
failure-before-consumer, alias/donation refusal and extra-throwing-operation
negative controls. No checked-stage runtime implementation is claimed here.

The next physical realization was first exercised as a test-only KernelBody
fixture, not a production schedule: compare the original Double to exact
`+/-Float/MAX_VALUE`, convert/write only valid lanes, and overwrite each lane's
status on both success and failure. NaN remains source-valid; infinities and the
immediate Double neighbours outside the finite Float interval fail. Checking
whether the narrowed Float is infinite would be incorrect at those neighbours.
The experiment runs failure -> mixed -> success -> mixed to detect stale status.
Local generated CUDA source compiled to PTX with nvcc 12.4, and HIP source to a
gfx1100 code object with the installed HIP/Clang toolchain, both without hardware.
These are physical feasibility checks, not public checked-map support, general
initialization evidence, CUDA/HIP execution or performance qualification.

A bit-only status cannot reproduce value-dependent source diagnostics. Preserve
the offending original Double in private stage diagnostics, and prove the selected
failure lane's correspondence to source traversal before reconstructing the source
exception. Diagnostic storage is only readable for a failure-selected lane whose
write completed; successful output initialization remains conditional on status
observation. Zero-length success must require neither launch nor status read.

The immediate JVM helper/ordered-reduction slice now has source qualification
on the unchanged marketing formulation: runtime and invoked AOT gradients agree
with nine-coordinate finite differences across changed parameters and data.
This is not full marketing-suite acceptance. Its broader scan-based formulation
still declines at temporary array allocation/scan ownership. AOT-created new
definitions also advance the global epoch, so a previously cached runtime
gradient may correctly refuse while its consumer cache retains that stale
callable. Numerical correctness and cache-lifecycle recovery are separate gates.

Continue in this order, interleaving independent small correctness cleanups:

1. Land the owned helper/reduction vertical with historical source, carrier,
   conversion and dependency checks intact. Extend allocation/scan through the
   same owned region producer and shared recurrence assembly; saved pullbacks
   come before any forward-replay optimization. Do not add a per-model rule.
2. Profile current boxing, native calls and staging with warmed JIT and matched
   scopes. Optimize measured residual/product boundaries while retaining absent
   cotangents, alias preflight and completion-backed ownership. Object signatures
   or redundant CHECKCASTs alone are not allocation evidence.
3. Integrate the optional MLIR/Pattern prototype at KernelBody through shared
   executable identity, native invocation, cache and lifetime contracts. Keep
   ordinary deftm/semantic lowering distinct from this scheduled backend entry;
   LLVM/MLIR must remain optional and must not add another runtime authority.
4. Close unchanged Gemma/SFT, bounded GSDM trajectories/statistics, city and PDE
   acceptance; reconcile TypedSOAC paths and retire covered adapters/tests.
   Preserve separate compile, prepare, submit, transfer and device measurements.
5. Requalify public CUDA/HIP fixtures and the external CUDA runtime against an
   immutable integrated main. Then complete the existing distributed planner,
   collective/halo and durable/multilevel simulator gates on this laptop, without
   claiming multi-host/fabric performance from co-located execution.
6. Consolidate support/evidence tables, reachable debt and test partitions after
   their independent positive and refusal obligations have replacement coverage.

The bounded JVM regions do not establish generic native/device AD or permission
to discard source owners. No SOTA performance or complete compiler claim follows
from this qualification. See `normalized-ad.md` for the explicit admission and
remaining representation boundaries.

### October 16 continuation

Fresh hardware-free planning qualification found that an explicit integer compute
estimate above 2^53 lost nanoseconds through a `double` conversion. The simulator
now preserves already-integral estimates exactly, and refuses integer durations
outside the long range. Controls cover timeline, aggregate busy time, makespan,
certificate costs and serial dependencies; fractional estimates keep their existing
rounding. This is an item-6 correctness increment, not measured fabric evidence or
closure of the remaining distributed, halo and durable-state acceptance gates.

The bounded normalized host AD composition now attaches an exact terminal additive sum
to an already-admitted canonical multi-equation pure producer. It reuses producer JVP/VJP,
owned residuals and the existing sum checks; a retained typed seed-broadcast equation joins
the transpose through the ordinary pure-spine composer. Independent controls cover both
contraction operands, nonsquare cotangent geometry, finite differences, repeated-input
map-node accumulation, changed inputs/seeds, mutation and replay/refusal. Direct map→sum
semantics remain unchanged. This is not general graph AD, device activation or default-engine
migration; product→sum without a map and scalar-active longer producers remain held.

The next resident slice composes canonical static producers with that scalar loss through the
existing sum adapter. Maps and explicitly marked fresh bilinear contractions share the phase
assembler, ordinary storage projection, strict binder, residual frontiers and lease owner.
Each marked primal/JVP/transpose independently reprojects its own axes, body, storage and
numerical contract. Unmarked contractions refuse before compilation; host-owned scalar losses
still fail ordinary resident promotion. Both selected contraction operands, both ABI orders and
a repeated-input three-node map producer are checked without allocation. OpenCL and Level Zero
exercise changed inputs, post-completion mutation, gradient checkpoints and lease failures.
No default-engine migration or model acceptance follows from this bounded composition.

Numerical evidence distinguishes source-expression order from target reduction association:
the retained checked commutative monoid permits a terminal Float tree, but does not authorize
factoring scalar/map or reverse-contribution checkpoints. A non-exact three-term fixture checks
every legal permutation and binary addition tree without tolerance or schedule assumptions.

The Gemma gate remains open. Same-device-input controls now narrow two GELU leaves:
the selected fused add→GELU forward is bit-identical on all 4096 elements, while the
selected backward has relative L2 error 2.67e-11 on identical operands. Saved activation
and incoming cotangent drift are approximately 6.53e-7 and 9.78e-7 respectively; changing
only the incoming cotangent explains more of that backward discrepancy than changing only
the saved activation. These are local attribution controls, not model acceptance. The
historical full-model 12/28 failure receipt has not been rerun after the derivative correction;
the fresh layer gate still fails 4/14. No tolerance or acceptance threshold has changed.

Continue with upstream same-operand attribution, representative normalized training graphs,
then workload closure and retirement. All eight original obligations below remain open.

### October 15 historical checkpoint

Normalized host AD now also admits one bounded bilinear segmented product, optionally
followed by a unary single-use map spine, through `linearize-equation-program`. The
product requires one selected active array, static positive dimensions, plain Float/Double
storage and checked operand/result AxisMaps. It uses the existing scalar derivative rules,
node composers and canonical runtime; there is no new rule registry or evaluator. Passive
operands and activation coefficients remain owned snapshots for delayed pullbacks. Scalar
captures may be shared across maps and retain inactive gradient slots.

Shared numerical-scope composition validates local and program declarations and preserves
every equation's effective policy; an absent/retained fallback join is not reassociation
permission. Equal-contract fingerprints remain unchanged. The mixed host controls cover
independent analytic JVP/VJP answers, finite differences, Float checkpoints, reversed ABI,
caller mutation, changed inputs and proof replay/refusals. This does not enable a device
contraction AD owner, arbitrary graph transposition, or default-engine migration. The real
Gemma, full GSDM optimizer/statistical and performance acceptance gates remain open.

The following AD consolidation sequence is interleaved with, not a replacement for,
the eight compiler obligations below. Further family-specific AD extensions are held
until the shared structural boundary is established.

### AD consolidation sequence

1. **Admission and evidence.** Extend the existing AD laws into an accepted/refused
   capability matrix. Distinguish known tangent spaces, proven NoTangent, unknown
   types, explicit inactivity and symbolic zero. Preserve actual precision checkpoints.
   Exercise constructor-time refusals without executing the user's primal.
2. **Shared structural transform.** Generalize the existing normalized program envelope
   to equation graphs: activity, topological JVP, reverse transposition and ordered
   fan-out accumulation. Start with test-only equivalence checks; retire superseded
   family orchestration only as coverage is demonstrated. Do not add another rule table.
   A bounded consolidation first replaces spine-only composition with the existing
   fusion authority for closed homogeneous map regions. It differentiates the exact
   rederived execution map, not the rounded node-wise source graph. This removes a
   topology special case but does not complete ordered graph cotangent accumulation.
   The next shared host-materialization prerequisite admits ordered homogeneous pure-map
   graphs with tuple/multiple declared outputs through existing per-map admission and
   equation realization. It preserves source order/checkpoints and adds no AD rules or
   runtime. Node-wise graph derivatives and precise residual-producer/borrowed-use
   relationships remain separate, consumed obligations in the existing envelope.
3. **Composition boundary.** Specify ordered parameter/result ports and owned residuals.
   Qualify an optional synchronous CPU foreign-AD adapter with independent analytic and
   finite-difference checks, changed inputs/seeds, mutation and cleanup tests. Keep
   VJP, JVP, batching and higher-order support independent. No core Python dependency,
   second compiler cache or competing GPU session convention.
4. **Operator and control coverage.** Compose contractions, broadcasting, reductions
   and indexed operations through shared linear operators, then admitted conditionals,
   scans/recurrences and higher-order cases. General effectful loops may remain refused;
   unsupported derivatives must never silently become zero.
5. **Residual/resource projection.** Track semantic value/version, dtype, checkpoint,
   consumers and lifetime through existing memory witnesses, buffer views, leases and
   completion events. Borrowing, donation and recomputation require these proofs; a live
   allocation alone does not prove it still contains the saved primal value.
6. **Workload closure and retirement.** Qualify real Gemma/SFT, GSDM optimizer and
   statistical training, density and PDE cases; measure preparation and warmed execution
   separately. Only then consider default-engine migration and remove obsolete paths.

Acceptance uses the existing analytic, finite-difference, pairing, higher-order and
mutation/refusal tests, plus cold Float-first/Double-first acquisition controls. Pairing
and CPU/device agreement using shared derivative rules are not independent calculus
oracles. Hardware-free tests, local device execution and actual distributed performance
remain distinct evidence tiers. The real Gemma discrepancy below remains open; changing
tolerances or abandoning strict floating-point semantics is not an acceptance shortcut.

### October 15 implementation boundaries

| Boundary | Current opt-in implementation | Still open |
| --- | --- | --- |
| Normalized host AD | Canonical pure maps/fully fusible closed homogeneous map regions, map→ordered-sum and functional indexed-map owners execute generated primal/JVP/VJP programs with exact residual plans. Maps, their checked fused execution regions and a single map→ordered-sum support ordered multiple active array inputs. | Ordered node-wise graph transposition, nonfusing graphs, multiple-active map-chain→sum and indexed operations, conditional/loop regions and default-source retirement. |
| Resident normalized AD | Public lease-oriented `prepare-owner`, `prepare-sum-owner` and `prepare-indexed-owner` reuse the generic resident binder, named-entry owner and LinkPlan/LinkModule validation on OpenCL and Level Zero. | Empty native buffers, general scatter/DAG execution, concurrent mutation, host readback and default-engine activation. |
| Shape and storage premises | A checked self-`ShapeProjection` derives a private dimension from one public buffer; closed external-storage and external-dimension premises cross separate resident phases without fake parameters or initialization. Program-wide typed shape values remain in `ProgramCall`, while kernels that do not consume them keep an empty scalar ABI. | Automatic source-free projection of compound/shared dimensions, per-call geometry changes and arbitrary dynamic shape programs. |
| Acceptance | The source AD engine remains the default structural reference; shared scalar rules mean it is not an independent calculus oracle. The unchanged real Gemma gate still fails 12/28 checks and the same-CPU-operand layer control 4/14. | Real-model closure, full GSDM statistical training, Adam/EMA qualification and the other original campaign gates. |

The latest selected SGEMM leaf diagnostic used the same GPU operands and measured maximum absolute
error 2.3841858e-7 and relative L2 error 3.8074463e-8 against the actual CPU SGEMM. Upstream input
drift remains, so this narrows one leaf and does not retire the Gemma gate. All eight campaign
obligations below remain active.

### October 14 historical checkpoint

- Compiler-generated pure TypedSOAC now enters the existing equation-first scheduling/emission
  tail without fabricated Clojure source or another type-inference authority. Named-entry storage
  contracts, the shared owner and generic resident binder now support a durable test-only
  canonical Float primal/residual/transpose composition on local OpenCL and Level Zero.
  Independent Float-bit oracles, changed seeds/generations, caller-input mutation and
  epoch/callee/lifetime refusals pass. Public normalized-device adapters, general DAGs,
  delayed public pullbacks and default-engine migration remain open.
- The fresh unchanged real-checkpoint Gemma gate fails 12 of 28 adapter gradient checks. A
  same-CPU-operand layer control still fails 4 of 14, so inter-layer input drift alone is not the
  explanation. The reduced two-layer public chain passes its unchanged numerical bounds on two
  replays. These are distinct controls: reduced acceptance does not replace real-model acceptance.
  Localize the remaining discrepancy with same-device-input operation oracles before changing
  schedules, precision or tolerances. All eight campaign obligations below remain active.

- #1364 activates the independently audited scalar vertical; #1365 gives normalized AD one
  deterministic map-stage constructor/reconstructor. #1366 preserves one unused public ScalarIR
  checkpoint alongside observable state outputs and later independent effects. Genuine public
  1-layer/8-dimensional batch-2 GSDM SGD preparation and three resident Level Zero updates now
  succeed: all 30 weight fields agree with CPU within 2.98e-8, loss within 1.43e-6, and captured
  host state remains unchanged. This supersedes the smaller forward-only receipt below, not the
  still-open full-size GSDM statistical, Adam/EMA or real-weight Gemma acceptance gates.
- Checked normalized AD now composes one bounded pure map→map chain through the existing typed
  fusion and scalar-rule authorities. The canonical owner retains the original semantic/public
  interface separately from the independently rederived execution source. Saved coefficient
  origins describe ordered source lineage and exact scalar-primal scope, including ANF values;
  they do not pretend every coefficient belongs to the first source equation. The existing CPU
  runtime executes generated primal/linear/transpose maps, with private snapshots, unchanged
  source argument order and exact Float checkpoints. Scalar-capture residual keys retain their
  already-inferred metadata at the shared map-stage owner. General DAGs, device normalized AD
  and default-engine migration remain open.
- A fresh larger public GSDM SGD probe (2 layers, embedding 32, batch 2, cosine-100 schedule)
  prepares and executes three resident Level Zero updates. All 48 weight fields remain finite
  and agree with compiled CPU within 1.20e-7; loss errors are 4.20e-5, 9.54e-7 and 2.86e-6.
  Captured host state remains unchanged and the artifact closes afterward. CPU generation took
  43.8 seconds and GPU preparation 146.3 seconds on the loaded warm process. These are bounded
  numerical/preparation receipts, not statistical training, steady-state performance, Adam/EMA,
  real Gemma acceptance or a SOTA kernel claim.

- #1358 checks Gemma's selected matrix dispatches against independently retained contraction
  identities, not a fixed kernel count. #1359 performs existing effect-aware DCE only after whole
  normalized-source admission and rebuilds coherent typed evidence afterward. Tiny Gemma retains
  its CPU/device numerical trajectories while seven unused contractions disappear (57 → 50).
- Buffered Adam and LoRA public replay tests freshly pass on CPU, OpenCL and Level Zero
  (6 tests / 310 assertions). #1360 adds explicit captured aggregate `:state` and exact
  path-keyed field outputs through that same boundary, merged after independent review and
  all seven CI contexts. Donation and whole-tree output ownership remain separate obligations.
- Terminal scalar activation now names only an independently admitted device continuation.
  Existing allocation/extent and complete-write authorities prove singleton capacity and a unique
  prior writer; the strict scalar lowerer proves retained types and nontrapping arithmetic.
  Production scheduling and public result admission independently reconstruct that proof from
  the enclosing semantic spine, not a resident marker or detached equation. Unsupported candidates,
  including residency lost after optimization, preserve the original source return and refusal.
  There is no loss-specific rule or host readback. A bounded one-layer Float GSDM public forward
  replay on Level Zero matches two changed CPU batches within 8.94e-7 and 1.42e-7; this is not
  training or statistical acceptance. Generic public tests also replay three exact Float-checkpoint
  to Double results on both local backends, including output lifetime and input immutability.
- Direct ScalarIR → extent-one SegMap scheduling now has an inactive substrate. Exact semantic
  correspondence retains type metadata and floating-point bits; the existing KernelBody and ABI
  paths own physical loads/stores. It requires an explicit resident singleton result and does not
  change scalar dispatch policy by itself. The separately audited terminal activation uses this
  same schedule, preserving logical rank zero and physical one-element storage. New tests cover
  all four target dialect emissions and changed-input
  bit-exact Float-to-Double execution on local OpenCL/Level Zero (7 tests / 76 assertions).
- CPU batch/conditioning GSDM controls pass (11 tests / 96 assertions). Full resident
  forward/VJP/update trajectory and statistical acceptance are still open. The historical
  real-weight Gemma 11/28 failure remains an unchanged acceptance obligation.
- Selected normalized-AD runtime/derivation gates freshly pass (30 tests / 541 assertions).
  General equation composition, normalized device execution and default-engine migration remain
  open; the source AD engine is still the independent default oracle.
- Distributed planning, cost, training, AMR-cycle contract/restore/state and compatibility
  hardware-free gates freshly pass (58 tests / 587 assertions). Local OpenCL/Level Zero cycle
  execution and mapped-file fresh-owner restart pass two existing tests (78 assertions), including
  independent coordinate/conservation and lineage checks. Local data-parallel AD → all-reduce → SGD,
  unequal batches, multistep updates and explicit rank-zero loss pass 4 tests / 196 assertions.
  These are not claims of production persistence, multi-host transport or fabric performance.
- A paired resident liveness measurement confirms 139 → 132 bound steps and exact equality of
  all 14 reset-state outputs. Host and device timing samples were nonstationary (CV 0.23–0.35),
  so timing is inconclusive. No speedup, regression or tuning decision follows from that run.

### October 13 historical checkpoint

- Main contains #1333 and #1334, each squash-merged only after all seven exact-head CI gates passed.
  Generated optimizer scalars follow each parameter leaf's precision. Successful static executable
  validation is reused only within a bounded, thread-confined staged request; dynamic checks remain.
- The corrected 2-layer, 32-dimensional Float GSDM train step compiles on Level Zero. Three seeded
  single-example steps execute and produce finite values. FP64 gradients and sampled central
  differences support the conditioning diagnosis, but FP32 near-zero key-bias gradients cause
  Adam-sensitive weight differences. This does not establish trajectory or statistical parity.
- Preliminary instrumented staging time improved from roughly 3.5–3.7 seconds to 1.7–2.0 seconds.
  This remains far too slow for the tiny workload and is not a controlled throughput benchmark.
  The step still crosses many separately staged graph boundaries.
- Whole-program resident compilation was interrupted at its 590-second diagnostic limit. Stack
  samples reached TypedSOAC fusion and final scalar DCE. No resident program or structured compiler
  refusal was obtained. Separate pass costs before another full-model attempt.
- The first bounded fusion baseline uses 8/16/32 independent maps with distinct extents and zero
  vertical candidates. Current enumeration nevertheless executes 7/15/31 producer exception
  analyses. This establishes avoidable analysis calls for that fixture, not the cause of the full
  timeout or a measured speedup. Retain candidate order and trapping controls when changing it.
- Normalized AD remains opt-in for bounded scalar/map/map-sum/indexed families. General equation
  composition, device execution, recurrence residuals and default migration remain open. The last
  recorded unchanged real Gemma gate failed 11 of 28 adapter checks; it requires a fresh coherent
  reproduction, not substitution by a microtest or a relaxed numerical oracle.

### Closed scheduling arithmetic safety

KernelBody index arithmetic denotes exact integer scheduling computations realized
at a declared machine width. A final zero, outer widening cast or inactive access
mask cannot waive an overflowing evaluated prefix. Closed literal/cast/add/sub/mul
trees use the existing bounded scalar-range proof, with recognition bounded before
child enumeration; proof exhaustion is an explicit refusal, not assumed safety.
Symbolic operands still need the separate actual lexical-guard/domain contract.

Verifier and C-family index emission share the existing launch integer-literal
width authority (Int when representable, otherwise Long). Emission delegates
signed literal spelling, including Long minimum, to the existing typed literal
emitter. This repairs an internal width disagreement, not source-level numerical
policy or a new type-inference registry. Typed source arithmetic keeps its original
checked/wrapping policy. Generated schedules must widen before multiplication;
this gate alone neither enables the symbolic CUDA workloads nor proves all index
arithmetic safe. Unchanged producer, inclusive-Long-loop and target source tests
are distinct controls alongside the negative prefix, budget and metadata cases.

### CUDA integration: bounded authority audit, not a replacement campaign

The October 13 compiler/runtime reviews of contributor PR #1338 (`063e7fc0`) identify
shared correctness/completeness work beyond transport: checked intermediate index arithmetic,
actual lexical guard scopes, known nested-return precision, conditional artifact obligations,
and rank-zero resident composition. Conservative refusals are distinct from numerical or
soundness defects. Contributor hardware receipts qualify that frozen tree only, not newer main.
Newer upstream observable-effect DCE and source/allocation placement fixes must survive reconciliation.

Continue the original eight obligations through these bounded integration gates:

1. **Ownership/completion/prepared replay.** Integrate through the existing root, Cleanup,
   ordered ABI and executable/session boundaries. Sync and async transfers use the same exact
   range validator and lifetime-use leases; retire duplicated backend borrow authorities.
   Check free-first and submission-first races, host publication, unknown outcomes, changed
   inputs and cleanup. Async return is not proof of independent queues or physical overlap.
2. **Shared compiler reconciliation.** Separate nested-return precision, lexical guard/index
   proof closure, condition-preserving artifact transport, and resident scalar/result-view
   composition. Retained source dtype evidence remains the primary type authority. Do not
   import recursive source-type reconstruction as a second inference engine. Negative tests
   must reject overflowing intermediate prefixes, borrowed/unrelated bounds, omitted guards,
   forged obligations and contradictory body/ABI/source facts. Minimize duplicated retained
   authority instead of perpetually expanding pairwise checks between copies.
3. **Exact-tree CUDA qualification.** Hand Sam an immutable integrated main commit and
   commands for generated public workload replay, numerical checks and sanitizer runs.
   Keep public runtime activation separate from source-only, mocked transport and memory
   foundations. Record host submission/staging, event device intervals and synchronized
   end-to-end timing separately; event units do not imply nanosecond precision. Reuse common
   measurement/oracle/tuning identity rather than creating a CUDA-specific cache or planner.
4. **Resume held workload/AD acceptance.** Close resident training and normalized AD with
   unchanged Gemma/GSDM/density/city/PDE oracles. CUDA integration does not waive these gates,
   AD obligations or the original distributed/durable completion work.

The contribution remains the credit/evidence umbrella until every source change and deferred
test has an integration or explicit deferral entry. Ask the contributor to pause broad rebasing
and implementation during extraction; request device verification when the exact commit is ready.
Multi-GPU execution additionally needs device-scoped roots, peer route admission and transport.
The existing topology/cost and empirical route contracts should consume those facts; synthetic
links, NVLink branding or a forecast never grant ownership, numerical permission or measured overlap.

### Loop import and vectorization consolidation (October 19 REPL exploration)

Explicit `par/` combinators remain the primary parallel-intent API. Ordinary Clojure loops
remain a best-effort import into the same typed operations and ordered control regions, not
a second scheduler or a promise to infer arbitrary parallel algorithms. Recognition, legality,
target representability and profitability must be reported separately.

The bounded source REPL exploration found:

| Surface case | Observed boundary | Remaining obligation |
|---|---|---|
| Explicit map / dense `dotimes` map | Both reach TypedSOAC; raw map reports one recovery | Numerical/device controls and declared alias contracts remain separate. |
| Explicit reduction / recovered sum loop | Explicit form imports; recovered form is recognized but its unbound terminal operation does not import | A flat terminal-result binding admits it in the REPL prototype; close this canonicalization handoff without changing evaluation context/order. |
| Explicit contraction | One generated kernel through public equation-first compilation | Compare schedules/performance under matched numerical permissions. |
| Outer `par/map-void!` with ordered inner reduction | One generated kernel | Do not confuse outer parallelism with cooperative reduction-axis scheduling. |
| Raw nested row loop | Imports after the flat terminal-binding prototype | Keep the recovered inner fold ordered unless separately authorized. |
| Data-dependent search inside a map | Semantic import succeeds; KernelBody lowering refuses | Import success is not end-to-end language coverage. |
| Dependent stores | Terminal-binding prototype imports an effect-map with `:iteration-order :sequential` | Never upgrade it solely because the surrounding source is counted. |

These observations used a synthetic target and small JVM numerical controls; they are not GPU
execution, bounds-completeness or performance receipts. Shared array snapshot discharge now
composes inline literal projection through the existing saved-product owner, followed by
sequential-only let normalization; it does not introduce another product lowerer or enable
broad copy propagation. A cold-loaded focused gate passes 161 tests / 924 assertions, and the
new public array-product integration check passes two assertions. Controls include eager sibling
evaluation/failure, reference identity, nullable arrays, contradictory carriers and preservation
of evaluation obligations across nested wrappers. Mixed-precision reduction completion and
actual device qualification remain separate obligations.

Independent read-only review found no confirmed defect in this shared-storage change; it
explicitly did not qualify emitted-device execution. The retained mixed-scale scalar-gradient
initially refused with `:equation-first-coverage` at a preserved constant seed after parallel
work. The next source-completion increment separates retention from ordering: marked successful
literal conversions remain executable roots, but their source-proved totality permits staging.
Canonical IEEE constant evidence is distinct from checked Clojure source evidence; Float
overflow cannot acquire infinity-on-overflow semantics through this proof. Both constant proofs
abstain at a bounded cast depth. Optional purity abstention retains the original independent
source obligations; incomplete obligation discovery produces a non-waivable unknown-obligation
denial. Review caught the initially conflated retention/order predicate and abstention interaction
before landing; both now have controls.

Focused gates pass 12 tests / 118 assertions for numeric/source staging and 163 tests / 940
assertions for evaluation/native behavior; array admission passes 1 test / 2 assertions. The full
affected frontend/storage namespaces report 145 tests / 794 passing assertions / one error. The
remaining mixed-gradient control now reaches the complete-source-return check and refuses
`:structured-control-source-result`, rather than declining the constant seed. Its conditional
Object-carried scalar completion and checked Float projection remain open. Do not substitute
payload metadata for source conversion or evaluation-sequencing proof. No check was waived,
no source precision contract changed, and no device-performance claim follows.

Downstream qualification of the downloaded 0.2.1690 jar now reports 49 tests / 757 assertions
and both fresh inline/packed-lagged marketing workers passing, including changed-data AOT and
MM allocation finite differences. This is workflow correctness evidence, not throughput evidence.
The separate preserved helper lifecycle still fails when a retained runtime gradient is reused
after AOT preparation: template revision stays 238 while compiler revision advances 1560→1564.
Keep that runtime→AOT→runtime sequence as a compatibility-debt acceptance case. Identify the
actual semantic/derived definition changes before changing invalidation granularity; never
reset or bypass the freshness guard to make the acceptance pass.

The read-only reference audit used local LLVM `585e2a01` and Valhalla JDK `775bcc5bdf2`.
LLVM LoopVectorize separates CFG/instruction/memory legality, runtime alias checks, VPlan and
costing; SLP is a separate packing optimization. C2 SuperWord likewise has dependency analysis,
speculative alias checks and profitability filtering. LLVM and C2 distinguish strict ordered
FP reductions from authorized reassociation; MLIR VectorToLLVM defaults reassociation off.
Do not rank these systems from unmatched numerical contracts or source inspection alone.

Priority within items 1–5: (1) close canonicalization/import handoffs with shared scope/context
machinery; (2) separate region numerical permissions from operator-level monoid facts;
(3) connect access/range, physical-view alias, ownership and completion evidence; (4) add narrow
guarded/costed vector plans only after differential legality controls. Existing raw-loop recovery
consults a global FP monoid permission; qualify and reconcile this with explicit-permission
semantics before widening it. The tested native scalar/requested-SIMD cancellation probe agreed,
so no changed numerical result was established by this exploration.

External acceptance may include a small synthesis/GA workload from Jack Rusher and, later,
Timestral-5M forecasting in pretrained-rstr. Timestral adds tests for mixed F32/F64 statistics,
masking/scans and robust existing math, not a new time-series compiler dialect. Its public
inference does not require AD; training qualification is separate. Existing naive `asinh`
cancellation is numerical-stdlib debt. These additions do not replace the original eight gates.

### Execution order

1. **Bound local compilation work (items 1–3).** Profile scalar-effect analysis, fusion candidate
   enumeration and final DCE separately on increasing, bounded components. Retain the candidate
   order, checked-conversion completion boundaries, effects and source provenance. Reuse existing
   analysis authorities; do not introduce an operation registry or persistent proof-skipping cache.
   Gate: repeatable cold/warm pass measurements, unchanged admitted/refused cases and numerical
   controls, followed by a bounded full-model compile rather than another blind timeout.
2. **Close resident training composition (items 3 and 5).** Use the existing EmittedParallelProgram,
   preparation plan, KernelExecutable and LinkPlan boundaries to keep intermediates resident and
   separate preparation from replay. Do not invent a GSDM session or linker. Gate: explicit
   parameter/state/data bindings, scalar results, alias/capacity/release counterexamples, CPU/device
   comparisons and reported kernel/allocation/transfer counts. Repeated calls must not rebind and
   restage every leaf. Resident execution alone does not close AD or performance acceptance.
3. **Continue normalized AD consolidation (item 5, with item 3).** Follow `normalized-ad.md`: checked
   source/primal/residual/linear/transpose programs, common equation composition, then the existing
   map/gather/reduce/contraction rules and ordered recurrence contracts. Keep source AD as the
   default structural reference until each migrated family passes its gates; shared derivative
   rules do not make it an independent calculus oracle. Gate: analytic/finite
   differences, JVP–VJP pairing, `:wrt`, mixed storage, zero domains, overwrite/tape lifetime and
   higher-order refusal/correctness checks. Arbitrary loops need not be accepted to finish the
   supported-language campaign, but must never silently differentiate to zero.
   Residual replay must identify the correct mutation version: ownership alone cannot authorize
   recomputation or remove an overwrite refusal. Retire a source rule only after common validators
   own equivalent obligations, including cold-load/cache-invalidation and independent refusal tests.
4. **Close local workload acceptance and measure (items 1–5).** Re-run unchanged real Gemma and
   generated SFT gates in a coherent compiler; add bounded GSDM trajectory/statistical checks;
   validate density composition, city kernels and a PDE rollout. Measure compile, preparation,
   host submission, transfer and device execution separately. Compare matched dtypes, numerical
   modes and residency against the relevant external baselines outside the hot loop. Only then
   tune schedules or expand precision policies. No peak/SOTA claim follows from emitted-source
   correctness alone.
5. **Consolidate the reached vertical (items 1–4).** Audit source-to-typed-to-schedule-to-body-to-ABI
   coverage; remove only the duplicate paths and tests whose independent obligations are covered.
   Update dependent-package compatibility notes and verify public OpenCL/CUDA/HIP compilation.
   Keep the external CUDA contributor informed of shared binder/ABI/ownership changes; GPU runtime
   parity and performance require its device measurements, not assumptions from this Intel GPU.
6. **Reconcile and finish distributed obligations (items 6–8).** First refresh the existing simulator
   and memory/route certificates, then public data-parallel/halo execution, then durable restart and
   the multilevel/AMR numerical vertical. Reuse the same value/version/access/completion facts.
   Gate each stage with hardware-free negative controls plus available local execution; explicitly
   defer real-fabric throughput, congestion calibration and larger physical deployments.

Memory planning is interleaved wherever residency, residuals, views or transfers expose a missing
obligation: observe, connect witnesses, shadow-plan, enable one proved reuse, then retire the old
guess. Distributed cost models may refine a legal plan, never its semantics or ownership proof.
Before enabling reuse, retain initialized-read, escaped-output/AD-tape and in-flight-transfer
obligations, as well as source-visible `alength` and view-range semantics. A logical tensor shape
must not replace a physical capacity or release/completion witness.

### Rhythm and stopping rules

- Keep one primary implementation slice and at most one focused review at a pivotal boundary.
  Use the bounded warm REPL for affected tests; restart after genuinely stale signatures, not for
  every test. Send the full suites to CI and fix actual failures before landing.
- Record each slice's input/output contract, admitted and refused cases, numerical oracle,
  physical obligations, compile/preparation/runtime measurements and removal target. If a slice
  cannot name those, narrow it before adding code.
- Squash only the reviewed exact head after all seven checks exist and pass. Continue independent
  work while CI runs; do not accumulate unreviewed PRs or confuse a merge with a release.
- Stop for user review only for material surface/numerical semantics changes, weaker acceptance,
  a new major dependency/target commitment or external resources/authority. Report such choices
  explicitly. Neither timeouts nor missing hardware justify bypassing a verifier.
- Completion requires fresh evidence for every original campaign gate and a documented remaining
  frontier. Frontier-scale training, PTX-specific peak tuning, FPGA support, learned planners and
  production data-center deployment remain north-star extensions, not implied claims of this
  laptop campaign.

### Bounded compile-cost increment

Auxiliary source-witness matching now filters by necessary binding identity before exact
semantic encoding. It still compares every same-binding source pair, requires exactly one,
validates the equation's complete witness set, and independently reconstructs allocation/alias
authority. Same-name binders with changed semantic metadata reach the canonical comparison;
ordinary identity equality is not evidence. Matching unsupported expressions still fail loudly.
Unrelated unsupported expressions are no longer encoded by this binding-scoped predicate;
whole-source admission remains a separate obligation. No cache or stored index is introduced.

A retained real GSDM training program has 539 source pairs and 66 auxiliary witnesses. Checking
three witness bindings preserved all three acceptance results while reducing canonical comparisons
from 55,002 to 11 (2.23 seconds to 1.19 milliseconds in that probe). A warm, loaded-machine
before/after small train-step compile measured 109.3 and 60.5 seconds, both with 316 semantic
equations and 260 emitted kernels. The second profile measured promotion at 1.06 seconds;
SOAC processing and emission still took 22.5 and 19.9 seconds. These single paired observations
are diagnostic, not a stable throughput or general speedup guarantee. Increasing independent-map
and map-chain probes remain separate from real-model compilation and device timing.

Vertical candidate enumeration now rejects extent/dataflow mismatches before forcing its existing
producer exceptional-conversion proof. Eligible pair order and all effect, completion, capture and
placement guards remain. Independent maps with zero vertical candidates now force zero proofs
instead of 7/15/31 for 8/16/32 maps; the matching-chain and trapping controls remain unchanged.
This is an analysis-call reduction, not evidence that the full resident compile now finishes.

DCE reuses only per-binding syntactic free-symbol sets inside its immutable input let. Mutation and
effect queries retain their original phases and order: effect analysis may initialize descriptors,
and descriptor effects do not all share the intrinsic revision counter. A controlled initialization
test therefore requires subsequent mutation queries to remain fresh. Differential alias, dead-local
buffer and mixed-effect forms match the previous pass. Focused fusion/placement/matrix-input/
epilogue/DCE suites pass 83 tests / 484 assertions in the warm REPL; cold CI remains separate.

Those differential controls also expose a pre-existing correctness gap: a checked cast or I/O
followed by a dead-local store is removed because known mutation targets suppress independent
observable effects. With `x = 2147483648`, a source cast-plus-store throws ArithmeticException while
the lowered form returns zero. Existing effect evidence retains `:checked-source-cast` or `:io`.
Repair this through that authority before claiming resident training acceptance; the cost increment
does not fix it or justify weaker exception/effect semantics.

### Observable effects are independent of local mutation targets

DCE's unconditional-root predicate now consults the existing effect descriptor before applying
purity conventions. Checked flags survive even in allocating expressions. Known-pure callees may
still evaluate effectful arguments, so the shortcut requires independently pure, unflagged shared
call arguments (including retained `.invk` transport). Nonlocal or unknown effects remain roots
regardless of a dead local write; unflagged local mutations remain governed by target liveness.
Mutation queries still follow descriptor analysis, preserving initialization-sensitive behavior.

This repairs observable behavior, not a new numerical policy: checked cast-plus-dead-store and
checked allocation arguments retain ArithmeticException; I/O survives plain, conditional and loop
regions and allocation arguments. Caller-owned aliases and verified local dead writes keep their
existing handling. Foreign GEMM calls whose current summary is `:io` now remain conservatively
live; this can retain previously eliminated work until a shared local-only effect contract exists.
No name-based exemption or numerical acceptance waiver is introduced. General exception/parallel
effect completeness and retirement of existing locally-pure conventions remain separate gates.

Cold CI exposed a materialization bug once DCE correctly retained a primal reduction: horizontal
fusion kept its earlier logical execution boundary, but source reconstruction placed it at the
latest constituent ID, after an intervening reader. Fusion ancestry is not execution placement.
Merged provenance now retains only the surviving equation's placement keys; a transferred result
explicitly adopts its consumer's boundary. Source reconstruction uses that existing source point,
with the historical ancestry fallback only for facts lacking a retained point. Independent JVM
comparisons cover intervening readers and a preceding vertical fusion with a later scalar
dependency; mixed legacy/current provenance cannot inherit the removed parent's placement.
The public mixed-storage gradient compile and device parity fixtures remain required gates.

Physical destination availability also constrains horizontal placement. The left site remains
preferred. When later private destinations are not allocated yet, a right-site fusion is legal
only if every destination is available there, no intervening equation references the left results
or storage, and all delayed left stores are allocation-backed private writes. Caller-visible
stores cannot move after an allocation that may fail. Missing allocation/input origin evidence
declines motion. Logical equation order and retained source placement move together; allocation
evaluation is not hoisted. Independent materialized-source/JVM controls check both reader denial
and caller writes completed before NegativeArraySizeException. The existing three-layout SDPA
integration retains its fused schedule rather than introducing a workload-specific rule.

### Post-emission host cleanup has a conservative submission boundary

Final resident DCE now uses its existing root predicate to retain bindings whose heads are in the
existing GPU submission inventory. It does not infer purity from marker names or metadata and does
not grant admission: extraction still verifies arity, registered artifacts and ordered ABI. DCE
retains initializer/argument dependencies in source order, other observable effects and source
metadata. Dead numerical work must be eliminated while semantic evidence is available, not by
reanalyzing physical driver bodies or discarding an unproved external effect.

On one already captured 962-binding GSDM input, unrestricted final DCE exceeded 32 seconds after
39 root analyses; several physical submission analyses individually cost about five seconds.
Conservatively rooting 427 submissions completed the same bounded diagnostic in 1.54 seconds,
with 535 ordinary analyses and two dead host bindings removed. This is not a full-model compile,
training, resident-execution or throughput acceptance result. Tests deny malformed-marker admission
and independently check once-only argument effects, checked casts and metadata preservation.

The subsequent bounded full GSDM resident compile completed all numerical/backend passes in
279 seconds and explicitly refused `:scalar-let-reads-device-buffer` on a converted reduction
result. No resident execution or training acceptance follows from this receipt. SOAC fusion still
cost about 112 seconds and backend emission about 48 seconds on this loaded laptop.

The next composition slice uses the existing explicit `:compiler :equation-first` vertical,
EmittedParallelProgram, invocation preparation and LinkPlan semantic outputs—not a new exception
in legacy scalar-let extraction. Keep device-produced scalar conversions/continuations in typed
device computation and project scalar/tensor results through existing resident views. Aggregate
parameter/moment leaves must acquire existing donation, mutable-sharing, alias, initialization
and output-lease contracts; the current read-only aggregate guard must not simply be removed.
Small gates are reduction → typed scalar conversion/sqrt → tensor update with scalar/tensor
outputs; mixed-storage frozen/parameter/moment leaves; and repeated invocation retaining old
outputs. Device-dependent host control remains an explicit refusal. Only after these contracts
compose should another bounded full training program be attempted.

### Independent distributed math intent reconciliation

Distributed planning, readiness (including retained-output readiness), resident root/scratch
projection, simulation, certification and one-shot execution now thread an independent caller
`:scalar-math` request through the existing numerical/LinkPlan authorities. Retained artifacts,
plan attributes and ambient compiler proof scopes cannot authorize selected math. Default
certificates retain their existing record shape and structural identity; only a nondefault
canonical request is retained in their extension map and independently required by verification.
Runtime intent is attached to the existing executable extension map before its original-owner
seal is issued. No record class, allocation/session owner, output lease or replay protocol changes.
Current storage-contract checks still precede output overlap/readiness checks.

Runtime options now reject unknown keys instead of silently ignoring them, and malformed math
requests refuse before hardware/session acquisition. Tests retain default compatibility, exact
request/certificate correspondence, acquisition ordering and actual OpenCL/Level Zero five-trip
selected-math execution as distinct obligations. This reconciles only the missing math seam of
held #1182; it does not replay historical ownership/output-watch changes or establish fabric
performance. The original eight-item compiler campaign and normalized-AD agenda remain open,
including exact generated SFT and the unchanged real Gemma gate (11/28 adapter failures).
Durable distributed capture/restore is deliberately not extended here: its producer-description
adapter still validates under default intent and refuses selected-math plans. Supporting that
composition requires a separate independent caller request, never consent inferred from the
runtime owner or its certificate. Existing default durable capture remains unchanged.

### Typed scan declines do not retry compatibility inference

Recognized inclusive/exclusive scans retain the complete typed singleton admission decision.
An intentional typed refusal now preserves its source, effects and exact structured diagnostic
without retrying compatibility SOAC inference; other parallel families' fallback behavior and
unexpected/invariant exception propagation are unchanged. The private fallback graph-construction
branch is retired. Public compatibility scan constructors remain for direct fixtures, while
admitted scans continue through the shared typed schedule, scalar interface and node validator.
This is a routing consolidation, not broader helper/algebra admission, new AD support or completion
of the unchanged failed real-model training gate. Native scan and independent source-prefix oracles
remain required.

Warm validation passes 30 focused boundary/source-packet tests / 570 assertions and
145 surrounding TypedSOAC, graph, emitter and actual-device tests / 1792 assertions.
Both local device backends were asserted available before the native controls. Partial
literal traversal retains padded backing storage and untouched suffixes; a concrete
element helper follows its independently obtained typed admission or exact refusal.
No test substitutes successful scheduling for an unsupported helper or recurrence.

### Original scalar closure for compatibility scan graphs

Compatibility scan graph construction now requires retained scalar dtype facts for dynamic
extents and scalar captures. It projects the exact selected operations' dependencies through
the existing SegOp operand authority and constructs ordinary GraphScalar declarations;
KernelGraph construction owns node dependency closure. Typed scan declarations retain their
existing order and dtype authority. Extent leaves must be integral, unknown/conflicting facts
refuse, and no emitted ABI or default `int` guess supplies original source proof.

The sole production legacy constructor caller forwards its existing scalar facts. Direct
constructor fixtures provide their established exact scalar interfaces; static no-capture
scans need no invented scalars. Scan scheduling now requires the original table and always
checks scheduled-body/node correspondence, retiring the legacy width fallback and validation
bypass. Manually assembled scan graphs without that table explicitly refuse. Public numerical
source forms and inclusive/exclusive arithmetic are unchanged; dynamic constructor calls now
require `:scalar-types`. This is source-interface closure, not new scan algebra, AD coverage,
GPU performance evidence or completion of the unchanged real-model training gate.

Warm validation passes 67 focused tests / 564 assertions and 38 affected graph/runtime
tests / 302 assertions. Explicitly available OpenCL and Level Zero execute 8 native
scan/replay/view tests / 40 assertions; source packet tests and their native controls
pass 10 tests / 681 assertions. Scan certificate serialization adds 9 passing assertions.
Unsupported retained semantic tags refuse rather than disappearing during dtype projection.
The remaining typed-decline-to-compatibility retry is still reachable and is a separate
retirement step; these checks do not establish general scan AD or training acceptance.

### Map store-callee derivative authority

An active map with a lexical nonlinear store callable named `double` kept its
correct primal but differentiated only the pre-store scalar body in reverse AD.
Source walking now retains the actual store-callee identity for map!/pmap using
the same scope-aware authority as scan. Both AD modes admit no store callable
or an established floating core conversion; active lexical, foreign and unknown
store callables refuse with `:ad-map-store-callee`. Inactive and reverse
`:wrt`-excluded maps retain their original primal effects. Runtime zero extent
does not establish construction-time callee authority. Existing Float conversion
semantics are unchanged; this is not general closure differentiation or a new
rounding policy. The original real-model acceptance gate remains failed.

A separate preexisting mixed-storage defect was exposed by the Float-store
positive: a Double reduction pullback accumulated a Double contribution into a
Float shadow array using generic core `aset`, which failed at invocation. Numeric
adjoint scatter in dotimes, additive/general reduction and scan now calls the
existing typed `raster.arrays/aset`
authority, whose Float-array/Double overload performs the established storage
narrowing. Accumulation, index replay and store order are unchanged; object-array
pullback tape stores are untouched. Mixed Float storage remains a mandatory
numerical control, not a narrowed or waived fixture.

Generated numeric map writes use the same typed setter, and backward lane/gather
maps derive their explicit destination conversion from retained array dtype via
the canonical dtype projections. Missing type evidence still refuses. Reduction
initializer hoisting retains existing operand metadata on its temporary instead
of dropping the carry proof; bound-before-initializer evaluation order is unchanged.
Focused controls pass 8 tests / 159 assertions, and affected admission, lexical,
soundness and ordered-normalization suites pass 61 tests / 462 assertions in the
existing warm REPL. These checks do not establish new Float-scan support or close
the unchanged real-model acceptance gate.

### Qualified reduction algebra identity

`typed-reduce-identity` now queries the exact declared operator rather than
discarding its namespace. Previously an unregistered qualified `+` inherited
the builtin zero identity, contradicting the shared algebra authority. Explicit
custom monoid identities remain authoritative, including custom operators named
`min`. Only a nil abstract identity consults the existing canonical intrinsic
authority for builtin min/max dtype extrema; a custom nil-identity operator
without that intrinsic declines. No new alias registry or source arithmetic
policy is introduced. Focused descriptor, scan, SOAC, product-consumer and
integral-reduction tests pass 46 tests / 506 assertions in the warm REPL.
This does not repair the separate mangled-descriptor basename compatibility
path, certify all reductions, or resolve the failed real-model training gate.

### Active parallel-effect AD admission

A public scalar objective that filled a fresh array with `x*x` using
`par/map-void!` and reduced that array returned the correct primal but zero
derivatives in both reverse AD and JVP. Scoped activity saw the capture, but
ordinary-call fallback inspected only the top-level call arguments and dropped
the captured dependency. Both modes now refuse an active parallel region that
reaches fallback without their own scoped derivative handler, through the
shared form/activity contract and `:ad-unsupported-active-parallel-region`.
Inactive and reverse `:wrt`-excluded effects retain primal execution; supported
map, reduction and scan handlers remain admitted. This is an intentional
surface behavior correction from a wrong derivative to explicit refusal.

Focused and surrounding AD suites pass 96 tests / 737 assertions in the warm
REPL, including an independent polynomial/finite-difference counterexample.
Complete differentiation of fresh-buffer effect maps remains open: canonical
single-assignment maps and ordered buffer versions must enter the existing
functional AD rules with their ownership, replay and precision obligations.
Enabling helper inlining alone is not sufficient; the unchanged SFT objective
is retained as the generated-AD acceptance target, separate from its existing
handwritten-backward and native replay tests. No fixture-specific derivative
or weaker real-model training oracle is introduced.

### October 10 non-executing call-signature evidence

The compiler's registered-method selection now exposes a non-executing tag-tuple
query. It shares the same implementation-Var/result-tag and promotion projection
as ordinary devirtualization, but never synthesizes arguments, invokes a helper,
or registers a parametric specialization. The existing parametric tag unifier
also exposes applicability separately from its declared result: a matching
signature with no result annotation is not the same as no matching signature.
The older result-only query retains its existing declared-result fallback.

Focused controls distinguish asymmetric `[Dual Double]` from unsupported
`[Double Dual]`/`[Dual Dual]`, preserve promotion casts, and recover the ordinary
Double result of `real-value` on a Dual. Queries leave registry identity and
helper execution counts unchanged. New controls pass 5 tests / 21 assertions;
existing dispatch/inference tests pass 29 tests / 133 assertions in the warm REPL.

This is shared evidence infrastructure, not a repaired forward-AD admission
claim. The lexical AD preparation must next propagate actual carrier result
facts and check every reachable argument tuple, rather than treating all seed
dependence as Dual or any Dual overload as family-wide coverage. Unknown results,
branch/carry alternatives, parametric cold/warm behavior and unsupported carrier
promotions remain explicit obligations. No surface numerical semantics, emitter
default, model pin or real-weight acceptance oracle changes here.

### October 9 physical compute lanes follow explicit worker placement

A reproduced simulator gap gave independent compute lanes to logical workers
explicitly placed on the same physical target. Compute steps and transfer
`:serialized-on` claims now share one nonrecursive worker-to-target projection.
Claims resolving to the same lane are deduplicated. Unplaced workers and
topology-only relay identities retain their original lanes; distinct physical
targets remain independent. Route-resource certificate witnesses use the same
projection and reject placement changes even when makespan is unchanged.

Logical compute totals, transfer bytes and per-worker peak-memory reports remain
unchanged. `:physical-compute-ns` additionally reports accumulated compute service
by physical target, not transfer-inclusive busy time. Physical memory-capacity
aggregation, link-sharing discovery and actual asynchronous distributed execution
remain separate obligations. This changes planning resource accounting, not the
surface numerical program, kernel emission or source arithmetic.

### October 9 explicit empirical route-cost simulation

The existing simulator now accepts optional live route context, supplied profile
reports and an explicit admission policy. Matching includes physical endpoint
placement, device/hardware signatures, directed route facts, bytes, realized
layout (not allocation identity), transport and staging budget. Evidence must
have consistent before/after snapshots, fresh endpoint sessions, finite positive
whole-step durations and bounded Unix-epoch age. The shared Measurement summary
supplies CV/spread and median costs. Insufficient, inconsistent, stale, repeated
or noisy evidence leaves an explicit per-route analytical fallback.
The policy's `:cold-warm` label is caller-declared summary metadata; these
observations do not independently establish sample warmness.

This is a pure diagnostic projection, not a new compiler cache or authority:
the original one-argument simulator and all plan certificates remain analytical.
Supplied reports are not authenticated, synchronous timings include observation
overhead, and admitted durations neither prove overlap nor model congestion.
Native fresh-owner OpenCL/Level Zero controls compare projected durations with
the actual captured samples using a deliberately permissive spread policy for
nonflaky plumbing coverage, not a production calibration claim. Production
stationarity, load-regime validation and overlap-conditioned route costs remain
open, along with the unchanged real-model training acceptance.

Focused simulator, compute-binding, AMR-plan and profile regression suites pass
80 tests / 625 assertions. The native route-admission control passes on both
local backends (1 test / 34 assertions); it checks projection of observed sample
medians, not held-out predictive accuracy. Reports can be supplied to
`simulate` with `:route-context` obtained from `raster.gpu.distributed/cost-context`
on a fresh ready owner and an explicit `:route-policy` containing `:now-ms` and
`:cold-warm`. Existing certificates never inherit these simulation options.

### October 9 distributed profiling hardware-evidence context

Distributed profiling now retains the existing HardwareDescriptor evidence
signature for each physical session, alongside live backend device/driver and
session identity. Queries use the session device, not a logical worker label.
Before/after calibration-version or admitted bandwidth drift invalidates the
observation; normal unprofiled execution is unchanged. Focused hardware-free
and real OpenCL/Level Zero context tests cover this boundary. These signatures
are necessary context for measured route-cost admission, not measurements of
the fabric, automatic topology discovery or an adaptive cost overlay. Exact
route/byte/transport/staging matching and noisy/stale sample admission remain
the next distributed calibration slice; analytical simulation remains a seed.

### October 9 shared physical service lanes

Directed links may now explicitly declare `:attributes :serialization-domains`,
a vector of distinct keyword identities. A transfer occupies its route links,
the union of their shared domains, and any existing endpoint `:serialized-on`
compute claims. One resource-occupancy update drives all these lanes; byte
accounting remains per route link and logical transfer, not per shared domain.
No domains means the existing independent-link behavior. This is conservative
exclusive serialization for declared physical sharing, not automatic discovery,
bandwidth sharing, switch congestion, or a calibrated fabric prediction.

Certificates retain each transfer's derived resource claims as well as costs:
renaming a shared domain invalidates an old witness even when numeric makespan
is unchanged. Warm-REPL distributed-plan, compute-binding and AMR-plan suites
pass 67 tests / 424 assertions, covering opposite-direction routes, independent
domains, empty/default declarations, malformed declarations and multi-hop claim
deduplication. Measured route service curves and production distributed execution
remain open alongside the original training and adaptive AMR obligations.

### October 9 physical topology admission

Reconstruction of a topology now reruns the existing device/link constructors,
not just their record-type and endpoint checks. Before this fix a modified link
with NaN bandwidth simulated and freshly certified with a 700 ns makespan in
the data-parallel fixture. Nonfinite/nonpositive bandwidth, invalid latency,
memory capacity, descriptor and attributes now fail through the existing field
contracts. Modified topology indexes must also match their contained record
identities and retain actual device/link maps. Constructor, simulation, fresh
certification and verification regression cases cover these boundaries. Pre-plan
route-cost queries also validate each visited link and its index identity, without
rescanning the entire topology for each transfer.

The affected hardware-free plan, compute-binding and AMR-plan suites pass
65 tests / 405 assertions. No valid constructor API or numerical policy changes;
this rejects malformed physical claims, not measured-but-inaccurate costs.
Explicit shared-link serialization domains, calibrated fabric costs, the held
training gate and the remaining distributed/AMR workload obligations remain open.

### October 9 distributed capture and physical-model reconciliation

PR #1136 is merged as `4fb7ec5b` and #1137 as `9f39cd6e`, each after independent
exact-head review and all seven registered CI gates. Original distributed owners
now retain physical allocation budgets and owner/session-bound measured storage
facts. Completed bounded AMR cycles capture both whole fields through the existing
manifest/provider contracts in a pinned read scope; every download and the final
provider callback boundary revalidate the live representation snapshot. Coordinates
are deliberately closed to nonnegative synchronized step advancement, not arbitrary
caller time labels. Final capture controls pass 5 tests / 40 assertions; the actual
OpenCL/Level Zero capture/close/decode/fresh-context oracle passes 1 test / 14
assertions without capability skips.

PR #1138's independently reviewed rebased head is
`843e04aab393a6b7e458aa1ff69a530f3fc242cc`; its fresh CI remains pending.
Source-compiler/target-semantic compatibility is checked before opening content,
then bounded fixture files are independently verified through read-only mmap leases
and decoded before fresh-context continuation. Pure and native tests pass 3 tests /
25 assertions without skips. The provider and explicit file materialization remain
test fixtures: this does not prove production durability, metadata atomicity or
durable parent existence. Use the existing availability finalizer before an
application-owned publication transaction.

The distributed physical model is not wholly overlap-blind: declared backend
transfer facts distinguish submission, physical queue independence/serialization,
event completion, staging/host lease requirements and peer mechanisms. An explicit
transfer `:serialized-on` claim occupies the relevant compute lane in simulation.
Directed-link bandwidth/latency remain analytical inputs, not discovered/measured
fabric facts. Shared NIC/PCIe/switch contention domains, message-size curves,
load-conditioned calibration and actual asynchronous distributed execution remain
open. Never infer physical overlap merely from asynchronous API shape.

Device-level calibration already feeds the single HardwareDescriptor and its
evidence signature; it is not yet distributed route calibration. The CPU
microbench retains nonstationary bandwidth as `:measured-noisy` for inspection.
Since #1139, those observations do not replace planning estimates: the existing
descriptor authority admits only positive finite fields tagged `:measured`,
rejects explicitly nonstationary bandwidth, and requires complete, alias-consistent
peak-FLOPS families. This admission does not establish an expiry policy or immunity
to systematic bias. Power/load regime, calibration validity and fabric contention
still need explicit evidence before treating estimates as reliable production-selection
facts. Reuse existing Link device-event profiling when
instrumenting distributed execution, rather than measuring repeated mutations of
a completed one-shot owner or inventing a second timing/cache convention.

The original campaign remains intact. The next compiler/training acceptance work
must retain the held real-weight oracle and shared typed realization contracts;
these successful scientific state boundaries neither close that gate nor establish
vendor-device performance. Local sibling dependencies are still old (`pretrained-rstr`
0.2.545, `finetune-rstr` 0.2.287), and the latter still calls retired
`bind-program!`; committed isolated acceptance sources, local checkout state and
published consumer migration must remain separate evidence.

### October 9 verified nonlinear boundary checkpoint

The shared elementary-math realization contract covers 23 C-family library operations while
preserving source precision. A captured real-model layer-0 diagnostic now verifies its taps
against the unchanged JVM pullback bit-for-bit. Supplying identical tanh values makes CPU/GPU
pullback arithmetic identical; target Float tanh differences reproduce strong derivative
cancellation sensitivity on both systems. This is local boundary evidence, not closure of the
held real-weight gate. See `local-compiler-evidence.md` for controls and the three-coordinate
distinction from the original GPU kernel. All eight campaign items retain their original
workload acceptance obligations; continue the general compiler, training and distributed
verticals without replacing them with isolated numerical regressions.

### October 9 typed arithmetic consolidation checkpoint

#1098–#1100 are merged: C-family scalar product boundaries are protected separately
from explicit FMA, transposed-left products use the shared register body, and JVM
helper extraction retains typed calls and their operand conversions. #1101 extends
that preservation to scalar simplification and partial evaluation; its follow-up
checks retained operation precision during JVM scheduled SegMap admission with an
active species. An executed mixed-width
surface map exposed different rounding in vector lanes and the scalar tail; the
correct scalar schedule is retained until mixed-species vector lowering exists.
These are shared semantic fixes, not replacement model kernels or a new type registry.
The follow-up threads active-species precision obligations into stencil and C-SIMD
consumers and checks complete reduction recurrences. C integer-widening syntax
admission remains emitter-specific; canonical conversion precision is checked before
source projection. A further reproduced Boolean comparison miscompile requires
retaining selected interface parameter tags through SIMD canonicalization. Declared
comparison domains are checked before compare-and-blend; missing signatures decline
rather than being inferred. Untyped compatibility and missing operation stamps still
need separate evidence, and mixed-species vector execution remains unimplemented.

The C-SIMD guard also exposes a quant-fold performance residual: Q4 x8 explicitly
computes Double scale products into Float storage, while its old vector assertion
required an all-Float implementation. Bit-sensitive changed-input JVM/native tests
now retain the source precision. The integer-dot override and homogeneous Float
widening tests remain; mixed-precision load/compute/store vector schedules are still
needed to recover the fold optimization without narrowing arithmetic.

The fresh unchanged real-weight two-layer training gate after #1100 passes loss and
input-gradient checks but still fails 12 of 28 adapter-gradient checks.
After the subsequent retained-precision fixes, a cold unchanged run fails eleven
of 28 adapter checks; a public strict-FP32 matrix-policy diagnostic produces the
identical metrics. This rules out that policy switch as a remedy for this case.
Acceptance therefore remains open; neither these narrow regressions nor synthetic
training acceptance closes campaign item 5. Continue precision-boundary consolidation
and identical-input matrix/VJP localization without loosening the pinned oracle.
The October 9 diagnostic below preserves that gate: an independent CPU model
using the default GPU's decomposed Float dot order reproduces the same eleven
failing adapter keys and the GPU loss exactly, but four adapter comparisons
still exceed the original coordinate threshold against that counterfactual.
Whole-tensor relative L2 errors against the unchanged native reference are much
smaller (at most 2.45e-5); this is evidence of sensitivity, not acceptance or
permission to relax the original oracle. See `local-compiler-evidence.md`.
CUDA/HIP runtime performance and multi-host execution likewise remain separate from
hardware-free vendor compilation and local logical-worker acceptance.

### October 8 collective execution and precision-boundary follow-up

The storage boundary fix is merged as #1095 after all seven required CI gates passed.
The next isolated checkpoint diagnostic uses identical layer-0 Q/K/V and a synthetic cotangent,
not a full-model replay. Raw causal scores and DW dots match the JVM exactly on local OpenCL;
the D reduction matches exactly for both CPU and GPU supplied intermediates. Same-input softmax
weights differ by at most one Float ULP. Selectively spelling exponential evaluation as
`float(Math.exp(double(delta)))` restores exact agreement for this sixteen-weight diagnostic
without changing Float max/sum carries, storage or the reference. This does not close the held
adapter-gradient gate or prove universal libm bitwise equality.

Native `:exp` realization is now stated separately from evaluation-order evidence in KernelBody.
Its target-library accuracy is implementation-defined, not a correctly-rounded or bounded-error
claim. Float calls remain Float; explicit Double calls followed by Float conversion retain that
typed sequence. Other transcendental realization contracts and a user-selected reproducible
algorithm remain follow-ups; do not globally widen math or introduce an attention-specific emitter.

The contribution refinement, ordinary generated-combine bindings and complete realization
certificate are merged (#1090–1094), with all required CI gates green. Two and three logical
workers execute complete all-reduces on the local OpenCL and Level Zero devices. Fractional
products and FP32/FP64 cancellation are checked against independently rounded evaluation trees;
optional FP64 skips retain all FP32 cases. This closes the local numerical collective seam,
not multi-host transport, distributed transformer training or performance acceptance.

Training diagnosis exposed a separate declared-storage inconsistency: the preservation option
reached parameter typing but was dropped before the TypedSOAC frontend, and resolved local
polymorphic allocations still followed ambient precision. Retaining their array tags also exposes
two existing restrictions: fold-map results assumed a single ambient storage dtype, and a canonical
scalar Fold required its consumer precision to equal its carry precision. The follow-up forwards
one existing policy, derives each destination's storage dtype and converts only completed Fold
results through the shared scalar conversion authority. Default allocation policy is unchanged;
the explicitly requested preservation option now applies to resolved locals as well as parameters.
Host fold-map projection spells the same per-destination completed-result conversions explicitly,
including mixed Float/Double outputs. Its executed oracle compares the projected form with typed
source execution; the shared mixed-storage fixture also enters CUDA/HIP compile gates. No fold
carry is narrowed merely because its consumer uses narrower storage.

An isolated real-checkpoint attention probe executes Double computation with Float storage after
these changes, with small nonzero differences from the existing CPU reference. The global `:dtype`
still selects generic source overloads; this probe fixes the Float source overload explicitly.
It is not a new independent precision-selection API and does not close the held real-model gate.
The next acceptance work must distinguish source specialization, intermediate arithmetic and
storage, without rewriting the pinned model or loosening its original gradient checks.

### October 8 native reference-environment admission

`blas/library-info` and `lapack/library-info` now report retained selection facts,
with no numerical smoke test or thread reconfiguration. Discovery retains its loader
input path and optional OpenBLAS configuration once; public diagnostics omit the foreign
lookup. Fortran LAPACK and LAPACKE report separately, including absent/rejected/error
outcomes, rather than inferring QR capability from a Fortran smoke test. Selected means
selected, not functional or fast. The requested loader path is not a binary hash, and
dependency-exported OpenBLAS metadata only advertises that provider's ABI, not a wrapper's
certified interface. MKL reports the LP64 interface filename as declared, not as an
OpenBLAS build fact. No second compiler cache or numerical descriptor registry is added.

Local retained evidence is threaded MKL for BLAS and OpenBLAS 0.3.30 (including
`NO_LAPACKE`) for Fortran LAPACK; the separately loaded LAPACKE wrapper resolves the
same dependency metadata. This differs from the external OpenBLAS 0.3.32 environment
and is not a preferred-version claim or crash diagnosis. Integer overflow probes confirm
existing core `int`, LAPACK `int-seg` and BLAS GEMM reject `INT_MAX + 1`; a hardware-free
GEMM-handle regression preserves that contract instead of adding a redundant conversion layer.

Explicit selection now shares conventional paths and one resolver across BLAS/LAPACK.
`raster.openblas.path` is read at first delayed selection rather than namespace load;
the captured pin bypasses preloaded/MKL/default paths. A missing file, required symbol,
or OpenBLAS build metadata rejects the pin instead of silently changing providers.
This intentionally tightens LAPACK's former preferred-path semantics and makes BLAS
honor the same setting. Pinned QR requires LAPACKE symbols in that selected library;
it does not silently bind a separate LAPACKE library with unknown provider linkage.
Without a pin, existing discovery order and metadata-absent compatibility remain.
The pin is not a guarantee against dynamic-loader symbol interposition. Provider/status
diagnostics and argument-range checks remain separate native admission work.

An external CUDA developer reports 4,602 tests / 57,807 assertions at `f7cbf12`,
with 16 failures and 45 errors. Their initial categories are missing BLAS/LAPACK
(37 checks), exact floating-point differences (10), unsupported portable trapping
arithmetic (11), attention dispatch (two), and a missing Clang case subsequently
resolved. These are reported baseline evidence, not reproduced diagnoses or a count
of independent root causes. A standalone C batched-GEMM crash with OpenBLAS 0.3.32
still requires the arguments, headers and linked-library integer ABI to be checked.
CUDA compiler fixtures and two changed-input device checks with clean memory-sanitizer
results are preliminary execution evidence, not completed runtime or performance gates.

Both Panama numerical libraries declare 32-bit integer slots. A shared optional
`openblas_get_config` probe now rejects advertised `USE64BITINT` libraries before
BLAS or LAPACK numerical downcalls. The rejection is intentional: changing integer
width requires different descriptors, not a tolerance adjustment. Unknown providers
without metadata retain existing admission and are not certified LP64 by this check.
Hardware-free native stubs test the pointer-returning metadata ABI, absent symbols,
null metadata and incompatible build flags. This does not explain the reported C crash.
Admission errors propagate through availability queries rather than being converted into
whole-namespace skips. Ordinary missing-library discovery retains the existing optional behavior.
The CI test job declares `RASTER_EXPECT_NATIVE_LIBRARIES=1`; an unconditional test
namespace then requires functional BLAS and LAPACK, without forcing either onto laptop
or compiler-only jobs. This preflight does not replace per-capability coverage accounting.

Actual binary identity/version and live thread metadata,
ABI treatment for unknown/preloaded providers/wrappers, and a pinned native reference environment
remain CI/compatibility debt. The shared pin resolves the former BLAS/LAPACK property
discrepancy; it does not provide a complete reproducibility manifest. Keep the remaining work
on items 1–2 without replacing the external training, distributed or PDE gates.

### October 8 resumed acceptance and AD review

The bounded JVM Float-seed correction (#1070) and position-sensitive helper capture (#1071)
are squash-merged after all seven registered CI gates passed. The unchanged real-checkpoint
gate after #1070 still declined: loss and input-gradient checks passed, but 12 of 28 adapter
checks exceeded the original componentwise threshold. Same-input raw gradients and a secondary
sequential-FP32 matrix diagnostic isolated an arithmetic-realization contribution without closing
the residual or replacing the pinned CPU oracle. The external training PR remains held.

Upstream #1072–1077 subsequently changed allocation defaults, buffer ownership, additive
reduction AD and gathered-read transposes. Recheck current main before carrying forward the old
numerical diagnosis. A code review flags two distinct replay obligations for focused reproduction:
additive replay must satisfy the existing purity proof rather than repeat effectful preludes;
storage replay/overwrite checks must follow aliases and include gathered index dependencies.
Symbolic aliases alone cannot prove distinct caller-provided arrays physically disjoint.
These correctness checks belong to the existing consolidation campaign, not a new compiler or
another AD rule registry. The distributed, multilevel/PDE and broader workload gates remain open.

The current-main checkpoint recheck reproduces the same failed numerical gate, without changing
the source pins or tolerances. Independent deterministic probes confirm the replay concerns:
an alias write changes a scalar gradient from 3 to 102, and an effectful additive prelude executes
twice (gradient 12 and scratch 6 instead of 3 and 3). Additive replay now reuses the ordered
route's existing purity proof; impure steps retain their closure tape and execute effects once.
This preserves support for effectful reductions rather than rejecting their additive shape.
Storage-alias and gathered-index replay protection remain the next distinct slice.

That slice derives direct storage representatives from ordinary aliases and each operation
record's result/output identity. The shared forward pass checks writes against those representatives;
gather, mapped and separately analyzed dotimes transposes declare their reread index dependencies.
Ordered closure reductions also declare scatter-index dependencies even though their primal read
values are taped. Explicit source writes inside the same operation are checked against its own
replay claims; normal record-declared output writes are not mistaken for such source effects.
Captured dotimes index values remain valid after source-index mutation, rather than inheriting
another representation's replay restriction. Focused source/AD oracles cover transitive inactive
aliases, map-result/output aliases, index mutation, captured indices and fresh-buffer writes.
This is not a physical no-alias proof for distinct caller arguments or a general may-alias
analysis of arbitrary conditional/opaque returned arrays. Keep those ownership obligations
explicit in the remaining consolidation work; no numerical gate or residual representation changes.

### October 6 contextual initializer precision follow-up

Real-weight boundary isolation exposed a remaining JVM/GPU arithmetic discrepancy in attention
scores: the GPU uses sequential FP32 products and sums, while the JVM's large-method helper
extraction retained Float products but accidentally accumulated in Double. Small unsplit methods
hid this defect because their typed Float calls rounded the accumulator on every iteration.
A four-map cancellation fixture reproduces the split-path failure independently of checkpoint
data: `[1e8, 1, -1e8]` incorrectly sums to one rather than zero.

The JVM loop emitter now realizes the retained contextual Float seed before forming its slot
type, only for a bare Double literal with both a Float binder stamp and Float recurrence width.
This adds no inference registry or attention rule. Explicit Double/nonliteral initializers and
genuinely wider recurrences remain unchanged. Canonical AD/SOAC forms are unchanged: an initial
attempt to insert casts in shared walking broke Double-storage AD islands and GQA proof
reconstruction in cold CI, so that broader change was withdrawn. Split/unsplit JVM tests, an
actual causal-weight cancellation oracle, and public OpenCL/Level Zero replay check the narrower
boundary; the affected AD and GQA cases are explicit non-regressions. Fresh checkpoint attention
weights match the independent sequential FP32 reference; full real-weight gradient acceptance
must be rerun and is not claimed by this correction. The original eight-item campaign stays open.

The split regression also exposed a separate lexical dependency defect: naming its length
parameter `count` can cause an extracted helper to read `clojure.core/count` instead. Keep that
follow-up distinct from initializer precision and cover core-shadowing parameters at helper
extraction; renaming a fixture to `cnt` isolates this slice, not a language restriction or fix.

The helper-capture follow-up uses the existing shared lexical free-symbol analysis with visible
parent locals declared, rather than allowing global core resolution to erase parameters. It also
prevents a numeric global constant from replacing a same-named local parameter. Visibility is
position-sensitive: each extracted `let` initializer sees only parameters and prior binders;
the body sees all completed bindings. True globals remain eligible for the existing constant
projection, and nested binders stay owned by the canonical scope analysis. Core-name/constant
shadowing, nested shadowing and early-global/later-local regressions cover this JVM boundary.
This does not introduce a new scanner, change surface signatures or close arbitrary-loop AD.

Current scalar AD consolidation: final parameter projection follows retained SSA alias bindings
to carried source types, keeping unknown/dynamically absent cotangents nil-safe. This removes the
cold projection-helper typedness gap without adding an intrinsic or adjoint registry. The
placement follow-up lets a direct Float/Double result conversion cross pure SSA equations and
replace only a same-dtype manifest identity transform. Effects, aliases, opaque host boundaries,
nonidentity transforms and general mixed-width arithmetic retain their existing barriers. The
mixed scalar-energy gradient now executes with exact JVM parity on both local backends, for
empty, small and two-phase reductions. This admitted reduction epilogue is not arbitrary scalar
consumer placement. Continue with external training acceptance and generic composition; do not
treat this slice as completion of the eight-item campaign.

Authorized scope is the eight-item campaign below. The original four-stage emitter sequence and
subsequent chronological notes remain the detailed history, not a reduction of that scope. Each
production migration must retain numerical, ABI, ownership and resource contracts; isolated emitter
coverage is not a completed vertical. The north star remains the architectural specification.

The current local-first landing order and completion gates are in
[compiler-consolidation.md](compiler-consolidation.md). The checkpoint below and chronological
notes retain the wider campaign and its implementation history.

## Current eight-item checkpoint — 2026-10-02

### October 4 public mixed-matrix admission slice

The public schedule and uniform Int/Long graph-carrier slices are now in review. The
hardware-free Gemma training compile is the next integration gate, before migrating its existing
mixed-precision device trajectory off the descriptor fixture. It exposed a reconstruction-context
gap: declared shape operands named like core functions (for example `seq`) disappeared from
standalone scalar-dependency analysis. The pure semantic constructor now derives its lexical
context from retained TypedSOAC values, just as production lowering does. Core-shadowing shape
regressions and the wider TypedSOAC route suite pass; exact source-law comparison is unchanged.

The same Gemma compile subsequently exposes an unresolved storage relation: the matrix axis
uses a retained scalar alias (`nqh`), while its output capacity expands to `nq * hd`. Do not assert
these equal from their spelling or relax capacity validation. The follow-up derives the dense
output-prefix minimum from the canonical reduction axes and enforces a checked allocation-capacity
guard at the independently reconstructed graph boundary. This proves storage safety without
asserting that independently supplied scalar values are equal. Unsupported physical layouts and
noncanonical folds remain outside this projection; declared capacities are not replaced.
Public mixed training acceptance, native numerical trajectories and performance remain open.

With the output-prefix guard, the hardware-free Gemma training compilation succeeds: 189
scheduled equations, 123 emitted tensor equations, 66 host scalar equations and 50 contraction
dispatches, with no candidate declines or legacy fallback. The other 73 emitted equations are
not single plain FP32 contractions; this is not evidence that every operation is matrix-tiled.
This measurement uses the synthetic Intel descriptor and establishes compilation only, not
native dispatch selection, training parity or throughput.

### October 5 invocation-local dispatch proof consolidation

The per-invocation boundary scan completes pure Gemma planning in 157 seconds after a
182-second fresh compile under variable background load: 123 tensor calls, all 50 mixed
`xmx-direct` selections (21 NT, 15 NN, 14 TN), and zero driver allocations. An inclusive
profile on the same compilation takes 136 seconds and records 905 boundary-equation queries
accounting for 68 seconds; nested timings are not additive and these are not matched benchmarks.

The next bounded consolidation extends the existing sealed program evidence, not the cache or
certificate hierarchy. One read-only operation-identity map retains the independently checked
fallback equation, common physical results and certified candidates. Every dispatch candidate,
numerical permission/model, common storage and exact fallback is checked during that proof.
Storage lowering and synchronous call construction reuse only those exact immutable facts.
Absent external evidence, lowering derives fresh invocation-local evidence itself. Copied or
modified owners/evidence reject; a missing scoped operation projection fails closed. Public
validation remains independent; runtime admission, scalar/range checks, result views, aliases,
host callbacks and final LinkPlan obligations remain fresh. No proof is attached to the plan,
and callbacks do not inherit the projection scope. Native Gemma trajectory acceptance and
the original external training/distributed/multilevel campaign gates remain open.

The first post-consolidation large check compiles in 118 seconds and lowers in 51 seconds,
again retaining 123 tensor calls and all 50 intended mixed selections without driver allocation.
An inclusive repeat takes 42 seconds: boundary queries fall from 905 to 290, KernelBody
validations from 70,122 to 20,883, and call construction from 32 seconds to 2 seconds. Timing
conditions are not controlled; call counts, unlike wall-time ratios, directly identify removed
duplicate work. Remaining boundary queries account for 18 seconds and justify auditing the
existing complete-write consumers next, before adding any further retained facts or changing
native/default schedules. No native trajectory or end-to-end training speedup is established.

The symbolic-write follow-up retains only already proved domains in that same operation report,
not a resolved complete-write flag. Invocation initialization and final LinkPlan effect analysis
resolve current scalar extents, capacities and views afresh. Ordinary SOAC/contractions keep
their prior semantic, alias-aware coverage path even when a report contains a stronger certified
contraction extent; sharing evidence does not promote another initialization-coverage family.
Protected SWR and dispatch admission retain their existing rules. Public LinkPlan validation
reconstructs independently, while retained validation checks the exact program seal before each
operation projection. Source-order, partial/prefix writes and structured loops are unchanged.
Fresh-versus-retained effect witnesses, varied capacities/extents, alias rejection and callback
isolation remain the oracles. Equal copied operations can be independently checked as equations,
but cannot substitute for a whole program's exact owned step; whole-call constructors should use
the owning program rather than manually replacing a step. No new cache or certificate is added.
Reusing the existing Gemma compilation, the large public plan lowers in 14.1 seconds and retains
123 tensor calls, all 50 mixed selections and zero driver allocations. This is a local diagnostic
observation under changing background/JIT conditions, not native or controlled performance data.

The existing mixed Gemma trajectory fixture is being migrated to public lowering and replay,
with donated adapters, shared constant weights and fixed-graph binding evidence. All 50 eligible
dispatch equations must select mixed execution; filtering only already-mixed selections would
hide fallback regressions. The initial pure public lowering exceeded a 590-second local check
budget; stack samples showed repeated semantic fingerprint/source-ABI validation and equation
reconstruction under `invocation-link/lower`'s buffer-by-equation boundary scans. After consolidating
those scans within one invocation, a fresh capped REPL compiled in 182 seconds and lowered in
157 seconds under varying background load. The resulting plan has 123 tensor calls and selects
all 50 mixed `:xmx-direct` dispatches (21 NT, 15 NN, 14 TN), with zero driver allocations.
This establishes public planning and selection on the synthetic Intel descriptor, not a matched
performance comparison or native training acceptance. Lowering remains too expensive; profile
the remaining repeated validation before extending the existing exact-owner evidence. Runtime
admission, independent public validation and native 25-step numerical acceptance remain required.

The subsequent native recheck passes the tiny public register-tiled replay on OpenCL and
Level Zero (six assertions), followed by a small Level Zero `:xmx-direct` matrix case with
relative L1 error `1.5e-8`. The migrated public mixed Gemma fixture then passes its complete
25-step trajectory: 35 assertions, zero failures/errors, all 50 mixed dispatch selections
(21 NT, 15 NN, 14 TN), zero planning allocations, and planned-versus-bound graph agreement.
FP32 loss decreases from 2.281172 to 1.261994; mixed loss ends at 1.262064. Every step remains
within the existing `5e-3` relative trajectory tolerance. This supersedes the unresolved local
native acceptance for this small fixture, not the historical queue errors or other workloads.
The capped reusable REPL completes the check in 248 seconds including compilation, planning,
binding, replay and host loss evaluation; this aggregate is not kernel timing or a performance
comparison. Full-model training and matched performance acceptance remain open. The fixture now
uses public equation-first lowering/replay rather than the resident descriptor lifecycle.

The next equation-first candidate constructor derives a full-K mixed schedule from the retained
typed algorithm and freshly reconstructed public graph. A pure planner owns target admission;
the existing matrix emitter consumes the resulting stage bodies. The shared target-schedule
owner checks descriptor/instruction agreement, divisible positive geometry and normalized
workgroup limits. The first row admits Intel DPAS with Intel OpenCL only, Int dimensions,
materialized dense storage or NN/NT tile-input fusion, and fused leading-batch slices. Unsupported
batched materialization, targets, layouts and precision policies return explicit declines.

This constructor is not numerical permission or public runtime selection. Every emitted artifact
still passes reconstruction, ordered ABI and complete-write checks, and physical scalar guards
remain enforced by graph preflight before allocation. Explicit public dispatch consent, retained
selection evidence and the Gemma training trajectory remain the next gates. Automatic defaults,
split-K coverage, external training and distributed/multilevel acceptance are not completed here.

The public opt-in is `:schedule {:precision :mixed-f16-f32 :typed-contraction
{:strategy :dispatch-mixed-matrix}}`. Both keys must be explicitly requested; inherited mixed
precision defaults do not grant consent. The existing equation dispatch owns one reconstructed
mixed candidate and an exact portable fallback, shares measured-selector consumption, and keeps
the full operational model in its numerical/tuning identity. Unsupported target/layout admission
retains the exact path with explicit decline statistics. Unconsumed selectors, fallback pruning,
tile-space expansion and split-K requests fail rather than being silently ignored. Automatic
defaults and vendor matrix schedules remain unchanged. This is public compiler/lowering and
artifact-roundtrip acceptance; native trajectory/performance acceptance is still pending.

Uniform Long shapes reuse the existing checked per-stage matrix specialization. The public
graph's integral carrier follows its declared GraphScalar width rather than the first consuming
node's physical width; exact and mixed alternatives therefore share one stable public ABI.
Individual kernel parameters and their checked int/long conversions are unchanged. Direct
physical range conditions are projected once by the existing owner and checked before node
conversion/scratch sizing, making a valid public Long outside an int specialization an explicit
schedule-admission failure. Invalid public representations remain caller errors; computed
private expressions still require their own checked preflight. Mixed-width dimensions remain
declined. This intentionally changes serialized declared-integral graph ABI; compiler/build
identity must invalidate older artifacts, and no compatibility wrapper is introduced.

### October 3 runtime/state integration update

This update supersedes the producer-integration gaps below only for the specifically validated
plain dense fields; it does not change the other seven campaign items or claim campaign completion.
PR #1027 connects original completed receipts and exact owner-bound storage measurements to the
existing numerical manifest/content-provider lifecycle. Selected output/post-state fields stream
through bounded aligned downloads under a private output lease. Physical producer, schedule,
representation and field/content bindings are derived; semantic coordinates/policy remain declared.
Independent review fixed a cross-thread provider deadlock by restricting the lifetime monitor to
validation/plan/lease acquisition. Affected content/capture/publication/skip tests pass 46 tests /
962 assertions, including actual OpenCL and Level Zero generated heat capture and public-codec
restored continuation. Exact-HEAD review and all seven CI gates preceded squash merge.

PR #1028 migrates the existing full-domain coarse/fine heat restart oracle away from handwritten
producer/byte-order metadata. Three witnessed replays capture both fields, verify availability,
record numerical-state parents and separately check execution-replay parents. The independent
JVM evolution, uninterrupted six-step device baseline, mapped-file restore and conservation checks
remain. Affected multilevel/state/capture tests pass 27 tests / 257 assertions; exact rebased-HEAD
review and all seven gates preceded squash merge. Both fixtures explicitly use synthetic packaged
build evidence and test providers/metadata, not release authentication or a production store.

#### Item-8 integration in review: captured partial-patch restore

The current strict `verify-restore!` compares complete producer provenance. Completed capture
retains dynamic execution fingerprints, schedules and representation observations; predicting
those from a target would either become circular or confuse producer and consumer programs.
Keep the existing strict four-facet API unchanged. Do not copy incoming provenance into expected
contracts, introduce arbitrary subset matching, or substitute the continuation artifact for its
source producer.

Use a fixed composed boundary instead: independently specified complete field/geometry, phase
and numerical-policy semantics, plus an expectation derived from the original sealed source
Prepared and ordered semantic output/post-state keys. Resolve field/node bindings through the
retained source interface/effect evidence; share the existing frontier calculation rather than
adding a second role/donation heuristic. Check exact program identity and ordered source/node
bindings, and require each captured field-producer content reference to match its full-field chunk.
Reject ambiguous public aliases because v1 capture retains nodes, not public keys. Check the
selected dtype's measured representation kind, dtype, source program and normalized byte order
against its stored chunk; these are internal consistency checks, not predictions of target facts.
Keep dynamic replay and numerical-state parent evidence distinct and auditable. Matching these
contracts remains consistency/compatibility, not authentication of an untrusted serialized claim.

The branch migrates the existing partial-patch remap/restart fixture while preserving its independent
layout/active-cell ordering, phase, wrong-same-extent rejection, host remap/evolution, mass and
same-backend exact continuation oracles. Add negative source-artifact, field-binding and content
counterexamples before any lease/upload. This slice remains unmerged pending final independent
review and all seven exact-head CI gates. Device packing, scalable hierarchy validation,
subcycling/reflux, external training migration and real fabric/device measurements remain open.

#### Resident patch packing acceptance

The next slice exercises production hierarchy-derived local/packed indices through the existing
typed gather/scatter source operations. A public compiled program packs a partially covered
coarse patch and a fine patch into active-cell order, then unpacks into caller-owned patch fields.
Independent host indexing and JVM execution agree bit-for-bit with OpenCL and Level Zero,
including signed zero and preservation of unselected covered coarse cells. The affected geometry,
packing and CI selection/skip ratchets pass 14 tests / 1,162 assertions in the capped REPL.
This adds no AMR-specific kernel, owner, codec or session convention. It is bounded resident
packing acceptance, not scalable hierarchy performance, general patch checkpoint materialization,
temporal interpolation, subcycling or reflux. Independent review and CI remain required.

#### Bounded subcycled diffusion and synchronized restart — October 4

PR #1031 is merged after independent review and all seven exact-head CI gates. It adds
temporal boundary interpolation and signed transport accumulation as ordinary source operations,
and retains physical capacities independently of checked traversal minima. Fusion still proves
the consumed output AxisMap; capacity guards do not substitute for source/index proofs.

The next bounded numerical vertical composes those operations with existing finite-volume,
CSR, gather/scatter and restriction operators: a complete coarse prediction, two ratio-2 fine
substeps, signed reflux, then average-down. Pure projection admits one strictly interior aligned
2D fine patch and periodic boundaries, checks enumeration budgets before materialization, and
retains physical face-integrated rate units. Projection does not certify timestep stability,
runtime ownership or a general adaptive cycle; synchronized inputs and a stable timestep remain
explicit caller obligations. No AMR-specific emitter, cache, allocation or session is added.

Independent coordinate-loop oracles cover square and anisotropic rectangular geometry,
conservation, transport signs, coarse-boundary interpolation and equilibrium scratch replay.
Completed capture stores both synchronized fields using original producer and owner-bound
representation evidence. The source closes before mapped-byte restoration into a fresh execution;
continuation matches uninterrupted resident feedback bit-for-bit on both OpenCL and Level Zero.
A different declared phase is rejected. The affected suites pass 12 tests / 316 assertions locally,
with both native backends available. Synthetic build/provider fixtures are not release provenance
or a production store. Independent review and CI remain required before merge. General hierarchy
scaling, adaptive/multi-patch or mid-cycle restart, external accuracy/performance comparisons,
external training migration and real distributed fabrics remain open.

#### Public mixed-precision training integration — October 4

The existing mixed Gemma training oracle still uses the compatibility descriptor fixture and
manually observes candidate selection. Do not replace it with a weaker public test: a reproduced
public `compiled/lower` at its `CFG-MP` shape with `:precision :mixed-f16-f32` binds portable
contractions, because `:auto` deliberately remains portable. An explicit `:matrix` request
correctly declines the FP32 storage with `:dtype-not-dpas`; that strategy is currently the direct
half-storage instruction request, not the packing-plus-matrix schedule. Policy permission does
not establish which instructions were emitted. This is a real integration gap, not an AD bug or
evidence that mixed training already executes through the equation-first path.

The reporting prerequisite adds ordered `:kernels` labels to the common executable description,
which public execution reports already forward. Each leaf retains only its own artifact labels;
missing precision remains nil rather than inheriting graph/request policy. Graph-level mixed
precision and a leaf's numerical contract are different facts. Labels describe bound artifacts,
not replay events, timing, or new compiler certification authority.

Continue in this order, reusing the existing typed mixed-matrix builder and graph refinement:

The reconstruction prerequisite is a target-neutral `mixed-matrix-schedule` pass. The existing
materialized, tile-input-fused, dynamic-LHS, split-K and batched NN/NT stage construction moves
there rather than being copied into another emitter. Target emission consumes the planned graph;
hardware admission and runtime selectors remain outside the pass. This extraction does not yet
prove a refined graph's complete-write or bounded-error law, admit it to equation dispatch, or
migrate the mixed Gemma gate. The next certification step must independently reconstruct from the
retained equation/source boundary and verify the terminal writer, dependencies and emitted bodies.

The reconstruction prerequisite now retains a closed versioned physical recipe, including input
fusion and batching decisions. Reference planning derives semantic operands, dimensions, layouts,
epilogues and the public boundary from the retained typed algorithm and an independently supplied
source graph. Split-K compiler-owned accumulator and flat work-item identities are capture-checked
by their constructors, allowing exact complete-graph comparison without printed-form equality or
a permissive alpha-equivalence check. This remains reference reconstruction, not certification of
the candidate graph, emitted stage bodies or its complete-write/numerical law. Those checks must
land before admitting public mixed-matrix equation dispatch.

The following extraction moves existing matrix-body and scheduled-certificate construction into
`mixed-matrix-body`, shared by stage emission and independent reference projection. Split-K and
batched views, dimension parameter allocation, epilogues and fused input regions retain the same
constructor path. Target emitters retain parameter spelling and artifact emission. This is still
not whole-graph certification. The following slice moves layout and canonical split-K combine
certificates through the same pass and adds reference projection for every stage node. Production
emission consumes these shared constructors; tests compare complete scheduled certificates, not
only emitted labels or ABI shapes. Before using this projection as algorithm validation, require
exact independent typed stage-graph reconstruction, including canonical range and slice geometry.
Terminal complete-write checks and graph-derived composed numerical policy remain required before
public mixed-matrix admission.

Local native validation is currently unresolved: the matrix suite reports queue/upload failures,
and an unchanged pre-extraction register-tiled baseline reproduces Level Zero queue error
`0x7ffffffe` and OpenCL upload error `-5`. Hardware-free checks and CI are separate evidence;
do not claim local device acceptance or mixed training progress from the extraction alone.

The source-law prerequisite shares the existing TypedSOAC segmented-reduction constructor between
ordinary lowering and exact semantic rederivation. The stronger contraction query compares the
reduction state, neutral values, recurrence, algebra, axes, scopes and retained semantic metadata;
only grid and schedule remain separately checked physical policy. It does not accept the
refinement's own source as independent evidence. Admission is deliberately restricted to canonical
lambda-local-free contractions; retaining arbitrary ordered lambda locals in the general reduction
region remains language-coverage work. This query is not yet wired into mixed dispatch admission:
fresh semantic graph derivation, stage reconstruction, body/projection checks and complete-write /
composed-numerics checks must be joined before that migration.

Mixed-matrix reconstruction validation now joins the exact typed reduction-law query with
complete stage-graph reconstruction and ordered stage-body projection. Semantic fingerprints
retain type metadata that ordinary collection equality ignores. A matching candidate/source pair
with a changed reduction neutral is rejected against the retained algorithm. This is deliberately
not complete-write or composed-error certification, and is not wired into public admission yet.
The remaining boundary integration must preserve the independently rederived public graph's
capacity expressions, scalar order, preconditions and optional inferred ABI, rather than impose
the compatibility descriptor's logical operand extents and explicit argument order.

The public graph's missing ordinary-fold read minima are addressed through the existing verified
operand AxisMaps, not a GEMM-specific recognizer or a second recurrence decomposition. Core reads
retain every distinct map, including repeated accesses to one buffer; unsupported shifted/gather
reads decline this optional query. Undeclared-map search is bounded. The graph owner's existing
contract remains: unresolved storage gets a proved minimum checked at binding, while known storage
capacities are preserved and guarded when insufficient. These logical minima do not authorize
physical address projection, rearrange the callable ABI, prove epilogue/neutral reads or admit a
mixed schedule. The mixed planner still needs to preserve this independently derived boundary.

The mixed planner now preserves an independently supplied graph boundary verbatim, including
inferred ABI, scalar order and capacity guards. Public input/output identities, storage roles and
dtypes must match its locally derived matrix interface; every logical read/write extent must be
covered by a capacity contract or explicit guard. Private layout/conversion/split buffers retain
their logical stage extents. Extra public scalars needed solely by capacity guards are preserved.
The compatibility/standalone interface construction remains a fallback, not authority for a
public equation graph. Reconstruction tests use a fresh public equation graph and reject matching
source/candidate undersized or wrong-dtype boundaries. This still does not admit public dispatch:
terminal complete-write and composed numerical policy are next.

EmittedParallelEquation now reconstructs mixed refinements from its independently derived public
source graph and compares every artifact's scheduled-body certificate with the pure stage-body
projection. Matching boundary or stage source alone cannot authorize a changed generated body.
Artifact graphs with an inferred public ABI use the same ordered ABI finalizer as ordinary SegOp
emission; compatibility graphs retain their explicit interface. This is artifact reconstruction,
not complete-write coverage, numerical composition, target admission or performance evidence.

The existing target-neutral matrix topology analyzer now belongs to the scheduling passes, not
the GPU backend. All three target-emission consumers reuse that owner. This mechanical move
prepares complete-write validation to share the same fragment/store/mask analysis without a
proof-to-backend dependency; it introduces no new accepted schedules or numerical claims.

The first mixed complete-write projection is deliberately direct, nonbatched and full-K only.
It reuses the shared matrix topology analyzer, checks the exact two-dimensional group count,
and joins the generated terminal/unique writer to the authoritative plain FP32 result. Its
domain is the retained free-axis product; public buffer overprovisioning does not enlarge it.
Changed stores/masks/launches cannot borrow a matching stage source's proof. Split-K combination,
batched result views, composed numerical permission and public mixed dispatch remain separate
follow-ups. Static empty KernelBody storage still rejects under its existing shape contract.

Full-K leading-batch results now compose the same per-slice store partition over the verified
contiguous `[B,M,N]` parent/view contract and exact group-axis-2 launch. NN/NT and either shared
input reuse this proof; split-K remains a reduction-composition follow-up, not an independent
batch. The coverage fixtures also caught and corrected the shared transposed RHS's internal
logical shape: `K×N` plus the permutation describes physical `N×K`, rather than applying the
transpose twice. This changes no surface semantics. Numerical permission remains separate.

Mixed numerical permission now consumes an operational model derived from the independent
reconstruction and matched to every emitted body, including fused conversions. It records stage
storage/numerics, actual matrix instruction/tiling, traversal, batching and result transforms.
Dispatch requires both `:approximate-model` permission and an exact canonical match in the
ordered `:permitted-models` vector; the default remains the exact portable candidate. An attached
candidate claim cannot replace rederivation. This is explicit precision-change consent, not a
universal finite error bound: IEEE half overflow and target-defined instruction exceptional
values remain visible. Existing refinement metadata's historical `:bounded-error` name does not
assert a proven finite bound. Public schedule selection and target admission remain next.

1. Extend equation dispatch and complete-write validation to consume independently validated
   mixed graph refinements. Preserve the graph's explicit bounded-error contract; inspecting only
   the first packing kernel is not a numerical proof. Output ABI is not a complete-write proof.
2. Share the existing FP32-to-FP16 packing/tile-local fusion, matrix, epilogue and split-K candidate
   construction with equation-first emission, bound to the retained typed equation, source graph,
   external ABI/effects and target descriptor. Add no training-specific emitter.
3. Introduce an explicit mixed-matrix dispatch request alongside the exact portable default.
   Do not redefine direct half-storage `:matrix` or change `:auto` before measurements justify it.
4. Preserve actual selection/admission evidence at the equation binding boundary, separately from
   its later fixed graph binding. Do not claim runtime candidate attempts from a requested policy.
5. Migrate the existing 25-step Gemma trajectory test only when the public path retains its
   selected matrix graph and leaf evidence. Keep loss reduction and FP32-versus-mixed trajectory
   oracles intact. The existing mixed builder is initially Intel/DPAS gated; vendor matrix rows
   must reuse this seam without claiming device acceptance from compiler-only CI.

This interleaves with item 5 and the emitter consolidation; it does not replace the original
eight-item campaign, external real-weight training, benchmarks or distributed execution work.

This section supersedes historical "next" and "still required" statements below where the cited
implementation now exists. It records evidence, not a declaration that the campaign is complete.
The detailed local execution and preparation evidence is in
[local-compiler-evidence.md](local-compiler-evidence.md). The wider campaign remains open;
do not infer a published release from a merge, or device performance from source compilation.

| Campaign item | Current evidence | Remaining acceptance |
|---|---|---|
| 1. CI feedback and coverage | `.circleci/config.yml` uses sixteen deterministic test shards plus public CUDA/HIP compile and OpenCL CPU gates. Targeted laptop checks use one capped REPL. The OpenCL corpus gate now writes its complete same-compile report under `test-results/coverage`, which CircleCI already retains. | Continue measuring shard balance and skip reasons; preserve whole-suite coverage while reducing feedback latency. Green compile jobs alone do not permit merge. |
| 2. Compatibility-debt ledger | `test/raster/compiler/compatibility_ledger.edn` has executable signature checks; `test/resources/coverage/gpu-corpus.edn` retains every corpus row, including errors. The baseline was refreshed through the real CPU OpenCL entry on 2026-09-21. | Keep the committed portable baseline and full target-specific CI report distinct; distinguish public invocation coverage from compatibility-entry census results. Never delete failed rows to improve the count. |
| 3. Direct TypedSOAC and retirement | The portable baseline now contains 252 source functions: 239 flat TypedSOAC, 1 typed structured-control, 5 scalar value helpers and 7 explicit host-only rows. The ASR resident-capture change advances only its two independently recompiled rows: the allocating wrapper becomes explicitly host-only and the caller-owned routine emits two independent KernelBody maps. Its local host/device acceptance is described below; #924 merged after all seven CI gates passed, including the same-compile corpus ratchet. Generated matrix/input-fusion, reductions, counted stores, AD/SGD, ABM effects and numerical operators use retained typed programs and fail closed instead of recovering through the source-shaped OpenCL emitter. Disabled schedule-family policy is now distinct from a real lowering decline in coverage reports. City two-exit/multiple-recur probes have exact local parity; P20f retains an explicit unbounded unchecked-int induction decline. | Finish external ASR consumer migration; keep source-language support broader than this fixed corpus visible through external regression cases. Preserve numerical, effect and overflow semantics rather than forcing unsupported loops through a fallback. Source coverage is not competitive scheduling or vendor device performance; matched accelerator execution and roofline evidence remain required. |
| 4. CUDA/HIP public-source verticals | Public numerical, attention and matrix fixtures compile in hardware-free vendor gates; allocation-free LinkPlan lowering is tested separately from source emission. | Vendor compilation is not vendor device correctness or performance. Keep matched execution/benchmark campaigns outside the laptop/CI dependency chain. |
| 5. Reusable training and external models | Direct equation-first AD/SGD and small resident Gemma/LoRA tests check state progress, numerical trajectories and admitted routes. Isolated committed external sources have LoRA/QLoRA 30-step parity and tiny Gemma forward/VJP/update replays against the JVM on both local backends, with shared mutable adapters (#950). Public typed scalar objectives now compose through ordinary rank-zero distributed domains (#1124–1125). No sibling checkout is changed. | The unchanged real-weight gate remains held: eleven of 28 adapter checks fail the pinned componentwise threshold despite passing loss/input-gradient checks. Local arithmetic/realization diagnostics are not acceptance. External adapter migration, save/load replay, longer training and preparation-cost acceptance remain; do not revive retired resident-descriptor APIs. Ranged sharing and asynchronous lifetime proofs remain separate. |
| 6. Hardware-free distributed simulation | `distributed_plan_test.clj` covers topology, dependency/resource overlap, directed routes, capacity gates, collectives, halos and certificate drift. | Calibrate against real execution and broaden workload projections without presenting analytic costs as measured performance. |
| 7. Data-parallel/halo execution | Checked DAGs and co-located logical workers use shared physical budgets; unequal heat partitions execute resident halo copies and match the monolithic reference. Collective refinements bind actual public reverse AD, row-weighted FP32 all-reduce and resident SGD for unequal batches on both local backends (#1117–1118). #1126 extends this to three explicitly composed updates with one parameter allocation per worker, fresh per-step scratch and checked dependencies; independent rounded oracles check trajectories and replica equality. | Real multi-device/fabric execution, asynchronous overlap and sustained/dynamic trainer execution remain. The finite three-step DAG is not replay of stale startup evidence or a released trainer. Co-location validates ownership, arithmetic and scheduling, not network throughput. |
| 8. Durable numerical state and multilevel/AMR | Heat and changed-layout coarse/fine evolution resume from addressed mapped bytes on both local backends, matching uninterrupted chains bit-for-bit (#926/#988–990). Bounded plain-patch device packing exists (#1030). Generated ratio-2 diffusion subcycling/reflux and synchronized restart have independent numerical oracles (#1032). #1127–1131 prepare the committed resident cycle through public composition and bind retained temporal stages to its exact Prepared, hierarchy-only workload, whole fields and private register lifecycle. The device follow-up executes two fresh-context cycles through the ordinary distributed owner on both local backends. | Generalized packing and scalable multi-patch/adaptive temporal planning, completion-backed distributed durable publication/lineage, general-rank/hyperbolic providers and mid-cycle restart remain. Structural certificates and bounded numerical results are not an arbitrary PDE theorem, external AMR accuracy baseline or production store. |

### Item-7 training reconciliation (2026-10-09)

`distributed_training_device_test.clj` now joins real local loss gradients to the collective
refinement and SGD consumer through allocation-free public lowering and checked LinkPlans.
Local batch sizes `[1 3]` and `[1 2 4]` exercise unequal contribution weights; each gradient is
scaled by its local row count before the declared FP32 sum tree, and SGD scales by the total
row count. Independent coordinate loops reproduce the leaf rounding, binary tree rounding
and final update. Both local backends pass, and final parameter replicas agree bit-for-bit.
The affected fixture passed four tests / 70 assertions, including hardware-free compilation
with session, allocation and instantiation tripwires.

This is one numerical update with co-located logical workers and synchronous resident copies.
It does not establish multi-step parameter-state reuse, asynchronous overlap, fabric throughput
or real-model training acceptance. The runtime deliberately rejects replay of a completed
one-shot owner; future training steps need an explicit checked parameter-state transition,
not a reset that silently reuses stale initializers. The independent real-weight Gemma gradient
gate remains open, and its original model, provider, numerical policy and tolerance are unchanged.

Item 8 now also has a production 2D hierarchy projection in `raster.ode.amr-geometry`: rectangular,
anisotropic, multilevel active rows retain patch provenance and match geometric sides into faces
and CSR incidence. Native evolution/remap/restart fixtures consume it, retaining independent
geometry and numerical oracles (28 tests / 1303 assertions). Fresh array materialization trusts
constructor output and adds no certification or ownership authority. Production cell-average
remap now uses exact normalized geometry, an overlap sweep and the existing FP64 CSR operator;
different finest lattices are covered by independent rational-overlap oracles. The affected
transfer/native evolution/remap/restart suite passes 16 tests / 375 assertions. Device patch packing,
scalable hierarchy validation, durable publication/lineage and general-rank numerical providers
remain open; the campaign is not complete. The reconciliation below supersedes historical
blanket statements about missing producer identity and subcycling/reflux.

### Item-8 reconciliation against implementation (2026-10-08)

PR #1032 already supplies `raster.ode.amr-subcycle`: a two-level ratio-2 diffusion cycle
with coarse prediction, temporal ghost interpolation, two fine steps, signed coarse/fine
flux-register accumulation, conservative reflux and average-down. It uses existing typed
finite-volume, CSR, gather/scatter and multilevel operations, not a domain-specific emitter.
`amr_subcycle_test.clj` compares complete fields and registers with independent coordinate
loops, including rectangular anisotropic geometry and incorrect-sign/frozen-boundary controls.
`amr_subcycle_device_test.clj` contains OpenCL/Level Zero parity and synchronized restart
checks, including rejection of a mismatched restart phase. Test presence is not fresh device
execution evidence; report each backend's actual result and capability skip separately.

The admitted numerical scope is one strictly interior fine patch, two levels, ratio 2,
periodic boundaries and caller-supplied stable timestep/synchronized initial state. Projection,
materialization and input preparation certify neither timestep stability nor initial coarse/fine
synchronization; both remain caller obligations. Adaptation, multiple fine patches,
additional levels, general-rank/hyperbolic providers and mid-cycle restart remain open.

Producer identity is also no longer wholly absent: `resident_state.clj` captures completed
program/storage/content facts from execution evidence and independently derives source-port
bindings/program identity from the prepared executable supplied at restore;
`build_manifest.clj` supplies packaged compiler identity. An unmanifested source checkout
does not gain persistent-cache eligibility. These facts do not establish durable publication,
distributed lineage realization or mathematical correctness of an arbitrary numerical operator.

The actual planning gap is narrower but important: `AMRPlan` still has only prolongation and
restriction operations and hierarchy-only/transfer-cycle modes. It does not retain the cycle's
temporal interpolation, substep dependencies, register lifetime or reflux/average-down ordering.
Integrate the existing numerical cycle into the compositional planning contracts before
claiming end-to-end adaptive AMR scheduling. Do not implement another diffusion/reflux kernel
vertical to compensate for this missing outer representation.

Fresh focused host validation on `b853cc5a` plus this documentation reconciliation:
`amr-subcycle-test`, `amr-geometry-test` and `amr-transfer-test` pass 14 tests / 1233
assertions in the capped REPL. This does not refresh the device/restart measurements or
close the real-checkpoint training gate.

Fresh local refresh on `5e3a7af6` exercises the five AMR geometry, transfer, subcycle,
device-packing and device-subcycle namespaces: 20 tests / 1359 assertions pass without
capability skips, including OpenCL and Level Zero packing and synchronized restart.
The bounded plain-patch packing vertical already exists in #1030; historical statements
that *all* device patch packing is missing are superseded by that evidence. Generalized
packing, temporal outer-plan integration and adaptive/multi-patch acceptance remain open.

### Rank-zero distributed compute composition (2026-10-09)

Explicit local compute domains now admit shape `[]` under the same dense single-leaf,
dtype, equal-volume, containment and ownership checks as positive-rank tensors. It projects
exactly one physical element; an absent shape, extra element or rank-mismatched offset
does not gain admission. Structural regressions preserve the original ABI leaf and allocation.
A complete public prediction/MSE objective also runs through an ordinary one-worker
DistributedPlan on OpenCL and Level Zero and matches the JVM scalar loss. No scalar-loss
opcode, trainer API, special transport or runtime replay convention is introduced.
This closes local scalar-output composition, not multi-step/fabric training or the held
real-weight numerical gate.

### External training publication readiness (2026-10-08)

The local `../finetune-rstr` repository at `9e9ba5d` has no configured Git remote;
the expected GitHub repository is not resolvable by the current account and the Clojars
artifact API returns 404. Its README badges are not release evidence. Raster's public
north-star names this workload, but that does not make its source available to collaborators.
The local README/lora GPU file and decision-training subtree have uncommitted work;
preserve that work and obtain a reviewed source snapshot rather than silently publishing it.

The public `pretrained-rstr` repository supplies model loading/inference and numerical-memory
orchestration. Its local source/test tree has no LoRA/QLoRA training implementation; the
local-private-model roadmap lists a reproducible LoRA fixture as future work. Keep checkpoint
loading there and adapters, objectives and training orchestration in finetune-rstr.

Before presenting finetune-rstr as an installable training package:

1. Migrate its existing head/layer/adapter programs from retired `bind-program!`/`run-program!`
   to the public Compiled/LinkPlan contract; do not revive a compatibility runtime.
2. Replace the old default Raster `0.2.287` dependency with a validated released version,
   and make clean-clone tests independent of sibling checkouts and a local Valhalla JDK.
3. Preserve the held real-checkpoint gradient gate and add an end-to-end adapter save/load
   and inference replay fixture with unchanged base weights. Record numerical/performance
   limits honestly; small synthetic parity is not real-model training acceptance.
4. Review uncommitted sources, license, build/release setup and model-data handling before
   separately authorizing GitHub/Clojars publication. No publication is implied by this ledger.

CUDA work can continue against Raster's committed external-source acceptance fixtures and
public pretrained-rstr while that migration proceeds. It must not require an unpublished
sibling package to execute Raster's hardware-free compiler gates.

### Distributed algebra witness consolidation (2026-10-08)

Review found that the analytic collective witness retained routes/algorithm names but omitted
the reduction algebra, broadcast root and numerical mode. Accumulating halo witnesses retained
only combine/identity/dtype, dropping the algebra's numerical facets. Retained records also
bypassed their constructors, so record identity could masquerade as validation after mutation.

The existing associative scalar authority now independently rederives retained certificates.
Distributed scheduling revalidates operations, schedules and communication legs even when they
are records. Collective certificates retain the reduction/root/numerical mode, and halo
certificates retain the complete associative certificate. Valid semantic changes require new
distributed evidence; missing/forged algebra facts reject before certification. Old witnesses
must be regenerated. This adds no operator registry, domain kernel emitter or runtime transport.

The execution boundary remains narrower than the analytic planner: readiness accepts only copy
halos, and refuses reducing collectives/accumulating halos before sessions or allocations.
Arithmetic execution must eventually transfer into explicit scratch and run a generated typed
combine as an ordinary bound compute step. The analytic witness is not a collective numerical
equivalence proof, a real-fabric measurement or completion of campaign items 6/7.

Focused scan/distributed-plan/physical-binding/AMR-plan validation passes 70 tests / 370
assertions in the capped REPL. Independent review found no blockers; its extra qualified-operator
and retained-leg mutation checks are included. Full cold CI remains a merge requirement.

### Distributed physical assignments before arithmetic lowering (2026-10-08)

The next seam binds ordinary copies to exact existing local graph regions through
`DistributedPlan.copy-bindings`; see [distributed-materialization.md](distributed-materialization.md).
It reuses BufferView projection, shared copy extent/alias checks and the existing readiness/runtime
authorities. Source storage must realize the transferred value and stay within its owned region;
constant destinations, unproven ghost sources, malformed or strided endpoints and stale witnesses
reject. Explicit copies cannot override semantic collective or halo legs. The assignment may
target a different mutable logical value: it is a physical copy contract, not value-version or
collective algebra equivalence evidence.

Focused scan/distributed-plan/compute/AMR-plan tests pass 74 tests / 392 assertions. A generated
`acopy!` → resident scratch copy → generated `axpy!` fixture passes on the actual Level Zero device.
The complete affected numerical namespace passes 13 tests / 61 assertions after separately fixing
independent owned LinkPlan instantiations' runtime root identity collision. That lifetime fix keeps
compiler and explicitly borrowed identities unchanged; it does not relax root retention guards.
Independent physical-copy review found no blockers. Exact-head cold CI remains required; semantic
collective lowering, packing, real fabric/overlap and full-model training acceptance remain open.

The strict semantic restore boundary now checks the independently supplied target field/layout,
logical phase, numerical policy and provenance before leases/uploads; bit-preserving certificate
verification also rejects opaque metadata and signed-zero drift. Affected state/native restart,
lease and AMR checks pass 40 tests / 454 assertions. This matches declared contracts, not omitted
application semantics or mathematical equivalence; durable provider/publication and distributed
lineage acceptance remain open, despite the producer/source evidence now available.

External source audit refreshed on 2026-09-30: authoritative pretrained-rstr main
`98bad4d3517ec05a3c495f019c80d8c553db02af` declares Raster 0.2.922 and uses the public
compiled boundary for Laya/ModernBERT. The older local `f4c2bb4` checkout (0.2.545) and
`pretrained-rstr-main` checkout (0.2.457) do not describe current upstream. An isolated
current-main snapshot is used for focused acceptance; no sibling code or dependency was
modified. The local finetune-rstr `9e9ba5d` checkout still declares 0.2.287 and its layer
forward/backward code names the retired binder, with local edits that must be preserved.
Existing in-repository training or external inference evidence must not be substituted for
the remaining external training gates.

### Execution order from this checkpoint

1. Finish the local public workload/support matrix and host-preparation audit before adding
   vendor-device claims. Keep complete-write evidence distinct from conditional initialized
   postconditions; fresh prefix results must remain legal without crediting untouched parent
   tails. Scalar reductions retain their exact one-element resident representation. Invalid
   shape diagnostics propagate; unsupported proofs may decline. Public validations remain
   independent; redundant proofs may be shared only inside an exact synchronous construction scope.
   Exact-owner static program evidence now reaches final synchronous construction, after the
   projection callback. Concrete call and memory/effect checks remain fresh; public verification
   stays independent. The October 2 affected suites pass 103 tests / 696 assertions, with both
   local native backends available. Warm Q4 preparation is about 11.7 ms in one diagnostic,
   not a controlled performance baseline. Continue auditing structured-call invariant completeness
   separately; this deduplication does not certify arbitrary forged carry-plan/staging metadata.
   The subsequent source-ordered call audit now retains entry buffer bindings and checks loop
   trip counts, carry rotations, numerical argument projection, final bindings and exports against
   each step's actual entry environment. It does not reselect dispatch alternatives or rerun host
   callbacks. Its affected suites pass 120 tests / 889 assertions, with six cache-owner checks
   passing another 55 assertions. Comprehensive host-step evaluation evidence remains separate.
2. Select the next compiler migration from a reproduced public workload, using the compatibility
   ledger to distinguish stale entry-point accounting from an actual semantic/emission gap. Preserve
   the corresponding JVM path and independent numerical tests while retiring the migrated fallback.
3. Coordinate the external training adapter boundary using public compiled artifacts/LinkPlans and
   semantic outputs. Validate chained gradients and real model state before claiming that item done.
4. Continue matched public GEMM/projection measurements under the protocol in
   `bench/comparison/generated-kernel-protocol.md`. Current noisy laptop samples do not promote a
   winner or establish SOTA parity; a cloud accelerator is optional later evidence, never a CI gate.
5. Extend numerical execution to a conservative PDE evolution with coarse/fine transfers and an
   uninterrupted-versus-restored oracle. Reuse existing field/lease/plan contracts. Introduce any
   additional semantic operation only when that workload exposes a concrete missing obligation.

The completed-reduction conversion follow-up removes the numerical-region projection gap for
one explicit Float/Double cast of the completed accumulator. Partial storage stays at the reduction
dtype. The scalar cotangent alias projection in #948 also removes the reproduced cold dynamic-helper
typedness gap; #949 admits placement across unrelated pure equations without crossing effects.
General mixed-width epilogue arithmetic and public Float-accumulator widening remain separate.

The immediate emphasis returns to unresolved compiler/training workload coverage after the durable
transfer acceptance; the distributed and scientific work does not replace the emitter agenda.

### Checked scalar conversion checkpoint

#565–#568 landed runtime VJP-cache invalidation, resident scalar composition, shape-only array
captures, and KernelBody signed narrowing with explicit target trap contracts. Source conversion
integration keeps checked casts distinct from unchecked wrapping, preserves ordered lexical
initializers and unused checked prefixes, and separates one-shot extent evaluation from CSE.
Beichte's external-effect classification remains separate from descriptor-derived exceptional
control obligations; only retained exact-conversion or checked constant evidence removes them.

The integration follow-up keeps implicit result/storage conversions out of the source-cast tree,
so explicit casts inside maps and stores retain their original checking and rounding behavior.
Padded map lanes guard the complete scalar region, not just memory operations. Retained branch
and loop-entry intervals may prove an integral narrowing exact; an unrestricted long-width argmax
still requires a trapping target or an explicit bound proof. Required exceptional prefix shapes
have one evaluator in the typed InvocationPlan, with source-order evidence retained across
non-equation host bindings. These changes are under PR #569 validation, not a green release claim.

The declared result cast of `par/map!` is itself source semantics, distinct from an implicit
array-store coercion. The integration uncovered a missing boundary when the body contains no
explicit inner cast. A typed scalar-conversion term is being validated through fusion and target
projection; it must retain materialization rounding and exceptional ordering, not merely the
destination dtype. Once-per-element evaluation alone does not justify moving a potentially
trapping producer across a consumer that writes observable storage. JVM AOT also now threads
declared scalar/array facts into the shared frontend, and explicit primitive long-to-int bytecode
uses checked narrowing. Focused JIT/AOT boundary tests pass; the full PR remains unmerged pending
conversion/fusion and CI regressions. Boxed numeric and native-C exceptional conversions remain
separate audit gaps, not covered by the primitive JVM result.

Typed scalar conversions also expose a JVM SIMD capability boundary: an exact float identity
conversion can disappear only inside a float vector computation. It remains a rounding boundary
when embedded in a double reduction. The shared typed scalar route preserves mixed-width programs;
all-float map/reduction cases must still vectorize. General mixed-width Vector API conversion and
lane-shape handling remain an explicit performance task, with numerical materialization-boundary
oracles required before admission is broadened. A scalar fallback here is not SIMD parity.
The historical raw source-cast SIMD path still lacks authoritative operand dtype evidence;
matching its destination to the vector species alone does not certify arbitrary nested casts.

Host-only scalar steps currently execute during preparation. Checked scalars after device work
therefore decline rather than moving a potential failure ahead of preceding writes. General
interleaved host/device exceptional control remains future work. Portable OpenCL still declines
unproven trapping device casts; CUDA/HIP/Intel termination is not a JVM exception-delivery promise.
This work does not certify the scalar simplifier's historical floating-point algebra under all
NaN/signed-zero cases, nor complete the remaining counted-map migration or compatibility census.

### External scientific comparisons

The [source-reviewed extension experiments](scientific-extension-validation.md) add Julia/SciML
and matrix-free references while preserving the existing ODE and iterative-solver library. They
distinguish library capabilities from verified resident/distributed compilation; do not infer
missing numerical algorithms from gaps in the new accelerator fixtures.

Use numerical frameworks as well as kernel compilers, with separate kernel and time-to-solution
measurements. Initial comparisons should match precision, discretization order, boundary conditions,
mesh, timestep/stability constraints, stopping tolerance and transfers. Also compare time to a fixed
solution accuracy, rather than treating a cheaper discretization as a compiler speedup.

| Reference | First Raster workload and question |
|---|---|
| [Devito](https://www.devitoproject.org/examples/userapi/01_dsl.html) | Heat/acoustic stencils: symbolic numerical programs, fusion, scheduling, bandwidth and halo work. |
| [AMReX](https://amrex-codes.github.io/amrex/docs_html/) | Block-structured coarse/fine transport: ghost exchange, conservation, refinement interfaces and placement. |
| [PETSc](https://petsc.org/release/manual/dmbase/) | Poisson/implicit evolution: compose operator application, residuals, preconditioners and convergence control; measure time to the same tolerance. |
| [MFEM](https://mfem.org/howto/assembly_levels/) | Later matrix-free finite elements: element/face gather, basis/quadrature contractions and assembly/scatter layouts. |
| [UFL](https://docs.fenicsproject.org/ufl/main/manual/form_language.html) | A semantic reference for domain-aware mathematical expressions and differentiation, not a standalone runtime benchmark. |

Start with Devito, then AMReX and PETSc; use MFEM/UFL to challenge regular-grid assumptions later.
Keep these comparisons out of the hot test loop. Record dependency revisions, numerical settings,
hardware and compiler/runtime revisions with each result. No installation, cloud service or sibling
source modification is implied by listing these baselines.

The architectural test is whether numerical problems lower through the same small functional core:
for example, gather states → compute face fluxes → signed accumulation → update cells. Mesh entity
relationships, geometric factors, boundary conditions, solver convergence and numerical error
contracts belong above that core when needed; scheduling, storage and communication remain explicit
below it. Changing resolution is a numerical approximation with a projection/error contract, not
automatically an equivalent compiler rewrite. Current rectangular cell-centred AMR and explicit
distributed DAGs do not yet establish arbitrary mesh, solver or adaptive-control generality.

## 1. Close product reduction production lowering — landed for the dense row subset

- Landed #382: shared typed multi-result scalar-region lowering.
- Landed #383: candidate row-product KernelBody, CPU OpenCL differential execution, CUDA/HIP
  compilation. At that stage production still used the retained source emitter.
- Landed #384: preserve TypedSOAC local dtypes through SegRed and subsequent
  scalar/index lowering; compare both direct and local-address candidates on CPU OpenCL.
- Landed #385: align the retained product KernelGrid with the segment-count launch,
  validate scratch/workgroup agreement, and replay the body refinement against the retained source.
- Certify the exact semantic source, storage boundaries and public bindings against the graph.
- Landed #386: derive product pointer contracts from the exact retained
  scheduled-equation projection, shared with fold-map. Unknown capacities and unlowered storage
  representations decline. This does not prove arbitrary indexed accesses in bounds or enable
  production selection; those obligations remain explicit.
- Exercise ordinary public compilation, resident execution and hidden/multiple tuple results.
- Cover required axis/bound variants and zero-row planning; reject unsupported cases explicitly.
- Empty-domain kernel follow-up uses one physical group and a workgroup-uniform guard enclosing
  every reduction operation, barrier and store. CPU differential tests execute this path and verify
  untouched outputs. This does not implement zero-byte device allocation or graph-node elision.
- Long-bound follow-up preserves independent row/column int or long facts through the public
  scalar ABI and widened induction. CPU differential execution includes Long dimensions and
  retained local addresses; CUDA/HIP fixtures compile the same candidate. This is still not the
  production route; source access requirements remain open, as does eventual zero-row elision.
- Before widening public dimension support, bind concrete launch geometry to KernelBody's int
  hardware-index representation, including scheduler overrides and compatibility staging. Logical
  Long dimensions must not silently overflow group/local indices; target resource limits remain
  a separate, potentially stricter contract.
- Select the certified body and delete the old product algorithm emitter once its coverage is
  replaced. Keep necessary target instruction lowering.
- Typed access follow-up extends the existing AxisMap relation with a bounded conditional
  intermediate-range proof over KernelBody indices. Exact polynomial coefficients prevent host
  overflow; every add/multiply subtree is checked before affine normalization. The proof assumes
  active positive axis domains and checked resident capacity; deriving those access requirements
  from scalar lowering and enforcing them in graph binding remain production obligations.
- Read-requirement follow-up shares product scalar-region lowering with scheduling and derives
  flat minimum capacities from actual typed dense loads, without invented input shapes. The
  graph preserves larger declared capacities and combines shared-storage requirements; existing
  checked resident/staged binding enforces graph capacities. Gathers, broadcasts and combine
  reads remain unsupported by this initial access proof. Production selection still requires
  successful access admission in addition to exact source/body/graph correspondence.
- Production follow-up admits dense row products through that combined check and the common
  C-family graph emitter. Public resident extraction uses the existing executable convention;
  effect-only staging has an explicit nil result policy, including multiple written outputs.
  Direct descriptor graph binding now enforces the same capacity premise as session/LinkPlan
  binding. Hidden/multiple outputs are checked across all three target dialects; public argmax
  has resident and staged CPU OpenCL numerical coverage.
- Retirement follow-up removes the handwritten product emitter from production sources and
  preserves its algorithm in `test/raster/compiler/reference/product_opencl.clj`. Existing source
  contract tests and CPU OpenCL differential execution retain that independent oracle. This does
  not promote its broader source-only cases (computed reduction bounds or multiple segment axes)
  into supported production coverage; the initial certified route remains dense and rank-one.

Exit: public product workloads use the typed route without reconstructed facts or source assembly.

## 2. Retire the remaining compatibility paths — active

The public staged Byte→Int32→Float contraction is now a separate compatibility-ledger
workload. Unlike the explicit-map Q4 row projection (already KernelBody), it declines
direct TypedSOAC admission and reaches retained `:staged-segred` emission. Its current
compatibility path requires explicit `:dtype :byte`; default Float policy emits Float
operand pointers despite declared Byte inputs, which the resident binder correctly
rejects. Do not count fixture-only staged KernelBody emission as this public vertical.

The next admission must retain the existing `SoacContract`/`ContractionFacts` payload,
including nested stage boundaries, typed lifts and storage dependencies, through the
typed program and projection. Flat ProductReduction components are simultaneous
accumulators, not nested stages. Check operand/lift/output types independently; do not
relax the frontend option gate or flatten stages into a scalar fold. Existing fusion
rules must decline staged nodes until their legality preserves those boundaries.
Acceptance includes exact fact transport without reparsing, public mixed-storage
execution, source return/effect preservation, and checked graph capacities/aliasing.

The dependency prerequisite projects core, stage-lift and epilogue storage plus scalar
captures once from ContractionFacts. SOAC and SegOp accessors share it; stage/epilogue
binders have local scope, and declared maps replace placeholder indices before capture
analysis. Decode expressions reading undeclared storage are explicitly rejected. A direct
lowering test checks equation operands/effects and Byte input, Float scale/output/result,
and integral scalar types. Floating declarations now specialize only under floating
kernel policies, and contraction result aliases inherit the destination type. This
repairs the existing dependency envelope, not direct TypedSOAC staged admission.

The next dialect slice represents a `contract` equation as a lexical closure over the
same ContractionFacts, with ordered external array/capture value IDs. SSA renames those
IDs, not the nested stage expressions or axis binders. Validation checks independent
storage dtypes/capacities, scalar closure, output shape and explicit read/write storage
contracts. Existing flat fusion rules recognize but do not rewrite this operation.
The binding projection retains the original facts for the generated staged body and
maps its arguments onto graph value IDs. This is a prerequisite: the public frontend
still needs admission and production routing before its compatibility ledger can close.
Closure validation is not an independent arithmetic type checker: it checks declared
storage/scalar types and lexical scope. The existing scheduling and scalar-body lowerers
must still prove instruction types and numerical legality. Unlowered representations,
layouts and sharding decline, and outer lifts cannot refer to inner-stage coordinates.

Production graph lowering now transports that closure as a bound SegContract, with
canonical dependency accessors used by graph construction. Before generated staged
emission, the existing graph projection certificate and an exact operation/binding
comparison both run; the resulting ScheduledKernelBody is checked against that node.
This route uses the common C-family graph emitter and resident graph binder. A small
Arc device test exercises vector SSA IDs and distinct row/column outputs. Public
frontend admission and the compatibility-ledger transition remain the next step.

Read-only inventory at #383 identifies these priorities; dynamic reachability must be measured
before treating a definition as live or dead:

| Priority | Boundary | Representative acceptance |
|---|---|---|
| 1 | General contraction fallback in `contract_route` / `segop_opencl` | General/scientific contractions, exact decline inventory |
| 2 | Unscheduled effect maps in `opencl_pass` / `par_opencl` | Resident mutation, aliasing, nil/buffer returns |
| 3 | Strided scatter/gather source emission | Block transfers, collision and view contracts |
| 4 | Seeded active-ID generation | ABM firms, stable order and count/capacity; not general predicate compaction |
| 5 | Compound-local algorithm emission | Structured loop dependencies, storage and barriers |
| 6 | Staged/quantized contractions | Decode/lift algebra, typed accumulators and overflow |

Audit apparent unused wrappers separately. A source oracle is not a production fallback; removing
one must not erase its independent numerical evidence. Track JVM and C/SIMD compatibility as well
as GPU paths, using the same production corpus and retained TypedSOAC facts.

A repository caller audit found four orphan NN source generators (row softmax, group norm,
vector scatter-reduce and scalar scatter-reduce): their only source callers were unreferenced
adapter wrappers, while tests inspected generated strings. The wrappers are removed and the
unchanged generator bodies live in a test-only reference namespace. Existing source assertions
remain, with scalar-scatter coverage added; these are source-shape oracles, not numerical or
accelerator performance evidence. Shared dtype/extension helpers remain production infrastructure.
This removes dead algorithms from production, not a live workload's fallback or an optimization.

The handwritten tiled GEMM generator and legacy direct DPAS source entry are now test-only
oracles. Production DPAS artifact construction has only the verified KernelBody branch, preserving
its source, ABI, epilogue metadata and launch geometry. Independent source comparisons and
historical benchmark rows still use the relocated generator; those rows are explicitly not public
compiler performance evidence. The legacy orientation gate remains shared with oracle tests for
now; live staged quantized contractions still require the epilogue-splice helper. This retirement
does not remove their source emission or the general gathered-contraction fallback.

The zero-reduction contraction router now attempts the shared portable contraction KernelBody
lowerer, using a semantic SegMap projected directly from verified facts. The map branch emits
scalar operations and one store without a synthetic reduction, and obtains each input extent from
its own proven AxisMap (not the output count). Initial admission is deliberately limited to
positive static free-axis extents whose product fits the int launch ABI, no options/result transforms or destination reads, and one
index expression per input array. Unsupported maps retain an explicit compatibility decline.
Focused CPU OpenCL execution uses exact input lengths and a partial final workgroup; the same
routed body is included in CUDA/HIP compile fixtures. This is a contraction-router migration, not
yet the public equation-first frontend: that frontend still excludes zero-reduction contractions.
Admitting them through retained map equations requires preserving independent input capacities;
the generic map lowerer's output-sized input shapes must not be reused as an outer-product proof.

The first generic map-capacity step preserves known positive static GraphBuffer capacities in
each KernelBody pointer shape/layout, including larger retained input capacities. An independent
check compares those static pointer contracts to the exact source graph node before target
emission. The typed-map fixture carries AbstractValues a[4], b[3], C[12] through ordinary typed
scheduling and graph construction; CPU execution uses exact buffers and rejects undersized inputs
and outputs. Source, launch and arguments are unchanged by enlarging a retained capacity. This is
storage-contract correspondence, not proof of arbitrary indexed access safety. Symbolic, unknown
and zero capacities still retain the compatibility behavior; no shape-only scalar ABI is added.
Public zero-reduction contraction admission remains a follow-up, as does access-derived capacity
refinement where the frontend has no declared shape.

CI also exposed target-registry leakage from test fixtures: matrix-capable synthetic targets can
change another test's automatic precision route. The direct contraction ABI test now requests its
FP32 policy explicitly. The isolation follow-up gives the hardware registry tests and the three
resetting cross-compilation fixtures fresh device, initialization and calibration atoms, restoring
the original identities even on exceptions. This is a serial-JVM fixture; parallelism remains
across CI processes. Other registration sites still need an ownership audit.

The first measured elementwise gap is unary subtraction in `scale-clamp-exp`: its typed source
reached the verified SegMap source fallback solely because the scalar lowerer required binary
subtraction. The follow-up uses the existing negation intrinsic and shared prefix spelling for
floating operands (preserving signed zero); integral operands retain checked subtraction and
its explicit portable-OpenCL trap limitation. Public route, CPU OpenCL IEEE cases and vendor
compile fixtures cover this increment. It does not retire the remaining ordered-loop fallback.

The next measured gap is `sum-kv-heads`: its ordered carry starts at one after loading the first
element. The origin follow-up retains nonnegative literal starts in the existing scalar ForLoop;
zero-origin SOAC recognition is unchanged. New nonzero coverage requires direct recursion, exact
binding-slot updates, one loop body and no narrowing induction test. Public GQA fan-in, group-one
signed-zero preservation and cancellation-sensitive accumulation are checked on CPU OpenCL.
This does not authorize reassociation or arbitrary symbolic/negative loop origins.

Coverage reporting now retains the normalized emission routes and decline details from each
existing corpus compilation. TypedSOAC frontend coverage alone cannot distinguish generated
KernelBody from a compatibility emitter. Aggregate counts explicitly describe emitted artifacts,
including dispatch alternatives, not executed launches. The portable baseline remains unchanged;
target-specific route evidence stays in the report and CI output. No extra compile is required.

Static caller audit found the old `par-hip` elementwise source emitter referenced only by its
historical compile-gate test, not by any production routing path. It is moved to the test-only
`reference.elementwise-cuda` namespace with the algorithm unchanged. Historical ABI/math compile
checks remain; mandatory CUDA/HIP CI continues to compile public equation-first KernelBody
fixtures. This removes a misleading second production emitter, not a supported runtime target.

The next caller audit distinguishes a remaining source edge from demonstrated runtime reachability:

| Boundary | Retained owner / consumer | Current evidence |
|---|---|---|
| General contractions | `generate-segmented-reduce-kernel` → `contract_route` | Nested gather selects the source fallback after `:operand-layout` decline. |
| Unscheduled effects | Typed mini-program → KernelBody or fail-loud decline | Raw/nested plain-array and logical-SoA effects schedule once; unsupported bodies cannot re-enter source emission. |
| Strided transfers | Typed mini-program → KernelBody | Bare and host-wrapped leaves share typed scheduling; source generators moved to test-only oracles. |
| Active IDs | `generate-par-active-ids-kernel` → `opencl_pass` | ABM firms uses seeded agent indices, not arbitrary visibility compaction. |
| Compound local | `generate-compound-local-kernel` → `opencl_pass` | Compatibility markers remain; structured typed programs bypass detection. |
| Staged/quantized contractions | `generate-staged-contraction-kernel` → `contract_route` | Device-tested staged/lift/quantized source assembly remains. |
| JVM SIMD / scalar | `pipeline` → `par_simd` or source expansion | Bound SegOps coexist with traversal of the retained source projection. |
| CPU C/SIMD | `cpu/aot` → `cpu/csimd` | Bound SegOps preferred; compatibility reconstruction and distinct vector emission remain. |

Before migrating staged contractions, their numerical admission must preserve the claimed
flat-equivalence law: stage initializers are proven zero using the shared checked-constant
semantics, and packed signed-byte accumulation requires every prefix to fit the DP4A instruction's
Int32 result, even when the surrounding accumulator is Long. The reusable interval-prefix proof
uses unbounded host arithmetic. This closes those admission gaps, not the remaining source
emitter's general integer overflow semantics or the typed staged lowering itself. Finite-precision
equivalence still needs a per-stage conversion/rounding contract; exact-arithmetic distribution
is not a proof of bitwise equivalence.

DP4A staged admission now belongs to `passes.parallel.staged-contraction-schedule`, not the
OpenCL emitter. Both routing and the retained source generator consume the same checked packed
maps; the old emitter-owned admission entry is removed. The next production slice is typed
packed-stage loops and lifts, using existing ScalarLoad/ScalarCompute/ForLoop operations and
shared scalar lowering. DP4A instruction acceptance itself proves no accumulation bounds: retain
the schedule's prefix gate. General scalar integer stages additionally need a checked recurrence
proof (including zero trips and cross-carry dependencies), not weakened KernelBody validation.

A REPL packing probe lowers four byte reads and existing bit operations into verified KernelBody
without reinterpreting the public byte-buffer ABI. Its target emission exposed an adjacent shift
contract gap: C-family word shifts now explicitly mask counts to the retained Int/Long width,
shift unsigned words, and restore arithmetic-right sign bits. Byte shifts decline centrally until
source lowering supplies an explicit promotion/result contract. This does not authorize narrowing
Long source shifts to Int. The shared fixture covers sign-bit packing and signed/count extrema on
OpenCL, with CUDA/HIP source compilation in CI. It is a portability prerequisite, not evidence that
four byte loads match the throughput of an aligned packed-word load or that staged emission is
already unified.

Staged route construction now reads the body and output dtype from the same verified contraction
facts as its axes and stages. It no longer reconstructs or reparses a compatibility form just to
build the existing staged emitter's spec. Regression tests disable both source entry points and
compare ordered ABI, scalar bindings, launch geometry and stage metadata for scalar and packed paths,
including a non-default output dtype. The staged algorithm is still source-emitted; this retires
an input-representation duplication, not the remaining target emission path.

The first typed packed-stage **candidate** now constructs ScheduledKernelBody directly from those
facts. Two-level Int32-dot/Float-lift reductions use byte loads, explicit word packing, DP4A, nested
typed carries and masked stores through the common target projector. Static AxisMap products bound
required storage and index arithmetic; semantic decode declarations, unproved prefixes, arbitrary
lift regions and unsupported domains decline. The required shapes are not proof that an arbitrary
raw backend caller supplied enough storage. Production admission must retain the checked resident
capacity boundary; the low-level OpenCL test binds known correctly sized buffers.

The generic `emit-static-dense-graph` projection now places a ScheduledKernelBody behind the
existing KernelGraph binding boundary. It derives every external buffer's required capacity from
positive static dense shapes, preserving logical storage dtypes, exact source/effects, public
scalar dependencies and target-private scalar expressions. Unknown/strided storage is rejected;
this is neither a new memory ABI nor an access proof for arbitrary producers. Tests exercise
operand, lift and output shortages through both session and descriptor binding, writable aliases,
and device replay with poisoned output. The staged candidate can use this boundary without
changing production schedule selection; performance admission remains outstanding.

Candidate validation covers signed-byte extremes, block widths 4/32/64, multiple blocks, distinct
row scales and masked launch tails on OpenCL; CUDA/HIP compile fixtures exercise the same typed
schedule. The old staged source emitter remains production-selected. Before promotion, compare
old scalar/packed and typed candidates on the same workload, including finite-precision rounding
and throughput, then migrate descriptor construction and remove the superseded emission. Four byte
loads are not an aligned packed-word throughput claim. More general stages still need the checked
recurrence and typed lift-region coverage described above.

The follow-up device comparison binds the same logical Byte ABI for typed, retained scalar and
retained packed emission, with independently poisoned outputs. Dyadic scale cases compare exactly;
non-dyadic cancellation compares against explicitly stage-rounded host arithmetic with tolerance
relative to contribution magnitudes. An all-zero result must fail that tolerance. Prepared kernel
handles are released after synchronous execution.

A small shared-laptop Arc probe (OpenCL profiling events, 3 warmup rounds and 12 interleaved timed
samples per candidate, 64 work-items/group, uploads/compilation excluded) gave the following median
milliseconds for `[rows outputs blocks block-width]`: `[1 128 32 32]` typed 0.052708, retained scalar
0.039166, retained packed 0.032708; `[4 128 32 32]` typed 0.041770, scalar 0.039062, packed 0.031458.
All outputs matched the independent reference. Samples were noisy (e.g. typed second-case range
0.040104–0.072187 ms); this is a directional probe, not a reproducible performance baseline or
regression in the unchanged production route. Do not promote the byte-load candidate on this
evidence. Investigate a generic, explicitly typed/aligned packed storage load (with byte fallback),
then repeat resident comparisons before retiring the production source emitter. Do not introduce
a quantization-specific memory ABI or bypass the common body verifier to obtain that load.

One general target-cleanup slice exposes canonical counted loops when their literal bounds prove
the final induction increment representable. The shared exact range helper includes that exit
increment, not just indices observed inside the loop. Ordinary and asynchronous pipelined loops
share one syntax helper; unknown bounds and explicit IndexCasts (including retained Long bounds)
keep the existing unsigned-distance guard. Scalar yields and event rotation stay before advance.
This does not strengthen recurrence facts or alter accumulation order, and the noisy laptop probes
do not establish a throughput improvement. Device tests cover zero/nonunit/reversed loop counts,
staged reductions and the existing asynchronous workgroup pipeline.

Public matmul/dA/dB probes on CPU OpenCL select generated portable contractions, not the
handwritten gather fallback. Their artifact adapter previously reported semantic `:segcontract`
provenance as the emission route. The follow-up propagates the actual emitter's route through
the adapter; semantic provenance remains separate. No schedule or precision changes.

Strided transfer retirement also closes the raw host-wrapper entry: `do`, conditional and
body-position leaves schedule their scalar extent beside the invocation, without moving it out
of the branch. Supplied typed programs are consumed without re-analysis and reject missing
equations. Existing source-oracle assertions remain test-only. CPU OpenCL checks inactive branches,
output identity, gather overwrite and repeated-index scatter accumulation into nonzero storage;
this is correctness evidence, not an accelerator performance measurement.

Active-ID retirement uncovered a shared source/ABI type bug: `derive-param-types` narrowed every
declared Long to int before TypedSOAC, including random seeds. The prerequisite now retains
declared integral widths at every caller of that common derivation. This changes affected scalar
ABIs; consumers must bind the emitted ABI, not assume an int slot for a Long parameter. Hardware
group/local index types are unchanged; schedules requiring int parameters still need explicit
checked narrowing. Public equation-first, resident and session paths are tested with 64-bit state
and wrapping arithmetic. The active-ID prototype remains unshipped until its checked int count
conversion is also retained; this prerequisite alone does not retire that kernel.

The next prerequisite distinguishes shape-value equality from unchecked cast erasure. Source
normalization removes integral identity/widening casts only when retained types prove them safe;
narrowing and unknown conversions remain scalar equations. Relational extent canonicalization
must not subsequently erase the same check. A public checked count is evaluated by the resident
descriptor before driver contact; horizontal fusion retains the scalar equation ahead of its
combined launch, and inactive host branches do not evaluate their local count. This does not yet
repair convenience operations that unconditionally unwrap counts or prove every compiler pass
preserves checked conversions.

Exit: classify all remaining routes; retire duplicated paths for covered workloads with explicit
production coverage and no silent legacy re-entry. Report any remaining gaps rather than declaring
the entire compiler unified from one successful vertical.

Scalar-emission audit removes the map/fold lowerer's private copy of primitive cast spellings in
favor of the shared operation descriptor and dtype vocabulary. This does not change narrowing or
rounding policies. Before sharing that lowerer with contraction elements, reconcile its explicit
device-wrap narrowing policy with reduction lowering's checked-cast rejection; do not widen
reduction admission accidentally. Indirect contraction loads also need independent storage
capacities and a data-dependent index-range contract. A dense AxisMap or an index array's integral
dtype does not prove its contents are in bounds. Keep the production operand-layout gate until
those obligations are represented and enforced, reusing existing typed loads and graph capacities.

The scalar conversion prerequisite now derives common rounding/overflow choices from dtype facets
in one pure helper, used by map/fold expression lowering and reduction element lowering. Integral
narrowing defaults to rejection; the existing map/fold device-wrap behavior is an explicit owner
opt-in, not an inferred equivalence to checked Clojure casts. Reduction's source checked-cast gate
is unchanged. Exhaustive primitive conversion pairs and owner-admission checks protect this seam.
This does not unify the full scalar lowerers, change epilogue conversion labeling, or admit new
indirect accesses; those remaining differences must be closed separately.

Retained scalar precision is a prerequisite for further sharing: a nested expression's declared
floating type must govern its arithmetic before a wider consumer converts its result. A direct
map/fold lowerer regression exposed early promotion of a Float-tagged addition inside a Double
addition. Lowering now respects retained expression metadata, including unary/variadic
normalization and branch results; missing metadata still uses the owner's contextual type.
The lowerer explicitly converts the completed floating result back to the owner's declared
dtype, including narrowing before stores and loop yields; preserving inner precision must not
violate those boundaries. Public softmax and heat-equation corpus checks exposed this obligation.
Integral contexts retain their existing behavior: the public Q4 projection has widened loop carries
around narrower dp4a result metadata. Reconciling that boundary is a separate prerequisite, not
permission to silently change accumulator widths in this floating-precision increment.
This is lowerer-level correctness coverage, not a demonstrated public-model miscompile or a
new inference rule. Full source reachability and remaining owner-policy differences remain open.

Reduction element lowering now delegates accepted loads, casts and arithmetic over typed child
values to the shared scalar-expression SSA builder. Its admission adapter still requires retained
nested arithmetic types, preserves literal dtypes, rejects checked integral casts/arithmetic and
comparisons, proves load coordinates through the existing owner callback, and checks the final
accumulator dtype. The shared builder accepts explicit owner conversion and masked-load policies;
map/fold defaults are unchanged. This serves existing scalar reductions and portable contractions,
without adding indirect-access admission or retiring their remaining source fallbacks. Result
transform admission still has its own explicit precision and overflow policy.

The shared scalar builder now reserves source, parameter and owner-scope identities before fresh
SSA allocation, including future local binders. Map and reduction fragment callers seed their
complete source region; generated coordinate names are reserved when the fragment arrives.
Reduction fragments pass only their typed child environment instead of all earlier SSA values,
and seeded calls do not repeatedly scan growing environments. Names remain deterministic, and
KernelBody's duplicate-identity validator remains intact. This closes the demonstrated scalar
capture cases; it is not a claim about every other schedule's allocator or target identifier mangling.

Result transforms now use the same typed load, conversion and scalar-compute builder as reduction
elements and map/fold source lowering. Reduction and result-transform adapters no longer rebuild
source fragments to emit accepted operations. The typed entries take approved coordinate vectors,
converted operands and explicit options; they do not infer source semantics or new range proofs.
Result-transform operand roles (including destination reads), multidimensional coordinates,
conversion labels and trapping integer arithmetic remain unchanged. Their separate admission
rules are not duplicate kernel emitters, and harmonizing those rules remains a semantic decision.

The active-ID prototype is deliberately **not shipped**. Review found that its remainder-based
body matches source `mod` only over positive populations, and its output conversion needs a proved
int range. It also inherited unconditional cast erasure on seed/population captures. Passing
ordinary ABM differential tests does not establish those preconditions. Keep the production route
until shared capture normalization and enforced scalar-domain admission preserve empty/inactive
execution as well as overflow. The local `compiler/active-indices-typed-map` branch retains the
prototype and count-check tests for that follow-up; no retirement is claimed.

The next prerequisite closes the existing RNG route's scalar inputs: source-signature count and
seed conversions become ordered host scalar equations, rather than unwrapping a captured checked
cast. Typed scalar materialization reapplies its retained result conversion, preserving
`long(int(seed))` even after value-equality simplification. Raw inactive host branches keep these
checks local; missing certified leaves fail once instead of recursively rescheduling. Public
descriptor preflight, CPU differential execution and vendor compile fixtures cover this boundary.
This does not admit the active-ID population domain or retire its production source kernel.

Raw effect-map follow-up uses the same one-attempt typed mini-program boundary as RNG and strided
transfers. Conditional leaves keep checked extents beside their invocation. Plain mixed-storage
maps and pre-store lexical snapshots have CPU OpenCL differential checks; tiny host-only bodies
still use the existing scalar fallback. A supplied typed program cannot re-enter source lowering.
Unscheduled raw bodies now fail with the retained typed scheduling diagnostics and `:fallback
:none`; they cannot enter the independent source emitter. Logical SoA values are scalar-replaced
before TypedSOAC construction, and bare, conditional and let-bound probes retain per-field dtypes,
stable field identities and one grouped logical binding through KernelABI projection. Tiny literal
host-only bodies still use the explicit scalar host fallback below the configured GPU threshold.
This retires the production compatibility edge; moving the orphaned generator and its historical
source tests out of production is the following mechanical cleanup.

## 3. General fusion and bounded specialization — price admission active; broader work queued

- Validate shared scalar-region composition, coupled reductions and post-reduction transforms.
- Audit recompute/materialize choices on fan-out, effects, aliases and observable results.
- Apply retained static structural facts before fusion, shape/layout/target facts before scheduling,
  and typed constant/range simplification afterward.
- Preserve ordered arithmetic and AD/effect boundaries. Use guards and bounded specialization
  caches; do not unroll large numerical iteration spaces merely because their sizes are known.

The first generic hardening rejects nonnumeric, nonpositive and nonfinite machine ridge prices,
and overflow of their derived recompute threshold. These prices cannot authorize duplicating a
fan-out producer. Sole-consumer elimination still needs no cost estimate. The witness policy is
versioned; production typed-route tests retain both materialized consumers with an explicit reason.
This is price admission, not a new fusion legality proof or a bounded specialization cache.

An observable-result audit found a separate compatibility-boundary bug: horizontally fused stored
maps submit through an effect-only kernel convention, but their source bindings still return
buffers. Returning the submission's nil value lost the first host alias despite correct device
writes. Typed materialization now emits one effect binding followed by every source buffer alias
as a flat binding; resident extraction needs no new nested-call convention. Effect-only source
maps remain nil. Tests check the public resident descriptor, buffer identity and CPU OpenCL values,
alongside retained checked-count rejection. Kernel ABI, launch count and fusion legality are unchanged.

Automatic Object-array element-class specialization now has a separate, explicit admission bound
(default 32 attempts per generic function, configurable down to zero), with generic dispatch at
capacity. One atomic claim owns each pending compilation, and clearing the cache revokes its old
completion token. Failed/pending entries consume the budget too. This bounds this existing host
specialization cache, not the number of functions, manually generated JVM classes, GPU shape
variants or global compilation memory. It is not yet the shared cluster/resource admission policy.

Exit: representative scientific and model expressions compile with explained fusion/placement
choices, differential correctness, and measured kernel/allocation/compile-cost changes.

## 4. Measure generated production kernels and selection — GEMM evidence active; broader ladder queued

The [generated-kernel comparison protocol](../bench/comparison/generated-kernel-protocol.md)
defines baseline selection, per-shape records and measurement boundaries. Baseline adapters
and external accelerator measurements remain follow-up implementation work.

Begin with existing production canaries, not independent handwritten builders:

- `test/raster/perf/production_canary.clj`: public AOT sumsq and resident generated GEMM;
  GPU replay currently measures host-synchronized latency, not pure device throughput.
- `bench/soac_contract_bench.clj`: contraction ladder; verify its entry point before reuse.
- `bench/resident_gemm_cold_bench.clj`: useful cold/warm methodology, but its source-oracle builder
  must be replaced before it counts as generated-production evidence.

Measure GEMM, quantized projections, reductions and attention with matched shape, dtype, numerical
policy, warmup and transfer boundaries. Record compiler/tuning time, kernel count, allocations,
peak storage, transferred bytes and execution time. External comparisons and heavier experiments
stay outside the local hot loop. Use device events where available and label CPU OpenCL results as
CPU results. No SOTA claim follows from successful vendor compilation alone.

The first scientific-stencil canary now exercises the public periodic heat `deftm` with an
independent oracle and resident Level Zero event timing. It exposed a one-work-item serial loop
at `128×128`; spelling independent cells as a SOAC map with the store indexed by the map lane
made uniqueness provable and changed the launch to 256 lanes/workgroup. An intermediate
stationary run measured 19.8 µs; the final portable source measured 15.4 µs median but its
sample series was nonstationary. Every run matched the oracle. This is a measured schedule
correction, not a stable speedup claim or vendor-library comparison. Raw samples are retained in
the generated-kernel comparison protocol.
The first attempted map stored through a derived `(int lane)` local and still scheduled one
work item. That is a real proof-coverage gap: a narrowing cast is injective only when its
runtime domain is proven to fit, so the next general ownership slice should carry an explicit
range fact through the local and certify the derived address. Do not strip the cast or trust a
`unique-index` claim without that proof. The current source uses the original lane for the store.

Use the measurements to shortlist legal schedules and validate selective autotuning/cache replay.
If comparable accelerator hardware is unavailable, complete harnesses and correctness gates but
leave accelerator competitiveness explicitly unmeasured.

The GEMM canary records candidate strategy, source/ABI signature, emission routes and entry-point
counts from its exact Prepared value, with no second compilation. These replace the parallel
strategy/signature arrays in the opt-in report. Candidate counts are not executed launches;
descriptor scratch counts exclude graph-owned temporaries and do not claim peak memory. The
register-tiled emitter now reports its actual KernelBody route through the artifact adapter,
separate from semantic contraction provenance. Host-synchronized timing remains labeled as such;
no device-event timing or accelerator comparison is inferred from this evidence.

Exit: reproducible production measurements identify wins/regressions and explain schedule choices;
claims are limited to tested hardware and workloads. Discuss the results before stages 5–6.

## Working loop

Dense workload follow-up: public parameterized GEMM and its explicit typed ReLU epilogue have
precision-selectable correctness canaries. An ordinary contraction-return alias followed by an
in-place map exposed two boundaries: lexical return-type propagation (now uses exact primitive
operand-return semantics, not suffix guessing), then stable-read alias normalization (still
rejected at binding). Closing the latter and measuring automatic fusion remains required; an
explicit epilogue is not evidence that source composition fuses automatically.

Future mechanized proofs may use the sibling Lean project. Start with small existing contracts
(bounded integer range transfer, access/capacity refinement and alias legality), replaying the
same counterexamples as executable tests. No proof assistant integration or whole-compiler
correctness claim is implied, and this does not block workload-driven retirement.

The returned-buffer follow-up separates same-type facts from exact identity facts. Known
destination-return aliases are normalized before access analysis; producers and public return
bindings remain intact. Scope-aware substitution and alpha-renaming of incoming free identities
prevent local shadowing from redirecting a physical buffer. The composed GEMM/ReLU canary now
executes safely through two generated stages. Automatic epilogue fusion remains the next gap.

Static same-destination contraction→map now uses the existing segmented-reduction result-transform
rule. Its sole lane-owned read becomes the completed accumulator; the fused destination is write-only.
Other destination operands, neighbor reads and externally live intermediates remain excluded.
The public static 3×4×5 GEMM/ReLU case is one generated stage and device-checked on CPU OpenCL.
Dynamic composed GEMM remains two stages: its normalized extent scalar separates the equations,
and symbolic extent equivalence plus legal scalar placement are not yet proved by this rule.

Real Intel Arc checks also cover the static composed case and awkward 127×65×33 explicit
GEMM/ReLU under both precision policies. These are numerical checks, not timing claims.
The public equation-first RK4 and symbolic GEMM device checks exposed a runtime boundary:
program-wide shape bookkeeping must remain available for capacity checks, but only the graph's
declared scalar arguments enter KernelGraphCall. The binder now projects that interface without
weakening exact call validation; both device cases execute after the fix.

The next real-device model probes exposed a distinct schedule legality gap: public linear
projection dimensions retain Long, while mixed-DPAS layout/matrix stages currently bind int
extents. That candidate now declines explicitly as `:mixed-dpas-index-width-not-lowered`,
leaving generated portable contraction schedules available. Tiny forward, input-gradient and
weight-gradient executions match CPU references, and the full AD train-step compiles resident.
The existing train-step gate also executes a 2×3×2 AD/SGD update on an available GPU, checking
output extent, numerical agreement with CPU AD and a nontrivial update from the saved input.
This is not an optimized-training result. Restore that optimization with width-preserving
matrix/layout schedules or guarded specialization whose range proof covers dimensions, products
and indexing; do not narrow retained types implicitly or relax ScheduledKernelBody validation.

The first width-preserving prerequisite gives shared layout cast and transpose schedules explicit
int/long extent parameters, with independent row/column widths and unchanged int defaults.
Exact widened address arithmetic and checked hardware launch limits remain unchanged. Small
OpenCL device tests exercise every width combination and poisoned tails; non-allocating checks
cover a logical extent above int range and overflowing shape products. CUDA/HIP fixtures include
long cast and mixed-width transpose. The mixed-DPAS Long admission gate remains closed until its
matrix, split-K and batched stages also retain widths consistently; this prerequisite alone does
not restore optimized Long-dimension GEMM.

The next prerequisite retires the last legacy KernelBody `Loop` record: matrix K traversal now
uses the same typed `ForLoop`/`Yield` vocabulary as scalar/control kernels. Matrix induction and
prefetch lookahead use long arithmetic with exact widening of leaves before computation, including
split-K bounds. The matrix-plan boundary rejects narrow induction and late casts of arithmetic;
it does not erase casts as an unchecked algebraic equivalence. Intel and CUDA emission retain
the induction width. Focused Arc execution covers direct, split-K, shared-weight batched GEMM and
fused epilogues. Public matrix dimensions and Intel block-I/O coordinates still have int contracts;
pitch, output-product and hardware-coordinate bounds must be addressed before admitting retained
Long dimensions. This change is correctness/unification evidence, not a performance claim.

Contiguous leading-slice BufferViews now carry exact long leaf products in KernelBody itself.
The shared validator checks scalar types and preserves parent-shape/launch correspondence;
late casts, narrow factors and non-integral extents are rejected. Scalar and matrix emission
consume that expression without a separate matrix-only widening helper. Offset correspondence
is not a standalone overflow proof: binding must check parent-capacity products, including every
intermediate multiplication (a trailing zero must not conceal an overflowing prefix). Focused
tests cover these non-allocating limits plus the existing real-device split/batched matrix routes.

The opt-in [matrix-width canary](matrix-width-canary.md) compares the generated DPAS leaf with
the test-only historical source using poisoned output and device events. The initial small Arc
run is numerically correct but partly nonstationary; retain raw samples rather than declaring
parity. Public projection performance remains a separate open gate: matrix pitch/coordinate and
linear-address bounds must be completed before restoring the retained-Long optimized route.

Static zero-reduction-axis public contractions now enter the existing typed SegMap path.
For unresolved plain input capacities, a bounded proof over the actual lowered loads derives
minimum storage independently of output size. Every integer arithmetic prefix must fit its
retained dtype; negative accesses, indirect coordinates and unknown domains do not establish
capacities. Larger declared storage remains intact. This is not general gather bounds checking
or symbolic shape inference. Exact-size CPU OpenCL execution is correctness evidence only.

Executable scalar preconditions now use the existing checked launch/storage expression algebra,
referencing physical ABI slot names. Binding validates scalar ranges and literal specializations;
graph preflight checks every node before scratch allocation. Registry and tuning identities retain
these constraints, and forced/cached selection cannot bypass them. Focused validation: 65 tests,
374 assertions. This supplies enforcement, not proof that target requirements have been supplied.

Next: derive Intel matrix restrictions from the verified schedule and share them between selection
and artifact binding. The [Intel 2D block-I/O specification](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html)
requires at least 64-byte row widths and 64-byte base alignment; the existing tiny FP16 N/K=16
device cases do not establish legality despite passing this driver. Validate surface maxima,
pitch and view-slice alignment as well, widen linear output arithmetic, and keep the retained-Long
DPAS admission gate closed until the complete contract is enforced.

The Intel surface-contract slice derives checked bounds and pitch/fragment divisibility once for
static admission, runtime fallback and artifact binding. A/B bases require 64-byte alignment;
leading input views additionally require 32-half-element slice strides, while shared operands and
scalar C stores are not over-aligned. Projection verification rejects stripped requirements.
Intel block-I/O now admits subgroup 16 only, and C linear stores widen before multiplication.
Focused Arc direct, split-K, batched/shared-weight and fused-SSA-epilogue execution passes with
legal surface widths. Non-allocating tests cover maximum surfaces and rejected odd-height/K48
batched slices. This does not admit retained Long dimensions.

Executable matrix stores now require ScalarSSARegion. The semantic ScalarRegion descriptor remains
an input to scheduling, but cannot bypass typed lowering into KernelBody. The old source-expression
store emitter, tag reconstruction and op-map-derived call whitelist are removed. Ordered epilogue
ABI checks remain shared; an explicit rejection test prevents reintroducing unlowered store regions.
The existing typed SSA storage-address emitter serves matrix epilogues on both Intel and CUDA.
Fresh capped-REPL validation: 68 tests, 688 assertions including Arc matrix execution. This removes
a duplicate lowering path; it is not a new tensor IR or a performance claim.

Mixed-precision graph emission now derives each node's logical scalar widths from the original
GraphScalar environment through the existing checked expression algebra. Matrix, cast, transpose,
split-combine and batched stages state identity or checked Long-to-int bindings before target
projection; emitted ABIs are not rewritten afterward. Mixed-width/all-Long graph tests cover all
four orientations, direct/split execution plans, shared-weight batches and pre-allocation overflow
rejection. Focused validation: 83 tests, 1084 assertions. The public Long-DPAS admission gate remains
closed. Existing int-sized layout/combine slots still impose capacity limits; complete their wide
lowering and selector admission before promising a fallback for every oversized specialization.

The following width slice preserves graph-derived physical extents in cast, transpose and generic
split-combine bodies. The segment-count ABI follows its own shape expression, independently of
wide coordinate uses; mixed Int-output/Long-reduction cases remain valid. Allocation-free checks
cover multi-billion-element layout and combine plans. Focused validation: 53 tests, 909 assertions,
plus three Arc matrix tests with eight assertions. Public Long-DPAS admission remains closed.

Intel split-K artifacts additionally require positive, K16-aligned chunks, derived from canonical
scheduled partition bounds. Literal terminal K bounds are tied to the physical K parameter;
unknown sliced forms fail closed. This protects direct artifact binding, not only graph-generated
chunks. Focused validation: 32 tests, 745 assertions including Arc execution. It does not prove that
an arbitrary caller supplied enough partitions to cover K, nor add CUDA partition constraints.

The first guarded Long admission is unbatched dense contraction with all three dimension
expressions retained as Long. Layout and combine extents stay wide; matrix coordinates are
checked specializations, and surface selectors retain the portable fallback. Batched and mixed
widths remain declined pending their separate admission proofs. All four graph orientations
execute on Arc through direct and explicit split schedules; ordinary public Long contraction
executes direct and dynamic split schedules. Selector fallback and forced-alternative rejection
are both tested. The affected route suite passed 85 tests / 1272 assertions before adding the
extra mixed-width rejection cases; forward/input-gradient/weight-gradient and a complete tiny
AD/SGD update also passed (two tests, 12 assertions). These are correctness results, not measured
projection throughput or a claim that every oversized workload has an executable fallback.

Graph admission can now reuse direct public/node scalar ABI ranges from the existing scalar-range
facts. The batched matrix selector consumes these guards, including its private Int grid-Z count,
before surface arithmetic. This helper deliberately does not evaluate computed arguments or
replace artifact preconditions and allocation preflight. Oversized Long batches select fallback;
forced graph binding still rejects them. Focused validation: 32 tests / 741 assertions, followed
by a six-assertion graph-only API check. Public batch admission remains a separate follow-up.

One small memory-capped REPL; focused affected tests locally. Full suites and hardware-free vendor
compilers run on CircleCI. Review candidate/certificate boundaries; squash only exact reviewed heads
with all required checks green. Never alter the concurrently edited main-checkout north-star file.

### Public staged contractions and result-prefix storage

The bounded Byte/Int32/Float staged contraction now enters the retained TypedSOAC closure from
ordinary public `par/contract` syntax, including canonical derivation of operand maps. Declared
Byte inputs and Float scale/output arrays remain independent under Float compilation policy.
Frontend admission and KernelBody lowering share a non-emitting schedule-domain check; unsupported
staged semantics are not silently flattened or accepted beyond the generated schedule's coverage.
The compatibility ledger records this workload as typed and KernelBody-emitted. This does not
retire all staged fallback schedules, generalize every precision/stage combination, or prove SOTA
performance. Those remain explicit follow-ups to the common generated schedule vertical.

Storage access requirements are lower bounds, aggregated across contractions. Known larger
capacities are preserved. A map reading a smaller static prefix uses the existing indexed capture
representation, not a weakened element-tensor shape. Logical prefix results get LinkPlan views
over their physical destination allocation, with no additional copy or initializer. Emitted calls
retain the semantic result/destination relation; planning and runtime independently check actual
view identity, type, extent and prefix containment. Runtime view resolution is mandatory before
binding such calls. A partial write establishes only its proven logical prefix, not the tail.

Validation includes hardware-free CUDA/HIP public compilation, short-buffer and forged-view
rejection, retained-call mutation checks, fresh-prefix initialization, and Arc execution of
contraction alone, in-place prefix map, and disjoint-destination map with unchanged tail values.
Known capacity is supplied at compilation for these larger-buffer cases. General runtime capacity
adaptation and dynamic view specialization still need their own invocation/view contracts; exact
invocation tensor-shape checks remain intact. Indexed captures can conservatively limit fusion;
recovering pointwise fusion through explicit views belongs with the broader typed fusion work.

### Generic floating staged reductions

Static floating staged contractions now use the same retained closure and resident graph route
as packed contractions. Their arbitrary-depth reduction loops and per-stage conversions are
constructed with the shared scalar SSA builder and emitted through the common target emitters;
there is no floating-stage source template. The packed schedule remains preferred when its
legality proof applies. This ordered scalar schedule establishes a general correctness baseline,
not a claim of competitive tiling or throughput.

Frontend capability analysis constructs the same typed stage plan as lowering, without target
emission or device work. Each buffer retains one declared storage dtype; captures require
authoritative types. Explicit and implicit compound scalar conversion boundaries require retained
type evidence, preserving checked integer arithmetic before floating conversion. This stricter
contract is opt-in for the new route; older packed scalar owners still need separate migration.
Portable OpenCL rejects checked trapping arithmetic, while Intel OpenCL, CUDA and HIP provide
the corresponding target helpers. Target-neutral admission does not imply universal target support.

Validation covers three floating stages, scalar captures, cancellation and signed-zero identities,
common resident binding, public compilation/execution, all four source dialects, unsupported-domain
declines, and retained checked integer arithmetic. Broader integer stage schedules, dynamic domains,
cooperative scheduling, target capability-driven alternatives and external performance comparisons
remain follow-ups; the older staged fallback is not yet fully retired.

### Staged scalar result transforms

Floating staged contractions can now execute their scalar epilogue after the entire nested
reduction, through the same strict scalar builder. The retained closure checks accumulator scope
and excludes closed reduction indices; shared storage requirements include declared epilogue
operand maps over free axes. Explicit scalar declarations must agree with authoritative capture
types, both at schedule admission and after SSA value binding. No stage/epilogue source template
or operator registry is added.

The generated path retains the current output dtype and casts the completed scalar transform
explicitly. Destination-reading transforms still require an inout storage proof and decline before
frontend admission. Decoded operands and broader integer stage schedules remain separate work.
Public validation includes post-reduction placement on Arc, all common source dialects, short
epilogue-buffer rejection before allocation, and CUDA/HIP vendor fixtures using the same public
workload. Whole staged-emitter retirement remains contingent on the uncovered numerical contracts.

### Shared load-transform semantics

The host contraction macro and retained staged emitter now share the same capture-avoiding
per-operand load-transform substitution. The host previously ignored `:decode`, so it was not a
valid numerical oracle for decoded kernels. Decode applies to the core summand before stage lifts
and the final result transform. Summand locals are alpha-renamed before substitution so they cannot
capture decode free variables or inherit a shadowed outer array's transformation.

This correctness prerequisite does not yet admit decoded loads to the generated staged schedule.
That migration still requires retained types for the raw-load binder, decoded scalar expressions,
and any integer widening/overflow semantics; no target-side type inference is introduced here.

### Contraction lexical typing (in progress)

The source walker now supplies contraction-local type contexts for axes, decoded raw loads,
child-stage accumulators and completed-result accumulators. It uses existing inference rather
than teaching target emitters to reconstruct these types. Explicitly unknown lexical bindings
mask stale reference metadata; absent bindings still retain metadata-based rewalk evidence.

Compiler-generated scalar casts are qualified so stage linearity can distinguish genuine
identity conversions from caller-local functions. Flattening only removes a cast when its
canonical identity and child accumulator dtype prove it redundant. Decoded GPU admission is
still gated separately; this prerequisite does not retire uncovered staged fallback cases.

### Decoded loads enter the ordinary typed scalar region

The source walker now normalizes load transforms before inferring contraction arithmetic.
Source aliases and macros are resolved before the shared capture-avoiding substitution; the
result is ordinary scalar arithmetic with explicit casts, not a second decoded KernelBody path.
Typing the transformed expression from the start is essential: substituting a Double decode
into an already Float-typed product would preserve an incorrect early rounding point.

Qualified primitive casts use the existing operator descriptor for their result types in shared
inference. The public decoded staged workloads use the generated staged-scalar schedule, with
zero fallback and zero driver allocations during compilation/lowering. Validation includes
alias/macro reads, an actual-device widening-rounding differential, and the same public-source
fixtures for CUDA/HIP compilation. The compatibility ledger records the generated route.

This does not claim mixed-storage or integer-overflow coverage, whole staged-emitter retirement,
or competitive kernel throughput. Unnormalized direct typed facts remain explicitly gated;
remaining precision/storage cases and measured schedule performance are still campaign work.

### Recursive floating stages around a proved packed inner fold

The generated staged schedule composes the existing verified byte-product Int32 fold beneath
multiple Float/Double stages. Its shared packed fragment is also used by the original two-stage
schedule; packing remains four byte loads and typed word operations, with no pointer reinterpretation.
The existing packed admission proves exact-product replacement, contiguous declared maps,
four-divisible extent, zero identity and every Int32 accumulation prefix. It is not a general
integer-loop invariant or a new quantization-specific rule.

Every outer stage explicitly converts its lifted term to its declared accumulator dtype, including
identity lifts. Public three-stage compilation and resident Arc execution use the generated route;
an independent reference rounds each Float stage with non-power-of-two scales. Hardware-free tests
retain mixed Float/Double outer dtypes and reject unproved integer, layout and lexical cases.
The public workload is included in vendor compile fixtures and the compatibility ledger.

Nonpacked integral folds, unproved overflow, decoded byte products and arbitrary mixed storage
remain separate work. This closes a recursive composition gap, not whole staged source-emitter
retirement or a throughput claim; vectorized packing and cooperative schedules still need measurement.

### Independently checked additive carry ranges

KernelBody validation can derive a static, single integral carry updated by `carry + term`.
Only carry-independent scalar compute/load prefixes participate. The verifier derives term ranges
through existing SSA validation, proves every prefix fits, and distinguishes body-entry prefixes
(N−1 updates) from result prefixes (N updates). Ordinary body validation is replayed with that
evidence; overflow policy is never changed by the proof.

External range provenance is deliberately discarded during this optional analysis, and
IndexCompute prefixes are excluded: mathematical index intervals alone do not prove intermediate
machine-width safety. If this weaker analysis fails, ordinary validation with original facts still
runs. Nested/control-flow/multi-carry recurrences retain conservative ranges. Tests cover fitting
and overflowing prefixes, nonzero initialization, zero-trip preservation, carry dependence,
post-loop facts, unsafe index-derived terms and safe fallback after speculative failure.

This is a verifier prerequisite, not yet broader integer-stage admission or general index-arithmetic
safety. Nonpacked/decoded byte reductions still need their source-to-schedule validation and device
comparisons before a migrated source emitter can be retired.

### Nonpacked integral stages through the shared verifier

Static Int/Long inner folds now use the recursive staged scalar schedule when the shared
KernelBody verifier proves every accumulation prefix fits. Four-byte packing is an optional
schedule, not the semantic admission boundary: widths 1, 3 and 5 and normalized byte decodes
are covered. Admission validates a normalized body specification through the same checks used
when materializing KernelBody, without allocating a record or invoking a target emitter.

Strict typed scalar lowering preserves retained integral operation widths before stage casts.
Checked or wrapping arithmetic retains a mathematical range only when the entire range fits its
machine dtype; this does not rewrite overflow policy. A public Long multiply/divide decode tests
the distinction from premature Int arithmetic. Independent Arc references cover byte extrema and
Float rounding, and both public decoded workloads enter the CUDA/HIP/OpenCL compile fixtures.
Possible overflowing folds still decline before schedule admission. This is correctness and
coverage work, not a throughput claim or proof that all staged source fallbacks can be removed.

### Separate accumulation, result-transform and storage precision

The recursive staged schedule converts its completed result to the declared output storage dtype
instead of requiring storage to equal the outer accumulator dtype. The shared conversion policy
rejects unsupported floating-to-integral conversions and unrequested integral narrowing.
An epilogue first converts to its own declared dtype, including a bare identity expression, then
converts to storage dtype. Neither conversion changes the preceding accumulator types or rounding.

Independent Arc cases distinguish Float accumulation then Double storage (0 rather than 1) and
Double accumulation then a Float identity epilogue then Double storage (16777216 rather than
16777217). Both are public vendor compile fixtures. Single-stage closure admission and the legacy
quant router's implicit accumulator semantics remain separate retirement work.

### Explicit single-stage contractions share the same closure

A nonempty stage list now admits its one-stage base case through the same lexical, storage,
legality and shared KernelBody proofs as deeper nests. No special kernel or quantization rule is
introduced. Public decoded width-three byte products use a proved Int fold followed by Float
storage; exact width-four products use the existing packed fragment through that same recursive
base case. Both have independent Arc references and public vendor compile fixtures. Extent
mismatch and seeded stages remain errors; unproved Int prefix bounds remain capability declines.
This admits explicit single-stage source contracts; the legacy router's implicit stage synthesis
and layout-retargeting still need normalization before the old staged emitter can be removed.

### Explicit accumulator policy enters the shared stage path

For explicit `:acc-dtype`, a single-axis additive reduction with a proved zero identity and
nonempty output axes can expose its canonical accumulator as a one-stage schedule. The helper
reads dtype, combine and neutral from the existing ProductReduction view; it does not infer
accumulation precision from Byte storage. Output dtype is retained independently. Integral zero
is spelled at the accumulator width after stage legality proves the identity is zero.

Source epilogue typing also retains `:acc-dtype` before output conversion. Public Arc cases cover
decoded Byte/Int accumulation and a Double accumulator with Float storage and a rounding-sensitive
epilogue. Both enter vendor compile fixtures. Rank-zero, multiple reduced axes, non-additive folds
and undeclared accumulator precision retain their existing paths. Implicit legacy Byte policy still
requires a numeric equivalence proof; this is not a blanket Byte-to-Int rewrite or emitter retirement.

### Input storage does not choose accumulation precision

Single-axis sums with input storage differing from the output can use the same canonical-stage
projection without changing accumulator dtype. The frontend selects it for mixed storage or
explicit accumulator policy; homogeneous contractions keep their existing schedule selection.
This still requires one common declared dtype among core input arrays, not arbitrary heterogeneous
operand storage. Existing rank/domain, layout, capture and numerical legality gates remain.

For an innermost floating stage over integral storage, untyped compound roots now decline rather
than borrowing arithmetic width from the buffer dtype. Retained source types govern operations;
typed loads/casts use their existing scalar contracts. Tests inspect Long byte-product arithmetic
separately from Float scale arithmetic. Public Arc execution of 1024 products of -128 by -128,
then two products of 1 by 1, gives ordered Float result16777216 rather than Int-then-Float16777218.
The public workload enters vendor compile fixtures. This is correctness/generalization work;
packed replacement of a floating fold still requires an independent equivalence proof.

### CUDA/HIP baseline findings integrated into the existing campaign (2026-09-12)

The reference review used local JAX/Pallas, Triton, CUTLASS `147295a3`, Composable Kernel
`704ff575`, and rocWMMA `97562d3`. It does not introduce a separate IR-redesign campaign.
Raster already schedules workgroup reduction/scan trees and typed matrix store epilogues.
The next portability increments belong to the current emitter-retirement and measurement work:

1. Finish indexed subgroup KernelBody emission and retain its numerical/dispatch tests.
2. Lower existing coordinate-free FP32 TileStore regions on CUDA WMMA, initially uniform
   scalar transforms such as scaling and ReLU. Logical row/column indices and tensor operands
   must decline: WMMA fragment slot mappings are opaque.
3. Add a direct HIP matrix row from the existing instruction/body contract when its selected
   instruction and wave geometry can be compiled and validated. rocWMMA is a reference for
   fragment load/MMA/store, not a mandatory semantic dependency.
4. Admit staged matrix mainloops and indexed epilogues only with schedule-visible storage,
   coordinate ownership, barriers, and resource accounting. CuTe's mapped output partitions
   and rocWMMA's double-buffered LDS GEMM are the concrete references.

Each row needs public compiler-path acceptance before being called a completed vertical;
target-only compile fixtures are prerequisites. Cold performance comparisons stay outside the
laptop hot loop. Resident training/model validation, distributed simulation and execution,
durable manifests, and numerical PDE/AMR acceptance retain their existing completion gates.
Fused GEMM plus auxiliary reductions (CK's mean/mean-square example), warp specialization,
and TMA remain workload-driven follow-ups rather than vocabulary added in anticipation.

### Shared fragment-emitter prerequisite

The CUDA direct-fragment mainloop spelling is separated from target admission. Ordered ABI,
instruction legality, aligned dimensions, view/slice declines and uniform store-region lowering
remain checked by the CUDA entry point. The internal renderer consumes the analyzed body plan;
it does not introduce a second schedule, an allocation, or source-level type inference. CUDA is
its only admitted dialect in this slice. This prepares HIP without copying the GEMM algorithm.

Hardware-free reference probes with rocWMMA `b5a884dc` (ROCm 5.7.1) and the laptop HIP 5.7 /
Clang 21 toolchain compiled f16×f16→f32 fragments for both gfx90a and gfx1100. Unbundling the
code objects and disassembling confirmed `v_mfma_f32_16x16x16f16` and
`v_wmma_f32_16x16x16_f16`, respectively. These are reference-library feasibility results, not
Raster-generated kernels, numerical validation or performance evidence. The newer reference
checkout requires ROCm 6.4 headers; it cannot be assumed compatible with the laptop installation.
The first Raster HIP matrix row should use the existing CDNA MFMA/wave64 witness, with a
separate architecture compile gate. RDNA WMMA/wave32 requires its own explicit capability;
the current gfx1100 scalar compile gate must not be mistaken for MFMA coverage. A pinned
header/toolchain contract, generated-body compilation, physical ABI requirements and public
route tests remain prerequisites before enabling either route.

### HIP MFMA source candidate shares CUDA's admission and fragment lowering

The direct-fragment backend now checks ordered pointer/scalar roles, matrix instruction,
aligned static dimensions, whole-K traversal and unsupported views once for CUDA and the HIP
candidate. CUDA's existing entry delegates to it. The HIP candidate consumes a verified
MFMA 16×16×16 / wave64 body and the same coordinate-free Float store region, including
scaling/ReLU. It does not copy a rocWMMA GEMM: only fragment load, multiply-accumulate and
store operations use the pinned library's target spelling. Prefetch remains a compiler hint,
not an asynchronous-copy or completion guarantee.

Mandatory HIP CI compiles that generated body for gfx90a, requires the expected MFMA in its
disassembly and rejects gfx1100 compilation. The header archive is pinned by revision and
SHA-256. The scalar RDNA3 gate stays separate. Local validation can use the same script and
archive without downloading or accessing a GPU.

This remains deliberately source-only. The common matrix-target boundary still declines HIP:
the current artifact compilation contract admits OpenCL language/extension requirements, not
external C++ header dependencies. Before production admission, represent the resolved header
and compiler/architecture identity in artifact compilation and caching, prove physical pointer
alignment/launch requirements, and preserve those through binding. Then require public-route
acceptance and AMD numerical validation. No throughput or CUDA/AMD numerical parity claim is
made from source compilation or disassembly.

### Workload-first continuation: public GEMM and targeted schedule work

The chosen next emphasis is public performance verticals (option B), with schedule/IR work only
where those workloads expose a need (targeted C). HIP production admission and PTX breadth do not
displace this work. The bounded composed-versus-explicit GEMM probe in the comparison protocol
keeps source/ABI evidence, dispatch declines, poisoned-output checks and rotating raw replay samples.
It labels host-synchronized timing explicitly and optionally uses existing descriptor-backed graph
device events. The earlier profiling blocker applied to `PreparedParallelProgram`, not this probe.
Observed replay events are retained separately from available dispatch alternatives.

The first Arc run exposed a real admission bug: the public descriptor's `:ocl` backend was excluded
from the existing mixed-DPAS schedule's `:opencl`/`:ze` check. Accepting the alias restores matrix
candidates under unchanged instruction, subgroup, precision and binding requirements. Focused
tests retain non-DPAS, CUDA/HIP and unsupported-wave declines. Before/after numerical checks pass;
both short timing runs are nonstationary, so no speedup, regression or tuning winner is asserted.

Next priorities are dynamic GEMM→activation fusion through symbolic extent equivalence and legal
scalar placement; correlating observed events with executable evidence; then the projection shape
ladder and a resident forward/VJP/update. The dynamic extent's checked multiplication cannot simply
be moved across an output write: fusion must preserve failure behavior as well as values.
The first dynamic slice handles an extent product already computed before the contraction. Shared
typed index lowering checks retained widths, and shared axis algebra establishes shape equality;
the fused result transform retains the extent as a dependency. This permits ordinary prebound
contraction→map source to reach one generated step without an explicit epilogue API. The original
post-contraction extent remains a barrier; no speculative motion or inferred range is introduced.
The same comparison probe now retains separate prebound-source identities for single-row and
multi-row projection observations. Both reach one generated XMX replay entry with exact output;
external baselines and stationary measurements remain required before a competitiveness claim.
The next measured boundary alternates activations while retaining constant weights. It validates
fresh results but reveals a separate generated A-conversion launch before each XMX contraction,
with substantial inter-kernel event-span gaps on OpenCL. A graph with one semantic contraction is
not automatically one physical launch. Investigate explicit typed tile-local representation
conversion and replay submission costs; do not relax TileLoad's dtype equality or conceal an
unrepresented conversion in a target emitter. This is a workload-driven extension of the existing
KernelBody/graph pipeline, not a new GEMM semantic API.

The same changing-activation workload also passes on Arc Level Zero. Its measured event spans
are much closer to kernel-duration sums than OpenCL's; both runs are nonstationary and retained
in the comparison protocol. This separates two investigations: runtime replay gaps and explicit
tile-input conversion. Do not attribute the gap to arithmetic or replace the current graph without
measurement. A tile-local conversion candidate must preserve RNE/IEEE semantics, masked zeros,
physical fragment layout, and public FP32 inputs; CUDA/HIP staging must be represented and charged
in the schedule, not silently invented by an emitter. Keep the existing global conversion as a
candidate because conversion/reloads repeated across output-column tiles can be more expensive.
An opt-in Arc oracle now validates the exact Intel 8r16 A-load mapping: lane `l`/component `r`
holds `[r,l]`, allowing eight FP32 loads with explicit RNE half conversion to construct `short8`
without shared staging for that shape. All 2,048 tagged-coordinate/rounding checks pass; test-only
source and reference identities are retained in the comparison protocol. This is a prerequisite,
not a production conversion route or performance admission. Next introduce a pure typed input
value region, preserve it in the structural matrix plan, lower only the proved mapping, and
test every other emitter's explicit decline before enabling a measured graph-fusion candidate.
The first contract slice adds `TileLoad.value-region` using the existing `ScalarSSARegion`, not
a new numerical dialect. It accepts only closed, pure, one-element regions, checks source/storage
and result/fragment dtypes, and runs shared SSA typing with no external initial values. Existing
loads retain nil regions and strict storage/fragment equality. Matrix targets currently reject
every nonnil input region explicitly: this contract alone does not enable conversion fusion.
Store-epilogue collection now excludes load regions. Production enablement must lower the
matching FP32 prefetch, retain Intel byte-pitch/extent constraints, and validate masked zeros.

The narrow Intel lowering now accepts an explicit lhs FP32→FP16 nearest-even/IEEE cast region
from a scheduled matrix body. Shared scalar lowering spells the conversion, while the tested
lane/component mapping packs the fragment. Warm-up and steady prefetches both use FP32 storage;
the physical contract caps K for four-byte surface widths. Row-tail loads are guarded and fill
zero. RHS transformations, views, partitioned K, and non-Intel targets remain explicit declines.
A tiny generated `[13 32 32]` Arc kernel matches all 416 outputs exactly with no temporary buffer.
This is target-lowering validation, not automatic public GEMM graph fusion or a speed claim.
Next: broaden numerical/schedule cases, add the eligible conversion→matrix graph rewrite as a
candidate preserving ABI/effects, then compare changing-activation workloads against the existing
global conversion strategy with stationary measurements and external baselines.

An additional host boundary probe exposed JVM AOT overflow debt: `compile-aot` of the prebound
canary with `m=Long/MAX_VALUE, n=2, k=0` previously returned an unchanged sentinel buffer instead
of throwing on the source long product. The bytecode emitter used `lmul`; the resident descriptor
binder evaluates scalar lets through Clojure and checked launch algebra. The focused follow-up
restores checked long add/subtract/multiply/negate/inc/dec with Math exact operations, preserving
the separate explicit unchecked emitters. Direct source-versus-bytecode boundary tests and the
public sentinel test retain the distinction. Int arithmetic, narrowing casts, and other targets
are not certified by this fix; backend-wide long arithmetic may pay overflow-check costs, so
cross-target semantic parity and performance remain separate acceptance work.
Shared-memory pipelines or indexed matrix epilogues should be introduced when those measurements
or workload requirements justify them. Existing scientific/distributed acceptance gates remain.

The input-conversion follow-up now connects that leaf to the ordinary typed contraction vertical:
the checked private-cast rewrite preserves the semantic source, scheduled graph and per-node body
certificates; eligible Intel NN/NT dispatches enumerate the candidate without changing the analytic
selector. Concrete graph dependencies and public/node ABI alias contracts govern admission before
graph-private allocation. Automatic or cached preferences may fall back only to the declared
default; explicit incompatible requests fail. Offline tuning retains source-bound alias rejection
rows without executing or timing them, excludes them from winners, and requires an applicable
default. This is alias admission, not a claim of complete scalar/alignment/target applicability.

The public prebound-extent GEMM→ReLU canary exercises lowering, schedule-based recompilation,
LinkPlan instantiation, constant B and changing A through this route. On the laptop Level Zero
device, two tiny `[13 32 32]` replays matched 832 outputs exactly and each profiled one generated
contract kernel; B conversion stayed in the one-time prologue. The non-prebound extent still
preserves its post-write checked-arithmetic barrier. These are correctness and replay-topology
results, not stationary throughput or a comparison with an external implementation. Next measure
the public route across the shape ladder against numerically matched external baselines, retain
the global-conversion alternative, and continue the resident forward/VJP/update and scientific
demonstrators. No source-form restriction should become a permanent semantic primitive merely
to make a benchmark fuse; safe scalar placement needs a proof or guarded schedule.

Training coverage now checks analytic dispatch selection rather than copying hardware pitch
predicates. The tiny Gemma LoRA forward/reverse-AD/SGD fixture uses dimensions admitting the
current matrix schedules: all 56 matrix choices select mixed precision across NN/NT/TN. On Arc,
the focused 25-update comparison passed 31 assertions; loss went from 2.281172 to 1.261994 for
FP32 and 1.262058 for mixed precision, within the existing per-step relative tolerance. This
validates the tiny block and selected preference, not a bound-route trace, stationary throughput,
or real-weight model convergence. Concrete alias admission remains a separate runtime check.

Resident binding now retains a compact admitted-executable report: strategy, precision, entry
points and admission attempt reason codes, without retaining live arguments or executable source.
The public compiled/linked `execution-info` query reads that evidence rather than running the
selector again. The alias-fallback fixture distinguishes preferred and bound strategies; the
public Arc fusion canary confirms bound fused selection separately from its measured replay.
Entry points include any prologue and must not be interpreted as replay events. Equation-first
program reporting remains explicitly unsupported; manual bindings without evidence return nil.

The training acceptance also compares the successfully replayed program's bound phase reports
against each selected matrix step, so runtime fallback cannot silently satisfy mixed coverage.
The numerical trajectory and tolerances are unchanged.

External training remains a separate acceptance item. A source audit of finetune-rstr
`9e9ba5d62f3822f056e01c37231d7eaa7c84947c` found that `finetune.train/bind-layer!`,
`forward-pass!` and `backward-pass!` still depend on the retired `gpu/bind-program!` and
`gpu/run-program!` APIs. Its released Raster dependency is 0.2.287, with a sibling override for
development. Migrate shared forward/VJP adapter state and output views onto public compiled
artifacts/LinkPlans before rerunning the two-layer chained-gradient and real-weight stack gates.
Do not restore the old binder or use the tiny Raster block as evidence those external gates pass.
The pretrained-rstr checkout inspected alongside it was `3b13ad42e7bf4c98e348cb779c28096848931ba0`;
no sibling source or dependency was changed by this audit.

2026-09-27 external-source probe, with both sibling checkouts left untouched: the locally cached
pretrained-rstr `origin/main` at `7553e2c9365548ce7593dc87d353198a0cfe475f` was archived into
temporary storage and loaded with Raster main `002abbf902f2ca8acce5bd3560e77aff3d884a51`
as its local dependency. The real descriptor generator's Gemma-3-270m-shaped B=2 layer
(`640×2048`, `:map-void`) compiled for `:ocl:0` to 29 resident `:map-void` stages and zero
compiler allocations, without weights or device allocation. This verifies source acceptance for
that exact external snapshot and the **descriptor** route only; it is not direct equation-first
lowering, numerical replay, or performance evidence. The active pretrained working branch instead
references `gqa-decode-attention-heads!`, presently found only in the dirty Raster root checkout,
so testing that pair against released Raster main fails at namespace load before compilation.
Separately, loading active `finetune.head-gpu` against this Raster main fails at
`finetune/head_gpu.clj:308` because it still refers to removed `gpu/bind-program!`. Next external
acceptance must migrate that adapter to public compiled artifacts/LinkPlans, then run its
real-weight forward/VJP/update gates; do not revive the retired binder to make the import pass.

The existing small `mse ∘ linear-nb` AD/SGD acceptance now executes through the direct equation-first
compile/lower/LinkPlan boundary. Its legacy descriptor compilation assertion remains a compatibility
check, but numerical validation uses two resident updates with CPU parity, per-replay state progress,
and an unchanged host initialization array. Direct lowering reports zero driver allocations; device
allocation starts only at LinkPlan instantiation. This replaces the old numerical descriptor run,
rather than adding another model-sized training loop. It is a reusable compiler capability check,
not external real-model training validation or a performance result.

The retired scatter invocation boundary is removed: its pipeline marker/accumulator metadata,
Level Zero-only resident convention, runtime positional binders and handwritten zero-fill source
had no remaining compiler producer or caller in the source/test/dev/benchmark inventory or the
inspected pretrained/finetune sources. Current scalar/strided scatter uses the existing typed
conflict algebra, scheduled KernelBody and ordinary executable binder. Existing tests remain:
7 typed-scatter tests (55 assertions), plus 3 retained-source/control and OpenCL device tests
(128 assertions), pass with retired vars unmapped from the REPL. In particular, collision updates
preserve nonzero destination contents rather than inheriting the old binder's implicit zero-fill.
This retires an orphan route, not scatter semantics or an independently exercised source oracle.
The retirement includes LinkPlan/ProgramStage certification: old positional descriptors are
rejected for lacking an executable interface, rather than being certified for a missing binder.
All retained descriptor effects use the executable ABI; the dead generic extraction tail and its
emitter-side scalar-type inference import are removed. The focused LinkPlan and ProgramStage
namespaces pass 9 tests/41 assertions, including rejection on both ZE and OpenCL. A follow-up must
validate internal scatter accumulator initialization end to end through ordinary typed fill IR;
do not reintroduce implicit zeroing in a special runtime convention.

## Distributed numerical acceptance checkpoint (2026-09-13)

The first numerical halo acceptance now uses unequal two-row/four-row owned partitions of a
2D heat field, each with explicit ghost rows. It interprets the existing ScheduledHalo source
and target-local rectangles into resident range copies, executes a composed `heat-rhs-2d!` plus
`numeric/axpy!` update through equation-first LinkPlans, and compares four time steps with the
unpartitioned calculation. A no-exchange negative control diverges. Both workers are emulated
on the same laptop GPU/session; this is not multi-host execution, transport overlap, or a bandwidth
benchmark. Intermediate state and halo exchange stay resident; only final numerical verification
downloads results. A hardware-free compile gate covers the composed update independently of
device availability. The composition exposed and fixed ANF result-type loss: existing typed
initializer metadata now survives onto new bindings, including structurally equal rewrites.
No function/type inference registry or census exemption was added.

Local checkpoint continuation is now exercised with actual files and scoped mmap leases. A
certified manifest carries an SHA-256 address and explicit little-endian raw-f64 format; reopening
checks payload length and digest before consuming bytes, and a corruption negative control fails.
The device acceptance advances two compiled steps, downloads directly into the mapped file, closes
the original session and mapping, opens a read-only lease, uploads directly into a fresh session,
closes the lease after synchronous transfer, and advances two more steps to the unpartitioned
reference. Temporary payloads are 448 bytes and deleted in `finally`. This is a local-file provider
fixture, not a production storage adapter, serialized manifest migration test, crash-consistent
publication protocol, asynchronous restore, or distributed checkpoint.
A read-only mapping is not an immutable snapshot against another process writing the backing file;
a production provider must pin an immutable version or hold an appropriate storage snapshot lease.

Still required: bind padded ghost storage explicitly to the global owned-shard contract, realize
the distributed compute/transfer DAG with device-scoped allocation and ownership, then carry the
local checkpoint acceptance through distributed publication and restore. The two-worker acceptance
must not be counted as evidence those cross-entry and multi-device runtime obligations are done.

`BufferView/rectangular-subview` now provides the checked physical region operation used by the
halo acceptance. It preserves dtype, allocation and element strides, proves per-axis containment,
and handles empty regions without extending the backing range. Global/ghost coordinates must be
translated explicitly to the base-local frame; fitting an allocation's byte span is not a proof
of logical rectangle containment. The next binding contract must distinguish owned cells, local
replicas of neighbor cells, and boundary-initialized ghost cells. Merely assigning the padded
buffer to the owned shard would hide the kernel's halo reads and would not prove transfer ordering.

Items 6–8 have checked planning components, but not yet multi-device numerical execution:
`distributed-plan` simulates topology, dependencies, collective/halo schedules and analytic costs;
`numerical-state` certifies chunk coverage and durable field identity; `numerical-content` tests
scoped leases and transfer ownership using fake providers; `amr-plan` composes hierarchy, fields,
regions and routes, but its coarse/fine operator requirements are declarations rather than bodies.
The distributed certificate now retains the optional structural local LinkPlan/ExecutionPlan
contracts instead of only their IDs and operation counts. Equally sized replacements with changed
operations, scalars, event dependencies, outputs or views invalidate the certificate. Persistent
compiler values are shared, not serialized or copied; this is not a content hash or snapshot of
mutable host/device buffers. The next structural binding relates named local entries and global
shards without introducing another ABI or memory representation:

```clojure
{:device-plans
 {:gpu-0
  {:entries {:stencil {:link-plan local-plan}}
   :steps {:advance-0
           {:entry :stencil
            :bindings {:tile-in  {:value :u-now  :shard :tile-0}
                       :tile-out {:value :u-next :shard :tile-0}}}}}}}
```

`distributed-plan/compute-bindings` returns validated entries, logical bindings, actual ABI-derived
accesses and physical leaf views, plus explicit unbound analytical compute steps. Shard identities
are qualified by their global value; exact local shape and the existing AbstractValue storage
contract must match. Public inputs/state/results and unsourced constants cannot disappear from
the binding. Sourced constants and private scratch may remain local. Repeated bindings of the
same resident shard must agree on allocation identity, range and ordered field packing.
Distinct shards cannot overlap physical ranges on one device without an explicit relation;
disjoint subviews remain legal. Declared global memory-space constraints must hold for every
physical leaf. Entry access facts are derived once per report, not once per invocation.
Bound leaves cannot alias private entry storage until an alias-aware access proof exists.
Private-private storage and unused entries remain entry-scoped; only bound shard storage gains
device-scoped identity. Unknown device-local keys are rejected rather than silently ignored.

This is not yet distributed execution: device-scoped allocation sharing, transfer realization,
submission/completion and ownership still need a numerical vertical. Separately instantiated
LinkPlans currently namespace their owned allocations per executable. Matching structural views
does not change that behavior or prove cross-entry zero-copy replay. Whole-shard binding also
does not stand in for a future explicit halo-subregion mapping. Optional flat ExecutionPlans remain
analytical; arbitrary event DAGs cannot be asserted as the realization of a named LinkPlan entry.

The next sequence is workload-driven:

1. Admit the existing `raster.ode.pde/heat-rhs-2d!` through direct TypedSOAC scheduling and emission,
   preserving its multi-store boundaries and offset interior domain. This local admission now
   passes CUDA/HIP compile/link and two-grid ZE numerical parity, through counted-store
   normalization described below. Parallel schedule quality remains unmeasured. Existing 1-D
   RK4 and the direct AD/SGD probe also succeed; the PDE source was not rewritten for admission.
2. Bind distributed compute to exact local program entries and shard/view contracts; make numerical
   program drift load-bearing in certification rather than comparing only IDs and counts.
3. Execute a hardware-free two-shard halo step and compare stitched values with the monolithic
   numerical oracle, alongside the certified region/byte/cost checks.
4. Checkpoint actual addressed bytes through a local immutable file/mmap provider, close/reopen,
   restore, continue, and compare with uninterrupted execution; test branch reuse of unchanged chunks.
5. Grow that seam into conservative 2-D shallow water and typed prolongation/restriction, then
   subcycling/reflux. Existing Dirichlet heat decay tests are not mass conservation, lake-at-rest,
   convergence-rate oracles, or numerical AMR acceptance.

The first numerical evolution oracle now uses an ordinary `deftm` periodic, cell-centred heat
step with full-domain ratio-two prolongation/restriction. It checks volume-weighted mass,
resolution convergence against an analytic eigenmode, and continuation from detached fine-field
bytes against uninterrupted evolution; direct TypedSOAC lowering is checked too. This is a
whole-domain refinement baseline, not partial-patch AMR, halo execution, a durable manifest,
subcycling across levels, or reflux. Device parity is tested only when OpenCL is available.

This keeps cloud hardware, native fabrics and object storage out of the development hot loop.
It does not replace the external training acceptance or measured kernel-comparison work.

The first loop-admission prerequisite fixes an existing unsound dense-map match: a multi-form
`dotimes` body beginning with `do` could select that `do`'s last store and drop its earlier stores
and all sibling forms. Dense matching now requires one accounted-for body; unproved ordered
effects remain intact. The numerical regression checks every destination. This closes an effect
preservation bug, not the broader 2-D heat admission gap; that still needs a complete typed
iteration-domain and cross-iteration write/read proof.

The next counted-domain prerequisite aligns `map-void!` and its compiler CPU expansion with
their documented `dotimes` semantics: evaluate the bound once as Long, execute no iterations for
nonpositive counts, and return nil. Explicit user narrowing casts remain executable. The former
implicit Int coercion could not faithfully represent a source Long-count loop. This changes no
kernel ABI and does not admit GPU launches beyond a target's index limits; unsupported domains
must still fail target checks rather than narrow. Raw store-loop normalization can subsequently
use one typed Long scalar extent `max(0, long(bound))` and the existing effect-region analysis,
with sequential order retained until a dependence proof permits parallel scheduling.

The source normalization now uses that existing effect-map boundary: a closed raw `dotimes`
store region keeps every body form, performs the Long conversion at its original site, and
clamps negative counts through an ordinary typed scalar conditional. Unknown/effectful or
mutable-array-derived bounds remain outside this normalization. No new operation registry or
semantic loop dialect is introduced. An explicit returned destination observes its latest write
even when the loop itself returns nil. Effect result shapes project the physical destination's
extent, and that projection participates in ordinary SSA remapping.

The existing `heat-rhs-2d!` now compiles and links through the direct CUDA/HIP boundary without
driver allocation. A local Level Zero numerical regression compares all cells against the CPU
for 2×3 (empty interior) and 5×7 grids with nonzero initial destination contents. This establishes
local execution correctness, not a competitive parallel stencil schedule, distributed execution,
or a performance result. The next distributed seam remains exact local-program/shard binding,
followed by numerical halo execution and real-byte durable restore.

The counted-loop migration also exposes `softmax-rows!` as an end-to-end scalar-sharing
regression. Pure, destination, and offset maps now project a complete typed lexical `let`
spine into ordered TypedSOAC locals instead of embedding it in one expression for later
substitution. Missing local type evidence declines admission rather than inventing a consumer
type. This preserves each intermediate once through SegMap and KernelBody. On the existing
softmax exponential polynomial, the generated kernel dropped from 25,599 scalar computes and
3,379,887 source bytes to 24 computes and 2,259 bytes. CUDA/HIP source-size ratchets and local
Level Zero numerical parity cover this case; these are not throughput or SOTA benchmark claims.
The shared inliner likewise retains actual argument types and once-only evaluation (including
unused checked conversions), and never flattens initializer bindings into loop/recur parameters.

Coverage accounting distinguishes newly admitted sequential store work from formerly parallel
work becoming serialized. The migration's new sequential regions include host loops previously
outside typed coverage; they do not establish a competitive schedule. The serialization ratchet
remains in force, with deliberate baseline changes requiring an audit of the previous workload
route and preservation of existing independent regions.
The refreshed 236-var baseline admits 23 such regions: 19 previously scalar programs,
previously compatible `softmax-rows!`, and three partially typed programs (`maxpool2d-bwd!`,
`mean-pool`, `dense-into!`) whose old typed kernels covered initialization/copy/final scaling
while the counted loops remained host work. Their existing independent kernels are retained.
The resulting route counts are 132 typed, 47 scalar, 14 compatible, and 43 errors; the 43 errors
remain visible debt rather than being excluded from the corpus.

Rectangular axis permutations now retain derived dimensions as opaque mixed-radix factors while
exposing their product spines to the ownership algebra. `im2col-1d` therefore lowers as a proved
unique traversal rather than an ordered compatibility effect. Its reverse, `col2im-1d`, exercises
the colliding case: walked typed addition is recognized through an identity result cast, and a
branch-local address region is represented as a guarded lexical effect rather than speculated out
of its source branch. That region lowers to ordinary KernelBody `IfRegion` plus `AtomicRMW` on the
shared OpenCL/CUDA/HIP path. A local Arc comparison covers the public equation-first LinkPlan and
the exact CPU result. This is general guarded reducing-scatter support; convolution names do not
participate in admission or lowering.

Fixpoint typedness now distinguishes statement bindings from value bindings through the existing
`:raster.effect/effectful` binder contract. Normalized loops, SOACs and effectful conditionals no
longer require an RHS-head whitelist merely to cross the typedness gate; an unmarked conditional
or parallel call still requires a retained result type and fails closed. The pure-map materializer
now attaches the same effect contract to its generated write step. This only admits statements to
the next conversion boundary: whole-program conditional branches still require explicit typed
program control and may not be mistaken for a scalar expression or a `KernelDispatch` selector.

Guarded lexical effect regions now admit counted store loops. The existing `effect-when` term
remains the only conditional effect scope; its body may contain the existing `effect-loop`, and
portable KernelBody lowering preserves the nesting as `IfRegion` around `ForLoop`. This closes an
obsolete frontend rejection without adding a loop or branch dialect. Integer values are not
implicitly predicates: Clojure treats zero as truthy, unlike C-family targets, so source must use
an explicit comparison until a first-class boolean ABI/value contract is carried end to end.

One-arm source conditionals whose active branch is a removable counted store loop are predicated
into that same effect-map contract. Pure branch-local scalar bindings remain typed effect-region
locals; ownership proof, rather than the rewrite, decides whether the traversal is independent.
A single walker-typed scalar reduction nested under a removable scalar expression is first exposed
as SSA, then handled by the existing reduction-to-scalar fusion rule. Thus gradient clipping lowers
`sqrt(reduce(...))` as a typed reduction result-transform followed by a guarded effect-map, without
a host scalar round trip or an optimizer-specific rule. Resident realization rewrites only actual
consumers of that rank-zero value, preserves effect-map destinations, and carries it physically as
a one-element device buffer. Full resident compilation emits two scheduled GPU stages and no
fallback; this is functional coverage, not a throughput claim.

Resident realization is now the default for non-escaping reduction scalars on every GPU compiler
entry, including diagnostic/staging compilation. Escaping loss values remain host-visible. This
removes the former policy split in which the resident-program API compiled RMSNorm, softmax
backward, and gradient clipping while `show-pipeline` reported contradictory scalar/buffer ABI
roles for the same typed programs.

Ordinary scalar reductions now delegate their complete element expression to the same strict
typed-SSA lowerer as maps, scans, stencils, contractions, and ordered fold-maps. The reduction
adapter retains only its algebra, coordinate, storage, and currently executable overflow gates;
it no longer reparses literals, loads, casts, and arithmetic through a second recursive lowerer.
Authoritative lexical binder types survive the shared pure-let normalization when an initializer
has no result stamp, without overwriting an initializer's own type. Consequently mixed-precision
value conditionals such as public Huber loss lower to KernelBody `IfRegion` plus explicit branch
conversions on OpenCL, CUDA, and HIP, while untyped compound arithmetic and unchecked coordinate
claims still decline.

The shared scalar language now normalizes Clojure's unary numeric predicates to typed comparison
SSA and expands `signum` into comparisons and selects. The expansion preserves signed zero and
returns NaN unchanged instead of relying on target builtins with differing edge semantics.
Compile-time Clojure truthiness is resolved before KernelBody boolean control, including the
literal `:else` arm produced by `cond`; runtime integers are still never treated as predicates.
Public Huber and L1 gradients therefore use the same KernelBody route on OpenCL, CUDA, and HIP
without source fallback or target-specific intrinsic spelling.

Effectful source loops now acquire the fixpoint statement contract only when they have no terminal
value and their enclosing binder is unused; a consumed nil or a value-carrying loop remains subject
to typed-value validation. TypedSOAC composes such an effect-only counted store loop with its later
lane-local continuation through the existing ordered effect-region algebra. This moves the public
GQA decode loop from an early untyped-fixpoint error to the explicit
`sequential-effect-continuation` production boundary. Crossing that boundary still requires a
read/write ownership proof for scratch written by the loop and read by later scalar folds; the
compiler does not claim parallel safety merely because the source came from an attention workload.

That ownership boundary now admits multiple writable destinations when every destination's complete
read/write set has one proved injective outer-item slice. Read-only tensor captures are irrelevant
unless they flow into an address, where the index algebra still declines data-dependent forms.
Dependency closure through typed locals determines which nested sequential loop digits actually
participate in each address; an address may be invariant in a surrounding inner loop without losing
outer-item ownership. Scalar loops embedded directly in effect-store values are canonicalized to
`Fold` before this proof, just like local initializers and carries. KernelGraph binding remains the
physical precondition and rejects overlapping writable views. With these pieces, public GQA decode
proves independent head slices over scratch and output and emits one ordinary KernelBody kernel;
the proof is general mixed-radix ownership, not attention recognition.

## Fresh-storage initialization through the typed vertical

The analogous OpenCL/Level Zero `invoke-registered-reduce-by-key-kernel` shortcuts are also
retired after an exact-symbol and dynamic-dispatch reachability audit. Production reduce-by-key
already enters the typed reducing-scatter conflict contract, scheduled SegMap/KernelBody, and
generic map-void invocation; the public combinator and CPU semantics are unchanged. The test-only
handwritten OpenCL source generator is retired separately: it accepted arbitrary operators but
always emitted addition. Its source assertions now exercise the existing public typed dispatch
test, which also asserts KernelBody emission and no longer skips on namespace-loading failures.
The retained Vulkan scaffold is unchanged; its analogous operator-validation gap must be closed
through the typed conflict contract before any future production activation.

Resident allocation does not itself implement Clojure's fresh-array zero semantics, and allocating
once is not enough for repeated execution. The frontend now retains allocation initialization,
element type, extent, and source order in TypedSOAC facts using the shared allocator descriptor.
GPU scheduling materializes required zeros as ordinary typed maps after fusion and before
ownership certification. Native fresh-array execution keeps its native initialization provider.
Explicit compound allocation lengths use the same scalar SSA normalization as launch extents,
at the original allocation site; the public invocation plan therefore sees a scalar dimension,
not a reconstructed arithmetic expression. Value remapping preserves the allocation contracts.

Overwrite elision uses complete logical result shapes for functional maps, stencils, contractions,
segmented/product reductions, segmented fold-maps, and scans, with no aliased destination read.
The shared extent proof follows retained integral scalar SSA and checked Long products, allowing
equal dimensions with reordered factors without erasing narrowing, floating, or wrapping
arithmetic. Allocation size must equal result volume: padding still requires initialization.
Coverage witnesses are consumed in equation order: only incoming scalars are initially available,
and scalar definitions extend the proof environment after execution. Allocation extent definitions
must precede their allocation; a later definition cannot justify an earlier fill or elision.
Write-only permission alone proves nothing for conditional effect stores or sparse updates.
Allocation cardinality is separate from a destination AbstractValue's logical consumer shape.
For example, an AD gradient may be allocated with `in*out` elements and later consumed over
`alength(weights)`. These remain independent symbolic identities: the initializer covers the
constructor extent, while concrete invocation/LinkPlan validation checks logical demand against
capacity. Known undersized allocations fail statically; neither symbolic equality nor extra
capacity is a license to elide a fill. Plain representation and absent logical layout remain
mandatory for this initialization route.
Unsupported extent, dtype, placement, alias, and observable-host-use contracts fail closed.
General effect-map dense-image proofs remain work; this does not change the matrix selector.
Storage first consumed by retained host bindings keeps the native allocation provider and host
writes; GPU staging uploads that state. Such host buffer accesses still disqualify straight-line
resident extraction. Observations between fused constituents cannot use this exemption.
The native-provider obligation is retained as semantic data and survives value remapping.
Source-independent SOAC promotion and invocation linking reject it until an explicit content
provider exists; an ABI `:write` role cannot silently discharge that obligation.

Validation covers ordinary and strided public scatter, holes, collisions, changed inputs, and
two replays on Arc; both stages are ordinary KernelBody kernels. CUDA/HIP checks cover public
source compilation **and allocation-free LinkPlan lowering**, not emission alone. Focused pure
checks cover full/partial overwrite, copy/unspecified storage, malformed facts, aliases, source
placement, generated names, and value remapping. Existing numerical tests are retained.

Compiler-generated scalar matrix and split-K combine algorithms now enter through semantic
contraction facts, without building and reparsing surface contraction forms. The typed-route
source-reparse guard remains intact; this is independent of initialization scheduling.

Invocation linking also consumes the retained typed functional write domain. A graph output's
`:write` role is insufficient to discard zero initialization: the domain must exactly cover
concrete materialized capacity, have plain layout, and have no graph input sharing its physical
storage token. The current proof admits single-equation algorithms only; multi-equation graphs
need an ordered first-touch proof. Unavailable dimensions and unsupported shape expressions
decline elision, while invalid negative dimensions still fail. This shares the functional-domain
classification with initialization scheduling, rather than introducing a backend operation list.

### 2026-09-14 — ordered product-valued scalar folds

The typed scalar language now retains exact counted recurrences with two or more carries as one
product-valued `Fold`. The source matcher preserves binding order, origins, inclusive/exclusive
bounds and the shared lexical update region without asserting reassociation. Static `nth`
projections are scalar SSA uses of that product region: KernelBody lowering memoizes the region and
emits one multi-carry `ForLoop`, while JVM projection binds the returned tuple once. Focused tests
cover frontend validation, one-loop OpenCL/CUDA/HIP emission, JVM numerical execution, and rejection
of changed exits and non-unit induction. This is general loop-language coverage, not a layer-norm
operation rule; algebra certification and parallel schedules remain separate follow-ups.

### 2026-09-20 — certified segmented fold-map association

The general per-segment fold-then-dense-map algebra now carries association per fold. Source must
explicitly request implementation-defined association; the frontend derives a typed monoid
certificate and the validator checks the serialized certificate against the retained scalar
region. Host projection keeps the request but preserves sequential interpretation. Ordered folds
and ordinary loop recurrences are unchanged. The next schedule may therefore assign a workgroup to
one segment and parallelize only certified reduction axes without recognizing normalization or
another library operation.

The first such schedule is now implemented for a single certified additive or multiplicative
fold. One workgroup owns a segment, lanes traverse the fold axis with a strided private partial,
and a workgroup-memory tree combines those partials before lanes traverse the dense result map.
KernelBody owns the allocation, barriers, scalar SSA and launch geometry, so the identical body
emits through OpenCL, CUDA and HIP. Unsupported fold shapes decline rather than silently becoming
cooperative. The full compiler uses the generic executable ABI for this schedule: unlike the old
map marker, it does not invent one distinguished element-count scalar and therefore preserves the
compiler-selected one-group-per-segment `LaunchSpec`. A public Intel OpenCL device test covers
non-power-of-two and greater-than-workgroup extents; nvcc and hipcc compile the same emitted body in
the hardware-free fixture gate. RMSNorm and quantized dots remain workload acceptance steps, not
special cases in this lowering.

The first workload acceptance expresses float RMSNorm itself as the same general SegFoldMap. The
multi-row operation and its one-row compatibility entry both compile to one cooperative executable
with no compiler allocation; host interpretation remains ordered. Intel Arc device-event
measurement at one row and width 640 observed a 169.8 microsecond ordered median versus a 3.1
microsecond cooperative median. The run was non-stationary and is recorded as directional local
evidence, not a portable throughput claim. Correctness covers widths 1, 17 and 513 plus a three-row
case, and the actual library kernel is part of the nvcc/hipcc fixture corpus. This validates the
algebra/schedule seam on a real decode bottleneck without introducing an RMSNorm compiler rule.

### 2026-09-25 — city language-coverage follow-up (not a current upgrade gate)

The city-rstr GPU day kernels now compile and agree with their JVM counterparts on Raster
0.2.951, including the multi-carry effectful episode loop. On current Raster main at #907,
the external P10 two-exit search and P14 multiple-`recur` probes also compile on Level Zero
and match the JVM exactly; Raster's city fixture independently covers a two-exit search and
branched effectful recurrences. P20f, which nests a seeded SplitMix draw inside a multi-carry
effectful episode walk, still declines with `:no-lowering-rule`. Direct source-region probing
isolated its `unchecked-add-int` induction against a `Long` bound: replacing only that step
with widened addition makes the whole region recognizable, while the other unchecked SplitMix
arithmetic stays unchanged. This is a legality gap, not evidence that SplitMix cannot be emitted;
accepting the int step without a bound proof would risk wraparound. Its full shape now has a
device parity-or-explicit-decline regression. This is not an upgrade gate for city, which
uses supported spellings, nor proof that every P13 variant is admitted. When revisiting the
remaining gap, retain a typed control-flow region and its effect/SSA ownership facts, then
check GPU/JVM results for the full city kernels. Do not reintroduce source-level fallback or
interpret a successful compile as numerical validation.

### 2026-09-25 — runtime-shaped cooperative register tiling

The public typed contraction keeps symbolic `m/n/k` through one semantic ABI while the
register-tiled KernelBody now consumes those scalar bounds directly. Its tile-local loads,
stores, barriers and accumulation are still compiler-generated and target-neutral; there is no
new GEMM source template. The router preserves source `int`/`long` scalar dtypes, widens local
K offsets explicitly when necessary, derives the output count as checked launch IR, and retains
the portable reduction as another executable alternative. Candidate graph composition now
matches logical buffer identities independent of physical A/B argument order. A flat read of a
2-D destination declines the tiled leaf rather than constructing an invalid rank-one load.
The selector sends products beyond int-sized resident capacity to the portable path, whose own
binding still checks capacity, so a Long-valued extent cannot silently overflow tiled address
arithmetic. General symbolic bound expressions remain a documented decline.

The opt-in public Arc oracle passes batch-one, multi-row, square and awkward-tail shapes, and
the generated symbolic body compiles to CUDA PTX and HIP syntax without hardware. The square
shape's local device-event median moved from approximately 393 to 112 microseconds, with
non-stationary early replays; see the raw comparison protocol for the measurement limits.

### 2026-09-29 — whole-plan replay measurement boundary

The opt-in production canary now composes three independently lowered typed effect maps through
the public `Compiled/compose` API and checks one recorded graph, three profiled kernels, and an
independent numerical reference. It reports component compilation, composition, instantiation,
and warm resident device-event replay separately. Ordinary CI tests only the canary contract;
no timing threshold enters the hot loop. One local Arc OpenCL run at width 1024 was numerically
exact, but replay timing was nonstationary (median 93 microseconds, CV 1.34); it cannot establish
a performance improvement. This is a mechanism canary, not a pretrained decoder substitute.

Next, run the same phase accounting on pretrained-rstr's resident decode and chunked prefill,
recording kernel/event attribution and transfer counts without changing its cache policy. The
committed GPU corpus reports 238 typed SOAC, one typed structured-control, six explicit host-only,
and six scalar routes out of 251 functions; it has no compatibility route. That is a coverage
ratchet for the selected corpus, not proof that the compatibility pipeline is dead for external
programs such as city-rstr. Remove compatibility lowering only after a production-path reachability
audit and explicit fail-loud replacements for its remaining admitted forms.

### 2026-10-03 — durable producer/storage authority seam

Items 1–8 remain the campaign, not just the currently active storage slice. Recent producer work
retains original compiler-owned Prepared/Compiled identity, tracks synchronous replay completion,
pins byte receipts and hashes actual resident initialization/output/post-state rather than caller
defaults. Verified raw-array decoding uses explicit dtype/byte order and bounded staging. Provider
availability finalization closes verified local leases and awaits matching durable placements
before invoking external metadata publication; provider/store realization is still separate.

Generated per-dtype KernelBody probes now establish measured storage order on local OpenCL and
Level Zero; all six sources also pass the exact-head CUDA/HIP compile gates (#1001). Owner-bound
measurement shares the artifact seal, lifetime guard, emitter and binder rather than adding a
cache/session convention. A live completed receipt can join only original same-owner/session/dtype
facts, then expose addressed physical leaf bytes with explicit raw-array storage. This is not
logical quantization inference, universal arithmetic/codec equivalence or a durable state commit.

Pivotal review caught loose byte coercion, dishonest optional-capability coverage, repeated ZE
registry-kernel leakage and source-only/explicit-SPIR-V cache aliasing. These have independent
regressions; focused storage/owner/registry/native/CI-ratchet suites pass 36 tests / 767 assertions.
Both local native backends execute actual retained compiler programs, not only emitted-source
tests. No timing claim is made under varying load and power-save settings.

Next acceptance order within item 8:

1. Make shared native graph/prepared destruction report failures and retain unresolved resources
   honestly. Current older destructors swallow some native errors; surfaced API cleanup tests do
   not prove native reclamation. Do not invent a separate checkpoint-only lifetime convention.
2. Bind semantic field/coordinate selection to the addressed producer leaves, stage/verify bounded
   immutable chunks and construct producer-derived manifests before real provider publication.
3. Extend scalable hierarchy/patch packing and temporal interpolation/subcycling/reflux validation.

Canonical external training/package migration, performance/selection evidence, real fabric and
multi-device collectives and production storage/AMR acceptance remain open as listed above. The
local evidence and compile-only vendor gates do not establish those broader end states.

#### Shared native teardown consolidation: acceptance slices

The resource-cleanup mechanism is owner-local runtime state, not compiler IR or a second authority
registry. Ordered destruction dependencies retain failed and blocked resources; successful releases
are removed exactly once. A thrown error defaults to an indeterminate outcome, so another close
reports it without repeating the native call. Only failures explicitly classified retry-safe by
the internal callback contract retry; this ordinary error marker is not authority or evidence.
No native OpenCL/Level Zero error is classified as retry-safe merely because it is nonzero.

Land this consolidation in dependency order, keeping items 1–8 above intact:

1. Validate the small owner-local mechanism with dependency, suppression, concurrency, recursive
   teardown and indeterminate-outcome tests. This alone makes no native reclamation claim.
2. Capture ownership immediately after each backend acquisition, including partial constructors;
   check native release statuses. Retain event arenas until all dependent events are released.
3. Integrate KernelGraph lifecycle under the session lock. Handle identity must include the session
   and generation, avoiding cross-session/key-replacement ABA. Retain provisional/retired owners on
   failed construction or replacement; drain events first; remove ownership only after success.
4. Extend the same ownership to descriptor prepared/recorded graphs and their exact dependencies,
   then root buffers, kernel arenas and LinkedExecutable/parallel-program close. Do not mark a
   wrapper closed before failed teardown resources have an honest retained disposition.

Pivotal review identified constructor rollback, stale graph handles and early wrapper-close flags
as correctness requirements, not optional cleanup. Native dependency ordering is graph recordings
before prepared kernels, kernels before private views/buffers, profiling events before pools and
pointer arenas, and session children before root storage. Independent siblings may continue after
failure only when that independence is actually established. Use of the owner must share the
surrounding session lock with release; a standalone live-status check does not pin resources.

Regression acceptance includes every acquisition/release fault, suppressed cleanup errors retaining
the primary failure, old graphs surviving failed new construction, stale-handle rejection, no double
release, and failed dependent destruction preventing buffer free. Performance acceptance remains
separate from correctness under laptop power-save/background-load conditions.

#### Native KernelCall acquisition and common graph ownership

Both resident backends now reserve a cleanup owner before creating each production KernelCall's
dedicated native kernel, then use checked native destruction. Argument-binding failures roll back
immediately; if destruction is unresolved, an explicit synchronous adoption callback transfers the
original owner into the surrounding session. Suppressed errors are diagnostics, never the lifetime
channel. The raw one-argument backend API preserves unresolved ownership in ExceptionInfo data and
keeps the original failure as its cause; it cannot promise identical top-level exception identity.

Common KernelGraph entries use the shared dependency plan for recordings, individual prepared
kernels, views and temporary buffers. Failed construction retains the same resident root views and
footprints, preventing root release. Successful cleanup is retired incrementally; session close
attempts all independent siblings in a layer, then stops before dependent roots/arena on failure.
Its sticky :releasing lifecycle marks :closed? true to reject further use, without pretending native
disposal succeeded. Repeated close remains available for retained cleanup. Modern bind/run sequences
are serialized by the session lock. Graph handles now carry session and generation; a stale or foreign
handle cannot drain or destroy a replacement's resources. A failed old-generation destruction keeps
that non-runnable owner, and does not publish the newly constructed replacement.

The migrated KernelGraph swallowing destructor is removed; a missing owner fails loudly. Backend
production prepared maps likewise cannot fall through the compatibility destructor if their owner is
missing. A duplicate shadowed ZE fresh-kernel factory is removed. Numeric surface semantics and IR
are unchanged; handle identity, close lifecycle and raw-backend exception handling become stricter.

#### Level Zero recording acquisition ownership

Regular Level Zero recordings reserve one cleanup plan before creating the queue, command list,
profiling pool or timestamp events. A create/readback exception is indeterminate, not proof that
no native allocation occurred: it retains the acquisition slot without guessing a handle to free.
Successful acquisitions are private callback captures, not mutable graph-map fields. Command-list
release precedes event release; all events precede pool release. An unsubmitted constructor's queue
is independent and is still attempted after another resource fails. Never-created dependent slots
may remain blocked plan entries; this does not assert those resources were allocated.

Recording rollback uses the same explicit adoption sink as KernelCall construction. Prepared kernels
also depend on adopted construction debts, preventing destruction while an unresolved recording may
still reference them. Fault tests cover every create/append/close point, independent destruction,
unknown-outcome exact-once behavior, and an append failure combined with failed list destruction.
Live profiled/unprofiled graph replay remains numerically checked on both local backends. Shadowed
older Level Zero record/replay/destroy definitions are removed. Direct backend users must establish
submission completion before destroying a recording; common session teardown drains tracked events.

The Level Zero slice alone does not close OpenCL per-replay ownership, descriptor/root-owner
consolidation or establish performance under varying laptop power/background load.

#### OpenCL recording and per-replay ownership

OpenCL recordings reserve their profiling queue and one retained submission dependency. Each replay,
ordinary or profiled, installs a nested owner before enqueue. Its dependency plan proves queue
completion before event release, and releases every event before closing the pointer arena. A
successful wait/query records completion so subsequent cleanup does not add an unnecessary finish.
If enqueue/flush fails before a public completion token exists, the recording still owns the partial
submission. An unknown drain blocks every dependent resource. Independent event releases may proceed
after successful drain; failed event release retains its arena and profiling queue, and the common
recording dependency keeps prepared kernels and resident roots alive. No unknown native operation is
automatically retried.

Timestamp reads validate the nested owner as well as the recording: a retained vector must not
permit queries on events already released by a partially failed cleanup. Nonempty graph tokens
cannot fall back to legacy release if their owner is missing. Transfers and empty-completion tokens
retain their existing contracts. Timestamp/await errors stay primary when cleanup also fails.
Raw backend callers still must serialize graph use/destruction; common sessions use their lock.

Hardware-free production fault tests cover partial enqueue, flush, unknown drain, queue creation,
independent event release, retained arenas, missing-owner tokens and timestamp access after partial
destruction. A common-session regression proves failed submission without a public event cannot free
the graph's prepared kernels or roots. Live profiled/unprofiled replay and storage/transfer oracles
remain separate checks. Ownership adds replay bookkeeping; performance acceptance remains pending
stable power/load measurements, not inferred from these correctness tests.

This is not complete native reclamation. Legacy descriptor bindings, backend view/event
partial constructors and destruction, asynchronous-event failure cleanup, root-buffer/arena failure
retention, and linked-wrapper close remain acceptance work. In particular native graph destructors
still contain swallowed-error debt, and general legacy concurrent operations are not certified by
the modern graph locking tests. No native error is automatically classified retry-safe.

#### Recorded wrappers and borrowed source retention

The common recorded wrapper reserves prologue and replay ownership before construction, adopts
unresolved backend acquisition debt synchronously, and publishes replacement recordings only after
old-generation teardown succeeds. Failed teardown keeps the registration non-live; independent
successful releases are not repeated. Failed provisional construction retains a private registration
when rollback cannot finish, preserving the exact primary exception and its borrowed sources.

Prepared phases and emitted graph handles cannot be released or rebound while any recorded wrapper
borrows them. Recording, replay, profiling and complete measurement serialize under the session
lock. The context-chain convenience API releases its previous recording before rebinding phases.
Attached cleanup stops before source teardown if recording teardown fails, and before root teardown
if any prepared source fails.

Surface lifecycle tightening: `free-buffer!` rejects releases overlapping a recorded wrapper's
retained source footprint, including failed provisional wrappers. Low-level preparation captures
resolved root objects and their registrations; descriptor materialization captures every resolved
leaf root, including views and composite arguments. A wrapper unions footprints only when every
source has one. Legacy/manual bindings without a footprint retain the conservative **all-buffer**
guard; missing is never interpreted as empty. Owned-root release checks keys, allocation identities
and exact resident-buffer object identity, including aliases registered after binding. Distinct
wrappers of the same native pointer require a shared allocation identity; no arbitrary pointer
equivalence is inferred. Detaching a new
borrowed/external alias checks its registration key because it cannot destroy the native root.
Unrelated staging buffers may therefore be released without special-case key exemptions.
Explicitly release recordings before freeing their buffers or replacing bindings; this does
not change numerical surface semantics. Backend prepared factories, sticky linked/parallel close,
and root/view/event owners remain unfinished; these wrapper checks do not certify their reclamation.

Fault oracles cover multiple recordings borrowing one source, unknown teardown with successful
independent sibling release, adopted constructor debt, failed replacement, and failed prologue replay.
These are lifecycle/correctness checks, not performance or distributed-ownership evidence.

#### Raw layout-binder retirement

The runtime-only convert/transpose binders and their separate Level Zero module/kernel caches
have no production callers. Their three native test consumers now construct target-neutral layout
bodies, project checked ScheduledKernelBody artifacts through the common target emitter, and bind
them through public KernelCall/session APIs. Byte-granularity transpose and all 36 nearest-even
FP16 conversion cases (ragged tails and extents below the unroll width) remain numerical oracles;
the int8 contraction route still checks its inserted transpose against the CPU reference.

The raw binders and OpenCL rejection stubs are removed rather than assigned another ownership
system. Tests release their modern graph bindings; migrated transpose/conversion storage follows
the session lifetime. This removes a duplicate acquisition/cache path, not an emitter capability.
Other legacy routing-test allocation/launch helpers and the raw map preparation binder remain
separate consolidation work. No performance or cross-vendor execution claim follows from this
Intel-native migration; hardware-free target compilation remains its own gate.

#### Plain prepared KernelCall ownership

The public `prepare!` convenience API now projects its split pointer/scalar/bound arguments into
the checked artifact's positional KernelCall ABI and full 1–3D launch geometry. It no longer uses
the raw map preparation binders, which are removed from both runtimes. Argument names are not
assumed unique: distinct physical positions survive projection, and scalar conversion uses the
shared runtime scalar contract. This tightens admission to checked artifacts without changing
the high-level argument convention or numerical surface semantics.

Plain prepared bindings require a live cleanup owner before invocation or recording. Replacement
builds the candidate before releasing the old generation, publishes only after successful release,
and retains failed old registrations as non-live. Failed construction adopts unresolved debt into
a private prepared registration, preserving the primary exception and pinning its resident roots.
Unrecorded prepared bindings now pin their captured root footprint too. Invocation serializes with
release under the session lock. Level Zero async KernelCall teardown drains the exact captured
command list before destroying its kernel; this is not a new completion-event API.

This slice does **not** certify all reclamation: legacy multi-child BoundExecutableStep teardown,
backend pre-reserved acquisition, modern graph alias guards, root/view/event ownership and sticky
linked/parallel close remain explicit follow-ups. Unknown cleanup outcomes are never retried just
because the public entry remains registered. Fault tests cover missing owners, positional duplicate
names, 3D launch projection, failed acquisition debt, replacement failure and launch/release ordering.
Quantized replay tests observe actual intermediate uploads rather than conflating an allocation's
one-time initialization source with hot-path transfers; CPU/device poison and replay oracles remain.

#### Composite prepared-step ownership

Descriptor-selected KernelArtifact and KernelGraph bindings now use the same checked composite
resource owner as modern graph bindings. BoundExecutableStep is only the ordered-child container,
not a teardown exception. Admission, recording and destruction require its live outer owner.
The former per-child swallowed destruction/free errors and publish-before-old-destruction ordering
are removed. A single outer construction accumulator receives every returned child binding,
backend-adopted debt, graph-private temporary and owned descriptor sub-buffer. Rollback reuses a
completed candidate's exact owner rather than wrapping its children in a second ownership plan.

Descriptor-private temporary and owned OpenCL sub-buffer constructors reserve acquisition slots
before native contact. Successful acquisition owners move atomically from conservative debt to
their post-kernel dependency layer. A throwing constructor remains indeterminate; no retry or
“nothing allocated” inference is allowed. Failed construction and failed replacement remain hidden
prepared registrations when cleanup cannot finish. Root footprints pin borrowed session storage,
and session close stops before root teardown if any prepared owner remains unresolved.

The dependency order is adopted debt → kernels → owned views → temporaries. Descriptor-owned view
owner/buffer vectors form matching ordered prefixes, with only a possible owner-only final tail if
bookkeeping fails; other graph paths provide no private view owners. Do not introduce a mixed raw
legacy prefix and migrated-owner suffix into this representation. Constructor tests cover first
and second allocation/slice failures, root/session retention, successful multi-node teardown, and
a later child-bind failure with unknown earlier child destruction. Conservative debt may retain
otherwise independent private storage; it does not justify freeing possibly referenced storage.

Scope remains explicit: public/root allocation constructors, modern graph external materialization,
global registries, root/view/event owners and sticky linked/parallel close are separate follow-ups.
This closes the descriptor-specific lifetime exception, not the entire memory-management campaign.

#### Retained linked/parallel close

Completed prepared programs, program sequences, direct prepared graphs and LinkedExecutables
retain the common cleanup owner across close attempts. Marking a value unusable no longer hides
failed cleanup on later calls. Successful sibling releases are not repeated; only failures
explicitly declaring retry safety are retried. Unknown outcomes preserve their original exception
and ownership. Attached linked teardown is recording → independent phase/program siblings → all
dependent allocation registrations. An owned linked executable has one session-close resource,
not a second competing resource plan for the same native children.

Prepared run/profile/report operations serialize with release, and active-use depth rejects
same-thread reentrant destruction before the closed flag changes. Linked execution scopes use
the same principle under their existing lifetime lock; live operations require a live cleanup
owner. Private, non-watchable active-use counters are installed before observable atom mutations
and balanced even when a watch throws; they do not alter numerical execution-state projections.
Returned raw handle reports remain **borrowed**, not lifetime leases: keeping an owner alive or
using callback-under-lock run/profile APIs remains the caller's responsibility.

Compiled close now performs pure close admission under the linked lifetime lock, invalidates
projected output wrappers, and always delegates to retained linked cleanup. It no longer mistakes
teardown for an active numerical execution scope or silently skips cleanup when the unusable flag
is already set. Output leases, active callbacks and missing owners decline before invalidation or
close-state mutation. This intentionally changes failed-close behavior from a later silent no-op
to a repeated report of retained ownership; numerical surface semantics are unchanged.

Fault oracles cover suppression and independent sibling attempts, nested sequences, explicitly
retry-safe cleanup, unknown outcomes without retry, concurrent use/release, reentrant callbacks,
owned/attached link dependency order, lease preflight and compiled output invalidation. Hand-built
host test fixtures share an ownership-complete helper; it does not validate plans or manufacture
native evidence. Native linked composition remains an affected numerical oracle.

Generic program/sequence staging now reserves the common
cleanup plan before binding and retains failed rollback via executor `:adopt-cleanup!` or the
existing `::resource-cleanup/unresolved` exception contract. Only successfully returned handles
enter that plan; an executor must retain indeterminate native acquisition that throws before
returning. Direct-graph sequence executors are admitted before acquisition, and their handle
slots/owners are reserved before binding. Completed values retain the exact construction owner.

Link program cleanup debt is adopted into the existing session prepared layer with a conservative
registered-root footprint. Original and late alias roots remain pinned. Session teardown treats
prepared programs and kernel graphs as distinct dependency layers, so parent failure cannot
trigger a second child retry in the same close. An adopted program owner contains binding
destruction only: session root allocations stay session-owned and are not freed twice against
the session's close snapshot. Owned Link construction reserves one session owner before target
and external-binding admission; unresolved close keeps that owner (and session) reachable on
the ordinary unresolved-cleanup exception. Successful construction retains that same owner.

Fault oracles cover independent siblings, sticky child failures, direct-graph wrapper failure,
owned admission/close failure and a real typed equation-program Link binding failure with both
attached and owned sessions. The last uses production lowering/validation/staging/views/adoption
and substitutes only native allocation/upload/bind/release. These checks do not certify all
public/root/view/event acquisition, output-value destruction, provider-backed durability,
distributed fabric execution or external model training, which remain explicit follow-ups.

The canonical backend-buffer ownership slice is implemented; final CI gates remain pending.
OpenCL/Level Zero root constructors
reserve a canonical backend buffer owner and can publish it through `:retain-owner!` before
native contact. Unresolved rollback uses `:adopt-cleanup!` (or the retention callback); containers
deduplicate those notifications by owner identity. Array initialization rolls back through that
same owner. OpenCL sub-buffers own a separate native reference; root and slice staging has
per-buffer shared arenas independent of the global runtime arena and parent staging segment.
Level Zero pointer slices remain non-owning and retain the root lifetime reference. Level Zero
captures the pointer before reinterpretation/record setup and destroys with its allocation
context; a retired context is rejected before native free. Transfer/binding admission checks
owner liveness and context identity. These are not concurrent-use leases or automatic reset
recovery. Slice extent arithmetic is checked for overflow.

Modern graph-private allocation/view construction now retains that exact backend owner rather
than creating a second native acquisition/destruction authority. Public session allocation
retains exact root owners before contact, publishes values/contracts after initialization, and
rolls back through those owners on any Throwable. Unknown outcomes remain session-owned and
make the session non-live. Close attempts independent root siblings and removes successful roots
incrementally, before kernel-arena teardown. Root lifecycle operations reject reentrant close,
free, registration and ordinary session use from synchronous atom watches.

Graph/call external materialization uses the same construction accumulator and canonical owners
as modern graph-private staging. The old temporary allocator, successful-prefix-only rollback,
raw session-root free and swallowed view-rollback helpers are removed. Fault doubles share one
ownership-complete constructor, not production inference from a partially returned buffer.

Synchronous session transfers hold the session monitor through resolution, validation and
execution. A scoped-use guard prevents a callback from reentrantly destroying roots. Common
`resource-cleanup/lifetime-owner` distinguishes destruction authority from borrowed lifetime
identity; registration validates the backend buffer, and free compares exact owner identity as
well as registration/object/allocation identity. A registered Level Zero slice or copied root
cannot hide the native alias behind a different allocation ID. Published root/table owner
mismatch declines before native free.

Asynchronous transfer submission uses the same scoped-use guard through event publication,
including retained host resources. Binding replacement uses one shared publication transaction
for prepare, step, graph and call: after successful old-generation teardown, a rejected or
throwing publication removes the exact retired/candidate registration before candidate rollback.
Successful publication also revalidates candidate identity, so a watch that releases the new
binding cannot return an already-dead handle. Replacement publication failure may leave the key
unbound; it cannot restore a destroyed old generation. A failed old-generation destruction still
retains that generation's cleanup debt. General atom validators that reject recovery mutations
can prevent deregistration; the primary error retains that recovery failure as a suppressed error.

The combined affected ownership, graph, transfer, Link composition, program-memory and adjacent
ABI/binding/native-recording checks pass (160 tests, 1572 assertions) in the existing REPL.
Oracles cover actual backend API calls after free/context
retirement, staging independence, setup/upload rollback, uncertain creation/free, independent
siblings, repeated close, reentrant/throwing watches and blocked transfer versus concurrent
free/close. Linked numerical replays retain their CPU parity anchors.

Surface lifetime behavior is intentionally stricter: directly fabricated backend records have
no native destruction authority; borrowed aliases must be detached before freeing their owned
root. `deftm` numerical semantics are unchanged. Raw backend callers, opaque native segments and
returned NIO views still require external lifetime/concurrency discipline; these checks do not
grant a lease or make concurrent runtime reset safe. Kernel-arena destruction itself still needs
retained outcome ownership; this slice only prevents an arena-close failure from repeating
successful root frees. Final review/CI remain pending. No performance claim follows from these
correctness tests, and the full distributed/training campaign remains incomplete.

Review-driven regressions cover retained-event publication with reentrant close, exact registry
removal on throwing watches/validator rejection, preserved unrelated generations and failed
old-generation destruction, plus public prepare rollback after a watch releases its candidate.
The next native lifetime consolidation is kernel-arena teardown: both backend arena closers still
swallow native release errors, and Level Zero scans arbitrary MemorySegment metadata rather than
declared owned resources. Replace these with retained outcome owners before claiming complete
native lifetime coverage; do not infer destructor authority from a value's representation.

The first #1013 full CI run exposed four remaining legacy fixture failures in dispatch, GEMM
Link topology and recorded-root ordering. Their mock constructors now use the shared canonical
acquisition/retention fixture, and the owned staging root carries its exact owner in the session.
All 45 tests / 256 assertions in those three namespaces pass in the existing REPL. No production
ownership check was weakened; the rerun must still pass all seven exact-head CI gates.

#### Kernel-arena ownership: next coherent vertical

The entry-generation publication transaction moves from session-private code into the existing
`resource-cleanup` namespace, with a path-based registry interface. Prepared steps, graphs and
calls use it directly; no adapter or second transaction remains. It checks the exact old entry
after teardown and inside the publication CAS, and checks the exact candidate after atom watches
return. An unrelated generation installed during teardown/publication is never overwritten or
removed. Root and nested registry paths share the same recovery rule. This prerequisite passes
106 tests / 1019 assertions across the affected publication, ownership and topology namespaces.
Admission rejects nil candidates and the exact old generation before any teardown; recovery
deduplicates secondary error identities. Publishing metadata for an identical registration must
preserve its lifetime explicitly, not invoke the retiring replacement transaction.

The rest of this vertical remains implementation work, not an achieved ownership claim:

1. Reserve one owner at registration, before lazy native contact. OpenCL owns its cached kernel,
   program and per-entry shared staging arena; Level Zero owns its fresh base kernel and declared
   cached staging buffers. Destruction orders kernel before program/staging. Dedicated bound
   KernelCall handles remain separately owned.
2. Acquire/load through reserved slots; retain unknown load/rollback outcomes in the exact
   registration. Publish only into that generation. Identical registration preserves the exact
   owner, handles and staging; incompatible replacement uses the shared transaction.
3. Level Zero `ensure-seg` uses canonical byte-buffer constructors/owners, not naked allocation
   and free. Growth publishes an acquired candidate only after safe old teardown, preserving or
   adopting every failure. OpenCL staging is per-entry, not runtime-global.
4. Close independent entries, remove only successfully released exact generations, and retain
   failed/blocked entries. Remove arbitrary MemorySegment scans and representation-based cache
   preservation. Audit/remove the unused ZE cached-kernel/invoke pair and duplicate registry
   accessor only after confirming there are no consumers.
5. Prove acquisition/build/release faults, staging growth, shared-module arena isolation, repeated
   close and registration replacement, plus a real-device close/re-register smoke check.

Level Zero modules are content-shared global-cache resources, *borrowed* by registrations. They
must not be freed by an arena. Their canonical cache ownership and shutdown-before-context order
are the immediately following runtime-root slice; the arena slice must not claim reset recovery
or silently drop those debts. These two slices precede the still-open training/model and durable
distributed acceptance items rather than replacing them.

The unused ZE `create-kernel`/`invoke-kernel` pair and its global mutable-kernel cache are removed,
along with the duplicate registry accessor and stale marker documentation. Source/test/dev/bench
search plus read-only audits of pretrained-rstr, finetune-rstr, umap-rstr, evoc-rstr, city and
spindel found no consumers of those two Raster functions. Registered ABI invocation, fresh bound
kernels and the shared module cache remain. The observable change is removal of those legacy
raw APIs; high-level `deftm`/prepared/Link semantics are unchanged. Adjacent emission and native
binding checks pass (32 tests / 386 assertions), with the removed vars unmapped in the REPL.
The old arena and global-module shutdown implementations still require the ownership migration
above; dead-cache removal alone is not that proof.

#### Registered native generations: implementation checkpoint (2026-10-03)

The shared publication prerequisite is merged as #1014. Native arena ownership is still WIP,
not released or accepted as complete. Registrations now reserve canonical cleanup before lazy
loading. OpenCL owns kernel, program and a per-registration shared staging Arena; Level Zero
owns its base kernel and canonical byte-buffer children while borrowing cached modules.
Staging caches (including ZE host short arrays) live only in registration-owned state, not in
compiler metadata. Identical compiler registration retains that exact state; importing native
handles or cleanup authority through registration metadata is rejected before contact.

Pivotal review identified and prompted tests for reentrant acquisition teardown, lost-generation
cleanup retention, forged metadata and watch reinsertion of released entries. A shared native-use
scope rejects same-thread public register/close callbacks before they can retire an acquiring
slot. If a registry watch loses a generation whose rollback is unresolved, its exact owner and
arena remain reachable through a hidden failed-registration entry. No uncertain native release
is retried. Independent arena entries still close even if one retains failure debt.

Persistent-REPL checks: 26 registration/load/arena tests with 228 assertions, plus 48 cleanup and
native-buffer/kernel tests with 654 assertions, all passing. New regressions cover exact-parent
loss on internal cache/array watches and same-owner refresh, including successful child rollback,
and reentrant register/close during destruction/publication. Pruning only proven-released ZE
children bounds historical owner retention; its final parent-generation postcheck has its own
remove/replace watch regression. The pivotal reviewer approved this scoped semantic change.

Actual OpenCL and Level Zero registration/load/staging/close/re-register smoke checks passed on
the local device. Two OpenCL mixed-storage invocation/identical-refresh tests also pass (six
assertions). These are ownership checks, not performance or reset-recovery claims. All seven
exact-head CI gates remain required before merge.
Global cached-module/runtime shutdown ownership remains the immediately following slice; neither
this checkpoint nor green ownership tests close the training/distributed campaign requirements.

The first OpenCL CPU CI run exposed a production boundary leak: the resident compiler extracts
artifacts via `kernel-registry-entry`, which was returning enriched runtime registrations. Those
descriptors imported the new owner/registration fields and were correctly rejected on binding.
The fix preserves the original admitted compiler artifact before runtime enrichment (and the
cloned explicit ZE payload), separately from the owning registration. Public registry reads now
return only that compiler artifact, never arena identity, loaded handles or cleanup authority.
Backend loading/binding stays on private runtime entries. Caller-supplied `:arena-id` is rejected;
arena selection belongs to the explicit registration argument/dynamic scope. Native-cache tests
inspect private state to verify handle reuse rather than depending on this compiler API leaking
handles. High-level compiler/Link semantics do not change. This is a boundary correction, not a
relaxation of ownership admission. Both-backend resident block-transfer and OpenCL compiled
composition/mixed-storage public compilation regressions pass locally after the correction; the
full exact-head CI gates must be rerun.

The next runtime-root survey confirms that arena ownership alone cannot justify safe shutdown.
ZE still caches raw module handles, can race module/async-list creation, and never destroys its
global immediate lists/context during shutdown. OCL reset still drops native authority. Root
teardown must not scan private session children or force-close live consumers. Landing order:

1. Canonical exact-generation ZE module cache, input-byte snapshots, acquisition/publication
   rollback and checked destruction; keep modules borrowed by kernel registrations. Migrate raw
   module/fresh-kernel benchmark consumers to existing Artifact/KernelCall execution boundaries.
2. Reserve retained backend root construction owners before context/queue/list contact. Unknown
   initialization outcome stays reachable; independent siblings close but context/Arena wait.
3. Generation-qualified session/resource leases and balanced synchronous-use tokens. A live or
   failed child pins the root. Shutdown preflight must decline without mutation/native contact
   while leases remain, following session -> registry -> state -> child lock order.
4. Cover or retire direct ownerless allocations/SoA/display/scan helpers, and adopt unreturned
   OpenCL async-transfer cleanup debt. Do not infer ownership from exposed MemorySegments.
5. Only after those gates replace shutdown/reset with retained root release and new-generation
   initialization. Native fault, lease, concurrency and differential workload tests precede use.

These are the runtime prerequisites of the existing training/distributed campaign, not a new
planner or a substitute completion target. The module-cache slice does not enable eviction or
claim safe global reset while outstanding kernel/buffer/event borrowers remain.

### Module-cache ownership continuation (2026-10-03, merged)

PR #1015 passed all seven exact-head gates and merged as `35a3a1af`. PR #1016 also
passed all seven exact-head gates and merged as `6b872966`. It replaces raw ZE module-cache handles with exact owning
cache references and gives each fresh kernel a `kernel -> module-borrow` cleanup
dependency. A copied/stale handle is not authority. Payload hashing and native
compilation consume the same snapshot; reservation/publication rollback uses the
common cleanup transactions, and hidden failure debt retains the content key.
Live or indeterminate kernel borrowers make module-cache teardown decline before
any module destruction. Checked destruction retains failure debt without retrying
unknown native outcomes. Shutdown admission holds registry -> state -> module-cache.

Low-level API change: `load-module!` returns a cache reference, and
`create-kernel-fresh` returns an owned kernel value. `destroy-kernel!` accepts that
owned value, not a raw pointer. Compiler/session/KernelCall callers retain their
existing public API. Both production kernel constructors are migrated. The two
low-level reference benchmarks share a construction/rollback helper rather than
duplicating success-only teardown. Moving these comparator harnesses entirely to
the public Artifact/KernelCall surface remains a cleanup opportunity, not a claimed
completed migration.

Current focused evidence: 57 hardware-free tests, 568 assertions pass, including
content snapshots, concurrent identical loads, creation/readback/destruction faults,
shared borrowers, hidden-debt identity, reservation watches/validators, and stale
cache/context references, loaded-publication rollback, destruction reentry, and
benchmark prefix/recording rollback. A fresh JVM confirmed 27 registration/boundary
tests (250 assertions), 38 ownership/benchmark fault tests (355 assertions), and
seven actual-device boundary tests (84 assertions) against the final exact-reference
representation. The pivotal reviewer approved the code subject to device/CI gates;
the local device and exact-head CI gates are met. This slice does not certify
root context, command-list, buffer or reset lifetime safety; those are the following
steps, and the full eight-item compiler campaign remains open.

### Root-construction ownership (2026-10-03, local implementation under review)

The shared runtime-root boundary reserves a cleanup DAG and exact generation before
OpenCL context/queue or Level Zero context/list acquisition. Native status, NULL,
readback, publication and destruction failures retain exact unresolved authority;
independent siblings may close, while their context and host Arena remain pinned.
Lazy ZE async-list acquisition uses the same root's pre-reserved slot, serialized
with initialization. Invalid construction callbacks fail before publication.

This slice deliberately changes live `shutdown!`/`reset!` to fail with
`:runtime-root-leases-incomplete`, without destroying or dropping resources.
Only failed, never-live initialization can be cleaned by this boundary. Compiler-only
development reload remains available; native runtime reload requires a fresh process.
This is a temporary explicit restriction, not completed root lifetime support.

Local root/module evidence is 34 tests and 381 assertions, including production
initializers under hardware-free native fault injection. The combined ownership
suite passes 84 tests and 890 assertions. Pivotal review found no remaining R1
correctness blocker, conditional on fresh-process device checks. A fresh capped
REPL now passes 27 registration/boundary tests (250 assertions), seven actual-device
execution tests (84 assertions), and six live-root/reset assertions across OpenCL
and Level Zero. PR #1017 passed all seven exact-head CI gates and merged as `66ace972`.
Next: generation-qualified session/buffer/registration/recording/event leases and
direct allocation migration, followed by checked live teardown/reset. Do not let
root-internal cache ownership masquerade as an external lease that prevents all
teardown. The training, distributed simulator/exchange and durable PDE/AMR campaign
requirements remain unchanged and incomplete.

### Resource-lease follow-up (R2, in progress)

PR #1017 landed R1; PR #1018 passed all seven exact-head CI gates and merged as
`d4cb8c17`. Its first hardware-free gate exercises
generation-qualified lease admission/retirement, concurrent independent leases,
failed lazy-child acquisition, and lost-generation cleanup debt. No live teardown
is enabled by introducing this primitive. Canonical OCL/ZE buffers now acquire a
lease before native contact; independently owned OCL sub-buffers own separate leases,
while ZE pointer views share their root allocation's owner. The lease is the final
dependency in the same canonical child cleanup DAG. Known cleanup retires it; unknown
creation/destruction retains the composite child owner and keeps the root pinned.

Shared `cleanup/build!` now marks the exact owner as constructing, without holding
its monitor across callbacks. Concurrent or reentrant release declines before mutation
with `:owner-construction-in-progress` and explicit retry-safe metadata. Marker install
and retirement check exact token identity after watches; mismatches retain unresolved
authority rather than clearing another generation. Rollback clears its own marker
before native cleanup. This closes a publish-before-contact race in the common owner
construction boundary, not a buffer-specific dispatch convention.

Current focused evidence: 169 tests, 1,770 assertions pass; seven real-device boundary
checks (84 assertions), six live-root/reset checks, and two real-device buffer/view
lease checks (eight assertions) pass on Intel Arc OpenCL and Level Zero. No timing or full live
root teardown claim follows from these tests.

The next slice, `runtime/registration-root-leases`, keeps metadata-only registrations
lazy. Loading acquires one exact root lease before native program/module/kernel contact;
the registration cleanup DAG releases it only after kernel, program and staging cleanup.
Identical registration reuses that authority. Known rollback retires the pin, while
unknown native acquisition/destruction retains the composite owner and blocks retirement.
Pure root-admission rejection leaves its reserved slot fresh; publication failure uses
the common owner transaction rather than pretending a native acquisition occurred.
Pre-lock lifecycle guards also reject native callbacks entering the registration registry,
including ZE module-cache callbacks, before lock-order inversion can occur.

Registration-slice acceptance: 176 focused tests / 1,844 assertions pass in the persistent
REPL; seven actual-device boundary checks / 84 assertions, six live-root/reset assertions,
and two buffer/view tests / eight assertions pass on the local OpenCL/Level Zero device.
PR #1019 passed all seven exact-head CI gates and merged as `8268509f`.
Prepared calls, recordings/events and raw allocation consumers are still outstanding.

The prepared-resource follow-up starts with ZE's public `create-kernel-fresh`: every
fresh kernel now has an independent root pin in its existing kernel/module-borrow
cleanup DAG. The module cache remains a root-internal resource, while a fresh kernel
is a lifetime that may survive the base registration. Kernel destruction must finish
before the module borrow retires, and both precede root-pin retirement. Unknown create,
NULL readback or destruction leaves both borrow and pin retained. Hardware-free tests
exercise two independent kernels and all three uncertain outcomes; 178 focused tests /
1,860 assertions and seven actual-device boundary tests / 84 assertions pass.
OpenCL's public binder now constructs its independent kernel through the canonical
root-child owner, with a reserved native acquisition slot and exact root Arena projection.
The registration lock covers loading through fresh creation; logical native-use admission
is scoped separately so loading does not recursively enter the same guarded registry.
Unknown create/readback or destruction retains the composite kernel/root owner; known
argument-binding failure retires both. Root-child construction shares the transactional
lease-slot publication helper, with before/after-write failure tests proving no native
contact or stranded root pin.

Final local acceptance: 180 focused tests / 1,878 assertions, seven actual-device boundary
tests / 84 assertions, and four buffer/view/prepared device tests / 16 assertions pass.
On both backends a prepared generated kernel survives base-registration retirement,
executes correctly, and balances its independent pin. This does not establish arbitrary
concurrent root reset: recordings/events, use admission and raw consumers remain next.
PR #1020 passed all seven exact-head CI gates and merged as `ebe849bc`.

Recording follow-up: both `record-graph!` implementations construct through the canonical
root-child owner. The OpenCL submission and optional profiling queue precede root release;
ordinary graphs pin their borrowed compute queue's root too. ZE derives its existing native
list/event/pool/independent-queue DAG as a resource plan, then uses the same root construction
transaction rather than building a second owner. Native descriptor allocations and queue
creation consume the exact admitted root projection. Uncertain acquisition or drain retains
the graph's composite owner and its root pin.

Acceptance so far: 182 focused tests / 1,898 assertions, seven actual boundary tests /
84 assertions, and six buffer/view/prepared/recording device tests / 36 assertions pass.
The latter replay ordinary and profiled graphs on OpenCL and ZE after retiring their base
registrations, and verify independent root-pin balance through graph/prepared/buffer teardown.
PR #1021 passed all seven exact-head CI gates and merged as `427d238f`. These pins do not automatically borrow
every child kernel or buffer: existing session ownership still controls those dependencies.
Standalone asynchronous range transfers remain debt, especially OpenCL's old exception path
that swallows drain/event-release failures before closing staging. Migrate that path to retained
completion cleanup before enabling any live reset; synchronous-use and raw-allocation debt also
remain. No throughput or complete distributed-runtime ownership claim follows from these gates.

Standalone transfer follow-up: OpenCL transfers now share the submission resource plan
(drain → event acquisition slots → staging Arena) and the canonical root-child transaction.
Unknown enqueue/readback/drain/event destruction retains the exact composite owner, staging
and root pin, without native retry. Successful cleanup releases events and staging before the
root. Nonempty OpenCL completion tokens require an owner; the raw-handle/finally-close fallback
is removed. ZE shared-allocation transfers remain inline and do not invent an in-flight lease.

The session event table is the sole outer authority: its exact pending event owner and resident
footprint are published before backend contact. Private slots retain returned tokens or adopted
backend cleanup debt, eliminating a post-contact Atom publication. One cleanup DAG orders backend
completion before independently owned host leases. Host ownership is armed only after the backend
returns successfully, preserving the public caller-owns-on-failed-submission contract. Failed
unknown submission keeps that pending event/footprint, so buffer aliases/free/session close cannot
drop dependencies. Failed cleanup is not reported as a successful data operation. The backend
submission contract adds a uniform optional third argument with `:adopt-cleanup!`; direct callers
can still use two arguments. Public GPUEvent and retained-range APIs keep their signatures.

Current acceptance: 182 focused ownership tests / 1,898 assertions and 48 affected range/content/
fault/device tests / 519 assertions pass. New native fault tests cover unknown enqueue, NULL
readback, drain/event-release failure, strict token ownership, caller host ownership, resident
alias blocking, session close and pre-contact publication watch rejection. Actual transfer tests
check OpenCL root-pin balance, staged download visibility and honest ZE inline completion; existing
ordinary/profiled recording and prepared/buffer cases still pass. Pivotal final review and all
seven exact-head CI gates remain required. Synchronous-use admission, raw allocation consumers and
general live reset remain incomplete; these ownership tests do not establish performance.

Review follow-up separates operation failure from destruction failure: an await/profiling error
is recorded once, backend drain/release is still attempted, and host leases retire only after known
backend retirement. A destruction failure retains the canonical debt without repeating await or
an indeterminate native release; the original operation error is preserved with cleanup failure
suppressed. Known retirement consumes a released event even when its operation failed. Session
teardown uses the same private consumption path without treating an already retired operation
error as remaining native debt. Public poll/await/release now admit through scoped session use,
preventing callback/watch reentry into buffer or session lifetime mutation.

The review-fix fault suite passes 12 tests / 166 assertions, including caught/uncaught native and
cleanup-publication watch reentry and known-await-failure session teardown. An uncaught cleanup
publication watch retains its exact owner and root pin before backend contact; this is fail-closed
retention, not a general recovery guarantee for arbitrary throwing watches. The 182-test /
1,898-assertion ownership suite also passes after these fixes. Final review and exact-head CI are
still required; live reset admission remains deliberately unchanged.

Host-lease handoff follows the entire successful construction transaction, not merely native
submission return. A final construction-marker watch can still reject submission after backend
success: rollback retires the backend token but leaves host leases caller-owned, removes known
retired event debt, and preserves the exact primary error. No fallible publication or callback
follows the private host-ownership handoff. The dedicated regression exercises this boundary.

PR #1022 passed all seven exact-head CI gates and merged as `4fdcf4c1`. The affected ownership
suite passes 182 tests / 1,898 assertions, and transfer/range/distributed/local-native lease
acceptance passes 39 tests / 320 assertions. The existing tiny resident Gemma LoRA workload also
passes 43 assertions: 25 Level Zero updates reduce loss from 2.800152 to 0.256915 while tracking
the independent JVM trajectory. This is the existing model twin, not external real-weight
training, throughput or live-reset evidence.

Integration must retain the authoritative child cleanup as well as its root pin;
a count or raw pointer alone is insufficient. Session construction currently creates
only a kernel-arena identifier and must remain lazy (no GPU initialization merely to
construct a session in hardware-free CI). Canonical native buffer owners, independently
owned OpenCL sub-buffers, kernel/program registrations, prepared bindings, graph
recordings and asynchronous events must retire their root lease only after all native
children are released. ZE non-owning views inherit the allocation's lifetime rather
than fabricating a separate owner. Root-owned cached modules are internal children,
not permanent external leases.

Direct allocation debt found in the production inventory: `runtime/display.clj`
resolves raw ZE `alloc-shared`; ZE SoA construction uses `alloc-shared`/`alloc-device`;
the obsolete scan helper allocated raw block-sum/offset scratch. The retirement below removes
that unreachable path rather than introducing another owner for it. Migrate the remaining
display/SoA consumers to canonical
owners or retire the paths, including their fault and view coverage, before changing
the fail-closed live reset gate. Buffer/registration/session lock ordering must be
reviewed against concurrent acquisition and teardown, not inferred from single-thread
test success.

### Retire backend-local scan runtime duplication — 2026-10-03

The production scan path already lowers TypedSOAC to SegScan, then a verified KernelGraph /
KernelDispatch behind the common executable ABI. The public `raster.par/scan`,
`raster.par/scan-exclusive` and `raster.gpu/invoke-scan!` surfaces retain their contracts. Raw
exclusive scan entering the source-shaped backend still fails closed with
`:exclusive-scan-requires-typed-schedule`; it cannot select a backend-local algorithm.

An independent reference/dispatch audit found no production, test, dev or benchmark caller,
generated marker or dynamic selector for either backend's `invoke-registered-scan-exclusive-kernel`.
The only ZE recursive-helper references were its definition/self-call and that obsolete entry.
Both entries and private ZE `invoke-full-gpu-scan!` are removed (235 lines), including raw
block-sum/offset allocation and finally-free behavior that could lose native acquisition debt or
mask a primary error. This is a break for direct callers of those internal runtime Vars, not a
change to supported source or session scan semantics. No raw allocation API is removed in this
slice: display and GpuSoA still use those APIs and require separate ownership migration.

Architectural ratchets reject these markers in program extraction and require the retired runtime
Vars to be absent. Existing typed graph/ABI, raw-source rejection and compatibility-ledger tests
remain the semantic oracles. The capped REPL passes 28 boundary/ledger tests / 150 assertions and
17 selected typed route, JVM, staged/resident and actual OpenCL/Level Zero scan tests / 88 assertions.
Persistent native namespaces are not reloaded: only the three deleted Vars were explicitly
unmapped. Cold CI remains necessary to validate source loading; final review and all seven
exact-head gates are required before merge. This retirement narrows ownership debt, not the
original training, distributed or AMR completion requirements.

### Display allocation uses canonical ownership — 2026-10-03

The live `runtime/display.clj` GPU constructor previously resolved raw ZE `alloc-shared`, exposed
an allocation without a release API, and called nonexistent JDK 25 `MemorySegment.copyInto`.
It now allocates through the existing canonical `make-buffer` transaction and retains that exact
child cleanup before native contact. The enclosing RenderBuffer cleanup delegates to the child
owner; it creates no second native destruction authority. Construction and initialization failure
retire known resources or return explicit unresolved cleanup containing the exact child debt.
Repeated retain/adopt callbacks must name the identical canonical child owner; a distinct
generation is rejected without overwriting the first. The constructor's unresolved-error chain
retains any rejected second generation, while the enclosing cleanup retires or retains the first.
Pixels are initialized to zero before publication; each axis and pixel count are checked, so
invalid/overflowing dimensions, including an oversized axis paired with zero, fail before
driver contact. The record's shape is unchanged and native namespaces/records are not reloaded.

The surface lifetime addition is `close-render-buffer!`: GPU callers must establish completion
and end all pointer borrows before closing. `:device-buffer` exposes the canonical allocation;
`:seg` remains a borrowed pointer, not ownership authority. CPU close is a no-op. Synchronization
checks both the render owner and allocation owner and copies completed pixels into the existing
AWT int array. It is not an implicit wait or a zero-copy GPU-to-AWT view; CPU-to-AWT remains
zero-copy. This does not establish arbitrary concurrent kernel borrowing or safe live reset.

Six hardware-free fault tests plus nine actual root-lease device tests pass 15 tests / 92
assertions, including real ZE render allocation/synchronization and balanced root-pin retirement.
The existing display compatibility cases pass 3 tests / 11 assertions and the affected canonical
ownership suite passes 182 tests / 1,898 assertions. Exact-head review and all seven CI gates are
required before merge. The GpuSoA review recommends retirement in favor of the already working
ResidentComposite/session-owned field path; do not extend raw composite allocation as a second
ownership framework. External training and distributed/AMR completion remain on the campaign.

### Retire ZE-only raw SoA ownership — 2026-10-03

The reference audit found no production/dev/benchmark caller of ZE `GpuSoA`, its raw shared/device
allocators, or its reflection-based copy helpers. The supported `defvalue` SoA vertical already
uses typed field projection, ordered `:binding`/`:field` ABI slots, session-owned DeviceBuffers
and non-owning ResidentComposite values. Retire the separate ZE record, constructors, predicate,
copy helpers, `n-elements` helper and binding arms rather than adding another composite owner.
Direct callers of these internal ZE Vars must migrate; supported SoA source semantics are unchanged.
Vulkan's separate GpuSoA implementation and general raw allocation APIs are untouched.

Five leaking/exception-swallowing raw-allocation tests are replaced by a cold namespace absence
ratchet. Ordered/subset composite binding checks now exercise both backends; actual public
aggregate device tests remain. New hardware-free tests exercise second-field create/upload
failure through canonical session allocation, including uncertain native destruction: no partial
value is published, known prefixes retire, only the exact uncertain child retains its root pin,
and subsequent close never repeats uncertain native destruction. A freed composite field is
rejected by both production binders before driver loading. This is not a new concurrent-borrow
or live-reset guarantee.

The capped REPL passes 74 affected tests / 749 assertions, including public aggregate execution
on OpenCL and Level Zero. Deleted source and test Vars were explicitly unmapped without reloading
native records; cold CI remains required. Exact-head independent review and all seven CI gates
remain mandatory before merge. The eight-item campaign, including external training acceptance,
distributed execution and multilevel/AMR validation, is not complete.

### Wide boxed JVM calls — 2026-10-06

Real-checkpoint training isolation exposed a separate JVM boundary defect: a dynamic
38-argument VJP call emitted a nonexistent `IFn.invoke` overload. Positional `IFn.invoke`
ends at twenty arguments. The three boxed `emit-fn-call` paths (current Var root, local
function and expression head) now share one emitter: up to twenty arguments retain
positional invocation; wider calls build an argument array and use `RT.seq`/`IFn.applyTo`.
The head and each argument are evaluated once, in source order. Typed static calls and
surface arity semantics are unchanged; this is not new GPU variadic support.

Regressions cover 20/21/38 arguments in all three paths, current Var-root replacement,
mixed primitive boxing, effect order, and a real compiled 38-argument typed callee's
`applyTo`. The capped REPL passes 39 JVM tests / 102 assertions and the combined wide-call
and compiled-AD tests pass 8 tests / 30 assertions. Independent review found no blockers;
cold CI remains required. Other guarded interop/wrapper fallback emitters remain separate
debt, not a claim that every boxed call path has been consolidated.

The real-weight acceptance gate remains held at its original tolerance. Its raw Ak
gradient already differs before SGD. On identical captured operands, the final Ak
contraction and preceding low-rank contraction match explicit sequential FP32 evaluation;
the normalized input also matches the CPU exactly. These checks narrow the investigation
to upstream cotangents, but do not establish full training acceptance or SOTA performance.

### Typed JVM loop carriers and lexical scope — 2026-10-06

The real-weight investigation exposed a JVM precision inconsistency, not a new GPU
schedule requirement. Ordinary lazy-JIT `sum-kv-heads` accumulated Float inputs in
Double, whereas the typed GPU/AOT path used Float. The cancellation probe
`[1e8, 1, -1e8]` returned 1 on the former path and 0 on the latter. Recurrence
analysis lacked the loop's own binding environment and the element type of expanded
`clojure.core/aget` reads. Both are now retained; dependent carry widths propagate
to a fixed point, and Long/Float joins use the existing Double promotion rule.

General structural let inference, recurrence scanning and loop seeding share a
source-ordered lexical binding environment. Unknown locals explicitly shadow outer
types rather than inheriting them, including lets inside recurrence expressions.
Array load emission and inference share their retained operand-type translation;
bare `aget` names are not treated as canonical intrinsics without source context.
Numeric recurrence branch joins do not change general boxed-if emission.

The numerical surface consequence is intentional: affected ordinary Float loops
now honor their declared per-add Float precision. Explicit Double loops and genuine
recurrence widening remain supported. This does not change AD rules, GPU reduction
association, BLAS arithmetic policy or acceptance tolerances. The independent review
found no remaining blockers; affected JVM suites pass 79 tests / 270 assertions in
the capped REPL. Warm emitter reloads required fresh anonymous class names in that
diagnostic process; no production counter/cache policy was changed. Cold CI is still
required. Rebuild the real-checkpoint CPU oracle before comparing again: previously
compiled ordinary functions retain the old arithmetic. The external real-weight
gate remains held, and the eight-item campaign remains incomplete.

Cold CI then exposed three `matrix-norm` verifier errors: improved lexical inference
made `case*` predict a uniform primitive result, but a loop arm still emitted a boxed
value. Case emission now reconciles each actual reaching branch, including its default,
with the chosen stack merge type. Unsupported coercions and void-to-primitive predictions
fail before verification instead of silently claiming a conversion. Named/default loop
arms, empty/nonempty inputs and a deliberately corrupt Boolean prediction are covered.
The final affected JVM plus dense-linear-algebra run passes 95 tests / 312 assertions;
the replacement exact-head cold CI run remains required before merge.

A fresh capped process rebuilding the pinned real-checkpoint oracle still declines:
loss error is 0.009375, input-gradient relative error 0.005526, and 11/28 adapter
coordinate-relative checks exceed the unchanged 0.02 threshold (maximum 1.06532).
The loop precision discrepancy is therefore fixed independently of the remaining
training mismatch. Continue identical-operand BLAS/sequential-FP32/public-GPU
triangulation and attention-cotangent isolation; do not silently change the oracle,
relax tolerance or promote schedules on this evidence.

### Same-operand contraction numerical evidence — 2026-10-06

NN, NT and TN BLAS projections now have a bounded public-device regression against
an independent sequential FP32 oracle, rounding every product and addition. It covers
reduction widths 3, 17 and 640 with nondyadic inputs, plus `[1e8, 1, -1e8]`
cancellation in each physical layout. The affected public matrix replay selection
passes 4 tests / 78 assertions on local OpenCL and Level Zero, with no native skips.
This checks the current portable evaluation order, not CPU BLAS bitwise equivalence
or an optimized matrix-family throughput claim.

A separate same-operand CPU/OpenCL diagnostic uses threaded MKL (thread environment
variables unset). GPU output matches explicit sequential FP32 exactly in all nine
smooth-input cases; MKL differs by up to 9.06e-6 absolute. The pipeline documentation
therefore no longer promises that `:f32-scalar` bit-tracks CPU BLAS. Exactness refers
to the retained typed evaluation order, and excluding FP16 does not guarantee a
full-gradient coordinate-relative bound near cancellation.

Remaining design obligation: state the arithmetic permission of projected BLAS
operations at their existing semantic boundary and preserve it through TypedSOAC,
scheduling, executable/tuning identity and reference selection. Ordered source folds,
abstract BLAS products and explicitly reassociated schedules must not silently share
an exactness claim. Do not introduce a second registry or weaken the real-checkpoint
gate to paper over this distinction. Continue same-input attention pullback isolation
before deciding whether that gate exposes a miscompile or an inadequate arithmetic
equivalence specification. External full-model training acceptance is still open.

### Source-arithmetic provenance through the existing contraction spine — 2026-10-06

The existing BLAS projection table now marks abstract product provenance. The concrete
Float/Double contract is constructed only after contextual specialization resolves the
contraction dtype; raw overloaded method spelling and stale call tags are not a second dtype
oracle. Ordinary and staged contractions retain their typed SSA semantics without fabricating
one global precision for a mixed-component fold. The closed descriptive schema distinguishes
operand identity conversion, the product-reduction precision floor, implementation-dependent
association/rounding inside that product, and the unchanged typed result transform.

The facet is explicit in TypedSOAC, projected contraction facts, both SegRed lowering paths,
scheduled numerics and the existing emitted certificate. A changed or omitted scheduled facet
rejects, as does a BLAS compatibility assertion on a source lacking that provenance. Existing
semantic identity therefore changes; old contraction cache artifacts rebuild. No new IR,
operation registry, schedule permission, selector or reference tolerance was introduced. Missing
facets in older typed equations conservatively project to retained SSA, never inferred BLAS.

Focused frontend/source tests pass 120 tests / 646 assertions; affected contraction, dialect
and certificate suites pass 84 tests / 636 assertions in the capped warm REPL. Review found
and corrected premature dtype resolution and lost producer-local provenance. The joined
hardware-free test covers the full semantic-to-artifact chain and rejects certificate mutation.
Affected public portable/register replay passes 4 tests / 78 assertions on local OpenCL and
Level Zero, with no skips. Full cold CI remains the merge gate.

Cold emitter-fixture CI identified an omitted facet in the staged packed/scalar candidate
constructors. Both now explicitly retain the same source facet before their graph rebinding;
the shared scheduled validator remains strict rather than hiding the omission with a default.
The same audit found split-K combine and parent mixed-matrix refinement propagation. The combine
retains its own ordered-SSA facet; physical cast/transpose/matrix stages do not acquire the parent
BLAS contract. Attention JVP additionally specializes an already staged closure after homogeneous
operand dtypes resolve: BLAS provenance is re-instantiated at that existing boundary, while
ordinary staged component semantics and the constructor mismatch guard remain unchanged.
Its focused suite passes 5 tests / 21 assertions. The exact public emitter-fixture path emits
157 artifacts each for synthetic CUDA and HIP; compiler-toolchain CI is still required.
The reviewed replacement also passes 53 GEMM/staged tests / 1338 assertions and 34 source-facet/
GEMM tests / 1243 assertions, including explicit local split-K and parent refinement provenance.

This is provenance, not a new equivalence theorem or schedule admission rule. Follow up with
restrictive refinement checks and reference-selection policy, preserving existing explicit
reassociation/mixed-precision consent. Before provenance could ever grant permission, distinguish
recognized compiler BLAS origins from user-supplied metadata assertions; the marker currently
authenticates neither. Initial numerical compatibility evidence remains alpha=1/beta=0; nontrivial
BLAS alpha/beta rounding is not proved by the product facet. The real-checkpoint gate (#1060)
remains held with unchanged tolerances and the original eight-item campaign remains incomplete.

### Declared product-accumulation floor — 2026-10-06

The shared numerical validator now applies a reject-only check when a contract carries abstract
BLAS source arithmetic and declares an accumulator dtype: a Float source permits Float/Double,
and a Double source permits Double. All single and component declarations are checked, including
when both fields appear. Naming an exact mode does not bypass a below-floor declaration. Ordered
typed SSA with no global accumulator declaration remains unchanged; ordinary staged and mixed
component folds do not acquire a fabricated BLAS floor. Existing mixed-matrix models still retain
Float accumulation and separately attest their explicit operand conversion.

This checks declared accumulator dtypes, not every instruction or operand conversion. Exact
source/body certificates and paired operational models remain their proof boundaries. No new
schedule permission, selector, error tolerance or numerical oracle is introduced. Independent
review found no blocker; focused source/numerical/GEMM suites pass 49 tests / 1390 assertions.
Native portable BLAS and explicitly mixed attention/projection checks
pass 2 tests / 72 assertions on OpenCL and Level Zero. The real-checkpoint gate remains held.

### Explicit register-product realization — 2026-10-08

The real-checkpoint diagnosis is recorded in `local-compiler-evidence.md`: generated
forward matrix products and matrix pullbacks match ordered Float multiply/add
references, while native BLAS uses different reduction/rounding realizations.
This is not sufficient to close the unchanged model-gradient gate.

The existing register-tiled schedule now exposes `:typed-contraction :multiply-add`
as `:decomposed` (default) or `:fused`. The latter produces canonical typed `:fma`
SSA without a new operator registry, emitter or model-specific compiler rule.
Only explicitly permissive register-tiled/fused-register dispatch requests admit
it; other strategies and restrictive precision reject the request. The resolved
schedule, body, legality, numerical certificate, tuning metadata and exact
complete-write rederivation retain the choice. Unknown nested schedule fields
fail closed, including misspelled arithmetic choices.

`Decomposed` describes the typed IR, not proof that a vendor compiler will refrain
from contraction. Numerical rounding remains implementation-defined. Explicit
FMA is currently a C-family capability; the existing WASM polynomial facet is not
a single-round FMA implementation. Before cross-target exact rounding can be
claimed, introduce checked target non-contraction controls or explicit rounded
operations and verify the resulting device/compiler instructions. This is a
correctness follow-up, not permission to weaken an ordered source reduction.

Bit-sensitive NN/NT projection checks on local OpenCL and Level Zero compare raw
Float bits against independent ordered oracles, replace inputs and compile
decomposed/fused/decomposed to exercise cache separation. Fused CUDA sm80 PTX and
HIP gfx1100 code compile locally; inspected PTX contains `fma.rn.f32`. No NVIDIA
or AMD execution/performance acceptance is implied. The real checkpoint's fused
Q projection matches the independent ordered FMA oracle exactly, but still differs
from native BLAS. Full-model gradient parity, sustained training and the original
distributed/AMR campaign residuals remain open.

### Finite collective composition and parameter transitions — 2026-10-09

`refinement-plan-fields` separates existing collective field construction from the
ordinary whole-plan validation boundary. Fields are neither a certificate nor an
executable. `refinement-plan` retains its existing single-request behavior.
`compose-refinement-plans` merges explicitly scoped requests: equal values, shards,
groups and local entries may be shared, but calls, copy bindings and refinements
must remain disjoint. The containing context supplies public outputs and metadata;
nonempty fragment attributes are rejected instead of silently losing evidence.
There is no inferred alias, ID renaming, parameter update or dependency.

The finite training regression composes three AD → all-reduce → SGD steps, with
unequal local batch counts. One parameter value/shard/allocation and startup source
per worker persists across steps; contributions and scratch are epoch-local. Each
new producer depends on the preceding update and only final updates are exported.
Existing readiness and runtime authorities still require one initialization phase
and reject replay of a completed owner. Numerical comparison uses an independently
rounded scalar oracle at the existing tolerance; replica equality is separately
bitwise. This is a co-located resident-copy fixture, not multi-node fabric,
dynamic epoch execution, a trainer API or acceptance of the held real-model gate.
Private allocations deliberately remain distinct throughout this finite DAG;
steady-state training and lifetime-proved scratch recycling remain follow-ups.

The affected refinement/training suites pass 20 tests / 371 assertions on the
capped warm REPL, including actual OpenCL and Level Zero execution with no
capability skips reported. Whole-repository fresh-process and CUDA/HIP compiler
checks remain CI obligations; this does not report a training throughput result.

### Whole AMR cycle as a local numerical provider — 2026-10-09

The existing ratio-2 diffusion cycle is lowered equation-first as one local
program. An ordinary certified LinkComposition connects its coarse/fine outputs
to the existing `arrays/acopy!` operation, committing both fields to initialized
state owners. The cycle borrows those owners read-only. Topology is constant and
intermediate storage is private scratch; the public outputs are the two state
fields. Preparation does not open a device session or allocate device storage.
Semantic input/output ports identify the connections, rather than inspection of
generated ABI names. No diffusion, interpolation, reflux or restriction kernel
is reimplemented by this provider. Public `compiled/compose` retains the prepared
artifact, component preparation reports and donated state boundary for subsequent
execution-identity checks; the provider does not implement another composition API.

Two consecutive invocations of the same local executable compare both complete
fields with the independent coordinate oracle at the existing 1e-11 tolerance.
They also check composite mass and exact average-down. The second invocation
uses the resident committed state without a host upload; original host arrays
remain unchanged. Focused tests pass 4 tests / 47 assertions on actual OpenCL and
Level Zero with no capability skips reported. PR #1127 landed after exact-head
independent approval and all seven fresh-process/compiler CI contexts succeeded.

This boundary is not yet an AMR temporal certificate or a distributed cycle.
The next planner slice must explicitly bind the exact existing program and its
ordered memory/effect evidence to coarse prediction, two fine substeps, temporal
boundary interpolation, register reset/accumulation/consumption, reflux and
average-down. The distributed plan should see one complete local compute call,
not duplicate its internal scratch scheduling. Keep the current schema-1
hierarchy/transfer modes unchanged; introduce temporal facts as a validated,
certificate-bound facet rather than unchecked attributes. Initial synchronization,
CFL stability and the mathematical meaning of the pinned numerical producer
remain separate obligations. No adaptive hierarchy, multi-patch cycle, mid-cycle
restart, storage recycling or asynchronous completion proof is claimed.

#### Temporal binding investigation

The current synthetic Intel preparation retains 26 source equations in the cycle
program: three host-only shape equations and 23 device submissions, followed by
the two state-commit submissions. These counts are observations of this retained
artifact, not a fixed schedule contract or a numerical correctness criterion.
Retained equation result-storage metadata and the validated ordered memory report
provide the storage/effect boundary; source equations must remain authoritative
if fusion later changes the physical submission count. Matching generated kernel
names, destination names or counts does not establish interpolation/reflux math.

No new patch-reshape machinery is needed for the next distributed binding.
`distributed-compute` already admits explicit local domains with one plain dense,
equal-volume ABI leaf, matching dtype and row-major coordinates. A flat numerical
ABI can therefore bind a rectangular patch through existing `:local-shape` and
owned `:placements`, with bounds and coverage checked by the same authority.
This does not permit arbitrary strides, quantized representations or aliasing.

The temporal facet must retain the hierarchy/state/distributed certificates and
exact prepared-program identity, distinguish the two state owners, and bind
register reset/update/consumption evidence to the complete-cycle compute call.
It must preserve the existing source-read/target-write transfer witness unchanged.
Source-level temporal stage meaning is explicit producer evidence; ordering,
storage coverage and exact artifact correspondence are independently checked.
The facet must reject changed geometry, timestep, stage coverage, field mappings,
completion call or retained program instead of accepting descriptive fingerprints
as a certificate. Publication still needs real completion and durable-byte
evidence; ordered memory accesses cannot authorize release or scratch recycling.

The planning review favors a separate certified temporal execution facet over a
schema-2 extension at this stage. Attach it to a verified schema-1
`:hierarchy-only` workload and retain the original sealed Prepared at the ODE /
execution layer, avoiding a compiler-IR dependency on `gpu.compiled`. Neither
the existing transfer-cycle mode nor its two-role implementation witness changes.
Start with pure stage/effect attestation and tamper verification, then bind the
actual provider and project one ordinary distributed compute entry. Generalized
multi-patch/adaptive temporal planning is the point to reconsider schema 2.

Exact program identity deliberately excludes initializer bytes. A same-shaped
topology/coefficient change need not alter it. For the first temporal facet,
geometry and numerical-stage meaning must remain explicitly trusted producer
attestations, bound into the certificate but not advertised as topology-byte or
PDE correctness proofs. A later content-addressed constant projection can make
the geometry-to-initializer correspondence independently checkable.

`emitted-parallel-program/retained-numerical-equations` now supplies the shared
top-level semantic projection needed by stage coverage. It validates the emitted
program, excludes host-only setup and physical operation sequences, and expands
the retained source equations of a compound emission. Dispatch alternatives
must agree under the existing canonical fingerprint relation, including NaN and
signed-zero semantics; expanded equation IDs must remain unique. Structured
loops retain their nested control under one outer equation. Physical fusion
does not erase the numerical spine or authorize an application interpretation.
Host-prefix validation and projection use one numerical-body extraction helper.
Affected provenance/fusion/dispatch/structured-control suites pass 72 tests /
685 assertions on the capped warm REPL; fresh CI remains a landing requirement.

The first ODE temporal projection now checks the bounded producer's nine-stage
attestation against that shared semantic spine: each numerical equation occurs
once in source order; every stage retains its predecessor, time-fraction roles
and register transition. Register evidence preserves declared result-storage
access roles as well as operand dependencies, so a read/write update cannot be
misclassified as a write-only reset. Its last semantic use is derived from those
events. The register must be a nonempty plain FP64 tensor, not an input scalar or
packed representation. These are structural checks over a numerical producer
attestation, not inference of its mathematical meaning from destination names.

The stage layer deliberately leaves full-write reset coverage, completion and
release unproven. It does not yet bind a Prepared, workload, geometry bytes or
distributed call. Those remain the next execution-facet obligations; no new
planner/runtime or alternate diffusion kernel is introduced. Tests compile the
actual numerical cycle without device allocation and include isolated access-role
and storage-facet controls. The focused stage suite passes 5 tests / 25 assertions
on the warm REPL; exact-head review and fresh CI remain landing requirements.
Together with the existing numerical/device/provider suites, the affected AMR
check passes 18 tests / 374 assertions, including actual OpenCL and Level Zero
cycles and synchronized restart with no capability skips reported.

The next execution-binding investigation confirms that an ordinary one-worker,
one-compute DistributedPlan can bind both resident cycle state owners through
existing explicit `[4,4]` local domains over their flat `[16]` ABI leaves. Normal
`compute-bindings` derives `:read-write` for both fields and normal readiness
accepts the plan's initialization/dataflow. Values still require an explicit
sharding facet; no layout or sharding inference was added to bypass that rule.
The probe allocates no driver resources and is not yet a certified AMR execution,
runtime demonstration, durable publication or measured cost result. Next bind
this exact local program and these whole-patch domains to a verified
hierarchy-only workload and the temporal producer evidence.

The bounded temporal execution facet now independently checks that workload,
its sole distributed compute binding, and the original sealed Prepared against
the producer's stage/timestep/field evidence. Both whole state owners require
shared-authority complete-write coverage. Physical register evidence requires
a complete private reset, coarse transport, fine transport and final reflux
consumption in order; release and observed completion remain unproven.

The new certificate does not retain mutable initializer arrays: its verified
workload witness omits the ordinary live distributed certificate, while its
source-free distributed projection replaces the single local LinkPlan with the
existing exact execution identity. The original workload and Prepared remain
owners outside this structural witness. Readiness still independently checks
initializer source identity, but the retained obligations exclude source objects.
Geometry fingerprints and synchronized input state remain explicit producer
attestations, not verified bytes, CFL or conservation proofs.

This investigation exposed missing Long result metadata in synthesized region-copy
address additions. Retaining the known counted ordinal width lets the existing
range/coverage authority prove full-field commits; no coverage predicate was
weakened. Whole copies with nonzero source offsets pass, while shifted/short
destination copies remain partial. The prerequisite is PR #1130. The focused
warm-REPL copy/coverage/execution suites pass 14 tests / 58 assertions with no
failures or errors. The temporal facet is still under review and has not yet
demonstrated distributed runtime execution or durable publication.

### Bounded temporal cycle through distributed ownership

PR #1130 is merged as `7609bedb`; PR #1131 is merged as `5228aa99`.
Both landed only after exact-head independent review and all seven registered
CI contexts succeeded. The stage/execution facet is now an available structural
boundary, not a theorem of PDE correctness or a device-completion receipt.

The follow-up executes that exact certified cycle through the existing one-shot
distributed runtime on both OpenCL and Level Zero. One logical worker is mapped
explicitly to each physical target with a declared aggregate memory budget.
Two cycles run in separately prepared plans and fresh contexts, with synchronous
readback between them. Both match the independent coordinate-loop oracle within
the existing `1e-11` bound, preserve composite mass, and retain exact average-down.
Outputs are rejected before completion and a completed execution cannot replay
stale startup evidence. This is local co-location, not a measured network/fabric
result; copied readback inputs are not durable manifest publication.

The runtime probe found a shared borrowing-boundary gap: a source-backed private
temporary lost its caller initialization obligation when its source was removed.
The existing projection now expresses that obligation using the existing `:state`
role and `:requires` authority. Original initialization sources/views remain with
the enclosing owner and are uploaded before any borrowed local call. Source-free
private temporaries still require ordered producers; ownership alone never proves
bytes initialized. Tests cover owner-view coverage, no spurious caller requirement
after a first local overwrite, idempotence and missing-source rejection.

The borrowing/device run passes 19 tests / 137 assertions with both FP64 probes
available and no skips. The expanded borrowing controls separately pass 18 tests /
103 assertions. This follow-up still requires exact-head review and fresh CI.
General multi-patch/adaptive temporal planning, distributed completion-backed
publication, physical cost calibration and the held real-model numerical gate
remain separate open campaign obligations.

### One actual storage probe, distinct owner capabilities

`gpu.storage-representation/observe!` extracts the existing generated sentinel
probe's session execution and resource cleanup from `Compiled`. It canonicalizes
the dtype, checks actual live storage capability and quiescence, emits for the
session dialect, measures exact bytes, and rejects session/device identity drift.
Temporary graph/buffer cleanup preserves the first failure and attempts every
release. The result is unsealed historical observation data, never a completion
receipt, ownership capability or a fact transferable to another owner.

Its synchronous policy callback lets the existing compiled wrapper retain
exclusive mutation, output invalidation/poisoning, and its original sealed
same-owner representation evidence. Capability rejection precedes that mutation
callback. The observation thunk is same-thread, one-shot and cannot be retained
past that callback; quiescence and identity are rechecked before allocation.
No new store, cache, type registry or handwritten kernel is introduced.
Distributed ownership/evidence admission remains a follow-up, not an implied
consequence of observing bytes on a session.

The existing ownership/fault oracles plus raw-observation controls pass 10 tests /
175 assertions. Existing actual owner-bound evidence checks pass on OpenCL and
Level Zero (2 tests / 48 assertions), including foreign facts, live output leases,
cached module reuse and output retirement. Independent exact-head review and
fresh CI are still required.

### Captured AMR restore consistency before mapped bytes

The follow-up restore verifier composes the existing fixed numerical-state semantic
boundary with independently revalidated source CertifiedCycleExecution evidence.
Re-certifying a changed manifest cannot replace exact producer/temporal/completion
identity, ordered source fields, synchronized next-step lineage, numerical policy or
the complete plain FP64 storage encoding. Historical representation snapshots are
not fresh-target evidence or authentication of serialized producer claims.

The hardware-free verifier controls pass 2 tests / 11 assertions. The native
continuation oracle is extended to check restore compatibility before opening bytes,
materialize the bounded captured chunks as local files, independently verify their
content through real read-only mmap leases, decode into fresh host startup inputs,
then execute a separately instantiated next cycle. This is explicit fixture
materialization, not a new production storage provider or a power-loss durability
receipt. Production provider availability finalization and metadata publication
remain application-owned, using the existing generic finalizer.

### Completion-backed bounded distributed AMR capture

The next item-8 slice connects the existing CertifiedCycleExecution to an original
completed distributed owner and the existing NumericalStateManifest/ContentProvider
contracts. It revalidates the exact compiler plan, pins its own output scope, and
captures the two complete fields with bounded element-aligned downloads and SHA-256.
The verified local domain explicitly relates a flat physical array to its dense
patch shape; identical element counts and byte extent do not license arbitrary
striding, packing or partial coverage. Generic distributed and bounded AMR capture
share one internal physical capture engine: storage evidence is checked before
every download, after the metadata-only manifest builder, after each provider
ingestion and at completion. The final manifest is certified before any provider
write; the builder cannot replace the derived fields or obtain resident readers.

Semantic admission remains adapter-specific. Generic capture still requires
placement/sharding-free portable tensors and canonicalizes their dtype/shape.
AMR retains its independently checked complete AbstractValues, including sharding,
coordinates and attributes. Shared physical admission checks plain dense encoding,
matching dtype, equal-volume shape and exact byte extent; it does not erase those
semantic facets or replace AMR's geometry, temporal and lineage proofs.

The captured manifest retains field geometry, numerical policy and producer-attested
meaning, with an advanced synchronized coordinate and the input state's identity as
its parent. All manifest fields are validated before ingestion. Ingestion failure
returns no state but may leave orphan blobs. The existing availability finalizer
must independently verify/promote content before application-owned metadata
publication. Neither observation nor hashing proves the producer's PDE semantics,
initial geometry bytes or parent existence in a durable catalog.

Hardware-free fault controls and an actual OpenCL/Level Zero capture/close/decode/
fresh-context continuation oracle are under validation. The native test's bounded
in-memory provider is explicitly not evidence of real durable storage. A mapped
provider realization and publication/restore compatibility controls remain next,
alongside the broader multi-patch, adaptive and physical-calibration obligations.

### Distributed representation evidence and explicit probe budgets

The distributed owner now uses the shared in-process identity-witness factory,
with a private issuer distinct from Compiled. Copies and facts from another
owner domain do not authenticate. This removes duplicated sealing mechanics,
not the independent compiler or lifetime checks required by each owner.

After successful execution, the existing generated storage probe can run in
the exclusive output scope. Its private allocation must fit the retained
physical-target resident-buffer budget. Admission declines preserve completion;
failure after probe entry marks the owner failed before releasing the scope.
Evidence binds the exact owner, session identity, target, dtype and observed
device snapshot. Consumption rechecks those facts inside the actual read scope.
Historical dereference alone grants no live-byte or durable-publication authority.

Focused identity, ownership, budget, cleanup and failure controls pass 14 tests /
270 assertions. The actual OpenCL and Level Zero representation/AMR oracles pass
3 tests / 104 assertions without skips. Independent review and fresh CI remain
required. Completion-backed distributed checkpoint capture is the next boundary;
this slice does not fabricate a local CompletedEvidence or certify program identity.

### Distributed output lifetime before durable capture

PR #1132 is merged as `4c7dc400` after independent exact-head review and all
seven registered CI gates. Its two-cycle local distributed execution is not
yet a durable checkpoint publication boundary.

The next prerequisite is `distributed/with-output-values!`: a synchronous read
scope on the existing enclosing owner. Successful completion is required;
close from any thread is refused, and callback failure releases the scope.
User code runs without the owner monitor held, avoiding deadlock when a provider
worker attempts close while the capture callback awaits that worker.
Asynchronous transfers must finish inside the callback and only
copied data may escape. Direct session mutation is outside this contract. This
does not manufacture a sealed compiler receipt or verify stored content.

The hardware-free lifecycle checks pass 4 tests / 30 assertions. The existing
AMR device oracle now copies its field bytes inside this scope; device validation
and exact-head review are still in progress.

Distributed hardware precision remains a distinct campaign obligation. The
topology represents device descriptors, memory capacities and directed links
with bandwidth and latency. The simulator accounts for dependencies and shared
route links, plus explicitly stated transfer/compute serialization. Its route
cost is an optimistic cut-through estimate, not measured fabric performance.
Local device calibration can refine kernel descriptors, but measured cluster
routes, shared physical bottlenecks, contention/load validity and realization
of predicted asynchronous overlap remain open. Keep capability legality and
ownership proofs separate from uncertain performance estimates.

### Shared bounded element-to-byte capture adapter

`numerical-content/element-byte-reader` extracts the existing local resident
capture's alignment adapter for reuse by distributed owners. An explicit
synchronous element downloader fills bounded scratch; arbitrary provider byte
windows receive only their requested raw bits. Extents must contain whole
elements, each byte window is at most 64 KiB, and subtraction-based admission
rejects overflow-sized requests before any source contact. Zero-byte windows
perform no download. Caller lifetime/immutability/representation obligations
remain explicit; this does not infer endianness or authenticate a producer.

The existing local capture uses this shared implementation instead of maintaining
its own alignment arithmetic. Focused adapter, local capture and storage suites
pass 27 tests / 522 assertions, including exact bytes at unaligned windows,
non-power-of-two element widths, overflow/range rejection, download failure,
content hashes and provider cleanup. Reloading the protocol owner required
reloading its test provider implementations in the warm REPL; the initial stale
provider failures were not numerical failures. Actual local capture/restart
checks pass on OpenCL and Level Zero (2 tests / 34 assertions). Independent
exact-head review and fresh CI remain pending.

### One dtype authority for hardware storage widths

Hardware planning now uses `compiler.core.dtype/bytes-of`, rather than a
second private width table. Canonical aliases and Half storage therefore have
the same widths in planning and lowering. Width alone does not grant native
arithmetic support. The planning helper now rejects `:short` and `:i16`: those
are not compiler scalar dtypes, even though Half uses JVM short-array storage.
No surface dtype was added or removed. The focused hardware, roofline and
calibration-admission suites pass 18 tests / 154 assertions in the warm REPL;
full CI and exact-head review remain separate gates.

### Shared physical resident-root accounting

`distributed-plan/resident-storage-plan` now owns the finite distributed runtime's
existing root-pool projection. Runtime allocation and optional simulator capacity
reporting consume the same independently validated bindings, exact root contracts
and overflow-safe physical budgets. Repeated calls share identical allocation
roots; co-located workers retain separate shard buffers and can share private
weights. Remapped workers require an explicit aggregate target budget. Distinct
shards still cannot silently alias storage; a test fixture attempting that was
rejected and corrected without weakening admission.

This models owned LinkPlan roots retained until owner close, not total physical
peak memory. KernelGraph scratch, backend temporaries, host staging, driver
allocations, available VRAM and lifetime reuse remain outside its scope. Default
analytical simulation and certificates are unchanged. The helper may retain
source-bearing LinkPlans and is not a portable report or allocation authority.

Focused planner/storage/cost/compute checks pass 71 tests / 552 assertions.
Existing actual AD -> all-reduce -> SGD checks pass on OpenCL and Level Zero
(2 tests / 154 assertions), including runtime-versus-simulator budget equality.
These are warm-REPL results; fresh CI and independent exact-head review remain
separate gates. The original real-model training numerical acceptance remains
open and is not replaced by these small distributed fixtures.

### Canonical graph-owned temporary storage requirements

`kernel-graph-call/temporary-storage-plan` now projects declared graph scratch
through the existing graph/scalar preflight and extent algebra. Runtime
`temporary-specs` delegates to this projection, rather than having a separate
planning evaluator. Each byte extent is checked against signed 64-bit storage
before allocation; aggregate requirements use exact arithmetic. Existing
out-of-range element values still fail in the canonical extent resolver.

The model counts all declared temporaries until graph unbind. It excludes
external roots, backend temporaries, host staging, alignment/driver overhead and
available device memory. It does not infer live-range reuse or grant allocation
authority. Distributed root budgets do not yet include these requirements;
binding/dispatch lifetime-aware aggregation is the remaining integration work.

Focused graph-call, GEMM and SegOp checks pass 87 tests / 1,605 assertions.
Existing native inclusive-scan tests pass on OpenCL and Level Zero (2 tests /
14 assertions), comparing planned allocation identities/byte sizes/totals with
actual private buffers as well as numerical output. These are warm-REPL checks;
fresh CI and independent exact-head review remain separate gates.

### One compiler-owned stage-once binding plan

The bounded graph/carry-variant calculation now lives in
`emitted-parallel-program-call/preparation-plan`; the runtime delegates to it.
This is the existing staging algorithm, not a new loop interpretation: zero-trip
loops bind no body variant, preserved initial carries plus parity rotation need
at most three body variants, and changing induction scalars still explicitly
decline stage-once preparation. Program-wide shape scalars and local physical
scalar overrides retain their existing merge order. Pure planning exposes the
same graph/scalar/buffer tuples that runtime binding consumes.

Runtime preparation validates the call once through this entry before acquiring
resources. Stage-once declines now precede invalid resolver/executor diagnostics;
accepted program arithmetic and surface semantics are unchanged. Existing
cleanup ownership and reverse rollback remain runtime responsibilities.

This supplies a shared input to upcoming memory accounting: local prepared
variants coexist, whereas the current synchronous distributed runner releases
each local LinkPlan after its compute action. A physical scratch peak must not
sum all distributed actions or count only a default dispatch alternative.
No distributed scratch-capacity admission is enabled by this extraction.

Focused cleanup and staging checks pass 35 tests / 233 assertions, including
compiler-plan versus runtime-bind comparisons and unchanged rollback behavior.
Actual co-located AD/all-reduce/SGD checks pass on OpenCL and Level Zero (2 tests /
154 assertions). Warm-REPL checks do not replace fresh CI or the original
real-model training numerical acceptance, which remains open.

### Dispatch-aware linked scratch and serial physical budgets

`LinkPlan/temporary-storage-plan` uses the same descriptor ABI binding facts,
linked physical extent facts, graph sizing and program staging as execution.
Separately prepared variants retain separate scratch even when their graph is
identical. Descriptor steps conservatively take the maximum across all declared
alternatives, not only the preferred/default schedule. Inapplicable or unresolved
alternative sizing refuses this optional report rather than silently contributing
zero; it does not change ordinary dispatch admission or claim exact selection.

Optional `:include-graph-temporaries? true` joins these requirements to the shared
DistributedPlan resident-root accounting, simulator report and runtime admission.
All roots remain resident, while the current synchronous runner releases each
local executable after its compute action: physical root bytes plus the maximum
local prepared scratch requirement is checked against the declared capacity.
The default remains the explicitly roots-only budget; no surface arithmetic or
ordinary execution behavior changes. Backend temporaries, host staging,
alignment/driver overhead and available VRAM remain excluded, so this is still
not a complete total-device memory proof or an asynchronous lifetime model.

Focused LinkPlan/distributed checks pass 94 tests / 677 assertions, with the
structured program check adding 1 test / 11 assertions. They cover a nonselected
scratch alternative, Half storage width, physical extent binding, private scratch
duplication, the three coexisting 512-byte loop variants, and serial max rather
than scratch sum. Native AD/all-reduce/SGD checks with the stronger optional
budget pass on OpenCL and Level Zero (2 tests / 154 assertions). These are
warm-REPL results; fresh CI and exact-head review remain separate gates. The
opt-in projection repeats structural validation; no preparation-time or kernel
performance improvement is claimed. The original real-model gate remains open.

Review caught a planner/runtime extent-environment mismatch for descriptor
scratch. The projection now preserves the current binder distinction: descriptor
scratch resolves only actual ABI scalars, while direct graphs/programs receive
linked view extents. A regression rejects invented descriptor `(extent x)`
values; the Half descriptor test sizes from its real `n` scalar, and the direct
graph test independently exercises physical view extents. Generalizing this
runtime boundary remains a separate task, not an inferred precision/shape fact.

Hardware-free runtime admission regressions additionally exercise the public
`instantiate!` entry with initialized inputs. Insufficient root or opt-in scratch
budgets reject before session creation, allocation or upload; exact-fit budgets
reach the deliberately stopped session boundary. Default roots-only behavior and
malformed scratch options are checked independently. The admission and storage
planner namespaces pass together (9 tests / 47 assertions). These boundary tests
do not execute their synthetic kernel or establish total physical memory safety.

### Producer-local fusion legality analysis

A live real-model VJP preparation sampled in scalar-effect descriptor-key
construction exposed repeated producer analysis during vertical candidate
enumeration. The exceptional-conversion check depends only on the producer and
the current immutable program, but previously ran again for each consumer.
It now uses one lazy result per producer in one enumeration. The original guard
order is unchanged, and subsequent enumerations/fixpoint iterations rederive it;
there is no persistent cache, retained certificate or new admission authority.

The four-map regression reproduces six checks on the previous implementation
and three on the replacement, with unchanged candidate pairs. A fresh all-trapping
enumeration rejects all candidates, and a producer without prospective consumers
never forces its check. Existing checked-conversion completion-boundary tests
remain. Focused fusion, placement, matrix-input and epilogue suites pass 65 tests /
417 assertions in the capped REPL. This is reduced analysis-call evidence, not a
measured end-to-end compile-time or device-performance improvement.

The new real-model layer-1 isolation reproduces all 14 original CPU adapter
gradients exactly with the original forward input and loss cotangent. Its GPU
preparation did not complete: the warm process had retained an eight-argument
math-policy invocation caller while a later staging reload restored the
seven-argument program-call constructor. This is a mixed-revision diagnostic
environment, not evidence of a production miscompile or a new numerical result.
Rebuild a coherent compiler source set before continuing that comparison. The
original componentwise real-model acceptance remains failed and unchanged.

Legacy scan graph constructors still omit the source GraphScalar table and, in some
cases, node scalar dependencies. Their emitted ABI projection remains the established
compatibility contract, not a certified original storage-scalar dependency proof.
Scheduled artifacts retain exact projection validation, but cannot authorize additional
source storage dependencies without the original typed scalar contract. Migration debt:
derive and retain the complete source scalar interface and per-node dependencies in
those constructors, then retire the ABI-only boundary with legacy/typed scan, carry,
empty, mixed-storage and math-policy regressions. A present scalar table (including an
empty one) with a missing node dependency contract must fail closed.

### Retained nested operand evidence at scalar dispatch

The shared source walker must consume already-proved operand return types before
speculating about contextual literal narrowing or selecting an exact-arity overload.
A known nested Double result remains Double even in a Float-array specialization;
only a genuinely compatible Float operation adopts the contextual Float literal.
This applies at the ordinary and reducible call boundaries, using the existing
inference/dispatch authority rather than a new backend-local type reconstruction.

The extracted correction from nanparth's #1338 tree (`063e7fc0`) is bounded to that
selection order. Unknown results and Object upper bounds retain the current dynamic
dispatch behavior; the change is not a claim of complete source inference. Tests
compare exact floating bits, both operand positions, effect order/single evaluation,
reducible calls, and split/unsplit loop carriers. Explicit Float addition boundaries
and uncast Double promotion are distinct oracles, not interchangeable tolerances.
The wider conditional-index proof, artifact quotienting, resident-scalar composition
and unchanged real-model acceptance obligations remain open.

### Explicit terminal host observation over the ordered invocation owner

A checked scalar return is an observable source computation, not permission to
expose its numerical dependency as the function result. Equation-first compilation
now retains the complete ordered terminal scalar suffix in InvocationPlan and
compiles it under the producer's historical source ownership. Double/Long ports
come from revalidated equation/result correspondence, not output-name or dtype
search. A later numerical consumer cannot be hidden in the host suffix.

`raster.gpu.link/instantiate-host-result!` and `run-host-result!` explicitly opt into
the host observation. They reuse the ordered executable's storage, lifetime lock,
initialization proof and failure boundary; numerical stages remain resident until
the declared scalar observation. Resident output/lease APIs refuse to expose these
private dependencies. Source ownership is checked before replay mutation. Terminal
exceptions or missing publication poison execution before replay credit/readiness.
This is a deliberate API distinction, not an implicit download in resident replay.

Projection into a GraphLinkInstance retains exact emitted-equation provenance as
revalidation input; ordinary graph/ABI/alias checks precede reuse of existing
complete-write analysis. It does not accept caller-authored coverage claims.
Terminal-owner persistence remains refused pending a reconstruction contract.
Equivalent lazy source-JIT definition transitions can still conservatively stale
an owner; general lifecycle/cache convergence is not established by this slice.

Bounded local qualification includes real OpenCL and Level Zero checked scalar
returns, changed-input replay, overflow, explicit mode refusal and injected missing
publication. No full campaign, device-performance or broad source-completeness
claim follows from these tests. Checked conversions inside numerical map stages
remain a separate open completion-before-publication obligation.

Follow-up terminal coverage includes binary core arithmetic over actual primitive
Double operands. The owner uses the shared source-arithmetic plan and method-name
translation to resolve exact Numbers overloads, retaining Double operations before
checked Float narrowing. Logical result tags do not select carriers. Other arities
and carriers still refuse. Core arithmetic Var/inline definitions join the existing
bootstrap ownership guard, not a new inference registry. A fresh-process affected
gate passes 27 tests / 1,067 assertions, including source arithmetic plans, terminal
ownership controls and the guarded Float hardware experiment; the original public
wide-add/narrow test passes ten assertions on OpenCL/Level Zero with changed replay
and explicit resident-API refusal. This is host-result qualification, not resident
composition or a broad green CI receipt.

### Observable scalar checkpoints alongside resident state

A returned loss can precede VJP and optimizer effects, while the typed envelope
also exposes mutated tensor outputs. The existing singleton ScalarIR projection
now selects exactly one unused public scalar among those observable outputs,
without moving its equation or dropping any tensor output. Complete source-use
auditing includes captures, host observations, aliases, shape and allocation
expressions, and result storage. Singleton writer and host-observation checks
follow the full physical alias component, including non-symbol value IDs.

The pre-component selection is transported only as a closed candidate hint;
production revalidation reconstructs the complete enclosing equation spine and
independently rederives the existing residency proof. Multiple scalar roots,
later consumers, malformed hints, alias-mediated writes and host observations
remain refused. No new scalar runtime, host readback or workload-specific rule
is introduced. Public OpenCL and Level Zero regressions compare changed-input
replay against an independent CPU oracle, preserving pre-update Float-to-Double
checkpoints, subsequent state effects and artifact lifetime checks.

A separate bounded real-workload probe prepared the generated public 1-layer,
8-dimensional, batch-2 GSDM SGD step and executed three resident Level Zero
updates. Across all 30 weight fields, maximum absolute error against the compiled
CPU reference was 2.98e-8; loss error was at most 1.43e-6. Captured host state
remained unchanged. Preparation took 114 seconds in a loaded warm process;
this is not a controlled compile-time benchmark. Full 2-layer/32-dimensional
GSDM statistical training, Adam/EMA coverage, real-weight Gemma adapter gates,
and normalized AD general composition/default migration remain separate open
acceptance obligations.
