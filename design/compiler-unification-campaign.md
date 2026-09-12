# Compiler unification campaign

Authorized scope: complete the four stages below in order. Each production migration must retain
its numerical, ABI, ownership and resource contracts; isolated emitter coverage is not a completed
vertical. The north star remains the architectural specification.

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
| Unscheduled effects | Typed mini-program first; `generate-par-map-void-kernel` on decline | Raw/nested plain-array effects schedule once; unsupported bodies retain a reported compatibility route. |
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
Unscheduled raw bodies retain the existing adapter with an explicit `:compatibility-effect-opencl`
emission route and `:effect-compatibility` count, including through nested host control. This is
not full effect-emitter retirement: logical SoA expansion and other unsupported effect bodies still
need their own typed closure. The independent source-emitter tests remain unchanged.
An explicit logical-SoA guard precedes plain-array scheduling: the typed frontend can otherwise
accept the container as one float pointer rather than decline. Bare, conditional and let-bound probes
retain the existing per-field dtypes, names and grouped logical binding through the source adapter.
Unscheduled effects within partially scheduled source programs also retain that reported adapter:
introducing leaf-local lets after emission breaks flat resident extraction, and alpha-renaming
only the host form would disconnect its artifact argument plan. Semantic-stage composition tests
exercise this boundary. Complete host-region normalization must precede emission before retiring it.

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
Shared-memory pipelines or indexed matrix epilogues should be introduced when those measurements
or workload requirements justify them. Existing scientific/distributed acceptance gates remain.
