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
| Unscheduled effects | `generate-par-map-void-kernel` → `opencl_pass` | Raw/unbound effect fallback still present. |
| Strided transfers | Typed mini-program → KernelBody | Bare and host-wrapped leaves share typed scheduling; source generators moved to test-only oracles. |
| Active IDs | `generate-par-active-ids-kernel` → `opencl_pass` | ABM firms uses seeded agent indices, not arbitrary visibility compaction. |
| Compound local | `generate-compound-local-kernel` → `opencl_pass` | Compatibility markers remain; structured typed programs bypass detection. |
| Staged/quantized contractions | `generate-staged-contraction-kernel` → `contract_route` | Device-tested staged/lift/quantized source assembly remains. |
| JVM SIMD / scalar | `pipeline` → `par_simd` or source expansion | Bound SegOps coexist with traversal of the retained source projection. |
| CPU C/SIMD | `cpu/aot` → `cpu/csimd` | Bound SegOps preferred; compatibility reconstruction and distinct vector emission remain. |

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

## 3. General fusion and bounded specialization — queued

- Validate shared scalar-region composition, coupled reductions and post-reduction transforms.
- Audit recompute/materialize choices on fan-out, effects, aliases and observable results.
- Apply retained static structural facts before fusion, shape/layout/target facts before scheduling,
  and typed constant/range simplification afterward.
- Preserve ordered arithmetic and AD/effect boundaries. Use guards and bounded specialization
  caches; do not unroll large numerical iteration spaces merely because their sizes are known.

Exit: representative scientific and model expressions compile with explained fusion/placement
choices, differential correctness, and measured kernel/allocation/compile-cost changes.

## 4. Measure generated production kernels and selection — queued

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

Exit: reproducible production measurements identify wins/regressions and explain schedule choices;
claims are limited to tested hardware and workloads. Discuss the results before stages 5–6.

## Working loop

One small memory-capped REPL; focused affected tests locally. Full suites and hardware-free vendor
compilers run on CircleCI. Review candidate/certificate boundaries; squash only exact reviewed heads
with all required checks green. Never alter the concurrently edited main-checkout north-star file.
