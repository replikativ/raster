# Raster compiler north star

Status: architectural direction, reconciled with the implementation on 2026-08-27.

Raster should become a compiler in which a typed Clojure program, its parallel
algorithm, its schedule, its device placement, and its executable artifact are
all inspectable values. The compiler may search and learn schedules, but it must
never learn around the type system, effect system, legality checks, or numerical
oracle.

The shortest description of the destination is:

```text
typed Clojure
    → semantic program IR
    → functional parallel IR (SOAC)
    → scheduled kernel graph
    → verified kernel IR
    → target module
    → composable Compiled artifact
```

This is an evolution of Raster's existing architecture, not a replacement with
Futhark, Triton, JAX, or MLIR. Raster keeps its typed multiple-dispatch frontend,
functional Clojure programming model, explicit AD rules, portable hardware
model, and vendor-JIT backends. The reference systems supply specific missing
ideas at specific layers.

## 1. What is already strong

Raster is not starting from a toy compiler:

- `deftm`/`ftm`, typed dispatch, monomorphization, AD expansion, and qualified
  semantic operator identities form a capable language frontend.
- The nanopass pipeline has explicit pass ordering, diagnostics, and declared
  dialect arrows.
- SOAC fusion already handles useful vertical and horizontal fusion, aliases,
  multiple consumers, and profitability vetoes.
- Contractions carry explicit free and contracted axes, checked index maps,
  contraction facts, layouts, staged accumulators, and routes for scalar,
  register-tiled, DPAS, DP4A, and block-quant execution.
- Hardware descriptors centralize launch geometry, matrix-instruction shape,
  register budgets, cache and bandwidth data, and measured calibration
  provenance.
- Schedules are data, have a compile-time resource feasibility gate, and can be
  searched and cached by hardware signature.
- `Compiled` and `DeviceArray` establish device-resident values, ownership,
  donation, multi-output projection, and functional invocation.
- The backend surface already includes JVM, C/SIMD, OpenCL, Level Zero, CUDA,
  HIP, Vulkan, WGSL, and WASM compile gates.
- The quantized path already covers Q4_0, Q8_0, Q4_K, and Q6_K, including
  staged contraction and DP4A-related work.

These pieces are the foundation. The SegOp boundary now uses a first-class typed
`ParallelProgram`; remaining gaps are concentrated in the earlier SOAC program,
the scheduled kernel-body representation, duplicated lowering, and string-template seams.

## 2. The load-bearing gaps

### 2.1 The middle end is not yet a sequence of real dialects

`segop-lower-pass` now produces a `ParallelProgram` with ordered equations,
SSA-like result IDs, `AbstractValue` contracts, effects, diagnostics, and provenance.
GPU and JVM backends consume the recorded operations; the source expression is retained only to
reconstruct scalar host control around them. The `:segop-lowered` validator performs a full
operation/type legality check, and source equality invalidates an equation after a backend-local
rewrite. Direct calls to a backend may still use the explicit, counted compatibility re-lowering
path.

The remaining nominal boundary is earlier: SOAC fusion still constructs records and a dependency
graph from source-shaped S-expressions, rewrites them, and reconstructs `par` forms before the typed
program boundary. The problem is not the use of S-expressions; it is that stable value identity,
types, effects, aliases and dialect legality are recovered from source spelling instead of being
explicit in a first-class typed SOAC program.

The first architectural correction is therefore:

> SOAC, scheduled kernel graph, and kernel IR must be first-class program values,
> and every conversion must have a legality target that fails loudly.

### 2.2 Algorithm, schedule, and emitted kernel are still entangled

Raster has schedule data and hardware-aware contraction routes, but there is no
single stage that applies a schedule to an algorithm and produces the kernel IR
consumed by every emitter. Several scheduling axes are costed or gated more
broadly than they are emitted. Resident BLAS GEMM calls now close over compiler-emitted
scalar/direct/split executable graphs, but their semantic front door still sits outside the
general SegOp contraction route and the XMX leaf remains a target-specific emitter template.

The required separation is:

- **algorithm** says what values are computed;
- **schedule** says where, in what order, with what tiling/layout/staging;
- **kernel IR** records the selected execution without losing the algorithm's
  semantic identity;
- **emission** is a target spelling of the verified kernel IR.

No schedule key is complete until it is declared, checked, costed, emitted, and
measured. A key that intentionally models feasibility before emission must be
labelled as such in the schema and excluded from claims of executable coverage.

The first general `KernelBody` vocabulary now extends the matrix-fragment path with typed scalar
SSA, explicit literals and conversions, rank-checked scalar loads/stores, structured `if` and
loop-carried regions, and full-participation subgroup reductions/broadcasts. Branch and loop yields
are typed products; use-before-definition, identity collisions, divergent collectives, implicit
masked-load values, conversion policy gaps, and launch/subgroup mismatches fail verification. Body
launches use the shared `LaunchSpec`, keeping host launch expressions separate from in-kernel index
arithmetic. Group, 1–3D local-thread, subgroup, and subgroup-lane indices are distinct hardware
sources. Uniform memory facts require an explicit body-level stable-read contract; the checked
KernelBody-to-ABI seam projects it to a no-write-alias precondition, artifacts reject equal compiler
bindings, direct calls check resident ranges, and graph/link binding retains its stronger
overlapping-write rejection. Floating narrowing distinguishes IEEE overflow from integral
wrap/saturate/trap policies. Whole-kernel workgroup allocations now have static typed shapes, named
layouts, explicit 1–16-byte alignment, and one deterministic packed-memory plan; launch shared-byte
accounting must match that plan exactly. Full-participation acquire/release workgroup barriers are
rejected outside workgroup-uniform control flow and lower from the same operation to OpenCL,
CUDA, and HIP. Workgroup-cooperative contiguous global-to-workgroup copies now carry explicit
source/destination coordinates, element and transfer widths, cache intent, full participation, and
preferred-versus-required overlap. Named commits close the exact copies issued since the previous
commit; waits consume an oldest group prefix and state the remaining pipeline depth. The verifier
rejects unstable sources, divergent base addresses, live destination reuse, lifetimes crossing a
structured-region boundary, and staged-memory consumption before a separate workgroup barrier.
Required overlap also proves coordinate alignment, projects its base-address alignment through the
ordered ABI, and direct calls validate resident ranges before launch; preferred CUDA overlap keeps
a uniform runtime-alignment fallback.
OpenCL lowers the dependency graph to events, CUDA targets with compute capability 8.0 or newer to
`cp.async` groups, and older/unknown CUDA or HIP targets report an honest synchronous cooperative
lowering; those targets reject a schedule that requires overlap. Masked collectives
remain absent until a concrete schedule supplies verifiable participation. An implemented OpenCL
target spelling now lowers this general scalar/control vocabulary directly: dense ranked loads and
stores (including contiguous leading-slice views), SSA expressions, structured branch/loop yields,
explicit numerical casts, and full-subgroup reductions/broadcasts, without recovering an algorithm
or schedule. Unsupported target semantics fail at this boundary instead of silently changing a
cast, layout, collective, or reduction-association contract.

The first production route now constructs
`SegmentedWeightedReductionSchedule -> KernelBody -> OpenCL` and projects the body's stable reads
onto the invariant ordered ABI. Attention contributes only the routed-paged storage descriptor used
by this first lowering row; neither KernelBody nor the target emitter contains an attention
operation. Dense/CSR physical page routing combines independently with bounded contiguous intervals
or bounded CSR membership. The generated leaf has replaced the handwritten cooperative source and
is checked against the reference algebra on a real device. Its subgroup builtin is recorded honestly
as implementation-defined association rather than claiming a fixed tree. A second production
schedule now partitions bounded membership into static contiguous tiles, emits one partial
KernelBody per segment/tile, and deterministically merges the explicit maximum/denominator/weighted
value state in increasing tile order. The four typed partial buffers are operation-derived,
graph-private temporaries; the source operation and external ABI are unchanged. This two-kernel
schedule is opt-in until measured selection can account for dynamic history length and temporary
traffic. Scalar/control KernelBody now has one C-family lowering core with thin portable OpenCL,
CUDA and HIP spelling descriptors. The same scheduled cooperative and tiled-history bodies retain
one ordered ABI while OpenCL uses subgroup builtins and CUDA/HIP select explicit shuffle-down trees;
the emitted artifacts record that target association. CUDA is compiled to PTX and HIP to an RDNA3
code object in mandatory hardware-free CI jobs, while real Intel execution remains the numerical
oracle. The compiler fixtures now stage values through both synchronous workgroup memory and the
verified async issue/commit/wait contract; the exact async body is compiled to CUDA sm_80 PTX and
an RDNA3 HIP code object in hardware-free CI. CUDA/HIP runtime registration, launch and on-device
numerical coverage remain deliberately separate from source legality. The verifier still forbids
pending asynchronous state and live staged storage from escaping an ordinary structured region,
but `PipelinedFor` now carries a complete ordered async queue across loop iterations, verifies each
rotating stage's layout and lifetime, and requires an explicit exact or separate-epilogue tail
policy. OpenCL lowers the queue through native event variables; CUDA maps it to `cp.async` groups;
HIP preserves the same verified schedule with an honest synchronous fallback. The next production
step is a double-buffered scheduled weighted-reduction body using this representation. Local
swizzles and measured stage-depth selection then let this collapse the tiled graph toward a
FlashAttention-like single kernel without changing the semantic plan or external ABI.

### 2.3 Executable steps and resident artifact values compose

Descriptor instances share one executable-step binder: a step may select a
single `KernelArtifact` or a multi-kernel `KernelGraph`, own its private
temporaries, and flatten with other instances into one resident replay graph.
This closes the mechanism gap for GEMM-containing layer composition.

`LinkPlan` lifts that mechanism into a public immutable compiler value. Stable
typed/shaped nodes replace name-decoding binders; ordered descriptor instances
carry scalar and schedule environments; pure validation proves views, ranges,
ownership, aliases and producer/consumer effects before runtime allocation.
Instantiation allocates internal nodes once and returns an owned executable
value over one replay graph. It can also attach to a caller session and import
borrowed/external allocations without taking ownership, which is the boundary
needed by paged cache managers.

`ProgramStage` supplies the corresponding replacement seam for external runtime
effects. A frontend declares stable state/output anchors; Raster derives the
unique minimal descriptor interval from checked executable ABIs, proves its
live boundary, and projects ordinary before/selected/after descriptors. Paged
attention is the first consumer, not a special compiler pass or linker mode.

The pretrained decoder now composes through declared `ProgramStage` boundaries and `LinkPlan`
node identities rather than handwritten buffer-name inspection. An ordinary resident descriptor
has a pure certifying conversion into a one-instance plan: the checkable witness preserves
parameter order, pointer/value identity, scalar specialization, roles, realized view contracts,
schedule, aliases, target, and outputs. `Compiled` owns that witness and invokes only its
`LinkedExecutable`; compatible device inputs use an exact-view no-op or a device-to-device copy,
never an implicit host round trip. The runtime preserves the certified allocation identities and
profiles the same stable recorded-graph boundary. Each instance also retains the complete ordered
specialization environment required by descriptor shape closures, while runtime pointers still
come only from certified node bindings. Independently lowered `Prepared` artifacts now compose
before allocation through semantic input/output keys. The composition certificate namespaces
component identities, unifies explicit dataflow and shared constant/input nodes, and re-derives one
validated plan; one later instantiation therefore removes even the device copy. The remaining
value-layer cleanup is ranged-view composition and explicit cross-component ownership transfer;
the duplicate legacy whole-program binding API is retired.

Pretrained-rstr's batch boundary is the current concrete shape test: weights bind once to shared
constant nodes, while residual, scratch, position, logits and token storage are lane-local nodes or
disjoint views. Packed Q/K/V and attention views already fit LinkPlan. The first projection slice now
makes Q8_K activation quantization and Q4_K DP4A projection uniformly row-capable: row count is an
ordinary outer shape/launch extent, activation leaves are row-major, and packed weight leaves remain
shared. B=1 is no longer a separate kernel contract. The remaining performance work is to derive
these projection, post-attention and head programs from the scheduled contraction/SOAC path rather
than treating the Q4_K source kernel as the final abstraction; batching is not a linker convention
or an attention special case.

Buffered RoPE follows the same rule: one kernel consumes row-major `[B,heads,head-dim]` values and
the routed attention `int positions[B]` buffer, so B=1 and B>1 neither select different kernels nor
upload duplicate position representations. Dense semantic rows whose logical width differs from a
quantized contraction's padded width use a generated padded-row quantization adapter. It synthesizes
zeroes while packing and therefore avoids a materialized padding buffer; the branch-free equal-width
kernel remains available for already padded layouts. Padding is layout, not quantization-specific
memory ownership.

Decoder selection is likewise an ordinary indexed row reduction, not an attention or quantization
primitive. Scalar reduction is now the one-component case of a canonical typed `ProductReduction`;
multi-result reductions carry ordered, independently typed components plus distinct element and
closed binary-combine regions. The first portable `ReductionSchedule` maps a segmented product to
strided lane folds, a fixed workgroup-local tree and segment stores. Its workgroup is constrained by
the target thread limit and the sum of every component's local-memory width, while numerical mode
and tuning candidates remain inspectable data. `argmax-rows!` therefore emits one mixed-type
`(value,index)` workgroup tree per row with no compiler-visible global scratch. The semantic ABI
contains only values, output indices, row count and width; ties select the lowest index, and the first
NaN outranks numeric values so corruption is visible and deterministic. Subgroup/shuffle and
multi-workgroup alternatives remain schedule candidates, not new semantic primitives.

Indexed storage movement remains a separate generic operation. Allocation-free `gather-blocks!`
and `scatter-blocks!` move contiguous typed blocks between dense staging and routed resident
storage with one ordinary SOAC map per direction. Routing is an `int[nblocks]` buffer; it does not
encode pages, attention, cache policy, or quantization. Gather permits repeated sources, while the
non-reducing scatter contract requires unique destinations. Host validation proves active extents,
index bounds, non-aliasing and the current 32-bit device-index limit; emitted kernels retain bounds
guards. FP16 is a bit-preserving short-array storage overload whose physical ABI is `half*`, while
index arithmetic remains in the ordinary scalar compiler domain. `gather-rows!` remains the
row-oriented compatibility spelling. Consequently greedy decode composes indexed reduction and
row gather as ordinary graph dataflow, while chunked paged prefill and cache persistence can compose
bulk host/device staging with block scatter/gather. Neither path introduces a decoder, attention,
page-manager, or quantization primitive into the compiler.

Artifact linking and value rebinding are not runtime conveniences. They are
compiler primitives required for competitive model execution.

### 2.4 Shape, layout, effects, ownership, and sharding are separate facts

The current metadata and descriptor registries contain much of this information,
but there is no one JAX-like abstract value threaded through transformations.
Consequently, shape polymorphism, layout propagation, aliasing, device
placement, and future distributed sharding do not share a single checked
contract.

The first value-layer slice now makes that separation concrete. `AbstractValue` records logical
dtype/shape/layout, numerical representation, memory space, placement, sharding, ownership and
effects without containing a buffer or backend handle. `LinkValue` records a later physical layout
and an ordered set of named `LinkNode` leaves. Kernel ABI field identities certify the flattening,
and the common resident binder expands the same composite contract for Level Zero and OpenCL. A
Q4_K value can therefore be represented by packed blocks, scales and sums while allocation remains
generic byte storage; the allocator does not branch on Q4_K. Existing dense plans receive implicit
one-leaf values. The follow-up slice makes resident parameter and allocation descriptors speak the
same contract: composite parameters have field initializers, composite scratch has independent
leaf size closures, and allocation still sees only typed byte ranges. Certified composition now
unifies and shares whole logical values atomically while retaining both logical and physical
certificate mappings; public composite outputs must flatten every leaf in field order. The
remaining value-layer boundary is ranged/subview composition with explicit ownership transfer.

### 2.5 Workload coverage is broad at the backend edge, not yet through one door

Triton's practical strength is not merely its set of operations. Masks, block
pointers, layouts, reductions, scans, atomics, gathers/scatters, tensor-core
forms, and software pipelines travel through one staged compilation path.
Raster has many of the leaves, but production reachability through the same
verified IR is the important coverage metric.

## 3. The settled architecture

### 3.1 Semantic program IR

The output of tracing/lowering should be an immutable, SSA-like program with
unique value identities, nested regions, source locations, and explicit
constants. S-expression syntax may remain the readable frontend form, but
passes that need semantic facts should operate on program values rather than
recovering them from spelling or symbol metadata.

Every value has an `AbstractValue`:

```clojure
{:dtype        :f16
 :shape        [batch seq hidden] ; dimensions may be symbolic
 :logical-layout {:order [0 1 2]}
 :memory-space :device
 :placement    {:device :ze:0}
 :sharding     nil
 :ownership    :owned
 :effects      #{}}
```

The program envelope and fact tables may be records or validated maps, while operation and region
syntax may remain compact S-expressions. The invariant matters more than the container: all facets
describe the same explicit value identity, and a pass cannot update one without revalidation.

Operations use one descriptor/interface system. In addition to the existing
buffer, device, shape, placement, algebra, comparison, and result facets, an
operation can define:

- abstract evaluation;
- effect and alias behavior;
- JVP, VJP/transpose, and batching rules;
- layout and sharding constraints;
- canonicalization and fusion traits;
- legal lowerings for each next dialect;
- cost features and measurement identity.

This borrows JAX's useful core—abstract values, primitives, explicit equations,
transformation rules, effects, and pytrees—without borrowing Python tracing
semantics or making retracing the programming model.

### 3.2 Functional parallel IR

The portable algorithm dialect should contain nested, typed parallel
operations. Its minimum useful family is:

- map;
- reduce and scan, including multiple results;
- screma;
- histogram;
- stream/chunked fold;
- gather and scatter;
- stencil;
- contraction;
- structured `if`, `while`, and call regions;
- explicit effect/token operations where functional dataflow is insufficient.

Futhark is the semantic reference here. In particular, Raster should adopt its
discipline of typed SOAC inputs/results, nested parallel distribution,
regularity and balance legality, and typed kernel results. Raster need not copy
Futhark's Haskell representation or uniqueness type system wholesale.

This dialect is hardware-independent. Hardware facts may be supplied to a
costing or scheduling pass, but must not change the meaning of an operation.

#### 3.2.1 Representation and Pattern integration

Raster keeps the small functional, Lisp-native character of the SOAC IR. The intended form is a
typed, dialect-validated S-expression program rather than a tools.analyzer-style map AST or a graph
reconstructed from arbitrary source forms. Conceptually:

```clojure
(soac-program
  {:inputs [%x]
   :values {%x x-value %y y-value %z z-value}}
  [(= %y (map {:index %i :extent %n} [%x]
              (lambda [%xi] (* %xi %xi))))
   (= %z (reduce {:operator + :identity 0.0} [%y]))]
  [%z])
```

`../pattern` remains the rewrite engine. Its nanopass dialects declare the accepted S-expression
grammar and pass arrows; fusion rules match the compact SOAC equations directly. The surrounding
`ParallelProgram`-style envelope supplies stable value/equation IDs, `AbstractValue`s, effects,
aliases, consumption, results, diagnostics and provenance. Essential facts are explicit fields or
table entries keyed by IDs, not Clojure metadata that ordinary list reconstruction can drop.

This representation is shared before backend selection. SOAC fusion changes the program consumed
by both JVM and accelerator lowerings; SegOp and schedule conversion then specialize it. The JVM
SIMD backend already consumes certified SegOps from `ParallelProgram`, but current SOAC fusion first
round-trips through `par` forms and the scalar fallback still expands retained source. Migration is
complete when JVM SIMD, scalar/JVM and GPU routes consume the same typed SOAC/SegOp facts and no
backend silently re-lowers an exact source form.

### 3.3 Schedule and transform IR

A schedule is an immutable transformation program over the functional parallel
IR. It is more than a bag of kernel kwargs. It can express:

- split, tile, reorder, fuse, compute placement, and materialization;
- thread/block/subgroup mapping;
- vectorization and matrix-instruction selection;
- operand layout and explicit layout conversion;
- staging space, copies, pipeline depth, and barriers;
- reduction decomposition and split-K;
- buffer donation, residency, and graph capture;
- device/mesh placement and resharding policy.

Halide supplies the algorithm/schedule separation and structural search model.
MLIR's Transform dialect supplies a useful safety model: typed handles,
preconditions, declared effects, and recoverable versus irrecoverable transform
failure. Raster can implement those properties in EDN and Clojure records.

Applying a schedule produces a candidate scheduled kernel graph. Candidate
generation, legality, feasibility, profitability, and measurement remain
separate:

```text
rewrite legality → hardware feasibility → analytic rank → measured rank
```

A cost model may abstain. A failed legality or resource check is never converted
to a different semantic program.

Runtime selection and offline tuning operate on an emitted executable
alternative, not necessarily one kernel. An alternative is either a single
`KernelArtifact` or an emitted `KernelGraph`. All alternatives in one
`KernelDispatch` share one ordered external ABI, compiler argument order,
target, and logical effects; graph topology, entry points, launch geometry,
derived scalars, and private temporaries are schedule-owned and may differ.
This is what permits a direct contraction and a split-K partial-plus-combine
graph to compete without making split-K storage part of the user-visible call.

The first resident GEMM slice uses this boundary directly. Compilation emits scalar f32, direct
mixed-precision XMX, and split-K graphs with one `(A B C M N K)` ABI. Checked integer-expression
IR derives the pitch gate, occupancy decision, private conversion/transpose/partials storage,
`KC`, and 1–3D launches from shape and schedule data. The resident binder supplies ABI values and
binds ordinary graph calls; it does not assemble a GEMM algorithm. Constant-only layout/conversion
nodes are hoisted by a graph-generic cacheable-transform rule. Remaining work is to originate the
same graphs from the canonical contraction/Screma route, add vendor matrix-instruction leaves, and
make measured shape tables override the analytic selector.

### 3.4 Scheduled kernel graph and kernel IR

The scheduled graph contains calls between kernels and explicit resident
buffers. Each kernel contains typed operations and regions sufficient to
represent:

- N-dimensional program and thread indices;
- named register, shared/local, cache-resident, and global layouts;
- masked loads/stores and boundary values;
- block views/pointers, strides, offsets, gathers, and scatters;
- scalar, subgroup, block, and grid reductions/scans;
- matrix/dot operands and accumulator fragments;
- shared-memory allocation and swizzles;
- asynchronous copy, commit, wait, and barrier;
- atomics;
- structured control flow;
- ordered kernel inputs, outputs, scalars, aliases, and temporary buffers.

SegMap, SegRed, and SegScan should migrate into this real dialect. Histograms and
the missing result forms should be added by workload demand.

Layouts should initially be a finite, named family with stride, permutation,
tile, vector, and swizzle parameters, plus explicit legal conversions. This is
enough for DPAS, WMMA/WGMMA, and MFMA families while Raster delegates register
allocation and final instruction scheduling to vendor compilers. A general
Triton GF(2) linear-layout algebra is not a prerequisite.

The kernel ABI is one ordered, typed slot vector:

```clojure
[{:name 'x   :kind :input  :dtype :f16 :shape [m k] :layout x-layout}
 {:name 'w   :kind :input  :dtype :q4-k :shape [n k] :layout w-layout}
 {:name 'out :kind :output :dtype :f32 :shape [m n] :layout out-layout}]
```

The signature renderer, binder, verifier, cache key, profiler, and debugger all
consume this same vector. Unordered input/output symbol sets and independently
maintained argument counts must disappear.

### 3.5 Dialect conversion and verification

Raster should adopt MLIR's conversion discipline without requiring MLIR as a
runtime dependency:

- a conversion target declares legal, dynamically legal, and illegal
  operations;
- full conversion succeeds only when no illegal operation remains;
- partial conversion is named explicitly and cannot masquerade as a completed
  dialect;
- type/layout conversions insert explicit adapters;
- each operation supplies interfaces/traits, avoiding repeated exact-op
  conditionals;
- every pass declares preserved analyses and invalidates the rest;
- pass diagnostics retain source and semantic value identities.

The immediate implication is that SegOp lowering may no longer warn and return
`nil`. Unsupported forms produce a structured diagnostic naming the operation,
source, missing rule, target dialect, and possible legal fallback. A fallback
is an explicit conversion rule, not exception handling.

Native MLIR remains an optional future interchange or target boundary. Raster
should not take a mandatory LLVM/MLIR/Triton build dependency merely to gain
dialect concepts it can express in its existing nanopass system.

### 3.6 Target lowering

Target backends select instruction families and render source/binaries from the
same verified kernel IR. Vendor differences that change the structure of a
kernel—DPAS versus WMMA/WGMMA versus MFMA—fork in target lowering. Dtype
spellings and instruction variants within a family are data tables.

The vendor compiler continues to own register allocation, low-level instruction
scheduling, and binary generation until measurements demonstrate that this
boundary is the limiting factor.

A direct PTX route, when justified by those measurements, is a late NVIDIA target dialect beneath
`KernelBody`, not a second scheduling IR and never a shortcut from SOAC. Full conversion first
legalizes a verified body to NVIDIA address spaces, predicates, cache policies, vector memory
operations, `cp.async`/`mbarrier`/TMA dependencies, exact MMA/WGMMA operand tuples and resource
directives; a separate renderer produces PTX and `ptxas` produces cubin. CUDA C remains a broad
differential backend. PTX gives Raster tighter instruction selection and memory-ordering control,
but physical registers, spills and final SASS scheduling remain vendor-compiler responsibilities.
The artifact records `ptxas` register, spill, stack and shared-memory reports so feasibility and
autotuning can use actual resource outcomes.

### 3.7 Executable artifact and runtime

`Compiled` evolves from a whole-program wrapper into a composable executable
graph:

- target module and kernel cache;
- typed input/output tree specifications;
- device values with shape, dtype, layout, placement, and ownership;
- ordered kernel ABI bindings;
- explicit aliases and donation;
- captured constants and parameters;
- events and asynchronous dependencies;
- schedule, hardware signature, compiler version, and measurement provenance;
- graph instantiation/linking without recompiling shared kernels;
- serialization of all non-live-resource state.

Execution scheduling is a distinct backend-neutral lowering. An `ExecutionPlan`
assigns operations to logical queue classes and connects them with logical
wait/completion events. Compiler events are stable identities, never OpenCL,
Level Zero, MPI, or vendor-library handles. A backend may initially realize the
plan on one in-order compute queue, but it must retain the dependency DAG so
later compute/transfer/collective overlap is a scheduling change rather than an
ABI rewrite.

The runtime event contract is submit, nonblocking status, host wait, and safe
release. Completion owns the lifetime of every referenced graph, kernel,
allocation, and view; destruction must first establish completion. Status
observation alone is not assumed to establish host memory visibility.

The first transfer realization uses that same public event contract for
bounds-checked batches of host/resident ranges. A completed transfer reports
bytes, command count, elapsed time, and timing provenance. OpenCL uses owned
native staging and device-event timestamps; a caller may reuse upload sources
after submission, while download destinations become visible only after the
host wait. Level Zero's current shared allocations complete their Panama copy
inline and report host-monotonic timing explicitly rather than pretending that
a device copy event exists. Logical compute and transfer queues initially map
to one physical in-order OpenCL queue, preserving ordering until execution-plan
lowering can express and realize cross-queue dependencies safely. Immediate-wait
measurements are the calibration path; `host-wall-ns` otherwise includes any
intentional host work between submission and wait.

Device-to-device binding is the normal path. Host transfer is an explicit graph
edge with a reason and byte count. The compiler's memory plan owns temporary
liveness, reuse, alignment, and peak-memory accounting.

JAX pytrees are the model for separating user structure from flat leaves, but
Raster trees should retain stable keyed paths and type/shape information. JAX
donation and sharding contracts are also useful references; Raster's existing
ownership checks should remain fail-loud.

## 4. Quantized computation

Quantization should be represented as contraction semantics plus an operand
storage layout, not as one opaque hand-written kernel per format. A format
descriptor must define:

- block shape and byte layout;
- scale, offset, and auxiliary metadata encoding;
- logical element-to-byte index map;
- decode computation and signedness;
- permitted activation and accumulator dtypes;
- reference pack/unpack implementation;
- legal instruction families;
- numerical tolerance and golden vectors.

Q4_0, Q8_0, Q4_K, and Q6_K establish the pattern. The next formats should be
selected from actual pretrained and finetuning model requirements and measured
quality/performance—not from a goal of mirroring every `llama.cpp` enum. Likely
candidates include high-use IQ variants and current low-bit floating formats
such as MXFP4/NVFP4, subject to model evidence and hardware support.

`llama.cpp-new` is the interoperability and CPU-reference oracle. Every added
format needs pack/unpack parity, randomized boundary cases, odd-tail coverage,
and end-to-end logits or loss comparisons. A format is not complete when only
its isolated emitter test passes.

## 5. Hardware-aware compilation and autotuning

Raster's existing hardware descriptor and schedule cache form the right base.
Runtime probes and catalogues may retain backend-native names, but their compiler projection
must normalize the execution hierarchy. In particular, the descriptor records a set of supported
subgroup/warp/wave widths separately from its preferred width, maximum workgroup geometry, and
per-field provenance. A schedule chooses one supported width; it must never inherit another
vendor's default. Matrix-operation scope remains an independent capability because a device may
support several subgroup widths while one DPAS, MMA, or MFMA operation requires exactly one.

The closed loop is:

```text
program + abstract values + hardware descriptor
    → legal schedules
    → resource-feasible schedules
    → analytic shortlist
    → compile and device-time benchmark
    → stationary winner
    → versioned cache and Compiled artifact
```

This is a staged JIT/runtime contract, not only an ahead-of-time compiler pipeline. Static analysis
may leave symbolic shapes, strides, placement choices and schedule alternatives in a partially
specialized artifact. At binding or execution time, Raster may propagate newly known shapes,
residency, allocation/topology, backend capabilities and calibrated measurements back into
specialization, candidate generation and tuning. Adaptation always returns or selects a newly
verified immutable artifact; it never mutates a running kernel around type, effect, ownership,
numerical or resource checks. Logical events, measurements, hardware descriptors, `Compiled`,
`LinkPlan` and `ExecutionPlan` are the compiler/runtime seam to OpenCL, Level Zero, CUDA/HIP and
future collective or cluster runtimes.

The search state should include both kernel and graph choices. Per-kernel axes
include tile, vector width, instruction family, workgroup geometry, layout,
staging, pipeline depth, and reduction strategy. Graph axes include fusion,
materialization, parameter/activation residency, recomputation, capture, and
placement.

The benchmark protocol must record:

- warmup and cache-state policy;
- device-event time, not host submission time;
- minimum and distribution summary;
- stationarity/variance result;
- compilation time and binary size;
- peak memory and temporary bytes;
- transfers and bytes moved;
- hardware, driver, compiler, program, schedule, and numerical-mode hashes.

Analytic models seed and prune; measurement decides among survivors. Learned
cost models may later replace parts of ranking, but the raw observations remain
the source of truth.

## 6. Distributed and resource-aware programming model

Distribution belongs in the abstract value and scheduled graph, not in a
separate orchestration wrapper. The program model needs:

- a device mesh and topology descriptor;
- sharding annotations on logical dimensions;
- sharding propagation rules per operation;
- explicit reshard, all-reduce, all-gather, reduce-scatter, broadcast, and
  point-to-point operations;
- compute/communication dependency events;
- memory capacity and bandwidth constraints per device/link;
- heterogeneous placement and legal dtype/layout capabilities.

Compilation minimizes a cost vector rather than a single kernel time:

```text
latency, throughput, peak memory, transferred bytes,
compile/tune budget, energy when measurable, and numerical error
```

The compiler hierarchy extends upward without making the single-device `LinkPlan` a cluster object:

```text
semantic program and AD
    → verified mesh, topology, resource envelope and allocation
    → distributed schedule with explicit sharding and communication
    → certified DistributedPlan with shard-local programs and cross-device events
    → one LinkPlan and ExecutionPlan per device
    → KernelGraph / KernelBody / target artifacts
```

`DistributedPlan` is the missing bridge. It owns global-to-shard value mappings, collective groups,
point-to-point routes, resource claims, per-device plans and a witness for shard coverage,
replication, ownership, capabilities and collective agreement. Native communicators and event
handles remain runtime resources. An outer `WorkloadPlan` may handle admission, continuous batching,
deadlines, retries, checkpoints and safe reallocation for dynamic services; it selects certified
compiled variants rather than injecting service policy into SOAC or kernel IR.

The first distributed target should be data-parallel training with explicit gradient all-reduce,
followed by tensor/sequence sharding for a transformer block and a scientific halo-exchange case.
Pipeline and expert parallelism should wait until the value, sharding, communication and failure
contracts can state them without runtime side protocols. MPI, NCCL/RCCL, oneCCL, UCX and eventually
GPU-initiated transports such as NVSHMEM are interchangeable execution backends, not Raster's
semantic memory model. Datahike may retain durable topology, plan, measurement and decision history;
it does not replace hot-path allocation or communication.

## 7. Reflection, structural self-modification, and learning

Raster can make reflection unusually powerful because Clojure data is a natural
representation for programs and schedules. The safe version has four rules:

1. Source operations, semantic values, SOAC nodes, kernel nodes, and measurements
   retain stable identities across lowering.
2. IRs, transforms, schedules, hardware descriptions, and measurements are
   immutable, content-addressed, inspectable data.
3. Every structural change is a declared transform with preconditions,
   provenance, and a verifier; rejected changes leave the payload unchanged.
4. Learned proposals compete in shadow mode before they may be cached or shipped,
   and no proposal bypasses legality, resource, or numerical gates.

This permits learning across representations: a model can associate source
structure, SOAC structure, chosen schedules, emitted kernel features, hardware,
and measurements. It does not require unrestricted mutation of compiler code or
live executables. Rewrite policies and cost models are replaceable artifacts;
the trusted verifier stays small.

## 8. Landing order

Each increment should be a thin vertical slice with a production-path test.

The immediate continuation after the verified pipelined-loop increment is:

1. Use `PipelinedFor` for the double-buffered segmented weighted-reduction production route.
2. In parallel, define a typed SOAC S-expression dialect with `../pattern` and differential-test
   map→map, map→reduce and horizontal-map fusion against the current graph.
3. Carry that dialect in the existing program/value envelope and route one ordinary fused reduction
   through JVM SIMD and GPU `KernelBody` without reconstructing compiler facts from source spelling.
4. Add a finite verified shared-memory swizzle family and bank-conflict/resource model.
5. Make stage count and swizzle measured schedule axes, retire the legacy tuner, and remove the
   attention-provenance gate from the generic weighted-reduction matcher.
6. Migrate the remaining SOAC fusion rules and scalar JVM fallback, then remove compatibility
   re-lowering for the covered forms.
7. Add a differential PTX target dialect/module boundary. Start topology and sharding values as a
   read-only distributed track without interrupting the kernel and typed-middle-end verticals.

The integrated roadmap has six cooperating tracks rather than one backend-only sequence:

| Track | Near-term completion gate | Unlock |
|---|---|---|
| Verified device scheduling | Pipelined loop, double-buffered weighted reduction, tails and honest fallbacks | General staged reductions and FlashAttention-like schedules |
| Typed functional middle end | Pattern-declared typed SOAC program; map/reduce fusion reaches JVM and GPU from one fact set | Reliable fusion, batching, AD and reflection across representations |
| Portable peak kernels | Verified swizzles/layouts, unified matrix fragments and quant storage descriptors | Competitive GEMM, attention and scientific tiles across Intel/NVIDIA/AMD |
| Measurement and selection | One tuner over coupled graph/kernel axes with numerical and resource gates | Hardware-aware schedule selection and selective autotuning |
| Precise target lowering | Format-neutral target modules, differential PTX, later AMD-specific lowering | Exact async/matrix/cache control where portable source is insufficient |
| Distributed/workload planning | Typed mesh/topology/sharding values, then certified `DistributedPlan` | End-to-end node, cluster and data-center optimization |

The table below is the durable architectural dependency ledger, not a claim that every increment is
unstarted. ABI, artifact composition and much of the kernel-IR foundation have landed; their stated
completion gates remain useful for finding legacy paths that have not joined the common route.

| Increment | Work | Completion gate |
|---|---|---|
| 1. Real SegOp boundary | Introduce a first-class program container for SOAC/SegOps; make lowering a full, fail-loud conversion; remove metadata-only and backend re-lowering for the migrated forms; validate operations and types, not only outer `let*`. | Map, reduction, scan, and one contraction travel through the same recorded dialect path on CPU and GPU; unsupported forms cannot silently bypass it. |
| 2. Kernel ABI | Replace unordered symbol sets and hand-maintained signatures/binders with one ordered typed ABI; derive emission, binding, cache identity, and diagnostics from it. | Every production GPU invocation compares the emitted and bound ABI structurally; syntax/compile checks cover every emitted kernel. |
| 3. Abstract values and operation interfaces | Add canonical shape/dtype/layout/placement/ownership/effect facts and operation rules; flatten/unflatten structured arguments with stable paths. | Passes no longer infer these facts from spellings or independent maps; differential type/effect tests cover nested and polymorphic shapes. |
| 4. Scheduled contraction slice | Apply an explicit schedule to contraction/Screma; bring one GEMM family through kernel IR with masks, named operand layouts, and fusible epilogue. | The same algorithm selects scalar, tiled, DPAS/DP4A, and quant leaves by legal schedule; emitted results match the flat semantic oracle. |
| 5. Composable artifacts | Lift executable/session ownership, add foreign `DeviceArray` rebinding, graph linking/instantiation, parameter capture, events, and N-D trees. | Two compiled transformer subgraphs compose with zero host transfers; a full forward/VJP/update step is one resident artifact. |
| 6. Closed tuning loop | Connect real device timing to schedule candidates and graph choices; make cache/provenance complete; wire or explicitly classify every schedule axis. | A cold compile tunes once, a warm compile reproduces the winner, numerical validation precedes timing, and the artifact explains why it won. |
| 7. Workload-driven coverage | Add stream/hist/gather/scatter, masking/block views, atomics, layouts/staging, and quant formats as demanded by model/scientific kernels. | Coverage is measured through production lowering on a published workload corpus, with differential and device compile/run gates. |
| 8. Distributed IR | Add mesh/sharding abstract values, collective nodes, communication cost, and placement scheduling. | A transformer training step runs data-parallel on multiple devices without hidden host copies and reports compute, communication, and memory costs. |

The first two increments are correctness infrastructure and should precede
additional backend breadth. Increment 5 is the first major model-execution
unlock. Increment 6 makes the hardware-aware work a closed system instead of a
collection of useful components.

## 9. Scientific and model acceptance ladder

Every tier compares semantics first and performance second:

1. generated small map/reduce/scan/stencil programs across interpreter, JVM,
   C/SIMD, and available GPU backends;
2. reductions with nontrivial identities, ragged sizes, aliases, multiple
   outputs, and exceptional values;
3. contractions over permutations, batching, split-K, transposes, layouts,
   epilogues, and supported quant formats;
4. FFT/PDE/sparse or irregular scientific kernels that force the non-GEMM
   coverage;
5. one transformer layer forward, VJP, optimizer update, and mixed precision;
6. `pretrained-rstr` prefill and decode with KV residency and logits parity;
7. `finetune-rstr` batched SFT/LoRA/QLoRA with a wholly resident train step;
8. full model execution and training, then multi-device versions.

For each tier record numerical error, kernel count, allocations, peak memory,
host/device and device/device bytes, compile time, tuning time, steady-state
time, and achieved bandwidth/compute. Comparisons with JAX/XLA, Triton,
`llama.cpp-new`, and vendor libraries must use matched dtypes, shapes, warmup,
transfer boundaries, and numerical modes.

## 10. Non-goals and anti-drift rules

- Do not port an entire reference compiler.
- Do not adopt native MLIR merely to obtain terminology or pass structure.
- Do not add an optimization knob without its schema, legality, resource,
  emission, measurement, and explanation path.
- Do not count isolated emitter tests as feature coverage.
- Do not infer semantics from mangled names, source spelling, or exception
  fallbacks.
- Do not create a second operation-property registry.
- Do not let tuning change numerical mode without making that mode part of the
  schedule, cache key, validation oracle, and returned artifact.
- Do not add distributed side protocols that are invisible to the value and
  effect model.
- Do not let self-modifying or learned components bypass the trusted verifier.

## 11. Reference mapping and study snapshot

| Reference | Borrow | Do not borrow |
|---|---|---|
| Futhark | typed SOAC semantics, nested distribution, balance/regularity legality, first-class SegOps and results | Haskell representation or full uniqueness system |
| Halide | algorithm/schedule separation, explicit schedule transformations, structural search states and hard resource filters | a separate DSL or every scheduling primitive before a workload needs it |
| Triton | staged high/kernel/target IR, explicit layouts and masks, broad kernel primitives, autotune protocol and failure handling | Python block-program frontend or a mandatory Triton backend dependency |
| `llama.cpp-new` | quant layout oracle, golden formats, model-driven kernel cases | one bespoke compiler architecture per quant format |
| JAX/XLA | abstract values, primitive transformation rules, effects, pytrees, donation, sharding contracts, layout constraints | Python retracing behavior or outsourcing Raster's programming model to XLA |
| MLIR | conversion legality, operation interfaces/traits, pass invalidation, Transform and DataLayout discipline | compulsory native MLIR integration before a concrete backend requires it |
| Mojo/MAX | one language spanning host and device, parametric low-level GPU libraries, explicit layouts/pipelines, heterogeneous cross-compilation | coupling Raster's semantics to a closed compiler stack or requiring imperative kernels as the main abstraction |
| TVM Relax/Disco | graph-level tensor distribution, SPMD worker sessions, explicit collective runtime boundary | making a controller/runtime protocol the distributed semantic IR |
| Legion/DaCe | correctness-independent mapping, topology-aware task/data placement, explicit data movement and transformations | adopting a second user programming model before Raster's value and schedule IRs are complete |

Local source revisions studied for this direction:

```text
futhark          f05e40c0f
Halide           2fad88f4c
triton           434aecbe9
llama.cpp-new    5c7c22c
jax              d36f6b3b7
xla              2aa87bb
mlir-reference   edff73e1
pretrained-rstr  3caf8a1
finetune-rstr    9e9ba5d
```

The durable decision is not any one revision's class hierarchy. It is the
separation of semantic program, transform/schedule, verified kernel, target
lowering, and composable artifact—with stable identities and measurements
connecting them.
