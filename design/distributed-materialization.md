# Distributed materialization and execution boundary

Status: owned-domain normalization, copy-replica geometry/coverage and exact boundary
provider bindings, contiguous halo endpoint projection and conditional DAG readiness are implemented.
A first synchronous, owning GPU executor now realizes sources and runs checked DAGs on actual
device identities. Co-located logical workers, asynchronous transports and reusable execution
epochs remain implementation work.

The current `distributed-plan/compute-bindings` accepts an explicit `:local-shape` with one owned
placement and optional copy replicas that together cover the entire local domain. Its report retains the original ABI leaf views and
adds a checked `:domain` with the reshaped view and owned placement. A flat `[6]` leaf can explicitly
realize a `[2 3]` shard. Copy replicas can now complete a padded domain: their destination rectangles
are derived from validated ScheduledHalo steps, anchored at the owned region, and checked for exact
coordinate-disjoint coverage. Each transfer must precede its consumer through the DAG, and repeated
consumers must agree on its physical destination. Periodic and nonperiodic halos now have structural
examples. A nonperiodic boundary names a bound preceding compute step and its ABI-written local
value declared as a public LinkPlan output; an internal/scratch write is not retained-output
evidence. The complete plain dense physical view must match the boundary region exactly. No implicit
fill is inferred from the boundary mode. Strided boundary outputs and selecting only part of a
producer's larger value remain refused until their write-region projection is certified.
An explicitly referenced boundary output is retained in the producing entry's `:boundary-outputs`
report rather than requiring a synthetic global ValueShard. Other required public values still
need global bindings. The boundary report records ABI write scope, not definite initialization of
every cell; that stronger obligation belongs to the execution-readiness proof below.
These checks do not prove initialization, freshness across intervening writes, or source endpoint
availability. The report is still structural and must not be treated as execution authorization.
Repeated owned realizations compare ordered physical regions: a contiguous ABI reshape preserves
identity when allocation, dtype, byte range, and field packing agree. Explicit domains use their
derived owned view; noncontiguous leaves retain their shape/stride mapping. Original ABI views
remain in the binding report and in the conservative cross-shard alias checks.

The current whole-shard binding deliberately requires one local LinkValue to realize one exact
owned shard. A halo-padded field cannot satisfy that contract by selecting only its owned cells:
the compiled kernel still reads the whole local buffer. Its neighbor replicas and physical-domain
boundary cells need explicit provenance as well. This extends the current compiler vertical;
it does not introduce another kernel ABI, quantization registry, or storage provider.

## One local domain, several placements

Keep the current `{ :value global-value :shard shard-id }` whole-shard spelling as a canonical
single-owned placement. The new spelling describes the full local materialization:

```clojure
{:local-shape [4 7]
 :placements
 [{:kind :owned :value :u :shard :left :local-offsets [1 0]}
  {:kind :replica :transfer :receive-right-face}
  {:kind :boundary :region {:offsets [0 0] :shape [1 7]}
   :provider {:step :initialize-left-boundary :local-value :boundary-out}}]}
```

The example's identifiers are schematic. Replica geometry comes from the referenced scheduled
transfer, not from duplicated user-provided source/shard coordinates. Boundary producers must be
verified executable writes or explicitly retained initialization evidence; a boundary-mode label
such as `:nonperiodic` is not a numerical zero-fill operation.

The first numerical implementation supports a single plain dense leaf. Check the declared
`local-shape` against the entire LinkValue/leaf element count, preserve dtype and allocation, and
construct an ordinary dense BufferView before selecting rectangles. Current heat LinkPlans carry
flat `[n]` arrays: do not infer a two-dimensional domain from scalar parameter names, and do not
confuse the acceptance test's separately constructed rectangular view with the compiled ABI view.
Composite/packed layouts keep their existing whole-shard route until their physical layout has a
certified region projection. Do not add per-quantization addressing knowledge to memory planning.
Enforce this restriction in validation: one plain dense injective leaf, equal element volume, and
no unresolved logical layout. Coordinate-disjoint regions are not physically disjoint for arbitrary
zero/overlapping strides. A boundary provider must resolve an exact local value and producing
region, with ABI-derived write evidence and physical subview correspondence, not merely a step ID.

For every local materialization:

- Derive owned shape from ValueShard and check its local offsets.
- Derive replica source/global and target-local regions from ScheduledHalo plus the owned anchor.
- Project every region with `BufferView/rectangular-subview`; check disjoint logical coordinate
  coverage of the whole local domain for any read/read-write ABI access.
- Keep ABI-derived access on the whole LinkValue. Owned publication is the owned region only;
  writes to ghost storage remain private unless an explicit combining operation publishes them.
- Include private aliases in access/lifetime reasoning. The current fail-closed bound/private
  overlap restriction must not be relaxed without replacement evidence.

## Identity and endpoint resolution

An owned realization remains keyed by `[device global-value shard-id]`, comparing its derived
owned subviews rather than the entire padded base. An incoming replica is
keyed by `[target-device transfer-step-id]`, not by its source's owned-shard key. Source and target
therefore may have different allocations, while repeated references to one materialization must
agree on its actual views. Transfer identity also distinguishes refreshed replicas across unrolled
time steps; the same backing allocation alone is not freshness evidence.

Resolve transfer endpoints through these placement identities. Avoid a second handwritten list
of ABI arguments or duplicate source/destination rectangle declarations. If the source owned
realization cannot be identified uniquely, or the target replica is absent, lowering must refuse
execution even when the topology-only plan remains useful for simulation.
`distributed-plan/transfer-bindings` now performs this strict geometric projection for every
transfer in a plan. It indexes owned sources by device/value/shard and replica destinations by
device/transfer, derives each source rectangle relative to its owned shard, and returns the existing
BufferView records with exact shape/dtype/byte agreement. Analytical plans are still accepted by
the ordinary planner when endpoints are absent; requesting strict projection rejects them.
The current projector supports only contiguous plain ScheduledHalo copies. Generic transfers,
combining transfers, and strided faces require additional lowerings and are rejected. This report
is not an execution certificate: it does not establish source initialization/freshness, actual
allocation sharing, event completion, or a usable transport implementation.
Only copy-mode halos create replica placements. Combining halo transfers must use their certified
reduction over the derived owned target face in global coordinates, not an absent ghost replica;
they must never be silently implemented as a copy. The initial copy-mode vertical rejects these
until its runtime can execute the stated reduction.

## Readiness is separate from geometry

`link-plan/initialization-contract` now exposes the shared effect validator's node-level
`:requires`, `:initializers`, `:produces`, `:reads`, `:writes`, and public `:outputs` sets.
Caller requirements arise from reads or pass-through outputs before a proven local writer;
alias requirements retain the actual requested subview. Declared source initializers are separate
from caller obligations and are not immutable snapshots or completed uploads. Produced storage
does not imply retained semantic identity for private temporary values. These are conditional
local facts; distributed lowering still needs to discharge them and establish freshness.
The facts are conservative, not minimal across all aliases. The `gpu.link` runtime now derives
owned `pending-inputs` from this same `:requires` set, so unused inputs and state fully produced
before reading do not demand redundant uploads. Pass-through outputs still require initialization.
Borrowed/external buffers retain their existing caller-initialized contract; importing a buffer
does not itself prove completion of a distributed producer. A distributed executor must discharge
those external preconditions through actual completed events, not bypass `run!` input checks.

Descriptor cacheable transforms can execute during graph recording. Runtime constant-role
admission therefore also requires captured data (an owned source or caller-ready borrowed/external
storage) with no overlapping local write. A late-uploaded owned constant is bound as a replay
input instead: its initialization gate remains active and its transform is not run prematurely.
Captured weights keep the existing one-time transform path. This does not authorize changing
captured constant contents after instantiation without rebuilding their derived transforms.

The proof reuses LinkPlan's ordered instance access facts and its existing
`produced-views`/`partial-writes` accounting. `value-accesses` deliberately summarizes ABI access
only; a `:write` entry is not an initialization postcondition. LinkPlan currently assumes caller
input/constant/state nodes are initialized when checking the local program. Distributed lowering
must discharge those caller preconditions using source leases and completed producers/transfers,
not inherit the assumption as evidence. The shared calculation is in the existing validator,
not a second kernel-effect registry.

Before an executable plan can allocate resources, prove:

1. Every compute step has a local executable binding, and every transfer has exact projected views.
2. Every read has an initialized owned region, completed incoming transfer, or completed boundary
   producer. Initial host data is evidence only until an overlapping physical write invalidates it;
   disjoint regional writes do not invalidate one another. Whole-buffer ABI effects remain
   conservative unless a narrower physical write region is actually proven.
3. Producers precede consumers through the DAG's actual dependencies/events. Vector order and
   analytical duration estimates alone are not completion events.
4. Writes and overlapping transfer ranges cannot race; refreshed replicas invalidate old readiness.
5. Allocation sharing follows device-scoped materialization identities, with release after all
   dependent operations complete. Existing separately instantiated owned LinkPlans do not do this.

Physical interval operations for the initial contiguous readiness proof now live in BufferView:
`subtract-contiguous` preserves unaffected typed fragments of an initialization fact after a write;
`covered-contiguous?` checks whether several compatible views jointly cover a required range.
They use device-scoped allocation identity, reject contradictory allocation contracts, and do not
mistake strided bounding spans for dense coverage. Invalidation is independent of the writer's dtype
when the cut is element-aligned; typed initialization coverage requires a matching dtype. A cut
through part of an element is rejected until a byte-validity or outward-rounded invalidation
policy exists. These helpers establish geometry only; the DAG checker must retain producer identity
on each fragment and reject unordered conflicting effects and stale replicas.
The existing `overlaps?` and `same-range?` predicates use the same device-scoped identity;
zero-byte views never overlap a nonempty range, including when their offset lies inside it.

`distributed-plan/check-readiness` now combines the local contracts, strict copy endpoints and
region operations into a conditional DAG check. Every compute must bind an executable local plan.
The checker rejects unordered read/write conflicts (including private local effect scopes),
requires complete typed coverage before a read, and checks that replica/boundary regions still
originate from their declared transfer/provider. A later writer removes that provenance while
preserving disjoint fragments. Being initialized is therefore not enough to satisfy a stale replica.

The returned `:initializers` are explicit obligations: all declared host sources must be realized
once before the DAG, with no implicit later reuploads. Overlapping initializers must identify the
same source object and exact view; this is deliberately conservative and is not source snapshot
certification. Initializer scopes are currently required at each local invocation conservatively.
`:actions` records required/read/write/produced scopes, and `:final-regions` retains producer IDs.
This is not a runnable distributed executable or a full ownership/semantic-equivalence certificate.
Allocation sharing, private-storage lifetime isolation, runtime pending-input reconciliation and
actual event/transport completion remain required. Unknown narrow write footprints retain whole-ABI
invalidation; a produced prefix alone is not substituted for a certified write footprint.

Lower dependencies to the existing ExecutionPlan logical queues/events and reuse the LinkPlan
executable binder. Keep a synchronous executor as an explicit backend capability, not an implicit
claim of asynchronous transport overlap. MPI/UCX/NCCL and storage/provider completions remain
realizations of the scheduling boundary, not native handles inside the semantic IR.

## Laptop acceptance and landing order

`raster.gpu.distributed/instantiate!` now performs readiness, storage-projection, complete shared
allocation-contract, resident-pool budget and retained-output checks before device contact. It
owns one session per actual target, realizes original allocation options, and uploads sources once.
`run!` interprets the existing ExecutionPlan wait/completion representation synchronously. Each
borrowed local executable is constructed only after its dependencies complete, run, then closed;
this also keeps constant prologues behind producer completion. The owner retains storage until
close. Failed/completed executions cannot replay stale readiness evidence. `output-values` exposes
retained logical values only after successful completion; later overwrites of retained outputs
are rejected during preflight.

Cross-device copies currently require explicit `:transport :host-staged`, with bounded native
staging (default 1 MiB) and synchronous download/upload chunks. This is not P2P, MPI, overlap or
evidence that the topology's modeled link performance was achieved. The pool budget check covers
the retained LinkPlan allocations, not all backend-private compilation/graph temporary memory.
Source stability through initialization and constant immutability remain caller obligations.
The actual DAG acceptance includes two logical workers on one physical GPU, with unequal owned
regions and periodic halo copies over four generated heat epochs. Bounded transfer tails,
allocation conflicts and failure cleanup have hardware-free tests; multi-device runtime execution
is not yet validated by that acceptance.

`link-plan/borrow-owned-storage` is the checked local ownership projection for an enclosing
allocation owner. It preserves executable instances, roles, logical shapes and physical views,
changes owned physical/logical ownership to borrowed, and extracts the original allocation
contracts and host initializers. The projected LinkPlan cannot silently reupload those sources.
Already borrowed/external allocations retain their contracts. The returned initialization facts
describe the original program's obligations; projection itself proves neither initialization nor
lifetime. The enclosing owner must reconcile shared initializers, realize them once, retain buffers
through completion, and release borrower registrations before freeing storage.

A real GPU acceptance now runs two successive generated heat executables over the same owning
session buffers. Closing the first removes only its borrowed registrations; the second continues
resident state without reuploading the original source and matches four CPU reference steps.
This validates the local storage seam, not DistributedPlan execution or cross-device transport.

DistributedPlan keeps shard identity on logical mesh workers. Each worker's device-plan may state
an explicit `:target` shared with other workers; local LinkPlans must agree with that physical
target. Remapped targets require an explicit aggregate `:device-capacities` runtime budget. Pool
accounting deduplicates physical allocation identities rather than granting each worker a separate
copy of the device's capacity. Cross-worker shard aliases are rejected, including immutable weight
sharing until an explicit sharing relation exists. Ordered private scratch with the same physical
allocation identity can be reused, subject to allocation-contract and readiness checks.

`:transport :resident-copy` performs actual same-target buffer copies and rejects cross-target
endpoints before device contact. Logical transfer routes and dependencies remain explicit; this
synchronous executor does not claim modeled route costs, overlap, or free transport. The periodic
heat acceptance checks 16 resident halo copies and six startup uploads with no later host uploads.

Land geometry/materialization normalization and validation first, then producer/freshness and
transfer endpoint checks, then device-scoped allocation and event execution. Extend the existing
unequal-shard heat and mapped-checkpoint acceptance to exercise that actual DAG. Only after this
integration should numerical coarse/fine operators be counted as an AMR execution vertical.
The full campaign still includes external model training validation and numerical AMR; none of
these structural design requirements substitutes for those numerical workload gates.
