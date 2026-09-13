# Distributed materialization and execution boundary

Status: reviewed implementation plan, not an implemented execution contract.

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
Only copy-mode halos create replica placements. Combining halo transfers must use their certified
reduction over the derived owned target face in global coordinates, not an absent ghost replica;
they must never be silently implemented as a copy. The initial copy-mode vertical rejects these
until its runtime can execute the stated reduction.

## Readiness is separate from geometry

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

Lower dependencies to the existing ExecutionPlan logical queues/events and reuse the LinkPlan
executable binder. Keep a synchronous executor as an explicit backend capability, not an implicit
claim of asynchronous transport overlap. MPI/UCX/NCCL and storage/provider completions remain
realizations of the scheduling boundary, not native handles inside the semantic IR.

## Laptop acceptance and landing order

The current DistributedPlan enforces one shard of a value per mesh device and requires distinct
transfer endpoints with nonempty routes. A real co-located two-shard execution therefore needs an
explicit local-copy lowering or a certified logical-to-runtime placement map. Do not silently map
invented topology devices to the same `:ze:0` target. Local copies must occupy stated resources and
have explicit cost/capability evidence; an empty route must not accidentally mean free transport.

Land geometry/materialization normalization and validation first, then producer/freshness and
transfer endpoint checks, then device-scoped allocation and event execution. Extend the existing
unequal-shard heat and mapped-checkpoint acceptance to exercise that actual DAG. Only after this
integration should numerical coarse/fine operators be counted as an AMR execution vertical.
The full campaign still includes external model training validation and numerical AMR; none of
these structural design requirements substitutes for those numerical workload gates.
