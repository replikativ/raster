# LinkPlan: validated resident program composition

`LinkPlan` is Raster's public, backend-neutral boundary for composing compiled resident program
descriptors without decoding kernel names or copying intermediate values through the host.

The boundary has three layers:

- `raster.compiler.ir.link-plan` contains immutable compiler values and pure validation;
- `raster.compiler.ir.resident-plan` certifies the lowering of an ordinary resident descriptor;
- `raster.compiler.ir.link-composition` certifies namespacing and boundary unification across
  independently lowered plans;
- `raster.gpu.link` instantiates a validated plan into resident allocations and one replay graph.

No attention, GEMM, quantization, or model-layer convention exists in the linker. A descriptor step
may select a `KernelArtifact` or `KernelGraph`; both pass through the common executable-step binder.
This is also the production route for certified inclusive scan: the compiler preserves its emitted
multi-kernel graph as one descriptor step, while graph-private block totals remain invisible to the
public LinkPlan value graph and are allocated by the executable binder.

The resident-descriptor conversion is proof-carrying in the compiler sense. It returns a
`CertifiedResidentPlan` containing the target plan and a `ResidentPlanCertificate`. The witness
records the source parameter order, pointer/value identity mapping, scalar specialization, roles,
resolved schedule, target, realized value contracts, aliases, and public outputs. `verify!`
independently re-derives those facts from the target plan and its embedded descriptor. Mutation or
omission therefore fails before runtime contact; later layers never reconstruct this information
from ABI spelling. This certificate proves preservation across this structural lowering boundary;
the earlier obligation that executable kernels implement the source program remains with their own
dialect conversions and numerical oracles.

`CertifiedLinkComposition` applies the same rule recursively. Ordered certified components are
namespaced before explicit output→input connections and equal input/constant sharing canonicalize
node and allocation identities. Its witness records source plan identities, connections, sharing,
node/allocation/instance mappings and outputs. Verification rechecks every source certificate and
re-derives the complete target plan. Composition therefore occurs before allocation: an
intermediate connected by identity cannot acquire an implicit host or device copy at runtime.

## Values

A `LinkNode` identifies one typed, realized `BufferView`, its ownership contract, dataflow role, and
optional host initializer. Distinct nodes may name disjoint or explicitly aliased ranges of one
allocation. Node identity, rather than a decoded ABI name, connects producers and consumers.

A `LinkInstance` binds every pointer-valued compiler symbol in one compiled descriptor to a node
identity. Scalar parameters, resolved schedule data, and residency-role overrides are ordinary
data. Instance order and descriptor step order define the initial serial schedule.

A `LinkPlan` owns the target, node table, ordered instances, ordered public outputs, explicit alias
pairs, and inspectable attributes. Constructing it validates, before any driver contact:

- allocation and view bounds, dtype, shape, stride, device and ownership contracts;
- exact descriptor pointer and scalar environments;
- descriptor scratch sizes and executable ABI storage dtypes;
- selected `KernelGraph` external ranges against its resolved scalar environment;
- producer-before-consumer ordering and read-only roles;
- declared physical overlap and same-step writable alias hazards;
- public output identity and availability.

## Example

```clojure
(require '[raster.compiler.ir.link-plan :as link]
         '[raster.gpu.link :as gpu-link])

(def plan
  (link/make
   {:id :two-layers
    :target :ze:0
    :nodes [(link/node {:id :x :dtype :float :shape [16 64]
                        :device :ze:0 :role :input})
            (link/node {:id :w0 :dtype :float :shape [64 64]
                        :device :ze:0 :role :constant :source w0})
            (link/node {:id :w1 :dtype :float :shape [64 64]
                        :device :ze:0 :role :constant :source w1})
            (link/node {:id :hidden :dtype :float :shape [16 64]
                        :device :ze:0 :role :internal})
            (link/node {:id :out :dtype :float :shape [16 64]
                        :device :ze:0 :role :output})]
    :instances [(link/instance
                 {:id :layer-0 :descriptor linear-descriptor
                  :bindings {'x :x, 'W :w0, 'y :hidden}
                  :scalars {'batch 16, 'in-f 64, 'out-f 64}})
                (link/instance
                 {:id :layer-1 :descriptor linear-descriptor
                  :bindings {'x :hidden, 'W :w1, 'y :out}
                  :scalars {'batch 16, 'in-f 64, 'out-f 64}})]
    :outputs [:out]}))

(with-open [executable (gpu-link/instantiate! plan)]
  (gpu-link/upload! executable :x x)
  (let [resident-outputs (gpu-link/run! executable)]
    ;; No host copy occurred. Download only at an explicit host boundary.
    (gpu-link/download executable :out)))
```

`instantiate!` may instead receive `{:session session}`. In that form the returned executable does
not own the caller's session. Its close releases only the recorded graph, prepared phases, private
kernel storage, materialized view handles, and allocation registrations that it created. Borrowed
or external plan allocations are supplied as `{allocation-id DeviceBuffer}` through
`:external-buffers` and are never freed by Raster.

Allocation ownership is deliberately independent of value readiness. Importing external storage
does not initialize an `:internal` or `:output` node; it still needs an ordered producer. Declare a
caller-supplied value as `:input`, `:constant`, or `:state` so validation records that semantic
boundary explicitly.

`raster.compiler.ir.link-module-plan` is a validation-only projection for several named entry
points over one canonical LinkPlan storage spine. The spine has no global outputs, and its named
entries form a complete non-empty contiguous partition of the ordered instances. Each validation
freshly rederives strict conditional read/write and complete-production facts from sealed LinkPlan
evidence. Legacy write permission and partial mutation preserve initialized bytes but cannot
initialize a retained value; only semantic complete writes and certified produced views can do so.
Cross-entry frontiers are whole logical values with one unique earlier complete producer, and
dense complete child views may jointly cover a parent view.

Every named entry passed to `link-module-plan/make` must now declare an exact ordered interface of
whole logical LinkValue identities: `{:inputs [...] :state [...] :outputs [...]}`. The former
omitted-interface form refuses. Input membership is freshly derived from resident reads and
state membership from resident reads or writes, using physical view overlap rather than node-name
coincidence; their caller-declared order is preserved. Outputs are caller intent, but each must use
output-role storage and be semantically complete-produced by that entry. Constants are implicit:
in particular, an owned constant without a host source remains an external initialization premise,
not ready storage. Scalars, certificates, partial logical values, and ambiguous cross-value aliases
are not interface fields. A retained frontier may also be a declared producer output; this creates
no second buffer or implicit lease.

This prerequisite declares resident entry I/O but does not yet execute entries or establish
sessions, generations, leases, replay, completion, failure recovery, or allocation lifetime. Host
source contents and native handles are live inputs, not fingerprinted certificate data. A later
runtime owner must realize those contracts without treating the validation-only spine as an eager
all-entry schedule.

Owned `:input`/`:constant`/`:state` nodes without a plan initializer are tracked as pending. Replay
fails until `gpu-link/upload!` initializes them. This makes input and first-use state readiness
explicit while still allowing pretrained runtimes to allocate first and publish or upload later.

## Current deliberate limits

- Runtime kernel pointer bindings are contiguous views. Non-contiguous logical views remain valid
  compiler values but must receive an explicit layout/materialization schedule before linking.
- One logical LinkNode represents one flat buffer value. Multi-slot SoA arguments need a future
  composite abstract value rather than an implicit tuple convention.
- The initial dependency schedule is serial. Queue/event partitioning can lower from the same
  proven effects later without changing node or instance identity.
- `raster.gpu.compiled/lower` returns an allocation-free `Prepared` value. Independently lowered
  artifacts compose through semantic input/output keys and only the final Prepared value is passed
  to `instantiate!`; `compile` remains the single-program lower+instantiate convenience.
- Composition currently rejects endpoints whose allocation contains several views and rejects
  cross-component donation. Ranged view unification and ownership threading belong to the next
  `AbstractValue`/view consolidation rather than an implicit whole-buffer convention.

For paged attention, page allocation, cache relocation, transactional publication, batching and
request scheduling remain runtime responsibilities. Raster receives borrowed/external node views
and composes routed attention or paged append descriptors exactly like any other semantic program.
When such a runtime replaces an interval of a compiled descriptor,
`raster.compiler.ir.program-stage` selects and validates that interval from declared state/output
effects and projects its before/selected/after descriptors. Those projections remain ordinary
`LinkInstance` inputs; the linker still contains no attention-specific path.
