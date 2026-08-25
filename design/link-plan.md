# LinkPlan: validated resident program composition

`LinkPlan` is Raster's public, backend-neutral boundary for composing compiled resident program
descriptors without decoding kernel names or copying intermediate values through the host.

The boundary has two layers:

- `raster.compiler.ir.link-plan` contains immutable compiler values and pure validation;
- `raster.gpu.link` instantiates a validated plan into resident allocations and one replay graph.

No attention, GEMM, quantization, or model-layer convention exists in the linker. A descriptor step
may select a `KernelArtifact` or `KernelGraph`; both pass through the common executable-step binder.

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
- A single `Compiled` value has not yet been rebased onto `LinkPlan`; this is the next convergence
  step after the pretrained decoder uses the semantic boundary.

For paged attention, page allocation, cache relocation, transactional publication, batching and
request scheduling remain runtime responsibilities. Raster receives borrowed/external node views
and composes routed attention or paged append descriptors exactly like any other semantic program.
