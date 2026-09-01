# Durable numerical state

Status: initial compiler contract and runtime direction, 2026-08-31.

Raster needs durable state for model parameters and optimizer state, KV continuations, PDE fields,
multilevel meshes, sampled agents, compiler measurements, and reproducible branches of a running
workflow. This is not a reason to make the numerical hot path transact through a database. It is a
reason to separate three planes and connect them with explicit lifecycle events.

```text
Datahike semantic/control plane
  state identity, parents, logical time, ownership, leases, decisions, provenance
                     |
                     | publish only after durable receipts
                     v
immutable numerical content plane
  local Boring/Konserve files or LMDB frontend <-> S3-compatible authoritative tier
                     |
                     | scoped local segment / asynchronous promotion
                     v
Raster resident compute plane
  BufferViews, block gather/scatter, device queues/events, direct peer communication
```

Hot node-to-node communication belongs to MPI, UCX, NCCL/RCCL, oneCCL, RDMA or another direct
transport selected from a certified `DistributedPlan`. Datahike records topology, intent, durable
state lineage, measurements and completed decisions. It must not relay halo cells, gradients or
activations.

## Identities and publication

Three identities must remain distinct:

- A **state identity** names a semantic checkpoint or branch, such as simulation time 1200 s or
  optimizer step 40. It has zero or more parents.
- A **content address** names immutable chunk content. Different states may reuse it.
- A **placement** says where a realization is available: local file, LMDB key, S3 object, host
  memory, GPU allocation, or a replica on another worker.

`NumericalStateManifest` is the storage-neutral compiler/runtime boundary. Version 1 contains
complete, regular, rectangular chunks for dense `AbstractValue` tensors. It certifies dtype-derived
logical byte counts, clipped edge cells, canonical grid order, content addresses, explicit byte
order, numerical compatibility, producer provenance and lineage. It does not contain paths,
buckets, object keys, mmap segments, buffers, queues, devices or store handles.

Publication is immutable and ordered:

1. gather changed resident blocks into bounded staging or writable local mapped files;
2. seal each chunk, compute and verify its content address;
3. write it to the local frontend and start durable backend promotion;
4. await the backend durability receipts required by policy;
5. publish one complete manifest/state transaction to Datahike;
6. release staging allocations and eventually garbage-collect unrooted content.

S3 has no transaction spanning many objects. Immutable chunks followed by one manifest commit give
readers an atomic semantic publication point. An S3 ETag is not a content identity, particularly
for multipart objects; Raster must retain its own digest.

## Konserve, mmap, LMDB and S3

The practical first composition is a Konserve tiered store with a local file frontend and an
S3-compatible authoritative backend. `konserve.mmap` can expose an uncompressed, unencrypted Boring
value in the local file store. A missing remote object must first be promoted to that frontend;
object storage itself cannot be memory-mapped.

This supports a useful restore path:

```text
S3 object -> local immutable file -> scoped MemorySegment -> Raster async upload/scatter -> GPU
```

The final arrow can avoid an intermediate JVM primitive array. The S3 arrow is not currently
zero-copy: the inspected `konserve-s3` implementation materializes reads with `readAllBytes` and
writes with `ByteArrayOutputStream`/`RequestBody.fromBytes`. Before using it for multi-GiB fields,
the storage layer needs byte-preserving streaming promotion, ranged/file downloads, multipart/file
uploads, bounded verification, and cancellation/backpressure. Tier promotion should not require
decoding and re-encoding a large numerical value.

`konserve-lmdb` is a useful alternative local frontend for many small or medium immutable chunks,
ordered scans, MVCC metadata and fast lookup. Its zero-copy segment is valid only inside the LMDB
read transaction that owns it. Long read transactions pin pages and delay reclamation; one writer
is serialized; map size and version garbage collection require operations policy; and the database
must not live on a network filesystem. A file mmap is usually the safer staging source for a long
asynchronous GPU transfer because its lifetime can be pinned without pinning an LMDB transaction.

The common runtime capability should consequently be scoped, not a naked segment:

```clojure
(with-local-content content-address
  (fn [{:keys [segment offset byte-length release]}]
    ... submit transfer ... retain until event ...))
```

File-backed Konserve and LMDB can implement that capability with different lifetime rules. Remote
stores implement promotion to a local provider, not a fictitious remote mmap.

`raster.runtime.numerical-content` now defines this seam as capability-described storage tiers,
provider-owned promotion/localization events, and an `AutoCloseable` `LocalContentLease`. A retained
Raster range-transfer event may take ownership of one or more leases after successful submission;
completion or session shutdown releases them exactly once. Failed validation leaves ownership with
the caller. This supports today's staging implementations and a future borrowed direct-DMA path
without changing the manifest or weakening mmap/LMDB lifetime rules. A nonblocking completion poll
does not release the lease: cancellation stops admission of new work, then `release-event!` waits at
the current transfer boundary and consumes the event and lease together.

In-place mmap editing is for ephemeral working copies. Mutating an object already published under a
content address violates snapshot immutability. Same-sized Boring edits may dirty only touched
pages, which is valuable while constructing the next chunk; the result must then be sealed under a
new address. Size-changing edits can move the tail and are unsuitable for large field hot paths.

## Deployment profiles

The contract is realistic for both institutional HPC and frontier-model clusters only if S3 is one
placement capability, not the semantic storage model:

- CERN publicly documents EOS as its disk namespace/buffer in front of CTA tape, accessed through
  XRootD or HTTP. A deployment there should implement an EOS/file promotion adapter and preserve
  CTA's existing lifecycle rather than require an S3 gateway.
- Jülich's JUST publicly documents tiered IBM Spectrum Scale/GPFS storage mounted by its compute
  systems, plus an S3 object service on JUDAC. The same manifest could therefore be realized through
  GPFS for the active campaign and S3 or tape-oriented policy for durable retention.
- LANL's public MarFS material describes a near-POSIX namespace over scalable object/file data
  stores, deployed alongside Lustre scratch and HPSS archive. That is structurally close to the
  content/placement separation here, including the need for parallel bulk movers.
- Cloud AI guidance pairs durable S3 with FSx for Lustre, local NVMe and a high-performance fabric.
  Checkpoints are written asynchronously and distributed hierarchically rather than independently
  downloaded by every accelerator worker. Public Anthropic material also confirms that frontier
  workloads span Trainium, TPUs and NVIDIA GPUs, so neither the device nor storage runtime can be
  assumed from the programming model.

Ceph is an attractive on-premises realization because one RADOS cluster can expose S3-compatible
RGW, CephFS and block devices. It also introduces a substantial operational system, correlated
failure/rebalancing domains and an S3 compatibility subset. Raster/Konserve should be able to use
it, but should not require it. Similar adapters can target MinIO, EOS/XRootD, Spectrum Scale,
Lustre, MarFS, DAOS or plain local files.

The runtime capability set should be tested rather than inferred from a product name: immutable
put, conditional publication, range read, multipart/streaming write, localize, scoped mmap, durable
receipt, checksum verification, listing/GC, and observed bandwidth/latency. A parallel filesystem
may implement `localize` as an already-mounted path; an object store stages into a node/rack cache;
an archive may return a delayed recall operation. This keeps the compiler and manifest stable while
the workload planner selects a suitable path for the current allocation.

## Relationship to scientific formats

Raster should provide adapters rather than claim that an EDN/object store replaces the scientific
data ecosystem.

| Format/system | Strong fit | Missing or awkward for Raster's goal |
|---|---|---|
| Boring + Konserve | Rich Clojure metadata, immutable values, local mmap navigation, tiering, natural Datahike references | No standard N-D coordinate/chunk schema; current S3 backend lacks bulk streaming/range paths |
| Raw slabs + small manifest | Minimal overhead, direct mmap and GPU transfer, simple checksums | Raster owns schema evolution, endian/layout rules, partial tiles and interoperability |
| Zarr v3 | Cloud-native chunked N-D arrays, sharding, broad Python/scientific tooling | Many-object overhead; weak cross-array snapshot/branch transaction and provenance semantics |
| HDF5 / NetCDF | Mature self-describing arrays, shared-filesystem and MPI-IO ecosystem | Monolithic files and concurrent/object-store mutation are awkward; not content-addressed lineage |
| ADIOS2 BP5/SST | HPC checkpoints, streaming and in-situ workflows, strong MPI integration | External native runtime; not a durable queryable semantic state graph |
| TileDB | Dense/sparse arrays, object storage, fragments and time travel | Heavyweight competing array/catalog/runtime model |
| TensorStore | Asynchronous N-D slicing and index transforms over cloud formats | Native C++ integration; not a provenance or workflow state store |
| Arrow | Excellent columnar interchange and analytics | Not an N-D checkpoint, halo or multilevel field format |
| safetensors | Immutable mmap-friendly tensor offsets and model weight interchange | Little chunk hierarchy, coordinates, sparse/AMR structure or branching |

Zarr import/export is the best early interoperability target for chunked simulation fields.
ADIOS2 is the strongest reference or adapter for MPI/in-situ checkpoint bandwidth. safetensors is a
useful model-weight boundary. Internally, Raster still benefits from content-addressed chunks and a
Datahike lineage graph because none of those formats supplies the whole programming model.

## Correctness and operational hazards

- Compression and encryption prevent direct mmap interpretation. Keep the hot local tier raw;
  allow compressed cold chunks only when the restore plan includes decode resources and cost.
- Chunk size trades per-object/API overhead against read amplification, retry cost and staging
  memory. It is a schedule/cost parameter, not a universal constant.
- Dtype, endian, layout, coordinate system, partial-edge shape, codec and checksum domain must be
  explicit. Shape and dtype alone do not establish restore compatibility.
- Async transfers must retain the mapped file or transaction until the event completes, unless the
  GPU backend has made an owned staging copy.
- Immutable versioning requires roots, leases and garbage collection. A state transaction may be
  removed only after no retained branch, running plan, replica or publication attempt can reach it.
- Exact restart also needs RNG state, input/dataset cursor, solver/controller state and numerical
  mode. Reconstructing tensor bytes alone is not reproducibility.
- Direct communication and durable checkpoint traffic contend for PCIe/NIC/storage bandwidth. The
  outer workload planner must schedule both, even though only the former is a collective/halo step.

## Adaptive mesh composition

`AMRWorkloadPlan` is the first checked composition of the three planes for adaptive scientific
workloads. Its `RefinementHierarchy` owns semantic level and patch coordinates. Every patch binds
one complete `NumericalStateManifest` field to one fully owned value in a certified
`DistributedPlan`. Prolongation and restriction name exact aligned rectangles, compatible storage
contracts, and explicit operator requirements; cross-device plans derive transfer byte counts from
the source field dtype and retain field identities, access roles, and regions on the generated
transfer and compute steps. The resulting AMR certificate embeds the complete durable-state
certificate, hierarchy witness, operation expansions, routes, and distributed certificate/cost
vector, so reusing a manifest identity cannot conceal changed content.

The hierarchy does not contain content placements, store handles, buffers, communicators, or
events. Conversely, the distributed plan does not reconstruct refinement meaning from transfer
attributes. This outer composition allows Datahike to version a simulation state and selected
certified plan while direct links carry hot patch data and Konserve-compatible tiers retain
immutable checkpoint chunks.

Version 1 is deliberately limited to cell-centred rectangular patches, adjacent-level operations,
and one owned distributed value per patch. Its schema and `:hierarchy-only`/`:transfer-cycle` mode
are load-bearing. A transfer cycle requires exactly one full-patch prolongation and restriction per
refined patch. It proves alignment, non-overlap, a strict one-parent proper-nesting margin (without
a physical-boundary exemption), durable coordinate identity, exact shard binding, and planning/
certificate agreement. Declared operator requirements are not yet proofs of an executable numeric
kernel. Conservative hyperbolic solvers require typed operator bodies, explicit time refinement,
flux-register accumulation, reflux, and average-down ordering before Raster may claim a complete
AMR step. The next demonstrator should make those contracts load-bearing in a 2-D shallow-water or
finite-volume workload and checkpoint only changed patches.

## Workload-driven landing order

1. Ratchet the existing RK4/PDE numerical oracle and remove its broad compound-kernel tolerance.
2. Land and use the certified storage-neutral `NumericalStateManifest`.
3. Define scoped local-content and asynchronous promotion receipts; implement local immutable file
   mmap first, then LMDB callbacks and S3 promotion.
4. Replace whole-object heap S3 operations with streaming/range/multipart paths and byte-preserving
   tier synchronization.
5. Bind manifest chunks to existing `ResidentBufferView`, async transfer and block gather/scatter
   contracts. KV cache state becomes one use case, not a separate memory ABI.
6. Generalize TypedSOAC stencil/index-space semantics to N-D neighborhoods and derive periodic,
   multidimensional halo regions in `DistributedPlan`.
7. Demonstrate a conservative 2-D shallow-water workload: mass/lake-at-rest/convergence oracles,
   chunk checkpoint, exact restore, branch to a changed parameter or boundary, and simulated
   multi-device halo overlap. Run local mmap in the hot loop; make MinIO/S3 an optional slow gate.
8. Extend the landed AMR hierarchy and typed prolongation/restriction schedule with subcycling and
   reflux contracts; checkpoint only changed patches. Validate the same state machinery with a
   transformer training checkpoint containing parameters, optimizer, RNG and data cursor.

The laptop/CI loop remains hardware-independent: semantic certification, restore compatibility,
storage fault injection, the distributed simulator and small numerical oracles run locally.
Cloud GPUs, a real MPI fabric and object storage provide periodic performance acceptance, not a
required compiler-development loop.
