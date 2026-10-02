# Durable numerical state

Status: compiler/runtime contracts and strict semantic restore gates, 2026-10-02. Production
publication, compiler-derived provenance and distributed durability acceptance remain open.

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
order, and binds declared numerical compatibility, producer provenance and lineage. It does not contain paths,
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

Manifest certification checks geometry and declared addresses, not bytes returned by a provider.
`raster.runtime.numerical-content/verify-chunk-lease!` checks the exact stored extent and SHA-256
over a scoped local lease before restore or upload. It hashes in bounded blocks rather than turning
a large mapped chunk into an int-indexed `ByteBuffer` or heap array. The caller still owns the lease
and must retain it through any asynchronous transfer. Provider adapters and publication paths must
call this verifier (or supply a separately certified equivalent); merely opening a lease is not a
digest check. The mapped PDE restart oracle now uses this shared boundary and injects corruption
and truncation faults.

## Restore compatibility

A valid manifest/certificate and correct chunk bytes can still describe the wrong target.
`numerical-state/restore-contract` declares the intended continuation's complete field set,
AbstractValues, coordinate spaces, logical step/stage, numerical policy and producer provenance.
`verify-restore!` independently verifies the incoming certificate and compares that expectation
before opening leases or uploading bytes. It returns no new authority or migration token.

Field IDs, canonical raw-storage facets, concrete shapes, coordinate spaces and the complete
logical/numerical/provenance contracts must agree. Field order, chunk partition, content identity,
storage placement, memory ownership and device placement do not imply or prevent semantic
compatibility. The expectation must come from the intended continuation, not blindly from the
incoming manifest. Omitted application semantics are not inferred; different program/numerical
contracts require an explicit future migration rather than an implicit relaxed comparison.
Target AbstractValues are projected onto shared logical storage facets and shape: runtime device
handles and allocator attributes are discarded, not serialized into the restore expectation.

Manifest and restore metadata must be supported pure canonical data, including extension maps.
Certificates use the shared bit-preserving comparison: signed-zero changes reject, same-payload
NaNs survive independent copies, and different payloads reject. This tightens previously permissive
metadata validation; it changes no surface numerical operation. Byte verification, stored-format
decoding, byte-order conversion and asynchronous lease retention remain separate obligations.
No complete persistence codec or mathematical program-equivalence proof is claimed.

The same-extent/different-cell-order PDE checkpoint is independently valid but rejects against
the wrong target layout before leasing. The correct target resumes from actual mapped bytes on
OpenCL and Level Zero. Affected state, restart, lease and AMR suites pass 40 tests / 454 assertions.
The fixture's producer fingerprint remains an explicit fixture label, not compiler-build evidence.

### Producer evidence: existing owners, distinct identities

An argument-independent template fingerprint is not a checkpoint producer identity. Template
lookup excludes concrete shapes, scalar arguments, array contents, roles and composition wiring.
Successful equation-first preparation reports now retain the existing four-part persistent artifact
identity on cold compilation, persistent loads and process hits. Ineligible source checkouts retain
their blockers and missing build evidence; reporting a fingerprint does not make them complete.
The same owner also retains the already sealed/opened compilation and payload fingerprints.
Store/load/process-hit paths agree; corrupt loads, ineligible builds and failed writes provide no
retained artifact hashes. Successful recompilation/storage after a corrupt load supplies hashes of
the replacement artifact. Ineligible builds expose no persistent artifact identity map. This
records exact artifact identity, not a durable numerical receipt.

`compiled/execution-identity` now joins that complete artifact identity with the bound invocation:
ordered retained program calls (including host-derived scalars and selected graphs), public shape/
storage contracts, roles, donation and schedule. For composition, use the existing
`CertifiedLinkComposition` components and normalized specification, not its nested timing reports.
It requires the original compiler-owned Prepared, independently reverifies the existing lowering
owners, and records source-free plan and default-free boundary contracts. No second build/source
cache or function registry is needed. Component IDs remain significant rather than claiming
equivalence under arbitrary graph renaming. This explicit inspection operation is not run on each
replay and introduces no identity cache.
Generated SSA names are anchored to that exact retained compilation artifact. Independently
recompiled artifacts may conservatively differ even for identical source; future typed-IR
alpha-normalization must be scope/definition/use aware, not a blanket symbol replacement.

Publication must additionally name addressed evidence for the actual final external inputs.
Prepared defaults are not sufficient: invocation can replace them. Constants, weights and external
inputs need verified immutable content evidence; mutable state needs verified parent field/state
evidence. Derive required slots from the final semantic input boundary so connected intermediates,
duplicate immutable shares and removed mutable borrowers do not become spurious dependencies.
Reject missing/extra slots and opaque runtime handles. Reuse content addresses and manifests;
parameter bytes are not folded into the template cache key.
In particular, a caller-supplied content-address map is not an authenticated binding receipt. The
runtime must join an exact node/view, verified source bytes or inherited device lineage, and a
completed upload/D2D operation before claiming those bytes produced the result. Structural
inspection alone cannot claim full producer attestation for externally initialized programs.

The synchronous linked-write boundary must also fail closed after a partially completed transfer:
the destination and all overlapping views require explicit reinitialization. A no-copy self-write
or a device copy reading a tainted resident range cannot repair those bytes. Successful complete
writes clear the repaired node only; disjoint views remain usable. Invocation, profiling and
measurement retire previous output wrappers after complete preflight but before backend writes.
Inputs rooted by object identity in those retired outputs retain private borrowed reads only
inside the existing executable lifetime lock, without extending independent or asynchronous
ownership. Donations commit at that same mutation boundary: backend failure consumes them too,
whereas pure preflight errors preserve handles. This readiness contract is a prerequisite for
future completed-input receipts, not byte attestation or rollback of partially executed kernels.

Synchronous replay, profiling and measurement share one completion boundary. Each warmup, probe
and measured replay counts as a completed replay only after success. An owner-local value epoch
also advances before uploads, device writes, restoration/replay and measurement flush callbacks;
it invalidates continuity, but is neither a portable state identity nor byte evidence. Pure option,
readiness, profiling and lease declines do not change it. A failed replay, profiling operation,
restore callback or flush poisons the executable: the original error is retained, subsequent use
rejects with `:link-execution-poisoned`, and callers must close and reinstantiate. Closing remains
allowed and idempotent. Unlike a repairable failed input transfer, an arbitrary partially executed
program has no proven restoration boundary. This is an intentional failure-contract tightening,
not automatic rollback, cross-owner mutation detection or completed producer attestation.

Offline linked-dispatch tuning also owns an exclusive mutation scope. Selection precedes the
scope; candidate construction, restoration, validation and timing remain under the existing
lifetime lock and cannot bypass output leases. Entry and successful exit invalidate value
continuity and output readiness, even when the tuning cache avoids launches. Candidate kernels
do not count as complete plan replays. Callback failure poisons the executable, and an output
lease cannot escape a callback that temporarily ran the full plan. This conservative contract
does not infer which candidates wrote which bytes or attach content evidence to tuning results.

The bound structure is implemented, not full producer certification. Its result has scope
`:exact-bound-program` and explicitly `:attests-input-bytes? false`; it supplies no numerical-state
`:program-fingerprint`. Affected artifact/cache/composition tests pass 28 tests / 189 assertions:
shape, role, scalar bit and wiring changes distinguish identities; cold/process-hit/persistent-load
paths agree for the same artifact; nested composition works; incomplete builds and modified owners
reject. Different host arrays and contents intentionally do not change this structural identity.
Completed resident-byte inspection is described below; full producer publication and independent
restore matching remain open.

An instantiated `Compiled` retains its original compiler-owned `Prepared` by reference and shares
the existing exact-object seal. `execution-identity` can inspect either original owner; copying or
associating fields, replacing the executable/boundary or substituting another Prepared declines.
An independently validated copied Prepared may still execute, but does not acquire exact retained
artifact provenance through instantiation. Inspection uses retained build/artifact evidence rather
than the current compiler cache or build state. No identity hashing is added to instantiation or
ordinary invocation. This remains structural evidence with `:attests-input-bytes? false`.
A conservative exact execution/build identity must not be presented as target-neutral mathematical
program equivalence.

### Offline completed resident-byte evidence

`compiled/invoke-with-evidence` executes an original sealed `Compiled` under the existing linked
lifetime lock. It hashes actual dense initialization-root bytes after all invocation writes and
declared outputs plus mutated roots after successful synchronous replay. This covers refreshed
host inputs and completed device copies, not mutable Prepared defaults or caller-supplied hashes.
The content reader shares the existing SHA-256 implementation with a fixed 64-KiB staging bound;
ordinary invocation adds no hashing or readback. Inspection is explicitly offline: downloading
large weights remains expensive despite bounded working memory.

The returned exact-owner `CompletedEvidence` pins output values until close. Dereferencing it
returns historical metadata even after close; accessing its device outputs requires the live
lease. Metadata records the retained program fingerprint, bound schedule descriptions, typed
view extents, raw content addresses and optional parent fingerprint. Raw bytes are labeled
`:device-native`, without an inferred byte order or canonical numerical codec. These receipts
are not cross-device restore manifests, compiler-binary attestations or mathematical proofs.

Admission requires an owned session and owned storage, retained initialization evidence, dense
views, no record-time prologue and no registered asynchronous events. A mutable-state receipt
names the previous same-owner receipt only when the owner epoch is uninterrupted and its
actual pre-state equals the previous actual post-state. Stateless receipts have no parent.
Unwitnessed replay, transfer, profiling, measurement or tuning breaks continuity. Pure input
preflight declines preserve the previous head; failures after mutation clear it. A readback
failure produces no receipt and releases the output lease, without pretending the successfully
completed replay never occurred. Only the latest head is retained, not an ancestor object chain.

This establishes a conservative completed-byte boundary inside linked ownership. Durable blob
publication, verified codec matching, scalable selective inspection and tracking mutation through
arbitrary raw-session access remain separate obligations on the original campaign.

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
