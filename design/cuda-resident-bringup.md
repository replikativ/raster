# CUDA resident bring-up

Raster already emits generated CUDA C++ from verified KernelBody and checks the resulting
sm_80 PTX with `nvcc` in hardware-free CI. This is **source legality**, not CUDA execution:
`raster.gpu.core/make-session` currently supports only Level Zero and OpenCL, and CUDA
hardware detection is a stub. Keep compile-only target descriptors separate from resident
runtime registration until the following acceptance gates pass.

## First device gate

### Async completion integration checkpoint

CUDA async range batches reuse the synchronous byte-plan validator and the common
Cleanup lifetime leases. The backend-local buffer borrow table is retired. Reservations
precede root/queue locking; one cleanup plan retains buffer uses, a root lease, staging,
events, completion and download publication. Query and await themselves reserve token
lifetime against concurrent retirement. Await/publication serialize; unknown completion
cannot be retried through a second public adapter.

An attempted free during active use now refuses before owner phase changes. It does not
poison a still-live buffer: independently submitted transfers may drain, followed by a
successful free. This is the common lifetime-admission contract, not a numerical change.

Capabilities use the existing validated transfer schema. CUDA currently shares an in-order
physical stream with compute, advertises no independent transfer queue and no peer transport.
Async host return does not certify physical overlap. Event timing excludes host allocation,
staging and publication; no CUDA runtime performance claim follows from mocked controls.

The contributor's backend-level transfer tests are integrated with additional lifetime and
polling-retirement controls. These four session-level tests remain deferred until public
CUDA registration and session integration; they are not counted as passed or silently skipped:

- `public-session-close-awaits-download-and-frees-transfer-staging`
- `host-publication-error-preserves-operation-failure-but-retires-native-resources`
- `public-await-retains-unknown-completion-with-one-native-attempt`
- `public-await-explicitly-retry-safe-completion-can-retire-after-retry`

Preserve their host-publication, close and unknown/retry-safe completion obligations in the
later session slice. Genuine CUDA/device/sanitizer qualification remains an exact-commit
handoff to Sam after the prepared execution and shared compiler fixes are reconciled.

The memory-foundation integration keeps CUDA session registration disabled. It adds
ordinal-zero allocation/views and checked synchronous ranges under the common root and
cleanup authorities, not graph binding or public execution. `retain-use!` supplies an
exact owner's lifetime lease; `using!` scopes one synchronously. Teardown refuses active
leases without changing phase. Reservation precedes the root/queue monitor, avoiding
owner/root lock inversion; no owner monitor is held through native work.

Manufactured byte plans are normalized through the exact typed slice/host range contract
before native contact. Copies use private pinned staging: a pageable “synchronous” upload
can return before DMA completes ([CUDA synchronization rules](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-driver-api/api-sync-behavior.html)).
Completion, staging retirement and buffer-use retirement form one cleanup dependency plan.
An uncertain copy outcome drains its actual default stream before cleanup, not the separate
nonblocking execution stream. Failed completion retains the operation owner, pinned memory,
buffer-use lease and supporting context without native retry. These are hardware-free
contract/fault controls; the integrated CUDA host must independently check real execution.

The independently integrated `raster.gpu.cuda-driver` leaf marshals native operations
and brackets each calling thread's exact context. It does not register a resident backend.
Native acquisition publication includes context restoration: allocations, modules, events
and pinned-host allocations that cannot be published are retired through the existing
common cleanup contract. Failed retirement retains an unresolved owner and its original
cause; indeterminate destruction is not retried. Initialization reserves a stream-before-
context cleanup dependency so a failed stream retirement cannot release its parent context.
Hardware-free fault tests qualify these contracts, not actual driver execution. The future
runtime adapter must adopt unresolved ownership and retain resources across submission;
the contributor's synchronous bounds/admission defects remain separate integration gates.

Use one public equation-first elementwise map from the existing compile-fixture corpus.
The path must be `deftm → typed SOAC → schedule → KernelBody → CUDA source → module`, then:

1. Probe the installed CUDA driver and enumerate a real device, recording device name,
   compute capability, driver version, toolkit/compiler version, and available memory.
   `raster.gpu.cuda-driver-probe/probe` now performs the read-only driver/name/capability
   part and reports `:no-device` on this laptop; it does not create a session or promise
   executable CUDA support. It now also queries total memory, block dimensions, warp width,
   SM count and distinct per-block/per-SM thread, register and shared-memory limits.
   An unavailable attribute is absent and diagnosed; unexpected driver failures remain errors.
   The hardware registry admits only ordinal zero at this stage. Observations override
   catalogue entries with per-field provenance; the thread-derived maximum warp count is
   a residency upper bound, not measured occupancy. Detection does not enable CUDA sessions.
   On a CUDA host, run its test with `RASTER_REQUIRE_CUDA_DEVICE=1` so a missing device fails.
2. Create a resident session; allocate input/output buffers; upload the input through the
   existing typed buffer contract. Compile the generated source, load its module, bind its
   **ordered KernelABI**, and launch its **KernelLaunch** 1–3D geometry. No fixture-specific
   binder or hand-written kernel may be substituted.
3. Complete the launch through the backend-neutral event contract; download output and
   compare against the uncompiled JVM result. Exercise an uneven extent and a nonzero
   BufferView offset so a correct-looking zero-origin launch is not enough.
4. Replay the bound call with changed resident input, then close the session and verify
   resource cleanup. A device event supplies the elapsed-kernel measurement; host timing
   is recorded separately and is not used to assert a speedup.

The initial adapter should use the CUDA Driver API for context, memory, module, launch and
event operations. The hardware-free `raster.gpu.cuda-nvrtc` transport now compiles generated
CUDA source to PTX in-process; the Driver API load/launch part is still missing. Loading
PTX or cubin through the driver API will avoid an `nvcc` subprocess on every compilation. Keep
the existing `nvcc` compile gate. Record compilation identity and driver/toolkit compatibility
in the artifact cache key. A precompiled PTX fixture is acceptable for the *first* transport
smoke, but the gate above is not complete until Raster compiles its generated source itself.
NVRTC requires explicit toolkit include paths and uses small virtual C headers for the
`stdint.h`/`math.h` names in the current CUDA preamble; host glibc headers are not a valid
substitute. The public equation-first map fixture has compiled to PTX with NVRTC 12.4 on
this machine, without an NVIDIA device. This is transport evidence only.
An NVRTC-equipped host should run the optional transport test with
`RASTER_REQUIRE_NVRTC=1` so missing native coverage is a failure, not a skip.

NVRTC now disables implicit multiply/add contraction with `--fmad=false` and returns
the complete ordered `:compiler-options` with its existing compilation identity.
Explicit FMA remains fused. A reassociated reduction or approximate matrix core does
not by itself grant contraction of its surrounding typed scalar expressions. Native
transport tests check exact/reassociated/bounded-error artifacts, include-path identity,
and emitted PTX containing separate ordinary multiply/add plus explicit FMA. These are
compiler-output checks, not CUDA device numerical or performance evidence.

## Integrating the contributed CUDA runtime

Sam (`nanparth`) contributed draft PR #1338. Integrate it as independently reviewed
increments against current main, retaining original authorship for unchanged commits
and explicit co-authorship/source-commit credit for extracted or reconciled changes.
The exact reviewed source is `063e7fc0bd8e4e22ad4e743c39a2cb1cdaeb458c`;
its hardware receipts do not automatically qualify a different integrated tree.

1. Numerical compilation policy: NVRTC options and independently checkable source
   rounding controls; no runtime registration. This is the first bounded increment.
2. Hardware discovery: observed driver/device limits and provenance, with mocked
   unavailable/partial probes; no catalogue estimates presented as measurements.
3. Driver transport and runtime ownership: context restoration, allocation, views,
   registration and destruction through the existing common ownership contracts.
4. Transfers and asynchronous execution: exact slice bounds, atomic admission versus
   retirement, host leases, completion, cancellation and rollback. The reviewed draft's
   synchronous forged-range and copy-after-free counterexamples must be regressions
   before affected runtime code is admitted.
5. Shared compiler proof changes: separate retained scalar, guarded index/range and
   scheduling increments. Reconcile newer upstream fixes rather than restoring the
   contributed tree wholesale; retain OpenCL/Level Zero and independent refusal controls.
6. Workload qualification: generated matrix/reduction/quantized/training replay on the
   final integrated tree, followed by fresh CUDA device and sanitizer receipts from
   the contributor's host. Compiler-only CI is not runtime or throughput acceptance.

The original draft remains the contribution/evidence umbrella until all slices are
accounted for. Do not merge it wholesale or enable a partially qualified CUDA backend.
The sequence may be refined by dependencies found during extraction; this does not
replace the general compiler-unification campaign or its unchanged numerical gates.

Do not mark `:cuda` a resident backend merely because a library or `nvcc` is present.
The backend entry must declare its memory space, coherence, slice ownership, transfer
capabilities, event behavior and physical-queue facts. Unsupported graph or async operations
should fail explicitly; they must not silently become synchronous or fall back to OpenCL.

`raster.gpu.core` currently resolves many historical runtime functions, including
`invoke-registered-map-void-kernel` and `bind-registered-map-void-kernel`. These are **not**
requirements for a new CUDA backend. The first gate uses the modern `KernelArtifact`,
`KernelCall` and ordered ABI path: arena creation/close, resident buffer allocation/free,
transfer, artifact registration, call binding, launch/completion and download. Add graph
record/replay and asynchronous range batches at the subsequent gate. Where the public
session layer still calls a legacy operation, report an unsupported-route diagnostic
with its operation and backend; do not implement a second CUDA source compiler to satisfy it.

## Subsequent gates

1. Ordered multi-kernel `KernelGraph` recording/replay with stable resident bindings,
   ranged views, explicit completion and cancellation-safe resource retention.
2. A generated reduction (RMSNorm) and generated Q4_K projection, each checked against
   its independent numerical oracle. The Q4_K integer component requires bit identity;
   floating reduction association is reported rather than assumed.
3. Generated GEMM and routed attention, with matching layout/precision/shape against
   NVIDIA library and framework baselines. Device-event timings, compile time, memory
   traffic and numerical error are reported separately. Tune only after correctness.
4. Expand device capabilities and selected schedule families from observed hardware;
   direct PTX remains a later target dialect if CUDA C++ plus vendor compilation is shown
   to leave important hardware behavior inaccessible.

The CUDA host need not be part of normal CI. CI continues to compile generated source
without a GPU; a short opt-in device suite runs on the CUDA host. Avoid renting one until
the first-gate smoke command and driver adapter are ready. At that point, access to a Linux
host with an NVIDIA GPU, working driver, CUDA toolkit/NVRTC and remote shell is enough;
record the exact versions rather than assuming an architecture in the compiler.
