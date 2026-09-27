# CUDA resident bring-up

Raster already emits generated CUDA C++ from verified KernelBody and checks the resulting
sm_80 PTX with `nvcc` in hardware-free CI. This is **source legality**, not CUDA execution:
`raster.gpu.core/make-session` currently supports only Level Zero and OpenCL, and CUDA
hardware detection is a stub. Keep compile-only target descriptors separate from resident
runtime registration until the following acceptance gates pass.

## First device gate

Use one public equation-first elementwise map from the existing compile-fixture corpus.
The path must be `deftm → typed SOAC → schedule → KernelBody → CUDA source → module`, then:

1. Probe the installed CUDA driver and enumerate a real device, recording device name,
   compute capability, driver version, toolkit/compiler version, and available memory.
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
event operations. NVRTC can compile generated CUDA source in-process at JIT time; loading
PTX or cubin through the driver API avoids an `nvcc` subprocess on every compilation. Keep
the existing `nvcc` compile gate. Record compilation identity and driver/toolkit compatibility
in the artifact cache key. A precompiled PTX fixture is acceptable for the *first* transport
smoke, but the gate above is not complete until Raster compiles its generated source itself.

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
