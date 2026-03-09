# GPU Computing

Raster compiles `deftm` functions to GPU kernels from the same Clojure source
that runs on the CPU. No separate kernel language is needed.

## Parallel Primitives

`raster.par` provides declarative parallel forms:

- `map!` / `map-void!` — parallel element-wise operations
- `reduce!` — parallel reduction
- `scan!` / `scan-exclusive` — parallel prefix sum
- `broadcast!` — array broadcasting
- `stencil` — neighborhood operations

At the REPL these expand to sequential loops. The compiler backend rewrites
them to SIMD, OpenCL, Vulkan, or Level Zero kernels.

## GPU Session API

A unified session manages kernel compilation, buffer allocation, invocation,
and cleanup:

```clojure
(require '[raster.gpu :as gpu])

(gpu/with-gpu-session [sess :ocl:0]   ;; or :ze:0 for Level Zero
  ;; Compile a deftm function to GPU kernels
  (gpu/compile! sess :step #'simulation-step!)

  ;; Allocate device buffers (topology-aware: zero-copy on integrated GPUs)
  (gpu/alloc! sess {:positions [:float n init-data]
                    :velocities [:float n nil]})

  ;; Invoke — buffers looked up by key
  (gpu/invoke! sess :step {"pos" :positions "vel" :velocities}
               [{:type :int :value grid-size}] n)

  ;; Download results
  (gpu/download sess :positions))
```

## Backends

| Backend | Module | Platforms |
|---------|--------|-----------|
| OpenCL ICD | `raster.gpu.ocl-runtime` | NVIDIA, AMD, Intel (portable) |
| Intel Level Zero | `raster.gpu.ze-runtime` | Intel GPUs, unified shared memory |
| Vulkan compute | `raster.compiler.backend.gpu.par-vulkan` | GLSL 450 -> SPIR-V |
| SIMD | `raster.compiler.backend.jvm.par-simd` | JDK Vector API (CPU) |

All backends use Panama FFM (JDK 22+) for native bindings — no JNI or
external build tools.

## SoA Memory Layout

GPU kernels automatically expand struct-of-arrays fields: a single `deftm`
parameter tagged with an SoA type becomes multiple flat arrays in the kernel
signature. Field access compiles to the correct array indexing, giving
GPU-friendly coalesced memory access without manual data layout work.

## SOAC Fusion

The compiler builds a dependency graph of parallel operations and fuses
compatible chains into single kernels:

- **Vertical fusion** — producer-consumer chains (map feeding into reduce)
- **Horizontal fusion** — independent maps over the same input

This minimizes memory traffic and kernel launch overhead.
