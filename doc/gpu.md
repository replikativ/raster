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

## Read-only aggregate input intent

The explicit `:compiler :equation-first` route in `raster.gpu.compiled/lower`
accepts complete public buffer-field paths in `:constants` and `:roles`:

```clojure
{:compiler :equation-first
 :constants [[:state :labels]]
 :roles {[:state :positions] :input}}
```

Paths follow the retained parameter projection, including nested fields and tree
indices; they do not name generated kernel parameters. Whole-root symbol selectors
still expand to retained buffer leaves. Scalar, unknown or ambiguous paths refuse,
as do overlapping selectors requesting conflicting roles.

A constant leaf is captured when the prepared program is instantiated; changing
the original host array does not silently refresh it on replay. Rebind an input
leaf through its full path, not a whole aggregate containing captured constants.
This is read-only intent, not mutable aggregate ownership: field donation, outputs,
taps and mutable aggregate writes remain unsupported. Field selectors on the
legacy resident-descriptor compiler fail explicitly; select equation-first.

## SOAC Fusion

The compiler builds a dependency graph of parallel operations and fuses
compatible chains into single kernels:

- **Vertical fusion** — producer-consumer chains (map feeding into reduce)
- **Horizontal fusion** — independent maps over the same input

This minimizes memory traffic and kernel launch overhead.
