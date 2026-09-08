# Target compilation requirements

A KernelArtifact may carry checked compiler requirements in its existing
`:attributes :compilation` map:

```clojure
{:language-standard "CL3.0"
 :extensions #{"cl_khr_integer_dot_product"}}
```

Absent or empty requirements retain backend defaults. Explicit requirements currently
support OpenCL C only, with CL1.2, CL2.0, or CL3.0 and optional extension-name sets.
They do not permit arbitrary command-line flags or changes to numerical policy.

The OpenCL runtime checks exact device extension tokens before creating a program,
then passes the requested language standard to the compiler. Extension presence
alone does not prove that an overload is supported or hardware accelerated; the
emitter must retain feature guards and schedule selection still needs evidence.

Requirements participate in leaf and graph tuning fingerprints and dispatch
entry-point collision checks. The public tensor ABI remains unchanged. Level Zero
currently rejects explicit requirements because its offline compiler and SPIR-V
cache do not yet consume this contract; silently ignoring it would be incorrect.

This is a prerequisite for target-selected intrinsic lowering, not a new semantic
IR, an optimized kernel promotion, or a change to the default OpenCL dialect.

## Explicit intrinsic candidates

The shared typed emitter accepts an opt-in physical implementation selection:

```clojure
{:target-features {:intrinsic-implementations {:dp4a :opencl-packed-dot}}}
```

The canonical intrinsic registry owns the helper and its compiler requirements.
Only consumed helpers contribute requirements. `emit-scalar-module` returns source
and requirements together; `emit-artifact` transports both automatically. The
source-only `emit-scalar-kernel` API rejects selections with nonempty requirements.
The native OpenCL helper uses signed packed dot followed by unsigned accumulation
and bit reinterpretation, preserving Int32 wrap rather than saturating the result.

This explicit candidate does not infer device acceleration or change automatic
selection. Its source feature guard additionally requires packed-dot frontend
support. `native-dot-validation/run!` (under the bench alias) explicitly validates
150 mixed-sign and accumulator-limit cases on a capable device, poisoning output
before each execution; unsupported devices fail rather than silently skip this
opt-in experiment. Generic CI remains independent of this hardware capability.

The portable helper also performs unsigned wrapping accumulation, then reconstructs
the signed result using only representable casts. The signed four-byte dot itself
fits Int32; adding the accumulator directly in signed C does not. The validation
runner accepts `:portable` for device execution, and `run-c!` runs the same 150
cases under an optimizing C compiler with undefined-behavior sanitization, without
requiring a GPU. This sanitizer check reproduces signed overflow with the former
helper and passes with the corrected helper. Prior benchmark files identify the
older source revision and must not be treated as measurements of this correction.
