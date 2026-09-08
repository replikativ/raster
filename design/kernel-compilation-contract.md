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
