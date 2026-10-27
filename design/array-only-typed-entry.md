# Primitive-array-only source entries

An original `deftm` declaration now selects the existing typed producer when
either an argument or its return annotation normalizes to a primitive-array
tag in `types/primitive-array-info`. Both `floats` and `(Array float)` use the
existing `annotation->tag` contract; `Param`/`Frozen` wrappers remain transparent.
Object arrays, nominal arrays and unresolved element types acquire no new typed
entry permission.

This is a producer migration, not an optimization annotation. Newly declared
array-only functions receive the same deferred source preparation, typed IFn
interface and physical entry/return adaptation as other typed functions.
Selected entry arguments are checked even when unused; null remains a valid
reference until the original body dereferences it. Return checks remain after
body evaluation. Explicit checked source casts remain checked. Arithmetic now
uses the existing typed producer's element dtype, just as declarations already
selected by a scalar annotation do. This is an observable surface change from
the former plain array-only producer: for example, Float multiplication overflow
produces infinity rather than a checked Double-to-Float conversion failure. Bare
floating literals can also select Float rather than Double arithmetic in a Float
body. This migration does not grant reassociation or enable fast-math. JVM AOT
must agree with this new original producer, not with the superseded plain path.

Existing plain frontends and emitted artifacts are not upgraded or restamped.
Their original source/producer identity guards remain authoritative. Driver-free
OpenCL/CUDA/HIP source qualification does not establish device execution; the
global routing change additionally requires fresh-process compiler and CI gates.

Ordinary and parametric dispatch now reuse the published source delegate rather
than a second equivalent wrapper. Return analyses must authenticate this routing
and the original semantic frontend; they cannot infer an arbitrary wrapper's
result from the separately registered typed implementation. Semantic-only facts
survive the same source's first physical JIT promotion, while substituted source
or receiver implementations remain guarded. This does not make nullable array
arguments sufficient to select a generic overload.
