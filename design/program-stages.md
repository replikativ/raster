# Semantic stages in resident program descriptors

`raster.compiler.ir.program-stage` defines the narrow boundary for replacing part of an already
compiled resident descriptor when a runtime owns effects inside that part. Paged cache management
is the first consumer, but the compiler contract contains no attention vocabulary.

A stage declaration names:

- a stable semantic identity;
- persistent state written by the stage;
- ordinary values exported by the stage;
- optional internal anchors needed to constrain the interval.

Raster derives ordered reads and writes from each step's checked executable ABI. Every anchor must
have exactly one writer. The unique minimal interval containing those writers is accepted only when
every value live after the interval is classified as state or output and every declared effect is
actually written. Selection therefore cannot silently use a first matching kernel name, output
spelling, or step index.

```clojure
(require '[raster.compiler.ir.program-stage :as stage])

(def routed-boundary
  (stage/select layer-descriptor
                {:id :routed-cache-attention
                 :state #{'key-cache 'value-cache}
                 :outputs #{'attention-output}}))

(def layer-parts (stage/partition layer-descriptor routed-boundary))
;; => {:before descriptor, :selected descriptor, :after descriptor}
```

The projected descriptors retain the compiler's original parameter order and scalar/size closures,
while unused scratch-allocation contracts are removed. Each non-empty projection can therefore be
instantiated as an ordinary `LinkInstance`; no second graph or attention-specific linker exists.

State and outputs may escape the complete descriptor and need not be locally live after the stage.
They must still be unique anchored writes. Conversely, an internal anchor may not be live outside
the selected interval. A runtime replacement consumes `ProgramStage.inputs`, supplies its declared
outputs/state effects, and composes the before/after projections through the same resident views.

This API deliberately does not promise arbitrary binary linking or infer semantic operations from
kernel names. The application/compiler frontend declares the semantic effect boundary, and Raster
proves that the emitted descriptor still realizes it. If fusion later moves or combines kernels,
the contract either resolves to the new unique effect span or fails loudly.
