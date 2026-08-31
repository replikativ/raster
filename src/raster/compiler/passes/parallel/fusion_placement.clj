(ns raster.compiler.passes.parallel.fusion-placement
  "Target-neutral placement policy for a functional value with more than one consumer.

   Fusion legality belongs to the SOAC pass.  This namespace answers the narrower scheduling
   question that remains once fusion is known to be legal: should a producer be recomputed inside
   its consumers, or should its intermediate stay materialized?  It reads only Raster's Abstract
   Machine and typed scalar expressions; vendor/device identities never cross this boundary."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]))

(def policy-version
  "Stable identifier recorded with every placement witness."
  :roofline-v1)

(defn expression-flops
  "Return the registered flop-equivalent cost of `expressions`.

   Cost facets are attached to semantic operations in op-descriptor.  Unpriced operations add no
   speculative cost: this policy is decline-only, so incomplete cost information must not invent a
   loss and disable an otherwise legal fusion."
  [expressions]
  (let [total (volatile! 0)]
    (letfn [(visit [form]
              (when (seq? form)
                (when-let [operation (descriptor/semantic-op form)]
                  (when-let [cost (descriptor/cost-facet operation)]
                    (vswap! total + (long (:flops cost 0)))))
                (run! visit form)))]
      (run! visit expressions))
    @total))

(defn placement-decision
  "Choose recomputation or materialization for one typed producer value.

   `consumer-count` includes equation consumers and externally observable program outputs.  A sole
   consumer always eliminates the intermediate: its computation moves rather than duplicates.
   For fan-out, recomputation is profitable when the producer's per-element work fits beneath the
   roofline cost of reading one materialized element:

       producer-flops <= element-bytes * ridge(dtype)

   With no Abstract Machine, fan-out remains materialized: the typed pass must not speculate that
   duplicated computation is cheaper. Missing dtype-specific information on an otherwise present
   machine abstains in the established decline-only direction (`:recompute`).
   The returned map is a serializable witness suitable for program facts and tuning records."
  [{:keys [abstract-machine dtype expressions consumer-count]}]
  (let [consumer-count (long consumer-count)
        flops (expression-flops expressions)
        element-bytes (when (dtype/known? dtype) (dtype/bytes-of dtype))
        ridge (get-in abstract-machine [:ridge dtype])
        threshold (when (and element-bytes ridge)
                    (* (long element-bytes) (double ridge)))
        base {:policy policy-version
              :consumer-count consumer-count
              :dtype dtype
              :producer-flops-per-element flops
              :element-bytes element-bytes
              :ridge-flops-per-byte ridge
              :recompute-threshold-flops threshold}]
    (cond
      (<= consumer-count 1)
      (assoc base :decision :eliminate
             :reason :sole-consumer
             :fuse? true)

      (nil? abstract-machine)
      (assoc base :decision :materialize
             :reason :no-abstract-machine
             :fuse? false)

      (nil? element-bytes)
      (assoc base :decision :recompute
             :reason :unknown-element-width
             :fuse? true)

      (nil? ridge)
      (assoc base :decision :recompute
             :reason :unknown-roofline-ridge
             :fuse? true)

      (<= flops threshold)
      (assoc base :decision :recompute
             :reason :recompute-cheaper-than-read
             :fuse? true)

      :else
      (assoc base :decision :materialize
             :reason :recompute-more-expensive-than-read
             :fuse? false))))
