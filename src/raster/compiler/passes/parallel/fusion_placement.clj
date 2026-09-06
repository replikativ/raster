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
  :roofline-v2)

(defn- positive-finite-price?
  [price]
  (and (number? price) (Double/isFinite (double price)) (pos? (double price))))

(defn expression-cost
  "Return registered flop-equivalent cost and its completeness for `expressions`.

   Cost facets are attached to semantic operations in op-descriptor. Array reads and primitive
   casts are transparent here: memory traffic is priced by the placement policy and casts do not
   add floating-point work. Every other unpriced call makes the estimate incomplete; unknown work
   must never look like zero work and enable duplicated computation."
  [expressions]
  (let [total (volatile! 0)
        unknown (volatile! #{})]
    (letfn [(visit [form]
              (when (seq? form)
                (when-let [operation (descriptor/semantic-op form)]
                  (if-let [cost (descriptor/cost-facet operation)]
                    (vswap! total + (long (:flops cost 0)))
                    (when-not (or (descriptor/aget-op? operation)
                                  (descriptor/cast-op? operation))
                      (vswap! unknown conj operation))))
                (run! visit form)))]
      (run! visit expressions))
    {:flops @total
     :complete? (empty? @unknown)
     :unknown-ops (vec (sort-by str @unknown))}))

(defn expression-flops
  "Return the registered portion of the flop-equivalent cost of `expressions`.

   Call `expression-cost` when completeness affects a decision."
  [expressions]
  (:flops (expression-cost expressions)))

(defn placement-decision
  "Choose recomputation or materialization for one typed producer value.

   `consumer-count` includes equation consumers and externally observable program outputs.  A sole
   consumer always eliminates the intermediate: its computation moves rather than duplicates.
   For fan-out, recomputation is profitable when the producer's per-element work fits beneath the
   roofline cost of reading one materialized element:

       producer-flops <= element-bytes * ridge(dtype)

   With incomplete producer cost or no complete Abstract Machine price, fan-out remains
   materialized: the typed pass must not speculate that duplicated computation is cheaper.
   The returned map is a serializable witness suitable for program facts and tuning records."
  [{:keys [abstract-machine dtype expressions consumer-count]}]
  (let [consumer-count (long consumer-count)
        {:keys [flops complete? unknown-ops]} (expression-cost expressions)
        element-bytes (when (dtype/known? dtype) (dtype/bytes-of dtype))
        ridge (get-in abstract-machine [:ridge dtype])
        threshold (when (and element-bytes (positive-finite-price? ridge))
                    (* (long element-bytes) (double ridge)))
        base {:policy policy-version
              :consumer-count consumer-count
              :dtype dtype
              :producer-flops-per-element flops
              :producer-cost-complete? complete?
              :unknown-cost-ops unknown-ops
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

      (not complete?)
      (assoc base :decision :materialize
             :reason :unknown-producer-cost
             :fuse? false)

      (nil? element-bytes)
      (assoc base :decision :materialize
             :reason :unknown-element-width
             :fuse? false)

      (nil? ridge)
      (assoc base :decision :materialize
             :reason :unknown-roofline-ridge
             :fuse? false)

      (not (positive-finite-price? ridge))
      (assoc base :decision :materialize
             :reason :invalid-roofline-ridge
             :fuse? false)

      (not (positive-finite-price? threshold))
      (assoc base :decision :materialize
             :reason :invalid-recompute-threshold
             :fuse? false)

      (<= flops threshold)
      (assoc base :decision :recompute
             :reason :recompute-cheaper-than-read
             :fuse? true)

      :else
      (assoc base :decision :materialize
             :reason :recompute-more-expensive-than-read
             :fuse? false))))
