(ns raster.compiler.passes.parallel.product-consumer-schedule
  "Target-aware physical schedule selection for a certified product/consumer region.

   The semantic region proof is target-neutral. This selector may choose a subgroup spelling only
   from stated device capabilities; otherwise the existing workgroup-memory tree remains the
   executable fallback."
  (:require [raster.compiler.core.hardware :as hardware]))

(def ^:private maximum-unrolled-local-volume 32)

(defn- power-of-two? [value]
  (and (integer? value) (pos? value) (zero? (bit-and value (dec value)))))

(defn- subgroup-width
  [desc reduced-width]
  (let [preferred (hardware/preferred-subgroup-size desc)
        supported (sort (hardware/supported-subgroup-sizes desc))
        candidates (filter #(and (power-of-two? %) (<= reduced-width %)) supported)]
    (cond
      (and (some #{preferred} candidates)) preferred
      (seq candidates) (first candidates)
      :else nil)))

(defn select
  "Attach one executable physical schedule to `plan`.

   The subgroup row uses exactly one subgroup per workgroup and pads inactive lanes with the
   certified identity. Static local suffixes are unrolled into SSA values. Missing or insufficient
   hardware facts deliberately retain the portable workgroup tree."
  [plan desc]
  (let [local-bounds (mapv :bound (get-in plan [:axes :local]))
        reduced-width (get-in plan [:axes :reduced :bound])
        static-local? (every? #(and (integer? %) (pos? %)) local-bounds)
        local-volume (when static-local? (reduce * 1 local-bounds))
        max-workgroup (hardware/maximum-workgroup-size desc)
        width (when (and desc (= :gpu (:device-type desc))
                         (integer? reduced-width) (pos? reduced-width))
                (subgroup-width desc reduced-width))
        subgroup? (and (:subgroup-collective plan)
                       width
                       (integer? max-workgroup)
                       (<= width max-workgroup)
                       static-local?
                       (<= local-volume maximum-unrolled-local-volume))]
    (assoc plan :physical-schedule
           (if subgroup?
             {:strategy :subgroup-product-ordered-consumer
              :workgroup-size width
              :subgroup-size width
              :reduced-width reduced-width
              :active-lanes reduced-width
              :neutral-padding (- width reduced-width)
              :local-volume local-volume
              :local-unrolling :static
              :intermediate-substitution :proved-local-offset}
             {:strategy :product-tree-ordered-consumer
              :workgroup-size (:workgroup-size plan)
              :fallback-reason
              (cond
                (nil? (:subgroup-collective plan)) :collective-contract-unavailable
                (nil? desc) :hardware-descriptor-unavailable
                (not= :gpu (:device-type desc)) :gpu-required
                (nil? width) :subgroup-width-unavailable
                (not (integer? max-workgroup)) :workgroup-limit-unavailable
                (not static-local?) :dynamic-local-volume
                (> local-volume maximum-unrolled-local-volume) :local-unroll-limit
                :else :subgroup-infeasible)}))))
