(ns raster.compiler.passes.parallel.segmented-weighted-reduction-route
  "Backend routing for schedule-neutral segmented weighted reductions.

   Each leaf proves its own representability from plan descriptors. The router composes decline
   trails and never infers a model-level semantic operation from the source spelling."
  (:require [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.passes.parallel.indexed-attention-route :as indexed-leaf]))

(def candidate-leaves
  [{:id :indexed-edge-list-subgroup-score-reuse :route indexed-leaf/route-dynamic-score-reuse}
   {:id :indexed-edge-list-reference :route indexed-leaf/route-dynamic}])

(defn- tuning-contract
  [plan]
  (let [{:keys [operands output accumulator-dtype] :as plan} (swr/validate! plan)
        schedule-key (swr/schedule-key plan)]
    {:schedule-path [:segmented-weighted-reduction :measured-selector]
     :numerical-mode {:operands (mapv :dtype operands)
                      :accumulate accumulator-dtype
                      :output (:dtype output)}
     ;; This is deliberately the compiler's physical schedule identity, not an attention label.
     ;; A different storage/membership layout must miss the tuning cache even when its algebra is
     ;; identical.
     :layout (assoc (dissoc schedule-key :algebra)
                    :membership (dissoc (:membership plan) :buffers))
     ;; Reference interpretation is compiler metadata. The generic GPU benchmark merely exposes
     ;; it to a caller-supplied oracle and never branches on this kind.
     :reference {:kind :segmented-weighted-reduction :plan plan}}))

(defn- selected-leaves
  [desc]
  (if (= :subgroup-score-reuse (:segmented-weighted-reduction-schedule desc))
    candidate-leaves
    [(peek candidate-leaves)]))

(defn route-dynamic-candidates
  "Return every representable dynamic leaf plus the complete decline trail.

   This is the emission seam for runtime dispatch/autotuning. It enumerates schedules but does not
   pick one; `route-dynamic` remains the single-artifact policy API."
  ([plan] (route-dynamic-candidates plan nil))
  ([plan desc]
   (let [plan (swr/validate! plan)
         results (mapv (fn [leaf]
                         (assoc ((:route leaf) plan desc) :leaf (:id leaf)))
                       candidate-leaves)]
     {:operation plan
      :candidates (filterv :strategy results)
      :declines (into [] (mapcat :declines) results)})))

(defn route-dynamic-candidates!
  ([plan] (route-dynamic-candidates! plan nil))
  ([plan desc]
   (let [result (route-dynamic-candidates plan desc)]
     (if (seq (:candidates result))
       result
       (throw (ex-info "no executable segmented weighted-reduction kernel candidates"
                       {:reason :segmented-weighted-reduction-no-kernel-candidates
                        :route result}))))))

(defn dynamic-dispatch
  "Package compatible dynamic schedules as runtime-selection IR.

   Returns nil when the target has only one legal leaf; callers then retain that ordinary artifact
   instead of manufacturing a degenerate dispatch."
  [plan desc schedule]
  (let [{:keys [candidates]} (route-dynamic-candidates! plan desc)
        artifacts (mapv :artifact candidates)
        by-strategy (into {} (map (juxt :strategy identity)) candidates)
        reference :indexed-segmented-reduction-reference
        score-reuse :indexed-segmented-reduction-subgroup-score-reuse
        subgroup-size (long (or (:subgroup-size desc) 16))
        multiple (long (get-in schedule
                               [:segmented-weighted-reduction
                                :score-reuse-subgroup-multiple]
                               16))
        measured-selector (get-in schedule
                                  [:segmented-weighted-reduction :measured-selector])
        components (get-in plan [:value :components])
        dispatch-key [(swr/algebra-key plan) subgroup-size
                      (or measured-selector multiple)]
        id (format "raster_segmented_weighted_reduction_dispatch_%08x"
                   (bit-and 0xffffffff (long (hash dispatch-key))))]
    (when (and (contains? by-strategy reference)
               (contains? by-strategy score-reuse))
      (let [dispatch
            (kdispatch/make
             {:id id
              :alternatives artifacts
              :default-strategy reference
              :selector {:kind :runtime-scalar-threshold
                         :argument components
                         :threshold (* subgroup-size multiple)
                         :at-least score-reuse
                         :otherwise reference}
              :provenance {:operation-id (get-in plan [:provenance :operation-id])
                           :semantic-op (get-in plan [:provenance :semantic-op])
                           :algebra-plan-id (:id plan)}
              :attributes {:algebra :segmented-weighted-reduction
                           :algebra-key (swr/algebra-key plan)
                           :tuning (tuning-contract plan)
                           :selection (if measured-selector
                                        :measured-runtime-shape
                                        :analytic-runtime-shape)}})]
        (if measured-selector
          (kdispatch/with-selector dispatch measured-selector)
          dispatch)))))

(defn route-dynamic
  ([plan] (route-dynamic plan nil))
  ([plan desc]
   (let [plan (swr/validate! plan)]
     (loop [[leaf & more] (selected-leaves desc)
            declines []]
       (if-not leaf
         {:operation plan :strategy nil :reference? false :declines declines}
         (let [result ((:route leaf) plan desc)]
           (if (:strategy result)
             (assoc result :leaf (:id leaf))
             (recur more (into declines (:declines result))))))))))

(defn route-dynamic!
  ([plan] (route-dynamic! plan nil))
  ([plan desc]
   (let [result (route-dynamic plan desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable segmented weighted-reduction kernel route"
                       {:reason :segmented-weighted-reduction-no-kernel-route
                        :route result}))))))
