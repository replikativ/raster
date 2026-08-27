(ns raster.compiler.ir.segmented-weighted-reduction-schedule
  "Target-neutral schedules for segmented normalized weighted reductions.

   The semantic operation remains SegmentedWeightedReductionPlan.  This value records how one
   legal implementation maps segments, score components, visible members and value components to
   cooperative hardware execution; it contains no attention buffer identities or target syntax.")

(defrecord SegmentedWeightedReductionSchedule
           [strategy workgroup-size segment-mapping membership-traversal score-reduction
            value-mapping state numerical-mode attributes])

(defn schedule?
  [value]
  (and value
       (= "raster.compiler.ir.segmented_weighted_reduction_schedule.SegmentedWeightedReductionSchedule"
          (.getName (class value)))))

(defn validate!
  [schedule]
  (when-not (schedule? schedule)
    (throw (ex-info "expected a SegmentedWeightedReductionSchedule"
                    {:reason :segmented-weighted-reduction-schedule-type
                     :schedule schedule})))
  (let [{:keys [strategy workgroup-size segment-mapping membership-traversal score-reduction
                value-mapping state numerical-mode attributes]} schedule]
    (when-not (= :subgroup-online-score-reuse strategy)
      (throw (ex-info "segmented weighted-reduction schedule has an unsupported strategy"
                      {:reason :segmented-weighted-reduction-schedule-strategy
                       :strategy strategy})))
    (when-not (and (integer? workgroup-size) (pos? workgroup-size))
      (throw (ex-info "segmented weighted-reduction workgroup size must be positive"
                      {:reason :segmented-weighted-reduction-schedule-workgroup
                       :workgroup-size workgroup-size})))
    (when-not (= :one-workgroup-per-segment segment-mapping)
      (throw (ex-info "cooperative weighted reduction requires one workgroup per segment"
                      {:reason :segmented-weighted-reduction-segment-mapping
                       :segment-mapping segment-mapping})))
    (when-not (= :sequential membership-traversal)
      (throw (ex-info "initial cooperative weighted reduction requires sequential membership traversal"
                      {:reason :segmented-weighted-reduction-membership-traversal
                       :membership-traversal membership-traversal})))
    (when-not (and (= :subgroup (:kind score-reduction))
                   (= workgroup-size (:width score-reduction))
                   (some? (:axis score-reduction)))
      (throw (ex-info "score reduction must occupy the schedule's complete hardware subgroup"
                      {:reason :segmented-weighted-reduction-score-reduction
                       :score-reduction score-reduction
                       :workgroup-size workgroup-size})))
    (when-not (and (= :lane-strided (:kind value-mapping))
                   (integer? (:components value-mapping))
                   (pos? (:components value-mapping))
                   (integer? (:components-per-lane value-mapping))
                   (= (:components-per-lane value-mapping)
                      (quot (+ (:components value-mapping) (dec workgroup-size))
                            workgroup-size)))
      (throw (ex-info "value mapping must cover every component by lane-strided register state"
                      {:reason :segmented-weighted-reduction-value-mapping
                       :value-mapping value-mapping
                       :workgroup-size workgroup-size})))
    (when-not (= {:kind :online-normalized-weighted-sum
                  :components [:maximum :denominator :weighted-values]}
                 state)
      (throw (ex-info "cooperative weighted reduction requires explicit online softmax state"
                      {:reason :segmented-weighted-reduction-online-state :state state})))
    (when-not (and (map? numerical-mode)
                   (keyword? (:score-accumulate numerical-mode))
                   (keyword? (:state-accumulate numerical-mode))
                   (= :subgroup-tree (:dot-order numerical-mode))
                   (true? (:online-rescale? numerical-mode))
                   (map? attributes))
      (throw (ex-info "segmented weighted-reduction numerical and attribute facets are incomplete"
                      {:reason :segmented-weighted-reduction-schedule-facets
                       :numerical-mode numerical-mode :attributes attributes}))))
  schedule)

(defn make
  [{:keys [strategy workgroup-size segment-mapping membership-traversal score-reduction
           value-mapping state numerical-mode attributes]
    :or {attributes {}}}]
  (validate!
   (->SegmentedWeightedReductionSchedule
    strategy workgroup-size segment-mapping membership-traversal score-reduction value-mapping
    state numerical-mode attributes)))
