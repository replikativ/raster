(ns raster.compiler.ir.segmented-weighted-reduction-schedule
  "Target-neutral schedules for segmented normalized weighted reductions.

   The semantic operation remains SegmentedWeightedReductionPlan.  This value records how one
   legal implementation maps segments, score components, visible members and value components to
   cooperative hardware execution; it contains no attention buffer identities or target syntax.")

(defrecord SegmentedWeightedReductionSchedule
           [strategy workgroup-size segment-mapping membership-traversal score-reduction
            membership-tiling value-mapping state staging numerical-mode attributes])

(def ^:private online-state-components
  [:maximum :denominator :weighted-values])

(def ^:private online-state-merge
  {:kind :maximum-rescale-sum
   :order :increasing-membership-tile
   :nan-policy :propagate
   :empty-policy :identity})

(defn online-state
  "The explicit mergeable state carried by sequential and tiled online schedules.

  This is scheduled reduction state, not an attention semantic.  Its merge law lets a later
  lowering partition any normalized exponential weighted reduction without exposing partial
  buffers through the source operation or public ABI."
  []
  {:kind :online-normalized-weighted-sum
   :components online-state-components
   :merge online-state-merge})

(declare validate!)

(defn tiled?
  [schedule]
  (= :static-contiguous-tiles (get-in (validate! schedule) [:membership-tiling :kind])))

(defn- valid-membership-tiling?
  [membership-tiling]
  (case (:kind membership-tiling)
    :sequential
    (= {:kind :sequential} membership-tiling)

    :static-contiguous-tiles
    (let [{:keys [tile-size tile-count membership-capacity merge-order]} membership-tiling]
      (and (pos-int? tile-size)
           (<= tile-size Integer/MAX_VALUE)
           (pos-int? tile-count)
           (pos-int? membership-capacity)
           (= tile-count (quot (+ membership-capacity (dec tile-size)) tile-size))
           (= :increasing-membership-tile merge-order)))

    false))

(defn- valid-staging?
  [strategy staging]
  (if (= :subgroup-online-pipelined-history strategy)
    (and (= :double-buffered-membership-rows (:kind staging))
         (= 2 (:stages staging))
         (= 2 (:members-per-iteration staging))
         (= :half (:element-dtype staging))
         (pos-int? (:key-elements staging))
         (pos-int? (:value-elements staging))
         (contains? #{4 8 16} (:transfer-bytes staging))
         (contains? #{:preferred :required} (:overlap staging))
         (= :separate-epilogue (:tail-policy staging))
         (pos-int? (:shared-memory-bytes staging))
         (= (:shared-memory-bytes staging)
            (* 2 (+ (:key-elements staging) (:value-elements staging)) 2)))
    (= {:kind :none} staging)))

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
                membership-tiling value-mapping state staging numerical-mode attributes]} schedule]
    (when-not (contains? #{:subgroup-online-score-reuse
                           :subgroup-online-pipelined-history
                           :subgroup-online-tiled-history}
                         strategy)
      (throw (ex-info "segmented weighted-reduction schedule has an unsupported strategy"
                      {:reason :segmented-weighted-reduction-schedule-strategy
                       :strategy strategy})))
    (when-not (and (integer? workgroup-size) (pos? workgroup-size))
      (throw (ex-info "segmented weighted-reduction workgroup size must be positive"
                      {:reason :segmented-weighted-reduction-schedule-workgroup
                       :workgroup-size workgroup-size})))
    (let [expected-mapping (if (= :subgroup-online-tiled-history strategy)
                             {:partial :one-workgroup-per-segment-tile
                              :merge :one-workgroup-per-segment}
                             :one-workgroup-per-segment)]
      (when-not (= expected-mapping segment-mapping)
        (throw (ex-info "cooperative weighted reduction has an invalid segment mapping"
                        {:reason :segmented-weighted-reduction-segment-mapping
                         :strategy strategy :segment-mapping segment-mapping
                         :expected expected-mapping}))))
    (when-not (contains? #{:contiguous-interval :csr-row} membership-traversal)
      (throw (ex-info "cooperative weighted reduction requires a bounded membership traversal"
                      {:reason :segmented-weighted-reduction-membership-traversal
                       :membership-traversal membership-traversal
                       :supported #{:contiguous-interval :csr-row}})))
    (when-not (and (= :subgroup (:kind score-reduction))
                   (= workgroup-size (:width score-reduction))
                   (some? (:axis score-reduction)))
      (throw (ex-info "score reduction must occupy the schedule's complete hardware subgroup"
                      {:reason :segmented-weighted-reduction-score-reduction
                       :score-reduction score-reduction
                       :workgroup-size workgroup-size})))
    (when-not (and (valid-membership-tiling? membership-tiling)
                   (= (= :subgroup-online-tiled-history strategy)
                      (= :static-contiguous-tiles (:kind membership-tiling))))
      (throw (ex-info "weighted-reduction membership tiling is incomplete or inconsistent"
                      {:reason :segmented-weighted-reduction-membership-tiling
                       :strategy strategy :membership-tiling membership-tiling})))
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
    (when-not (= (online-state) state)
      (throw (ex-info "cooperative weighted reduction requires explicit online softmax state"
                      {:reason :segmented-weighted-reduction-online-state :state state})))
    (when-not (valid-staging? strategy staging)
      (throw (ex-info "cooperative weighted reduction has an invalid workgroup staging contract"
                      {:reason :segmented-weighted-reduction-staging
                       :strategy strategy :staging staging})))
    (when-not (and (map? numerical-mode)
                   (keyword? (:score-accumulate numerical-mode))
                   (keyword? (:state-accumulate numerical-mode))
                   (= :implementation-defined (:dot-order numerical-mode))
                   (true? (:online-rescale? numerical-mode))
                   (= (if (= :subgroup-online-tiled-history strategy)
                        :increasing-members-within-tile-then-increasing-tile-left-fold
                        :increasing-membership)
                      (:online-state-order numerical-mode))
                   (map? attributes))
      (throw (ex-info "segmented weighted-reduction numerical and attribute facets are incomplete"
                      {:reason :segmented-weighted-reduction-schedule-facets
                       :numerical-mode numerical-mode :attributes attributes}))))
  schedule)

(defn make
  [{:keys [strategy workgroup-size segment-mapping membership-traversal score-reduction
           membership-tiling value-mapping state staging numerical-mode attributes]
    :or {staging {:kind :none} attributes {}}}]
  (validate!
   (->SegmentedWeightedReductionSchedule
    strategy workgroup-size segment-mapping membership-traversal score-reduction
    membership-tiling value-mapping state staging numerical-mode attributes)))
