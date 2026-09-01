(ns raster.compiler.ir.amr-plan
  "Certified block-structured adaptive-mesh workload plans.

   AMRPlan is an outer semantic IR.  It retains the refinement hierarchy and coarse/fine
   operators while delegating durable bytes to NumericalStateManifest and placement, routes,
   dependencies, and costs to DistributedPlan.  Version 1 is intentionally small: cell-centred
   rectangular patches, adjacent levels, aligned refinement, explicit prolongation/restriction,
   and one owned distributed value per patch. Operator requirements are declarations for later
   typed kernels, not proofs of numerical implementation. Subcycling and conservative reflux are
   not encoded by this version and must not be inferred from its certificates."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.numerical-state :as state]))

(def schema-version 1)
(def operation-kinds #{:prolongation :restriction})
(def plan-modes #{:hierarchy-only :transfer-cycle})

(defrecord RefinementPatch [id level device offsets shape field attributes])
(defrecord RefinementLevel [id index ratio-to-parent patches attributes])
(defrecord RefinementHierarchy [id base-shape centering proper-nesting-width levels attributes])
(defrecord CoarseFineOperation
           [id kind source-patch target-patch source-region target-region operator attributes])
(defrecord ScheduledCoarseFine
           [operation route duration-ns dependencies steps completion])
(defrecord AMRWorkloadPlan
           [id schema-version mode hierarchy state distributed-plan coarse-fine attributes])
(defrecord AMRWorkloadCertificate
           [plan-id schema-version mode hierarchy state-certificate operations
            distributed-certificate cost-vector attributes])
(defrecord CertifiedAMRWorkload [plan certificate])

(defn- record-type?
  [class-name value]
  (and value (= class-name (.getName (class value)))))

(defn patch? [value] (record-type? "raster.compiler.ir.amr_plan.RefinementPatch" value))
(defn level? [value] (record-type? "raster.compiler.ir.amr_plan.RefinementLevel" value))
(defn hierarchy? [value] (record-type? "raster.compiler.ir.amr_plan.RefinementHierarchy" value))
(defn coarse-fine-operation?
  [value]
  (record-type? "raster.compiler.ir.amr_plan.CoarseFineOperation" value))
(defn scheduled-coarse-fine?
  [value]
  (record-type? "raster.compiler.ir.amr_plan.ScheduledCoarseFine" value))
(defn amr-plan? [value] (record-type? "raster.compiler.ir.amr_plan.AMRWorkloadPlan" value))
(defn certificate?
  [value]
  (record-type? "raster.compiler.ir.amr_plan.AMRWorkloadCertificate" value))
(defn certified-plan?
  [value]
  (record-type? "raster.compiler.ir.amr_plan.CertifiedAMRWorkload" value))

(defn- fail!
  [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

(defn- exact-keys!
  [label reason candidate allowed]
  (let [unexpected (vec (remove allowed (keys candidate)))]
    (when (seq unexpected)
      (fail! (str label " contains fields outside its versioned schema")
             reason {:unexpected unexpected :allowed allowed})))
  candidate)

(defn- unique-by!
  [label reason key-fn values]
  (let [ids (mapv key-fn values)]
    (when-not (= (count ids) (count (distinct ids)))
      (fail! (str label " must have unique identities") reason {:ids ids})))
  values)

(defn- rectangle!
  [label reason {:keys [offsets shape] :as region} rank]
  (when-not (and (map? region)
                 (vector? offsets) (= rank (count offsets))
                 (every? #(and (integer? %) (not (neg? %))) offsets)
                 (vector? shape) (= rank (count shape))
                 (every? pos-int? shape))
    (fail! (str label " requires rank-matched non-negative offsets and positive shape")
           reason {:region region :rank rank}))
  {:offsets offsets :shape shape})

(defn- within?
  [outer-offsets outer-shape inner-offsets inner-shape]
  (every? true?
          (map (fn [outer-offset outer-extent inner-offset inner-extent]
                 (and (<= outer-offset inner-offset)
                      (<= (+ inner-offset inner-extent) (+ outer-offset outer-extent))))
               outer-offsets outer-shape inner-offsets inner-shape)))

(defn- overlap?
  [left right]
  (every? true?
          (map (fn [lo ls ro rs]
                 (and (< lo (+ ro rs)) (< ro (+ lo ls))))
               (:offsets left) (:shape left) (:offsets right) (:shape right))))

(defn patch
  [{:keys [id level device offsets shape field attributes] :or {attributes {}}}]
  (when (or (nil? id) (nil? device) (nil? field) (not (nat-int? level)))
    (fail! "refinement patch requires identities, a device, a field, and a level index"
           :amr-patch-identity {:id id :level level :device device :field field}))
  (let [{:keys [offsets shape]} (rectangle! "refinement patch" :amr-patch-region
                                            {:offsets offsets :shape shape}
                                            (count offsets))]
    (when-not (map? attributes)
      (fail! "refinement patch attributes must be a map"
             :amr-patch-attributes {:patch id :attributes attributes}))
    (->RefinementPatch id level device offsets shape field attributes)))

(defn level
  [{:keys [id index ratio-to-parent patches attributes] :or {attributes {}}}]
  (when (or (nil? id) (not (nat-int? index)))
    (fail! "refinement level requires an identity and non-negative index"
           :amr-level-identity {:id id :index index}))
  (when-not (and (vector? patches) (seq patches) (every? patch? patches))
    (fail! "refinement level requires an ordered non-empty patch vector"
           :amr-level-patches {:level id :patches patches}))
  (when-not (or (and (zero? index) (nil? ratio-to-parent))
                (and (pos? index) (vector? ratio-to-parent) (seq ratio-to-parent)
                     (every? #(and (integer? %) (<= 2 %)) ratio-to-parent)))
    (fail! "only refined levels carry a per-axis refinement ratio of at least two"
           :amr-level-ratio {:level id :index index :ratio ratio-to-parent}))
  (when-not (map? attributes)
    (fail! "refinement level attributes must be a map"
           :amr-level-attributes {:level id :attributes attributes}))
  (->RefinementLevel id index ratio-to-parent (vec patches) attributes))

(defn- domain-shapes
  [base-shape levels]
  (reduce (fn [shapes candidate]
            (conj shapes (mapv * (peek shapes) (:ratio-to-parent candidate))))
          [base-shape]
          (next levels)))

(defn- validate-level!
  [rank domain-shape parent candidate proper-nesting-width]
  (exact-keys! "refinement level" :amr-level-fields candidate
               #{:id :index :ratio-to-parent :patches :attributes})
  (level candidate)
  (when (and (pos? (:index candidate))
             (not= rank (count (:ratio-to-parent candidate))))
    (fail! "refinement ratio rank differs from the hierarchy"
           :amr-level-ratio-rank
           {:level (:id candidate) :rank rank :ratio (:ratio-to-parent candidate)}))
  (unique-by! "refinement patches" :amr-patch-identities :id (:patches candidate))
  (doseq [candidate-patch (:patches candidate)]
    (exact-keys! "refinement patch" :amr-patch-fields candidate-patch
                 #{:id :level :device :offsets :shape :field :attributes})
    (patch candidate-patch)
    (when-not (= (:index candidate) (:level candidate-patch))
      (fail! "patch level differs from its containing refinement level"
             :amr-patch-level
             {:patch (:id candidate-patch) :patch-level (:level candidate-patch)
              :level (:index candidate)}))
    (when-not (= rank (count (:shape candidate-patch)))
      (fail! "patch rank differs from its hierarchy"
             :amr-patch-rank {:patch (:id candidate-patch) :rank rank
                              :shape (:shape candidate-patch)}))
    (when-not (within? (vec (repeat rank 0)) domain-shape
                       (:offsets candidate-patch) (:shape candidate-patch))
      (fail! "patch lies outside its level domain"
             :amr-patch-bounds {:patch (:id candidate-patch)
                                :domain-shape domain-shape}))
    (when (pos? (:index candidate))
      (let [ratio (:ratio-to-parent candidate)]
        (when-not (every? zero? (concat (map mod (:offsets candidate-patch) ratio)
                                        (map mod (:shape candidate-patch) ratio)))
          (fail! "refined patch boundaries must align to parent cells"
                 :amr-patch-alignment
                 {:patch (:id candidate-patch) :ratio ratio
                  :offsets (:offsets candidate-patch) :shape (:shape candidate-patch)}))
        (let [coarse-offsets (mapv quot (:offsets candidate-patch) ratio)
              coarse-shape (mapv quot (:shape candidate-patch) ratio)
              nested-offsets (mapv #(- % proper-nesting-width) coarse-offsets)
              nested-shape (mapv #(+ % (* 2 proper-nesting-width)) coarse-shape)]
          (when-not (some #(and (every? (complement neg?) nested-offsets)
                                (within? (:offsets %) (:shape %)
                                         nested-offsets nested-shape))
                          (:patches parent))
            (fail! "refined patch lacks the requested proper-nesting parent margin"
                   :amr-proper-nesting
                   {:patch (:id candidate-patch) :margin proper-nesting-width
                    :coarsened {:offsets coarse-offsets :shape coarse-shape}}))))))
  (doseq [[left-index left] (map-indexed vector (:patches candidate))
          right (drop (inc left-index) (:patches candidate))]
    (when (overlap? left right)
      (fail! "patch interiors on one level must not overlap"
             :amr-patch-overlap {:level (:id candidate)
                                 :left (:id left) :right (:id right)})))
  candidate)

(defn validate-hierarchy!
  [candidate]
  (when-not (hierarchy? candidate)
    (fail! "expected a RefinementHierarchy" :amr-hierarchy-type
           {:actual (type candidate)}))
  (exact-keys! "refinement hierarchy" :amr-hierarchy-fields candidate
               #{:id :base-shape :centering :proper-nesting-width :levels :attributes})
  (let [{:keys [id base-shape centering proper-nesting-width levels attributes]} candidate]
    (when (nil? id)
      (fail! "refinement hierarchy requires an identity" :amr-hierarchy-id {}))
    (when-not (and (vector? base-shape) (seq base-shape) (every? pos-int? base-shape))
      (fail! "refinement hierarchy requires a positive concrete base shape"
             :amr-base-shape {:shape base-shape}))
    (when-not (= :cell centering)
      (fail! "version 1 supports cell-centred fields only"
             :amr-centering {:centering centering}))
    (when-not (nat-int? proper-nesting-width)
      (fail! "proper nesting width must be a non-negative integer"
             :amr-proper-nesting-width {:width proper-nesting-width}))
    (when-not (and (vector? levels) (seq levels) (every? level? levels))
      (fail! "refinement hierarchy requires an ordered non-empty level vector"
             :amr-levels {:levels levels}))
    (unique-by! "refinement levels" :amr-level-identities :id levels)
    (when-not (= (vec (range (count levels))) (mapv :index levels))
      (fail! "refinement levels must be in canonical contiguous index order"
             :amr-level-order {:indices (mapv :index levels)}))
    (let [base (first levels)
          base-patches (:patches base)]
      (when-not (and (= 1 (count base-patches))
                     (= (vec (repeat (count base-shape) 0)) (:offsets (first base-patches)))
                     (= base-shape (:shape (first base-patches))))
        (fail! "version 1 requires one base patch covering the complete domain"
               :amr-base-coverage {:base-shape base-shape :patches base-patches})))
    (let [shapes (domain-shapes base-shape levels)]
      (doseq [[index candidate] (map-indexed vector levels)]
        (validate-level! (count base-shape) (nth shapes index)
                         (when (pos? index) (nth levels (dec index)))
                         candidate proper-nesting-width)))
    (unique-by! "hierarchy patches" :amr-hierarchy-patch-identities
                :id (mapcat :patches levels))
    (unique-by! "hierarchy patch fields" :amr-hierarchy-field-identities
                :field (mapcat :patches levels))
    (when-not (map? attributes)
      (fail! "refinement hierarchy attributes must be a map"
             :amr-hierarchy-attributes {:attributes attributes})))
  candidate)

(defn hierarchy
  [{:keys [id base-shape centering proper-nesting-width levels attributes]
    :or {centering :cell proper-nesting-width 1 attributes {}}}]
  (validate-hierarchy!
   (->RefinementHierarchy id (vec base-shape) centering proper-nesting-width
                          (vec levels) attributes)))

(defn- patch-index
  [hierarchy]
  (into {} (map (juxt :id identity)) (mapcat :patches (:levels hierarchy))))

(defn coarse-fine-operation
  [{:keys [id kind source-patch target-patch source-region target-region operator attributes]
    :or {attributes {}}}]
  (when (or (nil? id) (not (contains? operation-kinds kind))
            (nil? source-patch) (nil? target-patch) (= source-patch target-patch))
    (fail! "coarse/fine operation requires an identity, kind, and distinct patches"
           :amr-operation-identity
           {:id id :kind kind :source source-patch :target target-patch}))
  (when-not (and (map? operator) (keyword? (:method operator))
                 (set? (:required-invariants operator))
                 (seq (:required-invariants operator))
                 (every? keyword? (:required-invariants operator)))
    (fail! "coarse/fine operator requires a method and non-empty requirement set"
           :amr-operation-operator {:operation id :operator operator}))
  (when-not (map? attributes)
    (fail! "coarse/fine operation attributes must be a map"
           :amr-operation-attributes {:operation id :attributes attributes}))
  (->CoarseFineOperation id kind source-patch target-patch source-region target-region
                         operator attributes))

(defn- absolute-region
  [patch region]
  {:offsets (mapv + (:offsets patch) (:offsets region))
   :shape (:shape region)})

(defn- validate-operation!
  [hierarchy candidate]
  (when-not (coarse-fine-operation? candidate)
    (fail! "expected a CoarseFineOperation" :amr-operation-type
           {:actual (type candidate)}))
  (exact-keys! "coarse/fine operation" :amr-operation-fields candidate
               #{:id :kind :source-patch :target-patch :source-region :target-region
                 :operator :attributes})
  (coarse-fine-operation candidate)
  (let [patches (patch-index hierarchy)
        source (get patches (:source-patch candidate))
        target (get patches (:target-patch candidate))]
    (when-not (and source target)
      (fail! "coarse/fine operation references an undeclared patch"
             :amr-operation-patch
             {:operation (:id candidate) :source (:source-patch candidate)
              :target (:target-patch candidate)}))
    (let [expected-levels (case (:kind candidate)
                            :prolongation [(:level source) (inc (:level source))]
                            :restriction [(:level source) (dec (:level source))])]
      (when-not (= expected-levels [(:level source) (:level target)])
        (fail! "coarse/fine operation must connect adjacent levels in the declared direction"
               :amr-operation-levels
               {:operation (:id candidate) :kind (:kind candidate)
                :source-level (:level source) :target-level (:level target)})))
    (let [source-region (rectangle! "source region" :amr-operation-source-region
                                    (:source-region candidate) (count (:shape source)))
          target-region (rectangle! "target region" :amr-operation-target-region
                                    (:target-region candidate) (count (:shape target)))]
      (when-not (within? (vec (repeat (count (:shape source)) 0)) (:shape source)
                         (:offsets source-region) (:shape source-region))
        (fail! "source region lies outside its patch"
               :amr-operation-source-bounds {:operation (:id candidate)}))
      (when-not (within? (vec (repeat (count (:shape target)) 0)) (:shape target)
                         (:offsets target-region) (:shape target-region))
        (fail! "target region lies outside its patch"
               :amr-operation-target-bounds {:operation (:id candidate)}))
      (let [[coarse fine coarse-region fine-region]
            (if (= :prolongation (:kind candidate))
              [source target source-region target-region]
              [target source target-region source-region])
            ratio (:ratio-to-parent (nth (:levels hierarchy) (:level fine)))
            coarse-absolute (absolute-region coarse coarse-region)
            fine-absolute (absolute-region fine fine-region)]
        (when-not (and (= (mapv * (:offsets coarse-absolute) ratio)
                          (:offsets fine-absolute))
                       (= (mapv * (:shape coarse-absolute) ratio)
                          (:shape fine-absolute)))
          (fail! "coarse and fine regions do not describe the same aligned cells"
                 :amr-operation-region-mapping
                 {:operation (:id candidate) :ratio ratio
                  :coarse coarse-absolute :fine fine-absolute}))))
    candidate))

(defn- region-elements
  [region]
  (reduce * 1 (:shape region)))

(defn schedule-coarse-fine
  "Plan one semantic adjacent-level operation as explicit distributed transfer/compute steps.

   `values` is the DistributedPlan value map. Cross-device operations require a directed route;
   same-device operations require an empty route. The source rectangle determines transfer bytes.
   The resulting compute step declares read/write roles but is not an executable kernel binding."
  [hierarchy values operation {:keys [route duration-ns dependencies]
                               :or {route [] dependencies []}}]
  (let [hierarchy (validate-hierarchy! hierarchy)
        operation (validate-operation! hierarchy operation)
        patches (patch-index hierarchy)
        source (get patches (:source-patch operation))
        target (get patches (:target-patch operation))
        source-value (get values (:field source))
        target-value (get values (:field target))
        cross-device? (not= (:device source) (:device target))]
    (when-not (and (abstract-value/abstract-value? source-value)
                   (abstract-value/abstract-value? target-value))
      (fail! "coarse/fine patches require distributed AbstractValues"
             :amr-operation-value
             {:operation (:id operation) :source-field (:field source)
              :target-field (:field target)}))
    (when-not (abstract-value/storage-contract-compatible? source-value target-value)
      (fail! "version 1 coarse/fine operations require one logical storage contract"
             :amr-operation-storage-contract
             {:operation (:id operation) :source-field (:field source)
              :target-field (:field target)}))
    (when-not (and (number? duration-ns) (Double/isFinite (double duration-ns))
                   (pos? duration-ns))
      (fail! "coarse/fine compute duration must be positive and finite"
             :amr-operation-duration {:operation (:id operation) :duration-ns duration-ns}))
    (when-not (and (vector? dependencies)
                   (= (count dependencies) (count (distinct dependencies))))
      (fail! "coarse/fine dependencies must be a unique ordered vector"
             :amr-operation-dependencies
             {:operation (:id operation) :dependencies dependencies}))
    (when-not (if cross-device? (and (vector? route) (seq route)) (empty? route))
      (fail! "coarse/fine route must be present exactly for cross-device placement"
             :amr-operation-route
             {:operation (:id operation) :source (:device source) :target (:device target)
              :route route}))
    (let [bytes (* (region-elements (:source-region operation))
                   (dtype/bytes-of (:dtype source-value)))
          transfer-id [(:id operation) :transfer]
          apply-id [(:id operation) :apply]
          common {:amr-operation (:id operation) :amr-kind (:kind operation)
                  :source-patch (:id source) :target-patch (:id target)
                  :source-field (:field source) :target-field (:field target)
                  :access {:source :read :target :write}
                  :source-region (:source-region operation)
                  :target-region (:target-region operation)
                  :operator (:operator operation)}
          transfer (when cross-device?
                     (distributed/transfer-step
                      {:id transfer-id :source (:device source) :target (:device target)
                       :route route :value (:field source) :bytes bytes
                       :dependencies dependencies :attributes common}))
          compute (distributed/compute-step
                   {:id apply-id :device (:device target) :duration-ns duration-ns
                    :dependencies (if transfer [transfer-id] dependencies)
                    :attributes common})
          steps (cond-> [] transfer (conj transfer) true (conj compute))]
      (->ScheduledCoarseFine operation (vec route) duration-ns (vec dependencies)
                             steps apply-id))))

(defn- validate-patch-bindings!
  [hierarchy certified-state certified-distributed]
  (let [manifest (:manifest certified-state)
        distributed-plan (:plan certified-distributed)
        fields (into {} (map (juxt :id identity)) (:fields manifest))
        values (:values distributed-plan)
        shards (:shards distributed-plan)
        base-patch (-> hierarchy :levels first :patches first)
        base-axes (get-in fields [(:field base-patch) :coordinate-space :axes])
        canonical-axis-names (when (vector? base-axes) (mapv :name base-axes))]
    (doseq [candidate (mapcat :patches (:levels hierarchy))]
      (let [field (get fields (:field candidate))
            value (get values (:field candidate))
            candidates (get shards (:field candidate))
            axes (get-in field [:coordinate-space :axes])
            expected-coordinates {:hierarchy (:id hierarchy)
                                  :level (:level candidate)
                                  :patch (:id candidate)}]
        (when-not (and field value)
          (fail! "every AMR patch must bind one durable and distributed field"
                 :amr-patch-field-binding {:patch (:id candidate) :field (:field candidate)}))
        (when-not (= (:shape candidate) (get-in field [:value :shape]))
          (fail! "durable field shape differs from its AMR patch"
                 :amr-patch-state-shape {:patch (:id candidate) :shape (:shape candidate)
                                         :field-shape (get-in field [:value :shape])}))
        (when-not (= expected-coordinates
                     (select-keys (:coordinate-space field) [:hierarchy :level :patch]))
          (fail! "durable field coordinate identity differs from its AMR patch"
                 :amr-patch-coordinate-space
                 {:patch (:id candidate) :expected expected-coordinates
                  :actual (:coordinate-space field)}))
        (when-not (and (vector? axes) (= (count (:base-shape hierarchy)) (count axes))
                       (every? #(and (map? %) (keyword? (:name %))
                                     (= (:centering hierarchy) (:centering %))) axes)
                       (= (count axes) (count (distinct (map :name axes))))
                       (= canonical-axis-names (mapv :name axes)))
          (fail! "AMR fields require canonical rank-matched cell-centred axes"
                 :amr-patch-coordinate-axes
                 {:patch (:id candidate) :expected-axis-names canonical-axis-names
                  :centering (:centering hierarchy) :axes axes}))
        (when-not (and (= (:shape candidate) (:shape value))
                       (abstract-value/storage-contract-compatible? (:value field) value))
          (fail! "distributed value disagrees with the durable patch field contract"
                 :amr-patch-value-contract
                 {:patch (:id candidate) :field (:field candidate)}))
        (when-not (and (= 1 (count candidates))
                       (= (:device candidate) (:device (first candidates)))
                       (= (vec (repeat (count (:shape candidate)) 0))
                          (:offsets (first candidates)))
                       (= (:shape candidate) (:shape (first candidates)))
                       (= :owned (:ownership (first candidates))))
          (fail! "version 1 requires one full owned distributed shard per AMR patch"
                 :amr-patch-shard
                 {:patch (:id candidate) :field (:field candidate)
                  :device (:device candidate) :shards candidates}))))))

(defn- full-local-region
  [candidate]
  {:offsets (vec (repeat (count (:shape candidate)) 0))
   :shape (:shape candidate)})

(defn- validate-plan-mode!
  [mode hierarchy coarse-fine]
  (case mode
    :hierarchy-only
    (when (seq coarse-fine)
      (fail! "hierarchy-only AMR plans cannot schedule coarse/fine execution"
             :amr-plan-mode {:mode mode :operations (mapv (comp :id :operation) coarse-fine)}))

    :transfer-cycle
    (let [patches (patch-index hierarchy)
          refined (vec (filter #(pos? (:level %)) (vals patches)))
          expected (set (mapcat (fn [candidate]
                                  [[:prolongation (:id candidate)]
                                   [:restriction (:id candidate)]])
                                refined))
          actual (mapv (fn [{:keys [operation]}]
                         [(:kind operation)
                          (case (:kind operation)
                            :prolongation (:target-patch operation)
                            :restriction (:source-patch operation))])
                       coarse-fine)]
      (when-not (and (seq refined) (= expected (set actual)) (= (count expected) (count actual)))
        (fail! "transfer-cycle requires exactly one prolongation and restriction per refined patch"
               :amr-plan-operation-coverage
               {:expected expected :actual actual}))
      (doseq [{:keys [operation]} coarse-fine]
        (let [fine-patch (get patches (case (:kind operation)
                                        :prolongation (:target-patch operation)
                                        :restriction (:source-patch operation)))
              fine-region (case (:kind operation)
                            :prolongation (:target-region operation)
                            :restriction (:source-region operation))]
          (when-not (= (full-local-region fine-patch) fine-region)
            (fail! "version 1 transfer-cycle operations must cover their complete fine patch"
                   :amr-plan-operation-region-coverage
                   {:operation (:id operation) :patch (:id fine-patch)
                    :expected (full-local-region fine-patch) :actual fine-region}))))))
  coarse-fine)

(defn- validate-operation-step-coverage!
  [coarse-fine steps]
  (let [expected (set (map :id (mapcat :steps coarse-fine)))
        tagged (set (map :id (filter #(contains? (:attributes %) :amr-operation) steps)))]
    (when-not (= expected tagged)
      (fail! "AMR-tagged distributed steps must exactly match scheduled coarse/fine expansions"
             :amr-plan-operation-step-coverage
             {:expected expected :actual tagged})))
  steps)

(defn validate!
  "Validate and return an AMRWorkloadPlan without realizing storage or communication resources."
  [candidate]
  (when-not (amr-plan? candidate)
    (fail! "expected an AMRWorkloadPlan" :amr-plan-type {:actual (type candidate)}))
  (exact-keys! "AMR workload plan" :amr-plan-fields candidate
               #{:id :schema-version :mode :hierarchy :state :distributed-plan
                 :coarse-fine :attributes})
  (let [{:keys [id schema-version mode hierarchy state distributed-plan coarse-fine attributes]}
        candidate]
    (when (nil? id)
      (fail! "AMR workload plan requires an identity" :amr-plan-id {}))
    (when-not (= raster.compiler.ir.amr-plan/schema-version schema-version)
      (fail! "AMR workload schema version is unsupported"
             :amr-schema-version
             {:expected raster.compiler.ir.amr-plan/schema-version :actual schema-version}))
    (when-not (contains? plan-modes mode)
      (fail! "AMR workload requires an explicit supported mode"
             :amr-plan-mode {:mode mode :allowed plan-modes}))
    (validate-hierarchy! hierarchy)
    (state/verify! state)
    (distributed/verify! distributed-plan)
    (validate-patch-bindings! hierarchy state distributed-plan)
    (when-not (and (vector? coarse-fine) (every? scheduled-coarse-fine? coarse-fine))
      (fail! "AMR coarse/fine operations must be an ordered ScheduledCoarseFine vector"
             :amr-plan-operations {:coarse-fine coarse-fine}))
    (unique-by! "coarse/fine operations" :amr-operation-identities
                (comp :id :operation) coarse-fine)
    (validate-plan-mode! mode hierarchy coarse-fine)
    (let [values (get-in distributed-plan [:plan :values])
          distributed-steps (get-in distributed-plan [:plan :steps])
          step-by-id (into {} (map (juxt :id identity))
                           distributed-steps)]
      (validate-operation-step-coverage! coarse-fine distributed-steps)
      (doseq [{:keys [operation route duration-ns dependencies steps completion] :as scheduled}
              coarse-fine]
        (let [expected (schedule-coarse-fine hierarchy values operation
                                             {:route route :duration-ns duration-ns
                                              :dependencies dependencies})]
          (when-not (= expected scheduled)
            (fail! "scheduled coarse/fine lowering differs from its semantic operation"
                   :amr-operation-lowering
                   {:operation (:id operation) :expected expected :actual scheduled}))
          (doseq [step steps]
            (when-not (= step (get step-by-id (:id step)))
              (fail! "coarse/fine expansion step is absent or differs from the distributed DAG"
                     :amr-operation-step {:operation (:id operation) :step (:id step)})))
          (when-not (= completion (:id (peek steps)))
            (fail! "coarse/fine completion must name its final compute step"
                   :amr-operation-completion
                   {:operation (:id operation) :completion completion})))))
    (when-not (map? attributes)
      (fail! "AMR workload attributes must be a map"
             :amr-plan-attributes {:attributes attributes})))
  candidate)

(defn plan
  [{:keys [id schema-version mode hierarchy state distributed-plan coarse-fine attributes]
    :or {schema-version raster.compiler.ir.amr-plan/schema-version
         coarse-fine [] attributes {}}}]
  (validate!
   (->AMRWorkloadPlan id schema-version mode hierarchy state distributed-plan
                      (vec coarse-fine) attributes)))

(defn simulate
  "Interpret the certified distributed schedule underlying an AMR workload."
  [candidate]
  (distributed/simulate (get-in (validate! candidate) [:distributed-plan :plan])))

(defn- hierarchy-witness
  [hierarchy]
  {:id (:id hierarchy)
   :base-shape (:base-shape hierarchy)
   :centering (:centering hierarchy)
   :proper-nesting-width (:proper-nesting-width hierarchy)
   :attributes (:attributes hierarchy)
   :levels
   (mapv (fn [candidate]
           {:id (:id candidate) :index (:index candidate)
            :ratio-to-parent (:ratio-to-parent candidate)
            :attributes (:attributes candidate)
            :patches (mapv #(select-keys % [:id :device :offsets :shape :field :attributes])
                           (:patches candidate))})
         (:levels hierarchy))})

(defn- derive-certificate
  [candidate]
  (let [distributed-certificate (get-in candidate [:distributed-plan :certificate])
        operations
        (mapv (fn [{:keys [operation route duration-ns dependencies steps completion]}]
                {:id (:id operation) :kind (:kind operation)
                 :source-patch (:source-patch operation)
                 :target-patch (:target-patch operation)
                 :source-region (:source-region operation)
                 :target-region (:target-region operation)
                 :operator (:operator operation)
                 :attributes (:attributes operation)
                 :route route :duration-ns duration-ns :dependencies dependencies
                 :steps (mapv :id steps) :completion completion
                 :bytes (reduce + 0 (keep :bytes steps))})
              (:coarse-fine candidate))]
    (->AMRWorkloadCertificate
     (:id candidate) (:schema-version candidate) (:mode candidate)
     (hierarchy-witness (:hierarchy candidate))
     (get-in candidate [:state :certificate]) operations distributed-certificate
     (:cost-vector distributed-certificate) (:attributes candidate))))

(defn certify
  "Validate an AMR workload and attach hierarchy, state, communication, and cost witnesses."
  [candidate]
  (let [candidate (validate! candidate)]
    (->CertifiedAMRWorkload candidate (derive-certificate candidate))))

(defn verify!
  "Revalidate a CertifiedAMRWorkload and independently derive its certificate."
  [certified]
  (when-not (certified-plan? certified)
    (fail! "expected a CertifiedAMRWorkload"
           :amr-certified-plan-type {:actual (type certified)}))
  (let [candidate (validate! (:plan certified))
        expected (derive-certificate candidate)]
    (when-not (certificate? (:certificate certified))
      (fail! "AMR workload certificate has the wrong type"
             :amr-certificate-type {:actual (type (:certificate certified))}))
    (when-not (= expected (:certificate certified))
      (fail! "AMR workload certificate does not match its plan"
             :amr-certificate {:expected expected :actual (:certificate certified)}))
    certified))
