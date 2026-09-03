(ns raster.compiler.ir.distributed-plan
  "Certified, backend-neutral planning across a device topology.

   A DistributedPlan does not contain communicator, queue, event, or buffer handles.  It maps
   AbstractValues to checked shard rectangles, embeds optional shard-local LinkPlan/ExecutionPlan
   values, and orders compute and point-to-point transfer steps.  `simulate` interprets that
   explicit schedule over exclusive per-device compute lanes and directed links, allowing compute
   and communication to overlap while accounting for link bandwidth, latency, bytes, and device
   memory capacity.

   Native MPI/NCCL/RCCL/oneCCL/UCX resources are later realizations of this contract."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.execution-plan :as execution-plan]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.scan :as scan]))

(def shard-kinds #{:replicated :partitioned})
(def shard-ownerships #{:owned :replica})
(def step-kinds #{:compute :transfer})
(def collective-kinds #{:all-reduce :all-gather :reduce-scatter :broadcast})
(def halo-boundaries #{:nonperiodic :periodic})
(def halo-destination-modes #{:copy :combine})

(defrecord DeviceMesh [axes devices])
(defrecord DeviceResource [id memory-capacity-bytes descriptor attributes])
(defrecord TopologyLink [id source target kind bandwidth-bytes-s latency-ns attributes])
(defrecord ClusterTopology [devices links])
(defrecord ValueShard [id value device offsets shape ownership])
(defrecord CollectiveGroup [id devices])
(defrecord CommunicationLeg [source target route bytes attributes])
(defrecord CollectiveOperation [id kind group value reduction root attributes])
(defrecord CollectiveSchedule [algorithm rounds numerical-mode attributes])
(defrecord ScheduledCollective [operation schedule dependencies steps completions])
(defrecord HaloExchange [id value axis width boundary combine attributes])
(defrecord ScheduledHalo [exchange routes dependencies steps completions])
(defrecord DistributedStep
           [id kind device source target route value bytes duration-ns dependencies
            peak-memory-bytes attributes])
(defrecord DistributedPlan
           [id mesh topology values shards collective-groups collectives
            halos device-plans steps outputs attributes])
(defrecord DistributedPlanCertificate
           [plan-id mesh-shape shard-coverage collectives halos
            route-costs cost-vector device-plans])
(defrecord CertifiedDistributedPlan [plan certificate])

(defn device-mesh? [value] (instance? DeviceMesh value))
(defn device-resource? [value] (instance? DeviceResource value))
(defn topology-link? [value] (instance? TopologyLink value))
(defn cluster-topology? [value] (instance? ClusterTopology value))
(defn value-shard? [value] (instance? ValueShard value))
(defn collective-group? [value] (instance? CollectiveGroup value))
(defn communication-leg? [value] (instance? CommunicationLeg value))
(defn collective-operation? [value] (instance? CollectiveOperation value))
(defn collective-schedule? [value] (instance? CollectiveSchedule value))
(defn scheduled-collective? [value] (instance? ScheduledCollective value))
(defn halo-exchange? [value] (instance? HaloExchange value))
(defn scheduled-halo? [value] (instance? ScheduledHalo value))
(defn distributed-step? [value] (instance? DistributedStep value))
(defn distributed-plan? [value] (instance? DistributedPlan value))
(defn certificate? [value] (instance? DistributedPlanCertificate value))
(defn certified-plan? [value] (instance? CertifiedDistributedPlan value))

(defn- fail!
  [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

(defn- finite-number?
  [value]
  (and (number? value) (Double/isFinite (double value))))

(defn- non-negative-number?
  [value]
  (and (finite-number? value) (not (neg? value))))

(defn- positive-number?
  [value]
  (and (finite-number? value) (pos? value)))

(defn- unique-by!
  [label reason key-fn values]
  (let [ids (mapv key-fn values)]
    (when-not (= (count ids) (count (distinct ids)))
      (fail! (str label " must have unique identities") reason {:ids ids})))
  values)

(defn mesh
  "Construct a rectangular row-major device mesh.

   `axes` is an ordered vector such as `[{:name :data :size 2} {:name :model :size 4}]`.
   `devices` contains exactly the product of axis sizes."
  [axes devices]
  (let [axes (vec axes)
        devices (vec devices)]
    (when-not (and (seq axes)
                   (every? #(and (map? %) (keyword? (:name %))
                                 (pos-int? (:size %))) axes))
      (fail! "device mesh axes must have keyword names and positive integer sizes"
             :distributed-mesh-axes {:axes axes}))
    (unique-by! "device mesh axes" :distributed-mesh-axis-identities :name axes)
    (when (some nil? devices)
      (fail! "device mesh identities must be non-nil"
             :distributed-mesh-device {:devices devices}))
    (unique-by! "device mesh devices" :distributed-mesh-device-identities identity devices)
    (let [expected (reduce * 1 (map :size axes))]
      (when-not (= expected (count devices))
        (fail! "device mesh extent differs from its row-major device vector"
               :distributed-mesh-extent
               {:axes axes :expected expected :actual (count devices)})))
    (->DeviceMesh axes devices)))

(defn device
  [{:keys [id memory-capacity-bytes descriptor attributes]
    :or {attributes {}}}]
  (when (nil? id)
    (fail! "topology device requires an identity" :distributed-device-id {}))
  (when-not (and (integer? memory-capacity-bytes) (not (neg? memory-capacity-bytes)))
    (fail! "topology device memory capacity must be a non-negative integer byte count"
           :distributed-device-memory
           {:device id :memory-capacity-bytes memory-capacity-bytes}))
  (when-not (or (nil? descriptor) (map? descriptor))
    (fail! "topology device descriptor must be a map or nil"
           :distributed-device-descriptor {:device id :descriptor descriptor}))
  (when-not (map? attributes)
    (fail! "topology device attributes must be a map"
           :distributed-device-attributes {:device id :attributes attributes}))
  (->DeviceResource id memory-capacity-bytes descriptor attributes))

(defn link
  "Construct one directed topology link. Bidirectional fabrics use two directed links."
  [{:keys [id source target kind bandwidth-bytes-s latency-ns attributes]
    :or {kind :interconnect attributes {}}}]
  (when (or (nil? id) (nil? source) (nil? target) (= source target))
    (fail! "topology link requires distinct non-nil endpoints and an identity"
           :distributed-link-identity
           {:id id :source source :target target}))
  (when-not (keyword? kind)
    (fail! "topology link kind must be a keyword"
           :distributed-link-kind {:link id :kind kind}))
  (when-not (positive-number? bandwidth-bytes-s)
    (fail! "topology link bandwidth must be positive and finite"
           :distributed-link-bandwidth {:link id :bandwidth-bytes-s bandwidth-bytes-s}))
  (when-not (non-negative-number? latency-ns)
    (fail! "topology link latency must be non-negative and finite"
           :distributed-link-latency {:link id :latency-ns latency-ns}))
  (when-not (map? attributes)
    (fail! "topology link attributes must be a map"
           :distributed-link-attributes {:link id :attributes attributes}))
  (->TopologyLink id source target kind bandwidth-bytes-s latency-ns attributes))

(defn topology
  "Construct a topology from DeviceResource and directed TopologyLink values."
  [devices links]
  (let [devices (vec devices)
        links (vec links)]
    (when-not (every? device-resource? devices)
      (fail! "cluster topology devices must be DeviceResource values"
             :distributed-topology-device-type {:devices devices}))
    (when-not (every? topology-link? links)
      (fail! "cluster topology links must be TopologyLink values"
             :distributed-topology-link-type {:links links}))
    (unique-by! "cluster topology devices" :distributed-topology-device-identities :id devices)
    (unique-by! "cluster topology links" :distributed-topology-link-identities :id links)
    (let [device-ids (set (map :id devices))]
      (doseq [candidate links]
        (when-not (and (contains? device-ids (:source candidate))
                       (contains? device-ids (:target candidate)))
          (fail! "topology link endpoint is not a declared device"
                 :distributed-link-endpoint
                 {:link (:id candidate) :source (:source candidate)
                  :target (:target candidate) :devices device-ids}))))
    (->ClusterTopology (into {} (map (juxt :id identity)) devices)
                       (into {} (map (juxt :id identity)) links))))

(defn shard
  [{:keys [id value device offsets shape ownership]
    :or {ownership :owned}}]
  (when (or (nil? id) (nil? value) (nil? device))
    (fail! "value shard requires shard, value, and device identities"
           :distributed-shard-identity {:id id :value value :device device}))
  (when-not (and (vector? offsets) (every? #(and (integer? %) (not (neg? %))) offsets)
                 (vector? shape) (every? pos-int? shape)
                 (= (count offsets) (count shape)))
    (fail! "value shard requires equal-rank non-negative offsets and positive shapes"
           :distributed-shard-shape {:id id :offsets offsets :shape shape}))
  (when-not (contains? shard-ownerships ownership)
    (fail! "value shard has an unsupported ownership contract"
           :distributed-shard-ownership
           {:id id :ownership ownership :allowed shard-ownerships}))
  (->ValueShard id value device offsets shape ownership))

(defn compute-step
  [{:keys [id device duration-ns dependencies peak-memory-bytes attributes]
    :or {dependencies [] peak-memory-bytes 0 attributes {}}}]
  (->DistributedStep id :compute device nil nil [] nil nil duration-ns
                     (vec dependencies) peak-memory-bytes attributes))

(defn transfer-step
  [{:keys [id source target route value bytes dependencies attributes]
    :or {dependencies [] attributes {}}}]
  (->DistributedStep id :transfer nil source target (vec route) value bytes nil
                     (vec dependencies) 0 attributes))

(defn collective-group
  [id devices]
  (let [devices (vec devices)]
    (when (or (nil? id) (< (count devices) 2) (some nil? devices)
              (not= (count devices) (count (distinct devices))))
      (fail! "collective group requires an identity and at least two unique devices"
             :distributed-collective-group {:id id :devices devices}))
    (->CollectiveGroup id devices)))

(defn communication-leg
  [{:keys [source target route bytes attributes] :or {attributes {}}}]
  (when-not (and source target (not= source target)
                 (vector? route) (seq route)
                 (integer? bytes) (not (neg? bytes))
                 (map? attributes))
    (fail! "communication leg requires distinct endpoints, a route, and non-negative bytes"
           :distributed-communication-leg
           {:source source :target target :route route :bytes bytes :attributes attributes}))
  (->CommunicationLeg source target route bytes attributes))

(defn collective-operation
  [{:keys [id kind group value reduction root attributes] :or {attributes {}}}]
  (when (or (nil? id) (not (contains? collective-kinds kind))
            (nil? group) (nil? value) (not (map? attributes)))
    (fail! "collective operation requires an identity, kind, group, value, and attributes"
           :distributed-collective-operation
           {:id id :kind kind :group group :value value :attributes attributes}))
  (when (and (contains? #{:all-reduce :reduce-scatter} kind)
             (not (scan/associative-scan? reduction)))
    (fail! "reducing collective requires a certified associative reduction"
           :distributed-collective-reduction {:id id :kind kind :reduction reduction}))
  (when (and (not (contains? #{:all-reduce :reduce-scatter} kind)) reduction)
    (fail! "non-reducing collective cannot carry a reduction contract"
           :distributed-collective-reduction {:id id :kind kind :reduction reduction}))
  (when (and (= :broadcast kind) (nil? root))
    (fail! "broadcast collective requires a root device"
           :distributed-collective-root {:id id}))
  (when (and (not= :broadcast kind) root)
    (fail! "only broadcast collectives may carry a root device"
           :distributed-collective-root {:id id :kind kind :root root}))
  (->CollectiveOperation id kind group value reduction root attributes))

(defn collective-schedule
  [{:keys [algorithm rounds numerical-mode attributes]
    :or {numerical-mode {} attributes {}}}]
  (let [rounds (mapv (fn [round] (mapv #(if (communication-leg? %)
                                          % (communication-leg %)) round)) rounds)]
    (when-not (and (keyword? algorithm) (seq rounds) (every? seq rounds)
                   (map? numerical-mode) (map? attributes))
      (fail! "collective schedule requires an algorithm and non-empty communication rounds"
             :distributed-collective-schedule
             {:algorithm algorithm :rounds rounds :numerical-mode numerical-mode
              :attributes attributes}))
    (doseq [[round-index round] (map-indexed vector rounds)]
      (let [link-ids (mapcat :route round)]
        (when-not (= (count link-ids) (count (distinct link-ids)))
          (fail! "parallel collective legs cannot claim the same directed link"
                 :distributed-collective-round-link-conflict
                 {:round round-index :links (vec link-ids)}))))
    (->CollectiveSchedule algorithm rounds numerical-mode attributes)))

(defn schedule-collective
  "Expand a semantic collective's communication schedule to topology-routed transfer steps.

   Each round is a barrier: its legs may overlap, and the next round waits for all of them. The
   semantic CollectiveOperation is retained beside these cost/execution steps."
  [operation schedule dependencies]
  (let [operation (if (collective-operation? operation) operation
                      (collective-operation operation))
        schedule (if (collective-schedule? schedule) schedule
                     (collective-schedule schedule))
        dependencies (vec dependencies)
        {:keys [steps completions]}
        (reduce
         (fn [{:keys [steps completions]} [round-index legs]]
           (let [waits (if (zero? round-index) dependencies completions)
                 round-steps
                 (mapv (fn [leg-index leg]
                         (transfer-step
                          {:id [(:id operation) :round round-index :leg leg-index]
                           :source (:source leg) :target (:target leg)
                           :route (:route leg) :value (:value operation) :bytes (:bytes leg)
                           :dependencies waits
                           :attributes (merge (:attributes leg)
                                              {:collective (:id operation)
                                               :collective-kind (:kind operation)
                                               :algorithm (:algorithm schedule)
                                               :round round-index :leg leg-index})}))
                       (range) legs)]
             {:steps (into steps round-steps)
              :completions (mapv :id round-steps)}))
         {:steps [] :completions []}
         (map-indexed vector (:rounds schedule)))]
    (->ScheduledCollective operation schedule dependencies steps completions)))

(defn halo-exchange
  "Declare a semantic neighbor exchange along one partitioned axis.

   `boundary` is `:nonperiodic` (only adjacent shards exchange) or `:periodic` (the last and first
   shards also exchange across the wrap edge). `combine` is nil for a copy into the target's ghost
   region, or a certified associative reduction when the source face accumulates into the target's
   owned face, as direct stiffness summation, deposition and force return require."
  [{:keys [id value axis width boundary combine attributes]
    :or {boundary :nonperiodic attributes {}}}]
  (when-not (and id value (integer? axis) (not (neg? axis))
                 (pos-int? width) (contains? halo-boundaries boundary) (map? attributes))
    (fail! "halo exchange requires a value, axis, positive width, and supported boundary policy"
           :distributed-halo-exchange
           {:id id :value value :axis axis :width width
            :boundary boundary :attributes attributes}))
  (when (and combine (not (scan/associative-scan? combine)))
    (fail! "accumulating halo exchange requires a certified associative reduction"
           :distributed-halo-combine {:id id :combine combine}))
  (->HaloExchange id value axis width boundary combine attributes))

(defn- halo-face
  "Global rectangle of a shard's owned face of `width` cells on `side` of `axis`."
  [candidate axis width side]
  (let [offsets (:offsets candidate)
        shape (:shape candidate)
        start (case side
                :lower (nth offsets axis)
                :upper (+ (nth offsets axis) (nth shape axis) (- width)))]
    {:offsets (assoc offsets axis start)
     :shape (assoc shape axis width)}))

(defn- halo-destination
  "Where a received face lands on the target.

   Copy mode writes the ghost strip beyond the target's owned extent, expressed in the target's
   local frame so a lower ghost has a negative offset. Combine mode accumulates into the target's
   owned face, expressed in the global frame."
  [target axis width side mode]
  (case mode
    :copy (let [rank (count (:shape target))
                offsets (vec (repeat rank 0))
                start (case side :lower (- width) :upper (nth (:shape target) axis))]
            {:frame :target-local
             :offsets (assoc offsets axis start)
             :shape (assoc (:shape target) axis width)})
    :combine (assoc (halo-face target axis width side) :frame :global)))

(defn- pack-halo-rounds
  "Greedily assign legs to the earliest round in which none of their directed links is claimed.

   Legs of one round overlap; a later round waits for the previous round's completions. The
   packing is deterministic in leg order, so a re-derived schedule reproduces the same rounds."
  [legs]
  (reduce
   (fn [rounds leg]
     (let [links (set (get-in leg [:leg :route]))
           index (or (some (fn [[i round]]
                             (when (empty? (set/intersection links (:links round))) i))
                           (map-indexed vector rounds))
                     (count rounds))
           round (get rounds index {:links #{} :legs []})]
       (assoc rounds index
              (-> round
                  (update :links set/union links)
                  (update :legs conj leg)))))
   []
   legs))

(defn schedule-halo
  "Derive a neighbor exchange from certified axis-partitioned shards.

   `routes` maps `[source-device target-device]` to an ordered directed-link path. Adjacent shard
   pairs exchange their faces in both directions; a periodic boundary adds the wrap edge between
   the last and first shards. Legs are packed into rounds so that no round claims one directed
   link twice; a later round depends on the completions of the previous one. Exact source
   rectangles, destination regions, byte counts and the round index are retained on each transfer
   step."
  [exchange abstract shards routes dependencies]
  (let [exchange (if (halo-exchange? exchange) exchange (halo-exchange exchange))
        {:keys [id value axis width boundary combine]} exchange
        mode (if combine :combine :copy)
        sharding (:sharding abstract)
        global-shape (:shape abstract)
        candidates (if (and (vector? global-shape) (< axis (count global-shape)))
                     (vec (sort-by #(nth (:offsets %) axis) shards))
                     [])]
    (when-not (and (abstract-value/abstract-value? abstract)
                   (= :partitioned (:kind sharding)) (= axis (:axis sharding))
                   (< axis (count global-shape)) (>= (count candidates) 2))
      (fail! "halo exchange requires a matching axis-partitioned AbstractValue"
             :distributed-halo-sharding
             {:halo id :axis axis :sharding sharding :shards candidates}))
    (when (and combine (not= (:dtype combine) (:dtype abstract)))
      (fail! "halo combine dtype differs from its AbstractValue"
             :distributed-halo-combine-dtype
             {:halo id :combine (:dtype combine) :value (:dtype abstract)}))
    (when-not (map? routes)
      (fail! "halo schedule routes must be keyed by device pairs"
             :distributed-halo-routes {:halo id :routes routes}))
    (doseq [candidate candidates]
      (when (< (nth (:shape candidate) axis) width)
        (fail! "halo width exceeds a local owned shard"
               :distributed-halo-width
               {:halo id :shard (:id candidate) :width width
                :local-extent (nth (:shape candidate) axis)})))
    (let [face-elements (* width (reduce * 1 (keep-indexed
                                              (fn [i extent] (when (not= i axis) extent))
                                              global-shape)))
          bytes (* face-elements (dtype/bytes-of (:dtype abstract)))
          adjacent (map-indexed (fn [i pair] [i false pair]) (partition 2 1 candidates))
          edges (cond-> (vec adjacent)
                  (= :periodic boundary)
                  (conj [(dec (count candidates)) true [(peek candidates) (first candidates)]]))
          legs
          (vec
           (mapcat
            (fn [[edge-index wrap? [left right]]]
              (mapv
               (fn [[direction source target side destination-side]]
                 (let [route (get routes [(:device source) (:device target)])]
                   (when-not route
                     (fail! "halo schedule lacks a route for an adjacent shard pair"
                            :distributed-halo-route
                            {:halo id :source (:device source) :target (:device target)}))
                   {:edge edge-index :direction direction :wrap wrap?
                    :leg (communication-leg
                          {:source (:device source) :target (:device target)
                           :route (vec route) :bytes bytes
                           :attributes {:source-shard (:id source)
                                        :target-shard (:id target)
                                        :source-region (halo-face source axis width side)
                                        :destination-side destination-side
                                        :destination-mode mode
                                        :destination-region
                                        (halo-destination target axis width
                                                          destination-side mode)}})}))
               [[:forward left right :upper :lower]
                [:backward right left :lower :upper]]))
            edges))
          rounds (pack-halo-rounds legs)
          {:keys [steps completions]}
          (reduce
           (fn [{:keys [steps completions]} [round-index round]]
             (let [waits (if (zero? round-index) (vec dependencies) completions)
                   round-steps
                   (mapv (fn [{:keys [edge direction wrap leg]}]
                           (transfer-step
                            {:id [id :edge edge direction]
                             :source (:source leg) :target (:target leg) :route (:route leg)
                             :value value :bytes (:bytes leg) :dependencies waits
                             :attributes (merge (:attributes leg)
                                                {:halo id :axis axis :width width
                                                 :boundary boundary
                                                 :edge edge :direction direction
                                                 :wrap wrap :round round-index})}))
                         (:legs round))]
               {:steps (into steps round-steps)
                :completions (mapv :id round-steps)}))
           {:steps [] :completions []}
           (map-indexed vector rounds))]
      (->ScheduledHalo exchange routes (vec dependencies) steps completions))))

(defn- concrete-shape!
  [value-id value]
  (let [shape (:shape value)]
    (when-not (and (seq shape) (every? #(and (integer? %) (pos? %)) shape))
      (fail! "distributed shard certification requires a positive concrete global shape"
             :distributed-value-shape {:value value-id :shape shape}))
    shape))

(defn- validate-shard-bounds!
  [value-id global-shape candidate]
  (when-not (value-shard? candidate)
    (fail! "distributed value mapping contains a non-ValueShard"
           :distributed-shard-type {:value value-id :shard candidate}))
  (when-not (= value-id (:value candidate))
    (fail! "value shard names a different global value"
           :distributed-shard-value
           {:value value-id :shard (:id candidate) :actual (:value candidate)}))
  (when-not (= (count global-shape) (count (:shape candidate)) (count (:offsets candidate)))
    (fail! "value shard rank differs from its global value"
           :distributed-shard-rank
           {:value value-id :global-shape global-shape :shard (:id candidate)
            :offsets (:offsets candidate) :shape (:shape candidate)}))
  (doseq [[axis global offset extent]
          (map vector (range) global-shape (:offsets candidate) (:shape candidate))]
    (when (> (+ offset extent) global)
      (fail! "value shard lies outside its global value"
             :distributed-shard-bounds
             {:value value-id :shard (:id candidate) :axis axis
              :global global :offset offset :extent extent}))))

(defn- validate-replicated-shards!
  [mesh value-id global-shape sharding candidates]
  (let [expected-devices (vec (or (:devices sharding) (:devices mesh)))
        actual-devices (mapv :device candidates)]
    (when-not (= (set expected-devices) (set actual-devices))
      (fail! "replicated value does not cover its declared device group"
             :distributed-replication-coverage
             {:value value-id :expected expected-devices :actual actual-devices}))
    (when-not (= (count expected-devices) (count actual-devices))
      (fail! "replicated value has duplicate device placements"
             :distributed-replication-duplicate
             {:value value-id :devices actual-devices}))
    (doseq [candidate candidates]
      (when-not (and (= (vec (repeat (count global-shape) 0)) (:offsets candidate))
                     (= global-shape (:shape candidate))
                     (= :replica (:ownership candidate)))
        (fail! "replicated shard must cover the full value and carry replica ownership"
               :distributed-replication-shape
               {:value value-id :shard candidate :global-shape global-shape})))))

(defn- validate-partitioned-shards!
  [value-id global-shape sharding candidates]
  (let [axis (:axis sharding)
        rank (count global-shape)]
    (when-not (and (integer? axis) (<= 0 axis) (< axis rank))
      (fail! "partitioned sharding requires an in-range integer axis"
             :distributed-partition-axis {:value value-id :axis axis :rank rank}))
    (when-let [expected-devices (:devices sharding)]
      (when-not (= (set expected-devices) (set (map :device candidates)))
        (fail! "partitioned value does not use its declared device group"
               :distributed-partition-devices
               {:value value-id :expected expected-devices
                :actual (mapv :device candidates)})))
    (doseq [candidate candidates
            other-axis (range rank)
            :when (not= axis other-axis)]
      (when-not (and (zero? (nth (:offsets candidate) other-axis))
                     (= (nth global-shape other-axis)
                        (nth (:shape candidate) other-axis)))
        (fail! "an axis-partitioned shard must cover every unsharded dimension"
               :distributed-partition-orthogonal-shape
               {:value value-id :axis axis :shard candidate
                :global-shape global-shape})))
    (when-not (every? #(= :owned (:ownership %)) candidates)
      (fail! "partitioned shards must carry owned, non-replicated storage"
             :distributed-partition-ownership {:value value-id :shards candidates}))
    (let [intervals (sort-by first
                             (map (fn [candidate]
                                    [(nth (:offsets candidate) axis)
                                     (+ (nth (:offsets candidate) axis)
                                        (nth (:shape candidate) axis))
                                     (:id candidate)])
                                  candidates))]
      (loop [cursor 0 remaining intervals]
        (if-let [[start end shard-id] (first remaining)]
          (if (= cursor start)
            (recur end (next remaining))
            (fail! "partitioned shards contain a gap or overlap"
                   :distributed-partition-coverage
                   {:value value-id :axis axis :expected-start cursor
                    :actual-start start :shard shard-id :intervals intervals}))
          (when-not (= cursor (nth global-shape axis))
            (fail! "partitioned shards do not cover their complete global dimension"
                   :distributed-partition-coverage
                   {:value value-id :axis axis :covered cursor
                    :global (nth global-shape axis) :intervals intervals})))))))

(defn- validate-value-shards!
  [mesh topology values shards]
  (when-not (= (set (keys values)) (set (keys shards)))
    (fail! "distributed values and shard mappings must have identical identities"
           :distributed-shard-values
           {:values (set (keys values)) :shards (set (keys shards))}))
  (doseq [[value-id value] values]
    (abstract-value/validate! value)
    (let [global-shape (concrete-shape! value-id value)
          candidates (vec (get shards value-id))
          sharding (:sharding value)]
      (when-not (seq candidates)
        (fail! "distributed value requires at least one physical shard"
               :distributed-shard-empty {:value value-id}))
      (unique-by! "value shards" :distributed-shard-identities :id candidates)
      (unique-by! "value shard devices" :distributed-shard-device-identities :device candidates)
      (doseq [candidate candidates]
        (validate-shard-bounds! value-id global-shape candidate)
        (when-not (contains? (set (:devices mesh)) (:device candidate))
          (fail! "value shard is placed outside the device mesh"
                 :distributed-shard-device
                 {:value value-id :shard (:id candidate) :device (:device candidate)}))
        (when-not (contains? (set (keys (:devices topology))) (:device candidate))
          (fail! "value shard is placed outside the cluster topology"
                 :distributed-shard-topology-device
                 {:value value-id :shard (:id candidate) :device (:device candidate)})))
      (when-not (and (map? sharding) (contains? shard-kinds (:kind sharding)))
        (fail! "distributed value requires an explicit supported sharding facet"
               :distributed-value-sharding
               {:value value-id :sharding sharding :allowed shard-kinds}))
      (case (:kind sharding)
        :replicated (validate-replicated-shards! mesh value-id global-shape sharding candidates)
        :partitioned (validate-partitioned-shards! value-id global-shape sharding candidates)))))

(defn- route-links!
  [topology {:keys [id source target route]}]
  (when-not (and (seq route) (vector? route))
    (fail! "cross-device transfer requires a non-empty ordered route"
           :distributed-transfer-route {:step id :source source :target target :route route}))
  (loop [current source remaining route resolved []]
    (if-let [link-id (first remaining)]
      (let [candidate (get (:links topology) link-id)]
        (when-not candidate
          (fail! "transfer route names an undeclared topology link"
                 :distributed-transfer-link {:step id :link link-id}))
        (when-not (= current (:source candidate))
          (fail! "transfer route is not directionally contiguous"
                 :distributed-transfer-continuity
                 {:step id :link link-id :expected-source current
                  :actual-source (:source candidate)}))
        (recur (:target candidate) (next remaining) (conj resolved candidate)))
      (do
        (when-not (= current target)
          (fail! "transfer route does not terminate at its target device"
                 :distributed-transfer-target
                 {:step id :expected target :actual current :route route}))
        resolved))))

(defn transfer-duration-ns
  "Topology cost for one transfer: path latency plus bytes at the route bottleneck bandwidth.

   This cut-through model is an optimistic analytic seed. Measured route costs can later replace
   it without changing the plan or its byte accounting."
  [topology transfer]
  (let [links (route-links! topology transfer)
        latency (reduce + 0.0 (map :latency-ns links))
        bottleneck (reduce min (map :bandwidth-bytes-s links))
        serialization (* (/ (double (:bytes transfer)) bottleneck) 1.0e9)]
    (long (Math/ceil (+ latency serialization)))))

(defn- validate-collectives!
  [mesh topology values groups collectives steps]
  (when-not (and (map? groups) (every? collective-group? (vals groups))
                 (every? (fn [[id group]] (= id (:id group))) groups))
    (fail! "collective groups must map their identities to CollectiveGroup values"
           :distributed-collective-groups {:groups groups}))
  (doseq [[group-id group] groups]
    (when-not (set/subset? (set (:devices group)) (set (:devices mesh)))
      (fail! "collective group contains a device outside the mesh"
             :distributed-collective-group-device
             {:group group-id :devices (:devices group) :mesh (:devices mesh)})))
  (when-not (and (vector? collectives) (every? scheduled-collective? collectives))
    (fail! "distributed collectives must be an ordered vector of ScheduledCollective values"
           :distributed-collectives {:collectives collectives}))
  (unique-by! "distributed collectives" :distributed-collective-identities
              (comp :id :operation) collectives)
  (let [step-by-id (into {} (map (juxt :id identity)) steps)]
    (doseq [{:keys [operation schedule dependencies steps completions] :as scheduled} collectives]
      (let [{:keys [id kind group value reduction root]} operation
            group-value (get groups group)
            abstract (get values value)
            devices (set (:devices group-value))
            sharding (:sharding abstract)
            sharding-devices (set (or (:devices sharding) (:devices mesh)))
            expected (schedule-collective operation schedule dependencies)]
        (when-not group-value
          (fail! "collective names an undeclared group"
                 :distributed-collective-group-reference {:collective id :group group}))
        (when-not abstract
          (fail! "collective names an undeclared AbstractValue"
                 :distributed-collective-value {:collective id :value value}))
        (when-not (= devices sharding-devices)
          (fail! "collective group and value sharding devices disagree"
                 :distributed-collective-sharding-group
                 {:collective id :group devices :sharding sharding-devices}))
        (if (contains? #{:all-reduce :all-gather :broadcast} kind)
          (when-not (= :replicated (:kind sharding))
            (fail! "collective output requires replicated sharding"
                   :distributed-collective-output-sharding
                   {:collective id :kind kind :sharding sharding}))

          (when-not (= :partitioned (:kind sharding))
            (fail! "reduce-scatter output requires partitioned sharding"
                   :distributed-collective-output-sharding
                   {:collective id :kind kind :sharding sharding})))
        (when (and reduction (not= (:dtype reduction) (:dtype abstract)))
          (fail! "collective reduction dtype differs from its AbstractValue"
                 :distributed-collective-dtype
                 {:collective id :reduction (:dtype reduction) :value (:dtype abstract)}))
        (when (and (= :broadcast kind) (not (contains? devices root)))
          (fail! "broadcast root must belong to its collective group"
                 :distributed-collective-root {:collective id :root root :devices devices}))
        (let [participants (set (mapcat (juxt :source :target)
                                        (mapcat identity (:rounds schedule))))]
          (when-not (= devices participants)
            (fail! "collective schedule does not involve every group member"
                   :distributed-collective-participation
                   {:collective id :expected devices :actual participants})))
        (doseq [leg (mapcat identity (:rounds schedule))]
          (when-not (and (contains? devices (:source leg))
                         (contains? devices (:target leg)))
            (fail! "collective communication leg leaves its group"
                   :distributed-collective-leg-group
                   {:collective id :leg leg :devices devices}))
          (transfer-duration-ns topology
                                (transfer-step {:id [:collective-validation id]
                                                :source (:source leg) :target (:target leg)
                                                :route (:route leg) :value value
                                                :bytes (:bytes leg)})))
        (when-not (= expected scheduled)
          (fail! "scheduled collective does not match its semantic operation and rounds"
                 :distributed-collective-lowering
                 {:collective id :expected expected :actual scheduled}))
        (doseq [step steps]
          (when-not (= step (get step-by-id (:id step)))
            (fail! "collective expansion step is absent or differs from the distributed DAG"
                   :distributed-collective-step
                   {:collective id :step (:id step)})))
        (when-not (= completions (:completions expected))
          (fail! "collective completions differ from its final communication round"
                 :distributed-collective-completions
                 {:collective id :completions completions})))))
  collectives)

(defn- validate-halos!
  [topology values shards halos steps]
  (when-not (and (vector? halos) (every? scheduled-halo? halos))
    (fail! "distributed halos must be an ordered vector of ScheduledHalo values"
           :distributed-halos {:halos halos}))
  (unique-by! "distributed halos" :distributed-halo-identities
              (comp :id :exchange) halos)
  (let [step-by-id (into {} (map (juxt :id identity)) steps)]
    (doseq [{:keys [exchange routes dependencies steps] :as scheduled} halos]
      (let [value-id (:value exchange)
            abstract (get values value-id)
            candidates (get shards value-id)
            expected (schedule-halo exchange abstract candidates routes dependencies)]
        (when-not (= expected scheduled)
          (fail! "scheduled halo does not match its value shards and routes"
                 :distributed-halo-lowering
                 {:halo (:id exchange) :expected expected :actual scheduled}))
        (doseq [step steps]
          (transfer-duration-ns topology step)
          (when-not (= step (get step-by-id (:id step)))
            (fail! "halo expansion step is absent or differs from the distributed DAG"
                   :distributed-halo-step
                   {:halo (:id exchange) :step (:id step)})))))
    halos))

(defn- validate-device-plans!
  [mesh device-plans]
  (when-not (map? device-plans)
    (fail! "distributed device plans must be a map keyed by device identity"
           :distributed-device-plans {:device-plans device-plans}))
  (when-not (set/subset? (set (keys device-plans)) (set (:devices mesh)))
    (fail! "a shard-local plan is assigned outside the device mesh"
           :distributed-device-plan-device
           {:devices (set (keys device-plans)) :mesh (set (:devices mesh))}))
  (doseq [[device-id {:keys [link-plan execution-plan] :as local}]
          device-plans]
    (when-not (map? local)
      (fail! "a shard-local plan must be a map"
             :distributed-device-plan {:device device-id :plan local}))
    (when link-plan
      (link-plan/validate! link-plan)
      (when-not (= device-id (:target link-plan))
        (fail! "shard-local LinkPlan target differs from its mesh device"
               :distributed-device-link-target
               {:device device-id :target (:target link-plan)})))
    (when execution-plan
      (execution-plan/validate! execution-plan))))

(defn- validate-steps!
  [mesh topology values steps outputs]
  (when-not (and (vector? steps) (every? distributed-step? steps))
    (fail! "distributed steps must be an ordered vector of DistributedStep values"
           :distributed-steps {:steps steps}))
  (unique-by! "distributed steps" :distributed-step-identities :id steps)
  (let [mesh-devices (set (:devices mesh))
        topology-devices (set (keys (:devices topology)))]
    (loop [remaining steps available #{}]
      (if-let [step (first remaining)]
        (do
          (when (or (nil? (:id step)) (not (contains? step-kinds (:kind step))))
            (fail! "distributed step requires an identity and supported kind"
                   :distributed-step-kind {:step step :allowed step-kinds}))
          (when-not (and (vector? (:dependencies step))
                         (= (count (:dependencies step))
                            (count (distinct (:dependencies step))))
                         (set/subset? (set (:dependencies step)) available))
            (fail! "distributed step dependencies must be unique earlier step identities"
                   :distributed-step-dependencies
                   {:step (:id step) :dependencies (:dependencies step)
                    :available available}))
          (when-not (map? (:attributes step))
            (fail! "distributed step attributes must be a map"
                   :distributed-step-attributes {:step (:id step)}))
          (case (:kind step)
            :compute
            (do
              (when-not (and (contains? mesh-devices (:device step))
                             (contains? topology-devices (:device step)))
                (fail! "compute step device must belong to the mesh and topology"
                       :distributed-compute-device
                       {:step (:id step) :device (:device step)}))
              (when-not (positive-number? (:duration-ns step))
                (fail! "compute step duration must be positive and finite"
                       :distributed-compute-duration
                       {:step (:id step) :duration-ns (:duration-ns step)}))
              (when-not (and (integer? (:peak-memory-bytes step))
                             (not (neg? (:peak-memory-bytes step))))
                (fail! "compute step peak memory must be a non-negative integer"
                       :distributed-compute-memory
                       {:step (:id step) :peak-memory-bytes (:peak-memory-bytes step)}))
              (let [capacity (get-in topology [:devices (:device step) :memory-capacity-bytes])]
                (when (> (:peak-memory-bytes step) capacity)
                  (fail! "compute step exceeds device memory capacity"
                         :distributed-device-memory-capacity
                         {:step (:id step) :device (:device step)
                          :required (:peak-memory-bytes step) :capacity capacity}))))

            :transfer
            (do
              (when-not (and (contains? mesh-devices (:source step))
                             (contains? mesh-devices (:target step))
                             (not= (:source step) (:target step)))
                (fail! "transfer endpoints must be distinct devices in the mesh"
                       :distributed-transfer-endpoints
                       {:step (:id step) :source (:source step) :target (:target step)}))
              (when-not (contains? values (:value step))
                (fail! "transfer step names an undeclared AbstractValue"
                       :distributed-transfer-value
                       {:step (:id step) :value (:value step)}))
              (when-not (and (integer? (:bytes step)) (not (neg? (:bytes step))))
                (fail! "transfer byte count must be a non-negative integer"
                       :distributed-transfer-bytes
                       {:step (:id step) :bytes (:bytes step)}))
              (route-links! topology step)))
          (recur (next remaining) (conj available (:id step))))
        (do
          (when-not (and (vector? outputs) (seq outputs)
                         (= (count outputs) (count (distinct outputs)))
                         (set/subset? (set outputs) available))
            (fail! "distributed outputs must be unique completed step identities"
                   :distributed-outputs {:outputs outputs :available available})))))))

(declare simulate)

(defn validate!
  "Validate and return a DistributedPlan without realizing any runtime resource."
  [plan]
  (when-not (distributed-plan? plan)
    (fail! "expected a DistributedPlan value"
           :distributed-plan-type {:actual (type plan)}))
  (let [{:keys [id mesh topology values shards collective-groups collectives halos
                device-plans steps outputs attributes]} plan]
    (when (nil? id)
      (fail! "distributed plan requires a stable identity" :distributed-plan-id {}))
    (when-not (device-mesh? mesh)
      (fail! "distributed plan requires a DeviceMesh"
             :distributed-plan-mesh {:mesh mesh}))
    ;; Reconstructing re-runs all constructor invariants even for a modified record.
    (raster.compiler.ir.distributed-plan/mesh (:axes mesh) (:devices mesh))
    (when-not (cluster-topology? topology)
      (fail! "distributed plan requires a ClusterTopology"
             :distributed-plan-topology {:topology topology}))
    (raster.compiler.ir.distributed-plan/topology
     (vals (:devices topology)) (vals (:links topology)))
    (when-not (set/subset? (set (:devices mesh)) (set (keys (:devices topology))))
      (fail! "device mesh is not contained in the cluster topology"
             :distributed-plan-mesh-topology
             {:mesh (set (:devices mesh)) :topology (set (keys (:devices topology)))}))
    (when-not (and (map? values) (every? abstract-value/abstract-value? (vals values)))
      (fail! "distributed values must map identities to AbstractValue records"
             :distributed-values {:values values}))
    (when-not (map? shards)
      (fail! "distributed shard mappings must be a map"
             :distributed-shards {:shards shards}))
    (validate-value-shards! mesh topology values shards)
    (validate-collectives! mesh topology values collective-groups collectives steps)
    (validate-halos! topology values shards halos steps)
    (validate-device-plans! mesh device-plans)
    (validate-steps! mesh topology values steps outputs)
    (when-not (map? attributes)
      (fail! "distributed plan attributes must be a map"
             :distributed-plan-attributes {:attributes attributes})))
  plan)

(defn plan
  [{:keys [id mesh topology values shards collective-groups collectives
           halos device-plans steps outputs attributes]
    :or {values {} shards {} collective-groups {} collectives []
         halos [] device-plans {} steps [] outputs [] attributes {}}}]
  (validate!
   (->DistributedPlan id mesh topology values shards collective-groups collectives (vec halos)
                      device-plans (vec steps) (vec outputs) attributes)))

(defn simulate
  "Simulate an explicit DistributedPlan schedule.

   Compute steps serialize on their device compute lane. Transfers serialize on every directed
   link in their route. The two resource classes are independent, so communication and compute
   overlap whenever dependencies permit. The result is deterministic and suitable as an analytic
   seed/pruner; measured costs should replace durations before production selection."
  [plan]
  (let [plan (validate! plan)
        topology (:topology plan)
        initial {:finish-by-step {}
                 :resource-free {}
                 :timeline {}
                 :device-compute-ns {}
                 :link-transfer-bytes {}
                 :link-busy-ns {}
                 :peak-memory-by-device {}}
        state
        (reduce
         (fn [state step]
           (let [dependency-ready (reduce max 0 (map #(get-in state [:finish-by-step %])
                                                     (:dependencies step)))
                 resources (case (:kind step)
                             :compute [[:compute (:device step)]]
                             :transfer (mapv (fn [link-id] [:link link-id]) (:route step)))
                 resource-ready (reduce max 0 (map #(get-in state [:resource-free %] 0)
                                                   resources))
                 start (max dependency-ready resource-ready)
                 duration (case (:kind step)
                            :compute (long (Math/ceil (double (:duration-ns step))))
                            :transfer (transfer-duration-ns topology step))
                 finish (+ start duration)
                 state (-> state
                           (assoc-in [:finish-by-step (:id step)] finish)
                           (assoc-in [:timeline (:id step)]
                                     {:kind (:kind step) :start-ns start
                                      :duration-ns duration :finish-ns finish
                                      :resources resources}))]
             (case (:kind step)
               :compute
               (-> state
                   (update-in [:device-compute-ns (:device step)] (fnil + 0) duration)
                   (update-in [:peak-memory-by-device (:device step)]
                              (fnil max 0) (:peak-memory-bytes step))
                   (assoc-in [:resource-free [:compute (:device step)]] finish))

               :transfer
               (reduce (fn [state link-id]
                         (-> state
                             (assoc-in [:resource-free [:link link-id]] finish)
                             (update-in [:link-transfer-bytes link-id]
                                        (fnil + 0) (:bytes step))
                             (update-in [:link-busy-ns link-id] (fnil + 0) duration)))
                       state (:route step)))))
         initial (:steps plan))
        makespan (reduce max 0 (vals (:finish-by-step state)))
        transferred-bytes (reduce + 0 (map :bytes (filter #(= :transfer (:kind %))
                                                          (:steps plan))))
        capacities (into {} (map (fn [[id resource]] [id (:memory-capacity-bytes resource)]))
                         (get-in plan [:topology :devices]))
        peak-memory (:peak-memory-by-device state)]
    {:plan-id (:id plan)
     :makespan-ns makespan
     :timeline (:timeline state)
     :device-compute-ns (:device-compute-ns state)
     :link-transfer-bytes (:link-transfer-bytes state)
     :link-busy-ns (:link-busy-ns state)
     :peak-memory-by-device peak-memory
     :memory-headroom-by-device
     (into {} (map (fn [[device-id capacity]]
                     [device-id (- capacity (get peak-memory device-id 0))])) capacities)
     :transferred-bytes transferred-bytes
     :cost-vector {:latency-ns makespan
                   :peak-memory-bytes (reduce + 0 (vals peak-memory))
                   :peak-memory-by-device peak-memory
                   :transferred-bytes transferred-bytes}}))

(defn- shard-coverage
  [plan]
  (into {}
        (map (fn [[value-id candidates]]
               [value-id
                (mapv #(select-keys % [:id :device :offsets :shape :ownership]) candidates)]))
        (:shards plan)))

(defn- route-costs
  [plan]
  (into {}
        (keep (fn [step]
                (when (= :transfer (:kind step))
                  [(:id step) {:route (:route step) :bytes (:bytes step)
                               :duration-ns (transfer-duration-ns (:topology plan) step)}])))
        (:steps plan)))

(defn- derive-certificate
  [plan]
  (let [simulation (simulate plan)]
    (->DistributedPlanCertificate
     (:id plan)
     (mapv (juxt :name :size) (get-in plan [:mesh :axes]))
     (shard-coverage plan)
     (mapv (fn [{:keys [operation schedule steps completions]}]
             {:id (:id operation) :kind (:kind operation) :group (:group operation)
              :value (:value operation) :algorithm (:algorithm schedule)
              :steps (mapv :id steps) :completions completions})
           (:collectives plan))
     (mapv (fn [{:keys [exchange steps completions]}]
             {:id (:id exchange) :value (:value exchange) :axis (:axis exchange)
              :width (:width exchange) :boundary (:boundary exchange)
              :combine (some-> (:combine exchange)
                               (select-keys [:combine :identity :dtype]))
              :rounds (inc (reduce max 0 (map #(get-in % [:attributes :round]) steps)))
              :steps (mapv :id steps) :completions completions
              :bytes (reduce + 0 (map :bytes steps))})
           (:halos plan))
     (route-costs plan)
     (:cost-vector simulation)
     (into {}
           (map (fn [[device-id local]]
                  [device-id
                   {:link-plan (some-> (:link-plan local) :id)
                    :execution-operations (some-> (:execution-plan local) :operations count)}]))
           (:device-plans plan)))))

(defn certify
  "Validate a plan and attach a reproducible coverage, route-cost, and resource witness."
  [plan]
  (let [plan (validate! plan)]
    (->CertifiedDistributedPlan plan (derive-certificate plan))))

(defn verify!
  "Revalidate a CertifiedDistributedPlan and independently derive its certificate."
  [certified]
  (when-not (certified-plan? certified)
    (fail! "expected a CertifiedDistributedPlan"
           :distributed-certified-plan-type {:actual (type certified)}))
  (let [plan (validate! (:plan certified))
        certificate (:certificate certified)
        expected (derive-certificate plan)]
    (when-not (certificate? certificate)
      (fail! "distributed plan certificate has the wrong type"
             :distributed-certificate-type {:actual (type certificate)}))
    (when-not (= expected certificate)
      (fail! "distributed plan certificate does not match its plan"
             :distributed-certificate {:expected expected :actual certificate}))
    certified))
