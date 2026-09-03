(ns raster.compiler.ir.distributed-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.scan :as scan]))

(defn- two-device-topology
  []
  (distributed/topology
   [(distributed/device {:id :gpu-0 :memory-capacity-bytes 16000000000})
    (distributed/device {:id :gpu-1 :memory-capacity-bytes 16000000000})]
   [(distributed/link {:id :gpu-0->gpu-1 :source :gpu-0 :target :gpu-1
                       :kind :pcie :bandwidth-bytes-s 25.0e9 :latency-ns 1000})
    (distributed/link {:id :gpu-1->gpu-0 :source :gpu-1 :target :gpu-0
                       :kind :pcie :bandwidth-bytes-s 25.0e9 :latency-ns 1000})]))

(defn- training-values
  []
  {:batch (abstract-value/tensor
           {:dtype :float :shape [8 4]
            :sharding {:kind :partitioned :axis 0 :devices [:gpu-0 :gpu-1]}
            :ownership :owned})
   :weights (abstract-value/tensor
             {:dtype :float :shape [4 4]
              :sharding {:kind :replicated :devices [:gpu-0 :gpu-1]}
              :ownership :owned})})

(defn- training-shards
  []
  {:batch [(distributed/shard {:id :batch-0 :value :batch :device :gpu-0
                               :offsets [0 0] :shape [4 4]})
           (distributed/shard {:id :batch-1 :value :batch :device :gpu-1
                               :offsets [4 0] :shape [4 4]})]
   :weights [(distributed/shard {:id :weights-0 :value :weights :device :gpu-0
                                 :offsets [0 0] :shape [4 4] :ownership :replica})
             (distributed/shard {:id :weights-1 :value :weights :device :gpu-1
                                 :offsets [0 0] :shape [4 4] :ownership :replica})]})

(defn- training-plan
  []
  (distributed/plan
   {:id :data-parallel-probe
    :mesh (distributed/mesh [{:name :data :size 2}] [:gpu-0 :gpu-1])
    :topology (two-device-topology)
    :values (training-values)
    :shards (training-shards)
    :steps [(distributed/compute-step
             {:id :gradient-0 :device :gpu-0 :duration-ns 100
              :peak-memory-bytes 1000000})
            (distributed/compute-step
             {:id :independent-gradient-1 :device :gpu-1 :duration-ns 500
              :peak-memory-bytes 2000000})
            (distributed/transfer-step
             {:id :send-gradient :source :gpu-0 :target :gpu-1
              :route [:gpu-0->gpu-1] :value :weights :bytes 1000000
              :dependencies [:gradient-0]})
            (distributed/compute-step
             {:id :apply-gradient :device :gpu-1 :duration-ns 200
              :peak-memory-bytes 3000000
              :dependencies [:independent-gradient-1 :send-gradient]})]
    :outputs [:apply-gradient]}))

(deftest topology-aware-simulation-overlaps-compute-and-transfer-resources
  (let [simulation (distributed/simulate (training-plan))]
    ;; transfer = 1us latency + 1MB / 25GB/s = 41us. It starts at t=100 after
    ;; gradient-0, while the independent gpu-1 compute occupies [0,500].
    (is (= 41300 (:makespan-ns simulation)))
    (is (= {:kind :transfer :start-ns 100 :duration-ns 41000 :finish-ns 41100
            :resources [[:link :gpu-0->gpu-1]]}
           (get-in simulation [:timeline :send-gradient])))
    (is (= 41100 (get-in simulation [:timeline :apply-gradient :start-ns])))
    (is (= {:gpu-0 100 :gpu-1 700} (:device-compute-ns simulation)))
    (is (= {:gpu-0->gpu-1 1000000} (:link-transfer-bytes simulation)))
    (is (= 1000000 (:transferred-bytes simulation)))
    (is (= 3000000 (get-in simulation [:peak-memory-by-device :gpu-1])))
    (is (= {:latency-ns 41300
            :peak-memory-bytes 4000000
            :peak-memory-by-device {:gpu-0 1000000 :gpu-1 3000000}
            :transferred-bytes 1000000}
           (:cost-vector simulation)))))

(deftest sharding-certifies-partition-coverage-and-replication
  (let [certified (distributed/certify (training-plan))
        certificate (:certificate certified)]
    (is (distributed/certified-plan? (distributed/verify! certified)))
    (is (= [[:data 2]] (:mesh-shape certificate)))
    (is (= [:batch-0 :batch-1]
           (mapv :id (get (:shard-coverage certificate) :batch))))
    (is (= 41000 (get-in certificate [:route-costs :send-gradient :duration-ns])))
    (is (= (:cost-vector (distributed/simulate (training-plan)))
           (:cost-vector certificate)))))

(deftest partition-gaps-and-overlaps-fail-before-scheduling
  (doseq [[label bad-offset expected-start]
          [["gap" 5 4]
           ["overlap" 3 4]]]
    (testing label
      (let [shards (assoc-in (training-shards) [:batch 1]
                             (distributed/shard
                              {:id :batch-1 :value :batch :device :gpu-1
                               :offsets [bad-offset 0] :shape [3 4]}))
            error (try
                    (distributed/plan
                     {:id :bad-shards
                      :mesh (distributed/mesh [{:name :data :size 2}] [:gpu-0 :gpu-1])
                      :topology (two-device-topology)
                      :values (training-values) :shards shards})
                    (catch clojure.lang.ExceptionInfo exception exception))]
        (is (= :distributed-partition-coverage (:reason (ex-data error))))
        (is (= expected-start (:expected-start (ex-data error))))))))

(deftest routes-are-directed-and-contiguous
  (let [base {:id :bad-route
              :mesh (distributed/mesh [{:name :data :size 2}] [:gpu-0 :gpu-1])
              :topology (two-device-topology)
              :values (training-values) :shards (training-shards)}
        error (try
                (distributed/plan
                 (assoc base
                        :steps [(distributed/transfer-step
                                 {:id :wrong-way :source :gpu-0 :target :gpu-1
                                  :route [:gpu-1->gpu-0] :value :weights :bytes 16})]
                        :outputs [:wrong-way]))
                (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-transfer-continuity (:reason (ex-data error))))))

(deftest memory-capacity-is-a-legality-gate-not-a-cost-penalty
  (let [error
        (try
          (distributed/plan
           {:id :oversubscribed
            :mesh (distributed/mesh [{:name :data :size 2}] [:gpu-0 :gpu-1])
            :topology (two-device-topology)
            :values (training-values) :shards (training-shards)
            :steps [(distributed/compute-step
                     {:id :too-large :device :gpu-0 :duration-ns 10
                      :peak-memory-bytes 16000000001})]
            :outputs [:too-large]})
          (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-device-memory-capacity (:reason (ex-data error))))
    (is (= 16000000000 (:capacity (ex-data error))))))

(deftest a-certificate-detects-plan-drift
  (let [certified (distributed/certify (training-plan))
        modified (assoc-in certified [:plan :steps 0 :duration-ns] 101)
        error (try (distributed/verify! modified)
                   (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-certificate (:reason (ex-data error))))))

(defn- two-device-all-reduce
  []
  (let [operation
        (distributed/collective-operation
         {:id :gradient-all-reduce :kind :all-reduce :group :data
          :value :weights
          :reduction (scan/certify {:acc 'acc :init 0.0 :lambda '(+ acc element)} :float)})
        schedule
        (distributed/collective-schedule
         {:algorithm :direct-exchange
          :rounds [[{:source :gpu-0 :target :gpu-1 :route [:gpu-0->gpu-1] :bytes 64}
                    {:source :gpu-1 :target :gpu-0 :route [:gpu-1->gpu-0] :bytes 64}]]})]
    (distributed/schedule-collective operation schedule [:gradient-0 :gradient-1])))

(deftest semantic-all-reduce-retains-its-topology-schedule
  (let [collective (two-device-all-reduce)
        compute [(distributed/compute-step
                  {:id :gradient-0 :device :gpu-0 :duration-ns 100})
                 (distributed/compute-step
                  {:id :gradient-1 :device :gpu-1 :duration-ns 150})]
        plan (distributed/plan
              {:id :all-reduce-probe
               :mesh (distributed/mesh [{:name :data :size 2}] [:gpu-0 :gpu-1])
               :topology (two-device-topology)
               :values (training-values) :shards (training-shards)
               :collective-groups {:data (distributed/collective-group
                                          :data [:gpu-0 :gpu-1])}
               :collectives [collective]
               :steps (into compute (:steps collective))
               :outputs (:completions collective)})
        simulation (distributed/simulate plan)
        certificate (:certificate (distributed/certify plan))]
    (is (= 2 (count (:completions collective))))
    (is (= 1153 (:makespan-ns simulation)))
    (is (= 128 (:transferred-bytes simulation)))
    (is (= {:id :gradient-all-reduce :kind :all-reduce :group :data
            :value :weights :algorithm :direct-exchange
            :steps [[:gradient-all-reduce :round 0 :leg 0]
                    [:gradient-all-reduce :round 0 :leg 1]]
            :completions [[:gradient-all-reduce :round 0 :leg 0]
                          [:gradient-all-reduce :round 0 :leg 1]]}
           (first (:collectives certificate))))))

(deftest reducing-collectives-require-an-associativity-certificate
  (let [error (try
                (distributed/collective-operation
                 {:id :unchecked :kind :all-reduce :group :data :value :weights})
                (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-collective-reduction (:reason (ex-data error))))))

(deftest parallel-collective-legs-cannot-oversubscribe-one-directed-link
  (let [error (try
                (distributed/collective-schedule
                 {:algorithm :invalid
                  :rounds [[{:source :gpu-0 :target :gpu-1
                             :route [:gpu-0->gpu-1] :bytes 64}
                            {:source :gpu-0 :target :gpu-1
                             :route [:gpu-0->gpu-1] :bytes 64}]]})
                (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-collective-round-link-conflict (:reason (ex-data error))))))

(defn- two-device-halo
  [routes]
  (distributed/schedule-halo
   (distributed/halo-exchange
    {:id :batch-halo :value :batch :axis 0 :width 1 :boundary :nonperiodic})
   (:batch (training-values)) (:batch (training-shards)) routes
   [:stencil-0 :stencil-1]))

(deftest halo-exchange-derives-neighbor-regions-and-byte-costs
  (let [halo (two-device-halo
              {[:gpu-0 :gpu-1] [:gpu-0->gpu-1]
               [:gpu-1 :gpu-0] [:gpu-1->gpu-0]})
        compute [(distributed/compute-step
                  {:id :stencil-0 :device :gpu-0 :duration-ns 100})
                 (distributed/compute-step
                  {:id :stencil-1 :device :gpu-1 :duration-ns 150})]
        plan (distributed/plan
              {:id :halo-probe
               :mesh (distributed/mesh [{:name :space :size 2}] [:gpu-0 :gpu-1])
               :topology (two-device-topology)
               :values (training-values) :shards (training-shards)
               :halos [halo]
               :steps (into compute (:steps halo))
               :outputs (:completions halo)})
        simulation (distributed/simulate plan)
        certificate (:certificate (distributed/certify plan))
        [forward backward] (:steps halo)]
    (is (= [16 16] (mapv :bytes (:steps halo))))
    (is (= {:offsets [3 0] :shape [1 4]}
           (get-in forward [:attributes :source-region])))
    (is (= {:offsets [4 0] :shape [1 4]}
           (get-in backward [:attributes :source-region])))
    (is (= 1151 (:makespan-ns simulation)))
    (is (= 32 (:transferred-bytes simulation)))
    (is (= {:id :batch-halo :value :batch :axis 0 :width 1
            :boundary :nonperiodic :combine nil :rounds 1
            :steps [[:batch-halo :edge 0 :forward]
                    [:batch-halo :edge 0 :backward]]
            :completions [[:batch-halo :edge 0 :forward]
                          [:batch-halo :edge 0 :backward]]
            :bytes 32}
           (first (:halos certificate))))))

(deftest halo-scheduling-requires-every-directed-neighbor-route
  (let [error (try
                (two-device-halo {[:gpu-0 :gpu-1] [:gpu-0->gpu-1]})
                (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-halo-route (:reason (ex-data error))))
    (is (= :gpu-1 (:source (ex-data error))))))

(deftest halo-steps-retain-destination-regions-in-both-frames
  (let [halo (two-device-halo
              {[:gpu-0 :gpu-1] [:gpu-0->gpu-1]
               [:gpu-1 :gpu-0] [:gpu-1->gpu-0]})
        [forward backward] (:steps halo)]
    ;; gpu-1 receives gpu-0's upper face into its lower ghost strip, one row before its origin.
    (is (= {:frame :target-local :offsets [-1 0] :shape [1 4]}
           (get-in forward [:attributes :destination-region])))
    (is (= :copy (get-in forward [:attributes :destination-mode])))
    (is (= {:frame :target-local :offsets [4 0] :shape [1 4]}
           (get-in backward [:attributes :destination-region])))
    (is (= [0 0] (mapv #(get-in % [:attributes :round]) (:steps halo))))
    (is (false? (get-in forward [:attributes :wrap])))))

(defn- three-device-topology
  []
  (distributed/topology
   [(distributed/device {:id :gpu-0 :memory-capacity-bytes 16000000000})
    (distributed/device {:id :gpu-1 :memory-capacity-bytes 16000000000})
    (distributed/device {:id :gpu-2 :memory-capacity-bytes 16000000000})]
   (for [[source target] [[:gpu-0 :gpu-1] [:gpu-1 :gpu-0] [:gpu-1 :gpu-2] [:gpu-2 :gpu-1]
                          [:gpu-2 :gpu-0] [:gpu-0 :gpu-2]]]
     (distributed/link {:id (keyword (str (name source) "->" (name target)))
                        :source source :target target
                        :kind :pcie :bandwidth-bytes-s 25.0e9 :latency-ns 1000}))))

(defn- ring-routes
  []
  (into {} (for [[source target] [[:gpu-0 :gpu-1] [:gpu-1 :gpu-0] [:gpu-1 :gpu-2] [:gpu-2 :gpu-1]
                                  [:gpu-2 :gpu-0] [:gpu-0 :gpu-2]]]
             [[source target] [(keyword (str (name source) "->" (name target)))]])))

(defn- three-shard-field
  []
  {:value (abstract-value/tensor
           {:dtype :float :shape [12 4]
            :sharding {:kind :partitioned :axis 0 :devices [:gpu-0 :gpu-1 :gpu-2]}
            :ownership :owned})
   :shards [(distributed/shard {:id :field-0 :value :field :device :gpu-0
                                :offsets [0 0] :shape [4 4]})
            (distributed/shard {:id :field-1 :value :field :device :gpu-1
                                :offsets [4 0] :shape [4 4]})
            (distributed/shard {:id :field-2 :value :field :device :gpu-2
                                :offsets [8 0] :shape [4 4]})]})

(deftest periodic-halo-adds-the-wrap-edge-in-one-round-on-a-ring
  (let [{:keys [value shards]} (three-shard-field)
        halo (distributed/schedule-halo
              (distributed/halo-exchange
               {:id :field-halo :value :field :axis 0 :width 1 :boundary :periodic})
              value shards (ring-routes) [])
        steps (:steps halo)
        wrap-forward (first (filter #(= [:field-halo :edge 2 :forward] (:id %)) steps))
        wrap-backward (first (filter #(= [:field-halo :edge 2 :backward] (:id %)) steps))]
    (is (= 6 (count steps)))
    (is (every? zero? (map #(get-in % [:attributes :round]) steps)))
    (is (= (mapv :id steps) (:completions halo)))
    ;; The wrap edge sends the last shard's upper face to the first shard's lower ghost.
    (is (= :gpu-2 (:source wrap-forward)))
    (is (= :gpu-0 (:target wrap-forward)))
    (is (true? (get-in wrap-forward [:attributes :wrap])))
    (is (= {:offsets [11 0] :shape [1 4]}
           (get-in wrap-forward [:attributes :source-region])))
    (is (= {:frame :target-local :offsets [-1 0] :shape [1 4]}
           (get-in wrap-forward [:attributes :destination-region])))
    (is (= {:offsets [0 0] :shape [1 4]}
           (get-in wrap-backward [:attributes :source-region])))
    (let [plan (distributed/plan
                {:id :periodic-ring
                 :mesh (distributed/mesh [{:name :space :size 3}] [:gpu-0 :gpu-1 :gpu-2])
                 :topology (three-device-topology)
                 :values {:field value} :shards {:field shards}
                 :halos [halo] :steps steps :outputs (:completions halo)})
          certificate (:certificate (distributed/certify plan))
          simulation (distributed/simulate plan)]
      (is (distributed/certified-plan? (distributed/verify! (distributed/certify plan))))
      (is (= :periodic (get-in certificate [:halos 0 :boundary])))
      (is (= 1 (get-in certificate [:halos 0 :rounds])))
      (is (nil? (get-in certificate [:halos 0 :combine])))
      (is (= 96 (:transferred-bytes simulation)))
      ;; Six distinct directed links, all legs overlap: one latency + 16 B serialization.
      (is (= 1001 (:makespan-ns simulation))))))

(deftest periodic-halo-on-two-shards-packs-the-wrap-edge-into-a-second-round
  (let [halo (distributed/schedule-halo
              (distributed/halo-exchange
               {:id :batch-halo :value :batch :axis 0 :width 1 :boundary :periodic})
              (:batch (training-values)) (:batch (training-shards))
              {[:gpu-0 :gpu-1] [:gpu-0->gpu-1]
               [:gpu-1 :gpu-0] [:gpu-1->gpu-0]}
              [:stencil-0 :stencil-1])
        steps (:steps halo)
        rounds (mapv #(get-in % [:attributes :round]) steps)
        second-round (filter #(= 1 (get-in % [:attributes :round])) steps)]
    (is (= [0 0 1 1] rounds))
    (is (= [[:batch-halo :edge 0 :forward] [:batch-halo :edge 0 :backward]]
           (:dependencies (first second-round))))
    (is (= [[:batch-halo :edge 1 :forward] [:batch-halo :edge 1 :backward]]
           (:completions halo)))
    (let [compute [(distributed/compute-step
                    {:id :stencil-0 :device :gpu-0 :duration-ns 100})
                   (distributed/compute-step
                    {:id :stencil-1 :device :gpu-1 :duration-ns 150})]
          plan (distributed/plan
                {:id :periodic-pair
                 :mesh (distributed/mesh [{:name :space :size 2}] [:gpu-0 :gpu-1])
                 :topology (two-device-topology)
                 :values (training-values) :shards (training-shards)
                 :halos [halo]
                 :steps (into compute steps)
                 :outputs (:completions halo)})
          simulation (distributed/simulate plan)]
      (is (= 2 (get-in (:certificate (distributed/certify plan)) [:halos 0 :rounds])))
      ;; Round 0 finishes at 150 + 1001; round 1 serializes behind it on the same links.
      (is (= 2152 (:makespan-ns simulation)))
      (is (= 64 (:transferred-bytes simulation))))))

(deftest accumulating-halo-lands-on-the-owned-face-and-records-its-monoid
  (let [combine (scan/certify {:acc 'acc :init 0.0 :lambda '(+ acc element)} :float)
        halo (distributed/schedule-halo
              (distributed/halo-exchange
               {:id :dss :value :batch :axis 0 :width 1 :combine combine})
              (:batch (training-values)) (:batch (training-shards))
              {[:gpu-0 :gpu-1] [:gpu-0->gpu-1]
               [:gpu-1 :gpu-0] [:gpu-1->gpu-0]}
              [])
        [forward backward] (:steps halo)
        plan (distributed/plan
              {:id :dss-probe
               :mesh (distributed/mesh [{:name :space :size 2}] [:gpu-0 :gpu-1])
               :topology (two-device-topology)
               :values (training-values) :shards (training-shards)
               :halos [halo] :steps (:steps halo) :outputs (:completions halo)})
        certificate (:certificate (distributed/certify plan))]
    (is (= :combine (get-in forward [:attributes :destination-mode])))
    ;; gpu-0's upper face accumulates into gpu-1's lower owned row, in global coordinates.
    (is (= {:frame :global :offsets [4 0] :shape [1 4]}
           (get-in forward [:attributes :destination-region])))
    (is (= {:frame :global :offsets [3 0] :shape [1 4]}
           (get-in backward [:attributes :destination-region])))
    (is (= {:combine (:combine combine) :identity (:identity combine) :dtype :float}
           (get-in certificate [:halos 0 :combine])))
    (is (distributed/certified-plan? (distributed/verify! (distributed/certify plan))))))

(deftest accumulating-halos-require-a-certified-monoid-of-the-value-dtype
  (let [uncertified (try
                      (distributed/halo-exchange
                       {:id :dss :value :batch :axis 0 :width 1 :combine '(+ acc element)})
                      (catch clojure.lang.ExceptionInfo exception exception))
        wrong-dtype (try
                      (distributed/schedule-halo
                       (distributed/halo-exchange
                        {:id :dss :value :batch :axis 0 :width 1
                         :combine (scan/certify {:acc 'acc :init 0.0 :lambda '(+ acc element)}
                                                :double)})
                       (:batch (training-values)) (:batch (training-shards))
                       {[:gpu-0 :gpu-1] [:gpu-0->gpu-1]
                        [:gpu-1 :gpu-0] [:gpu-1->gpu-0]}
                       [])
                      (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-halo-combine (:reason (ex-data uncertified))))
    (is (= :distributed-halo-combine-dtype (:reason (ex-data wrong-dtype))))))

(deftest halo-boundary-policy-is-validated
  (let [error (try
                (distributed/halo-exchange
                 {:id :bad :value :batch :axis 0 :width 1 :boundary :reflective})
                (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :distributed-halo-exchange (:reason (ex-data error))))))
