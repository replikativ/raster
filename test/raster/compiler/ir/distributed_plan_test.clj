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
            :boundary :nonperiodic
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
