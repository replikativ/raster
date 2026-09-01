(ns raster.compiler.ir.amr-plan-test
  (:refer-clojure :exclude [chunk])
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.numerical-state :as state]))

(defn- thrown-data
  [f]
  (try
    (f)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (ex-data exception))))

(defn- value
  ([shape] (value shape nil))
  ([shape sharding]
   (abstract-value/tensor
    {:dtype :float :shape shape :logical-layout {:order :row-major}
     :representation {:kind :plain} :sharding sharding :ownership :owned})))

(defn- patch-field
  [field-id patch-id level]
  (state/field
   {:id field-id
    :value (value [8])
    :chunk-shape [8]
    :coordinate-space {:hierarchy :heat/amr :level level :patch patch-id
                       :axes [{:name :x :centering :cell}]}
    :chunks
    [(state/chunk
      {:id [field-id :chunk 0] :offsets [0] :shape [8]
       :logical-byte-length 32 :stored-byte-length 32
       :content (state/content-address
                 :sha-256 (format "%064x" (inc level)))
       :storage {:format :raw-array :byte-order :little-endian}})]}))

(defn- numerical-state
  ([] (numerical-state {}))
  ([coordinate-overrides]
   (let [base (patch-field :temperature/l0 :patch/l0 0)
         fine (patch-field :temperature/l1 :patch/l1 1)
         fields (mapv (fn [candidate]
                        (if-let [coordinates (get coordinate-overrides (:id candidate))]
                          (assoc candidate :coordinate-space coordinates)
                          candidate))
                      [base fine])]
     (state/certify
      (state/manifest
       {:id :heat/step-8
        :parents [:heat/step-7]
        :logical-coordinate {:step 8 :time 0.125}
        :fields fields
        :numerical-contract
        {:mode :ieee-fp32 :determinism :reproducible-order
         :compatibility-id "heat-amr-v1:f32"}
        :provenance {:program-fingerprint "sha256:heat-amr-rk2"}})))))

(defn- hierarchy
  ([] (hierarchy {}))
  ([fine-overrides]
   (amr/hierarchy
    {:id :heat/amr :base-shape [8] :proper-nesting-width 1
     :levels
     [(amr/level
       {:id :level/l0 :index 0 :ratio-to-parent nil
        :patches [(amr/patch
                   {:id :patch/l0 :level 0 :device :gpu-0
                    :offsets [0] :shape [8] :field :temperature/l0})]})
      (amr/level
       {:id :level/l1 :index 1 :ratio-to-parent [2]
        :patches [(amr/patch
                   (merge {:id :patch/l1 :level 1 :device :gpu-1
                           :offsets [4] :shape [8] :field :temperature/l1}
                          fine-overrides))]})]})))

(defn- topology
  []
  (distributed/topology
   [(distributed/device {:id :gpu-0 :memory-capacity-bytes 1000000})
    (distributed/device {:id :gpu-1 :memory-capacity-bytes 1000000})]
   [(distributed/link {:id :gpu-0->gpu-1 :source :gpu-0 :target :gpu-1
                       :kind :fabric :bandwidth-bytes-s 25.0e9 :latency-ns 1000})
    (distributed/link {:id :gpu-1->gpu-0 :source :gpu-1 :target :gpu-0
                       :kind :fabric :bandwidth-bytes-s 25.0e9 :latency-ns 1000})]))

(defn- values
  []
  {:temperature/l0 (value [8] {:kind :partitioned :axis 0 :devices [:gpu-0]})
   :temperature/l1 (value [8] {:kind :partitioned :axis 0 :devices [:gpu-1]})})

(defn- shards
  []
  {:temperature/l0
   [(distributed/shard {:id :temperature/l0-shard :value :temperature/l0
                        :device :gpu-0 :offsets [0] :shape [8]})]
   :temperature/l1
   [(distributed/shard {:id :temperature/l1-shard :value :temperature/l1
                        :device :gpu-1 :offsets [0] :shape [8]})]})

(defn- operations
  [hierarchy]
  (let [prolong
        (amr/coarse-fine-operation
         {:id :fill-fine :kind :prolongation
          :source-patch :patch/l0 :target-patch :patch/l1
          :source-region {:offsets [2] :shape [4]}
          :target-region {:offsets [0] :shape [8]}
          :operator {:method :piecewise-linear
                     :required-invariants #{:constant-preserving}}})
        restrict
        (amr/coarse-fine-operation
         {:id :average-down :kind :restriction
          :source-patch :patch/l1 :target-patch :patch/l0
          :source-region {:offsets [0] :shape [8]}
          :target-region {:offsets [2] :shape [4]}
          :operator {:method :cell-average
                     :required-invariants #{:conservative :constant-preserving}}})
        distributed-values (values)
        scheduled-prolong
        (amr/schedule-coarse-fine
         hierarchy distributed-values prolong
         {:route [:gpu-0->gpu-1] :duration-ns 20
          :dependencies [:coarse-advance]})
        scheduled-restrict
        (amr/schedule-coarse-fine
         hierarchy distributed-values restrict
         {:route [:gpu-1->gpu-0] :duration-ns 15
          :dependencies [:fine-advance]})]
    [scheduled-prolong scheduled-restrict]))

(defn- workload
  ([] (workload (hierarchy) (numerical-state)))
  ([hierarchy numerical-state]
   (let [[prolong restrict :as coarse-fine] (operations hierarchy)
         steps (vec (concat
                     [(distributed/compute-step
                       {:id :coarse-advance :device :gpu-0 :duration-ns 100})]
                     (:steps prolong)
                     [(distributed/compute-step
                       {:id :fine-advance :device :gpu-1 :duration-ns 50
                        :dependencies [(:completion prolong)]})]
                     (:steps restrict)))
         distributed-plan
         (distributed/certify
          (distributed/plan
           {:id :heat/amr-step
            :mesh (distributed/mesh [{:name :level-placement :size 2}]
                                    [:gpu-0 :gpu-1])
            :topology (topology) :values (values) :shards (shards)
            :steps steps :outputs [(:completion restrict)]}))]
     (amr/plan
      {:id :heat/amr-workload :mode :transfer-cycle
       :hierarchy hierarchy :state numerical-state
       :distributed-plan distributed-plan :coarse-fine coarse-fine
       :attributes {:workload :heat-probe}}))))

(deftest two-level-workload-certifies-durable-state-routes-and-cost
  (let [candidate (workload)
        simulation (amr/simulate candidate)
        certified (amr/certify candidate)
        certificate (:certificate certified)]
    (is (amr/certified-plan? (amr/verify! certified)))
    (is (= :heat/step-8 (get-in certificate [:state-certificate :state-id])))
    (is (= [[:fill-fine 16] [:average-down 32]]
           (mapv (juxt :id :bytes) (:operations certificate))))
    (is (= {:offsets [2] :shape [4]}
           (get-in certificate [:operations 0 :source-region])))
    (is (= [:gpu-0->gpu-1]
           (get-in certificate [:operations 0 :route])))
    (is (= 48 (:transferred-bytes simulation)))
    (is (= 2188 (:makespan-ns simulation)))
    (is (= (:cost-vector simulation) (:cost-vector certificate)))
    (is (= {:offsets [2] :shape [4]}
           (get-in candidate [:coarse-fine 0 :steps 0 :attributes :source-region])))
    (is (= {:offsets [0] :shape [8]}
           (get-in candidate [:coarse-fine 0 :steps 0 :attributes :target-region])))))

(deftest refinement-is-aligned-nested-and-nonoverlapping
  (testing "fine patch boundaries align to parent cells"
    (is (= :amr-patch-alignment
           (:reason (thrown-data #(hierarchy {:offsets [3]}))))))
  (testing "proper nesting is a certified margin, not a scheduler hint"
    (is (= :amr-proper-nesting
           (:reason (thrown-data #(hierarchy {:offsets [0] :shape [8]}))))))
  (testing "base coverage remains exact"
    (is (= :amr-base-coverage
           (:reason
            (thrown-data
             #(amr/hierarchy
               {:id :bad :base-shape [8]
                :levels [(amr/level
                          {:id :level/l0 :index 0
                           :patches [(amr/patch
                                      {:id :short :level 0 :device :gpu-0
                                       :offsets [0] :shape [7] :field :short})]})]}))))))
  (testing "touching is allowed but overlapping fine patches are rejected"
    (is (= :amr-patch-overlap
           (:reason
            (thrown-data
             #(amr/hierarchy
               {:id :overlap :base-shape [8]
                :levels
                [(amr/level
                  {:id :l0 :index 0
                   :patches [(amr/patch
                              {:id :base :level 0 :device :gpu-0
                               :offsets [0] :shape [8] :field :base})]})
                 (amr/level
                  {:id :l1 :index 1 :ratio-to-parent [2]
                   :patches [(amr/patch
                              {:id :fine-a :level 1 :device :gpu-0
                               :offsets [2] :shape [4] :field :fine-a})
                             (amr/patch
                              {:id :fine-b :level 1 :device :gpu-1
                               :offsets [4] :shape [4] :field :fine-b})]})]})))))))

(deftest anisotropic-two-dimensional-regions-retain-rank-and-byte-cost
  (let [hierarchy
        (amr/hierarchy
         {:id :amr/two-d :base-shape [8 6]
          :levels
          [(amr/level
            {:id :l0 :index 0
             :patches [(amr/patch
                        {:id :coarse :level 0 :device :gpu-0 :offsets [0 0]
                         :shape [8 6] :field :coarse})]})
           (amr/level
            {:id :l1 :index 1 :ratio-to-parent [2 3]
             :patches [(amr/patch
                        {:id :fine :level 1 :device :gpu-1 :offsets [4 6]
                         :shape [8 6] :field :fine})]})]})
        values {:coarse (value [8 6]) :fine (value [8 6])}
        operation
        (amr/coarse-fine-operation
         {:id :fill-2d :kind :prolongation :source-patch :coarse :target-patch :fine
          :source-region {:offsets [2 2] :shape [4 2]}
          :target-region {:offsets [0 0] :shape [8 6]}
          :operator {:method :linear
                     :required-invariants #{:constant-preserving}}})
        scheduled (amr/schedule-coarse-fine
                   hierarchy values operation
                   {:route [:gpu-0->gpu-1] :duration-ns 10})]
    (is (= 32 (get-in scheduled [:steps 0 :bytes])))
    (is (= {:offsets [2 2] :shape [4 2]}
           (get-in scheduled [:steps 0 :attributes :source-region])))
    (is (= {:offsets [0 0] :shape [8 6]}
           (get-in scheduled [:steps 0 :attributes :target-region])))))

(deftest coarse-and-fine-regions-must-name-the-same-physical-cells
  (let [candidate
        (amr/coarse-fine-operation
         {:id :misregistered :kind :prolongation
          :source-patch :patch/l0 :target-patch :patch/l1
          :source-region {:offsets [1] :shape [4]}
          :target-region {:offsets [0] :shape [8]}
          :operator {:method :linear
                     :required-invariants #{:constant-preserving}}})]
    (is (= :amr-operation-region-mapping
           (:reason
            (thrown-data
             #(amr/schedule-coarse-fine
               (hierarchy) (values) candidate
               {:route [:gpu-0->gpu-1] :duration-ns 20})))))))

(deftest durable-coordinate-and-distributed-shard-bindings-are-load-bearing
  (testing "a durable tile cannot silently move to another hierarchy patch"
    (let [bad-state
          (numerical-state
           {:temperature/l1 {:hierarchy :heat/amr :level 1 :patch :different-patch}})]
      (is (= :amr-patch-coordinate-space
             (:reason (thrown-data #(workload (hierarchy) bad-state)))))))
  (testing "every scheduled expansion must occur exactly in the distributed DAG"
    (let [candidate (workload)
          omitted (distributed/certify
                   (distributed/plan
                    (assoc (into {} (get-in candidate [:distributed-plan :plan]))
                           :steps [(distributed/compute-step
                                    {:id :coarse-advance :device :gpu-0 :duration-ns 100})]
                           :outputs [:coarse-advance])))
          changed (assoc candidate :distributed-plan omitted)]
      (is (= :amr-plan-operation-step-coverage
             (:reason (thrown-data #(amr/validate! changed))))))))

(deftest schema-mode-centering-and-coordinate-axes-are-explicit
  (let [candidate (workload)]
    (testing "schema revisions fail instead of being interpreted as version 1"
      (is (= :amr-schema-version
             (:reason (thrown-data #(amr/validate! (assoc candidate :schema-version 2)))))))
    (testing "an executable cycle cannot be relabelled hierarchy-only"
      (is (= :amr-plan-mode
             (:reason (thrown-data #(amr/validate! (assoc candidate :mode :hierarchy-only)))))))
    (testing "version 1 cell geometry is load-bearing"
      (is (= :amr-centering
             (:reason
              (thrown-data
               #(amr/validate! (assoc-in candidate [:hierarchy :centering] :node))))))))
  (testing "durable axes must be rank-matched, canonical, and cell-centred"
    (doseq [axes [nil [{:name :x :centering :node}]
                  [{:name :y :centering :cell}]]]
      (let [bad-state
            (numerical-state
             {:temperature/l1
              (cond-> {:hierarchy :heat/amr :level 1 :patch :patch/l1}
                axes (assoc :axes axes))})]
        (is (= :amr-patch-coordinate-axes
               (:reason (thrown-data #(workload (hierarchy) bad-state)))))))))

(deftest coarse-fine-operators-have-typed-storage-and-access-boundaries
  (let [hierarchy (hierarchy)
        operation (:operation (first (operations hierarchy)))
        incompatible (assoc-in (values) [:temperature/l1 :dtype] :double)]
    (is (= :amr-operation-storage-contract
           (:reason
            (thrown-data
             #(amr/schedule-coarse-fine
               hierarchy incompatible operation
               {:route [:gpu-0->gpu-1] :duration-ns 20})))))
    (is (= :amr-operation-operator
           (:reason
            (thrown-data
             #(amr/coarse-fine-operation
               {:id :unproved :kind :prolongation
                :source-patch :patch/l0 :target-patch :patch/l1
                :source-region {:offsets [2] :shape [4]}
                :target-region {:offsets [0] :shape [8]}
                :operator {:method :linear :required-invariants #{}}})))))
    (is (= {:source :read :target :write}
           (get-in (first (operations hierarchy)) [:steps 0 :attributes :access])))
    (is (= :temperature/l1
           (get-in (first (operations hierarchy)) [:steps 0 :attributes :target-field])))))

(deftest same-device-coarse-fine-scheduling-needs-no-fabric-transfer
  (let [local-hierarchy (hierarchy {:device :gpu-0})
        operation (:operation (first (operations (hierarchy))))
        scheduled (amr/schedule-coarse-fine
                   local-hierarchy (values) operation
                   {:duration-ns 12 :dependencies [:coarse-advance]})]
    (is (= 1 (count (:steps scheduled))))
    (is (= :compute (:kind (first (:steps scheduled)))))
    (is (= [] (:route scheduled)))
    (is (= [[:fill-fine :apply]] [(:completion scheduled)]))))

(deftest amr-certificates-detect-semantic-or-cost-drift
  (let [certified (amr/certify (workload))
        changed-plan (assoc-in certified [:plan :coarse-fine 0 :duration-ns] 21)
        changed-certificate (assoc-in certified
                                      [:certificate :operations 0 :source-region :offsets]
                                      [1])
        replacement-state
        (state/certify
         (assoc-in (get-in certified [:plan :state :manifest])
                   [:fields 1 :chunks 0 :content]
                   (state/content-address :sha-256 (format "%064x" 99))))
        changed-state (assoc-in certified [:plan :state] replacement-state)]
    (is (= :amr-operation-lowering
           (:reason (thrown-data #(amr/verify! changed-plan)))))
    (is (= :amr-certificate
           (:reason (thrown-data #(amr/verify! changed-certificate)))))
    (is (= :amr-certificate
           (:reason (thrown-data #(amr/verify! changed-state)))))))
