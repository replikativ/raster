(ns raster.compiler.ir.amr-execution
  "Executable bindings for analytical AMR workloads.
   Structural checks bind producer attestations to exact compiler plans. They do not prove the
   producer's mathematical claim, and do not change the runtime ABI or recognize operator names."
  (:require [clojure.set :as set]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.numerical-contract :as numerical]
            [raster.compiler.ir.validate :refer [fail!]]))

(defrecord OperatorImplementation [operation contract link-plan source target producer invariants numerical])
(defrecord CertifiedAMRExecution [workload implementations bindings])

(defn- producer-identity? [x]
  (and (or (keyword? x) (symbol? x)) (some? (namespace x))))

(defn operation-contract
  "Derive the whole-patch semantic contract a numerical provider must implement."
  [workload operation-id]
  (let [plan (:plan (amr/verify! workload))
        hierarchy (:hierarchy plan)
        patches (into {} (map (juxt :id identity)) (mapcat :patches (:levels hierarchy)))
        scheduled (some #(when (= operation-id (get-in % [:operation :id])) %) (:coarse-fine plan))
        operation (:operation scheduled)
        source (patches (:source-patch operation)) target (patches (:target-patch operation))
        fine (if (> (:level source -1) (:level target -1)) source target)]
    (when-not scheduled
      (fail! "implementation names an absent AMR operation" :amr-execution-operation {:operation operation-id}))
    (doseq [[patch region] [[source (:source-region operation)] [target (:target-region operation)]]]
      (when-not (= {:offsets (vec (repeat (count (:shape patch)) 0)) :shape (:shape patch)} region)
        (fail! "executable AMR bindings require full patch regions until narrow effects are certified"
               :amr-execution-region {:operation operation-id :patch (:id patch) :region region})))
    {:operation operation :completion (:completion scheduled)
     :source source :target target :centering (:centering hierarchy)
     :ratio (:ratio-to-parent (nth (:levels hierarchy) (:level fine)))
     :source-value (get-in plan [:distributed-plan :plan :values (:field source)])
     :target-value (get-in plan [:distributed-plan :plan :values (:field target)])}))

(defn attest-implementation
  "Trusted producer boundary: attest that this exact LinkPlan implements the workload operation.
   Call from the implementation provider after compiling its numerical program. Invariants and
   numerical policy are explicit producer evidence, not consequences inferred from matching shapes."
  [workload operation-id plan {:keys [source target producer invariants numerical]}]
  (let [contract (operation-contract workload operation-id)]
    (link/validate! plan)
    (when-not (and (producer-identity? producer) (some? source) (some? target) (not= source target)
                   (set? invariants) (every? keyword? invariants))
      (fail! "implementation requires a producer, distinct roles and explicit invariants"
             :amr-execution-attestation {:operation operation-id}))
    (numerical/validate! numerical)
    (->OperatorImplementation operation-id contract plan source target producer invariants numerical)))

(defn certify
  "Bind every coarse/fine operation to exact executable storage and producer evidence.
   Returns a structural execution certificate; mathematical equivalence remains producer-attested."
  [workload implementations]
  (let [workload (amr/verify! workload)
        plan (get-in workload [:plan :distributed-plan :plan])
        operations (get-in workload [:plan :coarse-fine])
        _ (when-not (and (map? implementations)
                         (= (set (map #(get-in % [:operation :id]) operations)) (set (keys implementations))))
            (fail! "every AMR operation requires exactly one implementation witness"
                   :amr-execution-implementations {}))
        ready (distributed/check-readiness plan)
        bindings (:bindings (distributed/compute-bindings plan))
        checked
        (into {}
              (for [[id implementation] implementations
                    :let [contract (operation-contract workload id)
                          completion (:completion contract)
                          binding (bindings completion)
                          source-id (:source implementation) target-id (:target implementation)
                          source (get-in binding [:values source-id]) target (get-in binding [:values target-id])]]
                (do
                  (when-not (and (instance? OperatorImplementation implementation)
                                 (= id (:operation implementation))
                                 (= contract (:contract implementation))
                                 (= (:link-plan binding) (:link-plan implementation)))
                    (fail! "producer witness does not bind this operation's exact compiler plan"
                           :amr-execution-implementation {:operation id}))
                  (when-not (and (not= source-id target-id)
                                 (empty? (:boundary-outputs binding))
                                 (= (set [source-id target-id]) (set (keys (:values binding))))
                                 (= :read (:access source))
                                 (= :write (:access target))
                                 (= #{target-id} (set (link/output-value-ids (:link-plan binding))))
                                 (set/subset? (set (link/value-node-ids (:link-plan binding) target-id))
                                              (:produces (link/initialization-contract (:link-plan binding))))
                                 (= (:value source) (get-in contract [:source :field]))
                                 (= (:value target) (get-in contract [:target :field])))
                    (fail! "AMR implementation must read only its source field and write/export its target field"
                           :amr-execution-effects {:operation id :values (:values binding)}))
                  (doseq [[role value] [[:source source] [:target target]]
                          :let [patch (get contract role)
                                shard (some #(when (= (:shard value) (:id %)) %)
                                            (get-in plan [:shards (:field patch)]))]]
                    (when-not (and (= (:device patch) (:device shard))
                                   (= (:shape patch) (:shape shard))
                                   (every? zero? (:offsets shard))
                                   (some? (:domain value))
                                   (= (:shape patch) (get-in value [:domain :shape]))
                                   (= 1 (count (get-in value [:domain :placements])))
                                   (= :owned (get-in value [:domain :placements 0 :kind])))
                      (fail! "AMR role must realize the whole declared patch field"
                             :amr-execution-shard {:operation id :role role})))
                  (when-not (and (producer-identity? (:producer implementation))
                                 (set? (:invariants implementation))
                                 (every? keyword? (:invariants implementation))
                                 (set/subset? (get-in contract [:operation :operator :required-invariants])
                                              (:invariants implementation)))
                    (fail! "producer attestation does not establish all requested invariants"
                           :amr-execution-invariants {:operation id}))
                  (numerical/validate! (:numerical implementation))
                  [id {:completion completion :entry (:entry binding)
                       :source source-id :target target-id :contract contract}])))]
    (->CertifiedAMRExecution workload implementations
                             {:operations checked :readiness ready})))

(defn verify! [execution]
  (when-not (instance? CertifiedAMRExecution execution)
    (fail! "expected a CertifiedAMRExecution" :amr-execution-type {}))
  (let [expected (certify (:workload execution) (:implementations execution))]
    (when-not (= expected execution)
      (fail! "AMR execution certificate differs from its checked bindings" :amr-execution-certificate {}))
    execution))
