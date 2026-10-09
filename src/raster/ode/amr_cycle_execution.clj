(ns raster.ode.amr-cycle-execution
  "Exact executable bindings for a bounded co-located temporal AMR cycle.
   Reuses schema-1 hierarchy-only workloads and the ordinary distributed compute authority.
   Mathematical stage meaning, synchronized input bytes and geometry remain producer evidence."
  (:require [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.link-composition :as composition]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.numerical-contract :as numerical]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.validate :refer [fail! positive-number? non-blank-string?]]
            [raster.gpu.compiled :as compiled]
            [raster.ode.amr-cycle-contract :as temporal]))

(defrecord CertifiedCycleExecution [workload prepared attestation certificate])

(defn- require! [condition reason message data]
  (when-not condition (fail! message reason data)))

(defn- cycle-instance [lowering component-id]
  (composition/verify! lowering)
  (let [component (some #(when (= component-id (:id %)) %) (:components lowering))
        instances (get-in component [:lowering :plan :instances])
        original (first instances)
        id (get-in lowering [:certificate :instance-mapping [component-id (:id original)]])
        instance (some #(when (= id (:id %)) %) (get-in lowering [:plan :instances]))]
    (require! (and (= 1 (count instances)) (link/program-link-instance? instance))
              :amr-cycle-component "cycle component must retain one equation-first program" {})
    instance))

(defn- register-witness [local instance stages register]
  (let [value (get-in instance [:call :buffers register])
        nodes (link/value-node-ids local value)
        report (link/memory-report local)
        node (get-in local [:nodes (first nodes)])
        allocation (get-in node [:view :allocation :id])
        events (filterv #(= value (:value %)) (:accesses report))
        stage-by-equation (into {} (mapcat (fn [stage]
                                            (map #(vector % (:kind stage)) (:equations stage))) stages))
        step-stages (mapv (fn [step]
                           (let [e (:equation step)
                                 ids (or (seq (get-in e [:attributes :emitted-source-equations])) [(:id e)])]
                             (set (keep stage-by-equation ids))))
                         (get-in instance [:call :steps]))
        stage-of (fn [event]
                   (let [kinds (get step-stages (:step event))]
                     (require! (= 1 (count kinds)) :amr-cycle-register-stage
                               "register access must retain one unambiguous semantic stage" {})
                     (first kinds)))
        first-event (first events) last-event (last events)]
    (require! (and (= 1 (count nodes)) (= :scratch (:role node))
                   (= :owned (get-in node [:view :allocation :ownership]))
                   (not (some #{value} (link/output-value-ids local)))
                   (every? (fn [[id other]]
                             (or (= id (:id node))
                                 (not= allocation (get-in other [:view :allocation :id]))))
                           (:nodes local)))
              :amr-cycle-register-owner "register must retain one private nonaliased scratch owner" {})
    (require! (and (seq events) (every? #(= (:id instance) (:instance %)) events)
                   (= :register-reset (stage-of first-event))
                   (= :write (:access first-event)) (:complete-write? first-event)
                   (= :reflux (stage-of last-event)) (= :read (:access last-event)))
              :amr-cycle-register-lifetime "register requires complete reset before transport and final consumption" {})
    (doseq [event (butlast (rest events))]
      (require! (and (contains? #{:coarse-transport :fine-transport} (stage-of event))
                     (= :read-write (:access event)))
                :amr-cycle-register-lifetime "intermediate register accesses must be transport updates" {}))
    (require! (= [:register-reset :coarse-transport :fine-transport :reflux]
                 (mapv first (partition-by identity (map stage-of events))))
              :amr-cycle-register-lifetime "register must retain both transport stages in order" {})
    {:value value :node (:id node) :events events
     :last-order (:order last-event) :release :unproven}))

(defn- derive-evidence [workload prepared attestation]
  (let [workload (amr/verify! workload)
        request-keys #{:producer :geometry-fingerprint :dt :dt-value :numerical
                       :cycle-component :register :stages :fields :completion}
        _ (require! (and (map? attestation) (= request-keys (set (keys attestation))))
                    :amr-cycle-attestation "cycle attestation requires its closed execution schema" {})
        {:keys [producer geometry-fingerprint dt dt-value numerical cycle-component
                register stages fields completion]} attestation
        _ (require! (and (or (keyword? producer) (symbol? producer)) (namespace producer)
                         (non-blank-string? geometry-fingerprint) (positive-number? dt))
                    :amr-cycle-attestation "cycle requires explicit producer, geometry and positive timestep" {})
        _ (numerical/validate! numerical)
        local (compiled/plan prepared)
        instance (cycle-instance (:lowering prepared) cycle-component)
        _ (require! (= {:type :double :value (double dt)}
                       (get-in instance [:call :scalar-values dt-value]))
                    :amr-cycle-timestep "attested timestep must equal the exact compiled scalar" {})
        temporal (temporal/stage-projection (get-in instance [:call :program]) register stages)
        p (:plan workload) hierarchy (:hierarchy p)
        levels (:levels hierarchy)
        patches (mapv #(first (:patches %)) levels)
        _ (require! (and (= :hierarchy-only (:mode p)) (= 2 (count (:base-shape hierarchy)))
                         (= 2 (count levels)) (every? #(= 1 (count (:patches %))) levels)
                         (= [2 2] (:ratio-to-parent (second levels)))
                         (<= 1 (:proper-nesting-width hierarchy)))
                    :amr-cycle-hierarchy "temporal cycle requires a properly nested 2D two-level ratio-2 hierarchy" {})
        manifest (get-in p [:state :manifest])
        _ (require! (= :synchronized (get-in manifest [:logical-coordinate :phase]))
                    :amr-cycle-input-phase "temporal cycle starts from caller-attested synchronized state" {})
        dp (get-in p [:distributed-plan :plan])
        steps (:steps dp) step (first steps)
        _ (require! (and (= 1 (count steps)) (= :compute (:kind step)) (= completion (:id step))
                         (= [completion] (:outputs dp))
                         (every? #(= (:device step) (:device %)) patches))
                    :amr-cycle-completion "bounded cycle requires one co-located complete compute call" {})
        binding (get-in (distributed/compute-bindings dp) [:bindings completion])
        device-plan (get-in dp [:device-plans (:device step)])
        _ (require! (and (= #{(:device step)} (set (keys (:device-plans dp))))
                         (= #{(:entry binding)} (set (keys (:entries device-plan))))
                         (= #{completion} (set (keys (:steps device-plan))))
                         (not (contains? device-plan :link-plan))
                         (not (contains? device-plan :execution-plan)))
                    :amr-cycle-program "bounded cycle must retain exactly its single local entry and call" {})
        _ (require! (= local (:link-plan binding)) :amr-cycle-program
                    "workload compute call must bind the exact prepared LinkPlan" {})
        output-ids (zipmap (map :key (:out-tree prepared)) (link/output-value-ids local))
        _ (require! (and (= #{:coarse :fine} (set (keys fields)))
                         (= [:coarse :fine] (mapv :key (:out-tree prepared)))
                         (= 2 (count (distinct (vals output-ids))))
                         (= (set (vals output-ids)) (set (keys (:values binding))))
                         (empty? (:boundary-outputs binding)))
                    :amr-cycle-fields "cycle must bind and export only two distinct whole state fields" {})
        complete (:complete-writes (link/initialization-contract local))
        manifest-fields (into {} (map (juxt :id identity)) (:fields manifest))
        checked-fields
        (into {}
              (for [[role patch] (map vector [:coarse :fine] patches)
                    :let [request (get fields role) value (output-ids role)
                          bound (get-in binding [:values value])
                          nodes (link/value-node-ids local value)
                          node (get-in local [:nodes (first nodes)])]]
                (do
                  (require! (and (= #{:patch :output} (set (keys request)))
                                 (= (:id patch) (:patch request)) (= role (:output request))
                                 (= (:field patch) (:value bound)) (= :read-write (:access bound))
                                 (= (:shape patch) (get-in bound [:domain :shape]))
                                 (= 1 (count (get-in bound [:domain :placements])))
                                 (= :owned (get-in bound [:domain :placements 0 :kind]))
                                 (= geometry-fingerprint
                                    (get-in manifest-fields [(:field patch) :coordinate-space :layout-fingerprint]))
                                 (= 1 (count nodes)) (= :state (:role node))
                                 (= :double (get-in node [:view :dtype]))
                                 (= :owned (get-in node [:view :allocation :ownership]))
                                 (contains? complete (:id node)))
                            :amr-cycle-field-binding "cycle field requires its whole owned FP64 patch and complete post-state"
                            {:role role})
                  [role {:patch (:id patch) :field (:field patch) :value value
                         :node (:id node) :domain (:domain bound)}])))
        register-evidence (register-witness local instance stages register)
        program-identity (compiled/execution-identity prepared)
        producer-interface (compiled/producer-interface prepared)]
    ;; The ordinary workload certificate owns live LinkPlans and their host sources.
    ;; Revalidate that owner above, but use the existing source-free execution identity
    ;; for the sole local entry in this structural witness. Never fingerprint arrays.
    {:workload (dissoc (:certificate workload) :distributed-certificate)
     :distributed-projection (assoc-in dp [:device-plans (:device step) :entries (:entry binding) :link-plan]
                                      program-identity)
     :program program-identity :producer-interface producer-interface
     :attestation attestation :temporal temporal :fields checked-fields
     :register register-evidence :effects (link/memory-report local)
     :completion completion :entry (:entry binding)
     ;; Readiness checks initializer source identity itself. Retain structural obligations,
     ;; not mutable host arrays: this certificate is not a snapshot of input bytes.
     :initialization (update (distributed/check-readiness dp) :initializers
                             #(mapv (fn [initializer] (dissoc initializer :source)) %))
     :completion-observed? false :geometry-bytes-attested? false
     :mathematical-meaning :producer-attested :release :unproven}))

(defn certify
  "Bind temporal producer evidence to the exact prepared program and hierarchy-only workload.
   Structural validation proves binding/coverage/order, not CFL, input bytes or conservation.
   No runtime resources are created and logical completion is not a completed-device receipt."
  [workload prepared attestation]
  (->CertifiedCycleExecution workload prepared attestation (derive-evidence workload prepared attestation)))

(defn verify! [execution]
  (require! (and (instance? CertifiedCycleExecution execution)
                 (= #{:workload :prepared :attestation :certificate} (set (keys execution))))
            :amr-cycle-execution-type "expected an exact CertifiedCycleExecution" {})
  (let [expected (derive-evidence (:workload execution) (:prepared execution) (:attestation execution))]
    (require! (fingerprint/equivalent? expected (:certificate execution))
              :amr-cycle-execution-certificate "cycle execution certificate differs from checked evidence" {}))
  execution)
