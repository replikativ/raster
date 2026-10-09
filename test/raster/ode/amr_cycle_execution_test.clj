(ns raster.ode.amr-cycle-execution-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.equation-artifact-store :as artifact-store]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.numerical-state :as state]
            [raster.gpu.completed-evidence-device-test :as evidence]
            [raster.ode.amr-cycle-contract-test :as stages]
            [raster.ode.amr-cycle-execution :as execution]
            [raster.ode.amr-cycle-lowering :as lowering]
            [raster.ode.amr-subcycle :as subcycle]
            [raster.ode.amr-subcycle-test :as oracle])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(defn- with-fresh-store [f]
  ;; Exercise real artifact sealing without reusing an older development artifact
  ;; under the deliberately fixed synthetic build identity.
  (let [root (Files/createTempDirectory "raster-amr-execution-" (make-array FileAttribute 0))]
    (try
      (binding [raster.gpu.compiled/*equation-artifact-store*
                (artifact-store/make-store {:root (.toFile root)})]
        (f))
      (finally
        (doseq [file (reverse (file-seq (.toFile root)))]
          (Files/deleteIfExists (.toPath file)))))))

(defn- fixture []
  (let [worker :amr-worker target :ocl:analytic
        hierarchy (-> (#'oracle/hierarchy)
                      (assoc :proper-nesting-width 1)
                      (update :levels #(mapv (fn [level]
                                              (update level :patches
                                                      (fn [patches] (mapv (fn [p] (assoc p :device worker)) patches)))) %)))
        projection (subcycle/project hierarchy {:domain-lengths [1.0 1.0] :diffusivity 0.2})
        initial (#'oracle/initial-state)
        lowered (with-fresh-store
                  #(with-redefs [hardware/descriptor-for
                             (constantly {:device-id target :device-type :gpu :backend :ocl :fp64? true
                                          :subgroup-dialect :intel-opencl :max-workgroup-size 256})
                             build/current-identity #'evidence/test-build]
                  (lowering/lower projection (double-array (:coarse initial))
                                  (double-array (:fine initial)) 0.001 {:target target})))
        local (get-in lowered [:lowering :plan])
        patches (mapv #(first (:patches %)) (:levels hierarchy))
        values (into {} (for [patch patches]
                          [(:field patch) (av/tensor {:dtype :double :shape (:shape patch)
                                                     :sharding {:kind :partitioned :axis 0 :devices [worker]}})]))
        bindings (into {} (map (fn [role patch]
                                [(get-in lowered [:fields role])
                                 {:local-shape (:shape patch)
                                  :placements [{:kind :owned :value (:field patch) :shard (:id patch)
                                                :local-offsets [0 0]}]}]) [:coarse :fine] patches))
        dp (distributed/plan
            {:id :temporal-cycle :mesh (distributed/mesh [{:name :worker :size 1}] [worker])
             :topology (distributed/topology [(distributed/device {:id worker :memory-capacity-bytes 1048576})] [])
             :values values
             :shards (into {} (for [patch patches]
                               [(:field patch) [(distributed/shard {:id (:id patch) :value (:field patch)
                                                                   :device worker :offsets [0 0] :shape (:shape patch)})]]))
             :device-plans {worker {:target target :entries {:cycle {:link-plan local}}
                                    :steps {:cycle-complete {:entry :cycle :bindings bindings}}}}
             :steps [(distributed/compute-step {:id :cycle-complete :device worker :duration-ns 1})]
             :outputs [:cycle-complete]})
        manifest (state/manifest
                  {:id :cycle-state :parents [] :logical-coordinate {:step 0 :phase :synchronized}
                   :fields (mapv (fn [patch]
                                   (let [field (:field patch) shape (:shape patch) n (reduce * shape)]
                                     (state/field
                                      {:id field :value (values field) :chunk-shape shape
                                       :coordinate-space {:hierarchy (:id hierarchy) :level (:level patch)
                                                          :patch (:id patch) :layout-fingerprint (:geometry-fingerprint lowered)
                                                          :axes [{:name :x :centering :cell} {:name :y :centering :cell}]}
                                       ;; Structural fixture only, not a published byte snapshot.
                                       :chunks [(state/chunk {:id field :offsets [0 0] :shape shape
                                                              :logical-byte-length (* 8 n) :stored-byte-length (* 8 n)
                                                              :content (state/content-address :sha-256 (format "%064x" n))
                                                              :storage {:format :raw-array :byte-order :little-endian}})]}))) patches)
                   :numerical-contract {:mode :ieee-fp64 :determinism :reproducible-order :compatibility-id "amr-cycle-fixture"}
                   :provenance {:program-fingerprint "structural-cycle-fixture"}})
        workload (amr/certify (amr/plan {:id :cycle-workload :mode :hierarchy-only :hierarchy hierarchy
                                        :state (state/certify manifest) :distributed-plan (distributed/certify dp)
                                        :coarse-fine []}))
        attestation {:producer :test/diffusion :geometry-fingerprint (:geometry-fingerprint lowered)
                     :dt 0.001 :dt-value 'dt :numerical {:mode :exact :policy :retained-order}
                     :cycle-component :cycle :register 'register
                     :stages (#'stages/stages (get-in local [:instances 0 :call :program]))
                     :fields (into {} (map (fn [role patch] [role {:patch (:id patch) :output role}])
                                          [:coarse :fine] patches))
                     :completion :cycle-complete}]
    {:workload workload :prepared (:prepared lowered) :attestation attestation}))

(def ^:private retained-cycle (delay (fixture)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest exact-cycle-binding-retains-structural-obligations
  (let [{:keys [workload prepared attestation]} @retained-cycle
        certified (execution/certify workload prepared attestation)
        certificate (:certificate certified)]
    (is (identical? certified (execution/verify! certified)))
    (is (= #{:coarse :fine} (set (keys (:fields certificate)))))
    (is (= :reflux (get-in certificate [:temporal :register-last-stage])))
    (is (= :cycle-complete (:completion certificate)))
    (is (false? (:completion-observed? certificate)))
    (is (false? (:geometry-bytes-attested? certificate)))
    (is (= :producer-attested (:mathematical-meaning certificate)))
    (is (= :unproven (:release certificate)))
    (is (every? #(not (contains? % :source)) (get-in certificate [:initialization :initializers])))))

(deftest producer-evidence-must-match-exact-call-and-hierarchy
  (let [{:keys [workload prepared attestation]} @retained-cycle]
    (doseq [[bad expected] [[(assoc attestation :dt 0.002) :amr-cycle-timestep]
                            [(assoc attestation :completion :other) :amr-cycle-completion]
                            [(assoc attestation :geometry-fingerprint "other") :amr-cycle-field-binding]
                            [(assoc-in attestation [:fields :coarse :output] :fine) :amr-cycle-field-binding]
                            [(assoc attestation :extra true) :amr-cycle-attestation]
                            [(assoc attestation :register 'dt) :amr-temporal-register]]]
      (is (= expected (reason #(execution/certify workload prepared bad)))))))

(deftest execution-certificate-and-attestation-cannot-drift
  (let [{:keys [workload prepared attestation]} @retained-cycle
        certified (execution/certify workload prepared attestation)]
    (doseq [bad [(assoc-in certified [:certificate :completion-observed?] true)
                 (assoc-in certified [:certificate :register :release] :complete)
                 (assoc-in certified [:attestation :producer] :other/producer)
                 (assoc-in certified [:prepared :schedule] {:other true})
                 (assoc certified :extra true)]]
      (is (some? (reason #(execution/verify! bad)))))))

(deftest both-physical-register-transport-stages-are-required
  (let [{:keys [prepared attestation]} @retained-cycle
        local (raster.gpu.compiled/plan prepared)
        instance (first (:instances local))
        report (link/memory-report local)
        stages (:stages attestation)]
    ;; Isolate the physical lifecycle projection after ordinary plan verification.
    ;; Either omitted transport stage must be rejected, even if reset and reflux remain.
    (doseq [kind [:coarse-transport :fine-transport]
            :let [ids (set (:equations (some #(when (= kind (:kind %)) %) stages)))
                  incomplete (update report :accesses
                                     #(filterv (fn [event] (not (contains? ids (:phase event)))) %))]]
      (with-redefs [link/memory-report (constantly incomplete)]
        (is (= :amr-cycle-register-lifetime
               (reason #(#'execution/register-witness local instance stages 'register))))))))
