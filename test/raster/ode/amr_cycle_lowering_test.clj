(ns raster.ode.amr-cycle-lowering-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.link-composition :as composition]
            [raster.compiler.ir.link-plan :as plan]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.completed-evidence-device-test :as evidence]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as runtime]
            [raster.ode.amr-cycle-lowering :as cycle]
            [raster.ode.amr-subcycle :as subcycle]
            [raster.ode.amr-subcycle-test :as oracle]))

(defn- projection []
  (subcycle/project (#'oracle/hierarchy) {:domain-lengths [1.0 1.0] :diffusivity 0.2}))

(defn- field-node [lowered field]
  (let [local (get-in lowered [:lowering :plan])]
    (first (plan/value-node-ids local (get-in lowered [:fields field])))))

(deftest whole-cycle-lowering-preserves-two-state-owners-without-resources
  (let [target :ocl:analytic
        descriptor {:device-id target :device-type :gpu :backend :ocl :fp64? true
                    :subgroup-dialect :intel-opencl :max-workgroup-size 256}
        {:keys [coarse fine]} (#'oracle/initial-state)
        coarse (double-array coarse) fine (double-array fine)]
    (with-redefs [hardware/descriptor-for (constantly descriptor)
                  build/current-identity #'evidence/test-build
                  gpu/make-session (fn [& _] (throw (ex-info "lowering opened a device session" {})))
                  gpu/alloc! (fn [& _] (throw (ex-info "lowering allocated device storage" {})))
                  compiled/instantiate! (fn [& _] (throw (ex-info "lowering instantiated a program" {})))]
      (let [lowered (cycle/lower (projection) coarse fine 0.001 {:target target})
            local (get-in lowered [:lowering :plan])]
        (is (identical? (:lowering lowered) (composition/verify! (:lowering lowered))))
        (is (compiled/prepared? (:prepared lowered)))
        (is (identical? local (compiled/plan (:prepared lowered))))
        (is (= [:coarse :fine] (mapv :key (:out-tree (:prepared lowered)))))
        (is (= {[[:coarse :commit] :dst] :coarse
                 [[:fine :commit] :dst] :fine}
               (:donated (:prepared lowered))))
        (is (not-any? #{[:cycle :coarse] [:cycle :fine]}
                      (map :key (:in-tree (:prepared lowered)))))
        ;; Synthetic packaged evidence checks retention, not release provenance.
        (is (= :exact-bound-program (:scope (compiled/execution-identity (:prepared lowered)))))
        (is (= 3 (count (:instances local))))
        (is (= #{:coarse :fine} (set (keys (:fields lowered)))))
        (is (= (set (vals (:fields lowered))) (set (plan/output-value-ids local))))
        (is (= 0.001 (:dt lowered)))
        (doseq [[field source] [[:coarse coarse] [:fine fine]]]
          (let [node (get-in local [:nodes (field-node lowered field)])]
            (is (= :state (:role node)))
            (is (identical? source (:source node)))
            (is (= [16] (get-in node [:view :shape])))))
        (is (= 2 (count (get-in lowered [:lowering :certificate :mutable-shares]))))
        (is (= 2 (count (get-in lowered [:lowering :certificate :connections]))))
        (is (= :unproven (:release (plan/memory-report local))))
        (is (thrown? clojure.lang.ExceptionInfo
                     (composition/verify!
                      (update-in (:lowering lowered) [:plan :outputs] pop))))))))

(deftest cycle-lowering-rejects-invalid-options-before-preparation
  (doseq [options [nil {} {:target nil} {:target :ocl:analytic :id nil}
                   {:target :ocl:analytic :precision :float}]]
    (is (= :amr-cycle-options
           (:reason (ex-data (try (cycle/lower nil nil nil nil options)
                                 (catch clojure.lang.ExceptionInfo e e))))))))

(deftest coarse-and-fine-state-sources-must-be-distinct
  (let [state (double-array 16)]
    (is (= :amr-cycle-state-alias
           (:reason (ex-data (try (cycle/lower nil state state 0.001 {:target :ocl:analytic})
                                 (catch clojure.lang.ExceptionInfo e e))))))))

(defn- check-resident-cycle! [target]
  (let [{:keys [coarse fine]} (#'oracle/initial-state)
        coarse (double-array coarse) fine (double-array fine)
        initial-coarse (vec coarse) initial-fine (vec fine)
        p (projection)
        lowered (cycle/lower p coarse fine 0.001 {:target target})
        expected-first (#'oracle/reference-cycle initial-coarse initial-fine 0.001 false)
        expected-second (#'oracle/reference-cycle (:coarse expected-first) (:fine expected-first) 0.001 false)]
    (with-open [executable (runtime/instantiate! (get-in lowered [:lowering :plan]))]
      (doseq [expected [expected-first expected-second]]
        ;; The second invocation consumes the state committed by the first. No parameter
        ;; upload, source reinitialization or reset of a DistributedExecutable occurs.
        (runtime/run! executable)
        (let [actual (into {} (for [field [:coarse :fine]]
                                [field (vec (runtime/download executable (field-node lowered field)))]))]
          (doseq [field [:coarse :fine]]
            (is (every? #(< (Math/abs (double %)) 1.0e-11)
                        (map - (get expected field) (get actual field)))))
          (is (< (Math/abs (- (#'oracle/composite-mass initial-coarse initial-fine)
                              (#'oracle/composite-mass (:coarse actual) (:fine actual)))) 1.0e-11))
          (is (= (:coarse actual) (#'oracle/average-down (:coarse actual) (:fine actual))))))
      (is (= initial-coarse (vec coarse)))
      (is (= initial-fine (vec fine))))))

(deftest committed-cycle-retains-resident-state-on-local-devices
  (if @opencl/opencl-fp64-available?
    (check-resident-cycle! :ocl:0)
    (opencl/opencl-skip! "committed ratio-2 AMR cycle" :fp64))
  (if @ze/gpu-available?
    (let [caps ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
      (if (:fp64? caps)
        (check-resident-cycle! :ze:0)
        (ze/gpu-capability-skip! "committed ratio-2 AMR cycle" :fp64? caps)))
    (ze/gpu-skip! "committed ratio-2 AMR cycle")))
