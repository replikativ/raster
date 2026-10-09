(ns raster.ode.amr-cycle-state-device-test
  "Actual distributed AMR capture/restart; bounded in-memory provider is not real durability."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.distributed :as runtime]
            [raster.ode.amr-cycle-execution :as execution]
            [raster.ode.amr-cycle-execution-test :as fixture]
            [raster.ode.amr-cycle-state :as capture]
            [raster.ode.amr-subcycle-test :as oracle]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state-test :as provider-fixture])
  (:import [java.lang.foreign MemorySegment]
           [java.nio ByteOrder]))

(defn- captured-cycle [target initial provider]
  (let [{:keys [workload prepared attestation]} (#'fixture/fixture target initial)
        workload (if-let [parent (:captured-state initial)]
                   (amr/certify (assoc (:plan workload) :state parent))
                   workload)
        certified (execution/certify workload prepared attestation)
        plan (get-in (execution/verify! certified) [:workload :plan :distributed-plan :plan])]
    (with-open [owner (runtime/instantiate! plan {:device-capacities {target 1048576}})]
      (runtime/run! owner)
      (let [fact (runtime/measure-storage-representation! owner target :double)]
        (capture/capture! certified owner {target fact} provider :local
                          {:id (keyword (str "step-" (inc (get initial :step 0))))
                           :logical-coordinate {:step (inc (get initial :step 0)) :phase :synchronized}})))))

(defn- decode-fields [captured provider]
  (into {}
        (for [[role field] (map vector [:coarse :fine] (get-in captured [:state :manifest :fields]))
              :let [chunk (first (:chunks field)) result (double-array 16)]]
          [role (content/with-local-content
                 provider (:content chunk) {:tier :local}
                 (fn [lease]
                   (content/decode-raw-array-chunk!
                    chunk lease :double (MemorySegment/ofArray result)
                    (if (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder)) :little-endian :big-endian))
                   (vec result)))])))

(defn- check-restart [target]
  (let [{:keys [provider events]} (#'provider-fixture/provider (fn [& _]))
        initial (#'oracle/initial-state)
        first-capture (captured-cycle target initial provider)
        ;; The old owner and every device context have closed before localization and restart.
        restored (decode-fields first-capture provider)
        second-capture (captured-cycle target (assoc restored :step 1 :captured-state (:state first-capture)) provider)
        actual (decode-fields second-capture provider)
        first-expected (#'oracle/reference-cycle (:coarse initial) (:fine initial) 0.001 false)
        expected (#'oracle/reference-cycle (:coarse first-expected) (:fine first-expected) 0.001 false)]
    (doseq [role [:coarse :fine]]
      (is (every? #(< (Math/abs (double %)) 1.0e-11) (map - (role expected) (role actual)))))
    (is (empty? @events))
    (is (= [(get-in first-capture [:state :manifest :id])]
           (get-in second-capture [:state :manifest :parents])))
    (is (= 2 (get-in second-capture [:state :manifest :logical-coordinate :step])))))

(deftest distributed-capture-survives-owner-close-and-fresh-context-restart
  (if @opencl/opencl-fp64-available?
    (check-restart :ocl:0)
    (opencl/opencl-skip! "distributed AMR capture/restart" :fp64))
  (if @ze/gpu-available?
    (let [caps ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
      (if (:fp64? caps) (check-restart :ze:0)
          (ze/gpu-capability-skip! "distributed AMR capture/restart" :fp64? caps)))
    (ze/gpu-skip! "distributed AMR capture/restart")))
