(ns raster.ode.amr-cycle-distributed-device-test
  "Certified temporal cycle through the existing one-shot distributed runtime.
   Local co-location, synchronous readback and fresh contexts; no fabric or durable restart claim."
  (:require [clojure.test :refer [deftest is]]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.distributed :as runtime]
            [raster.ode.amr-cycle-execution :as execution]
            [raster.ode.amr-cycle-execution-test :as fixture]
            [raster.ode.amr-subcycle-test :as oracle]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- run-cycle! [target initial]
  (let [{:keys [workload prepared attestation]} (#'fixture/fixture target initial)
        certified (execution/certify workload prepared attestation)
        plan (get-in (execution/verify! certified) [:workload :plan :distributed-plan :plan])]
    (with-open [executable (runtime/instantiate! plan {:device-capacities {target 1048576}})]
      (is (= :distributed-runtime-state (reason #(runtime/output-values executable)))
          "outputs are unavailable before successful completion")
      (runtime/run! executable)
      (is (= :complete @(:state executable)))
      (is (= :distributed-runtime-state (reason #(runtime/run! executable)))
          "the one-shot runtime must not replay stale initialization evidence")
      (runtime/with-output-values!
       executable
       (fn [completed-outputs]
        (let [outputs (get completed-outputs (:completion attestation))
            session (get (:sessions executable) target)
            fields (get-in certified [:certificate :fields])]
        (is (= (set (map :value (vals fields))) (set (keys outputs))))
        (is (false? (get-in certified [:certificate :completion-observed?]))
            "structural certificate is not retroactively a runtime receipt")
        (into {} (for [role [:coarse :fine]
                       :let [result (double-array 16) value (get-in fields [role :value])]]
                   (do (gpu/download-range! session (outputs value) result {:elements 16})
                       [role (vec result)])))))))))

(defn- check-cycles! [target]
  (let [initial (#'oracle/initial-state)
        expected-first (#'oracle/reference-cycle (:coarse initial) (:fine initial) 0.001 false)
        actual-first (run-cycle! target initial)
        ;; run-cycle! closes every old session before this new prepared plan and context.
        ;; Readback arrays are copied startup inputs, not a durable manifest publication.
        actual-second (run-cycle! target (assoc actual-first :step 1))
        expected-second (#'oracle/reference-cycle (:coarse expected-first) (:fine expected-first) 0.001 false)
        mass (#'oracle/composite-mass (:coarse initial) (:fine initial))]
    (doseq [[expected actual] [[expected-first actual-first] [expected-second actual-second]]]
      (doseq [role [:coarse :fine]]
        (is (every? #(< (Math/abs (double %)) 1.0e-11)
                    (map - (role expected) (role actual)))
            (str target " independent coordinate oracle: " role)))
      (is (< (Math/abs (- mass (#'oracle/composite-mass (:coarse actual) (:fine actual)))) 1.0e-11))
      (is (= (:coarse actual) (#'oracle/average-down (:coarse actual) (:fine actual)))))))

(deftest certified-cycles-on-local-distributed-devices
  (if @opencl/opencl-fp64-available?
    (check-cycles! :ocl:0)
    (opencl/opencl-skip! "certified distributed temporal AMR cycles" :fp64))
  (if @ze/gpu-available?
    (let [caps ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
      (if (:fp64? caps)
        (check-cycles! :ze:0)
        (ze/gpu-capability-skip! "certified distributed temporal AMR cycles" :fp64? caps)))
    (ze/gpu-skip! "certified distributed temporal AMR cycles")))
