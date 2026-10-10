(ns raster.gpu.distributed-cost-device-test
  "Native exact-context route admission plumbing, not calibrated production performance."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.distributed-plan :as plan]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.distributed-training-device-test :as fixture]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn- check-costs! [device]
  (let [p (#'fixture/training-plan device [1 3])
        options {:transport :host-staged :max-staging-bytes 16
                 :device-capacities {device 1048576}}
        profiles (mapv (fn [_]
                         (with-open [owner (runtime/instantiate! p options)]
                           (let [context (runtime/cost-context owner)
                                 profile (runtime/profile! owner)]
                             (is (= context (:route-cost-context profile)))
                             (is (= :distributed-runtime-owner
                                    (try (runtime/cost-context owner) nil
                                         (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
                             profile))) (range 3))]
    (with-open [owner (runtime/instantiate! p options)]
      (let [context (runtime/cost-context owner)
            ;; Permissive spread is intentional: test native admission plumbing without a
            ;; flaky timing assertion on this shared laptop. Not production stationarity.
            policy {:now-ms (System/currentTimeMillis) :max-age-ms 600000
                    :cv-threshold 2.0 :cold-warm :cold}
            result (plan/simulate p {:route-context context :profiles profiles :route-policy policy})
            transfers (filter #(= :transfer (:kind %)) (:steps p))]
        (is (seq transfers))
        (doseq [step transfers
                :let [cost (get-in result [:route-cost-evidence (:id step)])
                      durations (mapv (fn [profile]
                                        (:host-wall-ns (first (filter #(= (:id step) (:step %))
                                                                     (:steps profile))))) profiles)
                      median (nth (sort durations) 1)]]
          (is (= :empirical-host-step (:source cost)))
          (is (= median (:duration-ns cost)))
          (is (= (mapv double durations) (get-in cost [:measurement :samples-ns])))
          (is (= median (get-in result [:timeline (:id step) :duration-ns]))))
        (is (false? (:empirical-costs-certified? result)))
        (is (= :ready @(:state owner)) "cost projection must not execute the current owner")))))

(deftest measured-routes-on-local-opencl-and-level-zero
  (if @opencl/opencl-available?
    (check-costs! :ocl:0)
    (opencl/opencl-skip! "native empirical distributed route admission"))
  (if @ze/gpu-available?
    (check-costs! :ze:0)
    (ze/gpu-skip! "native empirical distributed route admission")))
