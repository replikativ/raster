(ns raster.gpu.device-probe-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.device-probe :as probe]))

(defn- api
  [& {:keys [load! query-devices init! selected-device-info]
      :or {load! (constantly true)
           query-devices (constantly [{:name "enumerated"}])
           init! (constantly true)
           selected-device-info (constantly {:name "selected"})}}]
  {:load! load!
   :query-devices query-devices
   :init! init!
   :selected-device-info selected-device-info})

(deftest opencl-probe-distinguishes-absence-from-breakage
  (is (= :load-failed
         (:status
          (probe/probe-opencl-with
           (api :load! #(throw (Exception. "load")))))))
  (is (= :probe-error
         (:status
          (probe/probe-opencl-with
           (api :query-devices #(throw (Exception. "query")))))))
  (let [initialized? (atom false)
        result (probe/probe-opencl-with
                (api :query-devices (constantly [])
                     :init! #(reset! initialized? true)))]
    (is (= :no-device (:status result)))
    (is (false? @initialized?) "absence must not initialize a context")))

(deftest capability-gates-use-the-exact-selected-device
  (let [selected {:name "selected-without-fp16" :extensions "cl_khr_fp64"}
        result (probe/probe-opencl-with
                (api :query-devices
                     (constantly [{:name "other" :extensions "cl_khr_fp16"} selected])
                     :selected-device-info (constantly selected)))]
    (is (= :available (:status result)))
    (is (= selected (:device result)))
    (is (false? (probe/capability-supported? (:device result) :fp16))))
  (is (probe/capability-supported?
       {:extensions "cl_khr_fp64 cl_khr_fp16 cl_khr_subgroups"} :fp16)))

(defn- capture-skip
  [status expected?]
  (let [events (atom [])
        skip-log (var-get (ns-resolve 'raster.gpu.device-probe 'opencl-skip-log))
        prior-log @skip-log]
    (try
      (binding [probe/*expect-opencl?* expected?]
        (with-redefs [probe/opencl-status (delay status)
                      clojure.test/report (fn [event]
                                            (swap! events conj (:type event)))]
          (probe/opencl-skip! "probe-test")))
      @events
      (finally
        (reset! skip-log prior-log)))))

(deftest expected-opencl-mode-turns-absence-and-probe-errors-red
  (testing "ordinary CPU-only CI records a visible passing marker"
    (is (= [:pass] (capture-skip {:status :no-device} false))))
  (testing "an expected OpenCL lane fails on absence"
    (is (= [:fail] (capture-skip {:status :no-device} true))))
  (testing "an expected OpenCL lane fails on driver/query errors"
    (is (= [:fail]
           (capture-skip {:status :probe-error :error (Exception. "driver")} true)))))
