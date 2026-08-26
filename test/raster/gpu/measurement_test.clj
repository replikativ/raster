(ns raster.gpu.measurement-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]))

(deftest summarizes-device-samples
  (let [m (measurement/summarize [100.0 200.0 300.0 400.0]
                                 :cv-threshold 1.0
                                 :warmup-iterations 2
                                 :budget-ms 25
                                 :timing-source :synthetic
                                 :hashes {:artifact "abc"})]
    (is (measurement/measurement? m))
    (is (= 100.0 (:min-ns m)))
    (is (= 200.0 (:median-ns m)) "nearest-rank p50")
    (is (= 300.0 (:p75-ns m)))
    (is (= 250.0 (:mean-ns m)))
    (is (:stationary? m))
    (is (= 4 (:n m)))
    (is (= {:artifact "abc"} (:hashes m)))))

(deftest stationarity-is-explicit
  (is (:stationary? (measurement/summarize [100 101 99] :cv-threshold 0.02)))
  (is (false? (:stationary? (measurement/summarize [1 100 1] :cv-threshold 0.02)))))

(deftest bounded-device-sampling
  (let [calls (atom 0)
        flushes (atom 0)
        m (measurement/measure! #(do (swap! calls inc) 1000000.0)
                                :warmup-iterations 2
                                :budget-ms 4
                                :min-samples 3
                                :max-samples 10
                                :flush-fn #(swap! flushes inc)
                                :timing-source :synthetic)]
    (testing "warmup + five probes are excluded from the bounded reported sample set"
      (is (= 4 (:n m)))
      (is (= (+ 2 5 4) @calls))
      (is (= 4 @flushes)))
    (is (= :synthetic (:timing-source m)))
    (is (= 1000000.0 (:min-ns m)))))

(deftest rejects-invalid-measurements
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"at least one"
                        (measurement/summarize [])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"finite"
                        (measurement/summarize [Double/POSITIVE_INFINITY])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive"
                        (measurement/measure! (constantly 1.0) :budget-ms 0))))

(deftest core-measures-runtime-graphs-with-device-events
  (let [replays (atom 0)
        reads (atom 0)
        session (atom {:device-id :probe})]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "replay-graph!" (fn [_] (swap! replays inc))
           "read-graph-timestamps!" (fn [_]
                                      (swap! reads inc)
                                      {:wall-ms 0.001})
           (throw (ex-info "unexpected runtime function" {:function function-name}))))}
      #(let [m (gpu/measure-graph! session {:profile? true}
                                   :warmup-iterations 1
                                   :budget-ms 3
                                   :min-samples 3
                                   :max-samples 3)]
         (is (= :device-event (:timing-source m)))
         (is (= 1000.0 (:min-ns m)))
         (is (= (+ 1 5 3) @replays))
         (is (= @replays @reads))))))

(deftest stateful-linked-measurement-requires-restore
  (let [executable
        (link/map->LinkedExecutable
         {:plan {:nodes {:cache {:role :state}}}
          :pending-inputs (atom #{}) :closed? (atom false)})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"require :before-sample!"
                          (link/measure! executable :budget-ms 1)))))
