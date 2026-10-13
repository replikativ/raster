(ns raster.gpu.cuda-driver-probe-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver-probe :as driver])
  (:import [java.lang.foreign Arena ValueLayout]))

(def vendor-attribute-fixture
  {1 1024, 2 1024, 3 1024, 4 64, 8 49152, 10 32, 12 65536,
   16 40, 39 1536, 81 102400, 82 65536, 106 16})

(defn with-mocked-device-probe [errors f]
  (let [queried (atom [])]
    (with-open [arena (Arena/ofConfined)]
      (with-redefs-fn
        {#'driver/handles (delay {:device :device :name :name :capability :capability
                                 :attribute :attribute :total-memory :total-memory})
         #'driver/invoke
         (fn [operation & arguments]
           (case operation
             :device (do (.set (first arguments) ValueLayout/JAVA_INT (long 0) (int 0)) 0)
             :name (do (.setString (first arguments) (long 0) "fixture GPU") 0)
             :total-memory (do (.set (first arguments) ValueLayout/JAVA_LONG (long 0) (long 8222998528)) 0)
             :capability (do (.set (first arguments) ValueLayout/JAVA_INT (long 0) (int 8))
                             (.set (second arguments) ValueLayout/JAVA_INT (long 0) (int 6)) 0)
             :attribute (let [[pointer attribute device] arguments
                              code (get errors attribute 0)]
                          (swap! queried conj [attribute device])
                          (when (zero? code)
                            (.set pointer ValueLayout/JAVA_INT (long 0)
                                  (int (get vendor-attribute-fixture attribute))))
                          code)))}
        (fn [] (f (fn [] (#'driver/device-info arena 0)) queried))))))

(deftest device-probe-reads-distinct-vendor-attribute-limits
  (with-mocked-device-probe {}
    (fn [probe queried]
      (is (= {:sm-count 40 :warp-size 32 :max-threads-per-block 1024
              :max-threads-per-sm 1536 :max-blocks-per-sm 16
              :registers-per-block 65536 :registers-per-sm 65536
              :shared-memory-per-block 49152 :shared-memory-per-sm 102400
              :global-memory-bytes 8222998528 :max-workgroup-dims [1024 1024 64]}
             (:attributes (probe))))
      (is (= (set (keys vendor-attribute-fixture)) (set (map first @queried))))
      (is (every? #(zero? (second %)) @queried)))))

(deftest unavailable-attribute-is-absent-and-explicitly-diagnosed
  (with-mocked-device-probe {39 1}
    (fn [probe _]
      (let [info (probe)]
        (is (not (contains? (:attributes info) :max-threads-per-sm)))
        (is (= 40 (get-in info [:attributes :sm-count])))
        (is (= [{:reason :cuda-driver-attribute-unavailable :attribute 39 :error-code 1}]
               (get-in info [:attribute-errors :max-threads-per-sm])))))))

(deftest unexpected-attribute-driver-error-remains-a-typed-failure
  (with-mocked-device-probe {39 999}
    (fn [probe _]
      (let [error (try (probe) nil (catch clojure.lang.ExceptionInfo error error))]
        (is (= {:reason :cuda-driver-error :operation :device-attribute
                :attribute 39 :error-code 999}
               (ex-data error)))))))

(deftest partially-unavailable-dimensions-do-not-become-a-complete-launch-limit
  (with-mocked-device-probe {3 1}
    (fn [probe _]
      (let [info (probe)]
        (is (not (contains? (:attributes info) :max-workgroup-dims)))
        (is (= 1024 (get-in info [:attributes :max-threads-per-block])))
        (is (= [{:reason :cuda-driver-attribute-unavailable :attribute 3 :error-code 1}]
               (get-in info [:attribute-errors :max-workgroup-dims])))))))

(deftest no-library-does-not-claim-a-device
  (with-redefs [driver/available? (constantly false)]
    (is (= {:status :unavailable :devices []} (driver/probe)))))

(deftest initialization-failures-do-not-query-or-claim-devices
  (doseq [[code status] [[100 :no-device] [999 :driver-error]]]
    (with-redefs-fn {#'driver/available? (constantly true)
                    #'driver/handles (delay {:init :init})
                    #'driver/invoke (fn [operation & _]
                                      (is (= :init operation))
                                      code)}
      (fn []
        (is (= {:status status :devices [] :init-code code} (driver/probe)))))))

(deftest native-probe-classifies-physical-device-state
  (let [{:keys [status devices init-code driver-api-version]} (driver/probe)]
    (when (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
      (is (= :ready status) "a CUDA device is required for this test run"))
    (is (vector? devices))
    (case status
      :unavailable (is (empty? devices))
      :no-device (is (empty? devices))
      :driver-error (do (is (empty? devices))
                        (is (pos? init-code)))
      :ready (do (is (pos? (count devices)))
                 (is (pos? driver-api-version))
                 (is (every? (fn [{:keys [ordinal name compute-capability]}]
                               (and (nat-int? ordinal) (seq name)
                                    (= 2 (count compute-capability))
                                    (pos-int? (first compute-capability))
                                    (nat-int? (second compute-capability))))
                             devices)))
      (is false (str "unknown CUDA probe status: " status)))))
