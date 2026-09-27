(ns raster.gpu.cuda-driver-probe-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver-probe :as driver]))

(deftest no-library-does-not-claim-a-device
  (with-redefs [driver/available? (constantly false)]
    (is (= {:status :unavailable :devices []} (driver/probe)))))

(deftest native-probe-classifies-physical-device-state
  (let [{:keys [status devices init-code driver-api-version]} (driver/probe)]
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
