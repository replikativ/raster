(ns raster-strict-gemm-test
  "Host-only checks for the opt-in comparison harness; no accelerator required."
  (:require [clojure.test :refer [deftest is]]
            [raster-strict-gemm :as gemm]
            [raster.runtime.hardware :as hardware]))

(deftest output-oracle-rejects-nonfinite-and-missing-values
  (let [check-output! #'gemm/check-output!
        expected (float-array [1.0 2.0])]
    (is (zero? (check-output! expected [1.0 2.0])))
    (is (pos? (check-output! expected [1.0 2.00001])))
    (doseq [actual [[1.0 Float/NaN] [Float/NaN 2.0]
                    [1.0 Float/POSITIVE_INFINITY] [1.0 Float/NEGATIVE_INFINITY]
                    [1.0] [1.0 3.0]]]
      (is (thrown? clojure.lang.ExceptionInfo (check-output! expected actual))))))

(deftest shape-validation-precedes-hardware-initialization
  (with-redefs [hardware/init! #(throw (AssertionError. "hardware must not initialize"))]
    (doseq [shape [nil [] [1 2] [1 2 3 4] [0 2 3] [-1 2 3]
                   [1.0 2 3] [4097 2 3] [4096 4096 4096]]]
      (is (thrown? clojure.lang.ExceptionInfo (gemm/benchmark! shape))))))

(deftest device-timing-never-falls-back-or-saturates
  (let [duration #'gemm/profile-duration-ns]
    (is (= 125000 (duration {:device-wall-ms 0.125} :device-wall-ms)))
    (doseq [value [nil 0.0 -1.0 Double/NaN Double/POSITIVE_INFINITY
                   Double/NEGATIVE_INFINITY 1.0e15 1.0e-10]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (duration {:device-wall-ms value} :device-wall-ms))))))
