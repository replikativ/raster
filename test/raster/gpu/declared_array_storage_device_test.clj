(ns raster.gpu.declared-array-storage-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.value :as value]))

(defn- run-case [target]
  (doseq [compiler [nil :equation-first]]
    (let [artifact (compiled/compile
                    #'storage/mixed-storage!
                    [(float-array 4) (double-array 4) (double-array 4) 4]
                    (cond-> (merge storage/policy
                                   {:target target :dtype :double :outputs '[out]})
                      compiler (assoc :compiler compiler)))]
      (try
        (doseq [weights [[1.25 -2.5 0.125 8.0] [16.0 32.0 -4.0 0.5]]]
          (let [x (float-array weights)
                state (double-array [0.0000000001 1.0 4.0 -2.0])
                expected (double-array 4)]
            (storage/mixed-storage! x state expected 4)
            (is (= (vec expected)
                   (vec (value/->host (:out (artifact {:weights x :state state}))))))))
        (finally (compiled/close! artifact))))))

(deftest level-zero-mixed-storage-public-compilers-match-jvm
  (if @gp/gpu-available?
    (run-case :ze:0)
    (gp/gpu-skip! "mixed array storage on Level Zero")))

(deftest opencl-mixed-storage-public-compilers-match-jvm
  (if @opencl/opencl-available?
    (run-case :ocl:0)
    (opencl/opencl-skip! "mixed array storage on OpenCL")))
