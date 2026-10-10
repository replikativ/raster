(ns raster.gpu.distributed-math-request-device-test
  "Actual selected math in a five-trip emitted loop; not fabric or throughput evidence."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.distributed-math-request-test :as fixture]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn- expected-value []
  (nth (iterate (fn [x]
                  (float (* 2.0 (double (float (Math/tanh (double (float (+ (double x) 0.25)))))))))
                (float 0)) 5))

(defn- check-device! [target]
  (let [plan (fixture/selected-plan target (hardware/descriptor-for target))
        options (assoc fixture/request :device-capacities {target 1048576}
                                       :include-graph-temporaries? true)]
    (with-open [owner (runtime/instantiate! plan options)]
      (is (runtime/original-executable? owner))
      (is (= fixture/request (:caller-options owner)))
      (is (pos? (get-in owner [:allocation-budgets target :graph-temporary-bytes])))
      (is (identical? owner (runtime/run! owner)))
      (let [actual (float-array 64)]
        (runtime/with-output-values!
         owner (fn [outputs]
                 (gpu/download-range! (get (:sessions owner) target)
                                      (get-in outputs [:advance :output]) actual {:elements 64})))
        (is (every? #(<= (Math/abs (- (double %) (double (expected-value)))) 1.0e-6) actual))
        (is (= :complete @(:state owner)))))))

(deftest selected-loop-on-opencl
  (if @opencl/opencl-available?
    (check-device! :ocl:0)
    (opencl/opencl-skip! "distributed independently selected scalar math")))

(deftest selected-loop-on-level-zero
  (if @ze/gpu-available?
    (check-device! :ze:0)
    (println "SKIP: distributed independently selected scalar math on Level Zero")))
