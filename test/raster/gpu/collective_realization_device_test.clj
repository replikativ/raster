(ns raster.gpu.collective-realization-device-test
  "Co-located numerical all-reduce oracle, not a multi-host or fabric performance benchmark."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.distributed-refinement-test :as fixture]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.device-probe :as opencl]))

(defn- check-realization! [device]
  (doseq [n [2 3]]
    (let [plan (fixture/realized-plan n {:target-device device
                                        :target-descriptor (hardware/descriptor-for device)})
          expected (mapv #(float (+ (/ (* n (inc n)) 2) (* n 0.25 %))) (range 17))]
      (distributed/verify! (distributed/certify plan))
      (with-open [executable (runtime/instantiate! plan {:transport :resident-copy
                                                        :device-capacities {device 1048576}})]
        (runtime/run! executable)
        (let [outputs (runtime/output-values executable)
              session (get (:sessions executable) device)]
          (is (= n (count outputs)))
          (doseq [[_ values] outputs]
            (is (= 1 (count values)))
            (let [result (float-array 17)]
              (gpu/download-range! session (first (vals values)) result {:elements 17})
              (is (= expected (vec result))
                  "every participant receives the complete generated numerical reduction"))))))))

(deftest contribution-certified-all-reduce-on-colocated-opencl-workers
  (if @opencl/opencl-available?
    (check-realization! :ocl:0)
    (opencl/opencl-skip! "contribution-certified co-located all-reduce")))

(deftest contribution-certified-all-reduce-on-colocated-level-zero-workers
  (if @ze/gpu-available?
    (check-realization! :ze:0)
    (ze/gpu-skip! "contribution-certified co-located all-reduce")))
