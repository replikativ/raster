(ns raster.gpu.scan-schedule-device-test
  "Shared scan schedule execution, with independent CPU oracles and resident replay."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]))

(defn- check-scans! [device]
  (let [n 1025]
    (gpu/with-gpu-session [session device]
      (gpu/alloc! session {:input [:float n nil] :output [:float (inc n) nil]})
      (doseq [mode [:inclusive :exclusive]]
        (let [op (if (= mode :inclusive) 'raster.par/scan 'raster.par/scan-exclusive)
              form (list 'let* ['result (list op 'out 'acc 0.0 'i 'n 'float
                                             '(+ acc (clojure.core/aget values i)))] 'result)
              program (:program (route/attempt form :float {'values :float 'out :float}
                                               {:dtype :float :scalar-types {'n :int}}))
              graph (emit/generate-scan-kernel-graph
                     (:kernel-graph (lower/lower-typed-scan (get-in program [:equations 0 :algorithm]) nil :dtype :float
                                                          :array-types {'values :float 'out :float})))
              handle (gpu/bind-kernel-graph! session mode graph
                                            {'values :input 'out :output}
                                            {'n {:type :int :value n}})]
          (try
            (doseq [replay [0 1]]
              (let [input (float-array (map #(if (zero? replay) 1.0
                                               (* 0.25 (- (mod % 11) 5))) (range n)))
                    sums (reductions + 0.0 input)
                    expected (mapv float (if (= mode :inclusive) (rest sums) sums))]
                (gpu/upload! session :input input)
                (gpu/run-kernel-graph! session handle)
                (is (= expected (vec (take (count expected) (gpu/download session :output))))
                    (str device " " mode " full output, replay " replay))))
            (finally (gpu/release-kernel-graph! session handle))))))))

(deftest shared-scan-schedules-on-opencl
  (if @opencl/opencl-available?
    (check-scans! :ocl:0)
    (opencl/opencl-skip! "shared scan schedules")))

(deftest shared-scan-schedules-on-level-zero
  (if @ze/gpu-available?
    (check-scans! :ze:0)
    (ze/gpu-skip! "shared scan schedules")))
