(ns raster.compiler.passes.parallel.device-scalar-device-test
  "Actual execution of the inactive ScalarIR scheduling substrate, not public dispatch activation."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.device-scalar-substrate-test :as fixture]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.backend.gpu.parallel-program-c-family :as target]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn- check-device!
  ([device dialect] (check-device! device dialect (#'fixture/scalar-program)))
  ([device dialect algorithm]
  (let [operation (first (lower/lower-typed-scalar
                          algorithm device :target-descriptor @#'fixture/target-descriptor))
        emitted (target/emit-program (#'fixture/scheduled-program algorithm operation)
                                     {:target-dialect dialect})
        graph (get-in emitted [:program :equations 0 :operations 0 :graph])
        bias 0.125]
    (gpu/with-gpu-session [session device]
      (gpu/alloc! session {:input [:float 1 nil] :loss [:double 1 nil]})
      (let [handle (gpu/bind-kernel-graph! session :scalar-substrate graph
                                            {'out :input 'loss :loss}
                                            {'bias {:type :double :value bias}})]
        (try
          (doseq [sample [1.234567890123 -2.345678901234 16777217.0]]
            (let [input (float-array [sample])
                  expected (+ (double (float sample)) bias)]
              (gpu/upload! session :input input)
              (gpu/run-kernel-graph! session handle)
              (let [actual (first (gpu/download session :loss))]
                (is (Double/isFinite (double actual)))
                (is (= (Double/doubleToRawLongBits expected)
                       (Double/doubleToRawLongBits (double actual)))
                    (str device " preserves the Float checkpoint before Double widening")))))
          (finally (gpu/release-kernel-graph! session handle))))))))

(deftest resident-rank-zero-scalar-kernel-replays-changed-inputs-on-local-backends
  (if @opencl/opencl-fp64-available?
    (check-device! :ocl:0 :opencl-portable)
    (opencl/opencl-skip! "typed scalar substrate" :fp64))
  (if @ze/gpu-available?
    (let [caps ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
      (if (:fp64? caps)
        (check-device! :ze:0 :opencl-intel)
        (ze/gpu-capability-skip! "typed scalar substrate" :fp64? caps)))
    (ze/gpu-skip! "typed scalar substrate")))

(deftest required-scalar-payload-transport-executes-on-local-backends
  (let [algorithm (#'fixture/required-resident-payload-program)]
    (if @opencl/opencl-fp64-available?
      (check-device! :ocl:0 :opencl-portable algorithm)
      (opencl/opencl-skip! "required scalar payload" :fp64))
    (if @ze/gpu-available?
      (let [caps ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
        (if (:fp64? caps)
          (check-device! :ze:0 :opencl-intel algorithm)
          (ze/gpu-capability-skip! "required scalar payload" :fp64? caps)))
      (ze/gpu-skip! "required scalar payload"))))
