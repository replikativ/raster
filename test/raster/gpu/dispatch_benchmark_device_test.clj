(ns raster.gpu.dispatch-benchmark-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-benchmark :as benchmark]
            [raster.gpu.tuning-cache :as cache])
  (:import [java.nio.file Files]))

(def ^:private opencl-available?
  (delay
    (try
      (require 'raster.gpu.ocl-runtime)
      ((resolve 'raster.gpu.ocl-runtime/init!))
      true
      (catch Throwable _ false))))

(def ^:private abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'n :scalar :int :role :shape)])

(defn- affine-artifact
  [kernel-name strategy workgroup]
  (artifact/make
   {:kernel-name kernel-name
    :source
    (str "__kernel void " kernel-name
         "(__global const float* x, __global float* out, int n) {\n"
         "  for (int i = get_global_id(0); i < n; i += get_global_size(0))\n"
         "    out[i] = 2.0f * x[i] + 1.0f;\n"
         "}\n")
    :abi abi
    :arguments '[x out n]
    :launch (launch/spec {:workgroup-size [workgroup]
                          :group-count [(launch/ceil-div 'n workgroup)]})
    :effects {:kind :elementwise-map :reads ['x] :writes ['out]}
    :attributes {:strategy strategy}}))

(def ^:private dispatch
  (kdispatch/make
   {:id "device-affine-dispatch"
    :alternatives [(affine-artifact "device_affine_wg64" :wg64 64)
                   (affine-artifact "device_affine_wg128" :wg128 128)]
    :default-strategy :wg64
    :selector {:kind :runtime-scalar-threshold :argument 'n :threshold 512
               :at-least :wg128 :otherwise :wg64}}))

(defn- temporary-cache-root
  []
  (.toFile (Files/createTempDirectory "raster-device-dispatch-benchmark-"
                                      (make-array java.nio.file.attribute.FileAttribute 0))))

(defn- tune-affine!
  [device-id]
  (let [capacity 1024
        input (float-array (map #(float (* 0.25 %)) (range capacity)))]
    (binding [cache/*cache-root* (temporary-cache-root)]
      (gpu/with-gpu-session*
        device-id
        (fn [session]
          (gpu/alloc! session {:x [:float capacity input]
                               :out [:float capacity nil]})
          (benchmark/tune-dispatch!
           session dispatch (hardware/descriptor-for device-id) [127 capacity]
           (fn [n]
             {:arguments [:x :out {:type :int :value n}]
              :validate!
              (fn [_]
                (let [actual ^floats (gpu/download session :out)
                      max-error
                      (reduce max 0.0
                              (map (fn [index]
                                     (Math/abs
                                      (- (double (aget actual index))
                                         (+ (* 2.0 (double (aget input index))) 1.0))))
                                   (range n)))]
                  {:passed? (< max-error 1.0e-6)
                   :oracle-hash (str "affine-f32-v1-" n)
                   :max-error max-error}))
              :measurement {:warmup-iterations 0 :budget-ms 1
                            :min-samples 3 :max-samples 5 :cv-threshold 100.0}})
           :numerical-mode {:input :f32 :accumulate :f32 :output :f32}
           :layout {:input :contiguous :output :contiguous}
           :force? true))))))

(defn- assert-device-tuning!
  [device-id]
  (let [result (tune-affine! device-id)]
    (is (= 4 (count (:measurements result))))
    (is (every? #(true? (get-in % [:validation :passed?])) (:measurements result)))
    (is (= :runtime-scalar-ranges (get-in result [:selector :kind])))))

(deftest level-zero-validates-and-measures-emitted-dispatch-alternatives
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "validated KernelDispatch benchmark on Level Zero")
    (assert-device-tuning! :ze:0)))

(deftest opencl-validates-and-measures-emitted-dispatch-alternatives
  (if-not @opencl-available?
    (is true "OpenCL device unavailable")
    (assert-device-tuning! :ocl:0)))
