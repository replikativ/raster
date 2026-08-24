(ns raster.gpu.kernel-graph-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]))

(def ^:private ocl-available?
  (delay
    (try
      (require 'raster.gpu.ocl-runtime)
      ((resolve 'raster.gpu.ocl-runtime/init!))
      true
      (catch Throwable _ false))))

(defn- emitted-graph []
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              103)
        operations (lower/lower-scan node nil :dtype :float)]
    (emit/generate-scan-kernel-graph
     (lower/scan-kernel-graph
      node operations {:array-types {'values :float 'out :float}}))))

(defn- run-inclusive-scan
  [device-id]
  (let [n 1025
        graph (emitted-graph)
        input (float-array n 1.0)]
    (gpu/with-gpu-session [sess device-id]
      (gpu/alloc! sess {:values [:float n input]
                        :out [:float n nil]})
      (let [handle (gpu/bind-kernel-graph!
                    sess :inclusive-scan graph {'values :values 'out :out}
                    {'n {:type :int :value n}})]
        (try
          (gpu/run-kernel-graph! sess handle)
          (gpu/download sess :out)
          (finally
            (gpu/release-kernel-graph! sess handle)))))))

(defn- assert-prefix!
  [device-id]
  (let [result (run-inclusive-scan device-id)]
    (is (= 1025.0 (double (aget ^floats result 1024))))
    (is (every? true?
                (map-indexed (fn [i value] (= (double (inc i)) (double value))) result)))))

(deftest level-zero-kernel-graph-inclusive-scan
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "Level Zero KernelGraph inclusive scan")
    (assert-prefix! :ze:0)))

(deftest opencl-kernel-graph-inclusive-scan
  (if-not @ocl-available?
    (is true "OpenCL device unavailable")
    (assert-prefix! :ocl:0)))
