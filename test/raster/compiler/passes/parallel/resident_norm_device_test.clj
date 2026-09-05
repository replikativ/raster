(ns raster.compiler.passes.parallel.resident-norm-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.nn :as nn]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl-probe]))

(deftest public-rmsnorm-has-a-resident-reduction-and-map
  (doseq [target [:ocl:0 :ze:0]]
    (let [descriptor (pipeline/compile-gpu-program #'nn/rms-norm-1row! target :dtype :float)
          reduction (:artifact (first (:steps descriptor)))]
      (is (= [:executable :map-void] (mapv :convention (:steps descriptor))))
      (is (graph/kernel-graph? reduction))
      (is (= 2 (count (:nodes reduction))))
      (is (= 1 (count (:allocs descriptor))) "only the completed scalar crosses stages"))))

(defn- run-norm! [target]
  (let [descriptor (pipeline/compile-gpu-program #'nn/rms-norm-1row! target :dtype :float)]
    (gpu/with-gpu-session [session target]
      (doseq [width [1 17 513]]
        (let [x (float-array (map #(float (/ (- (mod % 11) 5) 7.0)) (range width)))
              weights (float-array (map #(float (/ (inc (mod % 7)) 9.0)) (range width)))
              output (float-array width)
              eps 0.0001
              gain 1.0
              arguments [x weights output (long width) eps gain]
              inverse (/ 1.0 (Math/sqrt (+ eps (/ (reduce + (map #(* (double %) %) x)) width))))
              expected (mapv #(* %1 inverse (+ gain %2)) x weights)
              program (fixture/instantiate! session descriptor arguments
                                            {'x :input 'weight :input 'out :output})]
          (try
            (dotimes [_ 2]
              (let [actual (get (fixture/run! program arguments) 'out)]
                (is (= width (count actual)))
                (doseq [[want got] (map vector expected actual)]
                  (is (< (Math/abs (- want got)) 0.00001)))))
            (finally (fixture/close! program))))))))

(deftest opencl-resident-rmsnorm-matches-reference
  (if @opencl-probe/opencl-available?
    (run-norm! :ocl:0)
    (opencl-probe/opencl-skip! "resident RMSNorm result transform")))

(deftest level-zero-resident-rmsnorm-matches-reference
  (if @gpu-probe/gpu-available?
    (run-norm! :ze:0)
    (gpu-probe/gpu-skip! "resident RMSNorm result transform")))
