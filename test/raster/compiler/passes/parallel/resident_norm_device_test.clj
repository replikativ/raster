(ns raster.compiler.passes.parallel.resident-norm-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.equation-first :as equation-first]
            [raster.nn :as numerical-nn]
            [raster.dl.nn :as nn]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
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

(deftest public-softmax-backward-composes-resident-reduction-and-map
  (if @opencl-probe/opencl-available?
    (let [compilation (equation-first/compile
                       #'numerical-nn/softmax-backward {:target :ocl:0 :dtype :double})]
      (doseq [width [1 17 513]]
        (let [dy (double-array (map #(/ (- (mod % 11) 5) 7.0) (range width)))
              weights (mapv #(double (inc (mod % 7))) (range width))
              denominator (reduce + weights)
              s (double-array (map #(/ % denominator) weights))
              dot (reduce + (map * (vec s) (vec dy)))
              expected (mapv #(* %1 (- %2 dot)) (vec s) (vec dy))
              plan (equation-first/lower compilation [dy s])]
          (is (= 0 (get-in plan [:attributes :driver-allocations])))
          (with-open [live (link/instantiate! plan)]
            (dotimes [_ 2]
              (link/run! live)
              (let [actual (vec (link/download live (first (:outputs plan))))]
                (is (= width (count actual)))
                (is (every? #(< (Math/abs (double %)) 1.0e-10)
                            (map - expected actual)))))))))
    (opencl-probe/opencl-skip! "public resident softmax backward composition")))

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
