(ns raster.compiler.passes.parallel.outer-product-device-test
  "Execute the routed zero-reduction KernelBody with exact, differently sized input buffers."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as device-probe]))

(deftest opencl-outer-product-exact-input-capacities
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "routed outer product")
    (doseq [[m n] [[4 3] [17 19]]]
      (let [form (list 'raster.par/contract 'C [['i m] ['j n]] []
                       '(* (aget a i) (aget b j)))
            routed (route/route-contraction form :dtype :float)
            artifact (emit/generate-contraction-kernel-artifact (:kernel-body routed))
            a (doto (float-array (map #(- (float %) 2.0) (range m)))
                (aset-float 0 -0.0))
            b (float-array (map #(* 0.5 (float %)) (range n)))]
        (is (= :kernel-body (:emission-route routed)))
        (gpu/with-gpu-session [session :ocl:0]
          (gpu/alloc! session {:a [:float m a] :b [:float n b] :c [:float (* m n) nil]})
          (let [handle (gpu/bind-kernel-executable!
                        session :outer-product artifact
                        [:a :b :c {:type :int :value (* m n)}])]
            (try
              (gpu/run-kernel-graph! session handle)
              ;; An artificial fold from +0 would erase negative zero. This is a map, not
              ;; a length-one reduction, so compare the actual FP32 product bits.
              (is (= (mapv #(Float/floatToRawIntBits (float %))
                           (for [x a y b] (float (* x y))))
                     (mapv #(Float/floatToRawIntBits (float %))
                           (gpu/download session :c))))
              (finally (gpu/release-kernel-graph! session handle)))))))))
