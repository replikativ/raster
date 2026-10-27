(ns raster.compiler.passes.parallel.typed-soac-initialization-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-compile-fixtures :as fixtures]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.value :as value]
            [raster.nn :as nn]))

(deftest source-only-allocation-does-not-add-a-numerical-zero-kernel
  (doseq [[target available? skip!] [[:ocl:0 @opencl/opencl-available? opencl/opencl-skip!]
                                    [:ze:0 @gpu-probe/gpu-available? gpu-probe/gpu-skip!]]]
    (if-not available?
      (skip! "source-only dense-gradient allocation")
      (let [prepared (compiled/lower #'nn/dense-backward-dW
                                     [(float-array [1.0 2.0]) (float-array [3.0 4.0 5.0])]
                                     {:compiler :equation-first :target target :dtype :float})
            live (compiled/instantiate! prepared)]
        (try
          (is (some? (:ordered-invocation prepared)))
          (is (= 1 (count (compiled/ir prepared))))
          (doseq [[dy x expected] [[[1.0 2.0] [3.0 4.0 5.0] [3.0 4.0 5.0 6.0 8.0 10.0]]
                                   [[-1.0 3.0] [2.0 -4.0 1.0] [-2.0 4.0 -1.0 6.0 -12.0 3.0]]]]
            (is (= expected
                   (vec (value/->host (:result (live {:dy (float-array dy) :x (float-array x)})))))))
          (finally (compiled/close! live)))))))

(deftest public-fresh-scatter-reestablishes-zeros-on-every-replay
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "public typed allocation initialization")
    (doseq [[function values dimensions expected]
            [[#'fixtures/public-c-family-scatter [1 2 3] [3] [3.0 0.0 3.0]]
             [#'fixtures/public-c-family-strided-scatter [1 2 3 4 5 6] [3 2]
              [4.0 6.0 0.0 0.0 5.0 6.0]]]]
      (let [input (float-array values)
            prepared (compiled/lower function
                                     (into [input (int-array [0 0 2])] dimensions)
                                     {:target :ze:0 :dtype :float :on-non-resident :throw})
            live (compiled/instantiate! prepared {:profile? true})]
        (try
          (is (= [:map-void :map-void] (mapv :convention (compiled/ir prepared))))
          (doseq [sign [1.0 -1.0]]
            (dotimes [i (alength input)] (aset-float input i (* sign (nth values i))))
            (let [result (compiled/profile live)]
              (is (= (mapv #(* sign %) expected)
                     (vec (get-in result [:result :output]))))
              (is (= 2 (count (:profile result)))
                  "the zero map is part of every replay, not an allocation-time upload")))
          (finally (compiled/close! live)))))))
