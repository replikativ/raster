(ns raster.gpu.symbolic-storage-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.symbolic-storage :as storage]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.device-probe :as ocl]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]))

(deftest symbolic-storage-maps-match-jvm-on-both-local-backends
  (doseq [[target available? skip!] [[:ocl:0 ocl/opencl-available? ocl/opencl-skip!]
                                    [:ze:0 ze/gpu-available? ze/gpu-skip!]]]
    (if-not @available?
      (skip! (str "symbolic storage traversal on " target))
      (doseq [[function args expected]
              [[#'storage/regrouped-map [(float-array [0.25 -2.5 8.0 1.0 2.0 3.0]) 1 2 3]
                [2.5 -3.0 18.0 4.0 6.0 8.0]]
               [#'storage/prefix-map [(float-array [0.25 -2.5 8.0 1.0 2.0 3.0]) 6 4]
                [0.25 -2.5 8.0 1.0]]]]
        (is (= expected (vec (apply function args))))
        (let [live (compiled/compile function args
                                     {:compiler :equation-first :dtype :float :target target})]
          (try
            (dotimes [_ 2]
              (let [outputs (live {})]
                (is (= 1 (count outputs)))
                (is (= expected (vec (value/->host (first (vals outputs))))))))
            (finally (compiled/close! live))))))))
