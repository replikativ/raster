(ns raster.gpu.scalar-helper-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.scalar-helpers :as helpers]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.device-probe :as ocl]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]))

(deftest cast-tail-helper-matches-source-through-both-local-backends
  (let [input (float-array [0.25 -2.5 8.0])
        expected (vec (helpers/map-cast-helper input 3))]
    (doseq [[target available? skip!] [[:ocl:0 ocl/opencl-available? ocl/opencl-skip!]
                                      [:ze:0 ze/gpu-available? ze/gpu-skip!]]]
      (if-not @available?
        (skip! (str "bare scalar helper on " target))
        (let [live (compiled/compile #'helpers/map-cast-helper [input 3]
                                     {:compiler :equation-first :dtype :float :target target})]
          (try
            (dotimes [_ 2]
              (let [outputs (live {})]
                (is (= 1 (count outputs)))
                (is (= expected (vec (value/->host (first (vals outputs))))) (str target))))
            (finally (compiled/close! live))))))))
