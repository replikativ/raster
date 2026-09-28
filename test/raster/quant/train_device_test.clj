(ns raster.quant.train-device-test
  "Q8 forward and pullback must use the same equation-first GPU path as adapter training."
  (:require [clojure.test :refer [deftest is]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.quant.train :as qt]))

(deftest q8-forward-and-backward-lower-through-typed-soac
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "equation-first Q8 forward and pullback")
    (let [in 32 out 2 rows 2
          weights (float-array (map #(float (/ (inc (mod % 9)) 64.0))
                                    (range (* in out))))
          {:keys [codes scales]} (qt/q8-quantize weights out in)
          x (float-array (map #(float (/ (inc (mod % 7)) 16.0))
                              (range (* rows in))))
          dy (float-array [0.2 -0.3 0.4 0.1])]
      (doseq [[kernel input expected input-key]
              [[#'qt/qlinear-q8 x (qt/qlinear-q8 x codes scales rows in out) 'x]
               [#'qt/qlinear-q8-dx dy (qt/qlinear-q8-dx dy codes scales rows in out) 'dy]]]
        (let [prepared (compiled/lower kernel
                                       [input codes scales (long rows) (long in) (long out)]
                                       {:compiler :equation-first :target :ze:0 :dtype :float
                                        :constants [input-key 'codes 'scales]
                                        :on-non-resident :throw})
              artifact (compiled/instantiate! prepared)]
          (try
            (let [actual (value/->host (:result (artifact {})))]
              (is (= (alength ^floats expected) (alength ^floats actual)))
              (is (every? true?
                          (map #(<= (Math/abs (- (double %1) (double %2))) 1.0e-5)
                               expected actual))))
            (finally (compiled/close! artifact))))))))
