(ns raster.gpu.typed-extrema-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.extrema :as extrema]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as device]
            [raster.gpu.value :as value]))

(deftest equation-first-extrema-agree-with-jvm-on-device
  (doseq [[v dtype array-fn bits _op] extrema/cases]
    (if-not (and @device/opencl-available?
                 (or (= :float dtype) @device/opencl-fp64-available?))
      (device/opencl-skip! (str "typed source extrema " dtype)
                          (when (= :double dtype) :fp64))
      (let [n 17
            artifact (compiled/compile v [(array-fn n) (array-fn n) (array-fn n) n]
                                       {:target :ocl:0 :compiler :equation-first :dtype dtype
                                        :outputs '[y] :on-non-resident :throw})]
        (try
          (doseq [shift [0 4 8]
                  :let [pairs (take n (drop shift (cycle extrema/operand-pairs)))
                        x (array-fn (map first pairs))
                        z (array-fn (map second pairs))
                        expected (array-fn n)]]
            (@v x z expected n)
            (let [actual (value/->host (first (vals (artifact {:x x :z z}))))]
              (is (extrema/same-result? bits actual expected)
                  (str (:name (meta v)) " shift=" shift))))
          (finally (compiled/close! artifact)))))))
