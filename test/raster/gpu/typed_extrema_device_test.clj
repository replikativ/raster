(ns raster.gpu.typed-extrema-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.extrema :as extrema]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as device]
            [raster.gpu.value :as value]))

(deftest subgroup-extrema-preserve-nans-and-zero-ties
  (if-not @device/opencl-subgroups-available?
    (device/opencl-skip! "source-semantic subgroup extrema" :subgroups)
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          resolve! #(ns-resolve ocl %)
          descriptor (hardware/descriptor-for :ocl:0)
          width (hardware/preferred-subgroup-size descriptor)]
      (assert (contains? (set (hardware/supported-subgroup-sizes descriptor)) width)
              "subgroup device oracle requires an authoritative supported width")
      (doseq [[_v dtype array-fn bits operator] extrema/cases
              :when (or (= :float dtype) @device/opencl-fp64-available?
                        (do (device/opencl-skip! "source-semantic Double subgroup extrema" :fp64)
                            false))
              :let [kernel-body (fixtures/floating-extrema-collective-body dtype operator width)
                    kernel-name (str "source_subgroup_" (name operator) "_" (name dtype))
                    compiled (artifact/make
                              {:kernel-name kernel-name :target :opencl-c
                               :source (emit/emit-scalar-kernel kernel-name kernel-body
                                         {:parameter-names {'x "rstr_x" 'out "rstr_out"}})
                               :abi [(abi/slot 'x :input dtype :c-name "rstr_x")
                                     (abi/slot 'out :output dtype :c-name "rstr_out")]
                               :arguments '[x out] :launch (:launch kernel-body)
                               :effects {:kind :test-extremum}
                               :provenance {:kernel-body (:id kernel-body)}})
                    output ((resolve! 'make-buffer) 1 dtype)]]
        (try
          ((resolve! 'register-kernel!) kernel-name compiled)
          (doseq [operands [(repeat width -0.0) (repeat width 0.0)
                            (take width (cycle [0.0 -0.0]))
                            (cons Float/NaN (repeat (dec width) 1.0))
                            (concat (repeat (dec width) 1.0) [Float/NaN])
                            (repeat width Float/NaN)
                            (range width)
                            (take width (cycle [Double/NEGATIVE_INFINITY Double/POSITIVE_INFINITY]))]
                  :let [input ((resolve! 'buffer-of-array) (array-fn operands) dtype)
                        combine (case [dtype operator]
                                  [:float :min] #(Math/min (unchecked-float %1) (unchecked-float %2))
                                  [:float :max] #(Math/max (unchecked-float %1) (unchecked-float %2))
                                  [:double :min] #(Math/min (double %1) (double %2))
                                  [:double :max] #(Math/max (double %1) (double %2)))
                        expected (reduce combine operands)]]
            (try
              ((resolve! 'launch-registered-bound!)
               ((resolve! 'bind-kernel-call) (call/make compiled [input output])))
              (is (extrema/same-result? bits
                                       ((resolve! 'buffer->array) output) [expected])
                  (str operator " operands=" operands))
              (finally ((resolve! 'free-buffer!) input))))
          (finally ((resolve! 'free-buffer!) output)))))))

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
