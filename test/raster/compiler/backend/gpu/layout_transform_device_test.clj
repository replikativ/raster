(ns raster.compiler.backend.gpu.layout-transform-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.passes.parallel.layout-transform-schedule :as schedule]
            [raster.gpu.device-probe :as probe]))

(defn- execute
  [kernel input initial-output scalars]
  (let [ocl (find-ns 'raster.gpu.ocl-runtime)
        runtime #(ns-resolve ocl %)
        parameters (:parameters kernel)
        names (zipmap (map :id parameters) ["x" "y" "rows" "cols"])
        compiled (artifact/make
                  {:kernel-name "layout_width_probe" :target :opencl-c
                   :source (emitter/emit-scalar-kernel
                            "layout_width_probe" kernel
                            {:target-dialect :opencl-portable :parameter-names names})
                   :abi (body-abi/project-contracts
                         (mapv #(abi/slot (:id %) (:kind %) (:dtype %)
                                          :c-name (get names (:id %))) parameters)
                         kernel)
                   :arguments (mapv :id parameters)
                   :launch (:launch kernel) :effects {:kind :layout-transform}})
        x ((runtime 'buffer-of-array) input :float)]
    (try
      (let [y ((runtime 'buffer-of-array) initial-output :float)]
        (try
          ((runtime 'register-kernel!) (:kernel-name compiled) compiled)
          (let [prepared ((runtime 'bind-kernel-call)
                          (call/make compiled
                                     (into [x y]
                                           (map (fn [parameter value]
                                                  {:type (:dtype parameter) :value value})
                                                (drop 2 parameters) scalars))))]
            (try
              ((runtime 'launch-registered-bound!) prepared)
              ((runtime 'buffer->array) y)
              (finally ((runtime 'destroy-prepared!) prepared))))
          (finally ((runtime 'free-buffer!) y))))
      (finally ((runtime 'free-buffer!) x)))))

(deftest layout-integer-widths-execute-with-masked-tails
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "layout transform retained integer widths")
    (do
      (doseq [extent-dtype [:int :long]]
        (let [n 1025
              input (float-array (map float (range n)))
              kernel (schedule/cast-body
                      {:id :cast-width :input 'x :output 'y
                       :extent-dtype extent-dtype
                       :source-dtype :float :destination-dtype :float
                       :vector-width 4 :rounding :exact :overflow :exact})
              actual (vec (execute kernel input (float-array (repeat (+ n 3) -1.0)) [n]))]
          (is (= (vec input) (subvec actual 0 n)))
          (is (= [-1.0 -1.0 -1.0] (subvec actual n)))))
      (doseq [row-dtype [:int :long] column-dtype [:int :long]]
        (let [kernel (schedule/transpose-body
                      {:id :transpose-width :input 'x :output 'y :element-dtype :float
                       :row-extent-dtype row-dtype :column-extent-dtype column-dtype})
              actual (vec (execute kernel (float-array [1 2 3 4 5 6])
                                   (float-array (repeat 8 -1.0)) [2 3]))]
          (is (= [1.0 4.0 2.0 5.0 3.0 6.0] (subvec actual 0 6)))
          (is (= [-1.0 -1.0] (subvec actual 6))))))))
