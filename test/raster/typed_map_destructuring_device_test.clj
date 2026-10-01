(ns raster.typed-map-destructuring-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.typed-map-destructuring-test :as surface]
            [raster.compiler.core.params-flatten :as pf]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn- run-flat-device [target]
  ;; This proves the existing flat kernel path, not a logical-map GPU facade.
  (let [flat-var (:raster.params/flat-var (meta #'surface/project-map!))
        input {:values (float-array (range 17)) :offset (float 2)}
        args (into (pf/flatten-value '(HMap {:values (Array float) :offset Float}) input)
                   [(float-array (repeat 17 Float/NaN)) 17])
        prepared (compiled/lower flat-var args
                                 {:target target :dtype :float :compiler :equation-first})
        program (compiled/instantiate! prepared)]
    (try
      (is (= 0 (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
      (doseq [_ (range 2)]
        (is (= (vec (surface/project-map! input (float-array 17) 17))
               (vec (value/->host (:result (program {})))))))
      (let [input-key (:key (first (filter #(= :input (:role %)) (:in-tree prepared))))
            changed (assoc input :values (float-array (range 17 34)))]
        (is (= (vec (surface/project-map! changed (float-array 17) 17))
               (vec (value/->host (:result (program {input-key (:values changed)})))))))
      (finally (compiled/close! program)))))

(deftest flattened-map-on-opencl
  (if @opencl/opencl-available? (run-flat-device :ocl:0)
      (opencl/opencl-skip! "flattened-map-opencl")))

(deftest flattened-map-on-level-zero
  (if @ze/gpu-available? (run-flat-device :ze:0)
      (ze/gpu-skip! "flattened-map-level-zero")))
