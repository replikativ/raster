(ns raster.typed-map-destructuring-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.typed-map-destructuring-test :as surface]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn- run-logical-device [target]
  (let [input {:values (float-array (range 17)) :offset (float 2)}
        args [input (float-array (repeat 17 Float/NaN)) 17]
        prepared (compiled/lower #'surface/project-map! args
                                 {:target target :dtype :float :compiler :equation-first})
        program (compiled/instantiate! prepared)]
    (try
      (is (= 0 (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
      (is (= #{[:arg0 :values] :out} (set (map :key (:in-tree prepared)))))
      (doseq [_ (range 2)]
        (is (= (vec (surface/project-map! input (float-array 17) 17))
               (vec (value/->host (:result (program {})))))))
      (let [changed (assoc input :values (float-array (range 17 34)))
            last-result (:result (program {:arg0 changed}))]
        (is (= (vec (surface/project-map! changed (float-array 17) 17))
               (vec (value/->host last-result))))
        (is (thrown? clojure.lang.ExceptionInfo
                     (program {:arg0 (assoc changed :offset (float 3))})))
        (is (thrown? clojure.lang.ExceptionInfo
                     (program {:arg0 (dissoc changed :offset)})))
        (is (thrown? clojure.lang.ExceptionInfo
                     (program {:arg0 changed [:arg0 :values] (:values changed)})))
        (is (= (vec (surface/project-map! changed (float-array 17) 17))
               (vec (value/->host last-result)))
            "invalid replacements must not run or mutate the graph"))
      (finally (compiled/close! program)))))

(deftest logical-map-on-opencl
  (if @opencl/opencl-available? (run-logical-device :ocl:0)
      (opencl/opencl-skip! "logical-map-opencl")))

(deftest logical-map-on-level-zero
  (if @ze/gpu-available? (run-logical-device :ze:0)
      (ze/gpu-skip! "logical-map-level-zero")))
