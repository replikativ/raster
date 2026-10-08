(ns raster.gpu.collective-realization-device-test
  "Co-located numerical all-reduce oracle, not a multi-host or fabric performance benchmark."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.distributed-refinement-test :as fixture]
            [raster.compiler.ir.scan :as scan]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.device-probe :as opencl]))

(defn- check-realization! [device]
  (doseq [{:keys [n dtype operation input-values] :or {dtype :float operation '+}}
          [{:n 2} {:n 3}
           ;; Exact real sum is one, but the declared FP32 left-associated tree yields zero.
           {:n 3 :input-values (mapv #(vec (repeat 17 %)) [16777216.0 1.0 -16777216.0])}
           {:n 3 :operation '*
            :input-values (mapv (fn [worker]
                                  (mapv #(+ 0.7 (* worker 0.11) (* % 0.013)) (range 17)))
                                (range 3))}
           {:n 3 :dtype :double
            :input-values (mapv #(vec (repeat 17 %)) [9007199254740992.0 1.0 -9007199254740992.0])}]]
    (let [plan (fixture/realized-plan n {:target-device device
                                        :target-descriptor (hardware/descriptor-for device)}
                                    {:algebra (scan/certify-reassociation
                                               {:acc 'acc :init (if (= '* operation) 1.0 0.0)
                                                :lambda (list operation 'acc 'element)} dtype)
                                     :input-values input-values})
          cast (case dtype :float float :double double)
          array (case dtype :float float-array :double double-array)
          combine (case operation + + * *)
          inputs (or input-values
                     (mapv (fn [worker] (mapv #(+ 1 worker (* 0.25 %)) (range 17))) (range n)))
          ;; Independent CPU oracle: round input storage and every declared binary tree node.
          ;; No high-precision/global-sum oracle may silently replace the admitted FP policy.
          expected (apply mapv (fn [& xs] (reduce #(cast (combine %1 %2)) (map cast xs))) inputs)]
      (distributed/verify! (distributed/certify plan))
      (with-open [executable (runtime/instantiate! plan {:transport :resident-copy
                                                        :device-capacities {device 1048576}})]
        (runtime/run! executable)
        (let [outputs (runtime/output-values executable)
              session (get (:sessions executable) device)]
          (is (= n (count outputs)))
          (doseq [[_ values] outputs]
            (is (= 1 (count values)))
            (let [result (array 17)]
              (gpu/download-range! session (first (vals values)) result {:elements 17})
              (is (= expected (vec result))
                  (str "every participant receives the declared " dtype " " operation " tree")))))))))

(deftest contribution-certified-all-reduce-on-colocated-opencl-workers
  (if @opencl/opencl-available?
    (check-realization! :ocl:0)
    (opencl/opencl-skip! "contribution-certified co-located all-reduce")))

(deftest contribution-certified-all-reduce-on-colocated-level-zero-workers
  (if @ze/gpu-available?
    (check-realization! :ze:0)
    (ze/gpu-skip! "contribution-certified co-located all-reduce")))
