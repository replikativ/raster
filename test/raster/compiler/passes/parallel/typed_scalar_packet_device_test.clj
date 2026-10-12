(ns raster.compiler.passes.parallel.typed-scalar-packet-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.passes.parallel.typed-scalar-packet-test :as packet]
            [raster.gpu.device-probe :as opencl-probe]
            [raster.dl.gpu-grad-parity :as level-zero]))

(defn- check-scalar! [target]
  (let [runtime (case target :ocl:0 'raster.gpu.ocl-runtime :ze:0 'raster.gpu.ze-runtime)
        register! (requiring-resolve (symbol (str runtime) "register-kernel-dispatch!"))
        scheduled (:form (lower/segop-lower-pass packet/source
                                                (assoc packet/options :target-device target)))
        emitted (opencl/opencl-pass scheduled :device-id target :dtype :float
                                   :compile-spirv? (= :ze:0 target) :min-elements 0)
        execute (eval (list 'fn '[xs] (:form emitted)))]
    (doseq [dispatch (:dispatches emitted)] (register! dispatch))
    (is (seq (:dispatches emitted)))
    (is (zero? (get-in emitted [:stats :fallback] 0)))
    (doseq [values [[-1 2 3.5] [7 -4 0.5] [] (repeat 1025 1)]]
      (let [input (float-array values)
            before (vec input)
            expected (Math/sqrt (reduce + 0.0 (map #(* (double %) (double %)) values)))]
        (is (< (Math/abs (- expected (double (execute input)))) 1.0e-6))
        (is (= before (vec input)))))))

(deftest opencl-host-scalar-consumer-uses-imported-reduction-graph
  (if @opencl-probe/opencl-available?
    (check-scalar! :ocl:0)
    (opencl-probe/opencl-skip! "imported scalar reduction result identity")))

(deftest level-zero-host-scalar-consumer-uses-imported-reduction-graph
  (if @level-zero/gpu-available?
    (check-scalar! :ze:0)
    (level-zero/gpu-skip! "imported scalar reduction result identity")))
