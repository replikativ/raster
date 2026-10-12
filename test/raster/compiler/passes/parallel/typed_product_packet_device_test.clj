(ns raster.compiler.passes.parallel.typed-product-packet-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.passes.parallel.typed-product-packet-test :as packet]
            [raster.gpu.device-probe :as opencl-probe]
            [raster.dl.gpu-grad-parity :as level-zero]))

(defn- check-product! [target]
  (let [runtime (case target :ocl:0 'raster.gpu.ocl-runtime :ze:0 'raster.gpu.ze-runtime)
        make-arena! (requiring-resolve (symbol (str runtime) "make-kernel-arena!"))
        register! (requiring-resolve (symbol (str runtime) "register-kernel!"))
        register-dispatch! (requiring-resolve (symbol (str runtime) "register-kernel-dispatch!"))
        close! (requiring-resolve (symbol (str runtime) "close-kernel-arena!"))
        source (list 'let* ['result (list 'do (list 'identity :prefix)
                                          (list 'let* (second packet/source) '[sums counts]))]
                     'result)
        options (assoc packet/options :target-device target)
        scheduled (:form (lower/segop-lower-pass source options))
        emitted (opencl/opencl-pass scheduled :device-id target :dtype :float
                                   :compile-spirv? (= :ze:0 target) :min-elements 0)
        execute (eval (list 'fn '[xs n] (:form emitted)))
        arena (make-arena!)]
    (try
      (doseq [artifact (:kernels emitted)] (register! (:kernel-name artifact) artifact arena))
      (doseq [dispatch (:dispatches emitted)] (register-dispatch! dispatch))
      (is (seq (:kernels emitted)))
      (is (zero? (get-in emitted [:stats :fallback] 0)))
      (doseq [values [[-1 2 3.5] [7 -4 0.5] []]]
        (let [[sums counts] (execute (float-array values) (count values))]
          (is (= [(reduce + 0.0 (map double values))] (vec sums)))
          (is (= [(count values)] (vec counts)))
          (is (= (class (double-array 0)) (class sums)))
          (is (= (class (long-array 0)) (class counts)))))
      (finally (close! arena)))))

(deftest opencl-nested-product-preserves-double-sum-and-long-count
  (if @opencl-probe/opencl-available?
    (check-product! :ocl:0)
    (opencl-probe/opencl-skip! "nested mixed-storage product reduction")))

(deftest level-zero-nested-product-preserves-double-sum-and-long-count
  (if @level-zero/gpu-available?
    (check-product! :ze:0)
    (level-zero/gpu-skip! "nested mixed-storage product reduction")))
