(ns raster.compiler.passes.parallel.typed-write-packet-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.passes.parallel.typed-write-packet-test :as packet]
            [raster.gpu.device-probe :as opencl-probe]
            [raster.dl.gpu-grad-parity :as level-zero]))

(defn- check-scatter! [target index-dtype]
  (let [runtime (case target :ocl:0 'raster.gpu.ocl-runtime :ze:0 'raster.gpu.ze-runtime)
        make-arena! (requiring-resolve (symbol (str runtime) "make-kernel-arena!"))
        register! (requiring-resolve (symbol (str runtime) "register-kernel!"))
        close! (requiring-resolve (symbol (str runtime) "close-kernel-arena!"))
        options (-> packet/options (assoc :target-device target)
                    (assoc-in [:array-types 'keys] index-dtype))
        scheduled (:form (lower/segop-lower-pass
                          (packet/source '(raster.par/scatter! out vals keys n stride)) options))
        emitted (opencl/opencl-pass scheduled :device-id target :dtype :float
                                   :compile-spirv? (= :ze:0 target) :min-elements 0)
        execute (eval (list 'fn '[out vals keys n stride] (:form emitted)))
        arena (make-arena!)]
    (try
      (doseq [artifact (:kernels emitted)] (register! (:kernel-name artifact) artifact arena))
      (is (= 1 (get-in emitted [:stats :segop-reused])))
      (is (= :reducing-scatter (get-in emitted [:kernels 0 :effects :kind])))
      (let [out (float-array [10 20 30 40])
            vals (float-array [1 2 3 4 5 6])
            keys ((case index-dtype :int int-array :long long-array) [1 0 1])]
        (is (identical? out (execute out vals keys 3 2)))
        (is (= [13.0 24.0 36.0 48.0] (vec out)))
        (is (identical? out (execute out vals keys 0 2)))
        (is (= [13.0 24.0 36.0 48.0] (vec out)))
        (aset vals 0 (float 9))
        (is (identical? out (execute out vals keys 3 2)))
        (is (= [16.0 28.0 50.0 56.0] (vec out)))
        (is (= [1 0 1] (vec keys))))
      (finally (close! arena)))))

(deftest opencl-recovered-strided-scatter-keeps-colliding-updates
  (if @opencl-probe/opencl-available?
    (doseq [dtype [:int :long]] (check-scatter! :ocl:0 dtype))
    (opencl-probe/opencl-skip! "recovered strided scatter")))

(deftest level-zero-recovered-strided-scatter-keeps-colliding-updates
  (if @level-zero/gpu-available?
    (doseq [dtype [:int :long]] (check-scatter! :ze:0 dtype))
    (level-zero/gpu-skip! "recovered strided scatter")))
