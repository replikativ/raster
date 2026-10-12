(ns raster.compiler.passes.parallel.typed-imperative-map-packet-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.par]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.passes.parallel.typed-imperative-map-packet-test :as packet]
            [raster.dl.gpu-grad-parity :as level-zero]
            [raster.gpu.device-probe :as opencl-probe]))

(def ^:dynamic *host-calls* nil)
(defn host-control [value]
  (when *host-calls* (swap! *host-calls* inc))
  value)

(defn- run-native-packet! [target]
  (let [runtime (case target :ocl:0 'raster.gpu.ocl-runtime :ze:0 'raster.gpu.ze-runtime)
        make-arena! (requiring-resolve (symbol (str runtime) "make-kernel-arena!"))
        register! (requiring-resolve (symbol (str runtime) "register-kernel!"))
        close-arena! (requiring-resolve (symbol (str runtime) "close-kernel-arena!"))
        options (assoc (#'packet/options :float) :target-device target)
        source (#'packet/source :float)
        scheduled (:form (lower/segop-lower-pass source options))
        emitted (opencl/opencl-pass scheduled :device-id target :dtype :float
                                   :compile-spirv? (= :ze:0 target) :min-elements 0)
        execute (binding [*ns* (the-ns 'raster.compiler.passes.parallel.typed-imperative-map-packet-device-test)]
                  (eval (list 'fn '[x y out rows width] (:form emitted))))
        original (binding [*ns* (the-ns 'raster.compiler.passes.parallel.typed-imperative-map-packet-device-test)]
                   (eval (list 'fn '[^floats x ^floats y ^floats out rows width] source)))
        arena (make-arena!)]
    (try
      (doseq [artifact (:kernels emitted)] (register! (:kernel-name artifact) artifact arena))
      (is (= 1 (count (:kernels emitted))))
      (is (= 1 (get-in emitted [:stats :segop-reused])))
      (doseq [[rows width expected changed]
              [[0 2 [] []]
               [3 0 [0.0 0.0 0.0] [0.0 0.0 0.0]]
               [3 2 [-4.0 -15.125 1.0] [10.0 -15.125 1.0]]]]
        (let [x (float-array [1.0 2.0 3.5 -4.0 0.0 0.5])
              y (float-array [2.0 -3.0 0.25 4.0 -1.0 2.0])
              out (float-array rows) cpu (float-array rows) calls (atom 0)]
          (is (= expected (vec (original x y cpu (long rows) (long width)))))
          (binding [*host-calls* calls]
            (is (identical? out (execute x y out (long rows) (long width))))
            (is (= expected (vec out)))
            (aset x 0 (float 8.0))
            (is (identical? out (execute x y out (long rows) (long width))))
            (is (= changed (vec out)))
            (is (= 2 @calls) "host prefix executes once on each invocation, not compilation"))))
      (finally (close-arena! arena)))))

(deftest opencl-dense-map-packet-preserves-jvm-results-and-host-order
  (if @opencl-probe/opencl-available?
    (run-native-packet! :ocl:0)
    (opencl-probe/opencl-skip! "dense imperative map packet")))

(deftest level-zero-dense-map-packet-preserves-jvm-results-and-host-order
  (if @level-zero/gpu-available?
    (run-native-packet! :ze:0)
    (level-zero/gpu-skip! "dense imperative map packet")))
