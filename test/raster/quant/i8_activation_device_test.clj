(ns raster.quant.i8-activation-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.quant.kernels-k :as qk]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.gpu.device-probe :as probe]))

(deftest typed-i8-activation-packing-is-row-local-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "typed I8 activation packing")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve ocl 'register-kernel!)
          buffer-of-array (ns-resolve ocl 'buffer-of-array)
          make-buffer (ns-resolve ocl 'make-buffer)
          bind-call (ns-resolve ocl 'bind-kernel-call)
          launch! (ns-resolve ocl 'launch-registered-bound!)
          read! (ns-resolve ocl 'buffer->array)
          free! (ns-resolve ocl 'free-buffer!)
          width 64 rows 3 blocks (* rows (quot width 32))
          input (float-array (mapcat (fn [scale]
                                       (take width (cycle (map #(* scale %) [-127 127 -2.5 -1.5 -0.5 0.5 1.5 2.5]))))
                                     [1.0 2.0 0.0]))
          expected-words (int-array (* rows (quot width 4)))
          expected-scales (float-array blocks)
          _ (qk/quant-act-i8-rows-gpu! input expected-words expected-scales width rows)
          compiled (first (:kernels (pipeline/show-pipeline #'qk/quant-act-i8-rows-gpu!
                                                            :target-device :ocl:0 :dtype :float)))
          x (buffer-of-array input :float)
          xp (make-buffer (alength expected-words) :int)
          xs (make-buffer blocks :float)]
      (try
        (register! (:kernel-name compiled) compiled)
        (launch! (bind-call (kcall/make compiled [x xp xs {:type :long :value width}
                                                {:type :long :value blocks}])))
        (is (= (vec expected-words) (vec (read! xp))))
        (is (= (vec expected-scales) (vec (read! xs))))
        (finally (free! xs) (free! xp) (free! x))))))
