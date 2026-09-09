(ns raster.compiler.passes.parallel.carried-effect-loop-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.carried-effect-loop-fixture :as fixture]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.gpu.device-probe :as probe]))

(deftest carried-store-loops-match-the-sequential-oracle-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "scheduled carried store loops")
    (let [runtime (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve runtime 'register-kernel!)
          buffer-of-array (ns-resolve runtime 'buffer-of-array)
          bind-call (ns-resolve runtime 'bind-kernel-call)
          launch! (ns-resolve runtime 'launch-registered-bound!)
          read! (ns-resolve runtime 'buffer->array)
          free! (ns-resolve runtime 'free-buffer!)]
      (doseq [[kind trips] (concat (for [kind [:scheduled :typed] trips [0 1 8]] [kind trips])
                                   [[:typed-index-collision 8]])]
        (let [operation (if (= kind :scheduled)
                          (fixture/scheduled-loop trips)
                          (first (lower/lower-typed-effect-map
                                  (cond-> (fixture/typed-program trips)
                                    (= kind :typed-index-collision) (dialect/remap-values {'seed 'i})) :ze:0)))
              compiled (fixture/artifact operation :opencl-portable
                                         :scalar-types {'rows :long 'seed :float 'i :float})
              values (vec (range 16))
              x (buffer-of-array (float-array values) :float)
              words (buffer-of-array (float-array (repeat 19 -77)) :float)
              totals (buffer-of-array (float-array (repeat 5 -77)) :float)]
          (try
            (register! (:kernel-name compiled) compiled)
            (launch! (bind-call (call/make compiled
                                          (mapv {'x x 'totals totals 'words words
                                                 'rows {:type :long :value 2}
                                                 'seed {:type :float :value 0.25}
                                                 'i {:type :float :value 0.25}}
                                                (:arguments compiled)))))
            (is (= (vec (concat (map-indexed (fn [i v] (if (< (mod i 8) trips) (float v) -77.0)) values)
                                (repeat 3 -77.0)))
                   (vec (read! words))) "only the selected row elements are written")
            (is (= (vec (concat (for [row (range 2)]
                                  (+ 0.25 (reduce + (take trips (drop (* row 8) values)))))
                                (repeat 3 -77.0)))
                   (vec (read! totals))) "zero-trip carry and inactive output tail are preserved")
            (finally (free! totals) (free! words) (free! x))))))))
