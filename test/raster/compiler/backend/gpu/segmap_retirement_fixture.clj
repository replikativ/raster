(ns raster.compiler.backend.gpu.segmap-retirement-fixture
  "Shared source-to-canonical-map fixtures; no historical source emitter oracle."
  (:require [raster.par]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]))

(def sources
  {:secondary-output
   '(let* [hout1 (raster.par/pmap i n float
                                (* (clojure.core/aget d i) (clojure.core/aget b i)))
           hout2 (raster.par/pmap j n float
                                (* (clojure.core/aget d j) (clojure.core/aget a j)))]
      [hout1 hout2])
   :integer-offset
   '(let* [out (raster.par/pmap i n float
                              (clojure.core/aget a (clojure.core/+ i offset)))] out)})

(defn emit [kind target compile-spirv?]
  (let [source (get sources kind)
        options {:dtype :float :target-device target
                 :array-types {'a :float 'b :float 'd :float}
                 :scalar-types {'n :long 'offset :int}}
        admitted (route/attempt source :float (:array-types options) options)
        _ (when-not (:program admitted)
            (throw (ex-info "retirement fixture requires complete typed admission"
                            {:kind kind :declined (:declined admitted)})))
        packet (:form (lower/segop-lower-pass (:program admitted) options))
        emitted (opencl/opencl-pass packet :device-id target :dtype :float
                                   :compile-spirv? compile-spirv? :min-elements 0)]
    {:source source :packet packet :admission admitted :emitted emitted
     :operation (first (get-in packet [:equations 0 :operations]))
     :artifact (first (:kernels emitted))}))
