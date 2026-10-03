(ns raster.gpu.layout-transform-device-support
  "Native layout oracles use the same scheduled-body/artifact boundary as production graphs."
  (:require [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.layout-transform :as layout]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]))

(defn- artifact [kernel-name body arguments legality numerics]
  (target/emit-artifact
   kernel-name
   (scheduled/make
    {:source {:kind :layout-transform-oracle :schedule (:schedule body)}
     :body body :arguments arguments
     :effects {:kind :layout-transform :uses (scheduled/derive-uses body arguments)}
     :legality legality :numerics numerics})
   :opencl-intel))

(defn cast-artifact [vector-width]
  (artifact
   (str "typed_layout_cast_" vector-width)
   (layout/cast-body {:id [:layout-oracle :cast vector-width] :input 'input :output 'output
                      :source-dtype :float :destination-dtype :half :vector-width vector-width
                      :rounding :nearest-even :overflow :ieee})
   '[input output elements]
   {:kind :dense-affine-cast :vector-width vector-width}
   {:mode :bounded-error :policy :f32-to-f16-storage :rounding :nearest-even
    :accumulator-dtype :half :error-model {:kind :ieee-f16-conversion :overflow :ieee}}))

(defn transpose-artifact [dtype]
  (artifact
   (str "typed_layout_transpose_" (name dtype))
   (layout/transpose-body {:id [:layout-oracle :transpose dtype] :input 'input :output 'output
                           :element-dtype dtype})
   '[input output rows cols]
   {:kind :bijective-affine-permutation :permutation [1 0]}
   {:mode :exact :policy :bit-preserving-permutation}))
