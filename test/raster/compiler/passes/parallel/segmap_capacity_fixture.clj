(ns raster.compiler.passes.parallel.segmap-capacity-fixture
  "A typed flattened map with independently declared physical input capacities."
  (:require [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]))

(defn graph
  ([] (graph 4))
  ([a-capacity]
   (let [options {:dtype :float :array-types {'a :float 'b :float 'C :float}
                  :values {'a (av/tensor {:dtype :float :shape [a-capacity]})
                           'b (av/tensor {:dtype :float :shape [3]})
                           'C (av/tensor {:dtype :float :shape [12]})}}
         typed (:program
                (typed-route/attempt
                 '(let* [r (raster.par/map! C i 12 nil
                                          (* (aget a (quot i 3)) (aget b (rem i 3))))] r)
                 :float (:array-types options) options))
         algorithm (get-in typed [:equations 0 :algorithm])
         scheduled (:form (segop-lower/segop-lower-pass typed options))]
     (equation-graph/make algorithm scheduled))))
