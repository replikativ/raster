(ns raster.compiler.backend.gpu.staged-contraction-fixtures
  (:require [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contraction-facts :as facts]))

(defn packed-facts [m n blocks width]
  (let [a-map (am/of-axes [['i m] ['blk blocks] ['t width]])
        b-map (am/of-axes [['j n] ['blk blocks] ['t width]])]
    (facts/from-components
     {:out 'out :free-axes [['i m] ['j n]]
      :contract-axes [['blk blocks] ['t width]] :dtype :byte
      :body (list 'raster.numeric/* (list 'aget 'a (am/index-expr a-map))
                  (list 'aget 'b (am/index-expr b-map)))
      :opts {:operands [{:sym 'a :map a-map} {:sym 'b :map b-map}]
             :stages [{:axis 'blk :extent blocks :dtype :float :init 0.0
                       :lift '(raster.numeric/* inner (aget da _) (aget db _))
                       :operands [{:sym 'da :dtype :float :map (am/of-axes [['i m] ['blk blocks]])}
                                  {:sym 'db :dtype :float :map (am/of-axes [['j n] ['blk blocks]])}]}
                      {:axis 't :extent width :dtype :int :init 0}]}})))
