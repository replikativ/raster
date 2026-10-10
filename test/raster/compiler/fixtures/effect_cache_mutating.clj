(ns raster.compiler.fixtures.effect-cache-definition)
(defn child [x] (swap! state inc) x)
