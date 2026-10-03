(ns raster.gpu.test-lifecycle
  "Ownership-complete host fixtures. This helper does not validate plans or acquire native state;
   tests using it must mock their corresponding resource release operations."
  (:require [raster.gpu.link :as link]))

(defn linked-executable [fields]
  ((ns-resolve 'raster.gpu.link 'own-linked-executable)
   (link/map->LinkedExecutable
    (merge {:closed? (atom false) :lifetime-lock (Object.)
            :execution-state (atom {:value-epoch 0})}
           fields))))
