(ns raster.gpu.test-lifecycle
  "Ownership-complete host fixtures. This helper does not validate plans or acquire native state;
   tests using it must mock their corresponding resource release operations."
  (:require [raster.gpu.link :as link]
            [raster.gpu.resource-cleanup :as cleanup]))

(defn native-buffer
  "Fault-double constructor with the production ownership protocol. acquire must return a map;
   uncertain acquisition is retained rather than guessed to have created nothing."
  [acquire release {:keys [retain-owner! adopt-cleanup!]}]
  (let [slot (cleanup/acquisition-slot)
        owner-ref (volatile! nil)
        owner (cleanup/owner [{:id :memory
                               :release #(cleanup/release-native!
                                          slot (fn [value] (release (assoc value ::cleanup/owner @owner-ref))))}])]
    (vreset! owner-ref owner)
    (cleanup/build! owner
                    #(do (when retain-owner! (retain-owner! owner))
                         (cleanup/acquire-native! slot acquire))
                    (or adopt-cleanup! retain-owner!))))

(defn linked-executable [fields]
  ((ns-resolve 'raster.gpu.link 'own-linked-executable)
   (link/map->LinkedExecutable
    (merge {:closed? (atom false) :lifetime-lock (Object.)
            :execution-state (atom {:value-epoch 0})}
           fields))))
