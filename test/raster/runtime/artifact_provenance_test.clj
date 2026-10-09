(ns raster.runtime.artifact-provenance-test
  (:require [clojure.test :refer [deftest is]]
            [raster.runtime.artifact-provenance :as provenance]))

(defrecord Witness [data provenance-seal])

(deftest exact-object-and-private-issuer-identity
  (let [left (provenance/issuer) right (provenance/issuer)
        sealed ((:seal left) (->Witness {:value 1} nil))
        other ((:seal right) sealed)]
    (is ((:authentic? left) sealed))
    (is (not ((:authentic? right) sealed)))
    (is ((:authentic? right) other))
    (is (not ((:authentic? left) other)))
    (is (not ((:authentic? left) (assoc sealed :data {:value 2}))))
    (is (not ((:authentic? left) (map->Witness (into {} sealed)))))
    (is (not ((:authentic? left) (into {} sealed))))
    (is (not ((:authentic? left) nil)))
    (is (not ((:authentic? left) {:provenance-seal (fn [_] (Object.))})))))
