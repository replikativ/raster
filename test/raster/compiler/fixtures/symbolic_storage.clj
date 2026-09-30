(ns raster.compiler.fixtures.symbolic-storage
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as a]
            [raster.par :as p]))

(deftm regrouped-map
  [x :- (Array float) batch :- Long heads :- Long width :- Long] :- (Array float)
  (let [heads-width (clojure.core/* heads width)
        batch-heads (clojure.core/* batch heads)
        producer-count (clojure.core/* batch heads-width)
        consumer-count (clojure.core/* batch-heads width)
        first-row (p/pmap i producer-count float
                         (float (clojure.core/+ (a/aget x i) (float 1.0))))]
    (p/pmap j consumer-count float
            (float (clojure.core/* (a/aget first-row j) (float 2.0))))))

(deftm prefix-map
  [x :- (Array float) n :- Long m :- Long] :- (Array float)
  (let [first-row (p/pmap i n float (a/aget x i))]
    (p/pmap j m float (a/aget first-row j))))
