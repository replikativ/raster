(ns raster.compiler.fixtures.contractions
  (:require [raster.core :refer [deftm]]
            [raster.arrays]
            [raster.numeric]
            [raster.par]))

(deftm fixed-matmul
  "Ragged NN contraction shared by source and device schedule oracles."
  [left :- (Array float) right :- (Array float)] :- (Array float)
  (let [output (float-array 35)]
    (raster.par/contract output [[i 5] [j 7]] [[k 3]]
                         (raster.numeric/*
                          (raster.arrays/aget left (clojure.core/+ (clojure.core/* i 3) k))
                          (raster.arrays/aget right (clojure.core/+ (clojure.core/* k 7) j)))
                         :init (float 0.0) :combine raster.numeric/+)
    output))

(deftm dynamic-matmul
  "The same NN contraction with public long extents, without shape specialization."
  [left :- (Array float) right :- (Array float)
   m :- Long n :- Long k :- Long] :- (Array float)
  (let [output (float-array (* m n))]
    (raster.par/contract output [[i m] [j n]] [[p k]]
                         (raster.numeric/*
                          (raster.arrays/aget left (+ (* i k) p))
                          (raster.arrays/aget right (+ (* p n) j)))
                         :init (float 0.0) :combine raster.numeric/+)
    output))
