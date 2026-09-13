(ns raster.compiler.fixtures.checked-casts
  "Public source conversion shared by semantic tests and target-compiler CI."
  (:require [raster.arrays]
            [raster.core :refer [deftm]]
            [raster.par]))

(deftm narrow-rows!
  [input :- (Array long) output :- (Array int) n :- Long] :- (Array int)
  (raster.par/map! output index n int
                   (int (raster.arrays/aget input index))))

(deftm unused-narrow-rows!
  [input :- (Array long) output :- (Array int) n :- Long] :- (Array int)
  (raster.par/map! output index n int
                   (let [checked (int (raster.arrays/aget input index))]
                     0)))

(deftm annihilated-narrow-rows!
  [input :- (Array long) output :- (Array int) n :- Long] :- (Array int)
  (raster.par/map! output index n int
                   (* (int (raster.arrays/aget input index)) 0)))
