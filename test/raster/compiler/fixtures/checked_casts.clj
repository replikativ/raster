(ns raster.compiler.fixtures.checked-casts
  "Public source conversion shared by semantic tests and target-compiler CI."
  (:require [raster.arrays]
            [raster.core :refer [deftm]]
            [raster.par]))

(deftm narrow-rows!
  [input :- (Array long) output :- (Array int) n :- Long] :- (Array int)
  (raster.par/map! output index n int
                   (int (raster.arrays/aget input index))))

(deftm narrow-stores!
  [input :- (Array long) output :- (Array int) n :- Long] :- (Array int)
  (raster.par/map-void! index n
    (raster.arrays/aset output index (int (raster.arrays/aget input index))))
  output)

(deftm unused-narrow-rows!
  [input :- (Array long) output :- (Array int) n :- Long] :- (Array int)
  (raster.par/map! output index n int
                   (let [checked (int (raster.arrays/aget input index))]
                     0)))

(deftm annihilated-narrow-rows!
  [input :- (Array long) output :- (Array int) n :- Long] :- (Array int)
  (raster.par/map! output index n int
                   (* (int (raster.arrays/aget input index)) 0)))

(deftm checked-prefix-rows!
  [input :- (Array long) output :- (Array int) n :- Long limit :- Long] :- (Array int)
  (let [checked (int limit)]
    (raster.par/map! output index n int
                     (clojure.core/unchecked-int (raster.arrays/aget input index)))))

(deftm checked-after-write!
  [output :- (Array int) n :- Long limit :- Long] :- (Array int)
  (let [written (raster.par/map! output index n int 1)
        checked (int limit)]
    written))

(deftm narrow-index!
  [output :- (Array byte) n :- Long] :- (Array byte)
  (raster.par/map! output index n byte (byte index)))
