(ns raster.ode.multilevel
  "Cell-centred numerical transfers expressed as ordinary typed programs.
   These operators do not perform adaptation, subcycling or flux correction. Callers supply
   disjoint dense row-major source/destination storage with the stated extents."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]))

(deftm prolong-constant-2d!
  [fine :- (Array double), coarse :- (Array double), nx :- Long, ny :- Long]
  (let [fine-width (* 2 ny)]
    (dotimes [i (* 2 nx)]
      (dotimes [j fine-width]
        (arrays/aset fine (+ (* i fine-width) j)
                     (arrays/aget coarse (+ (* (quot i 2) ny) (quot j 2))))))
    fine))

(deftm restrict-average-2d!
  [coarse :- (Array double), fine :- (Array double), nx :- Long, ny :- Long]
  (let [fine-width (* 2 ny)]
    (dotimes [i nx]
      (dotimes [j ny]
        (let [k (+ (* (* 2 i) fine-width) (* 2 j))]
          (arrays/aset coarse (+ (* i ny) j)
                       (* 0.25 (+ (arrays/aget fine k)
                                  (arrays/aget fine (+ k 1))
                                  (arrays/aget fine (+ k fine-width))
                                  (arrays/aget fine (+ k fine-width 1))))))))
    coarse))
