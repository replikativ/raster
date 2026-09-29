(ns raster.compiler.fixtures.mixed-storage
  (:require [raster.core :refer [deftm]]
            [raster.par :as par]))

(def policy {:preserve-declared-array-storage? true})

(deftm mixed-storage!
  [weights :- (Array float) state :- (Array double) out :- (Array double) n :- Long] :- Void
  (par/map-void! i n
    (aset out i (+ (double (aget weights i)) (aget state i)))))

(deftm mixed-scratch!
  [weights :- (Array float) out :- (Array double) n :- Long] :- Void
  (let [scratch (double-array n)]
    (par/map-void! i n (aset scratch i (double (aget weights i))))
    (par/map-void! i n (aset out i (aget scratch (- (- n i) 1))))))
