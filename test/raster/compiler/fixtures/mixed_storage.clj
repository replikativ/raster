(ns raster.compiler.fixtures.mixed-storage
  (:require [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]))

(def policy {:preserve-declared-array-storage? true})

(deftm mixed-scale
  [coefficient :- Double values :- (Array float)] :- (Array float)
  (par/scale coefficient values))

(deftm double-reduction-float-result
  [values :- (Array double)] :- Float
  (let [total (par/reduce acc 0.0 i (alength values)
                          (+ acc (aget values i)))
        ^float result (float total)]
    result))

(deftm mixed-scale-energy
  [coefficient :- Float values :- (Array double)] :- Double
  (par/dot-product (par/scale coefficient values) values))

(deftm mixed-scale-energy-gradient
  [coefficient :- Float values :- (Array double)] :- Float
  (let [gradient ((reverse/value+grad (var mixed-scale-energy) :wrt [0])
                  coefficient values)]
    (nth gradient 1)))

(deftm mixed-storage!
  [weights :- (Array float) state :- (Array double) out :- (Array double) n :- Long] :- Void
  (par/map-void! i n
    (aset out i (+ (double (aget weights i)) (aget state i)))))

(deftm mixed-scratch!
  [weights :- (Array float) out :- (Array double) n :- Long] :- Void
  (let [scratch (double-array n)]
    (par/map-void! i n (aset scratch i (double (aget weights i))))
    (par/map-void! i n (aset out i (aget scratch (- (- n i) 1))))))

(deftm mixed-fold-storage!
  "Sequentially replace each Float row element with a Double sum of the current row."
  [storage :- (Array float) nrows :- Long width :- Long] :- Void
  (par/map-void! row nrows
    (let [base (* row width)]
      (loop [col 0]
        (if (< col width)
          (do
            (aset storage (+ base col)
                  (float (loop [j 0 acc 0.0]
                           (if (< j width)
                             (recur (inc j) (+ acc (double (aget storage (+ base j)))))
                             acc))))
            (recur (inc col)))
          nil)))))
