(ns raster.compiler.fixtures.mixed-storage
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]))

(def policy {:preserve-declared-array-storage? true})

(deftm object-context-array-map
  [values :- (Array double)] :- (Array double)
  (let [mapped (par/map [i (alength values)] (* (aget values i) 2.0))]
    (nth [mapped mapped] 0)))

(deftm double-exponential-float-storage
  "Explicit Double transcendental evaluation followed by one Float storage conversion."
  [x :- (Array float)] :- (Array float)
  (let [out (arrays/zeros-like x (alength x))]
    (par/map! out i (alength x) nil
      (float (Math/exp (double (aget x i)))))))

(deftm double-fold-float-storage
  [x :- (Array float)] :- (Array float)
  (let [out (arrays/zeros-like x 1)]
    (par/map! out i 1 nil
      (float
       (loop [j 0 acc (double 0.0)]
         (if (< j 3)
           (recur (inc j) (+ acc (double (aget x j))))
           acc))))))

(deftm dynamic-double-fold-float-storage
  [x :- (Array float) n :- Long] :- (Array float)
  (let [out (arrays/zeros-like x 1)]
    (par/map! out i 1 nil
      (float
       (loop [j 0 acc (double 0.0)]
         (if (< j n)
           (recur (inc j) (+ acc (double (aget x j))))
           acc))))))

(deftm mixed-scale
  [coefficient :- Double values :- (Array float)] :- (Array float)
  (par/scale coefficient values))

(deftm checked-scale-effect!
  "A checked write-only map whose public result is Void, not the destination."
  [coefficient :- Double values :- (Array float) out :- (Array float) n :- Long] :- Void
  (par/map-void! i n
    (aset out i (float (* coefficient (double (aget values i)))))))

(deftm checked-scale-effect-result
  "Compose a Void checked helper with a genuinely private returned destination."
  [coefficient :- Double values :- (Array float) n :- Long] :- (Array float)
  (let [out (float-array n)]
    (checked-scale-effect! coefficient values out n)
    out))

(deftm double-reduction-float-result
  [values :- (Array double)] :- Float
  (let [total (par/reduce acc 0.0 i (alength values)
                          (+ acc (aget values i)))
        ^float result (float total)]
    result))

(deftm double-reduction-float-roundtrip
  [values :- (Array double)] :- Double
  (let [total (par/reduce acc 0.0 i (alength values)
                          (+ acc (aget values i)))]
    (double (float total))))

(deftm public-scaled-reduction-float-roundtrip
  [coefficient :- Double, values :- (Array double)] :- Double
  (let [total (par/reduce acc 0.0 i (alength values) (+ acc (aget values i)))]
    (double (float (* total coefficient)))))

(deftm public-affine-reduction-float-roundtrip
  [offset :- Long, values :- (Array double), coefficient :- Double] :- Double
  (let [total (par/reduce acc 0.0 i (alength values) (+ acc (aget values i)))]
    (double (float (+ (* total coefficient) (double offset))))))

(deftm scaled-reduction-float-roundtrip
  [values :- (Array double)] :- Double
  (let [total (par/reduce acc 0.0 i (alength values)
                          (+ acc (aget values i)))]
    (double (float (* total 1.5)))))

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
