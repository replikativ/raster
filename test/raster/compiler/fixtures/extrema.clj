(ns raster.compiler.fixtures.extrema
  "Shared source fixtures for native and typed GPU extrema differential gates."
  (:require [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.numeric :as numeric]))

(deftm max-float!
  [x :- (Array float), z :- (Array float), y :- (Array float), cnt :- Long] :- (Array float)
  (par/map! y i cnt float (Math/max (float (aget x i)) (float (aget z i))))
  y)

(deftm min-float!
  [x :- (Array float), z :- (Array float), y :- (Array float), cnt :- Long] :- (Array float)
  (par/map! y i cnt float (Math/min (float (aget x i)) (float (aget z i))))
  y)

(deftm max-double!
  [x :- (Array double), z :- (Array double), y :- (Array double), cnt :- Long] :- (Array double)
  (par/map! y i cnt double (numeric/max (double (aget x i)) (double (aget z i))))
  y)

(deftm min-double!
  [x :- (Array double), z :- (Array double), y :- (Array double), cnt :- Long] :- (Array double)
  (par/map! y i cnt double (numeric/min (double (aget x i)) (double (aget z i))))
  y)

(def cases
  [[#'max-float! :float float-array #(Float/floatToRawIntBits (float %)) :max]
   [#'min-float! :float float-array #(Float/floatToRawIntBits (float %)) :min]
   [#'max-double! :double double-array #(Double/doubleToRawLongBits (double %)) :max]
   [#'min-double! :double double-array #(Double/doubleToRawLongBits (double %)) :min]])

(def operand-pairs
  [[Double/NaN 1.0] [1.0 Double/NaN] [Double/NaN Double/NaN]
   [0.0 -0.0] [-0.0 0.0] [-0.0 -0.0] [0.0 0.0]
   [Double/POSITIVE_INFINITY 4.0] [-4.0 Double/NEGATIVE_INFINITY]
   [1.00000006 1.00000007] [-8.25 9.5]])

(defn same-result?
  "NaN classification is specified; non-NaNs and signed zeros retain exact bits."
  [bits actual expected]
  (every? true?
          (map (fn [a b]
                 (if (Double/isNaN (double b))
                   (Double/isNaN (double a))
                   (= (bits a) (bits b)))) actual expected)))
