(ns raster.compiler.fixtures.absolute
  "Public source fixtures for absolute-value precision and exceptional values."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as numeric]
            [raster.par :as par]))

(deftm float-math-abs [src :- (Array float) n :- Long] :- (Array float)
  (let [out (arrays/alloc-like src n)]
    (par/map! out i n float (Math/abs (arrays/aget src i)))
    out))

(deftm double-math-abs [src :- (Array double) n :- Long] :- (Array double)
  (let [out (arrays/alloc-like src n)]
    (par/map! out i n double (Math/abs (arrays/aget src i)))
    out))

(deftm float-numeric-abs [src :- (Array float) n :- Long] :- (Array float)
  (let [out (arrays/alloc-like src n)]
    (par/map! out i n float (numeric/abs (arrays/aget src i)))
    out))

(deftm double-numeric-abs [src :- (Array double) n :- Long] :- (Array double)
  (let [out (arrays/alloc-like src n)]
    (par/map! out i n double (numeric/abs (arrays/aget src i)))
    out))

(def cases
  [[#'float-math-abs :float float-array Float/floatToRawIntBits]
   [#'float-numeric-abs :float float-array Float/floatToRawIntBits]
   [#'double-math-abs :double double-array Double/doubleToRawLongBits]
   [#'double-numeric-abs :double double-array Double/doubleToRawLongBits]])

(defn operands [dtype]
  (case dtype
    :float [-0.0 Float/NaN Float/NEGATIVE_INFINITY Float/POSITIVE_INFINITY
            Float/MIN_VALUE (- Float/MIN_VALUE) (- Float/MAX_VALUE)
            -1.0000001192092896 -1.0 0.0]
    :double [-0.0 Double/NaN Double/NEGATIVE_INFINITY Double/POSITIVE_INFINITY
             Double/MIN_VALUE (- Double/MIN_VALUE) (- Double/MAX_VALUE)
             -1.0000000000000002 -1.0 0.0]))
