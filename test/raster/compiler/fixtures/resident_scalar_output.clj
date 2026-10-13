(ns raster.compiler.fixtures.resident-scalar-output
  "A public rank-zero reduction result that is also consumed by later device work."
  (:require [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm joint-norm-with-output
  [x :- (Array float), y :- (Array float), out :- (Array float), n :- Long] :- Double
  (let [nx (par/reduce sum (float 0.0) i n
                       (let [v (arrays/aget x i)] (n/+ sum (n/* v v))))
        ny (par/reduce sum (float 0.0) i n
                       (let [v (arrays/aget y i)] (n/+ sum (n/* v v))))
        norm (n/sqrt (n/+ (double nx) (double ny)))]
    (par/map-void! i n (arrays/aset out i (float norm)))
    (double norm)))

(defn inputs
  ([] (inputs 0))
  ([variant]
   (case variant
     0 {:x (float-array [1.0 2.0 3.0])
        :y (float-array [4.0 5.0 6.0])
        :out (float-array 3) :n (long 3)}
     1 {:x (float-array [2.0 0.0 1.0])
        :y (float-array [3.0 4.0 0.0])
        :out (float-array 3) :n (long 3)})))

(defn arguments [{:keys [x y out n]}]
  [x y out n])

(defn oracle [{:keys [x y n]}]
  (Math/sqrt
   (+ (reduce + 0.0 (map (fn [v] (* (double v) (double v))) (take n x)))
      (reduce + 0.0 (map (fn [v] (* (double v) (double v))) (take n y))))))
