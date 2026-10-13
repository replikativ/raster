(ns raster.compiler.fixtures.uniform-input-scalar-closure
  "A dynamic Long[1] input load evaluated once and reused by two device consumers."
  (:require [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.par :as par]))

(deftm buffered-clock-two-maps
  [time-buffer :- (Array long), out-a :- (Array long), out-b :- (Array long), n :- Long]
  :- (Array long)
  (let [^long clock-value (arrays/aget time-buffer 0)
        ^long step (clojure.core/unchecked-add clock-value (long 3))
        ^long scaled (clojure.core/unchecked-multiply step (long 2))]
    (par/map-void! i n
      (arrays/aset out-a i (clojure.core/unchecked-add step (long i))))
    (par/map-void! i n
      (arrays/aset out-b i (clojure.core/unchecked-add scaled (long i))))
    out-a))

(defn inputs [clock]
  {:time-buffer (long-array [(long clock)])
   :out-a (long-array 3)
   :out-b (long-array 3)
   :n (long 3)})

(defn arguments [{:keys [time-buffer out-a out-b n]}]
  [time-buffer out-a out-b n])

(defn oracle [clock]
  (let [step (+ (long clock) 3)
        scaled (* step 2)]
    {:out-a (mapv #(+ step %) (range 3))
     :out-b (mapv #(+ scaled %) (range 3))}))
