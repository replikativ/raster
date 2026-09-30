(ns raster.compiler.fixtures.scalar-helpers
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as a]
            [raster.numeric]
            [raster.par :as p]))

;; A bare cast tail is a pure scalar operation, not a scoped let/loop. All GPU frontend
;; entries must expose its existing typed body before SOAC admission.
(deftm cast-tail
  (All [T] [x :- T] :- T (T x)))

(deftm map-cast-helper
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (p/map-void! i n (a/aset output i (cast-tail (a/aget input i))))
    output))

(defn redefine-reload-tail! [factor]
  (binding [*ns* (the-ns 'raster.compiler.fixtures.scalar-helpers)]
    (eval (list 'deftm 'reload-tail
                (list 'All '[T] '[x :- T] ':- 'T
                      (list 'T (list 'raster.numeric/* 'x (list 'T factor))))))))

(redefine-reload-tail! 1.0)

(deftm map-reload-helper
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (p/map-void! i n (a/aset output i (reload-tail (a/aget input i))))
    output))
