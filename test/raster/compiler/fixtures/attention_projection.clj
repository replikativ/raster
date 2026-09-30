(ns raster.compiler.fixtures.attention-projection
  "A JVM-callable composed attention/projection oracle.
   Array helpers have Double source methods; FP32 device storage is an explicit compile policy."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.dl.array-ops :as array-ops]))

(deftm attention-projection
  [q :- (Array double) k :- (Array double) v :- (Array double)
   dst :- (Array long) src :- (Array long) weights :- (Array double)
   rows :- Long edges :- Long width :- Long heads :- Long] :- (Array float)
  (let [dk (quot width heads)
        raw (array-ops/indexed-dot q k dst src rows rows edges dk width heads)
        scaled (array-ops/scale-clamp-exp raw (/ 1.0 (n/sqrt dk)) 5.0 (* edges heads))
        denominator (array-ops/scatter-add scaled dst rows edges heads)
        weighted (array-ops/scatter-mul-add scaled v dst src rows rows edges dk width heads)
        normalized (array-ops/segment-div weighted denominator rows width heads 1.0e-6)
        result (float-array (* rows 7))]
    (par/contract result [[i rows] [j 7]] [[p width]]
                  (n/* (arrays/aget normalized (+ (* i width) p))
                       (arrays/aget weights (+ (* p 7) j)))
                  :init (float 0.0) :combine n/+)
    result))

(defn arguments []
  [(double-array (map #(/ (inc %) 16.0) (range 12)))
   (double-array (map #(/ (- % 4) 16.0) (range 12)))
   (double-array (map #(/ (mod % 7) 8.0) (range 12)))
   (long-array [0 0 2 2]) (long-array [1 2 0 2])
   (double-array (map #(/ (- (mod % 5) 2) 8.0) (range 28)))
   3 4 4 2])

(defn fp32-arguments [arguments]
  (mapv #(if (instance? (Class/forName "[D") %) (float-array %) %) arguments))
