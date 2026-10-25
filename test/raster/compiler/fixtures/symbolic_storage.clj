(ns raster.compiler.fixtures.symbolic-storage
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as a]
            [raster.par :as p]))

(deftm regrouped-map
  [x :- (Array float) batch :- Long heads :- Long width :- Long] :- (Array float)
  (let [heads-width (clojure.core/* heads width)
        batch-heads (clojure.core/* batch heads)
        producer-count (clojure.core/* batch heads-width)
        consumer-count (clojure.core/* batch-heads width)
        first-row (p/pmap i producer-count float
                         (float (clojure.core/+ (a/aget x i) (float 1.0))))]
    (p/pmap j consumer-count float
            (float (clojure.core/* (a/aget first-row j) (float 2.0))))))

(deftm prefix-map
  [x :- (Array float) n :- Long m :- Long] :- (Array float)
  (let [first-row (p/pmap i n float (a/aget x i))]
    (p/pmap j m float (a/aget first-row j))))

(deftm fill-and-read-prefix!
  [x :- (Array float), boundary :- (Array float), out :- (Array float),
   full :- Long, owned :- Long] :- Void
  (p/map-void! i full (a/aset boundary i (a/aget x i)))
  (p/map-void! j owned (a/aset out j (a/aget boundary j))))

(deftm return-written-prefix-buffer
  [out :- (Array float)] :- (Array float)
  (let [write (p/map! out i 2 float (float i))] write))

(deftm return-written-prefix-alias
  [out :- (Array float)] :- (Array float)
  (let [write (p/map! out i 2 float (float i)) alias write] alias))

(deftm return-written-prefix-aliased-destination
  [out :- (Array float)] :- (Array float)
  (let [count out destination count
        write (p/map! destination i 2 float (float i))] write))

(deftm return-written-prefix-tuple
  [out :- (Array float)] :- Object
  (let [write (p/map! out i 2 float (float i))] [write out]))

(deftm return-written-prefix-allocation
  [seed :- (Array float)] :- (Array float)
  (let [out (float-array 8) write (p/map! out i 2 float (float i))] write))

(deftm return-written-prefix-pair
  [a :- (Array float), b :- (Array float)] :- Object
  (let [left (p/map! a i 2 float (float i))
        right (p/map! b i 2 float (raster.numeric/+ (float 10.0) (float i)))]
    [right left]))

(deftm return-two-writes-to-one-buffer
  [out :- (Array float)] :- Object
  (let [left (p/map! out i 8 float (float i))
        right (p/map! out i 8 float (raster.numeric/+ (float 10.0) (float i)))]
    [left right]))

(deftm shifted-map
  [x :- (Array float), n :- Long] :- (Array float)
  (p/pmap i n float (a/aget x (inc i))))

(deftm indirect-map
  [x :- (Array float), indices :- (Array long), n :- Long] :- (Array float)
  (p/pmap i n float (a/aget x (a/aget indices i))))

(deftm shifted-scan
  [x :- (Array float), n :- Long] :- (Array float)
  (let [out (float-array n)]
    (p/scan out acc (float 0.0) i n float
            (raster.numeric/+ acc (a/aget x (inc i))))))

(deftm indirect-scan
  [x :- (Array float), indices :- (Array long), n :- Long] :- (Array float)
  (let [out (float-array n)]
    (p/scan out acc (float 0.0) i n float
            (raster.numeric/+ acc (a/aget x (a/aget indices i))))))

(deftm fused-bias-contract!
  [a :- (Array float), b :- (Array float), bias :- (Array float), out :- (Array float)] :- Void
  (p/map! out t 32 nil (a/aget bias (rem t 8)))
  (p/contract out [[i 4] [j 8]] [[k 16]]
              (raster.numeric/* (a/aget a (+ (* i 16) k))
                                (a/aget b (+ (* k 8) j)))
              :epilogue {:acc acc
                         :expr (+ acc (clojure.core/aget out (+ (* i 8) j)))
                         :operands [{:sym out :map {:groups [[[i 4]] [[j 8]]]}
                                     :dtype :float}]
                         :scalars [] :dtype :float}))
