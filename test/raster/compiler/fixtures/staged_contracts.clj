(ns raster.compiler.fixtures.staged-contracts
  "Public staged workloads shared by numerical tests, the debt ledger and vendor compile gates."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric]
            [raster.par]))

(deftm floating-three-stage!
  [a :- (Array float) b :- (Array float) weights :- (Array float)
   out :- (Array float) scale :- Float] :- Void
  (raster.par/contract out [[i 2] [j 3]] [[blk 2] [sub 3] [t 4]]
    (raster.numeric/* (raster.arrays/aget a (+ (* i 24) (* blk 12) (* sub 4) t))
                      (raster.arrays/aget b (+ (* j 24) (* blk 12) (* sub 4) t)))
    :stages [{:axis blk :extent 2 :dtype :float :init 0.0 :lift (* inner scale)}
             {:axis sub :extent 3 :dtype :float :init 0.0 :lift (* inner (aget weights _))
              :operands [{:sym weights :dtype :float :map {:groups [[[j 3] [blk 2] [sub 3]]]}}]}
             {:axis t :extent 4 :dtype :float :init 0.0}]))

(deftm floating-result-transform!
  [a :- (Array float) b :- (Array float) bias :- (Array float)
   out :- (Array float) gain :- Float] :- Void
  (raster.par/contract out [[i 2]] [[blk 2] [t 4]]
    (raster.numeric/* (raster.arrays/aget a (+ (* i 8) (* blk 4) t))
                      (raster.arrays/aget b (+ (* blk 4) t)))
    :stages [{:axis blk :extent 2 :dtype :float :init 0.0 :lift inner}
             {:axis t :extent 4 :dtype :float :init 0.0}]
    :epilogue {:acc value :expr (+ value (aget bias _) gain) :dtype :float
               :operands [{:sym bias :dtype :float :map {:groups [[[i 2]]]}}]}))

(defmacro decoded-read [array index]
  `(raster.arrays/aget ~array ~index))

(deftm floating-decoded-stages!
  [a :- (Array float) b :- (Array float) out :- (Array float)
   shift :- Float scale :- Float] :- Void
  (raster.par/contract out [[i 2]] [[blk 2] [t 4]]
    (raster.numeric/* (arrays/aget a (+ (* i 8) (* blk 4) t))
                      (decoded-read b (+ (* blk 4) t)))
    :decode {a (- x shift) b (* x scale)}
    :stages [{:axis blk :extent 2 :dtype :float :init 0.0 :lift inner}
             {:axis t :extent 4 :dtype :float :init 0.0}]))

(deftm widening-decoded-stages!
  [a :- (Array float) b :- (Array float) out :- (Array float) epsilon :- Double] :- Void
  (raster.par/contract out [[i 1]] [[blk 1] [t 1]]
    (raster.numeric/* (raster.arrays/aget a (+ i blk t))
                      (raster.arrays/aget b (+ blk t)))
    :decode {a (+ (double x) epsilon)}
    :stages [{:axis blk :extent 1 :dtype :float :init 0.0 :lift inner}
             {:axis t :extent 1 :dtype :float :init 0.0}]))

(deftm checked-long-stage!
  [a :- (Array float) b :- (Array float) out :- (Array float) gain :- Long] :- Void
  (raster.par/contract out [[i 1]] [[blk 2] [t 4]]
    (raster.numeric/* (raster.arrays/aget a (+ (* blk 4) t))
                      (raster.arrays/aget b (+ (* blk 4) t)))
    :stages [{:axis blk :extent 2 :dtype :float :init 0.0
              :lift (* inner (double (clojure.core/+ gain 1)))}
             {:axis t :extent 4 :dtype :float :init 0.0}]))
