(ns raster.par-scatter-test
  "Tests for par/scatter! as the scatter-reduce (scatter-ADD) combinator.

   Per the Futhark/JAX study, scalar-serial scatter is the correct lowering for
   a conflicting scatter-reduce (colliding indices accumulate via the commutative
   op; no atomics, no lost updates). par/scatter! is add-only, which is what the
   UMAP SGD needs. In a vectorized loop the scatter stays scalarized (lane
   extraction) — that is the SGD-vectorization step, not this combinator."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.par :as par]
            [raster.core :refer [deftm]]))

(deftest scatter-add-accumulates-collisions
  (testing "colliding indices accumulate (scatter-ADD, not last-wins)"
    (let [out (double-array 3)
          src (double-array [1.0 10.0 100.0 1000.0 5.0])
          idx (int-array [0 1 0 1 2])]   ; 0 twice, 1 twice, 2 once
      (par/scatter! out src idx 5)
      (is (= [101.0 1010.0 5.0] (vec out))))))

(deftest scatter-add-strided
  (testing "strided scatter-add accumulates per stride-row"
    (let [out (double-array 4)            ; 2 rows x 2
          src (double-array [1.0 2.0 3.0 4.0])
          idx (int-array [0 0])]          ; both rows -> row 0
      (par/scatter! out src idx 2 2)
      (is (= [4.0 6.0 0.0 0.0] (vec out))))))

(deftm scatter-kernel [out :- (Array double), src :- (Array double),
                       idx :- (Array int), n :- Long] :- (Array double)
  (raster.par/scatter! out src idx n))

(deftest scatter-add-in-deftm
  (testing "par/scatter! compiles in a deftm and accumulates collisions"
    (let [out (double-array 4)
          src (double-array [2.0 3.0 4.0 5.0 6.0 7.0])
          idx (int-array [0 0 1 3 3 0])]  ; 0 thrice, 3 twice
      (scatter-kernel out src idx 6)
      ;; idx0: 2+3+7=12, idx1: 4, idx3: 5+6=11
      (is (= [12.0 4.0 0.0 11.0] (vec out))))))
