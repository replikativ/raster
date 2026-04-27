(ns raster.deftm-tree-compose-test
  "Composition test for the unified deftm + tree-typed-args API.

  Validates that:
    1. deftm accepts HMap/HVec arg annotations directly (no Params wrapper).
    2. A sub-deftm taking an HMap arg can be called with a tree slice
       extracted from an outer deftm's HMap/HVec arg, and the cross-deftm
       splice rewrites the call to the flat-var with leaves spliced in.
    3. compile-aot of the outer deftm produces a working compiled function
       whose output matches a hand-rolled flat-arg equivalent."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core]
            [raster.params :as rp]))

;; --------------------------------------------------------------------------
;; A small "block": affine y = W*x + b operating on length-d vectors
;; --------------------------------------------------------------------------
(core/deftm affine-block
  [bw :- (HMap :mandatory {:W (Param (Array double))
                            :b (Param (Array double))})
   x  :- (Array double)
   d  :- Long]
  :- (Array double)
  (let [out (double-array d)]
    (dotimes [i d]
      (let [W (:W bw)
            b (:b bw)
            wi (clojure.core/aget W i)]
        (clojure.core/aset out i
                           (clojure.core/+ (clojure.core/aget b i)
                                           (clojure.core/* wi
                                                           (clojure.core/aget x i))))))
    out))

;; --------------------------------------------------------------------------
;; Outer model: HMap of two HVec-stacked blocks; calls affine-block per layer
;; --------------------------------------------------------------------------
(core/deftm two-block-net
  [w :- (HMap :mandatory
              {:layers (HVec [(HMap :mandatory {:W (Param (Array double))
                                                 :b (Param (Array double))})
                              (HMap :mandatory {:W (Param (Array double))
                                                 :b (Param (Array double))})])})
   x :- (Array double)
   d :- Long]
  :- (Array double)
  (let [h1 (affine-block (clojure.core/nth (:layers w) 0) x  d)
        h2 (affine-block (clojure.core/nth (:layers w) 1) h1 d)]
    h2))

;; --------------------------------------------------------------------------
;; Hand-rolled equivalent: same body but the layer params are flat positional
;; --------------------------------------------------------------------------
(core/deftm two-block-net-flat
  [w0 :- (Array double) b0 :- (Array double)
   w1 :- (Array double) b1 :- (Array double)
   x  :- (Array double) d :- Long]
  :- (Array double)
  (let [h1 (affine-block {:W w0 :b b0} x  d)
        h2 (affine-block {:W w1 :b b1} h1 d)]
    h2))

(deftest deftm-recognizes-bare-hmap-annotation
  (testing "deftm with bare HMap arg (no Params wrapper) becomes a tree-typed model"
    (let [m (meta #'two-block-net)]
      (is (some? (:raster.params/treedefs m))
          "bare HMap arg in deftm should produce ::treedefs metadata")
      (is (some? (:raster.params/flat-var m))
          "should generate a --flat var")
      (is (= '[w x d] (:raster.params/original-args m))
          "original-args should preserve the user-facing signature"))))

(deftest cross-deftm-splice-runs-at-runtime
  (testing "outer deftm calls sub-deftm with tree slices; runs correctly"
    (let [d 4
          x  (double-array [1.0 2.0 3.0 4.0])
          w0 (double-array [0.5 0.5 0.5 0.5])
          b0 (double-array [0.1 0.2 0.3 0.4])
          w1 (double-array [2.0 2.0 2.0 2.0])
          b1 (double-array [0.0 0.0 0.0 0.0])
          tree {:layers [{:W w0 :b b0} {:W w1 :b b1}]}
          structured-result (two-block-net tree x d)
          flat-result       (two-block-net-flat w0 b0 w1 b1 x d)]
      (is (= (vec structured-result) (vec flat-result))
          "structured-call result should match flat-call result"))))

(deftest compile-aot-on-tree-typed-deftm
  (testing "compile-aot of tree-typed deftm produces a working fn"
    (let [d 4
          compiled (rp/compile-aot #'two-block-net)
          x  (double-array [1.0 2.0 3.0 4.0])
          w0 (double-array [0.5 0.5 0.5 0.5])
          b0 (double-array [0.1 0.2 0.3 0.4])
          w1 (double-array [2.0 2.0 2.0 2.0])
          b1 (double-array [0.0 0.0 0.0 0.0])
          tree {:layers [{:W w0 :b b0} {:W w1 :b b1}]}
          compiled-result (compiled tree x d)
          eager-result    (two-block-net tree x d)]
      (is (= (vec compiled-result) (vec eager-result))
          "compile-aot output should match eager structured-call output"))))
