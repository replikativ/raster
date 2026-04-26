(ns raster.params-e2e-mlp-test
  "End-to-end equivalence: a 2-layer MLP defined via defmodel (HMap params)
  must produce bit-identical forward + gradient values to a hand-written
  flat-arg deftm with the same body. This is the gate before touching GSDM."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core]
            [raster.params :as rp]
            [raster.compiler.pipeline :as pipeline]
            [raster.ad.reverse :as ad]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]))

;; ---------------------------------------------------------------------------
;; Style A: defmodel with HMap params
;; ---------------------------------------------------------------------------
(rp/defmodel mlp-loss-A
  [w :- (Params (HMap :mandatory {:W1 (Param (Array double))
                                  :b1 (Param (Array double))
                                  :W2 (Param (Array double))
                                  :b2 (Param (Array double))}))
   x :- (Array double)
   y :- (Array double)
   batch :- Long
   d-in  :- Long
   d-hid :- Long
   d-out :- Long]
  :- Double
  (let [h   (raster.dl.nn/linear x   (:W1 w) (:b1 w) batch d-in  d-hid)
        out (raster.dl.nn/linear h   (:W2 w) (:b2 w) batch d-hid d-out)]
    (raster.dl.loss/mse-loss out y (clojure.core/* batch d-out))))

;; ---------------------------------------------------------------------------
;; Style B: hand-rolled flat deftm with the same body
;; ---------------------------------------------------------------------------
(core/deftm mlp-loss-B
  [W1 :- (Array double) b1 :- (Array double)
   W2 :- (Array double) b2 :- (Array double)
   x  :- (Array double) y  :- (Array double)
   batch :- Long d-in :- Long d-hid :- Long d-out :- Long]
  :- Double
  (let [h   (raster.dl.nn/linear x  W1 b1 batch d-in  d-hid)
        out (raster.dl.nn/linear h  W2 b2 batch d-hid d-out)]
    (raster.dl.loss/mse-loss out y (clojure.core/* batch d-out))))

;; ---------------------------------------------------------------------------
;; Fixed inputs / weights — both styles see the same data
;; ---------------------------------------------------------------------------
(def W1 (double-array [0.1 -0.2  0.3 -0.4  0.5 -0.6  0.7 -0.8])) ; 4×2
(def b1 (double-array [0.0 0.0 0.0 0.0]))
(def W2 (double-array [0.05 -0.05 0.05 -0.05
                       0.05  0.05 0.05  0.05
                       -0.05 0.05 -0.05 0.05]))                  ; 3×4
(def b2 (double-array [0.0 0.0 0.0]))
(def x (double-array [0.5 -0.3]))
(def y (double-array [1.0 0.0 -1.0]))
(def shape-args [1 2 4 3])  ; batch d-in d-hid d-out

(defn pytree-w [] {:W1 W1 :b1 b1 :W2 W2 :b2 b2})

(deftest forward-equivalence-lazy-jit
  (let [a (apply mlp-loss-A (pytree-w) x y shape-args)
        b (apply mlp-loss-B W1 b1 W2 b2 x y shape-args)]
    (is (= a b) "lazy JIT: defmodel and flat deftm produce identical loss")))

(deftest forward-equivalence-compile-aot
  (let [fast-A (rp/compile-aot #'mlp-loss-A)
        fast-B (pipeline/compile-aot #'mlp-loss-B)
        a (apply fast-A (pytree-w) x y shape-args)
        b (apply fast-B W1 b1 W2 b2 x y shape-args)]
    (is (= a b) "compile-aot: defmodel and flat deftm produce identical loss")))

(deftest gradient-equivalence-via-flat-var
  (testing "value+grad on the defmodel's underlying flat var matches the raw deftm"
    (let [flat-A (-> #'mlp-loss-A meta :raster.params/flat-var)
          vg-A (ad/value+grad flat-A)
          vg-B (ad/value+grad #'mlp-loss-B)
          ;; Style A canonical (sorted) order: W1 W2 b1 b2
          ;; Style B declared order: W1 b1 W2 b2
          out-A (apply vg-A W1 W2 b1 b2 x y shape-args)
          out-B (apply vg-B W1 b1 W2 b2 x y shape-args)]
      (is (= (first out-A) (first out-B)) "loss matches")
      (is (= (vec (nth out-A 1)) (vec (nth out-B 1))) "W1 grad")
      (is (= (vec (nth out-A 2)) (vec (nth out-B 3))) "W2 grad — A idx 2, B idx 3")
      (is (= (vec (nth out-A 3)) (vec (nth out-B 2))) "b1 grad — A idx 3, B idx 2")
      (is (= (vec (nth out-A 4)) (vec (nth out-B 4))) "b2 grad"))))

(deftest gradient-equivalence-via-pytree-api
  (testing "rp/value+grad returns structured grad pytrees matching raw deftm"
    (let [vg-pytree (rp/value+grad #'mlp-loss-A)
          [loss-A grad-w-A] (apply vg-pytree (pytree-w) x y shape-args)
          vg-B (ad/value+grad #'mlp-loss-B)
          out-B (apply vg-B W1 b1 W2 b2 x y shape-args)]
      (is (= loss-A (first out-B)) "loss matches raw deftm")
      (is (map? grad-w-A) "structured grad is a map")
      (is (= #{:W1 :W2 :b1 :b2} (set (keys grad-w-A))))
      (is (= (vec (nth out-B 1)) (vec (:W1 grad-w-A))))
      (is (= (vec (nth out-B 2)) (vec (:b1 grad-w-A))))
      (is (= (vec (nth out-B 3)) (vec (:W2 grad-w-A))))
      (is (= (vec (nth out-B 4)) (vec (:b2 grad-w-A)))))))
