(ns raster.compiler.passes.parallel.contract-lower-test
  "W1 step 2a: par/contract FORM → segmented SegRed (IR structure, device-free)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.ir.segop :as segop]))

(deftest contract-form-to-segred-nn
  (testing "matmul :nn form → segmented SegRed with free segments + reduced contract axis"
    (let [form '(raster.par/contract C [[i m] [j n]] [[l k]]
                                     (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))
          sr (cl/contract-form->segred form :id 5)]
      (is (instance? raster.compiler.ir.segop.SegRed sr))
      (testing "space: free axes are OUTER segment dims, contract axis is INNERMOST reduced"
        (is (= 3 (count (segop/seg-space-dims (:space sr)))))
        (is (= [{:name 'i :bound 'm} {:name 'j :bound 'n}]
               (segop/seg-space-segment-dims (:space sr))))
        (is (= {:name 'l :bound 'k} (segop/seg-space-reduced-dim (:space sr))))
        (is (= '(* m n) (segop/seg-space-num-segments-expr (:space sr)))))
      (testing "reduce-op: init 0.0, combine +, product in the element slot; map-lambda nil"
        (is (= 0.0 (:init (segop/scalar-reduce-op sr))))
        (is (= '+ (first (:lambda (segop/scalar-reduce-op sr)))))
        ;; combine is (+ acc <product>) — the product is the last operand
        (is (= '(* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))
               (last (:lambda (segop/scalar-reduce-op sr)))))
        (is (nil? (:lambda sr))))
      (testing "inputs {A B}, output {C}, scalars = the dim bounds (not index vars/arrays)"
        (is (= '#{A B} (:inputs sr)))
        (is (= '#{C} (:outputs sr)))
        (is (= '#{m n k} (:scalars sr)))
        (is (not (contains? (:scalars sr) 'i)))
        (is (not (contains? (:scalars sr) 'C))))
      (is (= :segmented (:phase sr)))
      (is (= 5 (:id sr))))))

(deftest contract-form-single-free-axis
  (testing "matvec: 1 free axis → 1 segment dim + reduced dim, no phantom segments"
    (let [form '(raster.par/contract y [[i m]] [[l k]]
                                     (* (aget A (+ (* i k) l)) (aget x l)))
          sr (cl/contract-form->segred form)]
      (is (= 2 (count (segop/seg-space-dims (:space sr)))))
      (is (= [{:name 'i :bound 'm}] (segop/seg-space-segment-dims (:space sr))))
      (is (= {:name 'l :bound 'k} (segop/seg-space-reduced-dim (:space sr))))
      (is (= 'm (segop/seg-space-num-segments-expr (:space sr))))
      (is (= '#{A x} (:inputs sr)))
      (is (= '#{y} (:outputs sr))))))

(deftest contract-form-custom-init-combine
  (testing "the exact typed identity and combine flow into the certified max-plus reduction"
    (let [form '(raster.par/contract C [[i m] [j n]] [[l k]]
                                     (+ (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))
                                     :init Double/NEGATIVE_INFINITY :combine max)
          sr (cl/contract-form->segred form)]
      (is (= 'Double/NEGATIVE_INFINITY (:init (segop/scalar-reduce-op sr))))
      (is (= 'max (first (:lambda (segop/scalar-reduce-op sr))))))))

(deftest contract-form-three-free-axes
  (testing "batched matmul: 3 free axes → 3 segment dims, num-segments = product"
    (let [form '(raster.par/contract C [[b btch] [i m] [j n]] [[l k]]
                                     (* (aget A x) (aget B y)))
          sr (cl/contract-form->segred form)]
      (is (= 4 (count (segop/seg-space-dims (:space sr)))))
      (is (= 3 (count (segop/seg-space-segment-dims (:space sr)))))
      (is (= {:name 'l :bound 'k} (segop/seg-space-reduced-dim (:space sr))))
      (is (= '(* (* btch m) n) (segop/seg-space-num-segments-expr (:space sr))))
      (is (= '#{btch m n k} (:scalars sr))))))
