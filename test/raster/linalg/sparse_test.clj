(ns raster.linalg.sparse-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.numeric :refer [+ - *]]
            [raster.linalg.sparse :refer [svec sparse-vector to-dense nnz dot norm]]))

(defn- close? [^double a ^double b]
  (< (Math/abs (- a b)) 1e-12))

;; ================================================================
;; Construction and conversion
;; ================================================================

(deftest svec-construction-test
  (testing "svec from map"
    (let [v (svec 10 {0 1.0, 5 2.0, 9 3.0})]
      (is (= 3 (nnz v)))
      (is (= [1.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0]
             (vec (to-dense v))))))
  (testing "empty sparse vector"
    (let [v (svec 5 {})]
      (is (= 0 (nnz v)))
      (is (= [0.0 0.0 0.0 0.0 0.0] (vec (to-dense v)))))))

;; ================================================================
;; Arithmetic
;; ================================================================

(deftest sparse-add-test
  (testing "add with overlapping indices"
    (let [a (svec 5 {0 1.0, 2 3.0})
          b (svec 5 {0 2.0, 3 4.0})
          r (+ a b)]
      (is (= [3.0 0.0 3.0 4.0 0.0] (vec (to-dense r))))
      (is (= 3 (nnz r)))))
  (testing "add with cancellation"
    (let [a (svec 5 {0 1.0, 2 3.0})
          b (svec 5 {0 -1.0, 3 4.0})
          r (+ a b)]
      ;; index 0 cancels: 1 + (-1) = 0
      (is (= [0.0 0.0 3.0 4.0 0.0] (vec (to-dense r))))
      (is (= 2 (nnz r))))))

(deftest sparse-sub-test
  (testing "negate"
    (let [a (svec 5 {1 2.0, 3 4.0})
          r (- a)]
      (is (= [0.0 -2.0 0.0 -4.0 0.0] (vec (to-dense r))))))
  (testing "subtract"
    (let [a (svec 5 {0 5.0, 2 3.0})
          b (svec 5 {0 2.0, 2 3.0})
          r (- a b)]
      ;; index 2 cancels
      (is (= [3.0 0.0 0.0 0.0 0.0] (vec (to-dense r))))
      (is (= 1 (nnz r))))))

(deftest sparse-scalar-mul-test
  (testing "scalar * sparse"
    (let [a (svec 5 {1 2.0, 3 4.0})
          r (* 3.0 a)]
      (is (= [0.0 6.0 0.0 12.0 0.0] (vec (to-dense r))))))
  (testing "sparse * scalar (integer)"
    (let [a (svec 5 {0 1.0, 4 5.0})
          r (* a 2)]
      (is (= [2.0 0.0 0.0 0.0 10.0] (vec (to-dense r)))))))

;; ================================================================
;; Dot product and norm
;; ================================================================

(deftest sparse-dot-test
  (testing "dot product skips non-overlapping"
    (let [a (svec 10 {0 1.0, 5 2.0, 9 3.0})
          b (svec 10 {0 4.0, 5 5.0, 7 6.0})]
      ;; 1*4 + 2*5 = 14
      (is (close? 14.0 (dot a b)))))
  (testing "dot with disjoint indices"
    (let [a (svec 10 {0 1.0, 1 2.0})
          b (svec 10 {5 3.0, 9 4.0})]
      (is (close? 0.0 (dot a b))))))

(deftest sparse-norm-test
  (testing "L2 norm"
    (let [a (svec 5 {0 3.0, 3 4.0})]
      (is (close? 5.0 (norm a))))))

;; ================================================================
;; Variadic (reducible dispatch)
;; ================================================================

(deftest sparse-variadic-test
  (testing "3-arg sparse +"
    (let [a (svec 5 {0 1.0})
          b (svec 5 {1 2.0})
          c (svec 5 {2 3.0})
          r (+ a b c)]
      (is (= [1.0 2.0 3.0 0.0 0.0] (vec (to-dense r)))))))
