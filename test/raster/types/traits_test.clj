(ns raster.types.traits-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm defvalue]]
            [raster.types.traits :refer [order-style ordered?
                                         arithmetic-style has-division? has-arithmetic?
                                         iterator-size has-length? finite?
                                         ->Ordered ->Unordered
                                         ->FieldArithmetic ->RingArithmetic
                                         ->HasLength ->SizeUnknown ->IsInfinite]]))

;; ================================================================
;; OrderStyle tests
;; ================================================================

(deftest order-style-test
  (testing "Long is ordered"
    (is (ordered? (order-style 42))))
  (testing "Double is ordered"
    (is (ordered? (order-style 3.14))))
  (testing "String is ordered"
    (is (ordered? (order-style "hello"))))
  (testing "Object default is unordered"
    (is (not (ordered? (order-style {:a 1}))))))

;; ================================================================
;; ArithmeticStyle tests
;; ================================================================

(deftest arithmetic-style-test
  (testing "Double has field arithmetic (supports division)"
    (is (has-division? (arithmetic-style 3.14)))
    (is (has-arithmetic? (arithmetic-style 3.14))))
  (testing "Long has ring arithmetic (no division)"
    (is (not (has-division? (arithmetic-style 42))))
    (is (has-arithmetic? (arithmetic-style 42))))
  (testing "String has no arithmetic"
    (is (not (has-arithmetic? (arithmetic-style "hello"))))
    (is (not (has-division? (arithmetic-style "hello"))))))

;; ================================================================
;; IteratorSize tests
;; ================================================================

(deftest iterator-size-test
  (testing "double arrays have known length"
    (is (has-length? (iterator-size (double-array 3))))
    (is (finite? (iterator-size (double-array 3)))))
  (testing "long arrays have known length"
    (is (has-length? (iterator-size (long-array 5)))))
  (testing "Object has unknown size"
    (is (not (has-length? (iterator-size [1 2 3])))))
  (testing "unknown size is still considered finite"
    (is (finite? (iterator-size [1 2 3]))))
  (testing "IsInfinite is not finite"
    (is (not (finite? (->IsInfinite))))))

;; ================================================================
;; Trait extension: user-defined types
;; ================================================================

(defvalue Quaternion [w :- Double, x :- Double, y :- Double, z :- Double])

;; Extend traits for Quaternion
(deftm raster.types.traits/order-style [q :- Quaternion] (->Unordered))
(deftm raster.types.traits/arithmetic-style [q :- Quaternion] (->FieldArithmetic))

(deftest trait-extension-test
  (testing "Quaternion is unordered"
    (is (not (ordered? (order-style (->Quaternion 1.0 0.0 0.0 0.0))))))
  (testing "Quaternion has field arithmetic"
    (is (has-division? (arithmetic-style (->Quaternion 1.0 0.0 0.0 0.0))))
    (is (has-arithmetic? (arithmetic-style (->Quaternion 1.0 0.0 0.0 0.0))))))

;; ================================================================
;; Trait-based dispatch (Holy trait trick pattern)
;; ================================================================

;; Example: describe-ordering uses trait dispatch
;; Import trait classes for use in deftm type annotations
(import 'raster.types.traits.Ordered)
(import 'raster.types.traits.Unordered)

(deftm describe-ordering [style :- Ordered, x :- Object]
  :ordered)
(deftm describe-ordering [style :- Unordered, x :- Object]
  :unordered)

(defn describe [x]
  (describe-ordering (order-style x) x))

(deftest holy-trait-trick-test
  (testing "trait-based dispatch selects correct implementation"
    (is (= :ordered (describe 42)))
    (is (= :ordered (describe 3.14)))
    (is (= :ordered (describe "hello")))
    (is (= :unordered (describe {:a 1})))
    (is (= :unordered (describe (->Quaternion 1.0 0.0 0.0 0.0))))))
