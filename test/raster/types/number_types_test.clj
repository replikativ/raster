(ns raster.types.number-types-test
  "Tests for number type predicates.

  Note: isinteger (AbstractInteger/AbstractFloat) and finite?/infinite?/nan?
  for integer types are defined on abstract types and only resolve through
  the compiler walker (compile-time dispatch). These tests cover the concrete
  Double/Float runtime dispatch methods from raster.numeric, and verify the
  abstract type hierarchy is correctly registered."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.types.number-types] ;; loads abstract type hierarchy
            [raster.numeric :refer [finite? infinite? nan?]]
            [raster.compiler.core.dispatch :refer [assignable-from?]])
  (:import [raster.types.number_types AbstractFloat AbstractInteger Signed Real]))

;; ================================================================
;; Abstract type hierarchy registration
;; ================================================================

(deftest abstract-type-hierarchy-test
  (testing "Double is registered as subtype of AbstractFloat"
    (is (assignable-from? AbstractFloat Double)))
  (testing "Float is registered as subtype of AbstractFloat"
    (is (assignable-from? AbstractFloat Float)))
  (testing "Long is registered as subtype of Signed"
    (is (assignable-from? Signed Long)))
  (testing "Integer is registered as subtype of Signed"
    (is (assignable-from? Signed Integer)))
  (testing "Short is registered as subtype of Signed"
    (is (assignable-from? Signed Short)))
  (testing "Byte is registered as subtype of Signed"
    (is (assignable-from? Signed Byte))))

;; ================================================================
;; finite? — Double and Float
;; ================================================================

(deftest finite-double-test
  (testing "finite? for normal Double values"
    (is (true? (finite? 0.0)))
    (is (true? (finite? 1.0)))
    (is (true? (finite? -1.0)))
    (is (true? (finite? Double/MAX_VALUE)))
    (is (true? (finite? Double/MIN_VALUE))))
  (testing "finite? returns false for Double infinity"
    (is (false? (finite? Double/POSITIVE_INFINITY)))
    (is (false? (finite? Double/NEGATIVE_INFINITY))))
  (testing "finite? returns false for Double NaN"
    (is (false? (finite? Double/NaN)))))

(deftest finite-float-test
  (testing "finite? for normal Float values"
    (is (true? (finite? (float 0.0))))
    (is (true? (finite? (float 1.0))))
    (is (true? (finite? (float -1.0))))
    (is (true? (finite? Float/MAX_VALUE)))
    (is (true? (finite? Float/MIN_VALUE))))
  (testing "finite? returns false for Float infinity"
    (is (false? (finite? Float/POSITIVE_INFINITY)))
    (is (false? (finite? Float/NEGATIVE_INFINITY))))
  (testing "finite? returns false for Float NaN"
    (is (false? (finite? Float/NaN)))))

;; ================================================================
;; infinite? — Double and Float
;; ================================================================

(deftest infinite-double-test
  (testing "infinite? for normal Double values"
    (is (false? (infinite? 0.0)))
    (is (false? (infinite? 1.0)))
    (is (false? (infinite? -1.0)))
    (is (false? (infinite? Double/MAX_VALUE))))
  (testing "infinite? for Double NaN (NaN is not infinite)"
    (is (false? (infinite? Double/NaN))))
  (testing "infinite? returns true for Double infinity"
    (is (true? (infinite? Double/POSITIVE_INFINITY)))
    (is (true? (infinite? Double/NEGATIVE_INFINITY)))))

(deftest infinite-float-test
  (testing "infinite? for normal Float values"
    (is (false? (infinite? (float 0.0))))
    (is (false? (infinite? (float 1.0)))))
  (testing "infinite? for Float NaN (NaN is not infinite)"
    (is (false? (infinite? Float/NaN))))
  (testing "infinite? returns true for Float infinity"
    (is (true? (infinite? Float/POSITIVE_INFINITY)))
    (is (true? (infinite? Float/NEGATIVE_INFINITY)))))

;; ================================================================
;; nan? — Double and Float
;; ================================================================

(deftest nan-double-test
  (testing "nan? for normal Double values"
    (is (false? (nan? 0.0)))
    (is (false? (nan? 1.0)))
    (is (false? (nan? -1.0)))
    (is (false? (nan? Double/MAX_VALUE))))
  (testing "nan? for Double infinity (infinity is not NaN)"
    (is (false? (nan? Double/POSITIVE_INFINITY)))
    (is (false? (nan? Double/NEGATIVE_INFINITY))))
  (testing "nan? returns true for Double NaN"
    (is (true? (nan? Double/NaN)))))

(deftest nan-float-test
  (testing "nan? for normal Float values"
    (is (false? (nan? (float 0.0))))
    (is (false? (nan? (float 1.0)))))
  (testing "nan? for Float infinity (infinity is not NaN)"
    (is (false? (nan? Float/POSITIVE_INFINITY)))
    (is (false? (nan? Float/NEGATIVE_INFINITY))))
  (testing "nan? returns true for Float NaN"
    (is (true? (nan? Float/NaN)))))

;; ================================================================
;; Edge cases: special Double/Float values
;; ================================================================

(deftest double-negative-zero-test
  (testing "negative zero is finite and not NaN"
    (is (true? (finite? -0.0)))
    (is (false? (nan? -0.0)))
    (is (false? (infinite? -0.0)))))

(deftest float-negative-zero-test
  (testing "Float negative zero is finite and not NaN"
    (is (true? (finite? (float -0.0))))
    (is (false? (nan? (float -0.0))))
    (is (false? (infinite? (float -0.0))))))

(deftest double-min-normal-test
  (testing "Double/MIN_NORMAL is finite"
    (is (true? (finite? Double/MIN_NORMAL)))
    (is (false? (nan? Double/MIN_NORMAL)))
    (is (false? (infinite? Double/MIN_NORMAL)))))
